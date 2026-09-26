use std::sync::Arc;

use crate::arrays::{Array, ArrayAddressing, ArrayElement, NumericArrayElement, StaticShape};
use crate::macros::dispatch_on_array_element_type;
use crate::operations::arithmetic::Add;
use crate::operations::manipulation::reshaping::Reshape;
use crate::operations::manipulation::slicing::Slice;

use super::*;

/// Canonical operation name for [`DotOperation`].
pub const DOT_OPERATION_NAME: &str = "dot";

/// Primitive representing a generalized dot (tensor contraction).
///
/// [`DotOperation`] is the unified primitive for matrix multiplication, batched matrix multiplication, vector inner
/// products, and arbitrary tensor contractions. It lowers to StableHLO's `dot_general` op in the XLA backend.
///
/// A dot is bilinear. Forward-mode differentiation applies
/// `d(dot(lhs, rhs)) = dot(dlhs, rhs) + dot(lhs, drhs)`. Transposition therefore accepts exactly one linear operand
/// and contracts the output cotangent with the other, known operand using the corresponding adjoint dimension
/// numbers. Accumulation-typed dots perform those contractions at the widened cotangent type before converting the
/// result back to the linear operand's cotangent representation.
///
/// Each forward-mode tangent term is staged as an ordinary dot that preserves the primal dimension numbers,
/// accumulation type, and requested output sharding without introducing captures. Transposition pins the adjoint
/// contraction's output sharding to the cotangent dual of the linear operand's sharding. A structural-zero output
/// cotangent remains structural zero.
///
/// Batching aligns the operands onto a common mapped axis and lifts the dimension numbers past it. Materializing an
/// axis on a replicated operand preserves a dynamic mapped extent as a first-class value. Two mapped operands must
/// describe the same mapped extent; for dynamic extents, they must share the same
/// [`DimensionVariable`](crate::arrays::DimensionVariable).
///
/// Every bounded ragged axis is either contracted or free. A contracted axis is zero-padded and consumed, and its
/// dimension variable is reported as [`BatchedOutputs`](crate::batching::BatchedOutputs) evidence so carrier validation
/// can distinguish deliberate consumption from a missing extent. Each operand is padded along only its own contracted
/// ragged axes because zeroing either factor removes the corresponding product. A free ragged axis propagates to the
/// result through the dot output layout: batching dimensions, then LHS free axes, then RHS free axes. A bounded ragged
/// axis on a batching dimension or a replicated operand is unsupported because no shared per-item extent identity
/// exists. Operands without bounded ragged axes use the dense path unchanged.
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

    /// Returns a [`DotOperation`] configured for standard rank-2 matrix multiplication.
    #[inline]
    pub fn matmul() -> Self {
        Self::new(DotDimensionNumbers::matmul())
    }

    /// Attaches a requested output [`Sharding`] to this operation, mirroring the `out_sharding` parameter of JAX's
    /// `dot_general`. When set, type inference validates the requested sharding (rank, mesh, no auto axes, and the
    /// unreduced-output rule) and uses it for the output instead of the inferred sharding, bypassing the batch and
    /// contracting dimension consistency checks. This is the only way to produce an output with unreduced axes
    /// (i.e., per-device partial results whose cross-device reduction is delayed).
    #[inline]
    pub fn with_output_sharding(mut self, output_sharding: impl Into<Option<Sharding>>) -> Self {
        self.output_sharding = output_sharding.into();
        self
    }

    /// Returns a copy of this [`DotOperation`] with the provided accumulation data type. The operand element types
    /// must still match each other and must promote to the accumulation type, which becomes the output element
    /// type: the backend upcasts the operands and accumulates the contraction at the wider type (XLA's
    /// `preferred_element_type` contract, which is what its low-precision matrix units implement natively — e.g.,
    /// `f8 × f8 → f32` and `bf16 × bf16 → f32`). Accumulation-typed dots differentiate like ordinary dots, with
    /// tangents and cotangents carried at the accumulation type (refer to the forward-mode and transpose rule
    /// documentation on this operation), and cannot yet be combined with a requested output sharding.
    #[inline]
    pub fn with_accumulation_type(mut self, accumulation_type: impl Into<Option<DataType>>) -> Self {
        self.accumulation_type = accumulation_type.into();
        self
    }

    /// Returns the optional accumulation data type. Refer to the documentation of
    /// [`Self::with_accumulation_type`].
    #[inline]
    pub fn accumulation_type(&self) -> Option<DataType> {
        self.accumulation_type
    }

    /// Returns the contracting and batching dimension specification.
    #[inline]
    pub fn dimensions(&self) -> &DotDimensionNumbers {
        &self.dimensions
    }

    /// Returns the requested output sharding, if any.
    #[inline]
    pub fn output_sharding(&self) -> Option<&Sharding> {
        self.output_sharding.as_ref()
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

impl<C: Domain<Type = ArrayType, Value: Dot>> InterpretableOperation<C> for DotOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        // The requested output sharding and accumulation type flow through the capability methods so that
        // interpretation over staging values (e.g., during program batching) preserves them; concrete values
        // ignore the sharding and upcast for the accumulation type. Type inference rejects combining the two.
        Ok(vec![match (&self.accumulation_type, &self.output_sharding) {
            (Some(accumulation_type), _) => {
                inputs[0].dot_with_accumulation_type(&inputs[1], &self.dimensions, *accumulation_type)?
            }
            (None, Some(output_sharding)) => {
                inputs[0].dot_with_output_sharding(&inputs[1], &self.dimensions, output_sharding)?
            }
            (None, None) => inputs[0].dot(&inputs[1], &self.dimensions)?,
        }])
    }
}

// Partial evaluation uses the default fold-or-residualize behavior.
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for DotOperation where
    C::Operation: From<DotOperation>
{
}

/// Value-level generalized dot capability.
///
/// [`Dot`] is the receiver-style entry point for staging or executing [`DotOperation`]. It performs the contraction
/// described by `dimensions`, supporting standard matrix multiplication, batched matrix multiplication, vector inner
/// products, and arbitrary tensor contractions.
pub trait Dot<Rhs = Self>: Sized {
    /// Computes the generalized dot product of `self` and `rhs` using `dimensions`, and returns a [`ProgramError`] if
    /// the operands are incompatible with `dimensions` or the contraction cannot be recorded in the value's context.
    fn dot(&self, rhs: &Rhs, dimensions: &DotDimensionNumbers) -> Result<Self, ProgramError>;

    /// Computes the generalized dot product of `self` and `rhs` using `dimensions`, requesting `output_sharding`
    /// for the result. The requested sharding overrides the inferred output sharding and is validated by the staged
    /// operation's type inference (refer to the documentation of [`DotOperation::with_output_sharding`]). The
    /// default implementation ignores the requested sharding and delegates to [`Self::dot`], which is correct for
    /// concrete (single-device) values, for which a sharding only describes distribution metadata; staging
    /// implementations override this method to attach the requested sharding to the staged operation.
    fn dot_with_output_sharding(
        &self,
        rhs: &Rhs,
        dimensions: &DotDimensionNumbers,
        output_sharding: &Sharding,
    ) -> Result<Self, ProgramError> {
        let _ = output_sharding;
        self.dot(rhs, dimensions)
    }

    /// Computes the generalized dot product of `self` and `rhs` using `dimensions`, upcasting the operands to
    /// `accumulation_type` and accumulating the contraction there, so the result carries the accumulation type.
    /// Refer to the documentation of [`DotOperation::with_accumulation_type`] for the exact contract.
    fn dot_with_accumulation_type(
        &self,
        rhs: &Rhs,
        dimensions: &DotDimensionNumbers,
        accumulation_type: DataType,
    ) -> Result<Self, ProgramError>;
}

impl Dot for Array {
    fn dot_with_accumulation_type(
        &self,
        rhs: &Self,
        dimensions: &DotDimensionNumbers,
        accumulation_type: DataType,
    ) -> Result<Self, ProgramError> {
        let lhs = self.convert_element_type(accumulation_type)?;
        let rhs = rhs.convert_element_type(accumulation_type)?;
        lhs.dot(&rhs, dimensions)
    }

    fn dot(&self, rhs: &Self, dimensions: &DotDimensionNumbers) -> Result<Self, ProgramError> {
        // TODO(eaplatanios): What about the accumulation type?
        let data_type = self.r#type().data_type();
        dispatch_on_array_element_type!(@numeric data_type, |Element| {
            self.dot_elements::<Element>(rhs, dimensions)
        })
    }
}

// Context-carrying values stage a dot through their context. The `From<DotOperation>` bound keeps this implementation
// disjoint from eager values, whose context operation is `ConstantOperation`.
impl<V: Value<Type = ArrayType> + ManualVariationAlignment<ArrayType>> Dot for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<DotOperation>,
{
    fn dot(&self, rhs: &Self, dimensions: &DotDimensionNumbers) -> Result<Self, ProgramError> {
        let inputs = [self.clone(), rhs.clone()];
        let inputs = ManualVariationAlignment::align_manual_variation(&inputs)?;
        let mut outputs = self.dispatch_domain().bind(DotOperation::new(dimensions.clone()), Vec::new(), &inputs)?;
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
        let mut outputs = self.dispatch_domain().bind(
            DotOperation::new(dimensions.clone()).with_accumulation_type(accumulation_type),
            Vec::new(),
            &inputs,
        )?;
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
        let mut outputs = self.dispatch_domain().bind(
            DotOperation::new(dimensions.clone()).with_output_sharding(output_sharding.clone()),
            Vec::new(),
            &inputs,
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Canonical operation name for [`RaggedDotOperation`].
pub const RAGGED_DOT_OPERATION_NAME: &str = "ragged_dot_general";

/// Primitive representing a grouped generalized dot with explicit group sizes.
///
/// Exactly one LHS dimension is ragged. Its role selects one of three modes:
///
///   - A non-contracting ragged dimension is partitioned into consecutive groups. The corresponding RHS group
///     dimension selects one RHS slice per group, and the grouped products are written back along the LHS result
///     dimension. A zero-size group contributes nothing and any uncovered suffix of that dimension is zero.
///   - A contracting ragged dimension partitions the paired contracting dimensions into consecutive groups. The
///     output gains a leading group dimension, and a zero-size group produces a zero slice.
///   - A batching ragged dimension has ordinary batched-dot semantics. `group_sizes` participates in type inference
///     but its values do not affect the result.
///
/// `group_sizes` is either a rank-one `[group_count]` array shared by every prefix or an array whose trailing axis is
/// `group_count` and whose prefix matches the dimensions preceding the ragged position in the grouped-dot iteration
/// space.
///
/// In non-contracting and contracting modes every size must be nonnegative. The eager interpreter rejects negative
/// metadata in those modes. XLA's decomposition lowering additionally clamps signed negatives to zero before unsigned
/// accumulation so invalid runtime metadata cannot become a large interval. The instruction lowering passes metadata
/// unchanged to `chlo.ragged_dot` and therefore relies on the nonnegative-input contract.
///
/// The sizes define consecutive raw cumulative intervals. Each interval is intersected with the physical LHS ragged
/// extent, so an over-covering group is clipped and every later group is empty once its raw start reaches or exceeds
/// that extent.
///
/// Grouped expansion modes require an element type that can represent zero; in particular, they reject `f8e8m0fnu`.
/// Refer to [`RaggedDotDimensionNumbers`] for the dimension-number contract.
///
/// The operation is linear in either data operand separately, while `group_sizes` is nondifferentiable metadata.
/// Forward-mode differentiation applies the two-term product rule with the same group metadata. Transposition is
/// defined only in non-contracting mode and applies another grouped dot followed by the inverse adjoint-axis
/// permutation.
///
/// Batching accepts either three replicated operands or three operands mapped over leading axis zero. It rejects
/// [`RaggedAxis`](crate::arrays::RaggedAxis) metadata because `group_sizes` is the sole source of ragged extents for
/// this operation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RaggedDotOperation {
    /// Grouped-dot dimension-number specification.
    dimensions: RaggedDotDimensionNumbers,
}

impl RaggedDotOperation {
    /// Creates a grouped generalized dot.
    #[inline]
    pub fn new(dimensions: RaggedDotDimensionNumbers) -> Self {
        Self { dimensions }
    }

    /// Returns the grouped-dot dimension-number specification.
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
        Ok(vec![ragged_dot_abstract(&input_types[0], &input_types[1], &input_types[2], &self.dimensions)?])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("dimensions", &self.dimensions)?;
            Ok(())
        })
    }
}

impl<C: Domain<Type = ArrayType, Value: RaggedDot>> InterpretableOperation<C> for RaggedDotOperation {
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

impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for RaggedDotOperation where
    C::Operation: From<RaggedDotOperation>
{
}

/// Value-level grouped generalized dot capability.
pub trait RaggedDot: Sized {
    /// Computes a grouped generalized dot using explicit `group_sizes`. Refer to [`RaggedDotOperation`] for the three
    /// modes, metadata shapes, cumulative-interval clipping, and zero-group and uncovered-position semantics.
    fn ragged_dot_general(
        &self,
        rhs: &Self,
        group_sizes: &Self,
        dimensions: &RaggedDotDimensionNumbers,
    ) -> Result<Self, ProgramError>;

    /// Computes the basic non-contracting form `[M, K] × [G, K, N] → [M, N]`. Refer to [`RaggedDotOperation`] for
    /// zero-size-group and uncovered-row behavior.
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
        dispatch_on_array_element_type!(@numeric data_type, |Element| {
            self.ragged_dot_elements::<Element>(rhs, group_sizes, dimensions)
        })
    }
}

impl<V: Value<Type = ArrayType> + ManualVariationAlignment<ArrayType>> RaggedDot for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<RaggedDotOperation>,
{
    fn ragged_dot_general(
        &self,
        rhs: &Self,
        group_sizes: &Self,
        dimensions: &RaggedDotDimensionNumbers,
    ) -> Result<Self, ProgramError> {
        let inputs = [self.clone(), rhs.clone(), group_sizes.clone()];
        let inputs = ManualVariationAlignment::align_manual_variation(&inputs)?;
        Ok(self
            .dispatch_domain()
            .bind(RaggedDotOperation::new(dimensions.clone()), Vec::new(), &inputs)?
            .remove(0))
    }
}

/// Combined generalized dot product and transposition capability.
///
/// This convenience trait groups the value-level [`Dot`] and [`Transpose`] operations used by the unified
/// [`DotOperation`] and [`TransposeOperation`](crate::operations::manipulation::TransposeOperation) primitives.
pub trait DotOps: Dot + Transpose {}

impl<T: Dot + Transpose> DotOps for T {}

impl Array {
    /// Allocates an array whose logical elements are initialized to the additive identity.
    fn zeroed<T: ArrayElement>(output_type: ArrayType) -> Result<Self, ProgramError> {
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        let zero = T::zero()?;
        for element in 0..output_addressing.element_count() {
            zero.encode(&mut bytes[output_addressing.byte_range_for_flat_index(element)]);
        }
        Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
    }

    /// Evaluates grouped generalized dot extent-exactly. Each concrete group's raw cumulative interval is clipped to
    /// the physical ragged extent, the resulting pair of operand slices is contracted by the ordinary generalized-dot
    /// kernel, and the result is written into its output window. This keeps temporary storage proportional to one
    /// group rather than the whole operand times the group count.
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
            return self.dot_elements::<T>(rhs, dot_dimensions);
        }
        let prefix_axes = dimensions.group_sizes_prefix_dimensions(self.r#type().rank())?;
        let prefix_shape = prefix_axes
            .iter()
            .map(|axis| self.r#type().shape().dimensions()[*axis].value().unwrap())
            .collect::<Vec<_>>();
        let prefix_count = prefix_shape.iter().product::<usize>();
        let group_count = group_sizes.r#type().shape().dimensions().last().unwrap().value().ok_or_else(|| {
            ProgramError::InvalidArgument {
                message: format!("`{RAGGED_DOT_OPERATION_NAME}` requires a static group count for eager evaluation"),
            }
        })?;
        let sizes = group_sizes.non_negative_integer_elements("group_sizes")?;
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
        let lhs_result = crate::operations::dot::lhs_result_axes(dot_dimensions, self.r#type().rank());
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
        let mut output = Self::zeroed::<T>(output_type)?;
        for prefix in 0..prefix_count {
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
                        lhs_slice.dot_elements::<T>(&rhs_slice, dense_dimensions)?
                    }
                    RaggedDotMode::Contracting => {
                        let rhs_ragged_axis = contracting_rhs_ragged_axis.unwrap();
                        rhs_starts[rhs_ragged_axis] = ragged_start;
                        rhs_limits[rhs_ragged_axis] = ragged_limit;
                        let rhs_slice = rhs.slice(&rhs_starts, &rhs_limits, &rhs_strides)?;
                        output_starts[0] = group;
                        output_limits[0] = group + 1;
                        let dot = lhs_slice.dot_elements::<T>(&rhs_slice, dot_dimensions)?;
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

    /// Contracts typed elements using each input's physical layout and the declared contraction dimensions.
    fn dot_elements<T: NumericArrayElement>(
        &self,
        rhs: &Self,
        dimensions: &DotDimensionNumbers,
    ) -> Result<Self, ProgramError> {
        debug_assert_eq!(self.r#type().data_type(), T::data_type());
        debug_assert_eq!(rhs.r#type().data_type(), T::data_type());
        let mut output_types = DotOperation::new(dimensions.clone())
            .infer_output_types(&[self.r#type().into_owned(), rhs.r#type().into_owned()], &[])?;
        let output_type = output_types.remove(0);
        let lhs_shape = self.r#type().static_shape().unwrap();
        let rhs_shape = rhs.r#type().static_shape().unwrap();
        let output_shape = output_type.static_shape().unwrap();
        let output_strides = output_shape.row_major_strides();
        let lhs_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let rhs_addressing = ArrayAddressing::new(rhs.r#type().into_owned())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;

        let lhs_batching = dimensions.lhs_batching_dimensions();
        let rhs_batching = dimensions.rhs_batching_dimensions();
        let lhs_contracting = dimensions.lhs_contracting_dimensions();
        let rhs_contracting = dimensions.rhs_contracting_dimensions();
        let lhs_result = (0..lhs_shape.rank())
            .filter(|axis| !lhs_batching.contains(axis) && !lhs_contracting.contains(axis))
            .collect::<Vec<_>>();
        let rhs_result = (0..rhs_shape.rank())
            .filter(|axis| !rhs_batching.contains(axis) && !rhs_contracting.contains(axis))
            .collect::<Vec<_>>();
        let contracting_shape =
            StaticShape::new(lhs_contracting.iter().map(|axis| lhs_shape[*axis]).collect::<Vec<_>>());
        let contracting_strides = contracting_shape.row_major_strides();
        let contracting_count = contracting_shape.dimensions().iter().product();

        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        let mut lhs_index = vec![0usize; lhs_shape.rank()];
        let mut rhs_index = vec![0usize; rhs_shape.rank()];
        for output_flat in 0..output_addressing.element_count() {
            // Decode the result coordinate directly into the corresponding batch and non-contracting operand axes.
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

            let mut accumulator = if T::data_type() == DataType::F8E8M0FNU { None } else { Some(T::zero()?) };
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
                accumulator = Some(match accumulator {
                    Some(accumulator) => accumulator.add(product)?,
                    None => product,
                });
            }
            let accumulator = match accumulator {
                Some(accumulator) => accumulator,
                None => T::zero()?,
            };
            accumulator.encode(&mut bytes[output_addressing.byte_range_for_flat_index(output_flat)]);
        }
        Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
    }
}
