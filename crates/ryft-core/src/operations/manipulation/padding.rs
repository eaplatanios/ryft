use std::borrow::Cow;
use std::fmt::Display;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::arrays::{
    Array, ArrayAddressing, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrBatch,
    ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayType, ArrayTypeRefinements, DataType,
    Dimension, DimensionOperation, DimensionType, DimensionValue, LinearResiduals, RaggedAxis, Shape,
    materialize_array_tangent,
};
use crate::axes::Axis;
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, ProjectedContext, StagingContext};
use crate::differentiation::{
    DifferentiableType, DifferentiationDual, DifferentiationError, ElementwiseDerivativeAlignment,
    ResidualZeroProvider, TransposableOperation, transpose_projected_operation,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation, impl_reference_dischargeable_operation};
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::constants::one::{One, OneOperation};
use crate::operations::constants::zero::{Zero, ZeroOperation};
use crate::operations::constants::zero_like::ZeroLikeOperation;
use crate::operations::control_flow::select::{Select, SelectOperation};
use crate::operations::differentiation::linear_call::LinearCallOperation;
use crate::operations::dimensions::dimension_add::DimensionAddOperation;
use crate::operations::dimensions::dimension_mul::DimensionMulOperation;
use crate::operations::dimensions::dimension_saturating_sub::DimensionSaturatingSubOperation;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::broadcasting::{Broadcast, DynamicBroadcastOperation};
use crate::operations::manipulation::slicing::{DynamicShapeSliceOperation, SliceOperation, resized_output_sharding};
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::math::reduce::{ReduceOperation, ReductionKind};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    EffectClass, EffectClasses, Effects, MaybeZero, Operation, OperationFormatter, OperationProjection, ProgramError,
    ProjectedValue, RegionInterface, Type, TypeError, Typed, Value, ValueProjection,
};
use crate::tracing::{NestedTracingContext, Tracer, TracingContext};

/// Canonical operation name for [`PadOperation`].
pub const PAD_OPERATION_NAME: &str = "pad";

/// [`Operation`] that expands its first input by adding edge and interior padding filled with its second input.
/// Refer to the documentation of [`Pad`] for more information.
///
/// The type parameter selects the input contract without introducing a separate padding operation. For example:
///
///   - `PadOperation<ArrayType>` accepts the input and padding-value arrays. It is used in programs over homogeneous
///     arrays whose output extents are fully described by the inferred array type.
///   - `PadOperation<ArrayIrType>` additionally accepts one first-class dimension input for each output axis. It is
///     used in mixed array/dimension programs that must carry those logical result extents explicitly.
///
/// Live reverse-mode differentiation of symbolic input shapes uses the mixed form and linearization, which retains the
/// runtime extents needed to restore cropped input positions and build the padding-position mask. Homogeneous direct
/// transposition supports static geometry as symbolic-zero cotangents do not require runtime extents.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PadOperation<T: Type> {
    /// Refer to the documentation of [`Self::edge_padding_low`] for more information.
    edge_padding_low: Vec<i64>,

    /// Refer to the documentation of [`Self::edge_padding_high`] for more information.
    edge_padding_high: Vec<i64>,

    /// Refer to the documentation of [`Self::interior_padding`] for more information.
    interior_padding: Vec<usize>,

    /// Whether the mixed signature needs an execution-time output extent assertion. Initialized conservatively to
    /// `true` and refined by [`PadOperation::with_input_types`]; ignored by the homogeneous operation.
    requires_runtime_assertion: bool,

    /// [`PhantomData`] marker tying this [`Operation`] to the [`Type`] universe in which it is valid.
    marker: PhantomData<fn() -> T>,
}

impl<T: Type> PadOperation<T> {
    /// Creates a new [`PadOperation`] with the provided edge and interior padding amounts. The three vectors must
    /// share one length (i.e., one entry per input axis). Whether that shared length matches the input rank is
    /// validated during type inference, once an input type is known. Mixed operations initially retain a runtime
    /// output-extent assertion. Use [`PadOperation::with_input_types`] to remove it when the input types prove the
    /// output extents. Homogeneous operations remain effect-free.
    pub fn new(
        edge_padding_low: Vec<i64>,
        edge_padding_high: Vec<i64>,
        interior_padding: Vec<usize>,
    ) -> Result<Self, ProgramError> {
        if edge_padding_low.len() != edge_padding_high.len() || edge_padding_low.len() != interior_padding.len() {
            return Err(TypeError::invalid(format!(
                "`{}` expects `edge_padding_low`, `edge_padding_high`, and `interior_padding` to \
                 share one length but got lengths {}, {}, and {}",
                PAD_OPERATION_NAME,
                edge_padding_low.len(),
                edge_padding_high.len(),
                interior_padding.len(),
            ))
            .into());
        }
        Ok(Self {
            edge_padding_low,
            edge_padding_high,
            interior_padding,
            requires_runtime_assertion: true,
            marker: PhantomData,
        })
    }

    /// Returns the signed padding amount at the beginning of each input axis. Positive amounts add padding elements,
    /// negative amounts crop elements from the beginning, and zero leaves that edge unchanged. Cropping applies after
    /// interior padding has been inserted. The amounts use [`i64`] because they can be negative.
    #[inline]
    pub fn edge_padding_low(&self) -> &[i64] {
        self.edge_padding_low.as_slice()
    }

    /// Returns the signed padding amount at the end of each input axis. Positive amounts add padding elements,
    /// negative amounts crop elements from the end, and zero leaves that edge unchanged. Cropping applies after
    /// interior padding has been inserted. The amounts use [`i64`] because they can be negative.
    #[inline]
    pub fn edge_padding_high(&self) -> &[i64] {
        self.edge_padding_high.as_slice()
    }

    /// Returns the number of padding elements inserted between each pair of adjacent input elements along each axis.
    /// Zero leaves adjacent elements contiguous, while one inserts a single padding element between them. These counts
    /// use [`usize`] because interior padding is nonnegative and cannot crop elements. No interior padding is inserted
    /// along an axis with fewer than two input elements.
    #[inline]
    pub fn interior_padding(&self) -> &[usize] {
        self.interior_padding.as_slice()
    }

    /// Renders this payload independently of its homogeneous or composite operation contract.
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, PAD_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("edge_padding_low", format_args!("{:?}", self.edge_padding_low))?;
            operation.field("edge_padding_high", format_args!("{:?}", self.edge_padding_high))?;
            operation.field("interior_padding", format_args!("{:?}", self.interior_padding))
        })
    }
}

// TODO(eaplatanios): Review from here onwards.

impl PadOperation<ArrayIrType> {
    /// Validates an input signature and removes the assertion effect when types alone prove every output extent.
    /// A refined payload rejects subsequent signatures that would require a runtime assertion.
    pub fn with_input_types(mut self, input_types: &[ArrayIrType]) -> Result<Self, TypeError> {
        self.requires_runtime_assertion = true;
        self.infer_output_types(input_types, &[])?;
        let input = <&ArrayType>::try_from(&input_types[0])?;
        let output_dimensions = ArrayIrType::extents(&input_types[2..])?;
        self.requires_runtime_assertion = !self.has_proven_output_extents(input, &output_dimensions)?;
        Ok(self)
    }

    /// Returns whether execution must validate the supplied output extents against the padded input geometry.
    pub fn requires_runtime_assertion(&self) -> bool {
        self.requires_runtime_assertion
    }

    /// Checks extent equality from an already validated signature without evaluating dimension inputs.
    fn has_proven_output_extents(&self, input: &ArrayType, output_dimensions: &[Dimension]) -> Result<bool, TypeError> {
        for (axis, (input_dimension, output_dimension)) in
            input.shape().dimensions().iter().zip(output_dimensions).enumerate()
        {
            let identity = self.edge_padding_low[axis] as i128 + self.edge_padding_high[axis] as i128 == 0
                && (self.interior_padding[axis] == 0
                    || input_dimension.bounds().upper().is_some_and(|upper| upper <= 2));
            if identity && input_dimension == output_dimension {
                continue;
            }
            if let (Some(input_extent), Some(output_extent)) = (input_dimension.value(), output_dimension.value())
                && static_padded_extent(
                    input_extent,
                    self.edge_padding_low[axis],
                    self.edge_padding_high[axis],
                    self.interior_padding[axis],
                    axis,
                )? == output_extent
            {
                continue;
            }
            return Ok(false);
        }
        Ok(true)
    }
}

impl From<PadOperation<ArrayType>> for PadOperation<ArrayIrType> {
    fn from(operation: PadOperation<ArrayType>) -> Self {
        Self {
            edge_padding_low: operation.edge_padding_low,
            edge_padding_high: operation.edge_padding_high,
            interior_padding: operation.interior_padding,
            requires_runtime_assertion: true,
            marker: PhantomData,
        }
    }
}

impl From<PadOperation<ArrayIrType>> for PadOperation<ArrayType> {
    fn from(operation: PadOperation<ArrayIrType>) -> Self {
        Self {
            edge_padding_low: operation.edge_padding_low,
            edge_padding_high: operation.edge_padding_high,
            interior_padding: operation.interior_padding,
            requires_runtime_assertion: true,
            marker: PhantomData,
        }
    }
}

impl<A: Value<Type = ArrayType>> From<PadOperation<ArrayType>> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: PadOperation<ArrayType>) -> Self {
        Self::Pad(operation.into())
    }
}

impl<T: Type> Display for PadOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for PadOperation<ArrayType> {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        PAD_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        match input_types[0].pad(
            &input_types[1],
            self.edge_padding_low.as_slice(),
            self.edge_padding_high.as_slice(),
            self.interior_padding.as_slice(),
        ) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError::invalid(error.to_string())),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        self.render(formatter, indentation)
    }
}

impl Operation for PadOperation<ArrayIrType> {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        PAD_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        if input_types.len() < 2 {
            return Err(TypeError::invalid(format!("expected at least 2 inputs but got {}", input_types.len())));
        }
        let input = <&ArrayType>::try_from(&input_types[0])?;
        let padding_value = <&ArrayType>::try_from(&input_types[1])?;
        let expected_input_count = input.rank() + 2;
        if input_types.len() != expected_input_count {
            return Err(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` expects an input, a padding value, and one output extent per axis \
                 ({expected_input_count} inputs total) but got {}",
                input_types.len(),
            )));
        }
        validate_pad_inputs(
            input,
            padding_value,
            self.edge_padding_low(),
            self.edge_padding_high(),
            self.interior_padding(),
        )
        .map_err(|error| match error {
            ProgramError::Type(error) => error,
            error => TypeError::invalid(error.to_string()),
        })?;
        let output_dimensions = ArrayIrType::extents(&input_types[2..])?;
        if !self.requires_runtime_assertion && !self.has_proven_output_extents(input, &output_dimensions)? {
            return Err(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` was constructed without a runtime extent check but these input types \
                 require one"
            )));
        }

        if is_effective_identity(input, self.edge_padding_low(), self.edge_padding_high(), self.interior_padding())
            && output_dimensions == input.shape().dimensions()
        {
            return Ok(vec![input.clone().into()]);
        }

        // Bounds only rule out impossible signatures. Runtime assertions establish the exact relation, including
        // when cropping is valid for only part of the input's declared range or result bounds narrow that range.
        for (axis, (input_dimension, output_dimension)) in
            input.shape().dimensions().iter().zip(&output_dimensions).enumerate()
        {
            let input_bounds = input_dimension.bounds();
            let minimum = padded_extent(
                input_bounds.lower(),
                self.edge_padding_low[axis],
                self.edge_padding_high[axis],
                self.interior_padding[axis],
                axis,
            )?
            .max(0);
            let maximum = input_bounds
                .upper()
                .map(|upper| {
                    padded_extent(
                        upper - 1,
                        self.edge_padding_low[axis],
                        self.edge_padding_high[axis],
                        self.interior_padding[axis],
                        axis,
                    )
                })
                .transpose()?;
            let output_bounds = output_dimension.bounds();
            if maximum.is_some_and(|maximum| maximum < output_bounds.lower() as i128)
                || output_bounds.upper().is_some_and(|upper| minimum >= upper as i128)
            {
                return Err(TypeError::invalid(format!(
                    "`{PAD_OPERATION_NAME}` output bounds {output_bounds} on axis {axis} cannot contain a padded \
                    extent derived from input bounds {input_bounds}",
                )));
            }
        }
        pad_output_type(
            input,
            padding_value,
            output_dimensions,
            self.edge_padding_low(),
            self.edge_padding_high(),
            self.interior_padding(),
        )
        .map(|r#type| vec![r#type.into()])
        .map_err(|error| match error {
            ProgramError::Type(error) => error,
            error => TypeError::invalid(error.to_string()),
        })
    }

    #[inline]
    fn effects(&self) -> Cow<'_, Effects> {
        Cow::Owned(Effects::explicit(if self.requires_runtime_assertion {
            EffectClasses::single(EffectClass::OrderedAssertion)
        } else {
            EffectClasses::NONE
        }))
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        self.render(formatter, indentation)
    }
}

impl_reference_dischargeable_operation!(@reference_free <T> PadOperation<T> where T: Type);

impl<C: Domain<Type = ArrayType, Value: Pad>> InterpretableOperation<C> for PadOperation<ArrayType> {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].pad(
            &inputs[1],
            self.edge_padding_low.as_slice(),
            self.edge_padding_high.as_slice(),
            self.interior_padding.as_slice(),
        )?])
    }
}

impl<C: Domain<Type = ArrayIrType, Value: DynamicPad>> InterpretableOperation<C> for PadOperation<ArrayIrType> {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, self.edge_padding_low().len() + 2, ProgramError);
        Ok(vec![inputs[0].dynamic_pad(
            &inputs[1],
            &inputs[2..],
            self.edge_padding_low(),
            self.edge_padding_high(),
            self.interior_padding(),
        )?])
    }
}

impl<T: Type, C: Context<Type = T, Operation: From<PadOperation<T>>>> PartiallyEvaluatableOperation<C>
    for PadOperation<T>
where
    PadOperation<T>: Operation<Type = T>,
{
}

// Batching rule for [`PadOperation`].
//
// A batched input with a replicated padding value keeps its batch axis by padding it with zero amounts: the lifted
// operation inserts `0` into all three padding vectors at the batch axis position. A batch-varying (batched) padding
// value is vectorized with a constant-size mask construction: pad the input with a representable placeholder, pad an
// all-true input mask with false, broadcast the per-item padding values over the padded result, and select those values
// at padding positions.
impl<C, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>> for PadOperation<ArrayType>
where
    C: Context<Type = ArrayType> + One<C::Value> + Zero<C::Value>,
    C::Value: Broadcast + Pad + Select + Transpose,
    PadOperation<ArrayType>: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        check_count!("input", inputs, 2, ProgramError);
        // Validate the padding contract first so that the ragged-axis check indexes the amounts with a known arity,
        // and reject unsupported ragged geometry before ordinary shape inference, which cannot express a changed
        // ragged extent. The amounts are indexed physically after inserting the mapped batch axis, and the lifted
        // vectors are reused by every branch below.
        validate_pad_inputs(
            &inputs[0].unbatched_type(),
            &inputs[1].unbatched_type(),
            self.edge_padding_low(),
            self.edge_padding_high(),
            self.interior_padding(),
        )?;
        let mut edge_padding_low = self.edge_padding_low().to_vec();
        let mut edge_padding_high = self.edge_padding_high().to_vec();
        let mut interior_padding = self.interior_padding().to_vec();
        if let Some(axis) = inputs[0].batch_axis_position() {
            edge_padding_low.insert(axis, 0);
            edge_padding_high.insert(axis, 0);
            interior_padding.insert(axis, 0);
        }
        let ragged_axes = validate_padding_ragged_axes(
            inputs[0].ragged_axes(),
            &edge_padding_low,
            &edge_padding_high,
            &interior_padding,
        )?;
        self.infer_output_types(&inputs.iter().map(ArrayBatch::unbatched_type).collect::<Vec<_>>(), &[])?;
        if inputs[1].batch_axis_position().is_none() {
            let Some(batch_axis) = inputs[0].batch_axis_position() else {
                let mut outputs = self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()])?;
                return Ok(vec![outputs.remove(0).with_ragged_axes(ragged_axes)?].into());
            };
            let lifted = PadOperation::new(edge_padding_low, edge_padding_high, interior_padding)?;
            let mut outputs =
                lifted.interpret_with_batch_axes(context, inputs, &[BatchAxis::from_position(batch_axis)])?;
            return Ok(vec![outputs.remove(0).with_ragged_axes(ragged_axes)?].into());
        }
        // A replicated input is aligned to a batch axis at position zero, so its amounts are lifted there as well.
        let batch_axis = inputs[0].batch_axis_position().unwrap_or(0);
        let input = P::match_axis(context, &inputs[0], Axis::from(batch_axis))?;
        if inputs[0].batch_axis_position().is_none() {
            edge_padding_low.insert(0, 0);
            edge_padding_high.insert(0, 0);
            interior_padding.insert(0, 0);
        }
        let ragged_axes = validate_padding_ragged_axes(
            input.ragged_axes(),
            &edge_padding_low,
            &edge_padding_high,
            &interior_padding,
        )?;
        let padding_type = inputs[1].unbatched_type();
        let placeholder_padding = context.parent().one(&padding_type)?;
        let padded = input.value().pad(
            &placeholder_padding,
            edge_padding_low.as_slice(),
            edge_padding_high.as_slice(),
            interior_padding.as_slice(),
        )?;
        let mask_input_type = input.r#type().into_owned().with_data_type(DataType::Boolean).with_layout(None);
        let mask_input = context.parent().one(&mask_input_type)?;
        let mask_padding_type = padding_type.with_data_type(DataType::Boolean).with_layout(None);
        let mask_padding = context.parent().zero(&mask_padding_type)?;
        let mask = mask_input.pad(
            &mask_padding,
            edge_padding_low.as_slice(),
            edge_padding_high.as_slice(),
            interior_padding.as_slice(),
        )?;
        let broadcasted_padding = inputs[1].value().broadcast(padded.r#type().into_owned(), &[batch_axis])?;
        let output = C::Value::select(&mask, &padded, &broadcasted_padding)?;
        Ok(vec![ArrayBatch::new(output, BatchAxis::from_position(batch_axis))?.with_ragged_axes(ragged_axes)?].into())
    }
}

// Batching rule for mixed [`PadOperation<ArrayIrType>`] instructions. Explicit result extents remain replicated. When
// the scalar padding value varies across the batch, the rule pads with a representable placeholder and uses a padded
// mask to select the broadcast per-item padding value without changing `pad`'s scalar input contract.
impl<C: Context<Type = ArrayIrType>> BatchableOperation<C, ArrayIrBatchingPolicy> for PadOperation<ArrayIrType>
where
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + ValueProjection<DimensionType, Projected: Value<Type = DimensionType>>,
    C::Value: DimensionSize + ValueProjection<ArrayType, Projected: Broadcast + Transpose + Value<Type = ArrayType>>,
    C::Operation: From<DynamicBroadcastOperation>
        + From<ConstantOperation<DimensionValue>>
        + From<DimensionSizeOperation>
        + From<OneOperation<ArrayType>>
        + From<PadOperation<ArrayIrType>>
        + OperationProjection<
            ArrayType,
            Projected: From<SelectOperation<ArrayType>>
                           + From<OneOperation<ArrayType>>
                           + From<ZeroOperation<ArrayType>>,
        >,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatchingPolicy>, BatchingError> {
        if inputs.len() < 2 {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
        }
        self.infer_output_types(&inputs.iter().map(ArrayIrBatch::unbatched_type).collect::<Vec<_>>(), &[])?;
        let (array_inputs, output_extents) = inputs.split_at(2);
        let [input, padding_value] = array_inputs else {
            unreachable!();
        };
        <&ArrayType>::try_from(&input.unbatched_type())?;
        <&ArrayType>::try_from(&padding_value.unbatched_type())?;
        for extent in output_extents {
            extent.validate_replicated_dimension()?;
        }
        let padding_value_batch = ArrayBatch::new(
            <C::Value as ValueProjection<ArrayType>>::into_projected(padding_value.value().clone())?,
            padding_value.batch_axis(),
        )?;
        let Some(batch_axis) = input
            .batch_axis_position()
            .or(Some(0).filter(|_| !padding_value_batch.batch_axis().is_replicated()))
        else {
            let ragged_axes = validate_padding_ragged_axes(
                input.ragged_axes(),
                self.edge_padding_low(),
                self.edge_padding_high(),
                self.interior_padding(),
            )?;
            let mut outputs = context.parent().bind(
                self.clone(),
                Vec::new(),
                &inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>(),
            )?;
            check_count!("output", outputs, 1, ProgramError);
            return Ok(vec![ArrayIrBatch::replicated(outputs.remove(0)).with_ragged_axes(ragged_axes)?].into());
        };

        let aligned_batch = driver.align_batch_axis(context, input.clone(), Axis::from(batch_axis))?;
        let aligned_ragged_axes = aligned_batch.ragged_axes().to_vec();
        let operand_batch = ArrayBatch::new(
            <C::Value as ValueProjection<ArrayType>>::into_projected(aligned_batch.into_value())?,
            BatchAxis::from_position(batch_axis),
        )?;
        let mut edge_padding_low = self.edge_padding_low().to_vec();
        edge_padding_low.insert(batch_axis, 0);
        let mut edge_padding_high = self.edge_padding_high().to_vec();
        edge_padding_high.insert(batch_axis, 0);
        let mut interior_padding = self.interior_padding().to_vec();
        interior_padding.insert(batch_axis, 0);
        let ragged_axes = validate_padding_ragged_axes(
            &aligned_ragged_axes,
            &edge_padding_low,
            &edge_padding_high,
            &interior_padding,
        )?;
        let operation = PadOperation::<ArrayIrType>::new(edge_padding_low, edge_padding_high, interior_padding)?;
        let mut lifted_output_extents = Vec::with_capacity(output_extents.len() + 1);
        lifted_output_extents.extend(output_extents[..batch_axis].iter().map(|extent| extent.value().clone()));
        lifted_output_extents.push(context.axis_extent().clone());
        lifted_output_extents.extend(output_extents[batch_axis..].iter().map(|extent| extent.value().clone()));

        if padding_value_batch.batch_axis().is_replicated() {
            let mut lifted_inputs = Vec::with_capacity(lifted_output_extents.len() + 2);
            lifted_inputs.push(<C::Value as ValueProjection<ArrayType>>::from_projected(operand_batch.into_value()));
            lifted_inputs.push(padding_value.value().clone());
            lifted_inputs.extend(lifted_output_extents);
            // The lifted payload is rebuilt through the conservative homogeneous-to-mixed conversion, so recompute
            // its proof against the actual batched signature to keep a proven pad effect-free.
            let operation = operation
                .with_input_types(&lifted_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>())?;
            let mut outputs = context.parent().bind(operation, Vec::new(), lifted_inputs.as_slice())?;
            check_count!("output", outputs, 1, ProgramError);
            return Ok(vec![
                ArrayIrBatch::new(outputs.remove(0), BatchAxis::from_position(batch_axis))?
                    .with_ragged_axes(ragged_axes)?,
            ]
            .into());
        }

        // `pad` requires a scalar padding input. Pad with a representable placeholder, build a Boolean mask for its
        // original positions, broadcast the mapped padding values across the result, and select them only outside those
        // positions.
        let array_context = ProjectedContext::<C, ArrayType>::new(context.parent().clone());
        let padding_scalar_type = padding_value_batch.unbatched_type();
        let placeholder_padding =
            <C::Value as ValueProjection<ArrayType>>::from_projected(array_context.one(&padding_scalar_type)?);
        let input = <C::Value as ValueProjection<ArrayType>>::from_projected(operand_batch.into_value());
        let mut padded_inputs = Vec::with_capacity(lifted_output_extents.len() + 2);
        padded_inputs.push(input.clone());
        padded_inputs.push(placeholder_padding);
        padded_inputs.extend(lifted_output_extents.iter().cloned());
        // Both pads of the decomposition share the lifted geometry, so one proof recomputation covers them.
        let operation = operation
            .with_input_types(&padded_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>())?;
        let mut padded = context.parent().bind(operation.clone(), Vec::new(), padded_inputs.as_slice())?;
        check_count!("output", padded, 1, ProgramError);
        let padded = padded.remove(0);

        let operand_type = <&ArrayType>::try_from(input.r#type().as_ref())?
            .clone()
            .with_data_type(DataType::Boolean)
            .with_layout(None);
        let mask_input_dimensions =
            operand_type
                .shape()
                .dimensions()
                .iter()
                .enumerate()
                .filter(|(_, dimension)| matches!(dimension, Dimension::Dynamic(_)))
                .map(|(axis, _)| {
                    if axis == batch_axis { Ok(context.axis_extent().clone()) } else { Ok(input.dimension_size(axis)?) }
                })
                .collect::<Result<Vec<_>, BatchingError>>()?;
        let mut mask_input =
            context
                .parent()
                .bind(OneOperation::new(operand_type), Vec::new(), mask_input_dimensions.as_slice())?;
        check_count!("output", mask_input, 1, ProgramError);
        let mask_input = mask_input.remove(0);
        let mask_padding_type = padding_scalar_type.with_data_type(DataType::Boolean).with_layout(None);
        let mask_padding =
            <C::Value as ValueProjection<ArrayType>>::from_projected(array_context.zero(&mask_padding_type)?);
        let mut mask_inputs = Vec::with_capacity(lifted_output_extents.len() + 2);
        mask_inputs.push(mask_input);
        mask_inputs.push(mask_padding);
        mask_inputs.extend(lifted_output_extents.iter().cloned());
        let mut mask = context.parent().bind(operation, Vec::new(), mask_inputs.as_slice())?;
        check_count!("output", mask, 1, ProgramError);
        let mask = mask.remove(0);

        let mut broadcast_inputs = Vec::with_capacity(lifted_output_extents.len() + 1);
        broadcast_inputs.push(<C::Value as ValueProjection<ArrayType>>::from_projected(
            padding_value_batch.move_axis(0)?.into_value(),
        ));
        broadcast_inputs.extend(lifted_output_extents);
        let mut broadcasted_padding = context.parent().bind(
            DynamicBroadcastOperation::new(vec![batch_axis]),
            Vec::new(),
            broadcast_inputs.as_slice(),
        )?;
        check_count!("output", broadcasted_padding, 1, ProgramError);
        let broadcasted_padding = broadcasted_padding.remove(0);

        let mask = <C::Value as ValueProjection<ArrayType>>::into_projected(mask)?;
        let padded = <C::Value as ValueProjection<ArrayType>>::into_projected(padded)?;
        let broadcasted_padding = <C::Value as ValueProjection<ArrayType>>::into_projected(broadcasted_padding)?;
        let mut output =
            array_context.bind(SelectOperation::new(), Vec::new(), &[mask, padded, broadcasted_padding])?;
        check_count!("output", output, 1, ProgramError);
        Ok(vec![
            ArrayIrBatch::new(
                <C::Value as ValueProjection<ArrayType>>::from_projected(output.remove(0)),
                BatchAxis::from_position(batch_axis),
            )?
            .with_ragged_axes(ragged_axes)?,
        ]
        .into())
    }
}

impl_differentiable_operation! {
    PadOperation<ArrayType>,
    jvp<C>
    where
        C: Context<Type = ArrayType> + Zero<C::Value>,
        C::Operation: From<PadOperation<ArrayType>>,
        C::Value: Pad,
    {
        |operation, context, _driver, inputs| {
            // Forward-mode rule for [`PadOperation`]: `pad` is linear in both the input and the padding value, so the
            // tangent pads the input tangent with the padding-value tangent using the same padding amounts.
            check_count!("input", inputs, 2, ProgramError);
            let primal = inputs[0].primal().pad(
                inputs[1].primal(),
                operation.edge_padding_low(),
                operation.edge_padding_high(),
                operation.interior_padding(),
            )?;
            // The pad needs both the input and padding-value tangents as real values, so materialize every structurally
            // zero side. The shared all-zero fast path normally short-circuits the case where both are zero; a direct
            // rule call with two structural zeros simply pads a materialized zero with a materialized zero.
            let operand_tangent = inputs[0].tangent().clone().materialize(context.tangent())?;
            let padding_tangent = inputs[1].tangent().clone().materialize(context.tangent())?;
            let tangent = operand_tangent.pad(
                &padding_tangent,
                operation.edge_padding_low(),
                operation.edge_padding_high(),
                operation.interior_padding(),
            )?;
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType>
            + From<OneOperation<ArrayType>>
            + From<PadOperation<ArrayType>>
            + From<SelectOperation<ArrayType>>
            + From<SliceOperation>
            + From<ReduceOperation>
            + From<ZeroOperation<ArrayType>>,
        Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<ArrayType>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // Transpose (vector-Jacobian product) for a [`PadOperation`].
            //
            // The forward map `(t, p) ↦ pad(t, p, low, high, interior)` writes input element `i` to output position
            // `low + i * (interior + 1)` along each axis and the padding value everywhere else, so its pullback splits
            // the output cotangent into two contributions:
            //
            //   - **Input cotangent**: slice the surviving input positions with stride `interior + 1`, then insert
            //     zeros at the cropped input positions.
            //   - **Padding-value cotangent**: pad an all-false input-shaped mask with `true`, select the output
            //     cotangent only at those padding positions, and sum the selected tensor. Selection rather than
            //     subtraction keeps non-finite cotangents at input positions from contaminating this contribution.
            //
            // A symbolic-zero output cotangent contributes nothing and is left to the accumulator defaults.
            let contributions = {
                // The rule stages into the tracing context only, so the transposition context is narrowed once up
                // front.
                let context: &mut TracingContext<V, O> = context;
                check_count!("input", inputs, 2, ProgramError);
                check_count!("output", outputs, 1, ProgramError);
                check_count!("accumulator", accumulators, 2, DifferentiationError);
                operation.infer_output_types(
                    &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
                    &[],
                )?;
                // A structural-zero output cotangent contributes nothing. Untouched accumulators default to structural
                // zeros when the transposition context collects its cotangents, so nothing is accumulated here.
                let MaybeZero::Value(cotangent) = &outputs[0] else {
                    return Ok(());
                };
                let input_cotangent = if inputs[0].is_unknown() {
                    let target_type = inputs[0].r#type().cotangent()?;
                    let mut starts = Vec::with_capacity(target_type.rank());
                    let mut limits = Vec::with_capacity(target_type.rank());
                    let mut strides = Vec::with_capacity(target_type.rank());
                    let mut low = Vec::with_capacity(target_type.rank());
                    let mut high = Vec::with_capacity(target_type.rank());
                    let mut empty = false;
                    for axis in 0..target_type.rank() {
                        let input_extent = target_type.dimension(axis).value().ok_or_else(|| {
                            TypeError::invalid(format!(
                                "`{PAD_OPERATION_NAME}` transpose requires a static input extent on axis {axis}"
                            ))
                        })? as i128;
                        let output_extent = cotangent.r#type().dimension(axis).value().ok_or_else(|| {
                            TypeError::invalid(format!(
                                "`{PAD_OPERATION_NAME}` transpose requires a static output extent on axis \
                                 {axis}"
                            ))
                        })? as i128;
                        let edge = operation.edge_padding_low[axis] as i128;
                        let stride = operation.interior_padding[axis] as i128 + 1;
                        // Keep only input indices whose padded coordinates survive cropping. Working in i128
                        // avoids negating i64::MIN and constructing an enormous intermediate dilated array.
                        let first = (-edge).div_euclid(stride) + i128::from((-edge).rem_euclid(stride) != 0);
                        let end = (output_extent - edge).div_euclid(stride)
                            + i128::from((output_extent - edge).rem_euclid(stride) != 0);
                        let first = first.clamp(0, input_extent);
                        let end = end.clamp(first, input_extent);
                        if first == end {
                            empty = true;
                            break;
                        }
                        // Surviving coordinates lie inside the static output extent, so they fit `usize`.
                        starts.push(usize::try_from(edge + first * stride).unwrap());
                        limits.push(usize::try_from(edge + (end - 1) * stride + 1).unwrap());
                        // With one surviving element, the stride is irrelevant and need not fit usize.
                        strides.push(if end - first == 1 { 1 } else { usize::try_from(stride).unwrap() });
                        low.push(i64::try_from(first).map_err(|_| {
                            TypeError::invalid(format!(
                                "`{PAD_OPERATION_NAME}` transpose low padding exceeds `i64` on axis {axis}"
                            ))
                        })?);
                        high.push(i64::try_from(input_extent - end).map_err(|_| {
                            TypeError::invalid(format!(
                                "`{PAD_OPERATION_NAME}` transpose high padding exceeds `i64` on axis {axis}"
                            ))
                        })?);
                    }
                    if empty {
                        MaybeZero::Zero(target_type)
                    } else {
                        let slice = SliceOperation::new(starts, limits).with_strides(strides)?;
                        let mut sliced =
                            context.stage_operation(slice, Vec::new(), std::slice::from_ref(cotangent))?;
                        check_count!("output", sliced, 1, ProgramError);
                        let zero = MaybeZero::Zero(cotangent.r#type().scalar_like()?)
                            .materialize(context)?;
                        let mut padded = context.stage_operation(
                            PadOperation::new(low, high, vec![0; target_type.rank()])?,
                            Vec::new(),
                            &[sliced.remove(0), zero],
                        )?;
                        check_count!("output", padded, 1, ProgramError);
                        MaybeZero::Value(padded.remove(0).unalign_cotangent(&target_type)?)
                    }
                } else {
                    MaybeZero::Zero(inputs[0].r#type().cotangent()?)
                };
                let padding_value_cotangent = if inputs[1].is_unknown() {
                    let mask_input_type =
                        inputs[0].r#type().cotangent()?.with_data_type(DataType::Boolean).with_layout(None);
                    let mask_padding_type =
                        inputs[1].r#type().cotangent()?.with_data_type(DataType::Boolean).with_layout(None);
                    let mask_input = MaybeZero::Zero(mask_input_type).materialize(context)?;
                    let no_inputs: [Tracer<TracingContext<V, O>>; 0] = [];
                    let mut mask_padding = context.stage_operation(
                        OneOperation::new(mask_padding_type),
                        Vec::new(),
                        &no_inputs,
                    )?;
                    check_count!("output", mask_padding, 1, ProgramError);
                    let mut mask = context.stage_operation(
                        operation.clone(),
                        Vec::new(),
                        &[mask_input, mask_padding.remove(0)],
                    )?;
                    check_count!("output", mask, 1, ProgramError);
                    let zero = MaybeZero::Zero(cotangent.r#type().into_owned()).materialize(context)?;
                    let mut selected = context.stage_operation(
                        SelectOperation::<ArrayType>::new(),
                        Vec::new(),
                        &[mask.remove(0), cotangent.clone(), zero],
                    )?;
                    check_count!("output", selected, 1, ProgramError);
                    let all_axes = (0..cotangent.r#type().rank()).collect::<Vec<_>>();
                    let mut reduced = context.stage_operation(
                        ReduceOperation::new(all_axes, ReductionKind::Sum),
                        Vec::new(),
                        &[selected.remove(0)],
                    )?;
                    check_count!("output", reduced, 1, ProgramError);
                    MaybeZero::Value(reduced.remove(0).unalign_cotangent(&inputs[1].r#type().cotangent()?)?)
                } else {
                    MaybeZero::Zero(inputs[1].r#type().cotangent()?)
                };
                vec![input_cotangent, padding_value_cotangent]
            };
            check_count!("input", contributions, accumulators.len(), ProgramError);
            for (accumulator, contribution) in accumulators.iter().zip(contributions) {
                accumulator.accumulate(context, contribution)?;
            }
            Ok(())
        }
    },
}

impl_differentiable_operation! {
    PadOperation<ArrayIrType>,
    jvp<C>
    where
        C: Context<Type = ArrayIrType> + Zero<C::Value>,
        C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
        C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
        C::Operation: ResidualZeroProvider<ArrayIrType, Operation = C::Operation>
            + From<DimensionSizeOperation>
            + From<DynamicShapeSliceOperation>
            + From<LinearCallOperation<ArrayIrType>>
            + From<PadOperation<ArrayIrType>>
            + From<ZeroOperation<ArrayType>>
            + From<ConstantOperation<DimensionValue>>
            + OperationProjection<
                ArrayType,
                Projected: From<OneOperation<ArrayType>>
                               + From<ReduceOperation>
                               + From<SelectOperation<ArrayType>>
                               + From<ZeroLikeOperation<ArrayType>>
                               + From<ZeroOperation<ArrayType>>,
            > + OperationProjection<DimensionType, Projected = DimensionOperation<DimensionValue>>,
        ProjectedValue<ArrayType, Tracer<NestedTracingContext<C>>>: ElementwiseDerivativeAlignment<ArrayType>,
    {
        |operation, context, _driver, inputs| {
            // Forward-mode rule for mixed pad. The explicit output extents are ordinary non-differentiated shape
            // values. Exact input geometry replays the mixed pad directly; dynamic geometry retains the exact input
            // shape and output extents so the linear transpose can reconstruct both the input and padding-value
            // cotangents.
            let destinations = context;
            if inputs.len() < 2 {
                return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
            }
            let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            let mut primal_outputs =
                destinations.primal().bind(operation.clone(), Vec::new(), primal_inputs.as_slice())?;
            check_count!("output", primal_outputs, 1, ProgramError);
            let output_primal = primal_outputs.remove(0);
            let tangent_primal = destinations.primal_to_tangent(output_primal.clone())?;
            let tangent_inputs = destinations.dual_primal_to_tangent(inputs)?;
            let (array_inputs, output_extents) = tangent_inputs.split_at(2);
            let tangent_context = destinations.tangent();
            let tangent = if array_inputs.iter().all(|input| input.tangent().is_zero()) {
                MaybeZero::Zero(tangent_primal.r#type().tangent()?)
            } else {
                let projected_context = ProjectedContext::<C, ArrayType>::new(tangent_context.clone());
                let mut materialized_inputs = array_inputs
                    .iter()
                    .map(|input| -> Result<C::Value, DifferentiationError> {
                        Ok(<C::Value as ValueProjection<ArrayType>>::from_projected(materialize_array_tangent(
                            &projected_context,
                            input,
                        )?))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let operand_cotangent_type =
                    <&ArrayType>::try_from(array_inputs[0].primal().r#type().as_ref())?.cotangent()?;
                if operand_cotangent_type
                    .shape()
                    .dimensions()
                    .iter()
                    .all(|dimension| matches!(dimension, Dimension::Static(_)))
                {
                    materialized_inputs.extend(output_extents.iter().map(|extent| extent.primal().clone()));
                    {
                        let mut outputs =
                            tangent_context.bind(operation.clone(), Vec::new(), materialized_inputs.as_slice())?;
                        check_count!("output", outputs, 1, ProgramError);
                        MaybeZero::Value(outputs.remove(0))
                    }
                } else {
                    let mut residuals = LinearResiduals::new();
                    let output_extents =
                        residuals.retain_all(output_extents.iter().map(|extent| extent.primal().clone()));
                    let operand_shape = residuals.retain_shape(tangent_context, array_inputs[0].primal())?;
                    let forward_operation = operation.clone();
                    let forward_output_extents = output_extents.clone();
                    let transpose_operation = operation.clone();
                    let transpose_operand_type = operand_cotangent_type.clone();
                    let transpose_padding_type =
                        <&ArrayType>::try_from(array_inputs[1].primal().r#type().as_ref())?.cotangent()?;
                    let transpose_output_type =
                        <&ArrayType>::try_from(tangent_primal.r#type().as_ref())?.cotangent()?;
                    let mut tangent = LinearCallOperation::stage(
                        tangent_context,
                        residuals.into_values(),
                        materialized_inputs,
                        move |residuals, linear_inputs| {
                            check_count!("input", linear_inputs, 2, ProgramError);
                            let mut pad_inputs = linear_inputs.to_vec();
                            pad_inputs.extend(forward_output_extents.iter().map(|index| residuals[*index].clone()));
                            linear_inputs[0].dispatch_domain().bind(
                                forward_operation,
                                Vec::new(),
                                pad_inputs.as_slice(),
                            )
                        },
                        move |residuals, output_cotangents| {
                            check_count!("output", output_cotangents, 1, ProgramError);
                            let transpose_context = output_cotangents[0].dispatch_domain();
                            let output_cotangent = output_cotangents[0].clone();
                            let input_extents = operand_shape.dimensions(&transpose_context, residuals)?;

                            let all_cropped = transpose_operand_type.shape().dimensions().iter().enumerate().any(
                                |(axis, dimension)| {
                                    dimension.bounds().upper().is_some_and(|upper| {
                                        if upper <= 1 {
                                            return true;
                                        }
                                        let last = ((upper - 2) as i128)
                                            .checked_mul(transpose_operation.interior_padding()[axis] as i128 + 1);
                                        last.and_then(|position| {
                                            position.checked_add(transpose_operation.edge_padding_low()[axis] as i128)
                                        })
                                        .is_some_and(|position| position < 0)
                                            || last
                                                .and_then(|position| {
                                                    position.checked_add(
                                                        1 + transpose_operation.edge_padding_high()[axis] as i128,
                                                    )
                                                })
                                                .is_some_and(|extent_after_high_crop| extent_after_high_crop <= 0)
                                    })
                                },
                            );
                            let input_cotangent = if all_cropped {
                                let dimensions = transpose_operand_type
                                    .shape()
                                    .dimensions()
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, dimension)| matches!(dimension, Dimension::Dynamic(_)))
                                    .map(|(axis, _)| input_extents[axis].clone())
                                    .collect::<Vec<_>>();
                                let mut zeros = transpose_context.bind(
                                    ZeroOperation::new(transpose_operand_type.clone()),
                                    Vec::new(),
                                    &dimensions,
                                )?;
                                check_count!("output", zeros, 1, ProgramError);
                                zeros.remove(0)
                            } else {
                                // Inverse edge padding first recovers the dilated input. Its exact result extents are
                                // `n + max(n - 1, 0) * interior`, derived from the retained input geometry.
                                let mut dilated_extents = Vec::with_capacity(transpose_operand_type.rank());
                                for (axis, input_extent) in input_extents.iter().enumerate() {
                                    let interior = transpose_operation.interior_padding()[axis];
                                    if interior == 0
                                        || transpose_operand_type
                                            .dimension(axis)
                                            .bounds()
                                            .upper()
                                            .is_some_and(|upper| upper <= 2)
                                    {
                                        dilated_extents.push(input_extent.clone());
                                        continue;
                                    }
                                    let mut one = transpose_context.bind(
                                        DimensionOperation::from(ConstantOperation::new(DimensionValue::constant(1)?)),
                                        Vec::new(),
                                        &[],
                                    )?;
                                    check_count!("output", one, 1, ProgramError);
                                    let one = one.remove(0);
                                    let input_type =
                                        <&DimensionType>::try_from(input_extent.r#type().as_ref())?.clone();
                                    let one_type = <&DimensionType>::try_from(one.r#type().as_ref())?.clone();
                                    let mut less_one = transpose_context.bind(
                                        DimensionOperation::SaturatingSub(DimensionSaturatingSubOperation::new(
                                            &input_type,
                                            &one_type,
                                        )?),
                                        Vec::new(),
                                        &[input_extent.clone(), one],
                                    )?;
                                    check_count!("output", less_one, 1, ProgramError);
                                    let less_one = less_one.remove(0);
                                    let mut interior_extent = transpose_context.bind(
                                        DimensionOperation::from(ConstantOperation::new(DimensionValue::constant(
                                            interior,
                                        )?)),
                                        Vec::new(),
                                        &[],
                                    )?;
                                    check_count!("output", interior_extent, 1, ProgramError);
                                    let interior_extent = interior_extent.remove(0);
                                    let less_one_type = <&DimensionType>::try_from(less_one.r#type().as_ref())?.clone();
                                    let interior_extent_type =
                                        <&DimensionType>::try_from(interior_extent.r#type().as_ref())?.clone();
                                    let mut gaps = transpose_context.bind(
                                        DimensionOperation::Mul(DimensionMulOperation::new(
                                            &less_one_type,
                                            &interior_extent_type,
                                        )?),
                                        Vec::new(),
                                        &[less_one, interior_extent],
                                    )?;
                                    check_count!("output", gaps, 1, ProgramError);
                                    let gaps = gaps.remove(0);
                                    let gaps_type = <&DimensionType>::try_from(gaps.r#type().as_ref())?.clone();
                                    let mut dilated_extent = transpose_context.bind(
                                        DimensionOperation::Add(DimensionAddOperation::new(&input_type, &gaps_type)?),
                                        Vec::new(),
                                        &[input_extent.clone(), gaps],
                                    )?;
                                    check_count!("output", dilated_extent, 1, ProgramError);
                                    dilated_extents.push(dilated_extent.remove(0));
                                }

                                let inverse_low = transpose_operation
                                    .edge_padding_low()
                                    .iter()
                                    .enumerate()
                                    .map(|(axis, padding)| {
                                        padding.checked_neg().ok_or_else(|| {
                                            TypeError::invalid(format!(
                                                "`{PAD_OPERATION_NAME}` transpose cannot negate `edge_padding_low` at \
                                                 axis {axis} with value {padding}",
                                            ))
                                        })
                                    })
                                    .collect::<Result<Vec<_>, _>>()?;
                                let inverse_high = transpose_operation
                                    .edge_padding_high()
                                    .iter()
                                    .enumerate()
                                    .map(|(axis, padding)| {
                                        padding.checked_neg().ok_or_else(|| {
                                            TypeError::invalid(format!(
                                                "`{PAD_OPERATION_NAME}` transpose cannot negate `edge_padding_high` at \
                                                 axis {axis} with value {padding}",
                                            ))
                                        })
                                    })
                                    .collect::<Result<Vec<_>, _>>()?;
                                let mut zero = transpose_context.bind(
                                    <C::Operation as OperationProjection<ArrayType>>::Projected::from(
                                        ZeroOperation::new(transpose_padding_type.clone()),
                                    ),
                                    Vec::new(),
                                    &[],
                                )?;
                                check_count!("output", zero, 1, ProgramError);
                                let zero = zero.remove(0);
                                let mut inverse_inputs = vec![output_cotangent.clone(), zero];
                                inverse_inputs.extend(dilated_extents);
                                // Recompute the proof against the actual inverse signature: static and unchanged
                                // axes need no runtime assertion, while derived dilated extents keep one.
                                let inverse_operation =
                                    PadOperation::<ArrayIrType>::new(
                                        inverse_low,
                                        inverse_high,
                                        vec![0; transpose_operand_type.rank()],
                                    )?
                                    .with_input_types(
                                        &inverse_inputs
                                            .iter()
                                            .map(|input| input.r#type().into_owned())
                                            .collect::<Vec<_>>(),
                                    )?;
                                let mut unpadded =
                                    transpose_context.bind(inverse_operation, Vec::new(), inverse_inputs.as_slice())?;
                                check_count!("output", unpadded, 1, ProgramError);
                                let unpadded = unpadded.remove(0);
                                let mut start_zero = transpose_context.bind(
                                    DimensionOperation::from(ConstantOperation::new(DimensionValue::constant(0)?)),
                                    Vec::new(),
                                    &[],
                                )?;
                                check_count!("output", start_zero, 1, ProgramError);
                                let start_zero = start_zero.remove(0);
                                let starts = vec![start_zero; transpose_operand_type.rank()];
                                let mut slice_inputs = Vec::with_capacity(1 + 2 * transpose_operand_type.rank());
                                slice_inputs.push(unpadded);
                                slice_inputs.extend(starts);
                                slice_inputs.extend(input_extents.iter().cloned());
                                let strides = transpose_operation
                                    .interior_padding()
                                    .iter()
                                    .enumerate()
                                    .map(|(axis, padding)| {
                                        if transpose_operand_type
                                            .dimension(axis)
                                            .bounds()
                                            .upper()
                                            .is_some_and(|upper| upper <= 2)
                                        {
                                            return Ok(1);
                                        }
                                        padding.checked_add(1).ok_or_else(|| {
                                            TypeError::invalid(format!(
                                                "`{PAD_OPERATION_NAME}` transpose stride overflows usize on axis \
                                                 {axis}",
                                            ))
                                        })
                                    })
                                    .collect::<Result<Vec<_>, _>>()?;
                                let mut input_cotangent = transpose_context.bind(
                                    DynamicShapeSliceOperation::new(transpose_operand_type.rank())
                                        .with_strides(strides)?,
                                    Vec::new(),
                                    slice_inputs.as_slice(),
                                )?;
                                check_count!("output", input_cotangent, 1, ProgramError);
                                input_cotangent.remove(0)
                            };

                            // Select padding positions before summing so non-finite cotangents at input positions
                            // cannot contaminate the padding-value contribution.
                            let mask_input_type =
                                transpose_operand_type.clone().with_data_type(DataType::Boolean).with_layout(None);
                            let mask_input_extents = mask_input_type
                                .shape()
                                .dimensions()
                                .iter()
                                .enumerate()
                                .filter(|(_, dimension)| matches!(dimension, Dimension::Dynamic(_)))
                                .map(|(axis, _)| input_extents[axis].clone())
                                .collect::<Vec<_>>();
                            let mut mask_input = transpose_context.bind(
                                ZeroOperation::new(mask_input_type),
                                Vec::new(),
                                mask_input_extents.as_slice(),
                            )?;
                            check_count!("output", mask_input, 1, ProgramError);
                            let mask_input = mask_input.remove(0);
                            let mut mask_padding = transpose_context.bind(
                                <C::Operation as OperationProjection<ArrayType>>::Projected::from(OneOperation::new(
                                    transpose_padding_type.clone().with_data_type(DataType::Boolean).with_layout(None),
                                )),
                                Vec::new(),
                                &[],
                            )?;
                            check_count!("output", mask_padding, 1, ProgramError);
                            let mask_padding = mask_padding.remove(0);
                            let mut mask_inputs = vec![mask_input, mask_padding];
                            mask_inputs.extend(output_extents.iter().map(|index| residuals[*index].clone()));
                            let mut mask =
                                transpose_context.bind(transpose_operation, Vec::new(), mask_inputs.as_slice())?;
                            check_count!("output", mask, 1, ProgramError);
                            let mask = mask.remove(0);
                            let output_zero_extents = transpose_output_type
                                .shape()
                                .dimensions()
                                .iter()
                                .enumerate()
                                .filter(|(_, dimension)| matches!(dimension, Dimension::Dynamic(_)))
                                .map(|(axis, _)| residuals[output_extents[axis]].clone())
                                .collect::<Vec<_>>();
                            let mut output_zero = transpose_context.bind(
                                ZeroOperation::new(transpose_output_type.clone()),
                                Vec::new(),
                                output_zero_extents.as_slice(),
                            )?;
                            check_count!("output", output_zero, 1, ProgramError);
                            let output_zero = output_zero.remove(0);
                            let mut selected = transpose_context.bind(
                                <C::Operation as OperationProjection<ArrayType>>::Projected::from(
                                    SelectOperation::new(),
                                ),
                                Vec::new(),
                                &[mask, output_cotangent, output_zero],
                            )?;
                            check_count!("output", selected, 1, ProgramError);
                            let selected = selected.remove(0);
                            let mut padding_cotangent = transpose_context.bind(
                                <C::Operation as OperationProjection<ArrayType>>::Projected::from(
                                    ReduceOperation::new(
                                        (0..transpose_output_type.rank()).collect(),
                                        ReductionKind::Sum,
                                    ),
                                ),
                                Vec::new(),
                                &[selected],
                            )?;
                            check_count!("output", padding_cotangent, 1, ProgramError);
                            let padding_cotangent = padding_cotangent.remove(0);
                            Ok(vec![
                                ValueProjection::<ArrayType>::into_projected(input_cotangent)?
                                    .unalign_cotangent(&transpose_operand_type)?
                                    .into_value(),
                                ValueProjection::<ArrayType>::into_projected(padding_cotangent)?
                                    .unalign_cotangent(&transpose_padding_type)?
                                    .into_value(),
                            ])
                        },
                    )?;
                    check_count!("output", tangent, 1, ProgramError);
                    MaybeZero::Value(tangent.remove(0))
                }
            };
            Ok(vec![DifferentiationDual::new(output_primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
        O: Operation<Type = ArrayIrType> + OperationProjection<ArrayType>,
        <O as OperationProjection<ArrayType>>::Projected: From<PadOperation<ArrayType>>
            + TransposableOperation<
                <V as ValueProjection<ArrayType>>::Projected,
                <O as OperationProjection<ArrayType>>::Projected,
            >,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // Direct transposition rule for mixed pad. Static input and output geometry delegate to the homogeneous
            // array pullback, while every explicit output extent receives a structural-zero cotangent. Dynamic geometry
            // requires linearization so [`DifferentiableOperation::jvp`] can retain the exact primal extents as
            // residuals.
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);

            if inputs.len() < 2 {
                return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
            }
            operation
                .infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(), &[])?;
            // A structural-zero output cotangent contributes nothing. Untouched accumulators default to structural
            // zeros when the transposition context collects its cotangents, so nothing is accumulated here.
            if outputs[0].is_zero() {
                return Ok(());
            }
            let (array_inputs, output_extents) = inputs.split_at(2);
            if array_inputs.iter().any(|input| {
                <&ArrayType>::try_from(input.r#type().as_ref()).is_ok_and(|r#type| {
                    r#type.shape().dimensions().iter().any(|dimension| matches!(dimension, Dimension::Dynamic(_)))
                })
            }) || output_extents.iter().any(|extent| {
                <&DimensionType>::try_from(extent.r#type().as_ref())
                    .is_ok_and(|r#type| matches!(r#type.to_dimension(), Dimension::Dynamic(_)))
            }) {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "direct `{PAD_OPERATION_NAME}` transposition with dynamic extents requires linearization so \
                         that the primal geometry can be retained as residuals",
                    ),
                }
                .into());
            }

            let projected_operation = <O as OperationProjection<ArrayType>>::Projected::from(
                PadOperation::<ArrayType>::from(operation.clone()),
            );
            // The explicit output extents are shape operands with no cotangent contribution, so their accumulators
            // are left untouched and default to structural zeros.
            transpose_projected_operation(context, &projected_operation, array_inputs, outputs, &accumulators[..2])?;
            Ok(())
        }
    },
}

/// Represents the ability to add edge and interior padding filled with a scalar value. Negative edge padding crops the
/// input after interior padding has been inserted. Along each axis, input coordinate `i` moves to
/// `edge_padding_low + i * (interior_padding + 1)`; coordinates outside the output are discarded, and every remaining
/// output position not occupied by an input element holds `padding_value`.
///
/// For an input extent `d`, the output extent is `d + max(d - 1, 0) * interior_padding + edge_padding_low +
/// edge_padding_high`. An empty input axis therefore contributes no interior padding. Each resulting extent must be
/// nonnegative and fit in [`usize`]. All three configuration slices must contain one entry per input axis, and the
/// padding value must be a scalar with the input's element data type and memory space.
///
/// An effective identity returns its input unchanged after validating the inputs. Otherwise, padding preserves memory
/// placement, clears explicit physical layout, and infers compatible sharding and distributed dependency state. Use
/// [`DynamicPad`] when changed output extents require first-class dimension inputs. Ordinary [`Self::pad`] also
/// preserves dynamic axes whose extent remains unchanged, including axes with balanced edge padding and no interior
/// padding.
///
/// These are the constant-value primitive semantics of StableHLO's [`pad`](https://openxla.org/stablehlo/spec#pad).
/// Reflection, wrapping, statistical padding, and per-edge values are higher-level operations, not modes of this
/// primitive. Interior padding is the transpose counterpart of strided [`SliceOperation`]: a stride of `s` corresponds
/// to inserting `s - 1` padding elements between adjacent input elements.
///
/// # Example
///
/// The following example pads a vector before, after, and between its input elements:
///
/// ```rust
/// # use ryft_core::{Array, ArrayType, DataType, Pad, ProgramError};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// let input = Array::from_elements(ArrayType::new_static(DataType::I32, [3]), &[1i32, 2, 3])?;
/// let value = Array::from_elements(ArrayType::scalar(DataType::I32), &[0i32])?;
/// let output = input.pad(&value, &[1], &[2], &[1])?;
/// assert_eq!(output.elements::<i32>()?, vec![0, 1, 0, 2, 0, 3, 0, 0]);
///
/// // Cropping is applied after inserting the interior zeros.
/// let cropped = input.pad_with_config(&value, &[(-1, 0, 1)])?;
/// assert_eq!(cropped.elements::<i32>()?, vec![0, 2, 0, 3]);
/// # Ok(())
/// # }
/// ```
pub trait Pad: Sized {
    /// Pads `self` with `padding_value` using the provided edge and interior padding amounts. Refer to the
    /// documentation of this trait for more information on what this operation does.
    ///
    /// # Parameters
    ///
    ///   - `padding_value`: Rank-0 scalar with the input's data type, written into every padding position.
    ///   - `edge_padding_low`: Padding added before the first element of each input axis.
    ///   - `edge_padding_high`: Padding added after the last element of each input axis.
    ///   - `interior_padding`: Padding added between any two adjacent elements of each input axis.
    fn pad(
        &self,
        padding_value: &Self,
        edge_padding_low: &[i64],
        edge_padding_high: &[i64],
        interior_padding: &[usize],
    ) -> Result<Self, ProgramError>;

    /// Pads using one `(low, high, interior)` tuple per input axis. This is equivalent to [`Self::pad`], with the three
    /// configuration slices assembled from those tuples. Negative edge amounts crop after interior padding; interior
    /// amounts are nonnegative. An empty configuration applies to a scalar input.
    ///
    /// # Parameters
    ///
    ///   - `padding_value`: Scalar value with the input's element data type and memory space.
    ///   - `padding_config`: One tuple per input axis, in axis order. Each tuple gives the number of values to add
    ///     before the input, after the input, and between adjacent input elements, respectively.
    fn pad_with_config(
        &self,
        padding_value: &Self,
        padding_config: &[(i64, i64, usize)],
    ) -> Result<Self, ProgramError> {
        let edge_padding_low = padding_config.iter().map(|&(low, _, _)| low).collect::<Vec<_>>();
        let edge_padding_high = padding_config.iter().map(|&(_, high, _)| high).collect::<Vec<_>>();
        let interior_padding = padding_config.iter().map(|&(_, _, interior)| interior).collect::<Vec<_>>();
        self.pad(padding_value, &edge_padding_low, &edge_padding_high, &interior_padding)
    }
}

impl Pad for ArrayType {
    fn pad(
        &self,
        padding_value: &Self,
        edge_padding_low: &[i64],
        edge_padding_high: &[i64],
        interior_padding: &[usize],
    ) -> Result<Self, ProgramError> {
        validate_pad_inputs(self, padding_value, edge_padding_low, edge_padding_high, interior_padding)?;
        let rank = self.rank();
        if is_effective_identity(self, edge_padding_low, edge_padding_high, interior_padding) {
            return Ok(self.clone());
        }
        let mut output_dimensions = Vec::with_capacity(rank);
        for axis in 0..rank {
            let dimension = self.dimension(axis);
            let output_dimension = match dimension {
                Dimension::Static(size) => Dimension::Static(static_padded_extent(
                    size,
                    edge_padding_low[axis],
                    edge_padding_high[axis],
                    interior_padding[axis],
                    axis,
                )?),
                Dimension::Dynamic(variable)
                    if i128::from(edge_padding_low[axis]) + i128::from(edge_padding_high[axis]) == 0
                        && (interior_padding[axis] == 0
                            || variable.bounds().upper().is_some_and(|upper| upper <= 2)) =>
                {
                    // This axis retains its extent, even if another axis is padded or balanced edge padding moves its
                    // elements. Keeping the existing identity requires no new runtime dimension computation.
                    Dimension::Dynamic(variable)
                }
                Dimension::Dynamic(variable) => {
                    if let Some(upper) = variable.bounds().upper() {
                        let maximum_input_extent = upper - 1;
                        let maximum_output_extent = padded_extent(
                            maximum_input_extent,
                            edge_padding_low[axis],
                            edge_padding_high[axis],
                            interior_padding[axis],
                            axis,
                        )?;
                        if maximum_output_extent < 0 {
                            return Err(TypeError::invalid(format!(
                                "`{PAD_OPERATION_NAME}` output size is negative ({maximum_output_extent}) on dynamic \
                                axis {axis} \
                                 even at its maximum input extent {maximum_input_extent}",
                            ))
                            .into());
                        }
                    }
                    return Err(TypeError::invalid(format!(
                        "`{PAD_OPERATION_NAME}` dynamic axis {axis} requires an explicit result-dimension input",
                    ))
                    .into());
                }
            };
            output_dimensions.push(output_dimension);
        }
        pad_output_type(self, padding_value, output_dimensions, edge_padding_low, edge_padding_high, interior_padding)
    }
}

impl Pad for Array {
    fn pad(
        &self,
        padding_value: &Self,
        edge_padding_low: &[i64],
        edge_padding_high: &[i64],
        interior_padding: &[usize],
    ) -> Result<Self, ProgramError> {
        let output_type = self.r#type().pad(
            padding_value.r#type().as_ref(),
            edge_padding_low,
            edge_padding_high,
            interior_padding,
        )?;
        if is_effective_identity(self.r#type().as_ref(), edge_padding_low, edge_padding_high, interior_padding) {
            return Ok(self.clone());
        }
        let output_shape = output_type.static_shape().unwrap();
        let rank = self.r#type().rank();
        let input_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let padding_addressing = ArrayAddressing::new(padding_value.r#type().into_owned())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let padding_bytes = &padding_value.storage_bytes()[padding_addressing.byte_range_for_flat_index(0)];
        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        // Structural-zero arrays have no element bytes, even when their logical shape is enormous. Empty outputs
        // likewise need no coordinate traversal or fill operation.
        if output_addressing.element_byte_width() == 0 || output_addressing.element_count() == 0 {
            return Ok(Self::new_unchecked(output_type, Arc::new(bytes)));
        }
        // The padded type carries no explicit layout, so its storage is dense row-major and is filled in bulk.
        for output_bytes in bytes.chunks_exact_mut(output_addressing.element_byte_width()) {
            output_bytes.copy_from_slice(padding_bytes);
        }
        if input_addressing.element_count() == 0 {
            return Ok(Self::new_unchecked(output_type, Arc::new(bytes)));
        }
        let mut input_index = vec![0usize; rank];
        let mut output_index = vec![0usize; rank];
        let mut written = 0usize;
        'elements: while written < input_addressing.element_count() {
            for axis in 0..rank {
                let input_coordinate = i128::try_from(input_index[axis]).map_err(|_| {
                    TypeError::invalid(format!("`{PAD_OPERATION_NAME}` input index is too large on axis {axis}"))
                })?;
                let stride =
                    i128::try_from(interior_padding[axis]).ok().and_then(|padding| padding.checked_add(1)).ok_or_else(
                        || TypeError::invalid(format!("`{PAD_OPERATION_NAME}` stride is too large on axis {axis}")),
                    )?;
                let output_coordinate = i128::from(edge_padding_low[axis])
                    .checked_add(input_coordinate.checked_mul(stride).ok_or_else(|| {
                        TypeError::invalid(format!("`{PAD_OPERATION_NAME}` output index overflows on axis {axis}"))
                    })?)
                    .ok_or_else(|| {
                        TypeError::invalid(format!("`{PAD_OPERATION_NAME}` output index overflows on axis {axis}"))
                    })?;
                let output_extent = i128::try_from(output_shape[axis]).map_err(|_| {
                    TypeError::invalid(format!("`{PAD_OPERATION_NAME}` output extent is too large on axis {axis}"))
                })?;
                if output_coordinate < 0 || output_coordinate >= output_extent {
                    written += 1;
                    input_addressing.advance_index(&mut input_index);
                    continue 'elements;
                }
                output_index[axis] = usize::try_from(output_coordinate).map_err(|_| {
                    TypeError::invalid(format!("`{PAD_OPERATION_NAME}` output index is too large on axis {axis}"))
                })?;
            }
            bytes[output_addressing.byte_range_unchecked(&output_index)]
                .copy_from_slice(&self.storage_bytes()[input_addressing.byte_range_unchecked(&input_index)]);
            written += 1;
            input_addressing.advance_index(&mut input_index);
        }
        Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
    }
}

impl<V: Value<Type = ArrayType>> Pad for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<PadOperation<ArrayType>>,
{
    fn pad(
        &self,
        padding_value: &Self,
        edge_padding_low: &[i64],
        edge_padding_high: &[i64],
        interior_padding: &[usize],
    ) -> Result<Self, ProgramError> {
        self.r#type()
            .pad(padding_value.r#type().as_ref(), edge_padding_low, edge_padding_high, interior_padding)?;
        if is_effective_identity(self.r#type().as_ref(), edge_padding_low, edge_padding_high, interior_padding) {
            return Ok(self.clone());
        }
        let mut outputs = self.dispatch_domain().bind(
            PadOperation::new(edge_padding_low.to_vec(), edge_padding_high.to_vec(), interior_padding.to_vec())?,
            Vec::new(),
            &[self.clone(), padding_value.clone()],
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Represents padding with explicit output dimensions and static padding configuration. The array and scalar inputs
/// follow [`Pad`]'s semantics. Each output axis additionally has a dimension input whose value must equal the padded
/// extent computed from the corresponding input axis. All output axes are supplied, including static and unchanged
/// axes, so the staged operation retains a complete description of its runtime geometry.
///
/// Exact dimension types yield static output axes. Other dimension types preserve their identities and bounds; bounds
/// constrain possible runtime extents rather than determining the actual result size. Runtime checks reject negative
/// padded sizes or supplied dimensions that disagree with the padding formula. The padding configuration itself is
/// static; this capability does not represent StableHLO's separate runtime-padding-amount operation `dynamic_pad`.
///
/// # Example
///
/// The following example pads a mixed array value using a first-class output extent. Context-carrying mixed values use
/// the same function to stage [`PadOperation<ArrayIrType>`]:
///
/// ```rust
/// # use ryft_core::{Array, ArrayIrValue, ArrayType, DataType, DimensionValue, DynamicPad, ProgramError};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// let input = ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[1i32, 2])?);
/// let value = ArrayIrValue::Array(Array::from_elements(ArrayType::scalar(DataType::I32), &[9i32])?);
/// let dimension = ArrayIrValue::Dimension(DimensionValue::constant(5)?);
/// let output = input.dynamic_pad(&value, &[dimension], &[1], &[1], &[1])?;
/// let expected = ArrayIrValue::Array(Array::from_elements(
///     ArrayType::new_static(DataType::I32, [5]),
///     &[9i32, 1, 9, 2, 9],
/// )?);
/// assert_eq!(output, expected);
/// # Ok(())
/// # }
/// ```
pub trait DynamicPad: Value<Type = ArrayIrType> + Sized {
    /// Pads `self` and validates the supplied result dimensions before returning the array value. Invalid input kinds,
    /// ranks, element data types, configuration lengths, and incompatible geometry return an error.
    ///
    /// # Parameters
    ///
    ///   - `padding_value`: Scalar array with the input's element data type and memory space.
    ///   - `output_dimensions`: One dimension value per output axis. Each value must equal
    ///     `d + max(d - 1, 0) * interior + low + high` for that axis's input extent `d`. Repeated dimension identities
    ///     must denote equal runtime sizes.
    ///   - `edge_padding_low`: Signed edge padding before each input axis; negative amounts crop after interior
    ///     padding.
    ///   - `edge_padding_high`: Signed edge padding after each input axis; negative amounts crop after interior
    ///     padding.
    ///   - `interior_padding`: Number of padding values between adjacent input elements on each axis. An empty input
    ///     axis has no adjacent pairs and contributes no interior padding.
    fn dynamic_pad(
        &self,
        padding_value: &Self,
        output_dimensions: &[Self],
        edge_padding_low: &[i64],
        edge_padding_high: &[i64],
        interior_padding: &[usize],
    ) -> Result<Self, ProgramError>;
}

impl<A: Value<Type = ArrayType> + Pad + DimensionSize<usize>> DynamicPad for ArrayIrValue<A> {
    fn dynamic_pad(
        &self,
        padding_value: &Self,
        output_dimensions: &[Self],
        edge_padding_low: &[i64],
        edge_padding_high: &[i64],
        interior_padding: &[usize],
    ) -> Result<Self, ProgramError> {
        let input = <Self as ValueProjection<ArrayType>>::projected(self)?;
        let padding_value = <Self as ValueProjection<ArrayType>>::projected(padding_value)?;
        validate_pad_inputs(
            input.r#type().as_ref(),
            padding_value.r#type().as_ref(),
            edge_padding_low,
            edge_padding_high,
            interior_padding,
        )?;
        check_count!("input", output_dimensions, input.r#type().rank(), ProgramError);
        // Binding every explicit extent to its dimension identity rejects repeated identities that denote different
        // runtime sizes.
        let mut refinements = ArrayTypeRefinements::default();
        for (axis, dimension) in output_dimensions.iter().enumerate() {
            let dimension = <Self as ValueProjection<DimensionType>>::projected(dimension)?;
            refinements.bind(dimension.r#type().variable(), dimension.extent())?;
            let actual_extent = static_padded_extent(
                input.dimension_size(axis)?,
                edge_padding_low[axis],
                edge_padding_high[axis],
                interior_padding[axis],
                axis,
            )?;
            if actual_extent != dimension.extent() {
                return Err(ProgramError::InvalidArgument {
                    message: format!(
                        "`{PAD_OPERATION_NAME}` output axis {axis} has extent {actual_extent}, but its explicit \
                         extent input is {}",
                        dimension.extent(),
                    ),
                });
            }
        }
        Ok(Self::Array(input.pad(padding_value, edge_padding_low, edge_padding_high, interior_padding)?))
    }
}

impl<V: Value<Type = ArrayIrType>> DynamicPad for V
where
    V::DispatchDomain: Context<Type = ArrayIrType, Operation: From<PadOperation<ArrayIrType>>>,
{
    fn dynamic_pad(
        &self,
        padding_value: &Self,
        output_dimensions: &[Self],
        edge_padding_low: &[i64],
        edge_padding_high: &[i64],
        interior_padding: &[usize],
    ) -> Result<Self, ProgramError> {
        let operation = PadOperation::<ArrayIrType>::new(
            edge_padding_low.to_vec(),
            edge_padding_high.to_vec(),
            interior_padding.to_vec(),
        )?;
        let mut inputs = Vec::with_capacity(2 + output_dimensions.len());
        inputs.push(self.clone());
        inputs.push(padding_value.clone());
        inputs.extend_from_slice(output_dimensions);
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let operation = operation.with_input_types(&input_types)?;
        let mut outputs = self.dispatch_domain().bind(operation, Vec::new(), &inputs)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Preserves ragged geometry only when padding leaves its data and extent-index axes unchanged.
fn validate_padding_ragged_axes<V: Value>(
    ragged_axes: &[RaggedAxis<V>],
    edge_padding_low: &[i64],
    edge_padding_high: &[i64],
    interior_padding: &[usize],
) -> Result<Vec<RaggedAxis<V>>, BatchingError> {
    for ragged_axis in ragged_axes {
        for axis in std::iter::once(ragged_axis.axis()).chain(ragged_axis.extent_axes().iter().copied()) {
            let (Some(low), Some(high), Some(interior)) =
                (edge_padding_low.get(axis), edge_padding_high.get(axis), interior_padding.get(axis))
            else {
                return Err(BatchingError::InvalidBatchMetadata {
                    message: format!(
                        "`{PAD_OPERATION_NAME}` batching found ragged axis {axis} outside the padded rank {}",
                        edge_padding_low.len(),
                    ),
                });
            };
            if *low != 0 || *high != 0 || *interior != 0 {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "`{PAD_OPERATION_NAME}` batching cannot change a ragged axis or an axis indexing its extents"
                    ),
                }
                .into());
            }
        }
    }
    Ok(ragged_axes.to_vec())
}

/// Returns whether this padding geometry leaves every possible element and its position unchanged.
fn is_effective_identity(
    input_type: &ArrayType,
    edge_padding_low: &[i64],
    edge_padding_high: &[i64],
    interior_padding: &[usize],
) -> bool {
    edge_padding_low.iter().all(|padding| *padding == 0)
        && edge_padding_high.iter().all(|padding| *padding == 0)
        && input_type.shape().dimensions().iter().zip(interior_padding).all(|(dimension, padding)| {
            *padding == 0
                || matches!(dimension, Dimension::Static(0 | 1))
                || matches!(
                    dimension,
                    Dimension::Dynamic(variable) if variable.bounds().upper().is_some_and(|upper| upper <= 2)
                )
        })
}

/// Validates the input types and padding-vector arity shared by both padding type contracts.
fn validate_pad_inputs(
    input: &ArrayType,
    padding_value: &ArrayType,
    edge_padding_low: &[i64],
    edge_padding_high: &[i64],
    interior_padding: &[usize],
) -> Result<(), ProgramError> {
    if input.data_type() != padding_value.data_type() {
        return Err(TypeError::invalid(format!(
            "`{}` input data type `{}` does not match padding value data type `{}`",
            PAD_OPERATION_NAME,
            input.data_type(),
            padding_value.data_type(),
        ))
        .into());
    }

    if padding_value.rank() != 0 {
        return Err(TypeError::invalid(format!(
            "`{PAD_OPERATION_NAME}` padding value must be a scalar but has type `{padding_value}`"
        ))
        .into());
    }

    if input.memory() != padding_value.memory() {
        return Err(TypeError::invalid(format!(
            "`{}` input and padding value must share one memory space but reside in `{}` and `{}`",
            PAD_OPERATION_NAME,
            input.memory(),
            padding_value.memory(),
        ))
        .into());
    }

    for (name, length) in [
        ("edge_padding_low", edge_padding_low.len()),
        ("edge_padding_high", edge_padding_high.len()),
        ("interior_padding", interior_padding.len()),
    ] {
        if length != input.rank() {
            return Err(TypeError::invalid(format!(
                "`{}` `{}` has length {} but input has rank {}",
                PAD_OPERATION_NAME,
                name,
                length,
                input.rank(),
            ))
            .into());
        }
    }

    Ok(())
}

/// Computes the output size of one padded axis as `input_size + max(input_size - 1, 0) * interior_padding +
/// edge_padding_low + edge_padding_high`. Interior padding contributes only between adjacent input elements, so it has
/// no effect on empty or singleton axes. Negative edge padding crops the axis after interior padding has been inserted.
///
/// The calculation uses checked [`i128`] arithmetic and returns an error if an intermediate value cannot be
/// represented. A negative result or a result larger than [`usize::MAX`] is otherwise returned unchanged: callers
/// that analyze bounds need the signed result even when it is not a valid array size. Use [`static_padded_extent`]
/// when the result must be a valid concrete extent. For example, an input size of `3` with interior padding `1` and
/// edge padding `1` on both sides produces `7` while an input size of `2` with low padding `-3` and no other padding
/// produces `-1` here rather than an error.
///
/// # Parameters
///
///   - `input_size`: Number of input elements along the axis, before any padding or cropping.
///   - `edge_padding_low`: Signed padding amount at the beginning of the axis; negative amounts crop that edge.
///   - `edge_padding_high`: Signed padding amount at the end of the axis; negative amounts crop that edge.
///   - `interior_padding`: Number of padding elements inserted between adjacent input elements along the axis.
///   - `axis`: Axis index included in error messages; it does not affect the calculation.
fn padded_extent(
    input_size: usize,
    edge_padding_low: i64,
    edge_padding_high: i64,
    interior_padding: usize,
    axis: usize,
) -> Result<i128, TypeError> {
    let gap_count = input_size.saturating_sub(1);
    let input_size = i128::try_from(input_size)
        .map_err(|_| TypeError::invalid(format!("`{PAD_OPERATION_NAME}` input size is too large on axis {axis}")))?;
    let gap_count = i128::try_from(gap_count)
        .map_err(|_| TypeError::invalid(format!("`{PAD_OPERATION_NAME}` input size is too large on axis {axis}")))?;
    let interior_padding = i128::try_from(interior_padding).map_err(|_| {
        TypeError::invalid(format!("`{PAD_OPERATION_NAME}` interior padding is too large on axis {axis}"))
    })?;
    let dilated_size = input_size
        .checked_add(gap_count.checked_mul(interior_padding).ok_or_else(|| {
            TypeError::invalid(format!("`{PAD_OPERATION_NAME}` output size overflows `usize` on axis {axis}"))
        })?)
        .and_then(|size| size.checked_add(i128::from(edge_padding_low)))
        .and_then(|size| size.checked_add(i128::from(edge_padding_high)))
        .ok_or_else(|| {
            TypeError::invalid(format!("`{PAD_OPERATION_NAME}` output size overflows `usize` on axis {axis}"))
        })?;
    Ok(dilated_size)
}

/// Computes the output size of one padded axis using [`padded_extent`] and requires the result to fit in [`usize`],
/// making it suitable for [`Dimension::Static`] or comparison with an explicit runtime extent.
///
/// Unlike [`padded_extent`], this function rejects a negative result or a result larger than [`usize::MAX`]. It does
/// not clamp excessive cropping to zero (e.g., an input size of `2` with low padding `-3` and no other padding is an
/// error, whereas low padding `-2` produces a valid empty axis). Arithmetic errors from [`padded_extent`] are
/// propagated. The parameters have the same meaning as in [`padded_extent`], including `axis`, which is used only
/// in diagnostics.
fn static_padded_extent(
    input_size: usize,
    edge_padding_low: i64,
    edge_padding_high: i64,
    interior_padding: usize,
    axis: usize,
) -> Result<usize, TypeError> {
    let output_size = padded_extent(input_size, edge_padding_low, edge_padding_high, interior_padding, axis)?;
    if output_size < 0 {
        return Err(TypeError::invalid(format!(
            "`{PAD_OPERATION_NAME}` output size is negative ({output_size}) on axis {axis}",
        )));
    }
    usize::try_from(output_size)
        .map_err(|_| TypeError::invalid(format!("`{PAD_OPERATION_NAME}` output size overflows `usize` on axis {axis}")))
}

/// Builds the output [`ArrayType`] for padding from dimensions already inferred or supplied by the caller. This
/// function does not compute those dimensions or verify the padding extent equation. Callers must first validate the
/// input and scalar padding-value types with [`validate_pad_inputs`] and provide one output dimension per input axis.
/// The padding vectors must likewise contain one entry per input axis.
///
/// # Parameters
///
///   - `input`: Validated input array type whose data type, memory space, and sharding determine the output metadata.
///   - `padding_value`: Validated scalar type, checked for compatible distributed dependencies when padding may occur.
///   - `output_dimensions`: Output dimensions to install unchanged; their relation to the padding amounts is the
///     caller's responsibility, including arranging runtime validation when it cannot be proved statically.
///   - `edge_padding_low`: Signed padding amounts at the beginning of each axis, used to detect possible padding.
///   - `edge_padding_high`: Signed padding amounts at the end of each axis, used to detect possible padding.
///   - `interior_padding`: Nonnegative padding counts between adjacent input elements, used together with input
///     dimension bounds to detect possible padding.
fn pad_output_type(
    input: &ArrayType,
    padding_value: &ArrayType,
    output_dimensions: Vec<Dimension>,
    edge_padding_low: &[i64],
    edge_padding_high: &[i64],
    interior_padding: &[usize],
) -> Result<ArrayType, ProgramError> {
    let padding_positions_may_exist = input.shape().dimensions().iter().enumerate().any(|(axis, dimension)| {
        edge_padding_low[axis] > 0
            || edge_padding_high[axis] > 0
            || (interior_padding[axis] > 0
                && !matches!(dimension, Dimension::Static(0 | 1))
                && !matches!(
                    dimension,
                    Dimension::Dynamic(variable)
                        if variable.bounds().upper().is_some_and(|upper| upper <= 2)
                ))
    });

    let sharding = resized_output_sharding(input, &output_dimensions, PAD_OPERATION_NAME)?;

    if padding_positions_may_exist {
        if input.unreduced_axes() != padding_value.unreduced_axes()
            || input.reduced_axes() != padding_value.reduced_axes()
        {
            return Err(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` input and padding value must have matching reduced and unreduced mesh axes \
                 but got input type `{input}` and padding value type `{padding_value}`",
            ))
            .into());
        }

        let input_varying_manual_axes = input.sharding().map(|sharding| sharding.varying_manual_axes());
        let padding_varying_manual_axes = padding_value.sharding().map(|sharding| sharding.varying_manual_axes());
        if input_varying_manual_axes.cloned().unwrap_or_default()
            != padding_varying_manual_axes.cloned().unwrap_or_default()
        {
            return Err(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` input and padding value must have matching varying manual axes but got input \
                 type `{input}` and padding value type `{padding_value}`",
            ))
            .into());
        }

        let has_distributed_dependencies = !input.unreduced_axes().is_empty()
            || !input.reduced_axes().is_empty()
            || input_varying_manual_axes.is_some_and(|axes| !axes.is_empty());
        if has_distributed_dependencies
            && input.sharding().map(|sharding| sharding.mesh())
                != padding_value.sharding().map(|sharding| sharding.mesh())
        {
            return Err(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` input and padding value with distributed dependencies must use the same mesh",
            ))
            .into());
        }
    }

    ArrayType::new(input.data_type(), Shape::new(output_dimensions))
        .with_memory(input.memory())
        .with_sharding(sharding)
        .map_err(|error| TypeError::invalid(format!("`{PAD_OPERATION_NAME}` output type is invalid: {error}")).into())
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation,
        DataType, DimensionBounds, DimensionError, DimensionType, DimensionValue, DimensionVariable, Layout,
        LogicalMesh, Memory, MeshAxis, MeshAxisType, RaggedAxis, Sharding, ShardingDimension, StridedLayout,
    };
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
    use crate::contexts::EagerContext;
    use crate::differentiation::{DifferentiableOperation, DifferentiationContext, TranspositionContext};
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::operations::constants::iota::IotaOperation;
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{EmptyRegionDriver, Program, ProgramBuilder, ProgramError, Typed};

    use super::*;

    /// Builds a rank-1 mixed pad of an `f32[size]` input with the provided padding amounts whose explicit output
    /// extent is the `output_size` program input, so the padded geometry is only known at runtime.
    fn dynamic_pad_program(
        input_bounds: DimensionBounds,
        output_bounds: DimensionBounds,
        edge_padding_low: i64,
        edge_padding_high: i64,
        interior_padding: usize,
    ) -> Program<ArrayIrValue<Array>, ArrayIrOperation<Array>, Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>> {
        let size = DimensionVariable::new("size", input_bounds);
        let output_size = DimensionVariable::new("output_size", output_bounds);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F32, Shape::new(vec![size.into()])).into());
        let padding_value = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let output_extent = builder.add_input(DimensionType::new(output_size).into());
        let output = builder
            .add_instruction(
                PadOperation::<ArrayType>::new(vec![edge_padding_low], vec![edge_padding_high], vec![interior_padding])
                    .unwrap(),
                Vec::new(),
                vec![input, padding_value, output_extent],
                None,
            )
            .unwrap()[0];
        builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap()
    }

    #[test]
    fn test_pad() {
        let operation = PadOperation::<ArrayType>::new(vec![1], vec![2], vec![1]).unwrap();
        // Operation identity and accessors.
        assert_eq!(operation.name(), PAD_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "pad [edge_padding_low=[1], edge_padding_high=[2], interior_padding=[1]]");
        assert_eq!(operation.edge_padding_low(), &[1]);
        assert_eq!(operation.edge_padding_high(), &[2]);
        assert_eq!(operation.interior_padding(), &[1]);
        assert_eq!(operation.effects().classes(), EffectClasses::NONE);
        assert_eq!(operation, operation.clone());
        assert_ne!(operation, PadOperation::new(vec![1], vec![2], vec![0]).unwrap());
        assert_eq!(
            PadOperation::<ArrayType>::new(vec![1], vec![2, 0], vec![1]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` expects `edge_padding_low`, `edge_padding_high`, and `interior_padding` to \
                 share one length but got lengths 1, 2, and 1"
            )))),
        );

        // Program rendering uses the canonical operation name and includes all three padding vectors.
        let mut builder = ProgramBuilder::<Array, PadOperation<ArrayType>>::new();
        let program_input = builder.add_input(ArrayType::new_static(DataType::F64, [3]));
        let program_padding_value = builder.add_input(ArrayType::scalar(DataType::F64));
        let program_output = builder
            .add_instruction(operation, Vec::new(), vec![program_input, program_padding_value], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Array>(vec![program_output], vec![Placeholder, Placeholder], Placeholder)
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[] .
                let %2:f64[8] = pad [edge_padding_low=[1], edge_padding_high=[2], interior_padding=[1]] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_pad_type_inference() {
        let input_type = ArrayType::new_static(DataType::F64, [3]);
        let padding_value_type = ArrayType::scalar(DataType::F64);
        // Type inference validates the padding geometry and returns the padded type: interior padding dilates the
        // input before the edges are added, an empty axis holds only its edge padding, and dynamic axes whose extent
        // changes need the explicit result-extent input of the mixed operation.
        check_operation_type_inference!(
            operation = PadOperation::<ArrayType>::new(vec![1], vec![2], vec![1]).unwrap(),
            cases = [
                {
                    input_types = [input_type.clone(), padding_value_type.clone()],
                    output_types = [ArrayType::new_static(DataType::F64, [8])],
                },
                {
                    input_types = [ArrayType::new_static(DataType::F64, [0]), padding_value_type.clone()],
                    output_types = [ArrayType::new_static(DataType::F64, [3])],
                },
                {
                    input_types = [input_type.clone()],
                    error = "expected 2 inputs but got 1",
                },
                {
                    input_types = [input_type.clone(), ArrayType::scalar(DataType::F32)],
                    error = format!(
                        "`{PAD_OPERATION_NAME}` input data type `f64` does not match padding value data type `f32`",
                    ),
                },
                {
                    input_types = [input_type.clone(), input_type.clone()],
                    error = format!("`{PAD_OPERATION_NAME}` padding value must be a scalar but has type `f64[3]`"),
                },
                {
                    input_types = [
                        ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                            "input",
                            DimensionBounds::unbounded(),
                        ))])),
                        padding_value_type.clone(),
                    ],
                    error = format!(
                        "`{PAD_OPERATION_NAME}` dynamic axis 0 requires an explicit result-dimension input",
                    ),
                },
                {
                    input_types = [
                        ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                            "input",
                            DimensionBounds::non_negative(Some(4)).unwrap(),
                        ))])),
                        padding_value_type.clone(),
                    ],
                    error = format!(
                        "`{PAD_OPERATION_NAME}` dynamic axis 0 requires an explicit result-dimension input",
                    ),
                },
                {
                    input_types = [ArrayType::new_static(DataType::F64, [usize::MAX]), padding_value_type.clone()],
                    error = format!("`{PAD_OPERATION_NAME}` output size overflows `usize` on axis 0"),
                },
            ],
        );
        assert_eq!(
            PadOperation::<ArrayType>::new(vec![1], vec![2], vec![1]).unwrap().infer_output_types(
                &[input_type.clone(), padding_value_type.clone()],
                &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1"))
        );
        // The padding vectors must have one entry per input axis.
        check_operation_type_inference!(
            operation = PadOperation::<ArrayType>::new(vec![1, 0], vec![2, 0], vec![1, 0]).unwrap(),
            cases = [{
                input_types = [input_type.clone(), padding_value_type.clone()],
                error = format!("`{PAD_OPERATION_NAME}` `edge_padding_low` has length 2 but input has rank 1"),
            }],
        );

        // Unchanged dynamic extents keep their identity even when another axis changes or edge shifts balance.
        let size = DimensionVariable::new("size", DimensionBounds::new(1, Some(5)).unwrap());
        check_operation_type_inference!(
            operation = PadOperation::<ArrayType>::new(vec![0, 1], vec![0, 1], vec![0, 0]).unwrap(),
            cases = [{
                input_types = [
                    ArrayType::new(DataType::F32, Shape::new(vec![size.clone().into(), 2.into()])),
                    ArrayType::scalar(DataType::F32),
                ],
                output_types = [ArrayType::new(DataType::F32, Shape::new(vec![size.clone().into(), 4.into()]))],
            }],
        );
        check_operation_type_inference!(
            operation = PadOperation::<ArrayType>::new(vec![-1], vec![1], vec![0]).unwrap(),
            cases = [{
                input_types = [
                    ArrayType::new(DataType::F32, Shape::new(vec![size.clone().into()])),
                    ArrayType::scalar(DataType::F32),
                ],
                output_types = [ArrayType::new(DataType::F32, Shape::new(vec![size.into()]))],
            }],
        );
        // Negative edges still validate their abstract extent on dynamic axes. A valid derived dynamic extent
        // requires the explicit result-dimension input introduced by the mixed operation signature, while an
        // always-negative extent is rejected immediately.
        check_operation_type_inference!(
            operation = PadOperation::<ArrayType>::new(vec![-1], vec![-2], vec![0]).unwrap(),
            cases = [
                {
                    input_types = [
                        ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                            "input",
                            DimensionBounds::non_negative(Some(9)).unwrap(),
                        ))])),
                        padding_value_type.clone(),
                    ],
                    error = format!(
                        "`{PAD_OPERATION_NAME}` dynamic axis 0 requires an explicit result-dimension input",
                    ),
                },
                {
                    input_types = [
                        ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                            "input",
                            DimensionBounds::unbounded(),
                        ))])),
                        padding_value_type.clone(),
                    ],
                    error = format!(
                        "`{PAD_OPERATION_NAME}` dynamic axis 0 requires an explicit result-dimension input",
                    ),
                },
            ],
        );
        check_operation_type_inference!(
            operation = PadOperation::<ArrayType>::new(vec![-5], vec![0], vec![0]).unwrap(),
            cases = [{
                input_types = [
                    ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                        "dynamic",
                        DimensionBounds::non_negative(Some(2)).unwrap(),
                    ))])),
                    padding_value_type,
                ],
                error = format!(
                    "`{PAD_OPERATION_NAME}` output size is negative (-4) on dynamic axis 0 even at its maximum input \
                     extent 1",
                ),
            }],
        );
    }

    #[test]
    fn test_pad_interpretation() {
        let context = EagerContext::<Array>::new();
        let padding_value = Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[9.0]).unwrap();
        // Interpretation writes the input elements at `low + i * (interior + 1)` (positions 1, 3, and 5) and fills
        // every other position with the padding value.
        let input = Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[1.0, 2.0, 3.0]).unwrap();
        let operation = PadOperation::<ArrayType>::new(vec![1], vec![2], vec![1]).unwrap();
        assert_eq!(
            operation.interpret(&context, &EmptyRegionDriver, &[input.clone(), padding_value.clone()]),
            Ok(vec![
                Array::from_elements::<f64>(
                    ArrayType::new_static(DataType::F64, [8]),
                    &[9.0, 1.0, 9.0, 2.0, 9.0, 3.0, 9.0, 9.0],
                )
                .unwrap()
            ]),
        );
        // Negative edge padding crops after the (empty) interior padding has been inserted.
        assert_eq!(
            PadOperation::new(vec![-1], vec![1], vec![0]).unwrap().interpret(
                &context,
                &EmptyRegionDriver,
                &[input, padding_value.clone()],
            ),
            Ok(vec![Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[2.0, 3.0, 9.0]).unwrap()]),
        );
        // Empty input axes hold only the edge padding (the `d == 0` case skips interior padding entirely) and rank-0
        // inputs pass through unchanged.
        assert_eq!(
            operation.interpret(
                &context,
                &EmptyRegionDriver,
                &[Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [0]), &[]).unwrap(), padding_value],
            ),
            Ok(vec![Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[9.0, 9.0, 9.0]).unwrap()]),
        );
        let scalar = Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[42.0]).unwrap();
        assert_eq!(
            PadOperation::new(Vec::new(), Vec::new(), Vec::new()).unwrap().interpret(
                &context,
                &EmptyRegionDriver,
                &[scalar.clone(), Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[7.0]).unwrap()],
            ),
            Ok(vec![scalar]),
        );
        assert_eq!(
            operation.interpret(&context, &EmptyRegionDriver, &[]),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );
    }

    #[test]
    fn test_pad_partial_evaluation() {
        // Check standard partial evaluation with known and residual inputs.
        let input = Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[1.0, 2.0, 3.0]).unwrap();
        let padding_value = Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[9.0]).unwrap();
        let expected = Array::from_elements::<f64>(
            ArrayType::new_static(DataType::F64, [8]),
            &[9.0, 1.0, 9.0, 2.0, 9.0, 3.0, 9.0, 9.0],
        )
        .unwrap();
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
            cases = [
                {
                    inputs = [(@known, input.clone()), (@known, padding_value.clone())],
                    outputs = [(@known, expected.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = input.r#type().into_owned(), replay = input.clone())),
                        (@known, padding_value.clone()),
                    ],
                    outputs = [(@residual, expected.clone())],
                    residual_instructions = 1,
                },
                {
                    inputs = [
                        (@known, input.clone()),
                        (@unknown(type = padding_value.r#type().into_owned(), replay = padding_value.clone())),
                    ],
                    outputs = [(@residual, expected.clone())],
                    residual_instructions = 1,
                },
                {
                    inputs = [
                        (@unknown(type = input.r#type().into_owned(), replay = input.clone())),
                        (@unknown(type = padding_value.r#type().into_owned(), replay = padding_value.clone())),
                    ],
                    outputs = [(@residual, expected.clone())],
                    residual_instructions = 1,
                },
            ],
        );
    }

    #[test]
    fn test_pad_batching() {
        // Batching inserts zero padding on the mapped axis and vectorizes a mapped padding value.
        check_operation_batching!(
            @exact,
            operation = PadOperation::new(vec![1], vec![0], vec![0]).unwrap(),
            axis_size = 2,
            cases = [
                {
                    inputs = [
                        (@mapped(axis = 0), Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [2, 2]),
                            &[1.0, 2.0, 3.0, 4.0],
                        ).unwrap()),
                        (@replicated, Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[0.0]).unwrap()),
                    ],
                    outputs = [
                        (@mapped(axis = 0), Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [2, 3]),
                            &[0.0, 1.0, 2.0, 0.0, 3.0, 4.0],
                        ).unwrap()),
                    ],
                },
                {
                    inputs = [
                        (@replicated, Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [2]),
                            &[1.0, 2.0],
                        ).unwrap()),
                        (@replicated, Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[0.0]).unwrap()),
                    ],
                    outputs = [
                        (@replicated, Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [3]),
                            &[0.0, 1.0, 2.0],
                        ).unwrap()),
                    ],
                },
                {
                    inputs = [
                        (@mapped(axis = 0), Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [2, 2]),
                            &[1.0, 2.0, 3.0, 4.0],
                        ).unwrap()),
                        (@mapped(axis = 0), Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [2]),
                            &[8.0, 9.0],
                        ).unwrap()),
                    ],
                    outputs = [
                        (@mapped(axis = 0), Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [2, 3]),
                            &[8.0, 1.0, 2.0, 9.0, 3.0, 4.0],
                        ).unwrap()),
                    ],
                },
                {
                    inputs = [
                        (@replicated, Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [2]),
                            &[1.0, 2.0],
                        ).unwrap()),
                        (@mapped(axis = 0), Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [2]),
                            &[8.0, 9.0],
                        ).unwrap()),
                    ],
                    outputs = [
                        (@mapped(axis = 0), Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [2, 3]),
                            &[8.0, 1.0, 2.0, 9.0, 1.0, 2.0],
                        ).unwrap()),
                    ],
                },
                {
                    inputs = [
                        (@mapped(axis = 1), Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [2, 2]),
                            &[1.0, 2.0, 3.0, 4.0],
                        ).unwrap()),
                        (@mapped(axis = 0), Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [2]),
                            &[8.0, 9.0],
                        ).unwrap()),
                    ],
                    outputs = [
                        (@mapped(axis = 1), Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [3, 2]),
                            &[8.0, 9.0, 1.0, 2.0, 3.0, 4.0],
                        ).unwrap()),
                    ],
                },
            ],
        );
    }

    #[test]
    fn test_pad_batching_sharding() {
        // The transform-owned mapped axis sharding rides along the batch axis, while the padded axis keeps its
        // replicated sharding, both for explicit mesh axes and for varying manual ones. `into_parts` splits the
        // batched-output carrier into the outputs and the ragged evidence, which padding never produces.
        let operation = PadOperation::new(vec![1], vec![0], vec![0]).unwrap();
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let explicit_sharding = Sharding::new(
            explicit_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let explicit_padding_sharding = Sharding::new(explicit_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let explicit_context =
            BatchingContext::new(EagerContext::<Array>::new(), 2).with_axis_sharding(ShardingDimension::sharded(["x"]));
        let input = ArrayBatch::new(
            Array::from_elements::<f64>(
                ArrayType::new_static(DataType::F64, [2, 2]).with_sharding(explicit_sharding.clone()).unwrap(),
                &[1.0, 2.0, 3.0, 4.0],
            )
            .unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let padding = ArrayBatch::new(
            Array::from_elements::<f64>(
                ArrayType::new_static(DataType::F64, [2]).with_sharding(explicit_padding_sharding.clone()).unwrap(),
                &[8.0, 9.0],
            )
            .unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let (outputs, _) =
            operation.batch(&explicit_context, &EmptyRegionDriver, &[input, padding]).unwrap().into_parts();
        assert_eq!(
            outputs,
            vec![
                ArrayBatch::new(
                    Array::from_elements::<f64>(
                        ArrayType::new_static(DataType::F64, [2, 3]).with_sharding(explicit_sharding.clone()).unwrap(),
                        &[8.0, 1.0, 2.0, 9.0, 3.0, 4.0],
                    )
                    .unwrap(),
                    BatchAxis::new(0),
                )
                .unwrap()
            ],
        );

        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let manual_sharding = Sharding::new(
            manual_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap()
        .with_varying_manual_axes(["x"])
        .unwrap();
        let manual_padding_sharding = Sharding::new(manual_mesh, vec![ShardingDimension::sharded(["x"])])
            .unwrap()
            .with_varying_manual_axes(["x"])
            .unwrap();
        let manual_context =
            BatchingContext::new(EagerContext::<Array>::new(), 2).with_axis_sharding(ShardingDimension::sharded(["x"]));
        let input = ArrayBatch::new(
            Array::from_elements::<f64>(
                ArrayType::new_static(DataType::F64, [2, 2]).with_sharding(manual_sharding.clone()).unwrap(),
                &[1.0, 2.0, 3.0, 4.0],
            )
            .unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let padding = ArrayBatch::new(
            Array::from_elements::<f64>(
                ArrayType::new_static(DataType::F64, [2]).with_sharding(manual_padding_sharding.clone()).unwrap(),
                &[8.0, 9.0],
            )
            .unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let (outputs, _) =
            operation.batch(&manual_context, &EmptyRegionDriver, &[input, padding]).unwrap().into_parts();
        assert_eq!(
            outputs,
            vec![
                ArrayBatch::new(
                    Array::from_elements::<f64>(
                        ArrayType::new_static(DataType::F64, [2, 3]).with_sharding(manual_sharding.clone()).unwrap(),
                        &[8.0, 1.0, 2.0, 9.0, 3.0, 4.0],
                    )
                    .unwrap(),
                    BatchAxis::new(0),
                )
                .unwrap()
            ],
        );

        // The vectorized mapped-padding rule handles an empty batch without inventing values or dropping placement.
        let explicit_context =
            BatchingContext::new(EagerContext::<Array>::new(), 0).with_axis_sharding(ShardingDimension::sharded(["x"]));
        let input = ArrayBatch::new(
            Array::from_elements::<f64>(
                ArrayType::new_static(DataType::F64, [0, 2]).with_sharding(explicit_sharding.clone()).unwrap(),
                &[],
            )
            .unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let padding = ArrayBatch::new(
            Array::from_elements::<f64>(
                ArrayType::new_static(DataType::F64, [0]).with_sharding(explicit_padding_sharding).unwrap(),
                &[],
            )
            .unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let (outputs, _) =
            operation.batch(&explicit_context, &EmptyRegionDriver, &[input, padding]).unwrap().into_parts();
        assert_eq!(
            outputs,
            vec![
                ArrayBatch::new(
                    Array::from_elements::<f64>(
                        ArrayType::new_static(DataType::F64, [0, 3]).with_sharding(explicit_sharding).unwrap(),
                        &[],
                    )
                    .unwrap(),
                    BatchAxis::new(0),
                )
                .unwrap()
            ],
        );
        let manual_context =
            BatchingContext::new(EagerContext::<Array>::new(), 0).with_axis_sharding(ShardingDimension::sharded(["x"]));
        let input = ArrayBatch::new(
            Array::from_elements::<f64>(
                ArrayType::new_static(DataType::F64, [0, 2]).with_sharding(manual_sharding.clone()).unwrap(),
                &[],
            )
            .unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let padding = ArrayBatch::new(
            Array::from_elements::<f64>(
                ArrayType::new_static(DataType::F64, [0]).with_sharding(manual_padding_sharding).unwrap(),
                &[],
            )
            .unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let (outputs, _) =
            operation.batch(&manual_context, &EmptyRegionDriver, &[input, padding]).unwrap().into_parts();
        assert_eq!(
            outputs,
            vec![
                ArrayBatch::new(
                    Array::from_elements::<f64>(
                        ArrayType::new_static(DataType::F64, [0, 3]).with_sharding(manual_sharding).unwrap(),
                        &[],
                    )
                    .unwrap(),
                    BatchAxis::new(0),
                )
                .unwrap()
            ],
        );
    }

    #[test]
    fn test_pad_batching_preserves_ragged_dimensions() {
        // Ragged geometry survives padding of an unrelated axis, while padding the ragged axis or an axis indexing its
        // extents is rejected before shape inference.
        let size = DimensionVariable::new("size", DimensionBounds::new(0, Some(4)).unwrap());
        let extents = Array::from_elements(ArrayType::new_static(DataType::I64, [2]), &[1_i64, 3]).unwrap();
        let ragged = RaggedAxis::new(1, extents, size, vec![0]);
        let input = ArrayBatch::new(
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3, 1]), &[1_f32, 0., 0., 2., 3., 4.])
                .unwrap(),
            BatchAxis::new(0),
        )
        .unwrap()
        .with_ragged_axes(vec![ragged.clone()])
        .unwrap();
        let padding = ArrayBatch::replicated(Array::from_elements(ArrayType::scalar(DataType::F32), &[9_f32]).unwrap());
        let context =
            BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);
        let batched = PadOperation::new(vec![0, 1], vec![0, 1], vec![0, 0])
            .unwrap()
            .batch(&context, &EmptyRegionDriver, &[input.clone(), padding.clone()])
            .unwrap();
        let (outputs, _) = batched.into_parts();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].ragged_axes(), &[ragged]);
        assert_eq!(
            outputs[0].value().elements::<f32>(),
            Ok(vec![9., 1., 9., 9., 0., 9., 9., 0., 9., 9., 2., 9., 9., 3., 9., 9., 4., 9.])
        );
        assert!(matches!(
            PadOperation::new(vec![1, 0], vec![0, 0], vec![0, 0])
                .unwrap()
                .batch(&context, &EmptyRegionDriver, &[input, padding]),
            Err(BatchingError::Program(ProgramError::UnsupportedOperation { message }))
                if message == format!(
                    "`{PAD_OPERATION_NAME}` batching cannot change a ragged axis or an axis indexing its extents",
                ),
        ));
    }

    #[test]
    fn test_pad_batching_mapped_padding_without_numeric_zero() {
        // The mapped padding-value decomposition pads with `one` as its placeholder, so element types without a
        // representable zero still vectorize and the selected bytes are copied verbatim.
        let input = ArrayBatch::new(
            Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [2, 1]), vec![127, 128]).unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let padding = ArrayBatch::new(
            Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [2]), vec![129, 130]).unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let context =
            BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);
        let batched = PadOperation::new(vec![1], vec![0], vec![0])
            .unwrap()
            .batch(&context, &EmptyRegionDriver, &[input, padding])
            .unwrap();
        let (outputs, _) = batched.into_parts();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().storage_bytes(), &[129, 127, 130, 128]);
    }

    #[test]
    fn test_pad_differentiation() {
        // Pad is linear in both inputs: its JVP pads tangent values and its pullback separates written and padding
        // positions.
        check_operation_differentiation!(
            @approx(step = 0.125, epsilon = 1e-9),
            operation = PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
            cases = [{
                primals = [
                    Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[1.0, 2.0, 3.0]).unwrap(),
                    Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[9.0]).unwrap(),
                ],
                tangents = [
                    Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[0.1, 0.2, 0.3]).unwrap(),
                    Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[0.5]).unwrap(),
                ],
                primal_outputs = [
                    Array::from_elements::<f64>(
                        ArrayType::new_static(DataType::F64, [8]),
                        &[9.0, 1.0, 9.0, 2.0, 9.0, 3.0, 9.0, 9.0],
                    ).unwrap(),
                ],
                tangent_outputs = [
                    Array::from_elements::<f64>(
                        ArrayType::new_static(DataType::F64, [8]),
                        &[0.5, 0.1, 0.5, 0.2, 0.5, 0.3, 0.5, 0.5],
                    ).unwrap(),
                ],
            }],
        );
    }

    #[test]
    fn test_pad_differentiation_structural_zero_tangent() {
        // The shared all-zero fast path lives in the differentiation context's bind, so a direct rule call reaches
        // the body with structural-zero tangents. The pad needs both tangents as real values, so each zero side is
        // materialized and padded like any other tangent.
        let operation = PadOperation::<ArrayType>::new(vec![1], vec![2], vec![1]).unwrap();
        let context = DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new());
        let input = Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[1.0, 2.0, 3.0]).unwrap();
        let padding_value = Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[9.0]).unwrap();
        let primal_output = Array::from_elements::<f64>(
            ArrayType::new_static(DataType::F64, [8]),
            &[9.0, 1.0, 9.0, 2.0, 9.0, 3.0, 9.0, 9.0],
        )
        .unwrap();
        let outputs = operation
            .jvp(
                &context,
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new(input.clone(), MaybeZero::Zero(ArrayType::new_static(DataType::F64, [3])))
                        .unwrap(),
                    DifferentiationDual::new(padding_value.clone(), MaybeZero::Zero(ArrayType::scalar(DataType::F64)))
                        .unwrap(),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(*outputs[0].primal(), primal_output);
        assert!(matches!(
            outputs[0].tangent(),
            MaybeZero::Value(tangent)
                if *tangent
                    == Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [8]), &[0.0; 8]).unwrap(),
        ));

        // A live input tangent is padded with the materialized zero padding-value tangent.
        let outputs = operation
            .jvp(
                &context,
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new(
                        input,
                        MaybeZero::Value(
                            Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[0.1, 0.2, 0.3])
                                .unwrap(),
                        ),
                    )
                    .unwrap(),
                    DifferentiationDual::new(padding_value, MaybeZero::Zero(ArrayType::scalar(DataType::F64))).unwrap(),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(*outputs[0].primal(), primal_output);
        assert!(matches!(
            outputs[0].tangent(),
            MaybeZero::Value(tangent)
                if *tangent == Array::from_elements::<f64>(
                    ArrayType::new_static(DataType::F64, [8]),
                    &[0.0, 0.1, 0.0, 0.2, 0.0, 0.3, 0.0, 0.0],
                )
                .unwrap(),
        ));
    }

    #[test]
    fn test_pad_transposition() {
        check_operation_transposition!(
            @exact,
            operation = PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
            cases = [
                {
                    inputs = [
                        (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![3.into()])))),
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                    ],
                    output_cotangents = [
                        Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [8]),
                            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                        ).unwrap(),
                    ],
                    input_cotangents = [
                        Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [3]),
                            &[2.0, 4.0, 6.0],
                        ).unwrap(),
                        Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[24.0]).unwrap(),
                    ],
                },
                {
                    inputs = [
                        (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![3.into()])))),
                        (@known, Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[9.0]).unwrap()),
                    ],
                    output_cotangents = [
                        Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [8]),
                            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                        ).unwrap(),
                    ],
                    input_cotangents = [
                        Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [3]),
                            &[2.0, 4.0, 6.0],
                        ).unwrap(),
                    ],
                },
                {
                    inputs = [
                        (@known, Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [3]),
                            &[1.0, 2.0, 3.0],
                        ).unwrap()),
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                    ],
                    output_cotangents = [
                        Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [8]),
                            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                        ).unwrap(),
                    ],
                    input_cotangents = [Array::from_elements::<f64>(
                        ArrayType::scalar(DataType::F64),
                        &[24.0],
                    ).unwrap()],
                },
            ],
        );
        // Selecting padding positions before reduction avoids the `infinity - infinity` contamination that a
        // total-sum-minus-input-sum formulation would introduce.
        check_operation_transposition!(
            @exact,
            operation = PadOperation::new(vec![1], vec![0], vec![0]).unwrap(),
            cases = [
                {
                    inputs = [
                        (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![2.into()])))),
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                    ],
                    output_cotangents = [
                        Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [3]),
                            &[5.0, f64::INFINITY, 7.0],
                        ).unwrap(),
                    ],
                    input_cotangents = [
                        Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [2]),
                            &[f64::INFINITY, 7.0],
                        ).unwrap(),
                        Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[5.0]).unwrap(),
                    ],
                },
                {
                    inputs = [
                        (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![2.into()])))),
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                    ],
                    output_cotangents = [
                        Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [3]),
                            &[3.0, 1e20, -1e20],
                        ).unwrap(),
                    ],
                    input_cotangents = [
                        Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2]), &[1e20, -1e20]).unwrap(),
                        Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[3.0]).unwrap(),
                    ],
                },
            ],
        );
        check_operation_transposition!(
            @exact,
            operation = PadOperation::new(vec![-1], vec![1], vec![0]).unwrap(),
            cases = [{
                inputs = [
                    (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![3.into()])))),
                    (@linear(type = ArrayType::scalar(DataType::F64))),
                ],
                output_cotangents = [
                    Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[2.0, 3.0, 5.0]).unwrap(),
                ],
                input_cotangents = [
                    Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[0.0, 2.0, 3.0]).unwrap(),
                    Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[5.0]).unwrap(),
                ],
            }],
        );
        check_operation_transposition!(
            @exact,
            operation = PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
            cases = [{
                inputs = [
                    (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![0.into()])))),
                    (@linear(type = ArrayType::scalar(DataType::F64))),
                ],
                output_cotangents = [
                    Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[1.0, 2.0, 3.0]).unwrap(),
                ],
                input_cotangents = [
                    Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [0]), &[]).unwrap(),
                    Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[6.0]).unwrap(),
                ],
            }],
        );
        check_operation_transposition!(
            @exact,
            operation = PadOperation::new(Vec::new(), Vec::new(), Vec::new()).unwrap(),
            cases = [{
                inputs = [
                    (@linear(type = ArrayType::scalar(DataType::F64))),
                    (@linear(type = ArrayType::scalar(DataType::F64))),
                ],
                output_cotangents = [
                    Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[f64::INFINITY]).unwrap(),
                ],
                input_cotangents = [
                    Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[f64::INFINITY]).unwrap(),
                    Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[0.0]).unwrap(),
                ],
            }],
        );

        // A pure crop never reads the padding scalar, so its dependency metadata may differ from the input's. The
        // inverse pad nevertheless introduces zeros for cropped input positions and must derive that internal zero's
        // dependencies from the input cotangent rather than from the unused primal padding scalar.
        let crop_mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let crop_input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]))
            .with_sharding(Sharding::replicated(crop_mesh.clone(), 1).with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        let crop_padding_type =
            ArrayType::scalar(DataType::F64).with_sharding(Sharding::replicated(crop_mesh, 0)).unwrap();
        let crop_output_type = crop_input_type.pad(&crop_padding_type, &[-1], &[0], &[0]).unwrap();
        check_operation_transposition!(
            @exact,
            operation = PadOperation::new(vec![-1], vec![0], vec![0]).unwrap(),
            cases = [{
                inputs = [
                    (@linear(type = crop_input_type.clone())),
                    (@known, Array::from_elements::<f64>(crop_padding_type, &[9.0]).unwrap()),
                ],
                output_cotangents = [Array::from_elements::<f64>(crop_output_type, &[2.0, 3.0]).unwrap()],
                input_cotangents = [Array::from_elements::<f64>(crop_input_type, &[0.0, 2.0, 3.0]).unwrap()],
            }],
        );

        // The pullback restores the complete cotangent types of both inputs after slicing and reducing the output
        // cotangent.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![3.into()]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
            .with_memory(Memory::Host { pinned: true });
        let padding_type = ArrayType::scalar(DataType::F64)
            .with_layout(Layout::Strided(StridedLayout::new(Vec::new())))
            .with_memory(Memory::Host { pinned: true });
        let output_type =
            ArrayType::new(DataType::F64, Shape::new(vec![8.into()])).with_memory(Memory::Host { pinned: true });
        check_operation_transposition!(
            @exact,
            operation = PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
            cases = [{
                inputs = [(@linear(type = input_type.clone())), (@linear(type = padding_type.clone()))],
                output_cotangents = [Array::from_elements::<f64>(
                    output_type,
                    &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                ).unwrap()],
                input_cotangents = [
                    Array::from_elements::<f64>(input_type, &[2.0, 4.0, 6.0]).unwrap(),
                    Array::from_elements::<f64>(padding_type, &[24.0]).unwrap(),
                ],
            }],
        );
    }

    #[test]
    fn test_pad_transposition_symbolic_zero() {
        // A structural-zero output cotangent contributes nothing: the rule returns before staging anything and leaves
        // both accumulators at their structural-zero defaults.
        let input_types = vec![ArrayType::new_static(DataType::F64, [3]), ArrayType::scalar(DataType::F64)];
        let operation = PadOperation::<ArrayType>::new(vec![1], vec![2], vec![1]).unwrap();
        let output_type = operation.infer_output_types(&input_types, &[]).unwrap().remove(0);
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let mut transpose = TranspositionContext::new(context.clone());
        let inputs = input_types.iter().cloned().map(PartialValue::Unknown).collect::<Vec<_>>();
        let accumulators = transpose.cotangent_accumulators(&inputs, &[]).unwrap();
        operation
            .transpose(
                &mut transpose,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Zero(output_type.cotangent().unwrap())],
                &accumulators,
            )
            .unwrap();
        let cotangents = transpose.take_cotangents(&accumulators).unwrap();
        assert_eq!(cotangents.len(), 2);
        for (cotangent, input_type) in cotangents.iter().zip(&input_types) {
            assert!(cotangent.is_zero());
            assert_eq!(cotangent.r#type().as_ref(), &input_type.cotangent().unwrap());
        }
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_pad_transposition_extreme_configuration() {
        // Interior padding is irrelevant when an axis contains fewer than two input elements.
        check_operation_transposition!(
            @exact,
            operation = PadOperation::new(vec![0], vec![0], vec![usize::MAX]).unwrap(),
            cases = [
                {
                    inputs = [(@linear(type = ArrayType::new_static(DataType::F32, [0]))),
                        (@linear(type = ArrayType::scalar(DataType::F32)))],
                    output_cotangents = [
                        Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0]), &[]).unwrap(),
                    ],
                    input_cotangents = [
                        Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0]), &[]).unwrap(),
                        Array::from_elements(ArrayType::scalar(DataType::F32), &[0_f32]).unwrap(),
                    ],
                },
                {
                    inputs = [(@linear(type = ArrayType::new_static(DataType::F32, [1]))),
                        (@linear(type = ArrayType::scalar(DataType::F32)))],
                    output_cotangents = [
                        Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[7_f32]).unwrap(),
                    ],
                    input_cotangents = [
                        Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[7_f32]).unwrap(),
                        Array::from_elements(ArrayType::scalar(DataType::F32), &[0_f32]).unwrap(),
                    ],
                },
            ],
        );
        // Wide coordinate arithmetic identifies the sole surviving input without allocating its enormous dilation.
        check_operation_transposition!(
            @exact,
            operation = PadOperation::new(vec![i64::MIN], vec![0], vec![i64::MAX as usize]).unwrap(),
            cases = [{
                inputs = [(@linear(type = ArrayType::new_static(DataType::F32, [2]))),
                    (@linear(type = ArrayType::scalar(DataType::F32)))],
                output_cotangents = [Array::from_elements(
                    ArrayType::new_static(DataType::F32, [1]),
                    &[7_f32],
                ).unwrap()],
                input_cotangents = [
                    Array::from_elements(ArrayType::new_static(DataType::F32, [2]), &[0_f32, 7.]).unwrap(),
                    Array::from_elements(ArrayType::scalar(DataType::F32), &[0_f32]).unwrap(),
                ],
            }],
        );
        // These edges crop every input position, despite their balanced finite output shape.
        check_operation_transposition!(
            @exact,
            operation = PadOperation::new(vec![i64::MIN], vec![i64::MAX], vec![0]).unwrap(),
            cases = [
                {
                    inputs = [(@linear(type = ArrayType::new_static(DataType::F32, [1]))),
                        (@linear(type = ArrayType::scalar(DataType::F32)))],
                    output_cotangents = [
                        Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0]), &[]).unwrap(),
                    ],
                    input_cotangents = [
                        Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[0_f32]).unwrap(),
                        Array::from_elements(ArrayType::scalar(DataType::F32), &[0_f32]).unwrap(),
                    ],
                },
                {
                    inputs = [(@linear(type = ArrayType::new_static(DataType::F32, [2]))),
                        (@linear(type = ArrayType::scalar(DataType::F32)))],
                    output_cotangents = [
                        Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[7_f32]).unwrap(),
                    ],
                    input_cotangents = [
                        Array::from_elements(ArrayType::new_static(DataType::F32, [2]), &[0_f32, 0.]).unwrap(),
                        Array::from_elements(ArrayType::scalar(DataType::F32), &[7_f32]).unwrap(),
                    ],
                },
            ],
        );
        check_operation_transposition!(
            @exact,
            operation = PadOperation::new(vec![i64::MAX], vec![i64::MIN], vec![0]).unwrap(),
            cases = [
                {
                    inputs = [(@linear(type = ArrayType::new_static(DataType::F32, [1]))),
                        (@linear(type = ArrayType::scalar(DataType::F32)))],
                    output_cotangents = [
                        Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0]), &[]).unwrap(),
                    ],
                    input_cotangents = [
                        Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[0_f32]).unwrap(),
                        Array::from_elements(ArrayType::scalar(DataType::F32), &[0_f32]).unwrap(),
                    ],
                },
                {
                    inputs = [(@linear(type = ArrayType::new_static(DataType::F32, [2]))),
                        (@linear(type = ArrayType::scalar(DataType::F32)))],
                    output_cotangents = [
                        Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[7_f32]).unwrap(),
                    ],
                    input_cotangents = [
                        Array::from_elements(ArrayType::new_static(DataType::F32, [2]), &[0_f32, 0.]).unwrap(),
                        Array::from_elements(ArrayType::scalar(DataType::F32), &[7_f32]).unwrap(),
                    ],
                },
            ],
        );
    }

    #[test]
    fn test_pad_transposition_rejects_unrepresentable_geometry() {
        // The homogeneous pullback slices static geometry only. A dynamic axis whose extent the padding leaves
        // unchanged passes type inference but has no static input extent to slice.
        let size = DimensionVariable::new("size", DimensionBounds::new(1, Some(5)).unwrap());
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![size.into()])));
        let padding_value = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(
                PadOperation::new(vec![-1], vec![1], vec![0]).unwrap(),
                Vec::new(),
                vec![input, padding_value],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            program.transpose_with_respect_to(&[0], &[]),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == format!("`{PAD_OPERATION_NAME}` transpose requires a static input extent on axis 0"),
        ));

        // Surviving input positions are computed in `i128`, but the inverse edge padding must fit `i64`: cropping
        // with `i64::MIN` keeps a surviving range that starts, or leaves a trailing crop that ends, beyond `i64::MAX`.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F32, [i64::MAX as usize + 2]));
        let padding_value = builder.add_input(ArrayType::scalar(DataType::F32));
        let output = builder
            .add_instruction(
                PadOperation::new(vec![i64::MIN], vec![0], vec![0]).unwrap(),
                Vec::new(),
                vec![input, padding_value],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(program.output_types(), vec![ArrayType::new_static(DataType::F32, [1])]);
        assert!(matches!(
            program.transpose_with_respect_to(&[0], &[]),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == format!("`{PAD_OPERATION_NAME}` transpose low padding exceeds `i64` on axis 0"),
        ));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F32, [i64::MAX as usize + 3]));
        let padding_value = builder.add_input(ArrayType::scalar(DataType::F32));
        let output = builder
            .add_instruction(
                PadOperation::new(vec![0], vec![i64::MIN], vec![0]).unwrap(),
                Vec::new(),
                vec![input, padding_value],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(program.output_types(), vec![ArrayType::new_static(DataType::F32, [2])]);
        assert!(matches!(
            program.transpose_with_respect_to(&[0], &[]),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == format!("`{PAD_OPERATION_NAME}` transpose high padding exceeds `i64` on axis 0"),
        ));
    }

    #[test]
    fn test_pad_pad() {
        // Context-carrying values pad through the blanket implementation, which validates the inputs and then binds
        // one homogeneous operation through the input's context. An effective identity is returned unchanged without
        // staging anything, but only after the same validation.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = context.input(ArrayType::new_static(DataType::F32, [3]));
        let padding_value = context.input(ArrayType::scalar(DataType::F32));
        assert_eq!(input.pad(&padding_value, &[0], &[0], &[0]), Ok(input.clone()));
        assert_eq!(
            input.pad(&input, &[0], &[0], &[0]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` padding value must be a scalar but has type `f32[3]`"
            )))),
        );
        assert_eq!(
            input.pad(&padding_value, &[], &[], &[]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` `edge_padding_low` has length 0 but input has rank 1"
            )))),
        );
        assert!(context.builder().borrow().instructions().is_empty());
        let output = input.pad(&padding_value, &[1], &[2], &[1]).unwrap();
        assert_eq!(output.r#type().as_ref(), &ArrayType::new_static(DataType::F32, [8]));
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![output.atom_id().unwrap()], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3], %1:f32[] .
                let %2:f32[8] = pad [edge_padding_low=[1], edge_padding_high=[2], interior_padding=[1]] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_pad_pad_with_config() {
        let input = Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[1_i32, 2]).unwrap();
        let padding = Array::from_elements(ArrayType::scalar(DataType::I32), &[9_i32]).unwrap();
        assert_eq!(
            input.pad_with_config(&padding, &[(1, 1, 1)]),
            Array::from_elements(ArrayType::new_static(DataType::I32, [5]), &[9_i32, 1, 9, 2, 9])
        );
        assert_eq!(
            input.pad_with_config(&padding, &[(-1, 0, 0)]),
            Array::from_elements(ArrayType::new_static(DataType::I32, [1]), &[2_i32])
        );
        assert_eq!(
            input.pad_with_config(&padding, &[]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` `edge_padding_low` has length 0 but input has rank 1"
            ))))
        );
    }

    #[test]
    fn test_array_type_pad() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        // [4] sharded over `x` and unreduced over the manual axis `m`.
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
            .unwrap()
            .with_unreduced_axes(["m"])
            .unwrap();
        let input = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let padding_value = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh, 0).with_unreduced_axes(["m"]).unwrap())
            .unwrap();

        // The abstract capability computes the padded type without consuming the borrowed input type, and each of
        // the three configuration slices must have one entry per input axis.
        let plain_input = ArrayType::new_static(DataType::F64, [3]);
        let plain_padding = ArrayType::scalar(DataType::F64);
        assert_eq!(plain_input.pad(&plain_padding, &[1], &[2], &[1]), Ok(ArrayType::new_static(DataType::F64, [8])));
        assert_eq!(
            plain_input.pad(&plain_padding, &[], &[0], &[0]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` `edge_padding_low` has length 0 but input has rank 1"
            )))),
        );
        assert_eq!(
            plain_input.pad(&plain_padding, &[0], &[], &[0]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` `edge_padding_high` has length 0 but input has rank 1"
            )))),
        );
        assert_eq!(
            plain_input.pad(&plain_padding, &[0], &[0], &[]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` `interior_padding` has length 0 but input has rank 1"
            )))),
        );

        // Padding preserves a common memory placement and rejects a padding scalar that would require an implicit
        // transfer. An effective identity keeps the complete input type, including its layout.
        let host_input = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_memory(Memory::Host { pinned: true });
        let host_padding = ArrayType::scalar(DataType::F32).with_memory(Memory::Host { pinned: true });
        assert_eq!(host_input.pad(&host_padding, &[0], &[1], &[0]).unwrap().memory(), Memory::Host { pinned: true });
        assert_eq!(
            host_input.pad(&padding_value, &[0], &[1], &[0]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` input and padding value must share one memory space but reside in \
                 `Host[Pinned]` and `Device`"
            )))),
        );
        let laid_out_input = host_input.with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        assert_eq!(laid_out_input.pad(&host_padding, &[0], &[0], &[0]), Ok(laid_out_input.clone()));

        // Padding to an evenly divisible size keeps the input sharding (including the unreduced manual axis): with low
        // = 0, interior = 0, and high = 4 the output is 0 + 4 + 4 = 8, divisible by the `x` mesh-axis size (2).
        assert_eq!(input.pad(&padding_value, &[0], &[4], &[0]).unwrap().sharding(), Some(&sharding));
        // Padding to a size not divisible by the explicit mesh-axis size (output 0 + 4 + 1 = 5) is rejected.
        assert_eq!(
            input.pad(&padding_value, &[0], &[1], &[0]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` on a dimension sharded over explicit mesh axes requires the output size (5) \
                 at axis 0 to be divisible by the mesh-axis product (2)",
            )))),
        );

        // Padding requires exact dependency metadata whenever the padding value can contribute. Neither reduced axes
        // nor varying manual axes are implicitly unioned from the scalar.
        let plain_padding = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(sharding.mesh().clone(), 0))
            .unwrap();
        assert_eq!(
            input.pad(&plain_padding, &[0], &[4], &[0]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` input and padding value must have matching reduced and unreduced mesh axes but \
                 got input type `{input}` and padding value type `{plain_padding}`",
            )))),
        );
        let reduced_padding = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(sharding.mesh().clone(), 0).with_reduced_axes(["m"]).unwrap())
            .unwrap();
        assert_eq!(
            input.pad(&reduced_padding, &[0], &[4], &[0]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` input and padding value must have matching reduced and unreduced mesh axes but \
                 got input type `{input}` and padding value type `{reduced_padding}`",
            )))),
        );
        let varying_input = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(
                Sharding::new(sharding.mesh().clone(), vec![ShardingDimension::sharded(["x"])])
                    .unwrap()
                    .with_varying_manual_axes(["m"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            varying_input.pad(&plain_padding, &[0], &[4], &[0]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` input and padding value must have matching varying manual axes but got input \
                 type `{varying_input}` and padding value type `{plain_padding}`",
            )))),
        );

        // Mesh identity is irrelevant for an ordinary scalar when neither side carries VMA or reduction metadata;
        // the result placement is derived solely from the input. Effective identities do not consult the unused padding
        // value's dependency metadata at all.
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("other", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let ordinary_input_sharding =
            Sharding::new(sharding.mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let ordinary_input = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(ordinary_input_sharding.clone())
            .unwrap();
        let other_mesh_padding =
            ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(other_mesh, 0)).unwrap();
        assert_eq!(
            ordinary_input.pad(&other_mesh_padding, &[0], &[4], &[0]).unwrap().sharding(),
            Some(&ordinary_input_sharding),
        );
        assert_eq!(varying_input.pad(&plain_padding, &[0], &[0], &[0]), Ok(varying_input));
    }

    #[test]
    fn test_array_pad() {
        // A rank-2 pad exercises the odometer across axes with different padding amounts: rows gain one interior row
        // and columns gain asymmetric edge padding.
        let input =
            Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2, 2]), &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = input
            .pad(
                &Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[0.0]).unwrap(),
                &[0, 1],
                &[1, 0],
                &[1, 0],
            )
            .unwrap();
        assert_eq!(
            *output.r#type(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(4), Dimension::Static(3)])),
        );
        assert_eq!(output.elements::<f64>(), Ok(vec![0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0]));

        // Signed edge padding crops the dilated input. Cropping can be asymmetric, can combine with interior dilation,
        // and must not be elided merely because the output shape happens to equal the input shape.
        assert_eq!(
            Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [5]), &[1.0, 2.0, 3.0, 4.0, 5.0])
                .unwrap()
                .pad(
                    &Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[0.0]).unwrap(),
                    &[-1],
                    &[-2],
                    &[0]
                )
                .unwrap()
                .elements::<f64>(),
            Ok(vec![2.0, 3.0]),
        );
        assert_eq!(
            Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[1.0, 2.0, 3.0])
                .unwrap()
                .pad(&Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[9.0]).unwrap(), &[-1], &[1], &[1])
                .unwrap()
                .elements::<f64>(),
            Ok(vec![9.0, 2.0, 9.0, 3.0, 9.0]),
        );
        assert_eq!(
            Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[1.0, 2.0, 3.0])
                .unwrap()
                .pad(&Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[9.0]).unwrap(), &[-1], &[1], &[0])
                .unwrap()
                .elements::<f64>(),
            Ok(vec![2.0, 3.0, 9.0]),
        );
        assert_eq!(
            Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [1]), &[1.0]).unwrap().pad(
                &Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[0.0]).unwrap(),
                &[-2],
                &[0],
                &[0]
            ),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` output size is negative (-1) on axis 0"
            )))),
        );

        // Interior padding is an effective identity on singleton axes. The eager and abstract fast paths preserve the
        // complete type, including its layout, and never compute a stride from an interior amount that no adjacent
        // pair of elements can use.
        let singleton_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![7])));
        let singleton = Array::from_elements::<f64>(singleton_type.clone(), &[3.0]).unwrap();
        let identity = singleton
            .pad(
                &Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[0.0]).unwrap(),
                &[0],
                &[0],
                &[usize::MAX],
            )
            .unwrap();
        assert_eq!(*identity.r#type(), singleton_type);
        assert_eq!(identity.elements::<f64>(), Ok(vec![3.0]));

        // The kernel validates the padding value shape eagerly.
        assert_eq!(
            Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2]), &[1.0, 2.0]).unwrap().pad(
                &Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [1]), &[0.0]).unwrap(),
                &[0],
                &[0],
                &[0]
            ),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` padding value must be a scalar but has type `f64[1]`"
            )))),
        );
    }

    #[test]
    fn test_array_pad_structural_zero() {
        // Structural-zero arrays have no element bytes, so even an enormous padded shape needs no traversal or fill.
        let input = Array::new(ArrayType::new_static(DataType::Zero, [usize::MAX - 1]), Vec::new()).unwrap();
        let padding = Array::new(ArrayType::scalar(DataType::Zero), Vec::new()).unwrap();
        let output = input.pad(&padding, &[1], &[0], &[0]).unwrap();
        assert_eq!(output.r#type().as_ref(), &ArrayType::new_static(DataType::Zero, [usize::MAX]));
        assert!(output.storage_bytes().is_empty());
    }

    #[test]
    fn test_array_pad_layouts() {
        // Narrow encodings and complex special values are copied verbatim, without numeric conversion.
        let input = Array::new(ArrayType::new_static(DataType::F8E8M0FNU, [1]), vec![127]).unwrap();
        let padding = Array::new(ArrayType::scalar(DataType::F8E8M0FNU), vec![255]).unwrap();
        assert_eq!(input.pad(&padding, &[1], &[1], &[0]).unwrap().storage_bytes(), &[255, 127, 255]);
        let values = [ComplexNumber::new(-0_f32, f32::INFINITY), ComplexNumber::new(f32::from_bits(0x7fc00001), -2.)];
        let input = Array::from_elements(ArrayType::new_static(DataType::C64, [2]), &values).unwrap();
        let padding_value = ComplexNumber::new(3_f32, -0.);
        let padding = Array::from_elements(ArrayType::scalar(DataType::C64), &[padding_value]).unwrap();
        let expected = Array::from_elements(
            ArrayType::new_static(DataType::C64, [5]),
            &[padding_value, values[0], padding_value, values[1], padding_value],
        )
        .unwrap();
        assert_eq!(input.pad(&padding, &[1], &[1], &[1]).unwrap().storage_bytes(), expected.storage_bytes());

        let vector = Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2]), &[1.0, 2.0]).unwrap();
        let padded = vector
            .pad(&Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[0.5]).unwrap(), &[1], &[2], &[1])
            .unwrap();
        assert_eq!(
            padded,
            Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [6]), &[0.5, 1.0, 0.5, 2.0, 0.5, 0.5])
                .unwrap()
        );

        // Padding copies both the reversed input layout and the rank-zero padding element by their exact bytes.
        let input_type =
            ArrayType::new_static(DataType::U16, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![-2])));
        let vector = Array::from_elements(input_type, &[1u16, 2]).unwrap();
        let padded = vector
            .pad(&Array::from_elements::<u16>(ArrayType::scalar(DataType::U16), &[9u16]).unwrap(), &[1], &[1], &[1])
            .unwrap();
        assert_eq!(padded.r#type().into_owned(), ArrayType::new_static(DataType::U16, [5]));
        assert_eq!(padded.elements::<u16>(), Ok(vec![9, 1, 9, 2, 9]));
        assert_eq!(padded.storage_bytes(), [9, 0, 1, 0, 9, 0, 2, 0, 9, 0]);
    }

    #[test]
    fn test_array_ir_pad() {
        // The homogeneous-to-mixed conversion cannot prove anything about its eventual inputs, so it conservatively
        // retains the runtime assertion, which participates in equality and effects but not in rendering: the renderer
        // prints only the three padding vectors.
        let homogeneous_operation = PadOperation::<ArrayType>::new(vec![1], vec![2], vec![1]).unwrap();
        let operation = PadOperation::<ArrayIrType>::new(vec![1], vec![2], vec![1]).unwrap();
        assert_eq!(operation, PadOperation::<ArrayIrType>::from(homogeneous_operation.clone()));
        assert_eq!(operation.name(), PAD_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "pad [edge_padding_low=[1], edge_padding_high=[2], interior_padding=[1]]");
        assert_eq!(operation.edge_padding_low(), &[1]);
        assert_eq!(operation.edge_padding_high(), &[2]);
        assert_eq!(operation.interior_padding(), &[1]);
        assert!(operation.requires_runtime_assertion());
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        assert_eq!(operation, operation.clone());

        // A complete static signature proves the output extent, so the operation becomes pure.
        let signature = [
            ArrayType::new_static(DataType::F64, [3]).into(),
            ArrayType::scalar(DataType::F64).into(),
            DimensionValue::constant(8).unwrap().r#type().into_owned().into(),
        ];
        let proven = operation.clone().with_input_types(&signature).unwrap();
        assert!(!proven.requires_runtime_assertion());
        assert_eq!(proven.effects().classes(), EffectClasses::NONE);
        assert_eq!(format!("{proven}"), format!("{operation}"));
        assert_ne!(proven, operation);

        // Converting to the homogeneous form drops the proof, and converting back is a one-way loss of provenness.
        assert_eq!(PadOperation::<ArrayType>::from(proven.clone()), homogeneous_operation);
        assert_eq!(PadOperation::<ArrayIrType>::from(PadOperation::<ArrayType>::from(proven.clone())), operation);
        assert!(matches!(
            ArrayIrOperation::<Array>::from(homogeneous_operation),
            ArrayIrOperation::Pad(operation) if operation.requires_runtime_assertion(),
        ));

        // Program rendering uses the canonical operation name and includes the trailing output-extent operand.
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let program_input = builder.add_input(ArrayType::new_static(DataType::F64, [3]).into());
        let program_padding_value = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let program_extent = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(8).unwrap()));
        let program_output = builder
            .add_instruction(proven, Vec::new(), vec![program_input, program_padding_value, program_extent], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![program_output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[] .
                let %2:dimension<8> = const 8
                    %3:f64[8] = pad [edge_padding_low=[1], edge_padding_high=[2], interior_padding=[1]] %0 %1 %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_array_ir_pad_with_input_types() {
        let operation = PadOperation::<ArrayIrType>::new(vec![1], vec![1], vec![0]).unwrap();
        let signature = [
            ArrayType::new_static(DataType::F32, [2]).into(),
            ArrayType::scalar(DataType::F32).into(),
            DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
        ];
        let proven = operation.clone().with_input_types(&signature).unwrap();
        assert!(!proven.requires_runtime_assertion());
        assert_eq!(proven.effects().classes(), EffectClasses::NONE);
        let size = DimensionVariable::new("size", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![size.clone().into()]));
        let result_size = DimensionVariable::new("result", DimensionBounds::new(3, Some(7)).unwrap());
        let dynamic_signature = [
            dynamic_type.clone().into(),
            ArrayType::scalar(DataType::F32).into(),
            DimensionType::new(result_size).into(),
        ];
        assert!(operation.clone().with_input_types(&dynamic_signature).unwrap().requires_runtime_assertion());
        assert_eq!(
            proven.infer_output_types(&dynamic_signature, &[]),
            Err(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` was constructed without a runtime extent check but these input types require \
                 one",
            )))
        );
        assert_eq!(
            operation.with_input_types(&[
                dynamic_type.clone().into(),
                ArrayType::scalar(DataType::F32).into(),
                DimensionType::new(DimensionVariable::new("disjoint", DimensionBounds::new(9, Some(10)).unwrap()))
                    .into(),
            ]),
            Err(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` output bounds [9, 10) on axis 0 cannot contain a padded extent derived from \
                 input bounds [1, 5)",
            )))
        );
        // Balanced edges that keep the input identity are proven without evaluating any dimension input.
        let identity = PadOperation::<ArrayIrType>::new(vec![-1], vec![1], vec![0])
            .unwrap()
            .with_input_types(&[
                dynamic_type.into(),
                ArrayType::scalar(DataType::F32).into(),
                DimensionType::new(size).into(),
            ])
            .unwrap();
        assert!(!identity.requires_runtime_assertion());
        assert_eq!(identity.effects().classes(), EffectClasses::NONE);
    }

    #[test]
    fn test_array_ir_pad_requires_runtime_assertion() {
        let operation = PadOperation::<ArrayIrType>::new(vec![1], vec![1], vec![0]).unwrap();
        assert!(operation.requires_runtime_assertion());
        let checked = operation
            .with_input_types(&[
                ArrayType::new_static(DataType::F32, [2]).into(),
                ArrayType::scalar(DataType::F32).into(),
                DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
            ])
            .unwrap();
        assert!(!checked.requires_runtime_assertion());
        let round_trip = PadOperation::<ArrayIrType>::from(PadOperation::<ArrayType>::from(checked));
        assert!(round_trip.requires_runtime_assertion());
    }

    #[test]
    fn test_array_ir_pad_type_inference() {
        let input_type = ArrayType::new_static(DataType::F64, [3]);
        let padding_value_type = ArrayType::scalar(DataType::F64);
        let eight = DimensionValue::constant(8).unwrap().r#type().into_owned();
        let dynamic_variable = DimensionVariable::new("dynamic", DimensionBounds::new(7, Some(10)).unwrap());
        let operation = PadOperation::<ArrayIrType>::new(vec![1], vec![2], vec![1]).unwrap();
        // The conservative form accepts every well-formed signature: a static extent must equal the padded extent, a
        // supplied dynamic identity names the output axis, layouts are cleared, and the array validation shared with
        // the homogeneous form applies. Malformed operand lists report exact errors.
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    type = ArrayIrType,
                    input_types = [input_type.clone().into(), padding_value_type.clone().into(), eight.clone().into()],
                    output_types = [ArrayType::new_static(DataType::F64, [8]).into()],
                },
                {
                    type = ArrayIrType,
                    input_types = [
                        input_type.clone().with_layout(Layout::Strided(StridedLayout::new(vec![2]))).into(),
                        padding_value_type.clone().into(),
                        eight.clone().into(),
                    ],
                    output_types = [ArrayType::new_static(DataType::F64, [8]).into()],
                },
                {
                    type = ArrayIrType,
                    input_types = [
                        input_type.clone().into(),
                        padding_value_type.clone().into(),
                        DimensionType::new(DimensionVariable::new("wrong", DimensionBounds::new(7, Some(8)).unwrap()))
                            .into(),
                    ],
                    error = format!(
                        "`{PAD_OPERATION_NAME}` output bounds [7, 8) on axis 0 cannot contain a padded extent derived \
                         from input bounds [3, 4)",
                    ),
                },
                {
                    type = ArrayIrType,
                    input_types = [
                        input_type.clone().into(),
                        padding_value_type.clone().into(),
                        DimensionType::new(dynamic_variable.clone()).into(),
                    ],
                    output_types = [
                        ArrayType::new(DataType::F64, Shape::new(vec![dynamic_variable.clone().into()])).into(),
                    ],
                },
                {
                    type = ArrayIrType,
                    input_types = [input_type.clone().into()],
                    error = "expected at least 2 inputs but got 1",
                },
                {
                    type = ArrayIrType,
                    input_types = [
                        input_type.clone().into(),
                        padding_value_type.clone().into(),
                        eight.clone().into(),
                        eight.clone().into(),
                    ],
                    error = format!(
                        "`{PAD_OPERATION_NAME}` expects an input, a padding value, and one output extent per axis (3 \
                         inputs total) but got 4",
                    ),
                },
                {
                    type = ArrayIrType,
                    input_types = [eight.clone().into(), padding_value_type.clone().into(), eight.clone().into()],
                    error = "expected array type but got dimension type",
                },
                {
                    type = ArrayIrType,
                    input_types = [input_type.clone().into(), eight.clone().into(), eight.clone().into()],
                    error = "expected array type but got dimension type",
                },
                {
                    type = ArrayIrType,
                    input_types = [
                        input_type.clone().into(),
                        padding_value_type.clone().into(),
                        padding_value_type.clone().into(),
                    ],
                    error = "expected dimension type but got array type",
                },
                {
                    type = ArrayIrType,
                    input_types = [
                        input_type.clone().into(),
                        ArrayType::scalar(DataType::F32).into(),
                        eight.clone().into(),
                    ],
                    error = format!(
                        "`{PAD_OPERATION_NAME}` input data type `f64` does not match padding value data type `f32`",
                    ),
                },
            ],
        );
        assert_eq!(
            operation.infer_output_types(
                &[input_type.clone().into(), padding_value_type.clone().into(), eight.into()],
                &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1"))
        );

        // An effective identity returns the input type unchanged, including its layout, when the supplied extents
        // equal the input dimensions.
        let laid_out_type = input_type.clone().with_layout(Layout::Strided(StridedLayout::new(vec![2])));
        check_operation_type_inference!(
            operation = PadOperation::<ArrayIrType>::new(vec![0], vec![0], vec![0]).unwrap(),
            cases = [{
                type = ArrayIrType,
                input_types = [
                    laid_out_type.clone().into(),
                    padding_value_type.clone().into(),
                    DimensionValue::constant(3).unwrap().r#type().into_owned().into(),
                ],
                output_types = [laid_out_type.into()],
            }],
        );

        // Intersecting result bounds admit valid runtime geometries, and a possibly empty input crops to a possibly
        // empty output. A proven operation rejects signatures whose extents it cannot prove statically.
        let input_variable = DimensionVariable::new("input", DimensionBounds::new(1, Some(5)).unwrap());
        let output_variable = DimensionVariable::new("output", DimensionBounds::new(3, Some(7)).unwrap());
        let narrow_variable = DimensionVariable::new("narrow", DimensionBounds::new(3, Some(6)).unwrap());
        let dynamic_input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(input_variable.clone())]));
        let dynamic_operation = PadOperation::<ArrayIrType>::new(vec![1], vec![1], vec![0]).unwrap();
        check_operation_type_inference!(
            operation = dynamic_operation.clone(),
            cases = [
                {
                    type = ArrayIrType,
                    input_types = [
                        dynamic_input_type.clone().into(),
                        padding_value_type.clone().into(),
                        DimensionType::new(output_variable.clone()).into(),
                    ],
                    output_types = [ArrayType::new(DataType::F64, Shape::new(vec![output_variable.into()])).into()],
                },
                {
                    type = ArrayIrType,
                    input_types = [
                        dynamic_input_type.clone().into(),
                        padding_value_type.clone().into(),
                        DimensionType::new(narrow_variable.clone()).into(),
                    ],
                    output_types = [ArrayType::new(DataType::F64, Shape::new(vec![narrow_variable.into()])).into()],
                },
            ],
        );
        let possibly_empty_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                "possibly_empty",
                DimensionBounds::new(0, Some(5)).unwrap(),
            ))]),
        );
        let cropped_variable = DimensionVariable::new("cropped", DimensionBounds::new(0, Some(4)).unwrap());
        check_operation_type_inference!(
            operation = PadOperation::<ArrayIrType>::new(vec![-1], vec![0], vec![0]).unwrap(),
            cases = [{
                type = ArrayIrType,
                input_types = [
                    possibly_empty_type.into(),
                    padding_value_type.clone().into(),
                    DimensionType::new(cropped_variable.clone()).into(),
                ],
                output_types = [ArrayType::new(DataType::F64, Shape::new(vec![cropped_variable.into()])).into()],
            }],
        );
        let proven_operation = dynamic_operation
            .with_input_types(&[
                ArrayType::new_static(DataType::F64, [2]).into(),
                padding_value_type.clone().into(),
                DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
            ])
            .unwrap();
        check_operation_type_inference!(
            operation = proven_operation,
            cases = [
                {
                    type = ArrayIrType,
                    input_types = [
                        ArrayType::new_static(DataType::F64, [2]).into(),
                        padding_value_type.clone().into(),
                        DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
                    ],
                    output_types = [ArrayType::new_static(DataType::F64, [4]).into()],
                },
                {
                    type = ArrayIrType,
                    input_types = [
                        dynamic_input_type.into(),
                        padding_value_type.into(),
                        DimensionType::new(DimensionVariable::new("output", DimensionBounds::new(3, Some(7)).unwrap()))
                            .into(),
                    ],
                    error = format!(
                        "`{PAD_OPERATION_NAME}` was constructed without a runtime extent check but these input types \
                         require one",
                    ),
                },
            ],
        );
    }

    #[test]
    fn test_array_ir_pad_interpretation() {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let operation = PadOperation::<ArrayIrType>::new(vec![1], vec![2], vec![1]).unwrap();
        let input = ArrayIrValue::Array(
            Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[1.0, 2.0, 3.0]).unwrap(),
        );
        let padding_value =
            ArrayIrValue::Array(Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[9.0]).unwrap());
        // A concrete mixed value validates the explicit extent against the padded input geometry before padding its
        // array member.
        assert_eq!(
            operation.interpret(
                &context,
                &EmptyRegionDriver,
                &[input.clone(), padding_value.clone(), ArrayIrValue::Dimension(DimensionValue::constant(8).unwrap()),],
            ),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements::<f64>(
                    ArrayType::new_static(DataType::F64, [8]),
                    &[9.0, 1.0, 9.0, 2.0, 9.0, 3.0, 9.0, 9.0],
                )
                .unwrap()
            )]),
        );
        assert_eq!(
            operation.interpret(
                &context,
                &EmptyRegionDriver,
                &[input, padding_value, ArrayIrValue::Dimension(DimensionValue::constant(7).unwrap())],
            ),
            Err(ProgramError::InvalidArgument {
                message: format!(
                    "`{PAD_OPERATION_NAME}` output axis 0 has extent 8, but its explicit extent input is 7"
                ),
            }),
        );
        // The mixed arity is one extent per configured axis on top of the two array inputs.
        assert_eq!(
            operation.interpret(&context, &EmptyRegionDriver, &[]),
            Err(ProgramError::InvalidInputCount { expected: 3, actual: 0 }),
        );
    }

    #[test]
    fn test_array_ir_pad_partial_evaluation() {
        let input = ArrayIrValue::Array(
            Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[1.0, 2.0, 3.0]).unwrap(),
        );
        let padding_value =
            ArrayIrValue::Array(Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[9.0]).unwrap());
        let extent = ArrayIrValue::Dimension(DimensionValue::constant(8).unwrap());
        let expected = ArrayIrValue::Array(
            Array::from_elements::<f64>(
                ArrayType::new_static(DataType::F64, [8]),
                &[9.0, 1.0, 9.0, 2.0, 9.0, 3.0, 9.0, 9.0],
            )
            .unwrap(),
        );
        // Partial evaluation folds a fully known pad and otherwise retains exactly one operation with the explicit
        // extent edge, including when only that extent is unknown.
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = PadOperation::<ArrayIrType>::new(vec![1], vec![2], vec![1]).unwrap(),
            cases = [
                {
                    inputs = [(@known, input.clone()), (@known, padding_value.clone()), (@known, extent.clone())],
                    outputs = [(@known, expected.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = input.r#type().into_owned(), replay = input.clone())),
                        (@known, padding_value.clone()),
                        (@known, extent.clone()),
                    ],
                    outputs = [(@residual, expected.clone())],
                    residual_instructions = 1,
                },
                {
                    inputs = [
                        (@known, input.clone()),
                        (@unknown(type = padding_value.r#type().into_owned(), replay = padding_value.clone())),
                        (@known, extent.clone()),
                    ],
                    outputs = [(@residual, expected.clone())],
                    residual_instructions = 1,
                },
                {
                    inputs = [
                        (@known, input.clone()),
                        (@known, padding_value.clone()),
                        (@unknown(type = extent.r#type().into_owned(), replay = extent.clone())),
                    ],
                    outputs = [(@residual, expected.clone())],
                    residual_instructions = 1,
                },
            ],
        );

        // An unused pad whose extent is not proven must survive simplification, because its ordered assertion still
        // validates the supplied extent, while a proven pad has no observable consequence and is eliminated.
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap()));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let program_input = builder.add_input(input.r#type().into_owned());
        let program_padding_value = builder.add_input(padding_value.r#type().into_owned());
        let program_extent = builder.add_input(extent_type.into());
        builder
            .add_instruction(
                PadOperation::<ArrayType>::new(vec![1], vec![2], vec![1]).unwrap(),
                Vec::new(),
                vec![program_input, program_padding_value, program_extent],
                None,
            )
            .unwrap();
        let unproven = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![program_input],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap()
            .into_simplified()
            .unwrap();
        assert_eq!(
            unproven.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[], %2:dimension<extent ∈ [1, 9)> .
                let %3:f64[extent] = pad [edge_padding_low=[1], edge_padding_high=[2], interior_padding=[1]] %0 %1 %2
                in (%0)
            "}
            .trim_end(),
        );
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let program_input = builder.add_input(input.r#type().into_owned());
        let program_padding_value = builder.add_input(padding_value.r#type().into_owned());
        let program_extent = builder.add_input(extent.r#type().into_owned());
        let proven_operation = PadOperation::<ArrayIrType>::new(vec![1], vec![2], vec![1])
            .unwrap()
            .with_input_types(&[
                input.r#type().into_owned(),
                padding_value.r#type().into_owned(),
                extent.r#type().into_owned(),
            ])
            .unwrap();
        builder
            .add_instruction(
                proven_operation,
                Vec::new(),
                vec![program_input, program_padding_value, program_extent],
                None,
            )
            .unwrap();
        let proven = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![program_input],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap()
            .into_simplified()
            .unwrap();
        assert!(proven.instructions().is_empty());
    }

    #[test]
    fn test_array_ir_pad_batching() {
        // A mapped input keeps its batch axis with zero padding amounts inserted at that position, while the
        // replicated padding value and explicit extents pass through and the batch extent joins the lifted extents.
        // The mixed batch and policy types differ from the array ones that `check_operation_batching!` constructs, so
        // these cases stay explicit.
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let operation = PadOperation::<ArrayIrType>::new(vec![1], vec![0], vec![0]).unwrap();
        let padding_value = ArrayIrBatch::replicated(ArrayIrValue::Array(
            Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[9.0]).unwrap(),
        ));
        let extent = ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        let input = ArrayIrBatch::new(
            ArrayIrValue::Array(
                Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2, 2]), &[1.0, 2.0, 3.0, 4.0])
                    .unwrap(),
            ),
            BatchAxis::new(0),
        )
        .unwrap();
        let batched = operation
            .batch(&context, &EmptyRegionDriver, &[input, padding_value.clone(), extent.clone()])
            .unwrap();
        let (outputs, _) = batched.into_parts();
        assert_eq!(
            outputs,
            vec![
                ArrayIrBatch::new(
                    ArrayIrValue::Array(
                        Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [2, 3]),
                            &[9.0, 1.0, 2.0, 9.0, 3.0, 4.0],
                        )
                        .unwrap(),
                    ),
                    BatchAxis::new(0),
                )
                .unwrap()
            ],
        );

        // A non-leading batch axis stays in place, so the padding amounts are lifted around it.
        let input = ArrayIrBatch::new(
            ArrayIrValue::Array(
                Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2, 2]), &[1.0, 2.0, 3.0, 4.0])
                    .unwrap(),
            ),
            BatchAxis::new(1),
        )
        .unwrap();
        let batched = operation
            .batch(&context, &EmptyRegionDriver, &[input, padding_value.clone(), extent.clone()])
            .unwrap();
        let (outputs, _) = batched.into_parts();
        assert_eq!(
            outputs,
            vec![
                ArrayIrBatch::new(
                    ArrayIrValue::Array(
                        Array::from_elements::<f64>(
                            ArrayType::new_static(DataType::F64, [3, 2]),
                            &[9.0, 9.0, 1.0, 2.0, 3.0, 4.0],
                        )
                        .unwrap(),
                    ),
                    BatchAxis::new(1),
                )
                .unwrap()
            ],
        );

        // Fully replicated inputs bind the operation as given and keep the output replicated.
        let input = ArrayIrBatch::replicated(ArrayIrValue::Array(
            Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2]), &[1.0, 2.0]).unwrap(),
        ));
        let batched = operation.batch(&context, &EmptyRegionDriver, &[input, padding_value, extent]).unwrap();
        let (outputs, _) = batched.into_parts();
        assert_eq!(
            outputs,
            vec![ArrayIrBatch::replicated(ArrayIrValue::Array(
                Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[9.0, 1.0, 2.0]).unwrap(),
            ))],
        );
    }

    #[test]
    fn test_array_ir_pad_batching_decomposes_mapped_padding_values() {
        // A mapped padding value is decomposed into placeholder padding, a padding-position mask, a broadcast of the
        // per-item scalar, and a select, so each batch item receives its own padding value.
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let operation =
            ArrayIrOperation::<Array>::from(PadOperation::<ArrayType>::new(vec![1], vec![0], vec![0]).unwrap());
        let input = ArrayIrBatch::new(
            ArrayIrValue::Array(
                Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [2, 2]), &[1.0_f32, 2.0, 3.0, 4.0])
                    .unwrap(),
            ),
            BatchAxis::new(0),
        )
        .unwrap();
        let padding_value = ArrayIrBatch::new(
            ArrayIrValue::Array(
                Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [2]), &[8.0_f32, 9.0]).unwrap(),
            ),
            BatchAxis::new(0),
        )
        .unwrap();
        let extent = ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        let batched = operation.batch(&context, &EmptyRegionDriver, &[input, padding_value, extent]).unwrap();
        let (outputs, _) = batched.into_parts();
        assert_eq!(
            outputs,
            vec![
                ArrayIrBatch::new(
                    ArrayIrValue::Array(
                        Array::from_elements::<f32>(
                            ArrayType::new_static(DataType::F32, [2, 3]),
                            &[8.0_f32, 1.0, 2.0, 9.0, 3.0, 4.0],
                        )
                        .unwrap(),
                    ),
                    BatchAxis::new(0),
                )
                .unwrap()
            ],
        );

        // Under a symbolic mapped extent, the replicated input is aligned through a dynamic broadcast and every
        // shape-changing instruction of the decomposition receives the same explicit output extents, including the
        // inserted batch extent.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_extent = trace.input(DimensionType::new(batch.clone()).into());
        let input = trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)])).into());
        let padding_value =
            trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch)])).into());
        let result_extent = trace.input(DimensionValue::constant(3).unwrap().r#type().into_owned().into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent);
        let outputs = context
            .bind(
                operation,
                Vec::new(),
                &[
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(input)),
                    BatchingTracer::new(context.clone(), ArrayIrBatch::new(padding_value, BatchAxis::new(0)).unwrap()),
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(result_extent)),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(0));
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![outputs[0].batch().value().atom_id().unwrap()],
                vec![Placeholder; 4],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f32[2], %2:f32[batch], %3:dimension<3> .
                let %4:dimension<2> = constant [value=2]
                    %5:f32[batch, 2] = broadcast [output_axes=[1]] %1 %0 %4
                    %6:f32[] = one [type=f32[]]
                    %7:f32[batch, 3] = pad [edge_padding_low=[0, 1], edge_padding_high=[0, 0], \
                        interior_padding=[0, 0]] %5 %6 %0 %3
                    %8:bool[batch, 2] = one [type=bool[batch, 2]] %0
                    %9:bool[] = zero [type=bool[]]
                    %10:bool[batch, 3] = pad [edge_padding_low=[0, 1], edge_padding_high=[0, 0], \
                        interior_padding=[0, 0]] %8 %9 %0 %3
                    %11:f32[batch, 3] = broadcast [output_axes=[0]] %2 %0 %3
                    %12:f32[batch, 3] = select %10 %7 %11
                in (%12)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_array_ir_pad_batching_preserves_runtime_assertion_proof() {
        // A proven, effect-free mixed pad must stay proven after batching over a mapped input, so that it can still be
        // eliminated when its output is unused.
        let parent = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = parent.input(ArrayType::new_static(DataType::F64, [2, 3]).into());
        let padding_value = parent.input(ArrayType::scalar(DataType::F64).into());
        let two = parent.lift(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap())).unwrap();
        let four = parent.lift(ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap())).unwrap();
        let operation = PadOperation::<ArrayIrType>::new(vec![1], vec![0], vec![0])
            .unwrap()
            .with_input_types(&[
                ArrayType::new_static(DataType::F64, [3]).into(),
                ArrayType::scalar(DataType::F64).into(),
                four.r#type().into_owned(),
            ])
            .unwrap();
        assert!(!operation.requires_runtime_assertion());
        assert_eq!(operation.effects().classes(), EffectClasses::NONE);
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent.clone(), two);
        let batched = operation
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    ArrayIrBatch::new(input.clone(), BatchAxis::new(0)).unwrap(),
                    ArrayIrBatch::replicated(padding_value.clone()),
                    ArrayIrBatch::replicated(four),
                ],
            )
            .unwrap();
        let (outputs, _) = batched.into_parts();
        assert_eq!(outputs.len(), 1);
        let program = parent
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![input.atom_id().unwrap()],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 3], %1:f64[] .
                let %2:dimension<2> = const 2
                    %3:dimension<4> = const 4
                    %4:f64[2, 4] = pad [edge_padding_low=[0, 1], edge_padding_high=[0, 0], interior_padding=[0, 0]] \
                        %0 %1 %2 %3
                in (%0)
            "}
            .trim_end(),
        );
        assert_eq!(program.instructions()[0].operation().effects().classes(), EffectClasses::NONE);
        assert!(program.into_simplified().unwrap().instructions().is_empty());

        // The mapped padding-value decomposition stages a placeholder pad and a mask pad with the same lifted
        // geometry; both keep the proof, so the whole unused decomposition is eliminated.
        let parent = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = parent.input(ArrayType::new_static(DataType::F64, [2, 3]).into());
        let padding_value = parent.input(ArrayType::new_static(DataType::F64, [2]).into());
        let two = parent.lift(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap())).unwrap();
        let four = parent.lift(ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap())).unwrap();
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent.clone(), two);
        let batched = operation
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    ArrayIrBatch::new(input.clone(), BatchAxis::new(0)).unwrap(),
                    ArrayIrBatch::new(padding_value.clone(), BatchAxis::new(0)).unwrap(),
                    ArrayIrBatch::replicated(four),
                ],
            )
            .unwrap();
        let (outputs, _) = batched.into_parts();
        assert_eq!(outputs.len(), 1);
        let program = parent
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![input.atom_id().unwrap()],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 3], %1:f64[2] .
                let %2:dimension<2> = const 2
                    %3:dimension<4> = const 4
                    %4:f64[] = one [type=f64[]]
                    %5:f64[2, 4] = pad [edge_padding_low=[0, 1], edge_padding_high=[0, 0], interior_padding=[0, 0]] \
                        %0 %4 %2 %3
                    %6:bool[2, 3] = one [type=bool[2, 3]]
                    %7:bool[] = zero [type=bool[]]
                    %8:bool[2, 4] = pad [edge_padding_low=[0, 1], edge_padding_high=[0, 0], interior_padding=[0, 0]] \
                        %6 %7 %2 %3
                    %9:f64[2, 4] = broadcast [output_axes=[0]] %1 %2 %3
                    %10:f64[2, 4] = select %8 %5 %9
                in (%0)
            "}
            .trim_end(),
        );
        let pads = program
            .instructions()
            .iter()
            .filter_map(|instruction| match instruction.operation() {
                ArrayIrOperation::Pad(operation) => Some(operation),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(pads.len(), 2);
        assert!(pads.iter().all(|pad| !pad.requires_runtime_assertion()));
        assert!(pads.iter().all(|pad| pad.effects().classes() == EffectClasses::NONE));
        assert!(program.into_simplified().unwrap().instructions().is_empty());

        // An unproven pad keeps its ordered assertion after batching: dead-result elimination retains it, and a wrong
        // runtime extent still fails even though nothing consumes the output.
        let parent = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = parent.input(ArrayType::new_static(DataType::F64, [2, 3]).into());
        let padding_value = parent.input(ArrayType::scalar(DataType::F64).into());
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap()));
        let extent = parent.input(extent_type.clone().into());
        let two = parent.lift(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap())).unwrap();
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent.clone(), two);
        let batched = PadOperation::<ArrayIrType>::new(vec![1], vec![0], vec![0])
            .unwrap()
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    ArrayIrBatch::new(input.clone(), BatchAxis::new(0)).unwrap(),
                    ArrayIrBatch::replicated(padding_value.clone()),
                    ArrayIrBatch::replicated(extent),
                ],
            )
            .unwrap();
        let (outputs, _) = batched.into_parts();
        assert_eq!(outputs.len(), 1);
        let program = parent
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![input.atom_id().unwrap()],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap()
            .into_simplified()
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 3], %1:f64[], %2:dimension<extent ∈ [1, 9)> .
                let %3:dimension<2> = const 2
                    %4:f64[2, extent] = pad [edge_padding_low=[0, 1], edge_padding_high=[0, 0], \
                        interior_padding=[0, 0]] %0 %1 %3 %2
                in (%0)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.instructions()[0].operation().effects().classes(),
            EffectClasses::single(EffectClass::OrderedAssertion),
        );
        let values = ArrayIrValue::Array(
            Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2, 3]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap(),
        );
        let padding_value =
            ArrayIrValue::Array(Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[9.0]).unwrap());
        assert_eq!(
            program.interpret(vec![
                values.clone(),
                padding_value.clone(),
                ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 4).unwrap()),
            ]),
            Ok(vec![values.clone()]),
        );
        assert_eq!(
            program.interpret(vec![
                values,
                padding_value,
                ArrayIrValue::Dimension(DimensionValue::new(extent_type, 5).unwrap()),
            ]),
            Err(ProgramError::InvalidArgument {
                message: format!(
                    "`{PAD_OPERATION_NAME}` output axis 1 has extent 4, but its explicit extent input is 5"
                ),
            }),
        );
    }

    #[test]
    fn test_array_ir_pad_batching_rejects_malformed_inputs() {
        // The array inputs are mandatory and must be arrays, and every explicit extent must stay replicated.
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let operation = PadOperation::<ArrayIrType>::new(vec![1], vec![0], vec![0]).unwrap();
        assert_eq!(
            operation.batch(&context, &EmptyRegionDriver, &[]).unwrap_err(),
            BatchingError::Program(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );
        let extent = ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        assert_eq!(
            operation
                .batch(&context, &EmptyRegionDriver, &[extent.clone(), extent.clone(), extent])
                .unwrap_err(),
            BatchingError::Type(TypeError::invalid("expected array type but got dimension type")),
        );
        let input = ArrayIrBatch::new(
            ArrayIrValue::Array(
                Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2, 2]), &[1.0, 2.0, 3.0, 4.0])
                    .unwrap(),
            ),
            BatchAxis::new(0),
        )
        .unwrap();
        let padding_value = ArrayIrBatch::replicated(ArrayIrValue::Array(
            Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[9.0]).unwrap(),
        ));
        let extent_type = DimensionValue::constant(3).unwrap().r#type().into_owned();
        let mapped_extent = ArrayIrBatch::mapped_dimension(
            ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::I64, [2]), &[3_i64, 3]).unwrap()),
            BatchAxis::new(0),
            extent_type.clone(),
        )
        .unwrap();
        assert_eq!(
            operation.batch(&context, &EmptyRegionDriver, &[input, padding_value, mapped_extent]).unwrap_err(),
            BatchingError::MappedDimension { r#type: Box::new(extent_type), axis: BatchAxis::new(0) },
        );
    }

    #[test]
    fn test_array_ir_pad_differentiation() {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F64, [3]).into());
        let padding_value = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let output_extent = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(8).unwrap()));
        let output = builder
            .add_instruction(
                PadOperation::<ArrayType>::new(vec![1], vec![2], vec![1]).unwrap(),
                Vec::new(),
                vec![input, padding_value, output_extent],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        // Static geometry replays the mixed pad directly on the tangents under the same constant extent, which is a
        // non-differentiated shape value. `check_operation_differentiation!` perturbs every program input numerically,
        // which the dimension input does not support, so the transform is checked explicitly here.
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[], %2:f64[3], %3:f64[] .
                let %4:dimension<8> = const 8
                    %5:f64[8] = pad [edge_padding_low=[1], edge_padding_high=[2], interior_padding=[1]] %0 %1 %4
                    %6:f64[8] = pad [edge_padding_low=[1], edge_padding_high=[2], interior_padding=[1]] %2 %3 %4
                in (%5, %6)
            "}
            .trim_end(),
        );
        assert_eq!(
            jvp.interpret(vec![
                ArrayIrValue::Array(
                    Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[1.0, 2.0, 3.0]).unwrap(),
                ),
                ArrayIrValue::Array(Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[9.0]).unwrap()),
                ArrayIrValue::Array(
                    Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[0.1, 0.2, 0.3]).unwrap(),
                ),
                ArrayIrValue::Array(Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[0.5]).unwrap()),
            ]),
            Ok(vec![
                ArrayIrValue::Array(
                    Array::from_elements::<f64>(
                        ArrayType::new_static(DataType::F64, [8]),
                        &[9.0, 1.0, 9.0, 2.0, 9.0, 3.0, 9.0, 9.0],
                    )
                    .unwrap()
                ),
                ArrayIrValue::Array(
                    Array::from_elements::<f64>(
                        ArrayType::new_static(DataType::F64, [8]),
                        &[0.5, 0.1, 0.5, 0.2, 0.5, 0.3, 0.5, 0.5],
                    )
                    .unwrap()
                ),
            ]),
        );

        // The rule requires at least the two array inputs.
        let context =
            DifferentiationContext::fused(TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new());
        assert!(matches!(
            PadOperation::<ArrayIrType>::new(vec![1], vec![2], vec![1])
                .unwrap()
                .jvp(&context, &EmptyRegionDriver, &[],),
            Err(DifferentiationError::Program(ProgramError::InvalidInputCount { expected: 2, actual: 0 })),
        ));
    }

    #[test]
    fn test_array_ir_pad_differentiation_dynamic_geometry() {
        // Dynamic input geometry retains the exact input extent and the explicit output extent as residuals, so the
        // linear tangent map can be transposed: the pullback undoes the edge padding, slices the dilated input with
        // stride `interior + 1`, and sums the cotangent at the padding positions selected through a padded mask.
        let source = DimensionVariable::new("source", DimensionBounds::new(0, Some(5)).unwrap());
        let result = DimensionVariable::new("result", DimensionBounds::new(3, Some(11)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(source.clone())]));
        let result_type = DimensionType::new(result);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let padding_value = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let output_extent = builder.add_input(result_type.clone().into());
        let output = builder
            .add_instruction(
                PadOperation::<ArrayType>::new(vec![1], vec![2], vec![1]).unwrap(),
                Vec::new(),
                vec![input, padding_value, output_extent],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 2);
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[source], %1:f64[], %2:dimension<result ∈ [3, 11)>, %3:dimension<source ∈ [0, 5)> .
                let %4:f64[result] = linear_call [residual_count=2] %2 %3 %0 %1 [
                    forward={
                        lambda %0:dimension<result ∈ [3, 11)>, %1:dimension<source ∈ [0, 5)>, %2:f64[source], \
                            %3:f64[] .
                        let %4:f64[result] = pad [edge_padding_low=[1], edge_padding_high=[2], interior_padding=[1]] \
                            %2 %3 %0
                        in (%4)
                    },
                    transpose={
                        lambda %0:dimension<result ∈ [3, 11)>, %1:dimension<source ∈ [0, 5)>, %2:f64[result] .
                        let %3:f64[] = zero [type=f64[]]
                            %4:dimension<1> = constant [value=1]
                            %5:dimension<max(0, source - 1) ∈ [0, 4)> = dimension_saturating_sub %1 %4
                            %6:dimension<1> = constant [value=1]
                            %7:dimension<max(0, source - 1) * 1 ∈ [0, 4)> = dimension_mul %5 %6
                            %8:dimension<source + max(0, source - 1) * 1 ∈ [0, 8)> = dimension_add %1 %7
                            %9:f64[source + max(0, source - 1) * 1] = pad [edge_padding_low=[-1], \
                                edge_padding_high=[-2], interior_padding=[0]] %2 %3 %8
                            %10:dimension<0> = constant [value=0]
                            %11:f64[source] = dynamic_shape_slice [strides=[2]] %9 %10 %1
                            %12:bool[source] = zero [type=bool[source]] %1
                            %13:bool[] = one [type=bool[]]
                            %14:bool[result] = pad [edge_padding_low=[1], edge_padding_high=[2], \
                                interior_padding=[1]] %12 %13 %0
                            %15:f64[result] = zero [type=f64[result]] %0
                            %16:f64[result] = select %14 %2 %15
                            %17:f64[] = reduce_sum [axes=[0]] %16
                        in (%11, %17)
                    },
                ]
                in (%4)
            "}
            .trim_end(),
        );
        let pullback = linearization.pullback().unwrap();

        let input = ArrayIrValue::Array(
            Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[10.0_f64, 20.0, 30.0]).unwrap(),
        );
        let padding_value =
            ArrayIrValue::Array(Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[-1.0_f64]).unwrap());
        let output_extent = ArrayIrValue::Dimension(DimensionValue::new(result_type.clone(), 8).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input, padding_value, output_extent]).unwrap();
        assert_eq!(
            primal_outputs[0],
            ArrayIrValue::Array(
                Array::from_elements::<f64>(
                    ArrayType::new_static(DataType::F64, [8]),
                    &[-1.0_f64, 10.0, -1.0, 20.0, -1.0, 30.0, -1.0, -1.0]
                )
                .unwrap()
            ),
        );
        let residuals = primal_outputs.split_off(1);

        let mut tangent_inputs = vec![
            ArrayIrValue::Array(
                Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[1.0_f64, 2.0, 3.0]).unwrap(),
            ),
            ArrayIrValue::Array(Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[4.0_f64]).unwrap()),
        ];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements::<f64>(
                    ArrayType::new_static(DataType::F64, [8]),
                    &[4.0_f64, 1.0, 4.0, 2.0, 4.0, 3.0, 4.0, 4.0]
                )
                .unwrap()
            )]),
        );

        let mut pullback_inputs = vec![ArrayIrValue::Array(
            Array::from_elements::<f64>(
                ArrayType::new_static(DataType::F64, [8]),
                &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            )
            .unwrap(),
        )];
        pullback_inputs.extend(residuals);
        assert_eq!(
            pullback.interpret(pullback_inputs),
            Ok(vec![
                ArrayIrValue::Array(
                    Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[2.0_f64, 4.0, 6.0])
                        .unwrap()
                ),
                ArrayIrValue::Array(
                    Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[24.0_f64]).unwrap()
                ),
            ]),
        );

        // An empty input contributes no interior padding, so every output position is a padding position.
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(
                    Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [0]), &[]).unwrap(),
                ),
                ArrayIrValue::Array(
                    Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[-1.0_f64]).unwrap(),
                ),
                ArrayIrValue::Dimension(DimensionValue::new(result_type, 3).unwrap()),
            ])
            .unwrap();
        assert_eq!(
            primal_outputs[0],
            ArrayIrValue::Array(
                Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[-1.0_f64, -1.0, -1.0])
                    .unwrap()
            ),
        );
        let residuals = primal_outputs.split_off(1);
        let mut pullback_inputs = vec![ArrayIrValue::Array(
            Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[1.0_f64, 2.0, 3.0]).unwrap(),
        )];
        pullback_inputs.extend(residuals);
        assert_eq!(
            pullback.interpret(pullback_inputs),
            Ok(vec![
                ArrayIrValue::Array(
                    Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [0]), &[]).unwrap()
                ),
                ArrayIrValue::Array(Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[6.0_f64]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_array_ir_pad_differentiation_dynamic_axis_after_static_axis() {
        // Explicit pad geometry retains one output extent per physical axis, including statically typed axes. Keep a
        // static leading axis to verify that the pullback selects dynamic constructor inputs from the right axis.
        let columns = DimensionVariable::new("columns", DimensionBounds::new(1, Some(5)).unwrap());
        let padded_columns = DimensionVariable::new("padded_columns", DimensionBounds::new(3, Some(7)).unwrap());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(columns)]));
        let padded_columns_type = DimensionType::new(padded_columns);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let padding_value = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let rows = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let output_extent = builder.add_input(padded_columns_type.clone().into());
        let output = builder
            .add_instruction(
                PadOperation::<ArrayType>::new(vec![0, 1], vec![0, 1], vec![0, 0]).unwrap(),
                Vec::new(),
                vec![input, padding_value, rows, output_extent],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(
                    Array::from_elements::<f64>(
                        ArrayType::new_static(DataType::F64, [2, 2]),
                        &[1.0_f64, 2.0, 3.0, 4.0],
                    )
                    .unwrap(),
                ),
                ArrayIrValue::Array(
                    Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[-1.0_f64]).unwrap(),
                ),
                ArrayIrValue::Dimension(DimensionValue::new(padded_columns_type, 4).unwrap()),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        let mut pullback_inputs = vec![ArrayIrValue::Array(
            Array::from_elements::<f64>(
                ArrayType::new_static(DataType::F64, [2, 4]),
                &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            )
            .unwrap(),
        )];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![
                ArrayIrValue::Array(
                    Array::from_elements::<f64>(
                        ArrayType::new_static(DataType::F64, [2, 2]),
                        &[2.0_f64, 3.0, 6.0, 7.0]
                    )
                    .unwrap()
                ),
                ArrayIrValue::Array(
                    Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[18.0_f64]).unwrap()
                ),
            ]),
        );
    }

    #[test]
    fn test_array_ir_pad_differentiation_disconnected_input_tangent() {
        // A structural-zero tangent has no runtime dimensions of its own. Read the input primal's extent so
        // materializing that tangent uses the same geometry rather than an input-free dynamic constructor.
        let source = DimensionVariable::new("source", DimensionBounds::new(1, Some(5)).unwrap());
        let result = DimensionVariable::new("result", DimensionBounds::new(3, Some(7)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(source.clone())]));
        let source_type = DimensionType::new(source);
        let result_type = DimensionType::new(result);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let source_extent = builder.add_input(source_type.clone().into());
        let padding_value = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let output_extent = builder.add_input(result_type.clone().into());
        // A mixed iota is a non-differentiable nullary constant, so its tangent is a structural zero of the input type
        // with symbolic extents while its primal is a non-zero exemplar and the padding-value tangent stays live. The
        // rule must still hand a concrete input tangent to the staged pad.
        let input = builder
            .add_instruction(
                ArrayIrOperation::<Array>::from(IotaOperation::new(input_type, 0).unwrap()),
                Vec::new(),
                vec![source_extent],
                None,
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(
                PadOperation::<ArrayType>::new(vec![1], vec![1], vec![0]).unwrap(),
                Vec::new(),
                vec![input, padding_value, output_extent],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::new(source_type, 2).unwrap()),
                ArrayIrValue::Array(
                    Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[-1.0_f64]).unwrap()
                ),
                ArrayIrValue::Dimension(DimensionValue::new(result_type, 4).unwrap()),
                ArrayIrValue::Array(Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[1.0_f64]).unwrap()),
            ]),
            Ok(vec![
                ArrayIrValue::Array(
                    Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [4]), &[-1.0_f64, 0.0, 1.0, -1.0])
                        .unwrap()
                ),
                ArrayIrValue::Array(
                    Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [4]), &[1.0_f64, 0.0, 0.0, 1.0])
                        .unwrap()
                ),
            ]),
        );
    }

    #[test]
    fn test_array_ir_pad_differentiation_restores_dynamic_layouts() {
        let size = DimensionVariable::new("size", DimensionBounds::new(1, Some(5)).unwrap());
        let output_size = DimensionVariable::new("output_size", DimensionBounds::new(3, Some(7)).unwrap());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![size.into()]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let padding_type = ArrayType::scalar(DataType::F32).with_layout(Layout::Strided(StridedLayout::new(vec![])));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.clone().into());
        let padding = builder.add_input(padding_type.clone().into());
        let output_dimension_type = DimensionType::new(output_size);
        let extent = builder.add_input(output_dimension_type.clone().into());
        let output = builder
            .add_instruction(
                PadOperation::<ArrayType>::new(vec![1], vec![1], vec![0]).unwrap(),
                Vec::new(),
                vec![input, padding, extent],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        let concrete_input_type = input_type.with_shape(Shape::new(vec![2.into()]));
        let primal_inputs = vec![
            ArrayIrValue::Array(Array::from_elements(concrete_input_type.clone(), &[1_f32, 2.]).unwrap()),
            ArrayIrValue::Array(Array::from_elements(padding_type.clone(), &[9_f32]).unwrap()),
            ArrayIrValue::Dimension(DimensionValue::new(output_dimension_type, 4).unwrap()),
        ];
        let mut primal_outputs = linearization.primal().interpret(primal_inputs.clone()).unwrap();
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = primal_inputs;
        tangent_inputs.pop();
        tangent_inputs.extend(residuals.clone());
        assert_eq!(linearization.tangent().interpret(tangent_inputs), Ok(primal_outputs));
        let mut pullback_inputs = vec![ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::F32, [4]), &[1_f32, 2., 3., 4.]).unwrap(),
        )];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![
                ArrayIrValue::Array(Array::from_elements(concrete_input_type, &[2_f32, 3.]).unwrap()),
                ArrayIrValue::Array(Array::from_elements(padding_type, &[5_f32]).unwrap()),
            ])
        );
    }

    #[test]
    fn test_array_ir_pad_differentiation_extreme_configuration() {
        // Interior padding is irrelevant when an axis is bounded to fewer than two input elements, so the pullback
        // skips the dilated-extent arithmetic and slices with stride one.
        let linearization = dynamic_pad_program(
            DimensionBounds::new(0, Some(2)).unwrap(),
            DimensionBounds::new(0, Some(2)).unwrap(),
            0,
            0,
            usize::MAX,
        )
        .linearize()
        .unwrap();
        let padding_value =
            ArrayIrValue::Array(Array::from_elements(ArrayType::scalar(DataType::F32), &[9_f32]).unwrap());
        let output_size =
            DimensionType::new(DimensionVariable::new("output_size", DimensionBounds::new(0, Some(2)).unwrap()));
        let mut outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(
                    Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0]), &[]).unwrap(),
                ),
                padding_value.clone(),
                ArrayIrValue::Dimension(DimensionValue::new(output_size.clone(), 0).unwrap()),
            ])
            .unwrap();
        let mut cotangents = vec![ArrayIrValue::Array(
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0]), &[]).unwrap(),
        )];
        cotangents.extend(outputs.split_off(1));
        assert_eq!(
            linearization.pullback().unwrap().interpret(cotangents),
            Ok(vec![
                ArrayIrValue::Array(
                    Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0]), &[]).unwrap()
                ),
                ArrayIrValue::Array(Array::from_elements(ArrayType::scalar(DataType::F32), &[0_f32]).unwrap()),
            ])
        );
        let mut outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[3_f32]).unwrap()),
                padding_value.clone(),
                ArrayIrValue::Dimension(DimensionValue::new(output_size.clone(), 1).unwrap()),
            ])
            .unwrap();
        let mut cotangents = vec![ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[7_f32]).unwrap(),
        )];
        cotangents.extend(outputs.split_off(1));
        assert_eq!(
            linearization.pullback().unwrap().interpret(cotangents),
            Ok(vec![
                ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[7_f32]).unwrap()),
                ArrayIrValue::Array(Array::from_elements(ArrayType::scalar(DataType::F32), &[0_f32]).unwrap()),
            ])
        );

        // Extreme edges crop every input position despite their balanced finite output shape, so the input cotangent
        // is zero and the whole output cotangent flows to the padding value.
        let linearization = dynamic_pad_program(
            DimensionBounds::new(1, Some(3)).unwrap(),
            DimensionBounds::new(0, Some(2)).unwrap(),
            i64::MIN,
            i64::MAX,
            0,
        )
        .linearize()
        .unwrap();
        let mut outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[3_f32]).unwrap()),
                padding_value.clone(),
                ArrayIrValue::Dimension(DimensionValue::new(output_size.clone(), 0).unwrap()),
            ])
            .unwrap();
        let mut cotangents = vec![ArrayIrValue::Array(
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0]), &[]).unwrap(),
        )];
        cotangents.extend(outputs.split_off(1));
        assert_eq!(
            linearization.pullback().unwrap().interpret(cotangents),
            Ok(vec![
                ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[0_f32]).unwrap()),
                ArrayIrValue::Array(Array::from_elements(ArrayType::scalar(DataType::F32), &[0_f32]).unwrap()),
            ])
        );
        let mut outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(
                    Array::from_elements(ArrayType::new_static(DataType::F32, [2]), &[3_f32, 3.]).unwrap(),
                ),
                padding_value.clone(),
                ArrayIrValue::Dimension(DimensionValue::new(output_size.clone(), 1).unwrap()),
            ])
            .unwrap();
        let mut cotangents = vec![ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[7_f32]).unwrap(),
        )];
        cotangents.extend(outputs.split_off(1));
        assert_eq!(
            linearization.pullback().unwrap().interpret(cotangents),
            Ok(vec![
                ArrayIrValue::Array(
                    Array::from_elements(ArrayType::new_static(DataType::F32, [2]), &[0_f32, 0.]).unwrap()
                ),
                ArrayIrValue::Array(Array::from_elements(ArrayType::scalar(DataType::F32), &[7_f32]).unwrap()),
            ])
        );
        let linearization = dynamic_pad_program(
            DimensionBounds::new(1, Some(3)).unwrap(),
            DimensionBounds::new(0, Some(2)).unwrap(),
            i64::MAX,
            i64::MIN,
            0,
        )
        .linearize()
        .unwrap();
        let mut outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[3_f32]).unwrap()),
                padding_value.clone(),
                ArrayIrValue::Dimension(DimensionValue::new(output_size.clone(), 0).unwrap()),
            ])
            .unwrap();
        let mut cotangents = vec![ArrayIrValue::Array(
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0]), &[]).unwrap(),
        )];
        cotangents.extend(outputs.split_off(1));
        assert_eq!(
            linearization.pullback().unwrap().interpret(cotangents),
            Ok(vec![
                ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[0_f32]).unwrap()),
                ArrayIrValue::Array(Array::from_elements(ArrayType::scalar(DataType::F32), &[0_f32]).unwrap()),
            ])
        );
        let mut outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(
                    Array::from_elements(ArrayType::new_static(DataType::F32, [2]), &[3_f32, 3.]).unwrap(),
                ),
                padding_value,
                ArrayIrValue::Dimension(DimensionValue::new(output_size, 1).unwrap()),
            ])
            .unwrap();
        let mut cotangents = vec![ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[7_f32]).unwrap(),
        )];
        cotangents.extend(outputs.split_off(1));
        assert_eq!(
            linearization.pullback().unwrap().interpret(cotangents),
            Ok(vec![
                ArrayIrValue::Array(
                    Array::from_elements(ArrayType::new_static(DataType::F32, [2]), &[0_f32, 0.]).unwrap()
                ),
                ArrayIrValue::Array(Array::from_elements(ArrayType::scalar(DataType::F32), &[7_f32]).unwrap()),
            ])
        );
    }

    #[test]
    fn test_array_ir_pad_differentiation_dilated_extents() {
        // An interior-padded dynamic axis that may hold more than two elements needs the dilated input extent
        // `n + max(n - 1, 0) * interior` at runtime, which the pullback derives from the retained input extent with
        // first-class dimension arithmetic before undoing the edge padding.
        let linearization = dynamic_pad_program(
            DimensionBounds::new(0, Some(5)).unwrap(),
            DimensionBounds::new(0, Some(9)).unwrap(),
            0,
            0,
            1,
        )
        .linearize()
        .unwrap();
        let pullback = linearization.pullback().unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[output_size], %1:dimension<output_size ∈ [0, 9)>, %2:dimension<size ∈ [0, 5)> .
                let %3:f32[size], %4:f32[] = linear_call [residual_count=2] %1 %2 %0 [
                    forward={
                        lambda %0:dimension<output_size ∈ [0, 9)>, %1:dimension<size ∈ [0, 5)>, %2:f32[output_size] .
                        let %3:f32[] = zero [type=f32[]]
                            %4:dimension<1> = constant [value=1]
                            %5:dimension<max(0, size - 1) ∈ [0, 4)> = dimension_saturating_sub %1 %4
                            %6:dimension<1> = constant [value=1]
                            %7:dimension<max(0, size - 1) * 1 ∈ [0, 4)> = dimension_mul %5 %6
                            %8:dimension<size + max(0, size - 1) * 1 ∈ [0, 8)> = dimension_add %1 %7
                            %9:f32[size + max(0, size - 1) * 1] = pad [edge_padding_low=[0], edge_padding_high=[0], \
                                interior_padding=[0]] %2 %3 %8
                            %10:dimension<0> = constant [value=0]
                            %11:f32[size] = dynamic_shape_slice [strides=[2]] %9 %10 %1
                            %12:bool[size] = zero [type=bool[size]] %1
                            %13:bool[] = one [type=bool[]]
                            %14:bool[output_size] = pad [edge_padding_low=[0], edge_padding_high=[0], \
                                interior_padding=[1]] %12 %13 %0
                            %15:f32[output_size] = zero [type=f32[output_size]] %0
                            %16:f32[output_size] = select %14 %2 %15
                            %17:f32[] = reduce_sum [axes=[0]] %16
                        in (%11, %17)
                    },
                    transpose={
                        lambda %0:dimension<output_size ∈ [0, 9)>, %1:dimension<size ∈ [0, 5)>, %2:f32[size], \
                            %3:f32[] .
                        let %4:f32[output_size] = pad [edge_padding_low=[0], edge_padding_high=[0], \
                            interior_padding=[1]] %2 %3 %0
                        in (%4)
                    },
                ]
                in (%3, %4)
            "}
            .trim_end(),
        );
        let padding_value =
            ArrayIrValue::Array(Array::from_elements(ArrayType::scalar(DataType::F32), &[9_f32]).unwrap());
        let output_size =
            DimensionType::new(DimensionVariable::new("output_size", DimensionBounds::new(0, Some(9)).unwrap()));

        // Three input elements dilate to five output positions, two of which hold the padding value.
        let mut outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(
                    Array::from_elements(ArrayType::new_static(DataType::F32, [3]), &[1_f32, 2., 3.]).unwrap(),
                ),
                padding_value.clone(),
                ArrayIrValue::Dimension(DimensionValue::new(output_size.clone(), 5).unwrap()),
            ])
            .unwrap();
        assert_eq!(
            outputs[0],
            ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::F32, [5]), &[1_f32, 9., 2., 9., 3.]).unwrap()
            ),
        );
        let mut cotangents = vec![ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::F32, [5]), &[1_f32, 2., 3., 4., 5.]).unwrap(),
        )];
        cotangents.extend(outputs.split_off(1));
        assert_eq!(
            pullback.interpret(cotangents),
            Ok(vec![
                ArrayIrValue::Array(
                    Array::from_elements(ArrayType::new_static(DataType::F32, [3]), &[1_f32, 3., 5.]).unwrap()
                ),
                ArrayIrValue::Array(Array::from_elements(ArrayType::scalar(DataType::F32), &[6_f32]).unwrap()),
            ])
        );

        // One element has no adjacent pair and an empty input has no elements, so neither dilates.
        let mut outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[1_f32]).unwrap()),
                padding_value.clone(),
                ArrayIrValue::Dimension(DimensionValue::new(output_size.clone(), 1).unwrap()),
            ])
            .unwrap();
        let mut cotangents = vec![ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[7_f32]).unwrap(),
        )];
        cotangents.extend(outputs.split_off(1));
        assert_eq!(
            pullback.interpret(cotangents),
            Ok(vec![
                ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[7_f32]).unwrap()),
                ArrayIrValue::Array(Array::from_elements(ArrayType::scalar(DataType::F32), &[0_f32]).unwrap()),
            ])
        );
        let mut outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(
                    Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0]), &[]).unwrap(),
                ),
                padding_value,
                ArrayIrValue::Dimension(DimensionValue::new(output_size, 0).unwrap()),
            ])
            .unwrap();
        let mut cotangents = vec![ArrayIrValue::Array(
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0]), &[]).unwrap(),
        )];
        cotangents.extend(outputs.split_off(1));
        assert_eq!(
            pullback.interpret(cotangents),
            Ok(vec![
                ArrayIrValue::Array(
                    Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0]), &[]).unwrap()
                ),
                ArrayIrValue::Array(Array::from_elements(ArrayType::scalar(DataType::F32), &[0_f32]).unwrap()),
            ])
        );
    }

    #[test]
    fn test_array_ir_pad_differentiation_preserves_runtime_assertion_proof() {
        // The inverse pad staged by the pullback recomputes its proof against the actual inverse signature. Static
        // axes and axes whose identity the padding leaves unchanged need no runtime assertion, so the inverse pad is
        // effect-free even though the forward pad's conservative conversion kept one.
        let columns = DimensionVariable::new("columns", DimensionBounds::new(1, Some(5)).unwrap());
        let columns_type = DimensionType::new(columns.clone());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(columns)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let padding_value = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let columns_extent = builder.add_input(columns_type.into());
        let rows = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap()));
        let output = builder
            .add_instruction(
                PadOperation::<ArrayType>::new(vec![1, 0], vec![1, 0], vec![0, 0]).unwrap(),
                Vec::new(),
                vec![input, padding_value, rows, columns_extent],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.instructions()[0].operation().effects().classes(),
            EffectClasses::single(EffectClass::OrderedAssertion),
        );
        let pullback = program.linearize().unwrap().pullback().unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[4, columns], %1:dimension<columns ∈ [1, 5)> .
                let %2:dimension<4> = const 4
                    %3:f64[2, columns], %4:f64[] = linear_call [residual_count=2] %2 %1 %0 [
                        forward={
                            lambda %0:dimension<4>, %1:dimension<columns ∈ [1, 5)>, %2:f64[4, columns] .
                            let %3:f64[] = zero [type=f64[]]
                                %4:dimension<2> = constant [value=2]
                                %5:f64[2, columns] = pad [edge_padding_low=[-1, 0], edge_padding_high=[-1, 0], \
                                    interior_padding=[0, 0]] %2 %3 %4 %1
                                %6:dimension<0> = constant [value=0]
                                %7:f64[2, columns] = dynamic_shape_slice [strides=[1, 1]] %5 %6 %6 %4 %1
                                %8:bool[2, columns] = zero [type=bool[2, columns]] %1
                                %9:bool[] = one [type=bool[]]
                                %10:bool[4, columns] = pad [edge_padding_low=[1, 0], edge_padding_high=[1, 0], \
                                    interior_padding=[0, 0]] %8 %9 %0 %1
                                %11:f64[4, columns] = zero [type=f64[4, columns]] %1
                                %12:f64[4, columns] = select %10 %2 %11
                                %13:f64[] = reduce_sum [axes=[0, 1]] %12
                            in (%7, %13)
                        },
                        transpose={
                            lambda %0:dimension<4>, %1:dimension<columns ∈ [1, 5)>, %2:f64[2, columns], %3:f64[] .
                            let %4:f64[4, columns] = pad [edge_padding_low=[1, 0], edge_padding_high=[1, 0], \
                                interior_padding=[0, 0]] %2 %3 %0 %1
                            in (%4)
                        },
                    ]
                in (%3, %4)
            "}
            .trim_end(),
        );
        // The inverse pad is the one with negated edge amounts; the mask pad and the replayed forward pad reuse the
        // forward operation and its conservative assertion.
        let inverse_pads = pullback
            .entry_region_ref()
            .instructions_in_closure()
            .filter_map(|(_, instruction)| match instruction.operation() {
                ArrayIrOperation::Pad(operation) if operation.edge_padding_low() == [-1, 0] => Some(operation),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(inverse_pads.len(), 1);
        assert!(!inverse_pads[0].requires_runtime_assertion());
        assert_eq!(inverse_pads[0].effects().classes(), EffectClasses::NONE);

        // Derived dilated extents are symbolic arithmetic that `with_input_types` cannot prove, so that inverse pad
        // legitimately keeps its assertion.
        let pullback = dynamic_pad_program(
            DimensionBounds::new(0, Some(5)).unwrap(),
            DimensionBounds::new(0, Some(9)).unwrap(),
            0,
            0,
            1,
        )
        .linearize()
        .unwrap()
        .pullback()
        .unwrap();
        let inverse_pads = pullback
            .entry_region_ref()
            .instructions_in_closure()
            .filter_map(|(_, instruction)| match instruction.operation() {
                ArrayIrOperation::Pad(operation) if operation.interior_padding() == [0] => Some(operation),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(inverse_pads.len(), 1);
        assert!(inverse_pads[0].requires_runtime_assertion());
        assert_eq!(inverse_pads[0].effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
    }

    #[test]
    fn test_array_ir_pad_differentiation_rejects_unrepresentable_inverse() {
        // The pullback negates the edge amounts to undo them, which `i64::MIN` cannot express. A finite input bound
        // would crop every position first, so an unbounded input reaches the negation. Linearization stages the
        // pullback eagerly, so the diagnostics surface from `linearize`.
        assert!(matches!(
            dynamic_pad_program(DimensionBounds::unbounded(), DimensionBounds::new(0, Some(5)).unwrap(), i64::MIN, 0, 0)
                .linearize(),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == format!(
                    "`{PAD_OPERATION_NAME}` transpose cannot negate `edge_padding_low` at axis 0 with value \
                     -9223372036854775808",
                ),
        ));
        assert!(matches!(
            dynamic_pad_program(DimensionBounds::unbounded(), DimensionBounds::new(0, Some(5)).unwrap(), 0, i64::MIN, 0)
                .linearize(),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == format!(
                    "`{PAD_OPERATION_NAME}` transpose cannot negate `edge_padding_high` at axis 0 with value \
                     -9223372036854775808",
                ),
        ));

        // The strided slice that recovers the input positions needs the stride `interior + 1` to fit `usize`, but
        // that overflow is unreachable: an interior amount that large is first staged as the dilated-extent constant,
        // which the dimension backend width rejects.
        assert_eq!(
            dynamic_pad_program(
                DimensionBounds::new(0, Some(5)).unwrap(),
                DimensionBounds::new(0, Some(2)).unwrap(),
                0,
                0,
                usize::MAX,
            )
            .linearize()
            .map(|_| ()),
            Err(DifferentiationError::Program(
                DimensionError::ExtentExceedsBackendWidth { value: usize::MAX, maximum: i64::MAX as usize }.into(),
            )),
        );
    }

    #[test]
    fn test_array_ir_pad_transposition() {
        // Static geometry delegates to the homogeneous pullback, so the pullback is the same strided slice, zero pad,
        // and masked sum. The explicit extent is a shape operand with no cotangent contribution.
        let eight = ArrayIrValue::Dimension(DimensionValue::constant(8).unwrap());
        let input_types = [
            ArrayIrType::Array(ArrayType::new_static(DataType::F64, [3])),
            ArrayIrType::Array(ArrayType::scalar(DataType::F64)),
            eight.r#type().into_owned(),
        ];
        let operation = PadOperation::<ArrayIrType>::new(vec![1], vec![2], vec![1]).unwrap();
        check_operation_transposition!(
            @exact,
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = operation.clone(),
            cases = [{
                inputs = [
                    (@linear(type = input_types[0].clone())),
                    (@linear(type = input_types[1].clone())),
                    (@known, eight.clone()),
                ],
                output_cotangents = [ArrayIrValue::Array(
                    Array::from_elements::<f64>(
                        ArrayType::new_static(DataType::F64, [8]),
                        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                    )
                    .unwrap(),
                )],
                input_cotangents = [
                    ArrayIrValue::Array(
                        Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[2.0, 4.0, 6.0])
                            .unwrap(),
                    ),
                    ArrayIrValue::Array(
                        Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[24.0]).unwrap(),
                    ),
                ],
                pullback = indoc! {"
                    lambda %0:f64[8], %1:dimension<8> .
                    let %2:f64[3] = slice [start_indices=[1], limit_indices=[6], strides=[2]] %0
                        %3:f64[] = zero [type=f64[]]
                        %4:f64[3] = pad [edge_padding_low=[0], edge_padding_high=[0], interior_padding=[0]] %2 %3
                        %5:bool[3] = zero [type=bool[3]]
                        %6:bool[] = one [type=bool[]]
                        %7:bool[8] = pad [edge_padding_low=[1], edge_padding_high=[2], interior_padding=[1]] %5 %6
                        %8:f64[8] = zero [type=f64[8]]
                        %9:f64[8] = select %7 %0 %8
                        %10:f64[] = reduce_sum [axes=[0]] %9
                    in (%4, %10)
                "},
            }],
        );

        // Under the direct rule the extent's accumulator is never touched, so it stays a structural zero once both
        // array cotangents have been accumulated.
        let output_type = operation.infer_output_types(&input_types, &[]).unwrap().remove(0);
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let output_cotangent = context.input(output_type.cotangent().unwrap());
        let mut transpose = TranspositionContext::new(context.clone());
        let inputs = input_types.iter().cloned().map(PartialValue::Unknown).collect::<Vec<_>>();
        let accumulators = transpose.cotangent_accumulators(&inputs, &[]).unwrap();
        operation
            .transpose(
                &mut transpose,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Value(output_cotangent)],
                &accumulators,
            )
            .unwrap();
        let cotangents = transpose.take_cotangents(&accumulators).unwrap();
        assert_eq!(cotangents.len(), 3);
        assert!(!cotangents[0].is_zero());
        assert!(!cotangents[1].is_zero());
        assert!(cotangents[2].is_zero());
        assert_eq!(cotangents[2].r#type().as_ref(), &input_types[2].cotangent().unwrap());

        // The rule requires at least the two array inputs.
        assert!(matches!(
            operation.transpose(
                &mut transpose,
                &EmptyRegionDriver,
                &[],
                &[MaybeZero::Zero(output_type.cotangent().unwrap())],
                &[],
            ),
            Err(DifferentiationError::Program(ProgramError::InvalidInputCount { expected: 2, actual: 0 })),
        ));
    }

    #[test]
    fn test_array_ir_pad_transposition_symbolic_zero() {
        // A structural-zero output cotangent contributes nothing: the rule returns before staging anything and leaves
        // every accumulator, including the extent's, at its structural-zero default.
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("size", DimensionBounds::new(1, Some(5)).unwrap()).into()]),
        );
        let output_dimension_type =
            DimensionType::new(DimensionVariable::new("output_size", DimensionBounds::new(3, Some(7)).unwrap()));
        let input_types =
            vec![input_type.into(), ArrayType::scalar(DataType::F32).into(), output_dimension_type.into()];
        let operation = PadOperation::<ArrayIrType>::new(vec![1], vec![1], vec![0]).unwrap();
        let output_type = operation.infer_output_types(&input_types, &[]).unwrap().remove(0);
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let mut transpose = TranspositionContext::new(context.clone());
        let inputs = input_types.iter().cloned().map(PartialValue::Unknown).collect::<Vec<_>>();
        let accumulators = transpose.cotangent_accumulators(&inputs, &[]).unwrap();
        operation
            .transpose(
                &mut transpose,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Zero(output_type.cotangent().unwrap())],
                &accumulators,
            )
            .unwrap();
        let cotangents = transpose.take_cotangents(&accumulators).unwrap();
        assert_eq!(cotangents.len(), 3);
        for (cotangent, input_type) in cotangents.iter().zip(&input_types) {
            assert!(cotangent.is_zero());
            assert_eq!(cotangent.r#type().as_ref(), &input_type.cotangent().unwrap());
        }
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_array_ir_pad_transposition_rejects_dynamic_extents() {
        // The direct rule delegates to the homogeneous pullback, which slices static geometry only. Dynamic geometry
        // is rejected by name; linearization is the supported route, because it retains the primal extents as
        // residuals.
        let program = dynamic_pad_program(
            DimensionBounds::new(0, Some(5)).unwrap(),
            DimensionBounds::new(3, Some(11)).unwrap(),
            1,
            2,
            1,
        );
        assert!(matches!(
            program.transpose_with_respect_to(&[0, 1], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == format!(
                    "direct `{PAD_OPERATION_NAME}` transposition with dynamic extents requires linearization so that \
                     the primal geometry can be retained as residuals",
                ),
        ));
        assert_eq!(
            program.linearize().unwrap().pullback().unwrap().output_types(),
            program.input_types()[..2].iter().map(|r#type| r#type.cotangent().unwrap()).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_dynamic_pad_dynamic_pad() {
        // Context-carrying mixed values stage one mixed pad through the blanket implementation, which proves what the
        // signature allows: a symbolic output extent keeps the runtime assertion.
        let size = DimensionVariable::new("size", DimensionBounds::new(1, Some(5)).unwrap());
        let output_size = DimensionVariable::new("output_size", DimensionBounds::new(3, Some(7)).unwrap());
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = context.input(ArrayType::new(DataType::F32, Shape::new(vec![size.into()])).into());
        let padding_value = context.input(ArrayType::scalar(DataType::F32).into());
        let extent = context.input(DimensionType::new(output_size.clone()).into());
        let output = input.dynamic_pad(&padding_value, std::slice::from_ref(&extent), &[1], &[1], &[0]).unwrap();
        assert_eq!(
            output.r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![output_size.into()])))
        );
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[size], %1:f32[], %2:dimension<output_size ∈ [3, 7)> .
                let %3:f32[output_size] = pad [edge_padding_low=[1], edge_padding_high=[1], interior_padding=[0]] \
                    %0 %1 %2
                in (%3)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.instructions()[0].operation().effects().classes(),
            EffectClasses::single(EffectClass::OrderedAssertion),
        );

        // A static signature proves the extent, so the staged pad is pure.
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = context.input(ArrayType::new_static(DataType::F32, [2]).into());
        let padding_value = context.input(ArrayType::scalar(DataType::F32).into());
        let extent = context.lift(ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap())).unwrap();
        let output = input.dynamic_pad(&padding_value, &[extent], &[1], &[1], &[0]).unwrap();
        assert_eq!(output.r#type().as_ref(), &ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4])));
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[] .
                let %2:dimension<4> = const 4
                    %3:f32[4] = pad [edge_padding_low=[1], edge_padding_high=[1], interior_padding=[0]] %0 %1 %2
                in (%3)
            "}
            .trim_end(),
        );
        assert_eq!(program.instructions()[0].operation().effects().classes(), EffectClasses::NONE);
    }

    #[test]
    fn test_array_ir_value_dynamic_pad() {
        let input =
            ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[1_i32, 2]).unwrap());
        let padding = ArrayIrValue::Array(Array::from_elements(ArrayType::scalar(DataType::I32), &[9_i32]).unwrap());
        let extent = ArrayIrValue::Dimension(DimensionValue::constant(5).unwrap());
        assert_eq!(
            input.dynamic_pad(&padding, &[extent], &[1], &[1], &[1]),
            Ok(ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::I32, [5]), &[9_i32, 1, 9, 2, 9]).unwrap()
            ))
        );
        assert_eq!(
            input.dynamic_pad(&padding, &[], &[1], &[1], &[1]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 })
        );
        let wrong_extent = ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap());
        assert_eq!(
            input.dynamic_pad(&padding, &[wrong_extent], &[1], &[1], &[1]),
            Err(ProgramError::InvalidArgument {
                message: format!(
                    "`{PAD_OPERATION_NAME}` output axis 0 has extent 5, but its explicit extent input is 4"
                ),
            })
        );
        assert_eq!(padding.dynamic_pad(&padding, &[], &[], &[], &[]), Ok(padding));
    }

    #[test]
    fn test_validate_padding_ragged_axes() {
        let ragged = RaggedAxis::new(
            1,
            Array::from_elements(ArrayType::new_static(DataType::I64, [2]), &[1_i64, 2]).unwrap(),
            DimensionVariable::new("size", DimensionBounds::new(0, Some(3)).unwrap()),
            vec![0],
        );
        assert_eq!(
            validate_padding_ragged_axes(std::slice::from_ref(&ragged), &[0, 0, 1], &[0, 0, 1], &[0, 0, 0]),
            Ok(vec![ragged.clone()])
        );
        assert!(matches!(
            validate_padding_ragged_axes(std::slice::from_ref(&ragged), &[1, 0, 0], &[0, 0, 0], &[0, 0, 0]),
            Err(BatchingError::Program(ProgramError::UnsupportedOperation { message }))
                if message == format!(
                    "`{PAD_OPERATION_NAME}` batching cannot change a ragged axis or an axis indexing its extents",
                ),
        ));
        // `ArrayBatch::with_ragged_axes` validates real inputs against the carrier rank, so an axis outside the padding
        // vectors is helper-level hardening rather than a reachable operation error.
        assert!(matches!(
            validate_padding_ragged_axes(&[ragged], &[0], &[0], &[0]),
            Err(BatchingError::InvalidBatchMetadata { message })
                if message == format!("`{PAD_OPERATION_NAME}` batching found ragged axis 1 outside the padded rank 1"),
        ));
    }

    #[test]
    fn test_is_effective_identity() {
        let static_type = ArrayType::new_static(DataType::F32, [3, 1, 0]);
        assert!(is_effective_identity(&static_type, &[0, 0, 0], &[0, 0, 0], &[0, 0, 0]));
        assert!(is_effective_identity(&ArrayType::scalar(DataType::F32), &[], &[], &[]));
        // Any edge amount moves or crops elements, even when the amounts balance.
        assert!(!is_effective_identity(&static_type, &[1, 0, 0], &[0, 0, 0], &[0, 0, 0]));
        assert!(!is_effective_identity(&static_type, &[-1, 0, 0], &[1, 0, 0], &[0, 0, 0]));
        assert!(!is_effective_identity(&static_type, &[0, 0, 0], &[0, 0, -1], &[0, 0, 0]));
        // Interior padding only matters on axes that can hold two adjacent elements.
        assert!(is_effective_identity(&static_type, &[0, 0, 0], &[0, 0, 0], &[0, usize::MAX, usize::MAX]));
        assert!(!is_effective_identity(&static_type, &[0, 0, 0], &[0, 0, 0], &[1, 0, 0]));
        let bounded_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                DimensionVariable::new("small", DimensionBounds::new(0, Some(2)).unwrap()).into(),
                DimensionVariable::new("large", DimensionBounds::new(0, Some(3)).unwrap()).into(),
                DimensionVariable::new("unbounded", DimensionBounds::unbounded()).into(),
            ]),
        );
        assert!(is_effective_identity(&bounded_type, &[0, 0, 0], &[0, 0, 0], &[1, 0, 0]));
        assert!(!is_effective_identity(&bounded_type, &[0, 0, 0], &[0, 0, 0], &[0, 1, 0]));
        assert!(!is_effective_identity(&bounded_type, &[0, 0, 0], &[0, 0, 0], &[0, 0, 1]));
    }

    #[test]
    fn test_padded_extent() {
        // `d + max(d - 1, 0) * interior + low + high` in `i128`, so negative results are representable here.
        assert_eq!(padded_extent(3, 1, 2, 1, 0), Ok(8));
        assert_eq!(padded_extent(0, 1, 2, 1, 0), Ok(3));
        assert_eq!(padded_extent(1, 0, 0, usize::MAX, 0), Ok(1));
        assert_eq!(padded_extent(1, -2, 0, 0, 0), Ok(-1));
        assert_eq!(padded_extent(2, i64::MIN, i64::MAX, 0, 0), Ok(1));
        // On 64-bit targets every `usize` fits `i128`, so only the dilation arithmetic can overflow: the gap product
        // itself, or the sum of the input size and a product just below `i128::MAX`.
        assert_eq!(
            padded_extent(usize::MAX, 0, 0, usize::MAX, 1),
            Err(TypeError::invalid(format!("`{PAD_OPERATION_NAME}` output size overflows `usize` on axis 1"))),
        );
        assert_eq!(
            padded_extent(i64::MAX as usize + 2, 0, 0, usize::MAX, 0),
            Err(TypeError::invalid(format!("`{PAD_OPERATION_NAME}` output size overflows `usize` on axis 0"))),
        );
    }

    #[test]
    fn test_static_padded_extent() {
        assert_eq!(static_padded_extent(3, 1, 2, 1, 0), Ok(8));
        assert_eq!(static_padded_extent(5, -1, -2, 0, 0), Ok(2));
        assert_eq!(static_padded_extent(0, 0, 0, usize::MAX, 0), Ok(0));
        assert_eq!(
            static_padded_extent(1, -2, 0, 0, 1),
            Err(TypeError::invalid(format!("`{PAD_OPERATION_NAME}` output size is negative (-1) on axis 1"))),
        );
        assert_eq!(
            static_padded_extent(usize::MAX, 1, 0, 0, 0),
            Err(TypeError::invalid(format!("`{PAD_OPERATION_NAME}` output size overflows `usize` on axis 0"))),
        );
        assert_eq!(
            static_padded_extent(2, 0, 0, usize::MAX, 0),
            Err(TypeError::invalid(format!("`{PAD_OPERATION_NAME}` output size overflows `usize` on axis 0"))),
        );
    }

    #[test]
    fn test_validate_pad_inputs() {
        let input = ArrayType::new_static(DataType::F32, [2, 3]);
        let padding_value = ArrayType::scalar(DataType::F32);
        assert_eq!(validate_pad_inputs(&input, &padding_value, &[0, 0], &[0, 0], &[0, 0]), Ok(()));
        assert_eq!(
            validate_pad_inputs(&input, &ArrayType::scalar(DataType::F64), &[0, 0], &[0, 0], &[0, 0]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` input data type `f32` does not match padding value data type `f64`"
            )))),
        );
        assert_eq!(
            validate_pad_inputs(&input, &ArrayType::new_static(DataType::F32, [1]), &[0, 0], &[0, 0], &[0, 0]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` padding value must be a scalar but has type `f32[1]`"
            )))),
        );
        assert_eq!(
            validate_pad_inputs(
                &input,
                &padding_value.clone().with_memory(Memory::Host { pinned: false }),
                &[0, 0],
                &[0, 0],
                &[0, 0],
            ),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` input and padding value must share one memory space but reside in `Device` \
                 and `Host[Unpinned]`"
            )))),
        );
        assert_eq!(
            validate_pad_inputs(&input, &padding_value, &[0], &[0, 0], &[0, 0]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` `edge_padding_low` has length 1 but input has rank 2"
            )))),
        );
        assert_eq!(
            validate_pad_inputs(&input, &padding_value, &[0, 0], &[0, 0, 0], &[0, 0]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` `edge_padding_high` has length 3 but input has rank 2"
            )))),
        );
        assert_eq!(
            validate_pad_inputs(&input, &padding_value, &[0, 0], &[0, 0], &[]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` `interior_padding` has length 0 but input has rank 2"
            )))),
        );
    }

    #[test]
    fn test_pad_output_type() {
        // The output keeps the input's data type and memory, clears its layout, and carries its sharding through the
        // resized dimensions.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let input = ArrayType::new_static(DataType::F32, [4])
            .with_layout(Layout::Strided(StridedLayout::new(vec![2])))
            .with_memory(Memory::Host { pinned: true });
        let padding_value = ArrayType::scalar(DataType::F32).with_memory(Memory::Host { pinned: true });
        assert_eq!(
            pad_output_type(&input, &padding_value, vec![Dimension::Static(6)], &[1], &[1], &[0]),
            Ok(ArrayType::new_static(DataType::F32, [6]).with_memory(Memory::Host { pinned: true })),
        );
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
            .unwrap()
            .with_unreduced_axes(["m"])
            .unwrap();
        let sharded_input = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap();
        let matching_padding_value = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.clone(), 0).with_unreduced_axes(["m"]).unwrap())
            .unwrap();
        assert_eq!(
            pad_output_type(&sharded_input, &matching_padding_value, vec![Dimension::Static(8)], &[0], &[4], &[0]),
            Ok(ArrayType::new_static(DataType::F32, [8]).with_sharding(sharding).unwrap()),
        );

        // Dependency metadata must agree only when a padding position can exist: a pure crop and interior padding of
        // a singleton axis never read the padding value.
        let plain_padding_value =
            ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
        assert!(matches!(
            pad_output_type(&sharded_input, &plain_padding_value, vec![Dimension::Static(6)], &[0], &[2], &[0]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == format!(
                    "`{PAD_OPERATION_NAME}` input and padding value must have matching reduced and unreduced mesh axes \
                     but got input type `{sharded_input}` and padding value type `{plain_padding_value}`",
                ),
        ));
        assert_eq!(
            pad_output_type(&sharded_input, &plain_padding_value, vec![Dimension::Static(2)], &[-1], &[-1], &[0])
                .map(|output| output.shape().clone()),
            Ok(Shape::new(vec![Dimension::Static(2)])),
        );
        let singleton_input = ArrayType::new_static(DataType::F32, [1])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["m"]).unwrap())
            .unwrap();
        assert_eq!(
            pad_output_type(&singleton_input, &plain_padding_value, vec![Dimension::Static(1)], &[0], &[0], &[3]),
            Ok(singleton_input.clone().with_layout(None)),
        );

        // Matching dependency metadata on different meshes cannot describe one distributed value.
        let other_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 4, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let other_mesh_padding_value = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(other_mesh, 0).with_unreduced_axes(["m"]).unwrap())
            .unwrap();
        assert_eq!(
            pad_output_type(&singleton_input, &other_mesh_padding_value, vec![Dimension::Static(2)], &[1], &[0], &[0]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` input and padding value with distributed dependencies must use the same mesh"
            )))),
        );
    }
}
