//! Edge and interior array padding, including explicit dynamic output extents.
//!
//! [`PadOperation`] supports cropping with negative edge padding and differentiates both the input and the scalar
//! padding value. Dynamic output extents are checked against the padding geometry before execution.

use std::borrow::Cow;
use std::fmt::Display;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::arrays::{
    Array, ArrayAddressing, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrBatch,
    ArrayIrBatchingPolicy, ArrayIrType, ArrayIrValue, ArrayType, ArrayTypeRefinements, DataType, Dimension,
    DimensionOperation, DimensionType, DimensionValue, LinearResiduals, RaggedAxis, Shape, Sharding,
    materialize_array_tangent,
};
use crate::axes::Axis;
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, ProjectedContext, StagingContext};
use crate::differentiation::{
    CotangentAccumulator, DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver,
    DifferentiationDual, DifferentiationError, DifferentiationPolicy, ElementwiseDerivativeAlignment,
    ResidualZeroProvider, TransposableOperation, TranspositionContext, TranspositionDriver,
    transpose_projected_operation,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_reference_dischargeable_operation};
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
use crate::operations::math::add::AddOperation;
use crate::operations::math::reduce::{ReduceOperation, ReductionKind};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    EffectClass, EffectClasses, Effects, MaybeZero, Operation, OperationFormatter, OperationProjection, ProgramError,
    ProjectedValue, RegionInterface, Type, TypeError, Typed, Value, ValueProjection,
};
use crate::tracing::{NestedTracingContext, Tracer, TracingContext};

// TODO(eaplatanios): Review this.

/// Canonical operation name for [`PadOperation`].
pub const PAD_OPERATION_NAME: &str = "pad";

/// [`Operation`] that expands its first input by adding edge and interior padding filled with its second (scalar)
/// input. Refer to the documentation of [`Pad`] for more information.
///
/// The type parameter selects the input contract without introducing a separate dynamic-padding operation:
///
///   - `PadOperation<ArrayType>` accepts the input and padding-value arrays. It is used in programs over homogeneous
///     arrays whose output extents are fully described by the inferred array type.
///   - `PadOperation<ArrayIrType>` additionally accepts one first-class dimension input for each output
///     axis. It is used in mixed array/dimension programs that must carry those logical result extents explicitly.
///
/// Live reverse-mode differentiation of symbolic input shapes uses the mixed form and linearization, which retains
/// the runtime extents needed to restore cropped input positions and build the padding-position mask. Homogeneous
/// direct transposition supports static geometry; symbolic-zero cotangents do not require runtime extents.
///
/// The padding amounts remain static configuration in both forms. This distinction is therefore unrelated to
/// StableHLO's `dynamic_pad`, whose padding amounts are runtime inputs. Converting between the two Ryft forms only
/// reparameterizes the operation family and moves the existing padding vectors without copying them.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PadOperation<T: Type> {
    /// Padding added before the first element of each input axis.
    edge_padding_low: Vec<i64>,

    /// Padding added after the last element of each input axis.
    edge_padding_high: Vec<i64>,

    /// Padding added between any two adjacent elements of each input axis.
    interior_padding: Vec<usize>,

    /// Whether the mixed signature needs an execution-time output extent assertion.
    requires_runtime_assertion: bool,

    /// Type universe that determines the operation's input contract.
    marker: PhantomData<fn() -> T>,
}

impl PadOperation<ArrayType> {
    /// Creates a new [`PadOperation`] with the provided edge and interior padding amounts. The three vectors must
    /// share one length (one entry per input axis); whether that shared length matches the input rank is validated
    /// during type inference, once an input type is known.
    pub fn new(
        edge_padding_low: Vec<i64>,
        edge_padding_high: Vec<i64>,
        interior_padding: Vec<usize>,
    ) -> Result<Self, ProgramError> {
        if edge_padding_low.len() != edge_padding_high.len() || edge_padding_low.len() != interior_padding.len() {
            return Err(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` expects `edge_padding_low`, `edge_padding_high`, and `interior_padding` to share one length but \
                    got lengths {}, {}, and {}",
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
            requires_runtime_assertion: false,
            marker: PhantomData,
        })
    }
}

impl PadOperation<ArrayIrType> {
    /// Validates an input signature and removes the assertion effect when types alone prove every output extent.
    /// A refined payload rejects subsequent signatures that would require a runtime assertion.
    pub fn with_input_types(mut self, input_types: &[ArrayIrType]) -> Result<Self, TypeError> {
        self.requires_runtime_assertion = true;
        self.infer_output_types(input_types, &[])?;
        self.requires_runtime_assertion = !self.has_proven_output_extents(input_types)?;
        Ok(self)
    }

    /// Returns whether execution must validate the supplied output extents against the padded input geometry.
    pub fn requires_runtime_assertion(&self) -> bool {
        self.requires_runtime_assertion
    }

    /// Checks extent equality from an already validated signature without evaluating dimension inputs.
    fn has_proven_output_extents(&self, input_types: &[ArrayIrType]) -> Result<bool, TypeError> {
        let input = <&ArrayType>::try_from(&input_types[0])?;
        let output_dimensions = ArrayIrType::extents(&input_types[2..])?;
        for (axis, (input_dimension, output_dimension)) in
            input.shape().dimensions().iter().zip(output_dimensions).enumerate()
        {
            let identity = self.edge_padding_low[axis] as i128 + self.edge_padding_high[axis] as i128 == 0
                && (self.interior_padding[axis] == 0
                    || input_dimension.bounds().upper().is_some_and(|upper| upper <= 2));
            if identity && *input_dimension == output_dimension {
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

impl<T: Type> PadOperation<T> {
    /// Returns the padding added before the first element of each input axis.
    #[inline]
    pub fn edge_padding_low(&self) -> &[i64] {
        self.edge_padding_low.as_slice()
    }

    /// Returns the padding added after the last element of each input axis.
    #[inline]
    pub fn edge_padding_high(&self) -> &[i64] {
        self.edge_padding_high.as_slice()
    }

    /// Returns the padding added between any two adjacent elements of each input axis.
    #[inline]
    pub fn interior_padding(&self) -> &[usize] {
        self.interior_padding.as_slice()
    }

    /// Renders this payload independently of its homogeneous or composite operation contract.
    fn render_operation(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, PAD_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("edge_padding_low", format_args!("{:?}", self.edge_padding_low))?;
            operation.field("edge_padding_high", format_args!("{:?}", self.edge_padding_high))?;
            operation.field("interior_padding", format_args!("{:?}", self.interior_padding))
        })
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
            requires_runtime_assertion: false,
            marker: PhantomData,
        }
    }
}

impl<T: Type> Display for PadOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render_operation(formatter, 0)
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
        self.render_operation(formatter, indentation)
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
                "`{PAD_OPERATION_NAME}` expects an operand, a padding value, and one output extent per axis \
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
        if !self.requires_runtime_assertion && !self.has_proven_output_extents(input_types)? {
            return Err(TypeError::invalid(
                "`pad` was constructed without a runtime extent check but these input types require one",
            ));
        }
        let output_dimensions = ArrayIrType::extents(&input_types[2..])?;

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
                    "`{PAD_OPERATION_NAME}` output bounds {output_bounds} on axis {axis} cannot contain a padded extent derived from input bounds {input_bounds}",
                )));
            }
        }
        padded_output_type(
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
        self.render_operation(formatter, indentation)
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
// A batched input with a replicated padding value keeps its batch axis by padding it with zero amounts: the
// lifted operation inserts `0` into all three padding vectors at the batch axis position. A batch-varying (batched)
// padding value is vectorized with a constant-size mask construction: pad the input with a representable placeholder, pad an all-true
// input mask with false, broadcast the per-item padding values over the padded result, and select those values at
// padding positions.
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
        // Reject unsupported ragged geometry before ordinary shape inference, which cannot express a changed
        // ragged extent. The configuration is indexed physically after inserting the mapped batch axis.
        validate_pad_inputs(
            &inputs[0].unbatched_type(),
            &inputs[1].unbatched_type(),
            self.edge_padding_low(),
            self.edge_padding_high(),
            self.interior_padding(),
        )?;
        let mut physical_low = self.edge_padding_low().to_vec();
        let mut physical_high = self.edge_padding_high().to_vec();
        let mut physical_interior = self.interior_padding().to_vec();
        if let Some(axis) = inputs[0].batch_axis_position() {
            physical_low.insert(axis, 0);
            physical_high.insert(axis, 0);
            physical_interior.insert(axis, 0);
        }
        validate_padding_ragged_axes(inputs[0].ragged_axes(), &physical_low, &physical_high, &physical_interior)?;
        self.infer_output_types(&inputs.iter().map(ArrayBatch::unbatched_type).collect::<Vec<_>>(), &[])?;
        if inputs[1].batch_axis_position().is_none() {
            let Some(batch_axis) = inputs[0].batch_axis_position() else {
                let ragged_axes = validate_padding_ragged_axes(
                    inputs[0].ragged_axes(),
                    self.edge_padding_low(),
                    self.edge_padding_high(),
                    self.interior_padding(),
                )?;
                let mut outputs = self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()])?;
                return Ok(vec![outputs.remove(0).with_ragged_axes(ragged_axes)?].into());
            };
            let mut edge_padding_low = self.edge_padding_low().to_vec();
            edge_padding_low.insert(batch_axis, 0);
            let mut edge_padding_high = self.edge_padding_high().to_vec();
            edge_padding_high.insert(batch_axis, 0);
            let mut interior_padding = self.interior_padding().to_vec();
            interior_padding.insert(batch_axis, 0);
            let lifted = PadOperation::new(edge_padding_low, edge_padding_high, interior_padding)?;
            let ragged_axes = validate_padding_ragged_axes(
                inputs[0].ragged_axes(),
                lifted.edge_padding_low(),
                lifted.edge_padding_high(),
                lifted.interior_padding(),
            )?;
            let mut outputs =
                lifted.interpret_with_batch_axes(context, inputs, &[BatchAxis::from_position(batch_axis)])?;
            return Ok(vec![outputs.remove(0).with_ragged_axes(ragged_axes)?].into());
        }
        let batch_axis = inputs[0].batch_axis_position().unwrap_or(0);
        let operand = P::match_axis(context, &inputs[0], Axis::from(batch_axis))?;
        let mut edge_padding_low = self.edge_padding_low().to_vec();
        edge_padding_low.insert(batch_axis, 0);
        let mut edge_padding_high = self.edge_padding_high().to_vec();
        edge_padding_high.insert(batch_axis, 0);
        let mut interior_padding = self.interior_padding().to_vec();
        interior_padding.insert(batch_axis, 0);

        let ragged_axes = validate_padding_ragged_axes(
            operand.ragged_axes(),
            &edge_padding_low,
            &edge_padding_high,
            &interior_padding,
        )?;
        let padding_type = inputs[1].unbatched_type();
        let placeholder_padding = context.parent().one(&padding_type)?;
        let padded = operand.value().pad(
            &placeholder_padding,
            edge_padding_low.as_slice(),
            edge_padding_high.as_slice(),
            interior_padding.as_slice(),
        )?;
        let mask_input_type = operand.r#type().into_owned().with_data_type(DataType::Boolean).with_layout(None);
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

// Batching rule for mixed [`PadOperation<ArrayIrType>`] instructions. Explicit result extents remain
// replicated. When the scalar padding value varies across the batch, the rule pads with a representable placeholder and uses a padded mask
// to select the broadcast per-item padding value without changing `pad`'s scalar operand contract.
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
        let [operand, padding_value] = array_inputs else {
            unreachable!();
        };
        <&ArrayType>::try_from(&operand.unbatched_type())?;
        <&ArrayType>::try_from(&padding_value.unbatched_type())?;
        for extent in output_extents {
            extent.validate_replicated_dimension()?;
        }
        let operand_batch = ArrayBatch::new(
            <C::Value as ValueProjection<ArrayType>>::into_projected(operand.value().clone())?,
            operand.batch_axis(),
        )?;
        let padding_value_batch = ArrayBatch::new(
            <C::Value as ValueProjection<ArrayType>>::into_projected(padding_value.value().clone())?,
            padding_value.batch_axis(),
        )?;
        let Some(batch_axis) = operand_batch
            .batch_axis_position()
            .or(Some(0).filter(|_| !padding_value_batch.batch_axis().is_replicated()))
        else {
            let ragged_axes = validate_padding_ragged_axes(
                operand.ragged_axes(),
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

        let operand_batch = driver.align_batch_axis(context, operand.clone(), Axis::from(batch_axis))?;
        let ragged_axes = operand_batch.ragged_axes().to_vec();
        let operand_batch = ArrayBatch::new(
            <C::Value as ValueProjection<ArrayType>>::into_projected(operand_batch.into_value())?,
            BatchAxis::from_position(batch_axis),
        )?;
        let mut edge_padding_low = self.edge_padding_low().to_vec();
        edge_padding_low.insert(batch_axis, 0);
        let mut edge_padding_high = self.edge_padding_high().to_vec();
        edge_padding_high.insert(batch_axis, 0);
        let mut interior_padding = self.interior_padding().to_vec();
        interior_padding.insert(batch_axis, 0);
        let ragged_axes =
            validate_padding_ragged_axes(&ragged_axes, &edge_padding_low, &edge_padding_high, &interior_padding)?;
        let operation = PadOperation::<ArrayIrType>::from(PadOperation::new(
            edge_padding_low,
            edge_padding_high,
            interior_padding,
        )?);
        let mut lifted_output_extents = Vec::with_capacity(output_extents.len() + 1);
        lifted_output_extents.extend(output_extents[..batch_axis].iter().map(|extent| extent.value().clone()));
        lifted_output_extents.push(context.axis_extent().clone());
        lifted_output_extents.extend(output_extents[batch_axis..].iter().map(|extent| extent.value().clone()));

        if padding_value_batch.batch_axis().is_replicated() {
            let mut lifted_inputs = Vec::with_capacity(lifted_output_extents.len() + 2);
            lifted_inputs.push(<C::Value as ValueProjection<ArrayType>>::from_projected(operand_batch.into_value()));
            lifted_inputs.push(padding_value.value().clone());
            lifted_inputs.extend(lifted_output_extents);
            let mut outputs = context.parent().bind(operation, Vec::new(), lifted_inputs.as_slice())?;
            check_count!("output", outputs, 1, ProgramError);
            return Ok(vec![
                ArrayIrBatch::new(outputs.remove(0), BatchAxis::from_position(batch_axis))?
                    .with_ragged_axes(ragged_axes)?,
            ]
            .into());
        }

        // `pad` requires a scalar padding input. Pad with a representable placeholder, build a Boolean mask for its
        // original positions, broadcast the mapped padding values across the result, and select them only outside
        // those positions.
        let array_context = ProjectedContext::<C, ArrayType>::new(context.parent().clone());
        let padding_scalar_type = padding_value_batch.unbatched_type();
        let placeholder_padding =
            <C::Value as ValueProjection<ArrayType>>::from_projected(array_context.one(&padding_scalar_type)?);
        let operand = <C::Value as ValueProjection<ArrayType>>::from_projected(operand_batch.into_value());
        let mut padded_inputs = Vec::with_capacity(lifted_output_extents.len() + 2);
        padded_inputs.push(operand.clone());
        padded_inputs.push(placeholder_padding);
        padded_inputs.extend(lifted_output_extents.iter().cloned());
        let mut padded = context.parent().bind(operation.clone(), Vec::new(), padded_inputs.as_slice())?;
        check_count!("output", padded, 1, ProgramError);
        let padded = padded.remove(0);

        let operand_type = <&ArrayType>::try_from(operand.r#type().as_ref())?
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
                    if axis == batch_axis {
                        Ok(context.axis_extent().clone())
                    } else {
                        Ok(operand.dimension_size(axis)?)
                    }
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

// Forward-mode rule for [`PadOperation`]: `pad` is linear in both the operand and the padding value, so the
// tangent pads the operand tangent with the padding-value tangent using the same padding amounts.
impl<C: Context<Type = ArrayType> + Zero<C::Value>> DifferentiableOperation<C> for PadOperation<ArrayType>
where
    C::Operation: From<PadOperation<ArrayType>>,
    C::Value: Pad,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        let primal = inputs[0].primal().pad(
            inputs[1].primal(),
            self.edge_padding_low(),
            self.edge_padding_high(),
            self.interior_padding(),
        )?;
        // The pad needs both the operand and padding-value tangents as real values, so materialize the structurally
        // zero side (the shared all-zero fast path already handled the case where both are zero).
        let operand_tangent = inputs[0].tangent().clone().materialize(context.tangent())?;
        let padding_tangent = inputs[1].tangent().clone().materialize(context.tangent())?;
        let tangent = operand_tangent.pad(
            &padding_tangent,
            self.edge_padding_low(),
            self.edge_padding_high(),
            self.interior_padding(),
        )?;
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

// Forward-mode rule for mixed pad. The explicit output extents are ordinary non-differentiated shape values. Exact
// operand geometry replays the mixed pad directly; dynamic geometry retains the exact operand shape and output
// extents so the linear transpose can reconstruct both the operand and padding-value cotangents.
impl<C> DifferentiableOperation<C> for PadOperation<ArrayIrType>
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
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let destinations = context;
        let context = destinations.primal();
        if inputs.len() < 2 {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
        }
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let mut primal = context.bind(self.clone(), Vec::new(), primal_inputs.as_slice())?;
        check_count!("output", primal, 1, ProgramError);
        let primal = primal.remove(0);
        let output_primal = primal;
        let primal = destinations.primal_to_tangent(output_primal.clone())?;
        let tangent_inputs = destinations.dual_primal_to_tangent(inputs)?;
        let inputs = tangent_inputs.as_slice();
        let (array_inputs, output_extents) = inputs.split_at(2);
        let context = destinations.tangent();
        let tangent = if array_inputs.iter().all(|input| input.tangent().is_zero()) {
            MaybeZero::Zero(primal.r#type().tangent()?)
        } else {
            let projected_context = ProjectedContext::<C, ArrayType>::new(context.clone());
            let mut tangent_inputs = array_inputs
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
                tangent_inputs.extend(output_extents.iter().map(|extent| extent.primal().clone()));
                {
                    let mut outputs = context.bind(self.clone(), Vec::new(), tangent_inputs.as_slice())?;
                    check_count!("output", outputs, 1, ProgramError);
                    MaybeZero::Value(outputs.remove(0))
                }
            } else {
                let mut residuals = LinearResiduals::new();
                let output_extents = residuals.retain_all(output_extents.iter().map(|extent| extent.primal().clone()));
                let operand_shape = residuals.retain_shape(context, array_inputs[0].primal())?;
                let forward_operation = self.clone();
                let forward_output_extents = output_extents.clone();
                let transpose_operation = self.clone();
                let transpose_operand_type = operand_cotangent_type.clone();
                let transpose_padding_type =
                    <&ArrayType>::try_from(array_inputs[1].primal().r#type().as_ref())?.cotangent()?;
                let transpose_output_type = <&ArrayType>::try_from(primal.r#type().as_ref())?.cotangent()?;
                let mut tangent = LinearCallOperation::stage(
                    context,
                    residuals.into_values(),
                    tangent_inputs,
                    move |residuals, linear_inputs| {
                        check_count!("input", linear_inputs, 2, ProgramError);
                        let mut pad_inputs = linear_inputs.to_vec();
                        pad_inputs.extend(forward_output_extents.iter().map(|index| residuals[*index].clone()));
                        linear_inputs[0].dispatch_domain().bind(forward_operation, Vec::new(), pad_inputs.as_slice())
                    },
                    move |residuals, output_cotangents| {
                        check_count!("output", output_cotangents, 1, ProgramError);
                        let transpose_context = output_cotangents[0].dispatch_domain();
                        let output_cotangent = output_cotangents[0].clone();
                        let input_extents = operand_shape.dimensions(&transpose_context, residuals)?;

                        let all_cropped =
                            transpose_operand_type.shape().dimensions().iter().enumerate().any(|(axis, dimension)| {
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
                            });
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
                                let input_type = <&DimensionType>::try_from(input_extent.r#type().as_ref())?.clone();
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
                                let mut interior = transpose_context.bind(
                                    DimensionOperation::from(ConstantOperation::new(DimensionValue::constant(
                                        interior,
                                    )?)),
                                    Vec::new(),
                                    &[],
                                )?;
                                check_count!("output", interior, 1, ProgramError);
                                let interior = interior.remove(0);
                                let less_one_type = <&DimensionType>::try_from(less_one.r#type().as_ref())?.clone();
                                let interior_type = <&DimensionType>::try_from(interior.r#type().as_ref())?.clone();
                                let mut gaps = transpose_context.bind(
                                    DimensionOperation::Mul(DimensionMulOperation::new(
                                        &less_one_type,
                                        &interior_type,
                                    )?),
                                    Vec::new(),
                                    &[less_one, interior],
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
                                            "`{PAD_OPERATION_NAME}` transpose cannot negate `edge_padding_low` at axis \
                                         {axis} with value {padding}",
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
                                        "`{PAD_OPERATION_NAME}` transpose cannot negate `edge_padding_high` at axis \
                                         {axis} with value {padding}",
                                    ))
                                })
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let mut zero = transpose_context.bind(
                                <C::Operation as OperationProjection<ArrayType>>::Projected::from(ZeroOperation::new(
                                    transpose_padding_type.clone(),
                                )),
                                Vec::new(),
                                &[],
                            )?;
                            check_count!("output", zero, 1, ProgramError);
                            let zero = zero.remove(0);
                            let mut inverse_inputs = vec![output_cotangent.clone(), zero];
                            inverse_inputs.extend(dilated_extents);
                            let inverse_operation = PadOperation::<ArrayIrType>::from(PadOperation::<ArrayType>::new(
                                inverse_low,
                                inverse_high,
                                vec![0; transpose_operand_type.rank()],
                            )?);
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
                                            "`{PAD_OPERATION_NAME}` transpose stride overflows usize on axis {axis}",
                                        ))
                                    })
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let mut input_cotangent = transpose_context.bind(
                                DynamicShapeSliceOperation::new(transpose_operand_type.rank()).with_strides(strides)?,
                                Vec::new(),
                                slice_inputs.as_slice(),
                            )?;
                            check_count!("output", input_cotangent, 1, ProgramError);
                            input_cotangent.remove(0)
                        };

                        // Select padding positions before summing so non-finite cotangents at operand positions cannot
                        // contaminate the padding-value contribution.
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
                            <C::Operation as OperationProjection<ArrayType>>::Projected::from(SelectOperation::new()),
                            Vec::new(),
                            &[mask, output_cotangent, output_zero],
                        )?;
                        check_count!("output", selected, 1, ProgramError);
                        let selected = selected.remove(0);
                        let mut padding_cotangent = transpose_context.bind(
                            <C::Operation as OperationProjection<ArrayType>>::Projected::from(ReduceOperation::new(
                                (0..transpose_output_type.rank()).collect(),
                                ReductionKind::Sum,
                            )),
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
}

// Transpose (vector-Jacobian product) for a [`PadOperation`].
//
// The forward map `(t, p) ↦ pad(t, p, low, high, interior)` writes input element `i` to output position
// `low + i * (interior + 1)` along each axis and the padding value everywhere else, so its pullback splits the
// output cotangent into two contributions:
//
//   - **Input cotangent**: slice the surviving input positions with stride `interior + 1`, then insert zeros
//     at the cropped input positions.
//   - **Padding-value cotangent**: pad an all-false input-shaped mask with `true`, select the output cotangent only
//     at those padding positions, and sum the selected tensor. Selection rather than subtraction keeps non-finite
//     cotangents at input positions from contaminating this contribution.
//
// Symbolic-zero cotangents propagate unchanged.
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for PadOperation<ArrayType>
where
    O: Operation<Type = ArrayType>
        + From<AddOperation<ArrayType>>
        + From<OneOperation<ArrayType>>
        + From<PadOperation<ArrayType>>
        + From<SelectOperation<ArrayType>>
        + From<SliceOperation>
        + From<ReduceOperation>
        + From<ZeroOperation<ArrayType>>,
    Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<ArrayType>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        let contributions = {
            // The rule stages into the tracing context only, so the transposition context is narrowed once up front.
            let context: &mut TracingContext<V, O> = context;
            check_count!("input", inputs, 2, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 2, DifferentiationError);
            self.infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(), &[])?;
            match &outputs[0] {
                MaybeZero::Zero(_) => vec![
                    MaybeZero::Zero(inputs[0].r#type().cotangent()?),
                    MaybeZero::Zero(inputs[1].r#type().cotangent()?),
                ],
                MaybeZero::Value(cotangent) => {
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
                                    "`{PAD_OPERATION_NAME}` transpose requires a static output extent on axis {axis}"
                                ))
                            })? as i128;
                            let edge = self.edge_padding_low[axis] as i128;
                            let stride = self.interior_padding[axis] as i128 + 1;
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
                            starts.push((edge + first * stride) as usize);
                            limits.push((edge + (end - 1) * stride + 1) as usize);
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
                            let zero = MaybeZero::Zero(dependency_scalar_type(cotangent.r#type().as_ref())?)
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
                        let mut mask_padding =
                            context.stage_operation(OneOperation::new(mask_padding_type), Vec::new(), &no_inputs)?;
                        check_count!("output", mask_padding, 1, ProgramError);
                        let mut mask =
                            context.stage_operation(self.clone(), Vec::new(), &[mask_input, mask_padding.remove(0)])?;
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
                }
            }
        };
        check_count!("input", contributions, accumulators.len(), ProgramError);
        for (accumulator, contribution) in accumulators.iter().zip(contributions) {
            accumulator.accumulate(context, contribution)?;
        }
        Ok(())
    }
}

// Direct transposition rule for mixed pad. Static operand and output geometry delegate to the homogeneous array
// pullback, while every explicit output extent receives a structural-zero cotangent. Dynamic geometry requires
// linearization so [`DifferentiableOperation::jvp`] can retain the exact primal extents as residuals.
impl<V, O> TransposableOperation<V, O> for PadOperation<ArrayIrType>
where
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    O: Operation<Type = ArrayIrType> + From<AddOperation<ArrayIrType>> + OperationProjection<ArrayType>,
    <O as OperationProjection<ArrayType>>::Projected: From<PadOperation<ArrayType>>
        + TransposableOperation<
            <V as ValueProjection<ArrayType>>::Projected,
            <O as OperationProjection<ArrayType>>::Projected,
        >,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        check_count!("output", outputs, 1, ProgramError);
        check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);

        if inputs.len() < 2 {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
        }
        self.infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(), &[])?;
        if outputs[0].is_zero() {
            for (input, accumulator) in inputs.iter().zip(accumulators) {
                accumulator.accumulate(context, MaybeZero::Zero(input.r#type().cotangent()?))?;
            }
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
                    "direct `{PAD_OPERATION_NAME}` transposition with dynamic extents requires linearization so that \
                     the primal geometry can be retained as residuals",
                ),
            }
            .into());
        }

        let operation =
            <O as OperationProjection<ArrayType>>::Projected::from(PadOperation::<ArrayType>::from(self.clone()));
        transpose_projected_operation(context, &operation, array_inputs, outputs, &accumulators[..2])?;
        for (extent, accumulator) in output_extents.iter().zip(&accumulators[2..]) {
            accumulator.accumulate(context, MaybeZero::Zero(extent.r#type().cotangent()?))?;
        }
        Ok(())
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
            if edge_padding_low[axis] != 0 || edge_padding_high[axis] != 0 || interior_padding[axis] != 0 {
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

/// Represents the ability to add edge and interior padding filled with a scalar value. Negative edge padding crops
/// the input after interior padding has been inserted. Along each axis, input coordinate `i` moves to
/// `edge_padding_low + i * (interior_padding + 1)`; coordinates outside the output are discarded, and every remaining
/// output position holds `padding_value`.
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
                    // This axis retains its extent, even if another axis is padded or balanced edge padding moves
                    // its elements. Keeping the existing identity requires no new runtime dimension computation.
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
                                "`{PAD_OPERATION_NAME}` output size is negative ({maximum_output_extent}) on dynamic axis {axis} \
                                 even at its maximum input extent {maximum_input_extent}",
                            ))
                            .into());
                        }
                    }
                    return Err(TypeError::invalid(format!(
                        "`{PAD_OPERATION_NAME}` dynamic axis {axis} requires an explicit result-dimension operand",
                    ))
                    .into());
                }
            };
            output_dimensions.push(output_dimension);
        }
        padded_output_type(
            self,
            padding_value,
            output_dimensions,
            edge_padding_low,
            edge_padding_high,
            interior_padding,
        )
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
        let input_shape = self.r#type().static_shape().unwrap();
        if edge_padding_low.iter().all(|padding| *padding == 0)
            && edge_padding_high.iter().all(|padding| *padding == 0)
            && input_shape
                .dimensions()
                .iter()
                .zip(interior_padding)
                .all(|(size, padding)| *padding == 0 || *size <= 1)
        {
            return Ok(self.clone());
        }
        let output_shape = output_type.static_shape().unwrap();
        let rank = input_shape.rank();
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
        if output_addressing.is_dense_row_major() && output_addressing.element_byte_width() != 0 {
            for output_bytes in bytes.chunks_exact_mut(output_addressing.element_byte_width()) {
                output_bytes.copy_from_slice(padding_bytes);
            }
        } else {
            for output_index in 0..output_addressing.element_count() {
                bytes[output_addressing.byte_range_for_flat_index(output_index)].copy_from_slice(padding_bytes);
            }
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
/// The following example pads a mixed array value using a first-class output extent. Context-carrying mixed values
/// use the same function to stage [`PadOperation<ArrayIrType>`]:
///
/// ```rust
/// # use ryft_core::{Array, ArrayIrValue, ArrayType, DataType, DimensionValue, DynamicPad, ProgramError};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// let input = ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[1i32, 2])?);
/// let value = ArrayIrValue::Array(Array::from_elements(ArrayType::scalar(DataType::I32), &[9i32])?);
/// let dimension = ArrayIrValue::Dimension(DimensionValue::constant(5)?);
/// let output = input.dynamic_pad(&value, &[dimension], &[1], &[1], &[1])?;
/// let expected = ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::I32, [5]), &[9i32, 1, 9, 2, 9])?);
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
    ///   - `edge_padding_low`: Signed edge padding before each input axis; negative amounts crop after interior padding.
    ///   - `edge_padding_high`: Signed edge padding after each input axis; negative amounts crop after interior padding.
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
                         extent operand is {}",
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
        let operation = PadOperation::<ArrayIrType>::from(PadOperation::new(
            edge_padding_low.to_vec(),
            edge_padding_high.to_vec(),
            interior_padding.to_vec(),
        )?);
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

/// Constructs a scalar type on the same mesh with the same non-dimensional dependency metadata as `source`. This is
/// the type a padding value must have to pad a `source`-typed operand, so rules that stage their own `pad` build
/// their padding constant against it.
pub(crate) fn dependency_scalar_type(source: &ArrayType) -> Result<ArrayType, TypeError> {
    let sharding = source
        .sharding()
        .map(|sharding| {
            Sharding::replicated(sharding.mesh().clone(), 0)
                .with_unreduced_axes(sharding.unreduced_axes().clone())
                .and_then(|output| output.with_reduced_axes(sharding.reduced_axes().clone()))
                .and_then(|output| output.with_varying_manual_axes(sharding.varying_manual_axes().clone()))
                .map_err(|error| {
                    TypeError::invalid(format!(
                        "`{PAD_OPERATION_NAME}` dependency scalar sharding construction failed: {error}"
                    ))
                })
        })
        .transpose()?;
    ArrayType::scalar(source.data_type())
        .with_memory(source.memory())
        .with_sharding(sharding)
        .map_err(|error| TypeError::invalid(error.to_string()))
}

/// Computes one concrete padded extent in a wide signed representation.
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

/// Computes one concrete padded extent and validates that it is representable by [`Dimension::Static`].
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
            "`{PAD_OPERATION_NAME}` output size is negative ({output_size}) on axis {axis}"
        )));
    }
    usize::try_from(output_size)
        .map_err(|_| TypeError::invalid(format!("`{PAD_OPERATION_NAME}` output size overflows `usize` on axis {axis}")))
}

/// Validates the operand types and padding-vector arity shared by both padding type contracts.
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

/// Builds the padded array type from already validated result dimensions while preserving array metadata semantics.
fn padded_output_type(
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
                "`{PAD_OPERATION_NAME}` input and padding value must have matching reduced and unreduced mesh axes but got input type \
                 {input} and padding value type {padding_value}",
            ))
            .into());
        }
        let input_varying_manual_axes = input.sharding().map(|sharding| sharding.varying_manual_axes());
        let padding_varying_manual_axes = padding_value.sharding().map(|sharding| sharding.varying_manual_axes());
        if input_varying_manual_axes.cloned().unwrap_or_default()
            != padding_varying_manual_axes.cloned().unwrap_or_default()
        {
            return Err(TypeError::invalid(format!(
                "`{PAD_OPERATION_NAME}` input and padding value must have matching varying manual axes but got input type {input} and \
                 padding value type {padding_value}",
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
                "`{PAD_OPERATION_NAME}` input and padding value with distributed dependencies must use the same mesh"
            ))
            .into());
        }
    }
    ArrayType::new(input.data_type(), Shape::new(output_dimensions))
        .with_memory(input.memory())
        .with_sharding(sharding)
        .map_err(|error| TypeError::invalid(error.to_string()).into())
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation,
        DataType, DimensionBounds, DimensionType, DimensionValue, DimensionVariable, Layout, LogicalMesh, Memory,
        MeshAxis, MeshAxisType, RaggedAxis, Sharding, ShardingDimension, StridedLayout,
    };
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
    use crate::contexts::EagerContext;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::operations::constants::iota::IotaOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{EffectClasses, EmptyRegionDriver, ProgramBuilder, ProgramError, Typed};

    use super::*;

    #[test]
    fn test_pad() {
        let operation = PadOperation::new(vec![1], vec![2], vec![1]).unwrap();
        // Operation identity and accessors.
        assert_eq!(operation.name(), PAD_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "pad [edge_padding_low=[1], edge_padding_high=[2], interior_padding=[1]]");
        assert_eq!(operation.edge_padding_low(), &[1]);
        assert_eq!(operation.edge_padding_high(), &[2]);
        assert_eq!(operation.interior_padding(), &[1]);

        assert_eq!(
            PadOperation::new(vec![1], vec![2, 0], vec![1]),
            Err(ProgramError::Type(TypeError::invalid(
                "`pad` expects `edge_padding_low`, `edge_padding_high`, and `interior_padding` to share one length \
                    but got lengths 1, 2, and 1"
                    .to_string()
            ))),
        );
        let input_type = ArrayType::new_static(DataType::F64, [3]);
        let padding_value_type = ArrayType::scalar(DataType::F64);
        // Program rendering uses the canonical operation name and includes all three padding vectors.
        let mut builder = ProgramBuilder::<Array, PadOperation<ArrayType>>::new();
        let program_input = builder.add_input(input_type);
        let program_padding_value = builder.add_input(padding_value_type);
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
    fn test_pad_operation_with_input_types() {
        let operation = PadOperation::<ArrayIrType>::from(PadOperation::new(vec![1], vec![1], vec![0]).unwrap());
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
            Err(TypeError::invalid(
                "`pad` was constructed without a runtime extent check but these input types require one",
            ))
        );
        assert_eq!(
            operation.with_input_types(&[
                dynamic_type.clone().into(),
                ArrayType::scalar(DataType::F32).into(),
                DimensionType::new(DimensionVariable::new("disjoint", DimensionBounds::new(9, Some(10)).unwrap()))
                    .into(),
            ]),
            Err(TypeError::invalid(
                "`pad` output bounds [9, 10) on axis 0 cannot contain a padded extent derived from input bounds [1, 5)",
            ))
        );
        let identity = PadOperation::<ArrayIrType>::from(PadOperation::new(vec![-1], vec![1], vec![0]).unwrap())
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
    fn test_pad_operation_requires_runtime_assertion() {
        let operation = PadOperation::<ArrayIrType>::from(PadOperation::new(vec![1], vec![1], vec![0]).unwrap());
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
    fn test_pad_type_inference() {
        assert_eq!(
            PadOperation::<ArrayType>::new(vec![0], vec![0], vec![0]).unwrap().infer_output_types(
                &[ArrayType::new_static(DataType::F32, [2]), ArrayType::scalar(DataType::F32)],
                &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1"))
        );
        assert_eq!(
            PadOperation::<ArrayIrType>::from(PadOperation::new(vec![0], vec![0], vec![0]).unwrap())
                .infer_output_types(
                    &[
                        ArrayType::new_static(DataType::F32, [2]).into(),
                        ArrayType::scalar(DataType::F32).into(),
                        DimensionValue::constant(2).unwrap().r#type().into_owned().into()
                    ],
                    &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)],
                ),
            Err(TypeError::invalid("expected 0 regions but got 1"))
        );
        // Unchanged dynamic extents keep their identity even when another axis changes or edge shifts balance.
        let size = DimensionVariable::new("size", DimensionBounds::new(1, Some(5)).unwrap());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![size.clone().into(), 2.into()]));
        assert_eq!(
            PadOperation::new(vec![0, 1], vec![0, 1], vec![0, 0])
                .unwrap()
                .infer_output_types(&[input_type, ArrayType::scalar(DataType::F32)], &[],),
            Ok(vec![ArrayType::new(DataType::F32, Shape::new(vec![size.clone().into(), 4.into()]))])
        );
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![size.into()]));
        assert_eq!(
            PadOperation::new(vec![-1], vec![1], vec![0])
                .unwrap()
                .infer_output_types(&[input_type.clone(), ArrayType::scalar(DataType::F32)], &[],),
            Ok(vec![input_type])
        );
        let operation = PadOperation::new(vec![1], vec![2], vec![1]).unwrap();
        // Type inference validates the padding geometry and returns the padded type, and the type-level (abstract)
        // capability backs it without consuming the borrowed input type. With d = 3, low = 1, high = 2, and
        // interior = 1, the output dimension is 1 + (3 - 1) * 2 + 1 + 2 = 8.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let padding_value_type = ArrayType::scalar(DataType::F64);
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(8)]));
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    input_types = [input_type.clone(), padding_value_type.clone()],
                    output_types = [output_type.clone()],
                },
                {
                    input_types = [input_type.clone()],
                    error = "expected 2 inputs but got 1",
                },
                {
                    input_types = [input_type.clone(), ArrayType::scalar(DataType::F32)],
                    error = "`pad` input data type `f64` does not match padding value data type `f32`",
                },
                {
                    input_types = [input_type.clone(), input_type.clone()],
                    error = "`pad` padding value must be a scalar but has type `f64[3]`",
                },
                {
                    input_types = [
                        ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                            "input",
                            DimensionBounds::unbounded(),
                        ))])),
                        padding_value_type.clone(),
                    ],
                    error = "`pad` dynamic axis 0 requires an explicit result-dimension operand",
                },
                {
                    input_types = [
                        ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                            "input",
                            DimensionBounds::non_negative(Some(4)).unwrap(),
                        ))])),
                        padding_value_type.clone(),
                    ],
                    error = "`pad` dynamic axis 0 requires an explicit result-dimension operand",
                },
                {
                    input_types = [
                        ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(usize::MAX)])),
                        padding_value_type.clone(),
                    ],
                    error = "`pad` output size overflows `usize` on axis 0",
                },
            ],
        );
        assert_eq!(input_type.pad(&padding_value_type, &[1], &[2], &[1]), Ok(output_type.clone()));
        let output_extent = DimensionValue::constant(8).unwrap();
        let composite_operation = PadOperation::<ArrayIrType>::from(operation.clone());
        assert_eq!(
            composite_operation.infer_output_types(
                &[
                    input_type.clone().into(),
                    padding_value_type.clone().into(),
                    output_extent.r#type().into_owned().into(),
                ],
                &[],
            ),
            Ok(vec![output_type.clone().into()]),
        );
        assert_eq!(
            composite_operation.infer_output_types(
                &[
                    input_type.clone().into(),
                    padding_value_type.clone().into(),
                    DimensionType::new(DimensionVariable::new("wrong", DimensionBounds::new(7, Some(8)).unwrap(),))
                        .into(),
                ],
                &[],
            ),
            Err(TypeError::invalid(
                "`pad` output bounds [7, 8) on axis 0 cannot contain a padded extent derived from input bounds [3, 4)"
            )),
        );
        let supplied_output = DimensionVariable::new("dynamic", DimensionBounds::new(7, Some(10)).unwrap());
        assert_eq!(
            composite_operation.infer_output_types(
                &[
                    input_type.clone().into(),
                    padding_value_type.clone().into(),
                    DimensionType::new(supplied_output.clone()).into(),
                ],
                &[],
            ),
            Ok(vec![ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![supplied_output.into()])))]),
        );

        // Intersecting result bounds admit valid runtime geometries; the checked operation enforces exact equality.
        let input_variable = DimensionVariable::new("input", DimensionBounds::new(1, Some(5)).unwrap());
        let output_variable = DimensionVariable::new("output", DimensionBounds::new(3, Some(7)).unwrap());
        let dynamic_input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(input_variable.clone())]));
        let dynamic_output_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(output_variable.clone())]));
        let dynamic_operation =
            PadOperation::<ArrayIrType>::from(PadOperation::new(vec![1], vec![1], vec![0]).unwrap());
        assert_eq!(
            dynamic_operation.infer_output_types(
                &[
                    dynamic_input_type.clone().into(),
                    padding_value_type.clone().into(),
                    DimensionType::new(output_variable).into(),
                ],
                &[],
            ),
            Ok(vec![dynamic_output_type.into()]),
        );
        let narrow_output = DimensionVariable::new("narrow", DimensionBounds::new(3, Some(6)).unwrap());
        assert_eq!(
            dynamic_operation.infer_output_types(
                &[
                    dynamic_input_type.into(),
                    padding_value_type.clone().into(),
                    DimensionType::new(narrow_output.clone()).into(),
                ],
                &[],
            ),
            Ok(vec![ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![narrow_output.into()])))]),
        );
        let zero_bounded_input = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                "possibly_empty",
                DimensionBounds::new(0, Some(5)).unwrap(),
            ))]),
        );
        let cropped = DimensionVariable::new("cropped", DimensionBounds::new(0, Some(4)).unwrap());
        assert_eq!(
            PadOperation::<ArrayIrType>::from(PadOperation::new(vec![-1], vec![0], vec![0]).unwrap())
                .infer_output_types(
                    &[
                        zero_bounded_input.into(),
                        padding_value_type.clone().into(),
                        DimensionType::new(cropped.clone()).into(),
                    ],
                    &[],
                ),
            Ok(vec![ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![cropped.into()])))]),
        );
        assert_eq!(
            input_type.pad(&padding_value_type, &[], &[0], &[0]),
            Err(ProgramError::Type(TypeError::invalid(
                "`pad` `edge_padding_low` has length 0 but input has rank 1".to_string()
            ))),
        );
        assert_eq!(
            input_type.pad(&padding_value_type, &[0], &[], &[0]),
            Err(ProgramError::Type(TypeError::invalid(
                "`pad` `edge_padding_high` has length 0 but input has rank 1".to_string()
            ))),
        );
        assert_eq!(
            input_type.pad(&padding_value_type, &[0], &[0], &[]),
            Err(ProgramError::Type(TypeError::invalid(
                "`pad` `interior_padding` has length 0 but input has rank 1".to_string()
            ))),
        );
        // Negative inverse edges still validate their abstract extent. A valid derived dynamic extent requires the
        // explicit result-dimension operand introduced by the mixed operation signature, while an always-negative
        // extent is rejected immediately.
        assert_eq!(
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                    "input",
                    DimensionBounds::non_negative(Some(9)).unwrap(),
                ))]),
            )
            .pad(&padding_value_type, &[-1], &[-2], &[0]),
            Err(ProgramError::Type(TypeError::invalid(
                "`pad` dynamic axis 0 requires an explicit result-dimension operand".to_string(),
            ))),
        );
        assert_eq!(
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("input", DimensionBounds::unbounded(),))]),
            )
            .pad(&padding_value_type, &[-1], &[-2], &[0]),
            Err(ProgramError::Type(TypeError::invalid(
                "`pad` dynamic axis 0 requires an explicit result-dimension operand".to_string(),
            ))),
        );
        assert_eq!(
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                    "dynamic",
                    DimensionBounds::non_negative(Some(2)).unwrap(),
                ))])
            )
            .pad(&padding_value_type, &[-5], &[0], &[0]),
            Err(ProgramError::Type(TypeError::invalid(
                "`pad` output size is negative (-4) on dynamic axis 0 even at its maximum input extent 1".to_string()
            ))),
        );

        assert_eq!(
            PadOperation::new(vec![1, 0], vec![2, 0], vec![1, 0])
                .unwrap()
                .infer_output_types(&[input_type.clone(), padding_value_type.clone()], &[]),
            Err(TypeError::invalid("`pad` `edge_padding_low` has length 2 but input has rank 1".to_string())),
        );
        // The homogeneous contract cannot manufacture a fresh result identity for a bounded-dynamic operand. The
        // canonical array IR contract supplies that result extent explicitly; keep this rejection to prevent
        // callers from falling back to implicit identity recovery.
        let dynamic_input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                "input",
                DimensionBounds::non_negative(Some(4)).unwrap(),
            ))]),
        );
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let dynamic_input = builder.add_input(dynamic_input_type);
        let dynamic_padding = builder.add_input(ArrayType::scalar(DataType::F64));
        assert_eq!(
            builder.add_instruction(
                PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
                Vec::new(),
                vec![dynamic_input, dynamic_padding],
                None
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "`pad` dynamic axis 0 requires an explicit result-dimension operand".to_string(),
            ))),
        );
    }

    #[test]
    fn test_pad_interpretation() {
        let shifted = PadOperation::new(vec![-1], vec![1], vec![0])
            .unwrap()
            .interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[
                    Array::from_elements(ArrayType::new_static(DataType::I32, [3]), &[1_i32, 2, 3]).unwrap(),
                    Array::from_elements(ArrayType::scalar(DataType::I32), &[9_i32]).unwrap(),
                ],
            )
            .unwrap();
        assert_eq!(
            shifted,
            vec![Array::from_elements(ArrayType::new_static(DataType::I32, [3]), &[2_i32, 3, 9]).unwrap()]
        );
        let operation = PadOperation::new(vec![1], vec![2], vec![1]).unwrap();
        let padding_value_type = ArrayType::scalar(DataType::F64);
        let output_type = ArrayType::new_static(DataType::F64, [8]);
        let output_extent = DimensionValue::constant(8).unwrap();
        let composite_operation = PadOperation::<ArrayIrType>::from(operation.clone());
        // Interpretation writes the input elements at `low + i * (interior + 1)` (positions 1, 3, and 5) and fills
        // every other position with the padding value.
        let input = Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[1.0, 2.0, 3.0]).unwrap();
        let output = operation
            .interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[input, Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[9.0]).unwrap()],
            )
            .unwrap();
        assert_eq!(*output[0].r#type(), output_type);
        assert_eq!(output[0].elements::<f64>(), Ok(vec![9.0, 1.0, 9.0, 2.0, 9.0, 3.0, 9.0, 9.0]));
        let output = InterpretableOperation::<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>::interpret(
            &composite_operation,
            &EagerContext::new(),
            &EmptyRegionDriver,
            &[
                ArrayIrValue::Array(
                    Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[1.0, 2.0, 3.0]).unwrap(),
                ),
                ArrayIrValue::Array(Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[9.0]).unwrap()),
                ArrayIrValue::Dimension(output_extent),
            ],
        )
        .unwrap();
        assert_eq!(
            output,
            vec![ArrayIrValue::Array(
                Array::from_elements::<f64>(
                    ArrayType::new_static(DataType::F64, [8]),
                    &[9.0, 1.0, 9.0, 2.0, 9.0, 3.0, 9.0, 9.0,]
                )
                .unwrap()
            )],
        );

        // Empty input axes hold only the edge padding (the `d == 0` case skips interior padding entirely) and
        // rank-0 inputs pass through unchanged.
        let empty_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0)]));
        assert_eq!(
            empty_type.pad(&padding_value_type, &[1], &[2], &[1]),
            Ok(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]))),
        );
        let empty = Array::from_elements::<f64>(empty_type, &[])
            .unwrap()
            .pad(&Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[7.0]).unwrap(), &[1], &[2], &[1])
            .unwrap();
        assert_eq!(empty.elements::<f64>(), Ok(vec![7.0, 7.0, 7.0]));
        let scalar = Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[42.0])
            .unwrap()
            .pad(&Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[7.0]).unwrap(), &[], &[], &[])
            .unwrap();
        assert_eq!(scalar.elements::<f64>(), Ok(vec![42.0]));

        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        );
    }

    #[test]
    fn test_pad_partial_evaluation() {
        // Check standard partial evaluation with known and residual operands.
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
                        (@mapped(axis = 0), Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2, 2]), &[1.0, 2.0, 3.0, 4.0]).unwrap()),
                        (@replicated, Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[0.0]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2, 3]), &[0.0, 1.0, 2.0, 0.0, 3.0, 4.0]).unwrap())],
                },
                {
                    inputs = [
                        (@replicated, Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2]), &[1.0, 2.0]).unwrap()),
                        (@replicated, Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[0.0]).unwrap()),
                    ],
                    outputs = [(@replicated, Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[0.0, 1.0, 2.0]).unwrap())],
                },
                {
                    inputs = [
                        (@mapped(axis = 0), Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2, 2]), &[1.0, 2.0, 3.0, 4.0]).unwrap()),
                        (@mapped(axis = 0), Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2]), &[8.0, 9.0]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2, 3]), &[8.0, 1.0, 2.0, 9.0, 3.0, 4.0]).unwrap())],
                },
                {
                    inputs = [
                        (@replicated, Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2]), &[1.0, 2.0]).unwrap()),
                        (@mapped(axis = 0), Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2]), &[8.0, 9.0]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 0), Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2, 3]), &[8.0, 1.0, 2.0, 9.0, 1.0, 2.0]).unwrap())],
                },
                {
                    inputs = [
                        (@mapped(axis = 1), Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2, 2]), &[1.0, 2.0, 3.0, 4.0]).unwrap()),
                        (@mapped(axis = 0), Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2]), &[8.0, 9.0]).unwrap()),
                    ],
                    outputs = [(@mapped(axis = 1), Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3, 2]), &[8.0, 9.0, 1.0, 2.0, 3.0, 4.0]).unwrap())],
                },
            ],
        );
    }

    #[test]
    fn test_pad_batching_sharding() {
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let physical_sharding =
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap()
                    .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                    .unwrap();
            let input_type =
                ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]))
                    .with_sharding(physical_sharding)
                    .unwrap();
            let input = ArrayBatch::new(
                Array::from_elements::<f64>(input_type, &[1.0, 2.0, 3.0, 4.0]).unwrap(),
                BatchAxis::new(0),
            )
            .unwrap();
            let padding_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))
                .with_sharding(
                    Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])])
                        .unwrap()
                        .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                        .unwrap(),
                )
                .unwrap();
            let padding =
                ArrayBatch::new(Array::from_elements::<f64>(padding_type, &[8.0, 9.0]).unwrap(), BatchAxis::new(0))
                    .unwrap();
            let context = BatchingContext::new(EagerContext::<Array>::new(), 2)
                .with_axis_sharding(ShardingDimension::sharded(["x"]));

            let outputs = PadOperation::new(vec![1], vec![0], vec![0])
                .unwrap()
                .batch(&context, &EmptyRegionDriver, &[input, padding])
                .unwrap()
                .into_parts()
                .0;

            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
            assert_eq!(
                outputs[0].r#type().sharding().unwrap().dimensions(),
                &[ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
            );
            assert_eq!(outputs[0].value().elements::<f64>(), Ok(vec![8.0, 1.0, 2.0, 9.0, 3.0, 4.0]));
        }

        // The vectorized mapped-padding rule handles an empty batch without inventing values or dropping placement.
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let physical_sharding =
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap()
                    .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                    .unwrap();
            let input_type =
                ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0), Dimension::Static(2)]))
                    .with_sharding(physical_sharding.clone())
                    .unwrap();
            let input =
                ArrayBatch::new(Array::from_elements::<f64>(input_type, &[]).unwrap(), BatchAxis::new(0)).unwrap();
            let padding_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0)]))
                .with_sharding(
                    Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])])
                        .unwrap()
                        .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                        .unwrap(),
                )
                .unwrap();
            let padding =
                ArrayBatch::new(Array::from_elements::<f64>(padding_type, &[]).unwrap(), BatchAxis::new(0)).unwrap();
            let context = BatchingContext::new(EagerContext::<Array>::new(), 0)
                .with_axis_sharding(ShardingDimension::sharded(["x"]));

            let outputs = PadOperation::new(vec![1], vec![0], vec![0])
                .unwrap()
                .batch(&context, &EmptyRegionDriver, &[input, padding])
                .unwrap()
                .into_parts()
                .0;

            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
            assert_eq!(outputs[0].r#type().sharding().unwrap().dimensions(), physical_sharding.dimensions(),);
            assert_eq!(outputs[0].r#type().shape().dimensions(), &[Dimension::Static(0), Dimension::Static(3)]);
            assert!(outputs[0].value().storage_bytes().is_empty());
        }
    }

    #[test]
    fn test_pad_batching_decomposes_mapped_padding_values() -> Result<(), ProgramError> {
        // A mapped padding value is decomposed into placeholder padding, a padding-position mask, a broadcast of
        // the per-item scalar, and a select, so each batch item receives its own padding value.
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2)?),
        );
        let pad = ArrayIrOperation::<Array>::from(PadOperation::new(vec![1], vec![0], vec![0])?);
        assert_eq!(
            pad.batch(
                &context,
                &EmptyRegionDriver,
                &[
                    ArrayIrBatch::new(
                        ArrayIrValue::Array(
                            Array::from_elements::<f32>(
                                ArrayType::new_static(DataType::F32, [2, 2]),
                                &[1.0_f32, 2.0, 3.0, 4.0]
                            )
                            .unwrap()
                        ),
                        BatchAxis::new(0),
                    )?,
                    ArrayIrBatch::new(
                        ArrayIrValue::Array(
                            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [2]), &[8.0_f32, 9.0])
                                .unwrap()
                        ),
                        BatchAxis::new(0)
                    )?,
                    ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(3)?)),
                ],
            )?
            .into_parts()
            .0,
            vec![ArrayIrBatch::new(
                ArrayIrValue::Array(
                    Array::from_elements::<f32>(
                        ArrayType::new_static(DataType::F32, [2, 3]),
                        &[8.0_f32, 1.0, 2.0, 9.0, 3.0, 4.0]
                    )
                    .unwrap()
                ),
                BatchAxis::new(0),
            )?],
        );

        // Under a symbolic mapped extent, the replicated operand is aligned through a dynamic broadcast and every
        // shape-changing instruction of the decomposition receives the same explicit output extents, including the
        // inserted batch extent.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
        let batch_extent = trace.input(DimensionType::new(batch.clone()).into());
        let operand = trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)])).into());
        let padding = trace.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch)])).into());
        let result_extent = trace.input(DimensionValue::constant(3)?.r#type().into_owned().into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent);
        let [output] = context
            .bind(
                ArrayIrOperation::from(PadOperation::new(vec![1], vec![0], vec![0])?),
                Vec::new(),
                &[
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(operand)),
                    BatchingTracer::new(context.clone(), ArrayIrBatch::new(padding, BatchAxis::new(0))?),
                    BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(result_extent)),
                ],
            )?
            .try_into()
            .unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(0));
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![output.batch().value().atom_id()?],
            vec![Placeholder; 4],
            vec![Placeholder],
        )?;
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
        Ok(())
    }

    #[test]
    fn test_pad_batching_preserves_ragged_dimensions() {
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
        let output = PadOperation::new(vec![0, 1], vec![0, 1], vec![0, 0])
            .unwrap()
            .batch(&context, &EmptyRegionDriver, &[input.clone(), padding.clone()])
            .unwrap()
            .into_parts()
            .0
            .remove(0);
        assert_eq!(output.ragged_axes(), &[ragged]);
        assert_eq!(
            output.value().elements::<f32>(),
            Ok(vec![9., 1., 9., 9., 0., 9., 9., 0., 9., 9., 2., 9., 9., 3., 9., 9., 4., 9.])
        );
        assert!(matches!(PadOperation::new(vec![1, 0], vec![0, 0], vec![0, 0]).unwrap()
            .batch(&context, &EmptyRegionDriver, &[input, padding]),
            Err(BatchingError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`pad` batching cannot change a ragged axis or an axis indexing its extents"));
    }

    #[test]
    fn test_pad_batching_mapped_padding_without_numeric_zero() {
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
        let output = PadOperation::new(vec![1], vec![0], vec![0])
            .unwrap()
            .batch(&context, &EmptyRegionDriver, &[input, padding])
            .unwrap()
            .into_parts()
            .0
            .remove(0);
        assert_eq!(output.value().storage_bytes(), &[129, 127, 130, 128]);
    }

    #[test]
    fn test_pad_differentiation() {
        // Pad is linear in both inputs: its JVP pads tangent values and its pullback separates written and padding
        // positions.
        check_operation_differentiation!(
            @approx(step = 0.125, epsilon = 1e-9),
            operation = PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
            cases = [{
                primals = [Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[1.0, 2.0, 3.0]).unwrap(), Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[9.0]).unwrap()],
                tangents = [Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[0.1, 0.2, 0.3]).unwrap(), Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[0.5]).unwrap()],
                primal_outputs = [Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [8]), &[9.0, 1.0, 9.0, 2.0, 9.0, 3.0, 9.0, 9.0]).unwrap()],
                tangent_outputs = [Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [8]), &[0.5, 0.1, 0.5, 0.2, 0.5, 0.3, 0.5, 0.5]).unwrap()],
            }],
        );
    }

    #[test]
    fn test_array_ir_pad_differentiation_dynamic_geometry() {
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
                PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
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
            linearization
                .tangent()
                .instructions()
                .iter()
                .filter(|instruction| matches!(instruction.operation(), ArrayIrOperation::LinearCall(_)))
                .count(),
            1
        );
        assert!(linearization.pullback().unwrap().entry_region_ref().instructions_in_closure().any(|(_, instruction)|
            matches!(instruction.operation(), ArrayIrOperation::DynamicShapeSlice(operation) if operation.strides() == [2])));

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
            linearization.pullback().unwrap().interpret(pullback_inputs),
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
            linearization.pullback().unwrap().interpret(pullback_inputs),
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
        // static leading axis to verify that the pullback selects dynamic constructor operands from the right axis.
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
                PadOperation::new(vec![0, 1], vec![0, 1], vec![0, 0]).unwrap(),
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
    fn test_array_ir_dynamic_pad_disconnected_operand_tangent_uses_a_primal_exemplar() {
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
        // A mixed iota is a non-differentiable nullary constant, so its tangent is a structural zero of the
        // operand type with symbolic extents while its primal is a non-zero exemplar and the padding-value tangent
        // stays live. The rule must still hand a concrete operand tangent to the staged pad.
        let operand = builder
            .add_instruction(
                ArrayIrOperation::<Array>::from(IotaOperation::new(input_type, 0).unwrap()),
                Vec::new(),
                vec![source_extent],
                None,
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(
                PadOperation::new(vec![1], vec![1], vec![0]).unwrap(),
                Vec::new(),
                vec![operand, padding_value, output_extent],
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
                PadOperation::new(vec![1], vec![1], vec![0]).unwrap(),
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
        for (low, high, interior, minimum) in
            [(0, 0, usize::MAX, 0), (i64::MIN, i64::MAX, 0, 1), (i64::MAX, i64::MIN, 0, 1)]
        {
            let size = DimensionVariable::new("size", DimensionBounds::new(minimum, Some(minimum + 2)).unwrap());
            let output_size = DimensionVariable::new("output_size", DimensionBounds::new(0, Some(2)).unwrap());
            let extent_type = DimensionType::new(output_size);
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let input = builder.add_input(ArrayType::new(DataType::F32, Shape::new(vec![size.into()])).into());
            let padding = builder.add_input(ArrayType::scalar(DataType::F32).into());
            let extent = builder.add_input(extent_type.clone().into());
            let output = builder
                .add_instruction(
                    PadOperation::new(vec![low], vec![high], vec![interior]).unwrap(),
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
            for output_extent in [0, 1] {
                let input_extent = output_extent + minimum;
                let input_type = ArrayType::new_static(DataType::F32, [input_extent]);
                let mut outputs = linearization
                    .primal()
                    .interpret(vec![
                        ArrayIrValue::Array(
                            Array::from_elements(input_type.clone(), &vec![3_f32; input_extent]).unwrap(),
                        ),
                        ArrayIrValue::Array(Array::from_elements(ArrayType::scalar(DataType::F32), &[9_f32]).unwrap()),
                        ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), output_extent).unwrap()),
                    ])
                    .unwrap();
                let mut cotangents = vec![ArrayIrValue::Array(
                    Array::from_elements(
                        ArrayType::new_static(DataType::F32, [output_extent]),
                        &vec![7_f32; output_extent],
                    )
                    .unwrap(),
                )];
                cotangents.extend(outputs.split_off(1));
                assert_eq!(
                    linearization.pullback().unwrap().interpret(cotangents),
                    Ok(vec![
                        ArrayIrValue::Array(
                            Array::from_elements(
                                input_type,
                                &vec![if minimum == 0 { 7_f32 } else { 0. }; input_extent]
                            )
                            .unwrap()
                        ),
                        ArrayIrValue::Array(
                            Array::from_elements(
                                ArrayType::scalar(DataType::F32),
                                &[if minimum == 0 { 0_f32 } else { 7. * output_extent as f32 }]
                            )
                            .unwrap()
                        ),
                    ])
                );
            }
        }
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
                    output_cotangents = [Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [8]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap()],
                    input_cotangents = [Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[2.0, 4.0, 6.0]).unwrap(), Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[24.0]).unwrap()],
                },
                {
                    inputs = [
                        (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![3.into()])))),
                        (@known, Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[9.0]).unwrap()),
                    ],
                    output_cotangents = [Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [8]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap()],
                    input_cotangents = [Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[2.0, 4.0, 6.0]).unwrap()],
                },
                {
                    inputs = [
                        (@known, Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[1.0, 2.0, 3.0]).unwrap()),
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                    ],
                    output_cotangents = [Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [8]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap()],
                    input_cotangents = [Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[24.0]).unwrap()],
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
                    output_cotangents = [Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[5.0, f64::INFINITY, 7.0]).unwrap()],
                    input_cotangents = [Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2]), &[f64::INFINITY, 7.0]).unwrap(), Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[5.0]).unwrap()],
                },
                {
                    inputs = [
                        (@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![2.into()])))),
                        (@linear(type = ArrayType::scalar(DataType::F64))),
                    ],
                    output_cotangents = [Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[3.0, 1e20, -1e20]).unwrap()],
                    input_cotangents = [Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2]), &[1e20, -1e20]).unwrap(), Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[3.0]).unwrap()],
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
                output_cotangents = [Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[2.0, 3.0, 5.0]).unwrap()],
                input_cotangents = [Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[0.0, 2.0, 3.0]).unwrap(), Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[5.0]).unwrap()],
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
                output_cotangents = [Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3]), &[1.0, 2.0, 3.0]).unwrap()],
                input_cotangents = [Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [0]), &[]).unwrap(), Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[6.0]).unwrap()],
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
                output_cotangents = [Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[f64::INFINITY]).unwrap()],
                input_cotangents = [Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[f64::INFINITY]).unwrap(), Array::from_elements::<f64>(ArrayType::scalar(DataType::F64), &[0.0]).unwrap()],
            }],
        );

        // A pure crop never reads the padding scalar, so its dependency metadata may differ from the operand's. The
        // inverse pad nevertheless introduces zeros for cropped input positions and must derive that internal zero's
        // dependencies from the operand cotangent rather than from the unused primal padding scalar.
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

        // The pullback restores the complete cotangent types of both operands after slicing and reducing the output
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
    fn test_pad_transposition_extreme_configuration() {
        // Interior padding is irrelevant when an axis contains fewer than two input elements.
        check_operation_transposition!(
            @exact,
            operation = PadOperation::new(vec![0], vec![0], vec![usize::MAX]).unwrap(),
            cases = [
                {
                    inputs = [(@linear(type = ArrayType::new_static(DataType::F32, [0]))),
                        (@linear(type = ArrayType::scalar(DataType::F32)))],
                    output_cotangents = [Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0]), &[]).unwrap()],
                    input_cotangents = [Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0]), &[]).unwrap(),
                        Array::from_elements(ArrayType::scalar(DataType::F32), &[0_f32]).unwrap()],
                },
                {
                    inputs = [(@linear(type = ArrayType::new_static(DataType::F32, [1]))),
                        (@linear(type = ArrayType::scalar(DataType::F32)))],
                    output_cotangents = [Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[7_f32]).unwrap()],
                    input_cotangents = [Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[7_f32]).unwrap(),
                        Array::from_elements(ArrayType::scalar(DataType::F32), &[0_f32]).unwrap()],
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
                output_cotangents = [Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[7_f32]).unwrap()],
                input_cotangents = [Array::from_elements(ArrayType::new_static(DataType::F32, [2]), &[0_f32, 7.]).unwrap(),
                    Array::from_elements(ArrayType::scalar(DataType::F32), &[0_f32]).unwrap()],
            }],
        );
        // These edges crop every input position, despite their balanced finite output shape.
        for (low, high) in [(i64::MIN, i64::MAX), (i64::MAX, i64::MIN)] {
            check_operation_transposition!(
                @exact,
                operation = PadOperation::new(vec![low], vec![high], vec![0]).unwrap(),
                cases = [
                    {
                        inputs = [(@linear(type = ArrayType::new_static(DataType::F32, [1]))),
                            (@linear(type = ArrayType::scalar(DataType::F32)))],
                        output_cotangents = [Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0]), &[]).unwrap()],
                        input_cotangents = [Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[0_f32]).unwrap(),
                            Array::from_elements(ArrayType::scalar(DataType::F32), &[0_f32]).unwrap()],
                    },
                    {
                        inputs = [(@linear(type = ArrayType::new_static(DataType::F32, [2]))),
                            (@linear(type = ArrayType::scalar(DataType::F32)))],
                        output_cotangents = [Array::from_elements(ArrayType::new_static(DataType::F32, [1]), &[7_f32]).unwrap()],
                        input_cotangents = [Array::from_elements(ArrayType::new_static(DataType::F32, [2]), &[0_f32, 0.]).unwrap(),
                            Array::from_elements(ArrayType::scalar(DataType::F32), &[7_f32]).unwrap()],
                    },
                ],
            );
        }
    }

    #[test]
    fn test_array_ir_pad_transposition() {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])).into());
        let padding_value = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let output_extent = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(8).unwrap()));
        let output = builder
            .add_instruction(
                PadOperation::new(vec![1], vec![2], vec![1]).unwrap(),
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

        assert_eq!(
            program.transpose_with_respect_to(&[0, 1], &[]).unwrap().interpret(vec![ArrayIrValue::Array(
                Array::from_elements::<f64>(
                    ArrayType::new_static(DataType::F64, [8]),
                    &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,]
                )
                .unwrap()
            )]),
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
    }

    #[test]
    fn test_array_ir_pad_transposition_symbolic_zero() {
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("size", DimensionBounds::new(1, Some(5)).unwrap()).into()]),
        );
        let output_dimension_type =
            DimensionType::new(DimensionVariable::new("output_size", DimensionBounds::new(3, Some(7)).unwrap()));
        let input_types =
            vec![input_type.into(), ArrayType::scalar(DataType::F32).into(), output_dimension_type.into()];
        let operation = PadOperation::<ArrayIrType>::from(PadOperation::new(vec![1], vec![1], vec![0]).unwrap());
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
        assert!(matches!(validate_padding_ragged_axes(&[ragged], &[1, 0, 0], &[0, 0, 0], &[0, 0, 0]),
            Err(BatchingError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`pad` batching cannot change a ragged axis or an axis indexing its extents"));
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

        // Padding preserves a common memory placement and rejects a padding scalar that would require an implicit
        // transfer.
        let host_input = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_memory(Memory::Host { pinned: true });
        let host_padding = ArrayType::scalar(DataType::F32).with_memory(Memory::Host { pinned: true });
        assert_eq!(host_input.pad(&host_padding, &[0], &[1], &[0]).unwrap().memory(), Memory::Host { pinned: true },);
        assert_eq!(
            host_input.pad(&padding_value, &[0], &[1], &[0]),
            Err(ProgramError::Type(TypeError::invalid(
                "`pad` input and padding value must share one memory space but reside in `Host[Pinned]` and \
                          `Device`"
                    .to_string()
            ))),
        );
        let laid_out_input = host_input.with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        assert_eq!(laid_out_input.pad(&host_padding, &[0], &[0], &[0]), Ok(laid_out_input.clone()));

        // Padding to an evenly divisible size keeps the operand sharding (including the unreduced manual axis): with
        // low = 0, interior = 0, and high = 4 the output is 0 + 4 + 4 = 8, divisible by the `x` mesh-axis size (2).
        assert_eq!(input.pad(&padding_value, &[0], &[4], &[0]).unwrap().sharding(), Some(&sharding));
        // Padding to a size not divisible by the explicit mesh-axis size (output 0 + 4 + 1 = 5) is rejected.
        assert!(input.pad(&padding_value, &[0], &[1], &[0]).is_err());

        // JAX requires exact dependency metadata whenever the padding value can contribute. Neither reduced axes nor
        // varying manual axes are implicitly unioned from the scalar.
        let plain_padding = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(sharding.mesh().clone(), 0))
            .unwrap();
        assert!(input.pad(&plain_padding, &[0], &[4], &[0]).is_err());
        let reduced_padding = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(sharding.mesh().clone(), 0).with_reduced_axes(["m"]).unwrap())
            .unwrap();
        assert!(input.pad(&reduced_padding, &[0], &[4], &[0]).is_err());
        let varying_input = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(
                Sharding::new(sharding.mesh().clone(), vec![ShardingDimension::sharded(["x"])])
                    .unwrap()
                    .with_varying_manual_axes(["m"])
                    .unwrap(),
            )
            .unwrap();
        assert!(varying_input.pad(&padding_value, &[0], &[4], &[0]).is_err());

        // Mesh identity is irrelevant for an ordinary scalar when neither side carries VMA or reduction metadata;
        // the result placement is derived solely from the operand. Effective identities do not consult the unused
        // padding value's dependency metadata at all.
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
        // A rank-2 pad exercises the odometer across axes with different padding amounts: rows gain one interior
        // row and columns gain asymmetric edge padding.
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
        assert_eq!(output.elements::<f64>(), Ok(vec![0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0]),);

        // Signed edge padding crops the dilated operand. Cropping can be asymmetric, can combine with interior
        // dilation, and must not be elided merely because the output shape happens to equal the input shape.
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
            Err(ProgramError::Type(TypeError::invalid("`pad` output size is negative (-1) on axis 0".to_string()))),
        );

        // Interior padding is an effective identity on singleton axes. The eager and abstract fast paths preserve
        // the complete type and avoid overflowing `interior + 1` for a value that can never be used as a stride.
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
            Err(ProgramError::Type(TypeError::invalid(
                "`pad` padding value must be a scalar but has type `f64[1]`".to_string()
            ))),
        );
    }

    #[test]
    fn test_array_pad_structural_zero() {
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
    fn test_array_pad_with_config() {
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
            Err(ProgramError::Type(TypeError::invalid("`pad` `edge_padding_low` has length 0 but input has rank 1",)))
        );
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
                message: "`pad` output axis 0 has extent 5, but its explicit extent operand is 4".to_string(),
            })
        );
        assert_eq!(padding.dynamic_pad(&padding, &[], &[], &[], &[]), Ok(padding));
    }

    #[test]
    fn test_dynamic_pad() {
        let size = DimensionVariable::new("size", DimensionBounds::new(1, Some(5)).unwrap());
        let output_size = DimensionVariable::new("output_size", DimensionBounds::new(3, Some(7)).unwrap());
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = context.input(ArrayType::new(DataType::F32, Shape::new(vec![size.into()])).into());
        let padding = context.input(ArrayType::scalar(DataType::F32).into());
        let extent = context.input(DimensionType::new(output_size.clone()).into());
        let output = input.dynamic_pad(&padding, &[extent.clone()], &[1], &[1], &[0]).unwrap();
        assert_eq!(
            output.r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![output_size.into()])),)
        );
        let builder = context.builder().borrow();
        let [instruction] = builder.instructions() else { panic!("expected one padding instruction") };
        assert!(matches!(instruction.operation(), ArrayIrOperation::Pad(_)));
        assert_eq!(
            instruction.inputs(),
            &[input.atom_id().unwrap(), padding.atom_id().unwrap(), extent.atom_id().unwrap()]
        );
    }
}
