use std::fmt::Display;

use crate::DimensionValue;
use crate::backends::scalars::Scalar;
use crate::batching::{ArrayBatch, BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain};
use crate::differentiation::elementwise::BroadcastDerivativeAlignment;
use crate::differentiation::forward::DifferentiationDual;
use crate::differentiation::types::DifferentiableType;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::constants::Fill;
use crate::operations::dimensions::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::concatenation::Concatenate;
use crate::operations::manipulation::conversion::ConvertElementTypeOperation;
use crate::operations::manipulation::reshaping::LegacyReshapeOperation;
use crate::operations::manipulation::transposition::TransposeOperation;
use crate::operations::math::ReduceOperation;
use crate::operations::sharding::ReshardOperation;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::ProgramError;
use crate::programs::atoms::MaybeZero;
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::Value;
use crate::sharding::{Sharding, ShardingDimension};
use crate::types::{ArrayProgramType, ArrayType, DataType, Dimension, DimensionType, Shape};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`BroadcastOperation`].
pub const BROADCAST_OPERATION_NAME: &str = "broadcast";

/// Mixed [`Operation`] that broadcasts one array using one explicit first-class dimension operand per output axis.
///
/// Operand zero is the array. Every remaining operand describes the corresponding output-axis extent, in order.
/// Exact dimension types produce static axes while non-exact dimension types retain their variables as dynamic axes.
/// `output_axes[input_axis]` names the output axis to which that input axis maps.
/// Refer to [`Broadcast`] for the underlying array broadcasting semantics.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BroadcastOperation {
    /// Vector that contains, for each input axis, the output axis to which it maps.
    output_axes: Vec<usize>,

    /// Optional requested output [`Sharding`].
    output_sharding: Option<Sharding>,
}

impl BroadcastOperation {
    /// Creates a broadcast with the supplied input-to-output axis mapping.
    #[inline]
    pub fn new(output_axes: Vec<usize>) -> Self {
        Self { output_axes, output_sharding: None }
    }

    /// Returns this operation with the requested output `sharding`.
    #[inline]
    pub fn with_output_sharding(mut self, sharding: impl Into<Option<Sharding>>) -> Self {
        self.output_sharding = sharding.into();
        self
    }

    /// Returns the output axes. The resulting slice contains, for each input axis, the output axis to which it maps.
    #[inline]
    pub fn output_axes(&self) -> &[usize] {
        self.output_axes.as_slice()
    }

    /// Returns the requested output sharding, if any.
    #[inline]
    pub fn output_sharding(&self) -> Option<&Sharding> {
        self.output_sharding.as_ref()
    }
}

impl Display for BroadcastOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as Operation<ArrayProgramType>>::render(self, formatter, 0)
    }
}

impl Operation<ArrayProgramType> for BroadcastOperation {
    #[inline]
    fn name(&self) -> &'static str {
        BROADCAST_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayProgramType],
        region_interfaces: &[RegionInterface<ArrayProgramType>],
    ) -> Result<Vec<ArrayProgramType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        let Some((input_type, output_extent_types)) = input_types.split_first() else {
            return Err(TypeError::invalid("'broadcast' expects an array followed by its output extents"));
        };
        let input_type = <&ArrayType>::try_from(input_type)?;
        let output_shape = Shape::new(
            output_extent_types
                .iter()
                .map(<&DimensionType>::try_from)
                .map(|result| result.map(DimensionType::to_dimension))
                .collect::<Result<Vec<_>, _>>()?,
        );
        Ok(vec![infer_explicit_broadcast_output_type(input_type, output_shape, self)?.into()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, BROADCAST_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("output_axes", format_args!("{:?}", self.output_axes))?;
            if let Some(output_sharding) = &self.output_sharding {
                operation.field("output_sharding", output_sharding)?;
            }
            Ok(())
        })
    }
}

/// Legacy homogeneous [`Operation`] that performs general N-dimensional broadcasting.
/// Refer to the documentation of [`LegacyBroadcast`] for more information.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LegacyBroadcastOperation {
    /// Output [`ArrayType`].
    output_type: ArrayType,

    /// Vector that contains, for each input axis `i`, the output axis that it maps to.
    output_axes: Vec<usize>,
}

impl LegacyBroadcastOperation {
    /// Creates a new [`LegacyBroadcastOperation`] with the supplied output type and output axes.
    #[inline]
    pub fn new(output_type: ArrayType, output_axes: Vec<usize>) -> Self {
        Self { output_type, output_axes }
    }

    /// Returns the output [`ArrayType`] of this [`LegacyBroadcastOperation`].
    #[inline]
    pub fn output_type(&self) -> &ArrayType {
        &self.output_type
    }

    /// Returns the output axes of this [`LegacyBroadcastOperation`]. The resulting slice contains, for each input axis,
    /// the output axis that it maps to.
    #[inline]
    pub fn output_axes(&self) -> &[usize] {
        self.output_axes.as_slice()
    }
}

impl Display for LegacyBroadcastOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for LegacyBroadcastOperation {
    #[inline]
    fn name(&self) -> &'static str {
        BROADCAST_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        match input_types[0].legacy_broadcast(self.output_type.clone(), self.output_axes.as_slice()) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError::invalid(error.to_string())),
        }
    }

    #[inline]
    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayType as crate::Type>::Identity>,
    ) -> Result<Self, TypeError> {
        Ok(Self { output_type: self.output_type.rename_identities(renaming)?, output_axes: self.output_axes.clone() })
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("output_type", &self.output_type)?;
            operation.field("output_axes", format_args!("{:?}", self.output_axes))
        })
    }
}

impl<C: Domain<Type = ArrayType, Value: LegacyBroadcast>> InterpretableOperation<C> for LegacyBroadcastOperation {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].legacy_broadcast(self.output_type.clone(), self.output_axes())?])
    }
}

impl<C: Context<Type = ArrayType, Operation: From<LegacyBroadcastOperation>>> PartiallyEvaluatableOperation<C>
    for LegacyBroadcastOperation
{
}

impl_differentiable_operation! {
    LegacyBroadcastOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType, Value: LegacyBroadcast, Operation: From<LegacyBroadcastOperation>>,
    {
        |operation, _context, _driver, inputs| {
            // Forward-mode differentiation rule for `LegacyBroadcastOperation`. Broadcasting is structural-linear, so
            // tangent follows the same axis mapping as the primal. A structural-zero input tangent remains structural
            // and acquires the primal output's tangent type.
            check_count!("input", inputs, 1, ProgramError);
            let primal =
                inputs[0].primal().legacy_broadcast(operation.output_type().clone(), operation.output_axes())?;
            let tangent_type = primal.r#type().tangent();
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(tangent_type),
                MaybeZero::Value(tangent) => {
                    MaybeZero::Value(tangent.legacy_broadcast(tangent_type, operation.output_axes())?)
                }
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<ArrayType>
            + From<LegacyBroadcastOperation>
            + From<ConvertElementTypeOperation>
            + From<ReduceOperation>
            + From<TransposeOperation>
            + From<LegacyReshapeOperation>
            + From<ReshardOperation>,
    {
        |operation, _context, _driver, inputs, outputs| {
            // Transposition rule for `LegacyBroadcastOperation`. The pullback of a broadcast is a sum-reduction over
            // output axis the input was replicated along (i.e., the axes of the target type that are not named in
            // `output_axes`, plus the mapped axes whose input extent is `1` stretched to a larger target extent).
            // After the reduction, the surviving axes are reordered into input-axis order when `output_axes` is not
            // monotonically increasing, and stretched unit axes are restored with a reshape so the cotangent matches
            // the input type exactly. Symbolic-zero cotangents propagate unchanged, and an input with no cotangent
            // space receives the structural zero of that space.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            let input_cotangent_type = inputs[0].r#type().cotangent();
            if input_cotangent_type.is_zero_space() {
                return Ok(vec![MaybeZero::Zero(input_cotangent_type)]);
            }
            let MaybeZero::Value(cotangent) = &outputs[0] else {
                return Ok(vec![MaybeZero::Zero(input_cotangent_type)]);
            };
            Ok(vec![MaybeZero::Value(
                cotangent.unalign_cotangent_along(&input_cotangent_type, operation.output_axes())?,
            )])
        }
    },
}

impl<C: Context<Type = ArrayType, Value: LegacyBroadcast>> BatchableOperation<C> for LegacyBroadcastOperation {
    fn batch<D: BatchingDriver<C>>(
        &self,
        _context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        match inputs[0].batch_axis_position() {
            None => {
                // A replicated input has no mapped axis to lift, so the original broadcast remains replicated.
                let output_value =
                    inputs[0].value().legacy_broadcast(self.output_type().clone(), self.output_axes())?;
                Ok(vec![ArrayBatch::replicated(output_value)])
            }
            Some(batch_axis) => {
                // Insert the mapped axis at the same physical output position and shift every existing broadcast-axis
                // mapping at or after that position around it.
                let axis_size = ArrayBatch::common_batch_size(inputs)?.expect("a mapped input pins the batch size");
                let mut output_type =
                    self.output_type().with_inserted_dimension(batch_axis, Dimension::Static(axis_size))?;
                let mut output_axes = self
                    .output_axes()
                    .iter()
                    .map(|&output_axis| if output_axis >= batch_axis { output_axis + 1 } else { output_axis })
                    .collect::<Vec<_>>();
                output_axes.insert(batch_axis, batch_axis);
                let axis_sharding = ArrayBatch::sharding_for_inputs(inputs)?;
                let output_sharding = self.output_type().sharding().cloned();
                let input_mesh = inputs.iter().find_map(|input| {
                    input.batch_axis_position()?;
                    input.r#type().sharding().map(|sharding| sharding.mesh().clone())
                });
                let output_sharding = match (output_sharding, input_mesh) {
                    (Some(sharding), _) => Some(sharding),
                    (None, Some(mesh)) if !matches!(&axis_sharding, ShardingDimension::Replicated) => {
                        Some(Sharding::replicated(mesh, self.output_type().rank()))
                    }
                    (None, None) => None,
                    (None, Some(_)) => None,
                };
                if let Some(sharding) = output_sharding {
                    output_type.sharding = Some(
                        sharding
                            .with_inserted_dimension(batch_axis, axis_sharding)
                            .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?,
                    );
                }
                let output_value = inputs[0].value().legacy_broadcast(output_type.clone(), output_axes.as_slice())?;
                Ok(vec![ArrayBatch::new(output_type, output_value, BatchAxis::from_position(batch_axis))?])
            }
        }
    }
}

/// Canonical operation name for [`DynamicBroadcastOperation`].
#[doc(hidden)]
pub const DYNAMIC_BROADCAST_OPERATION_NAME: &str = "dynamic_broadcast";

/// Legacy homogeneous [`Operation`] that broadcasts an array to dimensions packed into a runtime integer array.
/// Refer to the documentation of [`LegacyDynamicBroadcast`] for more information.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DynamicBroadcastOperation {
    /// Declared output [`ArrayType`], whose dynamic dimensions are refined by the runtime dimensions operand.
    output_type: ArrayType,

    /// Output dimension corresponding to each input dimension.
    output_axes: Vec<usize>,
}

impl DynamicBroadcastOperation {
    /// Creates a new [`DynamicBroadcastOperation`] with the supplied output type and output axes.
    #[inline]
    pub fn new(output_type: ArrayType, output_axes: Vec<usize>) -> Self {
        Self { output_type, output_axes }
    }

    /// Returns the declared output [`ArrayType`].
    #[inline]
    pub fn output_type(&self) -> &ArrayType {
        &self.output_type
    }

    /// Returns the output dimension corresponding to each input dimension.
    #[inline]
    pub fn output_axes(&self) -> &[usize] {
        self.output_axes.as_slice()
    }
}

impl Display for DynamicBroadcastOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for DynamicBroadcastOperation {
    #[inline]
    fn name(&self) -> &'static str {
        DYNAMIC_BROADCAST_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        let mut validation_output_dimensions = self.output_type.shape().dimensions().to_vec();
        if self.output_axes.len() == input_types[0].rank() {
            for (input_axis, &output_axis) in self.output_axes.iter().enumerate() {
                if output_axis < validation_output_dimensions.len() {
                    let input_dimension = input_types[0].dimension(input_axis);
                    if matches!(input_dimension, Dimension::Dynamic(_))
                        || matches!(validation_output_dimensions[output_axis], Dimension::Dynamic(_))
                    {
                        // The runtime dimensions operand determines this extent. Reuse the input extent here so
                        // that `ArrayType::broadcast` can validate the operation's structural broadcast contract.
                        validation_output_dimensions[output_axis] = input_dimension;
                    }
                }
            }

            for (output_axis, output_dimension) in validation_output_dimensions.iter_mut().enumerate() {
                if !self.output_axes.contains(&output_axis) && matches!(output_dimension, Dimension::Dynamic(_)) {
                    // Any positive representative extent is valid because the runtime dimensions operand supplies
                    // the actual replication count for an unmapped output axis.
                    *output_dimension = Dimension::Static(1);
                }
            }
        }

        input_types[0]
            .legacy_broadcast(
                self.output_type.clone().with_shape(Shape::new(validation_output_dimensions)),
                self.output_axes(),
            )
            .map_err(|error| match error {
                ProgramError::Type(error) => error,
                error => TypeError::invalid(error.to_string()),
            })?;

        let dimensions_type = &input_types[1];
        if dimensions_type.rank() != 1 {
            return Err(TypeError::invalid(format!(
                "dynamic broadcast output dimensions must have rank 1 but have rank {}",
                dimensions_type.rank(),
            )));
        }

        if !dimensions_type.data_type().is_integer() {
            return Err(TypeError::invalid(format!(
                "dynamic broadcast output dimensions must have integer elements but have data type {}",
                dimensions_type.data_type(),
            )));
        }

        if dimensions_type.dimension(0) != Dimension::Static(self.output_type.rank()) {
            return Err(TypeError::invalid(format!(
                "dynamic broadcast output dimensions has length {} but output has rank {}",
                dimensions_type.dimension(0),
                self.output_type.rank(),
            )));
        }

        Ok(vec![self.output_type.clone()])
    }

    #[inline]
    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayType as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        Ok(Self { output_type: self.output_type.rename_identities(renaming)?, output_axes: self.output_axes.clone() })
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("output_type", &self.output_type)?;
            operation.field("output_axes", format_args!("{:?}", self.output_axes))
        })
    }
}

impl<C: Domain<Type = ArrayType, Value: LegacyDynamicBroadcast>> InterpretableOperation<C>
    for DynamicBroadcastOperation
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].legacy_dynamic_broadcast(&inputs[1], self.output_type.clone(), self.output_axes())?])
    }
}

impl<C: Context<Type = ArrayType, Operation: From<DynamicBroadcastOperation>>> PartiallyEvaluatableOperation<C>
    for DynamicBroadcastOperation
{
}

impl_differentiable_operation! {
    DynamicBroadcastOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType, Value: LegacyDynamicBroadcast, Operation: From<DynamicBroadcastOperation>>,
    {
        |operation, _context, _driver, inputs| {
            check_count!("input", inputs, 2, ProgramError);
            let primal = inputs[0].primal().legacy_dynamic_broadcast(
                inputs[1].primal(),
                operation.output_type().clone(),
                operation.output_axes(),
            )?;
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
                MaybeZero::Value(tangent) => MaybeZero::Value(tangent.legacy_dynamic_broadcast(
                    inputs[1].primal(),
                    primal.r#type().tangent(),
                    operation.output_axes(),
                )?),
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<ArrayType>
            + From<LegacyBroadcastOperation>
            + From<ConvertElementTypeOperation>
            + From<ReduceOperation>
            + From<TransposeOperation>
            + From<LegacyReshapeOperation>
            + From<ReshardOperation>,
    {
        |operation, _context, _driver, inputs, outputs| {
            check_count!("input", inputs, 2, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            let input_cotangent_type = inputs[0].r#type().cotangent();
            if operation
                .output_axes()
                .iter()
                .copied()
                .enumerate()
                .any(|(input_axis, _)| matches!(input_cotangent_type.dimension(input_axis), Dimension::Dynamic(_)))
            {
                return Err(ProgramError::UnsupportedOperation {
                    message: "transposing a dynamic broadcast with a runtime-dependent input dimension is unsupported"
                        .to_string(),
                }
                .into());
            }
            let value_contribution = if input_cotangent_type.is_zero_space() || outputs[0].is_zero() {
                MaybeZero::Zero(input_cotangent_type)
            } else {
                let MaybeZero::Value(cotangent) = &outputs[0] else {
                    unreachable!("the structural-zero cotangent case was handled above")
                };
                MaybeZero::Value(cotangent.unalign_cotangent_along(&input_cotangent_type, operation.output_axes())?)
            };
            Ok(vec![value_contribution, MaybeZero::Zero(inputs[1].r#type().cotangent())])
        }
    },
}

impl<C: Context<Type = ArrayType, Value: LegacyDynamicBroadcast + Concatenate> + Fill<Scalar, C::Value>>
    BatchableOperation<C> for DynamicBroadcastOperation
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 2, ProgramError);
        if !inputs[1].batch_axis().is_replicated() {
            return Err(BatchingError::UnsupportedOperation {
                message: "dynamic broadcast does not support batched output dimensions".to_string(),
            });
        }
        let Some(input_batch_axis) = inputs[0].batch_axis_position() else {
            let output = inputs[0].value().legacy_dynamic_broadcast(
                inputs[1].value(),
                self.output_type.clone(),
                self.output_axes(),
            )?;
            return Ok(vec![ArrayBatch::replicated(output)]);
        };

        // Canonicalize the result's mapped axis at the front without moving the packed input. The original logical
        // mappings shift by one, while the packed input's existing mapped axis maps directly to output axis zero.
        let axis_size = context.axis_size();
        let mut output_type = self.output_type().with_inserted_dimension(0, Dimension::Static(axis_size))?;
        let mut output_axes = self.output_axes().iter().map(|&output_axis| output_axis + 1).collect::<Vec<_>>();
        output_axes.insert(input_batch_axis, 0);

        let axis_sharding = context.axis_sharding();
        let input_mesh = inputs[0].r#type().sharding().map(|sharding| sharding.mesh().clone());
        let base_sharding = match (self.output_type().sharding().cloned(), input_mesh) {
            (Some(sharding), _) => Some(sharding),
            (None, Some(mesh)) if !matches!(axis_sharding, ShardingDimension::Replicated) => {
                Some(Sharding::replicated(mesh, self.output_type().rank()))
            }
            (None, None) => None,
            (None, Some(_)) => None,
        };
        if let Some(sharding) = base_sharding {
            output_type.sharding = Some(
                sharding
                    .with_inserted_dimension(0, axis_sharding.clone())
                    .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?,
            );
        }

        let dimensions_type = inputs[1].r#type().into_owned();
        let inserted_type =
            dimensions_type.clone().with_shape(Shape::new(vec![Dimension::Static(1)])).with_layout(None);
        let dimension_overflow = || ProgramError::InvalidArgument {
            message: format!("dimension size {axis_size} does not fit in data type {}", dimensions_type.data_type(),),
        };
        let axis_size = match dimensions_type.data_type() {
            DataType::I8 => Scalar::I8(i8::try_from(axis_size).map_err(|_| dimension_overflow())?),
            DataType::I16 => Scalar::I16(i16::try_from(axis_size).map_err(|_| dimension_overflow())?),
            DataType::I32 => Scalar::I32(i32::try_from(axis_size).map_err(|_| dimension_overflow())?),
            DataType::I64 => Scalar::I64(i64::try_from(axis_size).map_err(|_| dimension_overflow())?),
            DataType::U8 => Scalar::U8(u8::try_from(axis_size).map_err(|_| dimension_overflow())?),
            DataType::U16 => Scalar::U16(u16::try_from(axis_size).map_err(|_| dimension_overflow())?),
            DataType::U32 => Scalar::U32(u32::try_from(axis_size).map_err(|_| dimension_overflow())?),
            DataType::U64 => Scalar::U64(u64::try_from(axis_size).map_err(|_| dimension_overflow())?),
            data_type => {
                return Err(ProgramError::InvalidArgument {
                    message: format!(
                        "dynamic broadcast output dimensions must have integer elements but have data type {data_type}"
                    ),
                }
                .into());
            }
        };
        let inserted = context.parent().fill(&inserted_type, axis_size)?;
        let output_dimensions = Concatenate::concatenate([&inserted, inputs[1].value()], 0)?;
        let output =
            inputs[0].value().legacy_dynamic_broadcast(&output_dimensions, output_type.clone(), &output_axes)?;
        Ok(vec![ArrayBatch::new(output_type, output, BatchAxis::from_position(0))?])
    }
}

/// Frozen homogeneous broadcast capability used only by the legacy array-only operation and transform language.
///
/// New program construction must use [`Broadcast`], whose output extents are explicit SSA operands. This capability
/// remains hidden from normal documentation only while its named Phase 4–9 consumers migrate.
#[doc(hidden)]
pub trait LegacyBroadcast: Sized {
    /// Broadcasts `self` to `output_type` using `output_axes`.
    ///
    /// # Parameters
    ///
    ///   - `output_type`: [`ArrayType`] of the output array.
    ///   - `output_axes`: Slice that contains, for each axis `i` of the input, the output axis that it maps to. This
    ///     slice must have length equal to the input's rank and contain distinct values in `0..output_type.rank()`.
    fn legacy_broadcast(&self, output_type: ArrayType, output_axes: &[usize]) -> Result<Self, ProgramError>;
}

impl LegacyBroadcast for ArrayType {
    fn legacy_broadcast(&self, output_type: ArrayType, output_axes: &[usize]) -> Result<ArrayType, ProgramError> {
        if self.data_type() != output_type.data_type() {
            return Err(TypeError::invalid(format!(
                "broadcasting input data type {} does not match output data type {}",
                self.data_type(),
                output_type.data_type(),
            ))
            .into());
        }

        if self.memory() != output_type.memory() {
            return Err(TypeError::invalid(format!(
                "broadcasting input memory {} does not match output memory {}",
                self.memory(),
                output_type.memory(),
            ))
            .into());
        }

        let input_rank = self.rank();
        let output_rank = output_type.rank();
        if output_axes.len() != input_rank {
            return Err(TypeError::invalid(format!(
                "broadcasting output axes has length {} but input has rank {}",
                output_axes.len(),
                input_rank,
            ))
            .into());
        }

        let mut seen = vec![false; output_rank];
        for (input_axis, &output_axis) in output_axes.iter().enumerate() {
            if output_axis >= output_rank {
                return Err(TypeError::invalid(format!(
                    "broadcasting `output_axes[{}] = {}` is out of bounds for output rank {}",
                    input_axis, output_axis, output_rank,
                ))
                .into());
            }
            if seen[output_axis] {
                return Err(TypeError::invalid(format!(
                    "broadcasting output axes map two input axes to output axis {output_axis}",
                ))
                .into());
            }
            seen[output_axis] = true;

            let input_dimension = self.dimension(input_axis);
            let output_dimension = output_type.dimension(output_axis);
            match (input_dimension, output_dimension.clone()) {
                // Identical sizes always map through, including identical dynamic sizes.
                (input_dimension, output_dimension) if input_dimension == output_dimension => {}
                // A static size-1 input dimension is replicated to match any static output extent. Expanding it
                // into a dynamic output dimension is unsupported because the replication count is unknown.
                (Dimension::Static(1), Dimension::Static(_)) => {}
                (Dimension::Static(1), Dimension::Dynamic(_)) => {
                    return Err(TypeError::invalid(format!(
                        "broadcasting cannot expand input axis {} of size 1 into dynamic output size {}",
                        input_axis, output_dimension,
                    ))
                    .into());
                }
                (Dimension::Static(input_size), Dimension::Static(output_size)) => {
                    return Err(TypeError::invalid(format!(
                        "broadcasting input axis {} has size {}, which is neither {} nor 1",
                        input_axis, input_size, output_size,
                    ))
                    .into());
                }
                // All remaining combinations pair a dynamic size with a mismatched size on the other side.
                (input_dimension, output_dimension) => {
                    return Err(TypeError::invalid(format!(
                        "broadcasting input axis {input_axis} has size {input_dimension} but the output has size \
                            {output_dimension}; a dynamic dimension only broadcasts to an identical dynamic dimension",
                    ))
                    .into());
                }
            }
        }

        // Output axes that no input axis maps to replicate the input along that axis, which requires a known
        // replication count and is therefore unsupported for dynamic output dimensions.
        for (output_axis, mapped) in seen.iter().enumerate() {
            let output_dimension = output_type.dimension(output_axis);
            if !mapped && matches!(output_dimension, Dimension::Dynamic(_)) {
                return Err(TypeError::invalid(format!(
                    "broadcasting cannot replicate the input into unmapped dynamic output axis {} of size {}",
                    output_axis, output_dimension,
                ))
                .into());
            }
        }
        Ok(output_type)
    }
}

/// Infers the result of the canonical mixed broadcast from its explicit output extent types.
pub(crate) fn infer_explicit_broadcast_output_type(
    input: &ArrayType,
    output_shape: Shape,
    operation: &BroadcastOperation,
) -> Result<ArrayType, TypeError> {
    let output_rank = output_shape.rank();
    if operation.output_axes().len() != input.rank() {
        return Err(TypeError::invalid(format!(
            "broadcasting output axes has length {} but input has rank {}",
            operation.output_axes().len(),
            input.rank(),
        )));
    }

    let mut mapped_output_axes = vec![false; output_rank];
    for (input_axis, &output_axis) in operation.output_axes().iter().enumerate() {
        if output_axis >= output_rank {
            return Err(TypeError::invalid(format!(
                "broadcasting `output_axes[{}] = {}` is out of bounds for output rank {}",
                input_axis, output_axis, output_rank,
            )));
        }
        if mapped_output_axes[output_axis] {
            return Err(TypeError::invalid(format!(
                "broadcasting output axes map two input axes to output axis {output_axis}",
            )));
        }
        mapped_output_axes[output_axis] = true;

        let input_dimension = input.dimension(input_axis);
        let output_dimension = output_shape.dimensions()[output_axis].clone();
        match (input_dimension, output_dimension) {
            // Equal axes preserve their extent, including the identity of an equal dynamic extent.
            (input_dimension, output_dimension) if input_dimension == output_dimension => {}
            // The explicit output extent operand supplies the runtime replication count that the legacy
            // metadata-only operation lacked, so a unit input may expand to either a static or dynamic extent.
            (Dimension::Static(1), _) => {}
            (Dimension::Static(input_extent), Dimension::Static(output_extent)) => {
                return Err(TypeError::invalid(format!(
                    "broadcasting input axis {} has size {}, which is neither {} nor 1",
                    input_axis, input_extent, output_extent,
                )));
            }
            (input_dimension, output_dimension) => {
                return Err(TypeError::invalid(format!(
                    "broadcasting input axis {input_axis} has size {input_dimension} but the output has size \
                        {output_dimension}; a dynamic dimension only broadcasts to an identical dynamic dimension",
                )));
            }
        }
    }

    let output_sharding = match operation.output_sharding() {
        Some(sharding) => Some(sharding.clone()),
        None => input
            .sharding()
            .map(|sharding| sharding.with_broadcasted_dimensions(output_rank, operation.output_axes()))
            .transpose()
            .map_err(|error| TypeError::invalid(error.to_string()))?,
    };

    if input.shape() == &output_shape && operation.output_axes().iter().copied().eq(0..input.rank()) {
        return input.clone().with_sharding(output_sharding).map_err(|error| TypeError::invalid(error.to_string()));
    }

    ArrayType::new(input.data_type(), output_shape)
        .with_memory(input.memory())
        .with_sharding(output_sharding)
        .map_err(|error| TypeError::invalid(error.to_string()))
}

impl<V: Value<Type = ArrayType, DispatchDomain: Context<Type = ArrayType, Operation: From<LegacyBroadcastOperation>>>>
    LegacyBroadcast for V
{
    #[inline]
    fn legacy_broadcast(&self, output_type: ArrayType, output_axes: &[usize]) -> Result<Self, ProgramError> {
        let output_type = self.r#type().legacy_broadcast(output_type, output_axes)?;
        if self.r#type().as_ref() == &output_type && output_axes.iter().copied().eq(0..output_type.rank()) {
            return Ok(self.clone());
        }
        Ok(self
            .dispatch_domain()
            .bind(
                LegacyBroadcastOperation::new(output_type, output_axes.to_vec()),
                Vec::new(),
                std::slice::from_ref(self),
            )?
            .remove(0))
    }
}

/// Broadcasts an array using one explicit first-class dimension value per output axis.
///
/// Exact dimension values describe static axes and computed dimension values describe dynamic axes. Both forms bind
/// the same [`BroadcastOperation`]; backend lowering chooses the appropriate static, bounded, or dynamic
/// representation from the inferred result type.
///
/// Exact host sizes can use [`Broadcast::broadcast_to_sizes`]:
///
/// ```rust
/// use ryft_core::operations::manipulation::Broadcast;
/// use ryft_core::{Array, ArrayProgramValue};
///
/// let input = ArrayProgramValue::Array(Array::vector(vec![1.0, 2.0, 3.0]));
/// let output = input.broadcast_to_sizes(&[2, 3]).unwrap();
/// assert_eq!(
///     output,
///     ArrayProgramValue::Array(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0])),
/// );
/// ```
///
/// Computed or input dimensions remain ordinary SSA operands:
///
/// ```rust
/// use ryft_core::operations::manipulation::Broadcast;
/// use ryft_core::{
///     Array, ArrayProgramOperation, ArrayProgramType, ArrayProgramValue, ArrayType, DataType,
///     DimensionBounds, DimensionType, DimensionVariable, TracingContext, Typed,
///     StagingContext,
/// };
///
/// type C = TracingContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;
///
/// let context = C::new();
/// let scalar = context.input(ArrayProgramType::Array(ArrayType::scalar(DataType::F32)));
/// let extent = context.input(ArrayProgramType::Dimension(DimensionType::new(DimensionVariable::new(
///     "extent",
///     DimensionBounds::new(1, Some(9)).unwrap(),
/// ))));
/// let output = scalar.broadcast_to(&[extent]).unwrap();
/// assert_eq!(output.r#type().to_string(), "f32[extent]");
/// ```
pub trait Broadcast: Value<Type = ArrayProgramType> + DimensionSize + Sized {
    /// Broadcasts `self` using an explicit input-to-output axis mapping and one first-class value per output extent.
    ///
    /// # Parameters
    ///
    ///   - `output_dimensions`: Ordered output extents, one per result axis.
    ///   - `output_axes`: Output axis corresponding to each input axis.
    fn broadcast(&self, output_dimensions: &[Self], output_axes: &[usize]) -> Result<Self, ProgramError> {
        self.broadcast_with_output_sharding(output_dimensions, output_axes, None)
    }

    /// Broadcasts `self` with an explicit requested output sharding.
    fn broadcast_with_output_sharding(
        &self,
        output_dimensions: &[Self],
        output_axes: &[usize],
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError>;

    /// Broadcasts `self` to `output_dimensions` using NumPy-style right alignment.
    fn broadcast_to(&self, output_dimensions: &[Self]) -> Result<Self, ProgramError> {
        let r#type = self.r#type();
        let input_type = <&ArrayType>::try_from(r#type.as_ref())?;
        let offset = output_dimensions.len().checked_sub(input_type.rank()).ok_or_else(|| {
            TypeError::invalid(format!(
                "cannot broadcast rank-{} input to {} output dimensions",
                input_type.rank(),
                output_dimensions.len(),
            ))
        })?;
        let output_axes = (0..input_type.rank()).map(|axis| axis + offset).collect::<Vec<_>>();
        self.broadcast(output_dimensions, output_axes.as_slice())
    }

    /// Broadcasts `self` by prepending the supplied first-class leading dimensions.
    fn broadcast_leading(&self, leading_dimensions: &[Self]) -> Result<Self, ProgramError>
    where
        Self::DispatchDomain: Context<Type = ArrayProgramType>,
        <Self::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    {
        let r#type = self.r#type();
        let input_type = <&ArrayType>::try_from(r#type.as_ref())?;
        let mut output_dimensions = Vec::with_capacity(leading_dimensions.len() + input_type.rank());
        output_dimensions.extend_from_slice(leading_dimensions);
        for (axis, dimension) in input_type.shape().dimensions().iter().enumerate() {
            output_dimensions.push(match dimension {
                Dimension::Static(extent) => self.dispatch_domain().lift(DimensionValue::constant(*extent)?.into())?,
                Dimension::Dynamic(_) => self.dimension_size(axis)?,
            });
        }
        let output_axes = (0..input_type.rank()).map(|axis| axis + leading_dimensions.len()).collect::<Vec<_>>();
        self.broadcast(output_dimensions.as_slice(), output_axes.as_slice())
    }

    /// Broadcasts `self` to an exact static shape.
    fn broadcast_to_sizes(&self, output_sizes: &[usize]) -> Result<Self, ProgramError>
    where
        Self::DispatchDomain: Context<Type = ArrayProgramType>,
        <Self::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    {
        let output_dimensions = output_sizes
            .iter()
            .map(|extent| self.dispatch_domain().lift(DimensionValue::constant(*extent)?.into()))
            .collect::<Result<Vec<_>, _>>()?;
        self.broadcast_to(output_dimensions.as_slice())
    }

    /// Broadcasts `self` by prepending exact static leading dimensions.
    fn broadcast_leading_sizes(&self, leading_sizes: &[usize]) -> Result<Self, ProgramError>
    where
        Self::DispatchDomain: Context<Type = ArrayProgramType>,
        <Self::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    {
        let context = self.dispatch_domain();
        let leading_dimensions = leading_sizes
            .iter()
            .map(|extent| context.lift(DimensionValue::constant(*extent)?.into()))
            .collect::<Result<Vec<_>, _>>()?;
        self.broadcast_leading(leading_dimensions.as_slice())
    }
}

impl<
    V: Value<
            Type = ArrayProgramType,
            DispatchDomain: Context<
                Type = ArrayProgramType,
                Operation: From<BroadcastOperation> + From<DimensionSizeOperation>,
            >,
        >,
> Broadcast for V
where
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
{
    #[inline]
    fn broadcast_with_output_sharding(
        &self,
        output_dimensions: &[Self],
        output_axes: &[usize],
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError> {
        let input_type = self.r#type();
        let input_type = <&ArrayType>::try_from(input_type.as_ref())?;
        let output_shape = Shape::new(
            output_dimensions
                .iter()
                .map(Typed::r#type)
                .map(|r#type| <&DimensionType>::try_from(r#type.as_ref()).map(DimensionType::to_dimension))
                .collect::<Result<Vec<_>, _>>()?,
        );
        let operation = BroadcastOperation::new(output_axes.to_vec()).with_output_sharding(output_sharding);
        let output_type = infer_explicit_broadcast_output_type(input_type, output_shape, &operation)?;
        if input_type == &output_type {
            return Ok(self.clone());
        }

        let mut inputs = Vec::with_capacity(output_dimensions.len() + 1);
        inputs.push(self.clone());
        inputs.extend_from_slice(output_dimensions);
        Ok(self.dispatch_domain().bind(operation, Vec::new(), inputs.as_slice())?.remove(0))
    }
}

/// Frozen homogeneous capability for the legacy packed integer-array broadcast representation.
#[doc(hidden)]
pub trait LegacyDynamicBroadcast: Sized {
    /// Broadcasts `self` using `output_axes`, refining `output_type` with the integer values in `output_dimensions`.
    ///
    /// # Parameters
    ///
    ///   - `output_dimensions`: Rank-one integer array containing one concrete size per output dimension.
    ///   - `output_type`: Declared output type refined by the runtime dimensions.
    ///   - `output_axes`: Output dimension corresponding to each input dimension.
    fn legacy_dynamic_broadcast(
        &self,
        output_dimensions: &Self,
        output_type: ArrayType,
        output_axes: &[usize],
    ) -> Result<Self, ProgramError>;
}

impl<V: Value<Type = ArrayType, DispatchDomain: Context<Type = ArrayType, Operation: From<DynamicBroadcastOperation>>>>
    LegacyDynamicBroadcast for V
{
    #[inline]
    fn legacy_dynamic_broadcast(
        &self,
        output_dimensions: &Self,
        output_type: ArrayType,
        output_axes: &[usize],
    ) -> Result<Self, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(
                DynamicBroadcastOperation::new(output_type, output_axes.to_vec()),
                Vec::new(),
                &[self.clone(), output_dimensions.clone()],
            )?
            .remove(0))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::dimensions::DimensionValue;
    use crate::contexts::EagerContext;
    use crate::differentiation::forward::{DifferentiableOperation, jvp};
    use crate::differentiation::reverse::TransposableOperation;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::Typed;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing::TracingContext;
    use crate::types::{DataType, DimensionBounds, DimensionVariable, Layout, Memory, StridedLayout};

    use super::LegacyBroadcastOperation as HomogeneousBroadcastOperation;
    use super::*;

    #[test]
    fn test_explicit_broadcast() {
        let operation = BroadcastOperation::new(vec![1]);
        assert_eq!(operation.name(), BROADCAST_OPERATION_NAME);
        assert_eq!(operation.to_string(), "broadcast [output_axes=[1]]");

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![4])))
            .with_memory(Memory::Host { pinned: true });
        let two = DimensionValue::constant(2).unwrap();
        let dynamic_extent =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap()));
        assert_eq!(
            operation.infer_output_types(
                &[input_type.into(), two.r#type().clone().into(), dynamic_extent.clone().into(),],
                &[],
            ),
            Ok(vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(dynamic_extent.variable().clone()),]),
                )
                .with_memory(Memory::Host { pinned: true })
                .into()
            ]),
        );

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![4])))
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap()
            .with_memory(Memory::Host { pinned: true });
        let three = DimensionValue::constant(3).unwrap();
        assert_eq!(
            operation.infer_output_types(
                &[input_type.into(), two.r#type().clone().into(), three.r#type().clone().into()],
                &[],
            ),
            Ok(vec![
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
                    .with_sharding(
                        Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],)
                            .unwrap(),
                    )
                    .unwrap()
                    .with_memory(Memory::Host { pinned: true })
                    .into(),
            ]),
        );

        assert_eq!(
            BroadcastOperation::new(vec![0, 0]).infer_output_types(
                &[
                    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1), Dimension::Static(1)]),).into(),
                    two.r#type().clone().into(),
                ],
                &[],
            ),
            Err(TypeError::invalid("broadcasting output axes map two input axes to output axis 0")),
        );
        assert_eq!(
            operation.infer_output_types(&[two.r#type().clone().into()], &[]),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        assert_eq!(
            operation.infer_output_types(
                &[ArrayType::scalar(DataType::F32).into(), ArrayType::scalar(DataType::I64).into()],
                &[],
            ),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );
    }

    #[test]
    fn test_broadcast() {
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let operation = HomogeneousBroadcastOperation::new(output_type.clone(), vec![1]);

        // Operation identity and accessors.
        assert_eq!(operation.name(), BROADCAST_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "broadcast [output_type=f64[2, 3], output_axes=[1]]");
        assert_eq!(*operation.output_type(), output_type);
        assert_eq!(operation.output_axes(), &[1]);

        // Type inference validates the axis mapping and returns the target type.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    input_types = [input_type.clone()],
                    output_types = [output_type.clone()],
                },
                {
                    input_types = [],
                    error = "expected 1 input but got 0",
                },
                {
                    input_types = [ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]))],
                    error = "broadcasting input data type f32 does not match output data type f64",
                },
                {
                    input_types = [output_type.clone()],
                    error = "broadcasting output axes has length 1 but input has rank 2",
                },
                {
                    input_types = [ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))],
                    error = "broadcasting input axis 0 has size 2, which is neither 3 nor 1",
                },
            ],
        );

        // Interpretation replicates the payload along the added axis.
        let input = Array::vector(vec![1.0, 2.0, 3.0]);
        let output = operation
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(*output[0].r#type(), output_type);
        assert_eq!(output[0].to_f64s(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        // Invalid interpreter arity reports the exact program error.
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // Program rendering uses the canonical operation name and includes the captured metadata.
        let mut builder = ProgramBuilder::<Array, HomogeneousBroadcastOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_output = builder.add_instruction(operation.clone(), Vec::new(), vec![program_input]).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![program_output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[2, 3] = broadcast [output_type=f64[2, 3], output_axes=[1]] %0
                in (%1)
            "}
            .trim_end(),
        );

        // Check standard partial evaluation with known and residual operands.
        let input = Array::vector(vec![1.0, 2.0, 3.0]);
        let expected = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = operation.clone(),
            cases = [
                {
                    inputs = [(@known, input.clone())],
                    outputs = [(@known, expected.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [(@unknown(type = input.r#type().into_owned(), replay = input.clone()))],
                    outputs = [(@residual, expected)],
                    residual_instructions = 1,
                },
            ],
        );

        // Batching preserves replicated inputs and lifts mapped inputs through the explicit output-axis mapping.
        check_operation_batching!(
            @exact,
            operation = HomogeneousBroadcastOperation::new(
                ArrayType::new(DataType::F64, Shape::new(vec![3.into(), 4.into()])),
                vec![0],
            ),
            axis_size = 2,
            cases = [
                {
                    inputs = [(@replicated, Array::vector(vec![1.0, 2.0, 3.0]))],
                    outputs = [(@replicated, Array::matrix(
                        3,
                        4,
                        vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0],
                    ))],
                },
                {
                    inputs = [(@mapped(axis = 0), Array::matrix(
                        2,
                        3,
                        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    ))],
                    outputs = [(@mapped(axis = 0), Array::from_f64s(
                        ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into(), 4.into()])),
                        vec![
                            1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
                            4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0,
                        ],
                    ))],
                },
            ],
        );

        // Broadcasting is structural-linear meaning that its JVP broadcasts both the primal and its tangent.
        check_operation_differentiation!(
            @approx(step = 0.125, epsilon = 1e-9),
            operation = HomogeneousBroadcastOperation::new(
                ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 2.into()])),
                vec![1],
            ),
            cases = [{
                primals = [Array::vector(vec![1.0, 2.0])],
                tangents = [Array::vector(vec![3.0, 4.0])],
                primal_outputs = [Array::matrix(2, 2, vec![1.0, 2.0, 1.0, 2.0])],
                tangent_outputs = [Array::matrix(2, 2, vec![3.0, 4.0, 3.0, 4.0])],
                jvp = indoc! {"
                    lambda %0:f64[2], %1:f64[2] .
                    let %2:f64[2, 2] = broadcast [output_type=f64[2, 2], output_axes=[1]] %0
                        %3:f64[2, 2] = broadcast [output_type=f64[2, 2], output_axes=[1]] %1
                    in (%2, %3)
                "},
            }],
        );

        // The pullback sums the output cotangent over every newly replicated output axis.
        check_operation_transposition!(
            @exact,
            operation = operation,
            cases = [{
                inputs = [(@linear(type = ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![Dimension::Static(3)]),
                )))],
                output_cotangents = [Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])],
                input_cotangents = [Array::vector(vec![5.0, 7.0, 9.0])],
                pullback = indoc! {"
                    lambda %0:f64[2, 3] .
                    let %1:f64[3] = reduce_sum [axes=[0]] %0
                    in (%1)
                "},
            }],
        );
    }

    #[test]
    fn test_broadcast_with_sharding() {
        // Explicit mapped-axis sharding remains attached to the lifted batch dimension.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let logical_output_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(4)]))
                .with_sharding(Sharding::replicated(mesh.clone(), 2))
                .unwrap();
        let expected_output_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(3), Dimension::Static(4)]),
        )
        .with_sharding(
            Sharding::new(
                mesh,
                vec![
                    ShardingDimension::sharded(["x"]),
                    ShardingDimension::replicated(),
                    ShardingDimension::replicated(),
                ],
            )
            .unwrap(),
        )
        .unwrap();

        check_operation_batching!(
            @exact,
            operation = HomogeneousBroadcastOperation::new(logical_output_type, vec![0]),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::from_f64s(
                    input_type,
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                ))],
                outputs = [(@mapped(axis = 0), Array::from_f64s(
                    expected_output_type,
                    vec![
                        1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
                        4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0,
                    ],
                ))],
            }],
        );

        // A mapped physical axis retains its mesh placement even when the logical operation result has no sharding.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let expected_output_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(3), Dimension::Static(4)]),
        )
        .with_sharding(
            Sharding::new(
                mesh,
                vec![
                    ShardingDimension::sharded(["x"]),
                    ShardingDimension::replicated(),
                    ShardingDimension::replicated(),
                ],
            )
            .unwrap(),
        )
        .unwrap();
        check_operation_batching!(
            @exact,
            operation = HomogeneousBroadcastOperation::new(
                ArrayType::new(DataType::F64, Shape::new(vec![3.into(), 4.into()])),
                vec![0],
            ),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::from_f64s(
                    input_type,
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                ))],
                outputs = [(@mapped(axis = 0), Array::from_f64s(
                    expected_output_type,
                    vec![
                        1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
                        4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0,
                    ],
                ))],
            }],
        );

        // A non-monotonic axis mapping keeps an explicitly sharded mapped axis at the beginning, middle, and end of
        // the physical result, even though the logical output itself has no sharding annotation.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);
        for batch_axis in 0..3 {
            let mut input_dimensions = vec![Dimension::Static(2), Dimension::Static(3)];
            input_dimensions.insert(batch_axis, Dimension::Static(2));
            let mut input_sharding = vec![ShardingDimension::replicated(); 3];
            input_sharding[batch_axis] = ShardingDimension::sharded(["x"]);
            let input_type = ArrayType::new(DataType::F64, Shape::new(input_dimensions))
                .with_sharding(Sharding::new(mesh.clone(), input_sharding).unwrap())
                .unwrap();
            let input = ArrayBatch::new(
                input_type.clone(),
                Array::from_f64s(input_type, vec![0.0; 12]),
                BatchAxis::from_position(batch_axis),
            )
            .unwrap();

            let output = HomogeneousBroadcastOperation::new(
                ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])),
                vec![1, 0],
            )
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap()
            .remove(0);
            let mut output_dimensions = vec![Dimension::Static(3), Dimension::Static(2)];
            output_dimensions.insert(batch_axis, Dimension::Static(2));
            let mut output_sharding = vec![ShardingDimension::replicated(); 3];
            output_sharding[batch_axis] = ShardingDimension::sharded(["x"]);
            assert_eq!(
                output.r#type().as_ref(),
                &ArrayType::new(DataType::F64, Shape::new(output_dimensions))
                    .with_sharding(Sharding::new(mesh.clone(), output_sharding).unwrap())
                    .unwrap(),
            );
            assert_eq!(output.batch_axis(), BatchAxis::from_position(batch_axis));
        }

        // Differentiation derives the tangent target from the primal output, preserving promoted tangent types.
        let primal_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(2)]));
        let tangent_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let primal_output_type =
            ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let tangent_output_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let (primal, tangent) = jvp(
            |value| value.legacy_broadcast(primal_output_type.clone(), &[1]),
            Array::from_f64s(primal_type, vec![2.0, 4.0]),
            Array::from_f64s(tangent_type, vec![1.0, 3.0]),
        )
        .unwrap();
        assert_eq!(primal.r#type().as_ref(), &primal_output_type);
        assert_eq!(tangent.r#type().as_ref(), &tangent_output_type);
        assert_eq!(tangent.to_f64s(), vec![1.0, 3.0, 1.0, 3.0]);

        // Input axis 0 (size 2) maps to output axis 2 and input axis 1 (size 3) maps to output axis 0, so the pullback
        // must sum over output axis 1 and swap the surviving axes back into input order.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let output_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(3), Dimension::Static(4), Dimension::Static(2)]),
        );
        check_operation_transposition!(
            @exact,
            operation = HomogeneousBroadcastOperation::new(output_type.clone(), vec![2, 0]),
            cases = [{
                inputs = [(@linear(type = input_type))],
                output_cotangents = [Array::from_f64s(
                    output_type,
                    (0..24).map(|value| value as f64).collect(),
                )],
                input_cotangents = [Array::matrix(2, 3, vec![12.0, 44.0, 76.0, 16.0, 48.0, 80.0])],
            }],
        );

        // Input axis 0 has extent 1 stretched to 2 in the target, so the pullback sums over it and restores the unit
        // axis with a reshape.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1), Dimension::Static(3)]));
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        check_operation_transposition!(
            @exact,
            operation = HomogeneousBroadcastOperation::new(output_type, vec![0, 1]),
            cases = [{
                inputs = [(@linear(type = input_type))],
                output_cotangents = [Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])],
                input_cotangents = [Array::matrix(1, 3, vec![5.0, 7.0, 9.0])],
            }],
        );

        // Symbolic-zero cotangents remain symbolic and acquire the input's promoted cotangent type.
        let input_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(3)]));
        let output_type =
            ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let operation = HomogeneousBroadcastOperation::new(output_type.clone(), vec![1]);
        let input_cotangent_type = input_type.cotangent();
        let output_cotangent_type = output_type.cotangent();

        let mut context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let contributions = operation
            .transpose(
                &mut context,
                &EmptyRegionDriver,
                &[PartialValue::Unknown(input_type)],
                &[MaybeZero::Zero(output_cotangent_type)],
            )
            .unwrap();
        assert_eq!(contributions.len(), 1);
        assert!(contributions[0].is_zero());
        assert_eq!(contributions[0].r#type().as_ref(), &input_cotangent_type);

        // Inputs with no cotangent space receive the structural zero of that space rather than being rejected.
        let input_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(3)]));
        let output_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let operation = HomogeneousBroadcastOperation::new(output_type.clone(), vec![1]);
        let contributions = operation
            .transpose(
                &mut context,
                &EmptyRegionDriver,
                &[PartialValue::Unknown(input_type.clone())],
                &[MaybeZero::Zero(output_type.cotangent())],
            )
            .unwrap();
        assert_eq!(contributions.len(), 1);
        assert!(contributions[0].is_zero());
        assert_eq!(contributions[0].r#type().as_ref(), &input_type.cotangent());
    }

    #[test]
    fn test_dynamic_broadcast() {
        let batch = DimensionVariable::new("batch", DimensionBounds::unbounded());
        let width = DimensionVariable::new("width", DimensionBounds::non_negative(Some(4)).unwrap());
        let output_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(3), Dimension::Dynamic(width)]),
        );
        let operation = DynamicBroadcastOperation::new(output_type.clone(), vec![2, 1]);
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1), Dimension::Static(3)]));
        let dimensions_type = ArrayType::new(DataType::I64, Shape::new(vec![Dimension::Static(3)]));

        // Type inference validates both the broadcast mapping and the runtime dimension-vector contract.
        assert_eq!(operation.name(), DYNAMIC_BROADCAST_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "dynamic_broadcast [output_type=f64[batch, 3, width], output_axes=[2, 1]]");
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    input_types = [input_type.clone(), dimensions_type.clone()],
                    output_types = [output_type.clone()],
                },
                {
                    input_types = [input_type.clone()],
                    error = "expected 2 inputs but got 1",
                },
                {
                    input_types = [input_type.clone(), ArrayType::scalar(DataType::I64)],
                    error = "dynamic broadcast output dimensions must have rank 1 but have rank 0",
                },
                {
                    input_types = [
                        input_type.clone(),
                        ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])),
                    ],
                    error = "dynamic broadcast output dimensions must have integer elements but have data type f64",
                },
                {
                    input_types = [
                        input_type.clone(),
                        ArrayType::new(DataType::I64, Shape::new(vec![Dimension::Static(2)])),
                    ],
                    error = "dynamic broadcast output dimensions has length 2 but output has rank 3",
                },
            ],
        );

        // Runtime dimensions refine the declared dynamic type and drive arbitrary-axis eager broadcasting.
        let input = Array::matrix(1, 3, vec![1.0, 2.0, 3.0]);
        let dimensions = Array::from_f64s(dimensions_type.clone(), vec![2.0, 3.0, 2.0]);
        let output = input.legacy_dynamic_broadcast(&dimensions, output_type.clone(), &[2, 1]).unwrap();
        assert_eq!(*output.r#type(), ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into(), 2.into()])),);
        assert_eq!(output.to_f64s(), [1.0, 1.0, 2.0, 2.0, 3.0, 3.0].repeat(2));

        // Runtime sizes must be nonnegative and refine every static or bounded declared dimension.
        let negative_dimensions = Array::from_f64s(dimensions_type.clone(), vec![-1.0, 3.0, 2.0]);
        assert_eq!(
            input.legacy_dynamic_broadcast(&negative_dimensions, output_type.clone(), &[2, 1]),
            Err(ProgramError::Type(TypeError::invalid(
                "dynamic broadcast output dimension 0 has invalid value -1".to_string(),
            ))),
        );
        let out_of_bounds_dimensions = Array::from_f64s(dimensions_type.clone(), vec![2.0, 3.0, 5.0]);
        assert_eq!(
            input.legacy_dynamic_broadcast(&out_of_bounds_dimensions, output_type, &[2, 1]),
            Err(ProgramError::Type(TypeError::invalid(
                "dynamic broadcast runtime shape [2, 3, 5] does not refine declared output shape [batch, 3, width]"
                    .to_string(),
            ))),
        );

        // The value tangent follows the same runtime broadcast while the integer dimensions carry a structural zero.
        let tangent = Array::matrix(1, 3, vec![4.0, 5.0, 6.0]);
        let dimensions = Array::from_f64s(
            ArrayType::new(DataType::I64, Shape::new(vec![Dimension::Static(3)])),
            vec![2.0, 3.0, 2.0],
        );
        let output_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
                Dimension::Static(3),
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::non_negative(Some(4)).unwrap())),
            ]),
        );
        let outputs = DynamicBroadcastOperation::new(output_type, vec![2, 1])
            .jvp(
                &EagerContext::<Array, ArrayOperation<Array>>::new(),
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new(input, tangent).unwrap(),
                    DifferentiationDual::new_with_zero_tangent(dimensions),
                ],
            )
            .unwrap();
        assert_eq!(outputs[0].primal().to_f64s(), [1.0, 1.0, 2.0, 2.0, 3.0, 3.0].repeat(2));
        assert_eq!(outputs[0].tangent().as_value().unwrap().to_f64s(), [4.0, 4.0, 5.0, 5.0, 6.0, 6.0].repeat(2),);

        // Partial evaluation folds concrete dimensions and residualizes the two-operand operation when the value is
        // unknown, retaining the dimensions for replay.
        let input = Array::matrix(1, 3, vec![1.0, 2.0, 3.0]);
        let dimensions = Array::from_f64s(
            ArrayType::new(DataType::I64, Shape::new(vec![Dimension::Static(3)])),
            vec![2.0, 3.0, 2.0],
        );
        let output = input
            .legacy_dynamic_broadcast(
                &dimensions,
                ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![
                        Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
                        Dimension::Static(3),
                        Dimension::Dynamic(DimensionVariable::new(
                            "dynamic",
                            DimensionBounds::non_negative(Some(4)).unwrap(),
                        )),
                    ]),
                ),
                &[2, 1],
            )
            .unwrap();
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = DynamicBroadcastOperation::new(
                ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![
                        Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
                        Dimension::Static(3),
                        Dimension::Dynamic(DimensionVariable::new(
                            "dynamic",
                            DimensionBounds::non_negative(Some(4)).unwrap(),
                        )),
                    ]),
                ),
                vec![2, 1],
            ),
            cases = [
                {
                    inputs = [(@known, input.clone()), (@known, dimensions.clone())],
                    outputs = [(@known, output.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = input.r#type().into_owned(), replay = input.clone())),
                        (@known, dimensions.clone()),
                    ],
                    outputs = [(@residual, output)],
                    residual_instructions = 1,
                },
            ],
        );

        // Batching prepends a canonical mapped output axis to both the declared result and the runtime dimensions
        // vector. The dimensions vector itself is a shape parameter and therefore must remain replicated.
        let expected_type = ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 2.into(), 3.into(), 2.into()]));
        let expected_values = vec![
            1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 4.0, 4.0, 5.0,
            5.0, 6.0, 6.0,
        ];
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);
        let input = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 1.into(), 3.into()])),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        let input = ArrayBatch::new(input.r#type().into_owned(), input, BatchAxis::from_position(0)).unwrap();
        let batched = operation
            .batch(&context, &EmptyRegionDriver, &[input, ArrayBatch::replicated(dimensions.clone())])
            .unwrap()
            .remove(0);
        assert_eq!(batched.batch_axis(), BatchAxis::from_position(0));
        assert_eq!(batched.unbatched_type().shape(), operation.output_type().shape());
        assert_eq!(*batched.value().r#type(), expected_type);
        assert_eq!(batched.value().to_f64s(), expected_values);

        // A non-leading packed input axis maps directly to the canonical leading output axis without transposing the
        // input or changing the per-item result.
        let input = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![1.into(), 2.into(), 3.into()])),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        let input = ArrayBatch::new(input.r#type().into_owned(), input, BatchAxis::from_position(1)).unwrap();
        let batched = operation
            .batch(&context, &EmptyRegionDriver, &[input, ArrayBatch::replicated(dimensions.clone())])
            .unwrap()
            .remove(0);
        assert_eq!(batched.batch_axis(), BatchAxis::from_position(0));
        assert_eq!(*batched.value().r#type(), expected_type);
        assert_eq!(batched.value().to_f64s(), expected_values);

        // The canonical output axis uses the active transform's placement and the mapped input's mesh when the logical
        // output itself has no sharding annotation.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![1.into(), 2.into(), 3.into()]))
            .with_sharding(
                Sharding::new(
                    mesh.clone(),
                    vec![
                        ShardingDimension::replicated(),
                        ShardingDimension::sharded(["x"]),
                        ShardingDimension::replicated(),
                    ],
                )
                .unwrap(),
            )
            .unwrap();
        let input = ArrayBatch::new(
            input_type.clone(),
            Array::from_f64s(input_type, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            BatchAxis::from_position(1),
        )
        .unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
            .with_axis_sharding(ShardingDimension::sharded(["x"]));
        let batched = operation
            .batch(&context, &EmptyRegionDriver, &[input, ArrayBatch::replicated(dimensions.clone())])
            .unwrap()
            .remove(0);
        let expected_sharding = Sharding::new(
            mesh,
            vec![
                ShardingDimension::sharded(["x"]),
                ShardingDimension::replicated(),
                ShardingDimension::replicated(),
                ShardingDimension::replicated(),
            ],
        )
        .unwrap();
        assert_eq!(batched.r#type().sharding(), Some(&expected_sharding));
        assert_eq!(batched.value().r#type().sharding(), Some(&expected_sharding));

        let mapped_dimensions = ArrayBatch::new(
            dimensions_type.clone(),
            Array::from_f64s(dimensions_type, vec![2.0, 3.0, 2.0]),
            BatchAxis::from_position(0),
        )
        .unwrap();
        assert!(matches!(
            DynamicBroadcastOperation::new(
                ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![
                        Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
                        3.into(),
                        Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
                    ]),
                ),
                vec![2, 1],
            )
            .batch(
                &context,
                &EmptyRegionDriver,
                &[ArrayBatch::replicated(Array::matrix(1, 3, vec![1.0, 2.0, 3.0])), mapped_dimensions],
            ),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "dynamic broadcast does not support batched output dimensions",
        ));

        // The pullback reduces both the unmapped runtime axis and the mapped axis expanded from a singleton.
        // The integer shape operand contributes no cotangent.
        check_operation_transposition!(
            @exact,
            operation = DynamicBroadcastOperation::new(
                ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![
                        Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
                        Dimension::Static(3),
                        Dimension::Dynamic(DimensionVariable::new(
                            "dynamic",
                            DimensionBounds::non_negative(Some(4)).unwrap(),
                        )),
                    ]),
                ),
                vec![2, 1],
            ),
            cases = [{
                inputs = [
                    (@linear(type = ArrayType::new(
                        DataType::F64,
                        Shape::new(vec![Dimension::Static(1), Dimension::Static(3)]),
                    ))),
                    (@known, Array::from_f64s(
                        ArrayType::new(DataType::I64, Shape::new(vec![Dimension::Static(3)])),
                        vec![2.0, 3.0, 2.0],
                    )),
                ],
                output_cotangents = [Array::from_f64s(
                    ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into(), 2.into()])),
                    (1..=12).map(f64::from).collect(),
                )],
                input_cotangents = [Array::matrix(1, 3, vec![18.0, 26.0, 34.0])],
            }],
        );
    }

    #[test]
    fn test_array_type_broadcast() {
        // Type-level broadcasting validates arbitrary mappings without consuming the input type.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        assert_eq!(input_type.legacy_broadcast(output_type.clone(), &[1]), Ok(output_type.clone()));
        assert_eq!(
            input_type.legacy_broadcast(output_type.clone().with_memory(Memory::Host { pinned: true }), &[1]),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting input memory Device does not match output memory Host[Pinned]".to_string(),
            ))),
        );
        assert_eq!(
            input_type.legacy_broadcast(output_type.clone(), &[2]),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting `output_axes[0] = 2` is out of bounds for output rank 2".to_string(),
            ))),
        );
        assert_eq!(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(3)]))
                .legacy_broadcast(output_type, &[1, 1]),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting output axes map two input axes to output axis 1".to_string(),
            ))),
        );

        // A dynamic input dimension maps only to the identical dynamic dimension, while replicated output axes must
        // remain static because their replication counts need to be known.
        let batch = DimensionVariable::new("batch", DimensionBounds::unbounded());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(3)]));
        let output_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(2), Dimension::Static(3)]),
        );
        assert_eq!(input_type.legacy_broadcast(output_type.clone(), &[0, 2]), Ok(output_type));

        let unbounded = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("input", DimensionBounds::unbounded()))]),
        );
        assert_eq!(
            unbounded.legacy_broadcast(
                ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                        "output",
                        DimensionBounds::non_negative(Some(4)).unwrap(),
                    ))]),
                ),
                &[0],
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting input axis 0 has size input but the output has size output; a dynamic dimension \
                    only broadcasts to an identical dynamic dimension"
                    .to_string(),
            ))),
        );
        assert_eq!(
            unbounded.legacy_broadcast(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])), &[0],),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting input axis 0 has size input but the output has size 3; a dynamic dimension \
                    only broadcasts to an identical dynamic dimension"
                    .to_string(),
            ))),
        );
        assert_eq!(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1)]))
                .legacy_broadcast(unbounded.clone(), &[0]),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting cannot expand input axis 0 of size 1 into dynamic output size input".to_string(),
            ))),
        );
        assert_eq!(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])).legacy_broadcast(unbounded, &[0]),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting input axis 0 has size 3 but the output has size input; a dynamic dimension \
                    only broadcasts to an identical dynamic dimension"
                    .to_string(),
            ))),
        );
        assert_eq!(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])).legacy_broadcast(
                ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![
                        Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
                        Dimension::Static(3),
                    ])
                ),
                &[1],
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "broadcasting cannot replicate the input into unmapped dynamic output axis 0 of size dynamic"
                    .to_string(),
            ))),
        );
    }

    #[test]
    fn test_array_broadcast() {
        // Arbitrary axis mappings replicate the payload along every unmapped target axis.
        let target = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(3), Dimension::Static(2)]),
        );
        let output = Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).legacy_broadcast(target, &[0, 2]).unwrap();
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]);

        // Static unit axes stretch to the target extent, and empty target dimensions produce empty payloads.
        let target = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let output = Array::matrix(1, 3, vec![1.0, 2.0, 3.0]).legacy_broadcast(target, &[0, 1]).unwrap();
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let target = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0), Dimension::Static(2)]));
        let output = Array::vector(vec![1.0, 2.0]).legacy_broadcast(target, &[1]).unwrap();
        assert_eq!(output.to_f64s(), Vec::<f64>::new());

        // The eager backend primitive preserves leading and right-aligned axis mappings.
        let output = Array::vector(vec![1.0, 2.0, 3.0])
            .legacy_broadcast(ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into()])), &[1])
            .unwrap();
        assert_eq!(*output.r#type(), ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into()])));
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let output = Array::scalar(7.0)
            .legacy_broadcast(ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into()])), &[])
            .unwrap();
        assert_eq!(output.to_f64s(), vec![7.0; 6]);
        let output = Array::vector(vec![10.0, 20.0, 30.0])
            .legacy_broadcast(ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into()])), &[1])
            .unwrap();
        assert_eq!(output.to_f64s(), vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0]);
        assert_eq!(
            Array::scalar(1.0)
                .legacy_broadcast(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0)])), &[],),
            Ok(Array::from_f64s(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0)])), Vec::new(),)),
        );

        // Convenience broadcasts preserve placement metadata, project dimension shardings, and clear a physical
        // layout only when the shape changes. An exact identity preserves the complete type.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap()
            .with_memory(Memory::Host { pinned: true });
        let input = Array::from_f64s(input_type.clone(), vec![1.0, 2.0, 3.0]);
        let identity = input.legacy_broadcast(input_type.clone(), &[0]).unwrap();
        assert_eq!(*identity.r#type(), input_type);
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into()]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                    .unwrap(),
            )
            .unwrap()
            .with_memory(Memory::Host { pinned: true });
        let output = input.legacy_broadcast(output_type, &[1]).unwrap();
        assert_eq!(output.r#type().memory(), Memory::Host { pinned: true });
        assert_eq!(output.r#type().layout(), None);
        assert_eq!(
            output.r#type().sharding(),
            Some(
                &Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],)
                    .unwrap(),
            ),
        );

        // Oversized target shapes fail through checked element-count arithmetic instead of panicking or wrapping.
        assert_eq!(
            Array::scalar(1.0).legacy_broadcast(
                ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(usize::MAX), Dimension::Static(2)]),),
                &[],
            ),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "shape [{}, 2] element count does not fit in usize",
                usize::MAX,
            )))),
        );
    }
}
