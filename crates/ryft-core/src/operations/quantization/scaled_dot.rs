use super::*;

/// Canonical operation name for [`ScaledDotOperation`].
pub const SCALED_DOT_OPERATION_NAME: &str = "scaled_dot";

/// Primitive representing a generalized block-scaled dot product.
///
/// The element operands occupy the first two input positions. Present scale operands follow in left-then-right
/// order, as recorded by [`Self::has_lhs_scale`] and [`Self::has_rhs_scale`]. A scale has the same rank as its
/// element operand. Its noncontracting dimensions match the operand exactly, while every contracting scale
/// dimension divides the corresponding operand dimension with an integer ratio of at least two. Ratios are inferred
/// independently for every side and contracting dimension.
///
/// Semantically this operation expands each present scale to its operand shape, converts elements and scales to
/// `bf16`, multiplies them, and applies [`DotOperation`] with [`Self::dimensions`] and
/// [`Self::preferred_element_type`]. That definition is implemented once by [`scaled_dot_composition`] and is also
/// the decomposition of the XLA `xla.scaled_dot` composite. Like the corresponding JAX primitive, scaled dot is not
/// differentiable. Batching inserts one leading batch pair and shifts the existing dimension numbers, without a rank
/// ceiling.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScaledDotOperation {
    /// Contracting and batching dimension specification.
    dimensions: DotDimensionNumbers,

    /// Data type of the result and dot accumulation.
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
    ///   - `dimensions`: Contracting and batching dimensions of the generalized dot.
    ///   - `preferred_element_type`: Data type used for the result and dot accumulation. Compatibility with the
    ///     `bf16` dequantized operands is validated by the generalized dot contract.
    ///   - `has_lhs_scale`: Whether the input list includes a left scale after the two element operands.
    ///   - `has_rhs_scale`: Whether the input list includes a right scale after the optional left scale.
    #[inline]
    pub fn new(
        dimensions: DotDimensionNumbers,
        preferred_element_type: DataType,
        has_lhs_scale: bool,
        has_rhs_scale: bool,
    ) -> Self {
        Self { dimensions, preferred_element_type, has_lhs_scale, has_rhs_scale }
    }

    /// Returns the default dimension numbers for operands of `rank`.
    pub fn default_dimensions(rank: usize) -> Result<DotDimensionNumbers, TypeError> {
        if rank < 2 {
            return Err(TypeError::invalid(format!(
                "'{SCALED_DOT_OPERATION_NAME}' does not support rank-{rank} operands"
            )));
        }
        Ok(DotDimensionNumbers::new(vec![rank - 1], vec![rank - 2], (0..rank - 2).collect(), (0..rank - 2).collect()))
    }

    /// Returns the contracting and batching dimensions.
    #[inline]
    pub fn dimensions(&self) -> &DotDimensionNumbers {
        &self.dimensions
    }

    /// Returns the data type used for the result and dot accumulation.
    #[inline]
    pub fn preferred_element_type(&self) -> DataType {
        self.preferred_element_type
    }

    /// Returns whether the operation has a left scale input.
    #[inline]
    pub fn has_lhs_scale(&self) -> bool {
        self.has_lhs_scale
    }

    /// Returns whether the operation has a right scale input.
    #[inline]
    pub fn has_rhs_scale(&self) -> bool {
        self.has_rhs_scale
    }

    /// Returns the number of inputs encoded by this operation.
    #[inline]
    fn input_count(&self) -> usize {
        2 + usize::from(self.has_lhs_scale) + usize::from(self.has_rhs_scale)
    }

    /// Splits an input list into the two element operands and the optional scales.
    fn inputs<'a, T>(&self, inputs: &'a [T]) -> Result<(&'a T, &'a T, Option<&'a T>, Option<&'a T>), TypeError> {
        if inputs.len() != self.input_count() {
            return Err(TypeError::invalid(format!(
                "'{SCALED_DOT_OPERATION_NAME}' expects {} inputs but got {}",
                self.input_count(),
                inputs.len(),
            )));
        }
        let lhs = &inputs[0];
        let rhs = &inputs[1];
        let mut scale_index = 2;
        let lhs_scale = self.has_lhs_scale.then(|| {
            let scale = &inputs[scale_index];
            scale_index += 1;
            scale
        });
        let rhs_scale = self.has_rhs_scale.then(|| &inputs[scale_index]);
        Ok((lhs, rhs, lhs_scale, rhs_scale))
    }
}

impl Display for ScaledDotOperation {
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
        let (lhs, rhs, lhs_scale, rhs_scale) = self.inputs(input_types)?;
        let rank = lhs.rank();
        if rank != rhs.rank()
            || lhs_scale.is_some_and(|scale| scale.rank() != rank)
            || rhs_scale.is_some_and(|scale| scale.rank() != rank)
        {
            return Err(TypeError::invalid(format!("'{SCALED_DOT_OPERATION_NAME}' inputs must have the same rank")));
        }
        for (descriptor, input_type) in input_types.iter().enumerate() {
            if !input_type.data_type().is_numeric() && input_type.data_type() != DataType::Boolean {
                return Err(TypeError::invalid(format!(
                    "'{SCALED_DOT_OPERATION_NAME}' input {descriptor} must have a numeric or Boolean element type but got {}",
                    input_type.data_type(),
                )));
            }
            if !input_type.unreduced_axes().is_empty() {
                return Err(TypeError::invalid(format!(
                    "'{SCALED_DOT_OPERATION_NAME}' does not support unreduced inputs"
                )));
            }
        }

        let dot_lhs = lhs.clone().with_data_type(DataType::BF16);
        let dot_rhs = rhs.clone().with_data_type(DataType::BF16);
        let output_type = dot_abstract(&dot_lhs, &dot_rhs, &self.dimensions, Some(self.preferred_element_type), None)?;
        for (side, operand, scale, contracting_dimensions) in [
            ("left", lhs, lhs_scale, self.dimensions.lhs_contracting_dimensions()),
            ("right", rhs, rhs_scale, self.dimensions.rhs_contracting_dimensions()),
        ] {
            let Some(scale) = scale else { continue };
            for axis in 0..rank {
                let operand_dimension = operand.dimension(axis);
                let scale_dimension = scale.dimension(axis);
                if contracting_dimensions.contains(&axis) {
                    match (operand_dimension.value(), scale_dimension.value()) {
                        (Some(operand_size), Some(scale_size)) if scale_size == 0 || operand_size % scale_size != 0 => {
                            return Err(TypeError::invalid(format!(
                                "'{SCALED_DOT_OPERATION_NAME}' {side} contracting axis {axis} of size {operand_size} must be divisible by its scale size {scale_size}",
                            )));
                        }
                        (Some(operand_size), Some(scale_size)) if operand_size / scale_size < 2 => {
                            return Err(TypeError::invalid(format!(
                                "'{SCALED_DOT_OPERATION_NAME}' {side} contracting axis {axis} to scale ratio must be at least 2 but got {}",
                                operand_size / scale_size,
                            )));
                        }
                        _ if operand_dimension == scale_dimension => {
                            return Err(TypeError::invalid(format!(
                                "'{SCALED_DOT_OPERATION_NAME}' {side} contracting axis {axis} to scale ratio must be at least 2"
                            )));
                        }
                        _ => {}
                    }
                } else if operand_dimension != scale_dimension {
                    return Err(TypeError::invalid(format!(
                        "'{SCALED_DOT_OPERATION_NAME}' {side} axis {axis} has size {operand_dimension} but its scale has size {scale_dimension}",
                    )));
                }
            }
        }
        Ok(vec![output_type])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, SCALED_DOT_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("dimensions", &self.dimensions)?;
            operation.field("preferred_element_type", &self.preferred_element_type)?;
            operation.field("lhs_scale", &self.has_lhs_scale)?;
            operation.field("rhs_scale", &self.has_rhs_scale)
        })
    }
}

impl<C: Domain<Type = ArrayType, Value: ScaledDot>> InterpretableOperation<C> for ScaledDotOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        let (lhs, rhs, lhs_scale, rhs_scale) = self.inputs(inputs)?;
        Ok(vec![lhs.scaled_dot(
            rhs,
            lhs_scale,
            rhs_scale,
            Some(&self.dimensions),
            Some(self.preferred_element_type),
        )?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for ScaledDotOperation where
    C::Operation: From<ScaledDotOperation>
{
}

/// Scaled dot intentionally has no differentiation rule. Its scales encode quantization metadata rather than
/// differentiable parameters, and silently treating them as constants would define a straight-through estimator that
/// is not part of the operation's semantics. Callers that need a differentiable approximation can spell out the
/// dequantization composition explicitly and choose their own gradient policy.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for ScaledDotOperation
where
    C::Operation: From<ScaledDotOperation>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        _inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!("'{SCALED_DOT_OPERATION_NAME}' does not support differentiation"),
        }
        .into())
    }
}

crate::impl_non_transposable_operation!(ScaledDotOperation);

/// Batching rule for [`ScaledDotOperation`]. Every input is aligned to one leading mapped axis. The rule then shifts
/// every existing dimension number past that new axis and records axis zero as an additional batching dimension on
/// both element operands. Repeating the transform applies the same lift again, so batching has no rank ceiling.
impl<C: Context<Type = ArrayType>, P: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>>
    for ScaledDotOperation
where
    ScaledDotOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<P>>, BatchingError> {
        if inputs.len() != self.input_count() {
            return Err(ProgramError::InvalidInputCount { expected: self.input_count(), actual: inputs.len() }.into());
        }
        if inputs.iter().all(|input| input.batch_axis().is_replicated()) {
            return Ok(self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()])?.into());
        }
        let mapped_dimensions = inputs
            .iter()
            .filter_map(|input| input.batch_axis_position().map(|axis| input.r#type().dimension(axis)))
            .collect::<Vec<_>>();
        if mapped_dimensions.windows(2).any(|pair| pair[0] != pair[1]) {
            return Err(BatchingError::MisalignedBatchAxes {
                message: format!("'{SCALED_DOT_OPERATION_NAME}' inputs map different batch extents"),
            });
        }
        let aligned_inputs = inputs
            .iter()
            .map(|input| P::match_axis(context, input, Axis::from(0)))
            .collect::<Result<Vec<_>, _>>()?;
        let (dimensions, output_axis) = lift_dot_dimensions(&self.dimensions, Some(0), Some(0)).unwrap();
        let lifted =
            ScaledDotOperation::new(dimensions, self.preferred_element_type, self.has_lhs_scale, self.has_rhs_scale);
        Ok(lifted
            .interpret_with_batch_axes(
                context,
                aligned_inputs.as_slice(),
                &[BatchAxis::from_optional_position(output_axis)],
            )?
            .into())
    }
}

/// Value-level generalized block-scaled dot capability.
pub trait ScaledDot: Typed<Type = ArrayType> + Sized {
    /// Computes `(self * lhs_scale) · (rhs * rhs_scale)` using a generalized dot contraction. Each absent scale is
    /// the multiplicative identity. When `dimensions` is absent, the final left axis contracts with the penultimate
    /// right axis and every preceding axis is batched. The preferred element type defaults to `bf16`.
    ///
    /// # Parameters
    ///
    ///   - `rhs`: Right element operand.
    ///   - `lhs_scale`: Optional block-scale tensor for the left operand.
    ///   - `rhs_scale`: Optional block-scale tensor for the right operand.
    ///   - `dimensions`: Optional generalized-dot dimension numbers.
    ///   - `preferred_element_type`: Optional result and accumulation data type.
    fn scaled_dot(
        &self,
        rhs: &Self,
        lhs_scale: Option<&Self>,
        rhs_scale: Option<&Self>,
        dimensions: Option<&DotDimensionNumbers>,
        preferred_element_type: Option<DataType>,
    ) -> Result<Self, ProgramError>;

    /// Computes the rank-3 block-scaled matrix product `[B, M, K] × [B, N, K] -> [B, M, N]`.
    #[inline]
    fn scaled_matmul(
        &self,
        rhs: &Self,
        lhs_scale: &Self,
        rhs_scale: &Self,
        preferred_element_type: Option<DataType>,
    ) -> Result<Self, ProgramError> {
        if [self, rhs, lhs_scale, rhs_scale].iter().any(|value| value.r#type().rank() != 3) {
            return Err(TypeError::invalid("'scaled_matmul' expects rank-3 inputs".to_string()).into());
        }
        self.scaled_dot(
            rhs,
            Some(lhs_scale),
            Some(rhs_scale),
            Some(&DotDimensionNumbers::new(vec![2], vec![2], vec![0], vec![0])),
            Some(preferred_element_type.unwrap_or(DataType::F32)),
        )
    }
}

/// Any context-carrying value computes a block-scaled dot by binding a [`ScaledDotOperation`] through its own
/// context. The `From<ScaledDotOperation>` bound makes this disjoint from the eager reference value types (whose
/// context operation is [`ConstantOperation`](crate::operations::constants::ConstantOperation)), so it covers the
/// transform tracers and backend-owned values without conflicting with concrete implementations.
impl<V: Value<Type = ArrayType>> ScaledDot for V
where
    V::DispatchDomain: Context<Operation: From<ScaledDotOperation>>,
{
    fn scaled_dot(
        &self,
        rhs: &Self,
        lhs_scale: Option<&Self>,
        rhs_scale: Option<&Self>,
        dimensions: Option<&DotDimensionNumbers>,
        preferred_element_type: Option<DataType>,
    ) -> Result<Self, ProgramError> {
        let dimensions = dimensions
            .cloned()
            .map(Ok)
            .unwrap_or_else(|| ScaledDotOperation::default_dimensions(self.r#type().rank()))?;
        let preferred_element_type = preferred_element_type.unwrap_or(DataType::BF16);
        let mut inputs = vec![self.clone(), rhs.clone()];
        inputs.extend(lhs_scale.cloned());
        inputs.extend(rhs_scale.cloned());
        let mut outputs = self.dispatch_domain().bind(
            ScaledDotOperation::new(dimensions, preferred_element_type, lhs_scale.is_some(), rhs_scale.is_some()),
            Vec::new(),
            inputs.as_slice(),
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Evaluates generalized scaled dot as its canonical portable composition.
///
/// Each present scale is expanded independently along every contracting axis, converted to `bf16`, and multiplied
/// into an element operand that is also converted to `bf16`. The resulting tensors are contracted using `dimensions`
/// and accumulated at `preferred_element_type`. An absent scale leaves its operand unchanged except for the `bf16`
/// conversion.
pub fn scaled_dot_composition<V>(
    lhs: &V,
    rhs: &V,
    lhs_scale: Option<&V>,
    rhs_scale: Option<&V>,
    dimensions: &DotDimensionNumbers,
    preferred_element_type: DataType,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + Broadcast + ConvertElementType + DimensionSize<usize> + Dot + Mul + Reshape,
{
    let lhs = ArrayIrValue::from(lhs.clone());
    let rhs = ArrayIrValue::from(rhs.clone());
    let lhs_scale = lhs_scale.cloned().map(ArrayIrValue::from);
    let rhs_scale = rhs_scale.cloned().map(ArrayIrValue::from);
    scaled_dot_ir_composition(&lhs, &rhs, lhs_scale.as_ref(), rhs_scale.as_ref(), dimensions, preferred_element_type)
}

/// Evaluates generalized scaled dot over the mixed array IR.
///
/// This is the authoritative staged composition. It obtains operand and scale extents through [`DimensionSize`],
/// proves each dynamic contracting ratio with [`DimensionRequirement`], and supplies every broadcast and reshape
/// extent as an ordinary first-class dimension operand. Array arithmetic remains delegated to the projected
/// [`ArrayType`] member, so this function introduces neither another value universe nor backend-specific shape logic.
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
        + DimensionSize
        + DynamicBroadcast
        + DynamicReshape
        + ValueProjection<ArrayType>
        + ValueProjection<DimensionType>,
    <V as ValueProjection<ArrayType>>::Projected: Value<Type = ArrayType> + ConvertElementType + Dot + Mul,
    <V as ValueProjection<DimensionType>>::Projected: Value<Type = DimensionType> + DimensionRequirement + Div,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
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
        lhs.dot(&rhs, dimensions)
    } else {
        lhs.dot_with_accumulation_type(&rhs, dimensions, preferred_element_type)
    })
}

/// Dequantizes one mixed-IR block-scaled operand for [`scaled_dot_ir_composition`].
fn dequantize_block_scaled_ir<V>(
    elements: &V,
    scale: Option<&V>,
    contracting_dimensions: &[usize],
) -> Result<<V as ValueProjection<ArrayType>>::Projected, ProgramError>
where
    V: Value<Type = ArrayIrType>
        + DimensionSize
        + DynamicBroadcast
        + DynamicReshape
        + ValueProjection<ArrayType>
        + ValueProjection<DimensionType>,
    <V as ValueProjection<ArrayType>>::Projected: Value<Type = ArrayType> + ConvertElementType + Mul,
    <V as ValueProjection<DimensionType>>::Projected: Value<Type = DimensionType> + DimensionRequirement + Div,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
{
    let context = elements.dispatch_domain();
    let element_type = elements.r#type();
    let element_type = <&ArrayType>::try_from(element_type.as_ref())?;
    let element_dimensions =
        (0..element_type.rank()).map(|axis| elements.dimension_size(axis)).collect::<Result<Vec<_>, _>>()?;
    let elements =
        <V as ValueProjection<ArrayType>>::into_projected(elements.clone())?.convert_element_type(DataType::BF16)?;
    let Some(scale) = scale else { return Ok(elements) };
    let scale_type = scale.r#type();
    let scale_type = <&ArrayType>::try_from(scale_type.as_ref())?;
    let scale_dimensions = (0..scale_type.rank())
        .map(|scale_axis| scale.dimension_size(scale_axis))
        .collect::<Result<Vec<_>, _>>()?;
    let mut expanded_dimensions = Vec::with_capacity(scale_type.rank() + contracting_dimensions.len());
    let mut output_axes = Vec::with_capacity(scale_type.rank());
    for (axis, scale_dimension) in scale_dimensions.iter().enumerate() {
        output_axes.push(expanded_dimensions.len());
        expanded_dimensions.push(scale_dimension.clone());
        if contracting_dimensions.contains(&axis) {
            let element_extent =
                <V as ValueProjection<DimensionType>>::into_projected(element_dimensions[axis].clone())?;
            let scale_extent = <V as ValueProjection<DimensionType>>::into_projected(scale_dimension.clone())?;
            element_extent.require_divisible_by(&scale_extent)?;
            let ratio = element_extent.div(&scale_extent)?;
            let two = <V as ValueProjection<DimensionType>>::into_projected(
                context.lift(DimensionValue::constant(2)?.into())?,
            )?;
            two.require_less_than_or_equal(&ratio)?;
            expanded_dimensions.push(<V as ValueProjection<DimensionType>>::from_projected(ratio));
        }
    }
    let expanded_scale = scale
        .dynamic_broadcast(expanded_dimensions.as_slice(), output_axes.as_slice())?
        .dynamic_reshape(element_dimensions.as_slice())?;
    let expanded_scale =
        <V as ValueProjection<ArrayType>>::into_projected(expanded_scale)?.convert_element_type(DataType::BF16)?;
    elements.mul(&expanded_scale)
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayBatch, ArrayType, DataType, Dimension, Shape};
    use crate::batching::{BatchAxis, BatchableOperation, BatchingContext};
    use crate::contexts::EagerContext;
    use crate::operations::dot::DotDimensionNumbers;
    use crate::programs::{Operation, TypeError};

    use super::*;

    #[test]
    fn test_scaled_dot_jax_contract() {
        // This fixture uses JAX's rank-2 default convention: the left trailing axis contracts with the right leading
        // axis. The two sides infer independent block ratios of two from different scale-axis positions.
        let lhs = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)])),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let rhs = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(3)])),
            (1..=12).map(|value| value as f64).collect(),
        );
        let lhs_scale = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
            vec![1.0, 2.0, 0.5, 1.0],
        );
        let rhs_scale = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        );
        let product = lhs.scaled_dot(&rhs, Some(&lhs_scale), Some(&rhs_scale), None, Some(DataType::F32)).unwrap();
        assert_eq!(
            product.r#type().as_ref(),
            &ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])),
        );
        assert_eq!(product.to_f64s(), vec![253.0, 284.0, 315.0, 272.5, 308.0, 343.5]);

        // Each scale is independently optional. Missing scales are the multiplicative identity, and omitting both
        // therefore reduces to a `bf16`-intermediate generalized dot with an `f32` result.
        assert_eq!(
            lhs.scaled_dot(&rhs, None, None, None, Some(DataType::F32)).unwrap().to_f64s(),
            vec![70.0, 80.0, 90.0, 158.0, 184.0, 210.0],
        );
        assert_eq!(
            lhs.scaled_dot(&rhs, Some(&lhs_scale), None, None, Some(DataType::F32)).unwrap().to_f64s(),
            vec![131.0, 148.0, 165.0, 143.5, 164.0, 184.5],
        );

        let dimensions = ScaledDotOperation::default_dimensions(2).unwrap();
        let operation = ScaledDotOperation::new(dimensions.clone(), DataType::F32, true, true);
        assert_eq!(operation.dimensions(), &dimensions);
        assert_eq!(operation.preferred_element_type(), DataType::F32);
        assert!(operation.has_lhs_scale());
        assert!(operation.has_rhs_scale());
        assert_eq!(
            operation.to_string(),
            indoc! {"
                scaled_dot [
                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    preferred_element_type=f32,
                    lhs_scale=true,
                    rhs_scale=true,
                ]
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_scaled_dot_inference() {
        let dimensions = DotDimensionNumbers::new(vec![2, 3], vec![1, 2], vec![0], vec![0]);
        let operation = ScaledDotOperation::new(dimensions, DataType::BF16, true, true);
        let lhs = ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![2.into(), 3.into(), 4.into(), 6.into()]));
        let rhs = ArrayType::new(DataType::F8E5M2, Shape::new(vec![2.into(), 4.into(), 6.into(), 5.into()]));
        let lhs_scale = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![2.into(), 3.into(), 2.into(), 2.into()]));
        let rhs_scale = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![2.into(), 2.into(), 2.into(), 5.into()]));
        assert_eq!(
            operation.infer_output_types(&[lhs.clone(), rhs.clone(), lhs_scale.clone(), rhs_scale.clone()], &[]),
            Ok(vec![ArrayType::new(DataType::BF16, Shape::new(vec![2.into(), 3.into(), 5.into()]))]),
        );
        assert_eq!(
            ScaledDotOperation::new(operation.dimensions().clone(), DataType::F64, true, true)
                .infer_output_types(&[lhs.clone(), rhs.clone(), lhs_scale.clone(), rhs_scale.clone()], &[]),
            Ok(vec![ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into(), 5.into()]))]),
        );
        assert_eq!(
            operation.infer_output_types(
                &[
                    lhs,
                    rhs,
                    lhs_scale.with_shape(Shape::new(vec![2.into(), 3.into(), 4.into(), 2.into()])),
                    ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![2.into(), 2.into(), 2.into(), 5.into()]),),
                ],
                &[],
            ),
            Err(TypeError::invalid(
                "'scaled_dot' left contracting axis 2 to scale ratio must be at least 2 but got 1".to_string(),
            )),
        );
    }

    #[test]
    fn test_scaled_dot_composition_supports_multiple_contracting_dimensions() {
        // Both contracting axes carry independent block ratios. Expanding all scale axes in one broadcast preserves
        // their original axis positions before the final reshape; every dequantized element is one in this fixture.
        let lhs = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![1.into(), 2.into(), 4.into(), 6.into()])),
            vec![1.0; 48],
        );
        let rhs = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![1.into(), 4.into(), 6.into(), 3.into()])),
            vec![1.0; 72],
        );
        let lhs_scale = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![1.into(), 2.into(), 2.into(), 2.into()])),
            vec![1.0; 8],
        );
        let rhs_scale = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![1.into(), 2.into(), 2.into(), 3.into()])),
            vec![1.0; 12],
        );
        let dimensions = DotDimensionNumbers::new(vec![2, 3], vec![1, 2], vec![0], vec![0]);
        let output = lhs
            .scaled_dot(&rhs, Some(&lhs_scale), Some(&rhs_scale), Some(&dimensions), Some(DataType::F32))
            .unwrap();
        assert_eq!(output.r#type().shape(), &Shape::new(vec![1.into(), 2.into(), 3.into()]));
        assert_eq!(output.to_f64s(), vec![24.0; 6]);

        // The ergonomic wrapper follows JAX's `[B, M, K] x [B, N, K]` convention and defaults to an `f32` result.
        let lhs = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![1.into(), 1.into(), 4.into()])),
            vec![1.0; 4],
        );
        let rhs = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![1.into(), 2.into(), 4.into()])),
            vec![1.0; 8],
        );
        let lhs_scale = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![1.into(), 1.into(), 2.into()])),
            vec![1.0; 2],
        );
        let rhs_scale = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![1.into(), 2.into(), 2.into()])),
            vec![1.0; 4],
        );
        let output = lhs.scaled_matmul(&rhs, &lhs_scale, &rhs_scale, None).unwrap();
        assert_eq!(output.r#type().data_type(), DataType::F32);
        assert_eq!(output.r#type().shape(), &Shape::new(vec![1.into(), 1.into(), 2.into()]));
        assert_eq!(output.to_f64s(), vec![4.0; 2]);

        // Ryft honors the wrapper's documented independent block-ratio semantics even though pinned JAX's wrapper
        // currently rejects unequal scale contracting dimensions before reaching `lax.scaled_dot`.
        let independently_scaled_rhs = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![1.into(), 2.into(), 1.into()])),
            vec![1.0; 2],
        );
        let output = lhs.scaled_matmul(&rhs, &lhs_scale, &independently_scaled_rhs, None).unwrap();
        assert_eq!(output.to_f64s(), vec![4.0; 2]);
    }

    #[test]
    fn test_scaled_dot_batching() {
        // Batching moves each mapped axis to the front and lifts it into the generalized-dot batch dimensions. Scale
        // operands follow the same rule, so each example retains its own block scales.
        let elements = ArrayType::new(DataType::F32, Shape::new(vec![2.into(), 4.into()]));
        let scales = ArrayType::new(DataType::F32, Shape::new(vec![2.into(), 2.into()]));
        let lhs = ArrayBatch::new(Array::from_f64s(elements.clone(), vec![1.0; 8]), BatchAxis::new(0)).unwrap();
        let rhs = ArrayBatch::new(Array::from_f64s(elements, vec![1.0; 8]), BatchAxis::new(0)).unwrap();
        let lhs_scale = ArrayBatch::new(Array::from_f64s(scales.clone(), vec![1.0; 4]), BatchAxis::new(0)).unwrap();
        let rhs_scale = ArrayBatch::new(Array::from_f64s(scales, vec![1.0; 4]), BatchAxis::new(0)).unwrap();
        let operation = ScaledDotOperation::new(DotDimensionNumbers::inner_product(), DataType::F32, true, true);

        let outputs = operation
            .batch(
                &BatchingContext::new(EagerContext::<Array>::new(), 2),
                &crate::EmptyRegionDriver,
                &[lhs, rhs, lhs_scale, rhs_scale],
            )
            .unwrap()
            .into_parts()
            .0;

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().to_f64s(), vec![4.0, 4.0]);
    }
}
