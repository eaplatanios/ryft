use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayElement, ArrayIrType, ArrayType, Dimension, DimensionValue,
};
use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
use crate::contexts::{Context, EagerContext, ProjectedContext, StagingContext};
use crate::differentiation::{
    DifferentiableType, DifferentiationContext, DifferentiationDual, DifferentiationPolicy, DifferentiationTracer,
};
use crate::macros::check_count;
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::constants::validate_dynamic_constant_dimensions;
use crate::operations::manipulation::broadcasting::{
    BROADCAST_OPERATION_NAME, Broadcast, BroadcastOperation, DynamicBroadcastOperation,
};
use crate::operations::manipulation::conversions::ConvertElementType;
use crate::operations::manipulation::memory::TransferToMemory;
use crate::partial::{PartialEvaluationContext, PartialTracer};
use crate::programs::{Operation, OperationProjection, ProgramError, Type, TypeError, Typed, Value, ValueProjection};
use crate::tracing::Tracer;

/// Represents the ability to synthesize a value filled with one typed host literal. [`ArrayType`] implementations
/// encode the literal as a rank-zero array, convert it to the requested element [`DataType`](crate::DataType) and
/// [`Memory`](crate::Memory), and use ordinary broadcasting for every rank-positive result. This keeps the fill value
/// explicit in Static Single Assignment (SSA) dataflow and avoids the need for a separate array fill operation type.
pub trait Fill<L, V: Typed> {
    /// Returns a value of [`Type`] `type` with every element it holds set to `value`.
    fn fill(&self, r#type: &V::Type, value: L) -> Result<V, ProgramError>;
}

impl<L: ArrayElement, O: Operation<Type = ArrayType>> Fill<L, Array> for EagerContext<Array, O> {
    fn fill(&self, r#type: &ArrayType, value: L) -> Result<Array, ProgramError> {
        if r#type.static_shape().is_none() {
            return Err(TypeError::invalid(format!(
                "cannot materialize a value of dynamically sized type {}; stage a rank-zero fill \
                 and expand it with a dynamic `{BROADCAST_OPERATION_NAME}` operation instead",
                r#type,
            ))
            .into());
        }
        Array::scalar(value)?
            .convert_element_type(r#type.data_type())?
            .transfer_to_memory(r#type.memory())
            .broadcast(r#type.clone(), &[])
    }
}

impl<L, C: Context, T: Type> Fill<L, <C::Value as ValueProjection<T>>::Projected> for ProjectedContext<C, T>
where
    C::Value: ValueProjection<T, Projected: Value<Type = T>>,
    ProjectedContext<C, T>: Context<Type = T, Value = <C::Value as ValueProjection<T>>::Projected> + FillLiteral<L, T>,
{
    #[inline]
    fn fill(&self, r#type: &T, value: L) -> Result<<C::Value as ValueProjection<T>>::Projected, ProgramError> {
        <ProjectedContext<C, T> as FillLiteral<L, T>>::fill_literal(self, r#type, value)
    }
}

impl<L, T: Type, C: StagingContext<Type = T> + FillLiteral<L, T>> Fill<L, Tracer<C>> for C {
    #[inline]
    fn fill(&self, r#type: &C::Type, value: L) -> Result<Tracer<C>, ProgramError> {
        self.fill_literal(r#type, value)
    }
}

impl<L, T: Type, C: Context<Type = T>> Fill<L, PartialTracer<C>> for PartialEvaluationContext<C>
where
    PartialEvaluationContext<C>: Context<Type = T, Value = PartialTracer<C>> + FillLiteral<L, T>,
{
    #[inline]
    fn fill(&self, r#type: &C::Type, value: L) -> Result<PartialTracer<C>, ProgramError> {
        <PartialEvaluationContext<C> as FillLiteral<L, T>>::fill_literal(self, r#type, value)
    }
}

impl<L, C: Context<Type = ArrayType> + Fill<L, C::Value>> Fill<L, BatchingTracer<C, ArrayBatchingPolicy>>
    for BatchingContext<C, ArrayBatchingPolicy>
{
    #[inline]
    fn fill(&self, r#type: &ArrayType, value: L) -> Result<BatchingTracer<C, ArrayBatchingPolicy>, ProgramError> {
        let batch = ArrayBatch::new(self.parent().fill(r#type, value)?, BatchAxis::replicated())?;
        Ok(BatchingTracer::new(self.clone(), batch))
    }
}

impl<L, C: Context<Type: DifferentiableType> + Fill<L, C::Value>, P: DifferentiationPolicy<C>>
    Fill<L, DifferentiationTracer<C, P>> for DifferentiationContext<C, P>
{
    #[inline]
    fn fill(&self, r#type: &C::Type, value: L) -> Result<DifferentiationTracer<C, P>, ProgramError> {
        let dual = DifferentiationDual::new_with_zero_tangent(self.primal().fill(r#type, value)?)?;
        Ok(DifferentiationTracer::new(dual, self.clone()))
    }
}

/// Internal trait used for implementing [`Fill`] by embedding a typed host literal in the active [`Context`]. For
/// arrays, the implementation converts `value` into a rank-zero [`Array`] and binds it as a [`ConstantOperation`],
/// then broadcasts that scalar to the requested type. The [`Array`] is only a portable literal payload that a backend
/// context interprets or lowers the constant into its own runtime value. It is distinct from the context's
/// [`Domain::Constant`](crate::Domain::Constant) representation, which stores lifted constants and program captures.
trait FillLiteral<L, T: Type>: Context<Type = T> {
    /// Embeds `value` as a host literal and expands it to a value of `type` in this [`Context`].
    fn fill_literal(&self, r#type: &T, value: L) -> Result<Self::Value, ProgramError>;
}

impl<L: ArrayElement, C: Context<Type = ArrayType>> FillLiteral<L, ArrayType> for C
where
    C::Value: Broadcast,
    C::Operation: From<ConstantOperation<Array>>,
{
    fn fill_literal(&self, r#type: &ArrayType, value: L) -> Result<Self::Value, ProgramError> {
        let value = Array::scalar(value)?.convert_element_type(r#type.data_type())?.transfer_to_memory(r#type.memory());
        self.bind(ConstantOperation::new(value), Vec::new(), &[])?.remove(0).broadcast(r#type.clone(), &[])
    }
}

/// Represents the ability to construct an [`Array`] filled with a supplied value whose shape includes _dynamic_ (i.e.,
/// runtime) dimensions. Unlike [`Fill`], this capability supplies each dynamic axis with an explicit dimension value.
/// Static axes retain their declared sizes. Dynamic axes take their sizes from the inputs in axis order. Repeated
/// dimension identities require a corresponding input for every occurrence. Each input must have the exact dimension
/// identity declared by its axis. The caller must thus provide the same dimension value for repeated occurrences.
///
/// Note that a fully static output type is also accepted with no dimension inputs. The same capability works with eager
/// mixed-IR values and with tracer values, where construction records the dimension inputs in the staged program.
///
/// The output type declares the element type, rank, static sizes, and identities and bounds of dynamic axes without
/// needing their runtime extents. For example, shape `[N, 2, M]` takes dimension values for `[N, M]`; extents `3` and
/// `4` produce shape `[3, 2, 4]`. The converted literal is embedded once and broadcast using the existing constant and
/// broadcast operations.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{
/// #     Array, ArrayIrOperation, ArrayIrValue, ArrayType, DataType, Dimension, DimensionBounds, DimensionType,
/// #     DimensionValue, DimensionVariable, DynamicFill, EagerContext, Shape,
/// # };
/// let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
/// let size = DimensionVariable::new("size", DimensionBounds::unbounded());
/// let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(size.clone())]));
/// let dimension = ArrayIrValue::Dimension(DimensionValue::new(DimensionType::new(size), 3).unwrap());
/// assert_eq!(
///     context.dynamic_fill(&output_type, 2.5f32, &[dimension]),
///     Ok(ArrayIrValue::Array(Array::vector(vec![2.5f32; 3]).unwrap())),
/// );
/// ```
pub trait DynamicFill<L, V: Typed> {
    /// Constructs an array filled with `value`, using explicit values for its dynamic dimensions. The literal is
    /// converted to the output element type before broadcasting. The result preserves the requested memory and
    /// sharding. Dynamic shapes use the broadcast operation's default layout; an explicit layout with a dynamic shape
    /// is rejected rather than silently discarded. Static shapes support the same output layouts as [`Fill`]. Invalid
    /// geometry is rejected before staging the literal.
    ///
    /// # Parameters
    ///
    ///   - `type`: Output [`ArrayType`] that may contain dynamic dimensions. Each dynamic axis names the dimension
    ///     identity that its corresponding input must carry.
    ///   - `value`: Host literal copied into every output element after element-type conversion.
    ///   - `dimensions`: Contains one dimension value per dynamic axis, in axis order. Static axes do not consume input
    ///     dimensions provided this way. Repeated identities still consume one input for each axis that uses them.
    fn dynamic_fill(&self, r#type: &ArrayType, value: L, dimensions: &[V]) -> Result<V, ProgramError>;
}

impl<L: ArrayElement, C: Context<Type = ArrayIrType>> DynamicFill<L, C::Value> for C
where
    C::Operation: OperationProjection<ArrayType, Projected: From<ConstantOperation<Array>> + From<BroadcastOperation>>
        + From<ConstantOperation<DimensionValue>>
        + From<DynamicBroadcastOperation>,
{
    fn dynamic_fill(&self, r#type: &ArrayType, value: L, dimensions: &[C::Value]) -> Result<C::Value, ProgramError> {
        validate_dynamic_constant_dimensions("fill", r#type, dimensions)?;
        if !dimensions.is_empty() && r#type.layout().is_some() {
            return Err(TypeError::invalid("dynamic fill does not support an explicit output layout").into());
        }
        let literal =
            Array::scalar(value)?.convert_element_type(r#type.data_type())?.transfer_to_memory(r#type.memory());
        let scalar_operation =
            <C::Operation as OperationProjection<ArrayType>>::Projected::from(ConstantOperation::new(literal));
        let mut outputs = self.bind(scalar_operation, Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        let scalar = outputs.remove(0);
        if dimensions.is_empty() {
            let mut outputs = self.bind(
                <C::Operation as OperationProjection<ArrayType>>::Projected::from(BroadcastOperation::new(
                    r#type.clone(),
                    Vec::new(),
                )),
                Vec::new(),
                &[scalar],
            )?;
            check_count!("output", outputs, 1, ProgramError);
            return Ok(outputs.remove(0));
        }

        // Dynamic broadcast consumes every output axis. Static axes become dimension constants, while the
        // caller's dynamic axes remain explicit operands and retain their declared identity.
        let mut dimensions = dimensions.iter();
        let mut inputs = Vec::with_capacity(r#type.rank() + 1);
        inputs.push(scalar);
        for dimension in r#type.shape().dimensions() {
            inputs.push(match dimension {
                Dimension::Static(extent) => {
                    let mut outputs =
                        self.bind(ConstantOperation::new(DimensionValue::constant(*extent)?), Vec::new(), &[])?;
                    check_count!("output", outputs, 1, ProgramError);
                    outputs.remove(0)
                }
                Dimension::Dynamic(_) => dimensions.next().unwrap().clone(),
            });
        }
        let operation = DynamicBroadcastOperation::new(Vec::new()).with_output_sharding(r#type.sharding().cloned());
        let mut outputs = self.bind(operation, Vec::new(), &inputs)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayBatchingPolicy, ArrayIrOperation, ArrayIrValue, ArrayOperation, ArrayType, DataType, Dimension,
        DimensionBounds, DimensionType, DimensionValue, DimensionVariable, Layout, LogicalMesh, Memory, MeshAxis,
        MeshAxisType, Shape, Sharding, StridedLayout, f6e2m3fn, u4,
    };
    use crate::operations::manipulation::broadcasting::DynamicBroadcastOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{AtomId, MaybeZero, ProgramBuilder};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_fill_interpretation() {
        let context = EagerContext::<Array>::new();

        // Rank-zero and rank-positive fills both apply the requested element conversion exactly once.
        assert_eq!(context.fill(&ArrayType::scalar(DataType::U32), 2.5f64), Ok(Array::scalar(2u32).unwrap()));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
            .with_memory(Memory::Host { pinned: true });
        assert_eq!(
            context.fill(&output_type, 2.5f64).unwrap(),
            Array::from_elements(output_type, &[2.5f32; 6]).unwrap(),
        );
        assert_eq!(
            context.fill(&ArrayType::new_static(DataType::F32, [2]), ComplexNumber::new(1.0f32, 2.0)),
            Ok(Array::vector(vec![1.0f32; 2]).unwrap()),
        );
        assert_eq!(
            context.fill(&ArrayType::new_static(DataType::Boolean, [2]), ComplexNumber::new(0.0f32, 2.0)),
            Array::from_elements(ArrayType::new_static(DataType::Boolean, [2]), &[true; 2]).map_err(Into::into),
        );

        // Fill uses the same checked element codecs for low-precision and sub-byte values as direct array creation.
        let low_precision = f6e2m3fn::from_bits(0x08).unwrap();
        assert_eq!(
            context.fill(&ArrayType::new_static(DataType::F6E2M3FN, [2]), low_precision),
            Array::from_elements(ArrayType::new_static(DataType::F6E2M3FN, [2]), &[low_precision; 2])
                .map_err(Into::into),
        );
        assert_eq!(
            context.fill(&ArrayType::new_static(DataType::U4, [2]), 2.5f64).unwrap().elements::<u4>(),
            Ok(vec![u4::new(2).unwrap(); 2]),
        );

        // Eager arrays have no runtime extent operands from which to materialize a dynamically shaped result.
        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("size", DimensionBounds::unbounded()))]),
        );
        assert!(matches!(
            context.fill(&dynamic_type, 1.0f32),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot materialize a value of dynamically sized type f32[size]; stage a rank-zero \
                               fill and expand it with a dynamic `broadcast` operation instead",
        ));
    }

    #[test]
    fn test_fill_partial_evaluation() {
        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let output_type = ArrayType::new_static(DataType::F32, [2]);
        let output = context.fill(&output_type, 3.5f32).unwrap();
        let expected = Array::from_elements(output_type, &[3.5f32; 2]).unwrap();
        assert_eq!(output.value().unwrap().as_known(), Some(&expected));
    }

    #[test]
    fn test_fill_batching() {
        let context =
            BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 4);
        let output_type = ArrayType::new_static(DataType::F32, [2]);
        let output = context.fill(&output_type, 4.5f32).unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(output.batch().value(), &Array::from_elements(output_type, &[4.5f32; 2]).unwrap(),);
    }

    #[test]
    fn test_fill_differentiation() {
        let context = DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new());
        let output_type = ArrayType::new_static(DataType::F32, [2]);
        let output = context.fill(&output_type, 5.5f32).unwrap();
        assert_eq!(output.primal(), &Array::from_elements(output_type.clone(), &[5.5f32; 2]).unwrap());
        assert!(matches!(output.tangent(), MaybeZero::Zero(r#type) if r#type == &output_type));
    }

    #[test]
    fn test_array_ir_dynamic_literal_fill_jvp_materializes_shaped_zero() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let extent = builder.add_input(extent_type.clone().into());
        let scalar = builder
            .add_instruction(
                ArrayOperation::from(ConstantOperation::new(Array::scalar(2.5_f64).unwrap())),
                Vec::new(),
                vec![],
                None,
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![scalar, extent], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let jvp = program.jvp().unwrap();
        let extent = ArrayIrValue::Dimension(DimensionValue::new(extent_type, 3).unwrap());
        assert_eq!(
            jvp.interpret(vec![extent]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![2.5_f64, 2.5, 2.5]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![0.0_f64, 0.0, 0.0]).unwrap()),
            ]),
        );
        let dynamic_zero = jvp
            .instructions()
            .iter()
            .find(|instruction| matches!(instruction.operation(), ArrayIrOperation::Zero(_)))
            .unwrap();
        assert_eq!(dynamic_zero.inputs(), &[AtomId::new(0)]);
    }

    #[test]
    fn test_projected_context_fill() {
        let parent = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let context = ProjectedContext::<_, ArrayType>::new(parent.clone());
        let output_type = ArrayType::new_static(DataType::F32, [2]);
        let output = context.fill(&output_type, 1.5f32).unwrap();
        assert_eq!(output.r#type().as_ref(), &output_type);
        let program = parent
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.into_value().atom_id().unwrap()],
                Vec::new(),
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f32[] = constant [value=1.5]
                    %1:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_staging_context_fill() {
        // A rank-zero fill is represented by its literal constant alone.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output = context.fill(&ArrayType::scalar(DataType::U32), 2.5f64).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![output.atom_id().unwrap()], Vec::new(), vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:u32[] = constant [value=2]
                in (%0)
            "}
            .trim_end(),
        );

        // A rank-positive fill broadcasts that literal, retaining the requested memory placement in both operations.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_type = ArrayType::new_static(DataType::F32, [2, 3]).with_memory(Memory::Host { pinned: false });
        let output = context.fill(&output_type, 2.5f64).unwrap();
        assert_eq!(output.r#type().as_ref(), &output_type);
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![output.atom_id().unwrap()], Vec::new(), vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f32[]@Host[Unpinned] = constant [value=2.5]
                    %1:f32[2, 3]@Host[Unpinned] = broadcast \
                        [output_type=f32[2, 3]@Host[Unpinned], output_axes=[]] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dynamic_fill() {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(5)).unwrap());
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 2);
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![rows.clone().into(), 2.into()]))
            .with_memory(Memory::Host { pinned: true })
            .with_sharding(sharding.clone())
            .unwrap();
        let extent = ArrayIrValue::Dimension(DimensionValue::new(DimensionType::new(rows), 3).unwrap());
        let result = context.dynamic_fill(&output_type, 2.5_f64, &[extent]).unwrap();
        let expected_type = ArrayType::new_static(DataType::F32, [3, 2])
            .with_memory(Memory::Host { pinned: true })
            .with_sharding(sharding.clone())
            .unwrap();
        assert_eq!(result, ArrayIrValue::Array(Array::from_elements(expected_type, &[2.5_f32; 6]).unwrap()));
        assert_eq!(
            context.dynamic_fill(&ArrayType::scalar(DataType::Boolean), true, &[]),
            Ok(ArrayIrValue::Array(Array::scalar(true).unwrap()))
        );
    }

    #[test]
    fn test_dynamic_fill_invalid_dimensions() {
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(5)).unwrap());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![rows.clone().into()]));
        assert!(matches!(context.dynamic_fill(&output_type, 1.0_f32, &[]),
            Err(ProgramError::Type(TypeError::Invalid { message, .. }))
                if message == "`fill` expects one dimension operand per dynamic output dimension (1) but got 0 operands"));
        assert!(context.builder().borrow().instructions().is_empty());
        let extent = context.input(DimensionType::new(rows).into());
        let output_type = output_type.with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        assert!(matches!(context.dynamic_fill(&output_type, 1.0_f32, &[extent]),
            Err(ProgramError::Type(TypeError::Invalid { message, .. }))
                if message == "dynamic fill does not support an explicit output layout"));
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_dynamic_fill_staging() {
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(5)).unwrap());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![rows.clone().into(), 2.into()]));
        let dimension_type = DimensionType::new(rows);
        let extent = context.input(dimension_type.clone().into());
        let result = context.dynamic_fill(&output_type, 3.0_f32, &[extent]).unwrap();
        assert_eq!(result.r#type().as_ref(), &ArrayIrType::Array(output_type));
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![result.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert!(matches!(program.instructions().last().unwrap().operation(), ArrayIrOperation::Broadcast(_)));
        let input = ArrayIrValue::Dimension(DimensionValue::new(dimension_type, 3).unwrap());
        assert_eq!(
            program.interpret(vec![input.clone()]),
            Ok(vec![ArrayIrValue::Array(Array::matrix(3, 2, vec![3.0_f32; 6]).unwrap())])
        );
        assert_eq!(
            program.jvp().unwrap().interpret(vec![input]),
            Ok(vec![
                ArrayIrValue::Array(Array::matrix(3, 2, vec![3.0_f32; 6]).unwrap()),
                ArrayIrValue::Array(Array::matrix(3, 2, vec![0.0_f32; 6]).unwrap())
            ])
        );
    }
}
