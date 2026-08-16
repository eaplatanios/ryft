use std::fmt::Display;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatching, ArrayElement, ArrayIrBatching, ArrayIrType, ArrayType,
    dispatch_on_array_element_type,
};
use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
use crate::contexts::{Context, Domain, EagerContext, ProjectedContext, StagingContext};
use crate::differentiation::{DifferentiationContext, DifferentiationDual, DifferentiationTracer};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{
    check_count, impl_non_differentiable_operation, impl_nullary_batchable_operation,
    impl_nullary_transposable_operation,
};
use crate::operations::constants::check_constructor_type_has_no_identity_references;
use crate::partial::{PartialEvaluationContext, PartialTracer, PartiallyEvaluatableOperation};
use crate::programs::{
    Operation, OperationFormatter, OperationProjection, ProgramError, RegionInterface, Type, TypeError,
    TypeIdentityRenaming, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`IotaOperation`].
pub const IOTA_OPERATION_NAME: &str = "iota";

/// [`Operation`] that has no inputs and that produces a single output of the [`Type`] it holds (i.e., its `r#type`
/// field) whose elements increase from `0` along a dimension chosen by [`dimension`](Self::dimension). Along that
/// dimension, the element at index `k` is `k`, and the value is constant along every other dimension. It is the
/// index-generating counterpart of constructing a scalar literal and broadcasting it. Rather than filling every
/// element with one scalar value, it synthesizes the per-position index through the [`Iota`] trait when interpreted.
/// It mirrors StableHLO's [`iota`](https://openxla.org/stablehlo/spec#iota).
#[derive(Copy, Clone, Debug)]
pub struct IotaOperation<T: Type> {
    /// [`Type`] of the value produced when this operation is interpreted.
    r#type: T,

    /// Dimension of `type` along which the produced values increase from `0`.
    dimension: usize,
}

impl<T: Type> IotaOperation<T> {
    /// Returns the type of the value produced by this [`IotaOperation`].
    #[inline]
    pub fn r#type(&self) -> &T {
        &self.r#type
    }

    /// Returns the dimension along which the produced values increase from `0`.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

impl IotaOperation<ArrayType> {
    /// Creates an [`IotaOperation`] after validating its element type and varying dimension.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if `r#type` does not have a numeric element type or if `dimension` is outside its rank.
    #[inline]
    pub fn new(r#type: ArrayType, dimension: usize) -> Result<Self, TypeError> {
        if !r#type.data_type().is_numeric() {
            return Err(TypeError::invalid(format!(
                "`{}` requires a numeric element type but has {}",
                IOTA_OPERATION_NAME,
                r#type.data_type(),
            )));
        }
        if dimension >= r#type.rank() {
            return Err(TypeError::invalid(format!(
                "`{}` dimension {} is out of bounds for rank {}",
                IOTA_OPERATION_NAME,
                dimension,
                r#type.rank(),
            )));
        }
        Ok(Self { r#type, dimension })
    }
}

impl Display for IotaOperation<ArrayType> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for IotaOperation<ArrayType> {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        IOTA_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 0, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        check_constructor_type_has_no_identity_references(IOTA_OPERATION_NAME, &self.r#type)?;
        Ok(vec![self.r#type.clone()])
    }

    #[inline]
    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayType as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        Self::new(self.r#type.rename_identities(renaming)?, self.dimension)
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, IOTA_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("type", &self.r#type)?;
            operation.field("dimension", &self.dimension)
        })
    }
}

impl<C: Domain<Type = ArrayType> + Iota<C::Value>> InterpretableOperation<C> for IotaOperation<ArrayType> {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 0, ProgramError);
        Ok(vec![context.iota(&self.r#type, self.dimension)?])
    }
}

impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for IotaOperation<ArrayType> where
    C::Operation: From<IotaOperation<ArrayType>>
{
}

impl_non_differentiable_operation!(IotaOperation<ArrayType>);
impl_nullary_transposable_operation!(IotaOperation<ArrayType>);
impl_nullary_batchable_operation!(@replicated IotaOperation<ArrayType>);
impl_nullary_batchable_operation!(@member<ArrayIrType, ArrayIrBatching> IotaOperation<ArrayType>);

impl_member_operation_for_array_ir_constant_operation!(IotaOperation<ArrayType>);
impl_member_interpretable_operation_for_array_ir_constant_operation!(
    IotaOperation<ArrayType>,
    Iota,
    |context, output_type, operation| context.iota(&output_type, operation.dimension()),
);

/// Represents the ability to synthesize a value for a given [`Type`] whose elements increase from `0` along a chosen
/// dimension in an interpretation context. [`Iota`] is the [`Type`]-driven capability needed by [`IotaOperation`] for
/// its [`InterpretableOperation`] implementation, sitting alongside [`Zero`](crate::Zero), [`One`](super::One), and
/// [`Fill`](super::Fill) in the same type-driven family.
pub trait Iota<V: Typed> {
    /// Returns a value of `type` whose elements increase from `0` along `dimension` and are constant along every other
    /// dimension.
    ///
    /// # Parameters
    ///
    ///   - `r#type`: Type of the value to produce.
    ///   - `dimension`: Dimension of `type` along which the produced values increase from `0`.
    fn iota(&self, r#type: &V::Type, dimension: usize) -> Result<V, ProgramError>;
}

impl<O: Operation<Type = ArrayType>> Iota<Array> for EagerContext<Array, O> {
    fn iota(&self, r#type: &ArrayType, dimension: usize) -> Result<Array, ProgramError> {
        if !r#type.data_type().is_numeric() {
            return Err(TypeError::invalid(format!(
                "`{}` requires a numeric element type but has {}",
                IOTA_OPERATION_NAME,
                r#type.data_type(),
            ))
            .into());
        }
        let sizes = r#type
            .shape()
            .dimensions()
            .iter()
            .map(|dimension| {
                dimension.value().ok_or_else(|| {
                    TypeError::invalid(format!(
                        "cannot materialize an iota of dynamically sized type {type}; stage it in an array program \
                         over `ArrayIrOperation`, whose `DynamicIota` constructor consumes one dimension operand per \
                         dynamic axis",
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if dimension >= sizes.len() {
            return Err(TypeError::invalid(format!(
                "iota dimension {dimension} is out of bounds for array type {type}",
            ))
            .into());
        }
        // In row-major order, the index along `dimension` at flat position `flat` is `(flat / stride) % size`, where
        // `stride` is the product of the sizes of the dimensions after `dimension`.
        let size = sizes[dimension];
        let stride: usize = sizes[dimension + 1..].iter().product();
        let data_type = r#type.data_type();
        dispatch_on_array_element_type!(data_type, |Element| {
            Array::from_fn_elements(r#type.clone(), |flat| Element::from_unsigned(((flat / stride) % size) as u64))
        })
    }
}

impl<C: Context> Iota<<C::Value as ValueProjection<ArrayType>>::Projected> for ProjectedContext<C, ArrayType>
where
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation: OperationProjection<ArrayType, Projected: From<IotaOperation<ArrayType>>>,
{
    #[inline]
    fn iota(
        &self,
        r#type: &ArrayType,
        dimension: usize,
    ) -> Result<<C::Value as ValueProjection<ArrayType>>::Projected, ProgramError> {
        Ok(self.bind(IotaOperation::new(r#type.clone(), dimension)?, Vec::new(), &[])?.remove(0))
    }
}

impl<C: StagingContext<Type = ArrayType, Operation: From<IotaOperation<ArrayType>>>> Iota<Tracer<C>> for C {
    #[inline]
    fn iota(&self, r#type: &ArrayType, dimension: usize) -> Result<Tracer<C>, ProgramError> {
        let mut outputs = self.stage_nullary_operation(IotaOperation::new(r#type.clone(), dimension)?)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Context<Type = ArrayType>> Iota<PartialTracer<C>> for PartialEvaluationContext<C>
where
    C::Operation: PartiallyEvaluatableOperation<C>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + From<IotaOperation<ArrayType>>,
{
    #[inline]
    fn iota(&self, r#type: &ArrayType, dimension: usize) -> Result<PartialTracer<C>, ProgramError> {
        let mut outputs = self.bind(IotaOperation::new(r#type.clone(), dimension)?, Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Context<Type = ArrayType> + Iota<C::Value>> Iota<BatchingTracer<C, ArrayBatching>>
    for BatchingContext<C, ArrayBatching>
{
    #[inline]
    fn iota(&self, r#type: &ArrayType, dimension: usize) -> Result<BatchingTracer<C, ArrayBatching>, ProgramError> {
        let batch = ArrayBatch::new(self.parent().iota(r#type, dimension)?, BatchAxis::replicated())?;
        Ok(BatchingTracer::new(self.clone(), batch))
    }
}

impl<C: Context<Type = ArrayType> + Iota<C::Value>> Iota<DifferentiationTracer<C>> for DifferentiationContext<C> {
    #[inline]
    fn iota(&self, r#type: &ArrayType, dimension: usize) -> Result<DifferentiationTracer<C>, ProgramError> {
        let dual = DifferentiationDual::new_with_zero_tangent(self.parent().iota(r#type, dimension)?);
        Ok(DifferentiationTracer::new(dual, self.clone()))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayBatching, ArrayIrOperation, ArrayIrValue, ArrayOperation, ArrayType, DataType, Dimension,
        DimensionBounds, DimensionVariable, Shape, u4,
    };
    use crate::batching::{BatchAxis, BatchingContext};
    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{EmptyRegionDriver, MaybeZero, Operation, ProgramBuilder};

    use super::*;

    #[test]
    fn test_iota() {
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));

        // Operation construction validates the varying dimension and element type.
        assert_eq!(
            IotaOperation::new(r#type.clone(), 2).unwrap_err(),
            TypeError::invalid("`iota` dimension 2 is out of bounds for rank 2"),
        );
        assert_eq!(
            IotaOperation::new(ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(2)])), 0,)
                .unwrap_err(),
            TypeError::invalid("`iota` requires a numeric element type but has bool"),
        );
        let complex_type = ArrayType::new(DataType::C64, Shape::new(vec![Dimension::Static(2)]));
        assert_eq!(
            IotaOperation::new(complex_type.clone(), 0).unwrap().infer_output_types(&[], &[]),
            Ok(vec![complex_type]),
        );
        let variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(variable.clone())]));
        assert_eq!(
            IotaOperation::new(dynamic_type, 0).unwrap().infer_output_types(&[], &[]),
            Err(TypeError::invalid(format!(
                "`iota` cannot construct type f64[extent] without operands because it references identity {variable}",
            ))),
        );

        // Verify the operation's stored type and axis, identity, and rendering.
        let operation = IotaOperation::new(r#type.clone(), 1).unwrap();
        assert_eq!(operation.name(), IOTA_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "iota [type=f64[2, 3], dimension=1]");
        assert_eq!(operation.r#type(), &r#type);
        assert_eq!(operation.dimension(), 1);
        assert_eq!(operation.infer_output_types(&[], &[]), Ok(vec![r#type.clone()]));

        // Eager interpretation along axis one varies between columns and repeats across rows.
        let context = EagerContext::<Array, IotaOperation<ArrayType>>::new();
        let expected = Array::from_f64s(r#type.clone(), vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
        assert_eq!(
            InterpretableOperation::<EagerContext<Array, IotaOperation<ArrayType>>>::interpret(
                &operation,
                &context,
                &EmptyRegionDriver,
                &[],
            ),
            Ok(vec![expected.clone()]),
        );

        // Verify the operation's textual form when it appears in a program.
        let mut builder = ProgramBuilder::<Array, IotaOperation<ArrayType>>::new();
        let output = builder.add_instruction(operation, Vec::new(), vec![]).unwrap()[0];
        let program = builder.build::<(), Array>(vec![output], (), Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f64[2, 3] = iota [type=f64[2, 3], dimension=1]
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_eager_context_iota() {
        let context = EagerContext::<Array>::new();

        // Each selected axis varies independently and repeats along every other axis.
        let output_type = ArrayType::new_static(DataType::I32, [2, 3]);
        assert_eq!(
            context.iota(&output_type, 0),
            Array::from_elements(output_type.clone(), &[0i32, 0, 0, 1, 1, 1]).map_err(Into::into),
        );
        assert_eq!(
            context.iota(&output_type, 1),
            Array::from_elements(output_type, &[0i32, 1, 2, 0, 1, 2]).map_err(Into::into),
        );

        // Iota uses the checked element codecs for sub-byte values.
        assert_eq!(
            context.iota(&ArrayType::new_static(DataType::U4, [2, 3]), 1).unwrap().elements::<u4>(),
            Ok(vec![
                u4::new(0).unwrap(),
                u4::new(1).unwrap(),
                u4::new(2).unwrap(),
                u4::new(0).unwrap(),
                u4::new(1).unwrap(),
                u4::new(2).unwrap(),
            ]),
        );

        // Non-numeric elements, out-of-range axes, and dynamic eager shapes are rejected.
        assert_eq!(
            context.iota(&ArrayType::new_static(DataType::Boolean, [2]), 0),
            Err(ProgramError::Type(TypeError::invalid("`iota` requires a numeric element type but has bool"))),
        );
        assert_eq!(
            context.iota(&ArrayType::new_static(DataType::F32, [2]), 1),
            Err(ProgramError::Type(TypeError::invalid("iota dimension 1 is out of bounds for array type f32[2]"))),
        );
        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("size", DimensionBounds::unbounded()))]),
        );
        assert_eq!(
            context.iota(&dynamic_type, 0),
            Err(ProgramError::Type(TypeError::invalid(
                "cannot materialize an iota of dynamically sized type f32[size]; stage it in an array program over \
                 `ArrayIrOperation`, whose `DynamicIota` constructor consumes one dimension operand per dynamic axis",
            ))),
        );
    }

    #[test]
    fn test_projected_context_iota() {
        let parent = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let context = ProjectedContext::<_, ArrayType>::new(parent.clone());
        let output_type = ArrayType::new_static(DataType::I32, [2, 3]);
        let output = context.iota(&output_type, 1).unwrap();
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
                let %0:i32[2, 3] = iota [type=i32[2, 3], dimension=1]
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_staging_context_iota() {
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_type = ArrayType::new_static(DataType::I32, [2, 3]);
        let output = context.iota(&output_type, 1).unwrap();
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
                let %0:i32[2, 3] = iota [type=i32[2, 3], dimension=1]
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_partial_evaluation_context_iota() {
        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let output_type = ArrayType::new_static(DataType::I32, [2, 3]);
        let output = context.iota(&output_type, 1).unwrap();
        let expected = Array::from_elements(output_type, &[0i32, 1, 2, 0, 1, 2]).unwrap();
        assert_eq!(output.value().unwrap().as_known(), Some(&expected));
    }

    #[test]
    fn test_batching_context_iota() {
        let context = BatchingContext::<_, ArrayBatching>::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 4);
        let output_type = ArrayType::new_static(DataType::I32, [2, 3]);
        let output = context.iota(&output_type, 1).unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(output.batch().value(), &Array::from_elements(output_type, &[0i32, 1, 2, 0, 1, 2]).unwrap(),);
    }

    #[test]
    fn test_differentiation_context_iota() {
        let context = DifferentiationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let output_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let output = context.iota(&output_type, 1).unwrap();
        assert_eq!(
            output.primal(),
            &Array::from_elements(output_type.clone(), &[0.0f32, 1.0, 2.0, 0.0, 1.0, 2.0]).unwrap(),
        );
        assert!(matches!(output.tangent(), MaybeZero::Zero(r#type) if r#type == &output_type));
    }
}
