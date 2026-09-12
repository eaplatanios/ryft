use std::fmt::Display;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayElement, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrType,
    ArrayOperation, ArrayType, Dimension, dispatch_on_array_element_type,
};
use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
use crate::contexts::{Context, Domain, EagerContext, ProjectedContext, StagingContext};
use crate::differentiation::{
    DifferentiationContext, DifferentiationDual, DifferentiationPolicy, DifferentiationTracer,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{
    check_count, impl_non_differentiable_operation, impl_nullary_batchable_operation,
    impl_nullary_transposable_operation,
};
use crate::operations::constants::{
    check_constructor_type_has_no_identity_references, validate_dynamic_constant_dimensions,
};
use crate::partial::{PartialEvaluationContext, PartialTracer, PartiallyEvaluatableOperation};
use crate::programs::{
    Operation, OperationFormatter, OperationProjection, ProgramError, RegionInterface, Type, TypeError,
    TypeIdentityRenaming, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

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
            operation.field("dimension", self.dimension)
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

impl_nullary_batchable_operation!(@replicated IotaOperation<ArrayType>);
impl_nullary_batchable_operation!(@member<ArrayIrType, ArrayIrBatchingPolicy> IotaOperation<ArrayType>);
impl_non_differentiable_operation!(IotaOperation<ArrayType>);
impl_nullary_transposable_operation!(IotaOperation<ArrayType>);

impl_member_operation_for_array_ir_constant_operation!(IotaOperation<ArrayType>);
impl_member_interpretable_operation_for_array_ir_constant_operation!(
    IotaOperation<ArrayType>,
    Iota,
    |context, output_type, operation| context.iota(&output_type, operation.dimension()),
);

impl<A: Value<Type = ArrayType>> From<IotaOperation<ArrayType>> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: IotaOperation<ArrayType>) -> Self {
        // Prefer the homogeneous member encoding for static iotas and the mixed dimension-operand encoding for dynamic
        // output types. Explicit mixed static constructors remain valid, but canonical lifts normalize them to the
        // homogeneous form.
        if operation
            .r#type()
            .shape()
            .dimensions()
            .iter()
            .any(|dimension| matches!(dimension, Dimension::Dynamic(_)))
        {
            Self::Iota(operation)
        } else {
            Self::Array(ArrayOperation::Iota(operation))
        }
    }
}

/// Represents the ability to synthesize a value for a given [`Type`] whose elements increase from `0` along a chosen
/// dimension in an interpretation context. [`Iota`] is the [`Type`]-driven capability needed by [`IotaOperation`] for
/// its [`InterpretableOperation`] implementation, sitting alongside [`Zero`](crate::Zero), [`One`](super::One), and
/// [`Fill`](super::Fill) in the same type-driven family.
///
/// For arrays, a zero-sized axis produces an empty result, including when products of other axis sizes would overflow.
/// Non-empty arrays must have representable strides and storage sizes. Integer coordinates narrow to the element type,
/// and complex coordinates have a zero imaginary component.
pub trait Iota<V: Typed> {
    /// Returns a value of `type` whose elements increase from `0` along `dimension` and are constant along every other
    /// dimension.
    ///
    /// # Parameters
    ///
    ///   - `r#type`: [`Type`] of the value to produce.
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
                         over `ArrayIrOperation`, whose `Iota` constructor consumes one dimension operand per \
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

        // Empty arrays never evaluate the element function, so their unused strides need not be representable.
        let stride = if sizes.contains(&0) {
            1
        } else {
            sizes[dimension + 1..]
                .iter()
                .try_fold(1usize, |stride, size| stride.checked_mul(*size))
                .ok_or_else(|| {
                    TypeError::invalid(format!("iota stride for array type `{type}` cannot be represented"))
                })?
        };

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
        let mut outputs = self.bind(IotaOperation::new(r#type.clone(), dimension)?, Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
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

impl<C: Context<Type = ArrayType> + Iota<C::Value>> Iota<BatchingTracer<C, ArrayBatchingPolicy>>
    for BatchingContext<C, ArrayBatchingPolicy>
{
    #[inline]
    fn iota(
        &self,
        r#type: &ArrayType,
        dimension: usize,
    ) -> Result<BatchingTracer<C, ArrayBatchingPolicy>, ProgramError> {
        let batch = ArrayBatch::new(self.parent().iota(r#type, dimension)?, BatchAxis::replicated())?;
        Ok(BatchingTracer::new(self.clone(), batch))
    }
}

impl<C: Context<Type = ArrayType> + Iota<C::Value>, P: DifferentiationPolicy<C>> Iota<DifferentiationTracer<C, P>>
    for DifferentiationContext<C, P>
{
    #[inline]
    fn iota(&self, r#type: &ArrayType, dimension: usize) -> Result<DifferentiationTracer<C, P>, ProgramError> {
        let dual = DifferentiationDual::new_with_zero_tangent(self.primal().iota(r#type, dimension)?)?;
        Ok(DifferentiationTracer::new(dual, self.clone()))
    }
}

/// Represents the ability to construct an [`Array`] of coordinates whose shape includes _dynamic_ (i.e., runtime)
/// dimensions. Unlike [`Iota`], this capability supplies each dynamic axis with an explicit dimension value. Static
/// axes retain their declared sizes. Dynamic axes take their sizes from the inputs in axis order. Repeated dimension
/// identities require a corresponding input for every occurrence. Each input must have the exact dimension identity
/// declared by its axis. The caller must thus provide the same dimension value for repeated occurrences.
///
/// Note that a fully static output type is also accepted with no dimension inputs. The same capability works with eager
/// mixed-IR values and with tracer values, where construction records the dimension inputs in the staged program.
/// When a dynamic axis has bounds that admit only one extent, the inferred result type represents that axis as static.
/// Its dimension input is still required because the declared output type contains a dynamic axis.
///
/// Each element is its zero-based index along the selected axis, converted to the output element data type;
/// coordinates repeat along the other axes. Complex outputs have zero imaginary components.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{
/// #     Array, ArrayIrOperation, ArrayIrValue, ArrayType, DataType, Dimension, DimensionBounds, DimensionType,
/// #     DimensionValue, DimensionVariable, DynamicIota, EagerContext, Shape,
/// # };
/// let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
/// let size = DimensionVariable::new("size", DimensionBounds::unbounded());
/// let output_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(size.clone())]));
/// let dimension = ArrayIrValue::Dimension(DimensionValue::new(DimensionType::new(size), 3).unwrap());
/// assert_eq!(
///     context.dynamic_iota(&output_type, 0, &[dimension]),
///     Ok(ArrayIrValue::Array(Array::vector(vec![0i32, 1, 2]).unwrap())),
/// );
/// ```
pub trait DynamicIota<V: Typed> {
    /// Constructs an array of coordinates with explicit values for its dynamic dimensions. Returns an error for a
    /// non-numeric output element type, an out-of-range axis, or dimension inputs that do not match the output type.
    ///
    /// # Parameters
    ///
    ///   - `type`: Output [`ArrayType`] that may contain dynamic dimensions. Each dynamic axis names the dimension
    ///     identity that its corresponding input must carry.
    ///   - `dimension`: Zero-based output axis along which the coordinates increase.
    ///   - `dimensions`: Contains one dimension value per dynamic axis, in axis order. Static axes do not consume input
    ///     dimensions provided this way. Repeated identities still consume one input for each axis that uses them.
    fn dynamic_iota(&self, r#type: &ArrayType, dimension: usize, dimensions: &[V]) -> Result<V, ProgramError>;
}

impl<C: Context<Type = ArrayIrType, Operation: From<IotaOperation<ArrayType>>>> DynamicIota<C::Value> for C {
    #[inline]
    fn dynamic_iota(
        &self,
        r#type: &ArrayType,
        dimension: usize,
        dimensions: &[C::Value],
    ) -> Result<C::Value, ProgramError> {
        let operation = IotaOperation::new(r#type.clone(), dimension)?;
        validate_dynamic_constant_dimensions(IOTA_OPERATION_NAME, r#type, dimensions)?;
        let mut outputs = self.bind(operation, Vec::new(), dimensions)?;
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
        DimensionBounds, DimensionType, DimensionValue, DimensionVariable, Shape, i4, u4,
    };
    use crate::batching::{BatchAxis, BatchingContext};
    use crate::contexts::EagerContext;
    use crate::differentiation::{TransposableOperation, TranspositionContext};
    use crate::interpretation::InterpretableOperation;
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{EmptyRegionDriver, MaybeZero, Operation, ProgramBuilder};
    use crate::tracing::TracingContext;

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
        // Verify the operation's stored type and axis, identity, and rendering.
        let operation = IotaOperation::new(r#type.clone(), 1).unwrap();
        assert_eq!(operation.name(), IOTA_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "iota [type=f64[2, 3], dimension=1]");
        assert_eq!(operation.r#type(), &r#type);
        assert_eq!(operation.dimension(), 1);

        // Verify the operation's textual form when it appears in a program.
        let mut builder = ProgramBuilder::<Array, IotaOperation<ArrayType>>::new();
        let output = builder.add_instruction(operation, Vec::new(), vec![], None).unwrap()[0];
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
    fn test_iota_type_inference() {
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let operation = IotaOperation::new(r#type.clone(), 1).unwrap();
        assert_eq!(operation.infer_output_types(&[], &[]), Ok(vec![r#type.clone()]));

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
    }

    #[test]
    fn test_iota_type_inference_dynamic_identity_instantiation() {
        let formal = DimensionVariable::new("formal", DimensionBounds::new(1, Some(5)).unwrap());
        let caller = DimensionVariable::new("caller", DimensionBounds::new(2, Some(4)).unwrap());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let extent = builder.add_input(DimensionType::new(formal.clone()).into());
        let output = builder
            .add_instruction(
                IotaOperation::new(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(formal)])), 0)
                    .unwrap(),
                Vec::new(),
                vec![extent],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let caller_input = ArrayIrType::Dimension(DimensionType::new(caller.clone()));
        let instantiated = program.with_instantiated_type_identities(std::slice::from_ref(&caller_input)).unwrap();
        let [instruction] = instantiated.instructions() else {
            panic!("expected one instantiated instruction");
        };
        let ArrayIrOperation::Iota(instantiated_iota) = instruction.operation() else {
            panic!("expected the instantiated operation to remain a dynamic iota");
        };
        assert_eq!(
            instantiated_iota.r#type(),
            &ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(caller.clone())])),
        );
        assert_eq!(
            instantiated
                .interpret(vec![ArrayIrValue::Dimension(DimensionValue::new(DimensionType::new(caller), 3).unwrap())]),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements(
                    ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(3)])),
                    &[0i32, 1, 2],
                )
                .unwrap(),
            )]),
        );
    }

    #[test]
    fn test_iota_interpretation() {
        let r#type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let operation = IotaOperation::new(r#type.clone(), 1).unwrap();
        // Eager interpretation along axis one varies between columns and repeats across rows.
        let context = EagerContext::<Array, IotaOperation<ArrayType>>::new();
        let expected = Array::from_elements::<f64>(r#type.clone(), &[0.0, 1.0, 2.0, 0.0, 1.0, 2.0]).unwrap();
        assert_eq!(
            InterpretableOperation::<EagerContext<Array, IotaOperation<ArrayType>>>::interpret(
                &operation,
                &context,
                &EmptyRegionDriver,
                &[],
            ),
            Ok(vec![expected.clone()]),
        );

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

        // Integer coordinates narrow in two's-complement form rather than failing at the first unrepresentable index.
        assert_eq!(
            context.iota(&ArrayType::new_static(DataType::I4, [10]), 0).unwrap().elements::<i4>(),
            Ok(vec![
                i4::new(0).unwrap(),
                i4::new(1).unwrap(),
                i4::new(2).unwrap(),
                i4::new(3).unwrap(),
                i4::new(4).unwrap(),
                i4::new(5).unwrap(),
                i4::new(6).unwrap(),
                i4::new(7).unwrap(),
                i4::new(-8).unwrap(),
                i4::new(-7).unwrap(),
            ]),
        );

        // Empty arrays do not compute coordinates or unused strides, even if the nonzero sizes would overflow.
        let empty_type = ArrayType::new_static(DataType::I32, [0, usize::MAX, usize::MAX]);
        assert_eq!(context.iota(&empty_type, 0), Array::from_elements::<i32>(empty_type, &[]).map_err(Into::into));
        let empty_type = ArrayType::new_static(DataType::I32, [usize::MAX, usize::MAX, 0]);
        assert_eq!(context.iota(&empty_type, 0), Array::from_elements::<i32>(empty_type, &[]).map_err(Into::into));

        // Nonempty oversized geometry is rejected before allocating storage, in both debug and release builds.
        let oversized_type = ArrayType::new_static(DataType::I32, [1, usize::MAX, 2]);
        assert_eq!(
            context.iota(&oversized_type, 0),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "iota stride for array type `{oversized_type}` cannot be represented",
            )))),
        );
        let oversized_type = ArrayType::new_static(DataType::I32, [usize::MAX]);
        assert_eq!(
            context.iota(&oversized_type, 0),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "array type {oversized_type} requires more bytes than can be represented",
            )))),
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
                 `ArrayIrOperation`, whose `Iota` constructor consumes one dimension operand per dynamic axis",
            ))),
        );

        // Iota materializes coordinates along the requested dimension in the declared element data type.
        assert_eq!(
            context.iota(&ArrayType::new_static(DataType::I32, [2, 3]), 1).unwrap().elements::<i32>(),
            Ok(vec![0, 1, 2, 0, 1, 2]),
        );
        assert_eq!(context.iota(&ArrayType::new_static(DataType::F64, [3]), 0).unwrap().to_f64s(), vec![0.0, 1.0, 2.0]);
        assert_eq!(
            context
                .iota(&ArrayType::new_static(DataType::C64, [3]), 0)
                .unwrap()
                .elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(0.0, 0.0), ComplexNumber::new(1.0, 0.0), ComplexNumber::new(2.0, 0.0),]),
        );

        // Kernels that materialize a payload from a type reject dynamically sized types. `zero` and `one` share the
        // storage-level rejection raised by `ArrayAddressing::new`, while `iota` names the array-program route that
        // admits dynamic extents.
        let dynamic_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
                Dimension::Static(3),
            ]),
        );
        assert_eq!(
            context.iota(&dynamic_type, 1).unwrap_err().to_string(),
            "cannot materialize an iota of dynamically sized type f64[dynamic, 3]; stage it in an array program over \
             `ArrayIrOperation`, whose `Iota` constructor consumes one dimension operand per dynamic axis",
        );
    }

    #[test]
    fn test_iota_partial_evaluation() {
        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let output_type = ArrayType::new_static(DataType::I32, [2, 3]);
        let output = context.iota(&output_type, 1).unwrap();
        let expected = Array::from_elements(output_type, &[0i32, 1, 2, 0, 1, 2]).unwrap();
        assert_eq!(output.value().unwrap().as_known(), Some(&expected));
    }

    #[test]
    fn test_iota_batching() {
        let context =
            BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 4);
        let output_type = ArrayType::new_static(DataType::I32, [2, 3]);
        let output = context.iota(&output_type, 1).unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(output.batch().value(), &Array::from_elements(output_type, &[0i32, 1, 2, 0, 1, 2]).unwrap(),);
    }

    #[test]
    fn test_iota_differentiation() {
        let context = DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new());
        let output_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let output = context.iota(&output_type, 1).unwrap();
        assert_eq!(
            output.primal(),
            &Array::from_elements(output_type.clone(), &[0.0f32, 1.0, 2.0, 0.0, 1.0, 2.0]).unwrap(),
        );
        assert!(matches!(output.tangent(), MaybeZero::Zero(r#type) if r#type == &output_type));
    }

    #[test]
    fn test_iota_differentiation_dynamic() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let extent = builder.add_input(extent_type.clone().into());
        let output = builder
            .add_instruction(
                IotaOperation::new(
                    ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())])),
                    0,
                )
                .unwrap(),
                Vec::new(),
                vec![extent],
                None,
            )
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
                ArrayIrValue::Array(Array::vector(vec![0.0_f64, 1.0, 2.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![0.0_f64, 0.0, 0.0]).unwrap()),
            ]),
        );
        assert_eq!(jvp.instructions().len(), 2);
        assert!(matches!(jvp.instructions()[0].operation(), ArrayIrOperation::Iota(_)));
        assert!(matches!(jvp.instructions()[1].operation(), ArrayIrOperation::Zero(_)));
        assert_eq!(jvp.instructions()[0].inputs(), jvp.instructions()[1].inputs());
    }

    #[test]
    fn test_iota_transposition() {
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = context.input(ArrayType::new_static(DataType::F64, [2]));
        let input_cotangents = IotaOperation::new(ArrayType::new_static(DataType::F64, [2]), 0)
            .unwrap()
            .transpose(
                &mut TranspositionContext::new(context),
                &EmptyRegionDriver,
                &[],
                &[MaybeZero::Value(output_cotangent)],
                &[],
            )
            .unwrap();
        assert_eq!(input_cotangents, ());
    }

    #[test]
    fn test_iota_transposition_dynamic() {
        // Dynamic constructors depend on their extent operands only as non-differentiable shape inputs, so every
        // extent receives a structural zero cotangent regardless of the output cotangent being live.
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let output_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]));
        let operation = ArrayIrOperation::<Array>::from(IotaOperation::new(output_type.clone(), 0).unwrap());
        let mut context =
            TranspositionContext::new(TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new());
        let output_cotangent = context.input(output_type.clone().into());
        let inputs = [PartialValue::Unknown(extent_type.clone().into())];
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        operation
            .transpose(&mut context, &EmptyRegionDriver, &inputs, &[MaybeZero::Value(output_cotangent)], &accumulators)
            .unwrap();
        let cotangents = context.take_cotangents(&accumulators).unwrap();
        let [cotangent] = cotangents.as_slice() else {
            panic!("expected one cotangent per operation input");
        };
        assert!(matches!(cotangent, MaybeZero::Zero(_)));
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
    fn test_dynamic_iota() {
        let extent_type = DimensionType::new(DimensionVariable::new("extent", DimensionBounds::unbounded()));
        let output_type = ArrayType::new(
            DataType::I32,
            Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone()), Dimension::Static(2)]),
        );
        let extent = ArrayIrValue::Dimension(DimensionValue::new(extent_type, 3).unwrap());
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        assert_eq!(
            context.dynamic_iota(&output_type, 0, &[extent]),
            Ok(ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::I32, [3, 2]), &[0i32, 0, 1, 1, 2, 2]).unwrap(),
            )),
        );
        assert_eq!(
            context.dynamic_iota(&ArrayType::new_static(DataType::I32, [3]), 0, &[]),
            Ok(ArrayIrValue::Array(Array::vector(vec![0i32, 1, 2]).unwrap())),
        );
    }

    #[test]
    fn test_dynamic_iota_invalid_axis() {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        assert_eq!(
            context.dynamic_iota(&ArrayType::new_static(DataType::I32, [3]), 1, &[]),
            Err(TypeError::invalid("`iota` dimension 1 is out of bounds for rank 1").into()),
        );
    }
    #[test]
    fn test_dynamic_iota_invalid_dimensions() {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let output_type = ArrayType::new(
            DataType::I32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("extent", DimensionBounds::unbounded()))]),
        );
        assert_eq!(
            context.dynamic_iota(&output_type, 0, &[]),
            Err(TypeError::invalid(
                "`iota` expects one dimension operand per dynamic output dimension (1) but got 0 operands",
            )
            .into()),
        );
        assert_eq!(
            context.dynamic_iota(&output_type, 0, &[ArrayIrValue::Array(Array::scalar(3.0f32).unwrap())]),
            Err(TypeError::invalid("`iota` operand 0 must be a dimension but has type f32[]").into()),
        );

        // Matching diagnostic names and bounds do not make separately created identities interchangeable.
        let distinct = DimensionVariable::new("extent", DimensionBounds::unbounded());
        let dimension = ArrayIrValue::Dimension(DimensionValue::new(DimensionType::new(distinct), 3).unwrap());
        assert_eq!(
            context.dynamic_iota(&output_type, 0, &[dimension]),
            Err(ProgramError::Type(TypeError::invalid(
                "`iota` operand 0 has type dimension<extent ∈ [0, ∞)> but the output shape requires \
                 dimension<extent ∈ [0, ∞)>",
            ))),
        );
    }

    #[test]
    fn test_dynamic_iota_staging() {
        let extent_type = DimensionType::new(DimensionVariable::new("extent", DimensionBounds::unbounded()));
        let output_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]));
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let extent = context.input(extent_type.clone().into());
        let output = context.dynamic_iota(&output_type, 0, &[extent]).unwrap();
        assert_eq!(output.r#type().as_ref(), &ArrayIrType::Array(output_type));
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let [instruction] = program.instructions() else {
            panic!("expected one dynamic iota instruction");
        };
        assert!(matches!(instruction.operation(), ArrayIrOperation::Iota(_)));
        let extent = ArrayIrValue::Dimension(DimensionValue::new(extent_type, 3).unwrap());
        assert_eq!(
            program.interpret(vec![extent.clone()]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0f64, 1.0, 2.0]).unwrap())]),
        );
        // Transform replay retains the extent operand for both coordinates and their zero tangent.
        assert_eq!(
            program.jvp().unwrap().interpret(vec![extent]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![0.0f64, 1.0, 2.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![0.0f64, 0.0, 0.0]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_dynamic_iota_staging_singleton_dimension() {
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let size = DimensionVariable::new("size", DimensionBounds::new(3, Some(4)).unwrap());
        let dimension_type = DimensionType::new(size.clone());
        let dimension = context.input(dimension_type.clone().into());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(size)]));
        let output = context.dynamic_iota(&output_type, 0, std::slice::from_ref(&dimension)).unwrap();

        // Singleton bounds refine the inferred result without removing the declared dynamic input.
        assert_eq!(output.r#type().as_ref(), &ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])));
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let [instruction] = program.instructions() else {
            panic!("expected one dynamic iota instruction");
        };
        assert!(
            matches!(instruction.operation(), ArrayIrOperation::Iota(operation) if operation.r#type() == &output_type)
        );
        assert_eq!(instruction.inputs(), &[dimension.atom_id().unwrap()]);
        let extent = ArrayIrValue::Dimension(DimensionValue::new(dimension_type, 3).unwrap());
        assert_eq!(
            program.interpret(vec![extent.clone()]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0f32, 1.0, 2.0]).unwrap())]),
        );
        assert_eq!(
            program.jvp().unwrap().interpret(vec![extent]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![0.0f32, 1.0, 2.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![0.0f32; 3]).unwrap()),
            ]),
        );
    }
}
