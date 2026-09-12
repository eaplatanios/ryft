use std::fmt::Display;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayElement, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrType,
    ArrayIrValue, ArrayOperation, ArrayType, DataType, Dimension, dispatch_on_array_element_type,
};
use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
use crate::contexts::{Context, Domain, EagerContext, ProjectedContext, StagingContext};
use crate::differentiation::{
    DifferentiableType, DifferentiationContext, DifferentiationDual, DifferentiationPolicy, DifferentiationTracer,
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
    Operation, OperationFormatter, OperationProjection, OperationProvider, ProgramError, RegionInterface, Type,
    TypeError, TypeIdentityRenaming, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

/// Canonical operation name for [`OneOperation`].
pub const ONE_OPERATION_NAME: &str = "one";

/// [`Operation`] that has no inputs and that produces a single output that corresponds to the _one_ value for the
/// [`Type`] that it holds (i.e., for its `r#type` field). Note that for arrays, this would typically correspond to an
/// array of the right type and shape filled with ones.
///
/// This operation also serves as an [`OperationProvider`] request: it carries the requested output type while the
/// provider receives no input types. Composite operation families select the appropriate member operation from this
/// type; homogeneous families use their ordinary `From<OneOperation<T>>` conversion.
#[derive(Clone, Debug)]
pub struct OneOperation<T: Type> {
    /// [`Type`] of the value produced when this operation is interpreted.
    r#type: T,
}

impl<T: Type> OneOperation<T> {
    /// Creates a new [`OneOperation`].
    #[inline]
    pub fn new(r#type: T) -> Self {
        Self { r#type }
    }

    /// Returns the type of the value produced by this operation.
    #[inline]
    pub fn r#type(&self) -> &T {
        &self.r#type
    }
}

impl<T: Type> Display for OneOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type> Operation for OneOperation<T> {
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        ONE_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[T],
        _region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 0, TypeError);
        check_constructor_type_has_no_identity_references(ONE_OPERATION_NAME, &self.r#type)?;
        Ok(vec![self.r#type.clone()])
    }

    #[inline]
    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<T::Identity>) -> Result<Self, TypeError> {
        Ok(Self { r#type: self.r#type.rename_identities(renaming)? })
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, ONE_OPERATION_NAME)?
            .bracketed(|operation| operation.field("type", &self.r#type))
    }
}

impl<T: Type, C: Domain<Type = T> + One<C::Value>> InterpretableOperation<C> for OneOperation<T> {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 0, ProgramError);
        Ok(vec![context.one(&self.r#type)?])
    }
}

impl<T: Type, C: Context<Type = T, Operation: From<OneOperation<T>>>> PartiallyEvaluatableOperation<C>
    for OneOperation<T>
{
}

impl_nullary_batchable_operation!(@replicated OneOperation<ArrayType>);
impl_nullary_batchable_operation!(@member<ArrayIrType, ArrayIrBatchingPolicy> OneOperation<ArrayType>);
impl_non_differentiable_operation!(<T> OneOperation<T> where T: Type);
impl_nullary_transposable_operation!(<T> OneOperation<T> where T: Type);

impl_member_operation_for_array_ir_constant_operation!(OneOperation<ArrayType>);
impl_member_interpretable_operation_for_array_ir_constant_operation!(
    OneOperation<ArrayType>,
    One,
    |context, output_type, _operation| context.one(&output_type),
);

impl<A: Value<Type = ArrayType>> From<OneOperation<ArrayType>> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: OneOperation<ArrayType>) -> Self {
        // Prefer the homogeneous member encoding for identity-free static ones and the mixed dimension-operand
        // encoding for dynamic output types. Explicit mixed static constructors remain valid, but canonical lifts
        // normalize them to the homogeneous form.
        if operation
            .r#type()
            .shape()
            .dimensions()
            .iter()
            .any(|dimension| matches!(dimension, Dimension::Dynamic(_)))
        {
            Self::One(operation)
        } else {
            Self::Array(ArrayOperation::One(operation))
        }
    }
}

impl<T: Type, O: Operation<Type = T> + From<OneOperation<T>>> OperationProvider<T, OneOperation<T>> for O {
    type Operation = Self;

    #[inline]
    fn provide(request: OneOperation<T>, input_types: &[&T]) -> Result<Self, ProgramError> {
        check_count!("input", input_types, 0, ProgramError);
        Ok(Self::from(request))
    }
}

impl<A: Value<Type = ArrayType>> OperationProvider<ArrayIrType, OneOperation<ArrayIrType>> for ArrayIrOperation<A> {
    type Operation = Self;

    fn provide(request: OneOperation<ArrayIrType>, input_types: &[&ArrayIrType]) -> Result<Self, ProgramError> {
        check_count!("input", input_types, 0, ProgramError);
        let r#type = match request.r#type {
            ArrayIrType::Array(r#type) => r#type,
            ArrayIrType::Dimension(_) => {
                return Err(TypeError::invalid("cannot materialize a one for a first-class dimension type").into());
            }
            ArrayIrType::Reference(r#type) => {
                return Err(TypeError::invalid(format!(
                    "cannot materialize a one for reference type `{}`; a reference denotes an allocation and has \
                     no one value, so tangent and cotangent references are allocated by the differentiation rules",
                    r#type,
                ))
                .into());
            }
        };
        check_constructor_type_has_no_identity_references(ONE_OPERATION_NAME, &r#type)?;
        Ok(Self::Array(ArrayOperation::One(OneOperation::new(r#type))))
    }
}

/// Represents the ability to synthesize a _one_ value for a given [`Type`] in an interpretation context. [`One`]
/// is the [`Type`]-driven counterpart to [`OneLike`](super::OneLike). It is what [`OneOperation`] needs for its
/// [`InterpretableOperation`] implementation, and it lives on the context because producing an eager value can be
/// backend- or context-dependent.
pub trait One<V: Typed> {
    /// Returns a _one_ value for the provided [`Type`].
    fn one(&self, r#type: &V::Type) -> Result<V, ProgramError>;
}

impl<O: Operation<Type = ArrayType>> One<Array> for EagerContext<Array, O> {
    fn one(&self, r#type: &ArrayType) -> Result<Array, ProgramError> {
        match r#type.data_type() {
            DataType::Token | DataType::Zero => {
                Err(TypeError::invalid(format!("data type `{}` cannot represent one", r#type.data_type())).into())
            }
            data_type => dispatch_on_array_element_type!(data_type, |Element| {
                Array::from_fn_elements(r#type.clone(), |_| Ok(Element::one()?))
            }),
        }
    }
}

impl<V: Value<Type = ArrayType>, O: Operation<Type = ArrayIrType>> One<ArrayIrValue<V>>
    for EagerContext<ArrayIrValue<V>, O>
where
    EagerContext<V, ArrayOperation<V>>: One<V>,
{
    #[inline]
    fn one(&self, r#type: &ArrayIrType) -> Result<ArrayIrValue<V>, ProgramError> {
        let r#type = <&ArrayType>::try_from(r#type)?;
        Ok(ArrayIrValue::Array(EagerContext::<V, ArrayOperation<V>>::new().one(r#type)?))
    }
}

impl<C: Context, T: Type> One<<C::Value as ValueProjection<T>>::Projected> for ProjectedContext<C, T>
where
    C::Value: ValueProjection<T, Projected: Value<Type = T>>,
    C::Constant: ValueProjection<T, Projected: Value<Type = T>>,
    C::Operation: OperationProjection<T, Projected: From<OneOperation<T>>>,
{
    #[inline]
    fn one(&self, r#type: &T) -> Result<<C::Value as ValueProjection<T>>::Projected, ProgramError> {
        Ok(self.bind(OneOperation::new(r#type.clone()), Vec::new(), &[])?.remove(0))
    }
}

impl<C: StagingContext> One<Tracer<C>> for C
where
    C::Operation: OperationProvider<C::Type, OneOperation<C::Type>, Operation = C::Operation>,
{
    #[inline]
    fn one(&self, r#type: &C::Type) -> Result<Tracer<C>, ProgramError> {
        let mut outputs =
            self.stage_nullary_operation(C::Operation::provide(OneOperation::new(r#type.clone()), &[])?)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Context> One<PartialTracer<C>> for PartialEvaluationContext<C>
where
    C::Operation: PartiallyEvaluatableOperation<C>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + OperationProvider<C::Type, OneOperation<C::Type>, Operation = C::Operation>,
{
    #[inline]
    fn one(&self, r#type: &C::Type) -> Result<PartialTracer<C>, ProgramError> {
        let mut outputs = self.bind(C::Operation::provide(OneOperation::new(r#type.clone()), &[])?, Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Context<Type = ArrayType> + One<C::Value>> One<BatchingTracer<C, ArrayBatchingPolicy>>
    for BatchingContext<C, ArrayBatchingPolicy>
{
    #[inline]
    fn one(&self, r#type: &ArrayType) -> Result<BatchingTracer<C, ArrayBatchingPolicy>, ProgramError> {
        let batch = ArrayBatch::new(self.parent().one(r#type)?, BatchAxis::replicated())?;
        Ok(BatchingTracer::new(self.clone(), batch))
    }
}

impl<C: Context<Type: DifferentiableType> + One<C::Value>, P: DifferentiationPolicy<C>> One<DifferentiationTracer<C, P>>
    for DifferentiationContext<C, P>
{
    #[inline]
    fn one(&self, r#type: &C::Type) -> Result<DifferentiationTracer<C, P>, ProgramError> {
        let dual = DifferentiationDual::new_with_zero_tangent(self.primal().one(r#type)?)?;
        Ok(DifferentiationTracer::new(dual, self.clone()))
    }
}

/// Represents the ability to construct an [`Array`] of ones whose shape includes _dynamic_ (i.e., runtime) dimensions.
/// Unlike [`One`], this capability supplies each dynamic axis with an explicit dimension value. Static axes retain
/// their declared sizes. Dynamic axes take their sizes from the inputs in axis order. Repeated dimension identities
/// require a corresponding input for every occurrence. Each input must have the exact dimension identity declared
/// by its axis. The caller must thus provide the same dimension value for repeated occurrences.
///
/// Note that a fully static output type is also accepted with no dimension inputs. The same capability works with eager
/// mixed-IR values and with tracer values, where construction records the dimension inputs in the staged program.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{
/// #     Array, ArrayIrOperation, ArrayIrValue, ArrayType, DataType, Dimension, DimensionBounds, DimensionType,
/// #     DimensionValue, DimensionVariable, DynamicOne, EagerContext, Shape,
/// # };
/// let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
/// let size = DimensionVariable::new("size", DimensionBounds::unbounded());
/// let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(size.clone())]));
/// let dimension = ArrayIrValue::Dimension(DimensionValue::new(DimensionType::new(size), 3).unwrap());
/// assert_eq!(
///     context.dynamic_one(&output_type, &[dimension]),
///     Ok(ArrayIrValue::Array(Array::vector(vec![1.0f32; 3]).unwrap())),
/// );
/// ```
pub trait DynamicOne<V: Typed> {
    /// Constructs an array of ones with explicit values for its dynamic dimensions.
    ///
    /// # Parameters
    ///
    ///   - `type`: Output [`ArrayType`] that may contain dynamic dimensions. Each dynamic axis names the dimension
    ///     identity that its corresponding input must carry.
    ///   - `dimensions`: Contains one dimension value per dynamic axis, in axis order. Static axes do not consume input
    ///     dimensions provided this way. Repeated identities still consume one input for each axis that uses them.
    fn dynamic_one(&self, r#type: &ArrayType, dimensions: &[V]) -> Result<V, ProgramError>;
}

impl<C: Context<Type = ArrayIrType, Operation: From<OneOperation<ArrayType>>>> DynamicOne<C::Value> for C {
    #[inline]
    fn dynamic_one(&self, r#type: &ArrayType, dimensions: &[C::Value]) -> Result<C::Value, ProgramError> {
        validate_dynamic_constant_dimensions(ONE_OPERATION_NAME, r#type, dimensions)?;
        let mut outputs = self.bind(OneOperation::new(r#type.clone()), Vec::new(), dimensions)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayBatch, ArrayBatchingPolicy, ArrayIrOperation, ArrayIrValue, ArrayOperation, DataType, Dimension,
        DimensionBounds, DimensionType, DimensionValue, DimensionVariable, Layout, Shape, StridedLayout, i4,
    };
    use crate::batching::{BatchAxis, BatchableOperation, BatchingContext};
    use crate::contexts::EagerContext;
    use crate::differentiation::{
        ForwardModeDifferentiate, LinearizationTracer, TransposableOperation, TranspositionContext, differentiate_at,
    };
    use crate::interpretation::InterpretableOperation;
    use crate::macros::check_operation_partial_evaluation;
    use crate::operations::constants::constant::ConstantOperation;
    use crate::operations::math::mul::Mul;
    use crate::operations::math::reduce::{Reduce, ReductionKind};
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{EmptyRegionDriver, MaybeZero, Operation, ProgramBuilder, ReferenceType};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_one() {
        // Verify the operation's stored type, identity, and rendering.
        let operation = OneOperation::new(ArrayType::scalar(DataType::F64));
        assert_eq!(operation.name(), ONE_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "one [type=f64[]]");
        assert_eq!(operation.r#type(), &ArrayType::scalar(DataType::F64));
        // Verify the operation's textual form when it appears in a program.
        let mut builder = ProgramBuilder::<Array, OneOperation<ArrayType>>::new();
        let output = builder.add_instruction(operation, Vec::new(), vec![], None).unwrap()[0];
        let program = builder.build::<(), Array>(vec![output], (), Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f64[] = one [type=f64[]]
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_one_type_inference() {
        let operation = OneOperation::new(ArrayType::scalar(DataType::F64));
        assert_eq!(operation.infer_output_types(&[], &[]), Ok(vec![ArrayType::scalar(DataType::F64)]));

        // Nullary construction rejects output types with ungrounded identity references, while a definition-position
        // identity remains valid because the constructed value establishes it itself.
        let rows = crate::arrays::DimensionVariable::new("rows", DimensionBounds::non_negative(Some(8)).unwrap());
        let dynamic_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Static(3)]));
        assert_eq!(
            OneOperation::new(dynamic_type).infer_output_types(&[], &[]),
            Err(TypeError::invalid(
                "`one` cannot construct type f32[rows, 3] without operands because it references identity rows",
            )),
        );
        let dimension_type = DimensionType::new(rows);
        assert_eq!(OneOperation::new(dimension_type.clone()).infer_output_types(&[], &[]), Ok(vec![dimension_type]),);
    }

    #[test]
    fn test_one_type_inference_dynamic_identity_instantiation() {
        let formal = DimensionVariable::new("formal", DimensionBounds::new(1, Some(5)).unwrap());
        let caller = DimensionVariable::new("caller", DimensionBounds::new(2, Some(4)).unwrap());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let extent = builder.add_input(DimensionType::new(formal.clone()).into());
        let output = builder
            .add_instruction(
                OneOperation::new(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(formal)]))),
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
        let ArrayIrOperation::One(instantiated_one) = instruction.operation() else {
            panic!("expected the instantiated operation to remain a dynamic one");
        };
        assert_eq!(
            instantiated_one.r#type(),
            &ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(caller.clone())])),
        );
        assert_eq!(
            instantiated
                .interpret(vec![ArrayIrValue::Dimension(DimensionValue::new(DimensionType::new(caller), 3).unwrap())]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f32, 1.0, 1.0]).unwrap())]),
        );
    }

    #[test]
    fn test_one_interpretation() {
        let operation = OneOperation::new(ArrayType::scalar(DataType::F64));
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Ok(vec![Array::scalar(1.0).unwrap()]),
        );

        let context = EagerContext::<Array>::new();

        // Verify canonical rank-zero one values across every supported data-type family.
        for (r#type, expected) in [
            (DataType::Boolean, Array::scalar(true).unwrap()),
            (DataType::I8, Array::scalar(1i8).unwrap()),
            (DataType::I16, Array::scalar(1i16).unwrap()),
            (DataType::I32, Array::scalar(1i32).unwrap()),
            (DataType::I64, Array::scalar(1i64).unwrap()),
            (DataType::U8, Array::scalar(1u8).unwrap()),
            (DataType::U16, Array::scalar(1u16).unwrap()),
            (DataType::U32, Array::scalar(1u32).unwrap()),
            (DataType::U64, Array::scalar(1u64).unwrap()),
            (DataType::BF16, Array::scalar(bf16::ONE).unwrap()),
            (DataType::F16, Array::scalar(f16::ONE).unwrap()),
            (DataType::F32, Array::scalar(1.0f32).unwrap()),
            (DataType::F64, Array::scalar(1.0f64).unwrap()),
        ] {
            assert_eq!(context.one(&ArrayType::scalar(r#type)), Ok(expected));
        }

        // Rank-positive arrays preserve the requested geometry.
        let output_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let expected = Array::from_elements(output_type.clone(), &[1.0f32; 6]).unwrap();
        assert_eq!(context.one(&output_type), Ok(expected.clone()));

        // Token, zero-space, and dynamically shaped eager arrays cannot be materialized as ones.
        for data_type in [DataType::Token, DataType::Zero] {
            assert_eq!(
                context.one(&ArrayType::scalar(data_type)),
                Err(ProgramError::Type(TypeError::invalid(format!("data type `{data_type}` cannot represent one",)))),
            );
        }
        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("size", DimensionBounds::unbounded()))]),
        );
        assert!(matches!(
            context.one(&dynamic_type),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot materialize a value of dynamically sized type f32[size]; dynamically shaped \
                               values exist only in array programs over `ArrayIrOperation`",
        ));

        let r#type = ArrayType::new_static(DataType::F32, [2, 2]);
        assert_eq!(
            context.one(&r#type),
            Array::from_elements(r#type.clone(), &[1.0f32; 4]).map_err(|_| unreachable!())
        );
        // Constructors dispatch over element codecs that have no scalar representation and honor physical layout.
        let strided_type =
            ArrayType::new_static(DataType::I4, [3]).with_layout(Layout::Strided(StridedLayout::new(vec![-1])));
        let one = context.one(&strided_type).unwrap();
        assert_eq!(one.elements::<i4>(), Ok(vec![i4::new(1).unwrap(); 3]));
        assert_eq!(one.storage_bytes(), [1, 1, 1]);

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
        assert!(matches!(
            context.one(&dynamic_type),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot materialize a value of dynamically sized type f64[dynamic, 3]; dynamically \
                               shaped values exist only in array programs over `ArrayIrOperation`",
        ));

        // Composite eager one materialization delegates array members and rejects first-class dimensions.
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        assert_eq!(context.one(&ArrayIrType::Array(output_type)), Ok(ArrayIrValue::Array(expected)));
        let dimension_type =
            ArrayIrType::Dimension(DimensionType::new(DimensionVariable::new("size", DimensionBounds::unbounded())));
        assert_eq!(
            context.one(&dimension_type),
            Err(ProgramError::Type(TypeError::invalid("expected array type but got dimension type"))),
        );
    }

    #[test]
    fn test_one_partial_evaluation() {
        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let output_type = ArrayType::new_static(DataType::F32, [2]);
        let output = context.one(&output_type).unwrap();
        let expected = Array::from_elements(output_type, &[1.0f32; 2]).unwrap();
        assert_eq!(output.value().unwrap().as_known(), Some(&expected));
    }

    #[test]
    fn test_one_partial_evaluation_dynamic() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let extent = ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap());
        let output = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 1.0, 1.0]).unwrap());
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = OneOperation::new(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]),
            )),
            cases = [
                {
                    inputs = [(@known, extent.clone())],
                    outputs = [(@known, output.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [(@unknown(type = extent_type.into(), replay = extent))],
                    outputs = [(@residual, output)],
                    residual_instructions = 1,
                },
            ],
        );
    }

    #[test]
    fn test_one_batching() {
        // A nullary one does not acquire a physical batch axis because the same value serves every batch item.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let outputs: Vec<ArrayBatch<Array>> = OneOperation::new(scalar_type.clone())
            .batch(
                &BatchingContext::new(EagerContext::<Array, ConstantOperation<Array>>::new(), 2),
                &EmptyRegionDriver,
                &[],
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].r#type().into_owned(), scalar_type);
        assert_eq!(outputs[0].value().to_f64s(), vec![1.0]);

        let context =
            BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 4);
        let output_type = ArrayType::new_static(DataType::F32, [2]);
        let output = context.one(&output_type).unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(output.batch().value(), &Array::from_elements(output_type, &[1.0f32; 2]).unwrap());
    }

    #[test]
    fn test_one_differentiation() {
        let context = DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new());
        let output_type = ArrayType::new_static(DataType::F32, [2]);
        let output = context.one(&output_type).unwrap();
        assert_eq!(output.primal(), &Array::from_elements(output_type.clone(), &[1.0f32; 2]).unwrap());
        assert!(matches!(output.tangent(), MaybeZero::Zero(r#type) if r#type == &output_type));
    }

    #[test]
    fn test_one_differentiation_dynamic() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let extent = builder.add_input(extent_type.clone().into());
        let output = builder
            .add_instruction(
                OneOperation::new(ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]),
                )),
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
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 1.0, 1.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![0.0_f64, 0.0, 0.0]).unwrap()),
            ]),
        );
        assert_eq!(jvp.instructions().len(), 2);
        assert!(matches!(jvp.instructions()[0].operation(), ArrayIrOperation::One(_)));
        assert!(matches!(jvp.instructions()[1].operation(), ArrayIrOperation::Zero(_)));
        assert_eq!(jvp.instructions()[0].inputs(), jvp.instructions()[1].inputs());

        // The direct transform context must likewise run the explicit rule rather than taking its all-structural-zero
        // shortcut: a nullary zero cannot recover the dynamic extent after the closure returns.
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let dynamic_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]));
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let (primal, tangent) = context
            .jvp(
                move |extent, ()| {
                    let context = extent.context().clone();
                    Ok(context.bind(OneOperation::new(dynamic_type), Vec::new(), &[extent])?.remove(0))
                },
                ArrayIrValue::Dimension(DimensionValue::new(extent_type, 3).unwrap()),
                ArrayIrValue::Array(Array::new(ArrayType::scalar(DataType::Zero), Vec::new()).unwrap()),
                (),
            )
            .unwrap();
        assert_eq!(primal, ArrayIrValue::Array(Array::vector(vec![1.0_f64, 1.0, 1.0]).unwrap()));
        assert_eq!(tangent, ArrayIrValue::Array(Array::vector(vec![0.0_f64, 0.0, 0.0]).unwrap()));
    }

    #[test]
    fn test_one_differentiation_composite_gradient_seed() {
        // Reverse-mode gradient terminals seed the output cotangent by binding a `one` of the composite cotangent
        // type through the fallible provider, which constructs the canonical array member encoding. The differentiated
        // function reaches ordinary array math through the array member projection, because homogeneous array
        // capabilities deliberately do not exist at the composite level.
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0]).unwrap());
        let squared_sum = |input: LinearizationTracer<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>| {
            let input = <_ as ValueProjection<ArrayType>>::into_projected(input)?;
            let squared = input.mul(&input)?;
            Ok::<_, ProgramError>(ValueProjection::<ArrayType>::from_projected(
                squared.reduce(&[0], ReductionKind::Sum),
            ))
        };

        let (value, gradient) = differentiate_at(input.clone()).value_and_gradient(squared_sum).unwrap();
        assert_eq!(value, ArrayIrValue::Array(Array::scalar(14.0_f64).unwrap()));
        assert_eq!(gradient, ArrayIrValue::Array(Array::vector(vec![2.0_f64, 4.0, 6.0]).unwrap()));
        assert_eq!(differentiate_at(input).gradient(squared_sum).unwrap(), gradient);
    }

    #[test]
    fn test_one_transposition() {
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = context.input(ArrayType::scalar(DataType::F64));
        let input_cotangents = OneOperation::new(ArrayType::scalar(DataType::F64))
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
    fn test_one_transposition_dynamic() {
        // Dynamic constructors depend on their extent operands only as non-differentiable shape inputs, so every
        // extent receives a structural-zero cotangent regardless of the output cotangent being live.
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let output_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]));
        let operation = ArrayIrOperation::<Array>::from(OneOperation::new(output_type.clone()));
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
    fn test_operation_provider_one() {
        // Homogeneous operation families construct nullary operations through their ordinary
        // `From<OneOperation<T>>` conversion.
        let static_type = ArrayType::new_static(DataType::F32, [2]);
        let ArrayOperation::<Array>::One(operation) =
            ArrayOperation::<Array>::provide(OneOperation::new(static_type.clone()), &[]).unwrap()
        else {
            panic!("expected a homogeneous one operation");
        };
        assert_eq!(operation.r#type(), &static_type);

        // Output types belong to the request; nullary construction rejects any operand types.
        assert_eq!(
            ArrayOperation::<Array>::provide(OneOperation::new(static_type.clone()), &[&static_type]).unwrap_err(),
            ProgramError::InvalidInputCount { expected: 0, actual: 1 },
        );
        let composite_type = ArrayIrType::Array(static_type.clone());
        assert_eq!(
            ArrayIrOperation::<Array>::provide(OneOperation::new(composite_type.clone()), &[&composite_type])
                .unwrap_err(),
            ProgramError::InvalidInputCount { expected: 0, actual: 1 },
        );

        // The composite provider projects a valid operand-free array one into the homogeneous member family.
        let ArrayIrOperation::<Array>::Array(ArrayOperation::One(operation)) =
            ArrayIrOperation::<Array>::provide(OneOperation::new(ArrayIrType::Array(static_type.clone())), &[])
                .unwrap()
        else {
            panic!("expected a composite homogeneous one operation");
        };
        assert_eq!(operation.r#type(), &static_type);

        // Operand-free construction cannot resolve a dynamic identity. Dynamic mixed ones must instead receive their
        // concrete extents as dimension operands.
        let size = DimensionVariable::new("size", DimensionBounds::unbounded());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(size.clone())]));
        assert_eq!(
            ArrayIrOperation::<Array>::provide(OneOperation::new(ArrayIrType::Array(dynamic_type)), &[]).unwrap_err(),
            ProgramError::Type(TypeError::invalid(
                "`one` cannot construct type f32[size] without operands because it references identity size",
            )),
        );

        // First-class dimensions and references are not algebraic values. In particular, a reference cannot be replaced
        // by a one of its referent type. The differentiation rules allocate tangent and cotangent references instead of
        // ever materializing a one reference.
        assert_eq!(
            ArrayIrOperation::<Array>::provide(
                OneOperation::new(ArrayIrType::Dimension(DimensionType::new(size))),
                &[],
            )
            .unwrap_err(),
            ProgramError::Type(TypeError::invalid("cannot materialize a one for a first-class dimension type")),
        );
        let reference_type = ReferenceType::new(static_type);
        assert_eq!(
            ArrayIrOperation::<Array>::provide(OneOperation::new(ArrayIrType::Reference(reference_type.clone())), &[])
                .unwrap_err(),
            ProgramError::Type(TypeError::invalid(format!(
                "cannot materialize a one for reference type `{reference_type}`; a reference denotes an allocation \
                 and has no one value, so tangent and cotangent references are allocated by the differentiation rules",
            ))),
        );
    }

    #[test]
    fn test_projected_context_one() {
        let parent = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let context = ProjectedContext::<_, ArrayType>::new(parent.clone());
        let output_type = ArrayType::new_static(DataType::F32, [2]);
        let output = context.one(&output_type).unwrap();
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
                let %0:f32[2] = one [type=f32[2]]
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_staging_context_one() {
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_type = ArrayType::new_static(DataType::F32, [2]);
        let output = context.one(&output_type).unwrap();
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
                let %0:f32[2] = one [type=f32[2]]
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dynamic_one_interpretation() {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let size = DimensionVariable::new("size", DimensionBounds::non_negative(Some(8)).unwrap());
        let dimension = ArrayIrValue::Dimension(DimensionValue::new(DimensionType::new(size.clone()), 2).unwrap());
        let output_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(size.clone()), Dimension::Static(3), Dimension::Dynamic(size)]),
        );

        // Static axes consume no operands; each occurrence of a dynamic identity consumes its own operand.
        assert_eq!(
            context.dynamic_one(&output_type, &[dimension.clone(), dimension]),
            Ok(ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3, 2]), &[1.0f32; 12]).unwrap(),
            )),
        );
        assert_eq!(
            context.dynamic_one(&ArrayType::scalar(DataType::F32), &[]),
            Ok(ArrayIrValue::Array(Array::scalar(1.0f32).unwrap())),
        );
    }

    #[test]
    fn test_dynamic_one_invalid_dimensions() {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let size = DimensionVariable::new("size", DimensionBounds::non_negative(Some(8)).unwrap());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(size)]));
        assert_eq!(
            context.dynamic_one(&output_type, &[]),
            Err(ProgramError::Type(TypeError::invalid(
                "`one` expects one dimension operand per dynamic output dimension (1) but got 0 operands",
            ))),
        );
        assert_eq!(
            context.dynamic_one(&output_type, &[ArrayIrValue::Array(Array::scalar(2.0f32).unwrap())]),
            Err(ProgramError::Type(TypeError::invalid("`one` operand 0 must be a dimension but has type f32[]"))),
        );
        let other = DimensionVariable::new("other", DimensionBounds::non_negative(Some(8)).unwrap());
        let dimension = ArrayIrValue::Dimension(DimensionValue::new(DimensionType::new(other), 2).unwrap());
        assert_eq!(
            context.dynamic_one(&output_type, &[dimension]),
            Err(ProgramError::Type(TypeError::invalid(
                "`one` operand 0 has type dimension<other ∈ [0, 8)> but the output shape requires \
                 dimension<size ∈ [0, 8)>",
            ))),
        );
    }

    #[test]
    fn test_dynamic_one_staging() {
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let size = DimensionVariable::new("size", DimensionBounds::non_negative(Some(8)).unwrap());
        let dimension_type = DimensionType::new(size.clone());
        let dimension = context.input(dimension_type.clone().into());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(size)]));
        let output = context.dynamic_one(&output_type, std::slice::from_ref(&dimension)).unwrap();
        assert_eq!(output.r#type().as_ref(), &ArrayIrType::Array(output_type.clone()));
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
            panic!("expected one dynamic one instruction");
        };
        assert!(
            matches!(instruction.operation(), ArrayIrOperation::One(operation) if operation.r#type() == &output_type)
        );
        assert_eq!(instruction.inputs(), &[dimension.atom_id().unwrap()]);
        let extent = ArrayIrValue::Dimension(DimensionValue::new(dimension_type, 3).unwrap());
        assert_eq!(
            program.interpret(vec![extent.clone()]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![1.0f32; 3]).unwrap())]),
        );

        // The staged operation retains its existing derivative rule: shape inputs receive no live tangent.
        assert_eq!(
            program.jvp().unwrap().interpret(vec![extent]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0f32; 3]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![0.0f32; 3]).unwrap()),
            ]),
        );
    }
}
