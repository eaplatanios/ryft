use std::fmt::Display;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatching, ArrayElement, ArrayIrBatching, ArrayIrOperation, ArrayIrType, ArrayIrValue,
    ArrayOperation, ArrayType, DataType, dispatch_on_array_element_type,
};
use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
use crate::contexts::{Context, Domain, EagerContext, ProjectedContext, StagingContext};
use crate::differentiation::{DifferentiableType, DifferentiationContext, DifferentiationDual, DifferentiationTracer};
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

/// Canonical operation name for [`OneOperation`].
pub const ONE_OPERATION_NAME: &str = "one";

/// [`Operation`] that has no inputs and that produces a single output that corresponds to the _one_ value for the
/// [`Type`] that it holds (i.e., for its `r#type` field). Note that for arrays, this would typically correspond to an
/// array of the right type and shape filled with ones.
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

impl_non_differentiable_operation!(<T> OneOperation<T> where T: Type);
impl_nullary_transposable_operation!(<T> OneOperation<T> where T: Type);
impl_nullary_batchable_operation!(@replicated OneOperation<ArrayType>);
impl_nullary_batchable_operation!(@member<ArrayIrType, ArrayIrBatching> OneOperation<ArrayType>);

impl_member_operation_for_array_ir_constant_operation!(OneOperation<ArrayType>);
impl_member_interpretable_operation_for_array_ir_constant_operation!(
    OneOperation<ArrayType>,
    One,
    |context, output_type, _operation| context.one(&output_type),
);

// TODO(eaplatanios): Restore the strict `Operation<Type = T>` super-trait bound once the next-generation trait solver
//  stabilizes. The current solver cannot discharge this projection equality at implementation heads whose context type
//  is built from `Self` (E0284). The equality is enforced per method through a `where` clause instead.
/// Supplies the canonical one [`Operation`] of a program type's operation family. [`Self::one_operation`] covers ones
/// that can be constructed from a type without operands, which is all that staging and reverse-mode gradient seeding
/// need.
///
/// The super-trait is plain [`Operation`] rather than `Operation<Type = T>` because the current trait solver cannot
/// discharge that projection equality where this provider is requested through a context's operation family. The
/// equality is instead required by [`one_operation`](Self::one_operation) itself, so a provider whose
/// [`Operation::Type`] disagrees with `T` cannot construct anything.
pub trait OneOperationProvider<T: Type>: Operation {
    /// Constructs an [`Operation`] that materializes a one of `r#type` without operands.
    fn one_operation(r#type: T) -> Result<Self, ProgramError>
    where
        Self: Operation<Type = T>;
}

impl<T: Type, O: Operation<Type = T> + From<OneOperation<T>>> OneOperationProvider<T> for O {
    #[inline]
    fn one_operation(r#type: T) -> Result<Self, ProgramError> {
        Ok(Self::from(OneOperation::new(r#type)))
    }
}

impl<A: Value<Type = ArrayType>> OneOperationProvider<ArrayIrType> for ArrayIrOperation<A> {
    fn one_operation(r#type: ArrayIrType) -> Result<Self, ProgramError> {
        let r#type = match r#type {
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
                Err(TypeError::invalid(format!("data type {} cannot represent one", r#type.data_type())).into())
            }
            data_type => dispatch_on_array_element_type!(data_type, |Element| {
                let element = Element::from_unsigned(1)?;
                Array::from_fn_elements(r#type.clone(), |_| Ok(element))
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

impl<C: StagingContext<Operation: OneOperationProvider<C::Type>>> One<Tracer<C>> for C {
    #[inline]
    fn one(&self, r#type: &C::Type) -> Result<Tracer<C>, ProgramError> {
        let mut outputs = self.stage_nullary_operation(C::Operation::one_operation(r#type.clone())?)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Context> One<PartialTracer<C>> for PartialEvaluationContext<C>
where
    C::Operation: PartiallyEvaluatableOperation<C>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + OneOperationProvider<C::Type>,
{
    #[inline]
    fn one(&self, r#type: &C::Type) -> Result<PartialTracer<C>, ProgramError> {
        let mut outputs = self.bind(C::Operation::one_operation(r#type.clone())?, Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Context<Type = ArrayType> + One<C::Value>> One<BatchingTracer<C, ArrayBatching>>
    for BatchingContext<C, ArrayBatching>
{
    #[inline]
    fn one(&self, r#type: &ArrayType) -> Result<BatchingTracer<C, ArrayBatching>, ProgramError> {
        let batch = ArrayBatch::new(self.parent().one(r#type)?, BatchAxis::replicated())?;
        Ok(BatchingTracer::new(self.clone(), batch))
    }
}

impl<C: Context<Type: DifferentiableType> + One<C::Value>> One<DifferentiationTracer<C>> for DifferentiationContext<C> {
    #[inline]
    fn one(&self, r#type: &C::Type) -> Result<DifferentiationTracer<C>, ProgramError> {
        let dual = DifferentiationDual::new_with_zero_tangent(self.parent().one(r#type)?)?;
        Ok(DifferentiationTracer::new(dual, self.clone()))
    }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayBatch, ArrayBatching, ArrayIrOperation, ArrayIrValue, ArrayOperation, DataType, Dimension,
        DimensionBounds, DimensionType, DimensionVariable, Shape,
    };
    use crate::batching::{BatchAxis, BatchableOperation, BatchingContext};
    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::operations::constants::constant::ConstantOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{EmptyRegionDriver, MaybeZero, Operation, ProgramBuilder, ReferenceType};

    use super::*;

    #[test]
    fn test_one() {
        // Verify the operation's stored type, identity, rendering, and eager interpretation.
        let operation = OneOperation::new(ArrayType::scalar(DataType::F64));
        assert_eq!(operation.name(), ONE_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "one [type=f64[]]");
        assert_eq!(operation.r#type(), &ArrayType::scalar(DataType::F64));
        assert_eq!(operation.infer_output_types(&[], &[]), Ok(vec![ArrayType::scalar(DataType::F64)]));
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Ok(vec![Array::scalar(1.0)]),
        );

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
    fn test_one_operation_provider() {
        // Homogeneous operation families receive the infallible provider implementation through their ordinary
        // `From<OneOperation<T>>` conversion.
        let static_type = ArrayType::new_static(DataType::F32, [2]);
        let ArrayOperation::<Array>::One(operation) = ArrayOperation::one_operation(static_type.clone()).unwrap()
        else {
            panic!("expected a homogeneous one operation");
        };
        assert_eq!(operation.r#type(), &static_type);

        // The composite provider projects a valid operand-free array one into the homogeneous member family.
        let ArrayIrOperation::<Array>::Array(ArrayOperation::One(operation)) =
            ArrayIrOperation::one_operation(ArrayIrType::Array(static_type.clone())).unwrap()
        else {
            panic!("expected a composite homogeneous one operation");
        };
        assert_eq!(operation.r#type(), &static_type);

        // Operand-free construction cannot resolve a dynamic identity. Dynamic mixed ones must instead receive their
        // concrete extents as dimension operands.
        let size = DimensionVariable::new("size", DimensionBounds::unbounded());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(size.clone())]));
        assert_eq!(
            ArrayIrOperation::<Array>::one_operation(ArrayIrType::Array(dynamic_type)).unwrap_err(),
            ProgramError::Type(TypeError::invalid(
                "`one` cannot construct type f32[size] without operands because it references identity size",
            )),
        );

        // First-class dimensions and references are not algebraic values. In particular, a reference cannot be replaced
        // by a one of its referent type. The differentiation rules allocate tangent and cotangent references instead of
        // ever materializing a one reference.
        assert_eq!(
            ArrayIrOperation::<Array>::one_operation(ArrayIrType::Dimension(DimensionType::new(size))).unwrap_err(),
            ProgramError::Type(TypeError::invalid("cannot materialize a one for a first-class dimension type")),
        );
        let reference_type = ReferenceType::new(static_type);
        assert_eq!(
            ArrayIrOperation::<Array>::one_operation(ArrayIrType::Reference(reference_type.clone())).unwrap_err(),
            ProgramError::Type(TypeError::invalid(format!(
                "cannot materialize a one for reference type `{reference_type}`; a reference denotes an allocation \
                 and has no one value, so tangent and cotangent references are allocated by the differentiation rules",
            ))),
        );
    }

    #[test]
    fn test_eager_context_one() {
        let context = EagerContext::<Array>::new();

        // Verify canonical rank-zero one values across every supported data-type family.
        for (r#type, expected) in [
            (DataType::Boolean, Array::scalar(true)),
            (DataType::I8, Array::scalar(1i8)),
            (DataType::I16, Array::scalar(1i16)),
            (DataType::I32, Array::scalar(1i32)),
            (DataType::I64, Array::scalar(1i64)),
            (DataType::U8, Array::scalar(1u8)),
            (DataType::U16, Array::scalar(1u16)),
            (DataType::U32, Array::scalar(1u32)),
            (DataType::U64, Array::scalar(1u64)),
            (DataType::BF16, Array::scalar(bf16::ONE)),
            (DataType::F16, Array::scalar(f16::ONE)),
            (DataType::F32, Array::scalar(1.0f32)),
            (DataType::F64, Array::scalar(1.0f64)),
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
                Err(ProgramError::Type(TypeError::invalid(format!("data type {data_type} cannot represent one",)))),
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
    fn test_partial_evaluation_context_one() {
        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let output_type = ArrayType::new_static(DataType::F32, [2]);
        let output = context.one(&output_type).unwrap();
        let expected = Array::from_elements(output_type, &[1.0f32; 2]).unwrap();
        assert_eq!(output.value().unwrap().as_known(), Some(&expected));
    }

    #[test]
    fn test_batching_context_one() {
        let context = BatchingContext::<_, ArrayBatching>::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 4);
        let output_type = ArrayType::new_static(DataType::F32, [2]);
        let output = context.one(&output_type).unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(output.batch().value(), &Array::from_elements(output_type, &[1.0f32; 2]).unwrap());
    }

    #[test]
    fn test_differentiation_context_one() {
        let context = DifferentiationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let output_type = ArrayType::new_static(DataType::F32, [2]);
        let output = context.one(&output_type).unwrap();
        assert_eq!(output.primal(), &Array::from_elements(output_type.clone(), &[1.0f32; 2]).unwrap());
        assert!(matches!(output.tangent(), MaybeZero::Zero(r#type) if r#type == &output_type));
    }
}
