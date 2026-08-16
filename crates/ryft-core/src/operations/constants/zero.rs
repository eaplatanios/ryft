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

/// Canonical operation name for [`ZeroOperation`].
pub const ZERO_OPERATION_NAME: &str = "zero";

/// [`Operation`] that has no inputs and that produces a single output that corresponds to the _zero_ value for the
/// [`Type`] that it holds (i.e., for its `r#type` field). For arrays, this would typically correspond to an array of
/// the right type and shape filled with zeros.
#[derive(Clone, Debug)]
pub struct ZeroOperation<T: Type> {
    /// [`Type`] of the value produced when this operation is interpreted.
    r#type: T,
}

impl<T: Type> ZeroOperation<T> {
    /// Creates a new [`ZeroOperation`].
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

impl<T: Type> Display for ZeroOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type> Operation for ZeroOperation<T> {
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        ZERO_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[T],
        _region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 0, TypeError);
        check_constructor_type_has_no_identity_references(ZERO_OPERATION_NAME, &self.r#type)?;
        Ok(vec![self.r#type.clone()])
    }

    #[inline]
    fn is_zero(&self, output_index: usize) -> bool {
        output_index == 0
    }

    #[inline]
    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<T::Identity>) -> Result<Self, TypeError> {
        Ok(Self { r#type: self.r#type.rename_identities(renaming)? })
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, ZERO_OPERATION_NAME)?
            .bracketed(|operation| operation.field("type", &self.r#type))
    }
}

impl<T: Type, C: Domain<Type = T> + Zero<C::Value>> InterpretableOperation<C> for ZeroOperation<T> {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 0, ProgramError);
        Ok(vec![context.zero(&self.r#type)?])
    }
}

impl<T: Type, C: Context<Type = T, Operation: From<ZeroOperation<T>>>> PartiallyEvaluatableOperation<C>
    for ZeroOperation<T>
{
}

impl_non_differentiable_operation!(<T> ZeroOperation<T> where T: Type);
impl_nullary_transposable_operation!(<T> ZeroOperation<T> where T: Type);
impl_nullary_batchable_operation!(@replicated ZeroOperation<ArrayType>);
impl_nullary_batchable_operation!(@member<ArrayIrType, ArrayIrBatching> ZeroOperation<ArrayType>);

impl_member_operation_for_array_ir_constant_operation!(ZeroOperation<ArrayType>);
impl_member_interpretable_operation_for_array_ir_constant_operation!(
    ZeroOperation<ArrayType>,
    Zero,
    |context, output_type, _operation| context.zero(&output_type),
);

// TODO(eaplatanios): Restore the strict `Operation<Type = T>` super-trait bound once the next-generation trait solver
//  stabilizes. The current solver cannot discharge this projection equality at implementation heads whose context type
//  is built from `Self` (E0284). The equality is enforced per method through a `where` clause instead.
/// Supplies the canonical zero [`Operation`] of a program type's operation family. [`Self::zero_operation`] covers
/// zeros that can be constructed from a type without operands, which is all that staging and eager materialization
/// need. Differentiation additionally must materialize zeros whose runtime geometry is unavailable from the type alone
/// (e.g., disconnected cotangents with dynamic axes). That residual protocol is transform-owned and lives on
/// [`ResidualZeroProvider`](crate::ResidualZeroProvider).
///
/// The super-trait is plain [`Operation`] rather than `Operation<Type = T>` because the current trait solver cannot
/// discharge that projection equality where this provider is requested through a context's operation family, which is
/// how every transform requests it. The equality is instead required by [`zero_operation`](Self::zero_operation)
/// itself, so a provider whose [`Operation::Type`] disagrees with `T` cannot construct anything: the requirement is
/// restated by the residual-zero protocol and by transform call sites, and any mismatched implementation is rejected
/// with a type-mismatch error there.
pub trait ZeroOperationProvider<T: Type>: Operation {
    /// Constructs an [`Operation`] that materializes a zero of `r#type` without operands.
    fn zero_operation(r#type: T) -> Result<Self, ProgramError>
    where
        Self: Operation<Type = T>;
}

impl<T: Type, O: Operation<Type = T> + From<ZeroOperation<T>>> ZeroOperationProvider<T> for O {
    #[inline]
    fn zero_operation(r#type: T) -> Result<Self, ProgramError> {
        Ok(Self::from(ZeroOperation::new(r#type)))
    }
}

impl<A: Value<Type = ArrayType>> ZeroOperationProvider<ArrayIrType> for ArrayIrOperation<A> {
    fn zero_operation(r#type: ArrayIrType) -> Result<Self, ProgramError> {
        let ArrayIrType::Array(r#type) = r#type else {
            // A first-class dimension is a symbolic runtime extent rather than an algebraic value. A zero dimension may
            // violate the type's bounds, and also assigning zero would bind its identity to an extent that may disagree
            // with the runtime definition. Dimension tangents and cotangents use the separate array `DataType::Zero`
            // representation.
            return Err(TypeError::invalid("cannot materialize a zero for a first-class dimension type").into());
        };
        check_constructor_type_has_no_identity_references(ZERO_OPERATION_NAME, &r#type)?;
        Ok(Self::Array(ArrayOperation::Zero(ZeroOperation::new(r#type))))
    }
}

/// Represents the ability to synthesize a _zero_ value for a given [`Type`] in an interpretation context. [`Zero`]
/// is the [`Type`]-driven counterpart to [`ZeroLike`](super::ZeroLike). It is what [`ZeroOperation`] needs for its
/// [`InterpretableOperation`] implementation, and it lives on the context because producing an eager value can be
/// backend- or context-dependent.
pub trait Zero<V: Typed> {
    /// Returns a _zero_ value for the provided [`Type`].
    fn zero(&self, r#type: &V::Type) -> Result<V, ProgramError>;
}

impl<O: Operation<Type = ArrayType>> Zero<Array> for EagerContext<Array, O> {
    fn zero(&self, r#type: &ArrayType) -> Result<Array, ProgramError> {
        match r#type.data_type() {
            DataType::Token => {
                Err(TypeError::invalid(format!("data type `{}` cannot represent zero", DataType::Token)).into())
            }
            DataType::Zero => Array::new(r#type.clone(), Vec::new()),
            data_type => dispatch_on_array_element_type!(data_type, |Element| {
                let element = Element::from_unsigned(0)?;
                Array::from_fn_elements(r#type.clone(), |_| Ok(element))
            }),
        }
    }
}

impl<V: Value<Type = ArrayType>, O: Operation<Type = ArrayIrType>> Zero<ArrayIrValue<V>>
    for EagerContext<ArrayIrValue<V>, O>
where
    EagerContext<V, ArrayOperation<V>>: Zero<V>,
{
    #[inline]
    fn zero(&self, r#type: &ArrayIrType) -> Result<ArrayIrValue<V>, ProgramError> {
        let r#type = <&ArrayType>::try_from(r#type)?;
        Ok(ArrayIrValue::Array(EagerContext::<V, ArrayOperation<V>>::new().zero(r#type)?))
    }
}

impl<C: Context, T: Type> Zero<<C::Value as ValueProjection<T>>::Projected> for ProjectedContext<C, T>
where
    C::Value: ValueProjection<T, Projected: Value<Type = T>>,
    C::Constant: ValueProjection<T, Projected: Value<Type = T>>,
    C::Operation: OperationProjection<T, Projected: From<ZeroOperation<T>>>,
{
    #[inline]
    fn zero(&self, r#type: &T) -> Result<<C::Value as ValueProjection<T>>::Projected, ProgramError> {
        Ok(self.bind(ZeroOperation::new(r#type.clone()), Vec::new(), &[])?.remove(0))
    }
}

impl<C: StagingContext<Operation: ZeroOperationProvider<C::Type>>> Zero<Tracer<C>> for C {
    #[inline]
    fn zero(&self, r#type: &C::Type) -> Result<Tracer<C>, ProgramError> {
        let mut outputs = self.stage_nullary_operation(C::Operation::zero_operation(r#type.clone())?)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Context> Zero<PartialTracer<C>> for PartialEvaluationContext<C>
where
    C::Operation: PartiallyEvaluatableOperation<C>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + ZeroOperationProvider<C::Type>,
{
    #[inline]
    fn zero(&self, r#type: &C::Type) -> Result<PartialTracer<C>, ProgramError> {
        let mut outputs = self.bind(C::Operation::zero_operation(r#type.clone())?, Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Context<Type = ArrayType> + Zero<C::Value>> Zero<BatchingTracer<C, ArrayBatching>>
    for BatchingContext<C, ArrayBatching>
{
    #[inline]
    fn zero(&self, r#type: &ArrayType) -> Result<BatchingTracer<C, ArrayBatching>, ProgramError> {
        let batch = ArrayBatch::new(self.parent().zero(r#type)?, BatchAxis::replicated())?;
        Ok(BatchingTracer::new(self.clone(), batch))
    }
}

impl<C: Context<Type: DifferentiableType> + Zero<C::Value>> Zero<DifferentiationTracer<C>>
    for DifferentiationContext<C>
{
    #[inline]
    fn zero(&self, r#type: &C::Type) -> Result<DifferentiationTracer<C>, ProgramError> {
        let dual = DifferentiationDual::new_with_zero_tangent(self.parent().zero(r#type)?)?;
        Ok(DifferentiationTracer::new(dual, self.clone()))
    }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayBatch, ArrayBatching, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, DataType,
        Dimension, DimensionBounds, DimensionType, DimensionVariable, Shape,
    };
    use crate::batching::{BatchAxis, BatchableOperation, BatchingContext};
    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::operations::constants::constant::ConstantOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{EmptyRegionDriver, MaybeZero, Operation, ProgramBuilder};

    use super::*;

    #[test]
    fn test_zero() {
        // Verify the operation's stored type, identity, zero metadata, rendering, and eager interpretation.
        let operation = ZeroOperation::new(ArrayType::scalar(DataType::F64));
        assert_eq!(operation.name(), ZERO_OPERATION_NAME);
        assert!(operation.is_zero(0));
        assert!(!operation.is_zero(1));
        assert_eq!(format!("{operation}"), "zero [type=f64[]]");
        assert_eq!(operation.r#type(), &ArrayType::scalar(DataType::F64));
        assert_eq!(operation.infer_output_types(&[], &[]), Ok(vec![ArrayType::scalar(DataType::F64)]));
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[]
            ),
            Ok(vec![Array::scalar(0.0)]),
        );

        // A nullary zero does not acquire a physical batch axis because the same value serves every batch item.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let outputs: Vec<ArrayBatch<Array>> = ZeroOperation::new(scalar_type.clone())
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
        assert_eq!(outputs[0].value().to_f64s(), vec![0.0]);

        // Nullary construction rejects output types with ungrounded identity _references_ (a dynamic array axis),
        // which must instead be constructed through the mixed dimension-operand contract owned by the composite
        // operation family. Definition-position identities remain constructible: a dimension value's type defines
        // its own variable, so nullary construction leaves no dangling reference.
        let rows = crate::arrays::DimensionVariable::new("rows", DimensionBounds::non_negative(Some(8)).unwrap());
        let dynamic_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Static(3)]));
        assert_eq!(
            ZeroOperation::new(dynamic_type.clone()).infer_output_types(&[], &[]),
            Err(TypeError::invalid(
                "`zero` cannot construct type f32[rows, 3] without operands because it references identity rows",
            )),
        );
        let dimension_type = DimensionType::new(rows);
        assert_eq!(ZeroOperation::new(dimension_type.clone()).infer_output_types(&[], &[]), Ok(vec![dimension_type]),);

        // Verify the operation's textual form when it appears in a program.
        let mut builder = ProgramBuilder::<Array, ZeroOperation<ArrayType>>::new();
        let output = builder.add_instruction(operation, Vec::new(), vec![]).unwrap()[0];
        let program = builder.build::<(), Array>(vec![output], (), Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f64[] = zero [type=f64[]]
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_eager_context_zero() {
        let context = EagerContext::<Array>::new();

        // Verify canonical rank-zero zero values across every supported data-type family.
        for (r#type, expected) in [
            (DataType::Boolean, Array::scalar(false)),
            (DataType::I8, Array::scalar(0i8)),
            (DataType::I16, Array::scalar(0i16)),
            (DataType::I32, Array::scalar(0i32)),
            (DataType::I64, Array::scalar(0i64)),
            (DataType::U8, Array::scalar(0u8)),
            (DataType::U16, Array::scalar(0u16)),
            (DataType::U32, Array::scalar(0u32)),
            (DataType::U64, Array::scalar(0u64)),
            (DataType::BF16, Array::scalar(bf16::ZERO)),
            (DataType::F16, Array::scalar(f16::ZERO)),
            (DataType::F32, Array::scalar(0.0f32)),
            (DataType::F64, Array::scalar(0.0f64)),
        ] {
            assert_eq!(context.zero(&ArrayType::scalar(r#type)), Ok(expected));
        }

        // Rank-positive arrays and the zero-space data type preserve the requested geometry.
        let output_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let expected = Array::from_elements(output_type.clone(), &[0.0f32; 6]).unwrap();
        assert_eq!(context.zero(&output_type), Ok(expected.clone()));
        let zero_space_type = ArrayType::new_static(DataType::Zero, [2, 3]);
        assert_eq!(context.zero(&zero_space_type), Array::new(zero_space_type, Vec::new()).map_err(Into::into));

        // Token arrays and dynamically shaped eager arrays cannot be materialized as zeros.
        assert_eq!(
            context.zero(&ArrayType::scalar(DataType::Token)),
            Err(ProgramError::Type(TypeError::invalid("data type `token` cannot represent zero"))),
        );
        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("size", DimensionBounds::unbounded()))]),
        );
        assert!(matches!(
            context.zero(&dynamic_type),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot materialize a value of dynamically sized type f32[size]; dynamically shaped \
                               values exist only in array programs over `ArrayIrOperation`",
        ));

        // Composite eager zero materialization delegates array members and rejects first-class dimensions.
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        assert_eq!(context.zero(&ArrayIrType::Array(output_type)), Ok(ArrayIrValue::Array(expected)));
        let dimension_type =
            ArrayIrType::Dimension(DimensionType::new(DimensionVariable::new("size", DimensionBounds::unbounded())));
        assert_eq!(
            context.zero(&dimension_type),
            Err(ProgramError::Type(TypeError::invalid("expected array type but got dimension type"))),
        );
    }

    #[test]
    fn test_projected_context_zero() {
        let parent = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let context = ProjectedContext::<_, ArrayType>::new(parent.clone());
        let output_type = ArrayType::new_static(DataType::F32, [2]);
        let output = context.zero(&output_type).unwrap();
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
                let %0:f32[2] = zero [type=f32[2]]
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_staging_context_zero() {
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_type = ArrayType::new_static(DataType::F32, [2]);
        let output = context.zero(&output_type).unwrap();
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
                let %0:f32[2] = zero [type=f32[2]]
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_partial_evaluation_context_zero() {
        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let output_type = ArrayType::new_static(DataType::F32, [2]);
        let output = context.zero(&output_type).unwrap();
        let expected = Array::from_elements(output_type, &[0.0f32; 2]).unwrap();
        assert_eq!(output.value().unwrap().as_known(), Some(&expected));
    }

    #[test]
    fn test_batching_context_zero() {
        let context = BatchingContext::<_, ArrayBatching>::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 4);
        let output_type = ArrayType::new_static(DataType::F32, [2]);
        let output = context.zero(&output_type).unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(output.batch().value(), &Array::from_elements(output_type, &[0.0f32; 2]).unwrap());
    }

    #[test]
    fn test_differentiation_context_zero() {
        let context = DifferentiationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let output_type = ArrayType::new_static(DataType::F32, [2]);
        let output = context.zero(&output_type).unwrap();
        assert_eq!(output.primal(), &Array::from_elements(output_type.clone(), &[0.0f32; 2]).unwrap());
        assert!(matches!(output.tangent(), MaybeZero::Zero(r#type) if r#type == &output_type));
    }
}
