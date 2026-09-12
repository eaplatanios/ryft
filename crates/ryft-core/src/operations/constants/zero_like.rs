use std::fmt::Display;
use std::marker::PhantomData;

use crate::arrays::{
    Array, ArrayElement, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType,
    dispatch_on_array_element_type,
};
use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_elementwise_operation};
use crate::operations::ElementwiseOperation;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{Operation, ProgramError, RegionInterface, Type, TypeError, Typed, Value, ValueProjection};

/// Canonical operation name for [`ZeroLikeOperation`].
pub const ZERO_LIKE_OPERATION_NAME: &str = "zero_like";

/// [`Operation`] that has one exemplar input and that produces a single output that corresponds to the _zero_ value
/// with the same [`Type`] as that input.
#[derive(Clone, Debug, Default)]
pub struct ZeroLikeOperation<T: Type>(PhantomData<fn() -> T>);

impl<T: Type> Copy for ZeroLikeOperation<T> {}

impl<T: Type> ZeroLikeOperation<T> {
    /// Constructs a new [`ZeroLikeOperation`].
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Type> Display for ZeroLikeOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(ZERO_LIKE_OPERATION_NAME)
    }
}

impl<T: Type> Operation for ZeroLikeOperation<T> {
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        ZERO_LIKE_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[T],
        _region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        input_types[0].validate_zero()?;
        Ok(vec![input_types[0].clone()])
    }

    #[inline]
    fn is_zero(&self, output_index: usize) -> bool {
        output_index == 0
    }
}

impl ElementwiseOperation for ZeroLikeOperation<ArrayType> {
    #[inline]
    fn input_count(&self) -> usize {
        1
    }
}

impl<C: Domain<Value: ZeroLike>> InterpretableOperation<C> for ZeroLikeOperation<C::Type> {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].zero_like()?])
    }
}

impl<C: Context<Operation: From<ZeroLikeOperation<C::Type>>>> PartiallyEvaluatableOperation<C>
    for ZeroLikeOperation<C::Type>
{
}

impl_differentiable_elementwise_operation!(@constant<T> ZeroLikeOperation<T>);

impl<A: Value<Type = ArrayType>> From<ZeroLikeOperation<ArrayIrType>> for ArrayIrOperation<A> {
    #[inline]
    fn from(_: ZeroLikeOperation<ArrayIrType>) -> Self {
        // A `zero_like` reads its complete output type, including every runtime extent, from its exemplar input, so
        // the composite family needs no mixed encoding for it: the homogeneous member constructor already expresses
        // the dynamic case. This conversion exists so that type-generic transform drivers can name the exemplar-based
        // zero in the composite universe with a plain `From<ZeroLikeOperation<C::Type>>` bound. A first-class dimension
        // exemplar is rejected by member type inference, which is correct because a dimension has no zero.
        Self::Array(ArrayOperation::ZeroLike(ZeroLikeOperation::new()))
    }
}

/// Represents the ability to construct a _zero_ value with the same type as an exemplar. [`ZeroLike`] is the
/// value-driven counterpart to [`Zero`](super::Zero) and supplies [`ZeroLikeOperation`]'s interpretation capability.
/// For arrays, the result preserves the exemplar's element data type, runtime shape, memory placement, layout, and
/// sharding. Its elements are zeros regardless of the exemplar's values; complex elements have a zero imaginary part.
/// Empty arrays remain empty, but their element data type must still support zero.
///
/// A staged call retains the exemplar as an input so that dynamic extents are read at execution time. It does not
/// replace dynamic dimensions with their allocation bounds. The exemplar's numerical values have no effect on the
/// result, so differentiation returns zero for its tangent or cotangent.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{Array, ZeroLike};
/// let input = Array::vector(vec![2.0f32, -3.0]).unwrap();
/// assert_eq!(input.zero_like(), Ok(Array::vector(vec![0.0f32, 0.0]).unwrap()));
/// ```
pub trait ZeroLike: Sized {
    /// Returns a _zero_ value with the same type and runtime shape as `self`. Returns an error if its element
    /// data type cannot represent zero, including when the exemplar is empty.
    fn zero_like(&self) -> Result<Self, ProgramError>;
}

impl ZeroLike for Array {
    fn zero_like(&self) -> Result<Self, ProgramError> {
        let data_type = self.r#type().data_type();
        match data_type {
            DataType::Token => Err(TypeError::invalid(format!("data type `{data_type}` cannot represent zero")).into()),
            DataType::Zero => Ok(self.clone()),
            data_type => dispatch_on_array_element_type!(data_type, |Element| {
                Self::from_fn_elements(self.r#type().into_owned(), |_| Ok(Element::zero()?))
            }),
        }
    }
}

impl<A: Value<Type = ArrayType> + ZeroLike> ZeroLike for ArrayIrValue<A> {
    #[inline]
    fn zero_like(&self) -> Result<Self, ProgramError> {
        let input = <Self as ValueProjection<ArrayType>>::projected(self)?;
        Ok(Self::Array(input.zero_like()?))
    }
}

impl<V: Value<DispatchDomain: Context<Operation: From<ZeroLikeOperation<V::Type>>>>> ZeroLike for V {
    #[inline]
    fn zero_like(&self) -> Result<Self, ProgramError> {
        let operation = ZeroLikeOperation::new();
        let mut outputs = self.dispatch_domain().bind(operation, Vec::new(), std::slice::from_ref(self))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayBatch, ArrayBatchingPolicy, ArrayOperation, ArrayType, DataType, Dimension, DimensionBounds,
        DimensionValue, DimensionVariable, Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType, Shape, Sharding,
        StridedLayout, f8e8m0fnu,
    };
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::differentiate_at;
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{check_operation_transposition, check_operation_type_inference};
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationValue, PartialTracer};
    use crate::programs::{EmptyRegionDriver, Operation, ProgramBuilder};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_zero_like() {
        // Verify the operation's identity, zero metadata, and rendering.
        let operation = ZeroLikeOperation::<ArrayType>::new();
        assert!(operation.is_zero(0));
        assert!(!operation.is_zero(1));
        assert_eq!(format!("{operation}"), ZERO_LIKE_OPERATION_NAME);

        // Verify the operation's textual form when it appears in a program.
        let mut builder = ProgramBuilder::<Array, ZeroLikeOperation<ArrayType>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                    lambda %0:f64[] .
                    let %1:f64[] = zero_like %0
                    in (%1)
                "}
            .trim_end(),
        );
    }

    #[test]
    fn test_zero_like_type_inference() {
        check_operation_type_inference!(
            operation = ZeroLikeOperation::<ArrayType>::new(),
            cases = [{
                input_types = [ArrayType::new_static(DataType::F32, [2])],
                output_types = [ArrayType::new_static(DataType::F32, [2])],
            }],
        );
    }

    #[test]
    fn test_zero_like_interpretation() {
        let operation = ZeroLikeOperation::<ArrayType>::new();
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::scalar(2.5).unwrap()],
            ),
            Ok(vec![Array::scalar(0.0).unwrap()]),
        );
        let input =
            Array::from_elements(ArrayType::scalar(DataType::F8E8M0FNU), &[f8e8m0fnu::from_bits(0x7f)]).unwrap();
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[input],
            ),
            Err(ProgramError::Type(TypeError::invalid("data type `f8e8m0fnu` cannot represent zero"))),
        );

        // Verify value-driven zero synthesis across representative rank-zero array data-type families.
        for (input, expected) in [
            (Array::scalar(false).unwrap(), Array::scalar(false).unwrap()),
            (Array::scalar(5i32).unwrap(), Array::scalar(0i32).unwrap()),
            (Array::scalar(5u32).unwrap(), Array::scalar(0u32).unwrap()),
            (Array::scalar(bf16::from_f32(5.0)).unwrap(), Array::scalar(bf16::ZERO).unwrap()),
            (Array::scalar(f16::from_f32(5.0)).unwrap(), Array::scalar(f16::ZERO).unwrap()),
            (Array::scalar(3.0f32).unwrap(), Array::scalar(0.0f32).unwrap()),
            (Array::scalar(7.0f64).unwrap(), Array::scalar(0.0f64).unwrap()),
        ] {
            assert_eq!(input.zero_like(), Ok(expected));
        }

        let input = Array::vector(vec![1.5f32, -2.5]).unwrap();
        let output = input.zero_like().unwrap();
        assert_eq!(output.elements::<f32>(), Ok(vec![0.0, 0.0]));
        assert_eq!(output.r#type().into_owned(), ArrayType::new_static(DataType::F32, [2]));

        // Complex identity values preserve physical layout and memory, including for empty arrays.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let r#type = ArrayType::new_static(DataType::C64, [2])
            .with_layout(Layout::Strided(StridedLayout::new(vec![16])))
            .with_memory(Memory::Host { pinned: true })
            .with_sharding(Sharding::replicated(mesh, 1))
            .unwrap();
        let input = Array::from_elements(r#type.clone(), &[Complex::new(2.0f32, 3.0); 2]).unwrap();
        assert_eq!(input.zero_like(), Ok(Array::from_elements(r#type, &[Complex::new(0.0f32, 0.0); 2]).unwrap()),);
        let empty = Array::from_elements::<Complex<f32>>(ArrayType::new_static(DataType::C64, [0]), &[]).unwrap();
        assert_eq!(empty.zero_like(), Ok(empty.clone()));

        // Unsupported data types report exact errors instead of fabricating a zero from the exemplar.
        let input = Array::from_elements(
            ArrayType::new_static(DataType::F8E8M0FNU, [2]),
            &[f8e8m0fnu::from_bits(0x7e), f8e8m0fnu::from_bits(0x80)],
        )
        .unwrap();
        assert_eq!(
            input.zero_like(),
            Err(ProgramError::Type(TypeError::invalid("data type `f8e8m0fnu` cannot represent zero"))),
        );
        let token = Array::new(ArrayType::scalar(DataType::Token), Vec::new()).unwrap();
        assert_eq!(
            token.zero_like(),
            Err(ProgramError::Type(TypeError::invalid("data type `token` cannot represent zero"))),
        );
        let zero = Array::new(ArrayType::new_static(DataType::Zero, [2]), Vec::new()).unwrap();
        assert_eq!(zero.zero_like(), Ok(zero.clone()));
    }

    #[test]
    fn test_zero_like_interpretation_mixed() {
        let input = ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::F32, [2]), &[2.0f32, 3.0]).unwrap(),
        );
        let expected = ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::F32, [2]), &[0.0f32, 0.0]).unwrap(),
        );
        assert_eq!(input.zero_like(), Ok(expected));
        let dimension = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(2).unwrap());
        assert_eq!(
            dimension.zero_like(),
            Err(ProgramError::Type(TypeError::invalid("expected array type but got dimension type"))),
        );
    }

    #[test]
    fn test_zero_like_partial_evaluation() {
        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let input =
            PartialTracer::new(context, PartialEvaluationValue::known(Array::vector(vec![1.5f32, -2.5]).unwrap()));
        let output = input.zero_like().unwrap();
        assert_eq!(output.value().unwrap().as_known(), Some(&Array::vector(vec![0.0f32, 0.0]).unwrap()));
    }

    #[test]
    fn test_zero_like_batching() {
        let context =
            BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 4);
        let input = BatchingTracer::new(
            context,
            ArrayBatch::new(Array::vector(vec![1.5f32, -2.5]).unwrap(), BatchAxis::replicated()).unwrap(),
        );
        let output = input.zero_like().unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(output.batch().value(), &Array::vector(vec![0.0f32, 0.0]).unwrap());

        // A nonleading mapped axis remains mapped rather than turning the result into a replicated constant.
        let context =
            BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 3);
        let r#type = ArrayType::new_static(DataType::C64, [2, 3]);
        let input = BatchingTracer::new(
            context,
            ArrayBatch::new(
                Array::from_elements(r#type.clone(), &[Complex::new(2.0f32, 3.0); 6]).unwrap(),
                BatchAxis::new(1),
            )
            .unwrap(),
        );
        let output = input.zero_like().unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::new(1));
        assert_eq!(output.batch().value(), &Array::from_elements(r#type, &[Complex::new(0.0f32, 0.0); 6]).unwrap(),);
    }

    #[test]
    fn test_zero_like_differentiation() {
        // Dense reverse-mode differentiation batches the constant rule while constructing the identity Jacobian.
        let jacobian = differentiate_at(Array::scalar(2.0).unwrap())
            .jacobian_reverse(|input| Ok(input.clone() + input.zero_like()?))
            .unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.value().to_f64s(), vec![1.0]);
    }

    #[test]
    fn test_zero_like_transposition() {
        check_operation_transposition!(
            @exact,
            operation = ZeroLikeOperation::<ArrayType>::new(),
            cases = [{
                inputs = [(@linear(type = ArrayType::scalar(DataType::F64)))],
                output_cotangents = [Array::scalar(3.0).unwrap()],
                input_cotangents = [Array::scalar(0.0).unwrap()],
            }],
        );
    }

    #[test]
    fn test_zero_like_staging() {
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = context.input(ArrayType::new_static(DataType::F32, [2]));
        let output = input.zero_like().unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![output.atom_id().unwrap()], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = zero_like %0
                in (%1)
            "}
            .trim_end(),
        );
    }
    #[test]
    fn test_zero_like_staging_mixed() {
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let size = DimensionVariable::new("size", DimensionBounds::non_negative(Some(8)).unwrap());
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(size)]));
        let input = context.input(r#type.clone().into());
        let output = input.zero_like().unwrap();
        assert_eq!(output.r#type().as_ref(), &ArrayIrType::Array(r#type));
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
            panic!("expected one exemplar-based zero instruction");
        };
        assert!(matches!(instruction.operation(), ArrayIrOperation::Array(ArrayOperation::ZeroLike(_))));
        assert_eq!(instruction.inputs(), &[input.atom_id().unwrap()]);
        assert_eq!(
            program.interpret(vec![ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::F32, [2]), &[2.0f32, 3.0]).unwrap(),
            )]),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::F32, [2]), &[0.0f32, 0.0]).unwrap(),
            )]),
        );
        // Runtime zero is a valid extent even though the exemplar's signature is dynamic.
        assert_eq!(
            program.interpret(vec![ArrayIrValue::Array(
                Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0]), &[]).unwrap(),
            )]),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0]), &[]).unwrap(),
            )]),
        );
    }
}
