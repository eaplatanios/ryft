use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use ryft_macros::capability;

use crate::arrays::{
    Array, ArrayElement, ArrayIrOperation, ArrayIrType, ArrayOperation, ArrayType, DataType,
    dispatch_on_array_element_type,
};
use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_elementwise_operation};
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::constants::fill::{FillLiteral, fill_literal_array};
use crate::operations::manipulation::broadcasting::Broadcast;
use crate::operations::manipulation::conversions::ConvertElementType;
use crate::operations::{Capability, ElementwiseOperation};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    LiteralIdentity, Operation, OperationFormatter, ProgramError, RegionInterface, Type, TypeError, Typed, Value,
    ValueDomainDispatch,
};

/// Canonical operation name for [`FillLikeOperation`].
pub const FILL_LIKE_OPERATION_NAME: &str = "fill_like";

/// [`Operation`] that has one exemplar input and that produces a single output with the [`Type`] of that input whose
/// elements all hold the value that this operation stores. It is the counterpart of [`Fill`](super::Fill) for outputs
/// whose shape is only known when the program runs. The exemplar stays an input, so the extents of its dynamic
/// dimensions are read at execution time, while its element values never affect the result. Refer to [`FillLike`]
/// for the value-level contract.
#[derive(Clone, Debug)]
pub struct FillLikeOperation<T: Type> {
    /// Rank-0 host [`Array`] that holds the fill value at the element data type of the exemplar.
    value: Array,

    /// [`PhantomData`] marker tying this [`Operation`] to the [`Type`] universe in which it is valid.
    marker: PhantomData<fn() -> T>,
}

impl<T: Type> FillLikeOperation<T> {
    /// Creates a new [`FillLikeOperation`] that fills its output with `value`, which must already have the element
    /// data type of the exemplars that this operation is applied to.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] when `value` is not a rank-0 array or when its data type has no elements (i.e., it is
    /// [`DataType::Token`] or [`DataType::Zero`]).
    pub fn new(value: Array) -> Result<Self, TypeError> {
        let value_type = value.r#type();
        if value_type.rank() != 0 {
            return Err(TypeError::invalid(format!(
                "`{FILL_LIKE_OPERATION_NAME}` requires a rank-0 value but got a value of type `{value_type}`",
            )));
        }
        if matches!(value_type.data_type(), DataType::Token | DataType::Zero) {
            return Err(TypeError::invalid(format!(
                "`{FILL_LIKE_OPERATION_NAME}` cannot fill elements of data type `{}`",
                value_type.data_type(),
            )));
        }
        Ok(Self { value, marker: PhantomData })
    }

    /// Returns the rank-0 [`Array`] that holds the fill value of this [`FillLikeOperation`].
    #[inline]
    pub fn value(&self) -> &Array {
        &self.value
    }
}

impl<T: Type> Display for FillLikeOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        OperationFormatter::new(formatter, 0, FILL_LIKE_OPERATION_NAME)?
            .bracketed(|operation| operation.field("value", &self.value))
    }
}

impl<T: Type> PartialEq for FillLikeOperation<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // Fill values are compared as literals, so that NaN values and signed zeros are identified by their bits.
        self.value.literal_eq(&other.value)
    }
}

impl<T: Type> Eq for FillLikeOperation<T> {}

impl<T: Type> Hash for FillLikeOperation<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.literal_hash(state);
    }
}

impl Operation for FillLikeOperation<ArrayType> {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        FILL_LIKE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let exemplar_type = &input_types[0];
        let value_data_type = self.value.r#type().data_type();
        if value_data_type != exemplar_type.data_type() {
            return Err(TypeError::invalid(format!(
                "`{}` value data type `{}` does not match the exemplar data type `{}`",
                FILL_LIKE_OPERATION_NAME,
                value_data_type,
                exemplar_type.data_type(),
            )));
        }

        // A fill holds the same value on every device, which is a valid value along reduced axes but not a partial
        // sum that still needs a cross-device reduction.
        if !exemplar_type.unreduced_axes().is_empty() {
            return Err(TypeError::invalid(format!(
                "`{FILL_LIKE_OPERATION_NAME}` cannot fill type `{exemplar_type}` because it is unreduced \
                 over mesh axes",
            )));
        }
        Ok(vec![exemplar_type.clone()])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, FILL_LIKE_OPERATION_NAME)?
            .bracketed(|operation| operation.field("value", &self.value))
    }
}

impl ElementwiseOperation for FillLikeOperation<ArrayType> {
    #[inline]
    fn input_count(&self) -> usize {
        1
    }
}

impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for FillLikeOperation<ArrayType>
where
    C::Value: FillLike,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // The capability takes a typed host literal, so the stored literal is read back at its own element type.
        // The constructor rejects the data types that have no elements, which the dispatch does not cover.
        check_count!("input", inputs, 1, ProgramError);
        dispatch_on_array_element_type!(self.value.r#type().data_type(), |Element| {
            let value = self.value.elements::<Element>()?.remove(0);
            Ok(vec![inputs[0].fill_like(value)?])
        })
    }
}

impl<C: Context<Type = ArrayType, Operation: From<FillLikeOperation<ArrayType>>>> PartiallyEvaluatableOperation<C>
    for FillLikeOperation<ArrayType>
{
}

impl_differentiable_elementwise_operation!(@constant<T> FillLikeOperation<T>);

impl<A: Value<Type = ArrayType>> From<FillLikeOperation<ArrayIrType>> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: FillLikeOperation<ArrayIrType>) -> Self {
        // The exemplar supplies the complete output type, including every runtime extent, so the homogeneous member
        // already represents dynamic arrays and the array IR family needs no mixed encoding. Projecting that member
        // also rejects dimension and reference exemplars.
        Self::Array(ArrayOperation::FillLike(FillLikeOperation { value: operation.value, marker: PhantomData }))
    }
}

/// Represents the ability to construct a value with the same type as an exemplar whose elements all hold
/// one typed host literal. [`FillLike`] is the value-driven counterpart to [`Fill`](super::Fill) and supplies
/// [`FillLikeOperation`]'s interpretation capability. For arrays, the result preserves the exemplar's element data
/// type, runtime shape, memory placement, layout, and sharding, and the literal is converted to that element data
/// type. The exemplar's element values have no effect on the result, so differentiation returns zero for its tangent
/// or cotangent. Empty arrays remain empty. A fill holds the same value on every device, which is not a partial sum,
/// so exemplars that are unreduced over a mesh axis are rejected.
///
/// A staged call binds a rank-0 [`ConstantOperation`] and broadcasts it to the exemplar's type when that type names no
/// runtime extents, exactly like [`Fill`](super::Fill), which has no data dependence on the exemplar. Otherwise, a
/// staged call binds [`FillLikeOperation`], which retains the exemplar as an input so that dynamic extents are read
/// at execution time rather than replaced by their allocation bounds. Eager calls always use the exemplar, whose
/// runtime placement may be more specific than its type.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and array IR values implement it for [`ArrayIrType`].
///
/// # Example
///
/// ```rust
/// # use ryft_core::{Array, FillLike};
/// let exemplar = Array::vector(vec![2.0f32, -3.0]).unwrap();
/// assert_eq!(exemplar.fill_like(1.5f64), Ok(Array::vector(vec![1.5f32, 1.5]).unwrap()));
/// ```
#[capability(projection(ArrayIrType => ArrayType))]
pub trait FillLike<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Returns a value with the same type and runtime shape as `self` whose elements all hold `value`, converted to
    /// the element data type of `self`.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when `value` cannot be converted to the element data type of `self`, or when `self`
    /// is unreduced over a mesh axis.
    fn fill_like<L: ArrayElement>(&self, value: L) -> Result<Self, ProgramError>;
}

impl FillLike for Array {
    fn fill_like<L: ArrayElement>(&self, value: L) -> Result<Self, ProgramError> {
        let r#type = self.r#type();
        fill_literal_array(&r#type, value)?.broadcast(r#type.into_owned(), &[])
    }
}

impl<V> FillLike<ArrayType> for V
where
    V: Value<Type = ArrayType, Dispatch = ValueDomainDispatch> + Broadcast,
    V::Domain:
        Context<Type = ArrayType, Operation: From<FillLikeOperation<ArrayType>> + From<ConstantOperation<Array>>>,
{
    fn fill_like<L: ArrayElement>(&self, value: L) -> Result<Self, ProgramError> {
        // A type that names no runtime extents determines the fill completely, so the staged fill is the same
        // constant and broadcast that `Fill` stages, which does not read the exemplar and so lets dead code
        // elimination drop it. Eager contexts keep `fill_like`, because the runtime placement of an eager value
        // (e.g., of an automatically sharded array) may be more specific than its type.
        let domain = self.domain();
        let r#type = self.r#type();
        if !domain.is_eager() && r#type.identities().next().is_none() {
            return domain.fill_literal(&r#type, value);
        }
        let value = Array::scalar(value)?.convert_element_type(r#type.data_type())?;
        let mut outputs = domain.bind(FillLikeOperation::new(value)?, Vec::new(), std::slice::from_ref(self))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrValue, Dimension, DimensionBounds, DimensionValue, DimensionVariable, Layout, LogicalMesh, Memory,
        MeshAxis, MeshAxisType, Shape, Sharding, ShardingDimension, StridedLayout,
    };
    use crate::batching::{BatchAxis, batch};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::differentiate_at;
    use crate::macros::{
        check_operation_batching, check_operation_partial_evaluation, check_operation_transposition,
        check_operation_type_inference,
    };
    use crate::operations::arithmetic::Add;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationOutput, PartialValue};
    use crate::programs::{EmptyRegionDriver, ProgramBuilder};
    use crate::tests::hash_of;
    use crate::tracing::TracingContext;

    use super::*;

    /// Returns a [`FillLikeOperation`] over [`ArrayType`] that fills its output with the `f32` literal `value`.
    fn f32_fill_like(value: f32) -> FillLikeOperation<ArrayType> {
        FillLikeOperation::new(Array::scalar(value).unwrap()).unwrap()
    }

    #[test]
    fn test_fill_like() {
        // Verify the operation's payload, validation, and literal identity, which distinguishes signed zeros and
        // identifies equal NaN payloads.
        let operation = f32_fill_like(2.5);
        assert_eq!(operation.value(), &Array::scalar(2.5f32).unwrap());
        assert_eq!(operation.to_string(), "fill_like [value=2.5]");
        assert_eq!(operation, f32_fill_like(2.5));
        assert_eq!(hash_of(&operation), hash_of(&f32_fill_like(2.5)));
        assert_ne!(operation, f32_fill_like(3.5));
        assert_ne!(f32_fill_like(0.0), f32_fill_like(-0.0));
        assert_eq!(f32_fill_like(f32::NAN), f32_fill_like(f32::NAN));
        assert_eq!(
            FillLikeOperation::<ArrayType>::new(Array::vector(vec![1.0f32, 2.0]).unwrap()).map(|_| ()),
            Err(TypeError::invalid(format!(
                "`{FILL_LIKE_OPERATION_NAME}` requires a rank-0 value but got a value of type `f32[2]`",
            ))),
        );
        assert_eq!(
            FillLikeOperation::<ArrayType>::new(Array::new(ArrayType::scalar(DataType::Zero), Vec::new()).unwrap())
                .map(|_| ()),
            Err(TypeError::invalid(format!("`{FILL_LIKE_OPERATION_NAME}` cannot fill elements of data type `zero`",))),
        );

        // Verify the operation's textual form when it appears in a program.
        let mut builder = ProgramBuilder::<Array, FillLikeOperation<ArrayType>>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F32, [2]));
        let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = fill_like [value=2.5] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_fill_like_type_inference() {
        // The output has the exemplar's type, including its dynamic dimensions, and the stored literal must already
        // have the exemplar's element data type.
        let size = DimensionVariable::new("size", DimensionBounds::non_negative(Some(8)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(size), 3.into()]));
        check_operation_type_inference!(
            operation = f32_fill_like(2.5),
            cases = [
                {
                    input_types = [ArrayType::new_static(DataType::F32, [2])],
                    output_types = [ArrayType::new_static(DataType::F32, [2])],
                },
                {
                    input_types = [dynamic_type.clone()],
                    output_types = [dynamic_type],
                },
                {
                    input_types = [ArrayType::new_static(DataType::F64, [2])],
                    error = format!(
                        "`{FILL_LIKE_OPERATION_NAME}` value data type `f32` does not match the exemplar data type \
                         `f64`",
                    ),
                },
            ],
        );

        // A fill is the same value on every device, so it keeps manual variation and rejects unreduced exemplars.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let unreduced = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh, 1).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        assert_eq!(Operation::infer_output_types(&f32_fill_like(2.5), &[varying.clone()], &[]), Ok(vec![varying]));
        assert_eq!(
            Operation::infer_output_types(&f32_fill_like(2.5), &[unreduced.clone()], &[]),
            Err(TypeError::invalid(format!(
                "`{FILL_LIKE_OPERATION_NAME}` cannot fill type `{unreduced}` because it is unreduced over mesh axes",
            ))),
        );

        // An explicitly sharded exemplar gives a fill with the same sharding.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["y"])]).unwrap())
            .unwrap();
        assert_eq!(Operation::infer_output_types(&f32_fill_like(2.5), &[sharded.clone()], &[]), Ok(vec![sharded]));
    }

    #[test]
    fn test_fill_like_interpretation() {
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &f32_fill_like(2.5),
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[Array::vector(vec![7.0f32, -1.0]).unwrap()],
            ),
            Ok(vec![Array::vector(vec![2.5f32, 2.5]).unwrap()]),
        );

        // The literal is converted to the exemplar's element data type, and the exemplar's values are ignored.
        assert_eq!(
            Array::vector(vec![1i32, 2, 3]).unwrap().fill_like(7.0f64),
            Ok(Array::vector(vec![7i32; 3]).unwrap())
        );
        assert_eq!(Array::scalar(false).unwrap().fill_like(true), Ok(Array::scalar(true).unwrap()));
        let output = Array::vector(vec![1.0f64, 2.0]).unwrap().fill_like(f64::NAN).unwrap();
        assert!(output.elements::<f64>().unwrap().iter().all(|element| element.is_nan()));

        // The result preserves the physical layout, memory, and sharding of the exemplar, and empty arrays stay empty.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2])
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
            .with_memory(Memory::Host { pinned: true })
            .with_sharding(Sharding::replicated(mesh, 1))
            .unwrap();
        let exemplar = Array::from_elements(r#type.clone(), &[2.0f32, 3.0]).unwrap();
        assert_eq!(exemplar.fill_like(-1.0f32), Ok(Array::from_elements(r#type, &[-1.0f32, -1.0]).unwrap()));
        let empty = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0, 3]), &[]).unwrap();
        assert_eq!(empty.fill_like(1.0f32), Ok(empty.clone()));
    }

    #[test]
    fn test_fill_like_interpretation_array_ir() {
        // Array IR values fill their array member, and a first-class dimension is not an array.
        let exemplar = ArrayIrValue::Array(Array::vector(vec![2.0f32, 3.0]).unwrap());
        assert_eq!(exemplar.fill_like(0.5f32), Ok(ArrayIrValue::Array(Array::vector(vec![0.5f32, 0.5]).unwrap())));
        let dimension = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(2).unwrap());
        assert_eq!(
            dimension.fill_like(0.5f32),
            Err(ProgramError::Type(TypeError::invalid("expected array type but got dimension type"))),
        );
    }

    #[test]
    fn test_fill_like_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = f32_fill_like(2.5),
            inputs = [Array::vector(vec![7.0f32, -1.0]).unwrap()],
            expected = Array::vector(vec![2.5f32, 2.5]).unwrap(),
        );

        // The fill of an unknown exemplar stays in the residual program, where it reads the exemplar's extents.
        let size = DimensionVariable::new("size", DimensionBounds::non_negative(Some(8)).unwrap());
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(size)]));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(r#type.clone());
        let output = builder.add_instruction(f32_fill_like(2.5), Vec::new(), vec![input], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let evaluation = program.partially_evaluate(&[PartialValue::Unknown(r#type)]).unwrap();
        assert_eq!(evaluation.outputs(), &[PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[size] .
                let %1:f32[size] = fill_like [value=2.5] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_fill_like_batching() {
        // A replicated exemplar gives a replicated fill, and a mapped exemplar keeps its batch axis.
        check_operation_batching!(
            @exact,
            operation = f32_fill_like(2.5),
            axis_size = 2,
            cases = [
                {
                    inputs = [(@replicated, Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap())],
                    outputs = [(@replicated, Array::vector(vec![2.5f32; 3]).unwrap())],
                },
                {
                    inputs = [(@mapped(axis = 1), Array::matrix(3, 2, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap())],
                    outputs = [(@mapped(axis = 1), Array::matrix(3, 2, vec![2.5f32; 6]).unwrap())],
                },
            ],
        );
    }

    #[test]
    fn test_fill_like_differentiation() {
        // The fill does not depend on the values of its exemplar, so its tangent is zero.
        assert_eq!(
            differentiate_at(Array::vector(vec![2.0f32, 3.0]).unwrap())
                .jvp(Array::vector(vec![5.0f32, 7.0]).unwrap(), |x| Ok(x.add(&x.fill_like(1.5f32)?)?)),
            Ok((Array::vector(vec![3.5f32, 4.5]).unwrap(), Array::vector(vec![5.0f32, 7.0]).unwrap())),
        );
    }

    #[test]
    fn test_fill_like_transposition() {
        check_operation_transposition!(
            @exact,
            operation = FillLikeOperation::<ArrayType>::new(Array::scalar(2.5f64).unwrap()).unwrap(),
            cases = [{
                inputs = [(@linear(type = ArrayType::scalar(DataType::F64)))],
                output_cotangents = [Array::scalar(3.0).unwrap()],
                input_cotangents = [Array::scalar(0.0).unwrap()],
            }],
        );
    }

    #[test]
    fn test_fill_like_staging() {
        // A static exemplar type determines the fill completely, so staging binds a constant and a broadcast and
        // leaves the exemplar unused.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = context.input(ArrayType::new_static(DataType::F32, [2]));
        let output = input.fill_like(2.5f64).unwrap();
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
                let %1:f32[] = constant [value=2.5]
                    %2:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %1
                in (%2)
            "}
            .trim_end(),
        );

        // Batching over a staging context keeps the batch axis of the exemplar mapped, and the staged fill of the
        // batched exemplar, whose type is static too, is again a constant and a broadcast.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = context.input(ArrayType::new_static(DataType::F32, [2, 3]));
        let output =
            batch(|item| Ok(item.fill_like(2.5f64)?), input, BatchAxis::new(1), BatchAxis::new(0), None).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![output.atom_id().unwrap()], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 3] .
                let %1:f32[] = constant [value=2.5]
                    %2:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %1
                    %3:f32[3, 2] = broadcast [output_type=f32[3, 2], output_axes=[1]] %2
                in (%3)
            "}
            .trim_end(),
        );

        // A dynamic exemplar type keeps the exemplar as an input, whose extents are read when the program runs.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let size = DimensionVariable::new("size", DimensionBounds::non_negative(Some(8)).unwrap());
        let input = context.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(size)])));
        let output = input.fill_like(2.5f64).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![output.atom_id().unwrap()], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[size] .
                let %1:f32[size] = fill_like [value=2.5] %0
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.interpret(vec![Array::vector(vec![7.0f32, 8.0, 9.0]).unwrap()]),
            Ok(vec![Array::vector(vec![2.5f32; 3]).unwrap()]),
        );
        assert_eq!(
            program.interpret(vec![Array::vector(Vec::<f32>::new()).unwrap()]),
            Ok(vec![Array::vector(Vec::<f32>::new()).unwrap()]),
        );
    }

    #[test]
    fn test_fill_like_staging_array_ir() {
        // Array IR values stage the fill of their array member, so a dynamic exemplar binds the homogeneous member
        // operation with the exemplar as its only input.
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let size = DimensionVariable::new("size", DimensionBounds::non_negative(Some(8)).unwrap());
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(size)]));
        let input = context.input(r#type.clone().into());
        let output = input.fill_like(2.5f64).unwrap();
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
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[size] .
                let %1:f32[size] = fill_like [value=2.5] %0
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.interpret(vec![ArrayIrValue::Array(Array::vector(vec![7.0f32, 8.0]).unwrap())]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![2.5f32, 2.5]).unwrap())]),
        );

        // The lifted array IR payload is the homogeneous member operation, and batching an array IR value fills
        // every batch item.
        let lifted = FillLikeOperation::<ArrayIrType>::new(Array::scalar(2.5f32).unwrap()).unwrap();
        assert!(matches!(
            ArrayIrOperation::<Array>::from(lifted),
            ArrayIrOperation::Array(ArrayOperation::FillLike(operation)) if operation == f32_fill_like(2.5),
        ));
        let vector = |values: Vec<f32>| ArrayIrValue::Array(Array::vector(values).unwrap());
        assert_eq!(
            batch(
                |item| Ok(item.fill_like(0.5f32)?),
                vector(vec![2.0, 3.0]),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None
            ),
            Ok(vector(vec![0.5, 0.5])),
        );
    }
}
