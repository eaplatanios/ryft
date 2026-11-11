use std::fmt::Display;
use std::marker::PhantomData;

use ryft_macros::capability;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayOperation,
    ArrayType, DataType,
};
use crate::batching::{BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_elementwise_operation};
use crate::operations::Capability;
use crate::operations::constants::iota::{Iota, IotaOperation};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    Operation, OperationFormatter, ProgramError, RegionInterface, Type, TypeError, Typed, Value, ValueDomainDispatch,
};

/// Canonical operation name for [`IotaLikeOperation`].
pub const IOTA_LIKE_OPERATION_NAME: &str = "iota_like";

/// [`Operation`] that has one exemplar input and that produces a single output with the shape of that input whose
/// elements increase from `0` along a dimension chosen by [`dimension`](Self::dimension), at the element data type
/// chosen by [`data_type`](Self::data_type). It is the counterpart of [`IotaOperation`] for outputs whose shape is
/// only known when the program runs. The exemplar stays an input, so the extents of its dynamic dimensions are read
/// at execution time, while its element values never affect the result. Refer to [`IotaLike`] for the value-level
/// contract.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IotaLikeOperation<T: Type> {
    /// Dimension of the exemplar along which the produced values increase from `0`.
    dimension: usize,

    /// Element [`DataType`] of the produced values.
    data_type: DataType,

    /// [`PhantomData`] marker tying this [`Operation`] to the [`Type`] universe in which it is valid.
    marker: PhantomData<fn() -> T>,
}

impl<T: Type> IotaLikeOperation<T> {
    /// Creates a new [`IotaLikeOperation`] whose output increases along `dimension` and has the element data type
    /// `data_type`. The dimension is validated against the rank of the exemplar during type inference.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] when `data_type` is not numeric.
    pub fn new(dimension: usize, data_type: DataType) -> Result<Self, TypeError> {
        if !data_type.is_numeric() {
            return Err(TypeError::invalid(format!(
                "`{IOTA_LIKE_OPERATION_NAME}` requires a numeric element type but has {data_type}",
            )));
        }
        Ok(Self { dimension, data_type, marker: PhantomData })
    }

    /// Returns the dimension along which the produced values increase from `0`.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the element [`DataType`] of the produced values.
    #[inline]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }
}

impl<T: Type> Copy for IotaLikeOperation<T> {}

impl<T: Type> Display for IotaLikeOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        OperationFormatter::new(formatter, 0, IOTA_LIKE_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("dimension", self.dimension)?;
            operation.field("data_type", self.data_type)
        })
    }
}

impl Operation for IotaLikeOperation<ArrayType> {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        IOTA_LIKE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let exemplar_type = &input_types[0];
        if self.dimension >= exemplar_type.rank() {
            return Err(TypeError::invalid(format!(
                "`{}` dimension {} is out of bounds for rank {}",
                IOTA_LIKE_OPERATION_NAME,
                self.dimension,
                exemplar_type.rank(),
            )));
        }

        // Coordinates are the same on every device, which is a valid value along reduced axes but not a partial
        // result that still needs a cross-device reduction.
        if !exemplar_type.unreduced_axes().is_empty() {
            return Err(TypeError::invalid(format!(
                "`{IOTA_LIKE_OPERATION_NAME}` cannot produce coordinates of type `{exemplar_type}` because it is \
                 unreduced over mesh axes",
            )));
        }

        // The output keeps the exemplar's shape, memory, and sharding. Its layout is kept only for an unchanged
        // element data type, because byte strides do not carry across element widths.
        Ok(vec![if exemplar_type.data_type() == self.data_type {
            exemplar_type.clone()
        } else {
            exemplar_type.clone().with_data_type(self.data_type).with_layout(None)
        }])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, IOTA_LIKE_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("dimension", self.dimension)?;
            operation.field("data_type", self.data_type)
        })
    }
}

impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for IotaLikeOperation<ArrayType>
where
    C::Value: IotaLike,
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].iota_like(self.dimension, self.data_type)?])
    }
}

impl<C: Context<Type = ArrayType, Operation: From<IotaLikeOperation<ArrayType>>>> PartiallyEvaluatableOperation<C>
    for IotaLikeOperation<ArrayType>
{
}

impl<C: Context<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>>
    for IotaLikeOperation<ArrayType>
where
    C::Value: IotaLike,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        _context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        self.infer_output_types(&[inputs[0].unbatched_type()], &[])?;
        match inputs[0].batch_axis_position() {
            // A replicated exemplar has no mapped axis, so every batch item has the same coordinates.
            None => Ok(vec![ArrayBatch::replicated(inputs[0].value().iota_like(self.dimension, self.data_type)?)]),
            Some(batch_axis) => {
                // The coordinates of every batch item increase along the same logical dimension, which sits one
                // physical position later when the mapped axis precedes it. The output keeps the mapped axis of
                // the exemplar, along which the coordinates are constant.
                let dimension = if self.dimension >= batch_axis { self.dimension + 1 } else { self.dimension };
                Ok(vec![ArrayBatch::new(
                    inputs[0].value().iota_like(dimension, self.data_type)?,
                    BatchAxis::from_position(batch_axis),
                )?])
            }
        }
    }
}

impl_differentiable_elementwise_operation!(@constant<T> IotaLikeOperation<T>);

impl<A: Value<Type = ArrayType>> From<IotaLikeOperation<ArrayIrType>> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: IotaLikeOperation<ArrayIrType>) -> Self {
        // The exemplar supplies the complete output shape, including every runtime extent, so the homogeneous member
        // already represents dynamic arrays and the array IR family needs no dimension inputs for it. Projecting that
        // member also rejects dimension and reference exemplars.
        Self::Array(ArrayOperation::IotaLike(IotaLikeOperation {
            dimension: operation.dimension,
            data_type: operation.data_type,
            marker: PhantomData,
        }))
    }
}

/// Represents the ability to construct coordinates with the shape of an exemplar (i.e., a value whose elements
/// increase from `0` along a chosen dimension and are constant along every other dimension). [`IotaLike`] is the
/// value-driven counterpart to [`Iota`] and supplies [`IotaLikeOperation`]'s interpretation capability. For arrays,
/// the result preserves the exemplar's runtime shape, memory placement, and sharding, and has the requested element
/// data type. It also preserves the exemplar's layout when the element data type is unchanged and takes the default
/// layout otherwise, because byte strides do not carry across element widths. The exemplar's element values have no
/// effect on the result, so differentiation returns zero for its tangent or cotangent. A zero-sized axis produces an
/// empty result, integer coordinates narrow to the element type, and complex coordinates have a zero imaginary
/// component.
///
/// A staged call binds the nullary [`IotaOperation`] of the output type when the exemplar's type names no runtime
/// extents, exactly like [`Iota`], which has no data dependence on the exemplar. Otherwise, a staged call binds
/// [`IotaLikeOperation`], which retains the exemplar as an input so that dynamic extents are read at execution time
/// rather than replaced by their allocation bounds. Eager calls always use the exemplar, whose runtime placement may
/// be more specific than its type.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and array IR values implement it for [`ArrayIrType`].
///
/// # Example
///
/// ```rust
/// # use ryft_core::{Array, DataType, IotaLike};
/// let exemplar = Array::matrix(2, 3, vec![0.5f32; 6]).unwrap();
/// assert_eq!(exemplar.iota_like(1, DataType::I32), Ok(Array::matrix(2, 3, vec![0i32, 1, 2, 0, 1, 2]).unwrap()));
/// ```
#[capability(projection(ArrayIrType => ArrayType))]
pub trait IotaLike<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Returns a value with the runtime shape of `self` and the element data type `data_type` whose elements increase
    /// from `0` along `dimension` and are constant along every other dimension.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when `data_type` is not numeric, when `dimension` is out of bounds for the rank of
    /// `self`, or when `self` is unreduced over a mesh axis.
    fn iota_like(&self, dimension: usize, data_type: DataType) -> Result<Self, ProgramError>;
}

impl IotaLike for Array {
    fn iota_like(&self, dimension: usize, data_type: DataType) -> Result<Self, ProgramError> {
        let operation = IotaLikeOperation::<ArrayType>::new(dimension, data_type)?;
        let output_type = operation.infer_output_types(&[self.r#type().into_owned()], &[])?.remove(0);
        self.domain().iota(&output_type, dimension)
    }
}

impl<V> IotaLike<ArrayType> for V
where
    V: Value<Type = ArrayType, Dispatch = ValueDomainDispatch>,
    V::Domain:
        Context<Type = ArrayType, Operation: From<IotaLikeOperation<ArrayType>> + From<IotaOperation<ArrayType>>>,
{
    fn iota_like(&self, dimension: usize, data_type: DataType) -> Result<Self, ProgramError> {
        // A type that names no runtime extents determines the coordinates completely, so the staged operation is the
        // nullary iota that `Iota` stages, which does not read the exemplar and so lets dead code elimination drop
        // it. Eager contexts keep `iota_like`, because the runtime placement of an eager value (e.g., of an
        // automatically sharded array) may be more specific than its type.
        let domain = self.domain();
        let r#type = self.r#type();
        let operation = IotaLikeOperation::<ArrayType>::new(dimension, data_type)?;
        let mut outputs = if !domain.is_eager() && r#type.identities().next().is_none() {
            let output_type = operation.infer_output_types(std::slice::from_ref(r#type.as_ref()), &[])?.remove(0);
            domain.bind(IotaOperation::new(output_type, dimension)?, Vec::new(), &[])?
        } else {
            domain.bind(operation, Vec::new(), std::slice::from_ref(self))?
        };
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
        MeshAxis, MeshAxisType, Shape, Sharding, StridedLayout,
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
    use crate::tracing::TracingContext;

    use super::*;

    /// Returns an [`IotaLikeOperation`] over [`ArrayType`] with the provided dimension and data type.
    fn iota_like(dimension: usize, data_type: DataType) -> IotaLikeOperation<ArrayType> {
        IotaLikeOperation::new(dimension, data_type).unwrap()
    }

    #[test]
    fn test_iota_like() {
        // Verify the operation's payload, validation, and rendering.
        let operation = iota_like(1, DataType::I64);
        assert_eq!(operation.dimension(), 1);
        assert_eq!(operation.data_type(), DataType::I64);
        assert_eq!(operation.to_string(), "iota_like [dimension=1, data_type=i64]");
        assert_eq!(
            IotaLikeOperation::<ArrayType>::new(0, DataType::Boolean),
            Err(TypeError::invalid(format!(
                "`{IOTA_LIKE_OPERATION_NAME}` requires a numeric element type but has bool",
            ))),
        );

        // Verify the operation's textual form when it appears in a program.
        let mut builder = ProgramBuilder::<Array, IotaLikeOperation<ArrayType>>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F32, [2, 3]));
        let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 3] .
                let %1:i64[2, 3] = iota_like [dimension=1, data_type=i64] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_iota_like_type_inference() {
        // The output has the exemplar's shape, including its dynamic dimensions, at the requested data type.
        let size = DimensionVariable::new("size", DimensionBounds::non_negative(Some(8)).unwrap());
        let dynamic_shape = Shape::new(vec![Dimension::Dynamic(size), 3.into()]);
        check_operation_type_inference!(
            operation = iota_like(1, DataType::I64),
            cases = [
                {
                    input_types = [ArrayType::new_static(DataType::F32, [2, 3])],
                    output_types = [ArrayType::new_static(DataType::I64, [2, 3])],
                },
                {
                    input_types = [ArrayType::new(DataType::F32, dynamic_shape.clone())],
                    output_types = [ArrayType::new(DataType::I64, dynamic_shape)],
                },
                {
                    input_types = [ArrayType::new_static(DataType::F32, [2])],
                    error = format!("`{IOTA_LIKE_OPERATION_NAME}` dimension 1 is out of bounds for rank 1"),
                },
            ],
        );

        // The coordinates keep the exemplar's sharding, including its manual variation. They keep its layout only
        // when the element data type is unchanged.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap();
        let layout = Layout::Strided(StridedLayout::new(vec![8]));
        let exemplar_type = ArrayType::new_static(DataType::F32, [2])
            .with_layout(layout)
            .with_sharding(sharding.clone())
            .unwrap();
        assert_eq!(
            iota_like(0, DataType::F32).infer_output_types(&[exemplar_type.clone()], &[]),
            Ok(vec![exemplar_type.clone()]),
        );
        assert_eq!(
            iota_like(0, DataType::I64).infer_output_types(&[exemplar_type], &[]),
            Ok(vec![ArrayType::new_static(DataType::I64, [2]).with_sharding(sharding).unwrap()]),
        );

        // Coordinates are the same on every device, so they cannot be partial results of an unreduced exemplar.
        let unreduced = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh, 1).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        assert_eq!(
            iota_like(0, DataType::I64).infer_output_types(&[unreduced.clone()], &[]),
            Err(TypeError::invalid(format!(
                "`{IOTA_LIKE_OPERATION_NAME}` cannot produce coordinates of type `{unreduced}` because it is \
                 unreduced over mesh axes",
            ))),
        );
    }

    #[test]
    fn test_iota_like_interpretation() {
        let exemplar = Array::matrix(2, 3, vec![0.5f32; 6]).unwrap();
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &iota_like(1, DataType::I64),
                &EagerContext::new(),
                &EmptyRegionDriver,
                std::slice::from_ref(&exemplar),
            ),
            Ok(vec![Array::matrix(2, 3, vec![0i64, 1, 2, 0, 1, 2]).unwrap()]),
        );

        // The coordinates follow the requested dimension and data type, and an empty exemplar gives empty coordinates.
        assert_eq!(
            exemplar.iota_like(0, DataType::F32),
            Ok(Array::matrix(2, 3, vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap())
        );
        let empty = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0, 3]), &[]).unwrap();
        assert_eq!(
            empty.iota_like(1, DataType::I32),
            Ok(Array::from_elements::<i32>(ArrayType::new_static(DataType::I32, [0, 3]), &[]).unwrap()),
        );

        // The coordinates preserve the memory and sharding of the exemplar, and its physical layout only for an
        // unchanged element data type.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2])
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
            .with_memory(Memory::Host { pinned: true })
            .with_sharding(Sharding::replicated(mesh, 1))
            .unwrap();
        let placed = Array::from_elements(r#type.clone(), &[2.0f32, 3.0]).unwrap();
        assert_eq!(
            placed.iota_like(0, DataType::F32),
            Ok(Array::from_elements(r#type.clone(), &[0.0f32, 1.0]).unwrap()),
        );
        assert_eq!(
            placed.iota_like(0, DataType::I64),
            Ok(Array::from_elements(r#type.with_data_type(DataType::I64).with_layout(None), &[0i64, 1]).unwrap()),
        );
        assert_eq!(
            exemplar.iota_like(2, DataType::I32),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{IOTA_LIKE_OPERATION_NAME}` dimension 2 is out of bounds for rank 2",
            )))),
        );
        assert_eq!(
            exemplar.iota_like(0, DataType::Boolean),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{IOTA_LIKE_OPERATION_NAME}` requires a numeric element type but has bool",
            )))),
        );
    }

    #[test]
    fn test_iota_like_interpretation_array_ir() {
        // Array IR values construct the coordinates of their array member, and a first-class dimension is not an
        // array.
        let exemplar = ArrayIrValue::Array(Array::vector(vec![2.0f32, 3.0, 4.0]).unwrap());
        assert_eq!(
            exemplar.iota_like(0, DataType::I32),
            Ok(ArrayIrValue::Array(Array::vector(vec![0i32, 1, 2]).unwrap())),
        );
        let dimension = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(2).unwrap());
        assert_eq!(
            dimension.iota_like(0, DataType::I32),
            Err(ProgramError::Type(TypeError::invalid("expected array type but got dimension type"))),
        );
    }

    #[test]
    fn test_iota_like_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = iota_like(0, DataType::I32),
            inputs = [Array::vector(vec![7.0f32, -1.0]).unwrap()],
            expected = Array::vector(vec![0i32, 1]).unwrap(),
        );

        // The coordinates of an unknown exemplar stay in the residual program, where they read its extents.
        let size = DimensionVariable::new("size", DimensionBounds::non_negative(Some(8)).unwrap());
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(size)]));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(r#type.clone());
        let output = builder.add_instruction(iota_like(0, DataType::I32), Vec::new(), vec![input], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let evaluation = program.partially_evaluate(&[PartialValue::Unknown(r#type)]).unwrap();
        assert_eq!(evaluation.outputs(), &[PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[size] .
                let %1:i32[size] = iota_like [dimension=0, data_type=i32] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_iota_like_batching() {
        // A replicated exemplar gives replicated coordinates. A mapped exemplar keeps its batch axis, and the
        // coordinates of every batch item increase along the same logical dimension, which shifts by one when the
        // mapped axis precedes it.
        check_operation_batching!(
            @exact,
            operation = iota_like(0, DataType::I32),
            axis_size = 2,
            cases = [
                {
                    inputs = [(@replicated, Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap())],
                    outputs = [(@replicated, Array::vector(vec![0i32, 1, 2]).unwrap())],
                },
                {
                    inputs = [(@mapped(axis = 0), Array::matrix(2, 3, vec![0.5f32; 6]).unwrap())],
                    outputs = [(@mapped(axis = 0), Array::matrix(2, 3, vec![0i32, 1, 2, 0, 1, 2]).unwrap())],
                },
                {
                    inputs = [(@mapped(axis = 1), Array::matrix(3, 2, vec![0.5f32; 6]).unwrap())],
                    outputs = [(@mapped(axis = 1), Array::matrix(3, 2, vec![0i32, 0, 1, 1, 2, 2]).unwrap())],
                },
            ],
        );
    }

    #[test]
    fn test_iota_like_differentiation() {
        // The coordinates do not depend on the values of their exemplar, so their tangent is zero.
        assert_eq!(
            differentiate_at(Array::vector(vec![2.0f32, 3.0]).unwrap())
                .jvp(Array::vector(vec![5.0f32, 7.0]).unwrap(), |x| Ok(x.add(&x.iota_like(0, DataType::F32)?)?)),
            Ok((Array::vector(vec![2.0f32, 4.0]).unwrap(), Array::vector(vec![5.0f32, 7.0]).unwrap())),
        );
    }

    #[test]
    fn test_iota_like_transposition() {
        check_operation_transposition!(
            @exact,
            operation = iota_like(0, DataType::F64),
            cases = [{
                inputs = [(@linear(type = ArrayType::new_static(DataType::F64, [2])))],
                output_cotangents = [Array::vector(vec![3.0, 4.0]).unwrap()],
                input_cotangents = [Array::vector(vec![0.0, 0.0]).unwrap()],
            }],
        );
    }

    #[test]
    fn test_iota_like_staging() {
        // A static exemplar type determines the coordinates completely, so staging binds the nullary iota of the
        // output type and leaves the exemplar unused.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = context.input(ArrayType::new_static(DataType::F32, [2, 3]));
        let output = input.iota_like(1, DataType::I64).unwrap();
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
                let %1:i64[2, 3] = iota [type=i64[2, 3], dimension=1]
                in (%1)
            "}
            .trim_end(),
        );

        // Batching over a staging context keeps the batch axis of the exemplar mapped, and the staged coordinates of
        // the batched exemplar, whose type is static too, are again a nullary iota along the shifted dimension.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = context.input(ArrayType::new_static(DataType::F32, [2, 3]));
        let output =
            batch(|item| item.iota_like(0, DataType::I64), input, BatchAxis::new(1), BatchAxis::new(0), None).unwrap();
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
                let %1:i64[2] = iota [type=i64[2], dimension=0]
                    %2:i64[3, 2] = broadcast [output_type=i64[3, 2], output_axes=[1]] %1
                in (%2)
            "}
            .trim_end(),
        );

        // A dynamic exemplar type keeps the exemplar as an input, whose extents are read when the program runs.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let size = DimensionVariable::new("size", DimensionBounds::non_negative(Some(8)).unwrap());
        let input = context.input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(size), 2.into()])));
        let output = input.iota_like(0, DataType::I64).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![output.atom_id().unwrap()], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[size, 2] .
                let %1:i64[size, 2] = iota_like [dimension=0, data_type=i64] %0
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.interpret(vec![Array::matrix(3, 2, vec![0.5f32; 6]).unwrap()]),
            Ok(vec![Array::matrix(3, 2, vec![0i64, 0, 1, 1, 2, 2]).unwrap()]),
        );
    }

    #[test]
    fn test_iota_like_staging_array_ir() {
        // Array IR values stage the coordinates of their array member, so a dynamic exemplar binds the homogeneous
        // member operation with the exemplar as its only input and without any dimension input.
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let size = DimensionVariable::new("size", DimensionBounds::non_negative(Some(8)).unwrap());
        let shape = Shape::new(vec![Dimension::Dynamic(size)]);
        let input = context.input(ArrayType::new(DataType::F32, shape.clone()).into());
        let output = input.iota_like(0, DataType::I32).unwrap();
        assert_eq!(output.r#type().as_ref(), &ArrayIrType::Array(ArrayType::new(DataType::I32, shape)));
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
                let %1:i32[size] = iota_like [dimension=0, data_type=i32] %0
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.interpret(vec![ArrayIrValue::Array(Array::vector(vec![7.0f32, 8.0, 9.0]).unwrap())]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0i32, 1, 2]).unwrap())]),
        );

        // The lifted array IR payload is the homogeneous member operation, and batching an array IR value constructs
        // the coordinates of every batch item.
        assert!(matches!(
            ArrayIrOperation::<Array>::from(IotaLikeOperation::<ArrayIrType>::new(0, DataType::I32).unwrap()),
            ArrayIrOperation::Array(ArrayOperation::IotaLike(operation)) if operation == iota_like(0, DataType::I32),
        ));
        let matrix = |values: Vec<f32>| ArrayIrValue::Array(Array::matrix(2, 2, values).unwrap());
        assert_eq!(
            batch(
                |item| Ok(item.iota_like(0, DataType::F32)?),
                matrix(vec![5.0, 6.0, 7.0, 8.0]),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            ),
            Ok(matrix(vec![0.0, 1.0, 0.0, 1.0])),
        );
    }
}
