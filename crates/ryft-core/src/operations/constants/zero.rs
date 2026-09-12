use std::fmt::Display;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayElement, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation,
    ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType, Dimension, dispatch_on_array_element_type,
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

/// Canonical operation name for [`ZeroOperation`].
pub const ZERO_OPERATION_NAME: &str = "zero";

/// [`Operation`] that has no inputs and that produces a single output that corresponds to the _zero_ value for the
/// [`Type`] that it holds (i.e., for its `r#type` field). For arrays, this would typically correspond to an array of
/// the right type and shape filled with zeros.
///
/// This operation also serves as an [`OperationProvider`] request: it carries the requested output type while the
/// provider receives no input types. Composite operation families select the appropriate member operation from this
/// type; homogeneous families use their ordinary `From<ZeroOperation<T>>` conversion.
/// This constructs zeros whose geometry is fully described by their type. Differentiation's separate
/// [`ResidualZeroProvider`](crate::ResidualZeroProvider) protocol handles zeros that require runtime geometry from
/// residual values, such as disconnected cotangents with dynamic axes.
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
        self.r#type.validate_zero()?;
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

impl_nullary_batchable_operation!(@replicated ZeroOperation<ArrayType>);
impl_nullary_batchable_operation!(@member<ArrayIrType, ArrayIrBatchingPolicy> ZeroOperation<ArrayType>);
impl_non_differentiable_operation!(<T> ZeroOperation<T> where T: Type);
impl_nullary_transposable_operation!(<T> ZeroOperation<T> where T: Type);

impl_member_operation_for_array_ir_constant_operation!(ZeroOperation<ArrayType>, validate_zero);
impl_member_interpretable_operation_for_array_ir_constant_operation!(
    ZeroOperation<ArrayType>,
    Zero,
    |context, output_type, _operation| context.zero(&output_type),
);

impl<A: Value<Type = ArrayType>> From<ZeroOperation<ArrayType>> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: ZeroOperation<ArrayType>) -> Self {
        // Prefer the homogeneous member encoding for identity-free static zeros and the mixed dimension-operand
        // encoding for dynamic output types. Explicit mixed static constructors remain valid, but canonical lifts
        // normalize them to the homogeneous form.
        if operation
            .r#type()
            .shape()
            .dimensions()
            .iter()
            .any(|dimension| matches!(dimension, Dimension::Dynamic(_)))
        {
            Self::Zero(operation)
        } else {
            Self::Array(ArrayOperation::Zero(operation))
        }
    }
}

impl<T: Type, O: Operation<Type = T> + From<ZeroOperation<T>>> OperationProvider<T, ZeroOperation<T>> for O {
    type Operation = Self;

    #[inline]
    fn provide(request: ZeroOperation<T>, input_types: &[&T]) -> Result<Self, ProgramError> {
        check_count!("input", input_types, 0, ProgramError);
        Ok(Self::from(request))
    }
}

impl<A: Value<Type = ArrayType>> OperationProvider<ArrayIrType, ZeroOperation<ArrayIrType>> for ArrayIrOperation<A> {
    type Operation = Self;

    fn provide(request: ZeroOperation<ArrayIrType>, input_types: &[&ArrayIrType]) -> Result<Self, ProgramError> {
        check_count!("input", input_types, 0, ProgramError);
        let r#type = match request.r#type {
            ArrayIrType::Array(r#type) => r#type,
            ArrayIrType::Dimension(_) => {
                // A first-class dimension is a symbolic runtime extent rather than an algebraic value. A zero dimension
                // may violate the type's bounds, and assigning zero would bind its identity to an extent that may
                // disagree with the runtime definition. Dimension tangents and cotangents use the separate array
                // `DataType::Zero` representation.
                return Err(TypeError::invalid("cannot materialize a zero for a first-class dimension type").into());
            }
            ArrayIrType::Reference(r#type) => {
                return Err(TypeError::invalid(format!(
                    "cannot materialize a zero for reference type `{}`; a reference denotes an allocation and has \
                     no zero value, so tangent and cotangent references are allocated by the differentiation rules",
                    r#type,
                ))
                .into());
            }
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
        r#type.validate_zero()?;
        match r#type.data_type() {
            DataType::Token => {
                Err(TypeError::invalid(format!("data type `{}` cannot represent zero", DataType::Token)).into())
            }
            DataType::Zero => Array::new(r#type.clone(), Vec::new()),
            data_type => dispatch_on_array_element_type!(data_type, |Element| {
                Array::from_fn_elements(r#type.clone(), |_| Ok(Element::zero()?))
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
        r#type.validate_zero()?;
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
        let mut outputs = self.bind(ZeroOperation::new(r#type.clone()), Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: StagingContext> Zero<Tracer<C>> for C
where
    C::Operation: OperationProvider<C::Type, ZeroOperation<C::Type>, Operation = C::Operation>,
{
    #[inline]
    fn zero(&self, r#type: &C::Type) -> Result<Tracer<C>, ProgramError> {
        let mut outputs =
            self.stage_nullary_operation(C::Operation::provide(ZeroOperation::new(r#type.clone()), &[])?)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Context> Zero<PartialTracer<C>> for PartialEvaluationContext<C>
where
    C::Operation: PartiallyEvaluatableOperation<C>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + OperationProvider<C::Type, ZeroOperation<C::Type>, Operation = C::Operation>,
{
    #[inline]
    fn zero(&self, r#type: &C::Type) -> Result<PartialTracer<C>, ProgramError> {
        let mut outputs =
            self.bind(C::Operation::provide(ZeroOperation::new(r#type.clone()), &[])?, Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Context<Type = ArrayType> + Zero<C::Value>> Zero<BatchingTracer<C, ArrayBatchingPolicy>>
    for BatchingContext<C, ArrayBatchingPolicy>
{
    #[inline]
    fn zero(&self, r#type: &ArrayType) -> Result<BatchingTracer<C, ArrayBatchingPolicy>, ProgramError> {
        let batch = ArrayBatch::new(self.parent().zero(r#type)?, BatchAxis::replicated())?;
        Ok(BatchingTracer::new(self.clone(), batch))
    }
}

impl<C: Context<Type = ArrayIrType> + Zero<C::Value>> Zero<BatchingTracer<C, ArrayIrBatchingPolicy>>
    for BatchingContext<C, ArrayIrBatchingPolicy>
{
    #[inline]
    fn zero(&self, r#type: &ArrayIrType) -> Result<BatchingTracer<C, ArrayIrBatchingPolicy>, ProgramError> {
        let batch = ArrayIrBatch::new(self.parent().zero(r#type)?, BatchAxis::replicated())?;
        Ok(BatchingTracer::new(self.clone(), batch))
    }
}

impl<C: Context<Type: DifferentiableType> + Zero<C::Value>, P: DifferentiationPolicy<C>>
    Zero<DifferentiationTracer<C, P>> for DifferentiationContext<C, P>
{
    #[inline]
    fn zero(&self, r#type: &C::Type) -> Result<DifferentiationTracer<C, P>, ProgramError> {
        let dual = DifferentiationDual::new_with_zero_tangent(self.primal().zero(r#type)?)?;
        Ok(DifferentiationTracer::new(dual, self.clone()))
    }
}

/// Represents the ability to construct an [`Array`] of zeros whose shape includes _dynamic_ (i.e., runtime) dimensions.
/// Unlike [`Zero`], this capability supplies each dynamic axis with an explicit dimension value. Static axes retain
/// their declared sizes. Dynamic axes take their sizes from the inputs in axis order. Repeated dimension identities
/// require a corresponding input for every occurrence. Each input must have the exact dimension identity declared
/// by its axis. Repeated inputs must carry the same runtime extent. Conflicting extents are rejected during
/// interpretation.
///
/// Note that a fully static output type is also accepted with no dimension inputs. The same capability works with eager
/// mixed-IR values and with tracer values, where construction records the dimension inputs in the staged program.
/// When a dynamic axis has bounds that admit only one extent, the inferred result type represents that axis as static.
/// Its dimension input is still required because the declared output type contains a dynamic axis.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{
/// #     Array, ArrayIrOperation, ArrayIrValue, ArrayType, DataType, Dimension, DimensionBounds, DimensionType,
/// #     DimensionValue, DimensionVariable, DynamicZero, EagerContext, Shape,
/// # };
/// let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
/// let size = DimensionVariable::new("size", DimensionBounds::unbounded());
/// let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(size.clone())]));
/// let dimension = ArrayIrValue::Dimension(DimensionValue::new(DimensionType::new(size), 3).unwrap());
/// assert_eq!(
///     context.dynamic_zero(&r#type, &[dimension]),
///     Ok(ArrayIrValue::Array(Array::vector(vec![0.0f32; 3]).unwrap())),
/// );
/// ```
pub trait DynamicZero<V: Typed> {
    /// Constructs an array of zeros with explicit values for its dynamic dimensions.
    ///
    /// # Parameters
    ///
    ///   - `type`: Output [`ArrayType`] that may contain dynamic dimensions. Each dynamic axis names the dimension
    ///     identity that its corresponding input must carry.
    ///   - `dimensions`: Contains one dimension value per dynamic axis, in axis order. Static axes do not consume input
    ///     dimensions provided this way. Repeated identities still consume one input for each axis that uses them.
    fn dynamic_zero(&self, r#type: &ArrayType, dimensions: &[V]) -> Result<V, ProgramError>;
}

impl<C: Context<Type = ArrayIrType, Operation: From<ZeroOperation<ArrayType>>>> DynamicZero<C::Value> for C {
    #[inline]
    fn dynamic_zero(&self, r#type: &ArrayType, dimensions: &[C::Value]) -> Result<C::Value, ProgramError> {
        r#type.validate_zero()?;
        validate_dynamic_constant_dimensions(ZERO_OPERATION_NAME, r#type, dimensions)?;
        let mut outputs = self.bind(ZeroOperation::new(r#type.clone()), Vec::new(), dimensions)?;
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
        Array, ArrayBatch, ArrayBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, DataType,
        Dimension, DimensionBounds, DimensionType, DimensionValue, DimensionVariable, Layout, Shape, StridedLayout, i4,
    };
    use crate::batching::{BatchAxis, BatchableOperation, BatchingContext};
    use crate::contexts::EagerContext;
    use crate::differentiation::{ForwardModeDifferentiate, TransposableOperation, TranspositionContext};
    use crate::interpretation::InterpretableOperation;
    use crate::macros::check_operation_partial_evaluation;
    use crate::operations::constants::constant::ConstantOperation;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationValue, PartialValue};
    use crate::programs::{EmptyRegionDriver, MaybeZero, Operation, ProgramBuilder, ReferenceType};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_zero() {
        // Verify the operation's stored type, identity, zero metadata, and rendering.
        let operation = ZeroOperation::new(ArrayType::scalar(DataType::F64));
        assert_eq!(operation.name(), ZERO_OPERATION_NAME);
        assert!(operation.is_zero(0));
        assert!(!operation.is_zero(1));
        assert_eq!(format!("{operation}"), "zero [type=f64[]]");
        assert_eq!(operation.r#type(), &ArrayType::scalar(DataType::F64));
        // Verify the operation's textual form when it appears in a program.
        let mut builder = ProgramBuilder::<Array, ZeroOperation<ArrayType>>::new();
        let output = builder.add_instruction(operation, Vec::new(), vec![], None).unwrap()[0];
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
    fn test_zero_type_inference() {
        // Representability is independent of the number of elements and is checked before staging.
        for data_type in [DataType::Token, DataType::F8E8M0FNU] {
            for shape in [vec![], vec![0], vec![2]] {
                let r#type = ArrayType::new_static(data_type, shape);
                let error = TypeError::invalid(format!("data type `{data_type}` cannot represent zero"));
                assert_eq!(ZeroOperation::new(r#type.clone()).infer_output_types(&[], &[]), Err(error.clone()));
                let context = TracingContext::<Array, ArrayOperation<Array>>::new();
                assert_eq!(context.zero(&r#type).unwrap_err(), ProgramError::Type(error.clone()));
                assert!(context.builder().borrow().instructions().is_empty());
                let context = EagerContext::<Array>::new();
                assert_eq!(context.zero(&r#type), Err(ProgramError::Type(error)));
            }
        }

        let operation = ZeroOperation::new(ArrayType::scalar(DataType::F64));
        assert_eq!(operation.infer_output_types(&[], &[]), Ok(vec![ArrayType::scalar(DataType::F64)]));

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
    }

    #[test]
    fn test_zero_type_inference_dynamic_identity_instantiation() {
        let formal = DimensionVariable::new("formal", DimensionBounds::new(1, Some(5)).unwrap());
        let caller = DimensionVariable::new("caller", DimensionBounds::new(2, Some(4)).unwrap());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let extent = builder.add_input(DimensionType::new(formal.clone()).into());
        let output = builder
            .add_instruction(
                ZeroOperation::new(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(formal)]))),
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

        // Genuine cross-program instantiation: deriving the caller renaming from the complete boundary signature
        // renames the whole program, including the dynamic zero's stored output type, and recloses its region
        // arena, so the instantiated payload stays consistent with the instantiated atom types.
        let caller_input = ArrayIrType::Dimension(DimensionType::new(caller.clone()));
        let instantiated = program.with_instantiated_type_identities(std::slice::from_ref(&caller_input)).unwrap();
        assert_eq!(instantiated.input_types(), vec![caller_input]);
        assert_eq!(
            instantiated.output_types(),
            vec![ArrayIrType::Array(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(caller.clone())]),
            ))],
        );
        let [instruction] = instantiated.instructions() else {
            panic!("expected one instantiated instruction");
        };
        let ArrayIrOperation::Zero(instantiated_zero) = instruction.operation() else {
            panic!("expected the instantiated operation to remain a dynamic zero");
        };
        assert_eq!(
            instantiated_zero.r#type(),
            &ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(caller.clone())])),
        );
        assert_eq!(
            instantiated.interpret(vec![ArrayIrValue::Dimension(
                DimensionValue::new(DimensionType::new(caller.clone()), 3).unwrap()
            )]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0_f32, 0.0, 0.0]).unwrap())]),
        );

        // A boundary interpretation of the *uninstantiated* program with an actual input type that uses a different
        // dimension identity takes the non-exact establishment path instead: the actual dimension member refines the
        // declared one by bounds alone, and the concrete static output then establishes its first fact for the declared
        // input identity through the closed identity signature.
        assert_eq!(
            program
                .interpret(vec![ArrayIrValue::Dimension(DimensionValue::new(DimensionType::new(caller), 3).unwrap())]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0_f32, 0.0, 0.0]).unwrap())]),
        );
    }

    #[test]
    fn test_zero_interpretation() {
        let operation = ZeroOperation::new(ArrayType::scalar(DataType::F64));
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[]
            ),
            Ok(vec![Array::scalar(0.0).unwrap()]),
        );

        let context = EagerContext::<Array>::new();

        // Verify canonical rank-zero zero values across every supported data-type family.
        for (r#type, expected) in [
            (DataType::Boolean, Array::scalar(false).unwrap()),
            (DataType::I8, Array::scalar(0i8).unwrap()),
            (DataType::I16, Array::scalar(0i16).unwrap()),
            (DataType::I32, Array::scalar(0i32).unwrap()),
            (DataType::I64, Array::scalar(0i64).unwrap()),
            (DataType::U8, Array::scalar(0u8).unwrap()),
            (DataType::U16, Array::scalar(0u16).unwrap()),
            (DataType::U32, Array::scalar(0u32).unwrap()),
            (DataType::U64, Array::scalar(0u64).unwrap()),
            (DataType::BF16, Array::scalar(bf16::ZERO).unwrap()),
            (DataType::F16, Array::scalar(f16::ZERO).unwrap()),
            (DataType::F32, Array::scalar(0.0f32).unwrap()),
            (DataType::F64, Array::scalar(0.0f64).unwrap()),
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

        // Composite eager zero materialization delegates array members and rejects first-class dimensions and
        // references. A reference names an allocation with persistent identity and cannot be conjured from its type.
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        assert_eq!(context.zero(&ArrayIrType::Array(output_type.clone())), Ok(ArrayIrValue::Array(expected)));
        let dimension_type =
            ArrayIrType::Dimension(DimensionType::new(DimensionVariable::new("size", DimensionBounds::unbounded())));
        assert_eq!(
            context.zero(&dimension_type),
            Err(ProgramError::Type(TypeError::invalid("cannot materialize a zero for a first-class dimension type"))),
        );
        assert_eq!(
            context.zero(&ArrayIrType::Reference(ReferenceType::new(output_type))),
            Err(ProgramError::Type(TypeError::invalid(
                "cannot materialize a zero for reference type `ref<f32[2, 3]>`; a reference denotes an allocation and has \
                 no zero value, so tangent and cotangent references are allocated by the differentiation rules",
            ))),
        );
    }

    #[test]
    fn test_zero_interpretation_layout_and_dynamic_type() {
        let context = EagerContext::<Array>::new();
        let r#type = ArrayType::new_static(DataType::F32, [2, 2]);
        assert_eq!(
            context.zero(&r#type),
            Array::from_elements(r#type.clone(), &[0.0f32; 4]).map_err(|_| unreachable!())
        );
        // Constructors dispatch over element codecs that have no scalar representation and honor physical layout.
        let strided_type =
            ArrayType::new_static(DataType::I4, [3]).with_layout(Layout::Strided(StridedLayout::new(vec![-1])));
        let zero = context.zero(&strided_type).unwrap();
        assert_eq!(zero.elements::<i4>(), Ok(vec![i4::new(0).unwrap(); 3]));
        assert_eq!(zero.storage_bytes(), [0, 0, 0]);

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
        let expected_message = "cannot materialize a value of dynamically sized type f64[dynamic, 3]; dynamically \
                                shaped values exist only in array programs over `ArrayIrOperation`";
        assert!(matches!(
            context.zero(&dynamic_type),
            Err(ProgramError::Type(TypeError::Invalid { message })) if message == expected_message,
        ));
    }

    #[test]
    fn test_zero_partial_evaluation() {
        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let output_type = ArrayType::new_static(DataType::F32, [2]);
        let output = context.zero(&output_type).unwrap();
        let expected = Array::from_elements(output_type, &[0.0f32; 2]).unwrap();
        assert_eq!(output.value().unwrap().as_known(), Some(&expected));
    }

    #[test]
    fn test_zero_partial_evaluation_dynamic() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let extent = ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap());
        let output = ArrayIrValue::Array(Array::vector(vec![0.0_f32, 0.0, 0.0]).unwrap());
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = ZeroOperation::new(ArrayType::new(
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
    fn test_zero_batching() {
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

        let context =
            BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 4);
        let output_type = ArrayType::new_static(DataType::F32, [2]);
        let output = context.zero(&output_type).unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(output.batch().value(), &Array::from_elements(output_type, &[0.0f32; 2]).unwrap());

        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap()),
        );
        let output = context.zero(&ArrayIrType::Array(ArrayType::scalar(DataType::F32))).unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(output.batch().value(), &ArrayIrValue::Array(Array::scalar(0.0_f32).unwrap()));
    }

    #[test]
    fn test_zero_differentiation() {
        let context = DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new());
        let output_type = ArrayType::new_static(DataType::F32, [2]);
        let output = context.zero(&output_type).unwrap();
        assert_eq!(output.primal(), &Array::from_elements(output_type.clone(), &[0.0f32; 2]).unwrap());
        assert!(matches!(output.tangent(), MaybeZero::Zero(r#type) if r#type == &output_type));
    }

    #[test]
    fn test_zero_differentiation_dynamic() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let extent = builder.add_input(extent_type.clone().into());
        let output = builder
            .add_instruction(
                ZeroOperation::new(ArrayType::new(
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
        let extent = ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap());
        assert_eq!(
            jvp.interpret(vec![extent]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![0.0_f64, 0.0, 0.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![0.0_f64, 0.0, 0.0]).unwrap()),
            ]),
        );
        assert_eq!(jvp.instructions().iter().filter(|instruction| instruction.operation().is_zero(0)).count(), 1);

        // Direct differentiation-context dispatch takes the same all-zero shortcut. Its tangent must reuse the shaped
        // primal SSA value rather than materializing a nullary zero that has no access to the runtime extent.
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let extent = context.input(extent_type.clone().into());
        let extent_tangent = context.input(ArrayType::scalar(DataType::Zero).into());
        let dynamic_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]));
        let (primal, tangent) = context
            .jvp(
                move |extent, ()| {
                    let context = extent.context().clone();
                    context.dynamic_zero(&dynamic_type, &[extent])
                },
                extent,
                extent_tangent,
                (),
            )
            .unwrap();
        assert_eq!(primal.atom_id(), tangent.atom_id());
        let builder = context.builder().borrow();
        let [instruction] = builder.instructions() else {
            panic!("expected one dynamic-zero instruction");
        };
        assert!(matches!(instruction.operation(), ArrayIrOperation::Zero(_)));
    }

    #[test]
    fn test_zero_transposition() {
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = context.input(ArrayType::scalar(DataType::F64));
        let input_cotangents = ZeroOperation::new(ArrayType::scalar(DataType::F64))
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
    fn test_zero_transposition_dynamic() {
        // Dynamic constructors depend on their extent operands only as non-differentiable shape inputs, so every
        // extent receives a structural-zero cotangent regardless of the output cotangent being live.
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let output_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]));
        let operation = ArrayIrOperation::<Array>::from(ZeroOperation::new(output_type.clone()));
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
    fn test_operation_provider_zero() {
        // Homogeneous operation families construct nullary operations through their ordinary
        // `From<ZeroOperation<T>>` conversion.
        let static_type = ArrayType::new_static(DataType::F32, [2]);
        let ArrayOperation::<Array>::Zero(operation) =
            ArrayOperation::<Array>::provide(ZeroOperation::new(static_type.clone()), &[]).unwrap()
        else {
            panic!("expected a homogeneous zero operation");
        };
        assert_eq!(operation.r#type(), &static_type);

        // Output types belong to the request; nullary construction rejects any operand types.
        assert_eq!(
            ArrayOperation::<Array>::provide(ZeroOperation::new(static_type.clone()), &[&static_type]).unwrap_err(),
            ProgramError::InvalidInputCount { expected: 0, actual: 1 },
        );
        let composite_type = ArrayIrType::Array(static_type.clone());
        assert_eq!(
            ArrayIrOperation::<Array>::provide(ZeroOperation::new(composite_type.clone()), &[&composite_type])
                .unwrap_err(),
            ProgramError::InvalidInputCount { expected: 0, actual: 1 },
        );

        // The composite provider projects a valid operand-free array zero into the homogeneous member family.
        let ArrayIrOperation::<Array>::Array(ArrayOperation::Zero(operation)) =
            ArrayIrOperation::<Array>::provide(ZeroOperation::new(ArrayIrType::Array(static_type.clone())), &[])
                .unwrap()
        else {
            panic!("expected a composite homogeneous zero operation");
        };
        assert_eq!(operation.r#type(), &static_type);

        // Operand-free construction cannot resolve a dynamic identity. Dynamic mixed zeros must instead receive their
        // concrete extents as dimension operands.
        let size = DimensionVariable::new("size", DimensionBounds::unbounded());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(size.clone())]));
        assert_eq!(
            ArrayIrOperation::<Array>::provide(ZeroOperation::new(ArrayIrType::Array(dynamic_type)), &[]).unwrap_err(),
            ProgramError::Type(TypeError::invalid(
                "`zero` cannot construct type f32[size] without operands because it references identity size",
            )),
        );

        // First-class dimensions and references are not algebraic values. In particular, a reference cannot be replaced
        // by a zero of its referent type. The differentiation rules allocate tangent and cotangent references instead
        // of ever materializing a zero reference.
        assert_eq!(
            ArrayIrOperation::<Array>::provide(
                ZeroOperation::new(ArrayIrType::Dimension(DimensionType::new(size))),
                &[],
            )
            .unwrap_err(),
            ProgramError::Type(TypeError::invalid("cannot materialize a zero for a first-class dimension type")),
        );
        let reference_type = ReferenceType::new(static_type);
        assert_eq!(
            ArrayIrOperation::<Array>::provide(ZeroOperation::new(ArrayIrType::Reference(reference_type.clone())), &[])
                .unwrap_err(),
            ProgramError::Type(TypeError::invalid(format!(
                "cannot materialize a zero for reference type `{reference_type}`; a reference denotes an allocation \
                 and has no zero value, so tangent and cotangent references are allocated by the differentiation rules",
            ))),
        );
    }

    #[test]
    fn test_operation_provider_zero_static_array_ir() {
        let r#type = ArrayType::scalar(DataType::F32);
        let operation = ArrayIrOperation::<Array>::provide(ZeroOperation::new(r#type.clone().into()), &[]).unwrap();
        assert!(
            matches!(operation, ArrayIrOperation::Array(ArrayOperation::Zero(operation)) if operation.r#type() == &r#type)
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
    fn test_zero_staging() {
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
    fn test_dynamic_zero() {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(5)).unwrap());
        let output_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Static(2)]));
        let extent = ArrayIrValue::Dimension(DimensionValue::new(DimensionType::new(rows), 3).unwrap());
        assert_eq!(
            context.dynamic_zero(&output_type, &[extent]),
            Ok(ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::F32, [3, 2]), &[0.0f32; 6]).unwrap(),
            )),
        );

        // The same API accepts static output types without fabricating dimension operands.
        assert_eq!(
            context.dynamic_zero(&ArrayType::new_static(DataType::I32, [2]), &[]),
            Ok(ArrayIrValue::Array(Array::vector(vec![0i32, 0]).unwrap())),
        );
    }

    #[test]
    fn test_dynamic_zero_invalid_dimensions() {
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let size = DimensionVariable::new("size", DimensionBounds::unbounded());
        let dimension = context.input(DimensionType::new(size.clone()).into());
        for data_type in [DataType::Token, DataType::F8E8M0FNU] {
            let r#type = ArrayType::new(data_type, Shape::new(vec![size.clone().into()]));
            assert_eq!(
                context.dynamic_zero(&r#type, &[dimension.clone()]).unwrap_err(),
                ProgramError::Type(TypeError::invalid(format!("data type `{data_type}` cannot represent zero"))),
            );
            assert!(context.builder().borrow().instructions().is_empty());
        }

        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(8)).unwrap());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows)]));
        assert_eq!(
            context.dynamic_zero(&output_type, &[]),
            Err(ProgramError::Type(TypeError::invalid(
                "`zero` expects one dimension operand per dynamic output dimension (1) but got 0 operands",
            ))),
        );
        assert_eq!(
            context.dynamic_zero(&output_type, &[ArrayIrValue::Array(Array::scalar(3.0f32).unwrap())]),
            Err(ProgramError::Type(TypeError::invalid("`zero` operand 0 must be a dimension but has type f32[]",))),
        );
        let other = DimensionVariable::new("other", DimensionBounds::non_negative(Some(8)).unwrap());
        let extent = ArrayIrValue::Dimension(DimensionValue::new(DimensionType::new(other), 3).unwrap());
        assert_eq!(
            context.dynamic_zero(&output_type, &[extent]),
            Err(ProgramError::Type(TypeError::invalid(
                "`zero` operand 0 has type dimension<other ∈ [0, 8)> but the output shape requires \
                 dimension<rows ∈ [0, 8)>",
            ))),
        );

        // Matching diagnostic names and bounds do not make separately created identities interchangeable.
        let distinct = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(8)).unwrap());
        let dimension = ArrayIrValue::Dimension(DimensionValue::new(DimensionType::new(distinct), 3).unwrap());
        assert_eq!(
            context.dynamic_zero(&output_type, &[dimension]),
            Err(ProgramError::Type(TypeError::invalid(
                "`zero` operand 0 has type dimension<rows ∈ [0, 8)> but the output shape requires dimension<rows ∈ [0, 8)>",
            ))),
        );

        // Invalid requests fail before staging any instruction or altering the builder.
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        assert_eq!(
            context.dynamic_zero(&output_type, &[]).unwrap_err(),
            ProgramError::Type(TypeError::invalid(
                "`zero` expects one dimension operand per dynamic output dimension (1) but got 0 operands",
            )),
        );
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_dynamic_zero_staging() {
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(5)).unwrap());
        let output_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Static(2)]));
        let extent_type = DimensionType::new(rows);
        let extent = context.input(extent_type.clone().into());
        let output = context.dynamic_zero(&output_type, &[extent]).unwrap();
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
            panic!("expected one dynamic-zero instruction");
        };
        assert!(matches!(instruction.operation(), ArrayIrOperation::Zero(_)));
        assert_eq!(
            program.interpret(vec![ArrayIrValue::Dimension(DimensionValue::new(extent_type, 2).unwrap())]),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::F32, [2, 2]), &[0.0f32; 4]).unwrap(),
            )]),
        );

        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        context.dynamic_zero(&ArrayType::scalar(DataType::F32), &[]).unwrap();
        let builder = context.builder().borrow();
        let [instruction] = builder.instructions() else {
            panic!("expected one static-zero instruction");
        };
        assert!(matches!(instruction.operation(), ArrayIrOperation::Array(ArrayOperation::Zero(_))));
    }

    #[test]
    fn test_dynamic_zero_staging_singleton_dimension() {
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let size = DimensionVariable::new("size", DimensionBounds::new(3, Some(4)).unwrap());
        let dimension_type = DimensionType::new(size.clone());
        let dimension = context.input(dimension_type.clone().into());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(size)]));
        let output = context.dynamic_zero(&output_type, std::slice::from_ref(&dimension)).unwrap();

        // The explicit dimension input binds the declared identity even when its bounds imply one extent.
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
            panic!("expected one dynamic zero instruction");
        };
        assert!(
            matches!(instruction.operation(), ArrayIrOperation::Zero(operation) if operation.r#type() == &output_type)
        );
        assert_eq!(instruction.inputs(), &[dimension.atom_id().unwrap()]);
        let extent = ArrayIrValue::Dimension(DimensionValue::new(dimension_type, 3).unwrap());
        assert_eq!(
            program.interpret(vec![extent.clone()]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0f32; 3]).unwrap())]),
        );
        assert_eq!(
            program.jvp().unwrap().interpret(vec![extent]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![0.0f32; 3]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![0.0f32; 3]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_dynamic_zero_partial_evaluation() {
        let context =
            PartialEvaluationContext::new(EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new());
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(5)).unwrap());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone())]));
        let extent_type = DimensionType::new(rows);
        let extent = ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap());
        let known = PartialTracer::new(context.clone(), PartialEvaluationValue::known(extent));
        let output = context.dynamic_zero(&output_type, &[known]).unwrap();
        assert_eq!(
            output.value().unwrap().as_known(),
            Some(&ArrayIrValue::Array(Array::vector(vec![0.0f32; 3]).unwrap()))
        );

        // Unknown dimensions remain operands of a residual constructor instead of forcing a static shape.
        let unknown = PartialTracer::new(context.clone(), context.unknown_input(extent_type.into(), 0));
        let output = context.dynamic_zero(&output_type, &[unknown]).unwrap();
        assert_eq!(output.value().unwrap().as_known(), None);
        assert_eq!(output.r#type().as_ref(), &ArrayIrType::Array(output_type));
    }

    #[test]
    fn test_dynamic_zero_batching() {
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap()),
        );
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(5)).unwrap());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone())]));
        let extent = ArrayIrValue::Dimension(DimensionValue::new(DimensionType::new(rows), 3).unwrap());
        let extent = BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(extent));
        let output = context.dynamic_zero(&output_type, &[extent]).unwrap();
        assert_eq!(output.batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(output.batch().value(), &ArrayIrValue::Array(Array::vector(vec![0.0f32; 3]).unwrap()));
    }

    #[test]
    fn test_dynamic_zero_transposition_context() {
        // Transposition exposes its parent trace's capabilities through dereferencing.
        let context = TranspositionContext::new(TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new());
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(5)).unwrap());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone())]));
        let extent = context.input(DimensionType::new(rows).into());
        let output = context.dynamic_zero(&output_type, &[extent]).unwrap();
        assert_eq!(output.r#type().as_ref(), &ArrayIrType::Array(output_type));
        let builder = context.builder().borrow();
        let [instruction] = builder.instructions() else {
            panic!("expected one dynamic-zero instruction");
        };
        assert!(matches!(instruction.operation(), ArrayIrOperation::Zero(_)));
    }
}
