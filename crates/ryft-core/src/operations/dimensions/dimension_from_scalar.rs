//! Checked conversion from ordinary scalar-array data into a first-class dimension value.
//!
//! The user-facing semantics and example live on [`DimensionFromScalar`]. The operation records the produced
//! [`DimensionType`] directly so its identity and authoritative bounds remain one structural SSA definition.

use std::fmt::Display;

use ryft_macros::Parameter;

use crate::arrays::batching::{align_array_batch, array_dimension};
use crate::arrays::{
    ArrayIrBatch, ArrayIrBatching, ArrayIrType, ArrayType, DimensionType, DimensionValue, DimensionVariable,
};
use crate::axes::Axis;
use crate::batching::{BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_differentiable_operation, impl_non_transposable_operation};
use crate::operations::{
    ConstantOperation, DimensionSizeOperation, DimensionToScalarOperation, DynamicBroadcastOperation, ScanOperation,
    TransposeOperation,
};
use crate::parameters::{Parameter, Placeholder};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    Effect, Effects, Operation, OperationFormatter, OperationProjection, ProgramBuilder, ProgramError, ProjectedValue,
    RegionInterface, Type, TypeError, TypeIdentityRenaming, Typed, Value, ValueProjection,
};

/// Canonical operation name for [`DimensionFromScalarOperation`].
pub const DIMENSION_FROM_SCALAR_OPERATION_NAME: &str = "dimension_from_scalar";

/// Converts ordinary rank-zero integer array data into a checked first-class dimension.
///
/// This is the explicit boundary that converts numerical data into a dimension value. `result` declares the fresh
/// [`DimensionVariable`] and authoritative bounds of the produced dimension. Eager execution rejects negative,
/// out-of-bounds, host-unrepresentable, and backend-width-incompatible values before returning the dimension.
///
/// The input may use any signed or unsigned integer element type. It must be rank zero. Under batching, a mapped
/// scalar produces one checked extent per item. Those extents remain packed scalar-array data on the transform-owned
/// batch carrier and become ragged geometry only when a shape-consuming batching rule accepts them.
///
/// # Example
///
/// ```rust
/// # use ryft_core::arrays::{DimensionBounds, DimensionVariable};
/// # use ryft_core::{ArrayIrValue, DimensionFromScalar, DimensionValue, Mul, ProgramError};
/// # use ryft_core::arrays::Array;
/// # fn main() -> Result<(), ProgramError> {
/// let scalar = ArrayIrValue::Array(Array::scalar(5_i32));
/// let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
/// let dimension = scalar.to_dimension(batch)?;
/// let ArrayIrValue::Dimension(dimension) = dimension else {
///     unreachable!("dimension_from_scalar always returns a dimension member");
/// };
/// assert_eq!(dimension.extent(), 5);
/// let doubled = dimension.mul(&DimensionValue::constant(2)?)?;
/// assert_eq!(doubled.extent(), 10);
/// # Ok(())
/// # }
/// ```
///
/// Extract vector elements with ordinary array operations before crossing this gateway. This keeps indexing a general
/// array concern and makes this operation the only numerical-data-to-dimension boundary:
///
/// ```rust
/// # use ryft_core::arrays::{DimensionBounds, DimensionVariable, Shape};
/// # use ryft_core::{DimensionFromScalar, ProgramError, Reshape, Slice};
/// # use ryft_core::arrays::Array;
/// # fn main() -> Result<(), ProgramError> {
/// let extents = Array::vector(vec![3_i32, 5_i32]);
/// let sequence = extents.slice(&[1], &[2], &[1])?.reshape(Shape::scalar())?;
/// let sequence = sequence.to_dimension(DimensionVariable::new(
///     "sequence",
///     DimensionBounds::new(1, Some(9))?,
/// ))?;
/// assert_eq!(sequence.extent(), 5);
/// # Ok(())
/// # }
/// ```
pub trait DimensionFromScalar<Output = Self>: Typed + Sized {
    /// Returns this rank-zero integer array as a first-class dimension described by `result`.
    fn to_dimension(&self, result: DimensionVariable) -> Result<Output, ProgramError>;
}

impl<V: Value<Type = ArrayIrType>> DimensionFromScalar<V> for V
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<DimensionFromScalarOperation>,
{
    fn to_dimension(&self, result: DimensionVariable) -> Result<V, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(DimensionFromScalarOperation::new(result), Vec::new(), std::slice::from_ref(self))?
            .remove(0))
    }
}

impl<V: Value<Type = ArrayIrType>> DimensionFromScalar<V> for ProjectedValue<ArrayType, V>
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<DimensionFromScalarOperation>,
{
    fn to_dimension(&self, result: DimensionVariable) -> Result<V, ProgramError> {
        Ok(self
            .value()
            .dispatch_domain()
            .bind(DimensionFromScalarOperation::new(result), Vec::new(), std::slice::from_ref(self.value()))?
            .remove(0))
    }
}

/// Mixed scalar-array-to-dimension operation used by [`DimensionFromScalar`].
///
/// Refer to [`DimensionFromScalar`] for semantic details and an example.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct DimensionFromScalarOperation {
    /// First-class dimension type produced by this operation.
    result_type: DimensionType,
}

impl DimensionFromScalarOperation {
    /// Creates a checked scalar-data gateway that produces a dimension described by `result`.
    #[inline]
    pub fn new(result: DimensionVariable) -> Self {
        Self { result_type: DimensionType::new(result) }
    }

    /// Returns the first-class dimension type produced by this operation.
    #[inline]
    pub fn result_type(&self) -> &DimensionType {
        &self.result_type
    }

    /// Validates the array member accepted by this gateway without allocating an inferred output collection.
    pub(crate) fn validate_input_type(input_type: &ArrayType) -> Result<(), TypeError> {
        if input_type.rank() != 0 || !input_type.data_type().is_integer() {
            return Err(TypeError::invalid(format!(
                "`{DIMENSION_FROM_SCALAR_OPERATION_NAME}` input must be a rank-0 integer array but has type \
                 {input_type}",
            )));
        }
        Ok(())
    }
}

impl Display for DimensionFromScalarOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for DimensionFromScalarOperation {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        DIMENSION_FROM_SCALAR_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let input_type = <&ArrayType>::try_from(&input_types[0])?;
        Self::validate_input_type(input_type)?;
        Ok(vec![self.result_type.clone().into()])
    }

    #[inline]
    fn effects(&self) -> Effects {
        Effects::single(Effect::OrderedAssertion)
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        Ok(Self { result_type: self.result_type.rename_identities(renaming)? })
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, DIMENSION_FROM_SCALAR_OPERATION_NAME)?
            .bracketed(|operation| operation.field("bounds", self.result_type.bounds()))
    }
}

impl<C: Domain<Type = ArrayIrType, Value: DimensionFromScalar<C::Value>>> InterpretableOperation<C>
    for DimensionFromScalarOperation
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        self.infer_output_types(&[inputs[0].r#type().into_owned()], &[])?;
        Ok(vec![inputs[0].to_dimension(self.result_type.variable().clone())?])
    }
}

impl<C: Context<Type = ArrayIrType, Operation: From<DimensionFromScalarOperation>>> PartiallyEvaluatableOperation<C>
    for DimensionFromScalarOperation
{
}

/// Batching converts a mapped scalar array into one checked extent per batch item. The extents remain ordinary packed
/// integer SSA data and are exposed as a mapped dimension only through [`ArrayIrBatch`]; no raggedness is added to
/// [`ArrayIrType`]. A carry-free scan applies this ordered-assertion gateway to every scalar and converts each checked
/// dimension back to scalar data for packing, so the gateway's bounds diagnostics remain exact.
impl<C> BatchableOperation<C, ArrayIrBatching> for DimensionFromScalarOperation
where
    C: Context<Type = ArrayIrType>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: crate::operations::Transpose + Value<Type = ArrayType>>,
    C::Operation: From<DimensionFromScalarOperation>
        + From<ConstantOperation<DimensionValue>>
        + From<DimensionSizeOperation>
        + From<DimensionToScalarOperation>
        + From<DynamicBroadcastOperation>
        + From<ScanOperation<C::Constant>>
        + OperationProjection<ArrayType>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<TransposeOperation>,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatching>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatching>,
        _driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatching>, BatchingError> {
        let [input] = inputs else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: inputs.len() }.into());
        };
        let unbatched_type = input.unbatched_type();
        let input_type = <&ArrayType>::try_from(&unbatched_type)?.clone();
        Self::validate_input_type(&input_type)?;
        if input.batch_axis().is_replicated() {
            return Ok(context
                .parent()
                .bind(self.clone(), Vec::new(), std::slice::from_ref(input.value()))?
                .into_iter()
                .map(ArrayIrBatch::replicated)
                .collect::<Vec<_>>()
                .into());
        }

        let input = align_array_batch(context, input.clone(), Axis::from(0))?;
        let scan_extent = array_dimension(context.parent(), input.value(), 0)?;
        let scan_extent_type = scan_extent.r#type();
        let length = <&DimensionType>::try_from(scan_extent_type.as_ref())?.to_dimension();
        let mut builder = ProgramBuilder::<C::Constant, C::Operation>::new();
        let scalar = builder.add_input(ArrayIrType::Array(input_type.clone()));
        let dimension = builder.add_instruction(self.clone(), Vec::new(), vec![scalar])?[0];
        let scalar = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![dimension])?[0];
        let body =
            builder.build::<Vec<C::Constant>, Vec<C::Constant>>(vec![scalar], vec![Placeholder], vec![Placeholder])?;

        let scan = ScanOperation::<C::Constant>::new(0, length.clone());
        let mut packed_inputs = vec![input.into_value()];
        if length.variable().is_some() {
            packed_inputs.push(scan_extent);
        }
        let mut outputs = context.parent().bind(scan, vec![body], packed_inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![ArrayIrBatch::mapped_dimension(
            outputs.remove(0),
            BatchAxis::from_position(0),
            self.result_type().clone(),
        )?]
        .into())
    }
}

impl_non_differentiable_operation!(DimensionFromScalarOperation);
impl_non_transposable_operation!(DimensionFromScalarOperation);

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, DataType, DimensionBounds, DimensionError,
        DimensionOperation, DimensionValue, MAX_DIMENSION_EXTENT, Shape,
    };
    use crate::contexts::{Context, EagerContext, StagingContext};
    use crate::differentiation::TransposableOperation;
    use crate::macros::check_operation_partial_evaluation;
    use crate::operations::dimensions::dimension_requirement::{
        DIMENSION_REQUIRE_BOUNDS_OPERATION_NAME, DimensionRequirementOperation,
    };
    use crate::operations::manipulation::broadcasting::DynamicBroadcastOperation;
    use crate::operations::math::sin::SinOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{Effects, EmptyRegionDriver, Program, ProgramBuilder, RegionInterface};
    use crate::tracing::TracingContext;

    use super::*;

    /// Returns the names of `program`'s ordered-assertion instructions, in program order.
    fn assertion_operation_names(
        program: &Program<
            ArrayIrValue<Array>,
            ArrayIrOperation<Array>,
            Vec<ArrayIrValue<Array>>,
            Vec<ArrayIrValue<Array>>,
        >,
    ) -> Vec<&'static str> {
        program
            .instructions()
            .iter()
            .filter(|instruction| instruction.operation().effects().contains(Effect::OrderedAssertion))
            .map(|instruction| instruction.operation().name())
            .collect()
    }

    #[test]
    fn test_dimension_from_scalar() {
        let bounds = DimensionBounds::new(0, Some(9)).unwrap();
        let variable = DimensionVariable::new("extent", bounds);
        let operation = DimensionFromScalarOperation::new(variable.clone());
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::I32));

        assert_eq!(operation.name(), DIMENSION_FROM_SCALAR_OPERATION_NAME);
        assert_eq!(operation.result_type(), &DimensionType::new(variable.clone()));
        assert_eq!(operation.to_string(), "dimension_from_scalar [bounds=[0, 9)]");
        assert_eq!(operation.effects(), Effects::single(Effect::OrderedAssertion));
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&scalar_type), &[]),
            Ok(vec![operation.result_type().clone().into()]),
        );
        for data_type in [
            DataType::I8,
            DataType::I16,
            DataType::I32,
            DataType::I64,
            DataType::U8,
            DataType::U16,
            DataType::U32,
            DataType::U64,
        ] {
            assert_eq!(
                operation.infer_output_types(&[ArrayType::scalar(data_type).into()], &[]),
                Ok(vec![operation.result_type().clone().into()]),
            );
        }
        assert_eq!(
            operation.infer_output_types(
                &[ArrayType::new(DataType::I32, Shape::new(vec![crate::arrays::Dimension::Static(1)])).into()],
                &[],
            ),
            Err(
                TypeError::invalid("`dimension_from_scalar` input must be a rank-0 integer array but has type i32[1]",)
            ),
        );
        assert_eq!(
            operation.infer_output_types(&[ArrayType::scalar(DataType::F32).into()], &[]),
            Err(TypeError::invalid("`dimension_from_scalar` input must be a rank-0 integer array but has type f32[]",)),
        );
        assert_eq!(
            operation.infer_output_types(&[operation.result_type().clone().into()], &[]),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        assert_eq!(operation.infer_output_types(&[], &[]), Err(TypeError::invalid("expected 1 input but got 0")));
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&scalar_type),
                &[RegionInterface::new(Vec::new(), Vec::new(), Effects::PURE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );

        let renamed_variable = DimensionVariable::new("renamed", bounds);
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(variable.clone(), renamed_variable.clone()).unwrap();
        let renamed_operation = operation.rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed_operation.result_type().variable(), &renamed_variable);
        assert_eq!(renamed_operation.result_type().bounds(), bounds);

        // Every integer element type uses the same checked conversion contract.
        for array in [
            Array::scalar(7_i8),
            Array::scalar(7_i16),
            Array::scalar(7_i32),
            Array::scalar(7_i64),
            Array::scalar(7_u8),
            Array::scalar(7_u16),
            Array::scalar(7_u32),
            Array::scalar(7_u64),
        ] {
            assert_eq!(array.to_dimension(variable.clone()).unwrap().extent(), 7);
        }
        assert_eq!(Array::scalar(0_i32).to_dimension(variable.clone()).unwrap().extent(), 0);
        let bounded_variable = DimensionVariable::new("bounded", DimensionBounds::new(2, Some(5)).unwrap());
        assert_eq!(Array::scalar(2_i32).to_dimension(bounded_variable.clone()).unwrap().extent(), 2);
        assert_eq!(Array::scalar(4_i32).to_dimension(bounded_variable.clone()).unwrap().extent(), 4);
        assert_eq!(
            Array::scalar(-1_i32).to_dimension(variable.clone()),
            Err(ProgramError::InvalidArgument {
                message: "`dimension_from_scalar` scalar input must be a nonnegative host-representable extent but is \
                          -1"
                .to_string(),
            }),
        );
        let error = Array::scalar(5_i32).to_dimension(bounded_variable.clone()).unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::BindingOutOfBounds {
                variable: "bounded".to_string(),
                value: 5,
                bounds: bounded_variable.bounds(),
            }),
        );
        if let Some(unrepresentable) = MAX_DIMENSION_EXTENT.checked_add(1) {
            let error = Array::scalar(u64::try_from(unrepresentable).unwrap())
                .to_dimension(DimensionVariable::new("wide", DimensionBounds::at_least(0)))
                .unwrap_err();
            assert_eq!(
                error.downcast_custom::<DimensionError>(),
                Some(&DimensionError::ExtentExceedsBackendWidth {
                    value: unrepresentable,
                    maximum: MAX_DIMENSION_EXTENT,
                }),
            );
        }

        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        assert_eq!(
            context.bind(operation.clone(), Vec::new(), &[ArrayIrValue::Array(Array::scalar(7_i32))],),
            Ok(vec![ArrayIrValue::Dimension(DimensionValue::new(operation.result_type().clone(), 7).unwrap(),)]),
        );
        assert_eq!(
            context.bind(
                operation.clone(),
                Vec::new(),
                &[ArrayIrValue::Dimension(DimensionValue::new(operation.result_type().clone(), 7).unwrap(),)],
            ),
            Err(TypeError::invalid("expected array type but got dimension type").into()),
        );
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = operation.clone(),
            cases = [
                {
                    inputs = [
                        (@known, ArrayIrValue::Array(Array::scalar(7_i32))),
                    ],
                    outputs = [
                        (@known, ArrayIrValue::Dimension(
                            DimensionValue::new(operation.result_type().clone(), 7).unwrap()
                        )),
                    ],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(
                            type = scalar_type.clone(),
                            replay = ArrayIrValue::Array(Array::scalar(7_i32))
                        )),
                    ],
                    outputs = [
                        (@residual, ArrayIrValue::Dimension(
                            DimensionValue::new(operation.result_type().clone(), 7).unwrap()
                        )),
                    ],
                    residual_instructions = 1,
                },
            ],
        );

        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let context = TestContext::new();
        let input = context.input(scalar_type);
        let input_id = input.atom_id().unwrap();
        let output = input.to_dimension(variable.clone()).unwrap();
        let output_id = output.atom_id().unwrap();
        let builder = context.builder().borrow();
        let [instruction] = builder.instructions() else {
            panic!("expected one dimension-from-scalar instruction");
        };
        assert_eq!(instruction.inputs(), &[input_id]);
        assert_eq!(instruction.outputs(), &[output_id]);
        assert!(instruction.regions().is_empty());
        assert!(matches!(instruction.operation(), ArrayIrOperation::DimensionFromScalar(_)));
        assert_eq!(output.r#type().as_ref(), &ArrayIrType::Dimension(DimensionType::new(variable.clone())));

        let program = builder
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output_id],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        drop(builder);
        assert_eq!(program.type_identity_signature().internal_identities(), std::slice::from_ref(&variable));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:i32[] .
                let %1:dimension<extent ∈ [0, 9)> = dimension_from_scalar [bounds=[0, 9)] %0
                in (%1)"},
        );

        let mut relocated_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let relocated_input = relocated_builder.add_input(ArrayType::scalar(DataType::I32).into());
        let relocated_outputs = relocated_builder.splice_program(&program, &[relocated_input]).unwrap();
        let [relocated_instruction] = relocated_builder.instructions() else {
            panic!("expected one relocated dimension-from-scalar instruction");
        };
        assert_eq!(relocated_instruction.inputs(), &[relocated_input]);
        assert_eq!(relocated_instruction.outputs(), relocated_outputs.as_slice());
        let ArrayIrOperation::DimensionFromScalar(relocated_operation) = relocated_instruction.operation() else {
            panic!("expected a relocated dimension-from-scalar operation");
        };
        assert_ne!(relocated_operation.result_type().variable(), &variable);
        assert_eq!(relocated_operation.result_type().variable().name(), variable.name());
        assert_eq!(relocated_operation.result_type().bounds(), bounds);

        // The relocated identity is nominally fresh but alpha-equivalent, so the relocated program renders exactly
        // like its source.
        let relocated = relocated_builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                relocated_outputs,
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(relocated.to_string(), program.to_string());

        let projected_context = TestContext::new();
        let projected_input = projected_context.input(ArrayType::scalar(DataType::I32).into());
        let projected_input =
            <crate::Tracer<TestContext> as crate::ValueProjection<ArrayType>>::into_projected(projected_input).unwrap();
        let projected_variable = DimensionVariable::new("projected", bounds);
        let projected_output = projected_input.to_dimension(projected_variable.clone()).unwrap();
        assert_eq!(projected_output.r#type().as_ref(), &ArrayIrType::Dimension(DimensionType::new(projected_variable)),);
        assert!(matches!(
            projected_context.builder().borrow().instructions(),
            [instruction] if matches!(instruction.operation(), ArrayIrOperation::DimensionFromScalar(_)),
        ));

        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_ids().len(), 1);
        assert_eq!(jvp.output_ids().len(), 1);

        let mut transposition_context = TestContext::new();
        assert!(matches!(
            <DimensionFromScalarOperation as TransposableOperation<
                ArrayIrValue<Array>,
                ArrayIrOperation<Array>,
            >>::transpose(
                &operation,
                &mut transposition_context,
                &EmptyRegionDriver,
                &[],
                &[],
            ),
            Err(crate::DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `dimension_from_scalar` is not transposable",
        ));
    }

    #[test]
    fn test_dimension_from_scalar_ordered_assertions_survive_differentiation() {
        // A tier-3 composite program whose differentiable array body is shaped by a data-derived dimension. The
        // gateway always carries an ordered assertion because its bounds check can only run against runtime data, and
        // the requirement is inconclusive from the declared `[1, 9)` bounds alone, so it residualizes as a second
        // ordered assertion. Their relative order defines which failure is observed first and must never change.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let value = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let extent_scalar = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let extent = builder
            .add_instruction(DimensionFromScalarOperation::new(rows.clone()), Vec::new(), vec![extent_scalar])
            .unwrap()[0];
        builder
            .add_instruction(
                DimensionOperation::Requirement(DimensionRequirementOperation::bounds(
                    &DimensionType::new(rows),
                    DimensionBounds::new(2, Some(8)).unwrap(),
                )),
                Vec::new(),
                vec![extent],
            )
            .unwrap();
        let broadcast = builder
            .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![value, extent])
            .unwrap()[0];
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Sin(SinOperation::new())),
                Vec::new(),
                vec![broadcast],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(program.effects(), Effects::single(Effect::OrderedAssertion));
        assert_eq!(
            assertion_operation_names(&program),
            vec![DIMENSION_FROM_SCALAR_OPERATION_NAME, DIMENSION_REQUIRE_BOUNDS_OPERATION_NAME],
        );

        // Forward differentiation stages both assertions once, in their original relative order: a dimension carries
        // no tangent, so neither assertion may be duplicated into a tangent computation or reordered against the
        // other by the interleaved dual program.
        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.effects(), Effects::single(Effect::OrderedAssertion));
        assert_eq!(
            assertion_operation_names(&jvp),
            vec![DIMENSION_FROM_SCALAR_OPERATION_NAME, DIMENSION_REQUIRE_BOUNDS_OPERATION_NAME],
        );

        // Linearization splits the dual program by known-ness. Both assertions are nonlinear primal work, so they
        // stay in the primal sub-program in their original order, and the compact linear tangent sub-program is left
        // pure: it consumes the checked extent as an ordinary residual instead of re-asserting it.
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.primal().effects(), Effects::single(Effect::OrderedAssertion));
        assert_eq!(
            assertion_operation_names(linearization.primal()),
            vec![DIMENSION_FROM_SCALAR_OPERATION_NAME, DIMENSION_REQUIRE_BOUNDS_OPERATION_NAME],
        );
        assert_eq!(linearization.tangent().effects(), Effects::PURE);
        assert_eq!(assertion_operation_names(linearization.tangent()), Vec::<&str>::new());
    }

    #[test]
    fn test_dimension_from_scalar_extent_controls_multiple_bounded_outputs() {
        // A data-dependent extent is an ordinary SSA value with one definition. Two sibling shape-carrying
        // constructors consume that one gateway result as an explicit operand instead of each recovering an extent of
        // its own, so both outputs are governed by a single identity and a single bounds assertion.
        let total = DimensionVariable::new("total", DimensionBounds::new(1, Some(9)).unwrap());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let extent_scalar = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let first_value = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let second_value = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let extent = builder
            .add_instruction(DimensionFromScalarOperation::new(total), Vec::new(), vec![extent_scalar])
            .unwrap()[0];
        let first = builder
            .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![first_value, extent])
            .unwrap()[0];
        let second = builder
            .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![second_value, extent])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![first, second],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:i32[], %1:f64[], %2:f64[] .
                let %3:dimension<total ∈ [1, 9)> = dimension_from_scalar [bounds=[1, 9)] %0
                    %4:f64[total] = broadcast [output_axes=[]] %1 %3
                    %5:f64[total] = broadcast [output_axes=[]] %2 %3
                in (%4, %5)"},
        );
        let output_types = program.output_types();
        assert_eq!(output_types[0], output_types[1]);

        // Both outputs follow the same runtime extent across every value the declared bounds admit.
        for extent in [1_usize, 5, 8] {
            assert_eq!(
                program.interpret(vec![
                    ArrayIrValue::Array(Array::scalar(i32::try_from(extent).unwrap())),
                    ArrayIrValue::Array(Array::scalar(2.0_f64)),
                    ArrayIrValue::Array(Array::scalar(3.0_f64)),
                ]),
                Ok(vec![
                    ArrayIrValue::Array(Array::vector(vec![2.0_f64; extent])),
                    ArrayIrValue::Array(Array::vector(vec![3.0_f64; extent])),
                ]),
            );
        }
    }
}
