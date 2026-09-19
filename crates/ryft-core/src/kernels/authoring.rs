//! Fallible authoring conveniences over canonical array operations.
//!
//! These functions bind through the value's existing context. They add no operation semantics or host evaluation
//! path; explicit builders, generated authoring code, and ordinary program replay see the same operation payloads.

use crate::arrays::{
    Array, ArrayElement, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DimensionBounds,
    DimensionType, DimensionValue, DimensionVariable,
};
use crate::contexts::{Context, Domain, StagingContext};
use crate::kernels::calls::{KernelCallOperation, KernelDefinition, KernelError, KernelParameter};
use crate::kernels::grids::{Grid, GridDimension, GridExecution};
use crate::kernels::indexing::TileLoadOperation;
use crate::kernels::interpretation::DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS;
use crate::kernels::mappings::{BlockMapping, BoundaryPolicy};
use crate::kernels::operations::KernelOperation;
use crate::kernels::validation::KernelParameterAccess;
use crate::operations::{
    AddOperation, CompareOperation, ComparisonDirection, ConditionOperation, DimensionFromScalarOperation,
    DimensionMulOperation, DotOperation, ReferenceWriteOperation, WhileOperation, ZeroOperation,
};
use crate::parameters::Placeholder;
use crate::programs::{ProgramBuilder, ProgramError, ProjectedValue, TypeError, Typed, Value};
use crate::tracing::{NestedTracingContext, Tracer};

/// Functional kernel invocation capability for array values. Concrete host arrays use the qualified reference
/// interpreter; projected symbolic arrays bind the same call and immutable body into their existing context.
/// Backend array families may implement this capability through their own compilation domain and selected adapter.
/// A definition remains independently inspectable and compilable through [`KernelDefinition`].
pub trait KernelCall: Clone + Typed<Type = ArrayType> {
    /// Invokes or stages one portable definition with its ordinary array inputs and results.
    fn call_kernel(definition: &KernelDefinition, inputs: &[Self]) -> Result<Vec<Self>, ProgramError>;
}

impl KernelCall for Array {
    fn call_kernel(definition: &KernelDefinition, inputs: &[Self]) -> Result<Vec<Self>, ProgramError> {
        definition.interpret(inputs.to_vec(), DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS)
    }
}

impl<V: Value<Type = ArrayIrType>> KernelCall for ProjectedValue<ArrayType, V>
where
    V::DispatchDomain: Context<Constant = ArrayIrValue<Array>, Operation = KernelOperation>,
{
    fn call_kernel(definition: &KernelDefinition, inputs: &[Self]) -> Result<Vec<Self>, ProgramError> {
        let first = inputs.first().ok_or_else(|| {
            ProgramError::Type(TypeError::invalid(
                "symbolic zero-input kernel calls require an explicit context and definition binding",
            ))
        })?;
        first
            .value()
            .dispatch_domain()
            .bind(
                definition.operation().clone(),
                vec![definition.body().clone()],
                &inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>(),
            )?
            .into_iter()
            .map(|value| {
                let r#type = value.r#type().into_owned();
                let ArrayIrType::Array(r#type) = r#type else {
                    return Err(ProgramError::Type(TypeError::invalid("kernel call returned a non-array result")));
                };
                Ok(ProjectedValue::new(value, r#type))
            })
            .collect()
    }
}

/// Returns one static full-array extent for generated shape metadata after checking the rank and static shape.
/// This checked helper prevents indexing panics in an annotated kernel's metadata expressions.
pub fn static_extent(r#type: &ArrayType, axis: usize) -> Result<usize, KernelError> {
    let shape = r#type.static_shape().ok_or_else(|| {
        KernelError::Type(TypeError::invalid("kernel shape metadata requires a static full-array shape"))
    })?;
    shape
        .dimensions()
        .get(axis)
        .copied()
        .ok_or_else(|| KernelError::Type(TypeError::invalid("kernel shape metadata axis exceeds the input rank")))
}

/// Declares one whole-array reference window on a rank-zero logical grid. Generated tiled definitions use explicit
/// mappings instead; this helper does not infer a tiled partition or bypass canonical slice validation.
pub fn whole_array_parameter(r#type: ArrayType, access: KernelParameterAccess) -> Result<KernelParameter, KernelError> {
    let shape = r#type
        .static_shape()
        .ok_or_else(|| KernelError::Type(TypeError::invalid("whole-array kernel parameters require a static shape")))?;
    let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
    let outputs = if r#type.rank() == 0 {
        vec![]
    } else {
        let zero =
            builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).map_err(ProgramError::from)?));
        vec![zero; r#type.rank()]
    };
    let program = builder.build(outputs, vec![], vec![Placeholder; r#type.rank()])?;
    KernelParameter::new(
        r#type,
        access,
        BlockMapping::new(program, shape.dimensions().to_vec(), BoundaryPolicy::InBounds)?,
    )
}

/// Constructs a parallel output-tile grid with whole-array read-only inputs and disjoint mapped output windows.
/// Every mapping consumes the same canonical grid-coordinate signature. Input mappings ignore those coordinates;
/// their full references remain available for dynamically selected loads inside the body.
pub fn tiled_call(
    input_types: &[ArrayType],
    output_type: ArrayType,
    block_shape: Vec<usize>,
    boundary: BoundaryPolicy,
) -> Result<KernelCallOperation, KernelError> {
    let output_shape = output_type
        .static_shape()
        .ok_or_else(|| TypeError::invalid("kernel output tiling requires a static shape"))?;
    if output_type.rank() != block_shape.len() {
        return Err(TypeError::invalid("kernel output tile rank must equal its array rank").into());
    }
    let extents = output_shape
        .dimensions()
        .iter()
        .zip(&block_shape)
        .map(|(&extent, &block)| shape_div_ceil(extent, block))
        .collect::<Result<Vec<_>, _>>()?;
    let grid =
        Grid::new(extents.iter().map(|&extent| GridDimension::new(extent.into(), GridExecution::Parallel)).collect())?;
    let coordinate_types = extents
        .iter()
        .enumerate()
        .map(|(axis, &extent)| {
            let bounds = DimensionBounds::new(0, Some(extent.max(1))).map_err(ProgramError::from)?;
            Ok(ArrayIrType::Dimension(DimensionType::new(format!("tile_axis_{axis}"), bounds)))
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;
    let mut parameters = Vec::with_capacity(input_types.len() + 1);
    for r#type in input_types.iter().chain(std::iter::once(&output_type)) {
        let is_output = parameters.len() == input_types.len();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let coordinates = coordinate_types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let shape = r#type
            .static_shape()
            .ok_or_else(|| TypeError::invalid("kernel input tiling requires a static shape"))?;
        let mut starts = Vec::with_capacity(r#type.rank());
        for axis in 0..r#type.rank() {
            let start = if is_output {
                let scale_value = DimensionValue::constant(block_shape[axis]).map_err(ProgramError::from)?;
                let ArrayIrType::Dimension(coordinate_type) = &coordinate_types[axis] else { unreachable!() };
                let operation = DimensionMulOperation::new(coordinate_type, scale_value.r#type().as_ref())
                    .map_err(ProgramError::from)?;
                let scale = builder.add_constant(ArrayIrValue::Dimension(scale_value));
                builder.add_instruction(
                    ArrayIrOperation::from(operation),
                    vec![],
                    vec![coordinates[axis], scale],
                    None,
                )?[0]
            } else {
                builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).map_err(ProgramError::from)?))
            };
            starts.push(start);
        }
        let program =
            builder.build(starts, vec![Placeholder; coordinate_types.len()], vec![Placeholder; r#type.rank()])?;
        parameters.push(KernelParameter::new(
            r#type.clone(),
            if is_output { KernelParameterAccess::WriteOnly } else { KernelParameterAccess::ReadOnly },
            BlockMapping::new(
                program,
                if is_output { block_shape.clone() } else { shape.dimensions().to_vec() },
                if is_output { boundary } else { BoundaryPolicy::InBounds },
            )?,
        )?);
    }
    KernelCallOperation::new(grid, parameters)
}

/// Stages a reference tile load, translating tile indices into checked element starts. Integer-array indices enter
/// the existing dimension universe through its checked scalar gateway; dimension-valued grid indices are reused.
pub fn tile_load<
    T: ArrayElement,
    C: Context<Type = ArrayIrType, Constant = ArrayIrValue<Array>, Operation = KernelOperation>,
>(
    context: &C,
    source: &C::Value,
    block_shape: Vec<usize>,
    indices: &[C::Value],
    other: T,
) -> Result<C::Value, ProgramError> {
    let source_type = source.r#type();
    let ArrayIrType::Reference(reference) = source_type.as_ref() else {
        return Err(TypeError::invalid("kernel tile source must be a reference").into());
    };
    let shape = reference
        .referent()
        .static_shape()
        .ok_or_else(|| TypeError::invalid("kernel tile source requires a static shape"))?;
    if indices.len() != block_shape.len() || indices.len() != shape.rank() {
        return Err(TypeError::invalid("kernel tile indices and block shape must match the source rank").into());
    }
    let mut inputs = vec![source.clone()];
    for (axis, index) in indices.iter().enumerate() {
        let dimension = match index.r#type().as_ref() {
            ArrayIrType::Dimension(_) => index.clone(),
            ArrayIrType::Array(_) => {
                let upper =
                    shape_div_ceil(shape.dimensions()[axis], block_shape[axis]).map_err(ProgramError::custom)?.max(1);
                let variable =
                    DimensionVariable::new(format!("tile_index_{axis}"), DimensionBounds::new(0, Some(upper))?);
                context
                    .bind(
                        ArrayIrOperation::DimensionFromScalar(DimensionFromScalarOperation::new(variable)),
                        vec![],
                        &[index.clone()],
                    )?
                    .remove(0)
            }
            _ => {
                return Err(TypeError::invalid("kernel tile index must be a dimension or integer scalar array").into());
            }
        };
        let scale_value = DimensionValue::constant(block_shape[axis])?;
        let dimension_type = dimension.r#type();
        let ArrayIrType::Dimension(dimension_type) = dimension_type.as_ref() else { unreachable!() };
        let operation = DimensionMulOperation::new(dimension_type, scale_value.r#type().as_ref())?;
        let scale = context.lift(ArrayIrValue::Dimension(scale_value))?;
        inputs.push(context.bind(ArrayIrOperation::from(operation), vec![], &[dimension, scale])?.remove(0));
    }
    inputs.push(context.lift(ArrayIrValue::Array(Array::scalar(other)?))?);
    Ok(context
        .bind(TileLoadOperation::new(block_shape, BoundaryPolicy::Masked)?, vec![], &inputs)?
        .remove(0))
}

/// Stores a complete tile into a private mapped output window through the canonical reference write. Every private
/// tile element is valid storage; the qualified invocation publishes only the intersection with the full output
/// shape. This boundary clips publication without requiring an allocated mask or weakening initialization checks.
pub fn tile_store<C: Context<Type = ArrayIrType, Constant = ArrayIrValue<Array>, Operation = KernelOperation>>(
    context: &C,
    output: &C::Value,
    value: &C::Value,
) -> Result<(), ProgramError> {
    context.bind(
        ArrayIrOperation::ReferenceWrite(ReferenceWriteOperation::new()),
        vec![],
        &[output.clone(), value.clone()],
    )?;
    Ok(())
}

/// Stages a bounded range loop with explicit carried values through the canonical while operation. The body is
/// traced exactly once even for an empty range. Its index is an ordinary signed 64-bit scalar array; callers use
/// canonical scalar-to-dimension conversion when an indexing operation needs a dimension value.
pub fn for_loop<C, F>(
    context: &C,
    range: std::ops::Range<usize>,
    inputs: Vec<C::Value>,
    function: F,
) -> Result<Vec<C::Value>, ProgramError>
where
    C: StagingContext<Type = ArrayIrType, Constant = ArrayIrValue<Array>, Operation = KernelOperation>,
    F: FnOnce(
        Tracer<NestedTracingContext<C>>,
        Vec<Tracer<NestedTracingContext<C>>>,
    ) -> Result<Vec<Tracer<NestedTracingContext<C>>>, ProgramError>,
{
    let start = i64::try_from(range.start)
        .map_err(|_| TypeError::invalid("kernel range start exceeds the signed 64-bit index domain"))?;
    let end = i64::try_from(range.end)
        .map_err(|_| TypeError::invalid("kernel range end exceeds the signed 64-bit index domain"))?;
    let index = context.lift(ArrayIrValue::Array(Array::scalar(start)?))?;
    let mut state = vec![index];
    state.extend(inputs);
    let types = state.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
    let (_, condition) = NestedTracingContext::trace(
        context.clone(),
        |state: Vec<_>| {
            let context = state[0].context();
            let end = context.lift(ArrayIrValue::Array(Array::scalar(end)?))?;
            context.bind(
                ArrayIrOperation::from(ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::LessThan))),
                vec![],
                &[state[0].clone(), end],
            )
        },
        types.clone(),
    )?;
    let (_, body) = NestedTracingContext::trace(
        context.clone(),
        |mut state: Vec<_>| {
            let index = state.remove(0);
            let values = function(index.clone(), state)?;
            let context = index.context();
            let one = context.lift(ArrayIrValue::Array(Array::scalar(1i64)?))?;
            let mut next = context.bind(
                ArrayIrOperation::from(ArrayOperation::Add(AddOperation::new())),
                vec![],
                &[index.clone(), one],
            )?;
            next.extend(values);
            Ok(next)
        },
        types,
    )?;
    let mut outputs = context.bind(
        ArrayIrOperation::While(
            WhileOperation::new().with_iteration_bound(range.end.saturating_sub(range.start).max(1))?,
        ),
        vec![condition, body],
        &state,
    )?;
    outputs.remove(0);
    Ok(outputs)
}

/// Stages both branches of a value-dependent condition with explicit carried values. Branches must return the same
/// types and preserve reference identities according to the canonical conditional operation's contract.
pub fn condition<C, Then, Else>(
    context: &C,
    predicate: &C::Value,
    inputs: Vec<C::Value>,
    then_function: Then,
    else_function: Else,
) -> Result<Vec<C::Value>, ProgramError>
where
    C: StagingContext<Type = ArrayIrType, Constant = ArrayIrValue<Array>, Operation = KernelOperation>,
    Then: FnOnce(Vec<Tracer<NestedTracingContext<C>>>) -> Result<Vec<Tracer<NestedTracingContext<C>>>, ProgramError>,
    Else: FnOnce(Vec<Tracer<NestedTracingContext<C>>>) -> Result<Vec<Tracer<NestedTracingContext<C>>>, ProgramError>,
{
    let types = inputs.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
    let (_, then_program) = NestedTracingContext::trace(context.clone(), then_function, types.clone())?;
    let (_, else_program) = NestedTracingContext::trace(context.clone(), else_function, types)?;
    let mut arguments = vec![predicate.clone()];
    arguments.extend(inputs);
    context.bind(ArrayIrOperation::Condition(ConditionOperation::new()), vec![then_program, else_program], &arguments)
}

/// Checked ceiling division used only for static kernel shape metadata.
pub fn shape_div_ceil(extent: usize, divisor: usize) -> Result<usize, KernelError> {
    if divisor == 0 {
        return Err(KernelError::Type(TypeError::invalid("kernel shape ceiling division requires a positive divisor")));
    }
    Ok(extent.div_ceil(divisor))
}

/// Constructs a statically shaped zero tile with the canonical element type of `T`. The active context records the
/// ordinary [`ZeroOperation`]; no host tile is allocated while tracing. Shape and placement validation remain with
/// the canonical array type and operation.
pub fn zeros<T: ArrayElement, C: Context<Type = ArrayIrType>>(
    context: &C,
    shape: impl Into<Vec<usize>>,
) -> Result<C::Value, ProgramError>
where
    C::Operation: From<ArrayIrOperation<Array>>,
{
    let operation = ArrayIrOperation::<Array>::from(ArrayOperation::Zero(ZeroOperation::new(ArrayType::new_static(
        T::data_type(),
        shape,
    ))));
    Ok(context.bind(operation, Vec::new(), &[])?.into_iter().next().unwrap())
}

/// Multiplies two rank-two tiles through [`DotOperation::matmul`], retaining its dtype, contraction, accumulation,
/// sharding, and numerical contracts. Incompatible tiles return the existing type diagnostic rather than panicking.
pub fn dot<V: Value<Type = ArrayIrType>>(left: &V, right: &V) -> Result<V, ProgramError>
where
    V::DispatchDomain: Context,
    <V::DispatchDomain as Domain>::Operation: From<ArrayIrOperation<Array>>,
{
    let operation = ArrayIrOperation::<Array>::from(ArrayOperation::Dot(DotOperation::matmul()));
    Ok(left
        .dispatch_domain()
        .bind(operation, Vec::new(), &[left.clone(), right.clone()])?
        .into_iter()
        .next()
        .unwrap())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{ArrayReference, DataType};
    use crate::kernels::calls::KernelCallOperation;
    use crate::kernels::grids::Grid;
    use crate::operations::{NegOperation, ReferenceRead, ReferenceWrite};
    use crate::programs::{Operation, ReferenceType};
    use crate::tracing::TracingContext;

    use super::*;

    /// Whole-array window for a single-program arithmetic fixture.
    fn parameter(shape: Vec<usize>, access: KernelParameterAccess) -> KernelParameter {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let zero = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()));
        let rank = shape.len();
        KernelParameter::new(
            ArrayType::new_static(DataType::F32, shape.clone()),
            access,
            BlockMapping::new(
                builder.build(vec![zero; rank], vec![], vec![Placeholder; rank]).unwrap(),
                shape,
                BoundaryPolicy::InBounds,
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn test_static_extent() {
        let r#type = ArrayType::new_static(DataType::F32, [2, 3]);
        assert_eq!(static_extent(&r#type, 1), Ok(3));
        assert_eq!(
            static_extent(&r#type, 2),
            Err(KernelError::Type(TypeError::invalid("kernel shape metadata axis exceeds the input rank",)))
        );
    }

    #[test]
    fn test_whole_array_parameter() {
        let r#type = ArrayType::new_static(DataType::F32, [2, 3]);
        let parameter = whole_array_parameter(r#type.clone(), KernelParameterAccess::ReadOnly).unwrap();
        assert_eq!(parameter.r#type().as_ref(), &r#type);
        assert_eq!(parameter.mapping().block_shape(), &[2, 3]);
        assert_eq!(parameter.mapping().evaluate(&[], &[2, 3]).unwrap().starts(), &[0, 0]);
    }

    #[test]
    fn test_tiled_call() {
        let operation = tiled_call(
            &[ArrayType::new_static(DataType::F32, [3, 5])],
            ArrayType::new_static(DataType::F32, [3, 7]),
            vec![2, 4],
            BoundaryPolicy::Masked,
        )
        .unwrap();
        assert_eq!(operation.grid().points(&[2, 2]).unwrap().len(), 4);
        assert_eq!(operation.parameters()[0].mapping().block_shape(), &[3, 5]);
        assert_eq!(operation.parameters()[1].mapping().block_shape(), &[2, 4]);
        assert_eq!(operation.coordinate_types().len(), 2);
        assert_eq!(
            tiled_call(&[], ArrayType::scalar(DataType::F32), vec![2], BoundaryPolicy::Masked).unwrap_err(),
            KernelError::Type(TypeError::invalid("kernel output tile rank must equal its array rank"))
        );
    }

    #[test]
    fn test_tile_load() {
        let (_, program) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<_>| {
                let context = inputs[0].context();
                let row = context.lift(ArrayIrValue::Dimension(DimensionValue::constant(0)?))?;
                let column = context.lift(ArrayIrValue::Array(Array::scalar(1i64)?))?;
                Ok(vec![tile_load(context, &inputs[0], vec![2, 2], &[row, column], -1i32)?])
            },
            vec![ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::I32, [3, 5])))],
        )
        .unwrap();
        let source = ArrayReference::new(
            Array::from_elements(ArrayType::new_static(DataType::I32, [3, 5]), &(0..15i32).collect::<Vec<_>>())
                .unwrap(),
        );
        assert_eq!(
            program.interpret(vec![ArrayIrValue::Reference(source)]),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::I32, [2, 2]), &[2i32, 3, 7, 8]).unwrap()
            )])
        );
    }

    #[test]
    fn test_tile_store() {
        let operation =
            tiled_call(&[], ArrayType::new_static(DataType::I32, [3]), vec![2], BoundaryPolicy::Masked).unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            let context = references[0].context();
            let value = zeros::<i32, _>(context, [2])?;
            tile_store(context, &references[0], &value)
        })
        .unwrap();
        assert_eq!(definition.interpret(vec![], 2), Ok(vec![Array::vector(vec![0i32, 0, 0]).unwrap()]));
    }

    #[test]
    fn test_for_loop() {
        let calls = std::cell::Cell::new(0);
        let (_, program) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<_>| {
                let context = inputs[0].context().clone();
                for_loop(&context, 0..3, inputs, |index, inputs| {
                    calls.set(calls.get() + 1);
                    index.context().bind(
                        ArrayIrOperation::from(ArrayOperation::Add(AddOperation::new())),
                        vec![],
                        &[inputs[0].clone(), index.clone()],
                    )
                })
            },
            vec![ArrayIrType::Array(ArrayType::scalar(DataType::I64))],
        )
        .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(program.instructions()[0].operation().name(), "while");
        assert_eq!(
            program.interpret(vec![ArrayIrValue::Array(Array::scalar(5i64).unwrap())]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(8i64).unwrap())])
        );
    }

    #[test]
    fn test_for_loop_empty_range_still_traces_once() {
        let calls = std::cell::Cell::new(0);
        let (_, program) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<_>| {
                let context = inputs[0].context().clone();
                for_loop(&context, 0..0, inputs, |_index, inputs| {
                    calls.set(calls.get() + 1);
                    Ok(inputs)
                })
            },
            vec![ArrayIrType::Array(ArrayType::scalar(DataType::I64))],
        )
        .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(program.instructions()[0].operation().name(), "while");
        assert_eq!(
            program.interpret(vec![ArrayIrValue::Array(Array::scalar(5i64).unwrap())]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(5i64).unwrap())])
        );
    }

    #[test]
    fn test_condition() {
        let then_calls = std::cell::Cell::new(0);
        let else_calls = std::cell::Cell::new(0);
        let (_, program) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<_>| {
                let context = inputs[0].context();
                condition(
                    context,
                    &inputs[0],
                    vec![inputs[1].clone()],
                    |inputs| {
                        then_calls.set(then_calls.get() + 1);
                        inputs[0].context().bind(
                            ArrayIrOperation::from(ArrayOperation::Add(AddOperation::new())),
                            vec![],
                            &[inputs[0].clone(), inputs[0].clone()],
                        )
                    },
                    |inputs| {
                        else_calls.set(else_calls.get() + 1);
                        inputs[0].context().bind(
                            ArrayIrOperation::from(ArrayOperation::Neg(NegOperation::new())),
                            vec![],
                            &inputs,
                        )
                    },
                )
            },
            vec![
                ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)),
                ArrayIrType::Array(ArrayType::scalar(DataType::I64)),
            ],
        )
        .unwrap();
        assert_eq!((then_calls.get(), else_calls.get()), (1, 1));
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Array(Array::scalar(true).unwrap()),
                ArrayIrValue::Array(Array::scalar(3i64).unwrap())
            ]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(6i64).unwrap())])
        );
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Array(Array::scalar(false).unwrap()),
                ArrayIrValue::Array(Array::scalar(3i64).unwrap())
            ]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(-3i64).unwrap())])
        );
    }

    #[test]
    fn test_shape_div_ceil() {
        assert_eq!(shape_div_ceil(33, 32), Ok(2));
        assert_eq!(shape_div_ceil(0, 32), Ok(0));
        assert_eq!(
            shape_div_ceil(1, 0),
            Err(KernelError::Type(TypeError::invalid("kernel shape ceiling division requires a positive divisor",)))
        );
    }

    #[test]
    fn test_zeros() {
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![parameter(vec![2, 3], KernelParameterAccess::WriteOnly)],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            let value = zeros::<f32, _>(references[0].context(), [2, 3])?;
            references[0].write(&value)
        })
        .unwrap();
        assert_eq!(definition.body().instructions()[0].operation().name(), "zero");
        assert_eq!(
            definition.interpret(vec![], 1).unwrap(),
            vec![Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[0.0f32; 6]).unwrap()],
        );
    }

    #[test]
    fn test_dot() {
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                parameter(vec![2, 3], KernelParameterAccess::ReadOnly),
                parameter(vec![3, 2], KernelParameterAccess::ReadOnly),
                parameter(vec![2, 2], KernelParameterAccess::WriteOnly),
            ],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            let result = dot(&references[0].read()?, &references[1].read()?)?;
            references[2].write(&result)
        })
        .unwrap();
        let left =
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let right =
            Array::from_elements(ArrayType::new_static(DataType::F32, [3, 2]), &[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0])
                .unwrap();
        assert_eq!(
            definition.interpret(vec![left, right], 1).unwrap(),
            vec![
                Array::from_elements(ArrayType::new_static(DataType::F32, [2, 2]), &[58.0f32, 64.0, 139.0, 154.0])
                    .unwrap()
            ],
        );
    }

    #[test]
    fn test_dot_invalid_contraction() {
        let input_types = vec![
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2, 3])),
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4, 2])),
        ];
        let expected = ArrayIrOperation::<Array>::from(ArrayOperation::Dot(DotOperation::matmul()))
            .infer_output_types(&input_types, &[])
            .unwrap_err();
        let result = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<_>| Ok(vec![dot(&inputs[0], &inputs[1])?]),
            input_types,
        );
        assert_eq!(result.map(|_| ()), Err(ProgramError::Type(expected)));
    }
}
