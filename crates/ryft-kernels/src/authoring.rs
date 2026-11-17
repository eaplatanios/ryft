//! Authoring functions that build kernel calls and stage operations into kernel bodies.
//!
//! The [`kernel`](crate::kernel) attribute macro expands into calls of these functions, and explicit builders that use
//! [`KernelDefinition::trace`] can call them directly. Each function binds its operations through the context of its
//! values and adds no operation semantics or host evaluation path of its own, so explicit builders, generated code, and
//! ordinary program replay all see the same operation payloads.
//!
//!   - [`KernelCall`] invokes a [`KernelDefinition`], which interprets concrete host arrays and binds a
//!     [`KernelCallOperation`] into the context of staged values.
//!   - [`mapped_call`] constructs a [`KernelCallOperation`] from a [`Grid`], one [`ParameterWindow`] per parameter, and
//!     the types of its scalar prefetch values. A window is the whole array or a block whose starts are
//!     [`MappingExpression`]s over the grid coordinates and the prefetched dimensions, which this function lowers into
//!     the [`BlockMapping`] of the parameter. [`whole_array_parameter`] declares a single [`KernelParameter`] whose
//!     window is its whole array, for a call with no grid axes and no scalar prefetch values.
//!   - [`tile_load`] and [`tile_store`] load a tile of a reference with a [`TileLoadOperation`] and store a tile with a
//!     [`ReferenceWriteOperation`].
//!   - [`masked_load`], [`masked_load_or`], [`masked_store`], [`masked_swap`], and [`window_mask`] stage the masked
//!     accesses of [`memory`](crate::memory) and the validity mask of a window. [`accumulate`] adds a value into a
//!     window with a [`ReferenceAddUpdateOperation`], or with a masked load, an addition, and a masked store when the
//!     window is masked. [`scratch`], [`async_copy`], and [`wait`] allocate scratch storage and copy between
//!     references asynchronously, and [`reference_slice`] selects a static sub-window of a reference as a
//!     [`ReferenceView`].
//!   - [`compare`] and [`select`] stage a [`CompareOperation`] and a [`SelectOperation`], and [`compare_dimension`]
//!     compares a dimension (e.g., a grid coordinate) with a constant, in the form that the first step initialization
//!     rule of [`validate_kernel_initialization`](crate::initialization::validate_kernel_initialization) recognizes.
//!   - [`for_loop`] stages a bounded range loop as a [`WhileOperation`].
//!   - [`zeros`], [`ones`], [`full`], and [`iota`] stage constant tiles, and [`dot`] stages a matrix multiplication
//!     [`DotOperation`].
//!
//! Operations that `ryft-core` already offers as capabilities of kernel body values are not repeated here. Bodies read
//! and write references with [`ReferenceRead`](ryft_core::ReferenceRead) and
//! [`ReferenceWrite`](ryft_core::ReferenceWrite), accumulate atomically with
//! [`ReferenceAtomicAddUpdate`](ryft_core::ReferenceAtomicAddUpdate), convert element types with
//! [`ConvertElementType`](ryft_core::ConvertElementType), and branch with [`Condition`](ryft_core::Condition). The
//! [`Compare`](ryft_core::Compare) and [`Select`](ryft_core::Select) capabilities are not available to kernel body
//! values, because [`KernelOperation`] does not provide their operations, which is why [`compare`] and [`select`]
//! exist.
//!
//! The generated `definition` functions also use three hidden functions that exist only for macro expansion. They
//! check static shape metadata and construct the call of a tiled kernel.

use std::ops::Range;

use ryft_core::{
    AddOperation, Array, ArrayElement, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation,
    ArrayReferenceTransform, ArraySliceAxis, ArrayType, AtomId, BroadcastOperation, CompareOperation,
    ComparisonDirection, ConstantOperation, Context, DataType, Dimension, DimensionAddOperation, DimensionBounds,
    DimensionDivOperation, DimensionFromScalarOperation, DimensionMaxOperation, DimensionMinOperation,
    DimensionMulOperation, DimensionRemOperation, DimensionSubOperation, DimensionType, DimensionValue,
    DimensionVariable, Domain, DotOperation, IotaOperation, NestedTracingContext, OneOperation, Placeholder,
    ProgramBuilder, ProgramError, ProjectedValue, ReferenceAddUpdateOperation, ReferenceView, ReferenceWriteOperation,
    SaturatingSubOperation, SelectOperation, StagingContext, Tracer, TypeError, Typed, Value, ValueDomainDispatch,
    WhileOperation, ZeroOperation,
};

use crate::calls::{KernelCallOperation, KernelDefinition, KernelError, KernelParameter};
use crate::grids::{Grid, GridDimension, GridExecution};
use crate::indexing::TileLoadOperation;
use crate::interpretation::DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS;
use crate::mappings::{BlockMapping, BlockMappingError, BoundaryPolicy};
use crate::memory::{
    AsyncCopyOperation, KernelMemoryError, MaskedLoadOperation, MaskedStoreOperation, MaskedSwapOperation,
    ScratchOperation, WaitOperation, WindowMaskOperation,
};
use crate::operations::KernelOperation;
use crate::validation::KernelParameterAccess;

/// Functional kernel invocation capability for array values. Concrete host arrays use the qualified reference
/// interpreter, and projected symbolic arrays bind the [`KernelCallOperation`] of the definition with its immutable
/// body into their existing context. Backend array families may implement this capability through their own
/// compilation domain and selected adapter. A definition remains independently inspectable and compilable through
/// [`KernelDefinition`].
///
/// Host arrays are interpreted through [`KernelDefinition::interpret`] with
/// [`DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS`] as the maximum grid size to enumerate when a parameter mapping
/// lacks a separable tiling proof (i.e., when [`BlockMapping::tiling_axes`] returns `None`). Grid points are otherwise
/// not capped, and replay has no step limit because kernel bodies admit only bounded loops.
pub trait KernelCall: Clone + Typed<Type = ArrayType> {
    /// Invokes or stages one portable definition with its ordinary array inputs and results.
    fn call_kernel(definition: &KernelDefinition, inputs: &[Self]) -> Result<Vec<Self>, ProgramError>;
}

impl KernelCall for Array {
    fn call_kernel(definition: &KernelDefinition, inputs: &[Self]) -> Result<Vec<Self>, ProgramError> {
        definition.interpret(inputs.to_vec(), DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS)
    }
}

impl<V: Value<Type = ArrayIrType, Dispatch = ValueDomainDispatch>> KernelCall for ProjectedValue<ArrayType, V>
where
    V::Domain: Context<Constant = ArrayIrValue<Array>, Operation = KernelOperation>,
{
    fn call_kernel(definition: &KernelDefinition, inputs: &[Self]) -> Result<Vec<Self>, ProgramError> {
        let first = inputs.first().ok_or_else(|| {
            ProgramError::Type(TypeError::invalid(
                "symbolic zero-input kernel calls require an explicit context and definition binding",
            ))
        })?;
        first
            .value()
            .domain()
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

/// Returns one static full array extent for generated shape metadata after checking the rank and static shape.
/// This checked helper prevents indexing panics in the metadata expressions of an annotated kernel.
#[doc(hidden)]
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

/// Declares one [`KernelParameter`] whose window is its whole array, using a [`BlockMapping`] with no inputs that
/// returns zero starts, which suits a rank-0 logical grid. Generated tiled definitions use explicit mappings instead,
/// and this helper does not infer a tiled partition or bypass canonical slice validation.
///
/// # Errors
///
/// Returns [`KernelError::Type`] when `type` has no static shape, and otherwise the errors of [`BlockMapping::new`]
/// and [`KernelParameter::new`].
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

/// Dimension arithmetic over the grid coordinates and the scalar prefetched dimensions of a kernel call, which
/// [`mapped_call`] lowers into the [`BlockMapping`] of a parameter. Every operation lowers to the checked dimension
/// operation of `ryft-core` with the same name (e.g., [`DimensionAddOperation`] for [`MappingExpression::Add`]), so
/// evaluating a mapping reports overflow, underflow of a plain subtraction, and division by zero as errors instead of
/// wrapping.
///
/// Expressions combine with the `+`, `-`, `*`, `/`, and `%` operators and with [`Self::min`], [`Self::max`], and
/// [`Self::saturating_sub`], so a start such as `i * 32 + offset` is written as
/// `MappingExpression::GridCoordinate(0) * MappingExpression::Constant(32) + MappingExpression::Prefetch(0)`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MappingExpression {
    /// Coordinate of the current program along the grid axis at this position.
    GridCoordinate(usize),

    /// Value of the scalar prefetched array at this position, converted to a dimension.
    Prefetch(usize),

    /// Constant extent.
    Constant(usize),

    /// Sum of two expressions.
    Add(Box<Self>, Box<Self>),

    /// Difference of two expressions, which is an error when the right-hand side input exceeds the left-hand side
    /// input.
    Sub(Box<Self>, Box<Self>),

    /// Difference of two expressions, clamped at zero.
    SaturatingSub(Box<Self>, Box<Self>),

    /// Product of two expressions.
    Mul(Box<Self>, Box<Self>),

    /// Flooring quotient of two expressions.
    Div(Box<Self>, Box<Self>),

    /// Remainder of the flooring division of two expressions.
    Rem(Box<Self>, Box<Self>),

    /// Minimum of two expressions.
    Min(Box<Self>, Box<Self>),

    /// Maximum of two expressions.
    Max(Box<Self>, Box<Self>),
}

impl MappingExpression {
    /// Returns the difference of this expression and `right`, clamped at zero.
    pub fn saturating_sub(self, right: Self) -> Self {
        Self::SaturatingSub(Box::new(self), Box::new(right))
    }

    /// Returns the minimum of this expression and `right`.
    pub fn min(self, right: Self) -> Self {
        Self::Min(Box::new(self), Box::new(right))
    }

    /// Returns the maximum of this expression and `right`.
    pub fn max(self, right: Self) -> Self {
        Self::Max(Box::new(self), Box::new(right))
    }
}

impl std::ops::Add for MappingExpression {
    type Output = Self;

    fn add(self, right: Self) -> Self {
        Self::Add(Box::new(self), Box::new(right))
    }
}

impl std::ops::Sub for MappingExpression {
    type Output = Self;

    fn sub(self, right: Self) -> Self {
        Self::Sub(Box::new(self), Box::new(right))
    }
}

impl std::ops::Mul for MappingExpression {
    type Output = Self;

    fn mul(self, right: Self) -> Self {
        Self::Mul(Box::new(self), Box::new(right))
    }
}

impl std::ops::Div for MappingExpression {
    type Output = Self;

    fn div(self, right: Self) -> Self {
        Self::Div(Box::new(self), Box::new(right))
    }
}

impl std::ops::Rem for MappingExpression {
    type Output = Self;

    fn rem(self, right: Self) -> Self {
        Self::Rem(Box::new(self), Box::new(right))
    }
}

/// Window of one parameter of a call that [`mapped_call`] constructs, which determines the [`BlockMapping`] of that
/// parameter.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ParameterWindow {
    /// Whole array at every grid point, which requires a static array shape. Every element start is a constant zero
    /// and the boundary policy is [`BoundaryPolicy::InBounds`].
    WholeArray,

    /// Block whose element start along each axis is a block index times the block extent of that axis. A constant
    /// index becomes a constant start, and every other index is multiplied by a constant equal to the block extent, so
    /// an index that is a grid coordinate or a prefetched dimension produces the forms that
    /// [`BlockMapping::tiling_axes`] recognizes.
    BlockIndices {
        /// Static extents of the block, including any elements that a masked window must exclude.
        block_shape: Vec<usize>,

        /// Required handling of a block that crosses the boundary of its array.
        boundary: BoundaryPolicy,

        /// Block index per array axis.
        indices: Vec<MappingExpression>,
    },

    /// Block whose element start along each axis is an arbitrary expression. A start that is a constant zero, or a
    /// grid coordinate along an axis whose block extent is 1, is recognized by [`BlockMapping::tiling_axes`], and
    /// every other start leaves the mapping without a separable tiling proof.
    ElementStarts {
        /// Static extents of the block, including any elements that a masked window must exclude.
        block_shape: Vec<usize>,

        /// Required handling of a block that crosses the boundary of its array.
        boundary: BoundaryPolicy,

        /// Element start per array axis.
        starts: Vec<MappingExpression>,
    },
}

/// Constructs a [`KernelCallOperation`] over `grid` whose parameters have the given full array types, accesses, and
/// windows, followed by the scalar prefetch values with the rank-0 integer types `prefetch_types`. Each window is
/// lowered into a [`BlockMapping`] whose inputs are the grid coordinates followed by the prefetched dimensions. Those
/// input types are shared by all parameters, and each admits every coordinate that `grid` may execute, so they
/// satisfy the identity and bound checks of [`KernelCallOperation::new_with_prefetch`]. Parameters, grid
/// coordinates, and prefetch values keep their order in the body inputs.
///
/// # Errors
///
/// Returns [`KernelError::Type`] when a whole array window has a type without a static shape, when an expression
/// references a grid axis or prefetch value that does not exist, or when a constant block index times its block
/// extent overflows, [`KernelError::Mapping`] when the number of indices or starts of a window differs from its block
/// rank, [`KernelError::Program`] when a dimension operation rejects its inputs, and the errors of
/// [`BlockMapping::new`], [`KernelParameter::new`], and [`KernelCallOperation::new_with_prefetch`].
pub fn mapped_call(
    grid: Grid,
    parameters: Vec<(ArrayType, KernelParameterAccess, ParameterWindow)>,
    prefetch_types: Vec<ArrayType>,
) -> Result<KernelCallOperation, KernelError> {
    let mut input_types = grid
        .dimensions()
        .iter()
        .enumerate()
        .map(|(axis, dimension)| {
            // A coordinate is at most the maximum extent of its axis minus one, and `Grid::new` guarantees that every
            // dynamic extent has a finite upper bound, which is exclusive.
            let maximum = match dimension.extent() {
                Dimension::Static(extent) => *extent,
                Dimension::Dynamic(variable) => variable.bounds().upper().unwrap().saturating_sub(1),
            };
            let name = dimension.name().map_or_else(|| format!("grid_axis_{axis}"), str::to_string);
            DimensionType::new(name, DimensionBounds::non_negative(Some(maximum.max(1))).unwrap())
        })
        .collect::<Vec<_>>();
    input_types.extend((0..prefetch_types.len()).map(|index| {
        DimensionType::new(format!("prefetch_{index}"), DimensionBounds::non_negative(None).unwrap())
    }));
    let grid_rank = grid.dimensions().len();
    let parameters = parameters
        .into_iter()
        .enumerate()
        .map(|(parameter, (r#type, access, window))| {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let inputs = input_types.iter().map(|r#type| builder.add_input(r#type.clone().into())).collect::<Vec<_>>();
            let (block_shape, boundary, starts) = match window {
                ParameterWindow::WholeArray => {
                    let shape = r#type.static_shape().ok_or_else(|| {
                        TypeError::invalid(format!(
                            "kernel parameter {parameter} with a whole array window requires a static shape",
                        ))
                    })?;
                    let starts = (0..shape.rank())
                        .map(|_| Ok(builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0)?))))
                        .collect::<Result<Vec<_>, ProgramError>>()?;
                    (shape.dimensions().to_vec(), BoundaryPolicy::InBounds, starts)
                }
                ParameterWindow::BlockIndices { block_shape, boundary, indices } => {
                    validate_window_rank(&block_shape, &indices)?;
                    let starts = indices
                        .iter()
                        .zip(&block_shape)
                        .map(|(index, &block)| {
                            if let MappingExpression::Constant(index) = index {
                                let start = index.checked_mul(block).ok_or_else(|| {
                                    TypeError::invalid(format!(
                                        "kernel parameter {parameter} block index {index} times block extent {block} \
                                         overflows",
                                    ))
                                })?;
                                let start = DimensionValue::constant(start).map_err(ProgramError::from)?;
                                return Ok(builder.add_constant(ArrayIrValue::Dimension(start)));
                            }
                            let index = lower_mapping_expression(&mut builder, &inputs, grid_rank, index)?;
                            let scale = DimensionValue::constant(block).map_err(ProgramError::from)?;
                            let operation =
                                DimensionMulOperation::new(&atom_dimension_type(&builder, index), &scale.r#type())
                                    .map_err(ProgramError::from)?;
                            let scale = builder.add_constant(ArrayIrValue::Dimension(scale));
                            Ok(builder.add_instruction(operation, vec![], vec![index, scale], None)?[0])
                        })
                        .collect::<Result<Vec<_>, KernelError>>()?;
                    (block_shape, boundary, starts)
                }
                ParameterWindow::ElementStarts { block_shape, boundary, starts } => {
                    validate_window_rank(&block_shape, &starts)?;
                    let starts = starts
                        .iter()
                        .map(|start| lower_mapping_expression(&mut builder, &inputs, grid_rank, start))
                        .collect::<Result<Vec<_>, KernelError>>()?;
                    (block_shape, boundary, starts)
                }
            };
            let rank = starts.len();
            let program = builder.build(starts, vec![Placeholder; inputs.len()], vec![Placeholder; rank])?;
            KernelParameter::new(r#type, access, BlockMapping::new(program, block_shape, boundary)?)
        })
        .collect::<Result<Vec<_>, KernelError>>()?;
    KernelCallOperation::new_with_prefetch(grid, parameters, prefetch_types)
}

/// Constructs a [`KernelCallOperation`] on a parallel grid with one point per output tile, with read-only inputs whose
/// windows are their whole arrays and a write-only output whose mapped windows are disjoint. Every mapping consumes the
/// same canonical grid coordinate signature. Input mappings ignore those coordinates, so the full input references
/// remain available for dynamically selected loads inside the body. The call is constructed by [`mapped_call`], with
/// the grid coordinates as the block indices of the output.
#[doc(hidden)]
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
    let mut parameters = input_types
        .iter()
        .map(|r#type| (r#type.clone(), KernelParameterAccess::ReadOnly, ParameterWindow::WholeArray))
        .collect::<Vec<_>>();
    let indices = (0..extents.len()).map(MappingExpression::GridCoordinate).collect();
    parameters.push((
        output_type,
        KernelParameterAccess::WriteOnly,
        ParameterWindow::BlockIndices { block_shape, boundary, indices },
    ));
    mapped_call(grid, parameters, vec![])
}

/// Stages a [`TileLoadOperation`] that loads one masked tile of the reference `source`, translating tile indices into
/// checked element starts with [`DimensionMulOperation`]s. Integer scalar array indices enter the dimension universe
/// through a checked [`DimensionFromScalarOperation`], and dimension indices (e.g., grid coordinates) are used
/// directly. The elements of the tile that lie outside `source` take the value `other`.
pub fn tile_load<T: ArrayElement, C>(
    context: &C,
    source: &C::Value,
    block_shape: Vec<usize>,
    indices: &[C::Value],
    other: T,
) -> Result<C::Value, ProgramError>
where
    C: Context<Type = ArrayIrType, Constant = ArrayIrValue<Array>, Operation = KernelOperation>,
{
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
                        ArrayIrOperation::DimensionFromScalar(DimensionFromScalarOperation::new(variable.into())),
                        vec![],
                        std::slice::from_ref(index),
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

/// Stores a complete tile into a private mapped output window through a [`ReferenceWriteOperation`]. Every private
/// tile element is valid storage, and the qualified invocation publishes only the intersection with the full output
/// shape. This boundary clips publication without requiring an allocated mask or weakening initialization checks.
pub fn tile_store<C>(context: &C, output: &C::Value, value: &C::Value) -> Result<(), ProgramError>
where
    C: Context<Type = ArrayIrType, Constant = ArrayIrValue<Array>, Operation = KernelOperation>,
{
    context.bind(
        ArrayIrOperation::ReferenceWrite(ReferenceWriteOperation::new()),
        vec![],
        &[output.clone(), value.clone()],
    )?;
    Ok(())
}

/// Stages a [`MaskedLoadOperation`] that loads the elements of the reference `reference` that `mask` selects and
/// returns `other` for every other element. `mask` is a Boolean array with the shape of the referent, and `other`
/// becomes a constant array of the referent type, so `T` must be the element type of the referent. Under kernel
/// qualification, the mask is also intersected with the validity of the window, so padding elements of a masked
/// window take the value `other` as well.
pub fn masked_load<T: ArrayElement, C>(
    context: &C,
    reference: &C::Value,
    mask: &C::Value,
    other: T,
) -> Result<C::Value, ProgramError>
where
    C: Context<Type = ArrayIrType, Constant = ArrayIrValue<Array>, Operation = KernelOperation>,
{
    let referent = static_referent(reference)?;
    let count = referent.static_shape().unwrap().dimensions().iter().product::<usize>();
    let other = context.lift(ArrayIrValue::Array(Array::from_elements(referent, &vec![other; count])?))?;
    Ok(context.bind(MaskedLoadOperation::new(), vec![], &[reference.clone(), mask.clone(), other])?.remove(0))
}

/// Stages a [`MaskedLoadOperation`] that loads the whole reference `reference` under a constant mask that selects
/// every element, which is the load of a window whose padding elements take the value `other`. Under kernel
/// qualification, that mask is intersected with the validity of the window, so exactly the valid elements are read.
/// `T` must be the element type of the referent.
pub fn masked_load_or<T: ArrayElement, C>(context: &C, reference: &C::Value, other: T) -> Result<C::Value, ProgramError>
where
    C: Context<Type = ArrayIrType, Constant = ArrayIrValue<Array>, Operation = KernelOperation>,
{
    let mask = context.lift(ArrayIrValue::Array(full_mask(&static_referent(reference)?)?))?;
    masked_load(context, reference, &mask, other)
}

/// Stages a [`MaskedStoreOperation`] that writes the elements of `value` that `mask` selects into the reference
/// `reference` and preserves every other element. `value` has exactly the referent type, and `mask` is a Boolean array
/// with the referent shape. A store under the [`window_mask`] of the same reference initializes a masked write-only
/// window completely.
pub fn masked_store<C>(context: &C, reference: &C::Value, value: &C::Value, mask: &C::Value) -> Result<(), ProgramError>
where
    C: Context<Type = ArrayIrType, Constant = ArrayIrValue<Array>, Operation = KernelOperation>,
{
    context.bind(MaskedStoreOperation::new(), vec![], &[reference.clone(), value.clone(), mask.clone()])?;
    Ok(())
}

/// Stages a [`MaskedSwapOperation`] that replaces the elements of the reference `reference` that `mask` selects with
/// the corresponding elements of `value` and returns their previous values, with `other` for every other element.
/// `value` has exactly the referent type, `mask` is a Boolean array with the referent shape, and `other` becomes a
/// constant array of the referent type, so `T` must be the element type of the referent.
pub fn masked_swap<T: ArrayElement, C>(
    context: &C,
    reference: &C::Value,
    value: &C::Value,
    mask: &C::Value,
    other: T,
) -> Result<C::Value, ProgramError>
where
    C: Context<Type = ArrayIrType, Constant = ArrayIrValue<Array>, Operation = KernelOperation>,
{
    let referent = static_referent(reference)?;
    let count = referent.static_shape().unwrap().dimensions().iter().product::<usize>();
    let other = context.lift(ArrayIrValue::Array(Array::from_elements(referent, &vec![other; count])?))?;
    let inputs = [reference.clone(), value.clone(), mask.clone(), other];
    Ok(context.bind(MaskedSwapOperation::new(), vec![], &inputs)?.remove(0))
}

/// Stages a [`WindowMaskOperation`] that returns which elements of the window of the reference `reference` are valid
/// for the current grid point, as a Boolean array with the referent shape. Every element of an in-bounds window is
/// valid.
pub fn window_mask<C>(context: &C, reference: &C::Value) -> Result<C::Value, ProgramError>
where
    C: Context<Type = ArrayIrType, Constant = ArrayIrValue<Array>, Operation = KernelOperation>,
{
    Ok(context.bind(WindowMaskOperation, vec![], std::slice::from_ref(reference))?.remove(0))
}

/// Adds `value` into the window of the reference `reference` in program order. A window with
/// [`BoundaryPolicy::InBounds`] uses a [`ReferenceAddUpdateOperation`]. A window with [`BoundaryPolicy::Masked`]
/// may contain padding that an unmasked read rejects, so it is instead loaded through a [`MaskedLoadOperation`] whose
/// padding elements are zero, added to `value`, and stored back through a [`MaskedStoreOperation`]. Both masks select
/// every element, and kernel qualification restricts them to the valid elements of the window. `value` has exactly
/// the referent type in both cases.
pub fn accumulate<C>(
    context: &C,
    reference: &C::Value,
    value: &C::Value,
    boundary: BoundaryPolicy,
) -> Result<(), ProgramError>
where
    C: Context<Type = ArrayIrType, Constant = ArrayIrValue<Array>, Operation = KernelOperation>,
{
    if boundary == BoundaryPolicy::InBounds {
        context.bind(ReferenceAddUpdateOperation::new(), vec![], &[reference.clone(), value.clone()])?;
        return Ok(());
    }
    let referent = static_referent(reference)?;
    let mask = context.lift(ArrayIrValue::Array(full_mask(&referent)?))?;
    let zero = ArrayIrOperation::<Array>::from(ArrayOperation::Zero(ZeroOperation::new(referent)));
    let zero = context.bind(zero, vec![], &[])?.remove(0);
    let current = context.bind(MaskedLoadOperation::new(), vec![], &[reference.clone(), mask.clone(), zero])?.remove(0);
    let sum = context
        .bind(ArrayIrOperation::from(ArrayOperation::Add(AddOperation::new())), vec![], &[current, value.clone()])?
        .remove(0);
    masked_store(context, reference, &sum, &mask)
}

/// Stages a [`ScratchOperation`] that allocates logically uninitialized storage with the canonical element type of
/// `T`, the static shape `shape`, and a minimum native alignment of `alignment` bytes, scoped to the enclosing kernel
/// region. The natural alignment of `T` is `align_of::<T>()`.
///
/// # Errors
///
/// Returns [`ProgramError::Custom`] holding [`KernelMemoryError::InvalidAlignment`] when `alignment` is not a positive
/// power of two, and the errors of binding the operation otherwise.
pub fn scratch<T: ArrayElement, C>(
    context: &C,
    shape: impl Into<Vec<usize>>,
    alignment: usize,
) -> Result<C::Value, ProgramError>
where
    C: Context<Type = ArrayIrType, Constant = ArrayIrValue<Array>, Operation = KernelOperation>,
{
    let operation = match ScratchOperation::new(ArrayType::new_static(T::data_type(), shape), alignment) {
        Ok(operation) => operation,
        Err(KernelMemoryError::Program(error)) => return Err(error),
        Err(error) => return Err(ProgramError::custom(error)),
    };
    Ok(context.bind(operation, vec![], &[])?.remove(0))
}

/// Stages an [`AsyncCopyOperation`] that starts copying the reference `source` into the reference `destination` and
/// returns its completion token, which a [`wait`] in the same region must consume. Until then, `source` may be read
/// but not changed, and `destination` may not be accessed.
pub fn async_copy<C>(context: &C, source: &C::Value, destination: &C::Value) -> Result<C::Value, ProgramError>
where
    C: Context<Type = ArrayIrType, Constant = ArrayIrValue<Array>, Operation = KernelOperation>,
{
    Ok(context.bind(AsyncCopyOperation::new(), vec![], &[source.clone(), destination.clone()])?.remove(0))
}

/// Stages a [`WaitOperation`] that completes the [`async_copy`] that returned `token`, after which its destination is
/// initialized.
pub fn wait<C>(context: &C, token: &C::Value) -> Result<(), ProgramError>
where
    C: Context<Type = ArrayIrType, Constant = ArrayIrValue<Array>, Operation = KernelOperation>,
{
    context.bind(WaitOperation, vec![], std::slice::from_ref(token))?;
    Ok(())
}

/// Compares `left` with `right` according to `direction` through a [`CompareOperation`], which returns Boolean data.
/// The comparison is selected by member kind, as the array IR operation family selects it for the
/// [`Compare`](ryft_core::Compare) capability. Two arrays compare through the array member, which broadcasts them to a
/// common shape and promotes their element types, and two dimensions compare through the array IR comparison, which
/// returns a rank-0 Boolean array. Kernel body values cannot use that capability itself, because the kernel operation
/// family does not provide its operation, so bodies compare through this function instead.
pub fn compare<C>(
    context: &C,
    left: &C::Value,
    right: &C::Value,
    direction: ComparisonDirection,
) -> Result<C::Value, ProgramError>
where
    C: Context<Type = ArrayIrType, Constant = ArrayIrValue<Array>, Operation = KernelOperation>,
{
    let operation = match (left.r#type().as_ref(), right.r#type().as_ref()) {
        (ArrayIrType::Array(_), ArrayIrType::Array(_)) => {
            ArrayIrOperation::from(ArrayOperation::Compare(CompareOperation::new(direction)))
        }
        _ => ArrayIrOperation::Compare(CompareOperation::new(direction)),
    };
    Ok(context.bind(operation, vec![], &[left.clone(), right.clone()])?.remove(0))
}

/// Selects the elements of `on_true` where the Boolean `mask` is true and the elements of `on_false` elsewhere
/// through a [`SelectOperation`]. The inputs are broadcast to a common shape and the element types of the two branches
/// are promoted. Kernel body values cannot use the [`Select`](ryft_core::Select) capability, because the kernel
/// operation family does not provide its operation, so bodies select through this function instead.
pub fn select<C>(
    context: &C,
    mask: &C::Value,
    on_true: &C::Value,
    on_false: &C::Value,
) -> Result<C::Value, ProgramError>
where
    C: Context<Type = ArrayIrType, Constant = ArrayIrValue<Array>, Operation = KernelOperation>,
{
    let operation = ArrayIrOperation::<Array>::from(ArrayOperation::Select(SelectOperation::new()));
    Ok(context.bind(operation, vec![], &[mask.clone(), on_true.clone(), on_false.clone()])?.remove(0))
}

/// Compares the dimension `dimension` with the constant `constant` through a [`CompareOperation`] and returns a rank-0
/// Boolean array. The constant is a lifted dimension constant and the comparison is always staged, so a comparison of
/// a grid coordinate with zero (e.g., `k == 0` on a sequential axis) has the form that
/// [`validate_kernel_initialization`](crate::initialization::validate_kernel_initialization) recognizes as the first
/// step of that axis.
pub fn compare_dimension<C>(
    context: &C,
    dimension: &C::Value,
    direction: ComparisonDirection,
    constant: usize,
) -> Result<C::Value, ProgramError>
where
    C: Context<Type = ArrayIrType, Constant = ArrayIrValue<Array>, Operation = KernelOperation>,
{
    let constant = context.lift(ArrayIrValue::Dimension(DimensionValue::constant(constant)?))?;
    Ok(context
        .bind(ArrayIrOperation::Compare(CompareOperation::new(direction)), vec![], &[dimension.clone(), constant])?
        .remove(0))
}

/// Returns a [`ReferenceView`] of the reference `reference` that selects one static range of elements per axis of its
/// referent, where `None` selects a whole axis. Constructing the view stages nothing. Accesses through the
/// [`ReferenceRead`](ryft_core::ReferenceRead), [`ReferenceWrite`](ryft_core::ReferenceWrite),
/// [`ReferenceSwap`](ryft_core::ReferenceSwap), and [`ReferenceAddUpdate`](ryft_core::ReferenceAddUpdate)
/// capabilities of the view apply an [`ArrayReferenceTransform::Slice`] to `reference`.
///
/// # Errors
///
/// Returns [`ProgramError::Type`] when `reference` is not a reference with a static referent shape, when the number
/// of ranges differs from the referent rank, when a range is decreasing, and when a range exceeds its axis.
pub fn reference_slice<V: Value<Type = ArrayIrType>>(
    reference: &V,
    ranges: &[Option<Range<usize>>],
) -> Result<ReferenceView<V, ArrayReferenceTransform, V>, ProgramError> {
    let referent = static_referent(reference)?;
    let shape = referent.static_shape().unwrap();
    if ranges.len() != shape.rank() {
        return Err(TypeError::invalid(format!(
            "kernel slice has {} ranges but its reference has rank {}",
            ranges.len(),
            shape.rank(),
        ))
        .into());
    }
    let axes = ranges
        .iter()
        .zip(shape.dimensions())
        .map(|(range, &extent)| {
            let range = range.clone().unwrap_or(0..extent);
            let size = range.end.checked_sub(range.start).ok_or_else(|| {
                TypeError::invalid(format!("kernel slice range `{}..{}` is decreasing", range.start, range.end))
            })?;
            Ok(ArraySliceAxis::new(range.start, size, 1))
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;
    ReferenceView::new(reference.clone())?.with_bound_transform(ArrayReferenceTransform::Slice { axes }, vec![])
}

/// Stages a bounded range loop with explicit carried values through a [`WhileOperation`], whose condition compares the
/// index with the end of the range through a [`CompareOperation`] and whose body increments the index through an
/// [`AddOperation`]. The body is traced exactly once even for an empty range. Its index is an ordinary signed 64-bit
/// scalar array, so callers use a [`DimensionFromScalarOperation`] when an indexing operation needs a dimension value.
pub fn for_loop<C, F>(
    context: &C,
    range: Range<usize>,
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

/// Returns the ceiling of `extent` divided by `divisor` for static kernel shape metadata, rejecting a zero divisor.
#[doc(hidden)]
pub fn shape_div_ceil(extent: usize, divisor: usize) -> Result<usize, KernelError> {
    if divisor == 0 {
        return Err(KernelError::Type(TypeError::invalid("kernel shape ceiling division requires a positive divisor")));
    }
    Ok(extent.div_ceil(divisor))
}

/// Constructs a statically shaped zero tile with the canonical element type of `T`. The active context records the
/// ordinary [`ZeroOperation`], and no host tile is allocated while tracing. Shape and placement validation remain with
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

/// Constructs a statically shaped tile of ones with the canonical element type of `T` through the ordinary
/// [`OneOperation`], without allocating a host tile while tracing.
pub fn ones<T: ArrayElement, C: Context<Type = ArrayIrType>>(
    context: &C,
    shape: impl Into<Vec<usize>>,
) -> Result<C::Value, ProgramError>
where
    C::Operation: From<ArrayIrOperation<Array>>,
{
    let operation = ArrayIrOperation::<Array>::from(ArrayOperation::One(OneOperation::new(ArrayType::new_static(
        T::data_type(),
        shape,
    ))));
    Ok(context.bind(operation, Vec::new(), &[])?.remove(0))
}

/// Constructs a statically shaped tile whose elements all equal `value`, with the canonical element type of `T`. The
/// value is staged as a rank-0 [`ConstantOperation`] and expanded by a [`BroadcastOperation`], so no host tile is
/// allocated while tracing.
pub fn full<T: ArrayElement, C: Context<Type = ArrayIrType>>(
    context: &C,
    shape: impl Into<Vec<usize>>,
    value: T,
) -> Result<C::Value, ProgramError>
where
    C::Operation: From<ArrayIrOperation<Array>>,
{
    let constant = ArrayIrOperation::<Array>::from(ArrayOperation::Constant(ConstantOperation::new(Array::scalar(
        value,
    )?)));
    let constant = context.bind(constant, Vec::new(), &[])?.remove(0);
    let broadcast = ArrayIrOperation::<Array>::from(ArrayOperation::Broadcast(BroadcastOperation::new(
        ArrayType::new_static(T::data_type(), shape),
        Vec::new(),
    )));
    Ok(context.bind(broadcast, Vec::new(), &[constant])?.remove(0))
}

/// Constructs a statically shaped tile with the canonical element type of `T` whose elements count up from zero along
/// the axis `dimension` and are constant along every other axis, through the ordinary [`IotaOperation`]. For example,
/// `iota::<i32, _>(context, [n], 0)` stages the vector `[0, 1, ..., n - 1]`.
pub fn iota<T: ArrayElement, C: Context<Type = ArrayIrType>>(
    context: &C,
    shape: impl Into<Vec<usize>>,
    dimension: usize,
) -> Result<C::Value, ProgramError>
where
    C::Operation: From<ArrayIrOperation<Array>>,
{
    let operation = IotaOperation::new(ArrayType::new_static(T::data_type(), shape), dimension)?;
    let operation = ArrayIrOperation::<Array>::from(ArrayOperation::Iota(operation));
    Ok(context.bind(operation, Vec::new(), &[])?.remove(0))
}

/// Multiplies two rank-2 tiles through [`DotOperation::matmul`], retaining its data type, contraction, accumulation,
/// sharding, and numerical contracts. Incompatible tiles return the existing type diagnostic rather than panicking.
pub fn dot<V: Value<Type = ArrayIrType>>(left: &V, right: &V) -> Result<V, ProgramError>
where
    V::Domain: Context,
    <V::Domain as Domain>::Operation: From<ArrayIrOperation<Array>>,
{
    let operation = ArrayIrOperation::<Array>::from(ArrayOperation::Dot(DotOperation::matmul()));
    Ok(left
        .domain()
        .bind(operation, Vec::new(), &[left.clone(), right.clone()])?
        .into_iter()
        .next()
        .unwrap())
}

/// Validates that a window has one block index or element start per block axis, with the error that
/// [`BlockMapping::new`] reports for a mapping whose outputs differ from its block rank.
fn validate_window_rank(block_shape: &[usize], expressions: &[MappingExpression]) -> Result<(), KernelError> {
    if expressions.len() != block_shape.len() {
        return Err(BlockMappingError::RankMismatch {
            boundary: "output",
            expected: block_shape.len(),
            actual: expressions.len(),
        }
        .into());
    }
    Ok(())
}

/// Lowers `expression` into `builder`, whose inputs `inputs` are the grid coordinates followed by the prefetched
/// dimensions, and returns the atom that holds its value. Leaves add no instructions, and every operation adds exactly
/// one unfolded dimension instruction, so the mapping keeps the structure that [`BlockMapping::tiling_axes`] inspects.
fn lower_mapping_expression(
    builder: &mut ProgramBuilder<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
    inputs: &[AtomId],
    grid_rank: usize,
    expression: &MappingExpression,
) -> Result<AtomId, KernelError> {
    let (left, right) = match expression {
        MappingExpression::GridCoordinate(axis) => {
            if *axis >= grid_rank {
                return Err(TypeError::invalid(format!(
                    "kernel mapping expression references grid axis {axis}, but the grid has rank {grid_rank}",
                ))
                .into());
            }
            return Ok(inputs[*axis]);
        }
        MappingExpression::Prefetch(index) => {
            return inputs.get(grid_rank + index).copied().ok_or_else(|| {
                TypeError::invalid(format!(
                    "kernel mapping expression references scalar prefetch {index}, but the call has {} scalar \
                     prefetch values",
                    inputs.len() - grid_rank,
                ))
                .into()
            });
        }
        MappingExpression::Constant(value) => {
            let value = DimensionValue::constant(*value).map_err(ProgramError::from)?;
            return Ok(builder.add_constant(ArrayIrValue::Dimension(value)));
        }
        MappingExpression::Add(left, right)
        | MappingExpression::Sub(left, right)
        | MappingExpression::SaturatingSub(left, right)
        | MappingExpression::Mul(left, right)
        | MappingExpression::Div(left, right)
        | MappingExpression::Rem(left, right)
        | MappingExpression::Min(left, right)
        | MappingExpression::Max(left, right) => (left, right),
    };
    let left = lower_mapping_expression(builder, inputs, grid_rank, left)?;
    let right = lower_mapping_expression(builder, inputs, grid_rank, right)?;
    let (left_type, right_type) = (atom_dimension_type(builder, left), atom_dimension_type(builder, right));
    let operation = match expression {
        MappingExpression::Add(..) => DimensionAddOperation::new(&left_type, &right_type).map(ArrayIrOperation::from),
        MappingExpression::Sub(..) => DimensionSubOperation::new(&left_type, &right_type).map(ArrayIrOperation::from),
        MappingExpression::SaturatingSub(..) => Ok(ArrayIrOperation::from(SaturatingSubOperation::new())),
        MappingExpression::Mul(..) => DimensionMulOperation::new(&left_type, &right_type).map(ArrayIrOperation::from),
        MappingExpression::Div(..) => DimensionDivOperation::new(&left_type, &right_type).map(ArrayIrOperation::from),
        MappingExpression::Rem(..) => DimensionRemOperation::new(&left_type, &right_type).map(ArrayIrOperation::from),
        MappingExpression::Min(..) => DimensionMinOperation::new(&left_type, &right_type).map(ArrayIrOperation::from),
        MappingExpression::Max(..) => DimensionMaxOperation::new(&left_type, &right_type).map(ArrayIrOperation::from),
        MappingExpression::GridCoordinate(_) | MappingExpression::Prefetch(_) | MappingExpression::Constant(_) => {
            unreachable!()
        }
    }
    .map_err(ProgramError::from)?;
    Ok(builder.add_instruction(operation, vec![], vec![left, right], None)?[0])
}

/// Returns the dimension type of an atom of a mapping program, all of whose atoms are dimensions.
fn atom_dimension_type(
    builder: &ProgramBuilder<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
    atom: AtomId,
) -> DimensionType {
    let ArrayIrType::Dimension(r#type) = builder.atoms()[atom.index()].r#type().into_owned() else { unreachable!() };
    r#type
}

/// Returns the referent type of the kernel reference `reference`, which must have a static shape.
fn static_referent<V: Typed<Type = ArrayIrType>>(reference: &V) -> Result<ArrayType, ProgramError> {
    let r#type = reference.r#type();
    let ArrayIrType::Reference(reference) = r#type.as_ref() else {
        return Err(TypeError::invalid(format!("kernel memory access requires a reference but got `{type}`")).into());
    };
    if reference.referent().static_shape().is_none() {
        return Err(TypeError::invalid(format!(
            "kernel memory access requires a static referent shape but got `{}`",
            reference.referent(),
        ))
        .into());
    }
    Ok(reference.referent().clone())
}

/// Returns a constant Boolean mask with the shape of the statically shaped `referent` that selects every element.
fn full_mask(referent: &ArrayType) -> Result<Array, ProgramError> {
    let count = referent.static_shape().unwrap().dimensions().iter().product::<usize>();
    Array::from_elements(ArrayType::new(DataType::Boolean, referent.shape().clone()), &vec![true; count])
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayReference, Condition, ConvertElementType, DataType, Operation, ReferenceAtomicAddUpdate,
        ReferenceRead, ReferenceType, ReferenceWrite, Shape, TracingContext,
    };

    use crate::initialization::KernelInitializationError;
    use crate::operations::NoKernelExtension;

    use super::*;

    /// Builds a matrix product of `f32[m, k]` and `f32[k, n]` inputs with square tiles of size `block` over the grid
    /// `(i, j, k)`, where only `k` is sequential, through [`mapped_call`] and the body functions of this module. The
    /// body zeroes its write-only output tile at `k == 0` and then accumulates one tile product into it at every `k`.
    /// Masked tiles are loaded with a zero fill and accumulated through masked accesses.
    fn carried_matmul_definition(shape: [usize; 3], block: usize, boundary: BoundaryPolicy) -> KernelDefinition {
        type BranchTracer = Tracer<NestedTracingContext<TracingContext<ArrayIrValue<Array>, KernelOperation>>>;
        let [m, k, n] = shape;
        let window = |axes: [usize; 2]| ParameterWindow::BlockIndices {
            block_shape: vec![block, block],
            boundary,
            indices: axes.map(MappingExpression::GridCoordinate).to_vec(),
        };
        let grid = Grid::new(vec![
            GridDimension::new(m.div_ceil(block).into(), GridExecution::Parallel).with_name("i").unwrap(),
            GridDimension::new(n.div_ceil(block).into(), GridExecution::Parallel).with_name("j").unwrap(),
            GridDimension::new(k.div_ceil(block).into(), GridExecution::Sequential).with_name("k").unwrap(),
        ])
        .unwrap();
        let operation = mapped_call(
            grid,
            vec![
                (ArrayType::new_static(DataType::F32, [m, k]), KernelParameterAccess::ReadOnly, window([0, 2])),
                (ArrayType::new_static(DataType::F32, [k, n]), KernelParameterAccess::ReadOnly, window([2, 1])),
                (ArrayType::new_static(DataType::F32, [m, n]), KernelParameterAccess::WriteOnly, window([0, 1])),
            ],
            vec![],
        )
        .unwrap();
        KernelDefinition::trace(operation, |(references, coordinates)| {
            let context = references[0].context();
            let first = compare_dimension(context, &coordinates[2], ComparisonDirection::Equal, 0)?;
            first.condition(
                |values: Vec<BranchTracer>| {
                    values[0].write(&zeros::<f32, _>(values[0].context(), [block, block])?)?;
                    Ok(Vec::<BranchTracer>::new())
                },
                |_values: Vec<BranchTracer>| Ok(Vec::<BranchTracer>::new()),
                vec![references[2].clone()],
            )?;
            let tiles = references[..2]
                .iter()
                .map(|reference| match boundary {
                    BoundaryPolicy::InBounds => reference.read(),
                    BoundaryPolicy::Masked => masked_load_or(context, reference, 0f32),
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            accumulate(context, &references[2], &dot(&tiles[0], &tiles[1])?, boundary)
        })
        .unwrap()
    }

    #[test]
    fn test_static_extent() {
        let r#type = ArrayType::new_static(DataType::F32, [2, 3]);
        assert_eq!(static_extent(&r#type, 1), Ok(3));
        assert_eq!(
            static_extent(&r#type, 2),
            Err(KernelError::Type(TypeError::invalid("kernel shape metadata axis exceeds the input rank"))),
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
    fn test_mapping_expression() {
        let coordinate = || Box::new(MappingExpression::GridCoordinate(0));
        let prefetch = || Box::new(MappingExpression::Prefetch(0));
        assert_eq!(
            MappingExpression::GridCoordinate(0) * MappingExpression::Constant(2) + MappingExpression::Prefetch(0),
            MappingExpression::Add(
                Box::new(MappingExpression::Mul(coordinate(), Box::new(MappingExpression::Constant(2)))),
                prefetch(),
            ),
        );
        assert_eq!(
            MappingExpression::GridCoordinate(0) - MappingExpression::Prefetch(0),
            MappingExpression::Sub(coordinate(), prefetch()),
        );
        assert_eq!(
            MappingExpression::GridCoordinate(0) / MappingExpression::Prefetch(0),
            MappingExpression::Div(coordinate(), prefetch()),
        );
        assert_eq!(
            MappingExpression::GridCoordinate(0) % MappingExpression::Prefetch(0),
            MappingExpression::Rem(coordinate(), prefetch()),
        );
        assert_eq!(
            MappingExpression::GridCoordinate(0).saturating_sub(MappingExpression::Prefetch(0)),
            MappingExpression::SaturatingSub(coordinate(), prefetch()),
        );
        assert_eq!(
            MappingExpression::GridCoordinate(0).min(MappingExpression::Prefetch(0)),
            MappingExpression::Min(coordinate(), prefetch()),
        );
        assert_eq!(
            MappingExpression::GridCoordinate(0).max(MappingExpression::Prefetch(0)),
            MappingExpression::Max(coordinate(), prefetch()),
        );
    }

    #[test]
    fn test_mapped_call() {
        // A whole array window, a tiled block, and an element offset window over a grid with one named axis.
        let grid = Grid::new(vec![
            GridDimension::new(3usize.into(), GridExecution::Parallel).with_name("i").unwrap(),
            GridDimension::new(2usize.into(), GridExecution::Sequential),
        ])
        .unwrap();
        let operation = mapped_call(
            grid,
            vec![
                (
                    ArrayType::new_static(DataType::F32, [2, 3]),
                    KernelParameterAccess::ReadOnly,
                    ParameterWindow::WholeArray,
                ),
                (
                    ArrayType::new_static(DataType::F32, [6, 8]),
                    KernelParameterAccess::WriteOnly,
                    ParameterWindow::BlockIndices {
                        block_shape: vec![2, 4],
                        boundary: BoundaryPolicy::InBounds,
                        indices: vec![MappingExpression::GridCoordinate(0), MappingExpression::GridCoordinate(1)],
                    },
                ),
                (
                    ArrayType::new_static(DataType::I32, [16]),
                    KernelParameterAccess::ReadWrite,
                    ParameterWindow::ElementStarts {
                        block_shape: vec![1],
                        boundary: BoundaryPolicy::Masked,
                        starts: vec![MappingExpression::GridCoordinate(0) + MappingExpression::Prefetch(0)],
                    },
                ),
            ],
            vec![ArrayType::scalar(DataType::I32)],
        )
        .unwrap();
        assert_eq!(operation.grid().named_axis("i"), Some(0));
        assert_eq!(operation.coordinate_types().len(), 2);
        assert_eq!(operation.prefetch_types(), &[ArrayType::scalar(DataType::I32)]);
        let mappings = operation.parameters().iter().map(KernelParameter::mapping).collect::<Vec<_>>();
        assert_eq!(
            mappings.iter().map(|mapping| mapping.block_shape().to_vec()).collect::<Vec<_>>(),
            vec![vec![2, 3], vec![2, 4], vec![1]],
        );
        assert_eq!(
            mappings.iter().map(|mapping| mapping.boundary_policy()).collect::<Vec<_>>(),
            vec![BoundaryPolicy::InBounds, BoundaryPolicy::InBounds, BoundaryPolicy::Masked],
        );
        assert_eq!(
            mappings.iter().map(|mapping| mapping.tiling_axes()).collect::<Vec<_>>(),
            vec![Some(vec![None, None]), Some(vec![Some(0), Some(1)]), None],
        );
        assert_eq!(
            mappings[1].program().to_string(),
            indoc! {"
                lambda %0:dimension<i ∈ [0, 3)>, %1:dimension<grid_axis_1 ∈ [0, 2)>, %2:dimension<prefetch_0 ∈ [0, ∞)> .
                let %3:dimension<2> = const 2
                    %4:dimension<i * 2 ∈ [0, 5)> = dimension_mul %0 %3
                    %5:dimension<4> = const 4
                    %6:dimension<grid_axis_1 * 4 ∈ [0, 5)> = dimension_mul %1 %5
                in (%4, %6)"
            },
        );
        let inputs = [2, 1, 5].map(|extent| DimensionValue::constant(extent).unwrap());
        assert_eq!(mappings[0].evaluate(&inputs, &[2, 3]).unwrap().starts(), &[0, 0]);
        assert_eq!(mappings[1].evaluate(&inputs, &[6, 8]).unwrap().starts(), &[4, 4]);
        assert_eq!(mappings[2].evaluate(&inputs, &[16]).unwrap().starts(), &[7]);
    }

    #[test]
    fn test_mapped_call_carried_matmul() {
        // Every output tile is zeroed at `k == 0` and then accumulates along the sequential axis `k`.
        let definition = carried_matmul_definition([4, 4, 4], 2, BoundaryPolicy::InBounds);
        let left = Array::matrix(4, 4, (1..=16).map(|value| value as f32).collect()).unwrap();
        let right = Array::matrix(4, 4, (1..=16).rev().map(|value| value as f32).collect()).unwrap();
        let expected = Array::matrix(
            4,
            4,
            vec![80f32, 70., 60., 50., 240., 214., 188., 162., 400., 358., 316., 274., 560., 502., 444., 386.],
        )
        .unwrap();
        assert_eq!(definition.interpret(vec![left, right], 8), Ok(vec![expected]));
    }

    #[test]
    fn test_mapped_call_carried_matmul_masked_edges() {
        // Every step after the first loads the valid elements that the previous step published into its fresh tile.
        let definition = carried_matmul_definition([3, 4, 3], 2, BoundaryPolicy::Masked);
        let left = Array::matrix(3, 4, (1..=12).map(|value| value as f32).collect()).unwrap();
        let right = Array::matrix(4, 3, (1..=12).map(|value| value as f32).collect()).unwrap();
        let expected = Array::matrix(3, 3, vec![70f32, 80., 90., 158., 184., 210., 246., 288., 330.]).unwrap();
        assert_eq!(definition.interpret(vec![left, right], 8), Ok(vec![expected]));
    }

    #[test]
    fn test_mapped_call_element_starts() {
        // The input windows run backwards through the input, which no separable tiling proof describes, so
        // qualification enumerates the grid.
        let grid = Grid::new(vec![GridDimension::new(3usize.into(), GridExecution::Parallel)]).unwrap();
        let reversed = MappingExpression::Constant(4)
            - MappingExpression::GridCoordinate(0) * MappingExpression::Constant(2);
        let operation = mapped_call(
            grid,
            vec![
                (
                    ArrayType::new_static(DataType::I32, [6]),
                    KernelParameterAccess::ReadOnly,
                    ParameterWindow::ElementStarts {
                        block_shape: vec![2],
                        boundary: BoundaryPolicy::InBounds,
                        starts: vec![reversed],
                    },
                ),
                (
                    ArrayType::new_static(DataType::I32, [6]),
                    KernelParameterAccess::WriteOnly,
                    ParameterWindow::BlockIndices {
                        block_shape: vec![2],
                        boundary: BoundaryPolicy::InBounds,
                        indices: vec![MappingExpression::GridCoordinate(0)],
                    },
                ),
            ],
            vec![],
        )
        .unwrap();
        assert_eq!(operation.parameters()[0].mapping().tiling_axes(), None);
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            references[1].write(&references[0].read()?)
        })
        .unwrap();
        let input = Array::vector(vec![0i32, 1, 2, 3, 4, 5]).unwrap();
        assert_eq!(
            definition.interpret(vec![input.clone()], 3),
            Ok(vec![Array::vector(vec![4i32, 5, 2, 3, 0, 1]).unwrap()]),
        );
        assert_eq!(
            definition.interpret(vec![input], 2).unwrap_err().downcast_custom::<KernelInitializationError>(),
            Some(&KernelInitializationError::QualificationLimit { programs: 3, maximum: 2 }),
        );
    }

    #[test]
    fn test_mapped_call_element_starts_arithmetic() {
        // Each start applies one dimension operation to the grid coordinate `g` and the prefetched dimension `p`.
        let coordinate = || MappingExpression::GridCoordinate(0);
        let prefetch = || MappingExpression::Prefetch(0);
        let constant = MappingExpression::Constant;
        let operation = mapped_call(
            Grid::new(vec![GridDimension::new(8usize.into(), GridExecution::Parallel)]).unwrap(),
            vec![(
                ArrayType::new_static(DataType::I32, [64; 8]),
                KernelParameterAccess::ReadOnly,
                ParameterWindow::ElementStarts {
                    block_shape: vec![1; 8],
                    boundary: BoundaryPolicy::InBounds,
                    starts: vec![
                        coordinate() + prefetch(),
                        constant(9) - coordinate(),
                        coordinate().saturating_sub(prefetch()),
                        coordinate() * prefetch(),
                        constant(20) / (prefetch() + constant(1)),
                        constant(23) % (prefetch() + constant(1)),
                        coordinate().min(prefetch()),
                        coordinate().max(prefetch()),
                    ],
                },
            )],
            vec![ArrayType::scalar(DataType::I32)],
        )
        .unwrap();
        let mapping = operation.parameters()[0].mapping();
        let starts = |coordinate: usize, prefetch: usize| {
            let inputs = [coordinate, prefetch].map(|extent| DimensionValue::constant(extent).unwrap());
            mapping.evaluate(&inputs, &[64; 8]).unwrap().starts().to_vec()
        };
        assert_eq!(starts(5, 3), vec![8, 4, 2, 15, 5, 3, 3, 5]);
        assert_eq!(starts(1, 3), vec![4, 8, 0, 3, 5, 3, 1, 3]);
    }

    #[test]
    fn test_mapped_call_prefetch() {
        // The input window starts at the block whose index is the prefetched scalar.
        let operation = mapped_call(
            Grid::new(vec![GridDimension::new(1usize.into(), GridExecution::Parallel)]).unwrap(),
            vec![
                (
                    ArrayType::new_static(DataType::I32, [8]),
                    KernelParameterAccess::ReadOnly,
                    ParameterWindow::BlockIndices {
                        block_shape: vec![2],
                        boundary: BoundaryPolicy::InBounds,
                        indices: vec![MappingExpression::Prefetch(0)],
                    },
                ),
                (
                    ArrayType::new_static(DataType::I32, [2]),
                    KernelParameterAccess::WriteOnly,
                    ParameterWindow::WholeArray,
                ),
            ],
            vec![ArrayType::scalar(DataType::I32)],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            references[1].write(&references[0].read()?)
        })
        .unwrap();
        let input = Array::vector((0..8i32).collect::<Vec<_>>()).unwrap();
        assert_eq!(
            definition.interpret(vec![input.clone(), Array::scalar(1i32).unwrap()], 1),
            Ok(vec![Array::vector(vec![2i32, 3]).unwrap()]),
        );
        let specialized = definition.specialize_prefetch(&[Array::scalar(3i32).unwrap()]).unwrap();
        assert_eq!(specialized.operation().prefetch_types(), &[]);
        assert_eq!(specialized.interpret(vec![input], 1), Ok(vec![Array::vector(vec![6i32, 7]).unwrap()]));
    }

    #[test]
    fn test_mapped_call_two_outputs() {
        // One input tile produces an element type conversion and a thresholded copy in two outputs.
        let window = || ParameterWindow::BlockIndices {
            block_shape: vec![2],
            boundary: BoundaryPolicy::InBounds,
            indices: vec![MappingExpression::GridCoordinate(0)],
        };
        let operation = mapped_call(
            Grid::new(vec![GridDimension::new(2usize.into(), GridExecution::Parallel)]).unwrap(),
            vec![
                (ArrayType::new_static(DataType::F32, [4]), KernelParameterAccess::ReadOnly, window()),
                (ArrayType::new_static(DataType::I32, [4]), KernelParameterAccess::WriteOnly, window()),
                (ArrayType::new_static(DataType::F32, [4]), KernelParameterAccess::WriteOnly, window()),
            ],
            vec![],
        )
        .unwrap();
        let definition = KernelDefinition::<NoKernelExtension>::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let input = references[0].read()?;
            references[1].write(&input.convert_element_type(DataType::I32)?)?;
            let threshold = full::<f32, _>(context, [2], 2.5f32)?;
            let zero = zeros::<f32, _>(context, [2])?;
            let mask = compare(context, &input, &threshold, ComparisonDirection::GreaterThan)?;
            references[2].write(&select(context, &mask, &input, &zero)?)
        })
        .unwrap();
        assert_eq!(
            definition.interpret(vec![Array::vector(vec![1.5f32, 2.0, 3.5, 4.0]).unwrap()], 2),
            Ok(vec![Array::vector(vec![1i32, 2, 3, 4]).unwrap(), Array::vector(vec![0f32, 0.0, 3.5, 4.0]).unwrap()]),
        );
    }

    #[test]
    fn test_mapped_call_read_write() {
        // Each grid point increments its own element of the first parameter and atomically increments the whole second
        // parameter, whose windows overlap across the parallel grid.
        let operation = mapped_call(
            Grid::new(vec![GridDimension::new(3usize.into(), GridExecution::Parallel)]).unwrap(),
            vec![
                (
                    ArrayType::new_static(DataType::I32, [3]),
                    KernelParameterAccess::ReadWrite,
                    ParameterWindow::BlockIndices {
                        block_shape: vec![1],
                        boundary: BoundaryPolicy::InBounds,
                        indices: vec![MappingExpression::GridCoordinate(0)],
                    },
                ),
                (ArrayType::scalar(DataType::I32), KernelParameterAccess::ReadWrite, ParameterWindow::WholeArray),
            ],
            vec![],
        )
        .unwrap();
        assert_eq!(operation.aliases(), vec![(0, 0), (1, 1)]);
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            accumulate(context, &references[0], &ones::<i32, _>(context, [1])?, BoundaryPolicy::InBounds)?;
            references[1].atomic_add_update(&ones::<i32, _>(context, [])?)
        })
        .unwrap();
        assert_eq!(
            definition.interpret(vec![Array::vector(vec![1i32, 2, 3]).unwrap(), Array::scalar(10i32).unwrap()], 3),
            Ok(vec![Array::vector(vec![2i32, 3, 4]).unwrap(), Array::scalar(13i32).unwrap()]),
        );
    }

    #[test]
    fn test_mapped_call_invalid() {
        let grid = || Grid::new(vec![GridDimension::new(2usize.into(), GridExecution::Parallel)]).unwrap();
        let vector = ArrayType::new_static(DataType::I32, [4]);
        let window = |indices: Vec<MappingExpression>| ParameterWindow::BlockIndices {
            block_shape: vec![2],
            boundary: BoundaryPolicy::InBounds,
            indices,
        };
        assert_eq!(
            mapped_call(
                grid(),
                vec![(
                    vector.clone(),
                    KernelParameterAccess::ReadOnly,
                    window(vec![MappingExpression::GridCoordinate(1)]),
                )],
                vec![],
            ),
            Err(KernelError::Type(TypeError::invalid(
                "kernel mapping expression references grid axis 1, but the grid has rank 1",
            ))),
        );
        assert_eq!(
            mapped_call(
                grid(),
                vec![(vector.clone(), KernelParameterAccess::ReadOnly, window(vec![MappingExpression::Prefetch(0)]))],
                vec![],
            ),
            Err(KernelError::Type(TypeError::invalid(
                "kernel mapping expression references scalar prefetch 0, but the call has 0 scalar prefetch values",
            ))),
        );
        assert_eq!(
            mapped_call(
                grid(),
                vec![(
                    vector,
                    KernelParameterAccess::ReadOnly,
                    window(vec![MappingExpression::GridCoordinate(0), MappingExpression::GridCoordinate(0)]),
                )],
                vec![],
            ),
            Err(KernelError::Mapping(BlockMappingError::RankMismatch { boundary: "output", expected: 1, actual: 2 })),
        );
        let dynamic = ArrayType::new(
            DataType::I32,
            Shape::new(vec![DimensionVariable::new("extent", DimensionBounds::non_negative(Some(4)).unwrap()).into()]),
        );
        assert_eq!(
            mapped_call(grid(), vec![(dynamic, KernelParameterAccess::ReadOnly, ParameterWindow::WholeArray)], vec![]),
            Err(KernelError::Type(TypeError::invalid(
                "kernel parameter 0 with a whole array window requires a static shape",
            ))),
        );
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
            KernelError::Type(TypeError::invalid("kernel output tile rank must equal its array rank")),
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
                Array::from_elements(ArrayType::new_static(DataType::I32, [2, 2]), &[2i32, 3, 7, 8]).unwrap(),
            )]),
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
    fn test_masked_load() {
        // The input window extends past its array, so its padding elements take the fill value as well.
        let operation = mapped_call(
            Grid::new(vec![GridDimension::new(1usize.into(), GridExecution::Parallel)]).unwrap(),
            vec![
                (
                    ArrayType::new_static(DataType::I32, [2]),
                    KernelParameterAccess::ReadOnly,
                    ParameterWindow::BlockIndices {
                        block_shape: vec![4],
                        boundary: BoundaryPolicy::Masked,
                        indices: vec![MappingExpression::GridCoordinate(0)],
                    },
                ),
                (
                    ArrayType::new_static(DataType::I32, [4]),
                    KernelParameterAccess::WriteOnly,
                    ParameterWindow::WholeArray,
                ),
            ],
            vec![],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let mask = context.lift(ArrayIrValue::Array(Array::vector(vec![true, false, true, true])?))?;
            references[1].write(&masked_load(context, &references[0], &mask, 7i32)?)
        })
        .unwrap();
        assert_eq!(
            definition.interpret(vec![Array::vector(vec![1i32, 2]).unwrap()], 1),
            Ok(vec![Array::vector(vec![1i32, 7, 7, 7]).unwrap()]),
        );
    }

    #[test]
    fn test_masked_load_or() {
        let operation = mapped_call(
            Grid::new(vec![GridDimension::new(2usize.into(), GridExecution::Parallel)]).unwrap(),
            vec![
                (
                    ArrayType::new_static(DataType::I32, [3]),
                    KernelParameterAccess::ReadOnly,
                    ParameterWindow::BlockIndices {
                        block_shape: vec![2],
                        boundary: BoundaryPolicy::Masked,
                        indices: vec![MappingExpression::GridCoordinate(0)],
                    },
                ),
                (
                    ArrayType::new_static(DataType::I32, [4]),
                    KernelParameterAccess::WriteOnly,
                    ParameterWindow::BlockIndices {
                        block_shape: vec![2],
                        boundary: BoundaryPolicy::InBounds,
                        indices: vec![MappingExpression::GridCoordinate(0)],
                    },
                ),
            ],
            vec![],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation.clone(), |(references, _coordinates)| {
            references[1].write(&masked_load_or(references[0].context(), &references[0], -1i32)?)
        })
        .unwrap();
        assert_eq!(
            definition.interpret(vec![Array::vector(vec![1i32, 2, 3]).unwrap()], 2),
            Ok(vec![Array::vector(vec![1i32, 2, 3, -1]).unwrap()]),
        );

        // Masked accesses require a reference.
        let result = KernelDefinition::<NoKernelExtension>::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let value = references[0].read()?;
            references[1].write(&masked_load_or(context, &value, -1i32)?)
        });
        assert_eq!(
            result.map(|_| ()),
            Err(KernelError::Program(ProgramError::Type(TypeError::invalid(
                "kernel memory access requires a reference but got `i32[2]`",
            )))),
        );
    }

    #[test]
    fn test_masked_store() {
        // A store under the validity mask of its own window initializes the masked write-only output.
        let operation = mapped_call(
            Grid::new(vec![GridDimension::new(2usize.into(), GridExecution::Parallel)]).unwrap(),
            vec![(
                ArrayType::new_static(DataType::I32, [3]),
                KernelParameterAccess::WriteOnly,
                ParameterWindow::BlockIndices {
                    block_shape: vec![2],
                    boundary: BoundaryPolicy::Masked,
                    indices: vec![MappingExpression::GridCoordinate(0)],
                },
            )],
            vec![],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let value = context.lift(ArrayIrValue::Array(Array::vector(vec![5i32, 7])?))?;
            let mask = window_mask(context, &references[0])?;
            masked_store(context, &references[0], &value, &mask)
        })
        .unwrap();
        assert_eq!(definition.interpret(vec![], 2), Ok(vec![Array::vector(vec![5i32, 7, 5]).unwrap()]));
    }

    #[test]
    fn test_masked_swap() {
        // The swap replaces only the selected valid elements and returns `other` for every other element.
        let window = || ParameterWindow::BlockIndices {
            block_shape: vec![4],
            boundary: BoundaryPolicy::Masked,
            indices: vec![MappingExpression::GridCoordinate(0)],
        };
        let r#type = ArrayType::new_static(DataType::I32, [5]);
        let operation = mapped_call(
            Grid::new(vec![GridDimension::new(2usize.into(), GridExecution::Parallel)]).unwrap(),
            vec![
                (r#type.clone(), KernelParameterAccess::ReadWrite, window()),
                (r#type, KernelParameterAccess::WriteOnly, window()),
            ],
            vec![],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let mask = context.lift(ArrayIrValue::Array(Array::vector(vec![true, false, true, true])?))?;
            let replacement = context.lift(ArrayIrValue::Array(Array::vector(vec![9i32, 10, 11, 12])?))?;
            let old = masked_swap(context, &references[0], &replacement, &mask, -1i32)?;
            references[1].write(&old)
        })
        .unwrap();
        assert_eq!(
            definition.interpret(vec![Array::vector(vec![1i32, 2, 3, 4, 5]).unwrap()], 2),
            Ok(vec![
                Array::vector(vec![9i32, 2, 11, 12, 9]).unwrap(),
                Array::vector(vec![1i32, -1, 3, 4, 5]).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_window_mask() {
        // The input window of the second grid point has one valid element, and every output window is in bounds.
        let window = |boundary| ParameterWindow::BlockIndices {
            block_shape: vec![2],
            boundary,
            indices: vec![MappingExpression::GridCoordinate(0)],
        };
        let operation = mapped_call(
            Grid::new(vec![GridDimension::new(2usize.into(), GridExecution::Parallel)]).unwrap(),
            vec![
                (
                    ArrayType::new_static(DataType::I32, [3]),
                    KernelParameterAccess::ReadOnly,
                    window(BoundaryPolicy::Masked),
                ),
                (
                    ArrayType::new_static(DataType::Boolean, [4]),
                    KernelParameterAccess::WriteOnly,
                    window(BoundaryPolicy::InBounds),
                ),
            ],
            vec![],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            references[1].write(&window_mask(references[0].context(), &references[0])?)
        })
        .unwrap();
        assert_eq!(
            definition.interpret(vec![Array::vector(vec![1i32, 2, 3]).unwrap()], 2),
            Ok(vec![Array::vector(vec![true, true, true, false]).unwrap()]),
        );
    }

    #[test]
    fn test_accumulate() {
        let operation = mapped_call(
            Grid::new(vec![GridDimension::new(3usize.into(), GridExecution::Parallel)]).unwrap(),
            vec![(
                ArrayType::new_static(DataType::I32, [3]),
                KernelParameterAccess::ReadWrite,
                ParameterWindow::BlockIndices {
                    block_shape: vec![1],
                    boundary: BoundaryPolicy::InBounds,
                    indices: vec![MappingExpression::GridCoordinate(0)],
                },
            )],
            vec![],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            accumulate(context, &references[0], &full::<i32, _>(context, [1], 10)?, BoundaryPolicy::InBounds)
        })
        .unwrap();
        assert_eq!(
            definition.body().to_string(),
            indoc! {"
                lambda %0:ref<i32[1]>, %1:dimension<program_0 ∈ [0, 3)> .
                let %2:i32[] = constant [value=10]
                    %3:i32[1] = broadcast [output_type=i32[1], output_axes=[]] %2
                    () = reference_add_update %0 %3
                in ()"
            },
        );
        assert_eq!(
            definition.interpret(vec![Array::vector(vec![1i32, 2, 3]).unwrap()], 3),
            Ok(vec![Array::vector(vec![11i32, 12, 13]).unwrap()]),
        );
    }

    #[test]
    fn test_accumulate_masked() {
        // The edge window of the second grid point loads, adds, and publishes only its one valid element.
        let operation = mapped_call(
            Grid::new(vec![GridDimension::new(2usize.into(), GridExecution::Parallel)]).unwrap(),
            vec![(
                ArrayType::new_static(DataType::I32, [3]),
                KernelParameterAccess::ReadWrite,
                ParameterWindow::BlockIndices {
                    block_shape: vec![2],
                    boundary: BoundaryPolicy::Masked,
                    indices: vec![MappingExpression::GridCoordinate(0)],
                },
            )],
            vec![],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            accumulate(context, &references[0], &full::<i32, _>(context, [2], 10)?, BoundaryPolicy::Masked)
        })
        .unwrap();
        assert_eq!(
            definition.body().to_string(),
            indoc! {"
                lambda %0:ref<i32[2]>, %1:dimension<program_0 ∈ [0, 2)> .
                let %2:i32[] = constant [value=10]
                    %3:i32[2] = broadcast [output_type=i32[2], output_axes=[]] %2
                    %4:bool[2] = const [true, true]
                    %5:i32[2] = zero [type=i32[2]]
                    %6:i32[2] = masked_load %0 %4 %5
                    %7:i32[2] = add %6 %3
                    () = masked_store %0 %7 %4
                in ()"
            },
        );
        assert_eq!(
            definition.interpret(vec![Array::vector(vec![1i32, 2, 3]).unwrap()], 2),
            Ok(vec![Array::vector(vec![11i32, 12, 13]).unwrap()]),
        );
    }

    #[test]
    fn test_scratch() {
        let operation = mapped_call(
            Grid::new(vec![]).unwrap(),
            vec![
                (ArrayType::scalar(DataType::I32), KernelParameterAccess::ReadOnly, ParameterWindow::WholeArray),
                (ArrayType::scalar(DataType::I32), KernelParameterAccess::WriteOnly, ParameterWindow::WholeArray),
            ],
            vec![],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation.clone(), |(references, _coordinates)| {
            let storage = scratch::<i32, _>(references[0].context(), [], 4)?;
            storage.write(&references[0].read()?)?;
            references[1].write(&storage.read()?)
        })
        .unwrap();
        assert_eq!(
            definition.interpret(vec![Array::scalar(23i32).unwrap()], 1),
            Ok(vec![Array::scalar(23i32).unwrap()]),
        );

        // The alignment must be a positive power of two.
        let result = KernelDefinition::<NoKernelExtension>::trace(operation, |(references, _coordinates)| {
            scratch::<i32, _>(references[0].context(), [], 3)?;
            Ok(())
        });
        assert_eq!(
            result.map(|_| ()),
            Err(KernelError::Program(ProgramError::custom(KernelMemoryError::InvalidAlignment { alignment: 3 }))),
        );
    }

    #[test]
    fn test_async_copy() {
        let vector = ArrayType::new_static(DataType::I32, [2]);
        let operation = mapped_call(
            Grid::new(vec![]).unwrap(),
            vec![
                (vector.clone(), KernelParameterAccess::ReadOnly, ParameterWindow::WholeArray),
                (vector, KernelParameterAccess::WriteOnly, ParameterWindow::WholeArray),
            ],
            vec![],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            let context = references[0].context();
            let token = async_copy(context, &references[0], &references[1])?;
            wait(context, &token)
        })
        .unwrap();
        let input = Array::vector(vec![17i32, 29]).unwrap();
        assert_eq!(definition.interpret(vec![input.clone()], 1), Ok(vec![input]));
    }

    #[test]
    fn test_wait() {
        // Only the token of an asynchronous copy completes one.
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::I32, [2])));
        let expected = WaitOperation.infer_output_types(std::slice::from_ref(&reference_type), &[]).unwrap_err();
        let result = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<_>| {
                wait(inputs[0].context(), &inputs[0])?;
                Ok(Vec::<Tracer<TracingContext<ArrayIrValue<Array>, KernelOperation>>>::new())
            },
            vec![reference_type],
        );
        assert_eq!(result.map(|_| ()), Err(ProgramError::Type(expected)));
    }

    #[test]
    fn test_compare() {
        // Arrays broadcast through the array comparison, and dimensions compare their extents.
        let (_, program) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<_>| {
                let context = inputs[0].context();
                Ok(vec![
                    compare(context, &inputs[0], &inputs[1], ComparisonDirection::LessThan)?,
                    compare(context, &inputs[2], &inputs[3], ComparisonDirection::GreaterThan)?,
                ])
            },
            vec![
                ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])),
                ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
                ArrayIrType::Dimension(DimensionType::new("left", DimensionBounds::non_negative(None).unwrap())),
                ArrayIrType::Dimension(DimensionType::new("right", DimensionBounds::non_negative(None).unwrap())),
            ],
        )
        .unwrap();
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1f32, 2.0, 3.0]).unwrap()),
                ArrayIrValue::Array(Array::scalar(2.5f32).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
                ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![true, true, false]).unwrap()),
                ArrayIrValue::Array(Array::scalar(true).unwrap()),
            ]),
        );

        // An array and a dimension do not compare.
        let input_types = vec![
            ArrayIrType::Array(ArrayType::scalar(DataType::I64)),
            ArrayIrType::Dimension(DimensionType::new("extent", DimensionBounds::non_negative(None).unwrap())),
        ];
        let expected = ArrayIrOperation::<Array>::Compare(CompareOperation::new(ComparisonDirection::Equal))
            .infer_output_types(&input_types, &[])
            .unwrap_err();
        let result = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<_>| {
                Ok(vec![compare(inputs[0].context(), &inputs[0], &inputs[1], ComparisonDirection::Equal)?])
            },
            input_types,
        );
        assert_eq!(result.map(|_| ()), Err(ProgramError::Type(expected)));
    }

    #[test]
    fn test_select() {
        let (_, program) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<_>| Ok(vec![select(inputs[0].context(), &inputs[0], &inputs[1], &inputs[2])?]),
            vec![
                ArrayIrType::Array(ArrayType::new_static(DataType::Boolean, [3])),
                ArrayIrType::Array(ArrayType::new_static(DataType::I32, [3])),
                ArrayIrType::Array(ArrayType::scalar(DataType::I32)),
            ],
        )
        .unwrap();
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![true, false, true]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![1i32, 2, 3]).unwrap()),
                ArrayIrValue::Array(Array::scalar(-1i32).unwrap()),
            ]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![1i32, -1, 3]).unwrap())]),
        );
    }

    #[test]
    fn test_compare_dimension() {
        // Each direction compares the dimension value 1 with the constant 1.
        for (direction, expected) in [
            (ComparisonDirection::Equal, true),
            (ComparisonDirection::NotEqual, false),
            (ComparisonDirection::LessThan, false),
            (ComparisonDirection::LessThanOrEqual, true),
            (ComparisonDirection::GreaterThan, false),
            (ComparisonDirection::GreaterThanOrEqual, true),
        ] {
            let (_, program) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
                |inputs: Vec<_>| Ok(vec![compare_dimension(inputs[0].context(), &inputs[0], direction, 1)?]),
                vec![ArrayIrType::Dimension(DimensionType::new(
                    "coordinate",
                    DimensionBounds::non_negative(Some(4)).unwrap(),
                ))],
            )
            .unwrap();
            assert_eq!(
                program.interpret(vec![ArrayIrValue::Dimension(DimensionValue::constant(1).unwrap())]),
                Ok(vec![ArrayIrValue::Array(Array::scalar(expected).unwrap())]),
            );
        }

        // The comparison of a grid coordinate with a constant stays a staged instruction over a constant atom.
        let operation = mapped_call(
            Grid::new(vec![GridDimension::new(2usize.into(), GridExecution::Sequential)]).unwrap(),
            vec![(ArrayType::scalar(DataType::I32), KernelParameterAccess::ReadOnly, ParameterWindow::WholeArray)],
            vec![],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, coordinates)| {
            compare_dimension(references[0].context(), &coordinates[0], ComparisonDirection::Equal, 0)?;
            Ok(())
        })
        .unwrap();
        assert_eq!(
            definition.body().to_string(),
            indoc! {"
                lambda %0:ref<i32[]>, %1:dimension<program_0 ∈ [0, 2)> .
                let %2:dimension<0> = const 0
                    %3:bool[] = compare [direction=Equal] %1 %2
                in ()"
            },
        );
    }

    #[test]
    fn test_reference_slice() {
        let operation = mapped_call(
            Grid::new(vec![]).unwrap(),
            vec![
                (
                    ArrayType::new_static(DataType::I32, [2, 3]),
                    KernelParameterAccess::ReadOnly,
                    ParameterWindow::WholeArray,
                ),
                (
                    ArrayType::new_static(DataType::I32, [2, 2]),
                    KernelParameterAccess::WriteOnly,
                    ParameterWindow::WholeArray,
                ),
            ],
            vec![],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation.clone(), |(references, _coordinates)| {
            let view = reference_slice(&references[0], &[None, Some(1..3)])?;
            let value: Tracer<_> = view.read()?;
            references[1].write(&value)
        })
        .unwrap();
        assert_eq!(
            definition.interpret(vec![Array::matrix(2, 3, vec![1i32, 2, 3, 4, 5, 6]).unwrap()], 1),
            Ok(vec![Array::matrix(2, 2, vec![2i32, 3, 5, 6]).unwrap()]),
        );

        // Slices take one non-decreasing range per axis.
        let result = KernelDefinition::<NoKernelExtension>::trace(operation.clone(), |(references, _coordinates)| {
            reference_slice(&references[0], &[None])?;
            Ok(())
        });
        assert_eq!(
            result.map(|_| ()),
            Err(KernelError::Program(ProgramError::Type(TypeError::invalid(
                "kernel slice has 1 ranges but its reference has rank 2",
            )))),
        );
        let result = KernelDefinition::<NoKernelExtension>::trace(operation, |(references, _coordinates)| {
            reference_slice(&references[0], &[None, Some(Range { start: 2, end: 1 })])?;
            Ok(())
        });
        assert_eq!(
            result.map(|_| ()),
            Err(KernelError::Program(ProgramError::Type(TypeError::invalid(
                "kernel slice range `2..1` is decreasing",
            )))),
        );
    }

    #[test]
    fn test_for_loop() {
        let calls = Cell::new(0);
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
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:i64[] .
                let %1:i64[] = const 0
                    %2:i64[], %3:i64[] = while [iteration_bound=3] %1 %0 [
                        condition={
                            lambda %0:i64[], %1:i64[] .
                            let %2:i64[] = const 3
                                %3:bool[] = compare [direction=LessThan] %0 %2
                            in (%3)
                        },
                        body={
                            lambda %0:i64[], %1:i64[] .
                            let %2:i64[] = add %1 %0
                                %3:i64[] = const 1
                                %4:i64[] = add %0 %3
                            in (%4, %2)
                        },
                    ]
                in (%3)"
            },
        );
        assert_eq!(
            program.interpret(vec![ArrayIrValue::Array(Array::scalar(5i64).unwrap())]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(8i64).unwrap())]),
        );
    }

    #[test]
    fn test_for_loop_empty_range_still_traces_once() {
        let calls = Cell::new(0);
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
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:i64[] .
                let %1:i64[] = const 0
                    %2:i64[], %3:i64[] = while [iteration_bound=1] %1 %0 [
                        condition={
                            lambda %0:i64[], %1:i64[] .
                            let %2:i64[] = const 0
                                %3:bool[] = compare [direction=LessThan] %0 %2
                            in (%3)
                        },
                        body={
                            lambda %0:i64[], %1:i64[] .
                            let %2:i64[] = const 1
                                %3:i64[] = add %0 %2
                            in (%3, %1)
                        },
                    ]
                in (%3)"
            },
        );
        assert_eq!(
            program.interpret(vec![ArrayIrValue::Array(Array::scalar(5i64).unwrap())]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(5i64).unwrap())]),
        );
    }

    #[test]
    fn test_shape_div_ceil() {
        assert_eq!(shape_div_ceil(33, 32), Ok(2));
        assert_eq!(shape_div_ceil(0, 32), Ok(0));
        assert_eq!(
            shape_div_ceil(1, 0),
            Err(KernelError::Type(TypeError::invalid("kernel shape ceiling division requires a positive divisor"))),
        );
    }

    #[test]
    fn test_zeros() {
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(ArrayType::new_static(DataType::F32, [2, 3]), KernelParameterAccess::WriteOnly)
                    .unwrap(),
            ],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            let value = zeros::<f32, _>(references[0].context(), [2, 3])?;
            references[0].write(&value)
        })
        .unwrap();
        assert_eq!(
            definition.body().to_string(),
            indoc! {"
                lambda %0:ref<f32[2, 3]> .
                let %1:f32[2, 3] = zero [type=f32[2, 3]]
                    () = reference_write %0 %1
                in ()"
            },
        );
        assert_eq!(
            definition.interpret(vec![], 1).unwrap(),
            vec![Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[0.0f32; 6]).unwrap()],
        );
    }

    #[test]
    fn test_ones() {
        let operation = mapped_call(
            Grid::new(vec![]).unwrap(),
            vec![(
                ArrayType::new_static(DataType::F32, [2, 3]),
                KernelParameterAccess::WriteOnly,
                ParameterWindow::WholeArray,
            )],
            vec![],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            references[0].write(&ones::<f32, _>(references[0].context(), [2, 3])?)
        })
        .unwrap();
        assert_eq!(
            definition.body().to_string(),
            indoc! {"
                lambda %0:ref<f32[2, 3]> .
                let %1:f32[2, 3] = one [type=f32[2, 3]]
                    () = reference_write %0 %1
                in ()"
            },
        );
        assert_eq!(definition.interpret(vec![], 1), Ok(vec![Array::matrix(2, 3, vec![1f32; 6]).unwrap()]));
    }

    #[test]
    fn test_full() {
        let operation = mapped_call(
            Grid::new(vec![]).unwrap(),
            vec![(
                ArrayType::new_static(DataType::I32, [2, 3]),
                KernelParameterAccess::WriteOnly,
                ParameterWindow::WholeArray,
            )],
            vec![],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            references[0].write(&full::<i32, _>(references[0].context(), [2, 3], 7)?)
        })
        .unwrap();
        assert_eq!(
            definition.body().to_string(),
            indoc! {"
                lambda %0:ref<i32[2, 3]> .
                let %1:i32[] = constant [value=7]
                    %2:i32[2, 3] = broadcast [output_type=i32[2, 3], output_axes=[]] %1
                    () = reference_write %0 %2
                in ()"
            },
        );
        assert_eq!(definition.interpret(vec![], 1), Ok(vec![Array::matrix(2, 3, vec![7i32; 6]).unwrap()]));
    }

    #[test]
    fn test_iota() {
        let operation = mapped_call(
            Grid::new(vec![]).unwrap(),
            vec![(
                ArrayType::new_static(DataType::I32, [2, 3]),
                KernelParameterAccess::WriteOnly,
                ParameterWindow::WholeArray,
            )],
            vec![],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _coordinates)| {
            references[0].write(&iota::<i32, _>(references[0].context(), [2, 3], 1)?)
        })
        .unwrap();
        assert_eq!(
            definition.body().to_string(),
            indoc! {"
                lambda %0:ref<i32[2, 3]> .
                let %1:i32[2, 3] = iota [type=i32[2, 3], dimension=1]
                    () = reference_write %0 %1
                in ()"
            },
        );
        assert_eq!(definition.interpret(vec![], 1), Ok(vec![Array::matrix(2, 3, vec![0i32, 1, 2, 0, 1, 2]).unwrap()]));
    }

    #[test]
    fn test_dot() {
        let operation = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(ArrayType::new_static(DataType::F32, [2, 3]), KernelParameterAccess::ReadOnly)
                    .unwrap(),
                whole_array_parameter(ArrayType::new_static(DataType::F32, [3, 2]), KernelParameterAccess::ReadOnly)
                    .unwrap(),
                whole_array_parameter(ArrayType::new_static(DataType::F32, [2, 2]), KernelParameterAccess::WriteOnly)
                    .unwrap(),
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
