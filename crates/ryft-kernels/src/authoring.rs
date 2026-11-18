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
/// [`DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS`] as the maximum grid size to enumerate during qualification.
/// Qualification enumerates the grid when the mapping of a parameter is not recognized by
/// [`BlockMapping::tiling_axes`] or when its recognized axes repeat a grid axis (e.g., the diagonal element starts
/// `(i, i)`, for which `tiling_axes` returns `Some`). Grid points are otherwise not capped, and replay has no step
/// limit because kernel bodies admit only bounded loops.
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
    input_types
        .extend((0..prefetch_types.len()).map(|index| {
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
/// the grid coordinates as the block indices of the output. The output is the parameter at `output_index` and the
/// inputs keep their order around it, so the body references of an annotated kernel follow its declaration order.
#[doc(hidden)]
pub fn tiled_call(
    input_types: &[ArrayType],
    output_type: ArrayType,
    output_index: usize,
    block_shape: Vec<usize>,
    boundary: BoundaryPolicy,
) -> Result<KernelCallOperation, KernelError> {
    if output_index > input_types.len() {
        return Err(TypeError::invalid("kernel output index exceeds its parameter count").into());
    }
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
    let window = ParameterWindow::BlockIndices { block_shape, boundary, indices };
    parameters.insert(output_index, (output_type, KernelParameterAccess::WriteOnly, window));
    mapped_call(grid, parameters, vec![])
}

/// Stages a [`TileLoadOperation`] that loads one masked tile of the reference `source`, translating tile indices into
/// checked element starts with [`DimensionMulOperation`]s. The elements of the tile that lie outside `source` take the
/// value `other`.
///
/// Every nonnegative tile index is accepted, whether it is a dimension (e.g., a grid coordinate) or an integer scalar
/// array (e.g., a loop index), and a tile that lies entirely beyond `source` contains only `other`. Every index at or
/// beyond the number of tiles that cover an axis selects such a tile, so an index that its type does not bound by that
/// number is first clamped to it, with a [`DimensionMinOperation`] for a dimension and with a [`CompareOperation`] and
/// a [`SelectOperation`] for a scalar. Every tile of a physically empty `source` contains only `other`, and a tile
/// without elements is empty at every index, so the indices of both are clamped to zero. The padding value `other`
/// takes the memory placement of `source`. The clamp preserves the result. It also keeps the multiplication by the tile
/// extent from overflowing and bounds the element starts so that the window limit provably fits the canonical dimension
/// width, whenever the limit of the first tile beyond `source` (i.e., its start plus the tile extent) fits that width.
/// It does not for a source that is not empty and reaches within one tile of that width, which no backend can
/// materialize, or for a tile that is longer than half that width. In those cases the multiplication or the tile load
/// keeps its runtime check, which reports an index that the clamp could not bound. Scalar indices then enter the
/// dimension universe through a checked [`DimensionFromScalarOperation`], which rejects negative values. The tile load
/// is refined with [`TileLoadOperation::with_input_types`], so it declares no ordered assertion whenever the bounded
/// element starts prove its windows valid.
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
    let empty = shape.dimensions().contains(&0) || block_shape.contains(&0);
    for (axis, index) in indices.iter().enumerate() {
        // Every index at or beyond `limit` selects a tile that lies entirely beyond the source. A tile without elements
        // is empty at every index, and every tile of a physically empty source contains only `other`, so both clamp
        // every index along every axis to zero. This also keeps the start of the first tile beyond a source axis whose
        // extent is near the canonical dimension width, or of a very long tile axis, from exceeding that width.
        let limit = if empty { 0 } else { shape.dimensions()[axis].div_ceil(block_shape[axis]) };
        let dimension = match index.r#type().as_ref() {
            ArrayIrType::Dimension(r#type) if r#type.maximum_extent().is_some_and(|maximum| maximum <= limit) => {
                index.clone()
            }
            ArrayIrType::Dimension(r#type) => {
                let limit = DimensionValue::constant(limit)?;
                let operation = DimensionMinOperation::new(r#type, limit.r#type().as_ref())?;
                let limit = context.lift(ArrayIrValue::Dimension(limit))?;
                context.bind(ArrayIrOperation::from(operation), vec![], &[index.clone(), limit])?.remove(0)
            }
            ArrayIrType::Array(r#type) if r#type.rank() == 0 && r#type.data_type().is_integer() => {
                // The clamp is needed only when `limit` is representable in the index type, which is exactly when it
                // survives a round trip through that type. The staged limit takes the complete index type, so that
                // the comparison and the selection see operands with the same placement.
                let limit_value = Array::scalar(limit as u64)?;
                let typed_limit = limit_value.converted_to(r#type.data_type())?;
                let index = if typed_limit.converted_to(DataType::U64)? == limit_value {
                    let typed_limit = Array::from_logical_bytes(r#type.clone(), &typed_limit.logical_bytes())?;
                    let typed_limit = context.lift(ArrayIrValue::Array(typed_limit))?;
                    let below = context
                        .bind(
                            ArrayIrOperation::from(ArrayOperation::Compare(CompareOperation::new(
                                ComparisonDirection::LessThan,
                            ))),
                            vec![],
                            &[index.clone(), typed_limit.clone()],
                        )?
                        .remove(0);
                    let operation = ArrayIrOperation::<Array>::from(ArrayOperation::Select(SelectOperation::new()));
                    context.bind(operation, vec![], &[below, index.clone(), typed_limit])?.remove(0)
                } else {
                    index.clone()
                };
                let bounds = DimensionBounds::new(0, limit.checked_add(1))?;
                let variable = DimensionVariable::new(format!("tile_index_{axis}"), bounds);
                context
                    .bind(
                        ArrayIrOperation::DimensionFromScalar(DimensionFromScalarOperation::new(variable.into())),
                        vec![],
                        &[index],
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
    // The padding value takes the memory placement of the source, which the type rule of the tile load requires.
    let other = Array::scalar(other)?;
    let other_type = other.r#type().into_owned().with_memory(reference.referent().memory());
    inputs.push(context.lift(ArrayIrValue::Array(Array::from_logical_bytes(other_type, &other.logical_bytes())?))?);
    let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
    let operation = TileLoadOperation::new(block_shape, BoundaryPolicy::Masked)?.with_input_types(&input_types)?;
    Ok(context.bind(operation, vec![], &inputs)?.remove(0))
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
/// returns `other` for every other element. `mask` is a Boolean array with the shape of the referent, and `other` is
/// staged as a rank-0 constant that a [`BroadcastOperation`] expands to the referent type, so `T` must be the element
/// type of the referent and no host tile is allocated while tracing. Under kernel qualification, the mask is also
/// intersected with the validity of the window, so padding elements of a masked window take the value `other` as well.
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
    let other = broadcast_scalar(context, Array::scalar(other)?, referent)?;
    Ok(context
        .bind(MaskedLoadOperation::new(), vec![], &[reference.clone(), mask.clone(), other])?
        .remove(0))
}

/// Stages a [`MaskedLoadOperation`] that loads the whole reference `reference` under a broadcast constant mask that
/// selects every element, which is the load of a window whose padding elements take the value `other`. Under kernel
/// qualification, that mask is intersected with the validity of the window, so exactly the valid elements are read.
/// `T` must be the element type of the referent.
pub fn masked_load_or<T: ArrayElement, C>(context: &C, reference: &C::Value, other: T) -> Result<C::Value, ProgramError>
where
    C: Context<Type = ArrayIrType, Constant = ArrayIrValue<Array>, Operation = KernelOperation>,
{
    let referent = static_referent(reference)?;
    let mask_type = ArrayType::new(DataType::Boolean, referent.shape().clone());
    let mask = broadcast_scalar(context, Array::scalar(true)?, mask_type)?;
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
/// `value` has exactly the referent type, `mask` is a Boolean array with the referent shape, and `other` is staged as a
/// rank-0 constant that a [`BroadcastOperation`] expands to the referent type, so `T` must be the element type of the
/// referent.
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
    let other = broadcast_scalar(context, Array::scalar(other)?, referent)?;
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
/// padding elements are zero, added to `value`, and stored back through a [`MaskedStoreOperation`]. Both use one
/// broadcast constant mask that selects every element, and kernel qualification restricts it to the valid elements of
/// the window. Under either policy, `value` broadcasts and promotes as in an [`AddOperation`] of the window and
/// `value`, so a rank-0 value, a value with unit extents, or a value with a narrower element type is added to every
/// element of the window. The sum must have exactly the referent type, and any other sum fails when it is staged, with
/// the type error of [`ReferenceAddUpdateOperation`] for an in-bounds window and of [`MaskedStoreOperation`] for a
/// masked window.
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
    let mask_type = ArrayType::new(DataType::Boolean, referent.shape().clone());
    let mask = broadcast_scalar(context, Array::scalar(true)?, mask_type)?;
    let zero = ArrayIrOperation::<Array>::from(ArrayOperation::Zero(ZeroOperation::new(referent)));
    let zero = context.bind(zero, vec![], &[])?.remove(0);
    let current = context
        .bind(MaskedLoadOperation::new(), vec![], &[reference.clone(), mask.clone(), zero])?
        .remove(0);
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
    broadcast_scalar(context, Array::scalar(value)?, ArrayType::new_static(T::data_type(), shape))
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
            return inputs[grid_rank..].get(*index).copied().ok_or_else(|| {
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

/// Stages the rank-0 array `value` as a [`ConstantOperation`] and expands it to `type` with a [`BroadcastOperation`],
/// whose canonical type rules check the data type and memory placement and keep the shape, layout, and sharding of
/// `type`. The constant takes the memory placement of `type` and a rank-0 sharding with the mesh, reduction state, and
/// varying manual axes of `type`, because broadcasting cannot introduce variation along a manual mesh axis. Every
/// element of the result has the exact encoding of `value`. No host array of the shape of `type` is allocated, and an
/// interpreter that materializes the result checks its storage size.
fn broadcast_scalar<C: Context<Type = ArrayIrType>>(
    context: &C,
    value: Array,
    r#type: ArrayType,
) -> Result<C::Value, ProgramError>
where
    C::Operation: From<ArrayIrOperation<Array>>,
{
    let sharding = r#type
        .sharding()
        .map(|sharding| sharding.with_dimensions(Vec::new()))
        .transpose()
        .map_err(|error| TypeError::invalid(error.to_string()))?;
    let value_type = value
        .r#type()
        .into_owned()
        .with_memory(r#type.memory())
        .with_sharding(sharding)
        .map_err(|error| TypeError::invalid(error.to_string()))?;
    let value = Array::from_logical_bytes(value_type, &value.logical_bytes())?;
    let constant = ArrayIrOperation::<Array>::from(ArrayOperation::Constant(ConstantOperation::new(value)));
    let constant = context.bind(constant, Vec::new(), &[])?.remove(0);
    let broadcast =
        ArrayIrOperation::<Array>::from(ArrayOperation::Broadcast(BroadcastOperation::new(r#type, Vec::new())));
    Ok(context.bind(broadcast, Vec::new(), &[constant])?.remove(0))
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use indoc::{formatdoc, indoc};
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayReference, Condition, ConvertElementType, DataType, Layout, LogicalMesh, MAX_DIMENSION_EXTENT, Memory,
        MeshAxis, MeshAxisType, Operation, Program, ReferenceAtomicAddUpdate, ReferenceRead, ReferenceType,
        ReferenceWrite, Shape, Sharding, TiledLayout, TracingContext,
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
        let reversed =
            MappingExpression::Constant(4) - MappingExpression::GridCoordinate(0) * MappingExpression::Constant(2);
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
    fn test_mapped_call_invalid_prefetch_index() {
        // The prefetch index is checked against the prefetch suffix, so it cannot overflow into the grid coordinates.
        let grid = Grid::new(vec![GridDimension::new(2usize.into(), GridExecution::Parallel)]).unwrap();
        let window = ParameterWindow::BlockIndices {
            block_shape: vec![2],
            boundary: BoundaryPolicy::InBounds,
            indices: vec![MappingExpression::Prefetch(usize::MAX)],
        };
        assert_eq!(
            mapped_call(
                grid,
                vec![(ArrayType::new_static(DataType::I32, [4]), KernelParameterAccess::ReadOnly, window)],
                vec![ArrayType::scalar(DataType::I32)],
            ),
            Err(KernelError::Type(TypeError::invalid(format!(
                "kernel mapping expression references scalar prefetch {}, but the call has 1 scalar prefetch values",
                usize::MAX,
            )))),
        );
    }

    #[test]
    fn test_tiled_call() {
        let input_types = [ArrayType::new_static(DataType::F32, [3, 5]), ArrayType::new_static(DataType::F32, [4])];
        let output_type = ArrayType::new_static(DataType::F32, [3, 7]);
        let operation = tiled_call(&input_types, output_type.clone(), 2, vec![2, 4], BoundaryPolicy::Masked).unwrap();
        assert_eq!(operation.grid().points(&[2, 2]).unwrap().len(), 4);
        assert_eq!(operation.parameters()[0].mapping().block_shape(), &[3, 5]);
        assert_eq!(operation.parameters()[1].mapping().block_shape(), &[4]);
        assert_eq!(operation.parameters()[2].mapping().block_shape(), &[2, 4]);
        assert_eq!(operation.coordinate_types().len(), 2);

        // The output takes its declared position among the parameters, and the inputs keep their order around it.
        for output_index in 0..=2 {
            let operation =
                tiled_call(&input_types, output_type.clone(), output_index, vec![2, 4], BoundaryPolicy::Masked)
                    .unwrap();
            let mut expected =
                vec![(KernelParameterAccess::ReadOnly, vec![3, 5]), (KernelParameterAccess::ReadOnly, vec![4])];
            expected.insert(output_index, (KernelParameterAccess::WriteOnly, vec![2, 4]));
            assert_eq!(
                operation
                    .parameters()
                    .iter()
                    .map(|parameter| (parameter.access(), parameter.mapping().block_shape().to_vec()))
                    .collect::<Vec<_>>(),
                expected,
            );
        }

        assert_eq!(
            tiled_call(&[], ArrayType::scalar(DataType::F32), 0, vec![2], BoundaryPolicy::Masked).unwrap_err(),
            KernelError::Type(TypeError::invalid("kernel output tile rank must equal its array rank")),
        );
        assert_eq!(
            tiled_call(&input_types, output_type, 3, vec![2, 4], BoundaryPolicy::Masked).unwrap_err(),
            KernelError::Type(TypeError::invalid("kernel output index exceeds its parameter count")),
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
    fn test_tile_load_index_domain() {
        // Loads one tile of width 2 with the fill -1 at a tile index that is an input of the given type.
        let program = |index_type: ArrayIrType, extent: usize| {
            TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
                |inputs: Vec<_>| {
                    let context = inputs[0].context();
                    Ok(vec![tile_load(context, &inputs[0], vec![2], &[inputs[1].clone()], -1i32)?])
                },
                vec![
                    ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::I32, [extent]))),
                    index_type,
                ],
            )
            .unwrap()
            .1
        };
        let load = |program: &Program<_, _, _, _>, source: Vec<i32>, index| {
            let source = ArrayReference::new(Array::vector(source).unwrap());
            program.interpret(vec![ArrayIrValue::Reference(source), index])
        };
        let tile = |elements: [i32; 2]| Ok(vec![ArrayIrValue::Array(Array::vector(elements.to_vec()).unwrap())]);
        let scalar = |index: i64| ArrayIrValue::Array(Array::scalar(index).unwrap());
        let dimension = |index: usize| ArrayIrValue::Dimension(DimensionValue::constant(index).unwrap());
        let coordinate = |upper: usize| {
            ArrayIrType::Dimension(DimensionType::new("index", DimensionBounds::non_negative(Some(upper)).unwrap()))
        };
        let unbounded =
            ArrayIrType::Dimension(DimensionType::new("index", DimensionBounds::non_negative(None).unwrap()));
        let scalar_index = program(ArrayIrType::Array(ArrayType::scalar(DataType::I64)), 2);
        let dimension_index = program(coordinate(2), 2);
        let unbounded_index = program(unbounded, 2);

        // A tile that lies entirely beyond its source takes the fill for every index kind.
        assert_eq!(load(&scalar_index, vec![3, 5], scalar(1)), tile([-1, -1]));
        assert_eq!(load(&dimension_index, vec![3, 5], dimension(1)), tile([-1, -1]));
        assert_eq!(load(&unbounded_index, vec![3, 5], dimension(1)), tile([-1, -1]));
        assert_eq!(load(&scalar_index, vec![3, 5], scalar(7)), tile([-1, -1]));
        assert_eq!(load(&unbounded_index, vec![3, 5], dimension(7)), tile([-1, -1]));

        // A loop index is a signed scalar, so it selects tiles beyond the source like every other index.
        let (_, looped) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<_>| {
                let context = inputs[0].context().clone();
                let initial = full::<i32, _>(&context, [2], 0)?;
                let outputs = for_loop(&context, 1..2, vec![inputs[0].clone(), initial], |index, values| {
                    let context = index.context().clone();
                    Ok(vec![values[0].clone(), tile_load(&context, &values[0], vec![2], &[index], -1i32)?])
                })?;
                Ok(vec![outputs[1].clone()])
            },
            vec![ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::I32, [2])))],
        )
        .unwrap();
        let source = ArrayReference::new(Array::vector(vec![3i32, 5]).unwrap());
        assert_eq!(looped.interpret(vec![ArrayIrValue::Reference(source)]), tile([-1, -1]));

        // The last valid tile of a length-3 source is partial.
        let partial_scalar_index = program(ArrayIrType::Array(ArrayType::scalar(DataType::I64)), 3);
        let partial_dimension_index = program(coordinate(2), 3);
        assert_eq!(load(&partial_scalar_index, vec![3, 5, 7], scalar(1)), tile([7, -1]));
        assert_eq!(load(&partial_dimension_index, vec![3, 5, 7], dimension(1)), tile([7, -1]));
        assert_eq!(load(&partial_dimension_index, vec![3, 5, 7], dimension(0)), tile([3, 5]));

        // Indices whose unclamped element start would overflow the canonical dimension width are clamped first, so
        // the checked multiplication never overflows and they select a tile beyond the source as well.
        assert_eq!(load(&scalar_index, vec![3, 5], scalar(i64::MAX)), tile([-1, -1]));
        assert_eq!(load(&unbounded_index, vec![3, 5], dimension(MAX_DIMENSION_EXTENT)), tile([-1, -1]));
        let unsigned_index = program(ArrayIrType::Array(ArrayType::scalar(DataType::U64)), 2);
        assert_eq!(
            load(&unsigned_index, vec![3, 5], ArrayIrValue::Array(Array::scalar(u64::MAX).unwrap())),
            tile([-1, -1]),
        );

        // Negative scalar indices fail their checked conversion.
        assert_eq!(
            load(&scalar_index, vec![3, 5], scalar(-1)),
            Err(ProgramError::Concretization {
                message: "cannot extract a concrete `usize` from `i64[]`; value `-1` is out of range".to_string(),
            }),
        );

        // Scalar indices are clamped by a comparison and a selection, and an index whose type already bounds it by the
        // number of tiles is used directly. The element starts of every index kind are bounded.
        assert_eq!(
            scalar_index.to_string(),
            indoc! {"
                lambda %0:ref<i32[2]>, %1:i64[] .
                let %2:i64[] = const 1
                    %3:bool[] = compare [direction=LessThan] %1 %2
                    %4:i64[] = select %3 %1 %2
                    %5:dimension<tile_index_0 ∈ [0, 2)> = dimension_from_scalar [bounds=[0, 2)] %4
                    %6:dimension<2> = const 2
                    %7:dimension<tile_index_0 * 2 ∈ [0, 3)> = dimension_mul %5 %6
                    %8:i32[] = const -1
                    %9:i32[2] = tile_load [block_shape=[2], boundary=Masked] %0 %7 %8
                in (%9)"
            },
        );
        assert_eq!(
            dimension_index.to_string(),
            indoc! {"
                lambda %0:ref<i32[2]>, %1:dimension<index ∈ [0, 2)> .
                let %2:dimension<2> = const 2
                    %3:dimension<index * 2 ∈ [0, 3)> = dimension_mul %1 %2
                    %4:i32[] = const -1
                    %5:i32[2] = tile_load [block_shape=[2], boundary=Masked] %0 %3 %4
                in (%5)"
            },
        );
        assert_eq!(
            unbounded_index.to_string(),
            indoc! {"
                lambda %0:ref<i32[2]>, %1:dimension<index ∈ [0, ∞)> .
                let %2:dimension<1> = const 1
                    %3:dimension<min(index, 1) ∈ [0, 2)> = dimension_min %1 %2
                    %4:dimension<2> = const 2
                    %5:dimension<min(index, 1) * 2 ∈ [0, 3)> = dimension_mul %3 %4
                    %6:i32[] = const -1
                    %7:i32[2] = tile_load [block_shape=[2], boundary=Masked] %0 %5 %6
                in (%7)"
            },
        );
    }

    #[test]
    fn test_tile_load_index_empty_source() {
        // An oversized index into a physically empty source of the widest extent selects the empty fill tile, even
        // though the start of the first tile beyond that extent does not fit the canonical dimension width.
        let shape = [MAX_DIMENSION_EXTENT, 0];
        let (_, program) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<_>| {
                let context = inputs[0].context();
                let column = context.lift(ArrayIrValue::Dimension(DimensionValue::constant(0)?))?;
                Ok(vec![tile_load(context, &inputs[0], vec![2, 0], &[inputs[1].clone(), column], -1i32)?])
            },
            vec![
                ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::I32, shape))),
                ArrayIrType::Array(ArrayType::scalar(DataType::I64)),
            ],
        )
        .unwrap();
        let source = Array::from_elements::<i32>(ArrayType::new_static(DataType::I32, shape), &[]).unwrap();
        let source = ArrayReference::new(source);
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Reference(source),
                ArrayIrValue::Array(Array::scalar(i64::MAX).unwrap()),
            ]),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements::<i32>(ArrayType::new_static(DataType::I32, [2, 0]), &[]).unwrap(),
            )]),
        );
    }

    #[test]
    fn test_tile_load_index_widest_source() {
        // The first tile beyond a source of the widest extent starts past the canonical dimension width, so the clamp
        // cannot map every large index to a padded tile. The multiplication then keeps its runtime check, which reports
        // such an index. No backend can materialize a source this wide unless it is empty.
        let (_, program) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<_>| {
                let context = inputs[0].context();
                let column = context.lift(ArrayIrValue::Dimension(DimensionValue::constant(0)?))?;
                Ok(vec![tile_load(context, &inputs[0], vec![2, 1], &[inputs[1].clone(), column], -1i32)?])
            },
            vec![
                ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(
                    DataType::I32,
                    [MAX_DIMENSION_EXTENT, 1],
                ))),
                ArrayIrType::Array(ArrayType::scalar(DataType::I64)),
            ],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ref<i32[9223372036854775807, 1]>, %1:i64[] .
                let %2:dimension<0> = const 0
                    %3:i64[] = const 4611686018427387904
                    %4:bool[] = compare [direction=LessThan] %1 %3
                    %5:i64[] = select %4 %1 %3
                    %6:dimension<tile_index_0 ∈ [0, 4611686018427387905)> = \
                        dimension_from_scalar [bounds=[0, 4611686018427387905)] %5
                    %7:dimension<2> = const 2
                    %8:dimension<tile_index_0 * 2 ∈ [0, 9223372036854775808)> = \
                        dimension_mul [requires_runtime_assertion=true] %6 %7
                    %9:dimension<1> = const 1
                    %10:i32[] = const -1
                    %11:i32[2, 1] = \
                        tile_load [block_shape=[2, 1], boundary=Masked, requires_runtime_assertion=true] %0 %8 %2 %10
                in (%11)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_tile_load_index_placement() {
        // The clamp limit of a scalar index takes the placement of that index, so a host-pinned index traces and loads.
        let index_type = ArrayType::scalar(DataType::I64).with_memory(Memory::Host { pinned: true });
        let (_, program) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<_>| {
                let context = inputs[0].context();
                let column = context.lift(ArrayIrValue::Dimension(DimensionValue::constant(0)?))?;
                Ok(vec![tile_load(context, &inputs[0], vec![2, 2], &[inputs[1].clone(), column], -1i32)?])
            },
            vec![
                ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::I32, [9, 11]))),
                ArrayIrType::Array(index_type.clone()),
            ],
        )
        .unwrap();
        let source = ArrayReference::new(
            Array::from_elements(ArrayType::new_static(DataType::I32, [9, 11]), &(0..99i32).collect::<Vec<_>>())
                .unwrap(),
        );
        let index = Array::from_logical_bytes(index_type, &0i64.to_ne_bytes()).unwrap();
        assert_eq!(
            program.interpret(vec![ArrayIrValue::Reference(source), ArrayIrValue::Array(index)]),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::I32, [2, 2]), &[0i32, 1, 11, 12]).unwrap(),
            )]),
        );
    }

    #[test]
    fn test_tile_load_index_empty_tile() {
        // A tile without elements is empty at every index, so even an index whose tile would start beyond the
        // canonical dimension width loads the empty tile.
        let (_, program) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<_>| {
                let context = inputs[0].context();
                let row = context.lift(ArrayIrValue::Dimension(DimensionValue::constant(1)?))?;
                let column = context.lift(ArrayIrValue::Dimension(DimensionValue::constant(0)?))?;
                Ok(vec![tile_load(context, &inputs[0], vec![1 << 62, 0], &[row, column], -1i32)?])
            },
            vec![ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::I32, [1, 1])))],
        )
        .unwrap();
        // The result is compared through its type and bytes, because formatting the debug view of an array with an
        // axis this long does not finish.
        let source = ArrayReference::new(Array::matrix(1, 1, vec![7i32]).unwrap());
        let tile = program.interpret(vec![ArrayIrValue::Reference(source)]).map(|mut values| {
            let ArrayIrValue::Array(tile) = values.remove(0) else { unreachable!() };
            (tile.r#type().into_owned(), tile.logical_bytes())
        });
        assert_eq!(tile, Ok((ArrayType::new_static(DataType::I32, [1 << 62, 0]), vec![])));
    }

    #[test]
    fn test_tile_load_pinned_host_source() {
        // The padding value takes the memory placement of the source, so a tile of a pinned host source loads.
        let source_type = ArrayType::new_static(DataType::I32, [3]).with_memory(Memory::Host { pinned: true });
        let (_, program) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<_>| {
                let context = inputs[0].context();
                let index = context.lift(ArrayIrValue::Dimension(DimensionValue::constant(0)?))?;
                Ok(vec![tile_load(context, &inputs[0], vec![4], &[index], -1i32)?])
            },
            vec![ArrayIrType::Reference(ReferenceType::new(source_type.clone()))],
        )
        .unwrap();
        let source = Array::from_elements(source_type, &[3i32, 5, 7]).unwrap();
        let tile_type = ArrayType::new_static(DataType::I32, [4]).with_memory(Memory::Host { pinned: true });
        assert_eq!(
            program.interpret(vec![ArrayIrValue::Reference(ArrayReference::new(source))]),
            Ok(vec![ArrayIrValue::Array(Array::from_elements(tile_type, &[3i32, 5, 7, -1]).unwrap())]),
        );
    }

    #[test]
    fn test_tile_store() {
        // Each program stores the tile of the input that its coordinate selects plus 10, where the element of the last
        // tile that lies past the end of the input is `-1`. The private tile of the last program holds `[19, 9]`, and
        // publication clips it to the output, so only its first element survives.
        let vector = ArrayType::new_static(DataType::I32, [3]);
        let operation =
            tiled_call(std::slice::from_ref(&vector), vector.clone(), 1, vec![2], BoundaryPolicy::Masked).unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, coordinates)| {
            let context = references[0].context();
            let tile = tile_load(context, &references[0], vec![2], std::slice::from_ref(&coordinates[0]), -1i32)?;
            let offset = full::<i32, _>(context, [2], 10)?;
            let operation = ArrayIrOperation::from(ArrayOperation::Add(AddOperation::new()));
            let value = context.bind(operation, vec![], &[tile, offset])?.remove(0);
            tile_store(context, &references[1], &value)
        })
        .unwrap();
        assert_eq!(
            definition.interpret(vec![Array::vector(vec![7i32, 8, 9]).unwrap()], 2),
            Ok(vec![Array::vector(vec![17i32, 18, 19]).unwrap()]),
        );
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
    fn test_masked_load_placement() {
        // The fill keeps the exact bits of a NaN payload and the memory placement and layout of the referent.
        let r#type = ArrayType::new_static(DataType::F32, [3])
            .with_memory(Memory::Host { pinned: true })
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0], vec![])));
        let (_, program) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<_>| {
                let context = inputs[0].context();
                let mask = context.lift(ArrayIrValue::Array(Array::vector(vec![true, false, true])?))?;
                Ok(vec![masked_load(context, &inputs[0], &mask, f32::from_bits(0x7fc00001))?])
            },
            vec![ArrayIrType::Reference(ReferenceType::new(r#type.clone()))],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ref<f32[3][layout=tiled{0}]@Host[Pinned]> .
                let %1:bool[3] = const [true, false, true]
                    %2:f32[]@Host[Pinned] = constant [value=NaN]
                    %3:f32[3][layout=tiled{0}]@Host[Pinned] = \
                        broadcast [output_type=f32[3][layout=tiled{0}]@Host[Pinned], output_axes=[]] %2
                    %4:f32[3][layout=tiled{0}]@Host[Pinned] = masked_load %0 %1 %3
                in (%4)"
            },
        );
        let source = Array::from_elements(r#type.clone(), &[1f32, 2.0, 3.0]).unwrap();
        let outputs = program.interpret(vec![ArrayIrValue::Reference(ArrayReference::new(source))]).unwrap();
        let [ArrayIrValue::Array(output)] = outputs.as_slice() else { unreachable!() };
        assert_eq!(output.r#type().as_ref(), &r#type);
        assert_eq!(
            output.elements::<f32>().unwrap().iter().map(|element| element.to_bits()).collect::<Vec<_>>(),
            vec![1f32.to_bits(), 0x7fc00001, 3f32.to_bits()],
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
    fn test_masked_load_or_manual_variation() {
        // The compact fill carries the variation of a referent that varies along a manual mesh axis, so the masked
        // load and the masked swap both trace on that reference.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 1).with_varying_manual_axes(["m"]).unwrap();
        let referent = ArrayType::new_static(DataType::F32, [3]).with_sharding(Some(sharding.clone())).unwrap();
        let mask = ArrayType::new_static(DataType::Boolean, [3]).with_sharding(Some(sharding)).unwrap();
        let (_, program) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<_>| {
                let context = inputs[0].context();
                let loaded = masked_load_or(context, &inputs[0], -1f32)?;
                let swapped = masked_swap(context, &inputs[0], &inputs[1], &inputs[2], -2f32)?;
                Ok(vec![loaded, swapped])
            },
            vec![
                ArrayIrType::Reference(ReferenceType::new(referent.clone())),
                ArrayIrType::Array(referent),
                ArrayIrType::Array(mask),
            ],
        )
        .unwrap();
        let sharded = "[sharding={mesh<['m'=2:manual]>, [{}], varying_manual={'m'}}]";
        let scalar = "[sharding={mesh<['m'=2:manual]>, [], varying_manual={'m'}}]";
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda %0:ref<f32[3]{sharded}>, %1:f32[3]{sharded}, %2:bool[3]{sharded} .
                let %3:bool[] = constant [value=true]
                    %4:bool[3] = broadcast [output_type=bool[3], output_axes=[]] %3
                    %5:f32[]{scalar} = constant [value=-1.0]
                    %6:f32[3]{sharded} = broadcast [
                        output_type=f32[3]{sharded},
                        output_axes=[],
                    ] %5
                    %7:f32[3]{sharded} = masked_load %0 %4 %6
                    %8:f32[]{scalar} = constant [value=-2.0]
                    %9:f32[3]{sharded} = broadcast [
                        output_type=f32[3]{sharded},
                        output_axes=[],
                    ] %8
                    %10:f32[3]{sharded} = masked_swap %0 %1 %2 %9
                in (%7, %10)"
            },
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
            Ok(
                vec![Array::vector(vec![9i32, 2, 11, 12, 9]).unwrap(), Array::vector(vec![1i32, -1, 3, 4, 5]).unwrap(),]
            ),
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
                    %4:bool[] = constant [value=true]
                    %5:bool[2] = broadcast [output_type=bool[2], output_axes=[]] %4
                    %6:i32[2] = zero [type=i32[2]]
                    %7:i32[2] = masked_load %0 %5 %6
                    %8:i32[2] = add %7 %3
                    () = masked_store %0 %8 %5
                in ()"
            },
        );
        assert_eq!(
            definition.interpret(vec![Array::vector(vec![1i32, 2, 3]).unwrap()], 2),
            Ok(vec![Array::vector(vec![11i32, 12, 13]).unwrap()]),
        );
    }

    #[test]
    fn test_accumulate_broadcasting() {
        // Adds `value` into a read-write vector through windows of 2 elements on a grid of 2 programs. The in-bounds
        // vector has 4 elements, and the masked vector has 3, so its second window has one valid element.
        let accumulate_into = |boundary: BoundaryPolicy, value: Array| {
            let extent = if boundary == BoundaryPolicy::InBounds { 4 } else { 3 };
            let operation = mapped_call(
                Grid::new(vec![GridDimension::new(2usize.into(), GridExecution::Parallel)]).unwrap(),
                vec![(
                    ArrayType::new_static(DataType::I32, [extent]),
                    KernelParameterAccess::ReadWrite,
                    ParameterWindow::BlockIndices {
                        block_shape: vec![2],
                        boundary,
                        indices: vec![MappingExpression::GridCoordinate(0)],
                    },
                )],
                vec![],
            )
            .unwrap();
            KernelDefinition::<NoKernelExtension>::trace(operation, |(references, _coordinates)| {
                let context = references[0].context();
                let value = context.lift(ArrayIrValue::Array(value.clone()))?;
                accumulate(context, &references[0], &value, boundary)
            })
            .map(|definition| definition.interpret(vec![Array::vector((1..=extent as i32).collect()).unwrap()], 2))
        };

        // Under both policies, the value broadcasts and promotes like the addition of the window and the value, so a
        // rank-0 value, a value with unit extents, and a value with a narrower element type are added to every
        // element of the window.
        for value in [Array::scalar(10i32).unwrap(), Array::vector(vec![10i32]).unwrap(), Array::scalar(10i8).unwrap()]
        {
            assert_eq!(
                accumulate_into(BoundaryPolicy::InBounds, value.clone()),
                Ok(Ok(vec![Array::vector(vec![11i32, 12, 13, 14]).unwrap()])),
            );
            assert_eq!(
                accumulate_into(BoundaryPolicy::Masked, value),
                Ok(Ok(vec![Array::vector(vec![11i32, 12, 13]).unwrap()])),
            );
        }

        // A sum whose type differs from the referent type is rejected under both policies, by the operation that
        // each policy stages to write it.
        assert_eq!(
            accumulate_into(BoundaryPolicy::InBounds, Array::scalar(10i64).unwrap()).map(|_| ()),
            Err(KernelError::Program(ProgramError::Type(TypeError::invalid(
                "`reference_add_update` addition output type `i64[2]` must exactly match reference referent type \
                 `i32[2]`",
            )))),
        );
        assert_eq!(
            accumulate_into(BoundaryPolicy::Masked, Array::scalar(10i64).unwrap()).map(|_| ()),
            Err(KernelError::Program(ProgramError::Type(TypeError::invalid(
                "masked memory input `1` type `i64[2]` must exactly match referent type `i32[2]`",
            )))),
        );
        assert_eq!(
            accumulate_into(BoundaryPolicy::InBounds, Array::matrix(1, 2, vec![10i32, 20]).unwrap()).map(|_| ()),
            Err(KernelError::Program(ProgramError::Type(TypeError::invalid(
                "`reference_add_update` addition output type `i32[1, 2]` must exactly match reference referent type \
                 `i32[2]`",
            )))),
        );
        assert_eq!(
            accumulate_into(BoundaryPolicy::Masked, Array::matrix(1, 2, vec![10i32, 20]).unwrap()).map(|_| ()),
            Err(KernelError::Program(ProgramError::Type(TypeError::invalid(
                "masked memory input `1` type `i32[1, 2]` must exactly match referent type `i32[2]`",
            )))),
        );
    }

    #[test]
    fn test_masked_accesses_shapes() {
        // Traces a masked load with and without an explicit mask, a masked swap, and a masked accumulation of the
        // reference with the given shape, whose fills and masks must be staged without host tiles of that shape.
        let trace = |shape: [usize; 2]| {
            TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
                |inputs: Vec<_>| {
                    let context = inputs[0].context();
                    let loaded = masked_load_or(context, &inputs[0], -1i32)?;
                    let mask = full::<bool, _>(context, shape, true)?;
                    let selected = masked_load(context, &inputs[0], &mask, -2i32)?;
                    let previous = masked_swap(context, &inputs[0], &loaded, &mask, -3i32)?;
                    accumulate(context, &inputs[0], &selected, BoundaryPolicy::Masked)?;
                    Ok(vec![loaded, selected, previous])
                },
                vec![ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::I32, shape)))],
            )
        };

        // Zero extents produce empty results.
        let (_, program) = trace([0, 3]).unwrap();
        let empty = Array::from_elements(ArrayType::new_static(DataType::I32, [0, 3]), &[0i32; 0]).unwrap();
        assert_eq!(
            program.interpret(vec![ArrayIrValue::Reference(ArrayReference::new(empty.clone()))]),
            Ok(vec![ArrayIrValue::Array(empty); 3]),
        );

        // A large representable shape traces without allocating host tiles of that shape.
        let (_, program) = trace([1 << 16, 1 << 16]).unwrap();
        let rendering = indoc! {"
            lambda %0:ref<i32[65536, 65536]> .
            let %1:bool[] = constant [value=true]
                %2:bool[65536, 65536] = broadcast [output_type=bool[65536, 65536], output_axes=[]] %1
                %3:i32[] = constant [value=-1]
                %4:i32[65536, 65536] = broadcast [output_type=i32[65536, 65536], output_axes=[]] %3
                %5:i32[65536, 65536] = masked_load %0 %2 %4
                %6:bool[] = constant [value=true]
                %7:bool[65536, 65536] = broadcast [output_type=bool[65536, 65536], output_axes=[]] %6
                %8:i32[] = constant [value=-2]
                %9:i32[65536, 65536] = broadcast [output_type=i32[65536, 65536], output_axes=[]] %8
                %10:i32[65536, 65536] = masked_load %0 %7 %9
                %11:i32[] = constant [value=-3]
                %12:i32[65536, 65536] = broadcast [output_type=i32[65536, 65536], output_axes=[]] %11
                %13:i32[65536, 65536] = masked_swap %0 %5 %7 %12
                %14:bool[] = constant [value=true]
                %15:bool[65536, 65536] = broadcast [output_type=bool[65536, 65536], output_axes=[]] %14
                %16:i32[65536, 65536] = zero [type=i32[65536, 65536]]
                %17:i32[65536, 65536] = masked_load %0 %15 %16
                %18:i32[65536, 65536] = add %17 %10
                () = masked_store %0 %18 %15
            in (%5, %10, %13)"
        };
        assert_eq!(program.to_string(), rendering);

        // A shape whose storage size is unrepresentable traces the same operations, and materializing one of its fills
        // is a typed error.
        let (_, program) = trace([MAX_DIMENSION_EXTENT, 4]).unwrap();
        assert_eq!(program.to_string(), rendering.replace("65536, 65536", &format!("{MAX_DIMENSION_EXTENT}, 4")));
        let (_, fill) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<_>| Ok(vec![full::<i32, _>(inputs[0].context(), [MAX_DIMENSION_EXTENT, 4], -1i32)?]),
            vec![ArrayIrType::Array(ArrayType::scalar(DataType::I32))],
        )
        .unwrap();
        assert_eq!(
            fill.interpret(vec![ArrayIrValue::Array(Array::scalar(0i32).unwrap())]),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "shape [{MAX_DIMENSION_EXTENT}, 4] element count does not fit in `usize`",
            )))),
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
