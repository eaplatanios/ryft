//! Dynamically positioned tile loads over canonical reference views.
//!
//! A tile load reads a fixed size window of a reference view whose start along each axis is a runtime dimension value,
//! so a kernel can read the tile that its grid point or loop iteration selects without knowing the tile position
//! statically. [`TileLoadOperation`] (i.e., the `tile_load` operation) owns this access. Its result is an ordinary
//! dense array, and an explicit [`BoundaryPolicy`] and padding value determine its contents where the window extends
//! beyond the edge of the source view.
//!
//! # Inputs and Types
//!
//! The inputs of a tile load are the source reference, one dimension start per tile axis, and a rank-0 padding value
//! with the data type and memory placement of the source, followed by the bindings of the source transforms. The
//! transforms select a reference view of the source before the tile window applies, and that view must have a static
//! shape with the rank of the tile. Type inference reuses the canonical rules rather than restating them. It infers the
//! type of an empty slice of the view and pads that slice to the tile shape with [`PadOperation`], so the rules for
//! distributed axes and padding values remain owned by those operations. The result drops any physical layout, because
//! a tile load declares a dense result.
//!
//! # Boundary Semantics
//!
//! Each axis of the window starts at its dimension start and spans the tile extent. Under [`BoundaryPolicy::InBounds`],
//! every window must lie inside the source view, and interpretation fails otherwise. Under
//! [`BoundaryPolicy::Masked`], the window is clipped to the source view, and the result elements beyond its edge
//! contain the exact encoding of the padding value, even when the clipped window is empty. Under either policy, only
//! the valid clipped reference view is constructed and read, so no out-of-bounds reference is ever created or accessed,
//! and a window limit beyond the canonical dimension width fails. Such a failure is observable even when the tile is
//! unused, so the operation declares an ordered assertion unless [`TileLoadOperation::with_input_types`] proves from
//! the bounds of the start types that every window is valid. Initialization validation conservatively requires the
//! entire source view to be initialized, because the selected window is dynamic.
//!
//! # Authoring
//!
//! In a `#[kernel]` body, `input.tiles(shape).pad(value)` declares a tiled view of an input, which generates no code by
//! itself. Each `load(indices)` of that view lowers to the [`tile_load`](crate::authoring::tile_load) authoring
//! function, which accepts every nonnegative tile index. An index at or beyond the number of tiles that cover an axis
//! selects a tile that lies entirely beyond the source and contains only the padding value, so that function first
//! clamps every index whose type does not bound it by that number. It then converts integer scalar tile indices into
//! checked dimension values, which rejects negative indices, multiplies each tile index by the tile extent to obtain
//! an element start, and stages a tile load with [`BoundaryPolicy::Masked`] and the declared padding value. The clamp
//! bounds the element starts, so the staged load of an ordinary tiled kernel is proved free of the window check.
//!
//! # Advanced Indexing
//!
//! This module defines no gather or scatter operations. Broadcast advanced indexing composes the existing
//! [`BroadcastOperation`](ryft_core::operations::BroadcastOperation),
//! [`ConcatenateOperation`](ryft_core::operations::ConcatenateOperation),
//! [`GatherOperation`](ryft_core::operations::GatherOperation), and
//! [`ScatterOperation`](ryft_core::operations::ScatterOperation). Broadcast signed integer query components to a common
//! shape, append their index vector axis, and then concatenate the components. Use
//! [`SelectOperation`](ryft_core::operations::SelectOperation) to replace an inactive query with a negative index, and
//! [`GatherMode::Fill`](ryft_core::operations::GatherMode::Fill) to fill loads or
//! [`ScatterMode::Drop`](ryft_core::operations::ScatterMode::Drop) to drop stores. Negative indices never wrap to the
//! opposite array edge, and the explicit scalar fill of a gather preserves its exact encoding.
//!
//! Read the qualified reference with [`ReferenceRead`](ryft_core::operations::ReferenceRead) before gathering or
//! scattering, and publish the result with [`ReferenceWrite`](ryft_core::ReferenceWrite). This composition reads the
//! complete source window. Indexed stores are read-modify-write operations over the whole window and require an already
//! initialized destination, even when the selected indices happen to cover every element. Existing reference effects
//! enforce that contract and reject unordered overlapping mutable windows. The composition does not establish partial
//! write-only initialization or create an additional reference view representation. More selective reference effects
//! require a separate proof before admission, and an inactive query mask alone does not waive the initialization of the
//! whole window.

use std::borrow::Cow;
use std::fmt::Display;
use std::sync::LazyLock;

use ryft_core::macros::check_count;
use ryft_core::{
    Array, ArrayIrType, ArrayIrValue, ArrayReferenceTransform, ArraySliceAxis, ArrayType, EagerContext, EffectClass,
    EffectClasses, Effects, InterpretableOperation, InterpretationDriver, MAX_DIMENSION_EXTENT, Operation,
    OperationFormatter, Pad, PadOperation, ProgramError, ReferenceAccessDescriptor, ReferenceAccessMode,
    ReferenceAccessOperation, ReferenceEffect, ReferenceTransform, RegionInterface, TypeError, Typed,
    infer_reference_view_type,
};

use crate::mappings::BoundaryPolicy;

/// Canonical operation name for [`TileLoadOperation`].
pub const TILE_LOAD_OPERATION_NAME: &str = "tile_load";

/// Reads a dynamically selected fixed size tile from a canonical reference as an ordinary dense array.
///
/// The inputs are the source reference, one checked dimension start per tile axis, and an explicit rank-0 padding value
/// with the data type and memory placement of the source, followed by the bindings of the source transforms, which
/// apply before the tile window. Under [`BoundaryPolicy::Masked`], the operation reads only a valid clipped reference
/// view, and every other result element contains the exact encoding of the padding value. No out-of-bounds reference is
/// constructed or accessed. Initialization validation conservatively requires the entire source view to be initialized,
/// because the selection is dynamic. The operation declares an ordered assertion for its window check unless
/// [`TileLoadOperation::with_input_types`] proves that check unnecessary.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TileLoadOperation {
    /// Static logical shape of each result tile.
    block_shape: Vec<usize>,

    /// Whether coordinates outside the source are rejected or filled explicitly.
    boundary_policy: BoundaryPolicy,

    /// Ordered selections applied to the source before selecting the tile window.
    transforms: Vec<ArrayReferenceTransform>,

    /// Whether execution must check the window of each axis. Initialized conservatively to `true` and refined by
    /// [`TileLoadOperation::with_input_types`].
    requires_runtime_assertion: bool,
}

impl TileLoadOperation {
    /// Declares a tile shape and boundary policy, checking each extent against the canonical dimension width. The
    /// operation initially retains a runtime window check. Use [`TileLoadOperation::with_input_types`] to remove it
    /// when the input types prove that every window is valid.
    pub fn new(block_shape: Vec<usize>, boundary_policy: BoundaryPolicy) -> Result<Self, TypeError> {
        if block_shape.iter().any(|&extent| extent > MAX_DIMENSION_EXTENT) {
            return Err(TypeError::invalid(format!(
                "`{TILE_LOAD_OPERATION_NAME}` block shape exceeds the canonical dimension width",
            )));
        }
        Ok(Self { block_shape, boundary_policy, transforms: Vec::new(), requires_runtime_assertion: true })
    }

    /// Replaces the source selections applied before the tile window.
    pub fn with_transforms(mut self, transforms: Vec<ArrayReferenceTransform>) -> Self {
        self.transforms = transforms;
        self
    }

    /// Returns this [`TileLoadOperation`] with its runtime window check requirement recomputed from the provided input
    /// types, after validating the complete input signature.
    ///
    /// The window of an axis is proved valid when the largest start that its dimension type admits plus the tile
    /// extent fits [`MAX_DIMENSION_EXTENT`] and, under [`BoundaryPolicy::InBounds`], also the extent of the source view
    /// along that axis. When every axis is proved, the returned operation is free of [`EffectClass::OrderedAssertion`].
    /// Otherwise, it retains that effect, because its execution can fail. For example, a masked tile of extent 2
    /// whose start has type `dimension<i * 2 ∈ [0, 3)>` is proved for every source, while an in-bounds tile of extent
    /// 1 at the constant start 4 of an `f32[4]` source is not.
    ///
    /// An operation without the check rejects subsequent type inference requests that would require it, rather than
    /// silently changing its effects. Calling this function again recomputes the requirement for a different
    /// signature and can restore the check.
    ///
    /// # Errors
    ///
    /// Returns the [`TypeError`] of type inference when `input_types` is not a valid input signature.
    pub fn with_input_types(mut self, input_types: &[ArrayIrType]) -> Result<Self, TypeError> {
        self.requires_runtime_assertion = true;
        self.infer_output_types(input_types, &[])?;
        let source = self.source_type(input_types)?;
        self.requires_runtime_assertion = !self.has_valid_windows(&input_types[1..self.block_shape.len() + 1], &source);
        Ok(self)
    }

    /// Returns the fixed result shape.
    pub fn block_shape(&self) -> &[usize] {
        &self.block_shape
    }

    /// Returns the transforms applied to the source reference before the tile window.
    pub fn transforms(&self) -> &[ArrayReferenceTransform] {
        &self.transforms
    }

    /// Returns the boundary contract applied before reference access.
    pub fn boundary_policy(&self) -> BoundaryPolicy {
        self.boundary_policy
    }

    /// Returns whether execution must check that the window of each axis fits the canonical dimension width and,
    /// under [`BoundaryPolicy::InBounds`], lies inside the source view.
    ///
    /// When `true`, the operation carries [`EffectClass::OrderedAssertion`], so a failing check is observable even
    /// when the loaded tile is unused. This is the conservative initial state, not a claim that a window is invalid.
    /// [`Self::with_input_types`] sets the flag to `false` when the input types prove every window valid.
    pub fn requires_runtime_assertion(&self) -> bool {
        self.requires_runtime_assertion
    }

    /// Returns the type of the source view that the transforms select, for a validated input signature.
    fn source_type(&self, input_types: &[ArrayIrType]) -> Result<ArrayType, TypeError> {
        let ArrayIrType::Reference(reference) = &input_types[0] else {
            return Err(TypeError::invalid(format!(
                "`{TILE_LOAD_OPERATION_NAME}` first input must be an array reference",
            )));
        };
        infer_reference_view_type(
            reference.referent(),
            &self.transforms,
            &input_types[self.block_shape.len() + 2..].iter().collect::<Vec<_>>(),
            ReferenceAccessMode::Read,
        )
    }

    /// Returns whether the dimension types `starts` prove that the window of every axis is valid in the static
    /// `source` view, using only their bounds. Refer to [`Self::with_input_types`] for the rule.
    fn has_valid_windows(&self, starts: &[ArrayIrType], source: &ArrayType) -> bool {
        let shape = source.static_shape().unwrap();
        starts.iter().zip(&self.block_shape).zip(shape.dimensions()).all(|((start, &block), &extent)| {
            let ArrayIrType::Dimension(start) = start else { unreachable!() };
            start.maximum_extent().and_then(|maximum| maximum.checked_add(block)).is_some_and(|limit| {
                limit <= MAX_DIMENSION_EXTENT && (self.boundary_policy == BoundaryPolicy::Masked || limit <= extent)
            })
        })
    }
}

impl Display for TileLoadOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for TileLoadOperation {
    type Type = ArrayIrType;

    fn name(&self) -> &'static str {
        TILE_LOAD_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        let rank = self.block_shape.len();
        let binding_count = self.transforms.iter().map(|transform| transform.binding_count()).sum::<usize>();
        check_count!("input", input_types, rank + 2 + binding_count, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let source = self.source_type(input_types)?;
        if source.rank() != rank || source.static_shape().is_none() {
            return Err(TypeError::invalid(format!(
                "`{TILE_LOAD_OPERATION_NAME}` source requires a static shape with the block rank",
            )));
        }
        if input_types[1..rank + 1].iter().any(|r#type| !matches!(r#type, ArrayIrType::Dimension(_))) {
            return Err(TypeError::invalid(format!(
                "`{TILE_LOAD_OPERATION_NAME}` starts must be canonical dimensions",
            )));
        }
        let ArrayIrType::Array(other) = &input_types[rank + 1] else {
            return Err(TypeError::invalid(format!(
                "`{TILE_LOAD_OPERATION_NAME}` padding value must be a scalar array",
            )));
        };
        if !self.requires_runtime_assertion && !self.has_valid_windows(&input_types[1..rank + 1], &source) {
            return Err(TypeError::invalid(format!(
                "`{TILE_LOAD_OPERATION_NAME}` was constructed without a runtime window check but these input types \
                 require one",
            )));
        }
        // Infer through an empty valid slice and canonical padding, so the rules for distributed axes and padding
        // values remain owned by the existing operations. Clear the physical layout for the declared dense result
        // contract.
        let empty =
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 0, 1); rank] }.output_type(&source)?;
        let padding = PadOperation::<ArrayType>::new(
            vec![0; rank],
            self.block_shape.iter().map(|&extent| extent as i64).collect(),
            vec![0; rank],
        )
        .unwrap();
        let mut output = padding.infer_output_types(&[empty, other.clone()], &[])?;
        Ok(vec![ArrayIrType::Array(output.remove(0).with_layout(None))])
    }

    fn effects(&self) -> Cow<'_, Effects> {
        // A failing window check is observable even when the loaded tile is unused, so it is an ordered assertion.
        static EFFECTS: LazyLock<[Effects; 2]> = LazyLock::new(|| {
            [EffectClasses::NONE, EffectClasses::single(EffectClass::OrderedAssertion)].map(|classes| {
                Effects::new(classes, vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read }])
                    .unwrap()
            })
        });
        Cow::Borrowed(&EFFECTS[usize::from(self.requires_runtime_assertion)])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("block_shape", format_args!("{:?}", self.block_shape))?;
            operation.field("boundary", format_args!("{:?}", self.boundary_policy))?;
            if !self.transforms.is_empty() {
                operation.list("transforms", &self.transforms)?;
            }
            if self.requires_runtime_assertion {
                operation.field("requires_runtime_assertion", true)?;
            }
            Ok(())
        })
    }
}

impl ReferenceAccessOperation for TileLoadOperation {
    type Transform = ArrayReferenceTransform;

    fn base_input_count(&self) -> usize {
        self.block_shape.len() + 2
    }

    fn reference_access_descriptor(
        &self,
        input_index: usize,
    ) -> Option<ReferenceAccessDescriptor<'_, ArrayReferenceTransform>> {
        (input_index == 0).then(|| {
            let start = self.base_input_count();
            ReferenceAccessDescriptor::new(
                &self.transforms,
                start..start + self.transforms.iter().map(|transform| transform.binding_count()).sum::<usize>(),
            )
        })
    }

    fn with_reference_access_transforms(
        &self,
        input_index: usize,
        transforms: Vec<ArrayReferenceTransform>,
    ) -> Result<Self, ProgramError> {
        if input_index != 0 {
            return Err(ProgramError::MalformedProgram(format!(
                "operation `{}` has no reference access at input {input_index}",
                self.name(),
            )));
        }
        Ok(self.clone().with_transforms(transforms))
    }
}

impl<O: Operation<Type = ArrayIrType>> InterpretableOperation<EagerContext<ArrayIrValue<Array>, O>>
    for TileLoadOperation
{
    fn interpret<D: InterpretationDriver<EagerContext<ArrayIrValue<Array>, O>>>(
        &self,
        _context: &EagerContext<ArrayIrValue<Array>, O>,
        _driver: &D,
        inputs: &[ArrayIrValue<Array>],
    ) -> Result<Vec<ArrayIrValue<Array>>, ProgramError> {
        let output_types =
            self.infer_output_types(&inputs.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>(), &[])?;
        let ArrayIrType::Array(output_type) = &output_types[0] else { unreachable!() };
        let ArrayIrValue::Reference(source) = &inputs[0] else { unreachable!() };
        let rank = self.block_shape.len();
        let ArrayIrValue::Array(other) = &inputs[rank + 1] else { unreachable!() };
        let source = source.with_transforms(&self.transforms, &inputs[rank + 2..])?;
        let source_type = source.r#type();
        let shape = source_type.referent().static_shape().unwrap();
        let mut axes = Vec::with_capacity(self.block_shape.len());
        let mut padding = Vec::with_capacity(self.block_shape.len());
        for ((start, &block), &extent) in inputs[1..rank + 1].iter().zip(&self.block_shape).zip(shape.dimensions()) {
            let ArrayIrValue::Dimension(start) = start else { unreachable!() };
            let start = start.extent();
            let limit = start.checked_add(block).filter(|&limit| limit <= MAX_DIMENSION_EXTENT).ok_or_else(|| {
                TypeError::invalid(format!(
                    "`{TILE_LOAD_OPERATION_NAME}` window limit exceeds the canonical dimension width",
                ))
            })?;
            if self.boundary_policy == BoundaryPolicy::InBounds && limit > extent {
                return Err(TypeError::invalid(format!(
                    "`{TILE_LOAD_OPERATION_NAME}` in-bounds window exceeds the source extent",
                ))
                .into());
            }
            let valid_start = start.min(extent);
            let valid_size = limit.min(extent).saturating_sub(valid_start);
            axes.push(ArraySliceAxis::new(valid_start, valid_size, 1));
            padding.push((block - valid_size) as i64);
        }
        let selected = source.with_transform(ArrayReferenceTransform::Slice { axes })?.read()?;
        let rank = self.block_shape.len();
        let padded = selected.pad(other, &vec![0; rank], &padding, &vec![0; rank])?;
        let output = Array::from_logical_bytes(output_type.clone(), &padded.logical_bytes())?;
        Ok(vec![ArrayIrValue::Array(output)])
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, ArrayIrOperation, ArrayOperation, ArrayReference, ArrayReferenceTransformIndex,
        BroadcastOperation, ConcatenateOperation, Context, DataType, DimensionBounds, DimensionType, DimensionValue,
        DotOperation, GatherDimensionNumbers, GatherMode, GatherOperation, InstructionId, Placeholder, ProgramBuilder,
        ReferenceReadOperation, ReferenceRoot, ReferenceType, ReferenceWriteOperation, ScatterDimensionNumbers,
        ScatterMode, ScatterOperation, ScatterReductionKind, SelectOperation, check_operation_type_inference,
    };

    use crate::authoring::{tiled_call, whole_array_parameter};
    use crate::calls::{KernelCallOperation, KernelDefinition, KernelError};
    use crate::grids::Grid;
    use crate::initialization::{KernelInitializationError, validate_kernel_initialization};
    use crate::operations::KernelOperation;
    use crate::validation::KernelParameterAccess;

    use super::*;

    /// Broadcasts row and column queries and a mask into canonical gather and scatter index vectors.
    fn advanced_index_definition(access: KernelParameterAccess) -> Result<KernelDefinition, KernelError> {
        let source_type = ArrayType::new_static(DataType::I32, [3, 4]);
        let query_type = ArrayType::new_static(DataType::I32, [2, 3]);
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(source_type, access).unwrap(),
                whole_array_parameter(ArrayType::new_static(DataType::I32, [2, 1]), KernelParameterAccess::ReadOnly)
                    .unwrap(),
                whole_array_parameter(ArrayType::new_static(DataType::I32, [1, 3]), KernelParameterAccess::ReadOnly)
                    .unwrap(),
                whole_array_parameter(
                    ArrayType::new_static(DataType::Boolean, [2, 3]),
                    KernelParameterAccess::ReadOnly,
                )
                .unwrap(),
                whole_array_parameter(query_type, KernelParameterAccess::WriteOnly).unwrap(),
            ],
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let inputs = call.body_input_types().into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let component_type = ArrayType::new_static(DataType::I32, [2, 3, 1]);
        let sentinel = builder
            .add_constant(ArrayIrValue::Array(Array::from_elements(component_type.clone(), &[-1i32; 6]).unwrap()));
        let increment = builder.add_constant(ArrayIrValue::Array(Array::matrix(2, 3, vec![100i32; 6]).unwrap()));
        let values = inputs[..4]
            .iter()
            .map(|&input| {
                builder
                    .add_instruction(
                        ArrayIrOperation::ReferenceRead(ReferenceReadOperation::new()),
                        vec![],
                        vec![input],
                        None,
                    )
                    .unwrap()[0]
            })
            .collect::<Vec<_>>();
        let mut components = Vec::new();
        for (input, r#type, axes) in [
            (values[1], component_type.clone(), vec![0, 2]),
            (values[2], component_type.clone(), vec![0, 1]),
            (values[3], component_type.with_data_type(DataType::Boolean), vec![0, 1]),
        ] {
            components.push(
                builder
                    .add_instruction(
                        ArrayIrOperation::Array(ArrayOperation::Broadcast(BroadcastOperation::new(r#type, axes))),
                        vec![],
                        vec![input],
                        None,
                    )
                    .unwrap()[0],
            );
        }
        // Fill and drop modes treat the negative sentinel as invalid, so it never wraps to the final source row.
        let rows = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Select(SelectOperation::new())),
                vec![],
                vec![components[2], components[0], sentinel],
                None,
            )
            .unwrap()[0];
        let indices = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Concatenate(
                    ConcatenateOperation::<ArrayType>::new(2, 3).unwrap(),
                )),
                vec![],
                vec![rows, components[1]],
                None,
            )
            .unwrap()[0];
        let gathered = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Gather(
                    GatherOperation::new(GatherDimensionNumbers::new(vec![], vec![0, 1], vec![0, 1]), vec![1, 1])
                        .with_mode(GatherMode::Fill { value: Some(Box::new(Array::scalar(-7i32).unwrap())) }),
                )),
                vec![],
                vec![values[0], indices],
                None,
            )
            .unwrap()[0];
        let updates = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Add(AddOperation::new())),
                vec![],
                vec![gathered, increment],
                None,
            )
            .unwrap()[0];
        let replaced = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Scatter(
                    ScatterOperation::new(
                        ScatterDimensionNumbers::new(vec![], vec![0, 1], vec![0, 1]),
                        ScatterReductionKind::Overwrite,
                    )
                    .with_mode(ScatterMode::Drop),
                )),
                vec![],
                vec![values[0], indices, updates],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(
                ArrayIrOperation::ReferenceWrite(ReferenceWriteOperation::new()),
                vec![],
                vec![inputs[0], replaced],
                None,
            )
            .unwrap();
        builder
            .add_instruction(
                ArrayIrOperation::ReferenceWrite(ReferenceWriteOperation::new()),
                vec![],
                vec![inputs[4], gathered],
                None,
            )
            .unwrap();
        let body = builder.build(vec![], vec![Placeholder; inputs.len()], vec![]).unwrap();
        KernelDefinition::new(call, body)
    }

    /// Builds a kernel without a grid that loads one tile of extent `block` of its `f32[4]` input under `boundary` at
    /// the constant start `start` and writes that tile to its output. When `refined` is set, the tile load is refined
    /// with [`TileLoadOperation::with_input_types`], and otherwise it keeps its conservative window check.
    fn tile_load_definition(block: usize, boundary: BoundaryPolicy, start: usize, refined: bool) -> KernelDefinition {
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(ArrayType::new_static(DataType::F32, [4]), KernelParameterAccess::ReadOnly)
                    .unwrap(),
                whole_array_parameter(ArrayType::new_static(DataType::F32, [block]), KernelParameterAccess::WriteOnly)
                    .unwrap(),
            ],
        )
        .unwrap();
        let input_types = call.body_input_types();
        let start = DimensionValue::constant(start).unwrap();
        let mut operation = TileLoadOperation::new(vec![block], boundary).unwrap();
        if refined {
            operation = operation
                .with_input_types(&[
                    input_types[0].clone(),
                    ArrayIrType::Dimension(start.r#type().into_owned()),
                    ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
                ])
                .unwrap();
        }
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let inputs = input_types.into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let start = builder.add_constant(ArrayIrValue::Dimension(start));
        let other = builder.add_constant(ArrayIrValue::Array(Array::scalar(-1f32).unwrap()));
        let tile = builder.add_instruction(operation, vec![], vec![inputs[0], start, other], None).unwrap()[0];
        builder
            .add_instruction(
                ArrayIrOperation::ReferenceWrite(ReferenceWriteOperation::new()),
                vec![],
                vec![inputs[1], tile],
                None,
            )
            .unwrap();
        KernelDefinition::new(call, builder.build(vec![], vec![Placeholder; 2], vec![]).unwrap()).unwrap()
    }

    #[test]
    fn test_tile_load_operation_new() {
        let operation = TileLoadOperation::new(vec![2, 3], BoundaryPolicy::Masked).unwrap();
        assert_eq!(operation.block_shape(), &[2, 3]);
        assert_eq!(operation.boundary_policy(), BoundaryPolicy::Masked);
        assert_eq!(operation.name(), TILE_LOAD_OPERATION_NAME);
        assert_eq!(
            format!("{operation}"),
            "tile_load [block_shape=[2, 3], boundary=Masked, requires_runtime_assertion=true]",
        );
        assert_eq!(
            TileLoadOperation::new(vec![MAX_DIMENSION_EXTENT + 1], BoundaryPolicy::Masked),
            Err(TypeError::invalid(format!(
                "`{TILE_LOAD_OPERATION_NAME}` block shape exceeds the canonical dimension width",
            ))),
        );
    }

    #[test]
    fn test_tile_load_operation_with_transforms() {
        let transforms = vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        let operation =
            TileLoadOperation::new(vec![2], BoundaryPolicy::Masked).unwrap().with_transforms(transforms.clone());
        assert_eq!(operation.transforms(), transforms);
    }

    #[test]
    fn test_tile_load_operation_with_input_types() {
        let source = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [4])));
        let padding = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let start = |upper: Option<usize>| {
            ArrayIrType::Dimension(DimensionType::new("start", DimensionBounds::non_negative(upper).unwrap()))
        };
        let refine = |boundary: BoundaryPolicy, start: ArrayIrType| {
            TileLoadOperation::new(vec![2], boundary)
                .unwrap()
                .with_input_types(&[source.clone(), start, padding.clone()])
                .map(|operation| operation.requires_runtime_assertion())
        };
        assert!(TileLoadOperation::new(vec![2], BoundaryPolicy::Masked).unwrap().requires_runtime_assertion());

        // The largest start plus the tile extent must fit the canonical dimension width, and an in-bounds window must
        // also fit the source view.
        assert_eq!(refine(BoundaryPolicy::Masked, start(Some(3))), Ok(false));
        assert_eq!(refine(BoundaryPolicy::Masked, start(Some(MAX_DIMENSION_EXTENT - 1))), Ok(false));
        assert_eq!(refine(BoundaryPolicy::Masked, start(Some(MAX_DIMENSION_EXTENT))), Ok(true));
        assert_eq!(refine(BoundaryPolicy::Masked, start(None)), Ok(true));
        assert_eq!(refine(BoundaryPolicy::InBounds, start(Some(3))), Ok(false));
        assert_eq!(refine(BoundaryPolicy::InBounds, start(Some(4))), Ok(true));
        assert_eq!(
            refine(BoundaryPolicy::Masked, padding.clone()),
            Err(TypeError::invalid(format!("`{TILE_LOAD_OPERATION_NAME}` starts must be canonical dimensions"))),
        );

        // An operation without the check rejects input types that require it rather than silently dropping its effect.
        let proved = TileLoadOperation::new(vec![2], BoundaryPolicy::Masked)
            .unwrap()
            .with_input_types(&[source.clone(), start(Some(3)), padding.clone()])
            .unwrap();
        assert_eq!(
            proved.infer_output_types(&[source, start(None), padding], &[]),
            Err(TypeError::invalid(format!(
                "`{TILE_LOAD_OPERATION_NAME}` was constructed without a runtime window check but these input types \
                 require one",
            ))),
        );
    }

    #[test]
    fn test_tile_load_operation_type_inference() {
        let source = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])));
        let start = ArrayIrType::Dimension(DimensionValue::constant(1).unwrap().r#type().into_owned());
        let padding = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let index = ArrayIrType::Array(ArrayType::scalar(DataType::I64));
        check_operation_type_inference!(
            operation = TileLoadOperation::new(vec![2], BoundaryPolicy::Masked).unwrap(),
            cases = [
                {
                    input_types = [source.clone(), start.clone(), padding.clone()],
                    output_types = [ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2]))],
                },
                {
                    input_types = [source.clone(), start.clone()],
                    error = "expected 3 inputs but got 2",
                },
                {
                    input_types = [padding.clone(), start.clone(), padding.clone()],
                    error = format!("`{TILE_LOAD_OPERATION_NAME}` first input must be an array reference"),
                },
                {
                    input_types = [source.clone(), index.clone(), padding.clone()],
                    error = format!("`{TILE_LOAD_OPERATION_NAME}` starts must be canonical dimensions"),
                },
                {
                    input_types = [source.clone(), start.clone(), start.clone()],
                    error = format!("`{TILE_LOAD_OPERATION_NAME}` padding value must be a scalar array"),
                },
                {
                    input_types = [
                        ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3, 1]))),
                        start.clone(),
                        padding.clone(),
                    ],
                    error = format!("`{TILE_LOAD_OPERATION_NAME}` source requires a static shape with the block rank"),
                },
            ],
        );

        // The transforms select the source view before the tile window, and the result is dense.
        let matrix = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2, 3])));
        check_operation_type_inference!(
            operation = TileLoadOperation::new(vec![2], BoundaryPolicy::InBounds).unwrap().with_transforms(vec![
                ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
            ]),
            cases = [
                {
                    input_types = [matrix.clone(), start.clone(), padding.clone(), index],
                    output_types = [ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2]))],
                },
                {
                    input_types = [matrix, start, padding],
                    error = "expected 4 inputs but got 3",
                },
            ],
        );
    }

    #[test]
    fn test_tile_load_operation_effects() {
        let operation = TileLoadOperation::new(vec![2], BoundaryPolicy::Masked).unwrap();
        assert_eq!(
            operation.effects().as_ref(),
            &Effects::new(
                EffectClasses::single(EffectClass::OrderedAssertion),
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read }],
            )
            .unwrap(),
        );
        let source = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])));
        let start = ArrayIrType::Dimension(DimensionValue::constant(1).unwrap().r#type().into_owned());
        let padding = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        assert_eq!(
            operation.with_input_types(&[source, start, padding]).unwrap().effects().as_ref(),
            &Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read }],
            )
            .unwrap(),
        );
    }

    #[test]
    fn test_tile_load_operation_effects_failing_windows() {
        // An in-bounds window at offset 4 of an `f32[4]` source and a masked window whose limit exceeds the canonical
        // dimension width both fail on the host, so the kernel that contains them must declare an ordered assertion.
        // Refinement keeps the check, because the start types do not prove these windows valid.
        let source = Array::vector(vec![1f32, 2.0, 3.0, 4.0]).unwrap();
        for refined in [false, true] {
            let in_bounds = tile_load_definition(1, BoundaryPolicy::InBounds, 4, refined);
            assert_eq!(
                in_bounds.interpret(vec![source.clone()], 1),
                Err(ProgramError::Type(TypeError::invalid(format!(
                    "`{TILE_LOAD_OPERATION_NAME}` in-bounds window exceeds the source extent",
                )))),
            );
            assert!(in_bounds.effects().classes().contains(EffectClass::OrderedAssertion));
            let masked = tile_load_definition(1, BoundaryPolicy::Masked, MAX_DIMENSION_EXTENT, refined);
            assert_eq!(
                masked.interpret(vec![source.clone()], 1),
                Err(ProgramError::Type(TypeError::invalid(format!(
                    "`{TILE_LOAD_OPERATION_NAME}` window limit exceeds the canonical dimension width",
                )))),
            );
            assert!(masked.effects().classes().contains(EffectClass::OrderedAssertion));
        }

        // A refined in-bounds window at the last valid offset is proved valid and keeps the kernel free of assertions.
        let safe = tile_load_definition(1, BoundaryPolicy::InBounds, 3, true);
        assert_eq!(safe.interpret(vec![source], 1), Ok(vec![Array::vector(vec![4f32]).unwrap()]));
        assert!(!safe.effects().classes().contains(EffectClass::OrderedAssertion));
    }

    #[test]
    fn test_tile_load_operation_render() {
        // Only an operation that retains its window check renders the flag.
        let operation = TileLoadOperation::new(vec![2], BoundaryPolicy::Masked).unwrap();
        assert_eq!(
            operation.to_string(),
            "tile_load [block_shape=[2], boundary=Masked, requires_runtime_assertion=true]",
        );
        let source = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2, 3])));
        let start = ArrayIrType::Dimension(DimensionValue::constant(1).unwrap().r#type().into_owned());
        let padding = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let operation = operation.with_transforms(vec![ArrayReferenceTransform::Index {
            axis: 0,
            index: ArrayReferenceTransformIndex::Static(1),
        }]);
        assert_eq!(
            operation.clone().with_input_types(&[source, start, padding]).unwrap().to_string(),
            "tile_load [block_shape=[2], boundary=Masked, transforms=[index(axis=0, index=1)]]",
        );

        // Long metadata wraps one field per line.
        let operation = operation.with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 3, 1)] },
        ]);
        assert_eq!(
            operation.to_string(),
            indoc! {"
                tile_load [
                    block_shape=[2],
                    boundary=Masked,
                    transforms=[index(axis=0, index=dynamic), slice(axes=[1:4])],
                    requires_runtime_assertion=true,
                ]"},
        );
    }

    #[test]
    fn test_tile_load_operation_interpretation() {
        let context = EagerContext::<ArrayIrValue<Array>, KernelOperation>::new();
        let source = ArrayReference::new(
            Array::from_elements(ArrayType::new_static(DataType::I32, [2, 3]), &[1i32, 2, 3, 4, 5, 6]).unwrap(),
        );
        let inputs = vec![
            ArrayIrValue::Reference(source.clone()),
            ArrayIrValue::Dimension(DimensionValue::constant(1).unwrap()),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
            ArrayIrValue::Array(Array::scalar(-1i32).unwrap()),
        ];
        assert_eq!(
            context.bind(TileLoadOperation::new(vec![2, 2], BoundaryPolicy::Masked).unwrap(), vec![], &inputs),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::I32, [2, 2]), &[6i32, -1, -1, -1]).unwrap(),
            )]),
        );
        assert_eq!(
            context.bind(TileLoadOperation::new(vec![1, 1], BoundaryPolicy::InBounds).unwrap(), vec![], &inputs),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::I32, [1, 1]), &[6i32]).unwrap(),
            )]),
        );
        assert_eq!(
            context.bind(TileLoadOperation::new(vec![2, 2], BoundaryPolicy::InBounds).unwrap(), vec![], &inputs),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{TILE_LOAD_OPERATION_NAME}` in-bounds window exceeds the source extent",
            )))),
        );
        assert_eq!(source.read().unwrap().elements::<i32>().unwrap(), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_tile_load_operation_interpretation_transforms() {
        let context = EagerContext::<ArrayIrValue<Array>, KernelOperation>::new();
        let source = ArrayReference::new(Array::matrix(2, 3, vec![1i32, 2, 3, 4, 5, 6]).unwrap());
        let operation = TileLoadOperation::new(vec![3], BoundaryPolicy::Masked).unwrap().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        assert_eq!(
            context.bind(
                operation,
                vec![],
                &[
                    ArrayIrValue::Reference(source),
                    ArrayIrValue::Dimension(DimensionValue::constant(1).unwrap()),
                    ArrayIrValue::Array(Array::scalar(-7i32).unwrap()),
                    ArrayIrValue::Array(Array::scalar(1i64).unwrap()),
                ],
            ),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![5i32, 6, -7]).unwrap())]),
        );
    }

    #[test]
    fn test_tile_load_operation_interpretation_empty_source_preserves_padding_bits() {
        let context = EagerContext::<ArrayIrValue<Array>, KernelOperation>::new();
        let other = Array::scalar(f32::from_bits(0x7fc01234)).unwrap();
        let inputs = vec![
            ArrayIrValue::Reference(ArrayReference::new(Array::vector(Vec::<f32>::new()).unwrap())),
            ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
            ArrayIrValue::Array(other.clone()),
        ];
        let values = context
            .bind(TileLoadOperation::new(vec![2], BoundaryPolicy::Masked).unwrap(), vec![], &inputs)
            .unwrap();
        let ArrayIrValue::Array(value) = &values[0] else { unreachable!() };
        assert_eq!(value.logical_bytes(), other.logical_bytes().repeat(2));
        assert_eq!(
            context.bind(TileLoadOperation::new(vec![0], BoundaryPolicy::Masked).unwrap(), vec![], &inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(Vec::<f32>::new()).unwrap())]),
        );
    }

    #[test]
    fn test_tile_load_operation_interpretation_overlapping_stencil_windows() {
        let source_type = ArrayType::new_static(DataType::F32, [1, 5]);
        let filter_type = ArrayType::new_static(DataType::F32, [3, 1]);
        let output_type = ArrayType::new_static(DataType::F32, [3, 1]);
        let operation = tiled_call(
            &[source_type.clone(), filter_type.clone()],
            output_type.clone(),
            2,
            vec![1, 1],
            BoundaryPolicy::Masked,
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let inputs =
            operation.body_input_types().into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let zero = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()));
        let other = builder.add_constant(ArrayIrValue::Array(Array::scalar(0.0f32).unwrap()));
        // Adjacent programs read overlapping source windows while their mapped outputs remain disjoint.
        let window = builder
            .add_instruction(
                TileLoadOperation::new(vec![1, 3], BoundaryPolicy::InBounds).unwrap(),
                vec![],
                vec![inputs[0], zero, inputs[3], other],
                None,
            )
            .unwrap()[0];
        let filter = builder
            .add_instruction(
                ArrayIrOperation::ReferenceRead(ReferenceReadOperation::new()),
                vec![],
                vec![inputs[1]],
                None,
            )
            .unwrap()[0];
        let result = builder
            .add_instruction(
                ArrayIrOperation::from(ArrayOperation::Dot(DotOperation::matmul())),
                vec![],
                vec![window, filter],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(
                ArrayIrOperation::ReferenceWrite(ReferenceWriteOperation::new()),
                vec![],
                vec![inputs[2], result],
                None,
            )
            .unwrap();
        let body = builder.build(vec![], vec![Placeholder; inputs.len()], vec![]).unwrap();
        let definition = KernelDefinition::new(operation, body).unwrap();
        assert_eq!(
            definition.interpret(
                vec![
                    Array::from_elements(source_type, &[1.0f32, 2.0, 3.0, 4.0, 5.0]).unwrap(),
                    Array::from_elements(filter_type, &[2.0f32, -1.0, 3.0]).unwrap(),
                ],
                3,
            ),
            Ok(vec![Array::from_elements(output_type, &[9.0f32, 13.0, 17.0]).unwrap()]),
        );
    }

    #[test]
    fn test_tile_load_operation_reference_access_descriptor() {
        let transforms = vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        let operation = TileLoadOperation::new(vec![2], BoundaryPolicy::Masked).unwrap().with_transforms(transforms);
        assert_eq!(operation.reference_access_descriptor(0).unwrap().bindings(), 3..4);
        assert!(operation.reference_access_descriptor(1).is_none());
    }

    #[test]
    fn test_kernel_advanced_indexing_broadcast_and_mask() {
        let definition = advanced_index_definition(KernelParameterAccess::ReadWrite).unwrap();
        let source = Array::matrix(3, 4, (0i32..12).collect()).unwrap();
        let rows = Array::matrix(2, 1, vec![2i32, 0]).unwrap();
        let columns = Array::matrix(1, 3, vec![3i32, 1, 0]).unwrap();
        let mask = Array::matrix(2, 3, vec![true, false, true, false, true, true]).unwrap();
        assert_eq!(
            definition.interpret(vec![source.clone(), rows, columns, mask], 1),
            Ok(vec![
                Array::matrix(3, 4, vec![100i32, 101, 2, 3, 4, 5, 6, 7, 108, 9, 10, 111]).unwrap(),
                Array::matrix(2, 3, vec![11i32, -7, 8, -7, 1, 0]).unwrap(),
            ]),
        );
        assert_eq!(source, Array::matrix(3, 4, (0i32..12).collect()).unwrap());
        // Explicit invalid active queries have exactly the same fill and drop semantics as inactive mask elements.
        assert_eq!(
            definition.interpret(
                vec![
                    source,
                    Array::matrix(2, 1, vec![-1i32, 0]).unwrap(),
                    Array::matrix(1, 3, vec![4i32, 1, 0]).unwrap(),
                    Array::matrix(2, 3, vec![true; 6]).unwrap(),
                ],
                1,
            ),
            Ok(vec![
                Array::matrix(3, 4, vec![100i32, 101, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).unwrap(),
                Array::matrix(2, 3, vec![-7i32, -7, -7, -7, 1, 0]).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_kernel_advanced_indexing_requires_initialized_destination() {
        // The boundary admits a read of a write-only destination, and the initialization analysis then rejects it
        // because nothing has written the destination before the scatter reads it.
        let definition = advanced_index_definition(KernelParameterAccess::WriteOnly).unwrap();
        let body = definition.body().entry_region_ref();
        assert_eq!(
            validate_kernel_initialization(body, definition.operation(), 1),
            Err(KernelInitializationError::UninitializedRead {
                instruction: InstructionId::new(body.id(), 0),
                root: ReferenceRoot::RegionInput { region: body.id(), input_index: 0 },
            }),
        );
    }
}
