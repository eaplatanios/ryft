//! Dynamically selected kernel tiles over canonical reference views and array padding.
//!
//! Broadcast advanced indexing composes the existing [`BroadcastOperation`](crate::operations::BroadcastOperation),
//! [`ConcatenateOperation`](crate::operations::ConcatenateOperation),
//! [`GatherOperation`](crate::operations::GatherOperation), and
//! [`ScatterOperation`](crate::operations::ScatterOperation). Broadcast signed integer query components to a common
//! shape, append their index-vector axis, then concatenate the components. Use
//! [`SelectOperation`](crate::operations::SelectOperation) to replace an inactive query with a negative index and
//! [`GatherMode::Fill`](crate::operations::GatherMode::Fill) to fill loads or
//! [`ScatterMode::Drop`](crate::operations::ScatterMode::Drop) to drop stores.
//! Negative indices never wrap to the opposite array edge. Gather's explicit scalar fill preserves its exact encoding.
//!
//! Read the qualified reference with [`ReferenceRead`](crate::operations::ReferenceRead) before gathering or
//! scattering; publish the result with [`ReferenceWrite`](crate::operations::ReferenceWrite). This composition reads
//! the complete source window. Indexed stores are full-window read-modify-write operations and require an already
//! initialized destination, even when the selected indices happen to cover every element. Existing reference effects
//! enforce that contract and reject unordered overlapping mutable windows. It does not establish partial write-only
//! initialization or create an additional reference-view representation. More selective reference effects require a
//! separate proof before admission; an inactive query mask alone does not waive full-window initialization.

use std::borrow::Cow;
use std::fmt::Display;
use std::sync::LazyLock;

use crate::arrays::{
    Array, ArrayIrType, ArrayIrValue, ArrayReferenceTransform, ArraySliceAxis, ArrayType, MAX_DIMENSION_EXTENT,
};
use crate::contexts::EagerContext;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::kernels::mappings::BoundaryPolicy;
use crate::macros::check_count;
use crate::operations::{Pad, PadOperation};
use crate::programs::{
    EffectClasses, Effects, Operation, OperationFormatter, ProgramError, ReferenceAccessDescriptor,
    ReferenceAccessMode, ReferenceAccessOperation, ReferenceEffect, ReferenceTransform, RegionInterface, TypeError,
    Typed, infer_reference_view_type,
};

/// Canonical operation name for [`TileLoadOperation`].
pub const TILE_LOAD_OPERATION_NAME: &str = "tile_load";

/// Reads a dynamically selected fixed-size tile from a canonical reference. Inputs are the source reference,
/// one checked dimension start per selected axis, and an explicit scalar padding value with the source dtype and
/// memory, followed by the source path bindings. The source path is applied before the tile window.
/// The result is an ordinary dense array. Masked loads create only a valid clipped reference view; all other result
/// positions contain the exact padding-value encoding. No out-of-bounds reference is constructed or accessed.
/// Initialization validation conservatively requires the entire source view initialized for dynamic selections.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TileLoadOperation {
    /// Static logical shape of each result tile.
    block_shape: Vec<usize>,

    /// Whether coordinates outside the source are rejected or filled explicitly.
    boundary_policy: BoundaryPolicy,

    /// Ordered selections applied to the source before selecting the tile window.
    transforms: Vec<ArrayReferenceTransform>,
}

impl TileLoadOperation {
    /// Declares a tile shape and boundary policy, checking the canonical dimension-width limit.
    pub fn new(block_shape: Vec<usize>, boundary_policy: BoundaryPolicy) -> Result<Self, TypeError> {
        if block_shape.iter().any(|&extent| extent > MAX_DIMENSION_EXTENT) {
            return Err(TypeError::invalid("`tile_load` block shape exceeds the canonical dimension width"));
        }
        Ok(Self { block_shape, boundary_policy, transforms: Vec::new() })
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

    /// Replaces the source selections applied before the tile window.
    pub fn with_transforms(mut self, transforms: Vec<ArrayReferenceTransform>) -> Self {
        self.transforms = transforms;
        self
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
        let ArrayIrType::Reference(reference) = &input_types[0] else {
            return Err(TypeError::invalid("`tile_load` first input must be an array reference"));
        };
        let source = infer_reference_view_type(
            reference.referent(),
            &self.transforms,
            &input_types[rank + 2..].iter().collect::<Vec<_>>(),
            ReferenceAccessMode::Read,
        )?;
        if source.rank() != rank || source.static_shape().is_none() {
            return Err(TypeError::invalid("`tile_load` source requires a static shape with the block rank"));
        }
        if input_types[1..rank + 1].iter().any(|r#type| !matches!(r#type, ArrayIrType::Dimension(_))) {
            return Err(TypeError::invalid("`tile_load` starts must be canonical dimensions"));
        }
        let ArrayIrType::Array(other) = &input_types[rank + 1] else {
            return Err(TypeError::invalid("`tile_load` padding value must be a scalar array"));
        };
        // Infer through an empty valid slice and canonical padding, so distributed-axis and padding-value rules
        // remain owned by the existing operations. Clear physical layout for the declared dense result contract.
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
        static EFFECTS: LazyLock<Effects> = LazyLock::new(|| {
            Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read }],
            )
            .unwrap()
        });
        Cow::Borrowed(&EFFECTS)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("block_shape", format_args!("{:?}", self.block_shape))?;
            operation.field("boundary", format_args!("{:?}", self.boundary_policy))?;
            if !self.transforms.is_empty() {
                operation.list("transforms", &self.transforms)?;
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
            let limit = start
                .checked_add(block)
                .filter(|&limit| limit <= MAX_DIMENSION_EXTENT)
                .ok_or_else(|| TypeError::invalid("`tile_load` window limit exceeds the canonical dimension width"))?;
            if self.boundary_policy == BoundaryPolicy::InBounds && limit > extent {
                return Err(TypeError::invalid("`tile_load` in-bounds window exceeds the source extent").into());
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

    use crate::arrays::{
        ArrayIrOperation, ArrayOperation, ArrayReference, ArrayReferenceTransformIndex, DataType, DimensionValue,
    };
    use crate::contexts::Context;
    use crate::kernels::authoring::{tiled_call, whole_array_parameter};
    use crate::kernels::calls::{KernelCallOperation, KernelDefinition, KernelError};
    use crate::kernels::grids::Grid;
    use crate::kernels::operations::KernelOperation;
    use crate::kernels::validation::{KernelParameterAccess, KernelValidationError};
    use crate::operations::{
        AddOperation, BroadcastOperation, ConcatenateOperation, DotOperation, GatherDimensionNumbers, GatherMode,
        GatherOperation, ReferenceReadOperation, ReferenceWriteOperation, ScatterDimensionNumbers, ScatterMode,
        ScatterOperation, ScatterReductionKind, SelectOperation,
    };
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ReferenceType};

    use super::*;

    /// Broadcasts row/column queries and a mask into canonical gather/scatter index vectors.
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
        // Fill and drop modes treat the negative sentinel as invalid; it never wraps to the final source row.
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

    #[test]
    fn test_tile_load_operation_new() {
        let operation = TileLoadOperation::new(vec![2, 3], BoundaryPolicy::Masked).unwrap();
        assert_eq!(operation.block_shape(), &[2, 3]);
        assert_eq!(operation.boundary_policy(), BoundaryPolicy::Masked);
        assert_eq!(operation.name(), "tile_load");
        assert_eq!(format!("{operation}"), "tile_load [block_shape=[2, 3], boundary=Masked]");
        assert_eq!(
            TileLoadOperation::new(vec![MAX_DIMENSION_EXTENT + 1], BoundaryPolicy::Masked),
            Err(TypeError::invalid("`tile_load` block shape exceeds the canonical dimension width"))
        );
    }

    #[test]
    fn test_tile_load_operation_with_transforms() {
        let transforms = vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        let operation =
            TileLoadOperation::new(vec![2], BoundaryPolicy::Masked).unwrap().with_transforms(transforms.clone());
        assert_eq!(operation.transforms(), transforms);
        assert_eq!(operation.reference_access_descriptor(0).unwrap().bindings(), 3..4);
        assert!(operation.reference_access_descriptor(1).is_none());
    }

    #[test]
    fn test_tile_load_operation_infer_output_types() {
        let operation = TileLoadOperation::new(vec![2], BoundaryPolicy::Masked).unwrap();
        let inputs = vec![
            ReferenceType::new(ArrayType::new_static(DataType::F32, [3])).into(),
            ArrayIrType::Dimension(DimensionValue::constant(1).unwrap().r#type().into_owned()),
            ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
        ];
        assert_eq!(
            operation.infer_output_types(&inputs, &[]),
            Ok(vec![ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2]))])
        );
        let mut invalid = inputs;
        invalid[1] = ArrayIrType::Array(ArrayType::scalar(DataType::I64));
        assert_eq!(
            operation.infer_output_types(&invalid, &[]),
            Err(TypeError::invalid("`tile_load` starts must be canonical dimensions"))
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
                Array::from_elements(ArrayType::new_static(DataType::I32, [2, 2]), &[6i32, -1, -1, -1]).unwrap()
            )])
        );
        assert_eq!(
            context.bind(TileLoadOperation::new(vec![1, 1], BoundaryPolicy::InBounds).unwrap(), vec![], &inputs),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::I32, [1, 1]), &[6i32]).unwrap()
            )])
        );
        assert_eq!(
            context.bind(TileLoadOperation::new(vec![2, 2], BoundaryPolicy::InBounds).unwrap(), vec![], &inputs),
            Err(ProgramError::Type(TypeError::invalid("`tile_load` in-bounds window exceeds the source extent")))
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
                ]
            ),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![5i32, 6, -7]).unwrap())])
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
            Ok(vec![ArrayIrValue::Array(Array::vector(Vec::<f32>::new()).unwrap())])
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
                3
            ),
            Ok(vec![Array::from_elements(output_type, &[9.0f32, 13.0, 17.0]).unwrap()])
        );
    }

    #[test]
    fn test_tile_load_operation_effects() {
        assert_eq!(
            TileLoadOperation::new(vec![2], BoundaryPolicy::Masked).unwrap().effects().as_ref(),
            &Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read }],
            )
            .unwrap()
        );
    }

    #[test]
    fn test_tile_load_operation_render() {
        let operation = TileLoadOperation::new(vec![2], BoundaryPolicy::Masked).unwrap();
        assert_eq!(operation.to_string(), "tile_load [block_shape=[2], boundary=Masked]");
        let operation = operation.with_transforms(vec![ArrayReferenceTransform::Index {
            axis: 0,
            index: ArrayReferenceTransformIndex::Static(1),
        }]);
        assert_eq!(
            operation.to_string(),
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
                ]"},
        );
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
        // Explicit invalid active queries have exactly the same fill/drop semantics as inactive mask lanes.
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
        let error = advanced_index_definition(KernelParameterAccess::WriteOnly).unwrap_err();
        let KernelError::Validation(KernelValidationError::DisallowedAccess {
            input_index,
            access,
            mode,
            operation,
            instruction,
        }) = &error
        else {
            panic!("unexpected error: {error}")
        };
        assert_eq!(*input_index, 0);
        assert_eq!(*access, KernelParameterAccess::WriteOnly);
        assert_eq!(*mode, ReferenceAccessMode::Read);
        assert_eq!(*operation, "reference_read");
        assert_eq!(instruction.region().index(), 0);
        assert_eq!(instruction.index(), 0);
        assert_eq!(
            error.to_string(),
            "operation `reference_read` at ^0[0] performs a `read` access on kernel input 0, which the boundary \
             contract declares write-only",
        );
    }
}
