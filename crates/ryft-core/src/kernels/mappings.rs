//! Pure dimension programs describing kernel operand windows.
//!
//! Mappings reuse ordinary Ryft programs and checked dimension arithmetic. Their boundary accepts host-bound grid
//! coordinates, extents, and static parameters, never arrays or mutable references. The initial straight-line subset
//! excludes control flow and array-to-dimension conversion. Prefetched scalar arrays require a separate, explicit
//! binding contract before they can enter this subset.

use thiserror::Error;

use crate::arrays::{
    Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayReferenceTransform, ArraySliceAxis, ArrayTypeRefinements,
    DimensionOperation, DimensionValue, MAX_DIMENSION_EXTENT,
};
use crate::contexts::EagerContext;
use crate::parameters::Placeholder;
use crate::programs::{
    Atom, FlatProgram, Operation, ProgramBuilder, ProgramError, ReferenceViewOverlap, TypeError, Typed,
};

/// Invalid mapping program, index arithmetic, or operand window.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum BlockMappingError {
    /// The existing interpreter rejected dimension bindings or arithmetic.
    #[error(transparent)]
    Program(#[from] ProgramError),

    /// A program value carries an array or reference instead of a dimension.
    #[error("block mapping atom {position} must be a dimension, but has type `{type}`")]
    NonDimension {
        /// Atom position in the mapping entry region.
        position: usize,
        /// Rejected canonical value type.
        r#type: ArrayIrType,
    },

    /// An operation is outside the supported pure, straight-line dimension subset.
    #[error("block mapping operation `{operation}` is not a supported straight-line dimension operation")]
    UnsupportedOperation {
        /// Canonical operation name.
        operation: &'static str,
    },

    /// A mapping output or operand shape has the wrong rank.
    #[error("block mapping {boundary} has rank {actual}, but the block shape has rank {expected}")]
    RankMismatch {
        /// Mapping output or operand shape being checked.
        boundary: &'static str,
        /// Static block rank.
        expected: usize,
        /// Observed boundary rank.
        actual: usize,
    },

    /// An extent or window limit exceeds the canonical dimension width.
    #[error("block mapping {boundary} at axis {axis} exceeds the maximum dimension extent {MAX_DIMENSION_EXTENT}")]
    Overflow {
        /// Block shape, operand shape, or window limit being checked.
        boundary: &'static str,
        /// Axis whose extent or arithmetic exceeded the dimension width.
        axis: usize,
    },

    /// An unmasked window reaches outside its operand.
    #[error("block mapping axis {axis} selects `{start}..{limit}` outside operand extent {extent}")]
    OutOfBounds {
        /// Operand axis whose window is invalid.
        axis: usize,
        /// Inclusive window start.
        start: usize,
        /// Exclusive window limit.
        limit: usize,
        /// Concrete operand extent.
        extent: usize,
    },
}

/// Required handling of a window that crosses an operand boundary.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BoundaryPolicy {
    /// Every selected coordinate must belong to the operand.
    InBounds,

    /// Consumers must explicitly mask invalid coordinates before memory access. This policy does not authorize
    /// unguarded out-of-bounds execution or choose a padding value for loads.
    Masked,
}

/// Immutable dimension program returning one element start per axis of a statically shaped block.
///
/// Program inputs retain their canonical dimension identities and bounds. Supplying concrete bindings checks those
/// bounds through ordinary program interpretation; checked arithmetic errors propagate without wrapping indices.
/// The program contains only dimension atoms and dimension-family operations, including dimension requirements.
#[derive(Clone, Debug)]
pub struct BlockMapping {
    /// Validated pure program; its outputs are element starts, not tile numbers.
    program: FlatProgram<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>,
    /// Logical extents of each block, including any explicitly masked coordinates.
    block_shape: Vec<usize>,
    /// Required treatment of coordinates outside the operand.
    boundary_policy: BoundaryPolicy,
}

impl BlockMapping {
    /// Validates a dimension-only mapping with one output per block axis. Zero-sized blocks and rank-zero blocks are
    /// allowed. Every instruction is checked, including instructions whose results are unused.
    pub fn new(
        program: FlatProgram<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>,
        block_shape: Vec<usize>,
        boundary_policy: BoundaryPolicy,
    ) -> Result<Self, BlockMappingError> {
        if program.output_types().len() != block_shape.len() {
            return Err(BlockMappingError::RankMismatch {
                boundary: "output",
                expected: block_shape.len(),
                actual: program.output_types().len(),
            });
        }
        for (axis, &extent) in block_shape.iter().enumerate() {
            if extent > MAX_DIMENSION_EXTENT {
                return Err(BlockMappingError::Overflow { boundary: "block shape", axis });
            }
        }
        let region = program.entry_region_ref();
        for instruction in region.instructions() {
            if !matches!(instruction.operation(), ArrayIrOperation::Dimension(_)) || !instruction.regions().is_empty() {
                return Err(BlockMappingError::UnsupportedOperation { operation: instruction.operation().name() });
            }
        }
        for (position, atom) in region.atoms().iter().enumerate() {
            if !matches!(atom.r#type().as_ref(), ArrayIrType::Dimension(_)) {
                return Err(BlockMappingError::NonDimension { position, r#type: atom.r#type().into_owned() });
            }
        }
        Ok(Self { program, block_shape, boundary_policy })
    }

    /// Returns the validated immutable mapping program for tracing, specialization, or lowering.
    pub fn program(&self) -> &FlatProgram<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>> {
        &self.program
    }

    /// Returns the static logical shape, including coordinates that require masking.
    pub fn block_shape(&self) -> &[usize] {
        &self.block_shape
    }

    /// Returns the required operand boundary handling.
    pub fn boundary_policy(&self) -> BoundaryPolicy {
        self.boundary_policy
    }

    /// Binds trailing static dimension parameters to canonical constants, leaving earlier grid coordinates as inputs.
    /// The ordinary program splicer checks dimension bounds and identity relationships and preserves checked
    /// arithmetic without executing it. A bound parameter cannot share its identity with a remaining grid input:
    /// that would impose a runtime equality which constant substitution alone cannot preserve. No reference or
    /// device effect can occur in a validated mapping.
    pub fn specialize(&self, values: &[DimensionValue]) -> Result<Self, BlockMappingError> {
        let input_count = self.program.input_types().len();
        let remaining = input_count.checked_sub(values.len()).ok_or(BlockMappingError::RankMismatch {
            boundary: "specialization",
            expected: input_count,
            actual: values.len(),
        })?;
        let mut builder = ProgramBuilder::new();
        let mut inputs = self
            .program
            .input_types()
            .into_iter()
            .take(remaining)
            .map(|r#type| builder.add_input(r#type))
            .collect::<Vec<_>>();
        let mut bindings = ArrayTypeRefinements::default();
        for (offset, (formal, value)) in self.program.input_types().into_iter().skip(remaining).zip(values).enumerate()
        {
            let ArrayIrType::Dimension(formal) = formal else { unreachable!() };
            let position = remaining + offset;
            for (input, r#type) in self.program.input_types().iter().take(remaining).enumerate() {
                let ArrayIrType::Dimension(r#type) = r#type else { unreachable!() };
                if r#type.variable() == formal.variable() {
                    return Err(ProgramError::from(TypeError::invalid(format!(
                        "block mapping parameter {position} shares a dimension identity with remaining input {input}",
                    )))
                    .into());
                }
            }
            let value = DimensionValue::new(formal.clone(), value.extent())
                .map_err(TypeError::from)
                .map_err(ProgramError::from)?;
            bindings.bind(formal.variable(), value.extent()).map_err(ProgramError::from)?;
            inputs.push(builder.add_constant(ArrayIrValue::Dimension(value)));
        }
        let outputs = builder.splice_program(&self.program, &inputs)?;
        let program =
            builder.build(outputs, vec![Placeholder; remaining], vec![Placeholder; self.block_shape.len()])?;
        Self::new(program, self.block_shape.clone(), self.boundary_policy)
    }

    /// Recognizes separable tiling directly from canonical mapping instructions. Each result is either a grid-input
    /// axis whose coordinate is multiplied by the corresponding block extent, or `None` for a constant zero start.
    /// A unit block also admits the coordinate directly. Unknown expressions return `None` for the complete mapping.
    ///
    /// This describes index geometry only: callers still check concrete bounds, repeated/unused grid axes, coverage,
    /// and access policy. Every instruction must contribute a recognized output, so this proof never skips a dead
    /// arithmetic instruction or a runtime dimension requirement that ordinary mapping evaluation would execute.
    pub fn tiling_axes(&self) -> Option<Vec<Option<usize>>> {
        if self.program.instructions().iter().any(|instruction| {
            instruction.outputs().len() != 1
                || !self.program.output_ids().contains(&instruction.outputs()[0])
                || !matches!(instruction.operation(), ArrayIrOperation::Dimension(DimensionOperation::Mul(_)))
        }) {
            return None;
        }
        self.program
            .output_ids()
            .iter()
            .zip(&self.block_shape)
            .map(|(&output, &extent)| {
                if let Atom::Constant(ArrayIrValue::Dimension(value)) = &self.program.atoms()[output.index()]
                    && value.extent() == 0
                {
                    return Some(None);
                }
                if extent == 1
                    && let Some(axis) = self.program.input_ids().iter().position(|&input| input == output)
                {
                    return Some(Some(axis));
                }
                let instruction =
                    self.program.instructions().iter().find(|instruction| instruction.outputs() == [output])?;
                let inputs = instruction.inputs();
                for (coordinate, factor) in [(inputs[0], inputs[1]), (inputs[1], inputs[0])] {
                    if let Some(axis) = self.program.input_ids().iter().position(|&input| input == coordinate)
                        && let Atom::Constant(ArrayIrValue::Dimension(value)) = &self.program.atoms()[factor.index()]
                        && value.extent() == extent
                    {
                        return Some(Some(axis));
                    }
                }
                None
            })
            .collect()
    }

    /// Interprets dimension bindings and intersects the resulting block with the concrete operand shape. The returned
    /// valid transform selects only valid memory; the original starts remain available for constructing explicit masks.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: dimension bindings in the mapping program's input order.
    ///   - `operand_shape`: concrete root extents in block-axis order.
    pub fn evaluate(
        &self,
        inputs: &[DimensionValue],
        operand_shape: &[usize],
    ) -> Result<BlockWindow, BlockMappingError> {
        if operand_shape.len() != self.block_shape.len() {
            return Err(BlockMappingError::RankMismatch {
                boundary: "operand",
                expected: self.block_shape.len(),
                actual: operand_shape.len(),
            });
        }
        let outputs = self.program.interpret(inputs.iter().cloned().map(ArrayIrValue::Dimension).collect())?;
        let mut starts = Vec::with_capacity(outputs.len());
        let mut axes = Vec::with_capacity(outputs.len());
        let mut requires_mask = false;
        for (axis, ((output, &size), &extent)) in
            outputs.into_iter().zip(&self.block_shape).zip(operand_shape).enumerate()
        {
            if extent > MAX_DIMENSION_EXTENT {
                return Err(BlockMappingError::Overflow { boundary: "operand shape", axis });
            }
            let ArrayIrValue::Dimension(start) = output else {
                unreachable!("validated mapping outputs are dimensions");
            };
            let start = start.extent();
            let limit = start
                .checked_add(size)
                .filter(|&limit| limit <= MAX_DIMENSION_EXTENT)
                .ok_or(BlockMappingError::Overflow { boundary: "window limit", axis })?;
            if start > extent || limit > extent {
                if self.boundary_policy == BoundaryPolicy::InBounds {
                    return Err(BlockMappingError::OutOfBounds { axis, start, limit, extent });
                }
                requires_mask = true;
            }
            starts.push(start);
            axes.push(ArraySliceAxis::new(start.min(extent), limit.min(extent) - start.min(extent), 1));
        }
        Ok(BlockWindow { starts, valid_transform: ArrayReferenceTransform::Slice { axes }, requires_mask })
    }
}

/// Evaluated block starts and their valid operand intersection. No storage or reference handle is owned here.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlockWindow {
    /// Original element starts before intersecting with the operand bounds.
    starts: Vec<usize>,
    /// Canonical unit-stride slice containing only valid operand coordinates.
    valid_transform: ArrayReferenceTransform,
    /// Whether explicit masks are required before using the full logical block.
    requires_mask: bool,
}

impl BlockWindow {
    /// Returns the original starts used with the mapping's logical block shape to construct masks.
    pub fn starts(&self) -> &[usize] {
        &self.starts
    }

    /// Returns the canonical transform that selects the valid intersection, which may be empty.
    pub fn valid_transform(&self) -> &ArrayReferenceTransform {
        &self.valid_transform
    }

    /// Returns whether the original logical window extends beyond the operand.
    pub fn requires_mask(&self) -> bool {
        self.requires_mask
    }

    /// Compares valid coordinates of windows on the same operand root. Empty intersections are disjoint, identical
    /// nonempty intersections are the same, and partially intersecting windows may overlap. Different ranks are
    /// conservatively reported as possibly overlapping; root identity remains the caller's responsibility.
    pub fn overlap(&self, other: &Self) -> ReferenceViewOverlap {
        let (ArrayReferenceTransform::Slice { axes: left }, ArrayReferenceTransform::Slice { axes: right }) =
            (&self.valid_transform, &other.valid_transform)
        else {
            unreachable!("block windows contain canonical slices");
        };
        if left.iter().chain(right).any(|axis| axis.size() == 0) {
            return ReferenceViewOverlap::Disjoint;
        }
        if left.len() != right.len() {
            return ReferenceViewOverlap::MayOverlap;
        }
        if left.iter().zip(right).any(|(left, right)| {
            left.start() + left.size() <= right.start() || right.start() + right.size() <= left.start()
        }) {
            return ReferenceViewOverlap::Disjoint;
        }
        if left == right { ReferenceViewOverlap::Same } else { ReferenceViewOverlap::MayOverlap }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{ArrayType, DataType, DimensionBounds, DimensionError, DimensionType};
    use crate::operations::{DimensionMulOperation, ZERO_OPERATION_NAME, ZeroOperation};
    use crate::parameters::{ParameterError, Placeholder};
    use crate::programs::{ProgramBuilder, ReferenceType};

    use super::*;

    /// Builds a dimension-only identity mapping with independently bounded coordinate inputs.
    fn identity_program(rank: usize) -> FlatProgram<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>> {
        let mut builder = ProgramBuilder::new();
        let outputs = (0..rank)
            .map(|axis| {
                builder.add_input(
                    DimensionType::new(format!("axis_{axis}"), DimensionBounds::non_negative(None).unwrap()).into(),
                )
            })
            .collect();
        builder.build(outputs, vec![Placeholder; rank], vec![Placeholder; rank]).unwrap()
    }

    /// Binds extents to the mapping's canonical input identities.
    fn inputs(mapping: &BlockMapping, extents: &[usize]) -> Vec<DimensionValue> {
        mapping
            .program()
            .input_types()
            .into_iter()
            .zip(extents)
            .map(|(r#type, &extent)| {
                let ArrayIrType::Dimension(r#type) = r#type else { unreachable!() };
                DimensionValue::new(r#type, extent).unwrap()
            })
            .collect()
    }

    #[test]
    fn test_block_mapping_new() {
        let mapping = BlockMapping::new(identity_program(2), vec![4, 8], BoundaryPolicy::InBounds).unwrap();
        assert_eq!(mapping.program().output_types().len(), 2);
        assert_eq!(mapping.block_shape(), &[4, 8]);
        assert_eq!(mapping.boundary_policy(), BoundaryPolicy::InBounds);
    }

    #[test]
    fn test_block_mapping_new_rejects_shape_errors() {
        assert!(matches!(
            BlockMapping::new(identity_program(1), vec![4, 8], BoundaryPolicy::InBounds),
            Err(BlockMappingError::RankMismatch { boundary: "output", expected: 2, actual: 1 }),
        ));
        assert!(matches!(
            BlockMapping::new(identity_program(1), vec![MAX_DIMENSION_EXTENT + 1], BoundaryPolicy::InBounds),
            Err(BlockMappingError::Overflow { boundary: "block shape", axis: 0 }),
        ));
    }

    #[test]
    fn test_block_mapping_new_rejects_arrays_and_references() {
        let array_type = ArrayType::scalar(DataType::I32);
        for r#type in [ArrayIrType::Array(array_type.clone()), ReferenceType::new(array_type).into()] {
            let mut builder = ProgramBuilder::new();
            builder.add_input(r#type.clone());
            let output = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()));
            let program = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
            assert!(matches!(
                BlockMapping::new(program, vec![1], BoundaryPolicy::InBounds),
                Err(BlockMappingError::NonDimension { position: 0, r#type: actual }) if actual == r#type,
            ));
        }
    }

    #[test]
    fn test_block_mapping_new_rejects_unused_array_operations() {
        let mut builder = ProgramBuilder::new();
        let output = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()));
        builder
            .add_instruction(ZeroOperation::new(ArrayType::scalar(DataType::I32)), Vec::new(), Vec::new(), None)
            .unwrap();
        let program = builder.build(vec![output], Vec::new(), vec![Placeholder]).unwrap();
        assert!(matches!(
            BlockMapping::new(program, vec![1], BoundaryPolicy::Masked),
            Err(BlockMappingError::UnsupportedOperation { operation: ZERO_OPERATION_NAME }),
        ));
    }

    #[test]
    fn test_block_mapping_program() {
        let program = identity_program(1);
        let input_types = program.input_types();
        let output_types = program.output_types();
        let mapping = BlockMapping::new(program, vec![4], BoundaryPolicy::InBounds).unwrap();
        assert_eq!(mapping.program().input_types(), input_types);
        assert_eq!(mapping.program().output_types(), output_types);
    }

    #[test]
    fn test_block_mapping_block_shape() {
        let mapping = BlockMapping::new(identity_program(2), vec![0, 4], BoundaryPolicy::InBounds).unwrap();
        assert_eq!(mapping.block_shape(), &[0, 4]);
    }

    #[test]
    fn test_block_mapping_boundary_policy() {
        let mapping = BlockMapping::new(identity_program(0), vec![], BoundaryPolicy::Masked).unwrap();
        assert_eq!(mapping.boundary_policy(), BoundaryPolicy::Masked);
    }

    #[test]
    fn test_block_mapping_specialize() {
        let mapping = BlockMapping::new(identity_program(2), vec![1, 1], BoundaryPolicy::InBounds).unwrap();
        let specialized = mapping.specialize(&[DimensionValue::constant(3).unwrap()]).unwrap();
        assert_eq!(specialized.program().input_types().len(), 1);
        assert_eq!(specialized.evaluate(&[DimensionValue::constant(2).unwrap()], &[4, 4]).unwrap().starts(), &[2, 3]);
        assert_eq!(mapping.program().input_types().len(), 2);
    }

    #[test]
    fn test_block_mapping_specialize_checks_shared_identity() {
        let variable = crate::arrays::DimensionVariable::new(
            "offset",
            crate::arrays::DimensionBounds::non_negative(Some(4)).unwrap(),
        );
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let first = builder.add_input(crate::arrays::DimensionType::from(variable.clone()).into());
        let second = builder.add_input(crate::arrays::DimensionType::from(variable.clone()).into());
        let mapping = BlockMapping::new(
            builder.build(vec![first, second], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap(),
            vec![1, 1],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        let expected = TypeError::from(crate::arrays::DimensionError::InputDimensionMismatch {
            dimension: "offset".to_owned(),
            expected: 1,
            actual: 2,
        });
        assert_eq!(
            mapping
                .specialize(&[DimensionValue::constant(1).unwrap(), DimensionValue::constant(2).unwrap()])
                .unwrap_err(),
            BlockMappingError::Program(ProgramError::from(expected)),
        );
        let expected = TypeError::from(crate::arrays::DimensionError::BindingOutOfBounds {
            variable: "offset".to_owned(),
            value: 4,
            bounds: variable.bounds(),
        });
        assert_eq!(
            mapping
                .specialize(&[DimensionValue::constant(4).unwrap(), DimensionValue::constant(4).unwrap()])
                .unwrap_err(),
            BlockMappingError::Program(ProgramError::from(expected))
        );
    }

    #[test]
    fn test_block_mapping_specialize_rejects_identity_shared_with_remaining_input() {
        let variable = crate::arrays::DimensionVariable::new(
            "coordinate",
            crate::arrays::DimensionBounds::non_negative(Some(4)).unwrap(),
        );
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        builder.add_input(crate::arrays::DimensionType::from(variable.clone()).into());
        let output = builder.add_input(crate::arrays::DimensionType::from(variable).into());
        let mapping = BlockMapping::new(
            builder.build(vec![output], vec![Placeholder; 2], vec![Placeholder]).unwrap(),
            vec![1],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        assert_eq!(
            mapping.specialize(&[DimensionValue::constant(1).unwrap()]).unwrap_err(),
            BlockMappingError::Program(ProgramError::from(TypeError::invalid(
                "block mapping parameter 1 shares a dimension identity with remaining input 0"
            )),)
        );
    }

    #[test]
    fn test_block_mapping_tiling_axes() {
        let mapping = BlockMapping::new(identity_program(2), vec![1, 1], BoundaryPolicy::InBounds).unwrap();
        assert_eq!(mapping.tiling_axes(), Some(vec![Some(0), Some(1)]));
        let mapping = BlockMapping::new(identity_program(2), vec![1, 2], BoundaryPolicy::InBounds).unwrap();
        assert_eq!(mapping.tiling_axes(), None);
        let mapping = BlockMapping::new(identity_program(0), vec![], BoundaryPolicy::InBounds).unwrap();
        assert_eq!(mapping.tiling_axes(), Some(vec![]));

        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let coordinate_type = DimensionType::new("coordinate", DimensionBounds::unbounded());
        let coordinate = builder.add_input(ArrayIrType::Dimension(coordinate_type.clone()));
        let factor = DimensionValue::constant(4).unwrap();
        let factor_type = factor.r#type().into_owned();
        let factor = builder.add_constant(ArrayIrValue::Dimension(factor));
        let start = builder
            .add_instruction(
                DimensionMulOperation::new(&factor_type, &coordinate_type).unwrap(),
                vec![],
                vec![factor, coordinate],
                None,
            )
            .unwrap()[0];
        let zero = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()));
        let program = builder.build(vec![start, zero], vec![Placeholder], vec![Placeholder; 2]).unwrap();
        let mapping = BlockMapping::new(program.clone(), vec![4, 8], BoundaryPolicy::InBounds).unwrap();
        assert_eq!(mapping.tiling_axes(), Some(vec![Some(0), None]));
        let mapping = BlockMapping::new(program, vec![2, 8], BoundaryPolicy::InBounds).unwrap();
        assert_eq!(mapping.tiling_axes(), None);
    }

    #[test]
    fn test_block_mapping_tiling_axes_rejects_dead_arithmetic() {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let coordinate_type = DimensionType::new("coordinate", DimensionBounds::unbounded());
        let coordinate = builder.add_input(ArrayIrType::Dimension(coordinate_type.clone()));
        let factor = DimensionValue::constant(4).unwrap();
        let factor_type = factor.r#type().into_owned();
        let factor = builder.add_constant(ArrayIrValue::Dimension(factor));
        builder
            .add_instruction(
                DimensionMulOperation::new(&coordinate_type, &factor_type).unwrap(),
                vec![],
                vec![coordinate, factor],
                None,
            )
            .unwrap();
        let zero = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()));
        let mapping = BlockMapping::new(
            builder.build(vec![zero], vec![Placeholder], vec![Placeholder]).unwrap(),
            vec![4],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        assert_eq!(mapping.tiling_axes(), None);
    }

    #[test]
    fn test_block_mapping_evaluate() {
        let mut builder = ProgramBuilder::new();
        let coordinate_type = DimensionType::new("coordinate", DimensionBounds::non_negative(Some(4)).unwrap());
        let tile_size = DimensionValue::constant(4).unwrap();
        let coordinate = builder.add_input(coordinate_type.clone().into());
        let size = builder.add_constant(ArrayIrValue::Dimension(tile_size.clone()));
        let start = builder
            .add_instruction(
                DimensionMulOperation::new(&coordinate_type, tile_size.r#type().as_ref()).unwrap(),
                Vec::new(),
                vec![coordinate, size],
                None,
            )
            .unwrap()[0];
        let program = builder.build(vec![start], vec![Placeholder], vec![Placeholder]).unwrap();
        let mapping = BlockMapping::new(program, vec![4], BoundaryPolicy::InBounds).unwrap();
        for coordinate in 0..4 {
            let window = mapping.evaluate(&inputs(&mapping, &[coordinate]), &[16]).unwrap();
            assert_eq!(window.starts(), &[coordinate * 4]);
            assert_eq!(
                window.valid_transform(),
                &ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(coordinate * 4, 4, 1)] },
            );
            assert!(!window.requires_mask());
        }
    }

    #[test]
    fn test_block_mapping_evaluate_partial_and_empty_tiles() {
        let mapping = BlockMapping::new(identity_program(1), vec![4], BoundaryPolicy::Masked).unwrap();
        for (start, valid_start, valid_size) in [(8, 8, 2), (10, 10, 0), (12, 10, 0)] {
            let window = mapping.evaluate(&inputs(&mapping, &[start]), &[10]).unwrap();
            assert_eq!(window.starts(), &[start]);
            assert_eq!(
                window.valid_transform(),
                &ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(valid_start, valid_size, 1)] },
            );
            assert!(window.requires_mask());
        }
        let empty = mapping.evaluate(&inputs(&mapping, &[0]), &[0]).unwrap();
        assert_eq!(
            empty.valid_transform(),
            &ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 0, 1)] }
        );
    }

    #[test]
    fn test_block_mapping_evaluate_rejects_unmasked_bounds_and_overflow() {
        let mapping = BlockMapping::new(identity_program(1), vec![4], BoundaryPolicy::InBounds).unwrap();
        assert!(matches!(
            mapping.evaluate(&inputs(&mapping, &[8]), &[10]),
            Err(BlockMappingError::OutOfBounds { axis: 0, start: 8, limit: 12, extent: 10 }),
        ));
        assert!(matches!(
            mapping.evaluate(&inputs(&mapping, &[MAX_DIMENSION_EXTENT]), &[MAX_DIMENSION_EXTENT]),
            Err(BlockMappingError::Overflow { boundary: "window limit", axis: 0 }),
        ));
        assert!(matches!(
            mapping.evaluate(&inputs(&mapping, &[0]), &[MAX_DIMENSION_EXTENT + 1]),
            Err(BlockMappingError::Overflow { boundary: "operand shape", axis: 0 }),
        ));
        assert!(matches!(
            mapping.evaluate(&inputs(&mapping, &[0]), &[4, 4]),
            Err(BlockMappingError::RankMismatch { boundary: "operand", expected: 1, actual: 2 }),
        ));
        assert!(matches!(
            mapping.evaluate(&[], &[4]),
            Err(BlockMappingError::Program(ProgramError::Parameter(ParameterError::MismatchedParameterStructures {
                left_structure,
                right_structure,
            }))) if left_structure == "[<Parameter>]" && right_structure == "[]",
        ));
    }

    #[test]
    fn test_block_mapping_evaluate_checked_dimension_arithmetic() {
        let mut builder = ProgramBuilder::new();
        let coordinate_type = DimensionType::new("coordinate", DimensionBounds::non_negative(None).unwrap());
        let coordinate = builder.add_input(coordinate_type.clone().into());
        let start = builder
            .add_instruction(
                DimensionMulOperation::new(&coordinate_type, &coordinate_type).unwrap(),
                Vec::new(),
                vec![coordinate, coordinate],
                None,
            )
            .unwrap()[0];
        let program = builder.build(vec![start], vec![Placeholder], vec![Placeholder]).unwrap();
        let mapping = BlockMapping::new(program, vec![1], BoundaryPolicy::Masked).unwrap();
        let error = mapping.evaluate(&inputs(&mapping, &[MAX_DIMENSION_EXTENT]), &[1]).unwrap_err();
        let BlockMappingError::Program(error) = error else { panic!("expected dimension arithmetic error") };
        assert!(matches!(error.downcast_custom::<DimensionError>(), Some(DimensionError::ArithmeticOverflow { .. })));
    }

    #[test]
    fn test_block_mapping_evaluate_rank_zero_and_zero_block() {
        let scalar = BlockMapping::new(identity_program(0), vec![], BoundaryPolicy::InBounds).unwrap();
        let window = scalar.evaluate(&[], &[]).unwrap();
        assert_eq!(window.starts(), &[] as &[usize]);
        assert_eq!(window.valid_transform(), &ArrayReferenceTransform::Slice { axes: vec![] });
        assert!(!window.requires_mask());
        let empty = BlockMapping::new(identity_program(1), vec![0], BoundaryPolicy::InBounds).unwrap();
        assert_eq!(
            empty.evaluate(&inputs(&empty, &[0]), &[0]).unwrap().valid_transform(),
            &ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 0, 1)] },
        );
    }

    #[test]
    fn test_block_window_starts() {
        let mapping = BlockMapping::new(identity_program(1), vec![4], BoundaryPolicy::Masked).unwrap();
        assert_eq!(mapping.evaluate(&inputs(&mapping, &[12]), &[10]).unwrap().starts(), &[12]);
    }

    #[test]
    fn test_block_window_valid_transform() {
        let mapping = BlockMapping::new(identity_program(2), vec![4, 8], BoundaryPolicy::Masked).unwrap();
        assert_eq!(
            mapping.evaluate(&inputs(&mapping, &[2, 4]), &[5, 7]).unwrap().valid_transform(),
            &ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(2, 3, 1), ArraySliceAxis::new(4, 3, 1)] },
        );
    }

    #[test]
    fn test_block_window_requires_mask() {
        let mapping = BlockMapping::new(identity_program(1), vec![4], BoundaryPolicy::Masked).unwrap();
        assert!(!mapping.evaluate(&inputs(&mapping, &[0]), &[4]).unwrap().requires_mask());
        assert!(mapping.evaluate(&inputs(&mapping, &[1]), &[4]).unwrap().requires_mask());
    }

    #[test]
    fn test_block_window_overlap() {
        let mapping = BlockMapping::new(identity_program(2), vec![4, 4], BoundaryPolicy::Masked).unwrap();
        let first = mapping.evaluate(&inputs(&mapping, &[0, 0]), &[8, 8]).unwrap();
        for (starts, expected) in [
            ([0, 0], ReferenceViewOverlap::Same),
            ([4, 0], ReferenceViewOverlap::Disjoint),
            ([0, 4], ReferenceViewOverlap::Disjoint),
            ([2, 2], ReferenceViewOverlap::MayOverlap),
            ([8, 8], ReferenceViewOverlap::Disjoint),
        ] {
            let other = mapping.evaluate(&inputs(&mapping, &starts), &[8, 8]).unwrap();
            assert_eq!(first.overlap(&other), expected);
            assert_eq!(other.overlap(&first), expected);
        }
        let scalar = BlockMapping::new(identity_program(0), vec![], BoundaryPolicy::InBounds).unwrap();
        let scalar = scalar.evaluate(&[], &[]).unwrap();
        assert_eq!(scalar.overlap(&scalar), ReferenceViewOverlap::Same);
        assert_eq!(first.overlap(&scalar), ReferenceViewOverlap::MayOverlap);
    }
}
