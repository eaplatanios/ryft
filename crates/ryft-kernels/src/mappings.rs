//! Block mappings, which select the window of a full array that each program of a kernel call accesses.
//!
//! Every [`KernelParameter`](crate::calls::KernelParameter) of a kernel call owns a [`BlockMapping`], which consists of
//! a pure dimension program (i.e., an ordinary Ryft [`FlatProgram`] over dimension values), a static block shape, and
//! a [`BoundaryPolicy`]. The inputs of the program are the grid coordinates of one program followed by the scalar
//! prefetched dimensions of the call, and its outputs are one element start per block axis (i.e., element offsets,
//! not tile numbers). The window of that program starts at those element starts and has the block shape as its
//! extents. Block extents are logical, so they include any coordinates that a masked window must exclude.
//!
//! [`BlockMapping::new`] admits only dimension values and dimension operations without attached regions, including
//! dimension requirements. A mapping therefore contains no arrays, references, or control flow, and it can have no
//! reference or device effect. Its arithmetic is the checked dimension arithmetic of `ryft-core`, so overflow
//! is reported as an error instead of wrapping.
//!
//! # Boundaries and Evaluation
//!
//! A window may extend past the end of its full array (e.g., the last tile of an extent that the block extent does
//! not divide). [`BoundaryPolicy::InBounds`] requires every window to lie within the full array, and
//! [`BoundaryPolicy::Masked`] admits such windows but requires consumers to mask the invalid coordinates before any
//! memory access. [`BlockMapping::evaluate`] interprets a mapping for concrete inputs and a concrete full array shape
//! and returns a [`BlockWindow`], which holds the original element starts, the slice that selects the valid
//! intersection with the full array, and whether the window requires a mask. It rejects an out-of-bounds window under
//! [`BoundaryPolicy::InBounds`], so that policy is checked whenever a window is evaluated rather than proved when the
//! mapping is constructed. [`BlockMapping::has_total_windows`] proves from the bounds of the mapping output types that
//! these checks pass for every binding, which lets a kernel call that cannot fail them declare no assertion of its
//! own. [`BlockWindow::overlap`] compares the valid intersections of two windows.
//!
//! # Separable Tiling
//!
//! [`BlockMapping::tiling_axes`] recognizes mappings whose windows tile the full array separably, which lets
//! consumers reason about every grid point at once instead of evaluating the mapping at each one. It recognizes a
//! mapping only when every instruction is a dimension multiplication that produces one of the mapping outputs, and
//! when every output has one of the following forms:
//!
//!   - a constant zero start, along which the mapping does not tile,
//!   - an input itself, when the block extent of that axis is 1, or
//!   - the product of an input and a constant equal to the block extent of that axis, in either operand order.
//!
//! Any other form, including an unused instruction, leaves the whole mapping unrecognized. This includes a nonzero
//! constant start (e.g., a specialized prefetched block index), because consumers rely on a start along an axis that
//! does not tile being zero (e.g., to treat such a window as covering its array from the beginning). The proof
//! describes indexing only, so callers still check concrete bounds, repeated or unused grid axes, coverage, and access
//! policy.
//!
//! Recognition therefore does not mean that the windows of distinct grid points are disjoint or that consumers can
//! avoid evaluating the mapping at every grid point. Several outputs may name the same grid axis. The diagonal element
//! starts `(i, i)` with blocks of one element are recognized as `Some(vec![Some(0), Some(0)])`, although the windows
//! follow the diagonal instead of tiling the array. Kernel qualification (i.e.,
//! [`validate_kernel_initialization`](crate::initialization::validate_kernel_initialization)) qualifies a parameter
//! without enumerating the grid only when its recognized axes are distinct. It evaluates every other mapping at every
//! grid point, including a recognized mapping that repeats an axis, and that enumeration is what the
//! `maximum_programs` bound of qualification limits.
//!
//! # Specialization and Identity
//!
//! [`BlockMapping::specialize`] binds the trailing inputs of a mapping (e.g., its scalar prefetched dimensions) to
//! constants and leaves the leading grid coordinates as inputs. It executes arithmetic whose inputs are all constants
//! and rebuilds every other dimension operation from the types of its actual inputs, so the specialized mapping has
//! exact bounds and requires a runtime assertion only where a failure remains possible (e.g., a start `prefetch * 128`
//! becomes a constant start).
//!
//! Two mappings are equal when their programs have the same region content identity and their block shapes and
//! boundary policies agree. Programs do not implement equality, and region content identity lets mappings key caches
//! without comparing program bodies. Every content changing construction mints a new identity and every content
//! preserving copy keeps it, so equality is exact for clones and conservative otherwise. Mappings built from separately
//! constructed but equal programs compare unequal, which affects only caching and never correctness.

use std::hash::{Hash, Hasher};

use ryft_core::{
    Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayReferenceTransform, ArraySliceAxis, Atom, AtomId, Context,
    Dimension, DimensionAddOperation, DimensionDivOperation, DimensionMaxOperation, DimensionMinOperation,
    DimensionMulOperation, DimensionOperation, DimensionPowOperation, DimensionRefinements, DimensionRemOperation,
    DimensionSubOperation, DimensionValue, EagerContext, FlatProgram, MAX_DIMENSION_EXTENT, Operation, Placeholder,
    ProgramBuilder, ProgramError, ReferenceViewOverlap, Shape, TypeError, Typed, TypedProgram,
};
use thiserror::Error;

/// Invalid mapping program, index arithmetic, or full array window.
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

    /// An operation is not a dimension operation without attached regions, which is the only kind a mapping supports.
    #[error("block mapping operation `{operation}` is not a supported straight-line dimension operation")]
    UnsupportedOperation {
        /// Canonical operation name.
        operation: &'static str,
    },

    /// A mapping output or full array shape has the wrong rank.
    #[error("block mapping {boundary} has rank {actual}, but the block shape has rank {expected}")]
    RankMismatch {
        /// Mapping output or full array shape being checked.
        boundary: &'static str,

        /// Static block rank.
        expected: usize,

        /// Observed boundary rank.
        actual: usize,
    },

    /// Specialization supplies more values than the mapping program has inputs.
    #[error("block mapping specialization value count {values} exceeds the mapping input count {inputs}")]
    SpecializationValueCount {
        /// Number of mapping program inputs.
        inputs: usize,

        /// Number of supplied specialization values.
        values: usize,
    },

    /// An extent or window limit exceeds the canonical dimension width.
    #[error("block mapping {boundary} at axis {axis} exceeds the maximum dimension extent {MAX_DIMENSION_EXTENT}")]
    Overflow {
        /// Block shape, full array shape, or window limit being checked.
        boundary: &'static str,

        /// Axis whose extent or arithmetic exceeded the dimension width.
        axis: usize,
    },

    /// An unmasked window reaches outside its full array.
    #[error("block mapping axis {axis} selects `{start}..{limit}` outside array extent {extent}")]
    OutOfBounds {
        /// Full array axis whose window is invalid.
        axis: usize,

        /// Inclusive window start.
        start: usize,

        /// Exclusive window limit.
        limit: usize,

        /// Concrete full array extent.
        extent: usize,
    },
}

/// Required handling of a window that crosses a full array boundary.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BoundaryPolicy {
    /// Every selected coordinate must belong to the full array.
    InBounds,

    /// Consumers must explicitly mask invalid coordinates before memory access. This policy does not authorize
    /// unguarded out-of-bounds execution or choose a padding value for loads.
    Masked,
}

/// Immutable dimension program returning one element start per axis of a statically shaped block.
///
/// Program inputs retain their canonical dimension identities and bounds. Supplying concrete bindings checks those
/// bounds through ordinary program interpretation, and checked arithmetic errors propagate without wrapping indices.
/// The program contains only dimension atoms and dimension family operations, including dimension requirements.
#[derive(Clone, Debug)]
pub struct BlockMapping {
    /// Validated pure program, whose outputs are element starts rather than tile numbers.
    program: FlatProgram<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>,

    /// Logical extents of each block, including any explicitly masked coordinates.
    block_shape: Vec<usize>,

    /// Required treatment of coordinates outside the full array.
    boundary_policy: BoundaryPolicy,
}

impl BlockMapping {
    /// Validates a mapping with one output per block axis whose atoms are all dimensions. Zero extent blocks and rank-0
    /// blocks are allowed. Every instruction is checked, including instructions whose results are unused.
    ///
    /// # Errors
    ///
    /// Returns [`BlockMappingError::RankMismatch`] when the number of outputs differs from the block rank,
    /// [`BlockMappingError::Overflow`] when a block extent exceeds [`MAX_DIMENSION_EXTENT`],
    /// [`BlockMappingError::UnsupportedOperation`] for an instruction that is not a regionless dimension operation, and
    /// [`BlockMappingError::NonDimension`] for an atom that is not a dimension.
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

    /// Returns the required full array boundary handling.
    pub fn boundary_policy(&self) -> BoundaryPolicy {
        self.boundary_policy
    }

    /// Binds trailing static dimension parameters to canonical constants, leaving earlier grid coordinates as inputs.
    /// Every bound value is checked against the bounds of its parameter, and parameters that share one identity must
    /// receive equal values. A bound parameter cannot share its identity with a remaining grid input, because that
    /// would impose a runtime equality which constant substitution alone cannot preserve.
    ///
    /// The instructions are then rebuilt in program order, because dimension operations derive their bounds and their
    /// runtime assertion effect from the operand types that they were constructed for. An instruction whose inputs are
    /// all constants is executed through the ordinary eager interpreter and replaced by its constant result, like known
    /// work in partial evaluation. Every other instruction is reconstructed through the checked `ryft-core`
    /// constructor of its operation from the types of its actual inputs, so refined bounds propagate through chains of
    /// arithmetic and an operation requires a runtime assertion only when its actual inputs leave a failure possible.
    /// No reference or device effect can occur in a validated mapping.
    ///
    /// # Errors
    ///
    /// Returns [`BlockMappingError::SpecializationValueCount`] when `values` has more elements than the mapping has
    /// inputs, and [`BlockMappingError::Program`] when a value lies outside the bounds of its parameter, when bound
    /// values disagree on a shared identity or share one with a remaining input, when checked arithmetic over constant
    /// inputs fails (e.g., overflow, underflow of a plain subtraction, or division by zero), and when a rebuilt
    /// operation rejects the bounds of its actual inputs (e.g., a subtraction whose right-hand side always exceeds its
    /// left-hand side).
    pub fn specialize(&self, values: &[DimensionValue]) -> Result<Self, BlockMappingError> {
        let input_count = self.program.input_types().len();
        let remaining = input_count
            .checked_sub(values.len())
            .ok_or(BlockMappingError::SpecializationValueCount { inputs: input_count, values: values.len() })?;
        let mut builder = ProgramBuilder::new();

        // Each source atom maps to an atom of the new program, to a constant value, or to both. Remaining inputs and
        // bound values are added first, and every other constant is added when an operand or output first uses it.
        let mut atoms = vec![None; self.program.atoms().len()];
        let mut constants = self.program.atoms().iter().map(|atom| atom.as_constant().cloned()).collect::<Vec<_>>();
        let input_types = self.program.input_types().into_owned();
        for (input, r#type) in self.program.input_ids().iter().zip(input_types).take(remaining) {
            atoms[input.index()] = Some(builder.add_input(r#type));
        }
        let mut bindings = DimensionRefinements::default();
        for (offset, (formal, value)) in self.program.input_types().iter().skip(remaining).zip(values).enumerate() {
            let ArrayIrType::Dimension(formal) = formal else { unreachable!() };
            let position = remaining + offset;
            for (input, r#type) in self.program.input_types().iter().take(remaining).enumerate() {
                let ArrayIrType::Dimension(r#type) = r#type else { unreachable!() };
                if formal.variable().is_some() && r#type.variable() == formal.variable() {
                    return Err(ProgramError::from(TypeError::invalid(format!(
                        "block mapping parameter {position} shares a dimension identity with remaining input {input}",
                    )))
                    .into());
                }
            }
            let value = DimensionValue::new(formal.clone(), value.extent())
                .map_err(TypeError::from)
                .map_err(ProgramError::from)?;
            if let Some(variable) = formal.variable() {
                bindings.record_declared_extent(variable, value.extent()).map_err(ProgramError::from)?;
            }
            let input = self.program.input_ids()[position].index();
            atoms[input] = Some(builder.add_constant(ArrayIrValue::Dimension(value.clone())));
            constants[input] = Some(ArrayIrValue::Dimension(value));
        }

        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        for instruction in self.program.instructions() {
            let values = instruction
                .inputs()
                .iter()
                .map(|input| constants[input.index()].clone())
                .collect::<Option<Vec<_>>>();
            if let Some(values) = values {
                let outputs = context.bind(instruction.operation().clone(), Vec::new(), &values)?;
                for (output, value) in instruction.outputs().iter().zip(outputs) {
                    constants[output.index()] = Some(value);
                }
                continue;
            }
            let inputs = instruction
                .inputs()
                .iter()
                .map(|&input| materialize(&mut builder, &mut atoms, &constants, input))
                .collect::<Vec<_>>();
            let input_types = inputs
                .iter()
                .map(|input| {
                    let ArrayIrType::Dimension(r#type) = builder.atoms()[input.index()].r#type().into_owned() else {
                        unreachable!("validated mapping atoms are dimensions");
                    };
                    r#type
                })
                .collect::<Vec<_>>();
            let ArrayIrOperation::Dimension(operation) = instruction.operation() else {
                unreachable!("validated mapping instructions are dimension operations");
            };
            // A constant has no inputs and so always folds above, and saturating subtraction is total and stores no
            // input metadata, so neither needs to be rebuilt.
            let operation = ArrayIrOperation::Dimension(match operation {
                DimensionOperation::Constant(_) | DimensionOperation::SaturatingSub(_) => operation.clone(),
                DimensionOperation::Max(_) => {
                    DimensionMaxOperation::new(&input_types[0], &input_types[1]).map_err(ProgramError::from)?.into()
                }
                DimensionOperation::Min(_) => {
                    DimensionMinOperation::new(&input_types[0], &input_types[1]).map_err(ProgramError::from)?.into()
                }
                DimensionOperation::Add(_) => {
                    DimensionAddOperation::new(&input_types[0], &input_types[1]).map_err(ProgramError::from)?.into()
                }
                DimensionOperation::Sub(_) => {
                    DimensionSubOperation::new(&input_types[0], &input_types[1]).map_err(ProgramError::from)?.into()
                }
                DimensionOperation::Mul(_) => {
                    DimensionMulOperation::new(&input_types[0], &input_types[1]).map_err(ProgramError::from)?.into()
                }
                DimensionOperation::Div(_) => {
                    DimensionDivOperation::new(&input_types[0], &input_types[1]).map_err(ProgramError::from)?.into()
                }
                DimensionOperation::Rem(_) => {
                    DimensionRemOperation::new(&input_types[0], &input_types[1]).map_err(ProgramError::from)?.into()
                }
                DimensionOperation::Pow(_) => {
                    DimensionPowOperation::new(&input_types[0], &input_types[1]).map_err(ProgramError::from)?.into()
                }
            });
            let outputs =
                builder.add_instruction(operation, vec![], inputs, Some(instruction.provenance().clone()))?.to_vec();
            for (output, atom) in instruction.outputs().iter().zip(outputs) {
                atoms[output.index()] = Some(atom);
            }
        }
        let outputs = self
            .program
            .output_ids()
            .iter()
            .map(|&output| materialize(&mut builder, &mut atoms, &constants, output))
            .collect();
        let program =
            builder.build(outputs, vec![Placeholder; remaining], vec![Placeholder; self.block_shape.len()])?;
        Self::new(program, self.block_shape.clone(), self.boundary_policy)
    }

    /// Recognizes separable tiling directly from canonical mapping instructions. Each result is either the input axis
    /// whose coordinate is multiplied by the corresponding block extent, or `None` for a constant zero start. A unit
    /// block also admits the coordinate directly. Unknown expressions return `None` for the complete mapping.
    ///
    /// This describes indexing only, so callers still check concrete bounds, repeated or unused grid axes, coverage,
    /// and access policy. A result may name the same input axis more than once (e.g., `Some(vec![Some(0), Some(0)])`
    /// for the diagonal element starts `(i, i)` with blocks of one element), and kernel qualification enumerates the
    /// grid for such a mapping. Every instruction must contribute a recognized output, so this proof never skips a dead
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

    /// Returns whether the window checks of [`Self::evaluate`] pass for every binding that the mapping input types
    /// admit and for every full array of shape `array_shape`. The proof uses only the bounds of the mapping output
    /// types, which the checked dimension operations of `ryft-core` derive from the bounds of their inputs. Along
    /// each axis, the largest start that its type admits plus the block extent must not exceed
    /// [`MAX_DIMENSION_EXTENT`], and neither may any extent that the array admits along that axis. Under
    /// [`BoundaryPolicy::InBounds`], that limit must also not exceed the smallest extent that the array admits along
    /// that axis. A start type without a finite upper bound, a dynamic array extent without one, and a shape whose rank
    /// differs from the block rank all leave the checks unproven.
    ///
    /// This covers the checks that follow the mapping program and not failures of the program itself (e.g., a division
    /// by zero), which the effects of the program declare. For example, the whole array windows of
    /// [`whole_array_parameter`](crate::authoring::whole_array_parameter) and the tiles `i * 2` of an `[4]` array over
    /// a grid of two programs pass, while the same tiles over a grid of three programs pass only when masked.
    pub fn has_total_windows(&self, array_shape: &Shape) -> bool {
        array_shape.rank() == self.block_shape.len()
            && self.program.output_types().iter().zip(&self.block_shape).zip(array_shape.dimensions()).all(
                |((r#type, &size), extent)| {
                    let ArrayIrType::Dimension(r#type) = r#type else {
                        unreachable!("validated mapping outputs are dimensions");
                    };
                    let (minimum, maximum) = match extent {
                        Dimension::Static(extent) => (*extent, Some(*extent)),
                        Dimension::Dynamic(variable) => {
                            (variable.bounds().lower(), variable.bounds().upper().map(|upper| upper - 1))
                        }
                    };
                    r#type.maximum_extent().and_then(|start| start.checked_add(size)).is_some_and(|limit| {
                        limit <= MAX_DIMENSION_EXTENT
                            && maximum.is_some_and(|maximum| maximum <= MAX_DIMENSION_EXTENT)
                            && (self.boundary_policy == BoundaryPolicy::Masked || limit <= minimum)
                    })
                },
            )
    }

    /// Interprets dimension bindings and intersects the resulting block with the concrete full array shape. The
    /// returned valid transform selects only valid memory, and the original starts remain available for constructing
    /// explicit masks.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: dimension bindings in the mapping program's input order.
    ///   - `array_shape`: concrete full array extents in block axis order.
    pub fn evaluate(&self, inputs: &[DimensionValue], array_shape: &[usize]) -> Result<BlockWindow, BlockMappingError> {
        if array_shape.len() != self.block_shape.len() {
            return Err(BlockMappingError::RankMismatch {
                boundary: "array shape",
                expected: self.block_shape.len(),
                actual: array_shape.len(),
            });
        }
        let outputs = self.program.interpret(inputs.iter().cloned().map(ArrayIrValue::Dimension).collect())?;
        let mut starts = Vec::with_capacity(outputs.len());
        let mut axes = Vec::with_capacity(outputs.len());
        let mut requires_mask = false;
        for (axis, ((output, &size), &extent)) in
            outputs.into_iter().zip(&self.block_shape).zip(array_shape).enumerate()
        {
            if extent > MAX_DIMENSION_EXTENT {
                return Err(BlockMappingError::Overflow { boundary: "array shape", axis });
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

// Two mappings are identical when their programs have the same `RegionContentIdentity` and their block shapes and
// boundary policies agree. Every construction that changes the contents of a program mints a new identity and every
// copy that preserves them keeps it, so the identity is a conservative but exact identity of the mapping program.
// Separately constructed mappings with equal programs therefore compare unequal, which only affects caching, never
// correctness.
impl PartialEq for BlockMapping {
    fn eq(&self, other: &Self) -> bool {
        self.program.entry_region().content_identity() == other.program.entry_region().content_identity()
            && self.block_shape == other.block_shape
            && self.boundary_policy == other.boundary_policy
    }
}

impl Eq for BlockMapping {}

impl Hash for BlockMapping {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.program.entry_region().content_identity().hash(state);
        self.block_shape.hash(state);
        self.boundary_policy.hash(state);
    }
}

/// Evaluated block starts and their valid full array intersection. No storage or reference handle is owned here.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlockWindow {
    /// Original element starts before intersecting with the full array bounds.
    starts: Vec<usize>,

    /// Canonical unit stride slice containing only valid full array coordinates.
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

    /// Returns whether the original logical window extends beyond the full array.
    pub fn requires_mask(&self) -> bool {
        self.requires_mask
    }

    /// Compares valid coordinates of windows on the same full array. Empty intersections are disjoint, identical
    /// non-empty intersections are the same, and partially intersecting windows may overlap. Different ranks are
    /// conservatively reported as possibly overlapping, and root identity remains the responsibility of the caller.
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

/// Returns the atom of `builder` that holds the source mapping atom `atom` during [`BlockMapping::specialize`], adding
/// its constant value to `builder` on first use. Every source atom without an atom in `builder` has a constant value.
fn materialize(
    builder: &mut ProgramBuilder<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
    atoms: &mut [Option<AtomId>],
    constants: &[Option<ArrayIrValue<Array>>],
    atom: AtomId,
) -> AtomId {
    *atoms[atom.index()].get_or_insert_with(|| {
        let value = constants[atom.index()].clone();
        builder.add_constant(value.unwrap_or_else(|| unreachable!("unmapped mapping atoms are constants")))
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::hash::DefaultHasher;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayType, DataType, DimensionBounds, DimensionError, DimensionType, DimensionVariable, EffectClass,
        EffectClasses, ParameterError, ReferenceType, ZERO_OPERATION_NAME, ZeroOperation,
    };

    use super::*;

    /// Returns the hash of `value` under the standard library's default hasher.
    fn hash_of<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    /// Builds an identity mapping over dimensions with independently bounded coordinate inputs.
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
            .into_owned()
            .into_iter()
            .zip(extents)
            .map(|(r#type, &extent)| {
                let ArrayIrType::Dimension(r#type) = r#type else { unreachable!() };
                DimensionValue::new(r#type, extent).unwrap()
            })
            .collect()
    }

    #[test]
    fn test_block_mapping_error() {
        assert_eq!(
            BlockMappingError::Program(ProgramError::from(TypeError::invalid("invalid mapping"))).to_string(),
            "invalid mapping",
        );
        let r#type = ArrayIrType::Array(ArrayType::scalar(DataType::I32));
        assert_eq!(
            BlockMappingError::NonDimension { position: 1, r#type }.to_string(),
            "block mapping atom 1 must be a dimension, but has type `i32[]`",
        );
        assert_eq!(
            BlockMappingError::UnsupportedOperation { operation: ZERO_OPERATION_NAME }.to_string(),
            "block mapping operation `zero` is not a supported straight-line dimension operation",
        );
        assert_eq!(
            BlockMappingError::RankMismatch { boundary: "output", expected: 2, actual: 1 }.to_string(),
            "block mapping output has rank 1, but the block shape has rank 2",
        );
        assert_eq!(
            BlockMappingError::SpecializationValueCount { inputs: 1, values: 2 }.to_string(),
            "block mapping specialization value count 2 exceeds the mapping input count 1",
        );
        assert_eq!(
            BlockMappingError::Overflow { boundary: "window limit", axis: 0 }.to_string(),
            format!("block mapping window limit at axis 0 exceeds the maximum dimension extent {MAX_DIMENSION_EXTENT}"),
        );
        assert_eq!(
            BlockMappingError::OutOfBounds { axis: 0, start: 8, limit: 12, extent: 10 }.to_string(),
            "block mapping axis 0 selects `8..12` outside array extent 10",
        );
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
        assert_eq!(
            BlockMapping::new(identity_program(1), vec![4, 8], BoundaryPolicy::InBounds).unwrap_err(),
            BlockMappingError::RankMismatch { boundary: "output", expected: 2, actual: 1 },
        );
        assert_eq!(
            BlockMapping::new(identity_program(1), vec![MAX_DIMENSION_EXTENT + 1], BoundaryPolicy::InBounds)
                .unwrap_err(),
            BlockMappingError::Overflow { boundary: "block shape", axis: 0 },
        );
    }

    #[test]
    fn test_block_mapping_new_rejects_arrays_and_references() {
        let array_type = ArrayType::scalar(DataType::I32);
        for r#type in [ArrayIrType::Array(array_type.clone()), ReferenceType::new(array_type).into()] {
            let mut builder = ProgramBuilder::new();
            builder.add_input(r#type.clone());
            let output = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()));
            let program = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
            assert_eq!(
                BlockMapping::new(program, vec![1], BoundaryPolicy::InBounds).unwrap_err(),
                BlockMappingError::NonDimension { position: 0, r#type },
            );
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
        assert_eq!(
            BlockMapping::new(program, vec![1], BoundaryPolicy::Masked).unwrap_err(),
            BlockMappingError::UnsupportedOperation { operation: ZERO_OPERATION_NAME },
        );
    }

    #[test]
    fn test_block_mapping_program() {
        let program = identity_program(1);
        let input_types = program.input_types().into_owned();
        let output_types = program.output_types().into_owned();
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
        let variable = DimensionVariable::new("offset", DimensionBounds::non_negative(Some(4)).unwrap());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let first = builder.add_input(DimensionType::from(variable.clone()).into());
        let second = builder.add_input(DimensionType::from(variable.clone()).into());
        let mapping = BlockMapping::new(
            builder.build(vec![first, second], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap(),
            vec![1, 1],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        let expected = TypeError::from(DimensionError::InputDimensionMismatch {
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
        let expected = TypeError::from(DimensionError::BindingOutOfBounds {
            variable: "offset".to_owned(),
            value: 4,
            bounds: variable.bounds(),
        });
        assert_eq!(
            mapping
                .specialize(&[DimensionValue::constant(4).unwrap(), DimensionValue::constant(4).unwrap()])
                .unwrap_err(),
            BlockMappingError::Program(ProgramError::from(expected)),
        );
    }

    #[test]
    fn test_block_mapping_specialize_rejects_identity_shared_with_remaining_input() {
        let variable = DimensionVariable::new("coordinate", DimensionBounds::non_negative(Some(4)).unwrap());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        builder.add_input(DimensionType::from(variable.clone()).into());
        let output = builder.add_input(DimensionType::from(variable).into());
        let mapping = BlockMapping::new(
            builder.build(vec![output], vec![Placeholder; 2], vec![Placeholder]).unwrap(),
            vec![1],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        assert_eq!(
            mapping.specialize(&[DimensionValue::constant(1).unwrap()]).unwrap_err(),
            BlockMappingError::Program(ProgramError::from(TypeError::invalid(
                "block mapping parameter 1 shares a dimension identity with remaining input 0",
            ))),
        );
    }

    #[test]
    fn test_block_mapping_specialize_rejects_excess_values() {
        let mapping = BlockMapping::new(identity_program(1), vec![1], BoundaryPolicy::InBounds).unwrap();
        let error = mapping
            .specialize(&[DimensionValue::constant(1).unwrap(), DimensionValue::constant(2).unwrap()])
            .unwrap_err();
        assert_eq!(error, BlockMappingError::SpecializationValueCount { inputs: 1, values: 2 });
        assert_eq!(error.to_string(), "block mapping specialization value count 2 exceeds the mapping input count 1");
    }

    #[test]
    fn test_block_mapping_specialize_rebuilds_dimension_operations() {
        // Both products may overflow for an unbounded prefetched dimension, so each requires a runtime assertion.
        let coordinate_type = DimensionType::new("coordinate", DimensionBounds::non_negative(Some(4)).unwrap());
        let prefetch_type = DimensionType::new("prefetch", DimensionBounds::non_negative(None).unwrap());
        let block = DimensionValue::constant(4).unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let coordinate = builder.add_input(coordinate_type.clone().into());
        let prefetch = builder.add_input(prefetch_type.clone().into());
        let start = builder
            .add_instruction(
                DimensionMulOperation::new(&coordinate_type, &prefetch_type).unwrap(),
                vec![],
                vec![coordinate, prefetch],
                None,
            )
            .unwrap()[0];
        let block_type = block.r#type().into_owned();
        let block = builder.add_constant(ArrayIrValue::Dimension(block));
        let block_start = builder
            .add_instruction(
                DimensionMulOperation::new(&prefetch_type, &block_type).unwrap(),
                vec![],
                vec![prefetch, block],
                None,
            )
            .unwrap()[0];
        let mapping = BlockMapping::new(
            builder.build(vec![start, block_start], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap(),
            vec![8, 4],
            BoundaryPolicy::Masked,
        )
        .unwrap();
        assert_eq!(mapping.program().effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));

        // The product of constants folds to a constant start, and the remaining product is rebuilt for its constant
        // factor, which proves it total.
        let specialized = mapping.specialize(&[DimensionValue::constant(8).unwrap()]).unwrap();
        assert_eq!(
            specialized.program().to_string(),
            indoc! {"
                lambda %0:dimension<coordinate ∈ [0, 4)> .
                let %1:dimension<8> = const 8
                    %2:dimension<coordinate * 8 ∈ [0, 25)> = dimension_mul %0 %1
                    %3:dimension<32> = const 32
                in (%2, %3)"
            },
        );
        assert_eq!(specialized.program().effects().classes(), EffectClasses::NONE);
        assert_eq!(
            specialized.evaluate(&inputs(&specialized, &[3]), &[32, 64]).unwrap().starts(),
            mapping.evaluate(&inputs(&mapping, &[3, 8]), &[32, 64]).unwrap().starts(),
        );
    }

    #[test]
    fn test_block_mapping_specialize_propagates_bounds_through_chains() {
        let coordinate_type = DimensionType::new("coordinate", DimensionBounds::non_negative(Some(4)).unwrap());
        let prefetch_type = DimensionType::new("prefetch", DimensionBounds::non_negative(None).unwrap());
        let factor = DimensionValue::constant(32).unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let coordinate = builder.add_input(coordinate_type.clone().into());
        let prefetch = builder.add_input(prefetch_type.clone().into());
        let operation = DimensionMulOperation::new(&coordinate_type, factor.r#type().as_ref()).unwrap();
        let factor = builder.add_constant(ArrayIrValue::Dimension(factor));
        let product = builder.add_instruction(operation, vec![], vec![coordinate, factor], None).unwrap()[0];
        let ArrayIrType::Dimension(product_type) = builder.atoms()[product.index()].r#type().into_owned() else {
            unreachable!()
        };
        let start = builder
            .add_instruction(
                DimensionAddOperation::new(&product_type, &prefetch_type).unwrap(),
                vec![],
                vec![product, prefetch],
                None,
            )
            .unwrap()[0];
        let mapping = BlockMapping::new(
            builder.build(vec![start], vec![Placeholder; 2], vec![Placeholder]).unwrap(),
            vec![32],
            BoundaryPolicy::Masked,
        )
        .unwrap();
        assert_eq!(
            mapping.program().to_string(),
            indoc! {"
                lambda %0:dimension<coordinate ∈ [0, 4)>, %1:dimension<prefetch ∈ [0, ∞)> .
                let %2:dimension<32> = const 32
                    %3:dimension<coordinate * 32 ∈ [0, 97)> = dimension_mul %0 %2
                    %4:dimension<coordinate * 32 + prefetch ∈ [0, 9223372036854775808)> = \
                        dimension_add [requires_runtime_assertion=true] %3 %1
                in (%4)"
            },
        );

        // The sum receives the bounds of the coordinate product shifted by the bound constant.
        let specialized = mapping.specialize(&[DimensionValue::constant(5).unwrap()]).unwrap();
        assert_eq!(
            specialized.program().to_string(),
            indoc! {"
                lambda %0:dimension<coordinate ∈ [0, 4)> .
                let %1:dimension<5> = const 5
                    %2:dimension<32> = const 32
                    %3:dimension<coordinate * 32 ∈ [0, 97)> = dimension_mul %0 %2
                    %4:dimension<coordinate * 32 + 5 ∈ [5, 102)> = dimension_add %3 %1
                in (%4)"
            },
        );
        assert_eq!(specialized.program().effects().classes(), EffectClasses::NONE);
        assert_eq!(specialized.evaluate(&inputs(&specialized, &[3]), &[128]).unwrap().starts(), &[101]);
        assert_eq!(mapping.evaluate(&inputs(&mapping, &[3, 5]), &[128]).unwrap().starts(), &[101]);
    }

    #[test]
    fn test_block_mapping_specialize_rejects_failing_arithmetic() {
        let coordinate_type = DimensionType::new("coordinate", DimensionBounds::non_negative(Some(4)).unwrap());
        let prefetch_type = DimensionType::new("prefetch", DimensionBounds::non_negative(None).unwrap());
        let block = DimensionValue::constant(4).unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let coordinate = builder.add_input(coordinate_type.clone().into());
        let prefetch = builder.add_input(prefetch_type.clone().into());
        let operation = DimensionMulOperation::new(&prefetch_type, block.r#type().as_ref()).unwrap();
        let block = builder.add_constant(ArrayIrValue::Dimension(block));
        let product = builder.add_instruction(operation, vec![], vec![prefetch, block], None).unwrap()[0];
        let quotient = builder
            .add_instruction(
                DimensionDivOperation::new(&coordinate_type, &prefetch_type).unwrap(),
                vec![],
                vec![coordinate, prefetch],
                None,
            )
            .unwrap()[0];
        let mapping = BlockMapping::new(
            builder.build(vec![product, quotient], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap(),
            vec![1, 1],
            BoundaryPolicy::Masked,
        )
        .unwrap();

        // Arithmetic over constants reports the error of eager execution.
        assert_eq!(
            mapping.specialize(&[DimensionValue::constant(MAX_DIMENSION_EXTENT).unwrap()]).unwrap_err(),
            BlockMappingError::Program(ProgramError::from(DimensionError::ArithmeticOverflow {
                message: format!(
                    "dimension arithmetic overflow in `dimension_mul`; got left={MAX_DIMENSION_EXTENT}, right=4",
                ),
            })),
        );

        // A rebuilt operation rejects a divisor that is zero at every grid point.
        assert_eq!(
            mapping.specialize(&[DimensionValue::constant(0).unwrap()]).unwrap_err(),
            BlockMappingError::Program(ProgramError::from(DimensionError::RequirementViolation {
                message: "0 > 0 is impossible from declared bounds".to_owned(),
            })),
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
    fn test_block_mapping_tiling_axes_after_specialization() {
        // The first start multiplies the coordinate by a prefetched factor and the second is a prefetched block index
        // times the block extent, so neither has a constant operand before specialization.
        let coordinate_type = DimensionType::new("coordinate", DimensionBounds::non_negative(Some(4)).unwrap());
        let factor_type = DimensionType::new("factor", DimensionBounds::non_negative(None).unwrap());
        let index_type = DimensionType::new("index", DimensionBounds::non_negative(None).unwrap());
        let block = DimensionValue::constant(4).unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let coordinate = builder.add_input(coordinate_type.clone().into());
        let factor = builder.add_input(factor_type.clone().into());
        let index = builder.add_input(index_type.clone().into());
        let start = builder
            .add_instruction(
                DimensionMulOperation::new(&coordinate_type, &factor_type).unwrap(),
                vec![],
                vec![coordinate, factor],
                None,
            )
            .unwrap()[0];
        let operation = DimensionMulOperation::new(&index_type, block.r#type().as_ref()).unwrap();
        let block = builder.add_constant(ArrayIrValue::Dimension(block));
        let block_start = builder.add_instruction(operation, vec![], vec![index, block], None).unwrap()[0];
        let mapping = BlockMapping::new(
            builder.build(vec![start, block_start], vec![Placeholder; 3], vec![Placeholder; 2]).unwrap(),
            vec![8, 4],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        assert_eq!(mapping.tiling_axes(), None);

        // A factor equal to the block extent tiles, and a zero block index folds to a constant zero start.
        let values = [8, 0].map(|extent| DimensionValue::constant(extent).unwrap());
        assert_eq!(mapping.specialize(&values).unwrap().tiling_axes(), Some(vec![Some(0), None]));

        // A nonzero block index folds to a nonzero constant start, which the proof does not describe.
        let values = [8, 2].map(|extent| DimensionValue::constant(extent).unwrap());
        assert_eq!(mapping.specialize(&values).unwrap().tiling_axes(), None);
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
            &ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 0, 1)] },
        );
    }

    #[test]
    fn test_block_mapping_evaluate_rejects_unmasked_bounds_and_overflow() {
        let mapping = BlockMapping::new(identity_program(1), vec![4], BoundaryPolicy::InBounds).unwrap();
        assert_eq!(
            mapping.evaluate(&inputs(&mapping, &[8]), &[10]),
            Err(BlockMappingError::OutOfBounds { axis: 0, start: 8, limit: 12, extent: 10 }),
        );
        assert_eq!(
            mapping.evaluate(&inputs(&mapping, &[MAX_DIMENSION_EXTENT]), &[MAX_DIMENSION_EXTENT]),
            Err(BlockMappingError::Overflow { boundary: "window limit", axis: 0 }),
        );
        assert_eq!(
            mapping.evaluate(&inputs(&mapping, &[0]), &[MAX_DIMENSION_EXTENT + 1]),
            Err(BlockMappingError::Overflow { boundary: "array shape", axis: 0 }),
        );
        assert_eq!(
            mapping.evaluate(&inputs(&mapping, &[0]), &[4, 4]),
            Err(BlockMappingError::RankMismatch { boundary: "array shape", expected: 1, actual: 2 }),
        );
        assert_eq!(
            mapping.evaluate(&[], &[4]),
            Err(BlockMappingError::Program(ProgramError::Parameter(ParameterError::MismatchedParameterStructures {
                left_structure: "[<Parameter>]".to_owned(),
                right_structure: "[]".to_owned(),
            }))),
        );
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
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::ArithmeticOverflow {
                message: format!(
                    "dimension arithmetic overflow in `dimension_mul`; got left={MAX_DIMENSION_EXTENT}, \
                     right={MAX_DIMENSION_EXTENT}",
                ),
            }),
        );
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
    fn test_block_mapping_identity() {
        // Clones share the region content identity of the program, so they are equal and hash identically.
        let program = identity_program(1);
        let mapping = BlockMapping::new(program.clone(), vec![4], BoundaryPolicy::InBounds).unwrap();
        assert_eq!(mapping, mapping.clone());
        assert_eq!(hash_of(&mapping), hash_of(&mapping.clone()));
        let same_mapping = BlockMapping::new(program.clone(), vec![4], BoundaryPolicy::InBounds).unwrap();
        assert_eq!(mapping, same_mapping);
        assert_eq!(hash_of(&mapping), hash_of(&same_mapping));
        // A mapping hashes by its program structure, which its interior caches never change.
        #[allow(clippy::mutable_key_type)]
        let mappings = HashMap::from([(mapping.clone(), "mapping")]);
        assert_eq!(mappings.get(&same_mapping), Some(&"mapping"));

        // The block shape and boundary policy participate in the identity.
        let wider_mapping = BlockMapping::new(program.clone(), vec![8], BoundaryPolicy::InBounds).unwrap();
        assert_ne!(mapping, wider_mapping);
        assert_eq!(mappings.get(&wider_mapping), None);
        assert_ne!(mapping, BlockMapping::new(program, vec![4], BoundaryPolicy::Masked).unwrap());

        // Separately constructed programs carry distinct region content identities, so mappings built from them are
        // conservatively unequal even when the programs render identically.
        let first_program = identity_program(0);
        let second_program = identity_program(0);
        assert_eq!(first_program.to_string(), second_program.to_string());
        assert_ne!(
            BlockMapping::new(first_program, vec![], BoundaryPolicy::InBounds).unwrap(),
            BlockMapping::new(second_program, vec![], BoundaryPolicy::InBounds).unwrap(),
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
