//! Typed TTIR construction from the verified portable kernel arena.
//!
//! Every parameter is a global F32 pointer, and each access of a window loads or stores a pointer tensor under the
//! validity mask of that window, which excludes its physical tile padding, its elements beyond the enclosing window,
//! and its elements beyond the array. Reference reads and [`TileLoadOperation`](ryft_kernels::TileLoadOperation)
//! lower to `tt.load` with that mask and an explicit `other` value, and reference writes lower to `tt.store` with that
//! mask. [`MaskedLoadOperation`](ryft_kernels::MaskedLoadOperation) and
//! [`MaskedStoreOperation`](ryft_kernels::MaskedStoreOperation) intersect that mask with their explicit mask, so they
//! never touch padding, and a masked load takes the corresponding element of its `other` input for every element
//! outside that intersection. A [`WindowMaskOperation`](ryft_kernels::WindowMaskOperation) returns the validity mask
//! itself, which is false on the physical padding. A [`MaskedSwapOperation`](ryft_kernels::MaskedSwapOperation) or a
//! reference swap lowers only when its old value is dead (i.e., when it is a
//! [`KernelSwapLowering::Store`](ryft_kernels::KernelSwapLowering::Store)), as a store. A reference add update loads
//! its window under the validity mask, adds the update, and stores the sum through the same pointers under the same
//! mask. A reference atomic add update is one masked `tt.atomic_rmw fadd` with `acq_rel` semantics at `gpu` scope,
//! and a scalar update of either is broadcast to the window.
//!
//! Sequential grid axes become `scf.for` loops inside each program, nested in declared axis order, so every program
//! visits its steps in the lexicographic order of the grid. The launch grid covers only the parallel axes.
//!
//! # Memory Ordering
//!
//! The pinned Triton keeps the global memory accesses of one thread in program order. Its passes honor the memory
//! effects of `tt.load` (a read of `GlobalMemory`) and `tt.store` (a write of it). CSE merges only loads that no write
//! separates, `triton-licm` hoists a load only out of an `scf.for` that writes nothing, and the instruction reordering
//! of TritonGPU moves no global access. The NVIDIA lowering emits every load and store as inline PTX with side effects
//! (i.e., `PTXBuilder::launch` by default), which LLVM keeps in order.
//!
//! It orders nothing across the threads of one program. Every operation distributes its tensor over the threads in a
//! layout that the compiler chooses for that operation, every thread that holds a copy of an element loads it, only
//! one thread stores it (i.e., `emitRedundantThreadPredicate` in the store lowering), and the membar analysis inserts
//! barriers for shared memory only. A load that follows a store of the same element may therefore run in another
//! thread before that store, and two stores of one element in different threads may land in either order. Only a
//! `ttg.barrier` with global address spaces (i.e., what `tl.debug_barrier` emits) would order them, and `ryft-mlir`
//! does not expose the TritonGPU dialect. The software pipeliner adds a hazard of its own. Under the default two
//! pipeline stages, it issues the loads that feed a dot in an innermost `scf.for` one iteration early, without any
//! alias analysis against the stores of the loop, and it never pipelines `scf.while`.
//!
//! Admission therefore rejects every body whose correctness would depend on such ordering:
//!
//!   - A write-only parameter is never read. Unless its window is a scalar, which only the first thread of a program
//!     ever stores, it is written by at most one access of the entry region, outside control flow.
//!   - A read-write parameter is accessed only by the entry region and only through its whole window, all of its
//!     reads precede its only write, and every value read from it reaches memory only through elementwise operations
//!     into that write. The thread that writes an element then computes it from a load of the same element, whose
//!     value the store depends on (in its own program order or through a layout conversion), so that load cannot
//!     observe the store, and every other load of that element feeds no memory. A swap whose old value is used fails
//!     this test because another thread may load that value after the swap stores over it.
//!   - A parameter that is accumulated atomically has no other access. Atomic updates are atomic among themselves in
//!     every order, and the interpreter admits every order of floating-point accumulation.
//!   - Every other mutable window changes at every step of each sequential axis with more than one step (i.e., its
//!     mapping tiles that axis), so no step of a program touches memory that another step writes and no load in a
//!     sequential loop reads memory that the loop writes.
//!
//! Parallel programs need no ordering, because the verifier proves that their mutable windows are disjoint or only
//! accumulated atomically.

use std::collections::{HashMap, HashSet};

use ryft_core::{
    Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReferenceTransform,
    ArrayReferenceTransformIndex, ArrayType, Atom, AtomId, ComparisonDirection, DataType, Dimension,
    DimensionOperation, DimensionValue, DotDimensionNumbers, InstructionId, Layout, Memory, Operation as CoreOperation,
    ReductionKind, ReferenceAccessMode, ReferenceAccessOperation, ReferenceRoot, RegionRef, Typed,
    validated_reference_access_descriptors,
};
use ryft_kernels::{
    GridExecution, KernelOperation, KernelParameterAccess, KernelSchedule, KernelSwapLowering, VerifiedKernel,
};
use ryft_mlir::dialects::{arith, scf, triton::tt};
use ryft_mlir::{
    Block, Context, DetachedBlock, DetachedOp, DialectHandle, Module, Operation, Size, Type, TypeRef,
    UnknownLocationRef, Value, ValueRef,
};

use crate::kernels::{Error, Options};

/// Verified native module and the exact pointer-only logical argument mapping.
pub(super) struct Lowered<'c, 't> {
    /// Typed module owned for the duration of native compilation.
    pub module: Module<'c, 't>,

    /// Physical parallel launch dimensions.
    pub grid: [u32; 3],

    /// Full logical buffer types in entry argument order.
    pub parameter_types: Vec<ArrayType>,
}

/// Native values remain owned by the caller-provided lowering context.
type Native<'c, 't> = ValueRef<'c, 'c, 't>;

/// Global window preserving physical coordinates when logical axes are indexed away.
#[derive(Clone, PartialEq)]
struct Reference<'c, 't> {
    /// Global scalar pointer.
    pointer: Native<'c, 't>,

    /// Full original allocation shape.
    allocation: Vec<usize>,

    /// Surviving original axes.
    axes: Vec<usize>,

    /// Coordinates fixed by reference indexing.
    fixed: Vec<Option<Native<'c, 't>>>,

    /// Logical window origins.
    starts: Vec<Native<'c, 't>>,

    /// Enclosing window exclusive limits.
    limits: Vec<Native<'c, 't>>,

    /// Current logical tile shape.
    shape: Vec<usize>,

    /// Validity of indexed-away coordinates.
    valid: Native<'c, 't>,
}

/// Canonical values keep references distinct from immutable scalar and tile values.
#[derive(Clone)]
enum LoweringValue<'c, 't> {
    Native(Native<'c, 't>),
    Reference(Reference<'c, 't>),
}

impl<'c, 't> LoweringValue<'c, 't> {
    /// Extracts an immutable value after canonical verification.
    fn native(&self) -> Result<Native<'c, 't>, Error> {
        match self {
            Self::Native(value) => Ok(*value),
            Self::Reference(_) => Err(unsupported("reference", "reference cannot escape as an ordinary value")),
        }
    }

    /// Extracts a global window after canonical verification.
    fn reference(&self) -> Result<&Reference<'c, 't>, Error> {
        match self {
            Self::Reference(value) => Ok(value),
            Self::Native(_) => Err(unsupported("reference", "expected a global reference")),
        }
    }
}

/// Bounded construction state; it owns no runtime buffers or compiler resources.
struct Lowering<'c, 't> {
    /// Context owning the complete generated module.
    context: &'c Context<'t>,

    /// Generated operation location.
    location: UnknownLocationRef<'c, 't>,

    /// Remaining canonical instruction and literal expansion allowance.
    remaining: usize,

    /// Maximum physical elements in one tile.
    maximum_tile_elements: usize,
}

/// Admits and constructs a complete native module without invoking a target compiler.
pub(super) fn lower<'c, 't>(
    context: &'c Context<'t>,
    kernel: &VerifiedKernel<'_>,
    options: &Options,
    _schedule: &KernelSchedule,
) -> Result<Lowered<'c, 't>, Error> {
    let call = kernel.definition().operation();
    validate_memory_ordering(kernel)?;
    if !call.prefetch_types().is_empty() {
        return Err(unsupported("kernel_call", "scalar prefetch must be specialized before Triton compilation"));
    }
    context.load_dialect(DialectHandle::triton_tt()?)?;
    context.load_dialect(DialectHandle::arith()?)?;
    context.load_dialect(DialectHandle::scf()?)?;
    let location = context.unknown_location();
    let module = context.module(location)?;
    let mut lowering = Lowering {
        context,
        location,
        remaining: options.maximum_instructions(),
        maximum_tile_elements: options.maximum_tile_elements(),
    };
    lowering.charge(call.parameters().len() + call.grid().dimensions().len())?;
    let mut parameter_types = Vec::new();
    for parameter in call.parameters() {
        let r#type = parameter.r#type().into_owned();
        if r#type.data_type() != DataType::F32
            || r#type.memory() != Memory::Device
            || r#type.sharding().is_some()
            || r#type.layout().is_some_and(|layout| {
                !matches!(layout, Layout::Tiled(layout)
                if layout.tiles().is_empty() && layout.minor_to_major().iter().copied().eq((0..r#type.rank()).rev()))
            })
        {
            return Err(unsupported("parameter", "requires unsharded dense row-major F32 device arrays"));
        }
        shape(&r#type)?;
        parameter_types.push(r#type);
    }
    let pointer = context.triton_tt_pointer_type(context.float32_type(), 1)?;
    let arguments = vec![(pointer, location); parameter_types.len()];
    let mut block = context.block(&arguments);
    let mut count = 1usize;
    for dimension in call.grid().dimensions() {
        let Dimension::Static(extent) = dimension.extent() else {
            return Err(unsupported("kernel_call", "grid extents must be static"));
        };
        if dimension.execution() == GridExecution::Parallel {
            count = count
                .checked_mul(*extent)
                .filter(|count| *count <= i32::MAX as usize)
                .ok_or_else(|| unsupported("kernel_call", "parallel grid exceeds signed 32-bit launch bounds"))?;
        } else if *extent > i64::MAX as usize {
            return Err(unsupported("kernel_call", "sequential grid extents exceed signed 64-bit loop bounds"));
        }
    }
    let mut coordinate = append(
        &mut block,
        tt::get_program_id(tt::ProgramIdDim::X, context.signless_integer_type(32).as_ref(), location)?,
    )?;
    coordinate = append(&mut block, arith::extsi(coordinate, context.signless_integer_type(64), location)?)?;
    // Sequential coordinates are placeholders until `Lowering::grid` binds them to loop induction variables.
    let mut coordinates = vec![coordinate; call.grid().dimensions().len()];
    for (axis, dimension) in call.grid().dimensions().iter().enumerate().rev() {
        let Dimension::Static(extent) = dimension.extent() else { unreachable!() };
        if dimension.execution() == GridExecution::Parallel {
            let extent = lowering.integer(&mut block, (*extent).max(1) as i64)?;
            coordinates[axis] = append(&mut block, arith::remsi(coordinate, extent, location)?)?;
            coordinate = append(&mut block, arith::divsi(coordinate, extent, location)?)?;
        }
    }
    let active = lowering.boolean(&mut block, count != 0)?;
    let pointers = (0..parameter_types.len())
        .map(|index| block.argument(index).map(|argument| argument.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    lowering.grid(&mut block, kernel, &pointers, &mut coordinates, active, 0)?;
    block.append_operation(tt::r#return(&[], location)?)?;
    let signature = context.function_type(&vec![pointer; parameter_types.len()], &[] as &[TypeRef<'_, '_>]);
    module.body()?.append_operation(tt::func(
        "ryft_kernel",
        signature.as_ref(),
        Some("public"),
        None,
        None,
        block.try_into()?,
        location,
    )?)?;
    if !module.verify()? {
        return Err(unsupported("kernel_call", "constructed Triton module failed native verification"));
    }
    Ok(Lowered { module, grid: [count.max(1) as u32, 1, 1], parameter_types })
}

impl<'c, 't> Lowering<'c, 't> {
    /// Charges work before growing native construction state.
    fn charge(&mut self, count: usize) -> Result<(), Error> {
        self.remaining = self
            .remaining
            .checked_sub(count)
            .ok_or_else(|| unsupported("kernel_call", "construction exceeds the instruction budget"))?;
        Ok(())
    }

    /// Computes a bounded physical tile shape without overflowing padding arithmetic.
    fn physical(&self, shape: &[usize]) -> Result<Vec<usize>, Error> {
        if shape.len() > 4 {
            return Err(unsupported("array", "tile rank exceeds four"));
        }
        let mut count = 1usize;
        let mut physical = Vec::new();
        for extent in shape {
            let extent = extent
                .max(&1)
                .checked_next_power_of_two()
                .ok_or_else(|| unsupported("array", "physical tile extent overflows"))?;
            count = count
                .checked_mul(extent)
                .filter(|count| *count <= self.maximum_tile_elements)
                .ok_or_else(|| unsupported("array", "physical tile exceeds the element budget"))?;
            physical.push(extent);
        }
        Ok(physical)
    }

    /// Constructs the native scalar or tensor type for one canonical shape.
    fn r#type(&self, data_type: DataType, shape: &[usize]) -> Result<TypeRef<'c, 't>, Error> {
        let element = match data_type {
            DataType::F32 => self.context.float32_type().as_ref(),
            DataType::I64 => self.context.signless_integer_type(64).as_ref(),
            DataType::Boolean => self.context.signless_integer_type(1).as_ref(),
            _ => return Err(unsupported("array", "only F32, I64 and Boolean tile values are supported")),
        };
        if shape.is_empty() {
            return Ok(element);
        }
        let dimensions = self.physical(shape)?.into_iter().map(Size::Static).collect::<Vec<_>>();
        Ok(self.context.tensor_type(element, &dimensions, None, self.location)?.as_ref())
    }

    /// Emits an exact signed indexing constant.
    fn integer(&self, block: &mut DetachedBlock<'c, 't>, value: i64) -> Result<Native<'c, 't>, Error> {
        append(
            block,
            arith::constant(
                self.context.integer_attribute(self.context.signless_integer_type(64), value),
                self.location,
            )?,
        )
    }

    /// Emits a scalar predicate.
    fn boolean(&self, block: &mut DetachedBlock<'c, 't>, value: bool) -> Result<Native<'c, 't>, Error> {
        append(
            block,
            arith::constant(
                self.context.integer_attribute(self.context.signless_integer_type(1), i64::from(value)),
                self.location,
            )?,
        )
    }

    /// Broadcasts a scalar to a physical tile, preserving scalar representation at rank zero.
    fn splat(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        value: Native<'c, 't>,
        data_type: DataType,
        shape: &[usize],
    ) -> Result<Native<'c, 't>, Error> {
        if shape.is_empty() {
            return Ok(value);
        }
        append(block, tt::splat(value, self.r#type(data_type, shape)?, self.location)?)
    }

    /// Emits exact uniform literal storage, rejecting nonuniform payloads before expansion.
    fn constant(&mut self, block: &mut DetachedBlock<'c, 't>, value: &Array) -> Result<Native<'c, 't>, Error> {
        let r#type = value.r#type();
        let shape = shape(&r#type)?;
        self.physical(&shape)?;
        let bytes = value.logical_bytes();
        let size = match r#type.data_type() {
            DataType::F32 => 4,
            DataType::I64 => 8,
            DataType::Boolean => 1,
            _ => return Err(unsupported("constant", "unsupported constant element type")),
        };
        self.charge(bytes.len() / size)?;
        if bytes.chunks_exact(size).any(|chunk| chunk != &bytes[..size]) {
            return Err(unsupported("constant", "nonuniform array constants are not yet supported"));
        }
        let mut bits = [0u8; 8];
        if !bytes.is_empty() {
            bits[..size].copy_from_slice(&bytes[..size]);
        }
        let value = match r#type.data_type() {
            DataType::F32 => {
                let integer = append(
                    block,
                    arith::constant(
                        self.context.integer_attribute(
                            self.context.signless_integer_type(32),
                            i64::from(u32::from_le_bytes(bits[..4].try_into().unwrap())),
                        ),
                        self.location,
                    )?,
                )?;
                append(block, arith::bitcast(integer, self.context.float32_type(), self.location)?)?
            }
            DataType::I64 => self.integer(block, i64::from_le_bytes(bits))?,
            _ => self.boolean(block, bits[0] != 0)?,
        };
        self.splat(block, value, r#type.data_type(), &shape)
    }

    /// Lowers canonical dimension-only mapping programs without evaluating symbolic values on the host.
    fn mapping(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        mapping: &ryft_kernels::BlockMapping,
        inputs: &[Native<'c, 't>],
    ) -> Result<Vec<Native<'c, 't>>, Error> {
        let program = mapping.program();
        self.charge(program.atoms().len())?;
        for atom in program.atoms() {
            if let ArrayIrType::Dimension(r#type) = atom.r#type().as_ref() {
                check_dimension(r#type)?;
            }
        }
        let mut values = vec![None; program.atoms().len()];
        for (id, value) in program.input_ids().iter().zip(inputs) {
            values[id.index()] = Some(*value);
        }
        for (index, atom) in program.atoms().iter().enumerate() {
            if let Atom::Constant(ArrayIrValue::Dimension(value)) = atom {
                values[index] = Some(self.integer(block, value.extent() as i64)?);
            }
        }
        for instruction in program.instructions() {
            self.charge(1)?;
            let ArrayIrOperation::Dimension(operation) = instruction.operation() else { unreachable!() };
            let inputs = instruction.inputs().iter().map(|id| values[id.index()].unwrap()).collect::<Vec<_>>();
            values[instruction.outputs()[0].index()] = Some(self.dimension(block, operation, &inputs)?);
        }
        Ok(program.output_ids().iter().map(|id| values[id.index()].unwrap()).collect())
    }

    /// Emits total checked dimension arithmetic; effectful operations retain explicit unsupported diagnostics.
    fn dimension(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        operation: &DimensionOperation<DimensionValue>,
        inputs: &[Native<'c, 't>],
    ) -> Result<Native<'c, 't>, Error> {
        if !operation.effects().is_pure() {
            return Err(unsupported(operation.name(), "dimension arithmetic must be proven total"));
        }
        let location = self.location;
        Ok(match operation {
            DimensionOperation::Constant(operation) => self.integer(block, operation.value().extent() as i64)?,
            DimensionOperation::Add(_) => append(block, arith::addi(inputs[0], inputs[1], location)?)?,
            DimensionOperation::Sub(_) => append(block, arith::subi(inputs[0], inputs[1], location)?)?,
            DimensionOperation::Mul(_) => append(block, arith::muli(inputs[0], inputs[1], location)?)?,
            DimensionOperation::Div(_) => append(block, arith::divsi(inputs[0], inputs[1], location)?)?,
            DimensionOperation::Rem(_) => append(block, arith::remsi(inputs[0], inputs[1], location)?)?,
            DimensionOperation::Min(_) => append(block, arith::minsi(inputs[0], inputs[1], location)?)?,
            DimensionOperation::Max(_) => append(block, arith::maxsi(inputs[0], inputs[1], location)?)?,
            DimensionOperation::SaturatingSub(_) => {
                let difference = append(block, arith::subi(inputs[0], inputs[1], location)?)?;
                let zero = self.integer(block, 0)?;
                append(block, arith::maxsi(difference, zero, location)?)?
            }
            _ => {
                return Err(unsupported(
                    operation.name(),
                    "dimension operation is outside the supported arithmetic subset",
                ));
            }
        })
    }

    /// Constructs per-axis tensor coordinates and a logical-padding mask.
    fn indices(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        shape: &[usize],
    ) -> Result<(Vec<Native<'c, 't>>, Native<'c, 't>), Error> {
        let physical = self.physical(shape)?;
        let truth = self.boolean(block, true)?;
        let mut mask = self.splat(block, truth, DataType::Boolean, shape)?;
        let mut indices = Vec::new();
        for (axis, extent) in physical.iter().enumerate() {
            let range_type = self.context.tensor_type(
                self.context.signless_integer_type(32),
                &[Size::Static(*extent)],
                None,
                self.location,
            )?;
            let range = append(block, tt::make_range(0, *extent as i64, range_type.as_ref(), self.location)?)?;
            let mut expanded = vec![1; shape.len()];
            expanded[axis] = *extent;
            let expanded_type = self.context.tensor_type(
                self.context.signless_integer_type(32),
                &expanded.iter().copied().map(Size::Static).collect::<Vec<_>>(),
                None,
                self.location,
            )?;
            let range = append(block, tt::reshape(range, false, false, expanded_type.as_ref(), self.location)?)?;
            let expanded_integer = self.r#type(DataType::I64, &expanded)?;
            let range = append(block, arith::extsi(range, expanded_integer, self.location)?)?;
            let range = append(block, tt::broadcast(range, self.r#type(DataType::I64, shape)?, self.location)?)?;
            let limit = self.integer(block, shape[axis] as i64)?;
            let limit = self.splat(block, limit, DataType::I64, shape)?;
            let valid = append(
                block,
                arith::cmpi(range, limit, arith::IntegerComparisonPredicate::SignedLessThan, self.location)?,
            )?;
            mask = append(block, arith::andi(mask, valid, self.location)?)?;
            indices.push(range);
        }
        Ok((indices, mask))
    }

    /// Emits the coordinates of a tile of type `output` along `dimension`. Coordinates in the physical padding exceed
    /// the logical extent, which is harmless because padding never reaches global memory.
    fn iota(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        operation: &ArrayOperation<Array>,
        dimension: usize,
        output: &ArrayType,
    ) -> Result<Native<'c, 't>, Error> {
        let output_shape = shape(output)?;
        let coordinates = self.indices(block, &output_shape)?.0[dimension];
        match output.data_type() {
            DataType::I64 => Ok(coordinates),
            DataType::F32 => {
                append(block, arith::sitofp(coordinates, self.r#type(DataType::F32, &output_shape)?, self.location)?)
            }
            _ => Err(unsupported(operation.name(), "iota requires an F32 or I64 element type")),
        }
    }

    /// Broadcasts `input` to a tile of type `output`, where `output_axes` contains the output axis of each input axis.
    fn broadcast(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        operation: &ArrayOperation<Array>,
        input: Native<'c, 't>,
        input_type: &ArrayType,
        output_axes: &[usize],
        output: &ArrayType,
    ) -> Result<Native<'c, 't>, Error> {
        let output_shape = shape(output)?;
        if input_type.rank() == 0 {
            return self.splat(block, input, output.data_type(), &output_shape);
        }
        if output_axes.windows(2).any(|axes| axes[0] >= axes[1]) {
            return Err(unsupported(operation.name(), "broadcast axes must preserve source order"));
        }
        let source = shape(input_type)?;
        let mut expanded = vec![1; output_shape.len()];
        for (axis, output_axis) in output_axes.iter().enumerate() {
            expanded[*output_axis] = source[axis];
        }
        let reshaped = append(
            block,
            tt::reshape(input, false, false, self.r#type(output.data_type(), &expanded)?, self.location)?,
        )?;
        append(block, tt::broadcast(reshaped, self.r#type(output.data_type(), &output_shape)?, self.location)?)
    }

    /// Computes the validity mask of a global window, intersecting root, enclosing window, and physical padding bounds,
    /// together with the global coordinate of each element along every allocation axis, paired with whether that
    /// coordinate lies inside its allocation axis. The mask is exactly the canonical window validity on the logical
    /// elements and false on the physical padding.
    fn validity(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        reference: &Reference<'c, 't>,
    ) -> Result<(Vec<(Native<'c, 't>, Native<'c, 't>)>, Native<'c, 't>), Error> {
        let (indices, mut mask) = self.indices(block, &reference.shape)?;
        let valid = self.splat(block, reference.valid, DataType::Boolean, &reference.shape)?;
        mask = append(block, arith::andi(mask, valid, self.location)?)?;
        let zero = self.integer(block, 0)?;
        let zero = self.splat(block, zero, DataType::I64, &reference.shape)?;
        let mut coordinates = Vec::new();
        for axis in 0..reference.allocation.len() {
            let coordinate = if let Some(coordinate) = reference.fixed[axis] {
                self.splat(block, coordinate, DataType::I64, &reference.shape)?
            } else {
                let position = reference.axes.iter().position(|value| *value == axis).unwrap();
                let start = self.splat(block, reference.starts[position], DataType::I64, &reference.shape)?;
                let coordinate = append(block, arith::addi(start, indices[position], self.location)?)?;
                let limit = self.splat(block, reference.limits[position], DataType::I64, &reference.shape)?;
                let valid = append(
                    block,
                    arith::cmpi(coordinate, limit, arith::IntegerComparisonPredicate::SignedLessThan, self.location)?,
                )?;
                mask = append(block, arith::andi(mask, valid, self.location)?)?;
                coordinate
            };
            let limit = self.integer(block, reference.allocation[axis] as i64)?;
            let limit = self.splat(block, limit, DataType::I64, &reference.shape)?;
            let nonnegative = append(
                block,
                arith::cmpi(
                    coordinate,
                    zero,
                    arith::IntegerComparisonPredicate::SignedGreaterThanOrEqual,
                    self.location,
                )?,
            )?;
            let below = append(
                block,
                arith::cmpi(coordinate, limit, arith::IntegerComparisonPredicate::SignedLessThan, self.location)?,
            )?;
            let inside = append(block, arith::andi(nonnegative, below, self.location)?)?;
            mask = append(block, arith::andi(mask, inside, self.location)?)?;
            coordinates.push((coordinate, inside));
        }
        Ok((coordinates, mask))
    }

    /// Creates valid pointer tensors and their validity mask, which [`Self::validity`] computes.
    fn pointers(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        reference: &Reference<'c, 't>,
    ) -> Result<(Native<'c, 't>, Native<'c, 't>), Error> {
        let (coordinates, mask) = self.validity(block, reference)?;
        let zero = self.integer(block, 0)?;
        let zero = self.splat(block, zero, DataType::I64, &reference.shape)?;
        let mut offset = zero;
        let mut stride = 1usize;
        for (axis, (coordinate, inside)) in coordinates.into_iter().enumerate().rev() {
            let coordinate = append(block, arith::select(inside, coordinate, zero, self.location)?)?;
            let stride_value = self.integer(block, stride as i64)?;
            let stride_value = self.splat(block, stride_value, DataType::I64, &reference.shape)?;
            let contribution = append(block, arith::muli(coordinate, stride_value, self.location)?)?;
            offset = append(block, arith::addi(offset, contribution, self.location)?)?;
            stride = stride
                .checked_mul(reference.allocation[axis])
                .filter(|value| *value <= i64::MAX as usize / 4)
                .ok_or_else(|| unsupported("reference", "allocation strides exceed signed 64-bit byte offsets"))?;
        }
        // Invalid lanes use the root pointer, so even masked arithmetic cannot create an overflowing address.
        offset = append(block, arith::select(mask, offset, zero, self.location)?)?;
        let pointer = if reference.shape.is_empty() {
            reference.pointer
        } else {
            let r#type = self.context.tensor_type(
                reference.pointer.r#type()?,
                &self.physical(&reference.shape)?.into_iter().map(Size::Static).collect::<Vec<_>>(),
                None,
                self.location,
            )?;
            append(block, tt::splat(reference.pointer, r#type.as_ref(), self.location)?)?
        };
        let pointers = append(block, tt::addptr(pointer, offset, pointer.r#type()?, self.location)?)?;
        Ok((pointers, mask))
    }

    /// Loads the valid elements of a clipped F32 window that the optional Boolean tile `mask` selects, and takes the
    /// corresponding element of the F32 tile `other` for every other element.
    fn read(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        reference: &Reference<'c, 't>,
        mask: Option<Native<'c, 't>>,
        other: Native<'c, 't>,
    ) -> Result<Native<'c, 't>, Error> {
        let (pointer, valid) = self.pointers(block, reference)?;
        let mask = match mask {
            Some(mask) => append(block, arith::andi(valid, mask, self.location)?)?,
            None => valid,
        };
        append(
            block,
            tt::load(
                pointer,
                Some(mask),
                Some(other),
                None,
                false,
                self.r#type(DataType::F32, &reference.shape)?,
                self.location,
            )?,
        )
    }

    /// Writes only the valid elements of a global window that the optional Boolean tile `mask` selects.
    fn write(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        reference: &Reference<'c, 't>,
        value: Native<'c, 't>,
        mask: Option<Native<'c, 't>>,
    ) -> Result<(), Error> {
        let (pointer, valid) = self.pointers(block, reference)?;
        let mask = match mask {
            Some(mask) => append(block, arith::andi(valid, mask, self.location)?)?,
            None => valid,
        };
        block.append_operation(tt::store(pointer, value, Some(mask), None, false, self.location)?)?;
        Ok(())
    }

    /// Prepares the F32 `value` of type `r#type` for an additive update of `reference`, broadcasting a scalar to the
    /// window shape.
    fn update(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        operation: &'static str,
        reference: &Reference<'c, 't>,
        value: Native<'c, 't>,
        r#type: &ArrayIrType,
    ) -> Result<Native<'c, 't>, Error> {
        let ArrayIrType::Array(r#type) = r#type else {
            return Err(unsupported(operation, "expected an ordinary array update"));
        };
        if r#type.data_type() != DataType::F32 {
            return Err(unsupported(operation, "updates require the F32 element type of their reference"));
        }
        if r#type.rank() == 0 {
            return self.splat(block, value, DataType::F32, &reference.shape);
        }
        if shape(r#type)? != reference.shape {
            return Err(unsupported(operation, "updates require the shape of their window or a scalar shape"));
        }
        Ok(value)
    }

    /// Applies one supported reference transform while retaining physical coordinates and enclosing validity.
    fn apply_transform(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        reference: &mut Reference<'c, 't>,
        transform: &ArrayReferenceTransform,
        operation: &'static str,
    ) -> Result<(), Error> {
        self.charge(1)?;
        let location = self.location;
        match transform {
            ArrayReferenceTransform::Index { axis, index: ArrayReferenceTransformIndex::Static(index) } => {
                let axis = *axis;
                let index = self.integer(block, *index as i64)?;
                let coordinate = append(block, arith::addi(reference.starts[axis], index, location)?)?;
                let valid = append(
                    block,
                    arith::cmpi(
                        coordinate,
                        reference.limits[axis],
                        arith::IntegerComparisonPredicate::SignedLessThan,
                        location,
                    )?,
                )?;
                reference.valid = append(block, arith::andi(reference.valid, valid, location)?)?;
                reference.fixed[reference.axes.remove(axis)] = Some(coordinate);
                reference.starts.remove(axis);
                reference.limits.remove(axis);
                reference.shape.remove(axis);
            }
            ArrayReferenceTransform::Slice { axes } => {
                if axes.iter().any(|axis| axis.stride() != 1) {
                    return Err(unsupported(operation, "strided reference transforms are unsupported"));
                }
                for (axis, selection) in axes.iter().enumerate() {
                    let offset = self.integer(block, selection.start() as i64)?;
                    let size = self.integer(block, selection.size() as i64)?;
                    let start = append(block, arith::addi(reference.starts[axis], offset, location)?)?;
                    let limit = append(block, arith::addi(start, size, location)?)?;
                    reference.starts[axis] = start;
                    reference.limits[axis] = append(block, arith::minsi(reference.limits[axis], limit, location)?)?;
                    reference.shape[axis] = selection.size();
                }
            }
            ArrayReferenceTransform::Index { .. } => {
                return Err(unsupported(operation, "dynamic reference transforms are unsupported"));
            }
        }
        Ok(())
    }

    /// Nests the sequential grid axes from `axis` onward as `scf.for` loops in declared axis order, and lowers the body
    /// once at the innermost step over windows evaluated at `coordinates`. Parallel coordinates are already bound.
    fn grid(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        kernel: &VerifiedKernel<'_>,
        pointers: &[Native<'c, 't>],
        coordinates: &mut [Native<'c, 't>],
        active: Native<'c, 't>,
        axis: usize,
    ) -> Result<(), Error> {
        let call = kernel.definition().operation();
        if let Some(dimension) = call.grid().dimensions().get(axis) {
            if dimension.execution() == GridExecution::Parallel {
                return self.grid(block, kernel, pointers, coordinates, active, axis + 1);
            }
            let Dimension::Static(extent) = dimension.extent() else { unreachable!() };
            let lower = self.integer(block, 0)?;
            let upper = self.integer(block, *extent as i64)?;
            let step = self.integer(block, 1)?;
            let mut body = self.context.block(&[(self.context.signless_integer_type(64).as_ref(), self.location)]);
            coordinates[axis] = body.argument(0)?.as_ref();
            self.grid(&mut body, kernel, pointers, coordinates, active, axis + 1)?;
            body.append_operation(scf::r#yield(&[], self.location)?)?;
            block.append_operation(scf::r#for(lower, upper, step, &[], false, body.try_into()?, self.location)?)?;
            return Ok(());
        }
        let mut inputs = Vec::new();
        for (index, parameter) in call.parameters().iter().enumerate() {
            let starts = self.mapping(block, parameter.mapping(), coordinates)?;
            let tile = parameter.mapping().block_shape().to_vec();
            let mut limits = Vec::new();
            for (start, extent) in starts.iter().zip(&tile) {
                let extent = self.integer(block, *extent as i64)?;
                limits.push(append(block, arith::addi(*start, extent, self.location)?)?);
            }
            let allocation = shape(&parameter.r#type())?;
            inputs.push(LoweringValue::Reference(Reference {
                pointer: pointers[index],
                axes: (0..tile.len()).collect(),
                fixed: vec![None; allocation.len()],
                allocation,
                starts,
                limits,
                shape: tile,
                valid: active,
            }));
        }
        inputs.extend(coordinates.iter().copied().map(LoweringValue::Native));
        self.region(block, kernel.definition().body().entry_region_ref(), &inputs)?;
        Ok(())
    }

    /// Replays canonical regions into typed SSA without changing portable operation semantics.
    fn region(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        region: RegionRef<'_, ArrayIrValue<Array>, KernelOperation>,
        inputs: &[LoweringValue<'c, 't>],
    ) -> Result<Vec<LoweringValue<'c, 't>>, Error> {
        self.charge(region.atoms().len())?;
        let mut values = vec![None; region.atoms().len()];
        for (id, value) in region.input_ids().iter().zip(inputs) {
            values[id.index()] = Some(value.clone());
        }
        for (index, atom) in region.atoms().iter().enumerate() {
            if let ArrayIrType::Dimension(r#type) = atom.r#type().as_ref() {
                check_dimension(r#type)?;
            }
            if let Atom::Constant(value) = atom {
                values[index] = Some(LoweringValue::Native(match value {
                    ArrayIrValue::Array(value) => self.constant(block, value)?,
                    ArrayIrValue::Dimension(value) => self.integer(block, value.extent() as i64)?,
                    ArrayIrValue::Reference(_) => {
                        return Err(unsupported("constant", "captured references are unsupported"));
                    }
                }));
            }
        }
        // A folded path may be repeated by many accesses. Cache only its immutable address calculations, never the
        // loaded value or memory effect. Atom and binding identities are local to this region, and the complete
        // ordered path is part of the key; separate region invocations construct independent caches and native SSA.
        let mut views = HashMap::<_, Reference<'c, 't>>::new();
        for instruction in region.instructions() {
            self.charge(1)?;
            let mut inputs =
                instruction.inputs().iter().map(|id| values[id.index()].clone().unwrap()).collect::<Vec<_>>();
            let operation = instruction.operation();
            let descriptors = validated_reference_access_descriptors(operation, instruction.inputs().len())?;
            for (input_index, descriptor) in descriptors.iter().enumerate() {
                let Some(descriptor) = descriptor else { continue };
                if descriptor.transforms().is_empty() {
                    continue;
                }
                let key = (
                    instruction.inputs()[input_index],
                    descriptor.transforms(),
                    &instruction.inputs()[descriptor.bindings()],
                );
                let reference = if let Some(reference) = views.get(&key) {
                    reference.clone()
                } else {
                    let mut reference = inputs[input_index].reference()?.clone();
                    for transform in descriptor.transforms() {
                        self.apply_transform(block, &mut reference, transform, operation.name())?;
                    }
                    views.insert(key, reference.clone());
                    reference
                };
                inputs[input_index] = LoweringValue::Reference(reference);
            }
            if descriptors.iter().any(Option::is_some) {
                inputs.truncate(operation.base_input_count());
            }
            let input_types = instruction
                .inputs()
                .iter()
                .map(|id| region.atoms()[id.index()].r#type().into_owned())
                .collect::<Vec<_>>();
            let output_types = instruction
                .outputs()
                .iter()
                .map(|id| region.atoms()[id.index()].r#type().into_owned())
                .collect::<Vec<_>>();
            let location = self.location;
            let native = |position: usize| inputs[position].native();
            let results = match instruction.operation() {
                KernelOperation::Portable(ArrayIrOperation::Assert(_))
                | KernelOperation::Portable(ArrayIrOperation::Array(ArrayOperation::Assert(_))) => {
                    return Err(unsupported("assert", "runtime assertions require native failure propagation"));
                }
                KernelOperation::Portable(ArrayIrOperation::Array(operation)) => {
                    let array_types = input_types
                        .iter()
                        .map(|r#type| match r#type {
                            ArrayIrType::Array(value) => Ok(value.clone()),
                            _ => Err(unsupported(operation.name(), "expected ordinary array operands")),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let ArrayIrType::Array(output) = &output_types[0] else { unreachable!() };
                    let values = inputs.iter().map(LoweringValue::native).collect::<Result<Vec<_>, _>>()?;
                    vec![LoweringValue::Native(self.array(block, operation, &values, &array_types, output)?)]
                }
                KernelOperation::Portable(ArrayIrOperation::Dimension(operation)) => {
                    let values = inputs.iter().map(LoweringValue::native).collect::<Result<Vec<_>, _>>()?;
                    vec![LoweringValue::Native(self.dimension(block, operation, &values)?)]
                }
                KernelOperation::Portable(ArrayIrOperation::Compare(operation)) => vec![LoweringValue::Native(append(
                    block,
                    arith::cmpi(native(0)?, native(1)?, integer_comparison(operation.direction(), false), location)?,
                )?)],
                KernelOperation::Portable(ArrayIrOperation::DimensionToScalar(_)) => vec![inputs[0].clone()],
                KernelOperation::Portable(ArrayIrOperation::DimensionFromScalar(operation)) => {
                    let bounds = operation.output_type().bounds();
                    let minimum = self.integer(block, bounds.lower() as i64)?;
                    let maximum = self.integer(block, bounds.upper().unwrap() as i64)?;
                    let above = append(
                        block,
                        arith::cmpi(
                            native(0)?,
                            minimum,
                            arith::IntegerComparisonPredicate::SignedGreaterThanOrEqual,
                            location,
                        )?,
                    )?;
                    let below = append(
                        block,
                        arith::cmpi(native(0)?, maximum, arith::IntegerComparisonPredicate::SignedLessThan, location)?,
                    )?;
                    let valid = append(block, arith::andi(above, below, location)?)?;
                    block.append_operation(tt::assert(valid, "dimension bounds", location)?)?;
                    vec![inputs[0].clone()]
                }
                KernelOperation::Portable(ArrayIrOperation::DimensionSize(operation)) => {
                    vec![LoweringValue::Native(
                        self.integer(
                            block,
                            operation.input_dimension().value().ok_or_else(|| {
                                unsupported("dimension_size", "dynamic dimensions require specialization")
                            })? as i64,
                        )?,
                    )]
                }
                KernelOperation::Portable(ArrayIrOperation::ReferenceRead(_)) => {
                    let zero = append(
                        block,
                        arith::constant(self.context.float_attribute(self.context.float32_type(), 0.0), location)?,
                    )?;
                    let reference = inputs[0].reference()?;
                    let zero = self.splat(block, zero, DataType::F32, &reference.shape)?;
                    vec![LoweringValue::Native(self.read(block, reference, None, zero)?)]
                }
                KernelOperation::Portable(ArrayIrOperation::ReferenceWrite(_))
                | KernelOperation::Portable(ArrayIrOperation::ReferenceSwap(_)) => {
                    // Admission rejects every swap whose old value is used, so a swap is a store whose dead old value
                    // is never bound.
                    self.write(block, inputs[0].reference()?, native(1)?, None)?;
                    vec![]
                }
                KernelOperation::Portable(ArrayIrOperation::ReferenceAddUpdate(_)) => {
                    let reference = inputs[0].reference()?;
                    let update = self.update(block, operation.name(), reference, native(1)?, &input_types[1])?;
                    let (pointer, valid) = self.pointers(block, reference)?;
                    let zero = append(
                        block,
                        arith::constant(self.context.float_attribute(self.context.float32_type(), 0.0), location)?,
                    )?;
                    let zero = self.splat(block, zero, DataType::F32, &reference.shape)?;
                    let r#type = self.r#type(DataType::F32, &reference.shape)?;
                    let old =
                        append(block, tt::load(pointer, Some(valid), Some(zero), None, false, r#type, location)?)?;
                    let sum = append(block, arith::addf(old, update, location)?)?;
                    block.append_operation(tt::store(pointer, sum, Some(valid), None, false, location)?)?;
                    vec![]
                }
                KernelOperation::Portable(ArrayIrOperation::ReferenceAtomicAddUpdate(_)) => {
                    let reference = inputs[0].reference()?;
                    let update = self.update(block, operation.name(), reference, native(1)?, &input_types[1])?;
                    let (pointer, valid) = self.pointers(block, reference)?;
                    block.append_operation(tt::atomic_rmw(
                        tt::RmwOp::FloatAdd,
                        pointer,
                        update,
                        Some(valid),
                        tt::MemSemantic::AcquireRelease,
                        tt::MemSyncScope::Gpu,
                        self.r#type(DataType::F32, &reference.shape)?,
                        location,
                    )?)?;
                    vec![]
                }
                KernelOperation::TileLoad(operation) => {
                    let mut reference = inputs[0].reference()?.clone();
                    reference.shape = operation.block_shape().to_vec();
                    for (axis, start) in reference.starts.iter_mut().enumerate() {
                        *start = append(block, arith::addi(*start, native(axis + 1)?, location)?)?;
                    }
                    let other = self.splat(block, native(inputs.len() - 1)?, DataType::F32, &reference.shape)?;
                    vec![LoweringValue::Native(self.read(block, &reference, None, other)?)]
                }
                KernelOperation::MaskedLoad(_) => {
                    vec![LoweringValue::Native(self.read(block, inputs[0].reference()?, Some(native(1)?), native(2)?)?)]
                }
                KernelOperation::MaskedStore(_) => {
                    self.write(block, inputs[0].reference()?, native(1)?, Some(native(2)?))?;
                    vec![]
                }
                KernelOperation::MaskedSwap(_) => {
                    // Admission rejects every swap whose old value is used, so this one is a masked store and its
                    // dead old value is never bound.
                    self.write(block, inputs[0].reference()?, native(1)?, Some(native(2)?))?;
                    vec![]
                }
                KernelOperation::WindowMask(_) => {
                    vec![LoweringValue::Native(self.validity(block, inputs[0].reference()?)?.1)]
                }
                KernelOperation::Portable(ArrayIrOperation::While(operation)) => {
                    let bound = operation
                        .iteration_bound()
                        .filter(|bound| *bound <= i64::MAX as usize)
                        .ok_or_else(|| unsupported("while", "requires a finite signed 64-bit iteration bound"))?;
                    let mut initial = inputs
                        .iter()
                        .filter_map(|input| match input {
                            LoweringValue::Native(value) => Some(*value),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    initial.push(self.integer(block, 0)?);
                    let types = initial.iter().map(|value| value.r#type()).collect::<Result<Vec<_>, _>>()?;
                    let arguments = types.iter().map(|r#type| (*r#type, location)).collect::<Vec<_>>();
                    let mut before = self.context.block(&arguments);
                    let before_values = carried_values(&before, &inputs)?;
                    let counter = before.argument(types.len() - 1)?.as_ref();
                    let limit = self.integer(&mut before, bound as i64)?;
                    let within = append(
                        &mut before,
                        arith::cmpi(counter, limit, arith::IntegerComparisonPredicate::SignedLessThan, location)?,
                    )?;
                    // The semantic bound is checked before the condition, including reference writes or assertions
                    // in that condition. A Boolean conjunction would evaluate the condition once too often.
                    let mut condition = self.context.block_with_no_arguments();
                    let predicate =
                        self.region(&mut condition, region.with_id(instruction.regions()[0])?, &before_values)?;
                    condition.append_operation(scf::r#yield(&[predicate[0].native()?], location)?)?;
                    let mut exhausted = self.context.block_with_no_arguments();
                    let stopped = self.boolean(&mut exhausted, false)?;
                    exhausted.append_operation(scf::r#yield(&[stopped], location)?)?;
                    let predicate = append(
                        &mut before,
                        scf::r#if(
                            within,
                            &[self.context.signless_integer_type(1).as_ref()],
                            condition.try_into()?,
                            Some(exhausted.try_into()?),
                            location,
                        )?,
                    )?;
                    let forwarded = (0..types.len())
                        .map(|index| before.argument(index).map(|value| value.as_ref()))
                        .collect::<Result<Vec<_>, _>>()?;
                    before.append_operation(scf::condition(predicate, &forwarded, location)?)?;
                    let mut after = self.context.block(&arguments);
                    let after_values = carried_values(&after, &inputs)?;
                    let next = self.region(&mut after, region.with_id(instruction.regions()[1])?, &after_values)?;
                    let mut yielded = Vec::new();
                    for (input, value) in inputs.iter().zip(&next) {
                        match (input, value) {
                            (LoweringValue::Native(_), LoweringValue::Native(value)) => yielded.push(*value),
                            (LoweringValue::Reference(left), LoweringValue::Reference(right)) if left == right => {}
                            _ => {
                                return Err(unsupported(
                                    "while",
                                    "loop-carried reference windows must remain unchanged",
                                ));
                            }
                        }
                    }
                    let counter = after.argument(types.len() - 1)?.as_ref();
                    let one = self.integer(&mut after, 1)?;
                    yielded.push(append(&mut after, arith::addi(counter, one, location)?)?);
                    after.append_operation(scf::r#yield(&yielded, location)?)?;
                    let operation = block.append_operation(scf::r#while(
                        &initial,
                        &types,
                        before.try_into()?,
                        after.try_into()?,
                        location,
                    )?)?;
                    let mut position = 0;
                    inputs
                        .iter()
                        .map(|input| match input {
                            LoweringValue::Native(_) => {
                                let value = operation.result(position)?.as_ref();
                                position += 1;
                                Ok(LoweringValue::Native(value))
                            }
                            LoweringValue::Reference(_) => Ok(input.clone()),
                        })
                        .collect::<Result<Vec<_>, Error>>()?
                }
                operation => {
                    return Err(unsupported(operation.name(), "operation is outside the portable Triton subset"));
                }
            };
            for (id, result) in instruction.outputs().iter().zip(results) {
                values[id.index()] = Some(result);
            }
        }
        Ok(region.output_ids().iter().map(|id| values[id.index()].clone().unwrap()).collect())
    }

    /// Constructs admitted arithmetic over canonical logical array types.
    fn array(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        operation: &ArrayOperation<Array>,
        inputs: &[Native<'c, 't>],
        types: &[ArrayType],
        output: &ArrayType,
    ) -> Result<Native<'c, 't>, Error> {
        let output_shape = shape(output)?;
        let output_type = self.r#type(output.data_type(), &output_shape)?;
        let location = self.location;
        let mut values = inputs.to_vec();
        if matches!(
            operation,
            ArrayOperation::Add(_)
                | ArrayOperation::Sub(_)
                | ArrayOperation::Mul(_)
                | ArrayOperation::Compare(_)
                | ArrayOperation::Select(_)
        ) {
            // Canonical inference broadcasts elementwise operands and promotes their element types, but the native
            // operations require operands of one type. Rank-0 operands are therefore splat to the result shape, and
            // every other broadcast or promotion is rejected. The predicate of a select is Boolean by construction.
            for (value, r#type) in values.iter_mut().zip(types) {
                if r#type.rank() == 0 && !output_shape.is_empty() {
                    *value = self.splat(block, *value, r#type.data_type(), &output_shape)?;
                } else if shape(r#type)? != output_shape {
                    return Err(unsupported(
                        operation.name(),
                        "elementwise operands require equal shapes or scalar broadcasting",
                    ));
                }
            }
            let data_types = if matches!(operation, ArrayOperation::Select(_)) { &types[1..] } else { types };
            if data_types.windows(2).any(|pair| pair[0].data_type() != pair[1].data_type()) {
                return Err(unsupported(operation.name(), "elementwise operands require equal element types"));
            }
        }
        Ok(match operation {
            ArrayOperation::Constant(operation) => self.constant(block, operation.value())?,
            ArrayOperation::Zero(_)
            | ArrayOperation::ZeroLike(_)
            | ArrayOperation::One(_)
            | ArrayOperation::OneLike(_) => {
                let one = matches!(operation, ArrayOperation::One(_) | ArrayOperation::OneLike(_));
                let scalar = match output.data_type() {
                    DataType::F32 => append(
                        block,
                        arith::constant(
                            self.context.float_attribute(self.context.float32_type(), if one { 1.0 } else { 0.0 }),
                            location,
                        )?,
                    )?,
                    DataType::I64 => self.integer(block, i64::from(one))?,
                    _ => self.boolean(block, one)?,
                };
                self.splat(block, scalar, output.data_type(), &output_shape)?
            }
            ArrayOperation::FillLike(operation) => {
                let scalar = self.constant(block, operation.value())?;
                self.splat(block, scalar, output.data_type(), &output_shape)?
            }
            ArrayOperation::Iota(iota) => self.iota(block, operation, iota.dimension(), output)?,
            ArrayOperation::IotaLike(iota) => self.iota(block, operation, iota.dimension(), output)?,
            ArrayOperation::Add(_) if output.data_type() == DataType::F32 => {
                append(block, arith::addf(values[0], values[1], location)?)?
            }
            ArrayOperation::Add(_) if output.data_type() == DataType::I64 => {
                append(block, arith::addi(values[0], values[1], location)?)?
            }
            ArrayOperation::Sub(_) if output.data_type() == DataType::F32 => {
                append(block, arith::subf(values[0], values[1], location)?)?
            }
            ArrayOperation::SaturatingSub(_) => {
                return Err(unsupported(operation.name(), "integer saturation has no portable Triton implementation"));
            }
            ArrayOperation::Mul(_) if output.data_type() == DataType::F32 => {
                append(block, arith::mulf(values[0], values[1], location)?)?
            }
            ArrayOperation::Neg(_) if output.data_type() == DataType::F32 => {
                append(block, arith::negf(inputs[0], location)?)?
            }
            ArrayOperation::Compare(operation) => {
                if types[0].data_type() == DataType::F32 {
                    append(
                        block,
                        arith::cmpf(values[0], values[1], float_comparison(operation.direction()), location)?,
                    )?
                } else {
                    append(
                        block,
                        arith::cmpi(
                            values[0],
                            values[1],
                            integer_comparison(operation.direction(), types[0].data_type() == DataType::Boolean),
                            location,
                        )?,
                    )?
                }
            }
            ArrayOperation::Select(_) => append(block, arith::select(values[0], values[1], values[2], location)?)?,
            ArrayOperation::Reshape(_)
                if shape(&types[0])?.iter().chain(output_shape.iter()).all(|extent| extent.is_power_of_two()) =>
            {
                if types[0].rank() == 0 {
                    self.splat(block, inputs[0], output.data_type(), &output_shape)?
                } else if output_shape.is_empty() {
                    append(block, tt::unsplat(inputs[0], output_type, location)?)?
                } else {
                    append(block, tt::reshape(inputs[0], false, false, output_type, location)?)?
                }
            }
            ArrayOperation::Broadcast(broadcast) => {
                self.broadcast(block, operation, inputs[0], &types[0], broadcast.output_axes(), output)?
            }
            ArrayOperation::BroadcastLike(broadcast) => {
                self.broadcast(block, operation, inputs[0], &types[0], broadcast.output_axes(), output)?
            }
            ArrayOperation::Transpose(operation) => {
                let permutation =
                    operation.permutation().normalize(output_shape.len()).map_err(ryft_core::ProgramError::from)?;
                append(
                    block,
                    tt::trans(
                        inputs[0],
                        &permutation.iter().map(|axis| *axis as i32).collect::<Vec<_>>(),
                        output_type,
                        location,
                    )?,
                )?
            }
            ArrayOperation::Reduce(operation)
                if operation.kind() == ReductionKind::Sum && output.data_type() == DataType::F32 =>
            {
                let mut current_shape = shape(&types[0])?;
                let (_, mask) = self.indices(block, &current_shape)?;
                let zero = append(
                    block,
                    arith::constant(self.context.float_attribute(self.context.float32_type(), 0.0), location)?,
                )?;
                let zero = self.splat(block, zero, DataType::F32, &current_shape)?;
                let mut result = append(block, arith::select(mask, inputs[0], zero, location)?)?;
                let mut axes = operation.axes().to_vec();
                axes.sort_unstable_by(|left, right| right.cmp(left));
                for axis in axes {
                    current_shape.remove(axis);
                    let float = self.context.float32_type();
                    let mut combine = self.context.block(&[(float, location); 2]);
                    let left = combine.argument(0)?.as_ref();
                    let right = combine.argument(1)?.as_ref();
                    let sum = append(&mut combine, arith::addf(left, right, location)?)?;
                    combine.append_operation(tt::reduce_return(&[sum], location)?)?;
                    result = append(
                        block,
                        tt::reduce(
                            &[result],
                            axis as i64,
                            &[self.r#type(DataType::F32, &current_shape)?],
                            combine.try_into()?,
                            location,
                        )?,
                    )?;
                }
                result
            }
            ArrayOperation::Dot(operation)
                if operation.dimensions() == &DotDimensionNumbers::matmul()
                    && types.len() == 2
                    && types.iter().all(|r#type| r#type.rank() == 2 && r#type.data_type() == DataType::F32)
                    && output.data_type() == DataType::F32
                    && operation.output_sharding().is_none() =>
            {
                let zero = append(
                    block,
                    arith::constant(self.context.float_attribute(self.context.float32_type(), 0.0), location)?,
                )?;
                let mut operands = Vec::new();
                for (value, r#type) in inputs.iter().zip(types) {
                    let shape = shape(r#type)?;
                    let (_, mask) = self.indices(block, &shape)?;
                    let padding = self.splat(block, zero, DataType::F32, &shape)?;
                    operands.push(append(block, arith::select(mask, *value, padding, location)?)?);
                }
                let accumulator = self.splat(block, zero, DataType::F32, &output_shape)?;
                append(
                    block,
                    tt::dot(operands[0], operands[1], accumulator, tt::InputPrecision::Ieee, 0, output_type, location)?,
                )?
            }
            _ => {
                return Err(unsupported(
                    operation.name(),
                    "array operation or geometry is outside the portable Triton subset",
                ));
            }
        })
    }
}

/// Rejects every body whose memory accesses need ordering between the threads of one program, which the module
/// documentation describes.
fn validate_memory_ordering(kernel: &VerifiedKernel<'_>) -> Result<(), Error> {
    let call = kernel.definition().operation();
    let body = kernel.definition().body().entry_region_ref();
    let references = kernel.definition().references();
    if let Some((_, instruction)) = body
        .instructions_in_closure()
        .find(|(id, _)| references.swap_lowering(*id) == Some(KernelSwapLowering::Exchange))
    {
        return Err(unordered(instruction.operation().name(), "swaps whose old value is used"));
    }

    // Roots of the parameters whose accesses the scan below orders, with whether each one is read-write.
    let mut ordered = HashMap::new();
    for (parameter, summary) in call.parameters().iter().zip(references.parameters()) {
        let Some(summary) = summary else { continue };
        let modes = summary.modes();
        if modes.contains(&ReferenceAccessMode::AtomicAccumulate) {
            if modes.len() > 1 {
                return Err(unordered("kernel_call", "atomic and other accesses of one parameter"));
            }
            continue;
        }
        let read_write = match parameter.access() {
            KernelParameterAccess::ReadOnly => continue,
            KernelParameterAccess::WriteOnly => {
                // A swap of a write-only parameter is a store, because swaps whose old value is used are rejected.
                if modes.iter().any(|mode| !matches!(mode, ReferenceAccessMode::Write | ReferenceAccessMode::ReadWrite))
                {
                    return Err(unordered("kernel_call", "reads after writes of one parameter"));
                }
                if parameter.mapping().block_shape().is_empty() {
                    continue;
                }
                false
            }
            KernelParameterAccess::ReadWrite => true,
        };
        let tiling = parameter.mapping().tiling_axes();
        if call.grid().dimensions().iter().enumerate().any(|(axis, dimension)| {
            dimension.execution() != GridExecution::Parallel
                && !matches!(dimension.extent(), Dimension::Static(0 | 1))
                && !tiling.as_ref().is_some_and(|axes| axes.contains(&Some(axis)))
        }) {
            return Err(unordered("kernel_call", "windows that repeat along a sequential grid axis"));
        }
        ordered.insert(summary.root(), read_write);
    }

    // Direct accesses of the ordered parameters by entry instruction, in program order.
    let analysis = references.analysis().analysis();
    let mut accesses = HashMap::<usize, Vec<_>>::new();
    for access in analysis.access_modes() {
        if access.instruction().region() == body.id() && ordered.contains_key(&access.root()) {
            accesses.entry(access.instruction().index()).or_default().push(*access);
        }
    }
    for (index, instruction) in body.instructions().iter().enumerate() {
        if !instruction.regions().is_empty()
            && analysis
                .transitive_access(InstructionId::new(body.id(), index))
                .is_some_and(|access| access.roots().any(|root| ordered.contains_key(&root)))
        {
            return Err(unordered(instruction.operation().name(), "accesses of a mutable window inside control flow"));
        }
    }
    let mut written = HashSet::new();
    // Values computed elementwise from reads of read-write parameters, with the root of the parameter read.
    let mut read = HashMap::<AtomId, ReferenceRoot>::new();
    for (index, instruction) in body.instructions().iter().enumerate() {
        let operation = instruction.operation();
        let mut target = None;
        for access in accesses.get(&index).into_iter().flatten() {
            let root = access.root();
            if ordered[&root]
                && (matches!(operation, KernelOperation::TileLoad(_))
                    || !references.path(access.instruction(), access.input_index()).is_some_and(|path| path.is_root()))
            {
                return Err(unsupported(
                    operation.name(),
                    "read-write parameters must be accessed through their whole window",
                ));
            }
            match access.mode() {
                ReferenceAccessMode::Read if written.contains(&root) => {
                    return Err(unordered(operation.name(), "reads after writes of one parameter"));
                }
                ReferenceAccessMode::Read => read.extend(instruction.outputs().iter().map(|output| (*output, root))),
                ReferenceAccessMode::Accumulate if written.contains(&root) => {
                    return Err(unordered(operation.name(), "reads after writes of one parameter"));
                }
                _ if !written.insert(root) => {
                    return Err(unordered(operation.name(), "repeated writes of one window"));
                }
                _ => target = Some(root),
            }
        }
        let sources = instruction.inputs().iter().filter_map(|input| read.get(input).copied()).collect::<HashSet<_>>();
        let elementwise = matches!(
            operation,
            KernelOperation::Portable(
                ArrayIrOperation::Compare(_)
                    | ArrayIrOperation::Array(
                        ArrayOperation::Add(_)
                            | ArrayOperation::Sub(_)
                            | ArrayOperation::Mul(_)
                            | ArrayOperation::Neg(_)
                            | ArrayOperation::Compare(_)
                            | ArrayOperation::Select(_)
                    ),
            )
        );
        match sources.into_iter().collect::<Vec<_>>().as_slice() {
            [] => {}
            [root] if elementwise => read.extend(instruction.outputs().iter().map(|output| (*output, *root))),
            [root] if target == Some(*root) => {}
            _ => {
                return Err(unsupported(
                    operation.name(),
                    "values read from a read-write parameter may reach memory only elementwise through its own write",
                ));
            }
        }
    }
    Ok(())
}

/// Appends one-result operations while preserving context-owned native lifetimes.
fn append<'c, 't, O: DetachedOp<'c, 'c, 't>>(
    block: &mut DetachedBlock<'c, 't>,
    operation: O,
) -> Result<Native<'c, 't>, Error> {
    Ok(block.append_operation(operation)?.result(0)?.as_ref())
}

/// Rebinds ordinary loop carries while preserving invariant reference windows.
fn carried_values<'c, 't>(
    block: &DetachedBlock<'c, 't>,
    inputs: &[LoweringValue<'c, 't>],
) -> Result<Vec<LoweringValue<'c, 't>>, Error> {
    let mut position = 0;
    inputs
        .iter()
        .map(|input| match input {
            LoweringValue::Native(_) => {
                let value = block.argument(position)?.as_ref();
                position += 1;
                Ok(LoweringValue::Native(value))
            }
            LoweringValue::Reference(_) => Ok(input.clone()),
        })
        .collect()
}

/// Checks logical shape and byte-offset bounds without applying tile limits to full global arrays.
fn shape(r#type: &ArrayType) -> Result<Vec<usize>, Error> {
    let shape = r#type.static_shape().ok_or_else(|| unsupported("array", "static shapes are required"))?;
    let mut count = 1usize;
    for extent in shape.dimensions() {
        if *extent > i32::MAX as usize {
            return Err(unsupported("array", "array extents exceed signed 32-bit indexing"));
        }
        count = count
            .checked_mul(*extent)
            .filter(|count| *count <= i64::MAX as usize / 4)
            .ok_or_else(|| unsupported("array", "array byte offsets exceed signed 64-bit indexing"))?;
    }
    Ok(shape.dimensions().to_vec())
}

/// Bounds symbolic indexing so window-origin additions remain representable before masking.
fn check_dimension(r#type: &ryft_core::DimensionType) -> Result<(), Error> {
    if r#type.bounds().upper().is_none_or(|upper| upper > i32::MAX as usize + 1) {
        return Err(unsupported("dimension", "dimension bounds must fit the signed 32-bit indexing domain"));
    }
    Ok(())
}

/// Maps canonical integer predicates without changing signed ordering.
fn integer_comparison(direction: ComparisonDirection, unsigned: bool) -> arith::IntegerComparisonPredicate {
    use arith::IntegerComparisonPredicate;
    match direction {
        ComparisonDirection::Equal => IntegerComparisonPredicate::Equal,
        ComparisonDirection::NotEqual => IntegerComparisonPredicate::NotEqual,
        ComparisonDirection::LessThan if unsigned => IntegerComparisonPredicate::UnsignedLessThan,
        ComparisonDirection::LessThan => IntegerComparisonPredicate::SignedLessThan,
        ComparisonDirection::LessThanOrEqual if unsigned => IntegerComparisonPredicate::UnsignedLessThanOrEqual,
        ComparisonDirection::LessThanOrEqual => IntegerComparisonPredicate::SignedLessThanOrEqual,
        ComparisonDirection::GreaterThan if unsigned => IntegerComparisonPredicate::UnsignedGreaterThan,
        ComparisonDirection::GreaterThan => IntegerComparisonPredicate::SignedGreaterThan,
        ComparisonDirection::GreaterThanOrEqual if unsigned => IntegerComparisonPredicate::UnsignedGreaterThanOrEqual,
        ComparisonDirection::GreaterThanOrEqual => IntegerComparisonPredicate::SignedGreaterThanOrEqual,
    }
}

/// Maps canonical floating predicates, retaining unordered inequality.
fn float_comparison(direction: ComparisonDirection) -> arith::FloatingPointComparisonPredicate {
    use arith::FloatingPointComparisonPredicate;
    match direction {
        ComparisonDirection::Equal => FloatingPointComparisonPredicate::Equal,
        ComparisonDirection::NotEqual => FloatingPointComparisonPredicate::NotEqual,
        ComparisonDirection::LessThan => FloatingPointComparisonPredicate::LessThan,
        ComparisonDirection::LessThanOrEqual => FloatingPointComparisonPredicate::LessThanOrEqual,
        ComparisonDirection::GreaterThan => FloatingPointComparisonPredicate::GreaterThan,
        ComparisonDirection::GreaterThanOrEqual => FloatingPointComparisonPredicate::GreaterThanOrEqual,
    }
}

/// Preserves the canonical operation owner in admission diagnostics.
fn unsupported(operation: &'static str, reason: &str) -> Error {
    Error::Unsupported { operation, reason: reason.to_owned() }
}

/// Rejects `accesses` whose correctness depends on ordering between the threads of one program.
fn unordered(operation: &'static str, accesses: &str) -> Error {
    unsupported(operation, &format!("{accesses} need a CTA barrier, which ryft-mlir does not expose"))
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArraySliceAxis, BroadcastLikeOperation, BroadcastOperation, Context as CoreContext, FillLikeOperation,
        IotaLikeOperation, IotaOperation, Placeholder, ProgramBuilder, ReferenceRead, ReferenceReadOperation,
        ReferenceWrite, ReferenceWriteOperation,
    };
    use ryft_kernels::{
        BoundaryPolicy, Grid, GridDimension, KernelCallOperation, KernelDefinition, KernelParameterAccess,
        MappingExpression, ParameterWindow, compare, kernel, mapped_call, masked_load, masked_load_or, masked_store,
        masked_swap, whole_array_parameter, window_mask,
    };
    use ryft_mlir::{OpRef, Region};

    use super::*;

    /// Builds a call over one parallel program whose parameters have the given types, accesses, and boundary policies.
    /// Every window has three elements, so the window of a two element array has one padding element, and its physical
    /// tile has four elements.
    fn tiled_call(parameters: Vec<(ArrayType, KernelParameterAccess, BoundaryPolicy)>) -> KernelCallOperation {
        mapped_call(
            Grid::new(vec![GridDimension::new(1usize.into(), GridExecution::Parallel)]).unwrap(),
            parameters
                .into_iter()
                .map(|(r#type, access, boundary)| {
                    let window = ParameterWindow::BlockIndices {
                        block_shape: vec![3],
                        boundary,
                        indices: vec![MappingExpression::GridCoordinate(0)],
                    };
                    (r#type, access, window)
                })
                .collect(),
            vec![],
        )
        .unwrap()
    }

    /// Lowers `definition` and returns its complete native module.
    fn lowered_module(definition: &KernelDefinition) -> String {
        let context = Context::new();
        let verified = VerifiedKernel::new(definition, 1).unwrap();
        lower(&context, &verified, &Options::default(), &KernelSchedule::default()).unwrap().module.to_string()
    }

    /// Lowers one array operation over native block arguments of the given input types. It returns the names of the
    /// emitted operations, the values of their non-Boolean integer literals, and the position of the operation that
    /// defines the result.
    fn lowered_array(
        operation: ArrayOperation<Array>,
        types: Vec<ArrayType>,
    ) -> Result<(Vec<String>, Vec<i64>, Option<usize>), Error> {
        let context = Context::new();
        let mut lowering = Lowering {
            context: &context,
            location: context.unknown_location(),
            remaining: 100,
            maximum_tile_elements: 1024,
        };
        let arguments = types
            .iter()
            .map(|r#type| (lowering.r#type(r#type.data_type(), &shape(r#type).unwrap()).unwrap(), lowering.location))
            .collect::<Vec<_>>();
        let mut block = context.block(&arguments);
        let inputs = (0..types.len()).map(|index| block.argument(index).unwrap().as_ref()).collect::<Vec<_>>();
        let output = operation.infer_output_types(&types, &[]).unwrap().remove(0);
        let result = lowering.array(&mut block, &operation, &inputs, &types, &output)?;
        let mut names = Vec::new();
        let mut literals = Vec::new();
        let mut position = None;
        for (index, native) in block.operations().unwrap().enumerate() {
            let native = native.unwrap();
            let name = native.name().as_str().unwrap().to_owned();
            let value = native.result(0).unwrap().as_ref();
            if let ("arith.constant", Ok(attribute)) = (name.as_str(), native.integer_attribute("value"))
                && value.r#type().unwrap() != context.signless_integer_type(1).as_ref()
            {
                literals.push(attribute.signless_value());
            }
            if value == result {
                position = Some(index);
            }
            names.push(name);
        }
        Ok((names, literals, position))
    }

    /// Covers global windows and the partially populated final tile through the public macro.
    #[kernel(crate = "ryft_kernels", requires = left.shape()[0] == right.shape()[0])]
    fn vector(
        #[input(data_type = F32, rank = 1)] left: &Array,
        #[input(data_type = F32, rank = 1)] right: &Array,
        #[output(data_type = F32, shape = [left.shape()[0]], tile = [256], boundary = masked)] output: &mut Array,
    ) {
        let [block] = output.tile_index();
        let left_tiles = left.tiles([256]).pad(0.0);
        let right_tiles = right.tiles([256]).pad(0.0);
        output.store(left_tiles.load([block]) + right_tiles.load([block]));
    }

    /// Covers physical padding in an odd-length reduction.
    #[kernel(crate = "ryft_kernels")]
    fn sum(
        #[input(data_type = F32, rank = 1)] input: &Array,
        #[output(data_type = F32, shape = [])] output: &mut Array,
    ) {
        output.store(input.load().sum([0]));
    }

    /// Covers invariant global references and an ordinary tile accumulator in a bounded loop.
    #[kernel(crate = "ryft_kernels", requires = left.shape()[1] == right.shape()[0])]
    fn matrix(
        #[input(data_type = F32, rank = 2)] left: &Array,
        #[input(data_type = F32, rank = 2)] right: &Array,
        #[output(data_type = F32, shape = [left.shape()[0], right.shape()[1]], tile = [32, 32], boundary = masked)]
        output: &mut Array,
    ) {
        let [row, column] = output.tile_index();
        let left_tiles = left.tiles([32, 32]).pad(0.0);
        let right_tiles = right.tiles([32, 32]).pad(0.0);
        let mut accumulator = zeros::<f32>([32, 32]);
        for depth in 0..left.shape()[1].div_ceil(32) {
            accumulator += left_tiles.load([row, depth]).dot(right_tiles.load([depth, column]));
        }
        output.store(accumulator);
    }

    /// Doubles every row on a sequential axis whose last window has one padding element.
    #[kernel(crate = "ryft_kernels", grid(i = input.shape()[0], k = sequential(input.shape()[1].div_ceil(2))))]
    fn double_rows(
        #[input(data_type = F32, rank = 2, block = [1, 2], index = [i, k], boundary = masked)] input: &Array,
        #[output(
            data_type = F32,
            shape = [input.shape()[0], input.shape()[1]],
            block = [1, 2],
            index = [i, k],
            boundary = masked,
        )]
        output: &mut Array,
    ) {
        let value = input.load_or(0.0);
        output.store(value + value);
    }

    /// Adds `input` into `accumulator` in place, reading the accumulator before its only write.
    #[kernel(crate = "ryft_kernels", requires = input.shape()[0] == accumulator.shape()[0])]
    fn add_in_place(
        #[input(data_type = F32, rank = 1)] input: &Array,
        #[inout(data_type = F32, rank = 1)] accumulator: &mut Array,
    ) {
        accumulator.store(accumulator.load() + input.load());
    }

    /// Adds `input` into `accumulator` through one add update.
    #[kernel(crate = "ryft_kernels", requires = input.shape()[0] == accumulator.shape()[0])]
    fn accumulate_in_place(
        #[input(data_type = F32, rank = 1)] input: &Array,
        #[inout(data_type = F32, rank = 1)] accumulator: &mut Array,
    ) {
        accumulator.accumulate(input.load());
    }

    /// Adds the sum of every window of `input` to `total` atomically.
    #[kernel(crate = "ryft_kernels", grid(i = input.shape()[0].div_ceil(2)))]
    fn parallel_sum(
        #[input(data_type = F32, rank = 1, block = [2], index = [i])] input: &Array,
        #[inout(data_type = F32, rank = 0)] total: &mut Array,
    ) {
        total.atomic_accumulate(input.load().sum([0]));
    }

    /// Zeroes the output at the first step of a sequential axis and accumulates every masked input window into it.
    #[kernel(crate = "ryft_kernels", grid(k = sequential(input.shape()[0].div_ceil(2))))]
    fn carried_sum(
        #[input(data_type = F32, rank = 1, block = [2], index = [k], boundary = masked)] input: &Array,
        #[output(data_type = F32, shape = [2])] output: &mut Array,
    ) {
        if k == 0 {
            output.store(zeros::<f32>([2]));
        }
        output.accumulate(input.load_or(0.0));
    }

    /// Stores every input window into the same output window, so the last step of the sequential axis wins.
    #[kernel(crate = "ryft_kernels", grid(k = sequential(input.shape()[0].div_ceil(2))))]
    fn carried_copy(
        #[input(data_type = F32, rank = 1, block = [2], index = [k])] input: &Array,
        #[output(data_type = F32, shape = [2])] output: &mut Array,
    ) {
        output.store(input.load());
    }

    /// Reads `values` back after writing it.
    #[kernel(crate = "ryft_kernels")]
    fn read_after_write(
        #[inout(data_type = F32, rank = 1)] values: &mut Array,
        #[output(data_type = F32, shape = [values.shape()[0]])] output: &mut Array,
    ) {
        values.store(values.load() + values.load());
        output.store(values.load());
    }

    /// Writes the same output window twice.
    #[kernel(crate = "ryft_kernels")]
    fn write_twice(
        #[input(data_type = F32, rank = 1)] input: &Array,
        #[output(data_type = F32, shape = [input.shape()[0]])] output: &mut Array,
    ) {
        output.store(input.load());
        output.store(input.load() + input.load());
    }

    /// Updates the first element of `values` in place through a slice of its window.
    #[kernel(crate = "ryft_kernels", requires = values.shape()[0] >= 1)]
    fn update_first(#[inout(data_type = F32, rank = 1)] values: &mut Array) {
        values.slice([0..1]).store(values.slice([0..1]).load());
    }

    /// Copies `values` into `output` before updating it in place.
    #[kernel(crate = "ryft_kernels")]
    fn copy_then_update(
        #[inout(data_type = F32, rank = 1)] values: &mut Array,
        #[output(data_type = F32, shape = [values.shape()[0]])] output: &mut Array,
    ) {
        let value = values.load();
        output.store(value);
        values.store(value + value);
    }

    /// Doubles `values` in place in every iteration of a loop.
    #[kernel(crate = "ryft_kernels")]
    fn double_repeatedly(#[inout(data_type = F32, rank = 1)] values: &mut Array) {
        for step in 0..values.shape()[0] {
            values.store(values.load() + values.load());
        }
    }

    /// Accumulates `input` into `total` atomically and then reads `total`.
    #[kernel(crate = "ryft_kernels")]
    fn read_atomic_total(
        #[input(data_type = F32, rank = 1)] input: &Array,
        #[inout(data_type = F32, rank = 0)] total: &mut Array,
        #[output(data_type = F32, shape = [])] output: &mut Array,
    ) {
        total.atomic_accumulate(input.load().sum([0]));
        output.store(total.load());
    }

    /// Returns the operation and reason of the admission failure of `definition`.
    fn rejection(definition: &KernelDefinition) -> (&'static str, String) {
        let context = Context::new();
        let verified = VerifiedKernel::new(definition, 1024).unwrap();
        match lower(&context, &verified, &Options::default(), &KernelSchedule::default()) {
            Err(Error::Unsupported { operation, reason }) => (operation, reason),
            Err(error) => panic!("expected an admission failure, got {error:?}"),
            Ok(_) => panic!("expected an admission failure"),
        }
    }

    /// The condition writes one and the body writes two; a bound of one must leave two in the output.
    fn condition_write_definition() -> ryft_kernels::KernelDefinition {
        use ryft_core::{Placeholder, ProgramBuilder, ReferenceWriteOperation, WhileOperation};
        use ryft_kernels::{Grid, KernelCallOperation, KernelDefinition, KernelParameterAccess, whole_array_parameter};

        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(ArrayType::scalar(DataType::F32), KernelParameterAccess::WriteOnly).unwrap()],
        )
        .unwrap();
        let mut condition = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = condition.add_input(call.parameters()[0].body_type());
        let one = condition.add_constant(ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()));
        condition
            .add_instruction(ReferenceWriteOperation::new(), vec![], vec![reference, one], None)
            .unwrap();
        let predicate = condition.add_constant(ArrayIrValue::Array(Array::scalar(true).unwrap()));
        let condition = condition
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![predicate],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let mut body = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = body.add_input(call.parameters()[0].body_type());
        let two = body.add_constant(ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()));
        body.add_instruction(ReferenceWriteOperation::new(), vec![], vec![reference, two], None).unwrap();
        let body = body
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![reference],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let mut entry = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = entry.add_input(call.parameters()[0].body_type());
        let condition = entry.import_region(condition.entry_region_ref());
        let body = entry.import_region(body.entry_region_ref());
        entry
            .add_instruction(
                ArrayIrOperation::While(WhileOperation::new().with_iteration_bound(1).unwrap()),
                vec![condition, body],
                vec![reference],
                None,
            )
            .unwrap();
        let body = entry
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder], vec![])
            .unwrap();
        KernelDefinition::new(call, body).unwrap()
    }

    #[test]
    fn test_lower() {
        let context = Context::new();
        let r#type = ArrayType::new_static(DataType::F32, [1003]);
        let definition = vector::definition(&r#type, &r#type).unwrap();
        let verified = VerifiedKernel::new(&definition, 1024).unwrap();
        let first = lower(&context, &verified, &Options::default(), &KernelSchedule::default()).unwrap();
        let second = lower(&context, &verified, &Options::default(), &KernelSchedule::default()).unwrap();
        assert_eq!(first.grid, [4, 1, 1]);
        assert_eq!(first.parameter_types, vec![r#type; 3]);
        assert_eq!(first.module.to_string(), second.module.to_string());
    }

    #[test]
    fn test_lower_empty_and_batched_windows() {
        let context = Context::new();
        let r#type = ArrayType::new_static(DataType::F32, [0]);
        let definition = vector::definition(&r#type, &r#type).unwrap();
        let lowered = lower(
            &context,
            &VerifiedKernel::new(&definition, 1024).unwrap(),
            &Options::default(),
            &KernelSchedule::default(),
        )
        .unwrap();
        assert_eq!(lowered.grid, [1, 1, 1]);
        let r#type = ArrayType::new_static(DataType::F32, [1003]);
        let definition = vector::definition(&r#type, &r#type)
            .unwrap()
            .batched(2, &[Some(0), Some(0), Some(0)], 1024)
            .unwrap();
        let lowered = lower(
            &context,
            &VerifiedKernel::new(&definition, 1024).unwrap(),
            &Options::default(),
            &KernelSchedule::default(),
        )
        .unwrap();
        assert_eq!(lowered.grid, [8, 1, 1]);
        assert_eq!(lowered.parameter_types, vec![ArrayType::new_static(DataType::F32, [2, 1003]); 3]);
    }

    #[test]
    fn test_lower_large_global_window() {
        let context = Context::new();
        let r#type = ArrayType::new_static(DataType::F32, [70_000]);
        let definition = vector::definition(&r#type, &r#type).unwrap();
        let lowered = lower(
            &context,
            &VerifiedKernel::new(&definition, 1024).unwrap(),
            &Options::default(),
            &KernelSchedule::default(),
        )
        .unwrap();
        assert_eq!(lowered.grid, [274, 1, 1]);
        assert_eq!(lowered.parameter_types, vec![r#type; 3]);
    }

    #[test]
    fn test_lower_reduction_and_matrix() {
        let context = Context::new();
        let definition = sum::definition(&ArrayType::new_static(DataType::F32, [257])).unwrap();
        let lowered = lower(
            &context,
            &VerifiedKernel::new(&definition, 1024).unwrap(),
            &Options::default(),
            &KernelSchedule::default(),
        )
        .unwrap();
        assert_eq!(lowered.grid, [1, 1, 1]);
        let definition = matrix::definition(
            &ArrayType::new_static(DataType::F32, [33, 35]),
            &ArrayType::new_static(DataType::F32, [35, 34]),
        )
        .unwrap();
        let lowered = lower(
            &context,
            &VerifiedKernel::new(&definition, 1024).unwrap(),
            &Options::default(),
            &KernelSchedule::default(),
        )
        .unwrap();
        assert_eq!(lowered.grid, [4, 1, 1]);
    }

    #[test]
    fn test_lower_condition_write_bound() {
        let context = Context::new();
        let definition = condition_write_definition();
        let lowered = lower(
            &context,
            &VerifiedKernel::new(&definition, 1).unwrap(),
            &Options::default(),
            &KernelSchedule::default(),
        )
        .unwrap();
        assert_eq!(lowered.parameter_types, vec![ArrayType::scalar(DataType::F32)]);
        let function = lowered.module.body().unwrap().operations().unwrap().next().unwrap().unwrap();
        let body = function.region(0).unwrap().blocks().unwrap().next().unwrap().unwrap();
        let loop_operation = body
            .operations()
            .unwrap()
            .map(Result::unwrap)
            .find(|operation| operation.name().as_str() == Ok("scf.while"))
            .unwrap();
        let condition = loop_operation.region(0).unwrap().blocks().unwrap().next().unwrap().unwrap();
        let names = condition
            .operations()
            .unwrap()
            .map(|operation| operation.unwrap().name().as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["arith.constant", "arith.cmpi", "scf.if", "scf.condition"]);
    }

    #[test]
    fn test_lower_sequential_grid() {
        // The sequential axis becomes a loop inside each program of the parallel axis, and the launch grid covers only
        // the two rows. The second window of every row has one padding element.
        let definition = double_rows::definition(&ArrayType::new_static(DataType::F32, [2, 3])).unwrap();
        assert_eq!(
            definition.interpret(vec![Array::matrix(2, 3, vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()], 1),
            Ok(vec![Array::matrix(2, 3, vec![2f32, 4.0, 6.0, 8.0, 10.0, 12.0]).unwrap()]),
        );
        assert_eq!(
            lowered_module(&definition),
            indoc! {"
            module {
              tt.func public @ryft_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
                %0 = tt.get_program_id x : i32
                %1 = arith.extsi %0 : i32 to i64
                %c2_i64 = arith.constant 2 : i64
                %2 = arith.remsi %1, %c2_i64 : i64
                %3 = arith.divsi %1, %c2_i64 : i64
                %true = arith.constant true
                %c0_i64 = arith.constant 0 : i64
                %c2_i64_0 = arith.constant 2 : i64
                %c1_i64 = arith.constant 1 : i64
                scf.for %arg2 = %c0_i64 to %c2_i64_0 step %c1_i64  : i64 {
                  %c1_i64_1 = arith.constant 1 : i64
                  %c2_i64_2 = arith.constant 2 : i64
                  %4 = arith.muli %2, %c1_i64_1 : i64
                  %5 = arith.muli %arg2, %c2_i64_2 : i64
                  %c1_i64_3 = arith.constant 1 : i64
                  %6 = arith.addi %4, %c1_i64_3 : i64
                  %c2_i64_4 = arith.constant 2 : i64
                  %7 = arith.addi %5, %c2_i64_4 : i64
                  %c1_i64_5 = arith.constant 1 : i64
                  %c2_i64_6 = arith.constant 2 : i64
                  %8 = arith.muli %2, %c1_i64_5 : i64
                  %9 = arith.muli %arg2, %c2_i64_6 : i64
                  %c1_i64_7 = arith.constant 1 : i64
                  %10 = arith.addi %8, %c1_i64_7 : i64
                  %c2_i64_8 = arith.constant 2 : i64
                  %11 = arith.addi %9, %c2_i64_8 : i64
                  %true_9 = arith.constant true
                  %12 = tt.splat %true_9 : i1 -> tensor<1x2xi1>
                  %c0_i32 = arith.constant 0 : i32
                  %13 = arith.bitcast %c0_i32 : i32 to f32
                  %14 = tt.splat %13 : f32 -> tensor<1x2xf32>
                  %true_10 = arith.constant true
                  %15 = tt.splat %true_10 : i1 -> tensor<1x2xi1>
                  %16 = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32>
                  %17 = tt.reshape %16 : tensor<1xi32> -> tensor<1x1xi32>
                  %18 = arith.extsi %17 : tensor<1x1xi32> to tensor<1x1xi64>
                  %19 = tt.broadcast %18 : tensor<1x1xi64> -> tensor<1x2xi64>
                  %c1_i64_11 = arith.constant 1 : i64
                  %20 = tt.splat %c1_i64_11 : i64 -> tensor<1x2xi64>
                  %21 = arith.cmpi slt, %19, %20 : tensor<1x2xi64>
                  %22 = arith.andi %15, %21 : tensor<1x2xi1>
                  %23 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                  %24 = tt.reshape %23 : tensor<2xi32> -> tensor<1x2xi32>
                  %25 = arith.extsi %24 : tensor<1x2xi32> to tensor<1x2xi64>
                  %26 = tt.broadcast %25 : tensor<1x2xi64> -> tensor<1x2xi64>
                  %c2_i64_12 = arith.constant 2 : i64
                  %27 = tt.splat %c2_i64_12 : i64 -> tensor<1x2xi64>
                  %28 = arith.cmpi slt, %26, %27 : tensor<1x2xi64>
                  %29 = arith.andi %22, %28 : tensor<1x2xi1>
                  %30 = tt.splat %true : i1 -> tensor<1x2xi1>
                  %31 = arith.andi %29, %30 : tensor<1x2xi1>
                  %c0_i64_13 = arith.constant 0 : i64
                  %32 = tt.splat %c0_i64_13 : i64 -> tensor<1x2xi64>
                  %33 = tt.splat %4 : i64 -> tensor<1x2xi64>
                  %34 = arith.addi %33, %19 : tensor<1x2xi64>
                  %35 = tt.splat %6 : i64 -> tensor<1x2xi64>
                  %36 = arith.cmpi slt, %34, %35 : tensor<1x2xi64>
                  %37 = arith.andi %31, %36 : tensor<1x2xi1>
                  %c2_i64_14 = arith.constant 2 : i64
                  %38 = tt.splat %c2_i64_14 : i64 -> tensor<1x2xi64>
                  %39 = arith.cmpi sge, %34, %32 : tensor<1x2xi64>
                  %40 = arith.cmpi slt, %34, %38 : tensor<1x2xi64>
                  %41 = arith.andi %39, %40 : tensor<1x2xi1>
                  %42 = arith.andi %37, %41 : tensor<1x2xi1>
                  %43 = tt.splat %5 : i64 -> tensor<1x2xi64>
                  %44 = arith.addi %43, %26 : tensor<1x2xi64>
                  %45 = tt.splat %7 : i64 -> tensor<1x2xi64>
                  %46 = arith.cmpi slt, %44, %45 : tensor<1x2xi64>
                  %47 = arith.andi %42, %46 : tensor<1x2xi1>
                  %c3_i64 = arith.constant 3 : i64
                  %48 = tt.splat %c3_i64 : i64 -> tensor<1x2xi64>
                  %49 = arith.cmpi sge, %44, %32 : tensor<1x2xi64>
                  %50 = arith.cmpi slt, %44, %48 : tensor<1x2xi64>
                  %51 = arith.andi %49, %50 : tensor<1x2xi1>
                  %52 = arith.andi %47, %51 : tensor<1x2xi1>
                  %c0_i64_15 = arith.constant 0 : i64
                  %53 = tt.splat %c0_i64_15 : i64 -> tensor<1x2xi64>
                  %54 = arith.select %51, %44, %53 : tensor<1x2xi1>, tensor<1x2xi64>
                  %c1_i64_16 = arith.constant 1 : i64
                  %55 = tt.splat %c1_i64_16 : i64 -> tensor<1x2xi64>
                  %56 = arith.muli %54, %55 : tensor<1x2xi64>
                  %57 = arith.addi %53, %56 : tensor<1x2xi64>
                  %58 = arith.select %41, %34, %53 : tensor<1x2xi1>, tensor<1x2xi64>
                  %c3_i64_17 = arith.constant 3 : i64
                  %59 = tt.splat %c3_i64_17 : i64 -> tensor<1x2xi64>
                  %60 = arith.muli %58, %59 : tensor<1x2xi64>
                  %61 = arith.addi %57, %60 : tensor<1x2xi64>
                  %62 = arith.select %52, %61, %53 : tensor<1x2xi1>, tensor<1x2xi64>
                  %63 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x2x!tt.ptr<f32>>
                  %64 = tt.addptr %63, %62 : tensor<1x2x!tt.ptr<f32>>, tensor<1x2xi64>
                  %65 = arith.andi %52, %12 : tensor<1x2xi1>
                  %66 = tt.load %64, %65, %14 : tensor<1x2x!tt.ptr<f32>>
                  %67 = arith.addf %66, %66 : tensor<1x2xf32>
                  %true_18 = arith.constant true
                  %68 = tt.splat %true_18 : i1 -> tensor<1x2xi1>
                  %69 = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32>
                  %70 = tt.reshape %69 : tensor<1xi32> -> tensor<1x1xi32>
                  %71 = arith.extsi %70 : tensor<1x1xi32> to tensor<1x1xi64>
                  %72 = tt.broadcast %71 : tensor<1x1xi64> -> tensor<1x2xi64>
                  %c1_i64_19 = arith.constant 1 : i64
                  %73 = tt.splat %c1_i64_19 : i64 -> tensor<1x2xi64>
                  %74 = arith.cmpi slt, %72, %73 : tensor<1x2xi64>
                  %75 = arith.andi %68, %74 : tensor<1x2xi1>
                  %76 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                  %77 = tt.reshape %76 : tensor<2xi32> -> tensor<1x2xi32>
                  %78 = arith.extsi %77 : tensor<1x2xi32> to tensor<1x2xi64>
                  %79 = tt.broadcast %78 : tensor<1x2xi64> -> tensor<1x2xi64>
                  %c2_i64_20 = arith.constant 2 : i64
                  %80 = tt.splat %c2_i64_20 : i64 -> tensor<1x2xi64>
                  %81 = arith.cmpi slt, %79, %80 : tensor<1x2xi64>
                  %82 = arith.andi %75, %81 : tensor<1x2xi1>
                  %83 = tt.splat %true : i1 -> tensor<1x2xi1>
                  %84 = arith.andi %82, %83 : tensor<1x2xi1>
                  %c0_i64_21 = arith.constant 0 : i64
                  %85 = tt.splat %c0_i64_21 : i64 -> tensor<1x2xi64>
                  %86 = tt.splat %8 : i64 -> tensor<1x2xi64>
                  %87 = arith.addi %86, %72 : tensor<1x2xi64>
                  %88 = tt.splat %10 : i64 -> tensor<1x2xi64>
                  %89 = arith.cmpi slt, %87, %88 : tensor<1x2xi64>
                  %90 = arith.andi %84, %89 : tensor<1x2xi1>
                  %c2_i64_22 = arith.constant 2 : i64
                  %91 = tt.splat %c2_i64_22 : i64 -> tensor<1x2xi64>
                  %92 = arith.cmpi sge, %87, %85 : tensor<1x2xi64>
                  %93 = arith.cmpi slt, %87, %91 : tensor<1x2xi64>
                  %94 = arith.andi %92, %93 : tensor<1x2xi1>
                  %95 = arith.andi %90, %94 : tensor<1x2xi1>
                  %96 = tt.splat %9 : i64 -> tensor<1x2xi64>
                  %97 = arith.addi %96, %79 : tensor<1x2xi64>
                  %98 = tt.splat %11 : i64 -> tensor<1x2xi64>
                  %99 = arith.cmpi slt, %97, %98 : tensor<1x2xi64>
                  %100 = arith.andi %95, %99 : tensor<1x2xi1>
                  %c3_i64_23 = arith.constant 3 : i64
                  %101 = tt.splat %c3_i64_23 : i64 -> tensor<1x2xi64>
                  %102 = arith.cmpi sge, %97, %85 : tensor<1x2xi64>
                  %103 = arith.cmpi slt, %97, %101 : tensor<1x2xi64>
                  %104 = arith.andi %102, %103 : tensor<1x2xi1>
                  %105 = arith.andi %100, %104 : tensor<1x2xi1>
                  %c0_i64_24 = arith.constant 0 : i64
                  %106 = tt.splat %c0_i64_24 : i64 -> tensor<1x2xi64>
                  %107 = arith.select %104, %97, %106 : tensor<1x2xi1>, tensor<1x2xi64>
                  %c1_i64_25 = arith.constant 1 : i64
                  %108 = tt.splat %c1_i64_25 : i64 -> tensor<1x2xi64>
                  %109 = arith.muli %107, %108 : tensor<1x2xi64>
                  %110 = arith.addi %106, %109 : tensor<1x2xi64>
                  %111 = arith.select %94, %87, %106 : tensor<1x2xi1>, tensor<1x2xi64>
                  %c3_i64_26 = arith.constant 3 : i64
                  %112 = tt.splat %c3_i64_26 : i64 -> tensor<1x2xi64>
                  %113 = arith.muli %111, %112 : tensor<1x2xi64>
                  %114 = arith.addi %110, %113 : tensor<1x2xi64>
                  %115 = arith.select %105, %114, %106 : tensor<1x2xi1>, tensor<1x2xi64>
                  %116 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x2x!tt.ptr<f32>>
                  %117 = tt.addptr %116, %115 : tensor<1x2x!tt.ptr<f32>>, tensor<1x2xi64>
                  tt.store %117, %67, %105 {operand_segment_sizes = array<i32: 1, 1, 1>} : tensor<1x2x!tt.ptr<f32>>
                }
                tt.return
              }
            }
        "},
        );
    }

    #[test]
    fn test_lower_read_write_parameter() {
        // The accumulator is one pointer that is loaded and then stored, and its loaded value reaches memory only
        // through the addition that its store writes.
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let definition = add_in_place::definition(&r#type, &r#type).unwrap();
        assert_eq!(
            definition.interpret(
                vec![Array::vector(vec![1f32, 2.0]).unwrap(), Array::vector(vec![10f32, 20.0]).unwrap()],
                1,
            ),
            Ok(vec![Array::vector(vec![11f32, 22.0]).unwrap()]),
        );
        assert_eq!(
            lowered_module(&definition),
            indoc! {"
            module {
              tt.func public @ryft_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
                %0 = tt.get_program_id x : i32
                %1 = arith.extsi %0 : i32 to i64
                %true = arith.constant true
                %c0_i64 = arith.constant 0 : i64
                %c2_i64 = arith.constant 2 : i64
                %2 = arith.addi %c0_i64, %c2_i64 : i64
                %c0_i64_0 = arith.constant 0 : i64
                %c2_i64_1 = arith.constant 2 : i64
                %3 = arith.addi %c0_i64_0, %c2_i64_1 : i64
                %cst = arith.constant 0.000000e+00 : f32
                %4 = tt.splat %cst : f32 -> tensor<2xf32>
                %true_2 = arith.constant true
                %5 = tt.splat %true_2 : i1 -> tensor<2xi1>
                %6 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                %7 = tt.reshape %6 : tensor<2xi32> -> tensor<2xi32>
                %8 = arith.extsi %7 : tensor<2xi32> to tensor<2xi64>
                %9 = tt.broadcast %8 : tensor<2xi64> -> tensor<2xi64>
                %c2_i64_3 = arith.constant 2 : i64
                %10 = tt.splat %c2_i64_3 : i64 -> tensor<2xi64>
                %11 = arith.cmpi slt, %9, %10 : tensor<2xi64>
                %12 = arith.andi %5, %11 : tensor<2xi1>
                %13 = tt.splat %true : i1 -> tensor<2xi1>
                %14 = arith.andi %12, %13 : tensor<2xi1>
                %c0_i64_4 = arith.constant 0 : i64
                %15 = tt.splat %c0_i64_4 : i64 -> tensor<2xi64>
                %16 = tt.splat %c0_i64_0 : i64 -> tensor<2xi64>
                %17 = arith.addi %16, %9 : tensor<2xi64>
                %18 = tt.splat %3 : i64 -> tensor<2xi64>
                %19 = arith.cmpi slt, %17, %18 : tensor<2xi64>
                %20 = arith.andi %14, %19 : tensor<2xi1>
                %c2_i64_5 = arith.constant 2 : i64
                %21 = tt.splat %c2_i64_5 : i64 -> tensor<2xi64>
                %22 = arith.cmpi sge, %17, %15 : tensor<2xi64>
                %23 = arith.cmpi slt, %17, %21 : tensor<2xi64>
                %24 = arith.andi %22, %23 : tensor<2xi1>
                %25 = arith.andi %20, %24 : tensor<2xi1>
                %c0_i64_6 = arith.constant 0 : i64
                %26 = tt.splat %c0_i64_6 : i64 -> tensor<2xi64>
                %27 = arith.select %24, %17, %26 : tensor<2xi1>, tensor<2xi64>
                %c1_i64 = arith.constant 1 : i64
                %28 = tt.splat %c1_i64 : i64 -> tensor<2xi64>
                %29 = arith.muli %27, %28 : tensor<2xi64>
                %30 = arith.addi %26, %29 : tensor<2xi64>
                %31 = arith.select %25, %30, %26 : tensor<2xi1>, tensor<2xi64>
                %32 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
                %33 = tt.addptr %32, %31 : tensor<2x!tt.ptr<f32>>, tensor<2xi64>
                %34 = tt.load %33, %25, %4 : tensor<2x!tt.ptr<f32>>
                %cst_7 = arith.constant 0.000000e+00 : f32
                %35 = tt.splat %cst_7 : f32 -> tensor<2xf32>
                %true_8 = arith.constant true
                %36 = tt.splat %true_8 : i1 -> tensor<2xi1>
                %37 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                %38 = tt.reshape %37 : tensor<2xi32> -> tensor<2xi32>
                %39 = arith.extsi %38 : tensor<2xi32> to tensor<2xi64>
                %40 = tt.broadcast %39 : tensor<2xi64> -> tensor<2xi64>
                %c2_i64_9 = arith.constant 2 : i64
                %41 = tt.splat %c2_i64_9 : i64 -> tensor<2xi64>
                %42 = arith.cmpi slt, %40, %41 : tensor<2xi64>
                %43 = arith.andi %36, %42 : tensor<2xi1>
                %44 = tt.splat %true : i1 -> tensor<2xi1>
                %45 = arith.andi %43, %44 : tensor<2xi1>
                %c0_i64_10 = arith.constant 0 : i64
                %46 = tt.splat %c0_i64_10 : i64 -> tensor<2xi64>
                %47 = tt.splat %c0_i64 : i64 -> tensor<2xi64>
                %48 = arith.addi %47, %40 : tensor<2xi64>
                %49 = tt.splat %2 : i64 -> tensor<2xi64>
                %50 = arith.cmpi slt, %48, %49 : tensor<2xi64>
                %51 = arith.andi %45, %50 : tensor<2xi1>
                %c2_i64_11 = arith.constant 2 : i64
                %52 = tt.splat %c2_i64_11 : i64 -> tensor<2xi64>
                %53 = arith.cmpi sge, %48, %46 : tensor<2xi64>
                %54 = arith.cmpi slt, %48, %52 : tensor<2xi64>
                %55 = arith.andi %53, %54 : tensor<2xi1>
                %56 = arith.andi %51, %55 : tensor<2xi1>
                %c0_i64_12 = arith.constant 0 : i64
                %57 = tt.splat %c0_i64_12 : i64 -> tensor<2xi64>
                %58 = arith.select %55, %48, %57 : tensor<2xi1>, tensor<2xi64>
                %c1_i64_13 = arith.constant 1 : i64
                %59 = tt.splat %c1_i64_13 : i64 -> tensor<2xi64>
                %60 = arith.muli %58, %59 : tensor<2xi64>
                %61 = arith.addi %57, %60 : tensor<2xi64>
                %62 = arith.select %56, %61, %57 : tensor<2xi1>, tensor<2xi64>
                %63 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
                %64 = tt.addptr %63, %62 : tensor<2x!tt.ptr<f32>>, tensor<2xi64>
                %65 = tt.load %64, %56, %35 : tensor<2x!tt.ptr<f32>>
                %66 = arith.addf %34, %65 : tensor<2xf32>
                %true_14 = arith.constant true
                %67 = tt.splat %true_14 : i1 -> tensor<2xi1>
                %68 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                %69 = tt.reshape %68 : tensor<2xi32> -> tensor<2xi32>
                %70 = arith.extsi %69 : tensor<2xi32> to tensor<2xi64>
                %71 = tt.broadcast %70 : tensor<2xi64> -> tensor<2xi64>
                %c2_i64_15 = arith.constant 2 : i64
                %72 = tt.splat %c2_i64_15 : i64 -> tensor<2xi64>
                %73 = arith.cmpi slt, %71, %72 : tensor<2xi64>
                %74 = arith.andi %67, %73 : tensor<2xi1>
                %75 = tt.splat %true : i1 -> tensor<2xi1>
                %76 = arith.andi %74, %75 : tensor<2xi1>
                %c0_i64_16 = arith.constant 0 : i64
                %77 = tt.splat %c0_i64_16 : i64 -> tensor<2xi64>
                %78 = tt.splat %c0_i64_0 : i64 -> tensor<2xi64>
                %79 = arith.addi %78, %71 : tensor<2xi64>
                %80 = tt.splat %3 : i64 -> tensor<2xi64>
                %81 = arith.cmpi slt, %79, %80 : tensor<2xi64>
                %82 = arith.andi %76, %81 : tensor<2xi1>
                %c2_i64_17 = arith.constant 2 : i64
                %83 = tt.splat %c2_i64_17 : i64 -> tensor<2xi64>
                %84 = arith.cmpi sge, %79, %77 : tensor<2xi64>
                %85 = arith.cmpi slt, %79, %83 : tensor<2xi64>
                %86 = arith.andi %84, %85 : tensor<2xi1>
                %87 = arith.andi %82, %86 : tensor<2xi1>
                %c0_i64_18 = arith.constant 0 : i64
                %88 = tt.splat %c0_i64_18 : i64 -> tensor<2xi64>
                %89 = arith.select %86, %79, %88 : tensor<2xi1>, tensor<2xi64>
                %c1_i64_19 = arith.constant 1 : i64
                %90 = tt.splat %c1_i64_19 : i64 -> tensor<2xi64>
                %91 = arith.muli %89, %90 : tensor<2xi64>
                %92 = arith.addi %88, %91 : tensor<2xi64>
                %93 = arith.select %87, %92, %88 : tensor<2xi1>, tensor<2xi64>
                %94 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
                %95 = tt.addptr %94, %93 : tensor<2x!tt.ptr<f32>>, tensor<2xi64>
                tt.store %95, %66, %87 {operand_segment_sizes = array<i32: 1, 1, 1>} : tensor<2x!tt.ptr<f32>>
                tt.return
              }
            }
        "},
        );
    }

    #[test]
    fn test_lower_add_update() {
        // The add update loads and stores its window through the same pointers under the same validity mask.
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let definition = accumulate_in_place::definition(&r#type, &r#type).unwrap();
        assert_eq!(
            definition.interpret(
                vec![Array::vector(vec![1f32, 2.0]).unwrap(), Array::vector(vec![10f32, 20.0]).unwrap()],
                1,
            ),
            Ok(vec![Array::vector(vec![11f32, 22.0]).unwrap()]),
        );
        assert_eq!(
            lowered_module(&definition),
            indoc! {"
            module {
              tt.func public @ryft_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
                %0 = tt.get_program_id x : i32
                %1 = arith.extsi %0 : i32 to i64
                %true = arith.constant true
                %c0_i64 = arith.constant 0 : i64
                %c2_i64 = arith.constant 2 : i64
                %2 = arith.addi %c0_i64, %c2_i64 : i64
                %c0_i64_0 = arith.constant 0 : i64
                %c2_i64_1 = arith.constant 2 : i64
                %3 = arith.addi %c0_i64_0, %c2_i64_1 : i64
                %cst = arith.constant 0.000000e+00 : f32
                %4 = tt.splat %cst : f32 -> tensor<2xf32>
                %true_2 = arith.constant true
                %5 = tt.splat %true_2 : i1 -> tensor<2xi1>
                %6 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                %7 = tt.reshape %6 : tensor<2xi32> -> tensor<2xi32>
                %8 = arith.extsi %7 : tensor<2xi32> to tensor<2xi64>
                %9 = tt.broadcast %8 : tensor<2xi64> -> tensor<2xi64>
                %c2_i64_3 = arith.constant 2 : i64
                %10 = tt.splat %c2_i64_3 : i64 -> tensor<2xi64>
                %11 = arith.cmpi slt, %9, %10 : tensor<2xi64>
                %12 = arith.andi %5, %11 : tensor<2xi1>
                %13 = tt.splat %true : i1 -> tensor<2xi1>
                %14 = arith.andi %12, %13 : tensor<2xi1>
                %c0_i64_4 = arith.constant 0 : i64
                %15 = tt.splat %c0_i64_4 : i64 -> tensor<2xi64>
                %16 = tt.splat %c0_i64 : i64 -> tensor<2xi64>
                %17 = arith.addi %16, %9 : tensor<2xi64>
                %18 = tt.splat %2 : i64 -> tensor<2xi64>
                %19 = arith.cmpi slt, %17, %18 : tensor<2xi64>
                %20 = arith.andi %14, %19 : tensor<2xi1>
                %c2_i64_5 = arith.constant 2 : i64
                %21 = tt.splat %c2_i64_5 : i64 -> tensor<2xi64>
                %22 = arith.cmpi sge, %17, %15 : tensor<2xi64>
                %23 = arith.cmpi slt, %17, %21 : tensor<2xi64>
                %24 = arith.andi %22, %23 : tensor<2xi1>
                %25 = arith.andi %20, %24 : tensor<2xi1>
                %c0_i64_6 = arith.constant 0 : i64
                %26 = tt.splat %c0_i64_6 : i64 -> tensor<2xi64>
                %27 = arith.select %24, %17, %26 : tensor<2xi1>, tensor<2xi64>
                %c1_i64 = arith.constant 1 : i64
                %28 = tt.splat %c1_i64 : i64 -> tensor<2xi64>
                %29 = arith.muli %27, %28 : tensor<2xi64>
                %30 = arith.addi %26, %29 : tensor<2xi64>
                %31 = arith.select %25, %30, %26 : tensor<2xi1>, tensor<2xi64>
                %32 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
                %33 = tt.addptr %32, %31 : tensor<2x!tt.ptr<f32>>, tensor<2xi64>
                %34 = tt.load %33, %25, %4 : tensor<2x!tt.ptr<f32>>
                %true_7 = arith.constant true
                %35 = tt.splat %true_7 : i1 -> tensor<2xi1>
                %36 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                %37 = tt.reshape %36 : tensor<2xi32> -> tensor<2xi32>
                %38 = arith.extsi %37 : tensor<2xi32> to tensor<2xi64>
                %39 = tt.broadcast %38 : tensor<2xi64> -> tensor<2xi64>
                %c2_i64_8 = arith.constant 2 : i64
                %40 = tt.splat %c2_i64_8 : i64 -> tensor<2xi64>
                %41 = arith.cmpi slt, %39, %40 : tensor<2xi64>
                %42 = arith.andi %35, %41 : tensor<2xi1>
                %43 = tt.splat %true : i1 -> tensor<2xi1>
                %44 = arith.andi %42, %43 : tensor<2xi1>
                %c0_i64_9 = arith.constant 0 : i64
                %45 = tt.splat %c0_i64_9 : i64 -> tensor<2xi64>
                %46 = tt.splat %c0_i64_0 : i64 -> tensor<2xi64>
                %47 = arith.addi %46, %39 : tensor<2xi64>
                %48 = tt.splat %3 : i64 -> tensor<2xi64>
                %49 = arith.cmpi slt, %47, %48 : tensor<2xi64>
                %50 = arith.andi %44, %49 : tensor<2xi1>
                %c2_i64_10 = arith.constant 2 : i64
                %51 = tt.splat %c2_i64_10 : i64 -> tensor<2xi64>
                %52 = arith.cmpi sge, %47, %45 : tensor<2xi64>
                %53 = arith.cmpi slt, %47, %51 : tensor<2xi64>
                %54 = arith.andi %52, %53 : tensor<2xi1>
                %55 = arith.andi %50, %54 : tensor<2xi1>
                %c0_i64_11 = arith.constant 0 : i64
                %56 = tt.splat %c0_i64_11 : i64 -> tensor<2xi64>
                %57 = arith.select %54, %47, %56 : tensor<2xi1>, tensor<2xi64>
                %c1_i64_12 = arith.constant 1 : i64
                %58 = tt.splat %c1_i64_12 : i64 -> tensor<2xi64>
                %59 = arith.muli %57, %58 : tensor<2xi64>
                %60 = arith.addi %56, %59 : tensor<2xi64>
                %61 = arith.select %55, %60, %56 : tensor<2xi1>, tensor<2xi64>
                %62 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
                %63 = tt.addptr %62, %61 : tensor<2x!tt.ptr<f32>>, tensor<2xi64>
                %cst_13 = arith.constant 0.000000e+00 : f32
                %64 = tt.splat %cst_13 : f32 -> tensor<2xf32>
                %65 = tt.load %63, %55, %64 : tensor<2x!tt.ptr<f32>>
                %66 = arith.addf %65, %34 : tensor<2xf32>
                tt.store %63, %66, %55 {operand_segment_sizes = array<i32: 1, 1, 1>} : tensor<2x!tt.ptr<f32>>
                tt.return
              }
            }
        "},
        );
    }

    #[test]
    fn test_lower_atomic_add_update() {
        // Every program adds the sum of its window to the scalar total with one masked atomic addition.
        let definition =
            parallel_sum::definition(&ArrayType::new_static(DataType::F32, [4]), &ArrayType::scalar(DataType::F32))
                .unwrap();
        assert_eq!(
            definition.interpret(
                vec![Array::vector(vec![1f32, 2.0, 3.0, 4.0]).unwrap(), Array::scalar(10f32).unwrap()],
                1,
            ),
            Ok(vec![Array::scalar(20f32).unwrap()]),
        );
        assert_eq!(
            lowered_module(&definition),
            indoc! {"
            module {
              tt.func public @ryft_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
                %0 = tt.get_program_id x : i32
                %1 = arith.extsi %0 : i32 to i64
                %c2_i64 = arith.constant 2 : i64
                %2 = arith.remsi %1, %c2_i64 : i64
                %3 = arith.divsi %1, %c2_i64 : i64
                %true = arith.constant true
                %c2_i64_0 = arith.constant 2 : i64
                %4 = arith.muli %2, %c2_i64_0 : i64
                %c2_i64_1 = arith.constant 2 : i64
                %5 = arith.addi %4, %c2_i64_1 : i64
                %cst = arith.constant 0.000000e+00 : f32
                %6 = tt.splat %cst : f32 -> tensor<2xf32>
                %true_2 = arith.constant true
                %7 = tt.splat %true_2 : i1 -> tensor<2xi1>
                %8 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                %9 = tt.reshape %8 : tensor<2xi32> -> tensor<2xi32>
                %10 = arith.extsi %9 : tensor<2xi32> to tensor<2xi64>
                %11 = tt.broadcast %10 : tensor<2xi64> -> tensor<2xi64>
                %c2_i64_3 = arith.constant 2 : i64
                %12 = tt.splat %c2_i64_3 : i64 -> tensor<2xi64>
                %13 = arith.cmpi slt, %11, %12 : tensor<2xi64>
                %14 = arith.andi %7, %13 : tensor<2xi1>
                %15 = tt.splat %true : i1 -> tensor<2xi1>
                %16 = arith.andi %14, %15 : tensor<2xi1>
                %c0_i64 = arith.constant 0 : i64
                %17 = tt.splat %c0_i64 : i64 -> tensor<2xi64>
                %18 = tt.splat %4 : i64 -> tensor<2xi64>
                %19 = arith.addi %18, %11 : tensor<2xi64>
                %20 = tt.splat %5 : i64 -> tensor<2xi64>
                %21 = arith.cmpi slt, %19, %20 : tensor<2xi64>
                %22 = arith.andi %16, %21 : tensor<2xi1>
                %c4_i64 = arith.constant 4 : i64
                %23 = tt.splat %c4_i64 : i64 -> tensor<2xi64>
                %24 = arith.cmpi sge, %19, %17 : tensor<2xi64>
                %25 = arith.cmpi slt, %19, %23 : tensor<2xi64>
                %26 = arith.andi %24, %25 : tensor<2xi1>
                %27 = arith.andi %22, %26 : tensor<2xi1>
                %c0_i64_4 = arith.constant 0 : i64
                %28 = tt.splat %c0_i64_4 : i64 -> tensor<2xi64>
                %29 = arith.select %26, %19, %28 : tensor<2xi1>, tensor<2xi64>
                %c1_i64 = arith.constant 1 : i64
                %30 = tt.splat %c1_i64 : i64 -> tensor<2xi64>
                %31 = arith.muli %29, %30 : tensor<2xi64>
                %32 = arith.addi %28, %31 : tensor<2xi64>
                %33 = arith.select %27, %32, %28 : tensor<2xi1>, tensor<2xi64>
                %34 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
                %35 = tt.addptr %34, %33 : tensor<2x!tt.ptr<f32>>, tensor<2xi64>
                %36 = tt.load %35, %27, %6 : tensor<2x!tt.ptr<f32>>
                %true_5 = arith.constant true
                %37 = tt.splat %true_5 : i1 -> tensor<2xi1>
                %38 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                %39 = tt.reshape %38 : tensor<2xi32> -> tensor<2xi32>
                %40 = arith.extsi %39 : tensor<2xi32> to tensor<2xi64>
                %41 = tt.broadcast %40 : tensor<2xi64> -> tensor<2xi64>
                %c2_i64_6 = arith.constant 2 : i64
                %42 = tt.splat %c2_i64_6 : i64 -> tensor<2xi64>
                %43 = arith.cmpi slt, %41, %42 : tensor<2xi64>
                %44 = arith.andi %37, %43 : tensor<2xi1>
                %cst_7 = arith.constant 0.000000e+00 : f32
                %45 = tt.splat %cst_7 : f32 -> tensor<2xf32>
                %46 = arith.select %44, %36, %45 : tensor<2xi1>, tensor<2xf32>
                %47 = \"tt.reduce\"(%46) <{axis = 0 : i32}> ({
                ^bb0(%arg2: f32, %arg3: f32):
                  %52 = arith.addf %arg2, %arg3 : f32
                  tt.reduce.return %52 : f32
                }) : (tensor<2xf32>) -> f32
                %true_8 = arith.constant true
                %48 = arith.andi %true_8, %true : i1
                %c0_i64_9 = arith.constant 0 : i64
                %c0_i64_10 = arith.constant 0 : i64
                %49 = arith.select %48, %c0_i64_10, %c0_i64_10 : i64
                %50 = tt.addptr %arg1, %49 : !tt.ptr<f32>, i64
                %51 = \
                    tt.atomic_rmw fadd, acq_rel, gpu, %50, %47, %48 \
                    {operand_segment_sizes = array<i32: 1, 1, 1>} : (!tt.ptr<f32>, f32, i1) -> f32
                tt.return
              }
            }
        "},
        );
    }

    #[test]
    fn test_lower_rejects_reads_of_write_only_parameters() {
        // The kernel verifier admits a read of a write-only parameter that follows a definite write, but another thread
        // of the program may load the element before the store lands, so admission rejects it.
        let r#type = ArrayType::scalar(DataType::F32);
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(r#type, KernelParameterAccess::WriteOnly).unwrap()],
        )
        .unwrap();
        let mut body = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = body.add_input(call.parameters()[0].body_type());
        let two = body.add_constant(ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()));
        body.add_instruction(ReferenceWriteOperation::new(), vec![], vec![reference, two], None).unwrap();
        body.add_instruction(ReferenceReadOperation::new(), vec![], vec![reference], None).unwrap();
        let body = body
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder], vec![])
            .unwrap();
        let definition = KernelDefinition::new(call, body).unwrap();
        assert_eq!(
            rejection(&definition),
            (
                "kernel_call",
                "reads after writes of one parameter need a CTA barrier, which ryft-mlir does not expose".to_owned(),
            ),
        );
    }

    #[test]
    fn test_lower_rejects_carried_accumulation() {
        // The first step of the sequential axis initializes the carried output, and every step then reads it back
        // through an add update over a masked input window.
        let definition = carried_sum::definition(&ArrayType::new_static(DataType::F32, [3])).unwrap();
        assert_eq!(
            definition.interpret(vec![Array::vector(vec![1f32, 2.0, 3.0]).unwrap()], 1),
            Ok(vec![Array::vector(vec![4f32, 2.0]).unwrap()]),
        );
        assert_eq!(
            rejection(&definition),
            (
                "kernel_call",
                "reads after writes of one parameter need a CTA barrier, which ryft-mlir does not expose".to_owned(),
            ),
        );
    }

    #[test]
    fn test_lower_rejects_carried_windows() {
        // Every step stores into the same window, so the stores of two steps could land in either order.
        let definition = carried_copy::definition(&ArrayType::new_static(DataType::F32, [4])).unwrap();
        assert_eq!(
            rejection(&definition),
            (
                "kernel_call",
                "windows that repeat along a sequential grid axis need a CTA barrier, which ryft-mlir does not expose"
                    .to_owned(),
            ),
        );
    }

    #[test]
    fn test_lower_rejects_reads_after_writes() {
        let definition = read_after_write::definition(&ArrayType::new_static(DataType::F32, [2])).unwrap();
        assert_eq!(
            rejection(&definition),
            (
                "reference_read",
                "reads after writes of one parameter need a CTA barrier, which ryft-mlir does not expose".to_owned(),
            ),
        );
    }

    #[test]
    fn test_lower_rejects_repeated_writes() {
        let definition = write_twice::definition(&ArrayType::new_static(DataType::F32, [2])).unwrap();
        assert_eq!(
            rejection(&definition),
            (
                "reference_write",
                "repeated writes of one window need a CTA barrier, which ryft-mlir does not expose".to_owned(),
            ),
        );
    }

    #[test]
    fn test_lower_rejects_partial_read_write_windows() {
        let definition = update_first::definition(&ArrayType::new_static(DataType::F32, [2])).unwrap();
        assert_eq!(
            rejection(&definition),
            ("reference_read", "read-write parameters must be accessed through their whole window".to_owned()),
        );
    }

    #[test]
    fn test_lower_rejects_read_write_values_reaching_other_writes() {
        // Another thread may load an element of `values` after the store of `values` overwrites it and copy that new
        // value into `output`.
        let definition = copy_then_update::definition(&ArrayType::new_static(DataType::F32, [2])).unwrap();
        assert_eq!(
            rejection(&definition),
            (
                "reference_write",
                "values read from a read-write parameter may reach memory only elementwise through its own write"
                    .to_owned(),
            ),
        );
    }

    #[test]
    fn test_lower_rejects_read_write_accesses_in_control_flow() {
        let definition = double_repeatedly::definition(&ArrayType::new_static(DataType::F32, [2])).unwrap();
        assert_eq!(
            rejection(&definition),
            (
                "while",
                "accesses of a mutable window inside control flow need a CTA barrier, which ryft-mlir does not expose"
                    .to_owned(),
            ),
        );
    }

    #[test]
    fn test_lower_rejects_mixed_atomic_accesses() {
        let definition =
            read_atomic_total::definition(&ArrayType::new_static(DataType::F32, [2]), &ArrayType::scalar(DataType::F32))
                .unwrap();
        assert_eq!(
            rejection(&definition),
            (
                "kernel_call",
                "atomic and other accesses of one parameter need a CTA barrier, which ryft-mlir does not expose"
                    .to_owned(),
            ),
        );
    }

    #[test]
    fn test_lower_masked_load() {
        // The load mask intersects the window validity with the explicit mask, so the padding element and the physical
        // padding take the value `other`.
        let operation = tiled_call(vec![
            (ArrayType::new_static(DataType::F32, [2]), KernelParameterAccess::ReadOnly, BoundaryPolicy::Masked),
            (ArrayType::new_static(DataType::F32, [3]), KernelParameterAccess::WriteOnly, BoundaryPolicy::InBounds),
        ]);
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            references[1].write(&masked_load_or(references[0].context(), &references[0], -1f32)?)
        })
        .unwrap();
        assert_eq!(
            definition.interpret(vec![Array::vector(vec![1f32, 2.0]).unwrap()], 1),
            Ok(vec![Array::vector(vec![1f32, 2.0, -1.0]).unwrap()]),
        );
        assert_eq!(
            lowered_module(&definition),
            indoc! {"
            module {
              tt.func public @ryft_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
                %0 = tt.get_program_id x : i32
                %1 = arith.extsi %0 : i32 to i64
                %c1_i64 = arith.constant 1 : i64
                %2 = arith.remsi %1, %c1_i64 : i64
                %3 = arith.divsi %1, %c1_i64 : i64
                %true = arith.constant true
                %c3_i64 = arith.constant 3 : i64
                %4 = arith.muli %2, %c3_i64 : i64
                %c3_i64_0 = arith.constant 3 : i64
                %5 = arith.addi %4, %c3_i64_0 : i64
                %c3_i64_1 = arith.constant 3 : i64
                %6 = arith.muli %2, %c3_i64_1 : i64
                %c3_i64_2 = arith.constant 3 : i64
                %7 = arith.addi %6, %c3_i64_2 : i64
                %true_3 = arith.constant true
                %8 = tt.splat %true_3 : i1 -> tensor<4xi1>
                %c-1082130432_i32 = arith.constant -1082130432 : i32
                %9 = arith.bitcast %c-1082130432_i32 : i32 to f32
                %10 = tt.splat %9 : f32 -> tensor<4xf32>
                %true_4 = arith.constant true
                %11 = tt.splat %true_4 : i1 -> tensor<4xi1>
                %12 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
                %13 = tt.reshape %12 : tensor<4xi32> -> tensor<4xi32>
                %14 = arith.extsi %13 : tensor<4xi32> to tensor<4xi64>
                %15 = tt.broadcast %14 : tensor<4xi64> -> tensor<4xi64>
                %c3_i64_5 = arith.constant 3 : i64
                %16 = tt.splat %c3_i64_5 : i64 -> tensor<4xi64>
                %17 = arith.cmpi slt, %15, %16 : tensor<4xi64>
                %18 = arith.andi %11, %17 : tensor<4xi1>
                %19 = tt.splat %true : i1 -> tensor<4xi1>
                %20 = arith.andi %18, %19 : tensor<4xi1>
                %c0_i64 = arith.constant 0 : i64
                %21 = tt.splat %c0_i64 : i64 -> tensor<4xi64>
                %22 = tt.splat %4 : i64 -> tensor<4xi64>
                %23 = arith.addi %22, %15 : tensor<4xi64>
                %24 = tt.splat %5 : i64 -> tensor<4xi64>
                %25 = arith.cmpi slt, %23, %24 : tensor<4xi64>
                %26 = arith.andi %20, %25 : tensor<4xi1>
                %c2_i64 = arith.constant 2 : i64
                %27 = tt.splat %c2_i64 : i64 -> tensor<4xi64>
                %28 = arith.cmpi sge, %23, %21 : tensor<4xi64>
                %29 = arith.cmpi slt, %23, %27 : tensor<4xi64>
                %30 = arith.andi %28, %29 : tensor<4xi1>
                %31 = arith.andi %26, %30 : tensor<4xi1>
                %c0_i64_6 = arith.constant 0 : i64
                %32 = tt.splat %c0_i64_6 : i64 -> tensor<4xi64>
                %33 = arith.select %30, %23, %32 : tensor<4xi1>, tensor<4xi64>
                %c1_i64_7 = arith.constant 1 : i64
                %34 = tt.splat %c1_i64_7 : i64 -> tensor<4xi64>
                %35 = arith.muli %33, %34 : tensor<4xi64>
                %36 = arith.addi %32, %35 : tensor<4xi64>
                %37 = arith.select %31, %36, %32 : tensor<4xi1>, tensor<4xi64>
                %38 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
                %39 = tt.addptr %38, %37 : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
                %40 = arith.andi %31, %8 : tensor<4xi1>
                %41 = tt.load %39, %40, %10 : tensor<4x!tt.ptr<f32>>
                %true_8 = arith.constant true
                %42 = tt.splat %true_8 : i1 -> tensor<4xi1>
                %43 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
                %44 = tt.reshape %43 : tensor<4xi32> -> tensor<4xi32>
                %45 = arith.extsi %44 : tensor<4xi32> to tensor<4xi64>
                %46 = tt.broadcast %45 : tensor<4xi64> -> tensor<4xi64>
                %c3_i64_9 = arith.constant 3 : i64
                %47 = tt.splat %c3_i64_9 : i64 -> tensor<4xi64>
                %48 = arith.cmpi slt, %46, %47 : tensor<4xi64>
                %49 = arith.andi %42, %48 : tensor<4xi1>
                %50 = tt.splat %true : i1 -> tensor<4xi1>
                %51 = arith.andi %49, %50 : tensor<4xi1>
                %c0_i64_10 = arith.constant 0 : i64
                %52 = tt.splat %c0_i64_10 : i64 -> tensor<4xi64>
                %53 = tt.splat %6 : i64 -> tensor<4xi64>
                %54 = arith.addi %53, %46 : tensor<4xi64>
                %55 = tt.splat %7 : i64 -> tensor<4xi64>
                %56 = arith.cmpi slt, %54, %55 : tensor<4xi64>
                %57 = arith.andi %51, %56 : tensor<4xi1>
                %c3_i64_11 = arith.constant 3 : i64
                %58 = tt.splat %c3_i64_11 : i64 -> tensor<4xi64>
                %59 = arith.cmpi sge, %54, %52 : tensor<4xi64>
                %60 = arith.cmpi slt, %54, %58 : tensor<4xi64>
                %61 = arith.andi %59, %60 : tensor<4xi1>
                %62 = arith.andi %57, %61 : tensor<4xi1>
                %c0_i64_12 = arith.constant 0 : i64
                %63 = tt.splat %c0_i64_12 : i64 -> tensor<4xi64>
                %64 = arith.select %61, %54, %63 : tensor<4xi1>, tensor<4xi64>
                %c1_i64_13 = arith.constant 1 : i64
                %65 = tt.splat %c1_i64_13 : i64 -> tensor<4xi64>
                %66 = arith.muli %64, %65 : tensor<4xi64>
                %67 = arith.addi %63, %66 : tensor<4xi64>
                %68 = arith.select %62, %67, %63 : tensor<4xi1>, tensor<4xi64>
                %69 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
                %70 = tt.addptr %69, %68 : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
                tt.store %70, %41, %62 {operand_segment_sizes = array<i32: 1, 1, 1>} : tensor<4x!tt.ptr<f32>>
                tt.return
              }
            }
        "},
        );
    }

    #[test]
    fn test_lower_masked_store() {
        let operation = tiled_call(vec![(
            ArrayType::new_static(DataType::F32, [2]),
            KernelParameterAccess::WriteOnly,
            BoundaryPolicy::Masked,
        )]);
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            let context = references[0].context();
            let value = context.lift(ArrayIrValue::Array(Array::vector(vec![5f32; 3])?))?;
            let mask = context.lift(ArrayIrValue::Array(Array::vector(vec![true; 3])?))?;
            masked_store(context, &references[0], &value, &mask)
        })
        .unwrap();
        assert_eq!(definition.interpret(vec![], 1), Ok(vec![Array::vector(vec![5f32, 5.0]).unwrap()]));
        assert_eq!(
            lowered_module(&definition),
            indoc! {"
            module {
              tt.func public @ryft_kernel(%arg0: !tt.ptr<f32>) {
                %0 = tt.get_program_id x : i32
                %1 = arith.extsi %0 : i32 to i64
                %c1_i64 = arith.constant 1 : i64
                %2 = arith.remsi %1, %c1_i64 : i64
                %3 = arith.divsi %1, %c1_i64 : i64
                %true = arith.constant true
                %c3_i64 = arith.constant 3 : i64
                %4 = arith.muli %2, %c3_i64 : i64
                %c3_i64_0 = arith.constant 3 : i64
                %5 = arith.addi %4, %c3_i64_0 : i64
                %c1084227584_i32 = arith.constant 1084227584 : i32
                %6 = arith.bitcast %c1084227584_i32 : i32 to f32
                %7 = tt.splat %6 : f32 -> tensor<4xf32>
                %true_1 = arith.constant true
                %8 = tt.splat %true_1 : i1 -> tensor<4xi1>
                %true_2 = arith.constant true
                %9 = tt.splat %true_2 : i1 -> tensor<4xi1>
                %10 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
                %11 = tt.reshape %10 : tensor<4xi32> -> tensor<4xi32>
                %12 = arith.extsi %11 : tensor<4xi32> to tensor<4xi64>
                %13 = tt.broadcast %12 : tensor<4xi64> -> tensor<4xi64>
                %c3_i64_3 = arith.constant 3 : i64
                %14 = tt.splat %c3_i64_3 : i64 -> tensor<4xi64>
                %15 = arith.cmpi slt, %13, %14 : tensor<4xi64>
                %16 = arith.andi %9, %15 : tensor<4xi1>
                %17 = tt.splat %true : i1 -> tensor<4xi1>
                %18 = arith.andi %16, %17 : tensor<4xi1>
                %c0_i64 = arith.constant 0 : i64
                %19 = tt.splat %c0_i64 : i64 -> tensor<4xi64>
                %20 = tt.splat %4 : i64 -> tensor<4xi64>
                %21 = arith.addi %20, %13 : tensor<4xi64>
                %22 = tt.splat %5 : i64 -> tensor<4xi64>
                %23 = arith.cmpi slt, %21, %22 : tensor<4xi64>
                %24 = arith.andi %18, %23 : tensor<4xi1>
                %c2_i64 = arith.constant 2 : i64
                %25 = tt.splat %c2_i64 : i64 -> tensor<4xi64>
                %26 = arith.cmpi sge, %21, %19 : tensor<4xi64>
                %27 = arith.cmpi slt, %21, %25 : tensor<4xi64>
                %28 = arith.andi %26, %27 : tensor<4xi1>
                %29 = arith.andi %24, %28 : tensor<4xi1>
                %c0_i64_4 = arith.constant 0 : i64
                %30 = tt.splat %c0_i64_4 : i64 -> tensor<4xi64>
                %31 = arith.select %28, %21, %30 : tensor<4xi1>, tensor<4xi64>
                %c1_i64_5 = arith.constant 1 : i64
                %32 = tt.splat %c1_i64_5 : i64 -> tensor<4xi64>
                %33 = arith.muli %31, %32 : tensor<4xi64>
                %34 = arith.addi %30, %33 : tensor<4xi64>
                %35 = arith.select %29, %34, %30 : tensor<4xi1>, tensor<4xi64>
                %36 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
                %37 = tt.addptr %36, %35 : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
                %38 = arith.andi %29, %8 : tensor<4xi1>
                tt.store %37, %7, %38 {operand_segment_sizes = array<i32: 1, 1, 1>} : tensor<4x!tt.ptr<f32>>
                tt.return
              }
            }
        "},
        );
    }

    #[test]
    fn test_lower_masked_swap() {
        // A swap whose old value is dead stores without reading, so it is admitted on a write-only parameter.
        let operation = tiled_call(vec![(
            ArrayType::new_static(DataType::F32, [2]),
            KernelParameterAccess::WriteOnly,
            BoundaryPolicy::Masked,
        )]);
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            let context = references[0].context();
            let value = context.lift(ArrayIrValue::Array(Array::vector(vec![5f32; 3])?))?;
            let mask = context.lift(ArrayIrValue::Array(Array::vector(vec![true; 3])?))?;
            masked_swap(context, &references[0], &value, &mask, -1f32)?;
            Ok(())
        })
        .unwrap();
        assert_eq!(definition.interpret(vec![], 1), Ok(vec![Array::vector(vec![5f32, 5.0]).unwrap()]));
        assert_eq!(
            lowered_module(&definition),
            indoc! {"
            module {
              tt.func public @ryft_kernel(%arg0: !tt.ptr<f32>) {
                %0 = tt.get_program_id x : i32
                %1 = arith.extsi %0 : i32 to i64
                %c1_i64 = arith.constant 1 : i64
                %2 = arith.remsi %1, %c1_i64 : i64
                %3 = arith.divsi %1, %c1_i64 : i64
                %true = arith.constant true
                %c3_i64 = arith.constant 3 : i64
                %4 = arith.muli %2, %c3_i64 : i64
                %c3_i64_0 = arith.constant 3 : i64
                %5 = arith.addi %4, %c3_i64_0 : i64
                %c1084227584_i32 = arith.constant 1084227584 : i32
                %6 = arith.bitcast %c1084227584_i32 : i32 to f32
                %7 = tt.splat %6 : f32 -> tensor<4xf32>
                %true_1 = arith.constant true
                %8 = tt.splat %true_1 : i1 -> tensor<4xi1>
                %c-1082130432_i32 = arith.constant -1082130432 : i32
                %9 = arith.bitcast %c-1082130432_i32 : i32 to f32
                %10 = tt.splat %9 : f32 -> tensor<4xf32>
                %true_2 = arith.constant true
                %11 = tt.splat %true_2 : i1 -> tensor<4xi1>
                %12 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
                %13 = tt.reshape %12 : tensor<4xi32> -> tensor<4xi32>
                %14 = arith.extsi %13 : tensor<4xi32> to tensor<4xi64>
                %15 = tt.broadcast %14 : tensor<4xi64> -> tensor<4xi64>
                %c3_i64_3 = arith.constant 3 : i64
                %16 = tt.splat %c3_i64_3 : i64 -> tensor<4xi64>
                %17 = arith.cmpi slt, %15, %16 : tensor<4xi64>
                %18 = arith.andi %11, %17 : tensor<4xi1>
                %19 = tt.splat %true : i1 -> tensor<4xi1>
                %20 = arith.andi %18, %19 : tensor<4xi1>
                %c0_i64 = arith.constant 0 : i64
                %21 = tt.splat %c0_i64 : i64 -> tensor<4xi64>
                %22 = tt.splat %4 : i64 -> tensor<4xi64>
                %23 = arith.addi %22, %15 : tensor<4xi64>
                %24 = tt.splat %5 : i64 -> tensor<4xi64>
                %25 = arith.cmpi slt, %23, %24 : tensor<4xi64>
                %26 = arith.andi %20, %25 : tensor<4xi1>
                %c2_i64 = arith.constant 2 : i64
                %27 = tt.splat %c2_i64 : i64 -> tensor<4xi64>
                %28 = arith.cmpi sge, %23, %21 : tensor<4xi64>
                %29 = arith.cmpi slt, %23, %27 : tensor<4xi64>
                %30 = arith.andi %28, %29 : tensor<4xi1>
                %31 = arith.andi %26, %30 : tensor<4xi1>
                %c0_i64_4 = arith.constant 0 : i64
                %32 = tt.splat %c0_i64_4 : i64 -> tensor<4xi64>
                %33 = arith.select %30, %23, %32 : tensor<4xi1>, tensor<4xi64>
                %c1_i64_5 = arith.constant 1 : i64
                %34 = tt.splat %c1_i64_5 : i64 -> tensor<4xi64>
                %35 = arith.muli %33, %34 : tensor<4xi64>
                %36 = arith.addi %32, %35 : tensor<4xi64>
                %37 = arith.select %31, %36, %32 : tensor<4xi1>, tensor<4xi64>
                %38 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
                %39 = tt.addptr %38, %37 : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
                %40 = arith.andi %31, %8 : tensor<4xi1>
                tt.store %39, %7, %40 {operand_segment_sizes = array<i32: 1, 1, 1>} : tensor<4x!tt.ptr<f32>>
                tt.return
              }
            }
        "},
        );
    }

    #[test]
    fn test_lower_masked_swap_rejects_used_old_values() {
        let operation = tiled_call(vec![
            (ArrayType::new_static(DataType::F32, [2]), KernelParameterAccess::WriteOnly, BoundaryPolicy::Masked),
            (ArrayType::new_static(DataType::F32, [3]), KernelParameterAccess::WriteOnly, BoundaryPolicy::InBounds),
        ]);
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            let context = references[0].context();
            let value = context.lift(ArrayIrValue::Array(Array::vector(vec![5f32; 3])?))?;
            let mask = window_mask(context, &references[0])?;
            masked_store(context, &references[0], &value, &mask)?;
            let previous = masked_swap(context, &references[0], &value, &mask, -1f32)?;
            references[1].write(&previous)
        })
        .unwrap();
        assert_eq!(
            rejection(&definition),
            (
                "masked_swap",
                "swaps whose old value is used need a CTA barrier, which ryft-mlir does not expose".to_owned(),
            ),
        );
    }

    #[test]
    fn test_lower_window_mask() {
        // The window mask is false at the padding element and at the physical padding, and it guards the store that
        // initializes the masked output.
        let operation = tiled_call(vec![(
            ArrayType::new_static(DataType::F32, [2]),
            KernelParameterAccess::WriteOnly,
            BoundaryPolicy::Masked,
        )]);
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            let context = references[0].context();
            let value = context.lift(ArrayIrValue::Array(Array::vector(vec![5f32; 3])?))?;
            let mask = window_mask(context, &references[0])?;
            masked_store(context, &references[0], &value, &mask)
        })
        .unwrap();
        assert_eq!(definition.interpret(vec![], 1), Ok(vec![Array::vector(vec![5f32, 5.0]).unwrap()]));
        assert_eq!(
            lowered_module(&definition),
            indoc! {"
            module {
              tt.func public @ryft_kernel(%arg0: !tt.ptr<f32>) {
                %0 = tt.get_program_id x : i32
                %1 = arith.extsi %0 : i32 to i64
                %c1_i64 = arith.constant 1 : i64
                %2 = arith.remsi %1, %c1_i64 : i64
                %3 = arith.divsi %1, %c1_i64 : i64
                %true = arith.constant true
                %c3_i64 = arith.constant 3 : i64
                %4 = arith.muli %2, %c3_i64 : i64
                %c3_i64_0 = arith.constant 3 : i64
                %5 = arith.addi %4, %c3_i64_0 : i64
                %c1084227584_i32 = arith.constant 1084227584 : i32
                %6 = arith.bitcast %c1084227584_i32 : i32 to f32
                %7 = tt.splat %6 : f32 -> tensor<4xf32>
                %true_1 = arith.constant true
                %8 = tt.splat %true_1 : i1 -> tensor<4xi1>
                %9 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
                %10 = tt.reshape %9 : tensor<4xi32> -> tensor<4xi32>
                %11 = arith.extsi %10 : tensor<4xi32> to tensor<4xi64>
                %12 = tt.broadcast %11 : tensor<4xi64> -> tensor<4xi64>
                %c3_i64_2 = arith.constant 3 : i64
                %13 = tt.splat %c3_i64_2 : i64 -> tensor<4xi64>
                %14 = arith.cmpi slt, %12, %13 : tensor<4xi64>
                %15 = arith.andi %8, %14 : tensor<4xi1>
                %16 = tt.splat %true : i1 -> tensor<4xi1>
                %17 = arith.andi %15, %16 : tensor<4xi1>
                %c0_i64 = arith.constant 0 : i64
                %18 = tt.splat %c0_i64 : i64 -> tensor<4xi64>
                %19 = tt.splat %4 : i64 -> tensor<4xi64>
                %20 = arith.addi %19, %12 : tensor<4xi64>
                %21 = tt.splat %5 : i64 -> tensor<4xi64>
                %22 = arith.cmpi slt, %20, %21 : tensor<4xi64>
                %23 = arith.andi %17, %22 : tensor<4xi1>
                %c2_i64 = arith.constant 2 : i64
                %24 = tt.splat %c2_i64 : i64 -> tensor<4xi64>
                %25 = arith.cmpi sge, %20, %18 : tensor<4xi64>
                %26 = arith.cmpi slt, %20, %24 : tensor<4xi64>
                %27 = arith.andi %25, %26 : tensor<4xi1>
                %28 = arith.andi %23, %27 : tensor<4xi1>
                %true_3 = arith.constant true
                %29 = tt.splat %true_3 : i1 -> tensor<4xi1>
                %30 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
                %31 = tt.reshape %30 : tensor<4xi32> -> tensor<4xi32>
                %32 = arith.extsi %31 : tensor<4xi32> to tensor<4xi64>
                %33 = tt.broadcast %32 : tensor<4xi64> -> tensor<4xi64>
                %c3_i64_4 = arith.constant 3 : i64
                %34 = tt.splat %c3_i64_4 : i64 -> tensor<4xi64>
                %35 = arith.cmpi slt, %33, %34 : tensor<4xi64>
                %36 = arith.andi %29, %35 : tensor<4xi1>
                %37 = tt.splat %true : i1 -> tensor<4xi1>
                %38 = arith.andi %36, %37 : tensor<4xi1>
                %c0_i64_5 = arith.constant 0 : i64
                %39 = tt.splat %c0_i64_5 : i64 -> tensor<4xi64>
                %40 = tt.splat %4 : i64 -> tensor<4xi64>
                %41 = arith.addi %40, %33 : tensor<4xi64>
                %42 = tt.splat %5 : i64 -> tensor<4xi64>
                %43 = arith.cmpi slt, %41, %42 : tensor<4xi64>
                %44 = arith.andi %38, %43 : tensor<4xi1>
                %c2_i64_6 = arith.constant 2 : i64
                %45 = tt.splat %c2_i64_6 : i64 -> tensor<4xi64>
                %46 = arith.cmpi sge, %41, %39 : tensor<4xi64>
                %47 = arith.cmpi slt, %41, %45 : tensor<4xi64>
                %48 = arith.andi %46, %47 : tensor<4xi1>
                %49 = arith.andi %44, %48 : tensor<4xi1>
                %c0_i64_7 = arith.constant 0 : i64
                %50 = tt.splat %c0_i64_7 : i64 -> tensor<4xi64>
                %51 = arith.select %48, %41, %50 : tensor<4xi1>, tensor<4xi64>
                %c1_i64_8 = arith.constant 1 : i64
                %52 = tt.splat %c1_i64_8 : i64 -> tensor<4xi64>
                %53 = arith.muli %51, %52 : tensor<4xi64>
                %54 = arith.addi %50, %53 : tensor<4xi64>
                %55 = arith.select %49, %54, %50 : tensor<4xi1>, tensor<4xi64>
                %56 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
                %57 = tt.addptr %56, %55 : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
                %58 = arith.andi %49, %28 : tensor<4xi1>
                tt.store %57, %7, %58 {operand_segment_sizes = array<i32: 1, 1, 1>} : tensor<4x!tt.ptr<f32>>
                tt.return
              }
            }
        "},
        );
    }

    #[test]
    fn test_lower_scalar_comparison() {
        // The rank-0 threshold is splat to the tile shape before the comparison, which selects the loaded elements.
        let operation = tiled_call(vec![
            (ArrayType::new_static(DataType::F32, [3]), KernelParameterAccess::ReadOnly, BoundaryPolicy::InBounds),
            (ArrayType::new_static(DataType::F32, [3]), KernelParameterAccess::ReadOnly, BoundaryPolicy::InBounds),
            (ArrayType::new_static(DataType::F32, [3]), KernelParameterAccess::WriteOnly, BoundaryPolicy::InBounds),
        ]);
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            let context = references[0].context();
            let threshold = context.lift(ArrayIrValue::Array(Array::scalar(0f32)?))?;
            let mask = compare(context, &references[1].read()?, &threshold, ComparisonDirection::GreaterThan)?;
            references[2].write(&masked_load(context, &references[0], &mask, 0f32)?)
        })
        .unwrap();
        assert_eq!(
            definition.interpret(
                vec![Array::vector(vec![1f32, 2.0, 3.0]).unwrap(), Array::vector(vec![1f32, -1.0, 2.0]).unwrap()],
                1,
            ),
            Ok(vec![Array::vector(vec![1f32, 0.0, 3.0]).unwrap()]),
        );
        assert_eq!(
            lowered_module(&definition),
            indoc! {"
            module {
              tt.func public @ryft_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
                %0 = tt.get_program_id x : i32
                %1 = arith.extsi %0 : i32 to i64
                %c1_i64 = arith.constant 1 : i64
                %2 = arith.remsi %1, %c1_i64 : i64
                %3 = arith.divsi %1, %c1_i64 : i64
                %true = arith.constant true
                %c3_i64 = arith.constant 3 : i64
                %4 = arith.muli %2, %c3_i64 : i64
                %c3_i64_0 = arith.constant 3 : i64
                %5 = arith.addi %4, %c3_i64_0 : i64
                %c3_i64_1 = arith.constant 3 : i64
                %6 = arith.muli %2, %c3_i64_1 : i64
                %c3_i64_2 = arith.constant 3 : i64
                %7 = arith.addi %6, %c3_i64_2 : i64
                %c3_i64_3 = arith.constant 3 : i64
                %8 = arith.muli %2, %c3_i64_3 : i64
                %c3_i64_4 = arith.constant 3 : i64
                %9 = arith.addi %8, %c3_i64_4 : i64
                %c0_i32 = arith.constant 0 : i32
                %10 = arith.bitcast %c0_i32 : i32 to f32
                %c0_i32_5 = arith.constant 0 : i32
                %11 = arith.bitcast %c0_i32_5 : i32 to f32
                %12 = tt.splat %11 : f32 -> tensor<4xf32>
                %cst = arith.constant 0.000000e+00 : f32
                %13 = tt.splat %cst : f32 -> tensor<4xf32>
                %true_6 = arith.constant true
                %14 = tt.splat %true_6 : i1 -> tensor<4xi1>
                %15 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
                %16 = tt.reshape %15 : tensor<4xi32> -> tensor<4xi32>
                %17 = arith.extsi %16 : tensor<4xi32> to tensor<4xi64>
                %18 = tt.broadcast %17 : tensor<4xi64> -> tensor<4xi64>
                %c3_i64_7 = arith.constant 3 : i64
                %19 = tt.splat %c3_i64_7 : i64 -> tensor<4xi64>
                %20 = arith.cmpi slt, %18, %19 : tensor<4xi64>
                %21 = arith.andi %14, %20 : tensor<4xi1>
                %22 = tt.splat %true : i1 -> tensor<4xi1>
                %23 = arith.andi %21, %22 : tensor<4xi1>
                %c0_i64 = arith.constant 0 : i64
                %24 = tt.splat %c0_i64 : i64 -> tensor<4xi64>
                %25 = tt.splat %6 : i64 -> tensor<4xi64>
                %26 = arith.addi %25, %18 : tensor<4xi64>
                %27 = tt.splat %7 : i64 -> tensor<4xi64>
                %28 = arith.cmpi slt, %26, %27 : tensor<4xi64>
                %29 = arith.andi %23, %28 : tensor<4xi1>
                %c3_i64_8 = arith.constant 3 : i64
                %30 = tt.splat %c3_i64_8 : i64 -> tensor<4xi64>
                %31 = arith.cmpi sge, %26, %24 : tensor<4xi64>
                %32 = arith.cmpi slt, %26, %30 : tensor<4xi64>
                %33 = arith.andi %31, %32 : tensor<4xi1>
                %34 = arith.andi %29, %33 : tensor<4xi1>
                %c0_i64_9 = arith.constant 0 : i64
                %35 = tt.splat %c0_i64_9 : i64 -> tensor<4xi64>
                %36 = arith.select %33, %26, %35 : tensor<4xi1>, tensor<4xi64>
                %c1_i64_10 = arith.constant 1 : i64
                %37 = tt.splat %c1_i64_10 : i64 -> tensor<4xi64>
                %38 = arith.muli %36, %37 : tensor<4xi64>
                %39 = arith.addi %35, %38 : tensor<4xi64>
                %40 = arith.select %34, %39, %35 : tensor<4xi1>, tensor<4xi64>
                %41 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
                %42 = tt.addptr %41, %40 : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
                %43 = tt.load %42, %34, %13 : tensor<4x!tt.ptr<f32>>
                %44 = tt.splat %10 : f32 -> tensor<4xf32>
                %45 = arith.cmpf ogt, %43, %44 : tensor<4xf32>
                %true_11 = arith.constant true
                %46 = tt.splat %true_11 : i1 -> tensor<4xi1>
                %47 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
                %48 = tt.reshape %47 : tensor<4xi32> -> tensor<4xi32>
                %49 = arith.extsi %48 : tensor<4xi32> to tensor<4xi64>
                %50 = tt.broadcast %49 : tensor<4xi64> -> tensor<4xi64>
                %c3_i64_12 = arith.constant 3 : i64
                %51 = tt.splat %c3_i64_12 : i64 -> tensor<4xi64>
                %52 = arith.cmpi slt, %50, %51 : tensor<4xi64>
                %53 = arith.andi %46, %52 : tensor<4xi1>
                %54 = tt.splat %true : i1 -> tensor<4xi1>
                %55 = arith.andi %53, %54 : tensor<4xi1>
                %c0_i64_13 = arith.constant 0 : i64
                %56 = tt.splat %c0_i64_13 : i64 -> tensor<4xi64>
                %57 = tt.splat %4 : i64 -> tensor<4xi64>
                %58 = arith.addi %57, %50 : tensor<4xi64>
                %59 = tt.splat %5 : i64 -> tensor<4xi64>
                %60 = arith.cmpi slt, %58, %59 : tensor<4xi64>
                %61 = arith.andi %55, %60 : tensor<4xi1>
                %c3_i64_14 = arith.constant 3 : i64
                %62 = tt.splat %c3_i64_14 : i64 -> tensor<4xi64>
                %63 = arith.cmpi sge, %58, %56 : tensor<4xi64>
                %64 = arith.cmpi slt, %58, %62 : tensor<4xi64>
                %65 = arith.andi %63, %64 : tensor<4xi1>
                %66 = arith.andi %61, %65 : tensor<4xi1>
                %c0_i64_15 = arith.constant 0 : i64
                %67 = tt.splat %c0_i64_15 : i64 -> tensor<4xi64>
                %68 = arith.select %65, %58, %67 : tensor<4xi1>, tensor<4xi64>
                %c1_i64_16 = arith.constant 1 : i64
                %69 = tt.splat %c1_i64_16 : i64 -> tensor<4xi64>
                %70 = arith.muli %68, %69 : tensor<4xi64>
                %71 = arith.addi %67, %70 : tensor<4xi64>
                %72 = arith.select %66, %71, %67 : tensor<4xi1>, tensor<4xi64>
                %73 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
                %74 = tt.addptr %73, %72 : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
                %75 = arith.andi %66, %45 : tensor<4xi1>
                %76 = tt.load %74, %75, %12 : tensor<4x!tt.ptr<f32>>
                %true_17 = arith.constant true
                %77 = tt.splat %true_17 : i1 -> tensor<4xi1>
                %78 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
                %79 = tt.reshape %78 : tensor<4xi32> -> tensor<4xi32>
                %80 = arith.extsi %79 : tensor<4xi32> to tensor<4xi64>
                %81 = tt.broadcast %80 : tensor<4xi64> -> tensor<4xi64>
                %c3_i64_18 = arith.constant 3 : i64
                %82 = tt.splat %c3_i64_18 : i64 -> tensor<4xi64>
                %83 = arith.cmpi slt, %81, %82 : tensor<4xi64>
                %84 = arith.andi %77, %83 : tensor<4xi1>
                %85 = tt.splat %true : i1 -> tensor<4xi1>
                %86 = arith.andi %84, %85 : tensor<4xi1>
                %c0_i64_19 = arith.constant 0 : i64
                %87 = tt.splat %c0_i64_19 : i64 -> tensor<4xi64>
                %88 = tt.splat %8 : i64 -> tensor<4xi64>
                %89 = arith.addi %88, %81 : tensor<4xi64>
                %90 = tt.splat %9 : i64 -> tensor<4xi64>
                %91 = arith.cmpi slt, %89, %90 : tensor<4xi64>
                %92 = arith.andi %86, %91 : tensor<4xi1>
                %c3_i64_20 = arith.constant 3 : i64
                %93 = tt.splat %c3_i64_20 : i64 -> tensor<4xi64>
                %94 = arith.cmpi sge, %89, %87 : tensor<4xi64>
                %95 = arith.cmpi slt, %89, %93 : tensor<4xi64>
                %96 = arith.andi %94, %95 : tensor<4xi1>
                %97 = arith.andi %92, %96 : tensor<4xi1>
                %c0_i64_21 = arith.constant 0 : i64
                %98 = tt.splat %c0_i64_21 : i64 -> tensor<4xi64>
                %99 = arith.select %96, %89, %98 : tensor<4xi1>, tensor<4xi64>
                %c1_i64_22 = arith.constant 1 : i64
                %100 = tt.splat %c1_i64_22 : i64 -> tensor<4xi64>
                %101 = arith.muli %99, %100 : tensor<4xi64>
                %102 = arith.addi %98, %101 : tensor<4xi64>
                %103 = arith.select %97, %102, %98 : tensor<4xi1>, tensor<4xi64>
                %104 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
                %105 = tt.addptr %104, %103 : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
                tt.store %105, %76, %97 {operand_segment_sizes = array<i32: 1, 1, 1>} : tensor<4x!tt.ptr<f32>>
                tt.return
              }
            }
        "},
        );
    }

    #[test]
    fn test_lowering_rejects_assertions() {
        for operation in [
            ArrayIrOperation::Assert(ryft_core::AssertOperation::new("check")),
            ArrayIrOperation::Array(ArrayOperation::Assert(ryft_core::AssertOperation::new("check"))),
        ] {
            let mut builder = ryft_core::ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
            builder
                .add_instruction(KernelOperation::Portable(operation), vec![], vec![predicate], None)
                .unwrap();
            let program = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![],
                    vec![ryft_core::Placeholder],
                    vec![],
                )
                .unwrap();
            let context = Context::new();
            let mut lowering = Lowering {
                context: &context,
                location: context.unknown_location(),
                remaining: 100,
                maximum_tile_elements: 1024,
            };
            let mut block = context.block_with_no_arguments();
            let predicate = lowering.boolean(&mut block, true).unwrap();
            assert!(
                matches!(lowering.region(&mut block, program.entry_region_ref(), &[LoweringValue::Native(predicate)]),
                Err(Error::Unsupported { operation: "assert", reason }) if reason == "runtime assertions require native failure propagation")
            );
        }
    }

    #[test]
    fn test_lowering_array_boolean_comparison() {
        use ryft_mlir::dialects::arith::CmpiOperation;

        let context = Context::new();
        let mut lowering = Lowering {
            context: &context,
            location: context.unknown_location(),
            remaining: 100,
            maximum_tile_elements: 1024,
        };
        let mut block = context.block_with_no_arguments();
        let left = lowering.boolean(&mut block, false).unwrap();
        let right = lowering.boolean(&mut block, true).unwrap();
        let boolean = ArrayType::scalar(DataType::Boolean);
        lowering
            .array(
                &mut block,
                &ArrayOperation::Compare(ryft_core::CompareOperation::new(ComparisonDirection::LessThan)),
                &[left, right],
                &[boolean.clone(), boolean.clone()],
                &boolean,
            )
            .unwrap();
        let operation = block.operations().unwrap().last().unwrap().unwrap();
        let comparison = unsafe { operation.cast::<arith::CmpiOperationRef>() }.unwrap();
        assert_eq!(comparison.predicate().unwrap(), arith::IntegerComparisonPredicate::UnsignedLessThan);
    }

    #[test]
    fn test_lowering_array_scalar_broadcasting() {
        // Rank-0 operands of comparisons and selects, including a rank-0 predicate, are splat to the result shape.
        let tile = ArrayType::new_static(DataType::F32, [4]);
        let scalar = ArrayType::scalar(DataType::F32);
        let compare = ArrayOperation::Compare(ryft_core::CompareOperation::new(ComparisonDirection::GreaterThan));
        assert_eq!(
            lowered_array(compare, vec![tile.clone(), scalar.clone()]).unwrap(),
            (vec!["tt.splat".to_owned(), "arith.cmpf".to_owned()], vec![], Some(1)),
        );
        let select = ArrayOperation::Select(ryft_core::SelectOperation::new());
        assert_eq!(
            lowered_array(select, vec![ArrayType::scalar(DataType::Boolean), scalar, tile]).unwrap(),
            (vec!["tt.splat".to_owned(), "tt.splat".to_owned(), "arith.select".to_owned()], vec![], Some(2)),
        );
    }

    #[test]
    fn test_lowering_array_mixed_element_types() {
        // Canonical inference promotes a Boolean operand to F32, which the native comparison cannot express.
        assert!(matches!(
            lowered_array(
                ArrayOperation::Compare(ryft_core::CompareOperation::new(ComparisonDirection::LessThan)),
                vec![ArrayType::new_static(DataType::F32, [4]), ArrayType::new_static(DataType::Boolean, [4])],
            ),
            Err(Error::Unsupported { operation: "compare", reason })
                if reason == "elementwise operands require equal element types",
        ));
    }

    #[test]
    fn test_lowering_array_fill_like() {
        let fill = |value: Array, r#type: ArrayType| {
            lowered_array(ArrayOperation::FillLike(FillLikeOperation::new(value).unwrap()), vec![r#type]).unwrap()
        };
        assert_eq!(
            fill(Array::scalar(2.5f32).unwrap(), ArrayType::new_static(DataType::F32, [2, 3])),
            (
                vec!["arith.constant".to_owned(), "arith.bitcast".to_owned(), "tt.splat".to_owned()],
                vec![i64::from(2.5f32.to_bits() as i32)],
                Some(2),
            ),
        );

        // Floating literals are emitted as integer bits, which preserves signs and NaN payloads exactly. Rank-0
        // outputs keep the scalar itself.
        assert_eq!(
            fill(Array::scalar(f32::from_bits(0xffc0_0001)).unwrap(), ArrayType::scalar(DataType::F32)),
            (
                vec!["arith.constant".to_owned(), "arith.bitcast".to_owned()],
                vec![i64::from(0xffc0_0001u32 as i32)],
                Some(1),
            ),
        );
        assert_eq!(
            fill(Array::scalar(-7i64).unwrap(), ArrayType::new_static(DataType::I64, [4])),
            (vec!["arith.constant".to_owned(), "tt.splat".to_owned()], vec![-7], Some(1)),
        );
    }

    #[test]
    fn test_lowering_array_iota() {
        let axis = [
            "tt.make_range",
            "tt.reshape",
            "arith.extsi",
            "tt.broadcast",
            "arith.constant",
            "tt.splat",
            "arith.cmpi",
            "arith.andi",
        ];
        let mut expected = vec!["arith.constant", "tt.splat"];
        expected.extend(axis);
        expected.extend(axis);
        let expected = expected.into_iter().map(str::to_owned).collect::<Vec<_>>();

        // The coordinates along dimension one are the second axis range, which follows the two mask operations and
        // the eight operations of the first axis.
        for (data_type, names, position) in [
            (DataType::I64, expected.clone(), 13),
            (DataType::F32, [expected, vec!["arith.sitofp".to_owned()]].concat(), 18),
        ] {
            let output = ArrayType::new_static(data_type, [2, 4]);
            let lowered = (names, vec![2, 4], Some(position));
            assert_eq!(
                lowered_array(ArrayOperation::Iota(IotaOperation::new(output, 1).unwrap()), vec![]).unwrap(),
                lowered,
            );
            assert_eq!(
                lowered_array(
                    ArrayOperation::IotaLike(IotaLikeOperation::new(1, data_type).unwrap()),
                    vec![ArrayType::new_static(DataType::Boolean, [2, 4])],
                )
                .unwrap(),
                lowered,
            );
        }
    }

    #[test]
    fn test_lowering_array_iota_like_unsupported_data_type() {
        assert!(matches!(
            lowered_array(
                ArrayOperation::IotaLike(IotaLikeOperation::new(0, DataType::I32).unwrap()),
                vec![ArrayType::new_static(DataType::F32, [4])],
            ),
            Err(Error::Unsupported { operation: "array", reason })
                if reason == "only F32, I64 and Boolean tile values are supported",
        ));
    }

    #[test]
    fn test_lowering_array_broadcast_like() {
        // Replicating along a new leading axis and stretching a unit axis both reshape before broadcasting, and the
        // exemplar's element type does not matter.
        for (value, exemplar, output_axes) in [
            (ArrayType::new_static(DataType::F32, [4]), ArrayType::new_static(DataType::F32, [2, 4]), vec![1]),
            (ArrayType::new_static(DataType::I64, [1, 4]), ArrayType::new_static(DataType::F32, [2, 4]), vec![0, 1]),
        ] {
            let output = ArrayType::new_static(value.data_type(), [2, 4]);
            let lowered = (vec!["tt.reshape".to_owned(), "tt.broadcast".to_owned()], vec![], Some(1));
            assert_eq!(
                lowered_array(
                    ArrayOperation::BroadcastLike(BroadcastLikeOperation::new(output_axes.clone())),
                    vec![value.clone(), exemplar],
                )
                .unwrap(),
                lowered,
            );
            assert_eq!(
                lowered_array(ArrayOperation::Broadcast(BroadcastOperation::new(output, output_axes)), vec![value])
                    .unwrap(),
                lowered,
            );
        }
        assert_eq!(
            lowered_array(
                ArrayOperation::BroadcastLike(BroadcastLikeOperation::new(vec![])),
                vec![ArrayType::scalar(DataType::F32), ArrayType::new_static(DataType::F32, [2, 4])],
            )
            .unwrap(),
            (vec!["tt.splat".to_owned()], vec![], Some(0)),
        );
    }

    #[test]
    fn test_lowering_array_broadcast_like_unordered_axes() {
        assert!(matches!(
            lowered_array(
                ArrayOperation::BroadcastLike(BroadcastLikeOperation::new(vec![1, 0])),
                vec![ArrayType::new_static(DataType::F32, [4, 2]), ArrayType::new_static(DataType::F32, [2, 4])],
            ),
            Err(Error::Unsupported { operation: "broadcast_like", reason })
                if reason == "broadcast axes must preserve source order",
        ));
    }

    #[test]
    fn test_lowering_region_reuses_only_identical_reference_transforms() {
        let context = Context::new();
        let lower_paths = |reference_transforms: &[(usize, usize)]| {
            let source_type = ArrayType::new_static(DataType::F32, [2, 2]);
            let call = KernelCallOperation::new(
                Grid::new(vec![]).unwrap(),
                vec![
                    whole_array_parameter(source_type.clone(), KernelParameterAccess::ReadOnly).unwrap(),
                    whole_array_parameter(source_type, KernelParameterAccess::ReadOnly).unwrap(),
                    whole_array_parameter(ArrayType::scalar(DataType::F32), KernelParameterAccess::WriteOnly).unwrap(),
                ],
            )
            .unwrap();
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let inputs = call
                .parameters()
                .iter()
                .map(|parameter| builder.add_input(parameter.body_type()))
                .collect::<Vec<_>>();
            let mut output = None;
            for &(root, index) in reference_transforms {
                let operation = ReferenceReadOperation::new().with_transforms(vec![
                    ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(index) },
                    ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) },
                ]);
                output = Some(builder.add_instruction(operation, vec![], vec![inputs[root]], None).unwrap()[0]);
            }
            builder
                .add_instruction(ReferenceWriteOperation::new(), vec![], vec![inputs[2], output.unwrap()], None)
                .unwrap();
            let body = builder.build(vec![], vec![Placeholder; 3], vec![]).unwrap();
            let definition = KernelDefinition::new(call, body).unwrap();
            let verified = VerifiedKernel::new(&definition, 1).unwrap();
            lower(&context, &verified, &Options::default(), &KernelSchedule::default())
                .unwrap()
                .module
                .to_string()
        };
        let single = lower_paths(&[(0, 0)]);
        let repeated = lower_paths(&[(0, 0), (0, 0), (0, 0)]);
        let different_path = lower_paths(&[(0, 0), (0, 0), (0, 1)]);
        let different_root = lower_paths(&[(0, 0), (0, 0), (1, 0)]);
        // Each load retains its own pointer arithmetic and memory effect; only the two applied transforms are shared.
        assert_eq!(repeated.matches("tt.load").count(), 3);
        assert_eq!(repeated.matches("arith.addi").count(), single.matches("arith.addi").count() + 4);
        assert_eq!(different_path.matches("tt.load").count(), 3);
        assert_eq!(different_path.matches("arith.addi").count(), repeated.matches("arith.addi").count() + 2);
        assert_eq!(different_root.matches("tt.load").count(), 3);
        assert_eq!(different_root.matches("arith.addi").count(), repeated.matches("arith.addi").count() + 2);
    }

    #[test]
    fn test_lowering_apply_transform() {
        let context = Context::new();
        context.load_dialect(DialectHandle::arith().unwrap()).unwrap();
        let mut lowering = Lowering {
            context: &context,
            location: context.unknown_location(),
            remaining: 100,
            maximum_tile_elements: 1024,
        };
        let mut block = context.block_with_no_arguments();
        let zero = lowering.integer(&mut block, 0).unwrap();
        let limit = lowering.integer(&mut block, 4).unwrap();
        let valid = lowering.boolean(&mut block, true).unwrap();
        let mut reference = Reference {
            pointer: zero,
            allocation: vec![4],
            axes: vec![0],
            fixed: vec![None],
            starts: vec![zero],
            limits: vec![limit],
            shape: vec![4],
            valid,
        };
        lowering
            .apply_transform(
                &mut block,
                &mut reference,
                &ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] },
                "reference_read",
            )
            .unwrap();
        assert_eq!(reference.shape, vec![2]);
        lowering
            .apply_transform(
                &mut block,
                &mut reference,
                &ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
                "reference_read",
            )
            .unwrap();
        assert_eq!(
            (reference.axes.len(), reference.starts.len(), reference.limits.len(), reference.shape.len()),
            (0, 0, 0, 0)
        );
        assert!(reference.fixed[0].is_some());
        assert!(matches!(lowering.apply_transform(&mut block, &mut reference, &ArrayReferenceTransform::Index {
            axis: 0, index: ArrayReferenceTransformIndex::Dynamic,
        }, "reference_read"), Err(Error::Unsupported { operation: "reference_read", reason })
            if reason == "dynamic reference transforms are unsupported"));
        assert!(matches!(lowering.apply_transform(&mut block, &mut reference, &ArrayReferenceTransform::Slice {
            axes: vec![ArraySliceAxis::new(0, 1, 2)],
        }, "reference_read"), Err(Error::Unsupported { operation: "reference_read", reason })
            if reason == "strided reference transforms are unsupported"));
    }

    #[test]
    fn test_lowering_physical() {
        let context = Context::new();
        let lowering = Lowering {
            context: &context,
            location: context.unknown_location(),
            remaining: 100,
            maximum_tile_elements: 1024,
        };
        assert_eq!(lowering.physical(&[0, 3]).unwrap(), vec![1, 4]);
        assert_eq!(lowering.physical(&[32, 32]).unwrap(), vec![32, 32]);
        assert!(matches!(lowering.physical(&[33, 32]), Err(Error::Unsupported { operation: "array", reason })
            if reason == "physical tile exceeds the element budget"));
        assert!(matches!(lowering.physical(&[1; 5]), Err(Error::Unsupported { operation: "array", reason })
            if reason == "tile rank exceeds four"));
        assert!(matches!(lowering.physical(&[usize::MAX]), Err(Error::Unsupported { operation: "array", reason })
            if reason == "physical tile extent overflows"));
    }

    #[test]
    fn test_lowering_charge() {
        let context = Context::new();
        let mut lowering = Lowering {
            context: &context,
            location: context.unknown_location(),
            remaining: 2,
            maximum_tile_elements: 1024,
        };
        lowering.charge(2).unwrap();
        assert_eq!(lowering.remaining, 0);
        assert!(matches!(lowering.charge(1), Err(Error::Unsupported { operation: "kernel_call", reason })
            if reason == "construction exceeds the instruction budget"));
    }

    #[test]
    fn test_shape() {
        assert_eq!(shape(&ArrayType::new_static(DataType::F32, [3, 0])).unwrap(), vec![3, 0]);
        assert!(matches!(shape(&ArrayType::new_static(DataType::F32, [i32::MAX as usize + 1])),
            Err(Error::Unsupported { operation: "array", reason })
                if reason == "array extents exceed signed 32-bit indexing"));
    }

    #[test]
    fn test_check_dimension() {
        let bounded = ryft_core::DimensionType::new(
            "index",
            ryft_core::DimensionBounds::non_negative(Some(i32::MAX as usize + 1)).unwrap(),
        );
        check_dimension(&bounded).unwrap();
        let unbounded = ryft_core::DimensionType::new("index", ryft_core::DimensionBounds::non_negative(None).unwrap());
        assert!(matches!(check_dimension(&unbounded), Err(Error::Unsupported { operation: "dimension", reason })
            if reason == "dimension bounds must fit the signed 32-bit indexing domain"));
    }
}
