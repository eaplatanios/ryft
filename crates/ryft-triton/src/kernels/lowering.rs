//! Typed TTIR construction from the verified portable kernel arena.
//!
//! Every parameter is a global F32 pointer, and each access of a window in global memory loads or stores a pointer
//! tensor under the validity mask of that window, which excludes its physical tile padding, its elements beyond the
//! enclosing window, and its elements beyond the array. Reference reads and
//! [`TileLoadOperation`](ryft_kernels::TileLoadOperation) lower to `tt.load` with that mask and an explicit `other`
//! value, and reference writes lower to `tt.store` with that mask. The host rejects an in-bounds tile load whose window
//! leaves its source view and any tile load whose window limit exceeds the canonical dimension width, and the lowering
//! cannot report such a failure at run time. Admission therefore requires the start bounds of every tile load to rule
//! it out, so an admitted in-bounds load reads only valid elements and lowers like a masked one. A dot lowers to one
//! `tt.dot` with IEEE input precision over unconverted F32 operands and an F32 accumulator, so admission accepts only
//! the precisions that request that computation and rejects every other precision, algorithm, and accumulation type.
//! [`MaskedLoadOperation`](ryft_kernels::MaskedLoadOperation) and
//! [`MaskedStoreOperation`](ryft_kernels::MaskedStoreOperation) intersect that mask with their explicit mask, so
//! they never touch padding, and a masked load takes the corresponding element of its `other` input for every element
//! outside that intersection. A [`WindowMaskOperation`](ryft_kernels::WindowMaskOperation) returns the validity mask
//! itself, which is false on the physical padding. A [`MaskedSwapOperation`](ryft_kernels::MaskedSwapOperation) or a
//! reference swap of a window in global memory lowers only when its old value is dead (i.e., when it is a
//! [`KernelSwapLowering::Store`](ryft_kernels::KernelSwapLowering::Store)), as a store. A reference add update loads
//! its window under the validity mask, adds the update, and stores the sum through the same pointers under the same
//! mask. A reference atomic add update is one masked `tt.atomic_rmw fadd` with `acq_rel` semantics at `gpu` scope,
//! and a scalar update of either is broadcast to the window.
//!
//! Sequential grid axes become `scf.for` loops inside each program, nested in declared axis order, so every program
//! visits its steps in the lexicographic order of the grid. The launch grid covers only the parallel axes. Bounded
//! `while` loops become `scf.while` loops and conditions become `scf.if` operations.
//!
//! # Promoted Windows
//!
//! Some windows of mutable parameters live in SSA values instead of global memory, like the accumulator tile of a
//! hand-written Triton kernel. Admission promotes a window when accessing it through global memory would need ordering
//! that Triton does not provide (refer to the next section) and every access of it uses its whole window (i.e., it is
//! a reference read, write, swap, or add update, or a masked load, store, or swap, without reference transforms). The
//! SSA value is one tile of the physical window shape that holds the current contents of the window, and every access
//! becomes a pure operation on it:
//!
//!   - A write replaces the value, and an add update adds the update to it.
//!   - A masked store selects the stored value where its mask and the window validity both hold, and it keeps the
//!     current value elsewhere.
//!   - A read returns the value, and a masked load selects the value under the same intersection and its `other`
//!     input elsewhere. A window mask is the validity mask, as for windows in global memory.
//!   - A swap returns the previous value, which a masked swap selects like a masked load, and then acts like the
//!     corresponding store.
//!
//! A read returns padding and invalid elements unmasked, which is unobservable, because the verifier rejects unmasked
//! reads of invalid elements and the lowering never lets physical padding reach memory, reductions, or dots.
//!
//! The scope of the value begins at the grid axis that follows the last sequential axis that the window tiles, so every
//! later sequential axis leaves the window unchanged. A window that a sequential axis tiles inside the loop of an axis
//! along which it repeats changes inside that loop and is not promoted. At the beginning of its scope, a read-write
//! window is loaded once under its validity mask, and a write-only window starts from zeros. No read observes those
//! zeros, because the verifier proves that every read follows a definite write (including through the first step rule
//! of a carry axis), so they stand for the uninitialized storage of a write-only parameter. Every
//! `scf.for` loop, `while` loop, and condition inside the scope carries the value. The loops carry it after their
//! ordinary carries and before the iteration counter of a `while` loop, the bounded condition of a `while` loop yields
//! it next to its predicate, and both branches of a condition yield it after their ordinary outputs. The scope ends
//! with one store of the value under the validity mask, so no padding element and no element beyond the array is ever
//! stored, and the verifier proves that a write-only window is completely written whenever its scope ends. When a grid
//! extent is zero, admission promotes nothing (refer to the next section), so the initial zeros of a write-only window
//! can never be stored.
//!
//! # Memory Ordering
//!
//! This section describes Triton as read from its sources at revision `678608832e2eed6a2d29462cc08e5c105089bea8`,
//! with the patches of the OpenXLA checkout that selects that revision. The linked compiler reports revision
//! `2122e24efc1886bfa36337c1f5b07c47d827b1d4`, so these statements have not been confirmed against the compiler that
//! actually runs, and only the NVIDIA lowering was read. No kernel that relies on them has been compiled natively or
//! run on a device.
//!
//! That Triton keeps the global memory accesses of one thread in program order. Its passes honor the memory
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
//! Admission therefore admits only bodies whose correctness does not depend on such ordering, promoting windows where
//! that removes the dependence:
//!
//!   - A write-only window stays in global memory when it is never read and, unless it is a scalar, which only the
//!     first thread of a program ever stores, it is written by at most one access of the entry region, outside control
//!     flow. A read-write window stays in global memory when it is accessed only by the entry region, all of its reads
//!     precede its only write, and no swap uses its old value. Every such window that is not a scalar changes at every
//!     step of each sequential axis with more than one step (i.e., its mapping tiles that axis).
//!   - Every other mutable window that is not accumulated atomically is promoted. A window that cannot be promoted
//!     (i.e., one with a partial access, or one that a sequential axis tiles inside the loop of an axis along which it
//!     repeats) is rejected with the reason why global memory would need ordering.
//!   - A promoted write-only window is never loaded, so nothing can observe its store out of order, and one thread
//!     stores each of its elements once per scope. The scopes of one program store distinct windows, because the axes
//!     outside a scope tile the window and the verifier proves that the windows of parallel programs are disjoint.
//!     The old value of a swap of such a window is an SSA value, so it may flow anywhere.
//!   - A read-write parameter, in global memory or promoted, is accessed only through its whole window, and every
//!     value read from it reaches memory only through elementwise operations into a write of that parameter. The
//!     values read from it include the old values of its swaps, and they may neither enter nor leave a region, which
//!     also keeps them out of every predicate. The thread that stores an element then computes it from a load of the
//!     same element, whose value the store depends on (in its own program order or through a layout conversion), so
//!     that load cannot observe the store, and every other load of that element feeds no memory. For a promoted
//!     window, that load is its entry load, that store is its exit store, and every update of the SSA value between
//!     them is elementwise, so the argument is unchanged. A swap whose old value flows anywhere else fails this test,
//!     because another thread may load that value after the store overwrites it.
//!   - A parameter that is accumulated atomically has no other access and is never promoted. Atomic updates are atomic
//!     among themselves in every order, and the interpreter admits every order of floating-point accumulation.
//!   - No load in a sequential loop reads memory that another iteration of that loop stores. A window in global memory
//!     changes at every step, and a promoted window is loaded and stored only outside the loops of the axes along which
//!     it repeats and at a different window for every step of the axes that tile it. Pipelining a load into an earlier
//!     iteration therefore never moves it past a store of the same memory. Within one step, the entry load of a
//!     promoted read-write window precedes its exit store as the previous rule requires.
//!
//! Parallel programs need no ordering, because the verifier proves that their mutable windows are disjoint or only
//! accumulated atomically. A grid with a zero extent runs the body at no grid point, and the single program that its
//! launch still runs masks every access by its inactive grid point, so admission applies no ordering rule to it.

use std::collections::{BTreeMap, HashMap, HashSet};

use ryft_core::{
    Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReferenceTransform,
    ArrayReferenceTransformIndex, ArrayType, Atom, AtomId, ComparisonDirection, DOT_OPERATION_NAME, DataType,
    Dimension, DimensionOperation, DimensionValue, DotAlgorithm, DotAlgorithmPreset, DotDimensionNumbers,
    DotInputPrecision, DotOperation, DotPrecision, InstructionId, Layout, MAX_DIMENSION_EXTENT, Memory,
    Operation as CoreOperation, ReductionKind, ReferenceAccess, ReferenceAccessMode, ReferenceAccessOperation,
    ReferenceRoot, RegionRef, Typed, infer_reference_view_type, validated_reference_access_descriptors,
};
use ryft_kernels::{
    ASYNC_COPY_OPERATION_NAME, BoundaryPolicy, GridExecution, KernelCallOperation, KernelOperation, KernelParameter,
    KernelParameterAccess, KernelReferenceSummary, KernelSchedule, KernelSwapLowering, TILE_LOAD_OPERATION_NAME,
    TileLoadOperation, VerifiedKernel,
};
use ryft_mlir::dialects::triton::tt;
use ryft_mlir::dialects::{arith, scf};
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

/// Whole window of a mutable parameter that the lowering keeps in an SSA value instead of global memory, which the
/// module documentation describes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Promotion {
    /// Position of the promoted parameter.
    parameter: usize,

    /// Grid axis at which the scope of the SSA value begins. Every sequential axis from this one onward leaves the
    /// window unchanged, so the window is loaded or initialized before the loops of those axes and stored after them.
    axis: usize,
}

/// Current SSA value of the window of a promoted parameter.
#[derive(Copy, Clone)]
struct PromotedWindow<'c, 't> {
    /// Global scalar pointer of the parameter, which identifies the references to its window.
    pointer: Native<'c, 't>,

    /// Value of the whole physical window at the current point of construction.
    value: Native<'c, 't>,
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

    /// Promoted windows that [`validate_memory_ordering`] admitted.
    promotions: Vec<Promotion>,

    /// Promoted windows whose scope encloses the current point of construction, in the order in which they started.
    windows: Vec<PromotedWindow<'c, 't>>,
}

/// Admits and constructs a complete native module without invoking a target compiler.
pub(super) fn lower<'c, 't>(
    context: &'c Context<'t>,
    kernel: &VerifiedKernel<'_>,
    options: &Options,
    _schedule: &KernelSchedule,
) -> Result<Lowered<'c, 't>, Error> {
    let call = kernel.definition().operation();
    let promotions = validate_memory_ordering(kernel)?;
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
        promotions,
        windows: Vec::new(),
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
        if operation.effects().has_execution_effects() {
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
    /// once at the innermost step over windows evaluated at `coordinates`. Parallel coordinates are already bound. The
    /// promoted windows whose scope begins at `axis` start before those loops and are stored after them, and every
    /// loop carries the values of all promoted windows whose scope encloses it.
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
        let opened = self.open_windows(block, call, pointers, coordinates, active, axis)?;
        if let Some(dimension) = call.grid().dimensions().get(axis) {
            if dimension.execution() == GridExecution::Parallel {
                self.grid(block, kernel, pointers, coordinates, active, axis + 1)?;
            } else {
                let Dimension::Static(extent) = dimension.extent() else { unreachable!() };
                let lower = self.integer(block, 0)?;
                let upper = self.integer(block, *extent as i64)?;
                let step = self.integer(block, 1)?;
                let carried = self.windows.iter().map(|window| window.value).collect::<Vec<_>>();
                let mut arguments = vec![(self.context.signless_integer_type(64).as_ref(), self.location)];
                for value in &carried {
                    arguments.push((value.r#type()?, self.location));
                }
                let mut body = self.context.block(&arguments);
                coordinates[axis] = body.argument(0)?.as_ref();
                for (index, window) in self.windows.iter_mut().enumerate() {
                    window.value = body.argument(index + 1)?.as_ref();
                }
                self.grid(&mut body, kernel, pointers, coordinates, active, axis + 1)?;
                let yielded = self.windows.iter().map(|window| window.value).collect::<Vec<_>>();
                body.append_operation(scf::r#yield(&yielded, self.location)?)?;
                let operation = block.append_operation(scf::r#for(
                    lower,
                    upper,
                    step,
                    &carried,
                    false,
                    body.try_into()?,
                    self.location,
                )?)?;
                for (index, window) in self.windows.iter_mut().enumerate() {
                    window.value = operation.result(index)?.as_ref();
                }
            }
        } else {
            let mut inputs = Vec::new();
            for (parameter, pointer) in call.parameters().iter().zip(pointers) {
                inputs.push(LoweringValue::Reference(self.window(block, parameter, *pointer, coordinates, active)?));
            }
            inputs.extend(coordinates.iter().copied().map(LoweringValue::Native));
            self.region(block, kernel.definition().body().entry_region_ref(), &inputs)?;
        }
        // Each scope ends in the block where it began, after every scope that began inside it has ended.
        let windows = self.windows.split_off(self.windows.len() - opened.len());
        for (window, reference) in windows.iter().zip(&opened) {
            self.write(block, reference, window.value, None)?;
        }
        Ok(())
    }

    /// Starts the SSA values of the promoted windows whose scope begins at grid axis `axis` and returns their windows,
    /// which [`Self::grid`] stores when the scope ends. Every sequential axis from `axis` onward leaves these windows
    /// unchanged, so they are evaluated with the coordinates of those axes, which are not bound yet, set to zero. A
    /// read-write window starts from one load of its contents. A write-only window starts from zeros, which no read
    /// observes because the verifier proves that every read follows a definite write, and which equal the zeroed
    /// storage that the host interpreter gives write-only parameters.
    fn open_windows(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        call: &KernelCallOperation,
        pointers: &[Native<'c, 't>],
        coordinates: &[Native<'c, 't>],
        active: Native<'c, 't>,
        axis: usize,
    ) -> Result<Vec<Reference<'c, 't>>, Error> {
        let promoted = self
            .promotions
            .iter()
            .filter(|promotion| promotion.axis == axis)
            .map(|promotion| promotion.parameter)
            .collect::<Vec<_>>();
        if promoted.is_empty() {
            return Ok(Vec::new());
        }
        let mut coordinates = coordinates.to_vec();
        let zero = self.integer(block, 0)?;
        for (position, dimension) in call.grid().dimensions().iter().enumerate().skip(axis) {
            if dimension.execution() != GridExecution::Parallel {
                coordinates[position] = zero;
            }
        }
        let mut opened = Vec::new();
        for index in promoted {
            let parameter = &call.parameters()[index];
            let reference = self.window(block, parameter, pointers[index], &coordinates, active)?;
            let zero = append(
                block,
                arith::constant(self.context.float_attribute(self.context.float32_type(), 0.0), self.location)?,
            )?;
            let zero = self.splat(block, zero, DataType::F32, &reference.shape)?;
            let value = match parameter.access() {
                KernelParameterAccess::ReadWrite => self.read(block, &reference, None, zero)?,
                _ => zero,
            };
            self.windows.push(PromotedWindow { pointer: reference.pointer, value });
            opened.push(reference);
        }
        Ok(opened)
    }

    /// Evaluates the window of `parameter` at the grid point `coordinates` as a global reference through `pointer`,
    /// which is valid only where `active` holds.
    fn window(
        &mut self,
        block: &mut DetachedBlock<'c, 't>,
        parameter: &KernelParameter,
        pointer: Native<'c, 't>,
        coordinates: &[Native<'c, 't>],
        active: Native<'c, 't>,
    ) -> Result<Reference<'c, 't>, Error> {
        let starts = self.mapping(block, parameter.mapping(), coordinates)?;
        let tile = parameter.mapping().block_shape().to_vec();
        let mut limits = Vec::new();
        for (start, extent) in starts.iter().zip(&tile) {
            let extent = self.integer(block, *extent as i64)?;
            limits.push(append(block, arith::addi(*start, extent, self.location)?)?);
        }
        let allocation = shape(&parameter.r#type())?;
        Ok(Reference {
            pointer,
            axes: (0..tile.len()).collect(),
            fixed: vec![None; allocation.len()],
            allocation,
            starts,
            limits,
            shape: tile,
            valid: active,
        })
    }

    /// Returns the position in [`Self::windows`] of the promoted window that `reference` addresses, if any. Promoted
    /// windows are accessed only through their whole window, so their references are never transformed.
    fn promoted(&self, reference: &Reference<'c, 't>) -> Option<usize> {
        self.windows.iter().position(|window| window.pointer == reference.pointer)
    }

    /// Restricts the Boolean tile `mask` to the valid elements of the global window `reference`.
    fn valid(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        reference: &Reference<'c, 't>,
        mask: Native<'c, 't>,
    ) -> Result<Native<'c, 't>, Error> {
        let (_, valid) = self.validity(block, reference)?;
        append(block, arith::andi(valid, mask, self.location)?)
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
                KernelOperation::Portable(ArrayIrOperation::ReferenceRead(_))
                    if let Some(window) = self.promoted(inputs[0].reference()?) =>
                {
                    // The verifier rejects unmasked reads of invalid elements, and the physical padding never reaches
                    // memory, so the whole value is the read.
                    vec![LoweringValue::Native(self.windows[window].value)]
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
                | KernelOperation::Portable(ArrayIrOperation::ReferenceSwap(_))
                    if let Some(window) = self.promoted(inputs[0].reference()?) =>
                {
                    // A write has no outputs, so the previous value binds only the old value of a swap.
                    vec![LoweringValue::Native(std::mem::replace(&mut self.windows[window].value, native(1)?))]
                }
                KernelOperation::Portable(ArrayIrOperation::ReferenceWrite(_))
                | KernelOperation::Portable(ArrayIrOperation::ReferenceSwap(_)) => {
                    // Admission promotes or rejects every window whose swap has a used old value, so a swap through
                    // memory is a store whose dead old value is never bound.
                    self.write(block, inputs[0].reference()?, native(1)?, None)?;
                    vec![]
                }
                KernelOperation::Portable(ArrayIrOperation::ReferenceAddUpdate(_))
                    if let Some(window) = self.promoted(inputs[0].reference()?) =>
                {
                    let reference = inputs[0].reference()?;
                    let update = self.update(block, operation.name(), reference, native(1)?, &input_types[1])?;
                    let sum = append(block, arith::addf(self.windows[window].value, update, location)?)?;
                    self.windows[window].value = sum;
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
                    validate_tile_load(operation, &input_types)?;
                    let mut reference = inputs[0].reference()?.clone();
                    reference.shape = operation.block_shape().to_vec();
                    for (axis, start) in reference.starts.iter_mut().enumerate() {
                        *start = append(block, arith::addi(*start, native(axis + 1)?, location)?)?;
                    }
                    let other = self.splat(block, native(inputs.len() - 1)?, DataType::F32, &reference.shape)?;
                    vec![LoweringValue::Native(self.read(block, &reference, None, other)?)]
                }
                KernelOperation::MaskedLoad(_) if let Some(window) = self.promoted(inputs[0].reference()?) => {
                    let mask = self.valid(block, inputs[0].reference()?, native(1)?)?;
                    let value = self.windows[window].value;
                    vec![LoweringValue::Native(append(block, arith::select(mask, value, native(2)?, location)?)?)]
                }
                KernelOperation::MaskedLoad(_) => {
                    vec![LoweringValue::Native(self.read(
                        block,
                        inputs[0].reference()?,
                        Some(native(1)?),
                        native(2)?,
                    )?)]
                }
                KernelOperation::MaskedStore(_) | KernelOperation::MaskedSwap(_)
                    if let Some(window) = self.promoted(inputs[0].reference()?) =>
                {
                    // The old value of a masked swap takes `other` outside the mask, like a masked load.
                    let mask = self.valid(block, inputs[0].reference()?, native(2)?)?;
                    let previous = self.windows[window].value;
                    let old = match instruction.operation() {
                        KernelOperation::MaskedSwap(_) => vec![LoweringValue::Native(append(
                            block,
                            arith::select(mask, previous, native(3)?, location)?,
                        )?)],
                        _ => vec![],
                    };
                    self.windows[window].value = append(block, arith::select(mask, native(1)?, previous, location)?)?;
                    old
                }
                KernelOperation::MaskedStore(_) => {
                    self.write(block, inputs[0].reference()?, native(1)?, Some(native(2)?))?;
                    vec![]
                }
                KernelOperation::MaskedSwap(_) => {
                    // Admission promotes or rejects every window whose swap has a used old value, so a masked swap
                    // through memory is a masked store whose dead old value is never bound.
                    self.write(block, inputs[0].reference()?, native(1)?, Some(native(2)?))?;
                    vec![]
                }
                KernelOperation::WindowMask(_) => {
                    vec![LoweringValue::Native(self.validity(block, inputs[0].reference()?)?.1)]
                }
                KernelOperation::Portable(ArrayIrOperation::Condition(_)) => {
                    // Both branches start from the values of the promoted windows before the condition and yield their
                    // ordinary outputs followed by the values of those windows.
                    let carried = self.windows.iter().map(|window| window.value).collect::<Vec<_>>();
                    let mut branches = Vec::new();
                    for slot in instruction.regions() {
                        for (window, value) in self.windows.iter_mut().zip(&carried) {
                            window.value = *value;
                        }
                        let mut branch = self.context.block_with_no_arguments();
                        let outputs = self.region(&mut branch, region.with_id(*slot)?, &inputs[1..])?;
                        let mut yielded = outputs
                            .iter()
                            .filter_map(|output| match output {
                                LoweringValue::Native(value) => Some(*value),
                                LoweringValue::Reference(_) => None,
                            })
                            .collect::<Vec<_>>();
                        yielded.extend(self.windows.iter().map(|window| window.value));
                        branch.append_operation(scf::r#yield(&yielded, location)?)?;
                        branches.push((branch, outputs, yielded));
                    }
                    let (else_branch, else_outputs, _) = branches.pop().unwrap();
                    let (then_branch, then_outputs, yielded) = branches.pop().unwrap();
                    let types = yielded.iter().map(|value| value.r#type()).collect::<Result<Vec<_>, _>>()?;
                    let operation = block.append_operation(scf::r#if(
                        native(0)?,
                        &types,
                        then_branch.try_into()?,
                        Some(else_branch.try_into()?),
                        location,
                    )?)?;
                    let mut position = 0;
                    let mut results = Vec::new();
                    for (then_output, else_output) in then_outputs.iter().zip(&else_outputs) {
                        match (then_output, else_output) {
                            (LoweringValue::Native(_), LoweringValue::Native(_)) => {
                                results.push(LoweringValue::Native(operation.result(position)?.as_ref()));
                                position += 1;
                            }
                            (LoweringValue::Reference(left), LoweringValue::Reference(right)) if left == right => {
                                results.push(then_output.clone());
                            }
                            _ => {
                                return Err(unsupported(
                                    "condition",
                                    "branch reference outputs must denote the same window",
                                ));
                            }
                        }
                    }
                    for window in &mut self.windows {
                        window.value = operation.result(position)?.as_ref();
                        position += 1;
                    }
                    results
                }
                KernelOperation::Portable(ArrayIrOperation::While(operation)) => {
                    let bound = operation
                        .iteration_bound()
                        .filter(|bound| *bound <= i64::MAX as usize)
                        .ok_or_else(|| unsupported("while", "requires a finite signed 64-bit iteration bound"))?;
                    // The loop carries its ordinary inputs, then the values of the promoted windows, and then its
                    // iteration counter.
                    let mut initial = inputs
                        .iter()
                        .filter_map(|input| match input {
                            LoweringValue::Native(value) => Some(*value),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    let count = initial.len();
                    initial.extend(self.windows.iter().map(|window| window.value));
                    initial.push(self.integer(block, 0)?);
                    let types = initial.iter().map(|value| value.r#type()).collect::<Result<Vec<_>, _>>()?;
                    let arguments = types.iter().map(|r#type| (*r#type, location)).collect::<Vec<_>>();
                    let mut before = self.context.block(&arguments);
                    let before_values = carried_values(&before, &inputs)?;
                    let entering = (count..types.len() - 1)
                        .map(|index| before.argument(index).map(|value| value.as_ref()))
                        .collect::<Result<Vec<_>, _>>()?;
                    let counter = before.argument(types.len() - 1)?.as_ref();
                    let limit = self.integer(&mut before, bound as i64)?;
                    let within = append(
                        &mut before,
                        arith::cmpi(counter, limit, arith::IntegerComparisonPredicate::SignedLessThan, location)?,
                    )?;
                    // The semantic bound is checked before the condition, including reference writes or assertions
                    // in that condition. A Boolean conjunction would evaluate the condition once too often.
                    for (window, value) in self.windows.iter_mut().zip(&entering) {
                        window.value = *value;
                    }
                    let mut condition = self.context.block_with_no_arguments();
                    let predicate =
                        self.region(&mut condition, region.with_id(instruction.regions()[0])?, &before_values)?;
                    let mut yielded = vec![predicate[0].native()?];
                    yielded.extend(self.windows.iter().map(|window| window.value));
                    condition.append_operation(scf::r#yield(&yielded, location)?)?;
                    let mut exhausted = self.context.block_with_no_arguments();
                    let mut stopped = vec![self.boolean(&mut exhausted, false)?];
                    stopped.extend(&entering);
                    exhausted.append_operation(scf::r#yield(&stopped, location)?)?;
                    let mut predicate_types = vec![self.context.signless_integer_type(1).as_ref()];
                    predicate_types.extend(&types[count..types.len() - 1]);
                    let predicate = before.append_operation(scf::r#if(
                        within,
                        &predicate_types,
                        condition.try_into()?,
                        Some(exhausted.try_into()?),
                        location,
                    )?)?;
                    let mut forwarded = (0..count)
                        .map(|index| before.argument(index).map(|value| value.as_ref()))
                        .collect::<Result<Vec<_>, _>>()?;
                    for index in 0..entering.len() {
                        forwarded.push(predicate.result(index + 1)?.as_ref());
                    }
                    forwarded.push(counter);
                    before.append_operation(scf::condition(predicate.result(0)?.as_ref(), &forwarded, location)?)?;
                    let mut after = self.context.block(&arguments);
                    let after_values = carried_values(&after, &inputs)?;
                    for (index, window) in self.windows.iter_mut().enumerate() {
                        window.value = after.argument(count + index)?.as_ref();
                    }
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
                    yielded.extend(self.windows.iter().map(|window| window.value));
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
                    for (index, window) in self.windows.iter_mut().enumerate() {
                        window.value = operation.result(count + index)?.as_ref();
                    }
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
            ArrayOperation::Dot(operation) => {
                validate_dot(operation, types, output)?;
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

/// Rejects every body whose memory accesses need ordering between the threads of one program and returns the windows
/// that the lowering promotes to SSA values instead, which the module documentation describes.
fn validate_memory_ordering(kernel: &VerifiedKernel<'_>) -> Result<Vec<Promotion>, Error> {
    let call = kernel.definition().operation();
    let body = kernel.definition().body().entry_region_ref();
    let references = kernel.definition().references();
    let analysis = references.analysis().analysis();

    // A grid with a zero extent runs the body at no grid point, and the single program that its launch still runs
    // masks every memory access by its inactive grid point, so no access needs ordering and nothing is promoted.
    if call.grid().dimensions().iter().any(|dimension| matches!(dimension.extent(), Dimension::Static(0))) {
        return Ok(Vec::new());
    }

    // Direct accesses by instruction, in program order within each region, together with the roots of the body that
    // they reach. Roots of nested region inputs resolve through the bindings of their attachments.
    let mut bindings = HashMap::<ReferenceRoot, Vec<ReferenceRoot>>::new();
    for binding in analysis.region_input_bindings() {
        // Bound inputs are reference-typed inputs of the attached region, which the analysis resolved.
        bindings.entry(analysis.root_of(binding.input()).unwrap()).or_default().push(binding.root());
    }
    let mut accesses = BTreeMap::<InstructionId, Vec<(ReferenceAccess, ReferenceRoot)>>::new();
    for access in analysis.access_modes() {
        let mut roots = vec![access.root()];
        while let Some(root) = roots.pop() {
            match bindings.get(&root) {
                Some(bound) => roots.extend(bound),
                None => accesses.entry(access.instruction()).or_default().push((*access, root)),
            }
        }
    }
    let operation = |instruction: InstructionId| -> Result<&'static str, Error> {
        Ok(body.with_id(instruction.region())?.instructions()[instruction.index()].operation().name())
    };

    let mut promotions = Vec::new();
    let mut read_write = HashSet::new();
    for (index, (parameter, summary)) in call.parameters().iter().zip(references.parameters()).enumerate() {
        let Some(summary) = summary else { continue };
        let root = summary.root();
        let modes = summary.modes();
        if modes.contains(&ReferenceAccessMode::AtomicAccumulate) {
            if modes.len() > 1 {
                return Err(unordered("kernel_call", "atomic and other accesses of one parameter"));
            }
            continue;
        }
        let read_write_parameter = match parameter.access() {
            KernelParameterAccess::ReadOnly => continue,
            KernelParameterAccess::WriteOnly => false,
            KernelParameterAccess::ReadWrite => true,
        };
        let parameter_accesses = accesses
            .iter()
            .flat_map(|(instruction, accesses)| {
                accesses.iter().filter(|(_, accessed)| *accessed == root).map(|(access, _)| (*instruction, *access))
            })
            .collect::<Vec<_>>();
        // Only reference reads, writes, swaps, and add updates and masked loads, stores, and swaps access a promoted
        // window, which they all do through their whole window.
        let mut partial = None;
        for (instruction, access) in &parameter_accesses {
            let whole = references.path(*instruction, access.input_index()).is_some_and(|path| path.is_root());
            let name = operation(*instruction)?;
            if partial.is_none() && (!whole || [TILE_LOAD_OPERATION_NAME, ASYNC_COPY_OPERATION_NAME].contains(&name)) {
                partial = Some(name);
            }
        }
        if read_write_parameter && let Some(name) = partial {
            return Err(unsupported(name, "read-write parameters must be accessed through their whole window"));
        }

        // The first reason, if any, why accessing the window through memory would need ordering.
        let reason = if let Some((instruction, _)) = parameter_accesses
            .iter()
            .find(|(instruction, _)| references.swap_lowering(*instruction) == Some(KernelSwapLowering::Exchange))
        {
            Some(unordered(operation(*instruction)?, "swaps whose old value is used"))
        } else if !read_write_parameter
            && modes
                .iter()
                .any(|mode| !matches!(mode, ReferenceAccessMode::Write | ReferenceAccessMode::ReadWrite))
        {
            Some(unordered("kernel_call", "reads after writes of one parameter"))
        } else if !read_write_parameter && parameter.mapping().block_shape().is_empty() {
            // Only the first thread of a program stores a scalar, in program order.
            continue;
        } else if call.grid().dimensions().iter().enumerate().any(|(axis, dimension)| {
            dimension.execution() != GridExecution::Parallel
                && !matches!(dimension.extent(), Dimension::Static(1))
                && !parameter.mapping().tiling_axes().is_some_and(|axes| axes.contains(&Some(axis)))
        }) {
            Some(unordered("kernel_call", "windows that repeat along a sequential grid axis"))
        } else if let Some(instruction) = body.instructions().iter().enumerate().find_map(|(index, instruction)| {
            (!instruction.regions().is_empty()
                && analysis
                    .transitive_access(InstructionId::new(body.id(), index))
                    .is_some_and(|access| access.roots().any(|accessed| accessed == root)))
            .then_some(instruction)
        }) {
            Some(unordered(instruction.operation().name(), "accesses of a mutable window inside control flow"))
        } else {
            let mut written = false;
            let mut reason = None;
            for (instruction, access) in &parameter_accesses {
                match access.mode() {
                    ReferenceAccessMode::Read | ReferenceAccessMode::Accumulate if written => {
                        reason = Some(unordered(operation(*instruction)?, "reads after writes of one parameter"));
                    }
                    ReferenceAccessMode::Read => continue,
                    _ if written => reason = Some(unordered(operation(*instruction)?, "repeated writes of one window")),
                    _ => {
                        written = true;
                        continue;
                    }
                }
                break;
            }
            reason
        };
        match reason {
            None => {}
            Some(reason) if partial.is_some() => return Err(reason),
            Some(reason) => {
                // The window stays the same along every sequential axis from the axis that follows the last one that
                // it tiles, so one SSA value can hold it across those axes. A window that some sequential axis tiles
                // inside the loop of an axis along which it repeats changes inside that loop and keeps its reason.
                let Some(tiling) = parameter.mapping().tiling_axes() else { return Err(reason) };
                let mut axis = 0;
                let mut repeats = false;
                for (position, dimension) in call.grid().dimensions().iter().enumerate() {
                    if dimension.execution() == GridExecution::Parallel
                        || matches!(dimension.extent(), Dimension::Static(1))
                    {
                        continue;
                    }
                    if !tiling.contains(&Some(position)) {
                        repeats = true;
                    } else if repeats {
                        return Err(reason);
                    } else {
                        axis = position + 1;
                    }
                }
                promotions.push(Promotion { parameter: index, axis });
            }
        }
        if read_write_parameter {
            read_write.insert(root);
        }
    }
    validate_read_write_values(body, None, &accesses, references, &read_write)?;
    Ok(promotions)
}

/// Rejects every value read from a read-write parameter in `region` that may reach memory other than elementwise
/// through a write of that parameter, which the module documentation describes. Such values may not enter or leave
/// a region, so `owner` names the operation that attaches a nested `region`, which must not return them.
fn validate_read_write_values(
    region: RegionRef<'_, ArrayIrValue<Array>, KernelOperation>,
    owner: Option<&'static str>,
    accesses: &BTreeMap<InstructionId, Vec<(ReferenceAccess, ReferenceRoot)>>,
    references: &KernelReferenceSummary,
    read_write: &HashSet<ReferenceRoot>,
) -> Result<(), Error> {
    let flow = |operation| {
        unsupported(
            operation,
            "values read from a read-write parameter may reach memory only elementwise through its own write",
        )
    };
    // Values computed elementwise from reads of read-write parameters, with the root of the parameter read.
    let mut read = HashMap::<AtomId, ReferenceRoot>::new();
    for (index, instruction) in region.instructions().iter().enumerate() {
        let operation = instruction.operation();
        let id = InstructionId::new(region.id(), index);
        let mut target = None;
        for (access, root) in accesses.get(&id).into_iter().flatten() {
            if !read_write.contains(root) {
                continue;
            }
            match access.mode() {
                ReferenceAccessMode::Read => read.extend(instruction.outputs().iter().map(|output| (*output, *root))),
                ReferenceAccessMode::ReadWrite
                    if references.swap_lowering(id) == Some(KernelSwapLowering::Exchange) =>
                {
                    read.extend(instruction.outputs().iter().map(|output| (*output, *root)));
                    target = Some(*root);
                }
                _ => target = Some(*root),
            }
        }
        let sources = instruction.inputs().iter().filter_map(|input| read.get(input).copied()).collect::<HashSet<_>>();
        if !instruction.regions().is_empty() {
            if !sources.is_empty() {
                return Err(flow(operation.name()));
            }
            for nested in instruction.regions() {
                validate_read_write_values(
                    region.with_id(*nested)?,
                    Some(operation.name()),
                    accesses,
                    references,
                    read_write,
                )?;
            }
            continue;
        }
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
            _ => return Err(flow(operation.name())),
        }
    }
    match owner {
        Some(owner) if region.output_ids().iter().any(|output| read.contains_key(output)) => Err(flow(owner)),
        _ => Ok(()),
    }
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

/// Validates that the dot that [`Lowering::array`] emits for `operation` over inputs of types `types` with the output
/// type `output` computes it as requested. That dot is one `tt.dot` with IEEE input precision over unconverted F32
/// operands that accumulates into an F32 tile, which every Triton target supports. It therefore implements the default
/// precision, the highest precision for either input, and the precise F32 algorithm, and admission rejects every other
/// precision or algorithm (e.g., [`DotAlgorithmPreset::BF16_BF16_F32`], which rounds the operands), every operand,
/// accumulation, or output type other than F32, and every output sharding. Parameter admission already rejects
/// sharded operands and layouts other than dense row-major ones.
fn validate_dot(operation: &DotOperation, types: &[ArrayType], output: &ArrayType) -> Result<(), Error> {
    if operation.dimensions() != &DotDimensionNumbers::matmul() || types.iter().any(|r#type| r#type.rank() != 2) {
        return Err(unsupported(DOT_OPERATION_NAME, "only rank-2 matrix multiplications are supported"));
    }
    if types.iter().chain([output]).any(|r#type| r#type.data_type() != DataType::F32) {
        return Err(unsupported(DOT_OPERATION_NAME, "operands, accumulation, and results must be F32"));
    }
    let ieee = match operation.precision() {
        DotPrecision::Inputs { lhs, rhs } => [lhs, rhs]
            .into_iter()
            .all(|precision| matches!(precision, DotInputPrecision::Default | DotInputPrecision::Highest)),
        DotPrecision::Algorithm(algorithm) => {
            algorithm == DotAlgorithm::new(DataType::F32, DataType::F32, DataType::F32)
        }
        DotPrecision::Preset(preset) => preset == DotAlgorithmPreset::F32_F32_F32,
    };
    if !ieee {
        let precision = operation.precision();
        return Err(unsupported(
            DOT_OPERATION_NAME,
            &format!("precision `{precision}` requires conversions or accumulation other than an IEEE F32 dot"),
        ));
    }
    if operation.output_sharding().is_some() {
        return Err(unsupported(DOT_OPERATION_NAME, "output shardings are unsupported"));
    }
    Ok(())
}

/// Validates that the start bounds of a tile load with the input types `input_types` prove that the host never rejects
/// its window, which the lowering cannot report at run time. Every window limit must fit the canonical dimension width,
/// and the window of an in-bounds load must also lie inside the view that its transforms select. An in-bounds load that
/// satisfies this check reads only valid elements, so it lowers like a masked load.
fn validate_tile_load(operation: &TileLoadOperation, input_types: &[ArrayIrType]) -> Result<(), Error> {
    let rank = operation.block_shape().len();
    let ArrayIrType::Reference(reference) = &input_types[0] else { unreachable!() };
    let bindings = input_types[rank + 2..].iter().collect::<Vec<_>>();
    let view =
        infer_reference_view_type(reference.referent(), operation.transforms(), &bindings, ReferenceAccessMode::Read)?;
    let shape = view.static_shape().unwrap();
    let starts = &input_types[1..rank + 1];
    for ((start, &block), &extent) in starts.iter().zip(operation.block_shape()).zip(shape.dimensions()) {
        let ArrayIrType::Dimension(start) = start else { unreachable!() };
        let limit = start.maximum_extent().and_then(|maximum| maximum.checked_add(block));
        let (bound, reason) = match operation.boundary_policy() {
            BoundaryPolicy::InBounds => {
                (extent, "in-bounds tile loads require start bounds that prove the window lies inside the source view")
            }
            BoundaryPolicy::Masked => (
                MAX_DIMENSION_EXTENT,
                "tile load start bounds must prove that the window limit fits the canonical dimension width",
            ),
        };
        if limit.is_none_or(|limit| limit > bound) {
            return Err(unsupported(TILE_LOAD_OPERATION_NAME, reason));
        }
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
        ArraySliceAxis, BroadcastLikeOperation, BroadcastOperation, ConditionOperation, Context as CoreContext,
        FillLikeOperation, IotaLikeOperation, IotaOperation, Placeholder, ProgramBuilder, ProgramError, ReferenceRead,
        ReferenceReadOperation, ReferenceWrite, ReferenceWriteOperation, TypeError,
    };
    use ryft_kernels::{
        Grid, GridDimension, KernelCallOperation, KernelDefinition, KernelParameterAccess, MappingExpression,
        ParameterWindow, compare, kernel, mapped_call, masked_load, masked_load_or, masked_store, masked_swap,
        whole_array_parameter, window_mask,
    };
    use ryft_mlir::{OpRef, Region, WalkOrder, WalkResult};

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
        lower(&context, &verified, &Options::default(), &KernelSchedule::default())
            .unwrap()
            .module
            .to_string()
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
            promotions: Vec::new(),
            windows: Vec::new(),
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

    /// Sums the rows of a matrix into masked output blocks of two rows, zeroing each block at the first step of the
    /// sequential axis that walks the masked column blocks of the input.
    #[kernel(
        crate = "ryft_kernels",
        grid(i = input.shape()[0].div_ceil(2), k = sequential(input.shape()[1].div_ceil(2))),
    )]
    fn row_sums(
        #[input(data_type = F32, rank = 2, block = [2, 2], index = [i, k], boundary = masked)] input: &Array,
        #[output(data_type = F32, shape = [input.shape()[0]], block = [2], index = [i], boundary = masked)]
        output: &mut Array,
    ) {
        if k == 0 {
            output.store(zeros::<f32>([2]));
        }
        output.accumulate(input.load_or(0.0).sum([1]));
    }

    /// Stores every input window of two elements into the same output window on one sequential axis, but tiles the
    /// output along a second sequential axis that the first one encloses.
    #[kernel(
        crate = "ryft_kernels",
        grid(repeat = sequential(2), k = sequential(input.shape()[0].div_ceil(2))),
    )]
    fn repeated_copy(
        #[input(data_type = F32, rank = 1, block = [2], index = [k])] input: &Array,
        #[output(data_type = F32, shape = [input.shape()[0]], block = [2], index = [k])] output: &mut Array,
    ) {
        output.store(input.load());
    }

    /// Copies `input` into `output` and then doubles `output` by reading it back.
    #[kernel(crate = "ryft_kernels")]
    fn copy_and_double(
        #[input(data_type = F32, rank = 1)] input: &Array,
        #[output(data_type = F32, shape = [input.shape()[0]])] output: &mut Array,
    ) {
        output.store(input.load());
        output.store(output.load() + output.load());
    }

    /// Copies every window of two elements of `input` and then zeroes the window of the first program.
    #[kernel(crate = "ryft_kernels", grid(i = input.shape()[0].div_ceil(2)))]
    fn zero_first(
        #[input(data_type = F32, rank = 1, block = [2], index = [i])] input: &Array,
        #[output(data_type = F32, shape = [input.shape()[0]], block = [2], index = [i])] output: &mut Array,
    ) {
        output.store(input.load());
        if i == 0 {
            output.store(zeros::<f32>([2]));
        }
    }

    /// Adds every window of two elements of `input` into `total` in place on a sequential axis.
    #[kernel(
        crate = "ryft_kernels",
        grid(k = sequential(input.shape()[0].div_ceil(2))),
        requires = total.shape()[0] == 2,
    )]
    fn accumulate_windows(
        #[input(data_type = F32, rank = 1, block = [2], index = [k])] input: &Array,
        #[inout(data_type = F32, rank = 1)] total: &mut Array,
    ) {
        total.accumulate(input.load());
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

    /// Writes the output window and then overwrites its first element through a slice.
    #[kernel(crate = "ryft_kernels", requires = input.shape()[0] >= 1)]
    fn overwrite_first(
        #[input(data_type = F32, rank = 1)] input: &Array,
        #[output(data_type = F32, shape = [input.shape()[0]])] output: &mut Array,
    ) {
        output.store(input.load());
        output.slice([0..1]).store(input.slice([0..1]).load() + input.slice([0..1]).load());
    }

    /// Reads `values` before a loop that stores twice that value into `values` in every iteration.
    #[kernel(crate = "ryft_kernels")]
    fn double_before_loop(#[inout(data_type = F32, rank = 1)] values: &mut Array) {
        let value = values.load();
        for step in 0..values.shape()[0] {
            values.store(value + value);
        }
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

    /// Reason for which admission rejects an in-bounds tile load whose window it cannot prove valid.
    const IN_BOUNDS_REASON: &str =
        "in-bounds tile loads require start bounds that prove the window lies inside the source view";

    /// Builds the common tile load fixture, which loads two elements with the padding value −1 at the constant element
    /// start `start` of the view that `transforms` select from the four element input, and stores them in the two
    /// element output.
    fn tile_load_definition(
        boundary: BoundaryPolicy,
        start: usize,
        transforms: Vec<ArrayReferenceTransform>,
    ) -> KernelDefinition {
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(ArrayType::new_static(DataType::F32, [4]), KernelParameterAccess::ReadOnly)
                    .unwrap(),
                whole_array_parameter(ArrayType::new_static(DataType::F32, [2]), KernelParameterAccess::WriteOnly)
                    .unwrap(),
            ],
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let inputs = call.body_input_types().into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let start = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(start).unwrap()));
        let other = builder.add_constant(ArrayIrValue::Array(Array::scalar(-1f32).unwrap()));
        let operation = TileLoadOperation::new(vec![2], boundary).unwrap().with_transforms(transforms);
        let tile = builder.add_instruction(operation, vec![], vec![inputs[0], start, other], None).unwrap()[0];
        builder
            .add_instruction(ReferenceWriteOperation::new(), vec![], vec![inputs[1], tile], None)
            .unwrap();
        KernelDefinition::new(call, builder.build(vec![], vec![Placeholder; 2], vec![]).unwrap()).unwrap()
    }

    /// Builds two definitions that contain the in-bounds tile load of [`tile_load_definition`] at the start 3, which
    /// the host never executes. The first runs it over an empty grid, which leaves its read-write output unchanged, and
    /// the second runs it in the untaken branch of a condition whose other branch returns the padding value.
    fn unreachable_tile_load_definitions() -> Vec<KernelDefinition> {
        let source = ArrayType::new_static(DataType::F32, [4]);
        let empty_grid = mapped_call(
            Grid::new(vec![GridDimension::new(0usize.into(), GridExecution::Parallel)]).unwrap(),
            vec![
                (source.clone(), KernelParameterAccess::ReadOnly, ParameterWindow::WholeArray),
                (
                    ArrayType::new_static(DataType::F32, [2]),
                    KernelParameterAccess::ReadWrite,
                    ParameterWindow::BlockIndices {
                        block_shape: vec![2],
                        boundary: BoundaryPolicy::Masked,
                        indices: vec![MappingExpression::GridCoordinate(0)],
                    },
                ),
            ],
            vec![],
        )
        .unwrap();
        let untaken_branch = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(source.clone(), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(ArrayType::new_static(DataType::F32, [2]), KernelParameterAccess::WriteOnly)
                    .unwrap(),
            ],
        )
        .unwrap();
        let load = |builder: &mut ProgramBuilder<ArrayIrValue<Array>, KernelOperation>, source| {
            let start = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
            let other = builder.add_constant(ArrayIrValue::Array(Array::scalar(-1f32).unwrap()));
            let operation = TileLoadOperation::new(vec![2], BoundaryPolicy::InBounds).unwrap();
            builder.add_instruction(operation, vec![], vec![source, start, other], None).unwrap()[0]
        };

        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let inputs = empty_grid
            .body_input_types()
            .into_iter()
            .map(|r#type| builder.add_input(r#type))
            .collect::<Vec<_>>();
        let tile = load(&mut builder, inputs[0]);
        builder
            .add_instruction(ReferenceWriteOperation::new(), vec![], vec![inputs[1], tile], None)
            .unwrap();
        let empty_grid =
            KernelDefinition::new(empty_grid, builder.build(vec![], vec![Placeholder; 3], vec![]).unwrap()).unwrap();

        let mut taken = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = taken.add_input(untaken_branch.parameters()[0].body_type());
        let tile = load(&mut taken, reference);
        let taken = taken
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![tile],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let mut padding = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        padding.add_input(untaken_branch.parameters()[0].body_type());
        let fill = padding.add_constant(ArrayIrValue::Array(Array::vector(vec![-1f32, -1.0]).unwrap()));
        let padding = padding
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![fill],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let inputs = untaken_branch
            .body_input_types()
            .into_iter()
            .map(|r#type| builder.add_input(r#type))
            .collect::<Vec<_>>();
        let predicate = builder.add_constant(ArrayIrValue::Array(Array::scalar(false).unwrap()));
        let taken = builder.import_region(taken.entry_region_ref());
        let padding = builder.import_region(padding.entry_region_ref());
        let tile = builder
            .add_instruction(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                vec![taken, padding],
                vec![predicate, inputs[0]],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(ReferenceWriteOperation::new(), vec![], vec![inputs[1], tile], None)
            .unwrap();
        let untaken_branch =
            KernelDefinition::new(untaken_branch, builder.build(vec![], vec![Placeholder; 2], vec![]).unwrap())
                .unwrap();
        vec![empty_grid, untaken_branch]
    }

    /// Builds the dot precision fixture, which multiplies its two 32×32 inputs with `operation`, and returns it
    /// together with the inputs `1.0009765625 · I` and `I`, where `1.0009765625 = 1 + 2⁻¹⁰` is exact in F32 and
    /// TensorFloat-32 and rounds to 1 in BF16.
    fn dot_definition(operation: DotOperation) -> (KernelDefinition, Vec<Array>) {
        let r#type = ArrayType::new_static(DataType::F32, [32, 32]);
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(r#type.clone(), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(r#type.clone(), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(r#type.clone(), KernelParameterAccess::WriteOnly).unwrap(),
            ],
        )
        .unwrap();
        let definition = KernelDefinition::trace(call, |(references, _)| {
            let inputs = [references[0].read()?, references[1].read()?];
            let operation = ArrayIrOperation::from(ArrayOperation::Dot(operation));
            references[2].write(&inputs[0].context().bind(operation, vec![], &inputs)?.remove(0))
        })
        .unwrap();
        let diagonal = |value: f32| {
            let elements = (0..32 * 32).map(|index| if index % 33 == 0 { value } else { 0.0 }).collect::<Vec<_>>();
            Array::from_elements(r#type.clone(), &elements).unwrap()
        };
        (definition, vec![diagonal(1.0 + 1.0 / 1024.0), diagonal(1.0)])
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
    fn test_lower_dot_precision() {
        // The host default preserves the diagonal value, and Triton emits one IEEE F32 dot without converting its
        // operands, which accumulates into an F32 tile.
        let (definition, inputs) = dot_definition(DotOperation::matmul());
        assert_eq!(definition.interpret(inputs.clone(), 1), Ok(vec![inputs[0].clone()]));
        let context = Context::new();
        let verified = VerifiedKernel::new(&definition, 1).unwrap();
        let lowered = lower(&context, &verified, &Options::default(), &KernelSchedule::default()).unwrap();
        let mut dots = Vec::new();
        lowered.module.as_operation().unwrap().walk(WalkOrder::PreOrder, |operation| {
            if operation.name().as_str() == Ok("tt.dot") {
                dots.push(
                    [tt::INPUT_PRECISION_ATTRIBUTE, tt::MAX_NUM_IMPRECISE_ACC_ATTRIBUTE]
                        .map(|name| operation.attribute(name).unwrap().unwrap().to_string()),
                );
            }
            WalkResult::Advance
        });
        let ieee = context.triton_tt_input_precision_attribute(tt::InputPrecision::Ieee).unwrap().to_string();
        assert_eq!(dots, vec![[ieee, "0 : i32".to_owned()]]);
        let module = lowered.module.to_string();
        assert_eq!(
            module.lines().filter(|line| line.contains("tt.dot")).collect::<Vec<_>>(),
            vec!["    %147 = tt.dot %128, %145, %146 : tensor<32x32xf32> * tensor<32x32xf32> -> tensor<32x32xf32>"],
        );
        assert_eq!(module.matches("arith.truncf").count() + module.matches("tt.fp_to_fp").count(), 0);

        // Every request of full F32 precision computes that same dot.
        for precision in [
            DotPrecision::from(DotInputPrecision::Highest),
            DotPrecision::from((DotInputPrecision::Default, DotInputPrecision::Highest)),
            DotPrecision::from(DotAlgorithmPreset::F32_F32_F32),
            DotPrecision::from(DotAlgorithm::new(DataType::F32, DataType::F32, DataType::F32)),
        ] {
            let (definition, _) = dot_definition(DotOperation::matmul().with_precision(precision));
            assert_eq!(lowered_module(&definition), module);
        }

        // Rounding the operands to BF16 rounds the diagonal value to 1 on the host, which the IEEE dot would not.
        let (definition, inputs) =
            dot_definition(DotOperation::matmul().with_precision(DotAlgorithmPreset::BF16_BF16_F32));
        assert_eq!(definition.interpret(inputs.clone(), 1), Ok(vec![inputs[1].clone()]));
        assert_eq!(
            rejection(&definition),
            (
                "dot",
                "precision `bf16_bf16_f32` requires conversions or accumulation other than an IEEE F32 dot".to_owned(),
            ),
        );
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
            definition
                .interpret(vec![Array::vector(vec![1f32, 2.0, 3.0, 4.0]).unwrap(), Array::scalar(10f32).unwrap()], 1,),
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
                %51 = tt.atomic_rmw fadd, acq_rel, gpu, %50, %47, %48 {operand_segment_sizes = array<i32: 1, 1, 1>} : \
                    (!tt.ptr<f32>, f32, i1) -> f32
                tt.return
              }
            }
        "},
        );
    }

    #[test]
    fn test_lower_promoted_carried_accumulation() {
        // The output repeats along the sequential axis, so its window lives in one SSA value that starts from zeros, is
        // zeroed again at the first step, and accumulates every masked input window. The loop stores nothing, and the
        // value is stored once after it.
        let definition = carried_sum::definition(&ArrayType::new_static(DataType::F32, [4])).unwrap();
        assert_eq!(
            definition.interpret(vec![Array::vector(vec![1f32, 2.0, 3.0, 4.0]).unwrap()], 1),
            Ok(vec![Array::vector(vec![4f32, 6.0]).unwrap()]),
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
                %c0_i64_0 = arith.constant 0 : i64
                %c2_i64 = arith.constant 2 : i64
                %2 = arith.addi %c0_i64_0, %c2_i64 : i64
                %cst = arith.constant 0.000000e+00 : f32
                %3 = tt.splat %cst : f32 -> tensor<2xf32>
                %c0_i64_1 = arith.constant 0 : i64
                %c2_i64_2 = arith.constant 2 : i64
                %c1_i64 = arith.constant 1 : i64
                %4 = scf.for %arg2 = %c0_i64_1 to %c2_i64_2 step %c1_i64 iter_args(%arg3 = %3) -> (tensor<2xf32>)  : \
                    i64 {
                  %c2_i64_9 = arith.constant 2 : i64
                  %34 = arith.muli %arg2, %c2_i64_9 : i64
                  %c2_i64_10 = arith.constant 2 : i64
                  %35 = arith.addi %34, %c2_i64_10 : i64
                  %c0_i64_11 = arith.constant 0 : i64
                  %c2_i64_12 = arith.constant 2 : i64
                  %36 = arith.addi %c0_i64_11, %c2_i64_12 : i64
                  %c0_i64_13 = arith.constant 0 : i64
                  %37 = arith.cmpi eq, %arg2, %c0_i64_13 : i64
                  %38 = scf.if %37 -> (tensor<2xf32>) {
                    %cst_20 = arith.constant 0.000000e+00 : f32
                    %74 = tt.splat %cst_20 : f32 -> tensor<2xf32>
                    scf.yield %74 : tensor<2xf32>
                  } else {
                    scf.yield %arg3 : tensor<2xf32>
                  }
                  %true_14 = arith.constant true
                  %39 = tt.splat %true_14 : i1 -> tensor<2xi1>
                  %c0_i32 = arith.constant 0 : i32
                  %40 = arith.bitcast %c0_i32 : i32 to f32
                  %41 = tt.splat %40 : f32 -> tensor<2xf32>
                  %true_15 = arith.constant true
                  %42 = tt.splat %true_15 : i1 -> tensor<2xi1>
                  %43 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                  %44 = tt.reshape %43 : tensor<2xi32> -> tensor<2xi32>
                  %45 = arith.extsi %44 : tensor<2xi32> to tensor<2xi64>
                  %46 = tt.broadcast %45 : tensor<2xi64> -> tensor<2xi64>
                  %c2_i64_16 = arith.constant 2 : i64
                  %47 = tt.splat %c2_i64_16 : i64 -> tensor<2xi64>
                  %48 = arith.cmpi slt, %46, %47 : tensor<2xi64>
                  %49 = arith.andi %42, %48 : tensor<2xi1>
                  %50 = tt.splat %true : i1 -> tensor<2xi1>
                  %51 = arith.andi %49, %50 : tensor<2xi1>
                  %c0_i64_17 = arith.constant 0 : i64
                  %52 = tt.splat %c0_i64_17 : i64 -> tensor<2xi64>
                  %53 = tt.splat %34 : i64 -> tensor<2xi64>
                  %54 = arith.addi %53, %46 : tensor<2xi64>
                  %55 = tt.splat %35 : i64 -> tensor<2xi64>
                  %56 = arith.cmpi slt, %54, %55 : tensor<2xi64>
                  %57 = arith.andi %51, %56 : tensor<2xi1>
                  %c4_i64 = arith.constant 4 : i64
                  %58 = tt.splat %c4_i64 : i64 -> tensor<2xi64>
                  %59 = arith.cmpi sge, %54, %52 : tensor<2xi64>
                  %60 = arith.cmpi slt, %54, %58 : tensor<2xi64>
                  %61 = arith.andi %59, %60 : tensor<2xi1>
                  %62 = arith.andi %57, %61 : tensor<2xi1>
                  %c0_i64_18 = arith.constant 0 : i64
                  %63 = tt.splat %c0_i64_18 : i64 -> tensor<2xi64>
                  %64 = arith.select %61, %54, %63 : tensor<2xi1>, tensor<2xi64>
                  %c1_i64_19 = arith.constant 1 : i64
                  %65 = tt.splat %c1_i64_19 : i64 -> tensor<2xi64>
                  %66 = arith.muli %64, %65 : tensor<2xi64>
                  %67 = arith.addi %63, %66 : tensor<2xi64>
                  %68 = arith.select %62, %67, %63 : tensor<2xi1>, tensor<2xi64>
                  %69 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
                  %70 = tt.addptr %69, %68 : tensor<2x!tt.ptr<f32>>, tensor<2xi64>
                  %71 = arith.andi %62, %39 : tensor<2xi1>
                  %72 = tt.load %70, %71, %41 : tensor<2x!tt.ptr<f32>>
                  %73 = arith.addf %38, %72 : tensor<2xf32>
                  scf.yield %73 : tensor<2xf32>
                }
                %true_3 = arith.constant true
                %5 = tt.splat %true_3 : i1 -> tensor<2xi1>
                %6 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                %7 = tt.reshape %6 : tensor<2xi32> -> tensor<2xi32>
                %8 = arith.extsi %7 : tensor<2xi32> to tensor<2xi64>
                %9 = tt.broadcast %8 : tensor<2xi64> -> tensor<2xi64>
                %c2_i64_4 = arith.constant 2 : i64
                %10 = tt.splat %c2_i64_4 : i64 -> tensor<2xi64>
                %11 = arith.cmpi slt, %9, %10 : tensor<2xi64>
                %12 = arith.andi %5, %11 : tensor<2xi1>
                %13 = tt.splat %true : i1 -> tensor<2xi1>
                %14 = arith.andi %12, %13 : tensor<2xi1>
                %c0_i64_5 = arith.constant 0 : i64
                %15 = tt.splat %c0_i64_5 : i64 -> tensor<2xi64>
                %16 = tt.splat %c0_i64_0 : i64 -> tensor<2xi64>
                %17 = arith.addi %16, %9 : tensor<2xi64>
                %18 = tt.splat %2 : i64 -> tensor<2xi64>
                %19 = arith.cmpi slt, %17, %18 : tensor<2xi64>
                %20 = arith.andi %14, %19 : tensor<2xi1>
                %c2_i64_6 = arith.constant 2 : i64
                %21 = tt.splat %c2_i64_6 : i64 -> tensor<2xi64>
                %22 = arith.cmpi sge, %17, %15 : tensor<2xi64>
                %23 = arith.cmpi slt, %17, %21 : tensor<2xi64>
                %24 = arith.andi %22, %23 : tensor<2xi1>
                %25 = arith.andi %20, %24 : tensor<2xi1>
                %c0_i64_7 = arith.constant 0 : i64
                %26 = tt.splat %c0_i64_7 : i64 -> tensor<2xi64>
                %27 = arith.select %24, %17, %26 : tensor<2xi1>, tensor<2xi64>
                %c1_i64_8 = arith.constant 1 : i64
                %28 = tt.splat %c1_i64_8 : i64 -> tensor<2xi64>
                %29 = arith.muli %27, %28 : tensor<2xi64>
                %30 = arith.addi %26, %29 : tensor<2xi64>
                %31 = arith.select %25, %30, %26 : tensor<2xi1>, tensor<2xi64>
                %32 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
                %33 = tt.addptr %32, %31 : tensor<2x!tt.ptr<f32>>, tensor<2xi64>
                tt.store %33, %4, %25 {operand_segment_sizes = array<i32: 1, 1, 1>} : tensor<2x!tt.ptr<f32>>
                tt.return
              }
            }
        "},
        );
    }

    #[test]
    fn test_lower_promoted_carried_accumulation_masked_edges() {
        // The second output window and the input windows of the last row and column cross the ends of their arrays.
        // The masked accumulation selects only valid elements of the promoted value, and its final store is masked by
        // the window validity, so the padding element of the output is never stored.
        let definition = row_sums::definition(&ArrayType::new_static(DataType::F32, [3, 3])).unwrap();
        assert_eq!(
            definition
                .interpret(vec![Array::matrix(3, 3, vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap()], 1,),
            Ok(vec![Array::vector(vec![6f32, 15.0, 24.0]).unwrap()]),
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
                %4 = arith.muli %2, %c2_i64_0 : i64
                %c2_i64_1 = arith.constant 2 : i64
                %5 = arith.addi %4, %c2_i64_1 : i64
                %cst = arith.constant 0.000000e+00 : f32
                %6 = tt.splat %cst : f32 -> tensor<2xf32>
                %c0_i64_2 = arith.constant 0 : i64
                %c2_i64_3 = arith.constant 2 : i64
                %c1_i64 = arith.constant 1 : i64
                %7 = scf.for %arg2 = %c0_i64_2 to %c2_i64_3 step %c1_i64 iter_args(%arg3 = %6) -> (tensor<2xf32>)  : \
                    i64 {
                  %c2_i64_9 = arith.constant 2 : i64
                  %c2_i64_10 = arith.constant 2 : i64
                  %37 = arith.muli %2, %c2_i64_9 : i64
                  %38 = arith.muli %arg2, %c2_i64_10 : i64
                  %c2_i64_11 = arith.constant 2 : i64
                  %39 = arith.addi %37, %c2_i64_11 : i64
                  %c2_i64_12 = arith.constant 2 : i64
                  %40 = arith.addi %38, %c2_i64_12 : i64
                  %c2_i64_13 = arith.constant 2 : i64
                  %41 = arith.muli %2, %c2_i64_13 : i64
                  %c2_i64_14 = arith.constant 2 : i64
                  %42 = arith.addi %41, %c2_i64_14 : i64
                  %c0_i64_15 = arith.constant 0 : i64
                  %43 = arith.cmpi eq, %arg2, %c0_i64_15 : i64
                  %44 = scf.if %43 -> (tensor<2xf32>) {
                    %cst_40 = arith.constant 0.000000e+00 : f32
                    %167 = tt.splat %cst_40 : f32 -> tensor<2xf32>
                    scf.yield %167 : tensor<2xf32>
                  } else {
                    scf.yield %arg3 : tensor<2xf32>
                  }
                  %true_16 = arith.constant true
                  %45 = tt.splat %true_16 : i1 -> tensor<2x2xi1>
                  %c0_i32 = arith.constant 0 : i32
                  %46 = arith.bitcast %c0_i32 : i32 to f32
                  %47 = tt.splat %46 : f32 -> tensor<2x2xf32>
                  %true_17 = arith.constant true
                  %48 = tt.splat %true_17 : i1 -> tensor<2x2xi1>
                  %49 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                  %50 = tt.reshape %49 : tensor<2xi32> -> tensor<2x1xi32>
                  %51 = arith.extsi %50 : tensor<2x1xi32> to tensor<2x1xi64>
                  %52 = tt.broadcast %51 : tensor<2x1xi64> -> tensor<2x2xi64>
                  %c2_i64_18 = arith.constant 2 : i64
                  %53 = tt.splat %c2_i64_18 : i64 -> tensor<2x2xi64>
                  %54 = arith.cmpi slt, %52, %53 : tensor<2x2xi64>
                  %55 = arith.andi %48, %54 : tensor<2x2xi1>
                  %56 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                  %57 = tt.reshape %56 : tensor<2xi32> -> tensor<1x2xi32>
                  %58 = arith.extsi %57 : tensor<1x2xi32> to tensor<1x2xi64>
                  %59 = tt.broadcast %58 : tensor<1x2xi64> -> tensor<2x2xi64>
                  %c2_i64_19 = arith.constant 2 : i64
                  %60 = tt.splat %c2_i64_19 : i64 -> tensor<2x2xi64>
                  %61 = arith.cmpi slt, %59, %60 : tensor<2x2xi64>
                  %62 = arith.andi %55, %61 : tensor<2x2xi1>
                  %63 = tt.splat %true : i1 -> tensor<2x2xi1>
                  %64 = arith.andi %62, %63 : tensor<2x2xi1>
                  %c0_i64_20 = arith.constant 0 : i64
                  %65 = tt.splat %c0_i64_20 : i64 -> tensor<2x2xi64>
                  %66 = tt.splat %37 : i64 -> tensor<2x2xi64>
                  %67 = arith.addi %66, %52 : tensor<2x2xi64>
                  %68 = tt.splat %39 : i64 -> tensor<2x2xi64>
                  %69 = arith.cmpi slt, %67, %68 : tensor<2x2xi64>
                  %70 = arith.andi %64, %69 : tensor<2x2xi1>
                  %c3_i64_21 = arith.constant 3 : i64
                  %71 = tt.splat %c3_i64_21 : i64 -> tensor<2x2xi64>
                  %72 = arith.cmpi sge, %67, %65 : tensor<2x2xi64>
                  %73 = arith.cmpi slt, %67, %71 : tensor<2x2xi64>
                  %74 = arith.andi %72, %73 : tensor<2x2xi1>
                  %75 = arith.andi %70, %74 : tensor<2x2xi1>
                  %76 = tt.splat %38 : i64 -> tensor<2x2xi64>
                  %77 = arith.addi %76, %59 : tensor<2x2xi64>
                  %78 = tt.splat %40 : i64 -> tensor<2x2xi64>
                  %79 = arith.cmpi slt, %77, %78 : tensor<2x2xi64>
                  %80 = arith.andi %75, %79 : tensor<2x2xi1>
                  %c3_i64_22 = arith.constant 3 : i64
                  %81 = tt.splat %c3_i64_22 : i64 -> tensor<2x2xi64>
                  %82 = arith.cmpi sge, %77, %65 : tensor<2x2xi64>
                  %83 = arith.cmpi slt, %77, %81 : tensor<2x2xi64>
                  %84 = arith.andi %82, %83 : tensor<2x2xi1>
                  %85 = arith.andi %80, %84 : tensor<2x2xi1>
                  %c0_i64_23 = arith.constant 0 : i64
                  %86 = tt.splat %c0_i64_23 : i64 -> tensor<2x2xi64>
                  %87 = arith.select %84, %77, %86 : tensor<2x2xi1>, tensor<2x2xi64>
                  %c1_i64_24 = arith.constant 1 : i64
                  %88 = tt.splat %c1_i64_24 : i64 -> tensor<2x2xi64>
                  %89 = arith.muli %87, %88 : tensor<2x2xi64>
                  %90 = arith.addi %86, %89 : tensor<2x2xi64>
                  %91 = arith.select %74, %67, %86 : tensor<2x2xi1>, tensor<2x2xi64>
                  %c3_i64_25 = arith.constant 3 : i64
                  %92 = tt.splat %c3_i64_25 : i64 -> tensor<2x2xi64>
                  %93 = arith.muli %91, %92 : tensor<2x2xi64>
                  %94 = arith.addi %90, %93 : tensor<2x2xi64>
                  %95 = arith.select %85, %94, %86 : tensor<2x2xi1>, tensor<2x2xi64>
                  %96 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
                  %97 = tt.addptr %96, %95 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi64>
                  %98 = arith.andi %85, %45 : tensor<2x2xi1>
                  %99 = tt.load %97, %98, %47 : tensor<2x2x!tt.ptr<f32>>
                  %true_26 = arith.constant true
                  %100 = tt.splat %true_26 : i1 -> tensor<2x2xi1>
                  %101 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                  %102 = tt.reshape %101 : tensor<2xi32> -> tensor<2x1xi32>
                  %103 = arith.extsi %102 : tensor<2x1xi32> to tensor<2x1xi64>
                  %104 = tt.broadcast %103 : tensor<2x1xi64> -> tensor<2x2xi64>
                  %c2_i64_27 = arith.constant 2 : i64
                  %105 = tt.splat %c2_i64_27 : i64 -> tensor<2x2xi64>
                  %106 = arith.cmpi slt, %104, %105 : tensor<2x2xi64>
                  %107 = arith.andi %100, %106 : tensor<2x2xi1>
                  %108 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                  %109 = tt.reshape %108 : tensor<2xi32> -> tensor<1x2xi32>
                  %110 = arith.extsi %109 : tensor<1x2xi32> to tensor<1x2xi64>
                  %111 = tt.broadcast %110 : tensor<1x2xi64> -> tensor<2x2xi64>
                  %c2_i64_28 = arith.constant 2 : i64
                  %112 = tt.splat %c2_i64_28 : i64 -> tensor<2x2xi64>
                  %113 = arith.cmpi slt, %111, %112 : tensor<2x2xi64>
                  %114 = arith.andi %107, %113 : tensor<2x2xi1>
                  %cst_29 = arith.constant 0.000000e+00 : f32
                  %115 = tt.splat %cst_29 : f32 -> tensor<2x2xf32>
                  %116 = arith.select %114, %99, %115 : tensor<2x2xi1>, tensor<2x2xf32>
                  %117 = \"tt.reduce\"(%116) <{axis = 1 : i32}> ({
                  ^bb0(%arg4: f32, %arg5: f32):
                    %167 = arith.addf %arg4, %arg5 : f32
                    tt.reduce.return %167 : f32
                  }) : (tensor<2x2xf32>) -> tensor<2xf32>
                  %true_30 = arith.constant true
                  %118 = tt.splat %true_30 : i1 -> tensor<2xi1>
                  %cst_31 = arith.constant 0.000000e+00 : f32
                  %119 = tt.splat %cst_31 : f32 -> tensor<2xf32>
                  %true_32 = arith.constant true
                  %120 = tt.splat %true_32 : i1 -> tensor<2xi1>
                  %121 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                  %122 = tt.reshape %121 : tensor<2xi32> -> tensor<2xi32>
                  %123 = arith.extsi %122 : tensor<2xi32> to tensor<2xi64>
                  %124 = tt.broadcast %123 : tensor<2xi64> -> tensor<2xi64>
                  %c2_i64_33 = arith.constant 2 : i64
                  %125 = tt.splat %c2_i64_33 : i64 -> tensor<2xi64>
                  %126 = arith.cmpi slt, %124, %125 : tensor<2xi64>
                  %127 = arith.andi %120, %126 : tensor<2xi1>
                  %128 = tt.splat %true : i1 -> tensor<2xi1>
                  %129 = arith.andi %127, %128 : tensor<2xi1>
                  %c0_i64_34 = arith.constant 0 : i64
                  %130 = tt.splat %c0_i64_34 : i64 -> tensor<2xi64>
                  %131 = tt.splat %41 : i64 -> tensor<2xi64>
                  %132 = arith.addi %131, %124 : tensor<2xi64>
                  %133 = tt.splat %42 : i64 -> tensor<2xi64>
                  %134 = arith.cmpi slt, %132, %133 : tensor<2xi64>
                  %135 = arith.andi %129, %134 : tensor<2xi1>
                  %c3_i64_35 = arith.constant 3 : i64
                  %136 = tt.splat %c3_i64_35 : i64 -> tensor<2xi64>
                  %137 = arith.cmpi sge, %132, %130 : tensor<2xi64>
                  %138 = arith.cmpi slt, %132, %136 : tensor<2xi64>
                  %139 = arith.andi %137, %138 : tensor<2xi1>
                  %140 = arith.andi %135, %139 : tensor<2xi1>
                  %141 = arith.andi %140, %118 : tensor<2xi1>
                  %142 = arith.select %141, %44, %119 : tensor<2xi1>, tensor<2xf32>
                  %143 = arith.addf %142, %117 : tensor<2xf32>
                  %true_36 = arith.constant true
                  %144 = tt.splat %true_36 : i1 -> tensor<2xi1>
                  %145 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                  %146 = tt.reshape %145 : tensor<2xi32> -> tensor<2xi32>
                  %147 = arith.extsi %146 : tensor<2xi32> to tensor<2xi64>
                  %148 = tt.broadcast %147 : tensor<2xi64> -> tensor<2xi64>
                  %c2_i64_37 = arith.constant 2 : i64
                  %149 = tt.splat %c2_i64_37 : i64 -> tensor<2xi64>
                  %150 = arith.cmpi slt, %148, %149 : tensor<2xi64>
                  %151 = arith.andi %144, %150 : tensor<2xi1>
                  %152 = tt.splat %true : i1 -> tensor<2xi1>
                  %153 = arith.andi %151, %152 : tensor<2xi1>
                  %c0_i64_38 = arith.constant 0 : i64
                  %154 = tt.splat %c0_i64_38 : i64 -> tensor<2xi64>
                  %155 = tt.splat %41 : i64 -> tensor<2xi64>
                  %156 = arith.addi %155, %148 : tensor<2xi64>
                  %157 = tt.splat %42 : i64 -> tensor<2xi64>
                  %158 = arith.cmpi slt, %156, %157 : tensor<2xi64>
                  %159 = arith.andi %153, %158 : tensor<2xi1>
                  %c3_i64_39 = arith.constant 3 : i64
                  %160 = tt.splat %c3_i64_39 : i64 -> tensor<2xi64>
                  %161 = arith.cmpi sge, %156, %154 : tensor<2xi64>
                  %162 = arith.cmpi slt, %156, %160 : tensor<2xi64>
                  %163 = arith.andi %161, %162 : tensor<2xi1>
                  %164 = arith.andi %159, %163 : tensor<2xi1>
                  %165 = arith.andi %164, %118 : tensor<2xi1>
                  %166 = arith.select %165, %143, %44 : tensor<2xi1>, tensor<2xf32>
                  scf.yield %166 : tensor<2xf32>
                }
                %true_4 = arith.constant true
                %8 = tt.splat %true_4 : i1 -> tensor<2xi1>
                %9 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                %10 = tt.reshape %9 : tensor<2xi32> -> tensor<2xi32>
                %11 = arith.extsi %10 : tensor<2xi32> to tensor<2xi64>
                %12 = tt.broadcast %11 : tensor<2xi64> -> tensor<2xi64>
                %c2_i64_5 = arith.constant 2 : i64
                %13 = tt.splat %c2_i64_5 : i64 -> tensor<2xi64>
                %14 = arith.cmpi slt, %12, %13 : tensor<2xi64>
                %15 = arith.andi %8, %14 : tensor<2xi1>
                %16 = tt.splat %true : i1 -> tensor<2xi1>
                %17 = arith.andi %15, %16 : tensor<2xi1>
                %c0_i64_6 = arith.constant 0 : i64
                %18 = tt.splat %c0_i64_6 : i64 -> tensor<2xi64>
                %19 = tt.splat %4 : i64 -> tensor<2xi64>
                %20 = arith.addi %19, %12 : tensor<2xi64>
                %21 = tt.splat %5 : i64 -> tensor<2xi64>
                %22 = arith.cmpi slt, %20, %21 : tensor<2xi64>
                %23 = arith.andi %17, %22 : tensor<2xi1>
                %c3_i64 = arith.constant 3 : i64
                %24 = tt.splat %c3_i64 : i64 -> tensor<2xi64>
                %25 = arith.cmpi sge, %20, %18 : tensor<2xi64>
                %26 = arith.cmpi slt, %20, %24 : tensor<2xi64>
                %27 = arith.andi %25, %26 : tensor<2xi1>
                %28 = arith.andi %23, %27 : tensor<2xi1>
                %c0_i64_7 = arith.constant 0 : i64
                %29 = tt.splat %c0_i64_7 : i64 -> tensor<2xi64>
                %30 = arith.select %27, %20, %29 : tensor<2xi1>, tensor<2xi64>
                %c1_i64_8 = arith.constant 1 : i64
                %31 = tt.splat %c1_i64_8 : i64 -> tensor<2xi64>
                %32 = arith.muli %30, %31 : tensor<2xi64>
                %33 = arith.addi %29, %32 : tensor<2xi64>
                %34 = arith.select %28, %33, %29 : tensor<2xi1>, tensor<2xi64>
                %35 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
                %36 = tt.addptr %35, %34 : tensor<2x!tt.ptr<f32>>, tensor<2xi64>
                tt.store %36, %7, %28 {operand_segment_sizes = array<i32: 1, 1, 1>} : tensor<2x!tt.ptr<f32>>
                tt.return
              }
            }
        "},
        );
    }

    #[test]
    fn test_lower_promoted_repeated_writes() {
        // The second write replaces the promoted value, and only that value is stored.
        let definition = write_twice::definition(&ArrayType::new_static(DataType::F32, [2])).unwrap();
        assert_eq!(
            definition.interpret(vec![Array::vector(vec![1f32, 2.0]).unwrap()], 1),
            Ok(vec![Array::vector(vec![2f32, 4.0]).unwrap()]),
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
                %c0_i64_0 = arith.constant 0 : i64
                %c2_i64 = arith.constant 2 : i64
                %2 = arith.addi %c0_i64_0, %c2_i64 : i64
                %cst = arith.constant 0.000000e+00 : f32
                %3 = tt.splat %cst : f32 -> tensor<2xf32>
                %c0_i64_1 = arith.constant 0 : i64
                %c2_i64_2 = arith.constant 2 : i64
                %4 = arith.addi %c0_i64_1, %c2_i64_2 : i64
                %c0_i64_3 = arith.constant 0 : i64
                %c2_i64_4 = arith.constant 2 : i64
                %5 = arith.addi %c0_i64_3, %c2_i64_4 : i64
                %cst_5 = arith.constant 0.000000e+00 : f32
                %6 = tt.splat %cst_5 : f32 -> tensor<2xf32>
                %true_6 = arith.constant true
                %7 = tt.splat %true_6 : i1 -> tensor<2xi1>
                %8 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                %9 = tt.reshape %8 : tensor<2xi32> -> tensor<2xi32>
                %10 = arith.extsi %9 : tensor<2xi32> to tensor<2xi64>
                %11 = tt.broadcast %10 : tensor<2xi64> -> tensor<2xi64>
                %c2_i64_7 = arith.constant 2 : i64
                %12 = tt.splat %c2_i64_7 : i64 -> tensor<2xi64>
                %13 = arith.cmpi slt, %11, %12 : tensor<2xi64>
                %14 = arith.andi %7, %13 : tensor<2xi1>
                %15 = tt.splat %true : i1 -> tensor<2xi1>
                %16 = arith.andi %14, %15 : tensor<2xi1>
                %c0_i64_8 = arith.constant 0 : i64
                %17 = tt.splat %c0_i64_8 : i64 -> tensor<2xi64>
                %18 = tt.splat %c0_i64_1 : i64 -> tensor<2xi64>
                %19 = arith.addi %18, %11 : tensor<2xi64>
                %20 = tt.splat %4 : i64 -> tensor<2xi64>
                %21 = arith.cmpi slt, %19, %20 : tensor<2xi64>
                %22 = arith.andi %16, %21 : tensor<2xi1>
                %c2_i64_9 = arith.constant 2 : i64
                %23 = tt.splat %c2_i64_9 : i64 -> tensor<2xi64>
                %24 = arith.cmpi sge, %19, %17 : tensor<2xi64>
                %25 = arith.cmpi slt, %19, %23 : tensor<2xi64>
                %26 = arith.andi %24, %25 : tensor<2xi1>
                %27 = arith.andi %22, %26 : tensor<2xi1>
                %c0_i64_10 = arith.constant 0 : i64
                %28 = tt.splat %c0_i64_10 : i64 -> tensor<2xi64>
                %29 = arith.select %26, %19, %28 : tensor<2xi1>, tensor<2xi64>
                %c1_i64 = arith.constant 1 : i64
                %30 = tt.splat %c1_i64 : i64 -> tensor<2xi64>
                %31 = arith.muli %29, %30 : tensor<2xi64>
                %32 = arith.addi %28, %31 : tensor<2xi64>
                %33 = arith.select %27, %32, %28 : tensor<2xi1>, tensor<2xi64>
                %34 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
                %35 = tt.addptr %34, %33 : tensor<2x!tt.ptr<f32>>, tensor<2xi64>
                %36 = tt.load %35, %27, %6 : tensor<2x!tt.ptr<f32>>
                %cst_11 = arith.constant 0.000000e+00 : f32
                %37 = tt.splat %cst_11 : f32 -> tensor<2xf32>
                %true_12 = arith.constant true
                %38 = tt.splat %true_12 : i1 -> tensor<2xi1>
                %39 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                %40 = tt.reshape %39 : tensor<2xi32> -> tensor<2xi32>
                %41 = arith.extsi %40 : tensor<2xi32> to tensor<2xi64>
                %42 = tt.broadcast %41 : tensor<2xi64> -> tensor<2xi64>
                %c2_i64_13 = arith.constant 2 : i64
                %43 = tt.splat %c2_i64_13 : i64 -> tensor<2xi64>
                %44 = arith.cmpi slt, %42, %43 : tensor<2xi64>
                %45 = arith.andi %38, %44 : tensor<2xi1>
                %46 = tt.splat %true : i1 -> tensor<2xi1>
                %47 = arith.andi %45, %46 : tensor<2xi1>
                %c0_i64_14 = arith.constant 0 : i64
                %48 = tt.splat %c0_i64_14 : i64 -> tensor<2xi64>
                %49 = tt.splat %c0_i64_1 : i64 -> tensor<2xi64>
                %50 = arith.addi %49, %42 : tensor<2xi64>
                %51 = tt.splat %4 : i64 -> tensor<2xi64>
                %52 = arith.cmpi slt, %50, %51 : tensor<2xi64>
                %53 = arith.andi %47, %52 : tensor<2xi1>
                %c2_i64_15 = arith.constant 2 : i64
                %54 = tt.splat %c2_i64_15 : i64 -> tensor<2xi64>
                %55 = arith.cmpi sge, %50, %48 : tensor<2xi64>
                %56 = arith.cmpi slt, %50, %54 : tensor<2xi64>
                %57 = arith.andi %55, %56 : tensor<2xi1>
                %58 = arith.andi %53, %57 : tensor<2xi1>
                %c0_i64_16 = arith.constant 0 : i64
                %59 = tt.splat %c0_i64_16 : i64 -> tensor<2xi64>
                %60 = arith.select %57, %50, %59 : tensor<2xi1>, tensor<2xi64>
                %c1_i64_17 = arith.constant 1 : i64
                %61 = tt.splat %c1_i64_17 : i64 -> tensor<2xi64>
                %62 = arith.muli %60, %61 : tensor<2xi64>
                %63 = arith.addi %59, %62 : tensor<2xi64>
                %64 = arith.select %58, %63, %59 : tensor<2xi1>, tensor<2xi64>
                %65 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
                %66 = tt.addptr %65, %64 : tensor<2x!tt.ptr<f32>>, tensor<2xi64>
                %67 = tt.load %66, %58, %37 : tensor<2x!tt.ptr<f32>>
                %cst_18 = arith.constant 0.000000e+00 : f32
                %68 = tt.splat %cst_18 : f32 -> tensor<2xf32>
                %true_19 = arith.constant true
                %69 = tt.splat %true_19 : i1 -> tensor<2xi1>
                %70 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                %71 = tt.reshape %70 : tensor<2xi32> -> tensor<2xi32>
                %72 = arith.extsi %71 : tensor<2xi32> to tensor<2xi64>
                %73 = tt.broadcast %72 : tensor<2xi64> -> tensor<2xi64>
                %c2_i64_20 = arith.constant 2 : i64
                %74 = tt.splat %c2_i64_20 : i64 -> tensor<2xi64>
                %75 = arith.cmpi slt, %73, %74 : tensor<2xi64>
                %76 = arith.andi %69, %75 : tensor<2xi1>
                %77 = tt.splat %true : i1 -> tensor<2xi1>
                %78 = arith.andi %76, %77 : tensor<2xi1>
                %c0_i64_21 = arith.constant 0 : i64
                %79 = tt.splat %c0_i64_21 : i64 -> tensor<2xi64>
                %80 = tt.splat %c0_i64_1 : i64 -> tensor<2xi64>
                %81 = arith.addi %80, %73 : tensor<2xi64>
                %82 = tt.splat %4 : i64 -> tensor<2xi64>
                %83 = arith.cmpi slt, %81, %82 : tensor<2xi64>
                %84 = arith.andi %78, %83 : tensor<2xi1>
                %c2_i64_22 = arith.constant 2 : i64
                %85 = tt.splat %c2_i64_22 : i64 -> tensor<2xi64>
                %86 = arith.cmpi sge, %81, %79 : tensor<2xi64>
                %87 = arith.cmpi slt, %81, %85 : tensor<2xi64>
                %88 = arith.andi %86, %87 : tensor<2xi1>
                %89 = arith.andi %84, %88 : tensor<2xi1>
                %c0_i64_23 = arith.constant 0 : i64
                %90 = tt.splat %c0_i64_23 : i64 -> tensor<2xi64>
                %91 = arith.select %88, %81, %90 : tensor<2xi1>, tensor<2xi64>
                %c1_i64_24 = arith.constant 1 : i64
                %92 = tt.splat %c1_i64_24 : i64 -> tensor<2xi64>
                %93 = arith.muli %91, %92 : tensor<2xi64>
                %94 = arith.addi %90, %93 : tensor<2xi64>
                %95 = arith.select %89, %94, %90 : tensor<2xi1>, tensor<2xi64>
                %96 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
                %97 = tt.addptr %96, %95 : tensor<2x!tt.ptr<f32>>, tensor<2xi64>
                %98 = tt.load %97, %89, %68 : tensor<2x!tt.ptr<f32>>
                %99 = arith.addf %67, %98 : tensor<2xf32>
                %true_25 = arith.constant true
                %100 = tt.splat %true_25 : i1 -> tensor<2xi1>
                %101 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                %102 = tt.reshape %101 : tensor<2xi32> -> tensor<2xi32>
                %103 = arith.extsi %102 : tensor<2xi32> to tensor<2xi64>
                %104 = tt.broadcast %103 : tensor<2xi64> -> tensor<2xi64>
                %c2_i64_26 = arith.constant 2 : i64
                %105 = tt.splat %c2_i64_26 : i64 -> tensor<2xi64>
                %106 = arith.cmpi slt, %104, %105 : tensor<2xi64>
                %107 = arith.andi %100, %106 : tensor<2xi1>
                %108 = tt.splat %true : i1 -> tensor<2xi1>
                %109 = arith.andi %107, %108 : tensor<2xi1>
                %c0_i64_27 = arith.constant 0 : i64
                %110 = tt.splat %c0_i64_27 : i64 -> tensor<2xi64>
                %111 = tt.splat %c0_i64_0 : i64 -> tensor<2xi64>
                %112 = arith.addi %111, %104 : tensor<2xi64>
                %113 = tt.splat %2 : i64 -> tensor<2xi64>
                %114 = arith.cmpi slt, %112, %113 : tensor<2xi64>
                %115 = arith.andi %109, %114 : tensor<2xi1>
                %c2_i64_28 = arith.constant 2 : i64
                %116 = tt.splat %c2_i64_28 : i64 -> tensor<2xi64>
                %117 = arith.cmpi sge, %112, %110 : tensor<2xi64>
                %118 = arith.cmpi slt, %112, %116 : tensor<2xi64>
                %119 = arith.andi %117, %118 : tensor<2xi1>
                %120 = arith.andi %115, %119 : tensor<2xi1>
                %c0_i64_29 = arith.constant 0 : i64
                %121 = tt.splat %c0_i64_29 : i64 -> tensor<2xi64>
                %122 = arith.select %119, %112, %121 : tensor<2xi1>, tensor<2xi64>
                %c1_i64_30 = arith.constant 1 : i64
                %123 = tt.splat %c1_i64_30 : i64 -> tensor<2xi64>
                %124 = arith.muli %122, %123 : tensor<2xi64>
                %125 = arith.addi %121, %124 : tensor<2xi64>
                %126 = arith.select %120, %125, %121 : tensor<2xi1>, tensor<2xi64>
                %127 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
                %128 = tt.addptr %127, %126 : tensor<2x!tt.ptr<f32>>, tensor<2xi64>
                tt.store %128, %99, %120 {operand_segment_sizes = array<i32: 1, 1, 1>} : tensor<2x!tt.ptr<f32>>
                tt.return
              }
            }
        "},
        );
    }

    #[test]
    fn test_lower_promoted_read_after_write() {
        // The reads of the output return the value that the first write left in its promoted value, so the output is
        // never loaded.
        let definition = copy_and_double::definition(&ArrayType::new_static(DataType::F32, [2])).unwrap();
        assert_eq!(
            definition.interpret(vec![Array::vector(vec![1f32, 2.0]).unwrap()], 1),
            Ok(vec![Array::vector(vec![2f32, 4.0]).unwrap()]),
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
                %c0_i64_0 = arith.constant 0 : i64
                %c2_i64 = arith.constant 2 : i64
                %2 = arith.addi %c0_i64_0, %c2_i64 : i64
                %cst = arith.constant 0.000000e+00 : f32
                %3 = tt.splat %cst : f32 -> tensor<2xf32>
                %c0_i64_1 = arith.constant 0 : i64
                %c2_i64_2 = arith.constant 2 : i64
                %4 = arith.addi %c0_i64_1, %c2_i64_2 : i64
                %c0_i64_3 = arith.constant 0 : i64
                %c2_i64_4 = arith.constant 2 : i64
                %5 = arith.addi %c0_i64_3, %c2_i64_4 : i64
                %cst_5 = arith.constant 0.000000e+00 : f32
                %6 = tt.splat %cst_5 : f32 -> tensor<2xf32>
                %true_6 = arith.constant true
                %7 = tt.splat %true_6 : i1 -> tensor<2xi1>
                %8 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                %9 = tt.reshape %8 : tensor<2xi32> -> tensor<2xi32>
                %10 = arith.extsi %9 : tensor<2xi32> to tensor<2xi64>
                %11 = tt.broadcast %10 : tensor<2xi64> -> tensor<2xi64>
                %c2_i64_7 = arith.constant 2 : i64
                %12 = tt.splat %c2_i64_7 : i64 -> tensor<2xi64>
                %13 = arith.cmpi slt, %11, %12 : tensor<2xi64>
                %14 = arith.andi %7, %13 : tensor<2xi1>
                %15 = tt.splat %true : i1 -> tensor<2xi1>
                %16 = arith.andi %14, %15 : tensor<2xi1>
                %c0_i64_8 = arith.constant 0 : i64
                %17 = tt.splat %c0_i64_8 : i64 -> tensor<2xi64>
                %18 = tt.splat %c0_i64_1 : i64 -> tensor<2xi64>
                %19 = arith.addi %18, %11 : tensor<2xi64>
                %20 = tt.splat %4 : i64 -> tensor<2xi64>
                %21 = arith.cmpi slt, %19, %20 : tensor<2xi64>
                %22 = arith.andi %16, %21 : tensor<2xi1>
                %c2_i64_9 = arith.constant 2 : i64
                %23 = tt.splat %c2_i64_9 : i64 -> tensor<2xi64>
                %24 = arith.cmpi sge, %19, %17 : tensor<2xi64>
                %25 = arith.cmpi slt, %19, %23 : tensor<2xi64>
                %26 = arith.andi %24, %25 : tensor<2xi1>
                %27 = arith.andi %22, %26 : tensor<2xi1>
                %c0_i64_10 = arith.constant 0 : i64
                %28 = tt.splat %c0_i64_10 : i64 -> tensor<2xi64>
                %29 = arith.select %26, %19, %28 : tensor<2xi1>, tensor<2xi64>
                %c1_i64 = arith.constant 1 : i64
                %30 = tt.splat %c1_i64 : i64 -> tensor<2xi64>
                %31 = arith.muli %29, %30 : tensor<2xi64>
                %32 = arith.addi %28, %31 : tensor<2xi64>
                %33 = arith.select %27, %32, %28 : tensor<2xi1>, tensor<2xi64>
                %34 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
                %35 = tt.addptr %34, %33 : tensor<2x!tt.ptr<f32>>, tensor<2xi64>
                %36 = tt.load %35, %27, %6 : tensor<2x!tt.ptr<f32>>
                %37 = arith.addf %36, %36 : tensor<2xf32>
                %true_11 = arith.constant true
                %38 = tt.splat %true_11 : i1 -> tensor<2xi1>
                %39 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                %40 = tt.reshape %39 : tensor<2xi32> -> tensor<2xi32>
                %41 = arith.extsi %40 : tensor<2xi32> to tensor<2xi64>
                %42 = tt.broadcast %41 : tensor<2xi64> -> tensor<2xi64>
                %c2_i64_12 = arith.constant 2 : i64
                %43 = tt.splat %c2_i64_12 : i64 -> tensor<2xi64>
                %44 = arith.cmpi slt, %42, %43 : tensor<2xi64>
                %45 = arith.andi %38, %44 : tensor<2xi1>
                %46 = tt.splat %true : i1 -> tensor<2xi1>
                %47 = arith.andi %45, %46 : tensor<2xi1>
                %c0_i64_13 = arith.constant 0 : i64
                %48 = tt.splat %c0_i64_13 : i64 -> tensor<2xi64>
                %49 = tt.splat %c0_i64_0 : i64 -> tensor<2xi64>
                %50 = arith.addi %49, %42 : tensor<2xi64>
                %51 = tt.splat %2 : i64 -> tensor<2xi64>
                %52 = arith.cmpi slt, %50, %51 : tensor<2xi64>
                %53 = arith.andi %47, %52 : tensor<2xi1>
                %c2_i64_14 = arith.constant 2 : i64
                %54 = tt.splat %c2_i64_14 : i64 -> tensor<2xi64>
                %55 = arith.cmpi sge, %50, %48 : tensor<2xi64>
                %56 = arith.cmpi slt, %50, %54 : tensor<2xi64>
                %57 = arith.andi %55, %56 : tensor<2xi1>
                %58 = arith.andi %53, %57 : tensor<2xi1>
                %c0_i64_15 = arith.constant 0 : i64
                %59 = tt.splat %c0_i64_15 : i64 -> tensor<2xi64>
                %60 = arith.select %57, %50, %59 : tensor<2xi1>, tensor<2xi64>
                %c1_i64_16 = arith.constant 1 : i64
                %61 = tt.splat %c1_i64_16 : i64 -> tensor<2xi64>
                %62 = arith.muli %60, %61 : tensor<2xi64>
                %63 = arith.addi %59, %62 : tensor<2xi64>
                %64 = arith.select %58, %63, %59 : tensor<2xi1>, tensor<2xi64>
                %65 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
                %66 = tt.addptr %65, %64 : tensor<2x!tt.ptr<f32>>, tensor<2xi64>
                tt.store %66, %37, %58 {operand_segment_sizes = array<i32: 1, 1, 1>} : tensor<2x!tt.ptr<f32>>
                tt.return
              }
            }
        "},
        );
    }

    #[test]
    fn test_lower_promoted_condition_write() {
        // The condition yields the promoted value from both branches, and the first program stores zeros.
        let definition = zero_first::definition(&ArrayType::new_static(DataType::F32, [4])).unwrap();
        assert_eq!(
            definition.interpret(vec![Array::vector(vec![1f32, 2.0, 3.0, 4.0]).unwrap()], 1),
            Ok(vec![Array::vector(vec![0f32, 0.0, 3.0, 4.0]).unwrap()]),
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
                %4 = arith.muli %2, %c2_i64_0 : i64
                %c2_i64_1 = arith.constant 2 : i64
                %5 = arith.addi %4, %c2_i64_1 : i64
                %cst = arith.constant 0.000000e+00 : f32
                %6 = tt.splat %cst : f32 -> tensor<2xf32>
                %c2_i64_2 = arith.constant 2 : i64
                %7 = arith.muli %2, %c2_i64_2 : i64
                %c2_i64_3 = arith.constant 2 : i64
                %8 = arith.addi %7, %c2_i64_3 : i64
                %c2_i64_4 = arith.constant 2 : i64
                %9 = arith.muli %2, %c2_i64_4 : i64
                %c2_i64_5 = arith.constant 2 : i64
                %10 = arith.addi %9, %c2_i64_5 : i64
                %c0_i64_6 = arith.constant 0 : i64
                %cst_7 = arith.constant 0.000000e+00 : f32
                %11 = tt.splat %cst_7 : f32 -> tensor<2xf32>
                %true_8 = arith.constant true
                %12 = tt.splat %true_8 : i1 -> tensor<2xi1>
                %13 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                %14 = tt.reshape %13 : tensor<2xi32> -> tensor<2xi32>
                %15 = arith.extsi %14 : tensor<2xi32> to tensor<2xi64>
                %16 = tt.broadcast %15 : tensor<2xi64> -> tensor<2xi64>
                %c2_i64_9 = arith.constant 2 : i64
                %17 = tt.splat %c2_i64_9 : i64 -> tensor<2xi64>
                %18 = arith.cmpi slt, %16, %17 : tensor<2xi64>
                %19 = arith.andi %12, %18 : tensor<2xi1>
                %20 = tt.splat %true : i1 -> tensor<2xi1>
                %21 = arith.andi %19, %20 : tensor<2xi1>
                %c0_i64_10 = arith.constant 0 : i64
                %22 = tt.splat %c0_i64_10 : i64 -> tensor<2xi64>
                %23 = tt.splat %7 : i64 -> tensor<2xi64>
                %24 = arith.addi %23, %16 : tensor<2xi64>
                %25 = tt.splat %8 : i64 -> tensor<2xi64>
                %26 = arith.cmpi slt, %24, %25 : tensor<2xi64>
                %27 = arith.andi %21, %26 : tensor<2xi1>
                %c4_i64 = arith.constant 4 : i64
                %28 = tt.splat %c4_i64 : i64 -> tensor<2xi64>
                %29 = arith.cmpi sge, %24, %22 : tensor<2xi64>
                %30 = arith.cmpi slt, %24, %28 : tensor<2xi64>
                %31 = arith.andi %29, %30 : tensor<2xi1>
                %32 = arith.andi %27, %31 : tensor<2xi1>
                %c0_i64_11 = arith.constant 0 : i64
                %33 = tt.splat %c0_i64_11 : i64 -> tensor<2xi64>
                %34 = arith.select %31, %24, %33 : tensor<2xi1>, tensor<2xi64>
                %c1_i64 = arith.constant 1 : i64
                %35 = tt.splat %c1_i64 : i64 -> tensor<2xi64>
                %36 = arith.muli %34, %35 : tensor<2xi64>
                %37 = arith.addi %33, %36 : tensor<2xi64>
                %38 = arith.select %32, %37, %33 : tensor<2xi1>, tensor<2xi64>
                %39 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
                %40 = tt.addptr %39, %38 : tensor<2x!tt.ptr<f32>>, tensor<2xi64>
                %41 = tt.load %40, %32, %11 : tensor<2x!tt.ptr<f32>>
                %42 = arith.cmpi eq, %2, %c0_i64_6 : i64
                %43 = scf.if %42 -> (tensor<2xf32>) {
                  %cst_18 = arith.constant 0.000000e+00 : f32
                  %73 = tt.splat %cst_18 : f32 -> tensor<2xf32>
                  scf.yield %73 : tensor<2xf32>
                } else {
                  scf.yield %41 : tensor<2xf32>
                }
                %true_12 = arith.constant true
                %44 = tt.splat %true_12 : i1 -> tensor<2xi1>
                %45 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                %46 = tt.reshape %45 : tensor<2xi32> -> tensor<2xi32>
                %47 = arith.extsi %46 : tensor<2xi32> to tensor<2xi64>
                %48 = tt.broadcast %47 : tensor<2xi64> -> tensor<2xi64>
                %c2_i64_13 = arith.constant 2 : i64
                %49 = tt.splat %c2_i64_13 : i64 -> tensor<2xi64>
                %50 = arith.cmpi slt, %48, %49 : tensor<2xi64>
                %51 = arith.andi %44, %50 : tensor<2xi1>
                %52 = tt.splat %true : i1 -> tensor<2xi1>
                %53 = arith.andi %51, %52 : tensor<2xi1>
                %c0_i64_14 = arith.constant 0 : i64
                %54 = tt.splat %c0_i64_14 : i64 -> tensor<2xi64>
                %55 = tt.splat %4 : i64 -> tensor<2xi64>
                %56 = arith.addi %55, %48 : tensor<2xi64>
                %57 = tt.splat %5 : i64 -> tensor<2xi64>
                %58 = arith.cmpi slt, %56, %57 : tensor<2xi64>
                %59 = arith.andi %53, %58 : tensor<2xi1>
                %c4_i64_15 = arith.constant 4 : i64
                %60 = tt.splat %c4_i64_15 : i64 -> tensor<2xi64>
                %61 = arith.cmpi sge, %56, %54 : tensor<2xi64>
                %62 = arith.cmpi slt, %56, %60 : tensor<2xi64>
                %63 = arith.andi %61, %62 : tensor<2xi1>
                %64 = arith.andi %59, %63 : tensor<2xi1>
                %c0_i64_16 = arith.constant 0 : i64
                %65 = tt.splat %c0_i64_16 : i64 -> tensor<2xi64>
                %66 = arith.select %63, %56, %65 : tensor<2xi1>, tensor<2xi64>
                %c1_i64_17 = arith.constant 1 : i64
                %67 = tt.splat %c1_i64_17 : i64 -> tensor<2xi64>
                %68 = arith.muli %66, %67 : tensor<2xi64>
                %69 = arith.addi %65, %68 : tensor<2xi64>
                %70 = arith.select %64, %69, %65 : tensor<2xi1>, tensor<2xi64>
                %71 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
                %72 = tt.addptr %71, %70 : tensor<2x!tt.ptr<f32>>, tensor<2xi64>
                tt.store %72, %43, %64 {operand_segment_sizes = array<i32: 1, 1, 1>} : tensor<2x!tt.ptr<f32>>
                tt.return
              }
            }
        "},
        );
    }

    #[test]
    fn test_lower_promoted_read_write_accumulation() {
        // The read-write total is loaded once before the sequential loop, accumulates every input window in the loop,
        // and is stored once after it.
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let definition = accumulate_windows::definition(&ArrayType::new_static(DataType::F32, [4]), &r#type).unwrap();
        assert_eq!(
            definition.interpret(
                vec![Array::vector(vec![1f32, 2.0, 3.0, 4.0]).unwrap(), Array::vector(vec![10f32, 20.0]).unwrap()],
                1,
            ),
            Ok(vec![Array::vector(vec![14f32, 26.0]).unwrap()]),
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
                %c0_i64_0 = arith.constant 0 : i64
                %c2_i64 = arith.constant 2 : i64
                %2 = arith.addi %c0_i64_0, %c2_i64 : i64
                %cst = arith.constant 0.000000e+00 : f32
                %3 = tt.splat %cst : f32 -> tensor<2xf32>
                %true_1 = arith.constant true
                %4 = tt.splat %true_1 : i1 -> tensor<2xi1>
                %5 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                %6 = tt.reshape %5 : tensor<2xi32> -> tensor<2xi32>
                %7 = arith.extsi %6 : tensor<2xi32> to tensor<2xi64>
                %8 = tt.broadcast %7 : tensor<2xi64> -> tensor<2xi64>
                %c2_i64_2 = arith.constant 2 : i64
                %9 = tt.splat %c2_i64_2 : i64 -> tensor<2xi64>
                %10 = arith.cmpi slt, %8, %9 : tensor<2xi64>
                %11 = arith.andi %4, %10 : tensor<2xi1>
                %12 = tt.splat %true : i1 -> tensor<2xi1>
                %13 = arith.andi %11, %12 : tensor<2xi1>
                %c0_i64_3 = arith.constant 0 : i64
                %14 = tt.splat %c0_i64_3 : i64 -> tensor<2xi64>
                %15 = tt.splat %c0_i64_0 : i64 -> tensor<2xi64>
                %16 = arith.addi %15, %8 : tensor<2xi64>
                %17 = tt.splat %2 : i64 -> tensor<2xi64>
                %18 = arith.cmpi slt, %16, %17 : tensor<2xi64>
                %19 = arith.andi %13, %18 : tensor<2xi1>
                %c2_i64_4 = arith.constant 2 : i64
                %20 = tt.splat %c2_i64_4 : i64 -> tensor<2xi64>
                %21 = arith.cmpi sge, %16, %14 : tensor<2xi64>
                %22 = arith.cmpi slt, %16, %20 : tensor<2xi64>
                %23 = arith.andi %21, %22 : tensor<2xi1>
                %24 = arith.andi %19, %23 : tensor<2xi1>
                %c0_i64_5 = arith.constant 0 : i64
                %25 = tt.splat %c0_i64_5 : i64 -> tensor<2xi64>
                %26 = arith.select %23, %16, %25 : tensor<2xi1>, tensor<2xi64>
                %c1_i64 = arith.constant 1 : i64
                %27 = tt.splat %c1_i64 : i64 -> tensor<2xi64>
                %28 = arith.muli %26, %27 : tensor<2xi64>
                %29 = arith.addi %25, %28 : tensor<2xi64>
                %30 = arith.select %24, %29, %25 : tensor<2xi1>, tensor<2xi64>
                %31 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
                %32 = tt.addptr %31, %30 : tensor<2x!tt.ptr<f32>>, tensor<2xi64>
                %33 = tt.load %32, %24, %3 : tensor<2x!tt.ptr<f32>>
                %c0_i64_6 = arith.constant 0 : i64
                %c2_i64_7 = arith.constant 2 : i64
                %c1_i64_8 = arith.constant 1 : i64
                %34 = scf.for %arg2 = %c0_i64_6 to %c2_i64_7 step %c1_i64_8 iter_args(%arg3 = %33) -> (tensor<2xf32>)  \
                    : i64 {
                  %c2_i64_15 = arith.constant 2 : i64
                  %64 = arith.muli %arg2, %c2_i64_15 : i64
                  %c2_i64_16 = arith.constant 2 : i64
                  %65 = arith.addi %64, %c2_i64_16 : i64
                  %c0_i64_17 = arith.constant 0 : i64
                  %c2_i64_18 = arith.constant 2 : i64
                  %66 = arith.addi %c0_i64_17, %c2_i64_18 : i64
                  %cst_19 = arith.constant 0.000000e+00 : f32
                  %67 = tt.splat %cst_19 : f32 -> tensor<2xf32>
                  %true_20 = arith.constant true
                  %68 = tt.splat %true_20 : i1 -> tensor<2xi1>
                  %69 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                  %70 = tt.reshape %69 : tensor<2xi32> -> tensor<2xi32>
                  %71 = arith.extsi %70 : tensor<2xi32> to tensor<2xi64>
                  %72 = tt.broadcast %71 : tensor<2xi64> -> tensor<2xi64>
                  %c2_i64_21 = arith.constant 2 : i64
                  %73 = tt.splat %c2_i64_21 : i64 -> tensor<2xi64>
                  %74 = arith.cmpi slt, %72, %73 : tensor<2xi64>
                  %75 = arith.andi %68, %74 : tensor<2xi1>
                  %76 = tt.splat %true : i1 -> tensor<2xi1>
                  %77 = arith.andi %75, %76 : tensor<2xi1>
                  %c0_i64_22 = arith.constant 0 : i64
                  %78 = tt.splat %c0_i64_22 : i64 -> tensor<2xi64>
                  %79 = tt.splat %64 : i64 -> tensor<2xi64>
                  %80 = arith.addi %79, %72 : tensor<2xi64>
                  %81 = tt.splat %65 : i64 -> tensor<2xi64>
                  %82 = arith.cmpi slt, %80, %81 : tensor<2xi64>
                  %83 = arith.andi %77, %82 : tensor<2xi1>
                  %c4_i64 = arith.constant 4 : i64
                  %84 = tt.splat %c4_i64 : i64 -> tensor<2xi64>
                  %85 = arith.cmpi sge, %80, %78 : tensor<2xi64>
                  %86 = arith.cmpi slt, %80, %84 : tensor<2xi64>
                  %87 = arith.andi %85, %86 : tensor<2xi1>
                  %88 = arith.andi %83, %87 : tensor<2xi1>
                  %c0_i64_23 = arith.constant 0 : i64
                  %89 = tt.splat %c0_i64_23 : i64 -> tensor<2xi64>
                  %90 = arith.select %87, %80, %89 : tensor<2xi1>, tensor<2xi64>
                  %c1_i64_24 = arith.constant 1 : i64
                  %91 = tt.splat %c1_i64_24 : i64 -> tensor<2xi64>
                  %92 = arith.muli %90, %91 : tensor<2xi64>
                  %93 = arith.addi %89, %92 : tensor<2xi64>
                  %94 = arith.select %88, %93, %89 : tensor<2xi1>, tensor<2xi64>
                  %95 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
                  %96 = tt.addptr %95, %94 : tensor<2x!tt.ptr<f32>>, tensor<2xi64>
                  %97 = tt.load %96, %88, %67 : tensor<2x!tt.ptr<f32>>
                  %98 = arith.addf %arg3, %97 : tensor<2xf32>
                  scf.yield %98 : tensor<2xf32>
                }
                %true_9 = arith.constant true
                %35 = tt.splat %true_9 : i1 -> tensor<2xi1>
                %36 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                %37 = tt.reshape %36 : tensor<2xi32> -> tensor<2xi32>
                %38 = arith.extsi %37 : tensor<2xi32> to tensor<2xi64>
                %39 = tt.broadcast %38 : tensor<2xi64> -> tensor<2xi64>
                %c2_i64_10 = arith.constant 2 : i64
                %40 = tt.splat %c2_i64_10 : i64 -> tensor<2xi64>
                %41 = arith.cmpi slt, %39, %40 : tensor<2xi64>
                %42 = arith.andi %35, %41 : tensor<2xi1>
                %43 = tt.splat %true : i1 -> tensor<2xi1>
                %44 = arith.andi %42, %43 : tensor<2xi1>
                %c0_i64_11 = arith.constant 0 : i64
                %45 = tt.splat %c0_i64_11 : i64 -> tensor<2xi64>
                %46 = tt.splat %c0_i64_0 : i64 -> tensor<2xi64>
                %47 = arith.addi %46, %39 : tensor<2xi64>
                %48 = tt.splat %2 : i64 -> tensor<2xi64>
                %49 = arith.cmpi slt, %47, %48 : tensor<2xi64>
                %50 = arith.andi %44, %49 : tensor<2xi1>
                %c2_i64_12 = arith.constant 2 : i64
                %51 = tt.splat %c2_i64_12 : i64 -> tensor<2xi64>
                %52 = arith.cmpi sge, %47, %45 : tensor<2xi64>
                %53 = arith.cmpi slt, %47, %51 : tensor<2xi64>
                %54 = arith.andi %52, %53 : tensor<2xi1>
                %55 = arith.andi %50, %54 : tensor<2xi1>
                %c0_i64_13 = arith.constant 0 : i64
                %56 = tt.splat %c0_i64_13 : i64 -> tensor<2xi64>
                %57 = arith.select %54, %47, %56 : tensor<2xi1>, tensor<2xi64>
                %c1_i64_14 = arith.constant 1 : i64
                %58 = tt.splat %c1_i64_14 : i64 -> tensor<2xi64>
                %59 = arith.muli %57, %58 : tensor<2xi64>
                %60 = arith.addi %56, %59 : tensor<2xi64>
                %61 = arith.select %55, %60, %56 : tensor<2xi1>, tensor<2xi64>
                %62 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
                %63 = tt.addptr %62, %61 : tensor<2x!tt.ptr<f32>>, tensor<2xi64>
                tt.store %63, %34, %55 {operand_segment_sizes = array<i32: 1, 1, 1>} : tensor<2x!tt.ptr<f32>>
                tt.return
              }
            }
        "},
        );
    }

    #[test]
    fn test_lower_promoted_read_write_loop() {
        // The `while` loop carries the promoted value after its ordinary carries and before its iteration counter,
        // and its condition yields the value unchanged.
        let definition = double_repeatedly::definition(&ArrayType::new_static(DataType::F32, [2])).unwrap();
        assert_eq!(
            definition.interpret(vec![Array::vector(vec![1f32, 2.0]).unwrap()], 1),
            Ok(vec![Array::vector(vec![4f32, 8.0]).unwrap()]),
        );
        assert_eq!(
            lowered_module(&definition),
            indoc! {"
            module {
              tt.func public @ryft_kernel(%arg0: !tt.ptr<f32>) {
                %0 = tt.get_program_id x : i32
                %1 = arith.extsi %0 : i32 to i64
                %true = arith.constant true
                %c0_i64 = arith.constant 0 : i64
                %c0_i64_0 = arith.constant 0 : i64
                %c2_i64 = arith.constant 2 : i64
                %2 = arith.addi %c0_i64_0, %c2_i64 : i64
                %cst = arith.constant 0.000000e+00 : f32
                %3 = tt.splat %cst : f32 -> tensor<2xf32>
                %true_1 = arith.constant true
                %4 = tt.splat %true_1 : i1 -> tensor<2xi1>
                %5 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                %6 = tt.reshape %5 : tensor<2xi32> -> tensor<2xi32>
                %7 = arith.extsi %6 : tensor<2xi32> to tensor<2xi64>
                %8 = tt.broadcast %7 : tensor<2xi64> -> tensor<2xi64>
                %c2_i64_2 = arith.constant 2 : i64
                %9 = tt.splat %c2_i64_2 : i64 -> tensor<2xi64>
                %10 = arith.cmpi slt, %8, %9 : tensor<2xi64>
                %11 = arith.andi %4, %10 : tensor<2xi1>
                %12 = tt.splat %true : i1 -> tensor<2xi1>
                %13 = arith.andi %11, %12 : tensor<2xi1>
                %c0_i64_3 = arith.constant 0 : i64
                %14 = tt.splat %c0_i64_3 : i64 -> tensor<2xi64>
                %15 = tt.splat %c0_i64_0 : i64 -> tensor<2xi64>
                %16 = arith.addi %15, %8 : tensor<2xi64>
                %17 = tt.splat %2 : i64 -> tensor<2xi64>
                %18 = arith.cmpi slt, %16, %17 : tensor<2xi64>
                %19 = arith.andi %13, %18 : tensor<2xi1>
                %c2_i64_4 = arith.constant 2 : i64
                %20 = tt.splat %c2_i64_4 : i64 -> tensor<2xi64>
                %21 = arith.cmpi sge, %16, %14 : tensor<2xi64>
                %22 = arith.cmpi slt, %16, %20 : tensor<2xi64>
                %23 = arith.andi %21, %22 : tensor<2xi1>
                %24 = arith.andi %19, %23 : tensor<2xi1>
                %c0_i64_5 = arith.constant 0 : i64
                %25 = tt.splat %c0_i64_5 : i64 -> tensor<2xi64>
                %26 = arith.select %23, %16, %25 : tensor<2xi1>, tensor<2xi64>
                %c1_i64 = arith.constant 1 : i64
                %27 = tt.splat %c1_i64 : i64 -> tensor<2xi64>
                %28 = arith.muli %26, %27 : tensor<2xi64>
                %29 = arith.addi %25, %28 : tensor<2xi64>
                %30 = arith.select %24, %29, %25 : tensor<2xi1>, tensor<2xi64>
                %31 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
                %32 = tt.addptr %31, %30 : tensor<2x!tt.ptr<f32>>, tensor<2xi64>
                %33 = tt.load %32, %24, %3 : tensor<2x!tt.ptr<f32>>
                %c0_i64_6 = arith.constant 0 : i64
                %c2_i64_7 = arith.constant 2 : i64
                %34 = arith.addi %c0_i64_6, %c2_i64_7 : i64
                %c0_i64_8 = arith.constant 0 : i64
                %c0_i64_9 = arith.constant 0 : i64
                %35:3 = scf.while (%arg1 = %c0_i64_8, %arg2 = %33, %arg3 = %c0_i64_9) : (i64, tensor<2xf32>, i64) -> \
                    (i64, tensor<2xf32>, i64) {
                  %c2_i64_16 = arith.constant 2 : i64
                  %65 = arith.cmpi slt, %arg3, %c2_i64_16 : i64
                  %66:2 = scf.if %65 -> (i1, tensor<2xf32>) {
                    %c2_i64_17 = arith.constant 2 : i64
                    %67 = arith.cmpi slt, %arg1, %c2_i64_17 : i64
                    scf.yield %67, %arg2 : i1, tensor<2xf32>
                  } else {
                    %false = arith.constant false
                    scf.yield %false, %arg2 : i1, tensor<2xf32>
                  }
                  scf.condition(%66#0) %arg1, %66#1, %arg3 : i64, tensor<2xf32>, i64
                } do {
                ^bb0(%arg1: i64, %arg2: tensor<2xf32>, %arg3: i64):
                  %c1_i64_16 = arith.constant 1 : i64
                  %65 = arith.addf %arg2, %arg2 : tensor<2xf32>
                  %66 = arith.addi %arg1, %c1_i64_16 : i64
                  %c1_i64_17 = arith.constant 1 : i64
                  %67 = arith.addi %arg3, %c1_i64_17 : i64
                  scf.yield %66, %65, %67 : i64, tensor<2xf32>, i64
                }
                %true_10 = arith.constant true
                %36 = tt.splat %true_10 : i1 -> tensor<2xi1>
                %37 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                %38 = tt.reshape %37 : tensor<2xi32> -> tensor<2xi32>
                %39 = arith.extsi %38 : tensor<2xi32> to tensor<2xi64>
                %40 = tt.broadcast %39 : tensor<2xi64> -> tensor<2xi64>
                %c2_i64_11 = arith.constant 2 : i64
                %41 = tt.splat %c2_i64_11 : i64 -> tensor<2xi64>
                %42 = arith.cmpi slt, %40, %41 : tensor<2xi64>
                %43 = arith.andi %36, %42 : tensor<2xi1>
                %44 = tt.splat %true : i1 -> tensor<2xi1>
                %45 = arith.andi %43, %44 : tensor<2xi1>
                %c0_i64_12 = arith.constant 0 : i64
                %46 = tt.splat %c0_i64_12 : i64 -> tensor<2xi64>
                %47 = tt.splat %c0_i64_0 : i64 -> tensor<2xi64>
                %48 = arith.addi %47, %40 : tensor<2xi64>
                %49 = tt.splat %2 : i64 -> tensor<2xi64>
                %50 = arith.cmpi slt, %48, %49 : tensor<2xi64>
                %51 = arith.andi %45, %50 : tensor<2xi1>
                %c2_i64_13 = arith.constant 2 : i64
                %52 = tt.splat %c2_i64_13 : i64 -> tensor<2xi64>
                %53 = arith.cmpi sge, %48, %46 : tensor<2xi64>
                %54 = arith.cmpi slt, %48, %52 : tensor<2xi64>
                %55 = arith.andi %53, %54 : tensor<2xi1>
                %56 = arith.andi %51, %55 : tensor<2xi1>
                %c0_i64_14 = arith.constant 0 : i64
                %57 = tt.splat %c0_i64_14 : i64 -> tensor<2xi64>
                %58 = arith.select %55, %48, %57 : tensor<2xi1>, tensor<2xi64>
                %c1_i64_15 = arith.constant 1 : i64
                %59 = tt.splat %c1_i64_15 : i64 -> tensor<2xi64>
                %60 = arith.muli %58, %59 : tensor<2xi64>
                %61 = arith.addi %57, %60 : tensor<2xi64>
                %62 = arith.select %56, %61, %57 : tensor<2xi1>, tensor<2xi64>
                %63 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
                %64 = tt.addptr %63, %62 : tensor<2x!tt.ptr<f32>>, tensor<2xi64>
                tt.store %64, %35#1, %56 {operand_segment_sizes = array<i32: 1, 1, 1>} : tensor<2x!tt.ptr<f32>>
                tt.return
              }
            }
        "},
        );
    }

    #[test]
    fn test_lower_rejects_partial_promoted_windows() {
        // Only whole windows are promoted, so the second write of the output through a slice keeps its reason.
        let definition = overwrite_first::definition(&ArrayType::new_static(DataType::F32, [2])).unwrap();
        assert_eq!(
            rejection(&definition),
            (
                "reference_write",
                "repeated writes of one window need a CTA barrier, which ryft-mlir does not expose".to_owned(),
            ),
        );
    }

    #[test]
    fn test_lower_rejects_carried_windows() {
        // The output window repeats along the outer sequential axis but changes along the inner one, so no SSA value
        // can hold it across the outer loop, and the stores of two of its steps could land in either order.
        let definition = repeated_copy::definition(&ArrayType::new_static(DataType::F32, [4])).unwrap();
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
        // value into `output`. Promoting `values` for its read after a write leaves one entry load and one exit store,
        // which are just as unordered.
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let reason = (
            "reference_write",
            "values read from a read-write parameter may reach memory only elementwise through its own write"
                .to_owned(),
        );
        assert_eq!(rejection(&copy_then_update::definition(&r#type).unwrap()), reason);
        assert_eq!(rejection(&read_after_write::definition(&r#type).unwrap()), reason);
    }

    #[test]
    fn test_lower_rejects_read_write_values_entering_control_flow() {
        // The value read before the loop enters it as a carry, and values read from a read-write parameter may not
        // cross region boundaries.
        let definition = double_before_loop::definition(&ArrayType::new_static(DataType::F32, [2])).unwrap();
        assert_eq!(
            rejection(&definition),
            (
                "while",
                "values read from a read-write parameter may reach memory only elementwise through its own write"
                    .to_owned(),
            ),
        );
    }

    #[test]
    fn test_lower_rejects_mixed_atomic_accesses() {
        let definition = read_atomic_total::definition(
            &ArrayType::new_static(DataType::F32, [2]),
            &ArrayType::scalar(DataType::F32),
        )
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
    fn test_lower_tile_load() {
        // A masked window at element 3 of `[3, 5, 7, 9]` loads its valid element and takes the padding value for the
        // element beyond the source view.
        let source = Array::vector(vec![3f32, 5.0, 7.0, 9.0]).unwrap();
        let definition = tile_load_definition(BoundaryPolicy::Masked, 3, vec![]);
        assert_eq!(definition.interpret(vec![source.clone()], 1), Ok(vec![Array::vector(vec![9f32, -1.0]).unwrap()]));
        assert_eq!(
            lowered_module(&definition),
            indoc! {"
            module {
              tt.func public @ryft_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
                %0 = tt.get_program_id x : i32
                %1 = arith.extsi %0 : i32 to i64
                %true = arith.constant true
                %c0_i64 = arith.constant 0 : i64
                %c4_i64 = arith.constant 4 : i64
                %2 = arith.addi %c0_i64, %c4_i64 : i64
                %c0_i64_0 = arith.constant 0 : i64
                %c2_i64 = arith.constant 2 : i64
                %3 = arith.addi %c0_i64_0, %c2_i64 : i64
                %c3_i64 = arith.constant 3 : i64
                %c-1082130432_i32 = arith.constant -1082130432 : i32
                %4 = arith.bitcast %c-1082130432_i32 : i32 to f32
                %5 = arith.addi %c0_i64, %c3_i64 : i64
                %6 = tt.splat %4 : f32 -> tensor<2xf32>
                %true_1 = arith.constant true
                %7 = tt.splat %true_1 : i1 -> tensor<2xi1>
                %8 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                %9 = tt.reshape %8 : tensor<2xi32> -> tensor<2xi32>
                %10 = arith.extsi %9 : tensor<2xi32> to tensor<2xi64>
                %11 = tt.broadcast %10 : tensor<2xi64> -> tensor<2xi64>
                %c2_i64_2 = arith.constant 2 : i64
                %12 = tt.splat %c2_i64_2 : i64 -> tensor<2xi64>
                %13 = arith.cmpi slt, %11, %12 : tensor<2xi64>
                %14 = arith.andi %7, %13 : tensor<2xi1>
                %15 = tt.splat %true : i1 -> tensor<2xi1>
                %16 = arith.andi %14, %15 : tensor<2xi1>
                %c0_i64_3 = arith.constant 0 : i64
                %17 = tt.splat %c0_i64_3 : i64 -> tensor<2xi64>
                %18 = tt.splat %5 : i64 -> tensor<2xi64>
                %19 = arith.addi %18, %11 : tensor<2xi64>
                %20 = tt.splat %2 : i64 -> tensor<2xi64>
                %21 = arith.cmpi slt, %19, %20 : tensor<2xi64>
                %22 = arith.andi %16, %21 : tensor<2xi1>
                %c4_i64_4 = arith.constant 4 : i64
                %23 = tt.splat %c4_i64_4 : i64 -> tensor<2xi64>
                %24 = arith.cmpi sge, %19, %17 : tensor<2xi64>
                %25 = arith.cmpi slt, %19, %23 : tensor<2xi64>
                %26 = arith.andi %24, %25 : tensor<2xi1>
                %27 = arith.andi %22, %26 : tensor<2xi1>
                %c0_i64_5 = arith.constant 0 : i64
                %28 = tt.splat %c0_i64_5 : i64 -> tensor<2xi64>
                %29 = arith.select %26, %19, %28 : tensor<2xi1>, tensor<2xi64>
                %c1_i64 = arith.constant 1 : i64
                %30 = tt.splat %c1_i64 : i64 -> tensor<2xi64>
                %31 = arith.muli %29, %30 : tensor<2xi64>
                %32 = arith.addi %28, %31 : tensor<2xi64>
                %33 = arith.select %27, %32, %28 : tensor<2xi1>, tensor<2xi64>
                %34 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
                %35 = tt.addptr %34, %33 : tensor<2x!tt.ptr<f32>>, tensor<2xi64>
                %36 = tt.load %35, %27, %6 : tensor<2x!tt.ptr<f32>>
                %true_6 = arith.constant true
                %37 = tt.splat %true_6 : i1 -> tensor<2xi1>
                %38 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
                %39 = tt.reshape %38 : tensor<2xi32> -> tensor<2xi32>
                %40 = arith.extsi %39 : tensor<2xi32> to tensor<2xi64>
                %41 = tt.broadcast %40 : tensor<2xi64> -> tensor<2xi64>
                %c2_i64_7 = arith.constant 2 : i64
                %42 = tt.splat %c2_i64_7 : i64 -> tensor<2xi64>
                %43 = arith.cmpi slt, %41, %42 : tensor<2xi64>
                %44 = arith.andi %37, %43 : tensor<2xi1>
                %45 = tt.splat %true : i1 -> tensor<2xi1>
                %46 = arith.andi %44, %45 : tensor<2xi1>
                %c0_i64_8 = arith.constant 0 : i64
                %47 = tt.splat %c0_i64_8 : i64 -> tensor<2xi64>
                %48 = tt.splat %c0_i64_0 : i64 -> tensor<2xi64>
                %49 = arith.addi %48, %41 : tensor<2xi64>
                %50 = tt.splat %3 : i64 -> tensor<2xi64>
                %51 = arith.cmpi slt, %49, %50 : tensor<2xi64>
                %52 = arith.andi %46, %51 : tensor<2xi1>
                %c2_i64_9 = arith.constant 2 : i64
                %53 = tt.splat %c2_i64_9 : i64 -> tensor<2xi64>
                %54 = arith.cmpi sge, %49, %47 : tensor<2xi64>
                %55 = arith.cmpi slt, %49, %53 : tensor<2xi64>
                %56 = arith.andi %54, %55 : tensor<2xi1>
                %57 = arith.andi %52, %56 : tensor<2xi1>
                %c0_i64_10 = arith.constant 0 : i64
                %58 = tt.splat %c0_i64_10 : i64 -> tensor<2xi64>
                %59 = arith.select %56, %49, %58 : tensor<2xi1>, tensor<2xi64>
                %c1_i64_11 = arith.constant 1 : i64
                %60 = tt.splat %c1_i64_11 : i64 -> tensor<2xi64>
                %61 = arith.muli %59, %60 : tensor<2xi64>
                %62 = arith.addi %58, %61 : tensor<2xi64>
                %63 = arith.select %57, %62, %58 : tensor<2xi1>, tensor<2xi64>
                %64 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
                %65 = tt.addptr %64, %63 : tensor<2x!tt.ptr<f32>>, tensor<2xi64>
                tt.store %65, %36, %57 {operand_segment_sizes = array<i32: 1, 1, 1>} : tensor<2x!tt.ptr<f32>>
                tt.return
              }
            }
        "},
        );

        // The same window under the in-bounds policy fails on the host. Triton cannot report that failure at run time,
        // so admission rejects every in-bounds load whose start bounds do not prove the window valid.
        let definition = tile_load_definition(BoundaryPolicy::InBounds, 3, vec![]);
        assert_eq!(
            definition.interpret(vec![source], 1),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{TILE_LOAD_OPERATION_NAME}` in-bounds window exceeds the source extent",
            )))),
        );
        assert_eq!(rejection(&definition), (TILE_LOAD_OPERATION_NAME, IN_BOUNDS_REASON.to_owned()));
    }

    #[test]
    fn test_lower_tile_load_boundaries() {
        let source = Array::vector(vec![3f32, 5.0, 7.0, 9.0]).unwrap();

        // An in-bounds load whose start bounds prove a wholly valid window lowers like the masked load of that window.
        let definition = tile_load_definition(BoundaryPolicy::InBounds, 2, vec![]);
        assert_eq!(definition.interpret(vec![source.clone()], 1), Ok(vec![Array::vector(vec![7f32, 9.0]).unwrap()]));
        assert_eq!(
            lowered_module(&definition),
            lowered_module(&tile_load_definition(BoundaryPolicy::Masked, 2, vec![])),
        );

        // Bounds apply to the selected view `[5, 7, 9]` rather than to the whole source, so the window at element 2 of
        // that view fails even though it lies inside the source.
        let slice = || vec![ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 3, 1)] }];
        let definition = tile_load_definition(BoundaryPolicy::InBounds, 1, slice());
        assert_eq!(definition.interpret(vec![source.clone()], 1), Ok(vec![Array::vector(vec![7f32, 9.0]).unwrap()]));
        assert_eq!(
            lowered_module(&definition),
            lowered_module(&tile_load_definition(BoundaryPolicy::Masked, 1, slice())),
        );
        let definition = tile_load_definition(BoundaryPolicy::InBounds, 2, slice());
        assert_eq!(
            definition.interpret(vec![source.clone()], 1),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{TILE_LOAD_OPERATION_NAME}` in-bounds window exceeds the source extent",
            )))),
        );
        assert_eq!(rejection(&definition), (TILE_LOAD_OPERATION_NAME, IN_BOUNDS_REASON.to_owned()));

        // A masked window that starts beyond the view consists entirely of padding.
        let definition = tile_load_definition(BoundaryPolicy::Masked, 4, vec![]);
        assert_eq!(definition.interpret(vec![source.clone()], 1), Ok(vec![Array::vector(vec![-1f32, -1.0]).unwrap()]));
        lowered_module(&definition);

        // A window limit beyond the canonical dimension width fails on the host under either policy, and Triton rejects
        // its start before it reaches the load because the start exceeds the signed 32-bit indexing range.
        let definition = tile_load_definition(BoundaryPolicy::Masked, MAX_DIMENSION_EXTENT - 1, vec![]);
        assert_eq!(
            definition.interpret(vec![source], 1),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{TILE_LOAD_OPERATION_NAME}` window limit exceeds the canonical dimension width",
            )))),
        );
        assert_eq!(
            rejection(&definition),
            ("dimension", "dimension bounds must fit the signed 32-bit indexing domain".to_owned()),
        );
    }

    #[test]
    fn test_lower_tile_load_unreachable() {
        // Admission cannot tell whether an in-bounds load runs, so it rejects the load even where the host never
        // executes it. This is a missing capability of the adapter, unlike a malformed load, which fails when it is
        // staged and never reaches an adapter.
        let source = Array::vector(vec![3f32, 5.0, 7.0, 9.0]).unwrap();
        let [empty_grid, untaken_branch] = unreachable_tile_load_definitions().try_into().ok().unwrap();
        let output = Array::vector(vec![1f32, 2.0]).unwrap();
        assert_eq!(empty_grid.interpret(vec![source.clone(), output.clone()], 1), Ok(vec![output]));
        assert_eq!(rejection(&empty_grid), (TILE_LOAD_OPERATION_NAME, IN_BOUNDS_REASON.to_owned()));
        assert_eq!(untaken_branch.interpret(vec![source], 1), Ok(vec![Array::vector(vec![-1f32, -1.0]).unwrap()]));
        assert_eq!(rejection(&untaken_branch), (TILE_LOAD_OPERATION_NAME, IN_BOUNDS_REASON.to_owned()));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = builder.add_input(
            whole_array_parameter(ArrayType::new_static(DataType::F32, [4]), KernelParameterAccess::ReadOnly)
                .unwrap()
                .body_type(),
        );
        let start = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()));
        let other = builder.add_constant(ArrayIrValue::Array(Array::scalar(-1f32).unwrap()));
        let operation = TileLoadOperation::new(vec![2, 2], BoundaryPolicy::InBounds).unwrap();
        assert_eq!(
            builder.add_instruction(operation, vec![], vec![reference, start, start, other], None),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{TILE_LOAD_OPERATION_NAME}` source requires a static shape with the block rank",
            )))),
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
    fn test_lower_masked_swap_used_old_value() {
        // The first output is promoted, so the swap takes its old value from the promoted value, with `other` at the
        // padding element and at the physical padding, and the second output stores that old value.
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
            definition.interpret(vec![], 1),
            Ok(vec![Array::vector(vec![5f32, 5.0]).unwrap(), Array::vector(vec![5f32, 5.0, -1.0]).unwrap()]),
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
                %c0_i64 = arith.constant 0 : i64
                %c3_i64 = arith.constant 3 : i64
                %4 = arith.muli %2, %c3_i64 : i64
                %c3_i64_0 = arith.constant 3 : i64
                %5 = arith.addi %4, %c3_i64_0 : i64
                %cst = arith.constant 0.000000e+00 : f32
                %6 = tt.splat %cst : f32 -> tensor<4xf32>
                %c3_i64_1 = arith.constant 3 : i64
                %7 = arith.muli %2, %c3_i64_1 : i64
                %c3_i64_2 = arith.constant 3 : i64
                %8 = arith.addi %7, %c3_i64_2 : i64
                %c3_i64_3 = arith.constant 3 : i64
                %9 = arith.muli %2, %c3_i64_3 : i64
                %c3_i64_4 = arith.constant 3 : i64
                %10 = arith.addi %9, %c3_i64_4 : i64
                %c1084227584_i32 = arith.constant 1084227584 : i32
                %11 = arith.bitcast %c1084227584_i32 : i32 to f32
                %12 = tt.splat %11 : f32 -> tensor<4xf32>
                %true_5 = arith.constant true
                %13 = tt.splat %true_5 : i1 -> tensor<4xi1>
                %14 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
                %15 = tt.reshape %14 : tensor<4xi32> -> tensor<4xi32>
                %16 = arith.extsi %15 : tensor<4xi32> to tensor<4xi64>
                %17 = tt.broadcast %16 : tensor<4xi64> -> tensor<4xi64>
                %c3_i64_6 = arith.constant 3 : i64
                %18 = tt.splat %c3_i64_6 : i64 -> tensor<4xi64>
                %19 = arith.cmpi slt, %17, %18 : tensor<4xi64>
                %20 = arith.andi %13, %19 : tensor<4xi1>
                %21 = tt.splat %true : i1 -> tensor<4xi1>
                %22 = arith.andi %20, %21 : tensor<4xi1>
                %c0_i64_7 = arith.constant 0 : i64
                %23 = tt.splat %c0_i64_7 : i64 -> tensor<4xi64>
                %24 = tt.splat %7 : i64 -> tensor<4xi64>
                %25 = arith.addi %24, %17 : tensor<4xi64>
                %26 = tt.splat %8 : i64 -> tensor<4xi64>
                %27 = arith.cmpi slt, %25, %26 : tensor<4xi64>
                %28 = arith.andi %22, %27 : tensor<4xi1>
                %c2_i64 = arith.constant 2 : i64
                %29 = tt.splat %c2_i64 : i64 -> tensor<4xi64>
                %30 = arith.cmpi sge, %25, %23 : tensor<4xi64>
                %31 = arith.cmpi slt, %25, %29 : tensor<4xi64>
                %32 = arith.andi %30, %31 : tensor<4xi1>
                %33 = arith.andi %28, %32 : tensor<4xi1>
                %true_8 = arith.constant true
                %34 = tt.splat %true_8 : i1 -> tensor<4xi1>
                %35 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
                %36 = tt.reshape %35 : tensor<4xi32> -> tensor<4xi32>
                %37 = arith.extsi %36 : tensor<4xi32> to tensor<4xi64>
                %38 = tt.broadcast %37 : tensor<4xi64> -> tensor<4xi64>
                %c3_i64_9 = arith.constant 3 : i64
                %39 = tt.splat %c3_i64_9 : i64 -> tensor<4xi64>
                %40 = arith.cmpi slt, %38, %39 : tensor<4xi64>
                %41 = arith.andi %34, %40 : tensor<4xi1>
                %42 = tt.splat %true : i1 -> tensor<4xi1>
                %43 = arith.andi %41, %42 : tensor<4xi1>
                %c0_i64_10 = arith.constant 0 : i64
                %44 = tt.splat %c0_i64_10 : i64 -> tensor<4xi64>
                %45 = tt.splat %7 : i64 -> tensor<4xi64>
                %46 = arith.addi %45, %38 : tensor<4xi64>
                %47 = tt.splat %8 : i64 -> tensor<4xi64>
                %48 = arith.cmpi slt, %46, %47 : tensor<4xi64>
                %49 = arith.andi %43, %48 : tensor<4xi1>
                %c2_i64_11 = arith.constant 2 : i64
                %50 = tt.splat %c2_i64_11 : i64 -> tensor<4xi64>
                %51 = arith.cmpi sge, %46, %44 : tensor<4xi64>
                %52 = arith.cmpi slt, %46, %50 : tensor<4xi64>
                %53 = arith.andi %51, %52 : tensor<4xi1>
                %54 = arith.andi %49, %53 : tensor<4xi1>
                %55 = arith.andi %54, %33 : tensor<4xi1>
                %56 = arith.select %55, %12, %6 : tensor<4xi1>, tensor<4xf32>
                %c-1082130432_i32 = arith.constant -1082130432 : i32
                %57 = arith.bitcast %c-1082130432_i32 : i32 to f32
                %58 = tt.splat %57 : f32 -> tensor<4xf32>
                %true_12 = arith.constant true
                %59 = tt.splat %true_12 : i1 -> tensor<4xi1>
                %60 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
                %61 = tt.reshape %60 : tensor<4xi32> -> tensor<4xi32>
                %62 = arith.extsi %61 : tensor<4xi32> to tensor<4xi64>
                %63 = tt.broadcast %62 : tensor<4xi64> -> tensor<4xi64>
                %c3_i64_13 = arith.constant 3 : i64
                %64 = tt.splat %c3_i64_13 : i64 -> tensor<4xi64>
                %65 = arith.cmpi slt, %63, %64 : tensor<4xi64>
                %66 = arith.andi %59, %65 : tensor<4xi1>
                %67 = tt.splat %true : i1 -> tensor<4xi1>
                %68 = arith.andi %66, %67 : tensor<4xi1>
                %c0_i64_14 = arith.constant 0 : i64
                %69 = tt.splat %c0_i64_14 : i64 -> tensor<4xi64>
                %70 = tt.splat %7 : i64 -> tensor<4xi64>
                %71 = arith.addi %70, %63 : tensor<4xi64>
                %72 = tt.splat %8 : i64 -> tensor<4xi64>
                %73 = arith.cmpi slt, %71, %72 : tensor<4xi64>
                %74 = arith.andi %68, %73 : tensor<4xi1>
                %c2_i64_15 = arith.constant 2 : i64
                %75 = tt.splat %c2_i64_15 : i64 -> tensor<4xi64>
                %76 = arith.cmpi sge, %71, %69 : tensor<4xi64>
                %77 = arith.cmpi slt, %71, %75 : tensor<4xi64>
                %78 = arith.andi %76, %77 : tensor<4xi1>
                %79 = arith.andi %74, %78 : tensor<4xi1>
                %80 = arith.andi %79, %33 : tensor<4xi1>
                %81 = arith.select %80, %56, %58 : tensor<4xi1>, tensor<4xf32>
                %82 = arith.select %80, %12, %56 : tensor<4xi1>, tensor<4xf32>
                %true_16 = arith.constant true
                %83 = tt.splat %true_16 : i1 -> tensor<4xi1>
                %84 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
                %85 = tt.reshape %84 : tensor<4xi32> -> tensor<4xi32>
                %86 = arith.extsi %85 : tensor<4xi32> to tensor<4xi64>
                %87 = tt.broadcast %86 : tensor<4xi64> -> tensor<4xi64>
                %c3_i64_17 = arith.constant 3 : i64
                %88 = tt.splat %c3_i64_17 : i64 -> tensor<4xi64>
                %89 = arith.cmpi slt, %87, %88 : tensor<4xi64>
                %90 = arith.andi %83, %89 : tensor<4xi1>
                %91 = tt.splat %true : i1 -> tensor<4xi1>
                %92 = arith.andi %90, %91 : tensor<4xi1>
                %c0_i64_18 = arith.constant 0 : i64
                %93 = tt.splat %c0_i64_18 : i64 -> tensor<4xi64>
                %94 = tt.splat %9 : i64 -> tensor<4xi64>
                %95 = arith.addi %94, %87 : tensor<4xi64>
                %96 = tt.splat %10 : i64 -> tensor<4xi64>
                %97 = arith.cmpi slt, %95, %96 : tensor<4xi64>
                %98 = arith.andi %92, %97 : tensor<4xi1>
                %c3_i64_19 = arith.constant 3 : i64
                %99 = tt.splat %c3_i64_19 : i64 -> tensor<4xi64>
                %100 = arith.cmpi sge, %95, %93 : tensor<4xi64>
                %101 = arith.cmpi slt, %95, %99 : tensor<4xi64>
                %102 = arith.andi %100, %101 : tensor<4xi1>
                %103 = arith.andi %98, %102 : tensor<4xi1>
                %c0_i64_20 = arith.constant 0 : i64
                %104 = tt.splat %c0_i64_20 : i64 -> tensor<4xi64>
                %105 = arith.select %102, %95, %104 : tensor<4xi1>, tensor<4xi64>
                %c1_i64_21 = arith.constant 1 : i64
                %106 = tt.splat %c1_i64_21 : i64 -> tensor<4xi64>
                %107 = arith.muli %105, %106 : tensor<4xi64>
                %108 = arith.addi %104, %107 : tensor<4xi64>
                %109 = arith.select %103, %108, %104 : tensor<4xi1>, tensor<4xi64>
                %110 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
                %111 = tt.addptr %110, %109 : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
                tt.store %111, %81, %103 {operand_segment_sizes = array<i32: 1, 1, 1>} : tensor<4x!tt.ptr<f32>>
                %true_22 = arith.constant true
                %112 = tt.splat %true_22 : i1 -> tensor<4xi1>
                %113 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
                %114 = tt.reshape %113 : tensor<4xi32> -> tensor<4xi32>
                %115 = arith.extsi %114 : tensor<4xi32> to tensor<4xi64>
                %116 = tt.broadcast %115 : tensor<4xi64> -> tensor<4xi64>
                %c3_i64_23 = arith.constant 3 : i64
                %117 = tt.splat %c3_i64_23 : i64 -> tensor<4xi64>
                %118 = arith.cmpi slt, %116, %117 : tensor<4xi64>
                %119 = arith.andi %112, %118 : tensor<4xi1>
                %120 = tt.splat %true : i1 -> tensor<4xi1>
                %121 = arith.andi %119, %120 : tensor<4xi1>
                %c0_i64_24 = arith.constant 0 : i64
                %122 = tt.splat %c0_i64_24 : i64 -> tensor<4xi64>
                %123 = tt.splat %4 : i64 -> tensor<4xi64>
                %124 = arith.addi %123, %116 : tensor<4xi64>
                %125 = tt.splat %5 : i64 -> tensor<4xi64>
                %126 = arith.cmpi slt, %124, %125 : tensor<4xi64>
                %127 = arith.andi %121, %126 : tensor<4xi1>
                %c2_i64_25 = arith.constant 2 : i64
                %128 = tt.splat %c2_i64_25 : i64 -> tensor<4xi64>
                %129 = arith.cmpi sge, %124, %122 : tensor<4xi64>
                %130 = arith.cmpi slt, %124, %128 : tensor<4xi64>
                %131 = arith.andi %129, %130 : tensor<4xi1>
                %132 = arith.andi %127, %131 : tensor<4xi1>
                %c0_i64_26 = arith.constant 0 : i64
                %133 = tt.splat %c0_i64_26 : i64 -> tensor<4xi64>
                %134 = arith.select %131, %124, %133 : tensor<4xi1>, tensor<4xi64>
                %c1_i64_27 = arith.constant 1 : i64
                %135 = tt.splat %c1_i64_27 : i64 -> tensor<4xi64>
                %136 = arith.muli %134, %135 : tensor<4xi64>
                %137 = arith.addi %133, %136 : tensor<4xi64>
                %138 = arith.select %132, %137, %133 : tensor<4xi1>, tensor<4xi64>
                %139 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
                %140 = tt.addptr %139, %138 : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
                tt.store %140, %82, %132 {operand_segment_sizes = array<i32: 1, 1, 1>} : tensor<4x!tt.ptr<f32>>
                tt.return
              }
            }
        "},
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
                %cst = arith.constant 0.000000e+00 : f32
                %11 = tt.splat %cst : f32 -> tensor<4xf32>
                %true_5 = arith.constant true
                %12 = tt.splat %true_5 : i1 -> tensor<4xi1>
                %13 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
                %14 = tt.reshape %13 : tensor<4xi32> -> tensor<4xi32>
                %15 = arith.extsi %14 : tensor<4xi32> to tensor<4xi64>
                %16 = tt.broadcast %15 : tensor<4xi64> -> tensor<4xi64>
                %c3_i64_6 = arith.constant 3 : i64
                %17 = tt.splat %c3_i64_6 : i64 -> tensor<4xi64>
                %18 = arith.cmpi slt, %16, %17 : tensor<4xi64>
                %19 = arith.andi %12, %18 : tensor<4xi1>
                %20 = tt.splat %true : i1 -> tensor<4xi1>
                %21 = arith.andi %19, %20 : tensor<4xi1>
                %c0_i64 = arith.constant 0 : i64
                %22 = tt.splat %c0_i64 : i64 -> tensor<4xi64>
                %23 = tt.splat %6 : i64 -> tensor<4xi64>
                %24 = arith.addi %23, %16 : tensor<4xi64>
                %25 = tt.splat %7 : i64 -> tensor<4xi64>
                %26 = arith.cmpi slt, %24, %25 : tensor<4xi64>
                %27 = arith.andi %21, %26 : tensor<4xi1>
                %c3_i64_7 = arith.constant 3 : i64
                %28 = tt.splat %c3_i64_7 : i64 -> tensor<4xi64>
                %29 = arith.cmpi sge, %24, %22 : tensor<4xi64>
                %30 = arith.cmpi slt, %24, %28 : tensor<4xi64>
                %31 = arith.andi %29, %30 : tensor<4xi1>
                %32 = arith.andi %27, %31 : tensor<4xi1>
                %c0_i64_8 = arith.constant 0 : i64
                %33 = tt.splat %c0_i64_8 : i64 -> tensor<4xi64>
                %34 = arith.select %31, %24, %33 : tensor<4xi1>, tensor<4xi64>
                %c1_i64_9 = arith.constant 1 : i64
                %35 = tt.splat %c1_i64_9 : i64 -> tensor<4xi64>
                %36 = arith.muli %34, %35 : tensor<4xi64>
                %37 = arith.addi %33, %36 : tensor<4xi64>
                %38 = arith.select %32, %37, %33 : tensor<4xi1>, tensor<4xi64>
                %39 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
                %40 = tt.addptr %39, %38 : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
                %41 = tt.load %40, %32, %11 : tensor<4x!tt.ptr<f32>>
                %42 = tt.splat %10 : f32 -> tensor<4xf32>
                %43 = arith.cmpf ogt, %41, %42 : tensor<4xf32>
                %c0_i32_10 = arith.constant 0 : i32
                %44 = arith.bitcast %c0_i32_10 : i32 to f32
                %45 = tt.splat %44 : f32 -> tensor<4xf32>
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
                %75 = arith.andi %66, %43 : tensor<4xi1>
                %76 = tt.load %74, %75, %45 : tensor<4x!tt.ptr<f32>>
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
                promotions: Vec::new(),
                windows: Vec::new(),
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
            promotions: Vec::new(),
            windows: Vec::new(),
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
    fn test_lowering_array_dot() {
        let matrix = ArrayType::new_static(DataType::F32, [2, 2]);
        let dot = |operation: DotOperation| lowered_array(ArrayOperation::Dot(operation), vec![matrix.clone(); 2]);
        assert!(matches!(dot(DotOperation::matmul()), Ok((_, _, Some(_)))));

        // Every precision that rounds the operands or relaxes the F32 accumulation is rejected.
        let f32 = DotAlgorithm::new(DataType::F32, DataType::F32, DataType::F32);
        for precision in [
            DotPrecision::from(DotInputPrecision::High),
            DotPrecision::from((DotInputPrecision::Highest, DotInputPrecision::High)),
            DotPrecision::from(DotAlgorithmPreset::BF16_BF16_F32),
            DotPrecision::from(DotAlgorithmPreset::TF32_TF32_F32),
            DotPrecision::from(DotAlgorithmPreset::BF16_BF16_F32_X3),
            DotPrecision::from(f32.with_imprecise_accumulation(true)),
            DotPrecision::from(f32.with_component_counts(2, 2).with_primitive_operation_count(3)),
        ] {
            assert!(matches!(
                dot(DotOperation::matmul().with_precision(precision)),
                Err(Error::Unsupported { operation: "dot", reason }) if reason == format!(
                    "precision `{precision}` requires conversions or accumulation other than an IEEE F32 dot",
                ),
            ));
        }

        // An F32 accumulation type is the default for F32 operands. Every other one is also the output type, which no
        // Triton tile represents.
        assert_eq!(
            dot(DotOperation::matmul().with_accumulation_type(DataType::F32)).ok(),
            dot(DotOperation::matmul()).ok(),
        );
        assert!(matches!(
            dot(DotOperation::matmul().with_accumulation_type(DataType::F16)),
            Err(Error::Unsupported { operation: "array", reason })
                if reason == "only F32, I64 and Boolean tile values are supported",
        ));
        assert!(matches!(
            lowered_array(
                ArrayOperation::Dot(DotOperation::matmul().with_accumulation_type(DataType::I64)),
                vec![ArrayType::new_static(DataType::I64, [2, 2]); 2],
            ),
            Err(Error::Unsupported { operation: "dot", reason })
                if reason == "operands, accumulation, and results must be F32",
        ));
        let batched = DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]));
        assert!(matches!(
            lowered_array(ArrayOperation::Dot(batched), vec![ArrayType::new_static(DataType::F32, [2, 2, 2]); 2]),
            Err(Error::Unsupported { operation: "dot", reason })
                if reason == "only rank-2 matrix multiplications are supported",
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
            promotions: Vec::new(),
            windows: Vec::new(),
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
            promotions: Vec::new(),
            windows: Vec::new(),
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
            promotions: Vec::new(),
            windows: Vec::new(),
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
