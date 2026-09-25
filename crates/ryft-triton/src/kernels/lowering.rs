//! Typed TTIR construction from the verified portable kernel arena.

use std::collections::HashMap;

use ryft_core::kernels::{GridExecution, KernelOperation, KernelSchedule, VerifiedKernel};
use ryft_core::{
    Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReferenceTransform,
    ArrayReferenceTransformIndex, ArrayType, Atom, ComparisonDirection, DataType, Dimension, DimensionOperation,
    DimensionValue, DotDimensionNumbers, Layout, Memory, Operation as CoreOperation, ReductionKind,
    ReferenceAccessOperation, RegionRef, Typed, validated_reference_access_descriptors,
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
    if !call.aliases().is_empty() {
        return Err(unsupported("kernel_call", "read-write alias parameters are not yet supported"));
    }
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
    let mut extents = Vec::new();
    for dimension in call.grid().dimensions() {
        if dimension.execution() != GridExecution::Parallel {
            return Err(unsupported("kernel_call", "sequential grid axes are not yet supported"));
        }
        let Dimension::Static(extent) = dimension.extent() else {
            return Err(unsupported("kernel_call", "grid extents must be static"));
        };
        count = count
            .checked_mul(*extent)
            .filter(|count| *count <= i32::MAX as usize)
            .ok_or_else(|| unsupported("kernel_call", "parallel grid exceeds signed 32-bit launch bounds"))?;
        extents.push(*extent);
    }
    let mut coordinate = append(
        &mut block,
        tt::get_program_id(tt::ProgramIdDim::X, context.signless_integer_type(32).as_ref(), location)?,
    )?;
    coordinate = append(&mut block, arith::extsi(coordinate, context.signless_integer_type(64), location)?)?;
    let mut coordinates = vec![coordinate; extents.len()];
    for axis in (0..extents.len()).rev() {
        let extent = lowering.integer(&mut block, extents[axis].max(1) as i64)?;
        coordinates[axis] = append(&mut block, arith::remsi(coordinate, extent, location)?)?;
        coordinate = append(&mut block, arith::divsi(coordinate, extent, location)?)?;
    }
    let active = lowering.boolean(&mut block, count != 0)?;
    let mut inputs = Vec::new();
    for (index, parameter) in call.parameters().iter().enumerate() {
        let starts = lowering.mapping(&mut block, parameter.mapping(), &coordinates)?;
        let tile = parameter.mapping().block_shape().to_vec();
        let mut limits = Vec::new();
        for (start, extent) in starts.iter().zip(&tile) {
            let extent = lowering.integer(&mut block, *extent as i64)?;
            limits.push(append(&mut block, arith::addi(*start, extent, location)?)?);
        }
        let allocation = shape(&parameter_types[index])?;
        inputs.push(LoweringValue::Reference(Reference {
            pointer: block.argument(index)?.as_ref(),
            axes: (0..tile.len()).collect(),
            fixed: vec![None; allocation.len()],
            allocation,
            starts,
            limits,
            shape: tile,
            valid: active,
        }));
    }
    inputs.extend(coordinates.into_iter().map(LoweringValue::Native));
    lowering.region(&mut block, kernel.definition().body().entry_region_ref(), &inputs)?;
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
        mapping: &ryft_core::kernels::BlockMapping,
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

    /// Creates valid pointer tensors, intersecting root, enclosing-window and physical-padding bounds.
    fn pointers(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        reference: &Reference<'c, 't>,
    ) -> Result<(Native<'c, 't>, Native<'c, 't>), Error> {
        let (indices, mut mask) = self.indices(block, &reference.shape)?;
        let valid = self.splat(block, reference.valid, DataType::Boolean, &reference.shape)?;
        mask = append(block, arith::andi(mask, valid, self.location)?)?;
        let zero = self.integer(block, 0)?;
        let mut offset = self.splat(block, zero, DataType::I64, &reference.shape)?;
        let mut stride = 1usize;
        for axis in (0..reference.allocation.len()).rev() {
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
            let nonnegative = self.splat(block, zero, DataType::I64, &reference.shape)?;
            let nonnegative = append(
                block,
                arith::cmpi(
                    coordinate,
                    nonnegative,
                    arith::IntegerComparisonPredicate::SignedGreaterThanOrEqual,
                    self.location,
                )?,
            )?;
            let below = append(
                block,
                arith::cmpi(coordinate, limit, arith::IntegerComparisonPredicate::SignedLessThan, self.location)?,
            )?;
            mask = append(block, arith::andi(mask, nonnegative, self.location)?)?;
            mask = append(block, arith::andi(mask, below, self.location)?)?;
            let coordinate_valid = append(block, arith::andi(nonnegative, below, self.location)?)?;
            let zero_coordinate = self.splat(block, zero, DataType::I64, &reference.shape)?;
            let coordinate =
                append(block, arith::select(coordinate_valid, coordinate, zero_coordinate, self.location)?)?;
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
        let zero = self.splat(block, zero, DataType::I64, &reference.shape)?;
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

    /// Loads a clipped F32 window with the exact caller-supplied scalar padding value.
    fn read(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        reference: &Reference<'c, 't>,
        other: Native<'c, 't>,
    ) -> Result<Native<'c, 't>, Error> {
        let (pointer, mask) = self.pointers(block, reference)?;
        let other = self.splat(block, other, DataType::F32, &reference.shape)?;
        append(
            block,
            tt::load(
                pointer,
                Some(mask),
                Some(other),
                tt::CacheModifier::None,
                tt::EvictionPolicy::Normal,
                false,
                self.r#type(DataType::F32, &reference.shape)?,
                self.location,
            )?,
        )
    }

    /// Writes only valid lanes of a global window.
    fn write(
        &self,
        block: &mut DetachedBlock<'c, 't>,
        reference: &Reference<'c, 't>,
        value: Native<'c, 't>,
    ) -> Result<(), Error> {
        let (pointer, mask) = self.pointers(block, reference)?;
        block.append_operation(tt::store(
            pointer,
            value,
            Some(mask),
            tt::CacheModifier::None,
            tt::EvictionPolicy::Normal,
            false,
            self.location,
        )?)?;
        Ok(())
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
            _ => return Err(unsupported(operation, "reference transform is outside the supported subset")),
        }
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
                    _ => return Err(unsupported("constant", "captured references are unsupported")),
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
                    vec![LoweringValue::Native(self.read(block, inputs[0].reference()?, zero)?)]
                }
                KernelOperation::Portable(ArrayIrOperation::ReferenceWrite(_)) => {
                    self.write(block, inputs[0].reference()?, native(1)?)?;
                    vec![]
                }
                KernelOperation::TileLoad(operation) => {
                    let mut reference = inputs[0].reference()?.clone();
                    reference.shape = operation.block_shape().to_vec();
                    for (axis, start) in reference.starts.iter_mut().enumerate() {
                        *start = append(block, arith::addi(*start, native(axis + 1)?, location)?)?;
                    }
                    vec![LoweringValue::Native(self.read(block, &reference, native(inputs.len() - 1)?)?)]
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
        if matches!(operation, ArrayOperation::Add(_) | ArrayOperation::Sub(_) | ArrayOperation::Mul(_)) {
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
            ArrayOperation::Add(_) if output.data_type() == DataType::F32 => {
                append(block, arith::addf(values[0], values[1], location)?)?
            }
            ArrayOperation::Add(_) if output.data_type() == DataType::I64 => {
                append(block, arith::addi(values[0], values[1], location)?)?
            }
            ArrayOperation::Sub(_) if output.data_type() == DataType::F32 => {
                append(block, arith::subf(values[0], values[1], location)?)?
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
                        arith::cmpf(inputs[0], inputs[1], float_comparison(operation.direction()), location)?,
                    )?
                } else {
                    append(
                        block,
                        arith::cmpi(
                            inputs[0],
                            inputs[1],
                            integer_comparison(operation.direction(), types[0].data_type() == DataType::Boolean),
                            location,
                        )?,
                    )?
                }
            }
            ArrayOperation::Select(_) => append(block, arith::select(inputs[0], inputs[1], inputs[2], location)?)?,
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
            ArrayOperation::Broadcast(operation) => {
                if types[0].rank() == 0 {
                    self.splat(block, inputs[0], output.data_type(), &output_shape)?
                } else {
                    if operation.output_axes().windows(2).any(|axes| axes[0] >= axes[1]) {
                        return Err(unsupported(operation.name(), "broadcast axes must preserve source order"));
                    }
                    let source = shape(&types[0])?;
                    let mut expanded = vec![1; output_shape.len()];
                    for (axis, output_axis) in operation.output_axes().iter().enumerate() {
                        expanded[*output_axis] = source[axis];
                    }
                    let reshaped = append(
                        block,
                        tt::reshape(inputs[0], false, false, self.r#type(output.data_type(), &expanded)?, location)?,
                    )?;
                    append(block, tt::broadcast(reshaped, output_type, location)?)?
                }
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

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::kernels::{
        Grid, KernelCallOperation, KernelDefinition, KernelParameterAccess, whole_array_parameter,
    };
    use ryft_core::{ArraySliceAxis, Placeholder, ProgramBuilder, ReferenceReadOperation, ReferenceWriteOperation};
    use ryft_mlir::{OpRef, Region};

    use super::*;

    /// Covers global windows and the partially populated final tile through the public macro.
    #[ryft_core::kernels::kernel(requires = left.shape()[0] == right.shape()[0])]
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
    #[ryft_core::kernels::kernel]
    fn sum(
        #[input(data_type = F32, rank = 1)] input: &Array,
        #[output(data_type = F32, shape = [])] output: &mut Array,
    ) {
        output.store(input.load().sum([0]));
    }

    /// Covers invariant global references and an ordinary tile accumulator in a bounded loop.
    #[ryft_core::kernels::kernel(requires = left.shape()[1] == right.shape()[0])]
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

    /// The condition writes one and the body writes two; a bound of one must leave two in the output.
    fn condition_write_definition() -> ryft_core::kernels::KernelDefinition {
        use ryft_core::kernels::{
            Grid, KernelCallOperation, KernelDefinition, KernelParameterAccess, whole_array_parameter,
        };
        use ryft_core::{Placeholder, ProgramBuilder, ReferenceWriteOperation, WhileOperation};

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
