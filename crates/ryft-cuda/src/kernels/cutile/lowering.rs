//! Deterministic cuTile source over the canonical portable kernel arena.
//!
//! Global windows use bounds-checked gather/scatter, including completely empty edge windows. Tile values use
//! power-of-two physical shapes; reductions mask logical padding with the reduction identity before arithmetic.
//!
//! The validity mask of a window excludes its logical tile padding, its elements beyond the enclosing window, and,
//! for a window with [`BoundaryPolicy::Masked`], its elements beyond the end of the array. Reads, writes, and add
//! updates, including [`TileLoadOperation`](ryft_kernels::TileLoadOperation), gather and scatter under that mask, and a
//! [`WindowMaskOperation`](ryft_kernels::WindowMaskOperation) returns it as a Boolean tile whose padding is false. The
//! masked accesses (i.e., [`MaskedLoadOperation`](ryft_kernels::MaskedLoadOperation),
//! [`MaskedStoreOperation`](ryft_kernels::MaskedStoreOperation), and
//! [`MaskedSwapOperation`](ryft_kernels::MaskedSwapOperation)) gather and scatter under the intersection of that mask
//! with their explicit mask, so they never touch padding, and a load or swap takes the corresponding element of its
//! `other` input for every element outside that intersection. A masked swap always reads before it writes, even when
//! its old value is dead, which is equivalent to a store because the read selects only the elements that the swap then
//! overwrites and its result is unused.
//!
//! The host rejects an in-bounds tile load whose window leaves its source view and any tile load whose window limit
//! exceeds the canonical dimension width, and the generated source cannot report such a failure at run time. Admission
//! therefore requires the start bounds of every tile load to rule it out, so an admitted in-bounds load reads only
//! valid elements and lowers like a masked one. For the same reason, admission rejects every assertion instead of
//! dropping it.
//!
//! A dot lowers to one `ct.mma` whose accumulator has the output element type, which is the requested accumulation
//! type. Admission accepts only the default precision, because the operand conversions of `ct.mma` for an explicit
//! precision or algorithm have not been qualified.
//!
//! A read that follows a write to the same parameter in one block is expected to observe that write. Tile IR orders
//! memory operations only through tokens, and the cuTile front end threads them so that every load waits for the last
//! store to any array that may alias its own. The exporter declares all parameters in one alias group, so that
//! ordering covers every pair of parameters. Each scatter writes distinct affine indices, so no scatter overlaps
//! itself. This relies on the token ordering pass of the front end, which its documentation does not state as a
//! contract. That pass was read from the main branch of the cuTile sources at commit `98cd15e`, while the pinned
//! release is 1.5.0, so it has not been confirmed against the pinned front end, and no generated source has been
//! compiled or run.

use ryft_core::{
    Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReferenceTransform,
    ArrayReferenceTransformIndex, ArrayType, Atom, ComparisonDirection, DataType, Dimension, DimensionOperation,
    DimensionValue, DotDimensionNumbers, MAX_DIMENSION_EXTENT, Operation, ProgramError, ReductionKind,
    ReferenceAccessMode, ReferenceAccessOperation, RegionRef, Typed, infer_reference_view_type,
    validated_reference_access_descriptors,
};
use ryft_kernels::{
    BoundaryPolicy, GridExecution, KernelOperation, KernelSchedule, NoKernelExtension, TILE_LOAD_OPERATION_NAME,
    TileLoadOperation, VerifiedKernel,
};

use crate::kernels::cutile::{Error, Options, Parameter, data_type_name, launch_grid};

/// Checked source and the exact logical parameter ABI consumed by the exporter.
pub(super) struct Source {
    /// Complete Python module defining the fixed exported symbol.
    pub code: String,

    /// Flattened parallel launch grid; sequential axes remain inside the kernel.
    pub grid: [u32; 3],

    /// Array constraints in canonical body-reference order.
    pub parameters: Vec<Parameter>,
}

/// A source expression or a reference into one declared global parameter.
#[derive(Clone, PartialEq)]
enum Value {
    /// Ordinary immutable tile or dimension expression.
    Expression(String),

    /// Global window retaining physical coordinates for dimensions removed by canonical indexing.
    Reference {
        /// Original global parameter position.
        parameter: usize,

        /// Logical window starts along surviving axes.
        starts: Vec<String>,

        /// Logical tile extents along surviving axes.
        shape: Vec<usize>,

        /// Exclusive enclosing-window limits along surviving axes.
        limits: Vec<String>,

        /// Physical parameter axis for each surviving logical axis.
        axes: Vec<usize>,

        /// Fixed physical coordinates for indexed axes; surviving axes contain `None`.
        fixed: Vec<Option<String>>,

        /// Validity constraints retained for indexed-away axes.
        validity: Vec<String>,
    },
}

impl Value {
    /// Extracts a non-reference expression after canonical type checking.
    fn expression(&self) -> Result<&str, Error> {
        match self {
            Self::Expression(value) => Ok(value),
            Self::Reference { .. } => Err(unsupported("reference", "reference cannot escape as a tile value")),
        }
    }
}

/// Source writer with deterministic temporary names and a checked expansion budget.
struct Lowering {
    /// Generated module text.
    code: String,

    /// Current Python indentation in levels.
    indentation: usize,

    /// Next temporary identifier.
    next: usize,

    /// Remaining canonical instruction and literal work.
    remaining: usize,
}

/// Validates and lowers a portable definition before any compiler process can start.
pub(super) fn lower(
    kernel: &VerifiedKernel<'_, NoKernelExtension>,
    options: &Options,
    schedule: &KernelSchedule,
) -> Result<Source, Error> {
    if schedule.maximum_scratch_bytes().is_some() {
        return Err(unsupported("kernel_call", "cuTile does not expose a validated compiler scratch-memory bound"));
    }
    let call = kernel.definition().operation();
    if !call.prefetch_types().is_empty() {
        return Err(unsupported("kernel_call", "scalar prefetch must be specialized before cuTile compilation"));
    }
    let mut lowering = Lowering {
        code: "import cuda.tile as ct\n\n@ct.kernel\n".to_owned(),
        indentation: 0,
        next: 0,
        remaining: options.maximum_instructions(),
    };
    lowering.charge(call.parameters().len())?;
    lowering.charge(call.grid().dimensions().len())?;
    let mut parameters = Vec::new();
    for parameter in call.parameters() {
        let r#type = parameter.r#type();
        lowering.charge(r#type.rank())?;
        checked_shape(&r#type)?;
        parameters.push(Parameter::from_type(&r#type)?);
    }
    lowering.line(format!(
        "def ryft_cutile_kernel({}):",
        (0..parameters.len()).map(|index| format!("p{index}")).collect::<Vec<_>>().join(", ")
    ));
    lowering.indentation += 1;
    let mut extents = Vec::new();
    let grid = launch_grid(kernel)?;
    for dimension in call.grid().dimensions() {
        let Dimension::Static(extent) = dimension.extent() else {
            return Err(unsupported("kernel_call", "grid extents must be static"));
        };
        if *extent > i32::MAX as usize {
            return Err(unsupported("kernel_call", "grid extent exceeds signed 32-bit indexing"));
        }
        extents.push(*extent);
    }
    if extents.contains(&0) {
        lowering.line("if ct.bid(0) < 0:".to_owned());
        lowering.indentation += 1;
    }
    let mut coordinates = vec![String::new(); extents.len()];
    let mut remaining = "ct.astype(ct.bid(0), ct.int64)".to_owned();
    for axis in (0..extents.len()).rev() {
        if call.grid().dimensions()[axis].execution() == GridExecution::Parallel {
            coordinates[axis] = lowering.assign(format!("({remaining} % {})", extents[axis].max(1)));
            remaining = lowering.assign(format!("({remaining} // {})", extents[axis].max(1)));
        }
    }
    for axis in 0..extents.len() {
        if call.grid().dimensions()[axis].execution() == GridExecution::Sequential {
            coordinates[axis] = lowering.name();
            lowering.line(format!("for {} in range({}):", coordinates[axis], extents[axis]));
            lowering.indentation += 1;
        }
    }
    let mut inputs = Vec::new();
    for (parameter, declaration) in call.parameters().iter().enumerate() {
        let starts = lowering.mapping(declaration.mapping(), &coordinates)?;
        let shape = declaration.mapping().block_shape().to_vec();
        // A masked window may cross the end of its array, so its limits also clip to the array extents. Its mask is
        // then its exact validity, which masked accesses and window masks require, instead of relying on the bounds
        // checks of `ct.gather` and `ct.scatter`.
        let limits = match declaration.mapping().boundary_policy() {
            BoundaryPolicy::InBounds => {
                starts.iter().zip(&shape).map(|(start, extent)| format!("({start} + {extent})")).collect()
            }
            BoundaryPolicy::Masked => starts
                .iter()
                .zip(&shape)
                .zip(checked_shape(&declaration.r#type())?)
                .map(|((start, extent), size)| format!("ct.minimum(({start} + {extent}), {size})"))
                .collect(),
        };
        let axes = (0..shape.len()).collect();
        let fixed = vec![None; shape.len()];
        inputs.push(Value::Reference { parameter, starts, shape, limits, axes, fixed, validity: vec![] });
    }
    inputs.extend(coordinates.into_iter().map(Value::Expression));
    lowering.region(kernel.definition().body().entry_region_ref(), &inputs)?;
    lowering.line("return".to_owned());
    Ok(Source { code: lowering.code, grid, parameters })
}

impl Lowering {
    /// Charges source work before expanding a canonical instruction or literal.
    fn charge(&mut self, count: usize) -> Result<(), Error> {
        self.remaining = self
            .remaining
            .checked_sub(count)
            .ok_or_else(|| unsupported("kernel_call", "source construction exceeds the instruction budget"))?;
        Ok(())
    }

    /// Allocates the next stable temporary name.
    fn name(&mut self) -> String {
        let result = format!("v{}", self.next);
        self.next += 1;
        result
    }

    /// Writes a source line at the current block depth.
    fn line(&mut self, text: String) {
        self.code.push_str(&"    ".repeat(self.indentation));
        self.code.push_str(&text);
        self.code.push('\n');
    }

    /// Materializes an immutable expression once.
    fn assign(&mut self, expression: String) -> String {
        let name = self.name();
        self.line(format!("{name} = {expression}"));
        name
    }

    /// Emits a checked dimension-only block mapping using its existing arena.
    fn mapping(&mut self, mapping: &ryft_kernels::BlockMapping, inputs: &[String]) -> Result<Vec<String>, Error> {
        let program = mapping.program();
        let mut values = vec![String::new(); program.atoms().len()];
        for (id, input) in program.input_ids().iter().zip(inputs) {
            values[id.index()] = input.clone();
        }
        for (index, atom) in program.atoms().iter().enumerate() {
            if let Atom::Constant(ArrayIrValue::Dimension(value)) = atom {
                values[index] = value.extent().to_string();
            }
        }
        for instruction in program.instructions() {
            self.charge(1)?;
            let ArrayIrOperation::Dimension(operation) = instruction.operation() else { unreachable!() };
            let inputs = instruction.inputs().iter().map(|id| values[id.index()].clone()).collect::<Vec<_>>();
            let result = dimension(operation, &inputs)?;
            values[instruction.outputs()[0].index()] = self.assign(result);
        }
        Ok(program.output_ids().iter().map(|id| values[id.index()].clone()).collect())
    }

    /// Replays canonical regions into Python syntax without executing or unrolling dynamic control flow.
    fn region(
        &mut self,
        region: RegionRef<'_, ArrayIrValue<Array>, KernelOperation>,
        inputs: &[Value],
    ) -> Result<Vec<Value>, Error> {
        let mut values = vec![None; region.atoms().len()];
        for (id, value) in region.input_ids().iter().zip(inputs) {
            values[id.index()] = Some(value.clone());
        }
        for (index, atom) in region.atoms().iter().enumerate() {
            if let Atom::Constant(value) = atom {
                values[index] = Some(Value::Expression(match value {
                    ArrayIrValue::Dimension(value) => value.extent().to_string(),
                    ArrayIrValue::Array(value) => self.constant(value)?,
                    ArrayIrValue::Reference(_) => {
                        return Err(unsupported("constant", "captured references are unsupported"));
                    }
                }));
            }
        }
        for instruction in region.instructions() {
            self.charge(1)?;
            let mut inputs =
                instruction.inputs().iter().map(|id| values[id.index()].clone().unwrap()).collect::<Vec<_>>();
            let operation = instruction.operation();
            let descriptors = validated_reference_access_descriptors(operation, instruction.inputs().len())?;
            for (input_index, descriptor) in descriptors.iter().enumerate() {
                for transform in descriptor.iter().flat_map(|descriptor| descriptor.transforms()) {
                    self.apply_transform(&mut inputs[input_index], transform, operation.name())?;
                }
            }
            if descriptors.iter().any(Option::is_some) {
                inputs.truncate(operation.base_input_count());
            }
            let types = instruction
                .inputs()
                .iter()
                .map(|id| region.atoms()[id.index()].r#type().into_owned())
                .collect::<Vec<_>>();
            let output_types = instruction
                .outputs()
                .iter()
                .map(|id| region.atoms()[id.index()].r#type().into_owned())
                .collect::<Vec<_>>();
            let expression = |index: usize| inputs[index].expression().map(str::to_owned);
            let results = match instruction.operation() {
                // An assertion has no results, and the generated source cannot report its failure to the caller.
                KernelOperation::Portable(ArrayIrOperation::Assert(_))
                | KernelOperation::Portable(ArrayIrOperation::Array(ArrayOperation::Assert(_))) => {
                    return Err(unsupported("assert", "runtime assertions require native failure propagation"));
                }
                KernelOperation::Portable(ArrayIrOperation::Array(operation)) => {
                    let input_types = types
                        .iter()
                        .map(|r#type| match r#type {
                            ArrayIrType::Array(r#type) => Ok(r#type.clone()),
                            _ => Err(unsupported(operation.name(), "array operation received a non-array value")),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let [ArrayIrType::Array(output)] = output_types.as_slice() else {
                        return Err(unsupported(
                            operation.name(),
                            "array operations must have exactly one array result",
                        ));
                    };
                    let inputs = inputs
                        .iter()
                        .map(|input| input.expression().map(str::to_owned))
                        .collect::<Result<Vec<_>, _>>()?;
                    vec![Value::Expression(self.array(operation, &inputs, &input_types, output)?)]
                }
                KernelOperation::Portable(ArrayIrOperation::Dimension(operation)) => {
                    let inputs = inputs
                        .iter()
                        .map(|input| input.expression().map(str::to_owned))
                        .collect::<Result<Vec<_>, _>>()?;
                    vec![Value::Expression(self.assign(dimension(operation, &inputs)?))]
                }
                KernelOperation::Portable(ArrayIrOperation::Compare(operation)) => {
                    vec![Value::Expression(self.assign(format!(
                        "({} {} {})",
                        expression(0)?,
                        comparison(operation.direction()),
                        expression(1)?
                    )))]
                }
                KernelOperation::Portable(ArrayIrOperation::DimensionToScalar(_)) => {
                    vec![Value::Expression(self.assign(format!("ct.astype({}, ct.int64)", expression(0)?)))]
                }
                KernelOperation::Portable(ArrayIrOperation::DimensionFromScalar(operation)) => {
                    let bounds = operation.output_type().bounds();
                    let value = expression(0)?;
                    let upper = bounds.upper().map_or_else(
                        || format!("({value} <= {})", ryft_core::MAX_DIMENSION_EXTENT),
                        |upper| format!("({value} < {upper})"),
                    );
                    self.line(format!("ct.assert_(({value} >= {}) & {upper}, 'dimension bounds')", bounds.lower()));
                    vec![Value::Expression(self.assign(format!("ct.astype({value}, ct.int64)")))]
                }
                KernelOperation::Portable(ArrayIrOperation::DimensionSize(operation)) => {
                    vec![Value::Expression(
                        operation
                            .input_dimension()
                            .value()
                            .ok_or_else(|| unsupported("dimension_size", "dynamic dimension size is unsupported"))?
                            .to_string(),
                    )]
                }
                KernelOperation::Portable(ArrayIrOperation::ReferenceRead(_)) => {
                    vec![Value::Expression(self.read(&inputs[0], "0")?)]
                }
                KernelOperation::Portable(ArrayIrOperation::ReferenceWrite(_)) => {
                    self.write(&inputs[0], &expression(1)?)?;
                    vec![]
                }
                KernelOperation::Portable(ArrayIrOperation::ReferenceSwap(_)) => {
                    let previous = self.read(&inputs[0], "0")?;
                    self.write(&inputs[0], &expression(1)?)?;
                    vec![Value::Expression(previous)]
                }
                KernelOperation::Portable(ArrayIrOperation::ReferenceAddUpdate(_)) => {
                    // The update is a gather, an addition, and a scatter of one window under its validity mask, and
                    // the scatter follows the gather because both access the same parameter. A rank-0 parameter has
                    // no window indices, so its only element is loaded and stored directly.
                    let update = expression(1)?;
                    if matches!(&inputs[0], Value::Reference { fixed, .. } if fixed.is_empty()) {
                        let current = self.read(&inputs[0], "0")?;
                        let sum = self.assign(format!("({current} + {update})"));
                        self.write(&inputs[0], &sum)?;
                    } else {
                        let (parameter, indices, mask) = self.window(&inputs[0])?;
                        let current = self.assign(format!(
                            "ct.gather(p{parameter}, {indices}, mask={mask}, padding_value=0, check_bounds=True)"
                        ));
                        let sum = self.assign(format!("({current} + {update})"));
                        self.line(format!(
                            "ct.scatter(p{parameter}, {indices}, {sum}, mask={mask}, check_bounds=True)"
                        ));
                    }
                    vec![]
                }
                KernelOperation::TileLoad(operation) => {
                    validate_tile_load(operation, &types)?;
                    let Value::Reference { parameter, starts, limits, axes, fixed, validity, .. } = &inputs[0] else {
                        unreachable!()
                    };
                    let starts = starts
                        .iter()
                        .enumerate()
                        .map(|(axis, start)| expression(axis + 1).map(|offset| format!("({start} + {offset})")))
                        .collect::<Result<Vec<_>, _>>()?;
                    let reference = Value::Reference {
                        parameter: *parameter,
                        starts,
                        shape: operation.block_shape().to_vec(),
                        limits: limits.clone(),
                        axes: axes.clone(),
                        fixed: fixed.clone(),
                        validity: validity.clone(),
                    };
                    vec![Value::Expression(self.read(&reference, &expression(inputs.len() - 1)?)?)]
                }
                KernelOperation::MaskedLoad(_) => {
                    let (parameter, indices, mask) = self.masked_window(&inputs[0], &expression(1)?)?;
                    vec![Value::Expression(self.masked_gather(parameter, &indices, &mask, &expression(2)?))]
                }
                KernelOperation::MaskedStore(_) => {
                    let (parameter, indices, mask) = self.masked_window(&inputs[0], &expression(2)?)?;
                    let value = expression(1)?;
                    self.line(format!("ct.scatter(p{parameter}, {indices}, {value}, mask={mask}, check_bounds=True)"));
                    vec![]
                }
                KernelOperation::MaskedSwap(_) => {
                    let (parameter, indices, mask) = self.masked_window(&inputs[0], &expression(2)?)?;
                    let previous = self.masked_gather(parameter, &indices, &mask, &expression(3)?);
                    let value = expression(1)?;
                    self.line(format!("ct.scatter(p{parameter}, {indices}, {value}, mask={mask}, check_bounds=True)"));
                    vec![Value::Expression(previous)]
                }
                KernelOperation::WindowMask(_) => {
                    let Value::Reference { shape, .. } = &inputs[0] else { unreachable!() };
                    let shape = physical_shape(shape);
                    let (_, _, mask) = self.window(&inputs[0])?;
                    vec![Value::Expression(self.assign(if mask == "True" {
                        format!("ct.full({shape}, True, ct.bool_)")
                    } else {
                        format!("ct.broadcast_to({mask}, {shape})")
                    }))]
                }
                KernelOperation::Portable(ArrayIrOperation::While(operation)) => {
                    let bound = operation
                        .iteration_bound()
                        .ok_or_else(|| unsupported("while", "a finite iteration bound is required"))?;
                    if bound > i32::MAX as usize {
                        return Err(unsupported("while", "iteration bound exceeds signed 32-bit indexing"));
                    }
                    let mut carries = Vec::new();
                    for input in &inputs {
                        carries.push(match input {
                            Value::Expression(value) => Value::Expression(self.assign(value.clone())),
                            Value::Reference { .. } => input.clone(),
                        });
                    }
                    let active = self.assign("ct.full((), True, ct.bool_)".to_owned());
                    let counter = self.name();
                    self.line(format!("for {counter} in range({bound}):"));
                    self.indentation += 1;
                    self.line(format!("if {active}:"));
                    self.indentation += 1;
                    let predicate = self.region(region.with_id(instruction.regions()[0])?, &carries)?;
                    self.line(format!("{active} = {}", predicate[0].expression()?));
                    self.line(format!("if {active}:"));
                    self.indentation += 1;
                    let next = self.region(region.with_id(instruction.regions()[1])?, &carries)?;
                    let mut destinations = Vec::new();
                    let mut sources = Vec::new();
                    for (carry, next) in carries.iter().zip(&next) {
                        match (carry, next) {
                            (Value::Expression(left), Value::Expression(right)) => {
                                destinations.push(left.clone());
                                sources.push(right.clone());
                            }
                            (Value::Reference { .. }, Value::Reference { .. }) if carry == next => {}
                            _ => {
                                return Err(unsupported(
                                    "while",
                                    "loop-carried reference windows must remain unchanged",
                                ));
                            }
                        }
                    }
                    if !destinations.is_empty() {
                        self.line(format!("{} = {}", tuple(&destinations), tuple(&sources)));
                    }
                    self.indentation -= 3;
                    carries
                }
                KernelOperation::Portable(ArrayIrOperation::Condition(_)) => {
                    if output_types.iter().any(|r#type| !matches!(r#type, ArrayIrType::Array(_))) {
                        return Err(unsupported("condition", "condition results must be ordinary arrays"));
                    }
                    let names = output_types.iter().map(|_| self.name()).collect::<Vec<_>>();
                    for (branch, id) in instruction.regions().iter().enumerate() {
                        self.line(if branch == 0 { format!("if {}:", expression(0)?) } else { "else:".to_owned() });
                        self.indentation += 1;
                        let results = self.region(region.with_id(*id)?, &inputs[1..])?;
                        if names.is_empty() {
                            self.line("pass".to_owned());
                        }
                        for (name, value) in names.iter().zip(results) {
                            self.line(format!("{name} = {}", value.expression()?));
                        }
                        self.indentation -= 1;
                    }
                    names.into_iter().map(Value::Expression).collect()
                }
                operation => {
                    return Err(unsupported(operation.name(), "operation is outside the portable cuTile subset"));
                }
            };
            for (id, result) in instruction.outputs().iter().zip(results) {
                values[id.index()] = Some(result);
            }
        }
        Ok(region.output_ids().iter().map(|id| values[id.index()].clone().unwrap()).collect())
    }

    /// Emits exact uniform literal bits without decimal floating-point conversion.
    fn constant(&mut self, value: &Array) -> Result<String, Error> {
        let r#type = value.r#type();
        let shape = checked_shape(&r#type)?;
        let bytes = value.logical_bytes();
        let size = match r#type.data_type() {
            DataType::Boolean => 1,
            DataType::F16 | DataType::BF16 => 2,
            DataType::I32 | DataType::U32 | DataType::F32 => 4,
            _ => 8,
        };
        self.charge(bytes.len() / size)?;
        if bytes.chunks_exact(size).any(|chunk| chunk != &bytes[..size]) {
            return Err(unsupported("constant", "nonuniform array literals are not yet supported"));
        }
        let mut bits = [0u8; 8];
        if !bytes.is_empty() {
            bits[..size].copy_from_slice(&bytes[..size]);
        }
        let scalar = literal_expression(r#type.data_type(), u64::from_le_bytes(bits))?;
        Ok(self.assign(format!("ct.broadcast_to({scalar}, {})", physical_shape(&shape))))
    }

    /// Applies one supported reference transform while preserving enclosing-window bounds.
    fn apply_transform(
        &mut self,
        reference: &mut Value,
        transform: &ArrayReferenceTransform,
        operation: &'static str,
    ) -> Result<(), Error> {
        self.charge(1)?;
        let Value::Reference { starts, shape, limits, axes, fixed, validity, .. } = reference else {
            return Err(unsupported(operation, "expected a global reference"));
        };
        match transform {
            ArrayReferenceTransform::Index { axis, index: ArrayReferenceTransformIndex::Static(index) } => {
                let coordinate = format!("({} + {index})", starts.remove(*axis));
                validity.push(format!("({coordinate} < {})", limits.remove(*axis)));
                fixed[axes.remove(*axis)] = Some(coordinate);
                shape.remove(*axis);
            }
            ArrayReferenceTransform::Slice { axes: selections } => {
                if selections.iter().any(|selection| selection.stride() != 1) {
                    return Err(unsupported(operation, "strided reference transforms are unsupported"));
                }
                for (axis, selection) in selections.iter().enumerate() {
                    starts[axis] = format!("({} + {})", starts[axis], selection.start());
                    limits[axis] = format!("ct.minimum({}, ({} + {}))", limits[axis], starts[axis], selection.size());
                    shape[axis] = selection.size();
                }
            }
            ArrayReferenceTransform::Index { .. } => {
                return Err(unsupported(operation, "dynamic reference transforms are unsupported"));
            }
        }
        Ok(())
    }

    /// Computes broadcasted element indices and the logical tile-padding mask.
    fn indices(&mut self, starts: &[String], shape: &[usize]) -> (String, String) {
        let mut indices = Vec::new();
        let mut masks = Vec::new();
        for (axis, (&extent, start)) in shape.iter().zip(starts).enumerate() {
            let mut dimensions = vec![1; shape.len()];
            dimensions[axis] = extent.max(1).next_power_of_two();
            let index = self.assign(format!(
                "ct.reshape(ct.arange({}, dtype=ct.int32), {})",
                dimensions[axis],
                tuple(&dimensions)
            ));
            masks.push(format!("({index} < {extent})"));
            indices.push(format!("({start} + {index})"));
        }
        (tuple(&indices), if masks.is_empty() { "True".to_owned() } else { masks.join(" & ") })
    }

    /// Adds the enclosing view bounds to the global bounds checked by cuTile.
    fn window_indices(&mut self, starts: &[String], shape: &[usize], limits: &[String]) -> (Vec<String>, String) {
        let mut indices = Vec::new();
        let mut masks = Vec::new();
        for (axis, ((&extent, start), limit)) in shape.iter().zip(starts).zip(limits).enumerate() {
            let mut dimensions = vec![1; shape.len()];
            dimensions[axis] = extent.max(1).next_power_of_two();
            let local = self.assign(format!(
                "ct.reshape(ct.arange({}, dtype=ct.int64), {})",
                dimensions[axis],
                tuple(&dimensions)
            ));
            let index = self.assign(format!("({start} + {local})"));
            masks.push(format!("(({local} < {extent}) & ({index} < {limit}))"));
            indices.push(index);
        }
        (indices, if masks.is_empty() { "True".to_owned() } else { masks.join(" & ") })
    }

    /// Returns the parameter of a global window, its global element indices, and its validity mask, which excludes
    /// logical tile padding, elements beyond the enclosing window, and indexed away coordinates outside their window.
    /// A window into a rank-0 parameter selects its only element, which is always valid.
    fn window(&mut self, reference: &Value) -> Result<(usize, String, String), Error> {
        let Value::Reference { parameter, starts, shape, limits, axes, fixed, validity } = reference else {
            unreachable!()
        };
        checked_tile(shape)?;
        if fixed.is_empty() {
            return Ok((*parameter, "(0,)".to_owned(), "True".to_owned()));
        }
        let (logical_indices, mut mask) = self.window_indices(starts, shape, limits);
        for condition in validity {
            mask = format!("({mask}) & ({condition})");
        }
        let mut indices = fixed.clone();
        for (&axis, index) in axes.iter().zip(logical_indices) {
            indices[axis] = Some(index);
        }
        Ok((*parameter, tuple(&indices.into_iter().map(Option::unwrap).collect::<Vec<_>>()), mask))
    }

    /// Returns the parameter and global element indices of a global window together with the name of the intersection
    /// of its validity mask with the explicit Boolean tile `mask`.
    fn masked_window(&mut self, reference: &Value, mask: &str) -> Result<(usize, String, String), Error> {
        let (parameter, indices, validity) = self.window(reference)?;
        let mask = if validity == "True" { mask.to_owned() } else { format!("({validity}) & ({mask})") };
        Ok((parameter, indices, self.assign(mask)))
    }

    /// Loads a clipped global window with an explicit exact padding value.
    fn read(&mut self, reference: &Value, other: &str) -> Result<String, Error> {
        let (parameter, indices, mask) = self.window(reference)?;
        if matches!(reference, Value::Reference { fixed, .. } if fixed.is_empty()) {
            return Ok(self.assign(format!("ct.reshape(ct.load(p{parameter}, (0,), (1,)), ())")));
        }
        Ok(self.assign(format!(
            "ct.gather(p{parameter}, {indices}, mask={mask}, padding_value={other}, check_bounds=True)"
        )))
    }

    /// Publishes only valid logical/global coordinates.
    fn write(&mut self, reference: &Value, value: &str) -> Result<(), Error> {
        let (parameter, indices, mask) = self.window(reference)?;
        if matches!(reference, Value::Reference { fixed, .. } if fixed.is_empty()) {
            self.line(format!("ct.store(p{parameter}, (0,), ct.reshape({value}, (1,)))"));
            return Ok(());
        }
        self.line(format!("ct.scatter(p{parameter}, {indices}, {value}, mask={mask}, check_bounds=True)"));
        Ok(())
    }

    /// Gathers the elements of parameter `parameter` at `indices` that `mask` selects, and takes the corresponding
    /// element of `other`, which has the type of the gathered tile, for every other element. The selection is explicit
    /// because `other` is a tile, while the padding value of `ct.gather` is a scalar.
    fn masked_gather(&mut self, parameter: usize, indices: &str, mask: &str, other: &str) -> String {
        let loaded =
            self.assign(format!("ct.gather(p{parameter}, {indices}, mask={mask}, padding_value=0, check_bounds=True)"));
        self.assign(format!("ct.where({mask}, {loaded}, {other})"))
    }

    /// Lowers ordinary array operations, retaining their canonical logical output type.
    fn array(
        &mut self,
        operation: &ArrayOperation<Array>,
        inputs: &[String],
        types: &[ArrayType],
        output: &ArrayType,
    ) -> Result<String, Error> {
        let shape = checked_shape(output)?;
        checked_tile(&shape)?;
        for r#type in types {
            checked_shape(r#type)?;
        }
        let data_type = dtype(output.data_type())?;
        let result = match operation {
            ArrayOperation::Constant(operation) => return self.constant(operation.value()),
            ArrayOperation::Zero(_) | ArrayOperation::ZeroLike(_) => {
                format!("ct.full({}, 0, ct.{data_type})", physical_shape(&shape))
            }
            ArrayOperation::One(_) | ArrayOperation::OneLike(_) => {
                format!("ct.full({}, 1, ct.{data_type})", physical_shape(&shape))
            }
            ArrayOperation::FillLike(operation) => {
                // The fill value is a rank-0 literal, so its logical bytes encode exactly one element.
                let bytes = operation.value().logical_bytes();
                let mut bits = [0u8; 8];
                bits[..bytes.len()].copy_from_slice(&bytes);
                let scalar = literal_expression(output.data_type(), u64::from_le_bytes(bits))?;
                format!("ct.broadcast_to({scalar}, {})", physical_shape(&shape))
            }
            ArrayOperation::Iota(iota) => iota_expression(&shape, iota.dimension(), data_type),
            ArrayOperation::IotaLike(iota) => iota_expression(&shape, iota.dimension(), data_type),
            ArrayOperation::Add(_) => format!("({} + {})", inputs[0], inputs[1]),
            ArrayOperation::Sub(_) => format!("({} - {})", inputs[0], inputs[1]),
            ArrayOperation::SaturatingSub(_) => {
                return Err(unsupported(operation.name(), "integer saturation has no qualified cuTile implementation"));
            }
            ArrayOperation::Mul(_) => format!("({} * {})", inputs[0], inputs[1]),
            ArrayOperation::Div(_) if matches!(output.data_type(), DataType::F16 | DataType::F32 | DataType::F64) => {
                format!("ct.truediv({}, {})", inputs[0], inputs[1])
            }
            ArrayOperation::Neg(_) => format!("(-{})", inputs[0]),
            ArrayOperation::Exp(_) => format!("ct.exp({})", inputs[0]),
            ArrayOperation::Max(_) => format!("ct.maximum({}, {})", inputs[0], inputs[1]),
            ArrayOperation::Min(_) => format!("ct.minimum({}, {})", inputs[0], inputs[1]),
            ArrayOperation::Compare(operation) => {
                format!("({} {} {})", inputs[0], comparison(operation.direction()), inputs[1])
            }
            ArrayOperation::Select(_) => format!("ct.where({}, {}, {})", inputs[0], inputs[1], inputs[2]),
            ArrayOperation::ConvertElementType(operation) => {
                if operation.bitcast() {
                    let input_shape = checked_shape(&types[0])?;
                    if input_shape != shape
                        || types[0].data_type() == DataType::Boolean
                        || output.data_type() == DataType::Boolean
                    {
                        return Err(unsupported(
                            operation.name(),
                            "bitcasts require equal-width non-Boolean element types",
                        ));
                    }
                    format!("ct.bitcast({}, ct.{data_type})", inputs[0])
                } else {
                    format!("ct.astype({}, ct.{data_type})", inputs[0])
                }
            }
            ArrayOperation::Reshape(_)
                if checked_shape(&types[0])?.iter().all(|extent| extent.is_power_of_two())
                    && shape.iter().all(|extent| extent.is_power_of_two()) =>
            {
                format!("ct.reshape({}, {})", inputs[0], physical_shape(&shape))
            }
            ArrayOperation::Broadcast(broadcast) => {
                broadcast_expression(operation.name(), &inputs[0], &types[0], broadcast.output_axes(), &shape)?
            }
            ArrayOperation::BroadcastLike(broadcast) => {
                broadcast_expression(operation.name(), &inputs[0], &types[0], broadcast.output_axes(), &shape)?
            }
            ArrayOperation::Transpose(operation) => {
                let permutation = operation.permutation().normalize(shape.len()).map_err(ProgramError::from)?;
                format!("ct.permute({}, {})", inputs[0], tuple(&permutation))
            }
            ArrayOperation::Reduce(operation)
                if matches!(operation.kind(), ReductionKind::Sum | ReductionKind::Max | ReductionKind::Min) =>
            {
                let input_shape = checked_shape(&types[0])?;
                let identity = match operation.kind() {
                    ReductionKind::Sum => "0",
                    ReductionKind::Max => "float('-inf')",
                    _ => "float('inf')",
                };
                if operation.kind() != ReductionKind::Sum
                    && !matches!(output.data_type(), DataType::F32 | DataType::F64)
                {
                    return Err(unsupported(operation.name(), "integer extremum reduction identities are unsupported"));
                }
                let (_, mask) = self.indices(&vec!["0".to_owned(); input_shape.len()], &input_shape);
                let mut result = self.assign(format!("ct.where({mask}, {}, {identity})", inputs[0]));
                let mut axes = operation.axes().to_vec();
                axes.sort_unstable_by(|left, right| right.cmp(left));
                for axis in axes {
                    let function = match operation.kind() {
                        ReductionKind::Sum => "sum",
                        ReductionKind::Max => "max",
                        _ => "min",
                    };
                    result = self.assign(format!("ct.{function}({result}, axis={axis})"));
                }
                result
            }
            ArrayOperation::Dot(operation)
                if operation.dimensions() == &DotDimensionNumbers::matmul()
                    && types[0].rank() == 2
                    && types[1].rank() == 2
                    && matches!(output.data_type(), DataType::F16 | DataType::F32 | DataType::F64) =>
            {
                // `ct.mma` accumulates in the element type of its accumulator, which is the output type and therefore
                // the requested accumulation type. How it converts the operands for an explicit precision or algorithm
                // has not been qualified, so only the default precision is admitted.
                if !operation.precision().is_default() {
                    let precision = operation.precision();
                    return Err(unsupported(
                        operation.name(),
                        &format!("precision `{precision}` has no qualified cuTile MMA conversion"),
                    ));
                }
                if operation.output_sharding().is_some() {
                    return Err(unsupported(operation.name(), "output shardings are unsupported"));
                }
                if types[0].data_type() != types[1].data_type()
                    || !matches!(
                        (types[0].data_type(), output.data_type()),
                        (DataType::F16, DataType::F16 | DataType::F32)
                            | (DataType::BF16, DataType::F32)
                            | (DataType::F32, DataType::F32)
                            | (DataType::F64, DataType::F64)
                    )
                {
                    return Err(unsupported(operation.name(), "unsupported cuTile MMA accumulation type"));
                }
                let left_shape = checked_shape(&types[0])?;
                let right_shape = checked_shape(&types[1])?;
                let (_, left_mask) = self.indices(&vec!["0".to_owned(); 2], &left_shape);
                let (_, right_mask) = self.indices(&vec!["0".to_owned(); 2], &right_shape);
                format!(
                    "ct.mma(ct.where({left_mask}, {}, 0), ct.where({right_mask}, {}, 0), ct.full({}, 0, ct.{data_type}))",
                    inputs[0],
                    inputs[1],
                    physical_shape(&shape)
                )
            }
            ArrayOperation::ScaledDot(_) => {
                return Err(unsupported(
                    operation.name(),
                    "canonical low-precision byte storage and scaled-dot rounding are not interchangeable with packed cuTile mma_scaled",
                ));
            }
            _ => return Err(unsupported(operation.name(), "array operation or geometry is outside the cuTile subset")),
        };
        Ok(self.assign(result))
    }
}

/// Renders a Python tuple, preserving singleton and empty syntax.
fn tuple<T: std::fmt::Display>(values: &[T]) -> String {
    format!(
        "({}{})",
        values.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
        if values.len() == 1 { "," } else { "" }
    )
}

/// Returns power-of-two physical tile extents.
fn physical_shape(shape: &[usize]) -> String {
    tuple(&shape.iter().map(|extent| extent.max(&1).next_power_of_two()).collect::<Vec<_>>())
}

/// Spells one scalar of `data_type` from its exact little-endian `bits` without decimal floating-point conversion.
fn literal_expression(data_type: DataType, bits: u64) -> Result<String, Error> {
    Ok(match data_type {
        DataType::F16 | DataType::BF16 => {
            format!("ct.bitcast(ct.full((), {bits}, ct.uint16), ct.{})", dtype(data_type)?)
        }
        DataType::F32 => format!("ct.bitcast(ct.full((), {bits}, ct.uint32), ct.float32)"),
        DataType::F64 => format!("ct.bitcast(ct.full((), {bits}, ct.uint64), ct.float64)"),
        DataType::I32 => format!("ct.full((), {}, ct.int32)", bits as u32 as i32),
        DataType::I64 => format!("ct.full((), {}, ct.int64)", bits as i64),
        DataType::Boolean => format!("ct.full((), {}, ct.bool_)", if bits == 0 { "False" } else { "True" }),
        data_type => format!("ct.full((), {bits}, ct.{})", dtype(data_type)?),
    })
}

/// Spells the coordinates of a tile with the logical `shape` along `dimension`, converted to the cuTile element type
/// named `data_type`. Coordinates in the physical padding exceed the logical extent, which is harmless because padding
/// never reaches global memory.
fn iota_expression(shape: &[usize], dimension: usize, data_type: &str) -> String {
    let mut expanded = vec![1; shape.len()];
    expanded[dimension] = shape[dimension].max(1).next_power_of_two();
    format!(
        "ct.astype(ct.broadcast_to(ct.reshape(ct.arange({}, dtype=ct.int64), {}), {}), ct.{data_type})",
        expanded[dimension],
        tuple(&expanded),
        physical_shape(shape),
    )
}

/// Spells the broadcast of `input` to a tile with the logical `shape`, where `output_axes` contains the output axis of
/// each input axis. The input is first reshaped so that it has unit extents along the axes it does not occupy.
fn broadcast_expression(
    operation: &'static str,
    input: &str,
    input_type: &ArrayType,
    output_axes: &[usize],
    shape: &[usize],
) -> Result<String, Error> {
    if output_axes.windows(2).any(|axes| axes[0] >= axes[1]) {
        return Err(unsupported(operation, "broadcast axes must preserve source order"));
    }
    let input_shape = checked_shape(input_type)?;
    let mut expanded = vec![1; shape.len()];
    for (axis, &output_axis) in output_axes.iter().enumerate() {
        expanded[output_axis] = input_shape[axis].max(1).next_power_of_two();
    }
    Ok(format!("ct.broadcast_to(ct.reshape({input}, {}), {})", tuple(&expanded), physical_shape(shape)))
}

/// Rejects unbounded source expansion and unsupported physical tiles before code generation.
fn checked_tile(shape: &[usize]) -> Result<(), Error> {
    if shape.len() > 4
        || shape
            .iter()
            .try_fold(1usize, |size, extent| size.checked_mul(extent.max(&1).next_power_of_two()))
            .is_none_or(|count| count > 1_048_576)
    {
        return Err(unsupported("array", "tile exceeds rank four or 1048576 physical elements"));
    }
    Ok(())
}

/// Checks the admitted static contiguous device ABI without dropping type metadata.
fn checked_shape(r#type: &ArrayType) -> Result<Vec<usize>, Error> {
    dtype(r#type.data_type())?;
    Parameter::from_type(r#type)?;
    let shape = r#type.static_shape().ok_or_else(|| unsupported("array", "static array shapes are required"))?;
    if shape.dimensions().iter().any(|extent| *extent > i32::MAX as usize) {
        return Err(unsupported("array", "array extent exceeds signed 32-bit indexing"));
    }
    Ok(shape.dimensions().to_vec())
}

/// Exact supported cuTile element spelling; packed and low-precision formats require separate qualification.
fn dtype(data_type: DataType) -> Result<&'static str, Error> {
    if let Some(name) = data_type_name(data_type) {
        return Ok(name);
    }
    match data_type {
        DataType::F4E2M1FN => Err(unsupported(
            "array",
            "canonical FP4 uses byte-per-value storage; packed cuTile FP4 requires an explicit storage and rounding contract",
        )),
        _ => Err(unsupported("array", "element dtype is outside the qualified cuTile subset")),
    }
}

/// Preserves exact ordered comparison semantics.
fn comparison(direction: ComparisonDirection) -> &'static str {
    match direction {
        ComparisonDirection::Equal => "==",
        ComparisonDirection::NotEqual => "!=",
        ComparisonDirection::LessThan => "<",
        ComparisonDirection::LessThanOrEqual => "<=",
        ComparisonDirection::GreaterThan => ">",
        ComparisonDirection::GreaterThanOrEqual => ">=",
    }
}

/// Emits arithmetic only when canonical bounds already prove that it cannot raise an assertion.
fn dimension(operation: &DimensionOperation<DimensionValue>, inputs: &[String]) -> Result<String, Error> {
    if operation.effects().has_execution_effects() {
        return Err(unsupported(operation.name(), "dimension arithmetic must be proven total"));
    }
    Ok(match operation {
        DimensionOperation::Constant(operation) => operation.value().extent().to_string(),
        DimensionOperation::Add(_) => format!("({} + {})", inputs[0], inputs[1]),
        DimensionOperation::Sub(_) => format!("({} - {})", inputs[0], inputs[1]),
        DimensionOperation::Mul(_) => format!("({} * {})", inputs[0], inputs[1]),
        DimensionOperation::Div(_) => format!("({} // {})", inputs[0], inputs[1]),
        DimensionOperation::Rem(_) => format!("({} % {})", inputs[0], inputs[1]),
        DimensionOperation::Min(_) => format!("ct.minimum({}, {})", inputs[0], inputs[1]),
        DimensionOperation::Max(_) => format!("ct.maximum({}, {})", inputs[0], inputs[1]),
        DimensionOperation::SaturatingSub(_) => format!("ct.maximum({} - {}, 0)", inputs[0], inputs[1]),
        _ => return Err(unsupported(operation.name(), "dimension operation is outside the arithmetic subset")),
    })
}

/// Validates that the start bounds of a tile load with the input types `input_types` prove that the host never rejects
/// its window, which the generated source cannot report at run time. Every window limit must fit the canonical
/// dimension width, which also keeps the 64-bit window arithmetic from overflowing, and the window of an in-bounds load
/// must also lie inside the view that its transforms select. An in-bounds load that satisfies this check reads only
/// valid elements, so it lowers like a masked load.
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

/// Attaches a precise owning-operation diagnostic to every unsupported path.
fn unsupported(operation: &'static str, reason: &str) -> Error {
    Error::Unsupported { operation, reason: reason.to_owned() }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArraySliceAxis, AssertOperation, AssertionError, BroadcastLikeOperation, ConditionOperation, Context,
        DimensionBounds, DimensionMulOperation, DimensionType, DotAlgorithm, DotAlgorithmPreset, DotInputPrecision,
        DotOperation, DotPrecision, FillLikeOperation, IotaLikeOperation, IotaOperation, Placeholder, ProgramBuilder,
        ReferenceRead, ReferenceReadOperation, ReferenceWrite, ReferenceWriteOperation, TypeError,
    };
    use ryft_kernels::{
        Grid, GridDimension, KernelCallOperation, KernelDefinition, KernelParameterAccess, MappingExpression,
        ParameterWindow, accumulate, mapped_call, masked_load_or, masked_store, masked_swap, whole_array_parameter,
        window_mask,
    };

    use super::*;

    /// Builds a call over two parallel programs whose parameters have the given types, accesses, and boundary policies.
    /// Every window has three elements, so the second window of a five element array has one padding element, and its
    /// physical tile has four elements.
    fn tiled_call(parameters: Vec<(ArrayType, KernelParameterAccess, BoundaryPolicy)>) -> KernelCallOperation {
        mapped_call(
            Grid::new(vec![GridDimension::new(2usize.into(), GridExecution::Parallel)]).unwrap(),
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

    /// Builds one source-owned scalar copy without backend or tool dependencies.
    fn scalar_copy() -> KernelDefinition {
        let r#type = ArrayType::scalar(DataType::F32);
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(r#type.clone(), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(r#type, KernelParameterAccess::WriteOnly).unwrap(),
            ],
        )
        .unwrap();
        KernelDefinition::trace(call, |(references, _)| references[1].write(&references[0].read()?)).unwrap()
    }

    /// Reason for which admission rejects an in-bounds tile load whose window it cannot prove valid.
    const IN_BOUNDS_REASON: &str =
        "in-bounds tile loads require start bounds that prove the window lies inside the source view";

    /// Lowers `definition` and returns its complete source.
    fn lowered_source(definition: &KernelDefinition) -> String {
        let kernel = VerifiedKernel::new(definition, 1).unwrap();
        lower(&kernel, &Options::default(), &KernelSchedule::default()).unwrap().code
    }

    /// Returns the operation and reason of the admission failure of `definition`.
    fn rejection(definition: &KernelDefinition) -> (&'static str, String) {
        let kernel = VerifiedKernel::new(definition, 1).unwrap();
        match lower(&kernel, &Options::default(), &KernelSchedule::default()) {
            Err(Error::Unsupported { operation, reason }) => (operation, reason),
            Err(error) => panic!("expected an admission failure, got {error:?}"),
            Ok(_) => panic!("expected an admission failure"),
        }
    }

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
        builder.add_instruction(ReferenceWriteOperation::new(), vec![], vec![inputs[1], tile], None).unwrap();
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
        let inputs =
            empty_grid.body_input_types().into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let tile = load(&mut builder, inputs[0]);
        builder.add_instruction(ReferenceWriteOperation::new(), vec![], vec![inputs[1], tile], None).unwrap();
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
        let inputs =
            untaken_branch.body_input_types().into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
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
        builder.add_instruction(ReferenceWriteOperation::new(), vec![], vec![inputs[1], tile], None).unwrap();
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

    /// Builds the assertion fixture, which checks the constant `condition` with the zero result `operation` and then
    /// copies its one element input to its output.
    fn assertion_definition(operation: KernelOperation, condition: bool) -> KernelDefinition {
        let r#type = ArrayType::new_static(DataType::F32, [1]);
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(r#type.clone(), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(r#type, KernelParameterAccess::WriteOnly).unwrap(),
            ],
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let inputs = call.body_input_types().into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let condition = builder.add_constant(ArrayIrValue::Array(Array::scalar(condition).unwrap()));
        builder.add_instruction(operation, vec![], vec![condition], None).unwrap();
        let value = builder.add_instruction(ReferenceReadOperation::new(), vec![], vec![inputs[0]], None).unwrap()[0];
        builder.add_instruction(ReferenceWriteOperation::new(), vec![], vec![inputs[1], value], None).unwrap();
        KernelDefinition::new(call, builder.build(vec![], vec![Placeholder; 2], vec![]).unwrap()).unwrap()
    }

    #[test]
    fn test_lower() {
        let definition = scalar_copy();
        let kernel = VerifiedKernel::new(&definition, 1).unwrap();
        let source = lower(&kernel, &Options::default(), &KernelSchedule::default()).unwrap();
        assert_eq!(
            source.code,
            indoc! {"
            import cuda.tile as ct

            @ct.kernel
            def ryft_cutile_kernel(p0, p1):
                v0 = ct.reshape(ct.load(p0, (0,), (1,)), ())
                ct.store(p1, (0,), ct.reshape(v0, (1,)))
                return
        "}
        );
        assert_eq!(source.grid, [1, 1, 1]);
        assert_eq!(
            source.parameters,
            vec![
                Parameter { dtype: "float32".to_owned(), shape: vec![1], strides: vec![1] },
                Parameter { dtype: "float32".to_owned(), shape: vec![1], strides: vec![1] },
            ]
        );
        let repeated = lower(&kernel, &Options::default(), &KernelSchedule::default()).unwrap();
        assert_eq!(source.code, repeated.code);
    }

    #[test]
    fn test_lower_batched_folded_transforms() {
        let definition = scalar_copy().batched(2, &[Some(0), Some(0)], 2).unwrap();
        let kernel = VerifiedKernel::new(&definition, 2).unwrap();
        let source = lower(&kernel, &Options::default(), &KernelSchedule::default()).unwrap();
        assert_eq!(source.grid, [2, 1, 1]);
        assert_eq!(
            source.parameters.iter().map(|parameter| parameter.shape.clone()).collect::<Vec<_>>(),
            vec![vec![2]; 2]
        );
        assert_eq!(definition.body().instructions().len(), 2);
    }

    #[test]
    fn test_lower_scratch_budget() {
        let definition = scalar_copy();
        let kernel = VerifiedKernel::new(&definition, 1).unwrap();
        assert!(matches!(
            lower(&kernel, &Options::default(), &KernelSchedule::default().with_maximum_scratch_bytes(0)),
            Err(Error::Unsupported { operation: "kernel_call", reason })
                if reason == "cuTile does not expose a validated compiler scratch-memory bound"
        ));
    }

    #[test]
    fn test_lower_signature_budget() {
        let definition = scalar_copy();
        let kernel = VerifiedKernel::new(&definition, 1).unwrap();
        let options = Options::default().with_maximum_instructions(1).unwrap();
        assert!(matches!(
            lower(&kernel, &options, &KernelSchedule::default()),
            Err(Error::Unsupported { operation: "kernel_call", reason })
                if reason == "source construction exceeds the instruction budget"
        ));
    }

    #[test]
    fn test_lower_dot_precision() {
        // The host default preserves the diagonal value, and the default precision lowers to one `ct.mma` over the
        // unconverted operands that accumulates into an F32 tile.
        let (definition, inputs) = dot_definition(DotOperation::matmul());
        assert_eq!(definition.interpret(inputs.clone(), 1), Ok(vec![inputs[0].clone()]));
        assert_eq!(
            lowered_source(&definition),
            indoc! {"
            import cuda.tile as ct

            @ct.kernel
            def ryft_cutile_kernel(p0, p1, p2):
                v0 = ct.reshape(ct.arange(32, dtype=ct.int64), (32, 1))
                v1 = (0 + v0)
                v2 = ct.reshape(ct.arange(32, dtype=ct.int64), (1, 32))
                v3 = (0 + v2)
                v4 = ct.gather(p0, (v1, v3), mask=((v0 < 32) & (v1 < (0 + 32))) & ((v2 < 32) & (v3 < (0 + 32))), padding_value=0, check_bounds=True)
                v5 = ct.reshape(ct.arange(32, dtype=ct.int64), (32, 1))
                v6 = (0 + v5)
                v7 = ct.reshape(ct.arange(32, dtype=ct.int64), (1, 32))
                v8 = (0 + v7)
                v9 = ct.gather(p1, (v6, v8), mask=((v5 < 32) & (v6 < (0 + 32))) & ((v7 < 32) & (v8 < (0 + 32))), padding_value=0, check_bounds=True)
                v10 = ct.reshape(ct.arange(32, dtype=ct.int32), (32, 1))
                v11 = ct.reshape(ct.arange(32, dtype=ct.int32), (1, 32))
                v12 = ct.reshape(ct.arange(32, dtype=ct.int32), (32, 1))
                v13 = ct.reshape(ct.arange(32, dtype=ct.int32), (1, 32))
                v14 = ct.mma(ct.where((v10 < 32) & (v11 < 32), v4, 0), ct.where((v12 < 32) & (v13 < 32), v9, 0), ct.full((32, 32), 0, ct.float32))
                v15 = ct.reshape(ct.arange(32, dtype=ct.int64), (32, 1))
                v16 = (0 + v15)
                v17 = ct.reshape(ct.arange(32, dtype=ct.int64), (1, 32))
                v18 = (0 + v17)
                ct.scatter(p2, (v16, v18), v14, mask=((v15 < 32) & (v16 < (0 + 32))) & ((v17 < 32) & (v18 < (0 + 32))), check_bounds=True)
                return
        "},
        );

        // Rounding the operands to BF16 rounds the diagonal value to 1 on the host. The conversions of `ct.mma` for an
        // explicit precision are not qualified, so admission rejects every precision other than the default.
        let (definition, inputs) =
            dot_definition(DotOperation::matmul().with_precision(DotAlgorithmPreset::BF16_BF16_F32));
        assert_eq!(definition.interpret(inputs.clone(), 1), Ok(vec![inputs[1].clone()]));
        assert_eq!(
            rejection(&definition),
            ("dot", "precision `bf16_bf16_f32` has no qualified cuTile MMA conversion".to_owned()),
        );
        for precision in [
            DotPrecision::from(DotInputPrecision::High),
            DotPrecision::from(DotInputPrecision::Highest),
            DotPrecision::from((DotInputPrecision::Default, DotInputPrecision::Highest)),
            DotPrecision::from(DotAlgorithmPreset::F32_F32_F32),
            DotPrecision::from(DotAlgorithmPreset::TF32_TF32_F32),
            DotPrecision::from(DotAlgorithm::new(DataType::F32, DataType::F32, DataType::F32)),
        ] {
            let (definition, _) = dot_definition(DotOperation::matmul().with_precision(precision));
            assert_eq!(
                rejection(&definition),
                ("dot", format!("precision `{precision}` has no qualified cuTile MMA conversion")),
            );
        }
    }

    #[test]
    fn test_lower_assertions() {
        // A true assertion has no results, so the copy that follows it returns `[5.0]` on the host, while a false
        // assertion fails. The generated source cannot report an assertion failure to its caller, so admission rejects
        // both forms of the assertion whatever its condition, instead of dropping it.
        let source = Array::vector(vec![5f32]).unwrap();
        let reason = "runtime assertions require native failure propagation".to_owned();
        for operation in [
            KernelOperation::Portable(ArrayIrOperation::Array(ArrayOperation::Assert(AssertOperation::new("check")))),
            KernelOperation::Portable(ArrayIrOperation::Assert(AssertOperation::new("check"))),
        ] {
            let passing = assertion_definition(operation.clone(), true);
            assert_eq!(passing.interpret(vec![source.clone()], 1), Ok(vec![source.clone()]));
            assert_eq!(rejection(&passing), ("assert", reason.clone()));
            let failing = assertion_definition(operation, false);
            assert_eq!(
                failing.interpret(vec![source.clone()], 1),
                Err(ProgramError::from(AssertionError::Failed { message: "check".to_owned(), observations: vec![] })),
            );
            assert_eq!(rejection(&failing), ("assert", reason.clone()));
        }
    }

    #[test]
    fn test_lower_tile_load() {
        // A masked window at element 3 of `[3, 5, 7, 9]` gathers its valid element and takes the padding value for
        // the element beyond the source view.
        let source = Array::vector(vec![3f32, 5.0, 7.0, 9.0]).unwrap();
        let definition = tile_load_definition(BoundaryPolicy::Masked, 3, vec![]);
        assert_eq!(definition.interpret(vec![source.clone()], 1), Ok(vec![Array::vector(vec![9f32, -1.0]).unwrap()]));
        assert_eq!(
            lowered_source(&definition),
            indoc! {"
            import cuda.tile as ct

            @ct.kernel
            def ryft_cutile_kernel(p0, p1):
                v0 = ct.broadcast_to(ct.bitcast(ct.full((), 3212836864, ct.uint32), ct.float32), ())
                v1 = ct.reshape(ct.arange(2, dtype=ct.int64), (2,))
                v2 = ((0 + 3) + v1)
                v3 = ct.gather(p0, (v2,), mask=((v1 < 2) & (v2 < (0 + 4))), padding_value=v0, check_bounds=True)
                v4 = ct.reshape(ct.arange(2, dtype=ct.int64), (2,))
                v5 = (0 + v4)
                ct.scatter(p1, (v5,), v3, mask=((v4 < 2) & (v5 < (0 + 2))), check_bounds=True)
                return
        "},
        );

        // The same window under the in-bounds policy fails on the host. cuTile cannot report that failure at run time,
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
            lowered_source(&definition),
            lowered_source(&tile_load_definition(BoundaryPolicy::Masked, 2, vec![])),
        );

        // Bounds apply to the selected view `[5, 7, 9]` rather than to the whole source, so the window at element 2 of
        // that view fails even though it lies inside the source.
        let slice = || vec![ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 3, 1)] }];
        let definition = tile_load_definition(BoundaryPolicy::InBounds, 1, slice());
        assert_eq!(definition.interpret(vec![source.clone()], 1), Ok(vec![Array::vector(vec![7f32, 9.0]).unwrap()]));
        assert_eq!(
            lowered_source(&definition),
            lowered_source(&tile_load_definition(BoundaryPolicy::Masked, 1, slice())),
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
        lowered_source(&definition);

        // A window limit beyond the canonical dimension width fails on the host under either policy, and the 64-bit
        // window arithmetic of cuTile would overflow, so admission rejects it even for a masked load.
        let definition = tile_load_definition(BoundaryPolicy::Masked, MAX_DIMENSION_EXTENT - 1, vec![]);
        assert_eq!(
            definition.interpret(vec![source], 1),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{TILE_LOAD_OPERATION_NAME}` window limit exceeds the canonical dimension width",
            )))),
        );
        assert_eq!(
            rejection(&definition),
            (
                TILE_LOAD_OPERATION_NAME,
                "tile load start bounds must prove that the window limit fits the canonical dimension width".to_owned(),
            ),
        );
    }

    #[test]
    fn test_lower_tile_load_unreachable() {
        // Admission cannot tell whether an in-bounds load runs, so it rejects the load even where the host never
        // executes it.
        let source = Array::vector(vec![3f32, 5.0, 7.0, 9.0]).unwrap();
        let [empty_grid, untaken_branch] = unreachable_tile_load_definitions().try_into().ok().unwrap();
        let output = Array::vector(vec![1f32, 2.0]).unwrap();
        assert_eq!(empty_grid.interpret(vec![source.clone(), output.clone()], 1), Ok(vec![output]));
        assert_eq!(rejection(&empty_grid), (TILE_LOAD_OPERATION_NAME, IN_BOUNDS_REASON.to_owned()));
        assert_eq!(untaken_branch.interpret(vec![source], 1), Ok(vec![Array::vector(vec![-1f32, -1.0]).unwrap()]));
        assert_eq!(rejection(&untaken_branch), (TILE_LOAD_OPERATION_NAME, IN_BOUNDS_REASON.to_owned()));
    }

    #[test]
    fn test_lower_masked_load() {
        // The masked window limits clip to the array, so the padding element of the second window and the physical
        // padding of every tile take the value `other`.
        let operation = tiled_call(vec![
            (ArrayType::new_static(DataType::F32, [5]), KernelParameterAccess::ReadOnly, BoundaryPolicy::Masked),
            (ArrayType::new_static(DataType::F32, [6]), KernelParameterAccess::WriteOnly, BoundaryPolicy::InBounds),
        ]);
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            references[1].write(&masked_load_or(references[0].context(), &references[0], -1f32)?)
        })
        .unwrap();
        let kernel = VerifiedKernel::new(&definition, 2).unwrap();
        let source = lower(&kernel, &Options::default(), &KernelSchedule::default()).unwrap();
        assert_eq!(
            source.code,
            indoc! {"
            import cuda.tile as ct

            @ct.kernel
            def ryft_cutile_kernel(p0, p1):
                v0 = (ct.astype(ct.bid(0), ct.int64) % 2)
                v1 = (ct.astype(ct.bid(0), ct.int64) // 2)
                v2 = (v0 * 3)
                v3 = (v0 * 3)
                v4 = ct.broadcast_to(ct.full((), True, ct.bool_), ())
                v5 = ct.broadcast_to(ct.reshape(v4, (1,)), (4,))
                v6 = ct.broadcast_to(ct.bitcast(ct.full((), 3212836864, ct.uint32), ct.float32), ())
                v7 = ct.broadcast_to(ct.reshape(v6, (1,)), (4,))
                v8 = ct.reshape(ct.arange(4, dtype=ct.int64), (4,))
                v9 = (v2 + v8)
                v10 = (((v8 < 3) & (v9 < ct.minimum((v2 + 3), 5)))) & (v5)
                v11 = ct.gather(p0, (v9,), mask=v10, padding_value=0, check_bounds=True)
                v12 = ct.where(v10, v11, v7)
                v13 = ct.reshape(ct.arange(4, dtype=ct.int64), (4,))
                v14 = (v3 + v13)
                ct.scatter(p1, (v14,), v12, mask=((v13 < 3) & (v14 < (v3 + 3))), check_bounds=True)
                return
        "}
        );
    }

    #[test]
    fn test_lower_masked_store() {
        let operation = tiled_call(vec![(
            ArrayType::new_static(DataType::F32, [5]),
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
        let kernel = VerifiedKernel::new(&definition, 2).unwrap();
        let source = lower(&kernel, &Options::default(), &KernelSchedule::default()).unwrap();
        assert_eq!(
            source.code,
            indoc! {"
            import cuda.tile as ct

            @ct.kernel
            def ryft_cutile_kernel(p0):
                v0 = (ct.astype(ct.bid(0), ct.int64) % 2)
                v1 = (ct.astype(ct.bid(0), ct.int64) // 2)
                v2 = (v0 * 3)
                v3 = ct.broadcast_to(ct.bitcast(ct.full((), 1084227584, ct.uint32), ct.float32), (4,))
                v4 = ct.reshape(ct.arange(4, dtype=ct.int64), (4,))
                v5 = (v2 + v4)
                v6 = ct.broadcast_to(((v4 < 3) & (v5 < ct.minimum((v2 + 3), 5))), (4,))
                v7 = ct.reshape(ct.arange(4, dtype=ct.int64), (4,))
                v8 = (v2 + v7)
                v9 = (((v7 < 3) & (v8 < ct.minimum((v2 + 3), 5)))) & (v6)
                ct.scatter(p0, (v8,), v3, mask=v9, check_bounds=True)
                return
        "}
        );
    }

    #[test]
    fn test_lower_masked_swap() {
        let operation = tiled_call(vec![
            (ArrayType::new_static(DataType::F32, [5]), KernelParameterAccess::ReadWrite, BoundaryPolicy::Masked),
            (ArrayType::new_static(DataType::F32, [6]), KernelParameterAccess::WriteOnly, BoundaryPolicy::InBounds),
        ]);
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            let context = references[0].context();
            let mask = context.lift(ArrayIrValue::Array(Array::vector(vec![true; 3])?))?;
            let replacement = context.lift(ArrayIrValue::Array(Array::vector(vec![9f32; 3])?))?;
            references[1].write(&masked_swap(context, &references[0], &replacement, &mask, -1f32)?)
        })
        .unwrap();
        let kernel = VerifiedKernel::new(&definition, 2).unwrap();
        let source = lower(&kernel, &Options::default(), &KernelSchedule::default()).unwrap();
        assert_eq!(
            source.code,
            indoc! {"
            import cuda.tile as ct

            @ct.kernel
            def ryft_cutile_kernel(p0, p1):
                v0 = (ct.astype(ct.bid(0), ct.int64) % 2)
                v1 = (ct.astype(ct.bid(0), ct.int64) // 2)
                v2 = (v0 * 3)
                v3 = (v0 * 3)
                v4 = ct.broadcast_to(ct.full((), True, ct.bool_), (4,))
                v5 = ct.broadcast_to(ct.bitcast(ct.full((), 1091567616, ct.uint32), ct.float32), (4,))
                v6 = ct.broadcast_to(ct.bitcast(ct.full((), 3212836864, ct.uint32), ct.float32), ())
                v7 = ct.broadcast_to(ct.reshape(v6, (1,)), (4,))
                v8 = ct.reshape(ct.arange(4, dtype=ct.int64), (4,))
                v9 = (v2 + v8)
                v10 = (((v8 < 3) & (v9 < ct.minimum((v2 + 3), 5)))) & (v4)
                v11 = ct.gather(p0, (v9,), mask=v10, padding_value=0, check_bounds=True)
                v12 = ct.where(v10, v11, v7)
                ct.scatter(p0, (v9,), v5, mask=v10, check_bounds=True)
                v13 = ct.reshape(ct.arange(4, dtype=ct.int64), (4,))
                v14 = (v3 + v13)
                ct.scatter(p1, (v14,), v12, mask=((v13 < 3) & (v14 < (v3 + 3))), check_bounds=True)
                return
        "}
        );
    }

    #[test]
    fn test_lower_window_mask() {
        // The mask is false at the padding element of the second window and at the physical padding of every tile.
        let operation = tiled_call(vec![
            (ArrayType::new_static(DataType::F32, [5]), KernelParameterAccess::ReadOnly, BoundaryPolicy::Masked),
            (ArrayType::new_static(DataType::Boolean, [6]), KernelParameterAccess::WriteOnly, BoundaryPolicy::InBounds),
        ]);
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            references[1].write(&window_mask(references[0].context(), &references[0])?)
        })
        .unwrap();
        let kernel = VerifiedKernel::new(&definition, 2).unwrap();
        let source = lower(&kernel, &Options::default(), &KernelSchedule::default()).unwrap();
        assert_eq!(
            source.code,
            indoc! {"
            import cuda.tile as ct

            @ct.kernel
            def ryft_cutile_kernel(p0, p1):
                v0 = (ct.astype(ct.bid(0), ct.int64) % 2)
                v1 = (ct.astype(ct.bid(0), ct.int64) // 2)
                v2 = (v0 * 3)
                v3 = (v0 * 3)
                v4 = ct.reshape(ct.arange(4, dtype=ct.int64), (4,))
                v5 = (v2 + v4)
                v6 = ct.broadcast_to(((v4 < 3) & (v5 < ct.minimum((v2 + 3), 5))), (4,))
                v7 = ct.reshape(ct.arange(4, dtype=ct.int64), (4,))
                v8 = (v3 + v7)
                ct.scatter(p1, (v8,), v6, mask=((v7 < 3) & (v8 < (v3 + 3))), check_bounds=True)
                return
        "}
        );
    }

    #[test]
    fn test_lower_reference_add_update() {
        // The update gathers the window, adds the update to it, and scatters the sum back under the same mask.
        let operation = tiled_call(vec![(
            ArrayType::new_static(DataType::F32, [6]),
            KernelParameterAccess::ReadWrite,
            BoundaryPolicy::InBounds,
        )]);
        let definition: KernelDefinition = KernelDefinition::trace(operation, |(references, _)| {
            let context = references[0].context();
            let value = context.lift(ArrayIrValue::Array(Array::vector(vec![5f32; 3])?))?;
            accumulate(context, &references[0], &value, BoundaryPolicy::InBounds)
        })
        .unwrap();
        let kernel = VerifiedKernel::new(&definition, 2).unwrap();
        let source = lower(&kernel, &Options::default(), &KernelSchedule::default()).unwrap();
        assert_eq!(
            source.code,
            indoc! {"
            import cuda.tile as ct

            @ct.kernel
            def ryft_cutile_kernel(p0):
                v0 = (ct.astype(ct.bid(0), ct.int64) % 2)
                v1 = (ct.astype(ct.bid(0), ct.int64) // 2)
                v2 = (v0 * 3)
                v3 = ct.broadcast_to(ct.bitcast(ct.full((), 1084227584, ct.uint32), ct.float32), (4,))
                v4 = ct.reshape(ct.arange(4, dtype=ct.int64), (4,))
                v5 = (v2 + v4)
                v6 = ct.gather(p0, (v5,), mask=((v4 < 3) & (v5 < (v2 + 3))), padding_value=0, check_bounds=True)
                v7 = (v6 + v3)
                ct.scatter(p0, (v5,), v7, mask=((v4 < 3) & (v5 < (v2 + 3))), check_bounds=True)
                return
        "}
        );

        // A rank-0 parameter has no window, so its only element is loaded and stored directly.
        let r#type = ArrayType::scalar(DataType::F32);
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(r#type, KernelParameterAccess::ReadWrite).unwrap()],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(call, |(references, _)| {
            let context = references[0].context();
            let value = context.lift(ArrayIrValue::Array(Array::scalar(5f32)?))?;
            accumulate(context, &references[0], &value, BoundaryPolicy::InBounds)
        })
        .unwrap();
        let kernel = VerifiedKernel::new(&definition, 1).unwrap();
        let source = lower(&kernel, &Options::default(), &KernelSchedule::default()).unwrap();
        assert_eq!(
            source.code,
            indoc! {"
            import cuda.tile as ct

            @ct.kernel
            def ryft_cutile_kernel(p0):
                v0 = ct.broadcast_to(ct.bitcast(ct.full((), 1084227584, ct.uint32), ct.float32), ())
                v1 = ct.reshape(ct.load(p0, (0,), (1,)), ())
                v2 = (v1 + v0)
                ct.store(p0, (0,), ct.reshape(v2, (1,)))
                return
        "}
        );
    }

    #[test]
    fn test_lowering_charge() {
        let mut lowering = Lowering { code: String::new(), indentation: 0, next: 0, remaining: 1 };
        assert!(lowering.charge(1).is_ok());
        assert!(matches!(lowering.charge(1), Err(Error::Unsupported { operation: "kernel_call", reason })
            if reason == "source construction exceeds the instruction budget"));
    }

    #[test]
    fn test_lowering_apply_transform() {
        let mut lowering = Lowering { code: String::new(), indentation: 0, next: 0, remaining: 10 };
        let mut reference = Value::Reference {
            parameter: 0,
            starts: vec!["8".to_owned()],
            shape: vec![4],
            limits: vec!["12".to_owned()],
            axes: vec![0],
            fixed: vec![None],
            validity: vec![],
        };
        lowering
            .apply_transform(
                &mut reference,
                &ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] },
                "reference_read",
            )
            .unwrap();
        lowering
            .apply_transform(
                &mut reference,
                &ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
                "reference_read",
            )
            .unwrap();
        let Value::Reference { starts, shape, limits, axes, fixed, validity, .. } = &reference else { unreachable!() };
        assert_eq!((starts.len(), shape.len(), limits.len(), axes.len()), (0, 0, 0, 0));
        assert_eq!(fixed, &vec![Some("((8 + 1) + 1)".to_owned())]);
        assert_eq!(validity, &vec!["(((8 + 1) + 1) < ct.minimum(12, ((8 + 1) + 2)))".to_owned()]);
        assert!(matches!(lowering.apply_transform(&mut reference, &ArrayReferenceTransform::Index {
            axis: 0, index: ArrayReferenceTransformIndex::Dynamic,
        }, "reference_read"), Err(Error::Unsupported { operation: "reference_read", reason })
            if reason == "dynamic reference transforms are unsupported"));
        assert!(matches!(lowering.apply_transform(&mut reference, &ArrayReferenceTransform::Slice {
            axes: vec![ArraySliceAxis::new(0, 1, 2)],
        }, "reference_read"), Err(Error::Unsupported { operation: "reference_read", reason })
            if reason == "strided reference transforms are unsupported"));
    }

    #[test]
    fn test_lowering_window_indices() {
        let mut lowering = Lowering { code: String::new(), indentation: 0, next: 0, remaining: 10 };
        let (indices, mask) = lowering.window_indices(&["16".to_owned()], &[3], &["18".to_owned()]);
        assert_eq!(indices, vec!["v1".to_owned()]);
        assert_eq!(mask, "((v0 < 3) & (v1 < 18))");
        assert_eq!(
            lowering.code,
            indoc! {"
            v0 = ct.reshape(ct.arange(4, dtype=ct.int64), (4,))
            v1 = (16 + v0)
        "}
        );
    }

    #[test]
    fn test_lowering_read_indexed_reference() {
        let mut lowering = Lowering { code: String::new(), indentation: 0, next: 0, remaining: 10 };
        let reference = Value::Reference {
            parameter: 0,
            starts: vec!["8".to_owned()],
            shape: vec![2],
            limits: vec!["10".to_owned()],
            axes: vec![1],
            fixed: vec![Some("3".to_owned()), None],
            validity: vec![],
        };
        assert_eq!(lowering.read(&reference, "0").unwrap(), "v2");
        assert_eq!(
            lowering.code,
            concat!(
                "v0 = ct.reshape(ct.arange(2, dtype=ct.int64), (2,))\n",
                "v1 = (8 + v0)\n",
                "v2 = ct.gather(p0, (3, v1), mask=((v0 < 2) & (v1 < 10)), padding_value=0, check_bounds=True)\n",
            )
        );
    }

    #[test]
    fn test_lowering_array_bitcast() {
        let mut lowering = Lowering { code: String::new(), indentation: 0, next: 0, remaining: 10 };
        let operation =
            ArrayOperation::ConvertElementType(ryft_core::ConvertElementTypeOperation::new(DataType::U32, true));
        assert_eq!(
            lowering
                .array(
                    &operation,
                    &["input".to_owned()],
                    &[ArrayType::scalar(DataType::F32)],
                    &ArrayType::scalar(DataType::U32),
                )
                .unwrap(),
            "v0"
        );
        assert_eq!(lowering.code, "v0 = ct.bitcast(input, ct.uint32)\n");
    }

    #[test]
    fn test_lowering_array_transpose_then_reshape() {
        let mut lowering = Lowering { code: String::new(), indentation: 0, next: 0, remaining: 10 };
        let matrix_type = ArrayType::new_static(DataType::F32, [2, 2]);
        let output = ArrayType::new_static(DataType::F32, [4]);
        let transpose = ArrayOperation::Transpose(ryft_core::TransposeOperation::new(vec![1, 0]));
        let transposed =
            lowering.array(&transpose, &["input".to_owned()], &[matrix_type.clone()], &matrix_type).unwrap();
        let reshape = ArrayOperation::Reshape(ryft_core::ReshapeOperation::new(
            matrix_type.shape().clone(),
            output.shape().clone(),
        ));
        assert_eq!(lowering.array(&reshape, &[transposed], &[matrix_type], &output).unwrap(), "v1");
        assert_eq!(
            lowering.code,
            indoc! {"
            v0 = ct.permute(input, (1, 0))
            v1 = ct.reshape(v0, (4,))
        "}
        );
    }

    #[test]
    fn test_lowering_array_fill_like() {
        // Floating literals are spelled as integer bits, which preserves signs and NaN payloads exactly.
        let mut lowering = Lowering { code: String::new(), indentation: 0, next: 0, remaining: 10 };
        for (value, shape) in [
            (Array::scalar(2.5f32).unwrap(), vec![2, 3]),
            (Array::scalar(f32::from_bits(0xffc0_0001)).unwrap(), vec![2, 3]),
            (Array::scalar(-7i64).unwrap(), vec![3]),
            (Array::scalar(true).unwrap(), vec![]),
        ] {
            let r#type = ArrayType::new_static(value.r#type().data_type(), shape);
            let operation = ArrayOperation::FillLike(FillLikeOperation::new(value).unwrap());
            lowering.array(&operation, &["exemplar".to_owned()], &[r#type.clone()], &r#type).unwrap();
        }
        assert_eq!(
            lowering.code,
            indoc! {"
            v0 = ct.broadcast_to(ct.bitcast(ct.full((), 1075838976, ct.uint32), ct.float32), (2, 4))
            v1 = ct.broadcast_to(ct.bitcast(ct.full((), 4290772993, ct.uint32), ct.float32), (2, 4))
            v2 = ct.broadcast_to(ct.full((), -7, ct.int64), (4,))
            v3 = ct.broadcast_to(ct.full((), True, ct.bool_), ())
        "}
        );
    }

    #[test]
    fn test_lowering_array_iota() {
        let mut lowering = Lowering { code: String::new(), indentation: 0, next: 0, remaining: 10 };
        let output = ArrayType::new_static(DataType::I32, [2, 3]);
        let iota = ArrayOperation::Iota(IotaOperation::new(output.clone(), 1).unwrap());
        assert_eq!(lowering.array(&iota, &[], &[], &output).unwrap(), "v0");
        let exemplar = ArrayType::new_static(DataType::Boolean, [2, 3, 5]);
        let iota_like = ArrayOperation::IotaLike(IotaLikeOperation::new(1, DataType::F32).unwrap());
        let output = ArrayType::new_static(DataType::F32, [2, 3, 5]);
        assert_eq!(lowering.array(&iota_like, &["exemplar".to_owned()], &[exemplar], &output).unwrap(), "v1");
        assert_eq!(
            lowering.code,
            indoc! {"
            v0 = ct.astype(ct.broadcast_to(ct.reshape(ct.arange(4, dtype=ct.int64), (1, 4)), (2, 4)), ct.int32)
            v1 = ct.astype(ct.broadcast_to(ct.reshape(ct.arange(4, dtype=ct.int64), (1, 4, 1)), (2, 4, 8)), ct.float32)
        "}
        );
    }

    #[test]
    fn test_lowering_array_iota_like_unsupported_data_type() {
        let mut lowering = Lowering { code: String::new(), indentation: 0, next: 0, remaining: 10 };
        let operation = ArrayOperation::IotaLike(IotaLikeOperation::new(0, DataType::I8).unwrap());
        assert!(matches!(
            lowering.array(
                &operation,
                &["exemplar".to_owned()],
                &[ArrayType::new_static(DataType::F32, [4])],
                &ArrayType::new_static(DataType::I8, [4]),
            ),
            Err(Error::Unsupported { operation: "array", reason })
                if reason == "element dtype is outside the qualified cuTile subset"
        ));
    }

    #[test]
    fn test_lowering_array_broadcast_like() {
        // A new leading axis and a stretched unit axis both reshape the value to unit extents along the axes it does
        // not occupy. The exemplar's element type does not matter.
        let mut lowering = Lowering { code: String::new(), indentation: 0, next: 0, remaining: 10 };
        let output = ArrayType::new_static(DataType::F32, [2, 3]);
        let exemplar = ArrayType::new_static(DataType::F16, [2, 3]);
        for (value, output_axes) in [
            (ArrayType::new_static(DataType::F32, [3]), vec![1]),
            (ArrayType::new_static(DataType::F32, [1, 3]), vec![0, 1]),
        ] {
            let operation = ArrayOperation::BroadcastLike(BroadcastLikeOperation::new(output_axes));
            let inputs = ["value".to_owned(), "exemplar".to_owned()];
            lowering.array(&operation, &inputs, &[value, exemplar.clone()], &output).unwrap();
        }
        assert_eq!(
            lowering.code,
            indoc! {"
            v0 = ct.broadcast_to(ct.reshape(value, (1, 4)), (2, 4))
            v1 = ct.broadcast_to(ct.reshape(value, (1, 4)), (2, 4))
        "}
        );
    }

    #[test]
    fn test_lowering_array_broadcast_like_unordered_axes() {
        let mut lowering = Lowering { code: String::new(), indentation: 0, next: 0, remaining: 10 };
        let operation = ArrayOperation::BroadcastLike(BroadcastLikeOperation::new(vec![1, 0]));
        assert!(matches!(
            lowering.array(
                &operation,
                &["value".to_owned(), "exemplar".to_owned()],
                &[ArrayType::new_static(DataType::F32, [4, 2]), ArrayType::new_static(DataType::F32, [2, 4])],
                &ArrayType::new_static(DataType::F32, [2, 4]),
            ),
            Err(Error::Unsupported { operation: "broadcast_like", reason })
                if reason == "broadcast axes must preserve source order"
        ));
    }

    #[test]
    fn test_checked_tile() {
        assert!(checked_tile(&[0, 3, 5]).is_ok());
        assert_eq!(physical_shape(&[0, 3, 5]), "(1, 4, 8)");
        assert!(matches!(checked_tile(&[1024, 2048]), Err(Error::Unsupported { operation: "array", reason })
            if reason == "tile exceeds rank four or 1048576 physical elements"));
    }

    #[test]
    fn test_dimension() {
        let left = DimensionType::new("left", DimensionBounds::new(0, Some(8)).unwrap());
        let right = DimensionType::new("right", DimensionBounds::new(0, Some(8)).unwrap());
        let operation = DimensionOperation::Mul(DimensionMulOperation::new(&left, &right).unwrap());
        assert_eq!(dimension(&operation, &["a".to_owned(), "b".to_owned()]).unwrap(), "(a * b)");
    }
}
