//! Deterministic cuTile source over the canonical portable kernel arena.
//!
//! Global windows use bounds-checked gather/scatter, including completely empty edge windows. Tile values use
//! power-of-two physical shapes; reductions mask logical padding with the reduction identity before arithmetic.

use ryft_core::kernels::{GridExecution, KernelOperation, KernelSchedule, NoKernelExtension, VerifiedKernel};
use ryft_core::{
    Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, Atom, ComparisonDirection, DataType,
    Dimension, DimensionOperation, DimensionValue, DotDimensionNumbers, Operation, ProgramError, ReductionKind,
    RegionRef, Typed,
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
        let limits = starts.iter().zip(&shape).map(|(start, extent)| format!("({start} + {extent})")).collect();
        let axes = (0..shape.len()).collect();
        let fixed = vec![None; shape.len()];
        inputs.push(Value::Reference { parameter, starts, shape, limits, axes, fixed });
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
    fn mapping(&mut self, mapping: &ryft_core::kernels::BlockMapping, inputs: &[String]) -> Result<Vec<String>, Error> {
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
            let inputs = instruction.inputs().iter().map(|id| values[id.index()].clone().unwrap()).collect::<Vec<_>>();
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
                KernelOperation::Portable(ArrayIrOperation::Array(operation)) => {
                    let input_types = types
                        .iter()
                        .map(|r#type| match r#type {
                            ArrayIrType::Array(r#type) => Ok(r#type.clone()),
                            _ => Err(unsupported(operation.name(), "array operation received a non-array value")),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let ArrayIrType::Array(output) = &output_types[0] else { unreachable!() };
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
                KernelOperation::Portable(ArrayIrOperation::ReferenceIndex(operation)) => {
                    let mut reference = inputs[0].clone();
                    let ryft_core::ArrayReferenceView::Index {
                        axis,
                        index: ryft_core::ArrayReferenceViewIndex::Static(index),
                    } = operation.transform()
                    else {
                        unreachable!()
                    };
                    let Value::Reference { starts, shape, limits, axes, fixed, .. } = &mut reference else {
                        unreachable!()
                    };
                    fixed[axes.remove(axis)] = Some(format!("({} + {index})", starts.remove(axis)));
                    shape.remove(axis);
                    limits.remove(axis);
                    vec![reference]
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
                KernelOperation::TileLoad(operation) => {
                    let Value::Reference { parameter, starts, limits, axes, fixed, .. } = &inputs[0] else {
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
                    };
                    vec![Value::Expression(self.read(&reference, &expression(inputs.len() - 1)?)?)]
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
        let bits = u64::from_le_bytes(bits);
        let scalar = match r#type.data_type() {
            DataType::F16 | DataType::BF16 => {
                format!("ct.bitcast(ct.full((), {bits}, ct.uint16), ct.{})", dtype(r#type.data_type())?)
            }
            DataType::F32 => format!("ct.bitcast(ct.full((), {bits}, ct.uint32), ct.float32)"),
            DataType::F64 => format!("ct.bitcast(ct.full((), {bits}, ct.uint64), ct.float64)"),
            DataType::I32 => format!("ct.full((), {}, ct.int32)", bits as u32 as i32),
            DataType::I64 => format!("ct.full((), {}, ct.int64)", bits as i64),
            DataType::Boolean => format!("ct.full((), {}, ct.bool_)", if bits == 0 { "False" } else { "True" }),
            data_type => format!("ct.full((), {bits}, ct.{})", dtype(data_type)?),
        };
        Ok(self.assign(format!("ct.broadcast_to({scalar}, {})", physical_shape(&shape))))
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

    /// Loads a clipped global window with an explicit exact padding value.
    fn read(&mut self, reference: &Value, other: &str) -> Result<String, Error> {
        let Value::Reference { parameter, starts, shape, limits, axes, fixed } = reference else { unreachable!() };
        checked_tile(shape)?;
        if fixed.is_empty() {
            return Ok(self.assign(format!("ct.reshape(ct.load(p{parameter}, (0,), (1,)), ())")));
        }
        let (logical_indices, mask) = self.window_indices(starts, shape, limits);
        let mut indices = fixed.clone();
        for (&axis, index) in axes.iter().zip(logical_indices) {
            indices[axis] = Some(index);
        }
        let indices = tuple(&indices.into_iter().map(Option::unwrap).collect::<Vec<_>>());
        Ok(self.assign(format!(
            "ct.gather(p{parameter}, {indices}, mask={mask}, padding_value={other}, check_bounds=True)"
        )))
    }

    /// Publishes only valid logical/global coordinates.
    fn write(&mut self, reference: &Value, value: &str) -> Result<(), Error> {
        let Value::Reference { parameter, starts, shape, limits, axes, fixed } = reference else { unreachable!() };
        checked_tile(shape)?;
        if fixed.is_empty() {
            self.line(format!("ct.store(p{parameter}, (0,), ct.reshape({value}, (1,)))"));
            return Ok(());
        }
        let (logical_indices, mask) = self.window_indices(starts, shape, limits);
        let mut indices = fixed.clone();
        for (&axis, index) in axes.iter().zip(logical_indices) {
            indices[axis] = Some(index);
        }
        let indices = tuple(&indices.into_iter().map(Option::unwrap).collect::<Vec<_>>());
        self.line(format!("ct.scatter(p{parameter}, {indices}, {value}, mask={mask}, check_bounds=True)"));
        Ok(())
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
            ArrayOperation::Add(_) => format!("({} + {})", inputs[0], inputs[1]),
            ArrayOperation::Sub(_) => format!("({} - {})", inputs[0], inputs[1]),
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
            ArrayOperation::Broadcast(operation) => {
                if operation.output_axes().windows(2).any(|axes| axes[0] >= axes[1]) {
                    return Err(unsupported(operation.name(), "broadcast axes must preserve source order"));
                }
                let input_shape = checked_shape(&types[0])?;
                let mut expanded = vec![1; shape.len()];
                for (axis, &output_axis) in operation.output_axes().iter().enumerate() {
                    expanded[output_axis] = input_shape[axis].max(1).next_power_of_two();
                }
                format!("ct.broadcast_to(ct.reshape({}, {}), {})", inputs[0], tuple(&expanded), physical_shape(&shape))
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
    if !operation.effects().is_pure() {
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

/// Attaches a precise owning-operation diagnostic to every unsupported path.
fn unsupported(operation: &'static str, reason: &str) -> Error {
    Error::Unsupported { operation, reason: reason.to_owned() }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::kernels::{
        Grid, KernelCallOperation, KernelDefinition, KernelParameterAccess, whole_array_parameter,
    };
    use ryft_core::{DimensionBounds, DimensionMulOperation, DimensionType, ReferenceRead, ReferenceWrite};

    use super::*;

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
    fn test_lowering_charge() {
        let mut lowering = Lowering { code: String::new(), indentation: 0, next: 0, remaining: 1 };
        assert!(lowering.charge(1).is_ok());
        assert!(matches!(lowering.charge(1), Err(Error::Unsupported { operation: "kernel_call", reason })
            if reason == "source construction exceeds the instruction budget"));
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
        let reshape = ArrayOperation::Reshape(ryft_core::ReshapeOperation::new(output.shape().clone()));
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
