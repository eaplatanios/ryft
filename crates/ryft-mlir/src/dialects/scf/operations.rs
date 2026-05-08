use crate::{
    ArrayAttributeRef, Block, DenseInteger64ArrayAttributeRef, DetachedOp, DetachedRegion, DialectHandle, Error,
    Location, Operation, OperationBuilder, Region, RegionRef, TypeRef, Value, ValueRef, mlir_op, mlir_op_trait,
};

/// Operation trait for `scf.condition`.
pub trait ConditionOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the loop continuation condition.
    fn condition(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the values forwarded to the next region or to the parent `scf.while` results.
    fn arguments(&self) -> Result<impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>>, Error> {
        Ok(self.operand_values().skip(1))
    }
}

mlir_op!(Condition);
mlir_op_trait!(Condition, AlwaysSpeculatable);
mlir_op_trait!(Condition, IsTerminator);
mlir_op_trait!(Condition, NoMemoryEffect);
mlir_op_trait!(Condition, Pure);
mlir_op_trait!(Condition, SingleBlockRegions);
mlir_op_trait!(Condition, ZeroRegions);
mlir_op_trait!(Condition, ZeroSuccessors);

/// Constructs a new detached/owned [`ConditionOperation`] at the specified [`Location`].
pub fn condition<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    condition: ValueRef<'v, 'c, 't>,
    arguments: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedConditionOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::scf()?)?;
    OperationBuilder::new("scf.condition", location)
        .add_operand(condition)
        .add_operands(arguments)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `scf::condition`"))
        })
}

/// Name of the `scf.execute_region` attribute that requests preservation until explicit lowering.
pub const NO_INLINE_ATTRIBUTE: &str = "no_inline";

/// Operation trait for `scf.execute_region`.
pub trait ExecuteRegionOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the region that is executed exactly once.
    fn execution_region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }

    /// Returns whether inlining should be delayed until an explicit lowering step.
    fn no_inline(&self) -> bool {
        self.has_attribute(NO_INLINE_ATTRIBUTE)
    }
}

mlir_op!(ExecuteRegion);
mlir_op_trait!(ExecuteRegion, OneRegion);
mlir_op_trait!(ExecuteRegion, ZeroOperands);
mlir_op_trait!(ExecuteRegion, ZeroSuccessors);

/// Constructs a new detached/owned [`ExecuteRegionOperation`] at the specified [`Location`].
pub fn execute_region<'c, 't: 'c, L: Location<'c, 't>>(
    result_types: &[TypeRef<'c, 't>],
    no_inline: bool,
    region: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedExecuteRegionOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::scf()?)?;
    let mut builder = OperationBuilder::new("scf.execute_region", location).add_results(result_types);
    if no_inline {
        builder = builder.add_attribute(NO_INLINE_ATTRIBUTE, context.unit_attribute());
    }
    builder.add_region(region).build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `scf::execute_region`"))
    })
}

/// Name of the `scf.for` attribute that switches loop-bound comparisons to unsigned integer comparisons.
pub const UNSIGNED_CMP_ATTRIBUTE: &str = "unsignedCmp";

/// Operation trait for `scf.for`.
pub trait ForOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the lower bound.
    fn lower_bound(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the upper bound.
    fn upper_bound(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the positive step value.
    fn step(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the loop-carried initial values.
    fn initial_values(&self) -> Result<impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>>, Error> {
        Ok(self.operand_values().skip(3))
    }

    /// Returns the induction variable block argument.
    fn induction_variable(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let block = self
            .as_ref()
            .region(0)?
            .blocks()?
            .next()
            .ok_or_else(|| Error::invalid_argument("missing induction variable block in `scf.for`"))??;
        Ok(ValueRef::from(block.argument(0)?))
    }

    /// Returns the region arguments for loop-carried values.
    fn region_iter_args(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.as_ref()
            .region(0)?
            .blocks()?
            .next()
            .ok_or_else(|| Error::invalid_argument("missing body block in `scf.for`"))??
            .arguments()
            .skip(1)
            .map(|argument| argument.map(ValueRef::from))
            .collect()
    }

    /// Returns the loop body region.
    fn body_region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }

    /// Returns whether integer loop-bound comparisons are unsigned.
    fn unsigned_cmp(&self) -> bool {
        self.has_attribute(UNSIGNED_CMP_ATTRIBUTE)
    }
}

mlir_op!(For);
mlir_op_trait!(For, AutomaticAllocationScope);
mlir_op_trait!(For, OneRegion);
mlir_op_trait!(For, SingleBlock);
mlir_op_trait!(For, SingleBlockRegions);
mlir_op_trait!(For, ZeroSuccessors);

/// Constructs a new detached/owned [`ForOperation`] at the specified [`Location`].
pub fn r#for<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    lower_bound: ValueRef<'v, 'c, 't>,
    upper_bound: ValueRef<'v, 'c, 't>,
    step: ValueRef<'v, 'c, 't>,
    initial_values: &[ValueRef<'v, 'c, 't>],
    unsigned_cmp: bool,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedForOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::scf()?)?;
    let result_types = initial_values.iter().map(|value| value.r#type()).collect::<Result<Vec<_>, _>>()?;
    let mut builder = OperationBuilder::new("scf.for", location)
        .add_operand(lower_bound)
        .add_operand(upper_bound)
        .add_operand(step)
        .add_operands(initial_values)
        .add_results(&result_types);
    if unsigned_cmp {
        builder = builder.add_attribute(UNSIGNED_CMP_ATTRIBUTE, context.unit_attribute());
    }
    builder.add_region(body).build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `scf::for`"))
    })
}

/// Name of the attribute used by SCF operations with multiple variadic operand groups.
pub const OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operand_segment_sizes";

/// Name of the `scf.forall` static lower-bound attribute.
pub const STATIC_LOWER_BOUND_ATTRIBUTE: &str = "staticLowerBound";

/// Name of the `scf.forall` static upper-bound attribute.
pub const STATIC_UPPER_BOUND_ATTRIBUTE: &str = "staticUpperBound";

/// Name of the `scf.forall` static step attribute.
pub const STATIC_STEP_ATTRIBUTE: &str = "staticStep";

/// Name of the `scf.forall` optional device mapping attribute.
pub const MAPPING_ATTRIBUTE: &str = "mapping";

/// Operation trait for `scf.forall`.
pub trait ForAllOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the dynamic lower-bound operands.
    fn dynamic_lower_bounds(&self) -> Result<impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>>, Error> {
        let lower_bound_count =
            self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        Ok(self.operand_values().take(lower_bound_count))
    }

    /// Returns the dynamic upper-bound operands.
    fn dynamic_upper_bounds(&self) -> Result<impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>>, Error> {
        let lower_bound_count =
            self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        let upper_bound_count =
            self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?;
        Ok(self.operand_values().skip(lower_bound_count).take(upper_bound_count))
    }

    /// Returns the dynamic step operands.
    fn dynamic_steps(&self) -> Result<impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>>, Error> {
        let lower_bound_count =
            self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        let upper_bound_count =
            self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?;
        let step_count = self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 2)?;
        Ok(self.operand_values().skip(lower_bound_count + upper_bound_count).take(step_count))
    }

    /// Returns the shared output operands.
    fn outputs(&self) -> Result<impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>>, Error> {
        let lower_bound_count =
            self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        let upper_bound_count =
            self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?;
        let step_count = self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 2)?;
        let output_count = self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 3)?;
        Ok(self.operand_values().skip(lower_bound_count + upper_bound_count + step_count).take(output_count))
    }

    /// Returns the static lower bounds.
    fn static_lower_bounds(&self) -> Result<DenseInteger64ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_64_array_attribute(STATIC_LOWER_BOUND_ATTRIBUTE)
    }

    /// Returns the static upper bounds.
    fn static_upper_bounds(&self) -> Result<DenseInteger64ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_64_array_attribute(STATIC_UPPER_BOUND_ATTRIBUTE)
    }

    /// Returns the static steps.
    fn static_steps(&self) -> Result<DenseInteger64ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_64_array_attribute(STATIC_STEP_ATTRIBUTE)
    }

    /// Returns the optional device mapping attributes.
    fn mapping(&self) -> Result<Option<ArrayAttributeRef<'c, 't>>, Error> {
        if self.has_attribute(MAPPING_ATTRIBUTE) { self.array_attribute(MAPPING_ATTRIBUTE).map(Some) } else { Ok(None) }
    }

    /// Returns the parallel body region.
    fn body_region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }
}

mlir_op!(ForAll);
mlir_op_trait!(ForAll, AutomaticAllocationScope);
mlir_op_trait!(ForAll, OneRegion);
mlir_op_trait!(ForAll, SingleBlock);
mlir_op_trait!(ForAll, SingleBlockRegions);
mlir_op_trait!(ForAll, ZeroSuccessors);

/// Constructs a new detached/owned [`ForAllOperation`] at the specified [`Location`].
pub fn for_all<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    dynamic_lower_bounds: &[ValueRef<'v, 'c, 't>],
    dynamic_upper_bounds: &[ValueRef<'v, 'c, 't>],
    dynamic_steps: &[ValueRef<'v, 'c, 't>],
    static_lower_bounds: &[i64],
    static_upper_bounds: &[i64],
    static_steps: &[i64],
    outputs: &[ValueRef<'v, 'c, 't>],
    mapping: Option<ArrayAttributeRef<'c, 't>>,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedForAllOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::scf()?)?;
    let segment_sizes = [
        dynamic_lower_bounds.len() as i32,
        dynamic_upper_bounds.len() as i32,
        dynamic_steps.len() as i32,
        outputs.len() as i32,
    ];
    let result_types = outputs.iter().map(|value| value.r#type()).collect::<Result<Vec<_>, _>>()?;
    let mut builder = OperationBuilder::new("scf.forall", location)
        .add_operands(dynamic_lower_bounds)
        .add_operands(dynamic_upper_bounds)
        .add_operands(dynamic_steps)
        .add_operands(outputs)
        .add_results(&result_types)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes)?)
        .add_attribute(STATIC_LOWER_BOUND_ATTRIBUTE, context.dense_i64_array_attribute(static_lower_bounds)?)
        .add_attribute(STATIC_UPPER_BOUND_ATTRIBUTE, context.dense_i64_array_attribute(static_upper_bounds)?)
        .add_attribute(STATIC_STEP_ATTRIBUTE, context.dense_i64_array_attribute(static_steps)?);
    if let Some(mapping) = mapping {
        builder = builder.add_attribute(MAPPING_ATTRIBUTE, mapping);
    }
    builder.add_region(body).build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `scf::for_all`"))
    })
}

/// Operation trait for `scf.forall.in_parallel`.
pub trait InParallelOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the region containing aggregate-combining operations.
    fn body_region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }
}

mlir_op!(InParallel);
mlir_op_trait!(InParallel, AlwaysSpeculatable);
mlir_op_trait!(InParallel, IsTerminator);
mlir_op_trait!(InParallel, NoMemoryEffect);
mlir_op_trait!(InParallel, NoTerminator);
mlir_op_trait!(InParallel, OneRegion);
mlir_op_trait!(InParallel, Pure);
mlir_op_trait!(InParallel, ReturnLike);
mlir_op_trait!(InParallel, SingleBlock);
mlir_op_trait!(InParallel, SingleBlockRegions);
mlir_op_trait!(InParallel, ZeroOperands);
mlir_op_trait!(InParallel, ZeroSuccessors);

/// Constructs a new detached/owned [`InParallelOperation`] at the specified [`Location`].
pub fn in_parallel<'c, 't: 'c, L: Location<'c, 't>>(
    region: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedInParallelOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::scf()?)?;
    OperationBuilder::new("scf.forall.in_parallel", location)
        .add_region(region)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `scf::in_parallel`"))
        })
}

/// Operation trait for `scf.if`.
pub trait IfOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the condition.
    fn condition(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the region executed when the condition is true.
    fn then_region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }

    /// Returns the region executed when the condition is false.
    fn else_region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(1)
    }
}

mlir_op!(If);
mlir_op_trait!(If, NoRegionArguments);
mlir_op_trait!(If, SingleBlockRegions);
mlir_op_trait!(If, ZeroSuccessors);

/// Constructs a new detached/owned [`IfOperation`] at the specified [`Location`].
pub fn r#if<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    condition: ValueRef<'v, 'c, 't>,
    result_types: &[TypeRef<'c, 't>],
    then_region: DetachedRegion<'c, 't>,
    else_region: Option<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedIfOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::scf()?)?;
    let else_region = else_region.unwrap_or_else(|| context.region());
    OperationBuilder::new("scf.if", location)
        .add_operand(condition)
        .add_results(result_types)
        .add_region(then_region)
        .add_region(else_region)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `scf::if`"))
        })
}

/// Operation trait for `scf.parallel`.
pub trait ParallelOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the lower-bound operands.
    fn lower_bounds(&self) -> Result<impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>>, Error> {
        let lower_bound_count =
            self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        Ok(self.operand_values().take(lower_bound_count))
    }

    /// Returns the upper-bound operands.
    fn upper_bounds(&self) -> Result<impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>>, Error> {
        let lower_bound_count =
            self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        let upper_bound_count =
            self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?;
        Ok(self.operand_values().skip(lower_bound_count).take(upper_bound_count))
    }

    /// Returns the step operands.
    fn steps(&self) -> Result<impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>>, Error> {
        let lower_bound_count =
            self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        let upper_bound_count =
            self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?;
        let step_count = self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 2)?;
        Ok(self.operand_values().skip(lower_bound_count + upper_bound_count).take(step_count))
    }

    /// Returns the initial values for reductions.
    fn initial_values(&self) -> Result<impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>>, Error> {
        let lower_bound_count =
            self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        let upper_bound_count =
            self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?;
        let step_count = self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 2)?;
        let initial_value_count =
            self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 3)?;
        Ok(self
            .operand_values()
            .skip(lower_bound_count + upper_bound_count + step_count)
            .take(initial_value_count))
    }

    /// Returns the parallel body region.
    fn body_region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }
}

mlir_op!(Parallel);
mlir_op_trait!(Parallel, AutomaticAllocationScope);
mlir_op_trait!(Parallel, OneRegion);
mlir_op_trait!(Parallel, SingleBlock);
mlir_op_trait!(Parallel, SingleBlockRegions);
mlir_op_trait!(Parallel, ZeroSuccessors);

/// Constructs a new detached/owned [`ParallelOperation`] at the specified [`Location`].
pub fn parallel<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    lower_bounds: &[ValueRef<'v, 'c, 't>],
    upper_bounds: &[ValueRef<'v, 'c, 't>],
    steps: &[ValueRef<'v, 'c, 't>],
    initial_values: &[ValueRef<'v, 'c, 't>],
    body: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedParallelOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::scf()?)?;
    let segment_sizes =
        [lower_bounds.len() as i32, upper_bounds.len() as i32, steps.len() as i32, initial_values.len() as i32];
    let result_types = initial_values.iter().map(|value| value.r#type()).collect::<Result<Vec<_>, _>>()?;
    OperationBuilder::new("scf.parallel", location)
        .add_operands(lower_bounds)
        .add_operands(upper_bounds)
        .add_operands(steps)
        .add_operands(initial_values)
        .add_results(&result_types)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes)?)
        .add_region(body)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `scf::parallel`"))
        })
}

/// Operation trait for `scf.reduce`.
pub trait ReduceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the values reduced by this terminator.
    fn values(&self) -> Result<impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>>, Error> {
        Ok(self.operand_values())
    }

    /// Returns the reduction regions.
    fn reductions(&self) -> Result<impl Iterator<Item = Result<RegionRef<'o, 'c, 't>, Error>>, Error> {
        Ok(self.regions())
    }
}

mlir_op!(Reduce);
mlir_op_trait!(Reduce, IsTerminator);
mlir_op_trait!(Reduce, SingleBlockRegions);
mlir_op_trait!(Reduce, ZeroSuccessors);

/// Constructs a new detached/owned [`ReduceOperation`] at the specified [`Location`].
pub fn reduce<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    values: &[ValueRef<'v, 'c, 't>],
    reductions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedReduceOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::scf()?)?;
    OperationBuilder::new("scf.reduce", location)
        .add_operands(values)
        .add_regions(reductions)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `scf::reduce`"))
        })
}

/// Operation trait for `scf.reduce.return`.
pub trait ReduceReturnOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the reduction value.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(ReduceReturn);
mlir_op_trait!(ReduceReturn, AlwaysSpeculatable);
mlir_op_trait!(ReduceReturn, IsTerminator);
mlir_op_trait!(ReduceReturn, NoMemoryEffect);
mlir_op_trait!(ReduceReturn, Pure);
mlir_op_trait!(ReduceReturn, SingleBlockRegions);
mlir_op_trait!(ReduceReturn, ZeroRegions);
mlir_op_trait!(ReduceReturn, ZeroSuccessors);

/// Constructs a new detached/owned [`ReduceReturnOperation`] at the specified [`Location`].
pub fn reduce_return<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    value: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedReduceReturnOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::scf()?)?;
    OperationBuilder::new("scf.reduce.return", location)
        .add_operand(value)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `scf::reduce_return`"))
        })
}

/// Operation trait for `scf.while`.
pub trait WhileOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the initial values passed to the before region.
    fn initial_values(&self) -> Result<impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>>, Error> {
        Ok(self.operand_values())
    }

    /// Returns the region that runs before the continuation condition.
    fn before_region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }

    /// Returns the region that runs after the continuation condition.
    fn after_region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(1)
    }
}

mlir_op!(While);
mlir_op_trait!(While, SingleBlockRegions);
mlir_op_trait!(While, ZeroSuccessors);

/// Constructs a new detached/owned [`WhileOperation`] at the specified [`Location`].
pub fn r#while<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    initial_values: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    before_region: DetachedRegion<'c, 't>,
    after_region: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedWhileOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::scf()?)?;
    OperationBuilder::new("scf.while", location)
        .add_operands(initial_values)
        .add_results(result_types)
        .add_region(before_region)
        .add_region(after_region)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `scf::while`"))
        })
}

/// Name of the `scf.index_switch` dense integer array attribute that stores case values.
pub const CASES_ATTRIBUTE: &str = "cases";

/// Operation trait for `scf.index_switch`.
pub trait IndexSwitchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the index value being matched.
    fn argument(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the case values in region order.
    fn cases(&self) -> Result<DenseInteger64ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_64_array_attribute(CASES_ATTRIBUTE)
    }

    /// Returns the default region.
    fn default_region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }

    /// Returns the case regions.
    fn case_regions(&self) -> Result<impl Iterator<Item = Result<RegionRef<'o, 'c, 't>, Error>>, Error> {
        Ok(self.regions().skip(1))
    }
}

mlir_op!(IndexSwitch);
mlir_op_trait!(IndexSwitch, SingleBlockRegions);
mlir_op_trait!(IndexSwitch, ZeroSuccessors);

/// Constructs a new detached/owned [`IndexSwitchOperation`] at the specified [`Location`].
pub fn index_switch<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    argument: ValueRef<'v, 'c, 't>,
    cases: &[i64],
    result_types: &[TypeRef<'c, 't>],
    default_region: DetachedRegion<'c, 't>,
    case_regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedIndexSwitchOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::scf()?)?;
    let mut regions = Vec::with_capacity(case_regions.len() + 1);
    regions.push(default_region);
    regions.extend(case_regions);
    OperationBuilder::new("scf.index_switch", location)
        .add_operand(argument)
        .add_results(result_types)
        .add_attribute(CASES_ATTRIBUTE, context.dense_i64_array_attribute(cases)?)
        .add_regions(regions)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `scf::index_switch`"))
        })
}

/// Operation trait for `scf.yield`.
pub trait YieldOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the values yielded to the parent operation.
    fn values(&self) -> Result<impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>>, Error> {
        Ok(self.operand_values())
    }
}

mlir_op!(Yield);
mlir_op_trait!(Yield, AlwaysSpeculatable);
mlir_op_trait!(Yield, IsTerminator);
mlir_op_trait!(Yield, NoMemoryEffect);
mlir_op_trait!(Yield, Pure);
mlir_op_trait!(Yield, ReturnLike);
mlir_op_trait!(Yield, SingleBlockRegions);
mlir_op_trait!(Yield, ZeroRegions);
mlir_op_trait!(Yield, ZeroSuccessors);

/// Constructs a new detached/owned [`YieldOperation`] at the specified [`Location`].
pub fn r#yield<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    values: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedYieldOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::scf()?)?;
    OperationBuilder::new("scf.yield", location)
        .add_operands(values)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `scf::yield`"))
        })
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Block, Context, Operation, Region, Type, Value};

    use super::*;

    #[test]
    fn test_condition_and_while() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let i1_type = context.signless_integer_type(1);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type.as_ref(), location), (i1_type.as_ref(), location)]);
                let initial_value = block.argument(0).unwrap().as_ref();
                let keep_going = block.argument(1).unwrap().as_ref();

                let mut before_region = context.region();
                let mut before_block = context.block(&[(index_type.as_ref(), location), (i1_type.as_ref(), location)]);
                let before_value = before_block.argument(0).unwrap().as_ref();
                let before_condition = before_block.argument(1).unwrap().as_ref();
                let condition_op = condition(before_condition, &[before_value], location).unwrap();
                assert_eq!(condition_op.condition().unwrap(), before_condition);
                assert_eq!(
                    condition_op.arguments().unwrap().collect::<Result<Vec<_>, _>>().unwrap(),
                    vec![before_value],
                );
                before_block.append_operation(condition_op).unwrap();
                before_region.append_block(before_block).unwrap();

                let mut after_region = context.region();
                let mut after_block = context.block(&[(index_type, location)]);
                let after_value = after_block.argument(0).unwrap().as_ref();
                after_block.append_operation(r#yield(&[after_value, keep_going], location).unwrap()).unwrap();
                after_region.append_block(after_block).unwrap();

                let while_op = r#while(
                    &[initial_value, keep_going],
                    &[index_type.as_ref()],
                    before_region,
                    after_region,
                    location,
                )
                .unwrap();
                assert_eq!(
                    while_op.initial_values().unwrap().collect::<Result<Vec<_>, _>>().unwrap(),
                    vec![initial_value, keep_going],
                );
                assert_eq!(while_op.before_region().unwrap().blocks().unwrap().count(), 1);
                assert_eq!(while_op.after_region().unwrap().blocks().unwrap().count(), 1);
                assert_eq!(while_op.result_count(), 1);
                let while_op = block.append_operation(while_op).unwrap();
                block
                    .append_operation(func::r#return(&[while_op.result(0).unwrap().as_ref()], location).unwrap())
                    .unwrap();
                func::func(
                    "while_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), i1_type.into()],
                        results: vec![index_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @while_test(%arg0: index, %arg1: i1) -> index {
                    %0 = scf.while (%arg2 = %arg0, %arg3 = %arg1) : (index, i1) -> index {
                      scf.condition(%arg3) %arg2 : index
                    } do {
                    ^bb0(%arg2: index):
                      scf.yield %arg2, %arg1 : index, i1
                    }
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_execute_region() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location)]);
                let argument = block.argument(0).unwrap().as_ref();

                let mut region = context.region();
                let mut region_block = context.block_with_no_arguments();
                let yield_op = r#yield(&[argument], location).unwrap();
                assert_eq!(yield_op.values().unwrap().collect::<Result<Vec<_>, _>>().unwrap(), vec![argument]);
                region_block.append_operation(yield_op).unwrap();
                region.append_block(region_block).unwrap();

                let execute_region_op = execute_region(&[index_type.as_ref()], true, region, location).unwrap();
                assert!(execute_region_op.no_inline());
                assert_eq!(execute_region_op.execution_region().unwrap().blocks().unwrap().count(), 1);
                assert_eq!(execute_region_op.result_count(), 1);
                let execute_region_op = block.append_operation(execute_region_op).unwrap();
                block
                    .append_operation(
                        func::r#return(&[execute_region_op.result(0).unwrap().as_ref()], location).unwrap(),
                    )
                    .unwrap();
                func::func(
                    "execute_region_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into()],
                        results: vec![index_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @execute_region_test(%arg0: index) -> index {
                    %0 = scf.execute_region -> index no_inline {
                      scf.yield %arg0 : index
                    }
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_for() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (index_type.as_ref(), location),
                    (index_type.as_ref(), location),
                    (index_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let lower_bound = block.argument(0).unwrap().as_ref();
                let upper_bound = block.argument(1).unwrap().as_ref();
                let step = block.argument(2).unwrap().as_ref();
                let initial_value = block.argument(3).unwrap().as_ref();

                let mut body = context.region();
                let mut body_block = context.block(&[(index_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let carried_value = body_block.argument(1).unwrap().as_ref();
                body_block.append_operation(r#yield(&[carried_value], location).unwrap()).unwrap();
                body.append_block(body_block).unwrap();

                let for_op = r#for(lower_bound, upper_bound, step, &[initial_value], false, body, location).unwrap();
                assert_eq!(for_op.lower_bound().unwrap(), lower_bound);
                assert_eq!(for_op.upper_bound().unwrap(), upper_bound);
                assert_eq!(for_op.step().unwrap(), step);
                assert_eq!(
                    for_op.initial_values().unwrap().collect::<Result<Vec<_>, _>>().unwrap(),
                    vec![initial_value],
                );
                assert_eq!(for_op.induction_variable().unwrap().r#type().unwrap(), index_type);
                assert_eq!(for_op.region_iter_args().unwrap()[0].r#type().unwrap(), i32_type);
                assert!(!for_op.unsigned_cmp());
                let for_op = block.append_operation(for_op).unwrap();
                block
                    .append_operation(func::r#return(&[for_op.result(0).unwrap().as_ref()], location).unwrap())
                    .unwrap();
                func::func(
                    "for_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into(), index_type.into(), i32_type.into()],
                        results: vec![i32_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @for_test(%arg0: index, %arg1: index, %arg2: index, %arg3: i32) -> i32 {
                    %0 = scf.for %arg4 = %arg0 to %arg1 step %arg2 iter_args(%arg5 = %arg3) -> (i32) {
                      scf.yield %arg5 : i32
                    }
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_for_all_and_in_parallel() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();

                let mut in_parallel_region = context.region();
                in_parallel_region.append_block(context.block_with_no_arguments()).unwrap();
                let in_parallel_op = in_parallel(in_parallel_region, location).unwrap();
                assert_eq!(in_parallel_op.body_region().unwrap().blocks().unwrap().count(), 1);

                let mut body = context.region();
                let mut body_block = context.block(&[(index_type, location)]);
                body_block.append_operation(in_parallel_op).unwrap();
                body.append_block(body_block).unwrap();

                let for_all_op = for_all(&[], &[], &[], &[0], &[4], &[1], &[], None, body, location).unwrap();
                assert_eq!(for_all_op.dynamic_lower_bounds().unwrap().count(), 0);
                assert_eq!(for_all_op.dynamic_upper_bounds().unwrap().count(), 0);
                assert_eq!(for_all_op.dynamic_steps().unwrap().count(), 0);
                assert_eq!(for_all_op.outputs().unwrap().count(), 0);
                assert_eq!(for_all_op.static_lower_bounds().unwrap().values().collect::<Vec<_>>(), vec![0]);
                assert_eq!(for_all_op.static_upper_bounds().unwrap().values().collect::<Vec<_>>(), vec![4]);
                assert_eq!(for_all_op.static_steps().unwrap().values().collect::<Vec<_>>(), vec![1]);
                assert!(for_all_op.mapping().unwrap().is_none());
                block.append_operation(for_all_op).unwrap();
                let return_values = Vec::<ValueRef<'_, '_, '_>>::new();
                block.append_operation(func::r#return(&return_values, location).unwrap()).unwrap();
                func::func("for_all_test", func::FuncAttributes::default(), block.try_into().unwrap(), location)
                    .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @for_all_test() {
                    scf.forall (%arg0) in (4) {
                    }
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_if() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (i1_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let condition_value = block.argument(0).unwrap().as_ref();
                let true_value = block.argument(1).unwrap().as_ref();
                let false_value = block.argument(2).unwrap().as_ref();

                let mut then_region = context.region();
                let mut then_block = context.block_with_no_arguments();
                then_block.append_operation(r#yield(&[true_value], location).unwrap()).unwrap();
                then_region.append_block(then_block).unwrap();

                let mut else_region = context.region();
                let mut else_block = context.block_with_no_arguments();
                else_block.append_operation(r#yield(&[false_value], location).unwrap()).unwrap();
                else_region.append_block(else_block).unwrap();

                let if_op =
                    r#if(condition_value, &[i32_type.as_ref()], then_region, Some(else_region), location).unwrap();
                assert_eq!(if_op.condition().unwrap(), condition_value);
                assert_eq!(if_op.then_region().unwrap().blocks().unwrap().count(), 1);
                assert_eq!(if_op.else_region().unwrap().blocks().unwrap().count(), 1);
                let if_op = block.append_operation(if_op).unwrap();
                block
                    .append_operation(func::r#return(&[if_op.result(0).unwrap().as_ref()], location).unwrap())
                    .unwrap();
                func::func(
                    "if_test",
                    func::FuncAttributes {
                        arguments: vec![i1_type.into(), i32_type.into(), i32_type.into()],
                        results: vec![i32_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @if_test(%arg0: i1, %arg1: i32, %arg2: i32) -> i32 {
                    %0 = scf.if %arg0 -> (i32) {
                      scf.yield %arg1 : i32
                    } else {
                      scf.yield %arg2 : i32
                    }
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_parallel_reduce_and_reduce_return() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (index_type.as_ref(), location),
                    (index_type.as_ref(), location),
                    (index_type.as_ref(), location),
                    (f32_type.as_ref(), location),
                ]);
                let lower_bound = block.argument(0).unwrap().as_ref();
                let upper_bound = block.argument(1).unwrap().as_ref();
                let step = block.argument(2).unwrap().as_ref();
                let initial_value = block.argument(3).unwrap().as_ref();

                let mut reduction_region = context.region();
                let mut reduction_block = context.block(&[(f32_type, location), (f32_type, location)]);
                let lhs = reduction_block.argument(0).unwrap().as_ref();
                let reduce_return_op = reduce_return(lhs, location).unwrap();
                assert_eq!(reduce_return_op.value().unwrap(), lhs);
                reduction_block.append_operation(reduce_return_op).unwrap();
                reduction_region.append_block(reduction_block).unwrap();

                let mut body = context.region();
                let mut body_block = context.block(&[(index_type, location)]);
                let reduce_op = reduce(&[initial_value], vec![reduction_region], location).unwrap();
                assert_eq!(reduce_op.values().unwrap().collect::<Result<Vec<_>, _>>().unwrap(), vec![initial_value]);
                assert_eq!(reduce_op.reductions().unwrap().into_iter().count(), 1);
                body_block.append_operation(reduce_op).unwrap();
                body.append_block(body_block).unwrap();

                let parallel_op =
                    parallel(&[lower_bound], &[upper_bound], &[step], &[initial_value], body, location).unwrap();
                assert_eq!(
                    parallel_op.lower_bounds().unwrap().collect::<Result<Vec<_>, _>>().unwrap(),
                    vec![lower_bound]
                );
                assert_eq!(
                    parallel_op.upper_bounds().unwrap().collect::<Result<Vec<_>, _>>().unwrap(),
                    vec![upper_bound]
                );
                assert_eq!(parallel_op.steps().unwrap().collect::<Result<Vec<_>, _>>().unwrap(), vec![step]);
                assert_eq!(
                    parallel_op.initial_values().unwrap().collect::<Result<Vec<_>, _>>().unwrap(),
                    vec![initial_value],
                );
                let parallel_op = block.append_operation(parallel_op).unwrap();
                block
                    .append_operation(func::r#return(&[parallel_op.result(0).unwrap().as_ref()], location).unwrap())
                    .unwrap();
                func::func(
                    "parallel_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into(), index_type.into(), f32_type.into()],
                        results: vec![f32_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @parallel_test(%arg0: index, %arg1: index, %arg2: index, %arg3: f32) -> f32 {
                    %0 = scf.parallel (%arg4) = (%arg0) to (%arg1) step (%arg2) init (%arg3) -> f32 {
                      scf.reduce(%arg3 : f32) {
                      ^bb0(%arg5: f32, %arg6: f32):
                        scf.reduce.return %arg5 : f32
                      }
                    }
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_index_switch() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (index_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let argument = block.argument(0).unwrap().as_ref();
                let default_value = block.argument(1).unwrap().as_ref();
                let case_0_value = block.argument(2).unwrap().as_ref();
                let case_1_value = block.argument(3).unwrap().as_ref();

                let mut default_region = context.region();
                let mut default_block = context.block_with_no_arguments();
                default_block.append_operation(r#yield(&[default_value], location).unwrap()).unwrap();
                default_region.append_block(default_block).unwrap();

                let mut case_0_region = context.region();
                let mut case_0_block = context.block_with_no_arguments();
                case_0_block.append_operation(r#yield(&[case_0_value], location).unwrap()).unwrap();
                case_0_region.append_block(case_0_block).unwrap();

                let mut case_1_region = context.region();
                let mut case_1_block = context.block_with_no_arguments();
                case_1_block.append_operation(r#yield(&[case_1_value], location).unwrap()).unwrap();
                case_1_region.append_block(case_1_block).unwrap();

                let switch_op = index_switch(
                    argument,
                    &[0, 1],
                    &[i32_type.as_ref()],
                    default_region,
                    vec![case_0_region, case_1_region],
                    location,
                )
                .unwrap();
                assert_eq!(switch_op.argument().unwrap(), argument);
                assert_eq!(switch_op.cases().unwrap().values().collect::<Vec<_>>(), vec![0, 1]);
                assert_eq!(switch_op.default_region().unwrap().blocks().unwrap().count(), 1);
                assert_eq!(switch_op.case_regions().unwrap().into_iter().count(), 2);
                let switch_op = block.append_operation(switch_op).unwrap();
                block
                    .append_operation(func::r#return(&[switch_op.result(0).unwrap().as_ref()], location).unwrap())
                    .unwrap();
                func::func(
                    "index_switch_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), i32_type.into(), i32_type.into(), i32_type.into()],
                        results: vec![i32_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            // The MLIR printer currently emits a trailing space after the `scf.index_switch` result type.
            concat!(
                "module {\n",
                "  func.func @index_switch_test(%arg0: index, %arg1: i32, %arg2: i32, %arg3: i32) -> i32 {\n",
                "    %0 = scf.index_switch %arg0 -> i32 \n",
                "    case 0 {\n",
                "      scf.yield %arg2 : i32\n",
                "    }\n",
                "    case 1 {\n",
                "      scf.yield %arg3 : i32\n",
                "    }\n",
                "    default {\n",
                "      scf.yield %arg1 : i32\n",
                "    }\n",
                "    return %0 : i32\n",
                "  }\n",
                "}\n",
            ),
        );
    }
}
