use crate::dialects::arith::{AtomicRmwKind, AtomicRmwKindAttributeRef};
use crate::{
    AffineMap, Attribute, DetachedOp, DetachedRegion, DialectHandle, Error, IntegerAttributeRef, IntegerSet, Location,
    Operation, OperationBuilder, RegionRef, TypeRef, ValueRef, VectorTypeDimension, mlir_op, mlir_op_trait,
};

/// Name of the affine map attribute used by single-map affine operations.
pub const MAP_ATTRIBUTE: &str = "map";

/// Trait representing the `affine.apply` operation.
pub trait ApplyOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the affine map applied by this operation.
    fn map(&self) -> Result<AffineMap<'c, 't>, Error> {
        self.affine_map_attribute(MAP_ATTRIBUTE)?.affine_map()
    }

    /// Returns the operands supplied to the affine map.
    fn map_operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().collect()
    }

    /// Returns the dimension operands supplied to the affine map.
    fn dimension_operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        (0..self.map()?.dimension_count()).map(|index| self.operand_value(index)).collect()
    }

    /// Returns the symbol operands supplied to the affine map.
    fn symbol_operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let dimension_count = self.map()?.dimension_count();
        (dimension_count..self.operand_count()).map(|index| self.operand_value(index)).collect()
    }
}

mlir_op!(Apply);
mlir_op_trait!(Apply, OneResult);
mlir_op_trait!(Apply, AlwaysSpeculatable);
mlir_op_trait!(Apply, ZeroRegions);
mlir_op_trait!(Apply, ZeroSuccessors);
mlir_op_trait!(Apply, Pure);
mlir_op_trait!(Apply, NoMemoryEffect);

/// Constructs a new detached [`ApplyOperation`].
pub fn apply<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    map: AffineMap<'c, 't>,
    map_operands: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedApplyOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::affine()?)?;
    OperationBuilder::new("affine.apply", location)
        .add_attribute(MAP_ATTRIBUTE, context.affine_map_attribute(map))
        .add_operands(map_operands)
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `affine::apply`"))
        })
}

/// Name of the lower-bound affine map attribute used by [`ForOperation`].
pub const LOWER_BOUND_MAP_ATTRIBUTE: &str = "lowerBoundMap";

/// Name of the upper-bound affine map attribute used by [`ForOperation`].
pub const UPPER_BOUND_MAP_ATTRIBUTE: &str = "upperBoundMap";

/// Name of the loop step attribute used by [`ForOperation`].
pub const STEP_ATTRIBUTE: &str = "step";

/// Name of the operand segment-size attribute used by affine operations with multiple variadic operand groups.
pub const OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operandSegmentSizes";

/// Trait representing the `affine.for` operation.
pub trait ForOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the affine map used to compute this loop's lower bound.
    fn lower_bound_map(&self) -> Result<AffineMap<'c, 't>, Error> {
        self.affine_map_attribute(LOWER_BOUND_MAP_ATTRIBUTE)?.affine_map()
    }

    /// Returns the operands supplied to this loop's lower-bound affine map.
    fn lower_bound_operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns the affine map used to compute this loop's upper bound.
    fn upper_bound_map(&self) -> Result<AffineMap<'c, 't>, Error> {
        self.affine_map_attribute(UPPER_BOUND_MAP_ATTRIBUTE)?.affine_map()
    }

    /// Returns the operands supplied to this loop's upper-bound affine map.
    fn upper_bound_operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns the initial values passed to this loop.
    fn inits(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 2)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns the positive integer step used by this loop.
    fn step(&self) -> Result<i64, Error> {
        Ok(self.integer_attribute(STEP_ATTRIBUTE)?.signless_value())
    }

    /// Returns this loop's body region.
    fn body(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }
}

mlir_op!(For);
mlir_op_trait!(For, AutomaticAllocationScope);
mlir_op_trait!(For, SingleBlockRegions);
mlir_op_trait!(For, ZeroSuccessors);

/// Constructs a new detached [`ForOperation`].
pub fn r#for<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    lower_bound_map: AffineMap<'c, 't>,
    lower_bound_operands: &[ValueRef<'v, 'c, 't>],
    upper_bound_map: AffineMap<'c, 't>,
    upper_bound_operands: &[ValueRef<'v, 'c, 't>],
    step: i64,
    inits: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    body: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedForOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::affine()?)?;
    let operand_segment_sizes =
        [lower_bound_operands.len() as i32, upper_bound_operands.len() as i32, inits.len() as i32];
    let builder = OperationBuilder::new("affine.for", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&operand_segment_sizes)?)
        .add_attribute(LOWER_BOUND_MAP_ATTRIBUTE, context.affine_map_attribute(lower_bound_map))
        .add_attribute(UPPER_BOUND_MAP_ATTRIBUTE, context.affine_map_attribute(upper_bound_map));
    builder
        .add_attribute(STEP_ATTRIBUTE, context.integer_attribute(context.index_type(), step))
        .add_operands(lower_bound_operands)
        .add_operands(upper_bound_operands)
        .add_operands(inits)
        .add_results(result_types)
        .add_region(body)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `affine::for`"))
        })
}

/// Name of the integer-set condition attribute used by [`IfOperation`].
pub const CONDITION_ATTRIBUTE: &str = "condition";

/// Trait representing the `affine.if` operation.
pub trait IfOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the integer-set condition evaluated by this operation.
    fn condition(&self) -> Result<IntegerSet<'c, 't>, Error> {
        Ok(self.integer_set_attribute(CONDITION_ATTRIBUTE)?.integer_set())
    }

    /// Returns the operands supplied to this operation's condition.
    fn condition_operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().collect()
    }

    /// Returns this operation's then-region.
    fn then_region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }

    /// Returns this operation's else-region.
    fn else_region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(1)
    }
}

mlir_op!(If);
mlir_op_trait!(If, ZeroSuccessors);

/// Constructs a new detached [`IfOperation`].
pub fn r#if<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    condition: IntegerSet<'c, 't>,
    condition_operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    then_region: DetachedRegion<'c, 't>,
    else_region: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedIfOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::affine()?)?;
    OperationBuilder::new("affine.if", location)
        .add_attribute(CONDITION_ATTRIBUTE, context.integer_set_attribute(condition))
        .add_operands(condition_operands)
        .add_results(result_types)
        .add_region(then_region)
        .add_region(else_region)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `affine::if`"))
        })
}

/// Trait representing the `affine.load` operation.
pub trait LoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the memref being loaded from.
    fn memref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the affine map used to index the memref.
    fn map(&self) -> Result<AffineMap<'c, 't>, Error> {
        self.affine_map_attribute(MAP_ATTRIBUTE)?.affine_map()
    }

    /// Returns the affine index operands used to access the memref.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        (1..self.operand_count()).map(|index| self.operand_value(index)).collect()
    }
}

mlir_op!(Load);
mlir_op_trait!(Load, OneResult);
mlir_op_trait!(Load, ZeroRegions);
mlir_op_trait!(Load, ZeroSuccessors);
mlir_op_trait!(Load, MemRefsNormalizable);

/// Constructs a new detached [`LoadOperation`].
pub fn load<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    memref: ValueRef<'v, 'c, 't>,
    map: AffineMap<'c, 't>,
    indices: &[ValueRef<'v, 'c, 't>],
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedLoadOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::affine()?)?;
    OperationBuilder::new("affine.load", location)
        .add_attribute(MAP_ATTRIBUTE, context.affine_map_attribute(map))
        .add_operand(memref)
        .add_operands(indices)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `affine::load`"))
        })
}

/// Trait representing the `affine.min` operation.
pub trait MinOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the affine map whose results are minimized by this operation.
    fn map(&self) -> Result<AffineMap<'c, 't>, Error> {
        self.affine_map_attribute(MAP_ATTRIBUTE)?.affine_map()
    }

    /// Returns the operands supplied to the affine map.
    fn map_operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().collect()
    }
}

mlir_op!(Min);
mlir_op_trait!(Min, OneResult);
mlir_op_trait!(Min, AlwaysSpeculatable);
mlir_op_trait!(Min, ZeroRegions);
mlir_op_trait!(Min, ZeroSuccessors);
mlir_op_trait!(Min, Pure);
mlir_op_trait!(Min, NoMemoryEffect);

/// Constructs a new detached [`MinOperation`].
pub fn min<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    map: AffineMap<'c, 't>,
    map_operands: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedMinOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::affine()?)?;
    OperationBuilder::new("affine.min", location)
        .add_attribute(MAP_ATTRIBUTE, context.affine_map_attribute(map))
        .add_operands(map_operands)
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `affine::min`"))
        })
}

/// Trait representing the `affine.max` operation.
pub trait MaxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the affine map whose results are maximized by this operation.
    fn map(&self) -> Result<AffineMap<'c, 't>, Error> {
        self.affine_map_attribute(MAP_ATTRIBUTE)?.affine_map()
    }

    /// Returns the operands supplied to the affine map.
    fn map_operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().collect()
    }
}

mlir_op!(Max);
mlir_op_trait!(Max, OneResult);
mlir_op_trait!(Max, AlwaysSpeculatable);
mlir_op_trait!(Max, ZeroRegions);
mlir_op_trait!(Max, ZeroSuccessors);
mlir_op_trait!(Max, Pure);
mlir_op_trait!(Max, NoMemoryEffect);

/// Constructs a new detached [`MaxOperation`].
pub fn max<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    map: AffineMap<'c, 't>,
    map_operands: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedMaxOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::affine()?)?;
    OperationBuilder::new("affine.max", location)
        .add_attribute(MAP_ATTRIBUTE, context.affine_map_attribute(map))
        .add_operands(map_operands)
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `affine::max`"))
        })
}

/// Name of the reduction-kind attribute used by [`ParallelOperation`].
pub const REDUCTIONS_ATTRIBUTE: &str = "reductions";

/// Name of the lower-bound affine map attribute used by [`ParallelOperation`].
pub const LOWER_BOUNDS_MAP_ATTRIBUTE: &str = "lowerBoundsMap";

/// Name of the lower-bound result-grouping attribute used by [`ParallelOperation`].
pub const LOWER_BOUNDS_GROUPS_ATTRIBUTE: &str = "lowerBoundsGroups";

/// Name of the upper-bound affine map attribute used by [`ParallelOperation`].
pub const UPPER_BOUNDS_MAP_ATTRIBUTE: &str = "upperBoundsMap";

/// Name of the upper-bound result-grouping attribute used by [`ParallelOperation`].
pub const UPPER_BOUNDS_GROUPS_ATTRIBUTE: &str = "upperBoundsGroups";

/// Name of the loop-step array attribute used by [`ParallelOperation`].
pub const STEPS_ATTRIBUTE: &str = "steps";

/// Trait representing the `affine.parallel` operation.
pub trait ParallelOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the reduction kinds used by this operation's results.
    fn reductions(&self) -> Result<Vec<AtomicRmwKind>, Error> {
        self.array_attribute(REDUCTIONS_ATTRIBUTE)?
            .elements()
            .map(|attribute| {
                attribute?
                    .cast::<AtomicRmwKindAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `reductions` attribute in `affine.parallel`"))?
                    .value()
            })
            .collect()
    }

    /// Returns the affine map used to compute this operation's lower bounds.
    fn lower_bounds_map(&self) -> Result<AffineMap<'c, 't>, Error> {
        self.affine_map_attribute(LOWER_BOUNDS_MAP_ATTRIBUTE)?.affine_map()
    }

    /// Returns the grouping of lower-bound affine map results by loop dimension.
    fn lower_bounds_groups(&self) -> Result<Vec<i32>, Error> {
        let attribute = self.dense_integer_elements_attribute(LOWER_BOUNDS_GROUPS_ATTRIBUTE)?;
        unsafe { attribute.i32_elements().collect() }
    }

    /// Returns the operands supplied to this operation's lower-bound affine map.
    fn lower_bounds_operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        (0..self.lower_bounds_map()?.input_count()).map(|index| self.operand_value(index)).collect()
    }

    /// Returns the affine map used to compute this operation's upper bounds.
    fn upper_bounds_map(&self) -> Result<AffineMap<'c, 't>, Error> {
        self.affine_map_attribute(UPPER_BOUNDS_MAP_ATTRIBUTE)?.affine_map()
    }

    /// Returns the grouping of upper-bound affine map results by loop dimension.
    fn upper_bounds_groups(&self) -> Result<Vec<i32>, Error> {
        let attribute = self.dense_integer_elements_attribute(UPPER_BOUNDS_GROUPS_ATTRIBUTE)?;
        unsafe { attribute.i32_elements().collect() }
    }

    /// Returns the operands supplied to this operation's upper-bound affine map.
    fn upper_bounds_operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let lower_bound_operand_count = self.lower_bounds_map()?.input_count();
        (lower_bound_operand_count..self.operand_count()).map(|index| self.operand_value(index)).collect()
    }

    /// Returns the positive integer steps for each parallel loop dimension.
    fn steps(&self) -> Result<Vec<i64>, Error> {
        self.array_attribute(STEPS_ATTRIBUTE)?
            .elements()
            .map(|attribute| {
                attribute?
                    .cast::<IntegerAttributeRef>()
                    .map(|attribute| attribute.signless_value())
                    .ok_or_else(|| Error::invalid_argument("invalid `steps` attribute in `affine.parallel`"))
            })
            .collect()
    }

    /// Returns this operation's body region.
    fn body(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }
}

mlir_op!(Parallel);
mlir_op_trait!(Parallel, AutomaticAllocationScope);
mlir_op_trait!(Parallel, MemRefsNormalizable);
mlir_op_trait!(Parallel, ZeroSuccessors);

/// Constructs a new detached [`ParallelOperation`].
pub fn parallel<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    lower_bounds_map: AffineMap<'c, 't>,
    lower_bounds_groups: &[i32],
    lower_bounds_operands: &[ValueRef<'v, 'c, 't>],
    upper_bounds_map: AffineMap<'c, 't>,
    upper_bounds_groups: &[i32],
    upper_bounds_operands: &[ValueRef<'v, 'c, 't>],
    steps: &[i64],
    reductions: &[AtomicRmwKind],
    result_types: &[TypeRef<'c, 't>],
    body: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedParallelOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::affine()?)?;
    let lower_bounds_groups_type = context.vector_type(
        context.signless_integer_type(32),
        &[VectorTypeDimension::Fixed(lower_bounds_groups.len())],
        context.unknown_location(),
    )?;
    let upper_bounds_groups_type = context.vector_type(
        context.signless_integer_type(32),
        &[VectorTypeDimension::Fixed(upper_bounds_groups.len())],
        context.unknown_location(),
    )?;
    let step_type = context.signless_integer_type(64);
    let step_attributes = steps.iter().map(|step| context.integer_attribute(step_type, *step)).collect::<Vec<_>>();
    let reduction_attributes = reductions
        .iter()
        .map(|reduction| context.arith_atomic_rmw_kind_attribute(*reduction))
        .collect::<Result<Vec<_>, _>>()?;
    let builder = OperationBuilder::new("affine.parallel", location)
        .add_attribute(REDUCTIONS_ATTRIBUTE, context.array_attribute(&reduction_attributes))
        .add_attribute(LOWER_BOUNDS_MAP_ATTRIBUTE, context.affine_map_attribute(lower_bounds_map))
        .add_attribute(UPPER_BOUNDS_MAP_ATTRIBUTE, context.affine_map_attribute(upper_bounds_map));
    builder
        .add_attribute(
            LOWER_BOUNDS_GROUPS_ATTRIBUTE,
            context.dense_i32_elements_attribute(lower_bounds_groups_type, lower_bounds_groups)?,
        )
        .add_attribute(
            UPPER_BOUNDS_GROUPS_ATTRIBUTE,
            context.dense_i32_elements_attribute(upper_bounds_groups_type, upper_bounds_groups)?,
        )
        .add_attribute(STEPS_ATTRIBUTE, context.array_attribute(&step_attributes))
        .add_operands(lower_bounds_operands)
        .add_operands(upper_bounds_operands)
        .add_results(result_types)
        .add_region(body)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `affine::parallel`"))
        })
}

/// Name of the read/write flag attribute used by [`PrefetchOperation`].
pub const IS_WRITE_ATTRIBUTE: &str = "isWrite";

/// Name of the locality hint attribute used by [`PrefetchOperation`].
pub const LOCALITY_HINT_ATTRIBUTE: &str = "localityHint";

/// Name of the data/instruction cache flag attribute used by [`PrefetchOperation`].
pub const IS_DATA_CACHE_ATTRIBUTE: &str = "isDataCache";

/// Trait representing the `affine.prefetch` operation.
pub trait PrefetchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the memref being prefetched.
    fn memref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the affine map used to index the memref.
    fn map(&self) -> Result<AffineMap<'c, 't>, Error> {
        self.affine_map_attribute(MAP_ATTRIBUTE)?.affine_map()
    }

    /// Returns the affine index operands used to access the memref.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        (1..self.operand_count()).map(|index| self.operand_value(index)).collect()
    }

    /// Returns `true` if this prefetch writes, and `false` if it reads.
    fn is_write(&self) -> Result<bool, Error> {
        Ok(self.boolean_attribute(IS_WRITE_ATTRIBUTE)?.value())
    }

    /// Returns the locality hint in the inclusive range `0..=3`.
    fn locality_hint(&self) -> Result<i32, Error> {
        Ok(self.integer_attribute(LOCALITY_HINT_ATTRIBUTE)?.signless_value() as i32)
    }

    /// Returns `true` if this prefetch targets the data cache, and `false` if it targets the instruction cache.
    fn is_data_cache(&self) -> Result<bool, Error> {
        Ok(self.boolean_attribute(IS_DATA_CACHE_ATTRIBUTE)?.value())
    }
}

mlir_op!(Prefetch);
mlir_op_trait!(Prefetch, ZeroRegions);
mlir_op_trait!(Prefetch, ZeroSuccessors);
mlir_op_trait!(Prefetch, MemRefsNormalizable);

/// Constructs a new detached [`PrefetchOperation`].
pub fn prefetch<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    memref: ValueRef<'v, 'c, 't>,
    map: AffineMap<'c, 't>,
    indices: &[ValueRef<'v, 'c, 't>],
    is_write: bool,
    locality_hint: i32,
    is_data_cache: bool,
    location: L,
) -> Result<DetachedPrefetchOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::affine()?)?;
    OperationBuilder::new("affine.prefetch", location)
        .add_attribute(MAP_ATTRIBUTE, context.affine_map_attribute(map))
        .add_operand(memref)
        .add_operands(indices)
        .add_attribute(IS_WRITE_ATTRIBUTE, context.boolean_attribute(is_write))
        .add_attribute(
            LOCALITY_HINT_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(32), locality_hint.into()),
        )
        .add_attribute(IS_DATA_CACHE_ATTRIBUTE, context.boolean_attribute(is_data_cache))
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `affine::prefetch`"))
        })
}

/// Trait representing the `affine.store` operation.
pub trait StoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the value being stored.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the memref being stored to.
    fn memref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the affine map used to index the memref.
    fn map(&self) -> Result<AffineMap<'c, 't>, Error> {
        self.affine_map_attribute(MAP_ATTRIBUTE)?.affine_map()
    }

    /// Returns the affine index operands used to access the memref.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        (2..self.operand_count()).map(|index| self.operand_value(index)).collect()
    }
}

mlir_op!(Store);
mlir_op_trait!(Store, ZeroRegions);
mlir_op_trait!(Store, ZeroSuccessors);
mlir_op_trait!(Store, MemRefsNormalizable);

/// Constructs a new detached [`StoreOperation`].
pub fn store<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    value: ValueRef<'v, 'c, 't>,
    memref: ValueRef<'v, 'c, 't>,
    map: AffineMap<'c, 't>,
    indices: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedStoreOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::affine()?)?;
    OperationBuilder::new("affine.store", location)
        .add_attribute(MAP_ATTRIBUTE, context.affine_map_attribute(map))
        .add_operand(value)
        .add_operand(memref)
        .add_operands(indices)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `affine::store`"))
        })
}

/// Trait representing the `affine.yield` operation.
pub trait YieldOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the values yielded by this operation.
    fn values(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().collect()
    }
}

mlir_op!(Yield);
mlir_op_trait!(Yield, SingleBlockRegions);
mlir_op_trait!(Yield, IsTerminator);
mlir_op_trait!(Yield, ZeroRegions);
mlir_op_trait!(Yield, ZeroSuccessors);

/// Constructs a new detached [`YieldOperation`].
pub fn r#yield<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    values: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedYieldOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::affine()?)?;
    OperationBuilder::new("affine.yield", location)
        .add_operands(values)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `affine::yield`"))
        })
}

/// Trait representing the `affine.vector_load` operation.
pub trait VectorLoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the memref being loaded from.
    fn memref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the affine map used to index the memref.
    fn map(&self) -> Result<AffineMap<'c, 't>, Error> {
        self.affine_map_attribute(MAP_ATTRIBUTE)?.affine_map()
    }

    /// Returns the affine index operands used to access the memref.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        (1..self.operand_count()).map(|index| self.operand_value(index)).collect()
    }
}

mlir_op!(VectorLoad);
mlir_op_trait!(VectorLoad, OneResult);
mlir_op_trait!(VectorLoad, ZeroRegions);
mlir_op_trait!(VectorLoad, ZeroSuccessors);
mlir_op_trait!(VectorLoad, MemRefsNormalizable);

/// Constructs a new detached [`VectorLoadOperation`].
pub fn vector_load<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    memref: ValueRef<'v, 'c, 't>,
    map: AffineMap<'c, 't>,
    indices: &[ValueRef<'v, 'c, 't>],
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedVectorLoadOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::affine()?)?;
    OperationBuilder::new("affine.vector_load", location)
        .add_attribute(MAP_ATTRIBUTE, context.affine_map_attribute(map))
        .add_operand(memref)
        .add_operands(indices)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `affine::vector_load`"))
        })
}

/// Trait representing the `affine.vector_store` operation.
pub trait VectorStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the vector value being stored.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the memref being stored to.
    fn memref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the affine map used to index the memref.
    fn map(&self) -> Result<AffineMap<'c, 't>, Error> {
        self.affine_map_attribute(MAP_ATTRIBUTE)?.affine_map()
    }

    /// Returns the affine index operands used to access the memref.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        (2..self.operand_count()).map(|index| self.operand_value(index)).collect()
    }
}

mlir_op!(VectorStore);
mlir_op_trait!(VectorStore, ZeroRegions);
mlir_op_trait!(VectorStore, ZeroSuccessors);
mlir_op_trait!(VectorStore, MemRefsNormalizable);

/// Constructs a new detached [`VectorStoreOperation`].
pub fn vector_store<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    value: ValueRef<'v, 'c, 't>,
    memref: ValueRef<'v, 'c, 't>,
    map: AffineMap<'c, 't>,
    indices: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedVectorStoreOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::affine()?)?;
    OperationBuilder::new("affine.vector_store", location)
        .add_attribute(MAP_ATTRIBUTE, context.affine_map_attribute(map))
        .add_operand(value)
        .add_operand(memref)
        .add_operands(indices)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `affine::vector_store`"))
        })
}

/// Name of the static-basis attribute used by affine index linearization operations.
pub const STATIC_BASIS_ATTRIBUTE: &str = "static_basis";

/// Trait representing the `affine.delinearize_index` operation.
pub trait DelinearizeIndexOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the linear index to delinearize.
    fn linear_index(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the dynamic basis operands.
    fn dynamic_basis(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        (1..self.operand_count()).map(|index| self.operand_value(index)).collect()
    }

    /// Returns the static basis values.
    fn static_basis(&self) -> Result<Vec<i64>, Error> {
        Ok(self.dense_integer_64_array_attribute(STATIC_BASIS_ATTRIBUTE)?.values().collect())
    }
}

mlir_op!(DelinearizeIndex);
mlir_op_trait!(DelinearizeIndex, ZeroRegions);
mlir_op_trait!(DelinearizeIndex, ZeroSuccessors);
mlir_op_trait!(DelinearizeIndex, AlwaysSpeculatable);
mlir_op_trait!(DelinearizeIndex, Pure);
mlir_op_trait!(DelinearizeIndex, NoMemoryEffect);

/// Constructs a new detached [`DelinearizeIndexOperation`].
pub fn delinearize_index<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    linear_index: ValueRef<'v, 'c, 't>,
    dynamic_basis: &[ValueRef<'v, 'c, 't>],
    static_basis: &[i64],
    result_types: &[TypeRef<'c, 't>],
    location: L,
) -> Result<DetachedDelinearizeIndexOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::affine()?)?;
    OperationBuilder::new("affine.delinearize_index", location)
        .add_operand(linear_index)
        .add_operands(dynamic_basis)
        .add_attribute(STATIC_BASIS_ATTRIBUTE, context.dense_i64_array_attribute(static_basis)?)
        .add_results(result_types)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `affine::delinearize_index`"))
        })
}

/// Name of the disjoint hint attribute used by [`LinearizeIndexOperation`].
pub const DISJOINT_ATTRIBUTE: &str = "disjoint";

/// Trait representing the `affine.linearize_index` operation.
pub trait LinearizeIndexOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the multi-index operands to linearize.
    fn multi_index(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns the dynamic basis operands.
    fn dynamic_basis(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns the static basis values.
    fn static_basis(&self) -> Result<Vec<i64>, Error> {
        Ok(self.dense_integer_64_array_attribute(STATIC_BASIS_ATTRIBUTE)?.values().collect())
    }

    /// Returns `true` if the linearization has the `disjoint` optimization hint.
    fn is_disjoint(&self) -> bool {
        self.has_attribute(DISJOINT_ATTRIBUTE)
    }
}

mlir_op!(LinearizeIndex);
mlir_op_trait!(LinearizeIndex, OneResult);
mlir_op_trait!(LinearizeIndex, AlwaysSpeculatable);
mlir_op_trait!(LinearizeIndex, ZeroRegions);
mlir_op_trait!(LinearizeIndex, ZeroSuccessors);
mlir_op_trait!(LinearizeIndex, Pure);
mlir_op_trait!(LinearizeIndex, NoMemoryEffect);

/// Constructs a new detached [`LinearizeIndexOperation`].
pub fn linearize_index<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    multi_index: &[ValueRef<'v, 'c, 't>],
    dynamic_basis: &[ValueRef<'v, 'c, 't>],
    static_basis: &[i64],
    disjoint: bool,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedLinearizeIndexOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::affine()?)?;
    let operand_segment_sizes = [multi_index.len() as i32, dynamic_basis.len() as i32];
    let builder = OperationBuilder::new("affine.linearize_index", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&operand_segment_sizes)?);
    let builder = builder
        .add_operands(multi_index)
        .add_operands(dynamic_basis)
        .add_attribute(STATIC_BASIS_ATTRIBUTE, context.dense_i64_array_attribute(static_basis)?);
    let builder = if disjoint { builder.add_attribute(DISJOINT_ATTRIBUTE, context.unit_attribute()) } else { builder };
    builder.add_result(result_type).build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `affine::linearize_index`"))
    })
}

/// Name of the source affine map attribute used by [`DmaStartOperation`].
pub const SOURCE_MAP_ATTRIBUTE: &str = "src_map";

/// Name of the destination affine map attribute used by [`DmaStartOperation`].
pub const DESTINATION_MAP_ATTRIBUTE: &str = "dst_map";

/// Name of the tag affine map attribute used by DMA affine operations.
pub const TAG_MAP_ATTRIBUTE: &str = "tag_map";

/// Trait representing the `affine.dma_start` operation.
pub trait DmaStartOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the affine map used to index the source memref.
    fn source_map(&self) -> Result<AffineMap<'c, 't>, Error> {
        self.affine_map_attribute(SOURCE_MAP_ATTRIBUTE)?.affine_map()
    }

    /// Returns the source memref indices.
    fn source_indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let end = 1 + self.source_map()?.input_count();
        (1..end).map(|index| self.operand_value(index)).collect()
    }

    /// Returns the destination memref.
    fn destination(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1 + self.source_map()?.input_count())
    }

    /// Returns the affine map used to index the destination memref.
    fn destination_map(&self) -> Result<AffineMap<'c, 't>, Error> {
        self.affine_map_attribute(DESTINATION_MAP_ATTRIBUTE)?.affine_map()
    }

    /// Returns the destination memref indices.
    fn destination_indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let start = 2 + self.source_map()?.input_count();
        let end = start + self.destination_map()?.input_count();
        (start..end).map(|index| self.operand_value(index)).collect()
    }

    /// Returns the tag memref.
    fn tag(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let index = 2 + self.source_map()?.input_count() + self.destination_map()?.input_count();
        self.operand_value(index)
    }

    /// Returns the affine map used to index the tag memref.
    fn tag_map(&self) -> Result<AffineMap<'c, 't>, Error> {
        self.affine_map_attribute(TAG_MAP_ATTRIBUTE)?.affine_map()
    }

    /// Returns the tag memref indices.
    fn tag_indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let start = 3 + self.source_map()?.input_count() + self.destination_map()?.input_count();
        let end = start + self.tag_map()?.input_count();
        (start..end).map(|index| self.operand_value(index)).collect()
    }

    /// Returns the number of elements to transfer.
    fn num_elements(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let index = 3
            + self.source_map()?.input_count()
            + self.destination_map()?.input_count()
            + self.tag_map()?.input_count();
        self.operand_value(index)
    }

    /// Returns `true` if this operation has stride operands.
    fn is_strided(&self) -> Result<bool, Error> {
        let base_operand_count = 4
            + self.source_map()?.input_count()
            + self.destination_map()?.input_count()
            + self.tag_map()?.input_count();
        Ok(self.operand_count() != base_operand_count)
    }

    /// Returns the optional stride operand.
    fn stride(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.is_strided()? { self.operand_value(self.operand_count() - 2).map(Some) } else { Ok(None) }
    }

    /// Returns the optional elements-per-stride operand.
    fn elements_per_stride(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.is_strided()? { self.operand_value(self.operand_count() - 1).map(Some) } else { Ok(None) }
    }
}

mlir_op!(DmaStart);
mlir_op_trait!(DmaStart, ZeroRegions);
mlir_op_trait!(DmaStart, ZeroSuccessors);
mlir_op_trait!(DmaStart, MemRefsNormalizable);

/// Constructs a new detached [`DmaStartOperation`].
pub fn dma_start<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    source_map: AffineMap<'c, 't>,
    source_indices: &[ValueRef<'v, 'c, 't>],
    destination: ValueRef<'v, 'c, 't>,
    destination_map: AffineMap<'c, 't>,
    destination_indices: &[ValueRef<'v, 'c, 't>],
    tag: ValueRef<'v, 'c, 't>,
    tag_map: AffineMap<'c, 't>,
    tag_indices: &[ValueRef<'v, 'c, 't>],
    num_elements: ValueRef<'v, 'c, 't>,
    stride: Option<ValueRef<'v, 'c, 't>>,
    elements_per_stride: Option<ValueRef<'v, 'c, 't>>,
    location: L,
) -> Result<DetachedDmaStartOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::affine()?)?;
    let builder = OperationBuilder::new("affine.dma_start", location)
        .add_operand(source)
        .add_attribute(SOURCE_MAP_ATTRIBUTE, context.affine_map_attribute(source_map));
    let builder = builder.add_operands(source_indices).add_operand(destination);
    let builder = builder.add_attribute(DESTINATION_MAP_ATTRIBUTE, context.affine_map_attribute(destination_map));
    let builder = builder.add_operands(destination_indices).add_operand(tag);
    let builder = builder.add_attribute(TAG_MAP_ATTRIBUTE, context.affine_map_attribute(tag_map));
    let builder = builder.add_operands(tag_indices).add_operand(num_elements);
    let builder = match (stride, elements_per_stride) {
        (Some(stride), Some(elements_per_stride)) => builder.add_operand(stride).add_operand(elements_per_stride),
        (None, None) => builder,
        _ => {
            return Err(Error::invalid_argument("`affine::dma_start` requires either both stride operands or neither"));
        }
    };
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `affine::dma_start`"))
    })
}

/// Trait representing the `affine.dma_wait` operation.
pub trait DmaWaitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the tag memref.
    fn tag(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the affine map used to index the tag memref.
    fn tag_map(&self) -> Result<AffineMap<'c, 't>, Error> {
        self.affine_map_attribute(TAG_MAP_ATTRIBUTE)?.affine_map()
    }

    /// Returns the tag memref indices.
    fn tag_indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let end = 1 + self.tag_map()?.input_count();
        (1..end).map(|index| self.operand_value(index)).collect()
    }

    /// Returns the number of elements associated with the DMA operation.
    fn num_elements(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1 + self.tag_map()?.input_count())
    }
}

mlir_op!(DmaWait);
mlir_op_trait!(DmaWait, ZeroRegions);
mlir_op_trait!(DmaWait, ZeroSuccessors);
mlir_op_trait!(DmaWait, MemRefsNormalizable);

/// Constructs a new detached [`DmaWaitOperation`].
pub fn dma_wait<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    tag: ValueRef<'v, 'c, 't>,
    tag_map: AffineMap<'c, 't>,
    tag_indices: &[ValueRef<'v, 'c, 't>],
    num_elements: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedDmaWaitOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::affine()?)?;
    OperationBuilder::new("affine.dma_wait", location)
        .add_operand(tag)
        .add_attribute(TAG_MAP_ATTRIBUTE, context.affine_map_attribute(tag_map))
        .add_operands(tag_indices)
        .add_operand(num_elements)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `affine::dma_wait`"))
        })
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{
        AffineExpression, Attribute, Block, Context, OneResult, Operation, Region, Size, Type, Value, ValueRef,
    };

    use super::*;

    #[test]
    fn test_affine_dialect_is_not_loaded_by_default() {
        let context = Context::new();

        assert_eq!(context.load_dialect_by_name("affine").unwrap(), None);
        assert!(!context.is_registered("affine.apply"));
    }

    #[test]
    fn test_apply() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let operands = block.arguments().map(|argument| argument.unwrap().as_ref()).collect::<Vec<_>>();
                let expression =
                    (context.dimension_affine_expression(0) + context.symbol_affine_expression(0)).as_ref();
                let map = context.affine_map(1, 1, &[expression]);
                let operation = apply(map, &operands, location).unwrap();
                assert_eq!(operation.map().unwrap(), map);
                assert_eq!(operation.map_operands().unwrap(), operands);
                assert_eq!(operation.dimension_operands().unwrap(), vec![operands[0]]);
                assert_eq!(operation.symbol_operands().unwrap(), vec![operands[1]]);
                let operation = apply(map, &operands, location).unwrap();
                let output = operation.output().unwrap();
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return(&[output], location).unwrap()).unwrap();
                func::func(
                    "test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
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
                #map = affine_map<(d0)[s0] -> (d0 + s0)>
                module {
                  func.func @test(%arg0: index, %arg1: index) -> index {
                    %0 = affine.apply #map(%arg0)[%arg1]
                    return %0 : index
                  }
                }
            "}
        );
    }

    #[test]
    fn test_for() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let mut body = context.block(&[(index_type, location)]);
                body.append_operation(r#yield(&[], location).unwrap()).unwrap();
                let lower_bound_map = context.constant_affine_map(0);
                let upper_bound_map = context.constant_affine_map(4);
                let operation =
                    r#for(lower_bound_map, &[], upper_bound_map, &[], 1, &[], &[], body.try_into().unwrap(), location)
                        .unwrap();
                assert_eq!(operation.lower_bound_map().unwrap(), lower_bound_map);
                assert_eq!(operation.upper_bound_map().unwrap(), upper_bound_map);
                assert_eq!(operation.lower_bound_operands().unwrap(), Vec::<ValueRef<'_, '_, '_>>::new());
                assert_eq!(operation.upper_bound_operands().unwrap(), Vec::<ValueRef<'_, '_, '_>>::new());
                assert_eq!(operation.inits().unwrap(), Vec::<ValueRef<'_, '_, '_>>::new());
                assert_eq!(operation.step().unwrap(), 1);
                assert_eq!(operation.body().unwrap().blocks().unwrap().next().unwrap().unwrap().argument_count(), 1);
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func("test", func::FuncAttributes::default(), block.try_into().unwrap(), location).unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @test() {
                    affine.for %arg0 = 0 to 4 {
                    }
                    return
                  }
                }
            "}
        );
    }

    #[test]
    fn test_if() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location)]);
                let condition_operand = block.argument(0).unwrap().as_ref();
                let condition_operands = [condition_operand];
                let mut then_block = context.block_with_no_arguments();
                then_block.append_operation(r#yield(&[], location).unwrap()).unwrap();
                let mut else_block = context.block_with_no_arguments();
                else_block.append_operation(r#yield(&[], location).unwrap()).unwrap();
                let condition = context.empty_integer_set(1, 0);
                let operation = r#if(
                    condition,
                    &condition_operands,
                    &[],
                    then_block.try_into().unwrap(),
                    else_block.try_into().unwrap(),
                    location,
                )
                .unwrap();
                assert_eq!(operation.condition().unwrap().dimension_count(), 1);
                assert_eq!(operation.condition_operands().unwrap(), vec![condition_operand]);
                assert_eq!(operation.then_region().unwrap().blocks().unwrap().count(), 1);
                assert_eq!(operation.else_region().unwrap().blocks().unwrap().count(), 1);
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test",
                    func::FuncAttributes { arguments: vec![index_type.into()], ..Default::default() },
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
                #set = affine_set<(d0) : (1 == 0)>
                module {
                  func.func @test(%arg0: index) {
                    affine.if #set(%arg0) {
                    } else {
                    }
                    return
                  }
                }
            "}
        );
    }

    #[test]
    fn test_load() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let index_type = context.index_type();
        let memref_type = context.mem_ref_type(i32_type, &[Size::Static(4)], None, None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(memref_type.as_ref(), location), (index_type.as_ref(), location)]);
                let memref = block.argument(0).unwrap().as_ref();
                let index = block.argument(1).unwrap().as_ref();
                let map = context.identity_affine_map(1);
                let operation = load(memref, map, &[index], i32_type.as_ref(), location).unwrap();
                assert_eq!(operation.memref().unwrap(), memref);
                assert_eq!(operation.map().unwrap(), map);
                assert_eq!(operation.indices().unwrap(), vec![index]);
                let operation = load(memref, map, &[index], i32_type.as_ref(), location).unwrap();
                let output = operation.output().unwrap();
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return(&[output], location).unwrap()).unwrap();
                func::func(
                    "test",
                    func::FuncAttributes {
                        arguments: vec![memref_type.into(), index_type.into()],
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
                  func.func @test(%arg0: memref<4xi32>, %arg1: index) -> i32 {
                    %0 = affine.load %arg0[%arg1] : memref<4xi32>
                    return %0 : i32
                  }
                }
            "}
        );
    }

    #[test]
    fn test_min() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let operands = block.arguments().map(|argument| argument.unwrap().as_ref()).collect::<Vec<_>>();
                let dimension = context.dimension_affine_expression(0);
                let symbol = context.symbol_affine_expression(0);
                let map = context.affine_map(1, 1, &[dimension.as_ref(), symbol.as_ref()]);
                let operation = min(map, &operands, location).unwrap();
                assert_eq!(operation.map().unwrap(), map);
                assert_eq!(operation.map_operands().unwrap(), operands);
                let operation = min(map, &operands, location).unwrap();
                let output = operation.output().unwrap();
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return(&[output], location).unwrap()).unwrap();
                func::func(
                    "test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
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
                #map = affine_map<(d0)[s0] -> (d0, s0)>
                module {
                  func.func @test(%arg0: index, %arg1: index) -> index {
                    %0 = affine.min #map(%arg0)[%arg1]
                    return %0 : index
                  }
                }
            "}
        );
    }

    #[test]
    fn test_max() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let operands = block.arguments().map(|argument| argument.unwrap().as_ref()).collect::<Vec<_>>();
                let dimension = context.dimension_affine_expression(0);
                let symbol = context.symbol_affine_expression(0);
                let map = context.affine_map(1, 1, &[dimension.as_ref(), symbol.as_ref()]);
                let operation = max(map, &operands, location).unwrap();
                assert_eq!(operation.map().unwrap(), map);
                assert_eq!(operation.map_operands().unwrap(), operands);
                let operation = max(map, &operands, location).unwrap();
                let output = operation.output().unwrap();
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return(&[output], location).unwrap()).unwrap();
                func::func(
                    "test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
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
                #map = affine_map<(d0)[s0] -> (d0, s0)>
                module {
                  func.func @test(%arg0: index, %arg1: index) -> index {
                    %0 = affine.max #map(%arg0)[%arg1]
                    return %0 : index
                  }
                }
            "}
        );
    }

    #[test]
    fn test_parallel() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type, location)]);
                let input = block.argument(0).unwrap().as_ref();
                let mut body = context.block(&[(index_type, location)]);
                body.append_operation(r#yield(&[input], location).unwrap()).unwrap();
                let lower_bounds_map = context.constant_affine_map(0);
                let upper_bounds_map = context.constant_affine_map(4);
                let operation = parallel(
                    lower_bounds_map,
                    &[1],
                    &[],
                    upper_bounds_map,
                    &[1],
                    &[],
                    &[1],
                    &[AtomicRmwKind::AddInteger],
                    &[i32_type.as_ref()],
                    body.try_into().unwrap(),
                    location,
                )
                .unwrap();
                assert_eq!(operation.reductions().unwrap(), vec![AtomicRmwKind::AddInteger]);
                assert_eq!(operation.lower_bounds_map().unwrap(), lower_bounds_map);
                assert_eq!(operation.upper_bounds_map().unwrap(), upper_bounds_map);
                assert_eq!(operation.lower_bounds_groups().unwrap(), vec![1]);
                assert_eq!(operation.upper_bounds_groups().unwrap(), vec![1]);
                assert_eq!(operation.lower_bounds_operands().unwrap(), Vec::<ValueRef<'_, '_, '_>>::new());
                assert_eq!(operation.upper_bounds_operands().unwrap(), Vec::<ValueRef<'_, '_, '_>>::new());
                assert_eq!(operation.steps().unwrap(), vec![1]);
                assert_eq!(operation.body().unwrap().blocks().unwrap().next().unwrap().unwrap().argument_count(), 1);
                let output = operation.result(0).unwrap();
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return(&[output], location).unwrap()).unwrap();
                func::func(
                    "test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into()],
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
                  func.func @test(%arg0: i32) -> i32 {
                    %0 = affine.parallel (%arg1) = (0) to (4) reduce (\"addi\") -> (i32) {
                      affine.yield %arg0 : i32
                    }
                    return %0 : i32
                  }
                }
            "}
        );
    }

    #[test]
    fn test_prefetch() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let index_type = context.index_type();
        let memref_type = context.mem_ref_type(i32_type, &[Size::Static(4)], None, None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(memref_type.as_ref(), location), (index_type.as_ref(), location)]);
                let memref = block.argument(0).unwrap().as_ref();
                let index = block.argument(1).unwrap().as_ref();
                let map = context.identity_affine_map(1);
                let operation = prefetch(memref, map, &[index], false, 3, true, location).unwrap();
                assert_eq!(operation.memref().unwrap(), memref);
                assert_eq!(operation.map().unwrap(), map);
                assert_eq!(operation.indices().unwrap(), vec![index]);
                assert!(!operation.is_write().unwrap());
                assert_eq!(operation.locality_hint().unwrap(), 3);
                assert!(operation.is_data_cache().unwrap());
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test",
                    func::FuncAttributes {
                        arguments: vec![memref_type.into(), index_type.into()],
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
                  func.func @test(%arg0: memref<4xi32>, %arg1: index) {
                    affine.prefetch %arg0[%arg1], read, locality<3>, data : memref<4xi32>
                    return
                  }
                }
            "}
        );
    }

    #[test]
    fn test_store() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let index_type = context.index_type();
        let memref_type = context.mem_ref_type(i32_type, &[Size::Static(4)], None, None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (memref_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                    (index_type.as_ref(), location),
                ]);
                let memref = block.argument(0).unwrap().as_ref();
                let value = block.argument(1).unwrap().as_ref();
                let index = block.argument(2).unwrap().as_ref();
                let map = context.identity_affine_map(1);
                let operation = store(value, memref, map, &[index], location).unwrap();
                assert_eq!(operation.value().unwrap(), value);
                assert_eq!(operation.memref().unwrap(), memref);
                assert_eq!(operation.map().unwrap(), map);
                assert_eq!(operation.indices().unwrap(), vec![index]);
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test",
                    func::FuncAttributes {
                        arguments: vec![memref_type.into(), i32_type.into(), index_type.into()],
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
                  func.func @test(%arg0: memref<4xi32>, %arg1: i32, %arg2: index) {
                    affine.store %arg1, %arg0[%arg2] : memref<4xi32>
                    return
                  }
                }
            "}
        );
    }

    #[test]
    fn test_yield() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type, location)]);
                let input = block.argument(0).unwrap().as_ref();
                let mut body = context.block(&[(index_type, location)]);
                let operation = r#yield(&[input], location).unwrap();
                assert_eq!(operation.values().unwrap(), vec![input]);
                body.append_operation(operation).unwrap();
                let parallel_operation = parallel(
                    context.constant_affine_map(0),
                    &[1],
                    &[],
                    context.constant_affine_map(4),
                    &[1],
                    &[],
                    &[1],
                    &[AtomicRmwKind::AddInteger],
                    &[i32_type.as_ref()],
                    body.try_into().unwrap(),
                    location,
                )
                .unwrap();
                let output = parallel_operation.result(0).unwrap();
                block.append_operation(parallel_operation).unwrap();
                block.append_operation(func::r#return(&[output], location).unwrap()).unwrap();
                func::func(
                    "test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into()],
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
                  func.func @test(%arg0: i32) -> i32 {
                    %0 = affine.parallel (%arg1) = (0) to (4) reduce (\"addi\") -> (i32) {
                      affine.yield %arg0 : i32
                    }
                    return %0 : i32
                  }
                }
            "}
        );
    }

    #[test]
    fn test_vector_load() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        let index_type = context.index_type();
        let memref_type = context.mem_ref_type(f32_type, &[Size::Static(16)], None, None, location).unwrap();
        let vector_type = context.vector_type(f32_type, &[VectorTypeDimension::Fixed(4)], location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(memref_type.as_ref(), location), (index_type.as_ref(), location)]);
                let memref = block.argument(0).unwrap().as_ref();
                let index = block.argument(1).unwrap().as_ref();
                let map = context.identity_affine_map(1);
                let operation = vector_load(memref, map, &[index], vector_type.as_ref(), location).unwrap();
                assert_eq!(operation.memref().unwrap(), memref);
                assert_eq!(operation.map().unwrap(), map);
                assert_eq!(operation.indices().unwrap(), vec![index]);
                assert_eq!(operation.output_type().unwrap(), vector_type);
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test",
                    func::FuncAttributes {
                        arguments: vec![memref_type.into(), index_type.into()],
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
                  func.func @test(%arg0: memref<16xf32>, %arg1: index) {
                    %0 = affine.vector_load %arg0[%arg1] : memref<16xf32>, vector<4xf32>
                    return
                  }
                }
            "}
        );
    }

    #[test]
    fn test_vector_store() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        let index_type = context.index_type();
        let memref_type = context.mem_ref_type(f32_type, &[Size::Static(16)], None, None, location).unwrap();
        let vector_type = context.vector_type(f32_type, &[VectorTypeDimension::Fixed(4)], location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (memref_type.as_ref(), location),
                    (vector_type.as_ref(), location),
                    (index_type.as_ref(), location),
                ]);
                let memref = block.argument(0).unwrap().as_ref();
                let value = block.argument(1).unwrap().as_ref();
                let index = block.argument(2).unwrap().as_ref();
                let map = context.identity_affine_map(1);
                let operation = vector_store(value, memref, map, &[index], location).unwrap();
                assert_eq!(operation.value().unwrap(), value);
                assert_eq!(operation.memref().unwrap(), memref);
                assert_eq!(operation.map().unwrap(), map);
                assert_eq!(operation.indices().unwrap(), vec![index]);
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test",
                    func::FuncAttributes {
                        arguments: vec![memref_type.into(), vector_type.into(), index_type.into()],
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
                  func.func @test(%arg0: memref<16xf32>, %arg1: vector<4xf32>, %arg2: index) {
                    affine.vector_store %arg1, %arg0[%arg2] : memref<16xf32>, vector<4xf32>
                    return
                  }
                }
            "}
        );
    }

    #[test]
    fn test_delinearize_index() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location)]);
                let linear_index = block.argument(0).unwrap().as_ref();
                let operation = delinearize_index(
                    linear_index,
                    &[],
                    &[4, 8],
                    &[index_type.as_ref(), index_type.as_ref()],
                    location,
                )
                .unwrap();
                assert_eq!(operation.linear_index().unwrap(), linear_index);
                assert_eq!(operation.dynamic_basis().unwrap(), Vec::<ValueRef<'_, '_, '_>>::new());
                assert_eq!(operation.static_basis().unwrap(), vec![4, 8]);
                assert_eq!(operation.result_count(), 2);
                let results =
                    operation.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().collect::<Vec<_>>();
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return(&results, location).unwrap()).unwrap();
                func::func(
                    "test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into()],
                        results: vec![index_type.into(), index_type.into()],
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
                  func.func @test(%arg0: index) -> (index, index) {
                    %0:2 = affine.delinearize_index %arg0 into (4, 8) : index, index
                    return %0#0, %0#1 : index, index
                  }
                }
            "}
        );
    }

    #[test]
    fn test_linearize_index() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let multi_index = block.arguments().map(|argument| argument.unwrap().as_ref()).collect::<Vec<_>>();
                let operation =
                    linearize_index(&multi_index, &[], &[4, 8], true, index_type.as_ref(), location).unwrap();
                assert_eq!(operation.multi_index().unwrap(), multi_index);
                assert_eq!(operation.dynamic_basis().unwrap(), Vec::<ValueRef<'_, '_, '_>>::new());
                assert_eq!(operation.static_basis().unwrap(), vec![4, 8]);
                assert!(operation.is_disjoint());
                assert_eq!(operation.output().unwrap().r#type().unwrap(), index_type);
                let output = operation.output().unwrap();
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return(&[output], location).unwrap()).unwrap();
                func::func(
                    "test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
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
                  func.func @test(%arg0: index, %arg1: index) -> index {
                    %0 = affine.linearize_index [%arg0, %arg1] by (4, 8) {disjoint} : index
                    return %0 : index
                  }
                }
            "}
        );
    }

    #[test]
    fn test_dma_start() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        let i32_type = context.signless_integer_type(32);
        let index_type = context.index_type();
        let source_memory_space = context.integer_attribute(index_type, 0);
        let destination_memory_space = context.integer_attribute(index_type, 1);
        let tag_memory_space = context.integer_attribute(index_type, 2);
        let source_type = context
            .mem_ref_type(f32_type, &[Size::Static(16)], None, Some(source_memory_space.as_ref()), location)
            .unwrap();
        let destination_type = context
            .mem_ref_type(f32_type, &[Size::Static(16)], None, Some(destination_memory_space.as_ref()), location)
            .unwrap();
        let tag_type = context
            .mem_ref_type(i32_type, &[Size::Static(1)], None, Some(tag_memory_space.as_ref()), location)
            .unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (source_type.as_ref(), location),
                    (destination_type.as_ref(), location),
                    (tag_type.as_ref(), location),
                    (index_type.as_ref(), location),
                    (index_type.as_ref(), location),
                ]);
                let source = block.argument(0).unwrap().as_ref();
                let destination = block.argument(1).unwrap().as_ref();
                let tag = block.argument(2).unwrap().as_ref();
                let num_elements = block.argument(3).unwrap().as_ref();
                let index = block.argument(4).unwrap().as_ref();
                let map = context.identity_affine_map(1);
                let operation = dma_start(
                    source,
                    map,
                    &[index],
                    destination,
                    map,
                    &[index],
                    tag,
                    map,
                    &[index],
                    num_elements,
                    None,
                    None,
                    location,
                )
                .unwrap();
                assert_eq!(operation.source().unwrap(), source);
                assert_eq!(operation.destination().unwrap(), destination);
                assert_eq!(operation.tag().unwrap(), tag);
                assert_eq!(operation.source_map().unwrap(), map);
                assert_eq!(operation.destination_map().unwrap(), map);
                assert_eq!(operation.tag_map().unwrap(), map);
                assert_eq!(operation.source_indices().unwrap(), vec![index]);
                assert_eq!(operation.destination_indices().unwrap(), vec![index]);
                assert_eq!(operation.tag_indices().unwrap(), vec![index]);
                assert_eq!(operation.num_elements().unwrap(), num_elements);
                assert!(!operation.is_strided().unwrap());
                assert!(operation.stride().unwrap().is_none());
                assert!(operation.elements_per_stride().unwrap().is_none());
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test",
                    func::FuncAttributes {
                        arguments: vec![
                            source_type.into(),
                            destination_type.into(),
                            tag_type.into(),
                            index_type.into(),
                            index_type.into(),
                        ],
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
            concat!(
                "module {\n",
                "  func.func @test(%arg0: memref<16xf32>, %arg1: memref<16xf32, 1 : index>, ",
                "%arg2: memref<1xi32, 2 : index>, %arg3: index, %arg4: index) {\n",
                "    affine.dma_start %arg0[%arg4], %arg1[%arg4], %arg2[%arg4], %arg3 : ",
                "memref<16xf32>, memref<16xf32, 1 : index>, memref<1xi32, 2 : index>\n",
                "    return\n",
                "  }\n",
                "}\n",
            )
        );
    }

    #[test]
    fn test_dma_wait() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let index_type = context.index_type();
        let tag_memory_space = context.integer_attribute(index_type, 2);
        let tag_type = context
            .mem_ref_type(i32_type, &[Size::Static(1)], None, Some(tag_memory_space.as_ref()), location)
            .unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (tag_type.as_ref(), location),
                    (index_type.as_ref(), location),
                    (index_type.as_ref(), location),
                ]);
                let tag = block.argument(0).unwrap().as_ref();
                let num_elements = block.argument(1).unwrap().as_ref();
                let index = block.argument(2).unwrap().as_ref();
                let map = context.identity_affine_map(1);
                let operation = dma_wait(tag, map, &[index], num_elements, location).unwrap();
                assert_eq!(operation.tag().unwrap(), tag);
                assert_eq!(operation.tag_map().unwrap(), map);
                assert_eq!(operation.tag_indices().unwrap(), vec![index]);
                assert_eq!(operation.num_elements().unwrap(), num_elements);
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test",
                    func::FuncAttributes {
                        arguments: vec![tag_type.into(), index_type.into(), index_type.into()],
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
                  func.func @test(%arg0: memref<1xi32, 2 : index>, %arg1: index, %arg2: index) {
                    affine.dma_wait %arg0[%arg2], %arg1 : memref<1xi32, 2 : index>
                    return
                  }
                }
            "}
        );
    }
}
