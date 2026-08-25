use super::*;

/// Returns whether `dimension` is sharded over at least one [`MeshAxisType::Explicit`] mesh axis of `mesh`.
///
/// The explicit-mode dot sharding rules — contracting-dimension ambiguity, batch-dimension consistency, and the
/// unreduced-operand rejection — apply only to Explicit axes. [`MeshAxisType::Manual`] axes are managed by the user
/// inside `shard_map` (a local per-shard dot over a manual-sharded contracting dimension is ordinary, not
/// ambiguous), and [`MeshAxisType::Auto`] axes are left to the compiler's propagation. A dimension sharded only over
/// those axis types therefore passes through these checks, mirroring how JAX gates its trace-time sharding rules to
/// Explicit mesh axes.
fn dimension_has_explicit_axis(mesh: &LogicalMesh, dimension: &ShardingDimension) -> bool {
    match dimension {
        ShardingDimension::Sharded(axis_names) => {
            axis_names.iter().any(|axis_name| mesh.axis_type(axis_name) == Some(MeshAxisType::Explicit))
        }
        ShardingDimension::Replicated | ShardingDimension::Unconstrained => false,
    }
}

/// Merges the [`ShardingDimension`]s of one aligned batch dimension pair, preferring the more informative entry
/// (`Sharded` over `Replicated` over `Unconstrained`). Returns [`None`] when the two entries are sharded over
/// different mesh axes, which the caller reports as an inconsistent-sharding error. Note that preferring a one-sided
/// `Sharded` entry over the other operand's entry intentionally diverges from JAX, which reads output batch
/// dimension specs from the LHS operand only.
fn merge_batch_sharding_dimensions(lhs: &ShardingDimension, rhs: &ShardingDimension) -> Option<ShardingDimension> {
    match (lhs, rhs) {
        (ShardingDimension::Sharded(left), ShardingDimension::Sharded(right)) if left == right => Some(lhs.clone()),
        (ShardingDimension::Sharded(_), ShardingDimension::Sharded(_)) => None,
        (ShardingDimension::Sharded(_), _) => Some(lhs.clone()),
        (_, ShardingDimension::Sharded(_)) => Some(rhs.clone()),
        (ShardingDimension::Replicated, _) | (_, ShardingDimension::Replicated) => Some(ShardingDimension::Replicated),
        (ShardingDimension::Unconstrained, ShardingDimension::Unconstrained) => Some(ShardingDimension::Unconstrained),
    }
}

/// Canonical operation name for [`DotOperation`].
pub const DOT_OPERATION_NAME: &str = "dot";

/// Canonical operation name for [`RaggedDotOperation`].
pub const RAGGED_DOT_OPERATION_NAME: &str = "ragged_dot_general";

/// Computes the abstract result type of one grouped generalized dot product.
pub(crate) fn ragged_dot_abstract(
    lhs: &ArrayType,
    rhs: &ArrayType,
    group_sizes: &ArrayType,
    dimensions: &RaggedDotDimensionNumbers,
) -> Result<ArrayType, TypeError> {
    if !group_sizes.data_type().is_integer() {
        return Err(TypeError::invalid(format!(
            "`{RAGGED_DOT_OPERATION_NAME}` group sizes must have an integer data type",
        )));
    }
    if group_sizes.rank() == 0 {
        return Err(TypeError::invalid(format!(
            "`{RAGGED_DOT_OPERATION_NAME}` group sizes must have rank at least one",
        )));
    }
    let mode = dimensions.mode(lhs.rank())?;
    let prefix_axes = dimensions.group_sizes_prefix_dimensions(lhs.rank())?;
    if group_sizes.rank() != 1 {
        let expected_prefix = Shape::new(prefix_axes.iter().map(|axis| lhs.dimension(*axis)).collect());
        let actual_prefix = Shape::new(group_sizes.shape().dimensions()[..group_sizes.rank() - 1].to_vec());
        if actual_prefix != expected_prefix {
            return Err(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` group sizes prefix must be `{expected_prefix}`, but got \
                 `{actual_prefix}`",
            )));
        }
    }
    let group_count = group_sizes.dimension(group_sizes.rank() - 1);
    let rhs_group_dimensions = dimensions.rhs_group_dimensions();
    match mode {
        RaggedDotMode::NonContracting => {
            if rhs_group_dimensions.len() != 1 {
                return Err(TypeError::invalid(format!(
                    "`{RAGGED_DOT_OPERATION_NAME}` requires exactly one RHS group dimension when the LHS ragged \
                     dimension is non-contracting",
                )));
            }
            let group_axis = rhs_group_dimensions[0];
            if group_axis >= rhs.rank() {
                return Err(TypeError::invalid(format!(
                    "`{RAGGED_DOT_OPERATION_NAME}` RHS group dimension {group_axis} is out of bounds for rank {}",
                    rhs.rank(),
                )));
            }
            let dot_dimensions = dimensions.dot_dimensions();
            if dot_dimensions.rhs_batching_dimensions().contains(&group_axis)
                || dot_dimensions.rhs_contracting_dimensions().contains(&group_axis)
            {
                return Err(TypeError::invalid(format!(
                    "`{RAGGED_DOT_OPERATION_NAME}` RHS group dimension must be distinct from batching and contracting \
                     dimensions",
                )));
            }
            if rhs.dimension(group_axis) != group_count {
                return Err(TypeError::invalid(format!(
                    "`{RAGGED_DOT_OPERATION_NAME}` RHS group dimension has extent {} but group sizes describe \
                     {group_count}",
                    rhs.dimension(group_axis),
                )));
            }
        }
        RaggedDotMode::Contracting | RaggedDotMode::Batch if !rhs_group_dimensions.is_empty() => {
            return Err(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` requires zero RHS group dimensions when the LHS ragged dimension is \
                 contracting or batching",
            )));
        }
        RaggedDotMode::Contracting | RaggedDotMode::Batch => {}
    }

    let mut output = dot_abstract(lhs, rhs, dimensions.dot_dimensions(), None, None)?;
    if mode != RaggedDotMode::Batch && lhs.data_type() == DataType::F8E8M0FNU {
        return Err(TypeError::invalid(format!(
            "`{RAGGED_DOT_OPERATION_NAME}` does not support element data type `f8e8m0fnu` in grouped expansion \
             modes because it cannot represent zero",
        )));
    }
    match mode {
        RaggedDotMode::NonContracting => {
            let rhs_results = rhs_result_axes(dimensions.dot_dimensions(), rhs.rank());
            let group_position = rhs_results.iter().position(|axis| *axis == rhs_group_dimensions[0]).unwrap();
            let output_axis = dimensions.dot_dimensions().lhs_batching_dimensions().len()
                + lhs_result_axes(dimensions.dot_dimensions(), lhs.rank()).len()
                + group_position;
            output = output.without_dimension(output_axis)?.0;
        }
        RaggedDotMode::Contracting => output = output.with_inserted_dimension(0, group_count)?,
        RaggedDotMode::Batch => {}
    }
    Ok(output)
}

/// Returns whether operands of `operand` element type may accumulate at `accumulation`.
///
/// The identical type is always valid. Floating-point operands may accumulate at `f32` or `f64`, and integer
/// operands may accumulate at a same-signedness integer type at least as wide.
fn accumulation_type_is_compatible(operand: DataType, accumulation: DataType) -> bool {
    /// Returns the signedness and bit width of an integer data type, or `None` for any other type.
    fn integer_parts(data_type: DataType) -> Option<(bool, usize)> {
        Some(match data_type {
            DataType::I1 => (true, 1),
            DataType::I2 => (true, 2),
            DataType::I4 => (true, 4),
            DataType::I8 => (true, 8),
            DataType::I16 => (true, 16),
            DataType::I32 => (true, 32),
            DataType::I64 => (true, 64),
            DataType::U1 => (false, 1),
            DataType::U2 => (false, 2),
            DataType::U4 => (false, 4),
            DataType::U8 => (false, 8),
            DataType::U16 => (false, 16),
            DataType::U32 => (false, 32),
            DataType::U64 => (false, 64),
            _ => return None,
        })
    }

    if operand == accumulation {
        return true;
    }
    if operand.is_floating_point() {
        return matches!(accumulation, DataType::F32 | DataType::F64);
    }
    match (integer_parts(operand), integer_parts(accumulation)) {
        (Some((operand_signed, operand_width)), Some((accumulation_signed, accumulation_width))) => {
            operand_signed == accumulation_signed && accumulation_width >= operand_width
        }
        _ => false,
    }
}

/// Computes the abstract output type of one generalized dot product.
///
/// The result shape is `[batching..., lhs_result..., rhs_result...]`, where the result dimensions are the operand
/// axes that are neither batching nor contracting, in their original order. The output element type is the requested
/// compatible accumulation type when one is provided, or otherwise the common operand element type.
///
/// The output [`Sharding`] follows JAX's `dot_general` sharding rule (`_dot_general_sharding_rule` in
/// `jax/_src/lax/lax.py`); refer to the
/// [StableHLO `dot_general` specification](https://openxla.org/stablehlo/spec#dot_general) for the underlying
/// operation semantics. Concretely:
///
///   - When `output_sharding` is provided, it is validated (rank, mesh, no auto axes, and the unreduced-output rule
///     requiring identically sharded contracting dimensions whose sharding axes equal the requested unreduced set)
///     and returned directly, bypassing the consistency checks below.
///   - Operands must not be unreduced. Reduced operands are legal, and reduced and varying manual axes are unioned
///     across the operands into the output sharding.
///   - When neither operand carries a sharding, the output carries none. When exactly one does, the rule runs with
///     the absent side treated as fully replicated on the present operand's mesh. Operand meshes must match.
///   - Batch dimension entries are merged preferring the more informative entry (`Sharded` over `Replicated` over
///     `Unconstrained`); two different `Sharded` entries are an error. This intentionally diverges from JAX, which
///     reads batch dimension specs from the LHS operand only.
///   - Contracting dimensions sharded on both operands are an error (identically sharded ones make the output
///     sharding ambiguous and require an explicit output sharding); a contracting dimension sharded on only one
///     operand is allowed and its sharding is dropped from the output.
///   - Result dimension entries are copied from the owning operand, and auto mesh axes are stripped from the final
///     sharding.
pub(crate) fn dot_abstract(
    lhs: &ArrayType,
    rhs: &ArrayType,
    dimensions: &DotDimensionNumbers,
    accumulation_type: Option<DataType>,
    output_sharding: Option<&Sharding>,
) -> Result<ArrayType, TypeError> {
    if lhs.data_type() != rhs.data_type() {
        return Err(TypeError::invalid(format!("`{DOT_OPERATION_NAME}` input element types are incompatible")));
    }
    if let Some(accumulation_type) = accumulation_type {
        if output_sharding.is_some() {
            return Err(TypeError::invalid(format!(
                "`{DOT_OPERATION_NAME}` does not support combining an accumulation type with a requested \
                     output sharding yet",
            )));
        }
        if !accumulation_type_is_compatible(lhs.data_type(), accumulation_type) {
            return Err(TypeError::invalid(format!(
                "`{DOT_OPERATION_NAME}` operand data type {} cannot accumulate at data type {accumulation_type}",
                lhs.data_type(),
            )));
        }
    }
    let lhs_rank = lhs.rank();
    let rhs_rank = rhs.rank();
    let lhs_batching = dimensions.lhs_batching_dimensions();
    let rhs_batching = dimensions.rhs_batching_dimensions();
    let lhs_contracting = dimensions.lhs_contracting_dimensions();
    let rhs_contracting = dimensions.rhs_contracting_dimensions();

    if lhs_batching.len() != rhs_batching.len() {
        return Err(TypeError::invalid(format!(
            "`{DOT_OPERATION_NAME}` batching dimensions have different lengths on the two operands"
        )));
    }
    if lhs_contracting.len() != rhs_contracting.len() {
        return Err(TypeError::invalid(format!(
            "`{DOT_OPERATION_NAME}` contracting dimensions have different lengths on the two operands"
        )));
    }
    if lhs_batching.iter().any(|axis| *axis >= lhs_rank) || lhs_contracting.iter().any(|axis| *axis >= lhs_rank) {
        return Err(TypeError::invalid(format!("`{DOT_OPERATION_NAME}` LHS dimension index out of bounds")));
    }
    if rhs_batching.iter().any(|axis| *axis >= rhs_rank) || rhs_contracting.iter().any(|axis| *axis >= rhs_rank) {
        return Err(TypeError::invalid(format!("`{DOT_OPERATION_NAME}` RHS dimension index out of bounds")));
    }

    for (lhs_axis, rhs_axis) in lhs_batching.iter().zip(rhs_batching.iter()) {
        if lhs.dimension(*lhs_axis) != rhs.dimension(*rhs_axis) {
            return Err(TypeError::invalid(format!(
                "`{DOT_OPERATION_NAME}` batching dimension sizes do not match (LHS axis {lhs_axis}, RHS axis {rhs_axis})"
            )));
        }
    }
    for (lhs_axis, rhs_axis) in lhs_contracting.iter().zip(rhs_contracting.iter()) {
        if lhs.dimension(*lhs_axis) != rhs.dimension(*rhs_axis) {
            return Err(TypeError::invalid(format!(
                "`{DOT_OPERATION_NAME}` contracting dimension sizes do not match (LHS axis {lhs_axis}, RHS axis {rhs_axis})"
            )));
        }
    }

    let lhs_result = lhs_result_axes(dimensions, lhs_rank);
    let rhs_result = rhs_result_axes(dimensions, rhs_rank);

    let output_dimensions: Vec<Dimension> = lhs_batching
        .iter()
        .map(|axis| lhs.dimension(*axis))
        .chain(lhs_result.iter().map(|axis| lhs.dimension(*axis)))
        .chain(rhs_result.iter().map(|axis| rhs.dimension(*axis)))
        .collect();

    // Output sharding (JAX's `_dot_general_sharding_rule`; see the rule summary above). When `output_sharding` is
    // provided it is validated and returned directly, bypassing the batch and contracting dimension consistency
    // checks (matching JAX's `out_sharding` behavior).
    let lhs_sharding = lhs.sharding();
    let rhs_sharding = rhs.sharding();

    // Operands unreduced over an Explicit axis are rejected on every path, including the explicit `output_sharding`
    // bypass: a pending cross-device reduction must be discharged (e.g., through a sharding constraint) before the
    // value is contracted. The check is gated to Explicit axes — a `shard_map` value unreduced over a Manual axis is
    // the user's to manage. Reduced operands are always legal; this is what lets adjoint dots consume reduced
    // cotangents.
    for sharding in [lhs_sharding, rhs_sharding].into_iter().flatten() {
        if sharding
            .unreduced_axes()
            .iter()
            .any(|axis_name| sharding.mesh().axis_type(axis_name) == Some(MeshAxisType::Explicit))
        {
            return Err(TypeError::invalid(format!("`{DOT_OPERATION_NAME}` operands cannot be unreduced")));
        }
    }

    let mesh = match (lhs_sharding, rhs_sharding) {
        (Some(left), Some(right)) if left.mesh() != right.mesh() => {
            return Err(TypeError::invalid(format!("`{DOT_OPERATION_NAME}` operand shardings must use the same mesh")));
        }
        (Some(left), _) => Some(left.mesh()),
        (_, Some(right)) => Some(right.mesh()),
        (None, None) => None,
    };

    // A missing operand sharding is treated as fully replicated so that one-sided shardings still propagate.
    let dimension_of = |sharding: Option<&Sharding>, axis: usize| -> ShardingDimension {
        sharding.map_or(ShardingDimension::Replicated, |sharding| sharding.dimensions()[axis].clone())
    };

    let sharding = if let Some(output_sharding) = output_sharding {
        let output_rank = dimensions.lhs_batching_dimensions().len() + lhs_result.len() + rhs_result.len();
        if output_sharding.rank() != output_rank {
            return Err(TypeError::invalid(format!(
                "`{DOT_OPERATION_NAME}` output sharding rank ({}) does not match the output rank ({output_rank})",
                output_sharding.rank(),
            )));
        }
        if let Some(mesh) = mesh
            && output_sharding.mesh() != mesh
        {
            return Err(TypeError::invalid(format!(
                "`{DOT_OPERATION_NAME}` output sharding must use the same mesh as the operands"
            )));
        }
        let mut referenced_axes: Vec<&String> = output_sharding.unreduced_axes().iter().collect();
        referenced_axes.extend(output_sharding.reduced_axes());
        for dimension in output_sharding.dimensions() {
            if let ShardingDimension::Sharded(axis_names) = dimension {
                referenced_axes.extend(axis_names);
            }
        }
        if referenced_axes
            .iter()
            .any(|name| output_sharding.mesh().axis_type(name) == Some(MeshAxisType::Auto))
        {
            return Err(TypeError::invalid(format!(
                "`{DOT_OPERATION_NAME}` output sharding cannot reference auto mesh axes"
            )));
        }
        // A requested unreduced output is a deferred all-reduce: contracting over a dimension sharded across mesh axes
        // leaves each device holding only a partial product-sum over its shard of the contraction, whose true value is
        // the cross-device sum. Marking the output unreduced over exactly those axes defers that sum to a later op
        // instead of materializing it here, so the request is valid only when it names precisely the axes that shard
        // the (identically sharded) contracting dimensions (JAX's `_dot_general_unreduced_rule`).
        if !output_sharding.unreduced_axes().is_empty() {
            let lhs_contracting_spec: Vec<ShardingDimension> = dimensions
                .lhs_contracting_dimensions()
                .iter()
                .map(|axis| dimension_of(lhs_sharding, *axis))
                .collect();
            let rhs_contracting_spec: Vec<ShardingDimension> = dimensions
                .rhs_contracting_dimensions()
                .iter()
                .map(|axis| dimension_of(rhs_sharding, *axis))
                .collect();
            if lhs_contracting_spec != rhs_contracting_spec {
                return Err(TypeError::invalid(format!(
                    "`{DOT_OPERATION_NAME}` contracting dimensions must be sharded identically when the output sharding is unreduced"
                )));
            }
            let mut contracting_axes = BTreeSet::new();
            for dimension in &lhs_contracting_spec {
                if let ShardingDimension::Sharded(axis_names) = dimension {
                    contracting_axes.extend(axis_names.iter().cloned());
                }
            }
            if output_sharding.unreduced_axes() != &contracting_axes {
                return Err(TypeError::invalid(format!(
                    "`{DOT_OPERATION_NAME}` output sharding unreduced axes must equal the axes that shard the contracting dimensions"
                )));
            }
        }
        Some(output_sharding.clone())
    } else if let Some(mesh) = mesh {
        for (lhs_axis, rhs_axis) in
            dimensions.lhs_contracting_dimensions().iter().zip(dimensions.rhs_contracting_dimensions())
        {
            let left = dimension_of(lhs_sharding, *lhs_axis);
            let right = dimension_of(rhs_sharding, *rhs_axis);
            // Only Explicit-axis sharding of both contracting operands triggers the ambiguity/consistency errors;
            // Manual/Auto contracting shardings fall through (handled by `shard_map` / the compiler).
            let both_explicitly_sharded =
                dimension_has_explicit_axis(mesh, &left) && dimension_has_explicit_axis(mesh, &right);
            if both_explicitly_sharded {
                let (ShardingDimension::Sharded(left_axes), ShardingDimension::Sharded(right_axes)) = (&left, &right)
                else {
                    unreachable!("dimension_has_explicit_axis only returns true for sharded dimensions")
                };
                if left_axes != right_axes {
                    return Err(TypeError::invalid(format!(
                        "`{DOT_OPERATION_NAME}` contracting dimensions must have consistent shardings, but got {left} and {right}"
                    )));
                }
                return Err(TypeError::invalid(format!(
                    "`{DOT_OPERATION_NAME}` contracting dimensions are sharded, making the output sharding ambiguous; request an \
                         explicit output sharding (e.g., one with unreduced axes) to resolve it"
                )));
            }
            // A contracting dimension sharded on only one operand (or only over Manual/Auto axes) is allowed and its
            // sharding is dropped from the output, matching JAX (the partitioner inserts the necessary communication).
        }

        let mut placement =
            Vec::with_capacity(dimensions.lhs_batching_dimensions().len() + lhs_result.len() + rhs_result.len());
        for (lhs_axis, rhs_axis) in
            dimensions.lhs_batching_dimensions().iter().zip(dimensions.rhs_batching_dimensions())
        {
            let left = dimension_of(lhs_sharding, *lhs_axis);
            let right = dimension_of(rhs_sharding, *rhs_axis);
            let merged = match merge_batch_sharding_dimensions(&left, &right) {
                Some(merged) => merged,
                // A batch-dimension conflict is an error only when an Explicit axis is involved; a conflict purely over
                // Manual/Auto axes drops to `Replicated` and is left to `shard_map` / the compiler.
                None if dimension_has_explicit_axis(mesh, &left) || dimension_has_explicit_axis(mesh, &right) => {
                    return Err(TypeError::invalid(format!(
                        "`{DOT_OPERATION_NAME}` batching dimensions must have consistent shardings, but got {left} and {right}"
                    )));
                }
                None => ShardingDimension::Replicated,
            };
            placement.push(merged);
        }
        placement.extend(lhs_result.iter().map(|axis| dimension_of(lhs_sharding, *axis)));
        placement.extend(rhs_result.iter().map(|axis| dimension_of(rhs_sharding, *axis)));

        let merged_axes = |select: fn(&Sharding) -> &BTreeSet<String>| -> BTreeSet<String> {
            match (lhs_sharding, rhs_sharding) {
                (Some(left), Some(right)) => select(left).union(select(right)).cloned().collect(),
                (Some(left), None) => select(left).clone(),
                (None, Some(right)) => select(right).clone(),
                (None, None) => BTreeSet::new(),
            }
        };
        let reduced_axes = merged_axes(Sharding::reduced_axes);
        let varying_manual_axes = merged_axes(Sharding::varying_manual_axes);
        let sharding = Sharding::new(mesh.clone(), placement)
            .and_then(|output| output.with_reduced_axes(reduced_axes))
            .and_then(|output| output.with_varying_manual_axes(varying_manual_axes))
            .map_err(|error| {
                TypeError::invalid(format!("`{DOT_OPERATION_NAME}` output sharding construction failed: {error}"))
            })?;
        Some(sharding.without_auto_axes())
    } else {
        None
    };

    ArrayType::new(accumulation_type.unwrap_or(lhs.data_type()), Shape::new(output_dimensions))
        .with_sharding(sharding)
        .map_err(|error| TypeError::invalid(error.to_string()))
}
