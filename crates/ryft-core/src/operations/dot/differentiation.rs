use super::*;

/// Applies the product rule shared by dense and grouped bilinear dot operations.
///
/// When one factor of a tangent term uses the widened tangent representation, both factors are converted to the
/// output tangent element type before applying the operation.
fn bilinear_array_jvp<V, Apply>(
    lhs: &DifferentiationDual<V>,
    rhs: &DifferentiationDual<V>,
    apply: Apply,
) -> Result<DifferentiationDual<V>, DifferentiationError>
where
    V: Value<Type = ArrayType> + ConvertElementType + std::ops::Add<Output = V>,
    Apply: Fn(&V, &V) -> Result<V, DifferentiationError>,
{
    let primal = apply(lhs.primal(), rhs.primal())?;
    let tangent_type = primal.r#type().tangent()?;
    let convert_to_tangent_type = |value: &V| {
        if value.r#type().data_type() == tangent_type.data_type() {
            Ok(value.clone())
        } else {
            value.convert_element_type(tangent_type.data_type()).map_err(DifferentiationError::from)
        }
    };
    let apply_tangent = |lhs: &V, rhs: &V| {
        if lhs.r#type().data_type() == rhs.r#type().data_type() {
            apply(lhs, rhs)
        } else {
            apply(&convert_to_tangent_type(lhs)?, &convert_to_tangent_type(rhs)?)
        }
    };
    let lhs_term = lhs.tangent().as_value().map(|tangent| apply_tangent(tangent, rhs.primal())).transpose()?;
    let rhs_term = rhs.tangent().as_value().map(|tangent| apply_tangent(lhs.primal(), tangent)).transpose()?;
    let tangent = lhs_term
        .into_iter()
        .chain(rhs_term)
        .reduce(|lhs, rhs| lhs + rhs)
        .map_or_else(|| MaybeZero::Zero(tangent_type), MaybeZero::Value);
    DifferentiationDual::new(primal, tangent)
}

// Forward-mode differentiation applies the product rule. Each term holds the corresponding primal operand fixed on
// its original contracting side and stages an ordinary dot with the primal operation's dimensions, accumulation type,
// and requested output sharding.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for DotOperation
where
    C::Operation: From<DotOperation>,
    C::Value: ConvertElementType + Dot + std::ops::Add<Output = C::Value>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        let left = &inputs[0];
        let right = &inputs[1];
        let stage_dot = |left: &C::Value, right: &C::Value| match (self.accumulation_type(), self.output_sharding()) {
            (Some(accumulation_type), _) => {
                left.dot_with_accumulation_type(right, self.dimensions(), accumulation_type)
            }
            (None, Some(output_sharding)) => left.dot_with_output_sharding(right, self.dimensions(), output_sharding),
            (None, None) => left.dot(right, self.dimensions()),
        };
        Ok(vec![bilinear_array_jvp(left, right, |left, right| Ok(stage_dot(left, right)))?])
    }
}

// A generalized dot is bilinear rather than jointly linear, so a valid pullback has exactly one linear operand. The
// known operand selects the corresponding adjoint dimensions, and the result is pinned to the linear operand's
// cotangent sharding and element representation.
impl<
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType> + From<ConvertElementTypeOperation<ArrayType>> + From<DotOperation>,
> TransposableOperation<V, O> for DotOperation
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        match (inputs[0].is_unknown(), inputs[1].is_unknown()) {
            // Both operands linear is a bilinear product, which is not a linear map in both operands jointly and so
            // never appears in a valid pushforward.
            (true, true) => Err(ProgramError::UnsupportedOperation {
                message: format!("bilinear `{DOT_OPERATION_NAME}` with two linear operands cannot be transposed"),
            }
            .into()),
            // Exactly one operand is linear: stage the adjoint dot reading the known operand's value, and emit a
            // structural zero for the known operand. A zero output cotangent stays a structural zero.
            (left_is_linear, _) => {
                let (linear_index, known_index) = if left_is_linear { (0, 1) } else { (1, 0) };
                let linear_cotangent_type = inputs[linear_index].r#type().cotangent()?;
                let contribution = match &outputs[0] {
                    MaybeZero::Zero(_) => MaybeZero::Zero(linear_cotangent_type),
                    MaybeZero::Value(output_cotangent) => {
                        // The dispatch guarantees a `Known` operand carries its pullback value, so read it directly.
                        let known_value = inputs[known_index]
                            .as_known()
                            .expect("dispatch guarantees a known operand carries its pullback value");
                        let known_value = if known_value.r#type().data_type() == output_cotangent.r#type().data_type() {
                            known_value.clone()
                        } else {
                            known_value.convert_element_type(output_cotangent.r#type().data_type())?
                        };
                        let left_rank = inputs[0].r#type().rank();
                        let right_rank = inputs[1].r#type().rank();
                        let adjoint_output_sharding = inputs[linear_index].r#type().sharding().map(Sharding::cotangent);
                        let adjoint = if left_is_linear {
                            // Known RHS: linear LHS cotangent is `dot(cotangent, rhs; adjoint_right)`.
                            let dimensions = adjoint_dimensions_for_right_dot(self.dimensions(), right_rank, left_rank);
                            DotOperation::new(dimensions).with_output_sharding(adjoint_output_sharding)
                        } else {
                            // Known LHS: linear RHS cotangent is `dot(lhs, cotangent; adjoint_left)`.
                            let dimensions = adjoint_dimensions_for_left_dot(self.dimensions(), left_rank);
                            DotOperation::new(dimensions).with_output_sharding(adjoint_output_sharding)
                        };
                        let operands = if left_is_linear {
                            [output_cotangent.clone(), known_value]
                        } else {
                            [known_value, output_cotangent.clone()]
                        };
                        let mut outputs = context.stage_operation(adjoint, Vec::new(), &operands)?;
                        check_count!("output", outputs, 1, ProgramError);
                        let adjoint_value = outputs.remove(0);
                        // An accumulation-typed primal contracts its adjoint at the widened cotangent type; convert
                        // the result back to the linear operand's cotangent element type when the two differ.
                        let adjoint_value = if adjoint_value.r#type().data_type() == linear_cotangent_type.data_type() {
                            adjoint_value
                        } else {
                            adjoint_value.convert_element_type(linear_cotangent_type.data_type())?
                        };
                        MaybeZero::Value(adjoint_value)
                    }
                };
                let mut contributions = inputs
                    .iter()
                    .map(|input| {
                        let input_type = input.r#type();
                        Ok(MaybeZero::Zero(input_type.cotangent()?))
                    })
                    .collect::<Result<Vec<_>, DifferentiationError>>()?;
                contributions[linear_index] = contribution;
                Ok(contributions)
            }
        }
    }
}

// A grouped dot is linear in either data operand separately. Group sizes are integer metadata and therefore carry no
// tangent; the two product-rule terms reuse the same grouped-dot dimensions.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for RaggedDotOperation
where
    C::Operation: From<ConvertElementTypeOperation<ArrayType>> + From<RaggedDotOperation>,
    C::Value: ConvertElementType + RaggedDot + std::ops::Add<Output = C::Value>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 3, ProgramError);
        let lhs = &inputs[0];
        let rhs = &inputs[1];
        let group_sizes = inputs[2].primal();
        let apply = |lhs: &C::Value, rhs: &C::Value| {
            lhs.ragged_dot_general(rhs, group_sizes, self.dimensions()).map_err(DifferentiationError::from)
        };
        Ok(vec![bilinear_array_jvp(lhs, rhs, apply)?])
    }
}

// JAX defines grouped-dot transposition only for the non-contracting mode. Each data-operand adjoint is another
// grouped dot followed by the inverse of the axis order produced by its adjoint dimension numbers.
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for RaggedDotOperation
where
    O: Operation<Type = ArrayType>
        + From<ConvertElementTypeOperation<ArrayType>>
        + From<RaggedDotOperation>
        + From<crate::operations::manipulation::TransposeOperation>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 3, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        let mode = self.dimensions().mode(inputs[0].r#type().rank())?;
        if mode != RaggedDotMode::NonContracting {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("`{RAGGED_DOT_OPERATION_NAME}` transposition is unsupported in `{mode}` mode"),
            }
            .into());
        }
        let group_sizes = inputs[2].as_known().ok_or_else(|| ProgramError::UnsupportedOperation {
            message: format!("`{RAGGED_DOT_OPERATION_NAME}` group sizes must be known during transposition"),
        })?;
        let zero_contributions = || {
            inputs
                .iter()
                .map(|input| Ok(MaybeZero::Zero(input.r#type().cotangent()?)))
                .collect::<Result<Vec<_>, DifferentiationError>>()
        };
        let MaybeZero::Value(cotangent) = &outputs[0] else {
            return zero_contributions();
        };
        if inputs[0].is_unknown() && inputs[1].is_unknown() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "bilinear `{RAGGED_DOT_OPERATION_NAME}` with two linear operands cannot be transposed",
                ),
            }
            .into());
        }
        let mut contributions = zero_contributions()?;
        let (linear_index, dimensions, output_axes, mut operands) = if inputs[0].is_unknown() {
            let known_rhs = inputs[1].as_known().unwrap();
            let (dimensions, output_axes) = adjoint_ragged_dimensions_for_lhs(
                self.dimensions(),
                inputs[0].r#type().rank(),
                inputs[1].r#type().rank(),
            );
            (0, dimensions, output_axes, [cotangent.clone(), known_rhs.clone(), group_sizes.clone()])
        } else if inputs[1].is_unknown() {
            let known_lhs = inputs[0].as_known().unwrap();
            let (dimensions, output_axes) = adjoint_ragged_dimensions_for_rhs(
                self.dimensions(),
                inputs[0].r#type().rank(),
                inputs[1].r#type().rank(),
            );
            (1, dimensions, output_axes, [known_lhs.clone(), cotangent.clone(), group_sizes.clone()])
        } else {
            return Ok(contributions);
        };
        let linear_cotangent_type = inputs[linear_index].r#type().cotangent()?;
        for operand in &mut operands[..2] {
            if operand.r#type().data_type() != cotangent.r#type().data_type() {
                *operand = operand.convert_element_type(cotangent.r#type().data_type())?;
            }
        }
        let mut adjoint = context.stage_operation(RaggedDotOperation::new(dimensions), Vec::new(), &operands)?;
        check_count!("output", adjoint, 1, ProgramError);
        let adjoint = adjoint.remove(0);
        let mut permutation = vec![0; output_axes.len()];
        for (axis, output_axis) in output_axes.into_iter().enumerate() {
            permutation[output_axis] = axis;
        }
        let adjoint = adjoint.transpose(permutation)?;
        let adjoint = if adjoint.r#type().data_type() == linear_cotangent_type.data_type() {
            adjoint
        } else {
            adjoint.convert_element_type(linear_cotangent_type.data_type())?
        };
        contributions[linear_index] = MaybeZero::Value(adjoint);
        Ok(contributions)
    }
}
