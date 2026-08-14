use super::*;

/// Forward-mode rule for [`DotOperation`]: the product rule for the contraction
/// `d(dot(a, b)) = dot(da, b) + dot(a, db)`. Each term holds the corresponding primal operand fixed on its original
/// contracting side, staged as an ordinary `Dot` whose dimension numbers, accumulation type, and requested output
/// sharding match the primal, so the tangent dots match the primal dot exactly and stay capture-free. For an
/// accumulation-typed dot the tangent terms stay accumulation-typed dots over the operand-typed tangents whenever a
/// term's operand element types agree (the common case, because every low-precision floating-point type except
/// `f8e8m0fnu` is its own tangent representation), so the output tangent lives at the accumulation type exactly like
/// the primal output; when a tangent arrives at a widened representation instead, both term operands are converted
/// to the output tangent element type first.
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
        let primal = stage_dot(left.primal(), right.primal());
        let tangent_type = primal.r#type().tangent();
        let convert_to_tangent_type = |value: &C::Value| {
            if value.r#type().data_type() == tangent_type.data_type() {
                Ok(value.clone())
            } else {
                value.convert_element_type(tangent_type.data_type()).map_err(DifferentiationError::from)
            }
        };
        // Each term's dot needs equal operand element types: matching pairs (including operand-typed tangents of an
        // accumulation-typed dot) stage directly, while a widened tangent representation pulls both operands up to
        // the output tangent element type.
        let stage_tangent_dot = |left: &C::Value, right: &C::Value| -> Result<C::Value, DifferentiationError> {
            if left.r#type().data_type() == right.r#type().data_type() {
                Ok(stage_dot(left, right))
            } else {
                Ok(stage_dot(&convert_to_tangent_type(left)?, &convert_to_tangent_type(right)?))
            }
        };
        let left_term =
            left.tangent().as_value().map(|tangent| stage_tangent_dot(tangent, right.primal())).transpose()?;
        let right_term =
            right.tangent().as_value().map(|tangent| stage_tangent_dot(left.primal(), tangent)).transpose()?;
        // Combine the surviving terms, falling back to a structural zero of the primal's type when both were dropped.
        let tangent = left_term
            .into_iter()
            .chain(right_term)
            .reduce(|left_term, right_term| left_term + right_term)
            .map_or_else(|| MaybeZero::Zero(tangent_type), MaybeZero::Value);
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

/// Partition-aware transpose rule for the primal [`DotOperation`]. A generalized dot is bilinear: it is linear in
/// each operand separately but not in both jointly, so in a valid pushforward exactly one operand is linear and the
/// other is a known runtime value. The known operand selects which adjoint the linear operand receives, reproducing
/// captured-factor transpose rules without first folding the known
/// operand into a captured factor: the known operand's value is read from the pullback through `operand_values` and
/// fed back into a primal `dot` with the adjoint dimension numbers.
///
///   - When the RHS operand is known, the forward map is `t ↦ dot(t, rhs; dimensions)` — the linear form modeled by
///     a right-factor dot — whose adjoint maps the output cotangent to `dot(cotangent, rhs; adjoint)` with
///     `adjoint = adjoint_dimensions_for_right_dot(dimensions, rhs_rank, lhs_rank)`. The LHS (linear) operand receives
///     that contribution and the RHS (known) operand receives a structural zero.
///   - When the LHS operand is known, the forward map is `t ↦ dot(lhs, t; dimensions)` — the linear form modeled by
///     a left-factor dot — whose adjoint maps the output cotangent to `dot(lhs, cotangent; adjoint)` with
///     `adjoint = adjoint_dimensions_for_left_dot(dimensions, lhs_rank)`. The RHS (linear) operand receives that
///     contribution and the LHS (known) operand receives a structural zero.
///
/// The adjoint dot's output sharding is pinned to the cotangent dual of the linear operand's sharding, matching the
/// captured-factor rules: the produced value *is* that operand's cotangent, so its sharding swaps the operand's
/// unreduced and reduced axes instead of being re-derived. A zero output cotangent stays a structural zero, and two
/// linear operands (a bilinear product that is not a linear map jointly) are rejected as unsupported.
///
/// For an accumulation-typed dot the output cotangent arrives at the accumulation type's cotangent representation,
/// the known operand is converted up to it, and the adjoint contraction runs at that widened type; the result is
/// then converted to the linear operand's cotangent element type (e.g., back down to `f8e4m3fn` for an
/// `f8 × f8 → f32` dot) so the produced cotangent matches the operand's cotangent representation exactly.
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
                message: "bilinear `dot` with two linear operands cannot be transposed".to_string(),
            }
            .into()),
            // Exactly one operand is linear: stage the adjoint dot reading the known operand's value, and emit a
            // structural zero for the known operand. A zero output cotangent stays a structural zero.
            (left_is_linear, _) => {
                let (linear_index, known_index) = if left_is_linear { (0, 1) } else { (1, 0) };
                let linear_cotangent_type = inputs[linear_index].r#type().cotangent();
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
                        MaybeZero::Zero(input_type.cotangent())
                    })
                    .collect::<Vec<_>>();
                contributions[linear_index] = contribution;
                Ok(contributions)
            }
        }
    }
}
