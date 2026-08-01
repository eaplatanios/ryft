use std::fmt::Display;

use crate::batching::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, BatchableOperation, BatchingContext, BatchingDriver, BatchingError,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::DifferentiationError;
use crate::differentiation::forward::{DifferentiableOperation, DifferentiationDriver, DifferentiationDual};
use crate::differentiation::reverse::{TransposableOperation, TranspositionDriver};
use crate::differentiation::types::DifferentiableType;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, check_types};
use crate::operations::constants::{Zero, ZeroOperation};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::ProgramError;
use crate::programs::atoms::MaybeZero;
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::{OutputRegionProvenance, RegionInterface, RegionSlot};
use crate::programs::types::{TypeError, Typed};
use crate::programs::values::Value;
use crate::tracing::{Tracer, TracingContext};
use crate::types::ArrayType;

/// Interface form implemented by a [`LinearCallOperation`].
#[derive(Clone, Debug, PartialEq)]
enum LinearCallInterface<T: DifferentiableType> {
    /// Executable linear map. Both the map and its transpose have attached regions (i.e., `forward` followed by
    /// `transpose`), and the complete operation interface is derived from the `forward` [`Region`](crate::Region)'s
    /// boundary, so no [`Type`](crate::Type)s are stored.
    ForwardAndTranspose,

    /// Reverse-only linear map that supplies a transpose program but no executable forward program, so exactly one
    /// region is attached, named `transpose`. The forward map `u ↦ Lᵣ(u)` therefore exists mathematically but has
    /// no region boundary to derive the operation interface from, and its input and output types are stored here
    /// explicitly. Interpreting this form is deliberately an error (i.e., the canonical reverse-only diagnostic).
    /// The call exists in a linearized program only so that reverse mode can transpose it by replaying the attached
    /// [`Region`](crate::Region). For example, [`CustomVjpOperation`](crate::CustomVjpOperation) stages this form
    /// because `custom_vjp` supplies a user-written backward program without a tangent program. Refer to the
    /// documentation of [`LinearCallOperation`] for how this form relates to [`Self::ForwardAndTranspose`].
    TransposeOnly {
        /// Input [`Type`](crate::Type)s of the unavailable forward map (one per linear operand).
        input_types: Vec<T>,

        /// Output [`Type`](crate::Type)s of the unavailable forward map.
        output_types: Vec<T>,
    },
}

/// [`Operation`] that represents a call to a residual-parameterized linear map together with its transpose. For
/// fixed residual values `r`, this operation represents a linear map `Lᵣ : U → V` and its transpose `Lᵣᵀ : V* → U*`.
/// Linearity applies only to `u ∈ U`: for scalars `a` and `b`, `Lᵣ(a · u₁ + b · u₂) = a · Lᵣ(u₁) + b · Lᵣ(u₂)`. The
/// residuals `r` are fixed primal values that parameterize the map, so transposition produces cotangents for `u` but
/// neither differentiates nor accumulates cotangents for `r`. Its two region interfaces are deliberately symmetric:
///
///   - `forward`: `(r, u) ↦ v = Lᵣ(u)`, and
///   - `transpose`: `(r, v̄) ↦ ū = Lᵣᵀ(v̄)`.
///
/// Operation operands are ordered as `[residuals..., linear_inputs...]`, matching both region interfaces, and
/// [`residual_count`](Self::residual_count) separates the two operand roles. That symmetry makes transposition a
/// *swap*: the transpose of `Lᵣ` staged with regions `(forward, transpose)` is the same operation staged with the
/// regions `(transpose, forward)` over `[residuals..., output_cotangents...]`, so transposing twice restores the
/// original call and a pullback program retains the linear-call boundary (and its explicit residual edges) instead
/// of dissolving it. This mirrors the transpose rule of [JAX's
/// `jax.custom_derivatives.linear_call`](https://github.com/jax-ml/jax/blob/main/jax/_src/custom_derivatives.py),
/// which has no rendered documentation page and is therefore linked at its source. The swap re-derives each side's
/// expected interface with the cotangent type mapping, which requires `cotangent(cotangent(u)) = u` for the linear
/// types. Tangent types (the only types linearization rules stage as linear operands) satisfy this even where primal
/// storage types do not (e.g., `f8e8m0fnu`, whose tangent and cotangent representations are both `f32`).
///
/// Every residual is an ordinary typed Single Static Assignment (SSA) edge rather than differentiation-only payload
/// metadata. Partial evaluation can lift those values into the enclosing [`Linearization`](crate::Linearization)
/// residual environment, and partition-aware transposition receives the same values as known operands in
/// deterministic order.
///
/// An explicit operation boundary is necessary because a tangent program must retain more than the computation of
/// `Lᵣ(u)`. After linearization, that program may be cloned, imported, simplified, differentiated again, or transposed
/// independently of the primal program. Merely inlining the forward computation would lose its association with both
/// the residual values and the program that implements `Lᵣᵀ`. Attaching both regions to this operation makes that
/// association survive through the ordinary [`Program`](crate::Program) region, operand, identity-renaming, and import
/// machinery, without an ambient value lookup or side table.
///
/// The dynamic reshape operation illustrates the need for this representation. Its tangent map reshapes an input
/// tangent using the output extents, whereas its transpose reshapes an output cotangent using the original input
/// extents. Those input extents cannot always be recovered from the output shape (e.g., `[n, 4] → [2, 2·n]`). The
/// reshape rule therefore carries both sets of extents as explicit residual operands and attaches forward and inverse
/// reshape regions to one [`LinearCallOperation`]. Other shape-dependent linear rules can use the same mechanism; the
/// operation itself has no array- or dimension-specific semantics.
///
/// The _forward-and-transpose_ form of this operation derives its complete interface from attached `forward` and
/// `transpose` [`Region`](crate::Region)s and can be interpreted, lowered, transposed, and differentiated again. The
/// _transpose-only_ form is the deliberate exception for linear maps that supply only a reverse rule: it stores the
/// unavailable forward input/output types and attaches only the transpose region, so attempting to execute or
/// differentiate it in forward mode is an error (e.g., [`CustomVjpOperation`](crate::CustomVjpOperation) stages
/// this form because it supplies a backward program without a tangent program).
#[derive(Clone, Debug, PartialEq)]
pub struct LinearCallOperation<T: DifferentiableType> {
    /// Number of leading residual operands.
    residual_count: usize,

    /// Specifies whether this call has an executable forward program or is a reverse-only transpose-only map.
    interface: LinearCallInterface<T>,
}

impl<T: DifferentiableType> LinearCallOperation<T> {
    /// Creates a _forward-and-transpose_ [`LinearCallOperation`] with two attached regions,
    /// `forward` and `transpose`.
    #[inline]
    pub fn new(residual_count: usize) -> Self {
        Self { residual_count, interface: LinearCallInterface::ForwardAndTranspose }
    }

    /// Creates a reverse-only _transpose-only_ [`LinearCallOperation`] for a linear map that supplies a transpose
    /// program but no executable forward program, stating the unavailable forward map's interface explicitly.
    #[inline]
    pub fn transpose_only(residual_count: usize, input_types: Vec<T>, output_types: Vec<T>) -> Self {
        Self { residual_count, interface: LinearCallInterface::TransposeOnly { input_types, output_types } }
    }

    /// Returns the number of leading residual operands.
    #[inline]
    pub fn residual_count(&self) -> usize {
        self.residual_count
    }

    /// Returns `true` if this call has the _transpose-only_ form, which attaches no executable forward
    /// [`Region`](crate::Region).
    #[inline]
    pub fn is_transpose_only(&self) -> bool {
        matches!(self.interface, LinearCallInterface::TransposeOnly { .. })
    }

    /// Splits the provided `input_types` into its leading residual types and trailing linear types.
    fn split_inputs<'a>(&self, input_types: &'a [T]) -> Result<(&'a [T], &'a [T]), TypeError> {
        if self.residual_count > input_types.len() {
            return Err(TypeError::invalid(format!(
                "linear call residual count {} exceeds input count {}",
                self.residual_count,
                input_types.len(),
            )));
        }
        Ok(input_types.split_at(self.residual_count))
    }
}

impl<T: DifferentiableType> Display for LinearCallOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: DifferentiableType> Operation<T> for LinearCallOperation<T> {
    #[inline]
    fn name(&self) -> &'static str {
        // The two forms render under distinct names so rendered programs and the diagnostics built from this name
        // distinguish a reverse-only call from an executable one without inspecting region counts.
        if self.is_transpose_only() { "transpose_only_linear_call" } else { "linear_call" }
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        if self.is_transpose_only() {
            const { &[RegionSlot::rule("transpose")] }
        } else {
            const { &[RegionSlot::rule("forward"), RegionSlot::rule("transpose")] }
        }
    }

    fn infer_region_input_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<Option<Vec<T>>>, TypeError> {
        match &self.interface {
            LinearCallInterface::ForwardAndTranspose => {
                check_count!("region", region_interfaces, 2, TypeError);
                let forward = &region_interfaces[0];
                let renaming = T::derive_identity_renaming(forward.input_types(), input_types)?;
                let output_types = forward
                    .output_types()
                    .iter()
                    .map(|r#type| r#type.rename_identities(&renaming))
                    .collect::<Result<Vec<_>, _>>()?;
                let (residual_types, _) = self.split_inputs(input_types)?;
                let transpose_input_types = residual_types
                    .iter()
                    .cloned()
                    .chain(output_types.iter().map(DifferentiableType::cotangent))
                    .collect();
                Ok(vec![Some(input_types.to_vec()), Some(transpose_input_types)])
            }
            LinearCallInterface::TransposeOnly { output_types, .. } => {
                check_count!("region", region_interfaces, 1, TypeError);
                let (residual_types, _) = self.split_inputs(input_types)?;
                Ok(vec![Some(
                    residual_types
                        .iter()
                        .cloned()
                        .chain(output_types.iter().map(DifferentiableType::cotangent))
                        .collect(),
                )])
            }
        }
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        match &self.interface {
            LinearCallInterface::ForwardAndTranspose => {
                check_count!("region", region_interfaces, 2, TypeError);
                let forward = &region_interfaces[0];
                let transpose = &region_interfaces[1];
                check_types!(@same, "linear call forward input", [input_types, forward.input_types()]);
                let (residual_types, linear_types) = self.split_inputs(input_types)?;
                let transpose_input_types = residual_types
                    .iter()
                    .cloned()
                    .chain(forward.output_types().iter().map(DifferentiableType::cotangent))
                    .collect::<Vec<_>>();
                let transpose_output_types = linear_types.iter().map(DifferentiableType::cotangent).collect::<Vec<_>>();
                check_types!(@same, "linear call transpose input", [
                    &transpose_input_types,
                    transpose.input_types(),
                ]);
                check_types!(@same, "linear call transpose output", [
                    &transpose_output_types,
                    transpose.output_types(),
                ]);
                Ok(forward.output_types().to_vec())
            }
            LinearCallInterface::TransposeOnly { input_types: linear_types, output_types } => {
                check_count!("region", region_interfaces, 1, TypeError);
                let transpose = &region_interfaces[0];
                let (residual_types, actual_linear_types) = self.split_inputs(input_types)?;
                check_types!(@same, "transpose-only linear call input", [linear_types, actual_linear_types]);
                let transpose_input_types = residual_types
                    .iter()
                    .cloned()
                    .chain(output_types.iter().map(DifferentiableType::cotangent))
                    .collect::<Vec<_>>();
                let transpose_output_types = linear_types.iter().map(DifferentiableType::cotangent).collect::<Vec<_>>();
                check_types!(@same, "transpose-only linear call transpose input", [
                    &transpose_input_types,
                    transpose.input_types(),
                ]);
                check_types!(@same, "transpose-only linear call transpose output", [
                    &transpose_output_types,
                    transpose.output_types(),
                ]);
                Ok(output_types.clone())
            }
        }
    }

    #[inline]
    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        // The forward-and-transpose form's outputs are exactly its forward region's outputs.
        // The transpose-only form has no forward region and therefore no region provenance.
        if self.is_transpose_only() {
            Vec::new()
        } else {
            vec![OutputRegionProvenance { region_index: 0, output_index }]
        }
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("residual_count", self.residual_count))
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<T::Identity>) -> Result<Self, TypeError> {
        Ok(Self {
            residual_count: self.residual_count,
            interface: match &self.interface {
                LinearCallInterface::ForwardAndTranspose => LinearCallInterface::ForwardAndTranspose,
                LinearCallInterface::TransposeOnly { input_types, output_types } => {
                    LinearCallInterface::TransposeOnly {
                        input_types: input_types
                            .iter()
                            .map(|r#type| r#type.rename_identities(renaming))
                            .collect::<Result<Vec<_>, _>>()?,
                        output_types: output_types
                            .iter()
                            .map(|r#type| r#type.rename_identities(renaming))
                            .collect::<Result<Vec<_>, _>>()?,
                    }
                }
            },
        })
    }
}

impl<C: Domain<Type: DifferentiableType>> InterpretableOperation<C> for LinearCallOperation<C::Type> {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        if self.is_transpose_only() {
            return Err(ProgramError::UnsupportedOperation {
                message: "a transpose-only linear call has no forward program to execute; it supports only \
                          reverse-mode differentiation (e.g., 'vjp', 'value_and_gradient', or 'jacobian_reverse')"
                    .to_string(),
            });
        }
        driver.interpret_region(context, 0, inputs.to_vec())
    }
}

impl<C: Context<Type: DifferentiableType, Operation: From<LinearCallOperation<C::Type>>>>
    PartiallyEvaluatableOperation<C> for LinearCallOperation<C::Type>
{
}

impl<C: Context<Type = ArrayType>, P: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>>
    for LinearCallOperation<ArrayType>
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        _context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        _inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        // TODO(eaplatanios): Fix this in phase 6.
        // Batching a linear call admits a principled rule (batch the attached forward and transpose regions and
        // replicate residual extents), which the Phase 6 extent-residual operation sweep owns; until then, `vmap`
        // over a linearized program containing an executable call reports this exact boundary.
        Err(BatchingError::UnsupportedOperation { message: format!("operation `{}` cannot be batched", self.name()) })
    }
}

impl<C: Context<Type: DifferentiableType> + Zero<C::Value>> DifferentiableOperation<C>
    for LinearCallOperation<C::Type>
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        if self.is_transpose_only() {
            return Err(ProgramError::UnsupportedOperation {
                message: "a transpose-only linear call has no forward-mode (JVP) rule; it supports only \
                          reverse-mode differentiation"
                    .to_string(),
            }
            .into());
        }

        inputs.len().checked_sub(self.residual_count).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "linear call residual count {} exceeds input count {}",
                self.residual_count,
                inputs.len(),
            ))
        })?;

        let forward = driver.region(0)?;
        let primals = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        if inputs.iter().all(|input| input.tangent().is_zero()) {
            let primal_outputs = forward.interpret_in_context(context, primals)?;
            return Ok(primal_outputs.into_iter().map(DifferentiationDual::new_with_zero_tangent).collect());
        }

        // Higher-order differentiation must include the dependence of the linear map on its residual parameters.
        // Replay the ordinary fused JVP of the attached forward region instead of assuming residual tangents are zero.
        let jvp = driver.jvp_program(forward)?;
        let mut jvp_inputs = primals;
        for input in inputs {
            if !input.tangent().r#type().is_zero_space() {
                jvp_inputs.push(input.tangent().clone().materialize(context)?);
            }
        }

        let mut outputs = jvp.interpret_in_context(context, jvp_inputs)?;
        let output_types = forward.output_types();
        let tangent_outputs = outputs.split_off(output_types.len());
        let mut tangent_outputs = tangent_outputs.into_iter();
        outputs
            .into_iter()
            .zip(output_types)
            .map(|(primal, output_type)| {
                if output_type.tangent().is_zero_space() {
                    Ok(DifferentiationDual::new_with_zero_tangent(primal))
                } else {
                    DifferentiationDual::new(primal, tangent_outputs.next().unwrap())
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }
}

impl<V: Value<Type: DifferentiableType>, O> TransposableOperation<V, O> for LinearCallOperation<V::Type>
where
    O: Operation<V::Type> + From<ZeroOperation<V::Type>> + From<LinearCallOperation<V::Type>>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        if self.residual_count > inputs.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "linear call residual count {} exceeds input count {}",
                self.residual_count,
                inputs.len(),
            ))
            .into());
        }
        let (residual_inputs, linear_inputs) = inputs.split_at(self.residual_count);
        let residuals = residual_inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                input.as_known().cloned().ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "linear call residual operand {index} is not known during transposition",
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        if outputs.iter().all(MaybeZero::is_zero) {
            return Ok(inputs.iter().map(|input| MaybeZero::Zero(input.r#type().cotangent())).collect());
        }

        // Classify each transpose-region output as structurally zero by inspecting its producing instruction in the
        // source region, so zero cotangents stay symbolic instead of accumulating. Outputs with no producing
        // instruction (forwarded region inputs and constants) conservatively classify as nonzero. The transpose
        // region follows the forward-and-transpose form's leading forward region (index 1) and is the transpose-only
        // form's only region (index 0).
        let transpose = driver.region(if self.is_transpose_only() { 0 } else { 1 })?;
        let output_is_zero = transpose
            .output_ids()
            .iter()
            .map(|output| {
                transpose
                    .instructions()
                    .iter()
                    .find_map(|instruction| {
                        instruction
                            .outputs()
                            .iter()
                            .position(|candidate| candidate == output)
                            .map(|output_index| instruction.operation().is_zero(output_index))
                    })
                    .unwrap_or(false)
            })
            .collect::<Vec<_>>();

        let mut transpose_inputs = residuals;
        transpose_inputs
            .extend(outputs.iter().cloned().map(|output| output.materialize(context)).collect::<Result<Vec<_>, _>>()?);
        let input_cotangents = if self.is_transpose_only() {
            // The transpose-only form's region is a user-supplied backward program with no linearity contract of its
            // own, so it cannot be re-transposed and is replayed inline into the pullback.
            transpose.interpret_in_context(context, transpose_inputs)?
        } else {
            // The forward-and-transpose form transposes by *swapping* its regions: the pullback stages the same
            // operation with the transpose region leading, over the same residuals followed by the output
            // cotangents. Transposing twice therefore restores the original call, and the pullback retains the
            // linear-call boundary together with its explicit residual edges.
            let swapped = LinearCallOperation::new(self.residual_count);
            let transpose_program = transpose.to_program();
            let forward_program = driver.region(0)?.to_program();
            context.bind(swapped, vec![transpose_program, forward_program], &transpose_inputs)?
        };
        check_count!("output", input_cotangents, linear_inputs.len(), ProgramError);

        let mut cotangents =
            residual_inputs.iter().map(|input| MaybeZero::Zero(input.r#type().cotangent())).collect::<Vec<_>>();
        cotangents.extend(linear_inputs.iter().zip(input_cotangents.into_iter().zip(output_is_zero)).map(
            |(input, (cotangent, is_zero))| {
                if input.is_unknown() {
                    if is_zero { MaybeZero::Zero(cotangent.r#type().into_owned()) } else { MaybeZero::Value(cotangent) }
                } else {
                    MaybeZero::Zero(input.r#type().cotangent())
                }
            },
        ));
        Ok(cotangents)
    }
}

// TODO(eaplatanios): Review from here onwards.

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::contexts::Context;
    use crate::contexts::tests::{
        ProjectedMemberType, ProjectedMemberValue, ProjectedProgramType, ProjectedProgramValue,
    };
    use crate::operations::math::MulOperation;
    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::values::Value;
    use crate::types::{ArrayType, DataType};

    use super::LinearCallOperation;

    #[test]
    fn test_linear_call_is_generic_over_a_third_composite_member() {
        type Operation = LinearCallOperation<ProjectedProgramType>;

        let linear_type = ProjectedProgramType::First(ProjectedMemberType);
        let residual_type = ProjectedProgramType::Third(ProjectedMemberType);
        let forward = {
            let mut builder = ProgramBuilder::<ProjectedProgramValue, Operation>::new();
            builder.add_input(residual_type.clone());
            let linear = builder.add_input(linear_type.clone());
            builder
                .build::<Vec<ProjectedProgramValue>, Vec<ProjectedProgramValue>>(
                    vec![linear],
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let transpose = {
            let mut builder = ProgramBuilder::<ProjectedProgramValue, Operation>::new();
            builder.add_input(residual_type.clone());
            let cotangent = builder.add_input(linear_type.clone());
            builder
                .build::<Vec<ProjectedProgramValue>, Vec<ProjectedProgramValue>>(
                    vec![cotangent],
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<ProjectedProgramValue, Operation>::new();
        let forward = builder.import_region(forward.entry_region_ref());
        let transpose = builder.import_region(transpose.entry_region_ref());
        let residual = builder.add_input(residual_type);
        let linear = builder.add_input(linear_type);
        let output = builder
            .add_instruction(LinearCallOperation::new(1), vec![forward, transpose], vec![residual, linear])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ProjectedProgramValue>, Vec<ProjectedProgramValue>>(
                vec![output],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();

        let linear = ProjectedProgramValue::First(ProjectedMemberValue(2));
        let residual = ProjectedProgramValue::Third(ProjectedMemberValue(7));
        assert_eq!(program.interpret(vec![residual, linear.clone()]), Ok(vec![linear]));
        assert_eq!(program.instructions()[0].regions().len(), 2);
    }

    #[test]
    fn test_linear_call_transposition_swaps_the_attached_regions() {
        use crate::contexts::StagingContext;
        use crate::differentiation::DifferentiationError;
        use crate::differentiation::reverse::{TransposableOperation, TranspositionDriver};
        use crate::partial::PartialValue;
        use crate::programs::Program;
        use crate::programs::atoms::MaybeZero;
        use crate::programs::regions::{RegionDriver, RegionRef};
        use crate::tracing::TracingContext;

        /// Exposes the executable carrier's two regions to the transpose rule without entering full reverse mode.
        struct TwoRegionDriver<'r> {
            forward: RegionRef<'r, Array, ArrayOperation<Array>>,
            transpose: RegionRef<'r, Array, ArrayOperation<Array>>,
        }

        impl RegionDriver<Array, ArrayOperation<Array>> for TwoRegionDriver<'_> {
            fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, Array, ArrayOperation<Array>>>
            where
                Array: 'r,
                ArrayOperation<Array>: 'r,
            {
                [self.forward, self.transpose].into_iter()
            }
        }

        impl TranspositionDriver<Array, ArrayOperation<Array>> for TwoRegionDriver<'_> {
            fn transpose_program(
                &self,
                _region: RegionRef<'_, Array, ArrayOperation<Array>>,
                _input_linearity: &[bool],
            ) -> Result<Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>, DifferentiationError>
            {
                unreachable!("linear call transposition swaps regions and never re-enters transposition")
            }
        }

        // For `Lᵣ(u) = r · u`, transposing the call must stage the *swapped* linear call over
        // `[residual, output cotangent]` instead of inlining the transpose region's body, so the pullback retains
        // the linear-call boundary and its residual edge, and transposing twice restores the original call.
        let r#type = ArrayType::scalar(DataType::F64);
        let region = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let residual = builder.add_input(r#type.clone());
            let linear = builder.add_input(r#type.clone());
            let output = builder.add_instruction(MulOperation, Vec::new(), vec![linear, residual]).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
                .unwrap()
        };
        let transpose_region = region.clone();
        let driver =
            TwoRegionDriver { forward: region.entry_region_ref(), transpose: transpose_region.entry_region_ref() };
        let mut context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let residual = context.input(r#type.clone());
        let output_cotangent = context.input(r#type.clone());
        let cotangents = LinearCallOperation::new(1)
            .transpose(
                &mut context,
                &driver,
                &[PartialValue::Known(residual), PartialValue::Unknown(r#type.clone())],
                &[MaybeZero::Value(output_cotangent)],
            )
            .unwrap();

        // The residual receives a structural zero and the linear input receives the staged swapped call's output.
        assert_eq!(cotangents.len(), 2);
        assert!(cotangents[0].is_zero());
        assert!(matches!(cotangents[1], MaybeZero::Value(_)));

        // The staged pullback contains one swapped linear call whose leading region is the transpose program.
        let output_atom_id = match &cotangents[1] {
            MaybeZero::Value(value) => value.atom_id().unwrap(),
            MaybeZero::Zero(_) => unreachable!(),
        };
        let builder = context.builder().clone();
        drop((context, cotangents));
        let builder =
            std::rc::Rc::try_unwrap(builder).expect("the builder is uniquely owned once every tracer is dropped");
        let staged = builder
            .into_inner()
            .build::<Vec<Array>, Vec<Array>>(vec![output_atom_id], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let rendered = staged.to_string();
        assert!(rendered.contains("linear_call [residual_count=1]"), "{rendered}");
        assert_eq!(rendered.matches("= mul").count(), 2, "{rendered}");
    }

    #[test]
    fn test_executable_linear_call_nested_jvp_differentiates_residual_parameters() {
        let r#type = ArrayType::scalar(DataType::F64);
        let forward = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let residual = builder.add_input(r#type.clone());
            let linear = builder.add_input(r#type.clone());
            let output = builder.add_instruction(MulOperation, Vec::new(), vec![linear, residual]).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
                .unwrap()
        };
        let transpose = forward.clone();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let forward = builder.import_region(forward.entry_region_ref());
        let transpose = builder.import_region(transpose.entry_region_ref());
        let residual = builder.add_input(r#type.clone());
        let linear = builder.add_input(r#type);
        let output = builder
            .add_instruction(LinearCallOperation::new(1), vec![forward, transpose], vec![residual, linear])
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();

        assert_eq!(
            program.jvp().unwrap().interpret(vec![
                Array::scalar(2.0),
                Array::scalar(3.0),
                Array::scalar(5.0),
                Array::scalar(7.0),
            ]),
            Ok(vec![Array::scalar(6.0), Array::scalar(29.0)]),
        );
    }
}
