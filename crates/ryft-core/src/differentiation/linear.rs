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
use crate::programs::operations::Operation;
use crate::programs::regions::{RegionInterface, RegionSlot};
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
    Executable,

    /// Reverse-only linear map staged for [`CustomVjpOperation`](crate::CustomVjpOperation). Exactly one region is
    /// attached (i.e., the `transpose` region, which is the user's `backward` program) because `custom_vjp` supplies
    /// a custom backward pass but no executable tangent program. The forward map `u ↦ Lᵣ(u)` therefore exists
    /// mathematically but has no region boundary to derive the operation interface from, so its input and output types
    /// are stored here explicitly. Interpreting this form is deliberately an error (i.e., the canonical reverse-only
    /// diagnostic). The call exists in a linearized program only so that reverse mode can transpose it by replaying the
    /// attached [`Region`](crate::Region). Refer to the documentation of [`LinearCallOperation`]'s for how this
    /// [`LinearCallInterface`] relates to [`Self::Executable`].
    Opaque {
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
/// neither differentiates nor accumulates cotangents for `r`. Its two region interfaces are therefore:
///
///   - `forward`: `(u, r) ↦ v = Lᵣ(u)`, and
///   - `transpose`: `(r, v̄) ↦ ū = Lᵣᵀ(v̄)`.
///
/// Operation operands are ordered as `[linear_inputs..., residuals...]`, while the transpose region receives
/// `[residuals..., output_cotangents...]`. [`residual_count`](Self::residual_count) separates the two operand
/// roles/ Every residual is consequently an ordinary typed Single Static Assignment (SSA) edge rather than
/// differentiation-only payload metadata. Partial evaluation can lift those values into the enclosing
/// [`Linearization`](crate::Linearization) residual environment, and partition-aware transposition
/// receives the same values as known operands in deterministic order.
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
/// The _executable_ form of this operation derives its complete interface from attached `forward` and `transpose`
/// [`Region`](crate::Region)s and can be interpreted, lowered, transposed, and differentiated again. The _opaque_
/// form is the deliberate exception: `custom_vjp` supplies only a reverse rule, so it stores the unavailable forward
/// input/output types and attaches only the transpose region. Attempting to execute or differentiate that reverse-only
/// map in forward mode is an error.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearCallOperation<T: DifferentiableType> {
    /// Number of trailing residual operands.
    residual_count: usize,

    /// Specifies whether this call has an executable forward program or represents an opaque custom
    /// Vector-Jacobian Product (VJP) tangent map.
    interface: LinearCallInterface<T>,
}

impl<T: DifferentiableType> LinearCallOperation<T> {
    /// Creates an _executable_ [`LinearCallOperation`] with two attached regions, `forward` and `transpose`.
    #[inline]
    pub fn new(residual_count: usize) -> Self {
        Self { residual_count, interface: LinearCallInterface::Executable }
    }

    /// Creates a reverse-only _opaque_ [`LinearCallOperation`] intended to be used by the custom
    /// Vector-Jacobian Product (VJP) operation.
    #[inline]
    pub fn opaque(residual_count: usize, input_types: Vec<T>, output_types: Vec<T>) -> Self {
        Self { residual_count, interface: LinearCallInterface::Opaque { input_types, output_types } }
    }

    /// Returns the number of trailing residual operands.
    #[inline]
    pub fn residual_count(&self) -> usize {
        self.residual_count
    }

    /// Returns `true` if this call has an executable forward [`Region`](crate::Region).
    #[inline]
    pub fn is_executable(&self) -> bool {
        matches!(self.interface, LinearCallInterface::Executable)
    }

    /// Splits the provided `input_types` into its leading linear types and trailing residual types.
    fn split_inputs<'a>(&self, input_types: &'a [T]) -> Result<(&'a [T], &'a [T]), TypeError> {
        Ok(input_types.split_at(input_types.len().checked_sub(self.residual_count).ok_or_else(|| {
            TypeError::invalid(format!(
                "linear call residual count {} exceeds input count {}",
                self.residual_count,
                input_types.len(),
            ))
        })?))
    }
}

// TODO(eaplatanios): Review from here onwards.

impl<T: DifferentiableType> Display for LinearCallOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl<T: DifferentiableType> Operation<T> for LinearCallOperation<T> {
    #[inline]
    fn name(&self) -> &'static str {
        if self.is_executable() { "linear_call" } else { "custom_vjp_tangent" }
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        if self.is_executable() {
            const { &[RegionSlot::rule("forward"), RegionSlot::rule("transpose")] }
        } else {
            const { &[RegionSlot::rule("transpose")] }
        }
    }

    fn infer_region_input_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<Option<Vec<T>>>, TypeError> {
        match &self.interface {
            LinearCallInterface::Executable => {
                check_count!("region", region_interfaces, 2, TypeError);
                let forward = &region_interfaces[0];
                let renaming = T::derive_identity_renaming(forward.input_types(), input_types)?;
                let output_types = forward
                    .output_types()
                    .iter()
                    .map(|r#type| r#type.rename_identities(&renaming))
                    .collect::<Result<Vec<_>, _>>()?;
                let (_, residual_types) = self.split_inputs(input_types)?;
                let transpose_input_types = residual_types
                    .iter()
                    .cloned()
                    .chain(output_types.iter().map(DifferentiableType::cotangent))
                    .collect();
                Ok(vec![Some(input_types.to_vec()), Some(transpose_input_types)])
            }
            LinearCallInterface::Opaque { output_types, .. } => {
                check_count!("region", region_interfaces, 1, TypeError);
                let (_, residual_types) = self.split_inputs(input_types)?;
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
            LinearCallInterface::Executable => {
                check_count!("region", region_interfaces, 2, TypeError);
                let forward = &region_interfaces[0];
                let transpose = &region_interfaces[1];
                check_types!(@same, "linear call forward input", [input_types, forward.input_types()]);
                let (linear_types, residual_types) = self.split_inputs(input_types)?;
                let expected_transpose_inputs = residual_types
                    .iter()
                    .cloned()
                    .chain(forward.output_types().iter().map(DifferentiableType::cotangent))
                    .collect::<Vec<_>>();
                let expected_transpose_outputs =
                    linear_types.iter().map(DifferentiableType::cotangent).collect::<Vec<_>>();
                check_types!(@same, "linear call transpose input", [
                    &expected_transpose_inputs,
                    transpose.input_types(),
                ]);
                check_types!(@same, "linear call transpose output", [
                    &expected_transpose_outputs,
                    transpose.output_types(),
                ]);
                Ok(forward.output_types().to_vec())
            }
            LinearCallInterface::Opaque { input_types: linear_types, output_types } => {
                check_count!("region", region_interfaces, 1, TypeError);
                let transpose = &region_interfaces[0];
                let (actual_linear_types, residual_types) = self.split_inputs(input_types)?;
                check_types!(@same, "opaque linear call input", [linear_types, actual_linear_types]);
                let expected_transpose_inputs = residual_types
                    .iter()
                    .cloned()
                    .chain(output_types.iter().map(DifferentiableType::cotangent))
                    .collect::<Vec<_>>();
                let expected_transpose_outputs =
                    linear_types.iter().map(DifferentiableType::cotangent).collect::<Vec<_>>();
                check_types!(@same, "opaque linear call transpose input", [
                    &expected_transpose_inputs,
                    transpose.input_types(),
                ]);
                check_types!(@same, "opaque linear call transpose output", [
                    &expected_transpose_outputs,
                    transpose.output_types(),
                ]);
                Ok(output_types.clone())
            }
        }
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<T::Identity>) -> Result<Self, TypeError> {
        Ok(Self {
            residual_count: self.residual_count,
            interface: match &self.interface {
                LinearCallInterface::Executable => LinearCallInterface::Executable,
                LinearCallInterface::Opaque { input_types, output_types } => LinearCallInterface::Opaque {
                    input_types: input_types.iter().map(|r#type| r#type.rename_identities(renaming)).collect::<Result<
                        Vec<_>,
                        _,
                    >>(
                    )?,
                    output_types: output_types
                        .iter()
                        .map(|r#type| r#type.rename_identities(renaming))
                        .collect::<Result<Vec<_>, _>>()?,
                },
            },
        })
    }
}

impl<C: Domain<Type: DifferentiableType>> InterpretableOperation<C> for LinearCallOperation<C::Type> {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        if !self.is_executable() {
            return Err(TypeError::invalid(
                "custom_vjp does not support forward-mode differentiation; use reverse mode (vjp, \
                 value_and_gradient, or jacobian_reverse) instead",
            )
            .into());
        }
        driver.interpret_region(context, 0, inputs.to_vec())
    }
}

impl<C: Context<Type: DifferentiableType>> PartiallyEvaluatableOperation<C> for LinearCallOperation<C::Type> where
    C::Operation: From<LinearCallOperation<C::Type>>
{
}

impl<C, P: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>> for LinearCallOperation<ArrayType>
where
    C: Context<Type = ArrayType>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        _context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        _inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        // Batching a linear call admits a principled rule (batch the attached forward and transpose regions and
        // replicate residual extents), which the Phase 6 extent-residual operation sweep owns; until then, `vmap`
        // over a linearized program containing an executable call reports this exact boundary.
        Err(BatchingError::UnsupportedOperation { message: format!("operation `{}` cannot be batched", self.name()) })
    }
}

impl<C> DifferentiableOperation<C> for LinearCallOperation<C::Type>
where
    C: Context<Type: DifferentiableType> + Zero<C::Value>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        if !self.is_executable() {
            return Err(ProgramError::UnsupportedOperation {
                message: "custom_vjp_tangent has no forward-mode (jvp) rule; custom_vjp is reverse-mode-only"
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

impl<V, O> TransposableOperation<V, O> for LinearCallOperation<V::Type>
where
    V: Value<Type: DifferentiableType>,
    O: Operation<V::Type> + From<ZeroOperation<V::Type>>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        let linear_count = inputs.len().checked_sub(self.residual_count).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "linear call residual count {} exceeds input count {}",
                self.residual_count,
                inputs.len(),
            ))
        })?;
        let residuals = inputs[linear_count..]
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
        // The transpose region follows the executable form's leading forward region (index 1) and is the opaque
        // form's only region (index 0).
        let transpose = driver.region(usize::from(self.is_executable()))?;
        let mut transpose_inputs = residuals;
        transpose_inputs
            .extend(outputs.iter().cloned().map(|output| output.materialize(context)).collect::<Result<Vec<_>, _>>()?);
        let input_cotangents = transpose.interpret_in_context(context, transpose_inputs)?;
        check_count!("output", input_cotangents, linear_count, ProgramError);
        // Classify each transpose output as structurally zero by inspecting its producing instruction in the source
        // region, so zero cotangents stay symbolic instead of accumulating. Outputs with no producing instruction
        // (forwarded region inputs and constants) conservatively classify as nonzero.
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
        let mut cotangents = inputs[..linear_count]
            .iter()
            .zip(input_cotangents.into_iter().zip(output_is_zero))
            .map(|(input, (cotangent, is_zero))| {
                if input.is_unknown() {
                    if is_zero { MaybeZero::Zero(cotangent.r#type().into_owned()) } else { MaybeZero::Value(cotangent) }
                } else {
                    MaybeZero::Zero(input.r#type().cotangent())
                }
            })
            .collect::<Vec<_>>();
        cotangents.extend(inputs[linear_count..].iter().map(|input| MaybeZero::Zero(input.r#type().cotangent())));
        Ok(cotangents)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::contexts::tests::{
        ProjectedMemberType, ProjectedMemberValue, ProjectedProgramType, ProjectedProgramValue,
    };
    use crate::operations::math::MulOperation;
    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::types::{ArrayType, DataType};

    use super::LinearCallOperation;

    #[test]
    fn test_linear_call_is_generic_over_a_third_composite_member() {
        type Operation = LinearCallOperation<ProjectedProgramType>;

        let linear_type = ProjectedProgramType::First(ProjectedMemberType);
        let residual_type = ProjectedProgramType::Third(ProjectedMemberType);
        let forward = {
            let mut builder = ProgramBuilder::<ProjectedProgramValue, Operation>::new();
            let linear = builder.add_input(linear_type.clone());
            builder.add_input(residual_type.clone());
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
        let linear = builder.add_input(linear_type);
        let residual = builder.add_input(residual_type);
        let output = builder
            .add_instruction(LinearCallOperation::new(1), vec![forward, transpose], vec![linear, residual])
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
        assert_eq!(program.interpret(vec![linear.clone(), residual]), Ok(vec![linear]));
        assert_eq!(program.instructions()[0].regions().len(), 2);
    }

    #[test]
    fn test_executable_linear_call_nested_jvp_differentiates_residual_parameters() {
        let r#type = ArrayType::scalar(DataType::F64);
        let forward = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let linear = builder.add_input(r#type.clone());
            let residual = builder.add_input(r#type.clone());
            let output = builder.add_instruction(MulOperation, Vec::new(), vec![linear, residual]).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
                .unwrap()
        };
        let transpose = forward.clone();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let forward = builder.import_region(forward.entry_region_ref());
        let transpose = builder.import_region(transpose.entry_region_ref());
        let linear = builder.add_input(r#type.clone());
        let residual = builder.add_input(r#type);
        let output = builder
            .add_instruction(LinearCallOperation::new(1), vec![forward, transpose], vec![linear, residual])
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
