//! Differentiation-specific operation families.
//!
//! This module owns custom JVP and VJP calls, residual-parameterized linear calls, and gradient barriers.
//! Differentiation algorithms and transform contexts live in [`crate::differentiation`], while the transform-wide
//! residual-zero protocol is owned separately by [`crate::differentiation::zeros`].

// TODO(eaplatanios): Review this module.

use crate::contexts::Context;
use crate::differentiation::{DifferentiableType, DifferentiationDual};
use crate::programs::{ProgramError, ReferenceBoundary, Type, TypeError, Typed};

pub mod custom_jvp;
pub mod custom_vjp;
pub mod linear_call;
pub mod stop_gradient;

pub use custom_jvp::{CUSTOM_JVP_OPERATION_NAME, CustomJvp, CustomJvpOperation, custom_jvp};
pub use custom_vjp::{CUSTOM_VJP_OPERATION_NAME, CustomVjp, CustomVjpOperation, custom_vjp};
pub use linear_call::LinearCallOperation;
pub use stop_gradient::{STOP_GRADIENT_OPERATION_NAME, StopGradient, StopGradientOperation, StopGradients};

/// Validates that the leading `non_differentiated_count` operand positions of a custom derivative call named `name` fit
/// within its `input_count` operands.
///
/// # Errors
///
/// Returns a [`TypeError`] when `non_differentiated_count` exceeds `input_count`.
pub(crate) fn validate_non_differentiated_count(
    name: &str,
    non_differentiated_count: usize,
    input_count: usize,
) -> Result<(), TypeError> {
    if non_differentiated_count > input_count {
        return Err(TypeError::invalid(format!(
            "{name} non-differentiated operand count {non_differentiated_count} exceeds input count {input_count}",
        )));
    }
    Ok(())
}

/// Validates the reference contract that the custom derivative operations share over one primal boundary. A
/// reference-typed input is accepted only in the leading `non_differentiated_count` positions, where it is plumbing
/// that every attached rule region receives unchanged: the rule interfaces define no tangent or cotangent slot for a
/// reference, so an active reference operand would have a derivative that no user-supplied rule can express. No
/// output may be a reference, because a rule region would then have to produce that output's tangent reference or
/// consume its cotangent reference, and a user-supplied rule can neither allocate nor receive one. Both operations
/// apply this contract during type inference, so it holds for every constructed call, and their forward-mode rules
/// apply it again to the replayed operands as defense in depth.
///
/// # Parameters
///
///   - `name`: Operation name used in diagnostics.
///   - `non_differentiated_count`: Number of leading inputs that parameterize the call without being differentiated.
///   - `input_types`: Primal input types in operand order.
///   - `output_types`: Primal output types in output order.
///
/// # Errors
///
/// Returns a [`TypeError`] naming the first reference-typed input in the differentiated segment, or otherwise the
/// first reference-typed output.
pub(crate) fn validate_custom_derivative_reference_boundary<T: Type>(
    name: &str,
    non_differentiated_count: usize,
    input_types: &[T],
    output_types: &[T],
) -> Result<(), TypeError> {
    if let Some((index, r#type)) = input_types
        .iter()
        .enumerate()
        .skip(non_differentiated_count)
        .find(|(_, r#type)| r#type.is_reference())
    {
        return Err(TypeError::invalid(format!(
            "{name} accepts reference inputs only in its leading non-differentiated segment; move input {index} of \
             type `{type}` before the differentiated inputs",
        )));
    }
    if let Some((index, r#type)) = output_types.iter().enumerate().find(|(_, r#type)| r#type.is_reference()) {
        return Err(TypeError::invalid(format!(
            "{name} cannot return a reference, but output {index} has type `{type}`",
        )));
    }
    Ok(())
}

/// Validates the operands before replaying a custom derivative rule. Rule regions bypass ordinary differentiation
/// dispatch, so replay checks the reference contract enforced during type inference and uses
/// [`ReferenceBoundary`](crate::programs::ReferenceBoundary) to reject aliased concrete or staged references.
/// Non-differentiated numeric operands must have zero tangents; non-differentiated references carry state whose tangent
/// the rule leaves untouched.
///
/// # Parameters
///
///   - `name`: Operation name used in diagnostics.
///   - `non_differentiated_count`: Number of leading operands that parameterize the call without being differentiated.
///   - `context`: Context the rule is replayed in, which resolves the operands.
///   - `inputs`: Dual operands of the replayed call, in operand order.
///   - `output_types`: Primal output types in output order.
///
/// # Errors
///
/// Returns the [`ProgramError`] of the first violated contract: the [`TypeError`] of
/// [`validate_custom_derivative_reference_boundary`], a reference boundary error, or an unsupported nonzero tangent.
pub(crate) fn validate_custom_derivative_replay<C: Context<Type: DifferentiableType>>(
    name: &str,
    non_differentiated_count: usize,
    context: &C,
    inputs: &[DifferentiationDual<C::Value>],
    output_types: &[C::Type],
) -> Result<(), ProgramError> {
    let primal_types = inputs.iter().map(|input| input.primal().r#type().into_owned()).collect::<Vec<_>>();
    validate_custom_derivative_reference_boundary(
        name,
        non_differentiated_count,
        primal_types.as_slice(),
        output_types,
    )?;
    ReferenceBoundary::new_for_differentiation(context, inputs.iter().map(DifferentiationDual::primal), [], [])?;
    if let Some(input) = inputs.iter().take(non_differentiated_count).find(|input| {
        !input.primal().r#type().is_reference()
            && !input.tangent().is_zero()
            && !input.tangent().r#type().is_zero_space()
    }) {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "{name} cannot propagate the nonzero tangent of type `{}` supplied for one of its \
                 {non_differentiated_count} leading non-differentiated operands, because its rule has no tangent slot \
                 for them",
                input.tangent().r#type(),
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;

    use pretty_assertions::assert_eq;

    use super::*;
    use crate::arrays::{Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayType, DataType};
    use crate::contexts::EagerContext;
    use crate::differentiation::{
        DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationError,
        DifferentiationPolicy, Linearization,
    };
    use crate::operations::references::{ReferenceNewOperation, ReferenceReadOperation};
    use crate::parameters::Placeholder;
    use crate::partial::PartitionedProgram;
    use crate::programs::{FlatProgram, Program, ProgramBuilder, ReferenceType, RegionDriver, RegionRef};
    use crate::tracing::Trace;

    /// Builds a reference-free program representing the identity function over `r#type`.
    pub(crate) fn array_ir_identity_program(
        r#type: &ArrayIrType,
    ) -> FlatProgram<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>> {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(r#type.clone());
        builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![input],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap()
    }

    /// Pairs an identity primal program with a custom JVP program that allocates and reads local reference state.
    pub(crate) fn custom_jvp_regions_with_reference_state(
        r#type: &ArrayIrType,
    ) -> Vec<FlatProgram<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>> {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(r#type.clone());
        let tangent = builder.add_input(r#type.clone());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let jvp_program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output, tangent],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        vec![array_ir_identity_program(r#type), jvp_program]
    }

    /// Builds a custom-derivative rule whose nested custom JVP closure contains local reference state.
    pub(crate) fn nested_custom_derivative_state_program(
        scalar_type: &ArrayIrType,
        include_tangent_output: bool,
    ) -> FlatProgram<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>> {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let regions = custom_jvp_regions_with_reference_state(scalar_type)
            .iter()
            .map(|region| builder.import_region(region.entry_region_ref()))
            .collect::<Vec<_>>();
        let input = builder.add_input(scalar_type.clone());
        let tangent = include_tangent_output.then(|| builder.add_input(scalar_type.clone()));
        let output = builder
            .add_instruction(CustomJvpOperation::<ArrayIrType>::new(), regions, vec![input], None)
            .unwrap()[0];
        let mut outputs = vec![output];
        if let Some(tangent) = tangent {
            outputs.push(tangent);
        }
        let input_count = usize::from(include_tangent_output) + 1;
        let output_count = outputs.len();
        builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                outputs,
                vec![Placeholder; input_count],
                vec![Placeholder; output_count],
            )
            .unwrap()
    }

    /// Supplies custom-derivative regions while making recursive differentiation an assertion failure: the custom
    /// derivative rules replay their rule regions directly and must never differentiate them recursively.
    pub(crate) struct ReferenceRuleDifferentiationDriver {
        pub(crate) programs: Vec<FlatProgram<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>>,
    }

    impl RegionDriver<ArrayIrValue<Array>, ArrayIrOperation<Array>> for ReferenceRuleDifferentiationDriver {
        fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, ArrayIrValue<Array>, ArrayIrOperation<Array>>>
        where
            ArrayIrValue<Array>: 'r,
            ArrayIrOperation<Array>: 'r,
        {
            self.programs.iter().map(Program::entry_region_ref)
        }
    }

    impl DifferentiationDriver<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>
        for ReferenceRuleDifferentiationDriver
    {
        fn jvp_program(
            &self,
            _region: RegionRef<'_, ArrayIrValue<Array>, ArrayIrOperation<Array>>,
            _activity: &[bool],
        ) -> Result<Arc<FlatProgram<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>>, DifferentiationError>
        {
            unreachable!("custom derivative rules replay their rule regions instead of differentiating them")
        }

        fn linearize_program(
            &self,
            _region: RegionRef<'_, ArrayIrValue<Array>, ArrayIrOperation<Array>>,
            _activity: &[bool],
        ) -> Result<Linearization<ArrayIrValue<Array>, ArrayIrOperation<Array>>, DifferentiationError> {
            unreachable!("custom derivative rules replay their rule regions instead of linearizing them")
        }

        fn partition_jvp_program(
            &self,
            region: RegionRef<'_, ArrayIrValue<Array>, ArrayIrOperation<Array>>,
            input_known: &[bool],
            required_known_outputs: &[usize],
        ) -> Result<PartitionedProgram<ArrayIrValue<Array>, ArrayIrOperation<Array>>, DifferentiationError> {
            Ok(region.partition_with_configuration(input_known, true, true, Some(required_known_outputs))?.0)
        }

        fn jvp_operation<P: DifferentiationPolicy<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>>(
            &self,
            _operation: &ArrayIrOperation<Array>,
            _programs: Vec<FlatProgram<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>>,
            _context: &DifferentiationContext<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>, P>,
            _inputs: &[DifferentiationDual<ArrayIrValue<Array>>],
        ) -> Result<Vec<DifferentiationDual<ArrayIrValue<Array>>>, DifferentiationError> {
            unreachable!("custom derivative rules replay their rule regions instead of differentiating them")
        }
    }

    #[test]
    fn test_validate_custom_derivative_replay() {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = DifferentiationDual::new(
            ArrayIrValue::Array(Array::scalar(3.0_f32)),
            ArrayIrValue::Array(Array::scalar(1.0_f32)),
        )
        .unwrap();
        assert!(matches!(
            validate_custom_derivative_replay("custom_jvp", 1, &context, &[input], &[]),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "custom_jvp cannot propagate the nonzero tangent of type `f32[]` supplied for one of \
                    its 1 leading non-differentiated operands, because its rule has no tangent slot for them",
        ));
    }

    #[test]
    fn test_validate_custom_derivative_replay_staged_references() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |inputs: Vec<_>| {
                let context = inputs[0].context();
                let mut duals = inputs
                    .iter()
                    .cloned()
                    .map(DifferentiationDual::new_with_zero_tangent)
                    .collect::<Result<Vec<_>, _>>()?;
                assert_eq!(validate_custom_derivative_replay("custom_jvp", 2, context, &duals, &[]), Ok(()));

                // Replaying the same valid boundary is allowed, including a fresh root allocated inside the trace.
                let local = context.bind(ReferenceNewOperation::new(), vec![], &inputs[2..])?.remove(0);
                duals[1] = DifferentiationDual::new_with_zero_tangent(local)?;
                assert_eq!(validate_custom_derivative_replay("custom_vjp", 2, context, &duals, &[]), Ok(()));
                assert_eq!(validate_custom_derivative_replay("custom_vjp", 2, context, &duals, &[]), Ok(()));

                // An ordinary staged operand must not hide two reference operands naming the same allocation.
                duals[1] = duals[0].clone();
                assert!(matches!(
                    validate_custom_derivative_replay("custom_jvp", 2, context, &duals, &[]),
                    Err(ProgramError::InvalidArgument { message })
                        if message == "input 1 and input 0 bind the same reference allocation",
                ));
                Ok(vec![inputs[2].clone()])
            },
            vec![reference_type.clone(), reference_type, scalar_type],
        )
        .unwrap();
    }
}
