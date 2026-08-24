//! Differentiation-specific operation families.
//!
//! This module owns custom JVP and VJP calls, residual-parameterized linear calls, and gradient barriers.
//! Differentiation algorithms and transform contexts remain in the parent module, while the transform-wide
//! residual-zero protocol is owned separately by `differentiation::zeros`.

// TODO(eaplatanios): Review this module.

pub mod custom_jvp;
pub mod custom_vjp;
pub mod linear_call;
pub mod stop_gradient;

pub use custom_jvp::{CUSTOM_JVP_OPERATION_NAME, CustomJvp, CustomJvpOperation, custom_jvp};
pub use custom_vjp::{CUSTOM_VJP_OPERATION_NAME, CustomVjp, CustomVjpOperation, custom_vjp};
pub use linear_call::LinearCallOperation;
pub use stop_gradient::{STOP_GRADIENT_OPERATION_NAME, StopGradient, StopGradientOperation, StopGradients};

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;

    use crate::arrays::{Array, ArrayIrOperation, ArrayIrType, ArrayIrValue};
    use crate::contexts::EagerContext;
    use crate::differentiation::DifferentiationError;
    use crate::differentiation::forward::{DifferentiationDriver, DifferentiationDual, Linearization};
    use crate::parameters::Placeholder;
    use crate::programs::{
        FlatProgram, NewReferenceOperation, Program, ProgramBuilder, ReferenceReadOperation, RegionDriver, RegionRef,
    };

    use super::*;

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

    /// Pairs an identity primal program with a custom JVP program that creates unresolved reference state.
    pub(crate) fn custom_jvp_regions_with_reference_state(
        r#type: &ArrayIrType,
    ) -> Vec<FlatProgram<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>> {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(r#type.clone());
        let tangent = builder.add_input(r#type.clone());
        let reference =
            builder.add_instruction(NewReferenceOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
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

    /// Builds a custom-derivative rule whose nested custom JVP closure contains unresolved reference state.
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

    /// Supplies custom-derivative regions while making recursive differentiation an assertion failure: the
    /// operation-local state guard must reject the fixture first.
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
        ) -> Result<Arc<FlatProgram<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>>, DifferentiationError>
        {
            unreachable!("the operation-local state guard must reject before recursive differentiation")
        }

        fn linearize_program(
            &self,
            _region: RegionRef<'_, ArrayIrValue<Array>, ArrayIrOperation<Array>>,
        ) -> Result<Linearization<ArrayIrValue<Array>, ArrayIrOperation<Array>>, DifferentiationError> {
            unreachable!("the operation-local state guard must reject before recursive linearization")
        }

        fn jvp_operation(
            &self,
            _operation: &ArrayIrOperation<Array>,
            _programs: Vec<FlatProgram<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>>,
            _context: &EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
            _inputs: &[DifferentiationDual<ArrayIrValue<Array>>],
        ) -> Result<Vec<DifferentiationDual<ArrayIrValue<Array>>>, DifferentiationError> {
            unreachable!("the operation-local state guard must reject before recursive differentiation")
        }
    }
}
