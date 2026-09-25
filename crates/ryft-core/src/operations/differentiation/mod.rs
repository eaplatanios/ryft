//! Differentiation-specific operation families.
//!
//! This module owns custom JVP and VJP calls, residual-parameterized linear calls, and gradient barriers.
//! Differentiation algorithms and transform contexts live in [`crate::differentiation`], while the transform-wide
//! residual-zero protocol is owned separately by [`crate::differentiation::zeros`].

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

    use crate::arrays::{Array, ArrayIrOperation, ArrayIrValue};
    use crate::contexts::EagerContext;
    use crate::differentiation::{
        DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationError,
        DifferentiationPolicy, Linearization,
    };
    use crate::parameters::Placeholder;
    use crate::partial::PartitionedProgram;
    use crate::programs::{FlatProgram, Operation, Program, ProgramBuilder, RegionDriver, RegionRef, Value};

    /// Builds one flat program that binds the custom-derivative `operation` with `regions` to inputs of `input_types`
    /// and returns all of its outputs.
    pub(crate) fn custom_derivative_call_program<V: Value, O: Clone + Operation<Type = V::Type>>(
        operation: impl Into<O>,
        regions: Vec<Program<V, O, Vec<V>, Vec<V>>>,
        input_types: Vec<V::Type>,
    ) -> Program<V, O, Vec<V>, Vec<V>> {
        let mut builder = ProgramBuilder::<V, O>::new();
        let regions = regions.into_iter().map(|region| builder.import_program(region)).collect::<Vec<_>>();
        let input_count = input_types.len();
        let inputs = input_types.into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let outputs = builder.add_instruction(operation, regions, inputs, None).unwrap().to_vec();
        let output_count = outputs.len();
        builder
            .build::<Vec<V>, Vec<V>>(outputs, vec![Placeholder; input_count], vec![Placeholder; output_count])
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
            _input_indices: &[usize],
        ) -> Result<Arc<FlatProgram<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>>, DifferentiationError>
        {
            unreachable!("custom derivative rules replay their rule regions instead of differentiating them")
        }

        fn linearize_program(
            &self,
            _region: RegionRef<'_, ArrayIrValue<Array>, ArrayIrOperation<Array>>,
            _input_indices: &[usize],
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

        fn bind_jvp_operation<P: DifferentiationPolicy<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>>(
            &self,
            _context: &DifferentiationContext<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>, P>,
            _operation: &ArrayIrOperation<Array>,
            _programs: Vec<FlatProgram<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>>,
            _inputs: &[DifferentiationDual<ArrayIrValue<Array>>],
        ) -> Result<Vec<DifferentiationDual<ArrayIrValue<Array>>>, DifferentiationError> {
            unreachable!("custom derivative rules replay their rule regions instead of differentiating them")
        }
    }
}
