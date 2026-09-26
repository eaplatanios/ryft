//! Operations that control how values are differentiated rather than what they compute. Each operation is defined by
//! an [`Operation`](crate::Operation) type (e.g., [`CustomJvpOperation`]) together with a user-facing function or
//! value capability trait (e.g., [`custom_jvp`](fn@custom_jvp) or [`StopGradient`]) that applies it to eager
//! [`Array`](crate::Array)s and traced values alike, so the same code executes immediately or records into a
//! program depending on the value it runs on. The operations fall into three groups:
//!
//!   - **Custom Derivative Rules:** [`custom_jvp`](fn@custom_jvp) pairs a function with a handwritten
//!     Jacobian-Vector Product (JVP) rule that governs both forward- and reverse-mode differentiation, and
//!     [`custom_vjp`](fn@custom_vjp) pairs a function with handwritten forward and backward rules that govern
//!     reverse-mode differentiation only. They stage a [`CustomJvpOperation`] and a [`CustomVjpOperation`],
//!     respectively, which carry the primal program and the rule programs as attached regions. [`custom_derivative_at`]
//!     stages either kind of rule at a known input, which lets the rule closures infer all of their parameter types
//!     from that input.
//!   - **Linear Maps:** [`LinearCallOperation`] calls a residual-parameterized linear map together with its transpose,
//!     which lets differentiation rules keep a linear map and its handwritten transpose together in tangent programs
//!     (e.g., for shape-dependent maps such as dynamic reshapes, or for the pullback of a [`custom_vjp`](fn@custom_vjp)
//!     call).
//!   - **Gradient Barriers:** [`StopGradient`] and [`StopGradients`] return values unchanged while replacing their
//!     tangents with structural zeros, so that no derivative flows through them.
//!
//! Outside of differentiation, all of these operations are transparent. That is, interpretation and backend lowering
//! replay the primal program of a custom derivative rule, execute the forward map of a linear call when it has one,
//! and pass the inputs of a gradient barrier through unchanged. Under differentiation, a custom derivative call replays
//! its rule programs instead of differentiating its primal program, and a gradient barrier produces zero tangents.
//! Refer to the documentation of [`custom_jvp`](fn@custom_jvp) and [`custom_vjp`](fn@custom_vjp) for how custom
//! derivative rules interact with references, batching, and partial evaluation. The differentiation transforms
//! themselves live in the [`differentiation`](crate::differentiation) module.
//!
//! # Examples
//!
//! ```rust
//! # use ryft_core::{Array, Cos, ProgramError, Sin, StopGradient, custom_derivative_at, differentiate_at};
//! # fn main() -> Result<(), ProgramError> {
//! // A custom JVP rule for `sin` that doubles the true derivative, so that its effect is visible.
//! let (value, tangent) = differentiate_at(Array::scalar(0.5f64)?).jvp(Array::scalar(1.0f64)?, |x| {
//!     custom_derivative_at(x).jvp(
//!         |x| Ok(x.sin()?),
//!         |x, tangent| {
//!             let tangent = x.cos()? * tangent;
//!             Ok((x.sin()?, tangent.clone() + tangent))
//!         },
//!     )
//! })?;
//! assert_eq!(value, Array::scalar(0.5f64.sin())?);
//! assert_eq!(tangent, Array::scalar(2.0 * 0.5f64.cos())?);
//!
//! // A gradient barrier treats its input as a constant, so the gradient of `x * stop_gradient(x)` is `x`.
//! let gradient = differentiate_at(Array::scalar(3.0f64)?).gradient(|x| x.clone() * x.stop_gradient().unwrap())?;
//! assert_eq!(gradient, Array::scalar(3.0f64)?);
//! # Ok(())
//! # }
//! ```

pub mod custom_derivatives;
pub mod custom_jvp;
pub mod custom_vjp;
pub mod linear_call;
pub mod stop_gradient;

pub use custom_derivatives::{CustomDerivativeBuilder, custom_derivative_at};
pub use custom_jvp::{CUSTOM_JVP_OPERATION_NAME, CustomJvp, CustomJvpOperation, custom_jvp};
pub use custom_vjp::{CUSTOM_VJP_OPERATION_NAME, CustomVjp, CustomVjpOperation, custom_vjp};
pub use linear_call::{LINEAR_CALL_OPERATION_NAME, LinearCallOperation};
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
