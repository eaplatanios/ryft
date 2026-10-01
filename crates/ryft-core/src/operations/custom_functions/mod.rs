//! Functions with custom transform rules. A custom function pairs a primal function with handwritten rules for the
//! transforms whose automatic treatment of that primal is inadequate (e.g., a fused or foreign kernel, a numerically
//! stabilized derivative, or a batching strategy that structural batching cannot discover). Every transform without a
//! configured rule treats the primal as it treats any other function, except that a function with only reverse-mode
//! rules rejects forward mode. This module has three layers:
//!
//!   - **Functions:** [`custom_function`] creates a [`CustomFunction`] from its primal, and configuration functions
//!     attach its rules: a Jacobian-Vector Product (JVP) rule that governs both forward- and reverse-mode
//!     differentiation ([`CustomFunction::with_jvp`], the analogue of JAX's `jax.custom_jvp`), forward and backward
//!     rules that govern reverse-mode differentiation ([`CustomFunction::with_vjp`], the analogue of JAX's
//!     `jax.custom_vjp`), or both, as well as a batching rule ([`CustomFunction::with_batching`], the analogue of
//!     JAX's `jax.custom_batching.custom_vmap`). [`CustomFunction::from_custom_call`] creates a function whose primal
//!     calls a foreign kernel.
//!   - **Operations:** each call of a custom function stages a [`CustomFunctionOperation`], which carries the primal
//!     program as an attached region and retains the rules as callbacks that are traced on the first request of each
//!     specialization. Callers whose rules are already programs (e.g., kernel staging with explicit rule programs)
//!     stage the same operation with the rule programs attached as regions instead. Reverse mode replaces either kind
//!     of call with a [`CustomFunctionTransposeOperation`] carrier whose transposition applies the backward rule.
//!   - **Rules:** [`CustomRuleDefinition`] and its [`CustomRuleRegistration`] are the flat, family-level rule sets
//!     behind the retained rules, which trace and cache the rule programs of each specialization.
//!
//! Outside of the transforms that their rules govern, custom functions are transparent: interpretation and backend
//! lowering replay the primal program. Refer to the documentation of [`custom_function`] for the calling convention
//! and for how the rules interact with references, batching, and partial evaluation.
//!
//! # Examples
//!
//! ```rust
//! # use ryft_core::{
//! #     Array, ArrayOperation, Cos, DomainTracer, EagerContext, ProgramError, Sin, custom_function, differentiate_at,
//! # };
//! # fn main() -> Result<(), ProgramError> {
//! type Tracer = DomainTracer<EagerContext<Array, ArrayOperation<Array>>>;
//!
//! // A custom JVP rule for `sin` that doubles the true derivative, so that its effect is visible.
//! let sine = custom_function(|x: Tracer| Ok(x.sin()?)).with_jvp(|x, tangent| {
//!     let tangent = x.cos()? * tangent;
//!     Ok((x.sin()?, tangent.clone() + tangent))
//! });
//! let (value, tangent) = differentiate_at(Array::scalar(0.5f64)?).jvp(Array::scalar(1.0f64)?, |x| sine.call(x))?;
//! assert_eq!(value, Array::scalar(0.5f64.sin())?);
//! assert_eq!(tangent, Array::scalar(2.0 * 0.5f64.cos())?);
//! # Ok(())
//! # }
//! ```

// TODO(eaplatanios): Review this module.

pub mod functions;
pub mod operations;
pub mod rules;

pub use functions::{
    CustomCallPrimal, CustomFunction, CustomFunctionBatching, CustomFunctionJvp, CustomFunctionVjp, DefaultBatching,
    DefaultJvp, DefaultVjp, JvpFromPrimal, WithAccumulatingVjp, WithBatching, WithJvp, WithSymbolicZeroJvp,
    WithSymbolicZeroVjp, WithVjp, custom_function,
};
pub use operations::{
    CUSTOM_FUNCTION_OPERATION_NAME, CUSTOM_FUNCTION_TRANSPOSE_OPERATION_NAME, CustomFunctionJvpRule,
    CustomFunctionOperation, CustomFunctionTransposeOperation,
};
pub use rules::{
    CustomRuleDefinition, CustomRuleReference, CustomRuleRegistration, CustomRuleSource, CustomRuleSpecializer,
    CustomRuleTracer, LiftedCustomRules, UnavailableCustomRules, WeakCustomRuleRegistration,
};

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::arrays::{Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, ArrayType};
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::{
        DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationError,
        DifferentiationPolicy, ForwardModeDifferentiate, Linearization,
    };
    use crate::operations::arithmetic::MulOperation;
    use crate::operations::custom_functions::operations::CustomFunctionOperation;
    use crate::operations::custom_functions::rules::{CustomRuleDefinition, CustomRuleRegistration};
    use crate::parameters::Placeholder;
    use crate::partial::PartitionedProgram;
    use crate::programs::{FlatProgram, Operation, Program, ProgramBuilder, RegionDriver, RegionRef, Typed, Value};
    use crate::tests::TestArrayOperation;

    /// Builds one flat program that binds the custom function `operation` with `regions` to inputs of `input_types`
    /// and returns all of its outputs.
    pub(crate) fn custom_function_call_program<V: Value, O: Clone + Operation<Type = V::Type>>(
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

    /// Supplies custom-function regions while making recursive differentiation an assertion failure: the custom
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
            Ok(region.partition_with_configuration(input_known, true, true, Some(required_known_outputs), None)?.0)
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

    // Fixtures of the tests of calls and carriers with retained rules.

    /// Eager context whose operation family registers the retained-rule operations.
    pub(crate) type TestContext = EagerContext<Array, TestArrayOperation>;

    /// Retained-rule definition over [`TestContext`].
    pub(crate) type TestDefinition = CustomRuleDefinition<Array, TestArrayOperation>;

    /// Registration of a [`TestDefinition`].
    pub(crate) type TestRegistration = CustomRuleRegistration<Array, TestArrayOperation>;

    /// Retained-rule definition over the production [`ArrayOperation`] member family.
    pub(crate) type MemberDefinition = CustomRuleDefinition<Array, ArrayOperation<Array>>;

    /// Invocation counters of the retained rules of a [`cube_definition`].
    #[derive(Default)]
    pub(crate) struct RuleCounters {
        /// Number of JVP rule invocations.
        pub(crate) jvp: AtomicUsize,

        /// Number of VJP forward rule invocations.
        pub(crate) forward: AtomicUsize,

        /// Number of VJP backward rule invocations.
        pub(crate) backward: AtomicUsize,
    }

    impl RuleCounters {
        /// Returns the JVP, forward, and backward invocation counts.
        pub(crate) fn counts(&self) -> (usize, usize, usize) {
            (self.jvp.load(Ordering::SeqCst), self.forward.load(Ordering::SeqCst), self.backward.load(Ordering::SeqCst))
        }
    }

    /// Builds the primal `f(x) = x³`.
    pub(crate) fn cube_program(r#type: &ArrayType) -> FlatProgram<TestContext> {
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(r#type.clone());
        let square = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
        let cube = builder.add_instruction(MulOperation::new(), Vec::new(), vec![square, input], None).unwrap()[0];
        builder.build(vec![cube], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Returns a definition of `f(x) = x³` whose deliberately wrong rules distinguish the selected rule: the JVP rule
    /// computes `ẏ = x² ẋ` and the VJP rule computes `x̄ = 2 x² ȳ`, while the true derivative is `3 x²`. Each rule
    /// counts its invocations in `counters`.
    pub(crate) fn cube_definition(counters: &Arc<RuleCounters>, jvp: bool, vjp: bool) -> TestDefinition {
        let mut definition = TestDefinition::new("cube").with_batching();
        if jvp {
            let counters = counters.clone();
            definition = definition.with_jvp(move |primals, tangents| {
                counters.jvp.fetch_add(1, Ordering::SeqCst);
                let square = primals[0].clone() * primals[0].clone();
                Ok((vec![square.clone() * primals[0].clone()], vec![square * tangents[0].clone()]))
            });
        }
        if vjp {
            let forward_counters = counters.clone();
            let backward_counters = counters.clone();
            definition = definition.with_vjp(
                move |primals| {
                    forward_counters.forward.fetch_add(1, Ordering::SeqCst);
                    let square = primals[0].clone() * primals[0].clone();
                    Ok((vec![square.clone() * primals[0].clone()], vec![square]))
                },
                move |leading_inputs, seeds| {
                    backward_counters.backward.fetch_add(1, Ordering::SeqCst);
                    let contribution = leading_inputs[0].clone() * seeds[0].clone();
                    Ok(vec![contribution.clone() + contribution])
                },
            );
        }
        definition
    }

    /// Returns the member-family counterpart of [`cube_definition`], whose rules are written against member tracers.
    pub(crate) fn member_cube_definition(counters: &Arc<RuleCounters>) -> MemberDefinition {
        let (jvp_counters, forward_counters, backward_counters) =
            (counters.clone(), counters.clone(), counters.clone());
        MemberDefinition::new("cube")
            .with_jvp(move |primals, tangents| {
                jvp_counters.jvp.fetch_add(1, Ordering::SeqCst);
                let square = primals[0].clone() * primals[0].clone();
                Ok((vec![square.clone() * primals[0].clone()], vec![square * tangents[0].clone()]))
            })
            .with_vjp(
                move |primals| {
                    forward_counters.forward.fetch_add(1, Ordering::SeqCst);
                    let square = primals[0].clone() * primals[0].clone();
                    Ok((vec![square.clone() * primals[0].clone()], vec![square]))
                },
                move |leading_inputs, seeds| {
                    backward_counters.backward.fetch_add(1, Ordering::SeqCst);
                    let contribution = leading_inputs[0].clone() * seeds[0].clone();
                    Ok(vec![contribution.clone() + contribution])
                },
            )
            .with_batching()
    }

    /// Stages one call of `definition` over an input of type `r#type`.
    pub(crate) fn custom_rule_program(definition: &TestRegistration, r#type: ArrayType) -> FlatProgram<TestContext> {
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let primal = builder.import_program(cube_program(&r#type));
        let input = builder.add_input(r#type);
        let output = builder
            .add_instruction(CustomFunctionOperation::new(definition.reference()), vec![primal], vec![input], None)
            .unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Returns the outputs and tangents of the forward-mode differentiation of one call of `definition` at `x` with
    /// tangent `tangent`.
    pub(crate) fn call_jvp(
        definition: &TestRegistration,
        x: Array,
        tangent: Array,
    ) -> Result<(Vec<Array>, Vec<Array>), DifferentiationError> {
        let r#type = x.r#type().into_owned();
        TestContext::new().jvp(
            |x, ()| {
                let operation = CustomFunctionOperation::new(definition.reference());
                x.context().bind(operation, vec![cube_program(&r#type)], &[x.clone()])
            },
            x,
            tangent,
            (),
        )
    }
}
