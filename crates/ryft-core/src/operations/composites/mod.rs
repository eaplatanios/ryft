//! Contains functionality for building _composite_ operations. Composites in Ryft are functions with custom transform
//! rules. Ryft composites follow the StableHLO [`composite`](https://openxla.org/stablehlo/spec#composite) design, with
//! the operation carrying its decomposition, where here the decomposition is the primal function. A composite pairs
//! that primal with handwritten rules for the transforms whose automatic treatment of it is inadequate (e.g., a fused
//! or foreign kernel, a numerically stabilized derivative, or a batching strategy that structural batching cannot
//! discover). Every transform without a configured rule treats the primal as it treats any other function, except that
//! a composite with only reverse mode rules rejects forward mode. This module has three layers:
//!
//!   - **Functions:** [`composite`] creates a [`Composite`] from its primal, and configuration functions attach its
//!     rules, including support for a Jacobian-Vector Product (JVP) rule that governs both forward and reverse mode
//!     differentiation (via [`Composite::with_jvp`]), forward and backward rules that govern reverse mode
//!     differentiation (via [`Composite::with_vjp`]), or both, and a batching rule (via [`Composite::with_batching`])
//!     or sequential batching (via [`Composite::with_sequential_batching`]). [`Composite::from_custom_call`] creates
//!     a composite whose primal calls a foreign kernel.
//!   - **Operations:** Each call of a composite stages a [`CompositeOperation`], which carries the primal program
//!     as an attached region and retains the rules as callbacks that are traced on the first request of each
//!     specialization. Callers whose rules are already programs (e.g., kernel staging with explicit rule programs)
//!     stage the same operation with the rule programs attached as regions instead. Reverse mode replaces a call
//!     with reverse mode rules, in either representation, with a [`CompositeTransposeOperation`] carrier whose
//!     transposition applies the backward rule, while it transposes the linearized forward mode rule of a call
//!     without reverse mode rules.
//!   - **Rules:** [`CustomRuleDefinition`] and its [`CustomRuleRegistration`] are the flat, family-level rule sets
//!     behind the retained rules, which trace and cache the rule programs of each specialization.
//!
//! Outside of the transforms that their rules govern, composites are _transparent_ meaning that interpretation and
//! backend lowering simply replay the primal program. Refer to the documentation of [`composite`] for the calling
//! convention and for how the rules interact with references, batching, and partial evaluation.
//!
//! # Examples
//!
//! ```rust
//! # use ryft_core::{
//! #     Array, ArrayOperation, Cos, DomainTracer, EagerContext, ProgramError, Sin, composite, differentiate_at,
//! # };
//! # fn main() -> Result<(), ProgramError> {
//! type Tracer = DomainTracer<EagerContext<Array, ArrayOperation<Array>>>;
//!
//! // A custom JVP rule for `sin` that computes the wrong derivative, so that its effect is visible.
//! let sine = composite(|x: Tracer| Ok(x.sin()?)).with_jvp(|x, tangent| {
//!     let tangent = x.cos()? * tangent;
//!     Ok((x.sin()?, tangent.clone() + tangent))
//! });
//! let (value, tangent) = differentiate_at(Array::scalar(0.5f64)?).jvp(Array::scalar(1.0f64)?, |x| sine.call(x))?;
//! assert_eq!(value, Array::scalar(0.5f64.sin())?);
//! assert_eq!(tangent, Array::scalar(2.0 * 0.5f64.cos())?);
//! # Ok(())
//! # }
//! ```

use crate::programs::{Type, TypeIdentityRenaming};

pub mod builders;
pub mod configurations;
pub mod operations;
pub mod rules;

pub use builders::{BatchedComposite, Composite, CompositeCallContext, JvpComposite, VjpComposite, composite};
pub use configurations::{
    AccumulatingVjp, BroadcastInvariantBatching, BroadcastSensitiveBatching, CompositeBatching, CompositeJvp,
    CompositeVjp, DefaultBatching, DefaultJvp, DefaultVjp, ExplicitJvp, ExplicitVjp, JvpFromPrimal, SequentialBatching,
    SymbolicZeroJvp, SymbolicZeroVjp,
};
pub use operations::{
    COMPOSITE_OPERATION_NAME, COMPOSITE_TRANSPOSE_OPERATION_NAME, CompositeJvpRule, CompositeOperation,
    CompositeTransposeOperation,
};
pub use rules::{
    CustomRuleDefinition, CustomRuleReference, CustomRuleRegistration, CustomRuleSource, CustomRuleSpecializer,
    CustomRuleTracer, CustomVjpOperation, LiftedCustomRules, UnavailableCustomRules, WeakCustomRuleRegistration,
};

/// Returns the renaming of the type identities of `source` onto those of `target` when the two type signatures are
/// equal up to a bijective renaming of their identities, which holds exactly when each renames into the other.
fn derive_bijective_identity_renaming<T: Type>(
    source: &[T],
    target: &[T],
) -> Option<TypeIdentityRenaming<T::Identity>> {
    let renames_into = |declared: &[T], actual: &[T]| {
        let renaming = T::derive_identity_renaming(declared, actual).ok()?;
        declared
            .iter()
            .zip(actual)
            .all(|(declared, actual)| declared.rename_identities(&renaming).is_ok_and(|renamed| &renamed == actual))
            .then_some(renaming)
    };
    if source.len() != target.len() {
        return None;
    }
    renames_into(target, source)?;
    renames_into(source, target)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayOperation, ArrayType, DataType, DimensionBounds, DimensionVariable, Shape};
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::{DifferentiationError, ForwardModeDifferentiate};
    use crate::operations::arithmetic::MulOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{FlatProgram, ProgramBuilder, Typed};

    use super::*;

    // Fixtures of the tests of calls and carriers with retained rules.

    /// Eager context whose operation family registers the retained-rule operations.
    pub(super) type TestContext = EagerContext<Array, ArrayOperation<Array>>;

    /// Retained-rule definition over [`TestContext`].
    pub(super) type TestDefinition = CustomRuleDefinition<Array, ArrayOperation<Array>>;

    /// Registration of a [`TestDefinition`].
    pub(super) type TestRegistration = CustomRuleRegistration<Array, ArrayOperation<Array>>;

    /// Retained-rule definition over the production [`ArrayOperation`] member family.
    pub(super) type MemberDefinition = CustomRuleDefinition<Array, ArrayOperation<Array>>;

    /// Invocation counters of the retained rules of a [`cube_definition`].
    #[derive(Default)]
    pub(super) struct RuleCounters {
        /// Number of JVP rule invocations.
        pub(super) jvp: AtomicUsize,

        /// Number of VJP forward rule invocations.
        pub(super) forward: AtomicUsize,

        /// Number of VJP backward rule invocations.
        pub(super) backward: AtomicUsize,
    }

    impl RuleCounters {
        /// Returns the JVP, forward, and backward invocation counts.
        pub(super) fn counts(&self) -> (usize, usize, usize) {
            (self.jvp.load(Ordering::SeqCst), self.forward.load(Ordering::SeqCst), self.backward.load(Ordering::SeqCst))
        }
    }

    /// Builds the primal `f(x) = x³`.
    pub(super) fn cube_program(r#type: &ArrayType) -> FlatProgram<TestContext> {
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(r#type.clone());
        let square = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
        let cube = builder.add_instruction(MulOperation::new(), Vec::new(), vec![square, input], None).unwrap()[0];
        builder.build(vec![cube], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Returns a definition of `f(x) = x³` whose deliberately wrong rules distinguish the selected rule: the JVP rule
    /// computes `ẏ = x² ẋ` and the VJP rule computes `x̄ = 2 x² ȳ`, while the true derivative is `3 x²`. Each rule
    /// counts its invocations in `counters`.
    pub(super) fn cube_definition(counters: &Arc<RuleCounters>, jvp: bool, vjp: bool) -> TestDefinition {
        let mut definition = TestDefinition::new("cube").with_batching_support();
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
    pub(super) fn member_cube_definition(counters: &Arc<RuleCounters>) -> MemberDefinition {
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
            .with_batching_support()
    }

    /// Stages one call of `definition` over an input of type `r#type`.
    pub(super) fn custom_rule_program(definition: &TestRegistration, r#type: ArrayType) -> FlatProgram<TestContext> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let primal = builder.import_program(cube_program(&r#type));
        let input = builder.add_input(r#type);
        let output = builder
            .add_instruction(CompositeOperation::new(definition.reference()), vec![primal], vec![input], None)
            .unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Returns the outputs and tangents of the forward mode differentiation of one call of `definition` at `x` with
    /// tangent `tangent`.
    pub(super) fn call_jvp(
        definition: &TestRegistration,
        x: Array,
        tangent: Array,
    ) -> Result<(Vec<Array>, Vec<Array>), DifferentiationError> {
        let r#type = x.r#type().into_owned();
        TestContext::new().jvp(
            |x, ()| {
                let operation = CompositeOperation::new(definition.reference());
                x.context().bind(operation, vec![cube_program(&r#type)], &[x.clone()])
            },
            x,
            tangent,
            (),
        )
    }

    #[test]
    fn test_derive_bijective_identity_renaming() {
        let bounds = DimensionBounds::new(1, Some(9)).unwrap();
        let (a, b, c) = (
            DimensionVariable::new("a", bounds),
            DimensionVariable::new("b", bounds),
            DimensionVariable::new("c", bounds),
        );
        let vector_type =
            |variable: &DimensionVariable| ArrayType::new(DataType::F32, Shape::new(vec![variable.clone().into()]));

        // Equal signatures are related by the identity renaming.
        let renaming = derive_bijective_identity_renaming(&[vector_type(&a)], &[vector_type(&a)]).unwrap();
        assert!(renaming.is_identity());

        // A bijective renaming maps each source identity onto its target identity.
        let renaming = derive_bijective_identity_renaming(
            &[vector_type(&a), vector_type(&b)],
            &[vector_type(&c), vector_type(&b)],
        )
        .unwrap();
        assert!(!renaming.is_identity());
        assert_eq!(vector_type(&a).rename_identities(&renaming), Ok(vector_type(&c)));
        assert_eq!(vector_type(&b).rename_identities(&renaming), Ok(vector_type(&b)));

        // Renamings that merge or split identities, and signatures that differ beyond their identities, have none.
        assert!(
            derive_bijective_identity_renaming(
                &[vector_type(&a), vector_type(&b)],
                &[vector_type(&c), vector_type(&c)]
            )
            .is_none()
        );
        assert!(
            derive_bijective_identity_renaming(
                &[vector_type(&a), vector_type(&a)],
                &[vector_type(&b), vector_type(&c)]
            )
            .is_none()
        );
        assert!(derive_bijective_identity_renaming(&[vector_type(&a)], &[vector_type(&a), vector_type(&a)]).is_none());
        assert!(
            derive_bijective_identity_renaming(
                &[vector_type(&a)],
                &[ArrayType::new(DataType::F64, Shape::new(vec![a.clone().into()]))]
            )
            .is_none()
        );
    }
}
