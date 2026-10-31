//! The [`CompositeOperation`] that calls a primal program with custom transform rules and the
//! [`CompositeTransposeOperation`] carrier that applies a retained or attached backward rule during transposition.

use std::hash::Hash;

use crate::batching::BatchAxis;
use crate::contexts::StagingContext;
use crate::differentiation::{DifferentiableType, DifferentiationError};
use crate::operations::composites::rules::specialization::CustomRuleSpecializer;
use crate::operations::composites::rules::{CustomRuleBatching, CustomRuleBatchingLevel, CustomRuleSpecializationKey};
use crate::programs::{Operation, Value};
use crate::tracing::TracingContext;

mod call;
mod transpose;

pub use call::{COMPOSITE_OPERATION_NAME, CompositeJvpRule, CompositeOperation};
pub use transpose::{COMPOSITE_TRANSPOSE_OPERATION_NAME, CompositeTransposeOperation};

// TODO(eaplatanios): Review from here onwards.

/// Returns the batching record of the [`CompositeTransposeOperation`] that reverse mode stages for a batched call with
/// retained rules whose forward rule specialization key is `key` (whose levels are those of the call), whose rules have
/// `rule_output_count` outputs, whose carrier receives seed extents of types `seed_extent_types`, and whose output
/// tangents have the unbatched types `extent_types`. At each level, the carrier's inputs have the batch axes of the
/// corresponding call inputs, except for the residuals, which have the axes of that level's forward specialization, and
/// for the seed extents, which are replicated, while its outputs have the call's batch axes, or those of the forward
/// specialization for the outputs that the call does not keep. The carrier's unbatched input types are those of the
/// unbatched forward rule.
fn transpose_carrier_batching<V, O, S>(
    rules: &S,
    key: &CustomRuleSpecializationKey<V::Type>,
    rule_output_count: usize,
    seed_extent_types: &[V::Type],
    extent_types: Vec<V::Type>,
) -> Result<CustomRuleBatching<V::Type>, DifferentiationError>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: Value<Type: DifferentiableType + Eq + Hash>,
    O: Operation<Type = V::Type>,
    S: CustomRuleSpecializer<V, O>,
{
    let unbatched = rules.forward_specialization(CustomRuleSpecializationKey { levels: Vec::new(), ..key.clone() })?;
    let non_differentiated_count = key.non_differentiated_count;
    let mut input_types = key.input_types[..non_differentiated_count].to_vec();
    input_types.extend_from_slice(&unbatched.program.output_types()[rule_output_count..]);
    input_types.extend_from_slice(seed_extent_types);
    for r#type in &key.input_types[non_differentiated_count..] {
        input_types.push(r#type.tangent()?);
    }
    let mut levels = Vec::with_capacity(key.levels.len());
    for (index, level) in key.levels.iter().enumerate() {
        let level_key = CustomRuleSpecializationKey { levels: key.levels[..=index].to_vec(), ..key.clone() };
        let level_specialization = rules.forward_specialization(level_key)?;
        let leading_input_count = non_differentiated_count
            + key.levels[..index].iter().map(|level| level.boundary_input_count).sum::<usize>();
        let mut input_axes = level.input_axes[..leading_input_count].to_vec();
        input_axes.extend_from_slice(&level_specialization.output_axes[rule_output_count..]);
        input_axes.extend(std::iter::repeat_n(BatchAxis::replicated(), seed_extent_types.len()));
        input_axes.extend_from_slice(&level.input_axes[leading_input_count..]);
        let output_axes = match key.kept_outputs {
            None => level.output_axes.clone(),
            Some(_) => level_specialization.output_axes[..rule_output_count].to_vec(),
        };
        levels.push(CustomRuleBatchingLevel { input_axes, output_axes, ..level.clone() });
    }
    Ok(CustomRuleBatching { input_types, output_types: extent_types, levels })
}

#[cfg(test)]
mod tests {
    use crate::arrays::{Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, ShardingDimension};
    use crate::batching::{BatchAxis, BatchedProgram, ProgramBatchingOutputAxesPolicy};
    use crate::contexts::EagerContext;
    use crate::operations::composites::rules::{CustomRuleDefinition, CustomRuleRegistration, CustomRuleTracer};
    use crate::operations::composites::tests::TestContext;
    use crate::parameters::Placeholder;
    use crate::programs::{FlatProgram, ProgramBuilder};

    use super::*;

    /// Eager context whose values are arrays.
    pub(super) type ArrayContext = EagerContext<Array, ArrayOperation<Array>>;

    /// Array IR program family into which member programs are converted.
    pub(super) type ConvertedProgram = FlatProgram<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>;

    /// Retained-rule definition over the production [`ArrayIrOperation`] family, which supports reference state.
    pub(super) type IrDefinition = CustomRuleDefinition<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Tracer over which the rules of an [`IrDefinition`] are written.
    pub(super) type IrTracer = CustomRuleTracer<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Stages one call of the array IR `definition` over `primal` with the provided input types.
    pub(super) fn ir_custom_rule_program(
        definition: &CustomRuleRegistration<ArrayIrValue<Array>, ArrayIrOperation<Array>>,
        non_differentiated_count: usize,
        primal: ConvertedProgram,
    ) -> ConvertedProgram {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input_types = primal.input_types();
        let primal = builder.import_program(primal);
        let inputs = input_types.into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let input_count = inputs.len();
        let operation = ArrayIrOperation::Composite(
            CompositeOperation::new(definition.reference())
                .with_non_differentiated_count(non_differentiated_count)
                .unwrap(),
        );
        let output = builder.add_instruction(operation, vec![primal], inputs, None).unwrap()[0];
        builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder; input_count],
                vec![Placeholder],
            )
            .unwrap()
    }

    /// Batches `program` over a batch axis of size `extent` with the provided input batch axes.
    pub(super) fn batched_program(
        program: &FlatProgram<TestContext>,
        extent: usize,
        input_axes: &[BatchAxis],
    ) -> FlatProgram<TestContext> {
        program
            .batched(extent, ShardingDimension::Replicated, input_axes, ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts()
            .0
    }
}
