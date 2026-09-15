//! Bounded exploration of host operation interleavings for qualified straight-line kernels.
//!
//! Each execution uses the canonical resumable interpreter and fresh private storage. Sequential grid dimensions
//! impose program-completion dependencies; parallel dimensions permit every merge of program instruction orders.
//! This is an executable concurrency specification for small kernels, not a model of device timing. Nested regions
//! and masked private windows remain executable by ordinary host interpretation but are outside this explorer.

use thiserror::Error;

use crate::arrays::{Array, ArrayIrType, ArrayIrValue};
use crate::contexts::EagerContext;
use crate::interpretation::InterpretableOperation;
use crate::kernels::calls::KernelDefinition;
use crate::kernels::grids::{Grid, GridExecution};
use crate::kernels::initialization::{KernelInitializationError, validate_kernel_initialization};
use crate::kernels::interpretation::{KernelDebugOptions, KernelTraceEntry};
use crate::kernels::mappings::BoundaryPolicy;
use crate::kernels::operations::{KernelExtension, KernelOperation};
use crate::programs::{Operation, ProgramError, ReferenceAccessMode, Typed};

/// Admission and resource-limit failures before interleaving execution begins.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum KernelSchedulingError {
    /// A valid kernel requires a continuation or window model not supported by this bounded explorer.
    #[error("kernel interleaving exploration does not support {reason}")]
    Unsupported {
        /// Concrete unsupported semantic structure.
        reason: &'static str,
    },

    /// Complete exploration would exceed the caller's bound. No partial list is presented as exhaustive.
    #[error("kernel interleaving count exceeds the limit {maximum}")]
    InterleavingLimit {
        /// Maximum complete instruction orders allowed by the caller.
        maximum: usize,
    },

    /// The total instruction and region-entry count cannot be represented by a host index.
    #[error("kernel interleaving step count exceeds the host index range")]
    StepCountOverflow,

    /// The straight-line invocation already exceeds the replay budget before any order is explored.
    #[error("kernel interleaving requires {required} replay steps, exceeding the limit {maximum}")]
    StepLimit {
        /// Instruction bindings plus one region entry per logical program.
        required: usize,
        /// Invocation-wide replay-step limit.
        maximum: usize,
    },
}

/// Outcome and operation trace of one admitted instruction order. Execution errors remain attached to their trace,
/// and do not stop independent orders from being checked. Allocation identities are normalized within each trace.
#[derive(Clone, Debug)]
pub struct KernelInterleaving {
    /// Functional outputs, or the exact execution diagnostic for this order.
    pub result: Result<Vec<Array>, ProgramError>,
    /// Attempted operation bindings in their actual execution order.
    pub trace: Vec<KernelTraceEntry>,
}

impl<Extension> KernelDefinition<Extension>
where
    Extension: KernelExtension + InterpretableOperation<EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>>,
{
    /// Explores all program-order-preserving instruction interleavings within the supplied bounds. Qualification
    /// rejects races and uninitialized accesses before exploration. Ordinary floating-point atomic accumulation can
    /// produce different legal results across orders; each result is retained without selecting a preferred answer.
    ///
    /// Atomic updates must select at most one element so an operation step represents one indivisible update.
    /// Nested regions and masked windows are explicitly unsupported here. Scalar prefetch is specialized from the
    /// supplied ordinary array inputs before planning, as in ordinary host interpretation. Dynamic grid extents must
    /// already have been specialized. Exceeding a planning bound returns an error before any execution begins.
    pub fn interpret_interleavings(
        &self,
        mut inputs: Vec<Array>,
        options: &KernelDebugOptions,
        maximum_interleavings: usize,
    ) -> Result<Vec<KernelInterleaving>, ProgramError> {
        self.operation().infer_output_types(
            &inputs.iter().map(|input| ArrayIrType::Array(input.r#type().into_owned())).collect::<Vec<_>>(),
            &[self.body().entry_region_ref().interface()],
        )?;
        let prefetch_count = self.operation().prefetch_types().len();
        if prefetch_count != 0 {
            let prefetched = inputs.split_off(inputs.len() - prefetch_count);
            return self.specialize_prefetch(&prefetched).map_err(ProgramError::custom)?.interpret_interleavings(
                inputs,
                options,
                maximum_interleavings,
            );
        }
        if self
            .body()
            .entry_region()
            .instructions()
            .iter()
            .any(|instruction| !instruction.regions().is_empty())
        {
            return Err(ProgramError::custom(KernelSchedulingError::Unsupported { reason: "attached regions" }));
        }
        if self
            .operation()
            .parameters()
            .iter()
            .any(|parameter| parameter.mapping().boundary_policy() != BoundaryPolicy::InBounds)
        {
            return Err(ProgramError::custom(KernelSchedulingError::Unsupported { reason: "masked windows" }));
        }
        let body = self.body().entry_region_ref();
        for instruction in body.instructions() {
            for (input, mode) in instruction.operation().effects().accesses() {
                if mode == ReferenceAccessMode::AtomicAccumulate {
                    let r#type = body.atoms()[instruction.inputs()[input].index()].r#type();
                    let ArrayIrType::Reference(reference) = r#type.as_ref() else { unreachable!() };
                    if reference.referent().element_count()?.is_none_or(|count| count > 1) {
                        return Err(ProgramError::custom(KernelSchedulingError::Unsupported {
                            reason: "non-scalar atomic updates",
                        }));
                    }
                }
            }
        }
        let extents = self
            .operation()
            .grid()
            .dimensions()
            .iter()
            .map(|dimension| {
                dimension.extent().value().ok_or_else(|| {
                    ProgramError::custom(KernelInitializationError::UnsupportedLaunch { boundary: "grid extents" })
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let points = self.operation().grid().points(&extents).map_err(ProgramError::custom)?;
        if points.len() > options.maximum_programs {
            return Err(ProgramError::custom(KernelInitializationError::QualificationLimit {
                programs: points.len(),
                maximum: options.maximum_programs,
            }));
        }
        validate_kernel_initialization(self.body().entry_region_ref(), self.operation(), options.maximum_programs)
            .map_err(ProgramError::custom)?;
        let instructions = self.body().entry_region().instructions().len();
        let required = instructions
            .checked_add(1)
            .and_then(|count| count.checked_mul(points.len()))
            .ok_or_else(|| ProgramError::custom(KernelSchedulingError::StepCountOverflow))?;
        if required > options.maximum_steps {
            return Err(ProgramError::custom(KernelSchedulingError::StepLimit {
                required,
                maximum: options.maximum_steps,
            }));
        }
        let points = points.collect::<Vec<_>>();
        let orders = self
            .operation()
            .grid()
            .interleavings(&extents, &points, instructions, maximum_interleavings)
            .map_err(ProgramError::custom)?;
        Ok(orders
            .into_iter()
            .map(|order| {
                let mut trace = Vec::new();
                let result = self.interpret_in_order(inputs.clone(), options, Some(&order), &mut trace);
                KernelInterleaving { result, trace }
            })
            .collect())
    }
}

impl Grid {
    /// Enumerates instruction-order merges iteratively, avoiding host recursion proportional to kernel size.
    /// Immediate predecessors along sequential axes suffice to enforce the grid's complete partial order.
    fn interleavings(
        &self,
        extents: &[usize],
        points: &[Vec<usize>],
        instructions: usize,
        maximum: usize,
    ) -> Result<Vec<Vec<usize>>, KernelSchedulingError> {
        if points.is_empty() {
            return if maximum == 0 {
                Err(KernelSchedulingError::InterleavingLimit { maximum })
            } else {
                Ok(vec![vec![]])
            };
        }
        let mut predecessors = vec![Vec::new(); points.len()];
        let mut stride = 1;
        for (axis, dimension) in self.dimensions().iter().enumerate().rev() {
            if dimension.execution() == GridExecution::Sequential {
                for (program, point) in points.iter().enumerate() {
                    if point[axis] != 0 {
                        predecessors[program].push(program - stride);
                    }
                }
            }
            stride *= extents[axis];
        }
        let steps = points.len() * instructions;
        let mut counts = vec![0; points.len()];
        let mut order = Vec::with_capacity(steps);
        let mut candidates = vec![0];
        let mut orders = Vec::new();
        loop {
            if order.len() == steps {
                if orders.len() == maximum {
                    return Err(KernelSchedulingError::InterleavingLimit { maximum });
                }
                orders.push(order.clone());
            } else {
                let next = candidates.last_mut().unwrap();
                let candidate = (*next..points.len()).find(|&program| {
                    counts[program] < instructions
                        && predecessors[program].iter().all(|&previous| counts[previous] == instructions)
                });
                if let Some(program) = candidate {
                    *next = program + 1;
                    counts[program] += 1;
                    order.push(program);
                    candidates.push(0);
                    continue;
                }
            }
            candidates.pop();
            let Some(program) = order.pop() else { break };
            counts[program] -= 1;
        }
        Ok(orders)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrOperation, ArrayOperation, ArrayType, DataType, Dimension, DimensionBounds, DimensionType,
        DimensionValue, DimensionVariable,
    };
    use crate::contexts::Context;
    use crate::kernels::calls::{KernelCallOperation, KernelParameter};
    use crate::kernels::grids::GridDimension;
    use crate::kernels::interpretation::KernelInterpretationError;
    use crate::kernels::mappings::BlockMapping;
    use crate::kernels::validation::KernelParameterAccess;
    use crate::operations::{
        CompareOperation, ComparisonDirection, ConditionOperation, ReferenceAtomicAddUpdate, SelectOperation,
    };
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, Provenance};

    use super::*;

    /// Two programs with opposite large updates followed by unit updates expose legal floating-point atomic orders.
    fn atomic_definition(execution: GridExecution) -> KernelDefinition {
        let mut mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        mapping.add_input(ArrayIrType::Dimension(DimensionType::new(DimensionVariable::new(
            "coordinate",
            DimensionBounds::non_negative(Some(2)).unwrap(),
        ))));
        let mapping = BlockMapping::new(
            mapping.build(vec![], vec![Placeholder], vec![]).unwrap(),
            vec![],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        let call = KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(2), execution)]).unwrap(),
            vec![
                KernelParameter::new(ArrayType::scalar(DataType::F32), KernelParameterAccess::ReadWrite, mapping)
                    .unwrap(),
            ],
        )
        .unwrap();
        KernelDefinition::trace(call, |(references, coordinates)| {
            let context = references[0].context();
            let zero = context.lift(ArrayIrValue::Dimension(DimensionValue::constant(0)?))?;
            let first = context
                .bind(
                    ArrayIrOperation::Compare(CompareOperation::new(ComparisonDirection::Equal)),
                    vec![],
                    &[coordinates[0].clone(), zero],
                )?
                .remove(0);
            let positive = context.lift(ArrayIrValue::Array(Array::scalar(1e20f32)?))?;
            let negative = context.lift(ArrayIrValue::Array(Array::scalar(-1e20f32)?))?;
            let update = context
                .bind(
                    ArrayIrOperation::Array(ArrayOperation::Select(SelectOperation::new())),
                    vec![],
                    &[first, positive, negative],
                )?
                .remove(0);
            references[0].atomic_add_update(&update)?;
            let one = context.lift(ArrayIrValue::Array(Array::scalar(1f32)?))?;
            references[0].atomic_add_update(&one)
        })
        .unwrap()
    }

    #[test]
    fn test_grid_interleavings() {
        let grid = Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Parallel)]).unwrap();
        let points = grid.points(&[2]).unwrap().collect::<Vec<_>>();
        assert_eq!(
            grid.interleavings(&[2], &points, 2, 6),
            Ok(vec![
                vec![0, 0, 1, 1],
                vec![0, 1, 0, 1],
                vec![0, 1, 1, 0],
                vec![1, 0, 0, 1],
                vec![1, 0, 1, 0],
                vec![1, 1, 0, 0],
            ]),
        );
        assert_eq!(
            grid.interleavings(&[2], &points, 2, 5),
            Err(KernelSchedulingError::InterleavingLimit { maximum: 5 }),
        );
        let grid = Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Sequential)]).unwrap();
        assert_eq!(grid.interleavings(&[2], &points, 2, 1), Ok(vec![vec![0, 0, 1, 1]]));
    }

    #[test]
    fn test_grid_interleavings_partial_order() {
        let grid = Grid::new(vec![
            GridDimension::new(Dimension::Static(2), GridExecution::Sequential),
            GridDimension::new(Dimension::Static(2), GridExecution::Sequential),
        ])
        .unwrap();
        let points = grid.points(&[2, 2]).unwrap().collect::<Vec<_>>();
        assert_eq!(grid.interleavings(&[2, 2], &points, 1, 2), Ok(vec![vec![0, 1, 2, 3], vec![0, 2, 1, 3]]));
        assert_eq!(grid.interleavings(&[2, 2], &points, 0, 1), Ok(vec![vec![]]));
        let grid = Grid::new(vec![GridDimension::new(Dimension::Static(0), GridExecution::Parallel)]).unwrap();
        assert_eq!(grid.interleavings(&[0], &[], 3, 1), Ok(vec![vec![]]));
        assert_eq!(grid.interleavings(&[0], &[], 3, 0), Err(KernelSchedulingError::InterleavingLimit { maximum: 0 }));
    }

    #[test]
    fn test_kernel_definition_interpret_interleavings() {
        let definition = atomic_definition(GridExecution::Parallel);
        let input = Array::scalar(0f32).unwrap();
        let options = KernelDebugOptions { maximum_programs: 2, maximum_steps: 10, check_nans: false };
        let executions = definition.interpret_interleavings(vec![input.clone()], &options, 70).unwrap();
        assert_eq!(executions.len(), 70);
        let results = executions
            .iter()
            .map(|execution| {
                assert_eq!(execution.trace.len(), 8);
                execution.result.as_ref().unwrap()[0].elements::<f32>().unwrap()[0].to_bits()
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(results, BTreeSet::from([1f32.to_bits(), 2f32.to_bits()]));
        assert_eq!(definition.interpret(vec![input.clone()], 2), Ok(vec![Array::scalar(1f32).unwrap()]));
        assert_eq!(input, Array::scalar(0f32).unwrap());
        let error = definition.interpret_interleavings(vec![input.clone()], &options, 69).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelSchedulingError>(),
            Some(&KernelSchedulingError::InterleavingLimit { maximum: 69 }),
        );
        assert_eq!(error.to_string(), "kernel interleaving count exceeds the limit 69");
        let error = definition
            .interpret_interleavings(vec![input], &KernelDebugOptions { maximum_steps: 9, ..options }, 70)
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelSchedulingError>(),
            Some(&KernelSchedulingError::StepLimit { required: 10, maximum: 9 }),
        );
        assert_eq!(error.to_string(), "kernel interleaving requires 10 replay steps, exceeding the limit 9");
    }

    #[test]
    fn test_kernel_definition_interpret_interleavings_sequential() {
        let definition = atomic_definition(GridExecution::Sequential);
        let executions = definition
            .interpret_interleavings(vec![Array::scalar(0f32).unwrap()], &KernelDebugOptions::default(), 1)
            .unwrap();
        assert_eq!(executions.len(), 1);
        assert_eq!(executions[0].result, Ok(vec![Array::scalar(1f32).unwrap()]));
        assert_eq!(
            executions[0].trace.iter().map(|entry| entry.coordinate.clone()).collect::<Vec<_>>(),
            vec![vec![0], vec![0], vec![0], vec![0], vec![1], vec![1], vec![1], vec![1],],
        );
    }
    #[test]
    fn test_kernel_definition_interpret_interleavings_rejects_masked_windows() {
        let original = atomic_definition(GridExecution::Parallel);
        let parameter = &original.operation().parameters()[0];
        let mapping = BlockMapping::new(parameter.mapping().program().clone(), vec![], BoundaryPolicy::Masked).unwrap();
        let call = KernelCallOperation::new(
            original.operation().grid().clone(),
            vec![
                KernelParameter::new(ArrayType::scalar(DataType::F32), KernelParameterAccess::ReadWrite, mapping)
                    .unwrap(),
            ],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(call, |(references, _)| {
            let update = references[0].context().lift(ArrayIrValue::Array(Array::scalar(1f32)?))?;
            references[0].atomic_add_update(&update)
        })
        .unwrap();
        let error = definition
            .interpret_interleavings(vec![Array::scalar(0f32).unwrap()], &KernelDebugOptions::default(), 2)
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelSchedulingError>(),
            Some(&KernelSchedulingError::Unsupported { reason: "masked windows" }),
        );
        assert_eq!(error.to_string(), "kernel interleaving exploration does not support masked windows");
    }

    #[test]
    fn test_kernel_definition_interpret_interleavings_rejects_attached_regions() {
        let call = KernelCallOperation::new(Grid::new(vec![]).unwrap(), vec![]).unwrap();
        let branch = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![], vec![])
            .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let first = builder.import_region(branch.entry_region_ref());
        let second = builder.import_region(branch.entry_region_ref());
        let predicate = builder.add_constant(ArrayIrValue::Array(Array::scalar(true).unwrap()));
        builder
            .add_instruction(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                vec![first, second],
                vec![predicate],
                None,
            )
            .unwrap();
        let body = builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![], vec![]).unwrap();
        let definition = KernelDefinition::new(call, body).unwrap();
        let error = definition.interpret_interleavings(vec![], &KernelDebugOptions::default(), 1).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelSchedulingError>(),
            Some(&KernelSchedulingError::Unsupported { reason: "attached regions" }),
        );
        assert_eq!(error.to_string(), "kernel interleaving exploration does not support attached regions");
    }

    #[test]
    fn test_kernel_definition_interpret_interleavings_rejects_vector_atomics() {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let zero = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()));
        let mapping = BlockMapping::new(
            builder.build(vec![zero], vec![], vec![Placeholder]).unwrap(),
            vec![2],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        let input = Array::vector(vec![0f32, 0f32]).unwrap();
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![KernelParameter::new(input.r#type().into_owned(), KernelParameterAccess::ReadWrite, mapping).unwrap()],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace(call, |(references, _)| {
            let update = references[0].context().lift(ArrayIrValue::Array(Array::vector(vec![1f32, 1f32])?))?;
            references[0].atomic_add_update(&update)
        })
        .unwrap();
        let error = definition.interpret_interleavings(vec![input], &KernelDebugOptions::default(), 1).unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelSchedulingError>(),
            Some(&KernelSchedulingError::Unsupported { reason: "non-scalar atomic updates" }),
        );
        assert_eq!(error.to_string(), "kernel interleaving exploration does not support non-scalar atomic updates");
    }

    #[test]
    fn test_kernel_definition_interpret_interleavings_retains_failed_traces() {
        let original = atomic_definition(GridExecution::Parallel);
        let definition: KernelDefinition = KernelDefinition::trace(original.operation().clone(), |(references, _)| {
            let update = references[0].context().lift(ArrayIrValue::Array(Array::scalar(f32::NAN)?))?;
            references[0].atomic_add_update(&update)
        })
        .unwrap();
        let input = Array::scalar(0f32).unwrap();
        let executions = definition
            .interpret_interleavings(
                vec![input.clone()],
                &KernelDebugOptions { maximum_programs: 2, maximum_steps: 4, check_nans: true },
                2,
            )
            .unwrap();
        assert_eq!(executions.len(), 2);
        for (program, execution) in executions.iter().enumerate() {
            let error = execution.result.as_ref().unwrap_err();
            assert_eq!(
                error.downcast_custom::<KernelInterpretationError>(),
                Some(&KernelInterpretationError::Nan {
                    operation: "reference_atomic_add_update",
                    position: "input",
                    value: 1,
                    element: 0,
                    data_type: DataType::F32,
                    coordinate: vec![program],
                    provenance: Provenance::unknown(),
                }),
            );
            assert_eq!(
                error.to_string(),
                format!(
                    "nan in `reference_atomic_add_update` input 1 element 0 of type `f32` at grid point [{program}]",
                ),
            );
            assert_eq!(execution.trace.len(), 1);
            assert_eq!(execution.trace[0].operation, "reference_atomic_add_update");
            assert_eq!(execution.trace[0].coordinate, vec![program]);
        }
        assert_eq!(input, Array::scalar(0f32).unwrap());
    }
}
