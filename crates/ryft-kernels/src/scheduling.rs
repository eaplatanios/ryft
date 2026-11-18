//! Contains the bounded exploration of host operation interleavings for qualified straight-line kernels.
//!
//! Ordinary host interpretation (i.e., [`KernelDefinition::interpret`]) replays the grid points of a kernel one after
//! another, which is only one of the orders that a device may choose. This module replays a small kernel under every
//! admitted order instead, so that results which depend on the order (e.g., floating-point atomic accumulation) become
//! observable. It is an executable concurrency specification for small kernels, not a model of device timing.
//!
//! # Interleavings
//!
//! An interleaving is one complete order of instruction steps across all grid points of a launch. Every grid point
//! executes the instructions of the kernel body in program order, and an interleaving merges these per point sequences
//! into one. Parallel grid dimensions permit every merge. Sequential grid dimensions impose completion dependencies, so
//! a grid point whose coordinate along a sequential axis is nonzero starts only after its immediate predecessor along
//! that axis has completed every instruction. These immediate predecessors suffice to enforce the complete partial
//! order of the grid. The specialized definition is qualified once, and each interleaving is then replayed with the
//! canonical resumable interpreter, fresh private storage and runtime state, and the same qualified context as
//! [`KernelDefinition::interpret_with_trace`].
//!
//! # Admission
//!
//! [`KernelDefinition::interpret_interleavings`] specializes scalar prefetch values from its inputs and then admits
//! only kernels whose interleavings are well defined at the granularity of one body instruction.
//!
//!   - The body must be a flat region, because the explorer schedules exactly one step per body instruction and plans
//!     its replay steps from the instruction count of the body, so it cannot interleave the nested steps of an
//!     attached region with the steps of other grid points.
//!   - Every parameter mapping must use the [`BoundaryPolicy::InBounds`] policy. A masked window is a private tile
//!     whose valid region ordinary interpretation publishes back after its grid point completes, and the ordered
//!     replay of an interleaving has no such publication step.
//!   - Every atomic update must select at most one element, so that one step represents one indivisible update.
//!   - Every extension operation of the body must declare its replay cost through [`KernelExtension::replay_cost`].
//!     The interpretation rule of an extension may bind further operations, each of which consumes a replay step, so
//!     without a declared cost the replay cost of an order could not be planned without executing it. The plan is
//!     sound only when that cost bounds every binding of the operation, including one whose rule binds diagnostic
//!     operations and then fails, because an order that exhausts its steps reports the step limit instead of its own
//!     execution error. An extension operation is one step of an interleaving, so the operations that its rule binds
//!     run without interruption.
//!
//! Grid extents must already be static, and qualification (i.e.,
//! [`validate_kernel_initialization`](crate::initialization::validate_kernel_initialization)) rejects races and
//! uninitialized accesses before exploration. Nested regions, masked windows, and extension operations remain
//! executable by ordinary host interpretation, and unsupported kernels (including extension operations of unknown
//! replay cost) fail with [`KernelSchedulingError::Unsupported`].
//!
//! # Limits and Results
//!
//! The [`KernelDebugOptions`] bound the number of grid points and the number of replay steps of each interleaving,
//! and the `maximum_interleavings` argument bounds the number of interleavings. The replay steps of each interleaving
//! are bounded by one step per body entry, one per portable or kernel memory instruction, which binds no other
//! operation, and the declared replay cost of every extension operation, for every grid point. The explorer computes
//! that bound with checked arithmetic and checks it against the step limit, so every order, including one that fails,
//! stays within that limit. Exceeding any of these bounds returns an error before any interleaving executes, so a
//! partial list is never presented as exhaustive. Each returned [`KernelInterleaving`] holds the functional outputs or
//! the exact execution error of its order, together with the trace of the operation bindings that it attempted. An
//! execution error stays attached to its interleaving and does not stop the remaining interleavings from executing.

use std::rc::Rc;

use ryft_core::{
    Array, ArrayIrType, ArrayIrValue, Domain, EagerContext, InterpretableOperation, Operation, ProgramError,
    ReferenceAccessMode, Typed, infer_reference_view_type, validated_reference_access_descriptors,
};
use thiserror::Error;

use crate::calls::{KernelCallOperation, KernelDefinition};
use crate::grids::{Grid, GridExecution};
use crate::initialization::{KernelInitializationError, validated_kernel_references};
use crate::interpretation::{KernelDebugOptions, KernelTraceEntry};
use crate::mappings::BoundaryPolicy;
use crate::operations::{KernelExtension, KernelOperation};

/// Admission and resource limit failures of [`KernelDefinition::interpret_interleavings`], all of which occur before
/// the execution of any interleaving begins.
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

    /// The total count of replay steps (i.e., operation bindings and region entries) cannot be represented by a host
    /// index.
    #[error("kernel interleaving step count exceeds the host index range")]
    StepCountOverflow,

    /// The straight-line invocation already exceeds the replay budget before any order is explored.
    #[error("kernel interleaving requires {required} replay steps, exceeding the limit {maximum}")]
    StepLimit {
        /// Instruction bindings plus one region entry per logical program.
        required: usize,

        /// Replay step limit of the whole invocation.
        maximum: usize,
    },
}

/// Outcome and operation trace of one admitted instruction order, which [`KernelDefinition::interpret_interleavings`]
/// returns. Execution errors remain attached to their trace and do not stop independent orders from being checked.
/// Allocation identities are normalized within each trace.
#[derive(Clone, Debug)]
pub struct KernelInterleaving {
    /// Functional outputs, or the exact execution diagnostic for this order.
    result: Result<Vec<Array>, ProgramError>,

    /// Attempted operation bindings in their actual execution order.
    trace: Vec<KernelTraceEntry>,
}

impl KernelInterleaving {
    /// Returns the functional outputs, or the exact execution diagnostic for this order.
    pub fn result(&self) -> &Result<Vec<Array>, ProgramError> {
        &self.result
    }

    /// Returns the attempted operation bindings in their actual execution order.
    pub fn trace(&self) -> &[KernelTraceEntry] {
        &self.trace
    }
}

impl<Extension> KernelDefinition<Extension>
where
    EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>: Domain<
            Type = ArrayIrType,
            Value = ArrayIrValue<Array>,
            Constant = ArrayIrValue<Array>,
            Operation = KernelOperation<Extension>,
        >,
    Extension: KernelExtension + InterpretableOperation<EagerContext<ArrayIrValue<Array>, KernelOperation<Extension>>>,
{
    /// Explores all instruction interleavings that preserve the program order of every grid point, within the supplied
    /// bounds. Qualification rejects races and uninitialized accesses before exploration. Ordinary floating-point
    /// atomic accumulation can produce different legal results across orders, and each result is retained without
    /// selecting a preferred answer.
    ///
    /// Atomic updates must select at most one element so an operation step represents one indivisible update.
    /// Nested regions, masked windows, and extension operations whose [`KernelExtension::replay_cost`] is unknown are
    /// explicitly unsupported here. Scalar prefetch is specialized from the supplied ordinary array inputs before
    /// planning, as in ordinary host interpretation. Dynamic grid extents must already have been specialized. Each
    /// order takes at most one replay step per body entry and per portable or kernel memory instruction, plus the
    /// declared replay cost of every extension operation, for every grid point, and exceeding a planning bound,
    /// including that step count, returns an error before any execution begins. That bound covers failing orders only
    /// when every declared replay cost also bounds the failing bindings of its operation.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: arrays in the definition's ordinary input order, followed by the scalar prefetch values.
    ///   - `options`: grid point and replay step limits, and whether NaN checks are enabled in each execution.
    ///   - `maximum_interleavings`: maximum number of complete instruction orders to execute.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when the inputs do not match the signature of the definition, when scalar prefetch
    /// specialization or qualification fails (e.g., with a [`KernelInitializationError`] for a dynamic grid extent or
    /// a grid with more than [`KernelDebugOptions::maximum_programs`] points), or with a [`KernelSchedulingError`]
    /// when the kernel is not admitted or a planning bound is exceeded. Failures of individual interleavings are
    /// returned in their [`KernelInterleaving::result`] instead.
    pub fn interpret_interleavings(
        &self,
        mut inputs: Vec<Array>,
        options: &KernelDebugOptions,
        maximum_interleavings: usize,
    ) -> Result<Vec<KernelInterleaving>, ProgramError> {
        let output_types = self.operation().infer_output_types(
            &inputs.iter().map(|input| ArrayIrType::Array(input.r#type().into_owned())).collect::<Vec<_>>(),
            std::slice::from_ref(self.body().entry_region_ref().interface()),
        )?;
        let prefetch_count = self.operation().prefetch_types().len();
        if prefetch_count != 0 {
            // Specialization binds the physical prefetch values, as in ordinary interpretation. The specialized call
            // types its results from the ordinary inputs alone, so the results take the types inferred from all inputs.
            let prefetched = inputs
                .split_off(inputs.len() - prefetch_count)
                .into_iter()
                .zip(self.operation().prefetch_types())
                .map(|(array, r#type)| KernelCallOperation::with_sharding_of(array, r#type))
                .collect::<Result<Vec<_>, _>>()?;
            let mut interleavings = self
                .specialize_prefetch(&prefetched)
                .map_err(ProgramError::custom)?
                .interpret_interleavings(inputs, options, maximum_interleavings)?;
            for interleaving in &mut interleavings {
                if let Ok(results) = &mut interleaving.result {
                    *results = std::mem::take(results)
                        .into_iter()
                        .zip(&output_types)
                        .map(|(array, r#type)| {
                            let ArrayIrType::Array(r#type) = r#type else { unreachable!("kernel results are arrays") };
                            KernelCallOperation::with_sharding_of(array, r#type)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                }
            }
            return Ok(interleavings);
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
        let mut costs = Vec::with_capacity(body.instructions().len());
        for instruction in body.instructions() {
            costs.push(match instruction.operation() {
                KernelOperation::Extension(extension) => extension.replay_cost().ok_or_else(|| {
                    ProgramError::custom(KernelSchedulingError::Unsupported {
                        reason: "extension operations of unknown replay cost",
                    })
                })?,
                _ => 1,
            });
            let descriptors =
                validated_reference_access_descriptors(instruction.operation(), instruction.inputs().len())?;
            for (input, mode) in instruction.operation().effects().reference_accesses() {
                if mode == ReferenceAccessMode::AtomicAccumulate {
                    let r#type = body.atoms()[instruction.inputs()[input].index()].r#type();
                    let ArrayIrType::Reference(reference) = r#type.as_ref() else { unreachable!() };
                    // Every declared access has a validated descriptor, and the update applies to the view that its
                    // transforms select rather than to the complete referent.
                    let Some(descriptor) = &descriptors[input] else { unreachable!() };
                    let bindings = instruction.inputs()[descriptor.bindings()]
                        .iter()
                        .map(|binding| body.atoms()[binding.index()].r#type())
                        .collect::<Vec<_>>();
                    let view = infer_reference_view_type(
                        reference.referent(),
                        descriptor.transforms(),
                        &bindings.iter().map(AsRef::as_ref).collect::<Vec<_>>(),
                        mode,
                    )?;
                    if view.element_count()?.is_none_or(|count| count > 1) {
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
        if points.len() > options.maximum_programs() {
            return Err(ProgramError::custom(KernelInitializationError::QualificationLimit {
                programs: points.len(),
                maximum: options.maximum_programs(),
            }));
        }
        let references = Rc::new(
            validated_kernel_references(self.body().entry_region_ref(), self.operation(), options.maximum_programs())
                .map_err(ProgramError::custom)?,
        );
        let instructions = costs.len();
        let required = replay_steps(costs, points.len()).map_err(ProgramError::custom)?;
        if required > options.maximum_steps() {
            return Err(ProgramError::custom(KernelSchedulingError::StepLimit {
                required,
                maximum: options.maximum_steps(),
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
                let result =
                    self.interpret_in_order(inputs.clone(), options, Some((&order, references.clone())), &mut trace);
                KernelInterleaving { result, trace }
            })
            .collect())
    }
}

impl Grid {
    /// Enumerates the merges of the instruction orders of all grid points iteratively, avoiding host recursion
    /// proportional to kernel size. Immediate predecessors along sequential axes suffice to enforce the grid's complete
    /// partial order.
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

/// Returns the number of replay steps that one order of a flat body takes at most across `points` grid points, which
/// is one body entry and the replay `costs` of the body instructions for every point. A portable or kernel memory
/// instruction without attached regions binds no other operation, so its cost is exactly one step, and an extension
/// operation costs its [`KernelExtension::replay_cost`], which must bound its failing bindings as well. An order that
/// fails stops early, so it takes fewer steps. Extensions of unknown cost are not admitted, because their rules may
/// bind further operations whose number is unknown before they run. A grid without points takes no steps, whatever
/// the costs of its instructions.
fn replay_steps(costs: impl IntoIterator<Item = usize>, points: usize) -> Result<usize, KernelSchedulingError> {
    if points == 0 {
        return Ok(0);
    }
    costs
        .into_iter()
        .try_fold(1usize, |steps, cost| steps.checked_add(cost))
        .and_then(|steps| steps.checked_mul(points))
        .ok_or(KernelSchedulingError::StepCountOverflow)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayIrOperation, ArrayOperation, ArrayReferenceTransform, ArrayReferenceTransformIndex, ArrayType,
        AttachedRegions, CompareOperation, ComparisonDirection, ConditionOperation, Context, DataType, Dimension,
        DimensionBounds, DimensionType, DimensionValue, InterpretationDriver, LogicalMesh, MeshAxis, MeshAxisType,
        Placeholder, ProgramBuilder, Provenance, ReferenceAccessDescriptor, ReferenceAccessOperation,
        ReferenceAtomicAddUpdate, ReferenceAtomicAddUpdateOperation, ReferenceRead, ReferenceWrite, RegionInterface,
        SelectOperation, Sharding, SubOperation, TypeError,
    };

    use crate::authoring::whole_array_parameter;
    use crate::calls::{KernelCallOperation, KernelParameter};
    use crate::grids::GridDimension;
    use crate::interpretation::KernelInterpretationError;
    use crate::mappings::BlockMapping;
    use crate::operations::KernelExtensionMemory;
    use crate::operations::tests::TestExtension;
    use crate::validation::KernelParameterAccess;

    use super::*;

    /// Two programs with opposite large updates followed by unit updates expose legal floating-point atomic orders.
    fn atomic_definition(execution: GridExecution) -> KernelDefinition {
        let mut mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        mapping.add_input(ArrayIrType::Dimension(DimensionType::new(
            "coordinate",
            DimensionBounds::non_negative(Some(2)).unwrap(),
        )));
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
    fn test_kernel_scheduling_error_display() {
        assert_eq!(
            KernelSchedulingError::Unsupported { reason: "masked windows" }.to_string(),
            "kernel interleaving exploration does not support masked windows",
        );
        assert_eq!(
            KernelSchedulingError::InterleavingLimit { maximum: 3 }.to_string(),
            "kernel interleaving count exceeds the limit 3",
        );
        assert_eq!(
            KernelSchedulingError::StepCountOverflow.to_string(),
            "kernel interleaving step count exceeds the host index range",
        );
        assert_eq!(
            KernelSchedulingError::StepLimit { required: 10, maximum: 9 }.to_string(),
            "kernel interleaving requires 10 replay steps, exceeding the limit 9",
        );
    }

    #[test]
    fn test_kernel_interleaving_result() {
        let definition = atomic_definition(GridExecution::Sequential);
        let executions = definition
            .interpret_interleavings(vec![Array::scalar(0f32).unwrap()], &KernelDebugOptions::default(), 1)
            .unwrap();
        assert_eq!(executions[0].result(), &Ok(vec![Array::scalar(1f32).unwrap()]));
    }

    #[test]
    fn test_kernel_interleaving_trace() {
        // The only order of a sequential grid replays the same bindings as ordinary traced interpretation.
        let definition = atomic_definition(GridExecution::Sequential);
        let input = Array::scalar(0f32).unwrap();
        let options = KernelDebugOptions::default();
        let executions = definition.interpret_interleavings(vec![input.clone()], &options, 1).unwrap();
        let mut trace = Vec::new();
        definition.interpret_with_trace(vec![input], &options, &mut trace).unwrap();
        assert_eq!(executions[0].trace(), trace.as_slice());
    }

    #[test]
    fn test_kernel_definition_interpret_interleavings() {
        let definition = atomic_definition(GridExecution::Parallel);
        let input = Array::scalar(0f32).unwrap();
        let options = KernelDebugOptions::default().with_maximum_programs(2).with_maximum_steps(10);
        let executions = definition.interpret_interleavings(vec![input.clone()], &options, 70).unwrap();
        assert_eq!(executions.len(), 70);
        let results = executions
            .iter()
            .map(|execution| {
                assert_eq!(execution.trace().len(), 8);
                execution.result().as_ref().unwrap()[0].elements::<f32>().unwrap()[0].to_bits()
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
        let error = definition.interpret_interleavings(vec![input], &options.with_maximum_steps(9), 70).unwrap_err();
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
        assert_eq!(executions[0].result(), &Ok(vec![Array::scalar(1f32).unwrap()]));
        assert_eq!(
            executions[0].trace().iter().map(|entry| entry.coordinate().to_vec()).collect::<Vec<_>>(),
            vec![vec![0], vec![0], vec![0], vec![0], vec![1], vec![1], vec![1], vec![1]],
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
    fn test_kernel_definition_interpret_interleavings_extension_replay_costs() {
        // The extension binds a nested `sub` through its interpretation rule and declares that cost, so the body
        // entry, the three body instructions, and the nested binding take five replay steps.
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![
                whole_array_parameter(ArrayType::scalar(DataType::F32), KernelParameterAccess::ReadOnly).unwrap(),
                whole_array_parameter(ArrayType::scalar(DataType::F32), KernelParameterAccess::WriteOnly).unwrap(),
            ],
        )
        .unwrap();
        let definition = KernelDefinition::<TestExtension>::trace(call, |(references, _coordinates)| {
            let input = references[0].read()?;
            let output = references[0]
                .context()
                .bind(KernelOperation::Extension(TestExtension::Rebinding), vec![], &[input])?
                .remove(0);
            references[1].write(&output)
        })
        .unwrap();
        let input = Array::scalar(3f32).unwrap();
        let options = KernelDebugOptions::default().with_maximum_steps(5);
        let mut trace = Vec::new();
        assert_eq!(
            definition.interpret_with_trace(vec![input.clone()], &options, &mut trace),
            Ok(vec![Array::scalar(0f32).unwrap()]),
        );
        assert_eq!(trace.len(), 4);
        let interleavings = definition.interpret_interleavings(vec![input.clone()], &options, 1).unwrap();
        assert_eq!(interleavings.len(), 1);
        assert_eq!(interleavings[0].result(), &Ok(vec![Array::scalar(0f32).unwrap()]));
        assert_eq!(
            interleavings[0].trace().iter().map(KernelTraceEntry::operation).collect::<Vec<_>>(),
            vec!["reference_read", "test_rebinding", "sub", "reference_write"],
        );

        // At a budget of four steps, planning fails with an outer error instead of running the order into its limit.
        let error = definition
            .interpret_interleavings(vec![input.clone()], &options.with_maximum_steps(4), 1)
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelSchedulingError>(),
            Some(&KernelSchedulingError::StepLimit { required: 5, maximum: 4 }),
        );

        // An extension of unknown replay cost is rejected before exploration, whatever the step budget.
        let call = KernelCallOperation::new(
            Grid::new(vec![]).unwrap(),
            vec![whole_array_parameter(ArrayType::scalar(DataType::F32), KernelParameterAccess::ReadWrite).unwrap()],
        )
        .unwrap();
        let definition = KernelDefinition::<TestExtension>::trace(call, |(references, _coordinates)| {
            references[0].context().bind(KernelOperation::Extension(TestExtension::Synchronous), vec![], &[])?;
            Ok(())
        })
        .unwrap();
        for maximum_steps in [1, usize::MAX] {
            let error = definition
                .interpret_interleavings(vec![input.clone()], &options.with_maximum_steps(maximum_steps), 1)
                .unwrap_err();
            assert_eq!(
                error.downcast_custom::<KernelSchedulingError>(),
                Some(&KernelSchedulingError::Unsupported { reason: "extension operations of unknown replay cost" }),
            );
        }
    }

    #[test]
    fn test_kernel_definition_interpret_interleavings_extension_failure_replay_costs() {
        // Returns its input in one step when that input is zero, and otherwise binds two diagnostic `sub` operations
        // before it fails, which takes three steps. It declares `cost` as its replay cost.
        #[derive(Clone, Debug)]
        struct DiagnosingExtension {
            cost: usize,
        }

        impl Operation for DiagnosingExtension {
            type Type = ArrayIrType;

            fn name(&self) -> &'static str {
                "test_diagnosing"
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayIrType],
                _region_interfaces: &[RegionInterface<ArrayIrType>],
            ) -> Result<Vec<ArrayIrType>, TypeError> {
                Ok(input_types.to_vec())
            }
        }

        impl ReferenceAccessOperation for DiagnosingExtension {
            type Transform = ArrayReferenceTransform;

            fn base_input_count(&self) -> usize {
                0
            }

            fn reference_access_descriptor(
                &self,
                _input_index: usize,
            ) -> Option<ReferenceAccessDescriptor<'_, ArrayReferenceTransform>> {
                None
            }

            fn with_reference_access_transforms(
                &self,
                _input_index: usize,
                _transforms: Vec<ArrayReferenceTransform>,
            ) -> Result<Self, ProgramError> {
                Err(ProgramError::UnsupportedOperation {
                    message: "test extension has no reference accesses".to_owned(),
                })
            }
        }

        impl KernelExtension for DiagnosingExtension {
            fn memory_semantics(&self) -> Result<KernelExtensionMemory, TypeError> {
                Ok(KernelExtensionMemory::Synchronous)
            }

            fn replay_cost(&self) -> Option<usize> {
                Some(self.cost)
            }
        }

        impl InterpretableOperation<EagerContext<ArrayIrValue<Array>, KernelOperation<DiagnosingExtension>>>
            for DiagnosingExtension
        {
            fn interpret<
                D: InterpretationDriver<EagerContext<ArrayIrValue<Array>, KernelOperation<DiagnosingExtension>>>,
            >(
                &self,
                context: &EagerContext<ArrayIrValue<Array>, KernelOperation<DiagnosingExtension>>,
                driver: &D,
                inputs: &[ArrayIrValue<Array>],
            ) -> Result<Vec<ArrayIrValue<Array>>, ProgramError> {
                let ArrayIrValue::Array(input) = &inputs[0] else { unreachable!() };
                if input.elements::<i32>()?[0] == 0 {
                    return Ok(inputs.to_vec());
                }
                for _ in 0..2 {
                    let operation =
                        KernelOperation::Portable(ArrayIrOperation::Array(ArrayOperation::Sub(SubOperation::new())));
                    driver.bind(context, operation, AttachedRegions::Empty, &[inputs[0].clone(), inputs[0].clone()])?;
                }
                Err(ProgramError::UnsupportedOperation { message: "nonzero diagnosing input".to_owned() })
            }
        }

        let definition = |cost: usize| {
            let call = KernelCallOperation::new(
                Grid::new(vec![]).unwrap(),
                vec![
                    whole_array_parameter(ArrayType::scalar(DataType::I32), KernelParameterAccess::ReadOnly).unwrap(),
                    whole_array_parameter(ArrayType::scalar(DataType::I32), KernelParameterAccess::WriteOnly).unwrap(),
                ],
            )
            .unwrap();
            KernelDefinition::<DiagnosingExtension>::trace(call, |(references, _coordinates)| {
                let input = references[0].read()?;
                let operation = KernelOperation::Extension(DiagnosingExtension { cost });
                let output = references[0].context().bind(operation, vec![], &[input])?.remove(0);
                references[1].write(&output)
            })
            .unwrap()
        };
        let operations = |interleaving: &KernelInterleaving| {
            interleaving.trace().iter().map(KernelTraceEntry::operation).collect::<Vec<_>>()
        };

        // A replay cost that bounds the failing binding too plans enough steps for the failing order to report its own
        // execution error. The body entry, the read, the three steps of the failing extension, and the write take six.
        let definition_with_bound = definition(3);
        let options = KernelDebugOptions::default().with_maximum_steps(6);
        let interleavings = definition_with_bound
            .interpret_interleavings(vec![Array::scalar(0i32).unwrap()], &options, 1)
            .unwrap();
        assert_eq!(interleavings[0].result(), &Ok(vec![Array::scalar(0i32).unwrap()]));
        assert_eq!(operations(&interleavings[0]), vec!["reference_read", "test_diagnosing", "reference_write"]);
        let interleavings = definition_with_bound
            .interpret_interleavings(vec![Array::scalar(1i32).unwrap()], &options, 1)
            .unwrap();
        assert_eq!(
            interleavings[0].result(),
            &Err(ProgramError::UnsupportedOperation { message: "nonzero diagnosing input".to_owned() }),
        );
        assert_eq!(operations(&interleavings[0]), vec!["reference_read", "test_diagnosing", "sub", "sub"]);
        let error = definition_with_bound
            .interpret_interleavings(vec![Array::scalar(1i32).unwrap()], &options.with_maximum_steps(5), 1)
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<KernelSchedulingError>(),
            Some(&KernelSchedulingError::StepLimit { required: 6, maximum: 5 }),
        );

        // A replay cost that covers only the successful binding violates that contract, so the failing order runs
        // into the step limit instead of reporting its execution error.
        let definition_without_bound = definition(1);
        let options = KernelDebugOptions::default().with_maximum_steps(4);
        let interleavings = definition_without_bound
            .interpret_interleavings(vec![Array::scalar(1i32).unwrap()], &options, 1)
            .unwrap();
        assert_eq!(
            interleavings[0].result().as_ref().unwrap_err().downcast_custom::<KernelInterpretationError>(),
            Some(&KernelInterpretationError::StepLimit {
                maximum_steps: 4,
                coordinate: vec![],
                operation: "sub",
                provenance: Provenance::unknown(),
            }),
        );
    }

    #[test]
    fn test_kernel_definition_interpret_interleavings_empty_grid() {
        // Declares the conservative replay cost `usize::MAX` and is never bound, because the grid has no points.
        #[derive(Clone, Debug)]
        struct UnboundedExtension;

        impl Operation for UnboundedExtension {
            type Type = ArrayIrType;

            fn name(&self) -> &'static str {
                "test_unbounded"
            }

            fn infer_output_types(
                &self,
                _input_types: &[ArrayIrType],
                _region_interfaces: &[RegionInterface<ArrayIrType>],
            ) -> Result<Vec<ArrayIrType>, TypeError> {
                Ok(vec![])
            }
        }

        impl ReferenceAccessOperation for UnboundedExtension {
            type Transform = ArrayReferenceTransform;

            fn base_input_count(&self) -> usize {
                0
            }

            fn reference_access_descriptor(
                &self,
                _input_index: usize,
            ) -> Option<ReferenceAccessDescriptor<'_, ArrayReferenceTransform>> {
                None
            }

            fn with_reference_access_transforms(
                &self,
                _input_index: usize,
                _transforms: Vec<ArrayReferenceTransform>,
            ) -> Result<Self, ProgramError> {
                Err(ProgramError::UnsupportedOperation {
                    message: "test extension has no reference accesses".to_owned(),
                })
            }
        }

        impl KernelExtension for UnboundedExtension {
            fn memory_semantics(&self) -> Result<KernelExtensionMemory, TypeError> {
                Ok(KernelExtensionMemory::Synchronous)
            }

            fn replay_cost(&self) -> Option<usize> {
                Some(usize::MAX)
            }
        }

        impl InterpretableOperation<EagerContext<ArrayIrValue<Array>, KernelOperation<UnboundedExtension>>>
            for UnboundedExtension
        {
            fn interpret<
                D: InterpretationDriver<EagerContext<ArrayIrValue<Array>, KernelOperation<UnboundedExtension>>>,
            >(
                &self,
                _context: &EagerContext<ArrayIrValue<Array>, KernelOperation<UnboundedExtension>>,
                _driver: &D,
                _inputs: &[ArrayIrValue<Array>],
            ) -> Result<Vec<ArrayIrValue<Array>>, ProgramError> {
                Err(ProgramError::UnsupportedOperation { message: "unbounded extension is never bound".to_owned() })
            }
        }

        let call = KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(0), GridExecution::Parallel)]).unwrap(),
            vec![],
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation<UnboundedExtension>>::new();
        builder.add_input(call.coordinate_types()[0].clone().into());
        builder
            .add_instruction(KernelOperation::Extension(UnboundedExtension), vec![], vec![], None)
            .unwrap();
        let definition =
            KernelDefinition::new(call, builder.build(vec![], vec![Placeholder], vec![]).unwrap()).unwrap();
        let options = KernelDebugOptions::default().with_maximum_steps(0);
        let mut trace = Vec::new();
        assert_eq!(definition.interpret_with_trace(vec![], &options, &mut trace), Ok(vec![]));
        assert_eq!(trace, Vec::<KernelTraceEntry>::new());
        let interleavings = definition.interpret_interleavings(vec![], &options, 1).unwrap();
        assert_eq!(interleavings.len(), 1);
        assert_eq!(interleavings[0].result(), &Ok(vec![]));
        assert_eq!(interleavings[0].trace(), &[]);
    }

    #[test]
    fn test_kernel_definition_interpret_interleavings_prefetch_placement() {
        // Copies a scalar prefetch into a write-only result. Ordinary interpretation accepts a prefetch that is placed
        // on a mesh, so exploration does as well, and its results take the types that the call infers from all of its
        // inputs (e.g., a result that varies along `x` when only the prefetch varies along `x`).
        let mut mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        mapping.add_input(DimensionType::new("offset", DimensionBounds::non_negative(Some(4)).unwrap()).into());
        let mapping = BlockMapping::new(
            mapping.build(vec![], vec![Placeholder], vec![]).unwrap(),
            vec![],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        let call = KernelCallOperation::new_with_prefetch(
            Grid::new(vec![]).unwrap(),
            vec![
                KernelParameter::new(ArrayType::scalar(DataType::I32), KernelParameterAccess::WriteOnly, mapping)
                    .unwrap(),
            ],
            vec![ArrayType::scalar(DataType::I32)],
        )
        .unwrap();
        let definition: KernelDefinition =
            KernelDefinition::trace_with_prefetch(call, |(references, _, values)| references[0].write(&values[0]))
                .unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let replicated = Sharding::replicated(mesh, 0);
        let bytes = Array::scalar(3i32).unwrap().logical_bytes();
        let options = KernelDebugOptions::default();
        for sharding in [replicated.clone(), replicated.with_varying_manual_axes(["x"]).unwrap()] {
            let input_type = ArrayType::scalar(DataType::I32).with_sharding(Some(sharding)).unwrap();
            let input = Array::from_logical_bytes(input_type.clone(), &bytes).unwrap();
            let output_types = definition.operation().output_types_for(&[ArrayIrType::Array(input_type)]).unwrap();
            let [ArrayIrType::Array(output_type)] = output_types.as_slice() else { unreachable!() };
            let expected = vec![Array::from_logical_bytes(output_type.clone(), &bytes).unwrap()];
            assert_eq!(definition.interpret(vec![input.clone()], 1), Ok(expected.clone()));
            let interleavings = definition.interpret_interleavings(vec![input], &options, 1).unwrap();
            assert_eq!(interleavings.len(), 1);
            assert_eq!(interleavings[0].result(), &Ok(expected));
        }
    }

    #[test]
    fn test_kernel_definition_interpret_interleavings_element_atomics() {
        // Both programs select the whole vector, and their atomic updates index the same single element of it.
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        builder.add_input(ArrayIrType::Dimension(DimensionType::new(
            "coordinate",
            DimensionBounds::non_negative(Some(2)).unwrap(),
        )));
        let zero = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(0).unwrap()));
        let mapping = BlockMapping::new(
            builder.build(vec![zero], vec![Placeholder], vec![Placeholder]).unwrap(),
            vec![2],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        let input = Array::vector(vec![0f32, 0f32]).unwrap();
        let call = KernelCallOperation::new(
            Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Parallel)]).unwrap(),
            vec![KernelParameter::new(input.r#type().into_owned(), KernelParameterAccess::ReadWrite, mapping).unwrap()],
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let reference = builder.add_input(call.parameters()[0].body_type());
        builder.add_input(call.coordinate_types()[0].clone().into());
        let update = builder.add_constant(ArrayIrValue::Array(Array::scalar(1f32).unwrap()));
        builder
            .add_instruction(
                ReferenceAtomicAddUpdateOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                    axis: 0,
                    index: ArrayReferenceTransformIndex::Static(0),
                }]),
                vec![],
                vec![reference, update],
                None,
            )
            .unwrap();
        let definition =
            KernelDefinition::new(call, builder.build(vec![], vec![Placeholder; 2], vec![]).unwrap()).unwrap();
        let executions = definition.interpret_interleavings(vec![input], &KernelDebugOptions::default(), 2).unwrap();
        assert_eq!(executions.len(), 2);
        let output = Array::vector(vec![2f32, 0f32]).unwrap();
        assert_eq!(executions[0].result(), &Ok(vec![output.clone()]));
        assert_eq!(
            executions[0]
                .trace()
                .iter()
                .map(|entry| (entry.coordinate().to_vec(), entry.operation()))
                .collect::<Vec<_>>(),
            vec![(vec![0], "reference_atomic_add_update"), (vec![1], "reference_atomic_add_update")],
        );
        assert_eq!(executions[1].result(), &Ok(vec![output]));
        assert_eq!(
            executions[1]
                .trace()
                .iter()
                .map(|entry| (entry.coordinate().to_vec(), entry.operation()))
                .collect::<Vec<_>>(),
            vec![(vec![1], "reference_atomic_add_update"), (vec![0], "reference_atomic_add_update")],
        );
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
                &KernelDebugOptions::default().with_maximum_programs(2).with_maximum_steps(4).with_nan_checks(true),
                2,
            )
            .unwrap();
        assert_eq!(executions.len(), 2);
        for (program, execution) in executions.iter().enumerate() {
            let error = execution.result().as_ref().unwrap_err();
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
            assert_eq!(execution.trace().len(), 1);
            assert_eq!(execution.trace()[0].operation(), "reference_atomic_add_update");
            assert_eq!(execution.trace()[0].coordinate(), vec![program]);
        }
        assert_eq!(input, Array::scalar(0f32).unwrap());
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
    fn test_replay_steps() {
        assert_eq!(replay_steps([1; 4], 2), Ok(10));
        assert_eq!(replay_steps([1, 2, 1], 1), Ok(5));
        assert_eq!(replay_steps([], 3), Ok(3));
        assert_eq!(replay_steps([1; 3], 0), Ok(0));
        assert_eq!(replay_steps([], usize::MAX), Ok(usize::MAX));
        assert_eq!(replay_steps([usize::MAX - 1], 1), Ok(usize::MAX));
        assert_eq!(replay_steps([1], usize::MAX), Err(KernelSchedulingError::StepCountOverflow));
        assert_eq!(replay_steps([usize::MAX], 1), Err(KernelSchedulingError::StepCountOverflow));
        assert_eq!(replay_steps([usize::MAX], 0), Ok(0));
        assert_eq!(replay_steps([usize::MAX - 1, 1], 0), Ok(0));
    }
}
