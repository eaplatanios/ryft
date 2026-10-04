use crate::arrays::{ArrayIrType, ArrayType};
use crate::operations::dimensions::dimension_from_scalar::DimensionFromScalarOperation;
use crate::operations::dimensions::dimension_to_scalar::{DIMENSION_DATA_TYPE, DimensionToScalarOperation};
use crate::programs::{Operation, Type, TypeError, TypeIdentityPosition, TypeRefinements};

// TODO(eaplatanios): Review this module and also add a module docstring that follows our established conventions.

pub mod condition;
pub mod scan;
pub mod select;
pub mod r#while;

pub use condition::{CONDITION_OPERATION_NAME, ConditionOperation, transpose_primal_condition};
pub use scan::{SCAN_OPERATION_NAME, ScanOperation};
pub use select::{SELECT_OPERATION_NAME, Select, SelectOperation};
pub use r#while::{WHILE_OPERATION_NAME, WhileOperation, WhilePredicate, WhileType};

/// Type-family storage policy for values that must cross an iteration boundary as stacked residuals.
///
/// Array residuals store themselves directly. A composite backend can instead assign a checked array-backed storage
/// type to metadata values such as first-class dimensions, keeping the temporal representation explicit in SSA.
pub(crate) trait TemporalResidualType: Type {
    /// Returns the per-iteration array-backed storage type for this residual.
    fn temporal_storage_type(&self) -> Result<Self, TypeError>;
}

/// Operation-family conversions paired with [`TemporalResidualType`].
///
/// Returning `None` means the residual already uses its storage representation. Returning an operation makes the
/// conversion visible in the generated program before stacking or after slicing one iteration's stored value.
pub(crate) trait TemporalResidualOperation<T: TemporalResidualType>: Operation<Type = T> {
    /// Returns the operation that converts a residual to temporal storage, if conversion is required.
    fn residual_to_storage(residual_type: &T) -> Result<Option<Self>, TypeError>;

    /// Returns the operation that restores a residual from temporal storage, if conversion is required.
    fn residual_from_storage(residual_type: &T) -> Result<Option<Self>, TypeError>;
}

impl TemporalResidualType for ArrayType {
    #[inline]
    fn temporal_storage_type(&self) -> Result<Self, TypeError> {
        Ok(self.clone())
    }
}

impl<O: Operation<Type = ArrayType>> TemporalResidualOperation<ArrayType> for O {
    #[inline]
    fn residual_to_storage(_residual_type: &ArrayType) -> Result<Option<Self>, TypeError> {
        Ok(None)
    }

    #[inline]
    fn residual_from_storage(_residual_type: &ArrayType) -> Result<Option<Self>, TypeError> {
        Ok(None)
    }
}

// Composite array IR residuals store arrays directly and first-class dimensions as scalar arrays. A reference never
// defines temporal storage because the transforms thread references through loops as carries and never save them as
// residuals.
impl TemporalResidualType for ArrayIrType {
    #[inline]
    fn temporal_storage_type(&self) -> Result<Self, TypeError> {
        Ok(match self {
            Self::Array(r#type) => Self::Array(r#type.clone()),
            Self::Dimension(_) => Self::Array(ArrayType::scalar(DIMENSION_DATA_TYPE)),
            Self::Reference(_) => {
                return Err(TypeError::invalid(
                    "a reference cannot be stored as a temporal residual; references are threaded as carries",
                ));
            }
        })
    }
}

impl<O> TemporalResidualOperation<ArrayIrType> for O
where
    O: Operation<Type = ArrayIrType> + From<DimensionFromScalarOperation> + From<DimensionToScalarOperation>,
{
    fn residual_to_storage(residual_type: &ArrayIrType) -> Result<Option<Self>, TypeError> {
        Ok(match residual_type {
            ArrayIrType::Array(_) => None,
            ArrayIrType::Dimension(_) => Some(Self::from(DimensionToScalarOperation)),
            ArrayIrType::Reference(_) => {
                return Err(TypeError::invalid(
                    "a reference cannot be stored as a temporal residual; references are threaded as carries",
                ));
            }
        })
    }

    fn residual_from_storage(residual_type: &ArrayIrType) -> Result<Option<Self>, TypeError> {
        Ok(match residual_type {
            ArrayIrType::Array(_) => None,
            ArrayIrType::Dimension(r#type) => {
                Some(Self::from(DimensionFromScalarOperation::new(r#type.variable().clone())))
            }
            ArrayIrType::Reference(_) => {
                return Err(TypeError::invalid(
                    "a reference cannot be stored as a temporal residual; references are threaded as carries",
                ));
            }
        })
    }
}

/// Returns the output types of a control-flow operation whose `input_types` refine the `declared_input_types` of its
/// regions, by applying the refinement facts that those inputs establish (e.g., `rows = 3` for an `f32[3]` input that
/// a region declares as `f32[rows]`) to the `declared_output_types`. The facts are established from the complete input
/// signature, so conflicting facts are rejected, but identities that an output type defines stay symbolic: a loop's
/// first-class dimension carry may change its extent across iterations, and a branch produces the identities that its
/// outputs define. A reference output whose aliased input is known (i.e., for which `aliased_input` returns that
/// input's index) takes exactly that input's type, so an alias family never mixes refined and declared reference
/// types, while any other reference output keeps its declared type. Metadata that is not a fact about an identity
/// (e.g., a sharding or layout that only the inputs carry) is never propagated to the outputs.
pub(crate) fn refine_output_types<T: Type>(
    declared_input_types: &[T],
    input_types: &[T],
    declared_output_types: &[T],
    aliased_input: impl Fn(usize) -> Option<usize>,
) -> Result<Vec<T>, TypeError> {
    let refinements = T::Refinements::establish(declared_input_types, input_types)?;
    let symbolic_identities = declared_output_types
        .iter()
        .flat_map(Type::identities)
        .filter_map(|(position, identity)| (position == TypeIdentityPosition::Definition).then(|| identity.clone()))
        .collect::<Vec<_>>();
    declared_output_types
        .iter()
        .enumerate()
        .map(|(index, declared_output_type)| {
            if declared_output_type.is_reference() {
                return Ok(aliased_input(index)
                    .map_or_else(|| declared_output_type.clone(), |input_index| input_types[input_index].clone()));
            }
            refinements.refine(declared_output_type, symbolic_identities.as_slice())
        })
        .collect()
}

/// Validates that every type identity that one of the `output_types` of a control-flow operation refers to is either
/// carried by one of its `input_types` or defined by one of its `output_types`, so that the instruction only produces
/// types whose identities it consumes or defines. [`refine_output_types`] replaces an identity that the inputs refine
/// away with its static extent, unless an output defines that identity, so this holds for the refined output types by
/// construction and is checked as a final invariant of `while`, `scan`, and `condition` type inference.
pub(crate) fn validate_output_identities<T: Type>(
    operation_name: &str,
    input_types: &[T],
    output_types: &[T],
) -> Result<(), TypeError> {
    let defined_identities = output_types
        .iter()
        .flat_map(Type::identities)
        .filter_map(|(position, identity)| (position == TypeIdentityPosition::Definition).then_some(identity))
        .collect::<Vec<_>>();
    let input_identities =
        input_types.iter().flat_map(Type::identities).map(|(_, identity)| identity).collect::<Vec<_>>();
    for (index, output_type) in output_types.iter().enumerate() {
        if let Some((_, identity)) = output_type
            .identities()
            .find(|(_, identity)| !defined_identities.contains(identity) && !input_identities.contains(identity))
        {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` output {index} has type `{output_type}`, which refers to the identity `{identity}` \
                 that no input carries and no output defines",
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use std::cell::Cell;

    use crate::arrays::{Array, ArrayIrType, ArrayIrValue, DimensionType, DimensionValue};
    use crate::axes::Axis;
    use crate::batching::{
        BatchAxis, BatchingContext, BatchingDriver, BatchingError, ProgramBatchingOutputAxesPolicy,
        RecursiveBatchingDriver, RecursiveBatchingPolicy,
    };
    use crate::captures::CaptureReference;
    use crate::contexts::Context;
    use crate::parameters::Placeholder;
    use crate::programs::{Atom, Operation, Program, Region, RegionDriver, RegionRef, Value};

    /// Wraps an [`Array`] as an array [`ArrayIrValue`].
    pub(crate) fn array(value: Array) -> ArrayIrValue<Array> {
        ArrayIrValue::Array(value)
    }

    /// Returns a first-class dimension [`ArrayIrValue`] of type `r#type` whose runtime extent is `extent`.
    pub(crate) fn dimension(r#type: &DimensionType, extent: usize) -> ArrayIrValue<Array> {
        ArrayIrValue::Dimension(DimensionValue::new(r#type.clone(), extent).unwrap())
    }

    /// Replaces every capture constant of a capture-lifted `program` with the concrete capture value it names, so that
    /// control-flow tests can interpret the discharged form of a captured program eagerly. Lifting turns the entry
    /// region's capture constants into leading inputs but keeps the capture constants of attached regions, which a
    /// backend resolves against the same leading capture arguments while lowering, and eager interpretation has no
    /// capture table to resolve them against.
    pub(crate) fn resolve_captures<O: Operation<Type = ArrayIrType>>(
        program: &Program<
            CaptureReference<ArrayIrType>,
            O,
            Vec<CaptureReference<ArrayIrType>>,
            Vec<CaptureReference<ArrayIrType>>,
        >,
        captures: &[ArrayIrValue<Array>],
    ) -> Program<ArrayIrValue<Array>, O, Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>> {
        let regions = program
            .regions()
            .iter()
            .map(|region| {
                let atoms = region
                    .atoms()
                    .iter()
                    .map(|atom| match atom {
                        Atom::Constant(capture) => Atom::Constant(captures[capture.index()].clone()),
                        Atom::Variable(r#type) => Atom::Variable(r#type.clone()),
                    })
                    .collect();
                Region::new(
                    atoms,
                    region.input_ids().to_vec(),
                    region.output_ids().to_vec(),
                    region.instructions().to_vec(),
                )
            })
            .collect();
        Program::new(
            vec![Placeholder; program.input_count()],
            vec![Placeholder; program.output_count()],
            regions,
            program.entry(),
        )
        .unwrap()
    }

    /// [`BatchingDriver`] that counts the structural [`batch_program`](BatchingDriver::batch_program) requests a
    /// region-carrying batching rule makes, delegating every request to the ordinary [`RecursiveBatchingDriver`] over
    /// the same regions. Region-carrying rules discover their nested programs' natural output axes before instantiating
    /// them at reconciled targets, and reuse a discovery program when its axes already match. This fixture lets rule
    /// tests pin how many structural passes such a rule actually performs, which a program rendering alone cannot
    /// observe.
    pub(crate) struct CountingBatchingDriver<'r, V: Value, O: Operation<Type = V::Type>> {
        /// [`Region`](crate::programs::Region)s attached to the operation application under test,
        /// in operation-defined order.
        regions: &'r Vec<Program<V, O, Vec<V>, Vec<V>>>,

        /// Number of structural program-batching requests observed so far.
        batch_program_calls: Cell<usize>,
    }

    impl<'r, V: Value, O: Operation<Type = V::Type>> CountingBatchingDriver<'r, V, O> {
        /// Creates a new [`CountingBatchingDriver`] over the provided attached regions.
        pub(crate) fn new(regions: &'r Vec<Program<V, O, Vec<V>, Vec<V>>>) -> Self {
            Self { regions, batch_program_calls: Cell::new(0) }
        }

        /// Returns the number of structural program-batching requests observed so far.
        pub(crate) fn batch_program_calls(&self) -> usize {
            self.batch_program_calls.get()
        }
    }

    impl<V: Value, O: Operation<Type = V::Type>> RegionDriver<V, O> for CountingBatchingDriver<'_, V, O> {
        fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, V, O>>
        where
            V: 'r,
            O: 'r,
        {
            self.regions.regions()
        }
    }

    impl<C: Context, P: RecursiveBatchingPolicy<C>> BatchingDriver<C, P>
        for CountingBatchingDriver<'_, C::Constant, C::Operation>
    {
        fn batch_program(
            &self,
            context: &BatchingContext<C, P>,
            region: RegionRef<'_, C::Constant, C::Operation>,
            input_axes: &[BatchAxis],
            output_axes_policy: ProgramBatchingOutputAxesPolicy,
        ) -> Result<P::BatchedProgram, BatchingError> {
            self.batch_program_calls.set(self.batch_program_calls.get() + 1);
            RecursiveBatchingDriver::new(self.regions).batch_program(context, region, input_axes, output_axes_policy)
        }

        fn batch_region(
            &self,
            context: &BatchingContext<C, P>,
            index: usize,
            inputs: Vec<P::Batch>,
        ) -> Result<Vec<P::Batch>, BatchingError> {
            RecursiveBatchingDriver::new(self.regions).batch_region(context, index, inputs)
        }

        fn restore_batch(
            &self,
            value: C::Value,
            batch_axis: BatchAxis,
            r#type: &C::Type,
            inputs: &[P::Batch],
        ) -> Result<P::Batch, BatchingError> {
            P::restore_batch(value, batch_axis, r#type, inputs)
        }

        fn align_batch_axis(
            &self,
            context: &BatchingContext<C, P>,
            batch: P::Batch,
            axis: Axis,
        ) -> Result<P::Batch, BatchingError> {
            P::align_batch_axis(context, batch, axis)
        }
    }
}
