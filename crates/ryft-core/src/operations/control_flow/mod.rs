pub mod condition;
pub mod scan;
pub mod select;
pub mod r#while;

use crate::programs::{Operation, Type, TypeError};

pub use condition::{CONDITION_OPERATION_NAME, ConditionOperation, transpose_primal_condition};
pub use scan::{SCAN_OPERATION_NAME, ScanOperation, ScanReferenceDischarge, transpose_primal_scan};
pub use select::{SELECT_OPERATION_NAME, Select, SelectOperation};
pub use r#while::{WHILE_OPERATION_NAME, WhileOperation, WhilePredicate, WhileTypeSemantics};
pub(crate) use r#while::{WhileResidualStackOperation, WhileResidualStackType};

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

#[cfg(test)]
pub(crate) mod tests {
    use std::cell::Cell;

    use crate::arrays::{Array, ArrayIrType, ArrayIrValue};
    use crate::axes::Axis;
    use crate::batching::{
        BatchAxis, BatchingContext, BatchingDriver, BatchingError, ProgramBatchingOutputAxesPolicy,
        RecursiveBatchingDriver, RecursiveBatchingPolicy,
    };
    use crate::captures::CaptureReference;
    use crate::contexts::Context;
    use crate::parameters::Placeholder;
    use crate::programs::{Atom, Operation, Program, Region, RegionDriver, RegionRef, Value};

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
        fn batch_region(
            &self,
            context: &BatchingContext<C, P>,
            index: usize,
            inputs: Vec<P::Batch>,
        ) -> Result<Vec<P::Batch>, BatchingError> {
            RecursiveBatchingDriver::new(self.regions).batch_region(context, index, inputs)
        }

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
