use std::fmt::Display;

use crate::batching::{ArrayBatch, BatchableOperation, BatchingError};
use crate::contexts::Context;
use crate::differentiation::{DifferentiableOperation, DifferentiationError, TransposableOperation};
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::constants::{
    Constant, ConstantOperation, Fill, FillOperation, Iota, IotaOperation, One, OneLikeOperation, OneOperation, Zero,
    ZeroLikeOperation, ZeroOperation,
};
use crate::partial::PartialValue;
use crate::programs::{MaybeZero, Value};
use crate::tracing::{Tracer, TracingContext};

use crate::batching::{BatchingContext, BatchingDriver};
use crate::differentiation::{DifferentiationDriver, DifferentiationDual, TranspositionDriver};
use crate::types::{ArrayType, Type, Typed};

/// [`ZeroOperation`] takes no inputs and produces a constant of its captured type. The same
/// constant is the right value for every batch item, so the rule interprets the operation once
/// under the parent context — constructing the constant eagerly under an eager parent and
/// staging a nullary operation under a staging parent — and wraps each output as a replicated
/// [`ArrayBatch`] (`batch_axis = None`). Downstream elementwise consumers that need the constant
/// materialized at the batched physical shape will broadcast it through the internal elementwise
/// batching rule.
impl<C: Context<Type = ArrayType> + Zero<C::Value>> BatchableOperation<C> for ZeroOperation<ArrayType> {
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        _inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        let outputs = self.interpret(&context.parent().clone(), &crate::EmptyRegionDriver, &[])?;
        Ok(outputs.into_iter().map(ArrayBatch::replicated).collect())
    }
}

/// See [`ZeroOperation`]'s impl above for the reasoning — [`OneOperation`] is replicated by the
/// same argument.
impl<C: Context<Type = ArrayType> + One<C::Value>> BatchableOperation<C> for OneOperation<ArrayType> {
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        _inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        let outputs = self.interpret(&context.parent().clone(), &crate::EmptyRegionDriver, &[])?;
        Ok(outputs.into_iter().map(ArrayBatch::replicated).collect())
    }
}

/// See [`ZeroOperation`]'s impl above for the reasoning — [`ConstantOperation`] is also replicated because it has no
/// data inputs. The stored constant type is decoupled from the flowing value type so the same rule serves both eager
/// batching (where the two coincide) and staged batching (where the stored constant lifts into a tracer).
impl<Stored, C> BatchableOperation<C> for ConstantOperation<Stored>
where
    Stored: Clone + Display + Typed<Type = ArrayType>,
    C: Context<Type = ArrayType> + Constant<C::Value, Stored>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        _inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        let outputs = self.interpret(&context.parent().clone(), &crate::EmptyRegionDriver, &[])?;
        Ok(outputs.into_iter().map(ArrayBatch::replicated).collect())
    }
}

/// See [`ZeroOperation`]'s impl above for the reasoning — [`FillOperation`] is also replicated because it has no
/// data inputs.
impl<F: Clone + Display, C: Context<Type = ArrayType> + Fill<F, C::Value>> BatchableOperation<C>
    for FillOperation<ArrayType, F>
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        _inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        let outputs = self.interpret(&context.parent().clone(), &crate::EmptyRegionDriver, &[])?;
        Ok(outputs.into_iter().map(ArrayBatch::replicated).collect())
    }
}

/// See [`ZeroOperation`]'s impl above for the reasoning — [`IotaOperation`] is also replicated because it has no data
/// inputs; a raw iota of a fixed type is the same value for every batch item. (The per-item batch index produced by
/// `axis_index` is materialized directly against the mapped axis instead of relying on this replicated rule.)
impl<C: Context<Type = ArrayType> + Iota<C::Value>> BatchableOperation<C> for IotaOperation<ArrayType> {
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        _inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        let outputs = self.interpret(&context.parent().clone(), &crate::EmptyRegionDriver, &[])?;
        Ok(outputs.into_iter().map(ArrayBatch::replicated).collect())
    }
}

impl<T: Type, V: Value<Type = T>, O: Operation<T>> TransposableOperation<V, O> for ZeroOperation<T> {
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("output", outputs, 1, ProgramError);
        Ok(Vec::new())
    }
}

impl<T: Type, V: Value<Type = T>, O: Operation<T>> TransposableOperation<V, O> for OneOperation<T> {
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("output", outputs, 1, ProgramError);
        Ok(Vec::new())
    }
}

impl<T, V, O, F, Mode> TransposableOperation<V, O> for ConstantOperation<F, Mode>
where
    T: Type,
    V: Value<Type = T>,
    O: Operation<T>,
    F: Clone + Display + Typed<Type = T>,
    Mode: Clone,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("output", outputs, 1, ProgramError);
        Ok(Vec::new())
    }
}

impl<V: Value, O: Operation<V::Type>> TransposableOperation<V, O> for ZeroLikeOperation {
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![MaybeZero::Zero(inputs[0].r#type().into_owned())])
    }
}

impl<V: Value, O: Operation<V::Type>> TransposableOperation<V, O> for OneLikeOperation {
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![MaybeZero::Zero(inputs[0].r#type().into_owned())])
    }
}

impl<T, V, O, F> TransposableOperation<V, O> for FillOperation<T, F>
where
    T: Type,
    V: Value<Type = T>,
    O: Operation<T>,
    F: Clone + Display,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("output", outputs, 1, ProgramError);
        Ok(Vec::new())
    }
}

impl<T: Type, V: Value<Type = T>, O: Operation<T>> TransposableOperation<V, O> for IotaOperation<T> {
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("output", outputs, 1, ProgramError);
        Ok(Vec::new())
    }
}

/// Forward-mode rule for [`ZeroOperation`]: the nullary constant is replayed to synthesize the primal value and
/// paired with a typed zero tangent, since constants carry no tangent.
impl<C: Context> DifferentiableOperation<C> for ZeroOperation<C::Type>
where
    C::Operation: From<ZeroOperation<C::Type>>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The outputs carry no tangent: replay the primal operation on the input primals and pair each output
        // with a structural zero tangent, which stays symbolic and stages nothing.
        let primal_inputs = inputs.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
        Ok(context
            .bind(self.clone(), Vec::new(), &primal_inputs)?
            .into_iter()
            .map(DifferentiationDual::new_with_zero_tangent)
            .collect())
    }
}

/// Forward-mode rule for [`OneOperation`]: the nullary constant is replayed to synthesize the primal value and
/// paired with a typed zero tangent, since constants carry no tangent.
impl<C: Context> DifferentiableOperation<C> for OneOperation<C::Type>
where
    C::Operation: From<OneOperation<C::Type>>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The outputs carry no tangent: replay the primal operation on the input primals and pair each output
        // with a structural zero tangent, which stays symbolic and stages nothing.
        let primal_inputs = inputs.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
        Ok(context
            .bind(self.clone(), Vec::new(), &primal_inputs)?
            .into_iter()
            .map(DifferentiationDual::new_with_zero_tangent)
            .collect())
    }
}

/// Forward-mode rule for [`ConstantOperation`]: the nullary constant is replayed to synthesize the primal value
/// and paired with a typed zero tangent, since constants carry no tangent.
impl<C: Context> DifferentiableOperation<C> for ConstantOperation<C::Constant>
where
    C::Operation: From<ConstantOperation<C::Constant>>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The outputs carry no tangent: replay the primal operation on the input primals and pair each output
        // with a structural zero tangent, which stays symbolic and stages nothing.
        let primal_inputs = inputs.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
        Ok(context
            .bind(self.clone(), Vec::new(), &primal_inputs)?
            .into_iter()
            .map(DifferentiationDual::new_with_zero_tangent)
            .collect())
    }
}

/// Forward-mode rule for [`ZeroLikeOperation`]: the exemplar-derived constant is replayed (its primal is
/// `zero_like` of the exemplar) and paired with a typed zero tangent regardless of the exemplar's tangent.
impl<C: Context> DifferentiableOperation<C> for ZeroLikeOperation
where
    C::Operation: From<ZeroLikeOperation>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The outputs carry no tangent: replay the primal operation on the input primals and pair each output
        // with a structural zero tangent, which stays symbolic and stages nothing.
        let primal_inputs = inputs.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
        Ok(context
            .bind(*self, Vec::new(), &primal_inputs)?
            .into_iter()
            .map(DifferentiationDual::new_with_zero_tangent)
            .collect())
    }
}

/// Forward-mode rule for [`OneLikeOperation`]: the exemplar-derived constant is replayed (its primal is
/// `one_like` of the exemplar) and paired with a typed zero tangent regardless of the exemplar's tangent.
impl<C: Context> DifferentiableOperation<C> for OneLikeOperation
where
    C::Operation: From<OneLikeOperation>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The outputs carry no tangent: replay the primal operation on the input primals and pair each output
        // with a structural zero tangent, which stays symbolic and stages nothing.
        let primal_inputs = inputs.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
        Ok(context
            .bind(*self, Vec::new(), &primal_inputs)?
            .into_iter()
            .map(DifferentiationDual::new_with_zero_tangent)
            .collect())
    }
}

/// Forward-mode rule for [`FillOperation`]: the nullary constant is replayed to synthesize the filled primal
/// value and paired with a typed zero tangent, since constants carry no tangent.
impl<C: Context, F: Clone + Display> DifferentiableOperation<C> for FillOperation<C::Type, F>
where
    C::Operation: From<FillOperation<C::Type, F>>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The outputs carry no tangent: replay the primal operation on the input primals and pair each output
        // with a structural zero tangent, which stays symbolic and stages nothing.
        let primal_inputs = inputs.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
        Ok(context
            .bind(self.clone(), Vec::new(), &primal_inputs)?
            .into_iter()
            .map(DifferentiationDual::new_with_zero_tangent)
            .collect())
    }
}

/// Forward-mode rule for [`IotaOperation`]: the nullary index constant is replayed to synthesize the primal value and
/// paired with a typed zero tangent, since constants carry no tangent.
impl<C: Context> DifferentiableOperation<C> for IotaOperation<C::Type>
where
    C::Operation: From<IotaOperation<C::Type>>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The outputs carry no tangent: replay the primal operation on the input primals and pair each output
        // with a structural zero tangent, which stays symbolic and stages nothing.
        let primal_inputs = inputs.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
        Ok(context
            .bind(self.clone(), Vec::new(), &primal_inputs)?
            .into_iter()
            .map(DifferentiationDual::new_with_zero_tangent)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;

    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::contexts::{Context, EagerContext};
    use crate::operations::math::{Cos, Sin};
    use crate::programs::Program;

    #[test]
    fn float_ext_matches_scalar_intrinsics() {
        let angle = 0.75f64;
        assert_eq!(Scalar::from(angle).sin().unwrap(), Scalar::from(angle.sin()));
        assert_eq!(Scalar::from(angle).cos().unwrap(), Scalar::from(angle.cos()));

        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (_, compiled): (Scalar, Program<Scalar, ScalarOperation<Scalar>, Scalar, Scalar>) =
            domain.interpret_and_trace(|x| Ok(x.sin()?), Scalar::from(2.0)).unwrap();

        assert_eq!(
            compiled.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = sin %0
                in (%1)
            "}
            .trim_end(),
        );
    }
}
