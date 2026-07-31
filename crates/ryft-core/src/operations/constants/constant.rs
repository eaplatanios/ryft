use std::borrow::Cow;
use std::fmt::{Debug, Display};

use crate::batching::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, BatchAxis, BatchableOperation, BatchingContext, BatchingDriver,
    BatchingError, BatchingTracer,
};
use crate::contexts::{Context, Domain, EagerContext, ProjectedContext, StagingContext};
use crate::differentiation::forward::{DifferentiationContext, DifferentiationDual, DifferentiationTracer};
use crate::differentiation::types::DifferentiableType;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_differentiable_operation, impl_nullary_transposable_operation};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::ProgramError;
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::operations::{Operation, OperationFormatter, OperationProjection};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::{Value, ValueProjection};
use crate::tracing::Tracer;
use crate::types::ArrayType;

/// Canonical operation name for [`ConstantOperation`].
pub const CONSTANT_OPERATION_NAME: &str = "constant";

/// [`Operation`] that has no inputs and produces a single output equal to a stored typed literal. It carries a `V`
/// [`Value`], so its output type is exactly the literal value's type and interpreting it materializes that value
/// through the active context. Program capture references are normally represented as constant [`Atom`](crate::Atom)s
/// instead (they name runtime values in a side table and are not literal operations).
#[derive(Copy, Clone)]
pub struct ConstantOperation<V: Value> {
    /// Literal value produced by this [`Operation`] when interpreted.
    value: V,
}

impl<V: Value> ConstantOperation<V> {
    /// Creates a new [`ConstantOperation`] storing the provided typed literal.
    #[inline]
    pub fn new(value: V) -> Self {
        Self { value }
    }

    /// Returns the type of the value produced by this operation.
    #[inline]
    pub fn r#type(&self) -> Cow<'_, V::Type> {
        self.value.r#type()
    }

    /// Returns the literal value produced by this operation.
    #[inline]
    pub fn value(&self) -> &V {
        &self.value
    }
}

impl<V: Value> Debug for ConstantOperation<V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("ConstantOperation").field("value", &self.value).finish()
    }
}

impl<V: Value> Display for ConstantOperation<V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<V: Value> Operation<V::Type> for ConstantOperation<V> {
    #[inline]
    fn name(&self) -> &'static str {
        CONSTANT_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[V::Type],
        _region_interfaces: &[RegionInterface<V::Type>],
    ) -> Result<Vec<V::Type>, TypeError> {
        check_count!("input", input_types, 0, TypeError);
        Ok(vec![self.value.r#type().into_owned()])
    }

    #[inline]
    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<V::Type as crate::Type>::Identity>,
    ) -> Result<Self, TypeError> {
        Ok(Self::new(self.value.rename_type_identities(renaming)?))
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, CONSTANT_OPERATION_NAME)?
            .bracketed(|operation| operation.field("value", &self.value))
    }
}

impl<Stored: Value, C: Domain<Type = Stored::Type> + Constant<C::Value, Stored>> InterpretableOperation<C>
    for ConstantOperation<Stored>
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 0, ProgramError);
        Ok(vec![context.constant(self.value.clone())?])
    }
}

impl<V: Value, C: Context<Type = V::Type, Operation: From<ConstantOperation<V>>>> PartiallyEvaluatableOperation<C>
    for ConstantOperation<V>
{
}

impl_non_differentiable_operation!(<V> ConstantOperation<V> where V: Value);
impl_nullary_transposable_operation!(<V> ConstantOperation<V> where V: Value);

impl<
    Stored: Value<Type = ArrayType>,
    C: Context<Type = ArrayType, Operation: From<ConstantOperation<Stored>>>,
    P: ArrayBatchingPolicy<C>,
> BatchableOperation<C, ArrayBatching<P>> for ConstantOperation<Stored>
{
    #[inline]
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 0, ProgramError);
        Ok(context
            .parent()
            .bind(self.clone(), Vec::new(), &[])?
            .into_iter()
            .map(ArrayBatch::replicated)
            .collect())
    }
}

/// Represents the ability to materialize a stored [`ConstantOperation`] payload and is typically implemented by
/// [`Context`]s. [`Constant`] is the literal value counterpart to [`Zero`](crate::Zero), [`One`](crate::One), and
/// [`Fill`](crate::Fill). It typically lives on [`Context`]s because producing a runtime value from a stored payload
/// can be context-dependent. For example, [`EagerContext`]s can return the value directly while [`StagingContext`]s
/// record a builder constant.
pub trait Constant<V, C> {
    /// Returns the runtime value represented by `value`.
    fn constant(&self, value: C) -> Result<V, ProgramError>;
}

impl<V: Value, O: Operation<V::Type>> Constant<V, V> for EagerContext<V, O> {
    #[inline]
    fn constant(&self, value: V) -> Result<V, ProgramError> {
        Ok(value)
    }
}

impl<C: Context, T: Type>
    Constant<<C::Value as ValueProjection<T>>::Projected, <C::Constant as ValueProjection<T>>::Projected>
    for ProjectedContext<C, T>
where
    C::Value: ValueProjection<T, Projected: Value<Type = T>>,
    C::Constant: ValueProjection<T, Projected: Value<Type = T>>,
    C::Operation: OperationProjection<T>,
{
    #[inline]
    fn constant(
        &self,
        value: <C::Constant as ValueProjection<T>>::Projected,
    ) -> Result<<C::Value as ValueProjection<T>>::Projected, ProgramError> {
        self.lift(value)
    }
}

impl<C: StagingContext> Constant<Tracer<C>, C::Constant> for C {
    #[inline]
    fn constant(&self, value: C::Constant) -> Result<Tracer<C>, ProgramError> {
        Ok(StagingContext::constant(self, value))
    }
}

impl<C: Context<Type = ArrayType> + Constant<C::Value, Stored>, Stored>
    Constant<BatchingTracer<C, ArrayBatching>, Stored> for BatchingContext<C, ArrayBatching>
{
    fn constant(&self, value: Stored) -> Result<BatchingTracer<C, ArrayBatching>, ProgramError> {
        let value = self.parent().constant(value)?;
        let r#type = value.r#type().into_owned();
        let batch = ArrayBatch::new(r#type, value, BatchAxis::replicated())?;
        Ok(BatchingTracer::new(self.clone(), batch))
    }
}

impl<C: Context<Type: DifferentiableType>> Constant<DifferentiationTracer<C>, C::Constant>
    for DifferentiationContext<C>
{
    #[inline]
    fn constant(&self, value: C::Constant) -> Result<DifferentiationTracer<C>, ProgramError> {
        let dual = DifferentiationDual::new_with_zero_tangent(self.parent().lift(value)?);
        Ok(DifferentiationTracer::new(dual, self.clone()))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::parameters::Placeholder;
    use crate::programs::atoms::{Atom, AtomId};
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::operations::Operation;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::tracing::DomainTracingContext;
    use crate::types::DataType;

    use super::*;

    #[test]
    fn test_constant() {
        // Verify the operation's literal value, identity, and rendering.
        let operation = ConstantOperation::<Scalar>::new(Scalar::from(3.5));
        assert_eq!(operation.name(), CONSTANT_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "constant [value=3.5]");
        assert_eq!(operation.value(), &Scalar::from(3.5));
        assert_eq!(operation.infer_output_types(&[], &[]), Ok(vec![DataType::F64]));

        // Eager interpretation returns the literal value unchanged.
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::<Scalar>::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Ok(vec![Scalar::from(3.5)]),
        );

        // Staged interpretation records the payload as a constant atom without emitting an instruction.
        let context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let output =
            InterpretableOperation::<DomainTracingContext<EagerContext<Scalar, ScalarOperation<Scalar>>>>::interpret(
                &operation,
                &context,
                &EmptyRegionDriver,
                &[],
            )
            .unwrap()
            .remove(0);
        assert_eq!(output.atom_id(), Ok(AtomId::new(0)));
        let staged_builder = context.builder().borrow();
        assert!(staged_builder.instructions().is_empty());
        assert!(matches!(&staged_builder.atoms()[0], Atom::Constant(value) if *value == 3.5));

        // Verify the operation's textual form when it appears in a program.
        let mut program_builder = ProgramBuilder::<Scalar, ConstantOperation<Scalar>>::new();
        let output = program_builder.add_instruction(operation, Vec::new(), vec![]).unwrap()[0];
        let program = program_builder.build::<(), Scalar>(vec![output], (), Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f64 = constant [value=3.5]
                in (%0)
            "}
            .trim_end(),
        );
    }
}
