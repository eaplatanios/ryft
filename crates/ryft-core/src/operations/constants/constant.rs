use std::borrow::Cow;
use std::fmt::{Debug, Display};

use crate::arrays::{ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayType, DimensionValue};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError, BatchingTracer,
};
use crate::contexts::{Context, Domain, EagerContext, ProjectedContext, StagingContext};
use crate::differentiation::{
    DifferentiableType, DifferentiationContext, DifferentiationDual, DifferentiationPolicy, DifferentiationTracer,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_differentiable_operation, impl_nullary_transposable_operation};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    Operation, OperationFormatter, OperationProjection, ProgramError, RegionInterface, Type, TypeError,
    TypeIdentityRenaming, Value, ValueProjection,
};
use crate::tracing::Tracer;

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

impl<V: Value> Display for ConstantOperation<V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<V: Value> Debug for ConstantOperation<V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("ConstantOperation").field("value", &self.value).finish()
    }
}

impl<V: Value> Operation for ConstantOperation<V> {
    type Type = V::Type;

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
        renaming: &TypeIdentityRenaming<<V::Type as Type>::Identity>,
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

impl<
    Stored: Value<Type = ArrayType>,
    C: Context<Type = ArrayType, Operation: From<ConstantOperation<Stored>>>,
    P: ArrayExtentBatchingPolicy<C>,
> BatchableOperation<C, ArrayBatchingPolicy<P>> for ConstantOperation<Stored>
{
    #[inline]
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        check_count!("input", inputs, 0, ProgramError);
        Ok(context
            .parent()
            .bind(self.clone(), Vec::new(), &[])?
            .into_iter()
            .map(ArrayBatch::replicated)
            .collect::<Vec<_>>()
            .into())
    }
}

impl_non_differentiable_operation!(<V> ConstantOperation<V> where V: Value);
impl_nullary_transposable_operation!(<V> ConstantOperation<V> where V: Value);

/// Represents the ability to materialize a stored [`ConstantOperation`] payload and is typically implemented by
/// [`Context`]s. [`Constant`] is the literal value counterpart to [`Zero`](crate::Zero), [`One`](crate::One), and
/// [`Fill`](crate::Fill). It typically lives on [`Context`]s because producing a runtime value from a stored payload
/// can be context-dependent. For example, [`EagerContext`]s can return the value directly while [`StagingContext`]s
/// record a builder constant.
pub trait Constant<V, C> {
    /// Returns the runtime value represented by `value`.
    fn constant(&self, value: C) -> Result<V, ProgramError>;
}

impl<V: Value, O: Operation<Type = V::Type>> Constant<V, V> for EagerContext<V, O> {
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
    Constant<BatchingTracer<C, ArrayBatchingPolicy>, Stored> for BatchingContext<C, ArrayBatchingPolicy>
{
    #[inline]
    fn constant(&self, value: Stored) -> Result<BatchingTracer<C, ArrayBatchingPolicy>, ProgramError> {
        let value = self.parent().constant(value)?;
        let batch = ArrayBatch::new(value, BatchAxis::replicated())?;
        Ok(BatchingTracer::new(self.clone(), batch))
    }
}

impl<C: Context<Type: DifferentiableType>, P: DifferentiationPolicy<C>>
    Constant<DifferentiationTracer<C, P>, C::Constant> for DifferentiationContext<C, P>
{
    #[inline]
    fn constant(&self, value: C::Constant) -> Result<DifferentiationTracer<C, P>, ProgramError> {
        let dual = DifferentiationDual::new_with_zero_tangent(self.primal().lift(value)?)?;
        Ok(DifferentiationTracer::new(dual, self.clone()))
    }
}

/// Capability to stage an exact first-class dimension literal in a [`Context`]. This is the dimension literal
/// counterpart of [`Constant`], and it is the one way a static extent enters a program as a value: the literal is
/// staged as a [`ConstantOperation`] carrying a [`DimensionValue`], so it is an ordinary instruction that partial
/// evaluation folds, rendering shows as `constant [value=n]`, and lowering emits as a constant. It is never stored as a
/// program constant atom, which is the representation of captured runtime values. The capability is blanket-implemented
/// for every context whose operation family includes [`ConstantOperation<DimensionValue>`], so it is available under
/// eager, tracing, batching, partial-evaluation, and differentiation contexts alike without per-context support.
pub trait DimensionConstant: Context {
    /// Stages the dimension literal `extent` in this context and returns the resulting value.
    fn dimension_constant(&self, extent: usize) -> Result<Self::Value, ProgramError>;
}

impl<C: Context<Operation: From<ConstantOperation<DimensionValue>>>> DimensionConstant for C {
    #[inline]
    fn dimension_constant(&self, extent: usize) -> Result<Self::Value, ProgramError> {
        let mut outputs = self.bind(ConstantOperation::new(DimensionValue::constant(extent)?), Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType, DimensionOperation,
    };
    use crate::contexts::EagerContext;
    use crate::differentiation::{TransposableOperation, TranspositionContext};
    use crate::interpretation::InterpretableOperation;
    use crate::macros::{check_operation_batching, check_operation_partial_evaluation};
    use crate::parameters::Placeholder;
    use crate::programs::{Atom, AtomId, EmptyRegionDriver, MaybeZero, Operation, ProgramBuilder, Typed};
    use crate::tracing::{DomainTracingContext, TracingContext};

    use super::*;

    #[test]
    fn test_constant() {
        // Verify the operation's literal value, identity, and rendering.
        let operation = ConstantOperation::<Array>::new(Array::scalar(3.5).unwrap());
        assert_eq!(operation.name(), CONSTANT_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "constant [value=3.5]");
        assert_eq!(operation.value(), &Array::scalar(3.5).unwrap());

        // Verify the operation's textual form when it appears in a program.
        let mut program_builder = ProgramBuilder::<Array, ConstantOperation<Array>>::new();
        let output = program_builder.add_instruction(operation, Vec::new(), vec![], None).unwrap()[0];
        let program = program_builder.build::<(), Array>(vec![output], (), Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f64[] = constant [value=3.5]
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_constant_type_inference() {
        let operation = ConstantOperation::<Array>::new(Array::scalar(3.5).unwrap());
        assert_eq!(operation.infer_output_types(&[], &[]), Ok(vec![ArrayType::scalar(DataType::F64)]));
    }

    #[test]
    fn test_constant_interpretation() {
        let operation = ConstantOperation::<Array>::new(Array::scalar(3.5).unwrap());
        // Eager interpretation returns the literal value unchanged.
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Ok(vec![Array::scalar(3.5).unwrap()]),
        );

        // Staged interpretation records the payload as a constant atom without emitting an instruction.
        let context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let output =
            InterpretableOperation::<DomainTracingContext<EagerContext<Array, ArrayOperation<Array>>>>::interpret(
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
        assert!(matches!(&staged_builder.atoms()[0], Atom::Constant(value) if *value == Array::scalar(3.5).unwrap()));
    }

    #[test]
    fn test_constant_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = ConstantOperation::new(Array::scalar(3.5).unwrap()),
            cases = [{
                inputs = [],
                outputs = [(@known, Array::scalar(3.5).unwrap())],
                residual_instructions = 0,
            }],
        );
    }

    #[test]
    fn test_constant_batching() {
        check_operation_batching!(
            @exact,
            operation = ConstantOperation::new(Array::scalar(3.5).unwrap()),
            axis_size = 2,
            cases = [{
                inputs = [],
                outputs = [(@replicated, Array::scalar(3.5).unwrap())],
            }],
        );
    }

    #[test]
    fn test_constant_differentiation() {
        let context = DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new());
        let output = context.constant(Array::scalar(3.5).unwrap()).unwrap();
        assert_eq!(output.primal(), &Array::scalar(3.5).unwrap());
        assert!(matches!(output.tangent(), MaybeZero::Zero(r#type) if r#type == &ArrayType::scalar(DataType::F64)));
    }

    #[test]
    fn test_constant_transposition() {
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = context.input(ArrayType::scalar(DataType::F64));
        let input_cotangents = ConstantOperation::new(Array::scalar(3.5).unwrap())
            .transpose(
                &mut TranspositionContext::new(context),
                &EmptyRegionDriver,
                &[],
                &[MaybeZero::Value(output_cotangent)],
                &[],
            )
            .unwrap();
        assert_eq!(input_cotangents, ());
    }

    #[test]
    fn test_dimension_constant() {
        // Dimension variables carry a distinct identity per construction, so the literals are compared by extent.
        let dimension_extent = |value: ArrayIrValue<Array>| match value {
            ArrayIrValue::Dimension(dimension) => (dimension.extent(), dimension.r#type().extent()),
            value => panic!("expected a dimension value but got {value}"),
        };

        // Eager contexts materialize the literal directly, both in the full array IR operation family and in the
        // minimal family that backs the eager dispatch domain of `ArrayIrValue`, whose only operation is a constant.
        let full = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new().dimension_constant(3).unwrap();
        assert_eq!(dimension_extent(full), (3, Some(3)));
        let minimal = EagerContext::<ArrayIrValue<Array>>::new().dimension_constant(3).unwrap();
        assert_eq!(dimension_extent(minimal), (3, Some(3)));

        // The capability is not tied to the array IR family: a pure dimension program materializes the same literal.
        let dimension = EagerContext::<DimensionValue, DimensionOperation<DimensionValue>>::new().dimension_constant(2);
        assert_eq!(dimension.map(|dimension| dimension.extent()), Ok(2));

        // Staging records the literal as a `constant` instruction with an exact dimension type, never as a program
        // constant atom.
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let output = context.dimension_constant(3).unwrap();
        let ArrayIrType::Dimension(output_type) = output.r#type().into_owned() else {
            panic!("expected a dimension type");
        };
        assert_eq!(output_type.extent(), Some(3));
        let builder = context.builder().borrow();
        let [instruction] = builder.instructions() else {
            panic!("expected exactly one staged instruction");
        };
        assert!(matches!(instruction.operation(), ArrayIrOperation::Dimension(DimensionOperation::Constant(_))));
        assert!(!builder.atoms().iter().any(|atom| matches!(atom, Atom::Constant(_))));
        let program = builder
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output.atom_id().unwrap()],
                vec![],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:dimension<3> = constant [value=3]
                in (%0)
            "}
            .trim_end(),
        );
    }
}
