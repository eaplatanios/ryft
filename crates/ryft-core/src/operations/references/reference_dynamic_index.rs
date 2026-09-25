use std::borrow::Cow;
use std::fmt::Display;
use std::sync::LazyLock;

use ryft_macros::Parameter;

use crate::arrays::{
    ArrayIrType, ArrayIrValue, ArrayReferenceView, ArrayReferenceViewIndex, ArrayReferenceViewPath, ArrayType,
};
use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    CotangentAccumulator, DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver,
    DifferentiationDual, DifferentiationError, DifferentiationPolicy, TransposableOperation, TranspositionContext,
    TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::arithmetic::AddOperation;
use crate::parameters::Parameter;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    BatchableReferenceView, Concretizable, EffectClasses, Effects, MaybeZero, Operation, OperationFormatter,
    ProgramError, ProjectedValue, ReferenceAlias, ReferenceAliasKind, ReferenceDischargeContext,
    ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeValue, ReferenceDischargeableOperation,
    ReferenceType, ReferenceViewOperation, RegionInterface, TypeError, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`ReferenceDynamicIndexOperation`].
pub const REFERENCE_DYNAMIC_INDEX_OPERATION_NAME: &str = "reference_dynamic_index";

/// Pure reference view selecting a clamped scalar integer index and removing the selected axis.
///
/// The reference is input zero and the scalar integer index is input one. The index and referent must occupy the
/// same memory space. For a nonempty axis of length `n`, the selected index is clamped to `0..=n - 1`; unsigned indices
/// retain their full value until clamping. The resulting reference aliases the same allocation, and constructing the
/// view does not access that allocation's contents.
///
/// Type inference permits an empty selected axis so that an unreachable zero-trip scan body remains well typed.
/// Executing a selection on an empty axis fails because there is no element to select.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct ReferenceDynamicIndexOperation {
    /// Axis selected in the input reference.
    axis: usize,
}

impl ReferenceDynamicIndexOperation {
    /// Creates a new dynamic reference index operation selecting `axis`.
    pub const fn new(axis: usize) -> Self {
        Self { axis }
    }

    /// Returns the allocation-preserving view, whose index is supplied by input one.
    pub const fn transform(&self) -> ArrayReferenceView {
        ArrayReferenceView::Index { axis: self.axis, index: ArrayReferenceViewIndex::Symbolic(1) }
    }
}

impl Display for ReferenceDynamicIndexOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ReferenceDynamicIndexOperation {
    type Type = ArrayIrType;

    fn name(&self) -> &'static str {
        REFERENCE_DYNAMIC_INDEX_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let reference = <&ReferenceType<ArrayType>>::try_from(&input_types[0])?;
        let index = <&ArrayType>::try_from(&input_types[1])?;
        if index.rank() != 0 || !index.data_type().is_integer() {
            return Err(TypeError::invalid(format!(
                "`reference_dynamic_index` requires a scalar integer index but received `{index}`"
            )));
        }
        if index.memory() != reference.referent().memory() {
            return Err(TypeError::invalid(format!(
                "`reference_dynamic_index` reference and index must share one memory space but index resides in {} and reference resides in {}",
                index.memory(),
                reference.referent().memory(),
            )));
        }
        Ok(vec![ReferenceType::new(self.transform().output_type(reference.referent())?).into()])
    }

    fn effects(&self) -> Cow<'_, Effects> {
        // A view preserves the canonical allocation without accessing its state, so it declares one view alias and no
        // effect class. Share one descriptor to avoid allocating and validating it on every query.
        static EFFECTS: LazyLock<Effects> = LazyLock::new(|| {
            Effects::new(EffectClasses::NONE, Vec::new(), vec![ReferenceAlias::new(0, 0, ReferenceAliasKind::View)])
                .unwrap()
        });
        Cow::Borrowed(&EFFECTS)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("axis", self.axis))
    }
}

impl<C, P> ReferenceDischargeableOperation<C, P> for ReferenceDynamicIndexOperation
where
    C: Context<Type = ArrayIrType, Operation: From<ReferenceDynamicIndexOperation>>,
    P: ReferenceDischargePolicy<C, Referent = ArrayType, Alias = ArrayReferenceViewPath<C::Value>>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        _driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        self.transform().discharge(self, context, inputs)
    }
}

impl<C: Domain<Type = ArrayIrType, Value: ReferenceDynamicIndex>> InterpretableOperation<C>
    for ReferenceDynamicIndexOperation
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].reference_dynamic_index(self.axis, &inputs[1])?])
    }
}

// The default partial-evaluation behavior applies: a view carries no effect of its own and is placed wherever its
// reference operand is.
impl<C: Context<Type = ArrayIrType, Operation: From<ReferenceDynamicIndexOperation>>> PartiallyEvaluatableOperation<C>
    for ReferenceDynamicIndexOperation
{
}

impl<
    C: Context<
            Type = ArrayIrType,
            Operation: ReferenceViewOperation<View: BatchableReferenceView> + From<ReferenceDynamicIndexOperation>,
        >,
    P: BatchingPolicy<C>,
> BatchableOperation<C, P> for ReferenceDynamicIndexOperation
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        _driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        ReferenceViewOperation::batch(&C::Operation::from(self.clone()), context, inputs)
    }
}

impl<C: Context<Type = ArrayIrType, Value: ReferenceDynamicIndex>> DifferentiableOperation<C>
    for ReferenceDynamicIndexOperation
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        let primal = inputs[0].primal().reference_dynamic_index(self.axis, inputs[1].primal())?;
        // Apply the same view to a live tangent reference. A reference without tangent storage keeps a symbolic
        // zero typed with the new view, without constructing a tangent reference.
        Ok(vec![match inputs[0].tangent() {
            MaybeZero::Value(reference) => {
                let tangent = reference
                    .reference_dynamic_index(self.axis, &context.primal_to_tangent(inputs[1].primal().clone())?)?;
                DifferentiationDual::new(primal, MaybeZero::Value(tangent))?
            }
            MaybeZero::Zero(_) => DifferentiationDual::new_with_zero_tangent(primal)?,
        }])
    }
}

impl<V: Value<Type = ArrayIrType>, O: Operation<Type = ArrayIrType> + From<AddOperation<ArrayIrType>>>
    TransposableOperation<V, O> for ReferenceDynamicIndexOperation
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        // A view is aliasing metadata rather than a linear map of its own: the cotangent of a view operand is reached by
        // reapplying the view path to its root's cotangent reference inside the transposition context, so the reverse
        // sweep never needs this rule to run and every operand receives a structural zero.
        check_count!("input", inputs, 2, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        check_count!("accumulator", accumulators, 2, DifferentiationError);
        for (accumulator, input) in accumulators.iter().zip(inputs) {
            accumulator.accumulate(context, MaybeZero::Zero(input.r#type().cotangent()?))?;
        }
        Ok(())
    }
}

/// Derives a reference view using a scalar integer index supplied as a value.
pub trait ReferenceDynamicIndex<Index = Self, Output = Self>: Sized {
    /// Returns a reference selecting one element on `axis`, removing that axis and sharing the original allocation.
    /// The index is clamped to the valid range, as for [`Slice`](crate::operations::DynamicSlice). Executing a
    /// selection on an empty axis fails; a staged selection can still appear in an unexecuted zero-trip scan body.
    ///
    /// # Parameters
    ///
    ///   - `axis`: Axis of the current reference's referent to select and remove.
    ///   - `index`: Scalar integer value in the same memory space as the referent. Negative values select the first
    ///     element and values beyond the axis length select the last. Eager reference handles read this value through
    ///     [`Concretizable<i128>`], while staging retains it as an ordinary operand.
    fn reference_dynamic_index(&self, axis: usize, index: &Index) -> Result<Output, ProgramError>;
}

impl<V: Value<Type = ArrayIrType>> ReferenceDynamicIndex for V
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceDynamicIndexOperation>,
{
    fn reference_dynamic_index(&self, axis: usize, index: &Self) -> Result<Self, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(ReferenceDynamicIndexOperation::new(axis), Vec::new(), &[self.clone(), index.clone()])?
            .remove(0))
    }
}

impl<V> ReferenceDynamicIndex<V, <V as ValueProjection<ReferenceType<ArrayType>>>::Projected>
    for ProjectedValue<ReferenceType<ArrayType>, V>
where
    V: Value<Type = ArrayIrType> + ValueProjection<ReferenceType<ArrayType>>,
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceDynamicIndexOperation>,
{
    fn reference_dynamic_index(
        &self,
        axis: usize,
        index: &V,
    ) -> Result<<V as ValueProjection<ReferenceType<ArrayType>>>::Projected, ProgramError> {
        self.value().reference_dynamic_index(axis, index)?.into_projected().map_err(Into::into)
    }
}

impl<A: Value<Type = ArrayType> + Concretizable<i128>> ReferenceDynamicIndex for ArrayIrValue<A> {
    fn reference_dynamic_index(&self, axis: usize, index: &Self) -> Result<Self, ProgramError> {
        ReferenceDynamicIndexOperation::new(axis)
            .infer_output_types(&[self.r#type().into_owned(), index.r#type().into_owned()], &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        let index = <Self as ValueProjection<ArrayType>>::projected(index)?;
        let referent = reference.r#type();
        let shape = referent
            .referent()
            .static_shape()
            .ok_or_else(|| TypeError::invalid("eager reference indexing requires a static shape"))?;
        let maximum = shape[axis]
            .checked_sub(1)
            .ok_or_else(|| TypeError::invalid("cannot dynamically index an empty reference axis"))?;
        let index = index.concretize()?.clamp(0, maximum as i128) as usize;
        Ok(Self::Reference(
            reference
                .with_transform(ArrayReferenceView::Index { axis, index: ArrayReferenceViewIndex::Static(index) })?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayIrOperation, ArrayReferenceDischarge, DataType, Memory};
    use crate::contexts::EagerContext;
    use crate::differentiation::DifferentiationTracer;
    use crate::operations::references::reference_new::ReferenceNew;
    use crate::operations::references::reference_read::ReferenceRead;
    use crate::operations::references::reference_write::ReferenceWrite;
    use crate::programs::EmptyRegionDriver;

    use super::*;

    type TestValue = ArrayIrValue<Array>;
    type TestOperation = ArrayIrOperation<Array>;
    type TestDestination = EagerContext<TestValue, TestOperation>;

    #[test]
    fn test_reference_dynamic_index() {
        let operation = ReferenceDynamicIndexOperation::new(1);
        assert_eq!(operation.to_string(), "reference_dynamic_index [axis=1]");
        assert_eq!(
            operation.transform(),
            ArrayReferenceView::Index { axis: 1, index: ArrayReferenceViewIndex::Symbolic(1) }
        );
    }

    #[test]
    fn test_reference_dynamic_index_type_inference() {
        let operation = ReferenceDynamicIndexOperation::new(0);
        let reference = ReferenceType::new(ArrayType::new_static(DataType::F32, [3, 2]));
        let index = ArrayType::scalar(DataType::I64);
        assert_eq!(
            operation.infer_output_types(&[reference.into(), index.clone().into()], &[]),
            Ok(vec![ReferenceType::new(ArrayType::new_static(DataType::F32, [2])).into()])
        );

        let root = ReferenceType::new(ArrayType::new_static(DataType::F32, [3, 2]));
        assert_eq!(
            ReferenceDynamicIndexOperation::new(2)
                .infer_output_types(&[root.clone().into(), index.clone().into()], &[]),
            Err(TypeError::invalid("reference index axis 2 is out of bounds for rank 2")),
        );
        assert_eq!(
            operation.infer_output_types(&[root.clone().into(), ArrayType::new_static(DataType::I64, [1]).into()], &[]),
            Err(TypeError::invalid("`reference_dynamic_index` requires a scalar integer index but received `i64[1]`")),
        );
        let host_index = index.clone().with_memory(Memory::Host { pinned: false });
        assert_eq!(
            operation.infer_output_types(&[root.into(), host_index.clone().into()], &[]),
            Err(TypeError::invalid(format!(
                "`reference_dynamic_index` reference and index must share one memory space but index resides in {} and reference resides in {}",
                host_index.memory(),
                index.memory(),
            ))),
        );

        // An empty-axis body can be constructed for a zero-trip scan; no index is executed in that case.
        let empty = ReferenceType::new(ArrayType::new_static(DataType::F32, [0, 2]));
        assert_eq!(
            operation.infer_output_types(&[empty.clone().into(), index.into()], &[]),
            Ok(vec![ReferenceType::new(ArrayType::new_static(DataType::F32, [2])).into()])
        );
        assert_eq!(
            operation.infer_output_types(&[empty.into(), ArrayType::scalar(DataType::F32).into()], &[]),
            Err(TypeError::invalid("`reference_dynamic_index` requires a scalar integer index but received `f32[]`"))
        );
    }

    #[test]
    fn test_reference_dynamic_index_interpretation() {
        let reference = TestValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap()).reference_new().unwrap();
        let negative = TestValue::Array(Array::scalar(-9_i64).unwrap());
        let large = TestValue::Array(Array::scalar(u64::MAX).unwrap());
        assert_eq!(
            reference.reference_dynamic_index(0, &negative).unwrap().read(),
            Ok(TestValue::Array(Array::scalar(1.0_f32).unwrap()))
        );
        let last = reference.reference_dynamic_index(0, &large).unwrap();
        last.write(&TestValue::Array(Array::scalar(7.0_f32).unwrap())).unwrap();
        assert_eq!(reference.read(), Ok(TestValue::Array(Array::vector(vec![1.0_f32, 2.0, 7.0]).unwrap())));

        // Empty axes remain well typed, but eager execution cannot select an element from them.
        let empty = TestValue::Array(Array::vector(Vec::<f32>::new()).unwrap()).reference_new().unwrap();
        assert_eq!(
            empty.reference_dynamic_index(0, &negative),
            Err(TypeError::invalid("cannot dynamically index an empty reference axis").into())
        );
    }

    #[test]
    fn test_reference_dynamic_index_differentiation() {
        let context = DifferentiationContext::fused(TestDestination::new());
        let allocation_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let reference = TestValue::Array(
            Array::from_elements::<f32>(allocation_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        )
        .reference_new()
        .unwrap();
        let tangent_reference =
            TestValue::Array(Array::from_elements::<f32>(allocation_type, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap())
                .reference_new()
                .unwrap();

        // An active reference's view is applied to its tangent reference at the same index, so both views select the
        // same row of their respective allocations. The index itself has no tangent.
        let active = DifferentiationTracer::new(
            DifferentiationDual::new(reference.clone(), tangent_reference).unwrap(),
            context.clone(),
        );
        let index = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(TestValue::Array(Array::scalar(1_i64).unwrap())).unwrap(),
            context.clone(),
        );
        let indexed =
            context.bind(ReferenceDynamicIndexOperation::new(0), Vec::new(), &[active, index.clone()]).unwrap();
        assert_eq!(indexed.len(), 1);
        assert_eq!(indexed[0].primal().read(), Ok(TestValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]).unwrap())));
        assert_eq!(
            indexed[0].tangent().as_value().unwrap().read(),
            Ok(TestValue::Array(Array::vector(vec![10.0_f32, 11.0, 12.0]).unwrap())),
        );

        // The views of a plumbing reference stay plumbing, typed with the view's own reference type.
        let plumbing =
            DifferentiationTracer::new(DifferentiationDual::new_with_zero_tangent(reference).unwrap(), context.clone());
        let indexed = context.bind(ReferenceDynamicIndexOperation::new(0), Vec::new(), &[plumbing, index]).unwrap();
        assert_eq!(indexed[0].primal().read(), Ok(TestValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]).unwrap())));
        assert!(matches!(
            indexed[0].tangent(),
            MaybeZero::Zero(r#type)
                if *r#type == ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3]))),
        ));
    }

    #[test]
    fn test_reference_dynamic_index_reference_discharge() {
        let context =
            ReferenceDischargeContext::<TestDestination, ArrayReferenceDischarge>::new(TestDestination::new());
        let value = TestValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap()).reference_new().unwrap();
        let reference = context
            .bind_preserved(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])), value)
            .unwrap();
        let index = TestValue::Array(Array::scalar(2_i64).unwrap());
        let outputs = ReferenceDynamicIndexOperation::new(0)
            .discharge_references(
                &context,
                &EmptyRegionDriver,
                &[reference.into(), ReferenceDischargeValue::Value(index)],
            )
            .unwrap();
        let view = outputs[0].try_as_reference("a dynamic reference view").unwrap();
        assert_eq!(view.preserved().unwrap().read(), Ok(TestValue::Array(Array::scalar(3.0_f32).unwrap())));
    }
}
