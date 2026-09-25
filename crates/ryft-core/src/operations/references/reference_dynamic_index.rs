use std::borrow::Cow;
use std::fmt::Display;
use std::sync::LazyLock;

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

/// Pure reference-to-reference operation selecting the hyperplane at a runtime scalar integer index along `axis` and
/// removing that axis. For example, index `1` on axis `0` of a `f32[2, 3]` reference selects its second row as a
/// `f32[3]` reference.
///
/// The reference is input zero and the scalar integer index is input one. The index and referent must occupy the
/// same memory space. For a nonempty axis of length `n`, a negative signed index `i` counts from the end of the axis
/// (i.e., it is replaced by `i + n` once), and the result is then clamped to `0..=n - 1`, so `-1` selects the last
/// hyperplane and an index that is still negative after that single wrap selects the first. Unsigned indices retain
/// their full value until clamping. This matches the negative-index policy of
/// [`DynamicSliceOperation`](crate::operations::DynamicSliceOperation), which reference discharge uses to materialize
/// the view. The resulting reference aliases the same allocation, and constructing the view does not access that
/// allocation's contents.
///
/// Type inference permits an empty selected axis so that an unreachable zero-trip scan body remains well typed.
/// Executing a selection on an empty axis fails because there is no element to select.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ReferenceDynamicIndexOperation {
    type Type = ArrayIrType;

    #[inline]
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
                "`reference_dynamic_index` reference and index must share one memory space but index resides in {} \
                 and reference resides in {}",
                index.memory(),
                reference.referent().memory(),
            )));
        }
        Ok(vec![ReferenceType::new(self.transform().output_type(reference.referent())?).into()])
    }

    #[inline]
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
        // A view is aliasing metadata rather than a linear map of its own: the cotangent of a view operand is reached
        // by reapplying the view path to its root's cotangent reference inside the transposition context, so the
        // reverse sweep never needs this rule to run and every operand receives a structural zero.
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
    /// Returns a reference selecting the hyperplane at `index` along `axis`, removing that axis and sharing the
    /// original allocation. Negative indices count from the end of the axis and the result is then clamped to the
    /// valid range, as for [`DynamicSlice`](crate::operations::DynamicSlice). Executing a selection on an empty axis
    /// fails; a staged selection can still appear in an unexecuted zero-trip scan body.
    ///
    /// # Parameters
    ///
    ///   - `axis`: Axis of the current reference's referent to select and remove.
    ///   - `index`: Scalar integer value in the same memory space as the referent. A negative value `i` is replaced by
    ///     `i + n` once, where `n` is the extent of `axis`, and the result is clamped to `0..=n - 1`. Eager reference
    ///     handles read this value through [`Concretizable<i128>`], while staging retains it as an ordinary operand.
    fn reference_dynamic_index(&self, axis: usize, index: &Index) -> Result<Output, ProgramError>;
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
        let extent = shape[axis] as i128;
        if extent == 0 {
            return Err(TypeError::invalid("cannot dynamically index an empty reference axis").into());
        }
        // Negative indices count from the end of the axis once before clamping, exactly as the dynamic slices that
        // reference discharge stages for this view do.
        let index = index.concretize()?;
        let index = if index < 0 { index + extent } else { index };
        let index = index.clamp(0, extent - 1) as usize;
        Ok(Self::Reference(
            reference
                .with_transform(ArrayReferenceView::Index { axis, index: ArrayReferenceViewIndex::Static(index) })?,
        ))
    }
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

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayIrOperation, ArrayReference, ArrayReferenceDischarge, DataType, Memory};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::DifferentiationTracer;
    use crate::macros::{check_operation_partial_evaluation, check_operation_type_inference};
    use crate::operations::references::reference_new::{ReferenceNew, ReferenceNewOperation};
    use crate::operations::references::reference_read::{ReferenceRead, ReferenceReadOperation};
    use crate::operations::references::reference_write::ReferenceWrite;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationValue, ReferencePlacement};
    use crate::programs::{EmptyRegionDriver, Program, ProgramBuilder, ReferenceAlias, ReferenceAliasKind};

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
            ArrayReferenceView::Index { axis: 1, index: ArrayReferenceViewIndex::Symbolic(1) },
        );
        assert!(operation.effects().is_pure());
        assert_eq!(operation.effects().reference_aliases(), &[ReferenceAlias::new(0, 0, ReferenceAliasKind::View)]);
        assert_eq!(operation.effects().reference_effects(), &[]);
    }

    #[test]
    fn test_reference_dynamic_index_type_inference() {
        let root = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3, 2])));
        let row = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2])));
        let index = ArrayType::scalar(DataType::I64);
        check_operation_type_inference!(
            operation = ReferenceDynamicIndexOperation::new(0),
            cases = [
                {
                    input_types = [root.clone(), ArrayIrType::Array(index.clone())],
                    output_types = [row.clone()],
                },
                {
                    input_types = [root.clone()],
                    error = "expected 2 inputs but got 1",
                },
                {
                    input_types = [
                        ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3, 2])),
                        ArrayIrType::Array(index.clone()),
                    ],
                    error = "expected reference type but got array type",
                },
                {
                    input_types = [root.clone(), ArrayIrType::Array(ArrayType::new_static(DataType::I64, [1]))],
                    error = "`reference_dynamic_index` requires a scalar integer index but received `i64[1]`",
                },
                {
                    input_types = [root.clone(), ArrayIrType::Array(ArrayType::scalar(DataType::F32))],
                    error = "`reference_dynamic_index` requires a scalar integer index but received `f32[]`",
                },
                // An empty-axis body can be constructed for a zero-trip scan, where no index is ever executed.
                {
                    input_types = [
                        ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [0, 2]))),
                        ArrayIrType::Array(index.clone()),
                    ],
                    output_types = [row],
                },
            ],
        );
        check_operation_type_inference!(
            operation = ReferenceDynamicIndexOperation::new(2),
            cases = [{
                input_types = [root.clone(), ArrayIrType::Array(index.clone())],
                error = "reference index axis 2 is out of bounds for rank 2",
            }],
        );

        // The index must reside in the memory space of the referent.
        let host_index = index.clone().with_memory(Memory::Host { pinned: false });
        assert_eq!(
            ReferenceDynamicIndexOperation::new(0).infer_output_types(&[root, host_index.clone().into()], &[]),
            Err(TypeError::invalid(format!(
                "`reference_dynamic_index` reference and index must share one memory space but index resides in {} \
                 and reference resides in {}",
                host_index.memory(),
                index.memory(),
            ))),
        );
    }

    #[test]
    fn test_reference_dynamic_index_interpretation() {
        let reference = TestValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap()).reference_new().unwrap();
        let select = |index: TestValue| reference.reference_dynamic_index(0, &index).unwrap().read().unwrap();
        assert_eq!(
            select(TestValue::Array(Array::scalar(1_i64).unwrap())),
            TestValue::Array(Array::scalar(2.0_f32).unwrap())
        );

        // A negative index counts from the end of the axis once, and the result is then clamped to the valid range.
        assert_eq!(
            select(TestValue::Array(Array::scalar(-1_i64).unwrap())),
            TestValue::Array(Array::scalar(3.0_f32).unwrap())
        );
        assert_eq!(
            select(TestValue::Array(Array::scalar(-3_i32).unwrap())),
            TestValue::Array(Array::scalar(1.0_f32).unwrap())
        );
        assert_eq!(
            select(TestValue::Array(Array::scalar(-9_i64).unwrap())),
            TestValue::Array(Array::scalar(1.0_f32).unwrap())
        );
        assert_eq!(
            select(TestValue::Array(Array::scalar(7_i64).unwrap())),
            TestValue::Array(Array::scalar(3.0_f32).unwrap())
        );

        // Unsigned indices retain their full value until clamping, and the selected view writes through to its root.
        let last = reference.reference_dynamic_index(0, &TestValue::Array(Array::scalar(u64::MAX).unwrap())).unwrap();
        last.write(&TestValue::Array(Array::scalar(7.0_f32).unwrap())).unwrap();
        assert_eq!(reference.read(), Ok(TestValue::Array(Array::vector(vec![1.0_f32, 2.0, 7.0]).unwrap())));

        // Empty axes remain well typed, but eager execution cannot select an element from them.
        let empty = TestValue::Array(Array::vector(Vec::<f32>::new()).unwrap()).reference_new().unwrap();
        assert_eq!(
            empty.reference_dynamic_index(0, &TestValue::Array(Array::scalar(0_i64).unwrap())),
            Err(TypeError::invalid("cannot dynamically index an empty reference axis").into())
        );
    }

    #[test]
    fn test_reference_dynamic_index_partial_evaluation() {
        let vector = || Array::vector(vec![1.0_f32, 2.0, 3.0]);
        let live = TestValue::Reference(ArrayReference::new(vector().unwrap()));
        let index = TestValue::Array(Array::scalar(-1_i64).unwrap());
        let inputs = [PartialEvaluationValue::known(live.clone()), PartialEvaluationValue::known(index.clone())];

        // Under the default `Execute` placement, a known reference and index fold the view into a handle that aliases
        // the same allocation.
        let executing = PartialEvaluationContext::new(TestDestination::new());
        let outputs =
            executing.fold_or_residualize(ReferenceDynamicIndexOperation::new(0), Vec::new(), &inputs).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_known(), Some(&live.reference_dynamic_index(0, &index).unwrap()));

        // Under the `Stage` placement, the view stays residual regardless of input knowledge, so the residual program
        // keeps the complete alias chain of every later access.
        let staging =
            PartialEvaluationContext::new(TestDestination::new()).with_reference_placement(ReferencePlacement::Stage);
        let outputs = staging.fold_or_residualize(ReferenceDynamicIndexOperation::new(0), Vec::new(), &inputs).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_unknown());

        // Program-level partial evaluation uses the `Stage` placement, so the view stays in the residual program
        // whether its reference and index are known or not, and replays against the runtime values.
        let known = TestValue::Reference(ArrayReference::new(vector().unwrap()));
        let replay = TestValue::Reference(ArrayReference::new(vector().unwrap()));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])));
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = ReferenceDynamicIndexOperation::new(0),
            cases = [
                {
                    inputs = [(@known, known.clone()), (@known, index.clone())],
                    outputs = [(@residual, known.reference_dynamic_index(0, &index).unwrap())],
                    residual_instructions = 1,
                },
                {
                    inputs = [
                        (@unknown(type = reference_type, replay = replay.clone())),
                        (@unknown(type = ArrayIrType::Array(ArrayType::scalar(DataType::I64)), replay = index.clone())),
                    ],
                    outputs = [(@residual, replay.reference_dynamic_index(0, &index).unwrap())],
                    residual_instructions = 1,
                },
            ],
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
    fn test_reference_dynamic_index_transposition() {
        let vector_type = ArrayType::new_static(DataType::F32, [3]);
        let index_type = ArrayIrType::Array(ArrayType::scalar(DataType::I64));

        // `r = new(x); y = read(dynamic_index(r, i))`: the view has no transpose of its own, because the read
        // accumulates `ȳ` into the selected element of the cotangent reference of `r`, which the allocation then
        // freezes into `x̄`.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(ArrayIrType::Array(vector_type.clone()));
        let index = builder.add_input(index_type.clone());
        let reference = builder
            .add_instruction(ReferenceNewOperation::<ArrayType, ArrayIrType>::new(), Vec::new(), vec![initial], None)
            .unwrap()[0];
        let element = builder
            .add_instruction(ReferenceDynamicIndexOperation::new(0), Vec::new(), vec![reference, index], None)
            .unwrap()[0];
        let output = builder
            .add_instruction(ReferenceReadOperation::<ArrayType, ArrayIrType>::new(), Vec::new(), vec![element], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[], %1:i64[] .
                let %2:f32[3] = zero [type=f32[3]]
                    %3:ref<f32[3]> = reference_new %2
                    %4:ref<f32[]> = reference_dynamic_index [axis=0] %3 %1
                    () = reference_add_update %4 %0
                    %5:f32[3] = reference_freeze %3
                in (%5)"},
        );
        assert_eq!(
            transposed.interpret(vec![
                TestValue::Array(Array::scalar(5.0_f32).unwrap()),
                TestValue::Array(Array::scalar(-1_i64).unwrap()),
            ]),
            Ok(vec![TestValue::Array(Array::vector(vec![0.0_f32, 0.0, 5.0]).unwrap())]),
        );

        // The rule itself gives both operands, including the integer index, a structural zero whatever its output
        // cotangent is, so it never stages anything into the transposition context.
        let inputs = [
            PartialValue::Unknown(ArrayIrType::Reference(ReferenceType::new(vector_type))),
            PartialValue::Unknown(index_type),
        ];
        let tracing = TracingContext::<TestValue, TestOperation>::new();
        let cotangent = tracing.input(ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let mut context = TranspositionContext::new(tracing);
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        assert_eq!(
            ReferenceDynamicIndexOperation::new(0).transpose(
                &mut context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Value(cotangent)],
                &accumulators,
            ),
            Ok(()),
        );
        assert!(context.builder().borrow().instructions().is_empty());
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

    #[test]
    fn test_reference_dynamic_index_negative_indices_agree_across_execution_paths() {
        // `y = read(dynamic_index(new(x), i))` selects the same element whether the view runs on eager reference
        // handles or is discharged into dynamic slices, including for negative and out-of-bounds indices.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])));
        let index = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::I64)));
        let reference = builder
            .add_instruction(ReferenceNewOperation::<ArrayType, ArrayIrType>::new(), Vec::new(), vec![initial], None)
            .unwrap()[0];
        let element = builder
            .add_instruction(ReferenceDynamicIndexOperation::new(0), Vec::new(), vec![reference, index], None)
            .unwrap()[0];
        let output = builder
            .add_instruction(ReferenceReadOperation::<ArrayType, ArrayIrType>::new(), Vec::new(), vec![element], None)
            .unwrap()[0];
        let program: Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>> =
            builder.build(vec![output], vec![Placeholder; 2], vec![Placeholder]).unwrap();
        let discharged = program
            .clone()
            .discharge_references::<ArrayReferenceDischarge>(0)
            .unwrap()
            .into_program_without_external_references()
            .unwrap();
        for (index, expected) in [(-9, 1.0_f32), (-3, 1.0), (-1, 3.0), (0, 1.0), (2, 3.0), (7, 3.0)] {
            let inputs = vec![
                TestValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap()),
                TestValue::Array(Array::scalar(index as i64).unwrap()),
            ];
            let expected = Ok(vec![TestValue::Array(Array::scalar(expected).unwrap())]);
            assert_eq!(program.interpret(inputs.clone()), expected, "eager execution with index {index}");
            assert_eq!(discharged.interpret(inputs), expected, "discharged execution with index {index}");
        }
    }
}
