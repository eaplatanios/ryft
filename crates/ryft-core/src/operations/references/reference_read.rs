use std::borrow::Cow;
use std::fmt::Display;
use std::marker::PhantomData;
use std::sync::LazyLock;

use crate::arrays::{ArrayIrType, ArrayIrValue, ArrayReferenceTransform, ArrayType};
use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    CotangentAccumulator, DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver,
    DifferentiationDual, DifferentiationError, DifferentiationPolicy, ResidualZeroProvider, TransposableOperation,
    TranspositionContext, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::manipulation::reshaping::Reshape;
use crate::operations::manipulation::slicing::Slice;
use crate::operations::references::reference_add_update::ReferenceAddUpdate;
use crate::operations::references::reference_new::ReferenceNewOperation;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    BatchableReferenceTransform, Concretizable, EffectClasses, Effects, MaybeZero, NoReferenceTransform, Operation,
    OperationFormatter, OperationProvider, ProgramError, ProjectedValue, ReferenceAccessDescriptor,
    ReferenceAccessMode, ReferenceAccessOperation, ReferenceDischargeContext, ReferenceDischargeDriver,
    ReferenceDischargePolicy, ReferenceDischargeValue, ReferenceDischargeableOperation, ReferenceEffect,
    ReferenceMemberType, ReferenceTransform, ReferenceType, RegionInterface, Type, TypeError, Typed, Value,
    ValueProjection, batch_reference_transforms, infer_reference_view_type,
};
use crate::tracing::{Tracer, TracingContext};

/// Canonical operation name for [`ReferenceReadOperation`].
pub const REFERENCE_READ_OPERATION_NAME: &str = "reference_read";

/// Reads the current value selected from a reference in the enclosing type universe `U`. The root is the first input.
/// Dynamic bindings follow it in transform order; the path is stored in `Transform` metadata. An empty path reads the
/// complete referent, and a nonempty path returns its selected referent type.
#[derive(Clone, Debug)]
pub struct ReferenceReadOperation<
    T: Type,
    U: Type,
    Transform: ReferenceTransform<Type = U, Referent = T> = NoReferenceTransform<T, U>,
> {
    /// Refer to the documentation of [`Self::transforms`].
    transforms: Vec<Transform>,

    /// [`PhantomData`] marker tying this [`Operation`] to its referent [`Type`] `T` and to the [`Type`] universe `U`
    /// in which it is valid.
    marker: PhantomData<fn() -> (T, U)>,
}

impl<T: Type, U: Type, Transform: ReferenceTransform<Type = U, Referent = T>> ReferenceReadOperation<T, U, Transform> {
    /// Creates a new [`ReferenceReadOperation`].
    pub const fn new() -> Self {
        Self { transforms: Vec::new(), marker: PhantomData }
    }

    /// Returns a copy of this [`ReferenceReadOperation`] with the provided transforms applied to the reference input.
    #[inline]
    pub fn with_transforms(mut self, transforms: Vec<Transform>) -> Self {
        self.transforms = transforms;
        self
    }

    /// Returns the transforms applied to the reference input.
    #[inline]
    pub fn transforms(&self) -> &[Transform] {
        &self.transforms
    }

    /// Returns the number of dynamic inputs supplied after the base inputs for this [`ReferenceReadOperation`].
    fn binding_count(&self) -> usize {
        self.transforms.iter().map(ReferenceTransform::binding_count).sum()
    }
}

impl<T: Type, U: Type, Transform: ReferenceTransform<Type = U, Referent = T>> Default
    for ReferenceReadOperation<T, U, Transform>
{
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Type, U: Type, Transform: ReferenceTransform<Type = U, Referent = T>> Display
    for ReferenceReadOperation<T, U, Transform>
where
    Self: Operation,
{
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type, U: Type + From<T>, Transform: ReferenceTransform<Type = U, Referent = T>> Operation
    for ReferenceReadOperation<T, U, Transform>
where
    for<'t> &'t ReferenceType<T>: TryFrom<&'t U, Error = TypeError>,
{
    type Type = U;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_READ_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[U],
        region_interfaces: &[RegionInterface<U>],
    ) -> Result<Vec<U>, TypeError> {
        check_count!("input", input_types, 1 + self.binding_count(), TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let reference = <&ReferenceType<T>>::try_from(&input_types[0])?;
        let binding_types = input_types[1..].iter().collect::<Vec<_>>();
        let referent = infer_reference_view_type(
            reference.referent(),
            &self.transforms,
            &binding_types,
            ReferenceAccessMode::Read,
        )?;
        Ok(vec![referent.into()])
    }

    #[inline]
    fn effects(&self) -> Cow<'_, Effects> {
        // Share one descriptor across all type instantiations to avoid allocating and validating it on every query.
        static EFFECTS: LazyLock<Effects> = LazyLock::new(|| {
            Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read }],
            )
            .unwrap()
        });
        Cow::Borrowed(&EFFECTS)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        // An empty transform path accesses the complete referent, so it renders as just the operation name.
        let operation = OperationFormatter::new(formatter, indentation, self.name())?;
        if self.transforms.is_empty() {
            return Ok(());
        }
        operation.bracketed(|operation| operation.list("transforms", &self.transforms))
    }
}

impl<T: Type, U: Type, Transform: ReferenceTransform<Type = U, Referent = T>> ReferenceAccessOperation
    for ReferenceReadOperation<T, U, Transform>
where
    ReferenceReadOperation<T, U, Transform>: Operation<Type = U>,
{
    type Transform = Transform;

    #[inline]
    fn base_input_count(&self) -> usize {
        1
    }

    #[inline]
    fn reference_access_descriptor(&self, input_index: usize) -> Option<ReferenceAccessDescriptor<'_, Transform>> {
        (input_index == 0).then(|| ReferenceAccessDescriptor::new(&self.transforms, 1..1 + self.binding_count()))
    }

    #[inline]
    fn with_reference_access_transforms(
        &self,
        input_index: usize,
        transforms: Vec<Transform>,
    ) -> Result<Self, ProgramError> {
        if input_index != 0 {
            return Err(ProgramError::InvalidArgument {
                message: format!("`{}` has no reference access at input {}", self.name(), input_index),
            });
        }
        Ok(self.clone().with_transforms(transforms))
    }
}

impl<
    T: Type,
    U: Type,
    Transform: ReferenceTransform<Type = U, Referent = T>,
    C: Context<Type = U>,
    P: ReferenceDischargePolicy<C, Referent = T, Transform = Transform>,
> ReferenceDischargeableOperation<C, P> for ReferenceReadOperation<T, U, Transform>
where
    ReferenceReadOperation<T, U, Transform>: Operation<Type = U>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        _driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        check_count!("input", inputs, 1 + self.binding_count(), ProgramError);
        let reference = inputs[0].try_as_reference("a reference to read")?;
        let bindings = inputs[1..]
            .iter()
            .map(|input| input.try_as_value("a reference transform binding").cloned())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(vec![ReferenceDischargeValue::Value(context.read_through(reference, &self.transforms, &bindings)?)])
    }
}

impl<
    T: Type,
    U: Type,
    Transform: ReferenceTransform<Type = U, Referent = T>,
    C: Domain<Type = U, Value: ReferenceRead<Transform, C::Value, C::Value>>,
> InterpretableOperation<C> for ReferenceReadOperation<T, U, Transform>
where
    ReferenceReadOperation<T, U, Transform>: Operation<Type = U>,
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1 + self.binding_count(), ProgramError);
        Ok(vec![inputs[0].read_through(&self.transforms, &inputs[1..])?])
    }
}

impl<
    T: Type,
    U: Type,
    Transform: ReferenceTransform<Type = U, Referent = T>,
    C: Context<Type = U, Operation: From<ReferenceReadOperation<T, U, Transform>>>,
> PartiallyEvaluatableOperation<C> for ReferenceReadOperation<T, U, Transform>
{
}

impl<
    T: Type,
    U: Type + From<ReferenceType<T>>,
    Transform: BatchableReferenceTransform<Type = U, Referent = T>,
    C: Context<Type = U, Operation: From<ReferenceReadOperation<T, U, Transform>>>,
    P: BatchingPolicy<C>,
> BatchableOperation<C, P> for ReferenceReadOperation<T, U, Transform>
where
    for<'t> &'t ReferenceType<T>: TryFrom<&'t U, Error = TypeError>,
    ReferenceReadOperation<T, U, Transform>: Operation<Type = U>,
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        _driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        // A read yields the selected referent, carrying the batch axis through every transform.
        check_count!("input", inputs, 1 + self.binding_count(), ProgramError);
        let binding_axes = inputs[1..].iter().map(P::batch_axis).collect::<Vec<_>>();
        let (transforms, output_axis) = batch_reference_transforms(
            P::value(&inputs[0]).r#type().as_ref(),
            P::batch_axis(&inputs[0]),
            &self.transforms,
            &binding_axes,
        )?;
        let values = inputs.iter().map(|input| P::value(input).clone()).collect::<Vec<_>>();
        let value = context.parent().bind(self.clone().with_transforms(transforms), Vec::new(), &values)?.remove(0);
        Ok(vec![P::batch(value, output_axis)?].into())
    }
}

impl<
    T: Type,
    U: DifferentiableType,
    Transform: ReferenceTransform<Type = U, Referent = T>,
    C: Context<Type = U, Operation: From<ReferenceReadOperation<T, U, Transform>>>,
> DifferentiableOperation<C> for ReferenceReadOperation<T, U, Transform>
where
    ReferenceReadOperation<T, U, Transform>: Operation<Type = U>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // Reading a reference reads its tangent reference alongside. A plumbing reference (i.e., a reference dual
        // whose tangent is a symbolic zero) carries no tangent reference, so the value read from it has a symbolic
        // zero tangent.
        check_count!("input", inputs, 1 + self.binding_count(), ProgramError);
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal = context.primal().bind(self.clone(), Vec::new(), &primal_inputs)?.remove(0);
        Ok(vec![match inputs[0].tangent() {
            MaybeZero::Value(reference) => {
                let mut tangent_inputs = vec![reference.clone()];
                for input in &inputs[1..] {
                    tangent_inputs.push(context.primal_to_tangent(input.primal().clone())?);
                }
                let tangent = context.tangent().bind(self.clone(), Vec::new(), &tangent_inputs)?.remove(0);
                DifferentiationDual::new(primal, MaybeZero::Value(tangent))?
            }
            MaybeZero::Zero(_) => DifferentiationDual::new_with_zero_tangent(primal)?,
        }])
    }
}

impl<
    T: Type,
    U: DifferentiableType + ReferenceMemberType,
    Transform: ReferenceTransform<Type = U, Referent = T>,
    V: Value<Type = U>,
    O: Operation<Type = U>
        + ResidualZeroProvider<U, Operation = O>
        + OperationProvider<U, ReferenceNewOperation<<U as ReferenceMemberType>::Referent, U>, Operation = O>,
> TransposableOperation<V, O> for ReferenceReadOperation<T, U, Transform>
where
    ReferenceReadOperation<T, U, Transform>: Operation<Type = U>,
    Tracer<TracingContext<V, O>>: ReferenceAddUpdate<Transform>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        // A read is the identity map from the referenced state to its output, so its transpose accumulates the output's
        // cotangent into the cotangent reference of the read root, viewed exactly as the input views it. The reference
        // input carries no value cotangent of its own; its state cotangent lives in that accumulator.
        check_count!("input", inputs, 1 + self.binding_count(), ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
        if let MaybeZero::Value(cotangent) = &outputs[0] {
            let reference = context.cotangent_reference(driver, 0)?;
            let bindings = inputs[1..]
                .iter()
                .map(|input| {
                    input.as_known().cloned().ok_or_else(|| ProgramError::UnsupportedOperation {
                        message: "reference transform bindings must be known when transposing an access".to_string(),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            reference.add_update_through(cotangent, &self.transforms, &bindings)?;
        }
        Ok(())
    }
}

/// Capability to read an immutable snapshot from a reference value.
pub trait ReferenceRead<Transform: ReferenceTransform, Binding = Self, Output = Self>: Sized {
    /// Reads an immutable snapshot through the supplied transforms.
    ///
    /// # Parameters
    ///
    ///   - `transforms`: Transforms applied in order to the reference.
    ///   - `bindings`: Dynamic inputs in transform order.
    fn read_through(&self, transforms: &[Transform], bindings: &[Binding]) -> Result<Output, ProgramError>;

    /// Reads an immutable snapshot of the complete reference.
    #[inline]
    fn read(&self) -> Result<Output, ProgramError> {
        self.read_through(&[], &[])
    }
}

impl<A: Value<Type = ArrayType> + Concretizable<i128> + Reshape + Slice> ReferenceRead<ArrayReferenceTransform>
    for ArrayIrValue<A>
{
    fn read_through(&self, transforms: &[ArrayReferenceTransform], bindings: &[Self]) -> Result<Self, ProgramError> {
        let operation = ReferenceReadOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
        let operation = operation.with_transforms(transforms.to_vec());
        let mut input_types = vec![self.r#type().into_owned()];
        input_types.extend(bindings.iter().map(|value| value.r#type().into_owned()));
        operation.infer_output_types(&input_types, &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        let reference = reference.with_transforms(transforms, bindings)?;
        Ok(Self::Array(reference.read()?))
    }
}

impl<Transform, V> ReferenceRead<Transform, V, V> for V
where
    Transform: ReferenceTransform<Type = ArrayIrType, Referent = ArrayType>,
    V: Value<
            Type = ArrayIrType,
            DispatchDomain: Context<Operation: From<ReferenceReadOperation<ArrayType, ArrayIrType, Transform>>>,
        >,
{
    fn read_through(&self, transforms: &[Transform], bindings: &[Self]) -> Result<Self, ProgramError> {
        let mut inputs = vec![self.clone()];
        inputs.extend_from_slice(bindings);
        let operation = ReferenceReadOperation::new().with_transforms(transforms.to_vec());
        Ok(self.dispatch_domain().bind(operation, Vec::new(), &inputs)?.remove(0))
    }
}

impl<Transform, V> ReferenceRead<Transform, V, <V as ValueProjection<ArrayType>>::Projected>
    for ProjectedValue<ReferenceType<ArrayType>, V>
where
    Transform: ReferenceTransform<Type = ArrayIrType, Referent = ArrayType>,
    V: Value<
            Type = ArrayIrType,
            DispatchDomain: Context<Operation: From<ReferenceReadOperation<ArrayType, ArrayIrType, Transform>>>,
        > + ValueProjection<ArrayType>,
{
    fn read_through(
        &self,
        transforms: &[Transform],
        bindings: &[V],
    ) -> Result<<V as ValueProjection<ArrayType>>::Projected, ProgramError> {
        let mut inputs = vec![self.value().clone()];
        inputs.extend_from_slice(bindings);
        let operation = ReferenceReadOperation::new().with_transforms(transforms.to_vec());
        self.value()
            .dispatch_domain()
            .bind(operation, Vec::new(), &inputs)?
            .remove(0)
            .into_projected()
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayReference,
        ArrayReferenceTransform, ArrayReferenceTransformIndex, ArraySliceAxis, ArrayType, DataType, Dimension,
        DimensionBounds, DimensionType, DimensionValue, DimensionVariable, Shape,
    };
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{DifferentiationContext, DifferentiationDual, DifferentiationTracer};
    use crate::macros::{check_operation_partial_evaluation, check_operation_type_inference};
    use crate::operations::references::reference_freeze::ReferenceFreeze;
    use crate::operations::references::reference_new::{ReferenceNew, ReferenceNewOperation};
    use crate::operations::references::tests::*;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationValue, ReferencePlacement};
    use crate::programs::{EffectClass, EmptyRegionDriver, ProgramBuilder, ReferenceError};

    use super::*;

    type TestIrValue = ArrayIrValue<Array>;
    type TestIrOperation = ArrayIrOperation<Array>;
    type TestIrReferenceReadOperation = ReferenceReadOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>;

    #[test]
    fn test_reference_read() {
        let operation = Read::new();
        assert_eq!(Read::default().to_string(), operation.to_string());
        assert_eq!(operation.name(), REFERENCE_READ_OPERATION_NAME);
        assert_eq!(operation.to_string(), REFERENCE_READ_OPERATION_NAME);
        assert_eq!(
            format!("{operation:?}"),
            format!(
                "ReferenceReadOperation {{ transforms: [], marker: {:?} }}",
                PhantomData::<fn() -> (TestReferent, TestType)>,
            ),
        );
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedState));
        assert_eq!(
            operation.effects().reference_effects(),
            &[ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read }],
        );
    }

    #[test]
    fn test_reference_read_wrapped_transforms() {
        // A transform path too long to render inline wraps over multiple lines, indented relative to the line that owns
        // the access, while shorter paths (see `test_reference_read_type_inference_transforms`) stay inline.
        let operation = TestIrReferenceReadOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Slice {
                axes: vec![ArraySliceAxis::new(0, 2, 1), ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(2, 3, 1)],
            },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        assert_eq!(
            operation.to_string(),
            indoc! {"
                reference_read [
                    transforms=[index(axis=0, index=1), slice(axes=[0:2, 1:3, 2:5]), index(axis=0, index=dynamic)],
                ]"},
        );
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference =
            builder.add_input(ReferenceType::new(ArrayType::new_static(DataType::F32, [4, 5, 6, 7])).into());
        let index = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let output = builder.add_instruction(operation, Vec::new(), vec![reference, index], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ref<f32[4, 5, 6, 7]>, %1:i32[] .
                let %2:f32[2, 3] = reference_read [
                    transforms=[index(axis=0, index=1), slice(axes=[0:2, 1:3, 2:5]), index(axis=0, index=dynamic)],
                ] %0 %1
                in (%2)"},
        );
    }

    #[test]
    fn test_reference_read_type_inference() {
        let referent = TestReferent::new(7, 16);
        let value = TestType::Value(referent);
        let reference = TestType::Reference(ReferenceType::new(referent));
        check_operation_type_inference!(
            operation = Read::new(),
            cases = [
                {
                    input_types = [reference.clone()],
                    output_types = [value.clone()],
                },
                {
                    input_types = [],
                    error = "expected 1 input but got 0",
                },
                {
                    input_types = [value],
                    error = "expected reference type but got value type",
                },
            ],
        );
        let region = RegionInterface::new(Vec::new(), Vec::new(), EffectClasses::NONE);
        assert_eq!(
            Read::new().infer_output_types(std::slice::from_ref(&reference), std::slice::from_ref(&region)),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );

        // Reading only requires a universe that embeds referents and projects reference types.
        check_operation_type_inference!(
            operation = ReferenceReadOperation::<TestReferent, ReadFreezeUniverse>::new(),
            cases = [{
                input_types = [ReadFreezeUniverse::Reference(ReferenceType::new(referent))],
                output_types = [ReadFreezeUniverse::Value(referent)],
            }],
        );

        // The array universe reads array referents and rejects its other members.
        let array_type = ArrayType::scalar(DataType::F32);
        let dimension_type = DimensionType::new("n", DimensionBounds::unbounded());
        check_operation_type_inference!(
            operation = TestIrReferenceReadOperation::new(),
            cases = [
                {
                    input_types = [ArrayIrType::Reference(ReferenceType::new(array_type.clone()))],
                    output_types = [ArrayIrType::Array(array_type.clone())],
                },
                {
                    input_types = [ArrayIrType::Array(array_type)],
                    error = "expected reference type but got array type",
                },
                {
                    input_types = [ArrayIrType::Dimension(dimension_type)],
                    error = "expected reference type but got dimension type",
                },
            ],
        );
    }

    #[test]
    fn test_reference_read_type_inference_transforms() {
        let operation = TestIrReferenceReadOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        let reference = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2, 3])));
        let scalar = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let index = ArrayIrType::Array(ArrayType::scalar(DataType::I32));
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                { input_types = [reference.clone(), index.clone()], output_types = [scalar.clone()], },
                {
                    input_types = [reference.clone(), scalar.clone()],
                    error = "reference transform requires a scalar integer index but received `f32[]`",
                },
                { input_types = [reference.clone()], error = "expected 2 inputs but got 1", },
                { input_types = [reference, index.clone(), index.clone()], error = "expected 2 inputs but got 3", },
            ],
        );
        assert_eq!(
            operation.to_string(),
            "reference_read [transforms=[index(axis=0, index=1), index(axis=0, index=dynamic)]]",
        );
        let descriptor = operation.reference_access_descriptor(0).unwrap();
        assert_eq!(descriptor.transforms(), operation.transforms());
        assert_eq!(descriptor.bindings(), 1..2);
        assert!(operation.reference_access_descriptor(1).is_none());
    }

    #[test]
    fn test_reference_read_interpretation() {
        let context = EagerContext::<TestIrValue, TestIrOperation>::new();
        let initial = TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap());
        let reference = TestIrValue::Reference(ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]).unwrap()));

        // A read returns a snapshot of the current referent and leaves the reference live.
        assert_eq!(
            InterpretableOperation::<EagerContext<TestIrValue, TestIrOperation>>::interpret(
                &TestIrReferenceReadOperation::new(),
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&reference),
            ),
            Ok(vec![initial.clone()]),
        );
        assert_eq!(reference.read(), Ok(initial.clone()));

        // Only reference members can be read.
        assert_eq!(
            InterpretableOperation::<EagerContext<TestIrValue, TestIrOperation>>::interpret(
                &TestIrReferenceReadOperation::new(),
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&initial),
            ),
            Err(TypeError::invalid("expected reference type but got array type").into()),
        );
        assert_eq!(initial.read(), Err(TypeError::invalid("expected reference type but got array type").into()));

        // A read of a dynamically typed referent returns exactly the stored dynamic type and physical bytes. Equality
        // over a dynamically typed `Array` cannot address its elements, so the referent is unwrapped and compared by
        // its declared type and storage. `Array`'s checked constructors reject dynamically shaped types, so the
        // referent comes from the test-only unchecked hatch.
        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("length", DimensionBounds::unbounded()))]),
        );
        let initial_bytes = 1.0_f32.to_le_bytes().to_vec();
        let reference = TestIrValue::Reference(ArrayReference::new(Array::with_unchecked_type(
            dynamic_type.clone(),
            initial_bytes.clone(),
        )));
        let read = InterpretableOperation::<EagerContext<TestIrValue, TestIrOperation>>::interpret(
            &TestIrReferenceReadOperation::new(),
            &context,
            &EmptyRegionDriver,
            std::slice::from_ref(&reference),
        )
        .unwrap()
        .remove(0);
        let read = <TestIrValue as ValueProjection<ArrayType>>::into_projected(read).unwrap();
        assert_eq!(read.r#type().into_owned(), dynamic_type);
        assert_eq!(read.storage_bytes(), initial_bytes.as_slice());

        // Reading a consumed reference fails against the shared allocation state.
        reference.clone().freeze().unwrap();
        let error = InterpretableOperation::<EagerContext<TestIrValue, TestIrOperation>>::interpret(
            &TestIrReferenceReadOperation::new(),
            &context,
            &EmptyRegionDriver,
            std::slice::from_ref(&reference),
        )
        .unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_reference_read_interpretation_transforms() {
        let context = EagerContext::<TestIrValue, TestIrOperation>::new();
        let reference =
            TestIrValue::Reference(ArrayReference::new(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap()));
        let index = TestIrValue::Array(Array::scalar(-1i32).unwrap());
        let operation = TestIrReferenceReadOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        assert_eq!(
            operation.interpret(&context, &EmptyRegionDriver, &[reference.clone(), index]),
            Ok(vec![TestIrValue::Array(Array::scalar(6f32).unwrap())]),
        );
        assert_eq!(
            reference.read(),
            Ok(TestIrValue::Array(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap())),
        );
    }

    #[test]
    fn test_reference_read_partial_evaluation() {
        type TestContext = EagerContext<TestIrValue, TestIrOperation>;

        let value = TestIrValue::Array(Array::scalar(1.0_f32).unwrap());
        let live = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        let reference = PartialEvaluationValue::known(TestIrValue::Reference(live.clone()));

        // Under the default `Execute` placement a known reference folds the read against the live state, which the read
        // leaves untouched.
        let executing = PartialEvaluationContext::new(TestContext::new());
        let outputs = executing
            .fold_or_residualize(TestIrReferenceReadOperation::new(), Vec::new(), std::slice::from_ref(&reference))
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_known(), Some(&value));
        assert_eq!(live.read(), Ok(Array::scalar(1.0_f32).unwrap()));

        // Under the `Stage` placement the read stays residual regardless of input knowledge, so eager specialization
        // never observes live reference state.
        let staging =
            PartialEvaluationContext::new(TestContext::new()).with_reference_placement(ReferencePlacement::Stage);
        let outputs =
            staging.fold_or_residualize(TestIrReferenceReadOperation::new(), Vec::new(), &[reference]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_unknown());

        // Program-level partial evaluation uses the `Stage` placement, so both a known and an unknown reference retain
        // the read in the residual program and replay it against the runtime reference.
        let known = TestIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32).unwrap()));
        let replay = TestIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32).unwrap()));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = TestIrReferenceReadOperation::new(),
            cases = [
                {
                    inputs = [(@known, known)],
                    outputs = [(@residual, value.clone())],
                    residual_instructions = 1,
                },
                {
                    inputs = [(@unknown(type = reference_type, replay = replay))],
                    outputs = [(@residual, value)],
                    residual_instructions = 1,
                },
            ],
        );
    }

    #[test]
    fn test_reference_read_partial_evaluation_transforms() {
        // The known root enters the residual program while the dynamic index remains an ordinary unknown input.
        let live = ArrayReference::new(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap());
        check_operation_partial_evaluation!(
            backend = (TestIrValue, TestIrOperation),
            operation = TestIrReferenceReadOperation::new().with_transforms(vec![
                ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
                ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
            ]),
            cases = [
                {
                    inputs = [
                        (@known, TestIrValue::Reference(live.clone())),
                        (@unknown(
                            type = ArrayType::scalar(DataType::I32).into(),
                            replay = TestIrValue::Array(Array::scalar(2i32).unwrap()),
                        )),
                    ],
                    outputs = [(@residual, TestIrValue::Array(Array::scalar(6f32).unwrap()))],
                    residual_instructions = 1,
                },
            ],
        );
        assert_eq!(live.read(), Ok(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap()));

        // Under the default `Execute` placement a known root and a known binding fold the viewed read against the live
        // state, while the `Stage` placement keeps it residual.
        let operation = TestIrReferenceReadOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        let inputs = [
            PartialEvaluationValue::known(TestIrValue::Reference(live.clone())),
            PartialEvaluationValue::known(TestIrValue::Array(Array::scalar(0i32).unwrap())),
        ];
        let executing = PartialEvaluationContext::new(EagerContext::<TestIrValue, TestIrOperation>::new());
        let outputs = executing.fold_or_residualize(operation.clone(), Vec::new(), &inputs).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_known(), Some(&TestIrValue::Array(Array::scalar(4f32).unwrap())));
        let staging = PartialEvaluationContext::new(EagerContext::<TestIrValue, TestIrOperation>::new())
            .with_reference_placement(ReferencePlacement::Stage);
        let outputs = staging.fold_or_residualize(operation, Vec::new(), &inputs).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_unknown());
        assert_eq!(live.read(), Ok(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap()));
    }

    #[test]
    fn test_reference_read_batching() {
        let extent = TestIrValue::Dimension(
            DimensionValue::new(DimensionType::new("batch", DimensionBounds::unbounded()), 2).unwrap(),
        );
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<TestIrValue, TestIrOperation>::new(),
            extent,
        );
        let packed_type = ArrayType::new_static(DataType::F32, [3, 2]);
        let packed =
            TestIrValue::Array(Array::from_elements::<f32>(packed_type, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        let reference = packed.reference_new().unwrap();

        // Reading a batched reference yields the packed referent at the reference's batch axis.
        let input =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(reference.clone(), BatchAxis::new(1)).unwrap());
        let outputs = context.bind(ReferenceReadOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(1));
        assert_eq!(outputs[0].batch().value(), &packed);
        assert_eq!(outputs[0].r#type().as_ref(), &ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])));

        // Reading a replicated reference stays replicated.
        let input = BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(reference));
        let outputs = context.bind(ReferenceReadOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].batch().value(), &packed);
    }

    #[test]
    fn test_reference_read_batching_transforms() {
        let extent = TestIrValue::Dimension(
            DimensionValue::new(DimensionType::new("batch", DimensionBounds::unbounded()), 2).unwrap(),
        );
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<TestIrValue, TestIrOperation>::new(),
            extent,
        );
        let packed_type = ArrayType::new_static(DataType::F32, [2, 2, 3]);
        let reference = TestIrValue::Array(
            Array::from_elements::<f32>(packed_type.clone(), &[1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.])
                .unwrap(),
        )
        .reference_new()
        .unwrap();
        let input =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(reference.clone(), BatchAxis::new(1)).unwrap());
        let index = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::replicated(TestIrValue::Array(Array::scalar(2i32).unwrap())),
        );
        let operation = TestIrReferenceReadOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        let outputs = context.bind(operation, Vec::new(), &[input, index]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].batch().value(), &TestIrValue::Array(Array::vector(vec![9f32, 12.]).unwrap()));
        assert_eq!(
            reference.read(),
            Ok(TestIrValue::Array(
                Array::from_elements::<f32>(packed_type, &[1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.])
                    .unwrap()
            )),
        );
    }

    #[test]
    fn test_reference_read_differentiation() {
        let context = DifferentiationContext::fused(EagerContext::<TestIrValue, TestIrOperation>::new());
        let reference = TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()).reference_new().unwrap();
        let tangent_reference = TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap()).reference_new().unwrap();

        // Reading an active reference reads its tangent reference alongside.
        let input = DifferentiationTracer::new(
            DifferentiationDual::new(reference.clone(), tangent_reference).unwrap(),
            context.clone(),
        );
        let outputs = context.bind(ReferenceReadOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()));
        assert_eq!(
            outputs[0].tangent().as_value(),
            Some(&TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap()))
        );

        // Reading a plumbing reference yields a symbolic zero tangent of the referent's tangent type.
        let input =
            DifferentiationTracer::new(DifferentiationDual::new_with_zero_tangent(reference).unwrap(), context.clone());
        let outputs = context.bind(ReferenceReadOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(outputs[0].primal(), &TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()));
        assert!(matches!(
            outputs[0].tangent(),
            MaybeZero::Zero(r#type) if *r#type == ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])),
        ));
    }

    #[test]
    fn test_reference_read_differentiation_transforms() {
        let context = DifferentiationContext::fused(EagerContext::<TestIrValue, TestIrOperation>::new());
        let reference = TestIrValue::Array(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap())
            .reference_new()
            .unwrap();
        let tangent_reference = TestIrValue::Array(Array::matrix(2, 3, vec![10f32, 20., 30., 40., 50., 60.]).unwrap())
            .reference_new()
            .unwrap();
        let input = DifferentiationTracer::new(
            DifferentiationDual::new(reference.clone(), tangent_reference.clone()).unwrap(),
            context.clone(),
        );
        let index = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(TestIrValue::Array(Array::scalar(-1i32).unwrap())).unwrap(),
            context.clone(),
        );
        let operation = TestIrReferenceReadOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        let outputs = context.bind(operation, Vec::new(), &[input, index]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &TestIrValue::Array(Array::scalar(6f32).unwrap()));
        assert_eq!(outputs[0].tangent().as_value(), Some(&TestIrValue::Array(Array::scalar(60f32).unwrap())));
        assert_eq!(
            reference.read(),
            Ok(TestIrValue::Array(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap())),
        );
        assert_eq!(
            tangent_reference.read(),
            Ok(TestIrValue::Array(Array::matrix(2, 3, vec![10f32, 20., 30., 40., 50., 60.]).unwrap())),
        );
    }

    #[test]
    fn test_reference_read_transposition() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));

        // `r = new(v); y = read(r)`: the read is the identity from the referenced state to its output, so its transpose
        // accumulates `ȳ` into the cotangent reference of the read root, which the allocation then freezes into `v̄`.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(scalar_type.clone());
        let reference = builder
            .add_instruction(ReferenceNewOperation::<ArrayType, ArrayIrType>::new(), Vec::new(), vec![initial], None)
            .unwrap()[0];
        let output = builder
            .add_instruction(TestIrReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = zero [type=f32[]]
                    %2:ref<f32[]> = reference_new %1
                    () = reference_add_update %2 %0
                    %3:f32[] = reference_freeze %2
                in (%3)"},
        );
        assert_eq!(
            transposed.interpret(vec![TestIrValue::Array(Array::scalar(5.0_f32).unwrap())]),
            Ok(vec![TestIrValue::Array(Array::scalar(5.0_f32).unwrap())]),
        );

        // The rule accumulates through the cotangent reference of its input's root, which only a transposition
        // context scoped to the read instruction can resolve, so a detached context rejects a live output cotangent.
        let inputs = [PartialValue::Unknown(reference_type)];
        let tracing = TracingContext::<TestIrValue, TestIrOperation>::new();
        let cotangent = tracing.input(scalar_type);
        let mut context = TranspositionContext::new(tracing);
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        assert!(matches!(
            TestIrReferenceReadOperation::new().transpose(
                &mut context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Value(cotangent)],
                &accumulators,
            ),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message)))
                if message == "input 0 has no reference root in a transposition context that is not scoped to a \
                               reference-carrying instruction",
        ));

        // A symbolic zero output cotangent contributes nothing, so the rule never touches the cotangent reference.
        assert_eq!(
            TestIrReferenceReadOperation::new().transpose(
                &mut context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Zero(ArrayIrType::Array(ArrayType::scalar(DataType::F32)))],
                &accumulators,
            ),
            Ok(()),
        );
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_reference_read_transposition_transforms() {
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(ArrayType::new_static(DataType::F32, [2, 3]).into());
        let index = builder.add_constant(TestIrValue::Array(Array::scalar(-1i32).unwrap()));
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let operation = TestIrReferenceReadOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        let outputs = builder.add_instruction(operation, Vec::new(), vec![reference, index], None).unwrap().to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 1], vec![Placeholder; 1])
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:i32[] = const -1
                    %2:f32[2, 3] = zero [type=f32[2, 3]]
                    %3:ref<f32[2, 3]> = reference_new %2
                    () = reference_add_update [transforms=[index(axis=0, index=1), index(axis=0, index=dynamic)]] %3 %0 %1
                    %4:f32[2, 3] = reference_freeze %3
                in (%4)"},
        );
        assert_eq!(
            transposed.interpret(vec![TestIrValue::Array(Array::scalar(5f32).unwrap())]),
            Ok(vec![TestIrValue::Array(Array::matrix(2, 3, vec![0f32, 0., 0., 0., 0., 5.]).unwrap())])
        );

        // The adjoint accumulation addresses the same elements as the read, so its dynamic binding must be known when
        // transposing. Transposing with respect to the index leaves that binding unknown.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(ArrayType::new_static(DataType::F32, [2, 3]).into());
        let index = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let operation = TestIrReferenceReadOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        let outputs = builder.add_instruction(operation, Vec::new(), vec![reference, index], None).unwrap().to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 1])
            .unwrap();
        assert!(matches!(
            program.transpose_with_respect_to(&[0, 1], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "reference transform bindings must be known when transposing an access",
        ));
    }

    #[test]
    fn test_reference_read_reference_discharge() {
        // A read observes the allocation's current state without changing it, so the allocation stays unmutated.
        let (context, reference) = allocated_reference(4);
        let handle = ReferenceDischargeValue::Reference(reference.clone());
        assert_eq!(
            Read::new().discharge_references(&context, &EmptyRegionDriver, std::slice::from_ref(&handle)),
            Ok(vec![ReferenceDischargeValue::Value(TestValue::new(REFERENT, 4))]),
        );
        assert_eq!(context.is_mutated(reference.allocation_id()), Ok(false));

        // A value input denotes no allocation, so the rule reports what it expected instead of reading a value.
        let pure: TestDischargeValue = ReferenceDischargeValue::Value(TestValue::new(REFERENT, 4));
        assert_eq!(
            Read::new().discharge_references(&context, &EmptyRegionDriver, std::slice::from_ref(&pure)),
            Err(ProgramError::MalformedProgram(
                "reference discharge expected a reference to read but received a value".to_string(),
            )),
        );
    }

    #[test]
    fn test_reference_read_reference_discharge_transforms() {
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(ArrayType::new_static(DataType::F32, [2, 3]).into());
        let index = builder.add_constant(TestIrValue::Array(Array::scalar(-1i32).unwrap()));
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let operation = TestIrReferenceReadOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        let outputs = builder.add_instruction(operation, Vec::new(), vec![reference, index], None).unwrap().to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 1], vec![Placeholder; 1])
            .unwrap();
        let discharged = program.discharge_references(0).unwrap();
        assert!(!discharged.program().entry_region_ref().contains_reference_accesses_in_closure());
        assert_eq!(
            discharged
                .program()
                .interpret(vec![TestIrValue::Array(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap())]),
            Ok(vec![TestIrValue::Array(Array::scalar(6f32).unwrap())])
        );
    }

    #[test]
    fn test_reference_read_reference_discharge_views_out_of_range_indices() {
        // `y = read(r[transforms=[index(axis=0, index=dynamic)]](i))` selects the same element whether the transform
        // runs on eager reference handles or is discharged into dynamic slices. A negative signed index counts from the
        // end of the axis once and is then clamped, while an unsigned index keeps its full value until it is clamped.
        let signed = [(-9i64, 1f32), (-3, 1.), (-1, 3.), (0, 1.), (2, 3.), (7, 3.)]
            .map(|(index, expected)| (Array::scalar(index).unwrap(), expected));
        let unsigned =
            [(u64::MAX, 3f32), (u64::MAX - 1, 3.)].map(|(index, expected)| (Array::scalar(index).unwrap(), expected));
        for (index, expected) in signed.into_iter().chain(unsigned) {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let initial = builder.add_input(ArrayType::new_static(DataType::F32, [3]).into());
            let binding = builder.add_input(index.r#type().into_owned().into());
            let reference =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
            let operation = TestIrReferenceReadOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                axis: 0,
                index: ArrayReferenceTransformIndex::Dynamic,
            }]);
            let output = builder.add_instruction(operation, Vec::new(), vec![reference, binding], None).unwrap()[0];
            let program = builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap();
            let discharged = program.clone().discharge_references(0).unwrap();
            let inputs =
                vec![TestIrValue::Array(Array::vector(vec![1f32, 2., 3.]).unwrap()), TestIrValue::Array(index.clone())];
            let expected = Ok(vec![TestIrValue::Array(Array::scalar(expected).unwrap())]);
            assert_eq!(program.interpret(inputs.clone()), expected, "eager execution with index `{index}`");
            assert_eq!(discharged.program().interpret(inputs), expected, "discharged execution with index `{index}`");
        }
    }

    #[test]
    fn test_reference_read_projected() {
        type TestContext = TracingContext<TestIrValue, TestIrOperation>;
        type TestTracer = Tracer<TestContext>;

        // Reading through a projected reference member binds the operation through the parent tracer's context and
        // hands back the projected array member.
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2])));
        let (output_type, program) = TestContext::trace(
            |input: TestTracer| {
                let reference = <TestTracer as ValueProjection<ReferenceType<ArrayType>>>::into_projected(input)?;
                let read: ProjectedValue<ArrayType, TestTracer> = reference.read()?;
                Ok(read.into_value())
            },
            reference_type,
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ref<f32[2]> .
                let %1:f32[2] = reference_read %0
                in (%1)"},
        );
    }

    #[test]
    fn test_reference_read_staging() {
        type TestContext = TracingContext<TestIrValue, TestIrOperation>;

        // A staged read is the native `reference_read` variant of the array operation family.
        let (output_type, program) = TestContext::trace(
            |input| input.read(),
            ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))),
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ref<f32[]> .
                let %1:f32[] = reference_read %0
                in (%1)"},
        );
        assert_eq!(program.effects().classes(), EffectClasses::single(EffectClass::OrderedState));
    }
}
