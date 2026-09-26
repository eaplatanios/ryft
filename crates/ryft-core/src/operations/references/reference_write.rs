use std::borrow::Cow;
use std::fmt::Display;
use std::marker::PhantomData;
use std::sync::LazyLock;

use crate::arrays::{ArrayIrType, ArrayIrValue, ArrayReferenceTransform, ArrayType};
use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{DifferentiableType, ResidualZeroProvider};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::constants::zero::Zero;
use crate::operations::manipulation::reshaping::Reshape;
use crate::operations::manipulation::slicing::{Slice, UpdateSlice};
use crate::operations::references::reference_swap::ReferenceSwapOperation;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    BatchableReferenceTransform, Concretizable, EffectClasses, Effects, MaybeZero, NoReferenceTransform, Operation,
    OperationFormatter, ProgramError, ProjectedValue, ReferenceAccessDescriptor, ReferenceAccessMode,
    ReferenceAccessOperation, ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy,
    ReferenceDischargeValue, ReferenceDischargeableOperation, ReferenceEffect, ReferenceTransform, ReferenceType,
    RegionInterface, Type, TypeError, Typed, Value, ValueProjection, batch_reference_transforms,
    infer_reference_view_type,
};

/// Canonical operation name for [`ReferenceWriteOperation`].
pub const REFERENCE_WRITE_OPERATION_NAME: &str = "reference_write";

/// Replaces the selected reference state with an exactly matching value without observing its previous contents.
/// The root and replacement are the first two inputs. Dynamic bindings follow them in transform order and the path is
/// stored in `Transform` metadata. An empty path accesses the complete referent.
#[derive(Clone, Debug)]
pub struct ReferenceWriteOperation<
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

impl<T: Type, U: Type, Transform: ReferenceTransform<Type = U, Referent = T>> ReferenceWriteOperation<T, U, Transform> {
    /// Creates a new [`ReferenceWriteOperation`].
    pub const fn new() -> Self {
        Self { transforms: Vec::new(), marker: PhantomData }
    }

    /// Returns a copy of this [`ReferenceWriteOperation`] with the provided transforms applied to the reference input.
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

    /// Returns the number of dynamic inputs supplied after the base inputs for this [`ReferenceWriteOperation`].
    fn binding_count(&self) -> usize {
        self.transforms.iter().map(ReferenceTransform::binding_count).sum()
    }
}

impl<T: Type, U: Type, Transform: ReferenceTransform<Type = U, Referent = T>> Default
    for ReferenceWriteOperation<T, U, Transform>
{
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Type, U: Type, Transform: ReferenceTransform<Type = U, Referent = T>> Display
    for ReferenceWriteOperation<T, U, Transform>
where
    Self: Operation,
{
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type, U: Type, Transform: ReferenceTransform<Type = U, Referent = T>> Operation
    for ReferenceWriteOperation<T, U, Transform>
where
    for<'t> &'t T: TryFrom<&'t U, Error = TypeError>,
    for<'t> &'t ReferenceType<T>: TryFrom<&'t U, Error = TypeError>,
{
    type Type = U;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_WRITE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[U],
        region_interfaces: &[RegionInterface<U>],
    ) -> Result<Vec<U>, TypeError> {
        check_count!("input", input_types, 2 + self.binding_count(), TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let reference = <&ReferenceType<T>>::try_from(&input_types[0])?;
        let binding_types = input_types[2..].iter().collect::<Vec<_>>();
        let referent = infer_reference_view_type(
            reference.referent(),
            &self.transforms,
            &binding_types,
            ReferenceAccessMode::Write,
        )?;
        let replacement = <&T>::try_from(&input_types[1])?;
        if replacement != &referent {
            return Err(TypeError::invalid(format!(
                "`{}` replacement type `{}` must exactly match reference referent type `{}`",
                REFERENCE_WRITE_OPERATION_NAME, replacement, referent,
            )));
        }
        Ok(Vec::new())
    }

    #[inline]
    fn effects(&self) -> Cow<'_, Effects> {
        // Share one descriptor across all type instantiations to avoid allocating and validating it on every query.
        static EFFECTS: LazyLock<Effects> = LazyLock::new(|| {
            Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Write }],
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
    for ReferenceWriteOperation<T, U, Transform>
where
    ReferenceWriteOperation<T, U, Transform>: Operation<Type = U>,
{
    type Transform = Transform;

    #[inline]
    fn base_input_count(&self) -> usize {
        2
    }

    #[inline]
    fn reference_access_descriptor(&self, input_index: usize) -> Option<ReferenceAccessDescriptor<'_, Transform>> {
        (input_index == 0).then(|| ReferenceAccessDescriptor::new(&self.transforms, 2..2 + self.binding_count()))
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
    U: Type + From<T> + From<ReferenceType<T>>,
    Transform: ReferenceTransform<Type = U, Referent = T>,
    C: Context<Type = U>,
    P: ReferenceDischargePolicy<C, Referent = T, Transform = Transform>,
> ReferenceDischargeableOperation<C, P> for ReferenceWriteOperation<T, U, Transform>
where
    ReferenceWriteOperation<T, U, Transform>: Operation<Type = U>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        _driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        check_count!("input", inputs, 2 + self.binding_count(), ProgramError);
        let reference = inputs[0].try_as_reference("a reference to write")?;
        let bindings = inputs[2..]
            .iter()
            .map(|input| input.try_as_value("a reference transform binding").cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let replacement = inputs[1].try_as_value("a replacement value")?.clone();

        // Rules can be called outside a validated program, so check the input types before changing reference state.
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(&input_types, &[])?;
        context.write_through(reference, replacement, &self.transforms, &bindings)?;
        Ok(Vec::new())
    }
}

impl<
    T: Type,
    U: Type,
    Transform: ReferenceTransform<Type = U, Referent = T>,
    C: Domain<Type = U, Value: ReferenceWrite<Transform, C::Value, C::Value>>,
> InterpretableOperation<C> for ReferenceWriteOperation<T, U, Transform>
where
    ReferenceWriteOperation<T, U, Transform>: Operation<Type = U>,
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2 + self.binding_count(), ProgramError);
        inputs[0].write_through(&inputs[1], &self.transforms, &inputs[2..])?;
        Ok(Vec::new())
    }
}

impl<
    T: Type,
    U: Type,
    Transform: ReferenceTransform<Type = U, Referent = T>,
    C: Context<Type = U, Operation: From<ReferenceWriteOperation<T, U, Transform>>>,
> PartiallyEvaluatableOperation<C> for ReferenceWriteOperation<T, U, Transform>
{
}

impl<
    T: Type,
    U: Type + From<ReferenceType<T>>,
    Transform: BatchableReferenceTransform<Type = U, Referent = T>,
    C: Context<Type = U, Operation: From<ReferenceWriteOperation<T, U, Transform>>>,
    P: BatchingPolicy<C>,
> BatchableOperation<C, P> for ReferenceWriteOperation<T, U, Transform>
where
    for<'t> &'t ReferenceType<T>: TryFrom<&'t U, Error = TypeError>,
    ReferenceWriteOperation<T, U, Transform>: Operation<Type = U>,
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        check_count!("input", inputs, 2 + self.binding_count(), ProgramError);
        let binding_axes = inputs[2..].iter().map(P::batch_axis).collect::<Vec<_>>();
        let (transforms, output_axis) = batch_reference_transforms(
            P::value(&inputs[0]).r#type().as_ref(),
            P::batch_axis(&inputs[0]),
            &self.transforms,
            &binding_axes,
        )?;

        // Broadcast or move the stored value to the final selected referent's batch axis.
        let replacement = match (output_axis.axis(), P::batch_axis(&inputs[1]).axis()) {
            (Some(axis), _) => driver.align_batch_axis(context, inputs[1].clone(), axis)?,
            (None, None) => inputs[1].clone(),
            (None, Some(_)) => {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "`{REFERENCE_WRITE_OPERATION_NAME}` cannot store a batched value into an unbatched reference; \
                         pass the reference as a batched input instead",
                    ),
                });
            }
        };

        let mut values = vec![P::value(&inputs[0]).clone(), P::value(&replacement).clone()];
        values.extend(inputs[2..].iter().map(|input| P::value(input).clone()));
        context.parent().bind(self.clone().with_transforms(transforms), Vec::new(), &values)?;
        Ok(Vec::new().into())
    }
}

impl_differentiable_operation! {
    <T, U, Transform> ReferenceWriteOperation<T, U, Transform>,
    jvp<C>
    where
        T: Type,
        U: DifferentiableType,
        Transform: ReferenceTransform<Type = U, Referent = T>,
        C: Context<
            Type = U,
            Operation: From<ReferenceWriteOperation<T, U, Transform>>
                + ResidualZeroProvider<U, Operation = C::Operation>,
        > + Zero<C::Value>,
    {
        |operation, context, _driver, inputs| {
            // Store the replacement tangent alongside the primal replacement.
            check_count!("input", inputs, 2 + operation.binding_count(), ProgramError);
            // Reject a live tangent without tangent storage before either reference can be changed.
            let stored = match (inputs[0].tangent(), inputs[1].tangent()) {
                (MaybeZero::Value(reference), tangent) => Some((reference, tangent.clone())),
                (MaybeZero::Zero(_), MaybeZero::Zero(_)) => None,
                (MaybeZero::Zero(_), MaybeZero::Value(_)) => {
                    return Err(ProgramError::InvalidArgument {
                        message: format!(
                            "`{REFERENCE_WRITE_OPERATION_NAME}` writes a live tangent into a reference that carries no \
                             tangent; pass the reference as a differentiated input instead of capturing it",
                        ),
                    }
                    .into());
                }
            };
            let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            context.primal().bind(operation.clone(), Vec::new(), &primal_inputs)?;
            if let Some((tangent_reference, tangent)) = stored {
                // A zero replacement tangent is instantiated because the tangent reference must observe the store.
                let source = context.primal_to_tangent(inputs[1].primal().clone())?;
                let tangent = C::Operation::materialize_zero_from_residual_sources(
                    context.tangent(), tangent, std::iter::once(&source),
                )?;
                let mut tangent_inputs = vec![tangent_reference.clone(), tangent];
                for input in &inputs[2..] {
                    tangent_inputs.push(context.primal_to_tangent(input.primal().clone())?);
                }
                context.tangent().bind(operation.clone(), Vec::new(), &tangent_inputs)?;
            }
            Ok(Vec::new())
        }
    },
    transpose<V, O>
    where
        T: Type,
        U: DifferentiableType,
        Transform: ReferenceTransform<Type = U, Referent = T>,
        V: Value<Type = U>,
        O: Operation<Type = U>
            + ResidualZeroProvider<U, Operation = O>
            + From<ReferenceSwapOperation<T, U, Transform>>,
        ReferenceSwapOperation<T, U, Transform>: Operation<Type = U>,
    {
        |operation, context, driver, inputs, outputs, accumulators| {
            // A write is a swap whose previous contents are discarded, so its transpose is the swap transpose with a
            // zero output cotangent: the cotangent reference is reset to zero (the pre-execution state no longer flows
            // into anything) and its previous contents become the cotangent of the stored value. An accumulator that
            // nothing has reached yet already holds zero, so nothing is staged and the stored value's cotangent stays
            // symbolic.
            check_count!("input", inputs, 2 + operation.binding_count(), ProgramError);
            check_count!("output", outputs, 0, ProgramError);
            check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
            let value_cotangent_type = inputs[1].r#type().cotangent()?;
            let Some(accumulator) = context.cotangent_reference_if_allocated(driver, 0)? else {
                return Ok(());
            };
            let zero = O::materialize_zero_from_residual_sources(
                &**context,
                MaybeZero::Zero(value_cotangent_type),
                context
                    .dimension_sources()
                    .chain(inputs.iter().filter_map(PartialValue::as_known))
                    .chain(std::iter::once(&accumulator)),
            )?;
            let mut adjoint_inputs = vec![accumulator, zero];
            for input in &inputs[2..] {
                adjoint_inputs.push(input.as_known().cloned().ok_or_else(|| ProgramError::UnsupportedOperation {
                    message: "reference transform bindings must be known when transposing an access".to_string(),
                })?);
            }
            let previous = context.bind(
                ReferenceSwapOperation::<T, U, Transform>::new().with_transforms(operation.transforms.clone()),
                Vec::new(), &adjoint_inputs,
            )?.remove(0);
            accumulators[1].accumulate(context, MaybeZero::Value(previous))
        }
    },
}

/// Capability to replace the value stored by a reference without observing the previous value.
pub trait ReferenceWrite<Transform: ReferenceTransform, Binding = Self, Replacement = Self>: Sized {
    /// Replaces the state selected by the supplied transforms with `replacement` in program order, without observing
    /// the previous value.
    ///
    /// # Parameters
    ///
    ///   - `replacement`: Value stored into the viewed reference state, whose type must exactly match the selected
    ///     referent type.
    ///   - `transforms`: Transforms applied in order to the reference.
    ///   - `bindings`: Dynamic inputs in transform order.
    fn write_through(
        &self,
        replacement: &Replacement,
        transforms: &[Transform],
        bindings: &[Binding],
    ) -> Result<(), ProgramError>;

    /// Replaces the complete referent with `replacement` in program order, without observing the previous value.
    #[inline]
    fn write(&self, replacement: &Replacement) -> Result<(), ProgramError> {
        self.write_through(replacement, &[], &[])
    }
}

impl<A: Value<Type = ArrayType> + Concretizable<i128> + Reshape + Slice + UpdateSlice>
    ReferenceWrite<ArrayReferenceTransform> for ArrayIrValue<A>
{
    fn write_through(
        &self,
        replacement: &Self,
        transforms: &[ArrayReferenceTransform],
        bindings: &[Self],
    ) -> Result<(), ProgramError> {
        let operation = ReferenceWriteOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
        let operation = operation.with_transforms(transforms.to_vec());
        let mut input_types = vec![self.r#type().into_owned(), replacement.r#type().into_owned()];
        input_types.extend(bindings.iter().map(|value| value.r#type().into_owned()));
        operation.infer_output_types(&input_types, &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        let reference = reference.with_transforms(transforms, bindings)?;
        let replacement = <Self as ValueProjection<ArrayType>>::projected(replacement)?;
        reference.write(replacement.clone())
    }
}

impl<Transform, V> ReferenceWrite<Transform, V, V> for V
where
    Transform: ReferenceTransform<Type = ArrayIrType, Referent = ArrayType>,
    V: Value<
            Type = ArrayIrType,
            DispatchDomain: Context<Operation: From<ReferenceWriteOperation<ArrayType, ArrayIrType, Transform>>>,
        >,
{
    fn write_through(
        &self,
        replacement: &Self,
        transforms: &[Transform],
        bindings: &[Self],
    ) -> Result<(), ProgramError> {
        let mut inputs = vec![self.clone(), replacement.clone()];
        inputs.extend_from_slice(bindings);
        let operation = ReferenceWriteOperation::new().with_transforms(transforms.to_vec());
        self.dispatch_domain().bind(operation, Vec::new(), &inputs)?;
        Ok(())
    }
}

impl<Transform, V> ReferenceWrite<Transform, V, ProjectedValue<ArrayType, V>>
    for ProjectedValue<ReferenceType<ArrayType>, V>
where
    Transform: ReferenceTransform<Type = ArrayIrType, Referent = ArrayType>,
    V: Value<
            Type = ArrayIrType,
            DispatchDomain: Context<Operation: From<ReferenceWriteOperation<ArrayType, ArrayIrType, Transform>>>,
        >,
{
    fn write_through(
        &self,
        replacement: &ProjectedValue<ArrayType, V>,
        transforms: &[Transform],
        bindings: &[V],
    ) -> Result<(), ProgramError> {
        let mut inputs = vec![self.value().clone(), replacement.value().clone()];
        inputs.extend_from_slice(bindings);
        let operation = ReferenceWriteOperation::new().with_transforms(transforms.to_vec());
        self.value().dispatch_domain().bind(operation, Vec::new(), &inputs)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrValue, ArrayReference,
        ArrayReferenceTransform, ArrayReferenceTransformIndex, ArrayType, DataType, DimensionBounds, DimensionType,
        DimensionValue,
    };
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
    use crate::contexts::EagerContext;
    use crate::differentiation::{
        DifferentiationContext, DifferentiationDual, DifferentiationError, DifferentiationTracer,
        TransposableOperation, TranspositionContext,
    };
    use crate::macros::{check_operation_partial_evaluation, check_operation_type_inference};
    use crate::operations::references::reference_freeze::{ReferenceFreeze, ReferenceFreezeOperation};
    use crate::operations::references::reference_new::{ReferenceNew, ReferenceNewOperation};
    use crate::operations::references::reference_read::ReferenceRead;
    use crate::operations::references::tests::*;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationValue, ReferencePlacement};
    use crate::programs::{EffectClass, EmptyRegionDriver, ProgramBuilder};
    use crate::tracing::{Tracer, TracingContext};

    use super::*;

    type TestIrValue = ArrayIrValue<Array>;
    type TestIrOperation = ArrayIrOperation<Array>;
    type TestIrContext = EagerContext<TestIrValue, TestIrOperation>;
    type TestIrReferenceWriteOperation = ReferenceWriteOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>;

    #[test]
    fn test_reference_write() {
        let operation = Write::new();
        assert_eq!(Write::default().to_string(), operation.to_string());
        assert_eq!(operation.name(), REFERENCE_WRITE_OPERATION_NAME);
        assert_eq!(operation.to_string(), REFERENCE_WRITE_OPERATION_NAME);
        assert_eq!(
            format!("{operation:?}"),
            format!(
                "ReferenceWriteOperation {{ transforms: [], marker: {:?} }}",
                PhantomData::<fn() -> (TestReferent, TestType)>
            ),
        );

        // A write orders against other state effects, accesses its reference input for writing, and aliases nothing.
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedState));
        assert_eq!(
            operation.effects().reference_effects(),
            &[ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Write }]
        );
    }

    #[test]
    fn test_reference_write_type_inference() {
        let referent = TestReferent::new(7, 16);
        let value = TestType::Value(referent);
        let reference = TestType::Reference(ReferenceType::new(referent));
        check_operation_type_inference!(
            operation = Write::new(),
            cases = [
                {
                    input_types = [reference.clone(), value.clone()],
                    output_types = [],
                },
                {
                    input_types = [reference.clone(), TestType::Value(TestReferent::new(7, 32))],
                    error = "`reference_write` replacement type `value<i7,p32>` must exactly match reference referent \
                             type `value<i7,p16>`",
                },
                {
                    input_types = [reference.clone()],
                    error = "expected 2 inputs but got 1",
                },
                {
                    input_types = [value.clone(), value.clone()],
                    error = "expected reference type but got value type",
                },
                {
                    input_types = [reference.clone(), reference.clone()],
                    error = "expected value type but got reference type",
                },
            ],
        );
        let region = RegionInterface::new(Vec::new(), Vec::new(), EffectClasses::NONE);
        assert_eq!(
            Write::new().infer_output_types(&[reference, value], std::slice::from_ref(&region)),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );

        // A universe only needs to project references and values out of itself for a write to type-check.
        check_operation_type_inference!(
            operation = ReferenceWriteOperation::<TestReferent, StoreUniverse>::new(),
            cases = [{
                input_types = [StoreUniverse::Reference(ReferenceType::new(referent)), StoreUniverse::Value(referent)],
                output_types = [],
            }],
        );

        let vector_type = ArrayType::new_static(DataType::F32, [2]);
        check_operation_type_inference!(
            operation = TestIrReferenceWriteOperation::new(),
            cases = [
                {
                    input_types = [
                        ArrayIrType::Reference(ReferenceType::new(vector_type.clone())),
                        ArrayIrType::Array(vector_type.clone()),
                    ],
                    output_types = [],
                },
                {
                    input_types = [
                        ArrayIrType::Reference(ReferenceType::new(vector_type.clone())),
                        ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])),
                    ],
                    error = "`reference_write` replacement type `f32[3]` must exactly match reference referent type \
                             `f32[2]`",
                },
                {
                    input_types = [ArrayIrType::Array(vector_type.clone()), ArrayIrType::Array(vector_type.clone())],
                    error = "expected reference type but got array type",
                },
                {
                    input_types = [
                        ArrayIrType::Reference(ReferenceType::new(vector_type.clone())),
                        ArrayIrType::Reference(ReferenceType::new(vector_type)),
                    ],
                    error = "expected array type but got reference type",
                },
            ],
        );
    }

    #[test]
    fn test_reference_write_type_inference_transforms() {
        let operation = TestIrReferenceWriteOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        let reference = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2, 3])));
        let scalar = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let index = ArrayIrType::Array(ArrayType::scalar(DataType::I32));
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                { input_types = [reference.clone(), scalar.clone(), index.clone()], output_types = [], },
                {
                    input_types = [reference.clone(), scalar.clone(), scalar.clone()],
                    error = "reference transform requires a scalar integer index but received `f32[]`",
                },
                { input_types = [reference.clone(), scalar.clone()], error = "expected 3 inputs but got 2", },
                {
                    input_types = [reference, scalar, index.clone(), index],
                    error = "expected 3 inputs but got 4",
                },
            ],
        );
        assert_eq!(
            operation.to_string(),
            "reference_write [transforms=[index(axis=0, index=1), index(axis=0, index=dynamic)]]",
        );
        let descriptor = operation.reference_access_descriptor(0).unwrap();
        assert_eq!(descriptor.transforms(), operation.transforms());
        assert_eq!(descriptor.bindings(), 2..3);
        assert!(operation.reference_access_descriptor(1).is_none());
    }

    #[test]
    fn test_reference_write_interpretation() {
        let live = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]).unwrap());
        let reference = TestIrValue::Reference(live.clone());

        // A replacement carrying exactly the referent type replaces the stored value and produces no output.
        assert_eq!(
            InterpretableOperation::<TestIrContext>::interpret(
                &TestIrReferenceWriteOperation::new(),
                &TestIrContext::new(),
                &EmptyRegionDriver,
                &[reference.clone(), TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap())],
            ),
            Ok(Vec::new()),
        );
        assert_eq!(live.read(), Ok(Array::vector(vec![3.0_f32, 4.0]).unwrap()));

        // Exact input inference runs before the store, so a rejected replacement leaves the stored value unchanged.
        assert_eq!(
            InterpretableOperation::<TestIrContext>::interpret(
                &TestIrReferenceWriteOperation::new(),
                &TestIrContext::new(),
                &EmptyRegionDriver,
                &[reference.clone(), TestIrValue::Array(Array::vector(vec![5.0_f32, 6.0, 7.0]).unwrap())],
            ),
            Err(TypeError::invalid(
                "`reference_write` replacement type `f32[3]` must exactly match reference referent type `f32[2]`",
            )
            .into()),
        );
        assert_eq!(live.read(), Ok(Array::vector(vec![3.0_f32, 4.0]).unwrap()));

        // Each input must be the member kind the operation expects.
        let array = TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap());
        assert_eq!(
            InterpretableOperation::<TestIrContext>::interpret(
                &TestIrReferenceWriteOperation::new(),
                &TestIrContext::new(),
                &EmptyRegionDriver,
                &[array.clone(), array],
            ),
            Err(TypeError::invalid("expected reference type but got array type").into()),
        );
        assert_eq!(
            InterpretableOperation::<TestIrContext>::interpret(
                &TestIrReferenceWriteOperation::new(),
                &TestIrContext::new(),
                &EmptyRegionDriver,
                &[reference.clone(), reference],
            ),
            Err(TypeError::invalid("expected array type but got reference type").into()),
        );
        assert_eq!(live.read(), Ok(Array::vector(vec![3.0_f32, 4.0]).unwrap()));
    }

    #[test]
    fn test_reference_write_interpretation_transforms() {
        let context = EagerContext::<TestIrValue, TestIrOperation>::new();
        let reference =
            TestIrValue::Reference(ArrayReference::new(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap()));
        let index = TestIrValue::Array(Array::scalar(-1i32).unwrap());
        let operation = TestIrReferenceWriteOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        assert_eq!(
            operation.interpret(
                &context,
                &EmptyRegionDriver,
                &[reference.clone(), TestIrValue::Array(Array::scalar(9f32).unwrap()), index]
            ),
            Ok(Vec::new()),
        );
        assert_eq!(
            reference.read(),
            Ok(TestIrValue::Array(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 9.]).unwrap())),
        );
    }

    #[test]
    fn test_reference_write_partial_evaluation() {
        // Program replay uses the `Stage` placement: a write stages regardless of input knowledge, the live handle
        // is passed to the residual program as a known reference input, and the store runs only when that program
        // runs.
        let live = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        check_operation_partial_evaluation!(
            backend = (TestIrValue, TestIrOperation),
            operation = TestIrReferenceWriteOperation::new(),
            cases = [
                {
                    inputs = [
                        (@known, TestIrValue::Reference(live.clone())),
                        (@known, TestIrValue::Array(Array::scalar(2.0_f32).unwrap())),
                    ],
                    outputs = [],
                    residual_instructions = 1,
                },
                {
                    inputs = [
                        (@known, TestIrValue::Reference(live.clone())),
                        (@unknown(
                            type = ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
                            replay = TestIrValue::Array(Array::scalar(3.0_f32).unwrap())
                        )),
                    ],
                    outputs = [],
                    residual_instructions = 1,
                },
            ],
        );
        assert_eq!(live.read(), Ok(Array::scalar(3.0_f32).unwrap()));

        // Under the `Stage` placement the live state is untouched at partial evaluation time even when every input
        // is known.
        let reference = PartialEvaluationValue::known(TestIrValue::Reference(live.clone()));
        let replacement = PartialEvaluationValue::known(TestIrValue::Array(Array::scalar(4.0_f32).unwrap()));
        let staging =
            PartialEvaluationContext::new(TestIrContext::new()).with_reference_placement(ReferencePlacement::Stage);
        let outputs = staging.fold_or_residualize(
            TestIrReferenceWriteOperation::new(),
            Vec::new(),
            &[reference.clone(), replacement.clone()],
        );
        assert_eq!(outputs.map(|outputs| outputs.len()), Ok(0));
        assert_eq!(live.read(), Ok(Array::scalar(3.0_f32).unwrap()));

        // Under the default `Execute` placement an all-known write folds: it runs against the live state in program
        // order at partial evaluation time.
        let executing = PartialEvaluationContext::new(TestIrContext::new());
        let outputs =
            executing.fold_or_residualize(TestIrReferenceWriteOperation::new(), Vec::new(), &[reference, replacement]);
        assert_eq!(outputs.map(|outputs| outputs.len()), Ok(0));
        assert_eq!(live.read(), Ok(Array::scalar(4.0_f32).unwrap()));
    }

    #[test]
    fn test_reference_write_partial_evaluation_transforms() {
        // The known root enters the residual program while the dynamic index remains an ordinary unknown input.
        let live = ArrayReference::new(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap());
        check_operation_partial_evaluation!(
            backend = (TestIrValue, TestIrOperation),
            operation = TestIrReferenceWriteOperation::new().with_transforms(vec![
                ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
                ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
            ]),
            cases = [
                {
                    inputs = [
                        (@known, TestIrValue::Reference(live.clone())),
                        (@known, TestIrValue::Array(Array::scalar(2f32).unwrap())),
                        (@unknown(
                            type = ArrayType::scalar(DataType::I32).into(),
                            replay = TestIrValue::Array(Array::scalar(2i32).unwrap()),
                        )),
                    ],
                    outputs = [],
                    residual_instructions = 1,
                },
            ],
        );
        assert_eq!(live.read(), Ok(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 2.]).unwrap()));

        // Under the `Stage` placement a known root and a known binding leave the live state untouched, while the
        // default `Execute` placement folds the viewed write by running it against the live state.
        let operation = TestIrReferenceWriteOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        let inputs = [
            PartialEvaluationValue::known(TestIrValue::Reference(live.clone())),
            PartialEvaluationValue::known(TestIrValue::Array(Array::scalar(7f32).unwrap())),
            PartialEvaluationValue::known(TestIrValue::Array(Array::scalar(0i32).unwrap())),
        ];
        let staging =
            PartialEvaluationContext::new(TestIrContext::new()).with_reference_placement(ReferencePlacement::Stage);
        let outputs = staging.fold_or_residualize(operation.clone(), Vec::new(), &inputs);
        assert_eq!(outputs.map(|outputs| outputs.len()), Ok(0));
        assert_eq!(live.read(), Ok(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 2.]).unwrap()));
        let executing = PartialEvaluationContext::new(TestIrContext::new());
        let outputs = executing.fold_or_residualize(operation, Vec::new(), &inputs);
        assert_eq!(outputs.map(|outputs| outputs.len()), Ok(0));
        assert_eq!(live.read(), Ok(Array::matrix(2, 3, vec![1f32, 2., 3., 7., 5., 2.]).unwrap()));
    }

    #[test]
    fn test_reference_write_batching() {
        let extent = TestIrValue::Dimension(
            DimensionValue::new(DimensionType::new("batch", DimensionBounds::unbounded()), 2).unwrap(),
        );
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(TestIrContext::new(), extent);
        let packed_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let reference = TestIrValue::Array(Array::from_elements::<f32>(packed_type.clone(), &[0.0; 6]).unwrap())
            .reference_new()
            .unwrap();
        let batched =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(reference.clone(), BatchAxis::new(0)).unwrap());

        // A replacement mapped at the reference's batch axis is stored packed.
        let aligned = TestIrValue::Array(
            Array::from_elements::<f32>(packed_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        );
        let replacement =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(aligned.clone(), BatchAxis::new(0)).unwrap());
        let outputs =
            context.bind(ReferenceWriteOperation::new(), Vec::new(), &[batched.clone(), replacement]).unwrap();
        assert!(outputs.is_empty());
        assert_eq!(reference.read(), Ok(aligned));

        // A replicated replacement is broadcast along the reference's batch axis.
        let replacement = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::replicated(TestIrValue::Array(Array::vector(vec![7.0_f32, 8.0, 9.0]).unwrap())),
        );
        context.bind(ReferenceWriteOperation::new(), Vec::new(), &[batched.clone(), replacement]).unwrap();
        assert_eq!(
            reference.read(),
            Ok(TestIrValue::Array(
                Array::from_elements::<f32>(packed_type.clone(), &[7.0, 8.0, 9.0, 7.0, 8.0, 9.0]).unwrap()
            )),
        );

        // A replacement mapped at another axis is moved to the reference's batch axis.
        let transposed_type = ArrayType::new_static(DataType::F32, [3, 2]);
        let transposed =
            TestIrValue::Array(Array::from_elements::<f32>(transposed_type, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap());
        let replacement =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(transposed, BatchAxis::new(1)).unwrap());
        context.bind(ReferenceWriteOperation::new(), Vec::new(), &[batched, replacement]).unwrap();
        assert_eq!(
            reference.read(),
            Ok(TestIrValue::Array(
                Array::from_elements::<f32>(packed_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()
            )),
        );

        // A replicated replacement is stored plainly into a replicated reference, while a batched replacement has no
        // batch axis to land in and the user is told to batch the reference instead.
        let unbatched = TestIrValue::Array(Array::vector(vec![0.0_f32, 0.0, 0.0]).unwrap()).reference_new().unwrap();
        let replicated = BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(unbatched.clone()));
        let replacement = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::replicated(TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap())),
        );
        context
            .bind(ReferenceWriteOperation::new(), Vec::new(), &[replicated.clone(), replacement])
            .unwrap();
        assert_eq!(unbatched.read(), Ok(TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap())));
        let replacement = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::new(
                TestIrValue::Array(Array::from_elements::<f32>(packed_type, &[0.0; 6]).unwrap()),
                BatchAxis::new(0),
            )
            .unwrap(),
        );
        let error = context.bind(ReferenceWriteOperation::new(), Vec::new(), &[replicated, replacement]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<BatchingError>(),
            Some(&BatchingError::UnsupportedOperation {
                message: "`reference_write` cannot store a batched value into an unbatched reference; pass the \
                          reference as a batched input instead"
                    .to_string(),
            }),
        );
        assert_eq!(unbatched.read(), Ok(TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap())));
    }

    #[test]
    fn test_reference_write_batching_transforms() {
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
        let update = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::new(TestIrValue::Array(Array::vector(vec![20f32, 30.]).unwrap()), BatchAxis::new(0)).unwrap(),
        );
        let operation = TestIrReferenceWriteOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        let outputs = context.bind(operation, Vec::new(), &[input, update, index]).unwrap();
        assert!(outputs.is_empty());
        assert_eq!(
            reference.read(),
            Ok(TestIrValue::Array(
                Array::from_elements::<f32>(packed_type, &[1f32, 2., 3., 4., 5., 6., 7., 8., 20., 10., 11., 30.])
                    .unwrap()
            )),
        );
    }

    #[test]
    fn test_reference_write_differentiation() {
        let context = DifferentiationContext::fused(TestIrContext::new());
        let reference = TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()).reference_new().unwrap();
        let tangent_reference = TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap()).reference_new().unwrap();
        let active = DifferentiationTracer::new(
            DifferentiationDual::new(reference.clone(), tangent_reference.clone()).unwrap(),
            context.clone(),
        );

        // A live replacement tangent is written into the tangent reference alongside the primal replacement.
        let replacement = DifferentiationTracer::new(
            DifferentiationDual::new(
                TestIrValue::Array(Array::vector(vec![5.0_f32, 6.0]).unwrap()),
                TestIrValue::Array(Array::vector(vec![7.0_f32, 8.0]).unwrap()),
            )
            .unwrap(),
            context.clone(),
        );
        let outputs = context.bind(ReferenceWriteOperation::new(), Vec::new(), &[active.clone(), replacement]).unwrap();
        assert!(outputs.is_empty());
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![5.0_f32, 6.0]).unwrap())));
        assert_eq!(tangent_reference.read(), Ok(TestIrValue::Array(Array::vector(vec![7.0_f32, 8.0]).unwrap())));

        // A symbolic zero replacement tangent is instantiated so that the tangent reference observes the store.
        let replacement = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(TestIrValue::Array(Array::vector(vec![9.0_f32, 10.0]).unwrap()))
                .unwrap(),
            context.clone(),
        );
        context.bind(ReferenceWriteOperation::new(), Vec::new(), &[active, replacement]).unwrap();
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![9.0_f32, 10.0]).unwrap())));
        assert_eq!(tangent_reference.read(), Ok(TestIrValue::Array(Array::vector(vec![0.0_f32, 0.0]).unwrap())));

        // A plumbing reference accepts a replacement without a live tangent and records no tangent store.
        let plumbing = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(reference.clone()).unwrap(),
            context.clone(),
        );
        let replacement = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(TestIrValue::Array(
                Array::vector(vec![11.0_f32, 12.0]).unwrap(),
            ))
            .unwrap(),
            context.clone(),
        );
        context.bind(ReferenceWriteOperation::new(), Vec::new(), &[plumbing.clone(), replacement]).unwrap();
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![11.0_f32, 12.0]).unwrap())));
        assert_eq!(tangent_reference.read(), Ok(TestIrValue::Array(Array::vector(vec![0.0_f32, 0.0]).unwrap())));

        // A live replacement tangent has no tangent reference to land in, and the rejection precedes the primal store.
        let replacement = DifferentiationTracer::new(
            DifferentiationDual::new(
                TestIrValue::Array(Array::vector(vec![13.0_f32, 14.0]).unwrap()),
                TestIrValue::Array(Array::vector(vec![15.0_f32, 16.0]).unwrap()),
            )
            .unwrap(),
            context.clone(),
        );
        let error = context.bind(ReferenceWriteOperation::new(), Vec::new(), &[plumbing, replacement]).unwrap_err();
        assert_eq!(
            error,
            ProgramError::InvalidArgument {
                message: "`reference_write` writes a live tangent into a reference that carries no tangent; pass the \
                          reference as a differentiated input instead of capturing it"
                    .to_string(),
            },
        );
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![11.0_f32, 12.0]).unwrap())));
    }

    #[test]
    fn test_reference_write_differentiation_transforms() {
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
        let update = DifferentiationTracer::new(
            DifferentiationDual::new(
                TestIrValue::Array(Array::scalar(9f32).unwrap()),
                TestIrValue::Array(Array::scalar(90f32).unwrap()),
            )
            .unwrap(),
            context.clone(),
        );
        let operation = TestIrReferenceWriteOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        let outputs = context.bind(operation, Vec::new(), &[input, update, index]).unwrap();
        assert!(outputs.is_empty());
        assert_eq!(
            reference.read(),
            Ok(TestIrValue::Array(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 9.]).unwrap())),
        );
        assert_eq!(
            tangent_reference.read(),
            Ok(TestIrValue::Array(Array::matrix(2, 3, vec![10f32, 20., 30., 40., 50., 90.]).unwrap())),
        );
    }

    #[test]
    fn test_reference_write_transposition() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));

        // The store transposes through the cotangent accumulator of its reference input, which only a transposition
        // context scoped to the instruction being transposed can resolve, so a detached context rejects it.
        let inputs = [PartialValue::Unknown(reference_type), PartialValue::Unknown(scalar_type.clone())];
        let mut context = TranspositionContext::new(TracingContext::<TestIrValue, TestIrOperation>::new());
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        assert!(matches!(
            TestIrReferenceWriteOperation::new().transpose(&mut context, &EmptyRegionDriver, &inputs, &[], &accumulators),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message)))
                if message == "input 0 has no reference root in a transposition context that is not scoped to a \
                               reference-carrying instruction",
        ));

        // `r = new(v); write(r, x); y = freeze(r)`: the freeze lands `ȳ` in the allocation's accumulator, the write
        // swaps a zero into it (the pre-execution state no longer flows anywhere) and hands its previous contents to
        // `x̄`, and the allocation freezes the cleared accumulator into `v̄ = 0`.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(scalar_type.clone());
        let replacement = builder.add_input(scalar_type.clone());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap();
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = zero [type=f32[]]
                    %2:ref<f32[]> = reference_new %1
                    () = reference_add_update %2 %0
                    %3:f32[] = zero [type=f32[]]
                    %4:f32[] = reference_swap %2 %3
                    %5:f32[] = reference_freeze %2
                in (%5, %4)
            "}
            .trim_end(),
        );
        assert_eq!(
            transposed.interpret(vec![TestIrValue::Array(Array::scalar(5.0_f32).unwrap())]),
            Ok(vec![
                TestIrValue::Array(Array::scalar(0.0_f32).unwrap()),
                TestIrValue::Array(Array::scalar(5.0_f32).unwrap()),
            ]),
        );

        // `r = new(v); write(r, x)` with no later access leaves the allocation's accumulator unallocated, so the write
        // stages nothing and both cotangents stay symbolic zeros.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(scalar_type.clone());
        let replacement = builder.add_input(scalar_type);
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder; 2], Vec::<Placeholder>::new())
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda  .
                let %0:f32[] = zero [type=f32[]]
                    %1:f32[] = zero [type=f32[]]
                in (%0, %1)
            "}
            .trim_end(),
        );
        assert_eq!(
            transposed.interpret(Vec::new()),
            Ok(vec![
                TestIrValue::Array(Array::scalar(0.0_f32).unwrap()),
                TestIrValue::Array(Array::scalar(0.0_f32).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_reference_write_transposition_transforms() {
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(ArrayType::new_static(DataType::F32, [2, 3]).into());
        let update = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let index = builder.add_constant(TestIrValue::Array(Array::scalar(-1i32).unwrap()));
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let operation = TestIrReferenceWriteOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        let outputs = builder
            .add_instruction(operation, Vec::new(), vec![reference, update, index], None)
            .unwrap()
            .to_vec();
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        assert!(outputs.is_empty());
        let output_ids = vec![frozen];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(output_ids, vec![Placeholder; 2], vec![Placeholder; 1])
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            transposed
                .interpret(vec![TestIrValue::Array(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap())]),
            Ok(vec![
                TestIrValue::Array(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 0.]).unwrap()),
                TestIrValue::Array(Array::scalar(6f32).unwrap())
            ])
        );
        // The adjoint swap addresses the same elements as the write, so its dynamic binding must be known when
        // transposing. Transposing with respect to the index leaves that binding unknown.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(ArrayType::new_static(DataType::F32, [2, 3]).into());
        let update = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let index = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let operation = TestIrReferenceWriteOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        let mut outputs = builder
            .add_instruction(operation, Vec::new(), vec![reference, update, index], None)
            .unwrap()
            .to_vec();
        outputs.push(
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0],
        );
        let output_count = outputs.len();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 3], vec![Placeholder; output_count])
            .unwrap();
        assert!(matches!(
            program.transpose_with_respect_to(&[0, 1, 2], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "reference transform bindings must be known when transposing an access",
        ));
    }

    #[test]
    fn test_reference_write_reference_discharge() {
        /// Reference policy that deliberately supports write discharge without supporting accumulation.
        #[derive(Copy, Clone, Debug)]
        struct WriteOnlyReferenceDischarge;

        impl<C: Context<Type = TestType, Operation: From<TestOperation>>> ReferenceDischargePolicy<C>
            for WriteOnlyReferenceDischarge
        {
            type Referent = TestReferent;
            type Transform = NoReferenceTransform<TestReferent, TestType>;
            type Alias = TestAlias;

            fn storage_alias(_referent: &TestReferent) -> TestAlias {
                TestAlias
            }

            fn read(_context: &C, current: &C::Value, _alias: &TestAlias) -> Result<C::Value, ProgramError> {
                Ok(current.clone())
            }

            fn write(
                _context: &C,
                _current: &C::Value,
                replacement: C::Value,
                _alias: &TestAlias,
            ) -> Result<C::Value, ProgramError> {
                Ok(replacement)
            }

            fn swap(
                _context: &C,
                _current: &C::Value,
                _replacement: C::Value,
                _alias: &TestAlias,
            ) -> Result<(C::Value, C::Value), ProgramError> {
                Err(ProgramError::MalformedProgram("write-only discharge policy must not swap".to_string()))
            }
        }

        // A policy with no accumulation capability replaces state through `write`, produces no old-value output,
        // and marks the allocation mutated. Its `swap` path is an error, making accidental swap dispatch visible.
        let context =
            ReferenceDischargeContext::<TestDestination, WriteOnlyReferenceDischarge>::new(TestDestination::new());
        let initial = TestValue::new(REFERENT, 4);
        let allocated =
            ReferenceDischargeValue::from(context.bind_discharged(ReferenceType::new(REFERENT), initial).unwrap());
        let reference = allocated.try_as_reference("the allocated reference").unwrap().clone();
        let inputs = vec![
            ReferenceDischargeValue::Reference(reference.clone()),
            ReferenceDischargeValue::Value(TestValue::new(REFERENT, 9)),
        ];
        assert_eq!(Write::new().discharge_references(&context, &EmptyRegionDriver, inputs.as_slice()), Ok(Vec::new()),);
        assert_eq!(context.read(&reference), Ok(TestValue::new(REFERENT, 9)));
        assert_eq!(context.is_mutated(reference.allocation_id()), Ok(true));

        // Exact input inference runs before mutation, so a rejected replacement leaves the allocation unchanged.
        let invalid = vec![
            ReferenceDischargeValue::Reference(reference.clone()),
            ReferenceDischargeValue::Value(TestValue::new(TestReferent::new(7, 32), 1)),
        ];
        assert_eq!(
            Write::new().discharge_references(&context, &EmptyRegionDriver, invalid.as_slice()),
            Err(TypeError::invalid(
                "`reference_write` replacement type `value<i7,p32>` must exactly match reference referent type \
             `value<i7,p16>`",
            )
            .into()),
        );
        assert_eq!(context.read(&reference), Ok(TestValue::new(REFERENT, 9)));
    }

    #[test]
    fn test_reference_write_reference_discharge_transforms() {
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(ArrayType::new_static(DataType::F32, [2, 3]).into());
        let update = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let index = builder.add_constant(TestIrValue::Array(Array::scalar(-1i32).unwrap()));
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let operation = TestIrReferenceWriteOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        let outputs = builder
            .add_instruction(operation, Vec::new(), vec![reference, update, index], None)
            .unwrap()
            .to_vec();
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        assert!(outputs.is_empty());
        let output_ids = vec![frozen];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(output_ids, vec![Placeholder; 2], vec![Placeholder; 1])
            .unwrap();
        let discharged = program.discharge_references(0).unwrap();
        assert!(!discharged.program().entry_region_ref().contains_reference_accesses_in_closure());
        assert_eq!(
            discharged.program().interpret(vec![
                TestIrValue::Array(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap()),
                TestIrValue::Array(Array::scalar(9f32).unwrap())
            ]),
            Ok(vec![TestIrValue::Array(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 9.]).unwrap())])
        );
    }

    #[test]
    fn test_reference_write_projected() {
        type TestTracingContext = TracingContext<TestIrValue, TestIrOperation>;
        type TestTracer = Tracer<TestTracingContext>;

        // A projected reference binds the write through its parent tracer's context, so the projected surface stages
        // exactly the native operation.
        let array_type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2]));
        let (_, program) = TestTracingContext::trace(
            |inputs| {
                let [initial, replacement]: [TestTracer; 2] = inputs.try_into().unwrap();
                let initial = <TestTracer as ValueProjection<ArrayType>>::into_projected(initial)?;
                let replacement = <TestTracer as ValueProjection<ArrayType>>::into_projected(replacement)?;
                let reference = initial.reference_new()?;
                let _: &ProjectedValue<ReferenceType<ArrayType>, TestTracer> = &reference;
                reference.write(&replacement)?;
                let frozen = reference.freeze()?;
                let _: &ProjectedValue<ArrayType, TestTracer> = &frozen;
                Ok(vec![frozen.into_value()])
            },
            vec![array_type.clone(), array_type],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[2] .
                let %2:ref<f32[2]> = reference_new %0
                    () = reference_write %2 %1
                    %3:f32[2] = reference_freeze %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_reference_write_staging() {
        // A traced reference stages the write as the native variant of its operation family, with no output and the
        // ordered-state effect of the program.
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let (output_types, program) = TracingContext::<TestIrValue, TestIrOperation>::trace(
            |inputs| {
                let [reference, replacement]: [Tracer<_>; 2] = inputs.try_into().unwrap();
                reference.write(&replacement)?;
                Ok(vec![reference])
            },
            vec![
                ArrayIrType::Reference(ReferenceType::new(array_type.clone())),
                ArrayIrType::Array(array_type.clone()),
            ],
        )
        .unwrap();
        assert_eq!(output_types, vec![ArrayIrType::Reference(ReferenceType::new(array_type))]);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ref<f32[2]>, %1:f32[2] .
                let () = reference_write %0 %1
                in (%0)
            "}
            .trim_end(),
        );
        assert_eq!(program.effects().classes(), EffectClasses::single(EffectClass::OrderedState));
    }
}
