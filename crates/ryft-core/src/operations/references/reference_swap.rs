use std::borrow::Cow;
use std::fmt::Display;
use std::marker::PhantomData;
use std::sync::LazyLock;

use crate::arrays::{ArrayIrType, ArrayIrValue, ArrayReferenceTransform, ArrayType};
use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{DifferentiableType, DifferentiationDual, ResidualZeroProvider};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::constants::zero::Zero;
use crate::operations::manipulation::reshaping::Reshape;
use crate::operations::manipulation::slicing::{Slice, UpdateSlice};
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

/// Canonical operation name for [`ReferenceSwapOperation`].
pub const REFERENCE_SWAP_OPERATION_NAME: &str = "reference_swap";

/// Replaces the selected reference state with an exactly matching value and returns its previous contents. The root
/// and replacement are the first two inputs. Dynamic bindings follow them in transform order and the path is stored
/// in `Transform` metadata. An empty path accesses the complete referent.
#[derive(Clone, Debug)]
pub struct ReferenceSwapOperation<
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

impl<T: Type, U: Type, Transform: ReferenceTransform<Type = U, Referent = T>> ReferenceSwapOperation<T, U, Transform> {
    /// Creates a new [`ReferenceSwapOperation`].
    pub const fn new() -> Self {
        Self { transforms: Vec::new(), marker: PhantomData }
    }

    /// Returns a copy of this [`ReferenceSwapOperation`] with the provided transforms applied to the reference input.
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

    /// Returns the number of dynamic inputs supplied after the base inputs for this [`ReferenceSwapOperation`].
    fn binding_count(&self) -> usize {
        self.transforms.iter().map(ReferenceTransform::binding_count).sum()
    }
}

impl<T: Type, U: Type, Transform: ReferenceTransform<Type = U, Referent = T>> Default
    for ReferenceSwapOperation<T, U, Transform>
{
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Type, U: Type, Transform: ReferenceTransform<Type = U, Referent = T>> Display
    for ReferenceSwapOperation<T, U, Transform>
where
    Self: Operation,
{
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type, U: Type + From<T>, Transform: ReferenceTransform<Type = U, Referent = T>> Operation
    for ReferenceSwapOperation<T, U, Transform>
where
    for<'t> &'t T: TryFrom<&'t U, Error = TypeError>,
    for<'t> &'t ReferenceType<T>: TryFrom<&'t U, Error = TypeError>,
{
    type Type = U;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_SWAP_OPERATION_NAME
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
            ReferenceAccessMode::ReadWrite,
        )?;
        let replacement = <&T>::try_from(&input_types[1])?;
        if replacement != &referent {
            return Err(TypeError::invalid(format!(
                "`{REFERENCE_SWAP_OPERATION_NAME}` replacement type `{replacement}` must exactly match reference \
                 referent type `{referent}`",
            )));
        }
        Ok(vec![referent.into()])
    }

    #[inline]
    fn effects(&self) -> Cow<'_, Effects> {
        // Share one descriptor across all type instantiations to avoid allocating and validating it on every query.
        static EFFECTS: LazyLock<Effects> = LazyLock::new(|| {
            Effects::new(
                EffectClasses::NONE,
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::ReadWrite }],
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
    for ReferenceSwapOperation<T, U, Transform>
where
    ReferenceSwapOperation<T, U, Transform>: Operation<Type = U>,
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
> ReferenceDischargeableOperation<C, P> for ReferenceSwapOperation<T, U, Transform>
where
    ReferenceSwapOperation<T, U, Transform>: Operation<Type = U>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        _driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        check_count!("input", inputs, 2 + self.binding_count(), ProgramError);
        let reference = inputs[0].try_as_reference("a reference to replace")?;
        let bindings = inputs[2..]
            .iter()
            .map(|input| input.try_as_value("a reference transform binding").cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let replacement = inputs[1].try_as_value("a replacement value")?.clone();

        // Rules can be called outside a validated program, so check the input types before changing reference state.
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(&input_types, &[])?;
        Ok(vec![ReferenceDischargeValue::Value(context.swap_through(
            reference,
            replacement,
            &self.transforms,
            &bindings,
        )?)])
    }
}

impl<
    T: Type,
    U: Type,
    Transform: ReferenceTransform<Type = U, Referent = T>,
    C: Domain<Type = U, Value: ReferenceSwap<Transform, C::Value, C::Value, C::Value>>,
> InterpretableOperation<C> for ReferenceSwapOperation<T, U, Transform>
where
    ReferenceSwapOperation<T, U, Transform>: Operation<Type = U>,
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2 + self.binding_count(), ProgramError);
        Ok(vec![inputs[0].swap_through(&inputs[1], &self.transforms, &inputs[2..])?])
    }
}

impl<
    T: Type,
    U: Type,
    Transform: ReferenceTransform<Type = U, Referent = T>,
    C: Context<Type = U, Operation: From<ReferenceSwapOperation<T, U, Transform>>>,
> PartiallyEvaluatableOperation<C> for ReferenceSwapOperation<T, U, Transform>
{
}

impl<
    T: Type,
    U: Type + From<ReferenceType<T>>,
    Transform: BatchableReferenceTransform<Type = U, Referent = T>,
    C: Context<Type = U, Operation: From<ReferenceSwapOperation<T, U, Transform>>>,
    P: BatchingPolicy<C>,
> BatchableOperation<C, P> for ReferenceSwapOperation<T, U, Transform>
where
    for<'t> &'t ReferenceType<T>: TryFrom<&'t U, Error = TypeError>,
    ReferenceSwapOperation<T, U, Transform>: Operation<Type = U>,
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
                        "`{REFERENCE_SWAP_OPERATION_NAME}` cannot store a batched value into an unbatched reference; \
                         pass the reference as a batched input instead",
                    ),
                });
            }
        };

        let mut values = vec![P::value(&inputs[0]).clone(), P::value(&replacement).clone()];
        values.extend(inputs[2..].iter().map(|input| P::value(input).clone()));
        let previous = context.parent().bind(self.clone().with_transforms(transforms), Vec::new(), &values)?.remove(0);
        Ok(vec![P::batch(previous, output_axis)?].into())
    }
}

impl_differentiable_operation! {
    <T, U, Transform> ReferenceSwapOperation<T, U, Transform>,
    jvp<C>
    where
        T: Type,
        U: DifferentiableType,
        Transform: ReferenceTransform<Type = U, Referent = T>,
        C: Context<
            Type = U,
            Operation: From<ReferenceSwapOperation<T, U, Transform>>
                + ResidualZeroProvider<U, Operation = C::Operation>,
        > + Zero<C::Value>,
    {
        |operation, context, _driver, inputs| {
            // Swap the tangent state alongside the primal and pair their previous values and reject a live tangent
            // without tangent storage before either reference can be changed.
            check_count!("input", inputs, 2 + operation.binding_count(), ProgramError);
            let stored = match (inputs[0].tangent(), inputs[1].tangent()) {
                (MaybeZero::Value(reference), tangent) => Some((reference, tangent.clone())),
                (MaybeZero::Zero(_), MaybeZero::Zero(_)) => None,
                (MaybeZero::Zero(_), MaybeZero::Value(_)) => {
                    return Err(ProgramError::InvalidArgument {
                        message: format!(
                            "`{REFERENCE_SWAP_OPERATION_NAME}` writes a live tangent into a reference that carries no \
                             tangent; pass the reference as a differentiated input instead of capturing it",
                        ),
                    }
                    .into());
                }
            };
            let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            let previous = context.primal().bind(operation.clone(), Vec::new(), &primal_inputs)?.remove(0);
            Ok(vec![match stored {
                Some((tangent_reference, tangent)) => {
                    // A zero replacement tangent is instantiated because the tangent reference must observe the store.
                    let source = context.primal_to_tangent(inputs[1].primal().clone())?;
                    let tangent = C::Operation::materialize_zero_from_residual_sources(
                        context.tangent(),
                        tangent,
                        std::iter::once(&source),
                    )?;
                    let mut tangent_inputs = vec![tangent_reference.clone(), tangent];
                    for input in &inputs[2..] {
                        tangent_inputs.push(context.primal_to_tangent(input.primal().clone())?);
                    }
                    DifferentiationDual::new(
                        previous,
                        MaybeZero::Value(
                            context
                                .tangent()
                                .bind(operation.clone(), Vec::new(), &tangent_inputs)?
                                .remove(0),
                        ),
                    )?
                }
                None => DifferentiationDual::new_with_zero_tangent(previous)?,
            }])
        }
    },
    transpose<V, O>
    where
        T: Type,
        U: DifferentiableType + ReferenceMemberType,
        Transform: ReferenceTransform<Type = U, Referent = T>,
        V: Value<Type = U>,
        O: Operation<Type = U>
            + ResidualZeroProvider<U, Operation = O>
            + OperationProvider<U, ReferenceNewOperation<<U as ReferenceMemberType>::Referent, U>, Operation = O>
            + From<ReferenceSwapOperation<T, U, Transform>>,
    {
        |operation, context, driver, inputs, outputs, accumulators| {
            // A swap maps `(state, x) ↦ (x, state)`, so its transpose swaps the output cotangent into the cotangent
            // reference and yields the previous contents as the cotangent of the stored value. A zero output cotangent
            // swapped into an accumulator that nothing has reached yet leaves both zero, so nothing is staged and the
            // stored value's cotangent stays symbolic.
            check_count!("input", inputs, 2 + operation.binding_count(), ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
            let accumulator = match &outputs[0] {
                MaybeZero::Value(_) => context.cotangent_reference(driver, 0)?,
                MaybeZero::Zero(_) => match context.cotangent_reference_if_allocated(driver, 0)? {
                    Some(accumulator) => accumulator,
                    None => return Ok(()),
                },
            };
            let cotangent = O::materialize_zero_from_residual_sources(
                &**context,
                outputs[0].clone(),
                context
                    .dimension_sources()
                    .chain(inputs.iter().filter_map(PartialValue::as_known))
                    .chain(std::iter::once(&accumulator)),
            )?;
            let mut adjoint_inputs = vec![accumulator, cotangent];
            for input in &inputs[2..] {
                adjoint_inputs.push(input.as_known().cloned().ok_or_else(|| ProgramError::UnsupportedOperation {
                    message: "reference transform bindings must be known when transposing an access".to_string(),
                })?);
            }
            let previous = context.bind(operation.clone(), Vec::new(), &adjoint_inputs)?.remove(0);
            accumulators[1].accumulate(context, MaybeZero::Value(previous))
        }
    },
}

/// Capability to replace the value stored by a reference in program order and return its previous immutable snapshot.
pub trait ReferenceSwap<Transform: ReferenceTransform, Binding = Self, Replacement = Self, Output = Replacement>:
    Sized
{
    /// Replaces the selected state and returns its previous immutable snapshot.
    ///
    /// # Parameters
    ///
    ///   - `replacement`: Value stored into the viewed reference.
    ///   - `transforms`: Transforms applied in order to the reference.
    ///   - `bindings`: Dynamic inputs in transform order.
    fn swap_through(
        &self,
        replacement: &Replacement,
        transforms: &[Transform],
        bindings: &[Binding],
    ) -> Result<Output, ProgramError>;

    /// Replaces the complete reference state and returns its previous immutable snapshot.
    #[inline]
    fn swap(&self, replacement: &Replacement) -> Result<Output, ProgramError> {
        self.swap_through(replacement, &[], &[])
    }
}

impl<A: Value<Type = ArrayType> + Concretizable<i128> + Reshape + Slice + UpdateSlice>
    ReferenceSwap<ArrayReferenceTransform> for ArrayIrValue<A>
{
    fn swap_through(
        &self,
        replacement: &Self,
        transforms: &[ArrayReferenceTransform],
        bindings: &[Self],
    ) -> Result<Self, ProgramError> {
        let operation = ReferenceSwapOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
        let operation = operation.with_transforms(transforms.to_vec());
        let mut input_types = vec![self.r#type().into_owned(), replacement.r#type().into_owned()];
        input_types.extend(bindings.iter().map(|value| value.r#type().into_owned()));
        operation.infer_output_types(&input_types, &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        let reference = reference.with_transforms(transforms, bindings)?;
        let replacement = <Self as ValueProjection<ArrayType>>::projected(replacement)?.clone();
        Ok(Self::Array(reference.swap(replacement)?))
    }
}

impl<Transform, V> ReferenceSwap<Transform, V, V, V> for V
where
    Transform: ReferenceTransform<Type = ArrayIrType, Referent = ArrayType>,
    V: Value<
            Type = ArrayIrType,
            DispatchDomain: Context<Operation: From<ReferenceSwapOperation<ArrayType, ArrayIrType, Transform>>>,
        >,
{
    fn swap_through(
        &self,
        replacement: &Self,
        transforms: &[Transform],
        bindings: &[Self],
    ) -> Result<Self, ProgramError> {
        let mut inputs = vec![self.clone(), replacement.clone()];
        inputs.extend_from_slice(bindings);
        let operation = ReferenceSwapOperation::new().with_transforms(transforms.to_vec());
        Ok(self.dispatch_domain().bind(operation, Vec::new(), &inputs)?.remove(0))
    }
}

impl<Transform, V>
    ReferenceSwap<Transform, V, ProjectedValue<ArrayType, V>, <V as ValueProjection<ArrayType>>::Projected>
    for ProjectedValue<ReferenceType<ArrayType>, V>
where
    Transform: ReferenceTransform<Type = ArrayIrType, Referent = ArrayType>,
    V: Value<
            Type = ArrayIrType,
            DispatchDomain: Context<Operation: From<ReferenceSwapOperation<ArrayType, ArrayIrType, Transform>>>,
        > + ValueProjection<ArrayType>,
{
    fn swap_through(
        &self,
        replacement: &ProjectedValue<ArrayType, V>,
        transforms: &[Transform],
        bindings: &[V],
    ) -> Result<<V as ValueProjection<ArrayType>>::Projected, ProgramError> {
        let mut inputs = vec![self.value().clone(), replacement.value().clone()];
        inputs.extend_from_slice(bindings);
        let operation = ReferenceSwapOperation::new().with_transforms(transforms.to_vec());
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
        Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrValue, ArrayReference,
        ArrayReferenceTransformIndex, ArrayType, DataType, Dimension, DimensionBounds, DimensionType, DimensionValue,
        DimensionVariable, Shape,
    };
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer};
    use crate::contexts::{EagerContext, StagingContext};
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
    type TestIrReferenceSwapOperation = ReferenceSwapOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>;

    #[test]
    fn test_reference_swap() {
        let operation = Swap::new();
        assert_eq!(Swap::default().to_string(), operation.to_string());
        assert_eq!(operation.name(), REFERENCE_SWAP_OPERATION_NAME);
        assert_eq!(operation.to_string(), REFERENCE_SWAP_OPERATION_NAME);
        assert_eq!(
            format!("{operation:?}"),
            format!(
                "ReferenceSwapOperation {{ transforms: [], marker: {:?} }}",
                PhantomData::<fn() -> (TestReferent, TestType)>
            ),
        );

        // A swap orders against other state effects, reads and writes its reference input, and aliases nothing.
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedState));
        assert_eq!(
            operation.effects().reference_effects(),
            &[ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::ReadWrite }]
        );
    }

    #[test]
    fn test_reference_swap_type_inference() {
        let referent = TestReferent::new(7, 16);
        let value = TestType::Value(referent);
        let reference = TestType::Reference(ReferenceType::new(referent));
        check_operation_type_inference!(
            operation = Swap::new(),
            cases = [
                {
                    input_types = [reference.clone(), value.clone()],
                    output_types = [value.clone()],
                },
                {
                    input_types = [reference.clone(), TestType::Value(TestReferent::new(7, 32))],
                    error = "`reference_swap` replacement type `value<i7,p32>` must exactly match reference referent \
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
            Swap::new().infer_output_types(&[reference, value], std::slice::from_ref(&region)),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );

        // A universe must project references and values out of itself and embed the previous value back into itself.
        check_operation_type_inference!(
            operation = ReferenceSwapOperation::<TestReferent, SwapUniverse>::new(),
            cases = [{
                input_types = [SwapUniverse::Reference(ReferenceType::new(referent)), SwapUniverse::Value(referent)],
                output_types = [SwapUniverse::Value(referent)],
            }],
        );

        let vector_type = ArrayType::new_static(DataType::F32, [2]);
        check_operation_type_inference!(
            operation = TestIrReferenceSwapOperation::new(),
            cases = [
                {
                    input_types = [
                        ArrayIrType::Reference(ReferenceType::new(vector_type.clone())),
                        ArrayIrType::Array(vector_type.clone()),
                    ],
                    output_types = [ArrayIrType::Array(vector_type.clone())],
                },
                {
                    input_types = [
                        ArrayIrType::Reference(ReferenceType::new(vector_type.clone())),
                        ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])),
                    ],
                    error = "`reference_swap` replacement type `f32[3]` must exactly match reference referent type \
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
    fn test_reference_swap_type_inference_transforms() {
        let operation = TestIrReferenceSwapOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        let reference = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2, 3])));
        let scalar = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let index = ArrayIrType::Array(ArrayType::scalar(DataType::I32));
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                { input_types = [reference.clone(), scalar.clone(), index.clone()], output_types = [scalar.clone()], },
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
            "reference_swap [transforms=[index(axis=0, index=1), index(axis=0, index=dynamic)]]",
        );
        let descriptor = operation.reference_access_descriptor(0).unwrap();
        assert_eq!(descriptor.transforms(), operation.transforms());
        assert_eq!(descriptor.bindings(), 2..3);
        assert!(operation.reference_access_descriptor(1).is_none());
    }

    #[test]
    fn test_reference_swap_interpretation() {
        let live = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]).unwrap());
        let reference = TestIrValue::Reference(live.clone());

        // A replacement carrying exactly the referent type replaces the stored value and returns the previous value.
        assert_eq!(
            InterpretableOperation::<TestIrContext>::interpret(
                &TestIrReferenceSwapOperation::new(),
                &TestIrContext::new(),
                &EmptyRegionDriver,
                &[reference.clone(), TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap())],
            ),
            Ok(vec![TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap())]),
        );
        assert_eq!(live.read(), Ok(Array::vector(vec![3.0_f32, 4.0]).unwrap()));

        // Exact input inference runs before the swap, so a rejected replacement leaves the stored value unchanged.
        assert_eq!(
            InterpretableOperation::<TestIrContext>::interpret(
                &TestIrReferenceSwapOperation::new(),
                &TestIrContext::new(),
                &EmptyRegionDriver,
                &[reference.clone(), TestIrValue::Array(Array::vector(vec![5.0_f32, 6.0, 7.0]).unwrap())],
            ),
            Err(TypeError::invalid(
                "`reference_swap` replacement type `f32[3]` must exactly match reference referent type `f32[2]`",
            )
            .into()),
        );
        assert_eq!(live.read(), Ok(Array::vector(vec![3.0_f32, 4.0]).unwrap()));

        // Each input must be the member kind the operation expects.
        let array = TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap());
        assert_eq!(
            InterpretableOperation::<TestIrContext>::interpret(
                &TestIrReferenceSwapOperation::new(),
                &TestIrContext::new(),
                &EmptyRegionDriver,
                &[array.clone(), array],
            ),
            Err(TypeError::invalid("expected reference type but got array type").into()),
        );
        assert_eq!(
            InterpretableOperation::<TestIrContext>::interpret(
                &TestIrReferenceSwapOperation::new(),
                &TestIrContext::new(),
                &EmptyRegionDriver,
                &[reference.clone(), reference],
            ),
            Err(TypeError::invalid("expected array type but got reference type").into()),
        );
        assert_eq!(live.read(), Ok(Array::vector(vec![3.0_f32, 4.0]).unwrap()));

        // Equality over a dynamically typed `Array` cannot address its elements, so a dynamically typed referent is
        // compared by its declared type and its exact physical storage. Both referents come from the test-only
        // unchecked hatch because the checked constructors reject dynamically shaped types, and they declare exactly
        // the same dynamic type, so the swap is observable only through the payloads it returns and installs.
        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("length", DimensionBounds::unbounded()))]),
        );
        let initial_bytes = 1.0_f32.to_le_bytes().to_vec();
        let replacement_bytes = 2.0_f32.to_le_bytes().to_vec();
        let live = ArrayReference::new(Array::with_unchecked_type(dynamic_type.clone(), initial_bytes.clone()));
        let replacement = Array::with_unchecked_type(dynamic_type.clone(), replacement_bytes.clone());
        let mut outputs = InterpretableOperation::<TestIrContext>::interpret(
            &TestIrReferenceSwapOperation::new(),
            &TestIrContext::new(),
            &EmptyRegionDriver,
            &[TestIrValue::Reference(live.clone()), TestIrValue::Array(replacement)],
        )
        .unwrap();
        assert_eq!(outputs.len(), 1);
        let previous = <TestIrValue as ValueProjection<ArrayType>>::into_projected(outputs.remove(0)).unwrap();
        assert_eq!(previous.r#type().into_owned(), dynamic_type);
        assert_eq!(previous.storage_bytes(), initial_bytes.as_slice());
        let installed = live.read().unwrap();
        assert_eq!(installed.r#type().into_owned(), dynamic_type);
        assert_eq!(installed.storage_bytes(), replacement_bytes.as_slice());
    }

    #[test]
    fn test_reference_swap_interpretation_transforms() {
        let context = EagerContext::<TestIrValue, TestIrOperation>::new();
        let reference =
            TestIrValue::Reference(ArrayReference::new(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap()));
        let index = TestIrValue::Array(Array::scalar(-1i32).unwrap());
        let operation = TestIrReferenceSwapOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        assert_eq!(
            operation.interpret(
                &context,
                &EmptyRegionDriver,
                &[reference.clone(), TestIrValue::Array(Array::scalar(9f32).unwrap()), index]
            ),
            Ok(vec![TestIrValue::Array(Array::scalar(6f32).unwrap())]),
        );
        assert_eq!(
            reference.read(),
            Ok(TestIrValue::Array(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 9.]).unwrap())),
        );
    }

    #[test]
    fn test_reference_swap_partial_evaluation() {
        // Program replay uses the `Stage` placement: a swap stages regardless of input knowledge, its previous value
        // is an unknown of the residual program, the live handle is passed to that program as a known reference input,
        // and the swap runs only when that program runs.
        let live = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        check_operation_partial_evaluation!(
            backend = (TestIrValue, TestIrOperation),
            operation = TestIrReferenceSwapOperation::new(),
            cases = [
                {
                    inputs = [
                        (@known, TestIrValue::Reference(live.clone())),
                        (@known, TestIrValue::Array(Array::scalar(2.0_f32).unwrap())),
                    ],
                    outputs = [(@residual, TestIrValue::Array(Array::scalar(1.0_f32).unwrap()))],
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
                    outputs = [(@residual, TestIrValue::Array(Array::scalar(2.0_f32).unwrap()))],
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
        let swapped = staging
            .fold_or_residualize(
                TestIrReferenceSwapOperation::new(),
                Vec::new(),
                &[reference.clone(), replacement.clone()],
            )
            .unwrap();
        assert_eq!(swapped.len(), 1);
        assert!(swapped[0].is_unknown());
        assert_eq!(live.read(), Ok(Array::scalar(3.0_f32).unwrap()));

        // Under the default `Execute` placement an all-known swap folds: it runs against the live state in program
        // order at partial evaluation time and its previous value is known.
        let executing = PartialEvaluationContext::new(TestIrContext::new());
        let swapped = executing
            .fold_or_residualize(TestIrReferenceSwapOperation::new(), Vec::new(), &[reference, replacement])
            .unwrap();
        assert_eq!(swapped.len(), 1);
        assert_eq!(swapped[0].as_known(), Some(&TestIrValue::Array(Array::scalar(3.0_f32).unwrap())));
        assert_eq!(live.read(), Ok(Array::scalar(4.0_f32).unwrap()));
    }

    #[test]
    fn test_reference_swap_partial_evaluation_transforms() {
        // The known root enters the residual program while the dynamic index remains an ordinary unknown input.
        let live = ArrayReference::new(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap());
        check_operation_partial_evaluation!(
            backend = (TestIrValue, TestIrOperation),
            operation = TestIrReferenceSwapOperation::new().with_transforms(vec![
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
                    outputs = [(@residual, TestIrValue::Array(Array::scalar(6f32).unwrap()))],
                    residual_instructions = 1,
                },
            ],
        );
        assert_eq!(live.read(), Ok(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 2.]).unwrap()));
    }

    #[test]
    fn test_reference_swap_batching() {
        let extent = TestIrValue::Dimension(
            DimensionValue::new(DimensionType::new("batch", DimensionBounds::unbounded()), 2).unwrap(),
        );
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(TestIrContext::new(), extent);
        let packed_type = ArrayType::new_static(DataType::F32, [3, 2]);
        let initial = TestIrValue::Array(
            Array::from_elements::<f32>(packed_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        );
        let reference = initial.reference_new().unwrap();
        let batched =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(reference.clone(), BatchAxis::new(1)).unwrap());

        // A replacement mapped at the reference's batch axis is swapped packed, and the previous packed value is
        // batched at the reference's axis.
        let aligned = TestIrValue::Array(
            Array::from_elements::<f32>(packed_type.clone(), &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap(),
        );
        let replacement =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(aligned.clone(), BatchAxis::new(1)).unwrap());
        let outputs = context.bind(ReferenceSwapOperation::new(), Vec::new(), &[batched, replacement]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(1));
        assert_eq!(outputs[0].batch().value(), &initial);
        assert_eq!(reference.read(), Ok(aligned));

        // A batched replacement cannot be swapped into an unbatched reference.
        let replicated = BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(reference.clone()));
        let replacement = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::new(
                TestIrValue::Array(Array::from_elements::<f32>(packed_type, &[0.0; 6]).unwrap()),
                BatchAxis::new(1),
            )
            .unwrap(),
        );
        let error = context.bind(ReferenceSwapOperation::new(), Vec::new(), &[replicated, replacement]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<BatchingError>(),
            Some(&BatchingError::UnsupportedOperation {
                message: "`reference_swap` cannot store a batched value into an unbatched reference; pass the \
                          reference as a batched input instead"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_reference_swap_batching_transforms() {
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
        let operation = TestIrReferenceSwapOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        let outputs = context.bind(operation, Vec::new(), &[input, update, index]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].batch().value(), &TestIrValue::Array(Array::vector(vec![9f32, 12.]).unwrap()));
        assert_eq!(
            reference.read(),
            Ok(TestIrValue::Array(
                Array::from_elements::<f32>(packed_type, &[1f32, 2., 3., 4., 5., 6., 7., 8., 20., 10., 11., 30.])
                    .unwrap()
            )),
        );
    }

    #[test]
    fn test_reference_swap_differentiation() {
        let context = DifferentiationContext::fused(TestIrContext::new());
        let reference = TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()).reference_new().unwrap();
        let tangent_reference = TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap()).reference_new().unwrap();
        let active = DifferentiationTracer::new(
            DifferentiationDual::new(reference.clone(), tangent_reference.clone()).unwrap(),
            context.clone(),
        );

        // Swapping an active reference swaps its tangent reference alongside, so the previous value pairs with the
        // previous tangent contents.
        let replacement = DifferentiationTracer::new(
            DifferentiationDual::new(
                TestIrValue::Array(Array::vector(vec![5.0_f32, 6.0]).unwrap()),
                TestIrValue::Array(Array::vector(vec![7.0_f32, 8.0]).unwrap()),
            )
            .unwrap(),
            context.clone(),
        );
        let outputs = context.bind(ReferenceSwapOperation::new(), Vec::new(), &[active, replacement]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()));
        assert_eq!(
            outputs[0].tangent().as_value(),
            Some(&TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap()))
        );
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![5.0_f32, 6.0]).unwrap())));
        assert_eq!(tangent_reference.read(), Ok(TestIrValue::Array(Array::vector(vec![7.0_f32, 8.0]).unwrap())));

        // Swapping a plumbing reference with a replacement without a live tangent yields a symbolic zero tangent.
        let plumbing = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(reference.clone()).unwrap(),
            context.clone(),
        );
        let replacement = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(TestIrValue::Array(Array::vector(vec![9.0_f32, 10.0]).unwrap()))
                .unwrap(),
            context.clone(),
        );
        let outputs =
            context.bind(ReferenceSwapOperation::new(), Vec::new(), &[plumbing.clone(), replacement]).unwrap();
        assert_eq!(outputs[0].primal(), &TestIrValue::Array(Array::vector(vec![5.0_f32, 6.0]).unwrap()));
        assert!(matches!(
            outputs[0].tangent(),
            MaybeZero::Zero(r#type) if *r#type == ArrayType::new_static(DataType::F32, [2]).into(),
        ));
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![9.0_f32, 10.0]).unwrap())));
        assert_eq!(tangent_reference.read(), Ok(TestIrValue::Array(Array::vector(vec![7.0_f32, 8.0]).unwrap())));

        // A live replacement tangent has no tangent reference to land in, and the rejection precedes the primal swap.
        let replacement = DifferentiationTracer::new(
            DifferentiationDual::new(
                TestIrValue::Array(Array::vector(vec![11.0_f32, 12.0]).unwrap()),
                TestIrValue::Array(Array::vector(vec![13.0_f32, 14.0]).unwrap()),
            )
            .unwrap(),
            context.clone(),
        );
        let error = context.bind(ReferenceSwapOperation::new(), Vec::new(), &[plumbing, replacement]).unwrap_err();
        assert_eq!(
            error,
            ProgramError::InvalidArgument {
                message: "`reference_swap` writes a live tangent into a reference that carries no tangent; pass the \
                          reference as a differentiated input instead of capturing it"
                    .to_string(),
            },
        );
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![9.0_f32, 10.0]).unwrap())));
    }

    #[test]
    fn test_reference_swap_differentiation_transforms() {
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
        let operation = TestIrReferenceSwapOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        let outputs = context.bind(operation, Vec::new(), &[input, update, index]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &TestIrValue::Array(Array::scalar(6f32).unwrap()));
        assert_eq!(outputs[0].tangent().as_value(), Some(&TestIrValue::Array(Array::scalar(60f32).unwrap())));
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
    fn test_reference_swap_transposition() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));

        // The swap transposes through the cotangent accumulator of its reference input, which only a transposition
        // context scoped to the instruction being transposed can resolve, so a detached context rejects it.
        let inputs = [PartialValue::Unknown(reference_type), PartialValue::Unknown(scalar_type.clone())];
        let tracing = TracingContext::<TestIrValue, TestIrOperation>::new();
        let cotangent = tracing.input(scalar_type.clone());
        let mut context = TranspositionContext::new(tracing);
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        assert!(matches!(
            TestIrReferenceSwapOperation::new().transpose(
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

        // `r = new(v); y = swap(r, x); z = freeze(r)`: the freeze lands `z̄` in the allocation's accumulator, the swap
        // swaps `ȳ` into it and hands its previous contents to `x̄ = z̄`, and the allocation freezes the accumulator
        // into `v̄ = ȳ`.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(scalar_type.clone());
        let replacement = builder.add_input(scalar_type.clone());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let previous = builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap()[0];
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![previous, frozen],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[] = zero [type=f32[]]
                    %3:ref<f32[]> = reference_new %2
                    () = reference_add_update %3 %1
                    %4:f32[] = reference_swap %3 %0
                    %5:f32[] = reference_freeze %3
                in (%5, %4)
            "}
            .trim_end(),
        );
        assert_eq!(
            transposed.interpret(vec![
                TestIrValue::Array(Array::scalar(2.0_f32).unwrap()),
                TestIrValue::Array(Array::scalar(5.0_f32).unwrap()),
            ]),
            Ok(vec![
                TestIrValue::Array(Array::scalar(2.0_f32).unwrap()),
                TestIrValue::Array(Array::scalar(5.0_f32).unwrap()),
            ]),
        );

        // A dead swap output whose accumulator a later freeze allocated instantiates its zero cotangent, because the
        // accumulator must observe the swap: `x̄ = z̄` and `v̄ = 0`.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(scalar_type.clone());
        let replacement = builder.add_input(scalar_type.clone());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap();
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![frozen], vec![Placeholder; 2], vec![Placeholder])
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

        // A dead swap output with an accumulator that nothing reached stages nothing and leaves both cotangents
        // symbolic zeros.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(scalar_type.clone());
        let replacement = builder.add_input(scalar_type);
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
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
    fn test_reference_swap_transposition_transforms() {
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(ArrayType::new_static(DataType::F32, [2, 3]).into());
        let update = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let index = builder.add_constant(TestIrValue::Array(Array::scalar(-1i32).unwrap()));
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let operation = TestIrReferenceSwapOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        let outputs = builder
            .add_instruction(operation, Vec::new(), vec![reference, update, index], None)
            .unwrap()
            .to_vec();
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let output_ids = vec![outputs[0], frozen];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(output_ids, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            transposed.interpret(vec![
                TestIrValue::Array(Array::scalar(7f32).unwrap()),
                TestIrValue::Array(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap())
            ]),
            Ok(vec![
                TestIrValue::Array(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 7.]).unwrap()),
                TestIrValue::Array(Array::scalar(6f32).unwrap())
            ])
        );
        // The adjoint swap addresses the same elements as the swap, so its dynamic binding must be known when
        // transposing. Transposing with respect to the index leaves that binding unknown.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(ArrayType::new_static(DataType::F32, [2, 3]).into());
        let update = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let index = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let operation = TestIrReferenceSwapOperation::new().with_transforms(vec![
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
    fn test_reference_swap_reference_discharge() {
        // A replacement returns the previous state and commits the successor, which marks the allocation mutated.
        let (context, reference) = allocated_reference(4);
        let inputs = vec![
            ReferenceDischargeValue::Reference(reference.clone()),
            ReferenceDischargeValue::Value(TestValue::new(REFERENT, 9)),
        ];
        assert_eq!(
            Swap::new().discharge_references(&context, &EmptyRegionDriver, inputs.as_slice()),
            Ok(vec![ReferenceDischargeValue::Value(TestValue::new(REFERENT, 4))]),
        );
        assert_eq!(context.read(&reference), Ok(TestValue::new(REFERENT, 9)));
        assert_eq!(context.is_mutated(reference.allocation_id()), Ok(true));

        // The replacement itself must be a value rather than a second reference handle.
        let handles =
            vec![ReferenceDischargeValue::Reference(reference.clone()), ReferenceDischargeValue::Reference(reference)];
        assert_eq!(
            Swap::new().discharge_references(&context, &EmptyRegionDriver, handles.as_slice()),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge expected a replacement value but received {}",
                handles[1],
            ))),
        );
    }

    #[test]
    fn test_reference_swap_reference_discharge_transforms() {
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(ArrayType::new_static(DataType::F32, [2, 3]).into());
        let update = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let index = builder.add_constant(TestIrValue::Array(Array::scalar(-1i32).unwrap()));
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let operation = TestIrReferenceSwapOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        let outputs = builder
            .add_instruction(operation, Vec::new(), vec![reference, update, index], None)
            .unwrap()
            .to_vec();
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let output_ids = vec![outputs[0], frozen];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(output_ids, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let discharged = program.discharge_references(0).unwrap();
        assert!(!discharged.program().entry_region_ref().contains_reference_accesses_in_closure());
        assert_eq!(
            discharged.program().interpret(vec![
                TestIrValue::Array(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap()),
                TestIrValue::Array(Array::scalar(9f32).unwrap())
            ]),
            Ok(vec![
                TestIrValue::Array(Array::scalar(6f32).unwrap()),
                TestIrValue::Array(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 9.]).unwrap())
            ])
        );
    }

    #[test]
    fn test_reference_swap_reference_discharge_views_out_of_range_indices() {
        // `y = swap(r[transforms=[index(axis=0, index=dynamic)]](i), x)` replaces the same element whether the
        // transform runs on eager reference handles or is discharged into dynamic slices and updates. A negative signed
        // index counts from the end of the axis once and is then clamped, while an unsigned index keeps its full value
        // until it is clamped.
        let signed = [(-9i64, 0), (-3, 0), (-1, 2), (0, 0), (2, 2), (7, 2)]
            .map(|(index, position)| (Array::scalar(index).unwrap(), position));
        let unsigned =
            [(u64::MAX, 2), (u64::MAX - 1, 2)].map(|(index, position)| (Array::scalar(index).unwrap(), position));
        for (index, position) in signed.into_iter().chain(unsigned) {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let initial = builder.add_input(ArrayType::new_static(DataType::F32, [3]).into());
            let replacement = builder.add_input(ArrayType::scalar(DataType::F32).into());
            let binding = builder.add_input(index.r#type().into_owned().into());
            let reference =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
            let operation = TestIrReferenceSwapOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                axis: 0,
                index: ArrayReferenceTransformIndex::Dynamic,
            }]);
            let previous =
                builder.add_instruction(operation, Vec::new(), vec![reference, replacement, binding], None).unwrap()[0];
            let frozen =
                builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            let program = builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                    vec![previous, frozen],
                    vec![Placeholder; 3],
                    vec![Placeholder; 2],
                )
                .unwrap();
            let discharged = program.clone().discharge_references(0).unwrap();
            let inputs = vec![
                TestIrValue::Array(Array::vector(vec![1f32, 2., 3.]).unwrap()),
                TestIrValue::Array(Array::scalar(9f32).unwrap()),
                TestIrValue::Array(index.clone()),
            ];
            let mut installed = vec![1f32, 2., 3.];
            installed[position] = 9.;
            let expected = Ok(vec![
                TestIrValue::Array(Array::scalar([1f32, 2., 3.][position]).unwrap()),
                TestIrValue::Array(Array::vector(installed).unwrap()),
            ]);
            assert_eq!(program.interpret(inputs.clone()), expected, "eager execution with index `{index}`");
            assert_eq!(discharged.program().interpret(inputs), expected, "discharged execution with index `{index}`");
        }
    }

    #[test]
    fn test_reference_swap_projected() {
        type TestTracingContext = TracingContext<TestIrValue, TestIrOperation>;
        type TestTracer = Tracer<TestTracingContext>;

        // A projected reference binds the swap through its parent tracer's context and projects the previous value,
        // so the projected surface stages exactly the native operation.
        let array_type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2]));
        let (_, program) = TestTracingContext::trace(
            |inputs| {
                let [initial, replacement]: [TestTracer; 2] = inputs.try_into().unwrap();
                let initial = <TestTracer as ValueProjection<ArrayType>>::into_projected(initial)?;
                let replacement = <TestTracer as ValueProjection<ArrayType>>::into_projected(replacement)?;
                let reference = initial.reference_new()?;
                let _: &ProjectedValue<ReferenceType<ArrayType>, TestTracer> = &reference;
                let previous = reference.swap(&replacement)?;
                let _: &ProjectedValue<ArrayType, TestTracer> = &previous;
                let frozen = reference.freeze()?;
                Ok(vec![previous.into_value(), frozen.into_value()])
            },
            vec![array_type.clone(), array_type],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[2] .
                let %2:ref<f32[2]> = reference_new %0
                    %3:f32[2] = reference_swap %2 %1
                    %4:f32[2] = reference_freeze %2
                in (%3, %4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_reference_swap_staging() {
        // A traced reference stages the swap as the native variant of its operation family, with the previous value
        // as its output and the ordered-state effect of the program.
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let (output_types, program) = TracingContext::<TestIrValue, TestIrOperation>::trace(
            |inputs| {
                let [reference, replacement]: [Tracer<_>; 2] = inputs.try_into().unwrap();
                Ok(vec![reference.swap(&replacement)?])
            },
            vec![
                ArrayIrType::Reference(ReferenceType::new(array_type.clone())),
                ArrayIrType::Array(array_type.clone()),
            ],
        )
        .unwrap();
        assert_eq!(output_types, vec![ArrayIrType::Array(array_type)]);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ref<f32[2]>, %1:f32[2] .
                let %2:f32[2] = reference_swap %0 %1
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(program.effects().classes(), EffectClasses::single(EffectClass::OrderedState));
    }
}
