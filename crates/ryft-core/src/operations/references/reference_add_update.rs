use std::borrow::Cow;
use std::fmt::Display;
use std::marker::PhantomData;
use std::sync::LazyLock;

use crate::arrays::{ArrayIrType, ArrayIrValue, ArrayReferenceTransform, ArrayType, DataType};
use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::DifferentiableType;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::arithmetic::{Add, AddOperation};
use crate::operations::manipulation::reshaping::Reshape;
use crate::operations::manipulation::slicing::{Slice, UpdateSlice};
use crate::operations::references::reference_read::ReferenceReadOperation;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    BatchableReferenceTransform, Concretizable, EffectClasses, Effects, MaybeZero, NoReferenceTransform, NoReferent,
    Operation, OperationFormatter, OperationProvider, ProgramError, ProjectedValue, ReferenceAccessDescriptor,
    ReferenceAccessMode, ReferenceAccessOperation, ReferenceAccumulationPolicy, ReferenceDischargeContext,
    ReferenceDischargeDriver, ReferenceDischargeValue, ReferenceDischargeableOperation, ReferenceEffect,
    ReferenceMemberType, ReferenceTransform, ReferenceType, RegionInterface, Type, TypeError, Typed, Value,
    ValueProjection, batch_reference_transforms, infer_reference_view_type,
};

/// Canonical operation name for [`ReferenceAddUpdateOperation`].
pub const REFERENCE_ADD_UPDATE_OPERATION_NAME: &str = "reference_add_update";

/// Applies an ordered additive update that preserves the selected referent's exact type. The root and update are the
/// first two inputs. Dynamic bindings follow them in transform order and the path is stored in `Transform` metadata.
/// An empty path accesses the complete referent.
#[derive(Clone, Debug)]
pub struct ReferenceAddUpdateOperation<
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

impl<T: Type, U: Type, Transform: ReferenceTransform<Type = U, Referent = T>>
    ReferenceAddUpdateOperation<T, U, Transform>
{
    /// Creates a new [`ReferenceAddUpdateOperation`].
    pub const fn new() -> Self {
        Self { transforms: Vec::new(), marker: PhantomData }
    }

    /// Returns a copy of this [`ReferenceAddUpdateOperation`] with the provided transforms applied
    /// to the reference input.
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

    /// Returns the number of dynamic inputs supplied after the base inputs for this [`ReferenceAddUpdateOperation`].
    fn binding_count(&self) -> usize {
        self.transforms.iter().map(ReferenceTransform::binding_count).sum()
    }
}

impl<T: Type, U: Type, Transform: ReferenceTransform<Type = U, Referent = T>> Default
    for ReferenceAddUpdateOperation<T, U, Transform>
{
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Type, U: Type, Transform: ReferenceTransform<Type = U, Referent = T>> Display
    for ReferenceAddUpdateOperation<T, U, Transform>
where
    Self: Operation,
{
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type, U: Type, Transform: ReferenceTransform<Type = U, Referent = T>> Operation
    for ReferenceAddUpdateOperation<T, U, Transform>
where
    for<'t> &'t T: TryFrom<&'t U, Error = TypeError>,
    for<'t> &'t ReferenceType<T>: TryFrom<&'t U, Error = TypeError>,
    AddOperation<T>: Operation<Type = T>,
{
    type Type = U;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_ADD_UPDATE_OPERATION_NAME
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
            ReferenceAccessMode::Accumulate,
        )?;
        let update = <&T>::try_from(&input_types[1])?;
        let addition_outputs = AddOperation::<T>::new().infer_output_types(&[referent.clone(), update.clone()], &[])?;
        check_count!("output", addition_outputs, 1, TypeError);
        let addition_output = &addition_outputs[0];
        if addition_output != &referent {
            return Err(TypeError::invalid(format!(
                "`{REFERENCE_ADD_UPDATE_OPERATION_NAME}` addition output type `{addition_output}` \
                 must exactly match reference referent type `{referent}`",
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
                vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Accumulate }],
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
    for ReferenceAddUpdateOperation<T, U, Transform>
where
    ReferenceAddUpdateOperation<T, U, Transform>: Operation<Type = U>,
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
    P: ReferenceAccumulationPolicy<C, Referent = T, Transform = Transform>,
> ReferenceDischargeableOperation<C, P> for ReferenceAddUpdateOperation<T, U, Transform>
where
    ReferenceAddUpdateOperation<T, U, Transform>: Operation<Type = U>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        _driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        check_count!("input", inputs, 2 + self.binding_count(), ProgramError);
        let reference = inputs[0].try_as_reference("a reference to accumulate into")?;
        let bindings = inputs[2..]
            .iter()
            .map(|input| input.try_as_value("a reference transform binding").cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let update = inputs[1].try_as_value("an update value")?.clone();

        // The sum of the handle's referent and the update must itself be the handle's referent, which is exactly what
        // this operation's own inference states and what a universe's addition alone does not guarantee. Rules can be
        // called outside a validated program, so check the input types before changing reference state.
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(&input_types, &[])?;
        context.accumulate_through(reference, update, &self.transforms, &bindings)?;
        Ok(Vec::new())
    }
}

impl<
    T: Type,
    U: Type,
    Transform: ReferenceTransform<Type = U, Referent = T>,
    C: Domain<Type = U, Value: ReferenceAddUpdate<Transform, C::Value, C::Value>>,
> InterpretableOperation<C> for ReferenceAddUpdateOperation<T, U, Transform>
where
    ReferenceAddUpdateOperation<T, U, Transform>: Operation<Type = U>,
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2 + self.binding_count(), ProgramError);
        inputs[0].add_update_through(&inputs[1], &self.transforms, &inputs[2..])?;
        Ok(Vec::new())
    }
}

impl<
    T: Type,
    U: Type,
    Transform: ReferenceTransform<Type = U, Referent = T>,
    C: Context<Type = U, Operation: From<ReferenceAddUpdateOperation<T, U, Transform>>>,
> PartiallyEvaluatableOperation<C> for ReferenceAddUpdateOperation<T, U, Transform>
{
}

impl<
    T: Type,
    U: Type + From<ReferenceType<T>>,
    Transform: BatchableReferenceTransform<Type = U, Referent = T>,
    C: Context<Type = U, Operation: From<ReferenceAddUpdateOperation<T, U, Transform>>>,
    P: BatchingPolicy<C>,
> BatchableOperation<C, P> for ReferenceAddUpdateOperation<T, U, Transform>
where
    for<'t> &'t ReferenceType<T>: TryFrom<&'t U, Error = TypeError>,
    ReferenceAddUpdateOperation<T, U, Transform>: Operation<Type = U>,
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError> {
        // Broadcast or move the stored value to the final selected referent's batch axis.
        check_count!("input", inputs, 2 + self.binding_count(), ProgramError);
        let binding_axes = inputs[2..].iter().map(P::batch_axis).collect::<Vec<_>>();
        let (transforms, output_axis) = batch_reference_transforms(
            P::value(&inputs[0]).r#type().as_ref(),
            P::batch_axis(&inputs[0]),
            &self.transforms,
            &binding_axes,
        )?;

        let update = match (output_axis.axis(), P::batch_axis(&inputs[1]).axis()) {
            (Some(axis), _) => driver.align_batch_axis(context, inputs[1].clone(), axis)?,
            (None, None) => inputs[1].clone(),
            (None, Some(_)) => {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "`{REFERENCE_ADD_UPDATE_OPERATION_NAME}` cannot store a batched value into an unbatched \
                         reference; pass the reference as a batched input instead",
                    ),
                });
            }
        };

        let mut values = vec![P::value(&inputs[0]).clone(), P::value(&update).clone()];
        values.extend(inputs[2..].iter().map(|input| P::value(input).clone()));
        context.parent().bind(self.clone().with_transforms(transforms), Vec::new(), &values)?;
        Ok(Vec::new().into())
    }
}

impl_differentiable_operation! {
    <T, U, Transform> ReferenceAddUpdateOperation<T, U, Transform>,
    jvp<C>
    where
        T: Type,
        U: DifferentiableType,
        Transform: ReferenceTransform<Type = U, Referent = T>,
        C: Context<Type = U, Operation: From<ReferenceAddUpdateOperation<T, U, Transform>>>,
    {
        |operation, context, _driver, inputs| {
            // Accumulate into the tangent reference alongside the primal as a zero update needs no tangent work, and
            // reject a live tangent without tangent storage before either reference can be changed.
            check_count!("input", inputs, 2 + operation.binding_count(), ProgramError);
            let stored = match (inputs[0].tangent(), inputs[1].tangent()) {
                (MaybeZero::Value(reference), tangent) => Some((reference, tangent.clone())),
                (MaybeZero::Zero(_), MaybeZero::Zero(_)) => None,
                (MaybeZero::Zero(_), MaybeZero::Value(_)) => {
                    return Err(ProgramError::InvalidArgument {
                        message: format!(
                            "`{REFERENCE_ADD_UPDATE_OPERATION_NAME}` writes a live tangent into a reference that \
                             carries no tangent; pass the reference as a differentiated input instead of capturing it",
                        ),
                    }
                    .into());
                }
            };
            let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            context.primal().bind(operation.clone(), Vec::new(), &primal_inputs)?;
            if let Some((tangent_reference, MaybeZero::Value(tangent))) = stored {
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
        O: Operation<Type = U> + From<ReferenceReadOperation<T, U, Transform>>,
        ReferenceReadOperation<T, U, Transform>: Operation<Type = U>,
    {
        |operation, context, driver, inputs, outputs, accumulators| {
            // An accumulation maps `(state, x) ↦ state + x`, so its transpose reads the cotangent reference as the
            // cotangent of the update and leaves the reference's contents unchanged for the earlier accesses. An
            // accumulator that nothing has reached yet holds zero, so nothing is staged and the update's cotangent
            // stays symbolic.
            check_count!("input", inputs, 2 + operation.binding_count(), ProgramError);
            check_count!("output", outputs, 0, ProgramError);
            check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
            let Some(accumulator) = context.cotangent_reference_if_allocated(driver, 0)? else {
                return Ok(());
            };
            if accumulators[1].is_needed() {
                let mut adjoint_inputs = vec![accumulator];
                for input in &inputs[2..] {
                    adjoint_inputs.push(input.as_known().cloned().ok_or_else(|| ProgramError::UnsupportedOperation {
                        message: "reference transform bindings must be known when transposing an access".to_string(),
                    })?);
                }
                let contribution = context
                    .bind(
                        ReferenceReadOperation::<T, U, Transform>::new().with_transforms(operation.transforms.clone()),
                        Vec::new(),
                        &adjoint_inputs,
                    )?
                    .remove(0);
                accumulators[1].accumulate(context, MaybeZero::Value(contribution))?;
            }
            Ok(())
        }
    },
}

// Reverse-mode transposition requires an accumulation provider for reference cotangents even when instantiated
// with a reference-free type universe. This implementation lets ordinary scalar/array gradients satisfy that
// bound without adding reference support; direct accumulation requests are unsupported.
impl<O: Operation<Type = DataType>> OperationProvider<DataType, ReferenceAddUpdateOperation<NoReferent, DataType>>
    for O
{
    type Operation = Self;

    fn provide(
        _request: ReferenceAddUpdateOperation<NoReferent, DataType>,
        input_types: &[&DataType],
    ) -> Result<Self, ProgramError> {
        check_count!("input", input_types, 2, ProgramError);
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "`{REFERENCE_ADD_UPDATE_OPERATION_NAME}` is not supported in a reference-free type universe",
            ),
        })
    }
}

// Reverse-mode transposition requires an accumulation provider for reference cotangents even when instantiated
// with a reference-free type universe. This implementation lets ordinary scalar/array gradients satisfy that
// bound without adding reference support; direct accumulation requests are unsupported.
impl<O: Operation<Type = ArrayType>> OperationProvider<ArrayType, ReferenceAddUpdateOperation<NoReferent, ArrayType>>
    for O
{
    type Operation = Self;

    fn provide(
        _request: ReferenceAddUpdateOperation<NoReferent, ArrayType>,
        input_types: &[&ArrayType],
    ) -> Result<Self, ProgramError> {
        check_count!("input", input_types, 2, ProgramError);
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "`{REFERENCE_ADD_UPDATE_OPERATION_NAME}` is not supported in a reference-free type universe",
            ),
        })
    }
}

impl<
    Transform: ReferenceTransform<Type = ArrayIrType, Referent = ArrayType>,
    O: Operation<Type = ArrayIrType> + From<ReferenceAddUpdateOperation<ArrayType, ArrayIrType, Transform>>,
> OperationProvider<ArrayIrType, ReferenceAddUpdateOperation<ArrayType, ArrayIrType, Transform>> for O
{
    type Operation = Self;

    fn provide(
        request: ReferenceAddUpdateOperation<ArrayType, ArrayIrType, Transform>,
        input_types: &[&ArrayIrType],
    ) -> Result<Self, ProgramError> {
        check_count!("input", input_types, 2 + request.binding_count(), ProgramError);
        let input_types = input_types.iter().map(|input| (*input).clone()).collect::<Vec<_>>();
        request.infer_output_types(&input_types, &[])?;
        Ok(request.into())
    }
}

/// Capability to add an update into the value stored by a reference in program order. Concrete values implement their
/// runtime update semantics directly. Values whose dispatch domain is a [`Context`] project the referent of their
/// reference type through [`ReferenceMemberType::referent`] and use the context's operation family to select and bind
/// the update operation through [`OperationProvider`]. The selected operation may use a downstream payload, and a
/// family without reference operations may reject construction. This capability supports arbitrary reference-aware type
/// universes so reverse-mode differentiation can accumulate cotangents through it without depending on [`ArrayIrType`].
pub trait ReferenceAddUpdate<Transform: ReferenceTransform, Binding = Self, Update = Self>: Sized {
    /// Adds `update` to the state selected by the supplied transforms in program order.
    ///
    /// # Parameters
    ///
    ///   - `update`: Value added into the viewed reference state. Its sum with the selected referent must have exactly
    ///     the selected referent type.
    ///   - `transforms`: Transforms applied in order to the reference.
    ///   - `bindings`: Dynamic inputs in transform order.
    fn add_update_through(
        &self,
        update: &Update,
        transforms: &[Transform],
        bindings: &[Binding],
    ) -> Result<(), ProgramError>;

    /// Adds `update` to the complete referent in program order.
    #[inline]
    fn add_update(&self, update: &Update) -> Result<(), ProgramError> {
        self.add_update_through(update, &[], &[])
    }
}

impl<A: Value<Type = ArrayType> + Concretizable<i128> + Add + Reshape + Slice + UpdateSlice>
    ReferenceAddUpdate<ArrayReferenceTransform> for ArrayIrValue<A>
{
    fn add_update_through(
        &self,
        update: &Self,
        transforms: &[ArrayReferenceTransform],
        bindings: &[Self],
    ) -> Result<(), ProgramError> {
        let operation = ReferenceAddUpdateOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new();
        let operation = operation.with_transforms(transforms.to_vec());
        let mut input_types = vec![self.r#type().into_owned(), update.r#type().into_owned()];
        input_types.extend(bindings.iter().map(|value| value.r#type().into_owned()));
        operation.infer_output_types(&input_types, &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        let reference = reference.with_transforms(transforms, bindings)?;
        let update = <Self as ValueProjection<ArrayType>>::projected(update)?;
        reference.add_update(update)
    }
}

// Staged values delegate selection to their operation family, including a downstream reference universe's family,
// over the referent of the reference being updated. A value that is not a reference member of its universe has no
// referent and is rejected before any selection.
impl<Transform, V> ReferenceAddUpdate<Transform, V, V> for V
where
    Transform: ReferenceTransform<Type = V::Type, Referent = <V::Type as ReferenceMemberType>::Referent>,
    V: Value<
            Type: ReferenceMemberType,
            DispatchDomain: Context<
                Operation: OperationProvider<
                    V::Type,
                    ReferenceAddUpdateOperation<<V::Type as ReferenceMemberType>::Referent, V::Type, Transform>,
                    Operation = <V::DispatchDomain as Domain>::Operation,
                >,
            >,
        >,
{
    fn add_update_through(
        &self,
        update: &Self,
        transforms: &[Transform],
        bindings: &[Self],
    ) -> Result<(), ProgramError> {
        let reference_type = self.r#type();
        reference_type
            .referent()
            .ok_or_else(|| TypeError::invalid(format!("expected reference type but got `{reference_type}`")))?;
        let mut inputs = vec![self.clone(), update.clone()];
        inputs.extend_from_slice(bindings);
        let input_types = inputs.iter().map(Typed::r#type).collect::<Vec<_>>();
        let operation = <V::DispatchDomain as Domain>::Operation::provide(
            ReferenceAddUpdateOperation::new().with_transforms(transforms.to_vec()),
            &input_types.iter().map(|r#type| r#type.as_ref()).collect::<Vec<_>>(),
        )?;
        self.dispatch_domain().bind(operation, Vec::new(), &inputs)?;
        Ok(())
    }
}

impl<Transform, V> ReferenceAddUpdate<Transform, V, ProjectedValue<ArrayType, V>>
    for ProjectedValue<ReferenceType<ArrayType>, V>
where
    Transform: ReferenceTransform<Type = ArrayIrType, Referent = ArrayType>,
    V: Value<
            Type = ArrayIrType,
            DispatchDomain: Context<Operation: From<ReferenceAddUpdateOperation<ArrayType, ArrayIrType, Transform>>>,
        >,
{
    fn add_update_through(
        &self,
        update: &ProjectedValue<ArrayType, V>,
        transforms: &[Transform],
        bindings: &[V],
    ) -> Result<(), ProgramError> {
        let mut inputs = vec![self.value().clone(), update.value().clone()];
        inputs.extend_from_slice(bindings);
        let operation = ReferenceAddUpdateOperation::new().with_transforms(transforms.to_vec());
        self.value().dispatch_domain().bind(operation, Vec::new(), &inputs)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrValue, ArrayOperation, ArrayReference,
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
    use crate::partial::{PartialEvaluationContext, PartialEvaluationValue, PartialValue, ReferencePlacement};
    use crate::programs::{EffectClass, EmptyRegionDriver, ProgramBuilder};
    use crate::tracing::{Tracer, TracingContext};

    use super::*;

    type TestIrValue = ArrayIrValue<Array>;
    type TestIrOperation = ArrayIrOperation<Array>;
    type TestIrContext = EagerContext<TestIrValue, TestIrOperation>;
    type TestIrReferenceAddUpdateOperation =
        ReferenceAddUpdateOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>;

    #[test]
    fn test_reference_add_update() {
        let operation = AddUpdate::new();
        assert_eq!(AddUpdate::default().to_string(), operation.to_string());
        assert_eq!(operation.name(), REFERENCE_ADD_UPDATE_OPERATION_NAME);
        assert_eq!(operation.to_string(), REFERENCE_ADD_UPDATE_OPERATION_NAME);
        assert_eq!(
            format!("{operation:?}"),
            format!(
                "ReferenceAddUpdateOperation {{ transforms: [], marker: {:?} }}",
                PhantomData::<fn() -> (TestReferent, TestType)>
            ),
        );

        // An accumulation orders against other state effects, accumulates into its reference input,
        // and aliases nothing.
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedState));
        assert_eq!(
            operation.effects().reference_effects(),
            &[ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Accumulate }],
        );
    }

    #[test]
    fn test_reference_add_update_type_inference() {
        let referent = TestReferent::new(7, 16);
        let value = TestType::Value(referent);
        let reference = TestType::Reference(ReferenceType::new(referent));
        check_operation_type_inference!(
            operation = AddUpdate::new(),
            cases = [
                {
                    input_types = [reference.clone(), value.clone()],
                    output_types = [],
                },
                {
                    input_types = [reference.clone(), TestType::Value(TestReferent::new(7, 32))],
                    error = "`reference_add_update` addition output type `value<i7,p32>` must exactly match reference \
                             referent type `value<i7,p16>`",
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
            AddUpdate::new().infer_output_types(&[reference, value], std::slice::from_ref(&region)),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );

        // A universe only needs to project references and values out of itself for an accumulation to type-check.
        check_operation_type_inference!(
            operation = ReferenceAddUpdateOperation::<TestReferent, StoreUniverse>::new(),
            cases = [{
                input_types = [
                    StoreUniverse::Reference(ReferenceType::new(referent)),
                    StoreUniverse::Value(referent),
                ],
                output_types = [],
            }],
        );

        // An array update may broadcast against the referent as long as the sum keeps the exact referent type.
        let vector_type = ArrayType::new_static(DataType::F32, [2]);
        check_operation_type_inference!(
            operation = TestIrReferenceAddUpdateOperation::new(),
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
                        ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
                    ],
                    output_types = [],
                },
                {
                    input_types = [
                        ArrayIrType::Reference(ReferenceType::new(vector_type.clone())),
                        ArrayIrType::Array(ArrayType::new_static(DataType::F64, [2])),
                    ],
                    error = "`reference_add_update` addition output type `f64[2]` must exactly match reference \
                             referent type `f32[2]`",
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
    fn test_reference_add_update_type_inference_transforms() {
        let operation = TestIrReferenceAddUpdateOperation::new().with_transforms(vec![
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
            "reference_add_update [transforms=[index(axis=0, index=1), dynamic_index(axis=0)]]",
        );
        let descriptor = operation.reference_access_descriptor(0).unwrap();
        assert_eq!(descriptor.transforms(), operation.transforms());
        assert_eq!(descriptor.bindings(), 2..3);
        assert!(operation.reference_access_descriptor(1).is_none());
    }

    #[test]
    fn test_reference_add_update_interpretation() {
        let live = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]).unwrap());
        let reference = TestIrValue::Reference(live.clone());

        // An update is added into the stored value in place and produces no output.
        assert_eq!(
            InterpretableOperation::<TestIrContext>::interpret(
                &TestIrReferenceAddUpdateOperation::new(),
                &TestIrContext::new(),
                &EmptyRegionDriver,
                &[reference.clone(), TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap())],
            ),
            Ok(Vec::new()),
        );
        assert_eq!(live.read(), Ok(Array::vector(vec![4.0_f32, 6.0]).unwrap()));

        // Broadcasting is valid only because the computed sum preserves the exact stored type.
        assert_eq!(
            InterpretableOperation::<TestIrContext>::interpret(
                &TestIrReferenceAddUpdateOperation::new(),
                &TestIrContext::new(),
                &EmptyRegionDriver,
                &[reference.clone(), TestIrValue::Array(Array::scalar(1.0_f32).unwrap())],
            ),
            Ok(Vec::new()),
        );
        assert_eq!(live.read(), Ok(Array::vector(vec![5.0_f32, 7.0]).unwrap()));

        // Exact input inference runs before the accumulation, so a rejected update leaves the stored value unchanged.
        assert_eq!(
            InterpretableOperation::<TestIrContext>::interpret(
                &TestIrReferenceAddUpdateOperation::new(),
                &TestIrContext::new(),
                &EmptyRegionDriver,
                &[reference.clone(), TestIrValue::Array(Array::vector(vec![3.0_f64, 4.0]).unwrap())],
            ),
            Err(TypeError::invalid(
                "`reference_add_update` addition output type `f64[2]` must exactly match reference referent type \
                 `f32[2]`",
            )
            .into()),
        );
        assert_eq!(live.read(), Ok(Array::vector(vec![5.0_f32, 7.0]).unwrap()));

        // Each input must be the member kind the operation expects.
        let array = TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap());
        assert_eq!(
            InterpretableOperation::<TestIrContext>::interpret(
                &TestIrReferenceAddUpdateOperation::new(),
                &TestIrContext::new(),
                &EmptyRegionDriver,
                &[array.clone(), array],
            ),
            Err(TypeError::invalid("expected reference type but got array type").into()),
        );
        assert_eq!(
            InterpretableOperation::<TestIrContext>::interpret(
                &TestIrReferenceAddUpdateOperation::new(),
                &TestIrContext::new(),
                &EmptyRegionDriver,
                &[reference.clone(), reference],
            ),
            Err(TypeError::invalid("expected array type but got reference type").into()),
        );
        assert_eq!(live.read(), Ok(Array::vector(vec![5.0_f32, 7.0]).unwrap()));
    }

    #[test]
    fn test_reference_add_update_interpretation_transforms() {
        let context = EagerContext::<TestIrValue, TestIrOperation>::new();
        let reference =
            TestIrValue::Reference(ArrayReference::new(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap()));
        let index = TestIrValue::Array(Array::scalar(-1i32).unwrap());
        let operation = TestIrReferenceAddUpdateOperation::new().with_transforms(vec![
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
            Ok(TestIrValue::Array(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 15.]).unwrap())),
        );
    }

    #[test]
    fn test_reference_add_update_partial_evaluation() {
        // Program replay uses the `Stage` placement: an accumulation stages regardless of input knowledge, the live
        // handle is passed to the residual program as a known reference input, and the sum is formed only when that
        // program runs.
        let live = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        check_operation_partial_evaluation!(
            backend = (TestIrValue, TestIrOperation),
            operation = TestIrReferenceAddUpdateOperation::new(),
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
        assert_eq!(live.read(), Ok(Array::scalar(6.0_f32).unwrap()));

        // Under the `Stage` placement the live state is untouched at partial evaluation time even when every input
        // is known.
        let reference = PartialEvaluationValue::known(TestIrValue::Reference(live.clone()));
        let update = PartialEvaluationValue::known(TestIrValue::Array(Array::scalar(4.0_f32).unwrap()));
        let staging =
            PartialEvaluationContext::new(TestIrContext::new()).with_reference_placement(ReferencePlacement::Stage);
        let outputs = staging.fold_or_residualize(
            TestIrReferenceAddUpdateOperation::new(),
            Vec::new(),
            &[reference.clone(), update.clone()],
        );
        assert_eq!(outputs.map(|outputs| outputs.len()), Ok(0));
        assert_eq!(live.read(), Ok(Array::scalar(6.0_f32).unwrap()));

        // Under the default `Execute` placement an all-known accumulation folds: it runs against the live state in
        // program order at partial evaluation time.
        let executing = PartialEvaluationContext::new(TestIrContext::new());
        let outputs =
            executing.fold_or_residualize(TestIrReferenceAddUpdateOperation::new(), Vec::new(), &[reference, update]);
        assert_eq!(outputs.map(|outputs| outputs.len()), Ok(0));
        assert_eq!(live.read(), Ok(Array::scalar(10.0_f32).unwrap()));
    }

    #[test]
    fn test_reference_add_update_partial_evaluation_transforms() {
        // The known root enters the residual program while the dynamic index remains an ordinary unknown input.
        let live = ArrayReference::new(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap());
        check_operation_partial_evaluation!(
            backend = (TestIrValue, TestIrOperation),
            operation = TestIrReferenceAddUpdateOperation::new().with_transforms(vec![
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
        assert_eq!(live.read(), Ok(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 8.]).unwrap()));
    }

    #[test]
    fn test_reference_add_update_batching() {
        let extent = TestIrValue::Dimension(
            DimensionValue::new(DimensionType::new("batch", DimensionBounds::unbounded()), 2).unwrap(),
        );
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(TestIrContext::new(), extent);
        let packed_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let initial = TestIrValue::Array(
            Array::from_elements::<f32>(packed_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        );
        let reference = initial.reference_new().unwrap();
        let batched =
            BatchingTracer::new(context.clone(), ArrayIrBatch::new(reference.clone(), BatchAxis::new(0)).unwrap());

        // An update mapped at the reference's batch axis accumulates packed.
        let update = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::new(
                TestIrValue::Array(Array::from_elements::<f32>(packed_type.clone(), &[1.0; 6]).unwrap()),
                BatchAxis::new(0),
            )
            .unwrap(),
        );
        let outputs = context.bind(ReferenceAddUpdateOperation::new(), Vec::new(), &[batched, update]).unwrap();
        assert!(outputs.is_empty());
        assert_eq!(
            reference.read(),
            Ok(TestIrValue::Array(
                Array::from_elements::<f32>(packed_type.clone(), &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap()
            )),
        );

        // A batched update cannot accumulate into an unbatched reference.
        let replicated = BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(reference.clone()));
        let update = BatchingTracer::new(
            context.clone(),
            ArrayIrBatch::new(
                TestIrValue::Array(Array::from_elements::<f32>(packed_type, &[1.0; 6]).unwrap()),
                BatchAxis::new(0),
            )
            .unwrap(),
        );
        let error = context.bind(ReferenceAddUpdateOperation::new(), Vec::new(), &[replicated, update]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<BatchingError>(),
            Some(&BatchingError::UnsupportedOperation {
                message: "`reference_add_update` cannot store a batched value into an unbatched reference; pass the \
                          reference as a batched input instead"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_reference_add_update_batching_transforms() {
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
        let operation = TestIrReferenceAddUpdateOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        let outputs = context.bind(operation, Vec::new(), &[input, update, index]).unwrap();
        assert!(outputs.is_empty());
        assert_eq!(
            reference.read(),
            Ok(TestIrValue::Array(
                Array::from_elements::<f32>(packed_type, &[1f32, 2., 3., 4., 5., 6., 7., 8., 29., 10., 11., 42.])
                    .unwrap()
            )),
        );
    }

    #[test]
    fn test_reference_add_update_differentiation() {
        let context = DifferentiationContext::fused(TestIrContext::new());
        let reference = TestIrValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()).reference_new().unwrap();
        let tangent_reference = TestIrValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap()).reference_new().unwrap();
        let active = DifferentiationTracer::new(
            DifferentiationDual::new(reference.clone(), tangent_reference.clone()).unwrap(),
            context.clone(),
        );

        // Addition is linear, so the update's tangent accumulates into the tangent reference.
        let update = DifferentiationTracer::new(
            DifferentiationDual::new(
                TestIrValue::Array(Array::vector(vec![5.0_f32, 6.0]).unwrap()),
                TestIrValue::Array(Array::vector(vec![7.0_f32, 8.0]).unwrap()),
            )
            .unwrap(),
            context.clone(),
        );
        let outputs = context.bind(ReferenceAddUpdateOperation::new(), Vec::new(), &[active.clone(), update]).unwrap();
        assert!(outputs.is_empty());
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![6.0_f32, 8.0]).unwrap())));
        assert_eq!(tangent_reference.read(), Ok(TestIrValue::Array(Array::vector(vec![10.0_f32, 12.0]).unwrap())));

        // A symbolic zero update tangent accumulates nothing and is not instantiated.
        let update = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(TestIrValue::Array(Array::vector(vec![1.0_f32, 1.0]).unwrap()))
                .unwrap(),
            context.clone(),
        );
        context.bind(ReferenceAddUpdateOperation::new(), Vec::new(), &[active, update]).unwrap();
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![7.0_f32, 9.0]).unwrap())));
        assert_eq!(tangent_reference.read(), Ok(TestIrValue::Array(Array::vector(vec![10.0_f32, 12.0]).unwrap())));

        // Staged, the zero-tangent accumulation is therefore elided from the tangent side entirely: the fused program
        // accumulates the constant into the primal reference and leaves the tangent reference untouched.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference_atom = builder.add_input(ReferenceType::new(ArrayType::scalar(DataType::F32)).into());
        let constant = builder.add_constant(TestIrValue::Array(Array::scalar(1.0_f32).unwrap()));
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference_atom, constant], None)
            .unwrap();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference_atom], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:ref<f32[]>, %1:ref<f32[]> .
                let %2:f32[] = const 1.0
                    () = reference_add_update %0 %2
                in (%0, %1)
            "}
            .trim_end(),
        );

        // A plumbing reference accepts an update without a live tangent, while a live update tangent has no tangent
        // reference to accumulate into and the rejection precedes the primal accumulation.
        let plumbing = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(reference.clone()).unwrap(),
            context.clone(),
        );
        let update = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(TestIrValue::Array(Array::vector(vec![1.0_f32, 1.0]).unwrap()))
                .unwrap(),
            context.clone(),
        );
        context.bind(ReferenceAddUpdateOperation::new(), Vec::new(), &[plumbing.clone(), update]).unwrap();
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![8.0_f32, 10.0]).unwrap())));
        let update = DifferentiationTracer::new(
            DifferentiationDual::new(
                TestIrValue::Array(Array::vector(vec![1.0_f32, 1.0]).unwrap()),
                TestIrValue::Array(Array::vector(vec![1.0_f32, 1.0]).unwrap()),
            )
            .unwrap(),
            context.clone(),
        );
        let error = context.bind(ReferenceAddUpdateOperation::new(), Vec::new(), &[plumbing, update]).unwrap_err();
        assert_eq!(
            error,
            ProgramError::InvalidArgument {
                message:
                    "`reference_add_update` writes a live tangent into a reference that carries no tangent; pass the \
                     reference as a differentiated input instead of capturing it"
                        .to_string(),
            },
        );
        assert_eq!(reference.read(), Ok(TestIrValue::Array(Array::vector(vec![8.0_f32, 10.0]).unwrap())));
        assert_eq!(tangent_reference.read(), Ok(TestIrValue::Array(Array::vector(vec![10.0_f32, 12.0]).unwrap())));
    }

    #[test]
    fn test_reference_add_update_differentiation_transforms() {
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
        let operation = TestIrReferenceAddUpdateOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        let outputs = context.bind(operation, Vec::new(), &[input, update, index]).unwrap();
        assert!(outputs.is_empty());
        assert_eq!(
            reference.read(),
            Ok(TestIrValue::Array(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 15.]).unwrap())),
        );
        assert_eq!(
            tangent_reference.read(),
            Ok(TestIrValue::Array(Array::matrix(2, 3, vec![10f32, 20., 30., 40., 50., 150.]).unwrap())),
        );
    }

    #[test]
    fn test_reference_add_update_transposition() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));

        // The accumulation transposes through the cotangent accumulator of its reference input, which only a
        // transposition context scoped to the instruction being transposed can resolve, so a detached context rejects
        // it.
        let inputs = [PartialValue::Unknown(reference_type), PartialValue::Unknown(scalar_type.clone())];
        let mut context = TranspositionContext::new(TracingContext::<TestIrValue, TestIrOperation>::new());
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        assert!(matches!(
            TestIrReferenceAddUpdateOperation::new().transpose(&mut context, &EmptyRegionDriver, &inputs, &[], &accumulators),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message)))
                if message == "input 0 has no reference root in a transposition context that is not scoped to a \
                               reference-carrying instruction",
        ));

        // `r = new(v); add_update(r, x); y = freeze(r)`: the freeze lands `ȳ` in the allocation's accumulator, the
        // accumulation reads it as `x̄ = ȳ` and leaves it unchanged, and the allocation freezes it into `v̄ = ȳ`.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(scalar_type.clone());
        let update = builder.add_input(scalar_type.clone());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
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
                    %3:f32[] = reference_read %2
                    %4:f32[] = reference_freeze %2
                in (%4, %3)
            "}
            .trim_end(),
        );
        assert_eq!(
            transposed.interpret(vec![TestIrValue::Array(Array::scalar(5.0_f32).unwrap())]),
            Ok(vec![
                TestIrValue::Array(Array::scalar(5.0_f32).unwrap()),
                TestIrValue::Array(Array::scalar(5.0_f32).unwrap()),
            ]),
        );

        // An update whose cotangent nobody requested stages no read of the accumulator.
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[] = zero [type=f32[]]
                    %3:ref<f32[]> = reference_new %2
                    () = reference_add_update %3 %0
                    %4:f32[] = reference_freeze %3
                in (%4)
            "}
            .trim_end(),
        );
        assert_eq!(
            transposed.interpret(vec![
                TestIrValue::Array(Array::scalar(5.0_f32).unwrap()),
                TestIrValue::Array(Array::scalar(7.0_f32).unwrap()),
            ]),
            Ok(vec![TestIrValue::Array(Array::scalar(5.0_f32).unwrap())]),
        );

        // `r = new(v); add_update(r, x)` with no later access leaves the allocation's accumulator unallocated, so the
        // accumulation stages nothing and both cotangents stay symbolic zeros.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(scalar_type.clone());
        let update = builder.add_input(scalar_type);
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
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
    fn test_reference_add_update_transposition_transforms() {
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(ArrayType::new_static(DataType::F32, [2, 3]).into());
        let update = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let index = builder.add_constant(TestIrValue::Array(Array::scalar(-1i32).unwrap()));
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let operation = TestIrReferenceAddUpdateOperation::new().with_transforms(vec![
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
            transposed.to_string(),
            indoc! {"
                lambda %0:f32[2, 3] .
                let %1:f32[2, 3] = zero [type=f32[2, 3]]
                    %2:ref<f32[2, 3]> = reference_new %1
                    %3:i32[] = const -1
                    () = reference_add_update %2 %0
                    %4:f32[] = reference_read [transforms=[index(axis=0, index=1), dynamic_index(axis=0)]] %2 %3
                    %5:f32[2, 3] = reference_freeze %2
                in (%5, %4)"},
        );
        assert_eq!(
            transposed
                .interpret(vec![TestIrValue::Array(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap())]),
            Ok(vec![
                TestIrValue::Array(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap()),
                TestIrValue::Array(Array::scalar(6f32).unwrap())
            ])
        );

        // The adjoint read addresses the same elements as the update, so its dynamic binding must be known when
        // transposing. Transposing with respect to the index leaves that binding unknown.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(ArrayType::new_static(DataType::F32, [2, 3]).into());
        let update = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let index = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let operation = TestIrReferenceAddUpdateOperation::new().with_transforms(vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ]);
        builder.add_instruction(operation, Vec::new(), vec![reference, update, index], None).unwrap();
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![frozen], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            program.transpose_with_respect_to(&[0, 1, 2], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "reference transform bindings must be known when transposing an access",
        ));
    }

    #[test]
    fn test_reference_add_update_reference_discharge() {
        // An accumulation produces no output and replaces the current state with its sum with the update.
        let (context, reference) = allocated_reference(4);
        let inputs = vec![
            ReferenceDischargeValue::Reference(reference.clone()),
            ReferenceDischargeValue::Value(TestValue::new(REFERENT, 9)),
        ];
        assert_eq!(
            AddUpdate::new().discharge_references(&context, &EmptyRegionDriver, inputs.as_slice()),
            Ok(Vec::new()),
        );
        assert_eq!(context.read(&reference), Ok(TestValue::new(REFERENT, 13)));
        assert_eq!(context.is_mutated(reference.allocation_id()), Ok(true));

        // An update whose sum with the referent would not itself be the referent is rejected by this operation's own
        // inference before the universe accumulates anything, so the allocation keeps its previous state.
        let promoted = vec![
            ReferenceDischargeValue::Reference(reference.clone()),
            ReferenceDischargeValue::Value(TestValue::new(TestReferent::new(7, 32), 1)),
        ];
        assert_eq!(
            AddUpdate::new().discharge_references(&context, &EmptyRegionDriver, promoted.as_slice()),
            Err(TypeError::invalid(
                "`reference_add_update` addition output type `value<i7,p32>` must exactly match reference referent \
             type `value<i7,p16>`",
            )
            .into()),
        );
        assert_eq!(context.read(&reference), Ok(TestValue::new(REFERENT, 13)));
    }

    #[test]
    fn test_reference_add_update_reference_discharge_transforms() {
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let initial = builder.add_input(ArrayType::new_static(DataType::F32, [2, 3]).into());
        let update = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let index = builder.add_constant(TestIrValue::Array(Array::scalar(-1i32).unwrap()));
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let operation = TestIrReferenceAddUpdateOperation::new().with_transforms(vec![
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
            Ok(vec![TestIrValue::Array(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 15.]).unwrap())])
        );
    }

    #[test]
    fn test_reference_add_update_provider() {
        // The composite array family selects its own accumulation operation for an array referent.
        assert!(matches!(
            TestIrOperation::provide(
                ReferenceAddUpdateOperation::new(),
                &[
                    &ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))),
                    &ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
                ],
            ),
            Ok(ArrayIrOperation::ReferenceAddUpdate(_)),
        ));

        let array_type = ArrayType::scalar(DataType::F32);
        assert!(matches!(
            ArrayOperation::<Array>::provide(ReferenceAddUpdateOperation::new(), &[&array_type, &array_type]),
            Err(ProgramError::UnsupportedOperation { .. }),
        ));
        assert!(matches!(
            AddOperation::<DataType>::provide(ReferenceAddUpdateOperation::new(), &[&DataType::F32, &DataType::F32]),
            Err(ProgramError::UnsupportedOperation { .. }),
        ));
        assert!(matches!(
            TestIrOperation::provide(ReferenceAddUpdateOperation::new(), &[]),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        ));
        assert!(matches!(
            TestIrOperation::provide(
                ReferenceAddUpdateOperation::new(),
                &[&ArrayIrType::Array(array_type.clone()), &ArrayIrType::Array(array_type.clone())],
            ),
            Err(ProgramError::Type(_)),
        ));

        // A request with a nonempty transform path expects its dynamic bindings after the root and the update.
        let viewed = TestIrReferenceAddUpdateOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
            axis: 0,
            index: ArrayReferenceTransformIndex::Dynamic,
        }]);
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2])));
        let value_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let index_type = ArrayIrType::Array(ArrayType::scalar(DataType::I32));
        assert!(matches!(
            TestIrOperation::provide(viewed.clone(), &[&reference_type, &value_type, &index_type]),
            Ok(ArrayIrOperation::ReferenceAddUpdate(operation)) if operation.transforms() == viewed.transforms(),
        ));
        assert!(matches!(
            TestIrOperation::provide(viewed.clone(), &[&reference_type, &value_type]),
            Err(ProgramError::InvalidInputCount { expected: 3, actual: 2 }),
        ));
        assert!(matches!(
            TestIrOperation::provide(viewed, &[&reference_type, &value_type, &index_type, &index_type]),
            Err(ProgramError::InvalidInputCount { expected: 3, actual: 4 }),
        ));
    }

    #[test]
    fn test_reference_add_update_projected() {
        type TestTracingContext = TracingContext<TestIrValue, TestIrOperation>;
        type TestTracer = Tracer<TestTracingContext>;

        // A projected reference binds the accumulation through its parent tracer's context, so the projected surface
        // stages exactly the native operation.
        let array_type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2]));
        let (_, program) = TestTracingContext::trace(
            |inputs| {
                let [initial, update]: [TestTracer; 2] = inputs.try_into().unwrap();
                let initial = <TestTracer as ValueProjection<ArrayType>>::into_projected(initial)?;
                let update = <TestTracer as ValueProjection<ArrayType>>::into_projected(update)?;
                let reference = initial.reference_new()?;
                let _: &ProjectedValue<ReferenceType<ArrayType>, TestTracer> = &reference;
                reference.add_update(&update)?;
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
                    () = reference_add_update %2 %1
                    %3:f32[2] = reference_freeze %2
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_reference_add_update_staging() {
        // A traced reference selects the accumulation through its operation family's provider and stages it as the
        // native variant, with no output and the ordered-state effect of the program.
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let (output_types, program) = TracingContext::<TestIrValue, TestIrOperation>::trace(
            |inputs| {
                let [reference, update]: [Tracer<_>; 2] = inputs.try_into().unwrap();
                reference.add_update(&update)?;
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
                let () = reference_add_update %0 %1
                in (%0)
            "}
            .trim_end(),
        );
        assert_eq!(program.effects().classes(), EffectClasses::single(EffectClass::OrderedState));
    }
}
