//! Whole-array reference [`Operation`]s: allocation ([`NewReferenceOperation`]), reading ([`ReferenceReadOperation`]),
//! replacement ([`ReferenceSwapOperation`]), ordered additive update ([`ReferenceAddUpdateOperation`]), and consuming
//! finalization ([`FreezeReferenceOperation`]), together with the capability traits that value families implement to
//! execute them eagerly.

use std::borrow::Cow;
use std::fmt::Display;
use std::sync::LazyLock;

use ryft_macros::Parameter;

use crate::arrays::{ArrayIrType, ArrayType};
use crate::batching::{
    BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiationDriver, DifferentiationDual, DifferentiationError,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_transposable_operation};
use crate::operations::math::add::AddOperation;
use crate::parameters::Parameter;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    Effect, Effects, Operation, ProgramError, ProjectedValue, ReferenceAccessMode, ReferenceInputAccess,
    ReferenceOperationSemantics, ReferenceOutputSemantics, ReferenceType, RegionInterface, TypeError, Value,
};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`NewReferenceOperation`].
pub const NEW_REFERENCE_OPERATION_NAME: &str = "new_reference";

/// Canonical operation name for [`ReferenceReadOperation`].
pub const REFERENCE_READ_OPERATION_NAME: &str = "reference_read";

/// Canonical operation name for [`ReferenceSwapOperation`].
pub const REFERENCE_SWAP_OPERATION_NAME: &str = "reference_swap";

/// Canonical operation name for [`ReferenceAddUpdateOperation`].
pub const REFERENCE_ADD_UPDATE_OPERATION_NAME: &str = "reference_add_update";

/// Canonical operation name for [`FreezeReferenceOperation`].
pub const FREEZE_REFERENCE_OPERATION_NAME: &str = "freeze_reference";

/// Rejects dynamic referents until runtime extent preservation is explicitly represented and validated.
fn require_static_referent(operation: &str, referent: &ArrayType) -> Result<(), TypeError> {
    if referent.static_shape().is_some() {
        return Ok(());
    }
    Err(TypeError::invalid(format!(
        "`{operation}` does not support dynamically shaped reference referent type `{referent}`",
    )))
}

/// Creates a new reference initialized from this value.
pub trait NewReference<Output = Self>: Sized {
    /// Creates an independent reference whose initial state is this value.
    fn new_reference(&self) -> Result<Output, ProgramError>;
}

impl<V: Value<Type = ArrayIrType>> NewReference<V> for V
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<NewReferenceOperation>,
{
    fn new_reference(&self) -> Result<V, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(NewReferenceOperation, Vec::new(), std::slice::from_ref(self))?
            .remove(0))
    }
}

impl<V: Value<Type = ArrayIrType>> NewReference<V> for ProjectedValue<ArrayType, V>
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<NewReferenceOperation>,
{
    fn new_reference(&self) -> Result<V, ProgramError> {
        Ok(self
            .value()
            .dispatch_domain()
            .bind(NewReferenceOperation, Vec::new(), std::slice::from_ref(self.value()))?
            .remove(0))
    }
}

/// Reads an immutable snapshot from a reference value.
pub trait ReferenceRead<Output = Self>: Sized {
    /// Returns the reference's current value as an immutable snapshot.
    fn read(&self) -> Result<Output, ProgramError>;
}

impl<V: Value<Type = ArrayIrType>> ReferenceRead<V> for V
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceReadOperation>,
{
    fn read(&self) -> Result<V, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(ReferenceReadOperation, Vec::new(), std::slice::from_ref(self))?
            .remove(0))
    }
}

impl<V: Value<Type = ArrayIrType>> ReferenceRead<V> for ProjectedValue<ReferenceType<ArrayType>, V>
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceReadOperation>,
{
    fn read(&self) -> Result<V, ProgramError> {
        Ok(self
            .value()
            .dispatch_domain()
            .bind(ReferenceReadOperation, Vec::new(), std::slice::from_ref(self.value()))?
            .remove(0))
    }
}

/// Replaces the value stored by a reference in program order and returns its previous immutable snapshot.
pub trait ReferenceSwap<Replacement = Self, Output = Replacement>: Sized {
    /// Installs `replacement` in program order and returns the previously stored value.
    fn swap(&self, replacement: &Replacement) -> Result<Output, ProgramError>;

    /// Installs `replacement` in program order and discards the previously stored value.
    #[inline]
    fn write(&self, replacement: &Replacement) -> Result<(), ProgramError> {
        self.swap(replacement).map(drop)
    }
}

impl<V: Value<Type = ArrayIrType>> ReferenceSwap<V, V> for V
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceSwapOperation>,
{
    fn swap(&self, replacement: &V) -> Result<V, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(ReferenceSwapOperation, Vec::new(), &[self.clone(), replacement.clone()])?
            .remove(0))
    }
}

impl<V: Value<Type = ArrayIrType>> ReferenceSwap<ProjectedValue<ArrayType, V>, V>
    for ProjectedValue<ReferenceType<ArrayType>, V>
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceSwapOperation>,
{
    fn swap(&self, replacement: &ProjectedValue<ArrayType, V>) -> Result<V, ProgramError> {
        Ok(self
            .value()
            .dispatch_domain()
            .bind(ReferenceSwapOperation, Vec::new(), &[self.value().clone(), replacement.value().clone()])?
            .remove(0))
    }
}

/// Adds an update into the value stored by a reference in program order.
pub trait ReferenceAddUpdate<Update = Self>: Sized {
    /// Adds `update` to the stored value in program order.
    fn add_update(&self, update: &Update) -> Result<(), ProgramError>;
}

impl<V: Value<Type = ArrayIrType>> ReferenceAddUpdate<V> for V
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceAddUpdateOperation>,
{
    fn add_update(&self, update: &V) -> Result<(), ProgramError> {
        self.dispatch_domain()
            .bind(ReferenceAddUpdateOperation, Vec::new(), &[self.clone(), update.clone()])?;
        Ok(())
    }
}

impl<V: Value<Type = ArrayIrType>> ReferenceAddUpdate<ProjectedValue<ArrayType, V>>
    for ProjectedValue<ReferenceType<ArrayType>, V>
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<ReferenceAddUpdateOperation>,
{
    fn add_update(&self, update: &ProjectedValue<ArrayType, V>) -> Result<(), ProgramError> {
        self.value().dispatch_domain().bind(
            ReferenceAddUpdateOperation,
            Vec::new(),
            &[self.value().clone(), update.value().clone()],
        )?;
        Ok(())
    }
}

/// Consumes a reference, returning its final value and invalidating its complete alias family.
pub trait FreezeReference<Output = Self>: Sized {
    /// Returns the final stored value and invalidates this reference and all aliases.
    fn freeze(&self) -> Result<Output, ProgramError>;
}

impl<V: Value<Type = ArrayIrType>> FreezeReference<V> for V
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<FreezeReferenceOperation>,
{
    fn freeze(&self) -> Result<V, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(FreezeReferenceOperation, Vec::new(), std::slice::from_ref(self))?
            .remove(0))
    }
}

impl<V: Value<Type = ArrayIrType>> FreezeReference<V> for ProjectedValue<ReferenceType<ArrayType>, V>
where
    V::DispatchDomain: Context<Type = ArrayIrType>,
    <V::DispatchDomain as Domain>::Operation: From<FreezeReferenceOperation>,
{
    fn freeze(&self) -> Result<V, ProgramError> {
        Ok(self
            .value()
            .dispatch_domain()
            .bind(FreezeReferenceOperation, Vec::new(), std::slice::from_ref(self.value()))?
            .remove(0))
    }
}

// Reference semantics descriptors are constant per operation type. Sharing them through `LazyLock` statics lets the
// per-instruction program analysis read them through `Cow::Borrowed` without allocating.
static NEW_REFERENCE_OPERATION_SEMANTICS: LazyLock<ReferenceOperationSemantics> = LazyLock::new(|| {
    ReferenceOperationSemantics::new(vec![ReferenceOutputSemantics::NewRoot { output_index: 0 }], Vec::new())
});

static REFERENCE_READ_OPERATION_SEMANTICS: LazyLock<ReferenceOperationSemantics> = LazyLock::new(|| {
    ReferenceOperationSemantics::new(Vec::new(), vec![ReferenceInputAccess::new(0, ReferenceAccessMode::Read)])
});

static REFERENCE_SWAP_OPERATION_SEMANTICS: LazyLock<ReferenceOperationSemantics> = LazyLock::new(|| {
    ReferenceOperationSemantics::new(Vec::new(), vec![ReferenceInputAccess::new(0, ReferenceAccessMode::Write)])
});

static REFERENCE_ADD_UPDATE_OPERATION_SEMANTICS: LazyLock<ReferenceOperationSemantics> = LazyLock::new(|| {
    ReferenceOperationSemantics::new(Vec::new(), vec![ReferenceInputAccess::new(0, ReferenceAccessMode::Accumulate)])
});

static FREEZE_REFERENCE_OPERATION_SEMANTICS: LazyLock<ReferenceOperationSemantics> = LazyLock::new(|| {
    ReferenceOperationSemantics::new(Vec::new(), vec![ReferenceInputAccess::new(0, ReferenceAccessMode::Consume)])
});

/// Composite array-to-reference allocation operation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct NewReferenceOperation;

impl Display for NewReferenceOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for NewReferenceOperation {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        NEW_REFERENCE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let referent = <&ArrayType>::try_from(&input_types[0])?;
        require_static_referent(NEW_REFERENCE_OPERATION_NAME, referent)?;
        Ok(vec![ReferenceType::new(referent.clone()).into()])
    }

    #[inline]
    fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
        Cow::Borrowed(&NEW_REFERENCE_OPERATION_SEMANTICS)
    }

    #[inline]
    fn effects(&self) -> Effects {
        Effects::single(Effect::OrderedState)
    }
}

impl<C: Domain<Type = ArrayIrType, Value: NewReference<C::Value>>> InterpretableOperation<C> for NewReferenceOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        // Canonical eager type validation and holder construction are owned by the value-level capability.
        Ok(vec![inputs[0].new_reference()?])
    }
}

/// Composite reference-to-array snapshot read operation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct ReferenceReadOperation;

impl Display for ReferenceReadOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ReferenceReadOperation {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_READ_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let reference = <&ReferenceType<ArrayType>>::try_from(&input_types[0])?;
        require_static_referent(REFERENCE_READ_OPERATION_NAME, reference.referent())?;
        Ok(vec![reference.referent().clone().into()])
    }

    #[inline]
    fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
        Cow::Borrowed(&REFERENCE_READ_OPERATION_SEMANTICS)
    }

    #[inline]
    fn effects(&self) -> Effects {
        Effects::single(Effect::OrderedState)
    }
}

impl<C: Domain<Type = ArrayIrType, Value: ReferenceRead<C::Value>>> InterpretableOperation<C>
    for ReferenceReadOperation
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        // Canonical eager type validation and holder access are owned by the value-level capability.
        Ok(vec![inputs[0].read()?])
    }
}

/// Composite whole-array replacement operation that returns the value stored before the replacement.
///
/// The replacement operand must have exactly the reference's declared referent type. No broadcasting, data-type
/// promotion, layout change, sharding change, or memory change is implicit at this storage boundary.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct ReferenceSwapOperation;

impl Display for ReferenceSwapOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ReferenceSwapOperation {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_SWAP_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let reference = <&ReferenceType<ArrayType>>::try_from(&input_types[0])?;
        let replacement = <&ArrayType>::try_from(&input_types[1])?;
        require_static_referent(REFERENCE_SWAP_OPERATION_NAME, reference.referent())?;
        if replacement != reference.referent() {
            return Err(TypeError::invalid(format!(
                "`{REFERENCE_SWAP_OPERATION_NAME}` replacement type `{replacement}` must exactly match reference \
                 referent type `{}`",
                reference.referent(),
            )));
        }
        Ok(vec![reference.referent().clone().into()])
    }

    #[inline]
    fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
        Cow::Borrowed(&REFERENCE_SWAP_OPERATION_SEMANTICS)
    }

    #[inline]
    fn effects(&self) -> Effects {
        Effects::single(Effect::OrderedState)
    }
}

impl<C: Domain<Type = ArrayIrType, Value: ReferenceSwap<C::Value>>> InterpretableOperation<C>
    for ReferenceSwapOperation
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        Ok(vec![inputs[0].swap(&inputs[1])?])
    }
}

/// Composite ordered additive-update operation over a whole-array reference.
///
/// The update uses ordinary array addition type inference, but it is legal only when that addition produces exactly
/// the reference's declared referent type. The operation has no result; later reads observe the updated state.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct ReferenceAddUpdateOperation;

impl Display for ReferenceAddUpdateOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ReferenceAddUpdateOperation {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        REFERENCE_ADD_UPDATE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let reference = <&ReferenceType<ArrayType>>::try_from(&input_types[0])?;
        let update = <&ArrayType>::try_from(&input_types[1])?;
        require_static_referent(REFERENCE_ADD_UPDATE_OPERATION_NAME, reference.referent())?;
        let addition_result = AddOperation::<ArrayType>::new()
            .infer_output_types(&[reference.referent().clone(), update.clone()], &[])?
            .remove(0);
        if &addition_result != reference.referent() {
            return Err(TypeError::invalid(format!(
                "`{REFERENCE_ADD_UPDATE_OPERATION_NAME}` addition result type `{addition_result}` must exactly match \
                 reference referent type `{}`",
                reference.referent(),
            )));
        }
        Ok(Vec::new())
    }

    #[inline]
    fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
        Cow::Borrowed(&REFERENCE_ADD_UPDATE_OPERATION_SEMANTICS)
    }

    #[inline]
    fn effects(&self) -> Effects {
        Effects::single(Effect::OrderedState)
    }
}

impl<C: Domain<Type = ArrayIrType, Value: ReferenceAddUpdate<C::Value>>> InterpretableOperation<C>
    for ReferenceAddUpdateOperation
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        inputs[0].add_update(&inputs[1])?;
        Ok(Vec::new())
    }
}

/// Composite consuming operation that returns a reference's final whole-array value and invalidates its complete alias
/// family.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct FreezeReferenceOperation;

impl Display for FreezeReferenceOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for FreezeReferenceOperation {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        FREEZE_REFERENCE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let reference = <&ReferenceType<ArrayType>>::try_from(&input_types[0])?;
        require_static_referent(FREEZE_REFERENCE_OPERATION_NAME, reference.referent())?;
        Ok(vec![reference.referent().clone().into()])
    }

    #[inline]
    fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
        Cow::Borrowed(&FREEZE_REFERENCE_OPERATION_SEMANTICS)
    }

    #[inline]
    fn effects(&self) -> Effects {
        Effects::single(Effect::OrderedState)
    }
}

impl<C: Domain<Type = ArrayIrType, Value: FreezeReference<C::Value>>> InterpretableOperation<C>
    for FreezeReferenceOperation
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].freeze()?])
    }
}

macro_rules! impl_unsupported_reference_transforms {
    // One invocation lists the whole family so a new reference operation cannot miss a transform rejection.
    ($($operation:ty),+ $(,)?) => {
        $(impl_unsupported_reference_transforms!(@each $operation);)+
    };

    // Installs the same conservative transform rejections for one unresolved reference operation. Transposition
    // reuses the shared non-transposable diagnostic: reference operations never transpose directly because
    // reverse-mode differentiation always discharges them first (refer to `plan-references.md`).
    (@each $operation:ty) => {
        impl_non_transposable_operation!($operation);

        // The default `partially_evaluate` routes through `fold_or_residualize`, whose ordered-state gate produces
        // the same discharge diagnostic for every reference operation, so only the trait obligation is declared here.
        impl<C: Context<Type = ArrayIrType, Operation: From<$operation>>> PartiallyEvaluatableOperation<C>
            for $operation
        {
        }

        impl<C: Context<Type = ArrayIrType, Operation: From<$operation>>, P: BatchingPolicy<C>> BatchableOperation<C, P>
            for $operation
        {
            fn batch<D: BatchingDriver<C, P>>(
                &self,
                _context: &BatchingContext<C, P>,
                _driver: &D,
                _inputs: &[P::Batch],
            ) -> Result<BatchedOutputs<C, P>, BatchingError> {
                Err(BatchingError::UnsupportedOperation {
                    message: format!("`{}` must be discharged before batching", self.name()),
                })
            }
        }

        impl<C: Context<Type = ArrayIrType, Operation: From<$operation>>> DifferentiableOperation<C> for $operation {
            fn jvp<D: DifferentiationDriver<C>>(
                &self,
                _context: &C,
                _driver: &D,
                _inputs: &[DifferentiationDual<C::Value>],
            ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
                Err(ProgramError::UnsupportedOperation {
                    message: format!("`{}` must be discharged before differentiation", self.name()),
                }
                .into())
            }
        }
    };
}

impl_unsupported_reference_transforms!(
    NewReferenceOperation,
    ReferenceReadOperation,
    ReferenceSwapOperation,
    ReferenceAddUpdateOperation,
    FreezeReferenceOperation,
);

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatching, ArrayIrOperation, ArrayIrValue, DataType, Dimension, DimensionBounds,
        DimensionVariable, ReferenceAnalysisError, ReferenceRoot, Shape,
    };
    use crate::contexts::EagerContext;
    use crate::differentiation::{
        CustomJvpOperation, CustomVjpOperation, DifferentiationContext, DifferentiationDual, DifferentiationError,
        DifferentiationTracer, ForwardModeDifferentiate, Linearization, TransposableOperation,
    };
    use crate::macros::check_operation_type_inference;
    use crate::operations::control_flow::condition::ConditionOperation;
    use crate::operations::control_flow::scan::ScanOperation;
    use crate::operations::control_flow::r#while::WhileOperation;
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialValue};
    use crate::programs::{
        EmptyRegionDriver, InstructionId, Program, ProgramBuilder, Reference, RegionDriver, RegionRef, ValueProjection,
    };
    use crate::tracing::{Tracer, TracingContext};

    use super::*;

    type TestValue = ArrayIrValue<Array>;
    type TestOperation = ArrayIrOperation<Array>;
    type TestProgram = Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>;

    // Dynamically shaped referent fixture shared by the five per-operation type-inference rejections.
    fn dynamic_referent_type() -> ArrayType {
        ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("length", DimensionBounds::unbounded()))]),
        )
    }

    // Builds the single-input identity program over `r#type`.
    fn identity_program(r#type: &ArrayIrType) -> TestProgram {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(r#type.clone());
        builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap()
    }

    // Builds the `[primal, rule]` region pair for a custom-derivative operation whose rule region hides unresolved
    // state: the primal is the identity over `r#type`, and the rule allocates a reference from its primal input, reads
    // it back, and maps `[primal, tangent]` to `[read, tangent]`.
    fn custom_derivative_state_regions(r#type: &ArrayIrType) -> Vec<TestProgram> {
        let rule = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let input = builder.add_input(r#type.clone());
            let tangent = builder.add_input(r#type.clone());
            let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![input]).unwrap()[0];
            let output = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(
                    vec![output, tangent],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        vec![identity_program(r#type), rule]
    }

    #[test]
    fn test_new_reference() {
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        check_operation_type_inference!(
            operation = NewReferenceOperation,
            cases = [
                {
                    input_types = [array_type.clone().into()],
                    output_types = [ReferenceType::new(array_type).into()],
                },
                {
                    input_types = [dynamic_referent_type().into()],
                    error = "`new_reference` does not support dynamically shaped reference referent type \
                             `f32[length]`",
                },
            ],
        );
        assert_eq!(NewReferenceOperation.effects(), Effects::single(Effect::OrderedState));
        assert_eq!(
            *NewReferenceOperation.reference_semantics(),
            ReferenceOperationSemantics::new(vec![ReferenceOutputSemantics::NewRoot { output_index: 0 }], Vec::new()),
        );
        assert_eq!(NewReferenceOperation.to_string(), NEW_REFERENCE_OPERATION_NAME);
    }

    #[test]
    fn test_reference_read() {
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        check_operation_type_inference!(
            operation = ReferenceReadOperation,
            cases = [
                {
                    input_types = [ReferenceType::new(array_type.clone()).into()],
                    output_types = [array_type.into()],
                },
                {
                    input_types = [ReferenceType::new(dynamic_referent_type()).into()],
                    error = "`reference_read` does not support dynamically shaped reference referent type \
                             `f32[length]`",
                },
            ],
        );
        assert_eq!(ReferenceReadOperation.effects(), Effects::single(Effect::OrderedState));
        assert_eq!(
            *ReferenceReadOperation.reference_semantics(),
            ReferenceOperationSemantics::new(Vec::new(), vec![ReferenceInputAccess::new(0, ReferenceAccessMode::Read)]),
        );
        assert_eq!(ReferenceReadOperation.to_string(), REFERENCE_READ_OPERATION_NAME);
    }

    #[test]
    fn test_reference_swap() {
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let dynamic_type = dynamic_referent_type();
        check_operation_type_inference!(
            operation = ReferenceSwapOperation,
            cases = [
                {
                    input_types = [ReferenceType::new(array_type.clone()).into(), array_type.clone().into()],
                    output_types = [array_type.clone().into()],
                },
                {
                    input_types = [
                        ReferenceType::new(array_type.clone()).into(),
                        ArrayType::new_static(DataType::F32, [3]).into(),
                    ],
                    error = "`reference_swap` replacement type `f32[3]` must exactly match reference referent type \
                             `f32[2]`",
                },
                {
                    input_types = [array_type.clone().into(), array_type.into()],
                    error = "expected reference type but got array type",
                },
                {
                    input_types = [ReferenceType::new(dynamic_type.clone()).into(), dynamic_type.into()],
                    error = "`reference_swap` does not support dynamically shaped reference referent type \
                             `f32[length]`",
                },
            ],
        );
        assert_eq!(ReferenceSwapOperation.effects(), Effects::single(Effect::OrderedState));
        assert_eq!(
            *ReferenceSwapOperation.reference_semantics(),
            ReferenceOperationSemantics::new(
                Vec::new(),
                vec![ReferenceInputAccess::new(0, ReferenceAccessMode::Write)],
            ),
        );
        assert_eq!(ReferenceSwapOperation.to_string(), REFERENCE_SWAP_OPERATION_NAME);
    }

    #[test]
    fn test_reference_add_update() {
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let dynamic_type = dynamic_referent_type();
        check_operation_type_inference!(
            operation = ReferenceAddUpdateOperation,
            cases = [
                {
                    input_types = [
                        ReferenceType::new(array_type.clone()).into(),
                        ArrayType::scalar(DataType::F32).into(),
                    ],
                    output_types = [],
                },
                {
                    input_types = [
                        ReferenceType::new(array_type.clone()).into(),
                        array_type.clone().with_data_type(DataType::F64).into(),
                    ],
                    error = "`reference_add_update` addition result type `f64[2]` must exactly match reference \
                             referent type `f32[2]`",
                },
                {
                    input_types = [
                        ReferenceType::new(array_type.clone()).into(),
                        ReferenceType::new(array_type).into(),
                    ],
                    error = "expected array type but got reference type",
                },
                {
                    input_types = [ReferenceType::new(dynamic_type.clone()).into(), dynamic_type.into()],
                    error = "`reference_add_update` does not support dynamically shaped reference referent type \
                             `f32[length]`",
                },
            ],
        );
        assert_eq!(ReferenceAddUpdateOperation.effects(), Effects::single(Effect::OrderedState));
        assert_eq!(
            *ReferenceAddUpdateOperation.reference_semantics(),
            ReferenceOperationSemantics::new(
                Vec::new(),
                vec![ReferenceInputAccess::new(0, ReferenceAccessMode::Accumulate)],
            ),
        );
        assert_eq!(ReferenceAddUpdateOperation.to_string(), REFERENCE_ADD_UPDATE_OPERATION_NAME);
    }

    #[test]
    fn test_freeze_reference() {
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        check_operation_type_inference!(
            operation = FreezeReferenceOperation,
            cases = [
                {
                    input_types = [ReferenceType::new(array_type.clone()).into()],
                    output_types = [array_type.clone().into()],
                },
                {
                    input_types = [array_type.into()],
                    error = "expected reference type but got array type",
                },
                {
                    input_types = [ReferenceType::new(dynamic_referent_type()).into()],
                    error = "`freeze_reference` does not support dynamically shaped reference referent type \
                             `f32[length]`",
                },
            ],
        );
        assert_eq!(FreezeReferenceOperation.effects(), Effects::single(Effect::OrderedState));
        assert_eq!(
            *FreezeReferenceOperation.reference_semantics(),
            ReferenceOperationSemantics::new(
                Vec::new(),
                vec![ReferenceInputAccess::new(0, ReferenceAccessMode::Consume)],
            ),
        );
        assert_eq!(FreezeReferenceOperation.to_string(), FREEZE_REFERENCE_OPERATION_NAME);
    }

    #[test]
    fn test_mutating_reference_operations_stage_as_composite_native_variants() {
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let reference_type = ReferenceType::new(array_type.clone());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let reference = builder.add_input(reference_type.into());
        let update = builder.add_input(array_type.into());
        let old = builder.add_instruction(ReferenceSwapOperation, Vec::new(), vec![reference, update]).unwrap()[0];
        builder.add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![reference, update]).unwrap();
        let frozen = builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![reference]).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![old, frozen],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        // The rendering pins both the staged instruction sequence and the composite-to-native variant selection that
        // each `From` conversion performs while the instructions are added.
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ref<f32[2]>, %1:f32[2] .
                let %2:f32[2] = reference_swap %0 %1
                    reference_add_update %0 %1
                    %3:f32[2] = freeze_reference %0
                in (%2, %3)
            "}
            .trim_end(),
        );
        assert_eq!(program.effects(), Effects::single(Effect::OrderedState));
    }

    #[test]
    fn test_mutating_reference_operations_execute_eagerly_and_reject_transforms_until_discharge() {
        type TestContext = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let context = TestContext::new();
        let reference = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0])).new_reference().unwrap();
        let update = ArrayIrValue::Array(Array::vector(vec![3.0_f32, 4.0]));
        assert_eq!(
            context.bind(ReferenceSwapOperation, Vec::new(), &[reference.clone(), update.clone()]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0]))]),
        );
        assert_eq!(context.bind(ReferenceAddUpdateOperation, Vec::new(), &[reference.clone(), update]), Ok(Vec::new()));
        assert_eq!(
            context.bind(FreezeReferenceOperation, Vec::new(), std::slice::from_ref(&reference)),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![6.0_f32, 8.0]))]),
        );

        // Eager semantics do not make an unresolved state operation safe for any generic transform. Every reference
        // operation shares one generated rejection body per transform, so one representative operation covers the
        // whole family (the remaining discharge diagnostics are checked at program scope by
        // `test_reference_operations_reject_transforms_until_discharge`).
        let partial_context = PartialEvaluationContext::new(TestContext::new());
        assert!(matches!(
            <ReferenceSwapOperation as PartiallyEvaluatableOperation<TestContext>>::partially_evaluate(
                &ReferenceSwapOperation,
                &partial_context,
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "`reference_swap` must be discharged before partial evaluation",
        ));

        let batching_context =
            BatchingContext::<_, ArrayIrBatching>::new(TestContext::new(), ArrayIrValue::Array(Array::scalar(2_i64)));
        assert!(matches!(
            <ReferenceSwapOperation as BatchableOperation<_, ArrayIrBatching>>::batch(
                &ReferenceSwapOperation,
                &batching_context,
                &EmptyRegionDriver,
                &[],
            ),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "`reference_swap` must be discharged before batching",
        ));

        assert!(matches!(
            <ReferenceSwapOperation as DifferentiableOperation<TestContext>>::jvp(
                &ReferenceSwapOperation,
                &TestContext::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`reference_swap` must be discharged before differentiation",
        ));

        // The transposition rule is checked directly rather than through `check_operation_transposition!(@rejected)`
        // because a one-instruction reference program never reaches the operation rule: program-level transposition
        // rejects the effectful linear instruction first, so the macro's shared diagnostic cannot hold here.
        assert!(matches!(
            <ReferenceSwapOperation as TransposableOperation<ArrayIrValue<Array>, ArrayIrOperation<Array>>>::transpose(
                &ReferenceSwapOperation,
                &mut TracingContext::new(),
                &EmptyRegionDriver,
                &[],
                &[],
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `reference_swap` is not transposable",
        ));
    }

    #[test]
    fn test_reference_allocation_and_read_stage_as_composite_native_variants() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let (output_type, program) = TestContext::trace(
            |input| ReferenceRead::read(&input.new_reference()?),
            ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:ref<f32[]> = new_reference %0
                    %2:f32[] = reference_read %1
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(program.effects(), Effects::single(Effect::OrderedState));
    }

    #[test]
    fn test_differentiation_context_rejects_intrinsic_state_before_zero_tangent_fast_path() {
        type TestContext = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let context = DifferentiationContext::new(TestContext::new());
        let input = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(ArrayIrValue::Array(Array::scalar(1.0_f32))).unwrap(),
            context.clone(),
        );

        assert!(matches!(
            context.bind(NewReferenceOperation, Vec::new(), &[input]),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "`new_reference` must be discharged before differentiation",
        ));
    }

    #[test]
    fn test_projected_reference_capabilities_bind_through_the_composite_parent() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        type TestTracer = Tracer<TestContext>;

        let array_type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2]));
        let (_, program) = TestContext::trace(
            |inputs| {
                let [initial, replacement, update]: [TestTracer; 3] = inputs.try_into().unwrap();
                let initial = <TestTracer as ValueProjection<ArrayType>>::into_projected(initial)?;
                let replacement = <TestTracer as ValueProjection<ArrayType>>::into_projected(replacement)?;
                let update = <TestTracer as ValueProjection<ArrayType>>::into_projected(update)?;
                let reference = initial.new_reference()?;
                let reference = <TestTracer as ValueProjection<ReferenceType<ArrayType>>>::into_projected(reference)?;
                reference.write(&replacement)?;
                reference.add_update(&update)?;
                Ok(vec![reference.freeze()?])
            },
            vec![array_type.clone(), array_type.clone(), array_type],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[2], %2:f32[2] .
                let %3:ref<f32[2]> = new_reference %0
                    %4:f32[2] = reference_swap %3 %1
                    reference_add_update %3 %2
                    %5:f32[2] = freeze_reference %3
                in (%5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_reference_program_eager_and_staged_semantics_are_equivalent() {
        type TestContext = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let inputs = (
            ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0])),
            ArrayIrValue::Array(Array::vector(vec![3.0_f32, 4.0])),
            ArrayIrValue::Array(Array::vector(vec![5.0_f32, 6.0])),
            ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0])),
        );
        let (eager_outputs, program) = TestContext::new()
            .interpret_and_trace(
                |(initial, replacement, written, update)| {
                    let reference = initial.new_reference()?;
                    let snapshot = reference.read()?;
                    let old = reference.swap(&replacement)?;
                    reference.write(&written)?;
                    reference.add_update(&update)?;
                    let final_value = reference.freeze()?;
                    Ok((snapshot, old, final_value))
                },
                inputs.clone(),
            )
            .unwrap();
        let expected = (
            ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0])),
            ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0])),
            ArrayIrValue::Array(Array::vector(vec![6.0_f32, 8.0])),
        );
        assert_eq!(eager_outputs, expected);
        program.analyze_references(0).unwrap();
        assert_eq!(program.interpret(inputs), Ok(expected));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[2], %2:f32[2], %3:f32[2] .
                let %4:ref<f32[2]> = new_reference %0
                    %5:f32[2] = reference_read %4
                    %6:f32[2] = reference_swap %4 %1
                    %7:f32[2] = reference_swap %4 %2
                    reference_add_update %4 %3
                    %8:f32[2] = freeze_reference %4
                in (%5, %6, %8)
            "}
            .trim_end(),
        );
        assert_eq!(program.effects(), Effects::single(Effect::OrderedState));
    }

    #[test]
    fn test_reference_program_preflight_rejects_before_external_mutation() {
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let external = builder.add_input(ReferenceType::new(array_type.clone()).into());
        let replacement = builder.add_input(array_type.into());
        builder.add_instruction(ReferenceSwapOperation, Vec::new(), vec![external, replacement]).unwrap();
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![external],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();

        let initial = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0]));
        let reference = initial.new_reference().unwrap();
        assert!(matches!(
            program.interpret(vec![
                reference.clone(),
                ArrayIrValue::Array(Array::vector(vec![3.0_f32, 4.0])),
            ]),
            Err(error)
                if error.downcast_custom::<ReferenceAnalysisError>()
                    == Some(&ReferenceAnalysisError::ReferenceOutput {
                        region: program.entry(),
                        output_index: 0,
                        root: ReferenceRoot::RegionInput { region: program.entry(), input_index: 0 },
                    }),
        ));
        assert_eq!(reference.read(), Ok(initial));

        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let external = builder.add_input(ReferenceType::new(ArrayType::new_static(DataType::F32, [2])).into());
        let replacement = builder.add_input(ArrayType::new_static(DataType::F32, [2]).into());
        let old = builder.add_instruction(ReferenceSwapOperation, Vec::new(), vec![external, replacement]).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![old],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        program.analyze_references(0).unwrap();
        assert_eq!(
            program.entry_region_ref().interpret_in_context(
                &EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
                vec![reference.clone(), ArrayIrValue::Array(Array::vector(vec![5.0_f32, 6.0]))],
                None,
            ),
            Err(ProgramError::UnsupportedOperation {
                message: "program replay of external reference public input 0 is not supported before external \
                          holder runtime integration"
                    .to_string(),
            }),
        );
        assert_eq!(reference.read(), Ok(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0]))));
        assert_eq!(
            program.interpret(vec![reference.clone(), ArrayIrValue::Array(Array::vector(vec![7.0_f32, 8.0])),]),
            Err(ProgramError::UnsupportedOperation {
                message: "program replay of external reference public input 0 is not supported before external \
                          holder runtime integration"
                    .to_string(),
            }),
        );
        assert_eq!(reference.read(), Ok(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0]))));
    }

    #[test]
    fn test_nested_reference_program_implicitly_discards_local_roots() {
        let array_type = ArrayType::new_static(DataType::F32, [2]);

        let mut true_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let true_input = true_builder.add_input(array_type.clone().into());
        let true_reference =
            true_builder.add_instruction(NewReferenceOperation, Vec::new(), vec![true_input]).unwrap()[0];
        let true_output =
            true_builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![true_reference]).unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![true_output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let mut false_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let false_input = false_builder.add_input(array_type.clone().into());
        let false_branch = false_builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![false_input],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(array_type.into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        program.analyze_references(0).unwrap();

        let value = ArrayIrValue::Array(Array::vector(vec![2.0_f32, 4.0]));
        assert_eq!(
            program.interpret(vec![ArrayIrValue::Array(Array::scalar(true)), value.clone()]),
            Ok(vec![value.clone()]),
        );
        assert_eq!(program.interpret(vec![ArrayIrValue::Array(Array::scalar(false)), value.clone()]), Ok(vec![value]),);
    }

    #[test]
    fn test_checked_root_replay_forwards_local_references_into_condition_branches() {
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let reference_type = ReferenceType::new(array_type.clone());
        let build_branch = || {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            let output = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let true_branch = build_branch();
        let false_branch = build_branch();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(array_type.into());
        let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![initial]).unwrap()[0];
        let output = builder
            .add_instruction(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, reference],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        program.analyze_references(0).unwrap();

        let context = EagerContext::<TestValue, TestOperation>::new();
        let value = TestValue::Array(Array::vector(vec![2.0_f32, 4.0]));
        assert_eq!(
            program.interpret(vec![TestValue::Array(Array::scalar(true)), value.clone()]),
            Ok(vec![value.clone()]),
        );
        assert_eq!(
            program.interpret(vec![TestValue::Array(Array::scalar(false)), value.clone()]),
            Ok(vec![value.clone()]),
        );
        assert_eq!(
            program.entry_region_ref().interpret_in_context(
                &context,
                vec![TestValue::Array(Array::scalar(true)), value.clone()],
                None,
            ),
            Ok(vec![value.clone()]),
        );
        assert_eq!(
            program.entry_region_ref().interpret_in_context(
                &context,
                vec![TestValue::Array(Array::scalar(false)), value.clone()],
                None,
            ),
            Ok(vec![value]),
        );
    }

    #[test]
    fn test_direct_eager_bind_validates_every_attached_region_before_selection() {
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let valid_branch = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let input = builder.add_input(array_type.clone().into());
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![input], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let invalid_branch = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let input = builder.add_input(array_type.clone().into());
            let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![input]).unwrap()[0];
            builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![reference]).unwrap();
            builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![input], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let invalid_region = invalid_branch.entry();

        let context = EagerContext::<TestValue, TestOperation>::new();
        let error = context
            .bind(
                ConditionOperation::new(),
                vec![valid_branch, invalid_branch],
                &[TestValue::Array(Array::scalar(true)), TestValue::Array(Array::vector(vec![1.0_f32, 2.0]))],
            )
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::UseAfterConsume {
                instruction: InstructionId::new(invalid_region, 2),
                operation: REFERENCE_READ_OPERATION_NAME.to_string(),
                input_index: 0,
                root: ReferenceRoot::Allocation { instruction: InstructionId::new(invalid_region, 0), output_index: 0 },
            }),
        );
    }

    #[test]
    fn test_while_recreates_and_discards_local_roots_per_invocation() {
        type Values = Vec<ArrayIrValue<Array>>;

        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let boolean_type = ArrayType::scalar(DataType::Boolean);

        // The condition executes twice and the body once. Every invocation creates a fresh local root whose read
        // snapshot remains inside that invocation, so releasing each region environment discards the holder.
        let condition = {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let state = builder.add_input(array_type.clone().into());
            let predicate = builder.add_input(boolean_type.clone().into());
            let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![state]).unwrap()[0];
            builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap();
            builder.build::<Values, Values>(vec![predicate], vec![Placeholder; 2], vec![Placeholder]).unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let state = builder.add_input(array_type.clone().into());
            builder.add_input(boolean_type.clone().into());
            let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![state]).unwrap()[0];
            let update = builder.add_constant(ArrayIrValue::Array(Array::scalar(1.0_f32)));
            builder.add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![reference, update]).unwrap();
            let state = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
            let done = builder.add_constant(ArrayIrValue::Array(Array::scalar(false)));
            builder
                .build::<Values, Values>(vec![state, done], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(body.entry_region_ref());
        let state = builder.add_input(array_type.into());
        let predicate = builder.add_input(boolean_type.clone().into());
        let outputs = builder
            .add_instruction(
                ArrayIrOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                vec![state, predicate],
            )
            .unwrap()
            .to_vec();
        let program = builder.build::<Values, Values>(outputs, vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();
        program.analyze_references(0).unwrap();
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0])),
                ArrayIrValue::Array(Array::scalar(true)),
            ]),
            Ok(
                vec![ArrayIrValue::Array(Array::vector(vec![2.0_f32, 3.0])), ArrayIrValue::Array(Array::scalar(false)),]
            ),
        );
    }

    #[test]
    fn test_scan_recreates_and_discards_local_roots_per_iteration() {
        type Values = Vec<ArrayIrValue<Array>>;

        let scalar_type = ArrayType::scalar(DataType::F32);
        let stacked_type = ArrayType::new_static(DataType::F32, [3]);
        let body = {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let carry = builder.add_input(scalar_type.clone().into());
            let item = builder.add_input(scalar_type.clone().into());
            let reference = builder.add_instruction(NewReferenceOperation, Vec::new(), vec![carry]).unwrap()[0];
            builder.add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![reference, item]).unwrap();
            let next = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
            builder
                .build::<Values, Values>(vec![next, next], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let carry = builder.add_input(scalar_type.into());
        let items = builder.add_input(stacked_type.into());
        let outputs = builder
            .add_instruction(ScanOperation::new(1, 3), vec![body_region], vec![carry, items])
            .unwrap()
            .to_vec();
        let program = builder.build::<Values, Values>(outputs, vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();

        program.analyze_references(0).unwrap();
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Array(Array::scalar(1.0_f32)),
                ArrayIrValue::Array(Array::vector(vec![1.0_f32, 3.0, 4.0])),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::scalar(9.0_f32)),
                ArrayIrValue::Array(Array::vector(vec![2.0_f32, 5.0, 9.0])),
            ]),
        );
    }

    #[test]
    fn test_reference_operations_reject_transforms_until_discharge() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let input_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let (_, program) =
            TestContext::trace(|input| ReferenceRead::read(&input.new_reference()?), input_type.clone()).unwrap();
        let program = program.into_flat_program();

        // Partial evaluation must never execute, fold, or split an unresolved state chain, regardless of whether the
        // reference-producing input is known or unknown.
        for input in [
            PartialValue::Unknown(input_type.clone()),
            PartialValue::Known(ArrayIrValue::Array(Array::scalar(1.0_f32))),
        ] {
            assert_eq!(
                program.partially_evaluate(std::slice::from_ref(&input)).map(|_| ()),
                Err(ProgramError::UnsupportedOperation {
                    message: "`new_reference` must be discharged before partial evaluation".to_string(),
                }),
            );
        }

        // Structural differentiation (direct `Program::jvp` and the linearization built on it) rejects at the fused
        // replay's entry, before its all-zero shortcut could stage any primal instruction.
        for result in [program.jvp().map(|_| ()), program.linearize().map(|_| ())] {
            assert!(matches!(
                result,
                Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                    if message
                        == "program carries unresolved state and must be discharged before differentiation",
            ));
        }

        // A lifted reference must not ride through batching as a replicated batch that could cross the output
        // boundary unchanged, so `BatchingContext::lift` routes through the policy's checked batch constructor.
        let batching_context = BatchingContext::<_, ArrayIrBatching>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Array(Array::scalar(2_i64)),
        );
        let error = batching_context
            .lift(ArrayIrValue::Reference(Reference::new(Array::scalar(1.0_f32))))
            .map(|_| ())
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<BatchingError>(),
            Some(&BatchingError::UnsupportedOperation {
                message: "references must be discharged before batching".to_string(),
            }),
        );
    }

    #[test]
    fn test_custom_derivative_rules_reject_unresolved_state() {
        // Custom derivative rule regions are interpreted directly rather than routed through the per-operation
        // transform rejections, so a rule whose body touches unresolved state must be rejected before any
        // of it executes during differentiation.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let regions = custom_derivative_state_regions(&scalar_type);

        // The central `DifferentiationContext::bind` guard rejects the state before the rule is ever consulted (the
        // operation-local rule-region guards remain as defense in depth behind it).
        let result = EagerContext::<TestValue, TestOperation>::new().jvp(
            {
                let regions = regions.clone();
                move |input: DifferentiationTracer<EagerContext<TestValue, TestOperation>>, ()| {
                    let operation = ArrayIrOperation::CustomJvp(CustomJvpOperation::new());
                    Ok(input.context().bind(operation, regions.clone(), std::slice::from_ref(&input))?.remove(0))
                }
            },
            ArrayIrValue::Array(Array::scalar(1.0_f32)),
            ArrayIrValue::Array(Array::scalar(1.0_f32)),
            (),
        );
        assert!(matches!(
            result,
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`custom_jvp` carries unresolved state in an attached region and must be \
                    discharged before differentiation",
        ));

        // The all-zero tangent fast path binds the primal directly without reaching any operation rule, so state must
        // be rejected on that path too: a lifted constant input carries a structural-zero tangent.
        let result = EagerContext::<TestValue, TestOperation>::new().jvp(
            move |input: DifferentiationTracer<EagerContext<TestValue, TestOperation>>, ()| {
                let lifted = input.context().lift(ArrayIrValue::Array(Array::scalar(1.0_f32)))?;
                let operation = ArrayIrOperation::CustomJvp(CustomJvpOperation::new());
                Ok(input.context().bind(operation, regions.clone(), std::slice::from_ref(&lifted))?.remove(0))
            },
            ArrayIrValue::Array(Array::scalar(1.0_f32)),
            ArrayIrValue::Array(Array::scalar(1.0_f32)),
            (),
        );
        assert!(matches!(
            result,
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`custom_jvp` carries unresolved state in an attached region and must be \
                    discharged before differentiation",
        ));
    }

    #[test]
    fn test_operation_local_custom_derivative_guards_reject_state_in_nested_dormant_rules() {
        type TestContext = EagerContext<TestValue, TestOperation>;

        struct TestDifferentiationDriver {
            programs: Vec<TestProgram>,
        }

        impl RegionDriver<TestValue, TestOperation> for TestDifferentiationDriver {
            fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, TestValue, TestOperation>>
            where
                TestValue: 'r,
                TestOperation: 'r,
            {
                self.programs.iter().map(Program::entry_region_ref)
            }
        }

        impl DifferentiationDriver<TestContext> for TestDifferentiationDriver {
            fn jvp_program(
                &self,
                _region: RegionRef<'_, TestValue, TestOperation>,
            ) -> Result<Arc<TestProgram>, DifferentiationError> {
                unreachable!("the operation-local state guard must reject before recursive differentiation")
            }

            fn linearize_program(
                &self,
                _region: RegionRef<'_, TestValue, TestOperation>,
            ) -> Result<Linearization<TestValue, TestOperation>, DifferentiationError> {
                unreachable!("the operation-local state guard must reject before recursive linearization")
            }

            fn jvp_operation(
                &self,
                _operation: &TestOperation,
                _programs: Vec<TestProgram>,
                _context: &TestContext,
                _inputs: &[DifferentiationDual<TestValue>],
            ) -> Result<Vec<DifferentiationDual<TestValue>>, DifferentiationError> {
                unreachable!("the operation-local state guard must reject before recursive differentiation")
            }
        }

        fn nested_rule_state_program(scalar_type: &ArrayIrType, include_tangent_output: bool) -> TestProgram {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let regions = custom_derivative_state_regions(scalar_type)
                .iter()
                .map(|region| builder.import_region(region.entry_region_ref()))
                .collect::<Vec<_>>();
            let input = builder.add_input(scalar_type.clone());
            let tangent = include_tangent_output.then(|| builder.add_input(scalar_type.clone()));
            let output =
                builder.add_instruction(CustomJvpOperation::<ArrayIrType>::new(), regions, vec![input]).unwrap()[0];
            let mut outputs = vec![output];
            if let Some(tangent) = tangent {
                outputs.push(tangent);
            }
            let input_count = usize::from(include_tangent_output) + 1;
            let output_count = outputs.len();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(
                    outputs,
                    vec![Placeholder; input_count],
                    vec![Placeholder; output_count],
                )
                .unwrap()
        }

        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let context = TestContext::new();
        let input = DifferentiationDual::new(
            TestValue::Array(Array::scalar(1.0_f32)),
            TestValue::Array(Array::scalar(1.0_f32)),
        )
        .unwrap();

        // The outer JVP rule is pure under ordinary effect aggregation because its nested custom-JVP call keeps the
        // state in a dormant rule. Direct operation-rule invocation must nevertheless inspect the complete closure.
        let primal = identity_program(&scalar_type);
        let jvp = nested_rule_state_program(&scalar_type, true);
        assert!(jvp.effects().is_pure());
        assert!(jvp.entry_region_ref().contains_effect_in_closure(Effect::OrderedState));
        let driver = TestDifferentiationDriver { programs: vec![primal, jvp] };
        assert!(matches!(
            CustomJvpOperation::<ArrayIrType>::new().jvp(&context, &driver, std::slice::from_ref(&input)),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`custom_jvp` rule regions must not contain unresolved state",
        ));

        // Custom VJP retains both its directly interpreted forward rule and its transpose-time backward rule in the
        // differentiated program. Exercise each position independently so neither can hide state in a dormant child.
        let primal = identity_program(&scalar_type);
        let forward = nested_rule_state_program(&scalar_type, false);
        let backward = identity_program(&scalar_type);
        assert!(forward.effects().is_pure());
        let driver = TestDifferentiationDriver { programs: vec![primal, forward, backward] };
        assert!(matches!(
            CustomVjpOperation::<ArrayIrType>::new().jvp(&context, &driver, std::slice::from_ref(&input)),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`custom_vjp` rule regions must not contain unresolved state",
        ));

        let primal = identity_program(&scalar_type);
        let forward = identity_program(&scalar_type);
        let backward = nested_rule_state_program(&scalar_type, false);
        assert!(backward.effects().is_pure());
        let driver = TestDifferentiationDriver { programs: vec![primal, forward, backward] };
        assert!(matches!(
            CustomVjpOperation::<ArrayIrType>::new().jvp(&context, &driver, std::slice::from_ref(&input)),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`custom_vjp` rule regions must not contain unresolved state",
        ));
    }

    #[test]
    fn test_program_jvp_rejects_reference_state_hidden_in_dormant_rule_regions() {
        // Sealed program effects deliberately exclude dormant rule regions, so a program whose only unresolved state
        // lives inside a custom-derivative rule is `Effects::PURE`. The fused JVP entry must therefore scan the whole
        // attached-region closure rather than trusting program effects.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let wrapped = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let regions = custom_derivative_state_regions(&scalar_type)
                .iter()
                .map(|region| builder.import_region(region.entry_region_ref()))
                .collect::<Vec<_>>();
            let input = builder.add_input(scalar_type.clone());
            let outputs = builder
                .add_instruction(ArrayIrOperation::CustomJvp(CustomJvpOperation::new()), regions, vec![input])
                .unwrap()
                .to_vec();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        assert!(wrapped.effects().is_pure());
        assert!(matches!(
            wrapped.jvp(),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message
                    == "program carries unresolved state and must be discharged before differentiation",
        ));
    }
}
