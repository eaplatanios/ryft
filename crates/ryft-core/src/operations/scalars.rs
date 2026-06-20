use std::fmt::Display;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::differentiation::{Cotangent, TransposableOperation};
use crate::domains::Domain;
use crate::operations::arithmetic::{
    AddOperation, DivOperation, MulOperation, NegOperation, ScaleOperation, SubOperation,
};
use crate::operations::compare::{Compare, CompareOperation};
use crate::operations::constants::{
    ConstantOperation, OneLike, OneLikeOperation, OneOperation, ZeroLike, ZeroLikeOperation, ZeroOperation,
};
use crate::operations::control_flow::{Select, SelectCondition, SelectOperation};
use crate::operations::differentiation::StopGradientOperation;
use crate::operations::trigonometric::{CosOperation, SinOperation};
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::Parameterized;
use crate::programs::{ProgramError, Value};
use crate::tracing::AbstractTracingContext;
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, TangentContext};
use crate::tracing_v2::operations::bounds::{SupportsLinearScalarOperation, SupportsTrigonometricOperations};
use crate::tracing_v2::operations::custom_derivatives::{
    CustomJvpOperation, CustomVjpCallOperation, CustomVjpOperation,
};
use crate::tracing_v2::operations::select::LinearSelectOperation;
use crate::tracing_v2::rematerialization::{MaybeRematerializationName, RematerializationNameOperation};
use crate::tracing_v2::{
    CapturedFactor, DifferentiableOperation, DifferentiationContext, ProgramLinearizableOperation,
};
use crate::types::{DataType, TypeError};

// TODO(eaplatanios): Review this file.

/// Closed scalar operation type for ordinary staged scalar programs.
///
/// [`ScalarOperation`] is intentionally limited to operations that are valid for scalar [`DataType`] metadata.
/// Array-only primitives such as reshaping and matrix multiplication remain available as standalone operations and
/// through array-based backends, but they are not variants of this enum.
#[derive(Clone, Debug)]
pub enum ScalarOperation<V: Value<DataType>> {
    /// Typed scalar zero.
    Zero(ZeroOperation<DataType>),

    /// Exemplar-derived scalar zero.
    ZeroLike(ZeroLikeOperation),

    /// Typed scalar one.
    One(OneOperation<DataType>),

    /// Exemplar-derived scalar one.
    OneLike(OneLikeOperation),

    /// Typed scalar constant.
    Constant(ConstantOperation<DataType, V>),

    /// Scalar negation.
    Neg(NegOperation),

    /// Scalar addition.
    Add(AddOperation),

    /// Scalar subtraction.
    Sub(SubOperation),

    /// Scaling by a captured scalar factor.
    Scale(ScaleOperation<DataType, V>),

    /// Scalar multiplication.
    Mul(MulOperation),

    /// Scalar division.
    Div(DivOperation),

    /// Scalar sine.
    Sin(SinOperation),

    /// Scalar cosine.
    Cos(CosOperation),

    /// Scalar comparison.
    Compare(CompareOperation),

    /// Scalar selection on a Boolean condition.
    Select(SelectOperation),

    /// Gradient barrier that passes its primal through unchanged.
    StopGradient(StopGradientOperation),

    /// Rematerialization tag attached to a scalar value.
    RematerializationName(RematerializationNameOperation),

    /// User-supplied `custom_jvp` operation with a closed scalar body.
    CustomJvp(Box<CustomJvpOperation<V, Self, DataType>>),

    /// User-supplied `custom_vjp` operation with a closed scalar body.
    CustomVjp(Box<CustomVjpOperation<V, Self, DataType>>),
}

impl<V: Value<DataType>> Operation<DataType> for ScalarOperation<V> {
    // Several variant payloads (the elementwise arithmetic and trigonometric operations) implement
    // [`Operation`](crate::operations::Operation) for both [`DataType`] and `ArrayType`, so plain method-call syntax
    // cannot infer the type parameter here. The arms therefore disambiguate to `Operation<DataType>` explicitly.
    fn name(&self) -> &'static str {
        match self {
            Self::Zero(operation) => <ZeroOperation<DataType> as Operation<DataType>>::name(operation),
            Self::ZeroLike(operation) => <ZeroLikeOperation as Operation<DataType>>::name(operation),
            Self::One(operation) => <OneOperation<DataType> as Operation<DataType>>::name(operation),
            Self::OneLike(operation) => <OneLikeOperation as Operation<DataType>>::name(operation),
            Self::Constant(operation) => <ConstantOperation<DataType, V> as Operation<DataType>>::name(operation),
            Self::Neg(operation) => <NegOperation as Operation<DataType>>::name(operation),
            Self::Add(operation) => <AddOperation as Operation<DataType>>::name(operation),
            Self::Sub(operation) => <SubOperation as Operation<DataType>>::name(operation),
            Self::Scale(operation) => <ScaleOperation<DataType, V> as Operation<DataType>>::name(operation),
            Self::Mul(operation) => <MulOperation as Operation<DataType>>::name(operation),
            Self::Div(operation) => <DivOperation as Operation<DataType>>::name(operation),
            Self::Sin(operation) => <SinOperation as Operation<DataType>>::name(operation),
            Self::Cos(operation) => <CosOperation as Operation<DataType>>::name(operation),
            Self::Compare(operation) => <CompareOperation as Operation<DataType>>::name(operation),
            Self::Select(operation) => <SelectOperation as Operation<DataType>>::name(operation),
            Self::StopGradient(operation) => <StopGradientOperation as Operation<DataType>>::name(operation),
            Self::RematerializationName(operation) => {
                <RematerializationNameOperation as Operation<DataType>>::name(operation)
            }
            Self::CustomJvp(operation) => {
                <CustomJvpOperation<V, Self, DataType> as Operation<DataType>>::name(&**operation)
            }
            Self::CustomVjp(operation) => {
                <CustomVjpOperation<V, Self, DataType> as Operation<DataType>>::name(&**operation)
            }
        }
    }

    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        match self {
            Self::Zero(operation) => {
                <ZeroOperation<DataType> as Operation<DataType>>::infer_output_types(operation, input_types)
            }
            Self::ZeroLike(operation) => {
                <ZeroLikeOperation as Operation<DataType>>::infer_output_types(operation, input_types)
            }
            Self::One(operation) => {
                <OneOperation<DataType> as Operation<DataType>>::infer_output_types(operation, input_types)
            }
            Self::OneLike(operation) => {
                <OneLikeOperation as Operation<DataType>>::infer_output_types(operation, input_types)
            }
            Self::Constant(operation) => {
                <ConstantOperation<DataType, V> as Operation<DataType>>::infer_output_types(operation, input_types)
            }
            Self::Neg(operation) => <NegOperation as Operation<DataType>>::infer_output_types(operation, input_types),
            Self::Add(operation) => <AddOperation as Operation<DataType>>::infer_output_types(operation, input_types),
            Self::Sub(operation) => <SubOperation as Operation<DataType>>::infer_output_types(operation, input_types),
            Self::Scale(operation) => {
                <ScaleOperation<DataType, V> as Operation<DataType>>::infer_output_types(operation, input_types)
            }
            Self::Mul(operation) => <MulOperation as Operation<DataType>>::infer_output_types(operation, input_types),
            Self::Div(operation) => <DivOperation as Operation<DataType>>::infer_output_types(operation, input_types),
            Self::Sin(operation) => <SinOperation as Operation<DataType>>::infer_output_types(operation, input_types),
            Self::Cos(operation) => <CosOperation as Operation<DataType>>::infer_output_types(operation, input_types),
            Self::Compare(operation) => {
                <CompareOperation as Operation<DataType>>::infer_output_types(operation, input_types)
            }
            Self::Select(operation) => {
                <SelectOperation as Operation<DataType>>::infer_output_types(operation, input_types)
            }
            Self::StopGradient(operation) => {
                <StopGradientOperation as Operation<DataType>>::infer_output_types(operation, input_types)
            }
            Self::RematerializationName(operation) => {
                <RematerializationNameOperation as Operation<DataType>>::infer_output_types(operation, input_types)
            }
            Self::CustomJvp(operation) => {
                <CustomJvpOperation<V, Self, DataType> as Operation<DataType>>::infer_output_types(
                    &**operation,
                    input_types,
                )
            }
            Self::CustomVjp(operation) => {
                <CustomVjpOperation<V, Self, DataType> as Operation<DataType>>::infer_output_types(
                    &**operation,
                    input_types,
                )
            }
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Zero(operation) => {
                <ZeroOperation<DataType> as Operation<DataType>>::render(operation, formatter, indentation)
            }
            Self::ZeroLike(operation) => {
                <ZeroLikeOperation as Operation<DataType>>::render(operation, formatter, indentation)
            }
            Self::One(operation) => {
                <OneOperation<DataType> as Operation<DataType>>::render(operation, formatter, indentation)
            }
            Self::OneLike(operation) => {
                <OneLikeOperation as Operation<DataType>>::render(operation, formatter, indentation)
            }
            Self::Constant(operation) => {
                <ConstantOperation<DataType, V> as Operation<DataType>>::render(operation, formatter, indentation)
            }
            Self::Neg(operation) => <NegOperation as Operation<DataType>>::render(operation, formatter, indentation),
            Self::Add(operation) => <AddOperation as Operation<DataType>>::render(operation, formatter, indentation),
            Self::Sub(operation) => <SubOperation as Operation<DataType>>::render(operation, formatter, indentation),
            Self::Scale(operation) => {
                <ScaleOperation<DataType, V> as Operation<DataType>>::render(operation, formatter, indentation)
            }
            Self::Mul(operation) => <MulOperation as Operation<DataType>>::render(operation, formatter, indentation),
            Self::Div(operation) => <DivOperation as Operation<DataType>>::render(operation, formatter, indentation),
            Self::Sin(operation) => <SinOperation as Operation<DataType>>::render(operation, formatter, indentation),
            Self::Cos(operation) => <CosOperation as Operation<DataType>>::render(operation, formatter, indentation),
            Self::Compare(operation) => {
                <CompareOperation as Operation<DataType>>::render(operation, formatter, indentation)
            }
            Self::Select(operation) => {
                <SelectOperation as Operation<DataType>>::render(operation, formatter, indentation)
            }
            Self::StopGradient(operation) => {
                <StopGradientOperation as Operation<DataType>>::render(operation, formatter, indentation)
            }
            Self::RematerializationName(operation) => {
                <RematerializationNameOperation as Operation<DataType>>::render(operation, formatter, indentation)
            }
            Self::CustomJvp(operation) => <CustomJvpOperation<V, Self, DataType> as Operation<DataType>>::render(
                &**operation,
                formatter,
                indentation,
            ),
            Self::CustomVjp(operation) => <CustomVjpOperation<V, Self, DataType> as Operation<DataType>>::render(
                &**operation,
                formatter,
                indentation,
            ),
        }
    }
}

impl<V: Value<DataType>> Display for ScalarOperation<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<V: Value<DataType>> From<ZeroOperation<DataType>> for ScalarOperation<V> {
    fn from(operation: ZeroOperation<DataType>) -> Self {
        Self::Zero(operation)
    }
}

impl<'a, V: Value<DataType>> TryFrom<&'a ScalarOperation<V>> for &'a ZeroOperation<DataType> {
    type Error = ();

    fn try_from(value: &'a ScalarOperation<V>) -> Result<Self, ()> {
        match value {
            ScalarOperation::Zero(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<V: Value<DataType>> From<ZeroLikeOperation> for ScalarOperation<V> {
    fn from(operation: ZeroLikeOperation) -> Self {
        Self::ZeroLike(operation)
    }
}

impl<'a, V: Value<DataType>> TryFrom<&'a ScalarOperation<V>> for &'a ZeroLikeOperation {
    type Error = ();

    fn try_from(value: &'a ScalarOperation<V>) -> Result<Self, ()> {
        match value {
            ScalarOperation::ZeroLike(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<V: Value<DataType>> From<OneOperation<DataType>> for ScalarOperation<V> {
    fn from(operation: OneOperation<DataType>) -> Self {
        Self::One(operation)
    }
}

impl<'a, V: Value<DataType>> TryFrom<&'a ScalarOperation<V>> for &'a OneOperation<DataType> {
    type Error = ();

    fn try_from(value: &'a ScalarOperation<V>) -> Result<Self, ()> {
        match value {
            ScalarOperation::One(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<V: Value<DataType>> From<OneLikeOperation> for ScalarOperation<V> {
    fn from(operation: OneLikeOperation) -> Self {
        Self::OneLike(operation)
    }
}

impl<'a, V: Value<DataType>> TryFrom<&'a ScalarOperation<V>> for &'a OneLikeOperation {
    type Error = ();

    fn try_from(value: &'a ScalarOperation<V>) -> Result<Self, ()> {
        match value {
            ScalarOperation::OneLike(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<V: Value<DataType>> From<ConstantOperation<DataType, V>> for ScalarOperation<V> {
    fn from(operation: ConstantOperation<DataType, V>) -> Self {
        Self::Constant(operation)
    }
}

impl<'a, V: Value<DataType>> TryFrom<&'a ScalarOperation<V>> for &'a ConstantOperation<DataType, V> {
    type Error = ();

    fn try_from(value: &'a ScalarOperation<V>) -> Result<Self, ()> {
        match value {
            ScalarOperation::Constant(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<V: Value<DataType>> From<NegOperation> for ScalarOperation<V> {
    fn from(operation: NegOperation) -> Self {
        Self::Neg(operation)
    }
}

impl<'a, V: Value<DataType>> TryFrom<&'a ScalarOperation<V>> for &'a NegOperation {
    type Error = ();

    fn try_from(value: &'a ScalarOperation<V>) -> Result<Self, ()> {
        match value {
            ScalarOperation::Neg(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<V: Value<DataType>> From<AddOperation> for ScalarOperation<V> {
    fn from(operation: AddOperation) -> Self {
        Self::Add(operation)
    }
}

impl<'a, V: Value<DataType>> TryFrom<&'a ScalarOperation<V>> for &'a AddOperation {
    type Error = ();

    fn try_from(value: &'a ScalarOperation<V>) -> Result<Self, ()> {
        match value {
            ScalarOperation::Add(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<V: Value<DataType>> From<SubOperation> for ScalarOperation<V> {
    fn from(operation: SubOperation) -> Self {
        Self::Sub(operation)
    }
}

impl<'a, V: Value<DataType>> TryFrom<&'a ScalarOperation<V>> for &'a SubOperation {
    type Error = ();

    fn try_from(value: &'a ScalarOperation<V>) -> Result<Self, ()> {
        match value {
            ScalarOperation::Sub(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<V: Value<DataType>> From<ScaleOperation<DataType, V>> for ScalarOperation<V> {
    fn from(operation: ScaleOperation<DataType, V>) -> Self {
        Self::Scale(operation)
    }
}

impl<'a, V: Value<DataType>> TryFrom<&'a ScalarOperation<V>> for &'a ScaleOperation<DataType, V> {
    type Error = ();

    fn try_from(value: &'a ScalarOperation<V>) -> Result<Self, ()> {
        match value {
            ScalarOperation::Scale(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<V: Value<DataType>> From<MulOperation> for ScalarOperation<V> {
    fn from(operation: MulOperation) -> Self {
        Self::Mul(operation)
    }
}

impl<'a, V: Value<DataType>> TryFrom<&'a ScalarOperation<V>> for &'a MulOperation {
    type Error = ();

    fn try_from(value: &'a ScalarOperation<V>) -> Result<Self, ()> {
        match value {
            ScalarOperation::Mul(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<V: Value<DataType>> From<DivOperation> for ScalarOperation<V> {
    fn from(operation: DivOperation) -> Self {
        Self::Div(operation)
    }
}

impl<'a, V: Value<DataType>> TryFrom<&'a ScalarOperation<V>> for &'a DivOperation {
    type Error = ();

    fn try_from(value: &'a ScalarOperation<V>) -> Result<Self, ()> {
        match value {
            ScalarOperation::Div(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<V: Value<DataType>> From<SinOperation> for ScalarOperation<V> {
    fn from(operation: SinOperation) -> Self {
        Self::Sin(operation)
    }
}

impl<'a, V: Value<DataType>> TryFrom<&'a ScalarOperation<V>> for &'a SinOperation {
    type Error = ();

    fn try_from(value: &'a ScalarOperation<V>) -> Result<Self, ()> {
        match value {
            ScalarOperation::Sin(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<V: Value<DataType>> From<CosOperation> for ScalarOperation<V> {
    fn from(operation: CosOperation) -> Self {
        Self::Cos(operation)
    }
}

impl<'a, V: Value<DataType>> TryFrom<&'a ScalarOperation<V>> for &'a CosOperation {
    type Error = ();

    fn try_from(value: &'a ScalarOperation<V>) -> Result<Self, ()> {
        match value {
            ScalarOperation::Cos(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<V: Value<DataType>> From<CompareOperation> for ScalarOperation<V> {
    fn from(operation: CompareOperation) -> Self {
        Self::Compare(operation)
    }
}

impl<'a, V: Value<DataType>> TryFrom<&'a ScalarOperation<V>> for &'a CompareOperation {
    type Error = ();

    fn try_from(value: &'a ScalarOperation<V>) -> Result<Self, ()> {
        match value {
            ScalarOperation::Compare(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<V: Value<DataType>> From<SelectOperation> for ScalarOperation<V> {
    fn from(operation: SelectOperation) -> Self {
        Self::Select(operation)
    }
}

impl<'a, V: Value<DataType>> TryFrom<&'a ScalarOperation<V>> for &'a SelectOperation {
    type Error = ();

    fn try_from(value: &'a ScalarOperation<V>) -> Result<Self, ()> {
        match value {
            ScalarOperation::Select(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<V: Value<DataType>> From<StopGradientOperation> for ScalarOperation<V> {
    fn from(operation: StopGradientOperation) -> Self {
        Self::StopGradient(operation)
    }
}

impl<'a, V: Value<DataType>> TryFrom<&'a ScalarOperation<V>> for &'a StopGradientOperation {
    type Error = ();

    fn try_from(value: &'a ScalarOperation<V>) -> Result<Self, ()> {
        match value {
            ScalarOperation::StopGradient(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<V: Value<DataType>> From<RematerializationNameOperation> for ScalarOperation<V> {
    fn from(operation: RematerializationNameOperation) -> Self {
        Self::RematerializationName(operation)
    }
}

impl<'a, V: Value<DataType>> TryFrom<&'a ScalarOperation<V>> for &'a RematerializationNameOperation {
    type Error = ();

    fn try_from(value: &'a ScalarOperation<V>) -> Result<Self, ()> {
        match value {
            ScalarOperation::RematerializationName(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<V: Value<DataType>> From<CustomJvpOperation<V, ScalarOperation<V>, DataType>> for ScalarOperation<V> {
    fn from(operation: CustomJvpOperation<V, ScalarOperation<V>, DataType>) -> Self {
        Self::CustomJvp(Box::new(operation))
    }
}

impl<'a, V: Value<DataType>> TryFrom<&'a ScalarOperation<V>>
    for &'a CustomJvpOperation<V, ScalarOperation<V>, DataType>
{
    type Error = ();

    fn try_from(value: &'a ScalarOperation<V>) -> Result<Self, ()> {
        match value {
            ScalarOperation::CustomJvp(operation) => Ok(&**operation),
            _ => Err(()),
        }
    }
}

impl<V: Value<DataType>> From<CustomVjpOperation<V, ScalarOperation<V>, DataType>> for ScalarOperation<V> {
    fn from(operation: CustomVjpOperation<V, ScalarOperation<V>, DataType>) -> Self {
        Self::CustomVjp(Box::new(operation))
    }
}

impl<'a, V: Value<DataType>> TryFrom<&'a ScalarOperation<V>>
    for &'a CustomVjpOperation<V, ScalarOperation<V>, DataType>
{
    type Error = ();

    fn try_from(value: &'a ScalarOperation<V>) -> Result<Self, ()> {
        match value {
            ScalarOperation::CustomVjp(operation) => Ok(&**operation),
            _ => Err(()),
        }
    }
}

impl<V: Value<DataType>, D> DifferentiableOperation<D> for ScalarOperation<V>
where
    ZeroOperation<DataType>: DifferentiableOperation<D>,
    ZeroLikeOperation: DifferentiableOperation<D>,
    OneOperation<DataType>: DifferentiableOperation<D>,
    OneLikeOperation: DifferentiableOperation<D>,
    ConstantOperation<DataType, V>: DifferentiableOperation<D>,
    NegOperation: DifferentiableOperation<D>,
    AddOperation: DifferentiableOperation<D>,
    SubOperation: DifferentiableOperation<D>,
    ScaleOperation<DataType, V>: DifferentiableOperation<D>,
    MulOperation: DifferentiableOperation<D>,
    DivOperation: DifferentiableOperation<D>,
    SinOperation: DifferentiableOperation<D>,
    CosOperation: DifferentiableOperation<D>,
    CompareOperation: DifferentiableOperation<D>,
    SelectOperation: DifferentiableOperation<D>,
    StopGradientOperation: DifferentiableOperation<D>,
    RematerializationNameOperation: DifferentiableOperation<D>,
    D: DifferentiationContext<Type = DataType, Constant = V> + Domain<Operation = ScalarOperation<V>>,
    D::Operation: From<ZeroOperation<DataType>> + From<OneOperation<DataType>>,
    D::Value: crate::tracing_v2::rematerialization::RematerializationName,
    D::Value: Add<Output = D::Value>
        + Sub<Output = D::Value>
        + Mul<Output = D::Value>
        + Div<Output = D::Value>
        + Neg<Output = D::Value>
        + SupportsTrigonometricOperations
        + ZeroLike
        + OneLike
        + Compare<Output = D::Value>
        + SelectCondition
        + Parameterized<D::Value>,
    D::Value: Select<Condition = <D::Value as SelectCondition>::Condition>,
    <D::Value as Parameterized<D::Value>>::ParameterStructure: std::fmt::Debug + PartialEq,
    Vec<D::Value>: Parameterized<D::Value, ParameterStructure: std::fmt::Debug + PartialEq>,
    ScalarOperation<V>: Clone + ProgramLinearizableOperation<D>,
    LinearOperationOf<D>: SupportsLinearScalarOperation<DataType, CapturedFactor<DataType, D::Value>>
        + From<LinearSelectOperation<CapturedFactor<DataType, D::Value>>>
        + crate::tracing_v2::ResidualizedOperation<D>
        + From<CustomVjpCallOperation<V, ScalarOperation<V>, CapturedFactor<DataType, D::Value>, DataType>>,
    Vec<V>: Parameterized<
            V,
            Family: crate::parameters::ParameterizedFamily<D::Tangent>
                        + crate::parameters::ParameterizedFamily<D::Value>,
            To<D::Value> = Vec<D::Value>,
            To<D::Tangent> = Vec<D::Tangent>,
            ParameterStructure: std::fmt::Debug + PartialEq,
        >,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        match self {
            Self::Zero(operation) => operation.jvp(context, inputs),
            Self::ZeroLike(operation) => operation.jvp(context, inputs),
            Self::One(operation) => operation.jvp(context, inputs),
            Self::OneLike(operation) => operation.jvp(context, inputs),
            Self::Constant(operation) => operation.jvp(context, inputs),
            Self::Neg(operation) => operation.jvp(context, inputs),
            Self::Add(operation) => operation.jvp(context, inputs),
            Self::Sub(operation) => operation.jvp(context, inputs),
            Self::Scale(operation) => operation.jvp(context, inputs),
            Self::Mul(operation) => operation.jvp(context, inputs),
            Self::Div(operation) => operation.jvp(context, inputs),
            Self::Sin(operation) => operation.jvp(context, inputs),
            Self::Cos(operation) => operation.jvp(context, inputs),
            Self::Compare(operation) => operation.jvp(context, inputs),
            Self::Select(operation) => operation.jvp(context, inputs),
            Self::StopGradient(operation) => operation.jvp(context, inputs),
            Self::RematerializationName(operation) => operation.jvp(context, inputs),
            Self::CustomJvp(operation) => <CustomJvpOperation<V, Self, DataType> as DifferentiableOperation<D>>::jvp(
                &**operation,
                context,
                inputs,
            ),
            Self::CustomVjp(operation) => <CustomVjpOperation<V, Self, DataType> as DifferentiableOperation<D>>::jvp(
                &**operation,
                context,
                inputs,
            ),
        }
    }
}

impl<V: Value<DataType>> InterpretableOperation<DataType, V> for ScalarOperation<V>
where
    ZeroOperation<DataType>: InterpretableOperation<DataType, V>,
    ZeroLikeOperation: InterpretableOperation<DataType, V>,
    OneOperation<DataType>: InterpretableOperation<DataType, V>,
    OneLikeOperation: InterpretableOperation<DataType, V>,
    ConstantOperation<DataType, V>: InterpretableOperation<DataType, V>,
    NegOperation: InterpretableOperation<DataType, V>,
    AddOperation: InterpretableOperation<DataType, V>,
    SubOperation: InterpretableOperation<DataType, V>,
    ScaleOperation<DataType, V>: InterpretableOperation<DataType, V>,
    MulOperation: InterpretableOperation<DataType, V>,
    DivOperation: InterpretableOperation<DataType, V>,
    SinOperation: InterpretableOperation<DataType, V>,
    CosOperation: InterpretableOperation<DataType, V>,
    CompareOperation: InterpretableOperation<DataType, V>,
    SelectOperation: InterpretableOperation<DataType, V>,
    StopGradientOperation: InterpretableOperation<DataType, V>,
    RematerializationNameOperation: InterpretableOperation<DataType, V>,
    V: Value<DataType>,
    Vec<V>: Parameterized<V, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret(
        &self,
        context: &<V as Value<DataType>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        match self {
            Self::Zero(operation) => operation.interpret(context, inputs),
            Self::ZeroLike(operation) => operation.interpret(context, inputs),
            Self::One(operation) => operation.interpret(context, inputs),
            Self::OneLike(operation) => operation.interpret(context, inputs),
            Self::Constant(operation) => operation.interpret(context, inputs),
            Self::Neg(operation) => operation.interpret(context, inputs),
            Self::Add(operation) => operation.interpret(context, inputs),
            Self::Sub(operation) => operation.interpret(context, inputs),
            Self::Scale(operation) => operation.interpret(context, inputs),
            Self::Mul(operation) => operation.interpret(context, inputs),
            Self::Div(operation) => operation.interpret(context, inputs),
            Self::Sin(operation) => operation.interpret(context, inputs),
            Self::Cos(operation) => operation.interpret(context, inputs),
            Self::Compare(operation) => operation.interpret(context, inputs),
            Self::Select(operation) => operation.interpret(context, inputs),
            Self::StopGradient(operation) => operation.interpret(context, inputs),
            Self::RematerializationName(operation) => operation.interpret(context, inputs),
            Self::CustomJvp(operation) => <CustomJvpOperation<V, Self, DataType> as InterpretableOperation<
                DataType,
                V,
            >>::interpret(&**operation, context, inputs),
            Self::CustomVjp(operation) => <CustomVjpOperation<V, Self, DataType> as InterpretableOperation<
                DataType,
                V,
            >>::interpret(&**operation, context, inputs),
        }
    }
}

/// Closed scalar operation type for staged linear scalar programs.
///
/// The `C` parameter is the constant type of the
/// [`DifferentiationContext`](crate::tracing_v2::DifferentiationContext) that stages the linear program: every
/// context pins `C` to its [`Domain::Constant`](crate::domains::Domain) in its `LinearOperation` associated-type
/// definition. It types the user-supplied backward programs captured by [`CustomVjpCall`](Self::CustomVjpCall),
/// which are written over context constants rather than over the factor type `F`.
///
/// The variants mirror the linear scalar primitives: typed [`Zero`](Self::Zero)/[`One`](Self::One) and their
/// exemplar-derived [`ZeroLike`](Self::ZeroLike)/[`OneLike`](Self::OneLike) maps, a typed
/// [`Constant`](Self::Constant), [`Neg`](Self::Neg)/[`Add`](Self::Add)/[`Sub`](Self::Sub), scaling by a captured
/// factor ([`Scale`](Self::Scale)), the captured-condition [`Select`](Self::Select)
/// ([`LinearSelectOperation`]), and the opaque [`CustomVjpCall`](Self::CustomVjpCall) staged by a `custom_vjp`
/// linearization (its transpose replays the user's backward program).
#[derive(Clone, Debug)]
pub enum LinearScalarOperation<C: Value<DataType>, F: Value<DataType> = C> {
    /// Typed scalar zero.
    Zero(ZeroOperation<DataType>),

    /// Exemplar-derived scalar zero.
    ZeroLike(ZeroLikeOperation),

    /// Typed scalar one.
    One(OneOperation<DataType>),

    /// Exemplar-derived scalar one.
    OneLike(OneLikeOperation),

    /// Typed scalar constant carried as a factor.
    Constant(ConstantOperation<DataType, F>),

    /// Scalar negation.
    Neg(NegOperation),

    /// Scalar addition.
    Add(AddOperation),

    /// Scalar subtraction.
    Sub(SubOperation),

    /// Scaling by a captured scalar factor.
    Scale(ScaleOperation<DataType, F>),

    /// Selection on a captured Boolean condition.
    Select(LinearSelectOperation<F>),

    /// Opaque `custom_vjp` call whose transpose replays the user's backward program.
    CustomVjpCall(Box<CustomVjpCallOperation<C, ScalarOperation<C>, F, DataType>>),
}

impl<C: Value<DataType>, F: Value<DataType>> Operation<DataType> for LinearScalarOperation<C, F> {
    // Several variant payloads (the elementwise arithmetic operations) implement
    // [`Operation`](crate::operations::Operation) for both [`DataType`] and `ArrayType`, so plain method-call syntax
    // cannot infer the type parameter here. The arms therefore disambiguate to `Operation<DataType>` explicitly.
    fn name(&self) -> &'static str {
        match self {
            Self::Zero(operation) => <ZeroOperation<DataType> as Operation<DataType>>::name(operation),
            Self::ZeroLike(operation) => <ZeroLikeOperation as Operation<DataType>>::name(operation),
            Self::One(operation) => <OneOperation<DataType> as Operation<DataType>>::name(operation),
            Self::OneLike(operation) => <OneLikeOperation as Operation<DataType>>::name(operation),
            Self::Constant(operation) => <ConstantOperation<DataType, F> as Operation<DataType>>::name(operation),
            Self::Neg(operation) => <NegOperation as Operation<DataType>>::name(operation),
            Self::Add(operation) => <AddOperation as Operation<DataType>>::name(operation),
            Self::Sub(operation) => <SubOperation as Operation<DataType>>::name(operation),
            Self::Scale(operation) => <ScaleOperation<DataType, F> as Operation<DataType>>::name(operation),
            Self::Select(operation) => <LinearSelectOperation<F> as Operation<DataType>>::name(operation),
            Self::CustomVjpCall(operation) => {
                <CustomVjpCallOperation<C, ScalarOperation<C>, F, DataType> as Operation<DataType>>::name(&**operation)
            }
        }
    }

    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        match self {
            Self::Zero(operation) => {
                <ZeroOperation<DataType> as Operation<DataType>>::infer_output_types(operation, input_types)
            }
            Self::ZeroLike(operation) => {
                <ZeroLikeOperation as Operation<DataType>>::infer_output_types(operation, input_types)
            }
            Self::One(operation) => {
                <OneOperation<DataType> as Operation<DataType>>::infer_output_types(operation, input_types)
            }
            Self::OneLike(operation) => {
                <OneLikeOperation as Operation<DataType>>::infer_output_types(operation, input_types)
            }
            Self::Constant(operation) => {
                <ConstantOperation<DataType, F> as Operation<DataType>>::infer_output_types(operation, input_types)
            }
            Self::Neg(operation) => <NegOperation as Operation<DataType>>::infer_output_types(operation, input_types),
            Self::Add(operation) => <AddOperation as Operation<DataType>>::infer_output_types(operation, input_types),
            Self::Sub(operation) => <SubOperation as Operation<DataType>>::infer_output_types(operation, input_types),
            Self::Scale(operation) => {
                <ScaleOperation<DataType, F> as Operation<DataType>>::infer_output_types(operation, input_types)
            }
            Self::Select(operation) => {
                <LinearSelectOperation<F> as Operation<DataType>>::infer_output_types(operation, input_types)
            }
            Self::CustomVjpCall(operation) => {
                <CustomVjpCallOperation<C, ScalarOperation<C>, F, DataType> as Operation<DataType>>::infer_output_types(
                    &**operation,
                    input_types,
                )
            }
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Zero(operation) => {
                <ZeroOperation<DataType> as Operation<DataType>>::render(operation, formatter, indentation)
            }
            Self::ZeroLike(operation) => {
                <ZeroLikeOperation as Operation<DataType>>::render(operation, formatter, indentation)
            }
            Self::One(operation) => {
                <OneOperation<DataType> as Operation<DataType>>::render(operation, formatter, indentation)
            }
            Self::OneLike(operation) => {
                <OneLikeOperation as Operation<DataType>>::render(operation, formatter, indentation)
            }
            Self::Constant(operation) => {
                <ConstantOperation<DataType, F> as Operation<DataType>>::render(operation, formatter, indentation)
            }
            Self::Neg(operation) => <NegOperation as Operation<DataType>>::render(operation, formatter, indentation),
            Self::Add(operation) => <AddOperation as Operation<DataType>>::render(operation, formatter, indentation),
            Self::Sub(operation) => <SubOperation as Operation<DataType>>::render(operation, formatter, indentation),
            Self::Scale(operation) => {
                <ScaleOperation<DataType, F> as Operation<DataType>>::render(operation, formatter, indentation)
            }
            Self::Select(operation) => {
                <LinearSelectOperation<F> as Operation<DataType>>::render(operation, formatter, indentation)
            }
            Self::CustomVjpCall(operation) => {
                <CustomVjpCallOperation<C, ScalarOperation<C>, F, DataType> as Operation<DataType>>::render(
                    &**operation,
                    formatter,
                    indentation,
                )
            }
        }
    }
}

impl<C: Value<DataType>, F: Value<DataType>> Display for LinearScalarOperation<C, F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<C: Value<DataType>, F: Value<DataType>> From<ZeroOperation<DataType>> for LinearScalarOperation<C, F> {
    fn from(operation: ZeroOperation<DataType>) -> Self {
        Self::Zero(operation)
    }
}

impl<'a, C: Value<DataType>, F: Value<DataType>> TryFrom<&'a LinearScalarOperation<C, F>>
    for &'a ZeroOperation<DataType>
{
    type Error = ();

    fn try_from(value: &'a LinearScalarOperation<C, F>) -> Result<Self, ()> {
        match value {
            LinearScalarOperation::Zero(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<C: Value<DataType>, F: Value<DataType>> From<ZeroLikeOperation> for LinearScalarOperation<C, F> {
    fn from(operation: ZeroLikeOperation) -> Self {
        Self::ZeroLike(operation)
    }
}

impl<'a, C: Value<DataType>, F: Value<DataType>> TryFrom<&'a LinearScalarOperation<C, F>> for &'a ZeroLikeOperation {
    type Error = ();

    fn try_from(value: &'a LinearScalarOperation<C, F>) -> Result<Self, ()> {
        match value {
            LinearScalarOperation::ZeroLike(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<C: Value<DataType>, F: Value<DataType>> From<OneOperation<DataType>> for LinearScalarOperation<C, F> {
    fn from(operation: OneOperation<DataType>) -> Self {
        Self::One(operation)
    }
}

impl<'a, C: Value<DataType>, F: Value<DataType>> TryFrom<&'a LinearScalarOperation<C, F>>
    for &'a OneOperation<DataType>
{
    type Error = ();

    fn try_from(value: &'a LinearScalarOperation<C, F>) -> Result<Self, ()> {
        match value {
            LinearScalarOperation::One(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<C: Value<DataType>, F: Value<DataType>> From<OneLikeOperation> for LinearScalarOperation<C, F> {
    fn from(operation: OneLikeOperation) -> Self {
        Self::OneLike(operation)
    }
}

impl<'a, C: Value<DataType>, F: Value<DataType>> TryFrom<&'a LinearScalarOperation<C, F>> for &'a OneLikeOperation {
    type Error = ();

    fn try_from(value: &'a LinearScalarOperation<C, F>) -> Result<Self, ()> {
        match value {
            LinearScalarOperation::OneLike(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<C: Value<DataType>, F: Value<DataType>> From<ConstantOperation<DataType, F>> for LinearScalarOperation<C, F> {
    fn from(operation: ConstantOperation<DataType, F>) -> Self {
        Self::Constant(operation)
    }
}

impl<'a, C: Value<DataType>, F: Value<DataType>> TryFrom<&'a LinearScalarOperation<C, F>>
    for &'a ConstantOperation<DataType, F>
{
    type Error = ();

    fn try_from(value: &'a LinearScalarOperation<C, F>) -> Result<Self, ()> {
        match value {
            LinearScalarOperation::Constant(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<C: Value<DataType>, F: Value<DataType>> From<NegOperation> for LinearScalarOperation<C, F> {
    fn from(operation: NegOperation) -> Self {
        Self::Neg(operation)
    }
}

impl<'a, C: Value<DataType>, F: Value<DataType>> TryFrom<&'a LinearScalarOperation<C, F>> for &'a NegOperation {
    type Error = ();

    fn try_from(value: &'a LinearScalarOperation<C, F>) -> Result<Self, ()> {
        match value {
            LinearScalarOperation::Neg(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<C: Value<DataType>, F: Value<DataType>> From<AddOperation> for LinearScalarOperation<C, F> {
    fn from(operation: AddOperation) -> Self {
        Self::Add(operation)
    }
}

impl<'a, C: Value<DataType>, F: Value<DataType>> TryFrom<&'a LinearScalarOperation<C, F>> for &'a AddOperation {
    type Error = ();

    fn try_from(value: &'a LinearScalarOperation<C, F>) -> Result<Self, ()> {
        match value {
            LinearScalarOperation::Add(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<C: Value<DataType>, F: Value<DataType>> From<SubOperation> for LinearScalarOperation<C, F> {
    fn from(operation: SubOperation) -> Self {
        Self::Sub(operation)
    }
}

impl<'a, C: Value<DataType>, F: Value<DataType>> TryFrom<&'a LinearScalarOperation<C, F>> for &'a SubOperation {
    type Error = ();

    fn try_from(value: &'a LinearScalarOperation<C, F>) -> Result<Self, ()> {
        match value {
            LinearScalarOperation::Sub(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<C: Value<DataType>, F: Value<DataType>> From<ScaleOperation<DataType, F>> for LinearScalarOperation<C, F> {
    fn from(operation: ScaleOperation<DataType, F>) -> Self {
        Self::Scale(operation)
    }
}

impl<'a, C: Value<DataType>, F: Value<DataType>> TryFrom<&'a LinearScalarOperation<C, F>>
    for &'a ScaleOperation<DataType, F>
{
    type Error = ();

    fn try_from(value: &'a LinearScalarOperation<C, F>) -> Result<Self, ()> {
        match value {
            LinearScalarOperation::Scale(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<C: Value<DataType>, F: Value<DataType>> From<LinearSelectOperation<F>> for LinearScalarOperation<C, F> {
    fn from(operation: LinearSelectOperation<F>) -> Self {
        Self::Select(operation)
    }
}

impl<'a, C: Value<DataType>, F: Value<DataType>> TryFrom<&'a LinearScalarOperation<C, F>>
    for &'a LinearSelectOperation<F>
{
    type Error = ();

    fn try_from(value: &'a LinearScalarOperation<C, F>) -> Result<Self, ()> {
        match value {
            LinearScalarOperation::Select(operation) => Ok(operation),
            _ => Err(()),
        }
    }
}

impl<C: Value<DataType>, F: Value<DataType>> From<CustomVjpCallOperation<C, ScalarOperation<C>, F, DataType>>
    for LinearScalarOperation<C, F>
{
    fn from(operation: CustomVjpCallOperation<C, ScalarOperation<C>, F, DataType>) -> Self {
        Self::CustomVjpCall(Box::new(operation))
    }
}

impl<'a, C: Value<DataType>, F: Value<DataType>> TryFrom<&'a LinearScalarOperation<C, F>>
    for &'a CustomVjpCallOperation<C, ScalarOperation<C>, F, DataType>
{
    type Error = ();

    fn try_from(value: &'a LinearScalarOperation<C, F>) -> Result<Self, ()> {
        match value {
            LinearScalarOperation::CustomVjpCall(operation) => Ok(&**operation),
            _ => Err(()),
        }
    }
}

impl<C: Value<DataType>, F: Value<DataType>, W, O> TransposableOperation<DataType, W, O> for LinearScalarOperation<C, F>
where
    ZeroOperation<DataType>: TransposableOperation<DataType, W, O>,
    ZeroLikeOperation: TransposableOperation<DataType, W, O>,
    OneOperation<DataType>: TransposableOperation<DataType, W, O>,
    OneLikeOperation: TransposableOperation<DataType, W, O>,
    ConstantOperation<DataType, F>: TransposableOperation<DataType, W, O>,
    NegOperation: TransposableOperation<DataType, W, O>,
    AddOperation: TransposableOperation<DataType, W, O>,
    SubOperation: TransposableOperation<DataType, W, O>,
    ScaleOperation<DataType, F>: TransposableOperation<DataType, W, O>,
    LinearSelectOperation<F>: TransposableOperation<DataType, W, O>,
    CustomVjpCallOperation<C, ScalarOperation<C>, F, DataType>: TransposableOperation<DataType, W, O>,
    W: Value<DataType>,
    O: Operation<DataType>,
    W: Add<Output = W> + Neg<Output = W> + ZeroLike + OneLike,
    Vec<W>: Parameterized<W, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<'transpose, DataType, W, O>,
        input_types: &[&DataType],
        output_cotangents: &[Cotangent<'transpose, DataType, W, O>],
    ) -> Result<Vec<Cotangent<'transpose, DataType, W, O>>, ProgramError> {
        match self {
            Self::Zero(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::ZeroLike(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::One(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::OneLike(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Constant(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Neg(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Add(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Sub(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Scale(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::Select(operation) => operation.transpose(context, input_types, output_cotangents),
            Self::CustomVjpCall(operation) => {
                <CustomVjpCallOperation<C, ScalarOperation<C>, F, DataType> as TransposableOperation<
                    DataType,
                    W,
                    O,
                >>::transpose(&**operation, context, input_types, output_cotangents)
            }
        }
    }
}

impl<V: Value<DataType>> MaybeRematerializationName for ScalarOperation<V> {
    #[inline]
    fn rematerialization_name(&self) -> Option<&str> {
        match self {
            Self::RematerializationName(operation) => Some(operation.tag()),
            _ => None,
        }
    }
}

impl<V: Value<DataType>> crate::tracing_v2::operations::dot::MaybeDot for ScalarOperation<V> {
    #[inline]
    fn dot_dimensions(&self) -> Option<&crate::tracing_v2::operations::dot::DotDimensionNumbers> {
        None
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::operations::compare::Compare;
    use crate::operations::control_flow::Select;
    use crate::scalars::ScalarDomain;
    use crate::tracing::trace;

    use super::*;

    #[test]
    fn test_scalar_compare_and_select_program() {
        // `f(x, y) = select(x > y, x + x, y)` staged through `ScalarOperation` tracers.
        let domain = ScalarDomain::<f64>::new();
        let (output_type, program) = trace(
            &domain,
            |(x, y)| {
                let mask = x.clone().greater_than(&y);
                Select::select(&mask, &(x.clone() + x), &y)
            },
            (DataType::F64, DataType::F64),
        )
        .unwrap();
        assert_eq!(output_type, DataType::F64);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:bool = compare [direction=GreaterThan] %0 %1
                    %3:f64 = add %0 %0
                    %4:f64 = select %2 %3 %1
                in (%4)
            "}
            .trim_end(),
        );

        // Interpreting the staged program exercises the in-band Boolean condition encoding of scalar values.
        assert_eq!(program.interpret((3.0, 2.0)), Ok(6.0));
        assert_eq!(program.interpret((1.0, 2.0)), Ok(2.0));
    }
}
