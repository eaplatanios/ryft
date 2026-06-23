use std::ops::{Add, Div, Mul, Neg, Sub};

use ryft_macros::{Operation, TransposableOperation};

use crate::contexts::Context;
use crate::domains::Domain;
use crate::operations::InterpretableOperation;
use crate::operations::arithmetic::{
    AddOperation, DivOperation, MulOperation, NegOperation, ScaleOperation, SubOperation,
};
use crate::operations::compare::{Compare, CompareOperation};
use crate::operations::constants::{
    ConstantOperation, MaybeZeroOperation, OneLike, OneLikeOperation, OneOperation, ZeroLike, ZeroLikeOperation,
    ZeroOperation,
};
use crate::operations::control_flow::{Select, SelectCondition, SelectOperation};
use crate::operations::differentiation::StopGradientOperation;
use crate::operations::trigonometric::{CosOperation, SinOperation};
use crate::parameters::Parameterized;
use crate::payloads::Input;
use crate::programs::{ProgramError, Value};
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
use crate::types::DataType;

// TODO(eaplatanios): Review this file.

/// Closed scalar operation type for ordinary staged scalar programs.
///
/// [`ScalarOperation`] is intentionally limited to operations that are valid for scalar [`DataType`] metadata.
/// Array-only primitives such as reshaping and matrix multiplication remain available as standalone operations and
/// through array-based backends, but they are not variants of this enum.
#[derive(Clone, Debug, Operation)]
pub enum ScalarOperation<V: Value<DataType>> {
    Zero(ZeroOperation<DataType>),
    ZeroLike(ZeroLikeOperation),
    One(OneOperation<DataType>),
    OneLike(OneLikeOperation),
    Constant(ConstantOperation<DataType, V>),
    Neg(NegOperation),
    Add(AddOperation),
    Sub(SubOperation),
    Scale(ScaleOperation<DataType, V>),
    Mul(MulOperation),
    Div(DivOperation),
    Sin(SinOperation),
    Cos(CosOperation),
    Compare(CompareOperation),
    Select(SelectOperation),
    StopGradient(StopGradientOperation),
    RematerializationName(RematerializationNameOperation),
    CustomJvp(Box<CustomJvpOperation<DataType, V, Self>>),
    CustomVjp(Box<CustomVjpOperation<DataType, V, Self>>),
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
        + From<CustomVjpCallOperation<DataType, V, ScalarOperation<V>, CapturedFactor<DataType, D::Value>>>,
    LinearOperationOf<D>: MaybeZeroOperation<DataType>,
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
            Self::CustomJvp(operation) => operation.jvp(context, inputs),
            Self::CustomVjp(operation) => operation.jvp(context, inputs),
        }
    }
}

impl<C, V> InterpretableOperation<DataType, V> for ScalarOperation<C>
where
    C: Value<DataType>,
    V: Value<DataType>,
    V::InterpretationContext: Context<Type = DataType, Constant = C, Value = V>,
    ZeroOperation<DataType>: InterpretableOperation<DataType, V>,
    ZeroLikeOperation: InterpretableOperation<DataType, V>,
    OneOperation<DataType>: InterpretableOperation<DataType, V>,
    OneLikeOperation: InterpretableOperation<DataType, V>,
    ConstantOperation<DataType, C>: InterpretableOperation<DataType, V>,
    NegOperation: InterpretableOperation<DataType, V>,
    AddOperation: InterpretableOperation<DataType, V>,
    SubOperation: InterpretableOperation<DataType, V>,
    ScaleOperation<DataType, C>: InterpretableOperation<DataType, V>,
    MulOperation: InterpretableOperation<DataType, V>,
    DivOperation: InterpretableOperation<DataType, V>,
    SinOperation: InterpretableOperation<DataType, V>,
    CosOperation: InterpretableOperation<DataType, V>,
    CompareOperation: InterpretableOperation<DataType, V>,
    SelectOperation: InterpretableOperation<DataType, V>,
    StopGradientOperation: InterpretableOperation<DataType, V>,
    RematerializationNameOperation: InterpretableOperation<DataType, V>,
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
            Self::CustomJvp(operation) => operation.interpret(context, inputs),
            Self::CustomVjp(operation) => operation.interpret(context, inputs),
        }
    }
}

/// Closed scalar operation type for staged linear scalar programs.
///
/// The `V` parameter is the scalar tangent/cotangent value type carried by the linear program. It is also the linear
/// program's constant-table type, so linear constants are typed as `V`. The `C` parameter is the primal context
/// constant type used by user-supplied programs captured by [`CustomVjpCall`](Self::CustomVjpCall), which are written
/// over context constants rather than over the linear value type `V` or captured-factor type `F`.
///
/// The variants mirror the linear scalar primitives: typed [`Zero`](Self::Zero)/[`One`](Self::One) and their
/// exemplar-derived [`ZeroLike`](Self::ZeroLike)/[`OneLike`](Self::OneLike) maps, a typed
/// [`Constant`](Self::Constant), [`Neg`](Self::Neg)/[`Add`](Self::Add)/[`Sub`](Self::Sub), scaling by a captured
/// factor ([`Scale`](Self::Scale)), the captured-condition [`Select`](Self::Select)
/// ([`LinearSelectOperation`]), and the opaque [`CustomVjpCall`](Self::CustomVjpCall) staged by a `custom_vjp`
/// linearization (its transpose replays the user's backward program).
#[derive(Clone, Debug, Operation, TransposableOperation)]
pub enum LinearScalarOperation<V: Value<DataType>, C: Value<DataType> = V, F: Value<DataType> = C> {
    Zero(ZeroOperation<DataType>),
    ZeroLike(ZeroLikeOperation),
    One(OneOperation<DataType>),
    OneLike(OneLikeOperation),
    Constant(ConstantOperation<DataType, V, Input>),
    Neg(NegOperation),
    Add(AddOperation),
    Sub(SubOperation),
    Scale(ScaleOperation<DataType, F, Input>),
    Select(LinearSelectOperation<F>),
    CustomVjpCall(Box<CustomVjpCallOperation<DataType, C, ScalarOperation<C>, F>>),
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
