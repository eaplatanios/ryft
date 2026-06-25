use ryft_macros::{Operation, TransposableOperation};

use crate::domains::Domain;
use crate::operations::BooleanLike;
use crate::operations::arithmetic::{
    AddOperation, DivOperation, MulOperation, NegOperation, ScaleOperation, SubOperation,
};
use crate::operations::compare::CompareOperation;
use crate::operations::constants::{
    ConstantOperation, MaybeZeroOperation, OneLikeOperation, OneOperation, ZeroLike, ZeroLikeOperation, ZeroOperation,
};
use crate::operations::control_flow::{SelectOperation, WhileOperation};
use crate::operations::differentiation::StopGradientOperation;
use crate::operations::trigonometric::{CosOperation, SinOperation};
use crate::parameters::Parameterized;
use crate::payloads::Input;
use crate::programs::{Program, ProgramError, Value};
use crate::tracing::Tracer;
use crate::tracing_v2::differentiation::{
    CaptureParameterizedOperation, JvpTracer, LinearOperationOf, LinearizationContextOf, NestedLinearization,
    TangentContext,
};
use crate::tracing_v2::operations::MaybeDot;
use crate::tracing_v2::operations::custom_derivatives::{
    CustomJvpOperation, CustomVjpCallOperation, CustomVjpOperation,
};
use crate::tracing_v2::operations::select::LinearSelectOperation;
use crate::tracing_v2::rematerialization::{MaybeRematerializationName, RematerializationNameOperation};
use crate::tracing_v2::{
    DifferentiableOperation, DifferentiationContext, DotDimensionNumbers, LinearizableProgramOperation,
    ResidualizedOperation, ValueOrCapture,
};
use crate::types::DataType;

// TODO(eaplatanios): Review this file.

/// Closed scalar operation type for ordinary staged scalar programs.
///
/// [`ScalarOperation`] is intentionally limited to operations that are valid for scalar [`DataType`] metadata.
/// Array-only primitives such as reshaping and matrix multiplication remain available as standalone operations and
/// through array-based backends, but they are not variants of this enum.
#[derive(Clone, Debug, Operation)]
#[ryft(bounds(interpretation(BooleanLike)))]
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
    While(Box<WhileOperation<DataType, V, Self>>),
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
    D::Value: ZeroLike + BooleanLike,
    ScalarOperation<V>: Clone + LinearizableProgramOperation<D>,
    LinearOperationOf<D>: From<AddOperation>
        + From<ZeroLikeOperation>
        + From<NegOperation>
        + From<SubOperation>
        + From<ScaleOperation<DataType, ValueOrCapture<DataType, D::Value>, Input>>
        + From<LinearSelectOperation<ValueOrCapture<DataType, D::Value>>>
        + ResidualizedOperation<D>
        + From<CustomVjpCallOperation<DataType, V, ScalarOperation<V>, ValueOrCapture<DataType, D::Value>>>,
    LinearOperationOf<D>: CaptureParameterizedOperation<
            DataType,
            ValueOrCapture<DataType, D::Value>,
            WithCapture<ValueOrCapture<DataType, D::Value>> = LinearOperationOf<D>,
        >,
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
            Self::While(operation) => operation.jvp(context, inputs),
            Self::StopGradient(operation) => operation.jvp(context, inputs),
            Self::RematerializationName(operation) => operation.jvp(context, inputs),
            Self::CustomJvp(operation) => operation.jvp(context, inputs),
            Self::CustomVjp(operation) => operation.jvp(context, inputs),
        }
    }
}

// TODO(eaplatanios): Should this be generated for all `DifferentiableOperation`s? Why do we need it?
/// Nested symbolic linearization for the [`ScalarOperation`] sum type.
///
/// The where clauses spell the leaf closure required by
/// [`Program::linearize`] instead of the recursive
/// `Self: DifferentiableOperation<LinearizationContextOf<E, Self>>` bound, which avoids pushing that recursive
/// obligation back into every consumer.
impl<F, E> LinearizableProgramOperation<E> for ScalarOperation<F>
where
    F: Value<DataType>,
    E: DifferentiationContext<Type = DataType, Constant = F>,
    E::LinearOperation<E::Tangent, F>:
        CaptureParameterizedOperation<DataType, F, WithCapture<F> = E::LinearOperation<E::Tangent, F>>,
    LinearOperationOf<LinearizationContextOf<E, Self>>: From<AddOperation>
        + From<ZeroLikeOperation>
        + From<NegOperation>
        + From<SubOperation>
        + From<ScaleOperation<DataType, ValueOrCapture<DataType, Tracer<LinearizationContextOf<E, Self>>>, Input>>
        + ResidualizedOperation<LinearizationContextOf<E, Self>>
        + From<ZeroOperation<DataType>>
        + From<LinearSelectOperation<ValueOrCapture<DataType, Tracer<LinearizationContextOf<E, Self>>>>>
        + From<
            CustomVjpCallOperation<
                DataType,
                F,
                Self,
                ValueOrCapture<DataType, Tracer<LinearizationContextOf<E, Self>>>,
            >,
        >,
    LinearOperationOf<LinearizationContextOf<E, Self>>: CaptureParameterizedOperation<
            DataType,
            ValueOrCapture<DataType, Tracer<LinearizationContextOf<E, Self>>>,
            WithCapture<ValueOrCapture<DataType, Tracer<LinearizationContextOf<E, Self>>>> = LinearOperationOf<
                LinearizationContextOf<E, Self>,
            >,
            WithCapture<ValueOrCapture<DataType, E::Value>> = LinearOperationOf<E>,
        > + MaybeZeroOperation<DataType>,
{
    fn linearize_program(
        differentiable: &E,
        program: &Program<DataType, F, Self, Vec<F>, Vec<F>>,
    ) -> Result<NestedLinearization<E, Self>, ProgramError> {
        program.linearize(differentiable)
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
/// ([`LinearSelectOperation`]), linearized [`While`](Self::While) payloads, and the opaque
/// [`CustomVjpCall`](Self::CustomVjpCall) staged by a `custom_vjp` linearization (its transpose replays the user's
/// backward program).
#[derive(Clone, Debug, Operation, TransposableOperation)]
#[ryft(bounds(interpretation(BooleanLike)))]
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
    While(Box<WhileOperation<DataType, V, Self, Input>>),
    CustomVjpCall(Box<CustomVjpCallOperation<DataType, C, ScalarOperation<C>, F>>),
}

impl<V: Value<DataType>, C: Value<DataType>, F: Value<DataType>> CaptureParameterizedOperation<DataType, F>
    for LinearScalarOperation<V, C, F>
{
    type WithCapture<MappedFactor: Value<DataType>> = LinearScalarOperation<V, C, MappedFactor>;

    fn try_map_captures<MappedFactor: Value<DataType>, MapFactorFn>(
        &self,
        map_factor: &mut MapFactorFn,
    ) -> Result<Self::WithCapture<MappedFactor>, ProgramError>
    where
        MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>,
    {
        match self {
            Self::Zero(operation) => Ok(operation.clone().into()),
            Self::ZeroLike(operation) => Ok(operation.clone().into()),
            Self::One(operation) => Ok(operation.clone().into()),
            Self::OneLike(operation) => Ok(operation.clone().into()),
            Self::Constant(operation) => Ok(operation.clone().into()),
            Self::Neg(operation) => Ok(operation.clone().into()),
            Self::Add(operation) => Ok(operation.clone().into()),
            Self::Sub(operation) => Ok(operation.clone().into()),
            Self::Scale(operation) => {
                Ok(ScaleOperation::<DataType, MappedFactor, Input>::new(map_factor(operation.factor())?).into())
            }
            Self::Select(operation) => Ok(LinearSelectOperation::new(map_factor(operation.condition())?).into()),
            Self::While(operation) => {
                let condition = operation
                    .condition()
                    .map_operations(|operation| operation.try_map_captures::<MappedFactor, _>(map_factor))?;
                let body = operation
                    .body()
                    .map_operations(|operation| operation.try_map_captures::<MappedFactor, _>(map_factor))?;
                Ok(LinearScalarOperation::While(Box::new(
                    WhileOperation::new(condition, body)?.with_iteration_bound(operation.iteration_bound())?,
                )))
            }
            Self::CustomVjpCall(call) => {
                Ok(LinearScalarOperation::CustomVjpCall(Box::new(call.map_captures(map_factor)?)))
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

impl<V: Value<DataType>> MaybeDot for ScalarOperation<V> {
    #[inline]
    fn dot_dimensions(&self) -> Option<&DotDimensionNumbers> {
        None
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::StagingContext;
    use crate::operations::compare::{Compare, ComparisonDirection};
    use crate::operations::control_flow::Select;
    use crate::operations::{InterpretableOperation, Operation};
    use crate::parameters::Placeholder;
    use crate::programs::{Program, ProgramBuilder};
    use crate::scalars::ScalarDomain;
    use crate::tracing::trace;
    use crate::tracing_v2::DifferentiationContext;
    use crate::types::TypeError;

    use super::*;

    /// Builds a carry-only scalar body program that maps `[carry]` to `[carry + carry]`.
    fn scalar_doubling_body() -> Program<DataType, f64, ScalarOperation<f64>, Vec<f64>, Vec<f64>> {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let carry = builder.add_input(DataType::F64);
        let doubled = builder.add_instruction(AddOperation, vec![carry, carry]).unwrap()[0];
        builder.build(vec![doubled], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a scalar while condition that maps `[carry]` to `[carry < 8]`.
    fn scalar_less_than_eight_condition() -> Program<DataType, f64, ScalarOperation<f64>, Vec<f64>, Vec<f64>> {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let carry = builder.add_input(DataType::F64);
        let eight = builder.add_constant(8.0);
        let predicate = builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), vec![carry, eight])
            .unwrap()[0];
        builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap()
    }

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

    #[test]
    fn test_scalar_while() {
        let operation = WhileOperation::<DataType, f64, ScalarOperation<f64>>::new(
            scalar_less_than_eight_condition(),
            scalar_doubling_body(),
        )
        .unwrap();

        assert_eq!(operation.name(), crate::operations::control_flow::WHILE_OPERATION_NAME);
        assert_eq!(operation.state_types(), vec![DataType::F64]);
        assert_eq!(operation.iteration_bound(), None);
        assert_eq!(operation.infer_output_types(&[DataType::F64]), Ok(vec![DataType::F64]));
        assert_eq!(operation.interpret(&crate::EagerContext::new(), &[1.0]), Ok(vec![8.0]));

        let domain = ScalarDomain::<f64>::new();
        let (output_type, program) = trace(
            &domain,
            |carry| {
                let operation = WhileOperation::<DataType, f64, ScalarOperation<f64>>::new(
                    scalar_less_than_eight_condition(),
                    scalar_doubling_body(),
                )
                .unwrap();
                let mut outputs = carry.context().stage_operation(operation, &[&carry])?;
                Ok(outputs.remove(0))
            },
            DataType::F64,
        )
        .unwrap();
        assert_eq!(output_type, DataType::F64);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = while [
                    condition={
                        lambda %0:f64 .
                        let %1:f64 = const
                            %2:bool = compare [direction=LessThan] %0 %1
                        in (%2)
                    },
                    body={
                        lambda %0:f64 .
                        let %1:f64 = add %0 %0
                        in (%1)
                    },
                ] %0
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(program.interpret(1.0), Ok(8.0));

        let (primal, tangent): (f64, f64) = domain
            .jvp(
                |carry| {
                    let operation = WhileOperation::<DataType, f64, ScalarOperation<f64>>::new(
                        scalar_less_than_eight_condition(),
                        scalar_doubling_body(),
                    )
                    .unwrap();
                    carry.unary(operation)
                },
                1.0,
                1.0,
            )
            .unwrap();
        assert_eq!(primal, 8.0);
        assert_eq!(tangent, 8.0);

        let (_, pushforward) = domain
            .linearize(
                |carry| {
                    let operation = WhileOperation::<DataType, f64, ScalarOperation<f64>>::new(
                        scalar_less_than_eight_condition(),
                        scalar_doubling_body(),
                    )
                    .unwrap();
                    Ok(carry.unary(operation))
                },
                1.0,
            )
            .unwrap();
        assert_eq!(pushforward.apply(&crate::EagerContext::new(), 1.0), Ok(8.0));
    }

    #[test]
    fn test_scalar_while_rejects_non_boolean_condition() {
        assert_eq!(
            WhileOperation::<DataType, f64, ScalarOperation<f64>>::new(scalar_doubling_body(), scalar_doubling_body())
                .map(|_| ()),
            Err(TypeError { message: "while condition output type must be bool, but got f64".to_string() }),
        );
    }
}
