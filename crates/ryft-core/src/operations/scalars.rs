use ryft_macros::{DifferentiableOperation, Operation, TransposableOperation};

use crate::operations::BooleanLike;
use crate::operations::arithmetic::{
    AbsOperation, AddOperation, DivOperation, MulOperation, NegOperation, SubOperation,
};
use crate::operations::compare::CompareOperation;
use crate::operations::complex::{ComplexOperation, ConjugateOperation, ImaginaryOperation, RealOperation};
use crate::operations::constants::{
    ConstantOperation, OneLikeOperation, OneOperation, ZeroLikeOperation, ZeroOperation,
};
use crate::operations::control_flow::{
    MaybeScan, MaybeWhile, SelectOperation, WhileOperation, WhileParts, WhilePredicate,
};
use crate::operations::debugging::PrintOperation;
use crate::operations::differentiation::StopGradientOperation;
use crate::operations::exponential::{ExponentialOperation, LogarithmOperation, SquareRootOperation};
use crate::operations::tag::{MaybeTag, TagOperation};
use crate::operations::trigonometric::{Atan2Operation, CosOperation, SinOperation};
use crate::programs::Value;
use crate::tracing_v2::DotDimensionNumbers;
use crate::tracing_v2::operations::MaybeDot;
use crate::tracing_v2::operations::custom_derivatives::{
    CustomJvpOperation, CustomVjpOperation, CustomVjpTangentOperation,
};
use crate::tracing_v2::rematerialization::RematerializeOperation;
use crate::types::DataType;

// TODO(eaplatanios): Review this file.

/// Closed scalar operation type for ordinary staged scalar programs.
///
/// [`ScalarOperation`] is intentionally limited to operations that are valid for scalar [`DataType`] metadata.
/// Array-only primitives such as reshaping and matrix multiplication remain available as standalone operations and
/// through array-based backends, but they are not variants of this enum.
#[derive(Clone, Debug, Operation, DifferentiableOperation, TransposableOperation)]
#[ryft(bounds(interpretation(BooleanLike + WhilePredicate)))]
pub enum ScalarOperation<V: Value<Type = DataType>> {
    Zero(ZeroOperation<DataType>),
    ZeroLike(ZeroLikeOperation),
    One(OneOperation<DataType>),
    OneLike(OneLikeOperation),
    Constant(ConstantOperation<V>),
    Abs(AbsOperation),
    Neg(NegOperation),
    Add(AddOperation),
    Sub(SubOperation),
    Mul(MulOperation),
    Div(DivOperation),
    Sin(SinOperation),
    Cos(CosOperation),
    Atan2(Atan2Operation),
    Exponential(ExponentialOperation),
    Logarithm(LogarithmOperation),
    SquareRoot(SquareRootOperation),
    Complex(ComplexOperation),
    Conjugate(ConjugateOperation),
    Real(RealOperation),
    Imaginary(ImaginaryOperation),
    Compare(CompareOperation),
    Select(SelectOperation),
    While(Box<WhileOperation<V, Self>>),
    StopGradient(StopGradientOperation),
    Tag(TagOperation),
    Print(PrintOperation),
    CustomJvp(Box<CustomJvpOperation<V, Self>>),
    CustomVjp(Box<CustomVjpOperation<V, Self>>),
    CustomVjpTangent(Box<CustomVjpTangentOperation<V, Self>>),
    Rematerialize(Box<RematerializeOperation<V, Self>>),
}

impl<V: Value<Type = DataType>> MaybeTag for ScalarOperation<V> {
    #[inline]
    fn key(&self) -> Option<&str> {
        match self {
            Self::Tag(operation) => Some(operation.key()),
            _ => None,
        }
    }
}

impl<V: Value<Type = DataType>> MaybeDot for ScalarOperation<V> {
    #[inline]
    fn dot_dimensions(&self) -> Option<&DotDimensionNumbers> {
        None
    }
}

impl<V: Value<Type = DataType>> MaybeWhile<V, ScalarOperation<V>> for ScalarOperation<V> {
    #[inline]
    fn as_while(&self) -> Option<WhileParts<'_, V, ScalarOperation<V>>> {
        match self {
            Self::While(operation) => operation.as_while(),
            _ => None,
        }
    }
}

/// [`ScalarOperation`] has no `scan` variant, so no residual ever needs scan-body provenance.
impl<V: Value<Type = DataType>> MaybeScan<V, ScalarOperation<V>> for ScalarOperation<V> {
    #[inline]
    fn scan_body(&self) -> Option<&crate::programs::Program<V, ScalarOperation<V>, Vec<V>, Vec<V>>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::Scalar;
    use crate::contexts::{Context, StagingContext};
    use crate::interpretation::InterpretableOperation;
    use crate::operations::Operation;
    use crate::operations::compare::{Compare, ComparisonDirection};
    use crate::operations::control_flow::Select;
    use crate::parameters::Placeholder;
    use crate::programs::{Program, ProgramBuilder};
    use crate::tracing::Trace;
    use crate::tracing_v2::ForwardModeDifferentiate;
    use crate::types::TypeError;

    use super::*;
    use crate::contexts::EagerContext;

    /// Builds a carry-only scalar body program that maps `[carry]` to `[carry + carry]`.
    fn scalar_doubling_body() -> Program<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>> {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let carry = builder.add_input(DataType::F64);
        let doubled = builder.add_instruction(AddOperation, vec![carry, carry]).unwrap()[0];
        builder.build(vec![doubled], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a scalar while condition that maps `[carry]` to `[carry < 8]`.
    fn scalar_less_than_eight_condition() -> Program<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>> {
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let carry = builder.add_input(DataType::F64);
        let eight = builder.add_constant(Scalar::from(8.0));
        let predicate = builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), vec![carry, eight])
            .unwrap()[0];
        builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_scalar_compare_and_select_program() {
        // `f(x, y) = select(x > y, x + x, y)` staged through `ScalarOperation` tracers.
        let (output_type, program) = EagerContext::<Scalar, ScalarOperation<Scalar>>::trace(
            |(x, y)| {
                let mask = x.clone().greater_than(&y)?;
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
        assert_eq!(program.interpret((Scalar::from(3.0), Scalar::from(2.0))), Ok(Scalar::from(6.0)));
        assert_eq!(program.interpret((Scalar::from(1.0), Scalar::from(2.0))), Ok(Scalar::from(2.0)));
    }

    #[test]
    fn test_scalar_while() {
        let operation = WhileOperation::<Scalar, ScalarOperation<Scalar>>::new(
            scalar_less_than_eight_condition(),
            scalar_doubling_body(),
        )
        .unwrap();

        assert_eq!(operation.name(), crate::operations::control_flow::WHILE_OPERATION_NAME);
        assert_eq!(operation.state_types(), vec![DataType::F64]);
        assert_eq!(operation.iteration_bound(), None);
        assert_eq!(operation.infer_output_types(&[DataType::F64]), Ok(vec![DataType::F64]));
        assert_eq!(
            operation.interpret(&crate::EagerContext::<Scalar>::new(), &[Scalar::from(1.0)]),
            Ok(vec![Scalar::from(8.0)])
        );

        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (output_type, program) = EagerContext::<Scalar, ScalarOperation<Scalar>>::trace(
            |carry| {
                let operation = WhileOperation::<Scalar, ScalarOperation<Scalar>>::new(
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
        assert_eq!(program.interpret(Scalar::from(1.0)), Ok(Scalar::from(8.0)));

        let (primal, tangent): (Scalar, Scalar) = domain
            .jvp(
                |carry| {
                    let operation = WhileOperation::<Scalar, ScalarOperation<Scalar>>::new(
                        scalar_less_than_eight_condition(),
                        scalar_doubling_body(),
                    )
                    .unwrap();
                    Ok(carry.context().bind(operation, &[carry.clone()])?.remove(0))
                },
                Scalar::from(1.0),
                Scalar::from(1.0),
            )
            .unwrap();
        assert_eq!(primal, 8.0);
        assert_eq!(tangent, 8.0);
    }

    #[test]
    fn test_scalar_while_rejects_non_boolean_condition() {
        assert_eq!(
            WhileOperation::<Scalar, ScalarOperation<Scalar>>::new(scalar_doubling_body(), scalar_doubling_body())
                .map(|_| ()),
            Err(TypeError { message: "'while' condition output type must be bool, but got f64".to_string() }),
        );
    }
}
