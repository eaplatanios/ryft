use half::{bf16, f16};

use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::constants::{
    OneLike, OneLikeOperation, OneOperation, SupportsZero, SupportsZeroLike, ZeroLike, ZeroLikeOperation, ZeroOperation,
};
use crate::tracing::transposition::{LinearOperation, TranspositionContext};
use crate::tracing::{AtomId, Traceable, TracingError};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableOperation, LinearizableEngine};
use crate::types::{DataType, Type};

impl<T: Type, V: Traceable<T>, LinearCarrier: Clone + Operation<T>> LinearOperation<T, V, LinearCarrier>
    for ZeroOperation<T>
{
    fn transpose(
        &self,
        _context: &mut TranspositionContext<T, V, LinearCarrier>,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        Ok(Vec::new())
    }
}

impl<E> DifferentiableOperation<E> for ZeroOperation<E::Type>
where
    E: LinearizableEngine,
    ZeroOperation<E::Type>: Operation<E::Type>,
    E::Value: Differentiable<E::Type, Tangent = E::Value>,
    E::LinearOperationCarrier: SupportsZero<E::Type, E::Value>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_count!("input", inputs, 0, TracingError);
        let tangent_outputs = context.stage(
            <E::LinearOperationCarrier as SupportsZero<E::Type, E::Value>>::zero_operation(self.r#type.clone()),
            &[],
        )?;
        check_count!("output", tangent_outputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: context.engine.zero(&self.r#type)?, tangent: tangent_outputs[0] }])
    }
}

impl<T: Type, V: Traceable<T>, LinearCarrier: Clone + Operation<T>> LinearOperation<T, V, LinearCarrier>
    for OneOperation<T>
{
    fn transpose(
        &self,
        _context: &mut TranspositionContext<T, V, LinearCarrier>,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        Ok(Vec::new())
    }
}

impl<E> DifferentiableOperation<E> for OneOperation<E::Type>
where
    E: LinearizableEngine,
    OneOperation<E::Type>: Operation<E::Type>,
    E::Value: Differentiable<E::Type, Tangent = E::Value>,
    E::LinearOperationCarrier: SupportsZero<E::Type, E::Value>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_count!("input", inputs, 0, TracingError);
        let tangent_outputs = context.stage(
            <E::LinearOperationCarrier as SupportsZero<E::Type, E::Value>>::zero_operation(self.r#type.clone()),
            &[],
        )?;
        check_count!("output", tangent_outputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: context.engine.one(&self.r#type)?, tangent: tangent_outputs[0] }])
    }
}

impl<T: Type, V: Traceable<T>, LinearCarrier: Clone + Operation<T>> LinearOperation<T, V, LinearCarrier>
    for ZeroLikeOperation
{
    fn transpose(
        &self,
        _context: &mut TranspositionContext<T, V, LinearCarrier>,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        Ok(vec![None])
    }
}

impl<E> DifferentiableOperation<E> for ZeroLikeOperation
where
    E: LinearizableEngine,
    ZeroLikeOperation: Operation<E::Type>,
    E::Value: ZeroLike + Differentiable<E::Type, Tangent = E::Value>,
    E::LinearOperationCarrier: SupportsZeroLike<E::Type, E::Value>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        let tangent_outputs = context.stage(
            <E::LinearOperationCarrier as SupportsZeroLike<E::Type, E::Value>>::zero_like_operation(),
            &[inputs[0].tangent],
        )?;
        check_count!("output", tangent_outputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: inputs[0].primal.zero_like(), tangent: tangent_outputs[0] }])
    }
}

impl<T: Type, V: Traceable<T>, LinearCarrier: Clone + Operation<T>> LinearOperation<T, V, LinearCarrier>
    for OneLikeOperation
{
    fn transpose(
        &self,
        _context: &mut TranspositionContext<T, V, LinearCarrier>,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        Ok(vec![None])
    }
}

impl<E> DifferentiableOperation<E> for OneLikeOperation
where
    E: LinearizableEngine,
    OneLikeOperation: Operation<E::Type>,
    E::Value: OneLike + Differentiable<E::Type, Tangent = E::Value>,
    E::LinearOperationCarrier: SupportsZeroLike<E::Type, E::Value>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        let tangent_outputs = context.stage(
            <E::LinearOperationCarrier as SupportsZeroLike<E::Type, E::Value>>::zero_like_operation(),
            &[inputs[0].tangent],
        )?;
        check_count!("output", tangent_outputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: inputs[0].primal.one_like(), tangent: tangent_outputs[0] }])
    }
}

macro_rules! impl_scalar_differentiable {
    ($ty:ty) => {
        impl crate::tracing_v2::differentiation::Differentiable<DataType> for $ty {
            type Tangent = Self;
        }
    };
}

impl_scalar_differentiable!(bool);
impl_scalar_differentiable!(i8);
impl_scalar_differentiable!(i16);
impl_scalar_differentiable!(i32);
impl_scalar_differentiable!(i64);
impl_scalar_differentiable!(u8);
impl_scalar_differentiable!(u16);
impl_scalar_differentiable!(u32);
impl_scalar_differentiable!(u64);
impl_scalar_differentiable!(bf16);
impl_scalar_differentiable!(f16);
impl_scalar_differentiable!(f32);
impl_scalar_differentiable!(f64);

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;

    use crate::tracing::Program;
    use crate::tracing::engines::{ScalarEngine, TracingEngine};
    use crate::tracing_v2::{Cos, Differentiable, ScalarOperation, Sin};
    use crate::types::DataType;

    #[test]
    fn test_scalar_types_are_differentiable() {
        let _: Option<<bool as Differentiable<DataType>>::Tangent> = None;
        let _: Option<<i8 as Differentiable<DataType>>::Tangent> = None;
        let _: Option<<i16 as Differentiable<DataType>>::Tangent> = None;
        let _: Option<<i32 as Differentiable<DataType>>::Tangent> = None;
        let _: Option<<i64 as Differentiable<DataType>>::Tangent> = None;
        let _: Option<<u8 as Differentiable<DataType>>::Tangent> = None;
        let _: Option<<u16 as Differentiable<DataType>>::Tangent> = None;
        let _: Option<<u32 as Differentiable<DataType>>::Tangent> = None;
        let _: Option<<u64 as Differentiable<DataType>>::Tangent> = None;
        let _: Option<<bf16 as Differentiable<DataType>>::Tangent> = None;
        let _: Option<<f16 as Differentiable<DataType>>::Tangent> = None;
        let _: Option<<f32 as Differentiable<DataType>>::Tangent> = None;
        let _: Option<<f64 as Differentiable<DataType>>::Tangent> = None;
    }

    #[test]
    fn float_ext_matches_scalar_intrinsics() {
        let angle = 0.75f64;
        assert_eq!(Sin::sin(angle), angle.sin());
        assert_eq!(Cos::cos(angle), angle.cos());

        let engine = ScalarEngine::<f64>::new();
        let (_, compiled): (f64, Program<DataType, f64, ScalarOperation<f64>, f64, f64>) =
            engine.interpret_and_trace(|x| Ok(x.sin()), 2.0f64).unwrap();

        assert_eq!(
            compiled.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = sin %0
                in (%1)
            "}
            .trim_end(),
        );
    }
}
