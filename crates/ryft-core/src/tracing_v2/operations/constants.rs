use std::convert::Infallible;

use half::{bf16, f16};

use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::constants::{
    OneLike, OneLikeOperation, OneOperation, SupportsZero, SupportsZeroLike, ZeroLike, ZeroLikeOperation, ZeroOperation,
};
use crate::tracing::transposition::{LinearOperation, TranspositionContext};
use crate::tracing::{AtomId, Traceable, TracingError};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer, Tangent};
use crate::tracing_v2::{DifferentiableEngine, DifferentiableOperation};
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
    E: DifferentiableEngine,
    ZeroOperation<E::Type>: Operation<E::Type>,
    E::Value: Differentiable<E::Type>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_count!("input", inputs, 0, TracingError);
        let tangent_outputs = context.stage(
            <<E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier as SupportsZero<
                E::Type,
                E::Tangent,
            >>::zero_operation(self.r#type.clone()),
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
    E: DifferentiableEngine,
    OneOperation<E::Type>: Operation<E::Type>,
    E::Value: Differentiable<E::Type>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_count!("input", inputs, 0, TracingError);
        let tangent_outputs = context.stage(
            <<E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier as SupportsZero<
                E::Type,
                E::Tangent,
            >>::zero_operation(self.r#type.clone()),
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
    E: DifferentiableEngine,
    ZeroLikeOperation: Operation<E::Type>,
    E::Value: ZeroLike + Differentiable<E::Type>,
    <E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier:
        SupportsZeroLike<E::Type, E::Tangent>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        let tangent_outputs = context.stage(
            <<E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier as SupportsZeroLike<
                E::Type,
                E::Tangent,
            >>::zero_like_operation(),
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
    E: DifferentiableEngine,
    OneLikeOperation: Operation<E::Type>,
    E::Value: OneLike + Differentiable<E::Type>,
    <E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier:
        SupportsZeroLike<E::Type, E::Tangent>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        let tangent_outputs = context.stage(
            <<E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier as SupportsZeroLike<
                E::Type,
                E::Tangent,
            >>::zero_like_operation(),
            &[inputs[0].tangent],
        )?;
        check_count!("output", tangent_outputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: inputs[0].primal.one_like(), tangent: tangent_outputs[0] }])
    }
}

macro_rules! impl_nondifferentiable_scalar {
    ($ty:ty, $data_type:path) => {
        impl crate::tracing_v2::differentiation::Differentiable<DataType> for $ty {
            type Tangent = Tangent<DataType, Infallible>;

            #[inline]
            fn tangent_type(&self) -> Result<Self::Tangent, TracingError> {
                Ok(Tangent::zero($data_type))
            }
        }
    };
}

macro_rules! impl_floating_scalar_differentiable {
    ($ty:ty, $zero:expr) => {
        impl crate::tracing_v2::differentiation::Differentiable<DataType> for $ty {
            type Tangent = Self;

            #[inline]
            fn tangent_type(&self) -> Result<Self::Tangent, TracingError> {
                Ok($zero)
            }
        }
    };
}

impl_nondifferentiable_scalar!(bool, DataType::Boolean);
impl_nondifferentiable_scalar!(i8, DataType::I8);
impl_nondifferentiable_scalar!(i16, DataType::I16);
impl_nondifferentiable_scalar!(i32, DataType::I32);
impl_nondifferentiable_scalar!(i64, DataType::I64);
impl_nondifferentiable_scalar!(u8, DataType::U8);
impl_nondifferentiable_scalar!(u16, DataType::U16);
impl_nondifferentiable_scalar!(u32, DataType::U32);
impl_nondifferentiable_scalar!(u64, DataType::U64);
impl_floating_scalar_differentiable!(bf16, bf16::ZERO);
impl_floating_scalar_differentiable!(f16, f16::ZERO);
impl_floating_scalar_differentiable!(f32, 0.0);
impl_floating_scalar_differentiable!(f64, 0.0);

#[cfg(test)]
mod tests {
    use std::any::TypeId;
    use std::convert::Infallible;

    use half::{bf16, f16};
    use indoc::indoc;

    use crate::tracing::Program;
    use crate::tracing::engines::{ScalarEngine, TracingEngine};
    use crate::tracing_v2::{Cos, Differentiable, ScalarOperation, Sin, Tangent};
    use crate::types::DataType;

    #[test]
    fn test_scalar_types_are_differentiable() {
        assert_eq!(
            TypeId::of::<<bool as Differentiable<DataType>>::Tangent>(),
            TypeId::of::<Tangent<DataType, Infallible>>()
        );
        assert_eq!(
            TypeId::of::<<i8 as Differentiable<DataType>>::Tangent>(),
            TypeId::of::<Tangent<DataType, Infallible>>()
        );
        assert_eq!(
            TypeId::of::<<i16 as Differentiable<DataType>>::Tangent>(),
            TypeId::of::<Tangent<DataType, Infallible>>()
        );
        assert_eq!(
            TypeId::of::<<i32 as Differentiable<DataType>>::Tangent>(),
            TypeId::of::<Tangent<DataType, Infallible>>()
        );
        assert_eq!(
            TypeId::of::<<i64 as Differentiable<DataType>>::Tangent>(),
            TypeId::of::<Tangent<DataType, Infallible>>()
        );
        assert_eq!(
            TypeId::of::<<u8 as Differentiable<DataType>>::Tangent>(),
            TypeId::of::<Tangent<DataType, Infallible>>()
        );
        assert_eq!(
            TypeId::of::<<u16 as Differentiable<DataType>>::Tangent>(),
            TypeId::of::<Tangent<DataType, Infallible>>()
        );
        assert_eq!(
            TypeId::of::<<u32 as Differentiable<DataType>>::Tangent>(),
            TypeId::of::<Tangent<DataType, Infallible>>()
        );
        assert_eq!(
            TypeId::of::<<u64 as Differentiable<DataType>>::Tangent>(),
            TypeId::of::<Tangent<DataType, Infallible>>()
        );
        assert_eq!(TypeId::of::<<bf16 as Differentiable<DataType>>::Tangent>(), TypeId::of::<bf16>());
        assert_eq!(TypeId::of::<<f16 as Differentiable<DataType>>::Tangent>(), TypeId::of::<f16>());
        assert_eq!(TypeId::of::<<f32 as Differentiable<DataType>>::Tangent>(), TypeId::of::<f32>());
        assert_eq!(TypeId::of::<<f64 as Differentiable<DataType>>::Tangent>(), TypeId::of::<f64>());
        assert_eq!(false.tangent_type().unwrap(), Tangent::zero(DataType::Boolean));
        assert_eq!(3i32.tangent_type().unwrap(), Tangent::zero(DataType::I32));
        assert_eq!(2.0f64.tangent_type().unwrap(), 0.0);
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
