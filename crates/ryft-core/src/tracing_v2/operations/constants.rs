use std::convert::Infallible;

use half::{bf16, f16};

use crate::differentiation::{Cotangent, LinearOperation};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::constants::{
    OneLike, OneLikeOperation, OneOperation, SupportsZero, SupportsZeroLike, ZeroLike, ZeroLikeOperation, ZeroOperation,
};
use crate::parameters::Parameter;
use crate::tracing::domains::Tracer;
use crate::tracing::{ProgramTracingContext, Traceable, TracingError};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer, Tangent};
use crate::tracing_v2::{DifferentiableDomain, DifferentiableOperation};
use crate::types::{DataType, Type};

impl<T: Parameter + Type, V: Traceable<T>, LinearCarrier: Clone + Operation<T>> LinearOperation<T, V, LinearCarrier>
    for ZeroOperation<T>
{
    fn transpose<'transpose>(
        &self,
        _context: &mut ProgramTracingContext<'transpose, T, V, LinearCarrier>,
        output_cotangents: &[Cotangent<'transpose, T, V, LinearCarrier>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, LinearCarrier>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        Ok(Vec::new())
    }
}

impl<D> DifferentiableOperation<D> for ZeroOperation<D::Type>
where
    D: DifferentiableDomain,
    ZeroOperation<D::Type>: Operation<D::Type>,
    D::Value: Differentiable<D::Type>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<D::Value, Tracer<'jvp, D::LinearDomain>>],
    ) -> Result<Vec<JvpTracer<D::Value, Tracer<'jvp, D::LinearDomain>>>, TracingError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 0, TracingError);
        let mut tangent_outputs = context.stage(D::LinearOperationCarrier::zero_operation(self.r#type.clone()), &[])?;
        check_count!("output", tangent_outputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: context.domain.zero(&self.r#type)?, tangent: tangent_outputs.remove(0) }])
    }
}

impl<T: Parameter + Type, V: Traceable<T>, LinearCarrier: Clone + Operation<T>> LinearOperation<T, V, LinearCarrier>
    for OneOperation<T>
{
    fn transpose<'transpose>(
        &self,
        _context: &mut ProgramTracingContext<'transpose, T, V, LinearCarrier>,
        output_cotangents: &[Cotangent<'transpose, T, V, LinearCarrier>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, LinearCarrier>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        Ok(Vec::new())
    }
}

impl<D> DifferentiableOperation<D> for OneOperation<D::Type>
where
    D: DifferentiableDomain,
    OneOperation<D::Type>: Operation<D::Type>,
    D::Value: Differentiable<D::Type>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<D::Value, Tracer<'jvp, D::LinearDomain>>],
    ) -> Result<Vec<JvpTracer<D::Value, Tracer<'jvp, D::LinearDomain>>>, TracingError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 0, TracingError);
        let mut tangent_outputs = context.stage(D::LinearOperationCarrier::zero_operation(self.r#type.clone()), &[])?;
        check_count!("output", tangent_outputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: context.domain.one(&self.r#type)?, tangent: tangent_outputs.remove(0) }])
    }
}

impl<T: Parameter + Type, V: Traceable<T>, LinearCarrier: Clone + Operation<T>> LinearOperation<T, V, LinearCarrier>
    for ZeroLikeOperation
{
    fn transpose<'transpose>(
        &self,
        _context: &mut ProgramTracingContext<'transpose, T, V, LinearCarrier>,
        output_cotangents: &[Cotangent<'transpose, T, V, LinearCarrier>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, LinearCarrier>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        Ok(vec![Cotangent::Zero])
    }
}

impl<D> DifferentiableOperation<D> for ZeroLikeOperation
where
    D: DifferentiableDomain,
    ZeroLikeOperation: Operation<D::Type>,
    D::Value: ZeroLike + Differentiable<D::Type>,
    D::LinearOperationCarrier: SupportsZeroLike<D::Type, D::Tangent>,
{
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<D::Value, Tracer<'jvp, D::LinearDomain>>],
    ) -> Result<Vec<JvpTracer<D::Value, Tracer<'jvp, D::LinearDomain>>>, TracingError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: inputs[0].primal.zero_like(), tangent: inputs[0].tangent.zero_like() }])
    }
}

impl<T: Parameter + Type, V: Traceable<T>, LinearCarrier: Clone + Operation<T>> LinearOperation<T, V, LinearCarrier>
    for OneLikeOperation
{
    fn transpose<'transpose>(
        &self,
        _context: &mut ProgramTracingContext<'transpose, T, V, LinearCarrier>,
        output_cotangents: &[Cotangent<'transpose, T, V, LinearCarrier>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, LinearCarrier>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        Ok(vec![Cotangent::Zero])
    }
}

impl<D> DifferentiableOperation<D> for OneLikeOperation
where
    D: DifferentiableDomain,
    OneLikeOperation: Operation<D::Type>,
    D::Value: OneLike + Differentiable<D::Type>,
    D::LinearOperationCarrier: SupportsZeroLike<D::Type, D::Tangent>,
{
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<D::Value, Tracer<'jvp, D::LinearDomain>>],
    ) -> Result<Vec<JvpTracer<D::Value, Tracer<'jvp, D::LinearDomain>>>, TracingError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: inputs[0].primal.one_like(), tangent: inputs[0].tangent.zero_like() }])
    }
}

macro_rules! impl_nondifferentiable_scalar {
    ($ty:ty, $data_type:path) => {
        impl crate::tracing_v2::differentiation::Differentiable<DataType> for $ty {
            type Tangent = Tangent<DataType, Infallible>;

            #[inline]
            fn zero_tangent(&self) -> Result<Self::Tangent, TracingError> {
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
            fn zero_tangent(&self) -> Result<Self::Tangent, TracingError> {
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

    use crate::operations::scalars::ScalarOperation;
    use crate::operations::trigonometric::{Cos, Sin};
    use crate::tracing::Program;
    use crate::tracing::domains::{ScalarDomain, TracingDomain};
    use crate::tracing_v2::{Differentiable, Tangent};
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
        assert_eq!(false.zero_tangent().unwrap(), Tangent::zero(DataType::Boolean));
        assert_eq!(3i32.zero_tangent().unwrap(), Tangent::zero(DataType::I32));
        assert_eq!(2.0f64.zero_tangent().unwrap(), 0.0);
    }

    #[test]
    fn float_ext_matches_scalar_intrinsics() {
        let angle = 0.75f64;
        assert_eq!(Sin::sin(angle), angle.sin());
        assert_eq!(Cos::cos(angle), angle.cos());

        let domain = ScalarDomain::<f64>::new();
        let (_, compiled): (f64, Program<DataType, f64, ScalarOperation<f64>, f64, f64>) =
            domain.interpret_and_trace(|x| Ok(x.sin()), 2.0f64).unwrap();

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
