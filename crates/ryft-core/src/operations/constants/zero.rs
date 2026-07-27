use std::fmt::Display;

use crate::batching::{ArrayBatch, BatchAxis, BatchingContext, BatchingTracer};
use crate::contexts::{Context, Domain, ProjectedContext, StagingContext};
use crate::differentiation::forward::{DifferentiationContext, DifferentiationDual, DifferentiationTracer};
use crate::differentiation::types::DifferentiableType;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{
    check_count, impl_non_differentiable_operation, impl_nullary_batchable_operation,
    impl_nullary_transposable_operation,
};
use crate::partial::{PartialEvaluationContext, PartialTracer, PartiallyEvaluatableOperation};
use crate::programs::ProgramError;
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::operations::{Operation, OperationFormatter, OperationProjection};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::{Value, ValueProjection};
use crate::tracing::{Tracer, TracingContext};
use crate::types::ArrayType;

/// Canonical operation name for [`ZeroOperation`].
pub const ZERO_OPERATION_NAME: &str = "zero";

/// [`Operation`] that has no inputs and that produces a single output that corresponds to the _zero_ value for the
/// [`Type`] that it holds (i.e., for its `r#type` field). Note that for arrays, this would typically correspond to an
/// array of the right type and shape filled with zeros.
#[derive(Clone, Debug)]
pub struct ZeroOperation<T: Type> {
    /// [`Type`] of the value produced when this operation is interpreted.
    r#type: T,
}

impl<T: Type> ZeroOperation<T> {
    /// Creates a new [`ZeroOperation`].
    #[inline]
    pub fn new(r#type: T) -> Self {
        Self { r#type }
    }

    /// Returns the type of the value produced by this operation.
    #[inline]
    pub fn r#type(&self) -> &T {
        &self.r#type
    }
}

impl<T: Type> Display for ZeroOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type> Operation<T> for ZeroOperation<T> {
    #[inline]
    fn name(&self) -> &'static str {
        ZERO_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[T],
        _region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 0, TypeError);
        Ok(vec![self.r#type.clone()])
    }

    #[inline]
    fn is_zero(&self, output_index: usize) -> bool {
        output_index == 0
    }

    #[inline]
    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<T::Identity>) -> Result<Self, TypeError> {
        Ok(Self { r#type: self.r#type.rename_identities(renaming)? })
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, ZERO_OPERATION_NAME)?
            .bracketed(|operation| operation.field("type", &self.r#type))
    }
}

impl<T: Type, C: Domain<Type = T> + Zero<C::Value>> InterpretableOperation<C> for ZeroOperation<T> {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 0, ProgramError);
        Ok(vec![context.zero(&self.r#type)?])
    }
}

impl<T: Type, C: Context<Type = T, Operation: From<ZeroOperation<T>>>> PartiallyEvaluatableOperation<C>
    for ZeroOperation<T>
{
}

impl_non_differentiable_operation!(ZeroOperation<C::Type>);
impl_nullary_transposable_operation!(ZeroOperation<T>);
impl_nullary_batchable_operation!(@replicated ZeroOperation<ArrayType>);

/// Represents the ability to synthesize a _zero_ value for a given [`Type`] in an interpretation context. [`Zero`]
/// is the [`Type`]-driven counterpart to [`ZeroLike`](super::ZeroLike). It is what [`ZeroOperation`] needs for its
/// [`InterpretableOperation`] implementation, and it lives on the context because producing an eager value can be
/// backend- or context-dependent.
pub trait Zero<V: Typed> {
    /// Returns a _zero_ value for the provided [`Type`].
    fn zero(&self, r#type: &V::Type) -> Result<V, ProgramError>;
}

impl<C: Context, T: Type> Zero<<C::Value as ValueProjection<T>>::Projected> for ProjectedContext<C, T>
where
    C::Value: ValueProjection<T, Projected: Value<Type = T>>,
    C::Constant: ValueProjection<T, Projected: Value<Type = T>>,
    C::Operation: OperationProjection<T, Projected: From<ZeroOperation<T>>>,
{
    #[inline]
    fn zero(&self, r#type: &T) -> Result<<C::Value as ValueProjection<T>>::Projected, ProgramError> {
        Ok(self.bind(ZeroOperation::new(r#type.clone()), Vec::new(), &[])?.remove(0))
    }
}

impl<C: StagingContext<Operation: From<ZeroOperation<C::Type>>>> Zero<Tracer<C>> for C {
    #[inline]
    fn zero(&self, r#type: &C::Type) -> Result<Tracer<C>, ProgramError> {
        let mut outputs = self.stage_nullary_operation(ZeroOperation::new(r#type.clone()))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Context> Zero<PartialTracer<C>> for PartialEvaluationContext<C>
where
    C::Operation: PartiallyEvaluatableOperation<C>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + From<ZeroOperation<C::Type>>,
{
    #[inline]
    fn zero(&self, r#type: &C::Type) -> Result<PartialTracer<C>, ProgramError> {
        let mut outputs = self.bind(ZeroOperation::new(r#type.clone()), Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Context<Type = ArrayType> + Zero<C::Value>> Zero<BatchingTracer<C>> for BatchingContext<C> {
    #[inline]
    fn zero(&self, r#type: &ArrayType) -> Result<BatchingTracer<C>, ProgramError> {
        let batch = ArrayBatch::new(r#type.clone(), self.parent().zero(r#type)?, BatchAxis::replicated())?;
        Ok(BatchingTracer::new(self.clone(), batch))
    }
}

impl<C: Context<Type: DifferentiableType> + Zero<C::Value>> Zero<DifferentiationTracer<C>>
    for DifferentiationContext<C>
{
    #[inline]
    fn zero(&self, r#type: &C::Type) -> Result<DifferentiationTracer<C>, ProgramError> {
        let dual = DifferentiationDual::new_with_zero_tangent(self.parent().zero(r#type)?);
        Ok(DifferentiationTracer::new(dual, self.clone()))
    }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::Array;
    use crate::backends::scalars::Scalar;
    use crate::batching::{ArrayBatch, BatchAxis, BatchableOperation, BatchingContext};
    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::operations::constants::ConstantOperation;
    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::operations::Operation;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::types::DataType;

    use super::*;

    #[test]
    fn test_zero() {
        // Verify canonical zero values across every supported scalar data-type family.
        let context = EagerContext::<Scalar, ZeroOperation<DataType>>::new();
        for (r#type, expected) in [
            (DataType::Boolean, Scalar::from(false)),
            (DataType::I8, Scalar::from(0i8)),
            (DataType::I16, Scalar::from(0i16)),
            (DataType::I32, Scalar::from(0i32)),
            (DataType::I64, Scalar::from(0i64)),
            (DataType::U8, Scalar::from(0u8)),
            (DataType::U16, Scalar::from(0u16)),
            (DataType::U32, Scalar::from(0u32)),
            (DataType::U64, Scalar::from(0u64)),
            (DataType::BF16, Scalar::from(bf16::ZERO)),
            (DataType::F16, Scalar::from(f16::ZERO)),
            (DataType::F32, Scalar::from(0.0f32)),
            (DataType::F64, Scalar::from(0.0f64)),
        ] {
            assert_eq!(context.zero(&r#type), Ok(expected));
        }

        // Verify the operation's stored type, identity, zero metadata, rendering, and eager interpretation.
        let operation = ZeroOperation::new(DataType::F64);
        assert_eq!(operation.name(), ZERO_OPERATION_NAME);
        assert!(Operation::<DataType>::is_zero(&operation, 0));
        assert!(!Operation::<DataType>::is_zero(&operation, 1));
        assert_eq!(format!("{operation}"), "zero [type=f64]");
        assert_eq!(operation.r#type(), &DataType::F64);
        assert_eq!(operation.infer_output_types(&[], &[]), Ok(vec![DataType::F64]));
        assert_eq!(
            InterpretableOperation::<EagerContext<Scalar>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[]
            ),
            Ok(vec![Scalar::from(0.0)]),
        );

        // A nullary zero does not acquire a physical batch axis because the same value serves every batch item.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let outputs: Vec<ArrayBatch<Array>> = ZeroOperation::new(scalar_type.clone())
            .batch(
                &BatchingContext::new(EagerContext::<Array, ConstantOperation<Array>>::new(), 2),
                &EmptyRegionDriver,
                &[],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].r#type().into_owned(), scalar_type);
        assert_eq!(outputs[0].value().to_f64s(), vec![0.0]);

        // Verify the operation's textual form when it appears in a program.
        let mut builder = ProgramBuilder::<Scalar, ZeroOperation<DataType>>::new();
        let output = builder.add_instruction(operation, Vec::new(), vec![]).unwrap()[0];
        let program = builder.build::<(), Scalar>(vec![output], (), Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f64 = zero [type=f64]
                in (%0)
            "}
            .trim_end(),
        );
    }
}
