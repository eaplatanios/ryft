use std::fmt::Display;

use crate::batching::{ArrayBatch, ArrayBatching, BatchAxis, BatchingContext, BatchingTracer};
use crate::contexts::{Context, Domain, ProjectedContext, StagingContext};
use crate::differentiation::forward::{DifferentiationContext, DifferentiationDual, DifferentiationTracer};
use crate::differentiation::types::DifferentiableType;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{
    check_count, impl_non_differentiable_operation, impl_nullary_batchable_operation,
    impl_nullary_transposable_operation,
};
use crate::operations::constants::check_constructor_type_has_no_identity_references;
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

impl<T: Type> Operation for ZeroOperation<T> {
    type Type = T;

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
        check_constructor_type_has_no_identity_references(ZERO_OPERATION_NAME, &self.r#type)?;
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

impl_non_differentiable_operation!(<T> ZeroOperation<T> where T: Type);
impl_nullary_transposable_operation!(<T> ZeroOperation<T> where T: Type);
impl_nullary_batchable_operation!(@replicated ZeroOperation<ArrayType>);

// TODO(eaplatanios): Restore the strict `Operation<Type = T>` super-trait bound once the next-generation trait solver
//  stabilizes. The current solver cannot discharge this projection equality at implementation heads whose context type
//  is built from `Self` (E0284); the equality is enforced per method through a `where` clause instead.
/// Supplies the canonical zero [`Operation`] of a program type's operation family. [`Self::zero_operation`] covers
/// zeros that can be constructed from a type without operands, which is all that staging and eager materialization
/// need. Differentiation additionally must materialize zeros whose runtime geometry is unavailable from the type alone
/// (e.g., disconnected cotangents with dynamic axes). That residual protocol is transform-owned and lives on
/// [`ResidualZeroProvider`](crate::ResidualZeroProvider).
///
/// The super-trait is plain [`Operation`] rather than `Operation<Type = T>` because the current trait solver cannot
/// discharge that projection equality where this provider is requested through a context's operation family, which is
/// how every transform requests it. The equality is instead required by [`zero_operation`](Self::zero_operation)
/// itself, so a provider whose [`Operation::Type`] disagrees with `T` cannot construct anything: the requirement is
/// restated by the residual-zero protocol and by transform call sites, and any mismatched implementation is rejected
/// with a type-mismatch error there.
pub trait ZeroOperationProvider<T: Type>: Operation {
    /// Constructs an [`Operation`] that materializes a zero of `r#type` without operands.
    fn zero_operation(r#type: T) -> Result<Self, ProgramError>
    where
        Self: Operation<Type = T>;
}

impl<T: Type, O: Operation<Type = T> + From<ZeroOperation<T>>> ZeroOperationProvider<T> for O {
    #[inline]
    fn zero_operation(r#type: T) -> Result<Self, ProgramError> {
        Ok(Self::from(ZeroOperation::new(r#type)))
    }
}

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

impl<C: StagingContext<Operation: ZeroOperationProvider<C::Type>>> Zero<Tracer<C>> for C {
    #[inline]
    fn zero(&self, r#type: &C::Type) -> Result<Tracer<C>, ProgramError> {
        let mut outputs = self.stage_nullary_operation(C::Operation::zero_operation(r#type.clone())?)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Context> Zero<PartialTracer<C>> for PartialEvaluationContext<C>
where
    C::Operation: PartiallyEvaluatableOperation<C>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + ZeroOperationProvider<C::Type>,
{
    #[inline]
    fn zero(&self, r#type: &C::Type) -> Result<PartialTracer<C>, ProgramError> {
        let mut outputs = self.bind(C::Operation::zero_operation(r#type.clone())?, Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Context<Type = ArrayType> + Zero<C::Value>> Zero<BatchingTracer<C, ArrayBatching>>
    for BatchingContext<C, ArrayBatching>
{
    #[inline]
    fn zero(&self, r#type: &ArrayType) -> Result<BatchingTracer<C, ArrayBatching>, ProgramError> {
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
    use crate::types::{DataType, Dimension, DimensionBounds, DimensionType, Shape};

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
        assert!(operation.is_zero(0));
        assert!(!operation.is_zero(1));
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

        // Nullary construction rejects output types with ungrounded identity *references* (a dynamic array axis),
        // which must instead be constructed through the mixed dimension-operand contract owned by the composite
        // operation family. Definition-position identities remain constructible: a dimension value's type defines
        // its own variable, so nullary construction leaves no dangling reference.
        let rows = crate::types::DimensionVariable::new("rows", DimensionBounds::non_negative(Some(8)).unwrap());
        let dynamic_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Static(3)]));
        assert_eq!(
            ZeroOperation::new(dynamic_type.clone()).infer_output_types(&[], &[]),
            Err(TypeError::invalid(
                "'zero' cannot construct type f32[rows, 3] without operands because it references identity rows",
            )),
        );
        let dimension_type = DimensionType::new(rows);
        assert_eq!(ZeroOperation::new(dimension_type.clone()).infer_output_types(&[], &[]), Ok(vec![dimension_type]),);

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
