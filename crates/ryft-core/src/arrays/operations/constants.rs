//! Array IR instantiations of the constants operation family contracts.
//!
//! Nullary constructors are the one place where a purely structural output type is not enough: materializing a zero,
//! one, or iota whose shape has dynamic axes requires the concrete runtime extents. This module supplies the array
//! universe's answers to those contracts — the mixed dynamic constructors that consume one first-class dimension
//! operand per dynamic axis, the canonical lifts that route each constructor to its static or dynamic encoding, and
//! the residual-aware zero construction that differentiation uses to rebuild a disconnected cotangent.

use crate::arrays::differentiation::ExactShape;
use crate::arrays::{
    ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayType, Dimension, DimensionError, DimensionType, DimensionValue,
    DimensionVariable, Shape,
};
use crate::backends::arrays::ArrayOperation;
use crate::contexts::{Context, Domain, EagerContext};
use crate::differentiation::ResidualZeroProvider;
use crate::interpretation::{InterpretationDriver, MemberInterpretableOperation};
use crate::operations::constants::{
    Iota, IotaOperation, One, OneOperation, ZERO_OPERATION_NAME, Zero, ZeroOperation, ZeroOperationProvider,
    check_constructor_type_has_no_identity_references, infer_dynamic_constructor_output_types,
};
use crate::operations::dimensions::DimensionSizeOperation;
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::operations::{MemberOperation, Operation};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{TypeError, Typed};
use crate::programs::values::{Value, ValueProjection};
use crate::programs::{AtomId, ProgramBuilder, ProgramError};

macro_rules! impl_dynamic_constructor_member_operation {
    // Implements the shared mixed array IR boundary for one canonical homogeneous constructor payload.
    ($operation:ty) => {
        impl MemberOperation<ArrayIrType> for $operation {
            fn infer_parent_region_input_types(
                &self,
                _input_types: &[ArrayIrType],
                region_interfaces: &[RegionInterface<ArrayIrType>],
            ) -> Result<Vec<Option<Vec<ArrayIrType>>>, TypeError> {
                Ok(vec![None; region_interfaces.len()])
            }

            fn infer_parent_output_types(
                &self,
                input_types: &[ArrayIrType],
                region_interfaces: &[RegionInterface<ArrayIrType>],
            ) -> Result<Vec<ArrayIrType>, TypeError> {
                infer_dynamic_constructor_output_types(self.name(), self.r#type(), input_types, region_interfaces)
            }

            fn rename_parent_type_identities(
                &self,
                renaming: &TypeIdentityRenaming<DimensionVariable>,
            ) -> Result<Self, TypeError> {
                self.rename_type_identities(renaming)
            }
        }
    };
}

impl_dynamic_constructor_member_operation!(ZeroOperation<ArrayType>);
impl_dynamic_constructor_member_operation!(OneOperation<ArrayType>);
impl_dynamic_constructor_member_operation!(IotaOperation<ArrayType>);

/// Resolves one mixed constructor's explicit dimension operands into the concrete static output type required by an
/// eager backend.
fn materialize_dynamic_constructor_type<V>(
    name: &str,
    r#type: &ArrayType,
    inputs: &[V],
) -> Result<ArrayType, ProgramError>
where
    V: ValueProjection<DimensionType, Projected = DimensionValue>,
{
    let expected = r#type
        .shape()
        .dimensions()
        .iter()
        .filter(|dimension| matches!(dimension, Dimension::Dynamic(_)))
        .count();
    if expected == 0 {
        return Err(TypeError::invalid(format!(
            "'{name}' with static output type {type} has no dynamic dimensions; use the homogeneous nullary \
             constructor instead",
            r#type = r#type,
        ))
        .into());
    }
    let mut extents = inputs.iter();
    let dimensions = r#type
        .shape()
        .dimensions()
        .iter()
        .map(|dimension| match dimension {
            Dimension::Static(extent) => Ok(Dimension::Static(*extent)),
            Dimension::Dynamic(variable) => {
                let extent =
                    extents.next().ok_or(ProgramError::InvalidInputCount { expected, actual: inputs.len() })?;
                let extent = <V as ValueProjection<DimensionType>>::into_projected(extent.clone())?;
                // Eager binds skip inference and intermediate results skip boundary refinement checks, so validate
                // each runtime extent against the stored output axis's authoritative bounds before allocation.
                // Identity equality is deliberately not required because interpreted inputs may be alpha-renamed.
                if !variable.bounds().contains(extent.extent()) {
                    return Err(DimensionError::BindingOutOfBounds {
                        variable: variable.to_string(),
                        value: extent.extent(),
                        bounds: variable.bounds(),
                    }
                    .into());
                }
                Ok(Dimension::Static(extent.extent()))
            }
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;
    if extents.next().is_some() {
        return Err(ProgramError::InvalidInputCount { expected, actual: inputs.len() });
    }
    Ok(r#type.clone().with_shape(Shape::new(dimensions)))
}

macro_rules! impl_dynamic_constant_interpretation {
    // Implements eager materialization for a dynamic nullary array constructor.
    ($operation:ty, $capability:ident, $method:ident) => {
        impl<C> MemberInterpretableOperation<C> for $operation
        where
            C: Domain<Type = ArrayIrType>,
            C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
                + ValueProjection<DimensionType, Projected = DimensionValue>,
            EagerContext<
                <C::Value as ValueProjection<ArrayType>>::Projected,
                ArrayOperation<<C::Value as ValueProjection<ArrayType>>::Projected>,
            >: $capability<<C::Value as ValueProjection<ArrayType>>::Projected>,
        {
            fn interpret_in_parent<D: InterpretationDriver<C>>(
                &self,
                _context: &C,
                driver: &D,
                inputs: &[C::Value],
            ) -> Result<Vec<C::Value>, ProgramError> {
                if driver.region_count() != 0 {
                    return Err(
                        TypeError::invalid(format!("expected 0 regions but got {}", driver.region_count(),)).into()
                    );
                }
                let output_type = materialize_dynamic_constructor_type(self.name(), self.r#type(), inputs)?;
                let output = EagerContext::<
                    <C::Value as ValueProjection<ArrayType>>::Projected,
                    ArrayOperation<<C::Value as ValueProjection<ArrayType>>::Projected>,
                >::new()
                .$method(&output_type)?;
                Ok(vec![<C::Value as ValueProjection<ArrayType>>::from_projected(output)])
            }
        }
    };
}

impl_dynamic_constant_interpretation!(ZeroOperation<ArrayType>, Zero, zero);
impl_dynamic_constant_interpretation!(OneOperation<ArrayType>, One, one);

impl<C> MemberInterpretableOperation<C> for IotaOperation<ArrayType>
where
    C: Domain<Type = ArrayIrType>,
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + ValueProjection<DimensionType, Projected = DimensionValue>,
    EagerContext<
        <C::Value as ValueProjection<ArrayType>>::Projected,
        ArrayOperation<<C::Value as ValueProjection<ArrayType>>::Projected>,
    >: Iota<<C::Value as ValueProjection<ArrayType>>::Projected>,
{
    fn interpret_in_parent<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        if driver.region_count() != 0 {
            return Err(TypeError::invalid(format!("expected 0 regions but got {}", driver.region_count())).into());
        }
        let output_type = materialize_dynamic_constructor_type(self.name(), self.r#type(), inputs)?;
        let output = EagerContext::<
            <C::Value as ValueProjection<ArrayType>>::Projected,
            ArrayOperation<<C::Value as ValueProjection<ArrayType>>::Projected>,
        >::new()
        .iota(&output_type, self.dimension())?;
        Ok(vec![<C::Value as ValueProjection<ArrayType>>::from_projected(output)])
    }
}

impl<A: Value<Type = ArrayType>> From<ZeroOperation<ArrayType>> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: ZeroOperation<ArrayType>) -> Self {
        // Each zero has one canonical encoding: identity-free static zeros already belong to the homogeneous array
        // member family, and only reference-bearing dynamic output types need the mixed dimension-operand variant.
        if operation
            .r#type()
            .shape()
            .dimensions()
            .iter()
            .any(|dimension| matches!(dimension, Dimension::Dynamic(_)))
        {
            Self::Zero(operation)
        } else {
            Self::Array(ArrayOperation::Zero(operation))
        }
    }
}

impl<A: Value<Type = ArrayType>> From<OneOperation<ArrayType>> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: OneOperation<ArrayType>) -> Self {
        // Each one has one canonical encoding: identity-free static ones already belong to the homogeneous array
        // member family, and only reference-bearing dynamic output types need the mixed dimension-operand variant.
        if operation
            .r#type()
            .shape()
            .dimensions()
            .iter()
            .any(|dimension| matches!(dimension, Dimension::Dynamic(_)))
        {
            Self::DynamicOne(operation)
        } else {
            Self::Array(ArrayOperation::One(operation))
        }
    }
}

impl<A: Value<Type = ArrayType>> From<IotaOperation<ArrayType>> for ArrayIrOperation<A> {
    #[inline]
    fn from(operation: IotaOperation<ArrayType>) -> Self {
        if operation
            .r#type()
            .shape()
            .dimensions()
            .iter()
            .any(|dimension| matches!(dimension, Dimension::Dynamic(_)))
        {
            Self::DynamicIota(operation)
        } else {
            Self::Array(ArrayOperation::Iota(operation))
        }
    }
}

// These residual-protocol algorithms are deliberately inherent methods duplicated by thin trait-impl delegations
// below rather than living only on the trait: the `XlaOperation` provider reuses them across operation families,
// which their `Self`-typed builder and context parameters cannot express. Do not fold them into the trait impl.
impl<A: Value<Type = ArrayType>> ArrayIrOperation<A> {
    /// Captures the runtime extents needed to materialize a disconnected cotangent zero from the primal `source`.
    pub fn capture_zero_residuals<
        V: Value<Type = ArrayIrType>,
        O: Operation<Type = ArrayIrType> + From<DimensionSizeOperation>,
    >(
        builder: &mut ProgramBuilder<V, O>,
        source: AtomId,
        r#type: &ArrayIrType,
    ) -> Result<Vec<AtomId>, ProgramError> {
        let r#type = <&ArrayType>::try_from(r#type)?;
        let (_, first_axes) = ExactShape::for_residual_zero(r#type.shape());
        if first_axes.is_empty() {
            return Ok(Vec::new());
        }
        let source_type = builder
            .atoms()
            .get(source.index())
            .ok_or(ProgramError::UnboundAtomId { id: source })?
            .r#type()
            .into_owned();
        let source_type = <&ArrayType>::try_from(&source_type)?;
        // Reading only each identity's first source axis establishes the same deduplicated residual ordering used by
        // zero construction, including when several axes share one identity.
        first_axes
            .into_iter()
            .map(|(axis, _)| {
                Ok(builder.add_instruction(
                    DimensionSizeOperation::new(source_type, axis)?,
                    Vec::new(),
                    vec![source],
                )?[0])
            })
            .collect()
    }

    /// Captures the value-level counterpart of
    /// [`capture_zero_residuals`](Self::capture_zero_residuals) in `context`.
    pub fn capture_zero_residual_values<C>(
        context: &C,
        source: &C::Value,
        r#type: &ArrayIrType,
    ) -> Result<Vec<C::Value>, ProgramError>
    where
        C: Context<Type = ArrayIrType>,
        C::Operation: From<DimensionSizeOperation>,
    {
        let r#type = <&ArrayType>::try_from(r#type)?;
        let (_, first_axes) = ExactShape::for_residual_zero(r#type.shape());
        if first_axes.is_empty() {
            return Ok(Vec::new());
        }
        let source_type = source.r#type();
        let source_type = <&ArrayType>::try_from(source_type.as_ref())?;
        // Capture one concrete extent per identity from its first source axis; repeated axes reuse that value when the
        // dynamic zero's per-axis operand list is reconstructed.
        first_axes
            .into_iter()
            .map(|(axis, _)| {
                Ok(context
                    .bind(DimensionSizeOperation::new(source_type, axis)?, Vec::new(), std::slice::from_ref(source))?
                    .remove(0))
            })
            .collect()
    }

    /// Returns the canonical zero operation for `r#type` and expands one explicit extent residual per distinct dynamic
    /// identity into the mixed constructor's per-axis operand order.
    pub fn zero_operation_with_residuals<R: Clone>(
        r#type: ArrayIrType,
        residuals: &[R],
    ) -> Result<(Self, Vec<R>), ProgramError> {
        let r#type = <&ArrayType>::try_from(&r#type)?.clone();
        let (shape, first_axes) = ExactShape::for_residual_zero(r#type.shape());
        let expected_residual_count = first_axes.len();
        if residuals.len() != expected_residual_count {
            return Err(ProgramError::InvalidArgument {
                message: format!(
                    "dynamic zero expected {expected_residual_count} extent residuals but got {}",
                    residuals.len(),
                ),
            });
        }
        if first_axes.is_empty() {
            return Ok((Self::zero_operation(r#type.into())?, Vec::new()));
        }
        let operands = shape.dynamic_dimensions(residuals);
        Ok((Self::Zero(ZeroOperation::new(r#type)), operands))
    }
}

impl<A: Value<Type = ArrayType>> ZeroOperationProvider<ArrayIrType> for ArrayIrOperation<A> {
    fn zero_operation(r#type: ArrayIrType) -> Result<Self, ProgramError> {
        let ArrayIrType::Array(r#type) = r#type else {
            return Err(TypeError::invalid("cannot materialize a zero for a first-class dimension type").into());
        };
        check_constructor_type_has_no_identity_references(ZERO_OPERATION_NAME, &r#type)?;
        Ok(Self::Array(ArrayOperation::Zero(ZeroOperation::new(r#type))))
    }
}

impl<A: Value<Type = ArrayType>> ResidualZeroProvider<ArrayIrType> for ArrayIrOperation<A> {
    fn zero_residual_types(r#type: &ArrayIrType) -> Vec<ArrayIrType> {
        match r#type {
            ArrayIrType::Array(r#type) => {
                let (_, first_axes) = ExactShape::for_residual_zero(r#type.shape());
                first_axes.into_iter().map(|(_, variable)| DimensionType::new(variable).into()).collect()
            }
            ArrayIrType::Dimension(_) => Vec::new(),
        }
    }

    fn capture_zero_residuals<V: Value<Type = ArrayIrType>>(
        builder: &mut ProgramBuilder<V, Self>,
        source: AtomId,
        r#type: &ArrayIrType,
    ) -> Result<Vec<AtomId>, ProgramError> {
        ArrayIrOperation::<A>::capture_zero_residuals(builder, source, r#type)
    }

    fn capture_zero_residual_values<C: Context<Type = ArrayIrType, Operation = Self>>(
        context: &C,
        source: &C::Value,
        r#type: &ArrayIrType,
    ) -> Result<Vec<C::Value>, ProgramError> {
        ArrayIrOperation::<A>::capture_zero_residual_values(context, source, r#type)
    }

    fn zero_operation_with_residuals<R: Clone>(
        r#type: ArrayIrType,
        residuals: &[R],
    ) -> Result<(Self, Vec<R>), ProgramError> {
        ArrayIrOperation::<A>::zero_operation_with_residuals(r#type, residuals)
    }
}

impl<A: Value<Type = ArrayType>, O: Operation<Type = ArrayIrType>> Zero<ArrayIrValue<A>>
    for EagerContext<ArrayIrValue<A>, O>
where
    EagerContext<A, ArrayOperation<A>>: Zero<A>,
{
    fn zero(&self, r#type: &ArrayIrType) -> Result<ArrayIrValue<A>, ProgramError> {
        let array_type = <&ArrayType>::try_from(r#type)?;
        Ok(ArrayIrValue::Array(EagerContext::<A, ArrayOperation<A>>::new().zero(array_type)?))
    }
}

#[cfg(test)]
mod tests {

    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayType, DataType, Dimension, DimensionBounds, DimensionType,
        DimensionValue, DimensionVariable, Shape,
    };

    use crate::backends::arrays::{Array, ArrayOperation};

    use crate::compilation::{
        CallRequest, CompilationDomain, CompilationTracer, CompileRequest, CompiledFunction, FlatCompilationProgram,
        JittedFunction, LoweredFunction, LoweringRequest, StageRequest, StagedFunction, try_jit,
    };
    use crate::contexts::{Context, EagerContext, StagingContext};
    use crate::differentiation::{
        DifferentiableType, ForwardModeDifferentiate, ReverseModeDifferentiate, TransposableOperation,
    };
    use crate::interpretation::InterpretableOperation;
    use crate::macros::check_operation_partial_evaluation;

    use crate::operations::constants::ZeroOperation;

    use crate::operations::differentiation::StopGradientOperation;

    use crate::operations::manipulation::BroadcastOperation;

    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::builders::ProgramBuilder;

    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::Typed;

    use crate::programs::{AtomId, MaybeZero, ProgramError};
    use crate::tracing::TracingContext;

    use super::*;

    /// Minimal composite compilation domain used to prove the retained-JIT contract over dimension inputs: it
    /// stages through the ordinary tracing path, "lowers" and "compiles" to the lifted flat program itself, counts
    /// backend compilations, and executes calls by eager interpretation of the compiled program.
    #[derive(Clone)]
    struct RetainedJitDomain {
        /// Number of backend compilations performed by this domain.
        compilations: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    /// Compilation options of [`RetainedJitDomain`], which requires none.
    #[derive(Clone, Debug, Default, PartialEq)]
    struct RetainedJitOptions;

    impl RetainedJitDomain {
        fn new() -> Self {
            Self { compilations: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)) }
        }

        fn compilation_count(&self) -> usize {
            self.compilations.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    impl crate::contexts::Domain for RetainedJitDomain {
        type Type = ArrayIrType;
        type Value = ArrayIrValue<Array>;
        type Constant = crate::captures::CaptureReference<ArrayIrType>;
        type Operation = ArrayIrOperation<Array>;
    }

    impl CompilationDomain for RetainedJitDomain {
        type LoweredProgram = FlatCompilationProgram<Self>;
        type CompiledProgram = FlatCompilationProgram<Self>;
        type Options = RetainedJitOptions;
        type Error = ProgramError;

        fn stage<Request>(
            &self,
            request: Request,
        ) -> Result<StagedFunction<Self, Request::Input, Request::Output>, ProgramError>
        where
            Request: StageRequest<Self>,
        {
            request.trace(|_, output_types| Ok(output_types))
        }

        fn lower<Request>(
            &self,
            staged: Request,
        ) -> Result<LoweredFunction<Self, Request::Input, Request::Output>, ProgramError>
        where
            Request: LoweringRequest<Self>,
        {
            let program = staged.lifted_program()?.as_ref().clone();
            let output_types = staged.staged().output_types().to_vec();
            Ok(staged.into_lowered(program, output_types))
        }

        fn compile<Request>(
            &self,
            lowered: Request,
        ) -> Result<CompiledFunction<Self, Request::Input, Request::Output>, ProgramError>
        where
            Request: CompileRequest<Self>,
        {
            self.compilations.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let program = lowered.lowered().lowered_program().clone();
            let output_types = lowered.lowered().output_types().to_vec();
            Ok(lowered.into_compiled(std::sync::Arc::new(program), output_types))
        }

        fn call<Request>(&self, request: Request) -> Result<Request::RuntimeOutput, ProgramError>
        where
            Request: CallRequest<Self>,
        {
            let executable = request.executable().clone();
            let outputs = executable.compiled_program().interpret_with(
                request.into_arguments(),
                |_, capture| {
                    Err(ProgramError::MalformedProgram(format!(
                        "retained-JIT test program retained capture {}",
                        capture.index(),
                    )))
                },
                |instruction, inputs| {
                    instruction.operation().interpret(
                        &EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
                        &EmptyRegionDriver,
                        inputs,
                    )
                },
            )?;
            Request::reconstruct(&executable, outputs)
        }
    }

    #[test]
    fn test_array_ir_dynamic_zero_retained_jit_reuses_one_specialization() {
        let domain = RetainedJitDomain::new();
        let function: JittedFunction<RetainedJitDomain, _, (), ArrayIrType, ArrayIrType> =
            try_jit(&domain, |(), extent: CompilationTracer<RetainedJitDomain>| {
                let ArrayIrType::Dimension(extent_type) = extent.r#type().into_owned() else {
                    return Err(ProgramError::InvalidArgument { message: "expected a dimension input".to_string() });
                };
                Ok(extent
                    .context()
                    .bind(
                        ZeroOperation::new(ArrayType::new(
                            DataType::F32,
                            Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]),
                        )),
                        Vec::new(),
                        std::slice::from_ref(&extent),
                    )?
                    .remove(0))
            });

        // Two calls with different runtime extents share one abstract input type, and therefore one retained trace,
        // lowering, and compiled specialization, while still producing outputs with different logical shapes. This is
        // the retained-JIT contract that would break if concrete extents ever became part of type or cache identity.
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        assert_eq!(
            function.call((), ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap())),
            Ok(ArrayIrValue::Array(Array::vector(vec![0.0_f32, 0.0, 0.0]))),
        );
        assert_eq!(
            function.call((), ArrayIrValue::Dimension(DimensionValue::new(extent_type, 4).unwrap())),
            Ok(ArrayIrValue::Array(Array::vector(vec![0.0_f32, 0.0, 0.0, 0.0]))),
        );
        assert_eq!(function.specialization_count(), 1);
        let statistics = function.statistics();
        assert_eq!(statistics.dispatch_misses, 1);
        assert_eq!(statistics.dispatch_hits, 1);
        assert_eq!(statistics.traces, 1);
        assert_eq!(statistics.lowerings, 1);
        assert_eq!(statistics.compilation_requests, 1);
        assert_eq!(domain.compilation_count(), 1);
    }

    #[test]
    fn test_array_ir_shaped_zero_partial_evaluation() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let extent = ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap());
        let output = ArrayIrValue::Array(Array::vector(vec![0.0_f32, 0.0, 0.0]));
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = ZeroOperation::new(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]),
            )),
            cases = [
                {
                    inputs = [(@known, extent.clone())],
                    outputs = [(@known, output.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [(@unknown(type = extent_type.into(), replay = extent))],
                    outputs = [(@residual, output)],
                    residual_instructions = 1,
                },
            ],
        );
    }

    #[test]
    fn test_array_ir_dynamic_one_partial_evaluation() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let extent = ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap());
        let output = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 1.0, 1.0]));
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = OneOperation::new(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]),
            )),
            cases = [
                {
                    inputs = [(@known, extent.clone())],
                    outputs = [(@known, output.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [(@unknown(type = extent_type.into(), replay = extent))],
                    outputs = [(@residual, output)],
                    residual_instructions = 1,
                },
            ],
        );
    }

    #[test]
    fn test_array_ir_shaped_zero_differentiation() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let extent = builder.add_input(extent_type.clone().into());
        let output = builder
            .add_instruction(
                ZeroOperation::new(ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]),
                )),
                Vec::new(),
                vec![extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let jvp = program.jvp().unwrap();
        let extent = ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap());
        assert_eq!(
            jvp.interpret(vec![extent]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![0.0_f64, 0.0, 0.0])),
                ArrayIrValue::Array(Array::vector(vec![0.0_f64, 0.0, 0.0])),
            ]),
        );
        assert_eq!(jvp.instructions().iter().filter(|instruction| instruction.operation().is_zero(0)).count(), 1);

        // Direct differentiation-context dispatch takes the same all-zero shortcut. Its tangent must reuse the shaped
        // primal SSA value rather than materializing a nullary zero that has no access to the runtime extent.
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let context = TestContext::new();
        let extent = context.input(extent_type.clone().into());
        let extent_tangent = context.input(ArrayType::scalar(DataType::Zero).into());
        let dynamic_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]));
        let (primal, tangent) = context
            .jvp(
                move |extent| {
                    let context = extent.context().clone();
                    Ok(context.bind(ZeroOperation::new(dynamic_type), Vec::new(), &[extent])?.remove(0))
                },
                extent,
                extent_tangent,
            )
            .unwrap();
        assert_eq!(primal.atom_id(), tangent.atom_id());
        let builder = context.builder().borrow();
        let [instruction] = builder.instructions() else {
            panic!("expected one dynamic-zero instruction");
        };
        assert!(matches!(instruction.operation(), ArrayIrOperation::Zero(_)));
    }

    #[test]
    fn test_array_ir_dynamic_one_differentiation() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let extent = builder.add_input(extent_type.clone().into());
        let output = builder
            .add_instruction(
                OneOperation::new(ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]),
                )),
                Vec::new(),
                vec![extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let jvp = program.jvp().unwrap();
        let extent = ArrayIrValue::Dimension(DimensionValue::new(extent_type, 3).unwrap());
        assert_eq!(
            jvp.interpret(vec![extent]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 1.0, 1.0])),
                ArrayIrValue::Array(Array::vector(vec![0.0_f64, 0.0, 0.0])),
            ]),
        );
        assert_eq!(jvp.instructions().len(), 2);
        assert!(matches!(jvp.instructions()[0].operation(), ArrayIrOperation::DynamicOne(_)));
        assert!(matches!(jvp.instructions()[1].operation(), ArrayIrOperation::Zero(_)));
        assert_eq!(jvp.instructions()[0].inputs(), jvp.instructions()[1].inputs());

        // The direct transform context must likewise run the explicit rule rather than taking its all-structural-zero
        // shortcut: a nullary zero cannot recover the dynamic extent after the closure returns.
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let dynamic_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]));
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let (primal, tangent) = context
            .jvp(
                move |extent| {
                    let context = extent.context().clone();
                    Ok(context.bind(OneOperation::new(dynamic_type), Vec::new(), &[extent])?.remove(0))
                },
                ArrayIrValue::Dimension(DimensionValue::new(extent_type, 3).unwrap()),
                ArrayIrValue::Array(Array::new(ArrayType::scalar(DataType::Zero), Vec::new()).unwrap()),
            )
            .unwrap();
        assert_eq!(primal, ArrayIrValue::Array(Array::vector(vec![1.0_f64, 1.0, 1.0])));
        assert_eq!(tangent, ArrayIrValue::Array(Array::vector(vec![0.0_f64, 0.0, 0.0])));
    }

    #[test]
    fn test_array_ir_dynamic_iota_differentiation() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let extent = builder.add_input(extent_type.clone().into());
        let output = builder
            .add_instruction(
                IotaOperation::new(
                    ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())])),
                    0,
                )
                .unwrap(),
                Vec::new(),
                vec![extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let jvp = program.jvp().unwrap();
        let extent = ArrayIrValue::Dimension(DimensionValue::new(extent_type, 3).unwrap());
        assert_eq!(
            jvp.interpret(vec![extent]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![0.0_f64, 1.0, 2.0])),
                ArrayIrValue::Array(Array::vector(vec![0.0_f64, 0.0, 0.0])),
            ]),
        );
        assert_eq!(jvp.instructions().len(), 2);
        assert!(matches!(jvp.instructions()[0].operation(), ArrayIrOperation::DynamicIota(_)));
        assert!(matches!(jvp.instructions()[1].operation(), ArrayIrOperation::Zero(_)));
        assert_eq!(jvp.instructions()[0].inputs(), jvp.instructions()[1].inputs());
    }

    #[test]
    fn test_array_ir_dynamic_zero_alpha_renamed_instantiation() {
        let formal = DimensionVariable::new("formal", DimensionBounds::new(1, Some(5)).unwrap());
        let caller = DimensionVariable::new("caller", DimensionBounds::new(2, Some(4)).unwrap());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let extent = builder.add_input(DimensionType::new(formal.clone()).into());
        let output = builder
            .add_instruction(
                ZeroOperation::new(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(formal)]))),
                Vec::new(),
                vec![extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        // Genuine cross-program instantiation: deriving the caller renaming from the complete boundary signature
        // renames the whole program — including the dynamic zero's stored output type — and recloses its region
        // arena, so the instantiated payload stays consistent with the instantiated atom types.
        let caller_input = ArrayIrType::Dimension(DimensionType::new(caller.clone()));
        let instantiated = program.with_instantiated_type_identities(std::slice::from_ref(&caller_input)).unwrap();
        assert_eq!(instantiated.input_types(), vec![caller_input]);
        assert_eq!(
            instantiated.output_types(),
            vec![ArrayIrType::Array(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(caller.clone())]),
            ))],
        );
        let [instruction] = instantiated.instructions() else {
            panic!("expected one instantiated instruction");
        };
        let ArrayIrOperation::Zero(instantiated_zero) = instruction.operation() else {
            panic!("expected the instantiated operation to remain a dynamic zero");
        };
        assert_eq!(
            instantiated_zero.r#type(),
            &ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(caller.clone())])),
        );
        assert_eq!(
            instantiated.interpret(vec![ArrayIrValue::Dimension(
                DimensionValue::new(DimensionType::new(caller.clone()), 3).unwrap()
            )]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0_f32, 0.0, 0.0]))]),
        );

        // A boundary interpretation of the *uninstantiated* program with an alpha-renamed actual input type takes
        // the non-exact establishment path instead: the actual dimension member refines the declared one by bounds
        // alone, and the concrete static output then establishes its first fact for the declared input identity
        // through the closed identity signature.
        assert_eq!(
            program
                .interpret(vec![ArrayIrValue::Dimension(DimensionValue::new(DimensionType::new(caller), 3).unwrap())]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0_f32, 0.0, 0.0]))]),
        );
    }

    #[test]
    fn test_array_ir_dynamic_one_identity_instantiation() {
        let formal = DimensionVariable::new("formal", DimensionBounds::new(1, Some(5)).unwrap());
        let caller = DimensionVariable::new("caller", DimensionBounds::new(2, Some(4)).unwrap());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let extent = builder.add_input(DimensionType::new(formal.clone()).into());
        let output = builder
            .add_instruction(
                OneOperation::new(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(formal)]))),
                Vec::new(),
                vec![extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let caller_input = ArrayIrType::Dimension(DimensionType::new(caller.clone()));
        let instantiated = program.with_instantiated_type_identities(std::slice::from_ref(&caller_input)).unwrap();
        let [instruction] = instantiated.instructions() else {
            panic!("expected one instantiated instruction");
        };
        let ArrayIrOperation::DynamicOne(instantiated_one) = instruction.operation() else {
            panic!("expected the instantiated operation to remain a dynamic one");
        };
        assert_eq!(
            instantiated_one.r#type(),
            &ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(caller.clone())])),
        );
        assert_eq!(
            instantiated
                .interpret(vec![ArrayIrValue::Dimension(DimensionValue::new(DimensionType::new(caller), 3).unwrap())]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f32, 1.0, 1.0]))]),
        );
    }

    #[test]
    fn test_array_ir_dynamic_iota_identity_instantiation() {
        let formal = DimensionVariable::new("formal", DimensionBounds::new(1, Some(5)).unwrap());
        let caller = DimensionVariable::new("caller", DimensionBounds::new(2, Some(4)).unwrap());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let extent = builder.add_input(DimensionType::new(formal.clone()).into());
        let output = builder
            .add_instruction(
                IotaOperation::new(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(formal)])), 0)
                    .unwrap(),
                Vec::new(),
                vec![extent],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let caller_input = ArrayIrType::Dimension(DimensionType::new(caller.clone()));
        let instantiated = program.with_instantiated_type_identities(std::slice::from_ref(&caller_input)).unwrap();
        let [instruction] = instantiated.instructions() else {
            panic!("expected one instantiated instruction");
        };
        let ArrayIrOperation::DynamicIota(instantiated_iota) = instruction.operation() else {
            panic!("expected the instantiated operation to remain a dynamic iota");
        };
        assert_eq!(
            instantiated_iota.r#type(),
            &ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(caller.clone())])),
        );
        assert_eq!(
            instantiated
                .interpret(vec![ArrayIrValue::Dimension(DimensionValue::new(DimensionType::new(caller), 3).unwrap())]),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements(
                    ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(3)])),
                    &[0i32, 1, 2],
                )
                .unwrap(),
            )]),
        );
    }

    #[test]
    fn test_array_ir_dynamic_constructor_transposition() {
        // Dynamic constructors depend on their extent operands only as non-differentiable shape inputs, so every
        // extent receives a structural-zero cotangent regardless of the output cotangent being live.
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let output_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]));
        for operation in [
            ArrayIrOperation::<Array>::from(ZeroOperation::new(output_type.clone())),
            ArrayIrOperation::<Array>::from(OneOperation::new(output_type.clone())),
            ArrayIrOperation::<Array>::from(IotaOperation::new(output_type.clone(), 0).unwrap()),
        ] {
            let mut context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let output_cotangent = context.input(output_type.clone().into());
            let cotangents = operation
                .transpose(
                    &mut context,
                    &EmptyRegionDriver,
                    &[PartialValue::Unknown(extent_type.clone().into())],
                    &[MaybeZero::Value(output_cotangent)],
                )
                .unwrap();
            let [cotangent] = cotangents.as_slice() else {
                panic!("expected one cotangent per operation input");
            };
            assert!(matches!(cotangent, MaybeZero::Zero(_)));
        }
    }

    #[test]
    fn test_array_ir_dynamic_literal_fill_jvp_materializes_shaped_zero() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let extent = builder.add_input(extent_type.clone().into());
        let scalar = builder
            .add_instruction(
                ArrayOperation::from(crate::ConstantOperation::new(Array::scalar(2.5_f64))),
                Vec::new(),
                vec![],
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(BroadcastOperation::new(Vec::new()), Vec::new(), vec![scalar, extent])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let jvp = program.jvp().unwrap();
        let extent = ArrayIrValue::Dimension(DimensionValue::new(extent_type, 3).unwrap());
        assert_eq!(
            jvp.interpret(vec![extent]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(vec![2.5_f64, 2.5, 2.5])),
                ArrayIrValue::Array(Array::vector(vec![0.0_f64, 0.0, 0.0])),
            ]),
        );
        let dynamic_zero = jvp
            .instructions()
            .iter()
            .find(|instruction| matches!(instruction.operation(), ArrayIrOperation::Zero(_)))
            .unwrap();
        assert_eq!(dynamic_zero.inputs(), &[AtomId::new(0)]);
    }

    #[test]
    fn test_array_ir_dynamic_projected_jvp_materializes_source_relative_widened_zero() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let input_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::StopGradient(StopGradientOperation::new())),
                Vec::new(),
                vec![input],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        // The projected constant derivative uses its primal result as the runtime-shape exemplar, then widens the
        // element type to the tangent representation. No type-only dynamic zero is present in the fused JVP.
        let jvp = program.jvp().unwrap();
        assert!(jvp.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ArrayIrOperation::Array(ArrayOperation::ZeroLike(_))
        )));
        assert!(jvp.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ArrayIrOperation::Array(ArrayOperation::ConvertElementType(_))
        )));
        assert!(
            !jvp.instructions()
                .iter()
                .any(|instruction| matches!(instruction.operation(), ArrayIrOperation::Zero(_)))
        );

        let primal = Array::from_f64s(
            ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(3)])),
            vec![1.0, 2.0, 4.0],
        );
        let tangent = Array::vector(vec![1.0_f32, 1.0, 1.0]);
        let expected_primal = primal.clone();
        assert_eq!(
            jvp.interpret(vec![ArrayIrValue::Array(primal), ArrayIrValue::Array(tangent)]),
            Ok(
                vec![ArrayIrValue::Array(expected_primal), ArrayIrValue::Array(Array::vector(vec![0.0_f32, 0.0, 0.0])),]
            ),
        );
    }

    #[test]
    fn test_array_ir_dynamic_disconnected_pullback_uses_explicit_extent_residual() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let dynamic_type = ArrayType::new(
            DataType::F8E8M0FNU,
            Shape::new(vec![
                Dimension::Dynamic(extent_type.variable().clone()),
                Dimension::Dynamic(extent_type.variable().clone()),
            ]),
        );
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        builder.add_input(dynamic_type.into());
        let scalar = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![scalar],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        // The dynamic input is disconnected from the output, so linearization retains its observed extent as one
        // ordinary residual and the pullback feeds that residual to the mixed zero constructor.
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        assert!(matches!(
            linearization.primal().instructions().last().unwrap().operation(),
            ArrayIrOperation::DimensionSize(_)
        ));
        let pullback = linearization.pullback().unwrap();
        let zero = pullback.instructions().last().unwrap();
        assert!(matches!(zero.operation(), ArrayIrOperation::Zero(_)));
        assert_eq!(zero.inputs(), &[AtomId::new(1), AtomId::new(1)]);
        assert_eq!(
            pullback.interpret(vec![
                ArrayIrValue::Array(Array::scalar(2.0_f64)),
                ArrayIrValue::Dimension(DimensionValue::new(extent_type, 3).unwrap()),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::matrix(3, 3, vec![0.0_f32; 9])),
                ArrayIrValue::Array(Array::scalar(2.0_f64)),
            ]),
        );
    }

    #[test]
    fn test_array_ir_nested_dynamic_disconnected_pullback_uses_explicit_extent_residual() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Dynamic(extent)]));
        let scalar_type = ArrayType::scalar(DataType::F64);
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let dynamic = context.input(dynamic_type.clone().into());
        let scalar = context.input(scalar_type.clone().into());

        // Value-level reverse mode runs inside the outer trace. It saves only the disconnected array's extent, and
        // the reusable pullback consumes that dimension residual through the mixed zero constructor.
        let (_, pullback) = context.vjp(|inputs: Vec<_>| Ok(vec![inputs[1].clone()]), vec![dynamic, scalar]).unwrap();
        assert_eq!(pullback.residuals().len(), 1);
        assert!(matches!(pullback.residuals()[0].r#type().as_ref(), ArrayIrType::Dimension(_)));
        let zero = pullback.program().instructions().last().unwrap();
        assert!(matches!(zero.operation(), ArrayIrOperation::Zero(_)));
        assert_eq!(zero.inputs(), &[AtomId::new(1)]);

        let cotangent = context.input(scalar_type.into());
        let cotangents = pullback.apply(vec![cotangent]).unwrap();
        assert_eq!(cotangents[0].r#type().as_ref(), &ArrayIrType::Array(dynamic_type.cotangent()));
        assert_eq!(cotangents[1].r#type().as_ref(), &ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
    }
}
