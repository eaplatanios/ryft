use std::fmt::Display;

use crate::backends::array_programs::{ArrayProgramOperation, ArrayProgramValue};
use crate::batching::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, BatchableOperation, BatchingContext, BatchingDriver, BatchingError,
};
use crate::contexts::{Context, Domain, EagerContext};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::dimensions::DimensionSize;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::ProgramError;
use crate::programs::effects::{Effect, Effects};
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::{Value, ValueProjection};
use crate::types::{ArrayProgramType, ArrayType, Dimension, DimensionType, DimensionVariable};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`CustomCallOperation`].
pub const CUSTOM_CALL_OPERATION_NAME: &str = "custom_call";

/// Typed configuration attribute value carried by a [`CustomCallOperation`] and forwarded to the foreign kernel.
/// The variants deliberately cover only the encodings every supporting backend must decode: strings, Booleans,
/// 64-bit signed integers, and 64-bit floating-point values. The `From` conversions let attribute values be passed
/// directly to [`CustomCallOperation::with_attribute`] (e.g., `.with_attribute("scale", 2.0)`); `&str` and `String`
/// convert into [`String`](Self::String), `bool` into [`Boolean`](Self::Boolean), `i64` into [`I64`](Self::I64),
/// and `f64` into [`F64`](Self::F64).
#[derive(Clone, Debug, PartialEq)]
pub enum CustomCallAttribute {
    /// UTF-8 string value.
    String(String),

    /// Boolean value.
    Boolean(bool),

    /// 64-bit signed-integer value.
    I64(i64),

    /// 64-bit floating-point value.
    F64(f64),
}

impl Display for CustomCallAttribute {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(string) => formatter.write_str(string),
            Self::Boolean(boolean) => write!(formatter, "{boolean}"),
            Self::I64(integer) => write!(formatter, "{integer}"),
            Self::F64(float) => write!(formatter, "{float:?}"),
        }
    }
}

impl From<&str> for CustomCallAttribute {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

impl From<String> for CustomCallAttribute {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<bool> for CustomCallAttribute {
    fn from(value: bool) -> Self {
        Self::Boolean(value)
    }
}

impl From<i64> for CustomCallAttribute {
    fn from(value: i64) -> Self {
        Self::I64(value)
    }
}

impl From<f64> for CustomCallAttribute {
    fn from(value: f64) -> Self {
        Self::F64(value)
    }
}

/// [`Operation`] that calls a foreign kernel registered with the executing backend under a target name — the
/// analogue of [`jax.ffi.ffi_call`](https://docs.jax.dev/en/latest/ffi.html). The operation is opaque to Ryft:
/// its output types are declared up front instead of inferred, and typed [`CustomCallAttribute`]s are forwarded
/// verbatim to the kernel as its configuration.
///
/// In an array program, each dynamic axis occurrence in the declared outputs requires one trailing first-class
/// dimension operand, ordered first by output and then by axis. Type inference verifies that each operand defines the
/// exact variable referenced by its corresponding output axis. These logical result extents do not enter the foreign
/// kernel ABI: only the leading array operands are passed to the kernel. Eager execution and backend lowering use the
/// trailing operands to verify or attach the declared logical sizes to the returned buffers.
///
/// The XLA backend lowers this operation to a
/// [`stablehlo.custom_call`](https://openxla.org/stablehlo/spec#custom_call) using the typed FFI calling convention
/// (`api_version = 4`), with the attributes carried as the `backend_config` dictionary. Handlers are registered with
/// the executing PJRT client under the same target name (e.g., via `ryft-pjrt`'s `Client::register_ffi_handler`).
/// The reference array backend cannot execute foreign kernels, so eager interpretation on it reports an error.
///
/// Because the kernel is opaque, the operation has no differentiation or batching rules: differentiating it reports
/// an error directing users to wrap the call with [`custom_jvp`](crate::tracing_v2::CustomJvp) or
/// [`custom_vjp`](crate::tracing_v2::CustomVjp), and batching it reports an error (invoke a kernel that understands
/// the batch axis instead). Marking the call as side-effecting via [`with_side_effect`](Self::with_side_effect)
/// reports [`Effect::OrderedIo`], which keeps the call alive through dead-code elimination and preserves its
/// execution order relative to other ordered effects; the lowered custom call is then also marked
/// `has_side_effect = true` so the XLA compiler never elides or reorders it.
///
/// # Backend Contract
///
/// This operation is backend-independent by design, and this payload is the entire portable contract: a target
/// name resolved in the executing backend's kernel registry, declared output types, typed configuration
/// attributes, and an effect flag. A backend supports the operation by providing (1) a process- or client-level
/// registry that resolves target names to executable kernels at execution time, (2) a calling convention that
/// hands the kernel its input buffers, output buffers matching the declared output types, and the decoded
/// attributes, and (3) an execution engine that honors [`Effect::OrderedIo`] for side-effecting calls. Backends
/// that cannot execute foreign kernels (like the reference array backend) reject interpretation with a clear
/// error instead of guessing.
///
/// Backend-specific vocabulary must never grow on this payload: encodings such as XLA's FFI API version or
/// `backend_config` layout, operand memory layouts, output-operand aliasing, or called-computation references
/// belong in the owning backend's lowering (or in a backend-owned wrapper operation), not here. If a
/// configuration knob only makes sense for one backend, it does not belong on this operation.
#[derive(Clone, Debug)]
pub struct CustomCallOperation {
    /// Name under which the foreign kernel is registered with the executing backend.
    target_name: String,

    /// Declared output types of the call, returned verbatim by type inference.
    output_types: Vec<ArrayType>,

    /// Typed configuration attributes forwarded to the kernel, in insertion order.
    attributes: Vec<(String, CustomCallAttribute)>,

    /// Whether the call has observable side effects beyond its returned outputs.
    has_side_effect: bool,
}

impl CustomCallOperation {
    /// Creates a new [`CustomCallOperation`] with the provided target name and declared output types.
    ///
    /// # Parameters
    ///
    ///   - `target_name`: Name under which the foreign kernel is registered with the executing backend.
    ///   - `output_types`: Declared output types of the call, returned verbatim by type inference.
    #[inline]
    pub fn new<N: Into<String>>(target_name: N, output_types: Vec<ArrayType>) -> Self {
        Self { target_name: target_name.into(), output_types, attributes: Vec::new(), has_side_effect: false }
    }

    /// Returns a copy of this [`CustomCallOperation`] with the provided typed configuration attribute appended.
    #[inline]
    pub fn with_attribute<N: Into<String>, V: Into<CustomCallAttribute>>(mut self, name: N, value: V) -> Self {
        self.attributes.push((name.into(), value.into()));
        self
    }

    /// Returns a copy of this [`CustomCallOperation`] marked as having observable side effects.
    #[inline]
    pub fn with_side_effect(mut self) -> Self {
        self.has_side_effect = true;
        self
    }

    /// Returns the name under which the foreign kernel is registered with the executing backend.
    #[inline]
    pub fn target_name(&self) -> &str {
        self.target_name.as_str()
    }

    /// Returns the declared output types of the call.
    #[inline]
    pub fn output_types(&self) -> &[ArrayType] {
        self.output_types.as_slice()
    }

    /// Returns the typed configuration attributes forwarded to the kernel, in insertion order.
    #[inline]
    pub fn attributes(&self) -> &[(String, CustomCallAttribute)] {
        self.attributes.as_slice()
    }

    /// Returns whether the call has observable side effects beyond its returned outputs.
    #[inline]
    pub fn has_side_effect(&self) -> bool {
        self.has_side_effect
    }
}

impl Display for CustomCallOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Operation::<ArrayProgramType>::render(self, formatter, 0)
    }
}

impl Operation<ArrayType> for CustomCallOperation {
    #[inline]
    fn name(&self) -> &'static str {
        CUSTOM_CALL_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        _input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        Ok(self.output_types.clone())
    }

    #[inline]
    fn effects(&self) -> Effects {
        if self.has_side_effect { Effects::single(Effect::OrderedIo) } else { Effects::PURE }
    }

    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayType as crate::Type>::Identity>,
    ) -> Result<Self, TypeError> {
        Ok(Self {
            target_name: self.target_name.clone(),
            output_types: self
                .output_types
                .iter()
                .map(|r#type| r#type.rename_identities(renaming))
                .collect::<Result<Vec<_>, _>>()?,
            attributes: self.attributes.clone(),
            has_side_effect: self.has_side_effect,
        })
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, CUSTOM_CALL_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("target", &self.target_name)?;
            for (name, value) in &self.attributes {
                operation.field(name, value)?;
            }
            if self.has_side_effect {
                operation.field("has_side_effect", true)?;
            }
            Ok(())
        })
    }
}

impl Operation<ArrayProgramType> for CustomCallOperation {
    #[inline]
    fn name(&self) -> &'static str {
        CUSTOM_CALL_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayProgramType],
        region_interfaces: &[RegionInterface<ArrayProgramType>],
    ) -> Result<Vec<ArrayProgramType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        let dynamic_output_dimensions = self
            .output_types
            .iter()
            .flat_map(|output_type| output_type.shape().dimensions())
            .filter_map(Dimension::variable)
            .collect::<Vec<_>>();
        let Some(array_input_count) = input_types.len().checked_sub(dynamic_output_dimensions.len()) else {
            return Err(TypeError::invalid(format!(
                "'{CUSTOM_CALL_OPERATION_NAME}' expects {} trailing output-extent dimensions but only {} inputs were \
                 provided",
                dynamic_output_dimensions.len(),
                input_types.len(),
            )));
        };
        for input_type in &input_types[..array_input_count] {
            <&ArrayType>::try_from(input_type)?;
        }
        for (input_type, expected_variable) in input_types[array_input_count..].iter().zip(dynamic_output_dimensions) {
            let actual_variable = <&crate::DimensionType>::try_from(input_type)?.variable();
            if actual_variable != expected_variable {
                return Err(TypeError::invalid(format!(
                    "'{CUSTOM_CALL_OPERATION_NAME}' output-extent operand defines dimension variable \
                     '{actual_variable}', but the corresponding declared output axis refers to \
                     '{expected_variable}'",
                )));
            }
        }
        Ok(self.output_types.iter().cloned().map(Into::into).collect())
    }

    #[inline]
    fn effects(&self) -> Effects {
        Operation::<ArrayType>::effects(self)
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        Operation::<ArrayType>::rename_type_identities(self, renaming)
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        Operation::<ArrayType>::render(self, formatter, indentation)
    }
}

impl<C: Domain<Type = ArrayType, Value: CustomCall>> InterpretableOperation<C> for CustomCallOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        C::Value::custom_call(self, inputs)
    }
}

impl<A: CustomCall + DimensionSize<usize> + Value<Type = ArrayType>>
    InterpretableOperation<EagerContext<ArrayProgramValue<A>, ArrayProgramOperation<A>>> for CustomCallOperation
{
    fn interpret<D: InterpretationDriver<EagerContext<ArrayProgramValue<A>, ArrayProgramOperation<A>>>>(
        &self,
        _context: &EagerContext<ArrayProgramValue<A>, ArrayProgramOperation<A>>,
        driver: &D,
        inputs: &[ArrayProgramValue<A>],
    ) -> Result<Vec<ArrayProgramValue<A>>, ProgramError> {
        if driver.region_count() != 0 {
            return Err(TypeError::invalid(format!("expected 0 regions but got {}", driver.region_count())).into());
        }
        self.infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(), &[])?;
        let dynamic_output_dimension_count = self
            .output_types
            .iter()
            .flat_map(|output_type| output_type.shape().dimensions())
            .filter(|dimension| matches!(dimension, Dimension::Dynamic(_)))
            .count();
        let array_input_count = inputs.len() - dynamic_output_dimension_count;
        let array_inputs = inputs[..array_input_count]
            .iter()
            .map(<ArrayProgramValue<A> as ValueProjection<ArrayType>>::projected)
            .collect::<Result<Vec<_>, _>>()?;
        let output_extents = inputs[array_input_count..]
            .iter()
            .map(<ArrayProgramValue<A> as ValueProjection<DimensionType>>::projected)
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = A::custom_call(self, array_inputs.iter().copied())?;
        check_count!("output", outputs, self.output_types.len(), ProgramError);
        let mut output_extents = output_extents.into_iter();
        for (output_index, (output, output_type)) in outputs.iter().zip(&self.output_types).enumerate() {
            for (axis, dimension) in output_type.shape().dimensions().iter().enumerate() {
                if matches!(dimension, Dimension::Dynamic(_)) {
                    let expected_extent = output_extents.next().unwrap().extent();
                    let actual_extent = output.dimension_size(axis)?;
                    if actual_extent != expected_extent {
                        return Err(ProgramError::InvalidArgument {
                            message: format!(
                                "'{CUSTOM_CALL_OPERATION_NAME}' output {output_index} axis {axis} has extent \
                                 {actual_extent}, but its explicit extent operand is {expected_extent}",
                            ),
                        });
                    }
                }
            }
        }
        Ok(outputs.into_iter().map(ArrayProgramValue::Array).collect())
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate). An all-known custom call folds into the
/// known side (executing there only if the known-side context can run foreign kernels), and a side-effecting
/// residual call survives dead-code elimination because [`Operation::effects`] is not [`Effects::PURE`].
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for CustomCallOperation where
    C::Operation: From<CustomCallOperation>
{
}

impl_differentiable_operation! {
    CustomCallOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Operation: From<CustomCallOperation>,
    {
        |operation, _context, _driver, _inputs| {
            // Foreign kernels are opaque, so there is no derivative to derive: differentiation reports an error
            // directing users to wrap the call with [`custom_jvp`](crate::tracing_v2::CustomJvp) or
            // [`custom_vjp`](crate::tracing_v2::CustomVjp), which is also how JAX handles `ffi_call` differentiation.
            Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "custom call '{}' has no differentiation rule; wrap it with `custom_jvp` or `custom_vjp` to \
                     provide one",
                    operation.target_name,
                ),
            }
            .into())
        }
    },
    transpose = @nonlinear,
}

/// Foreign kernels are opaque, so there is no batching rule to derive: batching reports an error, and callers
/// should invoke a kernel that understands the batch axis instead.
impl<C: Context<Type = ArrayType>, P: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>>
    for CustomCallOperation
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        _context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        _inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        Err(BatchingError::UnsupportedOperation {
            message: format!(
                "custom call '{}' has no batching rule; invoke a kernel that understands the batch axis instead",
                self.target_name,
            ),
        })
    }
}

/// Represents the ability to call foreign kernels registered with the executing backend. [`CustomCall`] stages or
/// executes a [`CustomCallOperation`]; refer to its documentation for the calling convention and the transform
/// rules. The capability method dispatches through the first input's context, so it needs at least one input
/// (zero-input custom calls can still be staged directly through a program builder).
pub trait CustomCall: Sized {
    /// Calls the foreign kernel described by `operation` with the provided inputs, returning one value per
    /// declared output type, and a [`ProgramError`] if something goes wrong.
    fn custom_call<'a, I: IntoIterator<Item = &'a Self>>(
        operation: &CustomCallOperation,
        inputs: I,
    ) -> Result<Vec<Self>, ProgramError>
    where
        Self: 'a;
}

/// Any context-carrying value calls foreign kernels by binding a [`CustomCallOperation`] through its own context.
/// The `From<CustomCallOperation>` bound makes this disjoint from the eager reference value types (whose context
/// operation is [`ConstantOperation`](crate::operations::constants::ConstantOperation)), so it covers the transform
/// tracers and backend-owned values without conflicting with concrete implementations.
impl<V: Value<Type = ArrayType>> CustomCall for V
where
    V::DispatchDomain: Context<Operation: From<CustomCallOperation>>,
{
    fn custom_call<'a, I: IntoIterator<Item = &'a Self>>(
        operation: &CustomCallOperation,
        inputs: I,
    ) -> Result<Vec<Self>, ProgramError>
    where
        Self: 'a,
    {
        let inputs = inputs.into_iter().cloned().collect::<Vec<_>>();
        let Some(first) = inputs.first() else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "the custom-call capability dispatches through its first input's context, so calling '{}' with \
                     no inputs requires staging the operation through a program builder instead",
                    operation.target_name(),
                ),
            });
        };
        first.dispatch_domain().bind(operation.clone(), Vec::new(), inputs.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::batching::{BatchAxis, ProgramBatchingOutputAxesPolicy};
    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::sharding::ShardingDimension;
    use crate::tracing::{DomainTracer, Trace};
    use crate::types::dimensions::{DimensionBounds, DimensionVariable};
    use crate::types::{DataType, Dimension, DimensionType, Shape};

    use super::*;

    /// Returns the `f32[2]` array type used throughout these tests.
    fn vector_type() -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]))
    }

    #[test]
    fn test_custom_call_operation_contract() {
        let operation = CustomCallOperation::new("ryft.test.add_one", vec![vector_type()])
            .with_attribute("scale", 2.0)
            .with_attribute("count", 4i64)
            .with_attribute("verbose", true)
            .with_attribute("label", "x")
            .with_side_effect();

        assert_eq!(operation.target_name(), "ryft.test.add_one");
        assert_eq!(operation.output_types(), &[vector_type()]);
        assert_eq!(
            operation.attributes(),
            &[
                ("scale".to_string(), CustomCallAttribute::F64(2.0)),
                ("count".to_string(), CustomCallAttribute::I64(4)),
                ("verbose".to_string(), CustomCallAttribute::Boolean(true)),
                ("label".to_string(), CustomCallAttribute::String("x".to_string())),
            ],
        );
        assert!(operation.has_side_effect());
        assert_eq!(Operation::<ArrayType>::name(&operation), CUSTOM_CALL_OPERATION_NAME);
        assert_eq!(Operation::<ArrayType>::effects(&operation), Effects::single(Effect::OrderedIo));
        assert_eq!(operation.infer_output_types(&[vector_type()], &[]), Ok(vec![vector_type()]),);
        // Long attribute lists wrap onto one line per field.
        assert_eq!(
            operation.to_string(),
            indoc! {"
                custom_call [
                    target=ryft.test.add_one,
                    scale=2.0,
                    count=4,
                    verbose=true,
                    label=x,
                    has_side_effect=true,
                ]
            "}
            .trim_end(),
        );

        let pure = CustomCallOperation::new("ryft.test.add_one", vec![vector_type()]);
        assert_eq!(Operation::<ArrayType>::effects(&pure), Effects::PURE);
        assert_eq!(pure.to_string(), "custom_call [target=ryft.test.add_one]");

        // Composite output extents are positional SSA operands: output-major and then axis-major.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::new(2, Some(17)).unwrap());
        let dynamic_output_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                Dimension::Dynamic(rows.clone()),
                Dimension::Static(3),
                Dimension::Dynamic(columns.clone()),
            ]),
        );
        let dynamic_operation = CustomCallOperation::new("ryft.test.dynamic", vec![dynamic_output_type.clone()]);
        let input_types = vec![
            vector_type().into(),
            DimensionType::new(rows.clone()).into(),
            DimensionType::new(columns.clone()).into(),
        ];
        assert_eq!(
            Operation::<ArrayProgramType>::infer_output_types(&dynamic_operation, &input_types, &[]),
            Ok(vec![dynamic_output_type.into()]),
        );
        assert_eq!(
            Operation::<ArrayProgramType>::infer_output_types(
                &dynamic_operation,
                &[vector_type().into(), DimensionType::new(columns).into(), DimensionType::new(rows).into()],
                &[],
            ),
            Err(TypeError::invalid(
                "'custom_call' output-extent operand defines dimension variable 'columns', but the corresponding \
                 declared output axis refers to 'rows'",
            )),
        );
        assert_eq!(
            Operation::<ArrayProgramType>::infer_output_types(
                &dynamic_operation,
                &[DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap(),))
                    .into()],
                &[],
            ),
            Err(TypeError::invalid(
                "'custom_call' expects 2 trailing output-extent dimensions but only 1 inputs were provided",
            )),
        );
    }

    #[test]
    fn test_custom_call_stages_through_the_tracer_capability() {
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| {
                let operation = CustomCallOperation::new("ryft.test.add_one", vec![vector_type()]);
                Ok(CustomCall::custom_call(&operation, std::slice::from_ref(&x))?.remove(0))
            },
            vector_type(),
        )
        .unwrap();
        let program = program.to_flat_program();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = custom_call [target=ryft.test.add_one] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_call_is_rejected_by_the_reference_backend() {
        let operation = CustomCallOperation::new("ryft.test.add_one", vec![vector_type()]);
        let result = InterpretableOperation::<EagerContext<Array>>::interpret(
            &operation,
            &EagerContext::new(),
            &EmptyRegionDriver,
            &[Array::vector(vec![1.0, 2.0])],
        );
        assert!(matches!(
            result,
            Err(ProgramError::UnsupportedOperation { message })
                if message == "the reference array backend cannot execute the foreign kernel 'ryft.test.add_one'",
        ));
    }

    #[test]
    fn test_custom_call_rejects_differentiation_and_batching() {
        let operation = CustomCallOperation::new("ryft.test.add_one", vec![vector_type()]);

        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(vector_type());
        let output = builder.add_instruction(operation.clone(), Vec::new(), vec![input]).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        assert!(matches!(
            program.jvp(),
            Err(error)
                if error.to_string().contains("custom call 'ryft.test.add_one' has no differentiation rule"),
        ));
        assert!(matches!(
            program.batched(2, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural),
            Err(error) if error.to_string().contains("custom call 'ryft.test.add_one' has no batching rule"),
        ));
    }
}
