use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops::Mul;

use crate::contexts::{EagerContext, StagingContext};
use crate::macros::{check_builders, check_count};
use crate::operations::{ElementwiseOperation, InterpretableOperation, Operation, OperationFormatter};
use crate::payloads::{Captured, Input};
use crate::programs::{ProgramError, Value};
use crate::tracing::Tracer;
use crate::types::{ArrayType, Type, TypeError, Typed};

/// Canonical operation name for [`ScaleOperation`].
pub const SCALE_OPERATION_NAME: &'static str = "scale";

/// Unary operation that multiplies its input by a captured factor. In ordinary programs this represents "multiply by a
/// closed-over constant." In linear programs the same semantic idea is reused to scale tangent and cotangent terms.
///
/// The `Payload` type parameter is a zero-sized semantic tag that tells interpretation how the factor should be
/// treated. The default [`Captured`] payload means that the factor is carried by the operation and can be staged as a
/// unary `scale` [`Instruction`](crate::Instruction). The [`Input`] payload is used when the factor is already part of
/// the active runtime or [`StagingContext`], such as a [`Tracer`] factor produced while rewriting a linear program. In
/// that case, interpretation lowers the operation to ordinary multiplication after validating that both operands belong
/// to the same builder. Keeping this distinction in the operation type lets [`Captured`]-factor and [`Input`]-factor
/// scaling share one operation struct without adding runtime fields or ambiguous interpretation implementations.
#[derive(Clone)]
pub struct ScaleOperation<T: Type, V: Typed<T>, Payload = Captured> {
    /// Captured factor applied to every input of this unary [`Operation`].
    factor: V,

    /// [`PhantomData`] marker tying the captured factor to the abstract type it is interpreted against and to its
    /// payload role. The `fn() -> ...` form indexes by type without owning one, and so this operation's `Send` and
    /// `Sync` depend only on the captured value.
    marker: PhantomData<fn() -> (T, Payload)>,
}

impl<T: Type, V: Typed<T>, Payload> ScaleOperation<T, V, Payload> {
    /// Creates a new [`ScaleOperation`] capturing the provided factor.
    #[inline]
    pub fn new(factor: V) -> Self {
        Self { factor, marker: PhantomData }
    }

    /// Returns the captured factor applied by this operation.
    #[inline]
    pub fn factor(&self) -> &V {
        &self.factor
    }
}

impl<T: Type, V: Debug + Typed<T>, Payload> Debug for ScaleOperation<T, V, Payload> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("ScaleOperation").field("factor", &self.factor).finish()
    }
}

impl<T: Type, V: Display + Typed<T>, Payload> Display for ScaleOperation<T, V, Payload> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type, V: Display + Typed<T>, Payload> Operation<T> for ScaleOperation<T, V, Payload> {
    #[inline]
    fn name(&self) -> &'static str {
        SCALE_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![input_types[0].clone()])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, Operation::name(self))?
            .bracketed(|operation| operation.field("factor", &self.factor))
    }
}

impl<V: Display + Typed<ArrayType>, Payload> ElementwiseOperation for ScaleOperation<ArrayType, V, Payload> {
    #[inline]
    fn input_count(&self) -> usize {
        1
    }
}

impl<
    T: Type,
    V: Clone + Display + Typed<T>,
    F: Clone + Value<T, InterpretationContext: Scale<T, F, V, Payload>>,
    Payload,
> InterpretableOperation<T, F> for ScaleOperation<T, V, Payload>
{
    #[inline]
    fn interpret(
        &self,
        context: &<F as Value<T>>::InterpretationContext,
        inputs: &[F],
    ) -> Result<Vec<F>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![context.scale(&inputs[0], self.factor.clone())?])
    }
}

/// Represents the ability to interpret a [`ScaleOperation`]. [`Captured`] factors stage a unary `scale`
/// [`Instruction`](crate::Instruction), while [`Input`] [`Tracer`] factors lower to ordinary multiplication in the
/// active context. [`EagerContext`]s interpret scaling through [`Mul`].
///
/// The `Payload` parameter mirrors [`ScaleOperation`]'s payload tag and selects the context-specific interpretation
/// path. [`Captured`] payloads are factors carried by the operation and may remain as scale payloads in staged linear
/// programs, while [`Input`] payloads are already context values and should be consumed as operands, typically by
/// lowering to multiplication. This type-level tag keeps those semantics explicit even when the factor type itself is
/// otherwise the same.
pub trait Scale<T: Type, V: Value<T>, F, Payload = Captured> {
    /// Scales `input` by `factor`.
    fn scale(&self, input: &V, factor: F) -> Result<V, ProgramError>;
}

impl<T: Type, V: Clone + Value<T> + Mul<F, Output = V>, F, O: Operation<T>, Payload> Scale<T, V, F, Payload>
    for EagerContext<T, V, O>
{
    #[inline]
    fn scale(&self, input: &V, factor: F) -> Result<V, ProgramError> {
        Ok(input.clone() * factor)
    }
}

impl<C: StagingContext<Operation: From<ScaleOperation<C::Type, F>>>, F: Clone + Display + Typed<C::Type>>
    Scale<C::Type, Tracer<C>, F, Captured> for C
{
    #[inline]
    fn scale(&self, input: &Tracer<C>, factor: F) -> Result<Tracer<C>, ProgramError> {
        let mut outputs = self.stage_operation(ScaleOperation::new(factor), &[input])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: StagingContext> Scale<C::Type, Tracer<C>, Tracer<C>, Input> for C
where
    Tracer<C>: Mul<Output = Tracer<C>>,
{
    #[inline]
    fn scale(&self, input: &Tracer<C>, factor: Tracer<C>) -> Result<Tracer<C>, ProgramError> {
        check_builders!(self.builder(), [[input.context().builder(), factor.context().builder()]])
            .map_err(|error| self.error(error))?;
        Ok(factor * input.clone())
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tests::TestArray;
    use crate::types::{ArrayType, DataType, Layout, Shape, Size, StridedLayout};

    use super::*;

    #[test]
    fn test_scale() {
        let operation = ScaleOperation::<DataType, f64>::new(3.0);

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<DataType>::name(&operation), SCALE_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "ScaleOperation { factor: 3.0 }");
        assert_eq!(format!("{operation}"), "scale [factor=3]");
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[DataType::F32]), Ok(vec![DataType::F32]),);
        assert_eq!(2.0 * 3.0, 6.0);
        assert_eq!(
            InterpretableOperation::<DataType, f64>::interpret(&operation, &EagerContext::new(), &[2.0]),
            Ok(vec![6.0]),
        );

        // Array-side rendering goes through the `ElementwiseOperation::render` override and must include the captured
        // factor, matching the scalar `Operation<DataType>::render` above so that wrapping enum variants delegate
        // faithfully.
        let array_operation = ScaleOperation::<ArrayType, TestArray>::new(TestArray::scalar(3.0));
        assert_eq!(format!("{array_operation}"), "scale [factor=[3.0]]");
        assert_eq!(
            InterpretableOperation::<ArrayType, TestArray>::interpret(
                &array_operation,
                &EagerContext::new(),
                &[TestArray::scalar(2.0)],
            ),
            Ok(vec![TestArray::scalar(6.0)]),
        );

        // Array type inference preserves shape, layout, and sharding metadata for its single input.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let input = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![3, 1])))
            .with_sharding(
                Sharding::with_manual_axes(
                    mesh,
                    vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["x"],
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&array_operation, std::slice::from_ref(&input)),
            Ok(vec![input]),
        );

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            Operation::<DataType>::infer_output_types(&operation, &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&array_operation, &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<DataType, f64>::interpret(&operation, &EagerContext::new(), &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, TestArray>::interpret(&array_operation, &EagerContext::new(), &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // Program rendering uses the canonical operation name and includes the captured factor.
        let mut builder = ProgramBuilder::<DataType, f64, ScaleOperation<DataType, f64>>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation, vec![input]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = scale [factor=3] %0
                in (%1)
            "}
            .trim_end(),
        );
    }
}
