use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use half::{bf16, f16};

use crate::differentiation::Tangent;
use crate::macros::check_count;
use crate::operations::{ElementwiseOperation, InterpretableOperation, Operation, OperationFormatter};
use crate::tracing::{Traceable, Tracer, TracingDomain, TracingError};
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

/// Canonical operation name for [`ScaleOperation`].
pub const SCALE_OPERATION_NAME: &'static str = "scale";

/// Unary operation that multiplies its input by a captured factor. In ordinary programs this represents "multiply by a
/// closed-over constant." In linear programs the same semantic idea is reused to scale tangent and cotangent terms.
#[derive(Clone, Debug)]
pub struct ScaleOperation<T: Type, V: Typed<T>> {
    /// Captured factor applied to every input of this unary [`Operation`].
    factor: V,

    /// [`PhantomData`] marker tying the captured factor to the abstract type it is interpreted against.
    marker: PhantomData<T>,
}

impl<T: Type, V: Typed<T>> ScaleOperation<T, V> {
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

impl<T: Type, V: Debug + Display + Typed<T>> Display for ScaleOperation<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(SCALE_OPERATION_NAME)
    }
}

impl<V: Debug + Display + Typed<DataType>> Operation<DataType> for ScaleOperation<DataType, V> {
    #[inline]
    fn name(&self) -> &'static str {
        SCALE_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![input_types[0].clone()])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, Operation::name(self))?
            .bracketed(|operation| operation.field("factor", &self.factor))
    }
}

impl<V: Debug + Display + Typed<ArrayType>> ElementwiseOperation for ScaleOperation<ArrayType, V> {
    #[inline]
    fn name(&self) -> &'static str {
        SCALE_OPERATION_NAME
    }

    #[inline]
    fn input_count(&self) -> usize {
        1
    }
}

impl<V: Clone + Typed<DataType> + Scale<F, Output = V>, F: Clone + Debug + Display + Typed<DataType>>
    InterpretableOperation<DataType, V> for ScaleOperation<DataType, F>
{
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![inputs[0].clone().scale(self.factor.clone())])
    }
}

impl<V: Clone + Typed<ArrayType> + Scale<F, Output = V>, F: Clone + Debug + Display + Typed<ArrayType>>
    InterpretableOperation<ArrayType, V> for ScaleOperation<ArrayType, F>
{
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![inputs[0].clone().scale(self.factor.clone())])
    }
}

/// Trait that represents [`Operation`] carrier types that support/include [`ScaleOperation`]. Backend-owned closed
/// [`Operation`] carrier types implement this trait so that generic transform code can stage [`ScaleOperation`] without
/// knowing which carrier is in use.
pub trait SupportsScale<T: Type, V: Traceable<T>, F: Traceable<T> = V> {
    /// Constructs the carrier-specific representation of [`ScaleOperation`].
    fn scale_operation(factor: F) -> Self;
}

/// Value-level scaling capability. [`Scale`] fills the same role for [`ScaleOperation`] that [`std::ops::Add`] and
/// [`std::ops::Neg`] fill for their corresponding arithmetic [`Operation`]s.
pub trait Scale<Factor = Self> {
    /// Resulting type after applying this operation.
    type Output;

    /// Scales this value by `factor`.
    fn scale(self, factor: Factor) -> Self::Output;
}

macro_rules! impl_scale_for_scalar {
    ($ty:ty) => {
        impl Scale for $ty {
            type Output = Self;

            #[inline]
            fn scale(self, factor: Self) -> Self::Output {
                factor * self
            }
        }
    };
}

impl_scale_for_scalar!(i8);
impl_scale_for_scalar!(i16);
impl_scale_for_scalar!(i32);
impl_scale_for_scalar!(i64);
impl_scale_for_scalar!(u8);
impl_scale_for_scalar!(u16);
impl_scale_for_scalar!(u32);
impl_scale_for_scalar!(u64);
impl_scale_for_scalar!(bf16);
impl_scale_for_scalar!(f16);
impl_scale_for_scalar!(f32);
impl_scale_for_scalar!(f64);

impl<'domain, D: TracingDomain<OperationCarrier: SupportsScale<D::Type, D::Value, F>>, F: Traceable<D::Type>> Scale<F>
    for Tracer<'domain, D>
{
    type Output = Self;

    #[inline]
    fn scale(self, factor: F) -> Self::Output {
        self.unary(D::OperationCarrier::scale_operation(factor))
    }
}

impl<T: Type, V: Traceable<T> + Scale<Factor, Output = V>, Factor> Scale<Factor> for Tangent<T, V> {
    type Output = Self;

    #[inline]
    fn scale(self, factor: Factor) -> Self::Output {
        match self {
            Self::Zero(r#type) => Self::Zero(r#type),
            Self::Value(value) => Self::Value(value.scale(factor)),
        }
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing::{ProgramBuilder, TracingError};
    use crate::types::{ArrayType, DataType, Layout, Shape, Size, StridedLayout};

    use super::*;

    #[test]
    fn test_scale() {
        let operation = ScaleOperation::<DataType, f64>::new(3.0);

        // Operation identity and concrete interpretation.
        assert_eq!(Operation::<DataType>::name(&operation), SCALE_OPERATION_NAME);
        assert_eq!(
            format!("{operation:?}"),
            "ScaleOperation { factor: 3.0, marker: PhantomData<ryft_core::types::data_types::DataType> }"
        );
        assert_eq!(format!("{operation}"), SCALE_OPERATION_NAME);
        assert_eq!(Operation::<DataType>::infer_output_types(&operation, &[DataType::F32]), Ok(vec![DataType::F32]),);
        assert_eq!(<f64 as Scale>::scale(2.0, 3.0), 6.0);
        assert_eq!(InterpretableOperation::<DataType, f64>::interpret(&operation, &[2.0]), Ok(vec![6.0]));

        let array_operation = ScaleOperation::<ArrayType, f64>::new(3.0);
        assert_eq!(InterpretableOperation::<ArrayType, f64>::interpret(&array_operation, &[2.0]), Ok(vec![6.0]),);

        // Array type inference preserves shape, layout, and sharding metadata for its single input.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let input = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Size::Static(2), Size::Static(3)]),
            Some(Layout::Strided(StridedLayout::new(vec![3, 1]))),
            Some(
                Sharding::with_manual_axes(
                    mesh,
                    vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["x"],
                )
                .unwrap(),
            ),
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
            InterpretableOperation::<DataType, f64>::interpret(&operation, &[]),
            Err(TracingError::InvalidInputCount { expected: 1, got: 0 }),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, f64>::interpret(&array_operation, &[]),
            Err(TracingError::InvalidInputCount { expected: 1, got: 0 }),
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
