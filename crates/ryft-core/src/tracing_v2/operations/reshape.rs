use std::fmt::{Debug, Display};

use half::{bf16, f16};

use crate::macros::check_input_count;
use crate::sharding::{Sharding, ShardingDimension};
use crate::tracing::engines::{Tracer, TracingEngine};
use crate::tracing::{AtomId, OperationFormatter, Traceable, TracingError};
use crate::tracing_v2::forward::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableEngine, LinearArrayOperation};
use crate::types::{ArrayType, Shape, Size, Type, TypeError, Typed};

use super::{DifferentiableOperation, InterpretableOperation, LinearOperation, Operation};

/// Hidden carrier capability for staging the reshape primitive.
#[doc(hidden)]
pub trait SupportsReshape<T: Type, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the reshape primitive.
    fn reshape_operation(input_shape: Shape, output_shape: Shape) -> Self;
}

/// Returns `true` when `dimension` is explicitly unsharded in the JAX sense.
fn is_effectively_unsharded_dimension(dimension: &ShardingDimension) -> bool {
    matches!(dimension, ShardingDimension::Replicated)
}

/// Returns the static element count of `shape`, or `None` when any dimension is dynamic or the product overflows.
fn static_shape_element_count(shape: &Shape) -> Option<usize> {
    shape.dimensions.iter().try_fold(1usize, |count, size| match size {
        Size::Static(value) => count.checked_mul(*value),
        Size::Dynamic(_) => None,
    })
}

/// Returns the non-singleton static dimensions of `shape` together with their original indices.
fn non_singleton_shape_dimensions(shape: &Shape) -> Vec<(usize, usize)> {
    shape
        .dimensions
        .iter()
        .enumerate()
        .filter_map(|(index, size)| match size {
            Size::Static(1) => None,
            Size::Static(value) => Some((index, *value)),
            Size::Dynamic(_) => None,
        })
        .collect()
}

/// Partitions two non-singleton shapes into corresponding contiguous groups with matching element counts.
fn reshape_dimension_groups(
    input_dimensions: &[(usize, usize)],
    output_dimensions: &[(usize, usize)],
) -> Option<Vec<(usize, usize, usize, usize)>> {
    let mut input_start_index = 0usize;
    let mut output_start_index = 0usize;
    let mut groups = Vec::new();

    while input_start_index < input_dimensions.len() || output_start_index < output_dimensions.len() {
        if input_start_index == input_dimensions.len() || output_start_index == output_dimensions.len() {
            return None;
        }

        let input_group_start_index = input_start_index;
        let output_group_start_index = output_start_index;
        let mut input_group_product = input_dimensions[input_start_index].1;
        let mut output_group_product = output_dimensions[output_start_index].1;
        input_start_index += 1;
        output_start_index += 1;

        while input_group_product != output_group_product {
            if input_group_product < output_group_product {
                if input_start_index == input_dimensions.len() {
                    return None;
                }
                input_group_product = input_group_product.checked_mul(input_dimensions[input_start_index].1)?;
                input_start_index += 1;
            } else {
                if output_start_index == output_dimensions.len() {
                    return None;
                }
                output_group_product = output_group_product.checked_mul(output_dimensions[output_start_index].1)?;
                output_start_index += 1;
            }
        }

        groups.push((input_group_start_index, input_start_index, output_group_start_index, output_start_index));
    }

    Some(groups)
}

/// Propagates reshape sharding using JAX-style singleton stripping and contiguous split/merge grouping.
fn reshape_array_sharding(
    input: &ArrayType,
    target_shape: &Shape,
    op: &'static str,
) -> Result<Option<Sharding>, TypeError> {
    let Some(sharding) = input.sharding.clone() else {
        return Ok(None);
    };
    if input.shape == *target_shape {
        return Ok(Some(sharding));
    }

    let input_non_singleton_dimensions = non_singleton_shape_dimensions(&input.shape);
    let output_non_singleton_dimensions = non_singleton_shape_dimensions(target_shape);
    let Some(groups) =
        reshape_dimension_groups(input_non_singleton_dimensions.as_slice(), output_non_singleton_dimensions.as_slice())
    else {
        return Err(TypeError { message: format!("{op} could not align static reshape dimension groups") });
    };

    let mut output_dimensions =
        std::iter::repeat_n(ShardingDimension::replicated(), target_shape.rank()).collect::<Vec<_>>();
    for (input_group_start_index, input_group_end_index, output_group_start_index, output_group_end_index) in groups {
        let input_group_length = input_group_end_index - input_group_start_index;
        let output_group_length = output_group_end_index - output_group_start_index;
        if input_group_length == 1 && output_group_length == 1 {
            let input_dimension_index = input_non_singleton_dimensions[input_group_start_index].0;
            let output_dimension_index = output_non_singleton_dimensions[output_group_start_index].0;
            output_dimensions[output_dimension_index] = sharding.dimensions[input_dimension_index].clone();
            continue;
        }

        if !input_non_singleton_dimensions[input_group_start_index..input_group_end_index]
            .iter()
            .map(|(index, _)| &sharding.dimensions[*index])
            .all(is_effectively_unsharded_dimension)
        {
            return Err(TypeError { message: format!("{op} cannot preserve sharding across the requested reshape") });
        }

        for (output_dimension_index, _) in
            output_non_singleton_dimensions[output_group_start_index..output_group_end_index].iter()
        {
            output_dimensions[*output_dimension_index] = ShardingDimension::replicated();
        }
    }

    Sharding::with_manual_axes(
        sharding.mesh.clone(),
        output_dimensions,
        sharding.unreduced_axes.clone(),
        sharding.reduced_manual_axes.clone(),
        sharding.varying_manual_axes.clone(),
    )
    .map(|sharding| Some(sharding.without_auto_axes()))
    .map_err(|_| TypeError { message: format!("{op} produced an invalid output sharding") })
}

/// Computes the abstract output type of one reshape application.
pub fn reshape_abstract(input: &ArrayType, target_shape: &Shape, op: &'static str) -> Result<ArrayType, TypeError> {
    if input.shape == *target_shape {
        return Ok(input.clone());
    }

    let Some(input_elements) = static_shape_element_count(&input.shape) else {
        return Err(TypeError { message: format!("{op} requires statically known input element counts") });
    };
    let Some(output_elements) = static_shape_element_count(target_shape) else {
        return Err(TypeError { message: format!("{op} requires statically known output element counts") });
    };
    if input_elements != output_elements {
        return Err(TypeError { message: format!("{op} changes the number of elements") });
    }

    ArrayType::new(input.data_type, target_shape.clone(), None, reshape_array_sharding(input, target_shape, op)?)
        .map_err(|_| TypeError { message: format!("{op} produced an invalid output type") })
}

/// Value-level reshape capability shared by concrete leaves and transform-local wrappers.
///
/// This trait is intentionally fallible because keeping the same Rust type before and after the
/// reshape may rule out some logically valid target shapes for a given leaf representation.
pub trait ReshapeOps: Sized {
    /// Reshapes `self` to `target_shape`.
    ///
    /// Implementors keep the same Rust type before and after the reshape, so some value types can only accept a
    /// subset of logically valid shapes.
    fn reshape(self, target_shape: Shape) -> Result<Self, TracingError>;
}

/// Convenience trait for traceable leaves that can serve as the concrete values of a staged reshape.
///
/// This is the trait bound most reshape-aware transforms use when they need both the abstract leaf
/// contract and the value-level reshape operation.
pub trait ReshapeValue: Traceable<ArrayType> + ReshapeOps {}

impl<T: Traceable<ArrayType> + ReshapeOps> ReshapeValue for T {}

impl<'engine, V: Traceable<ArrayType>, E> ReshapeOps for Tracer<'engine, E>
where
    E: TracingEngine<Type = ArrayType, Value = V> + ?Sized,
    E::Operation: SupportsReshape<ArrayType, V>,
{
    fn reshape(self, target_shape: Shape) -> Result<Self, TracingError> {
        let input_type = self.r#type().into_owned();
        let output_type = reshape_abstract(&input_type, &target_shape, "reshape")?;
        if input_type == output_type {
            return Ok(self);
        }
        let context = self.context.clone();
        Ok(context
            .trace(E::Operation::reshape_operation(input_type.shape.clone(), output_type.shape.clone()), &[&self])?
            .into_iter()
            .next()
            .expect("reshape should produce one traced output"))
    }
}

macro_rules! impl_scalar_reshape_ops {
    ($($ty:ty),* $(,)?) => {
        $(
            impl ReshapeOps for $ty {
                fn reshape(self, target_shape: Shape) -> Result<Self, TracingError> {
                    reshape_abstract(&self.r#type(), &target_shape, "reshape")?;
                    Ok(self)
                }
            }
        )*
    };
}

impl_scalar_reshape_ops!(bf16, f16, f32, f64);

/// Primitive representing one reshape between two [`Shape`]s.
#[derive(Clone)]
pub struct ReshapeOperation {
    /// Shape expected from the input.
    input_shape: Shape,

    /// Shape produced by the reshape.
    output_shape: Shape,
}

impl ReshapeOperation {
    /// Creates a reshape op from `input_shape` to `output_shape`.
    pub fn new(input_shape: Shape, output_shape: Shape) -> Self {
        Self { input_shape, output_shape }
    }

    /// Returns the expected input shape.
    pub fn input_shape(&self) -> &Shape {
        &self.input_shape
    }

    /// Returns the produced output shape.
    pub fn output_shape(&self) -> &Shape {
        &self.output_shape
    }
}

impl Debug for ReshapeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Reshape")
    }
}

impl Display for ReshapeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "reshape{}", self.output_shape())
    }
}

impl Operation<ArrayType> for ReshapeOperation {
    fn name(&self) -> &'static str {
        "reshape"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        if input_types.len() != 1 {
            return Err(TypeError { message: format!("reshape expected 1 input type but got {}", input_types.len()) });
        }
        if input_types[0].shape != *self.input_shape() {
            return Err(TypeError {
                message: format!(
                    "reshape expected input shape {} but got {}",
                    self.input_shape(),
                    input_types[0].shape
                ),
            });
        }
        Ok(vec![reshape_abstract(&input_types[0], self.output_shape(), "reshape")?])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("input_shape", self.input_shape())?;
            operation.field("output_shape", self.output_shape())
        })
    }
}

impl<V: ReshapeValue> InterpretableOperation<ArrayType, V> for ReshapeOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 1, TracingError);
        Ok(vec![inputs[0].clone().reshape(self.output_shape().clone())?])
    }
}

impl<V: ReshapeValue> LinearOperation<ArrayType, V, LinearArrayOperation<V>> for ReshapeOperation {
    fn transpose(
        &self,
        context: &mut crate::tracing_v2::operations::TranspositionContext<'_, ArrayType, V, LinearArrayOperation<V>>,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        check_input_count!(output_cotangents, 1, TracingError);
        let Some(atom) = output_cotangents[0] else {
            return Ok(vec![None]);
        };
        if self.input_shape() == self.output_shape() {
            return Ok(vec![Some(atom)]);
        }
        Ok(vec![Some(
            context
                .apply_operation(
                    &[atom],
                    LinearArrayOperation::Reshape {
                        input_shape: self.output_shape().clone(),
                        output_shape: self.input_shape().clone(),
                    },
                    1,
                )?
                .into_iter()
                .next()
                .expect("reshape should produce one cotangent contribution"),
        )])
    }
}

impl<E> DifferentiableOperation<E> for ReshapeOperation
where
    E: DifferentiableEngine<Type = ArrayType> + ?Sized,
    E::Value: ReshapeValue + Differentiable<ArrayType, Tangent = E::Value>,
    E::LinearOperation: SupportsReshape<ArrayType, E::Value>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_input_count!(inputs, 1, TracingError);
        let primal = inputs[0].primal.clone().reshape(self.output_shape().clone())?;
        let tangent = context
            .apply_operation(
                &[inputs[0].tangent],
                <E::LinearOperation as SupportsReshape<ArrayType, E::Value>>::reshape_operation(
                    self.input_shape().clone(),
                    self.output_shape().clone(),
                ),
                1,
            )?
            .into_iter()
            .next()
            .expect("reshape jvp should produce one tangent");
        Ok(vec![JvpTracer { primal, tangent }])
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding};
    use crate::types::{DataType, Shape};

    use super::*;

    /// Creates one small manual mesh used by reshape sharding tests.
    fn test_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap()
    }

    #[test]
    fn test_reshape_abstract_preserves_sharding_across_inserted_singleton_axes() {
        let mesh = test_mesh();
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(8)]),
            None,
            Some(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()),
        )
        .unwrap();

        assert_eq!(
            reshape_abstract(
                &input_type,
                &Shape::new(vec![Size::Static(1), Size::Static(8), Size::Static(1)]),
                "reshape",
            ),
            Ok(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Size::Static(1), Size::Static(8), Size::Static(1)]),
                None,
                Some(
                    Sharding::new(
                        mesh,
                        vec![
                            ShardingDimension::replicated(),
                            ShardingDimension::sharded(["x"]),
                            ShardingDimension::replicated(),
                        ],
                    )
                    .unwrap(),
                ),
            )
            .unwrap())
        );
    }

    #[test]
    fn test_reshape_abstract_merges_replicated_axes_and_preserves_unchanged_sharding() {
        let mesh = test_mesh();
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(8), Size::Static(2), Size::Static(3)]),
            None,
            Some(
                Sharding::new(
                    mesh.clone(),
                    vec![
                        ShardingDimension::sharded(["x"]),
                        ShardingDimension::replicated(),
                        ShardingDimension::replicated(),
                    ],
                )
                .unwrap(),
            ),
        )
        .unwrap();

        assert_eq!(
            reshape_abstract(&input_type, &Shape::new(vec![Size::Static(8), Size::Static(6)]), "reshape"),
            Ok(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Size::Static(8), Size::Static(6)]),
                None,
                Some(
                    Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],)
                        .unwrap(),
                ),
            )
            .unwrap())
        );
    }

    #[test]
    fn test_reshape_abstract_splits_replicated_axis_and_preserves_unchanged_sharding() {
        let mesh = test_mesh();
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(8), Size::Static(6)]),
            None,
            Some(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            ),
        )
        .unwrap();

        assert_eq!(
            reshape_abstract(
                &input_type,
                &Shape::new(vec![Size::Static(8), Size::Static(2), Size::Static(3)]),
                "reshape",
            ),
            Ok(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Size::Static(8), Size::Static(2), Size::Static(3)]),
                None,
                Some(
                    Sharding::new(
                        mesh,
                        vec![
                            ShardingDimension::sharded(["x"]),
                            ShardingDimension::replicated(),
                            ShardingDimension::replicated(),
                        ],
                    )
                    .unwrap(),
                ),
            )
            .unwrap())
        );
    }

    #[test]
    fn test_reshape_abstract_rejects_mismatched_element_counts() {
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]), None, None).unwrap();

        assert_eq!(
            reshape_abstract(&input_type, &Shape::new(vec![Size::Static(5)]), "reshape"),
            Err(TypeError { message: "reshape changes the number of elements".to_string() })
        );
    }

    #[test]
    fn test_reshape_abstract_rejects_partitioned_split() {
        let mesh = test_mesh();
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(8)]),
            None,
            Some(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap()),
        )
        .unwrap();

        assert_eq!(
            reshape_abstract(&input_type, &Shape::new(vec![Size::Static(2), Size::Static(4)]), "reshape"),
            Err(TypeError { message: "reshape cannot preserve sharding across the requested reshape".to_string() })
        );
    }

    #[test]
    fn test_reshape_abstract_rejects_partitioned_merge() {
        let mesh = test_mesh();
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(2), Size::Static(4)]),
            None,
            Some(
                Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])]).unwrap(),
            ),
        )
        .unwrap();

        assert_eq!(
            reshape_abstract(&input_type, &Shape::new(vec![Size::Static(8)]), "reshape"),
            Err(TypeError { message: "reshape cannot preserve sharding across the requested reshape".to_string() })
        );
    }

    #[test]
    fn test_reshape_abstract_allows_unsharded_many_to_many_group() {
        let mesh = test_mesh();
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(2), Size::Static(6)]),
            None,
            Some(
                Sharding::with_manual_axes(
                    mesh.clone(),
                    vec![ShardingDimension::replicated(), ShardingDimension::replicated()],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["x"],
                )
                .unwrap(),
            ),
        )
        .unwrap();

        assert_eq!(
            reshape_abstract(&input_type, &Shape::new(vec![Size::Static(3), Size::Static(4)]), "reshape"),
            Ok(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Size::Static(3), Size::Static(4)]),
                None,
                Some(
                    Sharding::with_manual_axes(
                        mesh,
                        vec![ShardingDimension::replicated(), ShardingDimension::replicated()],
                        Vec::<&str>::new(),
                        Vec::<&str>::new(),
                        ["x"],
                    )
                    .unwrap(),
                ),
            )
            .unwrap())
        );
    }
}
