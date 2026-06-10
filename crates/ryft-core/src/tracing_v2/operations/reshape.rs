use std::fmt::Display;

use crate::contexts::StagingContext;
use crate::differentiation::{Cotangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::programs::{ProgramError, Value};
use crate::sharding::{Sharding, ShardingDimension};
use crate::tracing::{AbstractTracingContext, Tracer};
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext};
use crate::types::{ArrayType, Shape, Size, Type, TypeError, Typed};

/// Trait for operation types that include or can wrap [`ReshapeOperation`]. Backend-owned closed
/// [`Operation`] operation types (such as [`ArrayOperation`](super::ArrayOperation), for example) implement this trait
/// so that generic transform code can stage [`ReshapeOperation`] without knowing the concrete operation enum.
#[doc(hidden)]
pub trait SupportsReshape<T: Type> {
    /// Constructs the backend-specific representation of the reshape [`Operation`].
    fn reshape_operation(output_shape: Shape) -> Self;
}

/// Returns `true` when `dimension` is explicitly unsharded in the JAX sense.
fn is_effectively_unsharded_dimension(dimension: &ShardingDimension) -> bool {
    matches!(dimension, ShardingDimension::Replicated)
}

/// Returns the non-singleton static dimensions of `shape` together with their original indices.
fn non_singleton_shape_dimensions(shape: &Shape) -> Vec<(usize, usize)> {
    shape
        .dimensions()
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
    let Some(sharding) = input.sharding().cloned() else {
        return Ok(None);
    };
    if input.shape() == target_shape {
        return Ok(Some(sharding));
    }

    let input_non_singleton_dimensions = non_singleton_shape_dimensions(input.shape());
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
            output_dimensions[output_dimension_index] = sharding.dimensions()[input_dimension_index].clone();
            continue;
        }

        if !input_non_singleton_dimensions[input_group_start_index..input_group_end_index]
            .iter()
            .map(|(index, _)| &sharding.dimensions()[*index])
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
        sharding.mesh().clone(),
        output_dimensions,
        sharding.unreduced_axes().clone(),
        sharding.reduced_manual_axes().clone(),
        sharding.varying_manual_axes().clone(),
    )
    .map(|sharding| Some(sharding.without_auto_axes()))
    .map_err(|_| TypeError { message: format!("{op} produced an invalid output sharding") })
}

/// Lifts a reshape's per-lane `input_shape` / `output_shape` pair through one batching level by
/// inserting a new dimension of size `axis_size` at the supplied input position and finding the
/// matching output position.
///
/// The lifted reshape preserves per-lane semantics in row-major order, which requires that the
/// element count to the left of the batch dimension is the same on both sides:
/// `product(input_shape[..k_in]) == product(output_shape[..k_out])`. When such a `k_out` exists,
/// the helper inserts `axis_size` at position `k_in` in the input shape and at position `k_out`
/// in the output shape, and returns `Some((lifted_input_shape, lifted_output_shape, k_out))`. If
/// no matching position can be found (for example, the batch axis falls in the middle of a
/// reshape that mixes dimensions on both sides), the helper returns `None` and the caller should
/// surface a [`BatchingError::UnsupportedOperation`](crate::batching::BatchingError::UnsupportedOperation)
/// pointing at a future fix that emits an explicit transpose before the reshape.
///
/// Dynamic dimensions in `input_shape[..k_in]` or in any candidate `output_shape[..k_out]` are
/// rejected (they make the prefix product undefined).
///
/// # Parameters
///
///   - `input_shape`: Per-lane shape of the reshape's input.
///   - `output_shape`: Per-lane shape produced by [`ReshapeOperation::output_shape`].
///   - `k_in`: Position of the batched axis in the parent-physical input.
///   - `axis_size`: Size of the batched lane this level introduces.
pub fn lift_reshape_shapes(
    input_shape: &Shape,
    output_shape: &Shape,
    k_in: usize,
    axis_size: usize,
) -> Option<(Shape, Shape, usize)> {
    if k_in > input_shape.rank() {
        return None;
    }
    let mut prefix_product = 1usize;
    for dim in &input_shape.dimensions()[..k_in] {
        let value = match dim {
            Size::Static(value) => *value,
            Size::Dynamic(_) => return None,
        };
        prefix_product = prefix_product.checked_mul(value)?;
    }

    let target_prefix_product = prefix_product;
    let mut output_prefix_product = 1usize;
    let mut k_out = None;
    for (index, dim) in output_shape.dimensions().iter().enumerate() {
        if output_prefix_product == target_prefix_product {
            k_out = Some(index);
            break;
        }
        let value = match dim {
            Size::Static(value) => *value,
            Size::Dynamic(_) => return None,
        };
        output_prefix_product = output_prefix_product.checked_mul(value)?;
    }
    if k_out.is_none() && output_prefix_product == target_prefix_product {
        k_out = Some(output_shape.rank());
    }
    let k_out = k_out?;

    let mut lifted_input_dimensions = input_shape.dimensions().to_vec();
    lifted_input_dimensions.insert(k_in, Size::Static(axis_size));
    let mut lifted_output_dimensions = output_shape.dimensions().to_vec();
    lifted_output_dimensions.insert(k_out, Size::Static(axis_size));

    Some((Shape::new(lifted_input_dimensions), Shape::new(lifted_output_dimensions), k_out))
}

/// Computes the abstract output type of one reshape application.
pub fn reshape_abstract(input: &ArrayType, target_shape: &Shape, op: &'static str) -> Result<ArrayType, TypeError> {
    if input.shape() == target_shape {
        return Ok(input.clone());
    }

    let Some(input_elements) = input.element_count().map_err(|error| TypeError { message: error.to_string() })? else {
        return Err(TypeError { message: format!("{op} requires statically known input element counts") });
    };
    let Some(output_elements) =
        target_shape.element_count().map_err(|error| TypeError { message: error.to_string() })?
    else {
        return Err(TypeError { message: format!("{op} requires statically known output element counts") });
    };
    if input_elements != output_elements {
        return Err(TypeError { message: format!("{op} changes the number of elements") });
    }

    ArrayType::new(input.data_type(), target_shape.clone(), None, reshape_array_sharding(input, target_shape, op)?)
        .map_err(|_| TypeError { message: format!("{op} produced an invalid output type") })
}

/// Value-level reshape capability shared by concrete leaves and transform-local wrappers.
///
/// This trait is intentionally fallible because keeping the same Rust type before and after the
/// reshape may rule out some logically valid target shapes for a given leaf representation.
pub trait Reshape: Sized {
    /// Reshapes `self` to `target_shape`.
    ///
    /// Implementors keep the same Rust type before and after the reshape, so some value types can only accept a
    /// subset of logically valid shapes.
    fn reshape(self, target_shape: Shape) -> Result<Self, ProgramError>;
}

/// Convenience trait for values that support reshape.
pub trait ReshapeOps: Reshape {}

impl<T: Reshape> ReshapeOps for T {}

/// Convenience trait for traceable leaves that can serve as the concrete values of a staged reshape.
///
/// This is the trait bound most reshape-aware transforms use when they need both the abstract leaf
/// contract and the value-level reshape operation.
pub trait ReshapeValue: Value<ArrayType> + ReshapeOps {}

impl<T: Value<ArrayType> + ReshapeOps> ReshapeValue for T {}

/// Symbolic-zero-aware tangent reshape. `Zero[input_shape].reshape(target_shape) -> Zero[target_shape]`
/// after validating the reshape via [`reshape_abstract`]; the symbolic-zero variant short-circuits
/// without staging the underlying reshape operation.
impl<V> Reshape for crate::differentiation::Tangent<ArrayType, V>
where
    V: crate::programs::Value<ArrayType> + Reshape,
{
    fn reshape(self, target_shape: Shape) -> Result<Self, ProgramError> {
        match self {
            Self::Zero(r#type) => {
                let output_type = reshape_abstract(&r#type, &target_shape, "reshape")?;
                Ok(Self::Zero(output_type))
            }
            Self::Value(value) => Ok(Self::Value(value.reshape(target_shape)?)),
        }
    }
}

impl<C> Reshape for Tracer<C>
where
    C: StagingContext<Type = ArrayType>,
    C::Operation: SupportsReshape<ArrayType>,
{
    fn reshape(self, target_shape: Shape) -> Result<Self, ProgramError> {
        let input_type = self.r#type().into_owned();
        let output_type = reshape_abstract(&input_type, &target_shape, "reshape")?;
        if input_type == output_type {
            return Ok(self);
        }
        let context = self.context().clone();
        Ok(context
            .stage_operation(C::Operation::reshape_operation(output_type.shape().clone()), &[&self])?
            .into_iter()
            .next()
            .expect("reshape should produce one traced output"))
    }
}

/// Primitive representing one reshape to a target [`Shape`]. The input shape is not part of the
/// operation payload: it is recoverable from the staged input types wherever a rule needs it.
#[derive(Clone, Debug)]
pub struct ReshapeOperation {
    /// Shape produced by the reshape.
    output_shape: Shape,
}

impl ReshapeOperation {
    /// Creates a reshape op producing `output_shape`.
    pub fn new(output_shape: Shape) -> Self {
        Self { output_shape }
    }

    /// Returns the shape produced by this reshape.
    #[inline]
    pub fn output_shape(&self) -> &Shape {
        &self.output_shape
    }
}

impl Display for ReshapeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}{}", self.name(), &self.output_shape)
    }
}

impl Operation<ArrayType> for ReshapeOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "reshape"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![reshape_abstract(&input_types[0], &self.output_shape, "reshape")?])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("output_shape", &self.output_shape))
    }
}

impl<V: ReshapeValue> InterpretableOperation<ArrayType, V> for ReshapeOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].clone().reshape(self.output_shape.clone())?])
    }
}

impl<V, O> TransposableOperation<ArrayType, V, O> for ReshapeOperation
where
    V: ReshapeValue,
    O: Operation<ArrayType> + SupportsReshape<ArrayType>,
{
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
        input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError> {
        check_count!("input", input_types, 1, ProgramError);
        check_count!("output", output_cotangents, 1, ProgramError);
        match &output_cotangents[0] {
            Cotangent::Staged(cotangent) => {
                Ok(vec![Cotangent::Staged(cotangent.clone().reshape(input_types[0].shape().clone())?)])
            }
            Cotangent::Zero => Ok(vec![Cotangent::Zero]),
        }
    }
}

impl<D> DifferentiableOperation<D> for ReshapeOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: ReshapeValue,
    LinearOperationOf<D>: SupportsReshape<ArrayType>,
{
    fn jvp<'jvp>(
        &self,
        _context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, ProgramError);
        let primal = inputs[0].primal().clone().reshape(self.output_shape.clone())?;
        let tangent = inputs[0].tangent().clone().reshape(self.output_shape.clone())?;
        Ok(vec![JvpTracer::new(primal, tangent)])
    }
}

impl<
    V: Value<ArrayType>
        + crate::tracing_v2::operations::broadcast::BroadcastInDim
        + crate::tracing_v2::operations::transpose::Transpose,
    C,
> crate::tracing_v2::batching::BatchableOperation<V, C> for ReshapeOperation
where
    ReshapeOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        _context: &C,
        inputs: &[crate::tracing_v2::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::tracing_v2::batching::ArrayBatch<V>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let (_, input_axes, axis_size) = crate::tracing_v2::batching::batch_input_metadata(inputs)?;
        let Some(k_in) = input_axes[0] else {
            // Lane-uniform: reshape is the same elementwise op (no axis arithmetic needed).
            return crate::tracing_v2::batching::apply_elementwise_batch(self, inputs);
        };
        let input_shape = inputs[0].logical_type()?.shape().clone();
        let Some((_, lifted_output_shape, k_out)) =
            lift_reshape_shapes(&input_shape, &self.output_shape, k_in, axis_size)
        else {
            return Err(crate::batching::BatchingError::UnsupportedOperation {
                message: format!(
                    "missing batching rule for ReshapeOperation with batch axis {k_in} crossing reshape group \
                    boundaries in {input_shape} -> {}",
                    self.output_shape,
                ),
            }
            .into());
        };
        let lifted_op = ReshapeOperation::new(lifted_output_shape);
        crate::tracing_v2::batching::apply_with_axes(&lifted_op, inputs, &[Some(k_out)])
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
            Err(TypeError { message: ("reshape changes the number of elements").into() })
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
            Err(TypeError { message: ("reshape cannot preserve sharding across the requested reshape").into() })
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
            Err(TypeError { message: ("reshape cannot preserve sharding across the requested reshape").into() })
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
