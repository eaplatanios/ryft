use std::fmt::Display;

use crate::contexts::StagingContext;
use crate::differentiation::Tangent;
use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::programs::{ProgramError, Value};
use crate::sharding::{Sharding, ShardingDimension};
use crate::tracing::Tracer;
use crate::types::{ArrayType, Shape, Size, Type, TypeError, Typed};

/// Canonical operation name for [`ReshapeOperation`].
pub const RESHAPE_OPERATION_NAME: &'static str = "reshape";

/// [`Operation`] that reshapes its input array to a target [`Shape`]. The input shape is not part of the operation
/// payload; it is recoverable from the staged input types wherever a rule needs it. Refer to the documentation of
/// [`Reshape`] for more information.
#[derive(Clone, Debug)]
pub struct ReshapeOperation {
    /// Output [`Shape`] of this [`ReshapeOperation`].
    shape: Shape,
}

impl ReshapeOperation {
    /// Creates a new [`ReshapeOperation`] with the provided output [`Shape`].
    #[inline]
    pub fn new(shape: Shape) -> Self {
        Self { shape }
    }

    /// Returns the output shape of this [`ReshapeOperation`].
    #[inline]
    pub fn output_shape(&self) -> &Shape {
        &self.shape
    }
}

impl Display for ReshapeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for ReshapeOperation {
    #[inline]
    fn name(&self) -> &'static str {
        RESHAPE_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        match input_types[0].reshape(self.shape.clone()) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError { message: error.to_string() }),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("shape", &self.shape))
    }
}

impl<V: Value<ArrayType> + Reshape<Output = V>> InterpretableOperation<ArrayType, V> for ReshapeOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].clone().reshape(self.shape.clone())?])
    }
}

/// Trait that represents [`Operation`] types that support/include [`ReshapeOperation`]. Backend-owned closed
/// [`Operation`] types implement this trait so that generic transform code can stage [`ReshapeOperation`]s without
/// knowing which operation type is in use.
pub trait SupportsReshape<T: Type> {
    /// Constructs an instance of [`ReshapeOperation`] for this [`Operation`] type with the provided output shape.
    fn reshape_operation(output_shape: Shape) -> Self;
}

/// Represents the ability to reshape an array to a target [`Shape`] without changing its element count or its layout.
/// This is the direct analogue of JAX's
/// [`jnp.reshape`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.reshape.html).
///
/// `t.reshape(target_shape)` reinterprets `t`'s payload under the specified target [`Shape`]. The input and target
/// shapes must have equal element counts. When the input carries a [`Sharding`], it is propagated using singleton
/// stripping and contiguous split/merge grouping: dimensions that map one-to-one keep their sharding, while dimensions
/// that split or merge must be replicated.
///
/// # Example
///
/// The following example shows how to use [`Reshape`] in practice:
///
/// ```rust
/// # use ryft_core::operations::manipulation::Reshape;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::tests::{TestArray as Array};
/// # use ryft_core::types::{Shape, Size};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Reshape a length-6 vector to a `[2, 3]` matrix while keeping the row-major payload unchanged.
/// let x = Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let y = x.reshape(Shape::new(vec![Size::Static(2), Size::Static(3)]))?;
/// assert_eq!(y.values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// # Ok(())
/// # }
/// ```
pub trait Reshape {
    /// Output type of the reshape operation.
    type Output;

    /// Reshapes `self` to `shape`. Refer to the documentation of this trait for more information on what this
    /// operation does.
    fn reshape(self, shape: Shape) -> Result<Self::Output, ProgramError>;
}

impl Reshape for &ArrayType {
    type Output = ArrayType;

    fn reshape(self, shape: Shape) -> Result<ArrayType, ProgramError> {
        if *self.shape() == shape {
            return Ok(self.clone());
        }

        let Some(input_elements) = self.element_count().map_err(|error| TypeError { message: error.to_string() })?
        else {
            return Err(
                TypeError { message: "reshape requires statically known input element counts".to_string() }.into()
            );
        };
        let Some(output_elements) = shape.element_count().map_err(|error| TypeError { message: error.to_string() })?
        else {
            return Err(
                TypeError { message: "reshape requires statically known output element counts".to_string() }.into()
            );
        };
        if input_elements != output_elements {
            return Err(TypeError { message: "reshape changes the number of elements".to_string() }.into());
        }

        // Propagate the input sharding (when present) to the target shape using JAX-style singleton stripping and
        // contiguous split/merge grouping.
        let sharding = if let Some(sharding) = self.sharding() {
            let alignment_error =
                || TypeError { message: "reshape could not align static reshape dimension groups".to_string() };

            // Strip singleton and dynamic dimensions on both sides. Only the remaining static dimensions take part
            // in the split/merge analysis, so shardings move freely across inserted or removed size-1 axes.
            let input_dimensions = self
                .shape()
                .dimensions()
                .iter()
                .enumerate()
                .filter_map(|(index, size)| match size {
                    Size::Static(1) => None,
                    Size::Static(value) => Some((index, *value)),
                    Size::Dynamic(_) => None,
                })
                .collect::<Vec<_>>();
            let output_dimensions = shape
                .dimensions()
                .iter()
                .enumerate()
                .filter_map(|(index, size)| match size {
                    Size::Static(1) => None,
                    Size::Static(value) => Some((index, *value)),
                    Size::Dynamic(_) => None,
                })
                .collect::<Vec<_>>();

            // Partition the two stripped shapes into corresponding contiguous groups with matching element counts.
            // Each group pairs the input dimensions that the reshape merges or splits into the paired output
            // dimensions. Starting a group with one dimension from each side, the side with the smaller running element
            // product absorbs its next dimension until the two products match. When one side runs out of dimensions
            // (or a product overflows) before the products match, the reshape mixes dimensions in a way that sharding
            // propagation cannot describe.
            let mut input_start_index = 0usize;
            let mut output_start_index = 0usize;
            let mut groups = Vec::new();
            while input_start_index < input_dimensions.len() || output_start_index < output_dimensions.len() {
                if input_start_index == input_dimensions.len() || output_start_index == output_dimensions.len() {
                    return Err(alignment_error().into());
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
                            return Err(alignment_error().into());
                        }
                        input_group_product = input_group_product
                            .checked_mul(input_dimensions[input_start_index].1)
                            .ok_or_else(alignment_error)?;
                        input_start_index += 1;
                    } else {
                        if output_start_index == output_dimensions.len() {
                            return Err(alignment_error().into());
                        }
                        output_group_product = output_group_product
                            .checked_mul(output_dimensions[output_start_index].1)
                            .ok_or_else(alignment_error)?;
                        output_start_index += 1;
                    }
                }
                groups.push((input_group_start_index, input_start_index, output_group_start_index, output_start_index));
            }

            // Distribute the input dimension shardings over the target dimensions. Output dimensions start out
            // replicated, which already covers the singleton axes stripped above and every split/merge group.
            // One-to-one groups then carry their input dimension's sharding over to the paired output dimension.
            // Dimensions that take part in an actual split or merge must be replicated on the input side, because
            // the reshape redistributes their elements across mesh shards.
            let mut output_sharding_dimensions =
                std::iter::repeat_n(ShardingDimension::replicated(), shape.rank()).collect::<Vec<_>>();
            for (input_group_start_index, input_group_end_index, output_group_start_index, output_group_end_index) in
                groups
            {
                let input_group_length = input_group_end_index - input_group_start_index;
                let output_group_length = output_group_end_index - output_group_start_index;
                if input_group_length == 1 && output_group_length == 1 {
                    let input_dimension_index = input_dimensions[input_group_start_index].0;
                    let output_dimension_index = output_dimensions[output_group_start_index].0;
                    output_sharding_dimensions[output_dimension_index] =
                        sharding.dimensions()[input_dimension_index].clone();
                    continue;
                }
                if !input_dimensions[input_group_start_index..input_group_end_index]
                    .iter()
                    .all(|(index, _)| matches!(sharding.dimensions()[*index], ShardingDimension::Replicated))
                {
                    return Err(TypeError {
                        message: "reshape cannot preserve sharding across the requested reshape".to_string(),
                    }
                    .into());
                }
            }

            // Rebuild the sharding over the target rank, preserving the mesh and the unreduced/manual axis metadata
            // that is unaffected by the dimension regrouping.
            Some(
                Sharding::with_manual_axes(
                    sharding.mesh().clone(),
                    output_sharding_dimensions,
                    sharding.unreduced_axes().clone(),
                    sharding.reduced_manual_axes().clone(),
                    sharding.varying_manual_axes().clone(),
                )
                .map(|sharding| sharding.without_auto_axes())
                .map_err(|_| TypeError { message: "reshape produced an invalid output sharding".to_string() })?,
            )
        } else {
            None
        };

        ArrayType::new(self.data_type(), shape)
            .with_sharding(sharding)
            .map_err(|_| TypeError { message: "reshape produced an invalid output type".to_string() }.into())
    }
}

impl<C: StagingContext<Type = ArrayType, Operation: SupportsReshape<ArrayType>>> Reshape for Tracer<C> {
    type Output = Self;

    fn reshape(self, shape: Shape) -> Result<Self, ProgramError> {
        let input_type = self.r#type().into_owned();
        let output_type = input_type.reshape(shape)?;
        if input_type == output_type {
            return Ok(self);
        }
        let mut outputs = self
            .context()
            .stage_operation(C::Operation::reshape_operation(output_type.shape().clone()), &[&self])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<V: Value<ArrayType> + Reshape<Output = V>> Reshape for Tangent<ArrayType, V> {
    type Output = Self;

    fn reshape(self, shape: Shape) -> Result<Self, ProgramError> {
        match self {
            Self::Zero(r#type) => Ok(Self::Zero(r#type.reshape(shape)?)),
            Self::Value(value) => Ok(Self::Value(value.reshape(shape)?)),
        }
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding};
    use crate::tests::TestArray;
    use crate::types::DataType;

    use super::*;

    #[test]
    fn test_reshape() {
        let shape = Shape::new(vec![Size::Static(2), Size::Static(3)]);
        let operation = ReshapeOperation::new(shape.clone());

        // Operation identity and accessors.
        assert_eq!(operation.name(), RESHAPE_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "reshape [shape=[2, 3]]");
        assert_eq!(*operation.output_shape(), shape);

        // Type inference validates the element count and returns the target shape.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(6)]));
        let output_type = ArrayType::new(DataType::F64, shape.clone());
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&input_type)), Ok(vec![output_type.clone()]));

        // Type-level (abstract) reshaping validates the target shape and returns the output type without consuming
        // the borrowed input type.
        assert_eq!(input_type.reshape(shape.clone()), Ok(output_type.clone()));

        // Interpretation reinterprets the row-major payload under the target shape.
        let input = TestArray::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let output = operation.interpret(std::slice::from_ref(&input)).unwrap();
        assert_eq!(*output[0].array_type(), output_type);
        assert_eq!(output[0].values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Invalid inputs report precise operation and interpreter errors.
        assert_eq!(
            operation.infer_output_types(&[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(5)]))]),
            Err(TypeError { message: "reshape changes the number of elements".to_string() }),
        );
        assert_eq!(
            InterpretableOperation::<ArrayType, TestArray>::interpret(&operation, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // Program rendering uses the canonical operation name and includes the captured output shape.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, ReshapeOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_output = builder.add_instruction(operation, vec![program_input]).unwrap()[0];
        let program = builder.build::<TestArray, TestArray>(vec![program_output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[6] .
                let %1:f64[2, 3] = reshape [shape=[2, 3]] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_reshape_preserves_sharding_across_inserted_singleton_axes() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert_eq!(
            input_type.reshape(Shape::new(vec![Size::Static(1), Size::Static(8), Size::Static(1)])),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(1), Size::Static(8), Size::Static(1)]))
                .with_sharding(
                    Sharding::new(
                        mesh,
                        vec![
                            ShardingDimension::replicated(),
                            ShardingDimension::sharded(["x"]),
                            ShardingDimension::replicated(),
                        ],
                    )
                    .unwrap(),
                )
                .unwrap())
        );
    }

    #[test]
    fn test_reshape_merges_replicated_axes_and_preserves_unchanged_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8), Size::Static(2), Size::Static(3)]))
                .with_sharding(
                    Sharding::new(
                        mesh.clone(),
                        vec![
                            ShardingDimension::sharded(["x"]),
                            ShardingDimension::replicated(),
                            ShardingDimension::replicated(),
                        ],
                    )
                    .unwrap(),
                )
                .unwrap();
        assert_eq!(
            input_type.reshape(Shape::new(vec![Size::Static(8), Size::Static(6)])),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8), Size::Static(6)]))
                .with_sharding(
                    Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                        .unwrap(),
                )
                .unwrap())
        );
    }

    #[test]
    fn test_reshape_splits_replicated_axis_and_preserves_unchanged_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8), Size::Static(6)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            input_type.reshape(Shape::new(vec![Size::Static(8), Size::Static(2), Size::Static(3)])),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8), Size::Static(2), Size::Static(3)]))
                .with_sharding(
                    Sharding::new(
                        mesh,
                        vec![
                            ShardingDimension::sharded(["x"]),
                            ShardingDimension::replicated(),
                            ShardingDimension::replicated(),
                        ],
                    )
                    .unwrap(),
                )
                .unwrap())
        );
    }

    #[test]
    fn test_reshape_rejects_partitioned_split() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert_eq!(
            input_type.reshape(Shape::new(vec![Size::Static(2), Size::Static(4)])),
            Err(ProgramError::Type(TypeError {
                message: "reshape cannot preserve sharding across the requested reshape".to_string(),
            })),
        );
    }

    #[test]
    fn test_reshape_rejects_partitioned_merge() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(4)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])]).unwrap(),
            )
            .unwrap();
        assert_eq!(
            input_type.reshape(Shape::new(vec![Size::Static(8)])),
            Err(ProgramError::Type(TypeError {
                message: "reshape cannot preserve sharding across the requested reshape".to_string(),
            })),
        );
    }

    #[test]
    fn test_reshape_allows_unsharded_many_to_many_group() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(6)]))
            .with_sharding(
                Sharding::with_manual_axes(
                    mesh.clone(),
                    vec![ShardingDimension::replicated(), ShardingDimension::replicated()],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["x"],
                )
                .unwrap(),
            )
            .unwrap();

        assert_eq!(
            input_type.reshape(Shape::new(vec![Size::Static(3), Size::Static(4)])),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3), Size::Static(4)]))
                .with_sharding(
                    Sharding::with_manual_axes(
                        mesh,
                        vec![ShardingDimension::replicated(), ShardingDimension::replicated()],
                        Vec::<&str>::new(),
                        Vec::<&str>::new(),
                        ["x"],
                    )
                    .unwrap(),
                )
                .unwrap())
        );
    }
}
