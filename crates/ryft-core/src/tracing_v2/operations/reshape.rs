use std::fmt::{Debug, Display};

use half::{bf16, f16};

#[cfg(test)]
use indoc::indoc;

use crate::{
    macros::check_input_count,
    sharding::{Sharding, ShardingDimension},
    tracing::{AtomId, OperationFormatter, Traceable, TracingError},
    tracing_v2::{
        DifferentiableEngine, LinearPrimitiveOperation,
        engines::StagingEngine,
        forward::{Differentiable, JvpContext, JvpTracer},
        jit::Tracer,
    },
    types::{ArrayType, Shape, Size, Type, TypeError, Typed},
};

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
    E: StagingEngine<Type = ArrayType, Value = V> + ?Sized,
    E::Operation: SupportsReshape<ArrayType, V>,
{
    fn reshape(self, target_shape: Shape) -> Result<Self, TracingError> {
        let input_type = self.r#type().into_owned();
        let output_type = reshape_abstract(&input_type, &target_shape, "reshape")?;
        if input_type == output_type {
            return Ok(self);
        }
        let engine = self.engine.clone();
        Ok(engine
            .apply_staged_op(
                std::slice::from_ref(&self),
                E::Operation::reshape_operation(input_type.shape.clone(), output_type.shape.clone()),
            )?
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

#[cfg(any(feature = "ndarray", test))]
mod ndarray_support {
    use ndarray::Array2;

    use super::{ReshapeOps, reshape_abstract};
    use crate::{
        tracing::TracingError,
        types::{Shape, Size, TypeError, Typed},
    };

    impl ReshapeOps for Array2<f32> {
        fn reshape(self, target_shape: Shape) -> Result<Self, TracingError> {
            let input_type = self.r#type().into_owned();
            let output_type = reshape_abstract(&input_type, &target_shape, "reshape")?;
            if input_type == output_type {
                return Ok(self);
            }
            let [Size::Static(rows), Size::Static(cols)] = output_type.shape.dimensions.as_slice() else {
                return Err(TracingError::Type(TypeError {
                    message: "reshape expected a rank-2 static ndarray target shape".to_string(),
                }));
            };
            let values = self.iter().copied().collect::<Vec<_>>();
            Array2::from_shape_vec((*rows, *cols), values).map_err(|_| {
                TracingError::Type(TypeError {
                    message: "reshape could not realize the requested ndarray target shape".to_string(),
                })
            })
        }
    }

    impl ReshapeOps for Array2<f64> {
        fn reshape(self, target_shape: Shape) -> Result<Self, TracingError> {
            let input_type = self.r#type().into_owned();
            let output_type = reshape_abstract(&input_type, &target_shape, "reshape")?;
            if input_type == output_type {
                return Ok(self);
            }
            let [Size::Static(rows), Size::Static(cols)] = output_type.shape.dimensions.as_slice() else {
                return Err(TracingError::Type(TypeError {
                    message: "reshape expected a rank-2 static ndarray target shape".to_string(),
                }));
            };
            let values = self.iter().copied().collect::<Vec<_>>();
            Array2::from_shape_vec((*rows, *cols), values).map_err(|_| {
                TracingError::Type(TypeError {
                    message: "reshape could not realize the requested ndarray target shape".to_string(),
                })
            })
        }
    }
}

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
        check_input_count!(inputs, 1);
        Ok(vec![inputs[0].clone().reshape(self.output_shape().clone())?])
    }
}

impl<V: ReshapeValue> LinearOperation<ArrayType, V> for ReshapeOperation {
    fn transpose(
        &self,
        context: &mut crate::tracing_v2::operations::TranspositionContext<
            '_,
            ArrayType,
            V,
            LinearPrimitiveOperation<V>,
        >,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        check_input_count!(output_cotangents, 1);
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
                    LinearPrimitiveOperation::Reshape {
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
        _engine: &E,
        context: &mut JvpContext<'_, E::Value, E::LinearOperation>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_input_count!(inputs, 1);
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
    use std::{cell::RefCell, rc::Rc};

    use ndarray::arr2;
    use pretty_assertions::assert_eq;

    use crate::{
        parameters::Placeholder,
        sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding},
        tracing::{Program, ProgramBuilder},
        tracing_v2::{
            LinearPrimitiveOperation, PrimitiveOperation, engines::StagingEngine, operations::TranspositionContext,
            operations::matrix::ndarray_support::Array2Engine,
        },
        types::{DataType, Shape},
    };

    use super::*;

    fn test_array_transposition_context(
        builder: Rc<
            RefCell<ProgramBuilder<ArrayType, ndarray::Array2<f64>, LinearPrimitiveOperation<ndarray::Array2<f64>>>>,
        >,
    ) -> TranspositionContext<'static, ArrayType, ndarray::Array2<f64>, LinearPrimitiveOperation<ndarray::Array2<f64>>>
    {
        TranspositionContext::new(builder)
    }

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

    #[test]
    fn test_reshape_eval_reorders_only_shape_metadata() {
        let input = arr2(&[[1.0f64, 2.0], [3.0, 4.0]]);

        assert_eq!(
            input.reshape(Shape::new(vec![Size::Static(1), Size::Static(4)])).unwrap(),
            arr2(&[[1.0f64, 2.0, 3.0, 4.0]])
        );
    }

    #[test]
    fn test_reshape_jit_rendering_includes_target_shape() {
        let input = arr2(&[[1.0f64, 2.0], [3.0, 4.0]]);
        let engine = Array2Engine::<f64>::new();
        let (_, compiled): (
            ndarray::Array2<f64>,
            Program<
                ArrayType,
                ndarray::Array2<f64>,
                PrimitiveOperation<ndarray::Array2<f64>>,
                ndarray::Array2<f64>,
                ndarray::Array2<f64>,
            >,
        ) = engine
            .interpret_and_trace(|x| x.reshape(Shape::new(vec![Size::Static(1), Size::Static(4)])), input)
            .unwrap();

        assert_eq!(
            compiled.to_string(),
            indoc! {"
                lambda %0:f64[2, 2] .
                let %1:f64[1, 4] = reshape [input_shape=[2, 2], output_shape=[1, 4]] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_reshape_transpose_restores_the_input_shape() {
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2)]), None, None).unwrap();
        let output_value = arr2(&[[1.0f64, 2.0, 3.0, 4.0]]);
        let transpose_builder = Rc::new(RefCell::new(ProgramBuilder::<
            ArrayType,
            ndarray::Array2<f64>,
            LinearPrimitiveOperation<ndarray::Array2<f64>>,
        >::new()));
        let output_cotangent_atom = transpose_builder.borrow_mut().add_input(output_value.r#type().into_owned());
        let mut context = test_array_transposition_context(transpose_builder.clone());
        let contribution_atom =
            ReshapeOperation::new(input_type.shape.clone(), Shape::new(vec![Size::Static(1), Size::Static(4)]))
                .transpose(&mut context, &[Some(output_cotangent_atom)])
                .unwrap()
                .into_iter()
                .next()
                .expect("transpose should return one contribution")
                .expect("transpose should produce one cotangent contribution");
        drop(context);

        let transpose_builder = Rc::try_unwrap(transpose_builder)
            .expect("transpose builder should not have outstanding linear terms")
            .into_inner();
        let transpose_program = transpose_builder
            .build::<ndarray::Array2<f64>, ndarray::Array2<f64>>(vec![contribution_atom], Placeholder, Placeholder)
            .unwrap();
        assert_eq!(
            transpose_program.interpret(arr2(&[[1.0f64, 2.0, 3.0, 4.0]])).unwrap(),
            arr2(&[[1.0f64, 2.0], [3.0, 4.0]])
        );
    }
}
