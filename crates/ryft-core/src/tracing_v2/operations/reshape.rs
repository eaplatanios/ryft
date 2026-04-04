//! Reshape primitive for [`crate::tracing_v2`].
//!
//! This module adds one linear unary primitive that changes only tensor shape metadata while preserving element order
//! and element count. The public trait surface is deliberately fallible because not every traceable leaf type can
//! represent every logical target shape with the same Rust type.

use std::{
    any::Any,
    fmt::{Debug, Display},
    sync::Arc,
};

#[cfg(test)]
use indoc::indoc;
#[cfg(feature = "xla")]
use ryft_mlir::dialects::stable_hlo;
#[cfg(feature = "xla")]
use ryft_mlir::{Block, Operation, Value, ValueRef};

#[cfg(feature = "xla")]
use crate::xla::{
    lowering::{LoweringError, MlirLowerableValue, PlainMlirLowerer, PlainMlirLoweringMode, ShardMapMlirLowerer},
    shard_map::{ShardMapTensor, trace},
};
use crate::{
    sharding::{Sharding, ShardingDimension},
    tracing_v2::{
        FloatExt, MatrixOps, TraceError, TraceValue, TransformLeaf, ZeroLike,
        batch::Batch,
        forward::{JvpTracer, TangentSpace},
        graph::AtomId,
        jit::JitTracer,
        linear::LinearTerm,
        ops::{BatchOp, Op},
        program::ProgramBuilder,
    },
    types::{ArrayType, Shape, Size, Typed},
};

use super::expect_input_count;

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
) -> Result<Option<Sharding>, TraceError> {
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
        return Err(TraceError::InternalInvariantViolation(
            "static reshape group alignment should succeed after element-count validation",
        ));
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
            return Err(TraceError::IncompatibleAbstractValues { op });
        }

        for (output_dimension_index, _) in
            output_non_singleton_dimensions[output_group_start_index..output_group_end_index].iter()
        {
            output_dimensions[*output_dimension_index] = ShardingDimension::replicated();
        }
    }

    Sharding::new(
        sharding.mesh.clone(),
        output_dimensions,
        sharding.unreduced_axes.clone(),
        sharding.reduced_manual_axes.clone(),
        sharding.varying_manual_axes.clone(),
    )
    .map(|sharding| Some(sharding.without_auto_axes()))
    .map_err(|_| TraceError::InternalInvariantViolation("reshape output sharding should match the target rank"))
}

/// Computes the abstract output type of one reshape application.
pub(crate) fn reshape_abstract(
    input: &ArrayType,
    target_shape: &Shape,
    op: &'static str,
) -> Result<ArrayType, TraceError> {
    if input.shape == *target_shape {
        return Ok(input.clone());
    }

    let Some(input_elements) = static_shape_element_count(&input.shape) else {
        return Err(TraceError::IncompatibleAbstractValues { op });
    };
    let Some(output_elements) = static_shape_element_count(target_shape) else {
        return Err(TraceError::IncompatibleAbstractValues { op });
    };
    if input_elements != output_elements {
        return Err(TraceError::IncompatibleAbstractValues { op });
    }

    ArrayType::new(input.data_type, target_shape.clone(), None, reshape_array_sharding(input, target_shape, op)?)
        .map_err(|_| TraceError::InternalInvariantViolation("reshape output sharding should match the target rank"))
}

/// Value-level reshape capability shared by concrete leaves and transform-local wrappers.
pub trait ReshapeOps: Sized {
    /// Reshapes `self` to `target_shape`.
    ///
    /// Implementors keep the same Rust type before and after the reshape, so some value types can only accept a
    /// subset of logically valid shapes.
    fn reshape(self, target_shape: Shape) -> Result<Self, TraceError>;
}

/// Convenience trait for traceable leaves that can serve as the concrete values of a staged reshape.
pub trait ReshapeValue: TraceValue + ReshapeOps {}

impl<T: TraceValue + ReshapeOps> ReshapeValue for T {}

/// Tangent-space reshape capability used by [`JvpTracer`].
pub trait ReshapeTangentSpace<V: ReshapeValue>: TangentSpace<V> {
    /// Reshapes one tangent value from `input_type` to `output_type`.
    fn reshape(input_type: &ArrayType, output_type: &ArrayType, tangent: Self) -> Result<Self, TraceError>;
}

impl<V: ReshapeValue + FloatExt + ZeroLike> ReshapeTangentSpace<V> for V {
    fn reshape(_input_type: &ArrayType, output_type: &ArrayType, tangent: Self) -> Result<Self, TraceError> {
        tangent.reshape(output_type.shape.clone())
    }
}

impl<V: ReshapeValue + FloatExt + ZeroLike + MatrixOps> ReshapeTangentSpace<V> for LinearTerm<V> {
    fn reshape(input_type: &ArrayType, output_type: &ArrayType, tangent: Self) -> Result<Self, TraceError> {
        if input_type == output_type {
            return Ok(tangent);
        }
        Ok(LinearTerm::apply_staged_op(
            std::slice::from_ref(&tangent),
            Arc::new(ReshapeOp::new(input_type.clone(), output_type.clone())),
            1,
        )?
        .into_iter()
        .next()
        .expect("reshape should produce one tangent output"))
    }
}

impl<V: ReshapeValue, T: ReshapeTangentSpace<V>> ReshapeOps for JvpTracer<V, T> {
    fn reshape(self, target_shape: Shape) -> Result<Self, TraceError> {
        let input_type = self.primal.tpe();
        let output_type = reshape_abstract(&input_type, &target_shape, "reshape")?;
        if input_type == output_type {
            return Ok(self);
        }
        let tangent = T::reshape(&input_type, &output_type, self.tangent)?;
        Ok(Self { primal: self.primal.reshape(target_shape)?, tangent })
    }
}

impl<V: TransformLeaf + ReshapeValue> ReshapeOps for JitTracer<V> {
    fn reshape(self, target_shape: Shape) -> Result<Self, TraceError> {
        let input_type = self.tpe();
        let output_type = reshape_abstract(&input_type, &target_shape, "reshape")?;
        if input_type == output_type {
            return Ok(self);
        }
        let output_value = self.value.clone().reshape(target_shape)?;
        Ok(JitTracer::apply_staged_op(
            std::slice::from_ref(&self),
            Arc::new(ReshapeOp::new(input_type, output_type)),
            vec![output_value],
        )?
        .into_iter()
        .next()
        .expect("reshape should produce one traced output"))
    }
}

impl<V: ReshapeValue> ReshapeOps for Batch<V> {
    fn reshape(self, target_shape: Shape) -> Result<Self, TraceError> {
        if self.lanes().iter().all(|lane| lane.tpe().shape == target_shape) {
            return Ok(self);
        }
        Ok(Self::new(
            self.into_lanes()
                .into_iter()
                .map(|lane| lane.reshape(target_shape.clone()))
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}

impl ReshapeOps for f32 {
    fn reshape(self, target_shape: Shape) -> Result<Self, TraceError> {
        reshape_abstract(&self.tpe(), &target_shape, "reshape")?;
        Ok(self)
    }
}

impl ReshapeOps for f64 {
    fn reshape(self, target_shape: Shape) -> Result<Self, TraceError> {
        reshape_abstract(&self.tpe(), &target_shape, "reshape")?;
        Ok(self)
    }
}

#[cfg(any(feature = "ndarray", test))]
mod ndarray_support {
    use ndarray::Array2;

    use super::{ReshapeOps, reshape_abstract};
    use crate::{
        tracing_v2::TraceError,
        types::{Shape, Size, Typed},
    };

    impl ReshapeOps for Array2<f32> {
        fn reshape(self, target_shape: Shape) -> Result<Self, TraceError> {
            let input_type = self.tpe();
            let output_type = reshape_abstract(&input_type, &target_shape, "reshape")?;
            if input_type == output_type {
                return Ok(self);
            }
            let [Size::Static(rows), Size::Static(cols)] = output_type.shape.dimensions.as_slice() else {
                return Err(TraceError::IncompatibleAbstractValues { op: "reshape" });
            };
            let values = self.iter().copied().collect::<Vec<_>>();
            Array2::from_shape_vec((*rows, *cols), values)
                .map_err(|_| TraceError::IncompatibleAbstractValues { op: "reshape" })
        }
    }

    impl ReshapeOps for Array2<f64> {
        fn reshape(self, target_shape: Shape) -> Result<Self, TraceError> {
            let input_type = self.tpe();
            let output_type = reshape_abstract(&input_type, &target_shape, "reshape")?;
            if input_type == output_type {
                return Ok(self);
            }
            let [Size::Static(rows), Size::Static(cols)] = output_type.shape.dimensions.as_slice() else {
                return Err(TraceError::IncompatibleAbstractValues { op: "reshape" });
            };
            let values = self.iter().copied().collect::<Vec<_>>();
            Array2::from_shape_vec((*rows, *cols), values)
                .map_err(|_| TraceError::IncompatibleAbstractValues { op: "reshape" })
        }
    }
}

#[cfg(feature = "xla")]
impl ReshapeOps for ShardMapTensor {
    fn reshape(self, target_shape: Shape) -> Result<Self, TraceError> {
        Ok(Self::new(reshape_abstract(self.r#type(), &target_shape, "reshape")?))
    }
}

/// Primitive representing one reshape from `input_type` to `output_type`.
#[derive(Clone)]
pub(crate) struct ReshapeOp {
    /// Abstract type expected for the operand.
    input_type: ArrayType,
    /// Abstract type produced by the reshape.
    output_type: ArrayType,
}

impl ReshapeOp {
    /// Creates a reshape op with explicit input and output abstract types.
    pub(crate) fn new(input_type: ArrayType, output_type: ArrayType) -> Self {
        Self { input_type, output_type }
    }

    /// Returns the captured input abstract type.
    pub(crate) fn input_type(&self) -> &ArrayType {
        &self.input_type
    }

    /// Returns the captured output abstract type.
    pub(crate) fn output_type(&self) -> &ArrayType {
        &self.output_type
    }
}

impl Debug for ReshapeOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Reshape")
    }
}

impl Display for ReshapeOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "reshape{}", self.output_type().shape)
    }
}

impl<V: ReshapeValue> Op<V> for ReshapeOp {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "reshape"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        if inputs[0] != *self.input_type() {
            return Err(TraceError::IncompatibleAbstractValues { op: "reshape" });
        }
        Ok(vec![self.output_type().clone()])
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().reshape(self.output_type().shape.clone())?])
    }

    fn replay_linearized_jit(
        &self,
        inputs: Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>,
    ) -> Result<Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>, TraceError>
    where
        V: TransformLeaf,
    {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![
            inputs
                .into_iter()
                .next()
                .expect("validated reshape input should exist")
                .reshape(self.output_type().shape.clone())?,
        ])
    }

    fn apply_program_jvp_rule(
        &self,
        inputs: &[JvpTracer<V, LinearTerm<V>>],
    ) -> Result<Vec<JvpTracer<V, LinearTerm<V>>>, TraceError>
    where
        V: FloatExt + ZeroLike + MatrixOps,
    {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().reshape(self.output_type().shape.clone())?])
    }

    fn transpose_program_op(
        &self,
        builder: &mut ProgramBuilder<V>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError>
    where
        V: FloatExt + ZeroLike + MatrixOps,
    {
        expect_input_count(inputs.len(), 1)?;
        expect_input_count(outputs.len(), 1)?;
        expect_input_count(output_cotangents.len(), 1)?;
        if self.input_type() == self.output_type() {
            return Ok(vec![Some(output_cotangents[0])]);
        }
        let contribution = builder.add_equation(
            Arc::new(Self::new(self.output_type().clone(), self.input_type().clone())),
            vec![output_cotangents[0]],
        )?[0];
        Ok(vec![Some(contribution)])
    }

    #[cfg(feature = "xla")]
    fn lower_plain_mlir<'b, 'c, 't>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        V: MlirLowerableValue,
    {
        let target_shape = self
            .output_type()
            .shape
            .dimensions
            .iter()
            .map(|size| match size {
                Size::Static(value) => Ok(*value),
                Size::Dynamic(_) => Err(LoweringError::UnsupportedOp { op: <Self as Op<V>>::name(self).to_string() }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let operation = lowerer.block.append_operation(stable_hlo::reshape(
            input_values[0],
            target_shape.as_slice(),
            lowerer.location,
        ));
        Ok(vec![operation.result(0).expect("stablehlo.reshape should return one result").as_ref()])
    }

    #[cfg(feature = "xla")]
    fn lower_shard_map_mlir<'b, 'c, 't>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        V: MlirLowerableValue,
    {
        let target_shape = self
            .output_type()
            .shape
            .dimensions
            .iter()
            .map(|size| match size {
                Size::Static(value) => Ok(*value),
                Size::Dynamic(_) => Err(LoweringError::UnsupportedOp { op: <Self as Op<V>>::name(self).to_string() }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let operation = lowerer.block.append_operation(stable_hlo::reshape(
            input_values[0],
            target_shape.as_slice(),
            lowerer.location,
        ));
        Ok(vec![operation.result(0).expect("stablehlo.reshape should return one result").as_ref()])
    }
}

impl<V: ReshapeValue> BatchOp<V> for ReshapeOp {
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().reshape(self.output_type().shape.clone())?])
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;
    use pretty_assertions::assert_eq;

    #[cfg(feature = "xla")]
    use crate::xla::shard_map::{
        ShardMapTracer, TracedShardMap, TracedXlaProgram, shard_map, with_sharding_constraint,
    };
    use crate::{
        parameters::Placeholder,
        sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding},
        tracing_v2::{CompiledFunction, JitTracer, ProgramBuilder, jit::try_jit},
        types::{DataType, Shape},
    };

    use super::*;

    /// Creates one small manual mesh used by reshape sharding tests.
    fn test_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap()
    }

    #[cfg(feature = "xla")]
    fn test_manual_mesh(axis_size: usize) -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::new("x", axis_size, MeshAxisType::Manual).unwrap()]).unwrap()
    }

    #[cfg(feature = "xla")]
    fn test_sharding(
        mesh: &LogicalMesh,
        dimensions: Vec<ShardingDimension>,
        varying_manual_axes: Vec<&str>,
    ) -> Sharding {
        Sharding::new(mesh.clone(), dimensions, Vec::<&str>::new(), Vec::<&str>::new(), varying_manual_axes).unwrap()
    }

    #[test]
    fn test_reshape_abstract_preserves_sharding_across_inserted_singleton_axes() {
        let mesh = test_mesh();
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(8)]),
            None,
            Some(
                Sharding::new(
                    mesh.clone(),
                    vec![ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                )
                .unwrap(),
            ),
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
                        Vec::<&str>::new(),
                        Vec::<&str>::new(),
                        Vec::<&str>::new(),
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
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
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
                    Sharding::new(
                        mesh,
                        vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
                        Vec::<&str>::new(),
                        Vec::<&str>::new(),
                        Vec::<&str>::new(),
                    )
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
                Sharding::new(
                    mesh.clone(),
                    vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                )
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
                        Vec::<&str>::new(),
                        Vec::<&str>::new(),
                        Vec::<&str>::new(),
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
            Err(TraceError::IncompatibleAbstractValues { op: "reshape" })
        );
    }

    #[test]
    fn test_reshape_abstract_rejects_partitioned_split() {
        let mesh = test_mesh();
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(8)]),
            None,
            Some(
                Sharding::new(
                    mesh,
                    vec![ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                )
                .unwrap(),
            ),
        )
        .unwrap();

        assert_eq!(
            reshape_abstract(&input_type, &Shape::new(vec![Size::Static(2), Size::Static(4)]), "reshape"),
            Err(TraceError::IncompatibleAbstractValues { op: "reshape" })
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
                Sharding::new(
                    mesh,
                    vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                )
                .unwrap(),
            ),
        )
        .unwrap();

        assert_eq!(
            reshape_abstract(&input_type, &Shape::new(vec![Size::Static(8)]), "reshape"),
            Err(TraceError::IncompatibleAbstractValues { op: "reshape" })
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
                Sharding::new(
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
                    Sharding::new(
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
        let (_, compiled): (
            ndarray::Array2<f64>,
            CompiledFunction<ndarray::Array2<f64>, ndarray::Array2<f64>, ndarray::Array2<f64>>,
        ) = try_jit(
            |x: JitTracer<ndarray::Array2<f64>>| x.reshape(Shape::new(vec![Size::Static(1), Size::Static(4)])),
            input,
        )
        .unwrap();

        assert_eq!(
            compiled.to_string(),
            indoc! {"
                lambda %0:f64[2, 2] .
                let %1:f64[1, 4] = reshape[1, 4] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_reshape_transpose_restores_the_input_shape() {
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2)]), None, None).unwrap();
        let output_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Static(4)]), None, None).unwrap();
        let mut forward_builder = ProgramBuilder::<ndarray::Array2<f64>>::new();
        let input = forward_builder.add_input_abstract(input_type.clone(), arr2(&[[1.0f64, 2.0], [3.0, 4.0]]));
        let output = forward_builder
            .add_equation(Arc::new(ReshapeOp::new(input_type.clone(), output_type.clone())), vec![input])
            .unwrap()[0];

        let mut transpose_builder = ProgramBuilder::<ndarray::Array2<f64>>::new();
        let output_cotangent = transpose_builder.add_input_abstract(output_type, arr2(&[[1.0f64, 2.0, 3.0, 4.0]]));
        let contribution = ReshapeOp::new(
            input_type.clone(),
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Static(4)]), None, None).unwrap(),
        )
        .transpose_program_op(&mut transpose_builder, &[input], &[output], &[output_cotangent])
        .unwrap()[0]
            .unwrap();

        let transpose_graph = transpose_builder.build::<ndarray::Array2<f64>, ndarray::Array2<f64>>(
            vec![contribution],
            Placeholder,
            Placeholder,
        );
        assert_eq!(transpose_graph.call(arr2(&[[1.0f64, 2.0, 3.0, 4.0]])).unwrap(), arr2(&[[1.0f64, 2.0], [3.0, 4.0]]));
    }

    #[cfg(feature = "xla")]
    #[test]
    fn test_trace_reshape_lowers_to_stablehlo_reshape() {
        let input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(3)]), None, None).unwrap();
        let traced: crate::xla::shard_map::TracedXlaProgram<ArrayType, ArrayType> = trace(
            |x: JitTracer<ShardMapTensor>| x.reshape(Shape::new(vec![Size::Static(3), Size::Static(2)])).unwrap(),
            input_type.clone(),
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
                    %0 = stablehlo.reshape %arg0 : (tensor<2x3xf32>) -> tensor<3x2xf32>
                    return %0 : tensor<3x2xf32>
                  }
                }
            "#}
        );
    }

    #[cfg(feature = "xla")]
    #[test]
    fn test_trace_reshape_with_sharding_constraint_renders_stablehlo_and_shardy() {
        let mesh = test_manual_mesh(4);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, None).unwrap();
        let sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], Vec::new());

        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let sharding = sharding.clone();
                move |x: JitTracer<ShardMapTensor>| {
                    with_sharding_constraint(x, sharding.clone())
                        .expect("with_sharding_constraint should stage before reshape")
                        .reshape(Shape::new(vec![Size::Static(1), Size::Static(8), Size::Static(1)]))
                        .unwrap()
                }
            },
            input_type,
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> tensor<1x8x1xf32> {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{"x"}]> : tensor<8xf32>
                    %1 = stablehlo.reshape %0 : (tensor<8xf32>) -> tensor<1x8x1xf32>
                    return %1 : tensor<1x8x1xf32>
                  }
                }
            "#}
        );
    }

    #[cfg(feature = "xla")]
    #[test]
    fn test_shard_map_reshape_renders_singleton_axis_sharding_propagation() {
        let mesh = test_manual_mesh(4);
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, None).unwrap();
        let input_sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], Vec::new());
        let output_sharding = test_sharding(
            &mesh,
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
            Vec::new(),
        );

        let traced: TracedShardMap<ArrayType, ArrayType> = shard_map(
            |x: ShardMapTracer| x.reshape(Shape::new(vec![Size::Static(1), Size::Static(2), Size::Static(1)])).unwrap(),
            global_input_type,
            mesh.clone(),
            input_sharding.clone(),
            output_sharding.clone(),
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<1x8x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{}, {"x"}, {}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = stablehlo.reshape %arg1 : (tensor<2xf32>) -> tensor<1x2x1xf32>
                      sdy.return %1 : tensor<1x2x1xf32>
                    } : (tensor<8xf32>) -> tensor<1x8x1xf32>
                    return %0 : tensor<1x8x1xf32>
                  }
                }
            "#}
        );
    }

    #[cfg(feature = "xla")]
    #[test]
    fn test_shard_map_reshape_renders_replicated_merge_sharding_propagation() {
        let mesh = test_manual_mesh(4);
        let global_input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(8), Size::Static(2), Size::Static(3)]),
            None,
            None,
        )
        .unwrap();
        let input_sharding = test_sharding(
            &mesh,
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated(), ShardingDimension::replicated()],
            Vec::new(),
        );
        let output_sharding =
            test_sharding(&mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()], Vec::new());

        let traced: TracedShardMap<ArrayType, ArrayType> = shard_map(
            |x: ShardMapTracer| x.reshape(Shape::new(vec![Size::Static(2), Size::Static(6)])).unwrap(),
            global_input_type,
            mesh.clone(),
            input_sharding.clone(),
            output_sharding.clone(),
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8x2x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) -> (tensor<8x6xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}, {}, {}]>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x"} (%arg1: tensor<2x2x3xf32>) {
                      %1 = stablehlo.reshape %arg1 : (tensor<2x2x3xf32>) -> tensor<2x6xf32>
                      sdy.return %1 : tensor<2x6xf32>
                    } : (tensor<8x2x3xf32>) -> tensor<8x6xf32>
                    return %0 : tensor<8x6xf32>
                  }
                }
            "#}
        );
    }

    #[cfg(feature = "xla")]
    #[test]
    fn test_shard_map_reshape_renders_replicated_split_sharding_propagation() {
        let mesh = test_manual_mesh(4);
        let global_input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8), Size::Static(6)]), None, None).unwrap();
        let input_sharding =
            test_sharding(&mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()], Vec::new());
        let output_sharding = test_sharding(
            &mesh,
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated(), ShardingDimension::replicated()],
            Vec::new(),
        );

        let traced: TracedShardMap<ArrayType, ArrayType> = shard_map(
            |x: ShardMapTracer| x.reshape(Shape::new(vec![Size::Static(2), Size::Static(2), Size::Static(3)])).unwrap(),
            global_input_type,
            mesh.clone(),
            input_sharding.clone(),
            output_sharding.clone(),
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8x6xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x2x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}, {}]>] out_shardings=[<@mesh, [{"x"}, {}, {}]>] manual_axes={"x"} (%arg1: tensor<2x6xf32>) {
                      %1 = stablehlo.reshape %arg1 : (tensor<2x6xf32>) -> tensor<2x2x3xf32>
                      sdy.return %1 : tensor<2x2x3xf32>
                    } : (tensor<8x6xf32>) -> tensor<8x2x3xf32>
                    return %0 : tensor<8x2x3xf32>
                  }
                }
            "#}
        );
    }
}
