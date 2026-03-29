//! Sharding-constraint primitive for traced XLA programs.

use std::{
    any::Any,
    fmt::{Debug, Display},
    sync::Arc,
};

use ryft_mlir::dialects::shardy;
use ryft_mlir::{Block, Operation, Value, ValueRef};

use crate::tracing_v2::{
    FloatExt, MatrixOps, TraceError, TraceValue, TransformLeaf, ZeroLike, forward::JvpTracer, graph::AtomId,
    jit::JitTracer, linear::LinearTerm, ops::Op, program::ProgramBuilder,
};
use crate::types::ArrayType;
use crate::xla::{
    lowering::{LoweringError, MlirLowerableValue, ShardMapMlirLowerer},
    sharding::Sharding,
};

use super::{expect_input_count, unary_abstract};

/// Unary primitive that constrains one traced XLA value to a requested sharding.
#[derive(Clone)]
pub(crate) struct WithShardingConstraintOp {
    sharding: Sharding,
}

impl WithShardingConstraintOp {
    /// Creates one sharding-constraint op with the provided target sharding.
    #[inline]
    pub(crate) fn new(sharding: Sharding) -> Self {
        Self { sharding }
    }

    /// Returns the target sharding carried by this op.
    #[inline]
    pub(crate) fn sharding(&self) -> &Sharding {
        &self.sharding
    }
}

impl Debug for WithShardingConstraintOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "WithShardingConstraint")
    }
}

impl Display for WithShardingConstraintOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "with_sharding_constraint")
    }
}

impl<V> Op<V> for WithShardingConstraintOp
where
    V: TraceValue,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "with_sharding_constraint"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        let mut output = unary_abstract(inputs)?;
        if output.rank() != self.sharding().rank() {
            return Err(TraceError::IncompatibleAbstractValues { op: "with_sharding_constraint" });
        }
        let mut sharding = self.sharding().clone();
        sharding.varying_manual_axes = output
            .sharding
            .as_ref()
            .map(|input_sharding| input_sharding.varying_manual_axes.clone())
            .unwrap_or_default();
        output.sharding = Some(sharding);
        Ok(vec![output])
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone()])
    }

    fn replay_linearized_jit(
        &self,
        inputs: Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>,
    ) -> Result<Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>, TraceError>
    where
        V: TransformLeaf,
    {
        expect_input_count(inputs.len(), 1)?;
        let input = inputs.into_iter().next().expect("validated single input should exist");
        let primal = JitTracer::apply_staged_op(
            std::slice::from_ref(&input.primal),
            Arc::new(self.clone()),
            vec![input.primal.value.clone()],
        )?
        .into_iter()
        .next()
        .expect("sharding constraint should produce one primal output");
        let tangent = LinearTerm::apply_staged_op(std::slice::from_ref(&input.tangent), Arc::new(self.clone()), 1)?
            .into_iter()
            .next()
            .expect("sharding constraint should produce one tangent output");
        Ok(vec![JvpTracer { primal, tangent }])
    }

    fn apply_program_jvp_rule(
        &self,
        inputs: &[JvpTracer<V, LinearTerm<V>>],
    ) -> Result<Vec<JvpTracer<V, LinearTerm<V>>>, TraceError>
    where
        V: FloatExt + ZeroLike + MatrixOps,
    {
        expect_input_count(inputs.len(), 1)?;
        let tangent = LinearTerm::apply_staged_op(std::slice::from_ref(&inputs[0].tangent), Arc::new(self.clone()), 1)?
            .into_iter()
            .next()
            .expect("sharding constraint should produce one tangent output");
        Ok(vec![JvpTracer { primal: inputs[0].primal.clone(), tangent }])
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
        let contribution = builder.add_equation(
            Arc::new(WithShardingConstraintOp::new(self.sharding().clone())),
            vec![output_cotangents[0]],
        )?[0];
        Ok(vec![Some(contribution)])
    }

    fn lower_shard_map_mlir<'b, 'c, 't>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        V: MlirLowerableValue,
    {
        let operation = lowerer.block.append_operation(shardy::sharding_constraint(
            input_values[0],
            self.sharding().to_shardy_tensor_sharding(lowerer.context),
            lowerer.location,
        ));
        Ok(vec![operation.result(0).expect("sdy.sharding_constraint should return one result").as_ref()])
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, ShardingDimension};
    use crate::tracing_v2::ProgramBuilder;
    use crate::types::{ArrayType, DataType, Shape, Size};
    use crate::xla::shard_map::ShardMapTensor;
    use crate::xla::sharding::Sharding;

    use super::*;

    fn test_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap()
    }

    fn test_sharding(mesh: &LogicalMesh) -> Sharding {
        Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::sharded(["x"])],
            Vec::<&str>::new(),
            Vec::<&str>::new(),
            Vec::<&str>::new(),
        )
        .unwrap()
    }

    #[test]
    fn test_with_sharding_constraint_abstract_eval_attaches_sharding() {
        let mesh = test_mesh();
        let sharding = test_sharding(&mesh);
        let op = WithShardingConstraintOp::new(sharding.clone());

        assert_eq!(
            <WithShardingConstraintOp as Op<ShardMapTensor>>::abstract_eval(
                &op,
                &[ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, None)],
            ),
            Ok(vec![ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, Some(sharding),)])
        );
    }

    #[test]
    fn test_with_sharding_constraint_abstract_eval_preserves_varying_axes() {
        let mesh = test_mesh();
        let target_sharding = test_sharding(&mesh);
        let input_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()], Vec::<&str>::new(), Vec::<&str>::new(), ["x"])
                .unwrap();
        let op = WithShardingConstraintOp::new(target_sharding);

        assert_eq!(
            <WithShardingConstraintOp as Op<ShardMapTensor>>::abstract_eval(
                &op,
                &[
                    ArrayType::new(
                        DataType::F32,
                        Shape::new(vec![Size::Static(8)]),
                        None,
                        Some(input_sharding.clone()),
                    )
                ],
            ),
            Ok(vec![ArrayType::new(
                DataType::F32,
                Shape::new(vec![Size::Static(8)]),
                None,
                Some(
                    Sharding::new(
                        input_sharding.mesh.clone(),
                        vec![ShardingDimension::sharded(["x"])],
                        Vec::<&str>::new(),
                        Vec::<&str>::new(),
                        ["x"],
                    )
                    .unwrap()
                ),
            )])
        );
    }

    #[test]
    fn test_with_sharding_constraint_abstract_eval_preserves_reduced_and_unreduced_axes() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 4, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 4, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let target_sharding =
            Sharding::new(
                mesh.clone(),
                vec![ShardingDimension::sharded(["x"])],
                Vec::<&str>::new(),
                Vec::<&str>::new(),
                Vec::<&str>::new(),
            )
            .unwrap();
        let input_sharding = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::replicated()],
            ["y"],
            ["z"],
            ["x"],
        )
        .unwrap();
        let op = WithShardingConstraintOp::new(target_sharding);

        assert_eq!(
            <WithShardingConstraintOp as Op<ShardMapTensor>>::abstract_eval(
                &op,
                &[ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, Some(input_sharding),)],
            ),
            Ok(vec![ArrayType::new(
                DataType::F32,
                Shape::new(vec![Size::Static(8)]),
                None,
                Some(
                    Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])], Vec::<&str>::new(), Vec::<&str>::new(), ["x"])
                        .unwrap()
                ),
            )])
        );
    }

    #[test]
    fn test_with_sharding_constraint_abstract_eval_rejects_rank_mismatch() {
        let mesh = test_mesh();
        let op = WithShardingConstraintOp::new(test_sharding(&mesh));

        assert_eq!(
            <WithShardingConstraintOp as Op<ShardMapTensor>>::abstract_eval(
                &op,
                &[ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8), Size::Static(4)]), None, None)],
            ),
            Err(TraceError::IncompatibleAbstractValues { op: "with_sharding_constraint" })
        );
    }

    #[test]
    fn test_with_sharding_constraint_transpose_preserves_the_constraint() {
        let mesh = test_mesh();
        let sharding = test_sharding(&mesh);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, None);

        let mut forward_builder = ProgramBuilder::<ShardMapTensor>::new();
        let input = forward_builder.add_input(&ShardMapTensor::new(input_type.clone()));
        let output = forward_builder
            .add_equation(Arc::new(WithShardingConstraintOp::new(sharding.clone())), vec![input])
            .unwrap()[0];

        let mut transpose_builder = ProgramBuilder::<ShardMapTensor>::new();
        let output_cotangent = transpose_builder.add_input(&ShardMapTensor::new(input_type));
        let contribution = WithShardingConstraintOp::new(sharding.clone())
            .transpose_program_op(&mut transpose_builder, &[input], &[output], &[output_cotangent])
            .unwrap()[0]
            .unwrap();

        let transpose_graph =
            transpose_builder.build::<ShardMapTensor, ShardMapTensor>(vec![contribution], Placeholder, Placeholder);
        assert_eq!(
            transpose_graph.to_string(),
            format!("lambda %0:f32[8] .\nlet %1:f32[8][sharding={sharding}] = with_sharding_constraint %0\nin (%1)")
                .trim_end(),
        );
    }
}
