//! Sharding-constraint primitive for traced XLA programs.

use std::{
    any::Any,
    fmt::{Debug, Display},
    sync::Arc,
};

use ryft_core::sharding::Sharding;
use ryft_core::tracing_v2::{
    Eval, PrimitiveOp, TraceError,
    forward::JvpTracer,
    graph::AtomId,
    jit::JitTracer,
    linear::{LinearTerm, Linearized},
    operations::{expect_input_count, unary_abstract},
    ops::{DifferentiableOp, LinearOp, Op},
    program::ProgramBuilder,
};
use ryft_core::types::ArrayType;

use crate::experimental::shard_map::{ShardMapTensor, ShardMapTracer};

/// Unary primitive that constrains one traced XLA value to a requested sharding.
#[derive(Clone)]
pub struct WithShardingConstraintOp {
    sharding: Sharding,
}

impl WithShardingConstraintOp {
    /// Creates one sharding-constraint op with the provided target sharding.
    #[inline]
    pub fn new(sharding: Sharding) -> Self {
        Self { sharding }
    }

    /// Returns the target sharding carried by this op.
    #[inline]
    pub fn sharding(&self) -> &Sharding {
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

impl Op for WithShardingConstraintOp {
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
}

impl Eval<ShardMapTensor> for WithShardingConstraintOp {
    fn eval(&self, inputs: &[ShardMapTensor]) -> Result<Vec<ShardMapTensor>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone()])
    }
}

impl LinearOp<ShardMapTensor> for WithShardingConstraintOp {
    fn transpose_program_op(
        &self,
        builder: &mut ProgramBuilder<ShardMapTensor>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        expect_input_count(outputs.len(), 1)?;
        expect_input_count(output_cotangents.len(), 1)?;
        let contribution = builder.add_equation(
            PrimitiveOp::Custom(Arc::new(WithShardingConstraintOp::new(self.sharding().clone()))),
            vec![output_cotangents[0]],
        )?[0];
        Ok(vec![Some(contribution)])
    }
}

impl DifferentiableOp<ShardMapTensor, LinearTerm<ShardMapTensor>> for WithShardingConstraintOp {
    fn jvp(
        &self,
        inputs: &[JvpTracer<ShardMapTensor, LinearTerm<ShardMapTensor>>],
    ) -> Result<Vec<JvpTracer<ShardMapTensor, LinearTerm<ShardMapTensor>>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        let tangent = LinearTerm::apply_staged_op(
            std::slice::from_ref(&inputs[0].tangent),
            PrimitiveOp::Custom(Arc::new(self.clone())),
            1,
        )?
        .into_iter()
        .next()
        .expect("sharding constraint should produce one tangent output");
        Ok(vec![JvpTracer { primal: inputs[0].primal.clone(), tangent }])
    }
}

impl Eval<Linearized<ShardMapTracer>> for WithShardingConstraintOp {
    fn eval(&self, inputs: &[Linearized<ShardMapTracer>]) -> Result<Vec<Linearized<ShardMapTracer>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        let input = &inputs[0];
        let primal = JitTracer::apply_staged_op(
            std::slice::from_ref(&input.primal),
            PrimitiveOp::Custom(Arc::new(self.clone())),
            vec![input.primal.value.clone()],
        )?
        .into_iter()
        .next()
        .expect("sharding constraint should produce one primal output");
        let tangent = LinearTerm::apply_staged_op(
            std::slice::from_ref(&input.tangent),
            PrimitiveOp::Custom(Arc::new(self.clone())),
            1,
        )?
        .into_iter()
        .next()
        .expect("sharding constraint should produce one tangent output");
        Ok(vec![Linearized { primal, tangent }])
    }
}

impl Eval<ShardMapTracer> for WithShardingConstraintOp {
    fn eval(&self, inputs: &[ShardMapTracer]) -> Result<Vec<ShardMapTracer>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone()])
    }
}

impl LinearOp<ShardMapTracer> for WithShardingConstraintOp {
    fn transpose_program_op(
        &self,
        builder: &mut ProgramBuilder<ShardMapTracer>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        expect_input_count(outputs.len(), 1)?;
        expect_input_count(output_cotangents.len(), 1)?;
        let contribution = builder.add_equation(
            PrimitiveOp::Custom(Arc::new(WithShardingConstraintOp::new(self.sharding().clone()))),
            vec![output_cotangents[0]],
        )?[0];
        Ok(vec![Some(contribution)])
    }
}

impl DifferentiableOp<ShardMapTracer, LinearTerm<ShardMapTracer>> for WithShardingConstraintOp {
    fn jvp(
        &self,
        inputs: &[JvpTracer<ShardMapTracer, LinearTerm<ShardMapTracer>>],
    ) -> Result<Vec<JvpTracer<ShardMapTracer, LinearTerm<ShardMapTracer>>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        let tangent = LinearTerm::apply_staged_op(
            std::slice::from_ref(&inputs[0].tangent),
            PrimitiveOp::Custom(Arc::new(self.clone())),
            1,
        )?
        .into_iter()
        .next()
        .expect("sharding constraint should produce one tangent output");
        Ok(vec![JvpTracer { primal: inputs[0].primal.clone(), tangent }])
    }
}

impl Eval<Linearized<JitTracer<ShardMapTracer>>> for WithShardingConstraintOp {
    fn eval(
        &self,
        _inputs: &[Linearized<JitTracer<ShardMapTracer>>],
    ) -> Result<Vec<Linearized<JitTracer<ShardMapTracer>>>, TraceError> {
        Err(TraceError::HigherOrderOpFailure {
            op: "eval_linearized_jit",
            message: "linearized JIT evaluation for 'with_sharding_constraint' at the JIT-tracer level is not \
                      supported"
                .to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::parameters::Placeholder;
    use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use ryft_core::tracing_v2::ProgramBuilder;
    use ryft_core::types::{ArrayType, DataType, Shape, Size};

    use super::*;

    fn test_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap()
    }

    fn test_sharding(mesh: &LogicalMesh) -> Sharding {
        Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()
    }

    #[test]
    fn test_with_sharding_constraint_abstract_eval_attaches_sharding() {
        let mesh = test_mesh();
        let sharding = test_sharding(&mesh);
        let op = WithShardingConstraintOp::new(sharding.clone());

        assert_eq!(
            <WithShardingConstraintOp as Op>::abstract_eval(
                &op,
                &[ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, None).unwrap()],
            ),
            Ok(vec![ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, Some(sharding)).unwrap()])
        );
    }

    #[test]
    fn test_with_sharding_constraint_abstract_eval_preserves_varying_axes() {
        let mesh = test_mesh();
        let target_sharding = test_sharding(&mesh);
        let input_sharding = Sharding::with_manual_axes(
            mesh.clone(),
            vec![ShardingDimension::replicated()],
            Vec::<&str>::new(),
            Vec::<&str>::new(),
            ["x"],
        )
        .unwrap();
        let op = WithShardingConstraintOp::new(target_sharding);

        assert_eq!(
            <WithShardingConstraintOp as Op>::abstract_eval(
                &op,
                &[
                    ArrayType::new(
                        DataType::F32,
                        Shape::new(vec![Size::Static(8)]),
                        None,
                        Some(input_sharding.clone()),
                    )
                    .unwrap()
                ],
            ),
            Ok(vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Size::Static(8)]),
                    None,
                    Some(
                        Sharding::with_manual_axes(
                            input_sharding.mesh.clone(),
                            vec![ShardingDimension::sharded(["x"])],
                            Vec::<&str>::new(),
                            Vec::<&str>::new(),
                            ["x"],
                        )
                        .unwrap()
                    ),
                )
                .unwrap()
            ])
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
        let target_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_sharding =
            Sharding::with_manual_axes(mesh.clone(), vec![ShardingDimension::replicated()], ["y"], ["z"], ["x"])
                .unwrap();
        let op = WithShardingConstraintOp::new(target_sharding);

        assert_eq!(
            <WithShardingConstraintOp as Op>::abstract_eval(
                &op,
                &[ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, Some(input_sharding),)
                    .unwrap()],
            ),
            Ok(vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Size::Static(8)]),
                    None,
                    Some(
                        Sharding::with_manual_axes(
                            mesh,
                            vec![ShardingDimension::sharded(["x"])],
                            Vec::<&str>::new(),
                            Vec::<&str>::new(),
                            ["x"]
                        )
                        .unwrap()
                    ),
                )
                .unwrap()
            ])
        );
    }

    #[test]
    fn test_with_sharding_constraint_abstract_eval_rejects_rank_mismatch() {
        let mesh = test_mesh();
        let op = WithShardingConstraintOp::new(test_sharding(&mesh));

        assert_eq!(
            <WithShardingConstraintOp as Op>::abstract_eval(
                &op,
                &[ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8), Size::Static(4)]), None, None)
                    .unwrap()],
            ),
            Err(TraceError::IncompatibleAbstractValues { op: "with_sharding_constraint" })
        );
    }

    #[test]
    fn test_with_sharding_constraint_transpose_preserves_the_constraint() {
        let mesh = test_mesh();
        let sharding = test_sharding(&mesh);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, None).unwrap();

        let mut forward_builder = ProgramBuilder::<ShardMapTensor>::new();
        let input = forward_builder.add_input(&ShardMapTensor::new(input_type.clone()));
        let output = forward_builder
            .add_equation(PrimitiveOp::Custom(Arc::new(WithShardingConstraintOp::new(sharding.clone()))), vec![input])
            .unwrap()[0];

        let mut transpose_builder = ProgramBuilder::<ShardMapTensor>::new();
        let output_cotangent = transpose_builder.add_input(&ShardMapTensor::new(input_type));
        let contribution = LinearOp::transpose_program_op(
            &WithShardingConstraintOp::new(sharding.clone()),
            &mut transpose_builder,
            &[input],
            &[output],
            &[output_cotangent],
        )
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
