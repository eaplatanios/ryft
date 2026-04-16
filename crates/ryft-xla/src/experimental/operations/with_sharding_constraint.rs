//! Sharding-constraint primitive for traced XLA programs.

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use ryft_core::sharding::Sharding;
use ryft_core::tracing_v2::{
    CustomPrimitive, DifferentiableOp, InterpretableOp, JitTracer, LinearOp, LinearPrimitiveOp, PrimitiveOp,
    TraceError, VectorizableOp,
    forward::JvpTracer,
    linear::{LinearTerm, Linearized},
    operations::{expect_input_count, unary_abstract},
    ops::Op,
};
use ryft_core::types::ArrayType;
use ryft_mlir::{Block, Operation, Value};

use crate::experimental::lowering::{
    LoweringError, ShardMapMlirLowerer, StableHloCustomLowering, StableHloCustomLoweringExtension,
};
use crate::experimental::shard_map::{ShardMapTensor, ShardMapTracer};
use crate::mlir::ToMlir;

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

    fn base_custom_primitive<V>(&self) -> CustomPrimitive<V>
    where
        V: ryft_core::tracing_v2::Traceable<ArrayType>,
        Self: InterpretableOp<V>
            + LinearOp<V>
            + DifferentiableOp<V, LinearTerm<V>>
            + VectorizableOp<V>
            + Clone
            + Send
            + Sync
            + 'static,
    {
        CustomPrimitive::new(self.clone())
            .with_transpose_rule(self.clone())
            .with_jvp_rule(self.clone())
            .with_vectorization_rule(self.clone())
    }

    /// Returns the tensor-leaf custom primitive registration for this op.
    pub(crate) fn to_tensor_custom_primitive(&self) -> CustomPrimitive<ShardMapTensor> {
        self.base_custom_primitive::<ShardMapTensor>()
            .with_linearized_jit_rule(self.clone())
            .with_extension(self.clone())
            .with_extension(StableHloCustomLoweringExtension::new(Arc::new(self.clone())))
    }

    /// Returns the traced-leaf custom primitive registration for this op.
    pub(crate) fn to_tracer_custom_primitive(&self) -> CustomPrimitive<ShardMapTracer> {
        self.base_custom_primitive::<ShardMapTracer>().with_linearized_jit_rule(self.clone())
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

impl InterpretableOp<ShardMapTensor> for WithShardingConstraintOp {
    fn interpret(&self, inputs: &[ShardMapTensor]) -> Result<Vec<ShardMapTensor>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone()])
    }
}

impl LinearOp<ShardMapTensor> for WithShardingConstraintOp {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ShardMapTensor>],
    ) -> Result<Vec<Option<LinearTerm<ShardMapTensor>>>, TraceError> {
        expect_input_count(output_cotangents.len(), 1)?;
        let contribution = LinearTerm::apply_staged_op(
            std::slice::from_ref(&output_cotangents[0]),
            LinearPrimitiveOp::custom(self.to_tensor_custom_primitive())?,
            1,
        )?
        .into_iter()
        .next()
        .expect("sharding constraint should produce one cotangent contribution");
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
            LinearPrimitiveOp::custom(self.to_tensor_custom_primitive())?,
            1,
        )?
        .into_iter()
        .next()
        .expect("sharding constraint should produce one tangent output");
        Ok(vec![JvpTracer { primal: inputs[0].primal.clone(), tangent }])
    }
}

impl InterpretableOp<Linearized<ShardMapTracer>> for WithShardingConstraintOp {
    fn interpret(&self, inputs: &[Linearized<ShardMapTracer>]) -> Result<Vec<Linearized<ShardMapTracer>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        let input = &inputs[0];
        let primal = JitTracer::apply_staged_op(
            std::slice::from_ref(&input.primal),
            PrimitiveOp::Custom(Arc::new(self.to_tensor_custom_primitive())),
            vec![input.primal.value.clone()],
        )?
        .into_iter()
        .next()
        .expect("sharding constraint should produce one primal output");
        let tangent = LinearTerm::apply_staged_op(
            std::slice::from_ref(&input.tangent),
            LinearPrimitiveOp::custom(self.to_tracer_custom_primitive())?,
            1,
        )?
        .into_iter()
        .next()
        .expect("sharding constraint should produce one tangent output");
        Ok(vec![Linearized { primal, tangent }])
    }
}

impl InterpretableOp<ShardMapTracer> for WithShardingConstraintOp {
    fn interpret(&self, inputs: &[ShardMapTracer]) -> Result<Vec<ShardMapTracer>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone()])
    }
}

impl LinearOp<ShardMapTracer> for WithShardingConstraintOp {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ShardMapTracer>],
    ) -> Result<Vec<Option<LinearTerm<ShardMapTracer>>>, TraceError> {
        expect_input_count(output_cotangents.len(), 1)?;
        let contribution = LinearTerm::apply_staged_op(
            std::slice::from_ref(&output_cotangents[0]),
            LinearPrimitiveOp::custom(self.to_tracer_custom_primitive())?,
            1,
        )?
        .into_iter()
        .next()
        .expect("sharding constraint should produce one cotangent contribution");
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
            LinearPrimitiveOp::custom(self.to_tracer_custom_primitive())?,
            1,
        )?
        .into_iter()
        .next()
        .expect("sharding constraint should produce one tangent output");
        Ok(vec![JvpTracer { primal: inputs[0].primal.clone(), tangent }])
    }
}

impl<V: ryft_core::tracing_v2::Traceable<ArrayType>> VectorizableOp<V> for WithShardingConstraintOp {
    fn batch(
        &self,
        inputs: &[ryft_core::tracing_v2::Batch<V>],
    ) -> Result<Vec<ryft_core::tracing_v2::Batch<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone()])
    }
}

impl InterpretableOp<Linearized<JitTracer<ShardMapTracer>>> for WithShardingConstraintOp {
    fn interpret(
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

impl StableHloCustomLowering<ShardMapTensor> for WithShardingConstraintOp {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        _op: &CustomPrimitive<ShardMapTensor>,
        input_values: &[ryft_mlir::ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ryft_mlir::ValueRef<'b, 'c, 't>>, LoweringError> {
        let operation = lowerer.block.append_operation(ryft_mlir::dialects::shardy::sharding_constraint(
            input_values[0],
            self.sharding().to_mlir(lowerer.location),
            lowerer.location,
        ));
        Ok(vec![operation.result(0).expect("sdy.sharding_constraint should return one result").as_ref()])
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use pretty_assertions::assert_eq;

    use ryft_core::parameters::Placeholder;
    use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use ryft_core::tracing_v2::{LinearOp, LinearProgramBuilder, LinearTerm};
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

        let transpose_builder = Rc::new(RefCell::new(LinearProgramBuilder::<ShardMapTensor>::new()));
        let output_cotangent_atom = transpose_builder.borrow_mut().add_input(&ShardMapTensor::new(input_type.clone()));
        let output_cotangent = LinearTerm::from_staged_parts(output_cotangent_atom, transpose_builder.clone());
        let contribution = LinearOp::transpose(&WithShardingConstraintOp::new(sharding.clone()), &[output_cotangent])
            .unwrap()
            .into_iter()
            .next()
            .expect("transpose should return one contribution")
            .expect("transpose should produce one cotangent contribution");
        let contribution_atom = contribution.atom();
        drop(contribution);

        let transpose_builder = Rc::try_unwrap(transpose_builder)
            .expect("transpose builder should not have outstanding linear terms")
            .into_inner();
        let transpose_graph = transpose_builder.build::<ShardMapTensor, ShardMapTensor>(
            vec![contribution_atom],
            Placeholder,
            Placeholder,
        );
        assert_eq!(
            transpose_graph.to_string(),
            format!("lambda %0:f32[8] .\nlet %1:f32[8][sharding={sharding}] = with_sharding_constraint %0\nin (%1)")
                .trim_end(),
        );
    }
}
