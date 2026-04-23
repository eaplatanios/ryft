use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use ryft_core::macros::check_input_count;
use ryft_core::sharding::Sharding;
use ryft_core::tracing::{InterpretableOperation, Operation, Traceable, TracingError};
use ryft_core::tracing_v2::{
    CustomPrimitive, DifferentiableOperation, LinearCustomPrimitive, LinearOperation, LinearPrimitiveOperation, Tracer,
    engine::Engine,
    forward::{Differentiable, EngineTangent, JvpTracer},
    linear::{LinearTerm, Linearized},
    operations::unary_abstract,
};
use ryft_core::types::{ArrayType, TypeError};
use ryft_mlir::{Block, Operation as MlirOperation, Value};

use crate::experimental::lowering::{
    LoweringError, ShardMapMlirLowerer, StableHloCustomLowering, StableHloCustomLoweringExtension,
};
use crate::experimental::operations::shard_map::ShardMapCustomReplayExtension;
use crate::experimental::ops::XlaPrimitiveOperation;
use crate::experimental::shard_map::{ShardMapTensor, ShardMapTracer};
use crate::mlir::ToMlir;

/// Unary primitive that constrains one traced XLA value to a requested sharding.
#[derive(Clone)]
pub struct WithShardingConstraintOperation {
    /// Requested sharding that the input leaf must satisfy after lowering.
    sharding: Sharding,
}

impl WithShardingConstraintOperation {
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

    fn base_custom_primitive<V>(&self) -> CustomPrimitive<ArrayType, V>
    where
        V: Traceable<ArrayType> + 'static,
        Self: Clone + InterpretableOperation<ArrayType, V> + LinearOperation<ArrayType, V> + Send + Sync + 'static,
    {
        CustomPrimitive::new(self.clone()).with_transpose_rule(self.clone())
    }

    /// Returns the tensor-leaf custom primitive registration for this op.
    pub(crate) fn to_tensor_custom_primitive(&self) -> CustomPrimitive<ArrayType, ShardMapTensor> {
        self.base_custom_primitive::<ShardMapTensor>()
            .with_jvp_rule_for::<crate::experimental::engine::XlaEngine<'static>, _>(self.clone())
            .with_linearized_jit_rule_for::<
                XlaPrimitiveOperation,
                LinearPrimitiveOperation<ArrayType, ShardMapTensor>,
                LinearPrimitiveOperation<ArrayType, ShardMapTracer>,
                crate::experimental::engine::XlaEngine<'static>,
                _,
            >(self.clone())
            .with_extension(self.clone())
            .with_extension(ShardMapCustomReplayExtension::<ShardMapTracer>::new(|_, inputs| Ok(vec![inputs[0].clone()])))
            .with_extension(ShardMapCustomReplayExtension::<Linearized<ShardMapTracer>>::new(|_, inputs| {
                Ok(vec![inputs[0].clone()])
            }))
            .with_extension(StableHloCustomLoweringExtension::new(Arc::new(self.clone())))
    }

    /// Returns the traced-leaf linear custom primitive registration used by tangent programs.
    pub(crate) fn to_tracer_linear_custom_primitive(&self) -> LinearCustomPrimitive<ArrayType, ShardMapTracer> {
        CustomPrimitive::new(self.clone())
            .with_transpose_rule(self.clone())
            .into_linear()
            .expect("with_sharding_constraint traced linear primitive should carry a transpose rule")
    }
}

impl Debug for WithShardingConstraintOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "WithShardingConstraint")
    }
}

impl Display for WithShardingConstraintOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "with_sharding_constraint")
    }
}

impl Operation for WithShardingConstraintOperation {
    fn name(&self) -> &'static str {
        "with_sharding_constraint"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        let mut output = unary_abstract(input_types)?;
        if output.rank() != self.sharding().rank() {
            return Err(TypeError {
                message: "with_sharding_constraint rank does not match the requested sharding rank".to_string(),
            });
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

impl InterpretableOperation<ArrayType, ShardMapTensor> for WithShardingConstraintOperation {
    fn interpret(&self, inputs: &[ShardMapTensor]) -> Result<Vec<ShardMapTensor>, TracingError> {
        check_input_count!(inputs, 1);
        Ok(vec![inputs[0].clone()])
    }
}

impl LinearOperation<ArrayType, ShardMapTensor> for WithShardingConstraintOperation {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, ShardMapTensor>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, ShardMapTensor>>>, TracingError> {
        check_input_count!(output_cotangents, 1);
        let contribution = LinearTerm::apply_staged_op(
            output_cotangents[0].builder.clone(),
            std::slice::from_ref(&output_cotangents[0]),
            LinearPrimitiveOperation::custom(self.to_tensor_custom_primitive())?,
            1,
        )?
        .into_iter()
        .next()
        .expect("sharding constraint should produce one cotangent contribution");
        Ok(vec![Some(contribution)])
    }
}

trait WithShardingConstraintJvpValue<E>: Clone + Differentiable<ArrayType> + Traceable<ArrayType>
where
    E: Engine<Type = ArrayType, Value = Self, LinearOperation = LinearPrimitiveOperation<ArrayType, Self>> + ?Sized,
    Self: Sized,
{
    fn apply_constraint_tangent(
        op: &WithShardingConstraintOperation,
        tangent: &LinearTerm<ArrayType, Self, LinearPrimitiveOperation<ArrayType, Self>>,
    ) -> Result<LinearTerm<ArrayType, Self, LinearPrimitiveOperation<ArrayType, Self>>, TracingError>;
}

impl<E> WithShardingConstraintJvpValue<E> for ShardMapTensor
where
    E: Engine<
            Type = ArrayType,
            Value = ShardMapTensor,
            LinearOperation = LinearPrimitiveOperation<ArrayType, ShardMapTensor>,
        > + ?Sized,
{
    fn apply_constraint_tangent(
        op: &WithShardingConstraintOperation,
        tangent: &LinearTerm<ArrayType, ShardMapTensor, LinearPrimitiveOperation<ArrayType, ShardMapTensor>>,
    ) -> Result<LinearTerm<ArrayType, ShardMapTensor, LinearPrimitiveOperation<ArrayType, ShardMapTensor>>, TracingError>
    {
        LinearTerm::apply_staged_op(
            tangent.builder.clone(),
            std::slice::from_ref(tangent),
            LinearPrimitiveOperation::custom(op.to_tensor_custom_primitive())?,
            1,
        )?
        .into_iter()
        .next()
        .ok_or(TracingError::InvalidOutputCount { expected: 1, got: 0 })
    }
}

impl<E> WithShardingConstraintJvpValue<E> for ShardMapTracer
where
    E: Engine<
            Type = ArrayType,
            Value = ShardMapTracer,
            LinearOperation = LinearPrimitiveOperation<ArrayType, ShardMapTracer>,
        > + ?Sized,
{
    fn apply_constraint_tangent(
        op: &WithShardingConstraintOperation,
        tangent: &LinearTerm<ArrayType, ShardMapTracer, LinearPrimitiveOperation<ArrayType, ShardMapTracer>>,
    ) -> Result<LinearTerm<ArrayType, ShardMapTracer, LinearPrimitiveOperation<ArrayType, ShardMapTracer>>, TracingError>
    {
        LinearTerm::apply_staged_op(
            tangent.builder.clone(),
            std::slice::from_ref(tangent),
            LinearPrimitiveOperation::Custom(Arc::new(op.to_tracer_linear_custom_primitive())),
            1,
        )?
        .into_iter()
        .next()
        .ok_or(TracingError::InvalidOutputCount { expected: 1, got: 0 })
    }
}

impl<V, E> DifferentiableOperation<E> for WithShardingConstraintOperation
where
    E: Engine<Type = ArrayType, Value = V, LinearOperation = LinearPrimitiveOperation<ArrayType, V>> + ?Sized,
    V: WithShardingConstraintJvpValue<E>,
    V: Differentiable<
            ArrayType,
            Tangent<LinearPrimitiveOperation<ArrayType, V>> = LinearTerm<
                ArrayType,
                V,
                LinearPrimitiveOperation<ArrayType, V>,
            >,
        >,
{
    fn jvp(
        &self,
        _engine: &E,
        inputs: &[JvpTracer<V, EngineTangent<E>>],
    ) -> Result<Vec<JvpTracer<V, EngineTangent<E>>>, TracingError> {
        check_input_count!(inputs, 1);
        let tangent = V::apply_constraint_tangent(self, &inputs[0].tangent)?;
        Ok(vec![JvpTracer { primal: inputs[0].primal.clone(), tangent }])
    }
}

impl InterpretableOperation<ArrayType, Linearized<ShardMapTracer>> for WithShardingConstraintOperation {
    fn interpret(
        &self,
        inputs: &[Linearized<ShardMapTracer>],
    ) -> Result<Vec<Linearized<ShardMapTracer>>, TracingError> {
        check_input_count!(inputs, 1);
        let input = &inputs[0];
        let primal = Tracer::apply_staged_op(
            input.primal.engine,
            input.primal.builder.clone(),
            std::slice::from_ref(&input.primal),
            XlaPrimitiveOperation::WithShardingConstraint(self.clone()),
        )?
        .into_iter()
        .next()
        .expect("sharding constraint should produce one primal output");
        let tangent = LinearTerm::apply_staged_op(
            input.tangent.builder.clone(),
            std::slice::from_ref(&input.tangent),
            LinearPrimitiveOperation::Custom(Arc::new(self.to_tracer_linear_custom_primitive())),
            1,
        )?
        .into_iter()
        .next()
        .expect("sharding constraint should produce one tangent output");
        Ok(vec![Linearized { primal, tangent }])
    }
}

impl InterpretableOperation<ArrayType, ShardMapTracer> for WithShardingConstraintOperation {
    fn interpret(&self, inputs: &[ShardMapTracer]) -> Result<Vec<ShardMapTracer>, TracingError> {
        check_input_count!(inputs, 1);
        Ok(vec![inputs[0].clone()])
    }
}

impl LinearOperation<ArrayType, ShardMapTracer> for WithShardingConstraintOperation {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, ShardMapTracer>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, ShardMapTracer>>>, TracingError> {
        check_input_count!(output_cotangents, 1);
        let contribution = LinearTerm::apply_staged_op(
            output_cotangents[0].builder.clone(),
            std::slice::from_ref(&output_cotangents[0]),
            LinearPrimitiveOperation::Custom(Arc::new(self.to_tracer_linear_custom_primitive())),
            1,
        )?
        .into_iter()
        .next()
        .expect("sharding constraint should produce one cotangent contribution");
        Ok(vec![Some(contribution)])
    }
}

impl StableHloCustomLowering<ShardMapTensor> for WithShardingConstraintOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        _op: &CustomPrimitive<ArrayType, ShardMapTensor>,
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
    use ryft_core::tracing::ProgramBuilder;
    use ryft_core::tracing_v2::{LinearOperation, LinearPrimitiveOperation, LinearTerm};
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
        let op = WithShardingConstraintOperation::new(sharding.clone());

        assert_eq!(
            <WithShardingConstraintOperation as Operation>::infer_output_types(
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
        let op = WithShardingConstraintOperation::new(target_sharding);

        assert_eq!(
            <WithShardingConstraintOperation as Operation>::infer_output_types(
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
        let op = WithShardingConstraintOperation::new(target_sharding);

        assert_eq!(
            <WithShardingConstraintOperation as Operation>::infer_output_types(
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
        let op = WithShardingConstraintOperation::new(test_sharding(&mesh));

        assert_eq!(
            <WithShardingConstraintOperation as Operation>::infer_output_types(
                &op,
                &[ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8), Size::Static(4)]), None, None)
                    .unwrap()],
            ),
            Err(TypeError {
                message: "with_sharding_constraint rank does not match the requested sharding rank".to_string()
            })
        );
    }

    #[test]
    fn test_with_sharding_constraint_transpose_preserves_the_constraint() {
        let mesh = test_mesh();
        let sharding = test_sharding(&mesh);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, None).unwrap();

        let transpose_builder = Rc::new(RefCell::new(ProgramBuilder::<
            ArrayType,
            ShardMapTensor,
            LinearPrimitiveOperation<ArrayType, ShardMapTensor>,
        >::new()));
        let output_cotangent_atom = transpose_builder.borrow_mut().add_input(input_type.clone());
        let output_cotangent = LinearTerm::from_staged_parts(output_cotangent_atom, transpose_builder.clone());
        let contribution =
            LinearOperation::transpose(&WithShardingConstraintOperation::new(sharding.clone()), &[output_cotangent])
                .unwrap()
                .into_iter()
                .next()
                .expect("transpose should return one contribution")
                .expect("transpose should produce one cotangent contribution");
        let contribution_atom = contribution.atom;
        drop(contribution);

        let transpose_builder = Rc::try_unwrap(transpose_builder)
            .expect("transpose builder should not have outstanding linear terms")
            .into_inner();
        let transpose_program = transpose_builder.build::<ShardMapTensor, ShardMapTensor>(
            vec![contribution_atom],
            Placeholder,
            Placeholder,
        );
        assert_eq!(
            transpose_program.to_string(),
            format!("lambda %0:f32[8] .\nlet %1:f32[8][sharding={sharding}] = with_sharding_constraint %0\nin (%1)")
                .trim_end(),
        );
    }

    #[test]
    fn test_with_sharding_constraint_traced_transpose_preserves_the_constraint() {
        let mesh = test_mesh();
        let sharding = test_sharding(&mesh);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, None).unwrap();

        let transpose_builder = Rc::new(RefCell::new(ProgramBuilder::<
            ArrayType,
            ShardMapTracer,
            LinearPrimitiveOperation<ArrayType, ShardMapTracer>,
        >::new()));
        let output_cotangent_atom = transpose_builder.borrow_mut().add_input(input_type.clone());
        let output_cotangent = LinearTerm::from_staged_parts(output_cotangent_atom, transpose_builder.clone());
        let contribution =
            LinearOperation::transpose(&WithShardingConstraintOperation::new(sharding.clone()), &[output_cotangent])
                .unwrap()
                .into_iter()
                .next()
                .expect("transpose should return one contribution")
                .expect("transpose should produce one cotangent contribution");
        let contribution_atom = contribution.atom;
        drop(contribution);

        let transpose_builder = Rc::try_unwrap(transpose_builder)
            .expect("transpose builder should not have outstanding linear terms")
            .into_inner();
        let transpose_program = transpose_builder.build::<ShardMapTracer, ShardMapTracer>(
            vec![contribution_atom],
            Placeholder,
            Placeholder,
        );
        assert_eq!(
            transpose_program.to_string(),
            format!("lambda %0:f32[8] .\nlet %1:f32[8][sharding={sharding}] = with_sharding_constraint %0\nin (%1)")
                .trim_end(),
        );
    }
}
