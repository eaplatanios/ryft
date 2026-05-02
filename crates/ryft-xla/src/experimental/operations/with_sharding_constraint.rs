use std::fmt::{Debug, Display};
use std::sync::Arc;

use ryft_core::macros::check_input_count;
use ryft_core::operations::{InterpretableOperation, Operation};
use ryft_core::sharding::Sharding;
use ryft_core::tracing::transposition::LinearOperation;
use ryft_core::tracing::{AtomId, Traceable, TracingError};
use ryft_core::tracing_v2::differentiation::JvpTracer;
use ryft_core::tracing_v2::{
    CustomPrimitive, DifferentiableEngine, DifferentiableOperation, JvpContext, LinearArrayOperation,
    LinearCustomPrimitive,
};
use ryft_core::types::{ArrayType, TypeError};
use ryft_mlir::{Block, Operation as MlirOperation, Value};

use crate::experimental::lowering::{
    LoweringError, ShardMapMlirLowerer, StableHloCustomLowering, StableHloCustomLoweringExtension,
};
use crate::experimental::operations::shard_map::ShardMapCustomReplayExtension;
use crate::experimental::shard_map::{ShardMapTensor, ShardMapTracer};
use crate::mlir::ToMlir;

/// Unary primitive that constrains one traced XLA value to a requested sharding.
#[derive(Clone)]
pub struct WithShardingConstraintOperation {
    /// Requested sharding that the input leaf must satisfy after lowering.
    pub sharding: Sharding,
}

impl WithShardingConstraintOperation {
    /// Creates one sharding-constraint op with the provided target sharding.
    #[inline]
    pub fn new(sharding: Sharding) -> Self {
        Self { sharding }
    }

    fn base_custom_primitive<V: Traceable<ArrayType> + 'static>(&self) -> CustomPrimitive<ArrayType, V>
    where
        Self: Clone
            + InterpretableOperation<ArrayType, V>
            + LinearOperation<ArrayType, V, LinearArrayOperation<V>>
            + Send
            + Sync
            + 'static,
    {
        CustomPrimitive::new(self.clone()).with_transpose_rule(self.clone())
    }

    /// Returns the tensor-leaf custom primitive registration for this op.
    pub(crate) fn to_tensor_custom_primitive(&self) -> CustomPrimitive<ArrayType, ShardMapTensor> {
        self.base_custom_primitive::<ShardMapTensor>()
            .with_jvp_rule_for::<crate::experimental::engines::XlaEngine<'static>, _>(self.clone())
            .with_extension(self.clone())
            .with_extension(ShardMapCustomReplayExtension::new(|_, inputs| Ok(vec![inputs[0].clone()])))
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

impl Operation<ArrayType> for WithShardingConstraintOperation {
    fn name(&self) -> &'static str {
        "with_sharding_constraint"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_input_count!(input_types, 1, TypeError);
        let mut output = input_types[0].clone();
        if output.rank() != self.sharding.rank() {
            return Err(TypeError {
                message: "with_sharding_constraint rank does not match the requested sharding rank".to_string(),
            });
        }
        let mut sharding = self.sharding.clone();
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
        check_input_count!(inputs, 1, TracingError);
        Ok(vec![inputs[0].clone()])
    }
}

impl LinearOperation<ArrayType, ShardMapTensor, LinearArrayOperation<ShardMapTensor>>
    for WithShardingConstraintOperation
{
    fn transpose(
        &self,
        context: &mut ryft_core::tracing::transposition::TranspositionContext<
            ArrayType,
            ShardMapTensor,
            LinearArrayOperation<ShardMapTensor>,
        >,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError> {
        check_input_count!(output_cotangents, 1, TracingError);
        match output_cotangents[0] {
            Some(atom) => {
                let contribution = context
                    .stage(LinearArrayOperation::custom(self.to_tensor_custom_primitive())?, &[atom])?
                    .into_iter()
                    .next()
                    .expect("sharding constraint should produce one cotangent contribution");
                Ok(vec![Some(contribution)])
            }
            None => Ok(vec![None]),
        }
    }
}

impl<E> DifferentiableOperation<E> for WithShardingConstraintOperation
where
    E: DifferentiableEngine<
            Type = ArrayType,
            Value = ShardMapTensor,
            LinearOperationCarrier = LinearArrayOperation<ShardMapTensor>,
        > + ?Sized,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<ShardMapTensor, AtomId>],
    ) -> Result<Vec<JvpTracer<ShardMapTensor, AtomId>>, TracingError> {
        check_input_count!(inputs, 1, TracingError);
        let tangent = context
            .apply_operation(&[inputs[0].tangent], LinearArrayOperation::custom(self.to_tensor_custom_primitive())?, 1)?
            .into_iter()
            .next()
            .expect("with_sharding_constraint jvp should produce one tangent");
        Ok(vec![JvpTracer { primal: inputs[0].primal.clone(), tangent }])
    }
}

impl InterpretableOperation<ArrayType, ShardMapTracer> for WithShardingConstraintOperation {
    fn interpret(&self, inputs: &[ShardMapTracer]) -> Result<Vec<ShardMapTracer>, TracingError> {
        check_input_count!(inputs, 1, TracingError);
        Ok(vec![inputs[0].clone()])
    }
}

impl LinearOperation<ArrayType, ShardMapTracer, LinearArrayOperation<ShardMapTracer>>
    for WithShardingConstraintOperation
{
    fn transpose(
        &self,
        context: &mut ryft_core::tracing::transposition::TranspositionContext<
            ArrayType,
            ShardMapTracer,
            LinearArrayOperation<ShardMapTracer>,
        >,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError> {
        check_input_count!(output_cotangents, 1, TracingError);
        match output_cotangents[0] {
            Some(atom) => {
                let contribution = context
                    .stage(LinearArrayOperation::Custom(Arc::new(self.to_tracer_linear_custom_primitive())), &[atom])?
                    .into_iter()
                    .next()
                    .expect("sharding constraint should produce one cotangent contribution");
                Ok(vec![Some(contribution)])
            }
            None => Ok(vec![None]),
        }
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
            self.sharding.to_mlir(lowerer.location),
            lowerer.location,
        ));
        Ok(vec![operation.result(0).expect("sdy.sharding_constraint should return one result").as_ref()])
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use pretty_assertions::assert_eq;

    use ryft_core::operations::Operation;
    use ryft_core::parameters::Placeholder;
    use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use ryft_core::tracing::transposition::{LinearOperation, TranspositionContext};
    use ryft_core::tracing::{ProgramBuilder, Traceable};
    use ryft_core::tracing_v2::LinearArrayOperation;
    use ryft_core::types::{ArrayType, DataType, Shape, Size};

    use super::*;

    fn test_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap()
    }

    fn test_sharding(mesh: &LogicalMesh) -> Sharding {
        Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()
    }

    fn test_transposition_context<V: Traceable<ArrayType>>(
        builder: Rc<RefCell<ProgramBuilder<ArrayType, V, LinearArrayOperation<V>>>>,
    ) -> TranspositionContext<ArrayType, V, LinearArrayOperation<V>> {
        TranspositionContext::new(builder)
    }

    #[test]
    fn test_with_sharding_constraint_abstract_eval_attaches_sharding() {
        let mesh = test_mesh();
        let sharding = test_sharding(&mesh);
        let op = WithShardingConstraintOperation::new(sharding.clone());

        assert_eq!(
            <WithShardingConstraintOperation as Operation<ArrayType>>::infer_output_types(
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
            <WithShardingConstraintOperation as Operation<ArrayType>>::infer_output_types(
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
            <WithShardingConstraintOperation as Operation<ArrayType>>::infer_output_types(
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
            <WithShardingConstraintOperation as Operation<ArrayType>>::infer_output_types(
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

        let transpose_builder =
            Rc::new(RefCell::new(
                ProgramBuilder::<ArrayType, ShardMapTensor, LinearArrayOperation<ShardMapTensor>>::new(),
            ));
        let output_cotangent_atom = transpose_builder.borrow_mut().add_input(input_type.clone());
        let mut context = test_transposition_context(transpose_builder.clone());
        let contribution_atom = LinearOperation::transpose(
            &WithShardingConstraintOperation::new(sharding.clone()),
            &mut context,
            &[Some(output_cotangent_atom)],
        )
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
            .build::<ShardMapTensor, ShardMapTensor>(vec![contribution_atom], Placeholder, Placeholder)
            .unwrap();
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

        let transpose_builder =
            Rc::new(RefCell::new(
                ProgramBuilder::<ArrayType, ShardMapTracer, LinearArrayOperation<ShardMapTracer>>::new(),
            ));
        let output_cotangent_atom = transpose_builder.borrow_mut().add_input(input_type.clone());
        let mut context = test_transposition_context(transpose_builder.clone());
        let contribution_atom = LinearOperation::transpose(
            &WithShardingConstraintOperation::new(sharding.clone()),
            &mut context,
            &[Some(output_cotangent_atom)],
        )
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
            .build::<ShardMapTracer, ShardMapTracer>(vec![contribution_atom], Placeholder, Placeholder)
            .unwrap();
        assert_eq!(
            transpose_program.to_string(),
            format!("lambda %0:f32[8] .\nlet %1:f32[8][sharding={sharding}] = with_sharding_constraint %0\nin (%1)")
                .trim_end(),
        );
    }
}
