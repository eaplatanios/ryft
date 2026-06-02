use ryft_core::differentiation::{Cotangent, TransposableOperation};
use ryft_core::macros::check_count;
use ryft_core::operations::{InterpretableOperation, Operation};
use ryft_core::sharding::Sharding;
use ryft_core::tracing::{Context, ProgramTracingContext, Traceable, TracingError};
use ryft_core::types::{ArrayType, TypeError};
use ryft_mlir::{Block, Operation as MlirOperation, Value, ValueRef};
use std::fmt::{Debug, Display};

use crate::experimental::lowering::{LoweringError, ShardMapMlirLowerer};
use crate::experimental::ops::{LinearXlaOperation, LinearXlaOperationExtension};
use crate::mlir::ToMlir;

/// Unary primitive that constrains one traced XLA value to a requested sharding.
#[derive(Clone, Debug)]
pub struct WithShardingConstraintOperation {
    /// Requested sharding that the input leaf must satisfy after lowering.
    sharding: Sharding,
}

impl WithShardingConstraintOperation {
    /// Creates one sharding-constraint op with the provided target sharding.
    #[inline]
    pub(crate) fn new(sharding: Sharding) -> Self {
        Self { sharding }
    }

    /// Returns the requested sharding constraint.
    #[inline]
    pub(crate) fn sharding(&self) -> &Sharding {
        &self.sharding
    }

    /// Lowers this sharding constraint to the corresponding Shardy operation.
    pub(crate) fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("input", input_values, 1, TracingError);
        let location = lowerer.location();
        let sharding = self.sharding.to_mlir(location)?;
        let operation = lowerer.block_mut().append_operation(ryft_mlir::dialects::shardy::sharding_constraint(
            input_values[0],
            sharding,
            location,
        )?)?;
        Ok(vec![operation.result(0).expect("sdy.sharding_constraint should return one result").as_ref()])
    }
}

impl Display for WithShardingConstraintOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl Operation<ArrayType> for WithShardingConstraintOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "with_sharding_constraint"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        let output = &input_types[0];
        if output.rank() != self.sharding.rank() {
            return Err(TypeError {
                message: ("with_sharding_constraint rank does not match the requested sharding rank").into(),
            });
        }
        let varying_manual_axes = output
            .sharding()
            .map(|input_sharding| input_sharding.varying_manual_axes().clone())
            .unwrap_or_default();
        let sharding = Sharding::with_manual_axes(
            self.sharding.mesh().clone(),
            self.sharding.dimensions().to_vec(),
            self.sharding.unreduced_axes().clone(),
            self.sharding.reduced_manual_axes().clone(),
            varying_manual_axes,
        )
        .map_err(|error| TypeError { message: error.to_string() })?;
        let output =
            ArrayType::new(output.data_type(), output.shape().clone(), output.layout().cloned(), Some(sharding))
                .map_err(|error| TypeError { message: error.to_string() })?;
        Ok(vec![output])
    }
}

impl<V: Traceable<ArrayType>> InterpretableOperation<ArrayType, V> for WithShardingConstraintOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![inputs[0].clone()])
    }
}

impl<V: Traceable<ArrayType>> TransposableOperation<ArrayType, V, LinearXlaOperation<V>>
    for WithShardingConstraintOperation
{
    fn transpose<'transpose>(
        &self,
        context: &mut ProgramTracingContext<'transpose, ArrayType, V, LinearXlaOperation<V>>,
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, LinearXlaOperation<V>>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, LinearXlaOperation<V>>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        match &output_cotangents[0] {
            Cotangent::Staged(cotangent) => {
                let cotangent_refs = [cotangent];
                let mut contribution_outputs = context.stage_operation(
                    LinearXlaOperation::Extension(LinearXlaOperationExtension::WithShardingConstraint(self.clone())),
                    cotangent_refs.as_slice(),
                )?;
                check_count!("output", contribution_outputs, 1, TracingError);
                Ok(vec![Cotangent::Staged(contribution_outputs.remove(0))])
            }
            Cotangent::Zero => Ok(vec![Cotangent::Zero]),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use pretty_assertions::assert_eq;

    use ryft_core::differentiation::{Cotangent, TransposableOperation};
    use ryft_core::operations::Operation;
    use ryft_core::parameters::Placeholder;
    use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use ryft_core::tracing::domains::ProgramTracingDomain;
    use ryft_core::tracing::{Context, ProgramBuilder, ProgramTracingContext, Traceable};
    use ryft_core::types::{ArrayType, DataType, Shape, Size};

    use crate::experimental::shard_map::ShardMapTracer;

    use super::*;

    fn test_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap()
    }

    fn test_sharding(mesh: &LogicalMesh) -> Sharding {
        Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()
    }

    fn test_transposition_context<'transpose, V: Traceable<ArrayType>>(
        domain: &'transpose ProgramTracingDomain<ArrayType, V, LinearXlaOperation<V>>,
        builder: Rc<RefCell<ProgramBuilder<ArrayType, V, LinearXlaOperation<V>>>>,
    ) -> ProgramTracingContext<'transpose, ArrayType, V, LinearXlaOperation<V>> {
        ProgramTracingContext::new(domain, builder)
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
                            input_sharding.mesh().clone(),
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
                message: ("with_sharding_constraint rank does not match the requested sharding rank").into()
            })
        );
    }

    #[test]
    fn test_with_sharding_constraint_transpose_preserves_the_constraint() {
        let mesh = test_mesh();
        let sharding = test_sharding(&mesh);
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]), None, None).unwrap();

        let transpose_builder =
            Rc::new(RefCell::new(ProgramBuilder::<ArrayType, ArrayType, LinearXlaOperation<ArrayType>>::new()));
        let output_cotangent_atom = transpose_builder.borrow_mut().add_input(input_type.clone());
        let domain = ProgramTracingDomain::new();
        let mut context = test_transposition_context(&domain, transpose_builder.clone());
        let output_cotangent = context.tracer(output_cotangent_atom, None);
        let contribution = TransposableOperation::transpose(
            &WithShardingConstraintOperation::new(sharding.clone()),
            &mut context,
            &[Cotangent::Staged(output_cotangent)],
        )
        .unwrap()
        .into_iter()
        .next()
        .expect("transpose should return one contribution");
        let Cotangent::Staged(contribution) = contribution else {
            panic!("transpose should produce one cotangent contribution");
        };
        let contribution_atom = contribution.atom_id().unwrap();
        drop(contribution);
        drop(context);

        let transpose_builder = Rc::try_unwrap(transpose_builder)
            .expect("transpose builder should not have outstanding linear terms")
            .into_inner();
        let transpose_program = transpose_builder
            .build::<ArrayType, ArrayType>(vec![contribution_atom], Placeholder, Placeholder)
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
                ProgramBuilder::<ArrayType, ShardMapTracer, LinearXlaOperation<ShardMapTracer>>::new(),
            ));
        let output_cotangent_atom = transpose_builder.borrow_mut().add_input(input_type.clone());
        let domain = ProgramTracingDomain::new();
        let mut context = test_transposition_context(&domain, transpose_builder.clone());
        let output_cotangent = context.tracer(output_cotangent_atom, None);
        let contribution = TransposableOperation::transpose(
            &WithShardingConstraintOperation::new(sharding.clone()),
            &mut context,
            &[Cotangent::Staged(output_cotangent)],
        )
        .unwrap()
        .into_iter()
        .next()
        .expect("transpose should return one contribution");
        let Cotangent::Staged(contribution) = contribution else {
            panic!("transpose should produce one cotangent contribution");
        };
        let contribution_atom = contribution.atom_id().unwrap();
        drop(contribution);
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
