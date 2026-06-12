use ryft_core::contexts::StagingContext;
use ryft_core::differentiation::{Cotangent, TransposableOperation};
use ryft_core::macros::check_count;
use ryft_core::operations::{InterpretableOperation, Operation};
use ryft_core::programs::{ProgramError, Value};
use ryft_core::sharding::Sharding;
use ryft_core::tracing::{AbstractTracingContext, Tracer};
use ryft_core::types::{ArrayType, TypeError};
use ryft_mlir::{Block, Operation as MlirOperation, Value as MlirValue, ValueRef};
use std::fmt::{Debug, Display};

use crate::experimental::lowering::{LoweringError, ShardMapMlirLowerer};
use crate::experimental::ops::{LinearXlaOperation, LinearXlaOperationExtension, XlaConstant};
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
        check_count!("input", input_values, 1, ProgramError);
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
            self.sharding.reduced_axes().clone(),
            varying_manual_axes,
        )
        .map_err(|error| TypeError { message: error.to_string() })?;
        let output = ArrayType::new(output.data_type(), output.shape().clone())
            .with_layout(output.layout().cloned())
            .with_memory(output.memory())
            .with_sharding(sharding)
            .map_err(|error| TypeError { message: error.to_string() })?;
        Ok(vec![output])
    }
}

/// Value-level sharding-constraint capability.
///
/// [`ConstrainSharding`] is the receiver-style entry point for staging or executing
/// [`WithShardingConstraintOperation`]. The provided default returns the value unchanged, which is correct for
/// concrete (single-device) values, for which a sharding constraint only describes distribution metadata. Staging
/// values override it to stage the constraint, which keeps transforms that apply operations through interpretation
/// (e.g., program batching and re-tracing) from silently dropping it.
pub trait ConstrainSharding: Sized {
    /// Returns this value constrained to the provided [`Sharding`].
    fn constrain_sharding(self, sharding: &Sharding) -> Self {
        let _ = sharding;
        self
    }
}

/// Trait for operation types that include or can wrap [`WithShardingConstraintOperation`]. Backend-owned closed
/// operation enums implement this trait so that generic transform code can stage the constraint without knowing the
/// concrete operation enum.
#[doc(hidden)]
pub trait SupportsWithShardingConstraint {
    /// Constructs the backend-specific representation of the sharding-constraint operation.
    fn with_sharding_constraint_operation(operation: WithShardingConstraintOperation) -> Self;
}

impl<C> ConstrainSharding for Tracer<C>
where
    C: StagingContext<Type = ArrayType>,
    C::Operation: SupportsWithShardingConstraint,
{
    fn constrain_sharding(self, sharding: &Sharding) -> Self {
        self.unary(C::Operation::with_sharding_constraint_operation(WithShardingConstraintOperation::new(
            sharding.clone(),
        )))
    }
}

impl<V: Value<ArrayType> + ConstrainSharding> InterpretableOperation<ArrayType, V> for WithShardingConstraintOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        // The constraint flows through the capability method so that interpretation over staging values (e.g.,
        // during program batching or re-tracing) preserves it; concrete values pass through unchanged.
        Ok(vec![inputs[0].clone().constrain_sharding(&self.sharding)])
    }
}

impl<V: Value<ArrayType>, Factor: Value<ArrayType>>
    TransposableOperation<ArrayType, V, LinearXlaOperation<V, XlaConstant, Factor>>
    for WithShardingConstraintOperation
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<'transpose, ArrayType, V, LinearXlaOperation<V, XlaConstant, Factor>>,
        input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, LinearXlaOperation<V, XlaConstant, Factor>>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, LinearXlaOperation<V, XlaConstant, Factor>>>, ProgramError>
    {
        check_count!("input", input_types, 1, ProgramError);
        check_count!("output", output_cotangents, 1, ProgramError);
        match &output_cotangents[0] {
            Cotangent::Staged(cotangent) => {
                // Mirroring JAX's `reshard` transpose, the cotangent is constrained to the cotangent dual of the
                // *input*'s sharding (swapping unreduced and reduced axes) rather than to this operation's target
                // sharding: the produced value is the input's cotangent and must be distributed like it. An input
                // that carries no sharding receives its cotangent unconstrained.
                let Some(input_sharding) = input_types[0].sharding() else {
                    return Ok(vec![Cotangent::Staged(cotangent.clone())]);
                };
                let adjoint_operation = WithShardingConstraintOperation::new(input_sharding.cotangent_dual());
                let cotangent_refs = [cotangent];
                let mut contribution_outputs = context.stage_operation(
                    LinearXlaOperation::Extension(LinearXlaOperationExtension::WithShardingConstraint(
                        adjoint_operation,
                    )),
                    cotangent_refs.as_slice(),
                )?;
                check_count!("output", contribution_outputs, 1, ProgramError);
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

    use ryft_core::contexts::StagingContext;
    use ryft_core::differentiation::{Cotangent, TransposableOperation};
    use ryft_core::domains::AbstractDomain;
    use ryft_core::operations::Operation;
    use ryft_core::parameters::Placeholder;
    use ryft_core::programs::{ProgramBuilder, Value};
    use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use ryft_core::tracing::AbstractTracingContext;
    use ryft_core::types::{ArrayType, DataType, Memory, Shape, Size};

    use crate::experimental::shard_map::ShardMapTracer;

    use super::*;

    fn test_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap()
    }

    fn test_sharding(mesh: &LogicalMesh) -> Sharding {
        Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()
    }

    fn test_transposition_context<'transpose, V: Value<ArrayType>>(
        domain: &'transpose AbstractDomain<ArrayType, V, LinearXlaOperation<V, XlaConstant>>,
        builder: Rc<RefCell<ProgramBuilder<ArrayType, V, LinearXlaOperation<V, XlaConstant>>>>,
    ) -> AbstractTracingContext<'transpose, ArrayType, V, LinearXlaOperation<V, XlaConstant>> {
        AbstractTracingContext::new(domain, builder)
    }

    #[test]
    fn test_with_sharding_constraint_abstract_eval_attaches_sharding() {
        let mesh = test_mesh();
        let sharding = test_sharding(&mesh);
        let op = WithShardingConstraintOperation::new(sharding.clone());

        assert_eq!(
            <WithShardingConstraintOperation as Operation<ArrayType>>::infer_output_types(
                &op,
                &[ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))],
            ),
            Ok(vec![
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
                    .with_sharding(sharding.clone())
                    .unwrap()
            ])
        );

        // The input memory placement must be preserved through the constraint instead of being reset to the default.
        assert_eq!(
            <WithShardingConstraintOperation as Operation<ArrayType>>::infer_output_types(
                &op,
                &[ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
                    .with_memory(Memory::Host { pinned: true })],
            ),
            Ok(vec![
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
                    .with_memory(Memory::Host { pinned: true })
                    .with_sharding(sharding)
                    .unwrap()
            ])
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
                &[ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
                    .with_sharding(input_sharding.clone())
                    .unwrap()],
            ),
            Ok(vec![
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
                    .with_sharding(
                        Sharding::with_manual_axes(
                            input_sharding.mesh().clone(),
                            vec![ShardingDimension::sharded(["x"])],
                            Vec::<&str>::new(),
                            Vec::<&str>::new(),
                            ["x"],
                        )
                        .unwrap()
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
                &[ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
                    .with_sharding(input_sharding)
                    .unwrap()],
            ),
            Ok(vec![
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
                    .with_sharding(
                        Sharding::with_manual_axes(
                            mesh,
                            vec![ShardingDimension::sharded(["x"])],
                            Vec::<&str>::new(),
                            Vec::<&str>::new(),
                            ["x"]
                        )
                        .unwrap()
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
                &[ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8), Size::Static(4)]))],
            ),
            Err(TypeError {
                message: ("with_sharding_constraint rank does not match the requested sharding rank").into()
            })
        );
    }

    #[test]
    fn test_with_sharding_constraint_transpose_constrains_to_the_input_sharding_dual() {
        let mesh = test_mesh();
        let target_sharding = test_sharding(&mesh);
        // The input is unreduced along `x`, so its cotangent must be constrained to the dual (reduced along `x`).
        let input_sharding =
            Sharding::with_unreduced_axes(mesh.clone(), vec![ShardingDimension::replicated()], ["x"]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(input_sharding.clone())
            .unwrap();

        let transpose_builder =
            Rc::new(RefCell::new(
                ProgramBuilder::<ArrayType, ArrayType, LinearXlaOperation<ArrayType, XlaConstant>>::new(),
            ));
        let output_cotangent_atom = transpose_builder
            .borrow_mut()
            .add_input(ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)])));
        let domain = AbstractDomain::new();
        let mut context = test_transposition_context(&domain, transpose_builder.clone());
        let output_cotangent = context.tracer(output_cotangent_atom, None);
        let contribution = TransposableOperation::transpose(
            &WithShardingConstraintOperation::new(target_sharding),
            &mut context,
            &[&input_type],
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
        let dual_sharding = input_sharding.cotangent_dual();
        assert_eq!(
            transpose_program.to_string(),
            format!(
                "lambda %0:f32[8] .\nlet %1:f32[8][sharding={dual_sharding}] = with_sharding_constraint %0\nin (%1)"
            )
            .trim_end(),
        );
    }

    #[test]
    fn test_with_sharding_constraint_batching_stages_the_lifted_constraint() {
        use ryft_core::tracing_v2::batching::{ArrayBatch, BatchableOperation};

        use crate::experimental::ops::{XlaOperation, XlaOperationExtension};

        let mesh = test_mesh();
        let sharding = test_sharding(&mesh); // Rank 1, sharded over `x`.
        let operation =
            XlaOperationExtension::WithShardingConstraint(WithShardingConstraintOperation::new(sharding.clone()));

        // Batch the operation over a tracer input, which is how program batching applies lifted operations: the
        // staged batched constraint must carry the lifted sharding (with a replicated entry inserted at the new
        // lane axis) instead of being erased.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, ArrayType, XlaOperation>::new()));
        let input_atom = builder
            .borrow_mut()
            .add_input(ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(8)])));
        let domain = AbstractDomain::new();
        let mut context = AbstractTracingContext::new(&domain, builder.clone());
        let input = ArrayBatch::mapped(context.tracer(input_atom, None), 0).unwrap();
        let outputs = operation.batch(&(), std::slice::from_ref(&input)).unwrap();
        assert_eq!(outputs[0].batch_axis(), Some(0));
        let output_atom = outputs[0].value().atom_id().unwrap();
        drop(input);
        drop(outputs);
        drop(context);

        let builder = Rc::try_unwrap(builder).expect("batching should not hold on to the builder").into_inner();
        let program = builder
            .build::<Vec<ArrayType>, Vec<ArrayType>>(vec![output_atom], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let lifted_sharding = sharding.inserting_dimension(0, ShardingDimension::Replicated).unwrap();
        assert_eq!(
            program.to_string(),
            format!(
                "lambda %0:f32[2, 8] .\nlet %1:f32[2, 8][sharding={lifted_sharding}] = with_sharding_constraint \
                 %0\nin (%1)"
            )
            .trim_end(),
        );
    }

    #[test]
    fn test_with_sharding_constraint_transpose_passes_through_unsharded_inputs() {
        let mesh = test_mesh();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]));

        let transpose_builder =
            Rc::new(RefCell::new(
                ProgramBuilder::<ArrayType, ArrayType, LinearXlaOperation<ArrayType, XlaConstant>>::new(),
            ));
        let output_cotangent_atom = transpose_builder.borrow_mut().add_input(input_type.clone());
        let domain = AbstractDomain::new();
        let mut context = test_transposition_context(&domain, transpose_builder.clone());
        let output_cotangent = context.tracer(output_cotangent_atom, None);
        let contribution = TransposableOperation::transpose(
            &WithShardingConstraintOperation::new(test_sharding(&mesh)),
            &mut context,
            &[&input_type],
            &[Cotangent::Staged(output_cotangent)],
        )
        .unwrap()
        .into_iter()
        .next()
        .expect("transpose should return one contribution");
        let Cotangent::Staged(contribution) = contribution else {
            panic!("transpose should produce one cotangent contribution");
        };
        // The input carries no sharding, so the cotangent flows back unconstrained (no operation is staged).
        assert_eq!(contribution.atom_id().unwrap(), output_cotangent_atom);
    }

    #[test]
    fn test_with_sharding_constraint_traced_transpose_constrains_to_the_input_sharding_dual() {
        let mesh = test_mesh();
        let target_sharding = test_sharding(&mesh);
        let input_sharding =
            Sharding::with_unreduced_axes(mesh.clone(), vec![ShardingDimension::replicated()], ["x"]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(input_sharding.clone())
            .unwrap();

        let transpose_builder = Rc::new(RefCell::new(ProgramBuilder::<
            ArrayType,
            ShardMapTracer,
            LinearXlaOperation<ShardMapTracer, XlaConstant>,
        >::new()));
        let output_cotangent_atom = transpose_builder
            .borrow_mut()
            .add_input(ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)])));
        let domain = AbstractDomain::new();
        let mut context = test_transposition_context(&domain, transpose_builder.clone());
        let output_cotangent = context.tracer(output_cotangent_atom, None);
        let contribution = TransposableOperation::transpose(
            &WithShardingConstraintOperation::new(target_sharding),
            &mut context,
            &[&input_type],
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
        let dual_sharding = input_sharding.cotangent_dual();
        assert_eq!(
            transpose_program.to_string(),
            format!(
                "lambda %0:f32[8] .\nlet %1:f32[8][sharding={dual_sharding}] = with_sharding_constraint %0\nin (%1)"
            )
            .trim_end(),
        );
    }
}
