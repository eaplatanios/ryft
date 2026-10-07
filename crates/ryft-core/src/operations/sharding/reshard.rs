use std::fmt::Display;

use ryft_macros::capability;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrType, ArrayType, MeshAxisType, Sharding,
    ShardingDimension,
};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{DifferentiableType, DifferentiationDual};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::Capability;
use crate::operations::manipulation::broadcasting::{Broadcast, BroadcastOperation};
use crate::operations::sharding::constrain_sharding::CONSTRAIN_SHARDING_OPERATION_NAME;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, ProgramError, RegionInterface, TypeError, Typed, Value,
    ValueDomainDispatch,
};

#[cfg(doc)]
use crate::operations::sharding::constrain_sharding::ConstrainShardingOperation;

/// Canonical operation name for [`ReshardOperation`].
pub const RESHARD_OPERATION_NAME: &str = "reshard";

/// [`Operation`] that reshards its input to a target [`Sharding`]. This operation is the Ryft analogue of JAX's
/// [`jax.sharding.reshard`](https://docs.jax.dev/en/latest/jax.sharding.html). The array's elements, shape, and data
/// type are unchanged, and the output type carries the target sharding in place of the input's, extended with the
/// manual variation and reduction facts of its input. The target may only name [`MeshAxisType::Explicit`] mesh axes.
/// Placement over [`MeshAxisType::Auto`] axes belongs to backend compilers and can be constrained using
/// [`ConstrainShardingOperation`]s while transitions over [`MeshAxisType::Manual`] axes require their corresponding
/// collectives. Refer to the documentation of [`Reshard`] for more information.
///
/// Interpretation passes the value through and records the target on its type. Batching inserts the mapped axis's
/// sharding into the target at the new batch axis. Differentiation reshards the tangent to the same target, and
/// transposition reshards the cotangent to the dual of the input's sharding, so that the input cotangent is
/// distributed like the input.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReshardOperation {
    /// Target [`Sharding`] that the input is resharded to.
    sharding: Sharding,
}

impl ReshardOperation {
    /// Creates a new [`ReshardOperation`] that reshards its input to `sharding`.
    #[inline]
    pub fn new(sharding: Sharding) -> Self {
        Self { sharding }
    }

    /// Returns the target [`Sharding`] that the input is resharded to.
    #[inline]
    pub fn sharding(&self) -> &Sharding {
        &self.sharding
    }
}

impl Display for ReshardOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ReshardOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        RESHARD_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let input = &input_types[0];
        if input.rank() != self.sharding.rank() {
            return Err(TypeError::invalid(format!(
                "`{}` target sharding rank ({}) does not match the input rank ({})",
                RESHARD_OPERATION_NAME,
                self.sharding.rank(),
                input.rank(),
            )));
        }

        // Every mesh axis the target references, whether it shards a dimension or carries reduction state,
        // must be one the type system governs.
        let sharded_axes = self.sharding.dimensions().iter().flat_map(|dimension| match dimension {
            ShardingDimension::Sharded(axis_names) => axis_names.as_slice(),
            ShardingDimension::Replicated | ShardingDimension::Unconstrained => &[],
        });
        let reduction_axes = self.sharding.unreduced_axes().iter().chain(self.sharding.reduced_axes());
        if sharded_axes
            .chain(reduction_axes)
            .any(|axis| self.sharding.mesh().axis_type(axis) == Some(MeshAxisType::Auto))
        {
            return Err(TypeError::invalid(format!(
                "`{RESHARD_OPERATION_NAME}` cannot target auto mesh axes; use \
                 `{CONSTRAIN_SHARDING_OPERATION_NAME}` to constrain placement over them",
            )));
        }

        if self
            .sharding
            .dimensions()
            .iter()
            .flat_map(|dimension| match dimension {
                ShardingDimension::Sharded(axes) => axes.as_slice(),
                _ => &[],
            })
            .chain(self.sharding.unreduced_axes())
            .chain(self.sharding.reduced_axes())
            .any(|axis| self.sharding.mesh().axis_type(axis) == Some(MeshAxisType::Manual))
        {
            return Err(TypeError::invalid(format!(
                "`{RESHARD_OPERATION_NAME}` cannot target manual mesh axes; use manual collectives for \
                 transitions over them",
            )));
        }

        // Explicit redistribution preserves manual variation and reduction obligations. Their transitions belong
        // to manual collectives, including when a cotangent dual swaps reduced and unreduced state.
        let input_sharding = input.sharding();
        let manual_unreduced = input_sharding
            .into_iter()
            .flat_map(|sharding| {
                sharding
                    .unreduced_axes()
                    .iter()
                    .filter(|axis| sharding.mesh().axis_type(axis) == Some(MeshAxisType::Manual))
            })
            .cloned();
        let manual_reduced = input_sharding
            .into_iter()
            .flat_map(|sharding| {
                sharding
                    .reduced_axes()
                    .iter()
                    .filter(|axis| sharding.mesh().axis_type(axis) == Some(MeshAxisType::Manual))
            })
            .cloned();
        let sharding = self
            .sharding
            .clone()
            .with_unreduced_axes(self.sharding.unreduced_axes().iter().cloned().chain(manual_unreduced))
            .and_then(|sharding| {
                sharding.with_reduced_axes(self.sharding.reduced_axes().iter().cloned().chain(manual_reduced))
            })
            .and_then(|sharding| {
                sharding.with_varying_manual_axes(
                    input_sharding.into_iter().flat_map(|sharding| sharding.varying_manual_axes()).cloned(),
                )
            })
            .map_err(|error| TypeError::invalid(error.to_string()))?;

        Ok(vec![input.clone().with_sharding(sharding).map_err(|error| TypeError::invalid(error.to_string()))?])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("sharding", &self.sharding))
    }
}

impl<C: Domain<Type = ArrayType, Value: Reshard>> InterpretableOperation<C> for ReshardOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // The resharding flows through the capability so interpretation over staging values (e.g., program batching,
        // re-tracing, etc.) preserves it. Concrete values pass through unchanged.
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].reshard(&self.sharding)?])
    }
}

impl<C: Context<Operation: From<ReshardOperation>>> PartiallyEvaluatableOperation<C> for ReshardOperation {}

impl<C: Context<Type = ArrayType, Value: Reshard>, P: ArrayExtentBatchingPolicy<C>>
    BatchableOperation<C, ArrayBatchingPolicy<P>> for ReshardOperation
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        // The lifted reshard's target sharding gains the mapped axis's sharding (derived from the batched inputs via
        // `ArrayBatch::sharding_for_inputs`) at the new batch dimension. Lifting needs only the batch axis's position
        // and placement, never its extent, so mapped axes with dynamic extents lift as well.
        check_count!("input", inputs, 1, ProgramError);
        let lifted_sharding = match inputs[0].batch_axis_position() {
            Some(batch_axis) => self.sharding().batched(batch_axis, ArrayBatch::sharding_for_inputs(inputs)?)?,
            None => self.sharding().clone(),
        };

        // Sharding changes preserve the packed shape, so the batch axis and ragged-axis metadata carry over unchanged.
        let batch_axis = BatchAxis::from_optional_position(inputs[0].batch_axis_position());
        let mut outputs =
            ReshardOperation::new(lifted_sharding).interpret_with_batch_axes(context, inputs, &[batch_axis])?;
        check_count!("output", outputs, 1, ProgramError);
        let output = ArrayBatch::new(outputs.remove(0).into_value(), batch_axis)?
            .with_ragged_axes(inputs[0].ragged_axes().to_vec())?;
        Ok(vec![output].into())
    }
}

impl_differentiable_operation! {
    ReshardOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Value: Reshard,
        C::Operation: From<ReshardOperation>,
    {
        |operation, _context, _driver, inputs| {
            // Resharding is linear, so the tangent is resharded to the same target sharding as the primal.
            check_count!("input", inputs, 1, ProgramError);
            let primal = inputs[0].primal().reshard(operation.sharding())?;
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                MaybeZero::Value(tangent) => MaybeZero::Value(tangent.reshard(operation.sharding())?),
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType> + From<BroadcastOperation> + From<ReshardOperation>,
    {
        |_operation, context, _driver, inputs, outputs, accumulators| {
            // The cotangent of a reshard is itself a reshard of the output cotangent to the cotangent dual of the
            // _input_'s sharding (swapping its unreduced and reduced axes), so the produced input cotangent is
            // distributed like the input. An input that carries no sharding receives an exactly unsharded cotangent
            // through an identity-axis broadcast.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 1, DifferentiationError);
            let input_cotangent_type = inputs[0].r#type().cotangent()?;
            match &outputs[0] {
                MaybeZero::Zero(_) => Ok(()),
                MaybeZero::Value(cotangent) => {
                    let contribution = match input_cotangent_type.sharding() {
                        Some(input_cotangent_sharding) => {
                            // The operation carries only explicit placement; the input cotangent already carries
                            // the preserved manual reduction state, so it must not be requested as a transition.
                            cotangent.reshard(&input_cotangent_sharding.without_manual_reduction_axes())?
                        },
                        None => cotangent.broadcast(
                            input_cotangent_type.clone(),
                            &(0..input_cotangent_type.shape().rank()).collect::<Vec<_>>(),
                        )?,
                    };
                    accumulators[0].accumulate(context, MaybeZero::Value(contribution))?;
                    Ok(())
                }
            }
        }
    },
}

/// Represents the ability to reshard a value to a target [`Sharding`]. [`Reshard`] stages a [`ReshardOperation`],
/// which is an identity function on the array's elements whose output type carries the target sharding. Concrete
/// single-device values record the target on their type and leave their payload untouched, and context-carrying values
/// stage the operation instead, so that transforms that apply operations through interpretation (e.g., program batching
/// and re-tracing) preserve the resharding. Every implementation validates the target exactly like [`ReshardOperation`]
/// type inference does, so that eager and staged evaluation accept the same programs.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and composite array IR values implement it for [`ArrayIrType`].
#[capability(projection(ArrayIrType => ArrayType))]
pub trait Reshard<T = <Self as Capability>::Universe>: Capability + Clone {
    /// Reshards `self` to `sharding`, and returns a [`ProgramError`] if `sharding` is not a valid target for `self` or
    /// the resharding cannot be recorded in the value's context.
    fn reshard(&self, sharding: &Sharding) -> Result<Self, ProgramError>;
}

impl<V: Value<Type = ArrayType, Dispatch = ValueDomainDispatch, Domain: Context<Type = ArrayType, Operation: From<ReshardOperation>>>>
    Reshard<ArrayType> for V
{
    fn reshard(&self, sharding: &Sharding) -> Result<Self, ProgramError> {
        // Any context-carrying value reshards by binding a `ReshardOperation` through its own context. The
        // `From<ReshardOperation>` bound makes this disjoint from the eager value types (whose context operation
        // is `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete
        // implementations.
        let mut outputs = self.domain().bind(
            ReshardOperation::new(sharding.clone()),
            Vec::new(),
            std::slice::from_ref(self),
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl Reshard for Array {
    fn reshard(&self, sharding: &Sharding) -> Result<Self, ProgramError> {
        // An `Array` is a concrete single-device value, so resharding is a no-op on its payload. Its type still records
        // the target, and the output type comes from the `ReshardOperation` type inference rule itself, so that eager
        // evaluation validates the target and carries the input's varying manual axes over exactly like staged programs
        // do.
        let input_type = self.r#type().into_owned();
        let mut output_types = ReshardOperation::new(sharding.clone()).infer_output_types(&[input_type], &[])?;
        check_count!("output", output_types, 1, ProgramError);
        Ok(Self::new_unchecked(output_types.remove(0), self.shared_storage_bytes().clone()))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, DataType, Dimension, DimensionBounds, DimensionType,
        DimensionVariable, DynamicArrayExtentBatchingPolicy, LogicalMesh, MeshAxis, RaggedAxis, Shape, f8e8m0fnu,
    };
    use crate::batching::{BatchAxis, batch};
    use crate::contexts::{EagerContext, ProjectedContext, StagingContext};
    use crate::differentiation::differentiate_at;
    use crate::macros::{check_operation_partial_evaluation, check_operation_type_inference};
    use crate::programs::{EffectClasses, EmptyRegionDriver, ValueProjection};
    use crate::tracing::TracingContext;

    use super::*;

    /// Returns a mesh with one axis of each type: `x` is explicit, `m` is manual, and `a` is auto.
    fn mesh() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap()
    }

    #[test]
    fn test_reshard() {
        let target = Sharding::new(mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let operation = ReshardOperation::new(target.clone());

        // Operation identity, accessors, and rendering.
        assert_eq!(operation.name(), RESHARD_OPERATION_NAME);
        assert_eq!(operation.sharding(), &target);
        assert_eq!(operation.to_string(), format!("{RESHARD_OPERATION_NAME} [sharding={target}]"));
    }

    #[test]
    fn test_reshard_type_inference() {
        let mesh = mesh();
        let target = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [8]);

        // The output keeps the input's shape and data type and adopts the target sharding;
        // an input varying over manual axes carries that variation over to the target.
        let varying_input_type = input_type
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        check_operation_type_inference!(
            operation = ReshardOperation::new(target.clone()),
            cases = [{
                input_types = [input_type.clone()],
                output_types = [input_type.clone().with_sharding(target.clone()).unwrap()],
            }, {
                input_types = [varying_input_type],
                output_types = [input_type
                    .clone()
                    .with_sharding(target.clone().with_varying_manual_axes(["m"]).unwrap())
                    .unwrap()],
            }, {
                input_types = [ArrayType::new_static(DataType::F32, [8, 2])],
                error = format!(
                    "`{RESHARD_OPERATION_NAME}` target sharding rank (1) does not match the input rank (2)",
                ),
            }, {
                input_types = [],
                error = "expected 1 input but got 0",
            }],
        );

        // Placement over auto axes belongs to the compiler, whether the axis shards a dimension
        // or carries reduction state.
        let auto_target = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let auto_error = format!(
            "`{RESHARD_OPERATION_NAME}` cannot target auto mesh axes; use \
             `{CONSTRAIN_SHARDING_OPERATION_NAME}` to constrain placement over them",
        );
        check_operation_type_inference!(
            operation = ReshardOperation::new(auto_target),
            cases = [{
                input_types = [input_type.clone()],
                error = auto_error.clone(),
            }],
        );
        let auto_unreduced_target = Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["a"]).unwrap();
        check_operation_type_inference!(
            operation = ReshardOperation::new(auto_unreduced_target),
            cases = [{
                input_types = [input_type.clone()],
                error = auto_error,
            }],
        );

        // Explicit redistribution preserves manual reduction obligations; targets cannot create or discharge them.
        let unreduced_input_type = input_type
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["m"]).unwrap())
            .unwrap();
        check_operation_type_inference!(
            operation = ReshardOperation::new(target.clone()),
            cases = [{
                input_types = [unreduced_input_type],
                output_types = [input_type.clone().with_sharding(target.clone().with_unreduced_axes(["m"]).unwrap()).unwrap()],
            }],
        );
        check_operation_type_inference!(
            operation = ReshardOperation::new(Sharding::replicated(mesh.clone(), 1).with_reduced_axes(["m"]).unwrap()),
            cases = [{
                input_types = [input_type.clone()],
                error = format!(
                    "`{RESHARD_OPERATION_NAME}` cannot target manual mesh axes; use manual collectives for \
                     transitions over them",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = ReshardOperation::new(Sharding::new(mesh, vec![ShardingDimension::sharded(["m"])]).unwrap()),
            cases = [{
                input_types = [input_type.clone()],
                error = format!(
                    "`{RESHARD_OPERATION_NAME}` cannot target manual mesh axes; use manual collectives for \
                     transitions over them",
                ),
            }],
        );

        // The operation cannot own nested regions.
        assert_eq!(
            ReshardOperation::new(target).infer_output_types(
                std::slice::from_ref(&input_type),
                &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_reshard_interpretation() {
        let target = Sharding::new(mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let operation = ReshardOperation::new(target.clone());
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();

        // Interpretation passes the elements through and records the target on the output type.
        assert_eq!(
            operation.interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input)),
            Ok(vec![
                Array::from_elements(
                    ArrayType::new_static(DataType::F32, [2]).with_sharding(target).unwrap(),
                    &[1.0f32, 2.0],
                )
                .unwrap(),
            ]),
        );
        assert_eq!(
            operation.interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
    }

    #[test]
    fn test_reshard_partial_evaluation() {
        let target = Sharding::new(mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        check_operation_partial_evaluation!(
            operation = ReshardOperation::new(target.clone()),
            inputs = [Array::vector(vec![1.0f32, 2.0]).unwrap()],
            expected = Array::from_elements(
                ArrayType::new_static(DataType::F32, [2]).with_sharding(target).unwrap(),
                &[1.0f32, 2.0],
            )
            .unwrap(),
        );
    }

    #[test]
    fn test_reshard_batching() {
        // The batch item reshards to a rank-1 sharding; batching over an unsharded input inserts a replicated entry
        // at the new batch axis, so the lifted reshard targets a rank-2 sharding.
        let target = Sharding::new(mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let expected_lifted = target.with_inserted_dimension(0, ShardingDimension::Replicated).unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x| {
                let target = target.clone();
                Ok(batch(move |item| item.reshard(&target), x, BatchAxis::new(0), BatchAxis::new(0), None)?)
            },
            ArrayType::new_static(DataType::F64, [2, 3]),
        )
        .unwrap();
        let shardings = program
            .instructions()
            .iter()
            .map(|instruction| match instruction.operation() {
                ArrayOperation::Reshard(operation) => operation.sharding().clone(),
                operation => panic!("unexpected operation `{operation}`"),
            })
            .collect::<Vec<_>>();
        assert_eq!(shardings, vec![expected_lifted]);
    }

    #[test]
    fn test_reshard_batching_ragged() {
        // Resharding preserves the packed geometry, so a bounded ragged axis on the input carries over to the output.
        let target = Sharding::new(mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let expected_lifted = target.with_inserted_dimension(0, ShardingDimension::Replicated).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array>::new(), 2);
        let array =
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[1f32, 2., 3., 4., 5., 6.]).unwrap();
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let extents = Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[1i32, 3]).unwrap();
        let input = ArrayBatch::new(array, BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, extents, variable, vec![0])])
            .unwrap();
        let ragged_axes = input.ragged_axes().to_vec();
        let (outputs, _) = ReshardOperation::new(target.clone())
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis_position(), Some(0));
        assert_eq!(outputs[0].ragged_axes(), ragged_axes.as_slice());
        assert_eq!(outputs[0].r#type().sharding(), Some(&expected_lifted));
    }

    #[test]
    fn test_reshard_batching_dynamic() {
        // Lifting needs only the batch axis's position and placement, so a mapped axis with a dynamic extent lifts.
        let target = Sharding::new(mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let expected_lifted = target.with_inserted_dimension(0, ShardingDimension::Replicated).unwrap();
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
        let extent = trace.input(DimensionType::from(items.clone()).into());
        let shape = Shape::new(vec![Dimension::Dynamic(items), Dimension::Static(3)]);
        let input = trace.input(ArrayType::new(DataType::F32, shape.clone()).into());
        let input = <_ as ValueProjection<ArrayType>>::into_projected(input).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
            ProjectedContext::new(trace),
            extent,
        );
        let input = ArrayBatch::new(input, BatchAxis::new(0)).unwrap();
        let (outputs, _) =
            ReshardOperation::new(target).batch(&context, &EmptyRegionDriver, &[input]).unwrap().into_parts();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis_position(), Some(0));
        assert_eq!(outputs[0].r#type().shape(), &shape);
        assert_eq!(outputs[0].r#type().sharding(), Some(&expected_lifted));
    }

    #[test]
    fn test_reshard_batching_replicated() {
        // An unmapped input keeps its original rank and remains replicated across the batch.
        let target = Sharding::new(mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array>::new(), 2);
        let array = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let input = ArrayBatch::new(array.clone(), BatchAxis::replicated()).unwrap();
        let (outputs, _) = ReshardOperation::new(target.clone())
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis_position(), None);
        assert!(outputs[0].ragged_axes().is_empty());
        assert_eq!(
            outputs[0].value(),
            &Array::from_elements(
                ArrayType::new_static(DataType::F32, [2]).with_sharding(target).unwrap(),
                &[1.0f32, 2.0],
            )
            .unwrap(),
        );
    }

    #[test]
    fn test_reshard_differentiation() {
        // Resharding is linear, so the tangent is resharded to the same target as the primal.
        let target = Sharding::new(mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let (primal, tangent) = differentiate_at(Array::vector(vec![1.0f32, 2.0]).unwrap())
            .jvp(Array::vector(vec![3.0f32, 4.0]).unwrap(), |x| x.reshard(&target))
            .unwrap();
        let expected_type = ArrayType::new_static(DataType::F32, [2]).with_sharding(target).unwrap();
        assert_eq!(primal, Array::from_elements(expected_type.clone(), &[1.0f32, 2.0]).unwrap());
        assert_eq!(tangent, Array::from_elements(expected_type, &[3.0f32, 4.0]).unwrap());
    }

    #[test]
    fn test_reshard_transposition() {
        let mesh = mesh();
        // The input is unreduced along the manual axis `m`, so its cotangent must be distributed like the input:
        // the dual sharding, which is reduced along `m`.
        let input_sharding = Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["m"]).unwrap();
        let input = Array::from_elements::<f64>(
            ArrayType::new_static(DataType::F64, [8]).with_sharding(input_sharding.clone()).unwrap(),
            &[1.0; 8],
        )
        .unwrap();
        let target = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_cotangent_type = input.r#type().cotangent().unwrap();
        let (output, pullback) = differentiate_at(input).vjp(|x| x.reshard(&target)).unwrap();
        let cotangent = pullback
            .apply(Array::from_elements::<f64>(output.r#type().cotangent().unwrap(), &[1.0; 8]).unwrap())
            .unwrap();
        assert_eq!(cotangent.r#type().as_ref(), &input_cotangent_type);
        assert_eq!(cotangent.to_f64s(), vec![1.0; 8]);
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        let shardings = pullback
            .instructions()
            .iter()
            .map(|instruction| match instruction.operation() {
                ArrayOperation::Reshard(operation) => operation.sharding().clone(),
                operation => panic!("unexpected operation `{operation}`"),
            })
            .collect::<Vec<_>>();
        assert_eq!(shardings, vec![Sharding::replicated(input_sharding.mesh().clone(), 1)]);
    }

    #[test]
    fn test_reshard_transposition_unsharded_input() {
        // An input without a sharding receives an exactly unsharded cotangent through an identity broadcast rather
        // than a reshard, including for element formats whose cotangent space widens to `f32`.
        let target = Sharding::new(mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new_static(DataType::F8E8M0FNU, [2]);
        let input = Array::from_elements::<f8e8m0fnu>(
            input_type.clone(),
            &[1.0; 2].map(|value| f8e8m0fnu::from_f64(value).unwrap()),
        )
        .unwrap();
        let (output, pullback) = differentiate_at(input.clone()).vjp(|x| x.reshard(&target)).unwrap();
        let cotangent = pullback
            .apply(Array::from_elements::<f32>(output.r#type().cotangent().unwrap(), &[1.0; 2]).unwrap())
            .unwrap();
        assert_eq!(cotangent.r#type().as_ref(), &input_type.cotangent().unwrap());
        assert_eq!(cotangent.to_f64s(), vec![1.0; 2]);

        // Reverse Jacobian construction preserves the widened element type and unsharded input space too.
        let jacobian = differentiate_at(input).jacobian_reverse(|x| x.reshard(&target)).unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.input_type(), &input_type);
        assert_eq!(block.value().r#type().data_type(), DataType::F32);
        assert_eq!(block.value().r#type().static_shape().unwrap().as_slice(), &[2, 2]);
        assert_eq!(block.value().r#type().sharding(), None);
        assert_eq!(block.value().to_f64s(), vec![1.0, 0.0, 0.0, 1.0],);
    }

    #[test]
    fn test_array_reshard() {
        let mesh = mesh();
        // Resharding records the target on the type, carrying the input's varying manual axes over exactly like the
        // `ReshardOperation` type-inference rule, and leaves the payload untouched.
        let input = Array::from_elements::<f64>(
            ArrayType::new_static(DataType::F64, [2])
                .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["m"]).unwrap())
                .unwrap(),
            &[1.0, 2.0],
        )
        .unwrap();
        let target = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let resharded = input.reshard(&target).unwrap();
        assert_eq!(resharded.r#type().sharding(), Some(&target.clone().with_varying_manual_axes(["m"]).unwrap()));
        assert_eq!(resharded.storage_bytes(), input.storage_bytes());

        // Eager evaluation validates the target exactly like staged programs do.
        assert_eq!(
            input.reshard(&Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["a"])]).unwrap()),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{RESHARD_OPERATION_NAME}` cannot target auto mesh axes; use \
                 `{CONSTRAIN_SHARDING_OPERATION_NAME}` to constrain placement over them",
            )))),
        );
        assert_eq!(
            input.reshard(&Sharding::replicated(mesh, 2)),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{RESHARD_OPERATION_NAME}` target sharding rank (2) does not match the input rank (1)",
            )))),
        );
    }
}
