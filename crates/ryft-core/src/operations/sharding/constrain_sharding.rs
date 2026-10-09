use std::fmt::Display;

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
use crate::operations::sharding::reshard::RESHARD_OPERATION_NAME;
use crate::parameters::{Parameter, Parameterized, ParameterizedFamily};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, ProgramError, RegionInterface, Type, TypeError, Typed, Value,
    ValueDirectDispatch, ValueDomainDispatch, ValueProjection,
};

#[cfg(doc)]
use crate::operations::sharding::reshard::{Reshard, ReshardOperation};

/// Canonical operation name for [`ConstrainShardingOperation`].
pub const CONSTRAIN_SHARDING_OPERATION_NAME: &str = "constrain_sharding";

/// [`Operation`] that constrains the placement of its input over [`MeshAxisType::Auto`] mesh axes.
/// This operation is the Ryft analogue of JAX's [`jax.lax.with_sharding_constraint`](
/// https://docs.jax.dev/en/latest/_autosummary/jax.lax.with_sharding_constraint.html). Type inference is the identity,
/// so the output type, sharding included, equals the input type and the constraint never becomes type-level state.
/// The constraint is nevertheless binding. When the program is lowered, the backend compiler must place the value as
/// requested over the auto axes and only remains free where the constraint is unconstrained. A constraint that shards
/// a dimension over a non-auto axis is rejected, because such placements are tracked transitions that belong to a
/// [`ReshardOperation`], and a constraint on a different mesh than the input's tracked sharding is rejected as well.
/// Refer to the documentation of [`ConstrainSharding`] for more information.
///
/// Interpretation passes the value through unchanged. Batching leaves the new batch axis unconstrained in the lifted
/// constraint, so that the compiler remains free to place it. Differentiation applies the same constraint to the
/// tangent, and the operation is self-adjoint under transposition, so the same constraint applies to the cotangent as
/// well. Backends lower [`lowered_sharding`](Self::lowered_sharding) rather than the constraint itself, so that the
/// emitted constraint also carries the input's tracked placement.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConstrainShardingOperation {
    /// Underlying [`Sharding`] constraint over [`MeshAxisType::Auto`] mesh axes.
    sharding: Sharding,
}

impl ConstrainShardingOperation {
    /// Creates a new [`ConstrainShardingOperation`] that constrains the placement of its input to `sharding`.
    #[inline]
    pub fn new(sharding: Sharding) -> Self {
        Self { sharding }
    }

    /// Returns the underlying [`Sharding`] constraint over [`MeshAxisType::Auto`] mesh axes.
    #[inline]
    pub fn sharding(&self) -> &Sharding {
        &self.sharding
    }

    /// Returns the [`Sharding`] a backend must constrain an input of type `input_type` to for this
    /// [`ConstrainShardingOperation`]. An input without a tracked sharding is constrained to [`Self::sharding`] as is.
    /// Otherwise, the tracked placement is merged into the constraint dimension by dimension, so that the emitted
    /// constraint never contradicts the type the program was checked against (i.e., a tracked sharded dimension keeps
    /// its axes and gains the constraint's auto axes after them, or stays as is where the constraint is replicated or
    /// unconstrained, whereas a tracked replicated dimension takes the constraint's entry, since a tracked type is
    /// only replicated over the axes the type system governs and leaves the auto axes to the compiler). The tracked
    /// unreduced, reduced, and varying manual axes are carried over as well.
    ///
    /// # Parameters
    ///
    ///   - `input_type`: Type of the input this operation is applied to, whose rank and mesh (if it carries a
    ///     sharding) must match those of this constraint.
    pub fn lowered_sharding(&self, input_type: &ArrayType) -> Result<Sharding, TypeError> {
        let Some(tracked) = input_type.sharding() else {
            return Ok(self.sharding.clone());
        };

        // A tracked type may still name auto axes (e.g., the actual placement of a concrete input), but placement
        // over those axes is the compiler's to decide and exactly what this constraint overrides, so only the
        // placement the type system governs is carried over.
        let tracked = tracked.without_auto_axes();
        if tracked.rank() != self.sharding.rank() {
            return Err(TypeError::invalid(format!(
                "`{}` rank ({}) does not match the input rank ({})",
                CONSTRAIN_SHARDING_OPERATION_NAME,
                self.sharding.rank(),
                tracked.rank(),
            )));
        }

        if tracked.mesh() != self.sharding.mesh() {
            return Err(TypeError::invalid(format!(
                "`{}` sharding {} is on a different mesh than the input sharding {}",
                CONSTRAIN_SHARDING_OPERATION_NAME, self.sharding, tracked,
            )));
        }

        let dimensions = tracked
            .dimensions()
            .iter()
            .zip(self.sharding.dimensions())
            .map(|(tracked, constrained)| match (tracked, constrained) {
                (ShardingDimension::Sharded(tracked_axes), ShardingDimension::Sharded(constrained_axes)) => {
                    ShardingDimension::Sharded(tracked_axes.iter().chain(constrained_axes).cloned().collect())
                }
                (ShardingDimension::Sharded(_), ShardingDimension::Replicated | ShardingDimension::Unconstrained) => {
                    tracked.clone()
                }
                (ShardingDimension::Replicated | ShardingDimension::Unconstrained, constrained) => constrained.clone(),
            })
            .collect();

        self.sharding
            .with_dimensions(dimensions)
            .and_then(|sharding| {
                sharding.with_unreduced_axes(tracked.unreduced_axes().union(self.sharding.unreduced_axes()).cloned())
            })
            .and_then(|sharding| {
                sharding.with_reduced_axes(tracked.reduced_axes().union(self.sharding.reduced_axes()).cloned())
            })
            .and_then(|sharding| {
                sharding.with_varying_manual_axes(
                    tracked.varying_manual_axes().union(self.sharding.varying_manual_axes()).cloned(),
                )
            })
            .map_err(|error| TypeError::invalid(error.to_string()))
    }
}

impl Display for ConstrainShardingOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ConstrainShardingOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        CONSTRAIN_SHARDING_OPERATION_NAME
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
                "`{}` rank ({}) does not match the input rank ({})",
                CONSTRAIN_SHARDING_OPERATION_NAME,
                self.sharding.rank(),
                input.rank(),
            )));
        }

        if let Some(tracked) = input.sharding()
            && tracked.mesh() != self.sharding.mesh()
        {
            return Err(TypeError::invalid(format!(
                "`{}` sharding {} is on a different mesh than the input sharding {}",
                CONSTRAIN_SHARDING_OPERATION_NAME, self.sharding, tracked,
            )));
        }

        // The constraint may only place dimensions over auto axes, which are the axes the compiler propagates.
        // Naming an explicit or manual axis is the mirror image of the auto-axis rejection of `reshard`.
        for dimension in self.sharding.dimensions() {
            let ShardingDimension::Sharded(axis_names) = dimension else {
                continue;
            };

            if let Some(axis_name) = axis_names
                .iter()
                .find(|axis_name| self.sharding.mesh().axis_type(axis_name) != Some(MeshAxisType::Auto))
            {
                return Err(TypeError::invalid(format!(
                    "`{CONSTRAIN_SHARDING_OPERATION_NAME}` can only constrain placement over auto mesh axes but \
                     `{axis_name}` is not one; use `{RESHARD_OPERATION_NAME}` for explicit axes",
                )));
            }
        }

        // The constraint is untracked. The output type, sharding included, is identical to the input, and the
        // constraint is enforced only when the backend lowers the operation.
        Ok(vec![input.clone()])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("sharding", &self.sharding))
    }
}

impl<C: Domain<Type = ArrayType, Value: ConstrainSharding>> InterpretableOperation<C> for ConstrainShardingOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // The constraint flows through the capability so interpretation over staging values preserves it.
        // Concrete values pass through unchanged.
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].constrain_sharding(&self.sharding)?])
    }
}

impl<C: Context<Operation: From<ConstrainShardingOperation>>> PartiallyEvaluatableOperation<C>
    for ConstrainShardingOperation
{
}

impl<C: Context<Type = ArrayType, Value: ConstrainSharding>, P: ArrayExtentBatchingPolicy<C>>
    BatchableOperation<C, ArrayBatchingPolicy<P>> for ConstrainShardingOperation
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        // The lifted constraint gains a `ShardingDimension::Unconstrained` entry at the new batch dimension as the
        // constraint governs only the compiler-propagated auto axes, and so the new dimension is left open for the
        // backend to fill rather than pinned to a derived or replicated entry. Like the `reshard` rule, lifting never
        // needs the batch axis's extent.
        check_count!("input", inputs, 1, ProgramError);
        let lifted_sharding = match inputs[0].batch_axis_position() {
            Some(batch_axis) => self.sharding().batched(batch_axis, ShardingDimension::Unconstrained)?,
            None => self.sharding().clone(),
        };

        // Sharding changes preserve the packed shape, so the batch axis and ragged-axis metadata carry over unchanged.
        let batch_axis = BatchAxis::from_optional_position(inputs[0].batch_axis_position());
        let mut outputs = ConstrainShardingOperation::new(lifted_sharding).interpret_with_batch_axes(
            context,
            inputs,
            &[batch_axis],
        )?;
        check_count!("output", outputs, 1, ProgramError);
        let output = ArrayBatch::new(outputs.remove(0).into_value(), batch_axis)?
            .with_ragged_axes(inputs[0].ragged_axes().to_vec())?;
        Ok(vec![output].into())
    }
}

impl_differentiable_operation! {
    ConstrainShardingOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Value: ConstrainSharding,
        C::Operation: From<ConstrainShardingOperation>,
    {
        |operation, _context, _driver, inputs| {
            // The constraint is linear, so the same constraint applies to the tangent as to the primal.
            check_count!("input", inputs, 1, ProgramError);
            let primal = inputs[0].primal().constrain_sharding(operation.sharding())?;
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                MaybeZero::Value(tangent) => MaybeZero::Value(tangent.constrain_sharding(operation.sharding())?),
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType> + From<ConstrainShardingOperation>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // The operation is self-adjoint, so the output cotangent is constrained by the same constraint (mirroring
            // JAX registering `with_sharding_constraint` with `ad.deflinear2`). Unlike the `reshard` rule, the input's
            // sharding is not consulted, because the constraint is the operation's own.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 1, DifferentiationError);
            match &outputs[0] {
                MaybeZero::Zero(_) => Ok(()),
                MaybeZero::Value(cotangent) => {
                    let contribution = MaybeZero::Value(cotangent.constrain_sharding(operation.sharding())?);
                    accumulators[0].accumulate(context, contribution)?;
                    Ok(())
                }
            }
        }
    },
}

/// Executes one [`ConstrainShardingOperation`] for a value through its [`Value::Dispatch`] policy. This backend
/// extension point separates primitive leaf execution from the structural [`ConstrainSharding`] capability.
/// Domain-dispatched array values bind the operation in their own context while direct eager arrays execute its checked
/// semantics without binding it again. Backend authors can implement this trait on [`ValueDirectDispatch`] for their
/// own value types. Implementations must validate targets through the operation's type inference contract and preserve
/// the input's elements, shape, data type, and domain. Composite implementations project the array member and lift the
/// result back, so dimensions and references retain their existing projection diagnostics.
///
/// `T` is the value's type descriptor, and `V` is its representation. Including `T` keeps dispatch over homogeneous
/// arrays and composite array IR values disjoint.
pub trait ConstrainShardingDispatch<T: Type, V: Value<Type = T>> {
    /// Applies the primitive sharding operation to `input` and returns its value.
    fn constrain_sharding(input: &V, sharding: &Sharding) -> Result<V, ProgramError>;
}

impl ConstrainShardingDispatch<ArrayType, Array> for ValueDirectDispatch {
    fn constrain_sharding(input: &Array, sharding: &Sharding) -> Result<Array, ProgramError> {
        // The constraint is untracked, and so the output of a concrete single-device value is the value itself
        // once the `ConstrainShardingOperation` type inference rule has accepted the constraint for its type.
        let input_type = input.r#type().into_owned();
        ConstrainShardingOperation::new(sharding.clone()).infer_output_types(&[input_type], &[])?;
        Ok(input.clone())
    }
}

impl<
    V: Value<
            Type = ArrayType,
            Dispatch = ValueDomainDispatch,
            Domain: Context<Type = ArrayType, Operation: From<ConstrainShardingOperation>>,
        >,
> ConstrainShardingDispatch<ArrayType, V> for ValueDomainDispatch
{
    fn constrain_sharding(input: &V, sharding: &Sharding) -> Result<V, ProgramError> {
        // Staging and transform values bind through their domain while direct eager execution
        // uses the separate policy above.
        let mut outputs = input.domain().bind(
            ConstrainShardingOperation::new(sharding.clone()),
            Vec::new(),
            std::slice::from_ref(input),
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<
    V: Value<Type = ArrayIrType>
        + ValueProjection<
            ArrayType,
            Projected: Value<Type = ArrayType, Dispatch: ConstrainShardingDispatch<ArrayType, V::Projected>>,
        >,
> ConstrainShardingDispatch<ArrayIrType, V> for V::Dispatch
{
    fn constrain_sharding(input: &V, sharding: &Sharding) -> Result<V, ProgramError> {
        let input = input.clone().into_projected()?;
        let output = <V::Projected as Value>::Dispatch::constrain_sharding(&input, sharding)?;
        Ok(V::from_projected(output))
    }
}

/// Constrains the placement of the leaves of a [`Parameterized`] receiver over [`Auto`](MeshAxisType::Auto) mesh axes
/// to the corresponding [`Sharding`]s. This preserves each leaf's elements, shape, data type, and tracked sharding
/// while recording a [`ConstrainShardingOperation`] that backend lowering must enforce. Lowering merges the requested
/// auto-axis placement with the input's tracked placement (refer to [`ConstrainShardingOperation::lowered_sharding`]),
/// so the compiler must honor both and remains free where placement is unconstrained. To perform a tracked transition
/// over [`Explicit`](MeshAxisType::Explicit) axes, use [`Reshard::reshard`].
///
/// Concrete single-device arrays are returned unchanged after type validation. Context-carrying values stage the
/// operation, preserving the constraint through interpretation and transformations. Differentiation applies the
/// same constraint to tangents and cotangents; the input's tracked sharding is not consulted for the transpose.
///
/// The leaf parameter `P` defaults to `Self`: a value is a one-leaf structure accepting one `Sharding`, while nested
/// tuples, vectors, maps, and custom structures accept the corresponding `Self::To<Sharding>`. Parameter paths must
/// agree, and the receiver's structure and static fields are preserved without requiring the receiver to be `Clone`.
/// Homogeneous and composite array values share this API through their [`ConstrainShardingDispatch`] implementations.
///
/// The universe parameter `T` defaults to the leaf's [`Capability`] universe, so that homogeneous array leaves use
/// [`ArrayType`] and composite array IR leaves use [`ArrayIrType`]. A structure uses the same universe as its leaves.
/// Implementations tie `T` to that universe, which selects their primitive dispatch family; specifying a different
/// universe does not grant the capability. The universe is distinct from `P`, which identifies the structural leaves.
pub trait ConstrainSharding<P: Parameter + Capability = Self, T = <P as Capability>::Universe>: Sized {
    /// Applies the sharding operation to every leaf with its corresponding [`Sharding`].
    ///
    /// # Parameters
    ///
    ///   - `shardings`: Shardings with the same ordered parameter paths as the receiver.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidArgument`] before executing any leaf if the parameter paths differ, and the
    /// primitive dispatch's type-inference, projection, or execution errors otherwise.
    fn constrain_sharding(&self, shardings: &<Self as Parameterized<P>>::To<Sharding>) -> Result<Self, ProgramError>
    where
        Self: Parameterized<P, Family: ParameterizedFamily<Sharding>>;
}

// The `Universe = T` constraint selects the leaf's actual execution family; its type view drives primitive dispatch.
// Keep the structural bound on the function as putting it on the trait shadows the concrete leaf family in generic
// `V: Value + ConstrainSharding` code, preventing its target from normalizing to a single `Sharding`.
impl<T, P: Value<Dispatch: ConstrainShardingDispatch<P::Type, P>> + Capability<Universe = T>, S: Parameterized<P>>
    ConstrainSharding<P, T> for S
{
    #[inline]
    fn constrain_sharding(&self, shardings: &S::To<Sharding>) -> Result<Self, ProgramError>
    where
        S::Family: ParameterizedFamily<Sharding>,
    {
        if !self.parameter_paths().eq(shardings.parameter_paths()) {
            return Err(ProgramError::InvalidArgument {
                message: "receiver and `shardings` must have the same parameter structure".to_string(),
            });
        }
        let outputs = self
            .parameters()
            .zip(shardings.parameters())
            .map(|(input, sharding)| P::Dispatch::constrain_sharding(input, sharding))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::from_parameters(self.parameter_structure(), outputs)?)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, ArrayReference, DataType, Dimension, DimensionBounds,
        DimensionType, DimensionValue, DimensionVariable, DynamicArrayExtentBatchingPolicy, LogicalMesh, MeshAxis,
        RaggedAxis, Shape,
    };
    use crate::batching::{BatchAxis, batch};
    use crate::contexts::{EagerContext, ProjectedContext, StagingContext};
    use crate::differentiation::differentiate_at;
    use crate::macros::{check_operation_partial_evaluation, check_operation_type_inference};
    use crate::parameters::Placeholder;
    use crate::programs::{EffectClasses, EmptyRegionDriver};
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
    fn test_constrain_sharding() {
        let constraint = Sharding::new(mesh(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let operation = ConstrainShardingOperation::new(constraint.clone());

        // Operation identity, accessors, and rendering.
        assert_eq!(operation.name(), CONSTRAIN_SHARDING_OPERATION_NAME);
        assert_eq!(operation.sharding(), &constraint);
        assert_eq!(operation.to_string(), format!("{CONSTRAIN_SHARDING_OPERATION_NAME} [sharding={constraint}]"));
    }

    #[test]
    fn test_constrain_sharding_lowered_sharding() {
        let mesh = mesh();
        let constraint = Sharding::new(
            mesh.clone(),
            vec![
                ShardingDimension::sharded(["a"]),
                ShardingDimension::replicated(),
                ShardingDimension::unconstrained(),
            ],
        )
        .unwrap();
        let operation = ConstrainShardingOperation::new(constraint.clone());
        let input_type = ArrayType::new_static(DataType::F32, [8, 4, 2]);

        // An input without a tracked sharding is constrained to the constraint as is.
        assert_eq!(operation.lowered_sharding(&input_type), Ok(constraint.clone()));

        // A tracked sharded dimension keeps its axes and gains the constraint's auto axes after them, or stays as is
        // where the constraint is replicated or unconstrained, whereas a tracked replicated dimension takes the
        // constraint's entry. Tracked variation and reduction state carries over.
        let tracked = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["m"]), ShardingDimension::replicated()],
        )
        .unwrap()
        .with_varying_manual_axes(["m"])
        .unwrap();
        assert_eq!(
            operation.lowered_sharding(&input_type.clone().with_sharding(tracked).unwrap()),
            Ok(Sharding::new(
                mesh.clone(),
                vec![
                    ShardingDimension::sharded(["x", "a"]),
                    ShardingDimension::sharded(["m"]),
                    ShardingDimension::unconstrained(),
                ],
            )
            .unwrap()
            .with_varying_manual_axes(["m"])
            .unwrap()),
        );
        let tracked = Sharding::replicated(mesh.clone(), 3).with_unreduced_axes(["x"]).unwrap();
        assert_eq!(
            operation.lowered_sharding(&input_type.clone().with_sharding(tracked).unwrap()),
            Ok(Sharding::new(
                mesh.clone(),
                vec![
                    ShardingDimension::sharded(["a"]),
                    ShardingDimension::replicated(),
                    ShardingDimension::unconstrained(),
                ],
            )
            .unwrap()
            .with_unreduced_axes(["x"])
            .unwrap()),
        );
        let tracked = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        assert_eq!(
            operation.lowered_sharding(&input_type.clone().with_sharding(tracked).unwrap()),
            Ok(Sharding::new(
                mesh.clone(),
                vec![
                    ShardingDimension::sharded(["a"]),
                    ShardingDimension::sharded(["x"]),
                    ShardingDimension::unconstrained(),
                ],
            )
            .unwrap()),
        );

        // Reduced axes are preserved independently of placement, just like unreduced and varying axes.
        let tracked = Sharding::replicated(mesh.clone(), 3).with_reduced_axes(["x"]).unwrap();
        assert_eq!(
            operation.lowered_sharding(&input_type.clone().with_sharding(tracked).unwrap()),
            Ok(constraint.clone().with_reduced_axes(["x"]).unwrap()),
        );

        // Unconstrained tracked dimensions leave each dimension's placement to the constraint.
        let tracked = Sharding::new(mesh.clone(), vec![ShardingDimension::unconstrained(); 3]).unwrap();
        assert_eq!(operation.lowered_sharding(&input_type.clone().with_sharding(tracked).unwrap()), Ok(constraint));

        // Tracked placement over auto axes is the compiler's to decide and is overridden by the constraint.
        let tracked = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["a"]), ShardingDimension::replicated()],
        )
        .unwrap();
        assert_eq!(
            operation.lowered_sharding(&input_type.clone().with_sharding(tracked).unwrap()),
            Ok(Sharding::new(
                mesh.clone(),
                vec![
                    ShardingDimension::sharded(["a"]),
                    ShardingDimension::replicated(),
                    ShardingDimension::unconstrained(),
                ],
            )
            .unwrap()),
        );

        // A tracked sharding on another mesh or of another rank cannot be merged.
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        assert_eq!(
            operation.lowered_sharding(&input_type.clone().with_sharding(Sharding::replicated(other_mesh, 3)).unwrap()),
            Err(TypeError::invalid(format!(
                "`{CONSTRAIN_SHARDING_OPERATION_NAME}` sharding {{mesh<['x'=2:explicit, 'm'=2:manual, \
                 'a'=2:auto]>, [{{'a'}}, {{}}, {{?}}]}} is on a different mesh than the input sharding \
                 {{mesh<['y'=2:explicit]>, [{{}}, {{}}, {{}}]}}",
            ))),
        );
        assert_eq!(
            operation.lowered_sharding(
                &ArrayType::new_static(DataType::F32, [8]).with_sharding(Sharding::replicated(mesh, 1)).unwrap(),
            ),
            Err(TypeError::invalid(format!(
                "`{CONSTRAIN_SHARDING_OPERATION_NAME}` rank (3) does not match the input rank (1)",
            ))),
        );
    }

    #[test]
    fn test_constrain_sharding_type_inference() {
        let mesh = mesh();
        let constraint = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [8]);
        let sharded_input_type = input_type
            .clone()
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();

        // Inference is the identity: the input type passes through untouched, whether or not it carries a sharding of
        // its own, and the constraint never appears on it.
        check_operation_type_inference!(
            operation = ConstrainShardingOperation::new(constraint.clone()),
            cases = [{
                input_types = [input_type.clone()],
                output_types = [input_type.clone()],
            }, {
                input_types = [sharded_input_type.clone()],
                output_types = [sharded_input_type],
            }, {
                input_types = [ArrayType::new_static(DataType::F32, [8, 2])],
                error = format!(
                    "`{CONSTRAIN_SHARDING_OPERATION_NAME}` rank (1) does not match the input rank (2)",
                ),
            }, {
                input_types = [],
                error = "expected 1 input but got 0",
            }],
        );

        // A tracked sharding on another mesh cannot be merged with the constraint, placement over explicit or manual
        // axes is a tracked transition, and the operation cannot own nested regions.
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other_input_type = input_type.clone().with_sharding(Sharding::replicated(other_mesh, 1)).unwrap();
        check_operation_type_inference!(
            operation = ConstrainShardingOperation::new(constraint.clone()),
            cases = [{
                input_types = [other_input_type],
                error = format!(
                    "`{CONSTRAIN_SHARDING_OPERATION_NAME}` sharding {{mesh<['x'=2:explicit, 'm'=2:manual, \
                     'a'=2:auto]>, [{{'a'}}]}} is on a different mesh than the input sharding \
                     {{mesh<['y'=2:explicit]>, [{{}}]}}",
                ),
            }],
        );
        let explicit_constraint = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        check_operation_type_inference!(
            operation = ConstrainShardingOperation::new(explicit_constraint),
            cases = [{
                input_types = [input_type.clone()],
                error = format!(
                    "`{CONSTRAIN_SHARDING_OPERATION_NAME}` can only constrain placement over auto mesh axes but `x` \
                     is not one; use `{RESHARD_OPERATION_NAME}` for explicit axes",
                ),
            }],
        );
        let manual_constraint = Sharding::new(mesh, vec![ShardingDimension::sharded(["m"])]).unwrap();
        check_operation_type_inference!(
            operation = ConstrainShardingOperation::new(manual_constraint),
            cases = [{
                input_types = [input_type.clone()],
                error = format!(
                    "`{CONSTRAIN_SHARDING_OPERATION_NAME}` can only constrain placement over auto mesh axes but `m` \
                     is not one; use `{RESHARD_OPERATION_NAME}` for explicit axes",
                ),
            }],
        );
        assert_eq!(
            ConstrainShardingOperation::new(constraint).infer_output_types(
                std::slice::from_ref(&input_type),
                &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_constrain_sharding_interpretation() {
        let mesh = mesh();
        let constraint = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let operation = ConstrainShardingOperation::new(constraint);

        // An untracked constraint preserves the payload and all existing type metadata,
        // including manual-axis variation.
        let input = Array::from_elements::<f64>(
            ArrayType::new_static(DataType::F64, [2])
                .with_sharding(Sharding::replicated(mesh, 1).with_varying_manual_axes(["m"]).unwrap())
                .unwrap(),
            &[1.0, 2.0],
        )
        .unwrap();
        assert_eq!(
            operation.interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input)),
            Ok(vec![input]),
        );
        assert_eq!(
            operation.interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
    }

    #[test]
    fn test_constrain_sharding_partial_evaluation() {
        let constraint = Sharding::new(mesh(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        check_operation_partial_evaluation!(
            operation = ConstrainShardingOperation::new(constraint),
            inputs = [Array::vector(vec![1.0f32, 2.0]).unwrap()],
            expected = Array::vector(vec![1.0f32, 2.0]).unwrap(),
        );
    }

    #[test]
    fn test_constrain_sharding_batching() {
        // The constraint governs only the compiler-propagated auto axes, so batching leaves the new batch axis
        // unconstrained for the backend to fill rather than pinning it to a derived or replicated entry.
        let constraint = Sharding::new(mesh(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let expected_lifted = constraint.with_inserted_dimension(0, ShardingDimension::Unconstrained).unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x| {
                let constraint = constraint.clone();
                Ok(batch(
                    move |item| item.constrain_sharding(&constraint),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )?)
            },
            ArrayType::new_static(DataType::F64, [2, 3]),
        )
        .unwrap();
        let shardings = program
            .instructions()
            .iter()
            .map(|instruction| match instruction.operation() {
                ArrayOperation::ConstrainSharding(operation) => operation.sharding().clone(),
                operation => panic!("unexpected operation `{operation}`"),
            })
            .collect::<Vec<_>>();
        assert_eq!(shardings, vec![expected_lifted]);
    }

    #[test]
    fn test_constrain_sharding_batching_ragged() {
        // The constraint preserves the packed geometry, so a bounded ragged axis on the input carries
        // over to the output.
        let constraint = Sharding::new(mesh(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array>::new(), 2);
        let array =
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[1f32, 2., 3., 4., 5., 6.]).unwrap();
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let extents = Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[1i32, 3]).unwrap();
        let input = ArrayBatch::new(array.clone(), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, extents, variable, vec![0])])
            .unwrap();
        let ragged_axes = input.ragged_axes().to_vec();
        let (outputs, _) = ConstrainShardingOperation::new(constraint.clone())
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis_position(), Some(0));
        assert_eq!(outputs[0].ragged_axes(), ragged_axes.as_slice());
        assert_eq!(outputs[0].value(), &array);
    }

    #[test]
    fn test_constrain_sharding_batching_dynamic() {
        // Lifting needs only the batch axis's position, so a mapped axis with a dynamic extent lifts.
        let constraint = Sharding::new(mesh(), vec![ShardingDimension::sharded(["a"])]).unwrap();
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
        let (outputs, _) = ConstrainShardingOperation::new(constraint)
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis_position(), Some(0));
        assert_eq!(outputs[0].r#type().shape(), &shape);
        assert_eq!(outputs[0].r#type().sharding(), None);
    }

    #[test]
    fn test_constrain_sharding_batching_replicated() {
        // An unmapped input keeps its original rank and remains replicated across the batch.
        let constraint = Sharding::new(mesh(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array>::new(), 2);
        let array = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let input = ArrayBatch::new(array.clone(), BatchAxis::replicated()).unwrap();
        let (outputs, _) = ConstrainShardingOperation::new(constraint.clone())
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis_position(), None);
        assert!(outputs[0].ragged_axes().is_empty());
        assert_eq!(outputs[0].value(), &array);
    }

    #[test]
    fn test_constrain_sharding_differentiation() {
        // The constraint is linear, so the JVP applies the same constraint to the primal and to the tangent.
        let constraint = Sharding::new(mesh(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x| x.constrain_sharding(&constraint),
            ArrayType::new_static(DataType::F32, [4]),
        )
        .unwrap();
        let jvp = program.to_flat_program().jvp().unwrap();
        let constraints = jvp
            .instructions()
            .iter()
            .map(|instruction| match instruction.operation() {
                ArrayOperation::ConstrainSharding(operation) => operation.sharding().clone(),
                operation => panic!("expected only sharding constraints but got `{operation}`"),
            })
            .collect::<Vec<_>>();
        assert_eq!(constraints, vec![constraint.clone(), constraint]);
    }

    #[test]
    fn test_constrain_sharding_transposition() {
        // The constraint is self-adjoint, so its transpose re-applies the same constraint to the cotangent rather than
        // dualizing it.
        let constraint = Sharding::new(mesh(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let (_, pullback) = differentiate_at(Array::vector(vec![1.0; 8]).unwrap())
            .vjp(|x| x.constrain_sharding(&constraint))
            .unwrap();
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        let shardings = pullback
            .instructions()
            .iter()
            .map(|instruction| match instruction.operation() {
                ArrayOperation::ConstrainSharding(operation) => operation.sharding().clone(),
                operation => panic!("unexpected operation `{operation}`"),
            })
            .collect::<Vec<_>>();
        assert_eq!(shardings, vec![constraint]);
    }

    #[test]
    fn test_array_constrain_sharding() {
        // The constraint is metadata for lowering only, so a concrete array is returned unchanged.
        let mesh = mesh();
        let constraint = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
        assert_eq!(input.constrain_sharding(&constraint), Ok(input.clone()));

        // Eager evaluation validates the constraint exactly like staged programs do.
        assert_eq!(
            input.constrain_sharding(&Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{CONSTRAIN_SHARDING_OPERATION_NAME}` can only constrain placement over auto mesh axes but `x` \
                 is not one; use `{RESHARD_OPERATION_NAME}` for explicit axes",
            )))),
        );
        assert_eq!(
            input.constrain_sharding(&Sharding::replicated(mesh, 2)),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{CONSTRAIN_SHARDING_OPERATION_NAME}` rank (2) does not match the input rank (1)",
            )))),
        );
    }

    #[test]
    fn test_constrain_sharding_structured() {
        // Every leaf of a structured value is constrained by its own sharding, and its tracked type is unchanged.
        let mesh = mesh();
        let tracked = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let constraint = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let input_types = (
            ArrayType::new_static(DataType::F32, [8]).with_sharding(tracked).unwrap(),
            ArrayType::new_static(DataType::F32, [4]),
        );
        let (output_types, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |inputs| inputs.constrain_sharding(&(constraint.clone(), constraint.clone())),
            input_types.clone(),
        )
        .unwrap();
        assert_eq!(output_types, input_types);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[8][sharding={mesh<['x'=2:explicit, 'm'=2:manual, 'a'=2:auto]>, [{'x'}]}], %1:f32[4] .
                let %2:f32[8][sharding={mesh<['x'=2:explicit, 'm'=2:manual, 'a'=2:auto]>, [{'x'}]}] = \
                        constrain_sharding [sharding={mesh<['x'=2:explicit, 'm'=2:manual, 'a'=2:auto]>, [{'a'}]}] %0
                    %3:f32[4] = \
                        constrain_sharding [sharding={mesh<['x'=2:explicit, 'm'=2:manual, 'a'=2:auto]>, [{'a'}]}] %1
                in (%2, %3)"
            },
        );

        // A constraint whose rank differs from the rank of its leaf is rejected before anything is staged.
        let result = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input| input.constrain_sharding(&Sharding::replicated(mesh, 2)),
            ArrayType::new_static(DataType::F32, [8]),
        );
        assert!(matches!(
            result,
            Err(ProgramError::Type(error))
                if error.to_string() == "`constrain_sharding` rank (2) does not match the input rank (1)",
        ));
    }

    #[test]
    fn test_constrain_sharding_structured_composite_tracing() {
        // Every leaf of a structured value is constrained by its own sharding, and its tracked type is unchanged.
        let mesh = mesh();
        let tracked = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let constraint = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let input_types = (
            ArrayType::new_static(DataType::F32, [8]).with_sharding(tracked).unwrap(),
            ArrayType::new_static(DataType::F32, [4]),
        );
        let (output_types, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |inputs| inputs.constrain_sharding(&(constraint.clone(), constraint.clone())),
            (ArrayIrType::Array(input_types.0.clone()), ArrayIrType::Array(input_types.1.clone())),
        )
        .unwrap();
        assert_eq!(output_types, (input_types.0.into(), input_types.1.into()));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[8][sharding={mesh<['x'=2:explicit, 'm'=2:manual, 'a'=2:auto]>, [{'x'}]}], %1:f32[4] .
                let %2:f32[8][sharding={mesh<['x'=2:explicit, 'm'=2:manual, 'a'=2:auto]>, [{'x'}]}] = \
                        constrain_sharding [sharding={mesh<['x'=2:explicit, 'm'=2:manual, 'a'=2:auto]>, [{'a'}]}] %0
                    %3:f32[4] = \
                        constrain_sharding [sharding={mesh<['x'=2:explicit, 'm'=2:manual, 'a'=2:auto]>, [{'a'}]}] %1
                in (%2, %3)"
            },
        );
    }

    #[test]
    fn test_constrain_sharding_structured_projection_errors() {
        let sharding = Sharding::replicated(mesh(), 0);
        let dimension = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(1).unwrap());
        assert!(matches!(
            dimension.constrain_sharding(&sharding),
            Err(ProgramError::Type(error)) if error.to_string() == "expected array type but got dimension type",
        ));
        let reference = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1.0f32).unwrap()));
        assert!(matches!(
            reference.constrain_sharding(&sharding),
            Err(ProgramError::Type(error)) if error.to_string() == "expected array type but got reference type",
        ));
    }

    #[test]
    fn test_constrain_sharding_structured_eager() {
        /// Structured arrays with a static field that sharding operations must preserve.
        #[derive(ryft_macros::Parameterized)]
        struct Inputs<P: Parameter> {
            /// Array leaves in a nested tuple and vector.
            values: (P, Vec<P>),

            /// Static label belonging to the receiver.
            label: &'static str,
        }

        // Only the placeholder structure is cloned; the array-containing receiver intentionally is not Clone.
        impl Clone for Inputs<Placeholder> {
            fn clone(&self) -> Self {
                Self { values: self.values.clone(), label: self.label }
            }
        }

        /// Exercises the public capability through an ordinary generic leaf bound.
        fn apply_leaf<V: Value + ConstrainSharding>(input: &V, sharding: &Sharding) -> Result<V, ProgramError> {
            input.constrain_sharding(sharding)
        }

        /// Exercises the same capability through a domain's generic value type.
        fn apply_domain<C: Domain<Value: ConstrainSharding>>(
            input: &C::Value,
            sharding: &Sharding,
        ) -> Result<C::Value, ProgramError> {
            input.constrain_sharding(sharding)
        }

        let mesh = mesh();
        let constraint = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let replicated = Sharding::replicated(mesh, 0);
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let scalar = Array::scalar(3.0f32).unwrap();
        let inputs = Inputs { values: (input.clone(), vec![scalar.clone()]), label: "input" };
        let shardings = Inputs { values: (constraint.clone(), vec![replicated.clone()]), label: "shardings" };
        let outputs = inputs.constrain_sharding(&shardings).unwrap();
        assert_eq!(outputs.label, "input");
        assert_eq!(outputs.values.0, input);
        assert_eq!(outputs.values.1[0], scalar);

        assert_eq!(apply_leaf(&input, &constraint), input.constrain_sharding(&constraint));
        assert_eq!(
            apply_domain::<EagerContext<Array, ArrayOperation<Array>>>(&input, &constraint),
            input.constrain_sharding(&constraint),
        );

        // Composite array members dispatch through the same leaf capability.
        let inputs = vec![ArrayIrValue::Array(input)];
        let expected = vec![inputs[0].constrain_sharding(&constraint).unwrap()];
        assert_eq!(inputs.constrain_sharding(&vec![constraint]).unwrap(), expected);

        assert!(Vec::<Array>::new().constrain_sharding(&Vec::new()).unwrap().is_empty());
        assert_eq!(<() as ConstrainSharding<Array>>::constrain_sharding(&(), &()), Ok(()));
    }

    #[test]
    fn test_constrain_sharding_structured_mismatched_parameter_paths() {
        let input = Array::scalar(1.0f32).unwrap();
        let sharding = Sharding::replicated(mesh(), 0);
        let missing = vec![input.clone(), input.clone()].constrain_sharding(&vec![sharding.clone()]);
        assert!(matches!(
            missing,
            Err(ProgramError::InvalidArgument { message, .. })
                if message == "receiver and `shardings` must have the same parameter structure",
        ));
        let extra = vec![input.clone()].constrain_sharding(&vec![sharding.clone(), sharding.clone()]);
        assert!(matches!(
            extra,
            Err(ProgramError::InvalidArgument { message, .. })
                if message == "receiver and `shardings` must have the same parameter structure",
        ));

        // Matching leaf counts do not imply matching paths.
        let inputs = BTreeMap::from([("left", input.clone())]);
        let shardings = BTreeMap::from([("right", sharding)]);
        assert!(matches!(
            inputs.constrain_sharding(&shardings),
            Err(ProgramError::InvalidArgument { message, .. })
                if message == "receiver and `shardings` must have the same parameter structure",
        ));

        // Structure validation happens before rank validation or leaf dispatch.
        let invalid = Sharding::replicated(mesh(), 1);
        let result = vec![input.clone(), input].constrain_sharding(&vec![invalid]);
        assert!(matches!(
            result,
            Err(ProgramError::InvalidArgument { message, .. })
                if message == "receiver and `shardings` must have the same parameter structure",
        ));
    }
}
