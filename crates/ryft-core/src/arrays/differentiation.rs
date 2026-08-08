use crate::arrays::batching::{ArrayBatching, ArrayBatchingPolicy, ArrayIrBatching};
use crate::arrays::dimensions::DimensionValue;
use crate::arrays::operations::DimensionOperation;
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::dimensions::{Dimension, DimensionType, DimensionVariable, Shape};
use crate::arrays::types::ir::ArrayIrType;
use crate::axes::Axis;
use crate::batching::BatchingError;
use crate::contexts::{Context, ProjectedContext};
use crate::differentiation::{DifferentiationDual, DifferentiationError, LinearCallBatchingPolicy};
use crate::macros::check_count;
use crate::operations::{
    ConstantOperation, DimensionSizeOperation, Permutation, Reduce, ReduceOperation, ReductionKind, Zero,
    ZeroLikeOperation, ZeroOperation,
};
use crate::programs::{MaybeZero, OperationProjection, ProgramError, Type, Typed, Value, ValueProjection};
use crate::tracing::{Tracer, TracingContext};

/// Ordered residual list accumulated by an extent-sensitive linearization rule (e.g., for slice, reshape, pad, reduce
/// gather, or a shape-changing collective) while it stages a [`LinearCallOperation`](crate::LinearCallOperation).
///
/// A linear call's attached forward and transpose [`Region`](crate::Region)s later run without access to the primal
/// trace, so everything they need from it must cross the call boundary as ordinary trailing Single Static Assignment
/// (SSA) operands, called _residuals_. A rule retains values one by one while building its regions, remembers the
/// returned indices, and finally passes [`Self::into_values`] as the staged linear call's residual operand list.
/// Inside a region, the same indices address the region's residual inputs.
///
/// The most important residuals in the array universe are exact runtime extents. A transpose region typically has to
/// construct values with the exact shape of a primal _operand_ (e.g., the zero-padded cotangent of a slice), and that
/// shape is neither recoverable from the region's cotangent inputs nor from any ambient side channel, because runtime
/// dimensions are ordinary Single Static Assignment (SSA) values. [`Self::retain_shape`] reads such extents from primal
/// arrays with [`DimensionSizeOperation`] on demand, and [`ExactShape`] is the compile-time plan that lets a region
/// rebuild an exact shape from the retained residuals.
///
/// Dynamic dimension definitions are deduplicated by identity. Retaining a dimension-typed value whose
/// [`DimensionType`] carries no concrete extent reuses the slot of any previously retained residual with the same
/// [`DimensionVariable`], because a variable that appears several times (across axes, or as both an axis and an
/// explicit dimension operand) denotes one runtime extent. This keeps operand lists minimal and, more importantly,
/// preserves the type-level equality between axes when shapes are reconstructed inside the attached regions. All
/// other values (i.e., ordinary arrays and dimensions whose types already pin a concrete extent) are purely positional:
/// every retention appends a new slot, and the values themselves are never inspected.
#[derive(Clone, Debug)]
pub struct LinearResiduals<V: Value<Type = ArrayIrType>> {
    /// Retained residual [`Value`]s, in the trailing-operand order of the staged linear call. Indices returned by the
    /// retention methods point into this list and stay valid because the list is append-only.
    values: Vec<V>,
}

impl<V: Value<Type = ArrayIrType>> LinearResiduals<V> {
    /// Creates a new empty [`LinearResiduals`] instance.
    #[inline]
    pub fn new() -> Self {
        Self { values: Vec::new() }
    }

    /// Returns the retained residual values, in residual-slot order.
    #[inline]
    pub fn values(&self) -> &[V] {
        self.values.as_slice()
    }

    /// Consumes this residual list and returns its values, in residual-slot order. The result is what a rule passes
    /// as the residual operand list when staging its [`LinearCallOperation`](crate::LinearCallOperation).
    #[inline]
    pub fn into_values(self) -> Vec<V> {
        self.values
    }

    /// Retains `value` and returns the residual slot index that will address it inside the attached
    /// [`Region`](crate::Region)s. When `value` is a dynamic dimension definition (i.e., its type is
    /// [`ArrayIrType::Dimension`] and that [`DimensionType`] has no concrete extent), retention deduplicates by
    /// identity: if a residual with the same [`DimensionVariable`] was already retained, its existing slot index is
    /// returned and `value` is dropped, since both values denote the same runtime extent. Every other value (i.e.,
    /// ordinary arrays, and dimensions whose types pin a concrete extent and therefore carry no identity worth sharing)
    /// is appended to a fresh slot unconditionally, even when it compares equal to an already-retained value.
    pub fn retain(&mut self, value: V) -> usize {
        if let ArrayIrType::Dimension(r#type) = value.r#type().as_ref()
            && r#type.extent().is_none()
            && let Some(index) = self.values.iter().position(|value| {
                matches!(
                    value.r#type().as_ref(),
                    ArrayIrType::Dimension(candidate) if candidate.variable() == r#type.variable()
                )
            })
        {
            return index;
        }
        let index = self.values.len();
        self.values.push(value);
        index
    }

    /// Retains an ordered value list and returns the residual slot index corresponding to each source value, applying
    /// the [`Self::retain`] deduplication rule value by value (so two source values may map to one shared slot).
    #[inline]
    pub fn retain_all<I: IntoIterator<Item = V>>(&mut self, values: I) -> Vec<usize> {
        values.into_iter().map(|value| self.retain(value)).collect()
    }

    /// Retains the exact runtime shape of `array` and returns the [`ExactShape`] plan that lets an attached region
    /// reconstruct it from the residual values in this [`LinearResiduals`] instance. Static axes contribute plan
    /// entries only and retain nothing. Each dynamic axis first looks for an already-retained dimension residual
    /// with the same [`DimensionVariable`] and reuses its slot. Only identities not yet represented bind a
    /// [`DimensionSizeOperation`] read of `array` in `context` (i.e., the primal trace) and retain its result.
    /// Repeated identities within the shape therefore share one residual and one read.
    ///
    /// # Parameters
    ///
    ///   - `context`: [`Context`] that owns the primal trace being linearized, in which any required
    ///     [`DimensionSizeOperation`] reads are bound.
    ///   - `array`: [`ArrayType`]-typed value owned by `context` whose exact runtime shape must become available inside
    ///     the attached regions. Passing a non-array value fails with a kind-mismatch [`TypeError`](crate::TypeError).
    pub fn retain_shape<C: Context<Type = ArrayIrType, Value = V, Operation: From<DimensionSizeOperation>>>(
        &mut self,
        context: &C,
        array: &V,
    ) -> Result<ExactShape, ProgramError> {
        let array_type = array.r#type();
        let array_type = <&ArrayType>::try_from(array_type.as_ref())?;
        array_type
            .shape()
            .dimensions()
            .iter()
            .enumerate()
            .map(|(axis, dimension)| match dimension {
                Dimension::Static(extent) => Ok(ExactShapeDimension::Static(*extent)),
                Dimension::Dynamic(variable) => {
                    if let Some(index) = self.values.iter().position(|value| {
                        matches!(
                            value.r#type().as_ref(),
                            ArrayIrType::Dimension(r#type) if r#type.variable() == variable
                        )
                    }) {
                        Ok(ExactShapeDimension::Residual(index))
                    } else {
                        Ok(ExactShapeDimension::Residual(
                            self.retain(
                                context
                                    .bind(
                                        DimensionSizeOperation::new(array_type, axis)?,
                                        Vec::new(),
                                        std::slice::from_ref(array),
                                    )?
                                    .remove(0),
                            ),
                        ))
                    }
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map(ExactShape)
    }
}

impl<V: Value<Type = ArrayIrType>> Default for LinearResiduals<V> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Exact runtime [`Shape`] expressed in the coordinate system of a [`LinearResiduals`] list, so that it can be
/// reconstructed inside a [`LinearCallOperation`](crate::LinearCallOperation)'s attached [`Region`](crate::Region)s.
/// A [`Shape`] describes an array _type_: each axis is either a static extent or a [`DimensionVariable`] identity.
/// What a staged region needs is one step more concrete (i.e., where the runtime extent of each axis lives) and the
/// only values in scope there are the region's residual inputs. [`ExactShape`] is that plan, containing one
/// [`ExactShapeDimension`] per axis, referring to static extents directly and to dynamic extents by residual slot
/// index. It is produced by [`LinearResiduals::retain_shape`] next to the residual list that gives those indices
/// meaning during rule staging or by [`Self::for_residual_zero`] when planning disconnected-cotangent zeros, and
/// consumed inside regions after the primal trace is out of reach.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactShape(Vec<ExactShapeDimension>);

impl ExactShape {
    /// Builds the canonical [`ExactShape`] plan for constructing a zero of [`Shape`] `shape` without any surrounding
    /// [`LinearResiduals`] list, and returns it together with the source axes a caller must read to populate those
    /// residuals. This is the planning half of the disconnected-cotangent protocol shared by
    /// [`ResidualZeroProvider`](crate::ResidualZeroProvider) and the dynamic-zero constructors: when a pullback input
    /// receives no cotangent, its zero must still be materialized with the primal input's exact runtime extents.
    /// Residual slots are assigned by first axis occurrence, and repeated uses of one dimension identity reuse the same
    /// slot, preserving equality between axes without retaining duplicate scalar values. The returned list contains one
    /// `(axis, variable)` entry per distinct dynamic identity, in slot order, telling the caller which source axis to
    /// read (e.g., with [`DimensionSizeOperation`]) to obtain each residual value.
    pub fn for_residual_zero(shape: &Shape) -> (Self, Vec<(usize, DimensionVariable)>) {
        // Residual slots are assigned by first axis occurrence. Repeated uses of one dimension identity reuse
        // that slot, preserving equality between axes without retaining duplicate scalar values.
        let mut first_axes = Vec::new();
        let dimensions = shape
            .dimensions()
            .iter()
            .enumerate()
            .map(|(axis, dimension)| match dimension {
                Dimension::Static(extent) => ExactShapeDimension::Static(*extent),
                Dimension::Dynamic(variable) => {
                    let residual =
                        first_axes.iter().position(|(_, candidate)| candidate == variable).unwrap_or_else(|| {
                            let residual = first_axes.len();
                            first_axes.push((axis, variable.clone()));
                            residual
                        });
                    ExactShapeDimension::Residual(residual)
                }
            })
            .collect();
        (Self(dimensions), first_axes)
    }

    /// Materializes one first-class dimension value per axis of this shape in `context` (typically an attached region
    /// body). Static axes stage a [`DimensionOperation`] constant, while dynamic axes clone the residual value their
    /// slot refers to. The result has exactly one value per axis, in axis order, ready to be consumed by operations
    /// that take one dimension operand per output axis.
    ///
    /// # Parameters
    ///
    ///   - `context`: [`Context`] in which static extents are staged as dimension constants.
    ///   - `residuals`: Residual values owned by `context`, indexed by this plan's residual slots (i.e., the region's
    ///     view of the [`LinearResiduals`] list this shape was built against).
    pub fn dimensions<
        C: Context<
                Type = ArrayIrType,
                Operation: OperationProjection<DimensionType, Projected = DimensionOperation<DimensionValue>>,
            >,
    >(
        &self,
        context: &C,
        residuals: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        self.0
            .iter()
            .map(|dimension| match dimension {
                ExactShapeDimension::Static(extent) => Ok(context
                    .bind(
                        DimensionOperation::from(ConstantOperation::new(DimensionValue::constant(*extent)?)),
                        Vec::new(),
                        &[],
                    )?
                    .remove(0)),
                ExactShapeDimension::Residual(index) => Ok(residuals[*index].clone()),
            })
            .collect()
    }

    /// Returns the residual values required by mixed dynamic array constructors, in dynamic-axis order. Constructors
    /// such as the dynamic zero consume one dimension operand per _dynamic_ axis, in axis order, while this plan stores
    /// deduplicated residual slots. This method expands the plan back into that operand convention: static axes
    /// contribute nothing, and repeated identities intentionally produce repeated operands referring to the one
    /// shared residual value.
    ///
    /// # Parameters
    ///
    ///   - `residuals`: Residual values indexed by this plan's residual slots.
    pub fn dynamic_dimensions<V: Clone>(&self, residuals: &[V]) -> Vec<V> {
        // Mixed array constructors consume one operand per dynamic axis. Expand deduplicated residual slots back into
        // axis order here, so repeated identities intentionally produce repeated operands.
        self.0
            .iter()
            .filter_map(|dimension| match dimension {
                ExactShapeDimension::Static(_) => None,
                ExactShapeDimension::Residual(index) => Some(residuals[*index].clone()),
            })
            .collect()
    }

    /// Returns this exact shape transposed by `permutation`, so that output axis `i` is copied from source axis
    /// `permutation[i]`. Rules whose transpose sees a permuted view of a retained shape (e.g., a reshape with a
    /// `dimensions` permutation) use this to derive that view without retaining any additional residuals. Residual
    /// slot indices are preserved, so the result addresses the same [`LinearResiduals`] list as `self`.
    #[inline]
    pub fn transposed(&self, permutation: &Permutation) -> Self {
        Self(permutation.iter().map(|axis| self.0[*axis]).collect())
    }
}

/// One dimension of an [`ExactShape`] that describes where the runtime extent of the corresponding axis lives, from
/// the point of view of a [`LinearCallOperation`](crate::LinearCallOperation)'s attached [`Region`](crate::Region).
/// This is the [`ExactShape`] counterpart of [`Dimension`], which is the per-axis entry of a [`Shape`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ExactShapeDimension {
    /// Compile-time extent that can be reconstructed as a dimension constant in either attached region.
    Static(usize),

    /// Index of the ordinary Single Static Assignment (SSA) residual that carries this dynamic extent.
    Residual(usize),
}

impl<C: Context<Type = ArrayType, Operation: From<ReduceOperation>>, P: ArrayBatchingPolicy<C>>
    LinearCallBatchingPolicy<C> for ArrayBatching<P>
{
    fn sum_mapped_cotangents(
        _context: &TracingContext<C::Constant, C::Operation>,
        cotangent: Tracer<TracingContext<C::Constant, C::Operation>>,
        axis: Axis,
    ) -> Result<Tracer<TracingContext<C::Constant, C::Operation>>, BatchingError> {
        let axis = axis.normalize(cotangent.r#type().rank()).map_err(|_| BatchingError::BatchAxisOutOfBounds {
            r#type: Box::new(cotangent.r#type().into_owned()),
            axis,
        })?;
        Ok(cotangent.reduce(&[axis], ReductionKind::Sum))
    }
}

impl<
    C: Context<
            Type = ArrayIrType,
            Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            Operation: OperationProjection<ArrayType, Projected: From<ReduceOperation>>,
        >,
> LinearCallBatchingPolicy<C> for ArrayIrBatching
{
    fn sum_mapped_cotangents(
        _context: &TracingContext<C::Constant, C::Operation>,
        cotangent: Tracer<TracingContext<C::Constant, C::Operation>>,
        axis: Axis,
    ) -> Result<Tracer<TracingContext<C::Constant, C::Operation>>, BatchingError> {
        // Projecting the replayed array cotangent gives it the ordinary `Reduce` capability,
        // whose staged operation lifts back through the composite operation family.
        let cotangent = ValueProjection::<ArrayType>::into_projected(cotangent)?;
        let axis = axis.normalize(cotangent.r#type().rank()).map_err(|_| BatchingError::BatchAxisOutOfBounds {
            r#type: Box::new(cotangent.r#type().into_owned()),
            axis,
        })?;
        Ok(ValueProjection::from_projected(cotangent.reduce(&[axis], ReductionKind::Sum)))
    }
}

/// Materializes one array operand's forward-mode tangent as a concrete projected array value, using the operand's
/// primal as the runtime-geometry exemplar whenever the tangent type cannot supply that geometry itself. A mixed array
/// rule that has to hand a concrete tangent to a staged operation cannot always materialize a [`MaybeZero::Zero`] from
/// its type: an identity-bearing [`ArrayType`] names its dynamic extents by [`DimensionVariable`] rather than pinning
/// them, so the type-only nullary [`ZeroOperation`] is unconstructible. The primal is a live value of exactly the
/// operand's shape, so [`ZeroLikeOperation`] over it produces the same zero with runtime geometry and no extra
/// residual. Identity-free tangent types keep the canonical nullary zero, whose zero-producing marker keeps
/// higher-order partial evaluation structural.
///
/// The exemplar is used only when the primal's type already equals the tangent type. Array families whose tangent
/// representation widens the element type (e.g., `f8e8m0fnu`) keep the nullary path, which their identity-free types
/// support; the widening exemplar path belongs to the generic projected-member dispatch rather than to an individual
/// mixed rule.
///
/// # Parameters
///
///   - `context`: Projected array view of the active mixed [`Context`] in which the zero is staged.
///   - `input`: Forward-mode dual whose tangent is materialized and whose primal is the geometry exemplar.
pub fn materialize_array_tangent<
    C: Context<
            Type = ArrayIrType,
            Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            Operation: OperationProjection<
                ArrayType,
                Projected: From<ZeroOperation<ArrayType>> + From<ZeroLikeOperation<ArrayType>>,
            >,
        >,
>(
    context: &ProjectedContext<C, ArrayType>,
    input: &DifferentiationDual<C::Value>,
) -> Result<<C::Value as ValueProjection<ArrayType>>::Projected, DifferentiationError> {
    let tangent_type = match input.tangent() {
        MaybeZero::Value(value) => {
            return Ok(<C::Value as ValueProjection<ArrayType>>::into_projected(value.clone())?);
        }
        MaybeZero::Zero(r#type) => <&ArrayType>::try_from(r#type)?.clone(),
    };
    let primal = <C::Value as ValueProjection<ArrayType>>::into_projected(input.primal().clone())?;
    if tangent_type.identities().next().is_some() && primal.r#type().as_ref() == &tangent_type {
        let mut zero = context.bind(ZeroLikeOperation::new(), Vec::new(), std::slice::from_ref(&primal))?;
        check_count!("output", zero, 1, ProgramError);
        return Ok(zero.remove(0));
    }
    Ok(context.zero(&tangent_type)?)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::Array;
    use crate::arrays::ir::ArrayIrValue;
    use crate::arrays::operations::{ArrayIrOperation, ArrayOperation};
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::DimensionBounds;
    use crate::contexts::{EagerContext, StagingContext};
    use crate::programs::TypeError;

    use super::*;

    #[test]
    fn test_linear_residuals() {
        let n = DimensionType::new(DimensionVariable::new("n", DimensionBounds::new(1, Some(9)).unwrap()));
        let m = DimensionType::new(DimensionVariable::new("m", DimensionBounds::new(1, Some(9)).unwrap()));
        let mut residuals = LinearResiduals::<ArrayIrValue<Array>>::new();
        assert!(residuals.values().is_empty());

        // Dynamic dimension definitions share one slot per identity, keyed by variable rather than by value, so a
        // repeated identity reuses its slot even when the retained value instance differs.
        let n_value = ArrayIrValue::<Array>::Dimension(DimensionValue::new(n.clone(), 4).unwrap());
        assert_eq!(residuals.retain(n_value.clone()), 0);
        assert_eq!(residuals.retain(ArrayIrValue::Dimension(DimensionValue::new(n, 5).unwrap())), 0);
        let m_value = ArrayIrValue::<Array>::Dimension(DimensionValue::new(m, 2).unwrap());
        assert_eq!(residuals.retain(m_value.clone()), 1);

        // Ordinary array residuals stay positional, so equal arrays still occupy distinct slots.
        let array = ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0]));
        assert_eq!(residuals.retain(array.clone()), 2);
        assert_eq!(residuals.retain(array), 3);

        // A dimension whose type pins a concrete extent carries no shareable identity and is never deduplicated.
        let constant = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(3).unwrap());
        assert_eq!(residuals.retain(constant.clone()), 4);
        assert_eq!(residuals.retain(constant), 5);

        // `retain_all` maps each source value to its (possibly shared) slot, in source order.
        assert_eq!(residuals.retain_all(vec![m_value, n_value.clone()]), vec![1, 0]);

        // The retained list keeps slot order, and a deduplicated slot keeps the first value retained for it.
        assert_eq!(residuals.values().len(), 6);
        assert_eq!(residuals.values()[0], n_value);
        assert_eq!(residuals.into_values().len(), 6);
    }

    #[test]
    fn test_linear_residuals_retain_shape() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let n = DimensionType::new(DimensionVariable::new("n", DimensionBounds::new(1, Some(9)).unwrap()));
        let array_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![
                Dimension::Static(2),
                Dimension::Dynamic(n.variable().clone()),
                Dimension::Dynamic(n.variable().clone()),
            ]),
        );

        // Reading an exact shape binds one `DimensionSize` read per distinct dynamic identity: the static axis
        // contributes a plan entry only, and the repeated identity reuses the first read's residual slot.
        let context = TestContext::new();
        let array = context.input(array_type.clone().into());
        let mut residuals = LinearResiduals::new();
        let shape = residuals.retain_shape(&context, &array).unwrap();
        assert_eq!(
            shape,
            ExactShape(vec![
                ExactShapeDimension::Static(2),
                ExactShapeDimension::Residual(0),
                ExactShapeDimension::Residual(0),
            ]),
        );
        assert_eq!(residuals.values().len(), 1);
        let builder = context.builder().borrow();
        let [instruction] = builder.instructions() else {
            panic!("expected exactly one dimension-size read");
        };
        assert!(matches!(instruction.operation(), ArrayIrOperation::DimensionSize(_)));
        drop(builder);

        // An already-retained residual with the same identity is reused without binding another read.
        let context = TestContext::new();
        let array = context.input(array_type.into());
        let dimension = context.input(n.into());
        let mut residuals = LinearResiduals::new();
        assert_eq!(residuals.retain(dimension), 0);
        let shape = residuals.retain_shape(&context, &array).unwrap();
        assert_eq!(
            shape,
            ExactShape(vec![
                ExactShapeDimension::Static(2),
                ExactShapeDimension::Residual(0),
                ExactShapeDimension::Residual(0),
            ]),
        );
        assert_eq!(residuals.values().len(), 1);
        assert!(context.builder().borrow().instructions().is_empty());

        // Non-array values are rejected with a kind mismatch.
        let context = TestContext::new();
        let dimension = context
            .input(DimensionType::new(DimensionVariable::new("k", DimensionBounds::new(1, Some(9)).unwrap())).into());
        let mut residuals = LinearResiduals::new();
        assert_eq!(
            residuals.retain_shape(&context, &dimension),
            Err(TypeError::invalid("expected array type but got dimension type").into()),
        );
    }

    #[test]
    fn test_exact_shape_for_residual_zero() {
        let n = DimensionType::new(DimensionVariable::new("n", DimensionBounds::new(1, Some(9)).unwrap()));
        let m = DimensionType::new(DimensionVariable::new("m", DimensionBounds::new(1, Some(9)).unwrap()));
        let shape = Shape::new(vec![
            Dimension::Static(2),
            Dimension::Dynamic(n.variable().clone()),
            Dimension::Static(3),
            Dimension::Dynamic(n.variable().clone()),
            Dimension::Dynamic(m.variable().clone()),
        ]);

        // Slots are assigned by first occurrence and the repeated identity reuses slot 0, while the first-axes list
        // names the source axis to read for each distinct identity, in slot order.
        let (plan, first_axes) = ExactShape::for_residual_zero(&shape);
        assert_eq!(
            plan,
            ExactShape(vec![
                ExactShapeDimension::Static(2),
                ExactShapeDimension::Residual(0),
                ExactShapeDimension::Static(3),
                ExactShapeDimension::Residual(0),
                ExactShapeDimension::Residual(1),
            ]),
        );
        assert_eq!(first_axes, vec![(1, n.variable().clone()), (4, m.variable().clone())]);

        // Dynamic-constructor operand expansion is in axis order and intentionally repeats shared slots.
        assert_eq!(plan.dynamic_dimensions(&["n", "m"]), vec!["n", "n", "m"]);

        // Transposing copies output axis `i` from source axis `permutation[i]`, preserving residual slot indices.
        assert_eq!(
            plan.transposed(&Permutation::from(vec![4, 0, 1, 2, 3])),
            ExactShape(vec![
                ExactShapeDimension::Residual(1),
                ExactShapeDimension::Static(2),
                ExactShapeDimension::Residual(0),
                ExactShapeDimension::Static(3),
                ExactShapeDimension::Residual(0),
            ]),
        );
    }

    #[test]
    fn test_exact_shape_dimensions() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let n = DimensionType::new(DimensionVariable::new("n", DimensionBounds::new(1, Some(9)).unwrap()));
        let shape = Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(n.variable().clone())]);
        let (plan, _) = ExactShape::for_residual_zero(&shape);

        // Static axes stage dimension constants, while dynamic axes reuse the residual values without staging
        // anything new.
        let context = TestContext::new();
        let residual = context.input(n.into());
        let dimensions = plan.dimensions(&context, std::slice::from_ref(&residual)).unwrap();
        let [static_dimension, dynamic_dimension] = dimensions.as_slice() else {
            panic!("expected one dimension value per axis");
        };
        assert!(matches!(
            static_dimension.r#type().as_ref(),
            ArrayIrType::Dimension(r#type) if r#type.extent() == Some(2),
        ));
        assert_eq!(dynamic_dimension.atom_id().unwrap(), residual.atom_id().unwrap());
        let builder = context.builder().borrow();
        let [instruction] = builder.instructions() else {
            panic!("expected exactly one staged dimension constant");
        };
        assert!(matches!(instruction.operation(), ArrayIrOperation::Dimension(DimensionOperation::Constant(_))));
    }

    #[test]
    fn test_array_batching_sum_mapped_cotangents() {
        type TestContext = EagerContext<Array, ArrayOperation<Array>>;

        // The array policy reduce-sums the packed per-item cotangents along the mapped axis, dropping that axis.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let cotangent_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let cotangent = context.input(cotangent_type);
        let summed = <ArrayBatching as LinearCallBatchingPolicy<TestContext>>::sum_mapped_cotangents(
            &context,
            cotangent.clone(),
            Axis::from(0),
        )
        .unwrap();
        assert_eq!(summed.r#type().as_ref(), &ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])));
        let builder = context.builder().borrow();
        let [instruction] = builder.instructions() else {
            panic!("expected exactly one staged reduction");
        };
        assert!(matches!(instruction.operation(), ArrayOperation::Reduce(_)));
        drop(builder);

        // An axis outside the cotangent's rank is rejected.
        assert!(matches!(
            <ArrayBatching as LinearCallBatchingPolicy<TestContext>>::sum_mapped_cotangents(
                &context,
                cotangent,
                Axis::from(5),
            ),
            Err(BatchingError::BatchAxisOutOfBounds { axis, .. }) if axis == Axis::from(5),
        ));
    }

    #[test]
    fn test_array_ir_batching_sum_mapped_cotangents() {
        type TestContext = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        // The composite policy projects the replayed cotangent to its array member, reduce-sums along the (here
        // negative and normalized) mapped axis, and lifts the sum back into the composite family.
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let cotangent_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let cotangent = context.input(cotangent_type.into());
        let summed = <ArrayIrBatching as LinearCallBatchingPolicy<TestContext>>::sum_mapped_cotangents(
            &context,
            cotangent,
            Axis::from(-2),
        )
        .unwrap();
        assert_eq!(
            summed.r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]))),
        );
        let builder = context.builder().borrow();
        let [instruction] = builder.instructions() else {
            panic!("expected exactly one staged reduction");
        };
        assert!(matches!(instruction.operation(), ArrayIrOperation::Array(ArrayOperation::Reduce(_))));
        drop(builder);

        // Dimension-typed cotangents cannot be projected to the array member.
        let dimension = context
            .input(DimensionType::new(DimensionVariable::new("k", DimensionBounds::new(1, Some(9)).unwrap())).into());
        assert!(matches!(
            <ArrayIrBatching as LinearCallBatchingPolicy<TestContext>>::sum_mapped_cotangents(
                &context,
                dimension,
                Axis::from(0),
            ),
            Err(BatchingError::Type(error)) if error == TypeError::invalid("expected array type but got dimension type"),
        ));
    }

    #[test]
    fn test_materialize_array_tangent() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let n = DimensionType::new(DimensionVariable::new("n", DimensionBounds::new(1, Some(9)).unwrap()));
        let dynamic_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(n.variable().clone())]));
        let static_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let context = TestContext::new();
        let projected_context = ProjectedContext::<TestContext, ArrayType>::new(context.clone());
        let primal = context.input(dynamic_type.clone().into());

        // A concrete tangent is projected and returned unchanged, staging nothing.
        let tangent = context.input(dynamic_type.clone().into());
        let input = DifferentiationDual::new(primal.clone(), MaybeZero::Value(tangent.clone())).unwrap();
        let materialized = materialize_array_tangent(&projected_context, &input).unwrap();
        assert_eq!(materialized.value().atom_id().unwrap(), tangent.atom_id().unwrap());
        assert!(context.builder().borrow().instructions().is_empty());

        // A structural zero whose identity-bearing type matches the primal's stages one zero-like over the primal
        // exemplar, because the type-only nullary zero cannot supply the runtime extent that `n` names.
        let input = DifferentiationDual::new(primal.clone(), MaybeZero::Zero(ArrayIrType::Array(dynamic_type.clone())))
            .unwrap();
        let materialized = materialize_array_tangent(&projected_context, &input).unwrap();
        assert_eq!(materialized.r#type().as_ref(), &dynamic_type);
        {
            let builder = context.builder().borrow();
            let [instruction] = builder.instructions() else {
                panic!("expected exactly one staged zero-like");
            };
            assert!(matches!(instruction.operation(), ArrayIrOperation::Array(ArrayOperation::ZeroLike(_))));
        }

        // An identity-free structural zero keeps the canonical nullary zero, whose zero-producing marker keeps
        // higher-order partial evaluation structural.
        let static_primal = context.input(static_type.clone().into());
        let input =
            DifferentiationDual::new(static_primal, MaybeZero::Zero(ArrayIrType::Array(static_type.clone()))).unwrap();
        let materialized = materialize_array_tangent(&projected_context, &input).unwrap();
        assert_eq!(materialized.r#type().as_ref(), &static_type);
        {
            let builder = context.builder().borrow();
            let [_, instruction] = builder.instructions() else {
                panic!("expected the staged zero-like followed by one staged nullary zero");
            };
            assert!(matches!(instruction.operation(), ArrayIrOperation::Array(ArrayOperation::Zero(_))));
        }

        // An identity-bearing zero whose type differs from the primal's cannot use the exemplar and fails as the
        // unconstructible nullary zero of an identity-referencing type.
        let m = DimensionType::new(DimensionVariable::new("m", DimensionBounds::new(1, Some(9)).unwrap()));
        let mismatched_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(m.variable().clone())]));
        let input = DifferentiationDual::new(primal, MaybeZero::Zero(ArrayIrType::Array(mismatched_type))).unwrap();
        assert!(matches!(
            materialize_array_tangent(&projected_context, &input),
            Err(DifferentiationError::Program(error))
                if error.to_string().contains("'zero' cannot construct type f64[m] without operands"),
        ));
    }
}
