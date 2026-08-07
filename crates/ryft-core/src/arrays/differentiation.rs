use crate::arrays::dimensions::DimensionValue;
use crate::arrays::operations::DimensionOperation;
use crate::arrays::types::{ArrayIrType, ArrayType, Dimension, DimensionType, DimensionVariable, Shape};
use crate::axes::Axis;
use crate::batching::{ArrayBatching, ArrayBatchingPolicy, ArrayIrBatching, BatchingError};
use crate::contexts::Context;
use crate::differentiation::LinearCallBatchingPolicy;
use crate::operations::constants::ConstantOperation;
use crate::operations::dimensions::DimensionSizeOperation;
use crate::operations::math::{Reduce, ReduceOperation, ReductionKind};
use crate::programs::ProgramError;
use crate::programs::operations::OperationProjection;
use crate::programs::types::Typed;
use crate::programs::values::{Value, ValueProjection};
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

// TODO(eaplatanios): Review from here onwards.

impl<V: Value<Type = ArrayIrType>> LinearResiduals<V> {
    /// Creates an empty residual list.
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
    /// as the residual operand list when staging its [`LinearCallOperation`](crate::differentiation::LinearCallOperation).
    #[inline]
    pub fn into_values(self) -> Vec<V> {
        self.values
    }

    /// Retains `value` and returns the residual slot index that will address it inside the attached regions.
    ///
    /// When `value` is a dynamic dimension definition (i.e., its type is [`ArrayIrType::Dimension`] and that
    /// [`DimensionType`] has no concrete extent), retention deduplicates by identity: if a residual with the same
    /// [`DimensionVariable`] was already retained, its existing slot index is returned and `value` is dropped, since
    /// both values denote the same runtime extent. Every other value—ordinary arrays, and dimensions whose types pin
    /// a concrete extent and therefore carry no identity worth sharing—is appended to a fresh slot unconditionally,
    /// even when it compares equal to an already-retained value.
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
    pub fn retain_all(&mut self, values: impl IntoIterator<Item = V>) -> Vec<usize> {
        values.into_iter().map(|value| self.retain(value)).collect()
    }

    /// Retains the exact runtime shape of `array` and returns the [`ExactShape`] plan that lets an attached region
    /// reconstruct it from this list's residuals.
    ///
    /// Static axes contribute plan entries only and retain nothing. Each dynamic axis first looks for an
    /// already-retained dimension residual with the same [`DimensionVariable`] and reuses its slot; only identities
    /// not yet represented bind a [`DimensionSizeOperation`] read of `array` in `context` (the primal trace) and
    /// retain its result. Repeated identities within the shape therefore share one residual and one read.
    ///
    /// # Parameters
    ///
    ///   - `context`: [`Context`] that owns the primal trace being linearized, in which any required
    ///     [`DimensionSizeOperation`] reads are bound.
    ///   - `array`: [`ArrayType`]-typed value owned by `context` whose exact runtime shape must become available
    ///     inside the attached regions. Passing a non-array value fails with a kind-mismatch [`TypeError`](crate::programs::types::TypeError).
    pub fn retain_shape<C>(&mut self, context: &C, array: &V) -> Result<ExactShape, ProgramError>
    where
        C: Context<Type = ArrayIrType, Value = V, Operation: From<DimensionSizeOperation>>,
    {
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
                        return Ok(ExactShapeDimension::Residual(index));
                    }
                    let extent = context
                        .bind(DimensionSizeOperation::new(array_type, axis)?, Vec::new(), std::slice::from_ref(array))?
                        .remove(0);
                    Ok(ExactShapeDimension::Residual(self.retain(extent)))
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map(ExactShape)
    }
}

/// Exact runtime shape expressed in the coordinate system of a [`LinearResiduals`] list, so it can be reconstructed
/// inside a linear call's attached regions.
///
/// A [`Shape`] describes an array *type*: each axis is either a static extent or a [`DimensionVariable`] identity.
/// What a staged region needs is one step more concrete—*where the runtime extent of each axis lives*—and the only
/// values in scope there are the region's residual inputs. [`ExactShape`] is that plan: one [`ExactShapeDimension`]
/// per axis, referring to static extents directly and to dynamic extents by residual slot index. It is produced next
/// to the residual list that gives those indices meaning, by [`LinearResiduals::retain_shape`] during rule staging or
/// by [`Self::for_residual_zero`] when planning disconnected-cotangent zeros, and consumed inside regions after the
/// primal trace is out of reach.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactShape(Vec<ExactShapeDimension>);

impl ExactShape {
    /// Builds the canonical residual plan for constructing a zero of `shape` without any surrounding residual list,
    /// and returns it together with the source axes a caller must read to populate those residuals.
    ///
    /// This is the planning half of the disconnected-cotangent protocol shared by
    /// [`ResidualZeroProvider`](crate::differentiation::ResidualZeroProvider) and the dynamic-zero constructors: when
    /// a pullback input receives no cotangent, its zero must still be materialized with the primal input's exact
    /// runtime extents. Residual slots are assigned by first axis occurrence, and repeated uses of one dimension
    /// identity reuse the same slot, preserving equality between axes without retaining duplicate scalar values. The
    /// returned list contains one `(axis, variable)` entry per distinct dynamic identity, in slot order, telling the
    /// caller which source axis to read (e.g., with [`DimensionSizeOperation`]) to obtain each residual value.
    pub fn for_residual_zero(shape: &Shape) -> (Self, Vec<(usize, DimensionVariable)>) {
        let mut first_axes = Vec::new();
        // Residual slots are assigned by first axis occurrence. Repeated uses of one dimension identity reuse that
        // slot, preserving equality between axes without retaining duplicate scalar values.
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
    /// body): static axes stage a [`DimensionOperation`] constant, while dynamic axes clone the residual value their
    /// slot refers to. The result has exactly one value per axis, in axis order, ready to be consumed by operations
    /// that take one dimension operand per output axis.
    ///
    /// # Parameters
    ///
    ///   - `context`: [`Context`] in which static extents are staged as dimension constants.
    ///   - `residuals`: Residual values owned by `context`, indexed by this plan's residual slots (i.e., the region's
    ///     view of the [`LinearResiduals`] list this shape was built against).
    pub fn dimensions<C>(&self, context: &C, residuals: &[C::Value]) -> Result<Vec<C::Value>, ProgramError>
    where
        C: Context<
                Type = ArrayIrType,
                Operation: OperationProjection<DimensionType, Projected = DimensionOperation<DimensionValue>>,
            >,
    {
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

    /// Returns the residual values required by mixed dynamic array constructors, in dynamic-axis order.
    ///
    /// Constructors such as the dynamic zero consume one dimension operand per *dynamic* axis, in axis order, while
    /// this plan stores deduplicated residual slots. This method expands the plan back into that operand convention:
    /// static axes contribute nothing, and repeated identities intentionally produce repeated operands referring to
    /// the one shared residual value.
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

    /// Returns this exact shape viewed through an axis selection, so that output axis `i` is copied from source axis
    /// `axes[i]`. Rules whose transpose sees a permuted, repeated, or reduced view of a retained shape (e.g., an axis
    /// transposition) use this to derive the viewed plan without retaining any additional residuals; residual slot
    /// indices are preserved, so the result addresses the same [`LinearResiduals`] list as `self`.
    pub fn reordered(&self, axes: &[usize]) -> Self {
        Self(axes.iter().map(|axis| self.0[*axis]).collect())
    }
}

/// One axis of an [`ExactShape`]: where the runtime extent of that axis lives, from the point of view of a linear
/// call's attached region.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ExactShapeDimension {
    /// Compile-time extent that can be reconstructed as a dimension constant in either attached region.
    Static(usize),

    /// Index of the ordinary SSA residual that carries this dynamic extent.
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

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{ArrayIrOperation, ArrayIrValue, DataType, DimensionBounds};
    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::programs::types::TypeError;

    use super::*;

    /// Returns a fresh dynamic [`DimensionType`] (i.e., identity plus bounds, no concrete extent) named `name`.
    fn dimension_type(name: &str) -> DimensionType {
        DimensionType::new(DimensionVariable::new(name, DimensionBounds::new(1, Some(9)).unwrap()))
    }

    #[test]
    fn test_linear_residuals() {
        let n = dimension_type("n");
        let m = dimension_type("m");
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

        let n = dimension_type("n");
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
        let dimension = context.input(dimension_type("k").into());
        let mut residuals = LinearResiduals::new();
        assert_eq!(
            residuals.retain_shape(&context, &dimension),
            Err(TypeError::invalid("expected array type but got dimension type").into()),
        );
    }

    #[test]
    fn test_exact_shape_for_residual_zero() {
        let n = dimension_type("n");
        let m = dimension_type("m");
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

        // Reordering copies output axis `i` from source axis `axes[i]`, preserving residual slot indices.
        assert_eq!(
            plan.reordered(&[4, 0, 1]),
            ExactShape(vec![
                ExactShapeDimension::Residual(1),
                ExactShapeDimension::Static(2),
                ExactShapeDimension::Residual(0),
            ]),
        );
    }

    #[test]
    fn test_exact_shape_dimensions() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let n = dimension_type("n");
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
        let dimension = context.input(dimension_type("k").into());
        assert!(matches!(
            <ArrayIrBatching as LinearCallBatchingPolicy<TestContext>>::sum_mapped_cotangents(
                &context,
                dimension,
                Axis::from(0),
            ),
            Err(BatchingError::Type(error)) if error == TypeError::invalid("expected array type but got dimension type"),
        ));
    }
}
