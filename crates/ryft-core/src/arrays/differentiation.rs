use std::borrow::Cow;

use crate::arrays::batching::{ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrBatchingPolicy};
use crate::arrays::dimensions::DimensionValue;
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::dimensions::{Dimension, DimensionType, DimensionVariable, Shape};
use crate::arrays::types::ir::ArrayIrType;
use crate::axes::Axis;
use crate::batching::BatchingError;
use crate::contexts::{Context, ProjectedContext};
use crate::differentiation::{
    CotangentBatchingPolicy, DifferentiationDual, DifferentiationError, ResidualZeroProvider,
};
use crate::operations::{
    ConstantOperation, DimensionSizeOperation, Permutation, Reduce, ReduceOperation, ReductionKind,
    ReferenceReadOperation, Zero, ZeroOperation,
};
use crate::programs::{
    AtomId, Operation, OperationProjection, OperationProvider, ProgramBuilder, ProgramError, Typed, Value,
    ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

// Homogeneous array operations construct input-free zeros from their output types. We keep this blanket implementation
// specific to `ArrayType` so that it is disjoint from the runtime-extent protocol for `ArrayIrType` below.
impl<O: Operation<Type = ArrayType> + From<ZeroOperation<ArrayType>>> ResidualZeroProvider<ArrayType> for O {}

// Array-IR operation families share declaration, capture, and assembly regardless of their backend representation.
// Static zeros use the family's `OperationProvider`. Dynamic zeros consume one extent operand per dynamic axis.
// Captured residuals contain one extent per distinct dimension identity, in first-occurrence order. Assembly expands
// repeated identities back into the constructor's per-axis operand order. Builder-level capture reads each identity's
// first axis from the ordinary primal array. Value-level capture also accepts first-class dimensions and references
// (i.e., a dimension with the requested identity is reused, an array supplies a `DimensionSizeOperation` read, and a
// reference is read before obtaining its referent's extent). A source is inspected before staging anything, so a
// candidate without the requested identity leaves no instructions behind.
impl<O> ResidualZeroProvider<ArrayIrType> for O
where
    O: Operation<Type = ArrayIrType>
        + OperationProvider<ArrayIrType, ZeroOperation<ArrayIrType>, Operation = O>
        + From<ZeroOperation<ArrayType>>
        + From<DimensionSizeOperation>
        + From<ReferenceReadOperation<ArrayType, ArrayIrType>>,
{
    #[inline]
    fn zero_residual_types(r#type: &ArrayIrType) -> Vec<ArrayIrType> {
        match r#type {
            ArrayIrType::Array(r#type) => ExactShape::for_residual_zero(r#type.shape())
                .1
                .into_iter()
                .map(|(_, variable)| DimensionType::new(variable).into())
                .collect(),
            ArrayIrType::Dimension(_) | ArrayIrType::Reference(_) => Vec::new(),
        }
    }

    #[inline]
    fn capture_zero_residuals<V: Value<Type = ArrayIrType>>(
        builder: &mut ProgramBuilder<V, O>,
        source: AtomId,
        r#type: &ArrayIrType,
    ) -> Result<Vec<AtomId>, ProgramError> {
        // Reading only each identity's first source axis establishes the same deduplicated residual ordering
        // used by zero construction, including when several axes share one identity.
        let r#type = <&ArrayType>::try_from(r#type)?;
        let (_, first_axes) = ExactShape::for_residual_zero(r#type.shape());
        first_axes
            .into_iter()
            .map(|(axis, _)| {
                Ok(builder.add_instruction(
                    DimensionSizeOperation::new(r#type, axis)?,
                    Vec::new(),
                    vec![source],
                    None,
                )?[0])
            })
            .collect()
    }

    #[inline]
    fn capture_zero_residual_value<C: Context<Type = ArrayIrType, Operation = O>>(
        context: &C,
        source: &C::Value,
        residual_type: &ArrayIrType,
    ) -> Result<Option<C::Value>, ProgramError> {
        let ArrayIrType::Dimension(residual_type) = residual_type else {
            return Ok(None);
        };

        let variable = residual_type.variable();
        let source_type = source.r#type();
        let array_type = match source_type.as_ref() {
            ArrayIrType::Dimension(source_type) => {
                // A matching first-class dimension already is the residual and so we reuse it without staging a read.
                return Ok((source_type.variable() == variable).then(|| source.clone()));
            }
            ArrayIrType::Reference(reference) => reference.referent(),
            ArrayIrType::Array(array_type) => array_type,
        };

        // Match the dimension identity before reading a reference or staging an extent query. The caller tries
        // candidate sources in order, so an unrelated candidate must leave the program unchanged.
        let Some(axis) = array_type
            .shape()
            .dimensions()
            .iter()
            .position(|dimension| matches!(dimension, Dimension::Dynamic(candidate) if candidate == variable))
        else {
            return Ok(None);
        };

        // An array supplies its extent directly. A reference must first supply its current array value.
        // Borrow ordinary arrays so that only the reference case needs an owned intermediate value.
        let source = if matches!(source_type.as_ref(), ArrayIrType::Reference(_)) {
            Cow::Owned(context.bind(ReferenceReadOperation::new(), Vec::new(), std::slice::from_ref(source))?.remove(0))
        } else {
            Cow::Borrowed(source)
        };

        Ok(Some(
            context
                .bind(
                    DimensionSizeOperation::new(array_type, axis)?,
                    Vec::new(),
                    std::slice::from_ref(source.as_ref()),
                )?
                .remove(0),
        ))
    }

    #[inline]
    fn zero_operation_with_residuals<R: Clone>(
        r#type: ArrayIrType,
        residuals: &[R],
    ) -> Result<(O, Vec<R>), ProgramError> {
        // Capture stores one residual per distinct dimension identity, even when that identity occurs on
        // several axes. Validate that compact list before expanding it into constructor operands.
        let array_type = <&ArrayType>::try_from(&r#type)?;
        let (shape, first_axes) = ExactShape::for_residual_zero(array_type.shape());
        let expected_residual_count = first_axes.len();
        if residuals.len() != expected_residual_count {
            return Err(ProgramError::InvalidArgument {
                message: format!(
                    "dynamic zero expected {expected_residual_count} extent residuals but got {}",
                    residuals.len(),
                ),
            });
        }

        if expected_residual_count == 0 {
            // Static zeros need no extent operands. Preserve the family's provider choice and reuse the owned type.
            return Ok((Self::provide(ZeroOperation::new(r#type), &[])?, Vec::new()));
        }

        // Dynamic constructors take one operand per dynamic axis: a shape [n, n, m] expands residuals [n, m]
        // into operands [n, n, m]. Static axes stay in the stored type and consume no operand.
        let operands = shape.dynamic_dimensions(residuals);
        Ok((O::from(ZeroOperation::new(array_type.clone())), operands))
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
    /// body). Static axes stage a [`DimensionValue`] [`ConstantOperation`], while dynamic axes clone the residual
    /// value their slot refers to. The result has exactly one value per axis, in axis order, ready to be consumed by
    /// operations that take one dimension operand per output axis.
    ///
    /// # Parameters
    ///
    ///   - `context`: [`Context`] in which static extents are staged as dimension constants.
    ///   - `residuals`: Residual values owned by `context`, indexed by this plan's residual slots (i.e., the region's
    ///     view of the [`LinearResiduals`] list this shape was built against).
    pub fn dimensions<C: Context<Type = ArrayIrType, Operation: From<ConstantOperation<DimensionValue>>>>(
        &self,
        context: &C,
        residuals: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        self.0
            .iter()
            .map(|dimension| match dimension {
                ExactShapeDimension::Static(extent) => Ok(context
                    .bind(ConstantOperation::new(DimensionValue::constant(*extent)?), Vec::new(), &[])?
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
/// [`DimensionType`](crate::DimensionType) carries no concrete extent reuses the slot of any previously retained
/// residual with the same [`DimensionVariable`], because a variable that appears several times (across axes, or as both
/// an axis and an explicit dimension operand) denotes one runtime extent. This keeps operand lists minimal and, more
/// importantly, preserves the type-level equality between axes when shapes are reconstructed inside the attached
/// regions. All other valid differential residuals (i.e., ordinary arrays and dimensions whose types already pin a
/// concrete extent) are purely positional: every retention appends a new slot, and the values themselves are never
/// inspected. References have no differential representation and are rejected by the fallible differentiation type
/// boundary (i.e., via [`DifferentiableType`](crate::DifferentiableType)) before a valid linearization can retain them.
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
    /// [`ArrayIrType::Dimension`] and that [`DimensionType`](crate::DimensionType) has no concrete extent), retention
    /// deduplicates by identity: if a residual with the same [`DimensionVariable`] was already retained, its existing
    /// slot index is returned and `value` is dropped, since both values denote the same runtime extent. Every other
    /// valid differential value (i.e., ordinary arrays, and dimensions whose types pin a concrete extent and therefore
    /// carry no identity worth sharing) is appended to a fresh slot unconditionally, even when it compares equal to an
    /// already-retained value. Unresolved references must never reach this method through a valid differentiation
    /// pipeline.
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

impl<C: Context<Type = ArrayType, Operation: From<ReduceOperation>>, P: ArrayExtentBatchingPolicy<C>>
    CotangentBatchingPolicy<C> for ArrayBatchingPolicy<P>
{
    fn sum_mapped_cotangents(
        _context: &TracingContext<C::Constant, C::Operation>,
        cotangent: Tracer<TracingContext<C::Constant, C::Operation>>,
        axis: Axis,
    ) -> Result<Tracer<TracingContext<C::Constant, C::Operation>>, BatchingError> {
        sum_mapped_array_cotangents(cotangent, axis)
    }
}

impl<
    C: Context<
            Type = ArrayIrType,
            Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            Operation: OperationProjection<ArrayType, Projected: From<ReduceOperation>>,
        >,
> CotangentBatchingPolicy<C> for ArrayIrBatchingPolicy
{
    fn sum_mapped_cotangents(
        _context: &TracingContext<C::Constant, C::Operation>,
        cotangent: Tracer<TracingContext<C::Constant, C::Operation>>,
        axis: Axis,
    ) -> Result<Tracer<TracingContext<C::Constant, C::Operation>>, BatchingError> {
        // Projecting the replayed array cotangent gives it the ordinary `Reduce` capability, whose staged operation
        // lifts back through the composite operation family.
        let cotangent = ValueProjection::<ArrayType>::into_projected(cotangent)?;
        Ok(ValueProjection::from_projected(sum_mapped_array_cotangents(cotangent, axis)?))
    }
}

/// Sums the per-item cotangents of the array-typed `cotangent` packed along `axis`, removing that axis. This is the
/// representation-independent core of [`CotangentBatchingPolicy::sum_mapped_cotangents`] for both array policies (the
/// homogeneous policy applies it to the replayed cotangent directly, while the composite policy applies it to the
/// cotangent's projected array member).
fn sum_mapped_array_cotangents<V: Typed<Type = ArrayType> + Reduce>(
    cotangent: V,
    axis: Axis,
) -> Result<V, BatchingError> {
    let normalized_axis = axis
        .normalize(cotangent.r#type().rank())
        .map_err(|_| BatchingError::BatchAxisOutOfBounds { r#type: Box::new(cotangent.r#type().into_owned()), axis })?;
    Ok(cotangent.reduce(&[normalized_axis], ReductionKind::Sum))
}

/// Materializes one array operand's forward-mode tangent as a concrete projected array value, reading whatever runtime
/// geometry the tangent type omits from the operand's primal. A mixed array rule that has to hand a concrete tangent to
/// a staged operation cannot always materialize a structural zero from its type (an [`ArrayType`] with symbolic extents
/// names its dynamic extents by [`DimensionVariable`] rather than pinning them, so the type-only nullary
/// [`ZeroOperation`](crate::ZeroOperation) is unconstructible for it). The primal names every one of those extents,
/// because the tangent type derivation preserves geometry exactly and rewrites only element representation, layout,
/// and sharding.
///
/// The zero is therefore staged through the *mixed* parent family's residual protocol rather than the projected array
/// view, and the result is projected back. That is deliberate: the mixed family owns the dynamic zero constructor that
/// consumes one first-class dimension operand per dynamic axis, while the projected homogeneous family has only the
/// nullary form. Routing through the parent is also what makes widened tangent representations (e.g., the `f32` tangent
/// of an `f8e8m0fnu` primal) work at a dynamic shape, which naming the primal's whole type as an exemplar could not.
/// Identity-free tangent types declare no residuals and keep the canonical nullary zero, whose zero-producing marker
/// keeps higher-order partial evaluation structural.
///
/// # Parameters
///
///   - `context`: Projected array view of the active mixed [`Context`], whose parent stages the zero.
///   - `input`: Forward-mode dual whose tangent is materialized and whose primal supplies its runtime geometry.
pub fn materialize_array_tangent<
    C: Context<
            Type = ArrayIrType,
            Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            Operation: ResidualZeroProvider<ArrayIrType, Operation = C::Operation> + OperationProjection<ArrayType>,
        > + Zero<C::Value>,
>(
    context: &ProjectedContext<C, ArrayType>,
    input: &DifferentiationDual<C::Value>,
) -> Result<<C::Value as ValueProjection<ArrayType>>::Projected, DifferentiationError> {
    let tangent = C::Operation::materialize_zero_from_residual_sources(
        context.parent(),
        input.tangent().clone(),
        std::iter::once(input.primal()),
    )?;
    Ok(<C::Value as ValueProjection<ArrayType>>::into_projected(tangent)?)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::Array;
    use crate::arrays::ir::ArrayIrValue;
    use crate::arrays::operations::{ArrayIrOperation, ArrayOperation, DimensionOperation};
    use crate::arrays::references::ArrayReference;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{DimensionBounds, DimensionType};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{DifferentiableType, ReverseModeDifferentiate};
    use crate::operations::StopGradientOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{MaybeZero, ReferenceType, TypeError};

    use super::*;

    #[test]
    fn test_array_ir_operation_zero_residual_types() {
        let rows = DimensionVariable::new("rows", DimensionBounds::unbounded());
        let columns = DimensionVariable::new("columns", DimensionBounds::unbounded());
        let r#type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![rows.clone().into(), rows.clone().into(), columns.clone().into(), 4.into()]),
        );
        assert_eq!(
            ArrayIrOperation::<Array>::zero_residual_types(&r#type.into()),
            vec![DimensionType::new(rows).into(), DimensionType::new(columns.clone()).into()],
        );
        assert_eq!(
            ArrayIrOperation::<Array>::zero_residual_types(&ArrayType::scalar(DataType::F32).into(),),
            Vec::<ArrayIrType>::new(),
        );
        assert_eq!(
            ArrayIrOperation::<Array>::zero_residual_types(&DimensionType::new(columns).into(),),
            Vec::<ArrayIrType>::new(),
        );
    }

    #[test]
    fn test_array_ir_operation_capture_zero_residuals() {
        let rows = DimensionVariable::new("rows", DimensionBounds::unbounded());
        let columns = DimensionVariable::new("columns", DimensionBounds::unbounded());
        let r#type =
            ArrayType::new(DataType::F32, Shape::new(vec![rows.clone().into(), rows.into(), columns.into(), 4.into()]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let source = builder.add_input(r#type.clone().into());
        let residuals =
            ArrayIrOperation::<Array>::capture_zero_residuals(&mut builder, source, &r#type.clone().into()).unwrap();

        // Repeated axes share one residual, captured from the first axis carrying each identity.
        assert_eq!(residuals, vec![AtomId::new(1), AtomId::new(2)]);
        assert_eq!(builder.instructions().len(), 2);
        for (instruction, axis) in builder.instructions().iter().zip([0, 2]) {
            assert!(
                matches!(instruction.operation(), ArrayIrOperation::DimensionSize(operation) if operation.axis() == axis)
            );
            assert_eq!(instruction.inputs(), &[source]);
        }
        let static_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        let static_source = builder.add_input(static_type.clone());
        assert_eq!(
            ArrayIrOperation::<Array>::capture_zero_residuals(&mut builder, static_source, &static_type,),
            Ok(Vec::new()),
        );
        assert_eq!(builder.instructions().len(), 2);
    }

    #[test]
    fn test_array_ir_operation_capture_zero_residual_value() {
        // Identity-directed capture answers per declared residual rather than per exemplar, so it must inspect a
        // candidate's type before staging anything. A candidate that does not name the requested quantity has to be
        // rejected without leaving a dead read behind, and a first-class dimension that already *is* the quantity has
        // to be reused rather than re-read.
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(8)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::positive(Some(8)).unwrap());
        let rows_residual_type = ArrayIrType::Dimension(DimensionType::new(rows.clone()));
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();

        // A first-class dimension of exactly the residual type is the extent already and is reused verbatim.
        let dimension = context.input(rows_residual_type.clone());
        let captured =
            ArrayIrOperation::<Array>::capture_zero_residual_value(&context, &dimension, &rows_residual_type).unwrap();
        assert_eq!(captured.unwrap().atom_id(), dimension.atom_id());
        assert!(context.builder().borrow().instructions().is_empty());

        // An array naming the quantity on a non-leading axis contributes a read of that axis, even though its element
        // type and its other axes differ from anything the zero's own type mentions.
        let array = context.input(
            ArrayType::new(
                DataType::F8E8M0FNU,
                Shape::new(vec![Dimension::Static(3), Dimension::Dynamic(rows.clone())]),
            )
            .into(),
        );
        let captured =
            ArrayIrOperation::<Array>::capture_zero_residual_value(&context, &array, &rows_residual_type).unwrap();
        assert_eq!(captured.unwrap().r#type().as_ref(), &rows_residual_type);
        {
            let builder = context.builder().borrow();
            let [instruction] = builder.instructions() else {
                panic!("expected exactly one staged extent read");
            };
            let ArrayIrOperation::DimensionSize(operation) = instruction.operation() else {
                panic!("expected a dimension-size read");
            };
            assert_eq!(operation.axis(), 1);
        }

        // Candidates that do not name the quantity, and residual types that are not first-class dimensions at all,
        // both answer `None` without staging anything.
        let unrelated =
            context.input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(columns)])).into());
        assert!(
            ArrayIrOperation::<Array>::capture_zero_residual_value(&context, &unrelated, &rows_residual_type)
                .unwrap()
                .is_none(),
        );
        let dimension_type = ArrayIrType::Dimension(DimensionType::new(rows));
        assert!(
            ArrayIrOperation::<Array>::capture_zero_residual_value(
                &context,
                &dimension,
                &ArrayType::scalar(DataType::F64).into(),
            )
            .unwrap()
            .is_none(),
        );
        assert_eq!(dimension.r#type().as_ref(), &dimension_type);
        assert_eq!(context.builder().borrow().instructions().len(), 1);
    }

    #[test]
    fn test_array_ir_operation_capture_zero_residual_value_reference() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(2, Some(8)).unwrap());
        let dimension_type = DimensionType::new(extent.clone());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let reference = context.input(ReferenceType::new(array_type).into());

        // A candidate with no matching dimension must not read the reference or stage an extent query.
        let unrelated = DimensionType::new(DimensionVariable::new("other", DimensionBounds::unbounded())).into();
        assert_eq!(ArrayIrOperation::<Array>::capture_zero_residual_value(&context, &reference, &unrelated), Ok(None));
        assert!(context.builder().borrow().instructions().is_empty());

        let dimension = ArrayIrOperation::<Array>::capture_zero_residual_value(
            &context,
            &reference,
            &dimension_type.clone().into(),
        )
        .unwrap()
        .unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![dimension.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let reference = ArrayReference::new(Array::vector(vec![3.0_f32, 5.0, 7.0]));
        let outputs = program.interpret(vec![ArrayIrValue::Reference(reference.clone())]).unwrap();
        let [ArrayIrValue::Dimension(dimension)] = outputs.as_slice() else {
            panic!("expected one dimension residual");
        };
        assert_eq!(dimension.extent(), 3);
        assert_eq!(dimension.r#type().extent(), Some(3));
        assert_eq!(reference.read(), Ok(Array::vector(vec![3.0_f32, 5.0, 7.0])));
    }

    #[test]
    fn test_array_ir_operation_zero_operation_with_residuals() {
        let rows = DimensionVariable::new("rows", DimensionBounds::unbounded());
        let columns = DimensionVariable::new("columns", DimensionBounds::unbounded());
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![rows.clone().into(), rows.into(), columns.into()]));
        let residuals = [AtomId::new(5), AtomId::new(8)];
        let (operation, operands) =
            ArrayIrOperation::<Array>::zero_operation_with_residuals(r#type.clone().into(), &residuals).unwrap();
        assert!(matches!(operation, ArrayIrOperation::Zero(operation) if operation.r#type() == &r#type));
        assert_eq!(operands, vec![residuals[0], residuals[0], residuals[1]]);
        assert_eq!(
            ArrayIrOperation::<Array>::zero_operation_with_residuals(r#type.into(), &residuals[..1],).map(|_| ()),
            Err(ProgramError::InvalidArgument { message: "dynamic zero expected 2 extent residuals but got 1".into() }),
        );
        let static_type = ArrayType::scalar(DataType::F32);
        let (operation, operands) =
            ArrayIrOperation::<Array>::zero_operation_with_residuals(static_type.clone().into(), &[] as &[AtomId])
                .unwrap();
        assert!(
            matches!(operation, ArrayIrOperation::Array(ArrayOperation::Zero(operation)) if operation.r#type() == &static_type)
        );
        assert_eq!(operands, Vec::<AtomId>::new());
    }

    #[test]
    fn test_array_ir_operation_materialize_zero_from_residual_sources() {
        let extent = DimensionVariable::new("extent", DimensionBounds::unbounded());
        let r#type: ArrayIrType =
            ArrayType::new(DataType::F32, Shape::new(vec![extent.clone().into(), extent.into()])).into();
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let source = context.input(r#type.clone());

        // The default boundary protocol shares the array-IR capture and assembly rules: one captured extent is reused
        // for both dynamic axes of the assembled zero.
        let zero = ArrayIrOperation::<Array>::materialize_zero_from_residual_sources(
            &context,
            MaybeZero::Zero(r#type.clone()),
            std::slice::from_ref(&source),
        )
        .unwrap();
        assert_eq!(zero.r#type().as_ref(), &r#type);
        let builder = context.builder().borrow();
        let [dimension_size, zero] = builder.instructions() else {
            panic!("expected one dimension-size instruction followed by one zero instruction");
        };
        assert!(
            matches!(dimension_size.operation(), ArrayIrOperation::DimensionSize(operation) if operation.axis() == 0)
        );
        assert_eq!(dimension_size.inputs(), &[source.atom_id().unwrap()]);
        assert!(matches!(zero.operation(), ArrayIrOperation::Zero(_)));
        assert_eq!(zero.inputs(), &[dimension_size.outputs()[0], dimension_size.outputs()[0]]);
    }

    #[test]
    fn test_array_ir_dynamic_disconnected_pullback_uses_explicit_extent_residual() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let dynamic_type = ArrayType::new(
            DataType::F8E8M0FNU,
            Shape::new(vec![
                Dimension::Dynamic(extent_type.variable().clone()),
                Dimension::Dynamic(extent_type.variable().clone()),
            ]),
        );
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        builder.add_input(dynamic_type.into());
        let scalar = builder.add_input(ArrayType::scalar(DataType::F64).into());
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![scalar],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        // The dynamic input is disconnected from the output, so linearization retains its observed extent as one
        // ordinary residual and the pullback feeds that residual to the mixed zero constructor.
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        assert!(matches!(
            linearization.primal().instructions().last().unwrap().operation(),
            ArrayIrOperation::DimensionSize(_)
        ));
        let pullback = linearization.pullback().unwrap();
        let zero = pullback.instructions().last().unwrap();
        assert!(matches!(zero.operation(), ArrayIrOperation::Zero(_)));
        assert_eq!(zero.inputs(), &[AtomId::new(1), AtomId::new(1)]);
        assert_eq!(
            pullback.interpret(vec![
                ArrayIrValue::Array(Array::scalar(2.0_f64)),
                ArrayIrValue::Dimension(DimensionValue::new(extent_type, 3).unwrap()),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::matrix(3, 3, vec![0.0_f32; 9])),
                ArrayIrValue::Array(Array::scalar(2.0_f64)),
            ]),
        );
    }

    #[test]
    fn test_array_ir_nested_dynamic_disconnected_pullback_uses_explicit_extent_residual() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Dynamic(extent)]));
        let scalar_type = ArrayType::scalar(DataType::F64);
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let dynamic = context.input(dynamic_type.clone().into());
        let scalar = context.input(scalar_type.clone().into());

        // Value-level reverse mode runs inside the outer trace. It saves only the disconnected array's extent,
        // and the reusable pullback consumes that dimension residual through the mixed zero constructor.
        let (_, pullback) =
            context.vjp(|inputs: Vec<_>, ()| Ok(vec![inputs[1].clone()]), vec![dynamic, scalar], ()).unwrap();
        assert_eq!(pullback.residuals().len(), 1);
        assert!(matches!(pullback.residuals()[0].r#type().as_ref(), ArrayIrType::Dimension(_)));
        let transposed = pullback
            .linear_program()
            .transpose_with_trailing_residuals_shared(pullback.residuals().len(), &[])
            .unwrap();
        let zero = transposed.instructions().last().unwrap();
        assert!(matches!(zero.operation(), ArrayIrOperation::Zero(_)));
        assert_eq!(zero.inputs(), &[AtomId::new(1)]);

        let cotangent = context.input(scalar_type.into());
        let cotangents = pullback.apply(vec![cotangent]).unwrap();
        assert_eq!(cotangents[0].r#type().as_ref(), &ArrayIrType::Array(dynamic_type.cotangent().unwrap()));
        assert_eq!(cotangents[1].r#type().as_ref(), &ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
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
    fn test_array_batching_sum_mapped_cotangents() {
        type TestContext = EagerContext<Array, ArrayOperation<Array>>;

        // The array policy reduce-sums the packed per-item cotangents along the mapped axis, dropping that axis.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let cotangent_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let cotangent = context.input(cotangent_type);
        let summed = <ArrayBatchingPolicy as CotangentBatchingPolicy<TestContext>>::sum_mapped_cotangents(
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
            <ArrayBatchingPolicy as CotangentBatchingPolicy<TestContext>>::sum_mapped_cotangents(
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
        let summed = <ArrayIrBatchingPolicy as CotangentBatchingPolicy<TestContext>>::sum_mapped_cotangents(
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
            <ArrayIrBatchingPolicy as CotangentBatchingPolicy<TestContext>>::sum_mapped_cotangents(
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

        // A structural zero whose type names a runtime extent reads that extent from the primal and stages the mixed
        // dynamic zero constructor over it, because the type-only nullary zero cannot supply what `n` names.
        let input = DifferentiationDual::new(primal.clone(), MaybeZero::Zero(ArrayIrType::Array(dynamic_type.clone())))
            .unwrap();
        let materialized = materialize_array_tangent(&projected_context, &input).unwrap();
        assert_eq!(materialized.r#type().as_ref(), &dynamic_type);
        {
            let builder = context.builder().borrow();
            let [size, zero] = builder.instructions() else {
                panic!("expected one staged extent read followed by one staged dynamic zero");
            };
            assert!(matches!(size.operation(), ArrayIrOperation::DimensionSize(_)));
            assert!(matches!(zero.operation(), ArrayIrOperation::Zero(_)));
            assert_eq!(zero.inputs(), size.outputs());
        }

        // An identity-free structural zero declares no residuals and keeps the canonical nullary zero, whose
        // zero-producing marker keeps higher-order partial evaluation structural.
        let static_primal = context.input(static_type.clone().into());
        let input =
            DifferentiationDual::new(static_primal, MaybeZero::Zero(ArrayIrType::Array(static_type.clone()))).unwrap();
        let materialized = materialize_array_tangent(&projected_context, &input).unwrap();
        assert_eq!(materialized.r#type().as_ref(), &static_type);
        {
            let builder = context.builder().borrow();
            let [_, _, instruction] = builder.instructions() else {
                panic!("expected the two earlier instructions followed by one staged nullary zero");
            };
            assert!(matches!(instruction.operation(), ArrayIrOperation::Array(ArrayOperation::Zero(_))));
        }

        // A widening element family is materialized from the same primal even though the two types differ. The `f32`
        // tangent of a dynamically shaped `f8e8m0fnu` primal has no exemplar of its own type anywhere, and naming the
        // extent instead of matching the whole type is what makes it constructible.
        let widening_type =
            ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Dynamic(n.variable().clone())]));
        let widening_primal = context.input(widening_type.clone().into());
        let widened_tangent_type = widening_type.tangent().unwrap();
        assert_eq!(widened_tangent_type.data_type(), DataType::F32);
        let input = DifferentiationDual::new(
            widening_primal,
            MaybeZero::Zero(ArrayIrType::Array(widened_tangent_type.clone())),
        )
        .unwrap();
        let materialized = materialize_array_tangent(&projected_context, &input).unwrap();
        assert_eq!(materialized.r#type().as_ref(), &widened_tangent_type);
        {
            let builder = context.builder().borrow();
            let [.., size, zero] = builder.instructions() else {
                panic!("expected one staged extent read followed by one staged widened dynamic zero");
            };
            assert!(matches!(size.operation(), ArrayIrOperation::DimensionSize(_)));
            assert!(matches!(zero.operation(), ArrayIrOperation::Zero(_)));
            assert_eq!(zero.inputs(), size.outputs());
        }
    }

    #[test]
    fn test_array_ir_dynamic_projected_jvp_materializes_source_relative_widened_zero() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let input_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::StopGradient(StopGradientOperation::new())),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        // The projected constant derivative uses its primal result as the runtime-shape exemplar, then widens the
        // element type to the tangent representation. No type-only dynamic zero is present in the fused JVP.
        let jvp = program.jvp().unwrap();
        assert!(jvp.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ArrayIrOperation::Array(ArrayOperation::ZeroLike(_))
        )));
        assert!(jvp.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ArrayIrOperation::Array(ArrayOperation::ConvertElementType(_))
        )));
        assert!(
            !jvp.instructions()
                .iter()
                .any(|instruction| matches!(instruction.operation(), ArrayIrOperation::Zero(_)))
        );

        let primal = Array::from_f64s(
            ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(3)])),
            vec![1.0, 2.0, 4.0],
        );
        let tangent = Array::vector(vec![1.0_f32, 1.0, 1.0]);
        let expected_primal = primal.clone();
        assert_eq!(
            jvp.interpret(vec![ArrayIrValue::Array(primal), ArrayIrValue::Array(tangent)]),
            Ok(
                vec![ArrayIrValue::Array(expected_primal), ArrayIrValue::Array(Array::vector(vec![0.0_f32, 0.0, 0.0])),]
            ),
        );
    }
}
