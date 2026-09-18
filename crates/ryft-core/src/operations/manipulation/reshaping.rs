use std::borrow::Cow;
use std::fmt::Display;
use std::sync::Arc;

use crate::arrays::{
    Array, ArrayAddressing, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrBatch,
    ArrayIrBatchingPolicy, ArrayIrType, ArrayIrValue, ArrayType, ArrayTypeRefinements, Dimension, DimensionType,
    DimensionValue, LinearResiduals, Shape, Sharding, ShardingDimension,
};
use crate::axes::{Axes, Axis};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableType, DifferentiationDual, ElementwiseDerivativeAlignment, TransposableOperation,
    transpose_projected_operation,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation, impl_reference_dischargeable_operation};
use crate::operations::constants::constant::{ConstantOperation, DimensionConstant};
use crate::operations::differentiation::linear_call::LinearCallOperation;
use crate::operations::dimensions::DimensionArithmetic;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::broadcasting::BroadcastOperation;
use crate::operations::manipulation::transposition::Transpose;
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue, PartiallyEvaluatableOperation,
};
use crate::programs::{
    EffectClass, EffectClasses, Effects, MaybeZero, Operation, OperationFormatter, OperationProjection, ProgramError,
    RegionInterface, Type, TypeError, TypeIdentityRenaming, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

/// Canonical operation name for [`ReshapeOperation`] and [`DynamicReshapeOperation`].
pub const RESHAPE_OPERATION_NAME: &str = "reshape";

/// [`Operation`] that reshapes its input array to a requested [`Shape`]. This is the member-family reshape primitive
/// of the homogeneous array language: complete output geometry is carried by the output [`Shape`], so the operation
/// has exactly one input and no explicit extent edges. The input shape is recoverable from the staged input types
/// and is therefore not duplicated in the payload. It and [`BroadcastOperation`] form the homogeneous baseline that
/// [`ProjectedContext`](crate::ProjectedContext) serves, which is why transform rules for mixed operations can delegate
/// to them once input geometry is resolved. Refer to the documentation of [`Reshape`] for the underlying
/// resolved-geometry contract. Programs that need first-class dynamic extents stage [`DynamicReshapeOperation`]
/// instead, which takes one explicit first-class dimension input per output axis.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReshapeOperation {
    /// Output [`Shape`] of this [`ReshapeOperation`].
    output_shape: Shape,

    /// Optional requested output [`Sharding`] of this [`ReshapeOperation`].
    output_sharding: Option<Sharding>,
}

impl ReshapeOperation {
    /// Creates a new [`ReshapeOperation`] with the provided output shape.
    #[inline]
    pub fn new<S: Into<Shape>>(output_shape: S) -> Self {
        Self { output_shape: output_shape.into(), output_sharding: None }
    }

    /// Returns this operation with the provided output [`Sharding`].
    #[inline]
    pub fn with_output_sharding<S: Into<Option<Sharding>>>(mut self, sharding: S) -> Self {
        self.output_sharding = sharding.into();
        self
    }

    /// Returns the output [`Shape`] of this [`ReshapeOperation`].
    #[inline]
    pub fn output_shape(&self) -> &Shape {
        &self.output_shape
    }

    /// Returns the requested output [`Sharding`] of this [`ReshapeOperation`], if any.
    #[inline]
    pub fn output_sharding(&self) -> Option<&Sharding> {
        self.output_sharding.as_ref()
    }
}

impl Display for ReshapeOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ReshapeOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        RESHAPE_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        match input_types[0].reshape_with_output_sharding(self.output_shape.clone(), self.output_sharding.clone()) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError::invalid(error.to_string())),
        }
    }

    #[inline]
    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayType as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        Ok(Self::new(self.output_shape.rename_type_identities(renaming))
            .with_output_sharding(self.output_sharding.clone()))
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("shape", self.output_shape())?;
            if let Some(output_sharding) = self.output_sharding() {
                operation.field("output_sharding", output_sharding)?;
            }
            Ok(())
        })
    }
}

impl<C: Domain<Type = ArrayType, Value: Reshape>> InterpretableOperation<C> for ReshapeOperation {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // Eager replay resolves a dynamic singleton-inserting or -removing output shape against the concrete input:
        // each dynamic output dimension takes the extent of its corresponding non-singleton input dimension, as the
        // broadcast rule refines its mapped dynamic dimensions. Any mismatch falls through to the type-level checks.
        check_count!("input", inputs, 1, ProgramError);
        let mut output_shape = self.output_shape.clone();
        if output_shape.dimensions().iter().any(|dimension| matches!(dimension, Dimension::Dynamic(_))) {
            let input_type = inputs[0].r#type();
            let mut input_dimensions =
                input_type.shape().dimensions().iter().filter(|dimension| **dimension != Dimension::Static(1));

            // Output singleton axes consume no input axis. Every other output axis consumes one non-singleton
            // input axis (static extents must match, while dynamic extents adopt the corresponding input extent).
            let refined = output_shape
                .dimensions()
                .iter()
                .map(|dimension| match dimension {
                    Dimension::Static(1) => Some(dimension.clone()),
                    Dimension::Static(_) => {
                        input_dimensions.next().filter(|input_dimension| *input_dimension == dimension).cloned()
                    }
                    Dimension::Dynamic(_) => input_dimensions.next().cloned(),
                })
                .collect::<Option<Vec<_>>>();

            // Commit only a complete match. Missing, mismatched, or leftover input axes leave the original shape
            // intact so the ordinary reshape validation reports the error rather than using a partial refinement.
            if let Some(refined) = refined
                && input_dimensions.next().is_none()
            {
                output_shape = Shape::new(refined);
            }
        }
        Ok(vec![inputs[0].reshape_with_output_sharding(output_shape, self.output_sharding.clone())?])
    }
}

impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for ReshapeOperation where
    C::Operation: From<ReshapeOperation>
{
}

impl<C: Context<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>>
    for ReshapeOperation
where
    C::Value: Transpose,
    ReshapeOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        if !inputs[0].ragged_axes().is_empty() {
            return Err(BatchingError::UnsupportedOperation {
                message: format!("`{RESHAPE_OPERATION_NAME}` does not support bounded ragged array inputs"),
            });
        }
        let Some(_) = inputs[0].batch_axis_position() else {
            // Replicated input meaning there is no batch axis to thread through the reshape, so interpret it as given
            // and report the output replicated.
            return Ok(self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()])?.into());
        };
        let Dimension::Static(axis_size) = P::axis_dimension(context)? else {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{RESHAPE_OPERATION_NAME}` with a dynamic mapped extent requires using a dynamic reshape \
                     operation and explicit result-dimension inputs",
                ),
            });
        };
        let input_axis_size = inputs[0].batch_size()?.unwrap();
        if input_axis_size != axis_size {
            return Err(BatchingError::MismatchedBatchSizes { expected: axis_size, actual: input_axis_size });
        }
        let moved_input = inputs[0].move_axis(0)?;
        let output_shape = self.output_shape();
        let mut lifted_output_dimensions = Vec::with_capacity(output_shape.rank() + 1);
        lifted_output_dimensions.push(Dimension::Static(axis_size));
        lifted_output_dimensions.extend_from_slice(output_shape.dimensions());
        let mut lifted_operation = ReshapeOperation::new(Shape::new(lifted_output_dimensions));
        if let Some(output_sharding) = self.output_sharding() {
            lifted_operation = lifted_operation.with_output_sharding(
                output_sharding.with_leading_batch_axis(ArrayBatch::sharding_for_inputs(inputs)?)?,
            );
        }
        Ok(lifted_operation
            .interpret_with_batch_axes(context, &[moved_input], &[BatchAxis::from_position(0)])?
            .into())
    }
}

impl_differentiable_operation! {
    ReshapeOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Operation: From<ReshapeOperation>,
        C::Value: Reshape,
    {
        |operation, _context, _driver, inputs| {
            // `reshape` is structural-linear, and so the tangent is the same reshape applied to the input tangent. A
            // structural-zero input tangent stays structural at the output tangent type: the shared all-zero fast path
            // applies only when the tangent type can be materialized, and direct rule calls reach this body with zero
            // tangents as well.
            check_count!("input", inputs, 1, ProgramError);
            let primal = inputs[0].primal().reshape_with_output_sharding(
                operation.output_shape().clone(),
                operation.output_sharding().cloned(),
            )?;
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                MaybeZero::Value(tangent) => MaybeZero::Value(tangent.reshape_with_output_sharding(
                    operation.output_shape().clone(),
                    operation.output_sharding().cloned(),
                )?),
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType> + From<ReshapeOperation>,
        Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<ArrayType> + Reshape,
    {
        |_operation, context, _driver, inputs, outputs, accumulators| {
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 1, DifferentiationError);
            let input_cotangent_type = inputs[0].r#type().cotangent()?;
            match &outputs[0] {
                MaybeZero::Value(cotangent) => {
                    let bridge_sharding = match (input_cotangent_type.sharding(), cotangent.r#type().sharding()) {
                        (Some(sharding), _) => Some(sharding.clone()),
                        (None, Some(sharding)) => Some(Sharding::replicated(
                            sharding.mesh().clone(),
                            input_cotangent_type.rank(),
                        )),
                        (None, None) => None,
                    };
                    let cotangent = cotangent.reshape_with_output_sharding(
                        input_cotangent_type.shape().clone(),
                        bridge_sharding,
                    )?;
                    let contribution = MaybeZero::Value(cotangent.unalign_cotangent(&input_cotangent_type)?);
                    accumulators[0].accumulate(context, contribution)?;
                    Ok(())
                }
                MaybeZero::Zero(_) => Ok(()),
            }
        }
    },
}

/// Reshapes an array without changing its element count, reading and writing elements in logical row-major order.
///
/// A [`Shape`] specifies the result dimensions, and [`Self::reshape_with_output_sharding`] can request explicit
/// output [`Sharding`]. The final axis varies fastest, independently of physical storage layout. For example,
/// reshaping a `[2, 3]` matrix to `[3, 2]` groups the same six elements into three consecutive pairs. An empty shape
/// requests a scalar and therefore requires exactly one input element. Zero-sized arrays can be reshaped to any fully
/// static shape whose element count is also zero. Reordering input axes is a separate [`Transpose`] operation applied
/// before reshaping.
///
/// The input and output element counts must be equal. An unchanged shape and the insertion or removal of static
/// singleton axes (as performed by [`Self::expand_dimensions`], [`Self::squeeze`], and [`Self::squeeze_all`]) preserve
/// existing dynamic identities through this capability, because the non-singleton dimensions keep their order. Other
/// dynamic result shapes, and shape changes whose input element count is unknown, require [`DynamicReshape`], which
/// takes one explicit dimension input per output axis. A statically zero axis proves a zero element count even when
/// another input axis is dynamic.
///
/// When the input carries sharding information, contiguous split/merge groups redistribute compatible mesh axes
/// over their output factors. Replicated singleton dimensions may be inserted or removed; sharded singletons retain
/// placement when its destination is unambiguous. Ambiguous dynamic, zero-sized, singleton, unconstrained, or
/// non-contiguous placement changes require explicit output sharding. A non-identity reshape preserves the input
/// memory space and clears explicit physical layout metadata because the logical shape change does not determine
/// a unique storage layout.
///
/// [`Reshape`] fills the same role for [`ReshapeOperation`] that [`std::ops::Add`] and [`std::ops::Neg`] fill for
/// their corresponding arithmetic [`Operation`]s.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::{Array, ProgramError, Reshape, Shape};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Shapes: input [6] -> output [2, 3].
/// let input = Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
/// let output = input.reshape(Shape::new(vec![2.into(), 3.into()]))?;
/// assert_eq!(output, Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?);
/// # Ok(())
/// # }
/// ```
pub trait Reshape: Sized {
    /// Reshapes `self` to `shape` with optional explicit output placement. Supplying `None` is equivalent to
    /// [`Self::reshape`]. An explicit placement is needed when a split, merge, or singleton change cannot infer an
    /// unambiguous placement from the input. The placement is attached to the reshape itself so backends can lower
    /// the requested result directly.
    ///
    /// # Parameters
    ///
    ///   - `shape`: Output shape, which must preserve the input element count.
    ///   - `output_sharding`: Requested output placement. When absent, compatible placement is inferred from the input.
    fn reshape_with_output_sharding<S: Into<Shape>>(
        &self,
        shape: S,
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError>;

    /// Reshapes `self` to `shape`, inferring compatible output sharding from the input placement.
    /// The output shape must preserve the element count.
    #[inline]
    fn reshape<S: Into<Shape>>(&self, shape: S) -> Result<Self, ProgramError> {
        self.reshape_with_output_sharding(shape, None)
    }

    /// Reshapes with signed sizes, accepting one inferred `-1` dimension. Values are visited in row-major order where
    /// the final axis varies fastest. A zero input count infers a zero axis if the other sizes have a non-zero product.
    /// Combining `-1` with an explicit zero is ambiguous and is rejected. The input element count must be known;
    /// first-class runtime dimensions use [`DynamicReshape`] instead.
    ///
    /// # Parameters
    ///
    ///   - `output_sizes`: Non-negative target sizes with at most one `-1` entry inferred from the input count.
    #[inline]
    fn reshape_to_sizes(&self, output_sizes: &[isize]) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType>,
    {
        let input_count = self.r#type().shape().element_count()?.ok_or_else(|| {
            TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` size inference requires a known input element count",
            ))
        })?;
        let mut inferred_axis = None;
        let mut sizes = Vec::with_capacity(output_sizes.len());
        for (axis, size) in output_sizes.iter().copied().enumerate() {
            if size == -1 {
                if inferred_axis.replace(axis).is_some() {
                    return Err(ProgramError::from(TypeError::invalid(format!(
                        "`{RESHAPE_OPERATION_NAME}` accepts at most one inferred `-1` dimension",
                    ))));
                }
                sizes.push(1);
            } else {
                sizes.push(usize::try_from(size).map_err(|_| {
                    TypeError::invalid(format!(
                        "`{RESHAPE_OPERATION_NAME}` dimensions must be nonnegative or the inferred size `-1`",
                    ))
                })?);
            }
        }
        let known_count = if sizes.contains(&0) {
            0
        } else {
            sizes.iter().try_fold(1usize, |count, size| count.checked_mul(*size)).ok_or_else(|| {
                TypeError::invalid(format!("`{RESHAPE_OPERATION_NAME}` output element count does not fit in `usize`"))
            })?
        };
        if let Some(axis) = inferred_axis {
            if known_count == 0 {
                return Err(ProgramError::from(TypeError::invalid(format!(
                    "cannot infer a `{RESHAPE_OPERATION_NAME}` dimension when another output dimension is zero",
                ))));
            }
            if !input_count.is_multiple_of(known_count) {
                return Err(ProgramError::from(TypeError::invalid(format!(
                    "`{RESHAPE_OPERATION_NAME}` inferred dimension does not divide the input element count",
                ))));
            }
            sizes[axis] = input_count / known_count;
        } else if known_count != input_count {
            return Err(ProgramError::from(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` output element count {known_count} differs from input element count \
                 {input_count}",
            ))));
        }
        self.reshape(Shape::new(sizes.into_iter().map(Dimension::Static).collect()))
    }

    /// Returns the input as a one-dimensional array in logical row-major order. An input that is already a vector
    /// retains its shape, including a dynamic extent. Other ranks require a known element count; an empty input
    /// produces shape `[0]`. Storage sharing or copying is determined by the backend.
    #[inline]
    fn flatten(&self) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType>,
    {
        let input_type = self.r#type();
        if input_type.rank() == 1 {
            // An existing vector needs no inferred element count or newly introduced dimension identity.
            self.reshape(input_type.shape().clone())
        } else {
            self.reshape_to_sizes(&[-1])
        }
    }

    /// Inserts one size-one axis without changing element order. `axis` addresses the result rank, so `0` inserts
    /// a leading axis and `-1` appends a trailing axis. Existing dynamic dimensions retain their identities and
    /// bounds; inserting a static singleton does not require explicit runtime extent inputs.
    ///
    /// # Parameters
    ///
    ///   - `axis`: Position of the inserted axis, normalized against the result rank.
    fn expand_dimensions<A: Into<Axis>>(&self, axis: A) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType>,
    {
        let input_type = self.r#type();
        let axis = axis
            .into()
            .normalize(input_type.rank() + 1)
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        let mut dimensions = input_type.shape().dimensions().to_vec();
        dimensions.insert(axis, Dimension::Static(1));
        self.reshape(Shape::new(dimensions))
    }

    /// Removes the selected size-one axes without changing element order. Negative axes address the input rank;
    /// duplicates and axes whose size is not exactly one are rejected. An empty selection leaves the shape unchanged.
    /// Use [`Self::squeeze_all`] to remove every singleton axis. Retained dynamic dimensions keep their identities
    /// and bounds, while the removed axes must have statically known size one.
    ///
    /// # Parameters
    ///
    ///   - `axes`: Input axes to remove; each must have the statically known size one.
    fn squeeze<A: Into<Axes>>(&self, axes: A) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType>,
    {
        let input_type = self.r#type();
        let axes = axes.into().normalize(input_type.rank()).map_err(|error| TypeError::invalid(error.to_string()))?;
        for axis in &axes {
            if input_type.dimension(*axis) != Dimension::Static(1) {
                return Err(TypeError::invalid(format!(
                    "`{RESHAPE_OPERATION_NAME}` cannot squeeze axis {axis} whose size is not one"
                ))
                .into());
            }
        }
        let dimensions = input_type
            .shape()
            .dimensions()
            .iter()
            .enumerate()
            .filter(|(axis, _)| !axes.contains(axis))
            .map(|(_, dimension)| dimension.clone())
            .collect();
        self.reshape(Shape::new(dimensions))
    }

    /// Removes every statically size-one axis using [`Self::squeeze`]. A shape consisting entirely of singleton axes
    /// becomes scalar; a shape with no singleton axes is unchanged.
    fn squeeze_all(&self) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType>,
    {
        let axes = self
            .r#type()
            .shape()
            .dimensions()
            .iter()
            .enumerate()
            .filter_map(|(axis, dimension)| (*dimension == Dimension::Static(1)).then_some(axis))
            .collect::<Vec<_>>();
        self.squeeze(axes)
    }
}

impl Reshape for ArrayType {
    fn reshape_with_output_sharding<S: Into<Shape>>(
        &self,
        shape: S,
        output_sharding: Option<Sharding>,
    ) -> Result<ArrayType, ProgramError> {
        let shape = shape.into();
        if self.shape() != &shape {
            if shape.dimensions().iter().any(|size| matches!(size, Dimension::Dynamic(_))) {
                // A dynamic output shape is accepted only when it inserts or removes static singleton axes around the
                // input's non-singleton dimensions, which keeps those dimensions and their identities in order and
                // therefore preserves the element count. Every other dynamic result shape needs explicit
                // result-dimension inputs.
                let is_non_singleton = |dimension: &&Dimension| **dimension != Dimension::Static(1);
                if !self
                    .shape()
                    .dimensions()
                    .iter()
                    .filter(is_non_singleton)
                    .eq(shape.dimensions().iter().filter(is_non_singleton))
                {
                    return Err(TypeError::invalid(format!(
                        "`{RESHAPE_OPERATION_NAME}` requires explicit result-dimension inputs for a dynamic output \
                         shape that does not only insert or remove singleton axes"
                    ))
                    .into());
                }
            } else {
                let Some(input_elements) = self
                    .element_count()
                    .map_err(|error| TypeError::invalid(format!("`{RESHAPE_OPERATION_NAME}` input {error}")))?
                else {
                    return Err(TypeError::invalid(format!(
                        "`{RESHAPE_OPERATION_NAME}` requires explicit result-dimension inputs for a dynamic input shape"
                    ))
                    .into());
                };
                let Some(output_elements) = shape
                    .element_count()
                    .map_err(|error| TypeError::invalid(format!("`{RESHAPE_OPERATION_NAME}` output {error}")))?
                else {
                    unreachable!("every output dimension is static on this path");
                };
                if input_elements != output_elements {
                    return Err(TypeError::invalid(format!(
                        "`{RESHAPE_OPERATION_NAME}` changes the number of elements",
                    ))
                    .into());
                }
            }
        }
        Ok(infer_reshape_output_type(self, shape, output_sharding.as_ref())?)
    }
}

impl Reshape for Array {
    fn reshape_with_output_sharding<S: Into<Shape>>(
        &self,
        shape: S,
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError> {
        // Delegate to the type-level reshape so all element-count and sharding validation remains shared with staged
        // execution.
        let output_type = self.r#type().reshape_with_output_sharding(shape, output_sharding)?;
        let input_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        if input_addressing.is_dense_row_major() && output_addressing.is_dense_row_major() {
            // Both shapes enumerate exactly the same contiguous encodings. Retyping can share storage; later
            // mutation detaches it through the array's existing copy-on-write boundary.
            return Ok(Self::new_unchecked(output_type, self.shared_storage().clone()));
        }
        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        for index in 0..input_addressing.element_count() {
            bytes[output_addressing.byte_range_for_flat_index(index)]
                .copy_from_slice(&self.storage_bytes()[input_addressing.byte_range_for_flat_index(index)]);
        }
        Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
    }
}

impl<V: Value<Type = ArrayType>> Reshape for V
where
    V::DispatchDomain: Context<Type = ArrayType, Operation: From<ReshapeOperation>>,
{
    #[inline]
    fn reshape_with_output_sharding<S: Into<Shape>>(
        &self,
        shape: S,
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError> {
        // Any context-carrying value reshapes by binding a `ReshapeOperation` through its own context. The
        // `From<ReshapeOperation>` bound makes this disjoint from the eager value types (whose context operation
        // is `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete
        // implementations.
        let operation = ReshapeOperation::new(shape).with_output_sharding(output_sharding);
        let input_type = self.r#type().into_owned();
        let output_type = input_type
            .reshape_with_output_sharding(operation.output_shape().clone(), operation.output_sharding().cloned())?;
        if input_type == output_type {
            return Ok(self.clone());
        }
        let mut outputs = self.dispatch_domain().bind(operation, Vec::new(), std::slice::from_ref(self))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Mixed [`Operation`] that reshapes arrays using one explicit first-class dimension input per output axis. The first
/// input value is the array being reshaped. Every remaining input describes the corresponding output-axis extent, in
/// order. Exact dimension types produce static axes while non-exact dimension types retain their variables as dynamic
/// axes. The operation therefore carries only reshape attributes; it does not duplicate its output shape or encode
/// shape arithmetic in its payload.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DynamicReshapeOperation {
    /// Refer to the documentation of [`requires_runtime_assertion`](Self::requires_runtime_assertion)
    /// for more information.
    requires_runtime_assertion: bool,

    /// Refer to the documentation of [`output_sharding`](Self::output_sharding) for more information.
    output_sharding: Option<Sharding>,
}

impl DynamicReshapeOperation {
    /// Creates a new [`DynamicReshapeOperation`] with no requested output [`Sharding`]. The operation initially
    /// retains a runtime element-count assertion. Use [`with_input_types`](Self::with_input_types) to remove it
    /// when the input types prove that the input and output shapes contain the same number of elements.
    #[inline]
    pub fn new() -> Self {
        Self { requires_runtime_assertion: true, output_sharding: None }
    }

    /// Returns a copy of this [`DynamicReshapeOperation`] with its runtime element-count assertion requirement
    /// recomputed from the provided `input_types`. The complete mixed signature is validated before determining
    /// whether the input and output shapes are guaranteed to contain the same number of elements.
    ///
    /// If that equality can be proved, the returned operation is effect-free. Otherwise, it retains
    /// [`EffectClass::OrderedAssertion`] so execution checks the element counts. For example, reshaping `[n, 4]` to
    /// `[n, 2, 2]` proves equality through the shared dimension identity and equal static factors. Reshaping to
    /// `[m, 2, 2]` with an independent dynamic dimension `m` generally still requires a runtime check.
    ///
    /// An effect-free operation rejects subsequent type inference requests that require a runtime assertion, rather
    /// than silently changing its effects. Calling this function again explicitly recomputes the requirement for a
    /// different signature and can restore the assertion. The requested output sharding is unchanged.
    ///
    /// # Parameters
    ///
    ///   - `input_types`: Complete mixed input signature containing the input array type followed by one
    ///     [`DimensionType`] wrapped in [`ArrayIrType`] per output axis, in axis order.
    pub fn with_input_types(mut self, input_types: &[ArrayIrType]) -> Result<Self, TypeError> {
        // Reset the requirement before validation so a previously effect-free operation can be explicitly
        // reconfigured for a signature whose element-count equality must be checked at runtime.
        self.requires_runtime_assertion = true;
        let output_types = self.infer_output_types(input_types, &[])?;
        let input = <&ArrayType>::try_from(&input_types[0])?;
        let output = <&ArrayType>::try_from(&output_types[0])?;
        self.requires_runtime_assertion = !input.shape().has_equal_element_count(output.shape())?;
        Ok(self)
    }

    /// Returns a copy of this [`DynamicReshapeOperation`] with the requested output `sharding`. Passing [`None`]
    /// restores inferred placement. The request is validated during type inference.
    #[inline]
    pub fn with_output_sharding<S: Into<Option<Sharding>>>(mut self, sharding: S) -> Self {
        self.output_sharding = sharding.into();
        self
    }

    /// Returns whether execution must check that the product of the explicit output-extent inputs equals the input
    /// array's element count. A mismatch is an error; reshaping cannot add or remove elements. When `true`, the
    /// operation carries [`EffectClass::OrderedAssertion`], making the check an observable effect even when the
    /// reshaped output is unused. This is the conservative initial state, not a claim that the extents are invalid.
    /// [`Self::with_input_types`] sets the flag to `false` when the input types prove equal element counts, allowing
    /// the operation to be effect-free. Such an operation rejects input signatures that would require a runtime check
    /// unless it is explicitly refined again with [`Self::with_input_types`].
    #[inline]
    pub fn requires_runtime_assertion(&self) -> bool {
        self.requires_runtime_assertion
    }

    /// Returns the requested output [`Sharding`], or [`None`] when placement is inferred from the input.
    #[inline]
    pub fn output_sharding(&self) -> Option<&Sharding> {
        self.output_sharding.as_ref()
    }
}

impl Default for DynamicReshapeOperation {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Display for DynamicReshapeOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for DynamicReshapeOperation {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        RESHAPE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        let Some((input_type, output_extent_types)) = input_types.split_first() else {
            return Err(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` expects an array followed by its output extents",
            )));
        };
        let input_type = <&ArrayType>::try_from(input_type)?;
        let output_shape = Shape::new(ArrayIrType::extents(output_extent_types)?);
        let output_type = infer_dynamic_reshape_output_type(input_type, output_shape, self.output_sharding())?;
        if !self.requires_runtime_assertion && !input_type.shape().has_equal_element_count(output_type.shape())? {
            return Err(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` was constructed without a runtime element-count check but these input \
                 types require one",
            )));
        }
        Ok(vec![output_type.into()])
    }

    fn effects(&self) -> Cow<'_, Effects> {
        Cow::Owned(Effects::explicit(if self.requires_runtime_assertion {
            EffectClasses::single(EffectClass::OrderedAssertion)
        } else {
            EffectClasses::NONE
        }))
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        if self.requires_runtime_assertion && self.output_sharding.is_none() {
            return formatter.write_str(RESHAPE_OPERATION_NAME);
        }
        OperationFormatter::new(formatter, indentation, RESHAPE_OPERATION_NAME)?.bracketed(|operation| {
            if !self.requires_runtime_assertion {
                operation.field("requires_runtime_assertion", false)?;
            }
            if let Some(output_sharding) = &self.output_sharding {
                operation.field("output_sharding", output_sharding)?;
            }
            Ok(())
        })
    }
}

impl_reference_dischargeable_operation!(@reference_free DynamicReshapeOperation);

impl<C: Domain<Type = ArrayIrType, Value: DynamicReshape>> InterpretableOperation<C> for DynamicReshapeOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        let Some((input, output_extents)) = inputs.split_first() else {
            return Err(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` expects an array followed by its output extents",
            ))
            .into());
        };
        Ok(vec![input.dynamic_reshape_with_output_sharding(output_extents, self.output_sharding().cloned())?])
    }
}

impl<C: Context<Type = ArrayIrType, Operation: From<DynamicReshapeOperation>>> PartiallyEvaluatableOperation<C>
    for DynamicReshapeOperation
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        if self.output_sharding().is_none()
            && driver.region_count() == 0
            && let Some(input) = inputs.first()
            && let Ok(input_type) = <&ArrayType>::try_from(input.r#type().as_ref())
            && input_type.static_shape().is_some()
            && self
                .infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(), &[])?
                == vec![input.r#type().into_owned()]
        {
            // A static identity reshape cannot observe its exact dimension inputs. Preserve the input directly so
            // an unknown array does not leave a redundant reshape in the residual program.
            return Ok(vec![input.clone()]);
        }
        context.fold_or_residualize(self.clone(), driver.regions().map(|region| region.to_program()).collect(), inputs)
    }
}

impl<C: Context<Type = ArrayIrType>> BatchableOperation<C, ArrayIrBatchingPolicy> for DynamicReshapeOperation
where
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
    C::Operation: From<DynamicReshapeOperation>,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        _driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatchingPolicy>, BatchingError> {
        // Explicit output extents remain replicated shape values. A mapped input is canonicalized to a leading batch
        // axis, and that axis is inserted into both the reshape geometry and the output sharding before the mixed
        // operation is replayed.
        let Some((input, output_extents)) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };

        <&ArrayType>::try_from(&input.unbatched_type())?;

        if !input.ragged_axes().is_empty() {
            return Err(BatchingError::UnsupportedOperation {
                message: format!("dynamic `{RESHAPE_OPERATION_NAME}` does not support bounded ragged array inputs"),
            });
        }

        for extent in output_extents {
            extent.validate_replicated_dimension()?;
        }

        if input.batch_axis().is_replicated() {
            return Ok(context
                .parent()
                .bind(self.clone(), Vec::new(), &inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>())?
                .into_iter()
                .map(ArrayIrBatch::replicated)
                .collect::<Vec<_>>()
                .into());
        }

        let moved_input = ArrayBatch::new(
            <C::Value as ValueProjection<ArrayType>>::into_projected(input.value().clone())?,
            input.batch_axis(),
        )?
        .move_axis(0)?;
        let moved_input = <C::Value as ValueProjection<ArrayType>>::from_projected(moved_input.into_value());

        let mut operation = Self::new();
        if let Some(output_sharding) = self.output_sharding() {
            operation = operation
                .with_output_sharding(output_sharding.with_leading_batch_axis(context.axis_sharding().clone())?);
        }

        let mut lifted_inputs = Vec::with_capacity(inputs.len() + 1);
        lifted_inputs.push(moved_input);
        lifted_inputs.push(context.axis_extent().clone());
        lifted_inputs.extend(output_extents.iter().map(|extent| extent.value().clone()));

        // Recompute the runtime assertion requirement for the lifted signature so proven equal counts stay effect-free.
        let operation = operation
            .with_input_types(&lifted_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>())?;
        Ok(context
            .parent()
            .bind(operation, Vec::new(), lifted_inputs.as_slice())?
            .into_iter()
            .map(|output| ArrayIrBatch::new(output, BatchAxis::from_position(0)))
            .collect::<Result<Vec<_>, _>>()?
            .into())
    }
}

impl_differentiable_operation! {
    DynamicReshapeOperation,
    jvp<C>
    where
        C: Context<Type = ArrayIrType>,
        C::Operation: From<DimensionSizeOperation>
            + From<LinearCallOperation<ArrayIrType>>
            + From<DynamicReshapeOperation>
            + From<ConstantOperation<DimensionValue>>
            + OperationProjection<ArrayType, Projected: From<BroadcastOperation>>,
    {
        |operation, context, _driver, inputs| {
            // The explicit output extents are ordinary non-differentiated shape values. Static input cotangent
            // geometry replays the mixed reshape directly. Dynamic geometry retains the exact input shape so the
            // linear transpose can reconstruct the inverse reshape from first-class dimension residuals.
            let destinations = context;
            if inputs.is_empty() {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
            }
            let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            let primal_operation = operation
                .clone()
                .with_input_types(&primal_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>())?;
            let mut primal_outputs =
                destinations.primal().bind(primal_operation, Vec::new(), primal_inputs.as_slice())?;
            check_count!("output", primal_outputs, 1, ProgramError);
            let output_primal = primal_outputs.remove(0);
            let tangent_primal = destinations.primal_to_tangent(output_primal.clone())?;
            let tangent_inputs = destinations.dual_primal_to_tangent(inputs)?;

            // Lifting the duals into the tangent space preserves their arity, so the array input is still present.
            let (array, output_extents) = tangent_inputs.split_first().unwrap();
            let tangent_context = destinations.tangent();
            let tangent = match array.tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(tangent_primal.r#type().tangent()?),
                MaybeZero::Value(array_tangent) => {
                    let input_type = <&ArrayType>::try_from(array.primal().r#type().as_ref())?.clone();
                    let input_cotangent_type = input_type.cotangent()?;

                    // Direct replay needs both the input geometry and the output geometry to be static, so that the
                    // transpose can reshape back without runtime extents. Otherwise, the exact extents are retained
                    // as linearization residuals; static extents become constants in the inverse shape.
                    let output_type = <&ArrayType>::try_from(output_primal.r#type().as_ref())?.clone();
                    if input_cotangent_type
                        .shape()
                        .dimensions()
                        .iter()
                        .chain(output_type.shape().dimensions())
                        .all(|dimension| matches!(dimension, Dimension::Static(_)))
                    {
                        let mut replay_inputs = Vec::with_capacity(tangent_inputs.len());
                        replay_inputs.push(array_tangent.clone());
                        replay_inputs.extend(output_extents.iter().map(|extent| extent.primal().clone()));
                        let tangent_operation = operation.clone().with_input_types(
                            &replay_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
                        )?;
                        let mut outputs =
                            tangent_context.bind(tangent_operation, Vec::new(), replay_inputs.as_slice())?;
                        check_count!("output", outputs, 1, ProgramError);
                        MaybeZero::Value(outputs.remove(0))
                    } else {
                        // Record each distinct dynamic input extent while the source array is available. Repeated type
                        // identities reuse one residual Single Static Assignment (SSA) value in first-use order.
                        let mut residuals = LinearResiduals::new();
                        let output_extent_residuals =
                            residuals.retain_all(output_extents.iter().map(|extent| extent.primal().clone()));
                        let input_shape = residuals.retain_shape(tangent_context, array.primal())?;

                        // Both linear regions share one deterministic residual boundary. The forward region consumes
                        // the retained output extents; the transpose region consumes the retained exact input geometry.
                        let forward_operation = operation.clone();
                        let forward_output_extents = output_extent_residuals.clone();
                        let transpose_target_type = input_cotangent_type.clone();
                        let tangent = LinearCallOperation::stage(
                            tangent_context,
                            residuals.into_values(),
                            vec![array_tangent.clone()],
                            move |residuals, linear_inputs| {
                                let mut reshape_inputs = Vec::with_capacity(1 + forward_output_extents.len());
                                reshape_inputs.push(linear_inputs[0].clone());
                                reshape_inputs
                                    .extend(forward_output_extents.iter().map(|index| residuals[*index].clone()));
                                linear_inputs[0].dispatch_domain().bind(
                                    forward_operation.with_input_types(
                                        &reshape_inputs
                                            .iter()
                                            .map(|input| input.r#type().into_owned())
                                            .collect::<Vec<_>>(),
                                    )?,
                                    Vec::new(),
                                    reshape_inputs.as_slice(),
                                )
                            },
                            move |residuals, output_cotangents| {
                                let transpose_context = output_cotangents[0].dispatch_domain();
                                let bridge_sharding = match (
                                    transpose_target_type.sharding(),
                                    <&ArrayType>::try_from(output_cotangents[0].r#type().as_ref())?.sharding(),
                                ) {
                                    (Some(sharding), _) => Some(sharding.clone()),
                                    (None, Some(sharding)) => Some(Sharding::replicated(
                                        sharding.mesh().clone(),
                                        transpose_target_type.rank(),
                                    )),
                                    (None, None) => None,
                                };
                                let mut inverse_operation = DynamicReshapeOperation::new();
                                if let Some(bridge_sharding) = bridge_sharding {
                                    inverse_operation = inverse_operation.with_output_sharding(bridge_sharding);
                                }
                                let mut inverse_inputs = Vec::with_capacity(transpose_target_type.rank() + 1);
                                inverse_inputs.push(output_cotangents[0].clone());
                                inverse_inputs.extend(input_shape.dimensions(&transpose_context, residuals)?);
                                let inverse_operation = inverse_operation.with_input_types(
                                    &inverse_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
                                )?;
                                let mut outputs =
                                    transpose_context.bind(inverse_operation, Vec::new(), inverse_inputs.as_slice())?;
                                check_count!("output", outputs, 1, ProgramError);
                                let cotangent = outputs.remove(0);

                                // The inverse geometry is exact, but reshape clears layouts and may need replicated
                                // bridge sharding. Restore the original cotangent's complete storage metadata after
                                // reconstructing the shape, just as the homogeneous rule's unalignment does.
                                let cotangent =
                                    if <&ArrayType>::try_from(cotangent.r#type().as_ref())? != &transpose_target_type {
                                        let mut outputs = transpose_context.bind(
                                            <C::Operation as OperationProjection<ArrayType>>::Projected::from(
                                                BroadcastOperation::new(
                                                    transpose_target_type.clone(),
                                                    (0..transpose_target_type.rank()).collect(),
                                                ),
                                            ),
                                            Vec::new(),
                                            std::slice::from_ref(&cotangent),
                                        )?;
                                        check_count!("output", outputs, 1, ProgramError);
                                        outputs.remove(0)
                                    } else {
                                        cotangent
                                    };
                                Ok(vec![cotangent])
                            },
                        )?
                        .remove(0);
                        MaybeZero::Value(tangent)
                    }
                }
            };

            Ok(vec![DifferentiationDual::new(output_primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
        O: Operation<Type = ArrayIrType>
            + OperationProjection<
                ArrayType,
                Projected: From<ReshapeOperation>
                    + TransposableOperation<
                        <V as ValueProjection<ArrayType>>::Projected,
                        <O as OperationProjection<ArrayType>>::Projected,
                    >,
            >,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // Static input geometry delegates to the homogeneous array pullback, while every explicit output
            // extent receives a structural-zero cotangent. Dynamic input geometry requires linearization so
            // that `DifferentiableOperation::jvp` can retain its exact extents as residuals.
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);

            let Some((input, _output_extents)) = inputs.split_first() else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
            };
            let input_cotangent_type = <&ArrayType>::try_from(input.r#type().as_ref())?.cotangent()?;

            // No inverse geometry is needed when there is no live contribution. In particular, dynamic input extents
            // must not force residual capture for a structural-zero cotangent or a nondifferentiable input.
            if input_cotangent_type.is_zero_space() || outputs[0].is_zero() {
                return Ok(());
            }
            let MaybeZero::Value(cotangent) = &outputs[0] else {
                unreachable!("a structural-zero output cotangent returns early above");
            };
            let output_type = <&ArrayType>::try_from(cotangent.r#type().as_ref())?.clone();

            // The homogeneous pullback reshapes the cotangent back through static geometry only. A dynamic extent on
            // either side is available exactly as a linearization residual, so direct transposition is rejected.
            if input_cotangent_type
                .shape()
                .dimensions()
                .iter()
                .chain(output_type.shape().dimensions())
                .any(|dimension| matches!(dimension, Dimension::Dynamic(_)))
            {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "direct transposition of a dynamic `{RESHAPE_OPERATION_NAME}` requires linearization so its \
                         input and output extents are available as explicit residuals",
                    ),
                }
                .into());
            }

            let projected_operation = <O as OperationProjection<ArrayType>>::Projected::from(
                ReshapeOperation::new(output_type.shape().clone())
                    .with_output_sharding(operation.output_sharding().cloned()),
            );

            // Dimension inputs do not receive cotangents; forward only the array inputs' handles.
            transpose_projected_operation(
                context,
                &projected_operation,
                std::slice::from_ref(input),
                outputs,
                &accumulators[..1],
            )
        }
    },
}

/// Reshapes an array using one explicit first-class dimension value per output axis.
///
/// This is the shape-polymorphic counterpart of [`Reshape`], which receives its complete output geometry as a
/// [`Shape`]. Exact dimension types describe static axes, including those produced by arithmetic, while non-exact
/// dimension types describe dynamic axes. Both forms bind the same [`DynamicReshapeOperation`], so runtime shape
/// arithmetic stays an ordinary graph computation instead of a type-level side condition; backend lowering chooses
/// the appropriate static, bounded, or dynamic representation from the inferred result type.
///
/// The output extents must multiply to the input element count, including when either shape contains a zero axis. When
/// the input types cannot prove this equality, the operation retains an ordered runtime assertion even if its result is
/// unused. Each dynamic extent also retains its declared bounds. Backends may reject unbounded geometry or a reshape
/// whose input and output physical capacities cannot be represented by their runtime reshape support; equal logical
/// element counts alone do not guarantee that every bounded shape can be compiled.
///
/// # Examples
///
/// Exact host sizes can use [`Self::dynamic_reshape_to_sizes`]:
///
/// ```rust
/// # use ryft_core::{Array, ArrayIrValue, DynamicReshape, ProgramError};
///
/// // Shapes: input [6] -> output [2, 3].
/// let input = ArrayIrValue::Array(Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?);
/// let output = input.dynamic_reshape_to_sizes(&[2, 3])?;
/// assert_eq!(output, ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?));
/// # Ok::<(), ProgramError>(())
/// ```
///
/// Computed or input dimensions remain ordinary Single Static Assignment (SSA) inputs, which is what makes a
/// runtime-derived output shape expressible. Here a `[batch, 6]` input is reshaped so that its dynamic leading extent
/// is read off the input while its trailing extent is an exact lifted dimension. Extents derived by first-class
/// dimension arithmetic work the same way, using the [`DimensionArithmetic`] capability (e.g.,
/// `rows.dimension_mul(&columns)?`) directly on the composite values.
///
/// ```rust
/// # use ryft_core::{
/// #     Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayType, Context, DataType, Dimension, DimensionBounds,
/// #     DimensionSize, DimensionValue, DimensionVariable, DynamicReshape, ProgramError, Shape, StagingContext,
/// #     TracingContext, Typed,
/// # };
///
/// let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
/// let input_type =
///     ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(6)]));
/// let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
///
/// // Shapes: input [batch, 6] -> output [batch, 2, 3], retaining the symbolic batch extent.
/// let input = context.input(ArrayIrType::Array(input_type));
/// let rows = input.dimension_size(0).unwrap();
/// let columns = context.lift(DimensionValue::constant(2).unwrap().into()).unwrap();
/// let depth = context.lift(DimensionValue::constant(3).unwrap().into()).unwrap();
/// let output = input.dynamic_reshape(&[rows, columns, depth]).unwrap();
/// assert_eq!(output.r#type().to_string(), "f32[batch, 2, 3]");
/// ```
pub trait DynamicReshape: Value<Type = ArrayIrType> + Sized {
    /// Reshapes `self` with optional explicit output sharding. Supplying `None` is equivalent to
    /// [`Self::dynamic_reshape`]. Explicit placement resolves ambiguous input-to-output sharding changes
    /// and is attached to the reshape itself for backend lowering.
    ///
    /// # Parameters
    ///
    ///   - `output_dimensions`: Non-negative first-class dimension values in output-axis order. Their product must
    ///     equal the input element count. An empty slice requests a scalar. Exact values infer static axes, while
    ///     non-exact values retain their dimension identities and bounds.
    ///   - `output_sharding`: Requested placement for the output axes. When absent, compatible input placement is
    ///     propagated. Ambiguous placement changes require an explicit sharding.
    fn dynamic_reshape_with_output_sharding(
        &self,
        output_dimensions: &[Self],
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError>;

    /// Reshapes `self` to the output shape described by `output_dimensions`, one first-class value per output axis.
    ///
    /// # Parameters
    ///
    ///   - `output_dimensions`: Non-negative first-class dimension values in output-axis order. Their product must
    ///     equal the input element count. An empty slice requests a scalar. Exact values infer static axes, while
    ///     non-exact values retain their dimension identities and bounds.
    #[inline]
    fn dynamic_reshape(&self, output_dimensions: &[Self]) -> Result<Self, ProgramError> {
        self.dynamic_reshape_with_output_sharding(output_dimensions, None)
    }

    /// Reshapes the input to an exact static shape by lifting every size into a dimension constant in its context.
    /// The sizes must multiply to the input element count (runtime inputs are checked when equality is not provable).
    ///
    /// # Parameters
    ///
    ///   - `output_sizes`: Output-axis sizes, including any zero axes. An empty slice requests a scalar. These are
    ///     exact extents rather than capacity bounds. Inferred `-1` sizes belong to [`Reshape::reshape_to_sizes`].
    fn dynamic_reshape_to_sizes(&self, output_sizes: &[usize]) -> Result<Self, ProgramError>
    where
        Self::DispatchDomain: Context<Type = ArrayIrType> + DimensionConstant,
    {
        // Validate the requested geometry before staging any dimension constant, so that an invalid element count
        // leaves no dead dimension literals behind in a trace.
        let input_type = self.r#type();
        let input_type = <&ArrayType>::try_from(input_type.as_ref())?;
        let output_shape = Shape::new(output_sizes.iter().map(|extent| Dimension::Static(*extent)).collect());
        infer_dynamic_reshape_output_type(input_type, output_shape, None)?;
        let output_dimensions = output_sizes
            .iter()
            .map(|extent| self.dispatch_domain().dimension_constant(*extent))
            .collect::<Result<Vec<_>, _>>()?;
        self.dynamic_reshape(output_dimensions.as_slice())
    }

    /// Returns the input as a vector in logical row-major order. Runtime extents are read from the input and multiplied
    /// using checked first-class dimension arithmetic. A vector retains its existing dimension identity while a scalar
    /// becomes a vector of size one, and an empty array becomes a vector of size zero.
    fn dynamic_flatten(&self) -> Result<Self, ProgramError>
    where
        Self: DimensionSize + DimensionArithmetic,
        Self::DispatchDomain: Context<Type = ArrayIrType> + DimensionConstant,
    {
        let input_type = self.r#type();
        let input_type = <&ArrayType>::try_from(input_type.as_ref())?;
        if input_type.rank() == 1 {
            return Ok(self.clone());
        }

        if input_type.shape().dimensions().iter().any(|dimension| dimension.value() == Some(0)) {
            // A known zero extent makes the complete product zero, even when an earlier partial product would
            // overflow the dimension ABI. Do not emit arithmetic for dimensions that cannot affect the result.
            return self.dynamic_reshape_to_sizes(&[0]);
        }

        if let Some(element_count) = input_type.element_count()? {
            // Static geometry needs no runtime size arithmetic.
            return self.dynamic_reshape_to_sizes(&[element_count]);
        }

        // Seed the product with the leading extent instead of a constant one, so that no multiplication by one is
        // staged. At least two axes remain here because vectors returned above.
        let mut size = self.dimension_size(0)?;
        for axis in 1..input_type.rank() {
            size = size.dimension_mul(&self.dimension_size(axis)?)?;
        }

        self.dynamic_reshape(&[size])
    }

    /// Inserts a size-one axis while preserving all existing extents, including runtime dimensions. The output
    /// dimensions are explicit inputs to [`Self::dynamic_reshape`], so retained programs can specialize them when
    /// concrete input shapes become available.
    ///
    /// # Parameters
    ///
    ///   - `axis`: Insertion position in the output rank. Zero prepends an axis and negative one appends an axis.
    fn dynamic_expand_dimensions<A: Into<Axis>>(&self, axis: A) -> Result<Self, ProgramError>
    where
        Self: DimensionSize,
        Self::DispatchDomain: Context<Type = ArrayIrType> + DimensionConstant,
    {
        let input_type = self.r#type();
        let input_type = <&ArrayType>::try_from(input_type.as_ref())?;
        let axis = axis
            .into()
            .normalize(input_type.rank() + 1)
            .map_err(|error| TypeError::invalid(error.to_string()))?;

        // Static extents are lifted as constants; only dynamic axes need a runtime size read.
        let mut dimensions = input_type
            .shape()
            .dimensions()
            .iter()
            .enumerate()
            .map(|(axis, dimension)| match dimension {
                Dimension::Static(extent) => self.dispatch_domain().dimension_constant(*extent),
                Dimension::Dynamic(_) => self.dimension_size(axis),
            })
            .collect::<Result<Vec<_>, _>>()?;
        dimensions.insert(axis, self.dispatch_domain().dimension_constant(1)?);
        self.dynamic_reshape(&dimensions)
    }
}

impl<A: Reshape + Value<Type = ArrayType>> DynamicReshape for ArrayIrValue<A> {
    fn dynamic_reshape_with_output_sharding(
        &self,
        output_dimensions: &[Self],
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError> {
        // Concrete composite values resolve every explicit extent input to its runtime value, so the mixed reshape
        // executes as the ordinary member reshape of the fully resolved output shape.
        let input = <Self as ValueProjection<ArrayType>>::projected(self)?;
        let mut refinements = ArrayTypeRefinements::default();
        let output_shape = Shape::new(
            output_dimensions
                .iter()
                .map(<Self as ValueProjection<DimensionType>>::projected)
                .map(|result| {
                    // Repeated identities describe one extent, even when separate eager values supply them.
                    let dimension = result?;
                    refinements.bind(dimension.r#type().variable(), dimension.extent())?;
                    Ok(Dimension::Static(dimension.extent()))
                })
                .collect::<Result<Vec<_>, ProgramError>>()?,
        );
        Ok(Self::Array(input.reshape_with_output_sharding(output_shape, output_sharding)?))
    }
}

impl<
    V: Value<Type = ArrayIrType, DispatchDomain: Context<Type = ArrayIrType, Operation: From<DynamicReshapeOperation>>>,
> DynamicReshape for V
{
    fn dynamic_reshape_with_output_sharding(
        &self,
        output_dimensions: &[Self],
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError> {
        let input_type = self.r#type();
        let input_type = <&ArrayType>::try_from(input_type.as_ref())?;
        let output_shape =
            Shape::new(ArrayIrType::extents(output_dimensions.iter().map(|dimension| dimension.r#type()))?);
        let operation = DynamicReshapeOperation::new().with_output_sharding(output_sharding);
        let output_type = infer_dynamic_reshape_output_type(input_type, output_shape, operation.output_sharding())?;

        // A static identity reshape cannot observe its exact extent inputs, so it stages nothing. Dynamic geometry
        // keeps the instruction because its inputs assert the runtime element-count relation.
        if input_type.static_shape().is_some() && &output_type == input_type {
            return Ok(self.clone());
        }

        let mut inputs = Vec::with_capacity(output_dimensions.len() + 1);
        inputs.push(self.clone());
        inputs.extend_from_slice(output_dimensions);
        let operation =
            operation.with_input_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>())?;
        let mut outputs = self.dispatch_domain().bind(operation, Vec::new(), inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Resolves the output sharding of a reshape from `input` to `output_shape`, validating a requested sharding or
/// inferring one from the input, and rebuilds the output type. When the shape is unchanged, the input's own type is
/// returned with the resolved sharding so that its layout survives. Any other reshape clears the layout, because the
/// logical shape change does not determine a unique storage layout. Both homogeneous validation and mixed inference
/// resolve their outputs here so that the two paths cannot drift.
fn infer_reshape_output_type(
    input: &ArrayType,
    output_shape: Shape,
    requested_sharding: Option<&Sharding>,
) -> Result<ArrayType, TypeError> {
    let sharding = match (requested_sharding, input.sharding()) {
        (Some(requested), _) => {
            // An explicit placement may redistribute dimensions, but must preserve the mesh, reduction state,
            // and manual-axis variation. Validate it before returning even for an identity reshape.
            if requested.rank() != output_shape.rank() {
                return Err(TypeError::invalid(format!(
                    "`{}` requested output sharding rank ({}) does not match the output rank ({})",
                    RESHAPE_OPERATION_NAME,
                    requested.rank(),
                    output_shape.rank(),
                )));
            }

            if input.sharding().is_some_and(|input| input.mesh() != requested.mesh()) {
                return Err(TypeError::invalid(format!(
                    "`{RESHAPE_OPERATION_NAME}` requested output sharding uses a different mesh",
                )));
            }

            if requested.references_auto_axis() {
                return Err(TypeError::invalid(format!(
                    "`{RESHAPE_OPERATION_NAME}` requested output sharding cannot reference auto mesh axes",
                )));
            }

            let input_unreduced = input.sharding().map(Sharding::unreduced_axes).cloned().unwrap_or_default();
            if requested.unreduced_axes() != &input_unreduced {
                return Err(TypeError::invalid(format!(
                    "`{RESHAPE_OPERATION_NAME}` requested output sharding changes the unreduced mesh axes",
                )));
            }

            let input_reduced = input.sharding().map(Sharding::reduced_axes).cloned().unwrap_or_default();
            if requested.reduced_axes() != &input_reduced {
                return Err(TypeError::invalid(format!(
                    "`{RESHAPE_OPERATION_NAME}` requested output sharding changes the reduced mesh axes",
                )));
            }

            let input_varying = input.sharding().map(Sharding::varying_manual_axes).cloned().unwrap_or_default();
            if requested.varying_manual_axes() != &input_varying {
                return Err(TypeError::invalid(format!(
                    "`{RESHAPE_OPERATION_NAME}` requested output sharding changes the varying manual mesh axes",
                )));
            }
            Some(requested.clone())
        }
        (None, Some(sharding)) => {
            // Infer placement from the input by preserving singleton placements first and then aligning
            // the remaining axes.
            let input_dimensions = input
                .shape()
                .dimensions()
                .iter()
                .cloned()
                .enumerate()
                .filter(|(_, size)| *size != Dimension::Static(1))
                .collect::<Vec<_>>();
            let output_dimensions = output_shape
                .dimensions()
                .iter()
                .cloned()
                .enumerate()
                .filter(|(_, size)| *size != Dimension::Static(1))
                .collect::<Vec<_>>();

            // Singleton axes do not change the element count, but a sharded singleton still records a placement. One
            // that stays at the same index keeps its placement. A moved sharded singleton keeps it only when exactly
            // one sharded input singleton and one output singleton remain unmatched. Any other loss of placement
            // requires an explicit output sharding instead of being dropped silently.
            let mut output_sharding_dimensions = vec![ShardingDimension::replicated(); output_shape.rank()];
            let mut matched_output_singletons = vec![false; output_shape.rank()];
            let mut unmatched_sharded_singletons = Vec::new();
            for axis in 0..input.rank() {
                if input.dimension(axis) != Dimension::Static(1) {
                    continue;
                }
                if axis < output_shape.rank() && output_shape.dimension(axis) == Dimension::Static(1) {
                    output_sharding_dimensions[axis] = sharding.dimensions()[axis].clone();
                    matched_output_singletons[axis] = true;
                } else if sharding.dimensions()[axis] != ShardingDimension::Replicated {
                    unmatched_sharded_singletons.push(axis);
                }
            }

            if let [input_axis, rest @ ..] = unmatched_sharded_singletons.as_slice() {
                let unmatched_output_singletons = (0..output_shape.rank())
                    .filter(|axis| {
                        output_shape.dimension(*axis) == Dimension::Static(1) && !matched_output_singletons[*axis]
                    })
                    .collect::<Vec<_>>();
                match unmatched_output_singletons.as_slice() {
                    [output_axis] if rest.is_empty() => {
                        output_sharding_dimensions[*output_axis] = sharding.dimensions()[*input_axis].clone();
                    }
                    _ => {
                        return Err(TypeError::invalid(format!(
                            "`{RESHAPE_OPERATION_NAME}` requires explicit output sharding to place the sharded \
                             singleton input axis {input_axis} in the output",
                        )));
                    }
                }
            }

            if input_dimensions.iter().map(|(_, size)| size).eq(output_dimensions.iter().map(|(_, size)| size)) {
                for ((input_axis, _), (output_axis, _)) in input_dimensions.iter().zip(&output_dimensions) {
                    output_sharding_dimensions[*output_axis] = sharding.dimensions()[*input_axis].clone();
                }
            } else if sharding.dimensions().iter().any(|dimension| *dimension != ShardingDimension::Replicated) {
                // Replicated inputs need no placement alignment. Non-replicated inputs with zero extents use a separate
                // path because multiplying through zero cannot identify the split/merge groups.
                if input_dimensions.iter().any(|(_, size)| *size == Dimension::Static(0))
                    || output_dimensions.iter().any(|(_, size)| *size == Dimension::Static(0))
                {
                    // Keep equal prefix and suffix axes in place. The unmatched middle must be replicated and static
                    // as its zero product cannot determine how to distribute sharding or align symbolic axes.
                    let mut prefix = 0usize;
                    while input_dimensions.get(prefix).map(|(_, size)| size)
                        == output_dimensions.get(prefix).map(|(_, size)| size)
                    {
                        let Some(((input_axis, _), (output_axis, _))) =
                            input_dimensions.get(prefix).zip(output_dimensions.get(prefix))
                        else {
                            break;
                        };
                        output_sharding_dimensions[*output_axis] = sharding.dimensions()[*input_axis].clone();
                        prefix += 1;
                    }

                    let mut input_end = input_dimensions.len();
                    let mut output_end = output_dimensions.len();
                    while input_end > prefix
                        && output_end > prefix
                        && input_dimensions[input_end - 1].1 == output_dimensions[output_end - 1].1
                    {
                        input_end -= 1;
                        output_end -= 1;
                        output_sharding_dimensions[output_dimensions[output_end].0] =
                            sharding.dimensions()[input_dimensions[input_end].0].clone();
                    }

                    if input_dimensions[prefix..input_end]
                        .iter()
                        .any(|(axis, _)| sharding.dimensions()[*axis] != ShardingDimension::Replicated)
                    {
                        return Err(TypeError::invalid(format!(
                            "`{RESHAPE_OPERATION_NAME}` requires explicit output sharding for an ambiguous \
                             zero-sized reshape",
                        )));
                    }

                    if input_dimensions[prefix..input_end].iter().any(|(_, size)| matches!(size, Dimension::Dynamic(_)))
                        || output_dimensions[prefix..output_end]
                            .iter()
                            .any(|(_, size)| matches!(size, Dimension::Dynamic(_)))
                    {
                        return Err(TypeError::invalid(format!(
                            "`{RESHAPE_OPERATION_NAME}` requires explicit output sharding for unaligned \
                             dynamic dimensions",
                        )));
                    }
                } else {
                    // Grow contiguous static groups until their products match, retaining equal symbolic axes
                    // as anchors.
                    let alignment_error = || {
                        TypeError::invalid(format!(
                            "`{RESHAPE_OPERATION_NAME}` could not align reshape dimension groups",
                        ))
                    };

                    let mut input_start = 0usize;
                    let mut output_start = 0usize;
                    while input_start < input_dimensions.len() || output_start < output_dimensions.len() {
                        if input_start == input_dimensions.len() || output_start == output_dimensions.len() {
                            return Err(alignment_error());
                        }

                        // Equal symbolic axes retain their placement independently of adjacent static
                        // split/merge groups.
                        if input_dimensions[input_start].1 == output_dimensions[output_start].1 {
                            output_sharding_dimensions[output_dimensions[output_start].0] =
                                sharding.dimensions()[input_dimensions[input_start].0].clone();
                            input_start += 1;
                            output_start += 1;
                            continue;
                        }

                        let input_group_start = input_start;
                        let output_group_start = output_start;
                        let mut input_product = static_positive_size(&input_dimensions[input_start].1)?;
                        let mut output_product = static_positive_size(&output_dimensions[output_start].1)?;
                        input_start += 1;
                        output_start += 1;
                        while input_product != output_product {
                            if input_product < output_product {
                                let (_, size) = input_dimensions.get(input_start).ok_or_else(alignment_error)?;
                                input_product = input_product
                                    .checked_mul(static_positive_size(size)?)
                                    .ok_or_else(alignment_error)?;
                                input_start += 1;
                            } else {
                                let (_, size) = output_dimensions.get(output_start).ok_or_else(alignment_error)?;
                                output_product = output_product
                                    .checked_mul(static_positive_size(size)?)
                                    .ok_or_else(alignment_error)?;
                                output_start += 1;
                            }
                        }

                        // Merge the contiguous sharded prefix of this group. A replicated axis followed by a
                        // sharded one would change which elements belong to each device.
                        let mut mesh_axes = Vec::new();
                        let mut saw_replicated = false;
                        for (axis, _) in &input_dimensions[input_group_start..input_start] {
                            match &sharding.dimensions()[*axis] {
                                ShardingDimension::Replicated => saw_replicated = true,
                                ShardingDimension::Unconstrained => {
                                    return Err(TypeError::invalid(format!(
                                        "`{RESHAPE_OPERATION_NAME}` requires explicit output sharding for \
                                         unconstrained dimensions",
                                    )));
                                }
                                ShardingDimension::Sharded(axis_names) => {
                                    if saw_replicated {
                                        return Err(TypeError::invalid(format!(
                                            "`{RESHAPE_OPERATION_NAME}` cannot preserve non-contiguous sharding \
                                             across a merge",
                                        )));
                                    }
                                    mesh_axes.extend(axis_names.iter().cloned());
                                }
                            }
                        }

                        // Distribute the merged mesh axes over output factors in order.
                        // Each axis must divide its factor.
                        let mut mesh_axis_index = 0usize;
                        for (output_axis, size) in &output_dimensions[output_group_start..output_start] {
                            let mut remaining = static_positive_size(size)?;
                            let start = mesh_axis_index;
                            while remaining > 1 && mesh_axis_index < mesh_axes.len() {
                                let mesh_axis = &mesh_axes[mesh_axis_index];
                                let mesh_axis_size = sharding.mesh().axis_size(mesh_axis).unwrap();
                                if remaining % mesh_axis_size != 0 {
                                    return Err(TypeError::invalid(format!(
                                        "`{RESHAPE_OPERATION_NAME}` cannot distribute sharding across the \
                                         requested split factors",
                                    )));
                                }
                                remaining /= mesh_axis_size;
                                mesh_axis_index += 1;
                            }

                            if mesh_axis_index > start {
                                output_sharding_dimensions[*output_axis] =
                                    ShardingDimension::sharded(mesh_axes[start..mesh_axis_index].iter().cloned());
                            }
                        }

                        if mesh_axis_index != mesh_axes.len() {
                            return Err(TypeError::invalid(format!(
                                "`{RESHAPE_OPERATION_NAME}` cannot distribute all input mesh axes \
                                 across the output dimensions",
                            )));
                        }
                    }
                }
            }

            // Only dimension placement changes; retain the input mesh, reduction state, and manual-axis variation.
            Some(
                Sharding::new(sharding.mesh().clone(), output_sharding_dimensions)
                    .and_then(|output| output.with_unreduced_axes(sharding.unreduced_axes().clone()))
                    .and_then(|output| output.with_reduced_axes(sharding.reduced_axes().clone()))
                    .and_then(|output| output.with_varying_manual_axes(sharding.varying_manual_axes().clone()))
                    .map_err(|error| {
                        TypeError::invalid(format!(
                            "`{RESHAPE_OPERATION_NAME}` inferred output sharding is invalid: {error}",
                        ))
                    })?,
            )
        }
        (None, None) => None,
    };

    if input.shape() == &output_shape {
        return input.clone().with_sharding(sharding).map_err(|error| {
            TypeError::invalid(format!("`{RESHAPE_OPERATION_NAME}` output sharding is invalid: {error}"))
        });
    }

    ArrayType::new(input.data_type(), output_shape)
        .with_memory(input.memory())
        .with_sharding(sharding)
        .map_err(|error| TypeError::invalid(format!("`{RESHAPE_OPERATION_NAME}` output sharding is invalid: {error}")))
}

/// Infers the output type of a dynamic reshape, rejecting known element-count mismatches before resolving its
/// placement through [`infer_reshape_output_type`]. Unknown element counts are checked at runtime when needed.
fn infer_dynamic_reshape_output_type(
    input: &ArrayType,
    output_shape: Shape,
    requested_sharding: Option<&Sharding>,
) -> Result<ArrayType, TypeError> {
    // Reject statically provable element-count mismatches immediately. Dynamic relationships remain explicit graph
    // facts; eager execution checks concrete sizes and lowering must enforce the runtime requirement explicitly.
    let input_elements = input
        .element_count()
        .map_err(|error| TypeError::invalid(format!("`{RESHAPE_OPERATION_NAME}` input {error}")))?;
    let output_elements = output_shape
        .element_count()
        .map_err(|error| TypeError::invalid(format!("`{RESHAPE_OPERATION_NAME}` output {error}")))?;
    if let (Some(input_elements), Some(output_elements)) = (input_elements, output_elements)
        && input_elements != output_elements
    {
        return Err(TypeError::invalid(format!("`{RESHAPE_OPERATION_NAME}` changes the number of elements")));
    }
    infer_reshape_output_type(input, output_shape, requested_sharding)
}

// TODO(eaplatanios): Review from here onwards.

/// Returns the positive static value of `size` for split/merge factorization.
fn static_positive_size(size: &Dimension) -> Result<usize, TypeError> {
    size.value().filter(|value| *value > 0).ok_or_else(|| {
        TypeError::invalid(format!(
            "`{RESHAPE_OPERATION_NAME}` requires explicit output sharding for unaligned dynamic dimensions"
        ))
    })
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;
    use std::marker::PhantomData;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::batching::DynamicArrayExtentBatchingPolicy;
    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, DataType, DimensionBounds, DimensionError,
        DimensionOperation, DimensionVariable, Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType, RaggedAxis,
        Sharding, StridedLayout,
    };
    use crate::contexts::{EagerContext, ProjectedContext, StagingContext};
    use crate::differentiation::{
        DifferentiableOperation, DifferentiationContext, DifferentiationError, TranspositionContext,
    };
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::operations::dimensions::dimension_from_scalar::DimensionFromScalarOperation;
    use crate::operations::dimensions::dimension_mul::DimensionMulOperation;
    use crate::operations::manipulation::transposition::TransposeOperation;
    use crate::parameters::{Parameter, Placeholder};
    use crate::partial::PartialValue;
    use crate::programs::{
        BindingRegionDriver, EmptyRegionDriver, Program, ProgramBuilder, ProgramError, Provenance, ProvenanceScope,
        Typed,
    };
    use crate::tracing::Trace;

    use super::*;

    /// Value wrapper that dispatches through a deliberately malformed context in either array language.
    #[derive(Clone, Debug)]
    struct DispatchValue<V: Value, O: Clone + Debug> {
        value: V,
        output_count: usize,
        operation: PhantomData<O>,
    }

    impl<V: Value, O: Clone + Debug> Display for DispatchValue<V, O> {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "{}", self.value)
        }
    }

    impl<V: Value, O: Clone + Debug> Parameter for DispatchValue<V, O> {}

    impl<V: Value, O: Clone + Debug> Typed for DispatchValue<V, O> {
        type Type = V::Type;

        fn r#type(&self) -> Cow<'_, Self::Type> {
            self.value.r#type()
        }
    }

    impl<V: Value, O: Debug + Operation<Type = V::Type>> Value for DispatchValue<V, O> {
        type DispatchDomain = InvalidOutputContext<V, O>;
        type ExecutionDomain = InvalidOutputContext<V, O>;

        fn dispatch_domain(&self) -> Self::DispatchDomain {
            InvalidOutputContext(self.output_count, PhantomData)
        }

        fn execution_domain(&self) -> Self::ExecutionDomain {
            self.dispatch_domain()
        }
    }

    /// Context that accepts valid reshape inputs but violates the single-output contract.
    #[derive(Clone)]
    struct InvalidOutputContext<V, O>(usize, PhantomData<(V, O)>);

    impl<V: Value, O: Debug + Operation<Type = V::Type>> Domain for InvalidOutputContext<V, O> {
        type Type = V::Type;
        type Value = DispatchValue<V, O>;
        type Constant = V;
        type Operation = O;
    }

    impl<V: Value, O: Debug + Operation<Type = V::Type>> Context for InvalidOutputContext<V, O> {
        fn lift(&self, value: V) -> Result<Self::Value, ProgramError> {
            Ok(DispatchValue { value, output_count: self.0, operation: PhantomData })
        }

        fn bind<Operation: Into<O>, D: BindingRegionDriver<V, O>>(
            &self,
            _operation: Operation,
            _driver: D,
            inputs: &[Self::Value],
        ) -> Result<Vec<Self::Value>, ProgramError> {
            Ok(vec![inputs[0].clone(); self.0])
        }

        fn is_eager(&self) -> bool {
            true
        }

        fn provenance(&self) -> Provenance {
            Provenance::unknown()
        }

        fn invoke_with_provenance_origin<R, F: FnOnce() -> R>(&self, _origin: Provenance, function: F) -> R {
            function()
        }

        fn invoke_with_provenance_scope<R, F: FnOnce() -> R>(&self, _scope: ProvenanceScope, function: F) -> R {
            function()
        }
    }

    /// Builds the `[source, 4] -> [2, source * 2]` mixed reshape whose inverse cannot recover `source` from its output
    /// shape without division, so that its differentiation must retain the source extent as an explicit residual.
    fn doubled_extent_reshape_program()
    -> Program<ArrayIrValue<Array>, ArrayIrOperation<Array>, Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>> {
        let source = DimensionVariable::new("source", DimensionBounds::new(0, Some(9)).unwrap());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(source.clone()), Dimension::Static(4)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.clone().into());
        let source_extent = builder
            .add_instruction(DimensionSizeOperation::new(&input_type, 0).unwrap(), Vec::new(), vec![input], None)
            .unwrap()[0];
        let two_value = DimensionValue::constant(2).unwrap();
        let two_type = two_value.r#type().into_owned();
        let two = builder.add_constant(ArrayIrValue::Dimension(two_value));
        let doubled_extent = builder
            .add_instruction(
                DimensionOperation::Mul(DimensionMulOperation::new(&DimensionType::new(source), &two_type).unwrap()),
                Vec::new(),
                vec![source_extent, two],
                None,
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![input, two, doubled_extent], None)
            .unwrap()[0];
        builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap()
    }

    #[test]
    fn test_reshape() {
        let shape = Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]);
        let operation = ReshapeOperation::new(shape.clone());

        // Operation identity and accessors.
        assert_eq!(operation.name(), RESHAPE_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "reshape [shape=[2, 3]]");
        assert_eq!(operation.output_shape(), &shape);

        // Program rendering uses the canonical operation name and includes the captured output shape.
        let mut builder = ProgramBuilder::<Array, ReshapeOperation>::new();
        let program_input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![6.into()])));
        let program_output = builder.add_instruction(operation, Vec::new(), vec![program_input], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![program_output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[6] .
                let %1:f64[2, 3] = reshape [shape=[2, 3]] %0
                in (%1)
            "}
            .trim_end(),
        );

        // The requested output sharding renders after the shape.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap();
        let operation = ReshapeOperation::new(shape).with_output_sharding(sharding.clone());
        assert_eq!(operation.output_sharding(), Some(&sharding));
        assert_eq!(
            operation.to_string(),
            indoc! {"
                reshape [shape=[2, 3], output_sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}]
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_reshape_type_inference() {
        let shape = Shape::new(vec![2.into(), 3.into()]);
        let operation = ReshapeOperation::new(shape.clone());
        // Type inference validates the element count and returns the target shape.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(6)]));
        let output_type = ArrayType::new(DataType::F64, shape.clone());
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    input_types = [input_type.clone()],
                    output_types = [output_type.clone()],
                },
                {
                    input_types = [],
                    error = "expected 1 input but got 0",
                },
                {
                    input_types = [ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(5)]))],
                    error = format!("`{RESHAPE_OPERATION_NAME}` changes the number of elements"),
                },
            ],
        );

        // Type-level (abstract) reshaping validates the target shape and returns the output type without consuming
        // the borrowed input type.
        assert_eq!(input_type.reshape(shape.clone()), Ok(output_type.clone()));

        assert_eq!(
            operation.infer_output_types(&[input_type], &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)]),
            Err(TypeError::invalid("expected 0 regions but got 1"))
        );
    }

    #[test]
    fn test_reshape_type_inference_identity_instantiation() {
        // The output shape stored in the operation names the same dimension identity as the input type, so
        // instantiating the program's identities renames both and the imported reshape keeps the caller's identity.
        let bounds = DimensionBounds::new(1, Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(source.clone())]));
        let output_shape = Shape::new(vec![Dimension::Static(1), Dimension::Dynamic(source)]);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(input_type);
        let output =
            builder.add_instruction(ReshapeOperation::new(output_shape), Vec::new(), vec![input], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        let target = DimensionVariable::new("target", bounds);
        let target_input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(target.clone())]));
        let target_output_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1), Dimension::Dynamic(target)]));
        let instantiated =
            program.with_instantiated_type_identities(&[target_input_type.clone()]).unwrap().into_owned();
        assert_eq!(instantiated.output_types(), vec![target_output_type.clone()]);
        assert_eq!(
            instantiated.to_string(),
            indoc! {"
                lambda %0:f64[target] .
                let %1:f64[1, target] = reshape [shape=[1, target]] %0
                in (%1)
            "}
            .trim_end(),
        );

        let mut destination = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let imported_input = destination.add_input(target_input_type);
        let imported_outputs = destination.splice_program(&instantiated, &[imported_input]).unwrap();
        let imported = destination
            .build::<Vec<Array>, Vec<Array>>(imported_outputs, vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(imported.output_types(), vec![target_output_type]);
        assert_eq!(imported.to_string(), instantiated.to_string());
    }

    #[test]
    fn test_reshape_interpretation() {
        let shape = Shape::new(vec![2.into(), 3.into()]);
        let operation = ReshapeOperation::new(shape.clone());
        let output_type = ArrayType::new(DataType::F64, shape);
        // Interpretation reinterprets the row-major payload under the target shape.
        let input = Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output = operation
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(*output[0].r#type(), output_type);
        assert_eq!(output[0].to_f64s(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(Arc::ptr_eq(input.shared_storage(), output[0].shared_storage()));

        // Storage sharing preserves value semantics: changing a reshaped copy must detach its payload.
        let mut changed = output[0].clone();
        changed.storage_bytes_mut()[..8].copy_from_slice(&9.0_f64.to_ne_bytes());
        assert_eq!(changed.to_f64s(), vec![9.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(input.to_f64s(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(output[0].to_f64s(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // An explicit transpose composes with the row-major reshape.
        assert_eq!(
            Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap()
                .transpose([-1, -2])
                .unwrap()
                .reshape(Shape::new(vec![Dimension::Static(6)]))
                .map(|array| array.to_f64s()),
            Ok(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]),
        );

        // Invalid interpreter arity reports the exact program error.
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[]
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
    }

    #[test]
    fn test_reshape_interpretation_refines_dynamic_singleton_axes() {
        // Eager replay of a dynamic singleton-inserting or -removing payload resolves the dynamic axes against the
        // concrete input, so specialized and interpreted programs execute the same reshape the trace admitted.
        let rows = Dimension::Dynamic(DimensionVariable::new("rows", DimensionBounds::new(0, Some(9)).unwrap()));
        let input = Array::matrix(3, 4, (0..12).map(|value| value as f64).collect()).unwrap();
        let inserted = ReshapeOperation::new(Shape::new(vec![1.into(), rows.clone(), 4.into()]))
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(*inserted[0].r#type(), ArrayType::new_static(DataType::F64, [1, 3, 4]));
        assert_eq!(inserted[0].to_f64s(), input.to_f64s());
        let removed = ReshapeOperation::new(Shape::new(vec![rows.clone(), 4.into()]))
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &inserted)
            .unwrap();
        assert_eq!(*removed[0].r#type(), ArrayType::new_static(DataType::F64, [3, 4]));
        assert_eq!(removed[0].to_f64s(), input.to_f64s());
        // A permutation is applied before the singleton change, and a non-singleton mismatch still reports the
        // type-level rejection rather than refining.
        let permuted = ReshapeOperation::new(Shape::new(vec![4.into(), 1.into(), rows.clone()]))
            .interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                std::slice::from_ref(&input.transpose([1, 0]).unwrap()),
            )
            .unwrap();
        assert_eq!(*permuted[0].r#type(), ArrayType::new_static(DataType::F64, [4, 1, 3]));
        assert_eq!(permuted[0].to_f64s(), vec![0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0]);
        assert_eq!(
            ReshapeOperation::new(Shape::new(vec![rows, 5.into()])).interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                std::slice::from_ref(&input),
            ),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` requires explicit result-dimension inputs for a dynamic output shape that \
                 does not only insert or remove singleton axes"
            )))),
        );
    }

    #[test]
    fn test_reshape_partial_evaluation() {
        // Check the standard partial-evaluation contract for both known and residual inputs.
        let input = Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let expected = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = ReshapeOperation::new(Shape::new(vec![2.into(), 3.into()])),
            cases = [
                {
                    inputs = [(@known, input.clone())],
                    outputs = [(@known, expected.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [(@unknown(type = input.r#type().into_owned(), replay = input.clone()))],
                    outputs = [(@residual, expected)],
                    residual_instructions = 1,
                },
            ],
        );
    }

    #[test]
    fn test_reshape_batching() {
        // A mapped axis is moved to the leading position before reshaping each batch item, while a replicated input
        // is reshaped as given and stays replicated.
        let batched_output = Array::from_elements::<f64>(
            ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 2.into(), 3.into()])),
            &(0..12).map(|value| value as f64).collect::<Vec<_>>(),
        )
        .unwrap();
        check_operation_batching!(
            @exact,
            operation = ReshapeOperation::new(Shape::new(vec![2.into(), 3.into()])),
            axis_size = 2,
            cases = [
                {
                    inputs = [(@mapped(axis = 0), Array::matrix(
                        2,
                        6,
                        (0..12).map(|value| value as f64).collect(),
                    ).unwrap())],
                    outputs = [(@mapped(axis = 0), batched_output.clone())],
                },
                {
                    inputs = [(@mapped(axis = 1), Array::matrix(
                        6,
                        2,
                        vec![0.0, 6.0, 1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 11.0],
                    ).unwrap())],
                    outputs = [(@mapped(axis = 0), batched_output)],
                },
                {
                    inputs = [(@replicated, Array::vector((0..6).map(|value| value as f64).collect()).unwrap())],
                    outputs = [(@replicated, Array::matrix(
                        2,
                        3,
                        (0..6).map(|value| value as f64).collect(),
                    ).unwrap())],
                },
            ],
        );

        // A requested per-item output sharding is lifted around the mapped axis, which keeps the input's own
        // placement on the physical batch dimension.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new_static(DataType::F64, [2, 6])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let output_type = ArrayType::new_static(DataType::F64, [2, 2, 3])
            .with_sharding(
                Sharding::new(
                    mesh.clone(),
                    vec![
                        ShardingDimension::sharded(["x"]),
                        ShardingDimension::replicated(),
                        ShardingDimension::replicated(),
                    ],
                )
                .unwrap(),
            )
            .unwrap();
        check_operation_batching!(
            @exact,
            operation = ReshapeOperation::new(Shape::new(vec![2.into(), 3.into()]))
                .with_output_sharding(Sharding::replicated(mesh, 2)),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::from_elements::<f64>(
                    input_type,
                    &(0..12).map(|value| value as f64).collect::<Vec<_>>(),
                ).unwrap())],
                outputs = [(@mapped(axis = 0), Array::from_elements::<f64>(
                    output_type,
                    &(0..12).map(|value| value as f64).collect::<Vec<_>>(),
                ).unwrap())],
            }],
        );
    }

    #[test]
    fn test_reshape_batching_dynamic_singleton_axes() {
        // Inserting a singleton axis around a dynamic per-item extent is a homogeneous reshape, so the mapped axis
        // is threaded through it like any other static geometry and the dynamic identity survives in the output.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let input = trace.input(
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Static(3), Dimension::Static(2)]),
            )
            .into(),
        );
        let input = <_ as ValueProjection<ArrayType>>::into_projected(input).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(ProjectedContext::new(trace.clone()), 3);
        let operation = ReshapeOperation::new(Shape::new(vec![
            Dimension::Dynamic(rows.clone()),
            Dimension::Static(1),
            Dimension::Static(2),
        ]));
        let (outputs, _) = operation
            .batch(&context, &EmptyRegionDriver, &[ArrayBatch::new(input, BatchAxis::new(1)).unwrap()])
            .unwrap()
            .into_parts();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(
            outputs[0].value().r#type().as_ref(),
            &ArrayType::new(
                DataType::F32,
                Shape::new(vec![
                    Dimension::Static(3),
                    Dimension::Dynamic(rows),
                    Dimension::Static(1),
                    Dimension::Static(2),
                ]),
            ),
        );
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![outputs[0].value().clone().into_value().atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[rows, 3, 2] .
                let %1:f32[3, rows, 2] = transpose [permutation=[1, 0, 2]] %0
                    %2:f32[3, rows, 1, 2] = reshape [shape=[3, rows, 1, 2]] %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_reshape_batching_unsupported_extents() {
        // Packed padding remains inaccessible: a reshape without a ragged contract must reject the input rather
        // than return a dense batch that forgets its per-item extents.
        let array =
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[1_f32, 2., 3., 4., 5., 6.]).unwrap();
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let extents = Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[1_i32, 3]).unwrap();
        let input = ArrayBatch::new(array, BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, extents, variable, vec![0])])
            .unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array>::new(), 2);
        let operation = ReshapeOperation::new(Shape::new(vec![Dimension::Static(3)]));
        assert_eq!(
            operation.batch(&context, &EmptyRegionDriver, &[input]).unwrap_err(),
            BatchingError::UnsupportedOperation {
                message: format!("`{RESHAPE_OPERATION_NAME}` does not support bounded ragged array inputs"),
            },
        );

        // A first-class mapped extent cannot be embedded into the homogeneous operation's static result geometry.
        // The diagnostic directs this case to the mixed operation, which carries that extent as an input.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
        let extent = trace.input(DimensionType::new(items.clone()).into());
        let input = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(items), Dimension::Static(6)])).into(),
        );
        let input = <_ as ValueProjection<ArrayType>>::into_projected(input).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
            ProjectedContext::new(trace),
            extent,
        );
        assert_eq!(
            ReshapeOperation::new(Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
                .batch(&context, &EmptyRegionDriver, &[ArrayBatch::new(input, BatchAxis::new(0)).unwrap()])
                .unwrap_err(),
            BatchingError::UnsupportedOperation {
                message: format!(
                    "`{RESHAPE_OPERATION_NAME}` with a dynamic mapped extent requires using a dynamic reshape \
                     operation and explicit result-dimension inputs"
                ),
            },
        );
    }

    #[test]
    fn test_reshape_differentiation() {
        check_operation_differentiation!(
            @approx(step = 0.125, epsilon = 1e-9),
            operation = ReshapeOperation::new(Shape::new(vec![2.into(), 2.into()])),
            cases = [{
                primals = [Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap()],
                tangents = [Array::vector(vec![5.0, 6.0, 7.0, 8.0]).unwrap()],
                primal_outputs = [Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap()],
                tangent_outputs = [Array::matrix(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap()],
            }],
        );
    }

    #[test]
    fn test_reshape_differentiation_structural_zero_tangent() {
        // The shared all-zero fast path only applies when the tangent type can be materialized, so a direct rule call
        // reaches the body with a structural-zero tangent. It must stay structural at the reshaped tangent type
        // instead of materializing or reshaping a zero array.
        let primal = Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let tangent_type = primal.r#type().tangent().unwrap();
        let outputs = ReshapeOperation::new(Shape::new(vec![2.into(), 3.into()]))
            .jvp(
                &DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new()),
                &EmptyRegionDriver,
                &[DifferentiationDual::new(primal, MaybeZero::Zero(tangent_type)).unwrap()],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(*outputs[0].primal(), Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        assert!(outputs[0].tangent().is_zero());
        assert_eq!(outputs[0].tangent().r#type().as_ref(), &ArrayType::new_static(DataType::F64, [2, 3]));
    }

    #[test]
    fn test_reshape_differentiation_dynamic_singleton_axes() {
        // Inserting a singleton axis around a dynamic extent is structural-linear like any static reshape, so the
        // tangent is the same reshape of the input tangent and the pullback reshapes the cotangent back to the exact
        // dynamic input type without any retained extent residuals.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Static(2)]));
        let output_shape =
            Shape::new(vec![Dimension::Static(2), Dimension::Static(1), Dimension::Dynamic(rows.clone())]);
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.clone().into());
        let transposed = builder
            .add_instruction(
                ArrayOperation::<Array>::from(TransposeOperation::new([1, 0])),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(
                ArrayOperation::<Array>::from(ReshapeOperation::new(output_shape)),
                Vec::new(),
                vec![transposed],
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

        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[rows, 2], %1:f64[rows, 2] .
                let %2:f64[2, rows] = transpose [permutation=[1, 0]] %0
                    %3:f64[2, rows] = transpose [permutation=[1, 0]] %1
                    %4:f64[2, 1, rows] = reshape [shape=[2, 1, rows]] %2
                    %5:f64[2, 1, rows] = reshape [shape=[2, 1, rows]] %3
                in (%4, %5)
            "}
            .trim_end(),
        );
        let output_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(1), Dimension::Dynamic(rows)]),
        );
        assert_eq!(jvp.output_types(), vec![output_type.clone().into(), output_type.tangent().unwrap().into()]);

        // Linearization keeps the reshape in the tangent program without residuals; its pullback undoes the
        // permutation and restores the exact dynamic input cotangent type.
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 0);
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[rows, 2] .
                let %1:f64[2, rows] = transpose [permutation=[1, 0]] %0
                    %2:f64[2, 1, rows] = reshape [shape=[2, 1, rows]] %1
                in (%2)
            "}
            .trim_end(),
        );
        let pullback = linearization.pullback().unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[2, 1, rows] .
                let %1:f64[2, rows] = reshape [shape=[2, rows]] %0
                    %2:f64[rows, 2] = transpose [permutation=[1, 0]] %1
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(pullback.output_types(), vec![input_type.cotangent().unwrap().into()]);
    }

    #[test]
    fn test_reshape_transposition() {
        check_operation_transposition!(
            @exact,
            operation = ReshapeOperation::new(Shape::new(vec![2.into(), 3.into()])),
            cases = [{
                inputs = [(@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![6.into()]))))],
                output_cotangents = [Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()],
                input_cotangents = [Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()],
            }],
        );
        check_operation_transposition!(
            @exact,
            operation = ReshapeOperation::new(
                Shape::new(vec![6.into()]),
            ),
            cases = [{
                inputs = [(@linear(type = ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![2.into(), 3.into()]),
                )))],
                output_cotangents = [Array::vector(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap()],
                input_cotangents = [Array::matrix(2, 3, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap()],
            }],
        );

        // Reshaping back to the input shape restores its complete cotangent type after the forward reshape has
        // intentionally cleared physical layout metadata.
        let layout = Layout::Strided(StridedLayout::new(vec![8]));
        let placed_input_type = ArrayType::new(DataType::F64, Shape::new(vec![6.into()]))
            .with_layout(layout)
            .with_memory(Memory::Host { pinned: true });
        let placed_output_type = ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into()]))
            .with_memory(Memory::Host { pinned: true });
        check_operation_transposition!(
            @exact,
            operation = ReshapeOperation::new(Shape::new(vec![2.into(), 3.into()])),
            cases = [{
                inputs = [(@linear(type = placed_input_type.clone()))],
                output_cotangents = [Array::from_elements::<f64>(
                    placed_output_type,
                    &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                ).unwrap()],
                input_cotangents = [Array::from_elements::<f64>(
                    placed_input_type,
                    &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                ).unwrap()],
            }],
        );
    }

    #[test]
    fn test_reshape_transposition_dynamic_singleton_axes() {
        // Removing singleton axes around a dynamic extent transposes directly: the inverse reshape reinserts the
        // singletons from the dynamic input cotangent type, so no runtime extent has to be retained.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(0, Some(9)).unwrap());
        let input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(1), Dimension::Dynamic(rows.clone()), Dimension::Static(2)]),
        );
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.clone().into());
        let output = builder
            .add_instruction(
                ArrayOperation::<Array>::from(ReshapeOperation::new(Shape::new(vec![
                    Dimension::Dynamic(rows),
                    Dimension::Static(2),
                ]))),
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
        let pullback = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[rows, 2] .
                let %1:f64[1, rows, 2] = reshape [shape=[1, rows, 2]] %0
                in (%1)
            "}
            .trim_end(),
        );
        assert_eq!(pullback.output_types(), vec![input_type.cotangent().unwrap().into()]);
    }

    #[test]
    fn test_reshape_reshape() {
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |input| input.reshape(Shape::new(vec![2.into(), 3.into()])),
            ArrayType::new_static(DataType::F64, [6]),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[6] .
                let %1:f64[2, 3] = reshape [shape=[2, 3]] %0
                in (%1)
            "}
            .trim_end(),
        );

        // An unchanged shape carries no instruction through context dispatch.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |input| input.reshape(input.r#type().shape().clone()),
            ArrayType::new_static(DataType::F64, [2, 3]),
        )
        .unwrap();
        assert!(program.instructions().is_empty());
    }

    #[test]
    fn test_reshape_reshape_invalid_output_count() {
        // Exercise both missing and extra results without involving larger operation families.
        let array = Array::from_elements(ArrayType::new_static(DataType::I32, [1, 2]), &[3_i32, 7]).unwrap();
        for output_count in [0, 2] {
            let context = InvalidOutputContext::<Array, ReshapeOperation>(output_count, PhantomData);
            let input = context.lift(array.clone()).unwrap();
            assert!(matches!(
                input.reshape(Shape::new(vec![2.into(), 1.into()])),
                Err(ProgramError::InvalidOutputCount { expected: 1, actual }) if actual == output_count,
            ));
        }
    }

    #[test]
    fn test_reshape_reshape_dynamic_singleton_axes() {
        // Inserting and removing static singleton axes around a dynamic extent stages the homogeneous reshape with
        // the dynamic identity carried in its shape; no explicit extent inputs are needed.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Static(4)]));
        let (output_types, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |input| {
                let input = <_ as ValueProjection<ArrayType>>::into_projected(input)?;
                let expanded = input.expand_dimensions(0)?.expand_dimensions(-1)?;
                let squeezed = expanded.squeeze(0)?;
                Ok(squeezed.squeeze_all()?.into_value())
            },
            ArrayIrType::Array(input_type.clone()),
        )
        .unwrap();
        assert_eq!(output_types, ArrayIrType::Array(input_type));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[rows, 4] .
                let %1:f64[1, rows, 4] = reshape [shape=[1, rows, 4]] %0
                    %2:f64[1, rows, 4, 1] = reshape [shape=[1, rows, 4, 1]] %1
                    %3:f64[rows, 4, 1] = reshape [shape=[rows, 4, 1]] %2
                    %4:f64[rows, 4] = reshape [shape=[rows, 4]] %3
                in (%4)
            "}
            .trim_end(),
        );

        // The rejected geometry is the same as at the type level, and nothing is staged for it.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = trace.input(ArrayType::new_static(DataType::F64, [6]).into());
        let input = <_ as ValueProjection<ArrayType>>::into_projected(input).unwrap();
        assert_eq!(
            input.reshape(Shape::new(vec![Dimension::Dynamic(rows), Dimension::Static(2)])),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` requires explicit result-dimension inputs for a dynamic output shape that \
                 does not only insert or remove singleton axes"
            )))),
        );
        assert!(trace.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_reshape_reshape_to_sizes() {
        let input = Array::vector(vec![1i32, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(input.reshape_to_sizes(&[3, -1]), Array::matrix(3, 2, vec![1i32, 2, 3, 4, 5, 6]));
        assert_eq!(Array::scalar(7i32).unwrap().reshape_to_sizes(&[-1]), Array::vector(vec![7i32]));
        assert_eq!(input.reshape_to_sizes(&[6]), Ok(input.clone()));
        assert_eq!(
            Array::vector(Vec::<i32>::new()).unwrap().reshape_to_sizes(&[2, -1]),
            Array::matrix(2, 0, Vec::<i32>::new()),
        );
        assert_eq!(Array::scalar(7i32).unwrap().reshape_to_sizes(&[]), Array::scalar(7i32));
    }

    #[test]
    fn test_reshape_reshape_to_sizes_invalid_dimensions() {
        let input = Array::vector(vec![1i32, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(
            input.reshape_to_sizes(&[-1, -1]),
            Err(ProgramError::from(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` accepts at most one inferred `-1` dimension"
            )))),
        );
        assert_eq!(
            input.reshape_to_sizes(&[-2]),
            Err(ProgramError::from(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` dimensions must be nonnegative or the inferred size `-1`"
            )))),
        );
        assert_eq!(
            input.reshape_to_sizes(&[4, -1]),
            Err(ProgramError::from(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` inferred dimension does not divide the input element count"
            )))),
        );
        assert_eq!(
            input.reshape_to_sizes(&[5]),
            Err(ProgramError::from(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` output element count 5 differs from input element count 6"
            )))),
        );
        assert_eq!(
            input.reshape_to_sizes(&[isize::MAX; 3]),
            Err(ProgramError::from(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` output element count does not fit in `usize`"
            )))),
        );
        assert_eq!(
            Array::vector(Vec::<i32>::new()).unwrap().reshape_to_sizes(&[0, -1]),
            Err(ProgramError::from(TypeError::invalid(format!(
                "cannot infer a `{RESHAPE_OPERATION_NAME}` dimension when another output dimension is zero"
            )))),
        );
        let dynamic = Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
            "size",
            DimensionBounds::new(0, Some(9)).unwrap(),
        ))]);
        assert_eq!(
            ArrayType::new(DataType::F32, dynamic).reshape_to_sizes(&[-1]),
            Err(ProgramError::from(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` size inference requires a known input element count"
            )))),
        );
    }

    #[test]
    fn test_reshape_flatten() {
        let input = Array::matrix(2, 2, vec![1i32, 2, 3, 4]).unwrap();
        assert_eq!(input.flatten(), Array::vector(vec![1i32, 2, 3, 4]));
        assert_eq!(Array::scalar(7i32).unwrap().flatten(), Array::vector(vec![7i32]));
        assert_eq!(Array::matrix(0, 2, Vec::<i32>::new()).unwrap().flatten(), Array::vector(Vec::<i32>::new()));

        // An existing dynamic vector needs no new dimension identity or element-count proof.
        let vector = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("length", DimensionBounds::unbounded()))]),
        )
        .with_memory(Memory::Host { pinned: true });
        assert_eq!(vector.flatten(), Ok(vector.clone()));
    }

    #[test]
    fn test_reshape_expand_dimensions() {
        let input = Array::vector(vec![1i32, 2]).unwrap();
        assert_eq!(input.expand_dimensions(0), Array::matrix(1, 2, vec![1i32, 2]));
        assert_eq!(input.expand_dimensions(-1), Array::matrix(2, 1, vec![1i32, 2]));
        assert_eq!(Array::scalar(7i32).unwrap().expand_dimensions(-1), Array::vector(vec![7i32]));
        assert!(matches!(
            input.expand_dimensions(2),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "axis 2 is out of bounds for rank 2",
        ));

        // A dynamic extent is preserved, together with the memory placement, because the inserted singleton keeps the
        // non-singleton dimensions in order.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let dynamic_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Static(3)]))
                .with_memory(Memory::Host { pinned: true });
        assert_eq!(
            dynamic_type.expand_dimensions(1),
            Ok(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(rows), Dimension::Static(1), Dimension::Static(3)]),
            )
            .with_memory(Memory::Host { pinned: true })),
        );
    }

    #[test]
    fn test_reshape_squeeze() {
        let input = Array::matrix(2, 1, vec![1i32, 2]).unwrap();
        assert_eq!(input.squeeze(-1), Array::vector(vec![1i32, 2]));
        assert_eq!(input.squeeze(Vec::<usize>::new()), Ok(input.clone()));
        assert!(matches!(
            input.squeeze(0),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == format!("`{RESHAPE_OPERATION_NAME}` cannot squeeze axis 0 whose size is not one"),
        ));
        assert!(matches!(
            input.squeeze([1, -1]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "axes contain duplicate axis 1",
        ));

        // Only statically known singleton axes can be squeezed; a dynamic axis is rejected even when its bounds admit
        // size one, while static singletons beside a dynamic extent are removed with the extent preserved.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(2)).unwrap());
        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Static(1), Dimension::Dynamic(rows.clone()), Dimension::Static(1)]),
        );
        assert_eq!(
            dynamic_type.squeeze([0, 2]),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows)]))),
        );
        assert!(matches!(
            dynamic_type.squeeze(1),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == format!("`{RESHAPE_OPERATION_NAME}` cannot squeeze axis 1 whose size is not one"),
        ));
    }

    #[test]
    fn test_reshape_squeeze_all() {
        let input = Array::matrix(1, 1, vec![7i32]).unwrap();
        assert_eq!(input.squeeze_all(), Array::scalar(7i32));
        let vector = Array::vector(vec![1i32, 2]).unwrap();
        assert_eq!(vector.squeeze_all(), Ok(vector));

        // Every static singleton is removed around a dynamic extent, which is itself never treated as a singleton.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(2)).unwrap());
        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                Dimension::Static(1),
                Dimension::Dynamic(rows.clone()),
                Dimension::Static(2),
                Dimension::Static(1),
            ]),
        );
        assert_eq!(
            dynamic_type.squeeze_all(),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows), Dimension::Static(2)]))),
        );
    }

    #[test]
    fn test_array_type_reshape() {
        // Dynamic dimensions can only be reshaped without explicit dimension inputs when the non-singleton
        // dimensions, and therefore their symbolic identities, keep their order. Other runtime relationships require
        // the mixed reshape operation and its explicit result-dimension inputs.
        let static_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(6)]));
        let dynamic_shape = Shape::new(vec![
            Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
            Dimension::Static(3),
        ]);
        let dynamic_type = ArrayType::new(DataType::F64, dynamic_shape.clone());
        let dynamic_output_error = format!(
            "`{RESHAPE_OPERATION_NAME}` requires explicit result-dimension inputs for a dynamic output shape that does \
             not only insert or remove singleton axes"
        );
        assert_eq!(
            dynamic_type.reshape(Shape::new(vec![Dimension::Static(6)])),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` requires explicit result-dimension inputs for a dynamic input shape"
            )))),
        );
        assert_eq!(
            static_type.reshape(dynamic_shape.clone()),
            Err(ProgramError::Type(TypeError::invalid(dynamic_output_error.clone()))),
        );
        assert_eq!(
            ReshapeOperation::new(dynamic_shape.clone()).infer_output_types(std::slice::from_ref(&static_type), &[]),
            Err(TypeError::invalid(dynamic_output_error.clone())),
        );
        assert_eq!(
            Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap().reshape(dynamic_shape),
            Err(ProgramError::Type(TypeError::invalid(dynamic_output_error.clone()))),
        );

        // Reshaping a dynamically sized type to its own shape short-circuits as the identity.
        assert_eq!(dynamic_type.reshape(dynamic_type.shape().clone()), Ok(dynamic_type.clone()));

        // Element counts that do not fit are reported with the side they belong to.
        let huge_shape = Shape::new(vec![Dimension::Static(usize::MAX), Dimension::Static(2)]);
        assert_eq!(
            ArrayType::new(DataType::F64, huge_shape.clone()).reshape(Shape::new(vec![Dimension::Static(2)])),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` input shape [18446744073709551615, 2] element count does not fit in usize"
            )))),
        );
        assert_eq!(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])).reshape(huge_shape),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` output shape [18446744073709551615, 2] element count does not fit in usize"
            )))),
        );

        // A static zero product does not justify anonymous dynamic output dimensions unless a permutation identifies
        // their runtime source. A fully static zero-sized target needs no runtime extent source.
        let trailing = DimensionVariable::new("trailing", DimensionBounds::unbounded());
        let zero_dynamic_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0), Dimension::Dynamic(trailing.clone())]));
        assert_eq!(
            zero_dynamic_type.reshape(Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
                Dimension::Static(0)
            ])),
            Err(ProgramError::Type(TypeError::invalid(dynamic_output_error))),
        );
        assert_eq!(
            zero_dynamic_type.reshape(Shape::new(vec![Dimension::Static(0)])),
            Ok(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0)]))),
        );
        assert_eq!(
            zero_dynamic_type
                .transpose([1, 0])
                .unwrap()
                .reshape(Shape::new(vec![Dimension::Dynamic(trailing.clone()), Dimension::Static(0),]),),
            Ok(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(trailing), Dimension::Static(0)]),)),
        );

        // A non-identity reshape preserves memory placement but clears a layout whose output strides cannot be
        // inferred from the logical target shape alone.
        let placed_type = static_type
            .clone()
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
            .with_memory(Memory::Host { pinned: true });
        assert_eq!(
            placed_type.reshape(Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])),
            Ok(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
                .with_memory(Memory::Host { pinned: true })),
        );
    }

    #[test]
    fn test_array_type_reshape_dynamic_singleton_axes() {
        // A dynamic output shape is accepted when it only inserts or removes static singleton axes, because the
        // ordered non-singleton dimensions, and with them the dynamic identities, are preserved. Like every other
        // non-identity reshape it keeps the memory placement and clears the layout.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::new(0, Some(4)).unwrap());
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                Dimension::Dynamic(rows.clone()),
                Dimension::Static(1),
                Dimension::Dynamic(columns.clone()),
            ]),
        )
        .with_layout(Layout::Strided(StridedLayout::new(vec![16, 16, 4])))
        .with_memory(Memory::Host { pinned: true });
        assert_eq!(
            input_type.reshape(Shape::new(vec![
                Dimension::Static(1),
                Dimension::Dynamic(rows.clone()),
                Dimension::Dynamic(columns.clone()),
                Dimension::Static(1),
            ])),
            Ok(ArrayType::new(
                DataType::F32,
                Shape::new(vec![
                    Dimension::Static(1),
                    Dimension::Dynamic(rows.clone()),
                    Dimension::Dynamic(columns.clone()),
                    Dimension::Static(1),
                ]),
            )
            .with_memory(Memory::Host { pinned: true })),
        );
        assert_eq!(
            input_type.reshape(Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Dynamic(columns.clone())])),
            Ok(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Dynamic(columns.clone())]),
            )
            .with_memory(Memory::Host { pinned: true })),
        );

        // The check applies to the permuted input, so a permutation may reorder the dynamic dimensions as long as the
        // output shape follows that permuted order.
        assert_eq!(
            input_type.transpose([2, 1, 0]).unwrap().reshape(Shape::new(vec![
                Dimension::Dynamic(columns.clone()),
                Dimension::Static(1),
                Dimension::Static(1),
                Dimension::Dynamic(rows.clone()),
            ]),),
            Ok(ArrayType::new(
                DataType::F32,
                Shape::new(vec![
                    Dimension::Dynamic(columns.clone()),
                    Dimension::Static(1),
                    Dimension::Static(1),
                    Dimension::Dynamic(rows.clone()),
                ]),
            )
            .with_memory(Memory::Host { pinned: true })),
        );

        // Reordering, merging, or resizing the non-singleton dimensions needs explicit result-dimension inputs, even
        // when a static singleton is inserted or removed at the same time.
        let dynamic_output_error = format!(
            "`{RESHAPE_OPERATION_NAME}` requires explicit result-dimension inputs for a dynamic output shape that does \
             not only insert or remove singleton axes"
        );
        assert_eq!(
            input_type.reshape(Shape::new(vec![Dimension::Dynamic(columns.clone()), Dimension::Dynamic(rows.clone())])),
            Err(ProgramError::Type(TypeError::invalid(dynamic_output_error.clone()))),
        );
        assert_eq!(
            input_type.reshape(Shape::new(vec![
                Dimension::Dynamic(rows.clone()),
                Dimension::Static(2),
                Dimension::Dynamic(columns.clone()),
            ])),
            Err(ProgramError::Type(TypeError::invalid(dynamic_output_error.clone()))),
        );
        assert_eq!(
            input_type.reshape(Shape::new(vec![Dimension::Dynamic(rows.clone())])),
            Err(ProgramError::Type(TypeError::invalid(dynamic_output_error))),
        );

        // Sharding is inferred over the non-singleton dimensions, so a sharded dynamic axis keeps its placement across
        // the inserted singleton.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Static(4)]))
                .with_sharding(
                    Sharding::new(
                        mesh.clone(),
                        vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
                    )
                    .unwrap(),
                )
                .unwrap();
        assert_eq!(
            sharded_type.reshape(Shape::new(vec![
                Dimension::Static(1),
                Dimension::Dynamic(rows.clone()),
                Dimension::Static(4),
            ])),
            Ok(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Static(1), Dimension::Dynamic(rows), Dimension::Static(4)]),
            )
            .with_sharding(
                Sharding::new(
                    mesh,
                    vec![
                        ShardingDimension::replicated(),
                        ShardingDimension::sharded(["x"]),
                        ShardingDimension::replicated(),
                    ],
                )
                .unwrap(),
            )
            .unwrap()),
        );
    }

    #[test]
    fn test_array_type_reshape_sharding() {
        // Singleton insertion preserves the corresponding non-singleton dimension placement.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert_eq!(
            input_type.reshape(Shape::new(vec![Dimension::Static(1), Dimension::Static(8), Dimension::Static(1)])),
            Ok(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Static(1), Dimension::Static(8), Dimension::Static(1)])
            )
            .with_sharding(
                Sharding::new(
                    mesh,
                    vec![
                        ShardingDimension::replicated(),
                        ShardingDimension::sharded(["x"]),
                        ShardingDimension::replicated(),
                    ],
                )
                .unwrap(),
            )
            .unwrap())
        );

        // A singleton that stays at the same position retains its own placement.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1), Dimension::Static(8)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap(),
            )
            .unwrap();
        assert_eq!(input_type.reshape(input_type.shape().clone()), Ok(input_type.clone()));
    }

    #[test]
    fn test_array_type_reshape_sharding_preserves_reduction_and_manual_axes() {
        // Reshape regroups ranked dimensions but leaves the reduction-state (unreduced/reduced) and varying-manual
        // axis sets untouched, since those describe mesh axes that do not correspond to ranked array dimensions.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("r", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8), Dimension::Static(6)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap()
                    .with_reduced_axes(["r"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            input_type.reshape(Shape::new(vec![Dimension::Static(8), Dimension::Static(2), Dimension::Static(3)])),
            Ok(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Static(8), Dimension::Static(2), Dimension::Static(3)])
            )
            .with_sharding(
                Sharding::new(
                    mesh,
                    vec![
                        ShardingDimension::sharded(["x"]),
                        ShardingDimension::replicated(),
                        ShardingDimension::replicated(),
                    ],
                )
                .unwrap()
                .with_reduced_axes(["r"])
                .unwrap(),
            )
            .unwrap())
        );

        // Many-to-many regrouping is supported when every participating dimension is replicated, and the varying
        // manual axes ride along unchanged.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(6)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
                    .unwrap()
                    .with_varying_manual_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            input_type.reshape(Shape::new(vec![Dimension::Static(3), Dimension::Static(4)])),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Static(4)]))
                .with_sharding(
                    Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::replicated()],)
                        .unwrap()
                        .with_varying_manual_axes(["x"])
                        .unwrap(),
                )
                .unwrap())
        );
    }

    #[test]
    fn test_array_type_reshape_with_output_sharding() {
        // Explicit output sharding can request a valid redistribution that inference would not choose.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let requested =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                .unwrap();
        assert_eq!(
            input_type.reshape_with_output_sharding(
                Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]),
                Some(requested.clone())
            ),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]))
                .with_sharding(requested)
                .unwrap()),
        );
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let auto_input = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(Sharding::new(auto_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert_eq!(
            auto_input.reshape(Shape::new(vec![Dimension::Static(2), Dimension::Static(4)])),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]))
                .with_sharding(
                    Sharding::new(auto_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],)
                        .unwrap(),
                )
                .unwrap()),
        );

        // A requested sharding is validated against the sharding contract (rank, mesh, mesh-axis kinds, and
        // reduction and manual-axis state) independently of the split/merge inference. A merge that inference rejects
        // as non-contiguous, and a split whose factor the mesh axis does not divide, are both accepted when the caller
        // states the placement explicitly, because the requested placement is the caller's own contract.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let non_contiguous_input =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]))
                .with_sharding(
                    Sharding::new(
                        mesh.clone(),
                        vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
                    )
                    .unwrap(),
                )
                .unwrap();
        assert_eq!(
            non_contiguous_input.reshape(Shape::new(vec![Dimension::Static(8)])),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` cannot preserve non-contiguous sharding across a merge"
            )))),
        );
        let merged_requested = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(
            non_contiguous_input
                .reshape_with_output_sharding(Shape::new(vec![Dimension::Static(8)]), Some(merged_requested.clone())),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
                .with_sharding(merged_requested)
                .unwrap()),
        );
        let odd_input = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(6)]))
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert_eq!(
            odd_input.reshape(Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` cannot distribute sharding across the requested split factors"
            )))),
        );
        let odd_requested =
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap();
        assert_eq!(
            odd_input.reshape_with_output_sharding(
                Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]),
                Some(odd_requested.clone())
            ),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]))
                .with_sharding(odd_requested)
                .unwrap()),
        );
    }

    #[test]
    fn test_array_type_reshape_sharding_zero_sized() {
        // Zero-product reshapes preserve fully replicated metadata. A sharded dynamic axis is ambiguous without an
        // explicit output request, and remains available to a caller that supplies one.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let replicated_input = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                Dimension::Static(0),
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
            ]),
        )
        .with_sharding(
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
                .unwrap()
                .with_varying_manual_axes(["x"])
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            replicated_input.reshape(Shape::new(vec![Dimension::Static(0)])),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(0)]))
                .with_sharding(
                    Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
                        .unwrap()
                        .with_varying_manual_axes(["x"])
                        .unwrap(),
                )
                .unwrap()),
        );
        let sharded_dynamic_input = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                Dimension::Static(0),
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
            ]),
        )
        .with_sharding(
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            sharded_dynamic_input.reshape(Shape::new(vec![Dimension::Static(0)])),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` requires explicit output sharding for an ambiguous zero-sized reshape"
            )))),
        );
        let zero_requested = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(
            sharded_dynamic_input
                .reshape_with_output_sharding(Shape::new(vec![Dimension::Static(0)]), Some(zero_requested.clone())),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(0)]))
                .with_sharding(zero_requested)
                .unwrap()),
        );
    }

    #[test]
    fn test_array_type_reshape_sharding_sharded_singleton_axes() {
        // A sharded singleton holds no elements but records a placement. It keeps that placement when it stays at the
        // same index, and follows an unambiguous move to the only unmatched output singleton. Dropping it or moving
        // it ambiguously is rejected instead of being replicated silently, and an explicit output sharding places it.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharded_type = |dimensions: Vec<Dimension>, sharding: Vec<ShardingDimension>| {
            ArrayType::new(DataType::F32, Shape::new(dimensions))
                .with_sharding(Sharding::new(mesh.clone(), sharding).unwrap())
                .unwrap()
        };
        let sharded = |axis: &str| ShardingDimension::sharded([axis]);
        let replicated = ShardingDimension::replicated;
        let input_type =
            sharded_type(vec![Dimension::Static(1), Dimension::Static(8)], vec![sharded("x"), replicated()]);

        // Same index kept, alongside a replicated split of the non-singleton dimension and a new output singleton.
        assert_eq!(
            input_type.reshape(Shape::new(vec![Dimension::Static(1), Dimension::Static(2), Dimension::Static(4)])),
            Ok(sharded_type(
                vec![Dimension::Static(1), Dimension::Static(2), Dimension::Static(4)],
                vec![sharded("x"), replicated(), replicated()],
            )),
        );
        assert_eq!(
            input_type.reshape(Shape::new(vec![Dimension::Static(1), Dimension::Static(8), Dimension::Static(1)])),
            Ok(sharded_type(
                vec![Dimension::Static(1), Dimension::Static(8), Dimension::Static(1)],
                vec![sharded("x"), replicated(), replicated()],
            )),
        );

        // Unambiguous move to the only unmatched output singleton.
        assert_eq!(
            input_type.reshape(Shape::new(vec![Dimension::Static(8), Dimension::Static(1)])),
            Ok(sharded_type(vec![Dimension::Static(8), Dimension::Static(1)], vec![replicated(), sharded("x")])),
        );

        // Removal of the sharded singleton, and an ambiguous move with two candidate output singletons.
        assert_eq!(
            input_type.reshape(Shape::new(vec![Dimension::Static(8)])),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` requires explicit output sharding to place the sharded singleton input \
                 axis 0 in the output"
            )))),
        );
        assert_eq!(
            input_type.reshape(Shape::new(vec![Dimension::Static(8), Dimension::Static(1), Dimension::Static(1)])),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` requires explicit output sharding to place the sharded singleton input \
                 axis 0 in the output"
            )))),
        );

        // A replicated singleton may be dropped freely, and only the sharded singletons participate in matching.
        let replicated_singleton_type =
            sharded_type(vec![Dimension::Static(1), Dimension::Static(8)], vec![replicated(), sharded("x")]);
        assert_eq!(
            replicated_singleton_type.reshape(Shape::new(vec![Dimension::Static(8)])),
            Ok(sharded_type(vec![Dimension::Static(8)], vec![sharded("x")])),
        );

        // Two moved sharded singletons are never assigned to output singletons by position, while a singleton kept at
        // its index leaves the remaining one free to move unambiguously.
        let two_singletons_type = sharded_type(
            vec![Dimension::Static(1), Dimension::Static(1), Dimension::Static(8)],
            vec![sharded("x"), sharded("y"), replicated()],
        );
        assert_eq!(
            two_singletons_type.reshape(Shape::new(vec![
                Dimension::Static(2),
                Dimension::Static(4),
                Dimension::Static(1),
                Dimension::Static(1),
            ])),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` requires explicit output sharding to place the sharded singleton input \
                 axis 0 in the output"
            )))),
        );
        assert_eq!(
            two_singletons_type.reshape(Shape::new(vec![
                Dimension::Static(8),
                Dimension::Static(1),
                Dimension::Static(1)
            ])),
            Ok(sharded_type(
                vec![Dimension::Static(8), Dimension::Static(1), Dimension::Static(1)],
                vec![replicated(), sharded("y"), sharded("x")],
            )),
        );

        // The explicit output sharding is the escape hatch for every rejected placement above.
        let requested = Sharding::new(mesh.clone(), vec![replicated(), sharded("x"), replicated()]).unwrap();
        assert_eq!(
            input_type.reshape_with_output_sharding(
                Shape::new(vec![Dimension::Static(8), Dimension::Static(1), Dimension::Static(1)]),
                Some(requested.clone())
            ),
            Ok(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Static(8), Dimension::Static(1), Dimension::Static(1)]),
            )
            .with_sharding(requested)
            .unwrap()),
        );
        let dropped = Sharding::new(mesh, vec![replicated()]).unwrap();
        assert_eq!(
            input_type.reshape_with_output_sharding(Shape::new(vec![Dimension::Static(8)]), Some(dropped.clone())),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
                .with_sharding(dropped)
                .unwrap()),
        );
    }

    #[test]
    fn test_array_reshape() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let reshaped = matrix.reshape(Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])).unwrap();
        assert_eq!(reshaped.r#type().into_owned(), ArrayType::new_static(DataType::F64, [3, 2]));
        assert_eq!(reshaped.to_f64s(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(matches!(
            matrix.reshape(Shape::new(vec![Dimension::Static(4)])),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == format!("`{RESHAPE_OPERATION_NAME}` changes the number of elements"),
        ));

        // Reshaping preserves logical order independently of the input's physical placement.
        let input_type =
            ArrayType::new_static(DataType::U16, [2, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![8, 2])));
        let matrix = Array::from_elements(input_type, &[1u16, 2, 3, 4, 5, 6]).unwrap();
        let reshaped = matrix.reshape(Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])).unwrap();
        assert_eq!(reshaped.elements::<u16>(), Ok(vec![1, 2, 3, 4, 5, 6]));
        assert_eq!(reshaped.storage_bytes(), [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0]);
    }

    #[test]
    fn test_array_reshape_with_output_sharding() {
        let input = Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(input.reshape_with_output_sharding([2, 3], None), input.reshape([2, 3]));

        // Explicit placement belongs to the reshape result while logical element order remains unchanged.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap();
        let output_type = ArrayType::new_static(DataType::F64, [2, 3]).with_sharding(sharding.clone()).unwrap();
        assert_eq!(
            input.reshape_with_output_sharding([2, 3], Some(sharding)),
            Ok(Array::from_elements::<f64>(output_type, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
        );
    }

    #[test]
    fn test_dynamic_reshape() {
        let operation = DynamicReshapeOperation::new();
        assert_eq!(operation.name(), RESHAPE_OPERATION_NAME);
        assert_eq!(operation.to_string(), "reshape");
        assert_eq!(operation.output_sharding(), None);
        let input_types = [
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4])),
            DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
        ];
        assert_eq!(
            operation.with_input_types(&input_types).unwrap().to_string(),
            "reshape [requires_runtime_assertion=false]"
        );

        // The requested output sharding renders after the assertion requirement and
        // switches an assertion-bearing operation to the bracketed form as well.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let operation = DynamicReshapeOperation::new().with_output_sharding(sharding.clone());
        assert_eq!(operation.output_sharding(), Some(&sharding));
        assert_eq!(operation.to_string(), "reshape [output_sharding={mesh<['x'=2:explicit]>, [{'x'}]}]");
        let input_types = [
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2, 2])),
            DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
        ];
        assert_eq!(
            operation.with_input_types(&input_types).unwrap().to_string(),
            indoc! {"
                reshape [
                    requires_runtime_assertion=false,
                    output_sharding={mesh<['x'=2:explicit]>, [{'x'}]},
                ]
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dynamic_reshape_requires_runtime_assertion() {
        let operation = DynamicReshapeOperation::new();
        assert_eq!(operation, DynamicReshapeOperation::default());
        assert!(operation.requires_runtime_assertion());
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));

        let exact = [
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4])),
            DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
        ];
        let operation = operation.with_input_types(&exact).unwrap();
        assert!(!operation.requires_runtime_assertion());
        assert_eq!(operation.effects().classes(), EffectClasses::NONE);

        // Reuse cannot silently drop a runtime check; explicit refinement can restore it for an independent extent.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let unproven = [exact[0].clone(), DimensionType::new(extent).into()];
        assert_eq!(
            operation.infer_output_types(&unproven, &[]),
            Err(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` was constructed without a runtime element-count check but these input \
                 types require one",
            ))),
        );
        let operation = operation.with_input_types(&unproven).unwrap();
        assert!(operation.requires_runtime_assertion());
        assert_eq!(operation.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        let operation = operation.with_input_types(&exact).unwrap();
        assert!(!operation.requires_runtime_assertion());
        assert_eq!(operation.effects().classes(), EffectClasses::NONE);
    }

    #[test]
    fn test_dynamic_reshape_type_inference() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone()), 4.into()]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let two = ArrayIrType::from(DimensionValue::constant(2).unwrap().r#type().into_owned());
        let input_types =
            vec![input.clone().into(), DimensionType::new(extent.clone()).into(), two.clone(), two.clone()];
        let operation = DynamicReshapeOperation::new().with_input_types(&input_types).unwrap();
        let output =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone()), 2.into(), 2.into()]))
                .with_sharding(
                    Sharding::new(
                        mesh,
                        vec![
                            ShardingDimension::sharded(["x"]),
                            ShardingDimension::replicated(),
                            ShardingDimension::replicated(),
                        ],
                    )
                    .unwrap(),
                )
                .unwrap();
        // The proof must not survive reuse with independent dynamic identities. Explicit replication lets the
        // count-proof diagnostic be reached before sharding inference for the unrelated identity.
        let other = DimensionType::new(DimensionVariable::new("other", DimensionBounds::new(1, Some(9)).unwrap()));
        let unsharded_input = ArrayIrType::Array(ArrayType::new(DataType::F32, input.shape().clone()));
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    input_types = [
                        input.clone().into(),
                        DimensionType::new(extent.clone()).into(),
                        two.clone(),
                        two.clone(),
                    ],
                    output_types = [output.into()],
                },
                {
                    input_types = [unsharded_input.clone(), other.into(), two.clone(), two.clone()],
                    error = format!(
                        "`{RESHAPE_OPERATION_NAME}` was constructed without a runtime element-count check but these \
                         input types require one",
                    ),
                },
                {
                    input_types = [],
                    error = format!("`{RESHAPE_OPERATION_NAME}` expects an array followed by its output extents"),
                },
            ],
        );
        check_operation_type_inference!(
            operation = DynamicReshapeOperation::new(),
            cases = [
                {
                    input_types = [
                        ArrayIrType::Array(ArrayType::new_static(DataType::F32, [6])),
                        two.clone(),
                        two.clone(),
                    ],
                    error = format!("`{RESHAPE_OPERATION_NAME}` changes the number of elements"),
                },
                {
                    input_types = [
                        ArrayIrType::Array(ArrayType::new_static(DataType::F32, [1, 1])),
                        two.clone(),
                        two.clone(),
                    ],
                    error = format!("`{RESHAPE_OPERATION_NAME}` changes the number of elements"),
                },
            ],
        );
        assert_eq!(operation.effects().classes(), EffectClasses::NONE);
        assert_eq!(
            DynamicReshapeOperation::new().effects().classes(),
            EffectClasses::single(EffectClass::OrderedAssertion)
        );
        assert_eq!(
            operation.infer_output_types(&input_types, &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)]),
            Err(TypeError::invalid("expected 0 regions but got 1"))
        );
        assert_eq!(
            operation.infer_output_types(&[two.clone(), two], &[]),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
    }

    #[test]
    fn test_dynamic_reshape_type_inference_identity_instantiation() {
        let bounds = DimensionBounds::new(1, Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let source_dimension_type = DimensionType::new(source.clone());
        let source_array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source.clone()), Dimension::Static(4)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = builder.add_input(source_array_type.clone().into());
        let extent = builder.add_input(source_dimension_type.into());
        let four = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap()));
        let output = builder
            .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![array, extent, four], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(program.output_types(), vec![source_array_type.into()]);

        // Instantiating the boundary identities renames the identity throughout: the reshape has no stored geometry of
        // its own, so its output type follows the renamed extent input.
        let target = DimensionVariable::new("target", bounds);
        let target_dimension_type = DimensionType::new(target.clone());
        let target_array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(target), Dimension::Static(4)]));
        let instantiated = program
            .with_instantiated_type_identities(&[
                target_array_type.clone().into(),
                target_dimension_type.clone().into(),
            ])
            .unwrap()
            .into_owned();
        assert_eq!(instantiated.output_types(), vec![target_array_type.clone().into()]);
        assert_eq!(
            instantiated.to_string(),
            indoc! {"
                lambda %0:f32[target, 4], %1:dimension<target ∈ [1, 9)> .
                let %2:dimension<4> = const 4
                    %3:f32[target, 4] = reshape %0 %1 %2
                in (%3)
            "}
            .trim_end(),
        );

        // Splicing the instantiated program into another builder keeps the caller's identity on the imported reshape.
        let mut destination = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = destination.add_input(target_array_type.clone().into());
        let extent = destination.add_input(target_dimension_type.into());
        let outputs = destination.splice_program(&instantiated, &[array, extent]).unwrap();
        let imported = destination
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                outputs,
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(imported.output_types(), vec![target_array_type.into()]);
        assert_eq!(imported.to_string(), instantiated.to_string());
    }

    #[test]
    fn test_dynamic_reshape_interpretation() {
        // A concrete composite value resolves every explicit extent input and reshapes its array member directly.
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        let rows = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let columns = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        assert_eq!(
            DynamicReshapeOperation::new().interpret(
                &EagerContext::<ArrayIrValue<Array>>::new(),
                &EmptyRegionDriver,
                &[input.clone(), rows.clone(), columns.clone()],
            ),
            Ok(vec![ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap())]),
        );
        assert!(matches!(
            DynamicReshapeOperation::new().interpret(
                &EagerContext::<ArrayIrValue<Array>>::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == format!("`{RESHAPE_OPERATION_NAME}` expects an array followed by its output extents"),
        ));

        // Repeating a nominal dimension cannot describe two different extents, even if their product happens
        // to match the input count. Interpretation must propagate the concrete capability's validation error.
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let dimensions = [
            ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 2).unwrap()),
            ArrayIrValue::Dimension(DimensionValue::new(extent_type, 3).unwrap()),
        ];
        assert_eq!(
            DynamicReshapeOperation::new().interpret(
                &EagerContext::<ArrayIrValue<Array>>::new(),
                &EmptyRegionDriver,
                &[input.clone(), dimensions[0].clone(), dimensions[1].clone()],
            ),
            Err(ProgramError::Type(
                DimensionError::InputDimensionMismatch { dimension: "extent".to_owned(), expected: 2, actual: 3 }
                    .into(),
            )),
        );
    }

    #[test]
    fn test_dynamic_reshape_partial_evaluation() {
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        let first_extent = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let second_extent = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        let output = ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        let input_type = input.r#type().into_owned();
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = DynamicReshapeOperation::new(),
            cases = [
                {
                    inputs = [
                        (@known, input.clone()),
                        (@known, first_extent.clone()),
                        (@known, second_extent.clone()),
                    ],
                    outputs = [(@known, output.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = input_type, replay = input)),
                        (@known, first_extent),
                        (@known, second_extent),
                    ],
                    outputs = [(@residual, output)],
                    residual_instructions = 1,
                },
            ],
        );

        let identity_input = ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        let identity_input_type = identity_input.r#type().into_owned();
        check_operation_partial_evaluation!(
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = DynamicReshapeOperation::new(),
            cases = [{
                inputs = [
                    (@unknown(type = identity_input_type, replay = identity_input.clone())),
                    (@known, ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap())),
                    (@known, ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap())),
                ],
                outputs = [(@residual, identity_input)],
                residual_instructions = 0,
            }],
        );

        // An unused runtime reshape whose element counts are not proven equal must survive simplification, because
        // its ordered assertion still validates the input-dependent element count.
        let input_type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4]));
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap()));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.clone());
        let extent = builder.add_input(ArrayIrType::Dimension(extent_type));
        builder
            .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![input, extent], None)
            .unwrap();
        let unproven = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![input],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap()
            .into_simplified()
            .unwrap();
        assert_eq!(
            unproven.to_string(),
            indoc! {"
                lambda %0:f32[4], %1:dimension<extent ∈ [1, 9)> .
                let %2:f32[extent] = reshape %0 %1
                in (%0)
            "}
            .trim_end(),
        );

        // A proven reshape has no observable consequence when unused and is eliminated.
        let extent_type = DimensionValue::constant(4).unwrap().r#type().into_owned();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.clone());
        let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let operation = DynamicReshapeOperation::new().with_input_types(&[input_type, extent_type.into()]).unwrap();
        builder.add_instruction(operation, Vec::new(), vec![input, extent], None).unwrap();
        let proven = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![input],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap()
            .into_simplified()
            .unwrap();
        assert!(proven.instructions().is_empty());
    }

    #[test]
    fn test_dynamic_reshape_batching() {
        // A mapped array input is lifted onto a leading batch axis while the replicated extents pass through. The
        // mixed batch and policy types differ from the array ones that `check_operation_batching!` constructs, so
        // these cases stay explicit.
        let two = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let three = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            two.clone(),
        );
        let values = (0..12).map(|value| value as f64).collect::<Vec<_>>();
        let expected_output = ArrayIrBatch::new(
            ArrayIrValue::Array(
                Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2, 2, 3]), &values).unwrap(),
            ),
            BatchAxis::new(0),
        )
        .unwrap();
        let inputs = [
            ArrayIrBatch::new(ArrayIrValue::Array(Array::matrix(2, 6, values.clone()).unwrap()), BatchAxis::new(0))
                .unwrap(),
            ArrayIrBatch::replicated(two.clone()),
            ArrayIrBatch::replicated(three.clone()),
        ];
        let batched = DynamicReshapeOperation::new().batch(&context, &EmptyRegionDriver, &inputs).unwrap();
        let (outputs, evidence) = batched.into_parts();
        assert_eq!(outputs, vec![expected_output.clone()]);
        assert_eq!(evidence, Vec::<DimensionVariable>::new());

        // A non-leading mapped axis is moved to the front before the per-item geometry is lifted.
        let transposed = ArrayIrBatch::new(
            ArrayIrValue::Array(
                Array::matrix(6, 2, vec![0.0, 6.0, 1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 11.0]).unwrap(),
            ),
            BatchAxis::new(1),
        )
        .unwrap();
        let inputs = [transposed, ArrayIrBatch::replicated(two), ArrayIrBatch::replicated(three)];
        let batched = DynamicReshapeOperation::new().batch(&context, &EmptyRegionDriver, &inputs).unwrap();
        let (outputs, _) = batched.into_parts();
        assert_eq!(outputs, vec![expected_output]);
    }

    #[test]
    fn test_dynamic_reshape_batching_replicated_input() {
        // A replicated array input has no batch axis to thread through, so the operation is bound as given and its
        // output stays replicated.
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let inputs = [
            ArrayIrBatch::replicated(ArrayIrValue::Array(Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap())),
            ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap())),
            ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap())),
        ];
        let batched = DynamicReshapeOperation::new().batch(&context, &EmptyRegionDriver, &inputs).unwrap();
        let (outputs, _) = batched.into_parts();
        assert_eq!(
            outputs,
            vec![ArrayIrBatch::replicated(ArrayIrValue::Array(
                Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
            ))],
        );
    }

    #[test]
    fn test_dynamic_reshape_batching_rejects_ragged_input() {
        // Packed padding remains inaccessible: a reshape without a ragged contract must reject the input rather
        // than return a dense batch that forgets its per-item extents.
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let array = ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[1_f32, 2., 3., 4., 5., 6.]).unwrap(),
        );
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let extents =
            ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[1_i32, 3]).unwrap());
        let input = ArrayIrBatch::new(array, BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, extents, variable, vec![0])])
            .unwrap();
        let three = ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        assert_eq!(
            DynamicReshapeOperation::new().batch(&context, &EmptyRegionDriver, &[input, three]).unwrap_err(),
            BatchingError::UnsupportedOperation {
                message: format!("dynamic `{RESHAPE_OPERATION_NAME}` does not support bounded ragged array inputs"),
            },
        );
    }

    #[test]
    fn test_dynamic_reshape_batching_rejects_mapped_extent() {
        // Output extents are replicated shape values; a mapped per-item dimension cannot describe one output geometry.
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let input = ArrayIrBatch::new(
            ArrayIrValue::Array(Array::matrix(2, 6, (0..12).map(|value| value as f64).collect()).unwrap()),
            BatchAxis::new(0),
        )
        .unwrap();
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap()));
        let mapped_extent = ArrayIrBatch::mapped_dimension(
            ArrayIrValue::Array(Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[6_i32, 6]).unwrap()),
            BatchAxis::new(0),
            extent_type.clone(),
        )
        .unwrap();
        assert!(matches!(
            DynamicReshapeOperation::new().batch(&context, &EmptyRegionDriver, &[input, mapped_extent]),
            Err(BatchingError::MappedDimension { r#type, axis })
                if *r#type == extent_type && axis == BatchAxis::new(0),
        ));
    }

    #[test]
    fn test_dynamic_reshape_batching_rejects_malformed_inputs() {
        // The array input is mandatory and must be an array; both checks run before any batch axis is inspected.
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        assert_eq!(
            DynamicReshapeOperation::new().batch(&context, &EmptyRegionDriver, &[]).unwrap_err(),
            BatchingError::Program(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
        let extent = ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(6).unwrap()));
        assert_eq!(
            DynamicReshapeOperation::new()
                .batch(&context, &EmptyRegionDriver, &[extent.clone(), extent])
                .unwrap_err(),
            BatchingError::Type(TypeError::invalid("expected array type but got dimension type")),
        );
    }

    #[test]
    fn test_dynamic_reshape_batching_lifts_output_sharding() {
        // A requested per-item output sharding is lifted around the transform-owned mapped axis sharding.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        )
        .with_axis_sharding(ShardingDimension::sharded(["x"]));
        let input_type = ArrayType::new_static(DataType::F64, [2, 6])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let output_type = ArrayType::new_static(DataType::F64, [2, 2, 3])
            .with_sharding(
                Sharding::new(
                    mesh.clone(),
                    vec![
                        ShardingDimension::sharded(["x"]),
                        ShardingDimension::replicated(),
                        ShardingDimension::replicated(),
                    ],
                )
                .unwrap(),
            )
            .unwrap();
        let values = (0..12).map(|value| value as f64).collect::<Vec<_>>();
        let input = ArrayIrBatch::new(
            ArrayIrValue::Array(Array::from_elements::<f64>(input_type, &values).unwrap()),
            BatchAxis::new(0),
        )
        .unwrap();
        let two = ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let three = ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        let batched = DynamicReshapeOperation::new()
            .with_output_sharding(Sharding::replicated(mesh, 2))
            .batch(&context, &EmptyRegionDriver, &[input, two, three])
            .unwrap();
        let (outputs, _) = batched.into_parts();
        assert_eq!(
            outputs,
            vec![
                ArrayIrBatch::new(
                    ArrayIrValue::Array(Array::from_elements::<f64>(output_type, &values).unwrap()),
                    BatchAxis::new(0),
                )
                .unwrap()
            ],
        );
    }

    #[test]
    fn test_dynamic_reshape_batching_preserves_element_count_proof() {
        // A proven, effect-free mixed reshape must stay proven after batching over a mapped input, so that it can
        // still be eliminated when its output is unused.
        let parent = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = parent.input(ArrayType::new_static(DataType::F64, [2, 6]).into());
        let two = parent.lift(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap())).unwrap();
        let three = parent.lift(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap())).unwrap();
        let operation = DynamicReshapeOperation::new()
            .with_input_types(&[
                ArrayType::new_static(DataType::F64, [6]).into(),
                two.r#type().into_owned(),
                three.r#type().into_owned(),
            ])
            .unwrap();
        assert_eq!(operation.effects().classes(), EffectClasses::NONE);
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent.clone(), two.clone());
        let batched = operation
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    ArrayIrBatch::new(input.clone(), BatchAxis::new(0)).unwrap(),
                    ArrayIrBatch::replicated(two),
                    ArrayIrBatch::replicated(three),
                ],
            )
            .unwrap();
        let (outputs, _) = batched.into_parts();
        assert_eq!(outputs.len(), 1);
        let program = parent
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![input.atom_id().unwrap()],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 6] .
                let %1:dimension<2> = const 2
                    %2:dimension<3> = const 3
                    %3:f64[2, 2, 3] = reshape [requires_runtime_assertion=false] %0 %1 %1 %2
                in (%0)
            "}
            .trim_end(),
        );
        assert_eq!(program.instructions()[0].operation().effects().classes(), EffectClasses::NONE);
        assert!(program.into_simplified().unwrap().instructions().is_empty());

        // An unproven reshape keeps its ordered assertion after batching: dead-result elimination retains it, and an
        // invalid runtime element count still fails even though nothing consumes the output.
        let parent = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = parent.input(ArrayType::new_static(DataType::F64, [2, 6]).into());
        let extent = parent.input(
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap())).into(),
        );
        let two = parent.lift(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap())).unwrap();
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent.clone(), two);
        let batched = DynamicReshapeOperation::new()
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    ArrayIrBatch::new(input.clone(), BatchAxis::new(0)).unwrap(),
                    ArrayIrBatch::replicated(extent.clone()),
                ],
            )
            .unwrap();
        let (outputs, _) = batched.into_parts();
        assert_eq!(outputs.len(), 1);
        let program = parent
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![input.atom_id().unwrap()],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap()
            .into_simplified()
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 6], %1:dimension<extent ∈ [1, 9)> .
                let %2:dimension<2> = const 2
                    %3:f64[2, extent] = reshape %0 %2 %1
                in (%0)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.instructions()[0].operation().effects().classes(),
            EffectClasses::single(EffectClass::OrderedAssertion),
        );
        let values = ArrayIrValue::Array(Array::matrix(2, 6, (0..12).map(|value| value as f64).collect()).unwrap());
        assert_eq!(
            program.interpret(vec![values.clone(), ArrayIrValue::Dimension(DimensionValue::constant(6).unwrap())]),
            Ok(vec![values.clone()]),
        );
        assert!(matches!(
            program.interpret(vec![values, ArrayIrValue::Dimension(DimensionValue::constant(5).unwrap())]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == format!("`{RESHAPE_OPERATION_NAME}` changes the number of elements"),
        ));
    }

    #[test]
    fn test_dynamic_reshape_differentiation() {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(6)])).into());
        let first_extent = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
        let second_extent = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()));
        let output = builder
            .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![input, first_extent, second_extent], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        // Static geometry on both sides replays the mixed reshape directly on the tangent under the same constant
        // extents, which are non-differentiated shape values.
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[6], %1:f64[6] .
                let %2:dimension<2> = const 2
                    %3:dimension<3> = const 3
                    %4:f64[2, 3] = reshape [requires_runtime_assertion=false] %0 %2 %3
                    %5:f64[2, 3] = reshape [requires_runtime_assertion=false] %1 %2 %3
                in (%4, %5)
            "}
            .trim_end(),
        );
        assert_eq!(
            jvp.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0]).unwrap()),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
                ArrayIrValue::Array(Array::matrix(2, 3, vec![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_dynamic_reshape_differentiation_structural_zero_tangent() {
        // A direct rule call may carry a structural-zero array tangent. The rule must keep it structural at the output
        // tangent type without staging a linear call or reading any extent, even for dynamic geometry on both sides.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let total = DimensionVariable::new("total", DimensionBounds::new(1, Some(37)).unwrap());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(rows), Dimension::Static(4)]));
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = context.input(input_type.clone().into());
        let extent = context.input(DimensionType::new(total.clone()).into());
        let inputs = vec![
            DifferentiationDual::new(input, MaybeZero::Zero(input_type.tangent().unwrap().into())).unwrap(),
            DifferentiationDual::new_with_zero_tangent(extent).unwrap(),
        ];
        let outputs = DynamicReshapeOperation::new()
            .jvp(&DifferentiationContext::fused(context.clone()), &EmptyRegionDriver, inputs.as_slice())
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].tangent().is_zero());
        assert_eq!(
            outputs[0].tangent().r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(total)]))),
        );
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![outputs[0].primal().atom_id().unwrap()],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[rows, 4], %1:dimension<total ∈ [1, 37)> .
                let %2:f64[total] = reshape %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dynamic_reshape_differentiation_static_input_dynamic_output() {
        // A static input reshaped to a dynamic output extent cannot replay the mixed reshape directly on the tangent,
        // because its transpose would need the runtime output extent. The JVP therefore takes the retained-shape
        // linearization path, whose inverse reshape lifts the static input extents as constants, and differentiates
        // end to end through `jvp`, `linearize`, and the transposed tangent program.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F64, [6]).into());
        let extent = builder.add_input(DimensionType::new(rows.clone()).into());
        let output = builder
            .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![input, extent], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let values = ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        let tangents = ArrayIrValue::Array(Array::vector(vec![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0]).unwrap());
        let extent = ArrayIrValue::Dimension(DimensionValue::constant(6).unwrap());

        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[6], %1:dimension<rows ∈ [1, 9)>, %2:f64[6] .
                let %3:f64[rows] = reshape %0 %1
                    %4:f64[rows] = linear_call [residual_count=1] %1 %2 [
                        forward={
                            lambda %0:dimension<rows ∈ [1, 9)>, %1:f64[6] .
                            let %2:f64[rows] = reshape %1 %0
                            in (%2)
                        },
                        transpose={
                            lambda %0:dimension<rows ∈ [1, 9)>, %1:f64[rows] .
                            let %2:dimension<6> = constant [value=6]
                                %3:f64[6] = reshape %1 %2
                            in (%3)
                        },
                    ]
                in (%3, %4)
            "}
            .trim_end(),
        );
        assert_eq!(
            jvp.interpret(vec![values.clone(), extent.clone(), tangents.clone()]),
            Ok(vec![values.clone(), tangents.clone()]),
        );

        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[6], %1:dimension<rows ∈ [1, 9)> .
                let %2:f64[rows] = reshape %0 %1
                in (%2, %1)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[6], %1:dimension<rows ∈ [1, 9)> .
                let %2:f64[rows] = linear_call [residual_count=1] %1 %0 [
                    forward={
                        lambda %0:dimension<rows ∈ [1, 9)>, %1:f64[6] .
                        let %2:f64[rows] = reshape %1 %0
                        in (%2)
                    },
                    transpose={
                        lambda %0:dimension<rows ∈ [1, 9)>, %1:f64[rows] .
                        let %2:dimension<6> = constant [value=6]
                            %3:f64[6] = reshape %1 %2
                        in (%3)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
        let mut primal_outputs = linearization.primal().interpret(vec![values.clone(), extent]).unwrap();
        let residuals = primal_outputs.split_off(1);
        assert_eq!(primal_outputs, vec![values.clone()]);
        assert_eq!(residuals.len(), linearization.residual_count());
        let mut tangent_inputs = vec![tangents.clone()];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(linearization.tangent().interpret(tangent_inputs), Ok(vec![tangents.clone()]));

        let pullback = linearization.tangent().transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(pullback.output_types(), vec![ArrayIrType::Array(ArrayType::new_static(DataType::F64, [6]))]);
        let mut pullback_inputs = vec![tangents.clone()];
        pullback_inputs.extend(residuals.clone());
        assert_eq!(pullback.interpret(pullback_inputs.clone()), Ok(vec![tangents.clone()]));
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![tangents]));

        // An input permutation is applied before the forward reshape and undone after the inverse reshape, so the
        // cotangent returns in the original input order.
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F64, [2, 3]).into());
        let extent = builder.add_input(DimensionType::new(rows).into());
        let input = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Transpose(TransposeOperation::new([1, 0]))),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![input, extent], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[2, 3], %1:dimension<rows ∈ [1, 9)> .
                let %2:f64[3, 2] = transpose [permutation=[1, 0]] %0
                    %3:f64[rows] = linear_call [residual_count=1] %1 %2 [
                        forward={
                            lambda %0:dimension<rows ∈ [1, 9)>, %1:f64[3, 2] .
                            let %2:f64[rows] = reshape %1 %0
                            in (%2)
                        },
                        transpose={
                            lambda %0:dimension<rows ∈ [1, 9)>, %1:f64[rows] .
                            let %2:dimension<3> = constant [value=3]
                                %3:dimension<2> = constant [value=2]
                                %4:f64[3, 2] = reshape %1 %2 %3
                            in (%4)
                        },
                    ]
                in (%3)
            "}
            .trim_end(),
        );
        let values = ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        let permuted_values = ArrayIrValue::Array(Array::vector(vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap());
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![values, ArrayIrValue::Dimension(DimensionValue::constant(6).unwrap())])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        assert_eq!(primal_outputs, vec![permuted_values]);
        let tangents = ArrayIrValue::Array(Array::matrix(2, 3, vec![10.0_f64, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap());
        let permuted_tangents =
            ArrayIrValue::Array(Array::vector(vec![10.0_f64, 40.0, 20.0, 50.0, 30.0, 60.0]).unwrap());
        let mut tangent_inputs = vec![tangents.clone()];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(linearization.tangent().interpret(tangent_inputs), Ok(vec![permuted_tangents.clone()]));
        let mut pullback_inputs = vec![permuted_tangents];
        pullback_inputs.extend(residuals);
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![tangents]));
    }

    #[test]
    fn test_dynamic_reshape_differentiation_dynamic_input_static_output() {
        // A dynamic input reshaped to a static output retains the input extent as a residual so that the pullback can
        // rebuild the exact dynamic input shape, while the static output extent stays a constant on both sides.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Static(4)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.clone().into());
        let twelve = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(12).unwrap()));
        let output = builder
            .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![input, twelve], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(program.output_types(), vec![ArrayIrType::Array(ArrayType::new_static(DataType::F64, [12]))]);

        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[rows, 4], %1:f64[rows, 4] .
                let %2:dimension<12> = const 12
                    %3:f64[12] = reshape %0 %2
                    %4:dimension<rows ∈ [1, 9)> = dimension_size [axis=0] %0
                    %5:f64[12] = linear_call [residual_count=2] %2 %4 %1 [
                        forward={
                            lambda %0:dimension<12>, %1:dimension<rows ∈ [1, 9)>, %2:f64[rows, 4] .
                            let %3:f64[12] = reshape %2 %0
                            in (%3)
                        },
                        transpose={
                            lambda %0:dimension<12>, %1:dimension<rows ∈ [1, 9)>, %2:f64[12] .
                            let %3:dimension<4> = constant [value=4]
                                %4:f64[rows, 4] = reshape %2 %1 %3
                            in (%4)
                        },
                    ]
                in (%3, %5)
            "}
            .trim_end(),
        );
        let values = (0..12).map(|value| value as f64).collect::<Vec<_>>();
        let tangent_values = (12..24).map(|value| value as f64).collect::<Vec<_>>();
        assert_eq!(
            jvp.interpret(vec![
                ArrayIrValue::Array(Array::matrix(3, 4, values.clone()).unwrap()),
                ArrayIrValue::Array(Array::matrix(3, 4, tangent_values.clone()).unwrap()),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(values.clone()).unwrap()),
                ArrayIrValue::Array(Array::vector(tangent_values.clone()).unwrap()),
            ]),
        );

        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[rows, 4] .
                let %1:dimension<12> = const 12
                    %2:f64[12] = reshape %0 %1
                    %3:dimension<rows ∈ [1, 9)> = dimension_size [axis=0] %0
                in (%2, %3)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[rows, 4], %1:dimension<rows ∈ [1, 9)> .
                let %2:dimension<12> = const 12
                    %3:f64[12] = linear_call [residual_count=2] %2 %1 %0 [
                        forward={
                            lambda %0:dimension<12>, %1:dimension<rows ∈ [1, 9)>, %2:f64[rows, 4] .
                            let %3:f64[12] = reshape %2 %0
                            in (%3)
                        },
                        transpose={
                            lambda %0:dimension<12>, %1:dimension<rows ∈ [1, 9)>, %2:f64[12] .
                            let %3:dimension<4> = constant [value=4]
                                %4:f64[rows, 4] = reshape %2 %1 %3
                            in (%4)
                        },
                    ]
                in (%3)
            "}
            .trim_end(),
        );
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::matrix(3, 4, values.clone()).unwrap())])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        assert_eq!(primal_outputs, vec![ArrayIrValue::Array(Array::vector(values).unwrap())]);
        assert_eq!(residuals.len(), linearization.residual_count());
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::matrix(3, 4, tangent_values.clone()).unwrap())];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(tangent_values.clone()).unwrap())]),
        );

        let pullback = linearization.tangent().transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(pullback.output_types(), vec![input_type.cotangent().unwrap().into()]);
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(tangent_values.clone()).unwrap())];
        pullback_inputs.extend(residuals);
        let expected_cotangent = ArrayIrValue::Array(Array::matrix(3, 4, tangent_values).unwrap());
        assert_eq!(pullback.interpret(pullback_inputs.clone()), Ok(vec![expected_cotangent.clone()]));
        assert_eq!(linearization.pullback().unwrap().interpret(pullback_inputs), Ok(vec![expected_cotangent]));
    }

    #[test]
    fn test_dynamic_reshape_differentiation_retains_input_extents() {
        // The inverse cannot recover `n` from the `[2, 2*n]` output shape without division. The reshape JVP must
        // therefore retain the original source extent as an explicit residual while it still has the source array.
        let program = doubled_extent_reshape_program();
        let jvp = program.jvp().unwrap();

        // The dual program derives the forward geometry once. The staged linear call receives the primal reshape's own
        // dimension inputs as its leading residuals instead of restaging extent arithmetic for the tangent, so no
        // dimension acquires a second forward definition just because the program was differentiated. The additional
        // geometry read is the transpose residual that the inverse reshape needs, not a duplicated derivation.
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[source, 4], %1:f64[source, 4] .
                let %2:dimension<2> = const 2
                    %3:dimension<source ∈ [0, 9)> = dimension_size [axis=0] %0
                    %4:dimension<source * 2 ∈ [0, 17)> = dimension_mul %3 %2
                    %5:f64[2, source * 2] = reshape %0 %2 %4
                    %6:dimension<source ∈ [0, 9)> = dimension_size [axis=0] %0
                    %7:f64[2, source * 2] = linear_call [residual_count=3] %2 %4 %6 %1 [
                        forward={
                            lambda %0:dimension<2>, %1:dimension<source * 2 ∈ [0, 17)>, %2:dimension<source ∈ [0, 9)>, \
                %3:f64[source, 4] .
                            let %4:f64[2, source * 2] = reshape %3 %0 %1
                            in (%4)
                        },
                        transpose={
                            lambda %0:dimension<2>, %1:dimension<source * 2 ∈ [0, 17)>, %2:dimension<source ∈ [0, 9)>, \
                %3:f64[2, source * 2] .
                            let %4:dimension<4> = constant [value=4]
                                %5:f64[source, 4] = reshape %3 %2 %4
                            in (%5)
                        },
                    ]
                in (%5, %7)
            "}
            .trim_end(),
        );

        for size in [0, 1, 3, 8] {
            let element_count = size * 4;
            let primal_values = (0..element_count).map(|value| value as f64).collect::<Vec<_>>();
            let tangent_values = (element_count..2 * element_count).map(|value| value as f64).collect::<Vec<_>>();
            assert_eq!(
                jvp.interpret(vec![
                    ArrayIrValue::Array(Array::matrix(size, 4, primal_values.clone()).unwrap()),
                    ArrayIrValue::Array(Array::matrix(size, 4, tangent_values.clone()).unwrap()),
                ]),
                Ok(vec![
                    ArrayIrValue::Array(Array::matrix(2, 2 * size, primal_values).unwrap()),
                    ArrayIrValue::Array(Array::matrix(2, 2 * size, tangent_values).unwrap()),
                ]),
            );
        }
        let linearization = program.linearize().unwrap();
        let rendered_primal = linearization.primal().to_string();
        let rendered_tangent = linearization.tangent().to_string();
        assert_eq!(
            rendered_primal,
            indoc! {"
                lambda %0:f64[source, 4] .
                let %1:dimension<2> = const 2
                    %2:dimension<source ∈ [0, 9)> = dimension_size [axis=0] %0
                    %3:dimension<source * 2 ∈ [0, 17)> = dimension_mul %2 %1
                    %4:f64[2, source * 2] = reshape %0 %1 %3
                    %5:dimension<source ∈ [0, 9)> = dimension_size [axis=0] %0
                in (%4, %3, %5)
            "}
            .trim_end(),
        );
        assert_eq!(
            rendered_tangent,
            indoc! {"
                lambda %0:f64[source, 4], %1:dimension<source * 2 ∈ [0, 17)>, %2:dimension<source ∈ [0, 9)> .
                let %3:dimension<2> = const 2
                    %4:f64[2, source * 2] = linear_call [residual_count=3] %3 %1 %2 %0 [
                        forward={
                            lambda %0:dimension<2>, %1:dimension<source * 2 ∈ [0, 17)>, %2:dimension<source ∈ [0, 9)>, \
                %3:f64[source, 4] .
                            let %4:f64[2, source * 2] = reshape %3 %0 %1
                            in (%4)
                        },
                        transpose={
                            lambda %0:dimension<2>, %1:dimension<source * 2 ∈ [0, 17)>, \
                %2:dimension<source ∈ [0, 9)>, %3:f64[2, source * 2] .
                            let %4:dimension<4> = constant [value=4]
                                %5:f64[source, 4] = reshape %3 %2 %4
                            in (%5)
                        },
                    ]
                in (%4)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().input_types().iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec!["f64[source, 4]", "dimension<source * 2 ∈ [0, 17)>", "dimension<source ∈ [0, 9)>"],
        );
        assert_eq!(linearization.tangent().input_types()[0], program.input_types()[0].tangent().unwrap(),);
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(
                Array::matrix(3, 4, (0..12).map(|value| value as f64).collect()).unwrap(),
            )])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        assert_eq!(residuals.len(), linearization.residual_count());
        assert_eq!(residuals.len(), 2);

        let tangent_values = (12..24).map(|value| value as f64).collect::<Vec<_>>();
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::matrix(3, 4, tangent_values.clone()).unwrap())];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs.clone()),
            Ok(vec![ArrayIrValue::Array(Array::matrix(2, 6, tangent_values).unwrap())]),
        );

        let mut pullback_inputs =
            vec![ArrayIrValue::Array(Array::matrix(2, 6, (24..36).map(|value| value as f64).collect()).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::matrix(3, 4, (24..36).map(|value| value as f64).collect(),).unwrap())]),
        );
    }

    #[test]
    fn test_dynamic_reshape_differentiation_retains_input_extents_replays_and_specializes() {
        // Replaying without concrete refinement still creates a fresh arithmetic definition. Its new identity must
        // be used consistently by the output shape rather than compared with the source definition by name.
        let program = doubled_extent_reshape_program();
        let (replayed_types, replayed) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |inputs| {
                let context = inputs[0].context().clone();
                program.interpret_in_context(&context, inputs)
            },
            program.input_types(),
        )
        .unwrap();
        assert_ne!(replayed_types, program.output_types());
        assert_eq!(replayed_types[0].to_string(), "f64[2, source * 2]");

        // Specialization replays this retained graph rather than retracing its construction closure. Both the
        // dimension arithmetic and the reshape must acquire the concrete geometry at each independent call.
        for size in [4, 5] {
            let input_type = ArrayType::new_static(DataType::F64, [size, 4]);
            let specialized = program.clone().specialize(&[input_type.clone().into()]).unwrap();
            let replayed_specialized = replayed.clone().specialize(&[input_type.clone().into()]).unwrap();
            assert_eq!(replayed_specialized.output_types(), specialized.output_types());
            assert_eq!(specialized.input_types(), vec![input_type.into()]);
            assert_eq!(specialized.output_types(), vec![ArrayType::new_static(DataType::F64, [2, 2 * size]).into()]);
            let values = (0..size * 4).map(|value| value as f64).collect::<Vec<_>>();
            assert_eq!(
                specialized.interpret(vec![ArrayIrValue::Array(Array::matrix(size, 4, values.clone()).unwrap())]),
                Ok(vec![ArrayIrValue::Array(Array::matrix(2, 2 * size, values).unwrap())]),
            );
        }
        assert!(matches!(
            program.clone().specialize(&[ArrayType::new_static(DataType::F64, [9, 4]).into()]),
            Err(ProgramError::Type(TypeError::Invalid { message, .. }))
                if message == "specialized input type f64[9, 4] does not refine declared input type f64[source, 4]",
        ));
    }

    #[test]
    fn test_dynamic_reshape_differentiation_retains_input_extents_imports_and_nests_the_linear_boundary() {
        // The executable linear boundary remains structural when imported, including both attached regions and every
        // residual edge. Nested forward differentiation likewise treats only the array input as differentiable.
        let program = doubled_extent_reshape_program();
        let linearization = program.linearize().unwrap();
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(
                Array::matrix(3, 4, (0..12).map(|value| value as f64).collect()).unwrap(),
            )])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs =
            vec![ArrayIrValue::Array(Array::matrix(3, 4, (12..24).map(|value| value as f64).collect()).unwrap())];
        tangent_inputs.extend(residuals.clone());
        let mut imported_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let imported_inputs = linearization
            .tangent()
            .input_types()
            .into_iter()
            .map(|r#type| imported_builder.add_input(r#type))
            .collect::<Vec<_>>();
        let imported_outputs =
            imported_builder.splice_program(linearization.tangent(), imported_inputs.as_slice()).unwrap();
        let imported = imported_builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                imported_outputs,
                vec![Placeholder; imported_inputs.len()],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(imported.to_string(), linearization.tangent().to_string());
        assert_eq!(imported.interpret(tangent_inputs.clone()), linearization.tangent().interpret(tangent_inputs));

        let nested_jvp = linearization.tangent().jvp().unwrap();
        let mut nested_inputs =
            vec![ArrayIrValue::Array(Array::matrix(3, 4, (12..24).map(|value| value as f64).collect()).unwrap())];
        nested_inputs.extend(residuals.clone());
        nested_inputs
            .push(ArrayIrValue::Array(Array::matrix(3, 4, (24..36).map(|value| value as f64).collect()).unwrap()));
        assert_eq!(nested_jvp.input_ids().len(), 2 + residuals.len());
        assert_eq!(
            nested_jvp.interpret(nested_inputs),
            Ok(vec![
                ArrayIrValue::Array(Array::matrix(2, 6, (12..24).map(|value| value as f64).collect(),).unwrap()),
                ArrayIrValue::Array(Array::matrix(2, 6, (24..36).map(|value| value as f64).collect(),).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_dynamic_reshape_differentiation_retains_input_extents_reuses_matching_extent_inputs() {
        // A matching explicit output-extent input is already the authoritative SSA value for the source axis, so
        // the residual path reuses it and does not read the source array again.
        let source = DimensionVariable::new("reused_source", DimensionBounds::new(1, Some(9)).unwrap());
        let source_type = DimensionType::new(source.clone());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(source), Dimension::Static(4)])).into(),
        );
        let source_extent = builder.add_input(source_type.into());
        let four = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap()));
        let output = builder
            .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![input, source_extent, four], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[reused_source, 4], %1:dimension<reused_source ∈ [1, 9)> .
                let %2:dimension<4> = const 4
                    %3:f64[reused_source, 4] = reshape [requires_runtime_assertion=false] %0 %1 %2
                in (%3, %1)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[reused_source, 4], %1:dimension<reused_source ∈ [1, 9)> .
                let %2:dimension<4> = const 4
                    %3:f64[reused_source, 4] = linear_call [residual_count=2] %1 %2 %0 [
                        forward={
                            lambda %0:dimension<reused_source ∈ [1, 9)>, %1:dimension<4>, %2:f64[reused_source, 4] .
                            let %3:f64[reused_source, 4] = reshape [requires_runtime_assertion=false] %2 %0 %1
                            in (%3)
                        },
                        transpose={
                            lambda %0:dimension<reused_source ∈ [1, 9)>, %1:dimension<4>, %2:f64[reused_source, 4] .
                            let %3:dimension<4> = constant [value=4]
                                %4:f64[reused_source, 4] = reshape [requires_runtime_assertion=false] %2 %0 %3
                            in (%4)
                        },
                    ]
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dynamic_reshape_differentiation_over_a_data_derived_extent() {
        // The reshape's output extent is produced by the `dimension_from_scalar` gateway over an ordinary integer
        // scalar array input, so it is a tier-3 data-derived dimension rather than a shape read off an input. The
        // linear-call residual contract must carry it exactly like any other primal dimension the tangent needs.
        let source = DimensionVariable::new("source", DimensionBounds::new(1, Some(9)).unwrap());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(source), Dimension::Static(4)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let extent_scalar = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let total = DimensionVariable::new("total", DimensionBounds::new(1, Some(33)).unwrap());
        let extent = builder
            .add_instruction(DimensionFromScalarOperation::new(total), Vec::new(), vec![extent_scalar], None)
            .unwrap()[0];
        let output = builder
            .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![input, extent], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[source, 4], %1:i32[] .
                let %2:dimension<total ∈ [1, 33)> = dimension_from_scalar [bounds=[1, 33)] %1
                    %3:f64[total] = reshape %0 %2
                    %4:dimension<source ∈ [1, 9)> = dimension_size [axis=0] %0
                in (%3, %2, %4)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[source, 4], %1:dimension<total ∈ [1, 33)>, %2:dimension<source ∈ [1, 9)> .
                let %3:f64[total] = linear_call [residual_count=2] %1 %2 %0 [
                    forward={
                        lambda %0:dimension<total ∈ [1, 33)>, %1:dimension<source ∈ [1, 9)>, %2:f64[source, 4] .
                        let %3:f64[total] = reshape %2 %0
                        in (%3)
                    },
                    transpose={
                        lambda %0:dimension<total ∈ [1, 33)>, %1:dimension<source ∈ [1, 9)>, %2:f64[total] .
                        let %3:dimension<4> = constant [value=4]
                            %4:f64[source, 4] = reshape %2 %1 %3
                        in (%4)
                    },
                ]
                in (%3)"},
        );

        // The data-derived extent rides the ordinary residual path: it is a primal output, a tangent input, and a
        // leading residual input of the linear call. Nothing about it is special-cased relative to the geometry-read
        // `source` residual beside it, and no dimension acquires a tangent input of its own.
        assert_eq!(linearization.residual_count(), 2);
        assert_eq!(
            linearization.tangent().input_types().iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec!["f64[source, 4]", "dimension<total ∈ [1, 33)>", "dimension<source ∈ [1, 9)>"],
        );

        // The complete contract executes: the primal emits the checked extent as a residual and the tangent consumes
        // it to reshape the live tangent array.
        let primal_values = (0..12).map(|value| value as f64).collect::<Vec<_>>();
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                ArrayIrValue::Array(Array::matrix(3, 4, primal_values.clone()).unwrap()),
                ArrayIrValue::Array(Array::scalar(12_i32).unwrap()),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        assert_eq!(primal_outputs, vec![ArrayIrValue::Array(Array::vector(primal_values).unwrap())]);
        assert_eq!(residuals.len(), 2);

        let tangent_values = (12..24).map(|value| value as f64).collect::<Vec<_>>();
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::matrix(3, 4, tangent_values.clone()).unwrap())];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::vector(tangent_values.clone()).unwrap())]),
        );

        let cotangent_values = (24..36).map(|value| value as f64).collect::<Vec<_>>();
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::vector(cotangent_values.clone()).unwrap())];
        pullback_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::matrix(3, 4, cotangent_values).unwrap())]),
        );

        // Nested forward differentiation of the linear program treats only the array input as differentiable, so the
        // two residual dimensions pass through the second-order boundary unchanged rather than acquiring tangents.
        let nested_jvp = linearization.tangent().jvp().unwrap();
        assert_eq!(nested_jvp.input_ids().len(), 2 + residuals.len());
        let mut nested_inputs = vec![ArrayIrValue::Array(Array::matrix(3, 4, tangent_values.clone()).unwrap())];
        nested_inputs.extend(residuals);
        nested_inputs
            .push(ArrayIrValue::Array(Array::matrix(3, 4, (36..48).map(|value| value as f64).collect()).unwrap()));
        assert_eq!(
            nested_jvp.interpret(nested_inputs),
            Ok(vec![
                ArrayIrValue::Array(Array::vector(tangent_values).unwrap()),
                ArrayIrValue::Array(Array::vector((36..48).map(|value| value as f64).collect()).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_dynamic_reshape_differentiation_deduplicates_repeated_permuted_extents() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(0, Some(5)).unwrap());
        let input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(extent.clone()), Dimension::Dynamic(extent)]),
        );
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.clone().into());
        let extent = builder
            .add_instruction(DimensionSizeOperation::new(&input_type, 0).unwrap(), Vec::new(), vec![input], None)
            .unwrap()[0];
        let input = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Transpose(TransposeOperation::new([1, 0]))),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![input, extent, extent], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();

        // Both output axes and both inverse axes use the same SSA extent. Partial evaluation carries it once even
        // though the linear call consumes it in multiple input positions.
        assert_eq!(linearization.residual_count(), 1);
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[extent, extent] .
                let %1:f64[extent, extent] = transpose [permutation=[1, 0]] %0
                    %2:dimension<extent ∈ [0, 5)> = dimension_size [axis=0] %0
                    %3:f64[extent, extent] = reshape [requires_runtime_assertion=false] %1 %2 %2
                in (%3, %2)
            "}
            .trim_end(),
        );
        let input = ArrayIrValue::Array(Array::matrix(3, 3, (0..9).map(|value| value as f64).collect()).unwrap());
        let mut primal_outputs = linearization.primal().interpret(vec![input]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let tangent = ArrayIrValue::Array(Array::matrix(3, 3, (9..18).map(|value| value as f64).collect()).unwrap());
        let mut tangent_inputs = vec![tangent];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(
                Array::matrix(3, 3, vec![9.0, 12.0, 15.0, 10.0, 13.0, 16.0, 11.0, 14.0, 17.0],).unwrap()
            )]),
        );
        let mut pullback_inputs =
            vec![ArrayIrValue::Array(Array::matrix(3, 3, (18..27).map(|value| value as f64).collect()).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(
                Array::matrix(3, 3, vec![18.0, 21.0, 24.0, 19.0, 22.0, 25.0, 20.0, 23.0, 26.0],).unwrap()
            )]),
        );

        // The same compiled programs accept the lower-bound zero without inventing an extent tangent input.
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![ArrayIrValue::Array(Array::matrix(0, 0, Vec::<f64>::new()).unwrap())])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![ArrayIrValue::Array(Array::matrix(0, 0, Vec::<f64>::new()).unwrap())];
        tangent_inputs.extend(residuals);
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![ArrayIrValue::Array(Array::matrix(0, 0, Vec::<f64>::new()).unwrap())]),
        );
    }

    #[test]
    fn test_dynamic_reshape_differentiation_preserves_sharding_through_the_inverse() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 2);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent.clone()), Dimension::Static(4)]))
                .with_sharding(sharding.clone())
                .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.clone().into());
        let extent = builder.add_input(DimensionType::new(extent).into());
        let four = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap()));
        let output = builder
            .add_instruction(
                DynamicReshapeOperation::new().with_output_sharding(sharding),
                Vec::new(),
                vec![input, extent, four],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();

        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.tangent().output_types(), vec![input_type.tangent().unwrap().into()]);
        assert_eq!(linearization.pullback().unwrap().output_types(), vec![input_type.cotangent().unwrap().into()]);

        // The inverse restores metadata that the forward reshape removed or introduced. In particular, replicated
        // bridge sharding must not escape into the cotangent of an originally unsharded input.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let plain_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent.clone()), Dimension::Static(4)]));
        for input_type in [plain_type.clone(), plain_type.with_layout(Layout::Strided(StridedLayout::new(vec![40, 8])))]
        {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let input = builder.add_input(input_type.clone().into());
            let extent = builder.add_input(DimensionType::new(extent.clone()).into());
            let two = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()));
            let output = builder
                .add_instruction(
                    DynamicReshapeOperation::new().with_output_sharding(Sharding::replicated(mesh, 3)),
                    Vec::new(),
                    vec![input, extent, two, two],
                    None,
                )
                .unwrap()[0];
            let program = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![output],
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap();
            let linearization = program.linearize().unwrap();
            assert_eq!(linearization.pullback().unwrap().output_types(), vec![input_type.cotangent().unwrap().into()]);
        }
    }

    #[test]
    fn test_dynamic_reshape_transposition() {
        // Static geometry delegates to the homogeneous pullback, which reshapes the cotangent back, while the known
        // extent inputs receive no cotangent.
        let two = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let three = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        check_operation_transposition!(
            @exact,
            backend = (ArrayIrValue<Array>, ArrayIrOperation<Array>),
            operation = DynamicReshapeOperation::new(),
            cases = [{
                inputs = [
                    (@linear(type = ArrayIrType::Array(ArrayType::new(
                        DataType::F64,
                        Shape::new(vec![Dimension::Static(6)]),
                    )))),
                    (@known, two.clone()),
                    (@known, three.clone()),
                ],
                output_cotangents = [ArrayIrValue::Array(
                    Array::matrix(2, 3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
                )],
                input_cotangents = [ArrayIrValue::Array(
                    Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
                )],
                pullback = indoc! {"
                    lambda %0:f64[2, 3], %1:dimension<2>, %2:dimension<3> .
                    let %3:f64[6] = reshape [shape=[6]] %0
                    in (%3)
                "},
            }],
        );
        // Structural zeros need no inverse shape reconstruction, even with unresolved input extents.
        let extent = DimensionVariable::new("rows", DimensionBounds::new(0, Some(5)).unwrap());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent.clone()), 4.into()]));
        let extent_type = ArrayIrType::Dimension(DimensionType::new(extent.clone()));
        let output_type = input_type.clone();
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let four = context.lift(ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap())).unwrap();
        let mut rule_context = TranspositionContext::new(context);
        let inputs = [
            PartialValue::Unknown(ArrayIrType::Array(input_type.clone())),
            PartialValue::Unknown(extent_type),
            PartialValue::Known(four),
        ];
        let accumulators = rule_context.cotangent_accumulators(&inputs, &[]).unwrap();
        DynamicReshapeOperation::new()
            .transpose(
                &mut rule_context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Zero(output_type.cotangent().unwrap().into())],
                &accumulators,
            )
            .unwrap();
        let contributions = rule_context.take_cotangents(&accumulators).unwrap();
        assert_eq!(contributions.len(), 3);
        assert!(contributions[0].is_zero());
        assert_eq!(contributions[0].r#type().as_ref(), &ArrayIrType::Array(input_type.cotangent().unwrap()));
        assert!(contributions[1].is_zero());
        assert_eq!(contributions[1].r#type().as_ref(), &inputs[1].r#type().cotangent().unwrap());
        assert!(contributions[2].is_zero());
        assert_eq!(contributions[2].r#type().as_ref(), &inputs[2].r#type().cotangent().unwrap());
    }

    #[test]
    fn test_dynamic_reshape_transposition_rejects_a_dynamic_input_extent() {
        // The direct rule reshapes the cotangent back through static geometry only, which cannot recover a runtime
        // input extent. It therefore rejects dynamic input geometry by name; linearization is the supported route,
        // because it retains that extent as an explicit residual.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let input_type =
            ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(rows), 4.into()])));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.clone());
        let twelve = builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(12).unwrap()));
        let output = builder
            .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![input, twelve], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert!(matches!(
            program.transpose_with_respect_to(&[0], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == format!(
                    "direct transposition of a dynamic `{RESHAPE_OPERATION_NAME}` requires linearization so its \
                     input and output extents are available as explicit residuals",
                ),
        ));
        assert_eq!(
            program.linearize().unwrap().pullback().unwrap().output_types(),
            vec![input_type.cotangent().unwrap()],
        );
    }

    #[test]
    fn test_dynamic_reshape_transposition_rejects_a_dynamic_output_extent() {
        // A dynamic output extent is rejected for the same reason: the inverse reshape of the homogeneous pullback
        // needs static geometry on both sides, and linearization retains the output extent as a residual instead.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let input_type = ArrayIrType::Array(ArrayType::new_static(DataType::F64, [6]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.clone());
        let extent = builder.add_input(DimensionType::new(rows).into());
        let output = builder
            .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![input, extent], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert!(matches!(
            program.transpose_with_respect_to(&[0], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == format!(
                    "direct transposition of a dynamic `{RESHAPE_OPERATION_NAME}` requires linearization so its \
                     input and output extents are available as explicit residuals",
                ),
        ));
        assert_eq!(
            program.linearize().unwrap().pullback().unwrap().output_types(),
            vec![input_type.cotangent().unwrap()],
        );
    }

    #[test]
    fn test_dynamic_reshape_dynamic_reshape() {
        // A staged reshape whose output shape is runtime-derived keeps each extent an ordinary input: the leading
        // extent is read off the input and the trailing one is first-class dimension arithmetic.
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(2), Dimension::Static(3)]),
        );
        let (output_type, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |input| {
                // Dimension arithmetic is a composite capability, so the two static extents multiply directly.
                let rows = input.dimension_size(0)?;
                let columns = input.dimension_size(1)?.dimension_mul(&input.dimension_size(2)?)?;
                input.dynamic_reshape(&[rows, columns])
            },
            ArrayIrType::Array(input_type),
        )
        .unwrap();
        assert_eq!(output_type.to_string(), "f64[batch, 6]");
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[batch, 2, 3] .
                let %1:dimension<batch ∈ [1, 9)> = dimension_size [axis=0] %0
                    %2:dimension<2> = constant [value=2]
                    %3:dimension<3> = constant [value=3]
                    %4:dimension<6> = dimension_mul %2 %3
                    %5:f64[batch, 6] = reshape [requires_runtime_assertion=false] %0 %1 %4
                in (%5)
            "}
            .trim_end(),
        );

        // A static identity reshape is elided, so the only staged instruction is the dimension literal it received and
        // never observed. Simplification drops that literal along with any other unused instruction.
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |input| input.dynamic_reshape_to_sizes(&[6]),
            ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(6)]))),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[6] .
                let %1:dimension<6> = constant [value=6]
                in (%0)
            "}
            .trim_end(),
        );
        assert!(program.into_simplified().unwrap().instructions().is_empty());
    }

    #[test]
    fn test_dynamic_reshape_dynamic_reshape_with_output_sharding() {
        // Requested output sharding is carried on the result, both eagerly and when staged.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input = ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        let six = ArrayIrValue::Dimension(DimensionValue::constant(6).unwrap());
        let output_type = ArrayType::new_static(DataType::F64, [6]).with_sharding(sharding.clone()).unwrap();
        assert_eq!(
            input.dynamic_reshape_with_output_sharding(std::slice::from_ref(&six), Some(sharding.clone()),),
            Ok(ArrayIrValue::Array(
                Array::from_elements::<f64>(output_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
            )),
        );

        let (staged_output_type, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |input| {
                let six = input.dispatch_domain().dimension_constant(6)?;
                input.dynamic_reshape_with_output_sharding(&[six], Some(sharding.clone()))
            },
            ArrayIrType::Array(ArrayType::new_static(DataType::F64, [2, 3])),
        )
        .unwrap();
        assert_eq!(staged_output_type, ArrayIrType::Array(output_type));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 3] .
                let %1:dimension<6> = constant [value=6]
                    %2:f64[6][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = reshape [
                        requires_runtime_assertion=false,
                        output_sharding={mesh<['x'=2:explicit]>, [{'x'}]},
                    ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dynamic_reshape_dynamic_reshape_invalid_output_count() {
        // Exercise both missing and extra results without involving larger operation families.
        let array = Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        for output_count in [0, 2] {
            let context =
                InvalidOutputContext::<ArrayIrValue<Array>, DynamicReshapeOperation>(output_count, PhantomData);
            let input = context.lift(ArrayIrValue::Array(array.clone())).unwrap();
            let two = context.lift(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap())).unwrap();
            let three = context.lift(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap())).unwrap();
            assert!(matches!(
                input.dynamic_reshape(&[two, three]),
                Err(ProgramError::InvalidOutputCount { expected: 1, actual }) if actual == output_count,
            ));
        }
    }

    #[test]
    fn test_dynamic_reshape_dynamic_reshape_to_sizes() {
        let input = ArrayIrValue::Array(Array::vector(vec![1_i32, 2, 3, 4, 5, 6]).unwrap());
        assert_eq!(
            input.dynamic_reshape_to_sizes(&[3, 2]),
            Ok(ArrayIrValue::Array(Array::matrix(3, 2, vec![1_i32, 2, 3, 4, 5, 6]).unwrap())),
        );
        assert_eq!(input.dynamic_reshape_to_sizes(&[6]), Ok(input.clone()));
        assert!(matches!(
            input.dynamic_reshape_to_sizes(&[5]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == format!("`{RESHAPE_OPERATION_NAME}` changes the number of elements"),
        ));

        // The geometry is validated before any dimension constant is staged, so an invalid element count leaves no
        // dead dimension literals behind in a trace.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = trace.input(ArrayType::new_static(DataType::I32, [6]).into());
        assert!(matches!(
            input.dynamic_reshape_to_sizes(&[5]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == format!("`{RESHAPE_OPERATION_NAME}` changes the number of elements"),
        ));
        assert!(trace.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_dynamic_reshape_dynamic_flatten() {
        let input = ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::F64, [2, 3]), &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap(),
        );
        assert_eq!(
            input.dynamic_flatten().unwrap(),
            ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::F64, [6]), &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],)
                    .unwrap()
            )
        );
        let empty = ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::F64, [i64::MAX as usize, 3, 0]), &[] as &[f64])
                .unwrap(),
        );
        assert_eq!(
            empty.dynamic_flatten().unwrap(),
            ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::F64, [0]), &[] as &[f64],).unwrap()
            )
        );

        // A static input needs no runtime size arithmetic: it stages exactly what `dynamic_reshape_to_sizes` stages.
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |input| input.dynamic_flatten(),
            ArrayIrType::Array(ArrayType::new_static(DataType::F64, [2, 3])),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 3] .
                let %1:dimension<6> = constant [value=6]
                    %2:f64[6] = reshape [requires_runtime_assertion=false] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
        let (_, expected) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |input| input.dynamic_reshape_to_sizes(&[6]),
            ArrayIrType::Array(ArrayType::new_static(DataType::F64, [2, 3])),
        )
        .unwrap();
        assert_eq!(program.to_string(), expected.to_string());

        // A vector is returned as is, and a scalar becomes a vector of size one.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let (output_type, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |input| input.dynamic_flatten(),
            ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(rows.clone())]))),
        )
        .unwrap();
        assert_eq!(
            output_type,
            ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(rows.clone())]))),
        );
        assert!(program.instructions().is_empty());
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |input| input.dynamic_flatten(),
            ArrayIrType::Array(ArrayType::scalar(DataType::F64)),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:dimension<1> = constant [value=1]
                    %2:f64[1] = reshape [requires_runtime_assertion=false] %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // A dynamic input seeds the product with its leading extent instead of a constant one, so that no
        // multiplication by one is staged, and folds the remaining extents with checked dimension arithmetic.
        let (output_type, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |input| input.dynamic_flatten(),
            ArrayIrType::Array(ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Dynamic(rows), Dimension::Static(4)]),
            )),
        )
        .unwrap();
        assert_eq!(output_type.to_string(), "f64[rows * 4]");
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[rows, 4] .
                let %1:dimension<rows ∈ [1, 9)> = dimension_size [axis=0] %0
                    %2:dimension<4> = constant [value=4]
                    %3:dimension<rows * 4 ∈ [4, 33)> = dimension_mul %1 %2
                    %4:f64[rows * 4] = reshape %0 %3
                in (%4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dynamic_reshape_dynamic_expand_dimensions() {
        // On a partly dynamic input, the static extents are lifted as constants and only the dynamic axis is read
        // back from the array. Replaying the traced program at two sizes shows that the inserted axis is a genuine
        // size-one axis rather than a baked-in physical bound.
        let dimension = DimensionVariable::new("rows", DimensionBounds::new(4, Some(6)).unwrap());
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |input| input.dynamic_expand_dimensions(0),
            ArrayIrType::Array(ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Dynamic(dimension), Dimension::Static(4)]),
            )),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[rows, 4] .
                let %1:dimension<rows ∈ [4, 6)> = dimension_size [axis=0] %0
                    %2:dimension<4> = constant [value=4]
                    %3:dimension<1> = constant [value=1]
                    %4:f64[1, rows, 4] = reshape [requires_runtime_assertion=false] %0 %3 %1 %2
                in (%4)
            "}
            .trim_end(),
        );
        for size in [4, 5] {
            let input = Array::from_elements(
                ArrayType::new_static(DataType::F64, [size, 4]),
                &(0..size * 4).map(|value| value as f64).collect::<Vec<_>>(),
            )
            .unwrap();
            let output = program.interpret(ArrayIrValue::Array(input.clone())).unwrap();
            let output = <ArrayIrValue<Array> as ValueProjection<ArrayType>>::into_projected(output).unwrap();
            assert_eq!(output.to_f64s(), input.to_f64s());
            assert_eq!(output.r#type().static_shape().unwrap().dimensions(), vec![1, size, 4]);
        }
        let input = ArrayIrValue::Array(
            Array::from_elements(ArrayType::new_static(DataType::F64, [2, 3]), &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap(),
        );
        assert_eq!(input.dynamic_expand_dimensions(-1).unwrap().r#type().to_string(), "f64[2, 3, 1]");
        assert!(matches!(
            input.dynamic_expand_dimensions(4),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "axis 4 is out of bounds for rank 3",
        ));
    }

    #[test]
    fn test_array_ir_value_dynamic_reshape() {
        // A concrete composite value resolves every extent input to its runtime value, binding repeated nominal
        // identities to one extent, and reshapes its array member with the resolved parameters.
        let input = ArrayIrValue::Array(Array::matrix(2, 3, vec![1_i32, 2, 3, 4, 5, 6]).unwrap());
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let three = ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap());
        let two = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        assert_eq!(
            input.dynamic_reshape_with_output_sharding(&[three.clone(), two.clone()], None),
            Ok(ArrayIrValue::Array(Array::matrix(3, 2, vec![1_i32, 2, 3, 4, 5, 6]).unwrap())),
        );
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 2);
        assert_eq!(
            input.dynamic_reshape_with_output_sharding(&[three.clone(), two.clone()], Some(sharding.clone())),
            Ok(ArrayIrValue::Array(
                Array::from_elements::<i32>(
                    ArrayType::new_static(DataType::I32, [3, 2]).with_sharding(sharding).unwrap(),
                    &[1, 2, 3, 4, 5, 6],
                )
                .unwrap(),
            )),
        );

        // Repeated nominal dimensions must resolve to one size even when different sizes have a valid product.
        let dimensions = [
            ArrayIrValue::Dimension(DimensionValue::new(extent_type.clone(), 2).unwrap()),
            ArrayIrValue::Dimension(DimensionValue::new(extent_type, 3).unwrap()),
        ];
        assert_eq!(
            input.dynamic_reshape(&dimensions),
            Err(ProgramError::Type(
                DimensionError::InputDimensionMismatch { dimension: "extent".to_owned(), expected: 2, actual: 3 }
                    .into(),
            )),
        );

        // Both the reshaped value and every extent must be the corresponding composite member.
        assert_eq!(
            two.dynamic_reshape_with_output_sharding(&[three.clone(), two.clone()], None),
            Err(ProgramError::Type(TypeError::invalid("expected array type but got dimension type"))),
        );
        assert_eq!(
            input.dynamic_reshape_with_output_sharding(&[three, input.clone()], None),
            Err(ProgramError::Type(TypeError::invalid("expected dimension type but got array type"))),
        );
    }

    #[test]
    fn test_reshape_output_type() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let sharded_type = |dimensions: Vec<Dimension>, sharding: Vec<ShardingDimension>| {
            ArrayType::new(DataType::F32, Shape::new(dimensions))
                .with_sharding(Sharding::new(mesh.clone(), sharding).unwrap())
                .unwrap()
        };
        let infer = |input: &ArrayType, output: Vec<Dimension>| {
            infer_reshape_output_type(input, Shape::new(output), None).map(|output| output.sharding().cloned().unwrap())
        };
        let sharding = |dimensions: Vec<ShardingDimension>| Sharding::new(mesh.clone(), dimensions).unwrap();

        // Equal non-singleton dimensions, including dynamic ones, carry their placement regardless of singletons.
        let input = sharded_type(
            vec![Dimension::Dynamic(rows.clone()), 1.into(), 4.into()],
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated(), ShardingDimension::sharded(["y"])],
        );
        assert_eq!(
            infer(&input, vec![1.into(), Dimension::Dynamic(rows.clone()), 4.into(), 1.into()]),
            Ok(sharding(vec![
                ShardingDimension::replicated(),
                ShardingDimension::sharded(["x"]),
                ShardingDimension::sharded(["y"]),
                ShardingDimension::replicated(),
            ])),
        );

        // A fully replicated input stays replicated through any regrouping.
        let input = sharded_type(vec![2.into(), 6.into()], vec![ShardingDimension::replicated(); 2]);
        assert_eq!(infer(&input, vec![3.into(), 4.into()]), Ok(sharding(vec![ShardingDimension::replicated(); 2])));

        // Equal dynamic axes anchor the alignment of adjacent static split/merge groups.
        let input = sharded_type(
            vec![Dimension::Dynamic(rows.clone()), 2.into(), 4.into()],
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"]), ShardingDimension::replicated()],
        );
        assert_eq!(
            infer(&input, vec![Dimension::Dynamic(rows.clone()), 8.into()]),
            Ok(sharding(vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])])),
        );

        // Groups whose products never meet, and dynamic dimensions inside a group, are reported.
        let input = sharded_type(vec![4.into()], vec![ShardingDimension::sharded(["x"])]);
        assert_eq!(
            infer(&input, vec![3.into()]),
            Err(TypeError::invalid(format!("`{RESHAPE_OPERATION_NAME}` could not align reshape dimension groups"))),
        );
        let input = sharded_type(
            vec![Dimension::Dynamic(rows.clone()), 2.into()],
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        );
        assert_eq!(
            infer(&input, vec![2.into(), Dimension::Dynamic(rows)]),
            Err(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` requires explicit output sharding for unaligned dynamic dimensions"
            ))),
        );
    }

    #[test]
    fn test_reshape_output_type_requested_sharding() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("r", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let output_shape = Shape::new(vec![2.into(), 4.into()]);
        let requested =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();

        // Both unsharded and compatibly sharded inputs produce the requested output type and placement.
        let expected = ArrayType::new(DataType::F32, output_shape.clone()).with_sharding(requested.clone()).unwrap();
        let unsharded_input = ArrayType::new_static(DataType::F32, [8]);
        assert_eq!(
            infer_reshape_output_type(&unsharded_input, output_shape.clone(), Some(&requested)),
            Ok(expected.clone())
        );
        let input = ArrayType::new_static(DataType::F32, [8])
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()]).unwrap())
            .unwrap();
        assert_eq!(infer_reshape_output_type(&input, output_shape.clone(), Some(&requested)), Ok(expected));

        // Rank, mesh, and mesh-axis kind are checked before the reduction and manual-axis state.
        assert_eq!(
            infer_reshape_output_type(&input, Shape::new(vec![8.into()]), Some(&requested)),
            Err(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` requested output sharding rank (2) does not match the output rank (1)"
            ))),
        );
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other_requested =
            Sharding::new(other_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        assert_eq!(
            infer_reshape_output_type(&input, output_shape.clone(), Some(&other_requested)),
            Err(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` requested output sharding uses a different mesh"
            ))),
        );
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let auto_requested =
            Sharding::new(auto_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap();
        assert_eq!(
            infer_reshape_output_type(&unsharded_input, output_shape.clone(), Some(&auto_requested)),
            Err(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` requested output sharding cannot reference auto mesh axes"
            ))),
        );
        assert_eq!(
            infer_reshape_output_type(
                &input,
                output_shape.clone(),
                Some(&requested.clone().with_unreduced_axes(["r"]).unwrap()),
            ),
            Err(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` requested output sharding changes the unreduced mesh axes"
            ))),
        );
        assert_eq!(
            infer_reshape_output_type(
                &input,
                output_shape.clone(),
                Some(&requested.clone().with_reduced_axes(["r"]).unwrap()),
            ),
            Err(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` requested output sharding changes the reduced mesh axes"
            ))),
        );
        assert_eq!(
            infer_reshape_output_type(
                &input,
                output_shape.clone(),
                Some(&requested.with_varying_manual_axes(["m"]).unwrap()),
            ),
            Err(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` requested output sharding changes the varying manual mesh axes"
            ))),
        );
    }

    #[test]
    fn test_infer_reshape_output_type_zero_extents() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let infer = |input: Vec<Dimension>, output: Vec<Dimension>, sharding: &Sharding| {
            let input = ArrayType::new(DataType::F32, Shape::new(input)).with_sharding(sharding.clone()).unwrap();
            infer_reshape_output_type(&input, Shape::new(output), None)
                .map(|output| output.sharding().unwrap().dimensions().to_vec())
        };

        // Equal prefix and suffix dimensions carry their placement around the zero-product middle, which itself must
        // be replicated and static.
        let sharding = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated(), ShardingDimension::replicated()],
        )
        .unwrap();
        assert_eq!(
            infer(vec![3.into(), 0.into(), 2.into()], vec![3.into(), 0.into(), 5.into(), 2.into()], &sharding),
            Ok(vec![
                ShardingDimension::sharded(["x"]),
                ShardingDimension::replicated(),
                ShardingDimension::replicated(),
                ShardingDimension::replicated(),
            ]),
        );
        let suffix_sharding = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
        )
        .unwrap();
        assert_eq!(
            infer(vec![0.into(), 4.into(), 2.into()], vec![0.into(), 2.into()], &suffix_sharding),
            Ok(vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])]),
        );

        // A sharded dimension inside the zero-product middle is ambiguous, and so is a dynamic one on either side.
        let middle_sharding = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        assert_eq!(
            infer(vec![0.into(), 4.into(), 2.into()], vec![0.into(), 2.into()], &middle_sharding),
            Err(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` requires explicit output sharding for an ambiguous zero-sized reshape"
            ))),
        );
        let dynamic_sharding = Sharding::new(
            mesh,
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated(), ShardingDimension::replicated()],
        )
        .unwrap();
        assert_eq!(
            infer(
                vec![2.into(), 0.into(), Dimension::Dynamic(rows.clone())],
                vec![2.into(), 0.into(), 2.into()],
                &dynamic_sharding,
            ),
            Err(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` requires explicit output sharding for unaligned dynamic dimensions"
            ))),
        );
        assert_eq!(
            infer(
                vec![2.into(), 0.into(), 2.into()],
                vec![2.into(), 0.into(), Dimension::Dynamic(rows)],
                &dynamic_sharding,
            ),
            Err(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` requires explicit output sharding for unaligned dynamic dimensions"
            ))),
        );
    }

    #[test]
    fn test_infer_reshape_output_type_static_groups() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let infer = |input: Vec<Dimension>, output: Vec<Dimension>, sharding: &Sharding| {
            let input = ArrayType::new(DataType::F32, Shape::new(input)).with_sharding(sharding.clone()).unwrap();
            infer_reshape_output_type(&input, Shape::new(output), None)
                .map(|output| output.sharding().unwrap().dimensions().to_vec())
        };

        // A split distributes the contiguous mesh axes over the leading output factors they divide; a merge
        // concatenates the contiguous sharded prefix of the merged dimensions.
        let split_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap();
        assert_eq!(
            infer(vec![8.into()], vec![2.into(), 4.into()], &split_sharding),
            Ok(vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])]),
        );
        assert_eq!(
            infer(vec![8.into()], vec![4.into(), 2.into()], &split_sharding),
            Ok(vec![ShardingDimension::sharded(["x", "y"]), ShardingDimension::replicated()]),
        );
        let merge_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])])
                .unwrap();
        assert_eq!(
            infer(vec![2.into(), 4.into()], vec![8.into()], &merge_sharding),
            Ok(vec![ShardingDimension::sharded(["x", "y"])]),
        );
        let prefix_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        assert_eq!(
            infer(vec![2.into(), 4.into()], vec![8.into()], &prefix_sharding),
            Ok(vec![ShardingDimension::sharded(["x"])]),
        );

        // A replicated dimension before a sharded one breaks contiguity, an unconstrained dimension cannot be placed,
        // a factor the next mesh axis does not divide cannot be split, and every input mesh axis must be consumed.
        let suffix_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                .unwrap();
        assert_eq!(
            infer(vec![2.into(), 4.into()], vec![8.into()], &suffix_sharding),
            Err(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` cannot preserve non-contiguous sharding across a merge"
            ))),
        );
        let unconstrained_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::unconstrained()]).unwrap();
        assert_eq!(
            infer(vec![8.into()], vec![2.into(), 4.into()], &unconstrained_sharding),
            Err(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` requires explicit output sharding for unconstrained dimensions"
            ))),
        );
        assert_eq!(
            infer(vec![6.into()], vec![3.into(), 2.into()], &split_sharding),
            Err(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` cannot distribute sharding across the requested split factors"
            ))),
        );
        // A trailing unit-size mesh axis still needs placement after all output factors are exhausted.
        let unit_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("z", 1, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let unit_sharding = Sharding::new(unit_mesh, vec![ShardingDimension::sharded(["x", "y", "z"])]).unwrap();
        assert_eq!(
            infer(vec![4.into()], vec![2.into(), 2.into()], &unit_sharding),
            Err(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` cannot distribute all input mesh axes across the output dimensions"
            ))),
        );
    }

    #[test]
    fn test_infer_dynamic_reshape_output_type() {
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(9)).unwrap());
        let input = ArrayType::new_static(DataType::F32, [2, 3])
            .with_layout(Layout::Strided(StridedLayout::new(vec![12, 4])))
            .with_memory(Memory::Host { pinned: true });

        // A shape change keeps the memory placement and clears the layout; an identity keeps the complete type.
        assert_eq!(
            infer_dynamic_reshape_output_type(&input, Shape::new(vec![6.into()]), None),
            Ok(ArrayType::new_static(DataType::F32, [6]).with_memory(Memory::Host { pinned: true })),
        );
        assert_eq!(infer_dynamic_reshape_output_type(&input, input.shape().clone(), None), Ok(input.clone()));

        // Statically provable mismatches are rejected; dynamic relationships remain explicit graph facts.
        assert_eq!(
            infer_dynamic_reshape_output_type(&input, Shape::new(vec![5.into()]), None),
            Err(TypeError::invalid(format!("`{RESHAPE_OPERATION_NAME}` changes the number of elements"))),
        );
        assert_eq!(
            infer_dynamic_reshape_output_type(&input, Shape::new(vec![Dimension::Dynamic(rows.clone())]), None),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows)]))
                .with_memory(Memory::Host { pinned: true })),
        );

        // A requested output sharding is validated and carried; otherwise the input placement is inferred.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded_input = ArrayType::new_static(DataType::F32, [2, 3])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let requested = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(
            infer_dynamic_reshape_output_type(&sharded_input, Shape::new(vec![6.into()]), Some(&requested)),
            Ok(ArrayType::new_static(DataType::F32, [6]).with_sharding(requested).unwrap()),
        );
        assert_eq!(
            infer_dynamic_reshape_output_type(&sharded_input, Shape::new(vec![2.into(), 1.into(), 3.into()]), None),
            Ok(ArrayType::new_static(DataType::F32, [2, 1, 3])
                .with_sharding(
                    Sharding::new(
                        mesh,
                        vec![
                            ShardingDimension::sharded(["x"]),
                            ShardingDimension::replicated(),
                            ShardingDimension::replicated(),
                        ],
                    )
                    .unwrap(),
                )
                .unwrap()),
        );
    }
}
