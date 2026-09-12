use std::borrow::Cow;
use std::fmt::Display;
use std::sync::Arc;

use crate::arrays::{
    Array, ArrayAddressing, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrBatch,
    ArrayIrBatchingPolicy, ArrayIrType, ArrayIrValue, ArrayType, Dimension, DimensionType, DimensionValue,
    LinearResiduals, Shape, Sharding, ShardingDimension,
};
use crate::axes::{Axes, Axis};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    CotangentAccumulator, DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver,
    DifferentiationDual, DifferentiationError, DifferentiationPolicy, ElementwiseDerivativeAlignment,
    TransposableOperation, TranspositionContext, TranspositionDriver, transpose_projected_operation,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation, impl_reference_dischargeable_operation};
use crate::operations::constants::constant::{ConstantOperation, DimensionConstant};
use crate::operations::differentiation::linear_call::LinearCallOperation;
use crate::operations::dimensions::dimension_size::DimensionSizeOperation;
use crate::operations::manipulation::broadcasting::BroadcastOperation;
use crate::operations::manipulation::transposition::{Permutation, Transpose, TransposeOperation};
use crate::operations::math::add::AddOperation;
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue, PartialValue,
    PartiallyEvaluatableOperation,
};
use crate::programs::{
    EffectClass, EffectClasses, Effects, MaybeZero, Operation, OperationFormatter, OperationProjection, ProgramError,
    RegionInterface, TypeError, TypeIdentityRenaming, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

// TODO(eaplatanios): Review this.

/// Canonical operation name for [`ReshapeOperation`] and [`DynamicReshapeOperation`].
pub const RESHAPE_OPERATION_NAME: &str = "reshape";

/// Logical element visitation order for [`Reshape::reshape_with_order`], independent of physical storage layout.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum ReshapeOrder {
    /// C ordering: the final axis varies fastest when reading and writing elements.
    #[default]
    RowMajor,

    /// Fortran ordering: the first axis varies fastest when reading and writing elements.
    ColumnMajor,
}

/// Semantic parameters accepted by [`Reshape`].
///
/// A [`Shape`] converts directly into `ReshapeParameters`, preserving the ordinary `value.reshape(shape)` spelling.
/// Callers that need an input permutation or an explicit output [`Sharding`] can construct these parameters and apply
/// the corresponding builder methods. Unlike [`ReshapeOperation`], this type contains no Intermediate Representation
/// (IR) behavior and can be consumed directly by eager backends and type inference.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReshapeParameters {
    /// Output shape of this reshape.
    output_shape: Shape,

    /// Optional permutation of the input dimensions applied before reshaping.
    dimensions: Option<Permutation>,

    /// Optional requested output [`Sharding`].
    output_sharding: Option<Sharding>,
}

impl ReshapeParameters {
    /// Creates reshape parameters with the provided output shape.
    #[inline]
    pub fn new(output_shape: impl Into<Shape>) -> Self {
        Self { output_shape: output_shape.into(), dimensions: None, output_sharding: None }
    }

    /// Resolves signed output sizes against the input's known element count. One size may be `-1`, which is replaced
    /// by the quotient of the input count and the product of the other sizes. All other sizes must be nonnegative.
    ///
    /// A zero input count infers a zero axis when the other sizes have a nonzero product. Combining `-1` with an
    /// explicit zero is ambiguous and is rejected. Unknown input counts require [`DynamicReshape`] instead.
    ///
    /// # Parameters
    ///
    ///   - `input_shape`: Shape supplying the element count used for validation and inference.
    ///   - `output_sizes`: Requested output sizes, with at most one inferred `-1` entry.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid negative sizes, repeated inferred axes, ambiguous zero products, overflow, an
    /// unknown input count, or a target count that differs from the input count.
    pub fn from_sizes(input_shape: &Shape, output_sizes: &[isize]) -> Result<Self, TypeError> {
        let input_count = input_shape
            .element_count()?
            .ok_or_else(|| TypeError::invalid("`reshape` size inference requires a known input element count"))?;
        let mut inferred_axis = None;
        let mut sizes = Vec::with_capacity(output_sizes.len());
        for (axis, size) in output_sizes.iter().copied().enumerate() {
            if size == -1 {
                if inferred_axis.replace(axis).is_some() {
                    return Err(TypeError::invalid("`reshape` accepts at most one inferred `-1` dimension"));
                }
                sizes.push(1);
            } else {
                sizes.push(usize::try_from(size).map_err(|_| {
                    TypeError::invalid("`reshape` dimensions must be nonnegative or the inferred size `-1`")
                })?);
            }
        }
        let known_count = if sizes.contains(&0) {
            0
        } else {
            sizes
                .iter()
                .try_fold(1usize, |count, size| count.checked_mul(*size))
                .ok_or_else(|| TypeError::invalid("`reshape` output element count does not fit in `usize`"))?
        };
        if let Some(axis) = inferred_axis {
            if known_count == 0 {
                return Err(TypeError::invalid(
                    "cannot infer a `reshape` dimension when another output dimension is zero",
                ));
            }
            if !input_count.is_multiple_of(known_count) {
                return Err(TypeError::invalid("`reshape` inferred dimension does not divide the input element count"));
            }
            sizes[axis] = input_count / known_count;
        } else if known_count != input_count {
            return Err(TypeError::invalid(format!(
                "`reshape` output element count {known_count} differs from input element count {input_count}"
            )));
        }
        Ok(Self::new(Shape::new(sizes.into_iter().map(Dimension::Static).collect())))
    }

    /// Returns the output shape.
    #[inline]
    pub fn output_shape(&self) -> &Shape {
        &self.output_shape
    }

    /// Returns the optional input-dimension permutation.
    #[inline]
    pub fn dimensions(&self) -> Option<&Permutation> {
        self.dimensions.as_ref()
    }

    /// Returns the requested output sharding, if any.
    #[inline]
    pub fn output_sharding(&self) -> Option<&Sharding> {
        self.output_sharding.as_ref()
    }

    /// Returns this operation with `dimensions` used to permute the input before reshaping.
    #[inline]
    pub fn with_dimensions<P: Into<Permutation>>(mut self, dimensions: P) -> Self {
        self.dimensions = Some(dimensions.into());
        self
    }

    /// Returns this operation with the requested output `sharding`.
    #[inline]
    pub fn with_output_sharding(mut self, sharding: impl Into<Option<Sharding>>) -> Self {
        self.output_sharding = sharding.into();
        self
    }

    /// Returns whether this operation leaves the input dimension order unchanged for an input of rank `rank`.
    #[inline]
    fn has_identity_dimensions(&self, rank: usize) -> bool {
        self.dimensions
            .as_ref()
            .is_none_or(|dimensions| dimensions.normalize(rank).is_ok_and(|axes| axes.into_iter().eq(0..rank)))
    }
}

impl From<Shape> for ReshapeParameters {
    #[inline]
    fn from(shape: Shape) -> Self {
        Self::new(shape)
    }
}

/// [`Operation`] that reshapes its input array according to semantic [`ReshapeParameters`].
///
/// This is the member-family reshape primitive of the homogeneous array language: complete output geometry is carried
/// by the [`ArrayType`] metadata that [`ReshapeParameters`] describes, so the operation has exactly one input and no
/// explicit extent edges. The input shape is recoverable from the staged input types and is therefore not duplicated
/// in the payload. It and [`BroadcastOperation`](crate::operations::manipulation::BroadcastOperation) form the
/// homogeneous baseline that [`ProjectedContext`](crate::contexts::ProjectedContext) serves, which is why transform
/// rules for mixed operations can delegate to them once input geometry is resolved. Refer to the documentation of
/// [`Reshape`] for the underlying resolved-geometry contract.
///
/// Programs that need first-class dynamic extents stage [`DynamicReshapeOperation`] instead, which takes one explicit
/// first-class dimension input per output axis.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReshapeOperation {
    /// Semantic parameters carried by this operation.
    parameters: ReshapeParameters,
}

impl ReshapeOperation {
    /// Creates a new [`ReshapeOperation`] from semantic reshape `parameters`.
    #[inline]
    pub fn new(parameters: impl Into<ReshapeParameters>) -> Self {
        Self { parameters: parameters.into() }
    }

    /// Returns the semantic parameters carried by this operation.
    #[inline]
    pub fn parameters(&self) -> &ReshapeParameters {
        &self.parameters
    }
}

impl Display for ReshapeOperation {
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

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        check_count!("input", input_types, 1, TypeError);
        match input_types[0].reshape(self.parameters.clone()) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError::invalid(error.to_string())),
        }
    }

    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayType as crate::Type>::Identity>,
    ) -> Result<Self, TypeError> {
        Ok(Self::new(ReshapeParameters {
            output_shape: self.parameters.output_shape().rename_type_identities(renaming),
            dimensions: self.parameters.dimensions().cloned(),
            output_sharding: self.parameters.output_sharding().cloned(),
        }))
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("shape", self.parameters.output_shape())?;
            if let Some(dimensions) = self.parameters.dimensions() {
                operation.field(
                    "dimensions",
                    format_args!("{:?}", dimensions.iter().map(|axis| axis.value()).collect::<Vec<_>>()),
                )?;
            }
            if let Some(output_sharding) = self.parameters.output_sharding() {
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
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].reshape(self.parameters.clone())?])
    }
}

// Partial evaluation defers to the default fold-or-residualize behavior of
// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
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
            // Replicated input: there is no batch axis to thread through the reshape, so interpret it as given and
            // report the output replicated.
            return Ok(self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()])?.into());
        };
        let Dimension::Static(axis_size) = P::axis_dimension(context)? else {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{RESHAPE_OPERATION_NAME}` with a dynamic mapped extent requires `DynamicReshape` and explicit \
                     result-dimension inputs",
                ),
            });
        };
        let input_axis_size = ArrayBatch::common_batch_size(inputs)?.unwrap();
        if input_axis_size != axis_size {
            return Err(BatchingError::MismatchedBatchSizes { expected: axis_size, actual: input_axis_size });
        }
        let moved_input = inputs[0].move_axis(0)?;
        let output_shape = self.parameters.output_shape();
        let mut lifted_output_dimensions = Vec::with_capacity(output_shape.rank() + 1);
        lifted_output_dimensions.push(Dimension::Static(axis_size));
        lifted_output_dimensions.extend_from_slice(output_shape.dimensions());
        let mut lifted_parameters = ReshapeParameters::new(Shape::new(lifted_output_dimensions));
        if let Some(dimensions) = self.parameters.dimensions() {
            let mut lifted_dimensions = Vec::with_capacity(dimensions.len() + 1);
            lifted_dimensions.push(0);
            lifted_dimensions.extend(
                dimensions.normalize(inputs[0].unbatched_type().rank())?.into_iter().map(|dimension| dimension + 1),
            );
            lifted_parameters = lifted_parameters.with_dimensions(lifted_dimensions);
        }
        if let Some(output_sharding) = self.parameters.output_sharding() {
            lifted_parameters = lifted_parameters.with_output_sharding(lift_output_sharding_for_leading_batch_axis(
                output_sharding,
                ArrayBatch::sharding_for_inputs(inputs)?,
            )?);
        }
        Ok(ReshapeOperation::new(lifted_parameters)
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
            // Forward-mode differentiation rule for `ReshapeOperation`. `reshape` is structural-linear, and so the
            // tangent is the same reshape applied to the input tangent. The shared all-zero fast path handles a zero
            // input tangent before this rule is consulted, so the input tangent reaching here is always live.
            check_count!("input", inputs, 1, ProgramError);
            let primal = inputs[0].primal().reshape(operation.parameters().clone())?;
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                MaybeZero::Value(tangent) => MaybeZero::Value(tangent.reshape(operation.parameters().clone())?),
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType> + From<ReshapeOperation> + From<TransposeOperation>,
        Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<ArrayType> + Reshape + Transpose,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 1, DifferentiationError);
            let input_cotangent_type = inputs[0].r#type().cotangent()?;
            let permuted_input_cotangent_type = match operation.parameters().dimensions() {
                Some(dimensions) => input_cotangent_type.transpose(dimensions)?,
                None => input_cotangent_type.clone(),
            };
            match &outputs[0] {
                MaybeZero::Value(cotangent) => {
                    let bridge_sharding =
                        match (permuted_input_cotangent_type.sharding(), cotangent.r#type().sharding()) {
                            (Some(sharding), _) => Some(sharding.clone()),
                            (None, Some(sharding)) => Some(Sharding::replicated(
                                sharding.mesh().clone(),
                                permuted_input_cotangent_type.rank(),
                            )),
                            (None, None) => None,
                        };
                    let mut inverse_parameters = ReshapeParameters::new(permuted_input_cotangent_type.shape().clone());
                    if let Some(bridge_sharding) = bridge_sharding {
                        inverse_parameters = inverse_parameters.with_output_sharding(bridge_sharding);
                    }
                    let mut cotangent = cotangent.reshape(inverse_parameters)?;
                    if let Some(dimensions) = operation.parameters().dimensions() {
                        cotangent = cotangent.transpose(dimensions.inverse()?)?;
                    }
                    {
                        let contribution = MaybeZero::Value(cotangent.unalign_cotangent(&input_cotangent_type)?);
                        accumulators[0].accumulate(context, contribution)?;
                        Ok(())
                    }
                }
                MaybeZero::Zero(_) => Ok(()),
            }
        }
    },
}

/// Represents the ability to reshape an array without changing its element count or row-major element order.
///
/// `t.reshape(target_shape)` reinterprets `t`'s payload under the specified target [`Shape`]. The input and target
/// shapes must have equal element counts, which the type system must be able to establish from the two shapes alone.
/// Shape-polymorphic programs whose output extents are not recoverable from anonymous dynamic [`Dimension`] values
/// stage [`DynamicReshapeOperation`] instead, which takes one explicit first-class dimension input per output axis
/// and so expresses runtime shape arithmetic as ordinary graph values. When the input carries a [`Sharding`], singleton
/// dimensions are ignored and contiguous split/merge groups redistribute compatible mesh axes over their output
/// factors. Ambiguous dynamic, zero-sized, unconstrained, or non-contiguous placement changes require an explicit
/// output sharding. A non-identity reshape preserves the input memory space and clears explicit physical layout
/// metadata because the logical shape change does not determine a unique output storage layout.
///
/// # Examples
///
/// The following example shows how to use [`Reshape`] in practice:
///
/// ```rust
/// # use ryft_core::operations::manipulation::Reshape;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::arrays::Array;
/// # use ryft_core::arrays::{Shape, Dimension};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Reshape a length-6 vector to a `[2, 3]` matrix while keeping the row-major payload unchanged.
/// let x = Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let y = x.reshape(Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))?;
/// assert_eq!(y.to_f64s(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// # Ok(())
/// # }
/// ```
pub trait Reshape: Sized {
    /// Reshapes `self` according to semantic `parameters`. A [`Shape`] converts directly into
    /// [`ReshapeParameters`], so ordinary calls remain `value.reshape(shape)`.
    fn reshape<P: Into<ReshapeParameters>>(&self, parameters: P) -> Result<Self, ProgramError>;

    /// Reshapes with signed sizes, accepting one inferred `-1` dimension. Values are visited in row-major order;
    /// the final axis varies fastest. Refer to [`ReshapeParameters::from_sizes`] for zero-size and inference rules.
    /// The input element count must be known; first-class runtime dimensions use [`DynamicReshape`] instead.
    ///
    /// # Parameters
    ///
    ///   - `output_sizes`: Nonnegative target sizes with at most one `-1` entry inferred from the input count.
    fn reshape_to_sizes(&self, output_sizes: &[isize]) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType>,
    {
        self.reshape(ReshapeParameters::from_sizes(self.r#type().shape(), output_sizes)?)
    }

    /// Reshapes while reading the input and filling the output in the requested logical axis order. Row-major
    /// ordering makes the final axis vary fastest; column-major ordering makes the first axis vary fastest. This
    /// controls element order, independently of physical layout, and accepts the same inferred `-1` dimension as
    /// [`Self::reshape_to_sizes`]. Column-major ordering composes an input permutation, reshape, and output transpose.
    ///
    /// # Parameters
    ///
    ///   - `output_sizes`: Nonnegative target sizes with at most one inferred `-1` entry.
    ///   - `order`: Logical visitation order shared by the input and output.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{Array, ProgramError, Reshape, ReshapeOrder};
    /// # fn main() -> Result<(), ProgramError> {
    /// let matrix = Array::matrix(2, 3, vec![1i32, 2, 3, 4, 5, 6])?;
    /// let output = matrix.reshape_with_order(&[3, -1], ReshapeOrder::ColumnMajor)?;
    /// assert_eq!(output.elements::<i32>()?, vec![1, 5, 4, 3, 2, 6]);
    /// # Ok(())
    /// # }
    /// ```
    fn reshape_with_order(&self, output_sizes: &[isize], order: ReshapeOrder) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType> + Transpose,
    {
        let input_type = self.r#type();
        let parameters = ReshapeParameters::from_sizes(input_type.shape(), output_sizes)?;
        match order {
            ReshapeOrder::RowMajor => self.reshape(parameters),
            ReshapeOrder::ColumnMajor => {
                let output_rank = parameters.output_shape().rank();
                let output_shape = Shape::new(parameters.output_shape().dimensions().iter().rev().cloned().collect());
                self.reshape(
                    ReshapeParameters::new(output_shape)
                        .with_dimensions((0..input_type.rank()).rev().collect::<Vec<_>>()),
                )?
                .transpose((0..output_rank).rev().collect::<Vec<_>>())
            }
        }
    }

    /// Returns the input as a one-dimensional array in logical row-major order. The element count must be known;
    /// an empty input produces shape `[0]`. Storage sharing or copying is determined by the backend.
    fn ravel(&self) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType>,
    {
        self.reshape_to_sizes(&[-1])
    }

    /// Returns a one-dimensional array in logical row-major order, with the same contract as [`Self::ravel`].
    /// This function does not require an independent storage allocation.
    fn flatten(&self) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType>,
    {
        self.ravel()
    }

    /// Inserts one size-one axis without changing element order. `axis` addresses the result rank, so `0` inserts
    /// a leading axis and `-1` appends a trailing axis. This reshape convenience requires resolved input geometry;
    /// it does not introduce a separate axis-identity primitive for transformations.
    ///
    /// # Parameters
    ///
    ///   - `axis`: Position of the inserted axis, normalized against the result rank.
    fn expand_dims<A: Into<Axis>>(&self, axis: A) -> Result<Self, ProgramError>
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
    /// Use [`Self::squeeze_all`] to remove every singleton axis. This convenience uses the existing reshape contract,
    /// including its restriction on shape-changing dynamic geometry.
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
                return Err(TypeError::invalid(format!("cannot squeeze axis {axis} whose size is not one")).into());
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
    fn reshape<P: Into<ReshapeParameters>>(&self, parameters: P) -> Result<ArrayType, ProgramError> {
        let parameters = parameters.into();
        let permuted_input = match parameters.dimensions() {
            Some(dimensions) => self.transpose(dimensions)?,
            None => self.clone(),
        };
        let shape = parameters.output_shape().clone();
        if permuted_input.shape() != &shape {
            if shape.dimensions().iter().any(|size| matches!(size, Dimension::Dynamic(_))) {
                return Err(TypeError::invalid(format!(
                    "`{RESHAPE_OPERATION_NAME}` requires explicit result-dimension inputs for a dynamic output shape"
                ))
                .into());
            }
            let Some(input_elements) =
                permuted_input.element_count().map_err(|error| TypeError::invalid(error.to_string()))?
            else {
                return Err(TypeError::invalid(format!(
                    "`{RESHAPE_OPERATION_NAME}` requires explicit result-dimension inputs for a dynamic input shape"
                ))
                .into());
            };
            let Some(output_elements) = shape.element_count().map_err(|error| TypeError::invalid(error.to_string()))?
            else {
                return Err(TypeError::invalid(format!(
                    "`{RESHAPE_OPERATION_NAME}` requires explicit result-dimension inputs for a dynamic output shape"
                ))
                .into());
            };
            if input_elements != output_elements {
                return Err(
                    TypeError::invalid(format!("`{RESHAPE_OPERATION_NAME}` changes the number of elements")).into()
                );
            }
        }

        let sharding = match parameters.output_sharding() {
            Some(requested) => Some(validate_requested_reshape_sharding(&permuted_input, &shape, requested)?),
            None => permuted_input
                .sharding()
                .map(|sharding| infer_reshape_sharding(&permuted_input, &shape, sharding))
                .transpose()?,
        };

        if parameters.has_identity_dimensions(self.rank()) && self.shape() == &shape {
            return self.clone().with_sharding(sharding).map_err(|error| TypeError::invalid(error.to_string()).into());
        }

        ArrayType::new(self.data_type(), shape)
            .with_memory(self.memory())
            .with_sharding(sharding)
            .map_err(|error| TypeError::invalid(error.to_string()).into())
    }
}

impl Reshape for Array {
    fn reshape<P: Into<ReshapeParameters>>(&self, parameters: P) -> Result<Self, ProgramError> {
        // Delegate to the type-level reshape so all element-count and sharding validation remains shared with staged
        // execution.
        let parameters = parameters.into();
        let output_type = self.r#type().reshape(parameters.clone())?;
        let transposed = parameters.dimensions().map(|dimensions| self.transpose(dimensions)).transpose()?;
        let input = transposed.as_ref().unwrap_or(self);
        let input_addressing = ArrayAddressing::new(input.r#type().into_owned())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        if input_addressing.is_dense_row_major() && output_addressing.is_dense_row_major() {
            bytes.copy_from_slice(input.storage_bytes());
        } else {
            for index in 0..input_addressing.element_count() {
                bytes[output_addressing.byte_range_for_flat_index(index)]
                    .copy_from_slice(&input.storage_bytes()[input_addressing.byte_range_for_flat_index(index)]);
            }
        }
        Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
    }
}

// Any context-carrying value reshapes by binding a [`ReshapeOperation`] through its own context. The
// `From<ReshapeOperation>` bound makes this disjoint from the eager value types (whose context operation is
// `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete implementations.
impl<V: Value<Type = ArrayType>> Reshape for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<ReshapeOperation>,
{
    #[inline]
    fn reshape<P: Into<ReshapeParameters>>(&self, parameters: P) -> Result<Self, ProgramError> {
        let operation = ReshapeOperation::new(parameters);
        let input_type = self.r#type().into_owned();
        let output_type = input_type.reshape(operation.parameters().clone())?;
        if operation.parameters().has_identity_dimensions(input_type.rank()) && input_type == output_type {
            return Ok(self.clone());
        }
        let mut outputs = self.dispatch_domain().bind(operation, Vec::new(), std::slice::from_ref(self))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Mixed [`Operation`] that reshapes one array using one explicit first-class dimension input per output axis.
///
/// Input zero is the array. Every remaining input describes the corresponding output-axis extent, in order.
/// Exact dimension types produce static axes while non-exact dimension types retain their variables as dynamic axes.
/// The operation therefore carries only reshape attributes; it does not duplicate its output shape or encode shape
/// arithmetic in its payload.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DynamicReshapeOperation {
    /// Whether the construction signature proves equal input and output element counts.
    element_count_proven: bool,

    /// Optional permutation of the input dimensions applied before reshaping.
    dimensions: Option<Permutation>,

    /// Optional requested output [`Sharding`].
    output_sharding: Option<Sharding>,
}

impl DynamicReshapeOperation {
    /// Creates a reshape with no input permutation or requested output sharding.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the optional input-dimension permutation.
    #[inline]
    pub fn dimensions(&self) -> Option<&Permutation> {
        self.dimensions.as_ref()
    }

    /// Returns the requested output sharding, if any.
    #[inline]
    pub fn output_sharding(&self) -> Option<&Sharding> {
        self.output_sharding.as_ref()
    }

    /// Validates the complete input signature and records whether its shapes prove equal element counts.
    ///
    /// Unproven operations retain an ordered runtime assertion, even when their output is unused. The proof is
    /// revalidated during type inference, so reusing an operation cannot silently weaken this requirement.
    ///
    /// # Parameters
    ///
    ///   - `input_types`: The array type followed by one dimension type per output axis.
    pub fn with_input_types(mut self, input_types: &[ArrayIrType]) -> Result<Self, TypeError> {
        self.element_count_proven = false;
        let output_types = self.infer_output_types(input_types, &[])?;
        let input = <&ArrayType>::try_from(&input_types[0])?;
        let output = <&ArrayType>::try_from(&output_types[0])?;
        self.element_count_proven = reshape_element_counts_equal(input.shape(), output.shape())?;
        Ok(self)
    }

    /// Returns this operation with `dimensions` used to permute the input before reshaping.
    #[inline]
    pub fn with_dimensions<P: Into<Permutation>>(mut self, dimensions: P) -> Self {
        self.dimensions = Some(dimensions.into());
        self
    }

    /// Returns this operation with the requested output `sharding`.
    #[inline]
    pub fn with_output_sharding(mut self, sharding: impl Into<Option<Sharding>>) -> Self {
        self.output_sharding = sharding.into();
        self
    }

    /// Returns whether this operation leaves the input dimension order unchanged for an input of rank `rank`.
    #[inline]
    fn has_identity_dimensions(&self, rank: usize) -> bool {
        self.dimensions
            .as_ref()
            .is_none_or(|dimensions| dimensions.normalize(rank).is_ok_and(|axes| axes.into_iter().eq(0..rank)))
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
                "`{RESHAPE_OPERATION_NAME}` expects an array followed by its output extents"
            )));
        };
        let input_type = <&ArrayType>::try_from(input_type)?;
        let output_shape = Shape::new(ArrayIrType::extents(output_extent_types)?);
        let output_type = infer_explicit_reshape_output_type(input_type, output_shape, self)?;
        if self.element_count_proven && !reshape_element_counts_equal(input_type.shape(), output_type.shape())? {
            return Err(TypeError::invalid("`reshape` input types do not preserve its element-count proof"));
        }
        Ok(vec![output_type.into()])
    }

    fn effects(&self) -> Cow<'_, Effects> {
        Cow::Owned(Effects::explicit(if self.element_count_proven {
            EffectClasses::NONE
        } else {
            EffectClasses::single(EffectClass::OrderedAssertion)
        }))
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        if !self.element_count_proven && self.dimensions.is_none() && self.output_sharding.is_none() {
            return formatter.write_str(RESHAPE_OPERATION_NAME);
        }
        OperationFormatter::new(formatter, indentation, RESHAPE_OPERATION_NAME)?.bracketed(|operation| {
            if self.element_count_proven {
                operation.field("element_count_proven", true)?;
            }
            if let Some(dimensions) = &self.dimensions {
                operation.field(
                    "dimensions",
                    format_args!("{:?}", dimensions.iter().map(|axis| axis.value()).collect::<Vec<_>>()),
                )?;
            }
            if let Some(output_sharding) = &self.output_sharding {
                operation.field("output_sharding", output_sharding)?;
            }
            Ok(())
        })
    }
}

impl_reference_dischargeable_operation!(@reference_free DynamicReshapeOperation);

impl<C> InterpretableOperation<C> for DynamicReshapeOperation
where
    C: Domain<Type = ArrayIrType>,
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType> + Reshape>
        + ValueProjection<DimensionType, Projected = DimensionValue>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        let Some((input, output_extents)) = inputs.split_first() else {
            return Err(TypeError::invalid(format!(
                "`{RESHAPE_OPERATION_NAME}` expects an array followed by its output extents"
            ))
            .into());
        };
        let input = <C::Value as ValueProjection<ArrayType>>::into_projected(input.clone())?;
        let output_shape = Shape::new(
            output_extents
                .iter()
                .cloned()
                .map(<C::Value as ValueProjection<DimensionType>>::into_projected)
                .map(|result| result.map(|extent| Dimension::Static(extent.extent())))
                .collect::<Result<Vec<_>, _>>()?,
        );
        let mut parameters = ReshapeParameters::new(output_shape);
        if let Some(dimensions) = self.dimensions() {
            parameters = parameters.with_dimensions(dimensions.clone());
        }
        if let Some(output_sharding) = self.output_sharding() {
            parameters = parameters.with_output_sharding(output_sharding.clone());
        }
        Ok(vec![<C::Value as ValueProjection<ArrayType>>::from_projected(input.reshape(parameters)?)])
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
            && self.has_identity_dimensions(input_type.rank())
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

// Batching rule for [`DynamicReshapeOperation`]. Explicit output extents remain replicated shape values. A mapped
// input is canonicalized to a leading batch axis, and that axis is inserted into both the reshape geometry and the
// output sharding before the mixed operation is replayed.
impl<C> BatchableOperation<C, ArrayIrBatchingPolicy> for DynamicReshapeOperation
where
    C: Context<Type = ArrayIrType, Operation: From<DynamicReshapeOperation>>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        _driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatchingPolicy>, BatchingError> {
        let Some((input, output_extents)) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        <&ArrayType>::try_from(&input.unbatched_type())?;
        if !input.ragged_axes().is_empty() {
            return Err(BatchingError::UnsupportedOperation {
                message: format!("dynamic {RESHAPE_OPERATION_NAME} does not support bounded ragged array inputs"),
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
        if let Some(dimensions) = self.dimensions() {
            let mut lifted_dimensions = Vec::with_capacity(dimensions.len() + 1);
            lifted_dimensions.push(0);
            lifted_dimensions.extend(
                dimensions
                    .normalize(<&ArrayType>::try_from(&input.unbatched_type())?.rank())?
                    .into_iter()
                    .map(|dimension| dimension + 1),
            );
            operation = operation.with_dimensions(lifted_dimensions);
        }
        if let Some(output_sharding) = self.output_sharding() {
            operation = operation.with_output_sharding(lift_output_sharding_for_leading_batch_axis(
                output_sharding,
                context.axis_sharding().clone(),
            )?);
        }

        let mut lifted_inputs = Vec::with_capacity(inputs.len() + 1);
        lifted_inputs.push(moved_input);
        lifted_inputs.push(context.axis_extent().clone());
        lifted_inputs.extend(output_extents.iter().map(|extent| extent.value().clone()));
        Ok(context
            .parent()
            .bind(operation, Vec::new(), lifted_inputs.as_slice())?
            .into_iter()
            .map(|output| ArrayIrBatch::new(output, BatchAxis::from_position(0)))
            .collect::<Result<Vec<_>, _>>()?
            .into())
    }
}

// Forward-mode rule for mixed reshape. The explicit output extents are ordinary non-differentiated shape values.
// Static input cotangent geometry replays the mixed reshape directly; dynamic geometry retains the exact input shape
// so the linear transpose can reconstruct the inverse reshape from first-class dimension residuals.
impl<C> DifferentiableOperation<C> for DynamicReshapeOperation
where
    C: Context<Type = ArrayIrType>,
    C::Operation: From<DimensionSizeOperation>
        + From<LinearCallOperation<ArrayIrType>>
        + From<DynamicReshapeOperation>
        + From<ConstantOperation<DimensionValue>>
        + OperationProjection<ArrayType, Projected: From<BroadcastOperation> + From<TransposeOperation>>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let destinations = context;
        let context = destinations.primal();
        let Some(_) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal_operation = self
            .clone()
            .with_input_types(&primal_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>())?;
        let mut primal_outputs = context.bind(primal_operation, Vec::new(), primal_inputs.as_slice())?;
        check_count!("output", primal_outputs, 1, ProgramError);
        let primal = primal_outputs.remove(0);
        let output_primal = primal;
        let primal = destinations.primal_to_tangent(output_primal.clone())?;
        let tangent_inputs = destinations.dual_primal_to_tangent(inputs)?;
        let inputs = tangent_inputs.as_slice();
        let (array, output_extents) = inputs.split_first().unwrap();
        let context = destinations.tangent();
        let tangent = match array.tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
            MaybeZero::Value(array_tangent) => {
                let input_type = <&ArrayType>::try_from(array.primal().r#type().as_ref())?.clone();
                let input_cotangent_type = input_type.cotangent()?;
                let permuted_input_cotangent_type = match self.dimensions() {
                    Some(dimensions) => input_cotangent_type.transpose(dimensions)?,
                    None => input_cotangent_type.clone(),
                };
                if permuted_input_cotangent_type
                    .shape()
                    .dimensions()
                    .iter()
                    .all(|dimension| matches!(dimension, Dimension::Static(_)))
                {
                    let mut tangent_inputs = Vec::with_capacity(inputs.len());
                    tangent_inputs.push(array_tangent.clone());
                    tangent_inputs.extend(output_extents.iter().map(|extent| extent.primal().clone()));
                    let operation = self.clone().with_input_types(
                        &tangent_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
                    )?;
                    let mut outputs = context.bind(operation, Vec::new(), tangent_inputs.as_slice())?;
                    check_count!("output", outputs, 1, ProgramError);
                    MaybeZero::Value(outputs.remove(0))
                } else {
                    // Record each distinct dynamic input extent while the source array is available. Repeated type
                    // identities reuse one residual SSA value in first-use order.
                    let mut residuals = LinearResiduals::new();
                    let output_extents =
                        residuals.retain_all(output_extents.iter().map(|extent| extent.primal().clone()));
                    let input_shape = residuals.retain_shape(context, array.primal())?;
                    let permuted_input_shape = match self.dimensions() {
                        Some(dimensions) => input_shape.transposed(dimensions)?,
                        None => input_shape,
                    };

                    // Both linear regions share one deterministic residual boundary. The forward region consumes the
                    // retained output extents; the transpose region consumes the retained exact input geometry.
                    let forward_operation = self.clone();
                    let forward_output_extents = output_extents.clone();
                    let transpose_operation = self.clone();
                    let transpose_target_type = input_cotangent_type.clone();
                    let transpose_permuted_type = permuted_input_cotangent_type.clone();
                    let tangent = LinearCallOperation::stage(
                        context,
                        residuals.into_values(),
                        vec![array_tangent.clone()],
                        move |residuals, linear_inputs| {
                            let mut reshape_inputs = Vec::with_capacity(1 + forward_output_extents.len());
                            reshape_inputs.push(linear_inputs[0].clone());
                            reshape_inputs.extend(forward_output_extents.iter().map(|index| residuals[*index].clone()));
                            linear_inputs[0].dispatch_domain().bind(
                                forward_operation.with_input_types(
                                    &reshape_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
                                )?,
                                Vec::new(),
                                reshape_inputs.as_slice(),
                            )
                        },
                        move |residuals, output_cotangents| {
                            let transpose_context = output_cotangents[0].dispatch_domain();
                            let bridge_sharding = match (
                                transpose_permuted_type.sharding(),
                                <&ArrayType>::try_from(output_cotangents[0].r#type().as_ref())?.sharding(),
                            ) {
                                (Some(sharding), _) => Some(sharding.clone()),
                                (None, Some(sharding)) => {
                                    Some(Sharding::replicated(sharding.mesh().clone(), transpose_permuted_type.rank()))
                                }
                                (None, None) => None,
                            };
                            let mut inverse_operation = DynamicReshapeOperation::new();
                            if let Some(bridge_sharding) = bridge_sharding {
                                inverse_operation = inverse_operation.with_output_sharding(bridge_sharding);
                            }
                            let mut inverse_inputs = Vec::with_capacity(transpose_permuted_type.rank() + 1);
                            inverse_inputs.push(output_cotangents[0].clone());
                            inverse_inputs.extend(permuted_input_shape.dimensions(&transpose_context, residuals)?);
                            let inverse_operation = inverse_operation.with_input_types(
                                &inverse_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
                            )?;
                            let mut outputs =
                                transpose_context.bind(inverse_operation, Vec::new(), inverse_inputs.as_slice())?;
                            check_count!("output", outputs, 1, ProgramError);
                            let cotangent = outputs.remove(0);
                            let cotangent = if let Some(dimensions) = transpose_operation.dimensions() {
                                transpose_context
                                    .bind(
                                        <C::Operation as OperationProjection<ArrayType>>::Projected::from(
                                            TransposeOperation::new(dimensions.inverse()?),
                                        ),
                                        Vec::new(),
                                        std::slice::from_ref(&cotangent),
                                    )?
                                    .remove(0)
                            } else {
                                cotangent
                            };
                            // The inverse geometry is exact, but reshape clears layouts and may need replicated
                            // bridge sharding. Restore the original cotangent's complete storage metadata after
                            // undoing the permutation, just as the homogeneous rule's unalignment does.
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
}

// Direct transposition rule for mixed reshape. Static input geometry delegates to the homogeneous array pullback,
// while every explicit output extent receives a structural-zero cotangent. Dynamic input geometry requires
// linearization so [`DifferentiableOperation::jvp`] can retain its exact extents as residuals.
impl<V, O> TransposableOperation<V, O> for DynamicReshapeOperation
where
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    O: Operation<Type = ArrayIrType> + From<AddOperation<ArrayIrType>> + OperationProjection<ArrayType>,
    <O as OperationProjection<ArrayType>>::Projected: From<ReshapeOperation>
        + From<TransposeOperation>
        + TransposableOperation<
            <V as ValueProjection<ArrayType>>::Projected,
            <O as OperationProjection<ArrayType>>::Projected,
        >,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        check_count!("output", outputs, 1, ProgramError);
        check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);

        let Some((input, _output_extents)) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        let input_cotangent_type = <&ArrayType>::try_from(input.r#type().as_ref())?.cotangent()?;
        let permuted_input_cotangent_type = match self.dimensions() {
            Some(dimensions) => input_cotangent_type.transpose(dimensions)?,
            None => input_cotangent_type.clone(),
        };
        if permuted_input_cotangent_type
            .shape()
            .dimensions()
            .iter()
            .any(|dimension| matches!(dimension, Dimension::Dynamic(_)))
        {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "direct transposition of a dynamic `{RESHAPE_OPERATION_NAME}` requires linearization so its input \
                     extents are available as explicit residuals",
                ),
            }
            .into());
        }

        let output_type = match outputs {
            [MaybeZero::Zero(r#type)] => <&ArrayType>::try_from(r#type)?.clone(),
            [MaybeZero::Value(value)] => <&ArrayType>::try_from(value.r#type().as_ref())?.clone(),
            _ => return Err(ProgramError::InvalidOutputCount { expected: 1, actual: outputs.len() }.into()),
        };
        let mut parameters = ReshapeParameters::new(output_type.shape().clone());
        if let Some(dimensions) = self.dimensions() {
            parameters = parameters.with_dimensions(dimensions.clone());
        }
        if let Some(output_sharding) = self.output_sharding() {
            parameters = parameters.with_output_sharding(output_sharding.clone());
        }
        let operation = <O as OperationProjection<ArrayType>>::Projected::from(ReshapeOperation::new(parameters));
        // Dimension inputs do not receive cotangents; forward only the array inputs' handles.
        transpose_projected_operation(context, &operation, std::slice::from_ref(input), outputs, &accumulators[..1])
    }
}

/// Reshapes an array using one explicit first-class dimension value per output axis.
///
/// This is the shape-polymorphic counterpart of [`Reshape`], which reads its complete output geometry from
/// [`ReshapeParameters`]. Exact dimension values describe static axes and computed dimension values describe dynamic
/// axes. Both forms bind the same [`DynamicReshapeOperation`], so runtime shape arithmetic stays an ordinary graph
/// computation instead of a type-level side condition; backend lowering chooses the appropriate static, bounded, or
/// dynamic representation from the inferred result type.
///
/// The output extents must multiply to the input element count, including when either shape contains a zero axis.
/// When the input types cannot prove this equality, the operation retains an ordered runtime assertion even if its
/// result is unused. Each dynamic extent also retains its declared bounds. Backends may reject unbounded geometry
/// or a reshape whose input and output physical capacities cannot be represented by their runtime reshape support;
/// equal logical element counts alone do not guarantee that every bounded shape can be compiled.
///
/// Exact host sizes can use [`DynamicReshape::dynamic_reshape_to_sizes`]:
///
/// ```rust
/// use ryft_core::operations::manipulation::DynamicReshape;
/// use ryft_core::{Array, ArrayIrValue};
///
/// let input = ArrayIrValue::Array(Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
/// let output = input.dynamic_reshape_to_sizes(&[2, 3]).unwrap();
/// assert_eq!(output, ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()));
/// ```
///
/// Computed or input dimensions remain ordinary SSA inputs, which is what makes a runtime-derived output shape
/// expressible. Here a `[batch, 6]` input is reshaped so that its dynamic leading extent is read off the input while
/// its trailing extent is an exact lifted dimension. Extents derived by first-class dimension arithmetic work the
/// same way, using the [`DimensionArithmetic`](crate::DimensionArithmetic) capability (e.g.,
/// `rows.dimension_mul(&columns)?`) directly on the composite values.
///
/// ```rust
/// use ryft_core::operations::manipulation::DynamicReshape;
/// use ryft_core::arrays::{
///     ArrayIrType, ArrayType, DataType, Dimension, DimensionBounds, DimensionValue, DimensionVariable, Shape,
/// };
/// use ryft_core::{
///     Array, ArrayIrOperation, ArrayIrValue, Context, DimensionSize, StagingContext, TracingContext, Typed,
/// };
///
/// type C = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
///
/// let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
/// let input_type =
///     ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(6)]));
/// let context = C::new();
/// let input = context.input(ArrayIrType::Array(input_type));
/// let rows = input.dimension_size(0).unwrap();
/// let columns = context.lift(DimensionValue::constant(2).unwrap().into()).unwrap();
/// let depth = context.lift(DimensionValue::constant(3).unwrap().into()).unwrap();
/// let output = input.dynamic_reshape(&[rows, columns, depth]).unwrap();
/// assert_eq!(output.r#type().to_string(), "f32[batch, 2, 3]");
/// ```
pub trait DynamicReshape: Value<Type = ArrayIrType> + Sized {
    /// Reshapes `self` to the output shape described by `output_dimensions`, one first-class value per output axis.
    ///
    /// # Parameters
    ///
    ///   - `output_dimensions`: Nonnegative first-class dimension values in output-axis order. Their product must
    ///     equal the input element count; an empty slice requests a scalar. Exact values infer static axes, while
    ///     non-exact values retain their dimension identities and bounds.
    fn dynamic_reshape(&self, output_dimensions: &[Self]) -> Result<Self, ProgramError> {
        self.dynamic_reshape_with_parameters(output_dimensions, None, None)
    }

    /// Reshapes `self` with an optional permutation applied to the input dimensions before the reshape and an
    /// explicit requested output sharding.
    ///
    /// # Parameters
    ///
    ///   - `output_dimensions`: Nonnegative first-class dimension values in output-axis order. Their product must
    ///     equal the input element count; an empty slice requests a scalar. Exact values infer static axes, while
    ///     non-exact values retain their dimension identities and bounds.
    ///   - `dimensions`: Optional permutation of all input axes, applied before reading elements in row-major order.
    ///     Negative axes are normalized against the input rank; each input axis must occur exactly once.
    ///   - `output_sharding`: Requested placement for the output axes. When absent, compatible input placement is
    ///     propagated; ambiguous placement changes require an explicit sharding.
    fn dynamic_reshape_with_parameters(
        &self,
        output_dimensions: &[Self],
        dimensions: Option<Permutation>,
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError>;

    /// Reshapes the input to an exact static shape by lifting every size into a dimension constant in its context.
    /// The sizes must multiply to the input element count; runtime inputs are checked when equality is not provable.
    ///
    /// # Parameters
    ///
    ///   - `output_sizes`: Output-axis sizes, including any zero axes. An empty slice requests a scalar. These are
    ///     exact extents rather than capacity bounds; inferred `-1` sizes belong to [`Reshape::reshape_to_sizes`].
    fn dynamic_reshape_to_sizes(&self, output_sizes: &[usize]) -> Result<Self, ProgramError>
    where
        Self::DispatchDomain: Context<Type = ArrayIrType>,
        Self::DispatchDomain: DimensionConstant,
    {
        let output_dimensions = output_sizes
            .iter()
            .map(|extent| self.dispatch_domain().dimension_constant(*extent))
            .collect::<Result<Vec<_>, _>>()?;
        self.dynamic_reshape(output_dimensions.as_slice())
    }
}

impl<A: Reshape + Value<Type = ArrayType>> DynamicReshape for ArrayIrValue<A> {
    fn dynamic_reshape_with_parameters(
        &self,
        output_dimensions: &[Self],
        dimensions: Option<Permutation>,
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError> {
        // Concrete composite values resolve every explicit extent input to its runtime value, so the mixed reshape
        // executes as the ordinary member reshape of the fully resolved output shape.
        let input = <Self as ValueProjection<ArrayType>>::projected(self)?;
        let output_shape = Shape::new(
            output_dimensions
                .iter()
                .map(<Self as ValueProjection<DimensionType>>::projected)
                .map(|result| result.map(|dimension| Dimension::Static(dimension.extent())))
                .collect::<Result<Vec<_>, _>>()?,
        );
        let mut parameters = ReshapeParameters::new(output_shape);
        if let Some(dimensions) = dimensions {
            parameters = parameters.with_dimensions(dimensions);
        }
        if let Some(output_sharding) = output_sharding {
            parameters = parameters.with_output_sharding(output_sharding);
        }
        Ok(Self::Array(input.reshape(parameters)?))
    }
}

impl<
    V: Value<Type = ArrayIrType, DispatchDomain: Context<Type = ArrayIrType, Operation: From<DynamicReshapeOperation>>>,
> DynamicReshape for V
{
    fn dynamic_reshape_with_parameters(
        &self,
        output_dimensions: &[Self],
        dimensions: Option<Permutation>,
        output_sharding: Option<Sharding>,
    ) -> Result<Self, ProgramError> {
        let input_type = self.r#type();
        let input_type = <&ArrayType>::try_from(input_type.as_ref())?;
        let output_shape =
            Shape::new(ArrayIrType::extents(output_dimensions.iter().map(|dimension| dimension.r#type()))?);
        let mut operation = DynamicReshapeOperation::new().with_output_sharding(output_sharding);
        if let Some(dimensions) = dimensions {
            operation = operation.with_dimensions(dimensions);
        }
        let output_type = infer_explicit_reshape_output_type(input_type, output_shape, &operation)?;

        // A static identity reshape cannot observe its exact extent inputs, so it stages nothing. Dynamic geometry
        // keeps the instruction because its inputs assert the runtime element-count relation.
        if operation.has_identity_dimensions(input_type.rank())
            && input_type.static_shape().is_some()
            && &output_type == input_type
        {
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

/// Inserts batching's physical leading dimension into a logical per-item output sharding.
pub(crate) fn lift_output_sharding_for_leading_batch_axis(
    output_sharding: &Sharding,
    batch_dimension: ShardingDimension,
) -> Result<Sharding, BatchingError> {
    let mut dimensions = output_sharding.dimensions().to_vec();
    dimensions.insert(0, batch_dimension.clone());
    let mut varying_manual_axes = output_sharding.varying_manual_axes().clone();
    if let ShardingDimension::Sharded(axis_names) = batch_dimension {
        for axis_name in axis_names {
            varying_manual_axes.remove(&axis_name);
        }
    }
    Sharding::new(output_sharding.mesh().clone(), dimensions)
        .and_then(|sharding| sharding.with_unreduced_axes(output_sharding.unreduced_axes().clone()))
        .and_then(|sharding| sharding.with_reduced_axes(output_sharding.reduced_axes().clone()))
        .and_then(|sharding| sharding.with_varying_manual_axes(varying_manual_axes))
        .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })
}

/// Proves equal products by comparing static coefficients and matching dynamic identities with multiplicity.
fn reshape_element_counts_equal(input: &Shape, output: &Shape) -> Result<bool, TypeError> {
    let coefficient = |shape: &Shape| {
        if shape.dimensions().contains(&Dimension::Static(0)) {
            return Ok(0usize);
        }
        shape.dimensions().iter().try_fold(1usize, |product, dimension| {
            product
                .checked_mul(dimension.value().unwrap_or(1))
                .ok_or_else(|| TypeError::invalid("`reshape` element count overflows the supported range"))
        })
    };
    let input_coefficient = coefficient(input)?;
    let output_coefficient = coefficient(output)?;
    if input_coefficient != output_coefficient {
        return Ok(false);
    }
    if input_coefficient == 0 {
        return Ok(true);
    }
    let mut remaining = output.dimensions().iter().filter(|dimension| dimension.value().is_none()).collect::<Vec<_>>();
    for dimension in input.dimensions().iter().filter(|dimension| dimension.value().is_none()) {
        let Some(index) = remaining.iter().position(|candidate| *candidate == dimension) else {
            return Ok(false);
        };
        remaining.swap_remove(index);
    }
    Ok(remaining.is_empty())
}

/// Infers the result of the canonical mixed reshape from its explicit output extent types.
fn infer_explicit_reshape_output_type(
    input: &ArrayType,
    output_shape: Shape,
    operation: &DynamicReshapeOperation,
) -> Result<ArrayType, TypeError> {
    let permuted_input = match operation.dimensions() {
        Some(dimensions) => input.transpose(dimensions).map_err(|error| TypeError::invalid(error.to_string()))?,
        None => input.clone(),
    };

    // Reject statically provable element-count mismatches immediately. Dynamic relationships remain explicit graph
    // facts; eager execution checks concrete sizes and lowering must enforce the runtime requirement explicitly.
    if let (Some(input_elements), Some(output_elements)) =
        (permuted_input.element_count()?, output_shape.element_count()?)
        && input_elements != output_elements
    {
        return Err(TypeError::invalid(format!("`{RESHAPE_OPERATION_NAME}` changes the number of elements")));
    }

    let sharding = match operation.output_sharding() {
        Some(requested) => Some(validate_requested_reshape_sharding(&permuted_input, &output_shape, requested)?),
        None => permuted_input
            .sharding()
            .map(|sharding| infer_reshape_sharding(&permuted_input, &output_shape, sharding))
            .transpose()?,
    };

    if operation.has_identity_dimensions(input.rank()) && input.shape() == &output_shape {
        return input.clone().with_sharding(sharding).map_err(|error| TypeError::invalid(error.to_string()));
    }

    ArrayType::new(input.data_type(), output_shape)
        .with_memory(input.memory())
        .with_sharding(sharding)
        .map_err(|error| TypeError::invalid(error.to_string()))
}

/// Validates an explicitly requested output sharding for a reshape.
fn validate_requested_reshape_sharding(
    input: &ArrayType,
    output_shape: &Shape,
    requested: &Sharding,
) -> Result<Sharding, TypeError> {
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
            "`{RESHAPE_OPERATION_NAME}` requested output sharding uses a different mesh"
        )));
    }
    if requested.references_auto_axis() {
        return Err(TypeError::invalid(format!(
            "`{RESHAPE_OPERATION_NAME}` requested output sharding cannot reference auto mesh axes"
        )));
    }
    let input_unreduced = input.sharding().map(Sharding::unreduced_axes).cloned().unwrap_or_default();
    let input_reduced = input.sharding().map(Sharding::reduced_axes).cloned().unwrap_or_default();
    let input_varying = input.sharding().map(Sharding::varying_manual_axes).cloned().unwrap_or_default();
    if requested.unreduced_axes() != &input_unreduced {
        return Err(TypeError::invalid(format!(
            "`{RESHAPE_OPERATION_NAME}` requested output sharding changes the unreduced mesh axes"
        )));
    }
    if requested.reduced_axes() != &input_reduced {
        return Err(TypeError::invalid(format!(
            "`{RESHAPE_OPERATION_NAME}` requested output sharding changes the reduced mesh axes"
        )));
    }
    if requested.varying_manual_axes() != &input_varying {
        return Err(TypeError::invalid(format!(
            "`{RESHAPE_OPERATION_NAME}` requested output sharding changes the varying manual mesh axes"
        )));
    }
    Ok(requested.clone())
}

/// Infers output placement by preserving equal axes and distributing contiguous static split/merge groups.
fn infer_reshape_sharding(input: &ArrayType, output_shape: &Shape, sharding: &Sharding) -> Result<Sharding, TypeError> {
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
    let mut output_sharding_dimensions = vec![ShardingDimension::replicated(); output_shape.rank()];
    for (axis, output_sharding) in output_sharding_dimensions.iter_mut().enumerate().take(input.rank()) {
        if input.dimension(axis) == Dimension::Static(1) && output_shape.dimension(axis) == Dimension::Static(1) {
            *output_sharding = sharding.dimensions()[axis].clone();
        }
    }

    if input_dimensions.iter().map(|(_, size)| size).eq(output_dimensions.iter().map(|(_, size)| size)) {
        for ((input_axis, _), (output_axis, _)) in input_dimensions.iter().zip(&output_dimensions) {
            output_sharding_dimensions[*output_axis] = sharding.dimensions()[*input_axis].clone();
        }
        return rebuild_reshape_sharding(sharding, output_sharding_dimensions);
    }

    if sharding.dimensions().iter().all(|dimension| *dimension == ShardingDimension::Replicated) {
        return rebuild_reshape_sharding(sharding, output_sharding_dimensions);
    }

    if input_dimensions.iter().any(|(_, size)| *size == Dimension::Static(0))
        || output_dimensions.iter().any(|(_, size)| *size == Dimension::Static(0))
    {
        propagate_zero_reshape_sharding(
            &input_dimensions,
            &output_dimensions,
            sharding,
            &mut output_sharding_dimensions,
        )?;
        return rebuild_reshape_sharding(sharding, output_sharding_dimensions);
    }

    let alignment_error =
        || TypeError::invalid(format!("`{RESHAPE_OPERATION_NAME}` could not align reshape dimension groups"));
    let mut input_start = 0usize;
    let mut output_start = 0usize;
    while input_start < input_dimensions.len() || output_start < output_dimensions.len() {
        if input_start == input_dimensions.len() || output_start == output_dimensions.len() {
            return Err(alignment_error());
        }
        // Equal symbolic axes retain their placement independently of adjacent static split/merge groups.
        if input_dimensions[input_start].1 == output_dimensions[output_start].1 {
            output_sharding_dimensions[output_dimensions[output_start].0] =
                sharding.dimensions()[input_dimensions[input_start].0].clone();
            input_start += 1;
            output_start += 1;
            continue;
        }
        let input_group_start = input_start;
        let output_group_start = output_start;
        let mut input_product = static_positive_size(input_dimensions[input_start].1.clone())?;
        let mut output_product = static_positive_size(output_dimensions[output_start].1.clone())?;
        input_start += 1;
        output_start += 1;
        while input_product != output_product {
            if input_product < output_product {
                let (_, size) = input_dimensions.get(input_start).ok_or_else(alignment_error)?;
                input_product =
                    input_product.checked_mul(static_positive_size(size.clone())?).ok_or_else(alignment_error)?;
                input_start += 1;
            } else {
                let (_, size) = output_dimensions.get(output_start).ok_or_else(alignment_error)?;
                output_product =
                    output_product.checked_mul(static_positive_size(size.clone())?).ok_or_else(alignment_error)?;
                output_start += 1;
            }
        }
        propagate_static_reshape_group(
            &input_dimensions[input_group_start..input_start],
            &output_dimensions[output_group_start..output_start],
            sharding,
            &mut output_sharding_dimensions,
        )?;
    }
    rebuild_reshape_sharding(sharding, output_sharding_dimensions)
}

/// Returns the positive static value of `size` for split/merge factorization.
fn static_positive_size(size: Dimension) -> Result<usize, TypeError> {
    match size {
        Dimension::Static(value) if value > 0 => Ok(value),
        Dimension::Static(_) | Dimension::Dynamic(_) => Err(TypeError::invalid(format!(
            "`{RESHAPE_OPERATION_NAME}` requires explicit output sharding for unaligned zero or dynamic dimensions"
        ))),
    }
}

/// Propagates placement around a zero-product reshape without multiplying through zero.
fn propagate_zero_reshape_sharding(
    input_dimensions: &[(usize, Dimension)],
    output_dimensions: &[(usize, Dimension)],
    sharding: &Sharding,
    output_sharding_dimensions: &mut [ShardingDimension],
) -> Result<(), TypeError> {
    let mut prefix = 0usize;
    while input_dimensions.get(prefix).map(|(_, size)| size) == output_dimensions.get(prefix).map(|(_, size)| size) {
        let Some(((input_axis, _), (output_axis, _))) = input_dimensions.get(prefix).zip(output_dimensions.get(prefix))
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
            "`{RESHAPE_OPERATION_NAME}` requires explicit output sharding for an ambiguous zero-sized reshape"
        )));
    }
    if input_dimensions[prefix..input_end].iter().any(|(_, size)| matches!(size, Dimension::Dynamic(_)))
        || output_dimensions[prefix..output_end].iter().any(|(_, size)| matches!(size, Dimension::Dynamic(_)))
    {
        return Err(TypeError::invalid(format!(
            "`{RESHAPE_OPERATION_NAME}` requires explicit output sharding for unaligned dynamic dimensions"
        )));
    }
    Ok(())
}

/// Propagates one positive-static reshape group, distributing contiguous mesh axes over output factors.
fn propagate_static_reshape_group(
    input_group: &[(usize, Dimension)],
    output_group: &[(usize, Dimension)],
    sharding: &Sharding,
    output_sharding_dimensions: &mut [ShardingDimension],
) -> Result<(), TypeError> {
    if input_group.len() == 1 && output_group.len() == 1 {
        output_sharding_dimensions[output_group[0].0] = sharding.dimensions()[input_group[0].0].clone();
        return Ok(());
    }

    let mut mesh_axes = Vec::new();
    let mut saw_replicated = false;
    for (axis, _) in input_group {
        match &sharding.dimensions()[*axis] {
            ShardingDimension::Replicated => saw_replicated = true,
            ShardingDimension::Unconstrained => {
                return Err(TypeError::invalid(format!(
                    "`{RESHAPE_OPERATION_NAME}` requires explicit output sharding for unconstrained dimensions"
                )));
            }
            ShardingDimension::Sharded(axis_names) => {
                if saw_replicated {
                    return Err(TypeError::invalid(format!(
                        "`{RESHAPE_OPERATION_NAME}` cannot preserve non-contiguous sharding across a merge"
                    )));
                }
                mesh_axes.extend(axis_names.iter().cloned());
            }
        }
    }

    let mut mesh_axis_index = 0usize;
    for (output_axis, size) in output_group {
        let mut remaining = static_positive_size(size.clone())?;
        let start = mesh_axis_index;
        while remaining > 1 && mesh_axis_index < mesh_axes.len() {
            let mesh_axis = &mesh_axes[mesh_axis_index];
            let mesh_axis_size = sharding.mesh().axis_size(mesh_axis).ok_or_else(|| {
                TypeError::invalid(format!("`{RESHAPE_OPERATION_NAME}` references unknown mesh axis `{mesh_axis}`"))
            })?;
            if remaining % mesh_axis_size != 0 {
                return Err(TypeError::invalid(format!(
                    "`{RESHAPE_OPERATION_NAME}` cannot distribute sharding across the requested split factors"
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
            "`{RESHAPE_OPERATION_NAME}` cannot distribute all input mesh axes across the output dimensions"
        )));
    }
    Ok(())
}

/// Rebuilds inferred reshape sharding while preserving reduction and manual-axis state.
fn rebuild_reshape_sharding(input: &Sharding, dimensions: Vec<ShardingDimension>) -> Result<Sharding, TypeError> {
    Sharding::new(input.mesh().clone(), dimensions)
        .and_then(|output| output.with_unreduced_axes(input.unreduced_axes().clone()))
        .and_then(|output| output.with_reduced_axes(input.reduced_axes().clone()))
        .and_then(|output| output.with_varying_manual_axes(input.varying_manual_axes().clone()))
        .map_err(|error| TypeError::invalid(error.to_string()))
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::batching::DynamicArrayExtentBatchingPolicy;
    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, DataType, DimensionBounds, DimensionOperation,
        DimensionVariable, Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType, RaggedAxis, Sharding, StridedLayout,
    };
    use crate::contexts::{EagerContext, ProjectedContext, StagingContext};
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::operations::dimensions::dimension_from_scalar::DimensionFromScalarOperation;
    use crate::operations::dimensions::dimension_mul::DimensionMulOperation;
    use crate::operations::{DimensionArithmetic, DimensionSize};
    use crate::parameters::Placeholder;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder, ProgramError, Typed};
    use crate::tracing::Trace;

    use super::*;

    #[test]
    fn test_reshape() {
        let shape = Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]);
        let operation = ReshapeOperation::new(shape.clone());

        // Operation identity and accessors.
        assert_eq!(operation.name(), RESHAPE_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "reshape [shape=[2, 3]]");
        assert_eq!(operation.parameters().output_shape(), &shape);

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
                    error = "`reshape` changes the number of elements",
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

        // The optional dimensions permutation is applied before the row-major reshape.
        assert_eq!(
            input.reshape(ReshapeParameters::new(Shape::new(vec![Dimension::Static(6)])).with_dimensions([-1, -2]),),
            Err(ProgramError::Type(TypeError::invalid("permutation has length 2 but input has rank 1".to_string()))),
        );
        assert_eq!(
            Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap()
                .reshape(ReshapeParameters::new(Shape::new(vec![Dimension::Static(6)])).with_dimensions([-1, -2]),)
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
        // A mapped axis is moved to the leading position before reshaping each batch item.
        let batched_input = Array::matrix(2, 6, (0..12).map(|value| value as f64).collect()).unwrap();
        let batched_output = Array::from_elements::<f64>(
            ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 2.into(), 3.into()])),
            &(0..12).map(|value| value as f64).collect::<Vec<_>>(),
        )
        .unwrap();
        check_operation_batching!(
            @exact,
            operation = ReshapeOperation::new(Shape::new(vec![2.into(), 3.into()])),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), batched_input)],
                outputs = [(@mapped(axis = 0), batched_output)],
            }],
        );
        check_operation_batching!(
            @exact,
            operation = ReshapeOperation::new(Shape::new(vec![2.into(), 3.into()])),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 1), Array::matrix(
                    6,
                    2,
                    vec![0.0, 6.0, 1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 11.0],
                ).unwrap())],
                outputs = [(@mapped(axis = 0), Array::from_elements::<f64>(
                    ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 2.into(), 3.into()])),
                    &(0..12).map(|value| value as f64).collect::<Vec<_>>(),
                ).unwrap())],
            }],
        );
        check_operation_batching!(
            @exact,
            operation = ReshapeOperation::new(
                ReshapeParameters::new(Shape::new(vec![6.into()])).with_dimensions([1, 0]),
            ),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::from_elements::<f64>(
                    ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 2.into(), 3.into()])),
                    &(1..=12).map(|value| value as f64).collect::<Vec<_>>(),
                ).unwrap())],
                outputs = [(@mapped(axis = 0), Array::matrix(
                    2,
                    6,
                    vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 7.0, 10.0, 8.0, 11.0, 9.0, 12.0],
                ).unwrap())],
            }],
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
                message: "`reshape` does not support bounded ragged array inputs".to_string(),
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
                message: "`reshape` with a dynamic mapped extent requires `DynamicReshape` and explicit result-dimension inputs".to_string(),
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
                ReshapeParameters::new(Shape::new(vec![6.into()])).with_dimensions([1, 0]),
            ),
            cases = [{
                inputs = [(@linear(type = ArrayType::new(
                    DataType::F64,
                    Shape::new(vec![2.into(), 3.into()]),
                )))],
                output_cotangents = [Array::vector(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap()],
                input_cotangents = [Array::matrix(2, 3, vec![10.0, 30.0, 50.0, 20.0, 40.0, 60.0]).unwrap()],
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
    fn test_array_type_reshape() {
        // Dynamic dimensions can only be reshaped without explicit dimension inputs when equality follows directly
        // from identical shapes carrying the same symbolic identities. Other runtime relationships require the mixed
        // reshape operation and its explicit result-dimension inputs.
        let static_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(6)]));
        let dynamic_shape = Shape::new(vec![
            Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
            Dimension::Static(3),
        ]);
        let dynamic_type = ArrayType::new(DataType::F64, dynamic_shape.clone());
        assert_eq!(
            dynamic_type.reshape(Shape::new(vec![Dimension::Static(6)])),
            Err(ProgramError::Type(TypeError::invalid(
                "`reshape` requires explicit result-dimension inputs for a dynamic input shape".to_string()
            ))),
        );
        assert_eq!(
            static_type.reshape(dynamic_shape.clone()),
            Err(ProgramError::Type(TypeError::invalid(
                "`reshape` requires explicit result-dimension inputs for a dynamic output shape".to_string()
            ))),
        );
        assert_eq!(
            ReshapeOperation::new(dynamic_shape.clone()).infer_output_types(std::slice::from_ref(&static_type), &[]),
            Err(TypeError::invalid(
                "`reshape` requires explicit result-dimension inputs for a dynamic output shape".to_string(),
            )),
        );
        assert_eq!(
            Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap().reshape(dynamic_shape),
            Err(ProgramError::Type(TypeError::invalid(
                "`reshape` requires explicit result-dimension inputs for a dynamic output shape".to_string()
            ))),
        );

        // Reshaping a dynamically sized type to its own shape short-circuits as the identity.
        assert_eq!(dynamic_type.reshape(dynamic_type.shape().clone()), Ok(dynamic_type.clone()));

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
            Err(ProgramError::Type(TypeError::invalid(
                "`reshape` requires explicit result-dimension inputs for a dynamic output shape".to_string()
            ))),
        );
        assert_eq!(
            zero_dynamic_type.reshape(Shape::new(vec![Dimension::Static(0)])),
            Ok(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0)]))),
        );
        assert_eq!(
            zero_dynamic_type.reshape(
                ReshapeParameters::new(Shape::new(vec![Dimension::Dynamic(trailing.clone()), Dimension::Static(0),]))
                    .with_dimensions([1, 0]),
            ),
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

        // Merging replicated axes preserves an independent unchanged sharded dimension.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Static(8), Dimension::Static(2), Dimension::Static(3)]),
        )
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
        assert_eq!(
            input_type.reshape(Shape::new(vec![Dimension::Static(8), Dimension::Static(6)])),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8), Dimension::Static(6)]))
                .with_sharding(
                    Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                        .unwrap(),
                )
                .unwrap())
        );

        // Splitting a replicated axis likewise preserves an unchanged sharded dimension.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8), Dimension::Static(6)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
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
                .unwrap(),
            )
            .unwrap())
        );

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

        // A sharded dimension can be split when its mesh axes divide a contiguous prefix of the output factors.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert_eq!(
            input_type.reshape(Shape::new(vec![Dimension::Static(2), Dimension::Static(4)])),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]))
                .with_sharding(
                    Sharding::new(
                        input_type.sharding().unwrap().mesh().clone(),
                        vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
                    )
                    .unwrap(),
                )
                .unwrap()),
        );

        // A genuinely merged sharded dimension cannot preserve its placement either.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])]).unwrap(),
            )
            .unwrap();
        assert_eq!(
            input_type.reshape(Shape::new(vec![Dimension::Static(8)])),
            Err(ProgramError::Type(TypeError::invalid(
                "`reshape` cannot preserve non-contiguous sharding across a merge".to_string()
            ))),
        );

        // Many-to-many regrouping is supported when every participating dimension is replicated.
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

        // A compatible merge keeps sharding from the contiguous outer prefix of the merged dimensions.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            input_type.reshape(Shape::new(vec![Dimension::Static(8)])),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
                .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
                .unwrap()),
        );

        // Explicit output sharding can request a valid redistribution that inference would not choose.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let requested =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                .unwrap();
        assert_eq!(
            input_type.reshape(
                ReshapeParameters::new(Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]))
                    .with_output_sharding(requested.clone()),
            ),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]))
                .with_sharding(requested)
                .unwrap()),
        );
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other_requested =
            Sharding::new(other_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        assert_eq!(
            input_type.reshape(
                ReshapeParameters::new(Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]))
                    .with_output_sharding(other_requested),
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "`reshape` requested output sharding uses a different mesh".to_string()
            ))),
        );
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let auto_requested =
            Sharding::new(auto_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap();
        let unsharded_input = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));
        assert_eq!(
            unsharded_input.reshape(
                ReshapeParameters::new(Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]))
                    .with_output_sharding(auto_requested),
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "`reshape` requested output sharding cannot reference auto mesh axes".to_string()
            ))),
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
            Err(ProgramError::Type(TypeError::invalid(
                "`reshape` requires explicit output sharding for an ambiguous zero-sized reshape".to_string()
            ))),
        );
        let zero_requested = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(
            sharded_dynamic_input.reshape(
                ReshapeParameters::new(Shape::new(vec![Dimension::Static(0)]))
                    .with_output_sharding(zero_requested.clone()),
            ),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(0)]))
                .with_sharding(zero_requested)
                .unwrap()),
        );

        // Lifting an explicit per-item sharding moves a manual mapped axis out of the varying set and onto the new
        // physical batch dimension.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let per_item_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
                .unwrap()
                .with_varying_manual_axes(["x"])
                .unwrap();
        assert_eq!(
            lift_output_sharding_for_leading_batch_axis(&per_item_sharding, ShardingDimension::sharded(["x"])),
            Ok(Sharding::new(
                mesh,
                vec![
                    ShardingDimension::sharded(["x"]),
                    ShardingDimension::replicated(),
                    ShardingDimension::replicated(),
                ],
            )
            .unwrap()),
        );
    }

    #[test]
    fn test_array_reshape() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let reshaped = matrix.reshape(Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])).unwrap();
        assert_eq!(reshaped.r#type().into_owned(), ArrayType::new_static(DataType::F64, [3, 2]));
        assert_eq!(reshaped.to_f64s(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(matrix.reshape(Shape::new(vec![Dimension::Static(4)])).is_err());

        // Reshaping preserves logical order independently of the input's physical placement.
        let input_type =
            ArrayType::new_static(DataType::U16, [2, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![8, 2])));
        let matrix = Array::from_elements(input_type, &[1u16, 2, 3, 4, 5, 6]).unwrap();
        let reshaped = matrix.reshape(Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])).unwrap();
        assert_eq!(reshaped.elements::<u16>(), Ok(vec![1, 2, 3, 4, 5, 6]));
        assert_eq!(reshaped.storage_bytes(), [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0]);
    }

    #[test]
    fn test_reshape_parameters_from_sizes() {
        let input = Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]);
        assert_eq!(
            ReshapeParameters::from_sizes(&input, &[3, -1]),
            Ok(ReshapeParameters::new(Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]))),
        );
        assert_eq!(
            ReshapeParameters::from_sizes(&input, &[6]),
            Ok(ReshapeParameters::new(Shape::new(vec![Dimension::Static(6)]))),
        );
        assert_eq!(
            ReshapeParameters::from_sizes(&Shape::new(vec![Dimension::Static(0)]), &[2, -1]),
            Ok(ReshapeParameters::new(Shape::new(vec![Dimension::Static(2), Dimension::Static(0)]))),
        );
        assert_eq!(
            ReshapeParameters::from_sizes(&Shape::new(Vec::new()), &[]),
            Ok(ReshapeParameters::new(Shape::new(Vec::new()))),
        );
    }

    #[test]
    fn test_reshape_parameters_from_sizes_invalid_dimensions() {
        let input = Shape::new(vec![Dimension::Static(6)]);
        assert_eq!(
            ReshapeParameters::from_sizes(&input, &[-1, -1]),
            Err(TypeError::invalid("`reshape` accepts at most one inferred `-1` dimension",))
        );
        assert_eq!(
            ReshapeParameters::from_sizes(&input, &[-2]),
            Err(TypeError::invalid("`reshape` dimensions must be nonnegative or the inferred size `-1`",))
        );
        assert_eq!(
            ReshapeParameters::from_sizes(&input, &[4, -1]),
            Err(TypeError::invalid("`reshape` inferred dimension does not divide the input element count",))
        );
        assert_eq!(
            ReshapeParameters::from_sizes(&input, &[5]),
            Err(TypeError::invalid("`reshape` output element count 5 differs from input element count 6",))
        );
        assert_eq!(
            ReshapeParameters::from_sizes(&input, &[isize::MAX; 3]),
            Err(TypeError::invalid("`reshape` output element count does not fit in `usize`",))
        );
        assert_eq!(
            ReshapeParameters::from_sizes(&Shape::new(vec![Dimension::Static(0)]), &[0, -1]),
            Err(TypeError::invalid("cannot infer a `reshape` dimension when another output dimension is zero",))
        );
        let dynamic = Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
            "size",
            DimensionBounds::new(0, Some(9)).unwrap(),
        ))]);
        assert_eq!(
            ReshapeParameters::from_sizes(&dynamic, &[-1]),
            Err(TypeError::invalid("`reshape` size inference requires a known input element count",))
        );
    }

    #[test]
    fn test_array_reshape_to_sizes() {
        let input = Array::vector(vec![1i32, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(input.reshape_to_sizes(&[3, -1]), Array::matrix(3, 2, vec![1i32, 2, 3, 4, 5, 6]));
        assert_eq!(Array::scalar(7i32).unwrap().reshape_to_sizes(&[-1]), Array::vector(vec![7i32]));
    }

    #[test]
    fn test_array_reshape_with_order() {
        let input = Array::matrix(2, 3, vec![1i32, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(
            input.reshape_with_order(&[3, -1], ReshapeOrder::RowMajor),
            Array::matrix(3, 2, vec![1i32, 2, 3, 4, 5, 6])
        );
        assert_eq!(
            input.reshape_with_order(&[3, -1], ReshapeOrder::ColumnMajor),
            Array::matrix(3, 2, vec![1i32, 5, 4, 3, 2, 6])
        );
        assert_eq!(
            input.reshape_with_order(&[-1], ReshapeOrder::ColumnMajor),
            Array::vector(vec![1i32, 4, 2, 5, 3, 6])
        );
        assert_eq!(
            Array::scalar(7i32).unwrap().reshape_with_order(&[], ReshapeOrder::ColumnMajor),
            Array::scalar(7i32)
        );
        assert_eq!(
            Array::vector(Vec::<i32>::new()).unwrap().reshape_with_order(&[2, -1], ReshapeOrder::ColumnMajor),
            Array::matrix(2, 0, Vec::<i32>::new())
        );
    }

    #[test]
    fn test_array_ravel() {
        let input = Array::matrix(2, 2, vec![1i32, 2, 3, 4]).unwrap();
        assert_eq!(input.ravel(), Array::vector(vec![1i32, 2, 3, 4]));
        assert_eq!(Array::scalar(7i32).unwrap().ravel(), Array::vector(vec![7i32]));
        assert_eq!(Array::matrix(0, 2, Vec::<i32>::new()).unwrap().ravel(), Array::vector(Vec::<i32>::new()));
    }

    #[test]
    fn test_array_flatten() {
        let input = Array::matrix(2, 2, vec![1i32, 2, 3, 4]).unwrap();
        assert_eq!(input.flatten(), Array::vector(vec![1i32, 2, 3, 4]));
    }

    #[test]
    fn test_array_expand_dims() {
        let input = Array::vector(vec![1i32, 2]).unwrap();
        assert_eq!(input.expand_dims(0), Array::matrix(1, 2, vec![1i32, 2]));
        assert_eq!(input.expand_dims(-1), Array::matrix(2, 1, vec![1i32, 2]));
        assert_eq!(Array::scalar(7i32).unwrap().expand_dims(-1), Array::vector(vec![7i32]));
        assert_eq!(input.expand_dims(2).unwrap_err().to_string(), "axis 2 is out of bounds for rank 2");
    }

    #[test]
    fn test_array_squeeze() {
        let input = Array::matrix(2, 1, vec![1i32, 2]).unwrap();
        assert_eq!(input.squeeze(-1), Array::vector(vec![1i32, 2]));
        assert_eq!(input.squeeze(Vec::<usize>::new()), Ok(input.clone()));
        assert!(
            matches!(input.squeeze(0), Err(ProgramError::Type(TypeError::Invalid { message })) if message == "cannot squeeze axis 0 whose size is not one")
        );
        assert_eq!(input.squeeze([1, -1]).unwrap_err().to_string(), "axes contain duplicate axis 1");
    }

    #[test]
    fn test_array_squeeze_all() {
        let input = Array::matrix(1, 1, vec![7i32]).unwrap();
        assert_eq!(input.squeeze_all(), Array::scalar(7i32));
        let vector = Array::vector(vec![1i32, 2]).unwrap();
        assert_eq!(vector.squeeze_all(), Ok(vector));
    }

    #[test]
    fn test_reshape_invalid_output_count() {
        use crate::parameters::Parameter;
        use crate::programs::{BindingRegionDriver, Provenance, ProvenanceScope};

        /// Array wrapper that dispatches through a deliberately malformed context.
        #[derive(Clone, Debug)]
        struct DispatchArray {
            array: Array,
            output_count: usize,
        }

        impl Display for DispatchArray {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "{}", self.array)
            }
        }

        impl Parameter for DispatchArray {}

        impl Typed for DispatchArray {
            type Type = ArrayType;

            fn r#type(&self) -> Cow<'_, ArrayType> {
                self.array.r#type()
            }
        }

        impl Value for DispatchArray {
            type DispatchDomain = InvalidOutputContext;
            type ExecutionDomain = InvalidOutputContext;

            fn dispatch_domain(&self) -> Self::DispatchDomain {
                InvalidOutputContext(self.output_count)
            }

            fn execution_domain(&self) -> Self::ExecutionDomain {
                self.dispatch_domain()
            }
        }

        /// Context that violates the reshape output arity while accepting otherwise valid inputs.
        #[derive(Clone)]
        struct InvalidOutputContext(usize);

        impl Domain for InvalidOutputContext {
            type Type = ArrayType;
            type Value = DispatchArray;
            type Constant = Array;
            type Operation = ReshapeOperation;
        }

        impl Context for InvalidOutputContext {
            fn lift(&self, array: Array) -> Result<DispatchArray, ProgramError> {
                Ok(DispatchArray { array, output_count: self.0 })
            }

            fn bind<O: Into<Self::Operation>, D: BindingRegionDriver<Self::Constant, Self::Operation>>(
                &self,
                _operation: O,
                _driver: D,
                inputs: &[DispatchArray],
            ) -> Result<Vec<DispatchArray>, ProgramError> {
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

        let array = Array::from_elements(ArrayType::new_static(DataType::I32, [1, 2]), &[3_i32, 7]).unwrap();
        let input = InvalidOutputContext(0).lift(array.clone()).unwrap();
        assert!(matches!(
            input.reshape(Shape::new(vec![2.into(), 1.into()])),
            Err(ProgramError::InvalidOutputCount { expected: 1, actual: 0 })
        ));
        let input = InvalidOutputContext(2).lift(array).unwrap();
        assert!(matches!(
            input.reshape(Shape::new(vec![2.into(), 1.into()])),
            Err(ProgramError::InvalidOutputCount { expected: 1, actual: 2 })
        ));
    }

    #[test]
    fn test_dynamic_reshape() {
        let operation = DynamicReshapeOperation::new();
        assert_eq!(operation.name(), RESHAPE_OPERATION_NAME);
        assert_eq!(operation.to_string(), "reshape");
        assert_eq!(operation.dimensions(), None);
        assert_eq!(operation.output_sharding(), None);
        let input_types = [
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4])),
            DimensionValue::constant(4).unwrap().r#type().into_owned().into(),
        ];
        assert_eq!(
            operation.with_input_types(&input_types).unwrap().to_string(),
            "reshape [element_count_proven=true]"
        );
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
        let input_types = vec![
            input.into(),
            DimensionType::new(extent.clone()).into(),
            DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
            DimensionValue::constant(2).unwrap().r#type().into_owned().into(),
        ];
        let operation = DynamicReshapeOperation::new().with_input_types(&input_types).unwrap();
        let output = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent), 2.into(), 2.into()]))
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
        assert_eq!(operation.infer_output_types(&input_types, &[]), Ok(vec![output.into()]));
        assert_eq!(operation.effects().classes(), EffectClasses::NONE);
        assert_eq!(
            DynamicReshapeOperation::new().effects().classes(),
            EffectClasses::single(EffectClass::OrderedAssertion)
        );

        // A proof must not survive reuse with independent dynamic identities.
        let mut unrelated = input_types.clone();
        unrelated[1] =
            DimensionType::new(DimensionVariable::new("other", DimensionBounds::new(1, Some(9)).unwrap())).into();
        let operation = operation.with_output_sharding(None);
        // Explicit replication lets the count-proof diagnostic be reached before sharding inference.
        unrelated[0] =
            ArrayType::new(DataType::F32, <&ArrayType>::try_from(&input_types[0]).unwrap().shape().clone()).into();
        assert_eq!(
            operation.infer_output_types(&unrelated, &[]),
            Err(TypeError::invalid("`reshape` input types do not preserve its element-count proof"))
        );
        assert_eq!(
            reshape_element_counts_equal(
                &Shape::new(vec![0.into(), usize::MAX.into(), 2.into()]),
                &Shape::new(vec![0.into()])
            ),
            Ok(true)
        );
    }

    #[test]
    fn test_dynamic_reshape_interpretation() {
        // A concrete composite value resolves every explicit extent input and reshapes its array member directly.
        let input = ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        let rows = ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap());
        let columns = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        assert_eq!(
            input.dynamic_reshape(&[rows, columns]).unwrap(),
            ArrayIrValue::Array(Array::matrix(2, 3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
        );
        assert_eq!(input.dynamic_reshape_to_sizes(&[3, 2]).unwrap().r#type().to_string(), "f64[3, 2]");
        assert_eq!(input.dynamic_reshape_to_sizes(&[6]).unwrap(), input);

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
                    %5:f64[batch, 6] = reshape [element_count_proven=true] %0 %1 %4
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
    fn test_dynamic_reshape_identity_instantiation() {
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
        assert_eq!(
            program.output_types(),
            vec![
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source), Dimension::Static(4)]),)
                    .into()
            ],
        );

        let target = DimensionVariable::new("target", bounds);
        let target_dimension_type = DimensionType::new(target.clone());
        let target_array_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(target.clone()), Dimension::Static(4)]));
        let instantiated = program
            .with_instantiated_type_identities(&[
                target_array_type.clone().into(),
                target_dimension_type.clone().into(),
            ])
            .unwrap()
            .into_owned();
        assert_eq!(
            instantiated.output_types(),
            vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(target.clone()), Dimension::Static(4)]),
                )
                .into()
            ],
        );

        let mut destination = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let array = destination.add_input(target_array_type.into());
        let extent = destination.add_input(target_dimension_type.into());
        let outputs = destination.splice_program(&instantiated, &[array, extent]).unwrap();
        let [instruction] = destination.instructions() else {
            panic!("expected the imported reshape instruction");
        };
        assert_eq!(instruction.inputs()[..2], [array, extent]);
        assert_eq!(instruction.outputs(), outputs.as_slice());
        assert_eq!(
            destination.atoms()[outputs[0].index()].r#type().as_ref(),
            &ArrayIrType::Array(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(target), Dimension::Static(4)]),
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

        // An unused runtime reshape must still validate its input-dependent element count.
        for proven in [false, true] {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let input_type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4]));
            let extent_type = if proven {
                DimensionValue::constant(4).unwrap().r#type().into_owned()
            } else {
                DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap()))
            };
            let input = builder.add_input(input_type.clone());
            let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
            let operation = if proven {
                DynamicReshapeOperation::new().with_input_types(&[input_type, extent_type.into()]).unwrap()
            } else {
                DynamicReshapeOperation::new()
            };
            builder.add_instruction(operation, Vec::new(), vec![input, extent], None).unwrap();
            let program = builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![input],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
                .into_simplified()
                .unwrap();
            assert_eq!(program.instructions().len(), usize::from(!proven));
        }
    }

    #[test]
    fn test_dynamic_reshape_batching() {
        let input = ArrayIrValue::Array(Array::matrix(2, 6, (0..12).map(|value| value as f64).collect()).unwrap());
        let output = ArrayIrValue::Array(
            Array::from_elements::<f64>(
                ArrayType::new_static(DataType::F64, [2, 2, 3]),
                &(0..12).map(|value| value as f64).collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let (outputs, evidence) = DynamicReshapeOperation::new()
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    ArrayIrBatch::new(input, BatchAxis::new(0)).unwrap(),
                    ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(2).unwrap())),
                    ArrayIrBatch::replicated(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap())),
                ],
            )
            .unwrap()
            .into_parts();
        assert_eq!(outputs, vec![ArrayIrBatch::new(output, BatchAxis::new(0)).unwrap()]);
        assert_eq!(evidence, Vec::<DimensionVariable>::new());
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

        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_types().len(), 2);
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

        let pullback = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback.interpret(vec![ArrayIrValue::Array(
                Array::matrix(2, 3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],).unwrap()
            )]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0,]).unwrap())]),
        );

        // The inverse cannot recover `n` from the `[2, 2*n]` output shape without division. The reshape JVP must
        // therefore retain the original source extent as an explicit residual while it still has the source array.
        let source = DimensionVariable::new("source", DimensionBounds::new(0, Some(9)).unwrap());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(source), Dimension::Static(4)]));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.clone().into());
        let source_extent = builder
            .add_instruction(DimensionSizeOperation::new(&input_type, 0).unwrap(), Vec::new(), vec![input], None)
            .unwrap()[0];
        let two_value = DimensionValue::constant(2).unwrap();
        let two_type = two_value.r#type().into_owned();
        let two = builder.add_constant(ArrayIrValue::Dimension(two_value));
        let source_type = DimensionType::new(input_type.shape().dimensions()[0].variable().unwrap().clone());
        let doubled_extent = builder
            .add_instruction(
                DimensionOperation::Mul(DimensionMulOperation::new(&source_type, &two_type).unwrap()),
                Vec::new(),
                vec![source_extent, two],
                None,
            )
            .unwrap()[0];
        let output = builder
            .add_instruction(DynamicReshapeOperation::new(), Vec::new(), vec![input, two, doubled_extent], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_types().len(), 2);

        // The dual program derives the forward geometry once. The staged linear call receives the primal reshape's own
        // dimension inputs as its leading residuals instead of restaging extent arithmetic for the tangent, so no
        // dimension acquires a second forward definition just because the program was differentiated. The additional
        // geometry read is the transpose residual that the inverse reshape needs, not a duplicated derivation.
        let staged = |name: &str| {
            jvp.instructions()
                .iter()
                .filter(|instruction| instruction.operation().name() == name)
                .collect::<Vec<_>>()
        };
        assert_eq!(staged("dimension_mul").len(), 1);
        assert_eq!(staged("dimension_size").len(), 2);
        let (reshapes, linear_calls) = (staged("reshape"), staged("linear_call"));
        let ([reshape], [linear_call]) = (reshapes.as_slice(), linear_calls.as_slice()) else {
            panic!("expected one staged primal reshape and one staged linear call");
        };
        assert_eq!(linear_call.inputs()[..2], reshape.inputs()[1..]);

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
            "
lambda %0:f64[source, 4] .
let %1:dimension<2> = const 2
    %2:dimension<source ∈ [0, 9)> = dimension_size [axis=0] %0
    %3:dimension<source * 2 ∈ [0, 17)> = dimension_mul %2 %1
    %4:f64[2, source * 2] = reshape %0 %1 %3
    %5:dimension<source ∈ [0, 9)> = dimension_size [axis=0] %0
in (%4, %3, %5)
            "
            .trim(),
        );
        assert_eq!(
            rendered_tangent,
            "
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
            "
            .trim(),
        );
        assert_eq!(linearization.tangent().input_types()[0], ArrayIrType::Array(input_type.tangent().unwrap()));
        assert!(
            linearization
                .tangent()
                .input_types()
                .iter()
                .skip(1)
                .all(|r#type| matches!(r#type, ArrayIrType::Dimension(_)))
        );
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

        // The executable linear boundary remains structural when imported, including both attached regions and every
        // residual edge. Nested forward differentiation likewise treats only the array input as differentiable.
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
        let [imported_call] = imported.instructions() else {
            panic!("expected one imported linear call");
        };
        assert!(matches!(imported_call.operation(), ArrayIrOperation::LinearCall(_)));
        assert_eq!(imported_call.regions().len(), 2);
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

        let mut pullback_inputs =
            vec![ArrayIrValue::Array(Array::matrix(2, 6, (24..36).map(|value| value as f64).collect()).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ArrayIrValue::Array(Array::matrix(3, 4, (24..36).map(|value| value as f64).collect(),).unwrap())]),
        );

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
        assert!(!linearization.primal().to_string().contains("dimension_size"));
        assert!(!linearization.tangent().to_string().contains("dimension_size"));
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
        assert_eq!(linearization.tangent().input_types().len(), 3);
        assert!(
            linearization
                .tangent()
                .input_types()
                .iter()
                .skip(1)
                .all(|r#type| matches!(r#type, ArrayIrType::Dimension(_)))
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
        let output = builder
            .add_instruction(
                DynamicReshapeOperation::new().with_dimensions([1, 0]),
                Vec::new(),
                vec![input, extent, extent],
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
        let linearization = program.linearize().unwrap();

        // Both output axes and both inverse axes use the same SSA extent. Partial evaluation carries it once even
        // though the linear call consumes it in multiple input positions.
        assert_eq!(linearization.residual_count(), 1);
        assert_eq!(linearization.primal().to_string().matches("dimension_size").count(), 1);
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
}
