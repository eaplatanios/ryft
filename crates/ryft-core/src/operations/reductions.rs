//! Operations that combine array elements along selected axes. Every reduction is defined by the [`ReduceOperation`]
//! type together with the [`Reduce`] value capability trait, whose functions apply it to eager [`Array`]s and traced
//! values alike, so the same code executes immediately or records into a program depending on the value it runs on. A
//! [`ReductionKind`] selects the computation:
//!
//!   - **Numeric Reductions:** [`Sum`](ReductionKind::Sum) adds the reduced elements, and
//!     [`Mean`](ReductionKind::Mean) additionally divides that sum by the number of reduced elements.
//!   - **Extrema:** [`Max`](ReductionKind::Max) and [`Min`](ReductionKind::Min) select the largest and smallest
//!     reduced elements, propagating NaNs, ordering negative zero below positive zero, and comparing complex
//!     elements by their real parts first and their imaginary parts second.
//!   - **Logarithmic Sums of Exponentials:** [`LogSumExp`](ReductionKind::LogSumExp) computes `log(sum(exp(x)))`
//!     without overflowing for large finite inputs.
//!   - **Boolean Reductions:** [`Any`](ReductionKind::Any) and [`All`](ReductionKind::All) compute the disjunction and
//!     conjunction of Boolean elements.
//!
//! The reduced axes are removed from the output shape, and the remaining axes keep their order, as for StableHLO's
//! [`reduce`](https://openxla.org/stablehlo/spec#reduce). Sums, extrema, and Boolean reductions start from the identity
//! of their combiner (e.g., `0` for sums and the smallest value of the element type for maxima), so reducing an empty
//! axis produces that identity. Bounded ragged-axis reductions support sums and logarithmic sums of exponentials;
//! their padding is replaced by zero or negative infinity, respectively. Other kinds reject ragged reduced axes.
//! Narrow floating-point sums, means, and logarithmic sums of exponentials compute in `f32` before converting the
//! output back to the input data type. Empty floating-point and complex means compute NaNs before output conversion.
//!
//! Floating-point and complex sums and means are linear: their transposes broadcast the cotangent over the reduced
//! axes, dividing it by the number of reduced elements for means. Extrema route the tangent through selected elements,
//! splitting it evenly between ties. Extrema and logarithmic sums of exponentials are differentiable but nonlinear;
//! logarithmic sums currently require statically shaped inputs for differentiation. Boolean reductions are not
//! differentiable.
//!
//! # Example
//!
//! ```rust
//! # use ryft_core::{Array, ProgramError, Reduce, ReductionKind};
//! # fn main() -> Result<(), ProgramError> {
//! let input = Array::matrix(2, 3, vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
//! let output = input.reduce(&[1], ReductionKind::Sum)?;
//! assert_eq!(output.elements::<f32>()?, vec![6.0, 15.0]);
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeSet;
use std::fmt::Display;
use std::sync::Arc;

use half::{bf16, f16};
use num_complex::Complex;

use crate::arrays::{
    Array, ArrayAddressing, ArrayBatch, ArrayBatchingPolicy, ArrayElement, ArrayIrType, ArrayType, DataType, Dimension,
    DimensionOperation, DimensionType, DimensionValue, FloatingPointArrayElement, LinearResiduals, MeshAxisType,
    NumericArrayElement, RaggedArrayExtentBatchingPolicy, RaggedMaskIdentity, Shape, Sharding, ShardingDimension,
    f4e2m1fn, f6e2m3fn, f6e3m2fn, f8e3m4, f8e4m3, f8e4m3b11fnuz, f8e4m3fn, f8e4m3fnuz, f8e5m2, f8e5m2fnuz, f8e8m0fnu,
    i1, i2, i4, u1, u2, u4,
};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, ProjectedContext};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver, DifferentiationDual,
    DifferentiationError, DifferentiationPolicy, ElementwiseDerivativeAlignment, MemberDifferentiableOperation,
    jvp_projected_operation,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, dispatch_on_array_element_type, impl_differentiable_operation};
use crate::operations::arithmetic::{Div, DivOperation, Mul, MulOperation, Sub};
use crate::operations::collectives::parallel_vary::ParallelVaryOperation;
use crate::operations::comparisons::{Compare, CompareOperation, ComparisonDirection};
use crate::operations::constants::constant::{ConstantOperation, DimensionConstant};
use crate::operations::constants::fill::Fill;
use crate::operations::constants::zero_like::ZeroLike;
use crate::operations::control_flow::select::Select;
use crate::operations::differentiation::linear_call::LinearCallOperation;
use crate::operations::dimensions::dimension_mul::DimensionMulOperation;
use crate::operations::dimensions::dimension_size::DimensionSizeOperation;
use crate::operations::dimensions::dimension_to_scalar::DimensionToScalarOperation;
use crate::operations::exponential::Exp;
use crate::operations::manipulation::broadcasting::{
    Broadcast, BroadcastOperation, DynamicBroadcast, DynamicBroadcastOperation,
};
use crate::operations::manipulation::conversions::{ConvertElementType, ConvertElementTypeOperation};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, OperationProjection, OperationProvider, ProgramError, RegionInterface,
    TypeError, Typed, Value, ValueProjection,
};

/// Name of [`ReduceOperation`]. The reduction's [`ReductionKind`] is rendered as its `kind` attribute.
pub const REDUCE_OPERATION_NAME: &str = "reduce";

/// Kind of reduction performed by a [`ReduceOperation`]. Reductions collapse selected axes while preserving the order
/// of the remaining axes. Sums, extrema, and Boolean reductions combine elements directly while means and logarithmic
/// sums of exponentials additionally normalize or transform those elements. Backends may implement reductions of a
/// specific kind by decomposing them into several primitive operations.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReductionKind {
    /// Numeric sum reduction. The identity is `0` and the combiner is addition. Floating-point inputs narrower
    /// than `f32` accumulate in `f32`. Also, formats without a zero representation cannot supply the identity.
    Sum,

    /// Numeric mean reduction defined as a [`Sum`](Self::Sum) divided by the product of reduced extents. Narrow
    /// floating-point inputs accumulate and divide in `f32` before conversion back to the input data type. Empty
    /// floating-point and complex reductions compute NaNs, subject to the output format's conversion rules. Integer
    /// inputs retain their data type, wrap the sum and element count to that type, and use truncating division;
    /// empty integer reductions produce zero. An integer divisor that wraps to zero or an overflowing signed quotient
    /// returns an error during eager evaluation.
    Mean,

    /// Numerically stable logarithm of a sum of exponentials, `log(sum(exp(x)))`. The computation guards the maximum
    /// shift before exponentiating, avoiding overflow for large finite inputs:
    ///
    /// ```text
    /// m      = reduce_max(x)                      // with a -∞ initial value
    /// safe_m = select(isfinite(m), m, 0)
    /// output = log(reduce_sum(exp(x - safe_m))) + safe_m
    /// ```
    ///
    /// The `safe_m` substitution is what the guard buys. Shifting by a raw maximum of `-∞` (the identity of a maximum,
    /// and therefore the output for an all-`-∞` slice or an empty reduction) would compute `-∞ - -∞ = NaN`.
    /// Substituting zero there leaves `log(0) + 0 = -∞`, which is the correct value of an empty or all-zero
    /// sum of exponentials. A maximum of `+∞` is guarded the same way, and a NaN input propagates as usual.
    ///
    /// Only real floating-point formats that represent negative infinity are supported. The ragged batching rule fills
    /// padding with negative infinity so its exponential stays zero after subtraction of any finite maximum. A finite
    /// sentinel cannot provide that guarantee, even when it is an identity of rounded pairwise `log_add_exp`:
    /// subtracting a nearby maximum makes padded entries contribute to the inner sum. This reduction is the unweighted,
    /// unmasked subset of [`jax.nn.logsumexp`](https://docs.jax.dev/en/latest/_autosummary/jax.nn.logsumexp.html);
    /// weights, masks, sign outputs, and complex inputs are not supported.
    LogSumExp,

    /// Maximum reduction. Boolean inputs use disjunction, real numeric inputs propagate NaNs and order negative zero
    /// below positive zero, and complex inputs compare lexicographically by `(real, imaginary)`. The identity is the
    /// data type's smallest value under that ordering.
    Max,

    /// Minimum reduction. Boolean inputs use conjunction, real numeric inputs propagate NaNs and order negative zero
    /// below positive zero, and complex inputs compare lexicographically by `(real, imaginary)`. The identity is the
    /// data type's largest value under that ordering.
    Min,

    /// Boolean disjunction reduction. The identity is `false` and the combiner is the logical _or_ operation.
    /// Inputs must have [`DataType::Boolean`].
    Any,

    /// Boolean conjunction reduction. The identity is `true` and the combiner is the logical _and_ operation.
    /// Inputs must have [`DataType::Boolean`].
    All,
}

impl ReductionKind {
    /// Returns the name of this [`ReductionKind`], which program renderings and diagnostics use as the `kind` attribute
    /// of a [`ReduceOperation`] (e.g., `reduce [kind=sum, axes=[0]]`).
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Mean => "mean",
            Self::LogSumExp => "log_sum_exp",
            Self::Max => "max",
            Self::Min => "min",
            Self::Any => "any",
            Self::All => "all",
        }
    }
}

impl Display for ReductionKind {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.name())
    }
}

/// Represents an N-dimensional axis-collapsing reduction. [`ReduceOperation`] collapses the input array along `axes`
/// using the reduction described by [`kind`](Self::kind). The output rank is the input rank minus the number of reduced
/// axes; non-reduced axes keep their relative order.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReduceOperation {
    /// Refer to the documentation of [`Self::axes`].
    axes: Vec<usize>,

    /// Refer to the documentation of [`Self::kind`].
    kind: ReductionKind,

    /// Refer to the documentation of [`Self::with_output_sharding`].
    output_sharding: Option<Sharding>,
}

impl ReduceOperation {
    /// Creates a new [`ReduceOperation`] reducing along `axes` with the supplied `kind`. The input shape is not part
    /// of the operation payload. Instead, it is recoverable from the staged input types wherever a rule needs it.
    #[inline]
    pub fn new(axes: Vec<usize>, kind: ReductionKind) -> Self {
        Self { axes, kind, output_sharding: None }
    }

    /// Returns this [`ReduceOperation`] with the requested output sharding, or without a requested output sharding
    /// when `output_sharding` is [`None`]. Only [`ReductionKind::Sum`] reductions support a requested output sharding.
    ///
    /// The rest of the request depends on the input type, so type inference validates it. Specifically, it must match
    /// the output rank and the input mesh and cannot reference automatic mesh axes. Requested unreduced axes defer
    /// cross-device sums and must be explicit axes that shard a reduced dimension or are already unreduced on the
    /// input. Reduced axes cannot be requested, because inference preserves the input's reduced state and manual
    /// variation independently of placement, and an explicit manual-variation request must match the input.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if `output_sharding` requests a sharding and this reduction's kind is not
    /// [`ReductionKind::Sum`].
    pub fn with_output_sharding<S: Into<Option<Sharding>>>(mut self, output_sharding: S) -> Result<Self, TypeError> {
        let output_sharding = output_sharding.into();
        if output_sharding.is_some() && self.kind != ReductionKind::Sum {
            return Err(TypeError::invalid(format!(
                "`{}` with kind `{}` does not support a requested output sharding (only kind `sum` does)",
                REDUCE_OPERATION_NAME, self.kind,
            )));
        }
        self.output_sharding = output_sharding;
        Ok(self)
    }

    /// Returns the axes reduced by this [`ReduceOperation`].
    #[inline]
    pub fn axes(&self) -> &[usize] {
        self.axes.as_slice()
    }

    /// Returns the kind of reduction used by this [`ReduceOperation`].
    #[inline]
    pub fn kind(&self) -> ReductionKind {
        self.kind
    }

    /// Returns the requested output sharding, if any, for this [`ReduceOperation`].
    #[inline]
    pub fn output_sharding(&self) -> Option<&Sharding> {
        self.output_sharding.as_ref()
    }
}

impl Display for ReduceOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ReduceOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        REDUCE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        let input = &input_types[0];
        let output = reduce_abstract(input, self.axes.as_slice(), self.kind)?;
        let Some(output_sharding) = &self.output_sharding else {
            return Ok(vec![output]);
        };

        // Only sums carry a requested output placement (refer to `with_output_sharding`), and the request must agree
        // with the inferred output in rank and with the input in mesh. Reduced state and manual variation belong to
        // the input's semantics, so the request can neither ask for reduced axes nor change manual variation.
        if output_sharding.rank() != output.rank() {
            return Err(TypeError::invalid(format!(
                "`{}` output sharding rank ({}) does not match the output rank ({})",
                REDUCE_OPERATION_NAME,
                output_sharding.rank(),
                output.rank(),
            )));
        }

        if let Some(input_sharding) = input.sharding()
            && output_sharding.mesh() != input_sharding.mesh()
        {
            return Err(TypeError::invalid(format!(
                "`{REDUCE_OPERATION_NAME}` output sharding must use the same mesh as the input",
            )));
        }

        if !output_sharding.reduced_axes().is_empty() {
            return Err(TypeError::invalid(format!(
                "`{REDUCE_OPERATION_NAME}` output sharding cannot request reduced axes",
            )));
        }

        if !output_sharding.varying_manual_axes().is_empty()
            && input
                .sharding()
                .is_none_or(|sharding| output_sharding.varying_manual_axes() != sharding.varying_manual_axes())
        {
            return Err(TypeError::invalid(format!(
                "`{REDUCE_OPERATION_NAME}` output sharding cannot change manual variation",
            )));
        }

        // Automatic mesh axes are placed by the partitioner rather than by the program, so a placement request cannot
        // name them, whether as an unreduced axis or as an axis that shards an output dimension.
        let mut referenced_axes = output_sharding.unreduced_axes().iter().collect::<Vec<_>>();
        for dimension in output_sharding.dimensions() {
            if let ShardingDimension::Sharded(axis_names) = dimension {
                referenced_axes.extend(axis_names);
            }
        }

        if referenced_axes
            .iter()
            .any(|name| output_sharding.mesh().axis_type(name) == Some(MeshAxisType::Auto))
        {
            return Err(TypeError::invalid(format!(
                "`{REDUCE_OPERATION_NAME}` output sharding cannot reference automatic mesh axes",
            )));
        }

        // Requested unreduced axes defer cross-device sums, so each of them must name an axis whose sum is actually
        // pending (i.e., an explicit axis that shards one of the summed-over dimensions, or an axis over which the
        // input is already unreduced).
        if !output_sharding.unreduced_axes().is_empty() {
            let mut reducible_axes = BTreeSet::new();
            if let Some(input_sharding) = input.sharding() {
                for axis in self.axes.as_slice() {
                    if let ShardingDimension::Sharded(axis_names) = &input_sharding.dimensions()[*axis] {
                        reducible_axes.extend(
                            axis_names
                                .iter()
                                .filter(|name| input_sharding.mesh().axis_type(name) == Some(MeshAxisType::Explicit))
                                .map(String::as_str),
                        );
                    }
                }
                reducible_axes.extend(input_sharding.unreduced_axes().iter().map(String::as_str));
            }
            if !output_sharding.unreduced_axes().iter().all(|name| reducible_axes.contains(name.as_str())) {
                return Err(TypeError::invalid(format!(
                    "`{REDUCE_OPERATION_NAME}` output sharding unreduced axes must be among the explicit axes \
                     sharding the reduced dimensions or the input's unreduced axes",
                )));
            }
        }

        // A placement request cannot discharge manual variation or manufacture reduction state.
        let output_sharding = output_sharding
            .clone()
            .with_reduced_axes(input.sharding().into_iter().flat_map(|sharding| sharding.reduced_axes()).cloned())
            .and_then(|sharding| {
                sharding.with_varying_manual_axes(
                    input.sharding().into_iter().flat_map(|sharding| sharding.varying_manual_axes()).cloned(),
                )
            })
            .map_err(|error| TypeError::invalid(error.to_string()))?;

        Ok(vec![output.with_sharding(output_sharding).map_err(|error| TypeError::invalid(error.to_string()))?])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("kind", self.kind)?;
            operation.field("axes", format_args!("{:?}", self.axes))?;
            if let Some(output_sharding) = &self.output_sharding {
                operation.field("output_sharding", output_sharding)?;
            }
            Ok(())
        })
    }
}

impl<D: Domain<Type = ArrayType, Value: Reduce>> InterpretableOperation<D> for ReduceOperation {
    fn interpret<I: InterpretationDriver<D>>(
        &self,
        _context: &D,
        _driver: &I,
        inputs: &[D::Value],
    ) -> Result<Vec<D::Value>, ProgramError> {
        // The requested output sharding flows through `Reduce::reduce_sum` so that interpretation over staging values
        // (e.g., during program batching) preserves it. Only sums can carry a requested output sharding, and so every
        // other kind reduces through `Reduce::reduce`.
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![match self.kind {
            ReductionKind::Sum => inputs[0].reduce_sum(self.axes.as_slice(), self.output_sharding.clone())?,
            kind => inputs[0].reduce(self.axes.as_slice(), kind)?,
        }])
    }
}

impl<C: Context<Type = ArrayType, Operation: From<ReduceOperation>>> PartiallyEvaluatableOperation<C>
    for ReduceOperation
{
}

impl<C: Context<Type = ArrayType>, P: RaggedArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>>
    for ReduceOperation
where
    ReduceOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        // The reduced axes are expressed in the per-item coordinate system, so the rule lifts them past the inserted
        // batch dimension and re-interprets the lifted reduction over the physical batched value, with a requested
        // output sharding gaining the mapped axis's sharding at the new output batch axis position.
        //
        // Reducing a bounded ragged axis away is the one array rule that legitimately consumes an input's per-item
        // extents: `RaggedArrayExtentBatchingPolicy::mask_reduction_input` first replaces the padding along that axis
        // with the reduction's identity, so the output no longer depends on those extents. The rule reports each such
        // `DimensionVariable` as its `BatchedOutputs` evidence, which is how the carrier-invariant validation boundary
        // tells a deliberate consumption apart from a silently dropped extent.
        check_count!("input", inputs, 1, ProgramError);
        let Some(batch_axis) = inputs[0].batch_axis_position() else {
            return Ok(self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()])?.into());
        };

        // The per-item axes cannot name the inserted batch dimension, so every axis at or after the batch axis shifts
        // past it, including an axis at the batch axis position itself. The output batch axis moves down by the number
        // of reduced axes before it, because the output drops those axes.
        let lifted_axes = self.axes.iter().map(|&axis| if axis < batch_axis { axis } else { axis + 1 }).collect();
        let output_axis = batch_axis - self.axes.iter().filter(|&&axis| axis < batch_axis).count();

        // A requested output sharding gains the mapped axis's sharding at the new output batch axis.
        let output_sharding = match &self.output_sharding {
            Some(output_sharding) => {
                Some(output_sharding.batched(output_axis, ArrayBatch::sharding_for_inputs(inputs)?)?)
            }
            None => None,
        };
        let operation = ReduceOperation::new(lifted_axes, self.kind).with_output_sharding(output_sharding)?;

        let input = &inputs[0];
        let reduced_axes = operation.axes();

        // The consumed extents are collected from the unmasked input, because masking rewrites the payload while
        // leaving in place the ragged metadata that the validation boundary is told about.
        let consumed_ragged_dimensions = input
            .ragged_axes()
            .iter()
            .filter(|ragged_axis| reduced_axes.contains(&ragged_axis.axis()))
            .map(|ragged_axis| ragged_axis.dimension().clone())
            .collect::<Vec<_>>();

        let masked = match self.kind {
            ReductionKind::LogSumExp => {
                P::mask_identity_input(context, input, reduced_axes, RaggedMaskIdentity::Lowest)?
            }
            _ => P::mask_reduction_input(context, input, reduced_axes, self.kind)?,
        };

        // Ragged axes outside the reduced axes survive onto the output.
        let remaining_ragged_axes = masked
            .ragged_axes()
            .iter()
            .cloned()
            .filter_map(|ragged_axis| ragged_axis.reduced(reduced_axes))
            .collect::<Vec<_>>();
        let output_batch_axis = BatchAxis::from_position(output_axis);
        let mut outputs = operation.interpret_with_batch_axes(
            context,
            std::slice::from_ref(&masked),
            std::slice::from_ref(&output_batch_axis),
        )?;
        check_count!("output", outputs, 1, ProgramError);
        let output = ArrayBatch::new(outputs.remove(0).into_value(), output_batch_axis)?
            .with_ragged_axes(remaining_ragged_axes)?;
        Ok(BatchedOutputs::new(vec![output], consumed_ragged_dimensions))
    }
}

impl_differentiable_operation! {
    ReduceOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Value: Reduce
            + Exp
            + Sub
            + Broadcast
            + Compare<C::Value>
            + ConvertElementType
            + Div
            + ElementwiseDerivativeAlignment<ArrayType>
            + Mul
            + Select
            + ZeroLike,
        C::Operation: From<ReduceOperation>
            + From<BroadcastOperation>
            + From<CompareOperation<ArrayType>>
            + From<DivOperation<ArrayType>>
            + From<MulOperation<ArrayType>>,
    {
        |operation, context, _driver, inputs| {
            // The additive reductions (i.e., `Sum` and `Mean`) are linear in the input, so the tangent is the same
            // reduction applied to the input tangent. `Max` and `Min` route their tangent through a primal-domain
            // argmax mask: the tangent of `reduce_max(x)` along the reduced axes is `reduce_sum(mask * Δx)`, where
            // `mask` equals `1` exactly at the per-reduction extremal positions (ties split evenly, matching the JAX
            // convention). The mask is staged capture-free as ordinary primal operations (a `compare` of the input
            // primal against the broadcast-back reduced value, followed by an ordinary `mul` against the input tangent)
            // so no residual factor is captured. `Any` and `All` are Boolean reductions with no tangent and are
            // rejected with `ProgramError::UnsupportedOperation`. The shared all-zero fast path handles a zero input
            // tangent before this rule is consulted, so the input tangent reaching every supported case is live.
            check_count!("input", inputs, 1, ProgramError);
            match operation.kind() {
                ReductionKind::Sum | ReductionKind::Mean => {
                    // Only sums can carry a requested output sharding, and the tangent sum must request the same one
                    // as the primal sum so that differentiation does not change how the output is distributed.
                    let reduce = |value: &C::Value| match operation.kind() {
                        ReductionKind::Sum => value.reduce_sum(operation.axes(), operation.output_sharding().cloned()),
                        kind => value.reduce(operation.axes(), kind),
                    };
                    let primal = reduce(inputs[0].primal())?;
                    let tangent = match inputs[0].tangent() {
                        MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                        MaybeZero::Value(tangent) => MaybeZero::Value(reduce(tangent)?),
                    };
                    Ok(vec![DifferentiationDual::new(primal, tangent)?])
                }
                ReductionKind::LogSumExp => {
                    // Differentiate through the normalized exponential weights. All-negative-infinity slices retain
                    // their undefined (i.e., NaN) derivative instead of concealing it with a special-case weight.
                    let primal_input = inputs[0].primal();
                    let primal = primal_input.reduce(operation.axes(), ReductionKind::LogSumExp)?;
                    let tangent = match inputs[0].tangent() {
                        MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                        MaybeZero::Value(input_tangent) => {
                            let primal_input = context.primal_to_tangent(primal_input.clone())?;
                            let tangent_data_type = input_tangent.r#type().data_type();
                            let working_data_type = if matches!(tangent_data_type, DataType::F32 | DataType::F64) {
                                tangent_data_type
                            } else {
                                DataType::F32
                            };

                            // Keep normalization and the weighted sum widened, just like the primal reduction.
                            let primal_input = primal_input.convert_element_type(working_data_type)?;
                            let input_tangent = input_tangent.convert_element_type(working_data_type)?;
                            let input_type = primal_input.r#type().into_owned();
                            let output_axes = output_to_input_axis_map(input_type.rank(), operation.axes.as_slice());

                            // Normalize before adding the maximum back as the rounded logarithmic output can lose the
                            // normalization term entirely when the inputs share a large finite offset.
                            let maximum = primal_input.reduce(operation.axes(), ReductionKind::Max)?;

                            // Non-finite maxima cannot be used as shifts. Preserve zero weights on finite inputs
                            // next to positive infinity, while the infinite inputs retain undefined derivatives.
                            let zero = maximum.zero_like()?;
                            let finite = maximum.sub(&maximum)?.equal(&zero)?;
                            let maximum = C::Value::select(&finite, &maximum, &zero)?;
                            let maximum = maximum.broadcast(input_type.clone(), output_axes.as_slice())?;
                            let exponentials = primal_input.sub(&maximum)?.exp()?;
                            let denominator = exponentials.reduce(operation.axes(), ReductionKind::Sum)?;
                            let denominator = denominator.broadcast(input_type, output_axes.as_slice())?;
                            let weights = exponentials.div(&denominator)?;
                            let weights = weights.align_tangent(input_tangent.r#type().as_ref(), &input_tangent)?;
                            let weighted = weights.mul(&input_tangent)?;
                            MaybeZero::Value(
                                weighted.reduce(operation.axes.as_slice(), ReductionKind::Sum)?
                                    .convert_element_type(tangent_data_type)?,
                            )
                        }
                    };
                    Ok(vec![DifferentiationDual::new(primal, tangent)?])
                }
                kind @ (ReductionKind::Max | ReductionKind::Min) => {
                    // Stage the argmax mask from the input primal capture-free: `compare` the input primal against the
                    // broadcast-back reduced value (an ordinary `compare`/`broadcast`), convert it to the tangent type,
                    // normalize it by the number of ties, and route the input tangent through that normalized mask.
                    let primal_input = inputs[0].primal();
                    let primal = primal_input.reduce(operation.axes(), kind)?;
                    let input_type = primal_input.r#type().into_owned();
                    let output_axes = output_to_input_axis_map(input_type.rank(), operation.axes());
                    let broadcast_primal = primal.broadcast(input_type, output_axes.as_slice())?;
                    let mask = primal_input.compare(&broadcast_primal, ComparisonDirection::Equal)?;
                    let tangent = match inputs[0].tangent() {
                        MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                        MaybeZero::Value(input_tangent) => {
                            let numeric_mask = context
                                .primal_to_tangent(mask.clone())?
                                .align_tangent(input_tangent.r#type().as_ref(), input_tangent)?;
                            let tie_count = numeric_mask.clone().reduce(operation.axes(), ReductionKind::Sum)?;
                            let masked_tangent = numeric_mask.mul(input_tangent)?;
                            MaybeZero::Value(
                                masked_tangent.reduce(operation.axes(), ReductionKind::Sum)?.div(&tie_count)?,
                            )
                        }
                    };
                    Ok(vec![DifferentiationDual::new(primal, tangent)?])
                }
                kind => Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "array operation `reduce with kind {kind:?}` is not supported by the forward-mode \
                         linearization slice; any and all are not differentiable",
                    ),
                }
                .into()),
            }
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: From<BroadcastOperation> + From<ConstantOperation<Array>> + From<MulOperation<ArrayType>>
            + OperationProvider<ArrayType, ParallelVaryOperation, Operation = O>
            + OperationProvider<ArrayType, BroadcastOperation, Operation = O>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // `Sum` transposes by broadcasting the cotangent over the reduced axes. `Mean` also divides by the reduced
            // element count. Non-linear reductions instead differentiate through the linear operations staged by their
            // JVP. A runtime-sized reduced axis requires linearization to retain its extent as a first-class residual.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 1, DifferentiationError);
            if operation.kind == ReductionKind::LogSumExp {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!("operation `{}` is not transposable", operation.name()),
                }.into());
            }

            let input_type = inputs[0].r#type();
            let input_shape = input_type.shape();
            match &outputs[0] {
                MaybeZero::Zero(_) => Ok(()),
                MaybeZero::Value(cotangent) => match operation.kind {
                    ReductionKind::Sum | ReductionKind::Mean => {
                        // Replicating the cotangent back over a reduced axis requires that axis's extent, which a
                        // directly transposed program cannot observe as it holds no primal value that carries it.
                        if let Some(axis) = operation
                            .axes
                            .iter()
                            .find(|axis| matches!(input_shape.dimension(**axis), Dimension::Dynamic(_)))
                        {
                            return Err(ProgramError::UnsupportedOperation {
                                message: format!(
                                    "direct transposition of `{}` with kind `{}` over reduced axis {} of {} requires \
                                     linearization so that the runtime extent can be retained as a residual",
                                    operation.name(),
                                    operation.kind(),
                                    axis,
                                    input_shape,
                                ),
                            }
                            .into());
                        }

                        if !accumulators[0].is_needed() {
                            return Ok(());
                        }

                        let output_type = input_type.cotangent()?;
                        let output_axes = output_to_input_axis_map(input_shape.rank(), &operation.axes);
                        let broadcasted = cotangent.broadcast(output_type, output_axes.as_slice())?;
                        let cotangent_input = match operation.kind {
                            ReductionKind::Sum => broadcasted,
                            ReductionKind::Mean => {
                                // The check above rejected every runtime-sized reduced axis, so each reduced extent
                                // is statically known here.
                                let reduced_extents = operation
                                    .axes
                                    .iter()
                                    .map(|axis| input_shape.dimension(*axis).value().unwrap())
                                    .collect::<Vec<_>>();
                                let element_count = if reduced_extents.contains(&0) {
                                    0
                                } else {
                                    reduced_extents.iter().try_fold(1usize, |count, extent| {
                                        count.checked_mul(*extent).ok_or_else(|| {
                                            TypeError::invalid(format!(
                                                "mean transpose reduced element count overflows `usize` for \
                                                 input shape `{input_shape}`",
                                            ))
                                        })
                                    })?
                                };
                                let inverse_count = 1.0 / element_count as f64;

                                // Stage a rank-zero literal holding `1 / N` and rely on implicit rank-zero broadcasting
                                // in the subsequent multiplication to scale the broadcast-back cotangent to the input
                                // shape.
                                let factor_type = ArrayType::new(cotangent.r#type().data_type(), Shape::scalar());
                                let factor = context.fill(&factor_type, inverse_count)?;
                                factor.mul(&broadcasted)?
                            }
                            _ => unreachable!("outer match handled the only two supported kinds"),
                        };
                        accumulators[0].accumulate(context, MaybeZero::Value(cotangent_input))
                    }
                    other => Err(
                        TypeError::invalid(format!("`{other}` reduction is not directly transposable")).into(),
                    ),
                },
            }
        }
    },
}

impl<C> MemberDifferentiableOperation<C> for ReduceOperation
where
    C: Context<Type = ArrayIrType>,
    C::Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation: From<DynamicBroadcastOperation>
        + From<DimensionSizeOperation>
        + From<DimensionToScalarOperation>
        + From<LinearCallOperation<ArrayIrType>>
        + From<ConstantOperation<DimensionValue>>
        + OperationProjection<
            ArrayType,
            Projected: DifferentiableOperation<ProjectedContext<C, ArrayType>>
                           + From<CompareOperation<ArrayType>>
                           + From<ConvertElementTypeOperation<ArrayType>>
                           + From<DivOperation<ArrayType>>
                           + From<MulOperation<ArrayType>>
                           + From<ReduceOperation>,
        > + OperationProjection<DimensionType, Projected = DimensionOperation<DimensionValue>>,
{
    fn jvp_in_parent<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // Fully static reductions delegate to the homogeneous projected rule. Dynamically shaped numeric reductions
        // retain their exact input extents as ordinary residual values so their transpose can broadcast cotangents
        // back to the runtime input shape. Logarithmic sums also delegate, because their projected rule broadcasts
        // to a static input type and supporting runtime-shaped softmax weights would need its own retained-shape
        // linearization. Boolean reductions delegate so that they report the projected rule's error.
        check_count!("input", inputs, 1, ProgramError);
        let input_type = <&ArrayType>::try_from(inputs[0].primal().r#type().as_ref())?.clone();
        if input_type.shape().dimensions().iter().all(|dimension| matches!(dimension, Dimension::Static(_)))
            || matches!(self.kind(), ReductionKind::LogSumExp | ReductionKind::Any | ReductionKind::All)
        {
            let operation = <C::Operation as OperationProjection<ArrayType>>::Projected::from(self.clone());
            return jvp_projected_operation(context, &operation, inputs);
        }

        let output = bind_array(context.primal(), self.clone(), std::slice::from_ref(inputs[0].primal()))?;
        let tangent_inputs = context.dual_primal_to_tangent(inputs)?;
        let input = &tangent_inputs[0];
        let input_tangent = match input.tangent() {
            MaybeZero::Zero(_) => {
                let tangent = MaybeZero::Zero(output.r#type().tangent()?);
                return Ok(vec![DifferentiationDual::new(output, tangent)?]);
            }
            MaybeZero::Value(input_tangent) => input_tangent.clone(),
        };

        let tangent_context = context.tangent();
        let mut residuals = LinearResiduals::new();
        let input_shape = residuals.retain_shape(tangent_context, input.primal())?;
        let output_axes = output_to_input_axis_map(input_type.rank(), self.axes());
        let cotangent_type = input_type.cotangent()?;
        let axes = self.axes().to_vec();
        let tangent = if matches!(self.kind(), ReductionKind::Max | ReductionKind::Min) {
            // Stage the extremum mask and its tie count in the primal trace, just like the static rule does.
            // The forward map sums the masked tangent and divides it by the tie count, which splits the derivative
            // evenly between tied extrema, and the transpose applies the same division and masking in reverse order.
            // The reduced output is broadcast back with the input sharding so that the comparison sees two identically
            // distributed inputs.
            let input_extents = input_shape.dimensions(tangent_context, residuals.values())?;
            let broadcast_output = tangent_context
                .bind(
                    DynamicBroadcastOperation::new(output_axes.clone())
                        .with_output_sharding(input_type.sharding().cloned()),
                    Vec::new(),
                    &[vec![context.primal_to_tangent(output.clone())?], input_extents].concat(),
                )?
                .remove(0);
            let mask = bind_array(
                tangent_context,
                CompareOperation::new(ComparisonDirection::Equal),
                &[input.primal().clone(), broadcast_output],
            )?;
            let mask = bind_array(
                tangent_context,
                ConvertElementTypeOperation::new(input_type.tangent()?.data_type(), false),
                &[mask],
            )?;
            let tie_count = bind_array(
                tangent_context,
                ReduceOperation::new(axes.clone(), ReductionKind::Sum),
                std::slice::from_ref(&mask),
            )?;
            let mask_index = residuals.retain(mask);
            let tie_count_index = residuals.retain(tie_count);
            LinearCallOperation::stage(
                tangent_context,
                residuals.into_values(),
                vec![input_tangent],
                move |residuals, linear_inputs| {
                    let context = linear_inputs[0].dispatch_domain();
                    let masked = bind_array(
                        &context,
                        MulOperation::new(),
                        &[residuals[mask_index].clone(), linear_inputs[0].clone()],
                    )?;
                    let sum = bind_array(&context, ReduceOperation::new(axes, ReductionKind::Sum), &[masked])?;
                    Ok(vec![bind_array(&context, DivOperation::new(), &[sum, residuals[tie_count_index].clone()])?])
                },
                move |residuals, output_cotangents| {
                    let context = output_cotangents[0].dispatch_domain();
                    let input_extents = input_shape.dimensions(&context, residuals)?;
                    let cotangent = bind_array(
                        &context,
                        DivOperation::new(),
                        &[output_cotangents[0].clone(), residuals[tie_count_index].clone()],
                    )?;
                    let cotangent = cotangent.dynamic_broadcast_with_output_sharding(
                        input_extents.as_slice(),
                        output_axes.as_slice(),
                        cotangent_type.sharding().cloned(),
                    )?;
                    Ok(vec![bind_array(&context, MulOperation::new(), &[residuals[mask_index].clone(), cotangent])?])
                },
            )?
        } else {
            // Only sum and mean reach this branch and both are linear, so the forward map applies this reduction,
            // including any requested output sharding, to the tangent. The transpose broadcasts the cotangent back
            // to the retained input shape and mean then divides it by the reduced element count. That count is the
            // product of the retained reduced-axis extents, which stays exact in dimension arithmetic until it is
            // converted to the cotangent data type.
            let forward_operation = self.clone();
            let kind = self.kind();
            LinearCallOperation::stage(
                tangent_context,
                residuals.into_values(),
                vec![input_tangent],
                move |_, linear_inputs| {
                    Ok(vec![bind_array(&linear_inputs[0].dispatch_domain(), forward_operation, linear_inputs)?])
                },
                move |residuals, output_cotangents| {
                    let context = output_cotangents[0].dispatch_domain();
                    let input_extents = input_shape.dimensions(&context, residuals)?;
                    let cotangent = output_cotangents[0].dynamic_broadcast_with_output_sharding(
                        input_extents.as_slice(),
                        output_axes.as_slice(),
                        cotangent_type.sharding().cloned(),
                    )?;
                    if kind == ReductionKind::Sum {
                        return Ok(vec![cotangent]);
                    }

                    let mut element_count = context.dimension_constant(1)?;
                    for axis in axes {
                        let operation = DimensionOperation::Mul(DimensionMulOperation::new(
                            <&DimensionType>::try_from(element_count.r#type().as_ref())?,
                            <&DimensionType>::try_from(input_extents[axis].r#type().as_ref())?,
                        )?);
                        element_count = context
                            .bind(operation, Vec::new(), &[element_count, input_extents[axis].clone()])?
                            .remove(0);
                    }
                    let element_count =
                        context.bind(DimensionToScalarOperation, Vec::new(), &[element_count])?.remove(0);
                    let element_count = bind_array(
                        &context,
                        ConvertElementTypeOperation::new(cotangent_type.data_type(), false),
                        &[element_count],
                    )?;
                    Ok(vec![bind_array(&context, DivOperation::new(), &[cotangent, element_count])?])
                },
            )?
        }
        .remove(0);

        Ok(vec![DifferentiationDual::new(output, MaybeZero::Value(tangent))?])
    }
}

/// Value-level reduction capability that collapses selected axes of an array using a [`ReductionKind`].
///
/// [`Reduce`] fills the same role for [`ReduceOperation`] that [`Broadcast`] fills for [`BroadcastOperation`]. Concrete
/// [`Array`]s reduce immediately, while context-carrying values bind a [`ReduceOperation`] through their own context.
/// The output rank is the input rank minus the number of reduced axes and the remaining axes keep their relative order.
/// Reducing over no axes validates the reduction and returns the input unchanged. Refer to [`ReductionKind`] for the
/// identity, accumulation precision, and supported data types of each kind.
///
/// Numeric reductions are differentiable, while [`ReductionKind::Any`] and [`ReductionKind::All`] have no derivative.
/// Differentiating a [`ReductionKind::LogSumExp`] reduction currently requires a statically shaped input. The other
/// numeric kinds retain the runtime extents of their inputs as residuals during linearization, and so they also support
/// dynamically shaped inputs.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{Array, ProgramError, Reduce, ReductionKind};
/// # fn main() -> Result<(), ProgramError> {
/// // Shapes: input [2, 3] -> output [3] when reducing axis 0 and output [2] when reducing axis 1.
/// let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// assert_eq!(input.reduce(&[0], ReductionKind::Sum)?.to_f64s(), vec![5.0, 7.0, 9.0]);
/// assert_eq!(input.reduce(&[1], ReductionKind::Max)?.to_f64s(), vec![3.0, 6.0]);
/// # Ok(())
/// # }
/// ```
pub trait Reduce: Sized {
    /// Reduces `self` along `axes` using the reduction selected by `kind`.
    ///
    /// # Parameters
    ///
    ///   - `axes`: Distinct axes of `self` to collapse, in any order. An empty list reduces nothing and returns `self`.
    ///   - `kind`: [`ReductionKind`] that determines how the elements along `axes` are combined.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if an axis is out of bounds or repeated, if `kind` does not support the data
    /// type of `self`, if `kind` cannot reduce the unreduced mesh axes of `self` (i.e., [`ReductionKind::Max`],
    /// [`ReductionKind::Min`], [`ReductionKind::LogSumExp`], and integer [`ReductionKind::Mean`] reductions),
    /// or if the context of `self` fails to bind the reduction.
    fn reduce(&self, axes: &[usize], kind: ReductionKind) -> Result<Self, ProgramError>;

    /// Sums `self` along `axes` using [`ReductionKind::Sum`], optionally requesting `output_sharding` for the output.
    /// Sums are the only reductions that support a requested output sharding, which is why this is the only shortcut
    /// function that takes one. For example, requesting an unreduced mesh axis that shards a reduced dimension defers
    /// the cross-device part of the sum. Refer to [`ReduceOperation::with_output_sharding`] for the complete set of
    /// valid requests.
    ///
    /// Every value validates a requested sharding and derives the output type through the type inference of the
    /// corresponding [`ReduceOperation`]. Context-carrying values attach the request to the staged [`ReduceOperation`],
    /// leaving the placement to the backend that executes it. Concrete [`Array`]s live on a single device, where a
    /// sharding only describes distribution metadata, and so they compute the same elements with or without a request,
    /// while their output type still records the requested sharding.
    ///
    /// # Parameters
    ///
    ///   - `axes`: Distinct axes of `self` to collapse, with the same semantics as in [`Self::reduce`].
    ///   - `output_sharding`: Requested [`Sharding`] of the output, which must match the output rank and the mesh
    ///     of `self`, or [`None`] to infer the output sharding from the input.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] under the same conditions as [`Self::reduce`] or if `output_sharding` is not a valid
    /// request for the type of `self`.
    fn reduce_sum(&self, axes: &[usize], output_sharding: Option<Sharding>) -> Result<Self, ProgramError>;

    /// Averages `self` along `axes` using [`ReductionKind::Mean`]. Refer to [`Self::reduce`] for the semantics of
    /// `axes` and for the errors that this function may return.
    #[inline]
    fn reduce_mean(&self, axes: &[usize]) -> Result<Self, ProgramError> {
        self.reduce(axes, ReductionKind::Mean)
    }

    /// Computes the numerically stable `log(sum(exp(self)))` along `axes` using [`ReductionKind::LogSumExp`], whose
    /// documentation describes the guarded computation and its data-type limits. Refer to [`Self::reduce`] for the
    /// semantics of `axes` and for the errors that this function may return.
    #[inline]
    fn reduce_log_sum_exp(&self, axes: &[usize]) -> Result<Self, ProgramError> {
        self.reduce(axes, ReductionKind::LogSumExp)
    }

    /// Computes the maximum of `self` along `axes` using [`ReductionKind::Max`]. Refer to [`Self::reduce`] for the
    /// semantics of `axes` and for the errors that this function may return.
    #[inline]
    fn reduce_max(&self, axes: &[usize]) -> Result<Self, ProgramError> {
        self.reduce(axes, ReductionKind::Max)
    }

    /// Computes the minimum of `self` along `axes` using [`ReductionKind::Min`]. Refer to [`Self::reduce`] for the
    /// semantics of `axes` and for the errors that this function may return.
    #[inline]
    fn reduce_min(&self, axes: &[usize]) -> Result<Self, ProgramError> {
        self.reduce(axes, ReductionKind::Min)
    }

    /// Computes the disjunction of the Boolean elements of `self` along `axes` using [`ReductionKind::Any`]. Refer to
    /// [`Self::reduce`] for the semantics of `axes` and for the errors that this function may return.
    #[inline]
    fn reduce_any(&self, axes: &[usize]) -> Result<Self, ProgramError> {
        self.reduce(axes, ReductionKind::Any)
    }

    /// Computes the conjunction of the Boolean elements of `self` along `axes` using [`ReductionKind::All`]. Refer to
    /// [`Self::reduce`] for the semantics of `axes` and for the errors that this function may return.
    #[inline]
    fn reduce_all(&self, axes: &[usize]) -> Result<Self, ProgramError> {
        self.reduce(axes, ReductionKind::All)
    }
}

impl Reduce for Array {
    fn reduce(&self, axes: &[usize], kind: ReductionKind) -> Result<Self, ProgramError> {
        if axes.is_empty() {
            ReduceOperation::new(Vec::new(), kind).infer_output_types(&[self.r#type().into_owned()], &[])?;
            return Ok(self.clone());
        }

        let data_type = self.r#type().data_type();

        // Reuse the abstract rule for validation and for the complete output metadata. The concrete kernel below
        // then decodes directly from the input's physical layout into the output's addressed storage.
        let output_type = reduce_abstract(self.r#type().as_ref(), axes, kind)?;
        if data_type == DataType::Zero {
            return Self::new(output_type, Vec::new());
        }

        // Narrow floating-point reductions accumulate and normalize in `f32`. Rounding only the final output
        // avoids losing small contributions and overflowing the element count used by a mean.
        if data_type.is_floating_point()
            && !matches!(data_type, DataType::F32 | DataType::F64 | DataType::F8E8M0FNU)
            && matches!(kind, ReductionKind::Sum | ReductionKind::Mean | ReductionKind::LogSumExp)
        {
            return self.convert_element_type(DataType::F32)?.reduce(axes, kind)?.convert_element_type(data_type);
        }

        match kind {
            ReductionKind::LogSumExp => {
                dispatch_on_array_element_type!(@float data_type, |Element| {
                    // Compute `log(sum(exp(input)))` by subtracting the maximum of each reduced slice before
                    // exponentiating. Non-finite maxima are replaced by zero, so that NaNs and infinities propagate
                    // through the exponentials instead of producing `-∞ - -∞ = NaN` for all-`-∞` or empty slices.
                    // Narrow inputs were already widened above, but reassociation in other backends can still
                    // change the final rounding.
                    let zero = Element::zero()?;
                    let maximums = self
                        .reduce_elements::<Element>(
                            output_type.clone(),
                            axes,
                            Element::max_identity(),
                            |left, right| Ok(ArrayElement::max(&left, &right)),
                        )?
                        .map_elements::<Element, Element>(output_type.clone(), |value| {
                            Ok(if value.convert_to::<f64>()?.is_finite() { value } else { zero })
                        })?;

                    // Accumulate the shifted exponentials of every input element into the output element that its
                    // non-reduced coordinates address, starting each sum from zero. Primitive floating-point types
                    // have inherent `exp` and `log` functions that shadow the fallible element functions, and so we
                    // call the latter through `FloatingPointArrayElement` explicitly.
                    let input_shape = self.r#type().static_shape().unwrap();
                    let input_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
                    let output_addressing = ArrayAddressing::new(output_type.clone())?;
                    let mut reduce_mask = vec![false; input_shape.rank()];
                    axes.iter().for_each(|axis| reduce_mask[*axis] = true);
                    let mut bytes = vec![0; output_addressing.storage_byte_len()];
                    for output in 0..output_addressing.element_count() {
                        zero.encode(&mut bytes[output_addressing.byte_range_for_flat_index(output)]);
                    }
                    let mut input_index = vec![0usize; input_shape.rank()];
                    let mut output_index = vec![0usize; output_type.rank()];
                    for _ in 0..input_addressing.element_count() {
                        let mut output_axis = 0usize;
                        for axis in 0..input_shape.rank() {
                            if !reduce_mask[axis] {
                                output_index[output_axis] = input_index[axis];
                                output_axis += 1;
                            }
                        }
                        let input_range = input_addressing.byte_range_unchecked(&input_index);
                        let input_value = Element::decode(&self.storage_bytes()[input_range]);
                        let output_range = output_addressing.byte_range_unchecked(&output_index);
                        let maximum = Element::decode(&maximums.storage_bytes()[output_range.clone()]);
                        let shifted = FloatingPointArrayElement::exp(input_value.sub(maximum)?)?;
                        let sum = Element::decode(&bytes[output_range.clone()]).add(shifted)?;
                        sum.encode(&mut bytes[output_range]);
                        input_addressing.advance_index(&mut input_index);
                    }

                    // Take the logarithm of each sum and add its maximum back.
                    for output in 0..output_addressing.element_count() {
                        let range = output_addressing.byte_range_for_flat_index(output);
                        let maximum = Element::decode(&maximums.storage_bytes()[range.clone()]);
                        let sum = Element::decode(&bytes[range.clone()]);
                        let value = FloatingPointArrayElement::log(sum)?.add(maximum)?;
                        value.encode(&mut bytes[range]);
                    }

                    Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
                })
            }
            ReductionKind::Sum | ReductionKind::Mean => {
                dispatch_on_array_element_type!(@numeric data_type, |Element| {
                    self.reduce_sum_or_mean_elements::<Element>(
                        output_type,
                        axes,
                        kind == ReductionKind::Mean,
                    )
                })
            }
            ReductionKind::Max | ReductionKind::Min => {
                dispatch_on_array_element_type!(data_type, |Element| {
                    let identity =
                        if kind == ReductionKind::Max { Element::max_identity() } else { Element::min_identity() };
                    self.reduce_elements::<Element>(output_type, axes, identity, |left, right| {
                        Ok(if kind == ReductionKind::Max {
                            ArrayElement::max(&left, &right)
                        } else {
                            ArrayElement::min(&left, &right)
                        })
                    })
                })
            }
            ReductionKind::Any => {
                self.reduce_elements::<bool>(output_type, axes, false, |left, right| Ok(left | right))
            }
            ReductionKind::All => self.reduce_elements::<bool>(output_type, axes, true, |left, right| Ok(left & right)),
        }
    }

    fn reduce_sum(&self, axes: &[usize], output_sharding: Option<Sharding>) -> Result<Self, ProgramError> {
        // A concrete array lives on a single device, so a requested sharding cannot change its elements and no data has
        // to move. Its type still records the requested sharding, and that output type comes from the type inference of
        // the same `ReduceOperation`, so that eager evaluation validates the request and produces exactly the output
        // type of a staged sum. Sharding does not affect storage, and so the sum's bytes can be reused.
        let Some(output_sharding) = output_sharding else {
            return self.reduce(axes, ReductionKind::Sum);
        };
        let mut output_types = ReduceOperation::new(axes.to_vec(), ReductionKind::Sum)
            .with_output_sharding(output_sharding)?
            .infer_output_types(&[self.r#type().into_owned()], &[])?;
        check_count!("output", output_types, 1, ProgramError);
        let output = self.reduce(axes, ReductionKind::Sum)?;
        Ok(Self::new_unchecked(output_types.remove(0), output.shared_storage().clone()))
    }
}

// Any context-carrying value reduces by binding a `ReduceOperation` through its context. The `From<ReduceOperation>`
// bound makes this disjoint from the eager value types (whose context operation is `ConstantOperation`), so it covers
// the transform tracers without conflicting with the concrete implementations.
impl<V: Value<Type = ArrayType, DispatchDomain: Context<Type = ArrayType, Operation: From<ReduceOperation>>>> Reduce
    for V
{
    fn reduce(&self, axes: &[usize], kind: ReductionKind) -> Result<Self, ProgramError> {
        if axes.is_empty() {
            ReduceOperation::new(Vec::new(), kind).infer_output_types(&[self.r#type().into_owned()], &[])?;
            return Ok(self.clone());
        }
        let mut outputs = self.dispatch_domain().bind(
            ReduceOperation::new(axes.to_vec(), kind),
            Vec::new(),
            std::slice::from_ref(self),
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }

    fn reduce_sum(&self, axes: &[usize], output_sharding: Option<Sharding>) -> Result<Self, ProgramError> {
        // Without a requested sharding, a sum is an ordinary reduction. A requested sharding is staged even when no
        // axes are reduced, because it may still change how the output is distributed.
        if output_sharding.is_none() {
            return self.reduce(axes, ReductionKind::Sum);
        }
        let mut outputs = self.dispatch_domain().bind(
            ReduceOperation::new(axes.to_vec(), ReductionKind::Sum).with_output_sharding(output_sharding)?,
            Vec::new(),
            std::slice::from_ref(self),
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

// TODO(eaplatanios): Review from here onwards.

impl Array {
    /// Reduces typed elements directly from addressed input storage into one addressed output buffer. `identity`
    /// initializes every output cell, including those whose reduced axes are empty.
    fn reduce_elements<T: ArrayElement>(
        &self,
        output_type: ArrayType,
        axes: &[usize],
        identity: T,
        combine: impl Fn(T, T) -> Result<T, ProgramError>,
    ) -> Result<Self, ProgramError> {
        debug_assert_eq!(self.r#type().data_type(), T::data_type());
        debug_assert_eq!(output_type.data_type(), T::data_type());
        let input_shape = self.r#type().static_shape().unwrap();
        let input_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let mut reduce_mask = vec![false; input_shape.rank()];
        axes.iter().for_each(|axis| reduce_mask[*axis] = true);

        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        for output in 0..output_addressing.element_count() {
            identity.encode(&mut bytes[output_addressing.byte_range_for_flat_index(output)]);
        }

        let mut input_index = vec![0usize; input_shape.rank()];
        let mut output_index = vec![0usize; output_type.rank()];
        for _ in 0..input_addressing.element_count() {
            let mut output_axis = 0usize;
            for axis in 0..input_shape.rank() {
                if !reduce_mask[axis] {
                    output_index[output_axis] = input_index[axis];
                    output_axis += 1;
                }
            }
            let input_value = T::decode(&self.storage_bytes()[input_addressing.byte_range_unchecked(&input_index)]);
            let output_range = output_addressing.byte_range_unchecked(&output_index);
            let value = combine(T::decode(&bytes[output_range.clone()]), input_value)?;
            value.encode(&mut bytes[output_range]);
            input_addressing.advance_index(&mut input_index);
        }
        Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
    }

    /// Accumulates a sum, optionally dividing each output by the reduced element count. Integer arithmetic wraps
    /// in the element type and retains zero for empty means; floating-point and complex empty means produce NaNs.
    fn reduce_sum_or_mean_elements<T: ElementDivideByCount>(
        &self,
        output_type: ArrayType,
        axes: &[usize],
        mean: bool,
    ) -> Result<Self, ProgramError> {
        let output = self.reduce_elements::<T>(output_type.clone(), axes, T::zero()?, T::add)?;
        if !mean {
            return Ok(output);
        }
        let shape = self.r#type().static_shape().unwrap();
        let count = axes.iter().map(|axis| shape[*axis]).product::<usize>();
        let count = if T::data_type().is_integer() { count.max(1) } else { count };
        output.map_elements::<T, T>(output_type, |value| value.divide_by_count(count))
    }
}

/// Element-level mean divisor, serving mean reductions, which have no capability analogue of their own because a
/// mean lowers to a sum followed by a division by the reduced element count.
trait ElementDivideByCount: NumericArrayElement {
    /// Divides this element by `count` after converting `count` to the element type.
    fn divide_by_count(self, count: usize) -> Result<Self, ProgramError>;
}

// Implements typed arithmetic for signed primitive integers with deterministic two's-complement wrapping.
macro_rules! impl_element_divide_by_count_for_signed_integer {
    ($type:ty) => {
        impl ElementDivideByCount for $type {
            fn divide_by_count(self, count: usize) -> Result<Self, ProgramError> {
                let divisor = count as Self;
                if divisor == 0 {
                    return Err(TypeError::invalid(format!(
                        "cannot divide an integer array element of data type `{}` by zero",
                        Self::data_type(),
                    ))
                    .into());
                }
                if self == Self::MIN && divisor == -1 {
                    return Err(TypeError::invalid(format!(
                        "cannot divide the minimum integer array element of data type `{}` by -1",
                        Self::data_type(),
                    ))
                    .into());
                }
                Ok(self / divisor)
            }
        }
    };
}

// Implements typed arithmetic for unsigned primitive integers with deterministic modular wrapping.
macro_rules! impl_element_divide_by_count_for_unsigned_integer {
    ($type:ty) => {
        impl ElementDivideByCount for $type {
            fn divide_by_count(self, count: usize) -> Result<Self, ProgramError> {
                let divisor = count as Self;
                if divisor == 0 {
                    return Err(TypeError::invalid(format!(
                        "cannot divide an integer array element of data type `{}` by zero",
                        Self::data_type(),
                    ))
                    .into());
                }
                Ok(self / divisor)
            }
        }
    };
}

impl_element_divide_by_count_for_signed_integer!(i8);
impl_element_divide_by_count_for_signed_integer!(i16);
impl_element_divide_by_count_for_signed_integer!(i32);
impl_element_divide_by_count_for_signed_integer!(i64);
impl_element_divide_by_count_for_unsigned_integer!(u8);
impl_element_divide_by_count_for_unsigned_integer!(u16);
impl_element_divide_by_count_for_unsigned_integer!(u32);
impl_element_divide_by_count_for_unsigned_integer!(u64);

// Implements modular arithmetic for a signed sub-byte integer's checked low-bit encoding.
macro_rules! impl_element_divide_by_count_for_signed_sub_byte_integer {
    ($type:ty) => {
        impl ElementDivideByCount for $type {
            fn divide_by_count(self, count: usize) -> Result<Self, ProgramError> {
                let bit_mask = Self::MIN.to_bits() | Self::MAX.to_bits();
                let divisor = Self::from_bits(count as u8 & bit_mask).unwrap().value();
                if divisor == 0 {
                    return Err(TypeError::invalid(format!(
                        "cannot divide an integer array element of data type `{}` by zero",
                        Self::data_type(),
                    ))
                    .into());
                }
                if self == Self::MIN && divisor == -1 {
                    return Err(TypeError::invalid(format!(
                        "cannot divide the minimum integer array element of data type `{}` by -1",
                        Self::data_type(),
                    ))
                    .into());
                }
                Ok(Self::new(self.value() / divisor).unwrap())
            }
        }
    };
}

// Implements modular arithmetic for an unsigned sub-byte integer's checked low-bit encoding.
macro_rules! impl_element_divide_by_count_for_unsigned_sub_byte_integer {
    ($type:ty) => {
        impl ElementDivideByCount for $type {
            fn divide_by_count(self, count: usize) -> Result<Self, ProgramError> {
                let divisor = Self::from_bits(count as u8 & Self::MAX.to_bits()).unwrap().value();
                if divisor == 0 {
                    return Err(TypeError::invalid(format!(
                        "cannot divide an integer array element of data type `{}` by zero",
                        Self::data_type(),
                    ))
                    .into());
                }
                Ok(Self::new(self.value() / divisor).unwrap())
            }
        }
    };
}

impl_element_divide_by_count_for_signed_sub_byte_integer!(i1);
impl_element_divide_by_count_for_signed_sub_byte_integer!(i2);
impl_element_divide_by_count_for_signed_sub_byte_integer!(i4);
impl_element_divide_by_count_for_unsigned_sub_byte_integer!(u1);
impl_element_divide_by_count_for_unsigned_sub_byte_integer!(u2);
impl_element_divide_by_count_for_unsigned_sub_byte_integer!(u4);

// Implements arithmetic for a low-precision floating-point format through its exact f64 conversion contract.
macro_rules! impl_element_divide_by_count_for_low_precision_float {
    ($type:ty) => {
        impl ElementDivideByCount for $type {
            #[inline]
            fn divide_by_count(self, count: usize) -> Result<Self, ProgramError> {
                let divisor = Self::from_f64(count as f64)?;
                Ok(Self::from_f64(self.to_f64() / divisor.to_f64())?)
            }
        }
    };
}

impl_element_divide_by_count_for_low_precision_float!(f4e2m1fn);
impl_element_divide_by_count_for_low_precision_float!(f6e2m3fn);
impl_element_divide_by_count_for_low_precision_float!(f6e3m2fn);
impl_element_divide_by_count_for_low_precision_float!(f8e3m4);
impl_element_divide_by_count_for_low_precision_float!(f8e4m3);
impl_element_divide_by_count_for_low_precision_float!(f8e4m3fn);
impl_element_divide_by_count_for_low_precision_float!(f8e4m3fnuz);
impl_element_divide_by_count_for_low_precision_float!(f8e4m3b11fnuz);
impl_element_divide_by_count_for_low_precision_float!(f8e5m2);
impl_element_divide_by_count_for_low_precision_float!(f8e5m2fnuz);
impl_element_divide_by_count_for_low_precision_float!(f8e8m0fnu);

// Implements ordinary arithmetic for a native or half-precision real floating-point type.
macro_rules! impl_element_divide_by_count_for_float {
    ($type:ty, $from_count:expr) => {
        impl ElementDivideByCount for $type {
            #[inline]
            fn divide_by_count(self, count: usize) -> Result<Self, ProgramError> {
                Ok(self / $from_count(count))
            }
        }
    };
}

impl_element_divide_by_count_for_float!(bf16, |count: usize| bf16::from_f64(count as f64));
impl_element_divide_by_count_for_float!(f16, |count: usize| f16::from_f64(count as f64));
impl_element_divide_by_count_for_float!(f32, |count: usize| count as f32);
impl_element_divide_by_count_for_float!(f64, |count: usize| count as f64);

// Implements complex arithmetic; division by a real count acts componentwise to avoid an unnecessary complex norm.
macro_rules! impl_element_divide_by_count_for_complex {
    ($component:ty) => {
        impl ElementDivideByCount for Complex<$component> {
            #[inline]
            fn divide_by_count(self, count: usize) -> Result<Self, ProgramError> {
                // Dividing by a real count is componentwise by definition, which also sidesteps the generic complex
                // division's norm computation, whose intermediate values can overflow for large counts.
                let divisor = count as $component;
                Ok(Complex::new(self.re / divisor, self.im / divisor))
            }
        }
    };
}

impl_element_divide_by_count_for_complex!(f32);
impl_element_divide_by_count_for_complex!(f64);

/// Returns the output [`ArrayType`] produced by reducing `input` along `axes` with `kind`.
///
/// Validates that:
///   - `axes` are unique and within `0..rank(input)`;
///   - `kind` matches the input data type (Boolean for Any/All, Boolean or numeric for Max/Min, and numeric for
///     Sum/Mean; logarithmic sums require the real floating-point domain documented on [`ReductionKind::LogSumExp`]).
///
/// The reduced axes are removed from the output shape; non-reduced axes keep their order. The output [`Sharding`]
/// drops the reduced axes' per-dimension [`ShardingDimension`] entries while
/// retaining the remaining entries in order. Reduction-state and manual-axis sets pass through unchanged; extrema,
/// logarithmic sums, and integer means of partial sums are rejected because they do not commute with the pending sum.
/// The backend partitioner owns cross-shard reductions over sharded dimensions; use
/// [`ReduceOperation::with_output_sharding`] to request an unreduced output that defers it. The
/// [`Layout`](crate::arrays::Layout) is dropped (it is rank-specific) and the [`Memory`](crate::arrays::Memory)
/// placement is preserved.
pub fn reduce_abstract(input: &ArrayType, axes: &[usize], kind: ReductionKind) -> Result<ArrayType, TypeError> {
    let rank = input.rank();
    let mut reduce_mask = vec![false; rank];
    for axis in axes {
        if *axis >= rank {
            return Err(TypeError::invalid(format!(
                "`{REDUCE_OPERATION_NAME}` axis {axis} is out of bounds for rank {rank}",
            )));
        }
        if reduce_mask[*axis] {
            return Err(TypeError::invalid(format!("`{REDUCE_OPERATION_NAME}` contains duplicate axis {axis}")));
        }
        reduce_mask[*axis] = true;
    }

    let data_type = input.data_type();

    // Validate axes before the element domain so malformed geometry retains diagnostic precedence.
    if kind == ReductionKind::LogSumExp {
        // Logarithmic sums are built from exponentials and logarithms, so they require the real floating-point domain
        // documented on `ReductionKind::LogSumExp`, restricted to formats that represent the negative infinity that the
        // maximum-shifted evaluation starts from.
        if !data_type.is_floating_point() {
            return Err(TypeError::invalid(format!(
                "`{REDUCE_OPERATION_NAME}` with kind `{kind}` requires real floating-point inputs but got \
                 `{data_type}`",
            )));
        }
        if !matches!(
            data_type,
            DataType::BF16
                | DataType::F16
                | DataType::F32
                | DataType::F64
                | DataType::F8E3M4
                | DataType::F8E4M3
                | DataType::F8E5M2,
        ) {
            return Err(TypeError::invalid(format!(
                "`{REDUCE_OPERATION_NAME}` with kind `{kind}` requires a floating-point format that represents \
                 negative infinity but got `{data_type}`",
            )));
        }
    } else {
        let (requirement, supports_kind) = if matches!(kind, ReductionKind::Any | ReductionKind::All) {
            ("Boolean", data_type.is_boolean())
        } else if matches!(kind, ReductionKind::Max | ReductionKind::Min) {
            ("Boolean or numeric", data_type.is_boolean() || data_type.is_numeric() || data_type == DataType::Zero)
        } else {
            ("numeric", data_type.is_numeric() || data_type == DataType::Zero)
        };
        if !supports_kind {
            return Err(TypeError::invalid(format!(
                "`{REDUCE_OPERATION_NAME}` with kind `{kind}` requires {requirement} inputs but got `{data_type}`",
            )));
        }
    }

    if !axes.is_empty()
        && (matches!(kind, ReductionKind::Max | ReductionKind::Min | ReductionKind::LogSumExp)
            || (kind == ReductionKind::Mean && data_type.is_integer()))
        && input.sharding().is_some_and(|sharding| !sharding.unreduced_axes().is_empty())
    {
        return Err(TypeError::invalid(format!(
            "`{REDUCE_OPERATION_NAME}` with kind `{kind}` cannot reduce inputs with unreduced axes",
        )));
    }

    let dimensions = input
        .shape()
        .dimensions()
        .iter()
        .enumerate()
        .filter_map(|(axis, size)| (!reduce_mask[axis]).then_some(size.clone()))
        .collect::<Vec<_>>();

    // The output drops the per-dimension sharding entries of the reduced axes and keeps the remaining entries in
    // order, while the reduction-state and manual-axis sets pass through unchanged.
    let sharding = input
        .sharding()
        .map(|sharding| {
            let dimensions = sharding
                .dimensions()
                .iter()
                .enumerate()
                .filter_map(|(axis, dimension)| (!reduce_mask[axis]).then(|| dimension.clone()))
                .collect::<Vec<_>>();
            Sharding::new(sharding.mesh().clone(), dimensions)
                .and_then(|output| output.with_unreduced_axes(sharding.unreduced_axes().clone()))
                .and_then(|output| output.with_reduced_axes(sharding.reduced_axes().clone()))
                .and_then(|output| output.with_varying_manual_axes(sharding.varying_manual_axes().clone()))
                .map_err(|error| {
                    TypeError::invalid(format!(
                        "`{REDUCE_OPERATION_NAME}` output sharding construction failed: {error}"
                    ))
                })
        })
        .transpose()?;
    ArrayType::new(data_type, Shape::new(dimensions))
        .with_memory(input.memory())
        .with_sharding(sharding)
        .map_err(|error| TypeError::invalid(error.to_string()))
}

/// Builds the `output_axes` vector that maps a reduced output's axes back to the
/// corresponding input axes. Output axis `j` corresponds to the `j`-th non-reduced input axis;
/// the returned vector lists those input-axis indices in order.
pub(crate) fn output_to_input_axis_map(input_rank: usize, reduced_axes: &[usize]) -> Vec<usize> {
    let mut reduce_mask = vec![false; input_rank];
    for axis in reduced_axes {
        reduce_mask[*axis] = true;
    }
    (0..input_rank).filter(|axis| !reduce_mask[*axis]).collect()
}

/// Binds the single-output array operation `operation` in `context`, lifting it into the context's operation family
/// through that family's [`OperationProjection<ArrayType>`](OperationProjection) member family, and returns its output.
fn bind_array<C: Context<Type = ArrayIrType>, O>(
    context: &C,
    operation: O,
    inputs: &[C::Value],
) -> Result<C::Value, ProgramError>
where
    C::Operation: OperationProjection<ArrayType, Projected: From<O>>,
{
    let operation = <C::Operation as OperationProjection<ArrayType>>::Projected::from(operation);
    let mut outputs = context.bind(operation, Vec::new(), inputs)?;
    check_count!("output", outputs, 1, ProgramError);
    Ok(outputs.remove(0))
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::batching::DynamicArrayExtentBatchingPolicy;
    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, ArrayType, DataType, Dimension, DimensionBounds,
        DimensionType, DimensionVariable, Layout, LogicalMesh, MeshAxis, MeshAxisType, RaggedAxis, Shape, Sharding,
        ShardingDimension, StridedLayout,
    };
    use crate::contexts::{EagerContext, ProjectedContext, StagingContext};
    use crate::differentiation::{DifferentiationError, TransposableOperation, TranspositionContext, differentiate_at};
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{EmptyRegionDriver, MaybeZero, ProgramBuilder, ProgramError, Typed, ValueProjection};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_reduction_kind_name() {
        for (kind, name) in [
            (ReductionKind::Sum, "sum"),
            (ReductionKind::Mean, "mean"),
            (ReductionKind::LogSumExp, "log_sum_exp"),
            (ReductionKind::Max, "max"),
            (ReductionKind::Min, "min"),
            (ReductionKind::Any, "any"),
            (ReductionKind::All, "all"),
        ] {
            assert_eq!(kind.name(), name);
            assert_eq!(kind.to_string(), name);
        }
    }

    #[test]
    fn test_reduce() {
        let operation = ReduceOperation::new(vec![0, 2], ReductionKind::Sum);
        assert_eq!(operation.axes(), &[0, 2]);
        assert_eq!(operation.kind(), ReductionKind::Sum);
        assert_eq!(operation.output_sharding(), None);
        assert_eq!(operation.to_string(), "reduce [kind=sum, axes=[0, 2]]");
    }

    #[test]
    fn test_reduce_log_sum_exp() {
        assert_eq!(
            ReduceOperation::new(vec![0, 2], ReductionKind::LogSumExp).to_string(),
            "reduce [kind=log_sum_exp, axes=[0, 2]]",
        );
        assert_eq!(ReduceOperation::new(vec![1], ReductionKind::LogSumExp).axes(), &[1]);
    }

    #[test]
    fn test_reduce_operation_with_output_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 1);

        // A sum accepts a requested output sharding, and passing `None` clears it again.
        let operation =
            ReduceOperation::new(vec![0], ReductionKind::Sum).with_output_sharding(sharding.clone()).unwrap();
        assert_eq!(operation.output_sharding(), Some(&sharding));
        assert_eq!(operation.with_output_sharding(None).unwrap().output_sharding(), None);

        // Every other reduction kind rejects a requested output sharding when the operation is constructed, while
        // still accepting the absence of one.
        for kind in [
            ReductionKind::Mean,
            ReductionKind::LogSumExp,
            ReductionKind::Max,
            ReductionKind::Min,
            ReductionKind::Any,
            ReductionKind::All,
        ] {
            assert_eq!(
                ReduceOperation::new(vec![0], kind).with_output_sharding(sharding.clone()),
                Err(TypeError::invalid(format!(
                    "`reduce` with kind `{kind}` does not support a requested output sharding (only kind `sum` does)",
                ))),
            );
            assert_eq!(
                ReduceOperation::new(vec![0], kind).with_output_sharding(None),
                Ok(ReduceOperation::new(vec![0], kind)),
            );
        }
    }

    #[test]
    fn test_reduce_type_inference() {
        check_operation_type_inference!(
            operation = ReduceOperation::new(vec![1], ReductionKind::Sum),
            cases = [
                {
                    input_types = [ArrayType::new_static(DataType::F64, [3, 2])],
                    output_types = [ArrayType::new_static(DataType::F64, [3])],
                },
                {
                    input_types = [ArrayType::new_static(DataType::F64, [3])],
                    error = "`reduce` axis 1 is out of bounds for rank 1",
                },
            ],
        );
    }

    #[test]
    fn test_reduce_type_inference_output_sharding() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let input = ArrayType::new_static(DataType::F64, [2, 3])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let unreduced = Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap();

        // The requested partial sum may defer only a sum that the input placement actually requires.
        check_operation_type_inference!(
            operation = ReduceOperation::new(vec![0], ReductionKind::Sum)
                .with_output_sharding(unreduced.clone())
                .unwrap(),
            cases = [{
                input_types = [input.clone()],
                output_types = [ArrayType::new_static(DataType::F64, [3]).with_sharding(unreduced.clone()).unwrap()],
            }],
        );
        check_operation_type_inference!(
            operation = ReduceOperation::new(vec![0], ReductionKind::Sum)
                .with_output_sharding(Sharding::replicated(mesh, 1).with_unreduced_axes(["y"]).unwrap())
                .unwrap(),
            cases = [{
                input_types = [input.clone()],
                error = "`reduce` output sharding unreduced axes must be among the explicit axes sharding the \
                         reduced dimensions or the input's unreduced axes",
            }],
        );
    }

    #[test]
    fn test_reduce_type_inference_output_sharding_state() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let input = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(
                Sharding::replicated(mesh.clone(), 1)
                    .with_reduced_axes(["x"])
                    .unwrap()
                    .with_varying_manual_axes(["m"])
                    .unwrap(),
            )
            .unwrap();

        // A placement request leaves the input's semantic state intact without staging a collective.
        check_operation_type_inference!(
            operation = ReduceOperation::new(vec![0], ReductionKind::Sum)
                .with_output_sharding(Sharding::replicated(mesh.clone(), 0))
                .unwrap(),
            cases = [{
                input_types = [input],
                output_types = [ArrayType::scalar(DataType::F32)
                    .with_sharding(Sharding::replicated(mesh.clone(), 0)
                        .with_reduced_axes(["x"]).unwrap()
                        .with_varying_manual_axes(["m"]).unwrap()).unwrap()],
            }],
        );
        let input = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1))
            .unwrap();
        check_operation_type_inference!(
            operation = ReduceOperation::new(vec![0], ReductionKind::Sum)
                .with_output_sharding(Sharding::replicated(mesh.clone(), 0).with_reduced_axes(["x"]).unwrap())
                .unwrap(),
            cases = [{
                input_types = [input.clone()],
                error = "`reduce` output sharding cannot request reduced axes",
            }],
        );
        check_operation_type_inference!(
            operation = ReduceOperation::new(vec![0], ReductionKind::Sum)
                .with_output_sharding(Sharding::replicated(mesh, 0).with_varying_manual_axes(["m"]).unwrap())
                .unwrap(),
            cases = [{
                input_types = [input],
                error = "`reduce` output sharding cannot change manual variation",
            }],
        );
    }

    #[test]
    fn test_reduce_type_inference_unreduced_inputs() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let output = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh, 0).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        check_operation_type_inference!(
            operation = ReduceOperation::new(vec![0], ReductionKind::Sum),
            cases = [{ input_types = [input.clone()], output_types = [output.clone()] }],
        );
        check_operation_type_inference!(
            operation = ReduceOperation::new(vec![0], ReductionKind::Mean),
            cases = [{ input_types = [input.clone()], output_types = [output] }],
        );
        // Extrema and logarithmic sums do not commute with the pending cross-device sum.
        check_operation_type_inference!(
            operation = ReduceOperation::new(vec![0], ReductionKind::Max),
            cases = [{
                input_types = [input.clone()],
                error = "`reduce` with kind `max` cannot reduce inputs with unreduced axes",
            }],
        );
        check_operation_type_inference!(
            operation = ReduceOperation::new(vec![0], ReductionKind::Min),
            cases = [{
                input_types = [input.clone()],
                error = "`reduce` with kind `min` cannot reduce inputs with unreduced axes",
            }],
        );
        check_operation_type_inference!(
            operation = ReduceOperation::new(vec![0], ReductionKind::LogSumExp),
            cases = [{
                input_types = [input.clone()],
                error = "`reduce` with kind `log_sum_exp` cannot reduce inputs with unreduced axes",
            }],
        );
        // Integer means truncate before the pending sum and therefore are not linear either.
        check_operation_type_inference!(
            operation = ReduceOperation::new(vec![0], ReductionKind::Mean),
            cases = [{
                input_types = [input.clone().with_data_type(DataType::I32)],
                error = "`reduce` with kind `mean` cannot reduce inputs with unreduced axes",
            }],
        );
        check_operation_type_inference!(
            operation = ReduceOperation::new(vec![], ReductionKind::Max),
            cases = [{ input_types = [input.clone()], output_types = [input] }],
        );
    }

    #[test]
    fn test_reduce_type_inference_log_sum_exp() {
        check_operation_type_inference!(
            operation = ReduceOperation::new(vec![1], ReductionKind::LogSumExp),
            cases = [
                {
                    input_types = [ArrayType::new_static(DataType::F32, [3, 2])],
                    output_types = [ArrayType::new_static(DataType::F32, [3])],
                },
                {
                    input_types = [ArrayType::new_static(DataType::F64, [3])],
                    error = "`reduce` axis 1 is out of bounds for rank 1",
                },
            ],
        );
    }

    #[test]
    fn test_reduce_interpretation() {
        use crate::arrays::{
            Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType, Sharding, ShardingDimension, StridedLayout,
        };

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let output_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new_static(DataType::F64, [2, 3])
            .with_layout(Layout::Strided(StridedLayout::new(vec![24, 8])))
            .with_memory(Memory::Host { pinned: true })
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap(),
            )
            .unwrap();
        let input = Array::from_elements::<f64>(input_type, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let outputs = ReduceOperation::new(vec![1], ReductionKind::Sum)
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        let output = outputs.into_iter().next().unwrap();
        // The payload kernel and abstract rule must agree on the complete output type: reduction projects sharding,
        // preserves memory placement, and clears the rank-specific layout.
        assert_eq!(
            output.r#type().as_ref(),
            &ArrayType::new_static(DataType::F64, [2])
                .with_memory(Memory::Host { pinned: true })
                .with_sharding(output_sharding)
                .unwrap(),
        );
        assert_eq!(output.elements::<f64>(), Ok(vec![6.0, 15.0]));
    }

    #[test]
    fn test_reduce_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = ReduceOperation::new(vec![0], ReductionKind::LogSumExp),
            inputs = [Array::vector(vec![0.0, 0.0]).unwrap()],
            expected = Array::scalar(std::f64::consts::LN_2).unwrap(),
        );
    }

    #[test]
    fn test_reduce_batching() {
        check_operation_batching!(
            @exact,
            operation = ReduceOperation::new(vec![1], ReductionKind::Sum),
            axis_size = 2,
            cases = [{
                inputs = [(@replicated, Array::matrix(2, 3, vec![1.0; 6]).unwrap())],
                outputs = [(@replicated, Array::vector(vec![3.0, 3.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_reduce_batching_non_batch_axis() {
        // Physical input is [3 batch items, 2 rows, 3 cols] mapped at axis 0. Per-item reduce over
        // axis 1 (the "cols" axis from the per-item view; physically axis 2 after batching).
        check_operation_batching!(
            @exact,
            operation = ReduceOperation::new(vec![1], ReductionKind::Sum),
            axis_size = 3,
            cases = [{
                inputs = [(@mapped(
                    axis = 0
                ), Array::from_elements::<f64>(
                    ArrayType::new_static(DataType::F64, [3, 2, 3]),
                    &(0..18).map(|index| index as f64).collect::<Vec<_>>(),
                ).unwrap())],
                outputs = [(@mapped(
                    axis = 0
                ), Array::matrix(3, 2, vec![3.0, 12.0, 21.0, 30.0, 39.0, 48.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_reduce_batching_non_leading_mapped_axis() {
        check_operation_batching!(
            @exact,
            operation = ReduceOperation::new(vec![0], ReductionKind::Sum),
            axis_size = 3,
            cases = [{
                inputs = [(@mapped(axis = 1), Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![5.0, 7.0, 9.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_reduce_batching_axes_around_mapped_axis() {
        // Per-item axes on both sides of the mapped axis lift independently: axis 0 keeps its physical position while
        // axis 2 shifts past the mapped axis, and the output batch axis moves down past the reduced axis 0.
        check_operation_batching!(
            @exact,
            operation = ReduceOperation::new(vec![0, 2], ReductionKind::Sum),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 1), Array::from_elements::<f64>(
                    ArrayType::new_static(DataType::F64, [2, 2, 2, 2]),
                    &(0..16).map(|index| index as f64).collect::<Vec<_>>(),
                ).unwrap())],
                outputs = [(@mapped(axis = 0), Array::matrix(2, 2, vec![18.0, 26.0, 34.0, 42.0]).unwrap())],
            }],
        );

        // A per-item axis at the mapped axis position names the per-item axis rather than the mapped one, so it
        // shifts past the mapped axis instead of reducing it.
        check_operation_batching!(
            @exact,
            operation = ReduceOperation::new(vec![0, 1], ReductionKind::Sum),
            axis_size = 3,
            cases = [{
                inputs = [(@mapped(axis = 1), Array::from_elements::<f64>(
                    ArrayType::new_static(DataType::F64, [2, 3, 2]),
                    &(0..12).map(|index| index as f64).collect::<Vec<_>>(),
                ).unwrap())],
                outputs = [(@mapped(axis = 0), Array::vector(vec![14.0, 22.0, 30.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_reduce_batching_log_sum_exp() {
        // Physical input is [2 batch items, 2 columns] mapped at axis 0, so the per-item axis 0 reduces physical
        // axis 1 and each batch item is reduced independently.
        check_operation_batching!(
            @approx(epsilon = 1e-12),
            operation = ReduceOperation::new(vec![0], ReductionKind::LogSumExp),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::matrix(2, 2, vec![0.0, 0.0, 1000.0, 1000.0]).unwrap())],
                outputs = [(@mapped(
                    axis = 0
                ), Array::vector(vec![std::f64::consts::LN_2, 1000.0 + std::f64::consts::LN_2]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_reduce_batching_log_sum_exp_replicated_input() {
        check_operation_batching!(
            @exact,
            operation = ReduceOperation::new(vec![0], ReductionKind::LogSumExp),
            axis_size = 2,
            cases = [{
                inputs = [(@replicated, Array::vector(vec![0.0, 0.0]).unwrap())],
                outputs = [(@replicated, Array::scalar(std::f64::consts::LN_2).unwrap())],
            }],
        );
    }

    #[test]
    fn test_reduce_batching_log_sum_exp_reduced_ragged_axis() {
        // Static array batching cannot neutralize ragged padding, and says so rather than summing the padding's
        // exponentials into the live output.
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
        let input = ArrayBatch::new(Array::matrix(2, 3, vec![0.0f32; 6]).unwrap(), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(
                1,
                Array::vector(vec![1i32, 3]).unwrap(),
                variable.clone(),
                vec![0],
            )])
            .unwrap();
        assert_eq!(
            ReduceOperation::new(vec![0], ReductionKind::LogSumExp).batch(
                &BatchingContext::new(EagerContext::<Array>::new(), 2),
                &EmptyRegionDriver,
                &[input],
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "static array batching cannot identity-mask bounded ragged dimension `length` on axis 1 \
                          with `Lowest`"
                    .to_string(),
            }),
        );

        // The composite dynamic policy can, and stages the mask ahead of the reduction: the padded positions of the
        // reduced axis are selected away in favor of negative infinity, whose exponential is the sum's zero identity.
        // The ragged axis is then genuinely consumed, so it leaves the output and is reported as evidence.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_extent = trace.input(DimensionType::from(items.clone()).into());
        let packed = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(items.clone()), Dimension::Static(3)]))
                .into(),
        );
        let extents = trace.input(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(items)])).into());
        let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
            ProjectedContext::new(trace.clone()),
            batch_extent,
        );
        let input = ArrayBatch::new(packed.into_projected().unwrap(), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, extents.into_projected().unwrap(), variable.clone(), vec![0])])
            .unwrap();
        // The per-item reduced axis 0 is the packed axis 1 that carries the ragged extents.
        let (outputs, evidence) = ReduceOperation::new(vec![0], ReductionKind::LogSumExp)
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert!(outputs[0].ragged_axes().is_empty());
        assert_eq!(evidence, vec![variable]);

        let output_id = outputs.into_iter().next().unwrap().into_value().into_value().atom_id().unwrap();
        drop(context);
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output_id],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<items ∈ [1, 9)>, %1:f32[items, 3], %2:i32[items] .
                let %3:dimension<items ∈ [1, 9)> = dimension_size [axis=0] %1
                    %4:dimension<3> = constant [value=3]
                    %5:i32[3] = iota [type=i32[3], dimension=0]
                    %6:i32[items, 3] = broadcast [output_axes=[1]] %5 %3 %4
                    %7:i32[items, 3] = broadcast [output_axes=[0]] %2 %3 %4
                    %8:bool[items, 3] = compare [direction=LessThan] %6 %7
                    %9:f32[] = constant [value=-inf]
                    %10:f32[items, 3] = broadcast [output_axes=[]] %9 %3 %4
                    %11:f32[items, 3] = select %8 %1 %10
                    %12:f32[items] = reduce [kind=log_sum_exp, axes=[1]] %11
                in (%12)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_reduce_differentiation() {
        for kind in [ReductionKind::Max, ReductionKind::Min] {
            check_operation_differentiation!(
                @approx(step = 1e-6, epsilon = 1e-6),
                operation = ReduceOperation::new(vec![0], kind),
                cases = [{
                    primals = [Array::vector(vec![1.0f64, 3.0, 2.0]).unwrap()],
                    tangents = [Array::vector(vec![2.0f64, 4.0, 6.0]).unwrap()],
                    primal_outputs = [Array::scalar(if kind == ReductionKind::Max { 3.0 } else { 1.0 }).unwrap()],
                    tangent_outputs = [Array::scalar(if kind == ReductionKind::Max { 4.0 } else { 2.0 }).unwrap()],
                }],
            );
        }
    }

    #[test]
    fn test_reduce_differentiation_ties() {
        for kind in [ReductionKind::Max, ReductionKind::Min] {
            let input = Array::vector(vec![1.0, 1.0]).unwrap();
            let (primal, tangent) = differentiate_at(input.clone())
                .jvp(Array::vector(vec![1.0, 3.0]).unwrap(), |input| Ok(input.reduce(&[0], kind)?))
                .unwrap();
            assert_eq!(primal.elements::<f64>(), Ok(vec![1.0]));
            assert_eq!(tangent.elements::<f64>(), Ok(vec![2.0]));

            let (primal, gradient) =
                differentiate_at(input).value_and_gradient(|input| Ok(input.reduce(&[0], kind)?)).unwrap();
            assert_eq!(primal.elements::<f64>(), Ok(vec![1.0]));
            assert_eq!(gradient.elements::<f64>(), Ok(vec![0.5, 0.5]));
        }
    }

    #[test]
    fn test_reduce_differentiation_log_sum_exp() {
        // The tangent is the softmax-weighted sum of the input tangents over the reduced axes.
        let primals = [1.0f64, 2.0, 3.0];
        let tangents = [0.5f64, -1.5, 2.0];
        let output = ((1.0f64 - 3.0).exp() + (2.0f64 - 3.0).exp() + 1.0).ln() + 3.0;
        let tangent = primals
            .iter()
            .zip(tangents.iter())
            .map(|(primal, tangent)| (primal - output).exp() * tangent)
            .sum::<f64>();
        check_operation_differentiation!(
            @approx(step = 1e-6, epsilon = 1e-6),
            operation = ReduceOperation::new(vec![0], ReductionKind::LogSumExp),
            cases = [{
                primals = [Array::vector(primals.to_vec()).unwrap()],
                tangents = [Array::vector(tangents.to_vec()).unwrap()],
                primal_outputs = [Array::scalar(output).unwrap()],
                tangent_outputs = [Array::scalar(tangent).unwrap()],
                jvp = indoc! {"
                    lambda %0:f64[3], %1:f64[3] .
                    let %2:f64[] = reduce [kind=log_sum_exp, axes=[0]] %0
                        %3:f64[] = reduce [kind=max, axes=[0]] %0
                        %4:f64[] = zero_like %3
                        %5:f64[] = sub %3 %3
                        %6:bool[] = compare [direction=Equal] %5 %4
                        %7:f64[] = select %6 %3 %4
                        %8:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %7
                        %9:f64[3] = sub %0 %8
                        %10:f64[3] = exp %9
                        %11:f64[] = reduce [kind=sum, axes=[0]] %10
                        %12:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %11
                        %13:f64[3] = div %10 %12
                        %14:f64[3] = mul %13 %1
                        %15:f64[] = reduce [kind=sum, axes=[0]] %14
                    in (%2, %15)
                "},
            }],
        );
    }

    #[test]
    fn test_reduce_differentiation_log_sum_exp_large_offset() {
        // Rounding the primal erases log(2), but must not erase the normalization of the derivative weights.
        let (primal, tangent) = differentiate_at(Array::vector(vec![1e20f64, 1e20]).unwrap())
            .jvp(Array::vector(vec![1.0f64, 1.0]).unwrap(), |input| input.reduce_log_sum_exp(&[0]))
            .unwrap();
        assert_eq!(primal.elements::<f64>(), Ok(vec![1e20]));
        assert_eq!(tangent.elements::<f64>(), Ok(vec![1.0]));
    }

    #[test]
    fn test_reduce_differentiation_log_sum_exp_narrow_floating_point() {
        // The normalization count exceeds the largest finite half value; the derivative must still sum to one.
        let (_, tangent) = differentiate_at(Array::vector(vec![f16::ZERO; 65_536]).unwrap())
            .jvp(Array::vector(vec![f16::ONE; 65_536]).unwrap(), |input| input.reduce_log_sum_exp(&[0]))
            .unwrap();
        assert_eq!(tangent.elements::<f16>(), Ok(vec![f16::ONE]));
    }

    #[test]
    fn test_reduce_differentiation_log_sum_exp_infinity() {
        let (_, pullback) = differentiate_at(Array::vector(vec![f64::INFINITY, 0.0]).unwrap())
            .vjp(|input| input.reduce_log_sum_exp(&[0]))
            .unwrap();
        let elements = pullback.apply(Array::scalar(1.0f64).unwrap()).unwrap().elements::<f64>().unwrap();
        assert!(elements[0].is_nan());
        assert_eq!(elements[1], 0.0);
    }

    #[test]
    fn test_reduce_differentiation_dynamic_reduced_axis() {
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(2)]));
        // Each pullback is traced once and replayed at different concrete extents; extrema also retain their masks.
        for kind in [ReductionKind::Sum, ReductionKind::Mean, ReductionKind::Max, ReductionKind::Min] {
            for (axes, cotangent) in
                [(vec![0], Array::vector(vec![1.0, 2.0]).unwrap()), (vec![0, 1], Array::scalar(3.0).unwrap())]
            {
                let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
                let input = builder.add_input(input_type.clone().into());
                let output = builder
                    .add_instruction(
                        ArrayIrOperation::Array(ArrayOperation::Reduce(ReduceOperation::new(axes.clone(), kind))),
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
                let linearization = program.linearize().unwrap();
                let pullback = linearization.pullback().unwrap();

                for rows in [4usize, 2] {
                    let values = (0..rows * 2).map(|index| index as f64).collect::<Vec<_>>();
                    let mut primal_outputs = linearization
                        .primal()
                        .interpret(vec![ArrayIrValue::Array(Array::matrix(rows, 2, values).unwrap())])
                        .unwrap();
                    let residuals = primal_outputs.split_off(1);
                    let mut pullback_inputs = vec![ArrayIrValue::Array(cotangent.clone())];
                    pullback_inputs.extend(residuals);

                    let cotangent_values = cotangent.elements::<f64>().unwrap();
                    let expected = (0..rows * 2)
                        .map(|index| {
                            let seed = cotangent_values[if axes.len() == 1 { index % 2 } else { 0 }];
                            match kind {
                                ReductionKind::Sum => seed,
                                ReductionKind::Mean => seed / (if axes.len() == 1 { rows } else { rows * 2 }) as f64,
                                ReductionKind::Max => {
                                    if if axes.len() == 1 { index / 2 == rows - 1 } else { index == rows * 2 - 1 } {
                                        seed
                                    } else {
                                        0.0
                                    }
                                }
                                ReductionKind::Min => {
                                    if if axes.len() == 1 { index / 2 == 0 } else { index == 0 } {
                                        seed
                                    } else {
                                        0.0
                                    }
                                }
                                _ => unreachable!(),
                            }
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(
                        pullback.interpret(pullback_inputs),
                        Ok(vec![ArrayIrValue::Array(Array::matrix(rows, 2, expected).unwrap())]),
                    );
                }
            }
        }
    }

    #[test]
    fn test_reduce_differentiation_dynamic_reduced_axis_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])]).unwrap();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(4)]))
                .with_sharding(sharding.clone())
                .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(input_type.into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Reduce(ReduceOperation::new(vec![1], ReductionKind::Max))),
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

        // The extremum mask compares the input against the reduced output broadcast back to the runtime input shape.
        // That broadcast requests the input sharding, just like the static rule's broadcast to the input type, instead
        // of replicating the sharded reduced axis, so that the comparison inputs are identically distributed.
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization
                .primal()
                .instructions()
                .iter()
                .filter_map(|instruction| match instruction.operation() {
                    ArrayIrOperation::Broadcast(operation) => Some(operation.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec![DynamicBroadcastOperation::new(vec![0]).with_output_sharding(sharding)],
        );
    }

    #[test]
    fn test_reduce_differentiation_output_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new_static(DataType::F64, [2, 3])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let unreduced = Sharding::new(mesh, vec![ShardingDimension::replicated()])
            .unwrap()
            .with_unreduced_axes(["x"])
            .unwrap();

        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = context.input(input_type);
        let output = input.reduce_sum(&[0], Some(unreduced.clone())).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![output.atom_id().unwrap()], vec![Placeholder], vec![Placeholder])
            .unwrap();
        // Linearization must preserve the requested sharding on both applications of the linear reduction: the
        // primal reduction and the same reduction applied to the tangent. Otherwise differentiation silently turns
        // a requested per-shard partial sum into the default reduced output.
        let linearization = program.linearize().unwrap();
        let expected_output_type = ArrayType::new_static(DataType::F64, [3]).with_sharding(unreduced.clone()).unwrap();
        assert_eq!(linearization.primal().output_types()[0], expected_output_type);
        assert_eq!(linearization.tangent().output_types()[0], expected_output_type);
        assert_eq!(
            linearization
                .primal()
                .instructions()
                .iter()
                .filter_map(|instruction| match instruction.operation() {
                    ArrayOperation::Reduce(operation) => Some(operation.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec![ReduceOperation::new(vec![0], ReductionKind::Sum).with_output_sharding(unreduced.clone()).unwrap()],
        );
        assert_eq!(
            linearization
                .tangent()
                .instructions()
                .iter()
                .filter_map(|instruction| match instruction.operation() {
                    ArrayOperation::Reduce(operation) => Some(operation.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec![ReduceOperation::new(vec![0], ReductionKind::Sum).with_output_sharding(unreduced.clone()).unwrap()],
        );
    }

    #[test]
    fn test_reduce_transposition_dynamic_reduced_axis() {
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(2)]));

        // Direct transposition observes no primal value, so the reduced axis's runtime extent is unavailable and the
        // replication cannot be staged. Both additive kinds report that instead of failing inside broadcast inference.
        for kind in [ReductionKind::Sum, ReductionKind::Mean] {
            let context = TracingContext::<Array, ArrayOperation<Array>>::new();
            let output_cotangent = {
                let atom = context
                    .builder()
                    .borrow_mut()
                    .add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])));
                context.tracer(atom, None)
            };
            let inputs = [PartialValue::Unknown(input_type.clone())];
            let mut transposition = TranspositionContext::new(context.clone());
            let accumulators = transposition.cotangent_accumulators(&inputs, &[]).unwrap();
            assert!(matches!(
                ReduceOperation::new(vec![0], kind).transpose(
                    &mut transposition,
                    &EmptyRegionDriver,
                    &inputs,
                    &[MaybeZero::Value(output_cotangent)],
                    &accumulators,
                ),
                Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                    if message == format!(
                        "direct transposition of `reduce` with kind `{kind}` over reduced axis 0 of [batch, 2] \
                         requires linearization so that the runtime extent can be retained as a residual",
                    ),
            ));
        }
    }

    #[test]
    fn test_reduce_transposition_mean() {
        check_operation_transposition!(
            @exact,
            operation = ReduceOperation::new(vec![0], ReductionKind::Mean),
            cases = [{
                inputs = [(@linear(type = ArrayType::new_static(DataType::F64, [4])))],
                output_cotangents = [Array::scalar(1.0).unwrap()],
                input_cotangents = [Array::vector(vec![0.25; 4]).unwrap()],
            }],
        );
    }

    #[test]
    fn test_reduce_transposition_mean_reduced_element_count() {
        use crate::differentiation::DifferentiationError;
        use crate::partial::PartialValue;
        use crate::programs::{MaybeZero, ProgramError, TypeError};
        use crate::tracing::TracingContext;

        let input_shape = Shape::new(vec![Dimension::Static(usize::MAX), Dimension::Static(2)]);
        let input_type = ArrayType::new(DataType::F64, input_shape.clone());
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = {
            let atom = context.builder().borrow_mut().add_input(ArrayType::scalar(DataType::F64));
            context.tracer(atom, None)
        };

        let inputs = [PartialValue::Unknown(input_type.clone())];
        let mut transposition = TranspositionContext::new(context.clone());
        let accumulators = transposition.cotangent_accumulators(&inputs, &[]).unwrap();
        assert!(matches!(
            ReduceOperation::new(vec![0, 1], ReductionKind::Mean).transpose(
                &mut transposition,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Value(output_cotangent)],
                &accumulators,
            ),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == format!(
                    "mean transpose reduced element count overflows `usize` for input shape `{input_shape}`",
                ),
        ));
    }

    #[test]
    fn test_reduce_transposition_mean_empty_reduction() {
        use crate::partial::PartialValue;
        use crate::programs::MaybeZero;
        use crate::tracing::TracingContext;

        let input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(usize::MAX), Dimension::Static(2), Dimension::Static(0)]),
        );
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = {
            let atom = context.builder().borrow_mut().add_input(ArrayType::scalar(DataType::F64));
            context.tracer(atom, None)
        };

        let contributions = {
            let mut context = TranspositionContext::new(context.clone());
            let inputs = &[PartialValue::Unknown(input_type.clone())];
            let accumulators = context.cotangent_accumulators(inputs, &[]).unwrap();
            ReduceOperation::new(vec![0, 1, 2], ReductionKind::Mean)
                .transpose(
                    &mut context,
                    &EmptyRegionDriver,
                    inputs,
                    &[MaybeZero::Value(output_cotangent)],
                    &accumulators,
                )
                .unwrap();
            context.take_cotangents(&accumulators).unwrap()
        };
        assert_eq!(contributions.len(), 1);
        assert_eq!(contributions[0].r#type().as_ref(), &input_type);
    }

    #[test]
    fn test_reduce_transposition_log_sum_exp() {
        check_operation_transposition!(
            @rejected,
            operation = ReduceOperation::new(vec![0], ReductionKind::LogSumExp),
            input_types = [ArrayType::new_static(DataType::F64, [3])],
        );
    }

    #[test]
    fn test_array_reduce() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(matrix.reduce(&[1], ReductionKind::Sum).unwrap(), Array::vector(vec![6.0, 15.0]).unwrap());
        assert_eq!(matrix.reduce(&[1], ReductionKind::Mean).unwrap(), Array::vector(vec![2.0, 5.0]).unwrap());
        assert_eq!(
            matrix.reduce(&[0, 1], ReductionKind::Sum).unwrap(),
            Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, []), &[21.0]).unwrap(),
        );
        assert_eq!(matrix.reduce(&[], ReductionKind::Sum).unwrap(), matrix);
        // Max and min use the data type's reduction identities and ordinary ordering.
        let integers = Array::vector(vec![3i32, -1, 2]).unwrap();
        assert_eq!(integers.reduce(&[0], ReductionKind::Max).unwrap().elements::<i32>(), Ok(vec![3]));
        assert_eq!(integers.reduce(&[0], ReductionKind::Min).unwrap().elements::<i32>(), Ok(vec![-1]));
        // Boolean reductions.
        let booleans = Array::vector(vec![true, false, true]).unwrap();
        assert_eq!(booleans.reduce(&[0], ReductionKind::Any).unwrap().elements::<bool>(), Ok(vec![true]));
        assert_eq!(booleans.reduce(&[0], ReductionKind::All).unwrap().elements::<bool>(), Ok(vec![false]));
        assert_eq!(booleans.reduce(&[0], ReductionKind::Max).unwrap().elements::<bool>(), Ok(vec![true]));
        assert_eq!(booleans.reduce(&[0], ReductionKind::Min).unwrap().elements::<bool>(), Ok(vec![false]));
    }

    #[test]
    fn test_array_reduce_layouts() {
        // Numeric and Boolean reductions traverse arbitrary layouts and produce the abstract rule's dense output.
        let r#type =
            ArrayType::new_static(DataType::U16, [2, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![-8, 2])));
        let matrix = Array::from_elements(r#type, &[1u16, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(matrix.reduce(&[1], ReductionKind::Sum).unwrap().elements::<u16>(), Ok(vec![6, 15]));
        let r#type =
            ArrayType::new_static(DataType::Boolean, [3]).with_layout(Layout::Strided(StridedLayout::new(vec![-1])));
        let booleans = Array::from_elements(r#type, &[true, false, true]).unwrap();
        assert_eq!(booleans.reduce(&[0], ReductionKind::Any).unwrap().elements::<bool>(), Ok(vec![true]));
    }

    #[test]
    fn test_array_reduce_narrow_elements() {
        // Sub-byte accumulation wraps in the declared width, and floating-point outputs retain their declared format.
        let narrow = Array::matrix(
            2,
            2,
            vec![i4::new(7).unwrap(), i4::new(2).unwrap(), i4::new(-8).unwrap(), i4::new(-3).unwrap()],
        )
        .unwrap();
        assert_eq!(
            narrow.reduce(&[1], ReductionKind::Sum).unwrap().elements::<i4>(),
            Ok(vec![i4::new(-7).unwrap(), i4::new(5).unwrap()]),
        );
        let low_precision =
            Array::vector(vec![f8e4m3fn::from_f64(1.0).unwrap(), f8e4m3fn::from_f64(0.5).unwrap()]).unwrap();
        assert_eq!(
            low_precision.reduce(&[0], ReductionKind::Sum).unwrap().elements::<f8e4m3fn>(),
            Ok(vec![f8e4m3fn::from_f64(1.5).unwrap()]),
        );
    }

    #[test]
    fn test_array_reduce_complex() {
        // Complex sums and means preserve both components, while empty sums materialize the numeric identity.
        let complex = Array::vector(vec![ComplexNumber::new(2.0f32, 4.0), ComplexNumber::new(4.0, 8.0)]).unwrap();
        assert_eq!(
            complex.reduce(&[0], ReductionKind::Sum).unwrap().elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(6.0, 12.0)]),
        );
        assert_eq!(
            complex.reduce(&[0], ReductionKind::Mean).unwrap().elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(3.0, 6.0)]),
        );
        let empty = Array::from_elements::<i32>(ArrayType::new_static(DataType::I32, [2, 0]), &[]).unwrap();
        assert_eq!(empty.reduce(&[1], ReductionKind::Sum).unwrap().elements::<i32>(), Ok(vec![0, 0]));
    }

    #[test]
    fn test_array_reduce_floating_point_extrema() {
        // Floating-point extrema propagate NaNs and order negative zero below positive zero.
        let nan = Array::vector(vec![1.0f32, f32::NAN]).unwrap();
        assert!(nan.reduce(&[0], ReductionKind::Max).unwrap().elements::<f32>().unwrap()[0].is_nan());
        let zeros = Array::vector(vec![-0.0f32, 0.0]).unwrap();
        assert_eq!(
            zeros.reduce(&[0], ReductionKind::Max).unwrap().elements::<f32>().unwrap()[0].to_bits(),
            0.0f32.to_bits(),
        );
        assert_eq!(
            zeros.reduce(&[0], ReductionKind::Min).unwrap().elements::<f32>().unwrap()[0].to_bits(),
            (-0.0f32).to_bits(),
        );
    }

    #[test]
    fn test_array_reduce_complex_extrema() {
        // Complex extrema compare `(real, imaginary)` lexicographically, with true lexicographic identities.
        let complex = Array::vector(vec![
            ComplexNumber::new(1.0f32, 5.0),
            ComplexNumber::new(2.0, -3.0),
            ComplexNumber::new(2.0, 4.0),
        ])
        .unwrap();
        assert_eq!(
            complex.reduce(&[0], ReductionKind::Max).unwrap().elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(2.0, 4.0)]),
        );
        assert_eq!(
            complex.reduce(&[0], ReductionKind::Min).unwrap().elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(1.0, 5.0)]),
        );
        let empty =
            Array::from_elements::<ComplexNumber<f32>>(ArrayType::new_static(DataType::C64, [2, 0]), &[]).unwrap();
        assert_eq!(
            empty.reduce(&[1], ReductionKind::Max).unwrap().elements::<ComplexNumber<f32>>(),
            Ok(vec![
                ComplexNumber::new(f32::NEG_INFINITY, f32::NEG_INFINITY),
                ComplexNumber::new(f32::NEG_INFINITY, f32::NEG_INFINITY),
            ]),
        );
        assert_eq!(
            empty.reduce(&[1], ReductionKind::Min).unwrap().elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(f32::INFINITY, f32::INFINITY); 2]),
        );
        let lower = Array::vector(vec![ComplexNumber::new(f32::NEG_INFINITY, -1.0)]).unwrap();
        assert_eq!(
            lower.reduce(&[0], ReductionKind::Max),
            Ok(Array::scalar(ComplexNumber::new(f32::NEG_INFINITY, -1.0)).unwrap()),
        );
        let upper = Array::vector(vec![ComplexNumber::new(f32::INFINITY, 1.0)]).unwrap();
        assert_eq!(
            upper.reduce(&[0], ReductionKind::Min),
            Ok(Array::scalar(ComplexNumber::new(f32::INFINITY, 1.0)).unwrap()),
        );
    }

    #[test]
    fn test_array_reduce_finite_extrema_identities() {
        let empty = Array::from_elements::<f8e8m0fnu>(ArrayType::new_static(DataType::F8E8M0FNU, [2, 0]), &[]).unwrap();
        assert_eq!(
            empty.reduce(&[1], ReductionKind::Max).unwrap().elements::<f8e8m0fnu>(),
            Ok(vec![f8e8m0fnu::MIN, f8e8m0fnu::MIN]),
        );
        assert_eq!(
            empty.reduce(&[1], ReductionKind::Min).unwrap().elements::<f8e8m0fnu>(),
            Ok(vec![f8e8m0fnu::MAX, f8e8m0fnu::MAX]),
        );
    }

    #[test]
    fn test_array_reduce_half_precision_accumulation() {
        // Accumulate and divide before converting back to the output format, including counts too large for f16.
        let ones = Array::vector(vec![f16::ONE; 4096]).unwrap();
        assert_eq!(ones.reduce(&[0], ReductionKind::Sum).unwrap().elements::<f16>(), Ok(vec![f16::from_f32(4096.0)]));
        assert_eq!(ones.reduce(&[0], ReductionKind::Mean).unwrap().elements::<f16>(), Ok(vec![f16::ONE]));
        let ones = Array::vector(vec![f16::ONE; 70_000]).unwrap();
        assert_eq!(ones.reduce(&[0], ReductionKind::Mean).unwrap().elements::<f16>(), Ok(vec![f16::ONE]));
    }

    #[test]
    fn test_array_reduce_empty_mean() {
        let empty = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0]), &[]).unwrap();
        assert!(empty.reduce(&[0], ReductionKind::Mean).unwrap().elements::<f32>().unwrap()[0].is_nan());
    }

    #[test]
    fn test_reduce_empty_axes_data_types() {
        for (input, kind, message) in [
            (
                Array::vector(vec![true]).unwrap(),
                ReductionKind::Sum,
                "`reduce` with kind `sum` requires numeric inputs but got `bool`",
            ),
            (
                Array::vector(vec![1i32]).unwrap(),
                ReductionKind::Any,
                "`reduce` with kind `any` requires Boolean inputs but got `i32`",
            ),
        ] {
            assert_eq!(input.reduce(&[], kind), Err(ProgramError::Type(TypeError::invalid(message.to_string()))));
            let context = TracingContext::<Array, ArrayOperation<Array>>::new();
            let input = context.input(input.r#type().into_owned());
            assert_eq!(input.reduce(&[], kind), Err(ProgramError::Type(TypeError::invalid(message.to_string()))));
        }
    }

    #[test]
    fn test_reduce_reduce_sum() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new_static(DataType::F64, [2, 3])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let unreduced = Sharding::new(mesh, vec![ShardingDimension::replicated()])
            .unwrap()
            .with_unreduced_axes(["x"])
            .unwrap();

        // Without a requested sharding, `reduce_sum` is an ordinary sum reduction.
        let input = Array::from_elements::<f64>(input_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(input.reduce_sum(&[0], None), input.reduce(&[0], ReductionKind::Sum));

        // A requested sharding is carried through the staged `ReduceOperation` into the built program.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output = context.input(input_type).reduce_sum(&[0], Some(unreduced)).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![output.atom_id().unwrap()], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 3][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}] .
                let %1:f64[3][sharding={mesh<['x'=2:explicit]>, [{}], unreduced={'x'}}] = reduce [
                    kind=sum,
                    axes=[0],
                    output_sharding={mesh<['x'=2:explicit]>, [{}], unreduced={'x'}},
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_reduce_reduce_sum_concrete_arrays() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new_static(DataType::F64, [2, 3])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let input = Array::from_elements::<f64>(input_type, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let unreduced = Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
            .unwrap()
            .with_unreduced_axes(["x"])
            .unwrap();

        // A valid request computes the same elements as an unsharded sum, because a concrete array lives on a single
        // device, while its output type records the requested sharding exactly like the output type of a staged sum.
        // An invalid request is rejected just like it is for staged sums.
        let output = input.reduce_sum(&[0], Some(unreduced.clone())).unwrap();
        assert_eq!(
            output.r#type().as_ref(),
            &ArrayType::new_static(DataType::F64, [3]).with_sharding(unreduced).unwrap()
        );
        assert_eq!(output.elements::<f64>(), Ok(vec![5.0, 7.0, 9.0]));
        assert_eq!(
            input.reduce_sum(&[0], Some(Sharding::replicated(mesh, 2))),
            Err(ProgramError::Type(TypeError::invalid(
                "`reduce` output sharding rank (2) does not match the output rank (1)",
            ))),
        );
    }

    #[test]
    fn test_reduce_reduce_mean() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(matrix.reduce_mean(&[1]), Ok(Array::vector(vec![2.0, 5.0]).unwrap()));
    }

    #[test]
    fn test_reduce_reduce_log_sum_exp() {
        // The expected values below spell out the guarded construction the primitive documents (shift by the safe
        // maximum, sum the exponentials, take the logarithm, add the shift back) so that they pin that construction
        // rather than an equivalent-in-exact-arithmetic alternative.
        let values = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        let expected = ((1.0f64 - 3.0).exp() + (2.0f64 - 3.0).exp() + 1.0).ln() + 3.0;
        assert_eq!(values.reduce_log_sum_exp(&[0]), Ok(Array::scalar(expected).unwrap()));

        // Reducing along no axes is the identity, matching `log(exp(x)) = x`, but only for the inputs the staged
        // operation accepts: the shortcut still validates the element data type.
        assert_eq!(values.reduce_log_sum_exp(&[]), Ok(values.clone()));

        // Equal inputs keep both shifted exponentials at one even when the naive composition would overflow.
        assert_eq!(
            Array::vector(vec![0.0, 0.0]).unwrap().reduce_log_sum_exp(&[0]),
            Ok(Array::scalar(std::f64::consts::LN_2).unwrap()),
        );
        assert_eq!(
            Array::vector(vec![1000.0, 1000.0]).unwrap().reduce_log_sum_exp(&[0]),
            Ok(Array::scalar(1000.0 + std::f64::consts::LN_2).unwrap()),
        );

        // The guard's reason to exist: an all-`-∞` slice and an empty reduction both pin to `-∞` (`log(0) + 0`)
        // instead of the `-∞ - -∞ = NaN` that shifting by the raw maximum would produce.
        assert_eq!(
            Array::vector(vec![f64::NEG_INFINITY, f64::NEG_INFINITY]).unwrap().reduce_log_sum_exp(&[0]),
            Ok(Array::scalar(f64::NEG_INFINITY).unwrap()),
        );
        assert_eq!(
            Array::new(ArrayType::new_static(DataType::F64, [0]), Vec::new()).unwrap().reduce_log_sum_exp(&[0]),
            Ok(Array::scalar(f64::NEG_INFINITY).unwrap()),
        );

        // A `+∞` element saturates the output, and NaN propagates.
        assert_eq!(
            Array::vector(vec![1.0, f64::INFINITY]).unwrap().reduce_log_sum_exp(&[0]),
            Ok(Array::scalar(f64::INFINITY).unwrap()),
        );
        assert!(
            Array::vector(vec![1.0, f64::NAN])
                .unwrap()
                .reduce_log_sum_exp(&[0])
                .unwrap()
                .elements::<f64>()
                .unwrap()[0]
                .is_nan(),
        );

        // Reducing one axis of a matrix leaves the other, in order.
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(
            matrix.reduce_log_sum_exp(&[1]),
            Ok(Array::vector(vec![expected, ((-2.0f64).exp() + (-1.0f64).exp() + 1.0).ln() + 6.0]).unwrap()),
        );

        // Validation errors are reported rather than panicking.
        assert_eq!(
            values.reduce_log_sum_exp(&[1]),
            Err(ProgramError::Type(TypeError::invalid("`reduce` axis 1 is out of bounds for rank 1".to_string()))),
        );
        assert_eq!(
            Array::vector(vec![1i32, 2]).unwrap().reduce_log_sum_exp(&[0]),
            Err(ProgramError::Type(TypeError::invalid(
                "`reduce` with kind `log_sum_exp` requires real floating-point inputs but got `i32`".to_string(),
            ))),
        );
    }

    #[test]
    fn test_reduce_reduce_log_sum_exp_half_precision_accumulation() {
        let zeros = Array::vector(vec![f16::ZERO; 4096]).unwrap();
        assert_eq!(zeros.reduce_log_sum_exp(&[0]).unwrap().elements::<f16>(), Ok(vec![f16::from_f32(8.3203125)]));
    }

    #[test]
    fn test_reduce_reduce_log_sum_exp_empty_axes() {
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        for input in [
            Array::vector(vec![1i32, 2]).unwrap(),
            Array::new(ArrayType::new_static(DataType::Zero, [2]), Vec::new()).unwrap(),
        ] {
            let expected = Err(ProgramError::Type(TypeError::invalid(format!(
                "`reduce` with kind `log_sum_exp` requires real floating-point inputs but got `{}`",
                input.r#type().data_type(),
            ))));
            assert_eq!(input.reduce(&[], ReductionKind::LogSumExp), expected);
            assert_eq!(input.reduce_log_sum_exp(&[]), expected);
            let input_atom = context.builder().borrow_mut().add_input(input.r#type().into_owned());
            let input = context.tracer(input_atom, None);
            let expected = Err(ProgramError::Type(TypeError::invalid(format!(
                "`reduce` with kind `log_sum_exp` requires real floating-point inputs but got `{}`",
                input.r#type().data_type(),
            ))));
            assert_eq!(input.reduce(&[], ReductionKind::LogSumExp), expected);
            assert_eq!(input.reduce_log_sum_exp(&[]), expected);
        }
    }

    #[test]
    fn test_reduce_reduce_max() {
        let matrix = Array::matrix(2, 3, vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]).unwrap();
        assert_eq!(matrix.reduce_max(&[0]), Ok(Array::vector(vec![4.0, 5.0, 6.0]).unwrap()));
    }

    #[test]
    fn test_reduce_reduce_min() {
        let matrix = Array::matrix(2, 3, vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]).unwrap();
        assert_eq!(matrix.reduce_min(&[0]), Ok(Array::vector(vec![1.0, 2.0, 3.0]).unwrap()));
    }

    #[test]
    fn test_reduce_reduce_any() {
        let matrix = Array::matrix(2, 3, vec![false, true, false, false, false, false]).unwrap();
        assert_eq!(matrix.reduce_any(&[1]), Ok(Array::vector(vec![true, false]).unwrap()));
    }

    #[test]
    fn test_reduce_reduce_all() {
        let matrix = Array::matrix(2, 3, vec![true, true, true, true, false, true]).unwrap();
        assert_eq!(matrix.reduce_all(&[1]), Ok(Array::vector(vec![true, false]).unwrap()));
    }

    #[test]
    fn test_reduce_abstract() {
        let input = ArrayType::new_static(DataType::F64, [2, 3, 4]);
        assert_eq!(reduce_abstract(&input, &[1], ReductionKind::Sum), Ok(ArrayType::new_static(DataType::F64, [2, 4])));
        assert_eq!(reduce_abstract(&input, &[0, 2], ReductionKind::Max), Ok(ArrayType::new_static(DataType::F64, [3])));
    }

    #[test]
    fn test_reduce_abstract_drops_sharded_reduced_axis_entries() {
        use crate::arrays::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};

        // Reducing over a sharded dimension deletes its entry without error (the partitioner owns the collective);
        // the surviving dimension keeps its sharding and the reduced manual axis set passes through.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("r", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let input = ArrayType::new_static(DataType::F64, [2, 3])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap()
                    .with_reduced_axes(["r"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            reduce_abstract(&input, &[0], ReductionKind::Sum),
            Ok(ArrayType::new_static(DataType::F64, [3])
                .with_sharding(
                    Sharding::new(mesh, vec![ShardingDimension::replicated()])
                        .unwrap()
                        .with_reduced_axes(["r"])
                        .unwrap(),
                )
                .unwrap()),
        );
    }

    #[test]
    fn test_reduce_abstract_propagates_dynamic_dimensions() {
        // Dynamic dimensions flow through reduce inference: reduced axes are dropped whether they are static or
        // dynamic, and the remaining dynamic dimensions are preserved in order.
        let batch = DimensionVariable::new("batch", DimensionBounds::unbounded());
        let width = DimensionVariable::new("width", DimensionBounds::non_negative(Some(4)).unwrap());
        let input = ArrayType::new(
            DataType::F64,
            Shape::new(vec![
                Dimension::Dynamic(batch.clone()),
                Dimension::Static(3),
                Dimension::Dynamic(width.clone()),
            ]),
        );
        assert_eq!(
            reduce_abstract(&input, &[1], ReductionKind::Sum),
            Ok(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Dynamic(width)]))),
        );
        assert_eq!(
            reduce_abstract(&input, &[0, 2], ReductionKind::Sum),
            Ok(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]))),
        );
    }

    #[test]
    fn test_reduce_abstract_rejects_out_of_bounds_and_duplicate_axes() {
        let input = ArrayType::new_static(DataType::F64, [2, 3]);
        assert_eq!(
            reduce_abstract(&input, &[2], ReductionKind::Sum),
            Err(TypeError::invalid("`reduce` axis 2 is out of bounds for rank 2".to_string())),
        );
        assert_eq!(
            reduce_abstract(&input, &[0, 0], ReductionKind::Sum),
            Err(TypeError::invalid("`reduce` contains duplicate axis 0".to_string())),
        );
    }

    #[test]
    fn test_reduce_abstract_enforces_reduction_data_types() {
        let numeric = ArrayType::new_static(DataType::F64, [2, 3]);
        assert_eq!(
            reduce_abstract(&numeric, &[1], ReductionKind::Any),
            Err(TypeError::invalid("`reduce` with kind `any` requires Boolean inputs but got `f64`".to_string())),
        );
        let boolean = ArrayType::new_static(DataType::Boolean, [2, 3]);
        assert_eq!(
            reduce_abstract(&boolean, &[1], ReductionKind::Sum),
            Err(TypeError::invalid("`reduce` with kind `sum` requires numeric inputs but got `bool`".to_string())),
        );
        assert_eq!(
            reduce_abstract(&boolean, &[1], ReductionKind::Any),
            Ok(ArrayType::new_static(DataType::Boolean, [2])),
        );
        assert_eq!(
            reduce_abstract(&boolean, &[1], ReductionKind::Max),
            Ok(ArrayType::new_static(DataType::Boolean, [2])),
        );
        let token = ArrayType::new_static(DataType::Token, [2, 3]);
        assert_eq!(
            reduce_abstract(&token, &[1], ReductionKind::Sum),
            Err(TypeError::invalid("`reduce` with kind `sum` requires numeric inputs but got `token`".to_string())),
        );
        // The structural-zero element type represents an already-known zero tangent and remains closed under numeric
        // reductions even though it has no numeric payload bytes.
        let zero = ArrayType::new_static(DataType::Zero, [2, 3]);
        assert_eq!(reduce_abstract(&zero, &[1], ReductionKind::Sum), Ok(ArrayType::new_static(DataType::Zero, [2])));
    }

    #[test]
    fn test_reduce_abstract_accepts_lexicographic_complex_extrema() {
        // Complex minimum and maximum use JAX's lexicographic `(real, imaginary)` ordering.
        let complex = ArrayType::new_static(DataType::C64, [2, 3]);
        assert_eq!(reduce_abstract(&complex, &[1], ReductionKind::Max), Ok(ArrayType::new_static(DataType::C64, [2])));
        assert_eq!(reduce_abstract(&complex, &[1], ReductionKind::Min), Ok(ArrayType::new_static(DataType::C64, [2])));
        assert_eq!(reduce_abstract(&complex, &[1], ReductionKind::Sum), Ok(ArrayType::new_static(DataType::C64, [2])));
    }

    #[test]
    fn test_reduce_abstract_log_sum_exp() {
        // The reduced axes are dropped and the remaining axes keep their order.
        let input = ArrayType::new_static(DataType::F64, [2, 3, 4]);
        assert_eq!(
            reduce_abstract(&input, &[1], ReductionKind::LogSumExp),
            Ok(ArrayType::new_static(DataType::F64, [2, 4])),
        );
        assert_eq!(
            reduce_abstract(&input, &[0, 2], ReductionKind::LogSumExp),
            Ok(ArrayType::new_static(DataType::F64, [3])),
        );
        assert_eq!(reduce_abstract(&input, &[], ReductionKind::LogSumExp), Ok(input.clone()));

        // Axis validation mirrors the reduction family's.
        assert_eq!(
            reduce_abstract(&input, &[3], ReductionKind::LogSumExp),
            Err(TypeError::invalid("`reduce` axis 3 is out of bounds for rank 3".to_string())),
        );
        assert_eq!(
            reduce_abstract(&input, &[1, 1], ReductionKind::LogSumExp),
            Err(TypeError::invalid("`reduce` contains duplicate axis 1".to_string())),
        );

        // Only real floating-point payloads have the exponential and logarithm this primitive is built from.
        for data_type in [DataType::I32, DataType::Boolean, DataType::C64, DataType::Token, DataType::Zero] {
            assert_eq!(
                reduce_abstract(&ArrayType::new_static(data_type, [2, 3]), &[1], ReductionKind::LogSumExp),
                Err(TypeError::invalid(format!(
                    "`reduce` with kind `log_sum_exp` requires real floating-point inputs but got `{data_type}`"
                ))),
            );
        }

        // Max-shifted padding requires true negative infinity, even for finite pairwise identities.
        for data_type in [
            DataType::F8E8M0FNU,
            DataType::F6E2M3FN,
            DataType::F4E2M1FN,
            DataType::F8E4M3B11FNUZ,
            DataType::F6E3M2FN,
            DataType::F8E4M3FN,
            DataType::F8E4M3FNUZ,
            DataType::F8E5M2FNUZ,
        ] {
            assert_eq!(
                reduce_abstract(&ArrayType::new_static(data_type, [2, 3]), &[1], ReductionKind::LogSumExp),
                Err(TypeError::invalid(format!(
                    "`reduce` with kind `log_sum_exp` requires a floating-point format that represents negative \
                     infinity but got `{data_type}`",
                ))),
            );
        }

        // Reducing over a sharded dimension deletes its entry without error (the partitioner owns the collective),
        // and the surviving dimension keeps its sharding.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded = ArrayType::new_static(DataType::F64, [2, 3])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            reduce_abstract(&sharded, &[0], ReductionKind::LogSumExp),
            Ok(ArrayType::new_static(DataType::F64, [3])
                .with_sharding(Sharding::new(mesh, vec![ShardingDimension::replicated()]).unwrap())
                .unwrap()),
        );
    }

    #[test]
    fn test_output_to_input_axis_map() {
        // Input rank 3, reduce axis 1: output axes [0, 1] map back to input axes [0, 2].
        assert_eq!(super::output_to_input_axis_map(3, &[1]), vec![0, 2]);
        // Input rank 3, reduce axes [0, 2]: output axis [0] maps back to input axis [1].
        assert_eq!(super::output_to_input_axis_map(3, &[0, 2]), vec![1]);
        // Input rank 4, reduce axes [1, 3]: output axes [0, 1] map back to input axes [0, 2].
        assert_eq!(super::output_to_input_axis_map(4, &[1, 3]), vec![0, 2]);
        // No reduction: identity map.
        assert_eq!(super::output_to_input_axis_map(3, &[]), vec![0, 1, 2]);
    }
}
