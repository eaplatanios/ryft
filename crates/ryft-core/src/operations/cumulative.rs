//! Operations that accumulate array elements along one axis (i.e., inclusive prefix scans). Every cumulative operation
//! is defined by the [`CumulativeOperation`] type together with the [`Cumulative`] value capability trait, whose
//! functions apply it to eager [`Array`]s and traced values alike, so the same code executes immediately or records
//! into a program depending on the value it runs on. A [`CumulativeKind`] selects the associative combining operator:
//!
//!   - **Arithmetic:** [`Sum`](CumulativeKind::Sum) and [`Product`](CumulativeKind::Product) add and multiply the
//!     accumulated elements.
//!   - **Extrema:** [`Max`](CumulativeKind::Max) and [`Min`](CumulativeKind::Min) select the largest and smallest
//!     accumulated elements.
//!   - **Logarithmic Sums of Exponentials:** [`LogSumExp`](CumulativeKind::LogSumExp) computes a running
//!     `log(sum(exp(x)))` without overflowing for large finite inputs.
//!
//! Element `i` along the scanned axis of the output holds the combination of the input elements `0..=i`
//! (i.e., an inclusive prefix), or of the input elements `i..` (i.e., an inclusive suffix, still in the
//! original output order) when the scan runs in reverse. The output has exactly the input type, so unlike
//! a [`ReduceOperation`](crate::ReduceOperation), a cumulative operation keeps the scanned axis, and unlike the
//! control-flow [`ScanOperation`](crate::ScanOperation), it applies one fixed combining operator rather than an
//! arbitrary loop body. The scanned dimension must be static and unsharded, the non-linear kinds reject unreduced
//! inputs, and floating-point reassociation can make a sequential eager scan and a compiled parallel scan round
//! differently in their last bits.
//!
//! # Batching
//!
//! The identity of each combining operator matters beyond the empty-prefix case: it is what the batching rule writes
//! over the padding of a bounded ragged scanned axis (refer to [`RaggedMaskIdentity`]), so that padded positions cannot
//! contribute to any live prefix. Masking is where cumulative batching parts company with a reduction's: a prefix scan
//! keeps every axis it touches, so it _consumes_ no bounded ragged axis. The input's ragged axes ride through onto
//! the result unchanged and the rule reports no consumption evidence. A scan whose axis is not ragged never reaches the
//! masking hook at all, and so passes through exactly as it would with no ragged metadata.
//!
//! # Differentiation
//!
//! [`CumulativeKind::Sum`] is the one _linear_ kind, so a cumulative sum differentiates and transposes as itself: its
//! tangent is the same prefix sum of the input tangent, and its transpose is the prefix sum in the opposite direction.
//! The non-linear kinds have no closed-form primitive derivative and none is invented for them: their forward-mode
//! rules differentiate through the parallel-prefix [`associative_scan`] decomposition, so each derivative is assembled
//! from the rules of the primitives that construction stages. None of them is directly transposable, and reverse mode
//! reaches them by transposing those staged primitives instead.
//!
//! # Example
//!
//! ```rust
//! # use ryft_core::{Array, Cumulative, CumulativeKind, ProgramError};
//! # fn main() -> Result<(), ProgramError> {
//! let input = Array::vector(vec![1.0, 3.0, 2.0])?;
//! assert_eq!(input.cumulative_sum(0)?, Array::vector(vec![1.0, 4.0, 6.0])?);
//! assert_eq!(input.reverse_cumulative_sum(0)?, Array::vector(vec![6.0, 5.0, 2.0])?);
//! assert_eq!(input.cumulative(0, CumulativeKind::Max, false)?, Array::vector(vec![1.0, 3.0, 3.0])?);
//! # Ok(())
//! # }
//! ```

use std::borrow::Cow;
use std::fmt::Display;

use num_complex::Complex;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayElement, ArrayType, DataType, Dimension, FloatingPointArrayElement,
    NumericArrayElement, RaggedArrayExtentBatchingPolicy, RaggedMaskIdentity, ShardingDimension,
};
use crate::axes::Axis;
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{DifferentiableType, DifferentiationDual, DifferentiationError};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, dispatch_on_array_element_type, impl_differentiable_operation};
use crate::operations::arithmetic::{AddOperation, Mul, MulOperation};
use crate::operations::collectives::parallel_vary::ParallelVaryOperation;
use crate::operations::constants::zero::ZeroOperation;
use crate::operations::control_flow::associative_scan::associative_scan;
use crate::operations::exponential::{LOG_ADD_EXP_OPERATION_NAME, LogAddExp, LogAddExpOperation};
use crate::operations::extrema::{Max, MaxOperation, Min, MinOperation};
use crate::operations::logical::OrOperation;
use crate::operations::manipulation::broadcasting::{Broadcast, BroadcastOperation};
use crate::operations::manipulation::concatenation::ConcatenateOperation;
use crate::operations::manipulation::padding::PadOperation;
use crate::operations::manipulation::slicing::SliceOperation;
use crate::operations::reductions::multiply_product_elements;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, OperationProvider, ProgramError, ProvenanceScope, RegionInterface,
    TypeError, Typed, Value,
};
use crate::tracing::{Tracer, TracingContext};

/// Name of [`CumulativeOperation`]. The operation's [`CumulativeKind`] is rendered as its `kind` attribute.
pub const CUMULATIVE_OPERATION_NAME: &str = "cumulative";

/// Associative combining operator of a [`CumulativeOperation`]. Every kind accumulates in the input's own element data
/// type, so each partial result is rounded to the input's element encoding (i.e., there is no widened accumulation),
/// and each kind has an identity that neutralizes the padding of a bounded ragged scanned axis.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CumulativeKind {
    /// Prefix sum. The identity is `0` and the combiner is addition. Real and complex numeric inputs are supported,
    /// as is the payload-free structural zero, every prefix sum of which is again zero. Integer inputs wrap in their
    /// element data type. This is the one linear kind, so its adjoint is the prefix sum in the opposite direction.
    Sum,

    /// Prefix product. The identity is `1` and the combiner is multiplication. Real and complex numeric inputs are
    /// supported, as is the payload-free structural zero, every prefix product of which is again zero. Integer inputs
    /// wrap in their element data type. Note that, like [`ReductionKind::Product`](crate::ReductionKind::Product), a
    /// complex factor that is exactly `1 + 0i` yields the other factor, because complex multiplication by it is not
    /// exact for infinite components.
    Product,

    /// Running numerically stable `log(sum(exp(x)))`. Each prefix is accumulated by folding the pairwise [`LogAddExp`]
    /// operation, which is stable over the whole real range but is a different expression from the max-shifted
    /// reduction that [`ReductionKind::LogSumExp`](crate::ReductionKind::LogSumExp) evaluates, so the two can round
    /// differently in their last bits.
    ///
    /// Real floating-point and complex inputs are supported. Complex prefixes use the principal logarithm of the
    /// elementwise [`LogAddExp`] operation, so their imaginary components stay in `[-π, π]`. An argument whose real
    /// component is negative infinity has a zero exponential whatever its imaginary component, and so the combiner
    /// returns the other argument unchanged. That shortcut is exact, and it is what keeps complex prefixes over such
    /// elements defined: the elementwise complex combination of two of them would subtract `-∞` from `-∞` and produce
    /// NaN (real combinations already return the other argument). A prefix that no combination produced (i.e., the
    /// first one, or one following a prefix whose real component is negative infinity) is wrapped onto the same
    /// principal branch, so that every prefix, and in particular the last one, agrees with the matching
    /// [`ReductionKind::LogSumExp`](crate::ReductionKind::LogSumExp) reduction. Differentiation goes through the
    /// elementwise combination instead, so under differentiation such complex prefixes are NaN, and the first prefix
    /// is the raw first input.
    ///
    /// Padding is filled with the element data type's lowest real value (i.e., [`RaggedMaskIdentity::LowestReal`]),
    /// which must be an identity of the rounded pairwise [`LogAddExp`] operation. True negative infinity satisfies this
    /// contract (paired with a zero imaginary component for complex inputs), as do finite sentinels that round back to
    /// the other input after each pairwise combination. A binary fold rounds after every pair, so once combining two
    /// sentinel values returns the sentinel, an all-sentinel subtree of any size does too, and such sentinels therefore
    /// remain neutral for any prefix length. [`DataType::F8E8M0FNU`] and [`DataType::F6E2M3FN`] have no suitable
    /// identity and are rejected, as are all integer and Boolean inputs. This criterion intentionally differs from that
    /// of the log-sum-exp reduction, whose padding must remain neutral after subtracting an arbitrary maximum and which
    /// therefore requires a format that represents negative infinity.
    LogSumExp,

    /// Running maximum. The identity is the element data type's lowest value (i.e., negative infinity for the
    /// floating-point formats that have infinities). Real and complex numeric inputs are supported, with the ordering
    /// and exceptional-value semantics of the elementwise [`Max`] operation: complex inputs compare their real parts
    /// first and their imaginary parts second, NaNs propagate, and `-0.0` orders below `+0.0`.
    Max,

    /// Running minimum. The identity is the element data type's highest value (i.e., positive infinity for the
    /// floating-point formats that have infinities). Real and complex numeric inputs are supported, with the ordering
    /// and exceptional-value semantics of the elementwise [`Min`] operation: complex inputs compare their real parts
    /// first and their imaginary parts second, NaNs propagate, and `-0.0` orders below `+0.0`.
    Min,
}

impl CumulativeKind {
    /// Returns the name of this [`CumulativeKind`], which program renderings and diagnostics use as the `kind`
    /// attribute of a [`CumulativeOperation`] (e.g., `cumulative [kind=sum, axis=0]`).
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Product => "product",
            Self::LogSumExp => "log_sum_exp",
            Self::Max => "max",
            Self::Min => "min",
        }
    }
}

impl Display for CumulativeKind {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.name())
    }
}

/// Represents one inclusive prefix scan along a single array axis. Output element `i` along [`axis`](Self::axis) holds
/// the combination of the input elements `0..=i` under the combining operator selected by [`kind`](Self::kind), or of
/// the input elements `i..` when [`reverse`](Self::reverse) is set. The output type is the input type, so a cumulative
/// operation is a shape-preserving unary primitive.
///
/// `reverse` belongs to the payload rather than being spelled by the caller as a pair of reversals precisely because it
/// makes a cumulative sum closed under transposition: the adjoint of a forward prefix sum is a reverse prefix sum of
/// the output cotangent and vice versa.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CumulativeOperation {
    /// Refer to the documentation of [`Self::axis`].
    axis: usize,

    /// Refer to the documentation of [`Self::kind`].
    kind: CumulativeKind,

    /// Refer to the documentation of [`Self::reverse`].
    reverse: bool,
}

impl CumulativeOperation {
    /// Creates a new forward [`CumulativeOperation`] scanning along `axis` with the supplied `kind`. The scanned extent
    /// is not part of the operation payload (it is recoverable from the staged input type wherever a rule needs it).
    #[inline]
    pub fn new(axis: usize, kind: CumulativeKind) -> Self {
        Self { axis, kind, reverse: false }
    }

    /// Returns this [`CumulativeOperation`] with its scan direction set to `reverse`, accumulating from the end of the
    /// scanned axis toward its start if `reverse` is `true`, and the opposite otherwise.
    #[inline]
    pub fn with_reverse(mut self, reverse: bool) -> Self {
        self.reverse = reverse;
        self
    }

    /// Returns the scanned axis of this [`CumulativeOperation`], in the input's own coordinate system.
    #[inline]
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// Returns the [`CumulativeKind`] of this [`CumulativeOperation`].
    #[inline]
    pub fn kind(&self) -> CumulativeKind {
        self.kind
    }

    /// Returns whether the scan accumulates from the end of the scanned axis toward its start,
    /// for this [`CumulativeOperation`].
    #[inline]
    pub fn reverse(&self) -> bool {
        self.reverse
    }
}

impl Display for CumulativeOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for CumulativeOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        CUMULATIVE_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![input_types[0].cumulative(self.axis, self.kind)?])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("kind", self.kind)?;
            operation.field("axis", self.axis)?;
            if self.reverse {
                // The scan direction renders only when it is set, keeping the common forward scan rendering compact.
                operation.field("reverse", self.reverse)?;
            }
            Ok(())
        })
    }
}

impl<C: Domain<Type = ArrayType, Value: Cumulative>> InterpretableOperation<C> for CumulativeOperation {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].cumulative(self.axis, self.kind, self.reverse)?])
    }
}

impl<C: Context<Type = ArrayType, Operation: From<CumulativeOperation>>> PartiallyEvaluatableOperation<C>
    for CumulativeOperation
{
}

impl<C: Context<Type = ArrayType>, P: RaggedArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>>
    for CumulativeOperation
where
    CumulativeOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        // The scanned axis is expressed in the per-item coordinate system and therefore cannot name the inserted batch
        // dimension, so an axis at or after the batch axis shifts past it. A prefix scan preserves its input's rank,
        // and so the output batch axis is the input batch axis itself. A replicated input carries no inserted batch
        // dimension, so its scanned axis needs no shift. Ragged axes are packed positions in both cases, and so the
        // masking and rewrapping below use the lifted axis.
        check_count!("input", inputs, 1, ProgramError);
        let (lifted_axis, output_batch_axis) = match inputs[0].batch_axis_position() {
            Some(batch_axis) => {
                (if self.axis < batch_axis { self.axis } else { self.axis + 1 }, BatchAxis::from_position(batch_axis))
            }
            None => (self.axis, BatchAxis::replicated()),
        };

        // Padding along a _scanned_ ragged axis is neutralized with the identity of the combining operator first, so
        // that padded positions cannot contribute to any live prefix. A payload-free structural zero needs no masking,
        // because every element of it (padding included) is the same zero, whose every prefix sum or product is zero.
        let scans_ragged_axis = inputs[0].ragged_axes().iter().any(|ragged_axis| ragged_axis.axis() == lifted_axis);
        let input = match scans_ragged_axis && !inputs[0].r#type().data_type().is_zero() {
            true => {
                let identity = match self.kind {
                    CumulativeKind::Sum => RaggedMaskIdentity::Zero,
                    CumulativeKind::Product => RaggedMaskIdentity::One,
                    CumulativeKind::LogSumExp => RaggedMaskIdentity::LowestReal,
                    CumulativeKind::Max => RaggedMaskIdentity::Lowest,
                    CumulativeKind::Min => RaggedMaskIdentity::Highest,
                };
                P::mask_identity_input(context, &inputs[0], &[lifted_axis], identity)?
            }
            false => inputs[0].clone(),
        };
        let lifted = Self { axis: lifted_axis, ..self.clone() };
        let mut outputs = lifted.interpret_with_batch_axes(
            context,
            std::slice::from_ref(&input),
            std::slice::from_ref(&output_batch_axis),
        )?;
        check_count!("output", outputs, 1, ProgramError);

        // Interpretation carries values rather than batch metadata, so the input's ragged axes are restored here.
        // A scan consumes none of them, and so the rule reports no consumption evidence.
        let output = ArrayBatch::new(outputs.remove(0).into_value(), output_batch_axis)?
            .with_ragged_axes(input.ragged_axes().to_vec())?;
        Ok(BatchedOutputs::new(vec![output], Vec::new()))
    }
}

impl_differentiable_operation! {
    CumulativeOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Value: Cumulative,
        C::Operation: From<AddOperation<ArrayType>>
            + From<MulOperation<ArrayType>>
            + From<OrOperation<ArrayType>>
            + From<MaxOperation<ArrayType>>
            + From<MinOperation<ArrayType>>
            + From<LogAddExpOperation<ArrayType>>
            + From<CumulativeOperation>
            + From<BroadcastOperation>
            + From<ConcatenateOperation<ArrayType>>
            + From<PadOperation<ArrayType>>
            + From<SliceOperation>
            + OperationProvider<ArrayType, ZeroOperation<ArrayType>, Operation = C::Operation>
            + OperationProvider<ArrayType, BroadcastOperation, Operation = C::Operation>
            + OperationProvider<ArrayType, ParallelVaryOperation, Operation = C::Operation>,
    {
        |operation, context, driver, inputs| {
            // A cumulative sum is linear in its input, so its tangent is the same prefix sum of the input tangent.
            // Every other kind is nonlinear, so its rule differentiates through the associative-scan decomposition
            // with the kind's own combining operator, and every primitive that construction stages contributes its
            // own forward-mode rule. The composite array universe reaches this rule through the default projected
            // fall-through of `MemberDifferentiableOperation`, because the operation is shape-preserving and its
            // input never needs the replication that a broadcasting elementwise member does.
            check_count!("input", inputs, 1, ProgramError);
            let (axis, kind, reverse) = (operation.axis, operation.kind, operation.reverse);
            let primal_input = inputs[0].primal();
            let MaybeZero::Value(tangent_input) = inputs[0].tangent() else {
                // Every JVP is linear in its tangent, so a structural zero tangent stays a structural zero one,
                // and the primal is cheaper to obtain from the primitive itself than from the decomposition.
                let primal = primal_input.cumulative(axis, kind, reverse)?;
                let tangent = MaybeZero::Zero(primal.r#type().tangent()?);
                return Ok(vec![DifferentiationDual::new(primal, tangent)?]);
            };

            let combine_fn: fn(
                &Tracer<TracingContext<C::Constant, C::Operation>>,
                &Tracer<TracingContext<C::Constant, C::Operation>>,
            ) -> Result<Tracer<TracingContext<C::Constant, C::Operation>>, ProgramError> = match kind {
                CumulativeKind::Sum => {
                    let primal = primal_input.cumulative(axis, kind, reverse)?;
                    let tangent = tangent_input.cumulative(axis, kind, reverse)?;
                    return Ok(vec![DifferentiationDual::new(primal, MaybeZero::Value(tangent))?]);
                }
                CumulativeKind::Product => |left, right| left.mul(right),
                CumulativeKind::LogSumExp => |left, right| left.log_add_exp(right),
                CumulativeKind::Max => |left, right| left.max(right),
                CumulativeKind::Min => |left, right| left.min(right),
            };

            // Only non-linear kinds with a live tangent reach this decomposition. Each staged primitive contributes
            // its own JVP through the instruction-scoped driver; the shortcuts above avoid unnecessary tracing. The
            // primal output of a non-linear kind comes from the decomposition, which reassociates combinations and can
            // change the last bits of floating-point results. It interleaves its halves by zero-padding and adding
            // them. That addition turns a `-0.0` result into `+0.0`, so under differentiation the primal output of
            // an extremum scan over signed zeros can differ from the undifferentiated one in the sign of a zero.
            // Similarly, the decomposition combines log-sum-exp prefixes with the unguarded elementwise `log_add_exp`,
            // so a complex prefix that combines two arguments with negative-infinite real components is NaN there,
            // while the primitive returns the other argument, and its first complex prefix is the raw first input,
            // while the primitive wraps it onto the principal branch.

            // Attribute the decomposition to this differentiation rule using provenance scopes. Its  manipulation
            // primitives do not preserve rank-specific layouts, so an identity broadcast restores the input layout
            // and makes the outputs have the cumulative primitive's type. Backends lower that broadcast as a layout
            // constraint (e.g., XLA's `LayoutConstraint`).
            let input_type = primal_input.r#type().into_owned();
            let (_, decomposition) = TracingContext::<C::Constant, C::Operation>::trace::<_, ArrayType, _>(
                |value: Tracer<TracingContext<C::Constant, C::Operation>>| {
                    let domain = value.dispatch_domain();
                    domain.invoke_with_provenance_scope(ProvenanceScope::new("ryft"), || {
                        domain.invoke_with_provenance_scope(ProvenanceScope::new("differentiation"), || {
                            let output = associative_scan(&value, axis, reverse, &combine_fn)?;
                            match input_type.layout() {
                                Some(_) => output.broadcast(
                                    input_type.clone(),
                                    &(0..input_type.rank()).collect::<Vec<_>>(),
                                ),
                                None => Ok(output),
                            }
                        })
                    })
                },
                input_type.clone(),
            )?;

            // A shared context can replay the fused program and obtain both primal and tangent in one pass.
            if std::ptr::eq(context.primal(), context.tangent()) {
                let fused = driver.jvp_program(decomposition.entry_region_ref(), &[0])?;
                let mut outputs = fused.interpret_in_context(
                    context.primal(),
                    vec![primal_input.clone(), tangent_input.clone()],
                )?;
                check_count!("output", outputs, 2, ProgramError);
                let output_tangent = outputs.remove(1);
                return Ok(vec![DifferentiationDual::new(outputs.remove(0), MaybeZero::Value(output_tangent))?]);
            }

            // Separate contexts replay the primal first, then transfer its saved residuals into the tangent context.
            // Linearizing through the driver preserves the active differentiation rules and policy.
            let linearization = driver.linearize_program(decomposition.entry_region_ref(), &[0])?;
            let mut primal_outputs =
                linearization.primal().interpret_in_context(context.primal(), vec![primal_input.clone()])?;
            check_count!("output", primal_outputs, 1 + linearization.residual_count(), ProgramError);
            let residuals = primal_outputs.split_off(1);
            let mut tangent_inputs = vec![tangent_input.clone()];
            tangent_inputs.extend(
                residuals.into_iter().map(|value| context.primal_to_tangent(value)).collect::<Result<Vec<_>, _>>()?,
            );
            let mut tangent_outputs =
                linearization.tangent().interpret_in_context(context.tangent(), tangent_inputs)?;
            check_count!("output", tangent_outputs, 1, ProgramError);
            Ok(vec![DifferentiationDual::new(
                primal_outputs.remove(0),
                MaybeZero::Value(tangent_outputs.remove(0)),
            )?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: From<CumulativeOperation>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // A forward prefix sum sends input element `i` into every output element `j >= i`, so the cotangent of
            // input `i` is the sum of the output cotangents `j >= i` (i.e., a reverse prefix sum). The adjoint of
            // a reverse prefix sum is symmetrically a forward one, which is why a cumulative sum is closed under
            // transposition and needs no companion primitive. Every other kind is nonlinear and is instead
            // differentiated through the linear operations staged by its JVP, so it is rejected here regardless
            // of its cotangent.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 1, DifferentiationError);
            if operation.kind != CumulativeKind::Sum {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "`{}` with kind `{}` is not directly transposable",
                        operation.name(),
                        operation.kind(),
                    ),
                }
                .into());
            }
            let contribution = match &outputs[0] {
                MaybeZero::Zero(_) => MaybeZero::Zero(inputs[0].r#type().cotangent()?),
                MaybeZero::Value(cotangent) => {
                    MaybeZero::Value(cotangent.cumulative(operation.axis, operation.kind, !operation.reverse)?)
                }
            };
            accumulators[0].accumulate(context, contribution)
        }
    },
}

/// Value-level cumulative capability that accumulates the elements of an array along one axis based on a chosen
/// [`CumulativeKind`]. [`Cumulative`] fills the same role for [`CumulativeOperation`] that [`Reduce`](crate::Reduce)
/// fills for [`ReduceOperation`](crate::ReduceOperation). Concrete [`Array`]s scan immediately, while context-carrying
/// values bind a [`CumulativeOperation`] through their own context. The output has the input's type, and element `i`
/// along the scanned axis holds the combination of the input elements `0..=i`, or of the input elements `i..` for the
/// reverse direction. Refer to [`CumulativeKind`] for the identity, supported data types, and exceptional values of
/// each kind.
///
/// Besides the general [`Self::cumulative`] function, this trait provides one forward and one reverse shortcut function
/// per kind, which all share the axis, direction, and error contract of [`Self::cumulative`].
pub trait Cumulative: Sized {
    /// Accumulates `self` along `axis` using the combining operator selected by `kind`.
    ///
    /// # Parameters
    ///
    ///   - `axis`: Axis of `self` to scan, with negative indices counted from the end. Its dimension must be static
    ///     and unsharded.
    ///   - `kind`: [`CumulativeKind`] that determines how the elements along `axis` are combined.
    ///   - `reverse`: Whether to accumulate from the end of `axis` toward its start (i.e., to compute inclusive
    ///     suffixes rather than inclusive prefixes).
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if `kind` does not support the data type of `self`, if `axis` is out of bounds, if
    /// the scanned dimension is dynamic or sharded, or if the context of `self` fails to bind the scan.
    fn cumulative<A: Into<Axis>>(&self, axis: A, kind: CumulativeKind, reverse: bool) -> Result<Self, ProgramError>;

    /// Returns the inclusive prefix sum of `self` along `axis` using [`CumulativeKind::Sum`]. Refer to
    /// [`Self::cumulative`] for the semantics of `axis` and for the errors that this function may return.
    #[inline]
    fn cumulative_sum<A: Into<Axis>>(&self, axis: A) -> Result<Self, ProgramError> {
        self.cumulative(axis, CumulativeKind::Sum, false)
    }

    /// Returns the inclusive suffix sum of `self` along `axis` using [`CumulativeKind::Sum`]. Refer to
    /// [`Self::cumulative`] for the semantics of `axis` and for the errors that this function may return.
    #[inline]
    fn reverse_cumulative_sum<A: Into<Axis>>(&self, axis: A) -> Result<Self, ProgramError> {
        self.cumulative(axis, CumulativeKind::Sum, true)
    }

    /// Returns the inclusive prefix product of `self` along `axis` using [`CumulativeKind::Product`]. Refer to
    /// [`Self::cumulative`] for the semantics of `axis` and for the errors that this function may return.
    #[inline]
    fn cumulative_product<A: Into<Axis>>(&self, axis: A) -> Result<Self, ProgramError> {
        self.cumulative(axis, CumulativeKind::Product, false)
    }

    /// Returns the inclusive suffix product of `self` along `axis` using [`CumulativeKind::Product`]. Refer to
    /// [`Self::cumulative`] for the semantics of `axis` and for the errors that this function may return.
    #[inline]
    fn reverse_cumulative_product<A: Into<Axis>>(&self, axis: A) -> Result<Self, ProgramError> {
        self.cumulative(axis, CumulativeKind::Product, true)
    }

    /// Returns the running maximum of `self` along `axis` using [`CumulativeKind::Max`]. Refer to [`Self::cumulative`]
    /// for the semantics of `axis` and for the errors that this function may return.
    #[inline]
    fn cumulative_max<A: Into<Axis>>(&self, axis: A) -> Result<Self, ProgramError> {
        self.cumulative(axis, CumulativeKind::Max, false)
    }

    /// Returns the reverse running maximum of `self` along `axis` using [`CumulativeKind::Max`]. Refer to
    /// [`Self::cumulative`] for the semantics of `axis` and for the errors that this function may return.
    #[inline]
    fn reverse_cumulative_max<A: Into<Axis>>(&self, axis: A) -> Result<Self, ProgramError> {
        self.cumulative(axis, CumulativeKind::Max, true)
    }

    /// Returns the running minimum of `self` along `axis` using [`CumulativeKind::Min`]. Refer to [`Self::cumulative`]
    /// for the semantics of `axis` and for the errors that this function may return.
    #[inline]
    fn cumulative_min<A: Into<Axis>>(&self, axis: A) -> Result<Self, ProgramError> {
        self.cumulative(axis, CumulativeKind::Min, false)
    }

    /// Returns the reverse running minimum of `self` along `axis` using [`CumulativeKind::Min`]. Refer to
    /// [`Self::cumulative`] for the semantics of `axis` and for the errors that this function may return.
    #[inline]
    fn reverse_cumulative_min<A: Into<Axis>>(&self, axis: A) -> Result<Self, ProgramError> {
        self.cumulative(axis, CumulativeKind::Min, true)
    }

    /// Returns the running numerically stable `log(sum(exp(self)))` along `axis` using [`CumulativeKind::LogSumExp`],
    /// whose documentation describes its data-type limits. Refer to [`Self::cumulative`] for the semantics of `axis`
    /// and for the errors that this function may return.
    #[inline]
    fn cumulative_log_sum_exp<A: Into<Axis>>(&self, axis: A) -> Result<Self, ProgramError> {
        self.cumulative(axis, CumulativeKind::LogSumExp, false)
    }

    /// Returns the reverse running numerically stable `log(sum(exp(self)))` along `axis` using
    /// [`CumulativeKind::LogSumExp`], whose documentation describes its data-type limits. Refer to
    /// [`Self::cumulative`] for the semantics of `axis` and for the errors that this function may return.
    #[inline]
    fn reverse_cumulative_log_sum_exp<A: Into<Axis>>(&self, axis: A) -> Result<Self, ProgramError> {
        self.cumulative(axis, CumulativeKind::LogSumExp, true)
    }
}

impl Cumulative for Array {
    fn cumulative<A: Into<Axis>>(&self, axis: A, kind: CumulativeKind, reverse: bool) -> Result<Self, ProgramError> {
        let axis = axis.into();
        let rank = self.r#type().rank();
        let axis = axis.normalize(rank).map_err(|_| {
            TypeError::invalid(format!("`{CUMULATIVE_OPERATION_NAME}` axis {axis} is out of bounds for rank {rank}"))
        })?;

        // The type rule validates the scan and supplies the complete output metadata. The kernels below then decode the
        // input's logical elements, run the sequential prefix scan over them with the kind's element-level combining
        // operator, and re-encode the result into the input's own type. Accumulation happens in the input's element
        // data type, so a low-precision payload rounds every partial result exactly as a staged program does.
        let output_type = self.r#type().cumulative(axis, kind)?;
        let data_type = output_type.data_type();

        // The structural-zero element type has no payload bytes, and every prefix sum or product of a zero is a zero.
        if data_type == DataType::Zero {
            return Self::new(output_type, Vec::new());
        }

        match kind {
            CumulativeKind::Sum | CumulativeKind::Product => {
                dispatch_on_array_element_type!(@numeric data_type, |Element| {
                    let scanned = self.cumulative_elements::<Element, _>(axis, reverse, |left, right| {
                        match kind {
                            CumulativeKind::Sum => NumericArrayElement::add(left, right),
                            _ => multiply_product_elements(left, right),
                        }
                    })?;
                    Self::from_elements(output_type, scanned.as_slice())
                })
            }
            CumulativeKind::LogSumExp => {
                dispatch_on_array_element_type!(@float_or_complex data_type, |Element| {
                    // Converting an element into `f64` keeps only its real component, which is all that decides whether
                    // its exponential is zero. The guard never changes a real combination.
                    let scanned = self.cumulative_elements::<Element, _>(axis, reverse, |left, right| {
                        if right.convert_to::<f64>()? == f64::NEG_INFINITY {
                            Ok(left)
                        } else if left.convert_to::<f64>()? == f64::NEG_INFINITY {
                            Ok(right)
                        } else {
                            FloatingPointArrayElement::log_add_exp(left, right)
                        }
                    })?;

                    // A prefix that no combination produced (i.e., the first element, or one that follows a prefix
                    // whose real component is negative infinity) is still a raw input, so its phase is brought onto
                    // the principal branch `[-π, π]` that every combined prefix already lies on. In-range phases are
                    // kept as they are, and others are reduced through `atan2` of their sine and cosine, which reduce
                    // even huge arguments exactly, unlike a floating-point remainder by `2π`.
                    let scanned = match data_type.is_complex() {
                        true => scanned
                            .into_iter()
                            .map(|value| {
                                let complex = value.convert_to::<Complex<f64>>()?;
                                let pi = std::f64::consts::PI;
                                if (-pi..=pi).contains(&complex.im) {
                                    return Ok(value);
                                }
                                let reduced = complex.im.sin().atan2(complex.im.cos());
                                Element::from_complex(Complex::new(complex.re, reduced))
                            })
                            .collect::<Result<Vec<_>, ProgramError>>()?,
                        false => scanned,
                    };
                    Self::from_elements(output_type, scanned.as_slice())
                })
            }
            CumulativeKind::Max | CumulativeKind::Min => {
                dispatch_on_array_element_type!(@numeric data_type, |Element| {
                    let scanned = self.cumulative_elements::<Element, _>(axis, reverse, |left, right| {
                        Ok(match kind {
                            CumulativeKind::Max => ArrayElement::max(&left, &right),
                            _ => ArrayElement::min(&left, &right),
                        })
                    })?;
                    Self::from_elements(output_type, scanned.as_slice())
                })
            }
        }
    }
}

// Any context-carrying value scans by binding a `CumulativeOperation` through its context. The
// `From<CumulativeOperation>` bound makes this disjoint from the eager value types (whose context operation is
// `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete implementations.
impl<V: Value<Type = ArrayType, DispatchDomain: Context<Type = ArrayType, Operation: From<CumulativeOperation>>>>
    Cumulative for V
{
    fn cumulative<A: Into<Axis>>(&self, axis: A, kind: CumulativeKind, reverse: bool) -> Result<Self, ProgramError> {
        let axis = axis.into();
        let rank = self.r#type().rank();
        let axis = axis.normalize(rank).map_err(|_| {
            TypeError::invalid(format!("`{CUMULATIVE_OPERATION_NAME}` axis {axis} is out of bounds for rank {rank}"))
        })?;
        let mut outputs = self.dispatch_domain().bind(
            CumulativeOperation::new(axis, kind).with_reverse(reverse),
            Vec::new(),
            std::slice::from_ref(self),
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl ArrayType {
    /// Returns the output [`ArrayType`] produced by scanning `self` along `axis` with `kind`. The result _is_ `self` as
    /// a prefix scan changes neither the element data type nor the shape, layout, memory placement, or sharding. Also,
    /// this function validates that:
    ///
    ///   - `kind` supports the element data type of `self`, as documented on [`CumulativeKind`],
    ///   - `axis` is within `0..self.rank()`,
    ///   - the scanned dimension is [`Dimension::Static`], because a prefix scan is defined by the exact number of
    ///     elements that it accumulates over,
    ///   - the scanned dimension is unsharded, because a prefix crosses shard boundaries and a cumulative operation
    ///     carries no cross-shard communication of its own, and
    ///   - `self` has no unreduced mesh axes unless `kind` is [`CumulativeKind::Sum`], because only a prefix sum
    ///     commutes with the pending cross-device sum of an unreduced value.
    ///
    /// A dynamically sized scanned axis could be supported in the future by physicalizing the scan at the dimension's
    /// declared upper bound and masking the elements past each runtime extent with the kind's identity, which is the
    /// same discipline that the ragged batching rule already uses. That extension is deliberately not implemented here
    /// as it would silently change the operation's cost model, and so it belongs to an explicit dynamic-scan surface.
    fn cumulative(&self, axis: usize, kind: CumulativeKind) -> Result<Self, TypeError> {
        // The element data type is validated before the scan geometry, and the diagnostic names the kind, because the
        // supported data types differ across kinds (e.g., summation accepts the structural zero while extrema do not).
        let data_type = self.data_type();
        let requirement: Option<Cow<'static, str>> = match kind {
            CumulativeKind::Sum | CumulativeKind::Product if !data_type.is_numeric() && data_type != DataType::Zero => {
                Some(Cow::Borrowed("numeric inputs"))
            }
            CumulativeKind::LogSumExp if !data_type.is_floating_point() && !data_type.is_complex() => {
                Some(Cow::Borrowed("floating-point or complex inputs"))
            }
            CumulativeKind::Max | CumulativeKind::Min if !data_type.is_numeric() => {
                Some(Cow::Borrowed("numeric inputs"))
            }
            CumulativeKind::LogSumExp if matches!(data_type, DataType::F8E8M0FNU | DataType::F6E2M3FN) => {
                Some(Cow::Owned(format!(
                    "a floating-point format whose lowest value is a `{LOG_ADD_EXP_OPERATION_NAME}` identity",
                )))
            }
            _ => None,
        };
        if let Some(requirement) = requirement {
            return Err(TypeError::invalid(format!(
                "`{CUMULATIVE_OPERATION_NAME}` with kind `{kind}` requires {requirement} but got `{data_type}`",
            )));
        }

        let rank = self.rank();
        if axis >= rank {
            return Err(TypeError::invalid(format!(
                "`{CUMULATIVE_OPERATION_NAME}` axis {axis} is out of bounds for rank {rank}",
            )));
        }

        if !matches!(self.dimension(axis), Dimension::Static(_)) {
            return Err(TypeError::invalid(format!(
                "`{CUMULATIVE_OPERATION_NAME}` requires a static scanned dimension but axis {axis} of `{self}` is \
                 dynamic",
            )));
        }

        if let Some(sharding) = self.sharding()
            && matches!(sharding.dimensions()[axis], ShardingDimension::Sharded(_))
        {
            return Err(TypeError::invalid(format!(
                "`{CUMULATIVE_OPERATION_NAME}` requires an unsharded scanned dimension but axis {axis} of `{self}` is \
                 sharded",
            )));
        }

        if kind != CumulativeKind::Sum && self.sharding().is_some_and(|sharding| !sharding.unreduced_axes().is_empty())
        {
            return Err(TypeError::invalid(format!(
                "`{CUMULATIVE_OPERATION_NAME}` with kind `{kind}` cannot scan inputs with unreduced axes",
            )));
        }

        Ok(self.clone())
    }
}

impl Array {
    /// Scans this array's logical elements sequentially along `axis`, returning a row-major payload of the same shape.
    /// Output element `i` along `axis` accumulates input elements `0..=i`, or elements `i..` when `reverse` is set.
    /// An axis shorter than two elements, including an empty axis, leaves the payload unchanged. Callers validate
    /// the static shape, axis, and element type before invoking this helper.
    ///
    /// `combine_fn` is fallible because reference-backend element arithmetic can fail (e.g., conversion into a
    /// low-precision encoding). Returning elements lets complex log-sum-exp normalize phases before encoding them.
    ///
    /// # Parameters
    ///
    ///   - `axis`: Validated nonnegative scanned axis.
    ///   - `reverse`: Whether to accumulate from the end of the scanned axis toward its start.
    ///   - `combine_fn`: Binary associative operator, receiving the accumulated prefix and the next element.
    fn cumulative_elements<T: ArrayElement, F: Fn(T, T) -> Result<T, ProgramError>>(
        &self,
        axis: usize,
        reverse: bool,
        combine_fn: F,
    ) -> Result<Vec<T>, ProgramError> {
        let shape = self.r#type().static_shape().unwrap();
        let values = self.elements::<T>()?;
        let mut output = values.clone();
        let extent = shape[axis];
        if extent < 2 {
            return Ok(output);
        }

        // Row-major storage splits into `outer` independent blocks of `extent` slices, each holding `inner` elements,
        // so one scan step moves by `inner` elements and the scan visits every `(outer, inner)` pair once. Both bounds
        // are computed as direct dimension products rather than from the payload length and the scanned axis stride,
        // because a zero-extent axis anywhere to the right of `axis` makes that stride zero.
        let inner = shape.dimensions()[axis + 1..].iter().product::<usize>();
        let outer = shape.dimensions()[..axis].iter().product::<usize>();
        for block in 0..outer {
            let base = block * extent * inner;
            for offset in 0..inner {
                let index = |position: usize| base + position * inner + offset;
                if reverse {
                    for position in (0..extent - 1).rev() {
                        output[index(position)] =
                            combine_fn(output[index(position + 1)].clone(), values[index(position)].clone())?;
                    }
                } else {
                    for position in 1..extent {
                        output[index(position)] =
                            combine_fn(output[index(position - 1)].clone(), values[index(position)].clone())?;
                    }
                }
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use half::bf16;
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::batching::DynamicArrayExtentBatchingPolicy;
    use crate::arrays::{
        ArrayIrOperation, ArrayIrValue, ArrayOperation, DimensionBounds, DimensionType, DimensionVariable, Layout,
        LogicalMesh, Memory, MeshAxis, MeshAxisType, RaggedAxis, Shape, Sharding, StridedLayout, f4e2m1fn, f8e4m3fn,
        f8e4m3fnuz, f8e5m2,
    };
    use crate::contexts::{EagerContext, ProjectedContext, StagingContext};
    use crate::differentiation::{
        DifferentiableOperation, DifferentiationContext, TransposableOperation, TranspositionContext, differentiate_at,
    };
    use crate::macros::{
        check_gradient, check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::operations::reductions::Reduce;
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{EmptyRegionDriver, Program, ProgramBuilder, ProgramRenderingMode, ValueProjection};

    use super::*;

    // Pairwise stable `log(exp(a) + exp(b))`, spelled out so that the expected values below pin the construction that
    // the log-sum-exp scan folds rather than an equivalent-in-exact-arithmetic alternative.
    fn log_add_exp(left: f64, right: f64) -> f64 {
        let delta = left - right;
        match delta.is_nan() {
            true => left + right,
            false => left.max(right) + (-delta.abs()).exp().ln_1p(),
        }
    }

    // Single-instruction program that applies `operation` to one input of type `input_type`, which the tests below
    // transform to inspect the programs that the differentiation rule stages.
    fn cumulative_program(
        operation: CumulativeOperation,
        input_type: ArrayType,
    ) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(input_type);
        let outputs = builder
            .add_instruction(ArrayOperation::from(operation), Vec::new(), vec![input], None)
            .unwrap()
            .to_vec();
        builder.build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_cumulative_kind_name() {
        for (kind, name) in [
            (CumulativeKind::Sum, "sum"),
            (CumulativeKind::Product, "product"),
            (CumulativeKind::LogSumExp, "log_sum_exp"),
            (CumulativeKind::Max, "max"),
            (CumulativeKind::Min, "min"),
        ] {
            assert_eq!(kind.name(), name);
            assert_eq!(kind.to_string(), name);
        }
    }

    #[test]
    fn test_cumulative() {
        let operation = CumulativeOperation::new(1, CumulativeKind::Sum);
        assert_eq!(operation.axis(), 1);
        assert_eq!(operation.kind(), CumulativeKind::Sum);
        assert!(!operation.reverse());
        assert!(operation.clone().with_reverse(true).reverse());

        // The kind always renders, while the scan direction renders only when it is set, keeping the common forward
        // scan compact.
        assert_eq!(operation.to_string(), "cumulative [kind=sum, axis=1]");
        assert_eq!(
            CumulativeOperation::new(0, CumulativeKind::LogSumExp).with_reverse(true).to_string(),
            "cumulative [kind=log_sum_exp, axis=0, reverse=true]",
        );

        // Equality and hashing distinguish axes, kinds, and directions.
        let reverse = operation.clone().with_reverse(true);
        assert_eq!(operation.clone().with_reverse(false), operation);
        assert_ne!(reverse, operation);
        assert_ne!(CumulativeOperation::new(0, CumulativeKind::Sum), operation);
        assert_ne!(CumulativeOperation::new(1, CumulativeKind::Max), operation);
        let operations = HashSet::from([operation.clone(), reverse.clone()]);
        assert_eq!(operations.len(), 2);
        assert!(operations.contains(&CumulativeOperation::new(1, CumulativeKind::Sum)));
        assert!(operations.contains(&reverse));
        assert!(!operations.contains(&CumulativeOperation::new(1, CumulativeKind::Max)));
    }

    #[test]
    fn test_cumulative_type_inference() {
        check_operation_type_inference!(
            operation = CumulativeOperation::new(1, CumulativeKind::Sum),
            cases = [
                {
                    input_types = [ArrayType::new_static(DataType::F64, [3, 2])],
                    output_types = [ArrayType::new_static(DataType::F64, [3, 2])],
                },
                {
                    input_types = [ArrayType::new_static(DataType::F64, [3])],
                    error = "`cumulative` axis 1 is out of bounds for rank 1",
                },
            ],
        );
    }

    #[test]
    fn test_cumulative_interpretation() {
        // Interpretation passes the operation's kind and direction through to `Cumulative::cumulative`, whose numerics
        // the eager kernel tests below own.
        let context = EagerContext::<Array>::new();
        let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(
            CumulativeOperation::new(1, CumulativeKind::Sum).interpret(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&input),
            ),
            Ok(vec![Array::matrix(2, 3, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]).unwrap()]),
        );
        assert_eq!(
            CumulativeOperation::new(1, CumulativeKind::Product).with_reverse(true).interpret(
                &context,
                &EmptyRegionDriver,
                std::slice::from_ref(&input),
            ),
            Ok(vec![Array::matrix(2, 3, vec![6.0, 6.0, 3.0, 120.0, 30.0, 6.0]).unwrap()]),
        );
    }

    #[test]
    fn test_cumulative_partial_evaluation() {
        check_operation_partial_evaluation!(
            operation = CumulativeOperation::new(0, CumulativeKind::Product),
            inputs = [Array::vector(vec![1.0, 2.0, 3.0]).unwrap()],
            expected = Array::vector(vec![1.0, 2.0, 6.0]).unwrap(),
        );
    }

    #[test]
    fn test_cumulative_batching() {
        // A replicated input carries no inserted batch dimension, so the scanned axis needs no shift.
        check_operation_batching!(
            @exact,
            operation = CumulativeOperation::new(0, CumulativeKind::Sum),
            axis_size = 2,
            cases = [{
                inputs = [(@replicated, Array::vector(vec![1.0, 2.0, 3.0]).unwrap())],
                outputs = [(@replicated, Array::vector(vec![1.0, 3.0, 6.0]).unwrap())],
            }],
        );

        // Physical input is [2 batch items, 3 columns] mapped at axis 0, so the per-item axis 0 scans physical axis 1
        // and each batch item accumulates independently, in either direction.
        check_operation_batching!(
            @exact,
            operation = CumulativeOperation::new(0, CumulativeKind::Sum),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::matrix(2, 3, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]).unwrap())],
            }],
        );
        check_operation_batching!(
            @exact,
            operation = CumulativeOperation::new(0, CumulativeKind::Max).with_reverse(true),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::matrix(2, 3, vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0]).unwrap())],
                outputs = [(@mapped(axis = 0), Array::matrix(2, 3, vec![4.0, 4.0, 4.0, 9.0, 9.0, 9.0]).unwrap())],
            }],
        );

        // With the batch dimension inserted after the scanned axis, the scanned axis keeps its position.
        check_operation_batching!(
            @exact,
            operation = CumulativeOperation::new(0, CumulativeKind::Sum),
            axis_size = 3,
            cases = [{
                inputs = [(@mapped(axis = 1), Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap())],
                outputs = [(@mapped(axis = 1), Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_cumulative_batching_ragged_unscanned_axis() {
        // Per item, a ragged `[length]` row is scanned along its dense trailing axis, so the ragged axis is not the
        // scanned one: nothing needs masking, the axis rides through onto the result unchanged, and, because the scan
        // consumes no axis, the rule claims no consumption evidence.
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
        let ragged_axes = vec![RaggedAxis::new(1, Array::vector(vec![1i32, 3]).unwrap(), variable, vec![0])];
        let batch = |values: &[f32]| {
            ArrayBatch::new(
                Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [2, 3, 2]), values).unwrap(),
                BatchAxis::new(0),
            )
            .unwrap()
            .with_ragged_axes(ragged_axes.clone())
            .unwrap()
        };
        let (outputs, evidence) = CumulativeOperation::new(1, CumulativeKind::Sum)
            .batch(
                &BatchingContext::new(EagerContext::<Array>::new(), 2),
                &EmptyRegionDriver,
                &[batch(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])],
            )
            .unwrap()
            .into_parts();
        assert_eq!(outputs, vec![batch(&[1.0, 3.0, 3.0, 7.0, 5.0, 11.0, 7.0, 15.0, 9.0, 19.0, 11.0, 23.0])]);
        assert!(evidence.is_empty());
    }

    #[test]
    fn test_cumulative_batching_ragged_scanned_axis() {
        // Scanning a ragged axis would fold its padding into every later live prefix, so the rule asks the policy to
        // neutralize that padding with the identity of the kind's combining operator first. Static array batching
        // cannot, and says so (naming that identity) rather than silently scanning padding.
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
        let ragged_axes = vec![RaggedAxis::new(1, Array::vector(vec![1i32, 3]).unwrap(), variable, vec![0])];
        let input = ArrayBatch::new(Array::matrix(2, 3, vec![1.0f32; 6]).unwrap(), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(ragged_axes.clone())
            .unwrap();
        for (kind, identity) in [
            (CumulativeKind::Sum, "Zero"),
            (CumulativeKind::Product, "One"),
            (CumulativeKind::LogSumExp, "LowestReal"),
            (CumulativeKind::Max, "Lowest"),
            (CumulativeKind::Min, "Highest"),
        ] {
            assert_eq!(
                CumulativeOperation::new(0, kind).batch(
                    &BatchingContext::new(EagerContext::<Array>::new(), 2),
                    &EmptyRegionDriver,
                    std::slice::from_ref(&input),
                ),
                Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "static array batching cannot identity-mask bounded ragged dimension `length` on axis 1 with \
                         `{identity}`",
                    ),
                }),
            );
        }

        // Every element of a payload-free structural zero, padding included, is the same zero, so its prefix sums and
        // products need no mask at all (and there is no `one` of that type to write) and pass the ragged axis through.
        let zero = ArrayBatch::new(
            Array::new(ArrayType::new_static(DataType::Zero, [2, 3]), Vec::new()).unwrap(),
            BatchAxis::new(0),
        )
        .unwrap()
        .with_ragged_axes(ragged_axes)
        .unwrap();
        for kind in [CumulativeKind::Sum, CumulativeKind::Product] {
            let (outputs, evidence) = CumulativeOperation::new(0, kind)
                .batch(
                    &BatchingContext::new(EagerContext::<Array>::new(), 2),
                    &EmptyRegionDriver,
                    std::slice::from_ref(&zero),
                )
                .unwrap()
                .into_parts();
            assert_eq!(outputs, vec![zero.clone()]);
            assert!(evidence.is_empty());
        }

        // The composite dynamic policy can, and stages the mask ahead of the scan: the padded positions of the scanned
        // axis are selected away in favor of the identity, and the ragged axis survives on the result because a scan
        // consumes no axis.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
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
        let ragged_axes = vec![RaggedAxis::new(1, extents.into_projected().unwrap(), length, vec![0])];
        let input = ArrayBatch::new(packed.into_projected().unwrap(), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(ragged_axes.clone())
            .unwrap();

        // The per-item scan of axis 0 is the packed axis 1 that carries the ragged extents.
        let (outputs, evidence) = CumulativeOperation::new(0, CumulativeKind::LogSumExp)
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].ragged_axes(), ragged_axes.as_slice());
        assert!(evidence.is_empty());

        let output_id = outputs.into_iter().next().unwrap().into_value().into_value().atom_id().unwrap();
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
                    %12:f32[items, 3] = cumulative [kind=log_sum_exp, axis=1] %11
                in (%12)"
            },
        );
    }

    #[test]
    fn test_cumulative_differentiation() {
        // Sums are linear, so their tangent is the same scan of the input tangent.
        check_operation_differentiation!(
            @approx(step = 1e-4, epsilon = 1e-6),
            operation = CumulativeOperation::new(0, CumulativeKind::Sum),
            cases = [{
                primals = [Array::vector(vec![1.0, 2.0, 3.0]).unwrap()],
                tangents = [Array::vector(vec![1.0, 1.0, 1.0]).unwrap()],
                primal_outputs = [Array::vector(vec![1.0, 3.0, 6.0]).unwrap()],
                tangent_outputs = [Array::vector(vec![1.0, 2.0, 3.0]).unwrap()],
            }],
        );
        check_operation_differentiation!(
            @approx(step = 1e-4, epsilon = 1e-6),
            operation = CumulativeOperation::new(0, CumulativeKind::Sum).with_reverse(true),
            cases = [{
                primals = [Array::vector(vec![1.0, 2.0, 3.0]).unwrap()],
                tangents = [Array::vector(vec![1.0, 1.0, 1.0]).unwrap()],
                primal_outputs = [Array::vector(vec![6.0, 5.0, 3.0]).unwrap()],
                tangent_outputs = [Array::vector(vec![3.0, 2.0, 1.0]).unwrap()],
            }],
        );

        // The other kinds differentiate through the associative-scan decomposition. A product's tangent applies the
        // product rule to each prefix, so a zero input zeroes every later prefix but still passes its own tangent,
        // scaled by the product of the other elements.
        check_operation_differentiation!(
            @approx(step = 1e-4, epsilon = 1e-6),
            operation = CumulativeOperation::new(0, CumulativeKind::Product),
            cases = [
                {
                    primals = [Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap()],
                    tangents = [Array::vector(vec![1.0, 1.0, 1.0, 1.0]).unwrap()],
                    primal_outputs = [Array::vector(vec![1.0, 2.0, 6.0, 24.0]).unwrap()],
                    tangent_outputs = [Array::vector(vec![1.0, 3.0, 11.0, 50.0]).unwrap()],
                },
                {
                    primals = [Array::vector(vec![2.0, 0.0, 3.0]).unwrap()],
                    tangents = [Array::vector(vec![1.0, 1.0, 1.0]).unwrap()],
                    primal_outputs = [Array::vector(vec![2.0, 0.0, 0.0]).unwrap()],
                    tangent_outputs = [Array::vector(vec![1.0, 2.0, 6.0]).unwrap()],
                },
            ],
        );
        check_operation_differentiation!(
            @approx(step = 1e-4, epsilon = 1e-6),
            operation = CumulativeOperation::new(0, CumulativeKind::Product).with_reverse(true),
            cases = [{
                primals = [Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap()],
                tangents = [Array::vector(vec![1.0, 1.0, 1.0, 1.0]).unwrap()],
                primal_outputs = [Array::vector(vec![24.0, 24.0, 12.0, 4.0]).unwrap()],
                tangent_outputs = [Array::vector(vec![50.0, 26.0, 7.0, 1.0]).unwrap()],
            }],
        );

        // A log-sum-exp tangent is the softmax-weighted average of the input tangents over each prefix, which is the
        // complex derivative for complex inputs.
        let e = std::f64::consts::E;
        let complex_first = ComplexNumber::new(0.5f64, 0.25);
        let complex_second = ComplexNumber::new(-0.3f64, 1.0);
        let complex_total = complex_first.exp() + complex_second.exp();
        check_operation_differentiation!(
            @approx(step = 1e-4, epsilon = 1e-6),
            operation = CumulativeOperation::new(0, CumulativeKind::LogSumExp),
            cases = [
                {
                    primals = [Array::vector(vec![0.0, 1.0, 2.0]).unwrap()],
                    tangents = [Array::vector(vec![1.0, 2.0, 3.0]).unwrap()],
                    primal_outputs = [Array::vector(vec![0.0, (1.0 + e).ln(), (1.0 + e + e * e).ln()]).unwrap()],
                    tangent_outputs = [Array::vector(vec![
                        1.0,
                        (1.0 + 2.0 * e) / (1.0 + e),
                        (1.0 + 2.0 * e + 3.0 * e * e) / (1.0 + e + e * e),
                    ])
                    .unwrap()],
                },
                {
                    primals = [Array::vector(vec![complex_first, complex_second]).unwrap()],
                    tangents = [
                        Array::vector(vec![ComplexNumber::new(1.0, 0.0), ComplexNumber::new(0.0, 1.0)]).unwrap(),
                    ],
                    primal_outputs = [Array::vector(vec![complex_first, complex_total.ln()]).unwrap()],
                    tangent_outputs = [Array::vector(vec![
                        ComplexNumber::new(1.0, 0.0),
                        (complex_first.exp() + complex_second.exp() * ComplexNumber::new(0.0, 1.0)) / complex_total,
                    ])
                    .unwrap()],
                },
            ],
        );
        check_operation_differentiation!(
            @approx(step = 1e-4, epsilon = 1e-6),
            operation = CumulativeOperation::new(0, CumulativeKind::LogSumExp).with_reverse(true),
            cases = [{
                primals = [Array::vector(vec![0.0, 1.0, 2.0]).unwrap()],
                tangents = [Array::vector(vec![1.0, 2.0, 3.0]).unwrap()],
                primal_outputs = [Array::vector(vec![(1.0 + e + e * e).ln(), (e + e * e).ln(), 2.0]).unwrap()],
                tangent_outputs = [Array::vector(vec![
                    (1.0 + 2.0 * e + 3.0 * e * e) / (1.0 + e + e * e),
                    (2.0 * e + 3.0 * e * e) / (e + e * e),
                    3.0,
                ])
                .unwrap()],
            }],
        );

        // At tie-free inputs, a running extremum's tangent is the tangent of the element that currently attains it.
        check_operation_differentiation!(
            @approx(step = 1e-4, epsilon = 1e-6),
            operation = CumulativeOperation::new(0, CumulativeKind::Max),
            cases = [{
                primals = [Array::vector(vec![3.0, 1.0, 4.0, 1.5, 5.0]).unwrap()],
                tangents = [Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap()],
                primal_outputs = [Array::vector(vec![3.0, 3.0, 4.0, 4.0, 5.0]).unwrap()],
                tangent_outputs = [Array::vector(vec![1.0, 1.0, 3.0, 3.0, 5.0]).unwrap()],
            }],
        );

        check_operation_differentiation!(
            @approx(step = 1e-4, epsilon = 1e-6),
            operation = CumulativeOperation::new(0, CumulativeKind::Max).with_reverse(true),
            cases = [{
                primals = [Array::vector(vec![5.0, 1.0, 4.0, 1.5, 3.0]).unwrap()],
                tangents = [Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap()],
                primal_outputs = [Array::vector(vec![5.0, 4.0, 4.0, 3.0, 3.0]).unwrap()],
                tangent_outputs = [Array::vector(vec![1.0, 3.0, 3.0, 5.0, 5.0]).unwrap()],
            }],
        );

        check_operation_differentiation!(
            @approx(step = 1e-4, epsilon = 1e-6),
            operation = CumulativeOperation::new(0, CumulativeKind::Min),
            cases = [{
                primals = [Array::vector(vec![3.0, 1.0, 4.0, 1.5, 5.0]).unwrap()],
                tangents = [Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap()],
                primal_outputs = [Array::vector(vec![3.0, 1.0, 1.0, 1.0, 1.0]).unwrap()],
                tangent_outputs = [Array::vector(vec![1.0, 2.0, 2.0, 2.0, 2.0]).unwrap()],
            }],
        );

        check_operation_differentiation!(
            @approx(step = 1e-4, epsilon = 1e-6),
            operation = CumulativeOperation::new(0, CumulativeKind::Min).with_reverse(true),
            cases = [{
                primals = [Array::vector(vec![3.0, 1.0, 4.0, 1.5, 5.0]).unwrap()],
                tangents = [Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap()],
                primal_outputs = [Array::vector(vec![1.0, 1.0, 1.5, 1.5, 5.0]).unwrap()],
                tangent_outputs = [Array::vector(vec![2.0, 2.0, 4.0, 4.0, 5.0]).unwrap()],
            }],
        );
    }

    #[test]
    fn test_cumulative_differentiation_zero_tangent() {
        // Every JVP is linear in its tangent, so a structural zero input tangent stays a structural zero output tangent
        // without differentiating through the associative-scan decomposition of a nonlinear kind.
        let outputs = CumulativeOperation::new(0, CumulativeKind::Product)
            .jvp(
                &DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new()),
                &EmptyRegionDriver,
                &[DifferentiationDual::new_with_zero_tangent(Array::vector(vec![1.0, 2.0, 3.0]).unwrap()).unwrap()],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &Array::vector(vec![1.0, 2.0, 6.0]).unwrap());
        assert!(matches!(
            outputs[0].tangent(),
            MaybeZero::Zero(r#type) if r#type == &ArrayType::new_static(DataType::F64, [3]),
        ));
    }

    #[test]
    fn test_cumulative_differentiation_extrema_ties() {
        // Running extrema are not differentiable where elements tie. The decomposition inherits the convention of the
        // elementwise extrema, which split the tangent evenly between tied elements.
        for kind in [CumulativeKind::Max, CumulativeKind::Min] {
            let (primal, tangent) = differentiate_at(Array::vector(vec![1.0, 1.0]).unwrap())
                .jvp(Array::vector(vec![1.0, 3.0]).unwrap(), |input| Ok(input.cumulative(0, kind, false)?))
                .unwrap();
            assert_eq!(primal, Array::vector(vec![1.0, 1.0]).unwrap());
            assert_eq!(tangent, Array::vector(vec![1.0, 2.0]).unwrap());
        }
    }

    #[test]
    fn test_cumulative_differentiation_associative_scan_decomposition() {
        // A nonlinear kind's forward mode differentiates *through* the decomposition, so the fused program holds no
        // `cumulative` instruction at all: it is the parallel-prefix construction (two halving levels over a
        // length-four axis) with each of its primitives' own rules interleaved. The primal half is recomputed there
        // rather than taken from the primitive.
        let jvp = cumulative_program(
            CumulativeOperation::new(0, CumulativeKind::Product),
            ArrayType::new_static(DataType::F64, [4]),
        )
        .jvp()
        .unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4] .
                let %2:f64[2] = slice [start_indices=[0], limits=[3], strides=[2]] %0
                    %3:f64[2] = slice [start_indices=[0], limits=[3], strides=[2]] %1
                    %4:f64[2] = slice [start_indices=[1], limits=[4], strides=[2]] %0
                    %5:f64[2] = slice [start_indices=[1], limits=[4], strides=[2]] %1
                    %6:f64[2] = mul %2 %4
                    %7:f64[2] = mul %4 %3
                    %8:f64[2] = mul %2 %5
                    %9:f64[2] = add %7 %8
                    %10:f64[1] = slice [start_indices=[0], limits=[1], strides=[2]] %6
                    %11:f64[1] = slice [start_indices=[0], limits=[1], strides=[2]] %9
                    %12:f64[1] = slice [start_indices=[1], limits=[2], strides=[2]] %6
                    %13:f64[1] = slice [start_indices=[1], limits=[2], strides=[2]] %9
                    %14:f64[1] = mul %10 %12
                    %15:f64[1] = mul %12 %11
                    %16:f64[1] = mul %10 %13
                    %17:f64[1] = add %15 %16
                    %18:f64[1] = slice [start_indices=[0], limits=[1]] %6
                    %19:f64[1] = slice [start_indices=[0], limits=[1]] %9
                    %20:f64[] = zero [type=f64[]]
                    %21:f64[2] = pad [edge_padding_low=[0], edge_padding_high=[1], interior_padding=[1]] %18 %20
                    %22:f64[] = zero [type=f64[]]
                    %23:f64[2] = pad [edge_padding_low=[0], edge_padding_high=[1], interior_padding=[1]] %19 %22
                    %24:f64[2] = pad [edge_padding_low=[1], edge_padding_high=[0], interior_padding=[1]] %14 %20
                    %25:f64[] = zero [type=f64[]]
                    %26:f64[2] = pad [edge_padding_low=[1], edge_padding_high=[0], interior_padding=[1]] %17 %25
                    %27:f64[2] = add %21 %24
                    %28:f64[2] = add %23 %26
                    %29:f64[1] = slice [start_indices=[0], limits=[1]] %0
                    %30:f64[1] = slice [start_indices=[0], limits=[1]] %1
                    %31:f64[1] = slice [start_indices=[0], limits=[1]] %27
                    %32:f64[1] = slice [start_indices=[0], limits=[1]] %28
                    %33:f64[1] = slice [start_indices=[2], limits=[4], strides=[2]] %0
                    %34:f64[1] = slice [start_indices=[2], limits=[4], strides=[2]] %1
                    %35:f64[1] = mul %31 %33
                    %36:f64[1] = mul %33 %32
                    %37:f64[1] = mul %31 %34
                    %38:f64[1] = add %36 %37
                    %39:f64[2] = concatenate [axis=0] %29 %35
                    %40:f64[2] = concatenate [axis=0] %30 %38
                    %41:f64[] = zero [type=f64[]]
                    %42:f64[4] = pad [edge_padding_low=[0], edge_padding_high=[1], interior_padding=[1]] %39 %41
                    %43:f64[] = zero [type=f64[]]
                    %44:f64[4] = pad [edge_padding_low=[0], edge_padding_high=[1], interior_padding=[1]] %40 %43
                    %45:f64[4] = pad [edge_padding_low=[1], edge_padding_high=[0], interior_padding=[1]] %27 %41
                    %46:f64[] = zero [type=f64[]]
                    %47:f64[4] = pad [edge_padding_low=[1], edge_padding_high=[0], interior_padding=[1]] %28 %46
                    %48:f64[4] = add %42 %45
                    %49:f64[4] = add %44 %47
                in (%48, %49)"
            },
        );
    }

    #[test]
    fn test_cumulative_differentiation_associative_scan_provenance() {
        // The decomposition that a nonlinear kind differentiates through is staged under the framework's
        // differentiation scope, which wraps the `associative_scan` scope of the function that stages it.
        let jvp = cumulative_program(
            CumulativeOperation::new(0, CumulativeKind::Max),
            ArrayType::new_static(DataType::F64, [2]),
        )
        .jvp()
        .unwrap();
        assert_eq!(
            std::fmt::from_fn(|formatter| jvp.render(formatter, 0, ProgramRenderingMode::WithProvenance)).to_string(),
            indoc! {"
                lambda %0:f64[2], %1:f64[2] .
                let %2:f64[1] = slice [start_indices=[0], limits=[1], strides=[2]] %0 ; provenance=ryft::differentiation::ryft::associative_scan
                    %3:f64[1] = slice [start_indices=[0], limits=[1], strides=[2]] %1 ; provenance=ryft::differentiation::ryft::associative_scan
                    %4:f64[1] = slice [start_indices=[1], limits=[2], strides=[2]] %0 ; provenance=ryft::differentiation::ryft::associative_scan
                    %5:f64[1] = slice [start_indices=[1], limits=[2], strides=[2]] %1 ; provenance=ryft::differentiation::ryft::associative_scan
                    %6:f64[1] = max %2 %4 ; provenance=ryft::differentiation::ryft::associative_scan
                    %7:bool[1] = compare [direction=GreaterThan] %2 %4 ; provenance=ryft::differentiation::ryft::associative_scan
                    %8:bool[1] = compare [direction=Equal] %2 %4 ; provenance=ryft::differentiation::ryft::associative_scan
                    %9:f64[1] = one_like %2 ; provenance=ryft::differentiation::ryft::associative_scan
                    %10:f64[1] = add %9 %9 ; provenance=ryft::differentiation::ryft::associative_scan
                    %11:f64[1] = div %9 %10 ; provenance=ryft::differentiation::ryft::associative_scan
                    %12:f64[1] = zero_like %2 ; provenance=ryft::differentiation::ryft::associative_scan
                    %13:f64[1] = select %8 %11 %12 ; provenance=ryft::differentiation::ryft::associative_scan
                    %14:f64[1] = select %7 %9 %13 ; provenance=ryft::differentiation::ryft::associative_scan
                    %15:f64[1] = mul %14 %3 ; provenance=ryft::differentiation::ryft::associative_scan
                    %16:bool[1] = compare [direction=GreaterThan] %4 %2 ; provenance=ryft::differentiation::ryft::associative_scan
                    %17:bool[1] = compare [direction=Equal] %4 %2 ; provenance=ryft::differentiation::ryft::associative_scan
                    %18:f64[1] = one_like %4 ; provenance=ryft::differentiation::ryft::associative_scan
                    %19:f64[1] = add %18 %18 ; provenance=ryft::differentiation::ryft::associative_scan
                    %20:f64[1] = div %18 %19 ; provenance=ryft::differentiation::ryft::associative_scan
                    %21:f64[1] = zero_like %4 ; provenance=ryft::differentiation::ryft::associative_scan
                    %22:f64[1] = select %17 %20 %21 ; provenance=ryft::differentiation::ryft::associative_scan
                    %23:f64[1] = select %16 %18 %22 ; provenance=ryft::differentiation::ryft::associative_scan
                    %24:f64[1] = mul %23 %5 ; provenance=ryft::differentiation::ryft::associative_scan
                    %25:f64[1] = add %15 %24 ; provenance=ryft::differentiation::ryft::associative_scan
                    %26:f64[1] = slice [start_indices=[0], limits=[1]] %0 ; provenance=ryft::differentiation::ryft::associative_scan
                    %27:f64[1] = slice [start_indices=[0], limits=[1]] %1 ; provenance=ryft::differentiation::ryft::associative_scan
                    %28:f64[] = zero [type=f64[]] ; provenance=ryft::differentiation::ryft::associative_scan
                    %29:f64[2] = pad [edge_padding_low=[0], edge_padding_high=[1], interior_padding=[1]] %26 %28 ; provenance=ryft::differentiation::ryft::associative_scan
                    %30:f64[] = zero [type=f64[]] ; provenance=ryft::differentiation::ryft::associative_scan
                    %31:f64[2] = pad [edge_padding_low=[0], edge_padding_high=[1], interior_padding=[1]] %27 %30 ; provenance=ryft::differentiation::ryft::associative_scan
                    %32:f64[2] = pad [edge_padding_low=[1], edge_padding_high=[0], interior_padding=[1]] %6 %28 ; provenance=ryft::differentiation::ryft::associative_scan
                    %33:f64[] = zero [type=f64[]] ; provenance=ryft::differentiation::ryft::associative_scan
                    %34:f64[2] = pad [edge_padding_low=[1], edge_padding_high=[0], interior_padding=[1]] %25 %33 ; provenance=ryft::differentiation::ryft::associative_scan
                    %35:f64[2] = add %29 %32 ; provenance=ryft::differentiation::ryft::associative_scan
                    %36:f64[2] = add %31 %34 ; provenance=ryft::differentiation::ryft::associative_scan
                in (%35, %36)"
            },
        );
    }

    #[test]
    fn test_cumulative_differentiation_layout() {
        // The decomposition's manipulation primitives drop an explicit layout, so the rule constrains its output back
        // onto the input's layout, and the primal and tangent outputs have exactly the primitive's output type. The
        // layout leaves their logical values, the running maximum of each row and its tangent, unchanged.
        let laid_out =
            ArrayType::new_static(DataType::F64, [2, 2]).with_layout(Layout::Strided(StridedLayout::new(vec![8, 16])));
        let jvp = cumulative_program(CumulativeOperation::new(1, CumulativeKind::Max), laid_out.clone())
            .jvp()
            .unwrap();
        assert_eq!(jvp.output_types(), vec![laid_out.clone(), laid_out.clone()]);
        assert_eq!(
            jvp.interpret(vec![
                Array::from_elements::<f64>(laid_out.clone(), &[1.0, 3.0, 2.0, 0.0]).unwrap(),
                Array::from_elements::<f64>(laid_out.clone(), &[1.0, 2.0, 3.0, 4.0]).unwrap(),
            ]),
            Ok(vec![
                Array::from_elements::<f64>(laid_out.clone(), &[1.0, 3.0, 2.0, 2.0]).unwrap(),
                Array::from_elements::<f64>(laid_out, &[1.0, 2.0, 3.0, 3.0]).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_cumulative_differentiation_dynamic_unscanned_axis() {
        // Only the scanned axis must be static: the decomposition keeps every other axis whole, so each nonlinear kind
        // differentiates over a dynamic unscanned axis in forward mode, and its linearized tangent transposes into a
        // pullback over the same dynamic type. On a concrete input, the dynamic programs compute exactly what the
        // statically shaped ones do.
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(3)]));
        let static_type = ArrayType::new_static(DataType::F64, [2, 3]);
        let primal = Array::matrix(2, 3, vec![0.5, 2.0, 1.0, -1.0, 3.0, 0.25]).unwrap();
        let tangent = Array::matrix(2, 3, vec![1.0, -2.0, 0.5, 3.0, 1.0, -1.0]).unwrap();
        let cotangent = Array::matrix(2, 3, vec![2.0, -1.0, 0.5, 1.0, 3.0, -2.0]).unwrap();

        // The pullback of a program's linearization at `primal`, applied to `cotangent`, along with its output types.
        let pullback = |program: &Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>| {
            let linearization = program.linearize().unwrap();
            let mut primal_outputs = linearization.primal().interpret(vec![primal.clone()]).unwrap();
            let residuals = primal_outputs.split_off(1);
            let transposed = linearization.tangent().transpose_with_respect_to(&[0], &[]).unwrap();
            let cotangents = transposed.interpret([vec![cotangent.clone()], residuals].concat());
            (transposed.output_types(), cotangents)
        };
        for kind in [CumulativeKind::Product, CumulativeKind::LogSumExp, CumulativeKind::Max, CumulativeKind::Min] {
            let dynamic = cumulative_program(CumulativeOperation::new(1, kind), dynamic_type.clone());
            let r#static = cumulative_program(CumulativeOperation::new(1, kind), static_type.clone());
            let jvp = dynamic.jvp().unwrap();
            assert_eq!(jvp.output_types(), vec![dynamic_type.clone(), dynamic_type.clone()], "{kind}");
            assert_eq!(
                jvp.interpret(vec![primal.clone(), tangent.clone()]),
                r#static.jvp().unwrap().interpret(vec![primal.clone(), tangent.clone()]),
                "{kind}",
            );
            let (dynamic_types, dynamic_cotangents) = pullback(&dynamic);
            let (_, static_cotangents) = pullback(&r#static);
            assert_eq!(dynamic_types, vec![dynamic_type.clone()], "{kind}");
            assert_eq!(dynamic_cotangents, static_cotangents, "{kind}");
        }
    }

    #[test]
    fn test_cumulative_differentiation_reverse_mode() {
        // Summing the prefix sums weights each input by the number of outputs it contributes to, which the transposed
        // sum computes directly. Nonlinear kinds reach reverse mode in either direction by transposing the linear
        // operations that their decomposition stages, which finite differences independently confirm.
        assert_eq!(
            differentiate_at(Array::vector(vec![1.0, 2.0, 3.0]).unwrap())
                .gradient(|input| Ok(input.cumulative_sum(0)?.reduce_sum(&[0], None)?))
                .unwrap(),
            Array::vector(vec![3.0, 2.0, 1.0]).unwrap(),
        );
        for kind in [CumulativeKind::Product, CumulativeKind::LogSumExp, CumulativeKind::Max, CumulativeKind::Min] {
            for reverse in [false, true] {
                check_gradient!(
                    |input| Ok(input.cumulative(0, kind, reverse)?.reduce_sum(&[0], None)?),
                    at = Array::vector(vec![1.0, 3.0, 2.0]).unwrap(),
                    step = 1e-3,
                    tolerance = 1e-6,
                );
            }
        }
    }

    #[test]
    fn test_cumulative_transposition() {
        // The adjoint of a forward prefix sum is a reverse prefix sum of the output cotangent, staged as the same
        // primitive with its `reverse` flag flipped, and vice versa.
        check_operation_transposition!(
            @exact,
            operation = CumulativeOperation::new(0, CumulativeKind::Sum),
            cases = [{
                inputs = [(@linear(type = ArrayType::new_static(DataType::F64, [3])))],
                output_cotangents = [Array::vector(vec![1.0, 1.0, 1.0]).unwrap()],
                input_cotangents = [Array::vector(vec![3.0, 2.0, 1.0]).unwrap()],
                pullback = indoc! {"
                    lambda %0:f64[3] .
                    let %1:f64[3] = cumulative [kind=sum, axis=0, reverse=true] %0
                    in (%1)
                "},
            }],
        );
        check_operation_transposition!(
            @exact,
            operation = CumulativeOperation::new(0, CumulativeKind::Sum).with_reverse(true),
            cases = [{
                inputs = [(@linear(type = ArrayType::new_static(DataType::F64, [3])))],
                output_cotangents = [Array::vector(vec![1.0, 1.0, 1.0]).unwrap()],
                input_cotangents = [Array::vector(vec![1.0, 2.0, 3.0]).unwrap()],
                pullback = indoc! {"
                    lambda %0:f64[3] .
                    let %1:f64[3] = cumulative [kind=sum, axis=0] %0
                    in (%1)
                "},
            }],
        );
    }

    #[test]
    fn test_cumulative_transposition_zero_cotangent() {
        // A structural zero output cotangent contributes a structural zero input cotangent without staging a scan.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let mut transposition = TranspositionContext::new(context.clone());
        let inputs = [PartialValue::Unknown(ArrayType::new_static(DataType::F64, [3]))];
        let accumulators = transposition.cotangent_accumulators(&inputs, &[]).unwrap();
        CumulativeOperation::new(0, CumulativeKind::Sum)
            .transpose(
                &mut transposition,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Zero(ArrayType::new_static(DataType::F64, [3]))],
                &accumulators,
            )
            .unwrap();
        let input_cotangents = transposition.take_cotangents(&accumulators).unwrap();
        assert_eq!(input_cotangents.len(), 1);
        assert!(matches!(
            &input_cotangents[0],
            MaybeZero::Zero(r#type) if r#type == &ArrayType::new_static(DataType::F64, [3]),
        ));
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_cumulative_transposition_nonlinear_kinds() {
        // Only sums are linear. Every other kind is differentiated through the linear operations staged by its JVP
        // instead, and so direct transposition rejects it, even when the cotangent is a structural zero.
        for kind in [CumulativeKind::Product, CumulativeKind::LogSumExp, CumulativeKind::Max, CumulativeKind::Min] {
            let context = TracingContext::<Array, ArrayOperation<Array>>::new();
            let output_cotangent = {
                let atom = context.builder().borrow_mut().add_input(ArrayType::new_static(DataType::F64, [3]));
                context.tracer(atom, None)
            };
            let inputs = [PartialValue::Unknown(ArrayType::new_static(DataType::F64, [3]))];
            for output_cotangent in
                [MaybeZero::Value(output_cotangent), MaybeZero::Zero(ArrayType::new_static(DataType::F64, [3]))]
            {
                let mut transposition = TranspositionContext::new(context.clone());
                let accumulators = transposition.cotangent_accumulators(&inputs, &[]).unwrap();
                assert!(matches!(
                    CumulativeOperation::new(0, kind).transpose(
                        &mut transposition,
                        &EmptyRegionDriver,
                        &inputs,
                        &[output_cotangent],
                        &accumulators,
                    ),
                    Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                        if message == format!("`cumulative` with kind `{kind}` is not directly transposable"),
                ));
            }
        }
    }

    #[test]
    fn test_cumulative_cumulative() {
        // Negative axes count from the end and select the same scan as the corresponding nonnegative axis, for every
        // kind and direction, both for concrete arrays and for context-carrying values that bind the operation.
        let input = Array::matrix(2, 3, vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        for (axis, position) in [(-1, 1usize), (-2, 0usize)] {
            for kind in [
                CumulativeKind::Sum,
                CumulativeKind::Product,
                CumulativeKind::LogSumExp,
                CumulativeKind::Max,
                CumulativeKind::Min,
            ] {
                for reverse in [false, true] {
                    let expected = input.cumulative(position, kind, reverse).unwrap();
                    assert_eq!(input.cumulative(axis, kind, reverse), Ok(expected.clone()));
                    let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
                        |value: Tracer<TracingContext<Array, ArrayOperation<Array>>>| {
                            value.cumulative(Axis::from(axis), kind, reverse)
                        },
                        input.r#type().into_owned(),
                    )
                    .unwrap();
                    assert_eq!(program.interpret(input.clone()), Ok(expected));
                }
            }
        }

        // Out-of-bounds axes are reported as given, before any operation is bound, on both paths.
        for (input, axis) in [(input.clone(), -3), (input, 2), (Array::scalar(1.0f64).unwrap(), -1)] {
            let error = ProgramError::Type(TypeError::invalid(format!(
                "`cumulative` axis {axis} is out of bounds for rank {}",
                input.r#type().rank(),
            )));
            assert_eq!(input.cumulative_sum(axis), Err(error.clone()));
            let result = TracingContext::<Array, ArrayOperation<Array>>::trace(
                |value: Tracer<TracingContext<Array, ArrayOperation<Array>>>| value.cumulative_sum(axis),
                input.r#type().into_owned(),
            );
            assert!(matches!(result, Err(actual) if actual == error));
        }
    }

    #[test]
    fn test_cumulative_cumulative_sum() {
        let input = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(input.cumulative_sum(0), Ok(Array::vector(vec![1.0, 3.0, 6.0]).unwrap()));
    }

    #[test]
    fn test_cumulative_reverse_cumulative_sum() {
        let input = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(input.reverse_cumulative_sum(0), Ok(Array::vector(vec![6.0, 5.0, 3.0]).unwrap()));
    }

    #[test]
    fn test_cumulative_cumulative_product() {
        let input = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(input.cumulative_product(0), Ok(Array::vector(vec![1.0, 2.0, 6.0]).unwrap()));
    }

    #[test]
    fn test_cumulative_reverse_cumulative_product() {
        let input = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(input.reverse_cumulative_product(0), Ok(Array::vector(vec![6.0, 6.0, 3.0]).unwrap()));
    }

    #[test]
    fn test_cumulative_cumulative_max() {
        let input = Array::vector(vec![3.0, 1.0, 4.0]).unwrap();
        assert_eq!(input.cumulative_max(0), Ok(Array::vector(vec![3.0, 3.0, 4.0]).unwrap()));
    }

    #[test]
    fn test_cumulative_reverse_cumulative_max() {
        let input = Array::vector(vec![3.0, 1.0, 4.0]).unwrap();
        assert_eq!(input.reverse_cumulative_max(0), Ok(Array::vector(vec![4.0, 4.0, 4.0]).unwrap()));
    }

    #[test]
    fn test_cumulative_cumulative_min() {
        let input = Array::vector(vec![3.0, 1.0, 4.0]).unwrap();
        assert_eq!(input.cumulative_min(0), Ok(Array::vector(vec![3.0, 1.0, 1.0]).unwrap()));
    }

    #[test]
    fn test_cumulative_reverse_cumulative_min() {
        let input = Array::vector(vec![3.0, 1.0, 4.0]).unwrap();
        assert_eq!(input.reverse_cumulative_min(0), Ok(Array::vector(vec![1.0, 1.0, 4.0]).unwrap()));
    }

    #[test]
    fn test_cumulative_cumulative_log_sum_exp() {
        let input = Array::vector(vec![0.0, 0.0]).unwrap();
        assert_eq!(input.cumulative_log_sum_exp(0), Ok(Array::vector(vec![0.0, std::f64::consts::LN_2]).unwrap()));
    }

    #[test]
    fn test_cumulative_reverse_cumulative_log_sum_exp() {
        let input = Array::vector(vec![0.0, 0.0]).unwrap();
        assert_eq!(
            input.reverse_cumulative_log_sum_exp(0),
            Ok(Array::vector(vec![std::f64::consts::LN_2, 0.0]).unwrap()),
        );
    }

    #[test]
    fn test_array_cumulative() {
        // Forward scans accumulate prefixes and reverse scans accumulate suffixes, along the selected axis only.
        let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        for (kind, forward_columns, reverse_columns, forward_rows) in [
            (
                CumulativeKind::Sum,
                vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0],
                vec![6.0, 5.0, 3.0, 15.0, 11.0, 6.0],
                vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0],
            ),
            (
                CumulativeKind::Product,
                vec![1.0, 2.0, 6.0, 4.0, 20.0, 120.0],
                vec![6.0, 6.0, 3.0, 120.0, 30.0, 6.0],
                vec![1.0, 2.0, 3.0, 4.0, 10.0, 18.0],
            ),
        ] {
            assert_eq!(input.cumulative(1, kind, false), Ok(Array::matrix(2, 3, forward_columns).unwrap()));
            assert_eq!(input.cumulative(1, kind, true), Ok(Array::matrix(2, 3, reverse_columns).unwrap()));
            assert_eq!(input.cumulative(0, kind, false), Ok(Array::matrix(2, 3, forward_rows).unwrap()));
        }
        let input = Array::matrix(2, 3, vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0]).unwrap();
        for (kind, forward_columns, reverse_columns, forward_rows) in [
            (
                CumulativeKind::Max,
                vec![3.0, 3.0, 4.0, 1.0, 5.0, 9.0],
                vec![4.0, 4.0, 4.0, 9.0, 9.0, 9.0],
                vec![3.0, 1.0, 4.0, 3.0, 5.0, 9.0],
            ),
            (
                CumulativeKind::Min,
                vec![3.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                vec![1.0, 1.0, 4.0, 1.0, 5.0, 9.0],
                vec![3.0, 1.0, 4.0, 1.0, 1.0, 4.0],
            ),
        ] {
            assert_eq!(input.cumulative(1, kind, false), Ok(Array::matrix(2, 3, forward_columns).unwrap()));
            assert_eq!(input.cumulative(1, kind, true), Ok(Array::matrix(2, 3, reverse_columns).unwrap()));
            assert_eq!(input.cumulative(0, kind, false), Ok(Array::matrix(2, 3, forward_rows).unwrap()));
        }

        // Integer payloads accumulate and select in their own element type, and their arithmetic wraps in it.
        let integers = Array::vector(vec![7i32, -2, 3, 4]).unwrap();
        assert_eq!(integers.cumulative_sum(0), Ok(Array::vector(vec![7i32, 5, 8, 12]).unwrap()));
        assert_eq!(integers.cumulative_product(0), Ok(Array::vector(vec![7i32, -14, -42, -168]).unwrap()));
        assert_eq!(integers.cumulative_max(0), Ok(Array::vector(vec![7i32, 7, 7, 7]).unwrap()));
        assert_eq!(integers.cumulative_min(0), Ok(Array::vector(vec![7i32, -2, -2, -2]).unwrap()));
        assert_eq!(
            Array::vector(vec![100i8, 100]).unwrap().cumulative_sum(0),
            Ok(Array::vector(vec![100i8, -56]).unwrap()),
        );

        // A zero-length scanned axis has nothing to accumulate and keeps the input's exact type.
        let empty = Array::new(ArrayType::new_static(DataType::F32, [0, 2]), Vec::new()).unwrap();
        for kind in [
            CumulativeKind::Sum,
            CumulativeKind::Product,
            CumulativeKind::LogSumExp,
            CumulativeKind::Max,
            CumulativeKind::Min,
        ] {
            assert_eq!(empty.cumulative(0, kind, false), Ok(empty.clone()));
        }

        // The kernel reports the type rule's validation errors instead of panicking.
        assert_eq!(
            input.cumulative(2, CumulativeKind::Sum, false),
            Err(ProgramError::Type(TypeError::invalid("`cumulative` axis 2 is out of bounds for rank 2"))),
        );
    }

    #[test]
    fn test_array_cumulative_element_encodings() {
        // Accumulation happens in the input's own encoding, so every partial sum is re-encoded rather than only the
        // final one. Each increment below is smaller than half a `bf16` step next to one, yet the running sum still
        // climbs by a full step each time, because each partial sum rounds up on its own. Summing the four increments
        // exactly and rounding once would stop one step short, which is what pins per-step re-encoding.
        let low_precision_type = ArrayType::new_static(DataType::BF16, [5]);
        let increment = f64::from(bf16::from_f64(0.005));
        assert_eq!(
            Array::from_elements::<bf16>(
                low_precision_type.clone(),
                &[1.0, 0.005, 0.005, 0.005, 0.005].map(bf16::from_f64),
            )
            .unwrap()
            .cumulative_sum(0),
            Ok(Array::from_elements::<bf16>(
                low_precision_type,
                &[1.0, 1.0078125, 1.015625, 1.0234375, 1.03125].map(bf16::from_f64),
            )
            .unwrap()),
        );
        assert_eq!(f64::from(bf16::from_f64(1.0 + 4.0 * increment)), 1.0234375);

        // `f8e4m3fn` cannot represent 15, so the last prefix sum rounds to the nearest representable value. The third
        // prefix product is exactly halfway between two values and rounds down to an even mantissa, which drags the
        // fourth prefix product one step below the exactly accumulated product (`2 * 1.25 * 1.25 * 1.25 = 3.90625`,
        // which itself rounds to 4).
        let low_precision_type = ArrayType::new_static(DataType::F8E4M3FN, [4]);
        let low_precision = |values: [f64; 4]| {
            Array::from_elements::<f8e4m3fn>(
                low_precision_type.clone(),
                &values.map(|value| f8e4m3fn::from_f64(value).unwrap()),
            )
            .unwrap()
        };
        assert_eq!(low_precision([1.0, 2.0, 4.0, 8.0]).cumulative_sum(0), Ok(low_precision([1.0, 3.0, 7.0, 15.0])));
        assert_eq!(
            low_precision([2.0, 1.25, 1.25, 1.25]).cumulative_product(0),
            Ok(low_precision([2.0, 2.5, 3.0, 3.75])),
        );

        // The payload-free structural zero has no bytes to scan, and every prefix sum or product of a zero is a zero.
        let structural_zero = Array::new(ArrayType::new_static(DataType::Zero, [3]), Vec::new()).unwrap();
        assert_eq!(structural_zero.cumulative_sum(0), Ok(structural_zero.clone()));
        assert_eq!(structural_zero.cumulative_product(0), Ok(structural_zero));

        // Complex payloads accumulate both components and multiply as complex numbers, except that an exact `1 + 0i`
        // factor returns the other factor, because multiplying an infinite component by its zero imaginary component
        // would otherwise produce NaN (e.g., `(∞ + 0i)(1 + 0i)` would become `∞ + NaNi`).
        let complex = Array::vector(vec![
            ComplexNumber::new(1.0f64, 1.0),
            ComplexNumber::new(2.0, -1.0),
            ComplexNumber::new(3.0, 5.0),
        ])
        .unwrap();
        assert_eq!(
            complex.cumulative_sum(0),
            Ok(Array::vector(vec![
                ComplexNumber::new(1.0f64, 1.0),
                ComplexNumber::new(3.0, 0.0),
                ComplexNumber::new(6.0, 5.0),
            ])
            .unwrap()),
        );
        assert_eq!(
            Array::vector(vec![ComplexNumber::new(0.0f64, 1.0); 3]).unwrap().cumulative_product(0),
            Ok(Array::vector(vec![
                ComplexNumber::new(0.0f64, 1.0),
                ComplexNumber::new(-1.0, 0.0),
                ComplexNumber::new(0.0, -1.0),
            ])
            .unwrap()),
        );
        let infinite =
            Array::vector(vec![ComplexNumber::new(f64::INFINITY, 0.0), ComplexNumber::new(1.0, 0.0)]).unwrap();
        assert_eq!(
            infinite.cumulative_product(0),
            Ok(Array::vector(vec![ComplexNumber::new(f64::INFINITY, 0.0); 2]).unwrap()),
        );
        assert_eq!(
            infinite.reverse_cumulative_product(0),
            Ok(Array::vector(vec![ComplexNumber::new(f64::INFINITY, 0.0), ComplexNumber::new(1.0, 0.0)]).unwrap()),
        );

        // The result carries the input's complete type, including a non-default physical layout.
        let laid_out =
            ArrayType::new_static(DataType::F32, [2, 2]).with_layout(Layout::Strided(StridedLayout::new(vec![4, 8])));
        assert_eq!(
            Array::from_elements::<f32>(laid_out.clone(), &[1.0, 2.0, 3.0, 4.0]).unwrap().cumulative_sum(1),
            Ok(Array::from_elements::<f32>(laid_out, &[1.0, 3.0, 3.0, 7.0]).unwrap()),
        );
    }

    #[test]
    fn test_array_cumulative_extrema() {
        // Complex elements compare their real parts first and their imaginary parts second.
        let complex = Array::vector(vec![
            ComplexNumber::new(1.0f32, 5.0),
            ComplexNumber::new(2.0, -3.0),
            ComplexNumber::new(2.0, 4.0),
        ])
        .unwrap();
        assert_eq!(
            complex.cumulative_max(0),
            Ok(Array::vector(vec![
                ComplexNumber::new(1.0f32, 5.0),
                ComplexNumber::new(2.0, -3.0),
                ComplexNumber::new(2.0, 4.0),
            ])
            .unwrap()),
        );
        let expected = Array::vector(vec![ComplexNumber::new(1.0f32, 5.0); 3]).unwrap();
        assert_eq!(complex.cumulative_min(0), Ok(expected));

        // Selection happens in the input's own element type, so a low-precision payload is returned bit for bit
        // rather than through a widened intermediate.
        let low_precision_type = ArrayType::new_static(DataType::F8E5M2, [3]);
        assert_eq!(
            Array::from_elements::<f8e5m2>(
                low_precision_type.clone(),
                &[0.5, 6.0, 1.5].map(|value| f8e5m2::from_f64(value).unwrap()),
            )
            .unwrap()
            .cumulative_max(0),
            Ok(Array::from_elements::<f8e5m2>(
                low_precision_type,
                &[0.5, 6.0, 6.0].map(|value| f8e5m2::from_f64(value).unwrap()),
            )
            .unwrap()),
        );

        // NaNs propagate through the elementwise extrema, so every later prefix of a NaN is NaN, and `-0.0` orders
        // below `+0.0`.
        for kind in [CumulativeKind::Max, CumulativeKind::Min] {
            let values = Array::vector(vec![1.0, f64::NAN, 2.0]).unwrap().cumulative(0, kind, false).unwrap().to_f64s();
            assert_eq!(values[0], 1.0);
            assert!(values[1].is_nan() && values[2].is_nan());
        }
        let zeros = Array::vector(vec![-0.0f64, 0.0]).unwrap();
        assert_eq!(zeros.cumulative_max(0).unwrap().to_f64s()[1].to_bits(), 0.0f64.to_bits());
        assert_eq!(zeros.reverse_cumulative_min(0).unwrap().to_f64s()[0].to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn test_array_cumulative_log_sum_exp() {
        // Forward scans accumulate prefixes and reverse scans accumulate suffixes, along the selected axis only, by
        // folding the pairwise stable `log_add_exp`.
        let input = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        let forward_second = log_add_exp(1.0, 2.0);
        let reverse_second = log_add_exp(3.0, 2.0);
        assert_eq!(
            input.cumulative_log_sum_exp(0),
            Ok(Array::vector(vec![1.0, forward_second, log_add_exp(forward_second, 3.0)]).unwrap()),
        );
        assert_eq!(
            input.reverse_cumulative_log_sum_exp(0),
            Ok(Array::vector(vec![log_add_exp(reverse_second, 1.0), reverse_second, 3.0]).unwrap()),
        );
        assert_eq!(
            Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap().cumulative_log_sum_exp(0),
            Ok(Array::matrix(
                2,
                3,
                vec![1.0, 2.0, 3.0, log_add_exp(1.0, 4.0), log_add_exp(2.0, 5.0), log_add_exp(3.0, 6.0)],
            )
            .unwrap()),
        );

        // Folding the guarded pairwise primitive keeps the scan exact where exponentiating directly would overflow: two
        // equal elements add exactly `log(2)` at any magnitude, in both directions.
        let large = Array::vector(vec![1000.0, 1000.0]).unwrap();
        assert_eq!(
            large.cumulative_log_sum_exp(0),
            Ok(Array::vector(vec![1000.0, 1000.0 + std::f64::consts::LN_2]).unwrap()),
        );
        assert_eq!(
            large.reverse_cumulative_log_sum_exp(0),
            Ok(Array::vector(vec![1000.0 + std::f64::consts::LN_2, 1000.0]).unwrap()),
        );

        // Negative infinity is the combining operator's identity, so it neither contributes to nor poisons a later
        // prefix, while a NaN input propagates.
        assert_eq!(
            Array::vector(vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 2.0]).unwrap().cumulative_log_sum_exp(0),
            Ok(Array::vector(vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 2.0]).unwrap()),
        );
        let with_nan = Array::vector(vec![1.0, f64::NAN, 2.0]).unwrap().cumulative_log_sum_exp(0).unwrap().to_f64s();
        assert_eq!(with_nan[0], 1.0);
        assert!(with_nan[1].is_nan() && with_nan[2].is_nan());

        // Accumulation happens in the input's own encoding, so each partial result is rounded to it, and the lowest
        // values of the supported finite-only formats remain identities for any number of rounded combinations.
        let single_precision = ArrayType::new_static(DataType::F32, [2]);
        assert_eq!(
            Array::from_elements::<f32>(single_precision.clone(), &[0.0, 0.0])
                .unwrap()
                .cumulative_log_sum_exp(0),
            Ok(Array::from_elements::<f32>(single_precision, &[0.0, std::f32::consts::LN_2]).unwrap()),
        );
        let lowest = Array::vector(vec![f4e2m1fn::MIN; 16]).unwrap();
        assert_eq!(lowest.cumulative_log_sum_exp(0), Ok(lowest));
        let lowest = Array::vector(vec![f8e4m3fnuz::MIN; 3000]).unwrap();
        assert_eq!(lowest.cumulative_log_sum_exp(0), Ok(lowest));

        // Complex prefixes use the principal logarithm, and an argument whose real component is negative infinity has
        // a zero exponential, so it leaves the other argument unchanged. That includes a pair of such arguments, whose
        // elementwise complex combination would subtract `-∞` from `-∞` and produce NaN; the reverse scan combines one.
        let first = ComplexNumber::new(1.0f64, 2.0);
        let doubled = FloatingPointArrayElement::log_add_exp(first, first).unwrap();
        assert!((doubled - ComplexNumber::new(1.0 + std::f64::consts::LN_2, 2.0)).norm() < 1e-15);
        let complex = Array::vector(vec![
            first,
            first,
            ComplexNumber::new(f64::NEG_INFINITY, 0.0),
            ComplexNumber::new(f64::NEG_INFINITY, 3.0),
        ])
        .unwrap();
        assert_eq!(
            complex.cumulative_log_sum_exp(0),
            Ok(Array::vector(vec![first, doubled, doubled, doubled]).unwrap()),
        );
        assert_eq!(
            complex.reverse_cumulative_log_sum_exp(0),
            Ok(Array::vector(vec![
                doubled,
                first,
                ComplexNumber::new(f64::NEG_INFINITY, 3.0),
                ComplexNumber::new(f64::NEG_INFINITY, 3.0),
            ])
            .unwrap()),
        );

        // A prefix that no combination produced is still wrapped onto the principal branch, like every combined prefix
        // and like the matching log-sum-exp reduction.
        let raw = Array::vector(vec![ComplexNumber::new(1.0f64, 4.0), ComplexNumber::new(f64::NEG_INFINITY, 0.0)])
            .unwrap()
            .cumulative_log_sum_exp(0)
            .unwrap()
            .elements::<ComplexNumber<f64>>()
            .unwrap();
        for value in raw {
            assert!((value - ComplexNumber::new(1.0, 4.0 - 2.0 * std::f64::consts::PI)).norm() < 1e-15);
        }

        // Out-of-range phases are reduced through `atan2` of their sine and cosine, as in the matching reduction, which
        // stays accurate for huge phases, where a remainder by the rounded `2π` would accumulate its rounding error
        // once per period.
        let huge = Array::vector(vec![ComplexNumber::new(1.0f64, 1e16)]).unwrap();
        let reduced = ComplexNumber::new(1.0f64, 1e16f64.sin().atan2(1e16f64.cos()));
        assert_eq!(huge.cumulative_log_sum_exp(0), Ok(Array::vector(vec![reduced]).unwrap()));
        assert_eq!(huge.reduce_log_sum_exp(&[0]), Ok(Array::scalar(reduced).unwrap()));
    }

    #[test]
    fn test_array_type_cumulative() {
        // A prefix scan preserves the complete input type, including its memory placement and the sharding of the
        // unscanned axes, and a dynamic unscanned axis passes through untouched.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input = ArrayType::new_static(DataType::F64, [2, 3])
            .with_memory(Memory::Host { pinned: true })
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap(),
            )
            .unwrap();
        assert_eq!(input.cumulative(1, CumulativeKind::Sum), Ok(input.clone()));
        let dynamic = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("batch", DimensionBounds::unbounded())),
                Dimension::Static(3),
            ]),
        );
        assert_eq!(dynamic.cumulative(1, CumulativeKind::Max), Ok(dynamic.clone()));
    }

    #[test]
    fn test_array_type_cumulative_rejects_invalid_geometry() {
        let input = ArrayType::new_static(DataType::F64, [2, 3]);
        assert_eq!(
            input.cumulative(2, CumulativeKind::Sum),
            Err(TypeError::invalid("`cumulative` axis 2 is out of bounds for rank 2")),
        );

        // The scan is defined by the exact number of accumulated elements, so both bounded and unbounded dynamic
        // scanned dimensions are rejected.
        let bounded = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let unbounded = DimensionVariable::new("batch", DimensionBounds::unbounded());
        for variable in [bounded, unbounded] {
            let input =
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable), Dimension::Static(3)]));
            assert_eq!(
                input.cumulative(0, CumulativeKind::Sum),
                Err(TypeError::invalid(format!(
                    "`cumulative` requires a static scanned dimension but axis 0 of `{input}` is dynamic",
                ))),
            );
        }

        // Scanning a sharded axis would need cross-shard communication that this operation does not carry.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded = ArrayType::new_static(DataType::F64, [2, 3])
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap(),
            )
            .unwrap();
        assert_eq!(
            sharded.cumulative(0, CumulativeKind::Sum),
            Err(TypeError::invalid(format!(
                "`cumulative` requires an unsharded scanned dimension but axis 0 of `{sharded}` is sharded",
            ))),
        );
    }

    #[test]
    fn test_array_type_cumulative_data_types() {
        // Arithmetic kinds accept real and complex numeric inputs and the structural zero.
        for kind in [CumulativeKind::Sum, CumulativeKind::Product] {
            for data_type in [DataType::F64, DataType::I32, DataType::C64, DataType::Zero] {
                let input = ArrayType::new_static(data_type, [3, 2]);
                assert_eq!(input.cumulative(1, kind), Ok(input.clone()));
            }
            assert_eq!(
                ArrayType::new_static(DataType::Boolean, [3, 2]).cumulative(1, kind),
                Err(TypeError::invalid(format!(
                    "`cumulative` with kind `{kind}` requires numeric inputs but got `bool`",
                ))),
            );
        }

        // Extrema accept real and complex numeric inputs, which they order like the elementwise extrema.
        for kind in [CumulativeKind::Max, CumulativeKind::Min] {
            for data_type in [DataType::F64, DataType::I32, DataType::C64] {
                let input = ArrayType::new_static(data_type, [3, 2]);
                assert_eq!(input.cumulative(1, kind), Ok(input.clone()));
            }
            for data_type in [DataType::Boolean, DataType::Token, DataType::Zero] {
                assert_eq!(
                    ArrayType::new_static(data_type, [3, 2]).cumulative(1, kind),
                    Err(TypeError::invalid(format!(
                        "`cumulative` with kind `{kind}` requires numeric inputs but got `{data_type}`",
                    ))),
                );
            }
        }

        // The exponential and the logarithm have no meaning for integer, Boolean, or payload-free inputs.
        for data_type in [DataType::I32, DataType::Boolean, DataType::Token, DataType::Zero] {
            assert_eq!(
                ArrayType::new_static(data_type, [3, 2]).cumulative(1, CumulativeKind::LogSumExp),
                Err(TypeError::invalid(format!(
                    "`cumulative` with kind `log_sum_exp` requires floating-point or complex inputs but got \
                     `{data_type}`",
                ))),
            );
        }

        // `f8e8m0fnu` encodes bare positive exponents, so its smallest element exponentiates to one instead of acting
        // as the combining operator's identity, and the lowest `f6e2m3fn` value already changes when combined with one
        // more copy of itself. The finite sentinels of the finite-only formats accepted below, however, are identities
        // after every rounded pairwise combination, and the complex types pair negative infinity with a zero imaginary
        // component.
        for data_type in [DataType::F8E8M0FNU, DataType::F6E2M3FN] {
            assert_eq!(
                ArrayType::new_static(data_type, [3, 2]).cumulative(1, CumulativeKind::LogSumExp),
                Err(TypeError::invalid(format!(
                    "`cumulative` with kind `log_sum_exp` requires a floating-point format whose lowest value is a \
                     `log_add_exp` identity but got `{data_type}`",
                ))),
            );
        }
        for data_type in [
            DataType::F32,
            DataType::F4E2M1FN,
            DataType::F8E4M3B11FNUZ,
            DataType::F6E3M2FN,
            DataType::C64,
            DataType::C128,
        ] {
            let input = ArrayType::new_static(data_type, [3, 2]);
            assert_eq!(input.cumulative(1, CumulativeKind::LogSumExp), Ok(input.clone()));
        }

        // The element data type is validated before the scan geometry.
        assert_eq!(
            ArrayType::new_static(DataType::Boolean, [3]).cumulative(1, CumulativeKind::Sum),
            Err(TypeError::invalid("`cumulative` with kind `sum` requires numeric inputs but got `bool`")),
        );
    }

    #[test]
    fn test_array_type_cumulative_unreduced_inputs() {
        // A prefix sum commutes with the pending cross-device sum of an unreduced input, and so it keeps its unreduced
        // axes. The other kinds do not commute with that sum, and so they reject unreduced inputs.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input = ArrayType::new_static(DataType::F32, [3])
            .with_sharding(Sharding::replicated(mesh, 1).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        assert_eq!(input.cumulative(0, CumulativeKind::Sum), Ok(input.clone()));
        for kind in [CumulativeKind::Product, CumulativeKind::LogSumExp, CumulativeKind::Max, CumulativeKind::Min] {
            assert_eq!(
                input.cumulative(0, kind),
                Err(TypeError::invalid(format!(
                    "`cumulative` with kind `{kind}` cannot scan inputs with unreduced axes",
                ))),
            );
        }
    }

    #[test]
    fn test_array_cumulative_elements() {
        let values = (1..=6).map(|value| value as f64).collect::<Vec<_>>();
        let input = Array::matrix(2, 3, values.clone()).unwrap();
        let add = |left: f64, right: f64| Ok(left + right);

        // Scan independent rows and the outer axis in both directions.
        assert_eq!(input.cumulative_elements(1, false, add), Ok(vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]));
        assert_eq!(input.cumulative_elements(1, true, add), Ok(vec![6.0, 5.0, 3.0, 15.0, 11.0, 6.0]));
        assert_eq!(input.cumulative_elements(0, false, add), Ok(vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]));
        assert_eq!(input.cumulative_elements(0, true, add), Ok(vec![5.0, 7.0, 9.0, 4.0, 5.0, 6.0]));

        // A scanned axis with fewer than two elements leaves the payload unchanged.
        let singleton = Array::matrix(6, 1, values.clone()).unwrap();
        assert_eq!(singleton.cumulative_elements(1, false, add), Ok(values));

        // An empty scanned axis or an empty trailing axis must not divide by a zero row stride.
        for (shape, reverse) in [(vec![0, 3], false), (vec![2, 0], false), (vec![3, 0, 2], true)] {
            let empty = Array::new(ArrayType::new_static(DataType::F64, shape), Vec::new()).unwrap();
            assert_eq!(empty.cumulative_elements(0, reverse, add), Ok(Vec::<f64>::new()));
        }
    }
}
