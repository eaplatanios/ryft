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
//! keeps every axis it touches, so it _consumes_ no bounded ragged axis. The operand's ragged axes ride through onto
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

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayElement, ArrayType, DataType, Dimension, FloatingPointArrayElement,
    NumericArrayElement, RaggedArrayExtentBatchingPolicy, RaggedMaskIdentity, ShardingDimension, StaticShape,
};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableType, DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    DifferentiationPolicy,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, dispatch_on_array_element_type, impl_differentiable_operation};
use crate::operations::arithmetic::{Add, AddOperation, Mul, MulOperation};
use crate::operations::collectives::parallel_vary::ParallelVaryOperation;
use crate::operations::constants::zero::{Zero, ZeroOperation};
use crate::operations::exponential::{LOG_ADD_EXP_OPERATION_NAME, LogAddExp, LogAddExpOperation};
use crate::operations::extrema::{Max, MaxOperation, Min, MinOperation};
use crate::operations::logical::{Or, OrOperation};
use crate::operations::manipulation::broadcasting::BroadcastOperation;
use crate::operations::manipulation::concatenation::{Concatenate, ConcatenateOperation};
use crate::operations::manipulation::padding::{Pad, PadOperation};
use crate::operations::manipulation::slicing::{Slice, SliceOperation};
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
    /// wrap in their element data type.
    Product,

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

    /// Running numerically stable `log(sum(exp(x)))`. Each prefix is accumulated by folding the pairwise [`LogAddExp`]
    /// operation, which is stable over the whole real range but is a different expression from the max-shifted
    /// reduction that [`ReductionKind::LogSumExp`](crate::ReductionKind::LogSumExp) evaluates, so the two can round
    /// differently in their last bits.
    ///
    /// Padding is filled with the element data type's lowest value, which must be an identity of the rounded pairwise
    /// [`LogAddExp`] operation. True negative infinity satisfies this contract, as do finite sentinels that round back
    /// to the other input after each pairwise combination. A binary fold rounds after every pair, so once combining two
    /// sentinel values returns the sentinel, an all-sentinel subtree of any size does too, and such sentinels therefore
    /// remain neutral for any prefix length. [`DataType::F8E8M0FNU`] and [`DataType::F6E2M3FN`] have no suitable
    /// identity and are rejected, as are all non-floating-point and complex inputs. This criterion intentionally
    /// differs from that of the log-sum-exp reduction, whose padding must remain neutral after subtracting an
    /// arbitrary maximum and which therefore requires a format that represents negative infinity.
    LogSumExp,
}

impl CumulativeKind {
    /// Returns the name of this [`CumulativeKind`], which program renderings and diagnostics use as the `kind`
    /// attribute of a [`CumulativeOperation`] (e.g., `cumulative [kind=sum, axis=0]`).
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Product => "product",
            Self::Max => "max",
            Self::Min => "min",
            Self::LogSumExp => "log_sum_exp",
        }
    }
}

impl Display for CumulativeKind {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.name())
    }
}

// TODO(eaplatanios): Review from here onwards.

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
    /// is not part of the operation payload: it is recoverable from the staged input type wherever a rule needs it.
    #[inline]
    pub fn new(axis: usize, kind: CumulativeKind) -> Self {
        Self { axis, kind, reverse: false }
    }

    /// Returns this operation with its scan direction set to `reverse`, accumulating from the end of the scanned axis
    /// toward its start.
    #[inline]
    pub fn with_reverse(mut self, reverse: bool) -> Self {
        self.reverse = reverse;
        self
    }

    /// Returns the scanned axis, in the operand's own coordinate system.
    #[inline]
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// Returns the combining operator of this [`CumulativeOperation`].
    #[inline]
    pub fn kind(&self) -> CumulativeKind {
        self.kind
    }

    /// Returns whether the scan accumulates from the end of the scanned axis toward its start.
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

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![input_types[0].cumulative(self.axis, self.kind)?])
    }

    // The scan direction renders only when it is set, keeping the common forward scan compact.
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("kind", self.kind)?;
            operation.field("axis", self.axis)?;
            if self.reverse {
                operation.field("reverse", self.reverse)?;
            }
            Ok(())
        })
    }
}

impl<C: Domain<Type = ArrayType, Value: Cumulative>> InterpretableOperation<C> for CumulativeOperation {
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

// Partial evaluation defers to the default fold-or-residualize behavior of `Program::partially_evaluate`.
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for CumulativeOperation where
    C::Operation: From<CumulativeOperation>
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
        // dimension, so an axis at or after the batch axis shifts past it. A prefix scan preserves its operand's rank,
        // and so the output batch axis is the input batch axis itself. A replicated operand carries no inserted batch
        // dimension, so its scanned axis needs no shift. Ragged axes are packed positions in both cases, and so the
        // masking and rewrapping below use the lifted axis.
        check_count!("input", inputs, 1, ProgramError);
        let (lifted_axis, output_batch_axis) = match inputs[0].batch_axis_position() {
            Some(batch_axis) => {
                (if self.axis < batch_axis { self.axis } else { self.axis + 1 }, BatchAxis::from_position(batch_axis))
            }
            None => (self.axis, BatchAxis::replicated()),
        };

        // Padding along a *scanned* ragged axis is neutralized with the identity of the combining operator first, so
        // that padded positions cannot contribute to any live prefix.
        let input = match inputs[0].ragged_axes().iter().any(|ragged_axis| ragged_axis.axis() == lifted_axis) {
            true => {
                let identity = match self.kind {
                    CumulativeKind::Sum => RaggedMaskIdentity::Zero,
                    CumulativeKind::Product => RaggedMaskIdentity::One,
                    CumulativeKind::Max | CumulativeKind::LogSumExp => RaggedMaskIdentity::Lowest,
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

        // Interpretation carries values rather than batch metadata, so the operand's ragged axes are restored here. A
        // scan consumes none of them, and so the rule reports no consumption evidence.
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
        C::Operation: From<CumulativeOperation>
            + From<AddOperation<ArrayType>>
            + From<ConcatenateOperation<ArrayType>>
            + From<LogAddExpOperation<ArrayType>>
            + From<MaxOperation<ArrayType>>
            + From<MinOperation<ArrayType>>
            + From<MulOperation<ArrayType>>
            + From<OrOperation<ArrayType>>
            + From<PadOperation<ArrayType>>
            + From<SliceOperation>
            + OperationProvider<ArrayType, ZeroOperation<ArrayType>, Operation = C::Operation>
            + OperationProvider<ArrayType, ParallelVaryOperation, Operation = C::Operation>
            + OperationProvider<ArrayType, BroadcastOperation, Operation = C::Operation>,
        C::Value: Cumulative,
    {
        |operation, context, driver, inputs| {
            // A cumulative sum is linear in its operand, so its tangent is the same prefix sum of the input tangent.
            // Every other kind is nonlinear, so its rule differentiates through the associative-scan decomposition with
            // the kind's own combining operator, and every primitive that construction stages contributes its own
            // forward-mode rule. The composite array universe reaches this rule through the default projected
            // fall-through of `MemberDifferentiableOperation`, because the operation is shape-preserving and its
            // operand never needs the replication that a broadcasting elementwise member does.
            check_count!("input", inputs, 1, ProgramError);
            let (axis, kind, reverse) = (operation.axis, operation.kind, operation.reverse);
            let primal_input = inputs[0].primal();
            let MaybeZero::Value(tangent_input) = inputs[0].tangent() else {
                // Every JVP is linear in its tangent, so a structural zero tangent stays a structural zero one, and the
                // primal is cheaper to obtain from the primitive itself than from the decomposition.
                let primal = primal_input.cumulative(axis, kind, reverse)?;
                let tangent = MaybeZero::Zero(primal.r#type().tangent()?);
                return Ok(vec![DifferentiationDual::new(primal, tangent)?]);
            };
            // The primal output of a nonlinear kind comes from the decomposition, which interleaves its halves by
            // adding zero-padded operands. That addition turns a `-0.0` result into `+0.0`, so under differentiation
            // the primal output of an extremum scan over signed zeros can differ from the undifferentiated one in the
            // sign of a zero (JAX's `_cumulative_jvp_rule` has the same property).
            let dual = match kind {
                CumulativeKind::Sum => {
                    let primal = primal_input.cumulative(axis, kind, reverse)?;
                    let tangent = tangent_input.cumulative(axis, kind, reverse)?;
                    DifferentiationDual::new(primal, MaybeZero::Value(tangent))?
                }
                CumulativeKind::Product => jvp_through_associative_scan(
                    context,
                    driver,
                    primal_input,
                    tangent_input,
                    axis,
                    reverse,
                    |left, right| left.mul(right),
                )?,
                CumulativeKind::Max => jvp_through_associative_scan(
                    context,
                    driver,
                    primal_input,
                    tangent_input,
                    axis,
                    reverse,
                    |left, right| left.max(right),
                )?,
                CumulativeKind::Min => jvp_through_associative_scan(
                    context,
                    driver,
                    primal_input,
                    tangent_input,
                    axis,
                    reverse,
                    |left, right| left.min(right),
                )?,
                CumulativeKind::LogSumExp => jvp_through_associative_scan(
                    context,
                    driver,
                    primal_input,
                    tangent_input,
                    axis,
                    reverse,
                    |left, right| left.log_add_exp(right),
                )?,
            };
            Ok(vec![dual])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: From<CumulativeOperation>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // A forward prefix sum sends input element `i` into every output element `j >= i`, so the cotangent of
            // input `i` is the sum of the output cotangents `j >= i` (i.e., a reverse prefix sum). The adjoint of a
            // reverse prefix sum is symmetrically a forward one, which is why a cumulative sum is closed under
            // transposition and needs no companion primitive. Every other kind is nonlinear and is instead
            // differentiated through the linear operations staged by its JVP, so it is rejected here regardless of its
            // cotangent.
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

/// Value-level cumulative capability that accumulates the elements of an array along one axis using a
/// [`CumulativeKind`].
///
/// [`Cumulative`] fills the same role for [`CumulativeOperation`] that
/// [`Reduce`](crate::operations::reductions::Reduce) fills for
/// [`ReduceOperation`](crate::operations::reductions::ReduceOperation). Concrete [`Array`]s scan immediately, while
/// context-carrying values bind a [`CumulativeOperation`] through their own context. The output has the input's type,
/// and element `i` along the scanned axis holds the combination of the input elements `0..=i`, or of the input elements
/// `i..` for the reverse direction. Refer to [`CumulativeKind`] for the identity, supported data types, and exceptional
/// values of each kind.
///
/// Besides the general [`Self::cumulative`] function, this trait provides one forward and one reverse shortcut function
/// per kind, which all share the axis, direction, and error contract of [`Self::cumulative`].
pub trait Cumulative: Sized {
    /// Accumulates `self` along `axis` using the combining operator selected by `kind`.
    ///
    /// # Parameters
    ///
    ///   - `axis`: Axis of `self` to scan. Its dimension must be static and unsharded.
    ///   - `kind`: [`CumulativeKind`] that determines how the elements along `axis` are combined.
    ///   - `reverse`: Whether to accumulate from the end of `axis` toward its start (i.e., to compute inclusive
    ///     suffixes rather than inclusive prefixes).
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if `kind` does not support the data type of `self`, if `axis` is out of bounds, if
    /// the scanned dimension is dynamic or sharded, or if the context of `self` fails to bind the scan.
    fn cumulative(&self, axis: usize, kind: CumulativeKind, reverse: bool) -> Result<Self, ProgramError>;

    /// Returns the inclusive prefix sum of `self` along `axis` using [`CumulativeKind::Sum`]. Refer to
    /// [`Self::cumulative`] for the semantics of `axis` and for the errors that this function may return.
    #[inline]
    fn cumulative_sum(&self, axis: usize) -> Result<Self, ProgramError> {
        self.cumulative(axis, CumulativeKind::Sum, false)
    }

    /// Returns the inclusive suffix sum of `self` along `axis` using [`CumulativeKind::Sum`]. Refer to
    /// [`Self::cumulative`] for the semantics of `axis` and for the errors that this function may return.
    #[inline]
    fn reverse_cumulative_sum(&self, axis: usize) -> Result<Self, ProgramError> {
        self.cumulative(axis, CumulativeKind::Sum, true)
    }

    /// Returns the inclusive prefix product of `self` along `axis` using [`CumulativeKind::Product`]. Refer to
    /// [`Self::cumulative`] for the semantics of `axis` and for the errors that this function may return.
    #[inline]
    fn cumulative_product(&self, axis: usize) -> Result<Self, ProgramError> {
        self.cumulative(axis, CumulativeKind::Product, false)
    }

    /// Returns the inclusive suffix product of `self` along `axis` using [`CumulativeKind::Product`]. Refer to
    /// [`Self::cumulative`] for the semantics of `axis` and for the errors that this function may return.
    #[inline]
    fn reverse_cumulative_product(&self, axis: usize) -> Result<Self, ProgramError> {
        self.cumulative(axis, CumulativeKind::Product, true)
    }

    /// Returns the running maximum of `self` along `axis` using [`CumulativeKind::Max`]. Refer to [`Self::cumulative`]
    /// for the semantics of `axis` and for the errors that this function may return.
    #[inline]
    fn cumulative_max(&self, axis: usize) -> Result<Self, ProgramError> {
        self.cumulative(axis, CumulativeKind::Max, false)
    }

    /// Returns the reverse running maximum of `self` along `axis` using [`CumulativeKind::Max`]. Refer to
    /// [`Self::cumulative`] for the semantics of `axis` and for the errors that this function may return.
    #[inline]
    fn reverse_cumulative_max(&self, axis: usize) -> Result<Self, ProgramError> {
        self.cumulative(axis, CumulativeKind::Max, true)
    }

    /// Returns the running minimum of `self` along `axis` using [`CumulativeKind::Min`]. Refer to [`Self::cumulative`]
    /// for the semantics of `axis` and for the errors that this function may return.
    #[inline]
    fn cumulative_min(&self, axis: usize) -> Result<Self, ProgramError> {
        self.cumulative(axis, CumulativeKind::Min, false)
    }

    /// Returns the reverse running minimum of `self` along `axis` using [`CumulativeKind::Min`]. Refer to
    /// [`Self::cumulative`] for the semantics of `axis` and for the errors that this function may return.
    #[inline]
    fn reverse_cumulative_min(&self, axis: usize) -> Result<Self, ProgramError> {
        self.cumulative(axis, CumulativeKind::Min, true)
    }

    /// Returns the running numerically stable `log(sum(exp(self)))` along `axis` using [`CumulativeKind::LogSumExp`],
    /// whose documentation describes its data-type limits. Refer to [`Self::cumulative`] for the semantics of `axis`
    /// and for the errors that this function may return.
    #[inline]
    fn cumulative_log_sum_exp(&self, axis: usize) -> Result<Self, ProgramError> {
        self.cumulative(axis, CumulativeKind::LogSumExp, false)
    }

    /// Returns the reverse running numerically stable `log(sum(exp(self)))` along `axis` using
    /// [`CumulativeKind::LogSumExp`], whose documentation describes its data-type limits. Refer to [`Self::cumulative`]
    /// for the semantics of `axis` and for the errors that this function may return.
    #[inline]
    fn reverse_cumulative_log_sum_exp(&self, axis: usize) -> Result<Self, ProgramError> {
        self.cumulative(axis, CumulativeKind::LogSumExp, true)
    }
}

impl Cumulative for Array {
    fn cumulative(&self, axis: usize, kind: CumulativeKind, reverse: bool) -> Result<Self, ProgramError> {
        // The type rule validates the scan and supplies the complete output metadata. The kernels below then decode the
        // operand's logical elements, run the sequential prefix scan over them with the kind's element-level combining
        // operator, and re-encode the result into the operand's own type. Accumulation happens in the operand's element
        // data type, so a low-precision payload rounds every partial result exactly as a staged program does.
        let output_type = self.r#type().cumulative(axis, kind)?;
        let data_type = output_type.data_type();

        // The structural-zero element type has no payload bytes, and every prefix sum or product of a zero is a zero.
        if data_type == DataType::Zero {
            return Self::new(output_type, Vec::new());
        }

        let shape = output_type.static_shape().unwrap();
        match kind {
            CumulativeKind::Sum | CumulativeKind::Product => {
                dispatch_on_array_element_type!(@numeric data_type, |Element| {
                    let elements = self.elements::<Element>()?;
                    let scanned = cumulative_evaluate(elements.as_slice(), &shape, axis, reverse, |left, right| {
                        match kind {
                            CumulativeKind::Sum => NumericArrayElement::add(left, right),
                            _ => NumericArrayElement::mul(left, right),
                        }
                    })?;
                    Self::from_elements(output_type, scanned.as_slice())
                })
            }
            CumulativeKind::Max | CumulativeKind::Min => {
                dispatch_on_array_element_type!(@numeric data_type, |Element| {
                    let elements = self.elements::<Element>()?;
                    let scanned = cumulative_evaluate(elements.as_slice(), &shape, axis, reverse, |left, right| {
                        Ok(match kind {
                            CumulativeKind::Max => ArrayElement::max(&left, &right),
                            _ => ArrayElement::min(&left, &right),
                        })
                    })?;
                    Self::from_elements(output_type, scanned.as_slice())
                })
            }
            CumulativeKind::LogSumExp => {
                dispatch_on_array_element_type!(@float data_type, |Element| {
                    let elements = self.elements::<Element>()?;
                    let scanned = cumulative_evaluate(
                        elements.as_slice(),
                        &shape,
                        axis,
                        reverse,
                        FloatingPointArrayElement::log_add_exp,
                    )?;
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
    fn cumulative(&self, axis: usize, kind: CumulativeKind, reverse: bool) -> Result<Self, ProgramError> {
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
    /// Returns the output [`ArrayType`] produced by scanning `self` along `axis` with `kind`. The result *is* `self`: a
    /// prefix scan changes neither the element data type nor the shape, layout, memory placement, or sharding.
    ///
    /// Validates that:
    ///   - `kind` supports the element data type of `self`, as documented on [`CumulativeKind`];
    ///   - `axis` is within `0..self.rank()`;
    ///   - the scanned dimension is [`Dimension::Static`], because a prefix scan is defined by the exact number of
    ///     elements that it accumulates over;
    ///   - the scanned dimension is unsharded, because a prefix crosses shard boundaries and a cumulative operation
    ///     carries no cross-shard communication of its own; and
    ///   - `self` has no unreduced mesh axes unless `kind` is [`CumulativeKind::Sum`], because only a prefix sum
    ///     commutes with the pending cross-device sum of an unreduced value.
    ///
    /// A dynamically sized scanned axis could be supported in the future by physicalizing the scan at the dimension's
    /// declared upper bound and masking the elements past each runtime extent with the kind's identity, which is the
    /// same discipline that the ragged batching rule already uses. That extension is deliberately not implemented here:
    /// it would silently change the operation's cost model, and so it belongs to an explicit dynamic-scan surface.
    fn cumulative(&self, axis: usize, kind: CumulativeKind) -> Result<Self, TypeError> {
        // The element data type is validated before the scan geometry, and the diagnostic names the kind, because the
        // supported data types differ across kinds (e.g., summation accepts the structural zero while extrema do not).
        let data_type = self.data_type();
        let requirement: Option<Cow<'static, str>> = match kind {
            CumulativeKind::Sum | CumulativeKind::Product if !data_type.is_numeric() && data_type != DataType::Zero => {
                Some(Cow::Borrowed("numeric inputs"))
            }
            CumulativeKind::Max | CumulativeKind::Min if !data_type.is_numeric() => {
                Some(Cow::Borrowed("numeric inputs"))
            }
            CumulativeKind::LogSumExp if !data_type.is_floating_point() => {
                Some(Cow::Borrowed("real floating-point inputs"))
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

/// Returns the inclusive prefix scan of `value` along `axis` under the associative operator `combine`, built out of
/// ordinary manipulation primitives instead of out of one [`CumulativeOperation`].
///
/// This is Ryft's port of the log-depth Blelloch construction that JAX's
/// [`lax.associative_scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html) implements
/// (`jax/_src/lax/control_flow/loops.py`), and it exists here for the reason it is reached for there: a cumulative
/// operation whose combining operator is nonlinear has no closed-form primitive derivative, so the nonlinear
/// [`CumulativeKind`]s define their forward mode by differentiating *through* this decomposition rather than by
/// carrying a bespoke gradient formula (JAX's `_cumulative_jvp_rule`). It is also useful on its own for combining
/// operators that no [`CumulativeKind`] covers.
///
/// The recursion combines adjacent pairs along `axis`, scans the halved sequence recursively, combines the scanned
/// halves back against the elements the pairing skipped, and interleaves the two halves into the result. `combine`
/// always receives its operands in scan order (the accumulated prefix first), so the construction stays correct for
/// associative operators that are not commutative. A `reverse` scan mirrors the same recursion around the end of the
/// axis (the pairing simply starts one element in when the extent is odd) instead of reversing the operand before and
/// after a forward scan, which saves two array reversals per scan. Boolean operands are interleaved with a disjunction
/// rather than an addition, because Booleans have no addition.
///
/// The whole operand shape must be static, because the construction slices at staging-time positions. A scanned axis
/// shorter than two elements is returned unchanged.
///
/// # Parameters
///
///   - `value`: Scanned operand.
///   - `axis`: Scanned axis.
///   - `reverse`: Whether to accumulate from the end of the scanned axis toward its start.
///   - `combine`: Associative binary operator, receiving the accumulated prefix and the next elements in scan order.
///
/// # Errors
///
/// Returns a [`ProgramError`] if `axis` is out of bounds, if the operand shape is not static, or if staging any of the
/// primitives of the construction (including those that `combine` stages) fails.
pub fn associative_scan<V, F>(value: &V, axis: usize, reverse: bool, combine: &F) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + Add + Concatenate + Or + Pad + Slice,
    V::DispatchDomain: Context<Type = ArrayType> + Zero<V>,
    F: Fn(&V, &V) -> Result<V, ProgramError>,
{
    let value_type = value.r#type().into_owned();
    let rank = value_type.rank();
    if axis >= rank {
        return Err(
            TypeError::invalid(format!("`associative_scan` axis {axis} is out of bounds for rank {rank}")).into()
        );
    }
    let shape = value_type.static_shape().ok_or_else(|| {
        TypeError::invalid(format!("`associative_scan` requires a statically shaped operand but got `{value_type}`"))
    })?;

    // The scopes below are purely diagnostic: they attribute every instruction the decomposition stages, and they are a
    // no-op under an eager context, which records no instructions at all.
    let domain = value.dispatch_domain();
    domain.invoke_with_provenance_scope(ProvenanceScope::new("ryft"), || {
        domain.invoke_with_provenance_scope(ProvenanceScope::new("differentiation"), || {
            domain.invoke_with_provenance_scope(ProvenanceScope::new("associative_scan"), || {
                associative_scan_recursively(value, &shape, axis, reverse, combine)
            })
        })
    })
}

/// Recursive half of [`associative_scan`], operating on an operand whose shape is already known to be static.
fn associative_scan_recursively<V, F>(
    value: &V,
    shape: &StaticShape,
    axis: usize,
    reverse: bool,
    combine: &F,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + Add + Concatenate + Or + Pad + Slice,
    V::DispatchDomain: Context<Type = ArrayType> + Zero<V>,
    F: Fn(&V, &V) -> Result<V, ProgramError>,
{
    let extent = shape[axis];
    if extent < 2 {
        return Ok(value.clone());
    }
    let half = extent / 2;

    // Pair adjacent elements. A forward scan pairs from the start of the axis and a reverse scan pairs from its end,
    // which is the one place the two directions differ: an odd extent leaves the first element unpaired going forward
    // and the *last* one unpaired going backward, so the pairing starts one element in.
    let pair_offset = match reverse {
        true => extent % 2,
        false => 0,
    };
    let earlier = scan_slice(value, shape, axis, pair_offset, extent - 1, 2)?;
    let later = scan_slice(value, shape, axis, pair_offset + 1, extent, 2)?;
    let reduced = match reverse {
        true => combine(&later, &earlier)?,
        false => combine(&earlier, &later)?,
    };

    // Scanning the pairwise reductions yields every other output element: the odd positions of a forward scan, and the
    // positions congruent to `pair_offset` of a reverse one.
    let halved_shape = StaticShape::new({
        let mut dimensions = shape.dimensions().to_vec();
        dimensions[axis] = half;
        dimensions
    });
    let aligned = associative_scan_recursively(&reduced, &halved_shape, axis, reverse, combine)?;

    // Each complementary position extends the aligned result before it by the one element that separates them, except
    // for the position at the scan's own start, which is just the operand element there. An even extent has one fewer
    // complementary combination than there are aligned results, so the aligned side is trimmed; an extent of exactly
    // two has none at all, and its complementary half is that lone start element.
    let complement_count = match extent % 2 {
        0 => half - 1,
        _ => half,
    };
    let (complement, aligned_leads) = match reverse {
        true => {
            let last = scan_slice(value, shape, axis, extent - 1, extent, 1)?;
            let complement = match complement_count {
                0 => last,
                _ => {
                    let trimmed = match extent % 2 {
                        0 => scan_slice(&aligned, &halved_shape, axis, 1, half, 1)?,
                        _ => aligned.clone(),
                    };
                    let start = (pair_offset + 1) % 2;
                    let operands = scan_slice(value, shape, axis, start, start + 2 * complement_count, 2)?;
                    V::concatenate([&combine(&trimmed, &operands)?, &last], axis)?
                }
            };
            (complement, extent % 2 == 0)
        }
        false => {
            let first = scan_slice(value, shape, axis, 0, 1, 1)?;
            let complement = match complement_count {
                0 => first,
                _ => {
                    let trimmed = match extent % 2 {
                        0 => scan_slice(&aligned, &halved_shape, axis, 0, half - 1, 1)?,
                        _ => aligned.clone(),
                    };
                    let operands = scan_slice(value, shape, axis, 2, (2 + 2 * complement_count).min(extent), 2)?;
                    V::concatenate([&first, &combine(&trimmed, &operands)?], axis)?
                }
            };
            (complement, false)
        }
    };

    match aligned_leads {
        true => scan_interleave(&aligned, &complement, shape, axis, half, extent - half),
        false => scan_interleave(&complement, &aligned, shape, axis, extent - half, half),
    }
}

/// Returns the elements of `value` at positions `start`, `start + stride`, ... below `limit` along `axis`, keeping
/// every other axis whole.
fn scan_slice<V: Slice>(
    value: &V,
    shape: &StaticShape,
    axis: usize,
    start: usize,
    limit: usize,
    stride: usize,
) -> Result<V, ProgramError> {
    let mut start_indices = vec![0; shape.rank()];
    let mut limit_indices = shape.dimensions().to_vec();
    let mut strides = vec![1; shape.rank()];
    start_indices[axis] = start;
    limit_indices[axis] = limit;
    strides[axis] = stride;
    value.slice(start_indices.as_slice(), limit_indices.as_slice(), strides.as_slice())
}

/// Returns `left` and `right` interleaved along `axis`, starting with `left`. `left` must hold either as many elements
/// along `axis` as `right` or exactly one more.
///
/// Both operands are dilated into the output extent with interior padding (writing zeros into the positions that the
/// other operand occupies) and then combined with an addition, or with a disjunction for Boolean operands, which have
/// no addition. The combination is exact because the two dilated operands have disjoint support and zero (i.e.,
/// `false`) is the identity of both combiners.
fn scan_interleave<V>(
    left: &V,
    right: &V,
    shape: &StaticShape,
    axis: usize,
    left_count: usize,
    right_count: usize,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + Add + Or + Pad,
    V::DispatchDomain: Zero<V>,
{
    if left_count != right_count && left_count != right_count + 1 {
        return Err(TypeError::invalid(format!(
            "`associative_scan` cannot interleave {left_count} elements with {right_count} elements"
        ))
        .into());
    }
    let padding_value = left.dispatch_domain().zero(&left.r#type().scalar_like()?)?;
    let mut edge_padding_low = vec![0; shape.rank()];
    let mut edge_padding_high = vec![0; shape.rank()];
    let mut interior_padding = vec![0; shape.rank()];
    interior_padding[axis] = 1;
    edge_padding_high[axis] = i64::from(left_count == right_count);
    let dilated_left = left.pad(&padding_value, &edge_padding_low, &edge_padding_high, &interior_padding)?;
    edge_padding_low[axis] = 1;
    edge_padding_high[axis] = i64::from(left_count != right_count);
    let dilated_right = right.pad(&padding_value, &edge_padding_low, &edge_padding_high, &interior_padding)?;
    match left.r#type().data_type() {
        DataType::Boolean => dilated_left.or(&dilated_right),
        _ => dilated_left.add(&dilated_right),
    }
}

/// Value that the nested trace staging an [`associative_scan`] decomposition flows.
type DecompositionTracer<C> = Tracer<TracingContext<<C as Domain>::Constant, <C as Domain>::Operation>>;

/// Applies the forward-mode rule of a nonlinear [`CumulativeKind`] by differentiating through the associative-scan
/// decomposition of [`associative_scan`], and returns the resulting primal/tangent pair.
///
/// The decomposition is traced once into its own program over the caller's operation family, that program is
/// differentiated through the instruction-scoped `driver` (which re-enters the active differentiation machinery, so
/// every primitive the construction stages contributes its *own* forward-mode rule), and the resulting fused program is
/// replayed in `context` over the operand's primal and tangent. The primal output comes back from the decomposition
/// too, rather than from the cumulative primitive, because the two are the same value (up to the sign of zero results)
/// and the fused program computes it on the way to the tangent. When the primal and tangent contexts differ, the
/// decomposition is linearized instead, its primal half is replayed in the primal context, and its tangent half is
/// replayed in the tangent context over the transferred residuals.
///
/// The caller is responsible for the structural-zero tangent shortcut; this function requires a live tangent because
/// the decomposition is pure overhead when there is nothing to propagate.
///
/// # Parameters
///
///   - `context`: [`DifferentiationContext`] that the forward-mode program is replayed in.
///   - `driver`: Instruction-scoped [`DifferentiationDriver`] serving the nested differentiation request.
///   - `primal`: Operand primal.
///   - `tangent`: Operand tangent.
///   - `axis`: Scanned axis.
///   - `reverse`: Whether the scan accumulates from the end of the scanned axis toward its start.
///   - `combine`: Associative combining operator of the kind, staged over the nested trace's values.
fn jvp_through_associative_scan<C, D, F, P: DifferentiationPolicy<C>>(
    context: &DifferentiationContext<C, P>,
    driver: &D,
    primal: &C::Value,
    tangent: &C::Value,
    axis: usize,
    reverse: bool,
    combine: F,
) -> Result<DifferentiationDual<C::Value>, DifferentiationError>
where
    C: Context<Type = ArrayType>,
    D: DifferentiationDriver<C>,
    C::Operation: From<AddOperation<ArrayType>>
        + From<ConcatenateOperation<ArrayType>>
        + From<OrOperation<ArrayType>>
        + From<PadOperation<ArrayType>>
        + From<SliceOperation>
        + OperationProvider<ArrayType, ZeroOperation<ArrayType>, Operation = C::Operation>
        + OperationProvider<ArrayType, ParallelVaryOperation, Operation = C::Operation>
        + OperationProvider<ArrayType, BroadcastOperation, Operation = C::Operation>,
    F: Fn(&DecompositionTracer<C>, &DecompositionTracer<C>) -> Result<DecompositionTracer<C>, ProgramError>,
{
    let (_, decomposition) = TracingContext::<C::Constant, C::Operation>::trace::<_, ArrayType, _>(
        |value: DecompositionTracer<C>| associative_scan(&value, axis, reverse, &combine),
        primal.r#type().into_owned(),
    )?;
    if std::ptr::eq(context.primal(), context.tangent()) {
        let fused = driver.jvp_program(decomposition.entry_region_ref(), &[0])?;
        let mut outputs = fused.interpret_in_context(context.primal(), vec![primal.clone(), tangent.clone()])?;
        check_count!("output", outputs, 2, ProgramError);
        let output_tangent = outputs.remove(1);
        return DifferentiationDual::new(outputs.remove(0), MaybeZero::Value(output_tangent));
    }
    let linearization = driver.linearize_program(decomposition.entry_region_ref(), &[0])?;
    let mut primal_outputs = linearization.primal().interpret_in_context(context.primal(), vec![primal.clone()])?;
    check_count!("output", primal_outputs, 1 + linearization.residual_count(), ProgramError);
    let residuals = primal_outputs.split_off(1);
    let mut tangent_inputs = vec![tangent.clone()];
    tangent_inputs
        .extend(residuals.into_iter().map(|value| context.primal_to_tangent(value)).collect::<Result<Vec<_>, _>>()?);
    let mut tangent_outputs = linearization.tangent().interpret_in_context(context.tangent(), tangent_inputs)?;
    check_count!("output", tangent_outputs, 1, ProgramError);
    DifferentiationDual::new(primal_outputs.remove(0), MaybeZero::Value(tangent_outputs.remove(0)))
}

/// Sequential prefix scan over a flat row-major payload with the provided `shape`, which the eager [`Array`] kernel
/// runs with the element-level combining operator of each [`CumulativeKind`].
///
/// Returns the scanned payload, which has the same length and shape as `values`. Output element `i` along `axis` holds
/// the accumulation of input elements `0..=i`, or of elements `i..` when `reverse` is set. A scanned axis shorter than
/// two elements (including a zero-length one) leaves the payload unchanged. The combiner is fallible because the
/// element-level arithmetic contracts of the reference backend are (e.g., a conversion into a low-precision encoding
/// can fail).
///
/// # Parameters
///
///   - `values`: Row-major input payload.
///   - `shape`: Input shape.
///   - `axis`: Scanned axis.
///   - `reverse`: Whether to accumulate from the end of the scanned axis toward its start.
///   - `combiner`: Binary associative operator, receiving the accumulated prefix and the next element.
fn cumulative_evaluate<T: Clone>(
    values: &[T],
    shape: &StaticShape,
    axis: usize,
    reverse: bool,
    combiner: impl Fn(T, T) -> Result<T, ProgramError>,
) -> Result<Vec<T>, ProgramError> {
    let mut output = values.to_vec();
    let extent = shape[axis];
    if extent < 2 {
        return Ok(output);
    }

    // Row-major storage splits into `outer` independent blocks of `extent` slices, each holding `inner` elements, so
    // one scan step moves by `inner` elements and the scan visits every `(outer, inner)` pair once. Both bounds are
    // computed as direct dimension products rather than from the payload length and the scanned axis stride, because a
    // zero-extent axis anywhere to the right of `axis` makes that stride zero.
    let inner = shape.dimensions()[axis + 1..].iter().product::<usize>();
    let outer = shape.dimensions()[..axis].iter().product::<usize>();
    for block in 0..outer {
        let base = block * extent * inner;
        for offset in 0..inner {
            let index = |position: usize| base + position * inner + offset;
            if reverse {
                for position in (0..extent - 1).rev() {
                    output[index(position)] =
                        combiner(output[index(position + 1)].clone(), values[index(position)].clone())?;
                }
            } else {
                for position in 1..extent {
                    output[index(position)] =
                        combiner(output[index(position - 1)].clone(), values[index(position)].clone())?;
                }
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
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
        DifferentiableOperation, TransposableOperation, TranspositionContext, differentiate_at,
    };
    use crate::macros::{
        check_gradient, check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::operations::reductions::Reduce;
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder, ProgramRenderingMode, ValueProjection};

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

    #[test]
    fn test_cumulative_kind_name() {
        for (kind, name) in [
            (CumulativeKind::Sum, "sum"),
            (CumulativeKind::Product, "product"),
            (CumulativeKind::Max, "max"),
            (CumulativeKind::Min, "min"),
            (CumulativeKind::LogSumExp, "log_sum_exp"),
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

        // Equality distinguishes kinds and directions.
        assert_eq!(operation.clone().with_reverse(false), operation);
        assert_ne!(operation.clone().with_reverse(true), operation);
        assert_ne!(CumulativeOperation::new(1, CumulativeKind::Max), operation);
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
        // A replicated operand carries no inserted batch dimension, so the scanned axis needs no shift.
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
    fn test_cumulative_batching_unscanned_ragged_axis() {
        // Per item, a ragged `[length]` row is scanned along its dense trailing axis, so the ragged axis is not the
        // scanned one: nothing needs masking, the axis rides through onto the result unchanged, and, because the scan
        // consumes no axis, the rule claims no consumption evidence.
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
        let extents = Array::vector(vec![1i32, 3]).unwrap();
        let input = ArrayBatch::new(
            Array::from_elements::<f32>(
                ArrayType::new_static(DataType::F32, [2, 3, 2]),
                &(1..=12).map(|value| value as f32).collect::<Vec<_>>(),
            )
            .unwrap(),
            BatchAxis::new(0),
        )
        .unwrap()
        .with_ragged_axes(vec![RaggedAxis::new(1, extents.clone(), variable.clone(), vec![0])])
        .unwrap();
        let (outputs, evidence) = CumulativeOperation::new(1, CumulativeKind::Sum)
            .batch(&BatchingContext::new(EagerContext::<Array>::new(), 2), &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].ragged_axes(), &[RaggedAxis::new(1, extents, variable, vec![0])]);
        let expected = vec![1.0, 3.0, 3.0, 7.0, 5.0, 11.0, 7.0, 15.0, 9.0, 19.0, 11.0, 23.0];
        assert_eq!(outputs[0].value().to_f64s(), expected);
        assert!(evidence.is_empty());
    }

    #[test]
    fn test_cumulative_batching_ragged_scanned_axis() {
        // Scanning a ragged axis would fold its padding into every later live prefix, so the rule asks the policy to
        // neutralize that padding with the identity of the kind's combining operator first. Static array batching
        // cannot, and says so (naming that identity) rather than silently scanning padding.
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
        let input = ArrayBatch::new(Array::matrix(2, 3, vec![1.0f32; 6]).unwrap(), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![1i32, 3]).unwrap(), variable, vec![0])])
            .unwrap();
        for (kind, identity) in [
            (CumulativeKind::Sum, "Zero"),
            (CumulativeKind::Product, "One"),
            (CumulativeKind::Max, "Lowest"),
            (CumulativeKind::Min, "Highest"),
            (CumulativeKind::LogSumExp, "Lowest"),
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
        let input = ArrayBatch::new(packed.into_projected().unwrap(), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, extents.into_projected().unwrap(), length.clone(), vec![0])])
            .unwrap();

        // The per-item scan of axis 0 is the packed axis 1 that carries the ragged extents.
        let (outputs, evidence) = CumulativeOperation::new(0, CumulativeKind::LogSumExp)
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].ragged_axes().len(), 1);
        assert_eq!(outputs[0].ragged_axes()[0].axis(), 1);
        assert_eq!(outputs[0].ragged_axes()[0].dimension(), &length);
        assert!(evidence.is_empty());

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
                    %12:f32[items, 3] = cumulative [kind=log_sum_exp, axis=1] %11
                in (%12)"
            },
        );
    }

    #[test]
    fn test_cumulative_differentiation() {
        // Sums are linear, so their tangent is the same scan of the input tangent. The other kinds differentiate
        // through the associative-scan decomposition, and their expected tangents are, respectively, the product rule
        // applied to each prefix (a zero input zeroes every later prefix but still passes its own tangent, scaled by
        // the product of the other elements), the tangent of the element that currently attains the extremum (at
        // tie-free inputs), and the softmax-weighted average of the input tangents over each prefix.
        let e = std::f64::consts::E;
        let extrema = Array::vector(vec![3.0, 1.0, 4.0, 1.5, 5.0]).unwrap();
        let ramp = Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        for (operation, primals, tangents, primal_outputs, tangent_outputs) in [
            (
                CumulativeOperation::new(0, CumulativeKind::Sum),
                Array::vector(vec![1.0, 2.0, 3.0]).unwrap(),
                Array::vector(vec![1.0, 1.0, 1.0]).unwrap(),
                Array::vector(vec![1.0, 3.0, 6.0]).unwrap(),
                Array::vector(vec![1.0, 2.0, 3.0]).unwrap(),
            ),
            (
                CumulativeOperation::new(0, CumulativeKind::Sum).with_reverse(true),
                Array::vector(vec![1.0, 2.0, 3.0]).unwrap(),
                Array::vector(vec![1.0, 1.0, 1.0]).unwrap(),
                Array::vector(vec![6.0, 5.0, 3.0]).unwrap(),
                Array::vector(vec![3.0, 2.0, 1.0]).unwrap(),
            ),
            (
                CumulativeOperation::new(0, CumulativeKind::Product),
                Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
                Array::vector(vec![1.0, 1.0, 1.0, 1.0]).unwrap(),
                Array::vector(vec![1.0, 2.0, 6.0, 24.0]).unwrap(),
                Array::vector(vec![1.0, 3.0, 11.0, 50.0]).unwrap(),
            ),
            (
                CumulativeOperation::new(0, CumulativeKind::Product),
                Array::vector(vec![2.0, 0.0, 3.0]).unwrap(),
                Array::vector(vec![1.0, 1.0, 1.0]).unwrap(),
                Array::vector(vec![2.0, 0.0, 0.0]).unwrap(),
                Array::vector(vec![1.0, 2.0, 6.0]).unwrap(),
            ),
            (
                CumulativeOperation::new(0, CumulativeKind::Product).with_reverse(true),
                Array::vector(vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
                Array::vector(vec![1.0, 1.0, 1.0, 1.0]).unwrap(),
                Array::vector(vec![24.0, 24.0, 12.0, 4.0]).unwrap(),
                Array::vector(vec![50.0, 26.0, 7.0, 1.0]).unwrap(),
            ),
            (
                CumulativeOperation::new(0, CumulativeKind::Max),
                extrema.clone(),
                ramp.clone(),
                Array::vector(vec![3.0, 3.0, 4.0, 4.0, 5.0]).unwrap(),
                Array::vector(vec![1.0, 1.0, 3.0, 3.0, 5.0]).unwrap(),
            ),
            (
                CumulativeOperation::new(0, CumulativeKind::Max).with_reverse(true),
                Array::vector(vec![5.0, 1.0, 4.0, 1.5, 3.0]).unwrap(),
                ramp.clone(),
                Array::vector(vec![5.0, 4.0, 4.0, 3.0, 3.0]).unwrap(),
                Array::vector(vec![1.0, 3.0, 3.0, 5.0, 5.0]).unwrap(),
            ),
            (
                CumulativeOperation::new(0, CumulativeKind::Min),
                extrema.clone(),
                ramp.clone(),
                Array::vector(vec![3.0, 1.0, 1.0, 1.0, 1.0]).unwrap(),
                Array::vector(vec![1.0, 2.0, 2.0, 2.0, 2.0]).unwrap(),
            ),
            (
                CumulativeOperation::new(0, CumulativeKind::Min).with_reverse(true),
                extrema,
                ramp,
                Array::vector(vec![1.0, 1.0, 1.5, 1.5, 5.0]).unwrap(),
                Array::vector(vec![2.0, 2.0, 4.0, 4.0, 5.0]).unwrap(),
            ),
            (
                CumulativeOperation::new(0, CumulativeKind::LogSumExp),
                Array::vector(vec![0.0, 1.0, 2.0]).unwrap(),
                Array::vector(vec![1.0, 2.0, 3.0]).unwrap(),
                Array::vector(vec![0.0, (1.0 + e).ln(), (1.0 + e + e * e).ln()]).unwrap(),
                Array::vector(vec![
                    1.0,
                    (1.0 + 2.0 * e) / (1.0 + e),
                    (1.0 + 2.0 * e + 3.0 * e * e) / (1.0 + e + e * e),
                ])
                .unwrap(),
            ),
            (
                CumulativeOperation::new(0, CumulativeKind::LogSumExp).with_reverse(true),
                Array::vector(vec![0.0, 1.0, 2.0]).unwrap(),
                Array::vector(vec![1.0, 2.0, 3.0]).unwrap(),
                Array::vector(vec![(1.0 + e + e * e).ln(), (e + e * e).ln(), 2.0]).unwrap(),
                Array::vector(vec![
                    (1.0 + 2.0 * e + 3.0 * e * e) / (1.0 + e + e * e),
                    (2.0 * e + 3.0 * e * e) / (e + e * e),
                    3.0,
                ])
                .unwrap(),
            ),
        ] {
            check_operation_differentiation!(
                @approx(step = 1e-4, epsilon = 1e-6),
                operation = operation.clone(),
                cases = [{
                    primals = [primals.clone()],
                    tangents = [tangents.clone()],
                    primal_outputs = [primal_outputs.clone()],
                    tangent_outputs = [tangent_outputs.clone()],
                }],
            );
        }
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
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F64, [4]));
        let outputs = builder
            .add_instruction(
                ArrayOperation::from(CumulativeOperation::new(0, CumulativeKind::Product)),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder.build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder], vec![Placeholder]).unwrap();
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4] .
                let %2:f64[2] = slice [start_indices=[0], limit_indices=[3], strides=[2]] %0
                    %3:f64[2] = slice [start_indices=[0], limit_indices=[3], strides=[2]] %1
                    %4:f64[2] = slice [start_indices=[1], limit_indices=[4], strides=[2]] %0
                    %5:f64[2] = slice [start_indices=[1], limit_indices=[4], strides=[2]] %1
                    %6:f64[2] = mul %2 %4
                    %7:f64[2] = mul %4 %3
                    %8:f64[2] = mul %2 %5
                    %9:f64[2] = add %7 %8
                    %10:f64[1] = slice [start_indices=[0], limit_indices=[1], strides=[2]] %6
                    %11:f64[1] = slice [start_indices=[0], limit_indices=[1], strides=[2]] %9
                    %12:f64[1] = slice [start_indices=[1], limit_indices=[2], strides=[2]] %6
                    %13:f64[1] = slice [start_indices=[1], limit_indices=[2], strides=[2]] %9
                    %14:f64[1] = mul %10 %12
                    %15:f64[1] = mul %12 %11
                    %16:f64[1] = mul %10 %13
                    %17:f64[1] = add %15 %16
                    %18:f64[1] = slice [start_indices=[0], limit_indices=[1]] %6
                    %19:f64[1] = slice [start_indices=[0], limit_indices=[1]] %9
                    %20:f64[] = zero [type=f64[]]
                    %21:f64[2] = pad [edge_padding_low=[0], edge_padding_high=[1], interior_padding=[1]] %18 %20
                    %22:f64[] = zero [type=f64[]]
                    %23:f64[2] = pad [edge_padding_low=[0], edge_padding_high=[1], interior_padding=[1]] %19 %22
                    %24:f64[2] = pad [edge_padding_low=[1], edge_padding_high=[0], interior_padding=[1]] %14 %20
                    %25:f64[] = zero [type=f64[]]
                    %26:f64[2] = pad [edge_padding_low=[1], edge_padding_high=[0], interior_padding=[1]] %17 %25
                    %27:f64[2] = add %21 %24
                    %28:f64[2] = add %23 %26
                    %29:f64[1] = slice [start_indices=[0], limit_indices=[1]] %0
                    %30:f64[1] = slice [start_indices=[0], limit_indices=[1]] %1
                    %31:f64[1] = slice [start_indices=[0], limit_indices=[1]] %27
                    %32:f64[1] = slice [start_indices=[0], limit_indices=[1]] %28
                    %33:f64[1] = slice [start_indices=[2], limit_indices=[4], strides=[2]] %0
                    %34:f64[1] = slice [start_indices=[2], limit_indices=[4], strides=[2]] %1
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
    fn test_cumulative_differentiation_reverse_mode() {
        // Summing the prefix sums weights each input by the number of outputs it contributes to, which the transposed
        // sum computes directly. Nonlinear kinds reach reverse mode by transposing the linear operations that their
        // decomposition stages, which finite differences independently confirm.
        assert_eq!(
            differentiate_at(Array::vector(vec![1.0, 2.0, 3.0]).unwrap())
                .gradient(|input| Ok(input.cumulative_sum(0)?.reduce_sum(&[0], None)?))
                .unwrap(),
            Array::vector(vec![3.0, 2.0, 1.0]).unwrap(),
        );
        check_gradient!(
            |input| Ok(input.cumulative_sum(0)?.reduce_sum(&[0], None)?),
            at = Array::vector(vec![1.0, 2.0, 3.0]).unwrap(),
            step = 1e-3,
            tolerance = 1e-6,
        );
        check_gradient!(
            |input| Ok(input.reverse_cumulative_product(0)?.reduce_sum(&[0], None)?),
            at = Array::vector(vec![1.0, 2.0, 3.0]).unwrap(),
            step = 1e-3,
            tolerance = 1e-6,
        );
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
    fn test_cumulative_transposition_nonlinear_kinds() {
        // Only sums are linear. Every other kind is differentiated through the linear operations staged by its JVP
        // instead, and so direct transposition rejects it, even when the cotangent is a structural zero.
        for kind in [CumulativeKind::Product, CumulativeKind::Max, CumulativeKind::Min, CumulativeKind::LogSumExp] {
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
            Ok(Array::vector(vec![100i8, -56]).unwrap())
        );

        // A zero-length scanned axis has nothing to accumulate and keeps the operand's exact type.
        let empty = Array::new(ArrayType::new_static(DataType::F32, [0, 2]), Vec::new()).unwrap();
        for kind in [
            CumulativeKind::Sum,
            CumulativeKind::Product,
            CumulativeKind::Max,
            CumulativeKind::Min,
            CumulativeKind::LogSumExp,
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
        // Accumulation happens in the operand's own encoding, so every partial sum is re-encoded rather than only the
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

        // Complex payloads accumulate both components and multiply as complex numbers.
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

        // The result carries the operand's complete type, including a non-default physical layout.
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

        // Selection happens in the operand's own element type, so a low-precision payload is returned bit for bit
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
        // equal operands add exactly `log(2)` at any magnitude, in both directions.
        let large = Array::vector(vec![1000.0, 1000.0]).unwrap();
        assert_eq!(
            large.cumulative_log_sum_exp(0),
            Ok(Array::vector(vec![1000.0, 1000.0 + std::f64::consts::LN_2]).unwrap())
        );
        assert_eq!(
            large.reverse_cumulative_log_sum_exp(0),
            Ok(Array::vector(vec![1000.0 + std::f64::consts::LN_2, 1000.0]).unwrap()),
        );

        // Negative infinity is the combining operator's identity, so it neither contributes to nor poisons a later
        // prefix, while a NaN operand propagates.
        assert_eq!(
            Array::vector(vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 2.0]).unwrap().cumulative_log_sum_exp(0),
            Ok(Array::vector(vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 2.0]).unwrap()),
        );
        let with_nan = Array::vector(vec![1.0, f64::NAN, 2.0]).unwrap().cumulative_log_sum_exp(0).unwrap().to_f64s();
        assert_eq!(with_nan[0], 1.0);
        assert!(with_nan[1].is_nan() && with_nan[2].is_nan());

        // Accumulation happens in the operand's own encoding, so each partial result is rounded to it, and the lowest
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
    }

    #[test]
    fn test_array_type_cumulative() {
        // A prefix scan preserves the complete operand type, including its memory placement and the sharding of the
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
                    "`cumulative` with kind `{kind}` requires numeric inputs but got `bool`"
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

        // The exponential and the logarithm have no meaning for non-floating-point or complex inputs.
        for data_type in [DataType::I32, DataType::Boolean, DataType::C64, DataType::Token, DataType::Zero] {
            assert_eq!(
                ArrayType::new_static(data_type, [3, 2]).cumulative(1, CumulativeKind::LogSumExp),
                Err(TypeError::invalid(format!(
                    "`cumulative` with kind `log_sum_exp` requires real floating-point inputs but got `{data_type}`",
                ))),
            );
        }

        // `f8e8m0fnu` encodes bare positive exponents, so its smallest element exponentiates to one instead of acting
        // as the combining operator's identity, and the lowest `f6e2m3fn` value already changes when combined with one
        // more copy of itself. These finite sentinels, however, are identities after every rounded pairwise
        // combination.
        for data_type in [DataType::F8E8M0FNU, DataType::F6E2M3FN] {
            assert_eq!(
                ArrayType::new_static(data_type, [3, 2]).cumulative(1, CumulativeKind::LogSumExp),
                Err(TypeError::invalid(format!(
                    "`cumulative` with kind `log_sum_exp` requires a floating-point format whose lowest value is a \
                     `log_add_exp` identity but got `{data_type}`",
                ))),
            );
        }
        for data_type in [DataType::F32, DataType::F4E2M1FN, DataType::F8E4M3B11FNUZ, DataType::F6E3M2FN] {
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
        for kind in [CumulativeKind::Product, CumulativeKind::Max, CumulativeKind::Min, CumulativeKind::LogSumExp] {
            assert_eq!(
                input.cumulative(0, kind),
                Err(TypeError::invalid(format!(
                    "`cumulative` with kind `{kind}` cannot scan inputs with unreduced axes"
                ))),
            );
        }
    }

    #[test]
    fn test_associative_scan() {
        // The decomposition is checked against the sequential scan of the same combiner, over both parities of the
        // scanned extent and in both directions. Summation pins the positions each output accumulates over, and the
        // left projection (which is associative but not commutative) additionally pins the operand order that the
        // construction passes to the combiner: its forward scan is the first element repeated and its reverse scan the
        // last.
        let add = |left: &Array, right: &Array| left.add(right);
        let first = |left: &Array, _right: &Array| Ok(left.clone());
        for extent in 0..=9usize {
            let values = (1..=extent).map(|value| value as f64).collect::<Vec<_>>();
            let input = Array::vector(values.clone()).unwrap();
            let shape = StaticShape::new(vec![extent]);
            for reverse in [false, true] {
                assert_eq!(
                    associative_scan(&input, 0, reverse, &add).map(|output| output.to_f64s()),
                    cumulative_evaluate(values.as_slice(), &shape, 0, reverse, |left, right| Ok(left + right)),
                    "summation over extent {extent}, reverse {reverse}",
                );
                assert_eq!(
                    associative_scan(&input, 0, reverse, &first).map(|output| output.to_f64s()),
                    cumulative_evaluate(values.as_slice(), &shape, 0, reverse, |left, _right| Ok(left)),
                    "left projection over extent {extent}, reverse {reverse}",
                );
            }
        }

        // Boolean operands are interleaved with a disjunction, because Booleans have no addition.
        let or = |left: &Array, right: &Array| left.or(right);
        let booleans = Array::vector(vec![false, false, true, false, false]).unwrap();
        assert_eq!(
            associative_scan(&booleans, 0, false, &or),
            Ok(Array::vector(vec![false, false, true, true, true]).unwrap()),
        );
        assert_eq!(
            associative_scan(&booleans, 0, true, &or),
            Ok(Array::vector(vec![true, true, true, false, false]).unwrap()),
        );

        // The construction scans one axis of a higher-rank operand independently per row.
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(
            associative_scan(&matrix, 1, false, &add),
            Ok(Array::matrix(2, 3, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]).unwrap()),
        );
        assert_eq!(
            associative_scan(&matrix, 1, true, &add),
            Ok(Array::matrix(2, 3, vec![6.0, 5.0, 3.0, 15.0, 11.0, 6.0]).unwrap()),
        );
        assert_eq!(
            associative_scan(&matrix, 0, false, &add),
            Ok(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]).unwrap()),
        );

        // The construction slices at staging-time positions, so it needs an in-bounds axis.
        assert_eq!(
            associative_scan(&matrix, 2, false, &add),
            Err(ProgramError::Type(TypeError::invalid("`associative_scan` axis 2 is out of bounds for rank 2"))),
        );
    }

    #[test]
    fn test_associative_scan_provenance() {
        // Every instruction that the decomposition stages carries the nested framework scopes, which attribute it to
        // the associative-scan decomposition in renderings that include provenance.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = context.input(ArrayType::new_static(DataType::F64, [2]));
        let output = associative_scan(&input, 0, false, &|left, right| left.add(right)).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![output.atom_id().unwrap()], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            std::fmt::from_fn(|formatter| program.render(formatter, 0, ProgramRenderingMode::WithProvenance))
                .to_string(),
            indoc! {"
                lambda %0:f64[2] .
                let %1:f64[1] = slice [start_indices=[0], limit_indices=[1], strides=[2]] %0 ; provenance=ryft::differentiation::associative_scan
                    %2:f64[1] = slice [start_indices=[1], limit_indices=[2], strides=[2]] %0 ; provenance=ryft::differentiation::associative_scan
                    %3:f64[1] = add %1 %2 ; provenance=ryft::differentiation::associative_scan
                    %4:f64[1] = slice [start_indices=[0], limit_indices=[1]] %0 ; provenance=ryft::differentiation::associative_scan
                    %5:f64[] = zero [type=f64[]] ; provenance=ryft::differentiation::associative_scan
                    %6:f64[2] = pad [edge_padding_low=[0], edge_padding_high=[1], interior_padding=[1]] %4 %5 ; provenance=ryft::differentiation::associative_scan
                    %7:f64[2] = pad [edge_padding_low=[1], edge_padding_high=[0], interior_padding=[1]] %3 %5 ; provenance=ryft::differentiation::associative_scan
                    %8:f64[2] = add %6 %7 ; provenance=ryft::differentiation::associative_scan
                in (%8)"
            },
        );
    }

    #[test]
    fn test_cumulative_evaluate() {
        let values = (1..=6).map(|value| value as f64).collect::<Vec<_>>();
        let shape = StaticShape::new(vec![2, 3]);
        let add = |left: f64, right: f64| Ok(left + right);

        // Forward scans accumulate prefixes and reverse scans accumulate suffixes, per row of the scanned axis, and
        // scanning the outer axis accumulates across the row stride instead.
        assert_eq!(
            cumulative_evaluate(values.as_slice(), &shape, 1, false, add),
            Ok(vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0])
        );
        assert_eq!(
            cumulative_evaluate(values.as_slice(), &shape, 1, true, add),
            Ok(vec![6.0, 5.0, 3.0, 15.0, 11.0, 6.0])
        );
        assert_eq!(
            cumulative_evaluate(values.as_slice(), &shape, 0, false, add),
            Ok(vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0])
        );

        // A scanned axis with fewer than two elements has nothing to accumulate.
        assert_eq!(cumulative_evaluate(values.as_slice(), &StaticShape::new(vec![6, 1]), 1, false, add), Ok(values));
        assert_eq!(cumulative_evaluate(&[], &StaticShape::new(vec![0, 3]), 0, false, add), Ok(Vec::<f64>::new()));

        // A zero-extent axis to the *right* of the scanned one leaves the payload empty while the scanned extent itself
        // is still at least two, so the block bounds are derived from dimension products rather than from the payload
        // length and the scanned axis stride, which is zero here.
        assert_eq!(cumulative_evaluate(&[], &StaticShape::new(vec![2, 0]), 0, false, add), Ok(Vec::<f64>::new()));
        assert_eq!(cumulative_evaluate(&[], &StaticShape::new(vec![3, 0, 2]), 0, true, add), Ok(Vec::<f64>::new()));
    }
}
