//! Operations that normalize logits into categorical distributions and score them against targets. These are
//! compositions of other operations rather than primitives: the [`Softmax`] value capability trait applies them to
//! eager [`Array`](crate::Array)s and traced values alike, so the same code executes immediately or records into a
//! program depending on the value it runs on, and every program transform supports them through the rules of the
//! operations that they stage. The capabilities fall into the following groups:
//!
//!   - **Normalization:** [`softmax`](Softmax::softmax) and [`log_softmax`](Softmax::log_softmax) normalize the
//!     exponentials of the logits along one or more class axes into probabilities and log-probabilities.
//!   - **Cross-Entropy Losses:** [`softmax_cross_entropy`](Softmax::softmax_cross_entropy) scores probability targets
//!     and [`softmax_cross_entropy_with_integer_labels`](Softmax::softmax_cross_entropy_with_integer_labels) scores
//!     one integer class label per row, both dropping the class axes.
//!   - **Masked Variants:** the `masked_*` functions (e.g., [`masked_softmax`](Softmax::masked_softmax))
//!     take a Boolean mask that selects the classes taking part, like the `where` argument of the corresponding
//!     [JAX](https://docs.jax.dev/en/latest/index.html) and [Optax](https://optax.readthedocs.io/en/latest/) functions.
//!     Unselected classes have zero probability, contribute nothing to a loss, and receive zero derivatives.
//!   - **Targets Supplied As Logits:** [`softmax_cross_entropy_with_target_logits`] creates a
//!     [`SoftmaxCrossEntropyWithTargetLogits`] function that scores predictions against a fixed target distribution
//!     supplied as logits rather than rounded probabilities.
//!
//! Inputs have real floating-point elements and static shapes, with at least one class along every class axis. Narrow
//! floating-point formats compute in `f32` while `f64` keeps its precision, and outputs retain the input element type.
//! Several class axes are normalized jointly as one flattened class axis in increasing axis order, which also orders
//! integer labels. Logits of negative infinity represent excluded classes, and a row that contains NaN or positive
//! infinity, or only excluded classes, produces NaNs. Inputs may vary over manual mesh axes, and inputs whose sharding
//! records unreduced or reduced mesh axes are rejected, because a nonlinear normalization preserves neither partial
//! sums nor zero-filled replicas. Batching follows the staged operations: every batch item is normalized independently
//! and the mapped axis keeps its position.
//!
//! # Differentiation
//!
//! [`log_softmax`](Softmax::log_softmax) separates one maximum logit from the sum of the other exponentials and uses
//! `ln_1p` for the small remaining mass. The selected slot is a constant zero after shifting, so its derivative is
//! accumulated from the other slots rather than subtracting a rounded probability from one, and small losses and
//! derivatives survive even when the largest probability rounds to one. This is the collapse-resistant construction
//! of ["A Fix for Softmax and Cross-Entropy"](https://jonbarron.info/softmaxfix/), which the probabilities and both
//! losses inherit. [`softmax_cross_entropy_with_target_logits`] additionally computes the prediction gradient from
//! the logits' difference before rounding either distribution to probabilities, through a retained custom derivative
//! rule that remains differentiable at higher orders, so nearly matching prediction and target distributions do not
//! cancel. None of these functions is linear, so reverse mode transposes the linearizations of the staged operations.
//! Probability targets are differentiable as well, whereas integer labels, masks, and target logits have
//! structural-zero derivatives.
//!
//! # Example
//!
//! ```rust
//! # use ryft_core::{Array, ProgramError, Softmax};
//! # fn main() -> Result<(), ProgramError> {
//! let logits = Array::vector(vec![1.0f64, 2.0, 3.0])?;
//! let probabilities = logits.softmax(0)?.to_f64s();
//! assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-15);
//! let mask = Array::vector(vec![true, false, true])?;
//! assert_eq!(logits.masked_softmax(&mask, 0)?.to_f64s()[1], 0.0);
//! let loss = logits.softmax_cross_entropy_with_integer_labels(&Array::scalar(2i32)?, 0)?;
//! assert_eq!(loss.to_f64s()[0], -logits.log_softmax(0)?.to_f64s()[2]);
//! # Ok(())
//! # }
//! ```

use ryft_macros::capability;

use crate::arrays::{ArrayIrType, ArrayType, DataType};
use crate::axes::{Axes, AxisError, NamedAxes};
use crate::batching::{BatchableType, RecursiveBatchingPolicy};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{CotangentBatchingPolicy, DifferentiableType};
use crate::operations::Capability;
use crate::operations::arithmetic::{Abs, Mul, Neg, Sub};
use crate::operations::collectives::parallel_vary::ManualVariationAlignment;
use crate::operations::comparisons::Compare;
use crate::operations::constants::fill::Fill;
use crate::operations::constants::iota::Iota;
use crate::operations::constants::zero_like::ZeroLike;
use crate::operations::control_flow::select::Select;
use crate::operations::custom_functions::functions::{CustomFunction, CustomFunctionJvp, WithJvp, custom_function};
use crate::operations::custom_functions::operations::CustomFunctionOperation;
use crate::operations::custom_functions::rules::CustomRuleRegistration;
use crate::operations::differentiation::stop_gradient::StopGradient;
use crate::operations::exponential::{Exp, ExpM1, Ln1p};
use crate::operations::logical::{And, Or};
use crate::operations::manipulation::broadcasting::Broadcast;
use crate::operations::manipulation::conversions::ConvertElementType;
use crate::operations::manipulation::reshaping::Reshape;
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::reductions::{ArgMax, Reduce};
use crate::programs::{ProgramError, TypeError, Typed, Value};
use crate::tracing::{DomainTracer, DomainTracingContext, TracingContext};

/// Capability that provides support for stable softmax normalization and cross-entropy along one or more class axes.
/// `axes` accepts one axis or a list of axes, and negative axes count from the end. Several class axes are normalized
/// jointly, as if they were flattened into one class axis in increasing axis order, which is also the order in which
/// an integer label indexes the classes. Inputs have real floating-point elements and static shapes, with at least one
/// class along every class axis. Half-precision and smaller formats use `f32` arithmetic internally; `f64` keeps its
/// precision. Normalization outputs retain the input element type. Logits of negative infinity represent excluded
/// classes. A row containing NaN or positive infinity, or consisting entirely of negative infinity, produces NaNs.
/// A finite singleton has probability one and log-probability zero, subject to the final element-format conversion.
/// Formats that cannot represent an output (e.g., zero or negative log-probabilities in `f8e8m0fnu`) follow
/// [`ConvertElementType`]'s conversion semantics and can produce NaNs.
///
/// The `masked_*` functions take a Boolean `mask` with the logits' shape that selects the classes taking part. A class
/// outside the mask has probability zero and log-probability negative infinity, contributes nothing to a loss, and
/// receives a zero derivative, whatever logit it holds; a row without any selected class has zero probabilities,
/// negative infinite log-probabilities, and a zero loss.
///
/// These functions preserve small losses and derivatives even when the largest probability rounds to one. Computing
/// `log(softmax(logits))` or subtracting a logit from a separately rounded log-sum-exp loses that information.
///
/// Inputs may vary over manual mesh axes, and the outputs vary the same way. Inputs whose sharding records unreduced
/// or reduced mesh axes are rejected, because a nonlinear normalization preserves neither partial sums nor zero-filled
/// replicas.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and composite array IR values implement it for [`ArrayIrType`].
///
/// # Examples
///
/// ```rust
/// # use ryft_core::{Array, ProgramError, Softmax};
/// # fn main() -> Result<(), ProgramError> {
/// let logits = Array::vector(vec![9.0f32, -9.0, -9.0])?;
/// let loss = logits.softmax_cross_entropy_with_integer_labels(&Array::scalar(0i32)?, -1)?;
/// assert!(loss.elements::<f32>()?[0] > 0.0);
/// # Ok(())
/// # }
/// ```
#[capability(projection(ArrayIrType => ArrayType))]
pub trait Softmax<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Returns class probabilities `qᵢ = exp(xᵢ) / ∑ⱼ exp(xⱼ)` for logits `x = self`. The indices range over the
    /// classes along `axes`; all other axes are processed independently. The output has the logits' shape and element
    /// type, and small derivatives survive even when a probability rounds to one.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] for non-real-floating inputs, dynamic shapes, invalid, duplicate, or empty class
    /// axes, inputs with unreduced or reduced mesh axes, or an operation binding failure.
    fn softmax<A: Into<Axes>>(&self, axes: A) -> Result<Self, ProgramError>;

    /// Returns probabilities `qᵢ = exp(xᵢ) / ∑ⱼ∈M exp(xⱼ)` for logits `x = self` and selected classes `i ∈ M`.
    /// Here `M = {j | maskⱼ}` contains the classes selected by the Boolean `mask` along `axes`; other axes are
    /// processed independently. Unselected classes have probability zero; an empty `M` gives zero for the whole row.
    /// The mask must have the logits' shape, and the output keeps the logits' shape and element type.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] under the same conditions as [`Self::softmax`], or when `mask` is not a Boolean
    /// array with the logits' shape.
    fn masked_softmax<A: Into<Axes>>(&self, mask: &Self, axes: A) -> Result<Self, ProgramError>;

    /// Returns log-probabilities `ℓᵢ = xᵢ − ln(∑ⱼ exp(xⱼ))` for logits `x = self`. The indices range over the
    /// classes along `axes`; all other axes are processed independently. The output has the logits' shape and element
    /// type. Computing directly in log space retains small negative log-probabilities that would be lost by taking
    /// the logarithm of a rounded [`Self::softmax`] result.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] under the same conditions as [`Self::softmax`].
    fn log_softmax<A: Into<Axes>>(&self, axes: A) -> Result<Self, ProgramError>;

    /// Returns log-probabilities `ℓᵢ = xᵢ − ln(∑ⱼ∈M exp(xⱼ))` for logits `x = self` and selected classes `i ∈ M`.
    /// Here `M = {j | maskⱼ}` contains the classes selected by the Boolean `mask` along `axes`; other axes are
    /// processed independently. Unselected classes have log-probability `−∞`; an empty `M` gives `−∞` for the whole
    /// row. The mask must have the logits' shape, and the output keeps the logits' shape and element type.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] under the same conditions as [`Self::masked_softmax`].
    fn masked_log_softmax<A: Into<Axes>>(&self, mask: &Self, axes: A) -> Result<Self, ProgramError>;

    /// Returns cross-entropy `L = −∑ᵢ pᵢ ℓᵢ`, where `p = probability_targets` and `ℓ` contains the log-probabilities
    /// of `self` from [`Self::log_softmax`]. The sum ranges over the classes along `axes`, producing one loss for each
    /// combination of remaining indices. The output drops the class axes and keeps the logits' element type. Targets
    /// must have the logits' shape and real floating-point elements; callers supply `pᵢ ≥ 0` with `∑ᵢ pᵢ = 1`. Targets
    /// are converted to the logits' working type, and both inputs are differentiable. For an excluded class,
    /// `0 · (−∞) = 0`, with zero derivative for that zero target. Inputs with different manual variation are aligned
    /// before computing the loss. For targets supplied as logits, use [`softmax_cross_entropy_with_target_logits`] to
    /// retain accurate prediction gradients when the distributions nearly coincide.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] under the same conditions as [`Self::softmax`], or for incompatible target shapes
    /// or element types.
    fn softmax_cross_entropy<A: Into<Axes>>(&self, probability_targets: &Self, axes: A) -> Result<Self, ProgramError>;

    /// Returns masked cross-entropy `L = −∑ᵢ∈M pᵢ ℓᵢ`, where `p = probability_targets` and `ℓ` contains the
    /// log-probabilities of `self` from [`Self::masked_log_softmax`]. The set `M = {i | maskᵢ}` contains the classes
    /// selected by the Boolean `mask` along `axes`; all other axes identify independent losses. Unselected classes
    /// contribute nothing; targets are not renormalized after masking, and an empty `M` gives zero loss. Target
    /// requirements, output shape and element type, and differentiation follow [`Self::softmax_cross_entropy`].
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] under the same conditions as [`Self::softmax_cross_entropy`], or when `mask` is not
    /// a Boolean array with the logits' shape.
    fn masked_softmax_cross_entropy<A: Into<Axes>>(
        &self,
        probability_targets: &Self,
        mask: &Self,
        axes: A,
    ) -> Result<Self, ProgramError>;

    /// Returns the negative log-probability of each labeled class: `L = −ℓ[y]`, where `y` is a class index from
    /// `labels` and `ℓ` contains the log-probabilities of `self` from [`Self::log_softmax`] along `axes`. The output
    /// drops the class axes and keeps the logits' element type. Labels have integer elements and the output shape;
    /// each label indexes the classes flattened along `axes` in increasing axis order. Labels have structural-zero
    /// derivatives; values outside `[0, class_count)` produce NaN. Inputs with different manual variation are
    /// aligned before looking up the labels.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] under the same conditions as [`Self::softmax`], or for incompatible label shapes
    /// or element types.
    fn softmax_cross_entropy_with_integer_labels<A: Into<Axes>>(
        &self,
        labels: &Self,
        axes: A,
    ) -> Result<Self, ProgramError>;

    /// Returns the negative masked log-probability of each labeled class: `L = −ℓ[y]`, where `y` is a class index from
    /// `labels` and `ℓ` contains the log-probabilities of `self` from [`Self::masked_log_softmax`]. Normalization uses
    /// the classes selected by the Boolean `mask` along `axes`; other axes identify independent losses. A valid label
    /// naming an unselected class gives infinite loss; a row with no selected class gives zero loss regardless of its
    /// label. Labels have integer elements and index the classes flattened along `axes` in increasing axis order.
    /// Label shape, output shape and element type, and differentiation follow
    /// [`Self::softmax_cross_entropy_with_integer_labels`].
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] under the same conditions as [`Self::softmax_cross_entropy_with_integer_labels`],
    /// or when `mask` is not a Boolean array with the logits' shape.
    fn masked_softmax_cross_entropy_with_integer_labels<A: Into<Axes>>(
        &self,
        labels: &Self,
        mask: &Self,
        axes: A,
    ) -> Result<Self, ProgramError>;
}

impl<
    V: Value<Type = ArrayType, Domain: Fill<f64, V> + Iota<V>>
        + ZeroLike
        + Neg
        + Sub
        + Mul
        + Exp
        + Ln1p
        + Compare
        + And
        + Select
        + Broadcast
        + ConvertElementType
        + Reshape
        + Transpose
        + Reduce
        + ArgMax
        + ManualVariationAlignment,
> Softmax<ArrayType> for V
{
    fn softmax<A: Into<Axes>>(&self, axes: A) -> Result<Self, ProgramError> {
        let (axes, working_type) = validate_normalization("softmax", &self.r#type(), axes.into())?;

        // Keep the exponential widened too as narrowing log-probabilities first would lose their small corrections.
        self.convert_element_type(working_type)?
            .log_softmax(axes)?
            .exp()?
            .convert_element_type(self.r#type().data_type())
    }

    fn masked_softmax<A: Into<Axes>>(&self, mask: &Self, axes: A) -> Result<Self, ProgramError> {
        let (axes, working_type) = validate_normalization("masked_softmax", &self.r#type(), axes.into())?;
        validate_normalization_mask("masked_softmax", &self.r#type(), &mask.r#type())?;

        // The exponential of a negative infinite log-probability is the zero probability of an unselected class.
        self.convert_element_type(working_type)?
            .masked_log_softmax(mask, axes)?
            .exp()?
            .convert_element_type(self.r#type().data_type())
    }

    fn log_softmax<A: Into<Axes>>(&self, axes: A) -> Result<Self, ProgramError> {
        let (axes, working_type) = validate_normalization("log_softmax", &self.r#type(), axes.into())?;
        let mut flattened = FlattenedClassInputs::new(vec![self.convert_element_type(working_type)?], &axes)?;
        let input = flattened.inputs.pop().unwrap();
        let axis = flattened.axis;
        let pivot = input.argmax_with_index_data_type(axis, DataType::I64)?;
        let mask = one_hot_mask(&input, &pivot, axis)?;
        let zero = input.zero_like()?;
        let pivot_value = mask.select(&input, &zero)?.reduce_sum(&[axis], None)?;
        let output_axes = (0..input.r#type().rank()).filter(|&index| index != axis).collect::<Vec<_>>();
        let pivot_value = pivot_value.broadcast(input.r#type().into_owned(), &output_axes)?;

        // The pivot must be a constant zero rather than `input[pivot] - input[pivot]`: the latter permits an automatic
        // differentiation implementation to subtract two rounded coefficients and erase its small gradient. Selecting
        // one index also preserves the full derivative at ties, unlike treating every maximum as a constant zero.
        let shifted = mask.select(&zero, &input.sub(&pivot_value)?)?;
        let tail = mask.select(&zero, &shifted.exp()?)?.reduce_sum(&[axis], None)?;
        let correction = tail.ln_1p()?.broadcast(input.r#type().into_owned(), &output_axes)?;
        let output = shifted.sub(&correction)?;

        // A singleton still follows the documented non-finite semantics. The finite test also propagates a NaN
        // chosen by argmax without relying on arithmetic in an otherwise excluded slot.
        let finite = pivot_value.sub(&pivot_value)?.equal(&zero)?;
        let nan = input.domain().fill(&input.r#type(), f64::NAN)?;
        let output = finite.select(&output, &nan)?;
        flattened.restore(output)?.convert_element_type(self.r#type().data_type())
    }

    fn masked_log_softmax<A: Into<Axes>>(&self, mask: &Self, axes: A) -> Result<Self, ProgramError> {
        let (axes, working_type) = validate_normalization("masked_log_softmax", &self.r#type(), axes.into())?;
        validate_normalization_mask("masked_log_softmax", &self.r#type(), &mask.r#type())?;
        let aligned = V::align_manual_variation(&[self.convert_element_type(working_type)?, mask.clone()])?;
        let mut flattened = FlattenedClassInputs::new(aligned, &axes)?;
        let mask = flattened.inputs.pop().unwrap();
        let input = flattened.inputs.pop().unwrap();

        // Unselected classes are excluded exactly like `-∞` logits, which also discards any NaN they hold. A row
        // without a selected class normalizes to NaNs, which the final selection replaces by `-∞`. The literal is
        // a non-differentiable constant, so it carries the input's placement and variation directly.
        let negative_infinity = input.domain().fill(&input.r#type(), f64::NEG_INFINITY)?;
        let log_probabilities = mask.select(&input, &negative_infinity)?.log_softmax(flattened.axis)?;
        let output = mask.select(&log_probabilities, &negative_infinity)?;
        flattened.restore(output)?.convert_element_type(self.r#type().data_type())
    }

    fn softmax_cross_entropy<A: Into<Axes>>(&self, probability_targets: &Self, axes: A) -> Result<Self, ProgramError> {
        let (axes, working_type) = validate_normalization("softmax_cross_entropy", &self.r#type(), axes.into())?;
        validate_normalization_targets("softmax_cross_entropy", &self.r#type(), &probability_targets.r#type())?;

        // Aligning once here stages a single variation transition instead of one per binary operation below.
        let aligned = V::align_manual_variation(&[
            self.convert_element_type(working_type)?,
            probability_targets.convert_element_type(working_type)?,
        ])?;
        let mut flattened = FlattenedClassInputs::new(aligned, &axes)?;
        let targets = flattened.inputs.pop().unwrap();
        let input = flattened.inputs.pop().unwrap();
        let log_probabilities = input.log_softmax(flattened.axis)?;
        cross_entropy_loss(&targets, &log_probabilities, None, flattened.axis)?
            .convert_element_type(self.r#type().data_type())
    }

    fn masked_softmax_cross_entropy<A: Into<Axes>>(
        &self,
        probability_targets: &Self,
        mask: &Self,
        axes: A,
    ) -> Result<Self, ProgramError> {
        let (axes, working_type) = validate_normalization("masked_softmax_cross_entropy", &self.r#type(), axes.into())?;
        validate_normalization_targets("masked_softmax_cross_entropy", &self.r#type(), &probability_targets.r#type())?;
        validate_normalization_mask("masked_softmax_cross_entropy", &self.r#type(), &mask.r#type())?;
        let aligned = V::align_manual_variation(&[
            self.convert_element_type(working_type)?,
            probability_targets.convert_element_type(working_type)?,
            mask.clone(),
        ])?;
        let mut flattened = FlattenedClassInputs::new(aligned, &axes)?;
        let mask = flattened.inputs.pop().unwrap();
        let targets = flattened.inputs.pop().unwrap();
        let input = flattened.inputs.pop().unwrap();
        let log_probabilities = input.masked_log_softmax(&mask, flattened.axis)?;
        cross_entropy_loss(&targets, &log_probabilities, Some(&mask), flattened.axis)?
            .convert_element_type(self.r#type().data_type())
    }

    fn softmax_cross_entropy_with_integer_labels<A: Into<Axes>>(
        &self,
        labels: &Self,
        axes: A,
    ) -> Result<Self, ProgramError> {
        let function = "softmax_cross_entropy_with_integer_labels";
        let (axes, working_type) = validate_normalization(function, &self.r#type(), axes.into())?;
        validate_normalization_labels(function, &self.r#type(), &labels.r#type(), &axes)?;

        // The class coordinates take the logits' type, including its varying manual axes, so the labels must vary over
        // the same axes before they are broadcast against them, just as a binary capability aligns its inputs before
        // binding. The transition that this stages on the logits, when they are the invariant input, transposes to
        // the cross-device sum that their replicated contributions require.
        let mut aligned = V::align_manual_variation(&[
            self.convert_element_type(working_type)?,
            labels.convert_element_type(DataType::I64)?,
        ])?;
        let labels = aligned.pop().unwrap();
        let mut flattened = FlattenedClassInputs::new(aligned, &axes)?;
        let log_probabilities = flattened.inputs.pop().unwrap().log_softmax(flattened.axis)?;
        cross_entropy_loss_with_integer_labels(&log_probabilities, &labels, None, flattened.axis)?
            .convert_element_type(self.r#type().data_type())
    }

    fn masked_softmax_cross_entropy_with_integer_labels<A: Into<Axes>>(
        &self,
        labels: &Self,
        mask: &Self,
        axes: A,
    ) -> Result<Self, ProgramError> {
        let function = "masked_softmax_cross_entropy_with_integer_labels";
        let (axes, working_type) = validate_normalization(function, &self.r#type(), axes.into())?;
        validate_normalization_labels(function, &self.r#type(), &labels.r#type(), &axes)?;
        validate_normalization_mask(function, &self.r#type(), &mask.r#type())?;
        let mut aligned = V::align_manual_variation(&[
            self.convert_element_type(working_type)?,
            mask.clone(),
            labels.convert_element_type(DataType::I64)?,
        ])?;
        let labels = aligned.pop().unwrap();
        let mut flattened = FlattenedClassInputs::new(aligned, &axes)?;
        let mask = flattened.inputs.pop().unwrap();
        let log_probabilities = flattened.inputs.pop().unwrap().masked_log_softmax(&mask, flattened.axis)?;
        cross_entropy_loss_with_integer_labels(&log_probabilities, &labels, Some(&mask), flattened.axis)?
            .convert_element_type(self.r#type().data_type())
    }
}

/// Array inputs with their class axes flattened for single-axis arithmetic, together with the information needed to
/// restore their original shape and axis order.
struct FlattenedClassInputs<V> {
    /// Flattened inputs, in their original input order.
    inputs: Vec<V>,

    /// Position of the (possibly merged) class axis in the flattened inputs.
    axis: usize,

    /// Inverse permutation and transposed shape that restore the original layout, present for several class axes.
    restoration: Option<(Vec<usize>, Vec<usize>)>,
}

impl<V: Value<Type = ArrayType> + Reshape + Transpose> FlattenedClassInputs<V> {
    /// Creates a new [`FlattenedClassInputs`] instance by flattening the class `axes` of `inputs`. Callers provide at
    /// least one input, a shared static shape, and non-empty, distinct axes validated for that shape in increasing
    /// order. Several class axes are moved to the end and merged into one; a single class axis leaves the inputs
    /// unchanged.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if a transpose or reshape fails.
    fn new(inputs: Vec<V>, axes: &[usize]) -> Result<Self, ProgramError> {
        if let [axis] = axes {
            return Ok(Self { inputs, axis: *axis, restoration: None });
        }
        let input_type = inputs[0].r#type();
        let rank = input_type.rank();
        let shape = input_type.static_shape().unwrap();
        let permutation = (0..rank).filter(|axis| !axes.contains(axis)).chain(axes.iter().copied()).collect::<Vec<_>>();
        let mut inverse_permutation = vec![0; rank];
        for (position, &axis) in permutation.iter().enumerate() {
            inverse_permutation[axis] = position;
        }
        let transposed_shape = permutation.iter().map(|&axis| shape[axis]).collect::<Vec<_>>();
        let class_axis = rank - axes.len();
        let mut flattened_shape = transposed_shape[..class_axis].to_vec();
        flattened_shape.push(transposed_shape[class_axis..].iter().product());
        let inputs = inputs
            .into_iter()
            .map(|input| input.transpose(permutation.clone())?.reshape(flattened_shape.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { inputs, axis: class_axis, restoration: Some((inverse_permutation, transposed_shape)) })
    }

    /// Restores the original shape and axis order of `value`, an output with the flattened inputs' shape.
    fn restore(&self, value: V) -> Result<V, ProgramError> {
        match &self.restoration {
            None => Ok(value),
            Some((inverse_permutation, transposed_shape)) => {
                value.reshape(transposed_shape.clone())?.transpose(inverse_permutation.clone())
            }
        }
    }
}

/// Validates the logits contract shared by every normalization entry point and returns the normalized class axes in
/// increasing order with the arithmetic data type: `f64` keeps its precision and every other real floating-point
/// format computes in `f32`. Diagnostics name `function`, the public entry point being validated.
fn validate_normalization(
    function: &str,
    input_type: &ArrayType,
    axes: Axes,
) -> Result<(Vec<usize>, DataType), ProgramError> {
    let rank = input_type.rank();
    let mut axes = axes.normalize(rank).map_err(|error| match error {
        AxisError::OutOfBounds { axis, rank } => {
            TypeError::invalid(format!("`{function}` axis {axis} is out of bounds for rank {rank}"))
        }
        AxisError::DuplicateAxis { axis } => {
            TypeError::invalid(format!("`{function}` axes contain duplicate axis {axis}"))
        }
        AxisError::UnboundAxisName { name } => {
            TypeError::invalid(format!("`{function}` axis name `{name}` is not bound by any enclosing transform"))
        }
    })?;

    if axes.is_empty() {
        return Err(TypeError::invalid(format!("`{function}` requires at least one class axis")).into());
    }

    axes.sort_unstable();

    if !input_type.data_type().is_floating_point() {
        return Err(TypeError::invalid(format!("`{function}` requires real floating-point logits")).into());
    }

    validate_normalization_placement(function, input_type)?;

    let shape = input_type
        .static_shape()
        .ok_or_else(|| TypeError::invalid(format!("`{function}` requires a static input shape")))?;
    if axes.iter().any(|&axis| shape[axis] == 0) {
        return Err(TypeError::invalid(format!("`{function}` requires a non-empty class axis")).into());
    }

    let r#type = if input_type.data_type() == DataType::F64 { DataType::F64 } else { DataType::F32 };
    Ok((axes, r#type))
}

/// Rejects inputs with partial sums or zero-filled replicas across mesh axes, which nonlinear normalization cannot
/// preserve. Diagnostics name `function`, the public entry point being validated.
fn validate_normalization_placement(function: &str, input_type: &ArrayType) -> Result<(), ProgramError> {
    if !input_type.unreduced_axes().is_empty() || !input_type.reduced_axes().is_empty() {
        return Err(TypeError::invalid(format!("`{function}` does not support unreduced or reduced inputs")).into());
    }
    Ok(())
}

/// Validates that `mask_type` is a Boolean array with the logits' shape and a supported placement.
fn validate_normalization_mask(
    function: &str,
    input_type: &ArrayType,
    mask_type: &ArrayType,
) -> Result<(), ProgramError> {
    if mask_type.shape() != input_type.shape() || mask_type.data_type() != DataType::Boolean {
        return Err(
            TypeError::invalid(format!("`{function}` mask must be a Boolean array with the logits' shape")).into()
        );
    }
    validate_normalization_placement(function, mask_type)
}

/// Validates that `targets_type` is a real floating-point array with the logits' shape and a supported placement.
fn validate_normalization_targets(
    function: &str,
    input_type: &ArrayType,
    targets_type: &ArrayType,
) -> Result<(), ProgramError> {
    if targets_type.shape() != input_type.shape() || !targets_type.data_type().is_floating_point() {
        return Err(TypeError::invalid(format!(
            "`{function}` probability targets must have the logits' shape and real floating-point elements",
        ))
        .into());
    }
    validate_normalization_placement(function, targets_type)
}

/// Validates that `labels_type` is an integer array with the logits' shape without the class `axes`
/// and a supported placement.
fn validate_normalization_labels(
    function: &str,
    input_type: &ArrayType,
    labels_type: &ArrayType,
    axes: &[usize],
) -> Result<(), ProgramError> {
    let expected_shape = input_type
        .shape()
        .dimensions()
        .iter()
        .enumerate()
        .filter(|(index, _)| !axes.contains(index))
        .map(|(_, dimension)| dimension.clone())
        .collect::<Vec<_>>();
    if labels_type.shape().dimensions() != expected_shape || !labels_type.data_type().is_integer() {
        return Err(TypeError::invalid(format!(
            "`{function}` labels must have the logits' shape with the class axes removed and integer elements",
        ))
        .into());
    }
    validate_normalization_placement(function, labels_type)
}

/// Reduces `targets` and precomputed `log_probabilities` over the single class `axis` to compute
/// [`Softmax::softmax_cross_entropy`]. With `mask`, log-probabilities must already be normalized over that mask as in
/// [`Softmax::masked_log_softmax`]; the loss follows [`Softmax::masked_softmax_cross_entropy`]. The caller supplies
/// validated, aligned inputs with targets and log-probabilities in working precision.
fn cross_entropy_loss<
    V: Value<Type = ArrayType, Domain: Fill<f64, V>> + ZeroLike + Neg + Mul + Compare + And + Select + Reduce,
>(
    targets: &V,
    log_probabilities: &V,
    mask: Option<&V>,
    axis: usize,
) -> Result<V, ProgramError> {
    let zero = targets.zero_like()?;
    let negative_infinity = log_probabilities.domain().fill(&log_probabilities.r#type(), f64::NEG_INFINITY)?;
    let excluded_zero = targets.equal(&zero)?.and(&log_probabilities.equal(&negative_infinity)?)?;
    let log_probabilities = excluded_zero.select(&zero, log_probabilities)?;
    let terms = targets.mul(&log_probabilities)?;
    let terms = match mask {
        Some(mask) => mask.select(&terms, &zero)?,
        None => terms,
    };
    terms.reduce_sum(&[axis], None)?.neg()
}

/// Selects each label's negative log-probability along `axis`, with the invalid-label
/// and empty-mask behavior of [`Softmax::softmax_cross_entropy_with_integer_labels`] and
/// [`Softmax::masked_softmax_cross_entropy_with_integer_labels`]. The caller supplies validated, aligned inputs,
/// `i64` labels, and log-probabilities computed in working precision over the same optional `mask`.
fn cross_entropy_loss_with_integer_labels<
    V: Value<Type = ArrayType, Domain: Fill<f64, V> + Iota<V>>
        + ZeroLike
        + Neg
        + Compare
        + Select
        + Broadcast
        + ConvertElementType
        + Reduce,
>(
    log_probabilities: &V,
    labels: &V,
    mask: Option<&V>,
    axis: usize,
) -> Result<V, ProgramError> {
    let label_mask = one_hot_mask(log_probabilities, labels, axis)?;
    let loss = label_mask
        .select(log_probabilities, &log_probabilities.zero_like()?)?
        .reduce_sum(&[axis], None)?
        .neg()?;
    let valid = label_mask
        .convert_element_type(DataType::I64)?
        .reduce_sum(&[axis], None)?
        .greater_than(&labels.zero_like()?)?;
    let nan = loss.domain().fill(&loss.r#type(), f64::NAN)?;
    let loss = valid.select(&loss, &nan)?;
    match mask {
        Some(mask) => mask.reduce_any(&[axis])?.select(&loss, &loss.zero_like()?),
        None => Ok(loss),
    }
}

/// Builds a Boolean mask with `input`'s shape, selecting the class named by `indices` in each row along `axis`.
/// The `indices` have `i64` elements and the input shape with `axis` removed; out-of-range indices select nothing.
/// The mask preserves the input's placement and manual variation. Indices must already vary over the input's manual
/// axes.
fn one_hot_mask<V: Value<Type = ArrayType, Domain: Iota<V>> + Compare + Broadcast>(
    input: &V,
    indices: &V,
    axis: usize,
) -> Result<V, ProgramError> {
    let index_type = input.r#type().into_owned().with_data_type(DataType::I64).with_layout(None);

    // Coordinates are non-differentiable constants, so they can carry the input's variation directly.
    // No skipped `parallel_vary` transpose is needed for these constants.
    let coordinates = input.domain().iota(&index_type, axis)?;
    let output_axes = (0..input.r#type().rank()).filter(|&index| index != axis).collect::<Vec<_>>();
    coordinates.equal(&indices.broadcast(index_type, &output_axes)?)
}

// TODO(eaplatanios): Review from here onwards.

/// Cross-entropy function created by [`softmax_cross_entropy_with_target_logits`].
///
/// Target logits parameterize a fixed distribution and are not differentiated. Prediction derivatives use the logits'
/// difference before rounding probabilities, preserving small gradients when the two distributions nearly match. A
/// retained custom JVP supplies both differentiation modes and remains differentiable at higher orders. Create the
/// function once and reuse it to share cached rule specializations across calls.
#[allow(clippy::type_complexity)]
pub struct SoftmaxCrossEntropyWithTargetLogits<D: Domain<Type = ArrayType>>
where
    DomainTracingContext<D>: StagingContext<Type = ArrayType, Constant = D::Constant, Operation = D::Operation>,
{
    /// Retained custom function whose leading target input is non-differentiated plumbing.
    function: CustomFunction<
        (DomainTracer<D>, DomainTracer<D>),
        DomainTracer<D>,
        Box<dyn Fn((DomainTracer<D>, DomainTracer<D>)) -> Result<DomainTracer<D>, ProgramError> + Send + Sync>,
        WithJvp<
            Box<
                dyn Fn(
                        (DomainTracer<D>, DomainTracer<D>),
                        (DomainTracer<D>, DomainTracer<D>),
                    ) -> Result<(DomainTracer<D>, DomainTracer<D>), ProgramError>
                    + Send
                    + Sync,
            >,
        >,
    >,
}

impl<D: Domain<Type = ArrayType>> SoftmaxCrossEntropyWithTargetLogits<D>
where
    D::Constant: 'static,
    D::Operation: 'static,
    DomainTracingContext<D>: StagingContext<Type = ArrayType, Constant = D::Constant, Operation = D::Operation>,
{
    /// Computes cross-entropy between the fixed distribution from `target_logits` and the predictions from `logits`,
    /// reducing the class axes. Both inputs must have the same real floating-point element type and static shape.
    /// Target derivatives are explicitly stopped, so differentiation over both inputs returns a zero target gradient.
    /// Manual variation is aligned before the custom function boundary, so its transpose sums replicated inputs'
    /// contributions once.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] for incompatible inputs, invalid, duplicate, or empty class axes, unsupported array
    /// metadata, or a failed operation while tracing or evaluating the primal or derivative rule.
    pub fn call<C, V>(&self, logits: V, target_logits: V) -> Result<V, ProgramError>
    where
        C: Context<Type = ArrayType, Value = V, Constant = D::Constant, Operation = D::Operation> + NamedAxes,
        C::Operation: From<CustomFunctionOperation<C::Constant, C::Operation>>,
        V: Value<Type = ArrayType, Domain = C> + ManualVariationAlignment + StopGradient,
        <ArrayType as BatchableType>::Policy: RecursiveBatchingPolicy<TracingContext<C::Constant, C::Operation>>
            + CotangentBatchingPolicy<TracingContext<C::Constant, C::Operation>>,
        CustomRuleRegistration<C::Constant, C::Operation>: Send,
        WithJvp<
            Box<
                dyn Fn(
                        (DomainTracer<D>, DomainTracer<D>),
                        (DomainTracer<D>, DomainTracer<D>),
                    ) -> Result<(DomainTracer<D>, DomainTracer<D>), ProgramError>
                    + Send
                    + Sync,
            >,
        >: CustomFunctionJvp<C::Constant, C::Operation, (DomainTracer<D>, DomainTracer<D>), DomainTracer<D>>,
    {
        // Placement is validated before the alignment below, which would otherwise report a `parallel_vary` error for
        // a reduced input instead of this function's own rejection. The primal validates the remaining contract.
        validate_normalization_placement("softmax_cross_entropy_with_target_logits", &logits.r#type())?;
        validate_normalization_placement("softmax_cross_entropy_with_target_logits", &target_logits.r#type())?;
        let target_logits = target_logits.stop_gradient()?;
        let mut inputs = V::align_manual_variation(&[target_logits, logits])?;
        let logits = inputs.pop().unwrap();
        let target_logits = inputs.pop().unwrap();
        self.function.call((target_logits, logits))
    }
}

/// Creates cross-entropy with a fixed target distribution supplied as logits rather than rounded probabilities.
///
/// Each call returns `-sum(softmax(target_logits) * log_softmax(logits), axes)` and removes the class axes,
/// which are normalized jointly as in [`Softmax`]. The inputs must have equal static shapes and real floating-point
/// element types; every class axis must be non0empty. Negative axes count from the end. Computation uses `f32` for
/// narrower inputs and restores the original output element type. Negative infinity can mask classes; NaN, positive
/// infinity, and rows containing only masked logits produce NaNs as in [`Softmax`].
///
/// Near matching distributions, the custom JVP computes the probability difference with `expm1` and `ln_1p` before
/// rounding either distribution to probabilities. Distant distributions use a direct difference with a correction
/// for its floating-point sum, so the gradient sums to zero up to the rounding of the target probabilities' own sum.
/// A class that both distributions mask contributes nothing to the loss or its derivatives and does not affect the
/// choice between the two formulas, so masked class sets keep the cancellation-resistant gradient; a class that
/// only one of them masks routes its row through the direct difference. Target logits are fixed, including under
/// higher-order differentiation. To differentiate probability targets themselves, use
/// [`Softmax::softmax_cross_entropy`] instead.
///
/// This implements the cancellation-resistant derivative in
/// [A Fix for Softmax and Cross-Entropy](https://jonbarron.info/softmaxfix/) as a differentiable JVP, so prediction
/// Hessians remain available. The returned function retains its rules and should be reused across calls; refer to
/// [`SoftmaxCrossEntropyWithTargetLogits`]. To embed its staged calls and retained rules in a composite value universe,
/// use [`Program::into_unprojected`](crate::programs::Program::into_unprojected).
///
/// # Example
///
/// ```rust
/// # use ryft_core::{Array, ArrayOperation, EagerContext, differentiate_at, softmax_cross_entropy_with_target_logits};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let loss = softmax_cross_entropy_with_target_logits::<EagerContext<Array, ArrayOperation<Array>>>(-1);
/// let target_logits = Array::vector(vec![2.0f32, 1.0, 0.0])?;
/// let gradient = differentiate_at(target_logits.clone()).with_captures(target_logits)
///     .gradient(|logits, target_logits| loss.call(logits, target_logits))?;
/// assert_eq!(gradient.elements::<f32>()?, vec![0.0f32, 0.0, 0.0]);
/// # Ok(())
/// # }
/// ```
pub fn softmax_cross_entropy_with_target_logits<D>(axes: impl Into<Axes>) -> SoftmaxCrossEntropyWithTargetLogits<D>
where
    D: Domain<Type = ArrayType>,
    D::Constant: 'static,
    D::Operation: 'static,
    DomainTracer<D>: Abs
        + And
        + Broadcast
        + Compare
        + ConvertElementType
        + Exp
        + ExpM1
        + Ln1p
        + Mul
        + Neg
        + Or
        + Reduce
        + Reshape
        + Select
        + Softmax
        + Sub
        + Transpose,
    DomainTracingContext<D>:
        StagingContext<Type = ArrayType, Constant = D::Constant, Operation = D::Operation> + Fill<f64, DomainTracer<D>>,
{
    let axes = axes.into();
    let primal_axes = axes.clone();
    let primal: Box<dyn Fn((DomainTracer<D>, DomainTracer<D>)) -> Result<DomainTracer<D>, ProgramError> + Send + Sync> =
        Box::new(move |(target_logits, logits)| {
            let inputs = CrossEntropyInputsWithTargetLogits::new(logits, target_logits, &primal_axes)?;
            soft_target_loss(
                &inputs.target_logits.softmax(inputs.axis)?,
                &inputs.logits.log_softmax(inputs.axis)?,
                inputs.axis,
            )?
            .convert_element_type(inputs.output_data_type)
        });
    let jvp: Box<
        dyn Fn(
                (DomainTracer<D>, DomainTracer<D>),
                (DomainTracer<D>, DomainTracer<D>),
            ) -> Result<(DomainTracer<D>, DomainTracer<D>), ProgramError>
            + Send
            + Sync,
    > = Box::new(move |(target_logits, logits), (_, tangent)| {
        let inputs = CrossEntropyInputsWithTargetLogits::new(logits, target_logits, &axes)?;
        let CrossEntropyInputsWithTargetLogits { target_logits, logits, class_axes, axis, output_data_type } = inputs;
        let log_probabilities = logits.log_softmax(axis)?;
        let probabilities = log_probabilities.exp()?;
        let target_probabilities = target_logits.softmax(axis)?;
        let loss = soft_target_loss(&target_probabilities, &log_probabilities, axis)?
            .convert_element_type(output_data_type)?;
        let gradient = soft_target_gradient(&target_logits, &logits, &target_probabilities, &probabilities, axis)?;
        let tangent = tangent.convert_element_type(logits.r#type().data_type())?;
        let mut flattened = FlattenedClassInputs::new(vec![tangent], &class_axes)?;
        let output_tangent = gradient
            .mul(&flattened.inputs.pop().unwrap())?
            .reduce_sum(&[axis], None)?
            .convert_element_type(loss.r#type().tangent()?.data_type())?;
        Ok((loss, output_tangent))
    });
    SoftmaxCrossEntropyWithTargetLogits {
        function: custom_function(primal)
            .with_jvp(jvp)
            .with_non_differentiated_count(1)
            .with_name("softmax_cross_entropy_with_target_logits"),
    }
}

/// Validated target and prediction logits in working precision, with flattened class axes and metadata for the loss
/// and its JVP.
struct CrossEntropyInputsWithTargetLogits<V> {
    /// Target logits in the working representation, with the class axes flattened.
    target_logits: V,

    /// Prediction logits in the working representation, with the class axes flattened.
    logits: V,

    /// Normalized class axes of the original inputs, in increasing order.
    class_axes: Vec<usize>,

    /// Position of the flattened class axis.
    axis: usize,

    /// Element type of the loss.
    output_data_type: DataType,
}

impl<V: Value<Type = ArrayType> + ConvertElementType + Reshape + Transpose> CrossEntropyInputsWithTargetLogits<V> {
    /// Validates `logits` and `target_logits`, widens them to working precision, and flattens their class `axes`.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] for incompatible logits, invalid class axes or placement, or a failed conversion,
    /// transpose, or reshape.
    fn new(logits: V, target_logits: V, axes: &Axes) -> Result<Self, ProgramError> {
        let logits_type = logits.r#type();
        let target_logits_type = target_logits.r#type();
        if logits_type.shape() != target_logits_type.shape()
            || logits_type.data_type() != target_logits_type.data_type()
        {
            return Err(TypeError::invalid(
                "`softmax_cross_entropy_with_target_logits` requires logits and target logits with the same shape and element type",
            )
            .into());
        }
        let (class_axes, working_data_type) =
            validate_normalization("softmax_cross_entropy_with_target_logits", &logits_type, axes.clone())?;
        validate_normalization_placement("softmax_cross_entropy_with_target_logits", &target_logits_type)?;
        let output_data_type = logits_type.data_type();
        let mut flattened = FlattenedClassInputs::new(
            vec![
                target_logits.convert_element_type(working_data_type)?,
                logits.convert_element_type(working_data_type)?,
            ],
            &class_axes,
        )?;
        let logits = flattened.inputs.pop().unwrap();
        let target_logits = flattened.inputs.pop().unwrap();
        Ok(Self { target_logits, logits, class_axes, axis: flattened.axis, output_data_type })
    }
}

/// Computes the primal loss without multiplying zero target probabilities by masked log-probabilities.
fn soft_target_loss<V>(teacher_probabilities: &V, log_probabilities: &V, axis: usize) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + Compare + Mul + Neg + Reduce + Select,
    V::Domain: Fill<f64, V>,
{
    // This literal has the logits' placement and variation because it is a non-differentiable constant.
    let zero = teacher_probabilities.domain().fill(teacher_probabilities.r#type().as_ref(), 0.0)?;
    let log_probabilities = teacher_probabilities.equal(&zero)?.select(&zero, log_probabilities)?;
    teacher_probabilities.mul(&log_probabilities)?.reduce_sum(&[axis], None)?.neg()
}

/// Computes the stable prediction probability difference, retaining differentiable dependence on the prediction logits.
fn soft_target_gradient<V>(
    teacher: &V,
    student: &V,
    teacher_probabilities: &V,
    student_probabilities: &V,
    axis: usize,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + Abs + And + Broadcast + Compare + ExpM1 + Ln1p + Mul + Or + Reduce + Select + Sub,
    V::Domain: Fill<f64, V>,
{
    let input_type = student.r#type().into_owned();
    let broadcast_axes = (0..input_type.rank()).filter(|&dimension| dimension != axis).collect::<Vec<_>>();
    // Broadcast outputs take the default layout, like the class coordinates, because the logits' layout need not apply
    // to another element type.
    let broadcast = |value: V| {
        let output_type = input_type.clone().with_data_type(value.r#type().data_type()).with_layout(None);
        value.broadcast(output_type, &broadcast_axes)
    };
    // Literals inherit the inputs' placement and variation; none can carry a tangent.
    let zero = student.domain().fill(&input_type, 0.0)?;
    let one = student.domain().fill(&input_type, 1.0)?;
    let infinity = student.domain().fill(&input_type, f64::INFINITY)?;
    let negative_infinity = student.domain().fill(&input_type, f64::NEG_INFINITY)?;
    // A class that both distributions mask has zero probability on both sides, so it contributes nothing to the
    // gradient. Its raw difference is `-∞ - -∞ = NaN`, which must neither take part in the arithmetic nor decide
    // which formula the row uses.
    let excluded = teacher.equal(&negative_infinity)?.and(&student.equal(&negative_infinity)?)?;
    let difference = student.sub(teacher)?;
    let finite_difference = difference.abs()?.less_than(&infinity)?;
    // Remove a common offset exactly before weighted centering. Rounded probabilities need not sum to exactly one,
    // so weighting a constant difference directly could otherwise leave a tiny nonzero gradient for equal softmaxes.
    // Only finite differences take part in the offset; a row without any keeps a zero offset, which keeps the
    // discarded branch finite.
    let offset = finite_difference.select(&difference, &negative_infinity)?.reduce_max(&[axis])?;
    let offset_type = offset.r#type().into_owned();
    let finite_offset = offset.abs()?.less_than(&student.domain().fill(&offset_type, f64::INFINITY)?)?;
    let offset = broadcast(finite_offset.select(&offset, &student.domain().fill(&offset_type, 0.0)?)?)?;
    let difference = finite_difference.select(&difference.sub(&offset)?, &zero)?;
    let centered = difference.sub(&broadcast(teacher_probabilities.mul(&difference)?.reduce_sum(&[axis], None)?)?)?;

    // Only close rows need the cancellation-resistant formula. Sanitize its input before expm1, rather than selecting
    // an overflowing result afterward: reverse-mode derivatives of an unused infinite branch can otherwise be NaN.
    // The all-finite test routes a class that only one distribution masks through the direct probability difference,
    // while a class that both mask is excluded from the test. Masking the raw differences before reduce_max keeps the
    // discarded branch's higher derivatives finite as well.
    let close = excluded.or(&finite_difference.and(&centered.abs()?.less_than(&one)?)?)?.reduce_all(&[axis])?;
    let close = broadcast(close)?;
    let centered = close.select(&centered, &zero)?;
    let correction = broadcast(teacher_probabilities.mul(&centered.exp_m1()?)?.reduce_sum(&[axis], None)?.ln_1p()?)?;
    let log_ratio = centered.sub(&correction)?;
    let near_difference = teacher_probabilities.mul(&log_ratio.exp_m1()?)?;
    let direct_difference = student_probabilities.sub(teacher_probabilities)?;
    let near = close.and(&log_ratio.abs()?.less_than(&one)?)?;
    let difference = near.select(&near_difference, &direct_difference)?;
    let residual = broadcast(difference.reduce_sum(&[axis], None)?)?;
    difference.sub(&student_probabilities.mul(&residual)?)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, Dimension, DimensionBounds, DimensionVariable,
        LogicalMesh, MeshAxis, MeshAxisType, Shape, Sharding, ShardingDimension, bf16,
    };
    use crate::axes::NamedAxis;
    use crate::batching::{BatchAxis, batch};
    use crate::contexts::{EagerContext, ProjectedContext};
    use crate::differentiation::differentiate_at;
    use crate::macros::check_gradient;
    use crate::operations::manipulation::reshaping::Reshape;
    use crate::operations::manipulation::slicing::Slice;
    use crate::operations::sharding::shard_map::{ShardMapTracer, shard_map};
    use crate::partial::{PartialEvaluationOutput, PartialValue};
    use crate::programs::ValueProjection;
    use crate::tracing::{Trace, Tracer};

    use super::*;

    /// Eager array domain used to trace, partially evaluate, and execute the staged programs.
    type ArrayContext = EagerContext<Array, ArrayOperation<Array>>;

    /// Eager composite array domain, whose `shard_map` invocations the reference emulation executes device by device.
    type ArrayIrContext = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Tracing context of [`ArrayIrContext`], whose array projections are the input leaves of `shard_map`.
    type ArrayIrTracingContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Returns the array held by a composite value.
    fn array(value: ArrayIrValue<Array>) -> Array {
        let ArrayIrValue::Array(array) = value else { panic!("expected an array") };
        array
    }

    /// Returns a mesh with one manual axis `x` of size two, with the varying and invariant versions of `type` on it.
    fn manual_types(r#type: ArrayType) -> (LogicalMesh, ArrayType, ArrayType) {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let rank = r#type.rank();
        let invariant = r#type.clone().with_sharding(Sharding::replicated(mesh.clone(), rank)).unwrap();
        let varying = r#type
            .with_sharding(Sharding::replicated(mesh.clone(), rank).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        (mesh, varying, invariant)
    }

    #[test]
    fn test_softmax() {
        let input = Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap();
        let actual = input.softmax(-1).unwrap().elements::<f64>().unwrap();
        let denominator = 1.0 + (-1.0f64).exp() + (-2.0f64).exp();
        assert_abs_diff_eq!(actual[0], (-2.0f64).exp() / denominator, epsilon = 1e-15);
        assert_abs_diff_eq!(actual[1], (-1.0f64).exp() / denominator, epsilon = 1e-15);
        assert_abs_diff_eq!(actual[2], 1.0 / denominator, epsilon = 1e-15);
        assert_eq!(
            Array::scalar(1.0f32).unwrap().reshape([1]).unwrap().softmax(0),
            Ok(Array::vector(vec![1.0f32]).unwrap())
        );
        assert_eq!(
            Array::matrix(0, 2, Vec::<f32>::new()).unwrap().softmax(1),
            Ok(Array::matrix(0, 2, Vec::<f32>::new()).unwrap())
        );
        // The composite universe projects to the same contract, including the singleton and empty edge cases.
        let composite = ArrayIrValue::Array(input.clone());
        assert_eq!(composite.softmax(0), Ok(ArrayIrValue::Array(input.softmax(0).unwrap())));
        assert_eq!(
            ArrayIrValue::Array(Array::vector(vec![1.0f32]).unwrap()).softmax(0),
            Ok(ArrayIrValue::Array(Array::vector(vec![1.0f32]).unwrap()))
        );
        assert_eq!(
            ArrayIrValue::Array(Array::matrix(0, 2, Vec::<f32>::new()).unwrap()).softmax(1),
            Ok(ArrayIrValue::Array(Array::matrix(0, 2, Vec::<f32>::new()).unwrap()))
        );
        assert_eq!(
            ArrayIrValue::Array(Array::vector(Vec::<f32>::new()).unwrap()).softmax(0),
            Err(ProgramError::Type(TypeError::invalid("`softmax` requires a non-empty class axis")))
        );
    }

    #[test]
    fn test_softmax_multiple_axes() {
        // Several class axes normalize jointly: every group of elements that share the remaining coordinates sums to
        // one, and normalizing over every axis sums the whole array to one.
        let values = (0..12).map(|index| (index as f64) * 0.3 - 1.0).collect::<Vec<_>>();
        let input = Array::from_elements(ArrayType::new_static(DataType::F64, [2, 3, 2]), &values).unwrap();
        let output = input.softmax([0, 2]).unwrap();
        assert_eq!(output.r#type().into_owned(), ArrayType::new_static(DataType::F64, [2, 3, 2]));
        let output = output.to_f64s();
        for j in 0..3 {
            let group = [(0, 0), (0, 1), (1, 0), (1, 1)].map(|(i, k)| (i * 3 + j) * 2 + k);
            let denominator = group.iter().map(|&index| values[index].exp()).sum::<f64>();
            for index in group {
                assert_abs_diff_eq!(output[index], values[index].exp() / denominator, epsilon = 1e-15);
            }
        }
        assert_abs_diff_eq!(input.softmax([0, 1, 2]).unwrap().to_f64s().iter().sum::<f64>(), 1.0, epsilon = 1e-15);
        assert_eq!(input.softmax([-1, -3]), input.softmax([0, 2]));
        assert_eq!(input.softmax(vec![2usize]), input.softmax(2));
    }

    #[test]
    fn test_softmax_validation() {
        assert_eq!(
            Array::vector(vec![1i32]).unwrap().softmax(0),
            Err(ProgramError::Type(TypeError::invalid("`softmax` requires real floating-point logits")))
        );
        assert_eq!(
            Array::vector(Vec::<f32>::new()).unwrap().softmax(0),
            Err(ProgramError::Type(TypeError::invalid("`softmax` requires a non-empty class axis")))
        );
        assert_eq!(
            Array::vector(vec![1.0f32]).unwrap().softmax(1),
            Err(ProgramError::Type(TypeError::invalid("`softmax` axis 1 is out of bounds for rank 1")))
        );
        let dynamic = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("n", DimensionBounds::unbounded()))]),
        );
        assert_eq!(
            ArrayContext::trace(|input| input.softmax(0), dynamic).map(|(output_type, _)| output_type),
            Err(ProgramError::Type(TypeError::invalid("`softmax` requires a static input shape")))
        );
        let (mesh, _, _) = manual_types(ArrayType::new_static(DataType::F32, [2]));
        let reduced = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh, 1).with_reduced_axes(["x"]).unwrap())
            .unwrap();
        assert_eq!(
            ArrayContext::trace(|input| input.softmax(0), reduced).map(|(output_type, _)| output_type),
            Err(ProgramError::Type(TypeError::invalid("`softmax` does not support unreduced or reduced inputs"))),
        );
        let input = Array::matrix(2, 2, vec![0.0f32; 4]).unwrap();
        assert_eq!(
            input.softmax([0, 0]),
            Err(ProgramError::Type(TypeError::invalid("`softmax` axes contain duplicate axis 0")))
        );
        assert_eq!(
            input.softmax(Vec::<usize>::new()),
            Err(ProgramError::Type(TypeError::invalid("`softmax` requires at least one class axis")))
        );
        assert_eq!(
            Array::matrix(2, 0, Vec::<f32>::new()).unwrap().softmax([0, 1]),
            Err(ProgramError::Type(TypeError::invalid("`softmax` requires a non-empty class axis")))
        );
    }

    #[test]
    fn test_softmax_staging() {
        // Narrow formats widen to `f32` before the normalization and narrow its exponential only at the end.
        let (_, program) =
            ArrayContext::trace(|input| input.softmax(0), ArrayType::new_static(DataType::BF16, [2])).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:bf16[2] .
            let %1:f32[2] = convert_element_type [data_type=f32] %0
                %2:i64[] = argmax [axis=0, index_data_type=i64] %1
                %3:i64[2] = iota [type=i64[2], dimension=0]
                %4:i64[2] = broadcast [output_type=i64[2], output_axes=[]] %2
                %5:bool[2] = compare [direction=Equal] %3 %4
                %6:f32[2] = zero_like %1
                %7:f32[2] = select %5 %1 %6
                %8:f32[] = reduce [kind=sum, axes=[0]] %7
                %9:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %8
                %10:f32[2] = sub %1 %9
                %11:f32[2] = select %5 %6 %10
                %12:f32[2] = exp %11
                %13:f32[2] = select %5 %6 %12
                %14:f32[] = reduce [kind=sum, axes=[0]] %13
                %15:f32[] = ln_1p %14
                %16:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %15
                %17:f32[2] = sub %11 %16
                %18:f32[2] = sub %9 %9
                %19:bool[2] = compare [direction=Equal] %18 %6
                %20:f32[] = constant [value=NaN]
                %21:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %20
                %22:f32[2] = select %19 %17 %21
                %23:f32[2] = exp %22
                %24:bf16[2] = convert_element_type [data_type=bf16] %23
            in (%24)
        "}
            .trim_end()
        );
        let input = Array::vector(vec![bf16::from_f32(4.0), bf16::from_f32(-4.0)]).unwrap();
        assert_eq!(program.interpret(input.clone()), input.softmax(0));
    }

    #[test]
    fn test_softmax_staging_sharded_class_axis() {
        // A class axis sharded over an explicit mesh axis stays sharded in the output: the argmax and the sums drop
        // the sharded axis and leave the cross-shard combination to the backend partitioner, and the broadcasts and
        // coordinates restore the input's placement.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("d", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["d"])]).unwrap())
            .unwrap();
        let (output_type, program) = ArrayContext::trace(|input| input.softmax(0), sharded.clone()).unwrap();
        assert_eq!(output_type, sharded);
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:f32[4][sharding={mesh<['d'=2:explicit]>, [{'d'}]}] .
            let %1:i64[][sharding={mesh<['d'=2:explicit]>, []}] = argmax [axis=0, index_data_type=i64] %0
                %2:i64[4][sharding={mesh<['d'=2:explicit]>, [{'d'}]}] = iota [type=i64[4][sharding={mesh<['d'=2:explicit]>, [{'d'}]}], dimension=0]
                %3:i64[4][sharding={mesh<['d'=2:explicit]>, [{'d'}]}] = broadcast [output_type=i64[4][sharding={mesh<['d'=2:explicit]>, [{'d'}]}], output_axes=[]] %1
                %4:bool[4][sharding={mesh<['d'=2:explicit]>, [{'d'}]}] = compare [direction=Equal] %2 %3
                %5:f32[4][sharding={mesh<['d'=2:explicit]>, [{'d'}]}] = zero_like %0
                %6:f32[4][sharding={mesh<['d'=2:explicit]>, [{'d'}]}] = select %4 %0 %5
                %7:f32[][sharding={mesh<['d'=2:explicit]>, []}] = reduce [kind=sum, axes=[0]] %6
                %8:f32[4][sharding={mesh<['d'=2:explicit]>, [{'d'}]}] = broadcast [output_type=f32[4][sharding={mesh<['d'=2:explicit]>, [{'d'}]}], output_axes=[]] %7
                %9:f32[4][sharding={mesh<['d'=2:explicit]>, [{'d'}]}] = sub %0 %8
                %10:f32[4][sharding={mesh<['d'=2:explicit]>, [{'d'}]}] = select %4 %5 %9
                %11:f32[4][sharding={mesh<['d'=2:explicit]>, [{'d'}]}] = exp %10
                %12:f32[4][sharding={mesh<['d'=2:explicit]>, [{'d'}]}] = select %4 %5 %11
                %13:f32[][sharding={mesh<['d'=2:explicit]>, []}] = reduce [kind=sum, axes=[0]] %12
                %14:f32[][sharding={mesh<['d'=2:explicit]>, []}] = ln_1p %13
                %15:f32[4][sharding={mesh<['d'=2:explicit]>, [{'d'}]}] = broadcast [output_type=f32[4][sharding={mesh<['d'=2:explicit]>, [{'d'}]}], output_axes=[]] %14
                %16:f32[4][sharding={mesh<['d'=2:explicit]>, [{'d'}]}] = sub %10 %15
                %17:f32[4][sharding={mesh<['d'=2:explicit]>, [{'d'}]}] = sub %8 %8
                %18:bool[4][sharding={mesh<['d'=2:explicit]>, [{'d'}]}] = compare [direction=Equal] %17 %5
                %19:f32[][sharding={mesh<['d'=2:explicit]>, []}] = constant [value=NaN]
                %20:f32[4][sharding={mesh<['d'=2:explicit]>, [{'d'}]}] = broadcast [output_type=f32[4][sharding={mesh<['d'=2:explicit]>, [{'d'}]}], output_axes=[]] %19
                %21:f32[4][sharding={mesh<['d'=2:explicit]>, [{'d'}]}] = select %18 %16 %20
                %22:f32[4][sharding={mesh<['d'=2:explicit]>, [{'d'}]}] = exp %21
            in (%22)
        "}
            .trim_end()
        );
    }

    #[test]
    fn test_softmax_staging_multiple_axes() {
        // Several class axes move to the end and merge into one class axis for the single-axis arithmetic, and the
        // output restores the input layout.
        let (_, program) =
            ArrayContext::trace(|input| input.softmax([0, 2]), ArrayType::new_static(DataType::F32, [2, 3, 2]))
                .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:f32[2, 3, 2] .
            let %1:f32[3, 2, 2] = transpose [permutation=[1, 0, 2]] %0
                %2:f32[3, 4] = reshape [shape=[3, 4]] %1
                %3:i64[3] = argmax [axis=1, index_data_type=i64] %2
                %4:i64[3, 4] = iota [type=i64[3, 4], dimension=1]
                %5:i64[3, 4] = broadcast [output_type=i64[3, 4], output_axes=[0]] %3
                %6:bool[3, 4] = compare [direction=Equal] %4 %5
                %7:f32[3, 4] = zero_like %2
                %8:f32[3, 4] = select %6 %2 %7
                %9:f32[3] = reduce [kind=sum, axes=[1]] %8
                %10:f32[3, 4] = broadcast [output_type=f32[3, 4], output_axes=[0]] %9
                %11:f32[3, 4] = sub %2 %10
                %12:f32[3, 4] = select %6 %7 %11
                %13:f32[3, 4] = exp %12
                %14:f32[3, 4] = select %6 %7 %13
                %15:f32[3] = reduce [kind=sum, axes=[1]] %14
                %16:f32[3] = ln_1p %15
                %17:f32[3, 4] = broadcast [output_type=f32[3, 4], output_axes=[0]] %16
                %18:f32[3, 4] = sub %12 %17
                %19:f32[3, 4] = sub %10 %10
                %20:bool[3, 4] = compare [direction=Equal] %19 %7
                %21:f32[] = constant [value=NaN]
                %22:f32[3, 4] = broadcast [output_type=f32[3, 4], output_axes=[]] %21
                %23:f32[3, 4] = select %20 %18 %22
                %24:f32[3, 2, 2] = reshape [shape=[3, 2, 2]] %23
                %25:f32[2, 3, 2] = transpose [permutation=[1, 0, 2]] %24
                %26:f32[2, 3, 2] = exp %25
            in (%26)
        "}
            .trim_end()
        );
        let values = (0..12).map(|index| (index as f32) * 0.3 - 1.0).collect::<Vec<_>>();
        let input = Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3, 2]), &values).unwrap();
        assert_eq!(program.interpret(input.clone()), input.softmax([0, 2]));
    }

    #[test]
    fn test_softmax_manual_variation() {
        let (_, varying, _) = manual_types(ArrayType::new_static(DataType::F32, [2]));
        let input = Array::from_elements(varying.clone(), &[0.0f32, 0.0]).unwrap();
        assert_eq!(input.softmax(0), Ok(Array::from_elements(varying, &[0.5f32, 0.5]).unwrap()));
    }

    #[test]
    fn test_softmax_partial_evaluation() {
        let (_, program) = ArrayContext::trace(
            |inputs: Vec<_>| inputs[0].softmax(0).map(|output| vec![output]),
            vec![ArrayType::new_static(DataType::F64, [3])],
        )
        .unwrap();
        let input = Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap();
        let known = program.partially_evaluate(&[PartialValue::Known(input.clone())]).unwrap();
        assert_eq!(known.program().to_string(), "lambda  .\nin ()");
        assert_eq!(known.outputs(), &[PartialEvaluationOutput::Known(input.softmax(0).unwrap())]);
        // With unknown logits, the class coordinates and the NaN fill fold into known residual inputs.
        let unknown = program.partially_evaluate(&[PartialValue::Unknown(input.r#type().into_owned())]).unwrap();
        assert_eq!(unknown.outputs(), &[PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(
            unknown.program().to_string(),
            indoc! {"
            lambda %0:f64[3], %1:i64[3], %2:f64[3] .
            let %3:i64[] = argmax [axis=0, index_data_type=i64] %0
                %4:i64[3] = broadcast [output_type=i64[3], output_axes=[]] %3
                %5:bool[3] = compare [direction=Equal] %1 %4
                %6:f64[3] = zero_like %0
                %7:f64[3] = select %5 %0 %6
                %8:f64[] = reduce [kind=sum, axes=[0]] %7
                %9:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %8
                %10:f64[3] = sub %9 %9
                %11:bool[3] = compare [direction=Equal] %10 %6
                %12:f64[3] = sub %0 %9
                %13:f64[3] = select %5 %6 %12
                %14:f64[3] = exp %13
                %15:f64[3] = select %5 %6 %14
                %16:f64[] = reduce [kind=sum, axes=[0]] %15
                %17:f64[] = ln_1p %16
                %18:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %17
                %19:f64[3] = sub %13 %18
                %20:f64[3] = select %11 %19 %2
                %21:f64[3] = exp %20
            in (%21)
        "}
            .trim_end()
        );
        assert_eq!(unknown.interpret(&EagerContext::new(), &[input.clone()]), Ok(vec![input.softmax(0).unwrap()]));
    }

    #[test]
    fn test_softmax_batching() {
        let input = Array::matrix(2, 2, vec![0.0f64, 0.0, 0.0, -20.0]).unwrap();
        let output = batch(|input| input.softmax(0), input.clone(), BatchAxis::new(0), BatchAxis::new(0), 2).unwrap();
        assert_eq!(output, input.softmax(1).unwrap());
    }

    #[test]
    fn test_softmax_differentiation() {
        check_gradient!(
            |input, weights| input.softmax(0)?.mul(&weights)?.reduce_sum(&[0], None),
            at = Array::vector(vec![-0.2f64, 0.5, 1.2]).unwrap(),
            with = Array::vector(vec![0.3f64, -1.0, 2.0]).unwrap(),
            step = 1e-5,
            tolerance = 1e-8,
        );
    }

    #[test]
    fn test_softmax_differentiation_saturation() {
        let input = Array::vector(vec![9.0f32, -9.0, -9.0]).unwrap();
        let (output, gradient) = differentiate_at(input)
            .value_and_gradient(|input| input.softmax(0)?.slice_axis(0, 0, 1, 1)?.reshape([]))
            .unwrap();
        let gradient = gradient.elements::<f32>().unwrap();
        let expected = (-(2.0 * (-18.0f64).exp()).ln_1p()).exp() as f32;
        assert_eq!(output.elements::<f32>(), Ok(vec![expected]));
        assert!(gradient[0] > 3.0e-8 && gradient[0] < 3.1e-8);
        assert_eq!(gradient[0], -(gradient[1] + gradient[2]));
    }

    #[test]
    fn test_softmax_differentiation_ties() {
        let input = Array::vector(vec![0.0f64, 0.0, 0.0]).unwrap();
        let (_, tangent) = differentiate_at(input)
            .jvp(Array::vector(vec![1.0f64, 0.0, 0.0]).unwrap(), |input| input.softmax(0))
            .unwrap();
        let tangent = tangent.elements::<f64>().unwrap();
        assert_abs_diff_eq!(tangent[0], 2.0 / 9.0, epsilon = 1e-15);
        assert_abs_diff_eq!(tangent[1], -1.0 / 9.0, epsilon = 1e-15);
        assert_abs_diff_eq!(tangent[2], -1.0 / 9.0, epsilon = 1e-15);
    }

    #[test]
    fn test_masked_softmax() {
        // Unselected classes have probability zero and do not take part in the normalization of the others, whatever
        // their logits hold; a row without any selected class is all zeros.
        let logits = Array::vector(vec![1.0f64, f64::NAN, 3.0]).unwrap();
        let mask = Array::vector(vec![true, false, true]).unwrap();
        let output = logits.masked_softmax(&mask, 0).unwrap().to_f64s();
        let denominator = 1.0 + (-2.0f64).exp();
        assert_abs_diff_eq!(output[0], (-2.0f64).exp() / denominator, epsilon = 1e-15);
        assert_eq!(output[1], 0.0);
        assert_abs_diff_eq!(output[2], 1.0 / denominator, epsilon = 1e-15);
        let unmasked = Array::vector(vec![1.0f64, 3.0]).unwrap().softmax(0).unwrap().to_f64s();
        assert_eq!([output[0], output[2]], [unmasked[0], unmasked[1]]);
        assert_eq!(
            Array::vector(vec![1.0f32, 2.0])
                .unwrap()
                .masked_softmax(&Array::vector(vec![false, false]).unwrap(), 0),
            Ok(Array::vector(vec![0.0f32, 0.0]).unwrap())
        );
        assert_eq!(
            Array::vector(vec![f32::NEG_INFINITY, 2.0])
                .unwrap()
                .masked_softmax(&Array::vector(vec![true, true]).unwrap(), 0),
            Ok(Array::vector(vec![0.0f32, 1.0]).unwrap())
        );
        let composite = ArrayIrValue::Array(logits.clone());
        assert_eq!(
            composite.masked_softmax(&ArrayIrValue::Array(mask.clone()), 0),
            Ok(ArrayIrValue::Array(logits.masked_softmax(&mask, 0).unwrap()))
        );
    }

    #[test]
    fn test_masked_softmax_validation() {
        let logits = Array::vector(vec![0.0f32, 0.0]).unwrap();
        let expected = Err(ProgramError::Type(TypeError::invalid(
            "`masked_softmax` mask must be a Boolean array with the logits' shape",
        )));
        assert_eq!(logits.masked_softmax(&Array::vector(vec![true]).unwrap(), 0), expected);
        assert_eq!(logits.masked_softmax(&Array::vector(vec![1i32, 0]).unwrap(), 0), expected);
        assert_eq!(
            Array::vector(vec![1i32, 0]).unwrap().masked_softmax(&Array::vector(vec![true, true]).unwrap(), 0),
            Err(ProgramError::Type(TypeError::invalid("`masked_softmax` requires real floating-point logits")))
        );
    }

    #[test]
    fn test_masked_softmax_differentiation() {
        check_gradient!(
            |logits, mask| logits.masked_softmax(&mask, 0)?.slice_axis(0, 0, 1, 1)?.reshape([]),
            at = Array::vector(vec![-0.2f64, 0.5, 1.2]).unwrap(),
            with = Array::vector(vec![true, false, true]).unwrap(),
            step = 1e-5,
            tolerance = 1e-8,
        );
        // Unselected classes receive zero derivatives, and the selected ones the derivatives of their own softmax.
        let logits = Array::vector(vec![1.0f64, 7.0, 3.0]).unwrap();
        let mask = Array::vector(vec![true, false, true]).unwrap();
        let (_, gradient) = differentiate_at(logits)
            .with_captures(mask)
            .value_and_gradient(|logits, mask| logits.masked_softmax(&mask, 0)?.slice_axis(0, 0, 1, 1)?.reshape([]))
            .unwrap();
        let probability = (-2.0f64).exp() / (1.0 + (-2.0f64).exp());
        let gradient = gradient.to_f64s();
        assert_abs_diff_eq!(gradient[0], probability * (1.0 - probability), epsilon = 1e-15);
        assert_eq!(gradient[1], 0.0);
        assert_abs_diff_eq!(gradient[2], -probability * (1.0 - probability), epsilon = 1e-15);
    }

    #[test]
    fn test_log_softmax() {
        let logits = Array::vector(vec![9.0f32, -9.0, -9.0]).unwrap();
        let output = logits.log_softmax(0).unwrap().elements::<f32>().unwrap();
        assert!(output[0] < -3.0e-8 && output[0] > -3.1e-8);
        assert_eq!(&output[1..], &[-18.0, -18.0]);
        let output = Array::vector(vec![bf16::from_f32(4.0), bf16::from_f32(-4.0)])
            .unwrap()
            .log_softmax(0)
            .unwrap()
            .elements::<bf16>()
            .unwrap();
        assert!(output[0].to_f32() < 0.0);
        let composite = ArrayIrValue::Array(logits.clone());
        assert_eq!(composite.log_softmax(0), Ok(ArrayIrValue::Array(logits.log_softmax(0).unwrap())));
        assert_eq!(
            ArrayIrValue::Array(Array::vector(vec![f32::NEG_INFINITY, 0.0]).unwrap()).log_softmax(0),
            Ok(ArrayIrValue::Array(Array::vector(vec![f32::NEG_INFINITY, 0.0]).unwrap()))
        );
        let ArrayIrValue::Array(output) =
            ArrayIrValue::Array(Array::vector(vec![f32::NAN, 0.0]).unwrap()).log_softmax(0).unwrap()
        else {
            panic!("expected an array")
        };
        assert!(output.to_f64s().iter().all(|value| value.is_nan()));
    }

    #[test]
    fn test_log_softmax_validation() {
        assert_eq!(
            Array::vector(vec![1.0f32]).unwrap().log_softmax(-2),
            Err(ProgramError::Type(TypeError::invalid("`log_softmax` axis -2 is out of bounds for rank 1")))
        );
        assert_eq!(
            Array::vector(vec![true]).unwrap().log_softmax(0),
            Err(ProgramError::Type(TypeError::invalid("`log_softmax` requires real floating-point logits")))
        );
        assert_eq!(
            Array::matrix(2, 0, Vec::<f64>::new()).unwrap().log_softmax(1),
            Err(ProgramError::Type(TypeError::invalid("`log_softmax` requires a non-empty class axis")))
        );
    }

    #[test]
    fn test_log_softmax_non_finite() {
        let input = Array::vector(vec![f32::NEG_INFINITY, 0.0]).unwrap();
        assert_eq!(input.log_softmax(0), Ok(Array::vector(vec![f32::NEG_INFINITY, 0.0]).unwrap()));
        assert_eq!(input.softmax(0), Ok(Array::vector(vec![0.0f32, 1.0]).unwrap()));
        for input in [
            vec![f32::NAN, 0.0],
            vec![f32::INFINITY, 0.0],
            vec![f32::NEG_INFINITY, f32::NEG_INFINITY],
            vec![f32::INFINITY],
        ] {
            assert!(
                Array::vector(input)
                    .unwrap()
                    .log_softmax(0)
                    .unwrap()
                    .elements::<f32>()
                    .unwrap()
                    .iter()
                    .all(|value| value.is_nan())
            );
        }
    }

    #[test]
    fn test_log_softmax_staging() {
        let (_, program) =
            ArrayContext::trace(|input| input.log_softmax(0), ArrayType::new_static(DataType::F32, [2])).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:f32[2] .
            let %1:i64[] = argmax [axis=0, index_data_type=i64] %0
                %2:i64[2] = iota [type=i64[2], dimension=0]
                %3:i64[2] = broadcast [output_type=i64[2], output_axes=[]] %1
                %4:bool[2] = compare [direction=Equal] %2 %3
                %5:f32[2] = zero_like %0
                %6:f32[2] = select %4 %0 %5
                %7:f32[] = reduce [kind=sum, axes=[0]] %6
                %8:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %7
                %9:f32[2] = sub %0 %8
                %10:f32[2] = select %4 %5 %9
                %11:f32[2] = exp %10
                %12:f32[2] = select %4 %5 %11
                %13:f32[] = reduce [kind=sum, axes=[0]] %12
                %14:f32[] = ln_1p %13
                %15:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %14
                %16:f32[2] = sub %10 %15
                %17:f32[2] = sub %8 %8
                %18:bool[2] = compare [direction=Equal] %17 %5
                %19:f32[] = constant [value=NaN]
                %20:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %19
                %21:f32[2] = select %18 %16 %20
            in (%21)
        "}
            .trim_end()
        );
        let input = Array::vector(vec![9.0f32, -9.0]).unwrap();
        assert_eq!(program.interpret(input.clone()), input.log_softmax(0));
    }

    #[test]
    fn test_log_softmax_partial_evaluation() {
        let (_, program) = ArrayContext::trace(
            |inputs: Vec<_>| inputs[0].log_softmax(0).map(|output| vec![output]),
            vec![ArrayType::new_static(DataType::F32, [2])],
        )
        .unwrap();
        let input = Array::vector(vec![9.0f32, -9.0]).unwrap();
        let known = program.partially_evaluate(&[PartialValue::Known(input.clone())]).unwrap();
        assert_eq!(known.program().to_string(), "lambda  .\nin ()");
        assert_eq!(known.outputs(), &[PartialEvaluationOutput::Known(input.log_softmax(0).unwrap())]);
        // With unknown logits, only the constants fold: the class coordinates and the NaN fill become known residual
        // inputs of the residual program.
        let unknown = program.partially_evaluate(&[PartialValue::Unknown(input.r#type().into_owned())]).unwrap();
        assert_eq!(
            unknown.program().to_string(),
            indoc! {"
            lambda %0:f32[2], %1:i64[2], %2:f32[2] .
            let %3:i64[] = argmax [axis=0, index_data_type=i64] %0
                %4:i64[2] = broadcast [output_type=i64[2], output_axes=[]] %3
                %5:bool[2] = compare [direction=Equal] %1 %4
                %6:f32[2] = zero_like %0
                %7:f32[2] = select %5 %0 %6
                %8:f32[] = reduce [kind=sum, axes=[0]] %7
                %9:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %8
                %10:f32[2] = sub %9 %9
                %11:bool[2] = compare [direction=Equal] %10 %6
                %12:f32[2] = sub %0 %9
                %13:f32[2] = select %5 %6 %12
                %14:f32[2] = exp %13
                %15:f32[2] = select %5 %6 %14
                %16:f32[] = reduce [kind=sum, axes=[0]] %15
                %17:f32[] = ln_1p %16
                %18:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %17
                %19:f32[2] = sub %13 %18
                %20:f32[2] = select %11 %19 %2
            in (%20)
        "}
            .trim_end()
        );
        assert_eq!(unknown.interpret(&EagerContext::new(), &[input.clone()]), Ok(vec![input.log_softmax(0).unwrap()]));
    }

    #[test]
    fn test_log_softmax_batching() {
        let input = Array::matrix(2, 2, vec![0.0f64, 0.0, 0.0, -20.0]).unwrap();
        let output =
            batch(|input| input.log_softmax(0), input.clone(), BatchAxis::new(0), BatchAxis::new(0), 2).unwrap();
        assert_eq!(output, input.log_softmax(1).unwrap());
    }

    #[test]
    fn test_log_softmax_differentiation() {
        check_gradient!(
            |input, weights| input.log_softmax(0)?.mul(&weights)?.reduce_sum(&[0], None),
            at = Array::vector(vec![-0.2f64, 0.5, 1.2]).unwrap(),
            with = Array::vector(vec![0.3f64, -1.0, 2.0]).unwrap(),
            step = 1e-5,
            tolerance = 1e-8,
        );
        // The dominant class keeps its small `1 - p` derivative, accumulated from the other classes' probabilities.
        let input = Array::vector(vec![9.0f32, -9.0, -9.0]).unwrap();
        let (output, gradient) = differentiate_at(input)
            .value_and_gradient(|input| input.log_softmax(0)?.slice_axis(0, 0, 1, 1)?.reshape([]))
            .unwrap();
        assert!(output.elements::<f32>().unwrap()[0] < -3.0e-8);
        let gradient = gradient.elements::<f32>().unwrap();
        assert!(gradient[0] > 3.0e-8 && gradient[0] < 3.1e-8);
        assert_eq!(gradient[0], -(gradient[1] + gradient[2]));
    }

    #[test]
    fn test_masked_log_softmax() {
        let logits = Array::vector(vec![1.0f64, f64::NAN, 3.0]).unwrap();
        let mask = Array::vector(vec![true, false, true]).unwrap();
        let output = logits.masked_log_softmax(&mask, 0).unwrap().to_f64s();
        let unmasked = Array::vector(vec![1.0f64, 3.0]).unwrap().log_softmax(0).unwrap().to_f64s();
        assert_eq!(output, vec![unmasked[0], f64::NEG_INFINITY, unmasked[1]]);
        assert_eq!(
            Array::vector(vec![1.0f32, 2.0])
                .unwrap()
                .masked_log_softmax(&Array::vector(vec![false, false]).unwrap(), 0),
            Ok(Array::vector(vec![f32::NEG_INFINITY, f32::NEG_INFINITY]).unwrap())
        );
        let composite = ArrayIrValue::Array(logits.clone());
        assert_eq!(
            composite.masked_log_softmax(&ArrayIrValue::Array(mask.clone()), 0),
            Ok(ArrayIrValue::Array(logits.masked_log_softmax(&mask, 0).unwrap()))
        );
    }

    #[test]
    fn test_masked_log_softmax_staging() {
        // Unselected classes enter the normalization as `-∞` logits and leave it as `-∞` log-probabilities.
        let (_, program) = ArrayContext::trace(
            |(logits, mask): (_, _)| logits.masked_log_softmax(&mask, 0),
            (ArrayType::new_static(DataType::F32, [2]), ArrayType::new_static(DataType::Boolean, [2])),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:f32[2], %1:bool[2] .
            let %2:f32[] = constant [value=-inf]
                %3:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %2
                %4:f32[2] = select %1 %0 %3
                %5:i64[] = argmax [axis=0, index_data_type=i64] %4
                %6:i64[2] = iota [type=i64[2], dimension=0]
                %7:i64[2] = broadcast [output_type=i64[2], output_axes=[]] %5
                %8:bool[2] = compare [direction=Equal] %6 %7
                %9:f32[2] = zero_like %4
                %10:f32[2] = select %8 %4 %9
                %11:f32[] = reduce [kind=sum, axes=[0]] %10
                %12:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %11
                %13:f32[2] = sub %4 %12
                %14:f32[2] = select %8 %9 %13
                %15:f32[2] = exp %14
                %16:f32[2] = select %8 %9 %15
                %17:f32[] = reduce [kind=sum, axes=[0]] %16
                %18:f32[] = ln_1p %17
                %19:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %18
                %20:f32[2] = sub %14 %19
                %21:f32[2] = sub %12 %12
                %22:bool[2] = compare [direction=Equal] %21 %9
                %23:f32[] = constant [value=NaN]
                %24:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %23
                %25:f32[2] = select %22 %20 %24
                %26:f32[2] = select %1 %25 %3
            in (%26)
        "}
            .trim_end()
        );
        let logits = Array::vector(vec![9.0f32, -9.0]).unwrap();
        let mask = Array::vector(vec![false, true]).unwrap();
        assert_eq!(program.interpret((logits.clone(), mask.clone())), logits.masked_log_softmax(&mask, 0));
    }

    #[test]
    fn test_masked_log_softmax_differentiation() {
        check_gradient!(
            |logits, mask| logits.masked_log_softmax(&mask, 0)?.slice_axis(0, 0, 1, 1)?.reshape([]),
            at = Array::vector(vec![-0.2f64, 0.5, 1.2]).unwrap(),
            with = Array::vector(vec![true, false, true]).unwrap(),
            step = 1e-5,
            tolerance = 1e-8,
        );
        let logits = Array::vector(vec![9.0f32, 20.0, -9.0]).unwrap();
        let mask = Array::vector(vec![true, false, true]).unwrap();
        let (output, gradient) = differentiate_at(logits)
            .with_captures(mask)
            .value_and_gradient(|logits, mask| logits.masked_log_softmax(&mask, 0)?.slice_axis(0, 0, 1, 1)?.reshape([]))
            .unwrap();
        // The dominant selected class keeps its small loss and derivative despite the larger unselected logit.
        assert!(output.elements::<f32>().unwrap()[0] < -1.5e-8);
        let gradient = gradient.elements::<f32>().unwrap();
        assert!(gradient[0] > 1.5e-8 && gradient[0] < 1.6e-8);
        assert_eq!(gradient[1], 0.0);
        assert_eq!(gradient[0], -gradient[2]);
    }

    #[test]
    fn test_softmax_cross_entropy() {
        let logits = Array::vector(vec![9.0f32, -9.0, -9.0]).unwrap();
        let targets = Array::vector(vec![1.0f32, 0.0, 0.0]).unwrap();
        let loss = logits.softmax_cross_entropy(&targets, 0).unwrap();
        assert!(loss.elements::<f32>().unwrap()[0] > 3.0e-8);
        assert_eq!(
            loss,
            logits.log_softmax(0).unwrap().slice_axis(0, 0, 1, 1).unwrap().reshape([]).unwrap().neg().unwrap()
        );
        // An excluded class with a zero target contributes `0 * log(0) = 0`.
        let logits = Array::vector(vec![f64::NEG_INFINITY, 0.0]).unwrap();
        let targets = Array::vector(vec![0.0f64, 1.0]).unwrap();
        assert_eq!(logits.softmax_cross_entropy(&targets, 0), Ok(Array::scalar(0.0f64).unwrap()));
        // Targets convert to the logits' working type and the result keeps the logits' element type.
        let logits = Array::vector(vec![bf16::from_f32(1.0), bf16::from_f32(0.0)]).unwrap();
        let targets = Array::vector(vec![0.5f64, 0.5]).unwrap();
        let loss = logits.softmax_cross_entropy(&targets, 0).unwrap();
        assert_eq!(loss.r#type().data_type(), DataType::BF16);
        assert_abs_diff_eq!(loss.to_f64s()[0], (1.0f64).exp().ln_1p() - 0.5, epsilon = 1e-2);
        let composite = ArrayIrValue::Array(logits.clone());
        assert_eq!(
            composite.softmax_cross_entropy(&ArrayIrValue::Array(targets.clone()), 0),
            Ok(ArrayIrValue::Array(logits.softmax_cross_entropy(&targets, 0).unwrap()))
        );
        assert_eq!(
            ArrayIrValue::Array(Array::vector(vec![f64::NEG_INFINITY, 0.0]).unwrap())
                .softmax_cross_entropy(&ArrayIrValue::Array(Array::vector(vec![0.0f64, 1.0]).unwrap()), 0),
            Ok(ArrayIrValue::Array(Array::scalar(0.0f64).unwrap()))
        );
    }

    #[test]
    fn test_softmax_cross_entropy_multiple_axes() {
        // Several class axes are the flattened class axis in increasing axis order.
        let logits = Array::from_elements(
            ArrayType::new_static(DataType::F64, [2, 2, 2]),
            &[1.0f64, -2.0, 0.5, 3.0, 0.0, 1.0, 2.0, -1.0],
        )
        .unwrap();
        let targets = Array::from_elements(
            ArrayType::new_static(DataType::F64, [2, 2, 2]),
            &[0.1f64, 0.2, 0.3, 0.4, 0.25, 0.25, 0.25, 0.25],
        )
        .unwrap();
        assert_eq!(
            logits.softmax_cross_entropy(&targets, [1, 2]),
            logits.reshape([2, 4]).unwrap().softmax_cross_entropy(&targets.reshape([2, 4]).unwrap(), 1)
        );
    }

    #[test]
    fn test_softmax_cross_entropy_validation() {
        let logits = Array::vector(vec![0.0f32, 0.0]).unwrap();
        let expected = Err(ProgramError::Type(TypeError::invalid(
            "`softmax_cross_entropy` probability targets must have the logits' shape and real floating-point elements",
        )));
        assert_eq!(logits.softmax_cross_entropy(&Array::vector(vec![1i32, 0]).unwrap(), 0), expected);
        assert_eq!(logits.softmax_cross_entropy(&Array::vector(vec![1.0f32]).unwrap(), 0), expected);
        assert_eq!(
            Array::vector(vec![1i32, 0]).unwrap().softmax_cross_entropy(&logits, 0),
            Err(ProgramError::Type(TypeError::invalid("`softmax_cross_entropy` requires real floating-point logits")))
        );
        let (mesh, _, _) = manual_types(ArrayType::new_static(DataType::F32, [2]));
        let reduced = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh, 1).with_reduced_axes(["x"]).unwrap())
            .unwrap();
        assert_eq!(
            ArrayContext::trace(
                |(logits, targets): (_, _)| logits.softmax_cross_entropy(&targets, 0),
                (ArrayType::new_static(DataType::F32, [2]), reduced),
            )
            .map(|(output_type, _)| output_type),
            Err(ProgramError::Type(TypeError::invalid(
                "`softmax_cross_entropy` does not support unreduced or reduced inputs"
            ))),
        );
    }

    #[test]
    fn test_softmax_cross_entropy_staging() {
        let (_, program) = ArrayContext::trace(
            |(logits, targets): (_, _)| logits.softmax_cross_entropy(&targets, 0),
            (ArrayType::new_static(DataType::F32, [2]), ArrayType::new_static(DataType::F32, [2])),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:f32[2], %1:f32[2] .
            let %2:i64[] = argmax [axis=0, index_data_type=i64] %0
                %3:i64[2] = iota [type=i64[2], dimension=0]
                %4:i64[2] = broadcast [output_type=i64[2], output_axes=[]] %2
                %5:bool[2] = compare [direction=Equal] %3 %4
                %6:f32[2] = zero_like %0
                %7:f32[2] = select %5 %0 %6
                %8:f32[] = reduce [kind=sum, axes=[0]] %7
                %9:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %8
                %10:f32[2] = sub %0 %9
                %11:f32[2] = select %5 %6 %10
                %12:f32[2] = exp %11
                %13:f32[2] = select %5 %6 %12
                %14:f32[] = reduce [kind=sum, axes=[0]] %13
                %15:f32[] = ln_1p %14
                %16:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %15
                %17:f32[2] = sub %11 %16
                %18:f32[2] = sub %9 %9
                %19:bool[2] = compare [direction=Equal] %18 %6
                %20:f32[] = constant [value=NaN]
                %21:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %20
                %22:f32[2] = select %19 %17 %21
                %23:f32[2] = zero_like %1
                %24:f32[] = constant [value=-inf]
                %25:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %24
                %26:bool[2] = compare [direction=Equal] %1 %23
                %27:bool[2] = compare [direction=Equal] %22 %25
                %28:bool[2] = and %26 %27
                %29:f32[2] = select %28 %23 %22
                %30:f32[2] = mul %1 %29
                %31:f32[] = reduce [kind=sum, axes=[0]] %30
                %32:f32[] = neg %31
            in (%32)
        "}
            .trim_end()
        );
        let logits = Array::vector(vec![9.0f32, -9.0]).unwrap();
        let targets = Array::vector(vec![0.0f32, 1.0]).unwrap();
        assert_eq!(program.interpret((logits.clone(), targets.clone())), logits.softmax_cross_entropy(&targets, 0));
    }

    #[test]
    fn test_softmax_cross_entropy_manual_variation() {
        // Varying logits with invariant targets are aligned once, before they are combined, so the staged program
        // carries a single variation transition and a loss that varies like the logits.
        let (mesh, varying, invariant) = manual_types(ArrayType::new_static(DataType::F32, [2]));
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |(logits, targets)| logits.softmax_cross_entropy(&targets, 0),
            (varying.clone(), invariant.clone()),
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh })],
        )
        .unwrap();
        let (_, expected_type, _) = manual_types(ArrayType::scalar(DataType::F32));
        assert_eq!(output_type, expected_type);
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}]}] .
            let %2:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_vary [axis_name=\"x\"] %1
                %3:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = argmax [axis=0, index_data_type=i64] %0
                %4:i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = iota [
                    type=i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    dimension=0,
                ]
                %5:i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %3
                %6:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=Equal] %4 %5
                %7:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = zero_like %0
                %8:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %6 %0 %7
                %9:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=sum, axes=[0]] %8
                %10:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %9
                %11:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %0 %10
                %12:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %6 %7 %11
                %13:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = exp %12
                %14:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %6 %7 %13
                %15:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=sum, axes=[0]] %14
                %16:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = ln_1p %15
                %17:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %16
                %18:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %12 %17
                %19:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %10 %10
                %20:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=Equal] %19 %7
                %21:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = constant [value=NaN]
                %22:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %21
                %23:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %20 %18 %22
                %24:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = zero_like %2
                %25:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = constant [value=-inf]
                %26:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %25
                %27:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=Equal] %2 %24
                %28:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=Equal] %23 %26
                %29:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = and %27 %28
                %30:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %29 %24 %23
                %31:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = mul %2 %30
                %32:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=sum, axes=[0]] %31
                %33:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = neg %32
            in (%33)
        "}
            .trim_end()
        );
    }

    #[test]
    fn test_softmax_cross_entropy_partial_evaluation() {
        // Known logits fold the whole normalization into residual inputs (the excluded-class mask and the
        // log-probabilities); the loss over the unknown targets remains.
        let (_, program) = ArrayContext::trace(
            |inputs: Vec<_>| inputs[0].softmax_cross_entropy(&inputs[1], 0).map(|output| vec![output]),
            vec![ArrayType::new_static(DataType::F32, [2]), ArrayType::new_static(DataType::F32, [2])],
        )
        .unwrap();
        let logits = Array::vector(vec![0.0f32, 0.0]).unwrap();
        let targets = Array::vector(vec![0.25f32, 0.75]).unwrap();
        let known = program
            .partially_evaluate(&[
                PartialValue::Known(logits.clone()),
                PartialValue::Unknown(targets.r#type().into_owned()),
            ])
            .unwrap();
        assert_eq!(
            known.program().to_string(),
            indoc! {"
            lambda %0:f32[2], %1:bool[2], %2:f32[2] .
            let %3:f32[2] = zero_like %0
                %4:bool[2] = compare [direction=Equal] %0 %3
                %5:bool[2] = and %4 %1
                %6:f32[2] = select %5 %3 %2
                %7:f32[2] = mul %0 %6
                %8:f32[] = reduce [kind=sum, axes=[0]] %7
                %9:f32[] = neg %8
            in (%9)
        "}
            .trim_end()
        );
        assert_eq!(
            known.interpret(&EagerContext::new(), &[targets.clone()]),
            Ok(vec![logits.softmax_cross_entropy(&targets, 0).unwrap()])
        );
    }

    #[test]
    fn test_softmax_cross_entropy_batching() {
        let logits = Array::matrix(2, 3, vec![9.0f64, -9.0, -9.0, 0.0, 1.0, 2.0]).unwrap();
        let targets = Array::vector(vec![1.0f64, 0.0, 0.0]).unwrap();
        let expected = logits
            .softmax_cross_entropy(&Array::matrix(2, 3, vec![1.0f64, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(), 1)
            .unwrap();
        assert_eq!(
            batch(
                |(logits, targets)| logits.softmax_cross_entropy(&targets, 0),
                (logits, targets),
                (BatchAxis::new(0), BatchAxis::replicated()),
                BatchAxis::new(0),
                None,
            ),
            Ok(expected),
        );
    }

    #[test]
    fn test_softmax_cross_entropy_differentiation() {
        check_gradient!(
            |logits, targets| logits.softmax_cross_entropy(&targets, 0),
            at = Array::vector(vec![-0.2f64, 0.5, 1.2]).unwrap(),
            with = Array::vector(vec![0.2f64, 0.3, 0.5]).unwrap(),
            step = 1e-5,
            tolerance = 1e-8,
        );
        // A saturated target class keeps its small loss and gradient, with the gradient summing to zero exactly.
        let logits = Array::vector(vec![9.0f32, -9.0, -9.0]).unwrap();
        let targets = Array::vector(vec![1.0f32, 0.0, 0.0]).unwrap();
        let (loss, gradient) = differentiate_at(logits)
            .with_captures(targets)
            .value_and_gradient(|logits, targets| logits.softmax_cross_entropy(&targets, 0))
            .unwrap();
        assert!(loss.elements::<f32>().unwrap()[0] > 3.0e-8);
        let gradient = gradient.elements::<f32>().unwrap();
        assert!(gradient[0] < -3.0e-8);
        assert_eq!(gradient[0], -(gradient[1] + gradient[2]));
    }

    #[test]
    fn test_softmax_cross_entropy_differentiation_probability_targets() {
        let logits = Array::vector(vec![0.0f64, 0.0]).unwrap();
        let targets = Array::vector(vec![0.0f64, 1.0]).unwrap();
        let (_, gradient) = differentiate_at(targets)
            .with_captures(logits)
            .value_and_gradient(|targets, logits| logits.softmax_cross_entropy(&targets, 0))
            .unwrap();
        assert_eq!(gradient.elements::<f64>(), Ok(vec![std::f64::consts::LN_2; 2]));
        // The derivative with respect to a zero target of an excluded class is defined as zero.
        let logits = Array::vector(vec![f64::NEG_INFINITY, 0.0]).unwrap();
        let targets = Array::vector(vec![0.0f64, 1.0]).unwrap();
        let (_, gradient) = differentiate_at(targets)
            .with_captures(logits)
            .value_and_gradient(|targets, logits| logits.softmax_cross_entropy(&targets, 0))
            .unwrap();
        assert_eq!(gradient, Array::vector(vec![0.0f64, 0.0]).unwrap());
    }

    #[test]
    fn test_masked_softmax_cross_entropy() {
        // Unselected classes contribute nothing, whatever their target, and the selected ones normalize alone.
        let logits = Array::vector(vec![1.0f64, 7.0, 3.0]).unwrap();
        let targets = Array::vector(vec![0.25f64, 1.0, 0.75]).unwrap();
        let mask = Array::vector(vec![true, false, true]).unwrap();
        let expected = Array::vector(vec![1.0f64, 3.0])
            .unwrap()
            .softmax_cross_entropy(&Array::vector(vec![0.25f64, 0.75]).unwrap(), 0)
            .unwrap();
        assert_eq!(logits.masked_softmax_cross_entropy(&targets, &mask, 0), Ok(expected));
        assert_eq!(
            logits.masked_softmax_cross_entropy(&targets, &Array::vector(vec![false; 3]).unwrap(), 0),
            Ok(Array::scalar(0.0f64).unwrap())
        );
        let composite = ArrayIrValue::Array(logits.clone());
        assert_eq!(
            composite.masked_softmax_cross_entropy(
                &ArrayIrValue::Array(targets.clone()),
                &ArrayIrValue::Array(mask.clone()),
                0
            ),
            Ok(ArrayIrValue::Array(logits.masked_softmax_cross_entropy(&targets, &mask, 0).unwrap()))
        );
    }

    #[test]
    fn test_masked_softmax_cross_entropy_validation() {
        let logits = Array::vector(vec![0.0f32, 0.0]).unwrap();
        let targets = Array::vector(vec![0.5f32, 0.5]).unwrap();
        assert_eq!(
            logits.masked_softmax_cross_entropy(&targets, &Array::vector(vec![true]).unwrap(), 0),
            Err(ProgramError::Type(TypeError::invalid(
                "`masked_softmax_cross_entropy` mask must be a Boolean array with the logits' shape"
            )))
        );
        assert_eq!(
            logits.masked_softmax_cross_entropy(
                &Array::vector(vec![1i32, 0]).unwrap(),
                &Array::vector(vec![true, true]).unwrap(),
                0
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "`masked_softmax_cross_entropy` probability targets must have the logits' shape and real floating-point \
                 elements"
            )))
        );
    }

    #[test]
    fn test_masked_softmax_cross_entropy_differentiation() {
        // The gradient is the selected probabilities minus the targets, and zero for unselected classes.
        let logits = Array::vector(vec![1.0f64, 7.0, 3.0]).unwrap();
        let targets = Array::vector(vec![0.25f64, 1.0, 0.75]).unwrap();
        let mask = Array::vector(vec![true, false, true]).unwrap();
        let (_, gradient) = differentiate_at(logits)
            .with_captures((targets, mask))
            .value_and_gradient(|logits, (targets, mask)| logits.masked_softmax_cross_entropy(&targets, &mask, 0))
            .unwrap();
        let probability = (-2.0f64).exp() / (1.0 + (-2.0f64).exp());
        let gradient = gradient.to_f64s();
        assert_abs_diff_eq!(gradient[0], probability - 0.25, epsilon = 1e-15);
        assert_eq!(gradient[1], 0.0);
        assert_abs_diff_eq!(gradient[2], (1.0 - probability) - 0.75, epsilon = 1e-15);
    }

    #[test]
    fn test_softmax_cross_entropy_with_integer_labels() {
        let logits = Array::matrix(2, 3, vec![9.0f32, -9.0, -9.0, -9.0, 9.0, -9.0]).unwrap();
        let labels = Array::vector(vec![0i32, 1]).unwrap();
        let loss = logits.softmax_cross_entropy_with_integer_labels(&labels, -1).unwrap().elements::<f32>().unwrap();
        assert!(loss.iter().all(|&value| value > 3.0e-8 && value < 3.1e-8));
        let singleton = Array::vector(vec![2.0f32]).unwrap();
        assert_eq!(
            singleton.softmax_cross_entropy_with_integer_labels(&Array::scalar(0i32).unwrap(), 0),
            Ok(Array::scalar(0.0f32).unwrap())
        );
        for label in [-1i64, 1] {
            assert!(
                singleton
                    .softmax_cross_entropy_with_integer_labels(&Array::scalar(label).unwrap(), 0)
                    .unwrap()
                    .elements::<f32>()
                    .unwrap()[0]
                    .is_nan()
            );
        }
        let composite = ArrayIrValue::Array(logits.clone());
        assert_eq!(
            composite.softmax_cross_entropy_with_integer_labels(&ArrayIrValue::Array(labels.clone()), -1),
            Ok(ArrayIrValue::Array(logits.softmax_cross_entropy_with_integer_labels(&labels, -1).unwrap()))
        );
        assert_eq!(
            ArrayIrValue::Array(singleton.clone())
                .softmax_cross_entropy_with_integer_labels(&ArrayIrValue::Array(Array::scalar(0i32).unwrap()), 0),
            Ok(ArrayIrValue::Array(Array::scalar(0.0f32).unwrap()))
        );
        let ArrayIrValue::Array(output) = ArrayIrValue::Array(singleton)
            .softmax_cross_entropy_with_integer_labels(&ArrayIrValue::Array(Array::scalar(1i32).unwrap()), 0)
            .unwrap()
        else {
            panic!("expected an array")
        };
        assert!(output.to_f64s()[0].is_nan());
    }

    #[test]
    fn test_softmax_cross_entropy_with_integer_labels_multiple_axes() {
        // A label indexes the classes flattened in increasing axis order.
        let logits = Array::from_elements(
            ArrayType::new_static(DataType::F64, [2, 2, 2]),
            &[1.0f64, -2.0, 0.5, 3.0, 0.0, 1.0, 2.0, -1.0],
        )
        .unwrap();
        let labels = Array::vector(vec![3i32, 2]).unwrap();
        assert_eq!(
            logits.softmax_cross_entropy_with_integer_labels(&labels, [1, 2]),
            logits.reshape([2, 4]).unwrap().softmax_cross_entropy_with_integer_labels(&labels, 1)
        );
        let labels = Array::vector(vec![1i32, 3]).unwrap();
        assert_eq!(
            logits.softmax_cross_entropy_with_integer_labels(&labels, [0, 2]),
            logits
                .transpose([1, 0, 2])
                .unwrap()
                .reshape([2, 4])
                .unwrap()
                .softmax_cross_entropy_with_integer_labels(&labels, 1)
        );
    }

    #[test]
    fn test_softmax_cross_entropy_with_integer_labels_validation() {
        let logits = Array::vector(vec![0.0f32, 0.0]).unwrap();
        let expected = Err(ProgramError::Type(TypeError::invalid(
            "`softmax_cross_entropy_with_integer_labels` labels must have the logits' shape with the class axes \
             removed and integer elements",
        )));
        assert_eq!(logits.softmax_cross_entropy_with_integer_labels(&Array::vector(vec![0i32]).unwrap(), 0), expected);
        assert_eq!(logits.softmax_cross_entropy_with_integer_labels(&Array::scalar(0.0f32).unwrap(), 0), expected);
        assert_eq!(
            logits.softmax_cross_entropy_with_integer_labels(&Array::scalar(0i32).unwrap(), 2),
            Err(ProgramError::Type(TypeError::invalid(
                "`softmax_cross_entropy_with_integer_labels` axis 2 is out of bounds for rank 1"
            )))
        );
        let (mesh, _, _) = manual_types(ArrayType::new_static(DataType::F32, [2]));
        let reduced = ArrayType::scalar(DataType::I32)
            .with_sharding(Sharding::replicated(mesh, 0).with_reduced_axes(["x"]).unwrap())
            .unwrap();
        assert_eq!(
            ArrayContext::trace(
                |(logits, labels): (_, _)| logits.softmax_cross_entropy_with_integer_labels(&labels, 0),
                (ArrayType::new_static(DataType::F32, [2]), reduced),
            )
            .map(|(output_type, _)| output_type),
            Err(ProgramError::Type(TypeError::invalid(
                "`softmax_cross_entropy_with_integer_labels` does not support unreduced or reduced inputs"
            ))),
        );
    }

    #[test]
    fn test_softmax_cross_entropy_with_integer_labels_staging() {
        let (_, program) = ArrayContext::trace(
            |(logits, labels)| logits.softmax_cross_entropy_with_integer_labels(&labels, -1),
            (ArrayType::new_static(DataType::F32, [2, 3]), ArrayType::new_static(DataType::I32, [2])),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:f32[2, 3], %1:i32[2] .
            let %2:i64[2] = convert_element_type [data_type=i64] %1
                %3:i64[2] = argmax [axis=1, index_data_type=i64] %0
                %4:i64[2, 3] = iota [type=i64[2, 3], dimension=1]
                %5:i64[2, 3] = broadcast [output_type=i64[2, 3], output_axes=[0]] %3
                %6:bool[2, 3] = compare [direction=Equal] %4 %5
                %7:f32[2, 3] = zero_like %0
                %8:f32[2, 3] = select %6 %0 %7
                %9:f32[2] = reduce [kind=sum, axes=[1]] %8
                %10:f32[2, 3] = broadcast [output_type=f32[2, 3], output_axes=[0]] %9
                %11:f32[2, 3] = sub %0 %10
                %12:f32[2, 3] = select %6 %7 %11
                %13:f32[2, 3] = exp %12
                %14:f32[2, 3] = select %6 %7 %13
                %15:f32[2] = reduce [kind=sum, axes=[1]] %14
                %16:f32[2] = ln_1p %15
                %17:f32[2, 3] = broadcast [output_type=f32[2, 3], output_axes=[0]] %16
                %18:f32[2, 3] = sub %12 %17
                %19:f32[2, 3] = sub %10 %10
                %20:bool[2, 3] = compare [direction=Equal] %19 %7
                %21:f32[] = constant [value=NaN]
                %22:f32[2, 3] = broadcast [output_type=f32[2, 3], output_axes=[]] %21
                %23:f32[2, 3] = select %20 %18 %22
                %24:i64[2, 3] = iota [type=i64[2, 3], dimension=1]
                %25:i64[2, 3] = broadcast [output_type=i64[2, 3], output_axes=[0]] %2
                %26:bool[2, 3] = compare [direction=Equal] %24 %25
                %27:f32[2, 3] = zero_like %23
                %28:f32[2, 3] = select %26 %23 %27
                %29:f32[2] = reduce [kind=sum, axes=[1]] %28
                %30:f32[2] = neg %29
                %31:i64[2, 3] = convert_element_type [data_type=i64] %26
                %32:i64[2] = reduce [kind=sum, axes=[1]] %31
                %33:i64[2] = zero_like %2
                %34:bool[2] = compare [direction=GreaterThan] %32 %33
                %35:f32[] = constant [value=NaN]
                %36:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %35
                %37:f32[2] = select %34 %30 %36
            in (%37)
        "}
            .trim_end()
        );
        let logits = Array::matrix(2, 3, vec![9.0f32, -9.0, -9.0, -9.0, 9.0, -9.0]).unwrap();
        let labels = Array::vector(vec![0i32, 2]).unwrap();
        assert_eq!(
            program.interpret((logits.clone(), labels.clone())),
            logits.softmax_cross_entropy_with_integer_labels(&labels, -1)
        );
    }

    #[test]
    fn test_softmax_cross_entropy_with_integer_labels_manual_variation() {
        // Labels and logits may vary over different manual axes: the invariant one is aligned before the class
        // coordinates, which carry the logits' variation, are compared with the labels.
        let (mesh, varying, invariant) = manual_types(ArrayType::new_static(DataType::F32, [2]));
        let (_, varying_labels, invariant_labels) = manual_types(ArrayType::scalar(DataType::I32));
        let named_axes = vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })];
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |(logits, labels)| logits.softmax_cross_entropy_with_integer_labels(&labels, 0),
            (varying, invariant_labels),
            named_axes.clone(),
        )
        .unwrap();
        let (_, varying_loss, _) = manual_types(ArrayType::scalar(DataType::F32));
        assert_eq!(output_type, varying_loss);
        assert_eq!(
            program.to_string(),
indoc! {"
            lambda %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], %1:i32[][sharding={mesh<['x'=2:manual]>, []}] .
            let %2:i64[][sharding={mesh<['x'=2:manual]>, []}] = convert_element_type [data_type=i64] %1
                %3:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = parallel_vary [axis_name=\"x\"] %2
                %4:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = argmax [axis=0, index_data_type=i64] %0
                %5:i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = iota [
                    type=i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    dimension=0,
                ]
                %6:i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %4
                %7:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=Equal] %5 %6
                %8:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = zero_like %0
                %9:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %7 %0 %8
                %10:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=sum, axes=[0]] %9
                %11:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %10
                %12:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %0 %11
                %13:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %7 %8 %12
                %14:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = exp %13
                %15:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %7 %8 %14
                %16:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=sum, axes=[0]] %15
                %17:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = ln_1p %16
                %18:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %17
                %19:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %13 %18
                %20:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %11 %11
                %21:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=Equal] %20 %8
                %22:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = constant [value=NaN]
                %23:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %22
                %24:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %21 %19 %23
                %25:i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = iota [
                    type=i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    dimension=0,
                ]
                %26:i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %3
                %27:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=Equal] %25 %26
                %28:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = zero_like %24
                %29:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %27 %24 %28
                %30:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=sum, axes=[0]] %29
                %31:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = neg %30
                %32:i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = convert_element_type [data_type=i64] %27
                %33:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=sum, axes=[0]] %32
                %34:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = zero_like %3
                %35:bool[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = compare [direction=GreaterThan] %33 %34
                %36:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = constant [value=NaN]
                %37:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = select %35 %31 %36
            in (%37)
        "}
            .trim_end()
        );

        // Varying labels with invariant logits vary the logits first, so the loss varies too.
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |(logits, labels)| logits.softmax_cross_entropy_with_integer_labels(&labels, 0),
            (invariant, varying_labels),
            named_axes,
        )
        .unwrap();
        assert_eq!(output_type, varying_loss);
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}]}], %1:i32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] .
            let %2:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = convert_element_type [data_type=i64] %1
                %3:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_vary [axis_name=\"x\"] %0
                %4:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = argmax [axis=0, index_data_type=i64] %3
                %5:i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = iota [
                    type=i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    dimension=0,
                ]
                %6:i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %4
                %7:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=Equal] %5 %6
                %8:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = zero_like %3
                %9:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %7 %3 %8
                %10:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=sum, axes=[0]] %9
                %11:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %10
                %12:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %3 %11
                %13:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %7 %8 %12
                %14:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = exp %13
                %15:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %7 %8 %14
                %16:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=sum, axes=[0]] %15
                %17:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = ln_1p %16
                %18:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %17
                %19:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %13 %18
                %20:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %11 %11
                %21:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=Equal] %20 %8
                %22:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = constant [value=NaN]
                %23:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %22
                %24:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %21 %19 %23
                %25:i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = iota [
                    type=i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    dimension=0,
                ]
                %26:i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %2
                %27:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=Equal] %25 %26
                %28:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = zero_like %24
                %29:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %27 %24 %28
                %30:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=sum, axes=[0]] %29
                %31:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = neg %30
                %32:i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = convert_element_type [data_type=i64] %27
                %33:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=sum, axes=[0]] %32
                %34:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = zero_like %2
                %35:bool[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = compare [direction=GreaterThan] %33 %34
                %36:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = constant [value=NaN]
                %37:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = select %35 %31 %36
            in (%37)
        "}
            .trim_end()
        );
    }

    #[test]
    fn test_softmax_cross_entropy_with_integer_labels_shard_map() {
        // The reference emulation of `shard_map` executes the manual-region program: each device holds two rows of the
        // logits and the same two replicated labels, which the loss aligns to the varying logits, and the per-device
        // losses and their gradient equal the unsharded ones.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharded_rows =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated])
                .unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let logits =
            Array::matrix(4, 3, vec![9.0f32, -9.0, -9.0, 0.0, 1.0, 2.0, -9.0, 9.0, -9.0, 2.0, 1.0, 0.0]).unwrap();
        let labels = Array::vector(vec![0i32, 2]).unwrap();
        let tiled_labels = Array::vector(vec![0i32, 2, 0, 2]).unwrap();
        let (_, program) = ArrayIrContext::trace(
            |inputs: Vec<Tracer<ArrayIrTracingContext>>| {
                let loss = shard_map(
                    |(logits, labels): (
                        ShardMapTracer<ArrayIrTracingContext>,
                        ShardMapTracer<ArrayIrTracingContext>,
                    )| { logits.softmax_cross_entropy_with_integer_labels(&labels, 1).unwrap() },
                    (
                        ValueProjection::<ArrayType>::into_projected(inputs[0].clone())?,
                        ValueProjection::<ArrayType>::into_projected(inputs[1].clone())?,
                    ),
                    mesh.clone(),
                    (sharded_rows.clone(), replicated.clone()),
                    sharded.clone(),
                )?;
                let gradient = differentiate_at(inputs[0].clone())
                    .with_captures(inputs[1].clone())
                    .gradient(|logits, labels| {
                        let loss = shard_map(
                            |(logits, labels): (
                                ShardMapTracer<ArrayIrTracingContext>,
                                ShardMapTracer<ArrayIrTracingContext>,
                            )| {
                                logits.softmax_cross_entropy_with_integer_labels(&labels, 1).unwrap()
                            },
                            (
                                ValueProjection::<ArrayType>::into_projected(logits)?,
                                ValueProjection::<ArrayType>::into_projected(labels)?,
                            ),
                            mesh.clone(),
                            (sharded_rows.clone(), replicated.clone()),
                            sharded.clone(),
                        )?;
                        Ok::<_, ProgramError>(loss.reduce_sum(&[0], None)?.into_value())
                    })
                    .map_err(ProgramError::from)?;
                Ok(vec![loss.into_value(), gradient])
            },
            vec![ArrayIrType::from(logits.r#type().into_owned()), ArrayIrType::from(labels.r#type().into_owned())],
        )
        .unwrap();
        let mut outputs =
            program.interpret(vec![ArrayIrValue::Array(logits.clone()), ArrayIrValue::Array(labels)]).unwrap();
        let gradient = array(outputs.pop().unwrap());
        let loss = array(outputs.pop().unwrap());
        assert_eq!(loss.r#type().sharding(), Some(&sharded));
        assert_eq!(
            loss.to_f64s(),
            logits.softmax_cross_entropy_with_integer_labels(&tiled_labels, 1).unwrap().to_f64s()
        );
        let expected_gradient = differentiate_at(logits)
            .with_captures(tiled_labels)
            .gradient(|logits, labels| {
                logits.softmax_cross_entropy_with_integer_labels(&labels, 1)?.reduce_sum(&[0], None)
            })
            .unwrap();
        assert_eq!(gradient.to_f64s(), expected_gradient.to_f64s());
    }

    #[test]
    fn test_softmax_cross_entropy_with_integer_labels_partial_evaluation() {
        // Known labels fold the class mask, the validity test, and the constants into residual inputs; the
        // normalization of the unknown logits remains.
        let (_, program) = ArrayContext::trace(
            |inputs: Vec<_>| {
                inputs[0].softmax_cross_entropy_with_integer_labels(&inputs[1], 0).map(|output| vec![output])
            },
            vec![ArrayType::new_static(DataType::F32, [2]), ArrayType::scalar(DataType::I32)],
        )
        .unwrap();
        let logits = Array::vector(vec![9.0f32, -9.0]).unwrap();
        let labels = Array::scalar(1i32).unwrap();
        let known = program
            .partially_evaluate(&[
                PartialValue::Unknown(logits.r#type().into_owned()),
                PartialValue::Known(labels.clone()),
            ])
            .unwrap();
        assert_eq!(
            known.program().to_string(),
            indoc! {"
            lambda %0:f32[2], %1:i64[2], %2:f32[2], %3:bool[2], %4:bool[], %5:f32[] .
            let %6:i64[] = argmax [axis=0, index_data_type=i64] %0
                %7:i64[2] = broadcast [output_type=i64[2], output_axes=[]] %6
                %8:bool[2] = compare [direction=Equal] %1 %7
                %9:f32[2] = zero_like %0
                %10:f32[2] = select %8 %0 %9
                %11:f32[] = reduce [kind=sum, axes=[0]] %10
                %12:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %11
                %13:f32[2] = sub %12 %12
                %14:bool[2] = compare [direction=Equal] %13 %9
                %15:f32[2] = sub %0 %12
                %16:f32[2] = select %8 %9 %15
                %17:f32[2] = exp %16
                %18:f32[2] = select %8 %9 %17
                %19:f32[] = reduce [kind=sum, axes=[0]] %18
                %20:f32[] = ln_1p %19
                %21:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %20
                %22:f32[2] = sub %16 %21
                %23:f32[2] = select %14 %22 %2
                %24:f32[2] = zero_like %23
                %25:f32[2] = select %3 %23 %24
                %26:f32[] = reduce [kind=sum, axes=[0]] %25
                %27:f32[] = neg %26
                %28:f32[] = select %4 %27 %5
            in (%28)
        "}
            .trim_end()
        );
        assert_eq!(
            known.interpret(&EagerContext::new(), &[logits.clone()]),
            Ok(vec![logits.softmax_cross_entropy_with_integer_labels(&labels, 0).unwrap()])
        );
    }

    #[test]
    fn test_softmax_cross_entropy_with_integer_labels_batching() {
        let logits = Array::matrix(2, 3, vec![9.0f64, -9.0, -9.0, 0.0, 1.0, 2.0]).unwrap();
        let labels = Array::vector(vec![0i32, 2]).unwrap();
        assert_eq!(
            batch(
                |(logits, labels)| logits.softmax_cross_entropy_with_integer_labels(&labels, 0),
                (logits.clone(), labels.clone()),
                (BatchAxis::new(0), BatchAxis::new(0)),
                BatchAxis::new(0),
                None,
            ),
            Ok(logits.softmax_cross_entropy_with_integer_labels(&labels, 1).unwrap()),
        );
    }

    #[test]
    fn test_softmax_cross_entropy_with_integer_labels_differentiation() {
        check_gradient!(
            |logits, labels| logits.softmax_cross_entropy_with_integer_labels(&labels, 0),
            at = Array::vector(vec![-0.2f64, 0.5, 1.2]).unwrap(),
            with = Array::scalar(1i32).unwrap(),
            step = 1e-5,
            tolerance = 1e-8,
        );
        let logits = Array::matrix(2, 3, vec![9.0f32, -9.0, -9.0, -9.0, 9.0, -9.0]).unwrap();
        let labels = Array::vector(vec![0i32, 1]).unwrap();
        let (_, gradient) = differentiate_at(logits)
            .with_captures(labels)
            .value_and_gradient(|logits, labels| {
                logits.softmax_cross_entropy_with_integer_labels(&labels, -1)?.reduce_sum(&[0], None)
            })
            .unwrap();
        let gradient = gradient.elements::<f32>().unwrap();
        assert_eq!(gradient[0], -(gradient[1] + gradient[2]));
        assert_eq!(gradient[4], -(gradient[3] + gradient[5]));
    }

    #[test]
    fn test_masked_softmax_cross_entropy_with_integer_labels() {
        // The loss is the negative masked log-probability of the labeled class: a label naming an unselected class
        // yields an infinite loss, a row without any selected class has zero loss, and a label outside the class
        // range remains NaN.
        let logits = Array::matrix(2, 3, vec![1.0f64, 7.0, 3.0, 1.0, 7.0, 3.0]).unwrap();
        let mask = Array::matrix(2, 3, vec![true, false, true, true, false, true]).unwrap();
        let labels = Array::vector(vec![0i32, 1]).unwrap();
        let loss = logits.masked_softmax_cross_entropy_with_integer_labels(&labels, &mask, 1).unwrap().to_f64s();
        let expected = Array::vector(vec![1.0f64, 3.0])
            .unwrap()
            .softmax_cross_entropy_with_integer_labels(&Array::scalar(0i32).unwrap(), 0)
            .unwrap()
            .to_f64s();
        assert_eq!(loss, vec![expected[0], f64::INFINITY]);
        let unselected = Array::matrix(2, 3, vec![false; 6]).unwrap();
        assert_eq!(
            logits.masked_softmax_cross_entropy_with_integer_labels(
                &Array::vector(vec![0i32, 5]).unwrap(),
                &unselected,
                1
            ),
            Ok(Array::vector(vec![0.0f64, 0.0]).unwrap())
        );
        assert!(
            logits
                .masked_softmax_cross_entropy_with_integer_labels(&Array::vector(vec![3i32, -1]).unwrap(), &mask, 1)
                .unwrap()
                .to_f64s()
                .iter()
                .all(|value| value.is_nan())
        );
        let composite = ArrayIrValue::Array(logits.clone());
        assert_eq!(
            composite.masked_softmax_cross_entropy_with_integer_labels(
                &ArrayIrValue::Array(labels.clone()),
                &ArrayIrValue::Array(mask.clone()),
                1
            ),
            Ok(ArrayIrValue::Array(
                logits.masked_softmax_cross_entropy_with_integer_labels(&labels, &mask, 1).unwrap()
            ))
        );
    }

    #[test]
    fn test_masked_softmax_cross_entropy_with_integer_labels_validation() {
        let logits = Array::vector(vec![0.0f32, 0.0]).unwrap();
        let labels = Array::scalar(0i32).unwrap();
        assert_eq!(
            logits.masked_softmax_cross_entropy_with_integer_labels(&labels, &Array::vector(vec![true]).unwrap(), 0),
            Err(ProgramError::Type(TypeError::invalid(
                "`masked_softmax_cross_entropy_with_integer_labels` mask must be a Boolean array with the logits' shape"
            )))
        );
        assert_eq!(
            logits.masked_softmax_cross_entropy_with_integer_labels(
                &Array::scalar(0.0f32).unwrap(),
                &Array::vector(vec![true, true]).unwrap(),
                0
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "`masked_softmax_cross_entropy_with_integer_labels` labels must have the logits' shape with the class \
                 axes removed and integer elements"
            )))
        );
    }

    #[test]
    fn test_masked_softmax_cross_entropy_with_integer_labels_differentiation() {
        let logits = Array::vector(vec![1.0f64, 7.0, 3.0]).unwrap();
        let mask = Array::vector(vec![true, false, true]).unwrap();
        let (_, gradient) = differentiate_at(logits)
            .with_captures((Array::scalar(0i32).unwrap(), mask))
            .value_and_gradient(|logits, (labels, mask)| {
                logits.masked_softmax_cross_entropy_with_integer_labels(&labels, &mask, 0)
            })
            .unwrap();
        let probability = (-2.0f64).exp() / (1.0 + (-2.0f64).exp());
        let gradient = gradient.to_f64s();
        assert_abs_diff_eq!(gradient[0], probability - 1.0, epsilon = 1e-15);
        assert_eq!(gradient[1], 0.0);
        assert_abs_diff_eq!(gradient[2], 1.0 - probability, epsilon = 1e-15);
    }

    #[test]
    fn test_differentiable_softmax_cross_entropy_with_target_logits_call() {
        let function = softmax_cross_entropy_with_target_logits::<ArrayContext>(-1);
        let teacher = Array::vector(vec![2.0f64, 1.0, 0.0]).unwrap();
        let output = function.call(teacher.clone(), teacher).unwrap();
        let denominator = 1.0 + (-1.0f64).exp() + (-2.0f64).exp();
        let expected = denominator.ln() + ((-1.0f64).exp() + 2.0 * (-2.0f64).exp()) / denominator;
        assert!((output.to_f64s()[0] - expected).abs() < 1e-15);

        let student = Array::matrix(2, 2, vec![0.0f64, 0.0, 0.0, 0.0]).unwrap();
        let teacher = Array::matrix(2, 2, vec![0.0f64, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(
            softmax_cross_entropy_with_target_logits::<ArrayContext>(0).call(student, teacher),
            Ok(Array::vector(vec![2.0f64.ln(), 2.0f64.ln()]).unwrap()),
        );
    }

    #[test]
    fn test_differentiable_softmax_cross_entropy_with_target_logits_call_validation() {
        let function = softmax_cross_entropy_with_target_logits::<ArrayContext>(-1);
        assert_eq!(
            function.call(Array::vector(vec![0.0f32]).unwrap(), Array::vector(vec![0.0f64]).unwrap()),
            Err(ProgramError::Type(TypeError::invalid(
                "`softmax_cross_entropy_with_target_logits` requires logits and target logits with the same shape and element type",
            ))),
        );
        assert_eq!(
            function.call(Array::vector(vec![0i32]).unwrap(), Array::vector(vec![0i32]).unwrap()),
            Err(ProgramError::Type(TypeError::invalid(
                "`softmax_cross_entropy_with_target_logits` requires real floating-point logits"
            ))),
        );
        assert_eq!(
            function.call(Array::vector(Vec::<f32>::new()).unwrap(), Array::vector(Vec::<f32>::new()).unwrap()),
            Err(ProgramError::Type(TypeError::invalid(
                "`softmax_cross_entropy_with_target_logits` requires a non-empty class axis"
            ))),
        );
        assert_eq!(
            softmax_cross_entropy_with_target_logits::<ArrayContext>(1)
                .call(Array::vector(vec![0.0f32]).unwrap(), Array::vector(vec![0.0f32]).unwrap()),
            Err(ProgramError::Type(TypeError::invalid(
                "`softmax_cross_entropy_with_target_logits` axis 1 is out of bounds for rank 1"
            ))),
        );
        // Placement is rejected before the inputs are aligned, so a reduced teacher beside a varying student reports
        // this function's rejection rather than a failed variation transition.
        let (mesh, varying, _) = manual_types(ArrayType::new_static(DataType::F32, [2]));
        let reduced = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_reduced_axes(["x"]).unwrap())
            .unwrap();
        assert_eq!(
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |(student, teacher): (_, _)| function.call(student, teacher),
                (varying, reduced),
                vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh })],
            )
            .map(|(output_type, _)| output_type),
            Err(ProgramError::Type(TypeError::invalid(
                "`softmax_cross_entropy_with_target_logits` does not support unreduced or reduced inputs"
            ))),
        );
    }

    #[test]
    fn test_softmax_cross_entropy_with_target_logits() {
        let function = softmax_cross_entropy_with_target_logits::<ArrayContext>(-1);
        let teacher = Array::vector(vec![2.0f32, 1.0, 0.0]).unwrap();
        let (loss, gradient) = differentiate_at(teacher.clone())
            .with_captures(teacher)
            .value_and_gradient(|student, teacher| function.call(student, teacher))
            .unwrap();
        assert!(loss.to_f64s()[0] > 0.0);
        assert_eq!(gradient, Array::vector(vec![0.0f32, 0.0, 0.0]).unwrap());
    }

    #[test]
    fn test_softmax_cross_entropy_with_target_logits_multiple_axes() {
        // Several class axes are flattened inside the retained rules, so the loss and the student gradient equal
        // those of the flattened logits.
        let function = softmax_cross_entropy_with_target_logits::<ArrayContext>([0, 1]);
        let flat = softmax_cross_entropy_with_target_logits::<ArrayContext>(0);
        let student = Array::matrix(2, 2, vec![1.0f64, 0.5, -0.5, 2.0]).unwrap();
        let teacher = Array::matrix(2, 2, vec![0.0f64, -0.5, -1.5, 1.0]).unwrap();
        let (loss, gradient) = differentiate_at(student.clone())
            .with_captures(teacher.clone())
            .value_and_gradient(|student, teacher| function.call(student, teacher))
            .unwrap();
        let (expected_loss, expected_gradient) = differentiate_at(student.reshape([4]).unwrap())
            .with_captures(teacher.reshape([4]).unwrap())
            .value_and_gradient(|student, teacher| flat.call(student, teacher))
            .unwrap();
        assert_eq!(loss, expected_loss);
        assert_eq!(gradient.reshape([4]), Ok(expected_gradient));
    }

    #[test]
    fn test_softmax_cross_entropy_with_target_logits_staging() {
        // The call stops the teacher's gradient and stages one retained custom function whose primal composes the
        // teacher's softmax with the student's log-softmax.
        let function = softmax_cross_entropy_with_target_logits::<ArrayContext>(-1);
        let (_, program) = ArrayContext::trace(
            |(student, teacher)| function.call(student, teacher),
            (ArrayType::new_static(DataType::F32, [2]), ArrayType::new_static(DataType::F32, [2])),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:f32[2], %1:f32[2] .
            let %2:f32[2] = stop_gradient %1
                %3:f32[] = custom_function [name=\"softmax_cross_entropy_with_target_logits\", non_differentiated_count=1] %2 %0 [
                    primal={
                        lambda %0:f32[2], %1:f32[2] .
                        let %2:i64[] = argmax [axis=0, index_data_type=i64] %0
                            %3:i64[2] = iota [type=i64[2], dimension=0]
                            %4:i64[2] = broadcast [output_type=i64[2], output_axes=[]] %2
                            %5:bool[2] = compare [direction=Equal] %3 %4
                            %6:f32[2] = zero_like %0
                            %7:f32[2] = select %5 %0 %6
                            %8:f32[] = reduce [kind=sum, axes=[0]] %7
                            %9:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %8
                            %10:f32[2] = sub %0 %9
                            %11:f32[2] = select %5 %6 %10
                            %12:f32[2] = exp %11
                            %13:f32[2] = select %5 %6 %12
                            %14:f32[] = reduce [kind=sum, axes=[0]] %13
                            %15:f32[] = ln_1p %14
                            %16:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %15
                            %17:f32[2] = sub %11 %16
                            %18:f32[2] = sub %9 %9
                            %19:bool[2] = compare [direction=Equal] %18 %6
                            %20:f32[] = constant [value=NaN]
                            %21:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %20
                            %22:f32[2] = select %19 %17 %21
                            %23:f32[2] = exp %22
                            %24:i64[] = argmax [axis=0, index_data_type=i64] %1
                            %25:i64[2] = iota [type=i64[2], dimension=0]
                            %26:i64[2] = broadcast [output_type=i64[2], output_axes=[]] %24
                            %27:bool[2] = compare [direction=Equal] %25 %26
                            %28:f32[2] = zero_like %1
                            %29:f32[2] = select %27 %1 %28
                            %30:f32[] = reduce [kind=sum, axes=[0]] %29
                            %31:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %30
                            %32:f32[2] = sub %1 %31
                            %33:f32[2] = select %27 %28 %32
                            %34:f32[2] = exp %33
                            %35:f32[2] = select %27 %28 %34
                            %36:f32[] = reduce [kind=sum, axes=[0]] %35
                            %37:f32[] = ln_1p %36
                            %38:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %37
                            %39:f32[2] = sub %33 %38
                            %40:f32[2] = sub %31 %31
                            %41:bool[2] = compare [direction=Equal] %40 %28
                            %42:f32[] = constant [value=NaN]
                            %43:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %42
                            %44:f32[2] = select %41 %39 %43
                            %45:f32[] = constant [value=0.0]
                            %46:f32[2] = broadcast [output_type=f32[2], output_axes=[]] %45
                            %47:bool[2] = compare [direction=Equal] %23 %46
                            %48:f32[2] = select %47 %46 %44
                            %49:f32[2] = mul %23 %48
                            %50:f32[] = reduce [kind=sum, axes=[0]] %49
                            %51:f32[] = neg %50
                        in (%51)
                    },
                ]
            in (%3)
        "}
            .trim_end()
        );
        let student = Array::vector(vec![1.0f32, 0.0]).unwrap();
        let teacher = Array::vector(vec![0.0f32, 0.0]).unwrap();
        assert_eq!(program.interpret((student.clone(), teacher.clone())), function.call(student, teacher));
    }

    #[test]
    fn test_softmax_cross_entropy_with_target_logits_manual_variation() {
        // Student and teacher may vary over different manual axes. The invariant teacher is aligned before the call
        // boundary, after its gradient is stopped, so the primal and its rules see inputs of one variation and the
        // constants they create carry it.
        let function = softmax_cross_entropy_with_target_logits::<ArrayContext>(-1);
        let (mesh, varying, invariant) = manual_types(ArrayType::new_static(DataType::F32, [2]));
        let named_axes = vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh })];
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |(student, teacher): (_, _)| function.call(student, teacher),
            (varying.clone(), invariant.clone()),
            named_axes.clone(),
        )
        .unwrap();
        let (_, varying_loss, _) = manual_types(ArrayType::scalar(DataType::F32));
        assert_eq!(output_type, varying_loss);
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}]}] .
            let %2:f32[2][sharding={mesh<['x'=2:manual]>, [{}]}] = stop_gradient %1
                %3:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_vary [axis_name=\"x\"] %2
                %4:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = custom_function [name=\"softmax_cross_entropy_with_target_logits\", non_differentiated_count=1] %3 %0 [
                    primal={
                        lambda %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                        let %2:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = argmax [axis=0, index_data_type=i64] %0
                            %3:i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = iota [
                                type=i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                                dimension=0,
                            ]
                            %4:i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                                output_type=i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                                output_axes=[],
                            ] %2
                            %5:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=Equal] %3 %4
                            %6:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = zero_like %0
                            %7:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %5 %0 %6
                            %8:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=sum, axes=[0]] %7
                            %9:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                                output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                                output_axes=[],
                            ] %8
                            %10:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %0 %9
                            %11:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %5 %6 %10
                            %12:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = exp %11
                            %13:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %5 %6 %12
                            %14:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=sum, axes=[0]] %13
                            %15:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = ln_1p %14
                            %16:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                                output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                                output_axes=[],
                            ] %15
                            %17:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %11 %16
                            %18:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %9 %9
                            %19:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=Equal] %18 %6
                            %20:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = constant [value=NaN]
                            %21:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                                output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                                output_axes=[],
                            ] %20
                            %22:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %19 %17 %21
                            %23:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = exp %22
                            %24:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = argmax [axis=0, index_data_type=i64] %1
                            %25:i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = iota [
                                type=i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                                dimension=0,
                            ]
                            %26:i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                                output_type=i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                                output_axes=[],
                            ] %24
                            %27:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=Equal] %25 %26
                            %28:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = zero_like %1
                            %29:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %27 %1 %28
                            %30:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=sum, axes=[0]] %29
                            %31:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                                output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                                output_axes=[],
                            ] %30
                            %32:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %1 %31
                            %33:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %27 %28 %32
                            %34:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = exp %33
                            %35:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %27 %28 %34
                            %36:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=sum, axes=[0]] %35
                            %37:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = ln_1p %36
                            %38:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                                output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                                output_axes=[],
                            ] %37
                            %39:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %33 %38
                            %40:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %31 %31
                            %41:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=Equal] %40 %28
                            %42:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = constant [value=NaN]
                            %43:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                                output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                                output_axes=[],
                            ] %42
                            %44:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %41 %39 %43
                            %45:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = constant [value=0.0]
                            %46:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                                output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                                output_axes=[],
                            ] %45
                            %47:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=Equal] %23 %46
                            %48:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %47 %46 %44
                            %49:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = mul %23 %48
                            %50:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=sum, axes=[0]] %49
                            %51:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = neg %50
                        in (%51)
                    },
                ]
            in (%4)
        "}
            .trim_end()
        );

        // An invariant student is varied before the call, and the gradient of the varying loss with respect to it
        // transposes that transition into a cross-device sum of the per-device cotangents. The staged gradient inlines
        // the retained rule, so every constant it creates is visible with the inputs' variation.
        let (gradient_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |(student, teacher): (_, _)| {
                differentiate_at(student)
                    .with_captures(teacher)
                    .gradient(|student, teacher| function.call(student, teacher))
                    .map_err(ProgramError::from)
            },
            (invariant.clone(), varying),
            named_axes,
        )
        .unwrap();
        assert_eq!(gradient_type, invariant);
        assert_eq!(
            program.to_string(),
            indoc! {"
            lambda %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}]}], %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
            let %2:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = stop_gradient %1
                %3:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = parallel_vary [axis_name=\"x\"] %0
                %4:i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = iota [
                    type=i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    dimension=0,
                ]
                %5:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = argmax [axis=0, index_data_type=i64] %2
                %6:i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %5
                %7:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=Equal] %4 %6
                %8:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = zero_like %2
                %9:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %7 %2 %8
                %10:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=sum, axes=[0]] %9
                %11:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %10
                %12:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %11 %11
                %13:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=Equal] %12 %8
                %14:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %2 %11
                %15:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %7 %8 %14
                %16:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = exp %15
                %17:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %7 %8 %16
                %18:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=sum, axes=[0]] %17
                %19:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = ln_1p %18
                %20:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %19
                %21:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %15 %20
                %22:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = constant [value=NaN]
                %23:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %22
                %24:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %13 %21 %23
                %25:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = exp %24
                %26:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = constant [value=0.0]
                %27:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %26
                %28:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=Equal] %25 %27
                %29:i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = iota [
                    type=i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    dimension=0,
                ]
                %30:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = argmax [axis=0, index_data_type=i64] %3
                %31:i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=i64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %30
                %32:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=Equal] %29 %31
                %33:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = zero_like %3
                %34:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %32 %3 %33
                %35:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=sum, axes=[0]] %34
                %36:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %35
                %37:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %36 %36
                %38:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=Equal] %37 %33
                %39:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %3 %36
                %40:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %32 %33 %39
                %41:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = exp %40
                %42:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %32 %33 %41
                %43:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=sum, axes=[0]] %42
                %44:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = ln_1p %43
                %45:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %44
                %46:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %40 %45
                %47:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = constant [value=NaN]
                %48:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %47
                %49:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %38 %46 %48
                %50:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %28 %27 %49
                %51:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = mul %25 %50
                %52:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=sum, axes=[0]] %51
                %53:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = neg %52
                %54:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = constant [value=-inf]
                %55:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %54
                %56:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=Equal] %2 %55
                %57:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=Equal] %3 %55
                %58:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = and %56 %57
                %59:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %3 %2
                %60:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = abs %59
                %61:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = constant [value=inf]
                %62:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %61
                %63:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=LessThan] %60 %62
                %64:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %63 %59 %55
                %65:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=max, axes=[0]] %64
                %66:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = abs %65
                %67:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = constant [value=inf]
                %68:bool[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = compare [direction=LessThan] %66 %67
                %69:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = constant [value=0.0]
                %70:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = select %68 %65 %69
                %71:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %70
                %72:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %59 %71
                %73:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = constant [value=0.0]
                %74:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %73
                %75:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %63 %72 %74
                %76:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = mul %25 %75
                %77:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=sum, axes=[0]] %76
                %78:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %77
                %79:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %75 %78
                %80:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = abs %79
                %81:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = constant [value=1.0]
                %82:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %81
                %83:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=LessThan] %80 %82
                %84:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = and %63 %83
                %85:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = or %58 %84
                %86:bool[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=all, axes=[0]] %85
                %87:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %86
                %88:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %87 %79 %74
                %89:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = exp_m1 %88
                %90:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = mul %25 %89
                %91:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=sum, axes=[0]] %90
                %92:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = ln_1p %91
                %93:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %92
                %94:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %88 %93
                %95:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = abs %94
                %96:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = compare [direction=LessThan] %95 %82
                %97:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = and %87 %96
                %98:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = exp_m1 %94
                %99:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = mul %25 %98
                %100:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = exp %49
                %101:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %100 %25
                %102:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %97 %99 %101
                %103:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = reduce [kind=sum, axes=[0]] %102
                %104:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %103
                %105:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = mul %100 %104
                %106:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sub %102 %105
                %107:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = one [type=f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]]
                %108:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                    output_type=f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                    output_axes=[],
                ] %107
                %109:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = mul %106 %108
                %110:f32[2][sharding={mesh<['x'=2:manual]>, [{}]}] = parallel_reduce [kind=sum, axis_name=\"x\", mesh=['x'=2:manual]] %109
            in (%110)
        "}
            .trim_end()
        );
    }

    #[test]
    fn test_softmax_cross_entropy_with_target_logits_shard_map() {
        // Inside a `shard_map`, each device holds two student rows beside the same two replicated teacher rows. The
        // call aligns the invariant teacher, and the reference emulation executes the primal and the student
        // gradient device by device with the results of the unsharded call.
        let function = softmax_cross_entropy_with_target_logits::<ProjectedContext<ArrayIrContext, ArrayType>>(-1);
        let plain = softmax_cross_entropy_with_target_logits::<ArrayContext>(-1);
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharded_rows =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated])
                .unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 2);
        let student =
            Array::matrix(4, 3, vec![2.0f32 - 1e-7, 1.0 + 1e-7, 0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, -9.0, 9.0, -9.0])
                .unwrap();
        let teacher = Array::matrix(2, 3, vec![2.0f32, 1.0, 0.0, 2.0, 1.0, 0.0]).unwrap();
        let tiled_teacher =
            Array::matrix(4, 3, vec![2.0f32, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0]).unwrap();
        let (_, program) = ArrayIrContext::trace(
            |inputs: Vec<Tracer<ArrayIrTracingContext>>| {
                let loss = shard_map(
                    |(student, teacher): (
                        ShardMapTracer<ArrayIrTracingContext>,
                        ShardMapTracer<ArrayIrTracingContext>,
                    )| { function.call(student, teacher).unwrap() },
                    (
                        ValueProjection::<ArrayType>::into_projected(inputs[0].clone())?,
                        ValueProjection::<ArrayType>::into_projected(inputs[1].clone())?,
                    ),
                    mesh.clone(),
                    (sharded_rows.clone(), replicated.clone()),
                    sharded.clone(),
                )?;
                let gradient = differentiate_at(inputs[0].clone())
                    .with_captures(inputs[1].clone())
                    .gradient(|student, teacher| {
                        let loss = shard_map(
                            |(student, teacher): (
                                ShardMapTracer<ArrayIrTracingContext>,
                                ShardMapTracer<ArrayIrTracingContext>,
                            )| { function.call(student, teacher).unwrap() },
                            (
                                ValueProjection::<ArrayType>::into_projected(student)?,
                                ValueProjection::<ArrayType>::into_projected(teacher)?,
                            ),
                            mesh.clone(),
                            (sharded_rows.clone(), replicated.clone()),
                            sharded.clone(),
                        )?;
                        Ok::<_, ProgramError>(loss.reduce_sum(&[0], None)?.into_value())
                    })
                    .map_err(ProgramError::from)?;
                Ok(vec![loss.into_value(), gradient])
            },
            vec![ArrayIrType::from(student.r#type().into_owned()), ArrayIrType::from(teacher.r#type().into_owned())],
        )
        .unwrap();
        let mut outputs =
            program.interpret(vec![ArrayIrValue::Array(student.clone()), ArrayIrValue::Array(teacher)]).unwrap();
        let gradient = array(outputs.pop().unwrap());
        let loss = array(outputs.pop().unwrap());
        assert_eq!(loss.r#type().sharding(), Some(&sharded));
        let (expected_loss, expected_gradient) = differentiate_at(student)
            .with_captures(tiled_teacher)
            .value_and_gradient(|student, teacher| plain.call(student, teacher)?.reduce_sum(&[0], None))
            .unwrap();
        assert_eq!(loss.reduce_sum(&[0], None).unwrap().to_f64s(), expected_loss.to_f64s());
        assert_eq!(gradient.to_f64s(), expected_gradient.to_f64s());
    }

    #[test]
    fn test_softmax_cross_entropy_with_target_logits_partial_evaluation() {
        let function = softmax_cross_entropy_with_target_logits::<ArrayContext>(-1);
        let (_, program) = ArrayContext::trace(
            |inputs: Vec<_>| function.call(inputs[0].clone(), inputs[1].clone()).map(|output| vec![output]),
            vec![ArrayType::new_static(DataType::F64, [2]), ArrayType::new_static(DataType::F64, [2])],
        )
        .unwrap();
        let student = Array::vector(vec![1.0f64, 0.0]).unwrap();
        let teacher = Array::vector(vec![0.0f64, 0.0]).unwrap();
        let known = program
            .partially_evaluate(&[PartialValue::Known(student.clone()), PartialValue::Known(teacher.clone())])
            .unwrap();
        assert_eq!(known.program().to_string(), "lambda  .\nin ()");
        assert_eq!(
            known.outputs(),
            &[PartialEvaluationOutput::Known(function.call(student.clone(), teacher.clone()).unwrap())]
        );
        // A known teacher alone cannot fold the call, whose primal needs both inputs, so the call stays residual.
        let unknown = program
            .partially_evaluate(&[
                PartialValue::Unknown(student.r#type().into_owned()),
                PartialValue::Known(teacher.clone()),
            ])
            .unwrap();
        assert_eq!(unknown.outputs(), &[PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(
            unknown.program().to_string(),
            indoc! {"
            lambda %0:f64[2], %1:f64[2] .
            let %2:f64[] = custom_function [name=\"softmax_cross_entropy_with_target_logits\", non_differentiated_count=1] %1 %0 [
                primal={
                    lambda %0:f64[2], %1:f64[2] .
                    let %2:i64[2] = iota [type=i64[2], dimension=0]
                        %3:i64[] = argmax [axis=0, index_data_type=i64] %0
                        %4:i64[2] = broadcast [output_type=i64[2], output_axes=[]] %3
                        %5:bool[2] = compare [direction=Equal] %2 %4
                        %6:f64[2] = zero_like %0
                        %7:f64[2] = select %5 %0 %6
                        %8:f64[] = reduce [kind=sum, axes=[0]] %7
                        %9:f64[2] = broadcast [output_type=f64[2], output_axes=[]] %8
                        %10:f64[2] = sub %9 %9
                        %11:bool[2] = compare [direction=Equal] %10 %6
                        %12:f64[2] = sub %0 %9
                        %13:f64[2] = select %5 %6 %12
                        %14:f64[2] = exp %13
                        %15:f64[2] = select %5 %6 %14
                        %16:f64[] = reduce [kind=sum, axes=[0]] %15
                        %17:f64[] = ln_1p %16
                        %18:f64[2] = broadcast [output_type=f64[2], output_axes=[]] %17
                        %19:f64[2] = sub %13 %18
                        %20:f64[] = constant [value=NaN]
                        %21:f64[2] = broadcast [output_type=f64[2], output_axes=[]] %20
                        %22:f64[2] = select %11 %19 %21
                        %23:f64[2] = exp %22
                        %24:f64[] = constant [value=0.0]
                        %25:f64[2] = broadcast [output_type=f64[2], output_axes=[]] %24
                        %26:bool[2] = compare [direction=Equal] %23 %25
                        %27:i64[2] = iota [type=i64[2], dimension=0]
                        %28:i64[] = argmax [axis=0, index_data_type=i64] %1
                        %29:i64[2] = broadcast [output_type=i64[2], output_axes=[]] %28
                        %30:bool[2] = compare [direction=Equal] %27 %29
                        %31:f64[2] = zero_like %1
                        %32:f64[2] = select %30 %1 %31
                        %33:f64[] = reduce [kind=sum, axes=[0]] %32
                        %34:f64[2] = broadcast [output_type=f64[2], output_axes=[]] %33
                        %35:f64[2] = sub %34 %34
                        %36:bool[2] = compare [direction=Equal] %35 %31
                        %37:f64[2] = sub %1 %34
                        %38:f64[2] = select %30 %31 %37
                        %39:f64[2] = exp %38
                        %40:f64[2] = select %30 %31 %39
                        %41:f64[] = reduce [kind=sum, axes=[0]] %40
                        %42:f64[] = ln_1p %41
                        %43:f64[2] = broadcast [output_type=f64[2], output_axes=[]] %42
                        %44:f64[2] = sub %38 %43
                        %45:f64[] = constant [value=NaN]
                        %46:f64[2] = broadcast [output_type=f64[2], output_axes=[]] %45
                        %47:f64[2] = select %36 %44 %46
                        %48:f64[2] = select %26 %25 %47
                        %49:f64[2] = mul %23 %48
                        %50:f64[] = reduce [kind=sum, axes=[0]] %49
                        %51:f64[] = neg %50
                    in (%51)
                },
            ]
            in (%2)
        "}
            .trim_end()
        );
        assert_eq!(
            unknown.interpret(&EagerContext::new(), &[student.clone()]),
            Ok(vec![function.call(student, teacher).unwrap()])
        );
    }

    #[test]
    fn test_softmax_cross_entropy_with_target_logits_batching() {
        let function = softmax_cross_entropy_with_target_logits::<ArrayContext>(-1);
        let student = Array::matrix(2, 2, vec![0.0f64, 0.0, 1000.0, -1000.0]).unwrap();
        let teacher = Array::vector(vec![0.0f64, 0.0]).unwrap();
        assert_eq!(
            batch(
                |(student, teacher)| function.call(student, teacher),
                (student, teacher),
                (BatchAxis::new(0), BatchAxis::replicated()),
                BatchAxis::new(0),
                None,
            ),
            Ok(Array::vector(vec![2.0f64.ln(), 1000.0]).unwrap()),
        );
    }

    #[test]
    fn test_softmax_cross_entropy_with_target_logits_differentiation() {
        let function = &softmax_cross_entropy_with_target_logits::<ArrayContext>(-1);
        check_gradient!(
            |student, teacher| function.call(student, teacher),
            at = Array::vector(vec![-0.2f64, 0.5, 1.2]).unwrap(),
            with = Array::vector(vec![0.0f64, 0.1, 0.3]).unwrap(),
            step = 1e-5,
            tolerance = 1e-8,
        );
        check_gradient!(
            |student, teacher| function.call(student, teacher),
            at = Array::vector(vec![-1.0f64, 2.0, 0.0]).unwrap(),
            with = Array::vector(vec![2.0f64, -2.0, 1.0]).unwrap(),
            step = 1e-5,
            tolerance = 1e-8,
        );
    }

    #[test]
    fn test_softmax_cross_entropy_with_target_logits_differentiation_teacher_gradient() {
        let function = softmax_cross_entropy_with_target_logits::<ArrayContext>(-1);
        let (student_gradient, teacher_gradient) =
            differentiate_at((Array::vector(vec![1.0f64, -1.0]).unwrap(), Array::vector(vec![0.0f64, 0.0]).unwrap()))
                .gradient(|(student, teacher)| function.call(student, teacher))
                .unwrap();
        let expected = 1.0 / (1.0 + (-2.0f64).exp()) - 0.5;
        assert!((student_gradient.to_f64s()[0] - expected).abs() < 1e-15);
        assert!((student_gradient.to_f64s()[1] + expected).abs() < 1e-15);
        assert_eq!(teacher_gradient, Array::vector(vec![0.0f64, 0.0]).unwrap());
    }

    #[test]
    fn test_softmax_cross_entropy_with_target_logits_differentiation_near_teacher() {
        let function = softmax_cross_entropy_with_target_logits::<ArrayContext>(-1);
        let teacher = Array::vector(vec![2.0f32, 1.0, 0.0]).unwrap();
        let student_values = [2.0f32 - 1e-7, 1.0f32 + 1e-7, 0.0];
        let gradient = differentiate_at(Array::vector(student_values.to_vec()).unwrap())
            .with_captures(teacher)
            .gradient(|student, teacher| function.call(student, teacher))
            .unwrap()
            .to_f64s();

        // Compute the oracle in f64 from the represented f32 inputs. This preserves the input rounding while leaving
        // enough precision to resolve the probability differences that ordinary f32 subtraction loses.
        let student_denominator = student_values.iter().map(|&value| (f64::from(value) - 2.0).exp()).sum::<f64>();
        let teacher_denominator = 1.0 + (-1.0f64).exp() + (-2.0f64).exp();
        for (axis, (actual, &student)) in gradient.iter().zip(&student_values).enumerate() {
            let expected =
                (f64::from(student) - 2.0).exp() / student_denominator - (-(axis as f64)).exp() / teacher_denominator;
            assert!((actual - expected).abs() < 1e-13, "class {axis}: {actual} differs from {expected}");
        }
        assert!(gradient[0] < 0.0);
        assert!(gradient[1] > 0.0);
        assert!(gradient[2] > 0.0);
        assert!(gradient.iter().sum::<f64>().abs() < 1e-14);
    }

    #[test]
    fn test_softmax_cross_entropy_with_target_logits_differentiation_shared_masks() {
        // A class that both distributions mask is excluded, so the remaining classes keep the cancellation-resistant
        // gradient instead of the rounded direct difference, whose `f32` resolution near `0.7` is about `6e-8`.
        let function = softmax_cross_entropy_with_target_logits::<ArrayContext>(-1);
        let teacher = Array::vector(vec![2.0f32, 1.0, f32::NEG_INFINITY]).unwrap();
        let student_values = [2.0f32 - 1e-7, 1.0f32 + 1e-7, f32::NEG_INFINITY];
        let (loss, gradient) = differentiate_at(Array::vector(student_values.to_vec()).unwrap())
            .with_captures(teacher)
            .value_and_gradient(|student, teacher| function.call(student, teacher))
            .unwrap();
        assert!(loss.to_f64s()[0].is_finite());
        let gradient = gradient.to_f64s();
        let student_denominator = student_values[..2].iter().map(|&value| (f64::from(value) - 2.0).exp()).sum::<f64>();
        let teacher_denominator = 1.0 + (-1.0f64).exp();
        for (axis, (actual, &student)) in gradient[..2].iter().zip(&student_values[..2]).enumerate() {
            let expected =
                (f64::from(student) - 2.0).exp() / student_denominator - (-(axis as f64)).exp() / teacher_denominator;
            assert!((actual - expected).abs() < 1e-13, "class {axis}: {actual} differs from {expected}");
        }
        assert_eq!(gradient[2], 0.0);
        assert!(gradient[0] < 0.0 && gradient[1] > 0.0);
    }

    #[test]
    fn test_softmax_cross_entropy_with_target_logits_differentiation_common_offset() {
        let function = softmax_cross_entropy_with_target_logits::<ArrayContext>(-1);
        let gradient = differentiate_at(Array::vector(vec![1.0f32, 0.5, -0.5]).unwrap())
            .with_captures(Array::vector(vec![0.0f32, -0.5, -1.5]).unwrap())
            .gradient(|student, teacher| function.call(student, teacher))
            .unwrap();
        assert_eq!(gradient, Array::vector(vec![0.0f32, 0.0, 0.0]).unwrap());
        // A shared mask does not take part in the offset, so a negative common offset also cancels exactly.
        let gradient = differentiate_at(Array::vector(vec![-1.0f32, -1.5, f32::NEG_INFINITY]).unwrap())
            .with_captures(Array::vector(vec![0.0f32, -0.5, f32::NEG_INFINITY]).unwrap())
            .gradient(|student, teacher| function.call(student, teacher))
            .unwrap();
        assert_eq!(gradient, Array::vector(vec![0.0f32, 0.0, 0.0]).unwrap());
    }

    #[test]
    fn test_softmax_cross_entropy_with_target_logits_differentiation_distant_teacher() {
        let function = softmax_cross_entropy_with_target_logits::<ArrayContext>(-1);
        let student = Array::vector(vec![-1000.0f32, 1000.0]).unwrap();
        let teacher = Array::vector(vec![1000.0f32, -1000.0]).unwrap();
        assert_eq!(
            differentiate_at(student)
                .with_captures(teacher)
                .value_and_gradient(|student, teacher| function.call(student, teacher)),
            Ok((Array::scalar(2000.0f32).unwrap(), Array::vector(vec![-1.0f32, 1.0]).unwrap())),
        );
    }

    #[test]
    fn test_softmax_cross_entropy_with_target_logits_differentiation_masks_and_singleton() {
        let function = softmax_cross_entropy_with_target_logits::<ArrayContext>(-1);
        let teacher = Array::vector(vec![0.0f32, f32::NEG_INFINITY]).unwrap();
        assert_eq!(
            differentiate_at(teacher.clone())
                .with_captures(teacher)
                .value_and_gradient(|student, teacher| function.call(student, teacher)),
            Ok((Array::scalar(0.0f32).unwrap(), Array::vector(vec![0.0f32, 0.0]).unwrap())),
        );
        // A class that only the teacher masks routes the row through the direct difference and keeps the loss finite,
        // while a class that only the student masks makes the loss infinite.
        let (loss, gradient) = differentiate_at(Array::vector(vec![0.0f32, 0.0]).unwrap())
            .with_captures(Array::vector(vec![0.0f32, f32::NEG_INFINITY]).unwrap())
            .value_and_gradient(|student, teacher| function.call(student, teacher))
            .unwrap();
        assert_eq!(loss, Array::scalar(std::f32::consts::LN_2).unwrap());
        assert_eq!(gradient, Array::vector(vec![-0.5f32, 0.5]).unwrap());
        let (loss, gradient) = differentiate_at(Array::vector(vec![0.0f32, f32::NEG_INFINITY]).unwrap())
            .with_captures(Array::vector(vec![0.0f32, 0.0]).unwrap())
            .value_and_gradient(|student, teacher| function.call(student, teacher))
            .unwrap();
        assert_eq!(loss, Array::scalar(f32::INFINITY).unwrap());
        assert_eq!(gradient, Array::vector(vec![0.5f32, -0.5]).unwrap());
        assert_eq!(
            differentiate_at(Array::vector(vec![1000.0f32]).unwrap())
                .with_captures(Array::vector(vec![-1000.0f32]).unwrap())
                .value_and_gradient(|student, teacher| function.call(student, teacher)),
            Ok((Array::scalar(0.0f32).unwrap(), Array::vector(vec![0.0f32]).unwrap())),
        );
    }

    #[test]
    fn test_softmax_cross_entropy_with_target_logits_differentiation_narrow_inputs() {
        let function = softmax_cross_entropy_with_target_logits::<ArrayContext>(-1);
        let teacher = Array::vector(vec![2.0f32, 1.0, 0.0]).unwrap().convert_element_type(DataType::BF16).unwrap();
        let (loss, gradient) = differentiate_at(teacher.clone())
            .with_captures(teacher)
            .value_and_gradient(|student, teacher| function.call(student, teacher))
            .unwrap();
        assert_eq!(loss.r#type().data_type(), DataType::BF16);
        assert_eq!(gradient.r#type().data_type(), DataType::BF16);
        assert_eq!(gradient.to_f64s(), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_softmax_cross_entropy_with_target_logits_differentiation_jvp() {
        let function = softmax_cross_entropy_with_target_logits::<ArrayContext>(-1);
        let (_, tangent) = differentiate_at(Array::vector(vec![1.0f64, 0.0]).unwrap())
            .with_captures(Array::vector(vec![0.0f64, 0.0]).unwrap())
            .jvp(Array::vector(vec![3.0f64, -2.0]).unwrap(), |student, teacher| function.call(student, teacher))
            .unwrap();
        let expected = 5.0 * (1.0 / (1.0 + (-1.0f64).exp()) - 0.5);
        assert!((tangent.to_f64s()[0] - expected).abs() < 1e-15);
    }

    #[test]
    fn test_softmax_cross_entropy_with_target_logits_differentiation_hessian() {
        let function = softmax_cross_entropy_with_target_logits::<ArrayContext>(-1);
        let hessian = differentiate_at(Array::vector(vec![0.0f64, 0.0]).unwrap())
            .with_captures(Array::vector(vec![0.0f64, 0.0]).unwrap())
            .hessian(|student, teacher| function.call(student, teacher))
            .unwrap();
        assert_eq!(hessian.values()[0], Array::matrix(2, 2, vec![0.25f64, -0.25, -0.25, 0.25]).unwrap());

        // Unselected centered differences may be NaN at shared masks or enormous at distant logits. The guarded
        // exponential branch must not contaminate either Hessian with non-finite coefficients.
        let masked = Array::vector(vec![0.0f64, f64::NEG_INFINITY]).unwrap();
        let hessian = differentiate_at(masked.clone())
            .with_captures(masked)
            .hessian(|student, teacher| function.call(student, teacher))
            .unwrap();
        assert_eq!(hessian.values()[0], Array::matrix(2, 2, vec![0.0f64; 4]).unwrap());

        // A shared mask beside two live classes leaves the live classes' curvature intact.
        let masked = Array::vector(vec![0.0f64, 0.0, f64::NEG_INFINITY]).unwrap();
        let hessian = differentiate_at(masked.clone())
            .with_captures(masked)
            .hessian(|student, teacher| function.call(student, teacher))
            .unwrap();
        assert_eq!(
            hessian.values()[0],
            Array::matrix(3, 3, vec![0.25f64, -0.25, 0.0, -0.25, 0.25, 0.0, 0.0, 0.0, 0.0]).unwrap()
        );

        let hessian = differentiate_at(Array::vector(vec![-1000.0f64, 1000.0]).unwrap())
            .with_captures(Array::vector(vec![1000.0f64, -1000.0]).unwrap())
            .hessian(|student, teacher| function.call(student, teacher))
            .unwrap();
        assert_eq!(hessian.values()[0], Array::matrix(2, 2, vec![0.0f64; 4]).unwrap());
    }
}
