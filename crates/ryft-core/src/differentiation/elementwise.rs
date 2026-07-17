//! Contains shared differentiation scaffolding for [`ElementwiseOperation`](crate::ElementwiseOperation)s. Elementwise
//! arithmetic rules all follow the same shape. They first compute the primal through the operation's own value-level
//! capability. Then, they resolve the tangent target type of the primal output, convert and broadcast every live
//! operand tangent (and any primal coefficients) into that widened differential representation, and combine the
//! per-operand contributions. Finally, they pair the primal with the combined tangent. This module owns that
//! scaffolding (including the structural-zero policy for outputs without a tangent space) so that each operation's
//! [`DifferentiableOperation`](crate::DifferentiableOperation) implementation only supplies its mathematical content
//! (i.e., the primal computation and the per-operand tangent terms), and the [`ElementwiseDerivativeAlignment`] trait
//! those rules (and the broadcasting, selection, and reduction rules) use to align derivative contributions between
//! operand types and the common type inferred for an implicitly broadcasting result.

use crate::differentiation::{DifferentiableType, DifferentiationDual, DifferentiationError};
use crate::macros::check_count;
use crate::operations::manipulation::{Broadcast, ConvertElementType, Reshape, Transpose};
use crate::operations::sharding::Reshard;
use crate::programs::ProgramError;
use crate::programs::atoms::MaybeZero;
use crate::programs::types::TypeError;
use crate::programs::values::Value;
use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};
use crate::types::{ArrayType, DataType, Size};

/// [`Value`] whose derivative contributions can be _aligned_ with the common [`Type`](crate::Type) inferred for an
/// implicitly broadcasting elementwise result and _unaligned_ back to an operand type. The two methods form an adjoint
/// pair: [`align_tangent`](Self::align_tangent) applies the implicit element-type promotion, broadcast, and resharding
/// an [`ElementwiseOperation`](crate::ElementwiseOperation) performs on an operand, and
/// [`unalign_cotangent`](Self::unalign_cotangent) applies the adjoint of that linear map
/// (i.e., demotion, sum-reduction over broadcasting axes, and the reverse reshard).
pub trait ElementwiseDerivativeAlignment<T: DifferentiableType>: Value<Type = T> {
    /// Aligns this tangent [`Value`] (or primal coefficient) with `target`, converting and broadcasting it
    /// into that [`Type`](crate::Type).
    fn align_tangent(&self, target: &T) -> Result<Self, DifferentiationError>;

    /// Unaligns this cotangent [`Value`] from `target` back to this [`Value`]'s type by applying the adjoint
    /// of the implicit conversion and broadcast.
    fn unalign_cotangent(&self, target: &T) -> Result<Self, DifferentiationError>;
}

impl<V: Value<Type = DataType> + ConvertElementType> ElementwiseDerivativeAlignment<DataType> for V {
    #[inline]
    fn align_tangent(&self, target: &DataType) -> Result<Self, DifferentiationError> {
        if self.r#type().as_ref() == target { Ok(self.clone()) } else { Ok(self.convert_element_type(*target)?) }
    }

    #[inline]
    fn unalign_cotangent(&self, target: &DataType) -> Result<Self, DifferentiationError> {
        if self.r#type().as_ref() == target { Ok(self.clone()) } else { Ok(self.convert_element_type(*target)?) }
    }
}

impl<V: Value<Type = ArrayType> + Broadcast + ConvertElementType + Reshape + Transpose + Reshard + Reduce>
    ElementwiseDerivativeAlignment<ArrayType> for V
{
    fn align_tangent(&self, target: &ArrayType) -> Result<Self, DifferentiationError> {
        let mut value = if self.r#type().data_type() == target.data_type() {
            self.clone()
        } else {
            self.convert_element_type(target.data_type())?
        };

        if value.r#type().as_ref() == target {
            return Ok(value);
        }

        let requires_reshard = value.r#type().sharding() != target.sharding();

        let rank = value.r#type().rank();
        if rank > target.rank() {
            return Err(TypeError {
                message: format!("cannot align tangent type {} to output type {}", value.r#type(), target),
            }
            .into());
        }

        let offset = target.rank() - rank;
        let output_axes = (0..rank).map(|axis| axis + offset).collect::<Vec<_>>();
        value = value.broadcast(target.clone(), output_axes.as_slice())?;

        // `BroadcastOperation` carries the requested output type, but changing an explicit/manual sharding is a
        // semantic redistribution rather than a metadata-only broadcast. Here we stage that transition explicitly so
        // that backend lowering cannot silently relabel the tangent when the primal result is placed differently from
        // this operand.
        if requires_reshard && let Some(sharding) = target.sharding() {
            value = value.reshard(sharding);
        }

        Ok(value)
    }

    fn unalign_cotangent(&self, target: &ArrayType) -> Result<Self, DifferentiationError> {
        let offset = self.r#type().rank().checked_sub(target.rank()).ok_or_else(|| TypeError {
            message: format!("cannot unalign cotangent type {} to input cotangent type {}", self.r#type(), target),
        })?;
        let output_axes = (0..target.rank()).map(|axis| axis + offset).collect::<Vec<_>>();
        self.unalign_cotangent_along(target, output_axes.as_slice())
    }
}

/// [`ArrayType`]-typed [`Value`] whose cotangents can additionally be _unaligned_ through the adjoint of an *explicit*
/// broadcast (e.g., [`BroadcastOperation`](crate::BroadcastOperation)) that placed the operand's axes at arbitrary
/// result positions. This extends [`ElementwiseDerivativeAlignment`] as a separate trait because axis placement is a
/// concept that is specific to [`ArrayType`]-typed values. Scalar types have no axes and participate only in implicit
/// suffix-aligned alignment maps.
pub trait BroadcastDerivativeAlignment: ElementwiseDerivativeAlignment<ArrayType> {
    /// Unaligns this cotangent [`Value`] back to `target` by applying the adjoint of an explicit broadcast that
    /// mapped each axis of `target` to the axis of this [`Value`] named by the corresponding entry of `output_axes`.
    /// Contributions along this [`Value`]'s remaining axes are sum-reduced away, the kept axes are permuted back
    /// into `target`'s axis order, and the element type, sharding, and stretched unit axes are restored.
    /// [`ElementwiseDerivativeAlignment::unalign_cotangent`] is the implicit special case that suffix-aligns
    /// the trailing axes.
    fn unalign_cotangent_along(&self, target: &ArrayType, output_axes: &[usize]) -> Result<Self, DifferentiationError>;
}

impl<V: Value<Type = ArrayType> + Broadcast + ConvertElementType + Reshape + Transpose + Reshard + Reduce>
    BroadcastDerivativeAlignment for V
{
    fn unalign_cotangent_along(&self, target: &ArrayType, output_axes: &[usize]) -> Result<Self, DifferentiationError> {
        // The broadcast being transposed mapped each `target` axis to the axis of this cotangent named by the
        // corresponding `output_axes` entry, so the mapping must name one in-range axis per `target` axis. Anything
        // else means the caller's axis mapping and the operand type disagree.
        let value_type = self.r#type();
        if output_axes.len() != target.rank() || output_axes.iter().any(|axis| *axis >= value_type.rank()) {
            return Err(TypeError {
                message: format!(
                    "cannot unalign cotangent type {value_type} to input cotangent type {target} using output axes \
                     {output_axes:?}",
                ),
            }
            .into());
        }

        // Classify each `target` axis. An axis whose extent survived the broadcast unchanged is *kept* (i.e., its
        // cotangent flows straight through), while an axis the broadcast stretched from extent one is dropped here and
        // summed over below, with its unit extent restored by the later reshaping (i.e., the adjoint of stretching is
        // summation). Any other extent mismatch means that the mapping never described a valid broadcast.
        let mut kept_axes = Vec::with_capacity(target.rank());
        for (target_axis, &output_axis) in output_axes.iter().enumerate() {
            let target_dimension = target.dimension(target_axis as isize);
            let value_dimension = value_type.dimension(output_axis as isize);
            if target_dimension != value_dimension {
                if target_dimension != Size::Static(1) {
                    return Err(TypeError {
                        message: format!(
                            "cannot unalign cotangent axis {output_axis} of size {value_dimension} to input axis \
                             {target_axis} of size {target_dimension}",
                        ),
                    }
                    .into());
                }
            } else {
                kept_axes.push((target_axis, output_axis));
            }
        }

        // Sum-reduce every non-kept axis of this cotangent: both the axes the broadcast introduced outright (i.e.,
        // that are never named by a kept `target` axis) and the stretched axes classified above. This is the core
        // adjoint step, since a broadcast duplicates values along exactly these axes.
        let reduce_axes = (0..value_type.rank())
            .filter(|axis| kept_axes.iter().all(|(_, value_axis)| value_axis != axis))
            .collect::<Vec<_>>();
        let mut contribution =
            if reduce_axes.is_empty() { self.clone() } else { self.reduce(reduce_axes.as_slice(), ReductionKind::Sum) };

        // The reduction leaves the kept axes in this cotangent's axis order but an explicit broadcast may have permuted
        // them relative to `target`, so compute the permutation that restores `target`'s axis order and transpose only
        // when it is not the identity.
        let mut kept_axes_by_value = kept_axes.clone();
        kept_axes_by_value.sort_by_key(|(_, value_axis)| *value_axis);
        let permutation = kept_axes
            .iter()
            .map(|kept| kept_axes_by_value.iter().position(|candidate| candidate == kept).unwrap())
            .collect::<Vec<_>>();
        if permutation.iter().enumerate().any(|(axis, position)| axis != *position) {
            contribution = Transpose::transpose(&contribution, permutation)?;
        }

        // Reinstate the stretched axes (reduced away entirely above) as unit axes so the shape matches `target`.
        if contribution.r#type().shape() != target.shape() {
            contribution = contribution.reshape(target.shape().clone())?;
        }

        // Convert back to the operand's cotangent element type (the adjoint of the element-type promotion the
        // broadcast side performed).
        if contribution.r#type().data_type() != target.data_type() {
            contribution = contribution.convert_element_type(target.data_type())?;
        }

        // Redistribute the contribution onto the operand cotangent's placement. Like in `align_tangent`, this is a
        // semantic redistribution that must be staged explicitly rather than relabeled by backend lowering.
        if contribution.r#type().sharding() != target.sharding()
            && let Some(sharding) = target.sharding()
        {
            contribution = contribution.reshard(sharding);
        }

        // Pin any remaining metadata-only difference (e.g., a layout that no step above can attach) with an
        // axis-identity broadcast, which carries its full requested output type.
        if contribution.r#type().as_ref() != target {
            let output_axes = (0..target.rank()).collect::<Vec<_>>();
            contribution = contribution.broadcast(target.clone(), output_axes.as_slice())?;
        }

        // As a defensive final check, report a mismatch instead of returning a mistyped cotangent, because a wrong
        // cotangent type would surface much later as a confusing accumulation or interpretation error.
        if contribution.r#type().as_ref() != target {
            return Err(TypeError {
                message: format!(
                    "unaligned cotangent type {} does not match required input cotangent type {}",
                    contribution.r#type(),
                    target,
                ),
            }
            .into());
        }

        Ok(contribution)
    }
}

// TODO(eaplatanios): Review from here onwards.

/// Operands handed to the tangent term of a unary elementwise JVP rule by [`unary_elementwise_jvp`]. The input primal
/// and tangent arrive already converted and broadcast to the tangent target type, while the primal output stays
/// at its own (possibly narrower) type so terms can reuse it when no widening is required.
pub(crate) struct UnaryTangentOperands<'a, T: DifferentiableType, V: ElementwiseDerivativeAlignment<T>> {
    /// Input primal converted and broadcast to the tangent target type.
    pub input: V,

    /// Live input tangent converted and broadcast to the tangent target type.
    pub tangent: V,

    /// Primal output of the operation, at its own type.
    pub primal: &'a V,

    /// Tangent target type of the primal output.
    pub target: &'a T,
}

impl<T: DifferentiableType, V: ElementwiseDerivativeAlignment<T>> UnaryTangentOperands<'_, T, V> {
    /// Returns the primal output reused as a tangent coefficient when its type already equals the tangent
    /// target, and otherwise recomputes the coefficient from the widened input via `recompute`. Reusing the primal
    /// keeps rules like `exp` and `sqrt` from staging a duplicate primal computation in tangent programs, while the
    /// widened recompute path preserves precision when the primal output lives in a narrower representation than its
    /// tangent (e.g., an `f8e8m0fnu` primal with an `f32` tangent).
    pub fn primal_coefficient(
        &self,
        recompute: impl FnOnce(&V) -> Result<V, ProgramError>,
    ) -> Result<V, DifferentiationError> {
        if self.primal.r#type().as_ref() == self.target { Ok(self.primal.clone()) } else { Ok(recompute(&self.input)?) }
    }
}

/// Operands handed to the per-side tangent terms of a binary elementwise JVP rule by [`binary_elementwise_jvp`].
/// Primal accessors convert and broadcast lazily so a term only stages the conversions it actually consumes.
pub(crate) struct BinaryTangentOperands<'a, T: DifferentiableType, V: ElementwiseDerivativeAlignment<T>> {
    /// Dual of the left operand.
    left: &'a DifferentiationDual<V>,

    /// Dual of the right operand.
    right: &'a DifferentiationDual<V>,

    /// Tangent target type of the primal output.
    target: &'a T,
}

impl<T: DifferentiableType, V: ElementwiseDerivativeAlignment<T>> BinaryTangentOperands<'_, T, V> {
    /// Returns the left operand's primal converted and broadcast to the tangent target type.
    pub fn left(&self) -> Result<V, DifferentiationError> {
        self.left.primal().align_tangent(self.target)
    }

    /// Returns the right operand's primal converted and broadcast to the tangent target type.
    pub fn right(&self) -> Result<V, DifferentiationError> {
        self.right.primal().align_tangent(self.target)
    }
}

/// Runs the shared unary elementwise JVP scaffolding: computes the primal via `primal`, resolves the tangent target
/// type, converts the operand primal and any live operand tangent into that representation, and delegates the
/// mathematical content to `tangent`. A structural-zero operand tangent propagates as a structural zero — including
/// through outputs without a tangent space (e.g., integer outputs), which only reject *live* tangents.
///
/// # Parameters
///
///   - `operation_name`: Canonical operation name used in the returned errors.
///   - `inputs`: The operand duals passed to the JVP rule.
///   - `primal`: Computes the primal output from the operand primal.
///   - `tangent`: Computes the live output tangent from the prepared [`UnaryTangentOperands`].
pub(crate) fn unary_elementwise_jvp<T, V>(
    operation_name: &str,
    inputs: &[DifferentiationDual<V>],
    primal: impl FnOnce(&V) -> Result<V, ProgramError>,
    tangent: impl FnOnce(UnaryTangentOperands<'_, T, V>) -> Result<V, DifferentiationError>,
) -> Result<Vec<DifferentiationDual<V>>, DifferentiationError>
where
    T: DifferentiableType,
    V: ElementwiseDerivativeAlignment<T>,
{
    check_count!("input", inputs, 1, ProgramError);
    let input = &inputs[0];
    let primal = primal(input.primal())?;
    let target = primal.r#type().tangent();
    let output_tangent = match input.tangent() {
        MaybeZero::Zero(_) => MaybeZero::Zero(target),
        MaybeZero::Value(_) if target.is_zero_space() => {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("'{operation_name}' output type {} has no tangent space", primal.r#type()),
            }
            .into());
        }
        MaybeZero::Value(input_tangent) => {
            let operands = UnaryTangentOperands {
                input: input.primal().align_tangent(&target)?,
                tangent: input_tangent.align_tangent(&target)?,
                primal: &primal,
                target: &target,
            };
            MaybeZero::Value(tangent(operands)?)
        }
    };
    Ok(vec![DifferentiationDual::new(primal, output_tangent)?])
}

/// Runs the shared binary elementwise JVP scaffolding: computes the primal via `primal`, resolves the tangent target
/// type, converts each live operand tangent into that representation, and delegates the mathematical content to
/// the per-side terms, invoking each term only when that side's tangent is live so terms stay lazy in staged
/// programs. Live contributions are summed; structural-zero operand tangents propagate as structural zeros —
/// including through outputs without a tangent space (e.g., integer outputs), which only reject *live* tangents.
///
/// # Parameters
///
///   - `operation_name`: Canonical operation name used in the returned errors.
///   - `inputs`: The operand duals passed to the JVP rule.
///   - `primal`: Computes the primal output from the left and right operand primals.
///   - `left_term`: Computes the left operand's tangent contribution from the prepared [`BinaryTangentOperands`] and
///     the left operand's live tangent (already converted and broadcast to the tangent target type).
///   - `right_term`: Computes the right operand's tangent contribution from the prepared [`BinaryTangentOperands`]
///     and the right operand's live tangent (already converted and broadcast to the tangent target type).
pub(crate) fn binary_elementwise_jvp<T, V>(
    operation_name: &str,
    inputs: &[DifferentiationDual<V>],
    primal: impl FnOnce(&V, &V) -> Result<V, ProgramError>,
    left_term: impl FnOnce(&BinaryTangentOperands<'_, T, V>, V) -> Result<V, DifferentiationError>,
    right_term: impl FnOnce(&BinaryTangentOperands<'_, T, V>, V) -> Result<V, DifferentiationError>,
) -> Result<Vec<DifferentiationDual<V>>, DifferentiationError>
where
    T: DifferentiableType,
    V: std::ops::Add<Output = V> + ElementwiseDerivativeAlignment<T>,
{
    check_count!("input", inputs, 2, ProgramError);
    let left = &inputs[0];
    let right = &inputs[1];
    let primal = primal(left.primal(), right.primal())?;
    let target = primal.r#type().tangent();
    if target.is_zero_space() {
        if left.tangent().as_value().is_some() || right.tangent().as_value().is_some() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("'{operation_name}' output type {} has no tangent space", primal.r#type()),
            }
            .into());
        }
        return Ok(vec![DifferentiationDual::new(primal, MaybeZero::Zero(target))?]);
    }
    let operands = BinaryTangentOperands { left, right, target: &target };
    let left_contribution = left
        .tangent()
        .as_value()
        .map(|tangent| left_term(&operands, tangent.align_tangent(&target)?))
        .transpose()?;
    let right_contribution = right
        .tangent()
        .as_value()
        .map(|tangent| right_term(&operands, tangent.align_tangent(&target)?))
        .transpose()?;
    let output_tangent = match (left_contribution, right_contribution) {
        (Some(left), Some(right)) => MaybeZero::Value(left + right),
        (Some(term), None) | (None, Some(term)) => MaybeZero::Value(term),
        (None, None) => MaybeZero::Zero(target),
    };
    Ok(vec![DifferentiationDual::new(primal, output_tangent)?])
}
