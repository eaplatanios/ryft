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
use crate::operations::manipulation::{ConvertElementType, LegacyBroadcast, Reshape, Transpose};
use crate::operations::math::{Reduce, ReductionKind};
use crate::operations::sharding::Reshard;
use crate::programs::ProgramError;
use crate::programs::atoms::MaybeZero;
use crate::programs::operations::Operation;
use crate::programs::types::TypeError;
use crate::programs::values::Value;
use crate::types::{ArrayType, DataType, Dimension};

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

impl<V: Value<Type = ArrayType> + LegacyBroadcast + ConvertElementType + Reshape + Transpose + Reshard + Reduce>
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
            return Err(TypeError::invalid(format!(
                "cannot align tangent type {} to output type {}",
                value.r#type(),
                target,
            ))
            .into());
        }

        let offset = target.rank() - rank;
        let output_axes = (0..rank).map(|axis| axis + offset).collect::<Vec<_>>();
        value = value.legacy_broadcast(target.clone(), output_axes.as_slice())?;

        // The broadcasting operation carries the requested output type, but changing an explicit/manual sharding is a
        // semantic redistribution rather than a metadata-only broadcast. Here we stage that transition explicitly so
        // that backend lowering cannot silently relabel the tangent when the primal result is placed differently from
        // this operand.
        if requires_reshard && let Some(sharding) = target.sharding() {
            value = value.reshard(sharding);
        }

        Ok(value)
    }

    fn unalign_cotangent(&self, target: &ArrayType) -> Result<Self, DifferentiationError> {
        let offset = self.r#type().rank().checked_sub(target.rank()).ok_or_else(|| {
            TypeError::invalid(format!(
                "cannot unalign cotangent type {} to input cotangent type {}",
                self.r#type(),
                target,
            ))
        })?;
        let output_axes = (0..target.rank()).map(|axis| axis + offset).collect::<Vec<_>>();
        self.unalign_cotangent_along(target, output_axes.as_slice())
    }
}

/// [`ArrayType`]-typed [`Value`] whose cotangents can additionally be _unaligned_ through the adjoint of an *explicit*
/// broadcast (e.g., using [`LegacyBroadcast`]) that placed the operand's axes at arbitrary result positions. This extends
/// [`ElementwiseDerivativeAlignment`] as a separate trait because axis placement is a concept that is specific to
/// [`ArrayType`]-typed values. Scalar types have no axes and participate only in implicit suffix-aligned alignment
/// maps.
pub trait BroadcastDerivativeAlignment: ElementwiseDerivativeAlignment<ArrayType> {
    /// Unaligns this cotangent [`Value`] back to `target` by applying the adjoint of an explicit broadcast that
    /// mapped each axis of `target` to the axis of this [`Value`] named by the corresponding entry of `output_axes`.
    /// Contributions along this [`Value`]'s remaining axes are sum-reduced away, the kept axes are permuted back
    /// into `target`'s axis order, and the element type, sharding, and stretched unit axes are restored.
    /// [`ElementwiseDerivativeAlignment::unalign_cotangent`] is the implicit special case that suffix-aligns
    /// the trailing axes.
    fn unalign_cotangent_along(&self, target: &ArrayType, output_axes: &[usize]) -> Result<Self, DifferentiationError>;
}

impl<V: Value<Type = ArrayType> + LegacyBroadcast + ConvertElementType + Reshape + Transpose + Reshard + Reduce>
    BroadcastDerivativeAlignment for V
{
    fn unalign_cotangent_along(&self, target: &ArrayType, output_axes: &[usize]) -> Result<Self, DifferentiationError> {
        // The broadcast being transposed mapped each `target` axis to the axis of this cotangent named by the
        // corresponding `output_axes` entry, so the mapping must name one in-range axis per `target` axis. Anything
        // else means the caller's axis mapping and the operand type disagree.
        let value_type = self.r#type();
        if output_axes.len() != target.rank() || output_axes.iter().any(|axis| *axis >= value_type.rank()) {
            return Err(TypeError::invalid(format!(
                "cannot unalign cotangent type {} to input cotangent type {} using output axes {:?}",
                value_type, target, output_axes,
            ))
            .into());
        }

        // Classify each `target` axis. An axis whose extent survived the broadcast unchanged is *kept* (i.e., its
        // cotangent flows straight through), while an axis the broadcast stretched from extent one is dropped here and
        // summed over below, with its unit extent restored by the later reshaping (i.e., the adjoint of stretching is
        // summation). Any other extent mismatch means that the mapping never described a valid broadcast.
        let mut kept_axes = Vec::with_capacity(target.rank());
        for (target_axis, &output_axis) in output_axes.iter().enumerate() {
            let target_dimension = target.dimension(target_axis);
            let value_dimension = value_type.dimension(output_axis);
            if target_dimension != value_dimension {
                if target_dimension != Dimension::Static(1) {
                    return Err(TypeError::invalid(format!(
                        "cannot unalign cotangent axis {} of size {} to input axis {} of size {}",
                        output_axis, value_dimension, target_axis, target_dimension,
                    ))
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
            contribution = contribution.legacy_broadcast(target.clone(), output_axes.as_slice())?;
        }

        // As a defensive final check, report a mismatch instead of returning a mistyped cotangent, because a wrong
        // cotangent type would surface much later as a confusing accumulation or interpretation error.
        if contribution.r#type().as_ref() != target {
            return Err(TypeError::invalid(format!(
                "unaligned cotangent type {} does not match required input cotangent type {}",
                contribution.r#type(),
                target,
            ))
            .into());
        }

        Ok(contribution)
    }
}

/// Represents operands handed to the tangent term of a unary elementwise JVP rule by [`unary_elementwise_jvp`]. The
/// input accessors convert and broadcast lazily to the output tangent [`Type`](crate::Type), while the output primal
/// stays at its own (possibly narrower) type so terms can reuse it when no widening is required.
pub struct UnaryElementwiseJvpOperands<'o, T: DifferentiableType, V: ElementwiseDerivativeAlignment<T>, F> {
    /// Input primal [`Value`] at its original [`Type`](crate::Type).
    input_primal: &'o V,

    /// Input tangent [`Value`] at its original [`Type`](crate::Type).
    input_tangent: &'o V,

    /// Primal output [`Value`] of the operation, at its own [`Type`](crate::Type).
    output_primal: &'o V,

    /// Tangent target [`Type`](crate::Type) of the primal output [`Value`].
    output_tangent_type: &'o T,

    /// Function that evaluates the operation's primal output from its input primal.
    evaluate_primal: &'o F,
}

impl<T: DifferentiableType, V: ElementwiseDerivativeAlignment<T>, F: Fn(&V) -> Result<V, ProgramError>>
    UnaryElementwiseJvpOperands<'_, T, V, F>
{
    /// Returns the input primal [`Value`] converted and broadcast to the output tangent [`Type`](crate::Type).
    #[inline]
    pub fn input_primal(&self) -> Result<V, DifferentiationError> {
        self.input_primal.align_tangent(self.output_tangent_type)
    }

    /// Returns the input tangent [`Value`] converted and broadcast to the output tangent [`Type`](crate::Type).
    #[inline]
    pub fn input_tangent(&self) -> Result<V, DifferentiationError> {
        self.input_tangent.align_tangent(self.output_tangent_type)
    }

    /// Returns the operation's output primal [`Value`] evaluated at the output tangent [`Type`](crate::Type). When the
    /// original output primal already has that type, this reuses it directly. Otherwise, it re-evaluates the operation
    /// from the converted input primal so that nonlinear rounding at the narrower primal type does not affect the
    /// tangent computation.
    #[inline]
    pub fn output_primal_at_tangent_type(&self) -> Result<V, DifferentiationError> {
        if self.output_primal.r#type().as_ref() == self.output_tangent_type {
            Ok(self.output_primal.clone())
        } else {
            Ok((self.evaluate_primal)(&self.input_primal()?)?)
        }
    }
}

/// Computes the Jacobian-Vector Product (JVP) for a unary elementwise [`Operation`]. This function computes the output
/// primal via `primal_fn`, resolves the tangent target [`Type`](crate::Type), converts the operand primal and any live
/// operand tangent into that representation, and delegates the mathematical content to `tangent_fn`. The prepared
/// operands retain `primal_fn` so that a rule can re-evaluate the output at the tangent type without repeating the
/// evaluator at the call site. A structural-zero operand tangent propagates as a structural zero, including through
/// outputs without a tangent space (e.g., integer outputs), which only reject *live* tangents.
///
/// # Parameters
///
///   - `operation`: [`Operation`] whose JVP rule is being evaluated.
///   - `inputs`: Input [`DifferentiationDual`]s passed to the JVP rule.
///   - `primal_fn`: Function that computes the primal output from the operand primal.
///   - `tangent_fn`: Function that computes the output tangent from the prepared [`UnaryElementwiseJvpOperands`].
pub fn unary_elementwise_jvp<
    T: DifferentiableType,
    V: ElementwiseDerivativeAlignment<T>,
    O: Operation<Type = T>,
    PrimalFn: Fn(&V) -> Result<V, ProgramError>,
    TangentFn: FnOnce(UnaryElementwiseJvpOperands<'_, T, V, PrimalFn>) -> Result<V, DifferentiationError>,
>(
    operation: &O,
    inputs: &[DifferentiationDual<V>],
    primal_fn: PrimalFn,
    tangent_fn: TangentFn,
) -> Result<Vec<DifferentiationDual<V>>, DifferentiationError> {
    check_count!("input", inputs, 1, ProgramError);
    let input = &inputs[0];
    let output_primal = primal_fn(input.primal())?;
    let target = output_primal.r#type().tangent();
    let output_tangent = match input.tangent() {
        MaybeZero::Zero(_) => MaybeZero::Zero(target),
        MaybeZero::Value(_) if target.is_zero_space() => {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("'{}' output type {} has no tangent space", operation.name(), output_primal.r#type()),
            }
            .into());
        }
        MaybeZero::Value(input_tangent) => {
            let operands = UnaryElementwiseJvpOperands {
                input_primal: input.primal(),
                input_tangent,
                output_primal: &output_primal,
                output_tangent_type: &target,
                evaluate_primal: &primal_fn,
            };
            MaybeZero::Value(tangent_fn(operands)?)
        }
    };
    Ok(vec![DifferentiationDual::new(output_primal, output_tangent)?])
}

/// Represents operands handed to the per-side tangent terms of a binary elementwise JVP rule by
/// [`binary_elementwise_jvp`]. Primal accessors convert and broadcast lazily so that a term only stages
/// the conversions that it actually consumes.
pub struct BinaryElementwiseJvpOperands<'o, T: DifferentiableType, V: ElementwiseDerivativeAlignment<T>> {
    /// Primal [`Value`] of the left operand.
    left_primal: &'o V,

    /// Primal [`Value`] of the right operand.
    right_primal: &'o V,

    /// Tangent target [`Type`](crate::Type) of the primal output [`Value`].
    output_tangent_type: &'o T,
}

impl<T: DifferentiableType, V: ElementwiseDerivativeAlignment<T>> BinaryElementwiseJvpOperands<'_, T, V> {
    /// Returns the left input's primal [`Value`] converted and broadcast to the tangent target [`Type`](crate::Type).
    pub fn left_primal(&self) -> Result<V, DifferentiationError> {
        self.left_primal.align_tangent(self.output_tangent_type)
    }

    /// Returns the right input's primal [`Value`] converted and broadcast to the tangent target [`Type`](crate::Type).
    pub fn right_primal(&self) -> Result<V, DifferentiationError> {
        self.right_primal.align_tangent(self.output_tangent_type)
    }
}

/// Computes the Jacobian-Vector Product (JVP) for a binary elementwise [`Operation`]. This function computes the output
/// primal via `primal_fn`, resolves the tangent target [`Type`](crate::Type), converts each live operand tangent into
/// that representation, and delegates the mathematical content to the per-side terms, invoking each term only when that
/// side's tangent is live so terms stay lazy in staged programs. Live contributions are summed, and structural-zero
/// operand tangents propagate as structural zeros,including through outputs without a tangent space (e.g., integer
/// outputs), which only reject *live* tangents.
///
/// # Parameters
///
///   - `operation`: [`Operation`] whose JVP rule is being evaluated.
///   - `inputs`: Input [`DifferentiationDual`]s passed to the JVP rule.
///   - `primal_fn`: Function that computes the primal output from the left and right operand primals.
///   - `left_tangent_term_fn`: Function that computes the left operand's tangent contribution from the prepared
///     [`BinaryElementwiseJvpOperands`] and the left operand's live tangent (already converted and broadcast to the
///     tangent target type).
///   - `right_tangent_term_fn`: Function that computes the right operand's tangent contribution from the prepared
///     [`BinaryElementwiseJvpOperands`] and the right operand's live tangent (already converted and broadcast to the
///     tangent target type).
pub fn binary_elementwise_jvp<
    T: DifferentiableType,
    V: std::ops::Add<Output = V> + ElementwiseDerivativeAlignment<T>,
    O: Operation<Type = T>,
    PrimalFn: FnOnce(&V, &V) -> Result<V, ProgramError>,
    LeftTangentTermFn: FnOnce(&BinaryElementwiseJvpOperands<'_, T, V>, V) -> Result<V, DifferentiationError>,
    RightTangentTermFn: FnOnce(&BinaryElementwiseJvpOperands<'_, T, V>, V) -> Result<V, DifferentiationError>,
>(
    operation: &O,
    inputs: &[DifferentiationDual<V>],
    primal_fn: PrimalFn,
    left_tangent_term_fn: LeftTangentTermFn,
    right_tangent_term_fn: RightTangentTermFn,
) -> Result<Vec<DifferentiationDual<V>>, DifferentiationError> {
    check_count!("input", inputs, 2, ProgramError);
    let left = &inputs[0];
    let right = &inputs[1];
    let primal = primal_fn(left.primal(), right.primal())?;
    let target = primal.r#type().tangent();
    if target.is_zero_space() {
        if left.tangent().as_value().is_some() || right.tangent().as_value().is_some() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("'{}' output type {} has no tangent space", operation.name(), primal.r#type()),
            }
            .into());
        }
        return Ok(vec![DifferentiationDual::new(primal, MaybeZero::Zero(target))?]);
    }
    let operands = BinaryElementwiseJvpOperands {
        left_primal: left.primal(),
        right_primal: right.primal(),
        output_tangent_type: &target,
    };
    let left_contribution = left
        .tangent()
        .as_value()
        .map(|tangent| left_tangent_term_fn(&operands, tangent.align_tangent(&target)?))
        .transpose()?;
    let right_contribution = right
        .tangent()
        .as_value()
        .map(|tangent| right_tangent_term_fn(&operands, tangent.align_tangent(&target)?))
        .transpose()?;
    let output_tangent = match (left_contribution, right_contribution) {
        (Some(left), Some(right)) => MaybeZero::Value(left + right),
        (Some(term), None) | (None, Some(term)) => MaybeZero::Value(term),
        (None, None) => MaybeZero::Zero(target),
    };
    Ok(vec![DifferentiationDual::new(primal, output_tangent)?])
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::differentiation::reverse::ReverseModeDifferentiate;
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::manipulation::ConvertElementType;
    use crate::operations::math::{AddOperation, SinOperation};
    use crate::programs::atoms::MaybeZero;
    use crate::programs::types::Typed;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::types::{ArrayType, DataType, Dimension, Shape};

    use super::*;

    #[test]
    fn test_scalar_elementwise_derivative_alignment() {
        let value = Scalar::from(1.5f64);
        assert_eq!(value.align_tangent(&DataType::F64), Ok(value));
        assert_eq!(value.align_tangent(&DataType::F32), Ok(Scalar::from(1.5f32)));

        let value = Scalar::from(1.5f32);
        assert_eq!(value.unalign_cotangent(&DataType::F32), Ok(value));
        assert_eq!(value.unalign_cotangent(&DataType::F64), Ok(Scalar::from(1.5f64)));
    }

    #[test]
    fn test_array_elementwise_derivative_alignment() {
        let scalar = Array::from_f64s(ArrayType::scalar(DataType::F32), vec![2.0]);
        let target = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        assert_eq!(scalar.align_tangent(&target), Ok(Array::from_f64s(target, vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0])),);

        let cotangent = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        let target = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]));
        assert_eq!(cotangent.unalign_cotangent(&target), Ok(Array::from_f64s(target, vec![5.0, 7.0, 9.0])),);

        let value = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let target = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        assert!(matches!(
            value.align_tangent(&target),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "cannot align tangent type f64[2, 3] to output type f64[3]",
        ));
    }

    #[test]
    fn test_broadcast_derivative_alignment() {
        let cotangent = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        let target = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        assert_eq!(
            cotangent.unalign_cotangent_along(&target, &[1, 0]),
            Ok(Array::from_f64s(target, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0])),
        );

        let cotangent = Array::from_f64s(
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(3), Dimension::Static(2), Dimension::Static(4)]),
            ),
            (1..=24).map(|value| value as f64).collect(),
        );
        let target = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]));
        assert_eq!(
            cotangent.unalign_cotangent_along(&target, &[1, 2]),
            Ok(Array::from_f64s(target, vec![126.0, 174.0]))
        );

        let target = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]));
        assert!(matches!(
            cotangent.unalign_cotangent_along(&target, &[3]),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message
                    == "cannot unalign cotangent type f64[3, 2, 4] to input cotangent type f64[2] using output axes [3]",
        ));
    }

    #[test]
    fn test_unary_elementwise_jvp() {
        #[derive(Clone)]
        struct BooleanOutputOperation;

        impl Operation for BooleanOutputOperation {
            type Type = DataType;

            fn name(&self) -> &'static str {
                "boolean_output"
            }

            fn infer_output_types(
                &self,
                _input_types: &[DataType],
                _region_interfaces: &[crate::programs::regions::RegionInterface<DataType>],
            ) -> Result<Vec<DataType>, TypeError> {
                Ok(vec![DataType::Boolean])
            }
        }

        let tangent_calls = Cell::new(0);
        let outputs = unary_elementwise_jvp(
            &SinOperation::<DataType>::new(),
            &[DifferentiationDual::new_with_zero_tangent(Scalar::from(2.0f64))],
            |input| Ok(*input),
            |_| {
                tangent_calls.set(tangent_calls.get() + 1);
                Ok(Scalar::from(1.0f64))
            },
        )
        .unwrap();
        assert_eq!(outputs[0].primal(), &Scalar::from(2.0f64));
        assert!(matches!(outputs[0].tangent(), MaybeZero::Zero(DataType::F64)));
        assert_eq!(tangent_calls.get(), 0);

        let primal_evaluations = Cell::new(0);
        let outputs = unary_elementwise_jvp(
            &SinOperation::<DataType>::new(),
            &[DifferentiationDual::new(Scalar::from(2.0f64), Scalar::from(3.0f64)).unwrap()],
            |input| {
                primal_evaluations.set(primal_evaluations.get() + 1);
                Ok(*input)
            },
            |operands| {
                tangent_calls.set(tangent_calls.get() + 1);
                Ok(operands.output_primal_at_tangent_type()? * operands.input_tangent()?)
            },
        )
        .unwrap();
        assert_eq!(outputs[0].primal(), &Scalar::from(2.0f64));
        assert!(matches!(outputs[0].tangent(), MaybeZero::Value(value) if value == &Scalar::from(6.0f64)));
        assert_eq!(tangent_calls.get(), 1);
        assert_eq!(primal_evaluations.get(), 1);

        let primal_evaluations = Cell::new(0);
        let input_primal = Scalar::from(2.0f32).convert_element_type(DataType::F8E8M0FNU).unwrap();
        let outputs = unary_elementwise_jvp(
            &SinOperation::<DataType>::new(),
            &[DifferentiationDual::new(input_primal, Scalar::from(3.0f32)).unwrap()],
            |input| {
                primal_evaluations.set(primal_evaluations.get() + 1);
                Ok(*input)
            },
            |operands| operands.output_primal_at_tangent_type(),
        )
        .unwrap();
        assert_eq!(outputs[0].primal(), &input_primal);
        assert!(matches!(outputs[0].tangent(), MaybeZero::Value(value) if value == &Scalar::from(2.0f32)));
        assert_eq!(primal_evaluations.get(), 2);

        let outputs = unary_elementwise_jvp(
            &BooleanOutputOperation,
            &[DifferentiationDual::new_with_zero_tangent(Scalar::from(2.0f64))],
            |_| Ok(Scalar::Bool(true)),
            |_| -> Result<Scalar, DifferentiationError> {
                panic!("zero-space output invoked its tangent function for a structural-zero input tangent")
            },
        )
        .unwrap();
        assert_eq!(outputs[0].primal(), &Scalar::Bool(true));
        assert!(matches!(outputs[0].tangent(), MaybeZero::Zero(DataType::Zero)));

        assert!(matches!(
            unary_elementwise_jvp(
                &BooleanOutputOperation,
                &[DifferentiationDual::new(Scalar::from(2.0f64), Scalar::from(3.0f64)).unwrap()],
                |_| Ok(Scalar::Bool(true)),
                |_| -> Result<Scalar, DifferentiationError> { panic!("zero-space output invoked its tangent function") },
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "'boolean_output' output type bool has no tangent space",
        ));
        assert!(matches!(
            unary_elementwise_jvp(
                &SinOperation::<DataType>::new(),
                &[],
                |input: &Scalar| Ok(*input),
                |_| Ok(Scalar::from(1.0f64)),
            ),
            Err(DifferentiationError::Program(ProgramError::InvalidInputCount { expected: 1, actual: 0 })),
        ));
    }

    #[test]
    fn test_binary_elementwise_jvp() {
        let left_calls = Cell::new(0);
        let right_calls = Cell::new(0);
        let left_primal = Scalar::from(2.0f64);
        let right_primal = Scalar::from(5.0f64);
        let compare = CompareOperation::new(ComparisonDirection::LessThan);

        let outputs = binary_elementwise_jvp(
            &AddOperation::<DataType>::new(),
            &[
                DifferentiationDual::new_with_zero_tangent(left_primal),
                DifferentiationDual::new_with_zero_tangent(right_primal),
            ],
            |left, right| Ok(*left + *right),
            |_, tangent| {
                left_calls.set(left_calls.get() + 1);
                Ok(tangent)
            },
            |_, tangent| {
                right_calls.set(right_calls.get() + 1);
                Ok(tangent)
            },
        )
        .unwrap();
        assert_eq!(outputs[0].primal(), &Scalar::from(7.0f64));
        assert!(matches!(outputs[0].tangent(), MaybeZero::Zero(DataType::F64)));
        assert_eq!((left_calls.get(), right_calls.get()), (0, 0));

        let outputs = binary_elementwise_jvp(
            &compare,
            &[
                DifferentiationDual::new_with_zero_tangent(left_primal),
                DifferentiationDual::new_with_zero_tangent(right_primal),
            ],
            |left, right| Ok(Scalar::Bool(left < right)),
            |_, _| -> Result<Scalar, DifferentiationError> {
                panic!("zero-space output invoked its left tangent term function for structural-zero input tangents")
            },
            |_, _| -> Result<Scalar, DifferentiationError> {
                panic!("zero-space output invoked its right tangent term function for structural-zero input tangents")
            },
        )
        .unwrap();
        assert_eq!(outputs[0].primal(), &Scalar::Bool(true));
        assert!(matches!(outputs[0].tangent(), MaybeZero::Zero(DataType::Zero)));

        let outputs = binary_elementwise_jvp(
            &AddOperation::<DataType>::new(),
            &[
                DifferentiationDual::new(left_primal, Scalar::from(3.0f64)).unwrap(),
                DifferentiationDual::new_with_zero_tangent(right_primal),
            ],
            |left, right| Ok(*left + *right),
            |_, tangent| {
                left_calls.set(left_calls.get() + 1);
                Ok(tangent)
            },
            |_, tangent| {
                right_calls.set(right_calls.get() + 1);
                Ok(tangent)
            },
        )
        .unwrap();
        assert!(matches!(outputs[0].tangent(), MaybeZero::Value(value) if value == &Scalar::from(3.0f64)));
        assert_eq!((left_calls.get(), right_calls.get()), (1, 0));

        let outputs = binary_elementwise_jvp(
            &AddOperation::<DataType>::new(),
            &[
                DifferentiationDual::new_with_zero_tangent(left_primal),
                DifferentiationDual::new(right_primal, Scalar::from(4.0f64)).unwrap(),
            ],
            |left, right| Ok(*left + *right),
            |_, tangent| {
                left_calls.set(left_calls.get() + 1);
                Ok(tangent)
            },
            |_, tangent| {
                right_calls.set(right_calls.get() + 1);
                Ok(tangent)
            },
        )
        .unwrap();
        assert!(matches!(outputs[0].tangent(), MaybeZero::Value(value) if value == &Scalar::from(4.0f64)));
        assert_eq!((left_calls.get(), right_calls.get()), (1, 1));

        let outputs = binary_elementwise_jvp(
            &AddOperation::<DataType>::new(),
            &[
                DifferentiationDual::new(left_primal, Scalar::from(3.0f64)).unwrap(),
                DifferentiationDual::new(right_primal, Scalar::from(4.0f64)).unwrap(),
            ],
            |left, right| Ok(*left + *right),
            |operands, tangent| {
                left_calls.set(left_calls.get() + 1);
                Ok(operands.right_primal()? * tangent)
            },
            |operands, tangent| {
                right_calls.set(right_calls.get() + 1);
                Ok(operands.left_primal()? * tangent)
            },
        )
        .unwrap();
        assert!(matches!(outputs[0].tangent(), MaybeZero::Value(value) if value == &Scalar::from(23.0f64)));
        assert_eq!((left_calls.get(), right_calls.get()), (2, 2));

        assert!(matches!(
            binary_elementwise_jvp(
                &compare,
                &[
                    DifferentiationDual::new(left_primal, Scalar::from(3.0f64)).unwrap(),
                    DifferentiationDual::new_with_zero_tangent(right_primal),
                ],
                |left, right| Ok(Scalar::Bool(left < right)),
                |_, _| -> Result<Scalar, DifferentiationError> {
                    panic!("zero-space output invoked its left tangent term function")
                },
                |_, _| -> Result<Scalar, DifferentiationError> {
                    panic!("zero-space output invoked its right tangent term function")
                },
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "'compare' output type bool has no tangent space",
        ));
        assert!(matches!(
            binary_elementwise_jvp(
                &AddOperation::<DataType>::new(),
                &[DifferentiationDual::new_with_zero_tangent(left_primal)],
                |left, right| Ok(*left + *right),
                |_, tangent| Ok(tangent),
                |_, tangent| Ok(tangent),
            ),
            Err(DifferentiationError::Program(ProgramError::InvalidInputCount { expected: 2, actual: 1 })),
        ));
    }

    #[test]
    fn test_binary_elementwise_vjp_restores_each_input_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let replicated_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))
            .with_sharding(Sharding::replicated(mesh, 1))
            .unwrap();
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let (output, pullback) = context
            .vjp(
                |(left, right)| Ok(left + right),
                (
                    Array::from_f64s(sharded_type.clone(), vec![1.0, 2.0]),
                    Array::from_f64s(replicated_type.clone(), vec![3.0, 4.0]),
                ),
            )
            .unwrap();
        assert!(pullback.program().to_string().contains("reshard"));
        let (left, right) = pullback.apply(Array::from_f64s(output.r#type().into_owned(), vec![1.0, 1.0])).unwrap();
        assert_eq!(left.r#type().as_ref(), &sharded_type);
        assert_eq!(right.r#type().as_ref(), &replicated_type);
    }
}
