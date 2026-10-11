use ryft_core::macros::check_count;
use ryft_core::{
    ArrayType, CUMULATIVE_OPERATION_NAME, CumulativeKind, CumulativeOperation, DataType, Dimension, ReductionKind,
};
use ryft_mlir::dialects::{chlo, stable_hlo};
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Region, Value as MlirValue,
    ValueRef,
};

use crate::experimental::lowering::types::{
    lower_f64_constant_splat, lower_tensor_type, lower_unplaced_constant_output,
};
use crate::operations::complex::lower_principal_phase_to_mlir;
use crate::operations::exponential::lower_log_add_exp_to_mlir;
use crate::operations::reductions::{
    build_reduce_body_region, build_reduction_identity_constant, lower_product_combiner_to_mlir,
};
use crate::{LowerableOperation, LowerableOperationFamily, LoweringError, XlaLowerer};

/// Builds the scalar `tensor<{element_type}>` value seeding every window of a prefix scan's `reduce_window`, which is
/// the identity of the combining operator selected by `kind`, so that the padded positions of a window cannot change
/// its prefix.
fn build_cumulative_initial_value<'b, 'c: 'b, 't: 'c>(
    kind: CumulativeKind,
    element_type: DataType,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    match kind {
        CumulativeKind::Sum => {
            build_reduction_identity_constant(ReductionKind::Sum, element_type, block, context, location)
        }
        // A scalar `one` reaches the same real and complex constant synthesis that the `one` primitive lowers through,
        // which the reduction identities do not cover because no reduction kind multiplies.
        CumulativeKind::Product => {
            Ok(lower_unplaced_constant_output(&[ArrayType::scalar(element_type)], 1, block, context, location)?[0])
        }
        CumulativeKind::Max => {
            build_reduction_identity_constant(ReductionKind::Max, element_type, block, context, location)
        }
        CumulativeKind::Min => {
            build_reduction_identity_constant(ReductionKind::Min, element_type, block, context, location)
        }
        // Complex types pair negative infinity with a zero imaginary component, whose exponential is exactly zero,
        // rather than using their lowest value, whose imaginary component is infinite too.
        CumulativeKind::LogSumExp if element_type.is_complex() => {
            let part_type =
                ArrayType::scalar(if element_type == DataType::C64 { DataType::F32 } else { DataType::F64 });
            let part_tensor_type = lower_tensor_type(&part_type, context, location)?;
            let real =
                lower_f64_constant_splat(f64::NEG_INFINITY, &part_type, part_tensor_type, block, context, location)?;
            let imaginary = lower_f64_constant_splat(0.0, &part_type, part_tensor_type, block, context, location)?;
            Ok(block.append_operation(stable_hlo::complex(real, imaginary, location)?)?.result(0).unwrap().as_ref())
        }
        // Real formats with infinity use negative infinity; supported finite-only formats use their lowest value,
        // which remains an identity after each rounded pairwise combination regardless of prefix length.
        CumulativeKind::LogSumExp => {
            build_reduction_identity_constant(ReductionKind::Max, element_type, block, context, location)
        }
    }
}

/// Builds the scalar reducer body region of a prefix scan's `reduce_window` for the combining operator selected by
/// `kind`. Sum and extrema delegate to [`build_reduce_body_region`] except for one-bit integers, whose predicate
/// carriers require integer arithmetic and signed ordering. Delegation keeps other extrema on the same portable
/// total-order expansion that `reduce` emits. Product, log-sum-exp, and one-bit integer combiners build their own
/// region over the same pair of scalar block arguments.
fn build_cumulative_body_region<'c, 't>(
    kind: CumulativeKind,
    element_type: DataType,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ryft_mlir::DetachedRegion<'c, 't>, LoweringError> {
    let reduction_kind = match kind {
        CumulativeKind::Sum => Some(ReductionKind::Sum),
        CumulativeKind::Max => Some(ReductionKind::Max),
        CumulativeKind::Min => Some(ReductionKind::Min),
        CumulativeKind::Product | CumulativeKind::LogSumExp => None,
    };
    if let Some(reduction_kind) = reduction_kind
        && !matches!(element_type, DataType::I1 | DataType::U1)
    {
        return build_reduce_body_region(reduction_kind, element_type, context, location);
    }

    // Product, log-sum-exp, and one-bit integer combiners build their body over a pair of scalar block arguments.
    let scalar_tensor_type = lower_tensor_type(&ArrayType::scalar(element_type), context, location)?;
    let block = context.block(&[(scalar_tensor_type, location), (scalar_tensor_type, location)]);
    let mut region = context.region();
    let mut block_ref = region.append_block(block)?;
    let left = block_ref.argument(0)?.as_ref();
    let right = block_ref.argument(1)?.as_ref();
    // One-bit integers use a predicate carrier. Addition wraps modulo two (XOR), and signed I1 extrema
    // reverse the Boolean order because the set bit represents -1 rather than +1.
    let body_value = match kind {
        CumulativeKind::Sum if matches!(element_type, DataType::I1 | DataType::U1) => {
            block_ref.append_operation(stable_hlo::xor(left, right, location)?)?.result(0).unwrap().as_ref()
        }
        CumulativeKind::Max | CumulativeKind::Min if matches!(element_type, DataType::I1 | DataType::U1) => {
            let use_or = (kind == CumulativeKind::Min) == (element_type == DataType::I1);
            let combined = if use_or {
                block_ref.append_operation(stable_hlo::or(left, right, location)?)?
            } else {
                block_ref.append_operation(stable_hlo::and(left, right, location)?)?
            };
            combined.result(0).unwrap().as_ref()
        }
        CumulativeKind::LogSumExp if element_type.is_complex() => {
            // An operand whose real component is negative infinity has a zero exponential, so it leaves the other
            // operand unchanged. The complex expansion cannot see this when both operands are such values (e.g., the
            // `-∞ + 0i` initial values that pad every window), because its lexicographic difference is then NaN, so
            // the body selects around it, just like the eager kernel does.
            let scalar_type = ArrayType::scalar(element_type);
            let combined = lower_log_add_exp_to_mlir(left, right, &scalar_type, &mut block_ref, context, location)?;
            let part_type =
                ArrayType::scalar(if element_type == DataType::C64 { DataType::F32 } else { DataType::F64 });
            let part_tensor_type = lower_tensor_type(&part_type, context, location)?;
            let negative_infinity = lower_f64_constant_splat(
                f64::NEG_INFINITY,
                &part_type,
                part_tensor_type,
                &mut block_ref,
                context,
                location,
            )?;
            let mut negative_infinity_masks = Vec::with_capacity(2);
            for operand in [left, right] {
                let real =
                    block_ref.append_operation(stable_hlo::real(operand, location)?)?.result(0).unwrap().as_ref();
                let mask = block_ref.append_operation(stable_hlo::compare(
                    real,
                    negative_infinity,
                    stable_hlo::ComparisonDirection::Equal,
                    stable_hlo::ComparisonType::Float,
                    location,
                )?)?;
                negative_infinity_masks.push(mask.result(0).unwrap().as_ref());
            }
            let (left_is_negative_infinity, right_is_negative_infinity) =
                (negative_infinity_masks[0], negative_infinity_masks[1]);
            let guarded = block_ref
                .append_operation(stable_hlo::select(left_is_negative_infinity, right, combined, location)?)?
                .result(0)
                .unwrap()
                .as_ref();
            block_ref
                .append_operation(stable_hlo::select(right_is_negative_infinity, left, guarded, location)?)?
                .result(0)
                .unwrap()
                .as_ref()
        }
        CumulativeKind::LogSumExp => {
            lower_log_add_exp_to_mlir(left, right, &ArrayType::scalar(element_type), &mut block_ref, context, location)?
        }
        _ => lower_product_combiner_to_mlir(left, right, element_type, &mut block_ref, context, location)?,
    };
    block_ref.append_operation(stable_hlo::r#return(&[body_value], location)?)?;
    Ok(region)
}

/// Lowers one prefix-scan dispatch to a `stablehlo.reduce_window` whose window spans the whole scanned axis.
///
/// The window has extent `n` along the scanned axis and extent one everywhere else, and the scan direction is
/// expressed purely through the padding of that axis: a forward scan pads `(n - 1, 0)` so that output position `i`
/// covers input positions `0..=i`, and a reverse scan pads `(0, n - 1)` so that it covers `i..n`. The reducer body
/// and the initial value come from the scan's [`CumulativeKind`], whose identity is what makes the padded positions
/// inert.
///
/// The window form looks quadratic, but it is the same emission JAX's `cumred_reduce_window_impl` produces, and XLA's
/// `TryOptimizeAssociativeScan` rewriter recognizes the full-prefix-window shape and rewrites it into a
/// logarithmic-depth tree of strided `reduce-window`s. On the pinned XLA version, the rewriter matches both the forward
/// and the reverse padding (observed in the optimized HLO of a CUDA compilation).
///
/// On GPU targets (i.e., when `target_platform` is `"cuda"` or `"rocm"`), a forward [`CumulativeKind::Sum`] over a
/// statically shaped real numeric operand lowers to a
/// [`chlo.scan`](https://openxla.org/stablehlo/generated/chlo#chloscan_chloscanop) instead, which is what JAX's
/// `_cumred_gpu_lowering` emits. The scan carries one running sum per row of the unscanned axes, seeded with the sum
/// identity, and its body adds each slice to that carry. Reverse, Boolean, and complex sums keep the window form, as
/// they do in JAX. XLA imports the scan as an associative HLO `scan`, which leaves the choice of a scan implementation
/// to the GPU backend; the pinned XLA version has no dedicated (e.g., CUB-backed) prefix-sum kernel and expands it
/// into the same logarithmic-depth `reduce-window` tree as the window form.
pub(crate) fn lower_cumulative_to_mlir<'b, 'c: 'b, 't: 'c>(
    kind: CumulativeKind,
    axis: usize,
    reverse: bool,
    input_value: ValueRef<'b, 'c, 't>,
    output_array_type: &ArrayType,
    target_platform: Option<&str>,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, LoweringError> {
    // The scanned extent is static by construction: `ryft-core` type inference rejects a dynamic scanned axis, so a
    // dynamic extent here means the staged program bypassed that rule.
    let extent = match output_array_type.shape().dimensions().get(axis) {
        Some(Dimension::Static(extent)) => *extent,
        _ => {
            return Err(LoweringError::UnsupportedOp {
                op: format!("`{CUMULATIVE_OPERATION_NAME}` with kind `{kind}` over dynamically sized axis {axis}"),
            });
        }
    };
    // An empty scanned axis accumulates nothing, and a zero-extent window is not a valid `reduce_window`.
    if extent == 0 {
        return Ok(input_value);
    }
    let element_type = output_array_type.data_type();
    // A structural zero has no payload and both its prefix sums and products stay zero. In particular, its
    // product must not construct a multiplicative identity, which this element type cannot represent.
    if element_type == DataType::Zero {
        return Ok(input_value);
    }
    if kind == CumulativeKind::Sum
        && !reverse
        && matches!(target_platform, Some("cuda" | "rocm"))
        && element_type.is_real()
        && let Some(shape) = output_array_type.static_shape()
    {
        let mut carry_dimensions = shape.dimensions().to_vec();
        carry_dimensions.remove(axis);
        let carry_tensor_type =
            lower_tensor_type(&ArrayType::new_static(element_type, carry_dimensions.clone()), context, location)?;
        let identity = build_reduction_identity_constant(ReductionKind::Sum, element_type, block, context, location)?;
        let initial_value = match carry_dimensions.is_empty() {
            true => identity,
            false => block
                .append_operation(stable_hlo::broadcast(identity, carry_tensor_type, &[], location)?)?
                .result(0)
                .unwrap()
                .as_ref(),
        };

        // The body receives one slice of the operand and the carry, and returns their sum both as the output slice and
        // as the next carry.
        let mut body_region = context.region();
        let mut body_block =
            body_region.append_block(context.block(&[(carry_tensor_type, location), (carry_tensor_type, location)]))?;
        let slice = body_block.argument(0)?.as_ref();
        let carry = body_block.argument(1)?.as_ref();
        let sum = if matches!(element_type, DataType::I1 | DataType::U1) {
            // Predicate-carrier addition is OR, while one-bit integer addition wraps modulo two.
            body_block.append_operation(stable_hlo::xor(slice, carry, location)?)?
        } else {
            body_block.append_operation(stable_hlo::add(slice, carry, location)?)?
        };
        let sum = sum.result(0).unwrap().as_ref();
        body_block.append_operation(stable_hlo::r#return(&[sum, sum], location)?)?;
        let scan = block.append_operation(chlo::scan(
            &[input_value],
            &[initial_value],
            axis,
            None,
            false,
            Some(true),
            body_region,
            location,
        )?)?;
        return Ok(scan.result(0).unwrap().as_ref());
    }
    // Rounded pairwise identities stay neutral in any binary tree, independent of the window extent.
    if matches!(kind, CumulativeKind::LogSumExp) && matches!(element_type, DataType::F8E8M0FNU | DataType::F6E2M3FN) {
        return Err(LoweringError::UnsupportedOp {
            op: format!(
                "`{CUMULATIVE_OPERATION_NAME}` with kind `{kind}` over `{element_type}`, whose lowest value is not a \
                 `log_add_exp` identity",
            ),
        });
    }
    let initial_value = build_cumulative_initial_value(kind, element_type, block, context, location)?;
    let body_region = build_cumulative_body_region(kind, element_type, context, location)?;
    let rank = output_array_type.rank();
    let mut window_dimensions = vec![1; rank];
    window_dimensions[axis] = extent;
    let mut padding = vec![(0, 0); rank];
    padding[axis] = match reverse {
        true => (0, extent - 1),
        false => (extent - 1, 0),
    };
    let result = block.append_operation(stable_hlo::reduce_window(
        &[input_value],
        &[initial_value],
        &window_dimensions,
        None,
        None,
        None,
        Some(&padding),
        body_region,
        location,
    )?)?;
    let result = result.result(0).expect("stablehlo.reduce_window should return one result").as_ref();
    match kind == CumulativeKind::LogSumExp && element_type.is_complex() {
        true => lower_principal_phase_to_mlir(result, output_array_type, block, context, location),
        false => Ok(result),
    }
}

impl LowerableOperation for CumulativeOperation {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut XlaLowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        check_count!("output", output_types, 1, ProgramError);
        let value = lower_cumulative_to_mlir(
            self.kind(),
            self.axis(),
            self.reverse(),
            input_values[0],
            &output_types[0],
            lowerer.collective_state.target_platform(),
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        Ok(vec![value])
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayOperation, CumulativeOperation, DimensionBounds, DimensionVariable, Instruction, Placeholder, Shape,
    };

    use crate::experimental::lowering::tests::{XlaProgramBuilder, lowered_unary_module, unproject_plain_program};
    use crate::experimental::lowering::{lower_mlir_module_for_program, to_mlir_module_for_plain_program};
    use crate::experimental::ops::XlaArrayConstant;

    use super::*;

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_cumulative_sum_as_a_full_prefix_reduce_window() {
        // A forward scan pads `(n - 1, 0)` on the scanned axis so that the full-extent window ending at output
        // position `i` covers input positions `0..=i`, with the padded positions made inert by the summation
        // identity that also seeds the window.
        assert_eq!(
            lowered_unary_module(
                ArrayOperation::Cumulative(CumulativeOperation::new(0, CumulativeKind::Sum)),
                DataType::F32,
                vec![4],
            )
            .unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %0 = \"stablehlo.reduce_window\"(%arg0, %cst) <{\
                      padding = dense<[[3, 0]]> : tensor<1x2xi64>, \
                      window_dimensions = array<i64: 4>\
                    }> ({
                    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
                      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
                      stablehlo.return %1 : tensor<f32>
                    }) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "},
        );

        // A reverse scan keeps the same window and flips the padding to `(0, n - 1)`, so the window starting at
        // output position `i` covers input positions `i..n`.
        assert_eq!(
            lowered_unary_module(
                ArrayOperation::Cumulative(CumulativeOperation::new(0, CumulativeKind::Sum).with_reverse(true)),
                DataType::F32,
                vec![4],
            )
            .unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %0 = \"stablehlo.reduce_window\"(%arg0, %cst) <{\
                      padding = dense<[[0, 3]]> : tensor<1x2xi64>, \
                      window_dimensions = array<i64: 4>\
                    }> ({
                    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
                      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
                      stablehlo.return %1 : tensor<f32>
                    }) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "},
        );

        // Every axis other than the scanned one keeps a unit window and no padding, so the scan runs independently
        // per row rather than mixing rows.
        assert_eq!(
            lowered_unary_module(
                ArrayOperation::Cumulative(CumulativeOperation::new(1, CumulativeKind::Sum)),
                DataType::F32,
                vec![2, 3],
            )
            .unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %0 = \"stablehlo.reduce_window\"(%arg0, %cst) <{\
                      padding = dense<[[0, 0], [2, 0]]> : tensor<2x2xi64>, \
                      window_dimensions = array<i64: 1, 3>\
                    }> ({
                    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
                      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
                      stablehlo.return %1 : tensor<f32>
                    }) : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
                    return %0 : tensor<2x3xf32>
                  }
                }
            "},
        );

        // An empty scanned axis accumulates nothing and would give `reduce_window` a degenerate zero-extent window,
        // so the lowering forwards the operand instead of emitting any operation.
        assert_eq!(
            lowered_unary_module(
                ArrayOperation::Cumulative(CumulativeOperation::new(0, CumulativeKind::Sum)),
                DataType::F32,
                vec![0, 2],
            )
            .unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<0x2xf32>) -> tensor<0x2xf32> {
                    return %arg0 : tensor<0x2xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_the_remaining_prefix_scans() {
        // A prefix product keeps the scan's window and padding shape and differs from the summation only in its
        // reducer body and in the multiplicative identity seeding every window.
        assert_eq!(
            lowered_unary_module(
                ArrayOperation::Cumulative(CumulativeOperation::new(0, CumulativeKind::Product)),
                DataType::F32,
                vec![4],
            )
            .unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                    %0 = \"stablehlo.reduce_window\"(%arg0, %cst) <{\
                      padding = dense<[[3, 0]]> : tensor<1x2xi64>, \
                      window_dimensions = array<i64: 4>\
                    }> ({
                    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
                      %1 = stablehlo.multiply %arg1, %arg2 : tensor<f32>
                      stablehlo.return %1 : tensor<f32>
                    }) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "},
        );

        // The running maximum seeds its windows with negative infinity and reuses the portable total-order reducer
        // body that `reduce` uses, so signed zeros order correctly and NaNs propagate on every backend.
        assert_eq!(
            lowered_unary_module(
                ArrayOperation::Cumulative(CumulativeOperation::new(0, CumulativeKind::Max)),
                DataType::F32,
                vec![4],
            )
            .unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
                    %0 = \"stablehlo.reduce_window\"(%arg0, %cst) <{\
                      padding = dense<[[3, 0]]> : tensor<1x2xi64>, \
                      window_dimensions = array<i64: 4>\
                    }> ({
                    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
                      %1 = stablehlo.compare GT, %arg1, %arg2, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %2 = stablehlo.select %1, %arg1, %arg2 : tensor<i1>, tensor<f32>
                      %3 = stablehlo.compare NE, %arg1, %arg1, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %4 = stablehlo.compare NE, %arg2, %arg2, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %5 = stablehlo.select %4, %arg2, %2 : tensor<i1>, tensor<f32>
                      %6 = stablehlo.select %3, %arg1, %5 : tensor<i1>, tensor<f32>
                      stablehlo.return %6 : tensor<f32>
                    }) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "},
        );

        // The running minimum is the mirrored construction, seeded with positive infinity, and its reverse direction
        // flips the padding exactly as the summation's does.
        assert_eq!(
            lowered_unary_module(
                ArrayOperation::Cumulative(CumulativeOperation::new(0, CumulativeKind::Min).with_reverse(true)),
                DataType::F32,
                vec![4],
            )
            .unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %cst = stablehlo.constant dense<0x7F800000> : tensor<f32>
                    %0 = \"stablehlo.reduce_window\"(%arg0, %cst) <{\
                      padding = dense<[[0, 3]]> : tensor<1x2xi64>, \
                      window_dimensions = array<i64: 4>\
                    }> ({
                    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
                      %1 = stablehlo.compare LT, %arg1, %arg2, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %2 = stablehlo.select %1, %arg1, %arg2 : tensor<i1>, tensor<f32>
                      %3 = stablehlo.compare NE, %arg1, %arg1, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %4 = stablehlo.compare NE, %arg2, %arg2, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %5 = stablehlo.select %4, %arg2, %2 : tensor<i1>, tensor<f32>
                      %6 = stablehlo.select %3, %arg1, %5 : tensor<i1>, tensor<f32>
                      stablehlo.return %6 : tensor<f32>
                    }) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "},
        );

        // The running `log(sum(exp(x)))` inlines the whole guarded `log_add_exp` expansion into the reducer body and
        // seeds its windows with negative infinity, whose exponential is the inner sum's zero identity.
        assert_eq!(
            lowered_unary_module(
                ArrayOperation::Cumulative(CumulativeOperation::new(0, CumulativeKind::LogSumExp)),
                DataType::F32,
                vec![4],
            )
            .unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
                    %0 = \"stablehlo.reduce_window\"(%arg0, %cst) <{\
                      padding = dense<[[3, 0]]> : tensor<1x2xi64>, \
                      window_dimensions = array<i64: 4>\
                    }> ({
                    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
                      %1 = stablehlo.subtract %arg1, %arg2 : tensor<f32>
                      %2 = stablehlo.compare NE, %1, %1, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                      %3 = stablehlo.add %arg1, %arg2 : tensor<f32>
                      %4 = stablehlo.abs %1 : tensor<f32>
                      %5 = stablehlo.negate %4 : tensor<f32>
                      %6 = stablehlo.exponential %5 : tensor<f32>
                      %7 = stablehlo.log_plus_one %6 : tensor<f32>
                      %8 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
                      %9 = stablehlo.add %8, %7 : tensor<f32>
                      %10 = stablehlo.select %2, %3, %9 : tensor<i1>, tensor<f32>
                      stablehlo.return %10 : tensor<f32>
                    }) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_rejects_a_dynamically_sized_scanned_axis() {
        // The window extent of the emitted `reduce_window` is the scanned extent, so the scan has no lowering when
        // that extent is only known at runtime. `ryft-core` type inference already keeps the scanned axis static,
        // which is why reaching the lowering's own guard takes an unchecked instruction whose output type carries a
        // dynamic scanned dimension. The diagnostic names the scan the caller wrote, not the shared window helper.
        let scanned = DimensionVariable::new("scanned", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(scanned)]));
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(dynamic_type.clone());
        let output = builder.add_variable(dynamic_type);
        builder.add_instruction_unchecked(Instruction::new(
            ArrayOperation::Cumulative(CumulativeOperation::new(0, CumulativeKind::LogSumExp)),
            vec![input],
            vec![output],
            Vec::new(),
        ));
        let program = builder
            .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main"),
            Err(LoweringError::UnsupportedOp {
                op: "`cumulative` with kind `log_sum_exp` over dynamically sized axis 0".to_string(),
            }),
        );
    }

    #[test]
    fn test_lower_mlir_module_for_program_lowers_cumulative_sum_to_chlo_scan_on_gpu() {
        // GPU targets lower a forward cumulative sum to `chlo.scan`, whose body adds each slice to the running sum of
        // its row, seeded with the sum identity broadcast over the unscanned axes. A reverse scan keeps the full-prefix
        // `reduce_window` form.
        let lower = |operation: CumulativeOperation, target_platform: &str| {
            let input_type = ArrayType::new_static(DataType::F32, [2, 4]);
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(input_type.clone());
            let output = builder
                .add_instruction(ArrayOperation::Cumulative(operation), Vec::new(), vec![input], None)
                .unwrap()[0];
            let program = builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                    vec![output],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap();
            let types = vec![input_type];
            let program = unproject_plain_program(program);
            lower_mlir_module_for_program(&program, &[], &types, &types, "main", None, None, Some(target_platform))
                .unwrap()
                .stable_hlo
        };
        for target_platform in ["cuda", "rocm"] {
            assert_eq!(
                lower(CumulativeOperation::new(1, CumulativeKind::Sum), target_platform),
                indoc! {"
                    module {
                      func.func @main(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
                        %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                        %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2xf32>
                        %1:2 = chlo.scan(%arg0) inits (%0) dimension=1  attributes {is_associative = true} {
                        ^bb0(%input: tensor<2xf32>, %carry: tensor<2xf32>):
                          %2 = stablehlo.add %input, %carry : tensor<2xf32>
                          stablehlo.return %2, %2 : tensor<2xf32>, tensor<2xf32>
                        } : (tensor<2x4xf32>, tensor<2xf32>) -> (tensor<2x4xf32>, tensor<2xf32>)
                        return %1#0 : tensor<2x4xf32>
                      }
                    }
                "},
            );
        }
        assert_eq!(
            lower(CumulativeOperation::new(1, CumulativeKind::Sum).with_reverse(true), "cuda"),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %0 = \"stablehlo.reduce_window\"(%arg0, %cst) <{\
                      padding = dense<[[0, 0], [0, 3]]> : tensor<2x2xi64>, \
                      window_dimensions = array<i64: 1, 4>\
                    }> ({
                    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
                      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
                      stablehlo.return %1 : tensor<f32>
                    }) : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
                    return %0 : tensor<2x4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_lower_cumulative_one_bit_sum_on_gpu() {
        for data_type in [DataType::I1, DataType::U1] {
            let input_type = ArrayType::new_static(data_type, [2, 4]);
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(input_type.clone());
            let output = builder
                .add_instruction(CumulativeOperation::new(1, CumulativeKind::Sum), Vec::new(), vec![input], None)
                .unwrap()[0];
            let program = builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                    vec![output],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap();
            let program = unproject_plain_program(program);
            let types = vec![input_type];
            for target_platform in ["cuda", "rocm"] {
                assert_eq!(
                    lower_mlir_module_for_program(
                        &program,
                        &[],
                        &types,
                        &types,
                        "main",
                        None,
                        None,
                        Some(target_platform),
                    )
                    .unwrap()
                    .stable_hlo,
                    indoc! {"
                        module {
                          func.func @main(%arg0: tensor<2x4xi1>) -> tensor<2x4xi1> {
                            %c = stablehlo.constant dense<false> : tensor<i1>
                            %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<2xi1>
                            %1:2 = chlo.scan(%arg0) inits (%0) dimension=1  attributes {is_associative = true} {
                            ^bb0(%input: tensor<2xi1>, %carry: tensor<2xi1>):
                              %2 = stablehlo.xor %input, %carry : tensor<2xi1>
                              stablehlo.return %2, %2 : tensor<2xi1>, tensor<2xi1>
                            } : (tensor<2x4xi1>, tensor<2xi1>) -> (tensor<2x4xi1>, tensor<2xi1>)
                            return %1#0 : tensor<2x4xi1>
                          }
                        }
                    "},
                );
            }
        }
    }

    #[test]
    fn test_lower_cumulative_structural_zero() {
        for kind in [CumulativeKind::Sum, CumulativeKind::Product] {
            for reverse in [false, true] {
                assert_eq!(
                    lowered_unary_module(
                        ArrayOperation::Cumulative(CumulativeOperation::new(0, kind).with_reverse(reverse)),
                        DataType::Zero,
                        vec![3],
                    ),
                    Ok(indoc! {"
                        module {
                          func.func @main(%arg0: tensor<3xi1>) -> tensor<3xi1> {
                            return %arg0 : tensor<3xi1>
                          }
                        }
                    "}
                    .to_string()),
                );
            }
        }
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_cumulative_log_sum_exp_identities() {
        // Invalid unchecked programs still cannot seed a fold with a non-identity value.
        for element_type in [DataType::F8E8M0FNU, DataType::F6E2M3FN] {
            let array_type = ArrayType::new(element_type, Shape::new(vec![Dimension::Static(4)]));
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(array_type.clone());
            let output = builder.add_variable(array_type);
            builder.add_instruction_unchecked(Instruction::new(
                ArrayOperation::Cumulative(CumulativeOperation::new(0, CumulativeKind::LogSumExp)),
                vec![input],
                vec![output],
                Vec::new(),
            ));
            let program = builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                    vec![output],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap();
            assert_eq!(
                to_mlir_module_for_plain_program(&program, "main"),
                Err(LoweringError::UnsupportedOp {
                    op: format!(
                        "`cumulative` with kind `log_sum_exp` over `{element_type}`, whose lowest value is not a \
                         `log_add_exp` identity",
                    ),
                }),
            );
        }

        // Finite pairwise identities remain neutral beyond the previously imposed scan-length limits.
        for data_type in [
            DataType::F4E2M1FN,
            DataType::F6E3M2FN,
            DataType::F8E4M3B11FNUZ,
            DataType::F8E4M3FNUZ,
            DataType::F8E5M2FNUZ,
            DataType::F32,
        ] {
            assert!(
                lowered_unary_module(
                    ArrayOperation::Cumulative(CumulativeOperation::new(0, CumulativeKind::LogSumExp)),
                    data_type,
                    vec![3000],
                )
                .is_ok(),
            );
        }
    }

    #[test]
    fn test_to_mlir_module_for_plain_program_lowers_nonlinear_cumulative_jvp_over_dynamic_unscanned_axis() {
        // The forward mode of a nonlinear cumulative kind differentiates through `associative_scan`, whose slices keep
        // a dynamic unscanned axis whole, so the whole derivative program lowers for every nonlinear kind.
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5)).unwrap());
        let input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(4)]));
        for kind in [CumulativeKind::Product, CumulativeKind::Max, CumulativeKind::Min, CumulativeKind::LogSumExp] {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(input_type.clone());
            let output = builder
                .add_instruction(
                    ArrayOperation::Cumulative(CumulativeOperation::new(1, kind)),
                    Vec::new(),
                    vec![input],
                    None,
                )
                .unwrap()[0];
            let program = builder
                .build::<Vec<XlaArrayConstant>, Vec<XlaArrayConstant>>(
                    vec![output],
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap();
            let jvp = program.jvp().unwrap();
            assert!(to_mlir_module_for_plain_program(&jvp, "main").is_ok(), "{kind}");
        }
    }
}
