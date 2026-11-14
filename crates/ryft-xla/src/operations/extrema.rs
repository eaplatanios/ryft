use ryft_core::macros::check_count;
use ryft_core::{ArrayType, ClampOperation, DataType, MaxOperation, MinOperation};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, LocationRef, Operation as MlirOperation, TensorTypeRef, Type, Value as MlirValue, ValueRef,
};

use crate::operations::{normalize_binary_elementwise_operands, normalize_elementwise_operand};
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer};

impl LowerableOperation for MaxOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let [left, right] = normalize_binary_elementwise_operands(
            input_values,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        // Floating-point values take the form of the target, because the native operation does not propagate NaNs
        // or order the signed zeros on every backend (e.g., XLA's CPU backend returns its left-hand side input when
        // the right-hand side one is a NaN and for a tie between the signed zeros).
        let form = FloatExtremumForm::for_elementwise(lowerer.collective_state.target_platform());
        Ok(vec![lower_extremum_to_mlir(
            true,
            output_types[0].data_type(),
            left,
            right,
            form,
            &mut lowerer.block,
            lowerer.location,
        )?])
    }
}

impl LowerableOperation for MinOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let [left, right] = normalize_binary_elementwise_operands(
            input_values,
            output_types,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        // Floating-point values take the form of the target, because the native operation does not propagate NaNs
        // or order the signed zeros on every backend (e.g., XLA's CPU backend returns its left-hand side input when
        // the right-hand side one is a NaN and for a tie between the signed zeros).
        let form = FloatExtremumForm::for_elementwise(lowerer.collective_state.target_platform());
        Ok(vec![lower_extremum_to_mlir(
            false,
            output_types[0].data_type(),
            left,
            right,
            form,
            &mut lowerer.block,
            lowerer.location,
        )?])
    }
}

impl LowerableOperation for ClampOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        _input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        check_count!("input", input_values, 3, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        let output_type = &output_types[0];
        let input = normalize_elementwise_operand(
            input_values[1],
            output_type,
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        let data_type = output_type.data_type();
        if data_type.is_complex() || data_type.is_floating_point() {
            // Floating-point and complex values compose `min(max(x, lower), upper)` explicitly. XLA's native `clamp`
            // does not propagate NaN operands on every backend (e.g., the CPU backend returns the lower bound for a
            // NaN input), and StableHLO leaves the ordering of complex `clamp` unspecified, whereas the extremum
            // lowerings below propagate the NaNs of the input and of both bounds, order the signed zeros, and order
            // complex values lexicographically like the reference backend.
            let [lower, upper] = [input_values[0], input_values[2]].map(|bound| {
                normalize_elementwise_operand(bound, output_type, &mut lowerer.block, lowerer.context, lowerer.location)
            });
            let (lower, upper) = (lower?, upper?);
            let form = FloatExtremumForm::for_elementwise(lowerer.collective_state.target_platform());
            let clamped_below =
                lower_extremum_to_mlir(true, data_type, input, lower, form, &mut lowerer.block, lowerer.location)?;
            return Ok(vec![lower_extremum_to_mlir(
                false,
                data_type,
                clamped_below,
                upper,
                form,
                &mut lowerer.block,
                lowerer.location,
            )?]);
        }

        // Integer and Boolean values use one `stablehlo.clamp`, which accepts scalar bounds or bounds with the
        // operand's shape, so rank-zero bounds keep their rank and only convert their element type, while every other
        // bound is broadcast to the result type.
        let mut bounds = [input_values[0], input_values[2]];
        for bound in &mut bounds {
            let is_scalar = bound.r#type()?.cast::<TensorTypeRef>().is_some_and(|r#type| r#type.rank() == 0);
            let bound_type = if is_scalar { ArrayType::scalar(output_type.data_type()) } else { output_type.clone() };
            *bound = normalize_elementwise_operand(
                *bound,
                &bound_type,
                &mut lowerer.block,
                lowerer.context,
                lowerer.location,
            )?;
        }
        let result =
            lowerer.block.append_operation(stable_hlo::clamp(bounds[0], input, bounds[1], lowerer.location)?)?;
        Ok(vec![result.result(0).expect("stablehlo.clamp should return one result").as_ref()])
    }
}

/// Form in which [`lower_extremum_to_mlir`] lowers a floating-point minimum or maximum. Every form propagates NaNs and
/// orders negative zero below positive zero. They differ in which XLA backends evaluate them with those semantics and
/// in which programs each backend can compile them.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum FloatExtremumForm {
    /// Explicit `TOTALORDER` comparison and selection, which orders the signed zeros on every XLA backend. XLA's
    /// dynamic padder rejects such comparisons over bounded dynamic arrays, and the Triton code generation of its GPU
    /// compiler fails on them, so this is the form of scalar combiner bodies (e.g., of reductions) that are not
    /// lowered for CUDA. XLA can only distribute a reduction whose combiner is a single operation, so
    /// [`lower_reduce_to_mlir`](crate::operations::reductions::lower_reduce_to_mlir) lowers the extrema of this form
    /// as reductions of integer keys that have the same order, for the floating-point types that have such keys.
    TotalOrder,

    /// Ordinary `FLOAT` comparison and selection, with a tie between two zeros resolved by arithmetic that orders the
    /// signed zeros. It supports bounded dynamic arrays, so it is the form of elementwise operations that are not
    /// lowered for CUDA. Its arithmetic is exact on backends that evaluate floating-point arithmetic strictly (e.g.,
    /// XLA's CPU backend), but not under the simplifications of XLA's GPU compiler, which ignore the sign of zero.
    Arithmetic,

    /// Native `stablehlo.maximum` or `stablehlo.minimum`, which orders the signed zeros on CUDA but not on every
    /// other XLA backend (e.g., XLA's CPU backend returns its left-hand side input for a tie between them). It is
    /// the form of every lowering for CUDA.
    Native,
}

impl FloatExtremumForm {
    /// Returns the [`FloatExtremumForm`] of scalar combiner bodies (e.g., of reductions) that are lowered for the
    /// PJRT platform named `target_platform`. CUDA uses [`FloatExtremumForm::Native`], because XLA's GPU compiler
    /// fuses reductions and the elementwise operations around them into Triton kernels (e.g., a maximum reduction
    /// that is followed by the rest of a softmax), and its Triton code generation fails on the total-order
    /// comparisons of [`FloatExtremumForm::TotalOrder`], which makes every such program fail to compile. Every other
    /// target, and the absence of a target, uses [`FloatExtremumForm::TotalOrder`]. The native form is not a combiner
    /// that XLA can distribute, so the lowering of a reduction whose dimensions can be sharded prefers the integer
    /// keys of [`FloatExtremumForm::TotalOrder`] for every target, which XLA's GPU compiler compiles as well.
    #[inline]
    pub(crate) fn for_combiner(target_platform: Option<&str>) -> Self {
        if target_platform == Some("cuda") { Self::Native } else { Self::TotalOrder }
    }

    /// Returns the [`FloatExtremumForm`] of elementwise operations that are lowered for the PJRT platform named
    /// `target_platform`. CUDA uses [`FloatExtremumForm::Native`] for the reason that [`Self::for_combiner`]
    /// describes. Every other target, and the absence of a target, uses [`FloatExtremumForm::Arithmetic`], because
    /// the inputs of elementwise operations may have bounded dynamic shapes.
    #[inline]
    pub(crate) fn for_elementwise(target_platform: Option<&str>) -> Self {
        if target_platform == Some("cuda") { Self::Native } else { Self::Arithmetic }
    }
}

/// Lowers one JAX-compatible minimum or maximum. Integer and Boolean values use the corresponding StableHLO
/// operation. Floating-point values use the provided [`FloatExtremumForm`] and explicit NaN propagation, so that every
/// XLA backend preserves IEEE signed-zero and NaN semantics. Complex values use JAX's lexicographic
/// `(real, imaginary)` compare/select sequence.
pub(crate) fn lower_extremum_to_mlir<'b, 'c: 'b, 't: 'c>(
    maximum: bool,
    element_type: DataType,
    left: ValueRef<'b, 'c, 't>,
    right: ValueRef<'b, 'c, 't>,
    form: FloatExtremumForm,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    if !element_type.is_complex() && !element_type.is_floating_point() {
        let result = if maximum {
            block.append_operation(stable_hlo::maximum(left, right, location)?)?
        } else {
            block.append_operation(stable_hlo::minimum(left, right, location)?)?
        };
        return Ok(result.result(0).expect("stablehlo extremum should return one result").as_ref());
    }
    if element_type.is_floating_point() {
        let direction = if maximum {
            stable_hlo::ComparisonDirection::GreaterThan
        } else {
            stable_hlo::ComparisonDirection::LessThan
        };
        let result = match form {
            FloatExtremumForm::TotalOrder | FloatExtremumForm::Arithmetic => {
                let comparison_type = if form == FloatExtremumForm::TotalOrder {
                    stable_hlo::ComparisonType::TotalOrder
                } else {
                    stable_hlo::ComparisonType::Float
                };
                let ordered =
                    block.append_operation(stable_hlo::compare(left, right, direction, comparison_type, location)?)?;
                let ordered = ordered.result(0).expect("stablehlo.compare should return one result").as_ref();
                block.append_operation(stable_hlo::select(ordered, left, right, location)?)?
            }
            FloatExtremumForm::Native if maximum => {
                block.append_operation(stable_hlo::maximum(left, right, location)?)?
            }
            FloatExtremumForm::Native => block.append_operation(stable_hlo::minimum(left, right, location)?)?,
        };
        let mut result = result.result(0).expect("stablehlo extremum should return one result").as_ref();

        if form == FloatExtremumForm::Arithmetic {
            // The ordinary comparison treats the signed zeros as equal, so a tie between two zeros is resolved by
            // arithmetic that orders negative zero below positive zero. A value is a zero exactly when it equals its
            // own negation, and two equal values of which one is a zero are both zeros. Their sum is negative zero
            // only when both are, which is their maximum, and the negated difference `-((-left) - right)` is positive
            // zero only when both are, which is their minimum.
            let negated_left = block.append_operation(stable_hlo::negate(left, location)?)?;
            let negated_left = negated_left.result(0).expect("stablehlo.negate should return one result").as_ref();
            let equal = block.append_operation(stable_hlo::compare(
                left,
                right,
                stable_hlo::ComparisonDirection::Equal,
                stable_hlo::ComparisonType::Float,
                location,
            )?)?;
            let equal = equal.result(0).expect("stablehlo.compare should return one result").as_ref();
            let left_is_zero = block.append_operation(stable_hlo::compare(
                left,
                negated_left,
                stable_hlo::ComparisonDirection::Equal,
                stable_hlo::ComparisonType::Float,
                location,
            )?)?;
            let left_is_zero = left_is_zero.result(0).expect("stablehlo.compare should return one result").as_ref();
            let both_zero = block.append_operation(stable_hlo::and(equal, left_is_zero, location)?)?;
            let both_zero = both_zero.result(0).expect("stablehlo.and should return one result").as_ref();
            let zero = if maximum {
                block.append_operation(stable_hlo::add(left, right, location)?)?
            } else {
                let difference = block.append_operation(stable_hlo::subtract(negated_left, right, location)?)?;
                let difference = difference.result(0).expect("stablehlo.subtract should return one result").as_ref();
                block.append_operation(stable_hlo::negate(difference, location)?)?
            };
            let zero = zero.result(0).expect("stablehlo arithmetic should return one result").as_ref();
            let selected = block.append_operation(stable_hlo::select(both_zero, zero, result, location)?)?;
            result = selected.result(0).expect("stablehlo.select should return one result").as_ref();
        }

        // Some XLA backends implement native min/max using instructions that drop NaNs or do not distinguish signed
        // zeros. StableHLO and JAX require IEEE maximum/minimum, so make both behaviors explicit. The form above
        // orders the signed zeros, and the selections below propagate a NaN of either input in every form.
        let left_is_nan = block.append_operation(stable_hlo::compare(
            left,
            left,
            stable_hlo::ComparisonDirection::NotEqual,
            stable_hlo::ComparisonType::Float,
            location,
        )?)?;
        let left_is_nan = left_is_nan.result(0).expect("stablehlo.compare should return one result").as_ref();
        let right_is_nan = block.append_operation(stable_hlo::compare(
            right,
            right,
            stable_hlo::ComparisonDirection::NotEqual,
            stable_hlo::ComparisonType::Float,
            location,
        )?)?;
        let right_is_nan = right_is_nan.result(0).expect("stablehlo.compare should return one result").as_ref();
        let right_or_result = block.append_operation(stable_hlo::select(right_is_nan, right, result, location)?)?;
        let right_or_result = right_or_result.result(0).expect("stablehlo.select should return one result").as_ref();
        let result = block.append_operation(stable_hlo::select(left_is_nan, left, right_or_result, location)?)?;
        return Ok(result.result(0).expect("stablehlo.select should return one result").as_ref());
    }

    let left_real = block.append_operation(stable_hlo::real(left, location)?)?;
    let left_real = left_real.result(0).expect("stablehlo.real should return one result").as_ref();
    let right_real = block.append_operation(stable_hlo::real(right, location)?)?;
    let right_real = right_real.result(0).expect("stablehlo.real should return one result").as_ref();
    let left_imaginary = block.append_operation(stable_hlo::imag(left, location)?)?;
    let left_imaginary = left_imaginary.result(0).expect("stablehlo.imag should return one result").as_ref();
    let right_imaginary = block.append_operation(stable_hlo::imag(right, location)?)?;
    let right_imaginary = right_imaginary.result(0).expect("stablehlo.imag should return one result").as_ref();
    let real_equal = block.append_operation(stable_hlo::compare(
        left_real,
        right_real,
        stable_hlo::ComparisonDirection::Equal,
        stable_hlo::ComparisonType::Float,
        location,
    )?)?;
    let real_equal = real_equal.result(0).expect("stablehlo.compare should return one result").as_ref();
    let direction =
        if maximum { stable_hlo::ComparisonDirection::GreaterThan } else { stable_hlo::ComparisonDirection::LessThan };
    let real_ordered = block.append_operation(stable_hlo::compare(
        left_real,
        right_real,
        direction,
        stable_hlo::ComparisonType::Float,
        location,
    )?)?;
    let real_ordered = real_ordered.result(0).expect("stablehlo.compare should return one result").as_ref();
    let imaginary_ordered = block.append_operation(stable_hlo::compare(
        left_imaginary,
        right_imaginary,
        direction,
        stable_hlo::ComparisonType::Float,
        location,
    )?)?;
    let imaginary_ordered = imaginary_ordered.result(0).expect("stablehlo.compare should return one result").as_ref();
    let ordered = block.append_operation(stable_hlo::select(real_equal, imaginary_ordered, real_ordered, location)?)?;
    let ordered = ordered.result(0).expect("stablehlo.select should return one result").as_ref();
    let result = block.append_operation(stable_hlo::select(ordered, left, right, location)?)?;
    Ok(result.result(0).expect("stablehlo.select should return one result").as_ref())
}

#[cfg(test)]
mod tests {
    use num_complex::Complex;
    use pretty_assertions::assert_eq;
    use ryft_core::{Clamp, DeviceMesh, Max, Min, Placeholder, Shape};

    use crate::domains::tests::domain_mesh;
    use crate::experimental::lowering::tests::{MixedValue, dynamic_dimension, execute_mixed_program};
    use crate::experimental::ops::{Constant, ProgramBuilder as ArrayIrProgramBuilder};
    use crate::tests::{execution_client, replicated_array, values_to_bytes};
    use crate::{Array, Domain, Session};

    use super::*;

    /// Uploads complex operands whose lexicographic order is decided by the real components, by the imaginary
    /// components, by signed zeros that compare equal, and by NaN payloads in either component.
    fn complex_operands<'c>(
        domain: &Domain<'c>,
        mesh: &DeviceMesh,
    ) -> ([Complex<f32>; 5], [Complex<f32>; 5], Array<'c>, Array<'c>) {
        let left = [
            Complex::new(1f32, 100.0),
            Complex::new(1.0, 2.0),
            Complex::new(-0.0, 0.0),
            Complex::new(f32::from_bits(0x7fc0_1234), 1.0),
            Complex::new(1.0, f32::from_bits(0x7fc0_5678)),
        ];
        let right = [
            Complex::new(2f32, 0.0),
            Complex::new(1.0, 3.0),
            Complex::new(0.0, -0.0),
            Complex::new(2.0, 3.0),
            Complex::new(1.0, 4.0),
        ];
        let left_array = replicated_array(domain, mesh, DataType::C64, &[5], values_to_bytes(&left).as_slice());
        let right_array = replicated_array(domain, mesh, DataType::C64, &[5], values_to_bytes(&right).as_slice());
        (left, right, left_array, right_array)
    }

    /// Uploads `f32` operands whose order is decided by their values, by signed zeros that compare equal (in both
    /// operand orders and with equal signs), by a NaN payload in the left operand, in the right operand, and in both,
    /// and by equal infinities.
    fn real_operands<'c>(domain: &Domain<'c>, mesh: &DeviceMesh) -> (Array<'c>, Array<'c>) {
        let left = [
            1f32,
            3.0,
            -0.0,
            0.0,
            -0.0,
            0.0,
            f32::from_bits(0x7fc0_1234),
            2.0,
            f32::from_bits(0x7fc0_1234),
            f32::NEG_INFINITY,
        ];
        let right = [
            2f32,
            1.0,
            0.0,
            -0.0,
            -0.0,
            0.0,
            7.0,
            f32::from_bits(0x7fc0_5678),
            f32::from_bits(0x7fc0_5678),
            f32::NEG_INFINITY,
        ];
        let left = replicated_array(domain, mesh, DataType::F32, &[10], values_to_bytes(&left).as_slice());
        let right = replicated_array(domain, mesh, DataType::F32, &[10], values_to_bytes(&right).as_slice());
        (left, right)
    }

    #[test]
    fn test_max_operation_lower() {
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);

        // Boolean maxima are disjunctions.
        let booleans = replicated_array(&domain, &mesh, DataType::Boolean, &[2], &[0, 1]);
        let truth = replicated_array(&domain, &mesh, DataType::Boolean, &[], &[1]);
        assert_eq!(booleans.max(&truth).unwrap().copy_to_host().unwrap(), values_to_bytes(&[true, true]));

        // Floating-point maxima order negative zero below positive zero in both operand orders, keep the sign of
        // equal zeros, and propagate a NaN of either operand with its payload, preferring the left one. Exact byte
        // comparisons check the signs of the zeros and the payloads.
        let (left, right) = real_operands(&domain, &mesh);
        assert_eq!(
            left.max(&right).unwrap().copy_to_host().unwrap(),
            values_to_bytes(&[
                2f32,
                3.0,
                0.0,
                0.0,
                -0.0,
                0.0,
                f32::from_bits(0x7fc0_1234),
                f32::from_bits(0x7fc0_5678),
                f32::from_bits(0x7fc0_1234),
                f32::NEG_INFINITY,
            ]),
        );
    }

    #[test]
    fn test_max_operation_lower_complex() {
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);

        // Complete operands compare lexicographically, and ties and unordered comparisons select the right operand.
        // Exact byte comparisons also check signed zeros and NaN payloads in both operand orders.
        let (left, right, left_array, right_array) = complex_operands(&domain, &mesh);
        assert_eq!(
            left_array.max(&right_array).unwrap().copy_to_host().unwrap(),
            values_to_bytes(&[right[0], right[1], right[2], right[3], right[4]]),
        );
        assert_eq!(
            right_array.max(&left_array).unwrap().copy_to_host().unwrap(),
            values_to_bytes(&[right[0], right[1], left[2], left[3], left[4]]),
        );

        // A real rank-0 operand broadcasts into a complex vector after promotion to complex double precision.
        let complex = replicated_array(
            &domain,
            &mesh,
            DataType::C128,
            &[2],
            values_to_bytes(&[Complex::new(1f64, 2.0), Complex::new(3.0, 4.0)]).as_slice(),
        );
        let real = replicated_array(&domain, &mesh, DataType::F64, &[], values_to_bytes(&[2f64]).as_slice());
        let output = complex.max(&real).unwrap();
        assert_eq!(output.data_type(), DataType::C128);
        assert_eq!(output.shape().dimensions(), &[2]);
        assert_eq!(output.copy_to_host().unwrap(), values_to_bytes(&[Complex::new(2f64, 0.0), Complex::new(3.0, 4.0)]));
    }

    #[test]
    fn test_min_operation_lower() {
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);

        // Boolean minima are conjunctions.
        let booleans = replicated_array(&domain, &mesh, DataType::Boolean, &[2], &[0, 1]);
        let truth = replicated_array(&domain, &mesh, DataType::Boolean, &[], &[1]);
        assert_eq!(booleans.min(&truth).unwrap().copy_to_host().unwrap(), values_to_bytes(&[false, true]));

        // Floating-point minima order negative zero below positive zero in both operand orders, keep the sign of
        // equal zeros, and propagate a NaN of either operand with its payload, preferring the left one.
        let (left, right) = real_operands(&domain, &mesh);
        assert_eq!(
            left.min(&right).unwrap().copy_to_host().unwrap(),
            values_to_bytes(&[
                1f32,
                1.0,
                -0.0,
                -0.0,
                -0.0,
                0.0,
                f32::from_bits(0x7fc0_1234),
                f32::from_bits(0x7fc0_5678),
                f32::from_bits(0x7fc0_1234),
                f32::NEG_INFINITY,
            ]),
        );
    }

    #[test]
    fn test_min_operation_lower_complex() {
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);

        // Complete operands compare lexicographically, and ties and unordered comparisons select the right operand.
        // Exact byte comparisons also check signed zeros and NaN payloads in both operand orders.
        let (left, right, left_array, right_array) = complex_operands(&domain, &mesh);
        assert_eq!(
            left_array.min(&right_array).unwrap().copy_to_host().unwrap(),
            values_to_bytes(&[left[0], left[1], right[2], right[3], right[4]]),
        );
        assert_eq!(
            right_array.min(&left_array).unwrap().copy_to_host().unwrap(),
            values_to_bytes(&[left[0], left[1], left[2], left[3], left[4]]),
        );

        // A real rank-0 operand broadcasts into a complex vector after promotion to complex double precision.
        let complex = replicated_array(
            &domain,
            &mesh,
            DataType::C128,
            &[2],
            values_to_bytes(&[Complex::new(1f64, 2.0), Complex::new(3.0, 4.0)]).as_slice(),
        );
        let real = replicated_array(&domain, &mesh, DataType::F64, &[], values_to_bytes(&[2f64]).as_slice());
        let output = complex.min(&real).unwrap();
        assert_eq!(output.data_type(), DataType::C128);
        assert_eq!(output.shape().dimensions(), &[2]);
        assert_eq!(output.copy_to_host().unwrap(), values_to_bytes(&[Complex::new(1f64, 2.0), Complex::new(2.0, 0.0)]));
    }

    #[test]
    fn test_clamp_operation_lower() {
        let client = execution_client();
        let domain = Session::new(&client).domain();
        let mesh = domain_mesh(&client, "x", 1);

        // Floating-point values below, inside, and above the interval keep their exact bytes, NaN inputs propagate,
        // and crossed bounds produce the upper bound.
        let input = replicated_array(
            &domain,
            &mesh,
            DataType::F32,
            &[4],
            values_to_bytes(&[-2.5f32, 0.5, 2.5, f32::NAN]).as_slice(),
        );
        let lower = replicated_array(&domain, &mesh, DataType::F32, &[], values_to_bytes(&[-1f32]).as_slice());
        let upper = replicated_array(&domain, &mesh, DataType::F32, &[], values_to_bytes(&[1f32]).as_slice());
        assert_eq!(
            input.clamp(&lower, &upper).unwrap().copy_to_host().unwrap(),
            values_to_bytes(&[-1f32, 0.5, 1.0, f32::NAN]),
        );
        assert_eq!(
            input.clamp(&upper, &lower).unwrap().copy_to_host().unwrap(),
            values_to_bytes(&[-1f32, -1.0, -1.0, f32::NAN]),
        );

        // A NaN bound propagates like a NaN input, and a zero bound of the other sign orders against a zero input.
        let zeros = replicated_array(&domain, &mesh, DataType::F32, &[2], values_to_bytes(&[0.5f32, -0.0]).as_slice());
        let nan = replicated_array(&domain, &mesh, DataType::F32, &[], values_to_bytes(&[f32::NAN]).as_slice());
        let zero = replicated_array(&domain, &mesh, DataType::F32, &[], values_to_bytes(&[0f32]).as_slice());
        assert_eq!(zeros.clamp(&nan, &upper).unwrap().copy_to_host().unwrap(), values_to_bytes(&[f32::NAN, f32::NAN]));
        assert_eq!(zeros.clamp(&lower, &nan).unwrap().copy_to_host().unwrap(), values_to_bytes(&[f32::NAN, f32::NAN]));
        assert_eq!(zeros.clamp(&zero, &upper).unwrap().copy_to_host().unwrap(), values_to_bytes(&[0.5f32, 0.0]));

        // Array bounds broadcast to the result.
        let bounds =
            replicated_array(&domain, &mesh, DataType::F32, &[4], values_to_bytes(&[0f32, 1.0, 2.0, 3.0]).as_slice());
        assert_eq!(
            input.clamp(&lower, &bounds).unwrap().copy_to_host().unwrap(),
            values_to_bytes(&[-1f32, 0.5, 2.0, f32::NAN]),
        );

        // Integer inputs clamp against rank-0 bounds.
        let integers =
            replicated_array(&domain, &mesh, DataType::I32, &[3], values_to_bytes(&[-5i32, 2, 9]).as_slice());
        let integer_lower = replicated_array(&domain, &mesh, DataType::I32, &[], values_to_bytes(&[0i32]).as_slice());
        let integer_upper = replicated_array(&domain, &mesh, DataType::I32, &[], values_to_bytes(&[5i32]).as_slice());
        assert_eq!(
            integers.clamp(&integer_lower, &integer_upper).unwrap().copy_to_host().unwrap(),
            values_to_bytes(&[0i32, 2, 5]),
        );

        // Boolean inputs clamp with conjunction and disjunction.
        let booleans = replicated_array(&domain, &mesh, DataType::Boolean, &[2], &[0, 1]);
        let truth = replicated_array(&domain, &mesh, DataType::Boolean, &[], &[1]);
        assert_eq!(booleans.clamp(&truth, &truth).unwrap().copy_to_host().unwrap(), values_to_bytes(&[true, true]));

        // All three inputs take part in promotion before selection, preserving integers that would round in `f32`.
        let input = replicated_array(&domain, &mesh, DataType::I64, &[], values_to_bytes(&[16777217i64]).as_slice());
        let lower = replicated_array(&domain, &mesh, DataType::F32, &[], values_to_bytes(&[0f32]).as_slice());
        let upper = replicated_array(&domain, &mesh, DataType::F64, &[], values_to_bytes(&[2e7f64]).as_slice());
        let output = input.clamp(&lower, &upper).unwrap();
        assert_eq!(output.data_type(), DataType::F64);
        assert_eq!(output.copy_to_host().unwrap(), values_to_bytes(&[16777217f64]));

        // Complex inputs compose the lexicographic extrema, so the imaginary part decides between equal real parts.
        let complex = replicated_array(
            &domain,
            &mesh,
            DataType::C64,
            &[3],
            values_to_bytes(&[Complex::new(1f32, -1.0), Complex::new(1.0, 1.0), Complex::new(1.0, 3.0)]).as_slice(),
        );
        let complex_lower = replicated_array(
            &domain,
            &mesh,
            DataType::C64,
            &[],
            values_to_bytes(&[Complex::new(1f32, 0.0)]).as_slice(),
        );
        let complex_upper = replicated_array(
            &domain,
            &mesh,
            DataType::C64,
            &[],
            values_to_bytes(&[Complex::new(1f32, 2.0)]).as_slice(),
        );
        assert_eq!(
            complex.clamp(&complex_lower, &complex_upper).unwrap().copy_to_host().unwrap(),
            values_to_bytes(&[Complex::new(1f32, 0.0), Complex::new(1.0, 1.0), Complex::new(1.0, 2.0)]),
        );
    }

    #[test]
    fn test_float_extremum_form() {
        // Only CUDA uses the native form. Elsewhere, and without a target, scalar combiner bodies use the total-order
        // form and elementwise operations use the arithmetic one, which supports bounded dynamic shapes.
        assert_eq!(FloatExtremumForm::for_combiner(Some("cuda")), FloatExtremumForm::Native);
        assert_eq!(FloatExtremumForm::for_combiner(Some("cpu")), FloatExtremumForm::TotalOrder);
        assert_eq!(FloatExtremumForm::for_combiner(None), FloatExtremumForm::TotalOrder);
        assert_eq!(FloatExtremumForm::for_elementwise(Some("cuda")), FloatExtremumForm::Native);
        assert_eq!(FloatExtremumForm::for_elementwise(Some("cpu")), FloatExtremumForm::Arithmetic);
        assert_eq!(FloatExtremumForm::for_elementwise(None), FloatExtremumForm::Arithmetic);
    }

    #[test]
    fn test_to_mlir_module_for_program_extrema_dynamic_differentiation() {
        let client = execution_client();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![dynamic_dimension("size", Some(5))]));
        let mut builder = ArrayIrProgramBuilder::new();
        let left = builder.add_input(input_type.clone().into());
        let right = builder.add_input(input_type.into());
        let minimum = builder.add_instruction(MinOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let maximum = builder.add_instruction(MaxOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![minimum, maximum], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap()
            .jvp()
            .unwrap();

        // The fourth element is padding. Both selection and tie weights must preserve the runtime extent.
        assert_eq!(
            execute_mixed_program(
                &client,
                &program,
                &[
                    MixedValue::Array(vec![1.0, 3.0, 2.0, 0.0], vec![4]),
                    MixedValue::Array(vec![2.0, 1.0, 2.0, 0.0], vec![4]),
                    MixedValue::Array(vec![2.0, 4.0, 6.0, 0.0], vec![4]),
                    MixedValue::Array(vec![8.0, 10.0, 12.0, 0.0], vec![4]),
                ],
                &[3, 3, 3, 3],
            )
            .unwrap(),
            vec![
                MixedValue::Array(vec![1.0, 1.0, 2.0], vec![3]),
                MixedValue::Array(vec![2.0, 3.0, 2.0], vec![3]),
                MixedValue::Array(vec![2.0, 10.0, 9.0], vec![3]),
                MixedValue::Array(vec![8.0, 4.0, 9.0], vec![3]),
                MixedValue::Dimension(3),
                MixedValue::Dimension(3),
                MixedValue::Dimension(3),
                MixedValue::Dimension(3),
            ],
        );
    }
}
