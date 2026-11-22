use ryft_core::macros::check_count;
use ryft_core::{ArrayType, ClampOperation, DataType, MaxOperation, MinOperation};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Block, BlockRef, LocationRef, Operation as MlirOperation, TensorTypeRef, Type as MlirType, Value as MlirValue,
    ValueRef,
};

use crate::lowering::{CollectiveLoweringState, constrain_unreduced_outputs};
use crate::operations::{normalize_binary_elementwise_operands, normalize_elementwise_operand};
use crate::{Error, LowerableOperation, LowerableOperationFamily, Lowerer, Type};

impl LowerableOperation for MaxOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = input_types;
        let [left, right] = normalize_binary_elementwise_operands(
            input_values,
            output_types,
            &mut lowerer.block(),
            lowerer.context(),
            lowerer.location(),
        )?;
        // Floating-point values take the form of the target, because the native operation does not propagate NaNs
        // or order the signed zeros on every backend (e.g., XLA's CPU backend returns its left-hand side input when
        // the right-hand side one is a NaN and for a tie between the signed zeros).
        let form = FloatExtremumForm::for_elementwise(lowerer.collective_state().target_platform());
        lower_elementwise_extremum(true, lowerer, left, right, &output_types[0], form)
    }
}

impl LowerableOperation for MinOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = input_types;
        let [left, right] = normalize_binary_elementwise_operands(
            input_values,
            output_types,
            &mut lowerer.block(),
            lowerer.context(),
            lowerer.location(),
        )?;
        // Floating-point values take the form of the target, because the native operation does not propagate NaNs
        // or order the signed zeros on every backend (e.g., XLA's CPU backend returns its left-hand side input when
        // the right-hand side one is a NaN and for a tie between the signed zeros).
        let form = FloatExtremumForm::for_elementwise(lowerer.collective_state().target_platform());
        lower_elementwise_extremum(false, lowerer, left, right, &output_types[0], form)
    }
}

impl LowerableOperation for ClampOperation<ArrayType> {
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = input_types;
        check_count!("input", input_values, 3, ProgramError);
        check_count!("output", output_types, 1, ProgramError);
        let output_type = &output_types[0];
        let input = normalize_elementwise_operand(
            input_values[1],
            output_type,
            &mut lowerer.block(),
            lowerer.context(),
            lowerer.location(),
        )?;
        let data_type = output_type.data_type();
        if data_type.is_complex() || data_type.is_floating_point() {
            // Floating-point and complex values compose `min(max(x, lower), upper)` explicitly. XLA's native `clamp`
            // does not propagate NaN operands on every backend (e.g., the CPU backend returns the lower bound for a
            // NaN input), and StableHLO leaves the ordering of complex `clamp` unspecified, whereas the extremum
            // lowerings below propagate the NaNs of the input and of both bounds, order the signed zeros, and order
            // complex values lexicographically like the reference backend.
            let [lower, upper] = [input_values[0], input_values[2]].map(|bound| {
                normalize_elementwise_operand(
                    bound,
                    output_type,
                    &mut lowerer.block(),
                    lowerer.context(),
                    lowerer.location(),
                )
            });
            let (lower, upper) = (lower?, upper?);
            let form = FloatExtremumForm::for_elementwise(lowerer.collective_state().target_platform());
            let clamped_below =
                lower_extremum_to_mlir(true, data_type, input, lower, form, &mut lowerer.block(), lowerer.location())?;
            return Ok(vec![lower_extremum_to_mlir(
                false,
                data_type,
                clamped_below,
                upper,
                form,
                &mut lowerer.block(),
                lowerer.location(),
            )?]);
        }

        // Integer and Boolean values use one `stablehlo.clamp`, which accepts scalar bounds or bounds with the
        // operand's shape, so rank-0 bounds keep their rank and only convert their element type, while every other
        // bound is broadcast to the result type.
        let mut bounds = [input_values[0], input_values[2]];
        for bound in &mut bounds {
            let is_scalar = bound.r#type()?.cast::<TensorTypeRef>().is_some_and(|r#type| r#type.rank() == 0);
            let bound_type = if is_scalar { ArrayType::scalar(output_type.data_type()) } else { output_type.clone() };
            *bound = normalize_elementwise_operand(
                *bound,
                &bound_type,
                &mut lowerer.block(),
                lowerer.context(),
                lowerer.location(),
            )?;
        }
        let result =
            lowerer
                .block()
                .append_operation(stable_hlo::clamp(bounds[0], input, bounds[1], lowerer.location())?)?;
        Ok(vec![result.result(0).unwrap().as_ref()])
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
    /// that is followed by the rest of a softmax), and its Triton code generation fails on the total order
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

/// Lowers the elementwise maximum (if `maximum` is set) or minimum of `left` and `right`, whose output has type
/// `output_type`, with [`lower_extremum_to_mlir`]. An output that is unreduced (i.e., an extremum of values that are
/// unreduced with the kind of the extremum) has intermediate values that are unreduced along the same mesh axes,
/// because each device computes the extremum of its own contributions, which commutes with the reduction that completes
/// them.
/// Shardy rejects an operation whose result drops an unreduced axis of its inputs, so those intermediate values are
/// constrained like the output (refer to [`constrain_unreduced_outputs`]). The constraints do not move any data, so the
/// values, including the NaNs and the signs of zeros, are the same as for values without reduction state.
fn lower_elementwise_extremum<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
    maximum: bool,
    lowerer: &mut Lowerer<'b, 'c, 't, O>,
    left: ValueRef<'b, 'c, 't>,
    right: ValueRef<'b, 'c, 't>,
    output_type: &ArrayType,
    form: FloatExtremumForm,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    let data_type = output_type.data_type();
    let mut block = lowerer.block();
    let location = lowerer.location();
    if output_type.unreduced_axes().is_empty() {
        return Ok(vec![lower_extremum_to_mlir(maximum, data_type, left, right, form, &mut block, location)?]);
    }
    let output_types = [Type::Array(output_type.clone())];
    let unreduced = Some((output_types.as_slice(), lowerer.collective_state()));
    Ok(vec![lower_constrained_extremum_to_mlir(
        maximum, data_type, left, right, form, unreduced, &mut block, location,
    )?])
}

/// Lowers one minimum or maximum that is compatible with JAX. Integer and Boolean values use the corresponding
/// StableHLO operation. Floating-point values use the provided [`FloatExtremumForm`] and explicit NaN propagation, so
/// that every XLA backend preserves the IEEE semantics of signed zeros and NaNs. Complex values use JAX's
/// lexicographic `(real, imaginary)` compare/select sequence.
pub(crate) fn lower_extremum_to_mlir<'b, 'c: 'b, 't: 'c>(
    maximum: bool,
    element_type: DataType,
    left: ValueRef<'b, 'c, 't>,
    right: ValueRef<'b, 'c, 't>,
    form: FloatExtremumForm,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    lower_constrained_extremum_to_mlir(maximum, element_type, left, right, form, None, block, location)
}

/// Lowers one minimum or maximum like [`lower_extremum_to_mlir`], constraining every intermediate value with
/// [`constrain_unreduced_outputs`] as if it had the output types and the lowering state in `unreduced`, when present.
/// The result itself is left for the lowering of the instruction to constrain.
#[allow(clippy::too_many_arguments)]
fn lower_constrained_extremum_to_mlir<'b, 'c: 'b, 't: 'c>(
    maximum: bool,
    element_type: DataType,
    left: ValueRef<'b, 'c, 't>,
    right: ValueRef<'b, 'c, 't>,
    form: FloatExtremumForm,
    unreduced: Option<(&[Type], &CollectiveLoweringState)>,
    block: &mut BlockRef<'b, 'c, 't>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    let constrain = |value: ValueRef<'b, 'c, 't>, block: &mut BlockRef<'b, 'c, 't>| {
        let mut values = [value];
        if let Some((output_types, collective_state)) = unreduced {
            constrain_unreduced_outputs(&mut values, output_types, collective_state, block, location)?;
        }
        Ok::<_, Error>(values[0])
    };
    if !element_type.is_complex() && !element_type.is_floating_point() {
        let result = if maximum {
            block.append_operation(stable_hlo::maximum(left, right, location)?)?
        } else {
            block.append_operation(stable_hlo::minimum(left, right, location)?)?
        };
        return Ok(result.result(0).unwrap().as_ref());
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
                let ordered = constrain(ordered.result(0).unwrap().as_ref(), block)?;
                block.append_operation(stable_hlo::select(ordered, left, right, location)?)?
            }
            FloatExtremumForm::Native if maximum => {
                block.append_operation(stable_hlo::maximum(left, right, location)?)?
            }
            FloatExtremumForm::Native => block.append_operation(stable_hlo::minimum(left, right, location)?)?,
        };
        let mut result = constrain(result.result(0).unwrap().as_ref(), block)?;

        if form == FloatExtremumForm::Arithmetic {
            // The ordinary comparison treats the signed zeros as equal, so a tie between two zeros is resolved by
            // arithmetic that orders negative zero below positive zero. A value is a zero exactly when it equals its
            // own negation, and two equal values of which one is a zero are both zeros. Their sum is negative zero
            // only when both are, which is their maximum, and the negated difference `-((-left) - right)` is positive
            // zero only when both are, which is their minimum.
            let negated_left = block.append_operation(stable_hlo::negate(left, location)?)?;
            let negated_left = constrain(negated_left.result(0).unwrap().as_ref(), block)?;
            let equal = block.append_operation(stable_hlo::compare(
                left,
                right,
                stable_hlo::ComparisonDirection::Equal,
                stable_hlo::ComparisonType::Float,
                location,
            )?)?;
            let equal = constrain(equal.result(0).unwrap().as_ref(), block)?;
            let left_is_zero = block.append_operation(stable_hlo::compare(
                left,
                negated_left,
                stable_hlo::ComparisonDirection::Equal,
                stable_hlo::ComparisonType::Float,
                location,
            )?)?;
            let left_is_zero = constrain(left_is_zero.result(0).unwrap().as_ref(), block)?;
            let both_zero = block.append_operation(stable_hlo::and(equal, left_is_zero, location)?)?;
            let both_zero = constrain(both_zero.result(0).unwrap().as_ref(), block)?;
            let zero = if maximum {
                block.append_operation(stable_hlo::add(left, right, location)?)?
            } else {
                let difference = block.append_operation(stable_hlo::subtract(negated_left, right, location)?)?;
                let difference = constrain(difference.result(0).unwrap().as_ref(), block)?;
                block.append_operation(stable_hlo::negate(difference, location)?)?
            };
            let zero = constrain(zero.result(0).unwrap().as_ref(), block)?;
            let selected = block.append_operation(stable_hlo::select(both_zero, zero, result, location)?)?;
            result = constrain(selected.result(0).unwrap().as_ref(), block)?;
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
        let left_is_nan = constrain(left_is_nan.result(0).unwrap().as_ref(), block)?;
        let right_is_nan = block.append_operation(stable_hlo::compare(
            right,
            right,
            stable_hlo::ComparisonDirection::NotEqual,
            stable_hlo::ComparisonType::Float,
            location,
        )?)?;
        let right_is_nan = constrain(right_is_nan.result(0).unwrap().as_ref(), block)?;
        let right_or_result = block.append_operation(stable_hlo::select(right_is_nan, right, result, location)?)?;
        let right_or_result = constrain(right_or_result.result(0).unwrap().as_ref(), block)?;
        let result = block.append_operation(stable_hlo::select(left_is_nan, left, right_or_result, location)?)?;
        return Ok(result.result(0).unwrap().as_ref());
    }

    let left_real = block.append_operation(stable_hlo::real(left, location)?)?;
    let left_real = constrain(left_real.result(0).unwrap().as_ref(), block)?;
    let right_real = block.append_operation(stable_hlo::real(right, location)?)?;
    let right_real = constrain(right_real.result(0).unwrap().as_ref(), block)?;
    let left_imaginary = block.append_operation(stable_hlo::imag(left, location)?)?;
    let left_imaginary = constrain(left_imaginary.result(0).unwrap().as_ref(), block)?;
    let right_imaginary = block.append_operation(stable_hlo::imag(right, location)?)?;
    let right_imaginary = constrain(right_imaginary.result(0).unwrap().as_ref(), block)?;
    let real_equal = block.append_operation(stable_hlo::compare(
        left_real,
        right_real,
        stable_hlo::ComparisonDirection::Equal,
        stable_hlo::ComparisonType::Float,
        location,
    )?)?;
    let real_equal = constrain(real_equal.result(0).unwrap().as_ref(), block)?;
    let direction =
        if maximum { stable_hlo::ComparisonDirection::GreaterThan } else { stable_hlo::ComparisonDirection::LessThan };
    let real_ordered = block.append_operation(stable_hlo::compare(
        left_real,
        right_real,
        direction,
        stable_hlo::ComparisonType::Float,
        location,
    )?)?;
    let real_ordered = constrain(real_ordered.result(0).unwrap().as_ref(), block)?;
    let imaginary_ordered = block.append_operation(stable_hlo::compare(
        left_imaginary,
        right_imaginary,
        direction,
        stable_hlo::ComparisonType::Float,
        location,
    )?)?;
    let imaginary_ordered = constrain(imaginary_ordered.result(0).unwrap().as_ref(), block)?;
    let ordered = block.append_operation(stable_hlo::select(real_equal, imaginary_ordered, real_ordered, location)?)?;
    let ordered = constrain(ordered.result(0).unwrap().as_ref(), block)?;
    let result = block.append_operation(stable_hlo::select(ordered, left, right, location)?)?;
    Ok(result.result(0).unwrap().as_ref())
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Clamp, Device, DeviceMesh, LogicalMesh, Max, MeshAxis, MeshAxisType, Min, Placeholder, ScanOptions, Shape,
        Sharding, Typed, UnreducedKind, ValueProjection, scan,
    };
    use ryft_pjrt::BufferType;

    use crate::lowering::tests::{MixedValue, dynamic_dimension, execute_mixed_program};
    use crate::operations::ProgramBuilder as ArrayIrProgramBuilder;
    use crate::sessions::tests::domain_mesh;
    use crate::tests::{cpu_client, execution_client, replicated_array, values_to_bytes};
    use crate::{Array, CompileTracer, CompiledFunction, Constant, FromPjrt, Options, Session, compile};

    use super::*;

    /// Uploads complex operands whose lexicographic order is decided by the real components, by the imaginary
    /// components, by signed zeros that compare equal, and by NaN payloads in either component.
    fn complex_operands(session: &Session, mesh: &DeviceMesh) -> ([Complex<f32>; 5], [Complex<f32>; 5], Array, Array) {
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
        let left_array = replicated_array(session, mesh, DataType::C64, &[5], values_to_bytes(&left).as_slice());
        let right_array = replicated_array(session, mesh, DataType::C64, &[5], values_to_bytes(&right).as_slice());
        (left, right, left_array, right_array)
    }

    /// Uploads `f32` operands whose order is decided by their values, by signed zeros that compare equal in both
    /// operand orders and with equal signs, by a NaN payload in the left operand, in the right operand, and in both,
    /// and by equal infinities.
    fn real_operands(session: &Session, mesh: &DeviceMesh) -> (Array, Array) {
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
        let left = replicated_array(session, mesh, DataType::F32, &[10], values_to_bytes(&left).as_slice());
        let right = replicated_array(session, mesh, DataType::F32, &[10], values_to_bytes(&right).as_slice());
        (left, right)
    }

    #[test]
    fn test_max_operation_lower() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);

        // Boolean maxima are disjunctions.
        let booleans = replicated_array(&session, &mesh, DataType::Boolean, &[2], &[0, 1]);
        let truth = replicated_array(&session, &mesh, DataType::Boolean, &[], &[1]);
        assert_eq!(booleans.max(&truth).unwrap().copy_to_host_bytes().unwrap(), values_to_bytes(&[true, true]));

        // Floating-point maxima order negative zero below positive zero in both operand orders, keep the sign of
        // equal zeros, and propagate a NaN of either operand with its payload, preferring the left one. Exact byte
        // comparisons check the signs of the zeros and the payloads.
        let (left, right) = real_operands(&session, &mesh);
        assert_eq!(
            left.max(&right).unwrap().copy_to_host_bytes().unwrap(),
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
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);

        // Complete operands compare lexicographically, and ties and unordered comparisons select the right operand.
        // Exact byte comparisons also check signed zeros and NaN payloads in both operand orders.
        let (left, right, left_array, right_array) = complex_operands(&session, &mesh);
        assert_eq!(
            left_array.max(&right_array).unwrap().copy_to_host_bytes().unwrap(),
            values_to_bytes(&[right[0], right[1], right[2], right[3], right[4]]),
        );
        assert_eq!(
            right_array.max(&left_array).unwrap().copy_to_host_bytes().unwrap(),
            values_to_bytes(&[right[0], right[1], left[2], left[3], left[4]]),
        );

        // A real rank-0 operand broadcasts into a complex vector after promotion to complex double precision.
        let complex = replicated_array(
            &session,
            &mesh,
            DataType::C128,
            &[2],
            values_to_bytes(&[Complex::new(1f64, 2.0), Complex::new(3.0, 4.0)]).as_slice(),
        );
        let real = replicated_array(&session, &mesh, DataType::F64, &[], values_to_bytes(&[2f64]).as_slice());
        let output = complex.max(&real).unwrap();
        assert_eq!(output.data_type(), DataType::C128);
        assert_eq!(output.shape().dimensions(), &[2]);
        assert_eq!(
            output.copy_to_host_bytes().unwrap(),
            values_to_bytes(&[Complex::new(2f64, 0.0), Complex::new(3.0, 4.0)])
        );
    }

    #[test]
    fn test_min_operation_lower() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);

        // Boolean minima are conjunctions.
        let booleans = replicated_array(&session, &mesh, DataType::Boolean, &[2], &[0, 1]);
        let truth = replicated_array(&session, &mesh, DataType::Boolean, &[], &[1]);
        assert_eq!(booleans.min(&truth).unwrap().copy_to_host_bytes().unwrap(), values_to_bytes(&[false, true]));

        // Floating-point minima order negative zero below positive zero in both operand orders, keep the sign of
        // equal zeros, and propagate a NaN of either operand with its payload, preferring the left one.
        let (left, right) = real_operands(&session, &mesh);
        assert_eq!(
            left.min(&right).unwrap().copy_to_host_bytes().unwrap(),
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
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);

        // Complete operands compare lexicographically, and ties and unordered comparisons select the right operand.
        // Exact byte comparisons also check signed zeros and NaN payloads in both operand orders.
        let (left, right, left_array, right_array) = complex_operands(&session, &mesh);
        assert_eq!(
            left_array.min(&right_array).unwrap().copy_to_host_bytes().unwrap(),
            values_to_bytes(&[left[0], left[1], right[2], right[3], right[4]]),
        );
        assert_eq!(
            right_array.min(&left_array).unwrap().copy_to_host_bytes().unwrap(),
            values_to_bytes(&[left[0], left[1], left[2], left[3], left[4]]),
        );

        // A real rank-0 operand broadcasts into a complex vector after promotion to complex double precision.
        let complex = replicated_array(
            &session,
            &mesh,
            DataType::C128,
            &[2],
            values_to_bytes(&[Complex::new(1f64, 2.0), Complex::new(3.0, 4.0)]).as_slice(),
        );
        let real = replicated_array(&session, &mesh, DataType::F64, &[], values_to_bytes(&[2f64]).as_slice());
        let output = complex.min(&real).unwrap();
        assert_eq!(output.data_type(), DataType::C128);
        assert_eq!(output.shape().dimensions(), &[2]);
        assert_eq!(
            output.copy_to_host_bytes().unwrap(),
            values_to_bytes(&[Complex::new(1f64, 2.0), Complex::new(2.0, 0.0)])
        );
    }

    #[test]
    fn test_min_and_max_operations_lower_dynamic_differentiation() {
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
                client,
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

    #[test]
    fn test_max_and_min_operations_lower_unreduced_values() {
        // Lowers the extremum of two vectors of type `type` for `mesh`.
        fn module(session: &Session, mesh: &DeviceMesh, maximum: bool, r#type: &ArrayType) -> String {
            let mut builder = ArrayIrProgramBuilder::new();
            let left = builder.add_input(r#type.clone().into());
            let right = builder.add_input(r#type.clone().into());
            let output = if maximum {
                builder.add_instruction(MaxOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0]
            } else {
                builder.add_instruction(MinOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0]
            };
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap();
            let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
            lowered.stable_hlo().to_string()
        }

        // Returns the extremum of `left` and `right` through the operation of the kind `kind`.
        fn extremum<V: Max + Min>(kind: UnreducedKind, left: &V, right: &V) -> V {
            if kind == UnreducedKind::Max { left.max(right).unwrap() } else { left.min(right).unwrap() }
        }

        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let pjrt_devices = client.addressable_devices().unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(
            logical_mesh.clone(),
            pjrt_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let single_device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![mesh.devices()[0]],
        )
        .unwrap();
        let r#type = |kind: UnreducedKind| {
            let sharding = Sharding::replicated(logical_mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap();
            let sharding = sharding.with_unreduced_kind(kind).unwrap();
            ArrayType::new_static(DataType::F32, [10]).with_sharding(sharding).unwrap()
        };

        // Each device holds the operands of `real_operands`, in both orders, so that executions check the signed zeros
        // and the NaN payloads of the extremum of every device against those of the same operation on values without
        // reduction state, at the top level of a compiled function, in the body of a `scan`, and in a called function.
        let (left, right) = real_operands(&session, &single_device_mesh);
        let (left, right) = (left.copy_to_host_bytes().unwrap(), right.copy_to_host_bytes().unwrap());
        let contributions = [[left.clone(), right.clone()], [right, left]];
        let mut outcomes = Vec::new();
        let mut expected_outcomes = Vec::new();
        for kind in [UnreducedKind::Max, UnreducedKind::Min] {
            let r#type = r#type(kind);
            let [left, right] = [0, 1].map(|operand| {
                let buffers = pjrt_devices
                    .iter()
                    .zip(&contributions)
                    .map(|(device, contributions)| {
                        let bytes = contributions[operand].as_slice();
                        client.buffer(bytes, BufferType::F32, &[10], None, device.clone(), None).unwrap()
                    })
                    .collect();
                Array::from_addressable_buffers(&session, r#type.clone(), mesh.clone(), None, buffers).unwrap()
            });
            let expected_shards = contributions
                .iter()
                .map(|[left, right]| {
                    let upload =
                        |bytes: &[u8]| replicated_array(&session, &single_device_mesh, DataType::F32, &[10], bytes);
                    extremum(kind, &upload(left), &upload(right)).copy_to_host_bytes().unwrap()
                })
                .collect::<Vec<_>>();
            let types = (r#type.clone(), r#type.clone());
            let top_level: CompiledFunction<(ArrayType, ArrayType), ArrayType> = compile(
                |(left, right): (CompileTracer, CompileTracer)| extremum(kind, &left, &right),
                types.clone(),
                &session,
                mesh.clone(),
            )
            .unwrap();
            let scanned: CompiledFunction<(ArrayType, ArrayType), ArrayType> = compile(
                |(left, right): (CompileTracer, CompileTracer)| {
                    let ((output, _), ()) = scan(
                        |_, (left, right), ()| {
                            let left = ValueProjection::<ArrayType>::into_projected(left)?;
                            let right = ValueProjection::<ArrayType>::into_projected(right)?;
                            Ok(((extremum(kind, &left, &right).into_value(), right.into_value()), ()))
                        },
                        (left.into_value(), right.into_value()),
                        (),
                        ScanOptions::new().with_iteration_count(1usize),
                    )
                    .unwrap();
                    ValueProjection::<ArrayType>::into_projected(output).unwrap()
                },
                types.clone(),
                &session,
                mesh.clone(),
            )
            .unwrap();
            let called: CompiledFunction<(ArrayType, ArrayType), ArrayType> =
                compile(|inputs| top_level.call(inputs).unwrap(), types.clone(), &session, mesh.clone()).unwrap();
            for (context, function) in
                [("top level", &top_level), ("scan body", &scanned), ("called function", &called)]
            {
                let output = function.executable().execute(&session, (left.clone(), right.clone())).unwrap();
                let shards =
                    output.shards().iter().map(|shard| shard.copy_to_host_bytes().unwrap()).collect::<Vec<_>>();
                outcomes.push(format!("{kind}, {context}: {} {shards:?}", output.r#type()));
                expected_outcomes.push(format!("{kind}, {context}: {type} {expected_shards:?}"));
            }
        }
        assert_eq!(outcomes.join("\n"), expected_outcomes.join("\n"));

        // The intermediate values of an extremum of unreduced values are unreduced along the same axes, because each
        // device computes the extremum of its own contributions, which commutes with the reduction that completes them.
        // Without their constraints, Shardy rejects the comparisons, which drop the unreduced axes of their inputs.
        assert_eq!(
            module(&session, &mesh, true, &r#type(UnreducedKind::Max)),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], unreduced=max{"x"}>}, %arg1: tensor<10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], unreduced=max{"x"}>}) -> (tensor<10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], unreduced=max{"x"}>}) {
                    %0 = stablehlo.compare GT, %arg0, %arg1, FLOAT : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{}], unreduced=max{"x"}> : tensor<10xi1>
                    %2 = stablehlo.select %1, %arg0, %arg1 : tensor<10xi1>, tensor<10xf32>
                    %3 = sdy.sharding_constraint %2 <@mesh, [{}], unreduced=max{"x"}> : tensor<10xf32>
                    %4 = stablehlo.negate %arg0 : tensor<10xf32>
                    %5 = sdy.sharding_constraint %4 <@mesh, [{}], unreduced=max{"x"}> : tensor<10xf32>
                    %6 = stablehlo.compare EQ, %arg0, %arg1, FLOAT : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
                    %7 = sdy.sharding_constraint %6 <@mesh, [{}], unreduced=max{"x"}> : tensor<10xi1>
                    %8 = stablehlo.compare EQ, %arg0, %5, FLOAT : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
                    %9 = sdy.sharding_constraint %8 <@mesh, [{}], unreduced=max{"x"}> : tensor<10xi1>
                    %10 = stablehlo.and %7, %9 : tensor<10xi1>
                    %11 = sdy.sharding_constraint %10 <@mesh, [{}], unreduced=max{"x"}> : tensor<10xi1>
                    %12 = stablehlo.add %arg0, %arg1 : tensor<10xf32>
                    %13 = sdy.sharding_constraint %12 <@mesh, [{}], unreduced=max{"x"}> : tensor<10xf32>
                    %14 = stablehlo.select %11, %13, %3 : tensor<10xi1>, tensor<10xf32>
                    %15 = sdy.sharding_constraint %14 <@mesh, [{}], unreduced=max{"x"}> : tensor<10xf32>
                    %16 = stablehlo.compare NE, %arg0, %arg0, FLOAT : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
                    %17 = sdy.sharding_constraint %16 <@mesh, [{}], unreduced=max{"x"}> : tensor<10xi1>
                    %18 = stablehlo.compare NE, %arg1, %arg1, FLOAT : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
                    %19 = sdy.sharding_constraint %18 <@mesh, [{}], unreduced=max{"x"}> : tensor<10xi1>
                    %20 = stablehlo.select %19, %arg1, %15 : tensor<10xi1>, tensor<10xf32>
                    %21 = sdy.sharding_constraint %20 <@mesh, [{}], unreduced=max{"x"}> : tensor<10xf32>
                    %22 = stablehlo.select %17, %arg0, %21 : tensor<10xi1>, tensor<10xf32>
                    %23 = sdy.sharding_constraint %22 <@mesh, [{}], unreduced=max{"x"}> : tensor<10xf32>
                    return %23 : tensor<10xf32>
                  }
                }
            "#},
        );
        assert_eq!(
            module(&session, &mesh, false, &r#type(UnreducedKind::Min)),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], unreduced=min{"x"}>}, %arg1: tensor<10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], unreduced=min{"x"}>}) -> (tensor<10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], unreduced=min{"x"}>}) {
                    %0 = stablehlo.compare LT, %arg0, %arg1, FLOAT : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{}], unreduced=min{"x"}> : tensor<10xi1>
                    %2 = stablehlo.select %1, %arg0, %arg1 : tensor<10xi1>, tensor<10xf32>
                    %3 = sdy.sharding_constraint %2 <@mesh, [{}], unreduced=min{"x"}> : tensor<10xf32>
                    %4 = stablehlo.negate %arg0 : tensor<10xf32>
                    %5 = sdy.sharding_constraint %4 <@mesh, [{}], unreduced=min{"x"}> : tensor<10xf32>
                    %6 = stablehlo.compare EQ, %arg0, %arg1, FLOAT : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
                    %7 = sdy.sharding_constraint %6 <@mesh, [{}], unreduced=min{"x"}> : tensor<10xi1>
                    %8 = stablehlo.compare EQ, %arg0, %5, FLOAT : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
                    %9 = sdy.sharding_constraint %8 <@mesh, [{}], unreduced=min{"x"}> : tensor<10xi1>
                    %10 = stablehlo.and %7, %9 : tensor<10xi1>
                    %11 = sdy.sharding_constraint %10 <@mesh, [{}], unreduced=min{"x"}> : tensor<10xi1>
                    %12 = stablehlo.subtract %5, %arg1 : tensor<10xf32>
                    %13 = sdy.sharding_constraint %12 <@mesh, [{}], unreduced=min{"x"}> : tensor<10xf32>
                    %14 = stablehlo.negate %13 : tensor<10xf32>
                    %15 = sdy.sharding_constraint %14 <@mesh, [{}], unreduced=min{"x"}> : tensor<10xf32>
                    %16 = stablehlo.select %11, %15, %3 : tensor<10xi1>, tensor<10xf32>
                    %17 = sdy.sharding_constraint %16 <@mesh, [{}], unreduced=min{"x"}> : tensor<10xf32>
                    %18 = stablehlo.compare NE, %arg0, %arg0, FLOAT : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
                    %19 = sdy.sharding_constraint %18 <@mesh, [{}], unreduced=min{"x"}> : tensor<10xi1>
                    %20 = stablehlo.compare NE, %arg1, %arg1, FLOAT : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
                    %21 = sdy.sharding_constraint %20 <@mesh, [{}], unreduced=min{"x"}> : tensor<10xi1>
                    %22 = stablehlo.select %21, %arg1, %17 : tensor<10xi1>, tensor<10xf32>
                    %23 = sdy.sharding_constraint %22 <@mesh, [{}], unreduced=min{"x"}> : tensor<10xf32>
                    %24 = stablehlo.select %19, %arg0, %23 : tensor<10xi1>, tensor<10xf32>
                    %25 = sdy.sharding_constraint %24 <@mesh, [{}], unreduced=min{"x"}> : tensor<10xf32>
                    return %25 : tensor<10xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_clamp_operation_lower() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);

        // Floating-point values below, inside, and above the interval keep their exact bytes, NaN inputs propagate,
        // and crossed bounds produce the upper bound.
        let input = replicated_array(
            &session,
            &mesh,
            DataType::F32,
            &[4],
            values_to_bytes(&[-2.5f32, 0.5, 2.5, f32::NAN]).as_slice(),
        );
        let lower = replicated_array(&session, &mesh, DataType::F32, &[], values_to_bytes(&[-1f32]).as_slice());
        let upper = replicated_array(&session, &mesh, DataType::F32, &[], values_to_bytes(&[1f32]).as_slice());
        assert_eq!(
            input.clamp(&lower, &upper).unwrap().copy_to_host_bytes().unwrap(),
            values_to_bytes(&[-1f32, 0.5, 1.0, f32::NAN]),
        );
        assert_eq!(
            input.clamp(&upper, &lower).unwrap().copy_to_host_bytes().unwrap(),
            values_to_bytes(&[-1f32, -1.0, -1.0, f32::NAN]),
        );

        // A NaN bound propagates like a NaN input, and a zero bound of the other sign orders against a zero input.
        let zeros = replicated_array(&session, &mesh, DataType::F32, &[2], values_to_bytes(&[0.5f32, -0.0]).as_slice());
        let nan = replicated_array(&session, &mesh, DataType::F32, &[], values_to_bytes(&[f32::NAN]).as_slice());
        let zero = replicated_array(&session, &mesh, DataType::F32, &[], values_to_bytes(&[0f32]).as_slice());
        assert_eq!(
            zeros.clamp(&nan, &upper).unwrap().copy_to_host_bytes().unwrap(),
            values_to_bytes(&[f32::NAN, f32::NAN])
        );
        assert_eq!(
            zeros.clamp(&lower, &nan).unwrap().copy_to_host_bytes().unwrap(),
            values_to_bytes(&[f32::NAN, f32::NAN])
        );
        assert_eq!(zeros.clamp(&zero, &upper).unwrap().copy_to_host_bytes().unwrap(), values_to_bytes(&[0.5f32, 0.0]));

        // Array bounds broadcast to the result.
        let bounds =
            replicated_array(&session, &mesh, DataType::F32, &[4], values_to_bytes(&[0f32, 1.0, 2.0, 3.0]).as_slice());
        assert_eq!(
            input.clamp(&lower, &bounds).unwrap().copy_to_host_bytes().unwrap(),
            values_to_bytes(&[-1f32, 0.5, 2.0, f32::NAN]),
        );

        // Integer inputs clamp against rank-0 bounds.
        let integers =
            replicated_array(&session, &mesh, DataType::I32, &[3], values_to_bytes(&[-5i32, 2, 9]).as_slice());
        let integer_lower = replicated_array(&session, &mesh, DataType::I32, &[], values_to_bytes(&[0i32]).as_slice());
        let integer_upper = replicated_array(&session, &mesh, DataType::I32, &[], values_to_bytes(&[5i32]).as_slice());
        assert_eq!(
            integers.clamp(&integer_lower, &integer_upper).unwrap().copy_to_host_bytes().unwrap(),
            values_to_bytes(&[0i32, 2, 5]),
        );

        // Boolean inputs clamp with conjunction and disjunction.
        let booleans = replicated_array(&session, &mesh, DataType::Boolean, &[2], &[0, 1]);
        let truth = replicated_array(&session, &mesh, DataType::Boolean, &[], &[1]);
        assert_eq!(
            booleans.clamp(&truth, &truth).unwrap().copy_to_host_bytes().unwrap(),
            values_to_bytes(&[true, true])
        );

        // All three inputs take part in promotion before selection, preserving integers that would round in `f32`.
        let input = replicated_array(&session, &mesh, DataType::I64, &[], values_to_bytes(&[16777217i64]).as_slice());
        let lower = replicated_array(&session, &mesh, DataType::F32, &[], values_to_bytes(&[0f32]).as_slice());
        let upper = replicated_array(&session, &mesh, DataType::F64, &[], values_to_bytes(&[2e7f64]).as_slice());
        let output = input.clamp(&lower, &upper).unwrap();
        assert_eq!(output.data_type(), DataType::F64);
        assert_eq!(output.copy_to_host_bytes().unwrap(), values_to_bytes(&[16777217f64]));

        // Complex inputs compose the lexicographic extrema, so the imaginary part decides between equal real parts.
        let complex = replicated_array(
            &session,
            &mesh,
            DataType::C64,
            &[3],
            values_to_bytes(&[Complex::new(1f32, -1.0), Complex::new(1.0, 1.0), Complex::new(1.0, 3.0)]).as_slice(),
        );
        let complex_lower = replicated_array(
            &session,
            &mesh,
            DataType::C64,
            &[],
            values_to_bytes(&[Complex::new(1f32, 0.0)]).as_slice(),
        );
        let complex_upper = replicated_array(
            &session,
            &mesh,
            DataType::C64,
            &[],
            values_to_bytes(&[Complex::new(1f32, 2.0)]).as_slice(),
        );
        assert_eq!(
            complex.clamp(&complex_lower, &complex_upper).unwrap().copy_to_host_bytes().unwrap(),
            values_to_bytes(&[Complex::new(1f32, 0.0), Complex::new(1.0, 1.0), Complex::new(1.0, 2.0)]),
        );
    }

    #[test]
    fn test_float_extremum_form_for_combiner() {
        // Only CUDA uses the native form. Elsewhere, and without a target, scalar combiner bodies use the total order
        // form.
        assert_eq!(FloatExtremumForm::for_combiner(Some("cuda")), FloatExtremumForm::Native);
        assert_eq!(FloatExtremumForm::for_combiner(Some("cpu")), FloatExtremumForm::TotalOrder);
        assert_eq!(FloatExtremumForm::for_combiner(None), FloatExtremumForm::TotalOrder);
    }

    #[test]
    fn test_float_extremum_form_for_elementwise() {
        // Only CUDA uses the native form. Elsewhere, and without a target, elementwise operations use the arithmetic
        // form, which supports bounded dynamic shapes.
        assert_eq!(FloatExtremumForm::for_elementwise(Some("cuda")), FloatExtremumForm::Native);
        assert_eq!(FloatExtremumForm::for_elementwise(Some("cpu")), FloatExtremumForm::Arithmetic);
        assert_eq!(FloatExtremumForm::for_elementwise(None), FloatExtremumForm::Arithmetic);
    }
}
