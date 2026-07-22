use ryft_core::contexts::Context;
use ryft_core::macros::check_count;
use ryft_core::operations::control_flow::{Select, WhilePredicate};
use ryft_core::operations::logical::{AndOperation, NotOperation, OrOperation, XorOperation};
use ryft_core::operations::manipulation::Broadcast;
use ryft_core::operations::manipulation::conversion::ElementType;
use ryft_core::operations::math::{Add, Div, Mul, Neg, Sub};
use ryft_core::programs::types::Typed;
use ryft_core::programs::{Concretizable, ProgramError, Value};
use ryft_core::types::DataType;

use crate::experimental::ops::XlaOperation;
use crate::{Array, ArrayShard};

/// Eagerly executes one `operation` over `inputs` through the first input's recovered
/// [`XlaDomain`](crate::XlaDomain) and extracts the operation's single output.
///
/// This is the JAX-style op-by-op dispatch shape behind every eager value capability of [`Array`]. The blanket
/// capability implementations in `ryft-core` (arithmetic, comparison, selection, manipulation, reductions, ...)
/// already follow it through [`Value::dispatch_domain`] — which for a concrete [`Array`] recovers the rich,
/// PJRT-backed [`XlaDomain`](crate::XlaDomain) that compiles a cached single-operation program and executes it —
/// so this module only implements the capabilities those blankets cannot cover: the foreign `std::ops` operator
/// sugar (per-type implementations required by the orphan rule) and the host-readback predicates
/// [`Concretizable<bool>`] and [`WhilePredicate`]. This helper is their shared bind-and-unwrap step; callers must pass
/// at least one input,
/// and the first input determines the domain (and thereby the client and compile cache) the operation executes
/// against.
fn bind_single_output<'o, P: Into<XlaOperation>>(
    operation: P,
    inputs: &[Array<'o>],
) -> Result<Array<'o>, ProgramError> {
    let domain = inputs.first().unwrap().execution_domain();
    let mut outputs = domain.bind(operation, Vec::new(), inputs)?;
    check_count!("output", outputs, 1, ProgramError);
    Ok(outputs.remove(0))
}

/// Copies the dense payload bytes of one addressable [`ArrayShard`] to the host, surfacing missing buffers and PJRT
/// transfer failures as [`ProgramError::Concretization`] errors.
fn shard_host_bytes(shard: &ArrayShard<'_>) -> Result<Vec<u8>, ProgramError> {
    let concretization = |message: String| ProgramError::Concretization { message };
    let buffer = shard
        .buffer()
        .ok_or_else(|| concretization("cannot read a non-addressable array shard from the current process".into()))?;
    buffer
        .copy_to_host(None)
        .map_err(|error| concretization(error.to_string()))?
        .r#await()
        .map_err(|error| concretization(error.to_string()))
}

impl Concretizable<bool> for Array<'_> {
    /// Extracts a concrete scalar Rust Boolean by copying one addressable shard of a rank-0 Boolean-typed [`Array`]
    /// to the host. Higher-rank or non-Boolean arrays error because they cannot collapse to a single Boolean, and
    /// arrays with no addressable shards error because the current process cannot read their payload.
    fn concretize(&self) -> Result<bool, ProgramError> {
        if self.r#type().rank() == 0 && self.data_type().is_boolean() {
            let shard = self.addressable_shards().next().ok_or_else(|| ProgramError::Concretization {
                message: format!(
                    "cannot extract a concrete boolean from an array of type {} with no addressable shards",
                    self.r#type(),
                ),
            })?;
            let bytes = shard_host_bytes(shard)?;
            return Ok(bytes.iter().any(|byte| *byte != 0));
        }
        Err(ProgramError::Concretization {
            message: format!(
                "cannot extract a concrete boolean from a value of type {}; expected bool[]",
                self.r#type(),
            ),
        })
    }
}

/// Batched while-predicate semantics for [`Array`], mirroring the reference semantics of
/// [`Array`](ryft_core::backends::arrays::Array): [`WhilePredicate::any_true`] reduces the whole Boolean payload with
/// `or` via device-to-host readback of every shard, and [`WhilePredicate::mask_select`] broadcasts the predicate
/// against the operands along its leading (prefix) axes on device before selecting.
impl WhilePredicate for Array<'_> {
    fn any_true(&self) -> Result<bool, ProgramError> {
        if !self.data_type().is_boolean() {
            return Err(ProgramError::Concretization {
                message: format!("cannot use a value of type {} as a Boolean while predicate", self.r#type()),
            });
        }
        // Reading every addressable shard covers all elements for both sharded placements (the shards partition the
        // payload) and replicated placements (`or` is idempotent over duplicates). Non-addressable shards would leave
        // remote elements unobserved, so they surface as concretization errors instead of a silently partial answer.
        let mut addressable_count = 0usize;
        let mut any = false;
        for shard in self.addressable_shards() {
            addressable_count += 1;
            if !any {
                any = shard_host_bytes(shard)?.iter().any(|byte| *byte != 0);
            }
        }
        if addressable_count != self.shards().len() {
            return Err(ProgramError::Concretization {
                message: format!(
                    "cannot decide a while predicate for an array of type {} whose shards are not all addressable \
                     from the current process",
                    self.r#type(),
                ),
            });
        }
        Ok(any)
    }

    fn mask_select(&self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        if !self.data_type().is_boolean() || on_true.r#type() != on_false.r#type() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "mask_select requires a Boolean prefix-shaped predicate and congruent operands, but got \
                     predicate {} with operands {} and {}",
                    self.r#type(),
                    on_true.r#type(),
                    on_false.r#type(),
                ),
            });
        }
        // Broadcast a prefix-shaped predicate up to the operands' full shape (its axes map to the leading output
        // axes) so the staged select receives shape-congruent operands.
        let condition = if self.r#type().shape() == on_true.r#type().shape() {
            self.clone()
        } else {
            let output_type = on_true.r#type().with_element_type(DataType::Boolean);
            let output_axes = (0..self.r#type().rank()).collect::<Vec<_>>();
            self.broadcast(output_type, output_axes.as_slice())?
        };
        Select::select(&condition, on_true, on_false)
    }
}

// The `std::ops` operator traits are foreign, so the blanket tracer implementations in `ryft-core` cannot cover
// concrete backend values; these per-type implementations provide the ergonomic panicking sugar by delegating to
// the fallible `ryft` capabilities (and, for the logical operators that have no fallible `ryft` counterpart
// traits, by binding the logical operations directly).

impl std::ops::Add for Array<'_> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Add::add(&self, &rhs).expect("`add` operation failed")
    }
}

impl std::ops::Sub for Array<'_> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Sub::sub(&self, &rhs).expect("`sub` operation failed")
    }
}

impl std::ops::Mul for Array<'_> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Mul::mul(&self, &rhs).expect("`mul` operation failed")
    }
}

impl std::ops::Div for Array<'_> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Div::div(&self, &rhs).expect("`div` operation failed")
    }
}

impl std::ops::Neg for Array<'_> {
    type Output = Self;

    fn neg(self) -> Self {
        Neg::neg(&self).expect("`neg` operation failed")
    }
}

impl std::ops::BitAnd for Array<'_> {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        bind_single_output(AndOperation, &[self, rhs]).expect("`and` operation failed")
    }
}

impl std::ops::BitOr for Array<'_> {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        bind_single_output(OrOperation, &[self, rhs]).expect("`or` operation failed")
    }
}

impl std::ops::BitXor for Array<'_> {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self {
        bind_single_output(XorOperation, &[self, rhs]).expect("`xor` operation failed")
    }
}

impl std::ops::Not for Array<'_> {
    type Output = Self;

    fn not(self) -> Self {
        bind_single_output(NotOperation, &[self]).expect("`not` operation failed")
    }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use pretty_assertions::assert_eq;

    use ryft_core::Sharding;
    use ryft_core::backends::arrays::Array as CpuArray;
    use ryft_core::backends::scalars::Scalar;
    use ryft_core::batching::{BatchAxis, batch};
    use ryft_core::operations::compare::{Compare, ComparisonDirection};
    use ryft_core::operations::constants::{OneLike, ZeroLike};
    use ryft_core::operations::differentiation::{CoordinateBasisOperation, StopGradient};
    use ryft_core::operations::manipulation::{
        Concatenate, ConvertElementType, Pad, Reshape, Slice, Transpose, UpdateSlice,
    };
    use ryft_core::operations::math::{
        Abs, Atan2, Ceil, Cos, Dot, Erf, Exp, Floor, Log, Logistic, Maximum, Minimum, Pow, Reduce, ReductionKind,
        Remainder, Round, Rsqrt, Sign, Sin, Sqrt, Tanh,
    };
    use ryft_core::operations::tag::Tag;
    use ryft_core::sharding::{Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, ShardingDimension};
    use ryft_core::tracing_v2::{
        ForwardModeDifferentiate, HessianDifferentiate, JacobianDifferentiate, ReverseModeDifferentiate,
        jacobian_reverse,
    };
    use ryft_core::types::{ArrayType, Shape, Size, StaticShape};
    use ryft_pjrt::{Client, ClientOptions, CpuClientOptions, load_cpu_plugin};

    use crate::tests::{
        ADD_ONE_CUSTOM_CALL_TARGET, ensure_add_one_handler_registered, values_from_bytes, values_to_bytes,
    };
    use crate::{Array, FromPjrt};

    use super::*;

    fn cpu_mesh(client: &Client<'_>) -> DeviceMesh {
        cpu_mesh_with_axis_size(client, 1)
    }

    fn cpu_mesh_with_axis_size(client: &Client<'_>, axis_size: usize) -> DeviceMesh {
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", axis_size, MeshAxisType::Auto).unwrap()]).unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .into_iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        DeviceMesh::new(logical_mesh, devices).unwrap()
    }

    fn replicated_type(mesh: &DeviceMesh, data_type: DataType, dimensions: &[usize]) -> ArrayType {
        let shape = Shape::new(dimensions.iter().map(|&dimension| Size::Static(dimension)).collect());
        ArrayType::new(data_type, shape)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), dimensions.len()))
            .unwrap()
    }

    fn f32_vector<'c>(client: &'c Client<'c>, mesh: &DeviceMesh, values: &[f32]) -> Array<'c> {
        let r#type = replicated_type(mesh, DataType::F32, &[values.len()]);
        Array::from_host_buffer(client, r#type, mesh.clone(), values_to_bytes::<f32>(values).as_slice()).unwrap()
    }

    fn f32_matrix<'c>(
        client: &'c Client<'c>,
        mesh: &DeviceMesh,
        rows: usize,
        columns: usize,
        values: &[f32],
    ) -> Array<'c> {
        assert_eq!(values.len(), rows * columns);
        let r#type = replicated_type(mesh, DataType::F32, &[rows, columns]);
        Array::from_host_buffer(client, r#type, mesh.clone(), values_to_bytes::<f32>(values).as_slice()).unwrap()
    }

    fn f32_scalar<'c>(client: &'c Client<'c>, mesh: &DeviceMesh, value: f32) -> Array<'c> {
        let r#type = replicated_type(mesh, DataType::F32, &[]);
        Array::from_host_buffer(client, r#type, mesh.clone(), value.to_ne_bytes().as_slice()).unwrap()
    }

    fn boolean_vector<'c>(client: &'c Client<'c>, mesh: &DeviceMesh, values: &[bool]) -> Array<'c> {
        let r#type = replicated_type(mesh, DataType::Boolean, &[values.len()]);
        let bytes = values.iter().map(|value| u8::from(*value)).collect::<Vec<_>>();
        Array::from_host_buffer(client, r#type, mesh.clone(), bytes.as_slice()).unwrap()
    }

    fn read_f32s(array: &Array<'_>) -> Vec<f32> {
        let shard = array.addressable_shards().next().unwrap();
        let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
        values_from_bytes::<f32>(bytes.as_slice())
    }

    fn read_i32s(array: &Array<'_>) -> Vec<i32> {
        let shard = array.addressable_shards().next().unwrap();
        let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
        values_from_bytes::<i32>(bytes.as_slice())
    }

    /// Copies an array to a dense row-major host vector for assertions only. Production dense differential assembly
    /// never calls this helper: its blocks remain [`Array`] values on device.
    fn read_f64_coordinates(array: &Array<'_>) -> Vec<f64> {
        let dimensions = array.shape().dimensions().to_vec();
        let count = dimensions.iter().product();
        let mut coordinates = vec![None; count];
        for shard in array.addressable_shards() {
            let bytes = shard_host_bytes(shard).unwrap();
            let values = match array.data_type() {
                DataType::BF16 => bytes
                    .chunks_exact(2)
                    .map(|bytes| bf16::from_ne_bytes(bytes.try_into().unwrap()).to_f64())
                    .collect::<Vec<_>>(),
                DataType::F16 => bytes
                    .chunks_exact(2)
                    .map(|bytes| f16::from_ne_bytes(bytes.try_into().unwrap()).to_f64())
                    .collect::<Vec<_>>(),
                DataType::F32 => bytes
                    .chunks_exact(4)
                    .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()) as f64)
                    .collect::<Vec<_>>(),
                DataType::F64 => {
                    bytes.chunks_exact(8).map(|bytes| f64::from_ne_bytes(bytes.try_into().unwrap())).collect::<Vec<_>>()
                }
                data_type => panic!("test coordinate readback does not support {data_type}"),
            };
            let slice = shard.slice();
            let shard_count = slice.iter().map(|range| range.end - range.start).product::<usize>();
            assert_eq!(values.len(), shard_count);
            for (local_flat, value) in values.into_iter().enumerate() {
                let mut local_stride = shard_count;
                let mut global_flat = 0usize;
                for (dimension, range) in slice.iter().enumerate() {
                    let shard_size = range.end - range.start;
                    local_stride /= shard_size;
                    let local_index = (local_flat / local_stride) % shard_size;
                    global_flat = global_flat * dimensions[dimension] + range.start + local_index;
                }
                coordinates[global_flat] = Some(value);
            }
        }
        coordinates.into_iter().map(|value| value.expect("all test shards are addressable")).collect()
    }

    fn read_booleans(array: &Array<'_>) -> Vec<bool> {
        let shard = array.addressable_shards().next().unwrap();
        let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
        bytes.iter().map(|byte| *byte != 0).collect()
    }

    fn c64_scalar<'c>(client: &'c Client<'c>, mesh: &DeviceMesh, value: num_complex::Complex<f32>) -> Array<'c> {
        let r#type = replicated_type(mesh, DataType::C64, &[]);
        Array::from_host_buffer(client, r#type, mesh.clone(), values_to_bytes(&[value]).as_slice()).unwrap()
    }

    fn read_c64s(array: &Array<'_>) -> Vec<num_complex::Complex<f32>> {
        let shard = array.addressable_shards().next().unwrap();
        let bytes = shard.buffer().unwrap().copy_to_host(None).unwrap().r#await().unwrap();
        values_from_bytes::<num_complex::Complex<f32>>(bytes.as_slice())
    }

    /// Asserts that `actual` is within a small absolute tolerance of `expected`, per component. XLA's complex
    /// arithmetic may associate differently than the host's, so exact equality is not required.
    fn assert_c64_close(actual: num_complex::Complex<f32>, expected: num_complex::Complex<f32>) {
        let close = (actual.re - expected.re).abs() < 1e-5 && (actual.im - expected.im).abs() < 1e-5;
        assert!(close, "expected {expected} but got {actual}");
    }

    #[test]
    fn test_eager_arithmetic_and_trigonometric_capabilities() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let a = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0, 4.0]);
        let b = f32_vector(&client, &mesh, &[10.0, 20.0, 30.0, 40.0]);

        assert_eq!(read_f32s(&a.add(&b).unwrap()), vec![11.0, 22.0, 33.0, 44.0]);
        assert_eq!(read_f32s(&b.sub(&a).unwrap()), vec![9.0, 18.0, 27.0, 36.0]);
        assert_eq!(read_f32s(&a.mul(&b).unwrap()), vec![10.0, 40.0, 90.0, 160.0]);
        assert_eq!(read_f32s(&b.div(&a).unwrap()), vec![10.0, 10.0, 10.0, 10.0]);
        assert_eq!(read_f32s(&a.neg().unwrap()), vec![-1.0, -2.0, -3.0, -4.0]);
        for (observed, input) in read_f32s(&a.sin().unwrap()).iter().zip([1.0f32, 2.0, 3.0, 4.0]) {
            assert!((observed - input.sin()).abs() < 1e-5);
        }
        for (observed, input) in read_f32s(&a.cos().unwrap()).iter().zip([1.0f32, 2.0, 3.0, 4.0]) {
            assert!((observed - input.cos()).abs() < 1e-5);
        }

        // Complex sine and cosine preserve their mathematically exact zero component even when the hyperbolic factor
        // overflows. A naive `0 * inf` decomposition would produce a NaN here.
        let extreme = c64_scalar(&client, &mesh, num_complex::Complex::new(0.0, 1000.0));
        let sine = read_c64s(&extreme.sin().unwrap())[0];
        assert_eq!(sine.re, 0.0);
        assert!(sine.im.is_infinite() && sine.im.is_sign_positive());
        let cosine = read_c64s(&extreme.cos().unwrap())[0];
        assert!(cosine.re.is_infinite() && cosine.re.is_sign_positive());
        assert_eq!(cosine.im, 0.0);
    }

    /// Asserts elementwise value agreement between the XLA-backed eager array backend and the `ryft-core`
    /// reference array backend ([`CpuArray`]) over a scoped operation list: the elementwise math operations,
    /// element-type conversion, selection, and reduction — including one complex and one `f8` case. This is the
    /// value-level counterpart of the reference-backend parity rule: the reference backend must agree with the
    /// pinned XLA semantics wherever both implement an operation.
    #[test]
    fn test_eager_value_parity_with_reference_backend() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);

        let left_values = [0.5f32, 1.25, 2.0, 3.75];
        let right_values = [4.0f32, 0.25, -1.5, 2.5];
        let left = f32_vector(&client, &mesh, &left_values);
        let right = f32_vector(&client, &mesh, &right_values);
        let reference_left = CpuArray::vector(left_values.to_vec());
        let reference_right = CpuArray::vector(right_values.to_vec());

        // Both backends compute in `f32`, so agreement is checked within a small `f32`-scale tolerance (the two
        // implementations may round transcendental functions differently in the last unit of precision).
        let assert_parity = |device: &Array<'_>, reference: &CpuArray| {
            let device_values = read_f32s(device);
            let reference_values = reference.to_f64s();
            assert_eq!(device_values.len(), reference_values.len());
            for (index, (device_value, reference_value)) in
                device_values.iter().zip(reference_values.iter()).enumerate()
            {
                assert!(
                    (f64::from(*device_value) - reference_value).abs() < 1e-5,
                    "element {index} disagrees: XLA computed {device_value} but the reference backend computed \
                     {reference_value}",
                );
            }
        };

        assert_parity(&left.add(&right).unwrap(), &reference_left.add(&reference_right).unwrap());
        assert_parity(&left.sub(&right).unwrap(), &reference_left.sub(&reference_right).unwrap());
        assert_parity(&left.mul(&right).unwrap(), &reference_left.mul(&reference_right).unwrap());
        assert_parity(&left.div(&right).unwrap(), &reference_left.div(&reference_right).unwrap());
        assert_parity(&right.neg().unwrap(), &reference_right.neg().unwrap());
        assert_parity(&right.abs().unwrap(), &reference_right.abs().unwrap());
        assert_parity(&left.sin().unwrap(), &reference_left.sin().unwrap());
        assert_parity(&left.cos().unwrap(), &reference_left.cos().unwrap());
        assert_parity(&left.atan2(&right).unwrap(), &reference_left.atan2(&reference_right).unwrap());
        assert_parity(&left.exp().unwrap(), &reference_left.exp().unwrap());
        assert_parity(&left.log().unwrap(), &reference_left.log().unwrap());
        assert_parity(&left.sqrt().unwrap(), &reference_left.sqrt().unwrap());
        assert_parity(&left.rsqrt().unwrap(), &reference_left.rsqrt().unwrap());
        assert_parity(&left.tanh().unwrap(), &reference_left.tanh().unwrap());
        assert_parity(&left.logistic().unwrap(), &reference_left.logistic().unwrap());
        assert_parity(&left.erf().unwrap(), &reference_left.erf().unwrap());
        assert_parity(&left.pow(&right).unwrap(), &reference_left.pow(&reference_right).unwrap());
        assert_parity(&right.sign().unwrap(), &reference_right.sign().unwrap());
        assert_parity(&right.floor().unwrap(), &reference_right.floor().unwrap());
        assert_parity(&right.ceil().unwrap(), &reference_right.ceil().unwrap());
        // The `right` vector contains the exact ties `-1.5` and `2.5`, checking the round-to-nearest-even policy
        // against the reference backend.
        assert_parity(&right.round().unwrap(), &reference_right.round().unwrap());
        assert_parity(&left.maximum(&right).unwrap(), &reference_left.maximum(&reference_right).unwrap());
        assert_parity(&left.minimum(&right).unwrap(), &reference_left.minimum(&reference_right).unwrap());
        assert_parity(&left.remainder(&right).unwrap(), &reference_left.remainder(&reference_right).unwrap());

        // Element-type conversion agrees, including the exact `f8e4m3fn` encodings: the device payload bytes match
        // the reference backend's encoded bits bit for bit.
        let converted = left.convert_element_type(DataType::F8E4M3FN).unwrap();
        let converted_bytes = shard_host_bytes(&converted.addressable_shards().next().unwrap()).unwrap();
        let reference_bits = reference_left
            .convert_element_type(DataType::F8E4M3FN)
            .unwrap()
            .values()
            .iter()
            .map(|value| value.low_precision_float_bits().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(converted_bytes, reference_bits);

        // Selection agrees.
        let condition_values = [true, false, true, false];
        let condition = boolean_vector(&client, &mesh, &condition_values);
        let reference_condition = CpuArray::vector(condition_values.to_vec());
        assert_parity(
            &Select::select(&condition, &left, &right).unwrap(),
            &Select::select(&reference_condition, &reference_left, &reference_right).unwrap(),
        );

        // Reductions agree, including the divide-by-count semantics of `Mean`.
        assert_parity(&left.reduce(&[0], ReductionKind::Sum), &reference_left.reduce(&[0], ReductionKind::Sum));
        assert_parity(&left.reduce(&[0], ReductionKind::Mean), &reference_left.reduce(&[0], ReductionKind::Mean));
        assert_parity(&left.reduce(&[0], ReductionKind::Max), &reference_left.reduce(&[0], ReductionKind::Max));
        assert_parity(&left.reduce(&[0], ReductionKind::Min), &reference_left.reduce(&[0], ReductionKind::Min));

        // Integer reductions agree exactly, including the truncating integer division of `Mean`.
        let integer_values = [5i32, -2, 7, 0];
        let integer = Array::from_host_buffer(
            &client,
            replicated_type(&mesh, DataType::I32, &[integer_values.len()]),
            mesh.clone(),
            values_to_bytes::<i32>(&integer_values).as_slice(),
        )
        .unwrap();
        let reference_integer = CpuArray::new(
            ArrayType::new(DataType::I32, Shape::new(vec![Size::Static(integer_values.len())])),
            integer_values.iter().copied().map(Scalar::I32).collect(),
        )
        .unwrap();
        for kind in [ReductionKind::Sum, ReductionKind::Mean, ReductionKind::Max, ReductionKind::Min] {
            let device_values = read_i32s(&integer.reduce(&[0], kind));
            let reference_values = reference_integer
                .reduce(&[0], kind)
                .values()
                .iter()
                .map(|value| match value {
                    Scalar::I32(value) => *value,
                    other => panic!("expected an i32 reduction result but got {other:?}"),
                })
                .collect::<Vec<_>>();
            assert_eq!(device_values, reference_values, "integer '{kind}' reduction disagrees");
        }

        // Complex multiplication agrees.
        let complex_left_value = num_complex::Complex::new(1.5f32, -2.0);
        let complex_right_value = num_complex::Complex::new(0.5f32, 3.0);
        let complex_left = c64_scalar(&client, &mesh, complex_left_value);
        let complex_right = c64_scalar(&client, &mesh, complex_right_value);
        let device_product = read_c64s(&complex_left.mul(&complex_right).unwrap())[0];
        let reference_product =
            CpuArray::scalar(complex_left_value).mul(&CpuArray::scalar(complex_right_value)).unwrap();
        let Scalar::C64(reference_product) = reference_product.values()[0] else {
            panic!("expected a c64 reference product");
        };
        assert!((device_product - reference_product).norm() < 1e-5);
    }

    /// The error function agrees between the XLA-backed eager array backend (which lowers to `chlo.erf` and relies
    /// on the XLA compiler legalizing it to a StableHLO polynomial approximation) and the reference backend's
    /// double-precision FDLIBM evaluation. The grid covers every rational-approximation regime of the reference
    /// implementation on both signs — the small-argument series, the primary interval, the interval around one,
    /// both complementary-function tail regimes — plus zero and the saturated |x| ≥ 6 regime.
    #[test]
    fn test_eager_erf_parity_with_reference_backend() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);

        let values = [
            -6.5f32, -4.0, -3.0, -2.0, -1.5, -1.0, -0.9, -0.5, -0.1, -1e-3, -1e-10, 0.0, 1e-10, 1e-3, 0.1, 0.5, 0.9,
            1.0, 1.2, 1.25, 2.0, 2.9, 3.0, 4.0, 6.5,
        ];
        let device_values = read_f32s(&f32_vector(&client, &mesh, &values).erf().unwrap());
        let reference_values = CpuArray::vector(values.to_vec()).erf().unwrap().to_f64s();
        assert_eq!(device_values.len(), reference_values.len());
        for ((input, device_value), reference_value) in
            values.iter().zip(device_values.iter()).zip(reference_values.iter())
        {
            // Both backends compute at `f32` precision (the device through XLA's polynomial legalization of
            // `chlo.erf` and the reference through a rounded double-precision evaluation), so agreement is checked
            // within an `f32`-scale relative tolerance with an absolute floor for the near-zero inputs.
            let tolerance = 1e-6f64.max(1e-6 * reference_value.abs());
            assert!(
                (f64::from(*device_value) - reference_value).abs() < tolerance,
                "erf({input}) disagrees: XLA computed {device_value} but the reference backend computed \
                 {reference_value}",
            );
        }

        // Both backends saturate exactly at the extremes.
        assert_eq!(device_values[0], -1.0);
        assert_eq!(*device_values.last().unwrap(), 1.0);
        assert_eq!(reference_values[0], -1.0);
        assert_eq!(*reference_values.last().unwrap(), 1.0);
    }

    /// The traced custom-call operation executes a registered XLA FFI handler through the eager capability path:
    /// dispatch compiles a cached single-operation program whose `stablehlo.custom_call` resolves to the
    /// `ryft.test.add_one` handler at execution time.
    #[test]
    fn test_eager_custom_call_executes_registered_ffi_handler() {
        use ryft_core::operations::custom_call::{CustomCall, CustomCallOperation};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        ensure_add_one_handler_registered(&client).unwrap();
        let mesh = cpu_mesh(&client);
        let input = f32_vector(&client, &mesh, &[1.5, 2.5]);

        let operation =
            CustomCallOperation::new(ADD_ONE_CUSTOM_CALL_TARGET, vec![replicated_type(&mesh, DataType::F32, &[2])]);
        let outputs = CustomCall::custom_call(&operation, std::slice::from_ref(&input)).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(read_f32s(&outputs[0]), vec![2.5, 3.5]);

        // A typed `f64` attribute reaches the handler through the `backend_config` dictionary.
        let operation = operation.with_attribute("increment", 2.5);
        let outputs = CustomCall::custom_call(&operation, std::slice::from_ref(&input)).unwrap();
        assert_eq!(read_f32s(&outputs[0]), vec![4.0, 5.0]);
    }

    /// A custom call wrapped with `custom_vjp` differentiates through the user-provided rule while the primal
    /// executes the registered FFI handler, which is the documented pairing for differentiable foreign kernels
    /// (the bare operation rejects differentiation).
    #[test]
    fn test_eager_custom_call_differentiates_through_custom_vjp() {
        use ryft_core::operations::custom_call::{CustomCall, CustomCallOperation};
        use ryft_core::tracing::DomainTracer;
        use ryft_core::tracing_v2::operations::custom_vjp;

        use crate::XlaDomain;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        ensure_add_one_handler_registered(&client).unwrap();
        let mesh = cpu_mesh(&client);
        let input = f32_vector(&client, &mesh, &[1.5, 2.5]);
        let output_type = replicated_type(&mesh, DataType::F32, &[2]);
        let domain = input.execution_domain();

        let add_one = move |x: &DomainTracer<XlaDomain<'_>>| {
            let operation = CustomCallOperation::new(ADD_ONE_CUSTOM_CALL_TARGET, vec![output_type.clone()]);
            Ok(CustomCall::custom_call(&operation, std::slice::from_ref(x))?.remove(0))
        };
        let function = custom_vjp::<XlaDomain<'_>, _, _, _, _, _, _>(
            {
                let add_one = add_one.clone();
                move |x: DomainTracer<XlaDomain<'_>>| add_one(&x)
            },
            move |x: DomainTracer<XlaDomain<'_>>| Ok((add_one(&x)?, ())),
            // d(x + 1)/dx is the identity, so the backward rule passes the cotangent through.
            |(), cotangent| Ok(cotangent),
        );
        let (value, gradient) = domain
            .value_and_gradient(|x| function.call(x).unwrap().reduce(&[0], ReductionKind::Sum), input)
            .unwrap();
        assert_eq!(read_f32s(&value), vec![6.0]);
        assert_eq!(read_f32s(&gradient), vec![1.0, 1.0]);
    }

    /// Sorting, top-k, and argmax agree between the XLA-backed eager array backend and the reference array backend,
    /// including stable-tie routing (equal keys keep their original order, so ranking ties select the lowest
    /// index) and NaN placement (NaNs order above `+∞` in the total order, so `argmax` reports a NaN's index).
    #[test]
    fn test_eager_sort_and_ranking_parity_with_reference_backend() {
        use ryft_core::operations::sort::{ArgMax, ArgMin, Sort, SortDirection, TopK};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);

        let key_values = [3.0f32, 1.0, 3.0, -0.0, 0.0, 2.0];
        let payload_values = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        let keys = f32_vector(&client, &mesh, &key_values);
        let payloads = f32_vector(&client, &mesh, &payload_values);
        let reference_keys = CpuArray::vector(key_values.to_vec());
        let reference_payloads = CpuArray::vector(payload_values.to_vec());

        for direction in [SortDirection::Ascending, SortDirection::Descending] {
            let sorted = Sort::sort(&[keys.clone(), payloads.clone()], 0, direction).unwrap();
            let reference_sorted =
                Sort::sort(&[reference_keys.clone(), reference_payloads.clone()], 0, direction).unwrap();
            for (device, reference) in sorted.iter().zip(reference_sorted.iter()) {
                // Bit-level comparison keeps the `-0.0` versus `+0.0` total-order placement observable.
                let device_bits = read_f32s(device).into_iter().map(f32::to_bits).collect::<Vec<_>>();
                let reference_bits =
                    reference.to_f64s().into_iter().map(|value| (value as f32).to_bits()).collect::<Vec<_>>();
                assert_eq!(device_bits, reference_bits);
            }
        }

        let (device_values, device_indices) = keys.top_k(3, 0).unwrap();
        let (reference_values, reference_indices) = reference_keys.top_k(3, 0).unwrap();
        let device_value_f64s = read_f32s(&device_values).iter().map(|value| f64::from(*value)).collect::<Vec<_>>();
        assert_eq!(device_value_f64s, reference_values.to_f64s());
        let device_index_f64s = read_i32s(&device_indices).iter().map(|index| f64::from(*index)).collect::<Vec<_>>();
        assert_eq!(device_index_f64s, reference_indices.to_f64s());
        // Ties select the lowest index: both threes appear before the two, and index 0 precedes index 2.
        assert_eq!(read_i32s(&device_indices), vec![0, 2, 5]);

        let nan_values = [1.0f32, f32::NAN, 3.0];
        let nan_keys = f32_vector(&client, &mesh, &nan_values);
        let reference_nan_keys = CpuArray::vector(nan_values.to_vec());
        assert_eq!(read_i32s(&nan_keys.argmax(0).unwrap()), vec![1]);
        assert_eq!(reference_nan_keys.argmax(0).unwrap().to_f64s(), vec![1.0]);
        assert_eq!(read_i32s(&nan_keys.argmin(0).unwrap()), vec![0]);
        assert_eq!(reference_nan_keys.argmin(0).unwrap().to_f64s(), vec![0.0]);
    }

    /// A two-key lexicographic sort agrees between the XLA-backed eager array backend and the reference array
    /// backend: duplicates in the `i32` primary key fall through to the `f32` secondary key (compared with
    /// `TOTALORDER` semantics), full ties keep their original order (the sort is stable), and the passenger
    /// co-permutes.
    #[test]
    fn test_eager_multi_key_sort_parity_with_reference_backend() {
        use ryft_core::operations::sort::{Sort, SortDirection};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);

        let primary_values = [2i32, 1, 2, 1, 2, 1];
        let secondary_values = [0.5f32, 3.0, -1.0, 1.0, 0.5, 2.0];
        let passenger_values = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        let primary_type = replicated_type(&mesh, DataType::I32, &[primary_values.len()]);
        let primary = Array::from_host_buffer(
            &client,
            primary_type,
            mesh.clone(),
            values_to_bytes::<i32>(&primary_values).as_slice(),
        )
        .unwrap();
        let secondary = f32_vector(&client, &mesh, &secondary_values);
        let passenger = f32_vector(&client, &mesh, &passenger_values);
        let reference_primary = CpuArray::new(
            ArrayType::new(DataType::I32, Shape::new(vec![Size::Static(primary_values.len())])),
            primary_values.iter().copied().map(Scalar::I32).collect(),
        )
        .unwrap();
        let reference_secondary = CpuArray::vector(secondary_values.iter().map(|value| f64::from(*value)).collect());
        let reference_passenger = CpuArray::vector(passenger_values.iter().map(|value| f64::from(*value)).collect());

        let cases = [
            // Ascending: primary 1s precede 2s, ties resolve by the secondary key, and the full `(2, 0.5)` tie
            // keeps element 0 before element 4 (the passenger shows the stable order).
            (
                SortDirection::Ascending,
                vec![1, 1, 1, 2, 2, 2],
                vec![1.0f32, 2.0, 3.0, -1.0, 0.5, 0.5],
                vec![40.0f32, 60.0, 20.0, 30.0, 10.0, 50.0],
            ),
            // Descending reverses both key comparisons while keeping the full tie in its original order.
            (
                SortDirection::Descending,
                vec![2, 2, 2, 1, 1, 1],
                vec![0.5f32, 0.5, -1.0, 3.0, 2.0, 1.0],
                vec![10.0f32, 50.0, 30.0, 20.0, 60.0, 40.0],
            ),
        ];
        for (direction, expected_primary, expected_secondary, expected_passenger) in cases {
            let sorted =
                Sort::sort_with_key_count(&[primary.clone(), secondary.clone(), passenger.clone()], 0, direction, 2)
                    .unwrap();
            assert_eq!(read_i32s(&sorted[0]), expected_primary);
            assert_eq!(read_f32s(&sorted[1]), expected_secondary);
            assert_eq!(read_f32s(&sorted[2]), expected_passenger);

            let reference_sorted = Sort::sort_with_key_count(
                &[reference_primary.clone(), reference_secondary.clone(), reference_passenger.clone()],
                0,
                direction,
                2,
            )
            .unwrap();
            let expected_primary_f64s = expected_primary.iter().map(|value| f64::from(*value)).collect::<Vec<_>>();
            let expected_secondary_f64s = expected_secondary.iter().map(|value| f64::from(*value)).collect::<Vec<_>>();
            let expected_passenger_f64s = expected_passenger.iter().map(|value| f64::from(*value)).collect::<Vec<_>>();
            assert_eq!(reference_sorted[0].to_f64s(), expected_primary_f64s);
            assert_eq!(reference_sorted[1].to_f64s(), expected_secondary_f64s);
            assert_eq!(reference_sorted[2].to_f64s(), expected_passenger_f64s);
        }
    }

    /// The reference backend's ThreeFry implementation is bit-exact with XLA's `rng_bit_generator` expansion:
    /// the same `[key, counter]` state produces identical `u32` and `u64` bits and identical advanced states on
    /// both backends.
    #[test]
    fn test_eager_rng_bit_generator_matches_reference_backend() {
        use ryft_core::operations::random::{RandomAlgorithm, RngBitGenerator};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);

        let state_values = [42u64, 7u64];
        let state_type = replicated_type(&mesh, DataType::U64, &[2]);
        let state = Array::from_host_buffer(
            &client,
            state_type,
            mesh.clone(),
            values_to_bytes::<u64>(&state_values).as_slice(),
        )
        .unwrap();
        let reference_state = CpuArray::new(
            ArrayType::new(DataType::U64, Shape::new(vec![Size::Static(2)])),
            state_values.iter().copied().map(Scalar::U64).collect(),
        )
        .unwrap();

        // An odd `u32` element count exercises the padded counter pair and the truncated word layout.
        let u32_output_type = ArrayType::new(DataType::U32, Shape::new(vec![Size::Static(5)]));
        let (device_state, device_bits) = state.rng_bit_generator(RandomAlgorithm::ThreeFry, &u32_output_type).unwrap();
        let (reference_new_state, reference_bits) =
            reference_state.rng_bit_generator(RandomAlgorithm::ThreeFry, &u32_output_type).unwrap();
        let device_words = values_from_bytes::<u32>(
            shard_host_bytes(&device_bits.addressable_shards().next().unwrap()).unwrap().as_slice(),
        );
        let reference_words = reference_bits
            .values()
            .iter()
            .map(|value| match value {
                Scalar::U32(word) => *word,
                _ => panic!("expected u32 reference bits"),
            })
            .collect::<Vec<_>>();
        assert_eq!(device_words, reference_words);
        // Generating five `u32` words runs three cipher invocations, and the counter advances by that invocation
        // count (`7 + 3 = 10`).
        let device_state_words = values_from_bytes::<u64>(
            shard_host_bytes(&device_state.addressable_shards().next().unwrap()).unwrap().as_slice(),
        );
        assert_eq!(device_state_words, vec![42u64, 10u64]);
        assert_eq!(reference_new_state.values(), &[Scalar::U64(42), Scalar::U64(10)]);

        let u64_output_type = ArrayType::new(DataType::U64, Shape::new(vec![Size::Static(3)]));
        let (device_state, device_bits) = state.rng_bit_generator(RandomAlgorithm::ThreeFry, &u64_output_type).unwrap();
        let (reference_new_state, reference_bits) =
            reference_state.rng_bit_generator(RandomAlgorithm::ThreeFry, &u64_output_type).unwrap();
        let device_state_words = values_from_bytes::<u64>(
            shard_host_bytes(&device_state.addressable_shards().next().unwrap()).unwrap().as_slice(),
        );
        assert_eq!(
            device_state_words,
            reference_new_state
                .values()
                .iter()
                .map(|value| match value {
                    Scalar::U64(word) => *word,
                    _ => panic!("expected a u64 reference state"),
                })
                .collect::<Vec<_>>(),
        );
        let device_words = values_from_bytes::<u64>(
            shard_host_bytes(&device_bits.addressable_shards().next().unwrap()).unwrap().as_slice(),
        );
        let reference_words = reference_bits
            .values()
            .iter()
            .map(|value| match value {
                Scalar::U64(word) => *word,
                _ => panic!("expected u64 reference bits"),
            })
            .collect::<Vec<_>>();
        assert_eq!(device_words, reference_words);
    }

    /// The reference backend's Philox implementation is bit-exact with XLA's `rng_bit_generator` expansion:
    /// the same `[key, counter]` state (with the 128-bit counter split into its low and high `u64` halves)
    /// produces identical `u32` and `u64` bits and identical advanced states on both backends.
    #[test]
    fn test_eager_philox_rng_bit_generator_matches_reference_backend() {
        use ryft_core::operations::random::{RandomAlgorithm, RngBitGenerator};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);

        let state_values = [42u64, 7u64, 9u64];
        let state_type = replicated_type(&mesh, DataType::U64, &[3]);
        let state = Array::from_host_buffer(
            &client,
            state_type,
            mesh.clone(),
            values_to_bytes::<u64>(&state_values).as_slice(),
        )
        .unwrap();
        let reference_state = CpuArray::new(
            ArrayType::new(DataType::U64, Shape::new(vec![Size::Static(3)])),
            state_values.iter().copied().map(Scalar::U64).collect(),
        )
        .unwrap();

        // An odd `u32` element count exercises the padded counter quad and the truncated word layout.
        let u32_output_type = ArrayType::new(DataType::U32, Shape::new(vec![Size::Static(5)]));
        let (device_state, device_bits) = state.rng_bit_generator(RandomAlgorithm::Philox, &u32_output_type).unwrap();
        let (reference_new_state, reference_bits) =
            reference_state.rng_bit_generator(RandomAlgorithm::Philox, &u32_output_type).unwrap();
        let device_words = values_from_bytes::<u32>(
            shard_host_bytes(&device_bits.addressable_shards().next().unwrap()).unwrap().as_slice(),
        );
        let reference_words = reference_bits
            .values()
            .iter()
            .map(|value| match value {
                Scalar::U32(word) => *word,
                _ => panic!("expected u32 reference bits"),
            })
            .collect::<Vec<_>>();
        assert_eq!(device_words, reference_words);
        // Generating five `u32` words runs two cipher invocations, and the low counter half advances by that
        // invocation count (`7 + 2 = 9`) while the key and high counter half are unchanged.
        let device_state_words = values_from_bytes::<u64>(
            shard_host_bytes(&device_state.addressable_shards().next().unwrap()).unwrap().as_slice(),
        );
        assert_eq!(device_state_words, vec![42u64, 9u64, 9u64]);
        assert_eq!(reference_new_state.values(), &[Scalar::U64(42), Scalar::U64(9), Scalar::U64(9)]);

        let u64_output_type = ArrayType::new(DataType::U64, Shape::new(vec![Size::Static(3)]));
        let (device_state, device_bits) = state.rng_bit_generator(RandomAlgorithm::Philox, &u64_output_type).unwrap();
        let (reference_new_state, reference_bits) =
            reference_state.rng_bit_generator(RandomAlgorithm::Philox, &u64_output_type).unwrap();
        let device_state_words = values_from_bytes::<u64>(
            shard_host_bytes(&device_state.addressable_shards().next().unwrap()).unwrap().as_slice(),
        );
        assert_eq!(
            device_state_words,
            reference_new_state
                .values()
                .iter()
                .map(|value| match value {
                    Scalar::U64(word) => *word,
                    _ => panic!("expected a u64 reference state"),
                })
                .collect::<Vec<_>>(),
        );
        let device_words = values_from_bytes::<u64>(
            shard_host_bytes(&device_bits.addressable_shards().next().unwrap()).unwrap().as_slice(),
        );
        let reference_words = reference_bits
            .values()
            .iter()
            .map(|value| match value {
                Scalar::U64(word) => *word,
                _ => panic!("expected u64 reference bits"),
            })
            .collect::<Vec<_>>();
        assert_eq!(device_words, reference_words);
    }

    /// The composed random distributions agree between the XLA-backed eager array backend and the reference
    /// array backend: uniform `f32` samples are bit-identical (every step of the composition is exact arithmetic
    /// over bit-identical ThreeFry draws), normal samples agree within floating-point tolerance (the Box–Muller
    /// transform routes through transcendental functions), and categorical samples with well-separated logits
    /// select identical indices.
    #[test]
    fn test_eager_random_distributions_parity_with_reference_backend() {
        use ryft_core::operations::random::Random;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);

        let state_values = [11u64, 3u64];
        let state = Array::from_host_buffer(
            &client,
            replicated_type(&mesh, DataType::U64, &[2]),
            mesh.clone(),
            values_to_bytes::<u64>(&state_values).as_slice(),
        )
        .unwrap();
        let reference_state = CpuArray::new(
            ArrayType::new(DataType::U64, Shape::new(vec![Size::Static(2)])),
            state_values.iter().copied().map(Scalar::U64).collect(),
        )
        .unwrap();

        let shape = Shape::new(vec![Size::Static(8)]);
        let (_, device_uniform) = state.uniform(shape.clone(), DataType::F32).unwrap();
        let (_, reference_uniform) = reference_state.uniform(shape.clone(), DataType::F32).unwrap();
        let device_bits = read_f32s(&device_uniform).into_iter().map(f32::to_bits).collect::<Vec<_>>();
        let reference_bits =
            reference_uniform.to_f64s().into_iter().map(|value| (value as f32).to_bits()).collect::<Vec<_>>();
        assert_eq!(device_bits, reference_bits);
        for value in reference_uniform.to_f64s() {
            assert!((0.0..1.0).contains(&value), "uniform sample {value} escapes [0, 1)");
        }

        let (_, device_normal) = state.normal(shape.clone(), DataType::F32).unwrap();
        let (_, reference_normal) = reference_state.normal(shape, DataType::F32).unwrap();
        for (device_value, reference_value) in read_f32s(&device_normal).iter().zip(reference_normal.to_f64s()) {
            assert!(
                (f64::from(*device_value) - reference_value).abs() < 1e-5,
                "normal samples disagree: XLA computed {device_value} but the reference backend computed \
                 {reference_value}",
            );
        }

        let logit_values = [0.0f32, 10.0, -3.0, 2.0];
        let device_logits = f32_vector(&client, &mesh, &logit_values);
        let reference_logits = CpuArray::vector(logit_values.iter().map(|value| f64::from(*value)).collect())
            .convert_element_type(DataType::F32)
            .unwrap();
        let (_, device_samples) = state.categorical(&device_logits, 0).unwrap();
        let (_, reference_samples) = reference_state.categorical(&reference_logits, 0).unwrap();
        assert_eq!(read_i32s(&device_samples), vec![1]);
        assert_eq!(reference_samples.values(), &[Scalar::I32(1)]);

        let (_, device_keys) = state.split_key(2).unwrap();
        let (_, reference_keys) = reference_state.split_key(2).unwrap();
        for (device_key, reference_key) in device_keys.iter().zip(reference_keys.iter()) {
            let device_words = values_from_bytes::<u64>(
                shard_host_bytes(&device_key.addressable_shards().next().unwrap()).unwrap().as_slice(),
            );
            let reference_words = reference_key
                .values()
                .iter()
                .map(|value| match value {
                    Scalar::U64(word) => *word,
                    _ => panic!("expected u64 key words"),
                })
                .collect::<Vec<_>>();
            assert_eq!(device_words, reference_words);
        }
    }

    /// An accumulation-typed dot (`f8e4m3fn × f8e4m3fn → f32`) agrees between the XLA-backed eager array backend
    /// and the reference array backend: the operands stay at the narrow element type on device and the contraction
    /// accumulates at `f32`. Every value used is exactly representable in `f8e4m3fn`, so both backends are exact.
    #[test]
    fn test_eager_accumulation_typed_dot_parity_with_reference_backend() {
        use ryft_core::operations::math::DotDimensionNumbers;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);

        let operand_type = ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Size::Static(2), Size::Static(2)]));
        let lhs_values = [0.5f64, 1.0, 1.5, 2.0];
        let rhs_values = [1.0f64, 0.5, 0.5, 1.0];
        let reference_lhs = CpuArray::from_f64s(operand_type.clone(), lhs_values.to_vec());
        let reference_rhs = CpuArray::from_f64s(operand_type.clone(), rhs_values.to_vec());
        let lhs_bytes = reference_lhs
            .values()
            .iter()
            .map(|value| value.low_precision_float_bits().unwrap())
            .collect::<Vec<_>>();
        let rhs_bytes = reference_rhs
            .values()
            .iter()
            .map(|value| value.low_precision_float_bits().unwrap())
            .collect::<Vec<_>>();
        let device_type = replicated_type(&mesh, DataType::F8E4M3FN, &[2, 2]);
        let lhs = Array::from_host_buffer(&client, device_type.clone(), mesh.clone(), lhs_bytes.as_slice()).unwrap();
        let rhs = Array::from_host_buffer(&client, device_type, mesh.clone(), rhs_bytes.as_slice()).unwrap();

        let device_product = lhs.dot_with_accumulation_type(&rhs, &DotDimensionNumbers::matmul(), DataType::F32);
        let reference_product =
            reference_lhs.dot_with_accumulation_type(&reference_rhs, &DotDimensionNumbers::matmul(), DataType::F32);
        assert_eq!(device_product.r#type().data_type(), DataType::F32);
        let device_value_f64s = read_f32s(&device_product).iter().map(|value| f64::from(*value)).collect::<Vec<_>>();
        assert_eq!(device_value_f64s, reference_product.to_f64s());
        assert_eq!(reference_product.to_f64s(), vec![1.0, 1.25, 2.5, 2.75]);
    }

    /// A block-scaled (NVFP4-style) dot agrees between the XLA-backed eager array backend and the reference array
    /// backend on CPU, where the operation lowers to the portable dequantization composition (the CUDA fast path
    /// emits the `__op$block_scaled_dot` custom call instead). Every element and scale is exactly representable,
    /// so both backends are exact.
    #[test]
    fn test_eager_scaled_dot_parity_with_reference_backend() {
        use ryft_core::operations::math::ScaledDot;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);

        let element_type = ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![Size::Static(2), Size::Static(16)]));
        let scale_type = ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Size::Static(2), Size::Static(1)]));
        const F4_CANDIDATES: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, -0.5, -1.0];
        let element_values =
            |seed: usize| (0..32).map(|index| F4_CANDIDATES[(index * 5 + seed) % 8]).collect::<Vec<_>>();
        let reference_lhs = CpuArray::from_f64s(element_type.clone(), element_values(0));
        let reference_rhs = CpuArray::from_f64s(element_type.clone(), element_values(3));
        let reference_lhs_scales = CpuArray::from_f64s(scale_type.clone(), vec![0.5, 2.0]);
        let reference_rhs_scales = CpuArray::from_f64s(scale_type.clone(), vec![2.0, 0.5]);
        let bits = |reference: &CpuArray| {
            reference
                .values()
                .iter()
                .map(|value| value.low_precision_float_bits().unwrap())
                .collect::<Vec<u8>>()
        };
        let device_element_type = replicated_type(&mesh, DataType::F4E2M1FN, &[2, 16]);
        let device_scale_type = replicated_type(&mesh, DataType::F8E4M3FN, &[2, 1]);
        let lhs = Array::from_host_buffer(
            &client,
            device_element_type.clone(),
            mesh.clone(),
            bits(&reference_lhs).as_slice(),
        )
        .unwrap();
        let rhs = Array::from_host_buffer(&client, device_element_type, mesh.clone(), bits(&reference_rhs).as_slice())
            .unwrap();
        let lhs_scales = Array::from_host_buffer(
            &client,
            device_scale_type.clone(),
            mesh.clone(),
            bits(&reference_lhs_scales).as_slice(),
        )
        .unwrap();
        let rhs_scales =
            Array::from_host_buffer(&client, device_scale_type, mesh.clone(), bits(&reference_rhs_scales).as_slice())
                .unwrap();

        let device_product = lhs.scaled_dot(&lhs_scales, &rhs, &rhs_scales, 16, DataType::F32).unwrap();
        let reference_product = reference_lhs
            .scaled_dot(&reference_lhs_scales, &reference_rhs, &reference_rhs_scales, 16, DataType::F32)
            .unwrap();
        assert_eq!(device_product.r#type().data_type(), DataType::F32);
        let device_value_f64s = read_f32s(&device_product).iter().map(|value| f64::from(*value)).collect::<Vec<_>>();
        assert_eq!(device_value_f64s, reference_product.to_f64s());
    }

    /// Dot-product attention agrees between the XLA-backed eager array backend and the reference array backend on
    /// CPU, where the operation lowers to the portable StableHLO composition (the CUDA fast path emits the
    /// `__cudnn$fmhaSoftmax` custom call instead), for both the unmasked and the causal variants.
    #[test]
    fn test_eager_dot_product_attention_parity_with_reference_backend() {
        use ryft_core::operations::attention::{AttentionMask, DotProductAttention};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);

        // Small `BTNH` shape `[1, 4, 2, 3]` in `f32` with deterministic pseudo-random operand values.
        let dimensions = [1usize, 4, 2, 3];
        let operand_type = ArrayType::new(
            DataType::F32,
            Shape::new(dimensions.iter().map(|&dimension| Size::Static(dimension)).collect()),
        );
        let device_type = replicated_type(&mesh, DataType::F32, &dimensions);
        let query_values: Vec<f64> = (0..24).map(|i| ((i * 7 % 11) as f64 - 5.0) * 0.25).collect();
        let key_values: Vec<f64> = (0..24).map(|i| ((i * 5 % 13) as f64 - 6.0) * 0.25).collect();
        let value_values: Vec<f64> = (0..24).map(|i| ((i * 3 % 7) as f64 - 3.0) * 0.5).collect();
        let device = |values: &[f64]| {
            let values = values.iter().map(|&value| value as f32).collect::<Vec<_>>();
            Array::from_host_buffer(&client, device_type.clone(), mesh.clone(), values_to_bytes(&values).as_slice())
                .unwrap()
        };
        let reference = |values: &[f64]| CpuArray::from_f64s(operand_type.clone(), values.to_vec());
        let scale = 0.5;

        for mask in [AttentionMask::None, AttentionMask::Causal] {
            let device_output = device(&query_values)
                .dot_product_attention(&device(&key_values), &device(&value_values), scale, mask, None)
                .unwrap();
            let reference_output = reference(&query_values)
                .dot_product_attention(&reference(&key_values), &reference(&value_values), scale, mask, None)
                .unwrap();
            assert_eq!(device_output.r#type().data_type(), DataType::F32);
            assert_eq!(device_output.shape().dimensions(), &dimensions);
            for (device_value, reference_value) in
                read_f32s(&device_output).iter().map(|value| f64::from(*value)).zip(reference_output.to_f64s())
            {
                assert!(
                    (device_value - reference_value).abs() < 1e-5,
                    "mask {mask}: expected {reference_value} but got {device_value}",
                );
            }
        }
    }

    /// The extended attention features — grouped-query heads, a broadcast bias, and a sliding window — agree
    /// between the XLA-backed eager array backend (the portable StableHLO composition on CPU) and the reference
    /// array backend, together with the activation (log-sum-exp) statistic.
    #[test]
    fn test_eager_dot_product_attention_extensions_parity_with_reference_backend() {
        use ryft_core::operations::attention::{AttentionMask, DotProductAttention};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);

        // Grouped-query `BTNH` shapes: `query [1, 4, 4, 3]` over `key`/`value [1, 5, 2, 3]` with a broadcast bias
        // `[1, 1, 4, 5]` and a sliding window of 2 under the causal mask.
        let query_dimensions = [1usize, 4, 4, 3];
        let key_value_dimensions = [1usize, 5, 2, 3];
        let bias_dimensions = [1usize, 1, 4, 5];
        let query_values: Vec<f64> = (0..48).map(|i| ((i * 7 % 11) as f64 - 5.0) * 0.25).collect();
        let key_values: Vec<f64> = (0..30).map(|i| ((i * 5 % 13) as f64 - 6.0) * 0.25).collect();
        let value_values: Vec<f64> = (0..30).map(|i| ((i * 3 % 7) as f64 - 3.0) * 0.5).collect();
        let bias_values: Vec<f64> = (0..20).map(|i| ((i * 11 % 17) as f64 - 8.0) * 0.125).collect();
        let device = |values: &[f64], dimensions: &[usize]| {
            let values = values.iter().map(|&value| value as f32).collect::<Vec<_>>();
            let r#type = replicated_type(&mesh, DataType::F32, dimensions);
            Array::from_host_buffer(&client, r#type, mesh.clone(), values_to_bytes(&values).as_slice()).unwrap()
        };
        let reference = |values: &[f64], dimensions: &[usize]| {
            let shape = Shape::new(dimensions.iter().map(|&dimension| Size::Static(dimension)).collect());
            CpuArray::from_f64s(ArrayType::new(DataType::F32, shape), values.to_vec())
        };
        let scale = 0.5;
        let mask = AttentionMask::Causal;
        let window = Some(2);

        let (device_output, device_activation) = device(&query_values, &query_dimensions)
            .dot_product_attention_with_activation(
                &device(&key_values, &key_value_dimensions),
                &device(&value_values, &key_value_dimensions),
                Some(&device(&bias_values, &bias_dimensions)),
                None,
                scale,
                mask,
                window,
                None,
            )
            .unwrap();
        let (reference_output, reference_activation) = reference(&query_values, &query_dimensions)
            .dot_product_attention_with_activation(
                &reference(&key_values, &key_value_dimensions),
                &reference(&value_values, &key_value_dimensions),
                Some(&reference(&bias_values, &bias_dimensions)),
                None,
                scale,
                mask,
                window,
                None,
            )
            .unwrap();
        for (device_values, reference_values) in [
            (read_f32s(&device_output), reference_output.to_f64s()),
            (read_f32s(&device_activation), reference_activation.to_f64s()),
        ] {
            for (device_value, reference_value) in device_values.iter().zip(reference_values) {
                assert!(
                    (f64::from(*device_value) - reference_value).abs() < 1e-5,
                    "expected {reference_value} but got {device_value}",
                );
            }
        }
    }

    /// Variable-sequence-length (padded) attention agrees between the XLA-backed eager array backend and the
    /// reference array backend, including the exact zeros in the out-of-range query rows of both the attended
    /// output and the activation statistic.
    #[test]
    fn test_eager_dot_product_attention_padding_parity_with_reference_backend() {
        use ryft_core::operations::attention::{AttentionMask, DotProductAttention};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);

        let dimensions = [2usize, 4, 2, 3];
        let key_value_dimensions = [2usize, 5, 2, 3];
        let query_values: Vec<f64> = (0..48).map(|i| ((i * 7 % 11) as f64 - 5.0) * 0.25).collect();
        let key_values: Vec<f64> = (0..60).map(|i| ((i * 5 % 13) as f64 - 6.0) * 0.25).collect();
        let value_values: Vec<f64> = (0..60).map(|i| ((i * 3 % 7) as f64 - 3.0) * 0.5).collect();
        let query_lengths = [3i32, 2];
        let key_value_lengths = [4i32, 2];
        let device = |values: &[f64], dimensions: &[usize]| {
            let values = values.iter().map(|&value| value as f32).collect::<Vec<_>>();
            let r#type = replicated_type(&mesh, DataType::F32, dimensions);
            Array::from_host_buffer(&client, r#type, mesh.clone(), values_to_bytes(&values).as_slice()).unwrap()
        };
        let device_lengths = |values: &[i32]| {
            let r#type = replicated_type(&mesh, DataType::I32, &[values.len()]);
            Array::from_host_buffer(&client, r#type, mesh.clone(), values_to_bytes(values).as_slice()).unwrap()
        };
        let reference = |values: &[f64], dimensions: &[usize]| {
            let shape = Shape::new(dimensions.iter().map(|&dimension| Size::Static(dimension)).collect());
            CpuArray::from_f64s(ArrayType::new(DataType::F32, shape), values.to_vec())
        };
        let reference_lengths = |values: &[i32]| {
            let shape = Shape::new(vec![Size::Static(values.len())]);
            CpuArray::from_f64s(
                ArrayType::new(DataType::I32, shape),
                values.iter().map(|&value| f64::from(value)).collect(),
            )
        };
        let scale = 0.5;
        let mask = AttentionMask::Causal;

        let (device_output, device_activation) = device(&query_values, &dimensions)
            .dot_product_attention_with_activation(
                &device(&key_values, &key_value_dimensions),
                &device(&value_values, &key_value_dimensions),
                None,
                Some((&device_lengths(&query_lengths), &device_lengths(&key_value_lengths))),
                scale,
                mask,
                None,
                None,
            )
            .unwrap();
        let (reference_output, reference_activation) = reference(&query_values, &dimensions)
            .dot_product_attention_with_activation(
                &reference(&key_values, &key_value_dimensions),
                &reference(&value_values, &key_value_dimensions),
                None,
                Some((&reference_lengths(&query_lengths), &reference_lengths(&key_value_lengths))),
                scale,
                mask,
                None,
                None,
            )
            .unwrap();
        let device_output_values = read_f32s(&device_output);
        let device_activation_values = read_f32s(&device_activation);
        for (device_values, reference_values) in [
            (&device_output_values, reference_output.to_f64s()),
            (&device_activation_values, reference_activation.to_f64s()),
        ] {
            for (device_value, reference_value) in device_values.iter().zip(reference_values) {
                assert!(
                    (f64::from(*device_value) - reference_value).abs() < 1e-5,
                    "expected {reference_value} but got {device_value}",
                );
            }
        }
        // Out-of-range query rows are exact zeros on the device path, mirroring the fused kernels' memzeroed
        // outputs.
        for b in 0..2 {
            for i in (query_lengths[b] as usize)..4 {
                for n in 0..2 {
                    for d in 0..3 {
                        assert_eq!(device_output_values[((b * 4 + i) * 2 + n) * 3 + d], 0.0);
                    }
                    assert_eq!(device_activation_values[(b * 2 + n) * 4 + i], 0.0);
                }
            }
        }
    }

    /// The attention backward operation agrees between the XLA-backed eager array backend (the backward composition
    /// fallback on CPU) and the reference array backend, including the bias cotangent and the padded
    /// (variable-sequence-length) gradient zeroing.
    #[test]
    fn test_eager_dot_product_attention_backward_parity_with_reference_backend() {
        use ryft_core::operations::attention::{AttentionMask, DotProductAttention, DotProductAttentionBackward};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);

        // Grouped-query `BTNH` shapes with a broadcast bias and per-item sequence lengths, exercising the group
        // sums, the bias cotangent, and the padded-gradient zeroing all at once.
        let query_dimensions = [2usize, 3, 2, 3];
        let key_value_dimensions = [2usize, 4, 1, 3];
        let bias_dimensions = [1usize, 2, 3, 4];
        let query_values: Vec<f64> = (0..36).map(|i| ((i * 7 % 11) as f64 - 5.0) * 0.25).collect();
        let key_values: Vec<f64> = (0..24).map(|i| ((i * 5 % 13) as f64 - 6.0) * 0.25).collect();
        let value_values: Vec<f64> = (0..24).map(|i| ((i * 3 % 7) as f64 - 3.0) * 0.5).collect();
        let bias_values: Vec<f64> = (0..24).map(|i| ((i * 11 % 17) as f64 - 8.0) * 0.125).collect();
        let seed_values: Vec<f64> = (0..36).map(|i| ((i * 13 % 19) as f64 - 9.0) * 0.125).collect();
        let query_lengths = [3i32, 2];
        let key_value_lengths = [4i32, 2];
        let device = |values: &[f64], dimensions: &[usize]| {
            let values = values.iter().map(|&value| value as f32).collect::<Vec<_>>();
            let r#type = replicated_type(&mesh, DataType::F32, dimensions);
            Array::from_host_buffer(&client, r#type, mesh.clone(), values_to_bytes(&values).as_slice()).unwrap()
        };
        let device_lengths = |values: &[i32]| {
            let r#type = replicated_type(&mesh, DataType::I32, &[values.len()]);
            Array::from_host_buffer(&client, r#type, mesh.clone(), values_to_bytes(values).as_slice()).unwrap()
        };
        let reference = |values: &[f64], dimensions: &[usize]| {
            let shape = Shape::new(dimensions.iter().map(|&dimension| Size::Static(dimension)).collect());
            CpuArray::from_f64s(ArrayType::new(DataType::F32, shape), values.to_vec())
        };
        let reference_lengths = |values: &[i32]| {
            let shape = Shape::new(vec![Size::Static(values.len())]);
            CpuArray::from_f64s(
                ArrayType::new(DataType::I32, shape),
                values.iter().map(|&value| f64::from(value)).collect(),
            )
        };
        let scale = 0.5;
        let mask = AttentionMask::Causal;

        let device_query = device(&query_values, &query_dimensions);
        let device_key = device(&key_values, &key_value_dimensions);
        let device_value = device(&value_values, &key_value_dimensions);
        let device_bias = device(&bias_values, &bias_dimensions);
        let device_query_lengths = device_lengths(&query_lengths);
        let device_key_value_lengths = device_lengths(&key_value_lengths);
        let device_sequence_lengths = Some((&device_query_lengths, &device_key_value_lengths));
        let (device_output, device_activation) = device_query
            .dot_product_attention_with_activation(
                &device_key,
                &device_value,
                Some(&device_bias),
                device_sequence_lengths,
                scale,
                mask,
                None,
                None,
            )
            .unwrap();
        let (device_dq, device_dk, device_dv, device_dbias) = device_query
            .dot_product_attention_backward_with_options(
                &device_key,
                &device_value,
                Some(&device_bias),
                device_sequence_lengths,
                &device_output,
                &device_activation,
                &device(&seed_values, &query_dimensions),
                scale,
                mask,
                None,
                None,
            )
            .unwrap();
        let reference_query = reference(&query_values, &query_dimensions);
        let reference_key = reference(&key_values, &key_value_dimensions);
        let reference_value = reference(&value_values, &key_value_dimensions);
        let reference_bias = reference(&bias_values, &bias_dimensions);
        let reference_query_lengths = reference_lengths(&query_lengths);
        let reference_key_value_lengths = reference_lengths(&key_value_lengths);
        let reference_sequence_lengths = Some((&reference_query_lengths, &reference_key_value_lengths));
        let (reference_output, reference_activation) = reference_query
            .dot_product_attention_with_activation(
                &reference_key,
                &reference_value,
                Some(&reference_bias),
                reference_sequence_lengths,
                scale,
                mask,
                None,
                None,
            )
            .unwrap();
        let (reference_dq, reference_dk, reference_dv, reference_dbias) = reference_query
            .dot_product_attention_backward_with_options(
                &reference_key,
                &reference_value,
                Some(&reference_bias),
                reference_sequence_lengths,
                &reference_output,
                &reference_activation,
                &reference(&seed_values, &query_dimensions),
                scale,
                mask,
                None,
                None,
            )
            .unwrap();
        for (device_cotangent, reference_cotangent) in [
            (&device_dq, &reference_dq),
            (&device_dk, &reference_dk),
            (&device_dv, &reference_dv),
            (&device_dbias.unwrap(), &reference_dbias.unwrap()),
        ] {
            for (device_value, reference_value) in read_f32s(device_cotangent).iter().zip(reference_cotangent.to_f64s())
            {
                assert!(
                    (f64::from(*device_value) - reference_value).abs() < 1e-5,
                    "expected {reference_value} but got {device_value}",
                );
            }
        }
        // Out-of-range gradient regions are exact zeros on the device path: query-cotangent rows at or beyond
        // `query_lengths[b]` and key/value-cotangent positions at or beyond `key_value_lengths[b]`.
        let device_dq_values = read_f32s(&device_dq);
        let device_dk_values = read_f32s(&device_dk);
        let device_dv_values = read_f32s(&device_dv);
        for b in 0..2 {
            for i in (query_lengths[b] as usize)..3 {
                for n in 0..2 {
                    for d in 0..3 {
                        assert_eq!(device_dq_values[((b * 3 + i) * 2 + n) * 3 + d], 0.0);
                    }
                }
            }
            for s in (key_value_lengths[b] as usize)..4 {
                for d in 0..3 {
                    assert_eq!(device_dk_values[(b * 4 + s) * 3 + d], 0.0);
                    assert_eq!(device_dv_values[(b * 4 + s) * 3 + d], 0.0);
                }
            }
        }
    }

    /// Reverse-mode differentiation through the `custom_vjp` attention training entry point on the XLA CPU eager
    /// path matches the reference backend's gradients (the primal executes the composition while the gradient
    /// replays the staged backward operation).
    #[test]
    fn test_eager_differentiable_dot_product_attention_gradient() {
        use ryft_core::operations::attention::{AttentionMask, differentiable_dot_product_attention};

        use crate::XlaDomain;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);

        let query_dimensions = [1usize, 2, 2, 3];
        let key_value_dimensions = [1usize, 3, 1, 3];
        let query_values: Vec<f64> = (0..12).map(|i| ((i * 7 % 11) as f64 - 5.0) * 0.25).collect();
        let key_values: Vec<f64> = (0..9).map(|i| ((i * 5 % 13) as f64 - 6.0) * 0.25).collect();
        let value_values: Vec<f64> = (0..9).map(|i| ((i * 3 % 7) as f64 - 3.0) * 0.5).collect();
        let device = |values: &[f64], dimensions: &[usize]| {
            let values = values.iter().map(|&value| value as f32).collect::<Vec<_>>();
            let r#type = replicated_type(&mesh, DataType::F32, dimensions);
            Array::from_host_buffer(&client, r#type, mesh.clone(), values_to_bytes(&values).as_slice()).unwrap()
        };
        let reference = |values: &[f64], dimensions: &[usize]| {
            let shape = Shape::new(dimensions.iter().map(|&dimension| Size::Static(dimension)).collect());
            CpuArray::from_f64s(ArrayType::new(DataType::F32, shape), values.to_vec())
        };
        let scale = 0.5;
        let mask = AttentionMask::Causal;

        let query = device(&query_values, &query_dimensions);
        let key = device(&key_values, &key_value_dimensions);
        let value = device(&value_values, &key_value_dimensions);
        let domain = query.execution_domain();
        let function = differentiable_dot_product_attention::<XlaDomain<'_>>(scale, mask, None, None);
        let (loss, (query_gradient, key_gradient, value_gradient)) = domain
            .value_and_gradient(
                |(query, key, value)| {
                    function.call((query, key, value)).unwrap().reduce(&[0, 1, 2, 3], ReductionKind::Sum)
                },
                (query, key, value),
            )
            .unwrap();

        use ryft_core::backends::arrays::ArrayOperation;
        use ryft_core::contexts::EagerContext;
        let reference_function = differentiable_dot_product_attention::<EagerContext<CpuArray, ArrayOperation<CpuArray>>>(
            scale, mask, None, None,
        );
        let (reference_loss, (reference_query_gradient, reference_key_gradient, reference_value_gradient)) =
            EagerContext::<CpuArray, ArrayOperation<CpuArray>>::new()
                .value_and_gradient(
                    |(query, key, value)| {
                        reference_function.call((query, key, value)).unwrap().reduce(&[0, 1, 2, 3], ReductionKind::Sum)
                    },
                    (
                        reference(&query_values, &query_dimensions),
                        reference(&key_values, &key_value_dimensions),
                        reference(&value_values, &key_value_dimensions),
                    ),
                )
                .unwrap();
        assert!((f64::from(read_f32s(&loss)[0]) - reference_loss.to_f64s()[0]).abs() < 1e-5);
        for (device_gradient, reference_gradient) in [
            (&query_gradient, &reference_query_gradient),
            (&key_gradient, &reference_key_gradient),
            (&value_gradient, &reference_value_gradient),
        ] {
            for (device_value, reference_value) in read_f32s(device_gradient).iter().zip(reference_gradient.to_f64s()) {
                assert!(
                    (f64::from(*device_value) - reference_value).abs() < 1e-5,
                    "expected {reference_value} but got {device_value}",
                );
            }
        }
    }

    #[test]
    fn test_eager_operator_sugar() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let a = f32_vector(&client, &mesh, &[1.0, 2.0]);
        let b = f32_vector(&client, &mesh, &[3.0, 5.0]);

        assert_eq!(read_f32s(&(a.clone() + b.clone())), vec![4.0, 7.0]);
        assert_eq!(read_f32s(&(b.clone() - a.clone())), vec![2.0, 3.0]);
        assert_eq!(read_f32s(&(a.clone() * b.clone())), vec![3.0, 10.0]);
        assert_eq!(read_f32s(&(b.clone() / a.clone())), vec![3.0, 2.5]);
        assert_eq!(read_f32s(&(-a)), vec![-1.0, -2.0]);

        let truths = boolean_vector(&client, &mesh, &[true, true, false, false]);
        let mixed = boolean_vector(&client, &mesh, &[true, false, true, false]);
        assert_eq!(read_booleans(&(truths.clone() & mixed.clone())), vec![true, false, false, false]);
        assert_eq!(read_booleans(&(truths.clone() | mixed.clone())), vec![true, true, true, false]);
        assert_eq!(read_booleans(&(truths.clone() ^ mixed.clone())), vec![false, true, true, false]);
        assert_eq!(read_booleans(&(!mixed)), vec![false, true, false, true]);
    }

    #[test]
    fn test_eager_transpose_and_reshape_round_trip() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let matrix = f32_matrix(&client, &mesh, 2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let transposed = matrix.transpose([1, 0]).unwrap();
        assert_eq!(transposed.shape(), StaticShape::new(vec![3, 2]));
        assert_eq!(read_f32s(&transposed), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        let round_tripped = transposed.transpose([1, 0]).unwrap();
        assert_eq!(round_tripped.shape(), StaticShape::new(vec![2, 3]));
        assert_eq!(read_f32s(&round_tripped), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let reshaped = matrix.reshape(Shape::new(vec![Size::Static(3), Size::Static(2)])).unwrap();
        assert_eq!(reshaped.shape(), StaticShape::new(vec![3, 2]));
        assert_eq!(read_f32s(&reshaped), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let flattened = reshaped.reshape(Shape::new(vec![Size::Static(6)])).unwrap();
        assert_eq!(flattened.shape(), StaticShape::new(vec![6]));
        assert_eq!(read_f32s(&flattened), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_eager_compare_and_select() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let a = f32_vector(&client, &mesh, &[1.0, 5.0, 3.0, 8.0]);
        let b = f32_vector(&client, &mesh, &[4.0, 2.0, 3.0, 9.0]);

        let less_than = a.compare(&b, ComparisonDirection::LessThan).unwrap();
        assert_eq!(less_than.data_type(), DataType::Boolean);
        assert_eq!(read_booleans(&less_than), vec![true, false, false, true]);

        let selected = Array::select(&less_than, &a, &b).unwrap();
        assert_eq!(read_f32s(&selected), vec![1.0, 2.0, 3.0, 8.0]);
    }

    #[test]
    fn test_eager_manipulation_capabilities() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let vector = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0, 4.0]);

        // Slice and update-slice round trip.
        let sliced = vector.slice(&[1], &[3], &[1]).unwrap();
        assert_eq!(read_f32s(&sliced), vec![2.0, 3.0]);
        let updated = vector.update_slice(&sliced, &[0]).unwrap();
        assert_eq!(read_f32s(&updated), vec![2.0, 3.0, 3.0, 4.0]);

        // Concatenation joins operands end to end along the requested axis.
        let concatenated = Array::concatenate([&vector, &sliced], 0).unwrap();
        assert_eq!(read_f32s(&concatenated), vec![1.0, 2.0, 3.0, 4.0, 2.0, 3.0]);

        // Broadcasting maps the input axis onto the trailing output axis.
        let broadcast_type = replicated_type(&mesh, DataType::F32, &[2, 6]);
        let broadcast = concatenated.broadcast(broadcast_type, &[1]).unwrap();
        assert_eq!(broadcast.shape(), StaticShape::new(vec![2, 6]));
        assert_eq!(read_f32s(&broadcast), vec![1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 2.0, 3.0],);

        // Padding writes the padding value around and between the payload elements.
        let padding_value = f32_scalar(&client, &mesh, 0.5);
        let padded = vector.pad(&padding_value, &[1], &[0], &[0]).unwrap();
        assert_eq!(read_f32s(&padded), vec![0.5, 1.0, 2.0, 3.0, 4.0]);

        // Reduction sums along the requested axis, and the like-constants match the receiver's shape.
        let total = vector.reduce(&[0], ReductionKind::Sum);
        assert_eq!(read_f32s(&total), vec![10.0]);
        assert_eq!(read_f32s(&vector.zero_like()), vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(read_f32s(&vector.one_like()), vec![1.0, 1.0, 1.0, 1.0]);

        // Tag and stop-gradient behave as eager identities on the payload.
        assert_eq!(read_f32s(&vector.clone().tag("residual")), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(read_f32s(&vector.stop_gradient()), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_eager_boolean_concretization() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let small = f32_scalar(&client, &mesh, 2.0);
        let large = f32_scalar(&client, &mesh, 5.0);

        // Python-style control flow: branch on a device-computed comparison result.
        let predicate = small.compare(&large, ComparisonDirection::LessThan).unwrap();
        let branch =
            if predicate.concretize().unwrap() { small.add(&large).unwrap() } else { small.mul(&large).unwrap() };
        assert_eq!(read_f32s(&branch), vec![7.0]);
        assert!(!large.compare(&small, ComparisonDirection::LessThan).unwrap().concretize().unwrap());

        // Elementwise truthiness is an explicit comparison against zero: zero maps to false and nonzero maps to true.
        let input = f32_vector(&client, &mesh, &[0.0, 2.0, 0.0]);
        let boolean = input.compare(&input.zero_like(), ComparisonDirection::NotEqual).unwrap();
        assert_eq!(boolean.data_type(), DataType::Boolean);
        assert_eq!(read_booleans(&boolean), vec![false, true, false]);

        // Rank-one predicates cannot collapse to a single Boolean.
        let vector_predicate = boolean_vector(&client, &mesh, &[true, false]);
        assert!(matches!(vector_predicate.concretize(), Err(ProgramError::Concretization { .. })));
    }

    #[test]
    fn test_eager_while_predicate() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);

        let mixed = boolean_vector(&client, &mesh, &[false, true]);
        let none = boolean_vector(&client, &mesh, &[false, false]);
        assert!(mixed.any_true().unwrap());
        assert!(!none.any_true().unwrap());
        assert!(matches!(f32_vector(&client, &mesh, &[1.0]).any_true(), Err(ProgramError::Concretization { .. }),));

        // The prefix-shaped predicate broadcasts along its leading axes: item 0 keeps `on_false`'s first row and
        // item 1 takes `on_true`'s second row.
        let on_true = f32_matrix(&client, &mesh, 2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let on_false = f32_matrix(&client, &mesh, 2, 2, &[5.0, 6.0, 7.0, 8.0]);
        let masked = mixed.mask_select(&on_true, &on_false).unwrap();
        assert_eq!(read_f32s(&masked), vec![5.0, 6.0, 3.0, 4.0]);
    }

    #[test]
    fn test_eager_chained_operations_propagate_client_and_cache() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let a = f32_vector(&client, &mesh, &[1.0, 2.0]);
        let b = f32_vector(&client, &mesh, &[3.0, 4.0]);

        // The result of an eager operation carries the executing client, so it feeds directly into further eager
        // operations, and it shares the producing domain's compile cache.
        let sum = a.add(&b).unwrap();
        assert!(std::ptr::eq(sum.client().unwrap(), &client));
        assert_eq!(sum.execution_domain().cache_size(), 1);

        let product = sum.mul(&b).unwrap();
        assert_eq!(read_f32s(&product), vec![12.0, 24.0]);
        assert!(std::ptr::eq(product.client().unwrap(), &client));

        // All values derived from `a` share one dispatch cache: `add` and `mul` each compiled once, and repeating
        // `add` at the same input signature is a cache hit.
        assert_eq!(product.execution_domain().cache_size(), 2);
        assert_eq!(a.execution_domain().cache_size(), 2);
        let repeated = a.add(&b).unwrap();
        assert_eq!(read_f32s(&repeated), vec![4.0, 6.0]);
        assert_eq!(a.execution_domain().cache_size(), 2);
    }

    #[test]
    fn test_execution_domain_recovery() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);

        // Arrays with an attached client recover a client-backed domain.
        let array = f32_vector(&client, &mesh, &[1.0, 2.0]);
        let domain = array.execution_domain();
        assert!(std::ptr::eq(domain.client().unwrap(), &client));

        // Arrays constructed without a client recover a clientless domain whose eager binds error clearly, which
        // also surfaces through the fallible value capabilities.
        let r#type = replicated_type(&mesh, DataType::F32, &[2]);
        let clientless = Array::from_addressable_buffers(None, r#type, mesh, Vec::new()).unwrap();
        assert!(clientless.client().is_none());
        assert!(matches!(
            clientless.neg(),
            Err(ProgramError::InvalidArgument { message })
                if message == "xla domain cannot eagerly execute operation `neg` without a PJRT client",
        ));
    }

    /// Top-level forward mode over concrete arrays: `jvp` of `f(x) = x * x` at `x = [1, 2, 3]` with unit tangents
    /// evaluates both the primal (`x²`) and the tangent (`2x·t`) through the eager engine.
    #[test]
    fn test_eager_jvp() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let x = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0]);
        let tangents = f32_vector(&client, &mesh, &[1.0, 1.0, 1.0]);
        let domain = x.execution_domain();
        let (value, tangent): (Array<'_>, Array<'_>) = domain.jvp(|x| Mul::mul(&x, &x), x.clone(), tangents).unwrap();
        assert_eq!(read_f32s(&value), vec![1.0, 4.0, 9.0]);
        assert_eq!(read_f32s(&tangent), vec![2.0, 4.0, 6.0]);
    }

    /// The absolute-value JVP uses its explicit origin convention after lowering the comparison and selection that
    /// define it, rather than exposing the NaN produced by the undefined `0 / 0` formula.
    #[test]
    fn test_eager_abs_jvp_at_zero() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let primal = f32_scalar(&client, &mesh, 0.0);
        let tangent = f32_scalar(&client, &mesh, 3.0);

        let (value, tangent) = primal.execution_domain().jvp(|input| input.abs(), primal.clone(), tangent).unwrap();
        assert_eq!(read_f32s(&value), vec![0.0]);
        assert_eq!(read_f32s(&tangent), vec![3.0]);

        let primal = c64_scalar(&client, &mesh, num_complex::Complex::new(0.0, 0.0));
        let tangent = c64_scalar(&client, &mesh, num_complex::Complex::new(1.0, 2.0));
        let (value, tangent) = primal.execution_domain().jvp(|input| input.abs(), primal.clone(), tangent).unwrap();
        assert_eq!(read_f32s(&value), vec![0.0]);
        assert_eq!(read_f32s(&tangent), vec![0.0]);
    }

    /// Batching a complex absolute value preserves the mapped axis's physical sharding while changing the element
    /// type from `c64` to `f32` and computing each complex magnitude independently.
    #[test]
    fn test_eager_batch_abs_preserves_mapped_axis_sharding() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = cpu_mesh_with_axis_size(&client, 2);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new(DataType::C64, Shape::new(vec![Size::Static(4)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let values = [
            num_complex::Complex::new(3.0f32, 4.0),
            num_complex::Complex::new(5.0f32, 12.0),
            num_complex::Complex::new(8.0f32, 15.0),
            num_complex::Complex::new(7.0f32, 24.0),
        ];
        let input =
            Array::from_host_buffer(&client, input_type, mesh.clone(), values_to_bytes(&values).as_slice()).unwrap();

        let output: Array<'_> = batch(|input| input.abs(), input, BatchAxis::new(0), BatchAxis::new(0), None).unwrap();
        let expected_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)])).with_sharding(sharding).unwrap();
        assert_eq!(output.r#type().as_ref(), &expected_type);
        for (actual, expected) in read_f64_coordinates(&output).into_iter().zip([5.0, 13.0, 17.0, 25.0]) {
            assert!((actual - expected).abs() < 1e-5, "expected {expected} but got {actual}");
        }
    }

    /// Top-level reverse mode over concrete arrays: `value_and_gradient` of `f(x) = sum(x * x)` at `x = [1, 2, 3]`
    /// returns the scalar value `14` and the gradient `2x` with the pullback replayed through the eager engine.
    #[test]
    fn test_eager_value_and_grad() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let x = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0]);
        let domain = x.execution_domain();
        let (value, gradient) = domain
            .value_and_gradient(
                |x| {
                    let squared = Mul::mul(&x, &x).unwrap();
                    squared.reduce(&[0], ReductionKind::Sum)
                },
                x.clone(),
            )
            .unwrap();
        assert_eq!(read_f32s(&value), vec![14.0]);
        assert_eq!(read_f32s(&gradient), vec![2.0, 4.0, 6.0]);
    }

    /// The free `grad` entry point recovers the eager XLA domain from a caller-supplied context and computes the
    /// gradient of `sum(x * x)` over concrete arrays.
    #[test]
    fn test_eager_free_grad() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let x = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0]);
        let domain = x.execution_domain();
        let gradient = domain
            .gradient(
                |x| {
                    let squared = Mul::mul(&x, &x).unwrap();
                    squared.reduce(&[0], ReductionKind::Sum)
                },
                x.clone(),
            )
            .unwrap();
        assert_eq!(read_f32s(&gradient), vec![2.0, 4.0, 6.0]);
    }

    /// Top-level `vjp` over concrete arrays: the pullback of `f(x) = x * x` at `x = [1, 2, 3]` maps an output
    /// cotangent `w` to `2x · w`, replayed through the eager engine with the recovered linearization-point residuals
    /// appended after the cotangent.
    #[test]
    fn test_eager_vjp_pullback() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let x = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0]);
        let domain = x.execution_domain();
        let (value, pullback) = domain.vjp(|x| Ok(vec![Mul::mul(&x, &x)?]), x.clone()).unwrap();
        let (pullback, residuals) = pullback.into_parts();
        assert_eq!(read_f32s(&value[0]), vec![1.0, 4.0, 9.0]);

        // The direct-transpose pullback consumes `[output_cotangents ++ residuals]` and produces the flat input
        // cotangents.
        let cotangent = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0]);
        let mut pullback_inputs = vec![cotangent];
        pullback_inputs.extend(residuals);
        let input_cotangents = pullback.interpret_in_context(&domain, pullback_inputs).unwrap();
        assert_eq!(input_cotangents.len(), 1);
        assert_eq!(read_f32s(&input_cotangents[0]), vec![2.0, 8.0, 18.0]);
    }

    /// Complex elementary operations and their holomorphic or real-output gradients execute through the XLA eager
    /// domain using the same principal-value and cotangent conventions as the scalar reference backend.
    #[test]
    fn test_eager_complex_gradients() {
        use ryft_core::operations::complex::{Conjugate, Real};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let z = num_complex::Complex::new(0.7f32, -0.3f32);
        let x = c64_scalar(&client, &mesh, z);
        let domain = x.execution_domain();

        // Holomorphic gradient of z² through the XLA eager domain: the `one` cotangent seed lowers through the
        // composed complex constant, and the pullback recovers ∂(z²)/∂z = 2z on device.
        let (value, gradient) = domain.value_and_gradient_holomorphic(|x| x.clone() * x, x.clone()).unwrap();
        assert_c64_close(read_c64s(&value)[0], z * z);
        assert_c64_close(read_c64s(&gradient)[0], z + z);

        // Complex `atan2(y, x)` uses XLA's principal value and its two holomorphic partial derivatives away from
        // singularities and branch cuts.
        let y = num_complex::Complex::new(0.7f32, -0.2f32);
        let x_value = num_complex::Complex::new(-0.3f32, 0.4f32);
        let y_array = c64_scalar(&client, &mesh, y);
        let x_array = c64_scalar(&client, &mesh, x_value);
        let (value, (y_gradient, x_gradient)) =
            domain.value_and_gradient_holomorphic(|(y, x)| y.atan2(&x), (y_array, x_array)).unwrap();
        let denominator = x_value * x_value + y * y;
        let imaginary_unit = num_complex::Complex::new(0.0f32, 1.0f32);
        assert_c64_close(
            read_c64s(&value)[0],
            -imaginary_unit * ((x_value + imaginary_unit * y) / denominator.sqrt()).ln(),
        );
        assert_c64_close(read_c64s(&y_gradient)[0], x_value / denominator);
        assert_c64_close(read_c64s(&x_gradient)[0], -y / denominator);

        // ℂ → ℝ gradient of |z|² = Re(z · z̄) through the plain entry point, exercising the `conjugate` (lowered as
        // `complex(real, -imag)`), `real`, and `complex` StableHLO lowerings in the pullback: the gradient is 2·z̄.
        let gradient = domain.gradient(|x| (x.clone() * x.conjugate().unwrap()).real().unwrap(), x.clone()).unwrap();
        assert_c64_close(read_c64s(&gradient)[0], (z + z).conj());
    }

    #[test]
    fn test_eager_value_and_grad_with_aux() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let x = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0]);
        let domain = x.execution_domain();
        let ((value, aux), gradient): ((Array<'_>, Array<'_>), Array<'_>) = domain
            .value_and_gradient_with_aux(
                |x| {
                    let squared = Mul::mul(&x, &x).unwrap();
                    let aux = Add::add(&x, &x).unwrap();
                    (squared.reduce(&[0], ReductionKind::Sum), aux)
                },
                x.clone(),
            )
            .unwrap();
        assert_eq!(read_f32s(&value), vec![14.0]);
        assert_eq!(read_f32s(&aux), vec![2.0, 4.0, 6.0]);
        assert_eq!(read_f32s(&gradient), vec![2.0, 4.0, 6.0]);
    }

    /// Nested transform composition over concrete arrays: `grad` of a function that internally maps its per-item
    /// square through the free `batch`. The batched square stages into the gradient trace, so the composition
    /// differentiates `sum(x * x)` and evaluates through the eager engine.
    #[test]
    fn test_eager_grad_of_batched_function() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let x = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0]);
        let domain = x.execution_domain();
        let gradient = domain
            .gradient(
                |x| {
                    let squared =
                        batch(|item| Ok(item.clone() * item), x, BatchAxis::new(0), BatchAxis::new(0), None).unwrap();
                    squared.reduce(&[0], ReductionKind::Sum)
                },
                x.clone(),
            )
            .unwrap();
        assert_eq!(read_f32s(&gradient), vec![2.0, 4.0, 6.0]);
    }

    /// Dense forward-mode Jacobian over concrete arrays: `jacobian_forward` of `f(x) = x * x` at
    /// `x = [1, 2, 3]` materializes the full `3x3` Jacobian `diag(2, 4, 6)`. Basis synthesis, batched replay, and block
    /// assembly stay on device.
    #[test]
    fn test_eager_jacobian_forward() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let x = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0]);
        let domain = x.execution_domain();
        let jacobian = domain.jacobian_forward(|x| Mul::mul(&x, &x), x).unwrap();

        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.output_type().static_shape().unwrap().as_slice(), &[3]);
        assert_eq!(block.input_type().static_shape().unwrap().as_slice(), &[3]);
        assert!(block.value().client().is_some(), "the derivative block must remain attached to its device client");
        assert_eq!(block.value().shape(), StaticShape::new(vec![3, 3]));
        assert_eq!(read_f64_coordinates(block.value()).as_slice(), &[2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 6.0]);
    }

    /// Packed coordinate-basis synthesis over half-precision element types stays on device and produces one exact
    /// identity matrix per leaf. Host readback occurs only in this assertion helper.
    #[test]
    fn test_eager_coordinate_basis_stays_on_device_across_element_types() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        for data_type in [DataType::F16, DataType::BF16] {
            let r#type = replicated_type(&mesh, data_type, &[3]);
            let bytes = match data_type {
                DataType::F16 => {
                    [1.0, 2.0, 3.0].iter().flat_map(|value| f16::from_f64(*value).to_ne_bytes()).collect::<Vec<_>>()
                }
                _ => [1.0, 2.0, 3.0].iter().flat_map(|value| bf16::from_f64(*value).to_ne_bytes()).collect::<Vec<_>>(),
            };
            let array = Array::from_host_buffer(&client, r#type, mesh.clone(), bytes.as_slice()).unwrap();
            assert_eq!(read_f64_coordinates(&array), vec![1.0, 2.0, 3.0]);

            let basis = array
                .execution_domain()
                .bind(CoordinateBasisOperation::new(array.r#type().into_owned(), 0, 3), Vec::new(), &[])
                .unwrap()
                .remove(0);
            assert_eq!(basis.data_type(), data_type);
            assert_eq!(basis.shape(), StaticShape::new(vec![3, 3]));
            assert_eq!(read_f64_coordinates(&basis), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        }

        // Complex leaves use the same integer index graph and select typed complex one/zero values.
        let complex_type = replicated_type(&mesh, DataType::C64, &[2]);
        let complex_values = [num_complex::Complex::new(2.0f32, 1.0), num_complex::Complex::new(-1.0, 3.0)];
        let complex =
            Array::from_host_buffer(&client, complex_type, mesh.clone(), values_to_bytes(&complex_values)).unwrap();
        let basis = complex
            .execution_domain()
            .bind(CoordinateBasisOperation::new(complex.r#type().into_owned(), 0, 2), Vec::new(), &[])
            .unwrap()
            .remove(0);
        assert_eq!(
            read_c64s(&basis),
            vec![
                num_complex::Complex::new(1.0, 0.0),
                num_complex::Complex::new(0.0, 0.0),
                num_complex::Complex::new(0.0, 0.0),
                num_complex::Complex::new(1.0, 0.0),
            ],
        );
    }

    /// Dense forward-mode Jacobian over a half-precision primal: `jacobian_forward` of `f(x) = x * x` at
    /// `x = [1, 2, 3]` with `f16` elements materializes the exact `3x3` Jacobian `diag(2, 4, 6)` end to end through the
    /// CPU PJRT backend (small integers and their doubles are exactly representable in `f16`).
    #[test]
    fn test_eager_jacobian_forward_over_f16_input() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let r#type = replicated_type(&mesh, DataType::F16, &[3]);
        let bytes = [1.0, 2.0, 3.0].iter().flat_map(|value| f16::from_f64(*value).to_ne_bytes()).collect::<Vec<_>>();
        let x = Array::from_host_buffer(&client, r#type, mesh.clone(), bytes.as_slice()).unwrap();
        let domain = x.execution_domain();
        let jacobian = domain.jacobian_forward(|x| Mul::mul(&x, &x), x).unwrap();

        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.output_type().static_shape().unwrap().as_slice(), &[3]);
        assert_eq!(block.input_type().static_shape().unwrap().as_slice(), &[3]);
        assert_eq!(read_f64_coordinates(block.value()).as_slice(), &[2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 6.0]);
    }

    #[test]
    fn test_eager_holomorphic_dense_differentiation() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let value = num_complex::Complex::new(1.0f32, 2.0);
        let input = c64_scalar(&client, &mesh, value);
        let context = input.execution_domain();

        let forward = context.jacobian_forward_holomorphic(|input| Mul::mul(&input, &input), input.clone()).unwrap();
        let reverse = context.jacobian_reverse_holomorphic(|input| Mul::mul(&input, &input), input.clone()).unwrap();
        let hessian = context.hessian_holomorphic(|input| Mul::mul(&input, &input), input).unwrap();

        assert_c64_close(read_c64s(forward.iter_blocks().next().unwrap().value())[0], 2.0 * value);
        assert_c64_close(read_c64s(reverse.iter_blocks().next().unwrap().value())[0], 2.0 * value);
        assert_c64_close(
            read_c64s(hessian.iter_blocks().next().unwrap().value())[0],
            num_complex::Complex::new(2.0, 0.0),
        );
    }

    /// Dense forward-mode Jacobian over a *sharded* primal: `jacobian_forward` of `f(x) = x * x` at `x = [1, 2, 3, 4]`
    /// sharded over a 2-device CPU mesh materializes the full `4x4` Jacobian `diag(2, 4, 6, 8)`. The batched basis
    /// tangents must carry the primal's sharding (with a replicated inserted batch axis) so the tangent replay
    /// type-checks against the tangent program's sharded declared input types.
    #[test]
    fn test_eager_jacobian_forward_over_sharded_input() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = cpu_mesh_with_axis_size(&client, 2);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)])).with_sharding(sharding).unwrap();
        let bytes = values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]);
        let x = Array::from_host_buffer(&client, r#type, mesh.clone(), bytes.as_slice()).unwrap();
        let domain = x.execution_domain();
        let jacobian = domain.jacobian_forward(|x| Mul::mul(&x, &x), x).unwrap();

        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.output_type().static_shape().unwrap().as_slice(), &[4]);
        assert_eq!(block.input_type().static_shape().unwrap().as_slice(), &[4]);
        assert_eq!(
            read_f64_coordinates(block.value()).as_slice(),
            &[2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 8.0],
        );
    }

    /// Dense reverse-mode Jacobian over a *sharded* primal: `jacobian_reverse` of `f(x) = x * x` at `x = [1, 2, 3, 4]`
    /// sharded over a 2-device CPU mesh matches the `jacobian_forward` matrix exactly, with the one-hot cotangent basis
    /// replayed through the pullback on device.
    #[test]
    fn test_eager_jacobian_reverse_over_sharded_input() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = cpu_mesh_with_axis_size(&client, 2);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)])).with_sharding(sharding).unwrap();
        let bytes = values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]);
        let x = Array::from_host_buffer(&client, r#type, mesh.clone(), bytes.as_slice()).unwrap();
        let jacobian = jacobian_reverse(|x| Mul::mul(&x, &x), x).unwrap();

        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.output_type().static_shape().unwrap().as_slice(), &[4]);
        assert_eq!(block.input_type().static_shape().unwrap().as_slice(), &[4]);
        assert_eq!(
            read_f64_coordinates(block.value()).as_slice(),
            &[2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 8.0],
        );
    }

    /// Dense reverse-mode Jacobian over concrete arrays: `jacobian_reverse` of `f(x) = x * x` at
    /// `x = [1, 2, 3]` replays the one-hot basis cotangents through the pullback on device and matches the
    /// `jacobian_forward` matrix exactly.
    #[test]
    fn test_eager_jacobian_reverse() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let x = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0]);
        let jacobian = jacobian_reverse(|x| Mul::mul(&x, &x), x).unwrap();

        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.output_type().static_shape().unwrap().as_slice(), &[3]);
        assert_eq!(block.input_type().static_shape().unwrap().as_slice(), &[3]);
        assert_eq!(read_f64_coordinates(block.value()).as_slice(), &[2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 6.0]);
    }

    /// Dense Hessian over concrete arrays. The differentiated function is the scalar-output sum of squares
    /// `f(x) = sum(x * x)` — the simplest function whose Hessian is a full (constant) matrix, `2·I` — evaluated at
    /// `x = [1, 2, 3]` as forward-over-reverse through the eager engine.
    #[test]
    fn test_eager_hessian() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let x = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0]);
        let domain = x.execution_domain();
        let hessian = domain
            .hessian(
                |x| {
                    let squared = Mul::mul(&x, &x).unwrap();
                    Ok(squared.reduce(&[0], ReductionKind::Sum))
                },
                x,
            )
            .unwrap();

        let block = hessian.iter_blocks().next().unwrap();
        assert!(block.output_type().static_shape().unwrap().as_slice().is_empty());
        assert_eq!(block.first_input_type().static_shape().unwrap().as_slice(), &[3]);
        assert_eq!(block.second_input_type().static_shape().unwrap().as_slice(), &[3]);
        assert_eq!(read_f64_coordinates(block.value()).as_slice(), &[2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]);
    }

    /// The callable `vjp` surface over concrete arrays: `Pullback::apply` replays the pullback program through the
    /// domain it was built in, reproducing the raw `vjp` numbers of [`test_eager_vjp_pullback`] (`2x · w` for
    /// `f(x) = x * x` at `x = [1, 2, 3]` with cotangent `w = [1, 2, 3]`).
    #[test]
    fn test_eager_vjp_pullback_apply() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let x = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0]);
        let domain = x.execution_domain();
        let (value, pullback) = domain.vjp(|x| Ok(vec![Mul::mul(&x, &x)?]), x).unwrap();
        assert_eq!(read_f32s(&value[0]), vec![1.0, 4.0, 9.0]);

        let cotangent = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0]);
        let input_cotangent = pullback.apply(vec![cotangent]).unwrap();
        assert_eq!(read_f32s(&input_cotangent), vec![2.0, 8.0, 18.0]);
    }

    #[test]
    fn test_free_batch_squares_vector_items() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let input = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0]);

        // The free `batch` recovers the eager XLA domain from the concrete input array and squares each item of the
        // length-3 batch through eager per-operation dispatch.
        let output: Array<'_> =
            batch(|x| Ok(x.clone() * x), input, BatchAxis::new(0), BatchAxis::new(0), None).unwrap();
        assert_eq!(read_f32s(&output), vec![1.0, 4.0, 9.0]);
    }
}
