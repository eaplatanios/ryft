use ryft_core::contexts::Context;
use ryft_core::macros::check_count;
use ryft_core::operations::BooleanLike;
use ryft_core::operations::compare::{Compare, ComparisonDirection};
use ryft_core::operations::constants::ZeroLike;
use ryft_core::operations::control_flow::{Select, WhilePredicate};
use ryft_core::operations::logical::{AndOperation, NotOperation, OrOperation, XorOperation};
use ryft_core::operations::manipulation::Broadcast;
use ryft_core::operations::math::{Add, Div, Mul, Neg, Sub};
use ryft_core::programs::types::Typed;
use ryft_core::programs::{ProgramError, Value};
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
/// sugar (per-type implementations required by the orphan rule) and the host-readback predicates [`BooleanLike`]
/// and [`WhilePredicate`]. This helper is their shared bind-and-unwrap step; callers must pass at least one input,
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

impl BooleanLike for Array<'_> {
    /// Returns the Boolean counterpart of this [`Array`]: already-Boolean arrays are returned as-is, while other
    /// element types are reinterpreted on device by comparing against a zero-filled counterpart (zero maps to
    /// `false` and any nonzero element maps to `true`). The conversion executes eagerly and panics if the underlying
    /// device execution fails, matching this method's infallible signature.
    fn as_boolean(&self) -> Self {
        if self.data_type() == DataType::Boolean {
            return self.clone();
        }
        let zero = self.zero_like();
        self.compare(&zero, ComparisonDirection::NotEqual).expect("`as_boolean` conversion failed")
    }

    /// Extracts a concrete scalar Rust Boolean by copying one addressable shard of a rank-0 Boolean-typed [`Array`]
    /// to the host. Higher-rank or non-Boolean arrays error because they cannot collapse to a single Boolean, and
    /// arrays with no addressable shards error because the current process cannot read their payload.
    fn boolean(&self) -> Result<bool, ProgramError> {
        if self.r#type().rank() == 0 && self.data_type() == DataType::Boolean {
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
/// [`TestArray`](ryft_core::tests::TestArray): [`WhilePredicate::any_true`] reduces the whole Boolean payload with
/// `or` via device-to-host readback of every shard, and [`WhilePredicate::mask_select`] broadcasts the predicate
/// against the operands along its leading (prefix) axes on device before selecting.
impl WhilePredicate for Array<'_> {
    fn any_true(&self) -> Result<bool, ProgramError> {
        if self.data_type() != DataType::Boolean {
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
        if self.data_type() != DataType::Boolean || on_true.r#type() != on_false.r#type() {
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
            let output_type = on_true.r#type().as_boolean();
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
    use ryft_core::batching::{BatchAxis, batch};
    use ryft_core::operations::constants::OneLike;
    use ryft_core::operations::control_flow::SelectCondition;
    use ryft_core::operations::differentiation::{CoordinateBasisOperation, StopGradient};
    use ryft_core::operations::manipulation::{Concatenate, Pad, Reshape, Slice, Transpose, UpdateSlice};
    use ryft_core::operations::math::{Abs, Atan2, Cos, Sin};
    use ryft_core::operations::tag::Tag;
    use ryft_core::sharding::{Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, ShardingDimension};
    use ryft_core::tracing_v2::operations::reduce::{Reduce, ReductionKind};
    use ryft_core::tracing_v2::{DenseDifferentiate, ForwardModeDifferentiate, ReverseModeDifferentiate, jacrev};
    use ryft_core::types::{ArrayType, Shape, Size, StaticShape};
    use ryft_pjrt::{Client, ClientOptions, CpuClientOptions, load_cpu_plugin};

    use crate::tests::{values_from_bytes, values_to_bytes};
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

        let condition = less_than.select_condition().unwrap();
        let selected = Array::select(&condition, &a, &b).unwrap();
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
        let concatenated = Array::concatenate(&[vector.clone(), sliced], 0).unwrap();
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
    fn test_eager_boolean_like_branching_on_device_predicate() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let small = f32_scalar(&client, &mesh, 2.0);
        let large = f32_scalar(&client, &mesh, 5.0);

        // Python-style control flow: branch on a device-computed comparison result.
        let predicate = small.compare(&large, ComparisonDirection::LessThan).unwrap();
        let branch = if predicate.boolean().unwrap() { small.add(&large).unwrap() } else { small.mul(&large).unwrap() };
        assert_eq!(read_f32s(&branch), vec![7.0]);
        assert!(!large.compare(&small, ComparisonDirection::LessThan).unwrap().boolean().unwrap());

        // Non-Boolean payloads reinterpret on device: zero maps to false and nonzero maps to true.
        let as_boolean = f32_vector(&client, &mesh, &[0.0, 2.0, 0.0]).as_boolean();
        assert_eq!(as_boolean.data_type(), DataType::Boolean);
        assert_eq!(read_booleans(&as_boolean), vec![false, true, false]);

        // Rank-one predicates cannot collapse to a single Boolean.
        let vector_predicate = boolean_vector(&client, &mesh, &[true, false]);
        assert!(matches!(vector_predicate.boolean(), Err(ProgramError::Concretization { .. })));
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

    /// Dense forward-mode Jacobian over concrete arrays: `jacfwd` of `f(x) = x * x` at `x = [1, 2, 3]` materializes
    /// the full `3x3` Jacobian `diag(2, 4, 6)`. Basis synthesis, batched replay, and block assembly stay on device.
    #[test]
    fn test_eager_jacfwd() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let x = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0]);
        let domain = x.execution_domain();
        let jacobian = domain.jacfwd(|x| Mul::mul(&x, &x), x).unwrap();

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

    /// Dense forward-mode Jacobian over a half-precision primal: `jacfwd` of `f(x) = x * x` at `x = [1, 2, 3]`
    /// with `f16` elements materializes the exact `3x3` Jacobian `diag(2, 4, 6)` end to end through the CPU PJRT
    /// backend (small integers and their doubles are exactly representable in `f16`).
    #[test]
    fn test_eager_jacfwd_over_f16_input() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let r#type = replicated_type(&mesh, DataType::F16, &[3]);
        let bytes = [1.0, 2.0, 3.0].iter().flat_map(|value| f16::from_f64(*value).to_ne_bytes()).collect::<Vec<_>>();
        let x = Array::from_host_buffer(&client, r#type, mesh.clone(), bytes.as_slice()).unwrap();
        let domain = x.execution_domain();
        let jacobian = domain.jacfwd(|x| Mul::mul(&x, &x), x).unwrap();

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

        let forward = context.jacfwd_holomorphic(|input| Mul::mul(&input, &input), input.clone()).unwrap();
        let reverse = context.jacrev_holomorphic(|input| Mul::mul(&input, &input), input.clone()).unwrap();
        let hessian = context.hessian_holomorphic(|input| Mul::mul(&input, &input), input).unwrap();

        assert_c64_close(read_c64s(forward.iter_blocks().next().unwrap().value())[0], 2.0 * value);
        assert_c64_close(read_c64s(reverse.iter_blocks().next().unwrap().value())[0], 2.0 * value);
        assert_c64_close(
            read_c64s(hessian.iter_blocks().next().unwrap().value())[0],
            num_complex::Complex::new(2.0, 0.0),
        );
    }

    /// Dense forward-mode Jacobian over a *sharded* primal: `jacfwd` of `f(x) = x * x` at `x = [1, 2, 3, 4]`
    /// sharded over a 2-device CPU mesh materializes the full `4x4` Jacobian `diag(2, 4, 6, 8)`. The batched basis
    /// tangents must carry the primal's sharding (with a replicated inserted batch axis) so the tangent replay
    /// type-checks against the tangent program's sharded declared input types.
    #[test]
    fn test_eager_jacfwd_over_sharded_input() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = cpu_mesh_with_axis_size(&client, 2);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)])).with_sharding(sharding).unwrap();
        let bytes = values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]);
        let x = Array::from_host_buffer(&client, r#type, mesh.clone(), bytes.as_slice()).unwrap();
        let domain = x.execution_domain();
        let jacobian = domain.jacfwd(|x| Mul::mul(&x, &x), x).unwrap();

        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.output_type().static_shape().unwrap().as_slice(), &[4]);
        assert_eq!(block.input_type().static_shape().unwrap().as_slice(), &[4]);
        assert_eq!(
            read_f64_coordinates(block.value()).as_slice(),
            &[2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 8.0],
        );
    }

    /// Dense reverse-mode Jacobian over a *sharded* primal: `jacrev` of `f(x) = x * x` at `x = [1, 2, 3, 4]`
    /// sharded over a 2-device CPU mesh matches the `jacfwd` matrix exactly, with the one-hot cotangent basis
    /// replayed through the pullback on device.
    #[test]
    fn test_eager_jacrev_over_sharded_input() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = cpu_mesh_with_axis_size(&client, 2);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)])).with_sharding(sharding).unwrap();
        let bytes = values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]);
        let x = Array::from_host_buffer(&client, r#type, mesh.clone(), bytes.as_slice()).unwrap();
        let jacobian = jacrev(|x| Mul::mul(&x, &x), x).unwrap();

        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.output_type().static_shape().unwrap().as_slice(), &[4]);
        assert_eq!(block.input_type().static_shape().unwrap().as_slice(), &[4]);
        assert_eq!(
            read_f64_coordinates(block.value()).as_slice(),
            &[2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 8.0],
        );
    }

    /// Dense reverse-mode Jacobian over concrete arrays: `jacrev` of `f(x) = x * x` at `x = [1, 2, 3]` replays the
    /// one-hot basis cotangents through the pullback on device and matches the `jacfwd` matrix exactly.
    #[test]
    fn test_eager_jacrev() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let x = f32_vector(&client, &mesh, &[1.0, 2.0, 3.0]);
        let jacobian = jacrev(|x| Mul::mul(&x, &x), x).unwrap();

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
