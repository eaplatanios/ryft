use ryft_core::macros::check_count;
use ryft_core::{
    Add, AndOperation, ArrayIrType, ArrayOperation, Broadcast, Concretizable, Context, DataType, DimensionFromScalar,
    DimensionFromScalarOperation, DimensionSize, DimensionSizeOperation, DimensionType, DimensionValue,
    DimensionVariable, Div, ElementType, Mul, Neg, NotOperation, Operation, OrOperation, ProgramError, Select, Sub,
    Typed, Value, WhilePredicate, XorOperation,
};

use crate::experimental::ops::XlaArrayConstant;
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
fn bind_single_output<'o, P: Into<ArrayOperation<XlaArrayConstant>>>(
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
                    self.r#type().as_ref(),
                ),
            })?;
            let bytes = shard_host_bytes(shard)?;
            return Ok(bytes.iter().any(|byte| *byte != 0));
        }
        Err(ProgramError::Concretization {
            message: format!(
                "cannot extract a concrete boolean from a value of type {}; expected bool[]",
                self.r#type().as_ref(),
            ),
        })
    }
}

impl DimensionSize<usize> for Array<'_> {
    /// Returns the concrete global extent of `axis` from this array's complete shard metadata.
    ///
    /// The descriptor list includes remote shards, so taking the greatest exclusive slice end recovers the global
    /// extent without synchronizing a device buffer. Constructing a [`DimensionValue`] validates that extent against
    /// the selected declared axis's identity and bounds before the host integer is returned.
    fn dimension_size<AxisValue: Into<ryft_core::Axis>>(&self, axis: AxisValue) -> Result<usize, ProgramError> {
        let operation = DimensionSizeOperation::new(self.r#type().as_ref(), axis)?;
        let extent = self.shards().iter().map(|shard| shard.slice()[operation.axis()].end).max().ok_or_else(|| {
            ProgramError::Concretization {
                message: format!("cannot read the extent of an array of type {} with no shard metadata", self.r#type()),
            }
        })?;
        DimensionValue::new(operation.result_type().clone(), extent)?;
        Ok(extent)
    }
}

impl DimensionFromScalar<DimensionValue> for Array<'_> {
    /// Copies this rank-zero integer array to the host and grants its checked value first-class dimension authority.
    fn to_dimension(&self, result: DimensionVariable) -> Result<DimensionValue, ProgramError> {
        let operation = DimensionFromScalarOperation::new(result);
        let mut output_types = operation.infer_output_types(&[ArrayIrType::Array(self.r#type().into_owned())], &[])?;
        let output_type = <&DimensionType>::try_from(&output_types.remove(0))?.clone();
        let shard = self.addressable_shards().next().ok_or_else(|| ProgramError::Concretization {
            message: format!(
                "cannot convert an array of type {} with no addressable shards to a dimension",
                self.r#type().as_ref(),
            ),
        })?;
        let bytes = shard_host_bytes(shard)?;
        let invalid_extent = |value: String| ProgramError::InvalidArgument {
            message: format!(
                "`{}` scalar input must be a nonnegative host-representable extent but is {value}",
                operation.name(),
            ),
        };
        let extent = match self.data_type() {
            DataType::I8 => {
                let value = i8::from_ne_bytes(bytes.as_slice().try_into().unwrap());
                usize::try_from(value).map_err(|_| invalid_extent(value.to_string()))?
            }
            DataType::I16 => {
                let value = i16::from_ne_bytes(bytes.as_slice().try_into().unwrap());
                usize::try_from(value).map_err(|_| invalid_extent(value.to_string()))?
            }
            DataType::I32 => {
                let value = i32::from_ne_bytes(bytes.as_slice().try_into().unwrap());
                usize::try_from(value).map_err(|_| invalid_extent(value.to_string()))?
            }
            DataType::I64 => {
                let value = i64::from_ne_bytes(bytes.as_slice().try_into().unwrap());
                usize::try_from(value).map_err(|_| invalid_extent(value.to_string()))?
            }
            DataType::U8 => usize::from(u8::from_ne_bytes(bytes.as_slice().try_into().unwrap())),
            DataType::U16 => usize::from(u16::from_ne_bytes(bytes.as_slice().try_into().unwrap())),
            DataType::U32 => {
                let value = u32::from_ne_bytes(bytes.as_slice().try_into().unwrap());
                usize::try_from(value).map_err(|_| invalid_extent(value.to_string()))?
            }
            DataType::U64 => {
                let value = u64::from_ne_bytes(bytes.as_slice().try_into().unwrap());
                usize::try_from(value).map_err(|_| invalid_extent(value.to_string()))?
            }
            _ => unreachable!("dimension_from_scalar input type is validated before reading its payload"),
        };
        Ok(DimensionValue::new(output_type, extent)?)
    }
}

/// Batched while-predicate semantics for [`Array`], mirroring the reference semantics of
/// [`Array`](ryft_core::arrays::Array): [`WhilePredicate::any_true`] reduces the whole Boolean payload with
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
                    self.r#type().as_ref(),
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
                    self.r#type().as_ref(),
                    on_true.r#type().as_ref(),
                    on_false.r#type().as_ref(),
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
        bind_single_output(AndOperation::new(), &[self, rhs]).expect("`and` operation failed")
    }
}

impl std::ops::BitOr for Array<'_> {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        bind_single_output(OrOperation::new(), &[self, rhs]).expect("`or` operation failed")
    }
}

impl std::ops::BitXor for Array<'_> {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self {
        bind_single_output(XorOperation::new(), &[self, rhs]).expect("`xor` operation failed")
    }
}

impl std::ops::Not for Array<'_> {
    type Output = Self;

    fn not(self) -> Self {
        bind_single_output(NotOperation::new(), &[self]).expect("`not` operation failed")
    }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use pretty_assertions::assert_eq;

    use ryft_core::{
        Abs, Array as CpuArray, ArrayType, Atan2, BatchAxis, Ceil, Compare, ComparisonDirection, Concatenate,
        ConvertElementType, CoordinateBasisOperation, Cos, Device, DeviceMesh, Differentiate, Dimension,
        DimensionBounds, Dot, Erf, Exp, Floor, ForwardModeDifferentiate, Log, LogicalMesh, Logistic, Max, MeshAxis,
        MeshAxisType, Min, OneLike, Pad, Pow, ProjectedContext, Reduce, ReductionKind, Rem, Reshape,
        ReverseModeDifferentiate, Round, Rsqrt, Scatter, ScatterDimensionNumbers, ScatterOperation,
        ScatterReductionKind, Shape, Sharding, ShardingDimension, Sign, Sin, Slice, Sqrt, StaticShape, StopGradient,
        Tag, Tanh, Transpose, TypeError, UpdateSlice, ZeroLike, batch, differentiate_at,
    };
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
        let shape = Shape::new(dimensions.iter().map(|&dimension| Dimension::Static(dimension)).collect());
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
        assert_parity(&left.max(&right).unwrap(), &reference_left.max(&reference_right).unwrap());
        assert_parity(&left.min(&right).unwrap(), &reference_left.min(&reference_right).unwrap());
        assert_parity(&left.rem(&right).unwrap(), &reference_left.rem(&reference_right).unwrap());

        // Element-type conversion agrees, including the exact `f8e4m3fn` encodings: the device payload bytes match
        // the reference backend's encoded bits bit for bit.
        let converted = left.convert_element_type(DataType::F8E4M3FN).unwrap();
        let converted_bytes = shard_host_bytes(&converted.addressable_shards().next().unwrap()).unwrap();
        let reference_bits = reference_left.convert_element_type(DataType::F8E4M3FN).unwrap().logical_bytes();
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
        let reference_integer = CpuArray::from_elements(
            ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(integer_values.len())])),
            &integer_values,
        )
        .unwrap();
        for kind in [ReductionKind::Sum, ReductionKind::Mean, ReductionKind::Max, ReductionKind::Min] {
            let device_values = read_i32s(&integer.reduce(&[0], kind));
            let reference_values = reference_integer.reduce(&[0], kind).elements::<i32>().unwrap();
            assert_eq!(device_values, reference_values, "integer `{kind}` reduction disagrees");
        }

        // Complex multiplication agrees.
        let complex_left_value = num_complex::Complex::new(1.5f32, -2.0);
        let complex_right_value = num_complex::Complex::new(0.5f32, 3.0);
        let complex_left = c64_scalar(&client, &mesh, complex_left_value);
        let complex_right = c64_scalar(&client, &mesh, complex_right_value);
        let device_product = read_c64s(&complex_left.mul(&complex_right).unwrap())[0];
        let reference_product =
            CpuArray::scalar(complex_left_value).mul(&CpuArray::scalar(complex_right_value)).unwrap();
        let reference_product = reference_product.elements::<num_complex::Complex<f32>>().unwrap()[0];
        assert!((device_product - reference_product).norm() < 1e-5);
    }

    #[test]
    fn test_eager_extrema_match_jax_semantics() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);

        // Boolean extrema use false/true identities, including when the reduced axis is empty.
        let empty_booleans =
            Array::from_host_buffer(&client, replicated_type(&mesh, DataType::Boolean, &[0]), mesh.clone(), &[])
                .unwrap();
        assert_eq!(read_booleans(&empty_booleans.reduce(&[0], ReductionKind::Max)), vec![false]);
        assert_eq!(read_booleans(&empty_booleans.reduce(&[0], ReductionKind::Min)), vec![true]);

        // Floating-point extrema propagate NaNs and order negative zero below positive zero.
        let nan_values = [1.0f32, f32::NAN];
        let nan = Array::from_host_buffer(
            &client,
            replicated_type(&mesh, DataType::F32, &[nan_values.len()]),
            mesh.clone(),
            values_to_bytes(&nan_values).as_slice(),
        )
        .unwrap();
        assert!(read_f32s(&nan.reduce(&[0], ReductionKind::Max))[0].is_nan());
        let zero_values = [-0.0f32, 0.0];
        let zeros = Array::from_host_buffer(
            &client,
            replicated_type(&mesh, DataType::F32, &[zero_values.len()]),
            mesh.clone(),
            values_to_bytes(&zero_values).as_slice(),
        )
        .unwrap();
        assert_eq!(read_f32s(&zeros.reduce(&[0], ReductionKind::Max))[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(read_f32s(&zeros.reduce(&[0], ReductionKind::Min))[0].to_bits(), (-0.0f32).to_bits());

        // Complex extrema compare `(real, imaginary)` lexicographically in reductions and scatter combiners.
        let complex_values = [
            num_complex::Complex::new(1.0f32, 5.0),
            num_complex::Complex::new(2.0, -3.0),
            num_complex::Complex::new(2.0, 4.0),
        ];
        let complex = Array::from_host_buffer(
            &client,
            replicated_type(&mesh, DataType::C64, &[complex_values.len()]),
            mesh.clone(),
            values_to_bytes(&complex_values).as_slice(),
        )
        .unwrap();
        assert_eq!(read_c64s(&complex.reduce(&[0], ReductionKind::Max)), vec![num_complex::Complex::new(2.0, 4.0)]);
        assert_eq!(read_c64s(&complex.reduce(&[0], ReductionKind::Min)), vec![num_complex::Complex::new(1.0, 5.0)]);
        let empty_complex =
            Array::from_host_buffer(&client, replicated_type(&mesh, DataType::C64, &[0]), mesh.clone(), &[]).unwrap();
        assert_eq!(
            read_c64s(&empty_complex.reduce(&[0], ReductionKind::Max)),
            vec![num_complex::Complex::new(f32::NEG_INFINITY, 0.0)],
        );

        let indices = Array::from_host_buffer(
            &client,
            replicated_type(&mesh, DataType::I32, &[1, 1]),
            mesh.clone(),
            values_to_bytes(&[0i32]).as_slice(),
        )
        .unwrap();
        let updates = Array::from_host_buffer(
            &client,
            replicated_type(&mesh, DataType::C64, &[1]),
            mesh.clone(),
            values_to_bytes(&[num_complex::Complex::new(1.0f32, 9.0)]).as_slice(),
        )
        .unwrap();
        let operation =
            ScatterOperation::new(ScatterDimensionNumbers::new(vec![], vec![0], vec![0]), ScatterReductionKind::Max);
        assert_eq!(
            read_c64s(&complex.scatter(&indices, &updates, &operation).unwrap()),
            vec![
                num_complex::Complex::new(1.0, 9.0),
                num_complex::Complex::new(2.0, -3.0),
                num_complex::Complex::new(2.0, 4.0),
            ],
        );
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
            let tolerance = 1e-6f64.max(1e-6 * f64::abs(*reference_value));
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
        use ryft_core::TiledLayout;
        use ryft_core::operations::custom_call::{CustomCall, CustomCallOperation};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        ensure_add_one_handler_registered(&client).unwrap();
        let mesh = cpu_mesh(&client);

        // The handler receives both matrix buffers in non-default column-major order. Its elementwise computation is
        // layout-independent because the input and output use the same physical ordering.
        let matrix_type = replicated_type(&mesh, DataType::F32, &[2, 2])
            .with_layout(Some(TiledLayout::new(vec![0, 1], Vec::new()).into()));
        let matrix_input = Array::from_host_buffer(
            &client,
            matrix_type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[1.5, 2.5, 3.5, 4.5]),
        )
        .unwrap();
        let operation = CustomCallOperation::new(ADD_ONE_CUSTOM_CALL_TARGET, vec![matrix_type]);
        let outputs = CustomCall::custom_call(&operation, std::slice::from_ref(&matrix_input)).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(read_f32s(&outputs[0]), vec![2.5, 3.5, 4.5, 5.5]);

        // A typed `f64` attribute reaches the handler through the `backend_config` dictionary.
        let operation = operation.with_attribute("increment", 2.5);
        let outputs = CustomCall::custom_call(&operation, std::slice::from_ref(&matrix_input)).unwrap();
        assert_eq!(read_f32s(&outputs[0]), vec![4.0, 5.0, 6.0, 7.0]);

        // An aliased side-effecting call executes with the same public array-only FFI contract while its lowering
        // uses the alias metadata and a hidden ordered-I/O token.
        let vector_input = f32_vector(&client, &mesh, &[1.5, 2.5]);
        let operation = CustomCallOperation::new(ADD_ONE_CUSTOM_CALL_TARGET, vec![vector_input.r#type().into_owned()])
            .with_input_output_alias(0, 0)
            .unwrap()
            .with_side_effect();
        let outputs = CustomCall::custom_call(&operation, std::slice::from_ref(&vector_input)).unwrap();
        assert_eq!(read_f32s(&outputs[0]), vec![2.5, 3.5]);
    }

    /// Batching a custom call with `CustomCallBatching::BroadcastAll` executes the registered handler on device:
    /// every operand is materialized on the batch axis and the elementwise `ryft.test.add_one` handler receives one
    /// batch-prefixed buffer in a single call, agreeing with the per-row result.
    ///
    /// `CustomCallBatching::Sequential` is not available through the *eager* XLA path, whose batching parent is a
    /// [`ProjectedContext`] that rejects every region-carrying operation. That restriction is a property of projected
    /// binding rather than of this rule (the scan-based `rng_bit_generator` batching rule meets the same wall), so
    /// the diagnostic is pinned here alongside the behavior that does execute.
    #[test]
    fn test_eager_custom_call_batching_executes_registered_ffi_handler() {
        use ryft_core::operations::custom_call::{CustomCall, CustomCallBatching, CustomCallOperation};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        ensure_add_one_handler_registered(&client).unwrap();
        let mesh = cpu_mesh(&client);

        let row_type = replicated_type(&mesh, DataType::F32, &[2]);
        let input = Array::from_host_buffer(
            &client,
            replicated_type(&mesh, DataType::F32, &[3, 2]),
            mesh.clone(),
            values_to_bytes::<f32>(&[1.5, 2.5, 3.5, 4.5, 5.5, 6.5]),
        )
        .unwrap();

        let broadcast_row_type = row_type.clone();
        let output: Array<'_> = batch(
            move |row| {
                let operation = CustomCallOperation::new(ADD_ONE_CUSTOM_CALL_TARGET, vec![broadcast_row_type])
                    .with_batching(CustomCallBatching::BroadcastAll);
                Ok(CustomCall::custom_call(&operation, std::slice::from_ref(&row))?.remove(0))
            },
            input.clone(),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(read_f32s(&output), vec![2.5, 3.5, 4.5, 5.5, 6.5, 7.5]);

        let sequential: Result<Array<'_>, _> = batch(
            move |row| {
                let operation = CustomCallOperation::new(ADD_ONE_CUSTOM_CALL_TARGET, vec![row_type])
                    .with_batching(CustomCallBatching::Sequential { unroll: None });
                Ok(CustomCall::custom_call(&operation, std::slice::from_ref(&row))?.remove(0))
            },
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        );
        assert!(
            matches!(&sequential, Err(error) if error.to_string().contains("`scan` cannot carry regions")),
            "{sequential:?}",
        );
    }

    /// A custom call wrapped with `custom_vjp` differentiates through the user-provided rule while the primal
    /// executes the registered FFI handler, which is the documented pairing for differentiable foreign kernels
    /// (the bare operation rejects differentiation).
    #[test]
    fn test_eager_custom_call_differentiates_through_custom_vjp() {
        use ryft_core::operations::custom_call::{CustomCall, CustomCallOperation};
        use ryft_core::{DomainTracer, custom_vjp};

        use crate::XlaDomain;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        ensure_add_one_handler_registered(&client).unwrap();
        let mesh = cpu_mesh(&client);
        let input = f32_vector(&client, &mesh, &[1.5, 2.5]);
        let output_type = replicated_type(&mesh, DataType::F32, &[2]);
        let domain = input.execution_domain();

        type ArrayXlaDomain<'c> = ProjectedContext<XlaDomain<'c>, ArrayType>;

        let add_one = move |x: &DomainTracer<ArrayXlaDomain<'_>>| {
            let operation = CustomCallOperation::new(ADD_ONE_CUSTOM_CALL_TARGET, vec![output_type.clone()]);
            Ok(CustomCall::custom_call(&operation, std::slice::from_ref(x))?.remove(0))
        };
        let function = custom_vjp(
            {
                let add_one = add_one.clone();
                move |x: DomainTracer<ArrayXlaDomain<'_>>| add_one(&x)
            },
            move |x: DomainTracer<ArrayXlaDomain<'_>>| Ok((add_one(&x)?, ())),
            // d(x + 1)/dx is the identity, so the backward rule passes the cotangent through.
            |(), cotangent| Ok(cotangent),
        );
        let (value, gradient) = domain
            .differentiate_at(input)
            .value_and_gradient(|x| function.call(x).unwrap().reduce(&[0], ReductionKind::Sum))
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
        let reference_primary = CpuArray::from_elements(
            ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(primary_values.len())])),
            &primary_values,
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
        let reference_state = CpuArray::from_elements(
            ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(2)])),
            &state_values,
        )
        .unwrap();

        // An odd `u32` element count exercises the padded counter pair and the truncated word layout.
        let u32_output_type = ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Static(5)]));
        let (device_state, device_bits) = state.rng_bit_generator(RandomAlgorithm::ThreeFry, &u32_output_type).unwrap();
        let (reference_new_state, reference_bits) =
            reference_state.rng_bit_generator(RandomAlgorithm::ThreeFry, &u32_output_type).unwrap();
        let device_words = values_from_bytes::<u32>(
            shard_host_bytes(&device_bits.addressable_shards().next().unwrap()).unwrap().as_slice(),
        );
        let reference_words = reference_bits.elements::<u32>().unwrap();
        assert_eq!(device_words, reference_words);
        // Generating five `u32` words runs three cipher invocations, and the counter advances by that invocation
        // count (`7 + 3 = 10`).
        let device_state_words = values_from_bytes::<u64>(
            shard_host_bytes(&device_state.addressable_shards().next().unwrap()).unwrap().as_slice(),
        );
        assert_eq!(device_state_words, vec![42u64, 10u64]);
        assert_eq!(reference_new_state.elements::<u64>().unwrap(), vec![42, 10]);

        let u64_output_type = ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(3)]));
        let (device_state, device_bits) = state.rng_bit_generator(RandomAlgorithm::ThreeFry, &u64_output_type).unwrap();
        let (reference_new_state, reference_bits) =
            reference_state.rng_bit_generator(RandomAlgorithm::ThreeFry, &u64_output_type).unwrap();
        let device_state_words = values_from_bytes::<u64>(
            shard_host_bytes(&device_state.addressable_shards().next().unwrap()).unwrap().as_slice(),
        );
        assert_eq!(device_state_words, reference_new_state.elements::<u64>().unwrap(),);
        let device_words = values_from_bytes::<u64>(
            shard_host_bytes(&device_bits.addressable_shards().next().unwrap()).unwrap().as_slice(),
        );
        let reference_words = reference_bits.elements::<u64>().unwrap();
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
        let reference_state = CpuArray::from_elements(
            ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(3)])),
            &state_values,
        )
        .unwrap();

        // An odd `u32` element count exercises the padded counter quad and the truncated word layout.
        let u32_output_type = ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Static(5)]));
        let (device_state, device_bits) = state.rng_bit_generator(RandomAlgorithm::Philox, &u32_output_type).unwrap();
        let (reference_new_state, reference_bits) =
            reference_state.rng_bit_generator(RandomAlgorithm::Philox, &u32_output_type).unwrap();
        let device_words = values_from_bytes::<u32>(
            shard_host_bytes(&device_bits.addressable_shards().next().unwrap()).unwrap().as_slice(),
        );
        let reference_words = reference_bits.elements::<u32>().unwrap();
        assert_eq!(device_words, reference_words);
        // Generating five `u32` words runs two cipher invocations, and the low counter half advances by that
        // invocation count (`7 + 2 = 9`) while the key and high counter half are unchanged.
        let device_state_words = values_from_bytes::<u64>(
            shard_host_bytes(&device_state.addressable_shards().next().unwrap()).unwrap().as_slice(),
        );
        assert_eq!(device_state_words, vec![42u64, 9u64, 9u64]);
        assert_eq!(reference_new_state.elements::<u64>().unwrap(), vec![42, 9, 9]);

        let u64_output_type = ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(3)]));
        let (device_state, device_bits) = state.rng_bit_generator(RandomAlgorithm::Philox, &u64_output_type).unwrap();
        let (reference_new_state, reference_bits) =
            reference_state.rng_bit_generator(RandomAlgorithm::Philox, &u64_output_type).unwrap();
        let device_state_words = values_from_bytes::<u64>(
            shard_host_bytes(&device_state.addressable_shards().next().unwrap()).unwrap().as_slice(),
        );
        assert_eq!(device_state_words, reference_new_state.elements::<u64>().unwrap(),);
        let device_words = values_from_bytes::<u64>(
            shard_host_bytes(&device_bits.addressable_shards().next().unwrap()).unwrap().as_slice(),
        );
        let reference_words = reference_bits.elements::<u64>().unwrap();
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
        let reference_state = CpuArray::from_elements(
            ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(2)])),
            &state_values,
        )
        .unwrap();

        let shape = Shape::new(vec![Dimension::Static(8)]);
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
        assert_eq!(reference_samples.elements::<i32>().unwrap(), vec![1]);

        let (_, device_keys) = state.split_key(2).unwrap();
        let (_, reference_keys) = reference_state.split_key(2).unwrap();
        for (device_key, reference_key) in device_keys.iter().zip(reference_keys.iter()) {
            let device_words = values_from_bytes::<u64>(
                shard_host_bytes(&device_key.addressable_shards().next().unwrap()).unwrap().as_slice(),
            );
            let reference_words = reference_key.elements::<u64>().unwrap();
            assert_eq!(device_words, reference_words);
        }
    }

    /// An accumulation-typed dot (`f8e4m3fn × f8e4m3fn → f32`) agrees between the XLA-backed eager array backend
    /// and the reference array backend: the operands stay at the narrow element type on device and the contraction
    /// accumulates at `f32`. Every value used is exactly representable in `f8e4m3fn`, so both backends are exact.
    #[test]
    fn test_eager_accumulation_typed_dot_parity_with_reference_backend() {
        use ryft_core::DotDimensionNumbers;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);

        let operand_type =
            ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let lhs_values = [0.5f64, 1.0, 1.5, 2.0];
        let rhs_values = [1.0f64, 0.5, 0.5, 1.0];
        let reference_lhs = CpuArray::from_f64s(operand_type.clone(), lhs_values.to_vec());
        let reference_rhs = CpuArray::from_f64s(operand_type.clone(), rhs_values.to_vec());
        let lhs_bytes = reference_lhs.logical_bytes();
        let rhs_bytes = reference_rhs.logical_bytes();
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
    /// backend on CPU, where XLA executes the portable decomposition attached to the `xla.scaled_dot` composite.
    /// CUDA may replace the same composite with a native implementation. Every element and scale is exactly
    /// representable, so both reference and CPU-XLA backends are exact.
    #[test]
    fn test_eager_scaled_dot_parity_with_reference_backend() {
        use ryft_core::{DotDimensionNumbers, ScaledDot};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);

        let element_type =
            ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(16)]));
        let scale_type =
            ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]));
        const F4_CANDIDATES: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, -0.5, -1.0];
        let element_values =
            |seed: usize| (0..32).map(|index| F4_CANDIDATES[(index * 5 + seed) % 8]).collect::<Vec<_>>();
        let reference_lhs = CpuArray::from_f64s(element_type.clone(), element_values(0));
        let reference_rhs = CpuArray::from_f64s(element_type.clone(), element_values(3));
        let reference_lhs_scales = CpuArray::from_f64s(scale_type.clone(), vec![0.5, 2.0]);
        let reference_rhs_scales = CpuArray::from_f64s(scale_type.clone(), vec![2.0, 0.5]);
        let bits = CpuArray::logical_bytes;
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

        let dimensions = DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new());
        let device_product = lhs
            .scaled_dot(&rhs, Some(&lhs_scales), Some(&rhs_scales), Some(&dimensions), Some(DataType::F32))
            .unwrap();
        let reference_product = reference_lhs
            .scaled_dot(
                &reference_rhs,
                Some(&reference_lhs_scales),
                Some(&reference_rhs_scales),
                Some(&dimensions),
                Some(DataType::F32),
            )
            .unwrap();
        assert_eq!(device_product.r#type().data_type(), DataType::F32);
        let device_value_f64s = read_f32s(&device_product).iter().map(|value| f64::from(*value)).collect::<Vec<_>>();
        assert_eq!(device_value_f64s, reference_product.to_f64s());
    }

    #[test]
    fn test_eager_dot_product_attention_parity_with_reference_backend() {
        use ryft_core::operations::attention::{AttentionConfiguration, AttentionInputs, DotProductAttention};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let dimensions = [1usize, 3, 2, 2];
        let host_type =
            ArrayType::new(DataType::F32, Shape::new(dimensions.iter().copied().map(Dimension::Static).collect()));
        let device_type = replicated_type(&mesh, DataType::F32, &dimensions);
        let query_values = (0..12).map(|index| (index as f32 - 5.0) * 0.25).collect::<Vec<_>>();
        let key_values = (0..12).map(|index| ((index * 5 % 13) as f32 - 6.0) * 0.25).collect::<Vec<_>>();
        let value_values = (0..12).map(|index| ((index * 3 % 7) as f32 - 3.0) * 0.5).collect::<Vec<_>>();
        let device = |values: &[f32]| {
            Array::from_host_buffer(&client, device_type.clone(), mesh.clone(), values_to_bytes(values).as_slice())
                .unwrap()
        };
        let reference = |values: &[f32]| {
            CpuArray::from_f64s(host_type.clone(), values.iter().map(|value| f64::from(*value)).collect())
        };

        for configuration in [
            AttentionConfiguration::new().with_scale(0.5),
            AttentionConfiguration::new().with_scale(0.5).with_causal(true).with_local_window((1, 0)),
        ] {
            let device_output = Array::dot_product_attention(
                AttentionInputs::new(device(&query_values), device(&key_values), device(&value_values)),
                configuration,
            )
            .unwrap()
            .0;
            let reference_output = CpuArray::dot_product_attention(
                AttentionInputs::new(reference(&query_values), reference(&key_values), reference(&value_values)),
                configuration,
            )
            .unwrap()
            .0;

            read_f32s(&device_output).iter().zip(reference_output.to_f64s()).for_each(|(actual, expected)| {
                assert!((f64::from(*actual) - expected).abs() < 1e-5, "expected {expected} but got {actual}");
            });
        }

        // The compiled composition also preserves the complete structural surface: a broadcast scalar bias, an
        // arbitrary mask, independent length vectors, the default scale, an asymmetric window, and the residual.
        let reference_bias = CpuArray::from_f64s(ArrayType::scalar(DataType::F32), vec![0.25]);
        let reference_mask = CpuArray::from_elements(
            ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(3), Dimension::Static(3)])),
            &[true, false, false, true, true, false, true, true, true],
        )
        .unwrap();
        let lengths_type = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(1)]));
        let reference_query_lengths = CpuArray::from_elements(lengths_type.clone(), &[2_i32]).unwrap();
        let reference_key_value_lengths = CpuArray::from_elements(lengths_type, &[3_i32]).unwrap();
        let device_value = |value: &CpuArray, dimensions: &[usize]| {
            Array::from_host_buffer(
                &client,
                replicated_type(&mesh, value.r#type().data_type(), dimensions),
                mesh.clone(),
                value.logical_bytes().as_slice(),
            )
            .unwrap()
        };
        let configuration =
            AttentionConfiguration::new().with_causal(true).with_local_window((2, 1)).with_residual(true);
        let device_inputs = AttentionInputs {
            query: device(&query_values),
            key: device(&key_values),
            value: device(&value_values),
            bias: Some(device_value(&reference_bias, &[])),
            mask: Some(device_value(&reference_mask, &[3, 3])),
            query_sequence_lengths: Some(device_value(&reference_query_lengths, &[1])),
            key_value_sequence_lengths: Some(device_value(&reference_key_value_lengths, &[1])),
        };
        let reference_inputs = AttentionInputs {
            query: reference(&query_values),
            key: reference(&key_values),
            value: reference(&value_values),
            bias: Some(reference_bias),
            mask: Some(reference_mask),
            query_sequence_lengths: Some(reference_query_lengths),
            key_value_sequence_lengths: Some(reference_key_value_lengths),
        };
        let (device_output, device_residual) = Array::dot_product_attention(device_inputs, configuration).unwrap();
        let (reference_output, reference_residual) =
            CpuArray::dot_product_attention(reference_inputs, configuration).unwrap();
        for (actual, expected) in [
            (read_f32s(&device_output), reference_output.to_f64s()),
            (read_f32s(&device_residual.unwrap()), reference_residual.unwrap().to_f64s()),
        ] {
            actual.iter().zip(expected).for_each(|(actual, expected)| {
                assert!((f64::from(*actual) - expected).abs() < 1e-5, "expected {expected} but got {actual}");
            });
        }
    }

    #[test]
    fn test_eager_differentiable_dot_product_attention_gradient() {
        use ryft_core::operations::attention::{
            AttentionConfiguration, AttentionInputs, differentiable_dot_product_attention,
        };
        use ryft_core::{ArrayOperation, EagerContext, ProjectedContext};

        use crate::XlaDomain;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);
        let dimensions = [1usize, 2, 1, 2];
        let values = [0.25_f32, -0.5, 0.75, 1.0];
        let device_type = replicated_type(&mesh, DataType::F32, &dimensions);
        let device = || {
            Array::from_host_buffer(&client, device_type.clone(), mesh.clone(), values_to_bytes(&values).as_slice())
                .unwrap()
        };
        let host_type =
            ArrayType::new(DataType::F32, Shape::new(dimensions.iter().copied().map(Dimension::Static).collect()));
        let reference =
            || CpuArray::from_f64s(host_type.clone(), values.iter().map(|value| f64::from(*value)).collect());
        let configuration = AttentionConfiguration::new().with_scale(0.5).with_causal(true);

        type ArrayXlaDomain<'c> = ProjectedContext<XlaDomain<'c>, ArrayType>;
        let function = differentiable_dot_product_attention::<ArrayXlaDomain<'_>>(configuration);
        let inputs = AttentionInputs::new(device(), device(), device());
        let domain = inputs.query.execution_domain();
        let (loss, gradients) = domain
            .differentiate_at(inputs)
            .value_and_gradient(|inputs| function.call(inputs).unwrap().reduce(&[0, 1, 2, 3], ReductionKind::Sum))
            .unwrap();

        let reference_function =
            differentiable_dot_product_attention::<EagerContext<CpuArray, ArrayOperation<CpuArray>>>(configuration);
        let (reference_loss, reference_gradients) = EagerContext::<CpuArray, ArrayOperation<CpuArray>>::new()
            .differentiate_at(AttentionInputs::new(reference(), reference(), reference()))
            .value_and_gradient(|inputs| {
                reference_function.call(inputs).unwrap().reduce(&[0, 1, 2, 3], ReductionKind::Sum)
            })
            .unwrap();

        assert!((f64::from(read_f32s(&loss)[0]) - reference_loss.to_f64s()[0]).abs() < 1e-5);
        for (actual, expected) in [
            (&gradients.query, &reference_gradients.query),
            (&gradients.key, &reference_gradients.key),
            (&gradients.value, &reference_gradients.value),
        ] {
            read_f32s(actual).iter().zip(expected.to_f64s()).for_each(|(actual, expected)| {
                assert!((f64::from(*actual) - expected).abs() < 1e-5, "expected {expected} but got {actual}");
            });
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

        let reshaped = matrix.reshape(Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])).unwrap();
        assert_eq!(reshaped.shape(), StaticShape::new(vec![3, 2]));
        assert_eq!(read_f32s(&reshaped), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let flattened = reshaped.reshape(Shape::new(vec![Dimension::Static(6)])).unwrap();
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
        let trimmed = vector.pad(&padding_value, &[-1], &[-1], &[0]).unwrap();
        assert_eq!(read_f32s(&trimmed), vec![2.0, 3.0]);
        let trimmed_and_dilated = vector.pad(&padding_value, &[-1], &[1], &[1]).unwrap();
        assert_eq!(read_f32s(&trimmed_and_dilated), vec![0.5, 2.0, 0.5, 3.0, 0.5, 4.0, 0.5]);

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
    fn test_eager_dimension_gateways() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = cpu_mesh_with_axis_size(&client, 2);

        let matrix = f32_matrix(&client, &mesh, 2, 3, &[0.0; 6]);
        assert_eq!(matrix.dimension_size(0), Ok(2));
        assert_eq!(matrix.dimension_size(-1), Ok(3));
        assert!(matches!(
            matrix.dimension_size(2),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`dimension_size` axis 2 is out of bounds for rank 2",
        ));

        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let sharded_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(5)]))
            .with_sharding(sharding)
            .unwrap();
        let sharded =
            Array::from_host_buffer(&client, sharded_type, mesh.clone(), values_to_bytes::<f32>(&[0.0; 5])).unwrap();
        assert_eq!(sharded.dimension_size(0), Ok(5));

        let scalar_type = replicated_type(&mesh, DataType::I32, &[]);
        let scalar = Array::from_host_buffer(&client, scalar_type.clone(), mesh.clone(), 5_i32.to_ne_bytes()).unwrap();
        let variable = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        assert_eq!(scalar.to_dimension(variable.clone()).unwrap().extent(), 5);

        let negative = Array::from_host_buffer(&client, scalar_type, mesh, (-1_i32).to_ne_bytes()).unwrap();
        assert!(matches!(
            negative.to_dimension(variable),
            Err(ProgramError::InvalidArgument { message })
                if message
                    == "`dimension_from_scalar` scalar input must be a nonnegative host-representable extent but is -1",
        ));
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
        assert_eq!(sum.execution_domain().parent().cache_size(), 1);

        let product = sum.mul(&b).unwrap();
        assert_eq!(read_f32s(&product), vec![12.0, 24.0]);
        assert!(std::ptr::eq(product.client().unwrap(), &client));

        // All values derived from `a` share one dispatch cache: `add` and `mul` each compiled once, and repeating
        // `add` at the same input signature is a cache hit.
        assert_eq!(product.execution_domain().parent().cache_size(), 2);
        assert_eq!(a.execution_domain().parent().cache_size(), 2);
        let repeated = a.add(&b).unwrap();
        assert_eq!(read_f32s(&repeated), vec![4.0, 6.0]);
        assert_eq!(a.execution_domain().parent().cache_size(), 2);
    }

    #[test]
    fn test_execution_domain_recovery() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })).unwrap();
        let mesh = cpu_mesh(&client);

        // Arrays with an attached client recover a client-backed domain.
        let array = f32_vector(&client, &mesh, &[1.0, 2.0]);
        let domain = array.execution_domain();
        assert!(std::ptr::eq(domain.parent().client().unwrap(), &client));

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
        let (value, tangent): (Array<'_>, Array<'_>) =
            domain.jvp(|x, ()| Mul::mul(&x, &x), x.clone(), tangents, ()).unwrap();
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

        let (value, tangent) =
            primal.execution_domain().jvp(|input, ()| input.abs(), primal.clone(), tangent, ()).unwrap();
        assert_eq!(read_f32s(&value), vec![0.0]);
        assert_eq!(read_f32s(&tangent), vec![3.0]);

        let primal = c64_scalar(&client, &mesh, num_complex::Complex::new(0.0, 0.0));
        let tangent = c64_scalar(&client, &mesh, num_complex::Complex::new(1.0, 2.0));
        let (value, tangent) =
            primal.execution_domain().jvp(|input, ()| input.abs(), primal.clone(), tangent, ()).unwrap();
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
        let input_type = ArrayType::new(DataType::C64, Shape::new(vec![Dimension::Static(4)]))
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
        let expected_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(sharding)
            .unwrap();
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
            .differentiate_at(x.clone())
            .value_and_gradient(|x| {
                let squared = Mul::mul(&x, &x).unwrap();
                squared.reduce(&[0], ReductionKind::Sum)
            })
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
            .differentiate_at(x.clone())
            .gradient(|x| {
                let squared = Mul::mul(&x, &x).unwrap();
                squared.reduce(&[0], ReductionKind::Sum)
            })
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
        let (value, pullback) = domain.vjp(|x, ()| Ok(vec![Mul::mul(&x, &x)?]), x.clone(), ()).unwrap();
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
        let (value, gradient) =
            domain.differentiate_at(x.clone()).holomorphic().value_and_gradient(|x| x.clone() * x).unwrap();
        assert_c64_close(read_c64s(&value)[0], z * z);
        assert_c64_close(read_c64s(&gradient)[0], z + z);

        // Complex `atan2(y, x)` uses XLA's principal value and its two holomorphic partial derivatives away from
        // singularities and branch cuts.
        let y = num_complex::Complex::new(0.7f32, -0.2f32);
        let x_value = num_complex::Complex::new(-0.3f32, 0.4f32);
        let y_array = c64_scalar(&client, &mesh, y);
        let x_array = c64_scalar(&client, &mesh, x_value);
        let (value, (y_gradient, x_gradient)) = domain
            .differentiate_at((y_array, x_array))
            .holomorphic()
            .value_and_gradient(|(y, x)| y.atan2(&x))
            .unwrap();
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
        let gradient = domain
            .differentiate_at(x.clone())
            .gradient(|x| (x.clone() * x.conjugate().unwrap()).real().unwrap())
            .unwrap();
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
            .differentiate_at(x.clone())
            .with_auxiliary_output()
            .value_and_gradient(|x| {
                let squared = Mul::mul(&x, &x).unwrap();
                let aux = Add::add(&x, &x).unwrap();
                (squared.reduce(&[0], ReductionKind::Sum), aux)
            })
            .unwrap();
        assert_eq!(read_f32s(&value), vec![14.0]);
        assert_eq!(read_f32s(&aux), vec![2.0, 4.0, 6.0]);
        assert_eq!(read_f32s(&gradient), vec![2.0, 4.0, 6.0]);
    }

    /// Differentiation captures remain device-resident residuals with their nontrivial sharding and placement.
    #[test]
    fn test_eager_differentiation_capture_preserves_nontrivial_sharding_and_placement() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin.client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) })).unwrap();
        let mesh = cpu_mesh_with_axis_size(&client, 2);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let primal = Array::from_host_buffer(
            &client,
            r#type.clone(),
            mesh.clone(),
            values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(),
        )
        .unwrap();
        let capture = Array::from_host_buffer(
            &client,
            r#type,
            mesh.clone(),
            values_to_bytes::<f32>(&[4.0, 5.0, 6.0, 7.0]).as_slice(),
        )
        .unwrap();

        let (value, gradient) = differentiate_at(primal.clone())
            .with_captures(capture.clone())
            .in_context(&primal.execution_domain())
            .value_and_gradient(|input, capture| Mul::mul(&input, &capture).unwrap().reduce(&[0], ReductionKind::Sum))
            .unwrap();
        assert_eq!(read_f32s(&value), vec![60.0]);
        assert_eq!(read_f64_coordinates(&gradient), vec![4.0, 5.0, 6.0, 7.0]);
        assert_eq!(gradient.sharding(), &sharding);
        assert!(gradient.client().is_some());

        let (_, pushforward) = differentiate_at(primal.clone())
            .with_captures(capture)
            .in_context(&primal.execution_domain())
            .linearize(|input, capture| Mul::mul(&input, &capture))
            .unwrap();
        let capture_residual = pushforward
            .residuals()
            .iter()
            .find(|residual| read_f64_coordinates(residual) == vec![4.0, 5.0, 6.0, 7.0])
            .expect("capture should survive as a pushforward residual");
        assert_eq!(capture_residual.sharding(), &sharding);
        assert!(capture_residual.client().is_some());
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
            .differentiate_at(x.clone())
            .gradient(|x| {
                let squared =
                    batch(|item| Ok(item.clone() * item), x, BatchAxis::new(0), BatchAxis::new(0), None).unwrap();
                squared.reduce(&[0], ReductionKind::Sum)
            })
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
        let jacobian = domain.differentiate_at(x).jacobian_forward(|x| Mul::mul(&x, &x)).unwrap();

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
        let jacobian = domain.differentiate_at(x).jacobian_forward(|x| Mul::mul(&x, &x)).unwrap();

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

        let forward = context
            .differentiate_at(input.clone())
            .holomorphic()
            .jacobian_forward(|input| Mul::mul(&input, &input))
            .unwrap();
        let reverse = context
            .differentiate_at(input.clone())
            .holomorphic()
            .jacobian_reverse(|input| Mul::mul(&input, &input))
            .unwrap();
        let hessian = context.differentiate_at(input).holomorphic().hessian(|input| Mul::mul(&input, &input)).unwrap();

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
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(sharding)
            .unwrap();
        let bytes = values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]);
        let x = Array::from_host_buffer(&client, r#type, mesh.clone(), bytes.as_slice()).unwrap();
        let domain = x.execution_domain();
        let jacobian = domain.differentiate_at(x).jacobian_forward(|x| Mul::mul(&x, &x)).unwrap();

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
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(sharding)
            .unwrap();
        let bytes = values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]);
        let x = Array::from_host_buffer(&client, r#type, mesh.clone(), bytes.as_slice()).unwrap();
        let jacobian = differentiate_at(x).jacobian_reverse(|x| Mul::mul(&x, &x)).unwrap();

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
        let jacobian = differentiate_at(x).jacobian_reverse(|x| Mul::mul(&x, &x)).unwrap();

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
            .differentiate_at(x)
            .hessian(|x| {
                let squared = Mul::mul(&x, &x).unwrap();
                Ok(squared.reduce(&[0], ReductionKind::Sum))
            })
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
        let (value, pullback) = domain.vjp(|x, ()| Ok(vec![Mul::mul(&x, &x)?]), x, ()).unwrap();
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
