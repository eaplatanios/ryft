//! Placement of whole trees of host values and arrays on devices, following `jax.device_put`.

use half::{bf16, f16};
use ryft_core::{
    ArrayType, DataType, Device, Dimension as CoreDimension, Parameter, Parameterized, ParameterizedFamily, Shape,
};

use crate::arrays::placement::DevicePutTarget;
use crate::arrays::{Array, ArrayError};
use crate::{Domain, Error, FromPjrt};

// TODO(eaplatanios): Review this module.

/// Options for the higher-level [`device_put()`] API.
///
/// Each field follows JAX's tree-prefix semantics: when a field is present, its structure is
/// broadcast over the input tree and applied leafwise.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DevicePutOptions<
    DeviceTarget = DevicePutTarget,
    SourceTarget = DevicePutTarget,
    Donate = bool,
    MayAlias = Option<bool>,
> {
    /// Destination placement tree prefix. When absent, host leaves are committed to the default
    /// local device and [`Array`] leaves preserve their current placement.
    device: Option<DeviceTarget>,

    /// Source placement tree prefix. This is validated for [`Array`] leaves and ignored for host
    /// leaves, which do not carry runtime placement metadata before upload.
    src: Option<SourceTarget>,

    /// Donation tree prefix. This is best-effort in the current `ryft` runtime.
    donate: Option<Donate>,

    /// May-alias tree prefix. `Some(false)` forces a fresh array result when possible.
    may_alias: Option<MayAlias>,
}

impl<DeviceTarget, SourceTarget, Donate, MayAlias> DevicePutOptions<DeviceTarget, SourceTarget, Donate, MayAlias> {
    /// Creates a new [`DevicePutOptions`].
    #[inline]
    pub fn new(
        device: Option<DeviceTarget>,
        src: Option<SourceTarget>,
        donate: Option<Donate>,
        may_alias: Option<MayAlias>,
    ) -> Self {
        Self { device, src, donate, may_alias }
    }

    /// Returns the destination placement tree prefix, if one was provided.
    #[inline]
    pub fn device(&self) -> Option<&DeviceTarget> {
        self.device.as_ref()
    }

    /// Returns the source placement tree prefix, if one was provided.
    #[inline]
    pub fn src(&self) -> Option<&SourceTarget> {
        self.src.as_ref()
    }

    /// Returns the donation tree prefix, if one was provided.
    #[inline]
    pub fn donate(&self) -> Option<&Donate> {
        self.donate.as_ref()
    }

    /// Returns the may-alias tree prefix, if one was provided.
    #[inline]
    pub fn may_alias(&self) -> Option<&MayAlias> {
        self.may_alias.as_ref()
    }

    pub(crate) fn into_parts(self) -> (Option<DeviceTarget>, Option<SourceTarget>, Option<Donate>, Option<MayAlias>) {
        (self.device, self.src, self.donate, self.may_alias)
    }

    /// Creates a new [`DevicePutOptions`] with all fields unset.
    #[inline]
    pub fn empty() -> Self {
        Self::new(None, None, None, None)
    }
}

impl<DeviceTarget, SourceTarget, Donate, MayAlias> Default
    for DevicePutOptions<DeviceTarget, SourceTarget, Donate, MayAlias>
{
    fn default() -> Self {
        Self::empty()
    }
}

impl DevicePutOptions<DevicePutTarget, DevicePutTarget, bool, Option<bool>> {
    /// Creates default high-level [`device_put()`] options without requiring generic type inference.
    #[inline]
    pub fn defaults() -> Self {
        Self::empty()
    }
}

/// Leaf types accepted by the higher-level [`device_put()`] API.
///
/// A [`DevicePutLeaf`] consumes one input leaf and materializes one runtime [`Array`] leaf. `ryft`
/// currently provides implementations for runtime [`Array`] leaves, primitive scalar host values,
/// and owned `ndarray::Array`s when the `ndarray` feature is enabled.
pub trait DevicePutLeaf<'c>: Parameter {
    /// Converts `self` into one runtime [`Array`] using the provided leafwise placement options.
    ///
    /// # Parameters
    ///
    ///   - `engine`: [`Domain`] wrapping the PJRT client used to materialize any needed
    ///     destination buffers and to cache any compiled reshard executables.
    ///   - `device`: Destination placement for this leaf, if one was specified.
    ///   - `src`: Source placement for this leaf, if one was specified.
    ///   - `donate`: Best-effort donation flag for this leaf.
    ///   - `may_alias`: Best-effort aliasing hint for this leaf.
    fn device_put_leaf(
        self,
        engine: &Domain<'c>,
        device: Option<DevicePutTarget>,
        src: Option<DevicePutTarget>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Array<'c>, Error>;
}

impl<'c, T: DenseHostDevicePutLeaf + Parameter> DevicePutLeaf<'c> for T {
    fn device_put_leaf(
        self,
        engine: &Domain<'c>,
        device: Option<DevicePutTarget>,
        _src: Option<DevicePutTarget>,
        _donate: bool,
        _may_alias: Option<bool>,
    ) -> Result<Array<'c>, Error> {
        let client = engine.client();
        let (shape, element_type, bytes) = self.into_dense_host_array();
        let (mesh, sharding) = match device {
            Some(device) => device.resolve(shape.len())?,
            None => {
                let device = client
                    .addressable_devices()?
                    .into_iter()
                    .next()
                    .ok_or_else(|| Error::from(ArrayError::MissingDefaultDevice))?;
                DevicePutTarget::device(Device::from_pjrt(device)?).resolve(shape.len())?
            }
        };
        let r#type =
            ArrayType::new(element_type, Shape::new(shape.iter().copied().map(CoreDimension::Static).collect()));
        Array::from_host_buffer_with_sharding(engine, r#type, sharding, mesh, bytes.as_slice())
    }
}

impl<'c> DevicePutLeaf<'c> for Array<'c> {
    fn device_put_leaf(
        self,
        engine: &Domain<'c>,
        device: Option<DevicePutTarget>,
        src: Option<DevicePutTarget>,
        _donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Array<'c>, Error> {
        let current_mesh = self.mesh().clone();
        let current_sharding = self.sharding().clone();
        if let Some(src) = src {
            let (expected_mesh, expected_sharding) = src.resolve(current_sharding.rank())?;
            if expected_mesh != current_mesh || expected_sharding != current_sharding {
                return Err(Error::from(ArrayError::SourcePlacementMismatch {
                    expected_mesh,
                    expected_sharding,
                    actual_mesh: current_mesh.clone(),
                    actual_sharding: current_sharding.clone(),
                }));
            }
        }

        let (target_mesh, target_sharding) = match device {
            Some(device) => device.resolve(self.sharding().rank())?,
            None => (current_mesh.clone(), current_sharding.clone()),
        };
        if target_mesh == current_mesh && target_sharding == current_sharding && may_alias != Some(false) {
            Ok(self)
        } else {
            self.to_placement(engine, DevicePutTarget::placement(target_mesh, target_sharding)?)
        }
    }
}

/// Higher-level `ryft` analogue of JAX's
/// [`jax.device_put`](https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html).
///
/// The input `x` may be one supported leaf or a `Parameterized` tree of supported leaves. Any
/// provided `device`, `src`, `donate`, and `may_alias` fields follow tree-prefix broadcasting
/// semantics over `x`.
///
/// Host leaves are committed to the default local device when `options.device` is absent. Existing
/// [`Array`] leaves preserve their current placement when `options.device` is absent.
pub fn device_put<
    'c,
    P: DevicePutLeaf<'c>,
    Input: Parameterized<P>,
    DeviceTarget: Parameterized<DevicePutTarget>,
    SourceTarget: Parameterized<DevicePutTarget>,
    Donate: Parameterized<bool>,
    MayAlias: Parameterized<Option<bool>>,
>(
    engine: &Domain<'c>,
    x: Input,
    options: DevicePutOptions<DeviceTarget, SourceTarget, Donate, MayAlias>,
) -> Result<<Input as Parameterized<P>>::To<Array<'c>>, Error>
where
    <Input as Parameterized<P>>::Family: ParameterizedFamily<Array<'c>>
        + ParameterizedFamily<DevicePutTarget>
        + ParameterizedFamily<bool>
        + ParameterizedFamily<Option<bool>>,
{
    let structure = x.parameter_structure();
    let leaf_count = structure.parameter_count();
    let (device, src, donate, may_alias) = options.into_parts();

    let device_values = match device {
        Some(device) => Input::To::<DevicePutTarget>::from_broadcasted_named_parameters(
            structure.clone(),
            device.into_named_parameters(),
        )?
        .into_parameters()
        .map(Some)
        .collect::<Vec<_>>(),
        None => vec![None; leaf_count],
    };
    let src_values = match src {
        Some(src) => Input::To::<DevicePutTarget>::from_broadcasted_named_parameters(
            structure.clone(),
            src.into_named_parameters(),
        )?
        .into_parameters()
        .map(Some)
        .collect::<Vec<_>>(),
        None => vec![None; leaf_count],
    };
    let donate_values = match donate {
        Some(donate) => {
            Input::To::<bool>::from_broadcasted_named_parameters(structure.clone(), donate.into_named_parameters())?
                .into_parameters()
                .collect::<Vec<_>>()
        }
        None => vec![false; leaf_count],
    };
    let may_alias_values = match may_alias {
        Some(may_alias) => Input::To::<Option<bool>>::from_broadcasted_named_parameters(
            structure.clone(),
            may_alias.into_named_parameters(),
        )?
        .into_parameters()
        .collect::<Vec<_>>(),
        None => vec![None; leaf_count],
    };

    let mut output_parameters = Vec::with_capacity(leaf_count);
    let mut device_values = device_values.into_iter();
    let mut src_values = src_values.into_iter();
    let mut donate_values = donate_values.into_iter();
    let mut may_alias_values = may_alias_values.into_iter();
    for parameter in x.into_parameters() {
        output_parameters.push(
            parameter.device_put_leaf(
                engine,
                device_values
                    .next()
                    .expect("device tree-prefix broadcasting should produce one placement per input leaf"),
                src_values.next().expect("src tree-prefix broadcasting should produce one placement per input leaf"),
                donate_values
                    .next()
                    .expect("donate tree-prefix broadcasting should produce one flag per input leaf"),
                may_alias_values
                    .next()
                    .expect("may_alias tree-prefix broadcasting should produce one flag per input leaf"),
            )?,
        );
    }
    Input::To::<Array<'c>>::from_parameters(structure, output_parameters).map_err(Into::into)
}

pub(crate) trait DenseHostDevicePutLeaf {
    fn into_dense_host_array(self) -> (Vec<usize>, DataType, Vec<u8>);
}

pub(crate) trait DenseHostElement {
    const DATA_TYPE: DataType;

    fn append_ne_bytes(&self, bytes: &mut Vec<u8>);
}

macro_rules! impl_dense_host_element {
    ($ty:ty, $data_type:expr) => {
        impl DenseHostElement for $ty {
            const DATA_TYPE: DataType = $data_type;

            fn append_ne_bytes(&self, bytes: &mut Vec<u8>) {
                bytes.extend_from_slice(&self.to_ne_bytes());
            }
        }
    };
}

impl DenseHostElement for bool {
    const DATA_TYPE: DataType = DataType::Boolean;

    fn append_ne_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.push(u8::from(*self));
    }
}

impl DenseHostElement for bf16 {
    const DATA_TYPE: DataType = DataType::BF16;

    fn append_ne_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&self.to_bits().to_ne_bytes());
    }
}

impl DenseHostElement for f16 {
    const DATA_TYPE: DataType = DataType::F16;

    fn append_ne_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&self.to_bits().to_ne_bytes());
    }
}

impl_dense_host_element!(i8, DataType::I8);
impl_dense_host_element!(i16, DataType::I16);
impl_dense_host_element!(i32, DataType::I32);
impl_dense_host_element!(i64, DataType::I64);
impl_dense_host_element!(u8, DataType::U8);
impl_dense_host_element!(u16, DataType::U16);
impl_dense_host_element!(u32, DataType::U32);
impl_dense_host_element!(u64, DataType::U64);
impl_dense_host_element!(f32, DataType::F32);
impl_dense_host_element!(f64, DataType::F64);

macro_rules! impl_scalar_dense_host_leaf {
    ($ty:ty) => {
        impl DenseHostDevicePutLeaf for $ty {
            fn into_dense_host_array(self) -> (Vec<usize>, DataType, Vec<u8>) {
                let mut bytes = Vec::with_capacity(size_of::<$ty>());
                self.append_ne_bytes(&mut bytes);
                (Vec::new(), <$ty as DenseHostElement>::DATA_TYPE, bytes)
            }
        }
    };
}

impl_scalar_dense_host_leaf!(bool);
impl_scalar_dense_host_leaf!(i8);
impl_scalar_dense_host_leaf!(i16);
impl_scalar_dense_host_leaf!(i32);
impl_scalar_dense_host_leaf!(i64);
impl_scalar_dense_host_leaf!(u8);
impl_scalar_dense_host_leaf!(u16);
impl_scalar_dense_host_leaf!(u32);
impl_scalar_dense_host_leaf!(u64);
impl_scalar_dense_host_leaf!(bf16);
impl_scalar_dense_host_leaf!(f16);
impl_scalar_dense_host_leaf!(f32);
impl_scalar_dense_host_leaf!(f64);

#[cfg(feature = "ndarray")]
impl<T: Clone + DenseHostElement, D: ndarray::Dimension> DenseHostDevicePutLeaf for ndarray::Array<T, D> {
    fn into_dense_host_array(self) -> (Vec<usize>, DataType, Vec<u8>) {
        let standard_layout = self.as_standard_layout().to_owned();
        let element_count = standard_layout.len();
        let mut bytes = Vec::with_capacity(element_count * size_of::<T>());
        for element in standard_layout.iter() {
            element.append_ne_bytes(&mut bytes);
        }
        (standard_layout.shape().to_vec(), T::DATA_TYPE, bytes)
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use ryft_pjrt::{BufferType, ClientOptions, CpuClientOptions, load_cpu_plugin};

    use crate::Session;
    use crate::tests::{values_from_bytes, values_to_bytes};

    use super::*;

    #[test]
    fn test_device_put_broadcasts_root_placement_over_array_tuple() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2), ..Default::default() }))
            .unwrap();
        let engine = Session::new(&client).domain();
        let client_devices = client.addressable_devices().unwrap();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(client_devices[0].id().unwrap(), client_devices[0].process_index().unwrap())],
        )
        .unwrap();
        let source_sharding = Sharding::replicated(source_mesh.logical_mesh().clone(), 1);
        let first_source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(source_sharding.clone())
            .unwrap();
        let first_source_array = Array::from_host_buffer(
            &engine,
            first_source_type,
            source_mesh.clone(),
            values_to_bytes::<f32>(&[0.0, 1.0, 2.0, 3.0]).as_slice(),
        )
        .unwrap();
        let second_source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(source_sharding)
            .unwrap();
        let second_source_array = Array::from_host_buffer(
            &engine,
            second_source_type,
            source_mesh,
            values_to_bytes::<f32>(&[10.0, 11.0, 12.0, 13.0]).as_slice(),
        )
        .unwrap();

        let target_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect(),
        )
        .unwrap();
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let moved_arrays = device_put(
            &engine,
            (first_source_array, second_source_array),
            DevicePutOptions::new(
                Some(DevicePutTarget::Placement { mesh: target_mesh.clone(), sharding: target_sharding.clone() }),
                Option::<DevicePutTarget>::None,
                Option::<bool>::None,
                Option::<Option<bool>>::None,
            ),
        )
        .unwrap();

        assert_eq!(
            moved_arrays.0.sharding().visualize().unwrap().render(false),
            indoc! {"
                    ┌─────┬─────┐
                    │  0  │  1  │
                    └─────┴─────┘
                "}
            .trim_end()
            .to_string()
        );
        assert_eq!(
            moved_arrays.1.sharding().visualize().unwrap().render(false),
            indoc! {"
                    ┌─────┬─────┐
                    │  0  │  1  │
                    └─────┴─────┘
                "}
            .trim_end()
            .to_string()
        );
        let first_moved_shard_bytes = moved_arrays
            .0
            .device_shard(client_devices[1].id().unwrap())
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(first_moved_shard_bytes.as_slice().len() % size_of::<f32>(), 0);
        assert_eq!(values_from_bytes::<f32>(first_moved_shard_bytes.as_slice()), vec![2.0, 3.0]);
        let second_moved_shard_bytes = moved_arrays
            .1
            .device_shard(client_devices[0].id().unwrap())
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(second_moved_shard_bytes.as_slice().len() % size_of::<f32>(), 0);
        assert_eq!(values_from_bytes::<f32>(second_moved_shard_bytes.as_slice()), vec![10.0, 11.0]);
    }

    #[test]
    fn test_device_put_preserves_partially_addressable_array_when_device_is_absent() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let engine = Session::new(&client).domain();
        let local_device = client.addressable_devices().unwrap().remove(0);
        let local_device_id = local_device.id().unwrap();
        let remote_device_id = local_device_id + 1;
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(local_device_id, local_device.process_index().unwrap()), Device::new(remote_device_id, 1)],
        )
        .unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let local_source_buffer = client
            .buffer(values_to_bytes::<f32>(&[0.0, 1.0]).as_slice(), BufferType::F32, [2u64], None, local_device, None)
            .unwrap();
        let source_array_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]));
        let source_array = Array::from_addressable_buffers_with_sharding(
            &engine,
            source_array_type,
            sharding.clone(),
            mesh.clone(),
            vec![local_source_buffer],
        )
        .unwrap();

        let copied_array = device_put(&engine, source_array, DevicePutOptions::defaults()).unwrap();
        let expected_visualization =
            format!("┌─────┬─────┐\n│{:^5}│{:^5}│\n└─────┴─────┘", local_device_id, remote_device_id);

        assert_eq!(copied_array.addressable_shards().count(), 1);
        assert_eq!(copied_array.sharding().visualize().unwrap().render(false), expected_visualization);
        let copied_shard_bytes = copied_array
            .device_shard(local_device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        assert_eq!(copied_shard_bytes.as_slice().len() % size_of::<f32>(), 0);
        assert_eq!(values_from_bytes::<f32>(copied_shard_bytes.as_slice()), vec![0.0, 1.0]);
        assert!(copied_array.device_shard(remote_device_id).unwrap().buffer().is_none());
    }

    #[test]
    fn test_device_put_rejects_mismatched_src_for_array_leaf() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() }))
            .unwrap();
        let engine = Session::new(&client).domain();
        let client_device = client.addressable_devices().unwrap().remove(0);
        let source_device = Device::from_pjrt(client_device).unwrap();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![source_device],
        )
        .unwrap();
        let source_sharding = Sharding::replicated(source_mesh.logical_mesh().clone(), 1);
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(2)]))
            .with_sharding(source_sharding.clone())
            .unwrap();
        let source_array = Array::from_host_buffer(
            &engine,
            source_type,
            source_mesh.clone(),
            values_to_bytes::<f32>(&[0.0, 1.0]).as_slice(),
        )
        .unwrap();
        let expected_src = DevicePutTarget::device(Device::new(source_device.id() + 1, 0)).resolve(1).unwrap();
        let actual_src = (source_mesh, source_sharding);

        assert!(matches!(
            device_put(
                &engine,
                source_array,
                DevicePutOptions::new(
                    Option::<DevicePutTarget>::None,
                    Some(DevicePutTarget::device(Device::new(source_device.id() + 1, 0))),
                    Option::<bool>::None,
                    Option::<Option<bool>>::None,
                ),
            ),
            Err(Error::Array(error)) if matches!(error.as_ref(),
                ArrayError::SourcePlacementMismatch {
                    expected_mesh: reported_expected_mesh,
                    expected_sharding: reported_expected_sharding,
                    actual_mesh: reported_actual_mesh,
                    actual_sharding: reported_actual_sharding,
                } if reported_expected_mesh == &expected_src.0
                    && reported_expected_sharding == &expected_src.1
                    && reported_actual_mesh == &actual_src.0
                    && reported_actual_sharding == &actual_src.1,
            ),
        ));
    }
}
