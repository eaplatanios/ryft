//! Placement of whole trees of host values and arrays on devices through [`Domain::device_put`], following
//! `jax.device_put`. [`DevicePutOptions`] holds the tree prefixes of the per-leaf options, and [`DevicePutLeaf`]
//! places one leaf, which is either an [`Array`] or a host value that is uploaded as one dense array.

use half::{bf16, f16};
use ryft_core::{
    Array as CoreArray, ArrayType, DataType, Device, Dimension as CoreDimension, Parameter, Parameterized,
    ParameterizedFamily, Shape, Typed,
};

use crate::arrays::placement::DevicePutTarget;
use crate::arrays::{Array, ArrayError};
use crate::{Domain, Error, FromPjrt};

// TODO(eaplatanios): Review this module.

/// Options of [`Domain::device_put`].
///
/// Each field follows JAX's tree prefix semantics, so when a field is present, its structure is broadcast over the
/// input tree and applied leafwise.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DevicePutOptions<
    DeviceTarget = DevicePutTarget,
    SourceTarget = DevicePutTarget,
    Donate = bool,
    MayAlias = Option<bool>,
> {
    /// Destination placement tree prefix. When absent, host leaves are committed to the default local device and
    /// [`Array`] leaves preserve their current placement.
    device: Option<DeviceTarget>,

    /// Source placement tree prefix. This is validated for [`Array`] leaves and ignored for host leaves, which do not
    /// carry runtime placement metadata before upload.
    source: Option<SourceTarget>,

    /// Donation tree prefix. A donated [`Array`] leaf lets a compiled reshard reuse its buffers for the result. The
    /// donation only takes effect when the leaf is the only handle of its buffers, and host leaves ignore it.
    donate: Option<Donate>,

    /// Aliasing tree prefix. With `Some(false)`, a resulting array shares no buffer with any other array, which
    /// copies the buffers that it would otherwise share.
    may_alias: Option<MayAlias>,
}

impl<DeviceTarget, SourceTarget, Donate, MayAlias> DevicePutOptions<DeviceTarget, SourceTarget, Donate, MayAlias> {
    /// Creates a new [`DevicePutOptions`].
    #[inline]
    pub fn new(
        device: Option<DeviceTarget>,
        source: Option<SourceTarget>,
        donate: Option<Donate>,
        may_alias: Option<MayAlias>,
    ) -> Self {
        Self { device, source, donate, may_alias }
    }

    /// Creates a new [`DevicePutOptions`] with all fields unset.
    #[inline]
    pub fn empty() -> Self {
        Self::new(None, None, None, None)
    }

    /// Returns the destination placement tree prefix, if one was provided.
    #[inline]
    pub fn device(&self) -> Option<&DeviceTarget> {
        self.device.as_ref()
    }

    /// Returns the source placement tree prefix, if one was provided.
    #[inline]
    pub fn source(&self) -> Option<&SourceTarget> {
        self.source.as_ref()
    }

    /// Returns the donation tree prefix, if one was provided.
    #[inline]
    pub fn donate(&self) -> Option<&Donate> {
        self.donate.as_ref()
    }

    /// Returns the aliasing tree prefix, if one was provided.
    #[inline]
    pub fn may_alias(&self) -> Option<&MayAlias> {
        self.may_alias.as_ref()
    }

    /// Returns the destination, source, donation, and aliasing tree prefixes of these options.
    pub(crate) fn into_parts(self) -> (Option<DeviceTarget>, Option<SourceTarget>, Option<Donate>, Option<MayAlias>) {
        (self.device, self.source, self.donate, self.may_alias)
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
    /// Creates default [`Domain::device_put`] options without requiring generic type inference.
    #[inline]
    pub fn defaults() -> Self {
        Self::empty()
    }
}

/// Leaf types accepted by [`Domain::device_put`].
///
/// A [`DevicePutLeaf`] consumes one input leaf and materializes one runtime [`Array`] leaf. `ryft` currently provides
/// implementations for runtime [`Array`] leaves, primitive scalar host values, and host arrays of the reference backend
/// ([`ryft_core::Array`]), which are uploaded as dense arrays whatever their host layout is.
pub trait DevicePutLeaf<'c>: Parameter {
    /// Converts `self` into one runtime [`Array`] using the provided leafwise placement options.
    ///
    /// # Parameters
    ///
    ///   - `domain`: [`Domain`] that provides the PJRT client used to materialize any needed destination buffers and
    ///     the cache of any compiled reshard executables.
    ///   - `device`: Destination placement for this leaf, if one was specified.
    ///   - `source`: Source placement for this leaf, if one was specified.
    ///   - `donate`: Whether a compiled reshard may reuse the buffers of this leaf for the result.
    ///   - `may_alias`: Whether the result may share buffers with other arrays. `Some(false)` forbids it.
    fn device_put_leaf(
        self,
        domain: &Domain<'c>,
        device: Option<DevicePutTarget>,
        source: Option<DevicePutTarget>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Array<'c>, Error>;
}

impl<'c, T: DenseHostDevicePutLeaf + Parameter> DevicePutLeaf<'c> for T {
    fn device_put_leaf(
        self,
        domain: &Domain<'c>,
        device: Option<DevicePutTarget>,
        source: Option<DevicePutTarget>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Array<'c>, Error> {
        // A host value has no placement to validate and no device buffers to donate or share.
        let _ = (source, donate, may_alias);
        let client = domain.client();
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
        Array::from_host_buffer_with_sharding(domain, r#type, sharding, mesh, bytes.as_slice())
    }
}

impl<'c> DevicePutLeaf<'c> for Array<'c> {
    fn device_put_leaf(
        self,
        domain: &Domain<'c>,
        device: Option<DevicePutTarget>,
        source: Option<DevicePutTarget>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Array<'c>, Error> {
        let current_mesh = self.mesh().clone();
        let current_sharding = self.sharding().clone();
        if let Some(source) = source {
            let (expected_mesh, expected_sharding) = source.resolve(current_sharding.rank())?;
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
        let output = if target_mesh == current_mesh && target_sharding == current_sharding {
            self
        } else {
            let target = DevicePutTarget::placement(target_mesh, target_sharding)?;
            if donate { self.into_placement(domain, target)? } else { self.to_placement(domain, target)? }
        };

        // The result is this leaf when the placement does not change, and it otherwise shares the buffers of the
        // shards that stay on their devices. Both are only observable through another handle of those buffers.
        if may_alias == Some(false) && !output.has_unique_shard_buffers() {
            output.with_copied_shard_buffers(domain)
        } else {
            Ok(output)
        }
    }
}

impl<'c> Domain<'c> {
    /// Places `x` on devices. This is the `ryft` analogue of JAX's
    /// [`jax.device_put`](https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html).
    ///
    /// The input `x` may be one supported leaf or a [`Parameterized`] tree of supported leaves (i.e., of
    /// [`DevicePutLeaf`]s). Any provided `device`, `source`, `donate`, and `may_alias` options follow tree prefix
    /// broadcasting semantics over `x`.
    ///
    /// Host leaves are committed to the default local device when `options.device()` is absent. Existing
    /// [`Array`] leaves preserve their current placement when `options.device()` is absent.
    ///
    /// # Parameters
    ///
    ///   - `x`: Leaf or tree of leaves to place.
    ///   - `options`: Destination, expected source, donation, and aliasing options for the leaves of `x`.
    pub fn device_put<
        P: DevicePutLeaf<'c>,
        Input: Parameterized<P>,
        DeviceTarget: Parameterized<DevicePutTarget>,
        SourceTarget: Parameterized<DevicePutTarget>,
        Donate: Parameterized<bool>,
        MayAlias: Parameterized<Option<bool>>,
    >(
        &self,
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
        let (device, source, donate, may_alias) = options.into_parts();

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
        let source_values = match source {
            Some(source) => Input::To::<DevicePutTarget>::from_broadcasted_named_parameters(
                structure.clone(),
                source.into_named_parameters(),
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
        let mut source_values = source_values.into_iter();
        let mut donate_values = donate_values.into_iter();
        let mut may_alias_values = may_alias_values.into_iter();
        for parameter in x.into_parameters() {
            output_parameters.push(parameter.device_put_leaf(
                self,
                device_values.next().unwrap(),
                source_values.next().unwrap(),
                donate_values.next().unwrap(),
                may_alias_values.next().unwrap(),
            )?);
        }
        Input::To::<Array<'c>>::from_parameters(structure, output_parameters).map_err(Into::into)
    }
}

/// Host value that [`Domain::device_put`] uploads as one dense array.
trait DenseHostDevicePutLeaf {
    /// Returns the shape, the data type, and the row-major bytes in native byte order of this value.
    fn into_dense_host_array(self) -> (Vec<usize>, DataType, Vec<u8>);
}

/// Element of a dense host value that [`Domain::device_put`] uploads.
trait DenseHostElement {
    /// [`DataType`] of the uploaded elements.
    const DATA_TYPE: DataType;

    /// Appends the bytes of this element in native byte order to `bytes`.
    fn append_ne_bytes(&self, bytes: &mut Vec<u8>);
}

macro_rules! impl_dense_host_element {
    // Implements `DenseHostElement` for the primitive type `$ty`, whose native-endian bytes encode `$data_type`.
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
    // Implements `DenseHostDevicePutLeaf` for the scalar `DenseHostElement` type `$ty` as a rank-0 host array.
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

impl DenseHostDevicePutLeaf for CoreArray {
    fn into_dense_host_array(self) -> (Vec<usize>, DataType, Vec<u8>) {
        // A host array holds materialized storage, which only a type with static extents can describe.
        let shape = self.r#type().static_shape().unwrap();
        (shape.as_slice().to_vec(), self.r#type().data_type(), self.logical_bytes())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension, Typed};
    use ryft_pjrt::BufferType;

    use crate::Session;
    use crate::domains::tests::domain_mesh;
    use crate::tests::{cpu_client, values_from_bytes, values_to_bytes};

    use super::*;

    // Returns the `f32` values that the devices of `array` hold, in the device order of its mesh.
    fn shard_values(array: &Array<'_>) -> Vec<Vec<f32>> {
        array
            .shards()
            .iter()
            .map(|shard| values_from_bytes::<f32>(&shard.copy_to_host().unwrap()))
            .collect()
    }

    #[test]
    fn test_device_put_options_new() {
        let target = DevicePutTarget::Device(Device::new(3, 0));
        let options =
            DevicePutOptions::new(Some(target.clone()), Option::<DevicePutTarget>::None, Some(true), Some(Some(false)));
        assert_eq!(options.device(), Some(&target));
        assert_eq!(options.source(), None);
        assert_eq!(options.donate(), Some(&true));
        assert_eq!(options.may_alias(), Some(&Some(false)));
        assert_eq!(options.into_parts(), (Some(target), None, Some(true), Some(Some(false))));
    }

    #[test]
    fn test_device_put_options_empty() {
        let options = DevicePutOptions::<DevicePutTarget, DevicePutTarget, bool, Option<bool>>::empty();
        assert_eq!(options.into_parts(), (None, None, None, None));
        assert_eq!(DevicePutOptions::defaults(), DevicePutOptions::empty());
        assert_eq!(DevicePutOptions::default(), DevicePutOptions::defaults());
    }

    #[test]
    fn test_domain_device_put_uploads_host_leaves() {
        let client = cpu_client(2);
        let domain = Session::new(&client).domain();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[1]).unwrap();

        // Without a destination, a host leaf is committed to the first local device.
        let scalar = domain.device_put(2.5f32, DevicePutOptions::defaults()).unwrap();
        let flag = domain.device_put(true, DevicePutOptions::defaults()).unwrap();
        assert_eq!(scalar.r#type().into_owned().with_sharding(None).unwrap(), ArrayType::scalar(DataType::F32));
        assert_eq!(scalar.mesh().devices(), [Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap()]);
        assert_eq!(values_from_bytes::<f32>(&scalar.copy_to_host().unwrap()), vec![2.5]);
        assert_eq!(flag.data_type(), DataType::Boolean);
        assert_eq!(flag.copy_to_host().unwrap(), vec![1u8]);

        // A destination device places the leaf on that device.
        let placed = domain
            .device_put(
                bf16::from_f32(1.5),
                DevicePutOptions::new(
                    Some(DevicePutTarget::Device(device)),
                    Option::<DevicePutTarget>::None,
                    Option::<bool>::None,
                    Option::<Option<bool>>::None,
                ),
            )
            .unwrap();
        assert_eq!(placed.mesh().devices(), [device]);
        assert_eq!(placed.copy_to_host().unwrap(), bf16::from_f32(1.5).to_bits().to_ne_bytes().to_vec());
    }

    #[test]
    fn test_domain_device_put_uploads_reference_backend_arrays() {
        let client = cpu_client(1);
        let domain = Session::new(&client).domain();
        let matrix = CoreArray::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap();
        let array = domain.device_put(matrix, DevicePutOptions::defaults()).unwrap();
        assert_eq!(
            array.r#type().into_owned().with_sharding(None).unwrap(),
            ArrayType::new_static(DataType::F32, [2, 3]),
        );
        assert_eq!(values_from_bytes::<f32>(&array.copy_to_host().unwrap()), vec![1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn test_domain_device_put_broadcasts_root_placement_over_array_tuple() {
        let client = cpu_client(2);
        let domain = Session::new(&client).domain();
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
            &domain,
            first_source_type,
            source_mesh.clone(),
            values_to_bytes::<f32>(&[0.0, 1.0, 2.0, 3.0]).as_slice(),
        )
        .unwrap();
        let second_source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(4)]))
            .with_sharding(source_sharding)
            .unwrap();
        let second_source_array = Array::from_host_buffer(
            &domain,
            second_source_type,
            source_mesh,
            values_to_bytes::<f32>(&[10.0, 11.0, 12.0, 13.0]).as_slice(),
        )
        .unwrap();

        let target_mesh = domain_mesh(&client, "x", 2);
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let moved_arrays = domain
            .device_put(
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
        assert_eq!(shard_values(&moved_arrays.0), vec![vec![0.0, 1.0], vec![2.0, 3.0]]);
        assert_eq!(shard_values(&moved_arrays.1), vec![vec![10.0, 11.0], vec![12.0, 13.0]]);
    }

    #[test]
    fn test_domain_device_put_donation_and_aliasing() {
        let client = cpu_client(2);
        let domain = Session::new(&client).domain();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![devices[0]],
        )
        .unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let source = Array::from_host_buffer(
            &domain,
            ArrayType::new_static(DataType::F32, [4]),
            source_mesh,
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        let shares_source_buffer = |array: &Array<'_>| {
            array
                .shards()
                .iter()
                .map(|shard| Arc::ptr_eq(shard.buffer().unwrap(), source.shards()[0].buffer().unwrap()))
                .collect::<Vec<_>>()
        };
        let options = |device: Option<DevicePutTarget>, donate: bool, may_alias: Option<bool>| {
            DevicePutOptions::new(device, Option::<DevicePutTarget>::None, Some(donate), Some(may_alias))
        };

        // Without a new placement, the result is the leaf itself unless aliasing is forbidden.
        let unchanged = domain.device_put(source.clone(), options(None, false, None)).unwrap();
        assert_eq!(shares_source_buffer(&unchanged), vec![true]);
        let copied = domain.device_put(source.clone(), options(None, false, Some(false))).unwrap();
        assert_eq!(copied.r#type(), source.r#type());
        assert_eq!(copied.mesh(), source.mesh());
        assert_eq!(shares_source_buffer(&copied), vec![false]);
        assert_eq!(shard_values(&copied), vec![vec![1., 2., 3., 4.]]);

        // A shard that stays on its device shares its buffer unless aliasing is forbidden.
        let target = DevicePutTarget::placement(mesh.clone(), replicated).unwrap();
        let placed = domain.device_put(source.clone(), options(Some(target.clone()), false, None)).unwrap();
        assert_eq!(shares_source_buffer(&placed), vec![true, false]);
        assert_eq!(shard_values(&placed), vec![vec![1., 2., 3., 4.]; 2]);
        let placed = domain.device_put(source.clone(), options(Some(target), false, Some(false))).unwrap();
        assert_eq!(shares_source_buffer(&placed), vec![false, false]);
        assert_eq!(shard_values(&placed), vec![vec![1., 2., 3., 4.]; 2]);

        // A donated leaf is consumed. This one shares its buffers with `source`, so the donation does not take effect
        // and `source` stays readable.
        let target = DevicePutTarget::placement(mesh, sharded.clone()).unwrap();
        let donated = domain.device_put(source.clone(), options(Some(target), true, None)).unwrap();
        assert_eq!(donated.sharding(), &sharded);
        assert_eq!(shard_values(&donated), vec![vec![1., 2.], vec![3., 4.]]);
        assert_eq!(shard_values(&source), vec![vec![1., 2., 3., 4.]]);
    }

    #[test]
    fn test_domain_device_put_preserves_partially_addressable_array_when_device_is_absent() {
        let client = cpu_client(1);
        let domain = Session::new(&client).domain();
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
            &domain,
            source_array_type,
            sharding.clone(),
            mesh.clone(),
            vec![local_source_buffer],
        )
        .unwrap();

        let copied_array = domain.device_put(source_array, DevicePutOptions::defaults()).unwrap();
        let expected_visualization =
            format!("┌─────┬─────┐\n│{:^5}│{:^5}│\n└─────┴─────┘", local_device_id, remote_device_id);

        assert_eq!(copied_array.addressable_shards().count(), 1);
        assert_eq!(copied_array.sharding().visualize().unwrap().render(false), expected_visualization);
        let copied_shard_bytes = copied_array.device_shard(local_device_id).unwrap().copy_to_host().unwrap();
        assert_eq!(values_from_bytes::<f32>(copied_shard_bytes.as_slice()), vec![0.0, 1.0]);
        assert!(copied_array.device_shard(remote_device_id).unwrap().buffer().is_none());
    }

    #[test]
    fn test_domain_device_put_rejects_mismatched_source_for_array_leaf() {
        let client = cpu_client(1);
        let domain = Session::new(&client).domain();
        let source_mesh = domain_mesh(&client, "x", 1);
        let source_device = source_mesh.devices()[0];
        let source_sharding = Sharding::replicated(source_mesh.logical_mesh().clone(), 1);
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(2)]))
            .with_sharding(source_sharding.clone())
            .unwrap();
        let source_array = Array::from_host_buffer(
            &domain,
            source_type,
            source_mesh.clone(),
            values_to_bytes::<f32>(&[0.0, 1.0]).as_slice(),
        )
        .unwrap();
        let expected_source = DevicePutTarget::device(Device::new(source_device.id() + 1, 0)).resolve(1).unwrap();
        let actual_source = (source_mesh, source_sharding);

        let error = domain
            .device_put(
                source_array,
                DevicePutOptions::new(
                    Option::<DevicePutTarget>::None,
                    Some(DevicePutTarget::device(Device::new(source_device.id() + 1, 0))),
                    Option::<bool>::None,
                    Option::<Option<bool>>::None,
                ),
            )
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            format!(
                "`device_put` source mesh `{:?}` with sharding `{:?}` does not match the array's current mesh `{:?}` \
                 with sharding `{:?}`",
                expected_source.0, expected_source.1, actual_source.0, actual_source.1,
            ),
        );
        assert!(matches!(
            &error,
            Error::Array(error) if matches!(error.as_ref(),
                ArrayError::SourcePlacementMismatch {
                    expected_mesh: reported_expected_mesh,
                    expected_sharding: reported_expected_sharding,
                    actual_mesh: reported_actual_mesh,
                    actual_sharding: reported_actual_sharding,
                } if reported_expected_mesh == &expected_source.0
                    && reported_expected_sharding == &expected_source.1
                    && reported_actual_mesh == &actual_source.0
                    && reported_actual_sharding == &actual_source.1,
            ),
        ));
    }
}
