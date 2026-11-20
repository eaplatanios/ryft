use half::{bf16, f16};

use ryft_core::{
    Array as CoreArray, ArrayType, DataType, Device, DeviceMesh, Dimension, LogicalMesh, MeshAxis, MeshAxisType,
    Parameter, Parameterized, ParameterizedFamily, Shape, Sharding, Typed, check_sharding,
};
use ryft_macros::Parameter;

use crate::arrays::{Array, ArrayError};
use crate::{Error, FromPjrt, Session};

/// Placement of an [`Array`] on devices. It specifies a destination for [`Session::place`],
/// [`Array::to_placement`], and [`Array::into_placement`], or an expected source in [`PlacementOptions`].
/// [`Self::Device`] uses replicated sharding on a single device. [`Self::Mesh`] pairs a concrete mesh with a
/// sharding. [`Self::Sharding`] uses the session's [default mesh](Session::device_mesh), whose logical topology
/// must match the sharding. Destination and source trees in [`PlacementOptions`] can use `Some(placement)` and
/// `None` together. A `None` leaf omits the option for its branch.
#[derive(Clone, Debug, PartialEq, Eq, Parameter)]
pub enum ArrayPlacement {
    /// Commit the value to one concrete device.
    Device(Device),

    /// Commit the value to the provided mesh and sharding.
    Mesh {
        /// Concrete destination mesh describing the device topology.
        mesh: DeviceMesh,

        /// Sharding to apply over `mesh`.
        sharding: Sharding,
    },

    /// Use this sharding on the session's configured default mesh.
    Sharding(Sharding),
}

impl ArrayPlacement {
    /// Creates a single-device placement.
    #[inline]
    pub fn device(device: Device) -> Self {
        Self::Device(device)
    }

    /// Creates a placement on an explicit mesh and sharding.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if `sharding` refers to a different logical mesh than `mesh`.
    #[inline]
    pub fn mesh(mesh: DeviceMesh, sharding: Sharding) -> Result<Self, Error> {
        check_sharding!(&mesh, &sharding);
        Ok(Self::Mesh { mesh, sharding })
    }

    /// Resolves this placement into the concrete [`DeviceMesh`] and [`Sharding`] of a value of rank `rank`.
    /// A [`Self::Device`] placement resolves to a mesh with one auto axis named `device` of size 1 and to the
    /// replicated sharding over that mesh.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::MissingMesh`](crate::SessionError::MissingMesh) when [`Self::Sharding`] needs a
    /// default mesh and the session has none, or [`ShardingError::MeshMismatch`](ryft_core::ShardingError::MeshMismatch)
    /// when the sharding and the concrete mesh have different logical topologies.
    pub(crate) fn resolve(self, session: &Session, rank: usize) -> Result<(DeviceMesh, Sharding), Error> {
        let (mesh, sharding) = match self {
            Self::Device(device) => {
                let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("device", 1, MeshAxisType::Auto)?])?;
                let mesh = DeviceMesh::new(logical_mesh, vec![device])?;
                let sharding = Sharding::replicated(mesh.logical_mesh().clone(), rank);
                return Ok((mesh, sharding));
            }
            Self::Mesh { mesh, sharding } => (mesh, sharding),
            Self::Sharding(sharding) => (session.device_mesh()?.clone(), sharding),
        };
        check_sharding!(&mesh, &sharding);
        Ok((mesh, sharding))
    }
}

impl From<Device> for ArrayPlacement {
    #[inline]
    fn from(value: Device) -> Self {
        Self::Device(value)
    }
}

// TODO(eaplatanios): Review from here onwards.

/// Options of [`Session::place`].
///
/// Each field follows JAX's tree prefix semantics, so when a field is present, its structure is broadcast over the
/// input tree and applied leafwise. Destination and source prefixes can contain [`ArrayPlacement`] leaves or
/// `Option<ArrayPlacement>` leaves. A `None` leaf omits the corresponding option for that branch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlacementOptions<Target = ArrayPlacement, Source = ArrayPlacement, Donate = bool, MayAlias = Option<bool>> {
    /// Destination placement tree prefix. When absent, host leaves are committed to the default local device and
    /// [`Array`] leaves preserve their current placement.
    target: Option<Target>,

    /// Source placement tree prefix. This is validated for [`Array`] leaves and ignored for host leaves, which do not
    /// carry runtime placement metadata before upload.
    source: Option<Source>,

    /// Donation tree prefix. A donated [`Array`] leaf lets a compiled reshard reuse its buffers for the result. The
    /// donation only takes effect when the leaf is the only handle of its buffers, and host leaves ignore it.
    donate: Option<Donate>,

    /// Aliasing tree prefix. With `Some(false)`, a resulting array shares no buffer with any other array, which
    /// copies the buffers that it would otherwise share.
    may_alias: Option<MayAlias>,
}

impl<Target, Source, Donate, MayAlias> PlacementOptions<Target, Source, Donate, MayAlias> {
    /// Creates a new [`PlacementOptions`].
    #[inline]
    pub fn new(
        target: Option<Target>,
        source: Option<Source>,
        donate: Option<Donate>,
        may_alias: Option<MayAlias>,
    ) -> Self {
        Self { target, source, donate, may_alias }
    }

    /// Creates a new [`PlacementOptions`] with all fields unset.
    #[inline]
    pub fn empty() -> Self {
        Self::new(None, None, None, None)
    }

    /// Returns the destination placement tree prefix, if one was provided.
    #[inline]
    pub fn target(&self) -> Option<&Target> {
        self.target.as_ref()
    }

    /// Returns the source placement tree prefix, if one was provided.
    #[inline]
    pub fn source(&self) -> Option<&Source> {
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
    pub(crate) fn into_parts(self) -> (Option<Target>, Option<Source>, Option<Donate>, Option<MayAlias>) {
        (self.target, self.source, self.donate, self.may_alias)
    }
}

impl<Target, Source, Donate, MayAlias> Default for PlacementOptions<Target, Source, Donate, MayAlias> {
    fn default() -> Self {
        Self::empty()
    }
}

impl PlacementOptions<ArrayPlacement, ArrayPlacement, bool, Option<bool>> {
    /// Creates default [`Session::place`] options without requiring generic type inference.
    #[inline]
    pub fn defaults() -> Self {
        Self::empty()
    }
}

/// Leaf types accepted by [`Session::place`].
///
/// A [`Placeable`] consumes one input leaf and materializes one runtime [`Array`] leaf. `ryft` currently provides
/// implementations for runtime [`Array`] leaves, primitive scalar host values, and host arrays of the reference backend
/// ([`ryft_core::Array`]), which are uploaded as dense arrays whatever their host layout is.
pub trait Placeable: Parameter {
    /// Converts `self` into one runtime [`Array`] using the provided leafwise placement options.
    ///
    /// # Parameters
    ///
    ///   - `session`: [`Session`] that provides the PJRT client used to materialize any needed destination buffers and
    ///     the cache of any compiled reshard executables.
    ///   - `target`: Destination placement for this leaf, if one was specified.
    ///   - `source`: Source placement for this leaf, if one was specified.
    ///   - `donate`: Whether a compiled reshard may reuse the buffers of this leaf for the result.
    ///   - `may_alias`: Whether the result may share buffers with other arrays. `Some(false)` forbids it.
    fn place(
        self,
        session: &Session,
        target: Option<ArrayPlacement>,
        source: Option<ArrayPlacement>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Array, Error>;
}

impl<T: IntoDenseHostArray + Parameter> Placeable for T {
    fn place(
        self,
        session: &Session,
        target: Option<ArrayPlacement>,
        source: Option<ArrayPlacement>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Array, Error> {
        // A host value has no placement to validate and no device buffers to donate or share.
        let _ = (source, donate, may_alias);
        let client = session.client();
        let (shape, element_type, bytes) = self.into_dense_host_array();
        let (mesh, sharding) = match target {
            Some(target) => target.resolve(session, shape.len())?,
            None => {
                let device = client
                    .addressable_devices()?
                    .into_iter()
                    .next()
                    .ok_or_else(|| Error::from(ArrayError::MissingDefaultDevice))?;
                ArrayPlacement::device(Device::from_pjrt(device)?).resolve(session, shape.len())?
            }
        };
        let r#type = ArrayType::new(element_type, Shape::new(shape.iter().copied().map(Dimension::Static).collect()));
        Array::from_host_buffer(session, r#type, mesh, Some(sharding), bytes.as_slice())
    }
}

impl Placeable for Array {
    fn place(
        self,
        session: &Session,
        target: Option<ArrayPlacement>,
        source: Option<ArrayPlacement>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Array, Error> {
        if let Some(source) = source {
            let (expected_mesh, expected_sharding) = source.resolve(session, self.sharding().rank())?;
            if &expected_mesh != self.mesh() || &expected_sharding != self.sharding() {
                return Err(Error::from(ArrayError::SourcePlacementMismatch {
                    expected_mesh,
                    expected_sharding,
                    actual_mesh: self.mesh().clone(),
                    actual_sharding: self.sharding().clone(),
                }));
            }
        }

        let output = if let Some(target) = target {
            let (mesh, sharding) = target.resolve(session, self.sharding().rank())?;
            if &mesh == self.mesh() && &sharding == self.sharding() {
                self
            } else {
                let target = ArrayPlacement::Mesh { mesh, sharding };
                if donate { self.into_placement(session, target)? } else { self.to_placement(session, target)? }
            }
        } else {
            self
        };

        // The result is this leaf when the placement does not change, and it otherwise shares the buffers of the
        // shards that stay on their devices. Both are only observable through another handle of those buffers.
        if may_alias == Some(false) && !output.has_unique_shard_buffers() {
            output.with_copied_shard_buffers(session)
        } else {
            Ok(output)
        }
    }
}

impl Session {
    /// Places `x` on devices. This is the `ryft` analogue of JAX's
    /// [`jax.device_put`](https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html).
    ///
    /// The input `x` may be one supported leaf or a [`Parameterized`] tree of supported leaves (i.e., of
    /// [`Placeable`]s). Any provided `target`, `source`, `donate`, and `may_alias` options follow tree prefix
    /// broadcasting semantics over `x`. The destination and source prefixes accept optional placement leaves,
    /// so a single tree can mix explicit placements with `None` for branches where the option is omitted.
    ///
    /// With no destination, a host leaf goes to the first addressable device and an existing [`Array`] leaf keeps
    /// its current placement. Use [`ArrayPlacement::Sharding`] to place a leaf on the session's default mesh.
    ///
    /// # Parameters
    ///
    ///   - `x`: Leaf or tree of leaves to place.
    ///   - `options`: Destination, expected source, donation, and aliasing options for the leaves of `x`.
    pub fn place<
        P: Placeable,
        Input: Parameterized<P>,
        TargetPlacement: Parameter + Into<Option<ArrayPlacement>>,
        Target: Parameterized<TargetPlacement>,
        SourcePlacement: Parameter + Into<Option<ArrayPlacement>>,
        Source: Parameterized<SourcePlacement>,
        Donate: Parameterized<bool>,
        MayAlias: Parameterized<Option<bool>>,
    >(
        &self,
        x: Input,
        options: PlacementOptions<Target, Source, Donate, MayAlias>,
    ) -> Result<<Input as Parameterized<P>>::To<Array>, Error>
    where
        <Input as Parameterized<P>>::Family: ParameterizedFamily<Array>
            + ParameterizedFamily<Option<ArrayPlacement>>
            + ParameterizedFamily<bool>
            + ParameterizedFamily<Option<bool>>,
    {
        let structure = x.parameter_structure();
        let leaf_count = structure.parameter_count();
        let (target, source, donate, may_alias) = options.into_parts();

        let target_values = match target {
            Some(target) => Input::To::<Option<ArrayPlacement>>::from_broadcasted_named_parameters(
                structure.clone(),
                target.into_named_parameters().map(|(path, placement)| (path, placement.into())),
            )?
            .into_parameters()
            .collect::<Vec<_>>(),
            None => vec![None; leaf_count],
        };
        let source_values = match source {
            Some(source) => Input::To::<Option<ArrayPlacement>>::from_broadcasted_named_parameters(
                structure.clone(),
                source.into_named_parameters().map(|(path, placement)| (path, placement.into())),
            )?
            .into_parameters()
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
        let mut target_values = target_values.into_iter();
        let mut source_values = source_values.into_iter();
        let mut donate_values = donate_values.into_iter();
        let mut may_alias_values = may_alias_values.into_iter();
        for parameter in x.into_parameters() {
            output_parameters.push(parameter.place(
                self,
                target_values.next().unwrap(),
                source_values.next().unwrap(),
                donate_values.next().unwrap(),
                may_alias_values.next().unwrap(),
            )?);
        }
        Input::To::<Array>::from_parameters(structure, output_parameters).map_err(Into::into)
    }
}

/// Host value that [`Session::place`] uploads as one dense array.
trait IntoDenseHostArray {
    /// Returns the shape, the data type, and the row-major bytes in native byte order of this value.
    fn into_dense_host_array(self) -> (Vec<usize>, DataType, Vec<u8>);
}

macro_rules! impl_into_dense_host_array {
    // Implements `IntoDenseHostArray` for the scalar `DenseHostElement` type `$ty` as a rank-0 host array.
    ($ty:ty) => {
        impl IntoDenseHostArray for $ty {
            #[inline]
            fn into_dense_host_array(self) -> (Vec<usize>, DataType, Vec<u8>) {
                let mut bytes = Vec::with_capacity(size_of::<$ty>());
                self.append_ne_bytes(&mut bytes);
                (Vec::new(), <$ty as DenseHostElement>::DATA_TYPE, bytes)
            }
        }
    };
}

impl_into_dense_host_array!(bool);
impl_into_dense_host_array!(i8);
impl_into_dense_host_array!(i16);
impl_into_dense_host_array!(i32);
impl_into_dense_host_array!(i64);
impl_into_dense_host_array!(u8);
impl_into_dense_host_array!(u16);
impl_into_dense_host_array!(u32);
impl_into_dense_host_array!(u64);
impl_into_dense_host_array!(bf16);
impl_into_dense_host_array!(f16);
impl_into_dense_host_array!(f32);
impl_into_dense_host_array!(f64);

impl IntoDenseHostArray for CoreArray {
    #[inline]
    fn into_dense_host_array(self) -> (Vec<usize>, DataType, Vec<u8>) {
        // A host array holds materialized storage, which only a type with static extents can describe.
        let shape = self.r#type().static_shape().unwrap();
        (shape.as_slice().to_vec(), self.r#type().data_type(), self.logical_bytes())
    }
}

/// Element of a dense host value that [`Session::place`] uploads.
trait DenseHostElement {
    /// [`DataType`] of the uploaded elements.
    const DATA_TYPE: DataType;

    /// Appends the bytes of this element in native byte order to `bytes`.
    fn append_ne_bytes(&self, bytes: &mut Vec<u8>);
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

macro_rules! impl_dense_host_element {
    // Implements `DenseHostElement` for the primitive type `$ty`, whose native-endian bytes encode `$data_type`.
    ($ty:ty, $data_type:expr) => {
        impl DenseHostElement for $ty {
            const DATA_TYPE: DataType = $data_type;

            #[inline]
            fn append_ne_bytes(&self, bytes: &mut Vec<u8>) {
                bytes.extend_from_slice(&self.to_ne_bytes());
            }
        }
    };
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        DeviceMesh, Error as CoreError, LogicalMesh, Memory, MeshAxis, MeshAxisType, Sharding, ShardingDimension,
        ShardingError, Typed,
    };
    use ryft_pjrt::BufferType;

    use crate::SessionError;
    use crate::sessions::tests::domain_mesh;
    use crate::tests::{cpu_client, device_mesh_2x2, values_from_bytes, values_to_bytes};

    use super::*;

    // Returns the `f32` values that the devices of `array` hold, in the device order of its mesh.
    fn shard_values(array: &Array) -> Vec<Vec<f32>> {
        array
            .shards()
            .iter()
            .map(|shard| values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()))
            .collect()
    }

    #[test]
    fn test_array_placement_device() {
        let device = Device::new(3, 0);
        assert_eq!(ArrayPlacement::device(device), ArrayPlacement::Device(device));
        assert_eq!(ArrayPlacement::from(device), ArrayPlacement::Device(device));
    }

    #[test]
    fn test_array_placement_mesh() {
        let mesh = device_mesh_2x2();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        assert_eq!(
            ArrayPlacement::mesh(mesh.clone(), sharding.clone()).unwrap(),
            ArrayPlacement::Mesh { mesh: mesh.clone(), sharding },
        );

        // A sharding over another logical mesh is rejected.
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("z", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let error = ArrayPlacement::mesh(mesh.clone(), Sharding::replicated(other_mesh.clone(), 1)).unwrap_err();
        assert_eq!(
            error.to_string(),
            format!("mesh mismatch; expected `{:?}` but got `{other_mesh:?}`", mesh.logical_mesh()),
        );
    }

    #[test]
    fn test_array_placement_resolve() {
        let session = Session::builder().client(cpu_client(1)).build().unwrap();

        // A device target resolves to a replicated placement on a mesh with one auto axis of size 1.
        let device = Device::new(3, 0);
        let device_mesh = LogicalMesh::new(vec![MeshAxis::new("device", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        assert_eq!(
            ArrayPlacement::Device(device).resolve(&session, 2).unwrap(),
            (DeviceMesh::new(device_mesh.clone(), vec![device]).unwrap(), Sharding::replicated(device_mesh, 2)),
        );

        // A placement target resolves to its own mesh and sharding.
        let mesh = device_mesh_2x2();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(
            ArrayPlacement::Mesh { mesh: mesh.clone(), sharding: sharding.clone() }
                .resolve(&session, 1)
                .unwrap(),
            (mesh.clone(), sharding.clone()),
        );

        // A bare sharding needs a configured mesh with matching topology.
        let placement = ArrayPlacement::Sharding(sharding.clone());
        assert!(matches!(placement.clone().resolve(&session, 1), Err(Error::Session(SessionError::MissingMesh))));
        let configured = session.with_device_mesh(mesh.clone());
        assert_eq!(placement.resolve(&configured, 1).unwrap(), (mesh.clone(), sharding));

        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("z", 4, MeshAxisType::Auto).unwrap()]).unwrap();
        let error = ArrayPlacement::Sharding(Sharding::replicated(other_mesh.clone(), 1))
            .resolve(&configured, 1)
            .unwrap_err();
        assert!(matches!(
            &error,
            Error::Core(error) if matches!(
                error.as_ref(),
                CoreError::Sharding(ShardingError::MeshMismatch { expected, actual })
                    if expected == mesh.logical_mesh() && actual == &other_mesh,
            ),
        ));
        assert_eq!(
            error.to_string(),
            format!("mesh mismatch; expected `{:?}` but got `{other_mesh:?}`", mesh.logical_mesh())
        );
    }

    #[test]
    fn test_placement_options_new() {
        let target = ArrayPlacement::Device(Device::new(3, 0));
        let options =
            PlacementOptions::new(Some(target.clone()), Option::<ArrayPlacement>::None, Some(true), Some(Some(false)));
        assert_eq!(options.target(), Some(&target));
        assert_eq!(options.source(), None);
        assert_eq!(options.donate(), Some(&true));
        assert_eq!(options.may_alias(), Some(&Some(false)));
        assert_eq!(options.into_parts(), (Some(target), None, Some(true), Some(Some(false))));
    }

    #[test]
    fn test_placement_options_empty() {
        let options = PlacementOptions::<ArrayPlacement, ArrayPlacement, bool, Option<bool>>::empty();
        assert_eq!(options.into_parts(), (None, None, None, None));
        assert_eq!(PlacementOptions::defaults(), PlacementOptions::empty());
        assert_eq!(PlacementOptions::default(), PlacementOptions::defaults());
    }

    #[test]
    fn test_session_place_uploads_host_leaves() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[1]).unwrap();

        // Without a destination, a host leaf is committed to the first local device.
        let scalar = session.place(2.5f32, PlacementOptions::defaults()).unwrap();
        let flag = session.place(true, PlacementOptions::defaults()).unwrap();
        assert_eq!(scalar.r#type().into_owned().with_sharding(None).unwrap(), ArrayType::scalar(DataType::F32));
        assert_eq!(scalar.mesh().devices(), [Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap()]);
        assert_eq!(scalar.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![2.5]);
        assert_eq!(flag.data_type(), DataType::Boolean);
        assert_eq!(flag.copy_to_host_bytes().unwrap(), vec![1u8]);

        // A destination device places the leaf on that device.
        let placed = session
            .place(
                bf16::from_f32(1.5),
                PlacementOptions::new(
                    Some(ArrayPlacement::Device(device)),
                    Option::<ArrayPlacement>::None,
                    Option::<bool>::None,
                    Option::<Option<bool>>::None,
                ),
            )
            .unwrap();
        assert_eq!(placed.mesh().devices(), [device]);
        assert_eq!(placed.copy_to_host_bytes().unwrap(), bf16::from_f32(1.5).to_bits().to_ne_bytes().to_vec());
    }

    #[test]
    fn test_session_place_uploads_reference_backend_arrays() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let matrix = CoreArray::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap();
        let array = session.place(matrix, PlacementOptions::defaults()).unwrap();
        assert_eq!(
            array.r#type().into_owned().with_sharding(None).unwrap(),
            ArrayType::new_static(DataType::F32, [2, 3]),
        );
        assert_eq!(array.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn test_session_place_broadcasts_root_placement_over_array_tuple() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(client_devices[0].id().unwrap(), client_devices[0].process_index().unwrap())],
        )
        .unwrap();
        let source_sharding = Sharding::replicated(source_mesh.logical_mesh().clone(), 1);
        let first_source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(source_sharding.clone())
            .unwrap();
        let first_source_array = Array::from_host_buffer(
            &session,
            first_source_type,
            source_mesh.clone(),
            None,
            values_to_bytes::<f32>(&[0.0, 1.0, 2.0, 3.0]).as_slice(),
        )
        .unwrap();
        let second_source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(source_sharding)
            .unwrap();
        let second_source_array = Array::from_host_buffer(
            &session,
            second_source_type,
            source_mesh,
            None,
            values_to_bytes::<f32>(&[10.0, 11.0, 12.0, 13.0]).as_slice(),
        )
        .unwrap();

        let target_mesh = domain_mesh(client, "x", 2);
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let moved_arrays = session
            .place(
                (first_source_array, second_source_array),
                PlacementOptions::new(
                    Some(ArrayPlacement::Mesh { mesh: target_mesh.clone(), sharding: target_sharding.clone() }),
                    Option::<ArrayPlacement>::None,
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
    fn test_session_place_broadcasts_optional_placement_prefixes() {
        let client = cpu_client(2);
        let mesh = domain_mesh(client, "x", 2);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let values = vec![1f32, 2., 3., 4.];
        let host = CoreArray::vector(values.clone()).unwrap();

        // An explicit sharding uses the session mesh. The None prefix sends both leaves to the first local device.
        let (partitioned, [local, other_local]) = session
            .place(
                (host.clone(), [host.clone(), host]),
                PlacementOptions::<_, ArrayPlacement>::new(
                    Some((Some(ArrayPlacement::Sharding(sharded.clone())), None)),
                    None,
                    None,
                    None,
                ),
            )
            .unwrap();
        assert_eq!(partitioned.mesh(), &mesh);
        assert_eq!(partitioned.sharding(), &sharded);
        assert_eq!(shard_values(&partitioned), vec![vec![1., 2.], vec![3., 4.]]);
        for array in [&local, &other_local] {
            assert_eq!(array.mesh().devices(), &mesh.devices()[..1]);
            assert_eq!(shard_values(array), vec![values.clone()]);
        }

        // For existing arrays, None leaves preserve their destination or omit their source check independently.
        let (moved, kept) = session
            .place(
                (local, partitioned.clone()),
                PlacementOptions::<_, _>::new(
                    Some((Some(ArrayPlacement::Sharding(replicated.clone())), None)),
                    Some((None, Some(ArrayPlacement::Sharding(sharded)))),
                    None,
                    None,
                ),
            )
            .unwrap();
        assert_eq!(moved.mesh(), &mesh);
        assert_eq!(moved.sharding(), &replicated);
        assert_eq!(shard_values(&moved), vec![values.clone(), values]);
        assert_eq!(kept.mesh(), partitioned.mesh());
        assert_eq!(kept.sharding(), partitioned.sharding());
        for (kept_shard, source_shard) in kept.shards().iter().zip(partitioned.shards()) {
            assert!(Arc::ptr_eq(kept_shard.buffer().unwrap(), source_shard.buffer().unwrap()));
        }
    }

    #[test]
    fn test_session_place_donation_and_aliasing() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
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
            &session,
            ArrayType::new_static(DataType::F32, [4]),
            source_mesh,
            None,
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        let shares_source_buffer = |array: &Array| {
            array
                .shards()
                .iter()
                .map(|shard| Arc::ptr_eq(shard.buffer().unwrap(), source.shards()[0].buffer().unwrap()))
                .collect::<Vec<_>>()
        };
        let options = |target: Option<ArrayPlacement>, donate: bool, may_alias: Option<bool>| {
            PlacementOptions::new(target, Option::<ArrayPlacement>::None, Some(donate), Some(may_alias))
        };

        // Without a new placement, the result is the leaf itself unless aliasing is forbidden.
        let unchanged = session.place(source.clone(), options(None, false, None)).unwrap();
        assert_eq!(shares_source_buffer(&unchanged), vec![true]);
        let copied = session.place(source.clone(), options(None, false, Some(false))).unwrap();
        assert_eq!(copied.r#type(), source.r#type());
        assert_eq!(copied.mesh(), source.mesh());
        assert_eq!(shares_source_buffer(&copied), vec![false]);
        assert_eq!(shard_values(&copied), vec![vec![1., 2., 3., 4.]]);

        // A forbidden alias copies every shard within its own memory, so an array in host memory stays there.
        let host = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]).with_memory(Memory::Host { pinned: true }),
            source.mesh().clone(),
            None,
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        let copied = session.place(host.clone(), options(None, false, Some(false))).unwrap();
        assert_eq!(copied.r#type(), host.r#type());
        assert!(!Arc::ptr_eq(copied.shards()[0].buffer().unwrap(), host.shards()[0].buffer().unwrap()));
        assert_eq!(copied.shards()[0].buffer().unwrap().memory().unwrap().kind().unwrap(), "pinned_host");
        assert_eq!(shard_values(&copied), vec![vec![1., 2., 3., 4.]]);

        // A shard that stays on its device shares its buffer unless aliasing is forbidden.
        let target = ArrayPlacement::mesh(mesh.clone(), replicated).unwrap();
        let placed = session.place(source.clone(), options(Some(target.clone()), false, None)).unwrap();
        assert_eq!(shares_source_buffer(&placed), vec![true, false]);
        assert_eq!(shard_values(&placed), vec![vec![1., 2., 3., 4.]; 2]);
        let placed = session.place(source.clone(), options(Some(target), false, Some(false))).unwrap();
        assert_eq!(shares_source_buffer(&placed), vec![false, false]);
        assert_eq!(shard_values(&placed), vec![vec![1., 2., 3., 4.]; 2]);

        // A donated leaf is consumed. This one shares its buffers with `source`, so the donation does not take effect
        // and `source` stays readable.
        let target = ArrayPlacement::mesh(mesh, sharded.clone()).unwrap();
        let donated = session.place(source.clone(), options(Some(target), true, None)).unwrap();
        assert_eq!(donated.sharding(), &sharded);
        assert_eq!(shard_values(&donated), vec![vec![1., 2.], vec![3., 4.]]);
        assert_eq!(shard_values(&source), vec![vec![1., 2., 3., 4.]]);
    }

    #[test]
    fn test_session_place_preserves_partially_addressable_array_when_target_is_absent() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
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
        let source_array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let source_array = Array::from_addressable_buffers(
            &session,
            source_array_type,
            mesh.clone(),
            Some(sharding.clone()),
            vec![local_source_buffer],
        )
        .unwrap();

        let copied_array = session.place(source_array, PlacementOptions::defaults()).unwrap();
        let expected_visualization =
            format!("┌─────┬─────┐\n│{:^5}│{:^5}│\n└─────┴─────┘", local_device_id, remote_device_id);

        assert_eq!(copied_array.addressable_shards().count(), 1);
        assert_eq!(copied_array.sharding().visualize().unwrap().render(false), expected_visualization);
        let copied_shard_bytes = copied_array.device_shard(local_device_id).unwrap().copy_to_host_bytes().unwrap();
        assert_eq!(values_from_bytes::<f32>(copied_shard_bytes.as_slice()), vec![0.0, 1.0]);
        assert!(copied_array.device_shard(remote_device_id).unwrap().buffer().is_none());
    }

    #[test]
    fn test_session_place_rejects_mismatched_source_for_array_leaf() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let source_mesh = domain_mesh(client, "x", 1);
        let source_device = source_mesh.devices()[0];
        let source_sharding = Sharding::replicated(source_mesh.logical_mesh().clone(), 1);
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]))
            .with_sharding(source_sharding.clone())
            .unwrap();
        let source_array = Array::from_host_buffer(
            &session,
            source_type,
            source_mesh.clone(),
            None,
            values_to_bytes::<f32>(&[0.0, 1.0]).as_slice(),
        )
        .unwrap();
        let expected_placement = ArrayPlacement::device(Device::new(source_device.id() + 1, 0));
        let expected_source = expected_placement.clone().resolve(&session, 1).unwrap();
        let actual_source = (source_mesh, source_sharding);

        let error = session
            .place(
                (source_array.clone(), source_array),
                PlacementOptions::new(
                    Option::<ArrayPlacement>::None,
                    Some((None, Some(expected_placement))),
                    Option::<bool>::None,
                    Option::<Option<bool>>::None,
                ),
            )
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            format!(
                "expected source mesh `{:?}` with sharding `{:?}` does not match the array's current mesh `{:?}` \
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
