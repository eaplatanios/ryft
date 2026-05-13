use super::*;

/// Concrete mesh/sharding target used by the higher-level [`device_put`] API.
#[derive(Clone, Debug, PartialEq, Eq, Parameter)]
pub struct ArrayPlacement {
    /// Concrete destination mesh describing the device topology.
    mesh: DeviceMesh,

    /// Sharding to apply over [`Self::mesh`].
    sharding: Sharding,
}

impl ArrayPlacement {
    /// Creates a new [`ArrayPlacement`].
    ///
    /// Returns an error if `sharding` refers to a different logical mesh than `mesh`.
    #[inline]
    pub fn new(mesh: DeviceMesh, sharding: Sharding) -> Result<Self, ArrayError> {
        if mesh.logical_mesh() != sharding.mesh() {
            return Err(ShardingError::MeshMismatch {
                expected: mesh.logical_mesh().clone(),
                actual: sharding.mesh().clone(),
            }
            .into());
        }
        Ok(Self { mesh, sharding })
    }

    /// Creates an [`ArrayPlacement`] from already matched mesh and sharding metadata.
    #[inline]
    pub(crate) fn from_parts_unchecked(mesh: DeviceMesh, sharding: Sharding) -> Self {
        Self { mesh, sharding }
    }

    /// Returns the concrete destination mesh.
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        &self.mesh
    }

    /// Returns the sharding applied over [`Self::mesh`].
    #[inline]
    pub fn sharding(&self) -> &Sharding {
        &self.sharding
    }

    /// Consumes this placement and returns its mesh and sharding.
    #[inline]
    pub(crate) fn into_parts(self) -> (DeviceMesh, Sharding) {
        (self.mesh, self.sharding)
    }

    pub(crate) fn single_device(device: MeshDevice, rank: usize) -> Result<Self, ArrayError> {
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("device", 1, MeshAxisType::Auto)?])?;
        let mesh = DeviceMesh::new(logical_mesh, vec![device])?;
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), rank);
        Self::new(mesh, sharding)
    }

    pub(crate) fn default_device(client: &Client<'_>, rank: usize) -> Result<Self, ArrayError> {
        let device = client.addressable_devices()?.into_iter().next().ok_or(ArrayError::MissingDefaultDevice)?;
        Self::single_device(MeshDevice::new(device.id()?, device.process_index()?), rank)
    }
}

/// Placement leaf accepted by the higher-level [`device_put`] API.
///
/// This models the current `ryft` subset of JAX's `device` / `src` arguments:
/// - [`Self::Device`] commits one leaf to a single concrete device, represented internally as a
///   size-1 mesh with fully replicated sharding, and
/// - [`Self::Placement`] commits one leaf to an explicit mesh/sharding pair.
#[derive(Clone, Debug, PartialEq, Eq, Parameter)]
pub enum DevicePutTarget {
    /// Commit the value to one concrete device.
    Device(MeshDevice),

    /// Commit the value to the provided mesh/sharding pair.
    Placement(ArrayPlacement),
}

impl DevicePutTarget {
    /// Creates a single-device placement.
    #[inline]
    pub fn device(device: MeshDevice) -> Self {
        Self::Device(device)
    }

    /// Creates an explicit mesh/sharding placement.
    ///
    /// Returns an error if `sharding` refers to a different logical mesh than `mesh`.
    #[inline]
    pub fn placement(mesh: DeviceMesh, sharding: Sharding) -> Result<Self, ArrayError> {
        ArrayPlacement::new(mesh, sharding).map(Self::Placement)
    }

    pub(crate) fn resolve(self, rank: usize) -> Result<ArrayPlacement, ArrayError> {
        match self {
            Self::Device(device) => ArrayPlacement::single_device(device, rank),
            Self::Placement(placement) => Ok(placement),
        }
    }
}

impl From<MeshDevice> for DevicePutTarget {
    fn from(value: MeshDevice) -> Self {
        Self::Device(value)
    }
}

impl From<ArrayPlacement> for DevicePutTarget {
    fn from(value: ArrayPlacement) -> Self {
        Self::Placement(value)
    }
}

/// Options for the higher-level [`device_put`] API.
///
/// Each field follows JAX's tree-prefix semantics: when a field is present, its structure is
/// broadcast over the input tree and applied leafwise.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DevicePutOptions<Device = DevicePutTarget, Src = DevicePutTarget, Donate = bool, MayAlias = Option<bool>> {
    /// Destination placement tree prefix. When absent, host leaves are committed to the default
    /// local device and [`Array`] leaves preserve their current placement.
    device: Option<Device>,

    /// Source placement tree prefix. This is validated for [`Array`] leaves and ignored for host
    /// leaves, which do not carry runtime placement metadata before upload.
    src: Option<Src>,

    /// Donation tree prefix. This is best-effort in the current `ryft` runtime.
    donate: Option<Donate>,

    /// May-alias tree prefix. `Some(false)` forces a fresh array result when possible.
    may_alias: Option<MayAlias>,
}

impl<Device, Src, Donate, MayAlias> DevicePutOptions<Device, Src, Donate, MayAlias> {
    /// Creates a new [`DevicePutOptions`].
    #[inline]
    pub fn new(device: Option<Device>, src: Option<Src>, donate: Option<Donate>, may_alias: Option<MayAlias>) -> Self {
        Self { device, src, donate, may_alias }
    }

    /// Returns the destination placement tree prefix, if one was provided.
    #[inline]
    pub fn device(&self) -> Option<&Device> {
        self.device.as_ref()
    }

    /// Returns the source placement tree prefix, if one was provided.
    #[inline]
    pub fn src(&self) -> Option<&Src> {
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

    pub(crate) fn into_parts(self) -> (Option<Device>, Option<Src>, Option<Donate>, Option<MayAlias>) {
        (self.device, self.src, self.donate, self.may_alias)
    }

    /// Creates a new [`DevicePutOptions`] with all fields unset.
    #[inline]
    pub fn empty() -> Self {
        Self::new(None, None, None, None)
    }
}

impl<Device, Src, Donate, MayAlias> Default for DevicePutOptions<Device, Src, Donate, MayAlias> {
    fn default() -> Self {
        Self::empty()
    }
}

impl DevicePutOptions<DevicePutTarget, DevicePutTarget, bool, Option<bool>> {
    /// Creates default high-level [`device_put`] options without requiring generic type inference.
    #[inline]
    pub fn defaults() -> Self {
        Self::empty()
    }
}
