use ryft_core::check_sharding;

use super::*;

/// Target leaf accepted by the higher-level [`device_put()`] API.
///
/// This models the current `ryft` subset of JAX's `device` / `src` arguments:
/// - [`Self::Device`] commits one leaf to a single concrete device, represented internally as a
///   size-1 mesh with fully replicated sharding, and
/// - [`Self::Placement`] commits one leaf to an explicit mesh/sharding pair.
#[derive(Clone, Debug, PartialEq, Eq, Parameter)]
pub enum DevicePutTarget {
    /// Commit the value to one concrete device.
    Device(Device),

    /// Commit the value to the provided mesh/sharding pair.
    Placement {
        /// Concrete destination mesh describing the device topology.
        mesh: DeviceMesh,

        /// Sharding to apply over `mesh`.
        sharding: Sharding,
    },
}

impl DevicePutTarget {
    /// Creates a single-device placement.
    #[inline]
    pub fn device(device: Device) -> Self {
        Self::Device(device)
    }

    /// Creates an explicit mesh/sharding placement.
    ///
    /// Returns an error if `sharding` refers to a different logical mesh than `mesh`.
    #[inline]
    pub fn placement(mesh: DeviceMesh, sharding: Sharding) -> Result<Self, ArrayError> {
        check_sharding!(&mesh, &sharding);
        Ok(Self::Placement { mesh, sharding })
    }

    pub(crate) fn resolve(self, rank: usize) -> Result<(DeviceMesh, Sharding), ArrayError> {
        match self {
            Self::Device(device) => {
                let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("device", 1, MeshAxisType::Auto)?])?;
                let mesh = DeviceMesh::new(logical_mesh, vec![device])?;
                let sharding = Sharding::replicated(mesh.logical_mesh().clone(), rank);
                Ok((mesh, sharding))
            }
            Self::Placement { mesh, sharding } => {
                check_sharding!(&mesh, &sharding);
                Ok((mesh, sharding))
            }
        }
    }
}

impl From<Device> for DevicePutTarget {
    fn from(value: Device) -> Self {
        Self::Device(value)
    }
}

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
