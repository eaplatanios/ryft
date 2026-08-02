use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use crate::sharding::ShardingError;

/// [`MeshAxis`] type which controls sharding constraint propagation. Each axis in a [`LogicalMesh`] can be tagged with
/// a [`MeshAxisType`] that tells the compiler (e.g., Shardy or [GSPMD](https://arxiv.org/abs/2105.04663)) how to treat
/// shardings along that axis during sharding constraint propagation. This type corresponds to
/// [`jax.sharding.AxisType`](https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.AxisType).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum MeshAxisType {
    /// Used for mesh axes whose sharding information is inferred automatically by the compiler
    /// (for example, by Shardy or [GSPMD](https://arxiv.org/abs/2105.04663)).
    #[default]
    Auto,

    /// Used for mesh axes whose sharding information is represented explicitly as part of the
    /// type-level sharding metadata and propagated before compilation.
    Explicit,

    /// Used for mesh axes for which the user manages all device communication explicitly
    /// (e.g., using an operation like `shard_map` which is analogous to
    /// [JAX's `shard_map`](https://docs.jax.dev/en/latest/notebooks/shard_map.html)).
    Manual,
}

impl Display for MeshAxisType {
    #[inline]
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(formatter, "auto"),
            Self::Explicit => write!(formatter, "explicit"),
            Self::Manual => write!(formatter, "manual"),
        }
    }
}

/// Named axis in a [`LogicalMesh`]. Each axis represents one dimension of the device grid with a human-readable name,
/// a size (i.e., the number of devices along that dimension), and a [`MeshAxisType`] that controls sharding propagation
/// behavior for that axis.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MeshAxis {
    /// Name of this [`MeshAxis`].
    pub(super) name: String,

    /// Number of devices along this [`MeshAxis`].
    pub(super) size: usize,

    /// Type of this [`MeshAxis`], controlling sharding propagation behavior.
    pub(super) r#type: MeshAxisType,
}

impl MeshAxis {
    /// Creates a new [`MeshAxis`].
    #[inline]
    pub fn new<N: Into<String>>(name: N, size: usize, r#type: MeshAxisType) -> Result<Self, ShardingError> {
        let name = name.into();
        if name.is_empty() {
            Err(ShardingError::EmptyMeshAxisName)
        } else if size == 0 {
            Err(ShardingError::EmptyMeshAxis { name })
        } else {
            Ok(Self { name, size, r#type })
        }
    }

    /// Returns the name of this [`MeshAxis`].
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the number of devices along this [`MeshAxis`].
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the type of this [`MeshAxis`], controlling sharding propagation behavior.
    #[inline]
    pub fn r#type(&self) -> MeshAxisType {
        self.r#type
    }
}

/// Key used to intern [`LogicalMesh`] instances.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct LogicalMeshKey {
    axes: Vec<MeshAxis>,
}

/// Interned immutable data for a [`LogicalMesh`].
#[doc(hidden)]
#[derive(Debug, PartialEq, Eq)]
pub struct LogicalMeshData {
    /// Named and sized axes that define this logical mesh topology.
    pub(super) axes: Vec<MeshAxis>,

    /// Mapping from [`MeshAxis`] names to their indices/positions in [`Self::axes`].
    pub(super) axis_indices: HashMap<String, usize>,
}

/// Returns all interned [`LogicalMesh`] instances.
#[inline]
fn interned_logical_meshes() -> &'static Mutex<HashMap<LogicalMeshKey, Weak<LogicalMeshData>>> {
    static INTERNER: OnceLock<Mutex<HashMap<LogicalMeshKey, Weak<LogicalMeshData>>>> = OnceLock::new();
    INTERNER.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Logical mesh that represent a device topology that is to be used for sharding. A [`LogicalMesh`] captures the mesh
/// axis names, sizes, and types of a device mesh without binding to physical devices. This is the compilation-time view
/// of a mesh: it provides enough information to validate partition specifications and generate sharding-related code
/// (e.g., [Shardy](https://openxla.org/shardy) MLIR attributes), but it does not carry any device-specific information.
/// Note that equivalent meshes are interned within the process and so repeated constructions share immutable storage.
#[derive(Clone, PartialEq, Eq)]
pub struct LogicalMesh(Arc<LogicalMeshData>);

impl LogicalMesh {
    /// Creates a new [`LogicalMesh`].
    #[inline]
    pub fn new(axes: Vec<MeshAxis>) -> Result<Self, ShardingError> {
        let mut axis_indices = HashMap::with_capacity(axes.len());
        for (axis_index, axis) in axes.iter().enumerate() {
            if axis_indices.insert(axis.name.clone(), axis_index).is_some() {
                return Err(ShardingError::DuplicateMeshAxisName { name: axis.name.clone() });
            }
        }
        let mut interner = interned_logical_meshes().lock().expect("poisoned logical mesh interner mutex");
        match interner.entry(LogicalMeshKey { axes }) {
            Entry::Occupied(mut occupied) => {
                if let Some(mesh) = occupied.get().upgrade() {
                    Ok(Self(mesh))
                } else {
                    let mesh = Arc::new(LogicalMeshData { axes: occupied.key().axes.clone(), axis_indices });
                    occupied.insert(Arc::downgrade(&mesh));
                    Ok(Self(mesh))
                }
            }
            Entry::Vacant(vacant) => {
                let mesh = Arc::new(LogicalMeshData { axes: vacant.key().axes.clone(), axis_indices });
                vacant.insert(Arc::downgrade(&mesh));
                Ok(Self(mesh))
            }
        }
    }

    /// Returns the rank (i.e., number of axes) of this [`LogicalMesh`].
    #[inline]
    pub fn rank(&self) -> usize {
        self.axes.len()
    }

    /// Returns the named and sized axes that define this logical mesh topology.
    #[inline]
    pub fn axes(&self) -> &[MeshAxis] {
        &self.axes
    }

    /// Returns the index of the [`MeshAxis`] with the provided name, if such an axis exists.
    #[inline]
    pub fn axis_index<S: AsRef<str>>(&self, axis_name: S) -> Option<usize> {
        self.axis_indices.get(axis_name.as_ref()).copied()
    }

    /// Returns the size of the [`MeshAxis`] in this [`LogicalMesh`] with the provided name, if such an axis exists.
    #[inline]
    pub fn axis_size<S: AsRef<str>>(&self, axis_name: S) -> Option<usize> {
        self.axis_indices.get(axis_name.as_ref()).map(|axis_index| self.axes[*axis_index].size)
    }

    /// Returns the type of the [`MeshAxis`] in this [`LogicalMesh`] with the provided name, if such an axis exists.
    #[inline]
    pub fn axis_type<S: AsRef<str>>(&self, axis_name: S) -> Option<MeshAxisType> {
        self.axis_indices.get(axis_name.as_ref()).map(|axis_index| self.axes[*axis_index].r#type)
    }

    /// Returns the total number of devices that the topology defined by this [`LogicalMesh`] contains.
    #[inline]
    pub fn device_count(&self) -> usize {
        self.axes.iter().fold(1usize, |count, axis| count * axis.size)
    }
}

impl Debug for LogicalMesh {
    #[inline]
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LogicalMesh")
            .field("axes", &self.axes)
            .field("axis_indices", &self.axis_indices)
            .finish()
    }
}

impl Hash for LogicalMesh {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.axes.hash(state);
    }
}

impl Deref for LogicalMesh {
    type Target = LogicalMeshData;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_mesh_axis_type() {
        assert_eq!(MeshAxisType::default(), MeshAxisType::Auto);
    }

    #[test]
    fn test_mesh_axis() {
        let axis = MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap();
        assert_eq!(axis.name, "x");
        assert_eq!(axis.size, 2);
        assert_eq!(axis.r#type, MeshAxisType::Auto);

        let axis = MeshAxis::new("y", 3, MeshAxisType::Manual).unwrap();
        assert_eq!(axis.name, "y");
        assert_eq!(axis.size, 3);
        assert_eq!(axis.r#type, MeshAxisType::Manual);

        assert!(matches!(MeshAxis::new("", 4, MeshAxisType::Auto), Err(ShardingError::EmptyMeshAxisName)));
        assert!(matches!(
            MeshAxis::new("x", 0, MeshAxisType::Auto),
            Err(ShardingError::EmptyMeshAxis { name }) if name == "x",
        ));
    }

    #[test]
    fn test_logical_mesh() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 3, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 1, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        assert_eq!(mesh.axes.iter().map(|axis| axis.name.as_str()).collect::<Vec<_>>(), vec!["x", "y", "z"]);
        assert_eq!(mesh.axes.iter().map(|axis| axis.size).collect::<Vec<_>>(), vec![2, 3, 1]);
        assert_eq!(
            mesh.axes.iter().map(|axis| axis.r#type).collect::<Vec<_>>(),
            vec![MeshAxisType::Auto, MeshAxisType::Manual, MeshAxisType::Explicit]
        );
        assert_eq!(mesh.axis_indices.get("x"), Some(&0));
        assert_eq!(mesh.axis_indices.get("y"), Some(&1));
        assert_eq!(mesh.axis_indices.get("z"), Some(&2));
        assert_eq!(mesh.axis_indices.get("w"), None);
        assert_eq!(mesh.rank(), 3);
        assert_eq!(mesh.axis_size("x"), Some(2));
        assert_eq!(mesh.axis_size("y"), Some(3));
        assert_eq!(mesh.axis_size("z"), Some(1));
        assert_eq!(mesh.axis_size("w"), None);
        assert_eq!(mesh.axis_type("x"), Some(MeshAxisType::Auto));
        assert_eq!(mesh.axis_type("y"), Some(MeshAxisType::Manual));
        assert_eq!(mesh.axis_type("z"), Some(MeshAxisType::Explicit));
        assert_eq!(mesh.axis_type("w"), None);
        assert_eq!(mesh.device_count(), 6);

        assert!(matches!(
            LogicalMesh::new(vec![
                MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
                MeshAxis::new("x", 3, MeshAxisType::Auto).unwrap(),
            ]),
            Err(ShardingError::DuplicateMeshAxisName { name }) if name == "x",
        ));

        let mesh_0 = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 3, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let mesh_1 = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 3, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let mesh_2 = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 4, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        assert!(Arc::ptr_eq(&mesh_0.0, &mesh_1.0));
        assert!(!Arc::ptr_eq(&mesh_0.0, &mesh_2.0));
    }
}
