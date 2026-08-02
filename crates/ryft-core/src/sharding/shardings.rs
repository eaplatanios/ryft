use std::collections::{BTreeSet, HashSet};
use std::fmt::{Display, Formatter};

use ryft_macros::Parameter;

use crate::parameters::Parameter;
use crate::sharding::ShardingError;
use crate::sharding::meshes::{LogicalMesh, MeshAxisType};

/// Describes how a single dimension of an array/tensor is distributed across [`LogicalMesh`] axes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ShardingDimension {
    /// Dimension that is replicated across the devices in a mesh instead of being sharded/partitioned.
    Replicated,

    /// Dimension that is sharded/partitioned by the mesh axes with the specified names. The dimension is sharded along
    /// the product of the specified axes, in major to minor order. For example, with a `4x2` mesh with `"data"` and
    /// `"model"` axes and `Sharded(["data", "model"])`, a dimension of size `24` is split into `4 * 2 = 8` partitions.
    Sharded(Vec<String>),

    /// Dimension that is unconstrained when it comes to sharding, meaning that the compiler is free to decide
    /// if and how to shard it.
    Unconstrained,
}

impl ShardingDimension {
    /// Creates a new [`Self::Replicated`].
    #[inline]
    pub fn replicated() -> Self {
        Self::Replicated
    }

    /// Creates a new [`Self::Sharded`].
    #[inline]
    pub fn sharded<N: Into<String>, I: IntoIterator<Item = N>>(axis_names: I) -> Self {
        Self::Sharded(axis_names.into_iter().map(Into::into).collect())
    }

    /// Creates a new [`Self::Unconstrained`].
    #[inline]
    pub fn unconstrained() -> Self {
        Self::Unconstrained
    }
}

impl Display for ShardingDimension {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Replicated => write!(formatter, "{{}}"),
            Self::Unconstrained => write!(formatter, "{{?}}"),
            Self::Sharded(axis_names) => {
                write!(formatter, "{{")?;
                if let Some((first_axis_name, remaining_axis_names)) = axis_names.split_first() {
                    write!(formatter, "'{}'", first_axis_name.replace('\'', "\\'"))?;
                    for axis_name in remaining_axis_names {
                        write!(formatter, ", '{}'", axis_name.replace('\'', "\\'"))?;
                    }
                }
                write!(formatter, "}}")
            }
        }
    }
}

/// [`LogicalMesh`]-bound sharding for a logical array value. This is the primary user-facing sharding type for
/// compilation-time annotations. It owns the [`LogicalMesh`] together with the per-dimension [`ShardingDimension`]
/// assignments and any additional state needed to model partial reductions and [`MeshAxisType::Manual`] mesh axes.
///
/// # Example
///
/// Consider the following [`Sharding`]:
///
/// ```ignore
/// Sharding {
///     mesh,
///     dimensions: vec![
///         ShardingDimension::sharded(["data"]),
///         ShardingDimension::replicated(),
///     ],
///     unreduced_axes: std::collections::BTreeSet::from(["model".to_string()]),
///     reduced_axes: std::collections::BTreeSet::new(),
///     varying_manual_axes: std::collections::BTreeSet::new(),
/// };
/// ```
///
/// In this case, the `"data"` [`MeshAxis`] shards array dimension `0`, while `"model"` does not shard any ranked
/// dimension and instead marks the value as still unreduced along the mesh axis `"model"`. Without `unreduced_axes`,
/// that unused mesh axis would be indistinguishable from a truly replicated axis.
///
/// # References
///
/// For more information on the approach Ryft takes to sharding, you can refer to the relevant JAX documentation that
/// inspired it. The following pages are particularly relevant:
///
/// - [Distributed Arrays and Automatic Parallelization](
///   https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
/// - [Explicit Sharding](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html)
/// - [Manual Parallelism with `shard_map`](https://docs.jax.dev/en/latest/notebooks/shard_map.html#so-let-s-see-a-shard-map).
/// - [Memories and Host Offloading](https://docs.jax.dev/en/latest/notebooks/host-offloading.html)
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct Sharding {
    /// Refer to the documentation of [`Self::mesh`] for information on this field.
    mesh: LogicalMesh,

    /// Refer to the documentation of [`Self::dimensions`] for information on this field.
    dimensions: Vec<ShardingDimension>,

    /// Refer to the documentation of [`Self::unreduced_axes`] for information on this field.
    unreduced_axes: BTreeSet<String>,

    /// Refer to the documentation of [`Self::reduced_axes`] for information on this field.
    reduced_axes: BTreeSet<String>,

    /// Refer to the documentation of [`Self::varying_manual_axes`] for information on this field.
    varying_manual_axes: BTreeSet<String>,
}

impl Sharding {
    /// Creates a new [`Sharding`] from a [`LogicalMesh`] and a per-dimension list of [`ShardingDimension`]s.
    /// Use [`Self::with_unreduced_axes`], [`Self::with_reduced_axes`], or [`Self::with_varying_manual_axes`] to
    /// configure the corresponding optional axis sets.
    pub fn new(mesh: LogicalMesh, dimensions: Vec<ShardingDimension>) -> Result<Self, ShardingError> {
        let sharding = Self {
            mesh,
            dimensions,
            unreduced_axes: BTreeSet::new(),
            reduced_axes: BTreeSet::new(),
            varying_manual_axes: BTreeSet::new(),
        };

        let mut used_axis_names = HashSet::new();
        for (dimension, partition_dimension) in sharding.dimensions.iter().enumerate() {
            if let ShardingDimension::Sharded(axis_names) = partition_dimension {
                if axis_names.is_empty() {
                    return Err(ShardingError::EmptySharding { dimension });
                }

                let mut seen_axis_names = HashSet::new();
                for axis_name in axis_names {
                    if sharding.mesh.axis_index(axis_name).is_none() {
                        return Err(ShardingError::UnknownMeshAxisName { name: axis_name.clone() });
                    }

                    if !seen_axis_names.insert(axis_name.clone()) || !used_axis_names.insert(axis_name.clone()) {
                        return Err(ShardingError::DuplicateMeshAxisName { name: axis_name.clone() });
                    }
                }
            }
        }

        Ok(sharding)
    }

    /// Creates a new _fully-replicated_ [`Sharding`] for an array with rank `rank`. All dimensions in the resulting
    /// sharding are going to be [`ShardingDimension::Replicated`], meaning that a copy of the full array will be
    /// present on every device.
    #[inline]
    pub fn replicated(mesh: LogicalMesh, rank: usize) -> Self {
        Self {
            mesh,
            dimensions: vec![ShardingDimension::Replicated; rank],
            unreduced_axes: BTreeSet::new(),
            reduced_axes: BTreeSet::new(),
            varying_manual_axes: BTreeSet::new(),
        }
    }

    /// Returns the [`LogicalMesh`] that describes the device topology underlying this [`Sharding`] and gives meaning
    /// to every [`MeshAxis`] name stored in it. This is effectively the coordinate system for the rest of this struct.
    /// Every axis name mentioned in [`Self::dimensions`], [`Self::unreduced_axes`], [`Self::reduced_axes`], and
    /// [`Self::varying_manual_axes`] is resolved against this mesh.
    #[inline]
    pub fn mesh(&self) -> &LogicalMesh {
        &self.mesh
    }

    /// Returns the ranked per-array dimension [`Sharding`] partition assignments. This is the array-rank-indexed part
    /// of this sharding: `dimensions[i]` describes how the logical array dimension `i` is partitioned across the mesh.
    /// For example, on a mesh with axes `("data", "model")`, the [`dimensions`](Self::dimensions) assignment
    /// `[ShardingDimension::sharded(["data"]), ShardingDimension::replicated()]` means that the first array dimension
    /// is split across `"data"` while the second array dimension is replicated on every device. This field
    /// intentionally does not try to encode every mesh-related fact about the value. Mesh axes that matter
    /// semantically but do not correspond to a ranked array dimension are stored separately in
    /// [`Self::unreduced_axes`], [`Self::reduced_axes`], and [`Self::varying_manual_axes`].
    #[inline]
    pub fn dimensions(&self) -> &[ShardingDimension] {
        &self.dimensions
    }

    /// Returns the mesh axes along which values carry per-device partial results. This is the "a cross-device reduction
    /// still needs to happen" marker. An axis can disappear from [`Self::dimensions`] after a local computation
    /// reduces over the corresponding array dimension, but the value may still not be truly replicated; each shard can
    /// still hold a different partial result that must later be combined across that mesh axis. Concretely, imagine a
    /// mesh with axes `("data", "model")` and a value whose first tensor dimension is sharded by `"data"`. If a local
    /// computation then sums over a `"model"`-partitioned feature dimension, the resulting value may have no ranked
    /// dimension left that mentions `"model"`, yet each `"model"` shard still owns a different partial sum. Setting
    /// `unreduced_axes` to `["model"]` preserves that fact. This is why this field is needed even though the mesh axis
    /// no longer appears in [`Self::dimensions`]; without it, an axis that is absent from ranked dimensions would be
    /// indistinguishable from ordinary replication.
    #[inline]
    pub fn unreduced_axes(&self) -> &BTreeSet<String> {
        &self.unreduced_axes
    }

    /// Returns the mesh axes across which values are known to have already been reduced. This is the dual of
    /// [`Self::unreduced_axes`]. A reduced axis is computationally indistinguishable from a replicated one (every
    /// device holds the same data along it), and the marker records that the value was produced by a reduction across
    /// that axis, even though that fact no longer has a direct ranked-dimension representation. The two sets are also
    /// duals under transposition (i.e., the cotangent of a value that is unreduced along an axis is reduced along that
    /// axis, and vice versa), mirroring how [JAX's `PartitionSpec`](
    /// https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.PartitionSpec) pairs `unreduced` with `reduced`.
    /// For [`MeshAxisType::Manual`] axes, the marker records that a manual mesh axis has already been consumed by a
    /// reduction inside a `shard_map` body: a concrete example is an output that is replicated in [`Self::dimensions`]
    /// but was produced by first summing across the active manual axis `"data"` inside the mapped computation, where
    /// `reduced_axes` being set to `["data"]` distinguishes "this value is already reduced across `data`" from both
    /// "this value is still unreduced across `data`" and "this axis was never relevant to the value".
    #[inline]
    pub fn reduced_axes(&self) -> &BTreeSet<String> {
        &self.reduced_axes
    }

    /// Returns the [`MeshAxisType::Manual`] mesh axes for which `shard_map` values are known to vary along. Unlike
    /// [`Self::dimensions`], this is not a placement description. It answers a typing question used while tracing
    /// `shard_map`: if we compared two otherwise identical devices that differ only along one of these axes, could this
    /// local value still be different? A concrete nested-`shard_map` example is an outer map that is manual over `"y"`
    /// and an inner map whose input sharding specifications additionally shard the value over manual axis `"x"`. Inside
    /// the inner body, the local array can still have the same rank and local shape as before, but it now semantically
    /// varies across both manual axes, and so the trace has `varying_manual_axes` set to `["y", "x"]`. This is needed
    /// because neither ranked sharding nor reduction-state fields can say whether a local value is uniform across the
    /// active manual shards. For example, constants created inside `shard_map` preserve [`Self::unreduced_axes`] and
    /// [`Self::reduced_axes`] but clear [`Self::varying_manual_axes`], because a constant does not vary from shard to
    /// shard even when it is traced under manual axes.
    #[inline]
    pub fn varying_manual_axes(&self) -> &BTreeSet<String> {
        &self.varying_manual_axes
    }

    /// Returns the rank (i.e., number of dimensions) of this [`Sharding`].
    #[inline]
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }

    /// Returns the names of the mesh axes that are implicitly or explicitly replicated by this [`Sharding`].
    pub fn replicated_axes(&self) -> Vec<&str> {
        let mut used_axes = HashSet::new();
        for dimension in &self.dimensions {
            if let ShardingDimension::Sharded(axis_names) = dimension {
                used_axes.extend(axis_names.iter().map(String::as_str));
            }
        }
        used_axes.extend(self.unreduced_axes.iter().map(String::as_str));
        used_axes.extend(self.reduced_axes.iter().map(String::as_str));
        self.mesh
            .axes()
            .iter()
            .filter_map(|axis| {
                let axis_name = axis.name();
                (matches!(self.mesh.axis_type(axis_name), Some(MeshAxisType::Explicit | MeshAxisType::Manual))
                    && !used_axes.contains(axis_name))
                .then_some(axis_name)
            })
            .collect()
    }

    /// Returns the partition index for the provided array dimension that is owned by the device at the provided
    /// mesh coordinates. Each dimension of a sharded array is partitioned independently; a device's full shard is the
    /// intersection of its per-dimension partitions. For example, with sharding `[Sharded(["x"]), Sharded(["y"])]` on
    /// a `2×2` mesh, the device at `(x=1, y=0)` owns partition `1` of dimension `0` (i.e., the second row-band) and
    /// partition `0` of dimension `1` (i.e., the first column-band). Together these identify the rectangular tile that
    /// device holds.
    ///
    /// The returned index is computed as follows:
    ///   - [`ShardingDimension::Replicated`] and [`ShardingDimension::Unconstrained`] always have partition index `0`,
    ///     since every device holds the full extent of that dimension.
    ///   - [`ShardingDimension::Sharded`] results in the row-major linearization of the device's mesh coordinates along
    ///     the sharding axes. For example, given `Sharded(["data", "model"])` where `data` has size `4` and `model` has
    ///     size `2`, a device at mesh coordinates `(data=2, model=1)` maps to partition index `2 * 2 + 1 = 5`.
    pub fn partition_index(&self, dimension: usize, device_mesh_coordinates: &[usize]) -> Result<usize, ShardingError> {
        let sharding_dimension = self
            .dimensions
            .get(dimension)
            .ok_or(ShardingError::DimensionOutOfBounds { dimension, rank: self.rank() })?;
        match sharding_dimension {
            ShardingDimension::Replicated | ShardingDimension::Unconstrained => Ok(0),
            ShardingDimension::Sharded(axis_names) => axis_names.iter().try_fold(0usize, |index, axis_name| {
                let axis_index = self
                    .mesh
                    .axis_index(axis_name)
                    .ok_or_else(|| ShardingError::UnknownMeshAxisName { name: axis_name.clone() })?;
                Ok(index * self.mesh.axes()[axis_index].size() + device_mesh_coordinates[axis_index])
            }),
        }
    }

    /// Returns this [`Sharding`] with its per-array dimension assignments replaced by `dimensions`, while preserving
    /// its mesh and auxiliary axis state. The resulting sharding is revalidated against all preserved axis sets.
    #[inline]
    pub fn with_dimensions(&self, dimensions: Vec<ShardingDimension>) -> Result<Self, ShardingError> {
        let mut sharding = Self::new(self.mesh.clone(), dimensions)?;
        for axis_name in self.unreduced_axes.iter().chain(&self.reduced_axes) {
            if sharding.dimensions_use_axis(axis_name) {
                return Err(ShardingError::DuplicateMeshAxisName { name: axis_name.clone() });
            }
        }
        sharding.unreduced_axes = self.unreduced_axes.clone();
        sharding.reduced_axes = self.reduced_axes.clone();
        sharding.varying_manual_axes = self.varying_manual_axes.clone();
        Ok(sharding)
    }

    /// Returns this [`Sharding`] with its unreduced mesh axes replaced by `unreduced_axes`. Use this when the
    /// sharding carries partial results along mesh axes that still need cross-device reduction.
    pub fn with_unreduced_axes<A: Into<String>, I: IntoIterator<Item = A>>(
        mut self,
        unreduced_axes: I,
    ) -> Result<Self, ShardingError> {
        let unreduced_axes = unreduced_axes.into_iter().map(Into::into).collect::<BTreeSet<_>>();
        for axis_name in &unreduced_axes {
            if self.mesh.axis_index(axis_name).is_none() {
                return Err(ShardingError::UnknownMeshAxisName { name: axis_name.clone() });
            }
            if self.dimensions_use_axis(axis_name) || self.reduced_axes.contains(axis_name) {
                return Err(ShardingError::DuplicateMeshAxisName { name: axis_name.clone() });
            }
            if self.varying_manual_axes.contains(axis_name) {
                return Err(ShardingError::ConflictingVaryingAndUnreducedMeshAxis { name: axis_name.clone() });
            }
        }
        self.unreduced_axes = unreduced_axes;
        Ok(self)
    }

    /// Returns this [`Sharding`] with its reduced mesh axes replaced by `reduced_axes`.
    pub fn with_reduced_axes<A: Into<String>, I: IntoIterator<Item = A>>(
        mut self,
        reduced_axes: I,
    ) -> Result<Self, ShardingError> {
        let reduced_axes = reduced_axes.into_iter().map(Into::into).collect::<BTreeSet<_>>();
        for axis_name in &reduced_axes {
            if self.mesh.axis_index(axis_name).is_none() {
                return Err(ShardingError::UnknownMeshAxisName { name: axis_name.clone() });
            }
            if self.dimensions_use_axis(axis_name) || self.unreduced_axes.contains(axis_name) {
                return Err(ShardingError::DuplicateMeshAxisName { name: axis_name.clone() });
            }
            if self.varying_manual_axes.contains(axis_name) {
                return Err(ShardingError::ConflictingVaryingAndReducedMeshAxis { name: axis_name.clone() });
            }
        }
        self.reduced_axes = reduced_axes;
        Ok(self)
    }

    /// Returns this [`Sharding`] with its varying manual mesh axes replaced by `varying_manual_axes`.
    pub fn with_varying_manual_axes<A: Into<String>, I: IntoIterator<Item = A>>(
        mut self,
        varying_manual_axes: I,
    ) -> Result<Self, ShardingError> {
        self.set_varying_manual_axes(varying_manual_axes)?;
        Ok(self)
    }

    /// Replaces this [`Sharding`]'s varying manual mesh axes with `varying_manual_axes` after validating the
    /// replacement. If validation fails, this sharding remains unchanged.
    pub fn set_varying_manual_axes<A: Into<String>, I: IntoIterator<Item = A>>(
        &mut self,
        varying_manual_axes: I,
    ) -> Result<(), ShardingError> {
        let varying_manual_axes = varying_manual_axes.into_iter().map(Into::into).collect::<BTreeSet<_>>();
        for axis_name in &varying_manual_axes {
            self.validate_varying_manual_axis(axis_name)?;
        }
        self.varying_manual_axes = varying_manual_axes;
        Ok(())
    }

    /// Adds `varying_manual_axes` to this [`Sharding`]'s varying manual mesh axes after validating every addition.
    /// If validation fails, this sharding remains unchanged.
    pub fn extend_varying_manual_axes<A: Into<String>, I: IntoIterator<Item = A>>(
        &mut self,
        varying_manual_axes: I,
    ) -> Result<(), ShardingError> {
        let varying_manual_axes = varying_manual_axes.into_iter().map(Into::into).collect::<Vec<_>>();
        for axis_name in &varying_manual_axes {
            if !self.varying_manual_axes.contains(axis_name) {
                self.validate_varying_manual_axis(axis_name)?;
            }
        }
        self.varying_manual_axes.extend(varying_manual_axes);
        Ok(())
    }

    /// Clears this [`Sharding`]'s varying manual mesh axes.
    #[inline]
    pub fn clear_varying_manual_axes(&mut self) {
        self.varying_manual_axes.clear();
    }

    /// Returns the [`Sharding`] that reverse-mode cotangents of values sharded like this one carry. It swaps
    /// [`Self::unreduced_axes`] with [`Self::reduced_axes`] and keeps all other state unchanged. The cotangent of a
    /// value that still carries per-device partial results along an axis is the same value on every device along that
    /// axis (i.e., marked reduced), while the cotangent of an already-reduced value carries per-device partial results
    /// that still need a reduction (i.e., marked unreduced). The swap is an **involution**, so that
    /// `sharding.cotangent().cotangent()` recovers `sharding`.
    #[inline]
    pub fn cotangent(&self) -> Self {
        Self { unreduced_axes: self.reduced_axes.clone(), reduced_axes: self.unreduced_axes.clone(), ..self.clone() }
    }

    /// Returns a copy of this [`Sharding`] with all of its [`MeshAxisType::Auto`] mesh axes removed.
    pub fn without_auto_axes(&self) -> Self {
        let dimensions = self
            .dimensions
            .iter()
            .map(|dimension| match dimension {
                ShardingDimension::Replicated => ShardingDimension::Replicated,
                ShardingDimension::Unconstrained => ShardingDimension::Unconstrained,
                ShardingDimension::Sharded(axis_names) => {
                    let axis_names = axis_names
                        .iter()
                        .filter(|name| {
                            matches!(self.mesh.axis_type(name), Some(MeshAxisType::Explicit | MeshAxisType::Manual))
                        })
                        .cloned()
                        .collect::<Vec<_>>();
                    if axis_names.is_empty() {
                        ShardingDimension::Replicated
                    } else {
                        ShardingDimension::Sharded(axis_names)
                    }
                }
            })
            .collect();
        let unreduced_axes = self
            .unreduced_axes
            .iter()
            .filter(|name| matches!(self.mesh.axis_type(name), Some(MeshAxisType::Explicit | MeshAxisType::Manual)))
            .cloned()
            .collect();
        let reduced_axes = self
            .reduced_axes
            .iter()
            .filter(|name| matches!(self.mesh.axis_type(name), Some(MeshAxisType::Explicit | MeshAxisType::Manual)))
            .cloned()
            .collect();
        Self { dimensions, unreduced_axes, reduced_axes, ..self.clone() }
    }

    /// Returns a copy of this [`Sharding`] with the provided [`ShardingDimension`] inserted at dimension `index`,
    /// shifting all subsequent dimensions one position to the right. Batching rules use this to extend an explicit
    /// output sharding with an entry for a newly introduced batch dimension. The resulting sharding is revalidated,
    /// and so inserting a [`ShardingDimension::Sharded`] entry that references unknown or already-used mesh axes fails.
    pub fn with_inserted_dimension(&self, index: usize, dimension: ShardingDimension) -> Result<Self, ShardingError> {
        if index > self.dimensions.len() {
            return Err(ShardingError::DimensionOutOfBounds { dimension: index, rank: self.rank() });
        }
        let mut dimensions = self.dimensions.clone();
        dimensions.insert(index, dimension);
        Self::new(self.mesh.clone(), dimensions)?
            .with_unreduced_axes(self.unreduced_axes.clone())?
            .with_reduced_axes(self.reduced_axes.clone())?
            .with_varying_manual_axes(self.varying_manual_axes.clone())
    }

    /// Returns a copy of this [`Sharding`] projected through an array broadcast. Each input dimension `i` is moved to
    /// output dimension `output_dimensions[i]`, while every output dimension not named by the mapping is replicated.
    /// The mesh and the unreduced, reduced, and varying-manual axis sets are preserved.
    ///
    /// # Parameters
    ///
    ///   - `output_rank`: Rank of the broadcast output.
    ///   - `output_dimensions`: Output dimension corresponding to each input dimension. Its length must equal this
    ///     sharding's rank, and its entries must be distinct and less than `output_rank`.
    pub fn with_broadcasted_dimensions(
        &self,
        output_rank: usize,
        output_dimensions: &[usize],
    ) -> Result<Self, ShardingError> {
        if output_dimensions.len() != self.rank() {
            return Err(ShardingError::BroadcastAxisCountMismatch {
                expected: self.rank(),
                actual: output_dimensions.len(),
            });
        }
        let mut dimensions = vec![ShardingDimension::Replicated; output_rank];
        let mut mapped = vec![false; output_rank];
        for (input_dimension, output_dimension) in output_dimensions.iter().copied().enumerate() {
            if output_dimension >= output_rank {
                return Err(ShardingError::BroadcastDimensionOutOfBounds {
                    dimension: output_dimension,
                    rank: output_rank,
                });
            }
            if mapped[output_dimension] {
                return Err(ShardingError::DuplicateBroadcastDimension { dimension: output_dimension });
            }
            mapped[output_dimension] = true;
            dimensions[output_dimension] = self.dimensions[input_dimension].clone();
        }
        Self::new(self.mesh.clone(), dimensions)?
            .with_unreduced_axes(self.unreduced_axes.clone())?
            .with_reduced_axes(self.reduced_axes.clone())?
            .with_varying_manual_axes(self.varying_manual_axes.clone())
    }

    /// Returns a copy of this [`Sharding`] with its `index`-th dimension removed, shifting
    /// subsequent dimensions one position to the left. This is the sharding-level analogue of
    /// [`ArrayType::without_dimension`](crate::ArrayType::without_dimension). The reduction axis sets are unchanged,
    /// but the removed entry's placement is reconciled with the manual-axis model. A dimension sharded over
    /// [`MeshAxisType::Manual`] axes moves those axes into the varying set (i.e., the value now varies across them
    /// rather than being placed along a ranked dimension), while a dimension sharded over a non-manual (e.g., a
    /// [`MeshAxisType::Explicit`]) axis cannot be dropped structurally (that would silently discard an explicit
    /// placement that only a reduction or collective can remove), and yields a
    /// [`ShardingError::NonManualShardedDimensionRemoval`]. [`ShardingDimension::Replicated`] and
    /// [`ShardingDimension::Unconstrained`] entries are dropped without any further effect.
    pub fn without_dimension(&self, index: usize) -> Result<Self, ShardingError> {
        if index >= self.dimensions.len() {
            return Err(ShardingError::DimensionOutOfBounds { dimension: index, rank: self.rank() });
        }
        let mut dimensions = self.dimensions.clone();
        let removed_dimension = dimensions.remove(index);
        let mut varying_manual_axes = self.varying_manual_axes.clone();
        if let ShardingDimension::Sharded(axis_names) = removed_dimension {
            for axis_name in axis_names {
                if self.mesh.axis_type(&axis_name) != Some(MeshAxisType::Manual) {
                    return Err(ShardingError::NonManualShardedDimensionRemoval { dimension: index, name: axis_name });
                }
                varying_manual_axes.insert(axis_name);
            }
        }
        Self::new(self.mesh.clone(), dimensions)?
            .with_unreduced_axes(self.unreduced_axes.clone())?
            .with_reduced_axes(self.reduced_axes.clone())?
            .with_varying_manual_axes(varying_manual_axes)
    }

    /// Returns `true` if this [`Sharding`] and `other`, which must share this sharding's mesh, disagree on any state
    /// that involves an [`Explicit`](MeshAxisType::Explicit) mesh axis: a per-dimension placement entry that differs
    /// while either side shards that dimension over an explicit axis, or an [`unreduced`](Self::unreduced_axes) /
    /// [`reduced`](Self::reduced_axes) axis set whose symmetric difference contains an explicit axis. Differences
    /// confined to [`Manual`](MeshAxisType::Manual) axes, [`Auto`](MeshAxisType::Auto) axes, and any
    /// [`varying_manual_axes`](Self::varying_manual_axes) differences are ignored. This function is used by the
    /// operations that need to determine whether explicit-axis state conflicts (e.g., dynamic slice updating).
    pub fn conflicts_on_explicit_axes_with(&self, other: &Sharding) -> bool {
        if self.dimensions.len() != other.dimensions.len() {
            return true;
        }
        let dimension_has_explicit_axis = |dimension: &ShardingDimension| {
            matches!(dimension, ShardingDimension::Sharded(axis_names)
                if axis_names.iter().any(|name| self.mesh.axis_type(name) == Some(MeshAxisType::Explicit)))
        };
        for (left, right) in self.dimensions.iter().zip(&other.dimensions) {
            if left != right && (dimension_has_explicit_axis(left) || dimension_has_explicit_axis(right)) {
                return true;
            }
        }
        let symmetric_difference_contains_explicit = |left: &BTreeSet<String>, right: &BTreeSet<String>| {
            left.symmetric_difference(right)
                .any(|name| self.mesh.axis_type(name) == Some(MeshAxisType::Explicit))
        };
        symmetric_difference_contains_explicit(&self.unreduced_axes, &other.unreduced_axes)
            || symmetric_difference_contains_explicit(&self.reduced_axes, &other.reduced_axes)
    }

    /// Returns `true` if any ranked dimension of this [`Sharding`] uses `axis_name` for sharding.
    fn dimensions_use_axis(&self, axis_name: &str) -> bool {
        self.dimensions.iter().any(|dimension| match dimension {
            ShardingDimension::Sharded(axis_names) => axis_names.iter().any(|name| name == axis_name),
            ShardingDimension::Replicated | ShardingDimension::Unconstrained => false,
        })
    }

    /// Validates that `axis_name` can be used as a varying manual mesh axis in this [`Sharding`].
    fn validate_varying_manual_axis(&self, axis_name: &str) -> Result<(), ShardingError> {
        if self.mesh.axis_index(axis_name).is_none() {
            return Err(ShardingError::UnknownMeshAxisName { name: axis_name.to_string() });
        }
        if self.mesh.axis_type(axis_name) != Some(MeshAxisType::Manual) {
            return Err(ShardingError::ExpectedManualMeshAxis { name: axis_name.to_string() });
        }
        if self.unreduced_axes.contains(axis_name) {
            return Err(ShardingError::ConflictingVaryingAndUnreducedMeshAxis { name: axis_name.to_string() });
        }
        if self.reduced_axes.contains(axis_name) {
            return Err(ShardingError::ConflictingVaryingAndReducedMeshAxis { name: axis_name.to_string() });
        }
        Ok(())
    }
}

impl Display for Sharding {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        fn write_names<I, S>(formatter: &mut Formatter<'_>, names: I) -> std::fmt::Result
        where
            I: IntoIterator<Item = S>,
            S: AsRef<str>,
        {
            write!(formatter, "{{")?;
            write!(
                formatter,
                "{}",
                names
                    .into_iter()
                    .map(|name| format!("'{}'", name.as_ref().replace('\'', "\\'")))
                    .collect::<Vec<_>>()
                    .join(", ")
            )?;
            write!(formatter, "}}")
        }

        write!(formatter, "{{mesh<[")?;
        write!(
            formatter,
            "{}",
            self.mesh
                .axes()
                .iter()
                .map(|axis| format!("'{}'={}:{}", axis.name().replace('\'', "\\'"), axis.size(), axis.r#type()))
                .collect::<Vec<_>>()
                .join(", ")
        )?;
        write!(formatter, "]>")?;

        write!(formatter, ", [")?;
        write!(formatter, "{}", self.dimensions.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "))?;
        write!(formatter, "]")?;

        if !self.unreduced_axes.is_empty() {
            write!(formatter, ", unreduced=")?;
            write_names(formatter, self.unreduced_axes.iter())?;
        }

        if !self.reduced_axes.is_empty() {
            write!(formatter, ", reduced=")?;
            write_names(formatter, self.reduced_axes.iter())?;
        }

        if !self.varying_manual_axes.is_empty() {
            write!(formatter, ", varying_manual=")?;
            write_names(formatter, self.varying_manual_axes.iter())?;
        }

        write!(formatter, "}}")
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::sharding::meshes::MeshAxis;

    use super::*;

    #[test]
    fn test_sharding_dimension() {
        assert_eq!(ShardingDimension::replicated().to_string(), "{}");
        assert_eq!(ShardingDimension::unconstrained().to_string(), "{?}");
        assert_eq!(ShardingDimension::sharded(["x"]).to_string(), "{'x'}");
        assert_eq!(ShardingDimension::sharded(["x", "y"]).to_string(), "{'x', 'y'}");
        assert_eq!(ShardingDimension::sharded([r"path\to", "x'y"]).to_string(), "{'path\\to', 'x\\'y'}");
    }

    #[test]
    fn test_sharding() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("data", 4, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("manual", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();

        let sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["data"]), ShardingDimension::replicated()])
                .unwrap()
                .with_reduced_axes(["manual"])
                .unwrap();
        assert_eq!(sharding.mesh, mesh.clone());
        assert_eq!(sharding.dimensions, vec![ShardingDimension::sharded(["data"]), ShardingDimension::replicated()]);
        assert_eq!(sharding.unreduced_axes, BTreeSet::new());
        assert_eq!(sharding.reduced_axes, BTreeSet::from(["manual".to_string()]));
        assert_eq!(sharding.varying_manual_axes, BTreeSet::new());
        assert_eq!(sharding.rank(), 2);
        assert_eq!(sharding.partition_index(0, &[0, 0]), Ok(0));
        assert_eq!(sharding.partition_index(0, &[2, 1]), Ok(2));
        assert_eq!(sharding.partition_index(0, &[3, 0]), Ok(3));
        assert_eq!(sharding.partition_index(1, &[0, 0]), Ok(0));
        assert_eq!(sharding.partition_index(1, &[3, 1]), Ok(0));
        assert_eq!(
            sharding.partition_index(2, &[0, 0]),
            Err(ShardingError::DimensionOutOfBounds { dimension: 2, rank: 2 })
        );
        assert_eq!(sharding.replicated_axes(), Vec::<&str>::new());
        assert_eq!(
            sharding.to_string(),
            "{mesh<['data'=4:explicit, 'manual'=2:manual]>, [{'data'}, {}], reduced={'manual'}}",
        );

        let auto = Sharding::replicated(
            LogicalMesh::new(vec![MeshAxis::new("data", 4, MeshAxisType::Auto).unwrap()]).unwrap(),
            1,
        );
        let explicit = Sharding::replicated(
            LogicalMesh::new(vec![MeshAxis::new("data", 4, MeshAxisType::Explicit).unwrap()]).unwrap(),
            1,
        );
        assert_ne!(auto.to_string(), explicit.to_string());

        let replicated = Sharding::replicated(mesh.clone(), 3);
        assert_eq!(replicated.mesh, mesh);
        assert_eq!(
            replicated.dimensions,
            vec![ShardingDimension::replicated(), ShardingDimension::replicated(), ShardingDimension::replicated(),]
        );
        assert_eq!(replicated.unreduced_axes, BTreeSet::new());
        assert_eq!(replicated.reduced_axes, BTreeSet::new());
        assert_eq!(replicated.varying_manual_axes, BTreeSet::new());
        assert_eq!(replicated.rank(), 3);
        assert_eq!(replicated.partition_index(0, &[0, 0]), Ok(0));
        assert_eq!(replicated.partition_index(1, &[3, 1]), Ok(0));
        assert_eq!(replicated.partition_index(2, &[2, 0]), Ok(0));
        assert_eq!(
            replicated.partition_index(3, &[0, 0]),
            Err(ShardingError::DimensionOutOfBounds { dimension: 3, rank: 3 })
        );
        assert_eq!(replicated.replicated_axes(), Vec::from(["data", "manual"]));
        assert_eq!(replicated.to_string(), "{mesh<['data'=4:explicit, 'manual'=2:manual]>, [{}, {}, {}]}");

        assert!(matches!(
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["z"])]),
            Err(ShardingError::UnknownMeshAxisName { name }) if name == "z",
        ));
        assert!(matches!(
            Sharding::new(mesh.clone(), vec![ShardingDimension::Sharded(Vec::new())]),
            Err(ShardingError::EmptySharding { dimension }) if dimension == 0,
        ));
    }

    #[test]
    fn test_sharding_axis_setters() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("n", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();

        let unreduced = sharding.clone().with_unreduced_axes(["y"]).unwrap().with_unreduced_axes(["m"]).unwrap();
        assert_eq!(unreduced.unreduced_axes(), &BTreeSet::from(["m".to_string()]));
        assert!(unreduced.reduced_axes().is_empty());
        assert!(unreduced.varying_manual_axes().is_empty());

        let reduced = sharding.clone().with_reduced_axes(["y"]).unwrap().with_reduced_axes(["m"]).unwrap();
        assert_eq!(reduced.reduced_axes(), &BTreeSet::from(["m".to_string()]));
        assert!(reduced.unreduced_axes().is_empty());
        assert!(reduced.varying_manual_axes().is_empty());

        let varying =
            sharding.clone().with_varying_manual_axes(["m"]).unwrap().with_varying_manual_axes(["n"]).unwrap();
        assert_eq!(varying.varying_manual_axes(), &BTreeSet::from(["n".to_string()]));
        assert!(varying.unreduced_axes().is_empty());
        assert!(varying.reduced_axes().is_empty());

        assert!(sharding.unreduced_axes().is_empty());
        assert!(sharding.reduced_axes().is_empty());
        assert!(sharding.varying_manual_axes().is_empty());
    }

    #[test]
    fn test_sharding_axis_setters_validate_replacements() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();

        assert!(matches!(
            sharding.clone().with_unreduced_axes(["unknown"]),
            Err(ShardingError::UnknownMeshAxisName { name }) if name == "unknown",
        ));
        assert!(matches!(
            sharding.clone().with_unreduced_axes(["x"]),
            Err(ShardingError::DuplicateMeshAxisName { name }) if name == "x",
        ));
        assert!(matches!(
            sharding.clone().with_reduced_axes(["y"]).unwrap().with_unreduced_axes(["y"]),
            Err(ShardingError::DuplicateMeshAxisName { name }) if name == "y",
        ));
        assert!(matches!(
            sharding.clone().with_varying_manual_axes(["m"]).unwrap().with_unreduced_axes(["m"]),
            Err(ShardingError::ConflictingVaryingAndUnreducedMeshAxis { name }) if name == "m",
        ));

        assert!(matches!(
            sharding.clone().with_reduced_axes(["unknown"]),
            Err(ShardingError::UnknownMeshAxisName { name }) if name == "unknown",
        ));
        assert!(matches!(
            sharding.clone().with_reduced_axes(["x"]),
            Err(ShardingError::DuplicateMeshAxisName { name }) if name == "x",
        ));
        assert!(matches!(
            sharding.clone().with_unreduced_axes(["y"]).unwrap().with_reduced_axes(["y"]),
            Err(ShardingError::DuplicateMeshAxisName { name }) if name == "y",
        ));
        assert!(matches!(
            sharding.clone().with_varying_manual_axes(["m"]).unwrap().with_reduced_axes(["m"]),
            Err(ShardingError::ConflictingVaryingAndReducedMeshAxis { name }) if name == "m",
        ));

        assert!(matches!(
            sharding.clone().with_varying_manual_axes(["unknown"]),
            Err(ShardingError::UnknownMeshAxisName { name }) if name == "unknown",
        ));
        assert!(matches!(
            sharding.clone().with_varying_manual_axes(["y"]),
            Err(ShardingError::ExpectedManualMeshAxis { name }) if name == "y",
        ));
        assert!(matches!(
            sharding.clone().with_unreduced_axes(["m"]).unwrap().with_varying_manual_axes(["m"]),
            Err(ShardingError::ConflictingVaryingAndUnreducedMeshAxis { name }) if name == "m",
        ));
        assert!(matches!(
            sharding.with_reduced_axes(["m"]).unwrap().with_varying_manual_axes(["m"]),
            Err(ShardingError::ConflictingVaryingAndReducedMeshAxis { name }) if name == "m",
        ));
    }

    #[test]
    fn test_sharding_without_auto_axes() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("data", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("model", 4, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("batch", 8, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("hidden", 16, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("reduction", 16, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("carry", 32, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(
            mesh.clone(),
            vec![
                ShardingDimension::sharded(["data", "model", "batch"]),
                ShardingDimension::sharded(["hidden"]),
                ShardingDimension::replicated(),
            ],
        )
        .unwrap()
        .with_unreduced_axes(["reduction", "carry"])
        .unwrap();
        assert_eq!(
            sharding.without_auto_axes(),
            Sharding::new(
                mesh,
                vec![
                    ShardingDimension::sharded(["data", "batch"]),
                    ShardingDimension::replicated(),
                    ShardingDimension::replicated(),
                ],
            )
            .unwrap()
            .with_unreduced_axes(["carry"])
            .unwrap(),
        );

        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("w", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "y", "z"])])
            .unwrap()
            .with_unreduced_axes(["w"])
            .unwrap()
            .without_auto_axes();
        assert_eq!(sharding, Sharding::new(mesh, vec![ShardingDimension::sharded(["x", "z"])]).unwrap(),);
        assert!(sharding.replicated_axes().is_empty());
        assert!(sharding.unreduced_axes.is_empty());

        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
            .unwrap()
            .with_reduced_axes(BTreeSet::from(["y".to_string(), "z".to_string()]))
            .unwrap()
            .with_varying_manual_axes(BTreeSet::from(["x".to_string()]))
            .unwrap();
        assert_eq!(
            sharding.without_auto_axes(),
            Sharding::new(mesh, vec![ShardingDimension::replicated()])
                .unwrap()
                .with_reduced_axes(["z"])
                .unwrap()
                .with_varying_manual_axes(["x"])
                .unwrap(),
        );
    }

    #[test]
    fn test_sharding_with_inserted_dimension() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("data", 4, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("model", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["data"])]).unwrap();

        assert_eq!(
            sharding.with_inserted_dimension(0, ShardingDimension::replicated()),
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["data"])],),
        );
        assert_eq!(
            sharding.with_inserted_dimension(1, ShardingDimension::sharded(["model"])),
            Sharding::new(
                mesh.clone(),
                vec![ShardingDimension::sharded(["data"]), ShardingDimension::sharded(["model"])],
            ),
        );
        assert!(matches!(
            sharding.with_inserted_dimension(2, ShardingDimension::replicated()),
            Err(ShardingError::DimensionOutOfBounds { dimension: 2, rank: 1 }),
        ));

        // The resulting sharding is revalidated, so reusing an already-used axis fails.
        assert!(matches!(
            sharding.with_inserted_dimension(0, ShardingDimension::sharded(["data"])),
            Err(ShardingError::DuplicateMeshAxisName { name }) if name == "data",
        ));
    }

    #[test]
    fn test_sharding_with_broadcasted_dimensions() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("data", 4, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("model", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("reduction", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::sharded(["data"]), ShardingDimension::sharded(["model"])],
        )
        .unwrap()
        .with_unreduced_axes(["reduction"])
        .unwrap()
        .with_varying_manual_axes(["model"])
        .unwrap();

        // Input dimensions can be reordered while newly introduced output dimensions remain replicated.
        // Non-ranked reduction and manual-axis state follows the projected sharding unchanged.
        assert_eq!(
            sharding.with_broadcasted_dimensions(3, &[2, 0]),
            Sharding::new(
                mesh,
                vec![
                    ShardingDimension::sharded(["model"]),
                    ShardingDimension::replicated(),
                    ShardingDimension::sharded(["data"]),
                ],
            )
            .unwrap()
            .with_unreduced_axes(["reduction"])
            .unwrap()
            .with_varying_manual_axes(["model"]),
        );

        // Malformed mappings fail before constructing an invalid projected sharding.
        assert!(matches!(
            sharding.with_broadcasted_dimensions(3, &[0]),
            Err(ShardingError::BroadcastAxisCountMismatch { expected: 2, actual: 1 }),
        ));
        assert!(matches!(
            sharding.with_broadcasted_dimensions(2, &[0, 2]),
            Err(ShardingError::BroadcastDimensionOutOfBounds { dimension: 2, rank: 2 }),
        ));
        assert!(matches!(
            sharding.with_broadcasted_dimensions(2, &[1, 1]),
            Err(ShardingError::DuplicateBroadcastDimension { dimension: 1 }),
        ));
    }

    #[test]
    fn test_sharding_without_dimension() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("data", 4, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("model", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["data"])])
                .unwrap();

        // Dropping a replicated dimension removes it cleanly and shifts the remaining dimensions one position left.
        assert_eq!(
            sharding.without_dimension(0),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["data"])]),
        );

        // A dimension sharded over a non-manual axis cannot be dropped structurally, since only a reduction or
        // collective can remove that explicit placement.
        assert!(matches!(
            sharding.without_dimension(1),
            Err(ShardingError::NonManualShardedDimensionRemoval { dimension: 1, name }) if name == "data",
        ));
        assert!(matches!(
            sharding.without_dimension(2),
            Err(ShardingError::DimensionOutOfBounds { dimension: 2, rank: 2 }),
        ));

        // Dropping a dimension sharded over a manual axis moves that axis into the varying set.
        let manual = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["model"])]).unwrap();
        assert_eq!(
            manual.without_dimension(0),
            Sharding::new(mesh.clone(), vec![]).unwrap().with_varying_manual_axes(["model"]),
        );
    }

    #[test]
    fn test_sharding_conflicts_on_explicit_axes_with() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("manual", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("auto", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let with_first_dimension = |dimension: ShardingDimension| {
            Sharding::new(mesh.clone(), vec![dimension, ShardingDimension::replicated()]).unwrap()
        };
        let with_reduction_axes = |unreduced: Vec<&str>, reduced: Vec<&str>, varying: Vec<&str>| {
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
                .unwrap()
                .with_unreduced_axes(unreduced)
                .unwrap()
                .with_reduced_axes(reduced)
                .unwrap()
                .with_varying_manual_axes(varying)
                .unwrap()
        };
        let replicated = with_first_dimension(ShardingDimension::replicated());

        // A sharding never conflicts with itself or with an equal sharding.
        let sharded_over_x = with_first_dimension(ShardingDimension::sharded(["x"]));
        assert!(!sharded_over_x.conflicts_on_explicit_axes_with(&sharded_over_x));
        assert!(!replicated.conflicts_on_explicit_axes_with(&replicated));

        // A rank mismatch is always a conflict, regardless of the axis types involved.
        let rank_one = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert!(sharded_over_x.conflicts_on_explicit_axes_with(&rank_one));

        // A per-dimension placement difference that touches an explicit axis is a conflict, whether that axis is
        // dropped or swapped for another explicit axis.
        let sharded_over_y = with_first_dimension(ShardingDimension::sharded(["y"]));
        assert!(sharded_over_x.conflicts_on_explicit_axes_with(&replicated));
        assert!(sharded_over_x.conflicts_on_explicit_axes_with(&sharded_over_y));

        // A per-dimension placement difference confined to manual or auto axes is tolerated.
        let sharded_over_manual = with_first_dimension(ShardingDimension::sharded(["manual"]));
        let sharded_over_auto = with_first_dimension(ShardingDimension::sharded(["auto"]));
        assert!(!sharded_over_manual.conflicts_on_explicit_axes_with(&replicated));
        assert!(!sharded_over_auto.conflicts_on_explicit_axes_with(&replicated));

        // A reduction-state difference is a conflict only when its symmetric difference contains an explicit axis.
        // Both the unreduced and reduced sets are checked, while manual/auto differences are tolerated.
        assert!(with_reduction_axes(vec!["x"], vec![], vec![]).conflicts_on_explicit_axes_with(&replicated));
        assert!(with_reduction_axes(vec![], vec!["y"], vec![]).conflicts_on_explicit_axes_with(&replicated));
        assert!(!with_reduction_axes(vec!["manual"], vec![], vec![]).conflicts_on_explicit_axes_with(&replicated));

        // A difference confined to the varying-manual-axis set is ignored entirely.
        assert!(!with_reduction_axes(vec![], vec![], vec!["manual"]).conflicts_on_explicit_axes_with(&replicated));
    }
}
