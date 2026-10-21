use std::borrow::Borrow;
use std::fmt::Display;
use std::sync::Arc;

use ryft_macros::Parameter;

use crate::arrays::types::arrays::{ArrayType, ArrayTypeRefinements};
use crate::arrays::types::dimensions::{
    Dimension, DimensionComparison, DimensionRefinementFact, DimensionRefinements, DimensionType,
    DimensionTypeRefinements, DimensionVariable, Shape,
};
use crate::operations::DIMENSION_DATA_TYPE;
use crate::parameters::Parameter;
use crate::programs::types::{derive_signature_identity_renaming, type_signature_pairs};
use crate::programs::{
    NoReferent, ReferenceMemberType, ReferenceType, Type, TypeError, TypeIdentityClass, TypeIdentityPosition,
    TypeIdentityRenaming, TypeRefinements, type_identities_needing_runtime_checks,
};

/// Kind of logical value represented by a [`RaggedType`]. Dimension values store integer extents themselves and admit
/// static and unbounded [`DimensionType`]s. Array values store numerical elements with separately described extents.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum RaggedValueKind {
    /// Ordinary array elements with bounded, individually described ragged axes.
    Array,

    /// First-class dimensions indexed by every coordinate of the storage shape.
    Dimension(DimensionType),
}

/// One occurrence of a bounded ragged dimension. Its [`extent_axes`](Self::extent_axes) map each axis of the dense
/// extents payload to the corresponding storage axis, preserving payload axis order. An empty mapping means a scalar
/// payload. Construction of [`RaggedType`] validates axes, finite bounds, and dependencies, then sorts occurrences by
/// their storage axis.
///
/// # Example
///
/// Consider padded storage with shape `[2, 3, 4]`, whose axes are `batch`, `time`, and `element`. Suppose
/// the `element` axis has a different logical length at each `batch`/`time` coordinate. Its [`RaggedDimension`] has
/// [`axis`](Self::axis) equal to `2` and `extent_axes = [0, 1]`: axis `0` of the extents payload corresponds to storage
/// axis `0` (i.e., `batch`), and axis `1` corresponds to storage axis `1` (i.e., `time`). The extents payload therefore
/// has shape `[2, 3]`. For example:
///
/// ```text
/// [[1, 4, 2],
///  [3, 0, 1]]
/// ```
///
/// The storage element at `[b, t, k]` is live when `k < extents[b, t]`. Thus, the slice at `batch=0` and `time=1` has
/// four live elements, while the one at `batch=1` and `time=1` has none. The [`RaggedDimension`] identifies the axis
/// whose length varies through `axis`, while `extent_axes` identifies the axes used to look up that length.
///
/// Note that order matters. In this case, `extent_axes = [1, 0]` would instead require a payload with shape `[3, 2]`,
/// indexed as `extents[t, b]`. An empty `extent_axes` would require a scalar payload giving one shared length for every
/// `batch`/`time` coordinate.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct RaggedDimension {
    /// Storage axis whose true extents are described by this occurrence.
    axis: usize,

    /// Nominal identity of the indexed logical extents, distinct from their padded storage capacity.
    dimension: DimensionVariable,

    /// Mapping from each axis of the dense extents payload to its corresponding storage axis, in payload order.
    extent_axes: Vec<usize>,
}

impl RaggedDimension {
    /// Creates a new [`RaggedDimension`] that describes one ragged axis occurrence.
    #[inline]
    pub fn new(axis: usize, dimension: DimensionVariable, extent_axes: Vec<usize>) -> Self {
        Self { axis, dimension, extent_axes }
    }

    /// Returns the padded storage axis described by this occurrence.
    #[inline]
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// Returns the nominal identity of this occurrence's logical extents.
    #[inline]
    pub fn dimension(&self) -> &DimensionVariable {
        &self.dimension
    }

    /// Returns the mapping from each axis of the dense extents payload to its corresponding storage axis, in payload
    /// order. An empty mapping means a scalar extent payload.
    #[inline]
    pub fn extent_axes(&self) -> &[usize] {
        &self.extent_axes
    }
}

/// Internal immutable representation shared by clones of one [`RaggedType`].
#[derive(Debug, PartialEq, Eq, Hash)]
struct RaggedTypePayload {
    /// Refer to the documentation of [`RaggedType::storage_type`].
    storage_type: ArrayType,

    /// Refer to the documentation of [`RaggedType::value_kind`].
    value_kind: RaggedValueKind,

    /// Refer to the documentation of [`RaggedType::ragged_dimensions`].
    ragged_dimensions: Vec<RaggedDimension>,

    /// Refer to the documentation of [`RaggedType::logical_shape`].
    logical_shape: Shape,

    /// Refer to the documentation of [`RaggedType::extent_types`].
    extent_types: Vec<ArrayType>,
}

/// Type of one logical value containing padded array storage and a flat list of dense indexed extents.
///
/// Every ragged storage axis has exactly its dimension's maximum admitted extent (e.g., an exclusive upper bound of `8`
/// requires storage extent `7`, including when the maximum admitted extent is zero). Other axes retain their ordinary
/// [`Dimension`]s. The [`logical_shape`](Self::logical_shape) replaces each padded axis by the variable of its
/// [`RaggedDimension`] without removing dependency-domain axes. The [`RaggedDimension`] entries are canonicalized in
/// storage-axis order. A domain may name an earlier ragged axis, whose sibling [`RaggedDimension`] determines which
/// payload entries are live; self and later ragged dependencies are rejected. Domain order itself is preserved because
/// it determines payload layout.
///
/// Extents have [`DIMENSION_DATA_TYPE`], full domain shape, the storage memory space, and projected sharding. Their
/// layouts are unspecified. This structural description does not establish numerical extent bounds; the checked
/// [`RaggedNew`](crate::RaggedNew) boundary establishes them. Reconstruction may omit checks only when an actual
/// checked producer proves the exact field relationships.
///
/// [`RaggedDimension`] entries of array values reference their indexed identities. A dimension value defines its
/// represented identity, independently of any [`RaggedDimension`] entries describing its domain. Indexed identities
/// cannot also describe ordinary scalar dimensions in one identity scope (i.e., a storage capacity never proves a
/// scalar logical extent). Ragged values do not implicitly project to [`ArrayType`], admit constants or references,
/// or support zero/one construction yet.
///
/// # Example
///
/// Consider `f32` storage with shape `[2, 3, 4]`, whose axes are `batch`, `time`, and `element`. The number of live
/// elements varies at each `batch`/`time` coordinate. Let `n` be a [`DimensionVariable`] with bounds `[0, 5)`,
/// admitting lengths `0` through `4`. The single [`RaggedDimension`] has `axis=2`, `dimension=n`, and
/// `extent_axes=[0, 1]`. The corresponding ragged type describes the following representation:
///
/// | Accessor                                       | Description                                         |
/// |------------------------------------------------|-----------------------------------------------------|
/// | [`storage_type`](Self::storage_type)           | Padded array type `f32[2, 3, 4]`.                   |
/// | [`value_kind`](Self::value_kind)               | [`RaggedValueKind::Array`].                         |
/// | [`ragged_dimensions`](Self::ragged_dimensions) | The [`RaggedDimension`] for the `element` axis.     |
/// | [`logical_shape`](Self::logical_shape)         | `[2, 3, n]`, retaining the `batch` and `time` axes. |
/// | [`extent_types`](Self::extent_types)           | One dense integer array type: `i64[2, 3]`.          |
///
/// A corresponding [`RaggedValue`](crate::RaggedValue) contains the padded `f32[2, 3, 4]` storage and one `i64[2, 3]`
/// extent payload. For example:
///
/// ```text
/// [[1, 4, 2],
///  [3, 0, 1]]
/// ```
///
/// Here, `storage[b, t, k]` is live when `k < extents[b, t]`. The slice at `batch=0` and `time=1` has four live
/// elements, while the slice at `batch=1` and `time=1` is entirely padding. The symbolic `n` records the indexed
/// lengths' shared identity and bounds; the concrete length at each coordinate belongs to the value's extent payload.
/// The same type can describe different length payloads across executions; occurrences sharing `n` within an execution
/// must still agree on their live extents.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct RaggedType {
    /// Shared immutable validated representation; clones retain its [`RaggedDimension`] and payload-type allocations.
    payload: Arc<RaggedTypePayload>,
}

impl RaggedType {
    /// Constructs a new structurally validated [`RaggedType`], canonicalizing [`RaggedDimension`] order.
    ///
    /// # Errors
    ///
    /// Returns [`TypeError`] for an array value with no ragged dimensions, non-integer dimension storage, invalid or
    /// duplicate axes, unbounded ragged dimensions, storage outside the declared maximum extent, cyclic or out-of-order
    /// ragged domain dependencies, or a projected sharding that cannot be represented.
    pub fn new(
        storage_type: ArrayType,
        value_kind: RaggedValueKind,
        mut ragged_dimensions: Vec<RaggedDimension>,
    ) -> Result<Self, TypeError> {
        match &value_kind {
            RaggedValueKind::Array if ragged_dimensions.is_empty() => {
                return Err(TypeError::invalid("a ragged array type requires at least one ragged dimension"));
            }
            RaggedValueKind::Array => {}
            RaggedValueKind::Dimension(_) if storage_type.data_type() != DIMENSION_DATA_TYPE => {
                return Err(TypeError::invalid(format!(
                    "ragged dimension storage must have data type `{}` but got `{}`",
                    DIMENSION_DATA_TYPE,
                    storage_type.data_type(),
                )));
            }
            RaggedValueKind::Dimension(_) => {}
        }

        ragged_dimensions.sort_by_key(RaggedDimension::axis);
        let mut logical_dimensions = storage_type.shape().dimensions().to_vec();
        let mut extent_types = Vec::with_capacity(ragged_dimensions.len());
        for (index, descriptor) in ragged_dimensions.iter().enumerate() {
            if descriptor.axis >= storage_type.rank() {
                return Err(TypeError::invalid(format!(
                    "ragged axis {} is out of bounds for rank-{} storage",
                    descriptor.axis,
                    storage_type.rank(),
                )));
            }

            if index > 0 && ragged_dimensions[index - 1].axis == descriptor.axis {
                return Err(TypeError::invalid(format!(
                    "ragged storage axis {} is described more than once",
                    descriptor.axis
                )));
            }

            descriptor.dimension.bounds().representable_extent_range()?;

            let maximum = DimensionType::from(descriptor.dimension.clone()).maximum_extent().ok_or_else(|| {
                TypeError::invalid(format!(
                    "ragged dimension `{}` must have a finite upper bound",
                    descriptor.dimension
                ))
            })?;

            if storage_type.dimension(descriptor.axis).value() != Some(maximum) {
                return Err(TypeError::invalid(format!(
                    "ragged axis {} requires storage extent {} for dimension `{}` but got `{}`",
                    descriptor.axis,
                    maximum,
                    descriptor.dimension,
                    storage_type.dimension(descriptor.axis),
                )));
            }

            for (extent_axis_index, axis) in descriptor.extent_axes.iter().copied().enumerate() {
                if axis >= storage_type.rank() {
                    return Err(TypeError::invalid(format!(
                        "ragged axis {} domain axis {} is out of bounds for rank-{} storage",
                        descriptor.axis,
                        axis,
                        storage_type.rank(),
                    )));
                }

                if descriptor.extent_axes[..extent_axis_index].contains(&axis) {
                    return Err(TypeError::invalid(format!(
                        "ragged axis {} repeats domain axis {}",
                        descriptor.axis, axis,
                    )));
                }

                if ragged_dimensions.iter().any(|dependency| dependency.axis == axis) && axis >= descriptor.axis {
                    return Err(TypeError::invalid(format!(
                        "ragged axis {} depends on ragged axis {}; dependencies must precede their consumer",
                        descriptor.axis, axis,
                    )));
                }
            }

            logical_dimensions[descriptor.axis] = descriptor.dimension.clone().into();
            let shape = Shape::new(descriptor.extent_axes.iter().map(|axis| storage_type.dimension(*axis)).collect());
            let sharding = storage_type
                .sharding()
                .map(|sharding| {
                    sharding.with_dimensions(
                        descriptor.extent_axes.iter().map(|axis| sharding.dimensions()[*axis].clone()).collect(),
                    )
                })
                .transpose()
                .map_err(|error| TypeError::invalid(format!("ragged extent sharding is invalid: {error}")))?;
            extent_types.push(
                ArrayType::new(DIMENSION_DATA_TYPE, shape)
                    .with_memory(storage_type.memory())
                    .with_sharding(sharding)
                    .map_err(|error| TypeError::invalid(format!("ragged extent type is invalid: {error}")))?,
            );
        }

        let r#type = Self {
            payload: Arc::new(RaggedTypePayload {
                storage_type,
                value_kind,
                ragged_dimensions,
                logical_shape: Shape::new(logical_dimensions),
                extent_types,
            }),
        };

        Self::validate_identity_scope([&r#type])?;
        Ok(r#type)
    }

    /// Returns the [`ArrayType`] of the physical storage, including its element data type, layout, memory space, and
    /// sharding. Each ragged axis has its dimension's maximum admitted extent; ordinary axes retain their original
    /// dimensions. Use [`Self::logical_shape`] for the shape with symbolic ragged dimensions in place of capacities.
    /// For dimension values, this storage contains the represented integer extents themselves, as described by
    /// [`Self::value_kind`].
    #[inline]
    pub fn storage_type(&self) -> &ArrayType {
        &self.payload.storage_type
    }

    /// Returns whether this representation contains array elements or indexed dimension values.
    /// [`RaggedValueKind::Array`] uses storage for numerical elements and separate payloads for ragged-axis extents.
    /// [`RaggedValueKind::Dimension`] uses integer storage for the dimension values themselves; its [`DimensionType`]
    /// supplies their bounds and optional symbolic identity. In either case, [`Self::ragged_dimensions`] describes
    /// ragged axes of that storage's coordinate shape, independently of the represented value kind.
    #[inline]
    pub fn value_kind(&self) -> &RaggedValueKind {
        &self.payload.value_kind
    }

    /// Returns the [`RaggedDimension`] entries in increasing storage-axis order, regardless of their construction
    /// order. Each [`RaggedDimension`] identifies a storage axis, its logical extent identity, and the ordered storage
    /// axes that index its extent payload. The [`RaggedDimension`] at index `i` corresponds to [`Self::extent_types`]
    /// entry `i`. Array values require at least one [`RaggedDimension`]; dimension values may have none when their
    /// coordinate shape is entirely ordinary. Dependencies on other ragged storage axes must precede the described
    /// axis; those axes' sibling [`RaggedDimension`] entries determine which extent-payload coordinates are live.
    #[inline]
    pub fn ragged_dimensions(&self) -> &[RaggedDimension] {
        &self.payload.ragged_dimensions
    }

    /// Returns the logical shape obtained by replacing each ragged storage axis's padded capacity with the
    /// [`DimensionVariable`] of its [`RaggedDimension`]. The shape has the same rank and axis order as
    /// [`Self::storage_type`], including all axes used to index extent payloads. Its ragged symbolic dimensions
    /// describe indexed extents; their concrete values at each coordinate reside in the value's payloads. For
    /// dimension values, this is the shape indexing those values; the type of the dimensions themselves is carried
    /// by [`Self::value_kind`]. The shape is computed once during construction and returned by reference here.
    #[inline]
    pub fn logical_shape(&self) -> &Shape {
        &self.payload.logical_shape
    }

    /// Returns one exact dense extent-payload [`ArrayType`] per [`Self::ragged_dimensions`] entry, in the same order.
    /// Each payload uses [`DIMENSION_DATA_TYPE`] and has the shape selected from [`Self::storage_type`] by the
    /// corresponding [`RaggedDimension::extent_axes`] mapping, preserving axis order. An empty mapping produces a
    /// scalar. Payloads share the storage memory space, have sharding projected onto their selected axes, and have
    /// no layout constraint. They cover the full padded shapes of their coordinate axes; sibling [`RaggedDimension`]
    /// entries determine live coordinates when those axes are themselves ragged. Compact or singleton payloads cannot
    /// replace a wider shape. These types establish representation structure; checked producers must still validate
    /// live numerical extents.
    #[inline]
    pub fn extent_types(&self) -> &[ArrayType] {
        &self.payload.extent_types
    }

    /// Returns each indexed [`DimensionVariable`] together with the storage axes used to look up its runtime extent.
    /// An entry `(n, [0, 2])` means that `n` has an extent at each coordinate of storage axes `0` and `2`, in that
    /// order. The returned integers are axis positions, not coordinates or extent values. An empty axis list means
    /// one shared extent, represented by a scalar payload.
    ///
    /// This function first yields one entry per [`RaggedDimension`], in increasing storage-axis order, pairing its
    /// [`dimension`](RaggedDimension::dimension) with its [`extent_axes`](RaggedDimension::extent_axes). For
    /// [`RaggedValueKind::Dimension`], it also yields the represented dimension's variable, if any, paired with every
    /// storage axis in order: the storage itself holds those dimension values. A static represented dimension adds
    /// no entry. Repeated identities are retained. Ordinary storage-shape variables are scalar extent identities of
    /// [`Self::storage_type`] and are not reported here.
    ///
    /// # Example
    ///
    /// Consider storage with shape `[2, 4]`, indexed by `batch` and `element`. A single [`RaggedDimension`] describes
    /// axis `1` using identity `r` with bounds `[0, 5)` and `extent_axes=[0]`. For [`RaggedValueKind::Array`], this
    /// function yields:
    ///
    /// ```text
    /// (r, [0])
    /// ```
    ///
    /// Here, `r[b]` gives the number of live elements in batch row `b`. If the same shape instead holds integer
    /// dimension values with kind `Dimension(n)` for a symbolic `n`, the function yields:
    ///
    /// ```text
    /// (r, [0])
    /// (n, [0, 1])
    /// ```
    ///
    /// Now `r[b]` gives the number of live dimension values in row `b`, and `n[b, j]` is the dimension value stored
    /// at each live coordinate `[b, j]`. This function reports the identities and their indexing axes; it does not
    /// read either payload or determine which coordinates are live.
    pub(crate) fn indexed_identities(&self) -> impl Iterator<Item = (&DimensionVariable, Vec<usize>)> {
        self.payload
            .ragged_dimensions
            .iter()
            .map(|descriptor| (&descriptor.dimension, descriptor.extent_axes.clone()))
            .chain(match &self.payload.value_kind {
                RaggedValueKind::Array => None,
                RaggedValueKind::Dimension(dimension) => {
                    dimension.variable().map(|identity| (identity, (0..self.payload.storage_type.rank()).collect()))
                }
            })
    }

    /// Compares the indexed extent requirements of a declared type with those of a supplied
    /// actual type, returning [`DimensionRefinementFact`]s that relate their identities. For example, a
    /// [`Correspondence(n, m)`](DimensionRefinementFact::Correspondence) requires the declared indexed
    /// extents `n` and supplied indexed extents `m` to agree at corresponding live coordinates.
    ///
    /// The types must have the same number of [`RaggedDimension`] entries, with matching storage axes and ordered
    /// [`extent_axes`](RaggedDimension::extent_axes) mappings. Each actual identity's bounds must fit within the
    /// declared identity's bounds. For each pair, this function records a correspondence, in storage-axis order.
    /// [`DimensionComparison::Refinement`] additionally requires the same identity on both sides;
    /// [`DimensionComparison::LocalDeclarations`] and [`DimensionComparison::IdentityRenaming`] allow distinct
    /// identities and record their relationship according to the selected comparison mode.
    ///
    /// Both types must also have the same value kind. For [`RaggedValueKind::Dimension`], facts about
    /// the represented dimension values follow the ragged-axis correspondences. These are derived by
    /// [`DimensionTypeRefinements::refinement_facts`], using `LocalDeclarations` for both `Refinement` and
    /// `LocalDeclarations`, or preserving `IdentityRenaming`. Thus, represented dimension values can have distinct
    /// identities even in `Refinement` mode. Their facts describe the values at live storage coordinates, separately
    /// from the extents of the axes indexing those values.
    ///
    /// This function inspects types only. Any equality requirements in the returned facts still depend on the required
    /// runtime checks succeeding. Dense storage-type validation is the caller's responsibility, and padded storage
    /// capacities never establish exact logical extents here.
    ///
    /// # Example
    ///
    /// Consider two array-value types with storage `f32[2, 4]`. Each has one [`RaggedDimension`] for axis `1`, with
    /// `extent_axes=[0]`. The declared type uses identity `n`, the actual type uses a distinct identity `m`, and both
    /// identities admit lengths `0` through `4`. With `LocalDeclarations`, this function returns:
    ///
    /// ```text
    /// [Correspondence(n, m)]
    /// ```
    ///
    /// This means that `n[b]` and `m[b]` must agree for each batch row `b` once the supplied value satisfies the
    /// declaration's runtime requirements. It does not imply that either length is `4`: that is only the padded
    /// capacity. With `Refinement`, the same comparison fails because `n` and `m` are distinct identities.
    ///
    /// # Errors
    ///
    /// Returns [`TypeError`] for mismatched ragged axes, extent-axis mappings, or value kinds, or when actual
    /// identities or represented dimension types do not satisfy the declared bounds and comparison rules.
    pub(crate) fn indexed_refinement_facts<'o>(
        declared: &'o Self,
        actual: &'o Self,
        comparison: DimensionComparison,
    ) -> Result<Vec<DimensionRefinementFact<'o>>, TypeError> {
        if declared.payload.ragged_dimensions.len() != actual.payload.ragged_dimensions.len() {
            return Err(TypeError::invalid("ragged types have different numbers of ragged dimensions"));
        }

        let mut facts = Vec::new();
        for (declared, actual) in declared.payload.ragged_dimensions.iter().zip(&actual.payload.ragged_dimensions) {
            if declared.axis != actual.axis || declared.extent_axes != actual.extent_axes {
                return Err(TypeError::invalid("ragged types have different occurrence axes or dependency domains"));
            }
            if !declared.dimension.bounds().contains_bounds(actual.dimension.bounds())
                || (comparison == DimensionComparison::Refinement && declared.dimension != actual.dimension)
            {
                return Err(TypeError::invalid(format!(
                    "indexed dimension `{}` does not refine declared indexed dimension `{}`",
                    actual.dimension, declared.dimension,
                )));
            }
            facts.push(DimensionRefinementFact::Correspondence(&declared.dimension, &actual.dimension));
        }

        match (&declared.payload.value_kind, &actual.payload.value_kind) {
            (RaggedValueKind::Array, RaggedValueKind::Array) => {}
            (RaggedValueKind::Dimension(declared), RaggedValueKind::Dimension(actual)) => {
                let comparison = match comparison {
                    DimensionComparison::Refinement | DimensionComparison::LocalDeclarations => {
                        DimensionComparison::LocalDeclarations
                    }
                    DimensionComparison::IdentityRenaming => DimensionComparison::IdentityRenaming,
                };
                facts.extend(
                    DimensionTypeRefinements::refinement_facts(declared, actual, comparison)
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
            (RaggedValueKind::Array, RaggedValueKind::Dimension(_))
            | (RaggedValueKind::Dimension(_), RaggedValueKind::Array) => {
                return Err(TypeError::invalid("ragged types have different value kinds"));
            }
        }

        Ok(facts)
    }

    /// Rejects an indexed identity (as reported by [`Self::indexed_identities`]) that occurs more than once across the
    /// provided types of one signature. Structural renaming relates identities by their types alone, while repeated
    /// indexed occurrences carry independent extent payloads whose equality only a runtime check can establish.
    pub(crate) fn validate_unrepeated_indexed_identities<'o>(
        types: impl IntoIterator<Item = &'o Self>,
    ) -> Result<(), TypeError> {
        let mut indexed = Vec::new();
        for r#type in types {
            for (identity, _) in r#type.indexed_identities() {
                if indexed.contains(&identity) {
                    return Err(TypeError::invalid(format!(
                        "indexed dimension `{identity}` has repeated extent occurrences \
                         whose equality requires runtime checking",
                    )));
                }
                indexed.push(identity);
            }
        }
        Ok(())
    }
}

impl Display for RaggedType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "ragged<{}", self.payload.storage_type)?;
        if let RaggedValueKind::Dimension(dimension) = &self.payload.value_kind {
            write!(formatter, "; item={dimension}")?;
        }
        for descriptor in &self.payload.ragged_dimensions {
            write!(formatter, "; {}:{} over {:?}", descriptor.axis, descriptor.dimension, descriptor.extent_axes)?;
        }
        write!(formatter, ">")
    }
}

impl Type for RaggedType {
    type Identity = DimensionVariable;
    type Refinements = RaggedTypeRefinements;

    fn validate_identity_scope<Types: IntoIterator<Item: Borrow<Self>>>(types: Types) -> Result<(), TypeError> {
        // Storage-shape identities are scalar extents, recorded before the indexed identities of the same member.
        let mut scope = RaggedIdentityScope::default();
        for (index, r#type) in types.into_iter().enumerate() {
            let r#type = r#type.borrow();
            for (_, identity) in r#type.storage_type().identities() {
                scope.record_scalar(identity)?;
            }
            scope.record_ragged(index, r#type)?;
        }
        Ok(())
    }

    #[inline]
    fn validate_zero(&self) -> Result<(), TypeError> {
        Err(TypeError::invalid("zero construction for first-class ragged types is not supported yet"))
    }

    #[inline]
    fn validate_one(&self) -> Result<(), TypeError> {
        Err(TypeError::invalid("one construction for first-class ragged types is not supported yet"))
    }

    fn identities(&self) -> impl Iterator<Item = (TypeIdentityPosition, &Self::Identity)> {
        self.payload
            .storage_type
            .identities()
            .chain(
                self.payload
                    .ragged_dimensions
                    .iter()
                    .map(|descriptor| (TypeIdentityPosition::Reference, &descriptor.dimension)),
            )
            .chain(match &self.payload.value_kind {
                RaggedValueKind::Array => None,
                RaggedValueKind::Dimension(dimension) => {
                    dimension.variable().map(|identity| (TypeIdentityPosition::Definition, identity))
                }
            })
    }

    fn derive_identity_renaming<
        'o,
        D: IntoIterator<Item = &'o Self, IntoIter: ExactSizeIterator>,
        A: IntoIterator<Item = &'o Self, IntoIter: ExactSizeIterator>,
    >(
        declared: D,
        actual: A,
    ) -> Result<TypeIdentityRenaming<Self::Identity>, TypeError>
    where
        Self: 'o,
    {
        derive_signature_identity_renaming(
            declared,
            actual,
            |declared, actual| {
                ArrayTypeRefinements::refinement_facts(
                    declared.storage_type(),
                    actual.storage_type(),
                    DimensionComparison::IdentityRenaming,
                )
                .chain(RaggedDimensionRefinements::refinement_facts(
                    declared,
                    actual,
                    DimensionComparison::IdentityRenaming,
                ))
            },
            |declared, actual| {
                Self::validate_identity_scope(declared.iter().copied())?;
                Self::validate_identity_scope(actual.iter().copied())?;
                Self::validate_unrepeated_indexed_identities(declared.iter().copied())?;
                Self::validate_unrepeated_indexed_identities(actual.iter().copied())
            },
            DimensionRefinements::derive_identity_renaming,
        )
    }

    fn rename_identities(&self, renaming: &TypeIdentityRenaming<Self::Identity>) -> Result<Self, TypeError> {
        let value_kind = match &self.payload.value_kind {
            RaggedValueKind::Array => RaggedValueKind::Array,
            RaggedValueKind::Dimension(dimension) => RaggedValueKind::Dimension(dimension.rename_identities(renaming)?),
        };
        Self::new(
            self.payload.storage_type.rename_identities(renaming)?,
            value_kind,
            self.payload
                .ragged_dimensions
                .iter()
                .map(|descriptor| {
                    Ok(RaggedDimension::new(
                        descriptor.axis,
                        descriptor.dimension.renamed(renaming)?,
                        descriptor.extent_axes.clone(),
                    ))
                })
                .collect::<Result<Vec<_>, TypeError>>()?,
        )
    }

    #[inline]
    fn is_compatible_with(&self, other: &Self) -> bool {
        self == other
    }

    #[inline]
    fn is_refined_by(&self, other: &Self) -> bool {
        self.payload.storage_type.is_refined_by(other.storage_type())
            && Self::indexed_refinement_facts(self, other, DimensionComparison::Refinement).is_ok()
    }

    #[inline]
    fn is_scalar(&self) -> bool {
        false
    }

    #[inline]
    fn is_complex(&self) -> bool {
        match self.payload.value_kind {
            RaggedValueKind::Array => self.payload.storage_type.is_complex(),
            RaggedValueKind::Dimension(_) => false,
        }
    }
}

impl ReferenceMemberType for RaggedType {
    type Referent = NoReferent;

    #[inline]
    fn referent(&self) -> Option<&NoReferent> {
        None
    }

    #[inline]
    fn as_referent(&self) -> Option<&NoReferent> {
        None
    }
}

impl From<NoReferent> for RaggedType {
    fn from(referent: NoReferent) -> Self {
        match referent {}
    }
}

impl From<ReferenceType<NoReferent>> for RaggedType {
    fn from(r#type: ReferenceType<NoReferent>) -> Self {
        match *r#type.referent() {}
    }
}

// TODO(eaplatanios): Review from here onwards.

/// Accumulator that validates one identity scope containing [`RaggedType`]s, possibly alongside members whose
/// identities are all scalar extents (e.g., the arrays and first-class dimensions of a composite universe). Callers
/// record each member's scalar extent identities with [`Self::record_scalar`] and then, for a ragged member, its
/// indexed identities with [`Self::record_ragged`], in member order.
///
/// An identity cannot describe both a scalar extent and indexed ragged extents, because a padded storage capacity never
/// proves a scalar logical extent. Occurrences of one indexed identity must also index compatible logical coordinate
/// domains. Reusing the same storage axes within one carrier is itself an explicit coordinate correspondence. Separate
/// carriers, or different axes of one carrier, correspond only through their coordinate dimensions, which must then be
/// unique so that equal physical positions alone never establish a correspondence.
#[derive(Debug, Default)]
pub(crate) struct RaggedIdentityScope {
    /// Recorded scalar extent identities, in first-occurrence order.
    scalar_identities: Vec<DimensionVariable>,

    /// Recorded indexed identities, in first-occurrence order, each with the logical coordinate dimensions, member
    /// index, and storage axes of its first occurrence.
    indexed_identities: Vec<(DimensionVariable, Vec<Dimension>, usize, Vec<usize>)>,
}

impl RaggedIdentityScope {
    /// Records one scalar extent identity of the scope.
    ///
    /// # Errors
    ///
    /// Returns [`TypeError`] if `identity` was already recorded as an indexed identity.
    pub(crate) fn record_scalar(&mut self, identity: &DimensionVariable) -> Result<(), TypeError> {
        if self.indexed_identities.iter().any(|(indexed, ..)| indexed == identity) {
            return Err(TypeError::invalid(format!(
                "dimension identity `{identity}` is used as both a scalar extent and indexed ragged extents",
            )));
        }
        if !self.scalar_identities.contains(identity) {
            self.scalar_identities.push(identity.clone());
        }
        Ok(())
    }

    /// Records the indexed identities of the ragged member at position `index` of the scope. The scalar extent
    /// identities of its storage shape must already have been recorded with [`Self::record_scalar`].
    ///
    /// # Errors
    ///
    /// Returns [`TypeError`] if an indexed identity was already recorded as a scalar extent, or if its coordinate
    /// domain is ambiguous or incompatible with that of an earlier occurrence.
    pub(crate) fn record_ragged(&mut self, index: usize, r#type: &RaggedType) -> Result<(), TypeError> {
        for (identity, domain) in r#type.indexed_identities() {
            if self.scalar_identities.contains(identity) {
                return Err(TypeError::invalid(format!(
                    "dimension identity `{identity}` is used as both a scalar extent and indexed ragged extents",
                )));
            }
            let coordinates = domain.iter().map(|axis| r#type.logical_shape().dimension(*axis)).collect::<Vec<_>>();
            if let Some((_, previous, previous_index, previous_axes)) =
                self.indexed_identities.iter().find(|(candidate, ..)| candidate == identity)
            {
                // Reusing an explicit axis correspondence within one carrier is proof. Separate carriers need
                // unique nominal/static coordinate dimensions; equal physical positions alone are insufficient.
                if *previous_index != index || previous_axes != &domain {
                    if previous
                        .iter()
                        .enumerate()
                        .any(|(position, coordinate)| previous[..position].contains(coordinate))
                        || coordinates
                            .iter()
                            .enumerate()
                            .any(|(position, coordinate)| coordinates[..position].contains(coordinate))
                    {
                        return Err(TypeError::invalid(format!(
                            "indexed dimension `{identity}` has ambiguous repeated domain coordinates"
                        )));
                    }
                    if coordinates.len() != previous.len()
                        || coordinates.iter().any(|coordinate| !previous.contains(coordinate))
                    {
                        return Err(TypeError::invalid(format!(
                            "indexed dimension `{identity}` has incompatible dependency domains"
                        )));
                    }
                }
            } else {
                self.indexed_identities.push((identity.clone(), coordinates, index, domain));
            }
        }
        Ok(())
    }
}

/// Indexed identity scope of [`RaggedType`] refinements. Identities referenced by [`RaggedDimension`] entries and
/// those defined by dimension-item values (i.e., those reported by [`RaggedType::indexed_identities`]) form indexed
/// classes, deliberately isolated from scalar extent facts, together with their logical coordinate domains.
///
/// This type owns every ragged-specific part of a refinement but not the scalar scope. Ragged storage shapes contribute
/// ordinary scalar facts to an [`ArrayTypeRefinements`] scope owned by the caller: [`RaggedTypeRefinements`] uses one
/// for the storage types of a ragged signature, while a composite universe may share one across all of its member
/// kinds. Functions that relate both scopes take that scalar scope as a parameter, so callers must always pass the same
/// scope that they used for construction.
///
/// Exact indexed facts may refine dimension items, but they never erase [`RaggedDimension`] entries or change storage
/// bounds. Repeated indexed payloads retain runtime-check obligations even when their nominal types agree, and a
/// scalar/indexed alias or an ambiguous coordinate correspondence is rejected before either environment can supply an
/// invalid substitution.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct RaggedDimensionRefinements {
    /// Indexed identity correspondences, deliberately isolated from scalar extent facts.
    indexed: DimensionRefinements,

    /// Declared indexed identities and their logical coordinate dimensions, in first-occurrence order.
    declared_domains: Vec<(DimensionVariable, Vec<Dimension>)>,

    /// Actual indexed identities and their logical coordinate dimensions, in first-occurrence order.
    actual_domains: Vec<(DimensionVariable, Vec<Dimension>)>,

    /// Mixed-scope first-observation order as `(indexed, class_index)` pairs, where `class_index` refers to the indexed
    /// classes or to the caller's scalar classes. Scopes without indexed identities use the ordinary scalar order
    /// directly, keeping this vector empty.
    identity_class_order: Vec<(bool, usize)>,
}

impl RaggedDimensionRefinements {
    /// Builds the indexed scope of matched declared and actual [`RaggedType`] pairs using
    /// [`DimensionComparison::LocalDeclarations`] facts, and records their coordinate domains. The pairs must belong to
    /// signatures whose identity scopes have already been validated. The mixed-scope class order starts empty; record
    /// it with [`Self::with_identity_class_order`].
    pub(crate) fn from_local_declarations<'o>(
        pairs: impl IntoIterator<Item = (&'o RaggedType, &'o RaggedType), IntoIter: Clone>,
    ) -> Result<Self, TypeError> {
        let pairs = pairs.into_iter();
        let indexed = DimensionRefinements::from_refinement_facts(pairs.clone().flat_map(|(declared, actual)| {
            Self::refinement_facts(declared, actual, DimensionComparison::LocalDeclarations)
        }))?;
        Ok(Self {
            indexed,
            declared_domains: Self::domains(pairs.clone().map(|(declared, _)| declared)),
            actual_domains: Self::domains(pairs.map(|(_, actual)| actual)),
            identity_class_order: Vec::new(),
        })
    }

    /// Records the first-observation order of the scalar classes of `scalar` and of the indexed classes of this scope,
    /// returning the updated scope. `facts` yields the complete signature's [`DimensionComparison::LocalDeclarations`]
    /// facts in pair order, each tagged with whether it is an indexed fact (`true`) or a scalar fact (`false`). Scopes
    /// without indexed identities keep the ordinary scalar order and do not consume `facts`.
    pub(crate) fn with_identity_class_order<'o>(
        mut self,
        scalar: &ArrayTypeRefinements,
        facts: impl IntoIterator<Item = (bool, Result<DimensionRefinementFact<'o>, TypeError>)>,
    ) -> Result<Self, TypeError> {
        if self.declared_domains.is_empty() && self.actual_domains.is_empty() {
            return Ok(self);
        }
        let mut identity_class_order = Vec::new();
        for (is_indexed, fact) in facts {
            let (identity, declared_scope) = match fact? {
                DimensionRefinementFact::Declared(identity, _)
                | DimensionRefinementFact::Correspondence(identity, _) => (identity, true),
                DimensionRefinementFact::Actual(identity, _) => (identity, false),
            };
            let matches = |class: TypeIdentityClass<'_, DimensionVariable>| {
                if declared_scope { class.declared().contains(identity) } else { class.actual().contains(identity) }
            };
            let class_index = if is_indexed {
                self.indexed.identity_classes().position(matches)
            } else {
                scalar.identity_classes().position(matches)
            }
            .unwrap();
            let entry = (is_indexed, class_index);
            if !identity_class_order.contains(&entry) {
                identity_class_order.push(entry);
            }
        }
        self.identity_class_order = identity_class_order;
        Ok(self)
    }

    /// Returns the indexed [`DimensionRefinementFact`]s for one declared and one actual [`RaggedType`], or their
    /// comparison error, as produced by [`RaggedType::indexed_refinement_facts`]. Storage facts belong to the
    /// caller's scalar scope and are not included.
    pub(crate) fn refinement_facts<'o>(
        declared: &'o RaggedType,
        actual: &'o RaggedType,
        comparison: DimensionComparison,
    ) -> impl Iterator<Item = Result<DimensionRefinementFact<'o>, TypeError>> {
        let (facts, error) = match RaggedType::indexed_refinement_facts(declared, actual, comparison) {
            Ok(facts) => (facts, None),
            Err(error) => (Vec::new(), Some(error)),
        };
        error.into_iter().map(Err).chain(facts.into_iter().map(Ok))
    }

    /// Returns the connected [`TypeIdentityClass`]es of the scalar scope `scalar` and of this indexed scope, in their
    /// mixed first-observation order.
    pub(crate) fn identity_classes<'o>(
        &'o self,
        scalar: &'o ArrayTypeRefinements,
    ) -> impl Iterator<Item = TypeIdentityClass<'o, DimensionVariable>> {
        let mut scalar_classes = scalar.identity_classes();
        let mut indexed_classes = self.indexed.identity_classes();
        let mut order = self.identity_class_order.iter();
        let ordinary_order = self.identity_class_order.is_empty();
        std::iter::from_fn(move || {
            if ordinary_order {
                // Dense scopes retain one linear pass without looking classes up again by position.
                scalar_classes.next().or_else(|| indexed_classes.next())
            } else {
                let (indexed, index) = order.next()?;
                Some(if *indexed {
                    self.indexed.identity_classes().nth(*index).unwrap()
                } else {
                    scalar.identity_classes().nth(*index).unwrap()
                })
            }
        })
    }

    /// Returns `unproven` extended with the declared indexed identities related to a repeated actual indexed
    /// identity, ordered by first occurrence in `declared_identities`.
    ///
    /// Distinct declarations can enter through one repeated actual indexed family. Its payload equality is no stronger
    /// a type-level proof than repeating a declared indexed family, so every related declared identity keeps its
    /// runtime-check obligation.
    ///
    /// # Parameters
    ///
    ///   - `unproven`: Declared identities whose relationships the supplied types do not prove, as computed by
    ///     [`type_identities_needing_runtime_checks`] for the complete signature.
    ///   - `declared_identities`: Every identity of the declared signature, in signature order, with repetitions.
    ///   - `actual`: The ragged members of the actual signature, in signature order.
    pub(crate) fn merge_identities_needing_runtime_checks<'o>(
        &self,
        mut unproven: Vec<DimensionVariable>,
        declared_identities: impl IntoIterator<Item = &'o DimensionVariable>,
        actual: impl IntoIterator<Item = &'o RaggedType>,
    ) -> Vec<DimensionVariable> {
        let mut observed = Vec::new();
        let mut repeated = Vec::new();
        for r#type in actual {
            for (identity, _) in r#type.indexed_identities() {
                if observed.contains(&identity) {
                    repeated.push(identity);
                }
                observed.push(identity);
            }
        }
        for class in self.indexed.identity_classes() {
            if class.actual().iter().any(|identity| repeated.contains(&identity)) {
                for identity in class.declared() {
                    if !unproven.contains(identity) {
                        unproven.push(identity.clone());
                    }
                }
            }
        }
        let mut ordered = Vec::new();
        for identity in declared_identities {
            if unproven.contains(identity) && !ordered.contains(identity) {
                ordered.push(identity.clone());
            }
        }
        ordered
    }

    /// Validates one scalar extent identity of an output signature against the indexed identities of the input
    /// signature in the selected scope.
    ///
    /// # Errors
    ///
    /// Returns [`TypeError`] if `identity` is an indexed identity of that scope.
    pub(crate) fn validate_scalar(&self, identity: &DimensionVariable, declared_scope: bool) -> Result<(), TypeError> {
        let domains = if declared_scope { &self.declared_domains } else { &self.actual_domains };
        if domains.iter().any(|(indexed, _)| indexed == identity) {
            return Err(TypeError::invalid(format!(
                "dimension identity `{identity}` is used as both a scalar extent and indexed ragged extents"
            )));
        }
        Ok(())
    }

    /// Validates the indexed identities of one ragged member of an output signature against the scalar classes of
    /// `scalar` and the coordinate domains of the input signature, both in the selected scope. Coordinate dimensions
    /// are compared after applying the scalar scope's exact facts.
    ///
    /// # Errors
    ///
    /// Returns [`TypeError`] if an indexed identity is a scalar identity of that scope, or if its coordinate domain is
    /// incompatible with, or ambiguous relative to, the recorded input domain.
    pub(crate) fn validate_ragged(
        &self,
        r#type: &RaggedType,
        scalar: &ArrayTypeRefinements,
        declared_scope: bool,
    ) -> Result<(), TypeError> {
        let domains = if declared_scope { &self.declared_domains } else { &self.actual_domains };
        let scalar_classes = scalar.dimension_refinements();
        for (identity, domain) in r#type.indexed_identities() {
            if scalar_classes.identity_classes().any(|class| {
                if declared_scope { class.declared().contains(identity) } else { class.actual().contains(identity) }
            }) {
                return Err(TypeError::invalid(format!(
                    "dimension identity `{identity}` is used as both a scalar extent and indexed ragged extents"
                )));
            }
            if let Some((_, previous)) = domains.iter().find(|(candidate, _)| candidate == identity) {
                let refine = |dimension: &Dimension| {
                    if declared_scope {
                        scalar_classes.refine_declared_dimension(dimension, &[])
                    } else {
                        scalar_classes.refine_actual_dimension(dimension, &[])
                    }
                };
                let previous = previous.iter().map(refine).collect::<Vec<_>>();
                let coordinates =
                    domain.into_iter().map(|axis| refine(&r#type.logical_shape().dimension(axis))).collect::<Vec<_>>();
                if previous.len() != coordinates.len()
                    || previous.iter().any(|coordinate| !coordinates.contains(coordinate))
                {
                    return Err(TypeError::invalid(format!(
                        "indexed dimension `{identity}` has incompatible dependency domains"
                    )));
                }
                if previous.iter().enumerate().any(|(index, coordinate)| previous[..index].contains(coordinate)) {
                    return Err(TypeError::invalid(format!(
                        "indexed dimension `{identity}` has ambiguous repeated domain coordinates"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Validates the indexed facts of one output pair against the declared-scope indexed facts, recording the first
    /// exact extent of an identity in `closed_identities` as [`DimensionRefinements::validate_declared_extent`]
    /// does. Callers validate against a clone, so that output validation never changes the input refinements.
    pub(crate) fn validate_declared_extents(
        &mut self,
        declared: &RaggedType,
        actual: &RaggedType,
        closed_identities: &[DimensionVariable],
    ) -> Result<(), TypeError> {
        Self::refinement_facts(declared, actual, DimensionComparison::Refinement).try_for_each(|fact| match fact? {
            DimensionRefinementFact::Declared(variable, extent) => {
                self.indexed.validate_declared_extent(variable, extent, closed_identities)
            }
            DimensionRefinementFact::Actual(..) | DimensionRefinementFact::Correspondence(..) => Ok(()),
        })
    }

    /// Applies declared-scope facts to a declared [`RaggedType`], refining its storage with the scalar scope `scalar`
    /// and its dimension item with the indexed scope, as described by [`TypeRefinements::refine_declared`].
    pub(crate) fn refine_declared(
        &self,
        r#type: &RaggedType,
        scalar: &ArrayTypeRefinements,
        preserved_identities: &[DimensionVariable],
    ) -> Result<RaggedType, TypeError> {
        let value_kind = match r#type.value_kind() {
            RaggedValueKind::Array => RaggedValueKind::Array,
            RaggedValueKind::Dimension(dimension) => RaggedValueKind::Dimension(DimensionType::from(
                self.indexed.refine_declared_dimension(dimension.dimension(), preserved_identities),
            )),
        };
        // An indexed exact fact does not remove a descriptor or its extent dependencies. Storage remains
        // at the declared bound; changing that representation requires an explicit cast, not refinement.
        RaggedType::new(
            scalar.refine_declared(r#type.storage_type(), preserved_identities)?,
            value_kind,
            r#type.ragged_dimensions().to_vec(),
        )
    }

    /// Applies actual-scope facts to a supplied [`RaggedType`], refining its storage with the scalar scope `scalar`
    /// and its dimension item with the indexed scope, as described by [`TypeRefinements::refine_actual`].
    pub(crate) fn refine_actual(
        &self,
        r#type: &RaggedType,
        scalar: &ArrayTypeRefinements,
        preserved_identities: &[DimensionVariable],
    ) -> Result<RaggedType, TypeError> {
        let value_kind = match r#type.value_kind() {
            RaggedValueKind::Array => RaggedValueKind::Array,
            RaggedValueKind::Dimension(dimension) => RaggedValueKind::Dimension(DimensionType::from(
                self.indexed.refine_actual_dimension(dimension.dimension(), preserved_identities),
            )),
        };
        // As on the declared side, exact indexed extents retain their padded descriptor representation.
        RaggedType::new(
            scalar.refine_actual(r#type.storage_type(), preserved_identities)?,
            value_kind,
            r#type.ragged_dimensions().to_vec(),
        )
    }

    /// Selects a representation for a supplied [`RaggedType`] using the classes of its declared counterpart, retyping
    /// its storage with the scalar scope `scalar` and its indexed identities with the indexed scope, as described by
    /// [`TypeRefinements::retype_actual`].
    pub(crate) fn retype_actual(
        &self,
        declared: &RaggedType,
        actual: &RaggedType,
        scalar: &ArrayTypeRefinements,
        preferred_identities: &[DimensionVariable],
    ) -> Result<RaggedType, TypeError> {
        RaggedType::indexed_refinement_facts(declared, actual, DimensionComparison::LocalDeclarations)?;
        let value_kind = match (declared.value_kind(), actual.value_kind()) {
            (RaggedValueKind::Array, RaggedValueKind::Array) => RaggedValueKind::Array,
            (RaggedValueKind::Dimension(declared), RaggedValueKind::Dimension(actual)) => {
                RaggedValueKind::Dimension(DimensionType::from(self.indexed.retype_actual_dimension(
                    declared.dimension(),
                    actual.dimension(),
                    preferred_identities,
                )))
            }
            // The comparison above already rejects mismatched value kinds with this diagnostic.
            (RaggedValueKind::Array, RaggedValueKind::Dimension(_))
            | (RaggedValueKind::Dimension(_), RaggedValueKind::Array) => {
                return Err(TypeError::invalid("ragged types have different value kinds"));
            }
        };
        let ragged_dimensions = declared
            .ragged_dimensions()
            .iter()
            .zip(actual.ragged_dimensions())
            .map(|(declared, actual)| {
                let selected = self.indexed.retype_actual_dimension(
                    &declared.dimension().clone().into(),
                    &actual.dimension().clone().into(),
                    preferred_identities,
                );
                // A static fact cannot discard a live payload or adapt padded storage. Dynamic selections
                // preserve nominal relationships; construction below rejects incompatible storage bounds.
                let dimension = match selected {
                    Dimension::Dynamic(variable) => variable,
                    Dimension::Static(_) => actual.dimension().clone(),
                };
                RaggedDimension::new(actual.axis(), dimension, actual.extent_axes().to_vec())
            })
            .collect();
        RaggedType::new(
            scalar.retype_actual(declared.storage_type(), actual.storage_type(), preferred_identities)?,
            value_kind,
            ragged_dimensions,
        )
    }

    /// Records the indexed domains of one already validated signature without retaining value or storage payloads.
    fn domains<'o>(types: impl IntoIterator<Item = &'o RaggedType>) -> Vec<(DimensionVariable, Vec<Dimension>)> {
        let mut domains = Vec::new();
        for r#type in types {
            for (identity, domain) in r#type.indexed_identities() {
                if !domains.iter().any(|(candidate, _)| candidate == identity) {
                    domains.push((
                        identity.clone(),
                        domain.into_iter().map(|axis| r#type.logical_shape().dimension(axis)).collect(),
                    ));
                }
            }
        }
        domains
    }
}

/// [`TypeRefinements`] for one complete [`RaggedType`] signature. Storage types follow the [`ArrayTypeRefinements`]
/// rules in one scalar scope shared by every member, so ordinary storage-shape identities relate across members exactly
/// like array axes. Identities referenced by [`RaggedDimension`] entries and those defined by dimension-item values
/// form separate indexed classes with retained logical coordinate domains, and an identity used in both interpretations
/// is rejected. The classes are reported in mixed first-observation order across both scopes.
///
/// Exact indexed facts may refine dimension items but do not erase [`RaggedDimension`] entries or change storage
/// bounds. Repeated indexed payloads retain runtime-check obligations even when their nominal types agree, because the
/// types do not inspect their independent extent payloads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RaggedTypeRefinements {
    /// [`ArrayTypeRefinements`] shared by the storage types of every member.
    storage: ArrayTypeRefinements,

    /// Indexed identity scope of the signature.
    ragged: RaggedDimensionRefinements,
}

impl TypeRefinements<RaggedType> for RaggedTypeRefinements {
    fn new<D: IntoIterator, A: IntoIterator>(declared: D, actual: A) -> Result<Self, TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<RaggedType>,
        A::Item: Borrow<RaggedType>,
    {
        // Ordinary admission retains nominal storage axes and indexed identities; comparing the complete signature then
        // checks the relationships within and between members.
        let declared = declared.into_iter().collect::<Vec<_>>();
        let actual = actual.into_iter().collect::<Vec<_>>();
        type_signature_pairs(declared.iter(), actual.iter())?.try_for_each(|(declared, actual)| {
            let (declared, actual): (&RaggedType, &RaggedType) = (declared.borrow(), actual.borrow());
            ArrayTypeRefinements::refinement_facts(
                declared.storage_type(),
                actual.storage_type(),
                DimensionComparison::Refinement,
            )
            .chain(RaggedDimensionRefinements::refinement_facts(declared, actual, DimensionComparison::Refinement))
            .try_for_each(|refinement_fact| refinement_fact.map(|_| ()))
        })?;
        Self::from_local_declarations(declared, actual)
    }

    fn from_local_declarations<D: IntoIterator, A: IntoIterator>(declared: D, actual: A) -> Result<Self, TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<RaggedType>,
        A::Item: Borrow<RaggedType>,
    {
        let declared = declared.into_iter().collect::<Vec<_>>();
        let actual = actual.into_iter().collect::<Vec<_>>();
        let pairs = type_signature_pairs(declared.iter().map(Borrow::borrow), actual.iter().map(Borrow::borrow))?;
        RaggedType::validate_identity_scope(declared.iter().map(Borrow::borrow))?;
        RaggedType::validate_identity_scope(actual.iter().map(Borrow::borrow))?;
        let storage = DimensionRefinements::from_refinement_facts(pairs.clone().flat_map(|(declared, actual)| {
            ArrayTypeRefinements::refinement_facts(
                declared.storage_type(),
                actual.storage_type(),
                DimensionComparison::LocalDeclarations,
            )
        }))?;
        let ragged = RaggedDimensionRefinements::from_local_declarations(pairs.clone())?;
        let actual = actual.iter().map(|r#type| Some(r#type.borrow())).collect::<Vec<_>>();
        let unproven = ragged.merge_identities_needing_runtime_checks(
            type_identities_needing_runtime_checks::<RaggedType, _>(&declared, &actual)?,
            declared.iter().flat_map(|r#type| r#type.borrow().identities().map(|(_, identity)| identity)),
            actual.iter().flatten().copied(),
        );
        let storage = ArrayTypeRefinements::from(storage.with_identities_needing_runtime_checks(unproven));
        let ragged = ragged.with_identity_class_order(
            &storage,
            pairs.flat_map(|(declared, actual)| {
                ArrayTypeRefinements::refinement_facts(
                    declared.storage_type(),
                    actual.storage_type(),
                    DimensionComparison::LocalDeclarations,
                )
                .map(|fact| (false, fact))
                .chain(
                    RaggedDimensionRefinements::refinement_facts(
                        declared,
                        actual,
                        DimensionComparison::LocalDeclarations,
                    )
                    .map(|fact| (true, fact)),
                )
            }),
        )?;
        Ok(Self { storage, ragged })
    }

    #[inline]
    fn identity_classes<'o>(&'o self) -> impl Iterator<Item = TypeIdentityClass<'o, DimensionVariable>>
    where
        DimensionVariable: 'o,
    {
        self.ragged.identity_classes(&self.storage)
    }

    #[inline]
    fn identities_needing_runtime_checks(&self) -> &[DimensionVariable] {
        self.storage.identities_needing_runtime_checks()
    }

    fn validate_refinement<D: IntoIterator, A: IntoIterator>(
        &self,
        declared: D,
        actual: A,
        closed_identities: &[DimensionVariable],
    ) -> Result<(), TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<RaggedType>,
        A::Item: Borrow<RaggedType>,
    {
        // Output validation uses declared-scope facts and never records actual-scope facts or correspondences.
        let mut storage = self.storage.dimension_refinements().clone();
        let mut ragged = self.ragged.clone();
        let declared = declared.into_iter().collect::<Vec<_>>();
        let actual = actual.into_iter().collect::<Vec<_>>();
        let mut pairs = type_signature_pairs(declared.iter().map(Borrow::borrow), actual.iter().map(Borrow::borrow))?;
        RaggedType::validate_identity_scope(declared.iter().map(Borrow::borrow))?;
        RaggedType::validate_identity_scope(actual.iter().map(Borrow::borrow))?;
        let validate_scope = |r#type: &RaggedType, declared_scope: bool| {
            for (_, identity) in r#type.storage_type().identities() {
                self.ragged.validate_scalar(identity, declared_scope)?;
            }
            self.ragged.validate_ragged(r#type, &self.storage, declared_scope)
        };
        declared.iter().try_for_each(|r#type| validate_scope(r#type.borrow(), true))?;
        actual.iter().try_for_each(|r#type| validate_scope(r#type.borrow(), false))?;
        pairs.try_for_each(|(declared, actual)| {
            ArrayTypeRefinements::refinement_facts(
                declared.storage_type(),
                actual.storage_type(),
                DimensionComparison::Refinement,
            )
            .try_for_each(|refinement_fact| match refinement_fact? {
                DimensionRefinementFact::Declared(variable, extent) => {
                    storage.validate_declared_extent(variable, extent, closed_identities)
                }
                DimensionRefinementFact::Actual(..) | DimensionRefinementFact::Correspondence(..) => Ok(()),
            })?;
            ragged.validate_declared_extents(declared, actual, closed_identities)
        })
    }

    #[inline]
    fn refine_declared(
        &self,
        r#type: &RaggedType,
        preserved_identities: &[DimensionVariable],
    ) -> Result<RaggedType, TypeError> {
        self.ragged.refine_declared(r#type, &self.storage, preserved_identities)
    }

    #[inline]
    fn refine_actual(
        &self,
        r#type: &RaggedType,
        preserved_identities: &[DimensionVariable],
    ) -> Result<RaggedType, TypeError> {
        self.ragged.refine_actual(r#type, &self.storage, preserved_identities)
    }

    #[inline]
    fn retype_actual(
        &self,
        declared: &RaggedType,
        actual: &RaggedType,
        preferred_identities: &[DimensionVariable],
    ) -> Result<RaggedType, TypeError> {
        self.ragged.retype_actual(declared, actual, &self.storage, preferred_identities)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;

    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{Dimension, DimensionBounds};
    use crate::arrays::types::memories::Memory;

    use super::*;

    #[test]
    fn test_ragged_dimension_new() {
        let dimension = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(5)).unwrap());
        let descriptor = RaggedDimension::new(1, dimension.clone(), vec![0]);
        assert_eq!(descriptor, RaggedDimension { axis: 1, dimension, extent_axes: vec![0] });
    }

    #[test]
    fn test_ragged_dimension_axis() {
        let dimension = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(5)).unwrap());
        assert_eq!(RaggedDimension::new(1, dimension, vec![0]).axis(), 1);
    }

    #[test]
    fn test_ragged_dimension_dimension() {
        let dimension = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(5)).unwrap());
        assert_eq!(RaggedDimension::new(1, dimension.clone(), vec![0]).dimension(), &dimension);
    }

    #[test]
    fn test_ragged_dimension_extent_axes() {
        let dimension = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(5)).unwrap());
        assert_eq!(RaggedDimension::new(2, dimension, vec![1, 0]).extent_axes(), &[1, 0]);
    }

    #[test]
    fn test_ragged_type_new() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(5)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::positive(Some(8)).unwrap());
        let storage_type = ArrayType::new_static(DataType::F32, [2, 4, 7]).with_memory(Memory::Host { pinned: false });
        let r#type = RaggedType::new(
            storage_type.clone(),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(2, columns.clone(), vec![1, 0]), RaggedDimension::new(1, rows.clone(), vec![0])],
        )
        .unwrap();
        assert_eq!(r#type.storage_type(), &storage_type);
        assert_eq!(
            r#type.ragged_dimensions(),
            &[RaggedDimension::new(1, rows.clone(), vec![0]), RaggedDimension::new(2, columns.clone(), vec![1, 0])]
        );
        assert_eq!(r#type.logical_shape(), &Shape::new(vec![Dimension::Static(2), rows.into(), columns.into()]));
        assert_eq!(
            r#type.extent_types(),
            &[
                ArrayType::new_static(DIMENSION_DATA_TYPE, [2]).with_memory(storage_type.memory()),
                ArrayType::new_static(DIMENSION_DATA_TYPE, [4, 2]).with_memory(storage_type.memory()),
            ]
        );
    }

    #[test]
    fn test_ragged_type_new_invalid_structure() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(5)).unwrap());
        let storage_type = ArrayType::new_static(DataType::F32, [2, 4]);
        assert_eq!(
            RaggedType::new(storage_type.clone(), RaggedValueKind::Array, vec![]),
            Err(TypeError::invalid("a ragged array type requires at least one ragged dimension"))
        );
        assert_eq!(
            RaggedType::new(
                storage_type.clone(),
                RaggedValueKind::Array,
                vec![RaggedDimension::new(2, rows.clone(), vec![0])]
            ),
            Err(TypeError::invalid("ragged axis 2 is out of bounds for rank-2 storage"))
        );
        assert_eq!(
            RaggedType::new(
                storage_type.clone(),
                RaggedValueKind::Array,
                vec![RaggedDimension::new(1, rows.clone(), vec![0]), RaggedDimension::new(1, rows.clone(), vec![])]
            ),
            Err(TypeError::invalid("ragged storage axis 1 is described more than once"))
        );
        assert_eq!(
            RaggedType::new(
                storage_type.clone(),
                RaggedValueKind::Array,
                vec![RaggedDimension::new(1, rows.clone(), vec![2])]
            ),
            Err(TypeError::invalid("ragged axis 1 domain axis 2 is out of bounds for rank-2 storage"))
        );
        assert_eq!(
            RaggedType::new(
                storage_type.clone(),
                RaggedValueKind::Array,
                vec![RaggedDimension::new(1, rows.clone(), vec![0, 0])]
            ),
            Err(TypeError::invalid("ragged axis 1 repeats domain axis 0"))
        );
        assert_eq!(
            RaggedType::new(
                storage_type.clone(),
                RaggedValueKind::Array,
                vec![RaggedDimension::new(1, rows.clone(), vec![1])]
            ),
            Err(TypeError::invalid("ragged axis 1 depends on ragged axis 1; dependencies must precede their consumer"))
        );
        assert_eq!(
            RaggedType::new(storage_type, RaggedValueKind::Dimension(DimensionType::from(rows)), vec![]),
            Err(TypeError::invalid("ragged dimension storage must have data type `i64` but got `f32`"))
        );
    }

    #[test]
    fn test_ragged_type_new_storage_bounds() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(5)).unwrap());
        assert_eq!(
            RaggedType::new(
                ArrayType::new_static(DataType::F32, [2, 3]),
                RaggedValueKind::Array,
                vec![RaggedDimension::new(1, rows.clone(), vec![0])]
            ),
            Err(TypeError::invalid("ragged axis 1 requires storage extent 4 for dimension `rows` but got `3`"))
        );
        assert_eq!(
            RaggedType::new(
                ArrayType::new_static(DataType::F32, [2, 5]),
                RaggedValueKind::Array,
                vec![RaggedDimension::new(1, rows, vec![0])]
            ),
            Err(TypeError::invalid("ragged axis 1 requires storage extent 4 for dimension `rows` but got `5`"))
        );
        let unbounded = DimensionVariable::new("unbounded", DimensionBounds::unbounded());
        assert_eq!(
            RaggedType::new(
                ArrayType::new_static(DataType::F32, [2, 4]),
                RaggedValueKind::Array,
                vec![RaggedDimension::new(1, unbounded, vec![0])]
            ),
            Err(TypeError::invalid("ragged dimension `unbounded` must have a finite upper bound"))
        );
        let empty = DimensionVariable::new("empty", DimensionBounds::non_negative(Some(1)).unwrap());
        assert_eq!(
            RaggedType::new(
                ArrayType::new_static(DataType::F32, [0]),
                RaggedValueKind::Array,
                vec![RaggedDimension::new(0, empty, vec![])]
            )
            .unwrap()
            .extent_types(),
            &[ArrayType::scalar(DIMENSION_DATA_TYPE)]
        );
        let singleton = DimensionVariable::new("singleton", DimensionBounds::positive(Some(2)).unwrap());
        assert_eq!(
            RaggedType::new(
                ArrayType::new_static(DataType::F32, [1]),
                RaggedValueKind::Array,
                vec![RaggedDimension::new(0, singleton, vec![])]
            )
            .unwrap()
            .storage_type(),
            &ArrayType::new_static(DataType::F32, [1])
        );
    }

    #[test]
    fn test_ragged_type_new_ragged_dependencies() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(5)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::non_negative(Some(8)).unwrap());
        assert_eq!(
            RaggedType::new(
                ArrayType::new_static(DataType::F32, [2, 4, 7]),
                RaggedValueKind::Array,
                vec![RaggedDimension::new(1, rows.clone(), vec![2]), RaggedDimension::new(2, columns.clone(), vec![0])]
            ),
            Err(TypeError::invalid("ragged axis 1 depends on ragged axis 2; dependencies must precede their consumer"))
        );
        assert_eq!(
            RaggedType::new(
                ArrayType::new_static(DataType::F32, [2, 4, 7]),
                RaggedValueKind::Array,
                vec![RaggedDimension::new(1, rows, vec![2]), RaggedDimension::new(2, columns, vec![1])]
            ),
            Err(TypeError::invalid("ragged axis 1 depends on ragged axis 2; dependencies must precede their consumer"))
        );
    }

    #[test]
    fn test_ragged_type_new_dimension_items() {
        let unbounded = DimensionType::new("item", DimensionBounds::unbounded());
        let r#type = RaggedType::new(
            ArrayType::new_static(DIMENSION_DATA_TYPE, [2]),
            RaggedValueKind::Dimension(unbounded.clone()),
            vec![],
        )
        .unwrap();
        assert_eq!(r#type.value_kind(), &RaggedValueKind::Dimension(unbounded));
        assert_eq!(r#type.logical_shape(), &Shape::from(vec![2]));
        assert_eq!(r#type.extent_types(), &[]);
        let exact = DimensionType::from(Dimension::Static(3));
        assert_eq!(
            RaggedType::new(ArrayType::scalar(DIMENSION_DATA_TYPE), RaggedValueKind::Dimension(exact.clone()), vec![])
                .unwrap()
                .value_kind(),
            &RaggedValueKind::Dimension(exact)
        );
    }

    #[test]
    fn test_ragged_type_storage_type() {
        let storage_type = ArrayType::new_static(DIMENSION_DATA_TYPE, [2]);
        let r#type = RaggedType::new(
            storage_type.clone(),
            RaggedValueKind::Dimension(DimensionType::from(Dimension::Static(1))),
            vec![],
        )
        .unwrap();
        assert_eq!(r#type.storage_type(), &storage_type);
    }

    #[test]
    fn test_ragged_type_value_kind() {
        let kind = RaggedValueKind::Dimension(DimensionType::from(Dimension::Static(1)));
        assert_eq!(
            RaggedType::new(ArrayType::scalar(DIMENSION_DATA_TYPE), kind.clone(), vec![]).unwrap().value_kind(),
            &kind
        );
    }

    #[test]
    fn test_ragged_type_ragged_dimensions() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(5)).unwrap());
        let descriptor = RaggedDimension::new(1, rows, vec![0]);
        assert_eq!(
            RaggedType::new(
                ArrayType::new_static(DataType::F32, [2, 4]),
                RaggedValueKind::Array,
                vec![descriptor.clone()]
            )
            .unwrap()
            .ragged_dimensions(),
            &[descriptor]
        );
    }

    #[test]
    fn test_ragged_type_logical_shape() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(5)).unwrap());
        assert_eq!(
            RaggedType::new(
                ArrayType::new_static(DataType::F32, [2, 4]),
                RaggedValueKind::Array,
                vec![RaggedDimension::new(1, rows.clone(), vec![0])]
            )
            .unwrap()
            .logical_shape(),
            &Shape::new(vec![Dimension::Static(2), rows.into()])
        );
    }

    #[test]
    fn test_ragged_type_extent_types() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(5)).unwrap());
        assert_eq!(
            RaggedType::new(
                ArrayType::new_static(DataType::F32, [2, 4]),
                RaggedValueKind::Array,
                vec![RaggedDimension::new(1, rows, vec![])]
            )
            .unwrap()
            .extent_types(),
            &[ArrayType::scalar(DIMENSION_DATA_TYPE)]
        );
    }

    #[test]
    fn test_ragged_type_formatting_equality_and_hashing() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(5)).unwrap());
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 4]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows.clone(), vec![0])],
        )
        .unwrap();
        assert_eq!(r#type.to_string(), "ragged<f32[2, 4]; 1:rows over [0]>");
        assert_eq!(r#type.clone(), r#type);
        let other = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 4]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, DimensionVariable::new("rows", rows.bounds()), vec![0])],
        )
        .unwrap();
        assert_ne!(r#type, other);
        let types = HashMap::from([(r#type.clone(), "first")]);
        assert_eq!(types.get(&r#type.clone()), Some(&"first"));
        assert_eq!(types.get(&other), None);
        let item = RaggedType::new(
            ArrayType::new_static(DIMENSION_DATA_TYPE, [2]),
            RaggedValueKind::Dimension(DimensionType::from(Dimension::Static(3))),
            vec![],
        )
        .unwrap();
        assert_eq!(item.to_string(), "ragged<i64[2]; item=dimension<3>>");
    }

    #[test]
    fn test_ragged_type_validate_identity_scope() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(5)).unwrap());
        let columns = DimensionVariable::new("columns", rows.bounds());
        let indexed = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 4]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows.clone(), vec![0])],
        )
        .unwrap();
        assert_eq!(RaggedType::validate_identity_scope([&indexed, &indexed]), Ok(()));

        // Storage-shape identities are scalar extents and cannot also describe indexed extents in the same scope,
        // including within the single type validated by construction.
        let scalar_storage = ArrayType::new(DataType::F32, Shape::new(vec![rows.clone().into(), Dimension::Static(4)]));
        let scalar = RaggedType::new(
            scalar_storage.clone(),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, columns, vec![0])],
        )
        .unwrap();
        let error =
            TypeError::invalid("dimension identity `rows` is used as both a scalar extent and indexed ragged extents");
        assert_eq!(RaggedType::validate_identity_scope([&scalar, &indexed]), Err(error.clone()));
        assert_eq!(RaggedType::validate_identity_scope([&indexed, &scalar]), Err(error.clone()));
        assert_eq!(
            RaggedType::new(
                scalar_storage,
                RaggedValueKind::Array,
                vec![RaggedDimension::new(1, rows.clone(), vec![0])]
            ),
            Err(error)
        );

        // Separate carriers must index compatible coordinate domains through unique coordinate dimensions.
        let different = RaggedType::new(
            ArrayType::new_static(DataType::F32, [3, 4]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows.clone(), vec![0])],
        )
        .unwrap();
        assert_eq!(
            RaggedType::validate_identity_scope([&indexed, &different]),
            Err(TypeError::invalid("indexed dimension `rows` has incompatible dependency domains"))
        );
        let ambiguous = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 2, 4]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(2, rows, vec![0, 1])],
        )
        .unwrap();
        assert_eq!(
            RaggedType::validate_identity_scope([&ambiguous, &ambiguous]),
            Err(TypeError::invalid("indexed dimension `rows` has ambiguous repeated domain coordinates"))
        );
    }

    #[test]
    fn test_ragged_type_identities_and_renaming() {
        let bounds = DimensionBounds::non_negative(Some(5)).unwrap();
        let first = DimensionVariable::new("first", bounds);
        let second = DimensionVariable::new("second", bounds);
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [4, 4]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(0, first.clone(), vec![]), RaggedDimension::new(1, second.clone(), vec![0])],
        )
        .unwrap();
        assert_eq!(
            r#type.identities().map(|(position, identity)| (position, identity.clone())).collect::<Vec<_>>(),
            vec![(TypeIdentityPosition::Reference, first.clone()), (TypeIdentityPosition::Reference, second.clone())]
        );
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(first.clone(), second.clone()).unwrap();
        renaming.insert(second.clone(), first.clone()).unwrap();
        let renamed = r#type.rename_identities(&renaming).unwrap();
        assert_eq!(
            renamed.ragged_dimensions(),
            &[RaggedDimension::new(0, second, vec![]), RaggedDimension::new(1, first, vec![0])]
        );
        assert_eq!(renamed.rename_identities(&renaming), Ok(r#type));
    }

    #[test]
    fn test_ragged_type_derive_identity_renaming() {
        let bounds = DimensionBounds::non_negative(Some(5)).unwrap();
        let [declared_batch, actual_batch, declared_rows, actual_rows] =
            ["declared_batch", "actual_batch", "declared_rows", "actual_rows"]
                .map(|name| DimensionVariable::new(name, bounds));
        let ragged = |batch: &DimensionVariable, rows: &DimensionVariable| {
            RaggedType::new(
                ArrayType::new(DataType::F32, Shape::new(vec![batch.clone().into(), Dimension::Static(4)])),
                RaggedValueKind::Array,
                vec![RaggedDimension::new(1, rows.clone(), vec![0])],
            )
            .unwrap()
        };
        let declared = ragged(&declared_batch, &declared_rows);
        let actual = ragged(&actual_batch, &actual_rows);

        // Storage and indexed identities are renamed together.
        let renaming = RaggedType::derive_identity_renaming([&declared], [&actual]).unwrap();
        assert_eq!(renaming.rename(&declared_batch), actual_batch);
        assert_eq!(renaming.rename(&declared_rows), actual_rows);
        assert_eq!(declared.rename_identities(&renaming), Ok(actual.clone()));

        // Identity-free members are still compared structurally.
        let item = |extent| {
            RaggedType::new(
                ArrayType::new_static(DIMENSION_DATA_TYPE, [2]),
                RaggedValueKind::Dimension(DimensionType::from(Dimension::Static(extent))),
                vec![],
            )
            .unwrap()
        };
        assert_eq!(RaggedType::derive_identity_renaming([&item(3)], [&item(3)]), Ok(TypeIdentityRenaming::new()));
        assert_eq!(
            RaggedType::derive_identity_renaming([&item(3)], [&item(2)]),
            Err(TypeError::invalid("dimension type dimension<2> cannot instantiate declared type dimension<3>"))
        );

        // Repeated indexed occurrences carry independent payloads, which a structural renaming cannot relate.
        assert_eq!(
            RaggedType::derive_identity_renaming([&declared, &declared], [&actual, &actual]),
            Err(TypeError::invalid(
                "indexed dimension `declared_rows` has repeated extent occurrences whose equality requires runtime \
                 checking"
            ))
        );
    }

    #[test]
    fn test_ragged_type_refinements_new() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(5)).unwrap());
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 4]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows.clone(), vec![0])],
        )
        .unwrap();
        let refinements = RaggedTypeRefinements::new([&r#type], [&r#type]).unwrap();
        assert_eq!(refinements.identities_needing_runtime_checks(), &[]);
        assert_eq!(refinements.refine_declared(&r#type, &[]), Ok(r#type.clone()));
        let other = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 4]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, DimensionVariable::new("other", rows.bounds()), vec![0])],
        )
        .unwrap();
        assert_eq!(
            RaggedTypeRefinements::new([&r#type], [&other]),
            Err(TypeError::invalid("indexed dimension `other` does not refine declared indexed dimension `rows`"))
        );
    }

    #[test]
    fn test_ragged_type_refinements_from_local_declarations() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(5)).unwrap());
        let other = DimensionVariable::new("other", rows.bounds());
        let declared = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 4]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows.clone(), vec![0])],
        )
        .unwrap();
        let actual = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 4]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, other.clone(), vec![0])],
        )
        .unwrap();
        let refinements = RaggedTypeRefinements::from_local_declarations([&declared], [&actual]).unwrap();
        assert_eq!(
            refinements.identity_classes().collect::<Vec<_>>(),
            vec![TypeIdentityClass::new(&[rows.clone()], &[other.clone()], false)]
        );
        assert_eq!(refinements.refine_actual(&actual, &[]), Ok(actual));
    }

    #[test]
    fn test_ragged_type_refinements_identity_classes() {
        let dimension = DimensionType::new("item", DimensionBounds::non_negative(Some(5)).unwrap());
        let declared = RaggedType::new(
            ArrayType::new_static(DIMENSION_DATA_TYPE, [2]),
            RaggedValueKind::Dimension(dimension.clone()),
            vec![],
        )
        .unwrap();
        let actual = RaggedType::new(
            ArrayType::new_static(DIMENSION_DATA_TYPE, [2]),
            RaggedValueKind::Dimension(DimensionType::from(Dimension::Static(3))),
            vec![],
        )
        .unwrap();
        let refinements = RaggedTypeRefinements::new([&declared], [&actual]).unwrap();
        assert_eq!(
            refinements.identity_classes().collect::<Vec<_>>(),
            vec![TypeIdentityClass::new(std::slice::from_ref(dimension.variable().unwrap()), &[], true)]
        );
    }

    #[test]
    fn test_ragged_type_refinements_identity_classes_mixed_observation_order() {
        let bounds = DimensionBounds::non_negative(Some(5)).unwrap();
        let [first, batch, second] = ["first", "batch", "second"].map(|name| DimensionVariable::new(name, bounds));
        let indexed = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 4]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, first.clone(), vec![0])],
        )
        .unwrap();
        let batched = RaggedType::new(
            ArrayType::new(DataType::F32, Shape::new(vec![batch.clone().into(), Dimension::Static(4)])),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, second.clone(), vec![0])],
        )
        .unwrap();
        let refinements = RaggedTypeRefinements::new([&indexed, &batched], [&indexed, &batched]).unwrap();

        // The first member's indexed class is observed before the second member's storage class.
        assert_eq!(
            refinements.identity_classes().collect::<Vec<_>>(),
            vec![
                TypeIdentityClass::new(std::slice::from_ref(&first), std::slice::from_ref(&first), false),
                TypeIdentityClass::new(std::slice::from_ref(&batch), std::slice::from_ref(&batch), false),
                TypeIdentityClass::new(std::slice::from_ref(&second), std::slice::from_ref(&second), false),
            ]
        );
    }

    #[test]
    fn test_ragged_type_refinements_identities_needing_runtime_checks() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(5)).unwrap());
        let r#type = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 4]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows.clone(), vec![0])],
        )
        .unwrap();
        let refinements = RaggedTypeRefinements::new([&r#type, &r#type], [&r#type, &r#type]).unwrap();
        assert_eq!(refinements.identities_needing_runtime_checks(), &[rows]);
    }

    #[test]
    fn test_ragged_type_refinements_validate_refinement() {
        let dimension = DimensionType::new("item", DimensionBounds::non_negative(Some(5)).unwrap());
        let declared = RaggedType::new(
            ArrayType::new_static(DIMENSION_DATA_TYPE, [2]),
            RaggedValueKind::Dimension(dimension.clone()),
            vec![],
        )
        .unwrap();
        let actual = RaggedType::new(
            ArrayType::new_static(DIMENSION_DATA_TYPE, [2]),
            RaggedValueKind::Dimension(DimensionType::from(Dimension::Static(3))),
            vec![],
        )
        .unwrap();
        let refinements = RaggedTypeRefinements::new([&declared], [&actual]).unwrap();
        assert_eq!(refinements.validate_refinement([&declared], [&actual], &[dimension.to_variable()]), Ok(()));
        let different = RaggedType::new(
            ArrayType::new_static(DIMENSION_DATA_TYPE, [2]),
            RaggedValueKind::Dimension(DimensionType::from(Dimension::Static(2))),
            vec![],
        )
        .unwrap();
        assert_eq!(
            refinements.validate_refinement([&declared], [&different], &[dimension.to_variable()]),
            Err(TypeError::from(crate::arrays::types::dimensions::DimensionError::InputDimensionMismatch {
                dimension: "item".to_owned(),
                expected: 3,
                actual: 2
            }))
        );
    }

    #[test]
    fn test_ragged_type_refinements_validate_refinement_ragged_scope() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(5)).unwrap());
        let columns = DimensionVariable::new("columns", rows.bounds());
        let indexed = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 4]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows.clone(), vec![0])],
        )
        .unwrap();
        let scalar = RaggedType::new(
            ArrayType::new(DataType::F32, Shape::new(vec![rows.clone().into(), Dimension::Static(4)])),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, columns.clone(), vec![0])],
        )
        .unwrap();
        let closed = [rows.clone(), columns];

        // Outputs cannot reinterpret an input's indexed identity as a storage extent, or the reverse.
        let error =
            TypeError::invalid("dimension identity `rows` is used as both a scalar extent and indexed ragged extents");
        let indexed_refinements = RaggedTypeRefinements::new([&indexed], [&indexed]).unwrap();
        assert_eq!(indexed_refinements.validate_refinement([&scalar], [&scalar], &closed), Err(error.clone()));
        let scalar_refinements = RaggedTypeRefinements::new([&scalar], [&scalar]).unwrap();
        assert_eq!(scalar_refinements.validate_refinement([&indexed], [&indexed], &closed), Err(error));

        // Output occurrences must index the coordinate domains of the corresponding inputs.
        let different = RaggedType::new(
            ArrayType::new_static(DataType::F32, [3, 4]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows, vec![0])],
        )
        .unwrap();
        assert_eq!(
            indexed_refinements.validate_refinement([&different], [&different], &closed),
            Err(TypeError::invalid("indexed dimension `rows` has incompatible dependency domains"))
        );
    }

    #[test]
    fn test_ragged_type_refinements_refine_declared() {
        let batch = DimensionVariable::new("batch", DimensionBounds::non_negative(Some(8)).unwrap());
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(5)).unwrap());
        let declared = RaggedType::new(
            ArrayType::new(DataType::F32, Shape::new(vec![batch.clone().into(), Dimension::Static(4)])),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows.clone(), vec![0])],
        )
        .unwrap();
        let actual = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 4]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows, vec![0])],
        )
        .unwrap();
        let refinements = RaggedTypeRefinements::new([&declared], [&actual]).unwrap();
        assert_eq!(refinements.refine_declared(&declared, &[]), Ok(actual));
        assert_eq!(refinements.refine_declared(&declared, &[batch]), Ok(declared));
    }

    #[test]
    fn test_ragged_type_refinements_refine_actual() {
        let dimension = DimensionType::new("item", DimensionBounds::non_negative(Some(5)).unwrap());
        let declared = RaggedType::new(
            ArrayType::new_static(DIMENSION_DATA_TYPE, [2]),
            RaggedValueKind::Dimension(dimension.clone()),
            vec![],
        )
        .unwrap();
        let exact = RaggedType::new(
            ArrayType::new_static(DIMENSION_DATA_TYPE, [2]),
            RaggedValueKind::Dimension(DimensionType::from(Dimension::Static(3))),
            vec![],
        )
        .unwrap();
        let refinements = RaggedTypeRefinements::new([&declared, &declared], [&declared, &exact]).unwrap();
        assert_eq!(refinements.refine_actual(&declared, &[]), Ok(exact));
        assert_eq!(refinements.refine_actual(&declared, &[dimension.to_variable()]), Ok(declared));
    }

    #[test]
    fn test_ragged_type_refinements_retype_actual() {
        let declared_item = DimensionType::new("declared", DimensionBounds::non_negative(Some(5)).unwrap());
        let actual_item = DimensionType::new("actual", declared_item.bounds());
        let declared = RaggedType::new(
            ArrayType::new_static(DIMENSION_DATA_TYPE, [2]),
            RaggedValueKind::Dimension(declared_item),
            vec![],
        )
        .unwrap();
        let actual = RaggedType::new(
            ArrayType::new_static(DIMENSION_DATA_TYPE, [2]),
            RaggedValueKind::Dimension(actual_item.clone()),
            vec![],
        )
        .unwrap();
        let refinements = RaggedTypeRefinements::from_local_declarations([&declared], [&actual]).unwrap();
        assert_eq!(refinements.retype_actual(&declared, &actual, &[actual_item.to_variable()]), Ok(actual));
    }
}
