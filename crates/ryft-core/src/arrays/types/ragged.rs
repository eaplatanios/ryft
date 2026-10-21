use std::borrow::Borrow;
use std::fmt::Display;
use std::sync::Arc;

use ryft_macros::Parameter;

use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::dimensions::{
    DimensionComparison, DimensionRefinementFact, DimensionType, DimensionTypeRefinements, DimensionVariable, Shape,
};
use crate::arrays::types::ir::{ArrayIrType, ArrayIrTypeRefinements};
use crate::operations::DIMENSION_DATA_TYPE;
use crate::parameters::Parameter;
use crate::programs::{
    NoReferent, ReferenceMemberType, ReferenceType, Type, TypeError, TypeIdentityClass, TypeIdentityPosition,
    TypeIdentityRenaming, TypeRefinements,
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
/// [`Dimension`](crate::Dimension)s. The [`logical_shape`](Self::logical_shape) replaces each padded axis by the
/// variable of its [`RaggedDimension`] without removing dependency-domain axes. The [`RaggedDimension`] entries are
/// canonicalized in storage-axis order. A domain may name an earlier ragged axis, whose sibling [`RaggedDimension`]
/// determines which payload entries are live; self and later ragged dependencies are rejected. Domain order itself is
/// preserved because it determines payload layout.
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

        ArrayIrType::validate_identity_scope([ArrayIrType::Ragged(r#type.clone())])?;
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
    /// no entry. Repeated identities are retained. Ordinary storage-shape variables are handled separately by
    /// [`ArrayIrType::scalar_extent_identities`].
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

// TODO(eaplatanios): Review from here onwards.

impl Type for RaggedType {
    type Identity = DimensionVariable;
    type Refinements = RaggedTypeRefinements;

    fn validate_identity_scope<Types: IntoIterator<Item: Borrow<Self>>>(types: Types) -> Result<(), TypeError> {
        ArrayIrType::validate_identity_scope(
            types.into_iter().map(|r#type| ArrayIrType::Ragged(r#type.borrow().clone())),
        )
    }

    fn validate_zero(&self) -> Result<(), TypeError> {
        Err(TypeError::invalid("zero construction for first-class ragged types is not supported yet"))
    }

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

    fn derive_identity_renaming<'o, D, A>(
        declared: D,
        actual: A,
    ) -> Result<TypeIdentityRenaming<Self::Identity>, TypeError>
    where
        Self: 'o,
        D: IntoIterator<Item = &'o Self, IntoIter: ExactSizeIterator>,
        A: IntoIterator<Item = &'o Self, IntoIter: ExactSizeIterator>,
    {
        let declared = declared.into_iter().cloned().map(ArrayIrType::Ragged).collect::<Vec<_>>();
        let actual = actual.into_iter().cloned().map(ArrayIrType::Ragged).collect::<Vec<_>>();
        ArrayIrType::derive_identity_renaming(&declared, &actual)
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

    fn is_compatible_with(&self, other: &Self) -> bool {
        self == other
    }

    fn is_refined_by(&self, other: &Self) -> bool {
        self.payload.storage_type.is_refined_by(other.storage_type())
            && Self::indexed_refinement_facts(self, other, DimensionComparison::Refinement).is_ok()
    }

    fn is_scalar(&self) -> bool {
        false
    }

    fn is_complex(&self) -> bool {
        match self.payload.value_kind {
            RaggedValueKind::Array => self.payload.storage_type.is_complex(),
            RaggedValueKind::Dimension(_) => false,
        }
    }
}

impl ReferenceMemberType for RaggedType {
    type Referent = NoReferent;

    fn referent(&self) -> Option<&NoReferent> {
        None
    }

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

/// Complete-signature refinements for [`RaggedType`], delegating to the same scalar/indexed separation used by the
/// composite [`ArrayIrType`] universe.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RaggedTypeRefinements {
    /// Shared composite refinement owner.
    composite: ArrayIrTypeRefinements,
}

impl TypeRefinements<RaggedType> for RaggedTypeRefinements {
    fn new<D: IntoIterator, A: IntoIterator>(declared: D, actual: A) -> Result<Self, TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<RaggedType>,
        A::Item: Borrow<RaggedType>,
    {
        Ok(Self {
            composite: ArrayIrTypeRefinements::new(
                declared.into_iter().map(|r#type| ArrayIrType::Ragged(r#type.borrow().clone())).collect::<Vec<_>>(),
                actual.into_iter().map(|r#type| ArrayIrType::Ragged(r#type.borrow().clone())).collect::<Vec<_>>(),
            )?,
        })
    }

    fn from_local_declarations<D: IntoIterator, A: IntoIterator>(declared: D, actual: A) -> Result<Self, TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<RaggedType>,
        A::Item: Borrow<RaggedType>,
    {
        Ok(Self {
            composite: ArrayIrTypeRefinements::from_local_declarations(
                declared.into_iter().map(|r#type| ArrayIrType::Ragged(r#type.borrow().clone())).collect::<Vec<_>>(),
                actual.into_iter().map(|r#type| ArrayIrType::Ragged(r#type.borrow().clone())).collect::<Vec<_>>(),
            )?,
        })
    }

    fn identity_classes<'o>(&'o self) -> impl Iterator<Item = TypeIdentityClass<'o, DimensionVariable>>
    where
        DimensionVariable: 'o,
    {
        self.composite.identity_classes()
    }

    fn identities_needing_runtime_checks(&self) -> &[DimensionVariable] {
        self.composite.identities_needing_runtime_checks()
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
        self.composite.validate_refinement(
            declared.into_iter().map(|r#type| ArrayIrType::Ragged(r#type.borrow().clone())).collect::<Vec<_>>(),
            actual.into_iter().map(|r#type| ArrayIrType::Ragged(r#type.borrow().clone())).collect::<Vec<_>>(),
            closed_identities,
        )
    }

    fn refine_declared(
        &self,
        r#type: &RaggedType,
        preserved_identities: &[DimensionVariable],
    ) -> Result<RaggedType, TypeError> {
        let ArrayIrType::Ragged(refined) =
            self.composite.refine_declared(&r#type.clone().into(), preserved_identities)?
        else {
            unreachable!()
        };
        Ok(refined)
    }

    fn refine_actual(
        &self,
        r#type: &RaggedType,
        preserved_identities: &[DimensionVariable],
    ) -> Result<RaggedType, TypeError> {
        let ArrayIrType::Ragged(refined) =
            self.composite.refine_actual(&r#type.clone().into(), preserved_identities)?
        else {
            unreachable!()
        };
        Ok(refined)
    }

    fn retype_actual(
        &self,
        declared: &RaggedType,
        actual: &RaggedType,
        preferred_identities: &[DimensionVariable],
    ) -> Result<RaggedType, TypeError> {
        let ArrayIrType::Ragged(refined) =
            self.composite
                .retype_actual(&declared.clone().into(), &actual.clone().into(), preferred_identities)?
        else {
            unreachable!()
        };
        Ok(refined)
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
