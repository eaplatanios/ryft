use std::borrow::{Borrow, Cow};
use std::collections::BTreeSet;
use std::fmt::Display;

use ryft_macros::Parameter;

use crate::arrays::addressing::ArrayAddressing;
use crate::arrays::broadcasting::Broadcastable;
use crate::arrays::sharding::meshes::{DeviceMesh, MeshAxisType};
use crate::arrays::sharding::shardings::{Sharding, ShardingDimension};
use crate::arrays::sharding::{ShardableType, ShardingError};
use crate::arrays::types::data::DataType;
use crate::arrays::types::dimensions::{
    Dimension, DimensionComparison, DimensionRefinementFact, DimensionRefinements, DimensionVariable, Shape,
    StaticShape,
};
use crate::arrays::types::layouts::Layout;
use crate::arrays::types::memories::Memory;
use crate::axes::Axis;
use crate::contexts::EagerContext;
use crate::operations::collectives::PARALLEL_VARY_OPERATION_NAME;
use crate::parameters::Parameter;
use crate::programs::types::type_signature_pairs;
use crate::programs::{
    NoReferent, ProgramError, ReferenceMemberType, ReferenceType, Type, TypeError, TypeIdentityClass,
    TypeIdentityPosition, TypeIdentityRenaming, TypeRefinements, Typed, Value, ValueDirectDispatch,
    type_identities_needing_runtime_checks,
};

// Shared empty batch axis set returned by `ArrayType::unreduced_axes` and `ArrayType::reduced_axes` for array types
// that carry no `Sharding`, so that both accessors can hand back a borrow without allocating.
static EMPTY_BATCH_AXES: BTreeSet<String> = BTreeSet::new();

/// Represents the [`Type`] of a potentially multi-dimensional array.
///
/// Note that the [`Display`] implementation of [`ArrayType`] renders array types simply as their [`DataType`]s
/// followed by their [`Shape`]s, optionally followed by their [`Layout`] and [`Sharding`], if present, and finally
/// an `@` followed by the [`Memory`] space when the array resides outside the default [`Memory::Device`] memory.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::arrays::{ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable, Memory, Shape};
///
/// // Boolean scalar.
/// assert_eq!(
///   ArrayType::new(DataType::Boolean, Shape::scalar()).to_string(),
///   "bool[]",
/// );
///
/// // 32-bit floating-point number vector with 42 elements residing in pinned host memory.
/// assert_eq!(
///   ArrayType::new_static(DataType::F32, [42])
///       .with_memory(Memory::Host { pinned: true })
///       .to_string(),
///   "f32[42]@Host[Pinned]",
/// );
///
/// // 64-bit unsigned integer vector with 42 elements.
/// assert_eq!(
///   ArrayType::new_static(DataType::U64, [42]).to_string(),
///   "u64[42]",
/// );
///
/// // 32-bit floating-point number matrix with 42 rows and up to 10 columns.
/// let columns = DimensionVariable::new("columns", DimensionBounds::non_negative(Some(10)).unwrap());
/// assert_eq!(
///   ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(42), columns.into()])).to_string(),
///   "f32[42, columns]",
/// );
///
/// // 64-bit complex number matrix with an unknown number of rows and 42 columns.
/// let rows = DimensionVariable::new("rows", DimensionBounds::unbounded());
/// assert_eq!(
///   ArrayType::new(DataType::C64, Shape::new(vec![rows.into(), Dimension::Static(42)])).to_string(),
///   "c64[rows, 42]",
/// );
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct ArrayType {
    /// [`DataType`] of the elements stored in the array.
    pub(crate) data_type: DataType,

    /// [`Shape`] of the array.
    pub(crate) shape: Shape,

    /// Optional physical memory/storage [`Layout`] of the array.
    pub(crate) layout: Option<Layout>,

    /// Optional [`Sharding`] information about the array.
    pub(crate) sharding: Option<Sharding>,

    /// [`Memory`] in which the array resides. For sharded arrays, the memory applies uniformly to every shard,
    /// each residing in its own device's memory of this kind.
    pub(crate) memory: Memory,
}

impl ArrayType {
    /// Constructs a new [`ArrayType`] with the provided [`DataType`] and [`Shape`], no [`Layout`] or [`Sharding`]
    /// information, and residing in the default [`Memory::Device`] memory space. Use [`Self::with_layout`],
    /// [`Self::with_sharding`], and [`Self::with_memory`] to attach optional metadata.
    #[inline]
    pub fn new(data_type: DataType, shape: Shape) -> Self {
        Self { data_type, shape, layout: None, sharding: None, memory: Memory::Device }
    }

    /// Constructs a new [`ArrayType`] whose dimensions are all statically known, with no [`Layout`] or [`Sharding`]
    /// information, and residing in the default [`Memory::Device`] memory space. Use [`Self::with_layout`],
    /// [`Self::with_sharding`], and [`Self::with_memory`] to attach optional metadata.
    #[inline]
    pub fn new_static<Dimensions: Into<Vec<usize>>>(data_type: DataType, dimensions: Dimensions) -> Self {
        Self::new(data_type, StaticShape::new(dimensions.into()).into())
    }

    /// Returns this [`ArrayType`] with its [`DataType`] replaced by the provided one, keeping its [`Shape`],
    /// [`Layout`], [`Sharding`], and [`Memory`] unchanged.
    #[inline]
    pub fn with_data_type<D: Into<DataType>>(mut self, data_type: D) -> Self {
        self.data_type = data_type.into();
        self
    }

    /// Returns this [`ArrayType`] with its [`Shape`] replaced by the provided one, keeping its [`DataType`],
    /// [`Layout`], [`Sharding`], and [`Memory`] unchanged.
    #[inline]
    pub fn with_shape<S: Into<Shape>>(mut self, shape: S) -> Self {
        self.shape = shape.into();
        self
    }

    /// Returns this [`ArrayType`] with the provided physical memory/storage [`Layout`] replacing its current layout
    /// (or without any [`Layout`] information when [`None`] is provided).
    #[inline]
    pub fn with_layout<L: Into<Option<Layout>>>(mut self, layout: L) -> Self {
        self.layout = layout.into();
        self
    }

    /// Returns this [`ArrayType`] with the provided [`Sharding`] replacing its current sharding metadata (or without
    /// any [`Sharding`] information when [`None`] is provided), after validating that any provided [`Sharding`] has the
    /// same rank as [`Self::shape`].
    #[inline]
    pub fn with_sharding<S: Into<Option<Sharding>>>(mut self, sharding: S) -> Result<Self, ShardingError> {
        let sharding = sharding.into();
        if let Some(sharding) = &sharding {
            let sharding_rank = sharding.rank();
            let array_rank = self.shape.rank();
            if sharding_rank != array_rank {
                return Err(ShardingError::ShardingRankMismatch { sharding_rank, array_rank });
            }
        }
        self.sharding = sharding;
        Ok(self)
    }

    /// Returns this [`ArrayType`] with the array residing in the provided [`Memory`]. This kind of placement
    /// information is metadata about where the array lives, and it does not affect the array's [`DataType`], [`Shape`],
    /// [`Layout`], or [`Sharding`] (for sharded arrays, every shard resides in its own device's memory of this kind).
    #[inline]
    pub fn with_memory<M: Into<Memory>>(mut self, memory: M) -> Self {
        self.memory = memory.into();
        self
    }

    /// Constructs a new scalar [`ArrayType`] with the provided [`DataType`]. The resulting [`ArrayType::shape`]
    /// will be a scalar (i.e., have rank 0).
    #[inline]
    pub fn scalar(data_type: DataType) -> Self {
        Self { data_type, shape: Shape::scalar(), layout: None, sharding: None, memory: Memory::default() }
    }

    /// Returns a scalar [`ArrayType`] with this array's element [`DataType`], [`Memory`] space, and mesh-level
    /// [`Sharding`] metadata. Unlike [`Self::scalar`], this function derives those properties from an existing
    /// [`ArrayType`]. The result has rank zero and no [`Layout`]. If this array type is sharded, the result keeps its
    /// mesh, reduced and unreduced mesh axes, and varying manual axes, but has no per-dimension sharding because it has
    /// no dimensions. An unsharded array produces an unsharded scalar. This is useful for constructing scalar fill
    /// values or zeros that must share an array's distributed dependencies without retaining its shape or storage
    /// layout.
    pub fn scalar_like(&self) -> Result<Self, TypeError> {
        ArrayType::scalar(self.data_type())
            .with_memory(self.memory())
            .with_sharding(
                self.sharding()
                    .map(|sharding| {
                        Sharding::replicated(sharding.mesh().clone(), 0)
                            .with_unreduced_axes(sharding.unreduced_axes().clone())
                            .and_then(|output| output.with_reduced_axes(sharding.reduced_axes().clone()))
                            .and_then(|output| output.with_varying_manual_axes(sharding.varying_manual_axes().clone()))
                            .map_err(|error| {
                                TypeError::invalid(format!("scalar sharding construction failed: {error}"))
                            })
                    })
                    .transpose()?,
            )
            .map_err(|error| TypeError::invalid(format!("scalar type is invalid: {error}")))
    }

    /// Returns the [`DataType`] of the elements stored in the array.
    #[inline]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Returns the [`Shape`] of the array.
    #[inline]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns a new [`StaticShape`] for this [`ArrayType`] if all dimensions of [`Self::shape`] have static size.
    /// This method computes the [`StaticShape`] from [`Self::shape`] on each call and returns an owned value, not a
    /// reference to cached shape metadata.
    #[inline]
    pub fn static_shape(&self) -> Option<StaticShape> {
        self.shape
            .dimensions()
            .iter()
            .map(Dimension::value)
            .collect::<Option<Vec<_>>>()
            .map(StaticShape::new)
    }

    /// Returns the rank (i.e., the number of dimensions) of this [`ArrayType`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::arrays::{ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable, Shape};
    ///
    /// // Boolean scalar.
    /// assert_eq!(ArrayType::new(DataType::Boolean, Shape::scalar()).rank(), 0);
    ///
    /// // 64-bit unsigned integer vector with 42 elements.
    /// assert_eq!(ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(42)])).rank(), 1);
    ///
    /// // 32-bit floating-point number matrix with 42 rows and up to 10 columns.
    /// let columns = DimensionVariable::new("columns", DimensionBounds::non_negative(Some(10)).unwrap());
    /// assert_eq!(
    ///     ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(42), columns.into()])).rank(),
    ///     2,
    /// );
    ///
    /// // 64-bit complex number matrix with an unknown number of rows and 42 columns.
    /// let rows = DimensionVariable::new("rows", DimensionBounds::unbounded());
    /// assert_eq!(
    ///     ArrayType::new(DataType::C64, Shape::new(vec![rows.into(), Dimension::Static(42)])).rank(),
    ///     2,
    /// );
    /// ```
    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Returns the [`Dimension`] of the `index`-th dimension of this array type's [`Shape`]. A negative `index` can be
    /// used to obtain dimension sizes using the end of the dimensions vector as the reference point. For example, an
    /// index value of `-1` will result in the last dimension (i.e., innermost) [`Dimension`] being returned.
    #[inline]
    pub fn dimension<A: Into<Axis>>(&self, index: A) -> Dimension {
        self.shape.dimension(index)
    }

    /// Returns the number of elements in arrays of this [`ArrayType`]. A statically zero dimension makes the result
    /// exactly zero even when another dimension is dynamic. Otherwise, a dynamic dimension produces `Ok(None)`.
    /// Returns a [`TypeError`] if the static element count does not fit in [`usize`].
    #[inline]
    pub fn element_count(&self) -> Result<Option<usize>, TypeError> {
        self.shape.element_count()
    }

    /// Returns the physical memory/storage [`Layout`] of the array if it is known.
    #[inline]
    pub fn layout(&self) -> Option<&Layout> {
        self.layout.as_ref()
    }

    /// Returns [`Sharding`] information about the array if it is known.
    #[inline]
    pub fn sharding(&self) -> Option<&Sharding> {
        self.sharding.as_ref()
    }

    /// Returns the mesh axes over which this array type's [`Sharding`] is *unreduced* (i.e.,
    /// [`Sharding::unreduced_axes`]), or an empty set when the array type carries no [`Sharding`].
    #[inline]
    pub fn unreduced_axes(&self) -> &BTreeSet<String> {
        self.sharding.as_ref().map(|sharding| sharding.unreduced_axes()).unwrap_or(&EMPTY_BATCH_AXES)
    }

    /// Returns the mesh axes over which this array type's [`Sharding`] is *reduced* (i.e.,
    /// [`Sharding::reduced_axes`]), or an empty set when the array type carries no [`Sharding`].
    #[inline]
    pub fn reduced_axes(&self) -> &BTreeSet<String> {
        self.sharding.as_ref().map(|sharding| sharding.reduced_axes()).unwrap_or(&EMPTY_BATCH_AXES)
    }

    /// Returns an [`ArrayType`] whose [`Sharding`] (if any) has its unreduced and reduced axis sets cleared while its
    /// per-dimension placement and varying-manual axes are preserved. Array types with no [`Sharding`] are returned
    /// unchanged. Bilinear type-inference rules (e.g., elementwise multiplication rules) use this so the shared
    /// elementwise broadcast does not reject inputs that only disagree on their reduction state, which those rules
    /// combine separately.
    #[inline]
    pub fn without_reduction_axes(&self) -> Self {
        let Some(sharding) = &self.sharding else {
            return self.clone();
        };
        let stripped = sharding
            .clone()
            .with_unreduced_axes(Vec::<String>::new())
            .expect("clearing unreduced axes preserves a valid sharding")
            .with_reduced_axes(Vec::<String>::new())
            .expect("clearing reduced axes preserves a valid sharding");
        self.clone().with_sharding(stripped).expect("a same-rank sharding stays valid")
    }

    /// Returns the [`Memory`] space in which the array resides.
    #[inline]
    pub fn memory(&self) -> Memory {
        self.memory
    }

    /// Returns an [`ArrayType`] with a dimension inserted at the provided index. Rank-changing operations clear
    /// explicit [`Layout`] information because [`Layout`]s do not carry enough information to infer a correct stride
    /// or tiling for a newly inserted logical axis. [`Sharding`] information is preserved by inserting a replicated
    /// sharding dimension at the same index and shifting the existing dimension annotations.
    pub fn with_inserted_dimension(&self, index: usize, dimension: Dimension) -> Result<Self, TypeError> {
        if index > self.rank() {
            return Err(TypeError::invalid(format!(
                "cannot insert dimension at index {} for rank-{} array type",
                index,
                self.rank()
            )));
        }
        let mut dimensions = self.shape.dimensions().to_vec();
        dimensions.insert(index, dimension);

        // The inserted array dimension is replicated. Reuse the sharding-level insertion so that this method stays a
        // thin wrapper. For more information, refer to the documentation of `Sharding::with_inserted_dimension`.
        let sharding = self
            .sharding
            .as_ref()
            .map(|sharding| sharding.with_inserted_dimension(index, ShardingDimension::Replicated))
            .transpose()
            .map_err(|error| TypeError::invalid(error.to_string()))?;

        Ok(Self {
            data_type: self.data_type,
            shape: Shape::new(dimensions),
            layout: None,
            sharding,
            memory: self.memory,
        })
    }

    /// Returns an [`ArrayType`] with its `index`-th dimension removed, paired with the [`Dimension`] of the removed
    /// dimension. Rank-changing operations clear explicit [`Layout`] information because [`Layout`]s do not carry
    /// enough information to infer a correct stride or tiling after removing a logical axis. [`Sharding`] information
    /// is preserved when the removed dimension is replicated or unconstrained. When the removed dimension is sharded
    /// over manual mesh axes, those axes become varying manual axes because the value can still differ across shards
    /// even though the ranked array dimension is gone. Removing a dimension sharded over non-manual axes is rejected
    /// because there is no equivalent rank-independent metadata field for those axes.
    pub fn without_dimension(&self, index: usize) -> Result<(Self, Dimension), TypeError> {
        if index >= self.rank() {
            return Err(TypeError::invalid(format!(
                "cannot remove dimension at index {} for rank-{} array type",
                index,
                self.rank()
            )));
        }
        let mut dimensions = self.shape.dimensions().to_vec();
        let dimension = dimensions.remove(index);

        // Delegate the per-dimension sharding bookkeeping (i.e., manual axes become varying and non-manual sharded
        // dimensions cannot be dropped) to the sharding itself. For more information, refer to the documentation of
        // `Sharding::without_dimension`.
        let sharding = self
            .sharding
            .as_ref()
            .map(|sharding| sharding.without_dimension(index))
            .transpose()
            .map_err(|error| TypeError::invalid(error.to_string()))?;

        Ok((
            Self {
                data_type: self.data_type,
                shape: Shape::new(dimensions),
                layout: None,
                sharding,
                memory: self.memory,
            },
            dimension,
        ))
    }

    /// Returns an [`ArrayType`] with a replicated [`Sharding`] over the provided [`DeviceMesh`]. The [`Layout`]
    /// information and [`Memory`] placement are preserved.
    pub fn replicated(&self, mesh: &DeviceMesh) -> Result<Self, ShardingError> {
        self.clone().with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), self.shape.rank()))
    }

    /// Checks that the array inputs of an ordinary operation vary over the same manual mesh axes, which is the
    /// standard variation rule of an operation that computes a function of its inputs on every device independently.
    /// Such an operation's output varies over exactly those axes. Its capability aligns the inputs before binding
    /// by calling [`align_manual_variation`](crate::ManualVariationAlignment::align_manual_variation), which inserts a
    /// [`ParallelVaryOperation`](crate::ParallelVaryOperation) on every input that lacks an axis, so this check fails
    /// only for a hand-built program or a capability that skipped the alignment. Every operation's type inference owns
    /// its variation rule: an operation that changes variation, such as a collective, computes its output variation
    /// itself, and any other operation with several array inputs should call this function. An input without a sharding
    /// counts as invariant, because ignoring it would let an unaligned constant bypass the check.
    ///
    /// # Parameters
    ///
    ///   - `operation_name`: Name of the operation whose inputs are checked, used in the error message.
    ///   - `inputs`: Array input types of the operation.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] naming the operation and the remedy if two inputs vary over different manual axes.
    pub fn check_matching_manual_variation(operation_name: &str, inputs: &[&Self]) -> Result<(), TypeError> {
        let Some(first) = inputs.first() else { return Ok(()) };
        let empty = BTreeSet::new();
        let expected = first.sharding().map(Sharding::varying_manual_axes).unwrap_or(&empty);
        if inputs
            .iter()
            .skip(1)
            .any(|input| input.sharding().map(Sharding::varying_manual_axes).unwrap_or(&empty) != expected)
        {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` inputs must have matching varying manual axes; insert \
                 `{PARALLEL_VARY_OPERATION_NAME}` on the inputs that lack an axis, as `align_manual_variation` does",
            )));
        }
        Ok(())
    }

    /// Validates this array type's explicit [`Layout`], if any, against its [`Shape`] using the storage addressing
    /// rules of [`ArrayAddressing`] (e.g., a strided or tiled layout must have the array's rank, and strides may not
    /// alias elements). Dynamic extents are checked at their known lower bounds, because a symbolic type does not
    /// know its runtime extents. Larger runtime extents can still expose additional stride aliasing or a storage size
    /// overflow, which interpretation validates against the resolved extents. Operations whose output types request
    /// a layout (e.g., [`BroadcastOperation`](crate::BroadcastOperation) and [`CastOperation`](crate::CastOperation))
    /// call this function during type inference, so that a malformed request fails there rather than when the output
    /// is materialized or lowered.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] describing the first layout or storage requirement that the type violates.
    pub fn validate_layout(&self) -> Result<(), TypeError> {
        if self.layout.is_some() {
            let dimensions = self
                .shape
                .dimensions()
                .iter()
                .map(|dimension| match dimension {
                    Dimension::Static(extent) => Dimension::Static(*extent),
                    Dimension::Dynamic(variable) => Dimension::Static(variable.bounds().lower()),
                })
                .collect::<Vec<_>>();
            ArrayAddressing::new(self.clone().with_shape(Shape::new(dimensions))).map_err(|error| match error {
                ProgramError::Type(error) => error,
                error => TypeError::invalid(error.to_string()),
            })?;
        }
        Ok(())
    }

    /// Returns this array type's sharding for a same-rank resize, preserving dimension placement and reduction state.
    /// Changed static extents must remain divisible by the product of their explicit mesh-axis sizes. Dynamic extents
    /// and manual or automatic mesh axes are left to their respective execution contracts. Callers validate the output
    /// rank and ensure their operation preserves the carried reduction state; this function only checks divisibility.
    ///
    /// # Parameters
    ///
    ///   - `output_sizes`: New dimensions, in input-axis order, with the same rank as this array type.
    ///   - `operation_name`: Name of the requesting operation, used to identify invalid resizes in diagnostics.
    pub(crate) fn resized_sharding(
        &self,
        output_sizes: &[Dimension],
        operation_name: &'static str,
    ) -> Result<Option<Sharding>, TypeError> {
        let Some(sharding) = self.sharding() else {
            return Ok(None);
        };

        for (axis, (input_size, output_size)) in self.shape().dimensions().iter().zip(output_sizes).enumerate() {
            let (ShardingDimension::Sharded(axis_names), Dimension::Static(output_size)) =
                (&sharding.dimensions()[axis], output_size)
            else {
                continue;
            };

            if input_size == &Dimension::Static(*output_size) {
                continue;
            }

            // Dimensions sharded only over manual or automatic axes have an explicit mesh-axis product of one.
            let explicit_axis_product: usize = axis_names
                .iter()
                .filter(|name| sharding.mesh().axis_type(name) == Some(MeshAxisType::Explicit))
                .filter_map(|name| sharding.mesh().axis_size(name))
                .try_fold(1usize, |product, size| product.checked_mul(size))
                .ok_or_else(|| TypeError::invalid(format!("`{operation_name}` mesh-axis product overflows `usize`")))?;
            if explicit_axis_product > 1 && output_size % explicit_axis_product != 0 {
                return Err(TypeError::invalid(format!(
                    "`{operation_name}` on a dimension sharded over explicit mesh axes requires the output size \
                     ({output_size}) at axis {axis} to be divisible by the mesh-axis product ({explicit_axis_product})",
                )));
            }
        }
        Ok(Some(sharding.clone()))
    }

    /// Returns whether `other` has the same element [`DataType`] and [`Memory`] placement as this [`ArrayType`] and
    /// refines this type's optional [`Layout`] and [`Sharding`] constraints. [`Shape`] refinement is deliberately
    /// excluded because ordinary type refinement, identity-renaming derivation, and cross-signature refinement
    /// validation have different requirements.
    fn non_shape_components_are_refined_by(&self, other: &Self) -> bool {
        let layout_is_refined = match (&self.layout, &other.layout) {
            (None, _) => true,
            (Some(_), None) => false,
            (Some(declared), Some(actual)) => declared == actual,
        };
        let sharding_is_refined = match (&self.sharding, &other.sharding) {
            (None, _) => true,
            (Some(_), None) => false,
            (Some(declared), Some(actual)) => declared == actual,
        };
        self.data_type == other.data_type && layout_is_refined && sharding_is_refined && self.memory == other.memory
    }
}

impl Display for ArrayType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}{}", self.data_type, self.shape)?;
        if let Some(layout) = &self.layout {
            write!(formatter, "[layout={layout}]")?;
        }
        if let Some(sharding) = &self.sharding {
            write!(formatter, "[sharding={sharding}]")?;
        }
        if self.memory != Memory::Device {
            write!(formatter, "@{}", self.memory)?;
        }
        Ok(())
    }
}

impl Type for ArrayType {
    type Identity = DimensionVariable;
    type Refinements = ArrayTypeRefinements;

    #[inline]
    fn validate_zero(&self) -> Result<(), TypeError> {
        if matches!(self.data_type(), DataType::Token | DataType::F8E8M0FNU) {
            return Err(TypeError::invalid(format!("data type `{}` cannot represent zero", self.data_type())));
        }
        Ok(())
    }

    #[inline]
    fn validate_one(&self) -> Result<(), TypeError> {
        if matches!(self.data_type(), DataType::Token | DataType::Zero) {
            return Err(TypeError::invalid(format!("data type `{}` cannot represent one", self.data_type())));
        }
        Ok(())
    }

    fn identities(&self) -> impl Iterator<Item = (TypeIdentityPosition, &Self::Identity)> {
        self.shape
            .dimensions()
            .iter()
            .filter_map(Dimension::variable)
            .map(|variable| (TypeIdentityPosition::Reference, variable))
    }

    fn derive_identity_renaming<
        't,
        D: IntoIterator<Item = &'t Self, IntoIter: ExactSizeIterator>,
        A: IntoIterator<Item = &'t Self, IntoIter: ExactSizeIterator>,
    >(
        declared: D,
        actual: A,
    ) -> Result<TypeIdentityRenaming<Self::Identity>, TypeError>
    where
        Self: 't,
    {
        DimensionRefinements::derive_identity_renaming(type_signature_pairs(declared, actual)?.flat_map(
            |(declared, actual)| {
                ArrayTypeRefinements::refinement_facts(declared, actual, DimensionComparison::IdentityRenaming)
            },
        ))
    }

    #[inline]
    fn rename_identities(&self, renaming: &TypeIdentityRenaming<Self::Identity>) -> Result<Self, TypeError> {
        Ok(Self { shape: self.shape.rename_type_identities(renaming), ..self.clone() })
    }

    #[inline]
    fn is_compatible_with(&self, other: &Self) -> bool {
        // Note that this compatibility relationship is defined here as a "broadcastability" relationship.
        self.is_broadcastable_to(other)
    }

    #[inline]
    fn is_refined_by(&self, other: &Self) -> bool {
        // `DataType`s and `Memory` placements must match exactly and shapes follow `Shape::is_refined_by`. The optional
        // `Layout` and `Sharding` metadata components follow the same directional declared-vs-actual reading as dynamic
        // dimensions where a declared `None` leaves the component unspecified and admits every actual value (including
        // one that carries the component), while a declared `Some` requires the actual type to carry the exact same
        // component. This is what lets declared types staged without placement information (e.g., program input types
        // traced from metadata-free exemplars) accept concrete runtime values whose types carry normalized shardings.
        self.non_shape_components_are_refined_by(other) && self.shape.is_refined_by(&other.shape)
    }

    #[inline]
    fn is_scalar(&self) -> bool {
        self.rank() == 0
    }

    #[inline]
    fn is_complex(&self) -> bool {
        self.data_type.is_complex()
    }
}

impl ShardableType for ArrayType {
    #[inline]
    fn sharding(&self) -> Option<&Sharding> {
        self.sharding()
    }
}

// Array types never describe references, so the universe has no referent family: `NoReferent` is uninhabited and both
// projections are `None` for every array type, which is the referent-free contract of `ReferenceMemberType`.
impl ReferenceMemberType for ArrayType {
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

// `ArrayType` describes itself. This fixed point (rather than a dummy unit-like type) is deliberate and load-bearing
// for metadata-only programs. The whole value of tracing with `ArrayType` as the carrier is that it inhabits the
// same type universe as real arrays, so every piece of machinery it reuses (e.g., `Operation<Type = ArrayType>` type
// inference, lowering bounds such as `MlirLowerableValue: Value<Type = ArrayType>` in `ryft-xla`, and tracing and
// staging code pinned on `V: Value<Type = ArrayType>`) accepts it anywhere a concrete array value would slot in.
// A dummy `Type` would place the carrier in a fresh operation universe with no operations or inference rules and an
// opaque unit type would additionally discard the shape, element-type, and sharding payload that `r#type()` feeds to
// type inference during a metadata trace. This is the standard abstract-interpretation move (e.g., JAX's `eval_shape`
// traces with `ShapeDtypeStruct` standing in for arrays, and an abstract value's abstract value is itself).
impl Typed for ArrayType {
    type Type = ArrayType;

    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayType> {
        Cow::Borrowed(self)
    }
}

// Some staged XLA programs use `ArrayType` itself as the value carrier (e.g., with `T = ArrayType` and `V = ArrayType`)
// because the program stores boundary metadata rather than runtime arrays. In that mode the abstract value is
// self-describing: its type is itself. This is not a type-theoretic universe claim (i.e., `ArrayType : ArrayType`).
// It is the `Typed` witness required by `Value<Type = ArrayType>` for metadata-only program storage, lowering, and
// transformation. Refer to the comment above the `Typed` implementation for `ArrayType` for more information.
impl Value for ArrayType {
    type Dispatch = ValueDirectDispatch;
    type Domain = EagerContext<Self>;

    #[inline]
    fn domain(&self) -> EagerContext<Self> {
        EagerContext::new()
    }

    #[inline]
    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<Self::Type as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        self.rename_identities(renaming)
    }
}

impl From<NoReferent> for ArrayType {
    #[inline]
    fn from(referent: NoReferent) -> Self {
        match referent {}
    }
}

impl From<ReferenceType<NoReferent>> for ArrayType {
    #[inline]
    fn from(r#type: ReferenceType<NoReferent>) -> Self {
        match *r#type.referent() {}
    }
}

/// [`TypeRefinements`] for a complete [`ArrayType`] signature, backed by a [`DimensionRefinements`] instance. This
/// adapter validates array metadata and shape pairs, then applies the resulting dimension facts to array axes. Refer
/// to [`DimensionRefinements`] for the distinction between declared and actual identity scopes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ArrayTypeRefinements {
    /// [`DimensionRefinements`] shared by array axes and, in the array IR universe, dimension and reference members.
    dimensions: DimensionRefinements,
}

impl ArrayTypeRefinements {
    /// Returns the underlying [`DimensionRefinements`]s.
    #[inline]
    pub(crate) fn dimension_refinements(&self) -> &DimensionRefinements {
        &self.dimensions
    }

    /// Returns the [`DimensionRefinementFact`]s for one declared and one actual array type, or their comparison error.
    /// Validates non-shape metadata and rank before comparing corresponding axes. [`DimensionComparison`] determines
    /// whether distinct dimension variables with unknown extents are accepted and whether identical dimensions
    /// contribute facts, as in [`DimensionRefinements::refinement_facts`]. Structural comparisons preserve
    /// substitutions rather than infer conditional exact extents from supplied variables.
    pub(crate) fn refinement_facts<'o>(
        declared: &'o ArrayType,
        actual: &'o ArrayType,
        comparison: DimensionComparison,
    ) -> impl Iterator<Item = Result<DimensionRefinementFact<'o>, TypeError>> {
        let error = if declared.non_shape_components_are_refined_by(actual) && declared.rank() == actual.rank() {
            None
        } else {
            let message = if comparison == DimensionComparison::IdentityRenaming {
                format!("type {actual} cannot instantiate declared type {declared}")
            } else {
                format!("type {actual} does not refine declared type {declared}")
            };
            Some(TypeError::invalid(message))
        };
        let dimensions = error.is_none().then(|| declared.shape().dimensions().iter().zip(actual.shape().dimensions()));
        error.into_iter().map(Err).chain(
            dimensions.into_iter().flatten().flat_map(move |(declared, actual)| {
                DimensionRefinements::refinement_facts(declared, actual, comparison)
            }),
        )
    }
}

impl TypeRefinements<ArrayType> for ArrayTypeRefinements {
    fn new<D: IntoIterator, A: IntoIterator>(declared: D, actual: A) -> Result<Self, TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<ArrayType>,
        A::Item: Borrow<ArrayType>,
    {
        // Ordinary admission retains nominal axes; comparing the complete signature below checks their relationships.
        let declared = declared.into_iter().collect::<Vec<_>>();
        let actual = actual.into_iter().collect::<Vec<_>>();
        type_signature_pairs(declared.iter(), actual.iter())?.try_for_each(|(declared, actual)| {
            Self::refinement_facts(declared.borrow(), actual.borrow(), DimensionComparison::Refinement)
                .try_for_each(|refinement_fact| refinement_fact.map(|_| ()))
        })?;
        Self::from_local_declarations(declared, actual)
    }

    fn from_local_declarations<D: IntoIterator, A: IntoIterator>(declared: D, actual: A) -> Result<Self, TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<ArrayType>,
        A::Item: Borrow<ArrayType>,
    {
        let declared = declared.into_iter().collect::<Vec<_>>();
        let actual = actual.into_iter().collect::<Vec<_>>();
        let refinements = DimensionRefinements::from_refinement_facts(
            type_signature_pairs(declared.iter().map(Borrow::borrow), actual.iter().map(Borrow::borrow))?.flat_map(
                |(declared, actual)| Self::refinement_facts(declared, actual, DimensionComparison::LocalDeclarations),
            ),
        )?;
        let actual = actual.iter().map(|r#type| Some(r#type.borrow())).collect::<Vec<_>>();
        let unproven = type_identities_needing_runtime_checks::<ArrayType, _>(&declared, &actual)?;
        Ok(Self::from(refinements.with_identities_needing_runtime_checks(unproven)))
    }

    #[inline]
    fn identity_classes<'o>(&'o self) -> impl Iterator<Item = TypeIdentityClass<'o, <ArrayType as Type>::Identity>>
    where
        <ArrayType as Type>::Identity: 'o,
    {
        self.dimensions.identity_classes()
    }

    #[inline]
    fn identities_needing_runtime_checks(&self) -> &[DimensionVariable] {
        self.dimensions.identities_needing_runtime_checks()
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
        D::Item: Borrow<ArrayType>,
        A::Item: Borrow<ArrayType>,
    {
        let mut refinements = self.clone();
        type_signature_pairs(declared, actual)?.try_for_each(|(declared, actual)| {
            Self::refinement_facts(declared.borrow(), actual.borrow(), DimensionComparison::Refinement).try_for_each(
                |refinement_fact| match refinement_fact? {
                    DimensionRefinementFact::Declared(variable, extent) => {
                        refinements.dimensions.validate_declared_extent(variable, extent, closed_identities)
                    }
                    DimensionRefinementFact::Actual(..) | DimensionRefinementFact::Correspondence(..) => Ok(()),
                },
            )
        })
    }

    fn refine_declared(
        &self,
        r#type: &ArrayType,
        preserved_identities: &[DimensionVariable],
    ) -> Result<ArrayType, TypeError> {
        let dimensions = r#type
            .shape()
            .dimensions()
            .iter()
            .map(|dimension| self.dimensions.refine_declared_dimension(dimension, preserved_identities))
            .collect();
        Ok(r#type.clone().with_shape(Shape::new(dimensions)))
    }

    fn refine_actual(
        &self,
        r#type: &ArrayType,
        preserved_identities: &[DimensionVariable],
    ) -> Result<ArrayType, TypeError> {
        let dimensions = r#type
            .shape()
            .dimensions()
            .iter()
            .map(|dimension| self.dimensions.refine_actual_dimension(dimension, preserved_identities))
            .collect();
        Ok(r#type.clone().with_shape(Shape::new(dimensions)))
    }

    fn retype_actual(
        &self,
        declared: &ArrayType,
        actual: &ArrayType,
        preferred_identities: &[DimensionVariable],
    ) -> Result<ArrayType, TypeError> {
        if declared.rank() != actual.rank() {
            return Err(TypeError::invalid(format!("type {actual} does not refine declared type {declared}")));
        }
        let dimensions = declared
            .shape()
            .dimensions()
            .iter()
            .zip(actual.shape().dimensions())
            .map(|(declared, actual)| self.dimensions.retype_actual_dimension(declared, actual, preferred_identities))
            .collect();
        Ok(actual.clone().with_shape(Shape::new(dimensions)))
    }
}

impl From<DimensionRefinements> for ArrayTypeRefinements {
    #[inline]
    fn from(dimensions: DimensionRefinements) -> Self {
        Self { dimensions }
    }
}

/// [`Type`] universe whose types may describe arrays. Universe-parameterized array capabilities use this view in their
/// provided functions to read the [`ArrayType`] of a value in either universe, so that those functions are available
/// for homogeneous array values and for array IR values alike.
pub trait ArrayMemberType: Type {
    /// Returns the [`ArrayType`] that this type describes, or a [`TypeError`] if it does not describe an array
    /// (e.g., a first-class dimension or a reference in the array IR universe).
    fn as_array_type(&self) -> Result<&ArrayType, TypeError>;
}

impl ArrayMemberType for ArrayType {
    #[inline]
    fn as_array_type(&self) -> Result<&ArrayType, TypeError> {
        Ok(self)
    }
}

impl ArrayMemberType for NoReferent {
    #[inline]
    fn as_array_type(&self) -> Result<&ArrayType, TypeError> {
        // Reference-aware array code can request this view, but a referent-free universe never supplies a referent.
        match *self {}
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use pretty_assertions::assert_eq;

    use crate::arrays::sharding::ShardingError;
    use crate::arrays::sharding::meshes::{Device, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType};
    use crate::arrays::sharding::shardings::{Sharding, ShardingDimension};
    use crate::arrays::types::data::DataType::{BF16, Boolean, C64, F8E3M4, F8E4M3FN, F16, F32, F64};
    use crate::arrays::types::dimensions::{DimensionBounds, DimensionError, DimensionType, DimensionVariable};
    use crate::arrays::types::ir::ArrayIrType;
    use crate::arrays::types::layouts::{StridedLayout, Tile, TileDimension, TiledLayout};

    use super::*;

    #[test]
    fn test_array_type_new_static() {
        let dimensions = &[2, 3][..];
        assert_eq!(ArrayType::new_static(Boolean, []), ArrayType::new(Boolean, Shape::scalar()));
        assert_eq!(ArrayType::new_static(F32, dimensions), ArrayType::new(F32, StaticShape::new(vec![2, 3]).into()));
    }

    #[test]
    fn test_array_type_scalar_like() {
        // The scalar keeps the source's data type, memory, mesh, and distributed dependency metadata,
        // but neither its shape nor its layout.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let plain_source = ArrayType::new_static(F64, [2, 3])
            .with_layout(Layout::Strided(StridedLayout::new(vec![1, 2])))
            .with_memory(Memory::Host { pinned: true });
        assert_eq!(plain_source.scalar_like(), Ok(ArrayType::scalar(F64).with_memory(Memory::Host { pinned: true })),);
        let unreduced_source = ArrayType::new_static(F32, [4])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
                    .unwrap()
                    .with_unreduced_axes(["m"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            unreduced_source.scalar_like(),
            Ok(ArrayType::scalar(F32)
                .with_sharding(Sharding::replicated(mesh.clone(), 0).with_unreduced_axes(["m"]).unwrap())
                .unwrap()),
        );
        let reduced_source = ArrayType::new_static(F32, [4])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_reduced_axes(["m"]).unwrap())
            .unwrap();
        assert_eq!(
            reduced_source.scalar_like(),
            Ok(ArrayType::scalar(F32)
                .with_sharding(Sharding::replicated(mesh.clone(), 0).with_reduced_axes(["m"]).unwrap())
                .unwrap()),
        );
        let varying_source = ArrayType::new_static(F32, [4])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
                    .unwrap()
                    .with_varying_manual_axes(["m"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            varying_source.scalar_like(),
            Ok(ArrayType::scalar(F32)
                .with_sharding(Sharding::replicated(mesh, 0).with_varying_manual_axes(["m"]).unwrap())
                .unwrap()),
        );
    }

    #[test]
    fn test_array_type_static_shape() {
        let static_shape = Shape::new(vec![Dimension::Static(42), Dimension::Static(4), Dimension::Static(2)]);
        let dynamic_shape = Shape::new(vec![
            Dimension::Static(42),
            Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
        ]);

        let scalar = ArrayType::scalar(Boolean);
        let static_array_type = ArrayType::new(F32, static_shape);
        let dynamic_array_type = ArrayType::new(F8E3M4, dynamic_shape);

        assert_eq!(scalar.static_shape(), Some(StaticShape::scalar()));
        assert_eq!(static_array_type.static_shape(), Some(StaticShape::new(vec![42, 4, 2])));
        assert_eq!(dynamic_array_type.static_shape(), None);
    }

    #[test]
    fn test_array_type_rank() {
        let s1 = Shape::new(vec![Dimension::Static(42), Dimension::Static(4), Dimension::Static(2)]);
        let s2 = Shape::new(vec![
            Dimension::Static(42),
            Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
        ]);

        let t0 = ArrayType::scalar(Boolean);
        let t1 = ArrayType::new(F32, s1);
        let t2 = ArrayType::new(F8E3M4, s2);

        assert_eq!(t0.rank(), 0);
        assert_eq!(t1.rank(), 3);
        assert_eq!(t2.rank(), 2);
    }

    #[test]
    fn test_array_type_dimension() {
        let columns = DimensionVariable::new("columns", DimensionBounds::unbounded());
        let s0 = Shape::new(vec![Dimension::Static(42), Dimension::Static(4), Dimension::Static(2)]);
        let s1 = Shape::new(vec![Dimension::Static(42), Dimension::Dynamic(columns.clone())]);

        let t0 = ArrayType::new(F32, s0);
        let t1 = ArrayType::new(F8E3M4, s1);

        assert_eq!(t0.dimension(0), Dimension::Static(42));
        assert_eq!(t0.dimension(2), Dimension::Static(2));
        assert_eq!(t0.dimension(-2), Dimension::Static(4));
        assert_eq!(t1.dimension(0), Dimension::Static(42));
        assert_eq!(t1.dimension(1), Dimension::Dynamic(columns.clone()));
        assert_eq!(t1.dimension(-1), Dimension::Dynamic(columns));
    }

    #[test]
    fn test_array_type_element_count() {
        let static_shape = Shape::new(vec![Dimension::Static(42), Dimension::Static(4), Dimension::Static(2)]);
        let dynamic_shape = Shape::new(vec![
            Dimension::Static(42),
            Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
        ]);

        let scalar = ArrayType::scalar(Boolean);
        let static_array_type = ArrayType::new(F32, static_shape);
        let dynamic_array_type = ArrayType::new(F8E3M4, dynamic_shape);

        assert_eq!(scalar.element_count(), Ok(Some(1)));
        assert_eq!(static_array_type.element_count(), Ok(Some(336)));
        assert_eq!(dynamic_array_type.element_count(), Ok(None));
    }

    #[test]
    fn test_array_type_sharding() {
        // An array type with no sharding reports `None` and empty reduction-axis sets, and stripping its reduction
        // axes leaves it unchanged.
        let unsharded = ArrayType::new(F32, Shape::new(vec![Dimension::Static(8)]));
        assert_eq!(unsharded.sharding(), None);
        assert_eq!(unsharded.unreduced_axes(), &BTreeSet::new());
        assert_eq!(unsharded.reduced_axes(), &BTreeSet::new());
        assert_eq!(unsharded.without_reduction_axes(), unsharded);

        // A single sharding can carry unreduced axes, reduced axes, and varying manual axes at once (over distinct
        // mesh axes), alongside a sharded placement dimension.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("a", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["a"])])
            .unwrap()
            .with_unreduced_axes(["x"])
            .unwrap()
            .with_reduced_axes(["y"])
            .unwrap()
            .with_varying_manual_axes(["m"])
            .unwrap();
        let sharded =
            ArrayType::new(F32, Shape::new(vec![Dimension::Static(8)])).with_sharding(sharding.clone()).unwrap();

        // `sharding` exposes the attached sharding, and the accessors surface its reduction-axis sets.
        assert_eq!(sharded.sharding(), Some(&sharding));
        assert_eq!(sharded.unreduced_axes(), &BTreeSet::from(["x".to_string()]));
        assert_eq!(sharded.reduced_axes(), &BTreeSet::from(["y".to_string()]));

        // `without_reduction_axes` clears both reduction-axis sets while preserving the placement dimensions and the
        // varying manual axes.
        let stripped = sharded.without_reduction_axes();
        let stripped_sharding = stripped.sharding().unwrap();
        assert_eq!(stripped_sharding.unreduced_axes(), &BTreeSet::new());
        assert_eq!(stripped_sharding.reduced_axes(), &BTreeSet::new());
        assert_eq!(stripped_sharding.dimensions(), sharding.dimensions());
        assert_eq!(stripped_sharding.varying_manual_axes(), &BTreeSet::from(["m".to_string()]));
        assert_eq!(stripped.unreduced_axes(), &BTreeSet::new());
        assert_eq!(stripped.reduced_axes(), &BTreeSet::new());
    }

    #[test]
    fn test_array_type_memory() {
        // Arrays reside in device memory by default, and `with_memory` re-places them.
        let t0 = ArrayType::new(F32, Shape::new(vec![2.into(), 3.into()]));
        assert_eq!(t0.memory(), Memory::Device);
        let t1 = t0.clone().with_memory(Memory::Host { pinned: true });
        assert_eq!(t1.memory(), Memory::Host { pinned: true });
        assert_eq!(t1.data_type(), t0.data_type());
        assert_eq!(t1.shape(), t0.shape());

        // Placement participates in type equality and is rendered only for non-default memories.
        assert_ne!(t0, t1);
        assert_eq!(t0.to_string(), "f32[2, 3]");
        assert_eq!(t1.to_string(), "f32[2, 3]@Host[Pinned]");
        assert_eq!(t0.clone().with_memory(Memory::Host { pinned: false }).to_string(), "f32[2, 3]@Host[Unpinned]");

        // Rank-changing helpers preserve the placement.
        let t2 = t1.with_inserted_dimension(1, 5.into()).unwrap();
        assert_eq!(t2.memory(), Memory::Host { pinned: true });
        let (t3, removed_dimension) = t2.without_dimension(1).unwrap();
        assert_eq!(removed_dimension, Dimension::Static(5));
        assert_eq!(t3, t1);

        // Replication preserves the placement.
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(0, 0), Device::new(1, 0)],
        )
        .unwrap();
        assert_eq!(t1.replicated(&mesh).unwrap().memory(), Memory::Host { pinned: true });
    }

    #[test]
    fn test_array_type_insert_and_remove_dimensions() {
        let t0 = ArrayType::new(F32, Shape::new(vec![2.into(), 3.into()]));
        let t1 = t0.with_inserted_dimension(1, 5.into()).unwrap();
        let t2 = ArrayType::new(F32, Shape::new(vec![2.into(), 5.into(), 3.into()]));

        assert_eq!(t1, t2);
        assert_eq!(t1.without_dimension(1).unwrap(), (t0, Dimension::Static(5)));

        let t3 = ArrayType::new(F32, Shape::new(vec![2.into(), 3.into()]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![12, 4])));
        let t4 = t3.with_inserted_dimension(1, 5.into()).unwrap();

        assert_eq!(t4.layout, None);
        assert_eq!(t4.shape, Shape::new(vec![2.into(), 5.into(), 3.into()]));

        let (t5, removed_dimension) = t4.without_dimension(1).unwrap();

        assert_eq!(removed_dimension, Dimension::Static(5));
        assert_eq!(t5.layout, None);
        assert_eq!(t5.shape, Shape::new(vec![2.into(), 3.into()]));

        let m0 = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let s0 = Sharding::new(m0.clone(), vec![ShardingDimension::sharded(["x"])])
            .unwrap()
            .with_varying_manual_axes(["x"])
            .unwrap();
        let t6 = ArrayType::new(F32, Shape::new(vec![8.into()])).with_sharding(s0).unwrap();
        let t7 = t6.with_inserted_dimension(0, 2.into()).unwrap();
        let s1 = t7.sharding().unwrap();

        assert_eq!(s1.dimensions(), &[ShardingDimension::replicated(), ShardingDimension::sharded(["x"])]);
        assert_eq!(s1.varying_manual_axes(), &["x".to_string()].into_iter().collect());

        let (t8, removed_dimension) = t7.without_dimension(0).unwrap();

        assert_eq!(removed_dimension, Dimension::Static(2));
        assert_eq!(t8, t6);

        let s2 = Sharding::new(m0, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let t9 = ArrayType::new(F32, Shape::new(vec![8.into()])).with_sharding(s2).unwrap();

        let (t10, removed_dimension) = t9.without_dimension(0).unwrap();
        let s3 = t10.sharding().unwrap();

        assert_eq!(removed_dimension, Dimension::Static(8));
        assert_eq!(s3.dimensions(), &Vec::<ShardingDimension>::new());
        assert_eq!(s3.varying_manual_axes(), &["x".to_string()].into_iter().collect());

        let m1 = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let t11 = ArrayType::new(F32, Shape::new(vec![8.into()]))
            .with_sharding(Sharding::new(m1, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();

        assert_eq!(
            t11.without_dimension(0),
            Err(TypeError::invalid(
                "cannot remove dimension 0 because it is sharded over the non-manual mesh axis `x`".to_string(),
            )),
        );
    }

    #[test]
    fn test_array_type_replicated() {
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(0, 0), Device::new(1, 0)],
        )
        .unwrap();
        let r#type = ArrayType::new(F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![12, 4])));
        let replicated = r#type.replicated(&mesh).unwrap();

        assert_eq!(replicated.data_type(), F32);
        assert_eq!(replicated.shape(), r#type.shape());
        assert_eq!(replicated.layout(), r#type.layout());
        assert_eq!(replicated.sharding(), Some(&Sharding::replicated(mesh.logical_mesh().clone(), 2)));
    }

    #[test]
    fn test_array_type_display() {
        let s1 = Shape::new(vec![Dimension::Static(42), Dimension::Static(4), Dimension::Static(2)]);
        let s2 = Shape::new(vec![Dimension::Static(4), Dimension::Static(1)]);
        let s3 = Shape::new(vec![
            Dimension::Static(4),
            Dimension::Dynamic(DimensionVariable::new("depth", DimensionBounds::non_negative(Some(1)).unwrap())),
        ]);
        let s4 = Shape::new(vec![
            Dimension::Dynamic(DimensionVariable::new("rows", DimensionBounds::unbounded())),
            Dimension::Static(42),
            Dimension::Dynamic(DimensionVariable::new("columns", DimensionBounds::unbounded())),
        ]);
        let s5 = Shape::new(vec![
            Dimension::Static(42),
            Dimension::Dynamic(DimensionVariable::new("columns", DimensionBounds::unbounded())),
        ]);

        let t0 = ArrayType::scalar(Boolean);
        let t1 = ArrayType::new(F32, s1);
        let t2 = ArrayType::new(BF16, s2);
        let t3 = ArrayType::new(F16, s3);
        let t4 = ArrayType::new(C64, s4);
        let t5 = ArrayType::new(F8E4M3FN, s5);
        let t6 = ArrayType::new(F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(2)]))
            .with_layout(Layout::Tiled(TiledLayout::new(vec![1, 0], vec![Tile::new(vec![TileDimension::Sized(2)])])));
        let t7 = ArrayType::new(F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(2)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![8, 4])));
        let t8 = ArrayType::new(F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(
                Sharding::new(
                    LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
                    vec![ShardingDimension::sharded(["x"])],
                )
                .unwrap()
                .with_varying_manual_axes(["x"])
                .unwrap(),
            )
            .unwrap();

        assert_eq!(format!("{t0}"), "bool[]");
        assert_eq!(format!("{t1}"), "f32[42, 4, 2]");
        assert_eq!(format!("{t2}"), "bf16[4, 1]");
        assert_eq!(format!("{t3}"), "f16[4, depth]");
        assert_eq!(format!("{t4}"), "c64[rows, 42, columns]");
        assert_eq!(format!("{t5}"), "f8e4m3fn[42, columns]");
        assert_eq!(format!("{t6}"), "f32[4, 2][layout=tiled{1,0:T(2)}]");
        assert_eq!(format!("{t7}"), "f32[4, 2][layout=strided{8,4}]");
        assert_eq!(format!("{t8}"), "f32[8][sharding={mesh<['x'=4:manual]>, [{'x'}], varying_manual={'x'}}]");
    }

    #[test]
    fn test_array_type_is_compatible_with() {
        // `Type::is_compatible_with` is the interoperability relation (i.e., the "broadcastability" relation),
        // and it is distinct from the refinement relation tested by `test_array_type_is_refined_by`.
        let vector = ArrayType::new(F32, Shape::new(vec![Dimension::Static(1), Dimension::Static(3)]));
        let matrix = ArrayType::new(F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        assert!(vector.is_compatible_with(&matrix));
        assert!(!matrix.is_compatible_with(&vector));
        assert!(matrix.is_compatible_with(&matrix));
    }

    #[test]
    fn test_array_type_is_refined_by() {
        let declared = ArrayType::new(
            F32,
            Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded())),
                Dimension::Static(3),
            ]),
        );
        let actual = ArrayType::new(F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));

        // Identical types and refining shapes are accepted; the relation is directional.
        assert!(declared.is_refined_by(&declared));
        assert!(actual.is_refined_by(&actual));
        assert!(declared.is_refined_by(&actual));
        assert!(!actual.is_refined_by(&declared));

        // Bounded dynamic declared dimensions enforce their exclusive bound on static actual sizes.
        let bounded = ArrayType::new(
            F32,
            Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::non_negative(Some(4)).unwrap())),
                Dimension::Static(3),
            ]),
        );
        assert!(
            bounded.is_refined_by(&ArrayType::new(F32, Shape::new(vec![Dimension::Static(3), Dimension::Static(3)])))
        );
        assert!(
            !bounded.is_refined_by(&ArrayType::new(F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(3)])))
        );

        // Data types must match exactly; broadcastable shapes do not make types compatible.
        assert!(
            !declared.is_refined_by(&ArrayType::new(F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])))
        );
        assert!(!actual.is_refined_by(&ArrayType::new(F32, Shape::new(vec![Dimension::Static(3)]))));

        // A declared type without optional layout metadata leaves the layout unspecified and is refined by actual
        // types that carry one, while a declared layout must be carried exactly by the actual type.
        let strided = actual.clone().with_layout(Layout::Strided(StridedLayout::new(vec![12, 4])));
        assert!(declared.is_refined_by(&strided));
        assert!(strided.is_refined_by(&strided));
        assert!(!strided.is_refined_by(&actual));
        assert!(!strided.is_refined_by(&actual.clone().with_layout(Layout::Strided(StridedLayout::new(vec![3, 1])))));

        // Optional sharding metadata follows the same directional reading: an unsharded declared type admits sharded
        // actual types, a declared sharding must match the actual sharding exactly, and a sharded declared type is
        // never refined by an unsharded actual type.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded = actual
            .clone()
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        assert!(declared.is_refined_by(&sharded));
        assert!(sharded.is_refined_by(&sharded));
        assert!(!sharded.is_refined_by(&actual));
        let replicated = actual.clone().with_sharding(Sharding::replicated(mesh, 2)).unwrap();
        assert!(!sharded.is_refined_by(&replicated));

        // Memory placement is not optional and must always match exactly.
        let pinned = actual.clone().with_memory(Memory::Host { pinned: true });
        assert!(!declared.is_refined_by(&pinned));
        assert!(pinned.is_refined_by(&pinned));
        assert!(!pinned.is_refined_by(&actual));
    }

    #[test]
    fn test_array_type_with_mismatched_sharding_rank() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(
            ArrayType::new(F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(2)])).with_sharding(sharding),
            Err(ShardingError::ShardingRankMismatch { sharding_rank: 1, array_rank: 2 }),
        );
    }

    #[test]
    fn test_array_type_validate_zero() {
        for data_type in [DataType::F32, DataType::Zero] {
            assert_eq!(ArrayType::scalar(data_type).validate_zero(), Ok(()));
        }
        for data_type in [DataType::Token, DataType::F8E8M0FNU] {
            assert_eq!(
                ArrayType::new_static(data_type, [0]).validate_zero(),
                Err(TypeError::invalid(format!("data type `{data_type}` cannot represent zero"))),
            );
        }
    }

    #[test]
    fn test_array_type_validate_one() {
        for data_type in [DataType::F32, DataType::F8E8M0FNU] {
            assert_eq!(ArrayType::scalar(data_type).validate_one(), Ok(()));
        }
        for data_type in [DataType::Token, DataType::Zero] {
            assert_eq!(
                ArrayType::new_static(data_type, [0]).validate_one(),
                Err(TypeError::invalid(format!("data type `{data_type}` cannot represent one"))),
            );
        }
    }

    #[test]
    fn test_array_type_derive_identity_renaming() {
        let declared_variable = DimensionVariable::new("declared", DimensionBounds::non_negative(Some(8)).unwrap());
        let actual_variable = DimensionVariable::new("actual", DimensionBounds::non_negative(Some(4)).unwrap());
        let declared = ArrayType::new(F32, Shape::new(vec![Dimension::Dynamic(declared_variable.clone())]));
        let actual = ArrayType::new(F32, Shape::new(vec![Dimension::Dynamic(actual_variable.clone())]));

        let renaming =
            ArrayType::derive_identity_renaming(std::slice::from_ref(&declared), std::slice::from_ref(&actual))
                .unwrap();
        assert_eq!(renaming.rename(&declared_variable), actual_variable);
        assert_eq!(
            ArrayType::derive_identity_renaming(std::slice::from_ref(&declared), &[] as &[ArrayType]),
            Err(TypeError::invalid("declared type count 1 does not match actual type count 0")),
        );

        // Exact actual bounds do not make an array identity structurally interchangeable with a static axis.
        let exact = DimensionVariable::new("exact", DimensionBounds::new(3, Some(4)).unwrap());
        let symbolic_actual = ArrayType::new(F32, Shape::new(vec![exact.into()]));
        let static_declared = ArrayType::new_static(F32, [3]);
        assert_eq!(
            ArrayType::derive_identity_renaming([&static_declared], [&symbolic_actual]),
            Err(TypeError::invalid("dimension exact does not instantiate declared dimension 3")),
        );
        let refinements = ArrayTypeRefinements::new([&static_declared], [&symbolic_actual]).unwrap();
        assert_eq!(refinements.refine_actual(&symbolic_actual, &[]), Ok(static_declared));

        // An earlier recorded extent conflict takes precedence over a later metadata mismatch.
        assert_eq!(
            ArrayType::derive_identity_renaming(
                [&declared, &declared, &ArrayType::scalar(F32)],
                [&ArrayType::new_static(F32, [2]), &ArrayType::new_static(F32, [3]), &ArrayType::scalar(F64)],
            ),
            Err(TypeError::from(DimensionError::InputDimensionMismatch {
                dimension: "declared".to_string(),
                expected: 2,
                actual: 3,
            })),
        );
    }

    #[test]
    fn test_array_type_derive_identity_renaming_rejects_retained_and_statically_bound_variable() {
        let variable = DimensionVariable::new("n", DimensionBounds::non_negative(Some(8)).unwrap());
        let dynamic_type = ArrayType::new(F32, Shape::new(vec![variable.into()]));
        let static_type = ArrayType::new_static(F32, [3]);
        let declared = [dynamic_type.clone(), dynamic_type.clone()];
        let error = TypeError::invalid(
            "dimension variable n is renamed to n by one signature member and bound to static extent 3 by another",
        );

        // Full refinement can retain the relationship, but a pure identity substitution cannot express it.
        // Structural consumers must reject the narrower signature rather than silently erase that obligation.
        for actual in [[dynamic_type.clone(), static_type.clone()], [static_type.clone(), dynamic_type.clone()]] {
            assert_eq!(ArrayType::derive_identity_renaming(&declared, &actual), Err(error.clone()));
            let refinements = ArrayTypeRefinements::from_local_declarations(&declared, &actual).unwrap();
            assert_eq!(refinements.refine_declared(&dynamic_type, &[]).unwrap(), static_type);
        }
    }

    #[test]
    fn test_array_type_derive_identity_renaming_rejects_renamed_and_statically_bound_variable() {
        let declared_variable = DimensionVariable::new("n", DimensionBounds::non_negative(Some(8)).unwrap());
        let actual_variable = DimensionVariable::new("m", DimensionBounds::non_negative(Some(8)).unwrap());
        let declared = ArrayType::new(F32, Shape::new(vec![Dimension::Dynamic(declared_variable.clone())]));
        let dynamic_actual = ArrayType::new(F32, Shape::new(vec![Dimension::Dynamic(actual_variable.clone())]));
        let static_actual = ArrayType::new(F32, Shape::new(vec![Dimension::Static(3)]));
        let error = TypeError::invalid(
            "dimension variable n is renamed to m by one signature member and bound to static extent 3 by another",
        );

        // The conflict is detected regardless of the order in which the two observations are made.
        assert_eq!(
            ArrayType::derive_identity_renaming(
                &[declared.clone(), declared.clone()],
                &[dynamic_actual.clone(), static_actual.clone()],
            ),
            Err(error.clone()),
        );
        assert_eq!(
            ArrayType::derive_identity_renaming(
                &[declared.clone(), declared.clone()],
                &[static_actual.clone(), dynamic_actual.clone()],
            ),
            Err(error.clone()),
        );

        // The conflict is also detected across array and dimension members of one array IR signature.
        assert_eq!(
            ArrayIrType::derive_identity_renaming(
                &[
                    ArrayIrType::Array(declared.clone()),
                    ArrayIrType::Dimension(DimensionType::from(declared_variable.clone())),
                ],
                &[
                    ArrayIrType::Array(static_actual),
                    ArrayIrType::Dimension(DimensionType::from(actual_variable.clone())),
                ],
            ),
            Err(error),
        );

        // A first-class dimension with exact bounds carries the same fact as the array's static extent. Both
        // signature orders admit that agreement, but an unequal exact extent or an unknown extent still conflicts.
        for (lower, upper, accepted) in [(3, 4, true), (4, 5, false), (2, 5, false)] {
            let target = DimensionVariable::new("target", DimensionBounds::new(lower, Some(upper)).unwrap());
            let declared_dimension = ArrayIrType::Dimension(DimensionType::from(declared_variable.clone()));
            let actual_dimension = ArrayIrType::Dimension(DimensionType::from(target.clone()));
            let static_array = ArrayIrType::Array(ArrayType::new_static(F32, [3]));
            let declared_array = ArrayIrType::Array(declared.clone());
            for (declared_types, actual_types) in [
                (
                    [declared_array.clone(), declared_dimension.clone()],
                    [static_array.clone(), actual_dimension.clone()],
                ),
                ([declared_dimension, declared_array], [actual_dimension, static_array]),
            ] {
                let result = ArrayIrType::derive_identity_renaming(&declared_types, &actual_types);
                if accepted {
                    assert_eq!(result.unwrap().rename(&declared_variable), target);
                } else {
                    assert_eq!(
                        result,
                        Err(TypeError::invalid(
                            "dimension variable n is renamed to target by one signature member and bound to \
                             static extent 3 by another",
                        )),
                    );
                }
            }
        }

        // Consistent repeated renamings of the same variable still succeed.
        let renaming = ArrayType::derive_identity_renaming(
            &[declared.clone(), declared],
            &[dynamic_actual.clone(), dynamic_actual],
        )
        .unwrap();
        assert_eq!(renaming.rename(&declared_variable), actual_variable);
    }

    #[test]
    fn test_array_type_referent() {
        let scalar = ArrayType::scalar(F32);
        let vector = ArrayType::new_static(F64, [3]);
        assert_eq!(scalar.referent(), None);
        assert_eq!(vector.referent(), None);
        assert!(!scalar.is_reference());
        assert!(!vector.is_reference());
    }

    #[test]
    fn test_array_type_as_referent() {
        assert_eq!(ArrayType::scalar(F32).as_referent(), None);
        assert_eq!(ArrayType::new_static(F64, [3]).as_referent(), None);
    }

    #[test]
    fn test_array_type_as_array_type() {
        let r#type = ArrayType::new_static(F32, [3]);
        assert_eq!(r#type.as_array_type(), Ok(&r#type));

        // Generic reference-aware code can use the same bound for a universe with no references.
        fn assert_array_projection<T: ReferenceMemberType<Referent: ArrayMemberType>>() {}
        assert_array_projection::<ArrayType>();
    }

    #[test]
    fn test_array_type_check_matching_manual_variation() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let plain = ArrayType::scalar(DataType::F32);
        let invariant = plain.clone().with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
        let varying = plain
            .clone()
            .with_sharding(Sharding::replicated(mesh, 0).with_varying_manual_axes(["devices"]).unwrap())
            .unwrap();
        assert_eq!(ArrayType::check_matching_manual_variation("add", &[]), Ok(()));
        assert_eq!(ArrayType::check_matching_manual_variation("add", &[&plain, &invariant]), Ok(()));
        assert_eq!(ArrayType::check_matching_manual_variation("add", &[&varying, &varying]), Ok(()));
        assert_eq!(
            ArrayType::check_matching_manual_variation("add", &[&varying, &plain]),
            Err(TypeError::invalid(
                "`add` inputs must have matching varying manual axes; insert `parallel_vary` on the inputs that lack \
                 an axis, as `align_manual_variation` does",
            )),
        );
        assert_eq!(
            ArrayType::check_matching_manual_variation("add", &[&invariant, &varying]),
            Err(TypeError::invalid(
                "`add` inputs must have matching varying manual axes; insert `parallel_vary` on the inputs that lack \
                 an axis, as `align_manual_variation` does",
            )),
        );
    }

    #[test]
    fn test_array_type_validate_layout() {
        // Unspecified layouts and layouts that address the shape without aliasing are valid.
        let plain = ArrayType::new_static(F32, [3]);
        assert_eq!(plain.validate_layout(), Ok(()));
        assert_eq!(plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![8]))).validate_layout(), Ok(()));
        let tiled = Layout::Tiled(TiledLayout::new(vec![0], vec![Tile::new(vec![TileDimension::Sized(2)])]));
        assert_eq!(ArrayType::new_static(F32, [4]).with_layout(tiled).validate_layout(), Ok(()));

        // Layouts must match the array rank and must not alias elements.
        assert_eq!(
            plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![4, 4]))).validate_layout(),
            Err(TypeError::invalid("strided layout rank 2 does not match array rank 1")),
        );
        assert_eq!(
            ArrayType::new_static(F32, [2, 3])
                .with_layout(Layout::Strided(StridedLayout::new(vec![4, 4])))
                .validate_layout(),
            Err(TypeError::invalid(
                "strided layout stride 4 on axis 1 is smaller than the 8-byte span occupied by more minor axes and \
                 may alias array elements",
            )),
        );

        // Dynamic extents are checked at their lower bounds.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(2, Some(5)).unwrap());
        assert_eq!(
            ArrayType::new(F32, Shape::new(vec![Dimension::Dynamic(extent)]))
                .with_layout(Layout::Strided(StridedLayout::new(vec![2])))
                .validate_layout(),
            Err(TypeError::invalid(
                "strided layout stride 2 on axis 0 is smaller than the 4-byte span occupied by more minor axes and \
                 may alias array elements",
            )),
        );
    }

    #[test]
    fn test_array_type_resized_sharding() {
        // An unsharded input has no sharding to carry.
        let input = ArrayType::new_static(DataType::F32, [4, 4]);
        assert_eq!(input.resized_sharding(&[Dimension::Static(2), Dimension::Static(4)], "slice"), Ok(None));

        // The input sharding, including its reduction state, is carried through unchanged when every resized dimension
        // sharded over explicit axes keeps a divisible static size.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let explicit =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap()
                .with_unreduced_axes(["m"])
                .unwrap();
        let explicit_input = input.clone().with_sharding(explicit.clone()).unwrap();
        assert_eq!(
            explicit_input.resized_sharding(&[Dimension::Static(2), Dimension::Static(1)], "slice"),
            Ok(Some(explicit.clone())),
        );
        assert_eq!(
            explicit_input.resized_sharding(&[Dimension::Static(3), Dimension::Static(4)], "slice"),
            Err(TypeError::invalid(
                "`slice` on a dimension sharded over explicit mesh axes requires the output size (3) at axis 0 to be \
                 divisible by the mesh-axis product (2)",
            )),
        );

        // A dimension sharded only over manual axes has an explicit product of one, so any size is accepted, and a
        // dynamic output extent is not checked.
        let manual =
            Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["m"])]).unwrap();
        let manual_input = input.with_sharding(manual.clone()).unwrap();
        assert_eq!(
            manual_input.resized_sharding(&[Dimension::Static(4), Dimension::Static(3)], "slice"),
            Ok(Some(manual)),
        );
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(4)).unwrap());
        assert_eq!(
            explicit_input.resized_sharding(&[Dimension::Dynamic(extent), Dimension::Static(4)], "slice"),
            Ok(Some(explicit)),
        );
    }

    #[test]
    fn test_array_type_resized_sharding_overflow() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", usize::MAX, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let input = ArrayType::new_static(DataType::I32, [1])
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x", "y"])]).unwrap())
            .unwrap();
        assert_eq!(
            input.resized_sharding(&[Dimension::Static(0)], "slice"),
            Err(TypeError::invalid("`slice` mesh-axis product overflows `usize`")),
        );
    }

    #[test]
    fn test_array_type_refinements_new() {
        let n = DimensionVariable::new("n", DimensionBounds::non_negative(Some(8)).unwrap());
        let declared = ArrayType::new(F32, Shape::new(vec![n.clone().into(), n.clone().into()]));
        let inconsistent = ArrayType::new_static(F32, [2, 3]);

        // Repeated dimensions must agree even when their occurrences belong to just one input type.
        let error = ArrayTypeRefinements::new([&declared], [&inconsistent]).unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::InputDimensionMismatch { dimension: "n".to_string(), expected: 2, actual: 3 }),
        );
        let vector = ArrayType::new(F32, Shape::new(vec![n.clone().into()]));
        assert!(
            ArrayTypeRefinements::new(
                [&vector, &vector],
                [ArrayType::new_static(F32, [2]), ArrayType::new_static(F32, [3])],
            )
            .is_err()
        );

        // A conditional entry is valid, but its actual type alone does not prove that the array is square.
        let conditional = ArrayType::new(F32, Shape::new(vec![n.clone().into(), 3.into()]));
        assert!(!declared.is_refined_by(&conditional));
        let refinements = ArrayTypeRefinements::new([&declared], [&conditional]).unwrap();
        assert_eq!(refinements.identities_needing_runtime_checks(), &[n]);
        assert_eq!(refinements.refine_declared(&declared, &[]).unwrap(), ArrayType::new_static(F32, [3, 3]));

        // Singleton identities prove exact extents without requiring a runtime equality check.
        let exact = DimensionVariable::new("exact", DimensionBounds::new(2, Some(3)).unwrap());
        let exact = ArrayType::new(F32, Shape::new(vec![exact.clone().into(), exact.into()]));
        assert!(!ArrayType::new_static(F32, [2, 2]).is_refined_by(&exact));
        let refinements = ArrayTypeRefinements::new([ArrayType::new_static(F32, [2, 2])], [&exact]).unwrap();
        assert!(refinements.identities_needing_runtime_checks().is_empty());
    }

    #[test]
    fn test_array_type_refinements_from_local_declarations() {
        let variable = DimensionVariable::new("length", DimensionBounds::non_negative(Some(8)).unwrap());
        let vector = ArrayType::new(F32, Shape::new(vec![variable.into()]));
        let declared = [vector.clone(), vector, ArrayType::scalar(F32)];
        let actual =
            [ArrayType::new_static(F32, [2]), ArrayType::new_static(F32, [3]), ArrayType::scalar(DataType::I32)];
        // Recording the conflicting length stops construction before the later data type mismatch.
        let error = ArrayTypeRefinements::from_local_declarations(&declared, &actual).unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::InputDimensionMismatch { dimension: "length".to_string(), expected: 2, actual: 3 }),
        );
        // Reject unequal signature lengths before consuming any member or its facts.
        assert_eq!(
            ArrayTypeRefinements::from_local_declarations(&declared, &actual[..2]),
            Err(TypeError::invalid("declared type count 3 does not match actual type count 2")),
        );
    }

    #[test]
    fn test_array_type_refinements_identity_classes() {
        let first = DimensionVariable::new("first", DimensionBounds::non_negative(Some(8)).unwrap());
        let second = DimensionVariable::new("second", DimensionBounds::non_negative(Some(8)).unwrap());
        let actual = DimensionVariable::new("actual", DimensionBounds::new(2, Some(3)).unwrap());
        let declared = ArrayType::new(F32, Shape::new(vec![first.clone().into(), second.clone().into(), 2.into()]));
        let supplied = ArrayType::new(F32, Shape::new(vec![2.into(), 2.into(), actual.clone().into()]));
        let refinements = ArrayTypeRefinements::from_local_declarations([&declared], [&supplied]).unwrap();

        // Equal exact extents do not join unrelated declared or actual identities, including isolated observations.
        assert_eq!(
            refinements.identity_classes().collect::<Vec<_>>(),
            vec![
                TypeIdentityClass::new(std::slice::from_ref(&first), &[], true),
                TypeIdentityClass::new(std::slice::from_ref(&second), &[], true),
                TypeIdentityClass::new(&[], std::slice::from_ref(&actual), true),
            ],
        );
    }

    #[test]
    fn test_array_type_refinements_identity_classes_transitive_order() {
        let first = DimensionVariable::new("first", DimensionBounds::non_negative(Some(8)).unwrap());
        let second = DimensionVariable::new("second", DimensionBounds::non_negative(Some(8)).unwrap());
        let third = DimensionVariable::new("third", DimensionBounds::non_negative(Some(8)).unwrap());
        let actual_first = DimensionVariable::new("actual_first", DimensionBounds::non_negative(Some(8)).unwrap());
        let actual_second = DimensionVariable::new("actual_second", DimensionBounds::non_negative(Some(8)).unwrap());
        let declared = ArrayType::new(
            F32,
            Shape::new(vec![
                first.clone().into(),
                second.clone().into(),
                second.clone().into(),
                third.clone().into(),
                third.clone().into(),
                third.clone().into(),
            ]),
        );
        let actual = ArrayType::new(
            F32,
            Shape::new(vec![
                actual_first.clone().into(),
                actual_first.clone().into(),
                actual_second.clone().into(),
                actual_second.clone().into(),
                actual_first.clone().into(),
                2.into(),
            ]),
        );
        let refinements = ArrayTypeRefinements::from_local_declarations([&declared], [&actual]).unwrap();
        // A cycle and a late exact refinement fact form one class, with order independent of temporary union roots.
        let formal_members = [first.clone(), second, third];
        let actual_members = [actual_first.clone(), actual_second.clone()];
        assert_eq!(
            refinements.identity_classes().collect::<Vec<_>>(),
            vec![TypeIdentityClass::new(&formal_members, &actual_members, true)],
        );
        assert_eq!(refinements.refine_actual(&actual, &[]), Ok(ArrayType::new_static(F32, [2; 6])));
        // Several immutable references may retain distinct actual names in the checked class.
        assert_eq!(
            refinements.retype_actual(
                &ArrayType::new(F32, Shape::new(vec![first.into()])),
                &ArrayType::new(F32, Shape::new(vec![actual_second.clone().into()])),
                &actual_members,
            ),
            Ok(ArrayType::new(F32, Shape::new(vec![actual_second.into()]))),
        );
    }

    #[test]
    fn test_array_type_refinements_identity_classes_validate_later_facts() {
        let formal = DimensionVariable::new("formal", DimensionBounds::non_negative(Some(8)).unwrap());
        let actual = DimensionVariable::new("actual", DimensionBounds::non_negative(Some(2)).unwrap());
        let declared = ArrayType::new(F32, Shape::new(vec![formal.clone().into()]));
        let supplied = ArrayType::new(F32, Shape::new(vec![actual.clone().into()]));
        let refinements = ArrayTypeRefinements::from_local_declarations([&declared], [&supplied]).unwrap();
        let error = refinements
            .validate_refinement([&declared], [&ArrayType::new_static(F32, [2])], std::slice::from_ref(&formal))
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::BindingOutOfBounds {
                variable: "actual".to_string(),
                value: 2,
                bounds: actual.bounds(),
            }),
        );
        assert_eq!(
            refinements.identity_classes().collect::<Vec<_>>(),
            vec![TypeIdentityClass::new(std::slice::from_ref(&formal), std::slice::from_ref(&actual), false)],
        );
        // Valid output facts belong to the validation clone, so the invocation's public classes stay unchanged.
        assert_eq!(
            refinements.validate_refinement(
                [&declared],
                [&ArrayType::new_static(F32, [1])],
                std::slice::from_ref(&formal)
            ),
            Ok(()),
        );
        assert!(!refinements.identity_classes().next().unwrap().is_exact());
    }

    #[test]
    fn test_array_type_refinements_validate_refinement() {
        let batch = DimensionVariable::new("batch", DimensionBounds::non_negative(Some(8)).unwrap());
        let declared = ArrayType::new(F32, Shape::new(vec![Dimension::Dynamic(batch.clone())]));
        let actual_two = ArrayType::new(F32, Shape::new(vec![Dimension::Static(2)]));
        let actual_three = ArrayType::new(F32, Shape::new(vec![Dimension::Static(3)]));

        let refinements = ArrayTypeRefinements::from_local_declarations(
            &[declared.clone(), declared.clone()],
            &[actual_two.clone(), actual_two.clone()],
        )
        .unwrap();
        assert_eq!(
            refinements.validate_refinement(std::slice::from_ref(&declared), std::slice::from_ref(&actual_two), &[]),
            Ok(()),
        );
        // Exact nominal dimensions prove extents without making distinct non-exact variables interchangeable.
        for extent in [2, 3] {
            let exact = DimensionVariable::new("exact", DimensionBounds::new(extent, Some(extent + 1)).unwrap());
            let actual = ArrayType::new(F32, Shape::new(vec![exact.into()]));
            let result = refinements.validate_refinement(std::slice::from_ref(&declared), &[actual], &[]);
            if extent == 2 {
                assert_eq!(result, Ok(()));
            } else {
                assert_eq!(
                    result.unwrap_err().downcast_custom::<DimensionError>(),
                    Some(&DimensionError::InputDimensionMismatch {
                        dimension: "batch".to_string(),
                        expected: 2,
                        actual: 3,
                    })
                );
            }
        }
        let unrelated = ArrayType::new(
            F32,
            Shape::new(vec![DimensionVariable::new("batch", DimensionBounds::non_negative(Some(8)).unwrap()).into()]),
        );
        assert!(refinements.validate_refinement(std::slice::from_ref(&declared), &[unrelated], &[]).is_err());

        let error = ArrayTypeRefinements::from_local_declarations(
            &[declared.clone(), declared.clone()],
            &[actual_two, actual_three],
        )
        .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::InputDimensionMismatch { dimension: "batch".to_string(), expected: 2, actual: 3 }),
        );

        let error = ArrayTypeRefinements::default()
            .validate_refinement(
                &[declared.clone(), declared],
                &[
                    ArrayType::new(F32, Shape::new(vec![Dimension::Static(2)])),
                    ArrayType::new(F32, Shape::new(vec![Dimension::Static(3)])),
                ],
                std::slice::from_ref(&batch),
            )
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::InputDimensionMismatch { dimension: "batch".to_string(), expected: 2, actual: 3 }),
        );
    }

    #[test]
    fn test_array_type_refinements_refine_declared() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(8)).unwrap());
        let cols = DimensionVariable::new("cols", DimensionBounds::non_negative(Some(8)).unwrap());
        let depth = DimensionVariable::new("depth", DimensionBounds::non_negative(Some(8)).unwrap());
        let refinements = ArrayTypeRefinements::from_local_declarations(
            &[ArrayType::new(F32, Shape::new(vec![rows.clone().into(), cols.clone().into()]))],
            &[ArrayType::new(F32, Shape::new(vec![Dimension::Static(3), Dimension::Static(4)]))],
        )
        .unwrap();

        // A variable with a recorded exact extent is replaced by that extent; other variables stay symbolic.
        let declared = ArrayType::new(F32, Shape::new(vec![rows.clone().into(), cols.clone().into(), depth.into()]));
        assert_eq!(
            refinements.refine_declared(&declared, &[]),
            Ok(declared.clone().with_shape(Shape::new(vec![
                Dimension::Static(3),
                Dimension::Static(4),
                declared.shape().dimensions()[2].clone(),
            ]))),
        );

        // Exclusion applies per identity, so a partially excluded type is still refined at its other identities.
        assert_eq!(
            refinements.refine_declared(&declared, std::slice::from_ref(&rows)),
            Ok(declared.clone().with_shape(Shape::new(vec![
                rows.clone().into(),
                Dimension::Static(4),
                declared.shape().dimensions()[2].clone(),
            ]))),
        );

        // Refinement only substitutes extents, so metadata such as a sharding is kept exactly as declared.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharded = ArrayType::new(F32, Shape::new(vec![rows.clone().into()]))
            .with_sharding(Sharding::replicated(mesh, 1))
            .unwrap();
        assert_eq!(
            refinements.refine_declared(&sharded, &[]),
            Ok(sharded.clone().with_shape(Shape::new(vec![Dimension::Static(3)]))),
        );

        // Facts are derived from the complete signature, so an identity that a caller excludes when refining still
        // takes part in conflict detection.
        assert_eq!(
            ArrayTypeRefinements::from_local_declarations(
                &[
                    ArrayType::new(F32, Shape::new(vec![rows.clone().into()])),
                    ArrayType::new(F32, Shape::new(vec![rows.into()])),
                ],
                &[
                    ArrayType::new(F32, Shape::new(vec![Dimension::Static(3)])),
                    ArrayType::new(F32, Shape::new(vec![Dimension::Static(4)])),
                ],
            )
            .unwrap_err()
            .downcast_custom::<DimensionError>(),
            Some(&DimensionError::InputDimensionMismatch { dimension: "rows".to_string(), expected: 3, actual: 4 }),
        );
    }
}
