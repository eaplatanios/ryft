use std::borrow::{Borrow, Cow};
use std::collections::BTreeSet;
use std::fmt::Display;

use ryft_macros::Parameter;

use crate::arrays::broadcasting::Broadcastable;
use crate::arrays::sharding::ShardingError;
use crate::arrays::sharding::meshes::DeviceMesh;
use crate::arrays::sharding::shardings::{Sharding, ShardingDimension};
use crate::arrays::types::data::DataType;
use crate::arrays::types::dimensions::{Dimension, DimensionError, DimensionVariable, Shape, StaticShape};
use crate::arrays::types::layouts::Layout;
use crate::arrays::types::memories::Memory;
use crate::axes::Axis;
use crate::contexts::EagerContext;
use crate::parameters::Parameter;
use crate::programs::types::visit_type_signature_pairs;
use crate::programs::{Type, TypeError, TypeIdentityPosition, TypeIdentityRenaming, TypeRefinements, Typed, Value};

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

    /// Returns a copy of this [`ArrayType`] with its [`DataType`] replaced by the provided one, keeping its [`Shape`],
    /// [`Layout`], [`Sharding`], and [`Memory`] unchanged.
    #[inline]
    pub fn with_data_type<D: Into<DataType>>(mut self, data_type: D) -> Self {
        self.data_type = data_type.into();
        self
    }

    /// Returns a copy of this [`ArrayType`] with its [`Shape`] replaced by the provided one, keeping its [`DataType`],
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

    /// Returns a copy of this [`ArrayType`] with the provided [`Sharding`] replacing its current sharding metadata
    /// (or without any [`Sharding`] information when [`None`] is provided), after validating that any provided
    /// [`Sharding`] has the same rank as [`Self::shape`].
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

    /// Returns a copy of this [`ArrayType`] with the array residing in the provided [`Memory`]. This kind of placement
    /// information is metadata about where the array lives, and it does not affect the array's [`DataType`], [`Shape`],
    /// [`Layout`], or [`Sharding`] (for sharded arrays, every shard resides in its own device's memory of this kind).
    #[inline]
    pub fn with_memory<M: Into<Memory>>(mut self, memory: M) -> Self {
        self.memory = memory.into();
        self
    }

    /// Constructs a new "scalar" [`ArrayType`] with the provided [`DataType`]. The resulting [`ArrayType::shape`]
    /// will be a scalar (i.e., have rank 0).
    #[inline]
    pub fn scalar(data_type: DataType) -> Self {
        Self { data_type, shape: Shape::scalar(), layout: None, sharding: None, memory: Memory::default() }
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

    /// Returns a copy of this [`ArrayType`] whose [`Sharding`] (if any) has its unreduced and reduced axis sets cleared
    /// while its per-dimension placement and varying-manual axes are preserved. Array types with no [`Sharding`] are
    /// returned unchanged. Bilinear type-inference rules (e.g., elementwise multiplication rules) use this so the
    /// shared elementwise broadcast does not reject operands that only disagree on their reduction state, which those
    /// rules combine separately.
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

    /// Returns a copy of this [`ArrayType`] with a dimension inserted at the provided index. Rank-changing operations
    /// clear explicit [`Layout`] information because [`Layout`]s do not carry enough information to infer a correct
    /// stride or tiling for a newly inserted logical axis. [`Sharding`] information is preserved by inserting a
    /// replicated sharding dimension at the same index and shifting the existing dimension annotations.
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

    /// Returns a copy of this [`ArrayType`] with its `index`-th dimension removed, paired with the [`Dimension`] of the
    /// removed dimension. Rank-changing operations clear explicit [`Layout`] information because [`Layout`]s do not
    /// carry enough information to infer a correct stride or tiling after removing a logical axis. [`Sharding`]
    /// information is preserved when the removed dimension is replicated or unconstrained. When the removed dimension
    /// is sharded over manual mesh axes, those axes become varying manual axes because the value can still differ
    /// across shards even though the ranked array dimension is gone. Removing a dimension sharded over non-manual
    /// axes is rejected because there is no equivalent rank-independent metadata field for those axes.
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

    /// Returns a copy of this [`ArrayType`] with a replicated [`Sharding`] over the provided [`DeviceMesh`]. The
    /// [`Layout`] information and [`Memory`] placement are preserved.
    pub fn replicated(&self, mesh: &DeviceMesh) -> Result<Self, ShardingError> {
        self.clone().with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), self.shape.rank()))
    }

    /// Checks that `actual` can instantiate `declared` and records what that instantiation implies for the declared
    /// [`DimensionVariable`]s (variable renamings in `renaming` and static extent bindings in `refinements`) for
    /// one declared/actual [`ArrayType`] pair.
    ///
    /// This is the single-pair step used by the [`Type::derive_identity_renaming`] implementations that fold over
    /// an entire declared/actual signature like [`ArrayType`]'s own, and [`ArrayIrType`](crate::ArrayIrType)'s, which
    /// encounters array types as individual elements of a mixed signature. It is a separate function so that those
    /// callers can thread the same two accumulators through every pair of the signature, which is what makes a declared
    /// [`DimensionVariable`] that appears in several signature positions get checked consistently rather than
    /// pair-by-pair.
    ///
    /// The `actual` type must have the `declared` type's element [`DataType`] and [`Memory`] placement, refine its
    /// optional [`Layout`] and [`Sharding`], and have the same rank. Each pair of corresponding dimensions then
    /// contributes:
    ///
    ///   - nothing, when both extents are static and equal,
    ///   - a `declared -> actual` variable renaming in `renaming`, when both dimensions are dynamic and the declared
    ///     variable's bounds contain the actual variable's bounds, or
    ///   - a static binding in `refinements`, when the declared dimension is dynamic and its bounds admit the actual
    ///     static extent; a later member observing a different extent for the same variable is rejected.
    ///
    /// Every other combination (e.g., mismatched static extents, a static declared dimension instantiated by a dynamic
    /// one, or actual bounds the declared variable's bounds do not contain) fails with a [`TypeError`]. A conflict
    /// *between* the two accumulators, where one signature position renames a variable that another position binds to
    /// a static extent, is deliberately left to the signature-folding callers, which call
    /// [`ArrayTypeRefinements::require_disjoint_from`] once after folding over the complete signature,
    /// so that detection does not depend on the order in which the two conflicting observations arrive.
    ///
    /// # Example
    ///
    /// Given the dynamic dimension variables `n: [0, 8)` and `m: [0, 8)`, folding over the declared signature
    /// `(f32[n, 2], f32[n])` behaves as follows for these actual signatures:
    ///
    ///   - `(f32[m, 2], f32[m])`: Records the renaming `n -> m` and accepts its consistent repetition.
    ///   - `(f32[3, 2], f32[4])`: Fails because `n` is bound to static extent `3` and then observed as `4`.
    ///   - `(f32[m, 2], f32[3])`: The extension itself succeeds, recording the renaming `n -> m` and the static binding
    ///     `n = 3`. The caller's subsequent [`ArrayTypeRefinements::require_disjoint_from`] check then rejects the
    ///     signature.
    pub fn extend_identity_renaming(
        declared: &Self,
        actual: &Self,
        renaming: &mut TypeIdentityRenaming<DimensionVariable>,
        refinements: &mut ArrayTypeRefinements,
    ) -> Result<(), TypeError> {
        if !declared.non_shape_components_are_refined_by(actual) || declared.rank() != actual.rank() {
            return Err(TypeError::invalid(format!("type {actual} cannot instantiate declared type {declared}")));
        }
        declared.shape().dimensions().iter().zip(actual.shape().dimensions()).try_for_each(
            |(declared_dimension, actual_dimension)| match (declared_dimension, actual_dimension) {
                (Dimension::Static(declared), Dimension::Static(actual)) if declared == actual => Ok(()),
                (Dimension::Dynamic(declared), Dimension::Dynamic(actual))
                    if declared.bounds().contains_bounds(actual.bounds()) =>
                {
                    renaming.insert(declared.clone(), actual.clone())
                }
                (Dimension::Dynamic(declared), Dimension::Static(actual)) if declared.bounds().contains(*actual) => {
                    refinements.bind(declared, *actual)
                }
                _ => Err(TypeError::invalid(format!(
                    "dimension {actual_dimension} does not instantiate declared dimension {declared_dimension}",
                ))),
            },
        )
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

    fn identities(&self) -> impl Iterator<Item = (TypeIdentityPosition, &Self::Identity)> {
        self.shape
            .dimensions()
            .iter()
            .filter_map(Dimension::variable)
            .map(|variable| (TypeIdentityPosition::Reference, variable))
    }

    fn derive_identity_renaming(
        declared: &[Self],
        actual: &[Self],
    ) -> Result<TypeIdentityRenaming<Self::Identity>, TypeError> {
        let mut renaming = TypeIdentityRenaming::new();
        let mut refinements = ArrayTypeRefinements::default();
        visit_type_signature_pairs(declared, actual, |declared, actual| {
            Self::extend_identity_renaming(declared, actual, &mut renaming, &mut refinements)
        })?;
        refinements.require_disjoint_from(&renaming)?;
        Ok(renaming)
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
    type DispatchDomain = EagerContext<Self>;
    type ExecutionDomain = EagerContext<Self>;

    #[inline]
    fn dispatch_domain(&self) -> EagerContext<Self> {
        EagerContext::new()
    }

    #[inline]
    fn execution_domain(&self) -> EagerContext<Self> {
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

/// [`TypeRefinements`] established while refining one complete [`ArrayType`] signature.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ArrayTypeRefinements {
    /// Dynamic-[`DimensionVariable`]-to-extent bindings in first-observation order.
    bindings: Vec<(DimensionVariable, usize)>,
}

impl ArrayTypeRefinements {
    /// Records one concrete extent for `variable`, rejecting a conflicting observation.
    pub fn bind(&mut self, variable: &DimensionVariable, extent: usize) -> Result<(), TypeError> {
        match self.bindings.iter().find_map(|(candidate, expected)| (candidate == variable).then_some(*expected)) {
            Some(expected) if expected != extent => Err(DimensionError::InputDimensionMismatch {
                dimension: variable.to_string(),
                expected,
                actual: extent,
            }
            .into()),
            Some(_) => Ok(()),
            None => {
                self.bindings.push((variable.clone(), extent));
                Ok(())
            }
        }
    }

    /// Validates one observed extent against an established input fact, or binds it when `variable` belongs to the
    /// validated boundary's closed identity signature. This is the validation-phase counterpart of [`Self::bind`].
    /// For a variable that already has a recorded extent, the two behave identically (i.e., a matching observation
    /// is accepted and a differing one is rejected). They differ on a variable with no recorded fact. [`Self::bind`]
    /// treats the refinement set as *open* and records the first observation as a new fact, which is correct while
    /// establishing facts from a complete input signature (i.e., as [`TypeRefinements::establish`] and
    /// [`ArrayType::extend_identity_renaming`] do, where every variable may legitimately appear for the first time).
    /// [`Self::validate_or_bind`] instead admits an unrecorded variable only when it appears in `closed_identities`,
    /// the boundary's complete closed identity set (i.e., an identity established by the formal input signature, whose
    /// first concrete fact may only become observable at an output, such as a first-class dimension input whose type is
    /// strictly identity plus bounds, or an identity defined by an instruction inside the program or region). In either
    /// case it delegates to [`Self::bind`] so repeated observations across several outputs must still agree; this is
    /// sound without inspecting runtime payloads because structural region closure already proves that every output
    /// reference identity is consumed or defined. Every variable outside the closed set is rejected, because an extent
    /// claimed for an identity foreign to the boundary cannot be justified. In short, use [`Self::bind`] where
    /// observations *create* facts, and use this function where observations must be *justified* against the boundary's
    /// identity set, as [`TypeRefinements::validate`] does when checking an output signature against the facts
    /// established from its input boundary.
    pub fn validate_or_bind(
        &mut self,
        variable: &DimensionVariable,
        extent: usize,
        closed_identities: &[DimensionVariable],
    ) -> Result<(), TypeError> {
        match self.bindings.iter().find_map(|(candidate, value)| (candidate == variable).then_some(*value)) {
            Some(expected) if expected != extent => Err(DimensionError::InputDimensionMismatch {
                dimension: variable.to_string(),
                expected,
                actual: extent,
            }
            .into()),
            Some(_) => Ok(()),
            None if closed_identities.contains(variable) => self.bind(variable, extent),
            None => Err(TypeError::invalid(format!(
                "dimension identity {variable} does not belong to the validated boundary signature",
            ))),
        }
    }

    /// Validates that `actual` refines `declared` while visiting each axis where a declared dynamic dimension is met
    /// by a concrete static extent (i.e., each *dynamic-to-static refinement*). This is the shared per-type validation
    /// engine behind both [`TypeRefinements::establish`] and [`TypeRefinements::validate`]. The structural refinement
    /// rules live here exactly once, while the provided `visitor` decides what an observed concrete extent *means*
    /// (e.g., [`Self::bind`] when establishing facts from an input signature versus [`Self::validate_or_bind`] when
    /// justifying an output signature against them). It first requires that `actual` refines every non-shape component
    /// of `declared` and matches its rank, and then walks the two shapes axis by axis:
    ///
    ///   - Equal static extents, or dynamic dimensions with the same [`DimensionVariable`], refine trivially and
    ///     contribute no visit, because no new concrete fact is observed.
    ///   - A declared dynamic dimension met by a static extent inside its declared bounds is the refinement this
    ///     function exists to surface: `visitor` is invoked with the declared variable and the observed extent, and
    ///     any error it returns aborts the walk and propagates to the caller.
    ///   - A declared dynamic dimension met by a static extent outside its declared bounds fails with
    ///     [`DimensionError::BindingOutOfBounds`], and every remaining combination (mismatched static extents, distinct
    ///     dynamic variables, or a static dimension met by a dynamic one) fails as not refining the declared type.
    ///
    /// # Parameters
    ///
    ///   - `declared`: [`ArrayType`] declared by the boundary signature, whose dynamic dimensions may legitimately be
    ///     met by concrete extents.
    ///   - `actual`: Observed [`ArrayType`] that must refine `declared`.
    ///   - `visitor`: Callback invoked once per in-bounds dynamic-to-static axis, in axis order, with the declared
    ///     [`DimensionVariable`] and the observed static extent. A repeated variable is visited once per axis that
    ///     observes it, so visitors are responsible for requiring that repeated observations agree.
    pub fn visit_dynamic_to_static_refinements(
        declared: &ArrayType,
        actual: &ArrayType,
        mut visitor: impl FnMut(&DimensionVariable, usize) -> Result<(), TypeError>,
    ) -> Result<(), TypeError> {
        if !declared.non_shape_components_are_refined_by(actual) || declared.rank() != actual.rank() {
            return Err(TypeError::invalid(format!("type {actual} does not refine declared type {declared}",)));
        }
        declared
            .shape()
            .dimensions()
            .iter()
            .zip(actual.shape().dimensions())
            .try_for_each(|(declared, actual)| match (declared, actual) {
                (Dimension::Static(declared), Dimension::Static(actual)) if declared == actual => Ok(()),
                (Dimension::Dynamic(declared), Dimension::Dynamic(actual)) if declared == actual => Ok(()),
                (Dimension::Dynamic(declared), Dimension::Static(actual)) if declared.bounds().contains(*actual) => {
                    visitor(declared, *actual)
                }
                (Dimension::Dynamic(declared), Dimension::Static(actual)) => Err(DimensionError::BindingOutOfBounds {
                    variable: declared.to_string(),
                    value: *actual,
                    bounds: declared.bounds(),
                }
                .into()),
                _ => Err(TypeError::invalid(format!(
                    "dimension {actual} does not refine declared dimension {declared}",
                ))),
            })
    }

    /// Rejects a [`DimensionVariable`] that one signature member binds to a static extent while another member renames
    /// it to a live [`DimensionVariable`], since no instantiation can satisfy both observations. Complete signature
    /// renaming derivations call this after folding over every member, so the check is independent of the order in
    /// which the two conflicting observations were recorded.
    pub fn require_disjoint_from(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<(), TypeError> {
        for (variable, extent) in &self.bindings {
            if let Some((_, target)) = renaming.replacements().iter().find(|(source, _)| source == variable) {
                return Err(TypeError::invalid(format!(
                    "dimension variable {variable} is renamed to {target} by one signature member and bound to \
                     static extent {extent} by another",
                )));
            }
        }
        Ok(())
    }
}

impl TypeRefinements<ArrayType> for ArrayTypeRefinements {
    fn establish<D: IntoIterator, A: IntoIterator>(declared: D, actual: A) -> Result<Self, TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<ArrayType>,
        A::Item: Borrow<ArrayType>,
    {
        let mut refinements = Self::default();
        visit_type_signature_pairs(declared, actual, |declared, actual| {
            Self::visit_dynamic_to_static_refinements(declared, actual, |variable, extent| {
                refinements.bind(variable, extent)
            })
        })?;
        Ok(refinements)
    }

    fn validate<D: IntoIterator, A: IntoIterator>(
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
        visit_type_signature_pairs(declared, actual, |declared, actual| {
            Self::visit_dynamic_to_static_refinements(declared, actual, |variable, extent| {
                refinements.validate_or_bind(variable, extent, closed_identities)
            })
        })
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
    use crate::arrays::types::dimensions::{DimensionBounds, DimensionType, DimensionVariable};
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
            ArrayType::derive_identity_renaming(std::slice::from_ref(&declared), &[]),
            Err(TypeError::invalid("declared type count 1 does not match actual type count 0")),
        );
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
                    ArrayIrType::Dimension(DimensionType::new(declared_variable.clone())),
                ],
                &[
                    ArrayIrType::Array(static_actual),
                    ArrayIrType::Dimension(DimensionType::new(actual_variable.clone())),
                ],
            ),
            Err(error),
        );

        // Consistent repeated renamings of the same variable still succeed.
        let renaming = ArrayType::derive_identity_renaming(
            &[declared.clone(), declared],
            &[dynamic_actual.clone(), dynamic_actual],
        )
        .unwrap();
        assert_eq!(renaming.rename(&declared_variable), actual_variable);
    }

    #[test]
    fn test_array_type_refinements_bind_repeated_dimensions_across_complete_signatures() {
        let batch = DimensionVariable::new("batch", DimensionBounds::non_negative(Some(8)).unwrap());
        let declared = ArrayType::new(F32, Shape::new(vec![Dimension::Dynamic(batch.clone())]));
        let actual_two = ArrayType::new(F32, Shape::new(vec![Dimension::Static(2)]));
        let actual_three = ArrayType::new(F32, Shape::new(vec![Dimension::Static(3)]));

        let refinements = ArrayTypeRefinements::establish(
            &[declared.clone(), declared.clone()],
            &[actual_two.clone(), actual_two.clone()],
        )
        .unwrap();
        assert_eq!(
            refinements.validate(std::slice::from_ref(&declared), std::slice::from_ref(&actual_two), &[]),
            Ok(()),
        );
        let error = ArrayTypeRefinements::establish(&[declared.clone(), declared.clone()], &[actual_two, actual_three])
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::InputDimensionMismatch { dimension: "batch".to_string(), expected: 2, actual: 3 }),
        );

        let error = ArrayTypeRefinements::default()
            .validate(
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
}
