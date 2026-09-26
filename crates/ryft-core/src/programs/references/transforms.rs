use std::convert::Infallible;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use ryft_macros::Parameter;

use crate::batching::{BatchAxis, BatchingError};
use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::programs::effects::ReferenceAccessMode;
use crate::programs::references::types::ReferenceType;
use crate::programs::types::{Type, TypeError};
use crate::programs::values::ValueId;

/// Represents whether two views of the same reference allocation cover separate parts, exactly the same part, or
/// potentially overlapping parts, as determined by [`ReferenceTransform::overlap`]. Both paths apply transforms
/// starting from the complete allocation. For example, `root[0]` and `root[1]` address different elements and are
/// disjoint, while `root[i]` and `root[j]` may overlap when the values of `i` and `j` are unknown. Each symbolic index
/// in a path has a binding identifying the program value that supplies it; that binding does not imply that the index's
/// runtime value is known.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReferenceViewOverlap {
    /// The two paths _provably_ address disjoint parts of the root.
    Disjoint,

    /// The two paths _provably_ address exactly the same part of the root.
    Same,

    /// The two paths may overlap: they address intersecting parts, or a transform depends on a symbol whose binding
    /// cannot prove the paths identical or disjoint.
    MayOverlap,
}

/// Uninhabited binding of [`ReferenceTransformPath`]s that only ever carry static [`BoundReferenceTransform`]s, such
/// as the path of an eager array reference handle. Every transform in such paths has empty bindings; consumers must
/// reject transforms that require dynamic bindings because no binding value can be supplied for them.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum NoReferenceTransformBinding {}

/// Metadata describing one selection applied to a reference allocation, carried by each access operation that applies
/// it. A transform stores information such as an array axis and a static index, or indicates that a dynamic binding
/// supplies an index. It stores neither the allocation nor the dynamic value or its instruction input position.
/// [`BoundReferenceTransform`] pairs the transform with its binding values or program identities and
/// [`ReferenceAccessDescriptor`](crate::ReferenceAccessDescriptor) identifies the corresponding range
/// of instruction inputs.
///
/// The `'static`, [`Send`], and [`Sync`] bounds allow [`ReferenceViewAnalysis`](crate::ReferenceViewAnalysis)
/// to be retained as type-erased metadata in the region's transform cache (refer to
/// [`RegionRef::transform`](crate::RegionRef::transform) for more information on that). In particular, `'static`
/// prevents transforms from borrowing temporary data; it does not require their instances to live forever. Owned
/// transform metadata satisfies this bound. Equality supports revalidation against a fresh analysis, and hashing lets
/// paths serve as part of eager reference handles' identities.
pub trait ReferenceTransform: 'static + Clone + Debug + Display + PartialEq + Eq + Hash + Send + Sync {
    /// Input type universe in which this [`ReferenceTransform`] binds its dynamic inputs.
    type Type: Type;

    /// Referent type family consumed and produced by this transform.
    type Referent: Type;

    /// Returns the number of dynamic inputs consumed by this transform.
    fn binding_count(&self) -> usize;

    /// Validates this transform's dynamic input types against the provided input referent. Implementations check the
    /// binding count as well as family-specific requirements, such as scalar integer array indices in the referent's
    /// memory space. Bindings belong to the input universe and need not themselves have referent types.
    ///
    /// # Parameters
    ///
    ///   - `input`: Referent type before this transform is applied.
    ///   - `bindings`: Types of this transform's dynamic inputs in the transform family's binding order.
    fn validate_bindings(&self, input: &Self::Referent, bindings: &[&Self::Type]) -> Result<(), TypeError>;

    /// Returns the referent type after applying this transform to `input`. Rejects transforms that cannot apply to that
    /// referent, including transforms through which an update could not be written back to `input`.
    fn output_type(&self, input: &Self::Referent) -> Result<Self::Referent, TypeError>;

    /// Returns the referent type that a read-only access selects by applying this transform to `input`. Unlike
    /// [`output_type`](Self::output_type), this function need not prove that an update through the transform could be
    /// written back to `input`, because a read-only access never writes back. Families whose write-back validation
    /// is expensive override this function to skip it. Note that, the default implementation delegates to
    /// [`output_type`](Self::output_type).
    fn read_type(&self, input: &Self::Referent) -> Result<Self::Referent, TypeError> {
        self.output_type(input)
    }

    /// Returns whether `lhs` and `rhs` address separate parts, exactly the same part, or potentially overlapping
    /// parts of the same reference allocation, without executing the program. Both paths must be relative to the
    /// allocation's root, meaning its entire referenced value, rather than to an intermediate view. For example, an
    /// access that selects `[2]` through a view of `root[1]` is compared through the full path `root[1][2]`, not `[2]`.
    /// Views of `root[0]` and `root[1]` are disjoint, while views of `root[i]` and `root[j]` may overlap when their
    /// indices are unknown. Equal symbol bindings identify the same program value, but different bindings do not prove
    /// that the runtime values differ. Comparing symbolic views must also account for the transforms themselves,
    /// including any clamping.
    ///
    /// Note that an empty path covers the complete allocation, two empty paths are considered
    /// [`Same`](ReferenceViewOverlap::Same), and an empty path may overlap with a path covering only part of the
    /// allocation. Paths are validated when they are derived, and implementations may conservatively return
    /// [`MayOverlap`](ReferenceViewOverlap::MayOverlap) for a malformed path instead of failing.
    ///
    /// # Parameters
    ///
    ///   - `type`: [`Type`] of the root reference from which both paths start.
    ///   - `lhs`: [`BoundReferenceTransform`]s describing the first view to compare, starting from the root.
    ///   - `rhs`: [`BoundReferenceTransform`]s describing the second view to compare, starting from the root.
    fn overlap(
        r#type: &Self::Type,
        lhs: &[BoundReferenceTransform<Self>],
        rhs: &[BoundReferenceTransform<Self>],
    ) -> ReferenceViewOverlap;
}

/// Optional batching capability for a [`ReferenceTransform`]. Implementations adjust a transform when its source
/// reference gains a batch axis. Reference families that support analysis and discharge without batching need only
/// implement [`ReferenceTransform`] but batching rules additionally require this trait.
pub trait BatchableReferenceTransform: ReferenceTransform {
    /// Moves the batch axis of a source reference through this [`ReferenceTransform`] mapping. The batch axis of a
    /// reference is an axis of its packed referent that the per-item transform never sees. The batched transform
    /// therefore addresses the same part of each packed item as the original transform addresses in the unbatched
    /// input, and the resulting view has its own batch axis. This is pure axis arithmetic; the transform's dynamic
    /// binding requirements are unchanged, and a replicated `batch_axis` returns the transform unchanged and
    /// replicated.
    ///
    /// # Parameters
    ///
    ///   - `type`: Packed reference [`Type`] of the batched source, with the batch axis inserted.
    ///   - `batch_axis`: Batch axis positioned in the packed referent of `type`.
    ///
    /// # Errors
    ///
    /// Returns a [`BatchingError`] when this family cannot carry `batch_axis` through the transform (e.g., a family
    /// without axes rejects every mapped axis, and a static array slice cannot span a dynamically sized batch axis).
    fn batch(&self, r#type: &Self::Type, batch_axis: BatchAxis) -> Result<(Self, BatchAxis), BatchingError>;
}

/// Uninhabited transform type for a referent family `T` in an input universe `U` that supports only whole-root
/// accesses. [`ReferenceTransformPath`]s containing this type are necessarily empty; the function-pointer marker
/// imposes no thread-safety or equality requirements on the type descriptors.
pub enum NoReferenceTransform<T: Type, U: Type> {
    #[doc(hidden)]
    Never(Infallible, PhantomData<fn() -> (T, U)>),
}

impl<T: Type, U: Type> Copy for NoReferenceTransform<T, U> {}

impl<T: Type, U: Type> Clone for NoReferenceTransform<T, U> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Type, U: Type> Debug for NoReferenceTransform<T, U> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Never(never, _) => Debug::fmt(never, formatter),
        }
    }
}

impl<T: Type, U: Type> Display for NoReferenceTransform<T, U> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Never(never, _) => Display::fmt(never, formatter),
        }
    }
}

impl<T: Type, U: Type> PartialEq for NoReferenceTransform<T, U> {
    #[inline]
    fn eq(&self, _other: &Self) -> bool {
        match self {
            Self::Never(never, _) => match *never {},
        }
    }
}

impl<T: Type, U: Type> Eq for NoReferenceTransform<T, U> {}

impl<T: Type, U: Type> Hash for NoReferenceTransform<T, U> {
    #[inline]
    fn hash<H: Hasher>(&self, _state: &mut H) {
        match self {
            Self::Never(never, _) => match *never {},
        }
    }
}

impl<T: 'static + Type, U: 'static + Type> ReferenceTransform for NoReferenceTransform<T, U> {
    type Type = U;
    type Referent = T;

    fn binding_count(&self) -> usize {
        match self {
            Self::Never(never, _) => match *never {},
        }
    }

    fn validate_bindings(&self, _input: &T, _bindings: &[&U]) -> Result<(), TypeError> {
        match self {
            Self::Never(never, _) => match *never {},
        }
    }

    fn output_type(&self, _input: &T) -> Result<T, TypeError> {
        match self {
            Self::Never(never, _) => match *never {},
        }
    }

    fn overlap(
        _type: &U,
        _lhs: &[BoundReferenceTransform<Self>],
        _rhs: &[BoundReferenceTransform<Self>],
    ) -> ReferenceViewOverlap {
        ReferenceViewOverlap::Same
    }
}

impl<T: 'static + Type, U: 'static + Type> BatchableReferenceTransform for NoReferenceTransform<T, U> {
    fn batch(&self, _type: &U, _batch_axis: BatchAxis) -> Result<(Self, BatchAxis), BatchingError> {
        match self {
            Self::Never(never, _) => match *never {},
        }
    }
}

/// A [`ReferenceTransform`] grouped together with its dynamic bindings. [`ReferenceTransformPath`] applies bound
/// transforms in order from the complete root. `Transform` owns transform metadata, such as an array axis or a static
/// slice while `Binding` supplies the ordinary values required by [`ReferenceTransform::binding_count`]. Static
/// transforms carry no bindings.
///
/// During analysis, bindings are [`ValueId`]s in the access instruction's region. During discharge, they are values
/// in the reconstruction context. For example, an array `Index { axis: 0, index: Dynamic }` consumes one binding and
/// that binding identifies the runtime index value, independently of its position among the instruction's inputs.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct BoundReferenceTransform<Transform: ReferenceTransform, Binding = ValueId> {
    /// [`ReferenceTransform`] of this [`BoundReferenceTransform`].
    transform: Transform,

    /// Dynamic bindings consumed by this transform, in the order defined by the transform family.
    bindings: Vec<Binding>,
}

impl<Transform: ReferenceTransform, Binding> BoundReferenceTransform<Transform, Binding> {
    /// Returns the [`ReferenceTransform`] of this [`BoundReferenceTransform`].
    #[inline]
    pub fn transform(&self) -> &Transform {
        &self.transform
    }

    /// Returns the binding of each symbol of the [`ReferenceTransform`], in the order returned
    /// by [`ReferenceTransform::binding_count`].
    #[inline]
    pub fn bindings(&self) -> &[Binding] {
        self.bindings.as_slice()
    }
}

// TODO(eaplatanios): Review from here onwards.

/// Sequence of [`BoundReferenceTransform`]s from a reference root to one derived reference (i.e., a view), in the order
/// they are applied. `Transform` is the type of each transform, such as
/// [`ArrayReferenceTransform`](crate::ArrayReferenceTransform), and `Binding` represents the inputs needed by a
/// symbolic transform. Each [`BoundReferenceTransform`] pairs a `Transform` with a vector of `Binding`s. The path
/// stores these bound transforms, but neither the root allocation nor its identity;
/// [`ReferenceAnalysis`](crate::ReferenceAnalysis) identifies the root when analyzing a [`Program`](crate::Program).
///
/// For example, `root[row][column]` produces two transforms: the first addresses a row in the root, and the second
/// addresses an element in that row. With `Transform = ArrayReferenceTransform` and `Binding = ValueId`, the bound
/// transforms describe the two indexing operations and store the program identities of `row` and `column`. During
/// discharge, `Binding = C::Value` stores their values in the reconstruction context instead, so the same path
/// traversal can reapply the transforms without looking up source program identities.
///
/// Eager reference handles resolve indices immediately into static transforms and use
/// [`NoReferenceTransformBinding`]. Their transforms have empty binding vectors. Dynamic paths instead bind each
/// transform to the consecutive inputs declared by the access descriptor, consuming exactly
/// [`ReferenceTransform::binding_count`] bindings per transform.
///
/// The empty path denotes the complete root. Complete root handles, capture constants, and forwarded complete
/// references carry it. A path belongs to one access instruction and therefore stays inside that instruction's region:
/// references that cross region boundaries always denote complete roots, and accesses in the receiving region carry
/// their own paths. Equality and hashing compare transforms and bindings, not the identities of the reference handles
/// or the array elements addressed by different transform sequences.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct ReferenceTransformPath<Transform: ReferenceTransform, Binding = ValueId> {
    /// [`BoundReferenceTransform`]s in this [`ReferenceTransformPath`] in the order they are applied, starting from
    /// the root.
    bound_transforms: Vec<BoundReferenceTransform<Transform, Binding>>,
}

impl<Transform: ReferenceTransform, Binding> ReferenceTransformPath<Transform, Binding> {
    /// Returns the empty [`ReferenceTransformPath`] denoting the complete root.
    pub const fn root() -> Self {
        Self { bound_transforms: Vec::new() }
    }

    /// Resolves an access's ordered transforms against consecutive dynamic bindings. The bindings must contain exactly
    /// the sum of [`ReferenceTransform::binding_count`] across all transforms. This function checks only the binding
    /// layout; [`infer_reference_view_type`] validates binding types and transform semantics.
    ///
    /// # Parameters
    ///
    ///   - `transforms`: Transforms applied in order from the complete root.
    ///   - `bindings`: Dynamic bindings grouped in transform order, with each transform consuming its declared binding
    ///     count.
    pub fn from_transforms(transforms: &[Transform], bindings: &[Binding]) -> Result<Self, ProgramError>
    where
        Binding: Clone,
    {
        let mut remaining = bindings;
        let mut bound_transforms = Vec::with_capacity(transforms.len());
        for transform in transforms {
            let count = transform.binding_count();
            if count > remaining.len() {
                return Err(ProgramError::MalformedProgram(format!(
                    "reference transform requires {count} bindings but only {} remain",
                    remaining.len(),
                )));
            }
            let (current, rest) = remaining.split_at(count);
            bound_transforms.push(BoundReferenceTransform { transform: transform.clone(), bindings: current.to_vec() });
            remaining = rest;
        }
        if !remaining.is_empty() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference transform path has {} extra bindings",
                remaining.len(),
            )));
        }
        Ok(Self { bound_transforms })
    }

    /// Returns whether this [`ReferenceTransformPath`] denotes the complete root (i.e., whether it is empty).
    #[inline]
    pub fn is_root(&self) -> bool {
        self.bound_transforms.is_empty()
    }

    /// Returns the [`BoundReferenceTransform`]s in this [`ReferenceTransformPath`] in the order they are applied,
    /// starting from the root.
    #[inline]
    pub fn bound_transforms(&self) -> &[BoundReferenceTransform<Transform, Binding>] {
        self.bound_transforms.as_slice()
    }

    /// Returns the ordered [`ReferenceTransform`]s applied from the root outward, without their bindings.
    #[inline]
    pub fn transforms(&self) -> impl ExactSizeIterator<Item = &Transform> + DoubleEndedIterator {
        self.bound_transforms.iter().map(BoundReferenceTransform::transform)
    }

    /// Returns a copy of this [`ReferenceTransformPath`] extended by one more [`BoundReferenceTransform`] applied to
    /// its current end, associating `transform` with `bindings`. The caller must supply the number of bindings declared
    /// by [`ReferenceTransform::binding_count`], in the transform family's order. This generic container does not
    /// validate the transform or its bindings.
    pub fn with_bound_transform(&self, transform: Transform, bindings: Vec<Binding>) -> Self
    where
        Binding: Clone,
    {
        let mut bound_transforms = Vec::with_capacity(self.bound_transforms.len() + 1);
        bound_transforms.extend(self.bound_transforms.iter().cloned());
        bound_transforms.push(BoundReferenceTransform { transform, bindings });
        Self { bound_transforms }
    }

    /// Returns a copy of this [`ReferenceTransformPath`] extended by one more static [`ReferenceTransform`] applied to
    /// its current end. This is the shorthand of [`with_bound_transform`](Self::with_bound_transform) with no bindings.
    /// The caller must ensure that `transform` requires no dynamic bindings.
    #[inline]
    pub fn with_transform(&self, transform: Transform) -> Self
    where
        Binding: Clone,
    {
        self.with_bound_transform(transform, Vec::new())
    }

    /// Extends this [`ReferenceTransformPath`] in place by the bound transforms of `suffix`, applied in order after its
    /// current end. Unlike [`with_bound_transform`](Self::with_bound_transform), which copies the existing bound
    /// transforms into a new path, this moves the bound transforms of `suffix` without cloning either path, so
    /// extending a path by `k` transforms costs `O(k)` amortized time regardless of its current length. Like
    /// [`with_bound_transform`](Self::with_bound_transform), it does not validate the transforms or their bindings.
    pub fn append(&mut self, suffix: Self) {
        self.bound_transforms.extend(suffix.bound_transforms);
    }
}

impl<Transform: ReferenceTransform> ReferenceTransformPath<Transform, ValueId> {
    /// Returns the [`ReferenceViewOverlap`] between the parts this [`ReferenceTransformPath`] and `other` address
    /// within one root of type `root`, through [`ReferenceTransform::overlap`]. Both paths must be relative to that
    /// same root rather than to an intermediate view. Callers resolve each access's allocation root through
    /// [`ReferenceAnalysis`](crate::ReferenceAnalysis) before comparing access paths.
    #[inline]
    pub fn overlap(&self, other: &Self, root: &Transform::Type) -> ReferenceViewOverlap {
        Transform::overlap(root, self.bound_transforms(), other.bound_transforms())
    }
}

impl<Transform: ReferenceTransform, Binding> Default for ReferenceTransformPath<Transform, Binding> {
    #[inline]
    fn default() -> Self {
        Self::root()
    }
}

/// Validates dynamic bindings and derives the referent of the view produced by an ordered path, applying its
/// transforms in order. Every transform consumes its own consecutive group of [`ReferenceTransform::binding_count`]
/// ordinary (non-reference) inputs, whose types are validated against the referent produced by the preceding transform
/// before the transform computes the next one. A [`Read`](ReferenceAccessMode::Read) access derives each referent
/// through [`ReferenceTransform::read_type`], and every other access mode also validates that updates can be written
/// back through [`ReferenceTransform::output_type`].
///
/// # Parameters
///
///   - `input`: Referent type of the root reference.
///   - `transforms`: Transforms applied in order from the root.
///   - `bindings`: Types of the transforms' dynamic inputs, in path order.
///   - `mode`: Access mode of the operation applying the path.
pub fn infer_reference_view_type<Transform: ReferenceTransform>(
    input: &Transform::Referent,
    transforms: &[Transform],
    bindings: &[&Transform::Type],
    mode: ReferenceAccessMode,
) -> Result<Transform::Referent, TypeError> {
    let mut output = input.clone();
    let mut remaining = bindings;
    for transform in transforms {
        let count = transform.binding_count();
        if count > remaining.len() {
            return Err(TypeError::invalid(format!(
                "reference transform requires {} bindings but only {} remain",
                count,
                remaining.len(),
            )));
        }

        let (current, rest) = remaining.split_at(count);
        transform.validate_bindings(&output, current)?;
        output = match mode {
            ReferenceAccessMode::Read => transform.read_type(&output)?,
            _ => transform.output_type(&output)?,
        };
        remaining = rest;
    }

    if !remaining.is_empty() {
        return Err(TypeError::invalid(format!("reference transform path has {} extra bindings", remaining.len())));
    }

    Ok(output)
}

/// Moves a packed reference's [`BatchAxis`] through an ordered transform path, one transform at a time in the same
/// order in which [`infer_reference_view_type`] applies them. Dynamic bindings must be replicated. The resulting batch
/// axis belongs to the referent of the final view, which may have fewer dimensions than the root.
pub fn batch_reference_transforms<
    Transform: BatchableReferenceTransform<Type: From<ReferenceType<Transform::Referent>>>,
>(
    root_type: &Transform::Type,
    root_axis: BatchAxis,
    transforms: &[Transform],
    binding_axes: &[BatchAxis],
) -> Result<(Vec<Transform>, BatchAxis), BatchingError>
where
    for<'t> &'t ReferenceType<Transform::Referent>: TryFrom<&'t Transform::Type, Error = TypeError>,
{
    let mut current_type = root_type.clone();
    let mut current_axis = root_axis;
    let mut remaining = binding_axes;
    let mut batched = Vec::with_capacity(transforms.len());
    for transform in transforms {
        let count = transform.binding_count();
        if count > remaining.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference transform requires {} bindings but only {} remain",
                count,
                remaining.len(),
            ))
            .into());
        }

        let (current, rest) = remaining.split_at(count);
        if current.iter().any(|axis| !axis.is_replicated()) {
            return Err(BatchingError::UnsupportedOperation {
                message: "batching a reference transform with a mapped index input is not supported".to_string(),
            });
        }

        let (transform, axis) = transform.batch(&current_type, current_axis)?;
        let reference = <&ReferenceType<Transform::Referent>>::try_from(&current_type)?;
        current_type = ReferenceType::new(transform.output_type(reference.referent())?).into();
        current_axis = axis;
        remaining = rest;
        batched.push(transform);
    }

    if !remaining.is_empty() {
        return Err(ProgramError::MalformedProgram(format!(
            "reference transform path has {} extra bindings",
            remaining.len(),
        ))
        .into());
    }

    Ok((batched, current_axis))
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;

    use crate::arrays::{ArrayIrType, ArrayReferenceTransform, ArrayReferenceTransformIndex, ArrayType, DataType};
    use crate::programs::atoms::AtomId;
    use crate::programs::effects::ReferenceAccessMode;
    use crate::programs::regions::RegionId;

    use super::*;

    /// Paths with instruction-local program identity bindings.
    type TestPath = ReferenceTransformPath<ArrayReferenceTransform>;

    /// Creates a static reference type for a test root.
    fn reference_type(dimensions: impl Into<Vec<usize>>) -> ArrayIrType {
        ReferenceType::new(ArrayType::new_static(DataType::F32, dimensions.into())).into()
    }

    /// Creates a static index transform.
    pub(crate) fn index(axis: usize, index: usize) -> ArrayReferenceTransform {
        ArrayReferenceTransform::Index { axis, index: ArrayReferenceTransformIndex::Static(index) }
    }

    /// Creates a dynamic index transform.
    pub(crate) fn dynamic() -> ArrayReferenceTransform {
        ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }
    }

    #[test]
    fn test_no_reference_transform() {
        let root = ReferenceTransformPath::<NoReferenceTransform<ArrayType, ArrayIrType>>::root();
        assert_eq!(root.overlap(&root, &reference_type([2])), ReferenceViewOverlap::Same);
        assert_eq!(
            batch_reference_transforms::<NoReferenceTransform<ArrayType, ArrayIrType>>(
                &reference_type([2]),
                BatchAxis::new(0),
                &[],
                &[],
            ),
            Ok((Vec::new(), BatchAxis::new(0))),
        );
    }

    #[test]
    fn test_bound_reference_transform_transform() {
        let transform = index(0, 1);
        let path = TestPath::root().with_transform(transform.clone());
        assert_eq!(path.bound_transforms()[0].transform(), &transform);
    }

    #[test]
    fn test_bound_reference_transform_bindings() {
        let static_path = TestPath::root().with_transform(index(0, 1));
        assert_eq!(static_path.bound_transforms()[0].bindings(), &[]);

        // The binding identifies the value supplying the dynamic index; it is not the index's runtime value.
        let path =
            TestPath::root().with_bound_transform(dynamic(), vec![ValueId::new(RegionId::new(0), AtomId::new(3))]);
        assert_eq!(path.bound_transforms()[0].bindings(), &[ValueId::new(RegionId::new(0), AtomId::new(3))]);
    }

    #[test]
    fn test_reference_transform_path_root() {
        let root = TestPath::root();
        assert!(root.is_root());
        assert_eq!(root.bound_transforms(), &[]);
        assert_eq!(root.transforms().count(), 0);
        assert_eq!(root, TestPath::default());

        assert_eq!(format!("{root:?}"), "ReferenceTransformPath { bound_transforms: [] }");
    }

    #[test]
    fn test_reference_transform_path_from_transforms() {
        let first = ValueId::new(RegionId::new(0), AtomId::new(2));
        let second = ValueId::new(RegionId::new(0), AtomId::new(3));
        let transforms = [index(0, 1), dynamic(), dynamic()];
        assert_eq!(
            TestPath::from_transforms(&transforms, &[first, second]),
            Ok(TestPath::root()
                .with_transform(index(0, 1))
                .with_bound_transform(dynamic(), vec![first])
                .with_bound_transform(dynamic(), vec![second])),
        );
        assert_eq!(TestPath::from_transforms(&[], &[]), Ok(TestPath::root()));
        assert_eq!(
            TestPath::from_transforms(&transforms, &[first]),
            Err(ProgramError::MalformedProgram(
                "reference transform requires 1 bindings but only 0 remain".to_string()
            )),
        );
        assert_eq!(
            TestPath::from_transforms(&[], &[first]),
            Err(ProgramError::MalformedProgram("reference transform path has 1 extra bindings".to_string())),
        );
    }

    #[test]
    fn test_reference_transform_path_is_root() {
        assert!(TestPath::root().is_root());
        assert!(!TestPath::root().with_transform(index(0, 1)).is_root());
    }

    #[test]
    fn test_reference_transform_path_bound_transforms() {
        let path = TestPath::root().with_transform(index(0, 1)).with_transform(index(0, 2));
        assert_eq!(
            path.bound_transforms(),
            &[
                BoundReferenceTransform { transform: index(0, 1), bindings: Vec::new() },
                BoundReferenceTransform { transform: index(0, 2), bindings: Vec::new() },
            ],
        );
    }

    #[test]
    fn test_reference_transform_path_transforms() {
        let path = TestPath::root().with_transform(index(0, 1)).with_transform(index(0, 2));
        assert_eq!(path.transforms().collect::<Vec<_>>(), vec![&index(0, 1), &index(0, 2)]);
        assert_eq!(path.transforms().rev().collect::<Vec<_>>(), vec![&index(0, 2), &index(0, 1)]);
    }

    #[test]
    fn test_reference_transform_path_with_bound_transform() {
        let row = TestPath::root().with_transform(index(0, 1));
        let symbolic = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic };
        let bound = row.with_bound_transform(symbolic.clone(), vec![ValueId::new(RegionId::new(0), AtomId::new(3))]);
        assert_eq!(bound.transforms().collect::<Vec<_>>(), vec![&index(0, 1), &symbolic]);
        assert_eq!(bound.bound_transforms()[1].bindings(), &[ValueId::new(RegionId::new(0), AtomId::new(3))]);
        assert_eq!(row.transforms().collect::<Vec<_>>(), vec![&index(0, 1)]);

        // Equal transforms can select different indices when their source bindings differ.
        assert_eq!(
            bound,
            row.with_bound_transform(symbolic.clone(), vec![ValueId::new(RegionId::new(0), AtomId::new(3))])
        );
        assert_ne!(
            bound,
            row.with_bound_transform(symbolic.clone(), vec![ValueId::new(RegionId::new(0), AtomId::new(4))])
        );
        assert_ne!(bound, row.with_bound_transform(symbolic, vec![ValueId::new(RegionId::new(1), AtomId::new(0))]));

        // Paths used as map keys distinguish bindings as well as transforms.
        let paths = HashMap::from([(bound.clone(), "bound")]);
        assert_eq!(paths.get(&bound), Some(&"bound"));
        assert_eq!(paths.get(&row), None);
    }

    #[test]
    fn test_reference_transform_path_with_transform() {
        let root = TestPath::root();
        let row = root.with_transform(index(0, 1));
        let element = row.with_transform(index(0, 2));

        // Appending preserves root-to-value order without modifying either source path.
        assert!(root.is_root());
        assert_eq!(row.transforms().collect::<Vec<_>>(), vec![&index(0, 1)]);
        assert_eq!(element.transforms().collect::<Vec<_>>(), vec![&index(0, 1), &index(0, 2)]);
        assert_eq!(row, TestPath::root().with_transform(index(0, 1)));
        assert_ne!(row, element);
        assert_ne!(row, TestPath::root().with_transform(index(1, 1)));
        assert_eq!(
            format!("{row:?}"),
            "ReferenceTransformPath { bound_transforms: [BoundReferenceTransform { transform: Index { axis: 0, index: \
             Static(1) }, bindings: [] }] }",
        );
    }

    #[test]
    fn test_reference_transform_path_overlap() {
        // The path query delegates to the family's rule against the caller-supplied root type: two different rows are
        // disjoint, a row is the same as itself, and the complete root may overlap with any row but is the same as
        // itself.
        let root = reference_type([2, 3]);
        let row_0 = TestPath::root().with_transform(index(0, 0));
        let row_1 = TestPath::root().with_transform(index(0, 1));
        assert_eq!(row_0.overlap(&row_1, &root), ReferenceViewOverlap::Disjoint);
        assert_eq!(row_0.overlap(&TestPath::root().with_transform(index(0, 0)), &root), ReferenceViewOverlap::Same);
        assert_eq!(TestPath::root().overlap(&row_0, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(TestPath::root().overlap(&TestPath::root(), &root), ReferenceViewOverlap::Same);

        // Symbolic transforms agree when their input bindings agree. Different bindings may still select the same row.
        let symbolic =
            TestPath::root().with_bound_transform(dynamic(), vec![ValueId::new(RegionId::new(0), AtomId::new(0))]);
        let other =
            TestPath::root().with_bound_transform(dynamic(), vec![ValueId::new(RegionId::new(1), AtomId::new(0))]);
        assert_eq!(symbolic.overlap(&symbolic, &root), ReferenceViewOverlap::Same);
        assert_eq!(symbolic.overlap(&TestPath::root(), &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(symbolic.overlap(&row_1, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(symbolic.overlap(&other, &root), ReferenceViewOverlap::MayOverlap);
    }

    #[test]
    fn test_infer_reference_view_type() {
        let input = ArrayType::new_static(DataType::F32, [3, 4]);
        let transforms = [index(0, 1), dynamic()];
        let binding = ArrayIrType::Array(ArrayType::scalar(DataType::I32));
        assert_eq!(
            infer_reference_view_type(&input, &transforms, &[&binding], ReferenceAccessMode::Read),
            Ok(ArrayType::scalar(DataType::F32))
        );
        assert_eq!(
            infer_reference_view_type(&input, &transforms, &[], ReferenceAccessMode::Read),
            Err(TypeError::invalid("reference transform requires 1 bindings but only 0 remain")),
        );
        assert_eq!(
            infer_reference_view_type::<ArrayReferenceTransform>(&input, &[], &[&binding], ReferenceAccessMode::Read),
            Err(TypeError::invalid("reference transform path has 1 extra bindings")),
        );
        assert_eq!(
            infer_reference_view_type(&input, &[index(0, 1), index(1, 0)], &[], ReferenceAccessMode::Read),
            Err(TypeError::invalid("reference index axis 1 is out of bounds for rank 1")),
        );
    }

    #[test]
    fn test_infer_reference_view_type_validates_write_back_only_for_mutating_accesses() {
        /// Transform that selects its input unchanged but whose selection can never be written back.
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        struct ReadOnlyTransform;

        impl std::fmt::Display for ReadOnlyTransform {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("read_only")
            }
        }

        impl ReferenceTransform for ReadOnlyTransform {
            type Type = ArrayIrType;
            type Referent = ArrayType;

            fn binding_count(&self) -> usize {
                0
            }

            fn validate_bindings(&self, _input: &ArrayType, _bindings: &[&ArrayIrType]) -> Result<(), TypeError> {
                Ok(())
            }

            fn output_type(&self, _input: &ArrayType) -> Result<ArrayType, TypeError> {
                Err(TypeError::invalid("`read_only` transforms cannot be written back"))
            }

            fn read_type(&self, input: &ArrayType) -> Result<ArrayType, TypeError> {
                Ok(input.clone())
            }

            fn overlap(
                _type: &ArrayIrType,
                _lhs: &[BoundReferenceTransform<Self>],
                _rhs: &[BoundReferenceTransform<Self>],
            ) -> ReferenceViewOverlap {
                ReferenceViewOverlap::MayOverlap
            }
        }

        // Only read-only accesses skip the write-back validation of `output_type`.
        let input = ArrayType::new_static(DataType::F32, [3]);
        assert_eq!(
            infer_reference_view_type(&input, &[ReadOnlyTransform], &[], ReferenceAccessMode::Read),
            Ok(input.clone())
        );
        for mode in [
            ReferenceAccessMode::Write,
            ReferenceAccessMode::ReadWrite,
            ReferenceAccessMode::Accumulate,
            ReferenceAccessMode::AtomicAccumulate,
        ] {
            assert_eq!(
                infer_reference_view_type(&input, &[ReadOnlyTransform], &[], mode),
                Err(TypeError::invalid("`read_only` transforms cannot be written back")),
            );
        }
    }

    #[test]
    fn test_batch_reference_transforms() {
        // Removing the first axis moves the packed batch axis before the dynamic transform is adjusted.
        let transforms = [index(0, 1), dynamic()];
        assert_eq!(
            batch_reference_transforms(
                &reference_type([3, 5, 4]),
                BatchAxis::new(1),
                &transforms,
                &[BatchAxis::replicated()]
            ),
            Ok((
                vec![
                    index(0, 1),
                    ArrayReferenceTransform::Index { axis: 1, index: ArrayReferenceTransformIndex::Dynamic }
                ],
                BatchAxis::new(0),
            )),
        );
        assert_eq!(
            batch_reference_transforms(
                &reference_type([3, 5, 4]),
                BatchAxis::new(1),
                &transforms,
                &[BatchAxis::new(0)]
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "batching a reference transform with a mapped index input is not supported".to_string(),
            }),
        );
        assert_eq!(
            batch_reference_transforms(&reference_type([3, 5, 4]), BatchAxis::new(1), &transforms, &[]),
            Err(ProgramError::MalformedProgram(
                "reference transform requires 1 bindings but only 0 remain".to_string()
            )
            .into()),
        );
    }
}
