use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use thiserror::Error;

use ryft_macros::Parameter;

use crate::arrays::addressing::ArraySliceAxis;
use crate::arrays::ir::{ArrayIrContext, ArrayIrValue};
use crate::arrays::operations::ArrayIrOperation;
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::dimensions::{Dimension, Shape, StaticShape};
use crate::arrays::types::ir::ArrayIrType;
use crate::batching::{BatchAxis, BatchingError};
use crate::contexts::Context;
use crate::macros::check_count;
use crate::operations::{
    Add, AddOperation, DynamicSliceOperation, DynamicUpdateSliceOperation, Reshape, ReshapeOperation, Slice,
    SliceOperation, UpdateSlice, UpdateSliceOperation,
};
use crate::parameters::Parameter;
use crate::programs::{
    BatchableReferenceTransform, BoundReferenceTransform, Concretizable, NoReferenceTransformBinding, Operation,
    OperationProjection, ProgramError, ReadyOrPendingReferenceGuard, Reference, ReferenceAccessDescriptor,
    ReferenceAccessOperation, ReferenceAccumulationPolicy, ReferenceDischargePolicy, ReferenceDischargeableType,
    ReferenceError, ReferenceId, ReferenceTransform, ReferenceTransformPath, ReferenceType, ReferenceView,
    ReferenceViewAnalysis, ReferenceViewOverlap, Type, TypeError, TypeIdentityRenaming, Typed, Value, ValueId,
};

/// Error produced by an invalid eager array-reference view operation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum ArrayReferenceViewError {
    #[error("cannot freeze a reference view; freeze the root reference instead")]
    CannotFreezeView,

    #[error("backend storage transactions require a root handle that uses the allocation's stored type identities")]
    NotStorageRoot,

    #[error("eager reference handles carry only static transforms; dynamic indices are resolved by each access")]
    DynamicTransformIndex,
}

/// Eager handle to a mutable array, which pairs one shared root allocation with a handle-local
/// [`ArrayReferenceTransformPath`] selecting the elements that the handle views. A root handle (i.e., one
/// created by [`ArrayReference::new`]) views the complete allocation, and [`ArrayReference::with_transform`]
/// and [`ArrayReference::with_transforms`] derive views of parts of it that share the same allocation.
///
/// The path contains only static transforms, so its binding type is [`NoReferenceTransformBinding`]. Reads extract the
/// elements that the path addresses, whereas mutations reconstruct the root through the same transforms in reverse
/// order and preserve the values outside the view. [`ArrayReferenceDischarge`] uses the same traversal to express
/// these accesses as immutable array operations in a context.
///
/// The underlying [`Reference`] owns the allocation's identity, lifetime, alias validation, and synchronization. This
/// handle adds the array-specific transform path and the referent type that it selects, without maintaining allocation
/// state of its own.
///
/// Equality and hashing identify the mutable location and the structural view, not the handle-local namespace of
/// type identities. Renaming type identities therefore preserves equality with the original handle when its view is
/// unchanged.
///
/// # Example
///
/// A view shares the allocation of its root, so writing through the view updates the root:
///
/// ```rust
/// # use ryft_core::{Array, ArrayReference, ArrayReferenceTransform, ArraySliceAxis, ProgramError};
/// # fn main() -> Result<(), ProgramError> {
/// let root = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0, 4.0])?);
/// let view = root.with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] })?;
/// view.write(Array::vector(vec![20.0f32, 30.0])?)?;
/// assert_eq!(view.id(), root.id());
/// assert_eq!(view.read()?, Array::vector(vec![20.0f32, 30.0])?);
/// assert_eq!(root.read()?, Array::vector(vec![1.0f32, 20.0, 30.0, 4.0])?);
/// # Ok(())
/// # }
/// ```
#[derive(Parameter)]
pub struct ArrayReference<A: Value<Type = ArrayType>> {
    /// Handle to the shared root allocation.
    root: Reference<A>,

    /// Ordered mapping from the shared root to this handle's referent. Eager handles only ever carry static
    /// transforms, so no dynamic index is ever bound on this path.
    path: ArrayReferenceTransformPath<NoReferenceTransformBinding>,

    /// Exact handle type derived once from the root type and path, so that repeated [`Typed`] type queries borrow this
    /// cached type instead of re-deriving it from the complete transform path.
    r#type: ReferenceType<ArrayType>,
}

impl<A: Value<Type = ArrayType>> ArrayReference<A> {
    /// Creates a new root [`ArrayReference`] to a fresh allocation initialized with `value`.
    /// The returned handle views the complete allocation (i.e., its transform path is empty).
    #[inline]
    pub fn new(value: A) -> Self {
        // `A::Type` is exactly `ArrayType`, whose type family cannot denote a reference, so the generic
        // nested-referent rejection is unreachable for this specialized constructor.
        let root = Reference::new(value).unwrap();
        let r#type = root.r#type().into_owned();
        Self { root, path: ArrayReferenceTransformPath::root(), r#type }
    }

    /// Returns the process-local identity of the allocation (i.e., the [`ReferenceId`]) that this handle shares with
    /// every clone and view of the same root.
    #[inline]
    pub fn id(&self) -> ReferenceId {
        self.root.id()
    }

    /// Returns the [`ArrayReferenceTransformPath`]that select this handle's elements from its shared root allocation.
    /// Note that the returned path is empty for root handles.
    #[inline]
    pub fn path(&self) -> &ArrayReferenceTransformPath<NoReferenceTransformBinding> {
        &self.path
    }

    /// Returns whether this handle is a _storage root_ (i.e., a root handle that also uses the allocation's stored type
    /// identities, so that its value is exactly the stored value). Backend storage transactions require a storage root,
    /// because they access the stored value directly, without applying a transform path or converting between the type
    /// identities of the handle and those used in storage.
    ///
    /// Backends use this predicate to decide whether a reference argument can enter a compiled call as-is, and must
    /// reject (or first materialize) views and aliases with renamed type identities, since [`Self::lock_storage`]
    /// refuses them. Ordinary value access through [`Self::read`] and [`Self::write`] has no such restriction.
    #[inline]
    pub fn is_storage_root(&self) -> bool {
        self.path.is_root() && self.root.uses_storage_type_identities()
    }

    /// Locks the storage of this handle's allocation for one backend-owned state transaction, which the returned guard
    /// holds until it is dropped. Backends use it to read the stored value as a compiled-call argument and to publish
    /// the submitted replacement of that value, following the protocol of [`Reference::lock`]:
    ///
    ///   - The function returns as soon as the allocation's mutex is acquired and does not wait for pending values or
    ///     active read leases, so the guard may observe a `Pending` value that the transaction must order after.
    ///   - The mutex is not reentrant. Locking the same allocation again while the guard is alive, whether through
    ///     this function on another alias or through a value access function such as [`Self::read`], deadlocks or
    ///     panics.
    ///   - A transaction that locks multiple allocations must lock them in ascending [`ReferenceId`] order (see
    ///     [`Self::id`]) and keep every guard, or the replacement typestate derived from it, alive until all of its
    ///     replacements have been validated and committed.
    ///
    /// # Errors
    ///
    /// Returns [`ArrayReferenceViewError::NotStorageRoot`] if this handle is not a storage root (as checked by
    /// [`Self::is_storage_root`]), and forwards the [`ReferenceError`] of an allocation that cannot be locked
    /// (e.g., because it is frozen or poisoned).
    pub fn lock_storage(&self) -> Result<ReadyOrPendingReferenceGuard<'_, A>, ProgramError> {
        if !self.is_storage_root() {
            return Err(ProgramError::custom(ArrayReferenceViewError::NotStorageRoot));
        }
        self.root.lock().map_err(ProgramError::custom)
    }

    /// Returns a view that shares this handle's allocation and appends `transform` to its path. Deriving a view
    /// reads no state, so the allocation is only validated when the returned handle is accessed.
    ///
    /// # Errors
    ///
    /// Returns [`ArrayReferenceViewError::DynamicTransformIndex`] if `transform` indexes dynamically, because an eager
    /// handle's path carries only static transforms ([`Self::with_transforms`] resolves dynamic indices instead), and
    /// a [`TypeError`] if `transform` does not apply to this handle's referent type.
    pub fn with_transform(&self, transform: ArrayReferenceTransform) -> Result<Self, ProgramError> {
        if transform.binding_count() != 0 {
            return Err(ProgramError::custom(ArrayReferenceViewError::DynamicTransformIndex));
        }

        // The cached handle type already reflects every earlier transform, so composition validates and derives
        // incrementally instead of re-folding the complete chain from the root type. Derivation is purely structural:
        // holder liveness is checked only when the resulting handle accesses state.
        let referent = transform.output_type(self.r#type.referent())?;
        let path = self.path.clone().with_transform(transform);
        Ok(Self { root: self.root.clone(), path, r#type: ReferenceType::new(referent) })
    }

    /// Returns a view that shares this handle's allocation and appends the ordered `transforms` of one access to its
    /// path, resolving every dynamic index into a static one. A negative dynamic index counts from the end of its axis
    /// once and is then clamped to that axis's valid range. Deriving a view reads no state, so the allocation is only
    /// validated when the returned handle is accessed.
    ///
    /// # Parameters
    ///
    ///   - `transforms`: Transforms to append, in order from this handle's referent outward.
    ///   - `bindings`: Dynamic indices of the transforms, in order, with one scalar integer array per dynamic index.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if the number of `bindings` does not match the dynamic indices of `transforms`, if a
    /// transform or binding does not apply to the referent that it receives, or if a dynamic index addresses an empty
    /// axis, and forwards the error of a binding that cannot be concretized.
    pub fn with_transforms(
        &self,
        transforms: &[ArrayReferenceTransform],
        bindings: &[ArrayIrValue<A>],
    ) -> Result<Self, ProgramError>
    where
        A: Concretizable<i128>,
    {
        // Validate and resolve every transform in one pass against the running referent, then append the resolved
        // transforms to one copy of this handle's path, rather than copying the growing path once per transform.
        let mut path = self.path.clone();
        let mut referent = self.r#type.referent().clone();
        let mut remaining = bindings;
        for transform in transforms {
            let count = transform.binding_count();
            if count > remaining.len() {
                return Err(TypeError::invalid(format!(
                    "reference transform requires {} bindings but only {} remain",
                    count,
                    remaining.len(),
                ))
                .into());
            }

            let (current, rest) = remaining.split_at(count);
            remaining = rest;
            let binding_types = current.iter().map(Typed::r#type).collect::<Vec<_>>();
            transform.validate_bindings(&referent, &binding_types.iter().map(AsRef::as_ref).collect::<Vec<_>>())?;
            let transform = match (transform, current) {
                (ArrayReferenceTransform::Index { axis, index: ArrayReferenceTransformIndex::Dynamic }, [index]) => {
                    // Binding validation has checked the scalar integer index; the transform's own validation checks
                    // the shape and axis.
                    transform.read_type(&referent)?;
                    let extent = referent.static_shape().unwrap()[*axis] as i128;
                    if extent == 0 {
                        return Err(TypeError::invalid("cannot dynamically index an empty reference axis").into());
                    }
                    let ArrayIrValue::Array(index) = index else { unreachable!() };
                    let index = index.concretize()?;
                    let index = if index < 0 { index + extent } else { index };
                    let index = index.clamp(0, extent - 1) as usize;
                    ArrayReferenceTransform::Index { axis: *axis, index: ArrayReferenceTransformIndex::Static(index) }
                }
                _ => transform.clone(),
            };
            referent = transform.output_type(&referent)?;
            path.push_transform(transform);
        }

        if !remaining.is_empty() {
            return Err(
                TypeError::invalid(format!("reference transform path has {} extra bindings", remaining.len())).into()
            );
        }

        Ok(Self { root: self.root.clone(), path, r#type: ReferenceType::new(referent) })
    }

    /// Returns an immutable snapshot of the elements that this handle views.
    ///
    /// # Errors
    ///
    /// Forwards the [`ReferenceError`] of an allocation that cannot be read (e.g., because it is frozen or poisoned).
    #[inline]
    pub fn read(&self) -> Result<A, ProgramError>
    where
        A: Reshape + Slice,
    {
        self.path.apply(self.root.read().map_err(ProgramError::custom)?)
    }

    /// Replaces the elements that this handle views with `replacement` and returns a snapshot of their previous
    /// values. Values outside the view are preserved.
    ///
    /// # Errors
    ///
    /// Forwards the [`ReferenceError`] of an allocation that cannot be updated (e.g., because it is frozen or
    /// poisoned), and returns [`ReferenceError::ReferentTypeMismatch`] if the type of `replacement` differs from this
    /// handle's referent type. Allocation errors take precedence over the type mismatch, for views and root handles
    /// alike.
    pub fn swap(&self, replacement: A) -> Result<A, ProgramError>
    where
        A: Reshape + Slice + UpdateSlice,
    {
        if self.path.is_root() {
            return self.root.swap(replacement).map_err(ProgramError::custom);
        }

        // Validation remains inside the holder transaction so frozen, poisoned, and leased-state diagnostics retain
        // precedence over replacement-type errors, matching the root write and swap paths.
        self.root.update(|current| {
            self.validate_view_referent_type(&replacement)?;
            let (previous, updated) =
                self.path.swap_in(&EagerTransformCarrier(PhantomData), current.clone(), replacement)?;
            Ok((updated, previous))
        })
    }

    /// Replaces the elements that this handle views with `replacement`, like [`Self::swap`], but without returning
    /// a snapshot of their previous values.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::swap`], with the same precedence.
    pub fn write(&self, replacement: A) -> Result<(), ProgramError>
    where
        A: Reshape + Slice + UpdateSlice,
    {
        if self.path.is_root() {
            return self.root.write(replacement).map_err(ProgramError::custom);
        }

        // Validation remains inside the holder transaction so frozen, poisoned, and leased-state diagnostics retain
        // precedence over replacement-type errors, matching the root write and swap paths.
        self.root.update(|current| {
            self.validate_view_referent_type(&replacement)?;
            self.path
                .write_in(&EagerTransformCarrier(PhantomData), current.clone(), replacement)
                .map(|updated| (updated, ()))
        })
    }

    /// Adds `update` elementwise into the elements that this handle views, preserving the values outside the view.
    ///
    /// # Errors
    ///
    /// Forwards the [`ReferenceError`] of an allocation that cannot be updated and the error of an addition whose
    /// inputs are incompatible, and returns [`ReferenceError::ReferentTypeMismatch`] if the sum does not have this
    /// handle's referent type (e.g., because `update` promotes or broadcasts the viewed elements).
    pub fn add_update(&self, update: &A) -> Result<(), ProgramError>
    where
        A: Add + Reshape + Slice + UpdateSlice,
    {
        if self.path.is_root() {
            return self.root.update(|current| current.add(update).map(|updated| (updated, ())));
        }

        self.root.update(|current| {
            let carrier = EagerTransformCarrier(PhantomData);
            let intermediates = self.path.intermediates_in(&carrier, current.clone())?;
            let updated_view = intermediates.last().unwrap().add(update)?;
            self.validate_view_referent_type(&updated_view)?;
            self.path
                .reconstruct_in(&carrier, &intermediates[..self.path.bound_transforms().len()], updated_view)
                .map(|updated| (updated, ()))
        })
    }

    /// Freezes the allocation of a root handle and returns its final value, invalidating every handle that shares the
    /// allocation.
    ///
    /// This function borrows the handle, whereas the value-level [`ReferenceFreeze`](crate::ReferenceFreeze)
    /// capability above it consumes one. The asymmetry is mechanical rather than semantic: the composite implementation
    /// reaches this handle through a projection of its owned value, which yields a borrow, and the capability already
    /// enforces the linearity one layer up.
    ///
    /// # Errors
    ///
    /// Returns [`ArrayReferenceViewError::CannotFreezeView`] without changing the shared state if this handle is a
    /// view, and forwards the [`ReferenceError`] of an allocation that cannot be frozen (e.g., because it is already
    /// frozen).
    pub fn freeze(&self) -> Result<A, ProgramError> {
        if !self.path.is_root() {
            return Err(ProgramError::custom(ArrayReferenceViewError::CannotFreezeView));
        }
        self.root.freeze().map_err(ProgramError::custom)
    }

    /// Returns a handle to the same allocation and view whose handle-local type identities are renamed by `renaming`,
    /// which applies in both directions. The renamed handle compares equal to this one.
    pub(crate) fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayType as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        let root = self.root.rename_type_identities(renaming)?;
        let referent = self.path.output_type(root.r#type().referent())?;
        Ok(Self { root, path: self.path.clone(), r#type: ReferenceType::new(referent) })
    }

    /// Checks that `value` has exactly this handle's referent type (i.e., the type of the elements that the handle
    /// views). Mutations through views must call this function themselves. A root-handle mutation replaces the complete
    /// stored value, which the underlying [`Reference`] already checks against its own referent type. A view mutation
    /// instead writes `value` into the root through [`UpdateSlice`], which only requires `value` to fit inside the part
    /// of the root that the view selects. The reconstructed root then passes the reference's check even when `value` is
    /// smaller than the view, so without this function, such a value would silently update only part of the view.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError::ReferentTypeMismatch`] if the type of `value` differs from this handle's
    /// referent type.
    fn validate_view_referent_type(&self, value: &A) -> Result<(), ProgramError> {
        let actual = value.r#type();
        if actual.as_ref() == self.r#type.referent() {
            return Ok(());
        }
        Err(ProgramError::custom(ReferenceError::ReferentTypeMismatch {
            expected: self.r#type.referent().to_string(),
            actual: actual.to_string(),
        }))
    }
}

impl<A: Value<Type = ArrayType>> Clone for ArrayReference<A> {
    #[inline]
    fn clone(&self) -> Self {
        Self { root: self.root.clone(), path: self.path.clone(), r#type: self.r#type.clone() }
    }
}

impl<A: Value<Type = ArrayType>> Debug for ArrayReference<A> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("ArrayReference").field("id", &self.id()).field("path", &self.path).finish()
    }
}

impl<A: Value<Type = ArrayType>> Display for ArrayReference<A> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.r#type(), formatter)
    }
}

impl<A: Value<Type = ArrayType>> PartialEq for ArrayReference<A> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root && self.path == other.path
    }
}

impl<A: Value<Type = ArrayType>> Eq for ArrayReference<A> {}

impl<A: Value<Type = ArrayType>> Hash for ArrayReference<A> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.root.hash(state);
        self.path.hash(state);
    }
}

impl<A: Value<Type = ArrayType>> Typed for ArrayReference<A> {
    type Type = ReferenceType<ArrayType>;

    #[inline]
    fn r#type(&self) -> Cow<'_, Self::Type> {
        Cow::Borrowed(&self.r#type)
    }
}

/// Immutable index mapping between a shared array-reference root and one view of it. This is the array specialization
/// of the generic [`ReferenceTransformPath`], whose transforms are [`ArrayReferenceTransform`]s.
///
/// The mapping stores validated transforms in root-to-handle order. The empty mapping (i.e., [`root`](Self::root)) is
/// the identity view and denotes the complete root. Each additional transform is applied to the preceding view, so
/// indexing or slicing an [`ArrayReference`] view composes onto the same shared root rather than creating another
/// mutable resource.
///
/// Traversals of a path call the input of each transform its _parent_ and the output its _child_. The _intermediates_
/// of a path are the root followed by each child, and the last of them, which is the value that the path selects, is
/// its _leaf_. Every intermediate before the leaf is a _strict parent_, and reconstruction needs each of them to
/// preserve the elements outside the view.
///
/// This type is structural metadata only: it owns neither the referenced array nor its resource identity, liveness,
/// or synchronization state. [`ArrayReference`] pairs it with a handle to the shared reference allocation. In staged
/// programs, [`ArrayReferenceAnalysis`] records one path per reference access, keyed by its instruction and root input
/// index. The path determines the viewed referent and addressed indices; mutations reconstruct the root by applying the
/// inverse update of each transform in reverse order. Overlapping views may address the same root indices and observe
/// one another's ordered mutations, while equality and hashing distinguish different transform sequences.
///
/// `Binding` supplies dynamic indices: [`ValueId`] identifies program values, the uninhabited
/// [`NoReferenceTransformBinding`] restricts eager handles to static transforms, and `C::Value` binds discharge indices
/// directly to context values. Supported index transforms are described by [`ArrayReferenceTransform`]. Attached-region
/// and external runtime boundaries pass complete root handles, and each access inside the receiving scope carries its
/// own path. For example, each access in a scan body selects one item of a stacked reference through a dynamic index
/// bound to the body's explicit index input.
pub type ArrayReferenceTransformPath<Binding = ValueId> = ReferenceTransformPath<ArrayReferenceTransform, Binding>;

/// Array specialization of [`ReferenceViewAnalysis`], associating each reference access with its ordered
/// [`ArrayReferenceTransformPath`]. Allocation roots and lifetimes come from the shared structural analysis.
/// Dynamic bindings name ordinary values in the access instruction's own region. The generic
/// [`references`](crate::programs::references) module owns allocation identity, lifetime and alias validation,
/// transform path storage, and analysis while this specialization supplies array shapes and indexing semantics.
pub type ArrayReferenceAnalysis = ReferenceViewAnalysis<ArrayReferenceTransform>;

impl ArrayReferenceTransformPath {
    /// Returns the part of a root of type `root_type` that this path selects, as one [`ArraySliceAxis`] per root axis.
    ///
    /// Static indices and unit-stride slices always select an axis-aligned box of the root. This function describes
    /// that box in root coordinates and at the root's rank: a sliced axis keeps its narrowed range, and an indexed
    /// axis, which the path removes from its result, becomes a size-one range. The ranges cover exactly the elements
    /// that the path selects, so consumers such as kernel validation can compute the elements or bytes that an access
    /// touches through [`ArrayAddressing`](crate::ArrayAddressing).
    ///
    /// Returns [`None`] if the path contains a dynamic index, whose position is only known at the access, if
    /// `root_type` does not have a static shape, or if the path does not fold against `root_type` (e.g., because
    /// an index is out of bounds).
    ///
    /// # Example
    ///
    /// The path `[slice(axes=[1:4, 2:5]), index(axis=0, index=1)]` first selects rows `1..4` and columns `2..5` of an
    /// `i32[4, 5]` root and then row `1` of that slice, which is row `2` of the root. In root coordinates, it
    /// therefore selects `[2:3, 2:5]`:
    ///
    /// ```rust
    /// # use ryft_core::{ArrayReferenceTransform, ArrayReferenceTransformIndex, ArrayReferenceTransformPath};
    /// # use ryft_core::{ArraySliceAxis, ArrayType, DataType};
    /// let path = ArrayReferenceTransformPath::root()
    ///     .with_transform(ArrayReferenceTransform::Slice {
    ///         axes: vec![ArraySliceAxis::new(1, 3, 1), ArraySliceAxis::new(2, 3, 1)],
    ///     })
    ///     .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) });
    /// assert_eq!(
    ///     path.root_slice_axes(&ArrayType::new_static(DataType::I32, vec![4, 5])),
    ///     Some(vec![ArraySliceAxis::new(2, 1, 1), ArraySliceAxis::new(2, 3, 1)]),
    /// );
    /// ```
    pub fn root_slice_axes(&self, root_type: &ArrayType) -> Option<Vec<ArraySliceAxis>> {
        RootIndexSelection::fold(&root_type.static_shape()?, self.bound_transforms())?
            .into_iter()
            .map(|selection| match selection {
                RootIndexSelection::Static { start, limit } => Some(ArraySliceAxis::new(start, limit - start, 1)),
                RootIndexSelection::Dynamic { .. } => None,
            })
            .collect()
    }
}

impl<Binding> ArrayReferenceTransformPath<Binding> {
    /// Returns the type of the value that this path selects from a root of type `root_type`, by applying
    /// the [`ArrayReferenceTransform::output_type`] of each transform in order. For example, the path
    /// `[slice(axes=[1:4, 2:5]), index(axis=0, index=1)]` turns an `i32[4, 5]` root into `i32[3, 3]` and
    /// then into `i32[3]`.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if a transform does not apply to the type that it receives (e.g., because an index
    /// is out of bounds), or if writing through the path would not reconstruct the root type exactly.
    #[inline]
    pub fn output_type(&self, root_type: &ArrayType) -> Result<ArrayType, TypeError> {
        self.transforms().try_fold(root_type.clone(), |r#type, transform| transform.output_type(&r#type))
    }

    /// Returns every value along this path: `root` first, followed by the child that each transform selects from the
    /// preceding value. The result has one more entry than the path has transforms, so an empty path returns only
    /// `root`. For example, for the path `[slice(axes=[1:4, 2:5]), index(axis=0, index=1)]` and an `i32[4, 5]` root,
    /// it returns:
    ///
    /// ```text
    ///     [root: i32[4, 5], root[1:4, 2:5]: i32[3, 3], root[2, 2:5]: i32[3]]
    /// ```
    ///
    /// The last entry is the selected value, and the others are the strict parents that
    /// [`reconstruct_in`](Self::reconstruct_in) needs to write a new selected value back. `carrier`
    /// performs the array operations and resolves dynamic indices from the bindings of each transform.
    #[inline]
    fn intermediates_in<C: TransformReadCarrier<Binding = Binding>>(
        &self,
        carrier: &C,
        root: C::Value,
    ) -> Result<Vec<C::Value>, ProgramError> {
        Self::intermediates_through(carrier, root, self.bound_transforms())
    }

    /// Returns the root with the value that this path selects replaced by `replacement`, keeping every element outside
    /// the view unchanged. The traversal runs from the leaf back to the root. Specifically, the last transform writes
    /// `replacement` into its parent, the transform before it writes that updated parent into its own parent, and so
    /// on, until the root is rebuilt:
    ///
    /// ```text
    ///     root  ──slice──▶  parent  ──index──▶  leaf           (intermediates_in)
    ///     root' ◀──update── parent' ◀──update── replacement    (reconstruct_in)
    /// ```
    ///
    /// # Parameters
    ///
    ///   - `carrier`: Array operations used to write each child back into its parent.
    ///   - `intermediates`: Strict parents of the leaf in root-to-leaf order, with one entry per transform. This is
    ///     the result of [`intermediates_in`](Self::intermediates_in) without its last entry.
    ///   - `replacement`: New selected value.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] if `intermediates` does not have one entry per transform,
    /// and forwards the errors of `carrier`.
    fn reconstruct_in<C: TransformWriteCarrier<Binding = Binding>>(
        &self,
        carrier: &C,
        intermediates: &[C::Value],
        replacement: C::Value,
    ) -> Result<C::Value, ProgramError> {
        let bound_transforms = self.bound_transforms();
        if intermediates.len() != bound_transforms.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference transform path reconstruction requires {} parent snapshots but received {}",
                bound_transforms.len(),
                intermediates.len(),
            )));
        }
        let mut reconstructed = replacement;
        for (bound_transform, intermediate) in bound_transforms.iter().zip(intermediates).rev() {
            let bindings = bound_transform.bindings();
            reconstructed = bound_transform.transform().replace_in(carrier, intermediate, &reconstructed, bindings)?;
        }
        Ok(reconstructed)
    }

    /// Replaces the value that this path selects from `root` with `replacement` and returns `(previous, updated)`,
    /// where `previous` is the selected value before the swap and `updated` is the rebuilt root (see
    /// [`reconstruct_in`](Self::reconstruct_in)). Both results come from one traversal, which the eager
    /// [`ArrayReference::swap`] and the discharge-time [`ReferenceDischargePolicy::swap`] share.
    fn swap_in<C: TransformWriteCarrier<Value: Clone, Binding = Binding>>(
        &self,
        carrier: &C,
        root: C::Value,
        replacement: C::Value,
    ) -> Result<(C::Value, C::Value), ProgramError> {
        let intermediates = self.intermediates_in(carrier, root)?;

        // The traversal always pushes the root itself first, so the chain is never empty and its last snapshot
        // is the value this view selects.
        let previous = intermediates.last().unwrap().clone();
        let intermediates = &intermediates[..self.bound_transforms().len()];
        let reconstructed = self.reconstruct_in(carrier, intermediates, replacement)?;
        Ok((previous, reconstructed))
    }

    /// Returns `root` with the value that this path selects replaced by `replacement`, like [`swap_in`](Self::swap_in)
    /// but without computing the previous selected value.
    ///
    /// The traversal stops at the parent of the leaf. Rebuilding the root needs every strict parent, so that elements
    /// outside the view survive, but applying the last transform would only produce the old selected value, which a
    /// write must not observe (e.g., in a staged program, it would stage a read that nothing uses). An empty path
    /// therefore returns `replacement` itself.
    fn write_in<C: TransformWriteCarrier<Binding = Binding>>(
        &self,
        carrier: &C,
        root: C::Value,
        replacement: C::Value,
    ) -> Result<C::Value, ProgramError> {
        let Some((_, parents)) = self.bound_transforms().split_last() else {
            return Ok(replacement);
        };
        let intermediates = Self::intermediates_through(carrier, root, parents)?;
        self.reconstruct_in(carrier, intermediates.as_slice(), replacement)
    }

    /// Returns `root` followed by the child that each of `bound_transforms` selects from the preceding value.
    /// This is [`intermediates_in`](Self::intermediates_in) for any prefix of this path's transforms, which lets
    /// [`write_in`](Self::write_in) stop at the parent of the leaf.
    fn intermediates_through<C: TransformReadCarrier<Binding = Binding>>(
        carrier: &C,
        root: C::Value,
        bound_transforms: &[BoundReferenceTransform<ArrayReferenceTransform, Binding>],
    ) -> Result<Vec<C::Value>, ProgramError> {
        let mut intermediates = Vec::with_capacity(bound_transforms.len() + 1);
        intermediates.push(root);
        for bound_transform in bound_transforms {
            let parent = intermediates.last().unwrap();
            let child = bound_transform.transform().apply_in(carrier, parent, bound_transform.bindings())?;
            intermediates.push(child);
        }
        Ok(intermediates)
    }
}

impl ArrayReferenceTransformPath<NoReferenceTransformBinding> {
    /// Returns the value that this static path selects from `root` (i.e., the last entry of
    /// [`intermediates_in`](Self::intermediates_in)) without keeping the values along the way.
    /// `root` is consumed, and so an empty path returns it without copying.
    fn apply<A: Value<Type = ArrayType> + Reshape + Slice>(&self, root: A) -> Result<A, ProgramError> {
        let carrier = EagerTransformCarrier(PhantomData);
        self.bound_transforms().iter().try_fold(root, |value, bound_transform| {
            bound_transform.transform().apply_in(&carrier, &value, bound_transform.bindings())
        })
    }
}

/// Index selected by an [`Index`](ArrayReferenceTransform::Index) transform.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum ArrayReferenceTransformIndex {
    /// An index known when the transform is described.
    Static(usize),

    /// An index supplied by the next ordinary input in the access's binding sequence.
    Dynamic,
}

/// One validated index [`ReferenceTransform`] in an [`ArrayReferenceTransformPath`]'s root-to-handle mapping.
///
/// A transform describes both directions of one selection: applying it extracts a selected child value from its parent,
/// while replacing that child reconstructs a value with exactly the parent's original type. This bidirectional contract
/// lets reference reads operate on the selected value and lets write-only replacements, swaps, or additive updates
/// reconstruct the shared root without changing its declared type. A write-only traversal materializes the strict
/// parents needed for reconstruction but deliberately skips extracting the overwritten leaf.
///
/// Transforms are interpreted in order from the root outward. [`Index`](Self::Index) removes one axis at a static or
/// dynamic index. [`Slice`](Self::Slice) preserves rank and selects one static unit-stride range per axis. A dynamic
/// index is supplied by the next input in the access's binding group. Analysis records that input's [`ValueId`] in
/// the corresponding [`BoundReferenceTransform`]. For example, the built-in scan binds its explicit body index to a
/// leading dynamic index on each access to a stacked reference. Eager handles resolve dynamic indices into static
/// transforms when an access applies its path. Discharge reconstructs dynamic indices with dynamic slicing and updates.
/// A negative runtime index counts from the end of the indexed axis once, then the result is clamped to that axis's
/// valid range, following the array dynamic-slicing contract. Strided slicing remains unsupported.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum ArrayReferenceTransform {
    /// Selects a position along one axis and removes that axis from the shape.
    Index {
        /// Axis of the transform's input that this transform indexes.
        axis: usize,

        /// Index selected on `axis`.
        index: ArrayReferenceTransformIndex,
    },

    /// Selects one static unit-stride range on every axis while preserving rank.
    Slice {
        /// Per-axis slices of the transform's input.
        axes: Vec<ArraySliceAxis>,
    },
}

impl ArrayReferenceTransform {
    /// Returns the exact canonical [`ArrayType`] produced from `input`. A dynamic index removes its axis exactly like
    /// a static one, without the static bounds check and write-back check, because the index it selects is only
    /// known to the access that applies the transform.
    pub fn output_type(&self, input: &ArrayType) -> Result<ArrayType, TypeError> {
        let (output, selection) = self.selected_type(input)?;
        let Some(selection) = selection else {
            return Ok(output);
        };

        // Check that the selected value can be written back into `input`. Shape arithmetic alone cannot guarantee this:
        // `ArrayType` also carries layouts, shardings, and other metadata whose slice and update-slice derivations are
        // owned by the type system, so an update-slice may reject a selection whose metadata does not fit back into
        // its parent. An accepted update-slice always produces `input` itself, so only its acceptance needs checking.
        // The check runs when a view is constructed and when an access that writes back derives its path type.
        // Read-only accesses derive their path types through `ReferenceTransform::read_type`, which skips it.
        let update = if selection.removed_axis.is_some() {
            output.reshape(selection.update_shape()).map_err(|error| TypeError::invalid(error.to_string()))?
        } else {
            output.clone()
        };

        input
            .update_slice(&update, selection.starts.as_slice())
            .map_err(|error| TypeError::invalid(error.to_string()))?;

        Ok(output)
    }

    /// Returns the exact canonical array type selected from `input` together with the static selection that produced
    /// it, or [`None`] for a dynamic index, whose selection is only known to the access that applies it. This is
    /// [`output_type`](Self::output_type) without the write-back check, which read-only accesses do not need.
    fn selected_type(&self, input: &ArrayType) -> Result<(ArrayType, Option<TransformSelection>), TypeError> {
        if let Self::Index { axis, index: ArrayReferenceTransformIndex::Dynamic } = self {
            Self::indexed_shape(*axis, input)?;
            return Ok((input.without_dimension(*axis)?.0, None));
        }
        let selection = self.selection(input)?;
        let sliced = input
            .slice(selection.starts.as_slice(), selection.limits.as_slice(), &vec![1; selection.starts.len()])
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        let output = if selection.removed_axis.is_some() {
            sliced.reshape(selection.output_shape()).map_err(|error| TypeError::invalid(error.to_string()))?
        } else {
            sliced
        };
        Ok((output, Some(selection)))
    }

    /// Validates the axis of an [`Index`](Self::Index) transform against `input`
    /// and returns the [`StaticShape`] of `input`.
    fn indexed_shape(axis: usize, input: &ArrayType) -> Result<StaticShape, TypeError> {
        let shape = input.static_shape().ok_or_else(|| {
            TypeError::invalid(format!("reference indexing requires a static referent type but got `{input}`"))
        })?;
        if axis >= shape.rank() {
            return Err(TypeError::invalid(format!(
                "reference index axis {} is out of bounds for rank {}",
                axis,
                shape.rank(),
            )));
        }
        Ok(shape)
    }

    /// Validates this transform against `input` and returns its normalized [`TransformSelection`].
    fn selection(&self, input: &ArrayType) -> Result<TransformSelection, TypeError> {
        match self {
            Self::Index { axis, index } => {
                let shape = Self::indexed_shape(*axis, input)?;
                let index = match index {
                    ArrayReferenceTransformIndex::Static(index) => *index,
                    ArrayReferenceTransformIndex::Dynamic => {
                        return Err(TypeError::invalid(
                            "a dynamic index has no static selection; apply its binding at the reference access",
                        ));
                    }
                };
                if index >= shape.dimension(*axis) {
                    return Err(TypeError::invalid(format!(
                        "reference index {} on axis {} is out of bounds for size {}",
                        index,
                        axis,
                        shape.dimension(*axis),
                    )));
                }
                let mut starts = vec![0; shape.rank()];
                starts[*axis] = index;
                let mut limits = shape.dimensions().to_vec();
                limits[*axis] = index + 1;
                Ok(TransformSelection { starts, limits, removed_axis: Some(*axis) })
            }
            Self::Slice { axes } => {
                let shape = input.static_shape().ok_or_else(|| {
                    TypeError::invalid(format!("reference slicing requires a static referent type but got `{input}`"))
                })?;
                if axes.len() != shape.rank() {
                    return Err(TypeError::invalid(format!(
                        "reference slice has {} axes but its input has rank {}",
                        axes.len(),
                        shape.rank(),
                    )));
                }
                let mut starts = Vec::with_capacity(axes.len());
                let mut limits = Vec::with_capacity(axes.len());
                for (axis, (slice_axis, input_size)) in axes.iter().copied().zip(shape.dimensions()).enumerate() {
                    if slice_axis.stride() != 1 {
                        return Err(TypeError::invalid(format!(
                            "reference slice axis {axis} stride must be 1 until scatter-backed strided updates are \
                             supported",
                        )));
                    }
                    let limit = slice_axis.start().checked_add(slice_axis.size()).ok_or_else(|| {
                        TypeError::invalid(format!("reference slice limit overflows `usize` on axis {axis}"))
                    })?;
                    if limit > *input_size {
                        return Err(TypeError::invalid(format!(
                            "reference slice on axis {} with start {} and size {} exceeds input size {}",
                            axis,
                            slice_axis.start(),
                            slice_axis.size(),
                            input_size,
                        )));
                    }
                    starts.push(slice_axis.start());
                    limits.push(limit);
                }
                Ok(TransformSelection { starts, limits, removed_axis: None })
            }
        }
    }

    /// Applies this transform to one carried parent value. A dynamic index is resolved by the carrier from the one
    /// value the transform's `bindings` close it over; a dynamic index that binds no value (an eager path, or a
    /// malformed closure) has no selection and is rejected by [`selection`](Self::selection).
    fn apply_in<C: TransformReadCarrier>(
        &self,
        carrier: &C,
        input: &C::Value,
        bindings: &[C::Binding],
    ) -> Result<C::Value, ProgramError> {
        if let (Self::Index { axis, index: ArrayReferenceTransformIndex::Dynamic }, [binding]) = (self, bindings) {
            return carrier.dynamic_index(input, *axis, binding);
        }
        let selection = self.selection(carrier.array_type(input)?.as_ref())?;
        let output_shape = selection.removed_axis.map(|_| selection.output_shape());
        let sliced = carrier.slice(input, selection.starts, selection.limits)?;
        match output_shape {
            Some(shape) => carrier.reshape(&sliced, shape),
            None => Ok(sliced),
        }
    }

    /// Reconstructs the carried parent after replacing exactly the elements selected by this transform, resolving
    /// a dynamic index exactly as [`apply_in`](Self::apply_in) does.
    fn replace_in<C: TransformWriteCarrier>(
        &self,
        carrier: &C,
        input: &C::Value,
        replacement: &C::Value,
        bindings: &[C::Binding],
    ) -> Result<C::Value, ProgramError> {
        if let (Self::Index { axis, index: ArrayReferenceTransformIndex::Dynamic }, [binding]) = (self, bindings) {
            return carrier.dynamic_update_index(input, replacement, *axis, binding);
        }
        let selection = self.selection(carrier.array_type(input)?.as_ref())?;
        if selection.removed_axis.is_some() {
            let update = carrier.reshape(replacement, selection.update_shape())?;
            carrier.update_slice(input, &update, selection.starts)
        } else {
            carrier.update_slice(input, replacement, selection.starts)
        }
    }
}

impl Display for ArrayReferenceTransform {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Index { axis, index: ArrayReferenceTransformIndex::Static(index) } => {
                write!(formatter, "index(axis={axis}, index={index})")
            }
            Self::Index { axis, index: ArrayReferenceTransformIndex::Dynamic } => {
                write!(formatter, "index(axis={axis}, index=dynamic)")
            }
            Self::Slice { axes } => {
                // Each axis renders as `start:limit`, with the tight exclusive limit one past its last selected index,
                // followed by `:stride` when the stride is not one.
                write!(formatter, "slice(axes=[")?;
                for (index, axis) in axes.iter().enumerate() {
                    if index > 0 {
                        write!(formatter, ", ")?;
                    }
                    let limit = axis.start() + axis.size().saturating_sub(1) * axis.stride() + axis.size().min(1);
                    write!(formatter, "{}:{limit}", axis.start())?;
                    if axis.stride() != 1 {
                        write!(formatter, ":{}", axis.stride())?;
                    }
                }
                write!(formatter, "])")
            }
        }
    }
}

impl ReferenceTransform for ArrayReferenceTransform {
    type Type = ArrayIrType;
    type Referent = ArrayType;

    #[inline]
    fn binding_count(&self) -> usize {
        usize::from(matches!(self, Self::Index { index: ArrayReferenceTransformIndex::Dynamic, .. }))
    }

    fn validate_bindings(&self, input: &ArrayType, bindings: &[&ArrayIrType]) -> Result<(), TypeError> {
        check_count!("binding", bindings, self.binding_count(), TypeError);
        if let Self::Index { index: ArrayReferenceTransformIndex::Dynamic, .. } = self {
            let index = <&ArrayType>::try_from(bindings[0])?;
            if index.rank() != 0 || !index.data_type().is_integer() {
                return Err(TypeError::invalid(format!(
                    "reference transform requires a scalar integer index but received `{index}`",
                )));
            }
            if index.memory() != input.memory() {
                return Err(TypeError::invalid(format!(
                    "reference transform and index must share one memory space but index resides in {} \
                     and reference resides in {}",
                    index.memory(),
                    input.memory(),
                )));
            }
        }
        Ok(())
    }

    #[inline]
    fn output_type(&self, input: &ArrayType) -> Result<ArrayType, TypeError> {
        self.output_type(input)
    }

    #[inline]
    fn read_type(&self, input: &ArrayType) -> Result<ArrayType, TypeError> {
        // Reads never write back through the view, so they skip the write-back check of `output_type`.
        Ok(self.selected_type(input)?.0)
    }

    fn overlap(
        r#type: &ArrayIrType,
        lhs: &[BoundReferenceTransform<Self>],
        rhs: &[BoundReferenceTransform<Self>],
    ) -> ReferenceViewOverlap {
        // Both paths fold to one static range or dynamic index per root axis. Non-intersecting static ranges prove
        // disjointness; identical static ranges or dynamic indices with equal bindings, offsets, and clamping extents
        // prove equality. Everything else may overlap. A malformed path cannot be folded and is treated as possibly
        // overlapping, because paths are validated when they are derived and this query must not fail.
        let Some(shape) = <&ReferenceType<ArrayType>>::try_from(r#type)
            .ok()
            .and_then(|r#type| r#type.referent().static_shape())
        else {
            return ReferenceViewOverlap::MayOverlap;
        };

        let (Some(lhs), Some(rhs)) = (RootIndexSelection::fold(&shape, lhs), RootIndexSelection::fold(&shape, rhs))
        else {
            return ReferenceViewOverlap::MayOverlap;
        };

        let mut overlap = ReferenceViewOverlap::Same;
        for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
            match lhs.overlap(rhs) {
                ReferenceViewOverlap::Disjoint => return ReferenceViewOverlap::Disjoint,
                ReferenceViewOverlap::Same => {}
                ReferenceViewOverlap::MayOverlap => overlap = ReferenceViewOverlap::MayOverlap,
            }
        }

        overlap
    }
}

impl BatchableReferenceTransform for ArrayReferenceTransform {
    fn batch(&self, r#type: &ArrayIrType, batch_axis: BatchAxis) -> Result<(Self, BatchAxis), BatchingError> {
        // The batch axis of a reference is an axis of its packed referent that the per-item transform never sees.
        // Indexing removes one per-item axis, so the packed transform cannot keep both axis positions unchanged: a
        // batch axis at or before the indexed axis shifts the packed indexed axis one position later while the output
        // keeps the batch axis, and a batch axis after the indexed axis leaves the packed indexed axis alone while the
        // output's batch axis moves one position earlier. Slicing preserves rank, so the packed transform selects the
        // complete batch axis through an identity selection inserted at the batch axis position and the output keeps
        // the batch axis.
        let Some(axis) = batch_axis.axis() else {
            return Ok((self.clone(), batch_axis));
        };
        let referent = <&ReferenceType<ArrayType>>::try_from(r#type)?.referent();
        let position = axis.normalize(referent.rank())?;

        // Normalizing the batch axis proves that the packed rank is nonzero. Validate the descriptor's per-item
        // axes before shifting an index or inserting a slice axis, so malformed public descriptors cannot panic.
        let rank = referent.rank() - 1;
        match self {
            Self::Index { axis: indexed_axis, .. } if *indexed_axis >= rank => Err(TypeError::invalid(format!(
                "reference index axis {indexed_axis} is out of bounds for rank {rank}",
            ))
            .into()),
            Self::Slice { axes } if axes.len() != rank => Err(TypeError::invalid(format!(
                "reference slice has {} axes but its input has rank {}",
                axes.len(),
                rank,
            ))
            .into()),
            Self::Index { axis: indexed_axis, index } if position <= *indexed_axis => {
                Ok((Self::Index { axis: indexed_axis + 1, index: *index }, BatchAxis::from_position(position)))
            }
            Self::Index { axis: indexed_axis, index } => {
                Ok((Self::Index { axis: *indexed_axis, index: *index }, BatchAxis::from_position(position - 1)))
            }
            Self::Slice { axes } => {
                let size = match &referent.shape().dimensions()[position] {
                    Dimension::Static(size) => *size,
                    Dimension::Dynamic(_) => {
                        return Err(BatchingError::DynamicBatchAxis { r#type: Box::new(referent.clone()), axis });
                    }
                };
                let mut axes = axes.clone();
                axes.insert(position, ArraySliceAxis::new(0, size, 1));
                Ok((Self::Slice { axes }, batch_axis))
            }
        }
    }
}

impl<Root: Typed, Binding: Clone + Typed<Type = ArrayIrType>> ReferenceView<Root, ArrayReferenceTransform, Binding> {
    /// Returns this view narrowed to position `index` on `axis`, with that axis removed from the viewed referent. For
    /// example, indexing axis `0` of an `f32[2, 3]` view at `1` produces an `f32[3]` view of its second row. The index
    /// is static, so it is validated against the viewed referent immediately.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if the viewed referent does not have a static shape, if `axis` is out of bounds for its
    /// rank, if `index` is out of bounds for `axis`, or if writing the selected elements back would not reconstruct the
    /// exact type of the viewed referent.
    #[inline]
    pub fn index(self, axis: usize, index: usize) -> Result<Self, ProgramError> {
        self.with_bound_transform(
            ArrayReferenceTransform::Index { axis, index: ArrayReferenceTransformIndex::Static(index) },
            Vec::new(),
        )
    }

    /// Returns this view narrowed to the runtime position that `index` holds on `axis`, with that axis removed from
    /// the viewed referent as in [`Self::index`]. Only the type of `index` is validated here. Each access through the
    /// returned view reads the value of `index`, counts a negative index from the end of `axis` once, and then clamps
    /// the result to the valid range of `axis`. An access fails if `axis` is empty.
    ///
    /// # Parameters
    ///
    ///   - `axis`: Axis of the viewed referent to index.
    ///   - `index`: Scalar integer value that holds the position, in the same memory space as the viewed referent.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if `index` is not a scalar integer, if it resides in a different memory space than the
    /// viewed referent, if the viewed referent does not have a static shape, or if `axis` is out of bounds for its
    /// rank.
    #[inline]
    pub fn dynamic_index(self, axis: usize, index: &Binding) -> Result<Self, ProgramError> {
        self.with_bound_transform(
            ArrayReferenceTransform::Index { axis, index: ArrayReferenceTransformIndex::Dynamic },
            vec![index.clone()],
        )
    }

    /// Returns this view narrowed to one static unit-stride range per axis, which preserves the rank of the viewed
    /// referent. `axes` holds exactly one [`ArraySliceAxis`] per axis. For example, slicing an `f32[4]` view with
    /// `ArraySliceAxis::new(1, 2, 1)` produces an `f32[2]` view of its two middle elements.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if the viewed referent does not have a static shape, if `axes` does not have one entry
    /// per axis, if an entry has a stride other than one, if a range extends past the end of its axis, or if writing
    /// the selected elements back would not reconstruct the exact type of the viewed referent.
    #[inline]
    pub fn slice(self, axes: &[ArraySliceAxis]) -> Result<Self, ProgramError> {
        self.with_bound_transform(ArrayReferenceTransform::Slice { axes: axes.to_vec() }, Vec::new())
    }
}

/// [`ReferenceDischargePolicy`] of the array reference universe. An array reference's referent is an array. Each access
/// applies an [`ArrayReferenceTransformPath`] to its allocation, with dynamic indices bound to context values. The
/// policy uses the same transform traversal as [`ArrayReference`]. Specifically, it reads extract the viewed array,
/// while replacements and accumulations reconstruct the root through the transforms in reverse order, preserving values
/// outside the view. Dynamic indices use dynamic slicing and updates with the negative index and clamping behavior
/// described by [`ArrayReferenceTransform`], so eager and discharged accesses address the same elements.
///
/// The generic [`references` module](crate::programs::references) owns discharge and state threading. This policy
/// supplies array-specific reconstruction operations rather than a separate state interpreter.
#[derive(Copy, Clone, Debug)]
pub struct ArrayReferenceDischarge;

impl<C: Context<Type = ArrayIrType>> ReferenceDischargePolicy<C> for ArrayReferenceDischarge
where
    C::Operation: OperationProjection<
            ArrayType,
            Projected: From<ReshapeOperation>
                           + From<SliceOperation>
                           + From<UpdateSliceOperation>
                           + From<DynamicSliceOperation>
                           + From<DynamicUpdateSliceOperation>,
        >,
{
    type Referent = ArrayType;
    type Transform = ArrayReferenceTransform;
    type Alias = ArrayReferenceTransformPath<C::Value>;

    #[inline]
    fn storage_alias(_referent: &ArrayType) -> ArrayReferenceTransformPath<C::Value> {
        ArrayReferenceTransformPath::root()
    }

    #[inline]
    fn apply_transforms(
        _context: &C,
        alias: &Self::Alias,
        transforms: &[ArrayReferenceTransform],
        bindings: &[C::Value],
    ) -> Result<Self::Alias, ProgramError> {
        let mut composed = alias.clone();
        composed.append(ArrayReferenceTransformPath::from_transforms(transforms, bindings)?);
        Ok(composed)
    }

    #[inline]
    fn read(
        context: &C,
        current: &C::Value,
        alias: &ArrayReferenceTransformPath<C::Value>,
    ) -> Result<C::Value, ProgramError> {
        // The traversal starts with the complete allocation, so the chain is non-empty and its final value is the part
        // selected by this handle.
        let mut intermediates = alias.intermediates_in(&ContextTransformCarrier { context }, current.clone())?;
        Ok(intermediates.pop().unwrap())
    }

    #[inline]
    fn write(
        context: &C,
        current: &C::Value,
        replacement: C::Value,
        alias: &ArrayReferenceTransformPath<C::Value>,
    ) -> Result<C::Value, ProgramError> {
        alias.write_in(&ContextTransformCarrier { context }, current.clone(), replacement)
    }

    #[inline]
    fn swap(
        context: &C,
        current: &C::Value,
        replacement: C::Value,
        alias: &ArrayReferenceTransformPath<C::Value>,
    ) -> Result<(C::Value, C::Value), ProgramError> {
        alias.swap_in(&ContextTransformCarrier { context }, current.clone(), replacement)
    }
}

impl<C: Context<Type = ArrayIrType>> ReferenceAccumulationPolicy<C> for ArrayReferenceDischarge
where
    C::Operation: From<AddOperation<ArrayIrType>>
        + OperationProjection<
            ArrayType,
            Projected: From<ReshapeOperation>
                           + From<SliceOperation>
                           + From<UpdateSliceOperation>
                           + From<DynamicSliceOperation>
                           + From<DynamicUpdateSliceOperation>,
        >,
{
    fn accumulate(
        context: &C,
        current: &C::Value,
        update: C::Value,
        alias: &ArrayReferenceTransformPath<C::Value>,
    ) -> Result<C::Value, ProgramError> {
        // Composite array IR values deliberately expose no value-level addition: the composite family carries array
        // payloads through `ArrayIrOperation::Array` and lifts the type-generic `AddOperation<ArrayIrType>` into that
        // member instead, which is the same seam generic reverse mode uses to accumulate cotangents. Accumulation
        // therefore binds the lifted addition through the context, requiring nothing beyond the conversion the
        // operation family already provides.
        let carrier = ContextTransformCarrier { context };
        let intermediates = alias.intermediates_in(&carrier, current.clone())?;

        // Add at the selected leaf, then rebuild each enclosing slice without reading the leaf a second time.
        let selected = intermediates.last().unwrap().clone();
        let mut outputs = context.bind(C::Operation::from(AddOperation::new()), Vec::new(), &[selected, update])?;
        check_count!("output", outputs, 1, ProgramError);
        let accumulated = outputs.remove(0);
        alias.reconstruct_in(&carrier, &intermediates[..alias.transforms().len()], accumulated)
    }
}

impl ReferenceDischargeableType for ArrayIrType {
    type Policy = ArrayReferenceDischarge;
}

impl<A: Value<Type = ArrayType>> ReferenceAccessOperation for ArrayIrOperation<A> {
    type Transform = ArrayReferenceTransform;

    fn base_input_count(&self) -> usize {
        match self {
            Self::ReferenceRead(operation) => operation.base_input_count(),
            Self::ReferenceWrite(operation) => operation.base_input_count(),
            Self::ReferenceAddUpdate(operation) => operation.base_input_count(),
            Self::ReferenceSwap(operation) => operation.base_input_count(),
            Self::ReferenceAtomicAddUpdate(operation) => operation.base_input_count(),
            Self::ReferenceFreeze(_) => 1,
            _ => 0,
        }
    }

    fn reference_access_descriptor(
        &self,
        input_index: usize,
    ) -> Option<ReferenceAccessDescriptor<'_, ArrayReferenceTransform>> {
        match self {
            Self::ReferenceRead(operation) => operation.reference_access_descriptor(input_index),
            Self::ReferenceWrite(operation) => operation.reference_access_descriptor(input_index),
            Self::ReferenceAddUpdate(operation) => operation.reference_access_descriptor(input_index),
            Self::ReferenceSwap(operation) => operation.reference_access_descriptor(input_index),
            Self::ReferenceAtomicAddUpdate(operation) => operation.reference_access_descriptor(input_index),
            Self::ReferenceFreeze(_) if input_index == 0 => Some(ReferenceAccessDescriptor::new(&[], 1..1)),
            _ => None,
        }
    }

    fn with_reference_access_transforms(
        &self,
        input_index: usize,
        transforms: Vec<ArrayReferenceTransform>,
    ) -> Result<Self, ProgramError> {
        match self {
            Self::ReferenceRead(operation) => {
                operation.with_reference_access_transforms(input_index, transforms).map(Self::ReferenceRead)
            }
            Self::ReferenceWrite(operation) => {
                operation.with_reference_access_transforms(input_index, transforms).map(Self::ReferenceWrite)
            }
            Self::ReferenceAddUpdate(operation) => {
                operation.with_reference_access_transforms(input_index, transforms).map(Self::ReferenceAddUpdate)
            }
            Self::ReferenceSwap(operation) => {
                operation.with_reference_access_transforms(input_index, transforms).map(Self::ReferenceSwap)
            }
            Self::ReferenceAtomicAddUpdate(operation) => operation
                .with_reference_access_transforms(input_index, transforms)
                .map(Self::ReferenceAtomicAddUpdate),
            _ if self.reference_access_descriptor(input_index).is_some() && transforms.is_empty() => Ok(self.clone()),
            _ => Err(ProgramError::UnsupportedOperation {
                message: format!("`{}` cannot replace the reference transforms at input {}", self.name(), input_index),
            }),
        }
    }
}

/// Normalized indices of one [`ArrayReferenceTransform`] applied to one statically shaped input. Both transform kinds
/// reduce to taking one static unit-stride slice of the input, optionally followed by squeezing the indexed axis.
/// Normalizing to this shared form lets every consumer (e.g., type derivation, eager reads, eager update
/// reconstruction, and staged discharge) share one validation and address computation.
struct TransformSelection {
    /// Inclusive slice start per input axis.
    starts: Vec<usize>,

    /// Exclusive slice limit per input axis.
    limits: Vec<usize>,

    /// Axis that an [`ArrayReferenceTransform::Index`] transform removes from the output after slicing it to size one,
    /// or [`None`] for rank-preserving [`ArrayReferenceTransform::Slice`]s.
    removed_axis: Option<usize>,
}

impl TransformSelection {
    /// Returns the static shape of the slice before squeezing (i.e., the update shape that writes back into the
    /// selected indices).
    fn update_shape(&self) -> Shape {
        Shape::new(
            self.starts
                .iter()
                .zip(self.limits.iter())
                .map(|(start, limit)| Dimension::Static(limit - start))
                .collect(),
        )
    }

    /// Returns the static shape of the value that the transform selects. This is either [`Self::update_shape`] without
    /// [`Self::removed_axis`], or exactly [`Self::update_shape`] when the transform removes no axis.
    fn output_shape(&self) -> Shape {
        Shape::new(
            self.starts
                .iter()
                .zip(self.limits.iter())
                .enumerate()
                .filter(|(axis, _)| Some(*axis) != self.removed_axis)
                .map(|(_, (start, limit))| Dimension::Static(limit - start))
                .collect(),
        )
    }
}

/// Indices that a folded [`ArrayReferenceTransformPath`] selects on one axis of its root,
/// used by [`ReferenceTransform::overlap`] to compare two paths of one root.
#[derive(Clone, Debug, PartialEq, Eq)]
enum RootIndexSelection {
    /// A static unit-stride range `[start, limit)` of the root axis. Before any transform touches the axis this is the
    /// complete axis, a slice narrows it, and a static index collapses it to one index.
    Static {
        /// Inclusive start of the range.
        start: usize,

        /// Exclusive limit of the range.
        limit: usize,
    },

    /// One index `offset + clamp(wrap(index), 0, extent - 1)` of the root axis, selected relative to the range that
    /// earlier transforms narrowed the axis to, where `index` is the runtime value of the binding and `wrap(index)`
    /// is `index + extent` for a negative `index` and `index` otherwise. Both wrapping and clamping depend on this
    /// extent, not just the binding.
    Dynamic {
        /// Binding that supplies the dynamic index.
        binding: ValueId,

        /// Start of the narrowed range that the dynamic index is relative to.
        offset: usize,

        /// Size of the narrowed axis against which the runtime index is wrapped and clamped.
        extent: usize,
    },
}

impl RootIndexSelection {
    /// Translates the transforms of a path into what they select on each axis of a root of static shape `shape`,
    /// in root coordinates. [`ReferenceTransform::overlap`] compares two paths through this translation, and
    /// [`ArrayReferenceTransformPath::root_slice_axes`] exposes it for static paths.
    ///
    /// Each transform is written in the coordinates of the view that it is applied to. A slice narrows each view axis
    /// relative to that axis's current start, and an index selects a position relative to that start and removes the
    /// axis from the view, so later transforms number the view axes without it. This function replays the path while
    /// tracking the root axes that remain in the view, in view axis order, each with the range that the path has
    /// narrowed it to. Every root axis starts as its complete [`Static`](Self::Static) range. A slice narrows the
    /// ranges of the axes that remain in the view, a static index collapses one range to a single index, and a
    /// dynamic index replaces one range with a [`Dynamic`](Self::Dynamic) index relative to it. Both kinds of index
    /// also remove their axis from the view.
    ///
    /// For example, over an `f32[4, 6]` root, the path `[slice(axes=[1:4, 2:6]), index(axis=0, index=dynamic),
    /// slice(axes=[1:3])]`, with its dynamic index bound to `%i`, folds as follows:
    ///
    /// ```text
    ///     transform                       root axis 0                  root axis 1    remaining view axes
    ///     (root)                          0:4                          0:6            [0, 1]
    ///     slice(axes=[1:4, 2:6])          1:4                          2:6            [0, 1]
    ///     index(axis=0, index=dynamic)    1 + clamp(wrap(%i), 0, 2)    2:6            [1]
    ///     slice(axes=[1:3])               1 + clamp(wrap(%i), 0, 2)    3:5            [1]
    /// ```
    ///
    /// The final `slice(axes=[1:3])` has one axis because the dynamic index removed view axis `0`, and it narrows
    /// root axis `1` from `2:6` to `3:5`. The path therefore selects root columns `3..5` of the root row
    /// `1 + clamp(wrap(%i), 0, 2)`.
    ///
    /// Returns [`None`] if the path does not fold against `shape`, which covers the paths that
    /// [`ArrayReferenceTransform::output_type`] rejects: an axis out of bounds, a static index or slice range that
    /// extends past its current range, a slice with the wrong number of axes or a stride other than one, and a dynamic
    /// index without a binding. It also returns [`None`] when a root coordinate overflows `usize`, so that malformed
    /// paths stay conservative.
    ///
    /// # Parameters
    ///
    ///   - `shape`: Static shape of the root.
    ///   - `bound_transforms`: Transforms of the path, in order from the root, with the program values that bind
    ///     their dynamic indices.
    fn fold(
        shape: &StaticShape,
        bound_transforms: &[BoundReferenceTransform<ArrayReferenceTransform>],
    ) -> Option<Vec<Self>> {
        let mut selections =
            shape.dimensions().iter().map(|size| Self::Static { start: 0, limit: *size }).collect::<Vec<_>>();

        // Root axes that remain in the view, in view axis order, each with the static range `start..limit` that the
        // path has narrowed it to so far. Indexing an axis removes it from this list and records its final selection,
        // so the ranges of the axes that are still listed are written back once the whole path has been folded.
        let mut remaining = shape
            .dimensions()
            .iter()
            .enumerate()
            .map(|(root_axis, size)| (root_axis, 0usize, *size))
            .collect::<Vec<_>>();
        for bound_transform in bound_transforms {
            match bound_transform.transform() {
                ArrayReferenceTransform::Index { axis, index } => {
                    if *axis >= remaining.len() {
                        return None;
                    }
                    let (root_axis, start, limit) = remaining.remove(*axis);
                    selections[root_axis] = match index {
                        ArrayReferenceTransformIndex::Static(index) => {
                            // Invalid paths must remain conservative even when the relative index overflows.
                            let index = start.checked_add(*index)?;
                            if index >= limit {
                                return None;
                            }
                            Self::Static { start: index, limit: index + 1 }
                        }
                        ArrayReferenceTransformIndex::Dynamic => {
                            let binding = *bound_transform.bindings().first()?;
                            Self::Dynamic { binding, offset: start, extent: limit - start }
                        }
                    };
                }
                ArrayReferenceTransform::Slice { axes } => {
                    if axes.len() != remaining.len() {
                        return None;
                    }
                    for (slice_axis, (_, start, limit)) in axes.iter().zip(remaining.iter_mut()) {
                        let narrowed_start = start.checked_add(slice_axis.start())?;
                        let narrowed_limit = narrowed_start.checked_add(slice_axis.size())?;
                        if slice_axis.stride() != 1 || narrowed_limit > *limit {
                            return None;
                        }
                        (*start, *limit) = (narrowed_start, narrowed_limit);
                    }
                }
            }
        }

        for (root_axis, start, limit) in remaining {
            selections[root_axis] = Self::Static { start, limit };
        }

        Some(selections)
    }

    /// Returns the [`ReferenceViewOverlap`] between the indices that this [`RootIndexSelection`] and `other`
    /// select on one root axis.
    fn overlap(&self, other: &Self) -> ReferenceViewOverlap {
        match (self, other) {
            (Self::Static { start: a_start, limit: a_limit }, Self::Static { start: b_start, limit: b_limit }) => {
                if a_limit <= b_start || b_limit <= a_start {
                    ReferenceViewOverlap::Disjoint
                } else if a_start == b_start && a_limit == b_limit {
                    ReferenceViewOverlap::Same
                } else {
                    ReferenceViewOverlap::MayOverlap
                }
            }
            (
                Self::Dynamic { binding: a_binding, offset: a_offset, extent: a_extent },
                Self::Dynamic { binding: b_binding, offset: b_offset, extent: b_extent },
            ) if a_binding == b_binding && a_offset == b_offset && a_extent == b_extent => ReferenceViewOverlap::Same,
            (Self::Dynamic { .. }, Self::Dynamic { .. })
            | (Self::Static { .. }, Self::Dynamic { .. })
            | (Self::Dynamic { .. }, Self::Static { .. }) => ReferenceViewOverlap::MayOverlap,
        }
    }
}

/// Array operations through which an [`ArrayReferenceTransformPath`] reads the value that it selects from a root.
///
/// Array references are accessed in two different settings. An eager [`ArrayReference`] reads and writes concrete
/// array values directly through their array-manipulation capabilities (e.g., [`Slice`] and [`Reshape`]), while
/// [`ArrayReferenceDischarge`] rewrites staged accesses into array operations that it binds into a program context,
/// whose operation family may be a backend-owned superset of the core array IR. This trait abstracts over the two,
/// so that the traversals which map a root to the value that a path selects (e.g.,
/// [`intermediates_in`](ArrayReferenceTransformPath::intermediates_in)) are written once, generically over their
/// carrier, and eager and discharged accesses cannot drift apart.
///
/// A static transform lowers to [`slice`](Self::slice), followed for an index by a [`reshape`](Self::reshape) that
/// removes the indexed axis, whereas a dynamic index lowers to [`dynamic_index`](Self::dynamic_index), which receives
/// the binding that supplies the index. Writing lives in the separate [`TransformWriteCarrier`], so that read-only
/// traversals do not require update capabilities (e.g., [`ArrayReference::read`] only requires [`Reshape`] and
/// [`Slice`]).
trait TransformReadCarrier {
    /// Representation of the values that the traversal reads and produces: concrete array values for eager handles,
    /// and context values for reference discharge.
    type Value;

    /// Value that supplies a dynamic index of the traversed path (i.e., the uninhabited [`NoReferenceTransformBinding`]
    /// for eager paths, which carry only static transforms, and the context value that holds the index for reference
    /// discharge).
    type Binding;

    /// Returns the [`ArrayType`] of `value`, which the traversal uses to validate each static transform and compute
    /// its slice bounds. The type is borrowed from the carrier or from `value` where possible.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if `value` is not an array (e.g., a reference-typed context value).
    fn array_type<'c>(&'c self, value: &'c Self::Value) -> Result<Cow<'c, ArrayType>, ProgramError>;

    /// Returns the unit-stride slice of `input` that starts at `starts` (inclusive) and ends at `limits` (exclusive),
    /// with one entry per axis of `input`.
    ///
    /// # Errors
    ///
    /// Forwards the error of a slice that does not apply to `input`.
    fn slice(&self, input: &Self::Value, starts: Vec<usize>, limits: Vec<usize>) -> Result<Self::Value, ProgramError>;

    /// Returns `input` reshaped to `shape`. The traversal uses it to remove the size-one axis that an index leaves
    /// behind after slicing, and to restore that axis before writing a value back.
    ///
    /// # Errors
    ///
    /// Forwards the error of a reshape that does not apply to `input`.
    fn reshape(&self, input: &Self::Value, shape: Shape) -> Result<Self::Value, ProgramError>;

    /// Returns the elements of `input` at the runtime position that `binding` supplies on `axis`, with that axis
    /// removed. A negative position counts from the end of `axis` once, and the result is then clamped to the valid
    /// range of `axis`.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if `input` does not have a static shape or `axis` is out of bounds for its rank, and
    /// forwards the errors of the operations that perform the selection.
    fn dynamic_index(
        &self,
        input: &Self::Value,
        axis: usize,
        binding: &Self::Binding,
    ) -> Result<Self::Value, ProgramError>;
}

/// A [`TransformReadCarrier`] that can also write a selected value back into its parent, which the traversals that
/// rebuild a root after a mutation need (e.g., [`reconstruct_in`](ArrayReferenceTransformPath::reconstruct_in)). Each
/// function of this trait inverts one read function of [`TransformReadCarrier`]: [`update_slice`](Self::update_slice)
/// inverts [`slice`](TransformReadCarrier::slice), and [`dynamic_update_index`](Self::dynamic_update_index) inverts
/// [`dynamic_index`](TransformReadCarrier::dynamic_index). It is a separate trait so that read-only traversals do not
/// require update capabilities from the carried values.
trait TransformWriteCarrier: TransformReadCarrier {
    /// Returns `target` with `update` written into the slice that starts at `starts`, which inverts
    /// [`slice`](TransformReadCarrier::slice). `update` has the shape of that slice, so the traversal
    /// restores the size-one axis of an index before calling this function.
    ///
    /// # Errors
    ///
    /// Forwards the error of an update that does not apply to `target`.
    fn update_slice(
        &self,
        target: &Self::Value,
        update: &Self::Value,
        starts: Vec<usize>,
    ) -> Result<Self::Value, ProgramError>;

    /// Returns `target` with `update` written at the runtime position that `binding` supplies on `axis`, which
    /// inverts [`dynamic_index`](TransformReadCarrier::dynamic_index). `update` has the shape of the value that
    /// `dynamic_index` selects (i.e., without `axis`), and the position is wrapped and clamped exactly as in
    /// `dynamic_index`, so that a write replaces the elements that the corresponding read selects.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if `target` does not have a static shape or `axis` is out of bounds for its rank,
    /// and forwards the errors of the operations that perform the update.
    fn dynamic_update_index(
        &self,
        target: &Self::Value,
        update: &Self::Value,
        axis: usize,
        binding: &Self::Binding,
    ) -> Result<Self::Value, ProgramError>;
}

/// Stateless eager carrier over one concrete array value family. Eager paths carry only static transforms,
/// so the dynamic-index hooks are unreachable by type.
struct EagerTransformCarrier<A>(PhantomData<A>);

impl<A: Value<Type = ArrayType> + Reshape + Slice> TransformReadCarrier for EagerTransformCarrier<A> {
    type Value = A;
    type Binding = NoReferenceTransformBinding;

    fn array_type<'c>(&'c self, value: &'c A) -> Result<Cow<'c, ArrayType>, ProgramError> {
        Ok(value.r#type())
    }

    fn slice(&self, input: &A, starts: Vec<usize>, limits: Vec<usize>) -> Result<A, ProgramError> {
        input.slice(starts.as_slice(), limits.as_slice(), &vec![1; starts.len()])
    }

    fn reshape(&self, input: &A, shape: Shape) -> Result<A, ProgramError> {
        input.reshape(shape)
    }

    fn dynamic_index(
        &self,
        _input: &A,
        _axis: usize,
        binding: &NoReferenceTransformBinding,
    ) -> Result<A, ProgramError> {
        match *binding {}
    }
}

impl<A: Value<Type = ArrayType> + Reshape + Slice + UpdateSlice> TransformWriteCarrier for EagerTransformCarrier<A> {
    fn update_slice(&self, target: &A, update: &A, starts: Vec<usize>) -> Result<A, ProgramError> {
        target.update_slice(update, starts.as_slice())
    }

    fn dynamic_update_index(
        &self,
        _target: &A,
        _update: &A,
        _axis: usize,
        binding: &NoReferenceTransformBinding,
    ) -> Result<A, ProgramError> {
        match *binding {}
    }
}

/// Transform carrier that binds the canonical slice, reshape, and update-slice operations of one array reference
/// transform path into a reference discharge context, sharing the single [`ArrayReferenceTransformPath`] traversal with
/// the eager value carrier, which keeps staged and eager reference semantics consistent. Dynamic indices arrive closed
/// over context values and select a size-one dynamic slice; updates restore the removed axis before replacing that
/// slice.
///
/// The carrier lifts each of these array operations into the context's operation family through that family's
/// [`OperationProjection<ArrayType>`](OperationProjection) member family. Any composite family that embeds the array
/// operations (e.g., one that derives `#[ryft(members(ArrayType))]`) therefore supports array reference discharge,
/// and core array IR and backend-owned supersets share one traversal without matching operation names.
struct ContextTransformCarrier<'c, C> {
    /// Context in which the slice, reshape, and update-slice operations are bound.
    context: &'c C,
}

impl<C: Context<Type = ArrayIrType>> TransformReadCarrier for ContextTransformCarrier<'_, C>
where
    C::Operation: OperationProjection<
            ArrayType,
            Projected: From<ReshapeOperation> + From<SliceOperation> + From<DynamicSliceOperation>,
        >,
{
    type Value = C::Value;
    type Binding = C::Value;

    #[inline]
    fn array_type<'c>(&'c self, value: &'c C::Value) -> Result<Cow<'c, ArrayType>, ProgramError> {
        match value.r#type() {
            Cow::Borrowed(r#type) => Ok(Cow::Borrowed(<&ArrayType>::try_from(r#type)?)),
            Cow::Owned(r#type) => Ok(Cow::Owned(<&ArrayType>::try_from(&r#type)?.clone())),
        }
    }

    #[inline]
    fn slice(&self, input: &C::Value, starts: Vec<usize>, limits: Vec<usize>) -> Result<C::Value, ProgramError> {
        self.context.bind_array(SliceOperation::new(starts, limits), &[input.clone()])
    }

    #[inline]
    fn reshape(&self, input: &C::Value, shape: Shape) -> Result<C::Value, ProgramError> {
        self.context.bind_array(ReshapeOperation::new(shape), &[input.clone()])
    }

    fn dynamic_index(&self, input: &C::Value, axis: usize, binding: &C::Value) -> Result<C::Value, ProgramError> {
        let input_type = self.array_type(input)?.into_owned();
        let mut sizes = ArrayReferenceTransform::indexed_shape(axis, &input_type)?.dimensions().to_vec();
        sizes[axis] = 1;

        // Unselected axes span their complete extent, so dynamic slicing clamps their start to zero. Reusing
        // the scalar index there avoids constructing redundant zero values in the context's value family.
        let mut inputs = vec![input.clone()];
        inputs.extend(std::iter::repeat_n(binding.clone(), sizes.len()));
        let selected = self.context.bind_array(DynamicSliceOperation::new(sizes), &inputs)?;
        self.reshape(&selected, input_type.without_dimension(axis)?.0.shape().clone())
    }
}

impl<C: Context<Type = ArrayIrType>> TransformWriteCarrier for ContextTransformCarrier<'_, C>
where
    C::Operation: OperationProjection<
            ArrayType,
            Projected: From<ReshapeOperation>
                           + From<SliceOperation>
                           + From<UpdateSliceOperation>
                           + From<DynamicSliceOperation>
                           + From<DynamicUpdateSliceOperation>,
        >,
{
    #[inline]
    fn update_slice(&self, target: &C::Value, update: &C::Value, starts: Vec<usize>) -> Result<C::Value, ProgramError> {
        self.context.bind_array(UpdateSliceOperation::new(starts), &[target.clone(), update.clone()])
    }

    fn dynamic_update_index(
        &self,
        target: &C::Value,
        update: &C::Value,
        axis: usize,
        binding: &C::Value,
    ) -> Result<C::Value, ProgramError> {
        let target_type = self.array_type(target)?.into_owned();
        let mut dimensions = ArrayReferenceTransform::indexed_shape(axis, &target_type)?.dimensions().to_vec();
        dimensions[axis] = 1;
        let rank = dimensions.len();
        let update = self.reshape(update, Shape::new(dimensions.into_iter().map(Dimension::Static).collect()))?;

        // Restore the indexed axis before writing back. Full-size axes clamp to zero just as in the read path,
        // while the selected axis uses the same runtime index and clamping extent as the original index transform.
        let mut inputs = vec![target.clone(), update];
        inputs.extend(std::iter::repeat_n(binding.clone(), rank));
        self.context.bind_array(DynamicUpdateSliceOperation::new(), &inputs)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::Array;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{DimensionBounds, DimensionVariable};
    use crate::arrays::types::memories::Memory;
    use crate::axes::{Axis, AxisError};
    use crate::captures::CaptureReference;
    use crate::contexts::EagerContext;
    use crate::operations::{
        ReferenceAddUpdate, ReferenceFreezeOperation, ReferenceNew, ReferenceRead, ReferenceReadOperation,
        ReferenceSwap, ReferenceWrite,
    };
    use crate::programs::{
        AtomId, Program, ProjectedValue, ReferenceCompletion, ReferenceReplacementPreparation, RegionId,
        ValueProjection,
    };
    use crate::tracing::{Trace, Tracer, TracingContext};

    use super::*;

    /// Array IR values that the eager and staged reference fixtures operate on.
    type TestValue = ArrayIrValue<Array>;

    /// Operation family of the array IR fixtures, which includes the reference operations.
    type TestOperation = ArrayIrOperation<Array>;

    /// Eager context over the array IR test values and operations.
    type TestEagerContext = EagerContext<TestValue, TestOperation>;

    /// Tracing context that stages array IR programs over the test values and operations.
    type TestContext = TracingContext<TestValue, TestOperation>;

    /// Tracer that stages array IR values into [`TestContext`].
    type TestTracer = Tracer<TestContext>;

    /// Eager view over an array IR reference value.
    type TestView = ReferenceView<TestValue, ArrayReferenceTransform, TestValue>;

    /// Array IR read operation over array reference transforms.
    type TestRead = ReferenceReadOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>;

    /// Returns the rendering of the program that `access` stages through [`ArrayReferenceDischarge`] over the
    /// composed view `root[1:3, 0:2][1]` of an `f32[3, 3]` allocation, given that allocation and an `f32[2]` value.
    fn render_composed_view_access(
        access: impl Fn(
            &TestContext,
            &TestTracer,
            TestTracer,
            &ArrayReferenceTransformPath<TestTracer>,
        ) -> Result<Vec<TestTracer>, ProgramError>,
    ) -> String {
        let alias = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Slice {
                axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 2, 1)],
            })
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) });
        let (_, staged): (_, Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>) =
            TestEagerContext::trace(
                |inputs: Vec<TestTracer>| {
                    let context = inputs[0].context().clone();
                    access(&context, &inputs[0], inputs[1].clone(), &alias)
                },
                vec![
                    ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3, 3])),
                    ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])),
                ],
            )
            .unwrap();
        staged.to_string()
    }

    #[test]
    fn test_array_reference_view_error() {
        for (error, message) in [
            (
                ArrayReferenceViewError::CannotFreezeView,
                "cannot freeze a reference view; freeze the root reference instead",
            ),
            (
                ArrayReferenceViewError::NotStorageRoot,
                "backend storage transactions require a root handle that uses the allocation's stored type identities",
            ),
            (
                ArrayReferenceViewError::DynamicTransformIndex,
                "eager reference handles carry only static transforms; dynamic indices are resolved by each access",
            ),
        ] {
            assert_eq!(error.to_string(), message);
        }
    }

    #[test]
    fn test_array_reference_new() {
        let root = ArrayReference::new(Array::vector(vec![1.0f32, 2.0]).unwrap());
        assert_eq!(root.r#type().as_ref(), &ReferenceType::new(ArrayType::new_static(DataType::F32, [2])));
        assert!(root.path().is_root());
        assert_eq!(root.read(), Ok(Array::vector(vec![1.0f32, 2.0]).unwrap()));
    }

    #[test]
    fn test_array_reference_id() {
        let root = ArrayReference::new(Array::vector(vec![1.0f32, 2.0]).unwrap());
        let view = root
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) })
            .unwrap();
        assert_eq!(root.id(), root.clone().id());
        assert_eq!(root.id(), view.id());
        assert_ne!(root.id(), ArrayReference::new(Array::vector(vec![1.0f32, 2.0]).unwrap()).id());
    }

    #[test]
    fn test_array_reference_path() {
        let root = ArrayReference::new(Array::vector(vec![1i32, 2, 3]).unwrap());
        assert!(root.path().is_root());
        let transform = ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] };
        let view = root.with_transform(transform.clone()).unwrap();
        assert_eq!(view.path().transforms().cloned().collect::<Vec<_>>(), vec![transform]);
    }

    #[test]
    fn test_array_reference_is_storage_root() {
        let root = ArrayReference::new(Array::vector(vec![1.0f32, 2.0]).unwrap());
        let view = root
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) })
            .unwrap();
        assert!(root.is_storage_root());
        assert!(!view.is_storage_root());
    }

    #[test]
    fn test_array_reference_lock_storage() {
        // A storage root locks its allocation and observes the stored value.
        let root = ArrayReference::new(Array::vector(vec![1.0f32, 2.0]).unwrap());
        let guard = root.lock_storage().unwrap();
        assert_eq!(guard.observe().unwrap().snapshot(), &Array::vector(vec![1.0f32, 2.0]).unwrap());
        drop(guard);

        // A view is not a storage root, so it is rejected before the allocation is locked.
        let view = root
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) })
            .unwrap();
        let error = view.lock_storage().err().unwrap();
        assert_eq!(error.downcast_custom::<ArrayReferenceViewError>(), Some(&ArrayReferenceViewError::NotStorageRoot));
        assert_eq!(
            error.to_string(),
            "backend storage transactions require a root handle that uses the allocation's stored type identities",
        );

        // An allocation that cannot be locked forwards its reference error.
        assert_eq!(root.freeze(), Ok(Array::vector(vec![1.0f32, 2.0]).unwrap()));
        assert_eq!(
            root.lock_storage().err().unwrap().downcast_custom::<ReferenceError>(),
            Some(&ReferenceError::Frozen),
        );
    }

    #[test]
    fn test_array_reference_with_transform() {
        let slice =
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 3, 1)] };
        let handle = ArrayReference::new(Array::matrix(3, 4, (1..=12).map(|value| value as f32).collect()).unwrap())
            .with_transform(slice)
            .unwrap();
        assert_eq!(handle.r#type().as_ref(), &ReferenceType::new(ArrayType::new_static(DataType::F32, [2, 3])));
        assert_eq!(handle.read(), Ok(Array::matrix(2, 3, vec![5.0f32, 6.0, 7.0, 9.0, 10.0, 11.0]).unwrap()));

        // Composition validates each appended transform against the preceding view's derived type, so an index that is
        // out of bounds for the view is rejected even though it exists in the root.
        assert_eq!(
            handle.with_transform(ArrayReferenceTransform::Index {
                axis: 0,
                index: ArrayReferenceTransformIndex::Static(2),
            }),
            Err(TypeError::invalid("reference index 2 on axis 0 is out of bounds for size 2").into()),
        );
    }

    #[test]
    fn test_array_reference_with_transform_rejects_dynamic_indices() {
        // An unresolved dynamic index cannot enter a static eager path. The access must supply its binding through
        // `with_transforms`, which resolves the index before extending the handle.
        let root = ArrayReference::new(Array::matrix(3, 4, (1..=12).map(|value| value as f32).collect()).unwrap());
        let error = root
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic })
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<ArrayReferenceViewError>(),
            Some(&ArrayReferenceViewError::DynamicTransformIndex),
        );
        assert_eq!(
            error.to_string(),
            "eager reference handles carry only static transforms; dynamic indices are resolved by each access",
        );
    }

    #[test]
    fn test_array_reference_with_transform_is_structural() {
        let root = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap());
        let guard = root.lock_storage().unwrap();
        let ReferenceReplacementPreparation::Prepared(prepared) = guard.prepare_replacement().unwrap() else {
            panic!("new reference unexpectedly has active read leases")
        };
        let transaction = prepared.begin(ReferenceCompletion::ready(Ok(())));

        // A view is pure structural metadata over a live reference, so composing one must never resolve its
        // submitted work. The reference is parked in its `Taken` state, where every value access is unavailable behind
        // this retained guard until replacement commit, and derivation still computes its exact referent type.
        let transform = ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] };
        let view = root.with_transform(transform).unwrap();
        assert_eq!(view.r#type().as_ref(), &ReferenceType::new(ArrayType::new_static(DataType::F32, [2])));

        // Poisoning the submitted mutation is terminal for the alias family, but further derivation remains structural
        // composition. The resulting handle reports the reference failure only when it attempts to access state.
        transaction.poison("submission failed");
        let poisoned = ReferenceError::ExecutionPoisoned { reason: "submission failed".to_string() };
        assert_eq!(root.read().unwrap_err().downcast_custom::<ReferenceError>(), Some(&poisoned));
        let composed = view
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) })
            .unwrap();
        assert_eq!(composed.r#type().as_ref(), &ReferenceType::new(ArrayType::scalar(DataType::F32)));
        assert_eq!(composed.read().unwrap_err().downcast_custom::<ReferenceError>(), Some(&poisoned));

        // Derivation stays structural after the allocation is frozen, so the frozen state surfaces only when the new
        // view accesses the allocation.
        let frozen = ArrayReference::new(Array::vector(vec![1.0f32, 2.0]).unwrap());
        assert_eq!(frozen.freeze(), Ok(Array::vector(vec![1.0f32, 2.0]).unwrap()));
        let frozen_view = frozen
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) })
            .unwrap();
        assert_eq!(frozen_view.read().unwrap_err().downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_array_reference_with_transforms() {
        let root = ArrayReference::new(Array::matrix(2, 3, vec![1i32, 2, 3, 4, 5, 6]).unwrap());
        let transforms = [
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ];

        // A negative index counts from the end of its axis once and an index past the end clamps to the last position,
        // so `[-1, 99]` resolves to the last element of the last row, and the resolved view shares the allocation.
        let selected = root
            .with_transforms(
                &transforms,
                &[
                    ArrayIrValue::Array(Array::scalar(-1i32).unwrap()),
                    ArrayIrValue::Array(Array::scalar(99i32).unwrap()),
                ],
            )
            .unwrap();
        assert_eq!(selected.id(), root.id());
        assert_eq!(selected.read(), Ok(Array::scalar(6i32).unwrap()));
        assert_eq!(selected.swap(Array::scalar(9i32).unwrap()), Ok(Array::scalar(6i32).unwrap()));
        assert_eq!(root.read(), Ok(Array::matrix(2, 3, vec![1i32, 2, 3, 4, 5, 9]).unwrap()));

        // An index that is still negative after wrapping clamps to the first position.
        let first = root
            .with_transforms(
                &transforms,
                &[
                    ArrayIrValue::Array(Array::scalar(-99i32).unwrap()),
                    ArrayIrValue::Array(Array::scalar(-99i32).unwrap()),
                ],
            )
            .unwrap();
        assert_eq!(first.read(), Ok(Array::scalar(1i32).unwrap()));

        // An empty access returns an equal handle.
        assert_eq!(root.with_transforms(&[], &[]), Ok(root.clone()));
    }

    #[test]
    fn test_array_reference_with_transforms_mixes_static_and_dynamic_transforms() {
        // Static transforms enter the path unchanged, while each dynamic index is resolved against the view that the
        // preceding transforms select, so `-1` names the last column of the slice rather than of the root.
        let root = ArrayReference::new(Array::matrix(2, 3, vec![1i32, 2, 3, 4, 5, 6]).unwrap());
        let slice =
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 2, 1), ArraySliceAxis::new(0, 2, 1)] };
        let view = root
            .with_transforms(
                &[
                    slice.clone(),
                    ArrayReferenceTransform::Index { axis: 1, index: ArrayReferenceTransformIndex::Dynamic },
                ],
                &[ArrayIrValue::Array(Array::scalar(-1i32).unwrap())],
            )
            .unwrap();
        assert_eq!(
            view.path().transforms().cloned().collect::<Vec<_>>(),
            vec![slice, ArrayReferenceTransform::Index { axis: 1, index: ArrayReferenceTransformIndex::Static(1) }],
        );
        assert_eq!(view.read(), Ok(Array::vector(vec![2i32, 5]).unwrap()));
    }

    #[test]
    fn test_array_reference_with_transforms_rejects_invalid_bindings() {
        let root = ArrayReference::new(Array::vector(vec![1i32, 2]).unwrap());
        let transforms = [ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];

        // Bindings must match the dynamic indices of the transforms exactly.
        assert_eq!(
            root.with_transforms(&transforms, &[]),
            Err(TypeError::invalid("reference transform requires 1 bindings but only 0 remain").into()),
        );
        assert_eq!(
            root.with_transforms(&[], &[ArrayIrValue::Array(Array::scalar(0i32).unwrap())]),
            Err(TypeError::invalid("reference transform path has 1 extra bindings").into()),
        );

        // A binding must be a scalar integer, and a dynamic index needs a non-empty axis to select from.
        assert_eq!(
            root.with_transforms(&transforms, &[ArrayIrValue::Array(Array::scalar(1f32).unwrap())]),
            Err(TypeError::invalid("reference transform requires a scalar integer index but received `f32[]`").into()),
        );
        let empty = ArrayReference::new(Array::vector(Vec::<i32>::new()).unwrap());
        assert_eq!(
            empty.with_transforms(&transforms, &[ArrayIrValue::Array(Array::scalar(0i32).unwrap())]),
            Err(TypeError::invalid("cannot dynamically index an empty reference axis").into()),
        );

        // Static transforms are validated against the view they apply to.
        assert_eq!(
            root.with_transforms(
                &[ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(5) }],
                &[],
            ),
            Err(TypeError::invalid("reference index 5 on axis 0 is out of bounds for size 2").into()),
        );
    }

    #[test]
    fn test_array_reference_read() {
        let root = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap());
        let view = root
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] })
            .unwrap();

        // Reading a view applies its path rather than exposing the complete allocation.
        assert_eq!(root.read(), Ok(Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap()));
        assert_eq!(view.read(), Ok(Array::vector(vec![2.0f32, 3.0]).unwrap()));
    }

    #[test]
    fn test_array_reference_swap() {
        // A root handle swaps the complete allocation.
        let root = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        assert_eq!(
            root.swap(Array::vector(vec![4.0f32, 5.0, 6.0]).unwrap()),
            Ok(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()),
        );

        // A view swaps only the elements that it selects and preserves the rest of the allocation.
        let view = root
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) })
            .unwrap();
        assert_eq!(view.swap(Array::scalar(7.0f32).unwrap()), Ok(Array::scalar(5.0f32).unwrap()));
        assert_eq!(root.read(), Ok(Array::vector(vec![4.0f32, 7.0, 6.0]).unwrap()));
    }

    #[test]
    fn test_array_reference_swap_rejects_wrong_referent_type() {
        let root = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap());
        let view = root
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 3, 1)] })
            .unwrap();

        // Reconstruction alone accepts smaller replacements, so the handle checks exact view type equality.
        let error = view.swap(Array::vector(vec![10.0f32, 20.0]).unwrap()).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceError>(),
            Some(&ReferenceError::ReferentTypeMismatch {
                expected: "f32[3]".to_string(),
                actual: "f32[2]".to_string(),
            }),
        );
        assert_eq!(root.read(), Ok(Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap()));

        // A frozen allocation reports its terminal state before checking a malformed replacement.
        assert_eq!(root.freeze(), Ok(Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap()));
        let error = view.swap(Array::vector(vec![1.0f32, 2.0]).unwrap()).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_array_reference_write() {
        // A root handle replaces the complete allocation.
        let root = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        assert_eq!(root.write(Array::vector(vec![4.0f32, 5.0, 6.0]).unwrap()), Ok(()));
        assert_eq!(root.read(), Ok(Array::vector(vec![4.0f32, 5.0, 6.0]).unwrap()));

        // A view replaces only the elements that it selects and preserves the rest of the allocation.
        let view = root
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) })
            .unwrap();
        assert_eq!(view.write(Array::scalar(7.0f32).unwrap()), Ok(()));
        assert_eq!(root.read(), Ok(Array::vector(vec![4.0f32, 7.0, 6.0]).unwrap()));
    }

    #[test]
    fn test_array_reference_write_rejects_wrong_referent_type() {
        let root = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap());
        let view = root
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 3, 1)] })
            .unwrap();

        // Reconstruction alone accepts smaller replacements, so the handle checks exact view type equality.
        let error = view.write(Array::vector(vec![10.0f32, 20.0]).unwrap()).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceError>(),
            Some(&ReferenceError::ReferentTypeMismatch {
                expected: "f32[3]".to_string(),
                actual: "f32[2]".to_string(),
            }),
        );
        assert_eq!(root.read(), Ok(Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap()));

        // A frozen allocation reports its terminal state before checking a malformed replacement.
        assert_eq!(root.freeze(), Ok(Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap()));
        let error = view.write(Array::vector(vec![1.0f32, 2.0]).unwrap()).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_array_reference_write_reconstructs_composed_transforms() {
        let root = ArrayReference::new(Array::matrix(3, 3, (1..=9).map(|value| value as f32).collect()).unwrap());
        let view = root
            .with_transform(ArrayReferenceTransform::Slice {
                axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 2, 1)],
            })
            .unwrap()
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) })
            .unwrap();
        assert_eq!(view.r#type().as_ref(), &ReferenceType::new(ArrayType::new_static(DataType::F32, [2])));

        // A write reconstructs both strict parents and preserves elements outside the composed view.
        assert_eq!(view.write(Array::vector(vec![70.0f32, 80.0]).unwrap()), Ok(()));
        assert_eq!(
            root.read(),
            Ok(Array::matrix(3, 3, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 70.0, 80.0, 9.0]).unwrap()),
        );
    }

    #[test]
    fn test_array_reference_add_update() {
        // A root handle adds into the complete allocation.
        let root = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        assert_eq!(root.add_update(&Array::vector(vec![1.0f32, 1.0, 1.0]).unwrap()), Ok(()));
        assert_eq!(root.read(), Ok(Array::vector(vec![2.0f32, 3.0, 4.0]).unwrap()));

        // A view adds into only the elements that it selects and preserves the rest of the allocation.
        let view = root
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) })
            .unwrap();
        assert_eq!(view.add_update(&Array::scalar(5.0f32).unwrap()), Ok(()));
        assert_eq!(root.read(), Ok(Array::vector(vec![2.0f32, 8.0, 4.0]).unwrap()));
    }

    #[test]
    fn test_array_reference_add_update_rejects_wrong_referent_type() {
        let root = ArrayReference::new(Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap());
        let view = root
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 3, 1)] })
            .unwrap();

        // The sum is checked against the view's referent type after the addition succeeds, so an update that promotes
        // the viewed elements is rejected and the allocation keeps its previous value.
        let error = view.add_update(&Array::vector(vec![1.0f64, 2.0, 3.0]).unwrap()).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceError>(),
            Some(&ReferenceError::ReferentTypeMismatch {
                expected: "f32[3]".to_string(),
                actual: "f64[3]".to_string(),
            }),
        );
        assert_eq!(root.read(), Ok(Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap()));

        // An addition whose inputs are incompatible fails before any referent type check.
        let error = view.add_update(&Array::vector(vec![1.0f32, 2.0]).unwrap()).unwrap_err();
        assert_eq!(error.to_string(), "failed to broadcast shape `[2]` to shape `[3]`");
        assert_eq!(root.read(), Ok(Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap()));

        // A frozen allocation reports its terminal state before attempting the addition.
        assert_eq!(root.freeze(), Ok(Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap()));
        let error = view.add_update(&Array::vector(vec![1.0f32, 2.0]).unwrap()).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_array_reference_freeze() {
        // A view cannot freeze its allocation, and rejecting it leaves the allocation available to the root.
        let root = ArrayReference::new(Array::vector(vec![1.0f32, 2.0]).unwrap());
        let view = root
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) })
            .unwrap();
        let error = view.freeze().unwrap_err();
        assert_eq!(
            error.downcast_custom::<ArrayReferenceViewError>(),
            Some(&ArrayReferenceViewError::CannotFreezeView),
        );
        assert_eq!(error.to_string(), "cannot freeze a reference view; freeze the root reference instead");

        // Freezing the root returns its final value and invalidates every handle that shares the allocation.
        assert_eq!(root.freeze(), Ok(Array::vector(vec![1.0f32, 2.0]).unwrap()));
        assert_eq!(view.read().unwrap_err().downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
        assert_eq!(root.freeze().unwrap_err().downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_array_reference_rename_type_identities() {
        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let target = DimensionVariable::new("target", bounds);
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source.clone())]));
        let reference = ArrayReference::new(CaptureReference::new(0, source_type.clone()));
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(source, target.clone()).unwrap();

        // A renamed handle is an alias of the same allocation with handle-local type identities, so it compares equal
        // to the original handle while no longer being a storage root, whose value must use the stored identities.
        let renamed = reference.rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed, reference);
        assert_eq!(renamed.id(), reference.id());
        assert_eq!(
            renamed.r#type().referent(),
            &ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(target)])),
        );
        assert_eq!(reference.r#type().referent(), &source_type);
        assert!(reference.is_storage_root());
        assert!(!renamed.is_storage_root());
        assert_eq!(
            renamed.lock_storage().err().unwrap().downcast_custom::<ArrayReferenceViewError>(),
            Some(&ArrayReferenceViewError::NotStorageRoot),
        );
    }

    #[test]
    fn test_array_reference_rename_type_identities_rejects_non_bijective_renamings() {
        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let second = DimensionVariable::new("second", bounds);
        let target = DimensionVariable::new("target", bounds);
        let reference = ArrayReference::new(CaptureReference::new(
            0,
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(source.clone()), Dimension::Dynamic(second.clone())]),
            ),
        ));

        // A renaming that merges two identities cannot reconstruct stored values, so it is rejected before any alias
        // exists.
        let mut merging = TypeIdentityRenaming::new();
        merging.insert(source.clone(), target.clone()).unwrap();
        merging.insert(second.clone(), target).unwrap();
        assert_eq!(
            reference.rename_type_identities(&merging),
            Err(TypeError::invalid("type identities `source` and `second` are both renamed to `target`")),
        );

        // The collision is reported in the caller's direction, so a handle that already carries a bijective
        // handle-local renaming names its own identities rather than the root identities behind them.
        let left = DimensionVariable::new("left", bounds);
        let right = DimensionVariable::new("right", bounds);
        let mut bijective = TypeIdentityRenaming::new();
        bijective.insert(source, left.clone()).unwrap();
        bijective.insert(second, right.clone()).unwrap();
        let renamed = reference.rename_type_identities(&bijective).unwrap();
        let merged = DimensionVariable::new("merged", bounds);
        let mut collapsing = TypeIdentityRenaming::new();
        collapsing.insert(left, merged.clone()).unwrap();
        collapsing.insert(right, merged).unwrap();
        assert_eq!(
            renamed.rename_type_identities(&collapsing),
            Err(TypeError::invalid("type identities `left` and `right` are both renamed to `merged`")),
        );
    }

    #[test]
    fn test_array_reference_clone() {
        // A clone shares the allocation, so a write through either handle is visible through the other.
        let root = ArrayReference::new(Array::vector(vec![1.0f32, 2.0]).unwrap());
        let alias = root.clone();
        assert_eq!(alias.id(), root.id());
        assert_eq!(alias.write(Array::vector(vec![3.0f32, 4.0]).unwrap()), Ok(()));
        assert_eq!(root.read(), Ok(Array::vector(vec![3.0f32, 4.0]).unwrap()));
    }

    #[test]
    fn test_array_reference_debug() {
        let root = ArrayReference::new(Array::vector(vec![1.0f32, 2.0]).unwrap());
        let view = root
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) })
            .unwrap();
        assert_eq!(format!("{root:?}"), format!("ArrayReference {{ id: {:?}, path: {:?} }}", root.id(), root.path()));
        assert_eq!(format!("{view:?}"), format!("ArrayReference {{ id: {:?}, path: {:?} }}", root.id(), view.path()));
    }

    #[test]
    fn test_array_reference_display() {
        let root = ArrayReference::new(Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        let view = root
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) })
            .unwrap();
        assert_eq!(root.to_string(), "ref<f32[2, 3]>");
        assert_eq!(view.to_string(), "ref<f32[3]>");
    }

    #[test]
    fn test_array_reference_eq_and_hash() {
        // Equality and hashing identify the allocation and the view, so clones are equal while separate allocations
        // with the same contents and different views of one allocation are not.
        let root = ArrayReference::new(Array::vector(vec![1.0f32, 2.0]).unwrap());
        let alias = root.clone();
        let separate = ArrayReference::new(Array::vector(vec![1.0f32, 2.0]).unwrap());
        let view = root
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) })
            .unwrap();
        assert_eq!(root, alias);
        assert_ne!(root, separate);
        assert_ne!(root, view);
        let references = HashMap::from([(root.clone(), "root"), (view.clone(), "view")]);
        assert_eq!(references.get(&alias), Some(&"root"));
        assert_eq!(references.get(&view), Some(&"view"));
        assert_eq!(references.get(&separate), None);
    }

    #[test]
    fn test_array_reference_type() {
        let root_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let root = ArrayReference::new(Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        let slice =
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 2, 1), ArraySliceAxis::new(1, 2, 1)] };
        let index = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) };
        let handle = root.with_transform(slice.clone()).unwrap().with_transform(index.clone()).unwrap();

        // Composition derives each handle type incrementally, which must agree with folding the complete mapping
        // over the root type in one pass.
        let path: ArrayReferenceTransformPath =
            ArrayReferenceTransformPath::root().with_transform(slice).with_transform(index);
        assert_eq!(root.r#type().as_ref(), &ReferenceType::new(root_type.clone()));
        assert_eq!(handle.r#type().as_ref(), &ReferenceType::new(path.output_type(&root_type).unwrap()));
        assert_eq!(handle.clone().r#type(), handle.r#type());
    }

    #[test]
    fn test_array_reference_transform_path_root_slice_axes() {
        let root_type = ArrayType::new_static(DataType::I32, vec![4, 5]);
        let path = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Slice {
                axes: vec![ArraySliceAxis::new(1, 3, 1), ArraySliceAxis::new(2, 3, 1)],
            })
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) });

        // A static path selects one range per root axis, in root coordinates and at the root's rank.
        assert_eq!(
            path.root_slice_axes(&root_type),
            Some(vec![ArraySliceAxis::new(2, 1, 1), ArraySliceAxis::new(2, 3, 1)]),
        );
        assert_eq!(
            ArrayReferenceTransformPath::root().root_slice_axes(&ArrayType::new_static(DataType::I32, vec![])),
            Some(vec![]),
        );

        // Dynamic indices, dynamically shaped roots, and paths that do not fold against the root have no static
        // selection.
        let dynamically_indexed = ArrayReferenceTransformPath::root().with_bound_transform(
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
            vec![ValueId::new(RegionId::new(0), AtomId::new(0))],
        );
        assert_eq!(dynamically_indexed.root_slice_axes(&root_type), None);
        let dynamically_shaped = ArrayType::new(
            DataType::I32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("length", DimensionBounds::unbounded()))]),
        );
        assert_eq!(ArrayReferenceTransformPath::root().root_slice_axes(&dynamically_shaped), None);
        let invalid = path
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(3) });
        assert_eq!(invalid.root_slice_axes(&root_type), None);
    }

    #[test]
    fn test_array_reference_transform_path_output_type() {
        let root_type = ArrayType::new_static(DataType::F32, [3, 4]);
        let root: ArrayReferenceTransformPath = ArrayReferenceTransformPath::root();
        assert_eq!(root.output_type(&root_type), Ok(root_type.clone()));

        // Each transform applies to the preceding view, so the slice narrows both axes and the index then removes
        // the leading axis of the already-narrowed view.
        let slice =
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 3, 1)] };
        let index = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) };
        let sliced = root.with_transform(slice);
        let indexed = sliced.clone().with_transform(index);
        assert_eq!(sliced.output_type(&root_type), Ok(ArrayType::new_static(DataType::F32, [2, 3])));
        assert_eq!(indexed.output_type(&root_type), Ok(ArrayType::new_static(DataType::F32, [3])));
    }

    #[test]
    fn test_array_reference_transform_path_intermediates_in() {
        let path: ArrayReferenceTransformPath<NoReferenceTransformBinding> = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] })
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) });
        let root = Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let carrier = EagerTransformCarrier::<Array>(PhantomData);
        assert_eq!(
            path.intermediates_in(&carrier, root.clone()),
            Ok(vec![root.clone(), Array::vector(vec![2.0f32, 3.0]).unwrap(), Array::scalar(3.0f32).unwrap()]),
        );
        assert_eq!(ArrayReferenceTransformPath::root().intermediates_in(&carrier, root.clone()), Ok(vec![root]));
    }

    #[test]
    fn test_array_reference_transform_path_reconstruct_in() {
        let path: ArrayReferenceTransformPath<NoReferenceTransformBinding> = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] })
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) });
        let carrier = EagerTransformCarrier::<Array>(PhantomData);

        // Reconstruction consumes strict parents in reverse order; the old selected scalar is unnecessary.
        assert_eq!(
            path.reconstruct_in(
                &carrier,
                &[Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap(), Array::vector(vec![2.0f32, 3.0]).unwrap()],
                Array::scalar(7.0f32).unwrap(),
            ),
            Ok(Array::vector(vec![1.0f32, 2.0, 7.0, 4.0]).unwrap()),
        );
        assert_eq!(
            ArrayReferenceTransformPath::root().reconstruct_in(&carrier, &[], Array::scalar(7.0f32).unwrap()),
            Ok(Array::scalar(7.0f32).unwrap()),
        );
    }

    #[test]
    fn test_array_reference_transform_path_reconstruct_in_rejects_invalid_parent_count() {
        let path: ArrayReferenceTransformPath<NoReferenceTransformBinding> = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) });
        let carrier = EagerTransformCarrier::<Array>(PhantomData);
        assert_eq!(
            path.reconstruct_in(&carrier, &[], Array::scalar(1.0f32).unwrap()),
            Err(ProgramError::MalformedProgram(
                "reference transform path reconstruction requires 1 parent snapshots but received 0".to_string(),
            )),
        );
        assert_eq!(
            path.reconstruct_in(
                &carrier,
                &[Array::vector(vec![1.0f32]).unwrap(), Array::scalar(1.0f32).unwrap()],
                Array::scalar(1.0f32).unwrap(),
            ),
            Err(ProgramError::MalformedProgram(
                "reference transform path reconstruction requires 1 parent snapshots but received 2".to_string(),
            )),
        );
    }

    #[test]
    fn test_array_reference_transform_path_apply_rejects_dynamic_indices() {
        // An eager path carries no bindings, so a dynamic index that reaches it has no position to select.
        let path: ArrayReferenceTransformPath<NoReferenceTransformBinding> = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic });
        assert_eq!(
            path.apply(Array::matrix(3, 4, (1..=12).map(|value| value as f32).collect()).unwrap()),
            Err(TypeError::invalid(
                "a dynamic index has no static selection; apply its binding at the reference access",
            )
            .into()),
        );
    }

    #[test]
    fn test_array_reference_transform_output_type() {
        let input = ArrayType::new_static(DataType::F32, [3, 4]);
        assert_eq!(
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) }
                .output_type(&input),
            Ok(ArrayType::new_static(DataType::F32, [4])),
        );
        assert_eq!(
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 3, 1)] }
                .output_type(&input),
            Ok(ArrayType::new_static(DataType::F32, [2, 3])),
        );
        // Empty selections remain valid array views and preserve rank.
        assert_eq!(
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(3, 0, 1), ArraySliceAxis::new(0, 4, 1)] }
                .output_type(&input),
            Ok(ArrayType::new_static(DataType::F32, [0, 4])),
        );
    }

    #[test]
    fn test_array_reference_transform_output_type_rejects_invalid_selections() {
        let matrix_type = ArrayType::new_static(DataType::F32, [3, 4]);
        let vector_type = ArrayType::new_static(DataType::F32, [3]);

        // Static indexing selects one existing index on one existing axis; a dynamic index still names an
        // existing axis.
        assert_eq!(
            ArrayReferenceTransform::Index { axis: 2, index: ArrayReferenceTransformIndex::Static(0) }
                .output_type(&matrix_type),
            Err(TypeError::invalid("reference index axis 2 is out of bounds for rank 2")),
        );
        assert_eq!(
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(3) }
                .output_type(&matrix_type),
            Err(TypeError::invalid("reference index 3 on axis 0 is out of bounds for size 3")),
        );
        assert_eq!(
            ArrayReferenceTransform::Index { axis: 2, index: ArrayReferenceTransformIndex::Dynamic }
                .output_type(&matrix_type),
            Err(TypeError::invalid("reference index axis 2 is out of bounds for rank 2")),
        );

        // Static slicing is rank-preserving, so it declares exactly one unit-stride selection per input axis and
        // stays inside every axis of the input.
        assert_eq!(
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 2, 1)] }.output_type(&matrix_type),
            Err(TypeError::invalid("reference slice has 1 axes but its input has rank 2")),
        );
        assert_eq!(
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 2, 2)] }.output_type(&vector_type),
            Err(TypeError::invalid(
                "reference slice axis 0 stride must be 1 until scatter-backed strided updates are supported",
            )),
        );
        assert_eq!(
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(2, 3, 1)] }.output_type(&vector_type),
            Err(TypeError::invalid("reference slice on axis 0 with start 2 and size 3 exceeds input size 3")),
        );

        // The exclusive limit is computed as `start + size`, so an unrepresentable limit is rejected before it can
        // wrap around into an apparently valid selection.
        assert_eq!(
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(usize::MAX, 1, 1)] }
                .output_type(&vector_type),
            Err(TypeError::invalid("reference slice limit overflows `usize` on axis 0")),
        );
    }

    #[test]
    fn test_array_reference_transform_output_type_rejects_dynamic_shapes() {
        let input = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("length", DimensionBounds::unbounded()))]),
        );
        assert_eq!(
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) }
                .output_type(&input),
            Err(TypeError::invalid(format!("reference indexing requires a static referent type but got `{input}`"))),
        );
        assert_eq!(
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 1, 1)] }.output_type(&input),
            Err(TypeError::invalid(format!("reference slicing requires a static referent type but got `{input}`"))),
        );
    }

    #[test]
    fn test_array_reference_transform_output_type_dynamic_index() {
        let matrix_type = ArrayType::new_static(DataType::F32, [3, 4]);
        let dynamic_index = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic };
        assert_eq!(dynamic_index.output_type(&matrix_type), Ok(ArrayType::new_static(DataType::F32, [4])));

        // A dynamic index removes its axis even when that axis is empty, where no static index could select it,
        // because its position is only known to the access that applies it.
        assert_eq!(
            dynamic_index.output_type(&ArrayType::new_static(DataType::F32, [0, 4])),
            Ok(ArrayType::new_static(DataType::F32, [4])),
        );
    }

    #[test]
    fn test_array_reference_transform_display() {
        assert_eq!(
            ArrayReferenceTransform::Index { axis: 1, index: ArrayReferenceTransformIndex::Static(2) }.to_string(),
            "index(axis=1, index=2)",
        );
        assert_eq!(
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }.to_string(),
            "index(axis=0, index=dynamic)",
        );
        assert_eq!(
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] }.to_string(),
            "slice(axes=[1:3])",
        );
        assert_eq!(
            ArrayReferenceTransform::Slice {
                axes: vec![ArraySliceAxis::new(0, 4, 1), ArraySliceAxis::new(1, 3, 2), ArraySliceAxis::new(2, 0, 1)],
            }
            .to_string(),
            "slice(axes=[0:4, 1:6:2, 2:2])",
        );
    }

    #[test]
    fn test_array_reference_transform_binding_count() {
        assert_eq!(
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(2) }.binding_count(),
            0,
        );
        assert_eq!(
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }.binding_count(),
            1,
        );
        assert_eq!(ArrayReferenceTransform::Slice { axes: vec![] }.binding_count(), 0);
    }

    #[test]
    fn test_array_reference_transform_validate_bindings() {
        let transform = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic };
        let input = ArrayType::new_static(DataType::F32, [3]);
        let integer = ArrayIrType::Array(ArrayType::scalar(DataType::I32));
        let floating = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let vector = ArrayIrType::Array(ArrayType::new_static(DataType::I32, [1]));
        let host = ArrayIrType::Array(ArrayType::scalar(DataType::I32).with_memory(Memory::Host { pinned: false }));
        assert_eq!(transform.validate_bindings(&input, &[&integer]), Ok(()));
        assert_eq!(transform.validate_bindings(&input, &[]), Err(TypeError::invalid("expected 1 binding but got 0")));
        assert_eq!(
            transform.validate_bindings(&input, &[&floating]),
            Err(TypeError::invalid("reference transform requires a scalar integer index but received `f32[]`")),
        );
        assert_eq!(
            transform.validate_bindings(&input, &[&vector]),
            Err(TypeError::invalid("reference transform requires a scalar integer index but received `i32[1]`")),
        );
        assert_eq!(
            transform.validate_bindings(&input, &[&host]),
            Err(TypeError::invalid(
                "reference transform and index must share one memory space but index resides in Host[Unpinned] \
                 and reference resides in Device",
            )),
        );
    }

    #[test]
    fn test_array_reference_transform_read_type() {
        // Read-only accesses derive the same types as `output_type`; they only skip its write-back check.
        let input = ArrayType::new_static(DataType::F32, [3, 4]);
        assert_eq!(
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) }
                .read_type(&input),
            Ok(ArrayType::new_static(DataType::F32, [4])),
        );
        assert_eq!(
            ArrayReferenceTransform::Index { axis: 1, index: ArrayReferenceTransformIndex::Dynamic }.read_type(&input),
            Ok(ArrayType::new_static(DataType::F32, [3])),
        );
        assert_eq!(
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 3, 1)] }
                .read_type(&input),
            Ok(ArrayType::new_static(DataType::F32, [2, 3])),
        );
        assert_eq!(
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(3) }
                .read_type(&input),
            Err(TypeError::invalid("reference index 3 on axis 0 is out of bounds for size 3")),
        );
    }

    #[test]
    fn test_array_reference_transform_overlap() {
        let root = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [4, 3])));
        let empty: ArrayReferenceTransformPath = ArrayReferenceTransformPath::root();
        let rows_0_1 = empty.clone().with_transform(ArrayReferenceTransform::Slice {
            axes: vec![ArraySliceAxis::new(0, 2, 1), ArraySliceAxis::new(0, 3, 1)],
        });
        let rows_1_2 = empty.clone().with_transform(ArrayReferenceTransform::Slice {
            axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 3, 1)],
        });
        let rows_2_3 = empty.clone().with_transform(ArrayReferenceTransform::Slice {
            axes: vec![ArraySliceAxis::new(2, 2, 1), ArraySliceAxis::new(0, 3, 1)],
        });
        let row_1 = empty
            .clone()
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) });
        let column_0 = empty
            .clone()
            .with_transform(ArrayReferenceTransform::Index { axis: 1, index: ArrayReferenceTransformIndex::Static(0) });

        // Static transforms fold to one static range per root axis: disjoint ranges on any axis make the paths
        // disjoint, identical ranges on every axis make them the same, and intersecting ranges may overlap. The trait
        // function and the path method agree.
        assert_eq!(
            ArrayReferenceTransform::overlap(&root, rows_0_1.bound_transforms(), rows_2_3.bound_transforms()),
            ReferenceViewOverlap::Disjoint,
        );
        assert_eq!(rows_0_1.overlap(&rows_2_3, &root), ReferenceViewOverlap::Disjoint);
        assert_eq!(row_1.overlap(&row_1, &root), ReferenceViewOverlap::Same);
        assert_eq!(rows_0_1.overlap(&rows_1_2, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(row_1.overlap(&rows_0_1, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(row_1.overlap(&rows_2_3, &root), ReferenceViewOverlap::Disjoint);

        // Rank changes are tracked while folding: an index removes its axis, so a slice that follows it addresses the
        // remaining root axes, and different transform sequences that select the same indices are the same.
        let row_1_columns_1_2 = row_1
            .clone()
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] });
        let row_1_column_1 = row_1
            .clone()
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) });
        let rows_1_columns_1_2_row_0 = empty
            .clone()
            .with_transform(ArrayReferenceTransform::Slice {
                axes: vec![ArraySliceAxis::new(1, 1, 1), ArraySliceAxis::new(1, 2, 1)],
            })
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) });
        assert_eq!(row_1_columns_1_2.overlap(&column_0, &root), ReferenceViewOverlap::Disjoint);
        assert_eq!(row_1_columns_1_2.overlap(&row_1, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(row_1_columns_1_2.overlap(&row_1_column_1, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(row_1_columns_1_2.overlap(&rows_1_columns_1_2_row_0, &root), ReferenceViewOverlap::Same);
        assert_eq!(row_1_column_1.overlap(&column_0, &root), ReferenceViewOverlap::Disjoint);

        // The complete root is the same as itself and as a slice spanning every axis, and may overlap with any
        // narrowing path.
        let complete = empty.clone().with_transform(ArrayReferenceTransform::Slice {
            axes: vec![ArraySliceAxis::new(0, 4, 1), ArraySliceAxis::new(0, 3, 1)],
        });
        assert_eq!(empty.overlap(&empty, &root), ReferenceViewOverlap::Same);
        assert_eq!(empty.overlap(&complete, &root), ReferenceViewOverlap::Same);
        assert_eq!(empty.overlap(&rows_0_1, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(empty.overlap(&row_1_column_1, &root), ReferenceViewOverlap::MayOverlap);

        // Dynamic indices agree only when their binding, offset, and clamping extent agree. Different
        // offsets can clamp to the same root element, so they cannot establish disjointness.
        let dynamic_index = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic };
        let first = ValueId::new(RegionId::new(0), AtomId::new(1));
        let second = ValueId::new(RegionId::new(0), AtomId::new(2));
        let other_region = ValueId::new(RegionId::new(1), AtomId::new(1));
        let row_first = empty.clone().with_bound_transform(dynamic_index.clone(), vec![first]);
        let row_second = empty.clone().with_bound_transform(dynamic_index.clone(), vec![second]);
        let row_other_region = empty.clone().with_bound_transform(dynamic_index.clone(), vec![other_region]);
        let shifted_row_first = rows_1_2.with_bound_transform(dynamic_index.clone(), vec![first]);
        assert_eq!(
            row_first.overlap(&empty.clone().with_bound_transform(dynamic_index.clone(), vec![first]), &root),
            ReferenceViewOverlap::Same,
        );
        assert_eq!(row_first.overlap(&row_second, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(row_first.overlap(&row_other_region, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(row_first.overlap(&row_1, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(row_first.overlap(&rows_2_3, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(row_first.overlap(&empty, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(row_first.overlap(&shifted_row_first, &root), ReferenceViewOverlap::MayOverlap);
        // Equal offsets with different extents also clamp differently: a large index selects row 3 in the
        // whole root but row 1 in its first two rows.
        let shortened_row_first = rows_0_1.clone().with_bound_transform(dynamic_index.clone(), vec![first]);
        assert_eq!(row_first.overlap(&shortened_row_first, &root), ReferenceViewOverlap::MayOverlap);

        // Folding continues past a dynamic index, so later transforms address the remaining root axes, and disjoint
        // selections on those axes make the paths disjoint whatever the dynamic indices select.
        let row_first_column_0 = row_first
            .clone()
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 1, 1)] });
        let row_first_columns_1_2 = row_first
            .clone()
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] });
        assert_eq!(row_first_column_0.overlap(&row_first_column_0, &root), ReferenceViewOverlap::Same);
        assert_eq!(row_first_column_0.overlap(&row_first_columns_1_2, &root), ReferenceViewOverlap::Disjoint);
        assert_eq!(
            row_first
                .with_transform(ArrayReferenceTransform::Index {
                    axis: 0,
                    index: ArrayReferenceTransformIndex::Static(0),
                })
                .overlap(
                    &row_second.with_transform(ArrayReferenceTransform::Index {
                        axis: 0,
                        index: ArrayReferenceTransformIndex::Static(2),
                    }),
                    &root,
                ),
            ReferenceViewOverlap::Disjoint,
        );

        // A path or root that cannot be folded (an out-of-bounds axis or index, a dynamic index without its
        // binding, a non-reference root, or a root without a static shape) is conservatively reported as possibly
        // overlapping rather than failing.
        let out_of_bounds = empty
            .clone()
            .with_transform(ArrayReferenceTransform::Index { axis: 2, index: ArrayReferenceTransformIndex::Static(0) });
        let unbound = empty.clone().with_transform(dynamic_index);
        assert_eq!(out_of_bounds.overlap(&rows_2_3, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(
            empty
                .with_transform(ArrayReferenceTransform::Index {
                    axis: 0,
                    index: ArrayReferenceTransformIndex::Static(4),
                })
                .overlap(&rows_2_3, &root),
            ReferenceViewOverlap::MayOverlap,
        );
        assert_eq!(unbound.overlap(&rows_2_3, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(
            rows_0_1.overlap(&rows_2_3, &ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4, 3]))),
            ReferenceViewOverlap::MayOverlap,
        );
        let dynamic = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("rows", DimensionBounds::unbounded())),
                Dimension::Static(3),
            ]),
        );
        assert_eq!(
            rows_0_1.overlap(&rows_2_3, &ArrayIrType::Reference(ReferenceType::new(dynamic))),
            ReferenceViewOverlap::MayOverlap,
        );
    }

    #[test]
    fn test_array_reference_transform_overlap_is_conservative_for_overflowing_coordinates() {
        // Malformed relative indices or slice starts cannot wrap around to become valid root coordinates, so a path
        // whose root coordinates overflow may overlap with anything.
        let root = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])));
        let tail: ArrayReferenceTransformPath = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] });
        let overflowing_index = tail.clone().with_transform(ArrayReferenceTransform::Index {
            axis: 0,
            index: ArrayReferenceTransformIndex::Static(usize::MAX),
        });
        let overflowing_slice =
            tail.with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(usize::MAX, 1, 1)] });
        assert_eq!(
            overflowing_index.overlap(&ArrayReferenceTransformPath::root(), &root),
            ReferenceViewOverlap::MayOverlap,
        );
        assert_eq!(
            overflowing_slice.overlap(&ArrayReferenceTransformPath::root(), &root),
            ReferenceViewOverlap::MayOverlap,
        );
    }

    #[test]
    fn test_array_reference_transform_batch() {
        let packed = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2, 3, 4])));
        let index = ArrayReferenceTransform::Index { axis: 1, index: ArrayReferenceTransformIndex::Static(2) };
        let slice =
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 1, 1), ArraySliceAxis::new(1, 2, 1)] };

        // A batch axis at or before the indexed axis shifts the packed indexed axis one position later and the output
        // keeps the batch axis, while a batch axis after the indexed axis leaves the packed indexed axis alone and the
        // output batch axis moves one position earlier. Negative batch axes normalize against the packed rank.
        assert_eq!(
            index.batch(&packed, BatchAxis::new(0)),
            Ok((
                ArrayReferenceTransform::Index { axis: 2, index: ArrayReferenceTransformIndex::Static(2) },
                BatchAxis::new(0),
            )),
        );
        assert_eq!(
            index.batch(&packed, BatchAxis::new(1)),
            Ok((
                ArrayReferenceTransform::Index { axis: 2, index: ArrayReferenceTransformIndex::Static(2) },
                BatchAxis::new(1),
            )),
        );
        assert_eq!(
            index.batch(&packed, BatchAxis::new(2)),
            Ok((
                ArrayReferenceTransform::Index { axis: 1, index: ArrayReferenceTransformIndex::Static(2) },
                BatchAxis::new(1),
            )),
        );
        assert_eq!(
            index.batch(&packed, BatchAxis::new(-1)),
            Ok((
                ArrayReferenceTransform::Index { axis: 1, index: ArrayReferenceTransformIndex::Static(2) },
                BatchAxis::new(1),
            )),
        );

        // Batching preserves the binding that supplies the index.
        let dynamic_index = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic };
        assert_eq!(
            dynamic_index.batch(&packed, BatchAxis::new(0)),
            Ok((
                ArrayReferenceTransform::Index { axis: 1, index: ArrayReferenceTransformIndex::Dynamic },
                BatchAxis::new(0),
            )),
        );

        // Slicing inserts the complete batch axis at the batch axis position and keeps the batch axis.
        assert_eq!(
            slice.batch(&packed, BatchAxis::new(1)),
            Ok((
                ArrayReferenceTransform::Slice {
                    axes: vec![
                        ArraySliceAxis::new(1, 1, 1),
                        ArraySliceAxis::new(0, 3, 1),
                        ArraySliceAxis::new(1, 2, 1),
                    ],
                },
                BatchAxis::new(1),
            )),
        );

        // A replicated source leaves both transforms unchanged and replicated.
        assert_eq!(index.batch(&packed, BatchAxis::replicated()), Ok((index.clone(), BatchAxis::replicated())));
        assert_eq!(slice.batch(&packed, BatchAxis::replicated()), Ok((slice.clone(), BatchAxis::replicated())));
    }

    #[test]
    fn test_array_reference_transform_batch_rejects_invalid_axes() {
        let packed = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2, 3, 4])));
        assert_eq!(
            ArrayReferenceTransform::Index { axis: 2, index: ArrayReferenceTransformIndex::Static(0) }
                .batch(&packed, BatchAxis::new(0)),
            Err(TypeError::invalid("reference index axis 2 is out of bounds for rank 2").into()),
        );
        // The indexed axis is validated before it is shifted past the batch axis, so a maximal axis cannot overflow.
        assert_eq!(
            ArrayReferenceTransform::Index { axis: usize::MAX, index: ArrayReferenceTransformIndex::Static(0) }
                .batch(&packed, BatchAxis::new(0)),
            Err(TypeError::invalid(format!("reference index axis {} is out of bounds for rank 2", usize::MAX)).into()),
        );
        // Inserting the batch selection requires exactly one selection per unbatched input axis.
        assert_eq!(
            ArrayReferenceTransform::Slice { axes: Vec::new() }.batch(&packed, BatchAxis::new(2)),
            Err(TypeError::invalid("reference slice has 0 axes but its input has rank 2").into()),
        );
        assert_eq!(
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 1, 1); 3] }
                .batch(&packed, BatchAxis::new(2)),
            Err(TypeError::invalid("reference slice has 3 axes but its input has rank 2").into()),
        );
    }

    #[test]
    fn test_array_reference_transform_batch_rejects_invalid_sources() {
        // The source must be a reference whose packed referent has the batch axis, and a static identity slice cannot
        // span a dynamically sized batch axis.
        let index = ArrayReferenceTransform::Index { axis: 1, index: ArrayReferenceTransformIndex::Static(2) };
        assert_eq!(
            index.batch(&ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2, 3, 4])), BatchAxis::new(0)),
            Err(BatchingError::Type(TypeError::invalid("expected reference type but got array type"))),
        );
        let packed = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2, 3, 4])));
        assert_eq!(
            index.batch(&packed, BatchAxis::new(3)),
            Err(BatchingError::Axis(AxisError::OutOfBounds { axis: Axis::from(3), rank: 3 })),
        );
        let batch = DimensionVariable::new("batch", DimensionBounds::unbounded());
        let dynamic = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(3)]));
        assert_eq!(
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 3, 1)] }
                .batch(&ArrayIrType::Reference(ReferenceType::new(dynamic.clone())), BatchAxis::new(0)),
            Err(BatchingError::DynamicBatchAxis { r#type: Box::new(dynamic), axis: Axis::from(0) }),
        );
    }

    #[test]
    fn test_reference_view_index() {
        let allocation = TestValue::Array(Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap())
            .reference_new()
            .unwrap();
        let row = TestView::new(allocation.clone()).unwrap().index(0, 1).unwrap();
        assert_eq!(row.read(), Ok(TestValue::Array(Array::vector(vec![4.0f32, 5.0, 6.0]).unwrap())));

        // The axis and the static index are validated against the viewed referent when the view is constructed.
        assert_eq!(
            TestView::new(allocation.clone()).unwrap().index(2, 0),
            Err(TypeError::invalid("reference index axis 2 is out of bounds for rank 2").into()),
        );
        assert_eq!(
            TestView::new(allocation).unwrap().index(0, 2),
            Err(TypeError::invalid("reference index 2 on axis 0 is out of bounds for size 2").into()),
        );
    }

    #[test]
    fn test_reference_view_index_reconstructs_removed_axis() {
        // Writing through an index restores the removed axis before updating the root, so mutations replace exactly
        // the indexed row and preserve every other row.
        let allocation = TestValue::Array(Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap())
            .reference_new()
            .unwrap();
        let row = TestView::new(allocation.clone()).unwrap().index(0, 1).unwrap();
        assert_eq!(
            row.swap(&TestValue::Array(Array::vector(vec![10.0f32, 20.0, 30.0]).unwrap())),
            Ok(TestValue::Array(Array::vector(vec![4.0f32, 5.0, 6.0]).unwrap())),
        );
        assert_eq!(row.add_update(&TestValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap())), Ok(()));
        assert_eq!(
            allocation.read(),
            Ok(TestValue::Array(Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, 11.0, 22.0, 33.0]).unwrap())),
        );
    }

    #[test]
    fn test_reference_view_dynamic_index() {
        // A dynamic index is resolved at the access against the view that the preceding transforms select, so `-1`
        // selects the last element of the sliced row rather than of the root.
        let root = TestValue::Reference(ArrayReference::new(Array::matrix(2, 3, vec![1i32, 2, 3, 4, 5, 6]).unwrap()));
        let selected = TestView::new(root)
            .unwrap()
            .index(0, 1)
            .unwrap()
            .slice(&[ArraySliceAxis::new(1, 2, 1)])
            .unwrap()
            .dynamic_index(0, &TestValue::Array(Array::scalar(-1i32).unwrap()))
            .unwrap();
        assert_eq!(selected.read(), Ok(TestValue::Array(Array::scalar(6i32).unwrap())));
    }

    #[test]
    fn test_reference_view_dynamic_index_rejects_invalid_indices() {
        let allocation = TestValue::Array(Array::vector(vec![1i32, 2, 3]).unwrap()).reference_new().unwrap();

        // Only the type of the index is known when the view is constructed, and it must be a scalar integer in the
        // viewed referent's memory space.
        assert_eq!(
            TestView::new(allocation.clone())
                .unwrap()
                .dynamic_index(0, &TestValue::Array(Array::scalar(1.0f32).unwrap())),
            Err(TypeError::invalid("reference transform requires a scalar integer index but received `f32[]`").into()),
        );
        assert_eq!(
            TestView::new(allocation.clone())
                .unwrap()
                .dynamic_index(0, &TestValue::Array(Array::vector(vec![1i32]).unwrap())),
            Err(TypeError::invalid("reference transform requires a scalar integer index but received `i32[1]`").into()),
        );
        let host_index = Array::with_unchecked_type(
            ArrayType::scalar(DataType::I32).with_memory(Memory::Host { pinned: false }),
            0i32.to_le_bytes().to_vec(),
        );
        assert_eq!(
            TestView::new(allocation).unwrap().dynamic_index(0, &TestValue::Array(host_index)),
            Err(TypeError::invalid(
                "reference transform and index must share one memory space but index resides in Host[Unpinned] \
                 and reference resides in Device",
            )
            .into()),
        );

        // An empty axis has no position to select, which only the access that applies the index can report.
        let empty = TestValue::Array(Array::vector(Vec::<i32>::new()).unwrap()).reference_new().unwrap();
        let view = TestView::new(empty)
            .unwrap()
            .dynamic_index(0, &TestValue::Array(Array::scalar(0i32).unwrap()))
            .unwrap();
        assert_eq!(view.read(), Err(TypeError::invalid("cannot dynamically index an empty reference axis").into()));
    }

    #[test]
    fn test_reference_view_dynamic_index_stages_projected_roots() {
        // A view over a projected tracer keeps the projected value types through every transform, stages each access
        // as one reference operation that carries the complete path and its binding, and interprets and discharges to
        // the same result.
        let (output_type, program) = TestContext::trace(
            |(input, index): (TestTracer, TestTracer)| {
                let input = <TestTracer as ValueProjection<ArrayType>>::into_projected(input)?;
                let reference = input.reference_new()?;
                let sliced = ReferenceView::<_, ArrayReferenceTransform, TestTracer>::new(reference)?
                    .slice(&[ArraySliceAxis::new(1, 2, 1)])?;
                let _: &ReferenceView<
                    ProjectedValue<ReferenceType<ArrayType>, TestTracer>,
                    ArrayReferenceTransform,
                    TestTracer,
                > = &sliced;
                let viewed = sliced.dynamic_index(0, &index)?;
                let selected: ProjectedValue<ArrayType, TestTracer> = viewed.read()?;
                viewed.write(&selected)?;
                viewed.add_update(&selected)?;
                let result: ProjectedValue<ArrayType, TestTracer> = viewed.read()?;
                Ok(result.into_value())
            },
            (
                ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])),
                ArrayIrType::Array(ArrayType::scalar(DataType::I32)),
            ),
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        assert_eq!(
            program.instructions().iter().map(|instruction| instruction.operation().name()).collect::<Vec<_>>(),
            vec!["reference_new", "reference_read", "reference_write", "reference_add_update", "reference_read"],
        );
        let read = &program.instructions()[1];
        assert_eq!(read.inputs().len(), 2);
        assert_eq!(read.operation().reference_access_descriptor(0).unwrap().transforms().len(), 2);
        let inputs = (
            TestValue::Array(Array::vector(vec![2.0f32, 3.0, 5.0]).unwrap()),
            TestValue::Array(Array::scalar(-1i32).unwrap()),
        );
        let expected = TestValue::Array(Array::scalar(10.0f32).unwrap());
        assert_eq!(program.clone().interpret(inputs.clone()), Ok(expected.clone()));
        let discharged = program.into_flat_program().discharge_references(0).unwrap();
        assert_eq!(discharged.program().interpret(vec![inputs.0, inputs.1]), Ok(vec![expected]));
    }

    #[test]
    fn test_reference_view_slice() {
        let allocation = TestValue::Array(Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap())
            .reference_new()
            .unwrap();
        let slice = TestView::new(allocation)
            .unwrap()
            .slice(&[ArraySliceAxis::new(0, 2, 1), ArraySliceAxis::new(1, 2, 1)])
            .unwrap();
        assert_eq!(slice.read(), Ok(TestValue::Array(Array::matrix(2, 2, vec![2.0f32, 3.0, 5.0, 6.0]).unwrap())));

        // Later transforms apply to the sliced view, so indexing its second row selects the second row of the slice.
        assert_eq!(slice.index(0, 1).unwrap().read(), Ok(TestValue::Array(Array::vector(vec![5.0f32, 6.0]).unwrap())));
    }

    #[test]
    fn test_reference_view_slice_shares_overlapping_allocation_state() {
        // Overlapping views address one allocation, so each observes the other's mutations of the shared elements.
        let allocation = TestValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap()).reference_new().unwrap();
        let left = TestView::new(allocation.clone()).unwrap().slice(&[ArraySliceAxis::new(0, 3, 1)]).unwrap();
        let right = TestView::new(allocation.clone()).unwrap().slice(&[ArraySliceAxis::new(1, 3, 1)]).unwrap();
        assert_eq!(
            left.swap(&TestValue::Array(Array::vector(vec![10.0f32, 20.0, 30.0]).unwrap())),
            Ok(TestValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap())),
        );
        assert_eq!(right.read(), Ok(TestValue::Array(Array::vector(vec![20.0f32, 30.0, 4.0]).unwrap())));
        assert_eq!(right.add_update(&TestValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap())), Ok(()));
        assert_eq!(allocation.read(), Ok(TestValue::Array(Array::vector(vec![10.0f32, 21.0, 32.0, 7.0]).unwrap())));
    }

    #[test]
    fn test_reference_view_slice_rejects_invalid_axes() {
        // Slices are validated against the viewed referent when the view is constructed: every range must stay inside
        // its axis and use a unit stride.
        let allocation = TestValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap()).reference_new().unwrap();
        assert_eq!(
            TestView::new(allocation.clone()).unwrap().slice(&[ArraySliceAxis::new(2, 2, 1)]),
            Err(TypeError::invalid("reference slice on axis 0 with start 2 and size 2 exceeds input size 3").into()),
        );
        assert_eq!(
            TestView::new(allocation).unwrap().slice(&[ArraySliceAxis::new(0, 2, 2)]),
            Err(TypeError::invalid(
                "reference slice axis 0 stride must be 1 until scatter-backed strided updates are supported",
            )
            .into()),
        );
    }

    #[test]
    fn test_array_reference_discharge_storage_alias() {
        let alias = <ArrayReferenceDischarge as ReferenceDischargePolicy<TestContext>>::storage_alias(
            &ArrayType::new_static(DataType::F32, [3]),
        );
        assert_eq!(alias, ArrayReferenceTransformPath::root());
    }

    #[test]
    fn test_array_reference_discharge_apply_transforms() {
        // Access transforms are appended to the alias with the context values that bind their dynamic indices.
        let context = TestEagerContext::new();
        let index = TestValue::Array(Array::scalar(1i32).unwrap());
        let alias = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] });
        let transform = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic };
        assert_eq!(
            ArrayReferenceDischarge::apply_transforms(&context, &alias, &[transform.clone()], &[index.clone()]),
            Ok(alias.with_bound_transform(transform, vec![index])),
        );
    }

    #[test]
    fn test_array_reference_discharge_apply_transforms_rejects_mismatched_bindings() {
        let context = TestEagerContext::new();
        let alias = ArrayReferenceTransformPath::root();
        let transform = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic };
        assert_eq!(
            ArrayReferenceDischarge::apply_transforms(&context, &alias, &[transform], &[]),
            Err(ProgramError::MalformedProgram(
                "reference transform requires 1 bindings but only 0 remain".to_string()
            )),
        );
        assert_eq!(
            ArrayReferenceDischarge::apply_transforms(
                &context,
                &alias,
                &[],
                &[TestValue::Array(Array::scalar(1i32).unwrap())],
            ),
            Err(ProgramError::MalformedProgram("reference transform path has 1 extra bindings".to_string())),
        );
    }

    #[test]
    fn test_array_reference_discharge_read() {
        let context = TestEagerContext::new();
        let current = TestValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let alias = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] });
        assert_eq!(
            ArrayReferenceDischarge::read(&context, &current, &alias),
            Ok(TestValue::Array(Array::vector(vec![2.0f32, 3.0]).unwrap())),
        );
        assert_eq!(
            ArrayReferenceDischarge::read(&context, &current, &ArrayReferenceTransformPath::root()),
            Ok(current)
        );
    }

    #[test]
    fn test_array_reference_discharge_read_stages_composed_views() {
        // A read materializes the allocation-to-leaf chain against the state it observes.
        assert_eq!(
            render_composed_view_access(|context, current, _, alias| {
                Ok(vec![ArrayReferenceDischarge::read(context, current, alias)?])
            }),
            indoc! {"
                lambda %0:f32[3, 3], %1:f32[2] .
                let %2:f32[2, 2] = slice [start_indices=[1, 0], limit_indices=[3, 2]] %0
                    %3:f32[1, 2] = slice [start_indices=[1, 0], limit_indices=[2, 2]] %2
                    %4:f32[2] = reshape [shape=[2]] %3
                in (%4)"},
        );
    }

    #[test]
    fn test_array_reference_discharge_write() {
        let context = TestEagerContext::new();
        let current = TestValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let alias = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] });
        assert_eq!(
            ArrayReferenceDischarge::write(
                &context,
                &current,
                TestValue::Array(Array::vector(vec![4.0f32, 5.0]).unwrap()),
                &alias,
            ),
            Ok(TestValue::Array(Array::vector(vec![1.0f32, 4.0, 5.0]).unwrap())),
        );

        // Writing through the root alias replaces the complete value.
        let replacement = TestValue::Array(Array::vector(vec![4.0f32, 5.0, 6.0]).unwrap());
        assert_eq!(
            ArrayReferenceDischarge::write(
                &context,
                &current,
                replacement.clone(),
                &ArrayReferenceTransformPath::root()
            ),
            Ok(replacement),
        );
    }

    #[test]
    fn test_array_reference_discharge_write_stages_composed_views() {
        // A write materializes only the strict parents of the leaf, so it stages no read of the old leaf value, and
        // writes the replacement back through both transforms in reverse order.
        assert_eq!(
            render_composed_view_access(|context, current, replacement, alias| {
                Ok(vec![ArrayReferenceDischarge::write(context, current, replacement, alias)?])
            }),
            indoc! {"
                lambda %0:f32[3, 3], %1:f32[2] .
                let %2:f32[2, 2] = slice [start_indices=[1, 0], limit_indices=[3, 2]] %0
                    %3:f32[1, 2] = reshape [shape=[1, 2]] %1
                    %4:f32[2, 2] = update_slice [start_indices=[1, 0]] %2 %3
                    %5:f32[3, 3] = update_slice [start_indices=[1, 0]] %0 %4
                in (%5)"},
        );
    }

    #[test]
    fn test_array_reference_discharge_swap() {
        let context = TestEagerContext::new();
        let current = TestValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let alias = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] });
        assert_eq!(
            ArrayReferenceDischarge::swap(
                &context,
                &current,
                TestValue::Array(Array::vector(vec![4.0f32, 5.0]).unwrap()),
                &alias,
            ),
            Ok((
                TestValue::Array(Array::vector(vec![2.0f32, 3.0]).unwrap()),
                TestValue::Array(Array::vector(vec![1.0f32, 4.0, 5.0]).unwrap()),
            )),
        );
    }

    #[test]
    fn test_array_reference_discharge_swap_stages_composed_views() {
        // A swap reads the old leaf value through the chain and writes the replacement back through both transforms in
        // reverse order, restoring the axis that the index removed.
        assert_eq!(
            render_composed_view_access(|context, current, replacement, alias| {
                let (previous, updated) = ArrayReferenceDischarge::swap(context, current, replacement, alias)?;
                Ok(vec![previous, updated])
            }),
            indoc! {"
                lambda %0:f32[3, 3], %1:f32[2] .
                let %2:f32[2, 2] = slice [start_indices=[1, 0], limit_indices=[3, 2]] %0
                    %3:f32[1, 2] = slice [start_indices=[1, 0], limit_indices=[2, 2]] %2
                    %4:f32[2] = reshape [shape=[2]] %3
                    %5:f32[1, 2] = reshape [shape=[1, 2]] %1
                    %6:f32[2, 2] = update_slice [start_indices=[1, 0]] %2 %5
                    %7:f32[3, 3] = update_slice [start_indices=[1, 0]] %0 %6
                in (%4, %7)"},
        );
    }

    #[test]
    fn test_array_reference_discharge_accumulate() {
        let context = TestEagerContext::new();
        let current = TestValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let alias = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] });
        assert_eq!(
            ArrayReferenceDischarge::accumulate(
                &context,
                &current,
                TestValue::Array(Array::vector(vec![4.0f32, 5.0]).unwrap()),
                &alias,
            ),
            Ok(TestValue::Array(Array::vector(vec![1.0f32, 6.0, 8.0]).unwrap())),
        );
    }

    #[test]
    fn test_array_reference_discharge_accumulate_stages_composed_views() {
        // An accumulation adds at the leaf and rebuilds each enclosing parent without reading the leaf a second time.
        assert_eq!(
            render_composed_view_access(|context, current, update, alias| {
                Ok(vec![ArrayReferenceDischarge::accumulate(context, current, update, alias)?])
            }),
            indoc! {"
                lambda %0:f32[3, 3], %1:f32[2] .
                let %2:f32[2, 2] = slice [start_indices=[1, 0], limit_indices=[3, 2]] %0
                    %3:f32[1, 2] = slice [start_indices=[1, 0], limit_indices=[2, 2]] %2
                    %4:f32[2] = reshape [shape=[2]] %3
                    %5:f32[2] = add %4 %1
                    %6:f32[1, 2] = reshape [shape=[1, 2]] %5
                    %7:f32[2, 2] = update_slice [start_indices=[1, 0]] %2 %6
                    %8:f32[3, 3] = update_slice [start_indices=[1, 0]] %0 %7
                in (%8)"},
        );
    }

    #[test]
    fn test_array_reference_discharge_accumulate_reconstructs_composed_slices() {
        // Rank-preserving slices need no reshapes, so composed slices update each parent in place with its child.
        let alias = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 3, 1)] })
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 2, 1)] });
        assert_eq!(
            ArrayReferenceDischarge::accumulate(
                &TestEagerContext::new(),
                &TestValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap()),
                TestValue::Array(Array::vector(vec![1.0f32, 2.0]).unwrap()),
                &alias,
            ),
            Ok(TestValue::Array(Array::vector(vec![1.0f32, 3.0, 5.0, 4.0]).unwrap())),
        );
        let alias: ArrayReferenceTransformPath<TestTracer> = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 3, 1)] })
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 2, 1)] });
        let (_, staged): (_, Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>) =
            TestEagerContext::trace(
                |inputs: Vec<TestTracer>| {
                    let context = inputs[0].context().clone();
                    Ok(vec![ArrayReferenceDischarge::accumulate(&context, &inputs[0], inputs[1].clone(), &alias)?])
                },
                vec![
                    ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4])),
                    ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])),
                ],
            )
            .unwrap();
        assert_eq!(
            staged.to_string(),
            indoc! {"
                lambda %0:f32[4], %1:f32[2] .
                let %2:f32[3] = slice [start_indices=[1], limit_indices=[4]] %0
                    %3:f32[2] = slice [start_indices=[0], limit_indices=[2]] %2
                    %4:f32[2] = add %3 %1
                    %5:f32[3] = update_slice [start_indices=[0]] %2 %4
                    %6:f32[4] = update_slice [start_indices=[1]] %0 %5
                in (%6)"},
        );
    }

    #[test]
    fn test_array_reference_discharge_stages_dynamic_indices() {
        // A dynamic index stages a size-one dynamic slice whose start is the binding on every axis, which the full
        // extent of the other axes clamps to zero, and a reshape that removes the indexed axis. Updates restore that
        // axis and write it back with the same starts, so they replace exactly the elements that the read selects.
        let stage = |inputs: Vec<TestTracer>| {
            let context = inputs[0].context().clone();
            let alias = ArrayReferenceTransformPath::root()
                .with_bound_transform(
                    ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
                    vec![inputs[1].clone()],
                )
                .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] });
            let selected = ArrayReferenceDischarge::read(&context, &inputs[0], &alias)?;
            let written = ArrayReferenceDischarge::write(&context, &inputs[0], inputs[2].clone(), &alias)?;
            let accumulated = ArrayReferenceDischarge::accumulate(&context, &inputs[0], inputs[2].clone(), &alias)?;
            Ok(vec![selected, written, accumulated])
        };
        let (_, staged): (_, Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>) =
            TestEagerContext::trace(
                stage,
                vec![
                    ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2, 3])),
                    ArrayIrType::Array(ArrayType::scalar(DataType::I32)),
                    ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])),
                ],
            )
            .unwrap();
        assert_eq!(
            staged.to_string(),
            indoc! {"
                lambda %0:f32[2, 3], %1:i32[], %2:f32[2] .
                let %3:f32[1, 3] = dynamic_slice [sizes=[1, 3]] %0 %1 %1
                    %4:f32[3] = reshape [shape=[3]] %3
                    %5:f32[2] = slice [start_indices=[1], limit_indices=[3]] %4
                    %6:f32[1, 3] = dynamic_slice [sizes=[1, 3]] %0 %1 %1
                    %7:f32[3] = reshape [shape=[3]] %6
                    %8:f32[3] = update_slice [start_indices=[1]] %7 %2
                    %9:f32[1, 3] = reshape [shape=[1, 3]] %8
                    %10:f32[2, 3] = dynamic_update_slice %0 %9 %1 %1
                    %11:f32[1, 3] = dynamic_slice [sizes=[1, 3]] %0 %1 %1
                    %12:f32[3] = reshape [shape=[3]] %11
                    %13:f32[2] = slice [start_indices=[1], limit_indices=[3]] %12
                    %14:f32[2] = add %13 %2
                    %15:f32[3] = update_slice [start_indices=[1]] %12 %14
                    %16:f32[1, 3] = reshape [shape=[1, 3]] %15
                    %17:f32[2, 3] = dynamic_update_slice %0 %16 %1 %1
                in (%5, %10, %17)"},
        );
        assert_eq!(
            staged.interpret(vec![
                TestValue::Array(Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
                TestValue::Array(Array::scalar(1i32).unwrap()),
                TestValue::Array(Array::vector(vec![20.0f32, 30.0]).unwrap()),
            ]),
            Ok(vec![
                TestValue::Array(Array::vector(vec![5.0f32, 6.0]).unwrap()),
                TestValue::Array(Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, 4.0, 20.0, 30.0]).unwrap()),
                TestValue::Array(Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, 4.0, 25.0, 36.0]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_array_reference_discharge_clamps_dynamic_indices() {
        // Dynamic slicing counts a negative index from the end once, so `-1` selects the last row while `-8` stays
        // negative and clamps to the first row, and an oversized index clamps to the last row.
        let (_, staged): (_, Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>) =
            TestEagerContext::trace(
                |inputs: Vec<TestTracer>| {
                    let context = inputs[0].context().clone();
                    let alias = ArrayReferenceTransformPath::root().with_bound_transform(
                        ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
                        vec![inputs[1].clone()],
                    );
                    Ok(vec![ArrayReferenceDischarge::read(&context, &inputs[0], &alias)?])
                },
                vec![
                    ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2, 3])),
                    ArrayIrType::Array(ArrayType::scalar(DataType::I32)),
                ],
            )
            .unwrap();
        let matrix = TestValue::Array(Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        let first_row = TestValue::Array(Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap());
        let last_row = TestValue::Array(Array::vector(vec![4.0f32, 5.0, 6.0]).unwrap());
        assert_eq!(
            staged.clone().interpret(vec![matrix.clone(), TestValue::Array(Array::scalar(-1i32).unwrap())]),
            Ok(vec![last_row.clone()]),
        );
        assert_eq!(
            staged.clone().interpret(vec![matrix.clone(), TestValue::Array(Array::scalar(-8i32).unwrap())]),
            Ok(vec![first_row]),
        );
        assert_eq!(
            staged.clone().interpret(vec![matrix.clone(), TestValue::Array(Array::scalar(1i32).unwrap())]),
            Ok(vec![last_row.clone()]),
        );
        assert_eq!(staged.interpret(vec![matrix, TestValue::Array(Array::scalar(80i32).unwrap())]), Ok(vec![last_row]));
    }

    #[test]
    fn test_array_ir_operation_base_input_count() {
        // Reference accesses count their reference and ordinary inputs before the trailing dynamic-index bindings, a
        // freeze consumes its single reference input, and pure operations have no binding groups.
        let transform = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic };
        assert_eq!(
            TestOperation::ReferenceRead(TestRead::new().with_transforms(vec![transform])).base_input_count(),
            1
        );
        assert_eq!(TestOperation::ReferenceFreeze(ReferenceFreezeOperation::new()).base_input_count(), 1);
        assert_eq!(TestOperation::from(AddOperation::<ArrayIrType>::new()).base_input_count(), 0);
    }

    #[test]
    fn test_array_ir_operation_reference_access_descriptor() {
        // An access describes its transforms and the input positions of their bindings.
        let transform = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic };
        let read = TestOperation::ReferenceRead(TestRead::new().with_transforms(vec![transform.clone()]));
        let descriptor = read.reference_access_descriptor(0).unwrap();
        assert_eq!(descriptor.transforms(), &[transform]);
        assert_eq!(descriptor.bindings(), 1..2);
        assert_eq!(read.reference_access_descriptor(1), None);

        // A freeze accesses its complete root without transforms, and pure operations access no reference.
        let freeze = TestOperation::ReferenceFreeze(ReferenceFreezeOperation::new());
        let descriptor = freeze.reference_access_descriptor(0).unwrap();
        assert!(descriptor.transforms().is_empty());
        assert_eq!(descriptor.bindings(), 1..1);
        assert_eq!(freeze.reference_access_descriptor(1), None);
        assert_eq!(TestOperation::from(AddOperation::<ArrayIrType>::new()).reference_access_descriptor(0), None);
    }

    #[test]
    fn test_array_ir_operation_with_reference_access_transforms() {
        // Replacing an access path changes the bindings that the access expects, and a whole-root access accepts only
        // the empty path that it already has.
        let transform = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic };
        let read = TestOperation::ReferenceRead(TestRead::new().with_transforms(vec![transform]));
        let replaced = read.with_reference_access_transforms(0, Vec::new()).unwrap();
        assert_eq!(replaced.reference_access_descriptor(0).unwrap().bindings(), 1..1);
        let freeze = TestOperation::ReferenceFreeze(ReferenceFreezeOperation::new());
        assert!(matches!(
            freeze.with_reference_access_transforms(0, Vec::new()),
            Ok(TestOperation::ReferenceFreeze(_))
        ));
    }

    #[test]
    fn test_array_ir_operation_with_reference_access_transforms_rejects_unsupported_paths() {
        // Delegated accesses report their own errors, while this family rejects paths on inputs that it does not
        // access and non-empty paths on whole-root accesses.
        let read = TestOperation::ReferenceRead(TestRead::new());
        assert!(matches!(
            read.with_reference_access_transforms(1, Vec::new()),
            Err(ProgramError::InvalidArgument { message })
                if message == "`reference_read` has no reference access at input 1",
        ));
        let transform = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) };
        assert!(matches!(
            TestOperation::ReferenceFreeze(ReferenceFreezeOperation::new())
                .with_reference_access_transforms(0, vec![transform]),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "`reference_freeze` cannot replace the reference transforms at input 0",
        ));
        assert!(matches!(
            TestOperation::from(AddOperation::<ArrayIrType>::new()).with_reference_access_transforms(0, Vec::new()),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "`add` cannot replace the reference transforms at input 0",
        ));
    }
}
