//! Array-reference handles and their ordered, root-preserving index and slice mappings.

// TODO(eaplatanios): Review this module.

use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use ryft_macros::Parameter;
use thiserror::Error;

use crate::arrays::addressing::ArraySliceAxis;
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::dimensions::{Dimension, Shape};
use crate::operations::{Add, Reshape, Slice, UpdateSlice};
use crate::parameters::Parameter;
use crate::programs::{
    ProgramError, ReadyOrPendingReferenceGuard, Reference, ReferenceError, ReferenceId, ReferenceType,
    ReferenceViewPath, Type, TypeError, TypeIdentityRenaming, Typed, Value,
};

/// Error produced by an invalid eager array-reference view operation.
#[derive(Clone, Debug, Error, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ArrayReferenceViewError {
    /// A consuming freeze was attempted through a derived view instead of the root handle.
    #[error("cannot freeze a reference view; freeze the root reference instead")]
    CannotFreezeView,

    /// A derived view was read through the bound-free root-only accessor.
    #[error("cannot read a reference view through the root-only snapshot accessor")]
    CannotReadRootThroughView,

    /// A derived or identity-renamed handle was used as a backend root-state transaction boundary.
    #[error("reference runtime transactions require an unrenamed root handle")]
    InvalidRuntimeRoot,
}

/// One validated coordinate transform in an [`ArrayReferenceView`]'s root-to-handle mapping.
///
/// A transform describes both directions of one view step: applying it extracts a selected child value from its
/// parent, while replacing that child reconstructs a value with exactly the parent's original type. This
/// bidirectional contract lets reference reads operate on the selected value and lets write-only replacements, swaps,
/// or additive updates reconstruct the shared root without changing its declared type. A write-only traversal
/// materializes the strict parents needed for reconstruction but deliberately skips extracting the overwritten leaf.
///
/// Transforms are interpreted in order from the root outward. [`Index`](Self::Index) removes one statically selected
/// axis; [`Slice`](Self::Slice) preserves rank and selects one static unit-stride range per axis.
/// Dynamic indexing and strided slicing are intentionally not represented until their inverse update semantics are
/// supported.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
#[non_exhaustive]
pub enum ArrayReferenceViewTransform {
    /// Selects one coordinate from one axis and removes that axis from the view shape.
    Index {
        /// Axis selected in the transform's input view.
        axis: usize,

        /// Coordinate selected on `axis`.
        index: usize,
    },

    /// Selects one static unit-stride range on every axis while preserving rank.
    Slice {
        /// Per-axis selections in the transform's input view.
        axes: Vec<ArraySliceAxis>,
    },
}

/// Normalized coordinates of one [`ArrayReferenceViewTransform`] applied to one statically shaped input.
///
/// Both transform kinds reduce to slicing one unit-stride hyper-rectangle out of the input, optionally followed by
/// squeezing the indexed axis. Normalizing to this shared form lets every consumer (type derivation, eager reads,
/// eager update reconstruction, and staged discharge) share one validation and address computation.
struct ViewSelection {
    /// Inclusive slice start per input axis.
    starts: Vec<usize>,

    /// Exclusive slice limit per input axis.
    limits: Vec<usize>,

    /// Exact static output shape after squeezing the indexed axis, for
    /// [`ArrayReferenceViewTransform::Index`] transforms only; [`None`] for rank-preserving slices, whose output
    /// shape is exactly [`Self::update_shape`].
    squeezed_output_shape: Option<Shape>,
}

impl ViewSelection {
    /// Returns the static shape of the sliced hyper-rectangle before squeezing (i.e., the update shape that writes
    /// back into the selected coordinates).
    fn update_shape(&self) -> Shape {
        Shape::new(
            self.starts
                .iter()
                .zip(self.limits.iter())
                .map(|(start, limit)| Dimension::Static(limit - start))
                .collect(),
        )
    }
}

impl ArrayReferenceViewTransform {
    /// Validates this transform against `input` and returns its normalized selection coordinates.
    fn selection(&self, input: &ArrayType) -> Result<ViewSelection, TypeError> {
        match self {
            Self::Index { axis, index } => {
                let shape = input.static_shape().ok_or_else(|| {
                    TypeError::invalid(format!("reference indexing requires a static referent type but got `{input}`"))
                })?;
                if *axis >= shape.rank() {
                    return Err(TypeError::invalid(format!(
                        "reference index axis {axis} is out of bounds for rank {}",
                        shape.rank(),
                    )));
                }
                if *index >= shape.dimension(*axis) {
                    return Err(TypeError::invalid(format!(
                        "reference index {index} on axis {axis} is out of bounds for size {}",
                        shape.dimension(*axis),
                    )));
                }
                let mut starts = vec![0; shape.rank()];
                starts[*axis] = *index;
                let mut limits = shape.dimensions().to_vec();
                limits[*axis] = index + 1;
                let output_shape = Shape::new(
                    shape
                        .dimensions()
                        .iter()
                        .enumerate()
                        .filter_map(|(candidate, size)| (candidate != *axis).then_some(Dimension::Static(*size)))
                        .collect(),
                );
                Ok(ViewSelection { starts, limits, squeezed_output_shape: Some(output_shape) })
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
                for (axis, (selection, input_size)) in axes.iter().copied().zip(shape.dimensions()).enumerate() {
                    if selection.stride() != 1 {
                        return Err(TypeError::invalid(format!(
                            "reference slice axis {axis} stride must be 1 until scatter-backed strided updates are \
                             supported",
                        )));
                    }
                    let limit = selection.start().checked_add(selection.size()).ok_or_else(|| {
                        TypeError::invalid(format!("reference slice limit overflows `usize` on axis {axis}"))
                    })?;
                    if limit > *input_size {
                        return Err(TypeError::invalid(format!(
                            "reference slice on axis {axis} with start {} and size {} exceeds input size {input_size}",
                            selection.start(),
                            selection.size(),
                        )));
                    }
                    starts.push(selection.start());
                    limits.push(limit);
                }
                Ok(ViewSelection { starts, limits, squeezed_output_shape: None })
            }
        }
    }

    /// Returns the exact canonical array type produced from `input`.
    pub fn output_type(&self, input: &ArrayType) -> Result<ArrayType, TypeError> {
        let selection = self.selection(input)?;
        let sliced = input
            .slice(selection.starts.as_slice(), selection.limits.as_slice(), &vec![1; selection.starts.len()])
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        let output = match &selection.squeezed_output_shape {
            Some(shape) => sliced.reshape(shape.clone()).map_err(|error| TypeError::invalid(error.to_string()))?,
            None => sliced,
        };
        self.validate_reconstruction(input, &output, &selection)?;
        Ok(output)
    }

    /// Applies this transform to one carried parent value.
    fn apply_in<C: ViewReadCarrier>(&self, carrier: &mut C, input: &C::Value) -> Result<C::Value, ProgramError> {
        let selection = self.selection(carrier.array_type(input)?.as_ref())?;
        let sliced = carrier.slice(input, selection.starts, selection.limits)?;
        match selection.squeezed_output_shape {
            Some(shape) => carrier.reshape(&sliced, shape),
            None => Ok(sliced),
        }
    }

    /// Reconstructs the carried parent after replacing exactly the coordinates selected by this transform.
    fn replace_in<C: ViewWriteCarrier>(
        &self,
        carrier: &mut C,
        input: &C::Value,
        replacement: &C::Value,
    ) -> Result<C::Value, ProgramError> {
        let selection = self.selection(carrier.array_type(input)?.as_ref())?;
        match selection.squeezed_output_shape {
            Some(_) => {
                let update = carrier.reshape(replacement, selection.update_shape())?;
                carrier.update_slice(input, &update, selection.starts)
            }
            None => carrier.update_slice(input, replacement, selection.starts),
        }
    }

    /// Proves that updating the selected child reconstructs the exact parent storage type.
    ///
    /// Shape arithmetic alone cannot guarantee this: [`ArrayType`] also carries layouts, shardings, and other
    /// metadata whose slice and update-slice derivations are owned by the type system, so this check catches any
    /// transform whose forward selection and inverse update do not round-trip on that metadata. Because derived
    /// view types are computed once when a handle or staged view is created, the proof runs once per composition
    /// step rather than per access.
    fn validate_reconstruction(
        &self,
        input: &ArrayType,
        output: &ArrayType,
        selection: &ViewSelection,
    ) -> Result<(), TypeError> {
        let update = match selection.squeezed_output_shape {
            Some(_) => {
                output.reshape(selection.update_shape()).map_err(|error| TypeError::invalid(error.to_string()))?
            }
            None => output.clone(),
        };
        let reconstructed = input
            .update_slice(&update, selection.starts.as_slice())
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        if &reconstructed == input {
            return Ok(());
        }
        Err(TypeError::invalid(format!(
            "reference view reconstruction changes root type from `{input}` to `{reconstructed}`",
        )))
    }
}

/// Immutable coordinate mapping between a shared array-reference root and one derived handle: the array
/// specialization of the generic [`ReferenceViewPath`], whose descriptions are [`ArrayReferenceViewTransform`]s.
///
/// The mapping stores validated transforms in root-to-handle order. The empty mapping ([`root`](Self::root)) is the
/// identity view and denotes the complete root. Each additional transform is applied to the preceding view, so
/// indexing or slicing an already-derived [`ArrayReference`] composes onto the same shared root rather than creating
/// another mutable resource.
///
/// This type is structural metadata only: it owns neither the referenced array nor its resource identity, liveness,
/// or synchronization state. [`ArrayReference`] pairs it with a handle to the shared reference allocation, and the
/// array view overlay ([`ArrayReferenceAnalysis`](crate::ArrayReferenceAnalysis)) records one per reference-typed
/// program value. The view determines that handle's referent type and selected coordinates; mutations reconstruct
/// the root by applying the inverse update of each transform in reverse order. Consequently, overlapping handles may
/// select the same root coordinates and observe one another's ordered mutations, while equality and hashing
/// distinguish different transform sequences.
///
/// Views currently support composed static indexing and static unit-stride slicing. Derived views cannot themselves
/// cross attached-region or external runtime state boundaries: pass the root handle across the boundary and recreate
/// the view within the destination scope.
pub type ArrayReferenceView = ReferenceViewPath<ArrayReferenceViewTransform>;

impl ArrayReferenceView {
    /// Returns the ordered transforms applied from the root outward.
    #[inline]
    pub fn transforms(&self) -> &[ArrayReferenceViewTransform] {
        self.views()
    }

    /// Returns the exact view type derived from `root_type`.
    pub fn output_type(&self, root_type: &ArrayType) -> Result<ArrayType, TypeError> {
        self.transforms()
            .iter()
            .try_fold(root_type.clone(), |r#type, transform| transform.output_type(&r#type))
    }

    /// Appends a transform whose local input/output types were already validated by the caller.
    pub(crate) fn with_transform_unchecked(&self, transform: ArrayReferenceViewTransform) -> Self {
        self.with_view(transform)
    }

    /// Applies the complete mapping to one root snapshot.
    pub(crate) fn apply<A>(&self, root: &A) -> Result<A, ProgramError>
    where
        A: Value<Type = ArrayType> + Reshape + Slice,
    {
        let mut carrier = EagerViewCarrier(PhantomData);
        self.transforms()
            .iter()
            .try_fold(root.clone(), |value, transform| transform.apply_in(&mut carrier, &value))
    }

    /// Replaces this view and returns the reconstructed root plus its old view snapshot.
    pub(crate) fn swap<A>(&self, root: &A, replacement: &A) -> Result<(A, A), ProgramError>
    where
        A: Value<Type = ArrayType> + Reshape + Slice + UpdateSlice,
    {
        let (old, reconstructed) =
            self.swap_in(&mut EagerViewCarrier(PhantomData), root.clone(), replacement.clone())?;
        Ok((reconstructed, old))
    }

    /// Materializes each parent-to-child snapshot once for update reconstruction, in root-to-view order starting
    /// with `root` itself.
    pub(crate) fn intermediates_in<C: ViewReadCarrier>(
        &self,
        carrier: &mut C,
        root: C::Value,
    ) -> Result<Vec<C::Value>, ProgramError> {
        let mut intermediates = Vec::with_capacity(self.transforms().len() + 1);
        intermediates.push(root);
        for transform in self.transforms() {
            let child = transform.apply_in(carrier, intermediates.last().unwrap())?;
            intermediates.push(child);
        }
        Ok(intermediates)
    }

    /// Reconstructs a root from precomputed view intermediates and a replacement leaf.
    pub(crate) fn reconstruct_in<C: ViewWriteCarrier>(
        &self,
        carrier: &mut C,
        intermediates: &[C::Value],
        replacement: C::Value,
    ) -> Result<C::Value, ProgramError> {
        let transforms = self.transforms();
        if intermediates.len() != transforms.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference view reconstruction requires {} parent snapshots but received {}",
                transforms.len(),
                intermediates.len(),
            )));
        }
        let mut reconstructed = replacement;
        for transform_index in (0..transforms.len()).rev() {
            reconstructed =
                transforms[transform_index].replace_in(carrier, &intermediates[transform_index], &reconstructed)?;
        }
        Ok(reconstructed)
    }

    /// Replaces this view's selected coordinates through `carrier`, returning their previous snapshot plus the
    /// reconstructed root, so that the eager swap and the discharge-time replacement share one traversal.
    pub(crate) fn swap_in<C: ViewWriteCarrier<Value: Clone>>(
        &self,
        carrier: &mut C,
        root: C::Value,
        replacement: C::Value,
    ) -> Result<(C::Value, C::Value), ProgramError> {
        let intermediates = self.intermediates_in(carrier, root)?;

        // The traversal always pushes the root itself first, so the chain is never empty and its last snapshot is
        // the value this view selects.
        let previous = intermediates.last().unwrap().clone();
        let reconstructed = self.reconstruct_in(carrier, &intermediates[..self.transforms().len()], replacement)?;
        Ok((previous, reconstructed))
    }

    /// Replaces this view's selected coordinates through `carrier` without materializing the selected old value.
    ///
    /// Immutable root reconstruction still needs each strict parent of the selected leaf so coordinates outside the
    /// logical view survive. The final transform is deliberately not applied: its output is exactly the old selected
    /// value that write-only semantics must not observe. An identity view therefore returns `replacement` directly.
    pub(crate) fn write_in<C: ViewWriteCarrier>(
        &self,
        carrier: &mut C,
        root: C::Value,
        replacement: C::Value,
    ) -> Result<C::Value, ProgramError> {
        let transforms = self.transforms();
        if transforms.is_empty() {
            return Ok(replacement);
        }
        let mut intermediates = Vec::with_capacity(transforms.len());
        intermediates.push(root);
        for transform in &transforms[..transforms.len() - 1] {
            let child = transform.apply_in(carrier, intermediates.last().unwrap())?;
            intermediates.push(child);
        }
        self.reconstruct_in(carrier, intermediates.as_slice(), replacement)
    }
}

/// One value carrier through which a reference view maps between a shared root and one derived handle.
///
/// The root-to-view push-forward and the reverse update-slice reconstruction each exist exactly once, on
/// [`ArrayReferenceView`], generically over this carrier: the eager carrier operates on concrete values with the
/// array-manipulation capabilities, while reference discharge binds the identical operation sequence through its
/// destination context. Keeping one traversal guarantees the staged and eager semantics cannot drift apart.
pub(crate) trait ViewReadCarrier {
    /// Value representation carried through the traversal.
    type Value;

    /// Returns the carried value's array type, borrowing from the carrier or the value where possible.
    fn array_type<'c>(&'c self, value: &'c Self::Value) -> Result<Cow<'c, ArrayType>, ProgramError>;

    /// Slices one unit-stride hyper-rectangle out of `input`.
    fn slice(
        &mut self,
        input: &Self::Value,
        starts: Vec<usize>,
        limits: Vec<usize>,
    ) -> Result<Self::Value, ProgramError>;

    /// Reshapes `input` to `shape`.
    fn reshape(&mut self, input: &Self::Value, shape: Shape) -> Result<Self::Value, ProgramError>;
}

/// A [`ViewReadCarrier`] that can also write a selected hyper-rectangle back into its parent.
pub(crate) trait ViewWriteCarrier: ViewReadCarrier {
    /// Returns `target` with `update` written at `starts`.
    fn update_slice(
        &mut self,
        target: &Self::Value,
        update: &Self::Value,
        starts: Vec<usize>,
    ) -> Result<Self::Value, ProgramError>;
}

/// Stateless eager carrier over one concrete array value family.
struct EagerViewCarrier<A>(PhantomData<A>);

impl<A: Value<Type = ArrayType> + Reshape + Slice> ViewReadCarrier for EagerViewCarrier<A> {
    type Value = A;

    fn array_type<'c>(&'c self, value: &'c A) -> Result<Cow<'c, ArrayType>, ProgramError> {
        Ok(value.r#type())
    }

    fn slice(&mut self, input: &A, starts: Vec<usize>, limits: Vec<usize>) -> Result<A, ProgramError> {
        input.slice(starts.as_slice(), limits.as_slice(), &vec![1; starts.len()])
    }

    fn reshape(&mut self, input: &A, shape: Shape) -> Result<A, ProgramError> {
        input.reshape(shape)
    }
}

impl<A: Value<Type = ArrayType> + Reshape + Slice + UpdateSlice> ViewWriteCarrier for EagerViewCarrier<A> {
    fn update_slice(&mut self, target: &A, update: &A, starts: Vec<usize>) -> Result<A, ProgramError> {
        target.update_slice(update, starts.as_slice())
    }
}

/// Eager array-reference handle pairing one shared root allocation with handle-local view metadata.
///
/// Equality and hashing identify the mutable location and structural view, not the handle-local type-identity
/// namespace.
/// Renaming type identities therefore preserves equality with the original handle when its view is unchanged.
pub struct ArrayReference<A: Value<Type = ArrayType>> {
    /// Handle to the shared root allocation.
    root: Reference<A>,

    /// Ordered mapping from the shared root to this handle's referent.
    view: ArrayReferenceView,

    /// Exact handle type derived once from the root type and view, so that repeated [`r#type`](Typed::type) calls
    /// stay borrow-cheap instead of re-deriving the complete transform chain.
    r#type: ReferenceType<ArrayType>,
}

impl<A: Value<Type = ArrayType>> ArrayReference<A> {
    /// Creates a new root reference initialized with `value`.
    #[inline]
    pub fn new(value: A) -> Self {
        // `A::Type` is exactly `ArrayType`, whose type family cannot denote a reference, so the generic nested-
        // referent rejection is unreachable for this specialized constructor.
        let root = Reference::new(value).unwrap();
        let r#type = root.r#type().into_owned();
        Self { root, view: ArrayReferenceView::root(), r#type }
    }

    /// Returns this shared reference allocation's process-local identity.
    #[inline]
    pub fn id(&self) -> ReferenceId {
        self.root.id()
    }

    /// Returns whether this is an unrenamed root handle accepted at a backend runtime state boundary.
    #[doc(hidden)]
    #[inline]
    pub fn is_runtime_root_handle(&self) -> bool {
        self.view.is_root() && self.root.uses_storage_type_identities()
    }

    /// Locks an unrenamed root for one backend-owned state transaction.
    #[doc(hidden)]
    pub fn lock_root(&self) -> Result<ReadyOrPendingReferenceGuard<'_, A>, ProgramError> {
        if !self.is_runtime_root_handle() {
            return Err(ProgramError::custom(ArrayReferenceViewError::InvalidRuntimeRoot));
        }
        self.root.lock().map_err(ProgramError::custom)
    }

    /// Returns a derived handle after appending `transform` without creating a resource.
    pub fn with_transform(&self, transform: ArrayReferenceViewTransform) -> Result<Self, ProgramError> {
        // The cached handle type already reflects every earlier transform, so composition validates and derives
        // incrementally instead of re-folding the complete chain from the root type. Derivation is purely structural:
        // holder liveness is checked only when the resulting handle accesses state.
        let referent = transform.output_type(self.r#type.referent())?;
        let view = self.view.with_transform_unchecked(transform);
        Ok(Self { root: self.root.clone(), view, r#type: ReferenceType::new(referent) })
    }

    /// Returns an immutable snapshot of this handle's selected coordinates.
    pub fn read(&self) -> Result<A, ProgramError>
    where
        A: Reshape + Slice,
    {
        self.view.apply(&self.root.read().map_err(ProgramError::custom)?)
    }

    /// Returns an immutable root snapshot without requiring array-manipulation capabilities.
    pub fn read_root(&self) -> Result<A, ProgramError> {
        if !self.view.is_root() {
            return Err(ProgramError::custom(ArrayReferenceViewError::CannotReadRootThroughView));
        }
        self.root.read().map_err(ProgramError::custom)
    }

    /// Validates that `value` exactly matches this handle's derived referent type. Root-handle mutations inherit this
    /// rule from the shared reference state, but derived-view mutations must enforce it themselves: update-slice
    /// reconstruction only requires the written value to fit inside the selected coordinates, so a smaller replacement
    /// would otherwise silently write a partial update.
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

    /// Replaces this handle's selected coordinates and returns their previous snapshot.
    ///
    /// Errors from the shared reference state take precedence over a replacement-type error, consistently with
    /// mutation through the root handle.
    pub fn swap(&self, replacement: A) -> Result<A, ProgramError>
    where
        A: Reshape + Slice + UpdateSlice,
    {
        if self.view.is_root() {
            return self.root.swap(replacement).map_err(ProgramError::custom);
        }
        // Validating inside the update keeps holder-state errors (frozen, poisoned, mid-transaction) ahead of the
        // replacement-type diagnostic, matching the root path.
        self.root.update(|current| {
            self.validate_view_referent_type(&replacement)?;
            self.view.swap(current, &replacement)
        })
    }

    /// Replaces this handle's selected coordinates without returning their previous snapshot.
    ///
    /// Errors from the shared reference state take precedence over a replacement-type error, consistently with
    /// mutation through the root handle.
    pub fn write(&self, replacement: A) -> Result<(), ProgramError>
    where
        A: Reshape + Slice + UpdateSlice,
    {
        if self.view.is_root() {
            return self.root.write(replacement).map_err(ProgramError::custom);
        }
        // Validation remains inside the holder transaction so frozen, poisoned, and leased-state diagnostics retain
        // precedence over replacement-type errors, matching the root write and swap paths.
        self.root.update(|current| {
            self.validate_view_referent_type(&replacement)?;
            self.view
                .write_in(&mut EagerViewCarrier(PhantomData), current.clone(), replacement)
                .map(|updated| (updated, ()))
        })
    }

    /// Adds `update` into this handle's selected coordinates.
    pub fn add_update(&self, update: &A) -> Result<(), ProgramError>
    where
        A: Add + Reshape + Slice + UpdateSlice,
    {
        if self.view.is_root() {
            return self.root.update(|current| current.add(update).map(|updated| (updated, ())));
        }
        self.root.update(|current| {
            let mut carrier = EagerViewCarrier(PhantomData);
            let intermediates = self.view.intermediates_in(&mut carrier, current.clone())?;
            let updated_view = intermediates.last().unwrap().add(update)?;
            self.validate_view_referent_type(&updated_view)?;
            self.view
                .reconstruct_in(&mut carrier, &intermediates[..self.view.transforms().len()], updated_view)
                .map(|updated| (updated, ()))
        })
    }

    /// Consumes the referenced root, invalidating its complete alias family, and rejects a derived view without
    /// changing shared state.
    ///
    /// This takes the handle by shared borrow while the value-level [`ReferenceFreeze`](crate::ReferenceFreeze)
    /// capability above it takes one by value. The asymmetry is mechanical rather than semantic: the composite
    /// implementation reaches this handle through a projection of its owned value, which yields a borrow, and the
    /// linearity the capability enforces is already enforced one layer up.
    pub fn freeze(&self) -> Result<A, ProgramError> {
        if !self.view.is_root() {
            return Err(ProgramError::custom(ArrayReferenceViewError::CannotFreezeView));
        }
        self.root.freeze().map_err(ProgramError::custom)
    }

    /// Returns this same root and view with handle-local identities renamed bidirectionally.
    pub(crate) fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayType as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        let root = self.root.rename_type_identities(renaming)?;
        let referent = self.view.output_type(root.r#type().referent())?;
        Ok(Self { root, view: self.view.clone(), r#type: ReferenceType::new(referent) })
    }
}

impl<A: Value<Type = ArrayType>> Clone for ArrayReference<A> {
    #[inline]
    fn clone(&self) -> Self {
        Self { root: self.root.clone(), view: self.view.clone(), r#type: self.r#type.clone() }
    }
}

impl<A: Value<Type = ArrayType>> Debug for ArrayReference<A> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("ArrayReference").field("id", &self.id()).field("view", &self.view).finish()
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
        self.root == other.root && self.view == other.view
    }
}

impl<A: Value<Type = ArrayType>> Eq for ArrayReference<A> {}

impl<A: Value<Type = ArrayType>> Hash for ArrayReference<A> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.root.hash(state);
        self.view.hash(state);
    }
}

impl<A: Value<Type = ArrayType>> Parameter for ArrayReference<A> {}

// The cached type is derived deterministically from the root type and view at construction, so equality and hashing
// over `(root, view)` remain consistent with it.
impl<A: Value<Type = ArrayType>> Typed for ArrayReference<A> {
    type Type = ReferenceType<ArrayType>;

    fn r#type(&self) -> Cow<'_, Self::Type> {
        Cow::Borrowed(&self.r#type)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::Array;
    use crate::arrays::types::data::DataType;
    use crate::programs::{ReferenceCompletion, ReferenceReplacementPreparation};

    use super::*;

    #[test]
    fn test_array_reference_view_transform_rejects_invalid_selections() {
        let matrix_type = ArrayType::new_static(DataType::F32, [3, 4]);
        let vector_type = ArrayType::new_static(DataType::F32, [3]);

        // Static indexing selects one existing coordinate on one existing axis.
        assert_eq!(
            ArrayReferenceViewTransform::Index { axis: 2, index: 0 }.output_type(&matrix_type),
            Err(TypeError::invalid("reference index axis 2 is out of bounds for rank 2")),
        );
        assert_eq!(
            ArrayReferenceViewTransform::Index { axis: 0, index: 3 }.output_type(&matrix_type),
            Err(TypeError::invalid("reference index 3 on axis 0 is out of bounds for size 3")),
        );

        // Static slicing is rank-preserving, so it declares exactly one unit-stride selection per input axis and
        // stays inside every axis of the input.
        assert_eq!(
            ArrayReferenceViewTransform::Slice { axes: vec![ArraySliceAxis::new(0, 2, 1)] }.output_type(&matrix_type),
            Err(TypeError::invalid("reference slice has 1 axes but its input has rank 2")),
        );
        assert_eq!(
            ArrayReferenceViewTransform::Slice { axes: vec![ArraySliceAxis::new(0, 2, 2)] }.output_type(&vector_type),
            Err(TypeError::invalid(
                "reference slice axis 0 stride must be 1 until scatter-backed strided updates are supported",
            )),
        );
        assert_eq!(
            ArrayReferenceViewTransform::Slice { axes: vec![ArraySliceAxis::new(2, 3, 1)] }.output_type(&vector_type),
            Err(TypeError::invalid("reference slice on axis 0 with start 2 and size 3 exceeds input size 3")),
        );

        // The exclusive limit is computed as `start + size`, so an unrepresentable limit is rejected before it can
        // wrap around into an apparently valid selection.
        assert_eq!(
            ArrayReferenceViewTransform::Slice { axes: vec![ArraySliceAxis::new(usize::MAX, 1, 1)] }
                .output_type(&vector_type),
            Err(TypeError::invalid("reference slice limit overflows `usize` on axis 0")),
        );
    }

    #[test]
    fn test_array_reference_view_composition() {
        let root_type = ArrayType::new_static(DataType::F32, [3, 4]);
        let root = ArrayReferenceView::root();
        assert!(root.is_root());
        assert!(root.transforms().is_empty());
        assert_eq!(root.output_type(&root_type), Ok(root_type.clone()));

        // Each transform applies to the preceding view, so the slice narrows both axes and the index then removes
        // the leading axis of the already-narrowed view.
        let slice = ArrayReferenceViewTransform::Slice {
            axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 3, 1)],
        };
        let index = ArrayReferenceViewTransform::Index { axis: 0, index: 1 };
        let sliced = root.with_transform_unchecked(slice.clone());
        let indexed = sliced.with_transform_unchecked(index.clone());
        assert!(!sliced.is_root());
        assert_eq!(sliced.transforms(), &[slice.clone()]);
        assert_eq!(sliced.output_type(&root_type), Ok(ArrayType::new_static(DataType::F32, [2, 3])));
        assert_eq!(indexed.transforms(), &[slice.clone(), index.clone()]);
        assert_eq!(indexed.output_type(&root_type), Ok(ArrayType::new_static(DataType::F32, [3])));

        // Composition validates each appended transform against the preceding view's derived type, so an
        // out-of-bounds coordinate of the derived view is rejected even though it exists in the root.
        let handle = ArrayReference::new(Array::matrix(3, 4, (1..=12).map(|value| value as f32).collect()))
            .with_transform(slice.clone())
            .unwrap();
        assert_eq!(
            handle.with_transform(ArrayReferenceViewTransform::Index { axis: 0, index: 2 }).unwrap_err(),
            TypeError::invalid("reference index 2 on axis 0 is out of bounds for size 2").into(),
        );

        // Equality and hashing distinguish transform sequences, including the two orders of the same two transforms.
        let reversed = root
            .with_transform_unchecked(ArrayReferenceViewTransform::Index { axis: 0, index: 1 })
            .with_transform_unchecked(ArrayReferenceViewTransform::Slice { axes: vec![ArraySliceAxis::new(0, 3, 1)] });
        assert_eq!(indexed, indexed.clone());
        assert_ne!(indexed, sliced);
        assert_ne!(indexed, reversed);
        assert_eq!(indexed.output_type(&root_type), reversed.output_type(&root_type));
        let views = HashMap::from([
            (root.clone(), "root"),
            (sliced.clone(), "sliced"),
            (indexed.clone(), "indexed"),
            (reversed.clone(), "reversed"),
        ]);
        assert_eq!(views.len(), 4);
        assert_eq!(views.get(&indexed), Some(&"indexed"));
        assert_eq!(views.get(&reversed), Some(&"reversed"));
    }

    #[test]
    fn test_array_reference_reads_root_and_derived_handles() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]));
        let derived = root
            .with_transform(ArrayReferenceViewTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] })
            .unwrap();

        // The ordinary accessor applies the handle's complete view, while the bound-free root-only accessor remains
        // available to generic code that has no array manipulation capabilities.
        assert_eq!(root.read(), Ok(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0])));
        assert_eq!(derived.read(), Ok(Array::vector(vec![2.0_f32, 3.0])));
        assert_eq!(root.read_root(), Ok(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0])));

        assert_eq!(
            derived.read_root().unwrap_err().downcast_custom::<ArrayReferenceViewError>(),
            Some(&ArrayReferenceViewError::CannotReadRootThroughView),
        );
        assert_eq!(
            derived.read_root().unwrap_err().to_string(),
            "cannot read a reference view through the root-only snapshot accessor",
        );
    }

    #[test]
    fn test_array_reference_view_mutation_requires_the_exact_derived_referent_type() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]));
        let view = root
            .with_transform(ArrayReferenceViewTransform::Slice { axes: vec![ArraySliceAxis::new(0, 3, 1)] })
            .unwrap();

        // The update-slice reconstruction only requires the written value to fit inside the selected coordinates, so
        // the derived handle enforces exact referent equality itself and leaves the shared root untouched.
        let error = view.swap(Array::vector(vec![10.0_f32, 20.0])).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceError>(),
            Some(&ReferenceError::ReferentTypeMismatch {
                expected: "f32[3]".to_string(),
                actual: "f32[2]".to_string(),
            }),
        );
        assert_eq!(root.read(), Ok(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0])));

        let error = view.write(Array::vector(vec![10.0_f32, 20.0])).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceError>(),
            Some(&ReferenceError::ReferentTypeMismatch {
                expected: "f32[3]".to_string(),
                actual: "f32[2]".to_string(),
            }),
        );
        assert_eq!(root.read(), Ok(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0])));

        // An additive update whose result type drifts away from the view's element data type is rejected by the same
        // check, after the addition itself succeeded, so the holder still retains its previous value.
        let error = view.add_update(&Array::vector(vec![1.0_f64, 2.0, 3.0])).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceError>(),
            Some(&ReferenceError::ReferentTypeMismatch {
                expected: "f32[3]".to_string(),
                actual: "f64[3]".to_string(),
            }),
        );
        assert_eq!(root.read(), Ok(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0])));

        // A write-only replacement of exactly the derived type preserves unaffected coordinates without returning the
        // selected snapshot. A following swap observes that replacement value and replaces it again.
        assert_eq!(view.write(Array::vector(vec![10.0_f32, 20.0, 30.0])), Ok(()));
        assert_eq!(root.read(), Ok(Array::vector(vec![10.0_f32, 20.0, 30.0, 4.0])));
        assert_eq!(view.swap(Array::vector(vec![40.0_f32, 50.0, 60.0])), Ok(Array::vector(vec![10.0_f32, 20.0, 30.0])));
        assert_eq!(root.read(), Ok(Array::vector(vec![40.0_f32, 50.0, 60.0, 4.0])));

        // Holder-state errors take precedence over the replacement-type diagnostic: a frozen root swapped through a
        // derived view with a wrongly typed replacement reports the terminal state, not a shape to fix.
        assert_eq!(root.freeze(), Ok(Array::vector(vec![40.0_f32, 50.0, 60.0, 4.0])));
        let error = view.write(Array::vector(vec![1.0_f32, 2.0])).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
        let error = view.swap(Array::vector(vec![1.0_f32, 2.0])).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_array_reference_composed_index_of_slice_update_reconstructs_the_root() {
        let root_type = ArrayType::new_static(DataType::F32, [3, 3]);
        let root = ArrayReference::new(Array::matrix(3, 3, (1..=9).map(|value| value as f32).collect()));
        let view = root
            .with_transform(ArrayReferenceViewTransform::Slice {
                axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 2, 1)],
            })
            .unwrap()
            .with_transform(ArrayReferenceViewTransform::Index { axis: 0, index: 1 })
            .unwrap();
        assert_eq!(view.r#type().as_ref(), &ReferenceType::new(ArrayType::new_static(DataType::F32, [2])));

        // The composed mapping selects row 2 of the sliced view. A write reconstructs both strict parents without
        // returning the selected leaf, and the following additive update reaches the same coordinates.
        view.write(Array::vector(vec![70.0_f32, 80.0])).unwrap();
        assert_eq!(
            root.read(),
            Ok(Array::from_f64s(root_type.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 70.0, 80.0, 9.0])),
        );
        view.add_update(&Array::vector(vec![10.0_f32, 20.0])).unwrap();
        assert_eq!(root.read(), Ok(Array::from_f64s(root_type, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 80.0, 100.0, 9.0])),);
    }

    #[test]
    fn test_array_reference_view_derivation_is_pure_structural_composition() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]));
        let guard = root.lock_root().unwrap();
        let ReferenceReplacementPreparation::Prepared(prepared) = guard.prepare_replacement().unwrap() else {
            panic!("new reference unexpectedly has active read leases")
        };
        let transaction = prepared.begin(ReferenceCompletion::ready(Ok(())));

        // A derived handle is pure structural metadata over a live reference, so composing one must never resolve its
        // submitted work. The reference is parked in its `Taken` state, where every value access is unavailable behind
        // this retained guard until replacement commit, and derivation still computes its exact referent type.
        let transform = ArrayReferenceViewTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] };
        let derived = root.with_transform(transform).unwrap();
        assert_eq!(derived.r#type().as_ref(), &ReferenceType::new(ArrayType::new_static(DataType::F32, [2])));

        // Poisoning the submitted mutation is terminal for the alias family, but further derivation remains structural
        // composition. The resulting handle reports the reference failure only when it attempts to access state.
        transaction.poison("submission failed");
        let poisoned = ReferenceError::ExecutionPoisoned { reason: "submission failed".to_string() };
        assert_eq!(root.read().unwrap_err().downcast_custom::<ReferenceError>(), Some(&poisoned));
        let composed = derived.with_transform(ArrayReferenceViewTransform::Index { axis: 0, index: 0 }).unwrap();
        assert_eq!(composed.r#type().as_ref(), &ReferenceType::new(ArrayType::scalar(DataType::F32)));
        assert_eq!(composed.read().unwrap_err().downcast_custom::<ReferenceError>(), Some(&poisoned));

        let frozen = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]));
        assert_eq!(frozen.freeze(), Ok(Array::vector(vec![1.0_f32, 2.0])));
        let frozen_view = frozen.with_transform(ArrayReferenceViewTransform::Index { axis: 0, index: 1 }).unwrap();
        assert_eq!(frozen_view.read().unwrap_err().downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_array_reference_view_reconstruction_validates_parent_count() {
        let view = ArrayReferenceView::root()
            .with_transform_unchecked(ArrayReferenceViewTransform::Index { axis: 0, index: 0 });
        let mut carrier = EagerViewCarrier::<Array>(PhantomData);
        assert_eq!(
            view.reconstruct_in(&mut carrier, &[], Array::scalar(1.0_f32)),
            Err(ProgramError::MalformedProgram(
                "reference view reconstruction requires 1 parent snapshots but received 0".to_string(),
            )),
        );
        assert_eq!(
            view.reconstruct_in(
                &mut carrier,
                &[Array::vector(vec![1.0_f32]), Array::scalar(1.0_f32)],
                Array::scalar(1.0_f32),
            ),
            Err(ProgramError::MalformedProgram(
                "reference view reconstruction requires 1 parent snapshots but received 2".to_string(),
            )),
        );
    }

    #[test]
    fn test_array_reference_caches_its_derived_type() {
        let root_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let root = ArrayReference::new(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let slice = ArrayReferenceViewTransform::Slice {
            axes: vec![ArraySliceAxis::new(0, 2, 1), ArraySliceAxis::new(1, 2, 1)],
        };
        let index = ArrayReferenceViewTransform::Index { axis: 0, index: 1 };
        let handle = root.with_transform(slice.clone()).unwrap().with_transform(index.clone()).unwrap();

        // Composition derives each handle type incrementally, which must agree with folding the complete mapping
        // over the root type in one step.
        let view = ArrayReferenceView::root().with_transform_unchecked(slice).with_transform_unchecked(index);
        assert_eq!(root.r#type().as_ref(), &ReferenceType::new(root_type.clone()));
        assert_eq!(handle.r#type().as_ref(), &ReferenceType::new(view.output_type(&root_type).unwrap()));
        assert_eq!(handle.clone().r#type(), handle.r#type());
        assert_eq!(handle.to_string(), "ref<f32[2]>");
        assert_eq!(root.to_string(), "ref<f32[2, 3]>");
    }
}
