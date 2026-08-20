//! Array-reference handles and their ordered, root-preserving index and slice mappings.

// TODO(eaplatanios): Review this module.

use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};

use ryft_macros::Parameter;
use thiserror::Error;

use crate::arrays::addressing::ArraySliceAxis;
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::dimensions::{Dimension, Shape};
use crate::operations::{Add, Reshape, Slice, UpdateSlice};
use crate::parameters::Parameter;
use crate::programs::{
    ProgramError, Reference, ReferenceGuard, ReferenceId, ReferenceType, Type, TypeError, TypeIdentityRenaming, Typed,
    Value,
};

/// Error produced by an invalid eager array-reference view operation.
#[derive(Clone, Debug, Error, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ArrayReferenceViewError {
    /// A consuming freeze was attempted through a derived view instead of the root handle.
    #[error("cannot freeze a reference view; freeze the root reference instead")]
    CannotFreezeView,

    /// A derived view was read through the bound-free root-only accessor.
    #[error("cannot read a reference view directly; use array reference operations instead")]
    CannotReadViewDirectly,

    /// A derived or identity-renamed handle was used as a backend root-state transaction boundary.
    #[error("reference runtime transactions require an unrenamed root handle")]
    InvalidRuntimeRoot,
}

/// One root-preserving coordinate transform in an [`ArrayReferenceView`].
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

impl ArrayReferenceViewTransform {
    /// Returns the exact canonical array type produced from `input`.
    pub fn output_type(&self, input: &ArrayType) -> Result<ArrayType, TypeError> {
        let output = match self {
            Self::Index { axis, index } => {
                let shape = input.static_shape().ok_or_else(|| {
                    TypeError::invalid(format!("reference indexing requires a static referent type but got `{input}`",))
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
                let sliced = input
                    .slice(starts.as_slice(), limits.as_slice(), &vec![1; shape.rank()])
                    .map_err(|error| TypeError::invalid(error.to_string()))?;
                let output_shape = Shape::new(
                    shape
                        .dimensions()
                        .iter()
                        .enumerate()
                        .filter_map(|(candidate, size)| (candidate != *axis).then_some(Dimension::Static(*size)))
                        .collect(),
                );
                sliced.reshape(output_shape).map_err(|error| TypeError::invalid(error.to_string()))?
            }
            Self::Slice { axes } => {
                let shape = input.static_shape().ok_or_else(|| {
                    TypeError::invalid(format!("reference slicing requires a static referent type but got `{input}`",))
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
                        TypeError::invalid(format!("reference slice limit overflows usize on axis {axis}"))
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
                input
                    .slice(starts.as_slice(), limits.as_slice(), &vec![1; shape.rank()])
                    .map_err(|error| TypeError::invalid(error.to_string()))?
            }
        };
        self.validate_reconstruction(input, &output)?;
        Ok(output)
    }

    /// Applies this transform to one immutable value snapshot.
    pub(crate) fn apply<A>(&self, input: &A) -> Result<A, ProgramError>
    where
        A: Value<Type = ArrayType> + Reshape + Slice,
    {
        match self {
            Self::Index { axis, index } => {
                let shape = input.r#type().static_shape().unwrap();
                let mut starts = vec![0; shape.rank()];
                starts[*axis] = *index;
                let mut limits = shape.dimensions().to_vec();
                limits[*axis] = index + 1;
                let sliced = input.slice(starts.as_slice(), limits.as_slice(), &vec![1; shape.rank()])?;
                sliced.reshape(Shape::new(
                    shape
                        .dimensions()
                        .iter()
                        .enumerate()
                        .filter_map(|(candidate, size)| (candidate != *axis).then_some(Dimension::Static(*size)))
                        .collect(),
                ))
            }
            Self::Slice { axes } => {
                let starts = axes.iter().map(|axis| axis.start()).collect::<Vec<_>>();
                let limits = axes.iter().map(|axis| axis.start() + axis.size()).collect::<Vec<_>>();
                input.slice(starts.as_slice(), limits.as_slice(), &vec![1; axes.len()])
            }
        }
    }

    /// Reconstructs `input` after replacing exactly the coordinates selected by this transform.
    pub(crate) fn replace<A>(&self, input: &A, replacement: &A) -> Result<A, ProgramError>
    where
        A: Value<Type = ArrayType> + Reshape + UpdateSlice,
    {
        match self {
            Self::Index { axis, index } => {
                let input_shape = input.r#type().static_shape().unwrap();
                let mut update_shape = input_shape.dimensions().to_vec();
                update_shape[*axis] = 1;
                let replacement =
                    replacement.reshape(Shape::new(update_shape.into_iter().map(Dimension::Static).collect()))?;
                let mut starts = vec![0; input_shape.rank()];
                starts[*axis] = *index;
                input.update_slice(&replacement, starts.as_slice())
            }
            Self::Slice { axes } => {
                let starts = axes.iter().map(|axis| axis.start()).collect::<Vec<_>>();
                input.update_slice(replacement, starts.as_slice())
            }
        }
    }

    /// Proves that updating the selected child reconstructs the exact parent storage type.
    fn validate_reconstruction(&self, input: &ArrayType, output: &ArrayType) -> Result<(), TypeError> {
        let reconstructed = match self {
            Self::Index { axis, .. } => {
                let input_shape = input.static_shape().unwrap();
                let mut update_shape = input_shape.dimensions().to_vec();
                update_shape[*axis] = 1;
                let update = output
                    .reshape(Shape::new(update_shape.into_iter().map(Dimension::Static).collect()))
                    .map_err(|error| TypeError::invalid(error.to_string()))?;
                input
                    .update_slice(&update, &vec![0; input.rank()])
                    .map_err(|error| TypeError::invalid(error.to_string()))?
            }
            Self::Slice { axes } => input
                .update_slice(output, &axes.iter().map(|axis| axis.start()).collect::<Vec<_>>())
                .map_err(|error| TypeError::invalid(error.to_string()))?,
        };
        if &reconstructed == input {
            return Ok(());
        }
        Err(TypeError::invalid(format!(
            "reference view reconstruction changes root type from `{input}` to `{reconstructed}`",
        )))
    }
}

/// Ordered coordinate mapping from one array-reference handle back to its shared root.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Parameter)]
pub struct ArrayReferenceView {
    /// Validated transforms applied from the root outward.
    transforms: Vec<ArrayReferenceViewTransform>,
}

impl ArrayReferenceView {
    /// Returns an identity view of a root reference.
    #[inline]
    pub const fn root() -> Self {
        Self { transforms: Vec::new() }
    }

    /// Returns the ordered transforms applied from the root outward.
    #[inline]
    pub fn transforms(&self) -> &[ArrayReferenceViewTransform] {
        self.transforms.as_slice()
    }

    /// Returns whether this mapping denotes the complete root.
    #[inline]
    pub fn is_root(&self) -> bool {
        self.transforms.is_empty()
    }

    /// Returns the exact view type derived from `root_type`.
    pub fn output_type(&self, root_type: &ArrayType) -> Result<ArrayType, TypeError> {
        self.transforms
            .iter()
            .try_fold(root_type.clone(), |r#type, transform| transform.output_type(&r#type))
    }

    /// Returns this mapping with `transform` appended after validating the complete composition.
    pub fn with_transform(
        &self,
        root_type: &ArrayType,
        transform: ArrayReferenceViewTransform,
    ) -> Result<Self, TypeError> {
        transform.output_type(&self.output_type(root_type)?)?;
        let mut transforms = self.transforms.clone();
        transforms.push(transform);
        Ok(Self { transforms })
    }

    /// Appends a transform whose local input/output types were validated by its owning operation.
    pub(crate) fn with_validated_transform(&self, transform: ArrayReferenceViewTransform) -> Self {
        let mut transforms = self.transforms.clone();
        transforms.push(transform);
        Self { transforms }
    }

    /// Applies the complete mapping to one root snapshot.
    pub(crate) fn apply<A>(&self, root: &A) -> Result<A, ProgramError>
    where
        A: Value<Type = ArrayType> + Reshape + Slice,
    {
        self.transforms.iter().try_fold(root.clone(), |value, transform| transform.apply(&value))
    }

    /// Replaces this view and returns the reconstructed root plus its old view snapshot.
    pub(crate) fn swap<A>(&self, root: &A, replacement: &A) -> Result<(A, A), ProgramError>
    where
        A: Value<Type = ArrayType> + Reshape + Slice + UpdateSlice,
    {
        let intermediates = self.intermediates(root)?;
        let old = intermediates.last().unwrap().clone();
        let reconstructed = self.reconstruct(intermediates.as_slice(), replacement)?;
        Ok((reconstructed, old))
    }

    /// Materializes each parent-to-child snapshot once for update reconstruction.
    fn intermediates<A>(&self, root: &A) -> Result<Vec<A>, ProgramError>
    where
        A: Value<Type = ArrayType> + Reshape + Slice,
    {
        let mut intermediates = Vec::with_capacity(self.transforms.len() + 1);
        intermediates.push(root.clone());
        for transform in &self.transforms {
            let child = transform.apply(intermediates.last().unwrap())?;
            intermediates.push(child);
        }
        Ok(intermediates)
    }

    /// Reconstructs a root from precomputed view intermediates and a replacement leaf.
    fn reconstruct<A>(&self, intermediates: &[A], replacement: &A) -> Result<A, ProgramError>
    where
        A: Value<Type = ArrayType> + Reshape + UpdateSlice,
    {
        let mut reconstructed = replacement.clone();
        for transform_index in (0..self.transforms.len()).rev() {
            reconstructed =
                self.transforms[transform_index].replace(&intermediates[transform_index], &reconstructed)?;
        }
        Ok(reconstructed)
    }
}

/// Eager array-reference handle pairing one shared root holder with handle-local view metadata.
pub struct ArrayReference<A: Value<Type = ArrayType>> {
    /// Shared identity-bearing root holder.
    root: Reference<A>,

    /// Ordered mapping from the shared root to this handle's referent.
    view: ArrayReferenceView,
}

impl<A: Value<Type = ArrayType>> ArrayReference<A> {
    /// Creates a new root reference initialized with `value`.
    #[inline]
    pub fn new(value: A) -> Self {
        Self { root: Reference::new(value), view: ArrayReferenceView::root() }
    }

    /// Returns this shared holder's process-local identity.
    #[inline]
    pub fn id(&self) -> ReferenceId {
        self.root.id()
    }

    /// Returns this handle's root-relative view mapping.
    #[inline]
    pub fn view(&self) -> &ArrayReferenceView {
        &self.view
    }

    /// Returns whether this handle denotes the complete root value.
    #[inline]
    pub fn is_root(&self) -> bool {
        self.view.is_root()
    }

    /// Returns whether this is an unrenamed root handle accepted at a backend runtime state boundary.
    #[doc(hidden)]
    #[inline]
    pub fn is_runtime_root_handle(&self) -> bool {
        self.view.is_root() && self.root.is_root_handle()
    }

    /// Locks an unrenamed root for one backend-owned state transaction.
    #[doc(hidden)]
    pub fn lock_root(&self) -> Result<ReferenceGuard<'_, A>, ProgramError> {
        if !self.is_runtime_root_handle() {
            return Err(ProgramError::custom(ArrayReferenceViewError::InvalidRuntimeRoot));
        }
        self.root.lock().map_err(ProgramError::custom)
    }

    /// Returns a derived handle after appending `transform` without creating a resource.
    pub fn with_transform(&self, transform: ArrayReferenceViewTransform) -> Result<Self, ProgramError> {
        self.root.validate_live().map_err(ProgramError::custom)?;
        let root_type = self.root.r#type();
        let view = self.view.with_transform(root_type.referent(), transform)?;
        Ok(Self { root: self.root.clone(), view })
    }

    /// Returns an immutable root snapshot without requiring array-manipulation capabilities.
    pub fn read(&self) -> Result<A, ProgramError> {
        if !self.view.is_root() {
            return Err(ProgramError::custom(ArrayReferenceViewError::CannotReadViewDirectly));
        }
        self.root.read().map_err(ProgramError::custom)
    }

    /// Returns an immutable snapshot of this handle's selected coordinates.
    pub(crate) fn read_view(&self) -> Result<A, ProgramError>
    where
        A: Reshape + Slice,
    {
        self.view.apply(&self.root.read().map_err(ProgramError::custom)?)
    }

    /// Replaces this handle's selected coordinates and returns their previous snapshot.
    pub fn swap(&self, replacement: A) -> Result<A, ProgramError>
    where
        A: Reshape + Slice + UpdateSlice,
    {
        if self.view.is_root() {
            return self.root.swap(replacement).map_err(ProgramError::custom);
        }
        self.root.update_with_result(|current| {
            let (updated, old) = self.view.swap(current, &replacement)?;
            Ok((updated, old))
        })
    }

    /// Adds `update` into this handle's selected coordinates.
    pub fn add_update(&self, update: &A) -> Result<(), ProgramError>
    where
        A: Add + Reshape + Slice + UpdateSlice,
    {
        if self.view.is_root() {
            return self.root.update_with(|current| current.add(update));
        }
        self.root.update_with(|current| {
            let intermediates = self.view.intermediates(current)?;
            let current_view = intermediates.last().unwrap();
            let updated_view = current_view.add(update)?;
            self.view.reconstruct(intermediates.as_slice(), &updated_view)
        })
    }

    /// Consumes a root handle, rejecting every derived view without changing shared state.
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
        self.view.output_type(root.r#type().referent())?;
        Ok(Self { root, view: self.view.clone() })
    }
}

impl<A: Value<Type = ArrayType>> Clone for ArrayReference<A> {
    #[inline]
    fn clone(&self) -> Self {
        Self { root: self.root.clone(), view: self.view.clone() }
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

impl<A: Value<Type = ArrayType>> Typed for ArrayReference<A> {
    type Type = ReferenceType<ArrayType>;

    fn r#type(&self) -> Cow<'_, Self::Type> {
        Cow::Owned(ReferenceType::new(self.view.output_type(self.root.r#type().referent()).unwrap()))
    }
}
