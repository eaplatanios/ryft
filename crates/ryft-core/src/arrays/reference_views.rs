//! Array-reference handles and their ordered, root-preserving index and slice mappings.
//!
//! # Symbolic Coordinates
//!
//! An [`Index`](ArrayReferenceViewTransform::Index) transform selects its coordinate either statically or through a
//! [`ViewSymbol`] ([`ViewIndex::Symbolic`]): a traced operand of the describing operation, or the iteration counter of
//! a region-carrying operation whose region input the view describes. A symbolic coordinate removes its axis exactly
//! like a static one, so the derived type is static, but it has no static selection: the operation that creates the
//! view resolves it, and the eager handles here (whose paths carry [`NoBinding`]) reject it up front. Refer to the
//! documentation of the [`views`](crate::programs::references) module for how the analysis closes symbols.

// TODO(eaplatanios): Review this module.

use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use ryft_macros::Parameter;
use thiserror::Error;

use crate::arrays::addressing::ArraySliceAxis;
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::dimensions::{Dimension, Shape, StaticShape};
use crate::arrays::types::ir::ArrayIrType;
use crate::batching::{BatchAxis, BatchingError};
use crate::operations::{Add, Reshape, Slice, UpdateSlice};
use crate::parameters::Parameter;
use crate::programs::{
    NoBinding, ProgramError, ReadyOrPendingReferenceGuard, Reference, ReferenceError, ReferenceId, ReferenceType,
    ReferenceView, ReferenceViewPath, ReferenceViewStep, Type, TypeError, TypeIdentityRenaming, Typed, Value,
    ViewOverlap, ViewSymbol, ViewSymbolBinding,
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

    /// A view with a symbolic coordinate was composed onto an eager handle, whose path carries only static steps.
    #[error("eager reference handles carry only static views; the operation that creates a symbolic view resolves it")]
    SymbolicViewCoordinate,
}

/// One validated coordinate transform in an [`ArrayReferenceView`]'s root-to-handle mapping.
///
/// A transform describes both directions of one view step: applying it extracts a selected child value from its
/// parent, while replacing that child reconstructs a value with exactly the parent's original type. This
/// bidirectional contract lets reference reads operate on the selected value and lets write-only replacements, swaps,
/// or additive updates reconstruct the shared root without changing its declared type. A write-only traversal
/// materializes the strict parents needed for reconstruction but deliberately skips extracting the overwritten leaf.
///
/// Transforms are interpreted in order from the root outward. [`Index`](Self::Index) removes one axis at a static or
/// symbolic coordinate; [`Slice`](Self::Slice) preserves rank and selects one static unit-stride range per axis.
/// Traced-operand indexing of arrays and strided slicing are intentionally not represented until their inverse update
/// semantics are supported.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
#[non_exhaustive]
pub enum ArrayReferenceViewTransform {
    /// Selects one coordinate from one axis and removes that axis from the view shape.
    Index {
        /// Axis selected in the transform's input view.
        axis: usize,

        /// Coordinate selected on `axis`.
        index: ViewIndex,
    },

    /// Selects one static unit-stride range on every axis while preserving rank.
    Slice {
        /// Per-axis selections in the transform's input view.
        axes: Vec<ArraySliceAxis>,
    },
}

/// Coordinate selected by an [`Index`](ArrayReferenceViewTransform::Index) transform.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum ViewIndex {
    /// A coordinate known when the view is described.
    Static(usize),

    /// A coordinate supplied by the named symbol, which the operation creating the view resolves.
    Symbolic(ViewSymbol),
}

// A transform depends on at most one symbol: the coordinate of a symbolic index.
impl ReferenceView for ArrayReferenceViewTransform {
    type Type = ArrayIrType;

    fn symbols(&self) -> Vec<ViewSymbol> {
        match self {
            Self::Index { index: ViewIndex::Symbolic(symbol), .. } => vec![*symbol],
            Self::Index { .. } | Self::Slice { .. } => Vec::new(),
        }
    }

    // The batch axis of a reference is an axis of its packed referent that the per-item view never sees. Indexing
    // removes one per-item axis, so the packed view cannot keep both axis positions unchanged: a batch axis at or
    // before the indexed axis shifts the packed indexed axis one position later while the output keeps the batch axis,
    // and a batch axis after the indexed axis leaves the packed indexed axis alone while the output's batch axis moves
    // one position earlier. Slicing preserves rank, so the packed view selects the complete batch axis through an
    // identity selection inserted at the batch axis position and the output keeps the batch axis.
    fn batch(&self, source: &ArrayIrType, batch_axis: BatchAxis) -> Result<(Self, BatchAxis), BatchingError> {
        let Some(axis) = batch_axis.axis() else {
            return Ok((self.clone(), batch_axis));
        };
        let referent = <&ReferenceType<ArrayType>>::try_from(source)?.referent();
        let position = axis.normalize(referent.rank())?;
        match self {
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

    // Both paths are folded to one coordinate per root axis and compared axis by axis: two static ranges that do not
    // intersect make the paths disjoint, identical static ranges and equal symbolic coordinates (same binding at the
    // same offset) are the same, and everything else may overlap. A malformed path cannot be folded and is treated as
    // possibly overlapping, because paths are validated when they are derived and this query must not fail.
    fn overlap(root: &ArrayIrType, a: &[ReferenceViewStep<Self>], b: &[ReferenceViewStep<Self>]) -> ViewOverlap {
        let Some(shape) =
            <&ReferenceType<ArrayType>>::try_from(root).ok().and_then(|root| root.referent().static_shape())
        else {
            return ViewOverlap::MayOverlap;
        };
        let (Some(a), Some(b)) = (RootCoordinate::fold(&shape, a), RootCoordinate::fold(&shape, b)) else {
            return ViewOverlap::MayOverlap;
        };
        let mut overlap = ViewOverlap::Same;
        for (a, b) in a.iter().zip(b.iter()) {
            match a.overlap(b) {
                ViewOverlap::Disjoint => return ViewOverlap::Disjoint,
                ViewOverlap::Same => {}
                ViewOverlap::MayOverlap => overlap = ViewOverlap::MayOverlap,
            }
        }
        overlap
    }
}

/// Coordinates that a folded [`ArrayReferenceView`] selects on one axis of its root, used by
/// [`ReferenceView::overlap`] to compare two paths of one root.
#[derive(Clone, Debug, PartialEq, Eq)]
enum RootCoordinate {
    /// A static unit-stride range `[start, limit)` of the root axis. Before any step touches the axis this is the
    /// complete axis, a slice narrows it, and a static index collapses it to one coordinate.
    Range {
        /// Inclusive start of the range.
        start: usize,

        /// Exclusive limit of the range.
        limit: usize,
    },

    /// One coordinate `offset + symbol` of the root axis, selected by a symbolic index whose value is `binding`
    /// relative to the range `[offset, ..)` that the earlier steps narrowed the axis to.
    Symbolic {
        /// Binding of the symbolic coordinate.
        binding: ViewSymbolBinding,

        /// Start of the narrowed range that the symbolic coordinate is relative to.
        offset: usize,
    },
}

impl RootCoordinate {
    /// Folds the closed `steps` of a path over a root of static shape `shape` into one coordinate per root axis, or
    /// [`None`] when the path is malformed for that root (an axis, coordinate, binding, or stride that the derivation
    /// would have rejected).
    fn fold(shape: &StaticShape, steps: &[ReferenceViewStep<ArrayReferenceViewTransform>]) -> Option<Vec<Self>> {
        let mut coordinates =
            shape.dimensions().iter().map(|size| Self::Range { start: 0, limit: *size }).collect::<Vec<_>>();
        // Root axes that the folded steps have not indexed away yet, in view axis order.
        let mut remaining = (0..shape.rank()).collect::<Vec<_>>();
        for step in steps {
            match step.view() {
                ArrayReferenceViewTransform::Index { axis, index } => {
                    if *axis >= remaining.len() {
                        return None;
                    }
                    let root_axis = remaining.remove(*axis);
                    let Self::Range { start, limit } = coordinates[root_axis] else {
                        return None;
                    };
                    coordinates[root_axis] = match index {
                        ViewIndex::Static(index) if start + *index < limit => {
                            Self::Range { start: start + *index, limit: start + *index + 1 }
                        }
                        ViewIndex::Static(_) => return None,
                        ViewIndex::Symbolic(_) => Self::Symbolic { binding: *step.bindings().first()?, offset: start },
                    };
                }
                ArrayReferenceViewTransform::Slice { axes } => {
                    if axes.len() != remaining.len() {
                        return None;
                    }
                    for (selection, root_axis) in axes.iter().zip(remaining.iter()) {
                        let Self::Range { start, limit } = coordinates[*root_axis] else {
                            return None;
                        };
                        let narrowed_start = start.checked_add(selection.start())?;
                        let narrowed_limit = narrowed_start.checked_add(selection.size())?;
                        if selection.stride() != 1 || narrowed_limit > limit {
                            return None;
                        }
                        coordinates[*root_axis] = Self::Range { start: narrowed_start, limit: narrowed_limit };
                    }
                }
            }
        }
        Some(coordinates)
    }

    /// Returns the relation between the coordinates that this and `other` select on one root axis.
    fn overlap(&self, other: &Self) -> ViewOverlap {
        match (self, other) {
            (Self::Range { start: a_start, limit: a_limit }, Self::Range { start: b_start, limit: b_limit }) => {
                if a_limit <= b_start || b_limit <= a_start {
                    ViewOverlap::Disjoint
                } else if a_start == b_start && a_limit == b_limit {
                    ViewOverlap::Same
                } else {
                    ViewOverlap::MayOverlap
                }
            }
            (
                Self::Symbolic { binding: a_binding, offset: a_offset },
                Self::Symbolic { binding: b_binding, offset: b_offset },
            ) if a_binding == b_binding => {
                if a_offset == b_offset {
                    ViewOverlap::Same
                } else {
                    ViewOverlap::Disjoint
                }
            }
            (Self::Symbolic { .. }, Self::Symbolic { .. })
            | (Self::Range { .. }, Self::Symbolic { .. })
            | (Self::Symbolic { .. }, Self::Range { .. }) => ViewOverlap::MayOverlap,
        }
    }
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
    /// Validates the axis of an [`Index`](Self::Index) transform against `input` and returns the static shape of
    /// `input`.
    fn indexed_shape(axis: usize, input: &ArrayType) -> Result<StaticShape, TypeError> {
        let shape = input.static_shape().ok_or_else(|| {
            TypeError::invalid(format!("reference indexing requires a static referent type but got `{input}`"))
        })?;
        if axis >= shape.rank() {
            return Err(TypeError::invalid(format!(
                "reference index axis {axis} is out of bounds for rank {}",
                shape.rank(),
            )));
        }
        Ok(shape)
    }

    /// Validates this transform against `input` and returns its normalized selection coordinates.
    fn selection(&self, input: &ArrayType) -> Result<ViewSelection, TypeError> {
        match self {
            Self::Index { axis, index } => {
                let shape = Self::indexed_shape(*axis, input)?;
                let index = match index {
                    ViewIndex::Static(index) => *index,
                    ViewIndex::Symbolic(_) => {
                        return Err(TypeError::invalid(
                            "a symbolic coordinate has no static selection; the operation that creates the view \
                             resolves it",
                        ));
                    }
                };
                if index >= shape.dimension(*axis) {
                    return Err(TypeError::invalid(format!(
                        "reference index {index} on axis {axis} is out of bounds for size {}",
                        shape.dimension(*axis),
                    )));
                }
                let mut starts = vec![0; shape.rank()];
                starts[*axis] = index;
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

    /// Returns the exact canonical array type produced from `input`. A symbolic coordinate removes its axis exactly
    /// like a static one, without the static bounds check and reconstruction proof, because the coordinate it selects
    /// is only known to the operation that creates the view.
    pub fn output_type(&self, input: &ArrayType) -> Result<ArrayType, TypeError> {
        if let Self::Index { axis, index: ViewIndex::Symbolic(_) } = self {
            Self::indexed_shape(*axis, input)?;
            return Ok(input.without_dimension(*axis)?.0);
        }
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

    /// Applies this transform to one carried parent value. A symbolic coordinate is resolved by the carrier from the
    /// one value the step's `bindings` close it over; a symbolic step that binds no value (an eager path, or a
    /// malformed closure) has no selection and is rejected by [`selection`](Self::selection).
    fn apply_in<C: ViewReadCarrier>(
        &self,
        carrier: &mut C,
        input: &C::Value,
        bindings: &[C::Binding],
    ) -> Result<C::Value, ProgramError> {
        if let (Self::Index { axis, index: ViewIndex::Symbolic(_) }, [binding]) = (self, bindings) {
            return carrier.index_symbolic(input, *axis, binding);
        }
        let selection = self.selection(carrier.array_type(input)?.as_ref())?;
        let sliced = carrier.slice(input, selection.starts, selection.limits)?;
        match selection.squeezed_output_shape {
            Some(shape) => carrier.reshape(&sliced, shape),
            None => Ok(sliced),
        }
    }

    /// Reconstructs the carried parent after replacing exactly the coordinates selected by this transform, resolving
    /// a symbolic coordinate exactly as [`apply_in`](Self::apply_in) does.
    fn replace_in<C: ViewWriteCarrier>(
        &self,
        carrier: &mut C,
        input: &C::Value,
        replacement: &C::Value,
        bindings: &[C::Binding],
    ) -> Result<C::Value, ProgramError> {
        if let (Self::Index { axis, index: ViewIndex::Symbolic(_) }, [binding]) = (self, bindings) {
            return carrier.update_index_symbolic(input, replacement, *axis, binding);
        }
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
/// Views currently support composed indexing (static, or symbolic where the binding admits it) and static unit-stride
/// slicing. `Binding` is what a symbolic coordinate is closed over: the array view overlay binds program identities
/// ([`ViewSymbolBinding`]), while eager handles only ever carry static steps and use [`NoBinding`]. Derived views
/// cannot themselves cross attached-region or external runtime state boundaries: pass the root handle across the
/// boundary and recreate the view within the destination scope.
pub type ArrayReferenceView<Binding = ViewSymbolBinding> = ReferenceViewPath<ArrayReferenceViewTransform, Binding>;

impl<Binding> ArrayReferenceView<Binding> {
    /// Returns the ordered transforms applied from the root outward, without their bindings.
    #[inline]
    pub fn transforms(&self) -> impl ExactSizeIterator<Item = &ArrayReferenceViewTransform> + DoubleEndedIterator {
        self.views()
    }

    /// Returns the exact view type derived from `root_type`.
    pub fn output_type(&self, root_type: &ArrayType) -> Result<ArrayType, TypeError> {
        self.transforms().try_fold(root_type.clone(), |r#type, transform| transform.output_type(&r#type))
    }

    /// Appends a static transform whose local input/output types were already validated by the caller.
    pub(crate) fn with_transform_unchecked(&self, transform: ArrayReferenceViewTransform) -> Self
    where
        Binding: Clone,
    {
        self.with_view(transform)
    }

    /// Materializes each parent-to-child snapshot once for update reconstruction, in root-to-view order starting
    /// with `root` itself. Each step's bindings are handed to `carrier`, which resolves symbolic coordinates.
    pub(crate) fn intermediates_in<C: ViewReadCarrier<Binding = Binding>>(
        &self,
        carrier: &mut C,
        root: C::Value,
    ) -> Result<Vec<C::Value>, ProgramError> {
        let mut intermediates = Vec::with_capacity(self.steps().len() + 1);
        intermediates.push(root);
        for step in self.steps() {
            let child = step.view().apply_in(carrier, intermediates.last().unwrap(), step.bindings())?;
            intermediates.push(child);
        }
        Ok(intermediates)
    }

    /// Reconstructs a root from precomputed view intermediates and a replacement leaf.
    pub(crate) fn reconstruct_in<C: ViewWriteCarrier<Binding = Binding>>(
        &self,
        carrier: &mut C,
        intermediates: &[C::Value],
        replacement: C::Value,
    ) -> Result<C::Value, ProgramError> {
        let steps = self.steps();
        if intermediates.len() != steps.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference view reconstruction requires {} parent snapshots but received {}",
                steps.len(),
                intermediates.len(),
            )));
        }
        let mut reconstructed = replacement;
        for (step, intermediate) in steps.iter().zip(intermediates).rev() {
            reconstructed = step.view().replace_in(carrier, intermediate, &reconstructed, step.bindings())?;
        }
        Ok(reconstructed)
    }

    /// Replaces this view's selected coordinates through `carrier`, returning their previous snapshot plus the
    /// reconstructed root, so that the eager swap and the discharge-time replacement share one traversal.
    pub(crate) fn swap_in<C: ViewWriteCarrier<Value: Clone, Binding = Binding>>(
        &self,
        carrier: &mut C,
        root: C::Value,
        replacement: C::Value,
    ) -> Result<(C::Value, C::Value), ProgramError> {
        let intermediates = self.intermediates_in(carrier, root)?;

        // The traversal always pushes the root itself first, so the chain is never empty and its last snapshot is
        // the value this view selects.
        let previous = intermediates.last().unwrap().clone();
        let reconstructed = self.reconstruct_in(carrier, &intermediates[..self.steps().len()], replacement)?;
        Ok((previous, reconstructed))
    }

    /// Replaces this view's selected coordinates through `carrier` without materializing the selected old value.
    ///
    /// Immutable root reconstruction still needs each strict parent of the selected leaf so coordinates outside the
    /// logical view survive. The final transform is deliberately not applied: its output is exactly the old selected
    /// value that write-only semantics must not observe. An identity view therefore returns `replacement` directly.
    pub(crate) fn write_in<C: ViewWriteCarrier<Binding = Binding>>(
        &self,
        carrier: &mut C,
        root: C::Value,
        replacement: C::Value,
    ) -> Result<C::Value, ProgramError> {
        let Some((_, parents)) = self.steps().split_last() else {
            return Ok(replacement);
        };
        let mut intermediates = Vec::with_capacity(self.steps().len());
        intermediates.push(root);
        for step in parents {
            let child = step.view().apply_in(carrier, intermediates.last().unwrap(), step.bindings())?;
            intermediates.push(child);
        }
        self.reconstruct_in(carrier, intermediates.as_slice(), replacement)
    }
}

impl ArrayReferenceView<NoBinding> {
    /// Applies the complete static mapping to one root snapshot.
    pub(crate) fn apply<A>(&self, root: &A) -> Result<A, ProgramError>
    where
        A: Value<Type = ArrayType> + Reshape + Slice,
    {
        let mut carrier = EagerViewCarrier(PhantomData);
        self.steps().iter().try_fold(root.clone(), |value, step| {
            step.view().apply_in(&mut carrier, &value, step.bindings())
        })
    }

    /// Replaces this static view and returns the reconstructed root plus its old view snapshot.
    pub(crate) fn swap<A>(&self, root: &A, replacement: &A) -> Result<(A, A), ProgramError>
    where
        A: Value<Type = ArrayType> + Reshape + Slice + UpdateSlice,
    {
        let (old, reconstructed) =
            self.swap_in(&mut EagerViewCarrier(PhantomData), root.clone(), replacement.clone())?;
        Ok((reconstructed, old))
    }
}

/// One value carrier through which a reference view maps between a shared root and one derived handle.
///
/// The root-to-view push-forward and the reverse update-slice reconstruction each exist exactly once, on
/// [`ArrayReferenceView`], generically over this carrier: the eager carrier operates on concrete values with the
/// array-manipulation capabilities, while reference discharge binds the identical operation sequence through its
/// destination context. Keeping one traversal guarantees the staged and eager semantics cannot drift apart. Static
/// steps lower to the carrier's slice and reshape; a symbolic index step hands the carrier the value its coordinate is
/// closed over, in the path's [`Binding`](Self::Binding).
pub(crate) trait ViewReadCarrier {
    /// Value representation carried through the traversal.
    type Value;

    /// What a symbolic coordinate of the traversed path is closed over.
    type Binding;

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

    /// Selects the coordinate that `binding` closes over on `axis` of `input` and removes that axis.
    fn index_symbolic(
        &mut self,
        input: &Self::Value,
        axis: usize,
        binding: &Self::Binding,
    ) -> Result<Self::Value, ProgramError>;
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

    /// Returns `target` with `update` written at the coordinate that `binding` closes over on `axis`, the inverse of
    /// [`index_symbolic`](ViewReadCarrier::index_symbolic).
    fn update_index_symbolic(
        &mut self,
        target: &Self::Value,
        update: &Self::Value,
        axis: usize,
        binding: &Self::Binding,
    ) -> Result<Self::Value, ProgramError>;
}

/// Stateless eager carrier over one concrete array value family. Eager paths carry only static steps, so the
/// symbolic-coordinate hooks are unreachable by type.
struct EagerViewCarrier<A>(PhantomData<A>);

impl<A: Value<Type = ArrayType> + Reshape + Slice> ViewReadCarrier for EagerViewCarrier<A> {
    type Value = A;
    type Binding = NoBinding;

    fn array_type<'c>(&'c self, value: &'c A) -> Result<Cow<'c, ArrayType>, ProgramError> {
        Ok(value.r#type())
    }

    fn slice(&mut self, input: &A, starts: Vec<usize>, limits: Vec<usize>) -> Result<A, ProgramError> {
        input.slice(starts.as_slice(), limits.as_slice(), &vec![1; starts.len()])
    }

    fn reshape(&mut self, input: &A, shape: Shape) -> Result<A, ProgramError> {
        input.reshape(shape)
    }

    fn index_symbolic(&mut self, _input: &A, _axis: usize, binding: &NoBinding) -> Result<A, ProgramError> {
        match *binding {}
    }
}

impl<A: Value<Type = ArrayType> + Reshape + Slice + UpdateSlice> ViewWriteCarrier for EagerViewCarrier<A> {
    fn update_slice(&mut self, target: &A, update: &A, starts: Vec<usize>) -> Result<A, ProgramError> {
        target.update_slice(update, starts.as_slice())
    }

    fn update_index_symbolic(
        &mut self,
        _target: &A,
        _update: &A,
        _axis: usize,
        binding: &NoBinding,
    ) -> Result<A, ProgramError> {
        match *binding {}
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

    /// Ordered mapping from the shared root to this handle's referent. Eager handles only ever carry static steps,
    /// so no symbol is ever bound on this path.
    view: ArrayReferenceView<NoBinding>,

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

    /// Returns a derived handle after appending `transform` without creating a resource. A transform with a symbolic
    /// coordinate is rejected with [`ArrayReferenceViewError::SymbolicViewCoordinate`]: an eager handle's path carries
    /// only static steps, and the coordinate is resolved by the operation that creates the view.
    pub fn with_transform(&self, transform: ArrayReferenceViewTransform) -> Result<Self, ProgramError> {
        if !transform.symbols().is_empty() {
            return Err(ProgramError::custom(ArrayReferenceViewError::SymbolicViewCoordinate));
        }
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
                .reconstruct_in(&mut carrier, &intermediates[..self.view.steps().len()], updated_view)
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
    use crate::arrays::types::dimensions::{DimensionBounds, DimensionVariable};
    use crate::axes::Axis;
    use crate::programs::{AtomId, ReferenceCompletion, ReferenceReplacementPreparation, RegionId, ValueId};

    use super::*;

    #[test]
    fn test_array_reference_view_transform_rejects_invalid_selections() {
        let matrix_type = ArrayType::new_static(DataType::F32, [3, 4]);
        let vector_type = ArrayType::new_static(DataType::F32, [3]);

        // Static indexing selects one existing coordinate on one existing axis; a symbolic coordinate still names an
        // existing axis.
        assert_eq!(
            ArrayReferenceViewTransform::Index { axis: 2, index: ViewIndex::Static(0) }.output_type(&matrix_type),
            Err(TypeError::invalid("reference index axis 2 is out of bounds for rank 2")),
        );
        assert_eq!(
            ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Static(3) }.output_type(&matrix_type),
            Err(TypeError::invalid("reference index 3 on axis 0 is out of bounds for size 3")),
        );
        assert_eq!(
            ArrayReferenceViewTransform::Index { axis: 2, index: ViewIndex::Symbolic(ViewSymbol::Iteration) }
                .output_type(&matrix_type),
            Err(TypeError::invalid("reference index axis 2 is out of bounds for rank 2")),
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
    fn test_array_reference_view_transform_symbolic_index() {
        let matrix_type = ArrayType::new_static(DataType::F32, [3, 4]);
        let symbolic =
            ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Symbolic(ViewSymbol::Operand(1)) };
        let r#static = ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Static(1) };
        let slice = ArrayReferenceViewTransform::Slice {
            axes: vec![ArraySliceAxis::new(0, 3, 1), ArraySliceAxis::new(0, 4, 1)],
        };

        // Only a symbolic index depends on a symbol, and it removes its axis exactly like a static one, including on
        // an axis that has no coordinate to check against.
        assert_eq!(symbolic.symbols(), vec![ViewSymbol::Operand(1)]);
        assert_eq!(r#static.symbols(), Vec::new());
        assert_eq!(slice.symbols(), Vec::new());
        assert_eq!(symbolic.output_type(&matrix_type), r#static.output_type(&matrix_type));
        assert_eq!(symbolic.output_type(&matrix_type), Ok(ArrayType::new_static(DataType::F32, [4])));
        assert_eq!(
            symbolic.output_type(&ArrayType::new_static(DataType::F32, [0, 4])),
            Ok(ArrayType::new_static(DataType::F32, [4])),
        );

        // A symbolic coordinate has no static selection, so neither an eager traversal nor an eager handle can carry
        // it: the operation that creates the view resolves the coordinate.
        let view: ArrayReferenceView<NoBinding> = ArrayReferenceView::root().with_transform_unchecked(symbolic.clone());
        assert_eq!(
            view.apply(&Array::matrix(3, 4, (1..=12).map(|value| value as f32).collect())),
            Err(TypeError::invalid(
                "a symbolic coordinate has no static selection; the operation that creates the view resolves it",
            )
            .into()),
        );
        let root = ArrayReference::new(Array::matrix(3, 4, (1..=12).map(|value| value as f32).collect()));
        let error = root.with_transform(symbolic).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ArrayReferenceViewError>(),
            Some(&ArrayReferenceViewError::SymbolicViewCoordinate),
        );
        assert_eq!(
            error.to_string(),
            "eager reference handles carry only static views; the operation that creates a symbolic view resolves it",
        );
    }

    #[test]
    fn test_array_reference_view_transform_batch() {
        let packed = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2, 3, 4])));
        let index = ArrayReferenceViewTransform::Index { axis: 1, index: ViewIndex::Static(2) };
        let slice = ArrayReferenceViewTransform::Slice {
            axes: vec![ArraySliceAxis::new(1, 1, 1), ArraySliceAxis::new(1, 2, 1)],
        };

        // A batch axis at or before the indexed axis shifts the packed indexed axis one position later and the output
        // keeps the batch axis, while a batch axis after the indexed axis leaves the packed indexed axis alone and the
        // output batch axis moves one position earlier. Negative batch axes normalize against the packed rank.
        assert_eq!(
            index.batch(&packed, BatchAxis::new(0)),
            Ok((ArrayReferenceViewTransform::Index { axis: 2, index: ViewIndex::Static(2) }, BatchAxis::new(0))),
        );
        assert_eq!(
            index.batch(&packed, BatchAxis::new(1)),
            Ok((ArrayReferenceViewTransform::Index { axis: 2, index: ViewIndex::Static(2) }, BatchAxis::new(1))),
        );
        assert_eq!(
            index.batch(&packed, BatchAxis::new(2)),
            Ok((ArrayReferenceViewTransform::Index { axis: 1, index: ViewIndex::Static(2) }, BatchAxis::new(1))),
        );
        assert_eq!(
            index.batch(&packed, BatchAxis::new(-1)),
            Ok((ArrayReferenceViewTransform::Index { axis: 1, index: ViewIndex::Static(2) }, BatchAxis::new(1))),
        );

        // The symbol of a symbolic coordinate rides along untouched.
        let symbolic =
            ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Symbolic(ViewSymbol::Operand(1)) };
        assert_eq!(
            symbolic.batch(&packed, BatchAxis::new(0)),
            Ok((
                ArrayReferenceViewTransform::Index { axis: 1, index: ViewIndex::Symbolic(ViewSymbol::Operand(1)) },
                BatchAxis::new(0),
            )),
        );

        // Slicing inserts the complete batch axis at the batch axis position and keeps the batch axis.
        assert_eq!(
            slice.batch(&packed, BatchAxis::new(1)),
            Ok((
                ArrayReferenceViewTransform::Slice {
                    axes: vec![
                        ArraySliceAxis::new(1, 1, 1),
                        ArraySliceAxis::new(0, 3, 1),
                        ArraySliceAxis::new(1, 2, 1)
                    ],
                },
                BatchAxis::new(1),
            )),
        );

        // A replicated source leaves both transforms unchanged and replicated.
        assert_eq!(index.batch(&packed, BatchAxis::replicated()), Ok((index.clone(), BatchAxis::replicated())));
        assert_eq!(slice.batch(&packed, BatchAxis::replicated()), Ok((slice.clone(), BatchAxis::replicated())));

        // The source must be a reference whose packed referent has the batch axis, and a static identity slice cannot
        // span a dynamically sized batch axis.
        let array = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2, 3, 4]));
        assert!(matches!(index.batch(&array, BatchAxis::new(0)), Err(BatchingError::Type(_))));
        assert!(matches!(index.batch(&packed, BatchAxis::new(3)), Err(BatchingError::Axis(_))));
        let batch = DimensionVariable::new("batch", DimensionBounds::unbounded());
        let dynamic = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(3)]));
        assert_eq!(
            ArrayReferenceViewTransform::Slice { axes: vec![ArraySliceAxis::new(0, 3, 1)] }
                .batch(&ArrayIrType::Reference(ReferenceType::new(dynamic.clone())), BatchAxis::new(0)),
            Err(BatchingError::DynamicBatchAxis { r#type: Box::new(dynamic), axis: Axis::from(0) }),
        );
    }

    #[test]
    fn test_array_reference_view_transform_overlap() {
        let root = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [4, 3])));
        let empty: ArrayReferenceView = ArrayReferenceView::root();
        let rows_0_1 = empty.with_view(ArrayReferenceViewTransform::Slice {
            axes: vec![ArraySliceAxis::new(0, 2, 1), ArraySliceAxis::new(0, 3, 1)],
        });
        let rows_1_2 = empty.with_view(ArrayReferenceViewTransform::Slice {
            axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 3, 1)],
        });
        let rows_2_3 = empty.with_view(ArrayReferenceViewTransform::Slice {
            axes: vec![ArraySliceAxis::new(2, 2, 1), ArraySliceAxis::new(0, 3, 1)],
        });
        let row_1 = empty.with_view(ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Static(1) });
        let column_0 = empty.with_view(ArrayReferenceViewTransform::Index { axis: 1, index: ViewIndex::Static(0) });

        // Static coordinates fold to one range per root axis: disjoint ranges on any axis make the paths disjoint,
        // identical ranges on every axis make them the same, and intersecting ranges may overlap. The trait function
        // and the path method agree.
        assert_eq!(
            ArrayReferenceViewTransform::overlap(&root, rows_0_1.steps(), rows_2_3.steps()),
            ViewOverlap::Disjoint,
        );
        assert_eq!(rows_0_1.overlap(&rows_2_3, &root), ViewOverlap::Disjoint);
        assert_eq!(row_1.overlap(&row_1, &root), ViewOverlap::Same);
        assert_eq!(rows_0_1.overlap(&rows_1_2, &root), ViewOverlap::MayOverlap);
        assert_eq!(row_1.overlap(&rows_0_1, &root), ViewOverlap::MayOverlap);
        assert_eq!(row_1.overlap(&rows_2_3, &root), ViewOverlap::Disjoint);

        // Rank changes are tracked while folding: an index removes its axis, so a slice that follows it addresses the
        // remaining root axes, and different step sequences that select the same coordinates are the same.
        let row_1_columns_1_2 =
            row_1.with_view(ArrayReferenceViewTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] });
        let row_1_column_1 =
            row_1.with_view(ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Static(1) });
        let rows_1_columns_1_2_row_0 = empty
            .with_view(ArrayReferenceViewTransform::Slice {
                axes: vec![ArraySliceAxis::new(1, 1, 1), ArraySliceAxis::new(1, 2, 1)],
            })
            .with_view(ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Static(0) });
        assert_eq!(row_1_columns_1_2.overlap(&column_0, &root), ViewOverlap::Disjoint);
        assert_eq!(row_1_columns_1_2.overlap(&row_1, &root), ViewOverlap::MayOverlap);
        assert_eq!(row_1_columns_1_2.overlap(&row_1_column_1, &root), ViewOverlap::MayOverlap);
        assert_eq!(row_1_columns_1_2.overlap(&rows_1_columns_1_2_row_0, &root), ViewOverlap::Same);
        assert_eq!(row_1_column_1.overlap(&column_0, &root), ViewOverlap::Disjoint);

        // The complete root is the same as itself and as a slice spanning every axis, and may overlap with any
        // narrowing path.
        let complete = empty.with_view(ArrayReferenceViewTransform::Slice {
            axes: vec![ArraySliceAxis::new(0, 4, 1), ArraySliceAxis::new(0, 3, 1)],
        });
        assert_eq!(empty.overlap(&empty, &root), ViewOverlap::Same);
        assert_eq!(empty.overlap(&complete, &root), ViewOverlap::Same);
        assert_eq!(empty.overlap(&rows_0_1, &root), ViewOverlap::MayOverlap);
        assert_eq!(empty.overlap(&row_1_column_1, &root), ViewOverlap::MayOverlap);

        // Symbolic coordinates compare by binding: the same binding at the same offset is the same coordinate, the
        // same binding relative to differently narrowed axes is provably different, and different bindings or a
        // binding against a static coordinate may overlap.
        let symbolic =
            ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Symbolic(ViewSymbol::Operand(1)) };
        let first = ViewSymbolBinding::Value(ValueId::new(RegionId::new(0), AtomId::new(1)));
        let second = ViewSymbolBinding::Value(ValueId::new(RegionId::new(0), AtomId::new(2)));
        let iteration = ViewSymbolBinding::Iteration(RegionId::new(1));
        let row_first = empty.with_step(symbolic.clone(), vec![first]);
        let row_second = empty.with_step(symbolic.clone(), vec![second]);
        let row_iteration = empty.with_step(symbolic.clone(), vec![iteration]);
        let shifted_row_first = rows_1_2.with_step(symbolic.clone(), vec![first]);
        assert_eq!(row_first.overlap(&empty.with_step(symbolic.clone(), vec![first]), &root), ViewOverlap::Same);
        assert_eq!(row_first.overlap(&row_second, &root), ViewOverlap::MayOverlap);
        assert_eq!(row_first.overlap(&row_iteration, &root), ViewOverlap::MayOverlap);
        assert_eq!(row_first.overlap(&row_1, &root), ViewOverlap::MayOverlap);
        assert_eq!(row_first.overlap(&rows_2_3, &root), ViewOverlap::MayOverlap);
        assert_eq!(row_first.overlap(&empty, &root), ViewOverlap::MayOverlap);
        assert_eq!(row_first.overlap(&shifted_row_first, &root), ViewOverlap::Disjoint);
        assert_eq!(
            row_first
                .with_view(ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Static(0) })
                .overlap(
                    &row_second.with_view(ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Static(2) }),
                    &root
                ),
            ViewOverlap::Disjoint,
        );

        // A path or root that cannot be folded (an out-of-bounds axis or coordinate, a symbolic step without its
        // binding, a non-reference root, or a root without a static shape) is conservatively reported as possibly
        // overlapping rather than failing.
        let out_of_bounds =
            empty.with_view(ArrayReferenceViewTransform::Index { axis: 2, index: ViewIndex::Static(0) });
        let unbound = empty.with_view(symbolic);
        assert_eq!(out_of_bounds.overlap(&rows_2_3, &root), ViewOverlap::MayOverlap);
        assert_eq!(
            empty
                .with_view(ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Static(4) })
                .overlap(&rows_2_3, &root),
            ViewOverlap::MayOverlap,
        );
        assert_eq!(unbound.overlap(&rows_2_3, &root), ViewOverlap::MayOverlap);
        assert_eq!(
            rows_0_1.overlap(&rows_2_3, &ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4, 3]))),
            ViewOverlap::MayOverlap,
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
            ViewOverlap::MayOverlap,
        );
    }

    #[test]
    fn test_array_reference_view_composition() {
        let root_type = ArrayType::new_static(DataType::F32, [3, 4]);
        let root: ArrayReferenceView = ArrayReferenceView::root();
        assert!(root.is_root());
        assert_eq!(root.transforms().len(), 0);
        assert_eq!(root.output_type(&root_type), Ok(root_type.clone()));

        // Each transform applies to the preceding view, so the slice narrows both axes and the index then removes
        // the leading axis of the already-narrowed view.
        let slice = ArrayReferenceViewTransform::Slice {
            axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 3, 1)],
        };
        let index = ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Static(1) };
        let sliced = root.with_transform_unchecked(slice.clone());
        let indexed = sliced.with_transform_unchecked(index.clone());
        assert!(!sliced.is_root());
        assert_eq!(sliced.transforms().collect::<Vec<_>>(), vec![&slice]);
        assert_eq!(sliced.output_type(&root_type), Ok(ArrayType::new_static(DataType::F32, [2, 3])));
        assert_eq!(indexed.transforms().collect::<Vec<_>>(), vec![&slice, &index]);
        assert_eq!(indexed.output_type(&root_type), Ok(ArrayType::new_static(DataType::F32, [3])));

        // Composition validates each appended transform against the preceding view's derived type, so an
        // out-of-bounds coordinate of the derived view is rejected even though it exists in the root.
        let handle = ArrayReference::new(Array::matrix(3, 4, (1..=12).map(|value| value as f32).collect()))
            .with_transform(slice.clone())
            .unwrap();
        assert_eq!(
            handle
                .with_transform(ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Static(2) })
                .unwrap_err(),
            TypeError::invalid("reference index 2 on axis 0 is out of bounds for size 2").into(),
        );

        // Equality and hashing distinguish transform sequences, including the two orders of the same two transforms.
        let reversed = root
            .with_transform_unchecked(ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Static(1) })
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
            .with_transform(ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Static(1) })
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
        let composed = derived
            .with_transform(ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Static(0) })
            .unwrap();
        assert_eq!(composed.r#type().as_ref(), &ReferenceType::new(ArrayType::scalar(DataType::F32)));
        assert_eq!(composed.read().unwrap_err().downcast_custom::<ReferenceError>(), Some(&poisoned));

        let frozen = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]));
        assert_eq!(frozen.freeze(), Ok(Array::vector(vec![1.0_f32, 2.0])));
        let frozen_view = frozen
            .with_transform(ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Static(1) })
            .unwrap();
        assert_eq!(frozen_view.read().unwrap_err().downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_array_reference_view_reconstruction_validates_parent_count() {
        let view: ArrayReferenceView<NoBinding> = ArrayReferenceView::root()
            .with_transform_unchecked(ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Static(0) });
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
        let index = ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Static(1) };
        let handle = root.with_transform(slice.clone()).unwrap().with_transform(index.clone()).unwrap();

        // Composition derives each handle type incrementally, which must agree with folding the complete mapping
        // over the root type in one step.
        let view: ArrayReferenceView =
            ArrayReferenceView::root().with_transform_unchecked(slice).with_transform_unchecked(index);
        assert_eq!(root.r#type().as_ref(), &ReferenceType::new(root_type.clone()));
        assert_eq!(handle.r#type().as_ref(), &ReferenceType::new(view.output_type(&root_type).unwrap()));
        assert_eq!(handle.clone().r#type(), handle.r#type());
        assert_eq!(handle.to_string(), "ref<f32[2]>");
        assert_eq!(root.to_string(), "ref<f32[2, 3]>");
    }
}
