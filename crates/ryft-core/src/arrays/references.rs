//! Array reference handles, index mappings, analysis, and discharge for the array IR.
//!
//! [`ArrayReferenceView`] describes array indexing and slicing. [`ArrayReferenceViewPath`] composes those transforms
//! into a mapping from a root array to a selected view, and [`ArrayReference`] pairs a static mapping with an eager
//! reference allocation. Reads select the elements at the mapped indices; mutations reconstruct the root through the
//! same transforms in reverse order, preserving values outside the view.
//!
//! [`ArrayReferenceAnalysis`] specializes the generic view analysis for these mappings. [`ArrayReferenceDischarge`]
//! uses the same traversal as eager handles to express reads and updates as immutable array operations in a context.
//! Sharing this traversal keeps eager view access and discharged array programs consistent.
//!
//! The [program reference module](crate::programs::references) owns reference identity, lifetime and alias validation,
//! symbolic path storage, and generic analysis and discharge. This module supplies array shapes, indices, eager array
//! handles, and array-IR reconstruction; it does not maintain a second reference analysis or state interpreter.
//!
//! # Symbolic Indices
//!
//! An [`Index`](ArrayReferenceView::Index) transform indexes one array axis using a static index or a
//! symbolic input position. Analysis binds each position to the [`ValueId`] of the corresponding instruction operand.
//! Eager handles carry [`NoReferenceViewBinding`] and accept only static transforms. Discharge paths can store context
//! values as bindings and reconstruct symbolic selections through dynamic slicing and updates. Runtime indices clamp
//! to the selected axis, following the array dynamic-slicing contract.

use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use ryft_macros::Parameter;
use thiserror::Error;

use crate::arrays::addressing::ArraySliceAxis;
use crate::arrays::operations::ArrayReferenceViewOperation;
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
    BatchableReferenceView, NoReferenceViewBinding, ProgramError, ReadyOrPendingReferenceGuard, Reference,
    ReferenceAccumulationPolicy, ReferenceDischargePolicy, ReferenceDischargeableType, ReferenceError, ReferenceId,
    ReferenceType, ReferenceView, ReferenceViewAnalysis, ReferenceViewOverlap, ReferenceViewPath, ReferenceViewStep,
    Type, TypeError, TypeIdentityRenaming, Typed, Value, ValueId,
};

// TODO(eaplatanios): Review this module.

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

    /// A view with a symbolic index was composed onto an eager handle, whose path carries only static steps.
    #[error("eager reference handles carry only static views; the operation that creates a symbolic view resolves it")]
    SymbolicViewIndex,
}

/// One validated index transform in an [`ArrayReferenceViewPath`]'s root-to-handle mapping.
///
/// A transform describes both directions of one view step: applying it extracts a selected child value from its
/// parent, while replacing that child reconstructs a value with exactly the parent's original type. This
/// bidirectional contract lets reference reads operate on the selected value and lets write-only replacements, swaps,
/// or additive updates reconstruct the shared root without changing its declared type. A write-only traversal
/// materializes the strict parents needed for reconstruction but deliberately skips extracting the overwritten leaf.
///
/// Transforms are interpreted in order from the root outward. [`Index`](Self::Index) removes one axis at a static or
/// symbolic index; [`Slice`](Self::Slice) preserves rank and selects one static unit-stride range per axis. The
/// built-in scan supplies its explicit body index to the dynamic reference-indexing operation. Eager handles
/// resolve runtime indices before creating static views. Discharge reconstructs symbolic indices with dynamic slicing;
/// strided slicing remains unsupported.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
#[non_exhaustive]
pub enum ArrayReferenceView {
    /// Selects a position along one axis and removes that axis from the view shape.
    Index {
        /// Axis selected in the transform's input view.
        axis: usize,

        /// Index selected on `axis`.
        index: ArrayReferenceViewIndex,
    },

    /// Selects one static unit-stride range on every axis while preserving rank.
    Slice {
        /// Per-axis selections in the transform's input view.
        axes: Vec<ArraySliceAxis>,
    },
}

/// Index selected by an [`Index`](ArrayReferenceView::Index) transform.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum ArrayReferenceViewIndex {
    /// An index known when the view is described.
    Static(usize),

    /// An index supplied by the instruction input at this position. Analysis binds it to that input's value.
    Symbolic(usize),
}

impl ArrayReferenceView {
    /// Returns the exact canonical array type produced from `input`. A symbolic index removes its axis exactly like a
    /// static one, without the static bounds check and reconstruction proof, because the index it selects is only known
    /// to the operation that creates the view.
    pub fn output_type(&self, input: &ArrayType) -> Result<ArrayType, TypeError> {
        if let Self::Index { axis, index: ArrayReferenceViewIndex::Symbolic(_) } = self {
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

    /// Validates this transform against `input` and returns its normalized selection indices.
    fn selection(&self, input: &ArrayType) -> Result<ViewSelection, TypeError> {
        match self {
            Self::Index { axis, index } => {
                let shape = Self::indexed_shape(*axis, input)?;
                let index = match index {
                    ArrayReferenceViewIndex::Static(index) => *index,
                    ArrayReferenceViewIndex::Symbolic(_) => {
                        return Err(TypeError::invalid(
                            "a symbolic index has no static selection; the operation that creates the view \
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

    /// Applies this transform to one carried parent value. A symbolic index is resolved by the carrier from the
    /// one value the step's `bindings` close it over; a symbolic step that binds no value (an eager path, or a
    /// malformed closure) has no selection and is rejected by [`selection`](Self::selection).
    fn apply_in<C: ViewReadCarrier>(
        &self,
        carrier: &mut C,
        input: &C::Value,
        bindings: &[C::Binding],
    ) -> Result<C::Value, ProgramError> {
        if let (Self::Index { axis, index: ArrayReferenceViewIndex::Symbolic(_) }, [binding]) = (self, bindings) {
            return carrier.index_symbolic(input, *axis, binding);
        }
        let selection = self.selection(carrier.array_type(input)?.as_ref())?;
        let sliced = carrier.slice(input, selection.starts, selection.limits)?;
        match selection.squeezed_output_shape {
            Some(shape) => carrier.reshape(&sliced, shape),
            None => Ok(sliced),
        }
    }

    /// Reconstructs the carried parent after replacing exactly the elements selected by this transform, resolving
    /// a symbolic index exactly as [`apply_in`](Self::apply_in) does.
    fn replace_in<C: ViewWriteCarrier>(
        &self,
        carrier: &mut C,
        input: &C::Value,
        replacement: &C::Value,
        bindings: &[C::Binding],
    ) -> Result<C::Value, ProgramError> {
        if let (Self::Index { axis, index: ArrayReferenceViewIndex::Symbolic(_) }, [binding]) = (self, bindings) {
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

impl ReferenceView for ArrayReferenceView {
    type Type = ArrayIrType;

    fn symbols(&self) -> Vec<usize> {
        // A transform depends on at most one symbol, which supplies the index for a symbolic indexing step.
        match self {
            Self::Index { index: ArrayReferenceViewIndex::Symbolic(symbol), .. } => vec![*symbol],
            Self::Index { .. } | Self::Slice { .. } => Vec::new(),
        }
    }

    // Both paths fold to one range or symbolic index per root axis. Nonintersecting static ranges prove disjointness;
    // identical static ranges or symbolic indices with equal bindings, offsets, and clamping extents prove equality.
    // Everything else may overlap. A malformed path cannot be folded and is treated as
    // possibly overlapping, because paths are validated when they are derived and this query must not fail.
    fn overlap(
        r#type: &ArrayIrType,
        lhs: &[ReferenceViewStep<Self>],
        rhs: &[ReferenceViewStep<Self>],
    ) -> ReferenceViewOverlap {
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

impl BatchableReferenceView for ArrayReferenceView {
    // The batch axis of a reference is an axis of its packed referent that the per-item view never sees. Indexing
    // removes one per-item axis, so the packed view cannot keep both axis positions unchanged: a batch axis at or
    // before the indexed axis shifts the packed indexed axis one position later while the output keeps the batch axis,
    // and a batch axis after the indexed axis leaves the packed indexed axis alone while the output's batch axis moves
    // one position earlier. Slicing preserves rank, so the packed view selects the complete batch axis through an
    // identity selection inserted at the batch axis position and the output keeps the batch axis.
    fn batch(&self, r#type: &ArrayIrType, batch_axis: BatchAxis) -> Result<(Self, BatchAxis), BatchingError> {
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
                "reference slice has {} axes but its input has rank {rank}",
                axes.len(),
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

/// Indices that a folded [`ArrayReferenceViewPath`] selects on one axis of its root, used by
/// [`ReferenceView::overlap`] to compare two paths of one root.
#[derive(Clone, Debug, PartialEq, Eq)]
enum RootIndexSelection {
    /// A static unit-stride range `[start, limit)` of the root axis. Before any step touches the axis this is the
    /// complete axis, a slice narrows it, and a static index collapses it to one index.
    Range {
        /// Inclusive start of the range.
        start: usize,

        /// Exclusive limit of the range.
        limit: usize,
    },

    /// One index `offset + clamp(symbol, 0, extent - 1)` of the root axis, selected relative to the
    /// range that earlier steps narrowed the axis to. Clamping depends on this extent, not just the binding.
    Symbolic {
        /// Binding of the symbolic index.
        binding: ValueId,

        /// Start of the narrowed range that the symbolic index is relative to.
        offset: usize,

        /// Size of the narrowed axis against which the runtime index is clamped.
        extent: usize,
    },
}

impl RootIndexSelection {
    /// Folds the closed `steps` of a path over a root of static shape `shape` into one range or symbolic index per root
    /// axis, or [`None`] when the path is malformed for that root (an axis, index, binding, or stride that the
    /// derivation would have rejected).
    fn fold(shape: &StaticShape, steps: &[ReferenceViewStep<ArrayReferenceView>]) -> Option<Vec<Self>> {
        let mut indices =
            shape.dimensions().iter().map(|size| Self::Range { start: 0, limit: *size }).collect::<Vec<_>>();
        // Root axes that the folded steps have not indexed away yet, in view axis order.
        let mut remaining = (0..shape.rank()).collect::<Vec<_>>();
        for step in steps {
            match step.view() {
                ArrayReferenceView::Index { axis, index } => {
                    if *axis >= remaining.len() {
                        return None;
                    }
                    let root_axis = remaining.remove(*axis);
                    let Self::Range { start, limit } = indices[root_axis] else {
                        return None;
                    };
                    indices[root_axis] = match index {
                        ArrayReferenceViewIndex::Static(index) => {
                            // Invalid paths must remain conservative even when the relative index overflows.
                            let index = start.checked_add(*index)?;
                            if index >= limit {
                                return None;
                            }
                            Self::Range { start: index, limit: index + 1 }
                        }
                        ArrayReferenceViewIndex::Symbolic(_) => {
                            Self::Symbolic { binding: *step.bindings().first()?, offset: start, extent: limit - start }
                        }
                    };
                }
                ArrayReferenceView::Slice { axes } => {
                    if axes.len() != remaining.len() {
                        return None;
                    }
                    for (selection, root_axis) in axes.iter().zip(remaining.iter()) {
                        let Self::Range { start, limit } = indices[*root_axis] else {
                            return None;
                        };
                        let narrowed_start = start.checked_add(selection.start())?;
                        let narrowed_limit = narrowed_start.checked_add(selection.size())?;
                        if selection.stride() != 1 || narrowed_limit > limit {
                            return None;
                        }
                        indices[*root_axis] = Self::Range { start: narrowed_start, limit: narrowed_limit };
                    }
                }
            }
        }
        Some(indices)
    }

    /// Returns the relation between the indices that this and `other` select on one root axis.
    fn overlap(&self, other: &Self) -> ReferenceViewOverlap {
        match (self, other) {
            (Self::Range { start: a_start, limit: a_limit }, Self::Range { start: b_start, limit: b_limit }) => {
                if a_limit <= b_start || b_limit <= a_start {
                    ReferenceViewOverlap::Disjoint
                } else if a_start == b_start && a_limit == b_limit {
                    ReferenceViewOverlap::Same
                } else {
                    ReferenceViewOverlap::MayOverlap
                }
            }
            (
                Self::Symbolic { binding: a_binding, offset: a_offset, extent: a_extent },
                Self::Symbolic { binding: b_binding, offset: b_offset, extent: b_extent },
            ) if a_binding == b_binding && a_offset == b_offset && a_extent == b_extent => ReferenceViewOverlap::Same,
            (Self::Symbolic { .. }, Self::Symbolic { .. })
            | (Self::Range { .. }, Self::Symbolic { .. })
            | (Self::Symbolic { .. }, Self::Range { .. }) => ReferenceViewOverlap::MayOverlap,
        }
    }
}

/// Normalized indices of one [`ArrayReferenceView`] applied to one statically shaped input.
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
    /// [`ArrayReferenceView::Index`] transforms only; [`None`] for rank-preserving slices, whose output
    /// shape is exactly [`Self::update_shape`].
    squeezed_output_shape: Option<Shape>,
}

impl ViewSelection {
    /// Returns the static shape of the sliced hyper-rectangle before squeezing (i.e., the update shape that writes back
    /// into the selected indices).
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

/// Immutable index mapping between a shared array-reference root and one derived handle: the array
/// specialization of the generic [`ReferenceViewPath`], whose views are [`ArrayReferenceView`]s.
///
/// The mapping stores validated transforms in root-to-handle order. The empty mapping ([`root`](Self::root)) is the
/// identity view and denotes the complete root. Each additional transform is applied to the preceding view, so
/// indexing or slicing an already-derived [`ArrayReference`] composes onto the same shared root rather than creating
/// another mutable resource.
///
/// This type is structural metadata only: it owns neither the referenced array nor its resource identity, liveness, or
/// synchronization state. [`ArrayReference`] pairs it with a handle to the shared reference allocation, and the array
/// view overlay ([`ArrayReferenceAnalysis`]) records one per reference-typed program value. The view determines that
/// handle's referent type and selected indices; mutations reconstruct the root by applying the inverse update of each
/// transform in reverse order. Consequently, overlapping handles may select the same root indices and observe one
/// another's ordered mutations, while equality and hashing distinguish different transform sequences.
///
/// `Binding` supplies symbolic indices: [`ValueId`] identifies program values, the uninhabited
/// [`NoReferenceViewBinding`] restricts eager handles to static steps, and `C::Value` binds discharge indices directly
/// to context values. Supported index transforms are described by [`ArrayReferenceView`]. Pass root handles
/// across attached-region and external runtime boundaries and recreate views inside the receiving scope. For example,
/// a scan body selects a view of a stacked reference using its explicit index input.
pub type ArrayReferenceViewPath<Binding = ValueId> = ReferenceViewPath<ArrayReferenceView, Binding>;

impl<Binding> ArrayReferenceViewPath<Binding> {
    /// Returns the exact view type derived from `root_type`.
    pub fn output_type(&self, root_type: &ArrayType) -> Result<ArrayType, TypeError> {
        self.views().try_fold(root_type.clone(), |r#type, transform| transform.output_type(&r#type))
    }

    /// Returns the root followed by each selected child, ending with this view's value. An empty path returns only
    /// the root. Each step's bindings are handed to `carrier`, which resolves symbolic indices; reconstruction uses
    /// every snapshot except the final child as its strict parents.
    fn intermediates_in<C: ViewReadCarrier<Binding = Binding>>(
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

    /// Reconstructs the root after replacing the selected leaf, working from the innermost view back to the root.
    ///
    /// # Parameters
    ///
    ///   - `carrier`: array operations used to reconstruct each parent.
    ///   - `intermediates`: one snapshot per strict parent, in root-to-view order. This is the sequence produced by
    ///     [`intermediates_in`](Self::intermediates_in) without its final selected value.
    ///   - `replacement`: new value of the selected view.
    fn reconstruct_in<C: ViewWriteCarrier<Binding = Binding>>(
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

    /// Replaces this view's selected elements through `carrier`, returning their previous snapshot plus the
    /// reconstructed root, so that the eager swap and the discharge-time replacement share one traversal.
    fn swap_in<C: ViewWriteCarrier<Value: Clone, Binding = Binding>>(
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

    /// Replaces this view's selected elements through `carrier` without materializing the selected old value.
    ///
    /// Immutable root reconstruction still needs each strict parent of the selected leaf so indices outside the logical
    /// view survive. The final transform is deliberately not applied: its output is exactly the old selected value that
    /// write-only semantics must not observe. An identity view therefore returns `replacement` directly.
    fn write_in<C: ViewWriteCarrier<Binding = Binding>>(
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

impl ArrayReferenceViewPath<NoReferenceViewBinding> {
    /// Applies the complete static mapping to one root snapshot.
    fn apply<A>(&self, root: &A) -> Result<A, ProgramError>
    where
        A: Value<Type = ArrayType> + Reshape + Slice,
    {
        let mut carrier = EagerViewCarrier(PhantomData);
        self.steps()
            .iter()
            .try_fold(root.clone(), |value, step| step.view().apply_in(&mut carrier, &value, step.bindings()))
    }

    /// Replaces this static view and returns the reconstructed root plus its old view snapshot.
    fn swap<A>(&self, root: &A, replacement: &A) -> Result<(A, A), ProgramError>
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
/// Reading the selected view and reconstructing the root with update-slice each exist exactly once, on
/// [`ArrayReferenceViewPath`], generically over this carrier: the eager carrier operates on concrete values with the
/// array-manipulation capabilities, while reference discharge binds the identical operation sequence through its
/// context. Keeping one traversal guarantees the staged and eager semantics cannot drift apart. Static steps lower to
/// the carrier's slice and reshape; a symbolic index step hands the carrier its index through the path's
/// [`Binding`](Self::Binding).
trait ViewReadCarrier {
    /// Value representation carried through the traversal.
    type Value;

    /// What a symbolic index of the traversed path is closed over.
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

    /// Selects the index that `binding` closes over on `axis` of `input` and removes that axis.
    fn index_symbolic(
        &mut self,
        input: &Self::Value,
        axis: usize,
        binding: &Self::Binding,
    ) -> Result<Self::Value, ProgramError>;
}

/// A [`ViewReadCarrier`] that can also write a selected hyper-rectangle back into its parent.
trait ViewWriteCarrier: ViewReadCarrier {
    /// Returns `target` with `update` written at `starts`.
    fn update_slice(
        &mut self,
        target: &Self::Value,
        update: &Self::Value,
        starts: Vec<usize>,
    ) -> Result<Self::Value, ProgramError>;

    /// Returns `target` with `update` written at the index that `binding` closes over on `axis`, the inverse of
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
/// symbolic-index hooks are unreachable by type.
struct EagerViewCarrier<A>(PhantomData<A>);

impl<A: Value<Type = ArrayType> + Reshape + Slice> ViewReadCarrier for EagerViewCarrier<A> {
    type Value = A;
    type Binding = NoReferenceViewBinding;

    fn array_type<'c>(&'c self, value: &'c A) -> Result<Cow<'c, ArrayType>, ProgramError> {
        Ok(value.r#type())
    }

    fn slice(&mut self, input: &A, starts: Vec<usize>, limits: Vec<usize>) -> Result<A, ProgramError> {
        input.slice(starts.as_slice(), limits.as_slice(), &vec![1; starts.len()])
    }

    fn reshape(&mut self, input: &A, shape: Shape) -> Result<A, ProgramError> {
        input.reshape(shape)
    }

    fn index_symbolic(
        &mut self,
        _input: &A,
        _axis: usize,
        binding: &NoReferenceViewBinding,
    ) -> Result<A, ProgramError> {
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
        binding: &NoReferenceViewBinding,
    ) -> Result<A, ProgramError> {
        match *binding {}
    }
}

/// Eager array-reference handle pairing one shared root allocation with handle-local view metadata.
///
/// Equality and hashing identify the mutable location and structural view, not the handle-local type-identity
/// namespace. Renaming type identities therefore preserves equality with the original handle when its view is
/// unchanged.
pub struct ArrayReference<A: Value<Type = ArrayType>> {
    /// Handle to the shared root allocation.
    root: Reference<A>,

    /// Ordered mapping from the shared root to this handle's referent. Eager handles only ever carry static steps,
    /// so no symbol is ever bound on this path.
    view: ArrayReferenceViewPath<NoReferenceViewBinding>,

    /// Exact handle type derived once from the root type and view, so that repeated [`Typed::r#type`] calls
    /// borrow the cached type instead of re-deriving the complete transform chain.
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
        Self { root, view: ArrayReferenceViewPath::root(), r#type }
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

    /// Returns a copy of this handle with `transform` appended to its view, sharing the same root allocation. A
    /// symbolic index is rejected with [`ArrayReferenceViewError::SymbolicViewIndex`]: an eager handle's path
    /// carries only static steps, and the index is resolved by the operation that creates the view.
    pub fn with_transform(&self, transform: ArrayReferenceView) -> Result<Self, ProgramError> {
        if !transform.symbols().is_empty() {
            return Err(ProgramError::custom(ArrayReferenceViewError::SymbolicViewIndex));
        }
        // The cached handle type already reflects every earlier transform, so composition validates and derives
        // incrementally instead of re-folding the complete chain from the root type. Derivation is purely structural:
        // holder liveness is checked only when the resulting handle accesses state.
        let referent = transform.output_type(self.r#type.referent())?;
        let view = self.view.with_view(transform);
        Ok(Self { root: self.root.clone(), view, r#type: ReferenceType::new(referent) })
    }

    /// Returns an immutable snapshot of this handle's selected elements.
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

    /// Replaces this handle's selected elements and returns their previous snapshot.
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

    /// Replaces this handle's selected elements without returning their previous snapshot.
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

    /// Adds `update` into this handle's selected elements.
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

    /// Validates that `value` exactly matches this handle's derived referent type. Root-handle mutations inherit this
    /// rule from the shared reference state, but derived-view mutations must enforce it themselves: update-slice
    /// reconstruction only requires the written value to fit inside the selected indices, so a smaller replacement
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

// TODO(eaplatanios): Review this module.

/// Array specialization of [`ReferenceViewAnalysis`], associating every reference-typed value in a region and its
/// attached computation regions with an [`ArrayReferenceViewPath`]. The generic analysis owns reference roots, aliases,
/// accesses, capture scopes, and lifetime validation. The specialization uses the generic
/// [`path`](ReferenceViewAnalysis::path) and [`paths`](ReferenceViewAnalysis::paths) accessors directly; it does not
/// perform a second analysis or keep a separate view table.
///
/// Root handles have an empty view, identity aliases copy their source view, and index or slice aliases append the
/// operation's [`ArrayReferenceView`]. The resulting view reproduces the value's declared referent shape when
/// applied to its root's array type. Attached-region inputs receive complete reference handles. Selections inside
/// that region are explicit instructions whose symbolic indices bind to the selecting operand's [`ValueId`].
///
/// Construct an analysis with [`ReferenceViewAnalysis::new`] or reuse a cached one through
/// [`RegionRef::reference_view_analysis`](crate::RegionRef::reference_view_analysis). Validation is explicit: consumers
/// such as kernel boundaries and lowering request this table when needed instead of independently reconstructing array
/// views. Program construction and eager reference operations continue to perform their own local validation.
pub type ArrayReferenceAnalysis = ReferenceViewAnalysis<ArrayReferenceView>;

// TODO(eaplatanios): Review this module.

/// [`ReferenceDischargePolicy`] of the array reference universe.
///
/// An array reference's referent is an ordinary [`ArrayType`]-typed array, and the alias one flowing handle carries is
/// the composed [`ArrayReferenceViewPath`] mapping its allocation to its own indices, with every symbolic index closed over
/// the context value it selects. Every access therefore reaches its indices through the same view traversal the eager
/// handles use, which is what keeps staged and eager reference semantics from drifting apart: reading materializes the
/// allocation-to-handle chain and takes its last snapshot, while a replacement or an accumulation writes the new leaf
/// back through that chain in reverse. Symbolic indices use dynamic slicing and updates with the same clamping
/// bounds, so reads and mutations always address the same elements.
///
/// The reconstruction context is bounded by [`Context`] rather than [`Domain`](crate::Domain) because the view
/// traversal binds canonical slicing, reshape, and update operations into it. Their value-level capabilities are
/// stated over [`ArrayType`]-typed values rather than the composite array IR, so the policy
/// constructs them through the context's operation family.
#[derive(Copy, Clone, Debug)]
pub struct ArrayReferenceDischarge;

impl ReferenceDischargeableType for ArrayIrType {
    type Policy = ArrayReferenceDischarge;
}

impl<C: Context<Type = ArrayIrType>> ReferenceDischargePolicy<C> for ArrayReferenceDischarge
where
    C::Operation: ArrayReferenceViewOperation,
{
    type Referent = ArrayType;
    type Alias = ArrayReferenceViewPath<C::Value>;

    fn storage_alias(_referent: &ArrayType) -> ArrayReferenceViewPath<C::Value> {
        ArrayReferenceViewPath::root()
    }

    fn read(
        context: &C,
        current: &C::Value,
        alias: &ArrayReferenceViewPath<C::Value>,
    ) -> Result<C::Value, ProgramError> {
        let mut intermediates = alias.intermediates_in(&mut ContextViewCarrier(context), current.clone())?;

        // The traversal starts with the complete allocation, so the chain is nonempty and its final value is the part
        // selected by this handle.
        Ok(intermediates.pop().unwrap())
    }

    fn write(
        context: &C,
        current: &C::Value,
        replacement: C::Value,
        alias: &ArrayReferenceViewPath<C::Value>,
    ) -> Result<C::Value, ProgramError> {
        alias.write_in(&mut ContextViewCarrier(context), current.clone(), replacement)
    }

    fn swap(
        context: &C,
        current: &C::Value,
        replacement: C::Value,
        alias: &ArrayReferenceViewPath<C::Value>,
    ) -> Result<(C::Value, C::Value), ProgramError> {
        alias.swap_in(&mut ContextViewCarrier(context), current.clone(), replacement)
    }
}

// Composite array-IR values deliberately expose no value-level addition: the composite family carries array payloads
// through `ArrayIrOperation::Array` and lifts the type-generic `AddOperation<ArrayIrType>` into that member instead,
// which is the same seam generic reverse mode uses to accumulate cotangents. Accumulation therefore binds the lifted
// addition through the context, requiring nothing beyond the conversion the operation family already provides.
impl<C: Context<Type = ArrayIrType>> ReferenceAccumulationPolicy<C> for ArrayReferenceDischarge
where
    C::Operation: ArrayReferenceViewOperation + From<AddOperation<ArrayIrType>>,
{
    fn accumulate(
        context: &C,
        current: &C::Value,
        update: C::Value,
        alias: &ArrayReferenceViewPath<C::Value>,
    ) -> Result<C::Value, ProgramError> {
        let mut carrier = ContextViewCarrier(context);
        let intermediates = alias.intermediates_in(&mut carrier, current.clone())?;
        // Add at the selected leaf, then rebuild each enclosing slice without reading the leaf a second time.
        let selected = intermediates.last().unwrap().clone();
        let accumulated = carrier.bind(C::Operation::from(AddOperation::new()), &[&selected, &update])?;
        alias.reconstruct_in(&mut carrier, &intermediates[..alias.views().len()], accumulated)
    }
}

/// View carrier that binds the canonical slice, reshape, and update-slice operations of one array reference view into
/// a reference discharge context, sharing the single [`ArrayReferenceViewPath`] traversal with the eager value carrier,
/// which keeps staged and eager reference semantics consistent. Symbolic indices arrive closed over context values
/// and select a size-one dynamic slice; updates restore the removed axis before replacing that slice.
struct ContextViewCarrier<'c, C>(
    /// Context in which the slice, reshape, and update-slice operations are bound.
    &'c C,
);

impl<C: Context<Type = ArrayIrType>> ContextViewCarrier<'_, C> {
    /// Binds one single-result view operation into the context and returns its result.
    ///
    /// # Parameters
    ///
    ///   - `operation`: Context-family operation to bind.
    ///   - `inputs`: Operands of the application, in operation-defined order.
    fn bind(&self, operation: C::Operation, inputs: &[&C::Value]) -> Result<C::Value, ProgramError> {
        let inputs = inputs.iter().map(|input| (*input).clone()).collect::<Vec<_>>();
        let mut outputs = self.0.bind(operation, Vec::new(), inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Context<Type = ArrayIrType>> ViewReadCarrier for ContextViewCarrier<'_, C>
where
    C::Operation: ArrayReferenceViewOperation,
{
    type Value = C::Value;
    type Binding = C::Value;

    fn array_type<'c>(&'c self, value: &'c C::Value) -> Result<Cow<'c, ArrayType>, ProgramError> {
        match value.r#type() {
            Cow::Borrowed(r#type) => Ok(Cow::Borrowed(<&ArrayType>::try_from(r#type)?)),
            Cow::Owned(r#type) => Ok(Cow::Owned(<&ArrayType>::try_from(&r#type)?.clone())),
        }
    }

    fn slice(&mut self, input: &C::Value, starts: Vec<usize>, limits: Vec<usize>) -> Result<C::Value, ProgramError> {
        self.bind(C::Operation::from_reference_slice(SliceOperation::new(starts, limits)), &[input])
    }

    fn reshape(&mut self, input: &C::Value, shape: Shape) -> Result<C::Value, ProgramError> {
        self.bind(C::Operation::from_reference_reshape(ReshapeOperation::new(shape)), &[input])
    }

    fn index_symbolic(&mut self, input: &C::Value, axis: usize, binding: &C::Value) -> Result<C::Value, ProgramError> {
        let input_type = self.array_type(input)?.into_owned();
        let mut sizes = ArrayReferenceView::indexed_shape(axis, &input_type)?.dimensions().to_vec();
        sizes[axis] = 1;
        // Unselected axes span their complete extent, so dynamic slicing clamps their start to zero. Reusing
        // the scalar index there avoids constructing redundant zero values in the context's value family.
        let mut inputs = vec![input];
        inputs.extend(std::iter::repeat_n(binding, sizes.len()));
        let selected =
            self.bind(C::Operation::from_reference_dynamic_slice(DynamicSliceOperation::new(sizes)), &inputs)?;
        self.reshape(&selected, input_type.without_dimension(axis)?.0.shape().clone())
    }
}

impl<C: Context<Type = ArrayIrType>> ViewWriteCarrier for ContextViewCarrier<'_, C>
where
    C::Operation: ArrayReferenceViewOperation,
{
    fn update_slice(
        &mut self,
        target: &C::Value,
        update: &C::Value,
        starts: Vec<usize>,
    ) -> Result<C::Value, ProgramError> {
        self.bind(C::Operation::from_reference_update_slice(UpdateSliceOperation::new(starts)), &[target, update])
    }

    fn update_index_symbolic(
        &mut self,
        target: &C::Value,
        update: &C::Value,
        axis: usize,
        binding: &C::Value,
    ) -> Result<C::Value, ProgramError> {
        let target_type = self.array_type(target)?.into_owned();
        let mut dimensions = ArrayReferenceView::indexed_shape(axis, &target_type)?.dimensions().to_vec();
        dimensions[axis] = 1;
        let rank = dimensions.len();
        let update = self.reshape(update, Shape::new(dimensions.into_iter().map(Dimension::Static).collect()))?;
        // Restore the indexed axis before writing back. Full-size axes clamp to zero just as in the read path,
        // while the selected axis uses the same runtime index and clamping extent as the original view.
        let mut inputs = vec![target, &update];
        inputs.extend(std::iter::repeat_n(binding, rank));
        self.bind(C::Operation::from_reference_dynamic_update_slice(DynamicUpdateSliceOperation), &inputs)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::addressing::ArraySliceAxis;
    use crate::arrays::arrays::Array;
    use crate::arrays::ir::ArrayIrValue;
    use crate::arrays::operations::{ArrayIrOperation, ReferenceIndexOperation, ReferenceSliceOperation};
    use crate::arrays::types::arrays::ArrayType;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{Dimension, DimensionBounds, DimensionVariable};
    use crate::arrays::types::ir::ArrayIrType;
    use crate::axes::Axis;
    use crate::contexts::EagerContext;
    use crate::operations::{
        ReferenceAddUpdateOperation, ReferenceFreezeOperation, ReferenceNewOperation, ReferenceReadOperation,
        ReferenceSwapOperation,
    };
    use crate::parameters::Placeholder;
    use crate::programs::{
        AtomId, Program, ProgramBuilder, ReferenceCompletion, ReferenceReplacementPreparation, ReferenceType, RegionId,
    };
    use crate::tracing::{Trace, Tracer, TracingContext};

    use super::*;

    /// Array IR values used to construct reference-view analysis fixtures.
    type TestValue = ArrayIrValue<Array>;

    /// Builder for array operations and reference operations in the same program.
    type TestBuilder = ProgramBuilder<TestValue, ArrayIrOperation<Array>>;

    /// Identifies an atom in a fixture's numbered region.
    fn value_id(region: usize, atom: usize) -> ValueId {
        ValueId::new(RegionId::new(region), AtomId::new(atom))
    }

    /// Operation family used by eager and staged array reference fixtures.
    type TestOperation = ArrayIrOperation<Array>;

    /// Context that records immutable array reconstruction.
    type TestContext = TracingContext<TestValue, TestOperation>;

    #[test]
    fn test_array_reference_view_error() {
        for (error, message) in [
            (
                ArrayReferenceViewError::CannotFreezeView,
                "cannot freeze a reference view; freeze the root reference instead",
            ),
            (
                ArrayReferenceViewError::CannotReadRootThroughView,
                "cannot read a reference view through the root-only snapshot accessor",
            ),
            (
                ArrayReferenceViewError::InvalidRuntimeRoot,
                "reference runtime transactions require an unrenamed root handle",
            ),
            (
                ArrayReferenceViewError::SymbolicViewIndex,
                concat!(
                    "eager reference handles carry only static views; ",
                    "the operation that creates a symbolic view resolves it",
                ),
            ),
        ] {
            assert_eq!(error.to_string(), message);
        }
    }

    #[test]
    fn test_array_reference_view_output_type() {
        let input = ArrayType::new_static(DataType::F32, [3, 4]);
        assert_eq!(
            ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(1) }.output_type(&input),
            Ok(ArrayType::new_static(DataType::F32, [4])),
        );
        assert_eq!(
            ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 3, 1)] }
                .output_type(&input),
            Ok(ArrayType::new_static(DataType::F32, [2, 3])),
        );
        // Empty selections remain valid array views and preserve rank.
        assert_eq!(
            ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(3, 0, 1), ArraySliceAxis::new(0, 4, 1)] }
                .output_type(&input),
            Ok(ArrayType::new_static(DataType::F32, [0, 4])),
        );
    }

    #[test]
    fn test_array_reference_view_output_type_rejects_invalid_selections() {
        let matrix_type = ArrayType::new_static(DataType::F32, [3, 4]);
        let vector_type = ArrayType::new_static(DataType::F32, [3]);

        // Static indexing selects one existing index on one existing axis; a symbolic index still names an
        // existing axis.
        assert_eq!(
            ArrayReferenceView::Index { axis: 2, index: ArrayReferenceViewIndex::Static(0) }.output_type(&matrix_type),
            Err(TypeError::invalid("reference index axis 2 is out of bounds for rank 2")),
        );
        assert_eq!(
            ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(3) }.output_type(&matrix_type),
            Err(TypeError::invalid("reference index 3 on axis 0 is out of bounds for size 3")),
        );
        assert_eq!(
            ArrayReferenceView::Index { axis: 2, index: ArrayReferenceViewIndex::Symbolic(1) }
                .output_type(&matrix_type),
            Err(TypeError::invalid("reference index axis 2 is out of bounds for rank 2")),
        );

        // Static slicing is rank-preserving, so it declares exactly one unit-stride selection per input axis and
        // stays inside every axis of the input.
        assert_eq!(
            ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(0, 2, 1)] }.output_type(&matrix_type),
            Err(TypeError::invalid("reference slice has 1 axes but its input has rank 2")),
        );
        assert_eq!(
            ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(0, 2, 2)] }.output_type(&vector_type),
            Err(TypeError::invalid(
                "reference slice axis 0 stride must be 1 until scatter-backed strided updates are supported",
            )),
        );
        assert_eq!(
            ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(2, 3, 1)] }.output_type(&vector_type),
            Err(TypeError::invalid("reference slice on axis 0 with start 2 and size 3 exceeds input size 3")),
        );

        // The exclusive limit is computed as `start + size`, so an unrepresentable limit is rejected before it can
        // wrap around into an apparently valid selection.
        assert_eq!(
            ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(usize::MAX, 1, 1)] }.output_type(&vector_type),
            Err(TypeError::invalid("reference slice limit overflows `usize` on axis 0")),
        );
    }

    #[test]
    fn test_array_reference_view_output_type_rejects_dynamic_shapes() {
        let input = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("length", DimensionBounds::unbounded()))]),
        );
        assert_eq!(
            ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(0) }.output_type(&input),
            Err(TypeError::invalid(format!("reference indexing requires a static referent type but got `{input}`"))),
        );
        assert_eq!(
            ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(0, 1, 1)] }.output_type(&input),
            Err(TypeError::invalid(format!("reference slicing requires a static referent type but got `{input}`"))),
        );
    }

    #[test]
    fn test_array_reference_view_output_type_symbolic_index() {
        let matrix_type = ArrayType::new_static(DataType::F32, [3, 4]);
        let symbolic = ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Symbolic(1) };
        let static_index = ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(1) };
        // Removing a symbolic axis derives the same type even when no static index could select it.
        assert_eq!(symbolic.output_type(&matrix_type), static_index.output_type(&matrix_type));
        assert_eq!(symbolic.output_type(&matrix_type), Ok(ArrayType::new_static(DataType::F32, [4])));
        assert_eq!(
            symbolic.output_type(&ArrayType::new_static(DataType::F32, [0, 4])),
            Ok(ArrayType::new_static(DataType::F32, [4])),
        );
    }

    #[test]
    fn test_array_reference_view_symbols() {
        let symbolic = ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Symbolic(1) };
        assert_eq!(symbolic.symbols(), vec![1]);
        assert_eq!(
            ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(1) }.symbols(),
            Vec::<usize>::new(),
        );
        assert_eq!(
            ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(0, 2, 1)] }.symbols(),
            Vec::<usize>::new(),
        );
    }

    #[test]
    fn test_array_reference_view_overlap() {
        let root = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [4, 3])));
        let empty: ArrayReferenceViewPath = ArrayReferenceViewPath::root();
        let rows_0_1 = empty.with_view(ArrayReferenceView::Slice {
            axes: vec![ArraySliceAxis::new(0, 2, 1), ArraySliceAxis::new(0, 3, 1)],
        });
        let rows_1_2 = empty.with_view(ArrayReferenceView::Slice {
            axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 3, 1)],
        });
        let rows_2_3 = empty.with_view(ArrayReferenceView::Slice {
            axes: vec![ArraySliceAxis::new(2, 2, 1), ArraySliceAxis::new(0, 3, 1)],
        });
        let row_1 = empty.with_view(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(1) });
        let column_0 =
            empty.with_view(ArrayReferenceView::Index { axis: 1, index: ArrayReferenceViewIndex::Static(0) });

        // Static indices fold to one range per root axis: disjoint ranges on any axis make the paths disjoint,
        // identical ranges on every axis make them the same, and intersecting ranges may overlap. The trait function
        // and the path method agree.
        assert_eq!(
            ArrayReferenceView::overlap(&root, rows_0_1.steps(), rows_2_3.steps()),
            ReferenceViewOverlap::Disjoint,
        );
        assert_eq!(rows_0_1.overlap(&rows_2_3, &root), ReferenceViewOverlap::Disjoint);
        assert_eq!(row_1.overlap(&row_1, &root), ReferenceViewOverlap::Same);
        assert_eq!(rows_0_1.overlap(&rows_1_2, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(row_1.overlap(&rows_0_1, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(row_1.overlap(&rows_2_3, &root), ReferenceViewOverlap::Disjoint);

        // Rank changes are tracked while folding: an index removes its axis, so a slice that follows it addresses the
        // remaining root axes, and different step sequences that select the same indices are the same.
        let row_1_columns_1_2 = row_1.with_view(ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] });
        let row_1_column_1 =
            row_1.with_view(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(1) });
        let rows_1_columns_1_2_row_0 = empty
            .with_view(ArrayReferenceView::Slice {
                axes: vec![ArraySliceAxis::new(1, 1, 1), ArraySliceAxis::new(1, 2, 1)],
            })
            .with_view(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(0) });
        assert_eq!(row_1_columns_1_2.overlap(&column_0, &root), ReferenceViewOverlap::Disjoint);
        assert_eq!(row_1_columns_1_2.overlap(&row_1, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(row_1_columns_1_2.overlap(&row_1_column_1, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(row_1_columns_1_2.overlap(&rows_1_columns_1_2_row_0, &root), ReferenceViewOverlap::Same);
        assert_eq!(row_1_column_1.overlap(&column_0, &root), ReferenceViewOverlap::Disjoint);

        // The complete root is the same as itself and as a slice spanning every axis, and may overlap with any
        // narrowing path.
        let complete = empty.with_view(ArrayReferenceView::Slice {
            axes: vec![ArraySliceAxis::new(0, 4, 1), ArraySliceAxis::new(0, 3, 1)],
        });
        assert_eq!(empty.overlap(&empty, &root), ReferenceViewOverlap::Same);
        assert_eq!(empty.overlap(&complete, &root), ReferenceViewOverlap::Same);
        assert_eq!(empty.overlap(&rows_0_1, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(empty.overlap(&row_1_column_1, &root), ReferenceViewOverlap::MayOverlap);

        // Symbolic indices agree only when their binding, offset, and clamping extent agree. Different
        // offsets can clamp to the same root element, so they cannot establish disjointness.
        let symbolic = ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Symbolic(1) };
        let first = ValueId::new(RegionId::new(0), AtomId::new(1));
        let second = ValueId::new(RegionId::new(0), AtomId::new(2));
        let other_region = ValueId::new(RegionId::new(1), AtomId::new(1));
        let row_first = empty.with_step(symbolic.clone(), vec![first]);
        let row_second = empty.with_step(symbolic.clone(), vec![second]);
        let row_other_region = empty.with_step(symbolic.clone(), vec![other_region]);
        let shifted_row_first = rows_1_2.with_step(symbolic.clone(), vec![first]);
        assert_eq!(
            row_first.overlap(&empty.with_step(symbolic.clone(), vec![first]), &root),
            ReferenceViewOverlap::Same
        );
        assert_eq!(row_first.overlap(&row_second, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(row_first.overlap(&row_other_region, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(row_first.overlap(&row_1, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(row_first.overlap(&rows_2_3, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(row_first.overlap(&empty, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(row_first.overlap(&shifted_row_first, &root), ReferenceViewOverlap::MayOverlap);
        // Equal offsets with different extents also clamp differently: a large index selects row 3 in the
        // whole root but row 1 in its first two rows.
        let shortened_row_first = rows_0_1.with_step(symbolic.clone(), vec![first]);
        assert_eq!(row_first.overlap(&shortened_row_first, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(
            row_first
                .with_view(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(0) })
                .overlap(
                    &row_second
                        .with_view(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(2) }),
                    &root,
                ),
            ReferenceViewOverlap::Disjoint,
        );

        // A path or root that cannot be folded (an out-of-bounds axis or index, a symbolic step without its
        // binding, a non-reference root, or a root without a static shape) is conservatively reported as possibly
        // overlapping rather than failing.
        let out_of_bounds =
            empty.with_view(ArrayReferenceView::Index { axis: 2, index: ArrayReferenceViewIndex::Static(0) });
        let unbound = empty.with_view(symbolic);
        assert_eq!(out_of_bounds.overlap(&rows_2_3, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(
            empty
                .with_view(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(4) })
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
    fn test_array_reference_view_overlap_overflow() {
        let root = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])));
        let view: ArrayReferenceViewPath = ArrayReferenceViewPath::root()
            .with_view(ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] })
            .with_view(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(usize::MAX) });
        // Malformed relative indices cannot wrap around to become valid root indices.
        assert_eq!(view.overlap(&ArrayReferenceViewPath::root(), &root), ReferenceViewOverlap::MayOverlap);
    }

    #[test]
    fn test_array_reference_view_batch() {
        let packed = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2, 3, 4])));
        let index = ArrayReferenceView::Index { axis: 1, index: ArrayReferenceViewIndex::Static(2) };
        let slice =
            ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(1, 1, 1), ArraySliceAxis::new(1, 2, 1)] };

        // A batch axis at or before the indexed axis shifts the packed indexed axis one position later and the output
        // keeps the batch axis, while a batch axis after the indexed axis leaves the packed indexed axis alone and the
        // output batch axis moves one position earlier. Negative batch axes normalize against the packed rank.
        assert_eq!(
            index.batch(&packed, BatchAxis::new(0)),
            Ok((ArrayReferenceView::Index { axis: 2, index: ArrayReferenceViewIndex::Static(2) }, BatchAxis::new(0))),
        );
        assert_eq!(
            index.batch(&packed, BatchAxis::new(1)),
            Ok((ArrayReferenceView::Index { axis: 2, index: ArrayReferenceViewIndex::Static(2) }, BatchAxis::new(1))),
        );
        assert_eq!(
            index.batch(&packed, BatchAxis::new(2)),
            Ok((ArrayReferenceView::Index { axis: 1, index: ArrayReferenceViewIndex::Static(2) }, BatchAxis::new(1))),
        );
        assert_eq!(
            index.batch(&packed, BatchAxis::new(-1)),
            Ok((ArrayReferenceView::Index { axis: 1, index: ArrayReferenceViewIndex::Static(2) }, BatchAxis::new(1))),
        );

        // Batching preserves the symbol that supplies the index.
        let symbolic = ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Symbolic(1) };
        assert_eq!(
            symbolic.batch(&packed, BatchAxis::new(0)),
            Ok(
                (ArrayReferenceView::Index { axis: 1, index: ArrayReferenceViewIndex::Symbolic(1) }, BatchAxis::new(0),)
            ),
        );

        // Slicing inserts the complete batch axis at the batch axis position and keeps the batch axis.
        assert_eq!(
            slice.batch(&packed, BatchAxis::new(1)),
            Ok((
                ArrayReferenceView::Slice {
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
            ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(0, 3, 1)] }
                .batch(&ArrayIrType::Reference(ReferenceType::new(dynamic.clone())), BatchAxis::new(0)),
            Err(BatchingError::DynamicBatchAxis { r#type: Box::new(dynamic), axis: Axis::from(0) }),
        );
    }

    #[test]
    fn test_array_reference_view_batch_rejects_invalid_axes() {
        let packed = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2, 3, 4])));
        assert_eq!(
            ArrayReferenceView::Index { axis: 2, index: ArrayReferenceViewIndex::Static(0) }
                .batch(&packed, BatchAxis::new(0)),
            Err(TypeError::invalid("reference index axis 2 is out of bounds for rank 2").into()),
        );
        // Shifting an unchecked maximum axis used to overflow before it could be rejected.
        assert_eq!(
            ArrayReferenceView::Index { axis: usize::MAX, index: ArrayReferenceViewIndex::Static(0) }
                .batch(&packed, BatchAxis::new(0)),
            Err(TypeError::invalid(format!("reference index axis {} is out of bounds for rank 2", usize::MAX,)).into()),
        );
        // Inserting the batch selection requires exactly one selection per unbatched input axis.
        assert_eq!(
            ArrayReferenceView::Slice { axes: Vec::new() }.batch(&packed, BatchAxis::new(2)),
            Err(TypeError::invalid("reference slice has 0 axes but its input has rank 2").into()),
        );
        assert_eq!(
            ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(0, 1, 1); 3] }.batch(&packed, BatchAxis::new(2)),
            Err(TypeError::invalid("reference slice has 3 axes but its input has rank 2").into()),
        );
    }

    #[test]
    fn test_array_reference_view_path_output_type() {
        let root_type = ArrayType::new_static(DataType::F32, [3, 4]);
        let root: ArrayReferenceViewPath = ArrayReferenceViewPath::root();
        assert_eq!(root.output_type(&root_type), Ok(root_type.clone()));

        // Each transform applies to the preceding view, so the slice narrows both axes and the index then removes
        // the leading axis of the already-narrowed view.
        let slice =
            ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 3, 1)] };
        let index = ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(1) };
        let sliced = root.with_view(slice);
        let indexed = sliced.with_view(index);
        assert_eq!(sliced.output_type(&root_type), Ok(ArrayType::new_static(DataType::F32, [2, 3])));
        assert_eq!(indexed.output_type(&root_type), Ok(ArrayType::new_static(DataType::F32, [3])));
    }

    #[test]
    fn test_array_reference_view_path_intermediates_in() {
        let view: ArrayReferenceViewPath<NoReferenceViewBinding> = ArrayReferenceViewPath::root()
            .with_view(ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] })
            .with_view(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(1) });
        let root = Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let mut carrier = EagerViewCarrier::<Array>(PhantomData);
        assert_eq!(
            view.intermediates_in(&mut carrier, root.clone()),
            Ok(vec![root.clone(), Array::vector(vec![2.0_f32, 3.0]).unwrap(), Array::scalar(3.0_f32).unwrap(),]),
        );
        assert_eq!(ArrayReferenceViewPath::root().intermediates_in(&mut carrier, root.clone()), Ok(vec![root]));
    }

    #[test]
    fn test_array_reference_view_path_reconstruct_in() {
        let view: ArrayReferenceViewPath<NoReferenceViewBinding> = ArrayReferenceViewPath::root()
            .with_view(ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] })
            .with_view(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(1) });
        let mut carrier = EagerViewCarrier::<Array>(PhantomData);
        // Reconstruction consumes strict parents in reverse order; the old selected scalar is unnecessary.
        assert_eq!(
            view.reconstruct_in(
                &mut carrier,
                &[Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap(), Array::vector(vec![2.0_f32, 3.0]).unwrap(),],
                Array::scalar(7.0_f32).unwrap()
            ),
            Ok(Array::vector(vec![1.0_f32, 2.0, 7.0, 4.0]).unwrap()),
        );
        assert_eq!(
            ArrayReferenceViewPath::root().reconstruct_in(&mut carrier, &[], Array::scalar(7.0_f32).unwrap()),
            Ok(Array::scalar(7.0_f32).unwrap()),
        );
    }

    #[test]
    fn test_array_reference_view_path_reconstruct_in_rejects_invalid_parent_count() {
        let view: ArrayReferenceViewPath<NoReferenceViewBinding> = ArrayReferenceViewPath::root()
            .with_view(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(0) });
        let mut carrier = EagerViewCarrier::<Array>(PhantomData);
        assert_eq!(
            view.reconstruct_in(&mut carrier, &[], Array::scalar(1.0_f32).unwrap()),
            Err(ProgramError::MalformedProgram(
                "reference view reconstruction requires 1 parent snapshots but received 0".to_string(),
            )),
        );
        assert_eq!(
            view.reconstruct_in(
                &mut carrier,
                &[Array::vector(vec![1.0_f32]).unwrap(), Array::scalar(1.0_f32).unwrap()],
                Array::scalar(1.0_f32).unwrap(),
            ),
            Err(ProgramError::MalformedProgram(
                "reference view reconstruction requires 1 parent snapshots but received 2".to_string(),
            )),
        );
    }

    #[test]
    fn test_array_reference_new() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]).unwrap());
        assert_eq!(root.r#type().as_ref(), &ReferenceType::new(ArrayType::new_static(DataType::F32, [2])));
        assert_eq!(root.read(), Ok(Array::vector(vec![1.0_f32, 2.0]).unwrap()));
        let alias = root.clone();
        let separate = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]).unwrap());
        let view = root
            .with_transform(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(0) })
            .unwrap();
        assert_eq!(root, alias);
        assert_ne!(root, separate);
        assert_ne!(root, view);
        let references = HashMap::from([(root.clone(), "root"), (view.clone(), "view")]);
        assert_eq!(references.get(&alias), Some(&"root"));
        assert_eq!(references.get(&view), Some(&"view"));
        assert_eq!(root.to_string(), "ref<f32[2]>");
        assert_eq!(format!("{root:?}"), format!("ArrayReference {{ id: {:?}, view: {:?} }}", root.id(), root.view));
    }

    #[test]
    fn test_array_reference_id() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]).unwrap());
        let view = root
            .with_transform(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(0) })
            .unwrap();
        assert_eq!(root.id(), root.clone().id());
        assert_eq!(root.id(), view.id());
        assert_ne!(root.id(), ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]).unwrap()).id());
    }

    #[test]
    fn test_array_reference_is_runtime_root_handle() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]).unwrap());
        let view = root
            .with_transform(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(0) })
            .unwrap();
        assert!(root.is_runtime_root_handle());
        assert!(!view.is_runtime_root_handle());
    }

    #[test]
    fn test_array_reference_lock_root() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]).unwrap());
        let view = root
            .with_transform(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(0) })
            .unwrap();
        drop(root.lock_root().unwrap());
        let error = view.lock_root().err().unwrap();
        assert_eq!(
            error.downcast_custom::<ArrayReferenceViewError>(),
            Some(&ArrayReferenceViewError::InvalidRuntimeRoot),
        );
        assert_eq!(error.to_string(), "reference runtime transactions require an unrenamed root handle");
    }

    #[test]
    fn test_array_reference_with_transform() {
        // Composition validates each appended transform against the preceding view's derived type, so an out-of-bounds
        // index of the derived view is rejected even though it exists in the root.
        let slice =
            ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 3, 1)] };
        let handle = ArrayReference::new(Array::matrix(3, 4, (1..=12).map(|value| value as f32).collect()).unwrap())
            .with_transform(slice.clone())
            .unwrap();
        assert_eq!(handle.r#type().as_ref(), &ReferenceType::new(ArrayType::new_static(DataType::F32, [2, 3])));
        assert_eq!(handle.read(), Ok(Array::matrix(2, 3, vec![5.0_f32, 6.0, 7.0, 9.0, 10.0, 11.0]).unwrap()));
        assert_eq!(
            handle
                .with_transform(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(2) })
                .unwrap_err(),
            TypeError::invalid("reference index 2 on axis 0 is out of bounds for size 2").into(),
        );
    }

    #[test]
    fn test_array_reference_with_transform_rejects_symbolic_indices() {
        // A symbolic index has no static selection, so neither an eager traversal nor an eager handle can carry it: the
        // operation that creates the view resolves the index.
        let symbolic = ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Symbolic(1) };
        let view: ArrayReferenceViewPath<NoReferenceViewBinding> =
            ArrayReferenceViewPath::root().with_view(symbolic.clone());
        assert_eq!(
            view.apply(&Array::matrix(3, 4, (1..=12).map(|value| value as f32).collect()).unwrap()),
            Err(TypeError::invalid(
                "a symbolic index has no static selection; the operation that creates the view resolves it",
            )
            .into()),
        );
        let root = ArrayReference::new(Array::matrix(3, 4, (1..=12).map(|value| value as f32).collect()).unwrap());
        let error = root.with_transform(symbolic).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ArrayReferenceViewError>(),
            Some(&ArrayReferenceViewError::SymbolicViewIndex),
        );
        assert_eq!(
            error.to_string(),
            "eager reference handles carry only static views; the operation that creates a symbolic view resolves it",
        );
    }

    #[test]
    fn test_array_reference_with_transform_is_structural() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap());
        let guard = root.lock_root().unwrap();
        let ReferenceReplacementPreparation::Prepared(prepared) = guard.prepare_replacement().unwrap() else {
            panic!("new reference unexpectedly has active read leases")
        };
        let transaction = prepared.begin(ReferenceCompletion::ready(Ok(())));

        // A derived handle is pure structural metadata over a live reference, so composing one must never resolve its
        // submitted work. The reference is parked in its `Taken` state, where every value access is unavailable behind
        // this retained guard until replacement commit, and derivation still computes its exact referent type.
        let transform = ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] };
        let derived = root.with_transform(transform).unwrap();
        assert_eq!(derived.r#type().as_ref(), &ReferenceType::new(ArrayType::new_static(DataType::F32, [2])));

        // Poisoning the submitted mutation is terminal for the alias family, but further derivation remains structural
        // composition. The resulting handle reports the reference failure only when it attempts to access state.
        transaction.poison("submission failed");
        let poisoned = ReferenceError::ExecutionPoisoned { reason: "submission failed".to_string() };
        assert_eq!(root.read().unwrap_err().downcast_custom::<ReferenceError>(), Some(&poisoned));
        let composed = derived
            .with_transform(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(0) })
            .unwrap();
        assert_eq!(composed.r#type().as_ref(), &ReferenceType::new(ArrayType::scalar(DataType::F32)));
        assert_eq!(composed.read().unwrap_err().downcast_custom::<ReferenceError>(), Some(&poisoned));

        let frozen = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]).unwrap());
        assert_eq!(frozen.freeze(), Ok(Array::vector(vec![1.0_f32, 2.0]).unwrap()));
        let frozen_view = frozen
            .with_transform(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(1) })
            .unwrap();
        assert_eq!(frozen_view.read().unwrap_err().downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_array_reference_read() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap());
        let derived =
            root.with_transform(ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] }).unwrap();

        // Reading a derived handle applies its selection rather than exposing the complete allocation.
        assert_eq!(root.read(), Ok(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap()));
        assert_eq!(derived.read(), Ok(Array::vector(vec![2.0_f32, 3.0]).unwrap()));
    }

    #[test]
    fn test_array_reference_read_root() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap());
        let derived =
            root.with_transform(ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] }).unwrap();
        assert_eq!(root.read_root(), Ok(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap()));

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
    fn test_array_reference_swap() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap());
        let view = root
            .with_transform(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(1) })
            .unwrap();
        assert_eq!(view.swap(Array::scalar(5.0_f32).unwrap()), Ok(Array::scalar(2.0_f32).unwrap()));
        assert_eq!(root.read(), Ok(Array::vector(vec![1.0_f32, 5.0, 3.0]).unwrap()));
    }

    #[test]
    fn test_array_reference_swap_rejects_wrong_referent_type() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap());
        let view = root.with_transform(ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(0, 3, 1)] }).unwrap();

        // Reconstruction alone accepts smaller replacements, so the handle checks exact view type equality.
        let error = view.swap(Array::vector(vec![10.0_f32, 20.0]).unwrap()).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceError>(),
            Some(&ReferenceError::ReferentTypeMismatch {
                expected: "f32[3]".to_string(),
                actual: "f32[2]".to_string(),
            }),
        );
        assert_eq!(root.read(), Ok(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap()));

        // A frozen allocation reports its terminal state before checking a malformed replacement.
        root.freeze().unwrap();
        let error = view.swap(Array::vector(vec![1.0_f32, 2.0]).unwrap()).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_array_reference_write() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap());
        let view = root
            .with_transform(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(1) })
            .unwrap();
        assert_eq!(view.write(Array::scalar(5.0_f32).unwrap()), Ok(()));
        assert_eq!(root.read(), Ok(Array::vector(vec![1.0_f32, 5.0, 3.0]).unwrap()));
    }

    #[test]
    fn test_array_reference_write_rejects_wrong_referent_type() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap());
        let view = root.with_transform(ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(0, 3, 1)] }).unwrap();

        let error = view.write(Array::vector(vec![10.0_f32, 20.0]).unwrap()).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceError>(),
            Some(&ReferenceError::ReferentTypeMismatch {
                expected: "f32[3]".to_string(),
                actual: "f32[2]".to_string(),
            }),
        );
        assert_eq!(root.read(), Ok(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap()));

        // A frozen allocation reports its terminal state before checking a malformed replacement.
        root.freeze().unwrap();
        let error = view.write(Array::vector(vec![1.0_f32, 2.0]).unwrap()).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_array_reference_write_reconstructs_composed_views() {
        let root = ArrayReference::new(Array::matrix(3, 3, (1..=9).map(|value| value as f32).collect()).unwrap());
        let view = root
            .with_transform(ArrayReferenceView::Slice {
                axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 2, 1)],
            })
            .unwrap()
            .with_transform(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(1) })
            .unwrap();
        assert_eq!(view.r#type().as_ref(), &ReferenceType::new(ArrayType::new_static(DataType::F32, [2])));

        // A write reconstructs both strict parents and preserves elements outside the composed view.
        assert_eq!(view.write(Array::vector(vec![70.0_f32, 80.0]).unwrap()), Ok(()));
        assert_eq!(
            root.read(),
            Ok(Array::matrix(3, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 70.0, 80.0, 9.0]).unwrap())
        );
    }

    #[test]
    fn test_array_reference_add_update() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap());
        let view = root
            .with_transform(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(1) })
            .unwrap();
        assert_eq!(view.add_update(&Array::scalar(5.0_f32).unwrap()), Ok(()));
        assert_eq!(root.read(), Ok(Array::vector(vec![1.0_f32, 7.0, 3.0]).unwrap()));
    }

    #[test]
    fn test_array_reference_add_update_rejects_wrong_referent_type() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap());
        let view = root.with_transform(ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(0, 3, 1)] }).unwrap();

        // An additive update whose result type drifts away from the view's element data type is rejected by the same
        // check, after the addition itself succeeded, so the holder still retains its previous value.
        let error = view.add_update(&Array::vector(vec![1.0_f64, 2.0, 3.0]).unwrap()).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceError>(),
            Some(&ReferenceError::ReferentTypeMismatch {
                expected: "f32[3]".to_string(),
                actual: "f64[3]".to_string(),
            }),
        );
        assert_eq!(root.read(), Ok(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap()));
    }

    #[test]
    fn test_array_reference_freeze() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]).unwrap());
        let view = root
            .with_transform(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(0) })
            .unwrap();
        let error = view.freeze().unwrap_err();
        assert_eq!(
            error.downcast_custom::<ArrayReferenceViewError>(),
            Some(&ArrayReferenceViewError::CannotFreezeView),
        );
        assert_eq!(error.to_string(), "cannot freeze a reference view; freeze the root reference instead");
        // Rejecting a derived handle leaves the allocation available for the root's consuming read.
        assert_eq!(root.freeze(), Ok(Array::vector(vec![1.0_f32, 2.0]).unwrap()));
        assert_eq!(view.read().unwrap_err().downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_array_reference_type() {
        let root_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let root = ArrayReference::new(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        let slice =
            ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(0, 2, 1), ArraySliceAxis::new(1, 2, 1)] };
        let index = ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(1) };
        let handle = root.with_transform(slice.clone()).unwrap().with_transform(index.clone()).unwrap();

        // Composition derives each handle type incrementally, which must agree with folding the complete mapping
        // over the root type in one step.
        let view: ArrayReferenceViewPath = ArrayReferenceViewPath::root().with_view(slice).with_view(index);
        assert_eq!(root.r#type().as_ref(), &ReferenceType::new(root_type.clone()));
        assert_eq!(handle.r#type().as_ref(), &ReferenceType::new(view.output_type(&root_type).unwrap()));
        assert_eq!(handle.clone().r#type(), handle.r#type());
        assert_eq!(handle.to_string(), "ref<f32[2]>");
        assert_eq!(root.to_string(), "ref<f32[2, 3]>");
    }

    #[test]
    fn test_array_reference_analysis_new() {
        // A root matrix reference is narrowed to a row slice, then to the contents of that row, while an overlapping
        // sibling view selects one column directly from the root; both leaves are read.
        let matrix_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let mut builder = TestBuilder::new();
        let matrix = builder.add_input(ArrayIrType::Reference(ReferenceType::new(matrix_type.clone())));
        let row_axes = vec![ArraySliceAxis::new(0, 1, 1), ArraySliceAxis::new(0, 3, 1)];
        let row = builder
            .add_instruction(ReferenceSliceOperation::new(row_axes.clone()), Vec::new(), vec![matrix], None)
            .unwrap()[0];
        let row_contents =
            builder.add_instruction(ReferenceIndexOperation::new(0, 0), Vec::new(), vec![row], None).unwrap()[0];
        let column =
            builder.add_instruction(ReferenceIndexOperation::new(1, 2), Vec::new(), vec![matrix], None).unwrap()[0];
        let row_contents_value = builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![row_contents], None)
            .unwrap()[0];
        let column_value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![column], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![row_contents_value, column_value],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        let analysis = ArrayReferenceAnalysis::new(program.entry_region_ref(), 0).unwrap();
        let row_view = ArrayReferenceViewPath::root().with_view(ArrayReferenceView::Slice { axes: row_axes.clone() });
        let row_contents_view =
            row_view.with_view(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(0) });
        let column_view = ArrayReferenceViewPath::root()
            .with_view(ArrayReferenceView::Index { axis: 1, index: ArrayReferenceViewIndex::Static(2) });
        assert_eq!(analysis.path(value_id(0, 0)), Some(&ArrayReferenceViewPath::root()));
        assert_eq!(analysis.path(value_id(0, 1)), Some(&row_view));
        assert_eq!(analysis.path(value_id(0, 2)), Some(&row_contents_view));
        assert_eq!(analysis.path(value_id(0, 3)), Some(&column_view));
        assert_eq!(analysis.path(value_id(0, 4)), None);
        assert_eq!(analysis.path(value_id(0, 5)), None);

        // Each composed view reproduces its declared array shape. Indexing the sliced row and indexing a column
        // directly from the matrix select different indices even though both refer to the same array.
        assert_eq!(row_view.output_type(&matrix_type), Ok(ArrayType::new_static(DataType::F32, [1, 3])));
        assert_eq!(row_contents_view.output_type(&matrix_type), Ok(ArrayType::new_static(DataType::F32, [3])));
        assert_eq!(column_view.output_type(&matrix_type), Ok(ArrayType::new_static(DataType::F32, [2])));
        assert_ne!(row_contents_view, column_view);
        assert_eq!(analysis.path(value_id(1, 0)), None);
    }

    #[test]
    fn test_array_reference_discharge_storage_alias() {
        let alias = <ArrayReferenceDischarge as ReferenceDischargePolicy<TestContext>>::storage_alias(
            &ArrayType::new_static(DataType::F32, [3]),
        );
        assert_eq!(alias, ArrayReferenceViewPath::root());
    }

    #[test]
    fn test_array_reference_discharge_read() {
        let context = EagerContext::<TestValue, TestOperation>::new();
        let current = TestValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap());
        let alias = ArrayReferenceViewPath::root()
            .with_view(ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] });
        assert_eq!(
            ArrayReferenceDischarge::read(&context, &current, &alias),
            Ok(TestValue::Array(Array::vector::<f32>(vec![2.0, 3.0]).unwrap())),
        );
        assert_eq!(ArrayReferenceDischarge::read(&context, &current, &ArrayReferenceViewPath::root()), Ok(current));
    }

    #[test]
    fn test_array_reference_discharge_write() {
        let alias = ArrayReferenceViewPath::root()
            .with_view(ArrayReferenceView::Slice {
                axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 2, 1)],
            })
            .with_view(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(1) });
        let stage = |inputs: Vec<Tracer<TracingContext<TestValue, TestOperation>>>| {
            let context = inputs[0].context().clone();
            Ok(vec![ArrayReferenceDischarge::write(&context, &inputs[0], inputs[1].clone(), &alias)?])
        };
        let (_, staged): (_, Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>) =
            EagerContext::<TestValue, TestOperation>::trace(
                stage,
                vec![
                    ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3, 3])),
                    ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])),
                ],
            )
            .unwrap();
        assert_eq!(
            staged.to_string(),
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
        let context = EagerContext::<TestValue, TestOperation>::new();
        let current = TestValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap());
        let alias = ArrayReferenceViewPath::root()
            .with_view(ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] });
        assert_eq!(
            ArrayReferenceDischarge::swap(
                &context,
                &current,
                TestValue::Array(Array::vector::<f32>(vec![4.0, 5.0]).unwrap()),
                &alias
            ),
            Ok((
                TestValue::Array(Array::vector::<f32>(vec![2.0, 3.0]).unwrap()),
                TestValue::Array(Array::vector(vec![1.0_f32, 4.0, 5.0]).unwrap())
            )),
        );
    }

    #[test]
    fn test_array_reference_discharge_swap_reconstructs_composed_view() {
        // Swapping through an index composed onto a slice must write back through both steps in reverse order, so the
        // discharged program reconstructs the sliced block from the squeezed row before writing it into the allocation.
        let matrix_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Static(3)]));
        let row_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(matrix_type.clone().into());
        let replacement = builder.add_input(row_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let block = builder
            .add_instruction(
                ReferenceSliceOperation::new(vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 2, 1)]),
                Vec::new(),
                vec![reference],
                None,
            )
            .unwrap()[0];
        let row =
            builder.add_instruction(ReferenceIndexOperation::new(0, 1), Vec::new(), vec![block], None).unwrap()[0];
        let old = builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![row, replacement], None)
            .unwrap()[0];
        let final_snapshot =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![old, final_snapshot],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let inputs = vec![
            TestValue::Array(
                Array::from_f64s(matrix_type.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap(),
            ),
            TestValue::Array(Array::vector::<f32>(vec![10.0, 20.0]).unwrap()),
        ];
        let expected = vec![
            TestValue::Array(Array::vector::<f32>(vec![7.0, 8.0]).unwrap()),
            TestValue::Array(
                Array::from_f64s(matrix_type, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 9.0]).unwrap(),
            ),
        ];
        assert_eq!(source.clone().interpret(inputs.clone()), Ok(expected.clone()));

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.program().interpret(inputs), Ok(expected));
        assert_eq!(
            discharged.program().to_string(),
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
        let context = EagerContext::<TestValue, TestOperation>::new();
        let current = TestValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap());
        let alias = ArrayReferenceViewPath::root()
            .with_view(ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] });
        assert_eq!(
            ArrayReferenceDischarge::accumulate(
                &context,
                &current,
                TestValue::Array(Array::vector::<f32>(vec![4.0, 5.0]).unwrap()),
                &alias
            ),
            Ok(TestValue::Array(Array::vector(vec![1.0_f32, 6.0, 8.0]).unwrap())),
        );
    }

    #[test]
    fn test_array_reference_discharge_accumulate_stages_composed_view_accesses() {
        // The policy is the interpreter-side half of array reference discharge, so this test pins the exact instruction
        // sequence each of its three alias applications stages, over a composed index-of-slice view of a 3x3
        // allocation. Each access materializes the allocation-to-handle chain against the state it observes, so the
        // chain is restaged per access rather than shared, and a replacement and an accumulation then write their new
        // leaf back through that chain in reverse. The alias is closed over context values, and a static chain
        // binds none of them.
        let alias: ArrayReferenceViewPath<Tracer<TestContext>> = ArrayReferenceViewPath::root()
            .with_view(ArrayReferenceView::Slice {
                axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 2, 1)],
            })
            .with_view(ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(1) });
        let stage = |inputs: Vec<Tracer<TracingContext<TestValue, TestOperation>>>| {
            let context = inputs[0].context().clone();
            let read = ArrayReferenceDischarge::read(&context, &inputs[0], &alias)?;
            let (previous, replaced) = ArrayReferenceDischarge::swap(&context, &inputs[0], inputs[1].clone(), &alias)?;
            let accumulated = ArrayReferenceDischarge::accumulate(&context, &replaced, inputs[1].clone(), &alias)?;
            Ok(vec![read, previous, replaced, accumulated])
        };
        let matrix_type = ArrayType::new_static(DataType::F32, [3, 3]);
        let (_, staged): (_, Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>) =
            EagerContext::<TestValue, TestOperation>::trace(
                stage,
                vec![
                    ArrayIrType::Array(matrix_type.clone()),
                    ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])),
                ],
            )
            .unwrap();
        assert_eq!(
            staged.to_string(),
            indoc! {"
                lambda %0:f32[3, 3], %1:f32[2] .
                let %2:f32[2, 2] = slice [start_indices=[1, 0], limit_indices=[3, 2]] %0
                    %3:f32[1, 2] = slice [start_indices=[1, 0], limit_indices=[2, 2]] %2
                    %4:f32[2] = reshape [shape=[2]] %3
                    %5:f32[2, 2] = slice [start_indices=[1, 0], limit_indices=[3, 2]] %0
                    %6:f32[1, 2] = slice [start_indices=[1, 0], limit_indices=[2, 2]] %5
                    %7:f32[2] = reshape [shape=[2]] %6
                    %8:f32[1, 2] = reshape [shape=[1, 2]] %1
                    %9:f32[2, 2] = update_slice [start_indices=[1, 0]] %5 %8
                    %10:f32[3, 3] = update_slice [start_indices=[1, 0]] %0 %9
                    %11:f32[2, 2] = slice [start_indices=[1, 0], limit_indices=[3, 2]] %10
                    %12:f32[1, 2] = slice [start_indices=[1, 0], limit_indices=[2, 2]] %11
                    %13:f32[2] = reshape [shape=[2]] %12
                    %14:f32[2] = add %13 %1
                    %15:f32[1, 2] = reshape [shape=[1, 2]] %14
                    %16:f32[2, 2] = update_slice [start_indices=[1, 0]] %11 %15
                    %17:f32[3, 3] = update_slice [start_indices=[1, 0]] %10 %16
                in (%4, %7, %10, %17)"},
        );
    }

    #[test]
    fn test_array_reference_discharge_accumulate_reconstructs_composed_slices() {
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let pair_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(vector_type.into());
        let replacement = builder.add_input(pair_type.clone().into());
        let update = builder.add_input(pair_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let indexed = builder
            .add_instruction(ReferenceIndexOperation::new(0, 3), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let indexed_snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![indexed], None).unwrap()[0];
        let outer = builder
            .add_instruction(
                ReferenceSliceOperation::new(vec![ArraySliceAxis::new(1, 3, 1)]),
                Vec::new(),
                vec![reference],
                None,
            )
            .unwrap()[0];
        let composed = builder
            .add_instruction(
                ReferenceSliceOperation::new(vec![ArraySliceAxis::new(0, 2, 1)]),
                Vec::new(),
                vec![outer],
                None,
            )
            .unwrap()[0];
        let old = builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![composed, replacement], None)
            .unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![composed, update], None)
            .unwrap();
        let final_snapshot =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![indexed_snapshot, old, final_snapshot],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();
        let inputs = vec![
            TestValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap()),
            TestValue::Array(Array::vector::<f32>(vec![10.0, 20.0]).unwrap()),
            TestValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()),
        ];
        let expected = source.interpret(inputs.clone()).unwrap();

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.program().interpret(inputs), Ok(expected),);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[4], %1:f32[2], %2:f32[2] .
                let %3:f32[1] = slice [start_indices=[3], limit_indices=[4]] %0
                    %4:f32[] = reshape [shape=[]] %3
                    %5:f32[3] = slice [start_indices=[1], limit_indices=[4]] %0
                    %6:f32[2] = slice [start_indices=[0], limit_indices=[2]] %5
                    %7:f32[3] = update_slice [start_indices=[0]] %5 %1
                    %8:f32[4] = update_slice [start_indices=[1]] %0 %7
                    %9:f32[3] = slice [start_indices=[1], limit_indices=[4]] %8
                    %10:f32[2] = slice [start_indices=[0], limit_indices=[2]] %9
                    %11:f32[2] = add %10 %2
                    %12:f32[3] = update_slice [start_indices=[0]] %9 %11
                    %13:f32[4] = update_slice [start_indices=[1]] %8 %12
                in (%4, %6, %13)"},
        );
    }

    #[test]
    fn test_array_reference_discharge_accumulate_reconstructs_removed_axis() {
        let matrix_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let row_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(matrix_type.clone().into());
        let replacement = builder.add_input(row_type.clone().into());
        let update = builder.add_input(row_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let row = builder
            .add_instruction(ReferenceIndexOperation::new(0, 1), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let old = builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![row, replacement], None)
            .unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![row, update], None)
            .unwrap();
        let final_snapshot =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![old, final_snapshot],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();
        let inputs = vec![
            TestValue::Array(Array::from_f64s(matrix_type.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
            TestValue::Array(Array::vector::<f32>(vec![10.0, 20.0, 30.0]).unwrap()),
            TestValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap()),
        ];
        let expected = vec![
            TestValue::Array(Array::vector::<f32>(vec![4.0, 5.0, 6.0]).unwrap()),
            TestValue::Array(Array::from_f64s(matrix_type, vec![1.0, 2.0, 3.0, 11.0, 22.0, 33.0]).unwrap()),
        ];
        assert_eq!(source.clone().interpret(inputs.clone()), Ok(expected.clone()));

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.program().interpret(inputs), Ok(expected));
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[2, 3], %1:f32[3], %2:f32[3] .
                let %3:f32[1, 3] = slice [start_indices=[1, 0], limit_indices=[2, 3]] %0
                    %4:f32[3] = reshape [shape=[3]] %3
                    %5:f32[1, 3] = reshape [shape=[1, 3]] %1
                    %6:f32[2, 3] = update_slice [start_indices=[1, 0]] %0 %5
                    %7:f32[1, 3] = slice [start_indices=[1, 0], limit_indices=[2, 3]] %6
                    %8:f32[3] = reshape [shape=[3]] %7
                    %9:f32[3] = add %8 %2
                    %10:f32[1, 3] = reshape [shape=[1, 3]] %9
                    %11:f32[2, 3] = update_slice [start_indices=[1, 0]] %6 %10
                in (%4, %11)"},
        );
    }

    #[test]
    fn test_array_reference_discharge_symbolic_indices() {
        // Stage a composed view: select a runtime row, then its last two columns. Updating the leaf must preserve
        // both the rest of that row and every other row of the shared root.
        let stage = |inputs: Vec<Tracer<TestContext>>| {
            let context = inputs[0].context().clone();
            let alias = ArrayReferenceViewPath::root()
                .with_step(
                    ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Symbolic(1) },
                    vec![inputs[1].clone()],
                )
                .with_view(ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] });
            let selected = ArrayReferenceDischarge::read(&context, &inputs[0], &alias)?;
            let written = ArrayReferenceDischarge::write(&context, &inputs[0], inputs[2].clone(), &alias)?;
            let (previous, swapped) = ArrayReferenceDischarge::swap(&context, &inputs[0], inputs[2].clone(), &alias)?;
            let accumulated = ArrayReferenceDischarge::accumulate(&context, &inputs[0], inputs[2].clone(), &alias)?;
            Ok(vec![selected, written, previous, swapped, accumulated])
        };
        let matrix_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let (_, staged): (_, Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>) =
            EagerContext::<TestValue, TestOperation>::trace(
                stage,
                vec![
                    ArrayIrType::Array(matrix_type.clone()),
                    ArrayIrType::Array(ArrayType::scalar(DataType::I32)),
                    ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])),
                ],
            )
            .unwrap();
        // Dynamic slicing clamps negative indices to the first row and oversized indices to the last row.
        for (index, selected, written, accumulated) in [
            (-8, vec![2.0, 3.0], vec![1.0, 20.0, 30.0, 4.0, 5.0, 6.0], vec![1.0, 22.0, 33.0, 4.0, 5.0, 6.0]),
            (1, vec![5.0, 6.0], vec![1.0, 2.0, 3.0, 4.0, 20.0, 30.0], vec![1.0, 2.0, 3.0, 4.0, 25.0, 36.0]),
            (80, vec![5.0, 6.0], vec![1.0, 2.0, 3.0, 4.0, 20.0, 30.0], vec![1.0, 2.0, 3.0, 4.0, 25.0, 36.0]),
        ] {
            let selected = TestValue::Array(Array::vector::<f32>(selected).unwrap());
            let written = TestValue::Array(Array::from_f64s(matrix_type.clone(), written).unwrap());
            assert_eq!(
                staged.clone().interpret(vec![
                    TestValue::Array(
                        Array::from_f64s(matrix_type.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()
                    ),
                    TestValue::Array(Array::scalar::<i32>(index).unwrap()),
                    TestValue::Array(Array::vector::<f32>(vec![20.0, 30.0]).unwrap()),
                ]),
                Ok(vec![
                    selected.clone(),
                    written.clone(),
                    selected,
                    written,
                    TestValue::Array(Array::from_f64s(matrix_type.clone(), accumulated).unwrap()),
                ]),
            );
        }
    }
}
