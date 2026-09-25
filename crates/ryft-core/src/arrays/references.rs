//! Array reference handles, index mappings, analysis, and discharge for the array IR.
//!
//! [`ArrayReferenceTransform`] describes array indexing and slicing. [`ArrayReferenceTransformPath`] composes those
//! transforms into a mapping from a root array to a selected view, and [`ArrayReference`] pairs a static mapping with
//! an eager reference allocation. Reads select the elements at the mapped indices; mutations reconstruct the root
//! through the same transforms in reverse order, preserving values outside the view.
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
//! An [`Index`](ArrayReferenceTransform::Index) transform indexes one array axis using a static index or a dynamic
//! binding. Each dynamic transform consumes the next ordinary input in the access's binding group; analysis records its
//! [`ValueId`] in the corresponding bound transform. Eager handles carry [`NoReferenceTransformBinding`] and accept
//! only static transforms. Discharge paths store context values as bindings and reconstruct dynamically indexed views
//! through dynamic slicing and updates. Following the array dynamic-slicing contract, a negative runtime index counts
//! from the end of the selected axis once, and the result is then clamped to that axis.

use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use thiserror::Error;

use ryft_macros::Parameter;

use crate::arrays::addressing::ArraySliceAxis;
use crate::arrays::ir::ArrayIrValue;
use crate::arrays::operations::{ArrayIrOperation, ArrayOperation};
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
    ProgramError, ReadyOrPendingReferenceGuard, Reference, ReferenceAccessDescriptor, ReferenceAccessOperation,
    ReferenceAccumulationPolicy, ReferenceDischargePolicy, ReferenceDischargeableType, ReferenceError, ReferenceId,
    ReferenceTransform, ReferenceTransformPath, ReferenceType, ReferenceView, ReferenceViewAnalysis,
    ReferenceViewOverlap, Type, TypeError, TypeIdentityRenaming, Typed, Value, ValueId,
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

    /// A transform with a dynamic index was composed onto an eager handle, whose path carries only static transforms.
    #[error("eager reference handles carry only static transforms; dynamic indices are resolved by each access")]
    DynamicTransformIndex,
}

impl<Root: Typed, Binding: Clone + Typed<Type = ArrayIrType>> ReferenceView<Root, ArrayReferenceTransform, Binding> {
    /// Selects a static position on `axis`, removing that dimension from the viewed referent.
    pub fn index(self, axis: usize, index: usize) -> Result<Self, ProgramError> {
        self.with_transform(
            ArrayReferenceTransform::Index { axis, index: ArrayReferenceTransformIndex::Static(index) },
            Vec::new(),
        )
    }

    /// Selects the supplied static unit-stride ranges from the currently viewed referent.
    pub fn slice(self, axes: &[ArraySliceAxis]) -> Result<Self, ProgramError> {
        self.with_transform(ArrayReferenceTransform::Slice { axes: axes.to_vec() }, Vec::new())
    }

    /// Selects a dynamic position on `axis`. Construction validates the scalar integer binding's type and memory
    /// space; concretization and negative-index normalization happen at the eventual access.
    pub fn dynamic_index(self, axis: usize, index: &Binding) -> Result<Self, ProgramError> {
        self.with_transform(
            ArrayReferenceTransform::Index { axis, index: ArrayReferenceTransformIndex::Dynamic },
            vec![index.clone()],
        )
    }
}

/// One validated index transform in an [`ArrayReferenceTransformPath`]'s root-to-handle mapping.
///
/// A transform describes both directions of one selection: applying it extracts a selected child value from its
/// parent, while replacing that child reconstructs a value with exactly the parent's original type. This
/// bidirectional contract lets reference reads operate on the selected value and lets write-only replacements, swaps,
/// or additive updates reconstruct the shared root without changing its declared type. A write-only traversal
/// materializes the strict parents needed for reconstruction but deliberately skips extracting the overwritten leaf.
///
/// Transforms are interpreted in order from the root outward. [`Index`](Self::Index) removes one axis at a static or
/// dynamic index; [`Slice`](Self::Slice) preserves rank and selects one static unit-stride range per axis. A dynamic
/// index is supplied by a binding input of the access that carries the transform. For example, the built-in scan binds
/// its explicit body index to a leading dynamic index on each access to a stacked reference. Eager handles resolve
/// dynamic indices into static transforms when an access applies its path. Discharge reconstructs dynamic indices with
/// dynamic slicing; strided slicing remains unsupported.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
#[non_exhaustive]
pub enum ArrayReferenceTransform {
    /// Selects a position along one axis and removes that axis from the view shape.
    Index {
        /// Axis selected in the transform's input view.
        axis: usize,

        /// Index selected on `axis`.
        index: ArrayReferenceTransformIndex,
    },

    /// Selects one static unit-stride range on every axis while preserving rank.
    Slice {
        /// Per-axis selections in the transform's input view.
        axes: Vec<ArraySliceAxis>,
    },
}

/// Index selected by an [`Index`](ArrayReferenceTransform::Index) transform.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum ArrayReferenceTransformIndex {
    /// An index known when the transform is described.
    Static(usize),

    /// An index supplied by the next ordinary input in the access's binding sequence.
    Dynamic,
}

impl ArrayReferenceTransform {
    /// Returns the exact canonical array type produced from `input`. A symbolic index removes its axis exactly like a
    /// static one, without the static bounds check and reconstruction proof, because the index it selects is only
    /// known to the access that applies the transform.
    pub fn output_type(&self, input: &ArrayType) -> Result<ArrayType, TypeError> {
        let (output, selection) = self.selected_type(input)?;
        if let Some(selection) = selection {
            self.validate_reconstruction(input, &output, &selection)?;
        }
        Ok(output)
    }

    /// Returns the exact canonical array type selected from `input` together with the static selection that produced
    /// it, or [`None`] for a dynamic index, whose selection is only known to the access that applies it. This is
    /// [`output_type`](Self::output_type) without the write-back proof, which read-only accesses do not need.
    fn selected_type(&self, input: &ArrayType) -> Result<(ArrayType, Option<TransformSelection>), TypeError> {
        if let Self::Index { axis, index: ArrayReferenceTransformIndex::Dynamic } = self {
            Self::indexed_shape(*axis, input)?;
            return Ok((input.without_dimension(*axis)?.0, None));
        }
        let selection = self.selection(input)?;
        let sliced = input
            .slice(selection.starts.as_slice(), selection.limits.as_slice(), &vec![1; selection.starts.len()])
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        let output = match &selection.squeezed_output_shape {
            Some(shape) => sliced.reshape(shape.clone()).map_err(|error| TypeError::invalid(error.to_string()))?,
            None => sliced,
        };
        Ok((output, Some(selection)))
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
                Ok(TransformSelection { starts, limits, squeezed_output_shape: Some(output_shape) })
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
                Ok(TransformSelection { starts, limits, squeezed_output_shape: None })
            }
        }
    }

    /// Applies this transform to one carried parent value. A symbolic index is resolved by the carrier from the one
    /// value the transform's `bindings` close it over; a symbolic transform that binds no value (an eager path, or a
    /// malformed closure) has no selection and is rejected by [`selection`](Self::selection).
    fn apply_in<C: TransformReadCarrier>(
        &self,
        carrier: &mut C,
        input: &C::Value,
        bindings: &[C::Binding],
    ) -> Result<C::Value, ProgramError> {
        if let (Self::Index { axis, index: ArrayReferenceTransformIndex::Dynamic }, [binding]) = (self, bindings) {
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
    fn replace_in<C: TransformWriteCarrier>(
        &self,
        carrier: &mut C,
        input: &C::Value,
        replacement: &C::Value,
        bindings: &[C::Binding],
    ) -> Result<C::Value, ProgramError> {
        if let (Self::Index { axis, index: ArrayReferenceTransformIndex::Dynamic }, [binding]) = (self, bindings) {
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
    /// transform whose forward selection and inverse update do not round-trip on that metadata. The proof runs when a
    /// view is constructed and when an access that writes back derives its path type. Read-only accesses derive their
    /// path types through [`ReferenceTransform::read_type`], which skips it.
    fn validate_reconstruction(
        &self,
        input: &ArrayType,
        output: &ArrayType,
        selection: &TransformSelection,
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
            "reference transform reconstruction changes root type from `{input}` to `{reconstructed}`",
        )))
    }
}

impl Display for ArrayReferenceTransform {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Index { axis, index: ArrayReferenceTransformIndex::Static(index) } => {
                write!(formatter, "index(axis={axis}, index={index})")
            }
            Self::Index { axis, index: ArrayReferenceTransformIndex::Dynamic } => {
                write!(formatter, "dynamic_index(axis={axis})")
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

    fn output_type(&self, input: &ArrayType) -> Result<ArrayType, TypeError> {
        self.output_type(input)
    }

    fn read_type(&self, input: &ArrayType) -> Result<ArrayType, TypeError> {
        // Reads never write back through the view, so they skip the reconstruction proof of `output_type`.
        Ok(self.selected_type(input)?.0)
    }

    // Both paths fold to one range or symbolic index per root axis. Nonintersecting static ranges prove disjointness;
    // identical static ranges or symbolic indices with equal bindings, offsets, and clamping extents prove equality.
    // Everything else may overlap. A malformed path cannot be folded and is treated as
    // possibly overlapping, because paths are validated when they are derived and this query must not fail.
    fn overlap(
        r#type: &ArrayIrType,
        lhs: &[BoundReferenceTransform<Self>],
        rhs: &[BoundReferenceTransform<Self>],
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

impl BatchableReferenceTransform for ArrayReferenceTransform {
    // The batch axis of a reference is an axis of its packed referent that the per-item transform never sees.
    // Indexing removes one per-item axis, so the packed transform cannot keep both axis positions unchanged: a batch
    // axis at or before the indexed axis shifts the packed indexed axis one position later while the output keeps the
    // batch axis, and a batch axis after the indexed axis leaves the packed indexed axis alone while the output's batch
    // axis moves one position earlier. Slicing preserves rank, so the packed transform selects the complete batch axis
    // through an identity selection inserted at the batch axis position and the output keeps the batch axis.
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

/// Indices that a folded [`ArrayReferenceTransformPath`] selects on one axis of its root, used by
/// [`ReferenceTransform::overlap`] to compare two paths of one root.
#[derive(Clone, Debug, PartialEq, Eq)]
enum RootIndexSelection {
    /// A static unit-stride range `[start, limit)` of the root axis. Before any transform touches the axis this is the
    /// complete axis, a slice narrows it, and a static index collapses it to one index.
    Range {
        /// Inclusive start of the range.
        start: usize,

        /// Exclusive limit of the range.
        limit: usize,
    },

    /// One index `offset + clamp(wrap(symbol), 0, extent - 1)` of the root axis, selected relative to the range that
    /// earlier transforms narrowed the axis to, where `wrap(symbol)` is `symbol + extent` for a negative `symbol` and
    /// `symbol` otherwise. Both wrapping and clamping depend on this extent, not just the binding.
    Symbolic {
        /// Binding of the symbolic index.
        binding: ValueId,

        /// Start of the narrowed range that the symbolic index is relative to.
        offset: usize,

        /// Size of the narrowed axis against which the runtime index is wrapped and clamped.
        extent: usize,
    },
}

impl RootIndexSelection {
    /// Folds the closed `bound_transforms` of a path over a root of static shape `shape` into one range or symbolic
    /// index per root axis, or [`None`] when the path is malformed for that root (an axis, index, binding, or stride
    /// that the derivation would have rejected).
    fn fold(
        shape: &StaticShape,
        bound_transforms: &[BoundReferenceTransform<ArrayReferenceTransform>],
    ) -> Option<Vec<Self>> {
        let mut indices =
            shape.dimensions().iter().map(|size| Self::Range { start: 0, limit: *size }).collect::<Vec<_>>();
        // Root axes that the folded transforms have not indexed away yet, in view axis order.
        let mut remaining = (0..shape.rank()).collect::<Vec<_>>();
        for bound_transform in bound_transforms {
            match bound_transform.transform() {
                ArrayReferenceTransform::Index { axis, index } => {
                    if *axis >= remaining.len() {
                        return None;
                    }
                    let root_axis = remaining.remove(*axis);
                    let Self::Range { start, limit } = indices[root_axis] else {
                        return None;
                    };
                    indices[root_axis] = match index {
                        ArrayReferenceTransformIndex::Static(index) => {
                            // Invalid paths must remain conservative even when the relative index overflows.
                            let index = start.checked_add(*index)?;
                            if index >= limit {
                                return None;
                            }
                            Self::Range { start: index, limit: index + 1 }
                        }
                        ArrayReferenceTransformIndex::Dynamic => {
                            let binding = *bound_transform.bindings().first()?;
                            Self::Symbolic { binding, offset: start, extent: limit - start }
                        }
                    };
                }
                ArrayReferenceTransform::Slice { axes } => {
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

/// Normalized indices of one [`ArrayReferenceTransform`] applied to one statically shaped input.
///
/// Both transform kinds reduce to slicing one unit-stride hyper-rectangle out of the input, optionally followed by
/// squeezing the indexed axis. Normalizing to this shared form lets every consumer (type derivation, eager reads,
/// eager update reconstruction, and staged discharge) share one validation and address computation.
struct TransformSelection {
    /// Inclusive slice start per input axis.
    starts: Vec<usize>,

    /// Exclusive slice limit per input axis.
    limits: Vec<usize>,

    /// Exact static output shape after squeezing the indexed axis, for
    /// [`ArrayReferenceTransform::Index`] transforms only; [`None`] for rank-preserving slices, whose output
    /// shape is exactly [`Self::update_shape`].
    squeezed_output_shape: Option<Shape>,
}

impl TransformSelection {
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

/// Immutable index mapping between a shared array-reference root and one derived handle: the array specialization of
/// the generic [`ReferenceTransformPath`], whose transforms are [`ArrayReferenceTransform`]s.
///
/// The mapping stores validated transforms in root-to-handle order. The empty mapping ([`root`](Self::root)) is the
/// identity view and denotes the complete root. Each additional transform is applied to the preceding view, so
/// indexing or slicing an already-derived [`ArrayReference`] composes onto the same shared root rather than creating
/// another mutable resource.
///
/// This type is structural metadata only: it owns neither the referenced array nor its resource identity, liveness, or
/// synchronization state. [`ArrayReference`] pairs it with a handle to the shared reference allocation. In staged
/// programs, [`ArrayReferenceAnalysis`] records one path per reference access, keyed by its instruction and root
/// input index. The path determines the viewed referent and addressed indices; mutations reconstruct the root by
/// applying the inverse update of each transform in reverse order. Overlapping views may address the same root indices
/// and observe one another's ordered mutations, while equality and hashing distinguish different transform sequences.
///
/// `Binding` supplies dynamic indices: [`ValueId`] identifies program values, the uninhabited
/// [`NoReferenceTransformBinding`] restricts eager handles to static transforms, and `C::Value` binds discharge indices
/// directly to context values. Supported index transforms are described by [`ArrayReferenceTransform`]. Attached-region
/// and external runtime boundaries pass complete root handles, and each access inside the receiving scope carries its
/// own path. For example, each access in a scan body selects one item of a stacked reference through a dynamic index
/// bound to the body's explicit index input.
pub type ArrayReferenceTransformPath<Binding = ValueId> = ReferenceTransformPath<ArrayReferenceTransform, Binding>;

impl ArrayReferenceTransformPath {
    /// Returns a static, rank-preserving slice of the root selecting the same elements as this path. Indexed axes
    /// become size-one ranges. Symbolic indices, dynamic root shapes, and invalid compositions return `None` rather
    /// than claiming a definite selection. The returned transform describes root coordinates, not the path's result
    /// rank.
    pub fn root_slice(&self, root_type: &ArrayType) -> Option<ArrayReferenceTransform> {
        let indices = RootIndexSelection::fold(&root_type.static_shape()?, self.bound_transforms())?;
        let axes = indices
            .into_iter()
            .map(|selection| match selection {
                RootIndexSelection::Range { start, limit } => Some(ArraySliceAxis::new(start, limit - start, 1)),
                RootIndexSelection::Symbolic { .. } => None,
            })
            .collect::<Option<Vec<_>>>()?;
        Some(ArrayReferenceTransform::Slice { axes })
    }
}

impl<Binding> ArrayReferenceTransformPath<Binding> {
    /// Returns the exact view type derived from `root_type`.
    pub fn output_type(&self, root_type: &ArrayType) -> Result<ArrayType, TypeError> {
        self.transforms().try_fold(root_type.clone(), |r#type, transform| transform.output_type(&r#type))
    }

    /// Returns the root followed by each selected child, ending with the value this path selects. An empty path returns
    /// only the root. The bindings of each bound transform are handed to `carrier`, which resolves symbolic indices;
    /// reconstruction uses every snapshot except the final child as its strict parents.
    fn intermediates_in<C: TransformReadCarrier<Binding = Binding>>(
        &self,
        carrier: &mut C,
        root: C::Value,
    ) -> Result<Vec<C::Value>, ProgramError> {
        let mut intermediates = Vec::with_capacity(self.bound_transforms().len() + 1);
        intermediates.push(root);
        for bound_transform in self.bound_transforms() {
            let parent = intermediates.last().unwrap();
            let child = bound_transform.transform().apply_in(carrier, parent, bound_transform.bindings())?;
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
    fn reconstruct_in<C: TransformWriteCarrier<Binding = Binding>>(
        &self,
        carrier: &mut C,
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

    /// Replaces this view's selected elements through `carrier`, returning their previous snapshot plus the
    /// reconstructed root, so that the eager swap and the discharge-time replacement share one traversal.
    fn swap_in<C: TransformWriteCarrier<Value: Clone, Binding = Binding>>(
        &self,
        carrier: &mut C,
        root: C::Value,
        replacement: C::Value,
    ) -> Result<(C::Value, C::Value), ProgramError> {
        let intermediates = self.intermediates_in(carrier, root)?;

        // The traversal always pushes the root itself first, so the chain is never empty and its last snapshot is
        // the value this view selects.
        let previous = intermediates.last().unwrap().clone();
        let reconstructed =
            self.reconstruct_in(carrier, &intermediates[..self.bound_transforms().len()], replacement)?;
        Ok((previous, reconstructed))
    }

    /// Replaces this view's selected elements through `carrier` without materializing the selected old value.
    ///
    /// Immutable root reconstruction still needs each strict parent of the selected leaf so indices outside the logical
    /// view survive. The final transform is deliberately not applied: its output is exactly the old selected value that
    /// write-only semantics must not observe. An identity view therefore returns `replacement` directly.
    fn write_in<C: TransformWriteCarrier<Binding = Binding>>(
        &self,
        carrier: &mut C,
        root: C::Value,
        replacement: C::Value,
    ) -> Result<C::Value, ProgramError> {
        let Some((_, parents)) = self.bound_transforms().split_last() else {
            return Ok(replacement);
        };
        let mut intermediates = Vec::with_capacity(self.bound_transforms().len());
        intermediates.push(root);
        for bound_transform in parents {
            let parent = intermediates.last().unwrap();
            let child = bound_transform.transform().apply_in(carrier, parent, bound_transform.bindings())?;
            intermediates.push(child);
        }
        self.reconstruct_in(carrier, intermediates.as_slice(), replacement)
    }
}

impl ArrayReferenceTransformPath<NoReferenceTransformBinding> {
    /// Applies the complete static mapping to one root snapshot.
    fn apply<A>(&self, root: &A) -> Result<A, ProgramError>
    where
        A: Value<Type = ArrayType> + Reshape + Slice,
    {
        let mut carrier = EagerTransformCarrier(PhantomData);
        self.bound_transforms().iter().try_fold(root.clone(), |value, bound_transform| {
            bound_transform.transform().apply_in(&mut carrier, &value, bound_transform.bindings())
        })
    }

    /// Replaces the elements this static path selects and returns the reconstructed root plus their old snapshot.
    fn swap<A>(&self, root: &A, replacement: &A) -> Result<(A, A), ProgramError>
    where
        A: Value<Type = ArrayType> + Reshape + Slice + UpdateSlice,
    {
        let (old, reconstructed) =
            self.swap_in(&mut EagerTransformCarrier(PhantomData), root.clone(), replacement.clone())?;
        Ok((reconstructed, old))
    }
}

/// Operation-family constructors for the canonical array operations that one array-reference transform path traversal
/// stages.
///
/// Mapping between a reference root and one derived handle's selected elements uses static or dynamic slices,
/// reshapes, and corresponding updates. Both eager handles and the
/// [`ArrayReferenceDischarge`] policy walk the same [`ArrayReferenceTransformPath`]. This
/// contract lets the staging consumer construct those operations in a closed operation family, so core array IR and
/// backend-owned supersets share one traversal without matching operation names.
///
/// Per-access transform metadata is exposed through [`ReferenceAccessOperation`]. These constructors stage
/// array-valued operations over discharged immutable state.
pub trait ArrayReferenceTransformOperation: Operation<Type = ArrayIrType> {
    /// Wraps a canonical homogeneous array reshape for reference-transform staging.
    fn from_reference_reshape(operation: ReshapeOperation) -> Self;

    /// Wraps a canonical homogeneous array slice for reference-transform staging.
    fn from_reference_slice(operation: SliceOperation) -> Self;

    /// Wraps a canonical homogeneous array update-slice for reference-transform staging.
    fn from_reference_update_slice(operation: UpdateSliceOperation) -> Self;

    /// Wraps a dynamic slice over discharged reference state.
    fn from_reference_dynamic_slice(operation: DynamicSliceOperation) -> Self;

    /// Wraps a dynamic update over discharged reference state.
    fn from_reference_dynamic_update_slice(operation: DynamicUpdateSliceOperation) -> Self;
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
                message: format!("`{}` cannot replace the reference transforms at input {input_index}", self.name()),
            }),
        }
    }
}

impl<A: Value<Type = ArrayType>> ArrayReferenceTransformOperation for ArrayIrOperation<A> {
    fn from_reference_reshape(operation: ReshapeOperation) -> Self {
        Self::Array(ArrayOperation::Reshape(operation))
    }

    fn from_reference_slice(operation: SliceOperation) -> Self {
        Self::Array(ArrayOperation::Slice(operation))
    }

    fn from_reference_update_slice(operation: UpdateSliceOperation) -> Self {
        Self::Array(ArrayOperation::UpdateSlice(operation))
    }

    fn from_reference_dynamic_slice(operation: DynamicSliceOperation) -> Self {
        Self::Array(ArrayOperation::DynamicSlice(operation))
    }

    fn from_reference_dynamic_update_slice(operation: DynamicUpdateSliceOperation) -> Self {
        Self::Array(ArrayOperation::DynamicUpdateSlice(operation))
    }
}

/// One value carrier through which a reference transform path maps between a shared root and one derived handle.
///
/// Reading the selected view and reconstructing the root with update-slice each exist exactly once, on
/// [`ArrayReferenceTransformPath`], generically over this carrier: the eager carrier operates on concrete values with
/// the array-manipulation capabilities, while reference discharge binds the identical operation sequence through its
/// context. Keeping one traversal guarantees the staged and eager semantics cannot drift apart. Static transforms lower
/// to the carrier's slice and reshape; a symbolic index transform hands the carrier its index through the path's
/// [`Binding`](Self::Binding).
trait TransformReadCarrier {
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

/// A [`TransformReadCarrier`] that can also write a selected hyper-rectangle back into its parent.
trait TransformWriteCarrier: TransformReadCarrier {
    /// Returns `target` with `update` written at `starts`.
    fn update_slice(
        &mut self,
        target: &Self::Value,
        update: &Self::Value,
        starts: Vec<usize>,
    ) -> Result<Self::Value, ProgramError>;

    /// Returns `target` with `update` written at the index that `binding` closes over on `axis`, the inverse of
    /// [`index_symbolic`](TransformReadCarrier::index_symbolic).
    fn update_index_symbolic(
        &mut self,
        target: &Self::Value,
        update: &Self::Value,
        axis: usize,
        binding: &Self::Binding,
    ) -> Result<Self::Value, ProgramError>;
}

/// Stateless eager carrier over one concrete array value family. Eager paths carry only static transforms, so the
/// symbolic-index hooks are unreachable by type.
struct EagerTransformCarrier<A>(PhantomData<A>);

impl<A: Value<Type = ArrayType> + Reshape + Slice> TransformReadCarrier for EagerTransformCarrier<A> {
    type Value = A;
    type Binding = NoReferenceTransformBinding;

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
        binding: &NoReferenceTransformBinding,
    ) -> Result<A, ProgramError> {
        match *binding {}
    }
}

impl<A: Value<Type = ArrayType> + Reshape + Slice + UpdateSlice> TransformWriteCarrier for EagerTransformCarrier<A> {
    fn update_slice(&mut self, target: &A, update: &A, starts: Vec<usize>) -> Result<A, ProgramError> {
        target.update_slice(update, starts.as_slice())
    }

    fn update_index_symbolic(
        &mut self,
        _target: &A,
        _update: &A,
        _axis: usize,
        binding: &NoReferenceTransformBinding,
    ) -> Result<A, ProgramError> {
        match *binding {}
    }
}

/// Eager array-reference handle pairing one shared root allocation with a handle-local transform path.
///
/// Equality and hashing identify the mutable location and structural view, not the handle-local type-identity
/// namespace. Renaming type identities therefore preserves equality with the original handle when its view is
/// unchanged.
pub struct ArrayReference<A: Value<Type = ArrayType>> {
    /// Handle to the shared root allocation.
    root: Reference<A>,

    /// Ordered mapping from the shared root to this handle's referent. Eager handles only ever carry static
    /// transforms, so no symbol is ever bound on this path.
    path: ArrayReferenceTransformPath<NoReferenceTransformBinding>,

    /// Exact handle type derived once from the root type and path, so that repeated [`Typed::r#type`] calls borrow the
    /// cached type instead of re-deriving the complete transform path.
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
        Self { root, path: ArrayReferenceTransformPath::root(), r#type }
    }

    /// Returns this shared reference allocation's process-local identity.
    #[inline]
    pub fn id(&self) -> ReferenceId {
        self.root.id()
    }

    /// Returns the canonical transforms selecting this handle's elements from its shared root allocation.
    pub fn path(&self) -> &ArrayReferenceTransformPath<NoReferenceTransformBinding> {
        &self.path
    }

    /// Returns whether this is an unrenamed root handle accepted at a backend runtime state boundary.
    #[doc(hidden)]
    #[inline]
    pub fn is_runtime_root_handle(&self) -> bool {
        self.path.is_root() && self.root.uses_storage_type_identities()
    }

    /// Locks an unrenamed root for one backend-owned state transaction.
    #[doc(hidden)]
    pub fn lock_root(&self) -> Result<ReadyOrPendingReferenceGuard<'_, A>, ProgramError> {
        if !self.is_runtime_root_handle() {
            return Err(ProgramError::custom(ArrayReferenceViewError::InvalidRuntimeRoot));
        }
        self.root.lock().map_err(ProgramError::custom)
    }

    /// Returns a copy of this handle with `transform` appended to its path, sharing the same root allocation. A
    /// dynamic index is rejected with [`ArrayReferenceViewError::DynamicTransformIndex`]: an eager handle's path
    /// carries only static transforms. [`Self::with_transforms`] resolves dynamic bindings while applying an access's
    /// path.
    pub fn with_transform(&self, transform: ArrayReferenceTransform) -> Result<Self, ProgramError> {
        if transform.binding_count() != 0 {
            return Err(ProgramError::custom(ArrayReferenceViewError::DynamicTransformIndex));
        }
        // The cached handle type already reflects every earlier transform, so composition validates and derives
        // incrementally instead of re-folding the complete chain from the root type. Derivation is purely structural:
        // holder liveness is checked only when the resulting handle accesses state.
        let referent = transform.output_type(self.r#type.referent())?;
        let path = self.path.with_transform(transform);
        Ok(Self { root: self.root.clone(), path, r#type: ReferenceType::new(referent) })
    }

    /// Returns a shared handle after applying an access's ordered transforms and dynamic bindings. Dynamic indices
    /// count negative values from the end of their current axis once, then clamp to its valid range. Each dynamic
    /// transform consumes the next binding. No state is read until the returned handle is accessed. Empty axes and
    /// failures to concretize dynamic bindings are rejected here.
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
                    "reference transform requires {count} bindings but only {} remain",
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
            path.append(ReferenceTransformPath::root().with_transform(transform));
        }
        if !remaining.is_empty() {
            return Err(
                TypeError::invalid(format!("reference transform path has {} extra bindings", remaining.len())).into()
            );
        }
        Ok(Self { root: self.root.clone(), path, r#type: ReferenceType::new(referent) })
    }

    /// Returns an immutable snapshot of this handle's selected elements.
    pub fn read(&self) -> Result<A, ProgramError>
    where
        A: Reshape + Slice,
    {
        self.path.apply(&self.root.read().map_err(ProgramError::custom)?)
    }

    /// Returns an immutable root snapshot without requiring array-manipulation capabilities.
    pub fn read_root(&self) -> Result<A, ProgramError> {
        if !self.path.is_root() {
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
        if self.path.is_root() {
            return self.root.swap(replacement).map_err(ProgramError::custom);
        }
        // Validating inside the update keeps holder-state errors (frozen, poisoned, mid-transaction) ahead of the
        // replacement-type diagnostic, matching the root path.
        self.root.update(|current| {
            self.validate_view_referent_type(&replacement)?;
            self.path.swap(current, &replacement)
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
        if self.path.is_root() {
            return self.root.write(replacement).map_err(ProgramError::custom);
        }
        // Validation remains inside the holder transaction so frozen, poisoned, and leased-state diagnostics retain
        // precedence over replacement-type errors, matching the root write and swap paths.
        self.root.update(|current| {
            self.validate_view_referent_type(&replacement)?;
            self.path
                .write_in(&mut EagerTransformCarrier(PhantomData), current.clone(), replacement)
                .map(|updated| (updated, ()))
        })
    }

    /// Adds `update` into this handle's selected elements.
    pub fn add_update(&self, update: &A) -> Result<(), ProgramError>
    where
        A: Add + Reshape + Slice + UpdateSlice,
    {
        if self.path.is_root() {
            return self.root.update(|current| current.add(update).map(|updated| (updated, ())));
        }
        self.root.update(|current| {
            let mut carrier = EagerTransformCarrier(PhantomData);
            let intermediates = self.path.intermediates_in(&mut carrier, current.clone())?;
            let updated_view = intermediates.last().unwrap().add(update)?;
            self.validate_view_referent_type(&updated_view)?;
            self.path
                .reconstruct_in(&mut carrier, &intermediates[..self.path.bound_transforms().len()], updated_view)
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
        if !self.path.is_root() {
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
        let referent = self.path.output_type(root.r#type().referent())?;
        Ok(Self { root, path: self.path.clone(), r#type: ReferenceType::new(referent) })
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
        Self { root: self.root.clone(), path: self.path.clone(), r#type: self.r#type.clone() }
    }
}

impl<A: Value<Type = ArrayType>> Debug for ArrayReference<A> {
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

impl<A: Value<Type = ArrayType>> Parameter for ArrayReference<A> {}

// The cached type is derived deterministically from the root type and path at construction, so equality and hashing
// over `(root, path)` remain consistent with it.
impl<A: Value<Type = ArrayType>> Typed for ArrayReference<A> {
    type Type = ReferenceType<ArrayType>;

    fn r#type(&self) -> Cow<'_, Self::Type> {
        Cow::Borrowed(&self.r#type)
    }
}

// TODO(eaplatanios): Review this module.

/// Array specialization of [`ReferenceViewAnalysis`], associating each reference access with its ordered
/// [`ArrayReferenceTransformPath`]. Allocation roots and lifetimes come from the shared structural analysis. Dynamic
/// bindings name ordinary values in the access instruction's own region.
pub type ArrayReferenceAnalysis = ReferenceViewAnalysis<ArrayReferenceTransform>;

// TODO(eaplatanios): Review this module.

/// [`ReferenceDischargePolicy`] of the array reference universe.
///
/// An array reference's referent is an ordinary [`ArrayType`]-typed array, and the alias one flowing handle carries is
/// the composed [`ArrayReferenceTransformPath`] mapping its allocation to its own indices, with every symbolic index
/// closed over the context value it selects. Every access therefore reaches its indices through the same transform path
/// traversal the eager handles use, which is what keeps staged and eager reference semantics from drifting apart:
/// reading materializes the allocation-to-handle chain and takes its last snapshot, while a replacement or an
/// accumulation writes the new leaf back through that chain in reverse. Symbolic indices use dynamic slicing and
/// updates with the same negative-index and clamping policy, so reads and mutations always address the same elements.
///
/// The reconstruction context is bounded by [`Context`] rather than [`Domain`](crate::Domain) because the transform
/// path traversal binds canonical slicing, reshape, and update operations into it. Their value-level capabilities are
/// stated over [`ArrayType`]-typed values rather than the composite array IR, so the policy constructs them through the
/// context's operation family.
#[derive(Copy, Clone, Debug)]
pub struct ArrayReferenceDischarge;

impl ReferenceDischargeableType for ArrayIrType {
    type Policy = ArrayReferenceDischarge;
}

impl<C: Context<Type = ArrayIrType>> ReferenceDischargePolicy<C> for ArrayReferenceDischarge
where
    C::Operation: ArrayReferenceTransformOperation,
{
    type Referent = ArrayType;
    type Alias = ArrayReferenceTransformPath<C::Value>;
    type Transform = ArrayReferenceTransform;

    fn compose_transforms(
        _context: &C,
        alias: &Self::Alias,
        transforms: &[ArrayReferenceTransform],
        bindings: &[C::Value],
    ) -> Result<Self::Alias, ProgramError> {
        let mut composed = alias.clone();
        composed.append(ArrayReferenceTransformPath::from_transforms(transforms, bindings)?);
        Ok(composed)
    }

    fn storage_alias(_referent: &ArrayType) -> ArrayReferenceTransformPath<C::Value> {
        ArrayReferenceTransformPath::root()
    }

    fn read(
        context: &C,
        current: &C::Value,
        alias: &ArrayReferenceTransformPath<C::Value>,
    ) -> Result<C::Value, ProgramError> {
        let mut intermediates = alias.intermediates_in(&mut ContextTransformCarrier(context), current.clone())?;

        // The traversal starts with the complete allocation, so the chain is nonempty and its final value is the part
        // selected by this handle.
        Ok(intermediates.pop().unwrap())
    }

    fn write(
        context: &C,
        current: &C::Value,
        replacement: C::Value,
        alias: &ArrayReferenceTransformPath<C::Value>,
    ) -> Result<C::Value, ProgramError> {
        alias.write_in(&mut ContextTransformCarrier(context), current.clone(), replacement)
    }

    fn swap(
        context: &C,
        current: &C::Value,
        replacement: C::Value,
        alias: &ArrayReferenceTransformPath<C::Value>,
    ) -> Result<(C::Value, C::Value), ProgramError> {
        alias.swap_in(&mut ContextTransformCarrier(context), current.clone(), replacement)
    }
}

// Composite array-IR values deliberately expose no value-level addition: the composite family carries array payloads
// through `ArrayIrOperation::Array` and lifts the type-generic `AddOperation<ArrayIrType>` into that member instead,
// which is the same seam generic reverse mode uses to accumulate cotangents. Accumulation therefore binds the lifted
// addition through the context, requiring nothing beyond the conversion the operation family already provides.
impl<C: Context<Type = ArrayIrType>> ReferenceAccumulationPolicy<C> for ArrayReferenceDischarge
where
    C::Operation: ArrayReferenceTransformOperation + From<AddOperation<ArrayIrType>>,
{
    fn accumulate(
        context: &C,
        current: &C::Value,
        update: C::Value,
        alias: &ArrayReferenceTransformPath<C::Value>,
    ) -> Result<C::Value, ProgramError> {
        let mut carrier = ContextTransformCarrier(context);
        let intermediates = alias.intermediates_in(&mut carrier, current.clone())?;
        // Add at the selected leaf, then rebuild each enclosing slice without reading the leaf a second time.
        let selected = intermediates.last().unwrap().clone();
        let accumulated = carrier.bind(C::Operation::from(AddOperation::new()), &[&selected, &update])?;
        alias.reconstruct_in(&mut carrier, &intermediates[..alias.transforms().len()], accumulated)
    }
}

/// Transform carrier that binds the canonical slice, reshape, and update-slice operations of one array reference
/// transform path into a reference discharge context, sharing the single [`ArrayReferenceTransformPath`] traversal with
/// the eager value carrier, which keeps staged and eager reference semantics consistent. Symbolic indices arrive closed
/// over context values and select a size-one dynamic slice; updates restore the removed axis before replacing that
/// slice.
struct ContextTransformCarrier<'c, C>(
    /// Context in which the slice, reshape, and update-slice operations are bound.
    &'c C,
);

impl<C: Context<Type = ArrayIrType>> ContextTransformCarrier<'_, C> {
    /// Binds one single-result operation of the traversal into the context and returns its result.
    ///
    /// # Parameters
    ///
    ///   - `operation`: Context-family operation to bind.
    ///   - `inputs`: Inputs of the application, in operation-defined order.
    fn bind(&self, operation: C::Operation, inputs: &[&C::Value]) -> Result<C::Value, ProgramError> {
        let inputs = inputs.iter().map(|input| (*input).clone()).collect::<Vec<_>>();
        let mut outputs = self.0.bind(operation, Vec::new(), inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

impl<C: Context<Type = ArrayIrType>> TransformReadCarrier for ContextTransformCarrier<'_, C>
where
    C::Operation: ArrayReferenceTransformOperation,
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
        let mut sizes = ArrayReferenceTransform::indexed_shape(axis, &input_type)?.dimensions().to_vec();
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

impl<C: Context<Type = ArrayIrType>> TransformWriteCarrier for ContextTransformCarrier<'_, C>
where
    C::Operation: ArrayReferenceTransformOperation,
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
        let mut dimensions = ArrayReferenceTransform::indexed_shape(axis, &target_type)?.dimensions().to_vec();
        dimensions[axis] = 1;
        let rank = dimensions.len();
        let update = self.reshape(update, Shape::new(dimensions.into_iter().map(Dimension::Static).collect()))?;
        // Restore the indexed axis before writing back. Full-size axes clamp to zero just as in the read path,
        // while the selected axis uses the same runtime index and clamping extent as the original index transform.
        let mut inputs = vec![target, &update];
        inputs.extend(std::iter::repeat_n(binding, rank));
        self.bind(C::Operation::from_reference_dynamic_update_slice(DynamicUpdateSliceOperation::new()), &inputs)
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
    use crate::arrays::operations::ArrayIrOperation;
    use crate::arrays::types::arrays::ArrayType;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{Dimension, DimensionBounds, DimensionVariable};
    use crate::arrays::types::ir::ArrayIrType;
    use crate::arrays::types::memories::Memory;
    use crate::axes::Axis;
    use crate::contexts::EagerContext;
    use crate::differentiation::differentiate_at;
    use crate::operations::{
        ConditionOperation, REFERENCE_NEW_OPERATION_NAME, REFERENCE_READ_OPERATION_NAME, ReferenceAddUpdate,
        ReferenceAddUpdateOperation, ReferenceFreeze, ReferenceFreezeOperation, ReferenceNew, ReferenceNewOperation,
        ReferenceRead, ReferenceReadOperation, ReferenceSwap, ReferenceSwapOperation, ReferenceWrite,
        ReferenceWriteOperation, ScanOperation, WhileOperation,
    };
    use crate::parameters::Placeholder;
    use crate::programs::{
        AtomId, EffectClass, EffectClasses, Operation, Program, ProgramBuilder, ProjectedValue, ReferenceCompletion,
        ReferenceReplacementPreparation, ReferenceType, RegionId, ValueProjection,
    };
    use crate::tracing::{Trace, Tracer, TracingContext};

    use super::*;

    /// Array IR values used to construct reference-view analysis fixtures.
    type TestValue = ArrayIrValue<Array>;

    /// Builder for array operations and reference operations in the same program.
    type TestBuilder = ProgramBuilder<TestValue, ArrayIrOperation<Array>>;

    /// Operation family used by eager and staged array reference fixtures.
    type TestOperation = ArrayIrOperation<Array>;

    /// Context that records immutable array reconstruction.
    type TestContext = TracingContext<TestValue, TestOperation>;

    /// Reference operations over the array IR used by the reference program fixtures.
    type TestNew = ReferenceNewOperation<ArrayType, ArrayIrType>;
    type TestRead = ReferenceReadOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>;
    type TestWrite = ReferenceWriteOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>;
    type TestSwap = ReferenceSwapOperation<ArrayType, ArrayIrType, crate::arrays::references::ArrayReferenceTransform>;
    type TestAddUpdate = ReferenceAddUpdateOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>;
    type TestFreeze = ReferenceFreezeOperation<ArrayType, ArrayIrType>;

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
                ArrayReferenceViewError::DynamicTransformIndex,
                "eager reference handles carry only static transforms; dynamic indices are resolved by each access",
            ),
        ] {
            assert_eq!(error.to_string(), message);
        }
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
    fn test_array_reference_transform_read_type() {
        // Read-only accesses derive the same types as `output_type`; they only skip its write-back proof.
        let input = ArrayType::new_static(DataType::F32, [3, 4]);
        for transform in [
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 1, index: ArrayReferenceTransformIndex::Dynamic },
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 3, 1)] },
        ] {
            assert_eq!(ReferenceTransform::read_type(&transform, &input), transform.output_type(&input));
        }
        assert_eq!(
            ReferenceTransform::read_type(
                &ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(3) },
                &input,
            ),
            Err(TypeError::invalid("reference index 3 on axis 0 is out of bounds for size 3")),
        );
    }

    #[test]
    fn test_array_reference_transform_output_type_rejects_invalid_selections() {
        let matrix_type = ArrayType::new_static(DataType::F32, [3, 4]);
        let vector_type = ArrayType::new_static(DataType::F32, [3]);

        // Static indexing selects one existing index on one existing axis; a symbolic index still names an
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
    fn test_array_reference_transform_output_type_symbolic_index() {
        let matrix_type = ArrayType::new_static(DataType::F32, [3, 4]);
        let symbolic = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic };
        let static_index = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) };
        // Removing a symbolic axis derives the same type even when no static index could select it.
        assert_eq!(symbolic.output_type(&matrix_type), static_index.output_type(&matrix_type));
        assert_eq!(symbolic.output_type(&matrix_type), Ok(ArrayType::new_static(DataType::F32, [4])));
        assert_eq!(
            symbolic.output_type(&ArrayType::new_static(DataType::F32, [0, 4])),
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
            "dynamic_index(axis=0)",
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
            0
        );
        assert_eq!(
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }.binding_count(),
            1
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
    fn test_array_reference_transform_overlap() {
        let root = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [4, 3])));
        let empty: ArrayReferenceTransformPath = ArrayReferenceTransformPath::root();
        let rows_0_1 = empty.with_transform(ArrayReferenceTransform::Slice {
            axes: vec![ArraySliceAxis::new(0, 2, 1), ArraySliceAxis::new(0, 3, 1)],
        });
        let rows_1_2 = empty.with_transform(ArrayReferenceTransform::Slice {
            axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 3, 1)],
        });
        let rows_2_3 = empty.with_transform(ArrayReferenceTransform::Slice {
            axes: vec![ArraySliceAxis::new(2, 2, 1), ArraySliceAxis::new(0, 3, 1)],
        });
        let row_1 = empty
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) });
        let column_0 = empty
            .with_transform(ArrayReferenceTransform::Index { axis: 1, index: ArrayReferenceTransformIndex::Static(0) });

        // Static indices fold to one range per root axis: disjoint ranges on any axis make the paths disjoint,
        // identical ranges on every axis make them the same, and intersecting ranges may overlap. The trait function
        // and the path method agree.
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
        let row_1_columns_1_2 =
            row_1.with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] });
        let row_1_column_1 = row_1
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) });
        let rows_1_columns_1_2_row_0 = empty
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
        let complete = empty.with_transform(ArrayReferenceTransform::Slice {
            axes: vec![ArraySliceAxis::new(0, 4, 1), ArraySliceAxis::new(0, 3, 1)],
        });
        assert_eq!(empty.overlap(&empty, &root), ReferenceViewOverlap::Same);
        assert_eq!(empty.overlap(&complete, &root), ReferenceViewOverlap::Same);
        assert_eq!(empty.overlap(&rows_0_1, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(empty.overlap(&row_1_column_1, &root), ReferenceViewOverlap::MayOverlap);

        // Symbolic indices agree only when their binding, offset, and clamping extent agree. Different
        // offsets can clamp to the same root element, so they cannot establish disjointness.
        let symbolic = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic };
        let first = ValueId::new(RegionId::new(0), AtomId::new(1));
        let second = ValueId::new(RegionId::new(0), AtomId::new(2));
        let other_region = ValueId::new(RegionId::new(1), AtomId::new(1));
        let row_first = empty.with_bound_transform(symbolic.clone(), vec![first]);
        let row_second = empty.with_bound_transform(symbolic.clone(), vec![second]);
        let row_other_region = empty.with_bound_transform(symbolic.clone(), vec![other_region]);
        let shifted_row_first = rows_1_2.with_bound_transform(symbolic.clone(), vec![first]);
        assert_eq!(
            row_first.overlap(&empty.with_bound_transform(symbolic.clone(), vec![first]), &root),
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
        let shortened_row_first = rows_0_1.with_bound_transform(symbolic.clone(), vec![first]);
        assert_eq!(row_first.overlap(&shortened_row_first, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(
            row_first
                .with_transform(ArrayReferenceTransform::Index {
                    axis: 0,
                    index: ArrayReferenceTransformIndex::Static(0)
                })
                .overlap(
                    &row_second.with_transform(ArrayReferenceTransform::Index {
                        axis: 0,
                        index: ArrayReferenceTransformIndex::Static(2)
                    }),
                    &root,
                ),
            ReferenceViewOverlap::Disjoint,
        );

        // A path or root that cannot be folded (an out-of-bounds axis or index, a symbolic transform without its
        // binding, a non-reference root, or a root without a static shape) is conservatively reported as possibly
        // overlapping rather than failing.
        let out_of_bounds = empty
            .with_transform(ArrayReferenceTransform::Index { axis: 2, index: ArrayReferenceTransformIndex::Static(0) });
        let unbound = empty.with_transform(symbolic);
        assert_eq!(out_of_bounds.overlap(&rows_2_3, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(
            empty
                .with_transform(ArrayReferenceTransform::Index {
                    axis: 0,
                    index: ArrayReferenceTransformIndex::Static(4)
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
    fn test_array_reference_transform_overlap_overflow() {
        let root = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])));
        let path: ArrayReferenceTransformPath = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] })
            .with_transform(ArrayReferenceTransform::Index {
                axis: 0,
                index: ArrayReferenceTransformIndex::Static(usize::MAX),
            });
        // Malformed relative indices cannot wrap around to become valid root indices.
        assert_eq!(path.overlap(&ArrayReferenceTransformPath::root(), &root), ReferenceViewOverlap::MayOverlap);
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
                BatchAxis::new(0)
            )),
        );
        assert_eq!(
            index.batch(&packed, BatchAxis::new(1)),
            Ok((
                ArrayReferenceTransform::Index { axis: 2, index: ArrayReferenceTransformIndex::Static(2) },
                BatchAxis::new(1)
            )),
        );
        assert_eq!(
            index.batch(&packed, BatchAxis::new(2)),
            Ok((
                ArrayReferenceTransform::Index { axis: 1, index: ArrayReferenceTransformIndex::Static(2) },
                BatchAxis::new(1)
            )),
        );
        assert_eq!(
            index.batch(&packed, BatchAxis::new(-1)),
            Ok((
                ArrayReferenceTransform::Index { axis: 1, index: ArrayReferenceTransformIndex::Static(2) },
                BatchAxis::new(1)
            )),
        );

        // Batching preserves the symbol that supplies the index.
        let symbolic = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic };
        assert_eq!(
            symbolic.batch(&packed, BatchAxis::new(0)),
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
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 3, 1)] }
                .batch(&ArrayIrType::Reference(ReferenceType::new(dynamic.clone())), BatchAxis::new(0)),
            Err(BatchingError::DynamicBatchAxis { r#type: Box::new(dynamic), axis: Axis::from(0) }),
        );
    }

    #[test]
    fn test_array_reference_transform_batch_rejects_invalid_axes() {
        let packed = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2, 3, 4])));
        assert_eq!(
            ArrayReferenceTransform::Index { axis: 2, index: ArrayReferenceTransformIndex::Static(0) }
                .batch(&packed, BatchAxis::new(0)),
            Err(TypeError::invalid("reference index axis 2 is out of bounds for rank 2").into()),
        );
        // Shifting an unchecked maximum axis used to overflow before it could be rejected.
        assert_eq!(
            ArrayReferenceTransform::Index { axis: usize::MAX, index: ArrayReferenceTransformIndex::Static(0) }
                .batch(&packed, BatchAxis::new(0)),
            Err(TypeError::invalid(format!("reference index axis {} is out of bounds for rank 2", usize::MAX,)).into()),
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
    fn test_array_reference_transform_path_root_slice() {
        let root_type = ArrayType::new_static(DataType::I32, vec![4, 5]);
        let path = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Slice {
                axes: vec![ArraySliceAxis::new(1, 3, 1), ArraySliceAxis::new(2, 3, 1)],
            })
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) });
        assert_eq!(
            path.root_slice(&root_type),
            Some(ArrayReferenceTransform::Slice {
                axes: vec![ArraySliceAxis::new(2, 1, 1), ArraySliceAxis::new(2, 3, 1)]
            }),
        );
        assert_eq!(
            ArrayReferenceTransformPath::root().root_slice(&ArrayType::new_static(DataType::I32, vec![])),
            Some(ArrayReferenceTransform::Slice { axes: vec![] }),
        );
        let symbolic = ArrayReferenceTransformPath::root().with_bound_transform(
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
            vec![ValueId::new(RegionId::new(0), AtomId::new(0))],
        );
        assert_eq!(symbolic.root_slice(&root_type), None);
        let dynamic = ArrayType::new(
            DataType::I32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("length", DimensionBounds::unbounded()))]),
        );
        assert_eq!(ArrayReferenceTransformPath::root().root_slice(&dynamic), None);
        let invalid = path
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(3) });
        assert_eq!(invalid.root_slice(&root_type), None);
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
        let indexed = sliced.with_transform(index);
        assert_eq!(sliced.output_type(&root_type), Ok(ArrayType::new_static(DataType::F32, [2, 3])));
        assert_eq!(indexed.output_type(&root_type), Ok(ArrayType::new_static(DataType::F32, [3])));
    }

    #[test]
    fn test_array_reference_transform_path_intermediates_in() {
        let path: ArrayReferenceTransformPath<NoReferenceTransformBinding> = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] })
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) });
        let root = Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let mut carrier = EagerTransformCarrier::<Array>(PhantomData);
        assert_eq!(
            path.intermediates_in(&mut carrier, root.clone()),
            Ok(vec![root.clone(), Array::vector(vec![2.0_f32, 3.0]).unwrap(), Array::scalar(3.0_f32).unwrap(),]),
        );
        assert_eq!(ArrayReferenceTransformPath::root().intermediates_in(&mut carrier, root.clone()), Ok(vec![root]));
    }

    #[test]
    fn test_array_reference_transform_path_reconstruct_in() {
        let path: ArrayReferenceTransformPath<NoReferenceTransformBinding> = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] })
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) });
        let mut carrier = EagerTransformCarrier::<Array>(PhantomData);
        // Reconstruction consumes strict parents in reverse order; the old selected scalar is unnecessary.
        assert_eq!(
            path.reconstruct_in(
                &mut carrier,
                &[Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap(), Array::vector(vec![2.0_f32, 3.0]).unwrap(),],
                Array::scalar(7.0_f32).unwrap()
            ),
            Ok(Array::vector(vec![1.0_f32, 2.0, 7.0, 4.0]).unwrap()),
        );
        assert_eq!(
            ArrayReferenceTransformPath::root().reconstruct_in(&mut carrier, &[], Array::scalar(7.0_f32).unwrap()),
            Ok(Array::scalar(7.0_f32).unwrap()),
        );
    }

    #[test]
    fn test_array_reference_transform_path_reconstruct_in_rejects_invalid_parent_count() {
        let path: ArrayReferenceTransformPath<NoReferenceTransformBinding> = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) });
        let mut carrier = EagerTransformCarrier::<Array>(PhantomData);
        assert_eq!(
            path.reconstruct_in(&mut carrier, &[], Array::scalar(1.0_f32).unwrap()),
            Err(ProgramError::MalformedProgram(
                "reference transform path reconstruction requires 1 parent snapshots but received 0".to_string(),
            )),
        );
        assert_eq!(
            path.reconstruct_in(
                &mut carrier,
                &[Array::vector(vec![1.0_f32]).unwrap(), Array::scalar(1.0_f32).unwrap()],
                Array::scalar(1.0_f32).unwrap(),
            ),
            Err(ProgramError::MalformedProgram(
                "reference transform path reconstruction requires 1 parent snapshots but received 2".to_string(),
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
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) })
            .unwrap();
        assert_eq!(root, alias);
        assert_ne!(root, separate);
        assert_ne!(root, view);
        let references = HashMap::from([(root.clone(), "root"), (view.clone(), "view")]);
        assert_eq!(references.get(&alias), Some(&"root"));
        assert_eq!(references.get(&view), Some(&"view"));
        assert_eq!(root.to_string(), "ref<f32[2]>");
        assert_eq!(format!("{root:?}"), format!("ArrayReference {{ id: {:?}, path: {:?} }}", root.id(), root.path));
    }

    #[test]
    fn test_array_reference_id() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]).unwrap());
        let view = root
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) })
            .unwrap();
        assert_eq!(root.id(), root.clone().id());
        assert_eq!(root.id(), view.id());
        assert_ne!(root.id(), ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]).unwrap()).id());
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
    fn test_array_reference_is_runtime_root_handle() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]).unwrap());
        let view = root
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) })
            .unwrap();
        assert!(root.is_runtime_root_handle());
        assert!(!view.is_runtime_root_handle());
    }

    #[test]
    fn test_array_reference_lock_root() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]).unwrap());
        let view = root
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) })
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
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 3, 1)] };
        let handle = ArrayReference::new(Array::matrix(3, 4, (1..=12).map(|value| value as f32).collect()).unwrap())
            .with_transform(slice.clone())
            .unwrap();
        assert_eq!(handle.r#type().as_ref(), &ReferenceType::new(ArrayType::new_static(DataType::F32, [2, 3])));
        assert_eq!(handle.read(), Ok(Array::matrix(2, 3, vec![5.0_f32, 6.0, 7.0, 9.0, 10.0, 11.0]).unwrap()));
        assert_eq!(
            handle
                .with_transform(ArrayReferenceTransform::Index {
                    axis: 0,
                    index: ArrayReferenceTransformIndex::Static(2)
                })
                .unwrap_err(),
            TypeError::invalid("reference index 2 on axis 0 is out of bounds for size 2").into(),
        );
    }

    #[test]
    fn test_array_reference_with_transform_rejects_symbolic_indices() {
        // An unresolved dynamic index cannot enter a static eager path. The access must supply its binding through
        // `with_transforms`, which resolves the index before extending the handle.
        let symbolic = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic };
        let path: ArrayReferenceTransformPath<NoReferenceTransformBinding> =
            ArrayReferenceTransformPath::root().with_transform(symbolic.clone());
        assert_eq!(
            path.apply(&Array::matrix(3, 4, (1..=12).map(|value| value as f32).collect()).unwrap()),
            Err(TypeError::invalid(
                "a dynamic index has no static selection; apply its binding at the reference access",
            )
            .into()),
        );
        let root = ArrayReference::new(Array::matrix(3, 4, (1..=12).map(|value| value as f32).collect()).unwrap());
        let error = root.with_transform(symbolic).unwrap_err();
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
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap());
        let guard = root.lock_root().unwrap();
        let ReferenceReplacementPreparation::Prepared(prepared) = guard.prepare_replacement().unwrap() else {
            panic!("new reference unexpectedly has active read leases")
        };
        let transaction = prepared.begin(ReferenceCompletion::ready(Ok(())));

        // A derived handle is pure structural metadata over a live reference, so composing one must never resolve its
        // submitted work. The reference is parked in its `Taken` state, where every value access is unavailable behind
        // this retained guard until replacement commit, and derivation still computes its exact referent type.
        let transform = ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] };
        let derived = root.with_transform(transform).unwrap();
        assert_eq!(derived.r#type().as_ref(), &ReferenceType::new(ArrayType::new_static(DataType::F32, [2])));

        // Poisoning the submitted mutation is terminal for the alias family, but further derivation remains structural
        // composition. The resulting handle reports the reference failure only when it attempts to access state.
        transaction.poison("submission failed");
        let poisoned = ReferenceError::ExecutionPoisoned { reason: "submission failed".to_string() };
        assert_eq!(root.read().unwrap_err().downcast_custom::<ReferenceError>(), Some(&poisoned));
        let composed = derived
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) })
            .unwrap();
        assert_eq!(composed.r#type().as_ref(), &ReferenceType::new(ArrayType::scalar(DataType::F32)));
        assert_eq!(composed.read().unwrap_err().downcast_custom::<ReferenceError>(), Some(&poisoned));

        let frozen = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]).unwrap());
        assert_eq!(frozen.freeze(), Ok(Array::vector(vec![1.0_f32, 2.0]).unwrap()));
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
        selected.swap(Array::scalar(9i32).unwrap()).unwrap();
        assert_eq!(root.read(), Ok(Array::matrix(2, 3, vec![1i32, 2, 3, 4, 5, 9]).unwrap()));
        assert_eq!(root.with_transforms(&[], &[]), Ok(root.clone()));
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
    }

    #[test]
    fn test_array_reference_with_transforms_rejects_invalid_bindings() {
        let root = ArrayReference::new(Array::vector(vec![1i32, 2]).unwrap());
        let transforms = [ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        assert_eq!(
            root.with_transforms(&transforms, &[]),
            Err(TypeError::invalid("reference transform requires 1 bindings but only 0 remain").into()),
        );
        assert_eq!(
            root.with_transforms(&transforms, &[ArrayIrValue::Array(Array::scalar(1f32).unwrap())]),
            Err(TypeError::invalid("reference transform requires a scalar integer index but received `f32[]`").into()),
        );
        let empty = ArrayReference::new(Array::vector(Vec::<i32>::new()).unwrap());
        assert_eq!(
            empty.with_transforms(&transforms, &[ArrayIrValue::Array(Array::scalar(0i32).unwrap())]),
            Err(TypeError::invalid("cannot dynamically index an empty reference axis").into()),
        );
    }

    #[test]
    fn test_array_reference_read() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap());
        let derived = root
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] })
            .unwrap();

        // Reading a derived handle applies its selection rather than exposing the complete allocation.
        assert_eq!(root.read(), Ok(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap()));
        assert_eq!(derived.read(), Ok(Array::vector(vec![2.0_f32, 3.0]).unwrap()));
    }

    #[test]
    fn test_array_reference_read_root() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap());
        let derived = root
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] })
            .unwrap();
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
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) })
            .unwrap();
        assert_eq!(view.swap(Array::scalar(5.0_f32).unwrap()), Ok(Array::scalar(2.0_f32).unwrap()));
        assert_eq!(root.read(), Ok(Array::vector(vec![1.0_f32, 5.0, 3.0]).unwrap()));
    }

    #[test]
    fn test_array_reference_swap_rejects_wrong_referent_type() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap());
        let view = root
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 3, 1)] })
            .unwrap();

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
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) })
            .unwrap();
        assert_eq!(view.write(Array::scalar(5.0_f32).unwrap()), Ok(()));
        assert_eq!(root.read(), Ok(Array::vector(vec![1.0_f32, 5.0, 3.0]).unwrap()));
    }

    #[test]
    fn test_array_reference_write_rejects_wrong_referent_type() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap());
        let view = root
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 3, 1)] })
            .unwrap();

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
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) })
            .unwrap();
        assert_eq!(view.add_update(&Array::scalar(5.0_f32).unwrap()), Ok(()));
        assert_eq!(root.read(), Ok(Array::vector(vec![1.0_f32, 7.0, 3.0]).unwrap()));
    }

    #[test]
    fn test_array_reference_add_update_rejects_wrong_referent_type() {
        let root = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap());
        let view = root
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 3, 1)] })
            .unwrap();

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
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) })
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
    fn test_eager_reference_index_slice_and_composition() {
        let matrix_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let initial =
            ArrayIrValue::Array(Array::from_elements::<f32>(matrix_type, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
        let allocation = initial.reference_new().unwrap();
        let row = ReferenceView::<_, ArrayReferenceTransform, TestValue>::new(allocation.clone())
            .unwrap()
            .index(0, 1)
            .unwrap();
        assert_eq!(row.read(), Ok(ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]).unwrap())));

        let slice = ReferenceView::<_, ArrayReferenceTransform, TestValue>::new(allocation.clone())
            .unwrap()
            .slice(&[ArraySliceAxis::new(0, 2, 1), ArraySliceAxis::new(1, 2, 1)])
            .unwrap();
        assert_eq!(
            slice.read(),
            Ok(ArrayIrValue::Array(
                Array::from_elements::<f32>(
                    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]),),
                    &[2.0, 3.0, 5.0, 6.0]
                )
                .unwrap()
            )),
        );
        let composed = slice.index(0, 1).unwrap();
        assert_eq!(composed.read(), Ok(ArrayIrValue::Array(Array::vector(vec![5.0_f32, 6.0]).unwrap())));
    }

    #[test]
    fn test_eager_reference_indexed_mutation_reconstructs_removed_axis() {
        let matrix_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let initial = ArrayIrValue::Array(
            Array::from_elements::<f32>(matrix_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        );
        let allocation = initial.reference_new().unwrap();
        let row = ReferenceView::<_, ArrayReferenceTransform, TestValue>::new(allocation.clone())
            .unwrap()
            .index(0, 1)
            .unwrap();

        assert_eq!(
            row.swap(&ArrayIrValue::Array(Array::vector(vec![10.0_f32, 20.0, 30.0]).unwrap())),
            Ok(ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]).unwrap())),
        );
        row.add_update(&ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap())).unwrap();
        assert_eq!(
            allocation.read(),
            Ok(ArrayIrValue::Array(
                Array::from_elements::<f32>(matrix_type, &[1.0, 2.0, 3.0, 11.0, 22.0, 33.0]).unwrap()
            )),
        );
    }

    #[test]
    fn test_eager_reference_views_share_overlapping_allocation_state() {
        let allocation =
            ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap()).reference_new().unwrap();
        let left = ReferenceView::<_, ArrayReferenceTransform, TestValue>::new(allocation.clone())
            .unwrap()
            .slice(&[ArraySliceAxis::new(0, 3, 1)])
            .unwrap();
        let right = ReferenceView::<_, ArrayReferenceTransform, TestValue>::new(allocation.clone())
            .unwrap()
            .slice(&[ArraySliceAxis::new(1, 3, 1)])
            .unwrap();

        assert_eq!(
            left.swap(&ArrayIrValue::Array(Array::vector(vec![10.0_f32, 20.0, 30.0]).unwrap())),
            Ok(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap())),
        );
        assert_eq!(right.read(), Ok(ArrayIrValue::Array(Array::vector(vec![20.0_f32, 30.0, 4.0]).unwrap())));
        right.add_update(&ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap())).unwrap();
        assert_eq!(allocation.read(), Ok(ArrayIrValue::Array(Array::vector(vec![10.0_f32, 21.0, 32.0, 7.0]).unwrap())));
    }

    #[test]
    fn test_eager_reference_view_validation_and_freeze_invalidation() {
        let allocation = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap()).reference_new().unwrap();
        assert_eq!(
            ReferenceView::<_, ArrayReferenceTransform, TestValue>::new(allocation.clone()).unwrap().index(1, 0),
            Err(TypeError::invalid("reference index axis 1 is out of bounds for rank 1").into()),
        );
        assert_eq!(
            ReferenceView::<_, ArrayReferenceTransform, TestValue>::new(allocation.clone()).unwrap().index(0, 3),
            Err(TypeError::invalid("reference index 3 on axis 0 is out of bounds for size 3").into()),
        );
        assert_eq!(
            ReferenceView::<_, ArrayReferenceTransform, TestValue>::new(allocation.clone())
                .unwrap()
                .slice(&[ArraySliceAxis::new(2, 2, 1)]),
            Err(TypeError::invalid("reference slice on axis 0 with start 2 and size 2 exceeds input size 3").into()),
        );
        assert_eq!(
            ReferenceView::<_, ArrayReferenceTransform, TestValue>::new(allocation.clone())
                .unwrap()
                .slice(&[ArraySliceAxis::new(0, 2, 2)]),
            Err(TypeError::invalid(
                "reference slice axis 0 stride must be 1 until scatter-backed strided updates are supported",
            )
            .into()),
        );

        let view = ReferenceView::<_, ArrayReferenceTransform, TestValue>::new(allocation.clone())
            .unwrap()
            .slice(&[ArraySliceAxis::new(0, 2, 1)])
            .unwrap();
        let same_view = ReferenceView::<_, ArrayReferenceTransform, TestValue>::new(allocation.clone())
            .unwrap()
            .slice(&[ArraySliceAxis::new(0, 2, 1)])
            .unwrap();
        let different_view = ReferenceView::<_, ArrayReferenceTransform, TestValue>::new(allocation.clone())
            .unwrap()
            .slice(&[ArraySliceAxis::new(1, 2, 1)])
            .unwrap();
        assert_eq!(view, same_view);
        assert_ne!(view, different_view);
        assert_eq!(view.root(), &allocation);
        assert_eq!(allocation.read(), Ok(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap())));

        assert_eq!(allocation.freeze(), Ok(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap())));
        let error = view.read().unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_array_reference_type() {
        let root_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let root = ArrayReference::new(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
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
        assert_eq!(handle.to_string(), "ref<f32[2]>");
        assert_eq!(root.to_string(), "ref<f32[2, 3]>");
    }

    #[test]
    fn test_array_reference_analysis_new() {
        let matrix_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let mut builder = TestBuilder::new();
        let matrix = builder.add_input(ReferenceType::new(matrix_type.clone()).into());
        let row_transforms = vec![
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 1, 1), ArraySliceAxis::new(0, 3, 1)] },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(0) },
        ];
        let column_transforms =
            vec![ArrayReferenceTransform::Index { axis: 1, index: ArrayReferenceTransformIndex::Static(2) }];
        let row = builder
            .add_instruction(TestRead::new().with_transforms(row_transforms.clone()), Vec::new(), vec![matrix], None)
            .unwrap()[0];
        let column = builder
            .add_instruction(TestRead::new().with_transforms(column_transforms.clone()), Vec::new(), vec![matrix], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![row, column], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        let region = program.entry_region_ref();
        let analysis = ArrayReferenceAnalysis::new(region, 0).unwrap();
        let row_path = ArrayReferenceTransformPath::from_transforms(&row_transforms, &[]).unwrap();
        let column_path = ArrayReferenceTransformPath::from_transforms(&column_transforms, &[]).unwrap();
        let row_access = crate::programs::InstructionId::new(region.id(), 0);
        let column_access = crate::programs::InstructionId::new(region.id(), 1);
        assert_eq!(analysis.path(row_access, 0), Some(&row_path));
        assert_eq!(analysis.path(column_access, 0), Some(&column_path));
        assert_eq!(analysis.path(row_access, 1), None);
        assert_eq!(row_path.output_type(&matrix_type), Ok(ArrayType::new_static(DataType::F32, [3])));
        assert_eq!(column_path.output_type(&matrix_type), Ok(ArrayType::new_static(DataType::F32, [2])));
        assert_eq!(
            analysis.overlap(region, (row_access, 0), (column_access, 0)),
            Some(ReferenceViewOverlap::MayOverlap)
        );
    }

    #[test]
    fn test_array_reference_discharge_compose_transforms() {
        let context = EagerContext::<TestValue, TestOperation>::new();
        let index = ArrayIrValue::Array(Array::scalar(1i32).unwrap());
        let alias = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] });
        let transform = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic };
        let composed =
            ArrayReferenceDischarge::compose_transforms(&context, &alias, &[transform.clone()], &[index.clone()]);
        assert_eq!(composed, Ok(alias.with_bound_transform(transform, vec![index])));
        let current = ArrayIrValue::Array(Array::vector(vec![1i32, 2, 3, 4]).unwrap());
        assert_eq!(
            ArrayReferenceDischarge::read(&context, &current, &composed.unwrap()),
            Ok(ArrayIrValue::Array(Array::scalar(3i32).unwrap())),
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
    fn test_array_reference_discharge_read() {
        let context = EagerContext::<TestValue, TestOperation>::new();
        let current = TestValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap());
        let alias = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] });
        assert_eq!(
            ArrayReferenceDischarge::read(&context, &current, &alias),
            Ok(TestValue::Array(Array::vector::<f32>(vec![2.0, 3.0]).unwrap())),
        );
        assert_eq!(
            ArrayReferenceDischarge::read(&context, &current, &ArrayReferenceTransformPath::root()),
            Ok(current)
        );
    }

    #[test]
    fn test_array_reference_discharge_write() {
        let alias = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Slice {
                axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 2, 1)],
            })
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) });
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
        let alias = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] });
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
        // Swapping through an index composed onto a slice must write back through both transforms in reverse order, so
        // the discharged program reconstructs the sliced block from the squeezed row before writing it into the
        // allocation.
        let matrix_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Static(3)]));
        let row_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(matrix_type.clone().into());
        let replacement = builder.add_input(row_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let old = builder
            .add_instruction(
                TestSwap::new().with_transforms(vec![
                    ArrayReferenceTransform::Slice {
                        axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 2, 1)],
                    },
                    ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
                ]),
                Vec::new(),
                vec![reference, replacement],
                None,
            )
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
                Array::from_elements::<f32>(matrix_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                    .unwrap(),
            ),
            TestValue::Array(Array::vector::<f32>(vec![10.0, 20.0]).unwrap()),
        ];
        let expected = vec![
            TestValue::Array(Array::vector::<f32>(vec![7.0, 8.0]).unwrap()),
            TestValue::Array(
                Array::from_elements::<f32>(matrix_type, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 9.0]).unwrap(),
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
        let alias = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] });
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
        let alias: ArrayReferenceTransformPath<Tracer<TestContext>> = ArrayReferenceTransformPath::root()
            .with_transform(ArrayReferenceTransform::Slice {
                axes: vec![ArraySliceAxis::new(1, 2, 1), ArraySliceAxis::new(0, 2, 1)],
            })
            .with_transform(ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) });
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
        let indexed_snapshot = builder
            .add_instruction(
                TestRead::new().with_transforms(vec![ArrayReferenceTransform::Index {
                    axis: 0,
                    index: ArrayReferenceTransformIndex::Static(3),
                }]),
                Vec::new(),
                vec![reference],
                None,
            )
            .unwrap()[0];
        let transforms = vec![
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 3, 1)] },
            ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(0, 2, 1)] },
        ];
        let old = builder
            .add_instruction(
                TestSwap::new().with_transforms(transforms.clone()),
                Vec::new(),
                vec![reference, replacement],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(
                TestAddUpdate::new().with_transforms(transforms),
                Vec::new(),
                vec![reference, update],
                None,
            )
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
        let transforms =
            vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) }];
        let old = builder
            .add_instruction(
                TestSwap::new().with_transforms(transforms.clone()),
                Vec::new(),
                vec![reference, replacement],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(
                TestAddUpdate::new().with_transforms(transforms),
                Vec::new(),
                vec![reference, update],
                None,
            )
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
            TestValue::Array(
                Array::from_elements::<f32>(matrix_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
            ),
            TestValue::Array(Array::vector::<f32>(vec![10.0, 20.0, 30.0]).unwrap()),
            TestValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]).unwrap()),
        ];
        let expected = vec![
            TestValue::Array(Array::vector::<f32>(vec![4.0, 5.0, 6.0]).unwrap()),
            TestValue::Array(Array::from_elements::<f32>(matrix_type, &[1.0, 2.0, 3.0, 11.0, 22.0, 33.0]).unwrap()),
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
            let alias = ArrayReferenceTransformPath::root()
                .with_bound_transform(
                    ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
                    vec![inputs[1].clone()],
                )
                .with_transform(ArrayReferenceTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] });
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

        // Dynamic slicing counts a negative index from the end once, so `-1` selects the last row while `-8` stays
        // negative and clamps to the first row; an oversized index clamps to the last row.
        for (index, selected, written, accumulated) in [
            (-1, vec![5.0, 6.0], vec![1.0, 2.0, 3.0, 4.0, 20.0, 30.0], vec![1.0, 2.0, 3.0, 4.0, 25.0, 36.0]),
            (-8, vec![2.0, 3.0], vec![1.0, 20.0, 30.0, 4.0, 5.0, 6.0], vec![1.0, 22.0, 33.0, 4.0, 5.0, 6.0]),
            (1, vec![5.0, 6.0], vec![1.0, 2.0, 3.0, 4.0, 20.0, 30.0], vec![1.0, 2.0, 3.0, 4.0, 25.0, 36.0]),
            (80, vec![5.0, 6.0], vec![1.0, 2.0, 3.0, 4.0, 20.0, 30.0], vec![1.0, 2.0, 3.0, 4.0, 25.0, 36.0]),
        ] {
            let selected = TestValue::Array(Array::vector::<f32>(selected).unwrap());
            let written = TestValue::Array(Array::from_elements::<f32>(matrix_type.clone(), &written).unwrap());
            assert_eq!(
                staged.clone().interpret(vec![
                    TestValue::Array(
                        Array::from_elements::<f32>(matrix_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()
                    ),
                    TestValue::Array(Array::scalar::<i32>(index).unwrap()),
                    TestValue::Array(Array::vector::<f32>(vec![20.0, 30.0]).unwrap()),
                ]),
                Ok(vec![
                    selected.clone(),
                    written.clone(),
                    selected,
                    written,
                    TestValue::Array(Array::from_elements::<f32>(matrix_type.clone(), &accumulated).unwrap()),
                ]),
            );
        }
    }

    #[test]
    fn test_reference_view_array_selection() {
        let root =
            ArrayIrValue::Reference(ArrayReference::new(Array::matrix(2, 3, vec![1i32, 2, 3, 4, 5, 6]).unwrap()));
        let viewed = ReferenceView::<_, ArrayReferenceTransform, TestValue>::new(root).unwrap();
        let selected = viewed
            .index(0, 1)
            .unwrap()
            .slice(&[ArraySliceAxis::new(1, 2, 1)])
            .unwrap()
            .dynamic_index(0, &ArrayIrValue::Array(Array::scalar(-1i32).unwrap()))
            .unwrap();
        assert_eq!(selected.read(), Ok(ArrayIrValue::Array(Array::scalar(6i32).unwrap())));
        let empty_root = ArrayIrValue::Reference(ArrayReference::new(Array::vector(Vec::<i32>::new()).unwrap()));
        let empty = ReferenceView::<_, ArrayReferenceTransform, TestValue>::new(empty_root)
            .unwrap()
            .dynamic_index(0, &ArrayIrValue::Array(Array::scalar(0i32).unwrap()))
            .unwrap();
        assert_eq!(empty.read(), Err(TypeError::invalid("cannot dynamically index an empty reference axis").into()));
    }

    #[test]
    fn test_reference_view_projected_tracer() {
        type TestTracer = Tracer<TestContext>;
        let (output_type, program) = TestContext::trace(
            |(input, index): (TestTracer, TestTracer)| {
                let input = <TestTracer as ValueProjection<ArrayType>>::into_projected(input)?;
                let reference = input.reference_new()?;
                let viewed = ReferenceView::<_, ArrayReferenceTransform, TestTracer>::new(reference)?
                    .slice(&[ArraySliceAxis::new(1, 2, 1)])?
                    .dynamic_index(0, &index)?;
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
            TestValue::Array(Array::vector(vec![2f32, 3., 5.]).unwrap()),
            TestValue::Array(Array::scalar(-1i32).unwrap()),
        );
        let expected = TestValue::Array(Array::scalar(10f32).unwrap());
        assert_eq!(program.clone().interpret(inputs.clone()), Ok(expected.clone()));
        let discharged = program.into_flat_program().discharge_references(0).unwrap();
        assert_eq!(discharged.program().interpret(vec![inputs.0, inputs.1]), Ok(vec![expected]));
    }

    #[test]
    fn test_repeated_folded_transform_metadata_cost() {
        for accesses in [1, 8, 64] {
            let mut builder = TestBuilder::new();
            let root = builder.add_input(ReferenceType::new(ArrayType::new_static(DataType::F32, [8])).into());
            let index = builder.add_input(ArrayType::scalar(DataType::I32).into());
            let mut outputs = Vec::new();
            for _ in 0..accesses {
                outputs.push(
                    builder
                        .add_instruction(
                            TestRead::new().with_transforms(vec![ArrayReferenceTransform::Index {
                                axis: 0,
                                index: ArrayReferenceTransformIndex::Dynamic,
                            }]),
                            Vec::new(),
                            vec![root, index],
                            None,
                        )
                        .unwrap()[0],
                );
            }
            let program = builder
                .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; accesses])
                .unwrap();
            assert_eq!(program.instructions().len(), accesses);
            assert_eq!(program.atoms().len(), 2 + accesses);
            assert_eq!(
                program.instructions().iter().map(|instruction| instruction.inputs().len()).sum::<usize>(),
                2 * accesses
            );
            assert_eq!(
                program
                    .instructions()
                    .iter()
                    .map(|instruction| instruction
                        .operation()
                        .reference_access_descriptor(0)
                        .unwrap()
                        .transforms()
                        .len())
                    .sum::<usize>(),
                accesses
            );
        }
    }

    #[test]
    fn test_array_ir_operation_reference_access_descriptor() {
        let transform = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic };
        let operation = TestOperation::ReferenceRead(TestRead::new().with_transforms(vec![transform.clone()]));
        assert_eq!(operation.base_input_count(), 1);
        let descriptor = operation.reference_access_descriptor(0).unwrap();
        assert_eq!(descriptor.transforms(), &[transform]);
        assert_eq!(descriptor.bindings(), 1..2);
        assert_eq!(operation.reference_access_descriptor(1), None);
        let replaced = operation.with_reference_access_transforms(0, Vec::new()).unwrap();
        assert_eq!(replaced.reference_access_descriptor(0).unwrap().bindings(), 1..1);
        assert!(matches!(
            operation.with_reference_access_transforms(1, Vec::new()),
            Err(ProgramError::InvalidArgument { message })
                if message == "`reference_read` has no reference access at input 1",
        ));
    }

    #[test]
    fn test_projected_reference_view_chains_preserve_projected_results() {
        type TestContext = TracingContext<TestValue, TestOperation>;
        type TestTracer = Tracer<TestContext>;

        let (output_type, program) = TestContext::trace(
            |input: TestTracer| {
                let input = <TestTracer as ValueProjection<ArrayType>>::into_projected(input)?;
                let reference = input.reference_new()?;
                let sliced = ReferenceView::<_, ArrayReferenceTransform, TestTracer>::new(reference)?
                    .slice(&[ArraySliceAxis::new(0, 2, 1)])?;
                let _: &ReferenceView<
                    ProjectedValue<ReferenceType<ArrayType>, TestTracer>,
                    ArrayReferenceTransform,
                    TestTracer,
                > = &sliced;
                let indexed = sliced.index(0, 1)?;
                let _: &ReferenceView<
                    ProjectedValue<ReferenceType<ArrayType>, TestTracer>,
                    ArrayReferenceTransform,
                    TestTracer,
                > = &indexed;
                let value = indexed.read()?;
                let _: &ProjectedValue<ArrayType, TestTracer> = &value;
                Ok(value.into_value())
            },
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2])),
        )
        .unwrap();

        assert_eq!(output_type, ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        assert_eq!(
            program.instructions().iter().map(|instruction| instruction.operation().name()).collect::<Vec<_>>(),
            vec![REFERENCE_NEW_OPERATION_NAME, REFERENCE_READ_OPERATION_NAME,],
        );
        let input = TestValue::Array(Array::vector(vec![3f32, 7.]).unwrap());
        let expected = TestValue::Array(Array::scalar(7f32).unwrap());
        assert_eq!(program.clone().interpret(input.clone()), Ok(expected.clone()));
        let discharged = program.into_flat_program().discharge_references(0).unwrap();
        assert_eq!(discharged.program().interpret(vec![input]), Ok(vec![expected]));
    }

    #[test]
    fn test_traced_reference_misuse_is_rejected_where_it_is_staged() {
        type TestContext = TracingContext<TestValue, TestOperation>;
        type TestTracer = Tracer<TestContext>;

        let array_type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2, 2]));
        let consumed = "`reference_read` reads a reference whose alias family `reference_freeze` already consumed";

        // Every clone of one tracer names the same staged atom, so a handle cloned before the freeze is invalidated
        // with the rest of the alias family and its next access is reported against the operation that performs it,
        // not against the freeze and not at discharge.
        let error = TestContext::trace(
            |input: TestTracer| {
                let reference = input.reference_new()?;
                let alias = reference.clone();
                reference.freeze()?;
                alias.read()
            },
            array_type.clone(),
        )
        .unwrap_err();
        assert_eq!(error, ProgramError::MalformedProgram(consumed.to_string()));

        // Consumption invalidates a view exactly as it invalidates the allocation, because the view is an alias
        // edge onto the same family rather than an independent resource.
        let error = TestContext::trace(
            |input: TestTracer| {
                let reference = input.reference_new()?;
                let row =
                    ReferenceView::<_, ArrayReferenceTransform, TestTracer>::new(reference.clone())?.index(0, 0)?;
                reference.freeze()?;
                row.read()
            },
            array_type.clone(),
        )
        .unwrap_err();
        assert_eq!(error, ProgramError::MalformedProgram(consumed.to_string()));

        // Independent allocations stay independent, and a whole-family consumption of one says nothing about the other.
        let (_, program) = TestContext::trace(
            |inputs: Vec<TestTracer>| {
                let first = inputs[0].reference_new()?;
                let second = inputs[1].reference_new()?;
                let frozen = first.freeze()?;
                Ok(vec![frozen, second.read()?])
            },
            vec![array_type.clone(), array_type],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 2], %1:f32[2, 2] .
                let %2:ref<f32[2, 2]> = reference_new %0
                    %3:ref<f32[2, 2]> = reference_new %1
                    %4:f32[2, 2] = reference_freeze %2
                    %5:f32[2, 2] = reference_read %3
                in (%4, %5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_traced_structured_reference_carry_joins_its_operand_alias_family() {
        type TestContext = TracingContext<TestValue, TestOperation>;
        type TestTracer = Tracer<TestContext>;

        // A `while` declares nothing in its reference semantics; that its carry output denotes the same reference as
        // its carry input is stated through `reference_output_identity_input` instead. The trace-time liveness state
        // honors that hook, so the loop's own result belongs to its operand's alias family and an access through it
        // after the allocation has been frozen is still reported at the access that performs it.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ArrayIrType::Reference(ReferenceType::new(scalar_type.clone()));
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        condition_builder.add_input(reference_type.clone());
        let predicate = condition_builder.add_constant(TestValue::Array(
            Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[false]).unwrap(),
        ));
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let carry = body_builder.add_input(reference_type);
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![carry], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let error = TestContext::trace(
            |input: TestTracer| {
                let context = input.context().clone();
                let reference = input.reference_new()?;
                let carried = context
                    .bind(
                        WhileOperation::new(),
                        vec![condition.clone(), body.clone()],
                        std::slice::from_ref(&reference),
                    )?
                    .remove(0);
                reference.freeze()?;
                carried.read()
            },
            ArrayIrType::Array(scalar_type),
        )
        .unwrap_err();
        assert_eq!(
            error,
            ProgramError::MalformedProgram(
                "`reference_read` reads a reference whose alias family `reference_freeze` already consumed".to_string(),
            ),
        );
    }

    #[test]
    fn test_array_ir_reference_program_matches_eager_execution() {
        type TestContext = EagerContext<TestValue, TestOperation>;

        let inputs = (
            TestValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()),
            TestValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap()),
            TestValue::Array(Array::vector(vec![5.0_f32, 6.0]).unwrap()),
            TestValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()),
        );
        let (eager_outputs, program) = TestContext::new()
            .interpret_and_trace(
                |(initial, replacement, written, update)| {
                    let reference = initial.reference_new()?;
                    let snapshot = reference.read()?;
                    let old = reference.swap(&replacement)?;
                    reference.write(&written)?;
                    reference.add_update(&update)?;
                    Ok((snapshot, old, reference.freeze()?))
                },
                inputs.clone(),
            )
            .unwrap();
        let expected = (
            TestValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()),
            TestValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()),
            TestValue::Array(Array::vector(vec![6.0_f32, 8.0]).unwrap()),
        );
        assert_eq!(eager_outputs, expected);
        assert_eq!(program.interpret(inputs), Ok(expected));
        assert_eq!(program.effects().classes(), EffectClasses::single(EffectClass::OrderedState));
    }

    #[test]
    fn test_array_ir_reference_jvp_read_modify_write() {
        // The tangent of a read-modify-write is the tangent reference's contents plus the update's tangent, and both
        // the primal and the tangent references observe their respective stores.
        let reference = TestValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()).reference_new().unwrap();
        let tangent_reference = TestValue::Array(Array::vector(vec![0.5_f32, 0.25]).unwrap()).reference_new().unwrap();
        let (primal, tangent) =
            differentiate_at((reference.clone(), TestValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap())))
                .jvp::<_, TestValue, _, _>(
                    (tangent_reference.clone(), TestValue::Array(Array::vector(vec![5.0_f32, 6.0]).unwrap())),
                    |(reference, value)| {
                        reference.add_update(&value)?;
                        reference.read()
                    },
                )
                .unwrap();
        assert_eq!(primal, TestValue::Array(Array::vector(vec![4.0_f32, 6.0]).unwrap()));
        assert_eq!(tangent, TestValue::Array(Array::vector(vec![5.5_f32, 6.25]).unwrap()));
        assert_eq!(reference.read(), Ok(TestValue::Array(Array::vector(vec![4.0_f32, 6.0]).unwrap())));
        assert_eq!(tangent_reference.read(), Ok(TestValue::Array(Array::vector(vec![5.5_f32, 6.25]).unwrap())));
    }

    #[test]
    fn test_array_ir_reference_program_jvp_matches_discharged_program() {
        // A program that allocates, writes, reads, accumulates into, and freezes a local reference has the same fused
        // JVP boundary and values as its discharged reference-free equivalent.
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(array_type.clone().into());
        let replacement = builder.add_input(array_type.into());
        let reference = builder.add_instruction(TestNew::new(), Vec::new(), vec![initial], None).unwrap()[0];
        builder.add_instruction(TestWrite::new(), Vec::new(), vec![reference, replacement], None).unwrap();
        let read = builder.add_instruction(TestRead::new(), Vec::new(), vec![reference], None).unwrap()[0];
        builder.add_instruction(TestAddUpdate::new(), Vec::new(), vec![reference, initial], None).unwrap();
        let frozen = builder.add_instruction(TestFreeze::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![read, frozen], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let jvp = program.jvp().unwrap();
        let discharged =
            program.clone().discharge_references(0).unwrap().into_program_without_external_references().unwrap();
        let discharged_jvp = discharged.jvp().unwrap();
        assert_eq!(jvp.input_types(), discharged_jvp.input_types());
        assert_eq!(jvp.output_types(), discharged_jvp.output_types());

        let inputs = vec![
            TestValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()),
            TestValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap()),
            TestValue::Array(Array::vector(vec![5.0_f32, 6.0]).unwrap()),
            TestValue::Array(Array::vector(vec![7.0_f32, 8.0]).unwrap()),
        ];
        let expected = vec![
            TestValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap()),
            TestValue::Array(Array::vector(vec![4.0_f32, 6.0]).unwrap()),
            TestValue::Array(Array::vector(vec![7.0_f32, 8.0]).unwrap()),
            TestValue::Array(Array::vector(vec![12.0_f32, 14.0]).unwrap()),
        ];
        assert_eq!(jvp.interpret(inputs.clone()), Ok(expected.clone()));
        assert_eq!(discharged_jvp.interpret(inputs), Ok(expected));
    }

    #[test]
    fn test_array_ir_reference_program_replay_binds_external_references() {
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let external = builder.add_input(ReferenceType::new(array_type.clone()).into());
        let replacement = builder.add_input(array_type.into());
        builder.add_instruction(TestSwap::new(), Vec::new(), vec![external, replacement], None).unwrap();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![external], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let initial = TestValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap());
        let reference = initial.reference_new().unwrap();
        let outputs = program
            .interpret(vec![reference.clone(), TestValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap())])
            .unwrap();
        assert_eq!(outputs, vec![reference.clone()]);
        assert_eq!(reference.read(), Ok(TestValue::Array(Array::vector(vec![3.0_f32, 4.0]).unwrap())));
    }

    #[test]
    fn test_array_ir_reference_program_discards_nested_local_allocations() {
        let array_type = ArrayType::new_static(DataType::F32, [2]);

        let mut true_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_input = true_builder.add_input(array_type.clone().into());
        let true_reference =
            true_builder.add_instruction(TestNew::new(), Vec::new(), vec![true_input], None).unwrap()[0];
        let true_output =
            true_builder.add_instruction(TestRead::new(), Vec::new(), vec![true_reference], None).unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![true_output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut false_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let false_input = false_builder.add_input(array_type.clone().into());
        let false_branch = false_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![false_input], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(array_type.into());
        let output = builder
            .add_instruction(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let value = TestValue::Array(Array::vector(vec![2.0_f32, 4.0]).unwrap());
        assert_eq!(
            program.interpret(vec![TestValue::Array(Array::scalar(true).unwrap()), value.clone()]),
            Ok(vec![value.clone()]),
        );
        assert_eq!(
            program.interpret(vec![TestValue::Array(Array::scalar(false).unwrap()), value.clone()]),
            Ok(vec![value])
        );
    }

    #[test]
    fn test_array_ir_reference_program_forwards_checked_allocations_into_condition_branches() {
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let reference_type = ReferenceType::new(array_type.clone());
        let build_branch = || {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            let output = builder.add_instruction(TestRead::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let true_branch = build_branch();
        let false_branch = build_branch();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(array_type.into());
        let reference = builder.add_instruction(TestNew::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let output = builder
            .add_instruction(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let context = EagerContext::<TestValue, TestOperation>::new();
        let value = TestValue::Array(Array::vector(vec![2.0_f32, 4.0]).unwrap());
        for predicate in [true, false] {
            assert_eq!(
                program.interpret(vec![TestValue::Array(Array::scalar(predicate).unwrap()), value.clone()]),
                Ok(vec![value.clone()]),
            );
            assert_eq!(
                program.entry_region_ref().interpret_in_context(
                    &context,
                    vec![TestValue::Array(Array::scalar(predicate).unwrap()), value.clone()],
                ),
                Ok(vec![value.clone()]),
            );
        }
    }

    #[test]
    fn test_array_ir_reference_while_recreates_and_discards_local_allocations_per_invocation() {
        type Values = Vec<TestValue>;

        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let boolean_type = ArrayType::scalar(DataType::Boolean);
        let condition = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let state = builder.add_input(array_type.clone().into());
            let predicate = builder.add_input(boolean_type.clone().into());
            let reference = builder.add_instruction(TestNew::new(), Vec::new(), vec![state], None).unwrap()[0];
            builder.add_instruction(TestRead::new(), Vec::new(), vec![reference], None).unwrap();
            builder.build::<Values, Values>(vec![predicate], vec![Placeholder; 2], vec![Placeholder]).unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let state = builder.add_input(array_type.clone().into());
            builder.add_input(boolean_type.clone().into());
            let reference = builder.add_instruction(TestNew::new(), Vec::new(), vec![state], None).unwrap()[0];
            let update = builder.add_constant(TestValue::Array(Array::scalar(1.0_f32).unwrap()));
            builder.add_instruction(TestAddUpdate::new(), Vec::new(), vec![reference, update], None).unwrap();
            let state = builder.add_instruction(TestRead::new(), Vec::new(), vec![reference], None).unwrap()[0];
            let done = builder.add_constant(TestValue::Array(Array::scalar(false).unwrap()));
            builder
                .build::<Values, Values>(vec![state, done], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(body.entry_region_ref());
        let state = builder.add_input(array_type.into());
        let predicate = builder.add_input(boolean_type.into());
        let outputs = builder
            .add_instruction(
                ArrayIrOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                vec![state, predicate],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder.build::<Values, Values>(outputs, vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();
        assert_eq!(
            program.interpret(vec![
                TestValue::Array(Array::vector(vec![1.0_f32, 2.0]).unwrap()),
                TestValue::Array(Array::scalar(true).unwrap()),
            ]),
            Ok(vec![
                TestValue::Array(Array::vector(vec![2.0_f32, 3.0]).unwrap()),
                TestValue::Array(Array::scalar(false).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_array_ir_reference_scan_recreates_and_discards_local_allocations_per_iteration() {
        type Values = Vec<TestValue>;

        let scalar_type = ArrayType::scalar(DataType::F32);
        let stacked_type = ArrayType::new_static(DataType::F32, [3]);
        let body = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carry = builder.add_input(scalar_type.clone().into());
            let item = builder.add_input(scalar_type.clone().into());
            let reference = builder.add_instruction(TestNew::new(), Vec::new(), vec![carry], None).unwrap()[0];
            builder.add_instruction(TestAddUpdate::new(), Vec::new(), vec![reference, item], None).unwrap();
            let next = builder.add_instruction(TestRead::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Values, Values>(vec![next, next], vec![Placeholder; 3], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let body_region = builder.import_region(body.entry_region_ref());
        let carry = builder.add_input(scalar_type.into());
        let items = builder.add_input(stacked_type.into());
        let outputs = builder
            .add_instruction(ScanOperation::new(1, 3), vec![body_region], vec![carry, items], None)
            .unwrap()
            .to_vec();
        let program = builder.build::<Values, Values>(outputs, vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();

        assert_eq!(
            program.interpret(vec![
                TestValue::Array(Array::scalar(1.0_f32).unwrap()),
                TestValue::Array(Array::vector(vec![1.0_f32, 3.0, 4.0]).unwrap()),
            ]),
            Ok(vec![
                TestValue::Array(Array::scalar(9.0_f32).unwrap()),
                TestValue::Array(Array::vector(vec![2.0_f32, 5.0, 9.0]).unwrap()),
            ]),
        );
    }
}
