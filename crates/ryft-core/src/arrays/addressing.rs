use std::ops::Range;

use crate::programs::ProgramError;
use crate::programs::types::TypeError;
use crate::types::{ArrayType, DataType, Dimension, Layout, Tile, TileDimension, TiledLayout};

/// Checked mapping from a static [`ArrayType`]'s logical indices to its storage addresses. Addressing includes both
/// logical element offsets and physical byte ranges, and so it is broader than indexing alone. An array without an
/// explicit [`Layout`] uses dense row-major storage. [`Layout::Strided`] supports positive and negative byte strides,
/// deriving the base offset that keeps every addressed byte inside the allocation. [`Layout::Tiled`] follows
/// [XLA's tiled layout semantics](https://openxla.org/xla/tiled_layout), including minor-to-major ordering, nested
/// tiling, dimension combination, and tile padding semantics.
///
/// The [`ArrayType`] is the sole stored source of truth. Construction validates its static shape, layout structure,
/// non-aliasing storage, and checked storage span without caching parallel shape or layout metadata.
#[derive(Clone, Debug)]
pub struct ArrayAddressing {
    /// Static [`ArrayType`] whose storage is addressed by this [`ArrayAddressing`].
    r#type: ArrayType,
}

impl ArrayAddressing {
    /// Creates a new [`ArrayAddressing`] for the provided [`ArrayType`].
    pub fn new(r#type: ArrayType) -> Result<Self, ProgramError> {
        if r#type.shape().dimensions().iter().any(|dimension| matches!(dimension, Dimension::Dynamic(_))) {
            return Err(
                TypeError::invalid(format!("cannot materialize a value of dynamically sized type {}", r#type)).into()
            );
        }

        let addressing = Self { r#type };

        // Validate the logical size.
        addressing.logical_byte_len_checked()?;

        // Validate the layout.
        match addressing.r#type.layout() {
            None => {}
            Some(Layout::Tiled(layout)) => {
                let rank = addressing.r#type.rank();
                if layout.rank() != rank {
                    return Err(TypeError::invalid(format!(
                        "tiled layout rank {} does not match array rank {}",
                        layout.rank(),
                        rank,
                    ))
                    .into());
                }
                let mut seen = vec![false; rank];
                for axis in layout.minor_to_major() {
                    if *axis >= rank || seen[*axis] {
                        return Err(TypeError::invalid(format!(
                            "tiled layout minor-to-major dimensions must be a permutation of 0..{rank}",
                        ))
                        .into());
                    }
                    seen[*axis] = true;
                }
                let mut dimension_count = rank;
                for (tile_index, tile) in layout.tiles().iter().enumerate() {
                    if tile.dimensions().is_empty() || tile.dimensions().len() > dimension_count {
                        return Err(TypeError::invalid(format!(
                            "tile {} has {} dimensions but the tiled shape has {}",
                            tile_index,
                            tile.dimensions().len(),
                            dimension_count,
                        ))
                        .into());
                    }
                    let mut sized_count = 0usize;
                    for (position, dimension) in tile.dimensions().iter().enumerate() {
                        match dimension {
                            TileDimension::Sized(0) => {
                                return Err(TypeError::invalid(format!(
                                    "tile {tile_index} dimension {position} must have positive size",
                                ))
                                .into());
                            }
                            TileDimension::Sized(_) => sized_count += 1,
                            TileDimension::Combined if position + 1 == tile.dimensions().len() => {
                                return Err(TypeError::invalid(format!(
                                    "tile {tile_index} cannot combine its most minor dimension",
                                ))
                                .into());
                            }
                            TileDimension::Combined => {}
                        }
                    }
                    dimension_count = dimension_count - tile.dimensions().len() + 2 * sized_count;
                }
            }
            Some(Layout::Strided(layout)) => {
                let rank = addressing.r#type.rank();
                if layout.rank() != rank {
                    return Err(TypeError::invalid(format!(
                        "strided layout rank {} does not match array rank {}",
                        layout.rank(),
                        rank,
                    ))
                    .into());
                }
                if addressing.element_count() != 0 && addressing.element_byte_width() != 0 {
                    let mut axes = layout
                        .strides()
                        .iter()
                        .enumerate()
                        .filter(|(axis, _)| addressing.dimension(*axis) > 1)
                        .map(|(axis, stride)| (stride.unsigned_abs(), axis))
                        .collect::<Vec<_>>();
                    axes.sort_unstable();
                    let mut occupied_span = addressing.element_byte_width();
                    for (stride, axis) in axes {
                        if stride < occupied_span {
                            return Err(TypeError::invalid(format!(
                                "strided layout stride {} on axis {} is smaller than the {}-byte span occupied by more minor axes and may alias array elements",
                                layout.strides()[axis], axis, occupied_span,
                            ))
                            .into());
                        }
                        occupied_span = (addressing.dimension(axis) - 1)
                            .checked_mul(stride)
                            .and_then(|span| occupied_span.checked_add(span))
                            .ok_or_else(|| {
                                TypeError::invalid(format!(
                                    "physical storage span for array type {} cannot be represented",
                                    addressing.r#type,
                                ))
                            })?;
                    }
                }
            }
        }

        // Validate the physical size.
        addressing.storage_byte_len_checked()?;

        Ok(addressing)
    }

    /// Returns the static [`ArrayType`] whose storage is addressed by this descriptor.
    #[inline]
    pub fn r#type(&self) -> &ArrayType {
        &self.r#type
    }

    /// Returns the number of bytes used by each logical element. Every [`DataType`] occupies a whole number of bytes:
    /// sub-byte types such as [`DataType::I4`] store one element per byte with the unused high bits set to zero,
    /// unlike XLA's packed host representation that stores two 4-bit elements per byte, and so sub-byte buffers must
    /// be repacked wherever they cross a backend buffer boundary.
    #[inline]
    pub fn element_byte_width(&self) -> usize {
        Self::element_byte_width_for_data_type(self.r#type.data_type())
    }

    /// Returns the number of logical elements in the addressed array.
    #[inline]
    pub fn element_count(&self) -> usize {
        self.r#type.element_count().unwrap().unwrap()
    }

    /// Returns the static size of the provided array axis.
    #[inline]
    fn dimension(&self, axis: usize) -> usize {
        match self.r#type.shape().dimensions()[axis] {
            Dimension::Static(dimension) => dimension,
            Dimension::Dynamic(_) => unreachable!(),
        }
    }

    /// Returns the number of bytes occupied by the encoded logical elements, excluding layout holes and padding.
    #[inline]
    pub fn logical_byte_len(&self) -> usize {
        self.logical_byte_len_checked().unwrap()
    }

    /// Returns the number of bytes required by the physical storage of the addressed array.
    #[inline]
    pub fn storage_byte_len(&self) -> usize {
        self.storage_byte_len_checked().unwrap()
    }

    /// Maps the provided logical multi-index to its flat row-major element index.
    pub fn index(&self, index: &[usize]) -> Result<usize, ProgramError> {
        self.validate_index(index)?;
        Ok(self.logical_index_unchecked(|axis| index[axis]))
    }

    /// Maps one logical multi-index to the byte range occupied by its element in physical storage.
    pub fn byte_range(&self, index: &[usize]) -> Result<Range<usize>, ProgramError> {
        self.validate_index(index)?;
        Ok(self.byte_range_unchecked(|axis| index[axis]))
    }

    /// Maps a prevalidated flat logical row-major index to its physical byte range.
    pub fn byte_range_for_flat_index(&self, index: usize) -> Range<usize> {
        let start = self.byte_offset_unchecked(|axis| {
            let inner = (axis + 1..self.r#type.rank()).fold(1usize, |stride, axis| stride * self.dimension(axis));
            (index / inner) % self.dimension(axis)
        });
        start..start + self.element_byte_width()
    }

    /// Maps a contiguous flat logical element index range to its corresponding physical byte range. This operation
    /// returns an error when the logical elements are not stored contiguously in ascending physical address order.
    /// Use [`Self::ranges`] when the [`Layout`] may split a logical selection across storage ranges.
    pub fn range(&self, elements: Range<usize>) -> Result<ArrayIndexRange, ProgramError> {
        let element_count = self.element_count();
        if elements.start > elements.end || elements.end > element_count {
            return Err(TypeError::invalid(format!(
                "array element range {}..{} is out of bounds for {} elements",
                elements.start, elements.end, element_count,
            ))
            .into());
        }
        if elements.is_empty() {
            return Ok(ArrayIndexRange { elements, bytes: 0..0 });
        }
        if self.is_dense_row_major() {
            let element_byte_width = self.element_byte_width();
            let bytes = elements.start * element_byte_width..elements.end * element_byte_width;
            return Ok(ArrayIndexRange { elements, bytes });
        }
        let first_bytes = self.byte_range_for_flat_index(elements.start);
        let byte_start = first_bytes.start;
        let mut byte_end = first_bytes.end;
        for element in elements.start + 1..elements.end {
            let element_bytes = self.byte_range_for_flat_index(element);
            if element_bytes.start != byte_end {
                return Err(TypeError::invalid(format!(
                    "logical array element range {}..{} is not contiguous in physical storage",
                    elements.start, elements.end,
                ))
                .into());
            }
            byte_end = element_bytes.end;
        }
        let bytes = byte_start..byte_end;
        Ok(ArrayIndexRange { elements, bytes })
    }

    /// Returns an [`Iterator`] over the contiguous storage ranges covered by a multidimensional slice. `starts` and
    /// `sizes` define the slice along each axis. `strides` optionally specifies the step along each axis. [`None`] uses
    /// a stride of one for every axis.
    #[inline]
    pub fn ranges<'a>(
        &'a self,
        starts: &'a [usize],
        sizes: &'a [usize],
        strides: Option<&'a [usize]>,
    ) -> Result<ArrayIndexRanges<'a>, ProgramError> {
        ArrayIndexRanges::new(self, starts, sizes, strides)
    }

    /// Returns `true` when the flat logical row-major element order coincides with dense, gap-free physical storage for
    /// this [`ArrayAddressing`] instance. This holds for arrays without an explicit [`Layout`], for [`Layout::Strided`]
    /// layouts whose strides equal the dense row-major byte strides, for [`Layout::Tiled`] layouts with a descending
    /// minor-to-major permutation and no tiles, and trivially for arrays without payload bytes. Callers can use this
    /// function to replace per-element addressing with bulk byte ranges and copies.
    pub fn is_dense_row_major(&self) -> bool {
        if self.element_count() == 0 || self.element_byte_width() == 0 {
            return true;
        }
        match self.r#type.layout() {
            None => true,
            Some(Layout::Strided(layout)) => {
                // Every dimension is positive here, so the accumulated dense stride stays within the validated
                // logical byte length and cannot overflow.
                let mut dense_stride = self.element_byte_width();
                for axis in (0..self.r#type.rank()).rev() {
                    let stride = layout.strides()[axis];
                    if stride < 0 || stride.unsigned_abs() != dense_stride {
                        return false;
                    }
                    dense_stride *= self.dimension(axis);
                }
                true
            }
            Some(Layout::Tiled(layout)) => {
                layout.tiles().is_empty() && layout.minor_to_major().iter().rev().copied().eq(0..self.r#type.rank())
            }
        }
    }

    /// Returns the checked logical payload byte length, rejecting static element counts or byte lengths
    /// that do not fit in [`usize`].
    fn logical_byte_len_checked(&self) -> Result<usize, ProgramError> {
        self.r#type
            .element_count()
            .ok()
            .flatten()
            .and_then(|element_count| element_count.checked_mul(self.element_byte_width()))
            .ok_or_else(|| {
                TypeError::invalid(format!("array type {} requires more bytes than can be represented", self.r#type))
                    .into()
            })
    }

    /// Returns the checked physical storage byte length.
    fn storage_byte_len_checked(&self) -> Result<usize, ProgramError> {
        let element_byte_width = self.element_byte_width();
        if self.element_count() == 0 || element_byte_width == 0 {
            return Ok(0);
        }
        match self.r#type.layout() {
            None => self.logical_byte_len_checked(),
            Some(Layout::Strided(layout)) => {
                let span = layout.strides().iter().enumerate().try_fold(0usize, |span, (axis, stride)| {
                    (self.dimension(axis) - 1)
                        .checked_mul(stride.unsigned_abs())
                        .and_then(|axis_span| span.checked_add(axis_span))
                        .ok_or_else(|| {
                            TypeError::invalid(format!(
                                "physical storage span for array type {} cannot be represented",
                                self.r#type,
                            ))
                        })
                })?;
                span.checked_add(element_byte_width).ok_or_else(|| {
                    TypeError::invalid(format!(
                        "physical storage span for array type {} cannot be represented",
                        self.r#type,
                    ))
                    .into()
                })
            }
            Some(Layout::Tiled(layout)) => {
                let level = layout.tiles().len();
                let dimension_count = self.tiled_dimension_count(layout, level);
                let padded_element_count = (0..dimension_count).try_fold(1usize, |count, position| {
                    count.checked_mul(self.tiled_component(layout, level, position, &|_| 0)?.1)
                });
                padded_element_count.and_then(|count| count.checked_mul(element_byte_width)).ok_or_else(|| {
                    TypeError::invalid(format!(
                        "physical storage span for array type {} cannot be represented",
                        self.r#type,
                    ))
                    .into()
                })
            }
        }
    }

    /// Validates the provided logical multi-index is within bounds for the underlying array.
    fn validate_index(&self, index: &[usize]) -> Result<(), ProgramError> {
        let rank = self.r#type.rank();
        if index.len() != rank {
            return Err(TypeError::invalid(format!(
                "array index rank {} does not match array rank {}",
                index.len(),
                rank,
            ))
            .into());
        }
        for (axis, coordinate) in index.iter().enumerate() {
            let dimension = self.dimension(axis);
            if *coordinate >= dimension {
                return Err(TypeError::invalid(format!(
                    "array index {coordinate} on axis {axis} is out of bounds for dimension size {dimension}",
                ))
                .into());
            }
        }
        Ok(())
    }

    /// Returns a flat row-major index without validating its coordinates. `coordinate_fn` is called with each logical
    /// axis and must return the selected coordinate along that axis.
    fn logical_index_unchecked<CoordinateFn: Fn(usize) -> usize>(&self, coordinate_fn: CoordinateFn) -> usize {
        let mut index = 0usize;
        for axis in 0..self.r#type.rank() {
            index = index * self.dimension(axis) + coordinate_fn(axis);
        }
        index
    }

    /// Returns a physical byte range without validating its coordinates. `coordinate_fn` is called with each logical
    /// axis and must return the selected coordinate along that axis.
    fn byte_range_unchecked<CoordinateFn: Fn(usize) -> usize>(&self, coordinate_fn: CoordinateFn) -> Range<usize> {
        let start = self.byte_offset_unchecked(coordinate_fn);
        start..start + self.element_byte_width()
    }

    /// Returns a physical byte offset without validating its coordinates. `coordinate_fn` is called with each logical
    /// axis and must return the selected coordinate along that axis.
    fn byte_offset_unchecked<CoordinateFn: Fn(usize) -> usize>(&self, coordinate_fn: CoordinateFn) -> usize {
        let element_byte_width = self.element_byte_width();
        if element_byte_width == 0 {
            return 0;
        }
        match self.r#type.layout() {
            None => self.logical_index_unchecked(coordinate_fn) * element_byte_width,
            Some(Layout::Strided(layout)) => {
                let mut offset = layout.strides().iter().enumerate().fold(0usize, |offset, (axis, stride)| {
                    if *stride < 0 { offset + (self.dimension(axis) - 1) * stride.unsigned_abs() } else { offset }
                });
                for (axis, stride) in layout.strides().iter().enumerate() {
                    let delta = coordinate_fn(axis) * stride.unsigned_abs();
                    if *stride < 0 {
                        offset -= delta;
                    } else {
                        offset += delta;
                    }
                }
                offset
            }
            Some(Layout::Tiled(layout)) => {
                let level = layout.tiles().len();
                let dimension_count = self.tiled_dimension_count(layout, level);
                let mut element = 0usize;
                for position in 0..dimension_count {
                    let (component, bound) = self.tiled_component(layout, level, position, &coordinate_fn).unwrap();
                    element = element.checked_mul(bound).and_then(|element| element.checked_add(component)).unwrap();
                }
                element * element_byte_width
            }
        }
    }

    // TODO(eaplatanios): Should this be moved to `TiledLayout`?
    /// Returns the number of physical dimensions after applying the first `level` nested tiles. Every applied tile
    /// removes its input dimensions and adds one tile-count and one within-tile dimension per sized tile dimension.
    fn tiled_dimension_count(&self, layout: &TiledLayout, level: usize) -> usize {
        layout.tiles()[..level].iter().fold(self.r#type.rank(), |count, tile| {
            count - tile.dimensions().len()
                + 2 * tile.dimensions().iter().filter(|dimension| dimension.is_sized()).count()
        })
    }

    // TODO(eaplatanios): Should this be moved to `TiledLayout`?
    /// Evaluates the physical coordinate and dimension bound at `position` after applying the first `level` nested
    /// tiles, with positions ordering physical dimensions from most major to most minor. At level zero, positions map
    /// logical dimensions through the layout's minor-to-major permutation. Each tile level then passes its untiled
    /// prefix dimensions through unchanged and replaces the tiled suffix with the tile-count coordinates of all sized
    /// tile dimensions followed by their within-tile coordinates, where each sized tile dimension first absorbs the
    /// run of [`TileDimension::Combined`] dimensions immediately preceding it. Returns [`None`] when an intermediate
    /// coordinate or bound does not fit in [`usize`]. `coordinate_fn` is called with a logical axis and must return
    /// the selected coordinate along that axis.
    fn tiled_component<CoordinateFn: Fn(usize) -> usize>(
        &self,
        layout: &TiledLayout,
        level: usize,
        position: usize,
        coordinate_fn: &CoordinateFn,
    ) -> Option<(usize, usize)> {
        if level == 0 {
            let axis = layout.minor_to_major()[self.r#type.rank() - 1 - position];
            return Some((coordinate_fn(axis), self.dimension(axis)));
        }
        let tile = &layout.tiles()[level - 1];
        let prior_count = self.tiled_dimension_count(layout, level - 1);
        let prefix_count = prior_count - tile.dimensions().len();
        if position < prefix_count {
            return self.tiled_component(layout, level - 1, position, coordinate_fn);
        }
        let sized_count = tile.dimensions().iter().filter(|dimension| dimension.is_sized()).count();
        let tiled_position = position - prefix_count;
        let group = tiled_position % sized_count;
        let within_tile = tiled_position >= sized_count;
        let (start, end, tile_size) = Self::tile_group(tile, group);
        let mut combined_coordinate = 0usize;
        let mut combined_bound = 1usize;
        for prior_position in prefix_count + start..prefix_count + end {
            let (component, bound) = self.tiled_component(layout, level - 1, prior_position, coordinate_fn)?;
            combined_coordinate = combined_coordinate.checked_mul(bound)?.checked_add(component)?;
            combined_bound = combined_bound.checked_mul(bound)?;
        }
        if within_tile {
            Some((combined_coordinate % tile_size, tile_size))
        } else {
            Some((combined_coordinate / tile_size, combined_bound.div_ceil(tile_size)))
        }
    }

    /// Returns the tiled-dimension positions absorbed by the `group`-th sized dimension of `tile` as a `(start, end,
    /// size)` tuple, where `start..end` spans the combined dimensions preceding the sized dimension together with the
    /// sized dimension itself.
    fn tile_group(tile: &Tile, group: usize) -> (usize, usize, usize) {
        let mut start = 0usize;
        let mut current_group = 0usize;
        for (position, dimension) in tile.dimensions().iter().enumerate() {
            if let TileDimension::Sized(size) = dimension {
                if current_group == group {
                    return (start, position + 1, *size);
                }
                current_group += 1;
                start = position + 1;
            }
        }

        // Layout validation ensures that every tile ends in a sized dimension, while callers compute `group` modulo
        // the number of sized dimensions. The loop must therefore return upon visiting the requested sized dimension.
        unreachable!()
    }

    /// Returns the byte width of one element of `data_type`.
    const fn element_byte_width_for_data_type(data_type: DataType) -> usize {
        match data_type {
            DataType::Token | DataType::Zero => 0,
            DataType::Boolean
            | DataType::I1
            | DataType::I2
            | DataType::I4
            | DataType::I8
            | DataType::U1
            | DataType::U2
            | DataType::U4
            | DataType::U8
            | DataType::F4E2M1FN
            | DataType::F6E2M3FN
            | DataType::F6E3M2FN
            | DataType::F8E3M4
            | DataType::F8E4M3
            | DataType::F8E4M3FN
            | DataType::F8E4M3FNUZ
            | DataType::F8E4M3B11FNUZ
            | DataType::F8E5M2
            | DataType::F8E5M2FNUZ
            | DataType::F8E8M0FNU => 1,
            DataType::I16 | DataType::U16 | DataType::BF16 | DataType::F16 => 2,
            DataType::I32 | DataType::U32 | DataType::F32 => 4,
            DataType::I64 | DataType::U64 | DataType::F64 | DataType::C64 => 8,
            DataType::C128 => 16,
        }
    }
}

/// One contiguous logical element range in an array and its corresponding physical byte range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArrayIndexRange {
    /// Contiguous flat logical element range.
    elements: Range<usize>,

    /// Physical byte range containing `elements`.
    bytes: Range<usize>,
}

impl ArrayIndexRange {
    /// Returns the contiguous flat logical element range.
    #[inline]
    pub fn elements(&self) -> Range<usize> {
        self.elements.clone()
    }

    /// Returns the physical byte range containing [`Self::elements`].
    #[inline]
    pub fn bytes(&self) -> Range<usize> {
        self.bytes.clone()
    }
}

/// Allocation-free [`Iterator`] over the contiguous [`ArrayIndexRange`]s that make up one logical array slice.
/// Elements are visited in logical row-major slice order. Consecutive logical elements are coalesced only when their
/// physical byte ranges are also consecutive in ascending address order, so dense slices use bulk ranges while
/// strided, permuted, reversed, and tiled layouts split exactly where their storage does.
///
/// The iterator borrows its slice metadata and tracks its position as one flat logical slice index. It allocates
/// nothing, and every fallible rank, bounds, stride, and overflow check happens during construction.
#[derive(Clone, Debug)]
pub struct ArrayIndexRanges<'a> {
    /// [`ArrayAddressing`] used to map each emitted logical range to physical storage.
    addressing: &'a ArrayAddressing,

    /// First selected coordinate along each logical axis.
    starts: &'a [usize],

    /// Number of selected coordinates along each logical axis.
    sizes: &'a [usize],

    /// Distance between selected coordinates along each logical axis, or [`None`] when every stride is one.
    strides: Option<&'a [usize]>,

    /// Total number of logical elements selected by this slice.
    element_count: usize,

    /// Flat logical slice indices that have not yet been mapped to physical storage.
    ordinals: Range<usize>,

    /// First non-coalesced element already read while constructing the previous output range.
    pending: Option<ArrayIndexRange>,
}

impl<'a> ArrayIndexRanges<'a> {
    /// Creates a new [`ArrayIndexRanges`] [`Iterator`].
    pub fn new(
        addressing: &'a ArrayAddressing,
        starts: &'a [usize],
        sizes: &'a [usize],
        strides: Option<&'a [usize]>,
    ) -> Result<Self, ProgramError> {
        let rank = addressing.r#type.rank();
        let stride_count = strides.map_or(rank, <[usize]>::len);
        if starts.len() != rank || sizes.len() != rank || stride_count != rank {
            return Err(TypeError::invalid(format!(
                "array selection for rank {} requires {} starts, sizes, and strides but got {}, {}, and {}",
                rank,
                rank,
                starts.len(),
                sizes.len(),
                stride_count,
            ))
            .into());
        }

        let stride = |axis| strides.map_or(1, |strides: &[usize]| strides[axis]);
        let mut empty = false;
        for axis in 0..rank {
            let axis_stride = stride(axis);
            if axis_stride == 0 {
                return Err(
                    TypeError::invalid(format!("array selection stride must be positive on axis {axis}")).into()
                );
            }
            let dimension = addressing.dimension(axis);
            if sizes[axis] == 0 {
                empty = true;
                if starts[axis] > dimension {
                    return Err(TypeError::invalid(format!(
                        "empty array selection starts at {} on axis {}, past dimension size {}",
                        starts[axis], axis, dimension,
                    ))
                    .into());
                }
                continue;
            }
            let last = (sizes[axis] - 1)
                .checked_mul(axis_stride)
                .and_then(|offset| starts[axis].checked_add(offset))
                .ok_or_else(|| {
                TypeError::invalid(format!("array selection index calculation overflowed on axis {axis}"))
            })?;
            if last >= dimension {
                return Err(TypeError::invalid(format!(
                    "array selection reaches index {last} on axis {axis}, past dimension size {dimension}",
                ))
                .into());
            }
        }

        // A nonempty selection cannot contain more coordinates than the array itself, whose checked element count is
        // representable. Avoid multiplying irrelevant huge dimensions after any zero selection size.
        let element_count =
            if empty { 0 } else { sizes.iter().try_fold(1usize, |count, size| count.checked_mul(*size)).unwrap() };

        Ok(Self { addressing, starts, sizes, strides, element_count, ordinals: 0..element_count, pending: None })
    }

    /// Returns the total number of selected logical elements across all emitted ranges.
    #[inline]
    pub fn element_count(&self) -> usize {
        self.element_count
    }

    /// Maps one flat logical slice ordinal to its single-element logical and physical ranges.
    fn element_range(&self, ordinal: usize) -> ArrayIndexRange {
        let coordinate = |axis: usize| {
            let inner = self.sizes[axis + 1..].iter().product::<usize>();
            let position = (ordinal / inner) % self.sizes[axis];
            let stride = self.strides.map_or(1, |strides| strides[axis]);
            self.starts[axis] + position * stride
        };
        let element = self.addressing.logical_index_unchecked(coordinate);
        let bytes = self.addressing.byte_range_unchecked(coordinate);
        ArrayIndexRange { elements: element..element + 1, bytes }
    }
}

impl Iterator for ArrayIndexRanges<'_> {
    type Item = ArrayIndexRange;

    fn next(&mut self) -> Option<Self::Item> {
        let mut range =
            self.pending.take().or_else(|| self.ordinals.next().map(|ordinal| self.element_range(ordinal)))?;
        while let Some(ordinal) = self.ordinals.next() {
            let next = self.element_range(ordinal);
            if next.elements.start == range.elements.end && next.bytes.start == range.bytes.end {
                range.elements.end = next.elements.end;
                range.bytes.end = next.bytes.end;
            } else {
                self.pending = Some(next);
                break;
            }
        }
        Some(range)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.ordinals.len() + usize::from(self.pending.is_some());
        (usize::from(remaining > 0), Some(remaining))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::programs::ProgramError;
    use crate::programs::types::TypeError;
    use crate::types::{
        ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable, Layout, Shape, StridedLayout, Tile,
        TileDimension, TiledLayout,
    };

    use super::*;

    #[test]
    fn test_array_addressing() {
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let addressing = ArrayAddressing::new(r#type.clone()).unwrap();
        assert_eq!(addressing.r#type(), &r#type);
        assert_eq!(addressing.element_byte_width(), 4);
        assert_eq!(addressing.element_count(), 6);
        assert_eq!(addressing.logical_byte_len(), 24);
        assert_eq!(addressing.storage_byte_len(), 24);
        assert_eq!(addressing.index(&[0, 0]), Ok(0));
        assert_eq!(addressing.index(&[1, 2]), Ok(5));
        assert_eq!(addressing.byte_range(&[1, 2]), Ok(20..24));
        let range = addressing.range(1..4).unwrap();
        assert_eq!(range.elements(), 1..4);
        assert_eq!(range.bytes(), 4..16);

        // Scalar and empty shapes have well-defined addressing, including empty shapes whose irrelevant suffix
        // products would overflow an ordinary row-major-stride calculation.
        let scalar = ArrayAddressing::new(ArrayType::scalar(DataType::C128)).unwrap();
        assert_eq!(scalar.element_count(), 1);
        assert_eq!(scalar.logical_byte_len(), 16);
        assert_eq!(scalar.storage_byte_len(), 16);
        assert_eq!(scalar.index(&[]), Ok(0));
        let empty = ArrayAddressing::new(ArrayType::new(
            DataType::C128,
            Shape::new(vec![Dimension::Static(0), Dimension::Static(usize::MAX), Dimension::Static(usize::MAX)]),
        ))
        .unwrap();
        assert_eq!(empty.element_count(), 0);
        assert_eq!(empty.logical_byte_len(), 0);
        assert_eq!(empty.storage_byte_len(), 0);

        // Malformed external indices and unmaterializable types fail before payload access.
        assert!(matches!(
            addressing.index(&[0]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array index rank 1 does not match array rank 2",
        ));
        assert!(matches!(
            addressing.index(&[2, 0]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array index 2 on axis 0 is out of bounds for dimension size 2",
        ));
        assert!(matches!(
            addressing.range(4..3),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array element range 4..3 is out of bounds for 6 elements",
        ));
        assert!(matches!(
            addressing.range(5..7),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array element range 5..7 is out of bounds for 6 elements",
        ));
        let dynamic = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded()))]),
        );
        assert!(matches!(
            ArrayAddressing::new(dynamic),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot materialize a value of dynamically sized type f32[dynamic]",
        ));
        let oversized = ArrayType::new(DataType::C128, Shape::new(vec![Dimension::Static(usize::MAX)]));
        assert!(matches!(
            ArrayAddressing::new(oversized),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == format!("array type c128[{}] requires more bytes than can be represented", usize::MAX),
        ));

        // Element counts that overflow before the byte multiplication are rejected instead of panicking.
        let overflowing_element_count =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(usize::MAX), Dimension::Static(2)]));
        assert!(matches!(
            ArrayAddressing::new(overflowing_element_count.clone()),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == format!(
                    "array type {overflowing_element_count} requires more bytes than can be represented",
                ),
        ));
    }

    #[test]
    fn test_array_strided_addressing() {
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));

        // Positive byte strides preserve an inner contiguous row while leaving a four-byte hole between rows.
        let positive =
            ArrayAddressing::new(r#type.clone().with_layout(Layout::Strided(StridedLayout::new(vec![16, 4])))).unwrap();
        assert_eq!(positive.logical_byte_len(), 24);
        assert_eq!(positive.storage_byte_len(), 28);
        assert_eq!(positive.byte_range(&[0, 0]), Ok(0..4));
        assert_eq!(positive.byte_range(&[0, 2]), Ok(8..12));
        assert_eq!(positive.byte_range(&[1, 0]), Ok(16..20));
        assert_eq!(positive.byte_range(&[1, 2]), Ok(24..28));
        assert_eq!(positive.range(0..3), Ok(ArrayIndexRange { elements: 0..3, bytes: 0..12 }));
        assert!(matches!(
            positive.range(0..6),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "logical array element range 0..6 is not contiguous in physical storage",
        ));

        // Negative byte strides derive a base offset at the opposite end of storage without changing logical order.
        let negative =
            ArrayAddressing::new(r#type.clone().with_layout(Layout::Strided(StridedLayout::new(vec![-16, 4]))))
                .unwrap();
        assert_eq!(negative.storage_byte_len(), 28);
        assert_eq!(negative.byte_range(&[0, 0]), Ok(16..20));
        assert_eq!(negative.byte_range(&[0, 2]), Ok(24..28));
        assert_eq!(negative.byte_range(&[1, 0]), Ok(0..4));
        assert_eq!(negative.byte_range(&[1, 2]), Ok(8..12));

        // Permuted byte strides make the first logical axis physically minor.
        let permuted =
            ArrayAddressing::new(r#type.clone().with_layout(Layout::Strided(StridedLayout::new(vec![4, 8])))).unwrap();
        assert_eq!(permuted.storage_byte_len(), 24);
        assert_eq!(permuted.byte_range(&[0, 1]), Ok(8..12));
        assert_eq!(permuted.byte_range(&[1, 0]), Ok(4..8));

        // Only strides that exactly reproduce dense row-major storage qualify for bulk dense addressing.
        let dense =
            ArrayAddressing::new(r#type.clone().with_layout(Layout::Strided(StridedLayout::new(vec![12, 4])))).unwrap();
        assert!(dense.is_dense_row_major());
        assert_eq!(dense.range(0..6), Ok(ArrayIndexRange { elements: 0..6, bytes: 0..24 }));
        assert!(!positive.is_dense_row_major());
        assert!(!negative.is_dense_row_major());
        assert!(!permuted.is_dense_row_major());

        // Invalid ranks, potentially aliasing strides, and unrepresentable storage spans fail at construction.
        assert!(matches!(
            ArrayAddressing::new(
                r#type.clone().with_layout(Layout::Strided(StridedLayout::new(vec![4]))),
            ),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "strided layout rank 1 does not match array rank 2",
        ));
        assert!(matches!(
            ArrayAddressing::new(
                r#type.clone().with_layout(Layout::Strided(StridedLayout::new(vec![4, 4]))),
            ),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "strided layout stride 4 on axis 1 is smaller than the 8-byte span occupied by more minor axes and may alias array elements",
        ));
        let overflowing = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![isize::MAX])));
        assert!(matches!(
            ArrayAddressing::new(overflowing),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "physical storage span for array type f32[3][layout=strided{9223372036854775807}] cannot be represented",
        ));
    }

    #[test]
    fn test_array_tiled_addressing() {
        let matrix_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Static(5)]));

        // A minor-to-major permutation without tiles is column-major and introduces no padding, while a descending
        // permutation without tiles reproduces dense row-major storage.
        let permuted = ArrayAddressing::new(
            matrix_type.clone().with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new()))),
        )
        .unwrap();
        assert_eq!(permuted.storage_byte_len(), 60);
        assert_eq!(permuted.byte_range(&[0, 1]), Ok(12..16));
        assert_eq!(permuted.byte_range(&[1, 0]), Ok(4..8));
        assert_eq!(permuted.byte_range(&[2, 4]), Ok(56..60));
        assert!(!permuted.is_dense_row_major());
        let row_major = ArrayAddressing::new(
            matrix_type.clone().with_layout(Layout::Tiled(TiledLayout::new(vec![1, 0], Vec::new()))),
        )
        .unwrap();
        assert!(row_major.is_dense_row_major());
        assert_eq!(row_major.range(0..15), Ok(ArrayIndexRange { elements: 0..15, bytes: 0..60 }));

        // A 2-by-2 tile pads the physical shape to 4-by-6 and follows XLA's tile-major, then within-tile order.
        let tiled = ArrayAddressing::new(matrix_type.clone().with_layout(Layout::Tiled(TiledLayout::new(
            vec![1, 0],
            vec![Tile::new(vec![TileDimension::Sized(2), TileDimension::Sized(2)])],
        ))))
        .unwrap();
        assert_eq!(tiled.logical_byte_len(), 60);
        assert_eq!(tiled.storage_byte_len(), 96);
        assert_eq!(tiled.byte_range(&[0, 0]), Ok(0..4));
        assert_eq!(tiled.byte_range(&[1, 0]), Ok(8..12));
        assert_eq!(tiled.byte_range(&[0, 2]), Ok(16..20));
        assert_eq!(tiled.byte_range(&[2, 3]), Ok(68..72));
        assert_eq!(tiled.byte_range(&[2, 4]), Ok(80..84));

        // Repeated tiling may rearrange the within-tile dimensions produced by an earlier tile.
        let nested = ArrayAddressing::new(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(8)])).with_layout(
                Layout::Tiled(TiledLayout::new(
                    vec![1, 0],
                    vec![
                        Tile::new(vec![TileDimension::Sized(2), TileDimension::Sized(4)]),
                        Tile::new(vec![TileDimension::Sized(2), TileDimension::Sized(1)]),
                    ],
                )),
            ),
        )
        .unwrap();
        assert_eq!(nested.storage_byte_len(), 128);
        assert_eq!(nested.byte_range(&[1, 0]), Ok(4..8));
        assert_eq!(nested.byte_range(&[0, 1]), Ok(8..12));

        // Combined tile dimensions flatten adjacent physical dimensions before ordinary padded tiling is applied.
        let combined = ArrayAddressing::new(
            ArrayType::new(
                DataType::U8,
                Shape::new(vec![
                    Dimension::Static(2),
                    Dimension::Static(7),
                    Dimension::Static(8),
                    Dimension::Static(11),
                    Dimension::Static(10),
                ]),
            )
            .with_layout(Layout::Tiled(TiledLayout::new(
                vec![4, 3, 2, 1, 0],
                vec![Tile::new(vec![
                    TileDimension::Combined,
                    TileDimension::Combined,
                    TileDimension::Sized(2),
                    TileDimension::Combined,
                    TileDimension::Sized(3),
                ])],
            ))),
        )
        .unwrap();
        assert_eq!(combined.logical_byte_len(), 12_320);
        assert_eq!(combined.storage_byte_len(), 12_432);
        assert_eq!(combined.byte_range(&[1, 0, 0, 0, 0]), Ok(6216..6217));

        // Invalid permutations, tile dimensions, and padded storage spans fail during descriptor construction.
        assert!(matches!(
            ArrayAddressing::new(
                matrix_type.clone().with_layout(Layout::Tiled(TiledLayout::new(vec![0], Vec::new()))),
            ),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "tiled layout rank 1 does not match array rank 2",
        ));
        assert!(matches!(
            ArrayAddressing::new(
                matrix_type.clone().with_layout(Layout::Tiled(TiledLayout::new(vec![0, 0], Vec::new()))),
            ),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "tiled layout minor-to-major dimensions must be a permutation of 0..2",
        ));
        assert!(matches!(
            ArrayAddressing::new(matrix_type.clone().with_layout(Layout::Tiled(TiledLayout::new(
                vec![1, 0],
                vec![Tile::new(vec![TileDimension::Sized(0)])],
            )))),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "tile 0 dimension 0 must have positive size",
        ));
        assert!(matches!(
            ArrayAddressing::new(matrix_type.clone().with_layout(Layout::Tiled(TiledLayout::new(
                vec![1, 0],
                vec![Tile::new(vec![TileDimension::Combined])],
            )))),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "tile 0 cannot combine its most minor dimension",
        ));
        assert!(matches!(
            ArrayAddressing::new(matrix_type.clone().with_layout(Layout::Tiled(TiledLayout::new(
                vec![1, 0],
                vec![Tile::new(vec![
                    TileDimension::Sized(1),
                    TileDimension::Sized(1),
                    TileDimension::Sized(1),
                ])],
            )))),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "tile 0 has 3 dimensions but the tiled shape has 2",
        ));
        let overflowing = ArrayType::new(DataType::U8, Shape::new(vec![Dimension::Static(usize::MAX)]))
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0], vec![Tile::new(vec![TileDimension::Sized(2)])])));
        assert!(matches!(
            ArrayAddressing::new(overflowing),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == format!(
                    "physical storage span for array type u8[{}][layout=tiled{{0:T(2)}}] cannot be represented",
                    usize::MAX,
                ),
        ));
    }

    #[test]
    fn test_array_addressing_data_type_widths() {
        let cases = [
            (DataType::Token, 0),
            (DataType::Zero, 0),
            (DataType::Boolean, 1),
            (DataType::I1, 1),
            (DataType::I2, 1),
            (DataType::I4, 1),
            (DataType::I8, 1),
            (DataType::I16, 2),
            (DataType::I32, 4),
            (DataType::I64, 8),
            (DataType::U1, 1),
            (DataType::U2, 1),
            (DataType::U4, 1),
            (DataType::U8, 1),
            (DataType::U16, 2),
            (DataType::U32, 4),
            (DataType::U64, 8),
            (DataType::F4E2M1FN, 1),
            (DataType::F6E2M3FN, 1),
            (DataType::F6E3M2FN, 1),
            (DataType::F8E3M4, 1),
            (DataType::F8E4M3, 1),
            (DataType::F8E4M3FN, 1),
            (DataType::F8E4M3FNUZ, 1),
            (DataType::F8E4M3B11FNUZ, 1),
            (DataType::F8E5M2, 1),
            (DataType::F8E5M2FNUZ, 1),
            (DataType::F8E8M0FNU, 1),
            (DataType::BF16, 2),
            (DataType::F16, 2),
            (DataType::F32, 4),
            (DataType::F64, 8),
            (DataType::C64, 8),
            (DataType::C128, 16),
        ];
        for (data_type, byte_width) in cases {
            let addressing =
                ArrayAddressing::new(ArrayType::new(data_type, Shape::new(vec![Dimension::Static(2)]))).unwrap();
            assert_eq!(addressing.element_byte_width(), byte_width);
            assert_eq!(addressing.logical_byte_len(), 2 * byte_width);
            assert_eq!(addressing.storage_byte_len(), 2 * byte_width);
        }
    }

    #[test]
    fn test_array_index_ranges() {
        let addressing = ArrayAddressing::new(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Static(3), Dimension::Static(4)]),
        ))
        .unwrap();

        // A complete selection coalesces into one range, while a partial innermost dimension emits one range per row.
        let ranges = addressing.ranges(&[0, 0], &[3, 4], Some(&[1, 1])).unwrap();
        assert_eq!(ranges.element_count(), 12);
        assert_eq!(ranges.collect::<Vec<_>>(), vec![ArrayIndexRange { elements: 0..12, bytes: 0..48 }],);
        assert_eq!(
            addressing.ranges(&[0, 1], &[3, 2], Some(&[1, 1])).unwrap().collect::<Vec<_>>(),
            vec![
                ArrayIndexRange { elements: 1..3, bytes: 4..12 },
                ArrayIndexRange { elements: 5..7, bytes: 20..28 },
                ArrayIndexRange { elements: 9..11, bytes: 36..44 },
            ],
        );

        // A strided outer dimension retains contiguous rows; a strided inner dimension emits individual elements.
        assert_eq!(
            addressing.ranges(&[0, 0], &[2, 4], Some(&[2, 1])).unwrap().collect::<Vec<_>>(),
            vec![ArrayIndexRange { elements: 0..4, bytes: 0..16 }, ArrayIndexRange { elements: 8..12, bytes: 32..48 },],
        );
        assert_eq!(
            addressing.ranges(&[0, 0], &[2, 2], Some(&[1, 2])).unwrap().collect::<Vec<_>>(),
            vec![
                ArrayIndexRange { elements: 0..1, bytes: 0..4 },
                ArrayIndexRange { elements: 2..3, bytes: 8..12 },
                ArrayIndexRange { elements: 4..5, bytes: 16..20 },
                ArrayIndexRange { elements: 6..7, bytes: 24..28 },
            ],
        );

        // Coalescing follows physical storage and therefore splits permuted and reversed layouts.
        let permuted = ArrayAddressing::new(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
                .with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new()))),
        )
        .unwrap();
        assert_eq!(
            permuted.ranges(&[0, 0], &[2, 3], None).unwrap().collect::<Vec<_>>(),
            vec![
                ArrayIndexRange { elements: 0..1, bytes: 0..4 },
                ArrayIndexRange { elements: 1..2, bytes: 8..12 },
                ArrayIndexRange { elements: 2..3, bytes: 16..20 },
                ArrayIndexRange { elements: 3..4, bytes: 4..8 },
                ArrayIndexRange { elements: 4..5, bytes: 12..16 },
                ArrayIndexRange { elements: 5..6, bytes: 20..24 },
            ],
        );
        let reversed = ArrayAddressing::new(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]))
                .with_layout(Layout::Strided(StridedLayout::new(vec![-4]))),
        )
        .unwrap();
        assert_eq!(
            reversed.ranges(&[0], &[3], None).unwrap().collect::<Vec<_>>(),
            vec![
                ArrayIndexRange { elements: 0..1, bytes: 8..12 },
                ArrayIndexRange { elements: 1..2, bytes: 4..8 },
                ArrayIndexRange { elements: 2..3, bytes: 0..4 },
            ],
        );

        // Rank-zero selections contain one element, while any zero selection size yields no ranges.
        let scalar = ArrayAddressing::new(ArrayType::scalar(DataType::F32)).unwrap();
        assert_eq!(
            scalar.ranges(&[], &[], None).unwrap().collect::<Vec<_>>(),
            vec![ArrayIndexRange { elements: 0..1, bytes: 0..4 }],
        );
        assert_eq!(
            addressing.ranges(&[3, 0], &[0, 4], Some(&[1, 1])).unwrap().collect::<Vec<_>>(),
            Vec::<ArrayIndexRange>::new(),
        );

        // Invalid selection metadata is rejected completely before iteration begins.
        assert!(matches!(
            addressing.ranges(&[0], &[1, 1], Some(&[1, 1])),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array selection for rank 2 requires 2 starts, sizes, and strides but got 1, 2, and 2",
        ));
        assert!(matches!(
            addressing.ranges(&[0, 0], &[1, 1], Some(&[1, 0])),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array selection stride must be positive on axis 1",
        ));
        assert!(matches!(
            addressing.ranges(&[0, 3], &[1, 2], Some(&[1, 1])),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array selection reaches index 4 on axis 1, past dimension size 4",
        ));
        let zero_width =
            ArrayAddressing::new(ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(usize::MAX)])))
                .unwrap();
        assert!(matches!(
            zero_width.ranges(&[0], &[usize::MAX], Some(&[2])),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array selection index calculation overflowed on axis 0",
        ));
    }
}
