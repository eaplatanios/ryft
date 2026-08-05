use std::ops::Range;

use crate::programs::ProgramError;
use crate::programs::types::TypeError;
use crate::types::{ArrayType, DataType, Dimension};

// TODO(eaplatanios): Extend this to address arbitrary physical layouts, including negative strided and tiled layouts,
//  so reference arrays physically honor the layout declared by their `ArrayType`.
/// Checked mapping from a static [`ArrayType`]'s logical indices to its storage addresses. Addressing includes both
/// logical element offsets and byte ranges, and so it is broader than indexing alone. The descriptor currently covers
/// the reference backend's interim dense row-major representation and therefore ignores [`ArrayType::layout`]. The
/// array type is the sole source of truth; construction validates its static shape, element count, and byte length
/// without caching duplicate shape or stride state. The pending arbitrary-layout implementation will make explicit
/// layouts determine the reference array's physical storage addresses.
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
        let element_count = r#type.element_count()?.unwrap();
        element_count
            .checked_mul(Self::element_byte_width_for_data_type(r#type.data_type()))
            .ok_or_else(|| {
                TypeError::invalid(format!("array type {} requires more bytes than can be represented", r#type))
            })?;
        Ok(Self { r#type })
    }

    /// Returns the number of bytes used by each logical element.
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
        self.element_count() * self.element_byte_width()
    }

    /// Returns the number of bytes required by the physical storage of the addressed array.
    #[inline]
    pub fn storage_byte_len(&self) -> usize {
        self.logical_byte_len()
    }

    /// Maps the provided logical multi-index to its flat row-major element index.
    pub fn index(&self, index: &[usize]) -> Result<usize, ProgramError> {
        let rank = self.r#type.rank();
        if index.len() != rank {
            return Err(TypeError::invalid(format!(
                "array index rank {} does not match array rank {}",
                index.len(),
                rank,
            ))
            .into());
        }
        let mut flat_index = 0usize;
        let mut element_stride = 1usize;
        for axis in (0..rank).rev() {
            let coordinate = index[axis];
            let dimension = self.dimension(axis);
            if coordinate >= dimension {
                return Err(TypeError::invalid(format!(
                    "array index {coordinate} on axis {axis} is out of bounds for dimension size {dimension}",
                ))
                .into());
            }
            flat_index =
                flat_index
                    .checked_add(coordinate.checked_mul(element_stride).ok_or_else(|| {
                        TypeError::invalid(format!("array index calculation overflowed on axis {axis}"))
                    })?)
                    .ok_or_else(|| TypeError::invalid(format!("array index calculation overflowed on axis {axis}")))?;
            element_stride = element_stride
                .checked_mul(dimension)
                .ok_or_else(|| TypeError::invalid(format!("array index calculation overflowed on axis {axis}")))?;
        }
        Ok(flat_index)
    }

    /// Maps a contiguous flat logical element index range to its corresponding element and byte ranges.
    pub fn range(&self, elements: Range<usize>) -> Result<ArrayIndexRange, ProgramError> {
        let element_count = self.element_count();
        if elements.start > elements.end || elements.end > element_count {
            return Err(TypeError::invalid(format!(
                "dense array element range {}..{} is out of bounds for {} elements",
                elements.start, elements.end, element_count,
            ))
            .into());
        }
        let element_byte_width = self.element_byte_width();
        let bytes = elements.start * element_byte_width..elements.end * element_byte_width;
        debug_assert!(bytes.end <= self.storage_byte_len());
        Ok(ArrayIndexRange { elements, bytes })
    }

    /// Returns an [`Iterator`] over the contiguous storage [`ArrayIndexRange`] covered by a multidimensional slice.
    /// `starts` and `sizes` define the slice along each axis. `strides` optionally specifies the step along each axis.
    /// Passing [`None`] for `strides` will result in using a stride of one for every axis.
    #[inline]
    pub fn ranges<'a>(
        &'a self,
        starts: &'a [usize],
        sizes: &'a [usize],
        strides: Option<&'a [usize]>,
    ) -> Result<ArrayIndexRanges<'a>, ProgramError> {
        ArrayIndexRanges::new(self, starts, sizes, strides)
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

/// Allocation-free [`Iterator`] over contiguous [`ArrayIndexRange`]s in one logical array selection. The iterator
/// borrows its selection metadata and represents its current position as one mixed-radix ordinal. It therefore
/// allocates no coordinate or stride vectors and performs no fallible validation while iterating.
#[derive(Clone, Debug)]
pub struct ArrayIndexRanges<'a> {
    /// Addressing used to map each emitted logical range to physical storage.
    addressing: &'a ArrayAddressing,

    /// Number of selected coordinates along each logical axis.
    sizes: &'a [usize],

    /// Distance between selected coordinates, or [`None`] when every stride is one.
    strides: Option<&'a [usize]>,

    /// First axis included in each emitted contiguous range.
    run_axis: usize,

    /// Number of logical elements in each emitted contiguous range.
    run_length: usize,

    /// Row-major element stride of the innermost prefix axis.
    prefix_element_stride: usize,

    /// Flat row-major index of the selection's first element.
    base_element: usize,

    /// Ordinal of the next outer-prefix coordinate to emit.
    next_prefix: usize,

    /// Total number of outer-prefix coordinates, and therefore emitted ranges.
    prefix_count: usize,
}

impl<'a> ArrayIndexRanges<'a> {
    /// Validates and constructs a logical range iterator.
    fn new(
        addressing: &'a ArrayAddressing,
        starts: &'a [usize],
        sizes: &'a [usize],
        strides: Option<&'a [usize]>,
    ) -> Result<Self, ProgramError> {
        let rank = addressing.r#type.rank();
        let stride_count = strides.map_or(rank, <[usize]>::len);
        if starts.len() != rank || sizes.len() != rank || stride_count != rank {
            return Err(TypeError::invalid(format!(
                "dense array selection for rank {rank} requires {rank} starts, sizes, and strides but got {}, {}, and {}",
                starts.len(),
                sizes.len(),
                stride_count,
            ))
            .into());
        }

        let stride = |axis| strides.map_or(1, |strides| strides[axis]);
        let mut empty = false;
        for axis in 0..rank {
            let axis_stride = stride(axis);
            if axis_stride == 0 {
                return Err(TypeError::invalid(format!(
                    "dense array selection stride must be positive on axis {axis}"
                ))
                .into());
            }
            let dimension = addressing.dimension(axis);
            if sizes[axis] == 0 {
                empty = true;
                if starts[axis] > dimension {
                    return Err(TypeError::invalid(format!(
                        "empty dense array selection starts at {} on axis {axis}, past dimension size {dimension}",
                        starts[axis],
                    ))
                    .into());
                }
                continue;
            }
            let last = (sizes[axis] - 1)
                .checked_mul(axis_stride)
                .and_then(|offset| starts[axis].checked_add(offset))
                .ok_or_else(|| {
                TypeError::invalid(format!("dense array selection index calculation overflowed on axis {axis}"))
            })?;
            if last >= dimension {
                return Err(TypeError::invalid(format!(
                    "dense array selection reaches index {last} on axis {axis}, past dimension size {dimension}",
                ))
                .into());
            }
        }

        if empty {
            return Ok(Self {
                addressing,
                sizes,
                strides,
                run_axis: rank,
                run_length: 0,
                prefix_element_stride: 0,
                base_element: 0,
                next_prefix: 0,
                prefix_count: 0,
            });
        }

        let base_element = addressing.index(starts)?;
        // Grow the inner contiguous run across each selected suffix axis. We may include one partially selected axis,
        // but can continue into its outer neighbor only when the current axis is selected in full.
        let mut run_axis = rank;
        let mut run_length = 1usize;
        for axis in (0..rank).rev() {
            if sizes[axis] > 1 && stride(axis) != 1 {
                break;
            }
            run_axis = axis;
            run_length = run_length.checked_mul(sizes[axis]).unwrap();
            let covers_axis = starts[axis] == 0 && sizes[axis] == addressing.dimension(axis);
            if !covers_axis {
                break;
            }
        }
        let prefix_count = sizes[..run_axis].iter().try_fold(1usize, |count, size| count.checked_mul(*size)).unwrap();
        let prefix_element_stride = (run_axis..rank)
            .try_fold(1usize, |stride, axis| stride.checked_mul(addressing.dimension(axis)))
            .unwrap();
        Ok(Self {
            addressing,
            sizes,
            strides,
            run_axis,
            run_length,
            prefix_element_stride,
            base_element,
            next_prefix: 0,
            prefix_count,
        })
    }

    /// Returns the [`ArrayAddressing`] used to map the emitted logical ranges to physical storage.
    #[inline]
    pub fn addressing(&self) -> &ArrayAddressing {
        self.addressing
    }

    /// Returns the number of selected coordinates along each logical axis.
    #[inline]
    pub fn sizes(&self) -> &[usize] {
        self.sizes
    }

    /// Returns the distance between selected coordinates, or [`None`] when every stride is one.
    #[inline]
    pub fn strides(&self) -> Option<&[usize]> {
        self.strides
    }

    /// Returns the total number of selected logical elements.
    #[inline]
    pub fn element_count(&self) -> usize {
        self.prefix_count * self.run_length
    }
}

impl Iterator for ArrayIndexRanges<'_> {
    type Item = ArrayIndexRange;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_prefix == self.prefix_count {
            return None;
        }
        let mut ordinal = self.next_prefix;
        self.next_prefix += 1;
        let mut element = self.base_element;
        let mut element_stride = self.prefix_element_stride;
        for axis in (0..self.run_axis).rev() {
            let position = ordinal % self.sizes[axis];
            ordinal /= self.sizes[axis];
            let stride = self.strides.map_or(1, |strides| strides[axis]);
            // Construction validated the complete selection and its total element count, so these products and sums
            // are bounded by the represented array's element count.
            element += position * stride * element_stride;
            element_stride *= self.addressing.dimension(axis);
        }
        Some(self.addressing.range(element..element + self.run_length).unwrap())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.prefix_count - self.next_prefix;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ArrayIndexRanges<'_> {}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::programs::ProgramError;
    use crate::programs::types::TypeError;
    use crate::types::{
        ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable, Layout, Shape, StridedLayout, TiledLayout,
    };

    use super::*;

    /// Creates a static [`ArrayType`] with the provided element data type and dimension sizes.
    fn array_type(data_type: DataType, dimensions: &[usize]) -> ArrayType {
        ArrayType::new(data_type, Shape::new(dimensions.iter().map(|size| Dimension::Static(*size)).collect()))
    }

    #[test]
    fn test_array_addressing() {
        let r#type = array_type(DataType::F32, &[2, 3]);
        let addressing = ArrayAddressing::new(r#type.clone()).unwrap();
        assert_eq!(addressing.r#type, r#type);
        assert_eq!(addressing.element_byte_width(), 4);
        assert_eq!(addressing.element_count(), 6);
        assert_eq!(addressing.logical_byte_len(), 24);
        assert_eq!(addressing.storage_byte_len(), 24);
        assert_eq!(addressing.index(&[0, 0]), Ok(0));
        assert_eq!(addressing.index(&[1, 2]), Ok(5));
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
        let empty = ArrayAddressing::new(array_type(DataType::C128, &[0, usize::MAX, usize::MAX])).unwrap();
        assert_eq!(empty.element_count(), 0);
        assert_eq!(empty.logical_byte_len(), 0);
        assert_eq!(empty.storage_byte_len(), 0);

        // Until arbitrary-layout support lands, explicit layouts leave the interim dense addresses unchanged.
        let strided_type = r#type.clone().with_layout(Layout::Strided(StridedLayout::new(vec![-12, 4])));
        let tiled_type = r#type.clone().with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new())));
        for layout_type in [strided_type, tiled_type] {
            let layout_addressing = ArrayAddressing::new(layout_type).unwrap();
            assert_eq!(layout_addressing.index(&[1, 2]), addressing.index(&[1, 2]));
            assert_eq!(layout_addressing.range(1..4), addressing.range(1..4));
        }

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
                if message == "dense array element range 4..3 is out of bounds for 6 elements",
        ));
        assert!(matches!(
            addressing.range(5..7),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "dense array element range 5..7 is out of bounds for 6 elements",
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
        let oversized = array_type(DataType::C128, &[usize::MAX]);
        assert!(matches!(
            ArrayAddressing::new(oversized),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == format!("array type c128[{}] requires more bytes than can be represented", usize::MAX),
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
            let addressing = ArrayAddressing::new(array_type(data_type, &[2])).unwrap();
            assert_eq!(addressing.element_byte_width(), byte_width);
            assert_eq!(addressing.logical_byte_len(), 2 * byte_width);
            assert_eq!(addressing.storage_byte_len(), 2 * byte_width);
        }
    }

    #[test]
    fn test_array_index_ranges() {
        let addressing = ArrayAddressing::new(array_type(DataType::F32, &[3, 4])).unwrap();

        // A complete selection coalesces into one range, while a partial innermost dimension emits one range per row.
        let ranges = addressing.ranges(&[0, 0], &[3, 4], Some(&[1, 1])).unwrap();
        assert!(std::ptr::eq(ranges.addressing(), &addressing));
        assert_eq!(ranges.sizes(), &[3, 4]);
        assert_eq!(ranges.strides(), Some([1, 1].as_slice()));
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
                if message == "dense array selection for rank 2 requires 2 starts, sizes, and strides but got 1, 2, and 2",
        ));
        assert!(matches!(
            addressing.ranges(&[0, 0], &[1, 1], Some(&[1, 0])),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "dense array selection stride must be positive on axis 1",
        ));
        assert!(matches!(
            addressing.ranges(&[0, 3], &[1, 2], Some(&[1, 1])),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "dense array selection reaches index 4 on axis 1, past dimension size 4",
        ));
        let zero_width = ArrayAddressing::new(array_type(DataType::Zero, &[usize::MAX])).unwrap();
        assert!(matches!(
            zero_width.ranges(&[0], &[usize::MAX], Some(&[2])),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "dense array selection index calculation overflowed on axis 0",
        ));
    }
}
