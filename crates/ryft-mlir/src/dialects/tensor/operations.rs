use crate::{
    ArrayAttributeRef, Attribute, DenseInteger32ArrayAttributeRef, DenseInteger64ArrayAttributeRef, DetachedOp,
    DetachedRegion, DialectHandle, IntegerAttributeRef, Location, OneResult, Operation, OperationBuilder, RegionRef,
    Size, Type, Value, ValueRef, mlir_op, mlir_op_trait,
};

/// An index entry that is either known statically or provided by an SSA value at runtime.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum StaticOrDynamicIndex<'v, 'c: 'v, 't: 'c> {
    /// Statically known index value.
    Static(i64),

    /// Dynamically provided index value.
    Dynamic(ValueRef<'v, 'c, 't>),
}

impl<'v, 'c: 'v, 't: 'c> StaticOrDynamicIndex<'v, 'c, 't> {
    /// Returns the static value, if this entry is statically known.
    pub fn static_value(&self) -> Option<i64> {
        match self {
            Self::Static(value) => Some(*value),
            Self::Dynamic(_) => None,
        }
    }

    /// Returns the dynamic SSA value, if this entry is dynamically provided.
    pub fn dynamic_value(&self) -> Option<ValueRef<'v, 'c, 't>> {
        match self {
            Self::Static(_) => None,
            Self::Dynamic(value) => Some(*value),
        }
    }
}

/// Operation trait for `tensor.bitcast`.
pub trait BitcastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the tensor being bitcast.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the bitcast tensor.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(Bitcast);
mlir_op_trait!(Bitcast, AlwaysSpeculatable);
mlir_op_trait!(Bitcast, NoMemoryEffect);
mlir_op_trait!(Bitcast, OneResult);
mlir_op_trait!(Bitcast, Pure);
mlir_op_trait!(Bitcast, ZeroRegions);
mlir_op_trait!(Bitcast, ZeroSuccessors);

/// Constructs a new detached/owned [`BitcastOperation`] at the specified [`Location`].
pub fn bitcast<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    result_type: T,
    location: L,
) -> DetachedBitcastOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::tensor());
    OperationBuilder::new("tensor.bitcast", location)
        .add_operand(source)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tensor::bitcast`")
}

/// Operation trait for `tensor.cast`.
pub trait CastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the tensor being cast.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the cast tensor.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(Cast);
mlir_op_trait!(Cast, AlwaysSpeculatable);
mlir_op_trait!(Cast, NoMemoryEffect);
mlir_op_trait!(Cast, OneResult);
mlir_op_trait!(Cast, Pure);
mlir_op_trait!(Cast, ZeroRegions);
mlir_op_trait!(Cast, ZeroSuccessors);

/// Constructs a new detached/owned [`CastOperation`] at the specified [`Location`].
pub fn cast<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    result_type: T,
    location: L,
) -> DetachedCastOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::tensor());
    OperationBuilder::new("tensor.cast", location)
        .add_operand(source)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tensor::cast`")
}

/// Name of the `tensor.concat` dimension attribute.
pub const DIM_ATTRIBUTE: &str = "dim";

/// Operation trait for `tensor.concat`.
pub trait ConcatOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the concatenation dimension.
    fn dimension(&self) -> i64 {
        self.attribute(DIM_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value())
            .unwrap_or_else(|| panic!("invalid '{DIM_ATTRIBUTE}' attribute in `tensor.concat`"))
    }

    /// Returns the input tensors being concatenated.
    fn inputs(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }

    /// Returns the concatenated tensor.
    fn concatenated(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(Concat);
mlir_op_trait!(Concat, AlwaysSpeculatable);
mlir_op_trait!(Concat, NoMemoryEffect);
mlir_op_trait!(Concat, OneResult);
mlir_op_trait!(Concat, Pure);
mlir_op_trait!(Concat, ZeroRegions);
mlir_op_trait!(Concat, ZeroSuccessors);

/// Constructs a new detached/owned [`ConcatOperation`] at the specified [`Location`].
pub fn concat<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    inputs: &[ValueRef<'v, 'c, 't>],
    dimension: i64,
    result_type: T,
    location: L,
) -> DetachedConcatOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::tensor());
    OperationBuilder::new("tensor.concat", location)
        .add_operands(inputs)
        .add_attribute(DIM_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(64), dimension))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tensor::concat`")
}

/// Operation trait for `tensor.dim`.
pub trait DimOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the tensor whose dimension is queried.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the dimension index.
    fn index(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the dimension size.
    fn size(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(Dim);
mlir_op_trait!(Dim, NoMemoryEffect);
mlir_op_trait!(Dim, OneResult);
mlir_op_trait!(Dim, ZeroRegions);
mlir_op_trait!(Dim, ZeroSuccessors);

/// Constructs a new detached/owned [`DimOperation`] at the specified [`Location`].
pub fn dim<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    index: ValueRef<'v, 'c, 't>,
    location: L,
) -> DetachedDimOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::tensor());
    OperationBuilder::new("tensor.dim", location)
        .add_operand(source)
        .add_operand(index)
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tensor::dim`")
}

/// Operation trait for `tensor.empty`.
pub trait EmptyOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the dynamic size operands.
    fn dynamic_sizes(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }

    /// Returns the materialized tensor.
    fn tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(Empty);
mlir_op_trait!(Empty, AlwaysSpeculatable);
mlir_op_trait!(Empty, NoMemoryEffect);
mlir_op_trait!(Empty, OneResult);
mlir_op_trait!(Empty, Pure);
mlir_op_trait!(Empty, ZeroRegions);
mlir_op_trait!(Empty, ZeroSuccessors);

/// Constructs a new detached/owned [`EmptyOperation`] at the specified [`Location`].
pub fn empty<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    dynamic_sizes: &[ValueRef<'v, 'c, 't>],
    result_type: T,
    location: L,
) -> DetachedEmptyOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::tensor());
    OperationBuilder::new("tensor.empty", location)
        .add_operands(dynamic_sizes)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tensor::empty`")
}

/// Operation trait for `tensor.extract`.
pub trait ExtractOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the tensor being read.
    fn tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the index operands.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        (1..self.operand_count()).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns the extracted element.
    fn element(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(Extract);
mlir_op_trait!(Extract, AlwaysSpeculatable);
mlir_op_trait!(Extract, NoMemoryEffect);
mlir_op_trait!(Extract, OneResult);
mlir_op_trait!(Extract, Pure);
mlir_op_trait!(Extract, ZeroRegions);
mlir_op_trait!(Extract, ZeroSuccessors);

/// Constructs a new detached/owned [`ExtractOperation`] at the specified [`Location`].
pub fn extract<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    tensor: ValueRef<'v, 'c, 't>,
    indices: &[ValueRef<'v, 'c, 't>],
    result_type: T,
    location: L,
) -> DetachedExtractOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::tensor());
    OperationBuilder::new("tensor.extract", location)
        .add_operand(tensor)
        .add_operands(indices)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tensor::extract`")
}

/// Name of the operand segment-size attribute used by Tensor operations with variadic operand groups.
pub const OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operandSegmentSizes";

/// Name of the static offset entries attribute.
pub const STATIC_OFFSETS_ATTRIBUTE: &str = "static_offsets";

/// Name of the static size entries attribute.
pub const STATIC_SIZES_ATTRIBUTE: &str = "static_sizes";

/// Name of the static stride entries attribute.
pub const STATIC_STRIDES_ATTRIBUTE: &str = "static_strides";

/// Operation trait for `tensor.extract_slice`.
pub trait ExtractSliceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source tensor.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the mixed static and dynamic offsets.
    fn offsets(&self) -> Vec<StaticOrDynamicIndex<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| {
                panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `tensor.extract_slice`")
            });
        let dynamic_start = 1;
        let dynamic_end = dynamic_start + segment_sizes[1] as usize;
        let dynamic_offsets =
            (dynamic_start..dynamic_end).map(|index| self.operand_value(index).unwrap()).collect::<Vec<_>>();
        let static_offsets = self
            .attribute(STATIC_OFFSETS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| panic!("invalid '{STATIC_OFFSETS_ATTRIBUTE}' attribute in `tensor.extract_slice`"));
        let dynamic_index = unsafe { Size::Dynamic.to_c_api() };
        let mut dynamic_offsets = dynamic_offsets.into_iter();
        static_offsets
            .into_iter()
            .map(|index| {
                (index == dynamic_index)
                    .then(|| dynamic_offsets.next().expect("missing dynamic offset operand"))
                    .map_or_else(|| StaticOrDynamicIndex::Static(index), StaticOrDynamicIndex::Dynamic)
            })
            .collect()
    }

    /// Returns the mixed static and dynamic sizes.
    fn sizes(&self) -> Vec<StaticOrDynamicIndex<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| {
                panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `tensor.extract_slice`")
            });
        let dynamic_start = 1 + segment_sizes[1] as usize;
        let dynamic_end = dynamic_start + segment_sizes[2] as usize;
        let dynamic_sizes =
            (dynamic_start..dynamic_end).map(|index| self.operand_value(index).unwrap()).collect::<Vec<_>>();
        let static_sizes = self
            .attribute(STATIC_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| panic!("invalid '{STATIC_SIZES_ATTRIBUTE}' attribute in `tensor.extract_slice`"));
        let dynamic_index = unsafe { Size::Dynamic.to_c_api() };
        let mut dynamic_sizes = dynamic_sizes.into_iter();
        static_sizes
            .into_iter()
            .map(|index| {
                (index == dynamic_index)
                    .then(|| dynamic_sizes.next().expect("missing dynamic size operand"))
                    .map_or_else(|| StaticOrDynamicIndex::Static(index), StaticOrDynamicIndex::Dynamic)
            })
            .collect()
    }

    /// Returns the mixed static and dynamic strides.
    fn strides(&self) -> Vec<StaticOrDynamicIndex<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| {
                panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `tensor.extract_slice`")
            });
        let dynamic_start = 1 + segment_sizes[1] as usize + segment_sizes[2] as usize;
        let dynamic_end = dynamic_start + segment_sizes[3] as usize;
        let dynamic_strides =
            (dynamic_start..dynamic_end).map(|index| self.operand_value(index).unwrap()).collect::<Vec<_>>();
        let static_strides = self
            .attribute(STATIC_STRIDES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| panic!("invalid '{STATIC_STRIDES_ATTRIBUTE}' attribute in `tensor.extract_slice`"));
        let dynamic_index = unsafe { Size::Dynamic.to_c_api() };
        let mut dynamic_strides = dynamic_strides.into_iter();
        static_strides
            .into_iter()
            .map(|index| {
                (index == dynamic_index)
                    .then(|| dynamic_strides.next().expect("missing dynamic stride operand"))
                    .map_or_else(|| StaticOrDynamicIndex::Static(index), StaticOrDynamicIndex::Dynamic)
            })
            .collect()
    }

    /// Returns the extracted slice.
    fn slice(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(ExtractSlice);
mlir_op_trait!(ExtractSlice, AlwaysSpeculatable);
mlir_op_trait!(ExtractSlice, NoMemoryEffect);
mlir_op_trait!(ExtractSlice, OneResult);
mlir_op_trait!(ExtractSlice, Pure);
mlir_op_trait!(ExtractSlice, ZeroRegions);
mlir_op_trait!(ExtractSlice, ZeroSuccessors);

/// Constructs a new detached/owned [`ExtractSliceOperation`] at the specified [`Location`].
pub fn extract_slice<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    offsets: &[StaticOrDynamicIndex<'v, 'c, 't>],
    sizes: &[StaticOrDynamicIndex<'v, 'c, 't>],
    strides: &[StaticOrDynamicIndex<'v, 'c, 't>],
    result_type: T,
    location: L,
) -> DetachedExtractSliceOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::tensor());
    let dynamic_index = unsafe { Size::Dynamic.to_c_api() };
    let static_offsets = offsets.iter().map(|index| index.static_value().unwrap_or(dynamic_index)).collect::<Vec<_>>();
    let dynamic_offsets = offsets.iter().filter_map(StaticOrDynamicIndex::dynamic_value).collect::<Vec<_>>();
    let static_sizes = sizes.iter().map(|index| index.static_value().unwrap_or(dynamic_index)).collect::<Vec<_>>();
    let dynamic_sizes = sizes.iter().filter_map(StaticOrDynamicIndex::dynamic_value).collect::<Vec<_>>();
    let static_strides = strides.iter().map(|index| index.static_value().unwrap_or(dynamic_index)).collect::<Vec<_>>();
    let dynamic_strides = strides.iter().filter_map(StaticOrDynamicIndex::dynamic_value).collect::<Vec<_>>();
    OperationBuilder::new("tensor.extract_slice", location)
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context
                .dense_i32_array_attribute(&[
                    1,
                    dynamic_offsets.len() as i32,
                    dynamic_sizes.len() as i32,
                    dynamic_strides.len() as i32,
                ])
                .unwrap(),
        )
        .add_attribute(STATIC_OFFSETS_ATTRIBUTE, context.dense_i64_array_attribute(&static_offsets).unwrap())
        .add_attribute(STATIC_SIZES_ATTRIBUTE, context.dense_i64_array_attribute(&static_sizes).unwrap())
        .add_attribute(STATIC_STRIDES_ATTRIBUTE, context.dense_i64_array_attribute(&static_strides).unwrap())
        .add_operand(source)
        .add_operands(&dynamic_offsets)
        .add_operands(&dynamic_sizes)
        .add_operands(&dynamic_strides)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tensor::extract_slice`")
}

/// Operation trait for `tensor.from_elements`.
pub trait FromElementsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the scalar elements used to construct the tensor.
    fn elements(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }

    /// Returns the constructed tensor.
    fn tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(FromElements);
mlir_op_trait!(FromElements, AlwaysSpeculatable);
mlir_op_trait!(FromElements, NoMemoryEffect);
mlir_op_trait!(FromElements, OneResult);
mlir_op_trait!(FromElements, Pure);
mlir_op_trait!(FromElements, ZeroRegions);
mlir_op_trait!(FromElements, ZeroSuccessors);

/// Constructs a new detached/owned [`FromElementsOperation`] at the specified [`Location`].
pub fn from_elements<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    elements: &[ValueRef<'v, 'c, 't>],
    result_type: T,
    location: L,
) -> DetachedFromElementsOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::tensor());
    OperationBuilder::new("tensor.from_elements", location)
        .add_operands(elements)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tensor::from_elements`")
}

/// Name of the `tensor.gather` dimension-list attribute.
pub const GATHER_DIMS_ATTRIBUTE: &str = "gather_dims";

/// Name of the unit attribute marking unique gather/scatter coordinates.
pub const UNIQUE_ATTRIBUTE: &str = "unique";

/// Operation trait for `tensor.gather`.
pub trait GatherOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source tensor.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the indices tensor.
    fn indices(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the gathered source dimensions.
    fn gather_dimensions(&self) -> DenseInteger64ArrayAttributeRef<'c, 't> {
        self.attribute(GATHER_DIMS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast())
            .unwrap_or_else(|| panic!("invalid '{GATHER_DIMS_ATTRIBUTE}' attribute in `tensor.gather`"))
    }

    /// Returns whether coordinates are statically guaranteed to be unique.
    fn unique(&self) -> bool {
        self.has_attribute(UNIQUE_ATTRIBUTE)
    }

    /// Returns the gathered tensor.
    fn gathered(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(Gather);
mlir_op_trait!(Gather, AlwaysSpeculatable);
mlir_op_trait!(Gather, NoMemoryEffect);
mlir_op_trait!(Gather, OneResult);
mlir_op_trait!(Gather, Pure);
mlir_op_trait!(Gather, ZeroRegions);
mlir_op_trait!(Gather, ZeroSuccessors);

/// Constructs a new detached/owned [`GatherOperation`] at the specified [`Location`].
pub fn gather<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    indices: ValueRef<'v, 'c, 't>,
    gather_dimensions: &[i64],
    unique: bool,
    result_type: T,
    location: L,
) -> DetachedGatherOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::tensor());
    let mut builder = OperationBuilder::new("tensor.gather", location)
        .add_operand(source)
        .add_operand(indices)
        .add_attribute(GATHER_DIMS_ATTRIBUTE, context.dense_i64_array_attribute(gather_dimensions).unwrap())
        .add_result(result_type);
    if unique {
        builder = builder.add_attribute(UNIQUE_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tensor::gather`")
}

/// Operation trait for `tensor.generate`.
pub trait GenerateOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the dynamic extent operands.
    fn dynamic_extents(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }

    /// Returns the region that yields tensor elements.
    fn body_region(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }

    /// Returns the generated tensor.
    fn tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(Generate);
mlir_op_trait!(Generate, OneRegion);
mlir_op_trait!(Generate, OneResult);
mlir_op_trait!(Generate, SingleBlockRegions);
mlir_op_trait!(Generate, ZeroSuccessors);

/// Constructs a new detached/owned [`GenerateOperation`] at the specified [`Location`].
pub fn generate<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    dynamic_extents: &[ValueRef<'v, 'c, 't>],
    result_type: T,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedGenerateOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::tensor());
    OperationBuilder::new("tensor.generate", location)
        .add_operands(dynamic_extents)
        .add_result(result_type)
        .add_region(body)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tensor::generate`")
}

/// Operation trait for `tensor.insert`.
pub trait InsertOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the scalar value being inserted.
    fn scalar(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the destination tensor.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the index operands.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        (2..self.operand_count()).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns the tensor containing the inserted scalar.
    fn updated(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(Insert);
mlir_op_trait!(Insert, AlwaysSpeculatable);
mlir_op_trait!(Insert, NoMemoryEffect);
mlir_op_trait!(Insert, OneResult);
mlir_op_trait!(Insert, Pure);
mlir_op_trait!(Insert, ZeroRegions);
mlir_op_trait!(Insert, ZeroSuccessors);

/// Constructs a new detached/owned [`InsertOperation`] at the specified [`Location`].
pub fn insert<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    scalar: ValueRef<'v, 'c, 't>,
    destination: ValueRef<'v, 'c, 't>,
    indices: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> DetachedInsertOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::tensor());
    OperationBuilder::new("tensor.insert", location)
        .add_operand(scalar)
        .add_operand(destination)
        .add_operands(indices)
        .add_result(destination.r#type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tensor::insert`")
}

/// Operation trait for `tensor.insert_slice`.
pub trait InsertSliceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the inserted source tensor.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the destination tensor.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the mixed static and dynamic offsets.
    fn offsets(&self) -> Vec<StaticOrDynamicIndex<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| {
                panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `tensor.insert_slice`")
            });
        let dynamic_start = 2;
        let dynamic_end = dynamic_start + segment_sizes[2] as usize;
        let dynamic_offsets =
            (dynamic_start..dynamic_end).map(|index| self.operand_value(index).unwrap()).collect::<Vec<_>>();
        let static_offsets = self
            .attribute(STATIC_OFFSETS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| panic!("invalid '{STATIC_OFFSETS_ATTRIBUTE}' attribute in `tensor.insert_slice`"));
        let dynamic_index = unsafe { Size::Dynamic.to_c_api() };
        let mut dynamic_offsets = dynamic_offsets.into_iter();
        static_offsets
            .into_iter()
            .map(|index| {
                (index == dynamic_index)
                    .then(|| dynamic_offsets.next().expect("missing dynamic offset operand"))
                    .map_or_else(|| StaticOrDynamicIndex::Static(index), StaticOrDynamicIndex::Dynamic)
            })
            .collect()
    }

    /// Returns the mixed static and dynamic sizes.
    fn sizes(&self) -> Vec<StaticOrDynamicIndex<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| {
                panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `tensor.insert_slice`")
            });
        let dynamic_start = 2 + segment_sizes[2] as usize;
        let dynamic_end = dynamic_start + segment_sizes[3] as usize;
        let dynamic_sizes =
            (dynamic_start..dynamic_end).map(|index| self.operand_value(index).unwrap()).collect::<Vec<_>>();
        let static_sizes = self
            .attribute(STATIC_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| panic!("invalid '{STATIC_SIZES_ATTRIBUTE}' attribute in `tensor.insert_slice`"));
        let dynamic_index = unsafe { Size::Dynamic.to_c_api() };
        let mut dynamic_sizes = dynamic_sizes.into_iter();
        static_sizes
            .into_iter()
            .map(|index| {
                (index == dynamic_index)
                    .then(|| dynamic_sizes.next().expect("missing dynamic size operand"))
                    .map_or_else(|| StaticOrDynamicIndex::Static(index), StaticOrDynamicIndex::Dynamic)
            })
            .collect()
    }

    /// Returns the mixed static and dynamic strides.
    fn strides(&self) -> Vec<StaticOrDynamicIndex<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| {
                panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `tensor.insert_slice`")
            });
        let dynamic_start = 2 + segment_sizes[2] as usize + segment_sizes[3] as usize;
        let dynamic_end = dynamic_start + segment_sizes[4] as usize;
        let dynamic_strides =
            (dynamic_start..dynamic_end).map(|index| self.operand_value(index).unwrap()).collect::<Vec<_>>();
        let static_strides = self
            .attribute(STATIC_STRIDES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| panic!("invalid '{STATIC_STRIDES_ATTRIBUTE}' attribute in `tensor.insert_slice`"));
        let dynamic_index = unsafe { Size::Dynamic.to_c_api() };
        let mut dynamic_strides = dynamic_strides.into_iter();
        static_strides
            .into_iter()
            .map(|index| {
                (index == dynamic_index)
                    .then(|| dynamic_strides.next().expect("missing dynamic stride operand"))
                    .map_or_else(|| StaticOrDynamicIndex::Static(index), StaticOrDynamicIndex::Dynamic)
            })
            .collect()
    }

    /// Returns the updated tensor.
    fn updated(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(InsertSlice);
mlir_op_trait!(InsertSlice, AlwaysSpeculatable);
mlir_op_trait!(InsertSlice, NoMemoryEffect);
mlir_op_trait!(InsertSlice, OneResult);
mlir_op_trait!(InsertSlice, Pure);
mlir_op_trait!(InsertSlice, ZeroRegions);
mlir_op_trait!(InsertSlice, ZeroSuccessors);

/// Constructs a new detached/owned [`InsertSliceOperation`] at the specified [`Location`].
pub fn insert_slice<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    destination: ValueRef<'v, 'c, 't>,
    offsets: &[StaticOrDynamicIndex<'v, 'c, 't>],
    sizes: &[StaticOrDynamicIndex<'v, 'c, 't>],
    strides: &[StaticOrDynamicIndex<'v, 'c, 't>],
    location: L,
) -> DetachedInsertSliceOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::tensor());
    let dynamic_index = unsafe { Size::Dynamic.to_c_api() };
    let static_offsets = offsets.iter().map(|index| index.static_value().unwrap_or(dynamic_index)).collect::<Vec<_>>();
    let dynamic_offsets = offsets.iter().filter_map(StaticOrDynamicIndex::dynamic_value).collect::<Vec<_>>();
    let static_sizes = sizes.iter().map(|index| index.static_value().unwrap_or(dynamic_index)).collect::<Vec<_>>();
    let dynamic_sizes = sizes.iter().filter_map(StaticOrDynamicIndex::dynamic_value).collect::<Vec<_>>();
    let static_strides = strides.iter().map(|index| index.static_value().unwrap_or(dynamic_index)).collect::<Vec<_>>();
    let dynamic_strides = strides.iter().filter_map(StaticOrDynamicIndex::dynamic_value).collect::<Vec<_>>();
    OperationBuilder::new("tensor.insert_slice", location)
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context
                .dense_i32_array_attribute(&[
                    1,
                    1,
                    dynamic_offsets.len() as i32,
                    dynamic_sizes.len() as i32,
                    dynamic_strides.len() as i32,
                ])
                .unwrap(),
        )
        .add_attribute(STATIC_OFFSETS_ATTRIBUTE, context.dense_i64_array_attribute(&static_offsets).unwrap())
        .add_attribute(STATIC_SIZES_ATTRIBUTE, context.dense_i64_array_attribute(&static_sizes).unwrap())
        .add_attribute(STATIC_STRIDES_ATTRIBUTE, context.dense_i64_array_attribute(&static_strides).unwrap())
        .add_operand(source)
        .add_operand(destination)
        .add_operands(&dynamic_offsets)
        .add_operands(&dynamic_sizes)
        .add_operands(&dynamic_strides)
        .add_result(destination.r#type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tensor::insert_slice`")
}

/// Operation trait for `tensor.rank`.
pub trait RankOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the tensor whose rank is queried.
    fn tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the tensor rank.
    fn rank(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(Rank);
mlir_op_trait!(Rank, AlwaysSpeculatable);
mlir_op_trait!(Rank, NoMemoryEffect);
mlir_op_trait!(Rank, OneResult);
mlir_op_trait!(Rank, Pure);
mlir_op_trait!(Rank, ZeroRegions);
mlir_op_trait!(Rank, ZeroSuccessors);

/// Constructs a new detached/owned [`RankOperation`] at the specified [`Location`].
pub fn rank<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    tensor: ValueRef<'v, 'c, 't>,
    location: L,
) -> DetachedRankOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::tensor());
    OperationBuilder::new("tensor.rank", location)
        .add_operand(tensor)
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tensor::rank`")
}

/// Operation trait for `tensor.reshape`.
pub trait ReshapeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source tensor.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the tensor containing the new shape.
    fn shape(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the reshaped tensor.
    fn reshaped(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(Reshape);
mlir_op_trait!(Reshape, AlwaysSpeculatable);
mlir_op_trait!(Reshape, NoMemoryEffect);
mlir_op_trait!(Reshape, OneResult);
mlir_op_trait!(Reshape, Pure);
mlir_op_trait!(Reshape, ZeroRegions);
mlir_op_trait!(Reshape, ZeroSuccessors);

/// Constructs a new detached/owned [`ReshapeOperation`] at the specified [`Location`].
pub fn reshape<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    shape: ValueRef<'v, 'c, 't>,
    result_type: T,
    location: L,
) -> DetachedReshapeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::tensor());
    OperationBuilder::new("tensor.reshape", location)
        .add_operand(source)
        .add_operand(shape)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tensor::reshape`")
}

/// Name of the reassociation groups attribute.
pub const REASSOCIATION_ATTRIBUTE: &str = "reassociation";

/// Name of the static output-shape entries attribute.
pub const STATIC_OUTPUT_SHAPE_ATTRIBUTE: &str = "static_output_shape";

/// Operation trait shared by Tensor reassociative reshape operations.
pub trait ReassociativeReshapeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source tensor.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the reassociation groups.
    fn reassociation(&self) -> ArrayAttributeRef<'c, 't> {
        self.attribute(REASSOCIATION_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<ArrayAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{REASSOCIATION_ATTRIBUTE}' attribute in `{}`", self.name()))
    }

    /// Returns the reshaped tensor.
    fn reshaped(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

/// Operation trait for `tensor.expand_shape`.
pub trait ExpandShapeOperation<'o, 'c: 'o, 't: 'c>: ReassociativeReshapeOperation<'o, 'c, 't> {
    /// Returns the mixed static and dynamic output-shape entries.
    fn output_shape(&self) -> Vec<StaticOrDynamicIndex<'o, 'c, 't>> {
        let dynamic_shape =
            (1..self.operand_count()).map(|index| self.operand_value(index).unwrap()).collect::<Vec<_>>();
        let static_shape = self
            .attribute(STATIC_OUTPUT_SHAPE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| panic!("invalid '{STATIC_OUTPUT_SHAPE_ATTRIBUTE}' attribute in `tensor.expand_shape`"));
        let dynamic_index = unsafe { Size::Dynamic.to_c_api() };
        let mut dynamic_shape = dynamic_shape.into_iter();
        static_shape
            .into_iter()
            .map(|index| {
                (index == dynamic_index)
                    .then(|| dynamic_shape.next().expect("missing dynamic output-shape operand"))
                    .map_or_else(|| StaticOrDynamicIndex::Static(index), StaticOrDynamicIndex::Dynamic)
            })
            .collect()
    }
}

mlir_op!(ExpandShape);
mlir_op_trait!(ExpandShape, AlwaysSpeculatable);
mlir_op_trait!(ExpandShape, NoMemoryEffect);
mlir_op_trait!(ExpandShape, OneResult);
mlir_op_trait!(ExpandShape, Pure);
mlir_op_trait!(ExpandShape, ZeroRegions);
mlir_op_trait!(ExpandShape, ZeroSuccessors);
mlir_op_trait!(ExpandShape, @local ReassociativeReshapeOperation);

/// Constructs a new detached/owned [`ExpandShapeOperation`] at the specified [`Location`].
pub fn expand_shape<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    reassociation: &[&[i64]],
    output_shape: &[StaticOrDynamicIndex<'v, 'c, 't>],
    result_type: T,
    location: L,
) -> DetachedExpandShapeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::tensor());
    let dynamic_index = unsafe { Size::Dynamic.to_c_api() };
    let index_attribute_type = context.signless_integer_type(64);
    let reassociation = reassociation
        .iter()
        .map(|group| {
            let group = group
                .iter()
                .map(|index| context.integer_attribute(index_attribute_type, *index))
                .collect::<Vec<_>>();
            context.array_attribute(&group).as_ref()
        })
        .collect::<Vec<_>>();
    let static_output_shape =
        output_shape.iter().map(|index| index.static_value().unwrap_or(dynamic_index)).collect::<Vec<_>>();
    let dynamic_output_shape = output_shape.iter().filter_map(StaticOrDynamicIndex::dynamic_value).collect::<Vec<_>>();
    OperationBuilder::new("tensor.expand_shape", location)
        .add_operand(source)
        .add_operands(&dynamic_output_shape)
        .add_attribute(REASSOCIATION_ATTRIBUTE, context.array_attribute(&reassociation))
        .add_attribute(STATIC_OUTPUT_SHAPE_ATTRIBUTE, context.dense_i64_array_attribute(&static_output_shape).unwrap())
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tensor::expand_shape`")
}

/// Operation trait for `tensor.collapse_shape`.
pub trait CollapseShapeOperation<'o, 'c: 'o, 't: 'c>: ReassociativeReshapeOperation<'o, 'c, 't> {}

mlir_op!(CollapseShape);
mlir_op_trait!(CollapseShape, AlwaysSpeculatable);
mlir_op_trait!(CollapseShape, NoMemoryEffect);
mlir_op_trait!(CollapseShape, OneResult);
mlir_op_trait!(CollapseShape, Pure);
mlir_op_trait!(CollapseShape, ZeroRegions);
mlir_op_trait!(CollapseShape, ZeroSuccessors);
mlir_op_trait!(CollapseShape, @local ReassociativeReshapeOperation);

/// Constructs a new detached/owned [`CollapseShapeOperation`] at the specified [`Location`].
pub fn collapse_shape<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    reassociation: &[&[i64]],
    result_type: T,
    location: L,
) -> DetachedCollapseShapeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::tensor());
    let index_attribute_type = context.signless_integer_type(64);
    let reassociation = reassociation
        .iter()
        .map(|group| {
            let group = group
                .iter()
                .map(|index| context.integer_attribute(index_attribute_type, *index))
                .collect::<Vec<_>>();
            context.array_attribute(&group).as_ref()
        })
        .collect::<Vec<_>>();
    OperationBuilder::new("tensor.collapse_shape", location)
        .add_operand(source)
        .add_attribute(REASSOCIATION_ATTRIBUTE, context.array_attribute(&reassociation))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tensor::collapse_shape`")
}

/// Name of the static low-padding entries attribute.
pub const STATIC_LOW_ATTRIBUTE: &str = "static_low";

/// Name of the static high-padding entries attribute.
pub const STATIC_HIGH_ATTRIBUTE: &str = "static_high";

/// Name of the `tensor.pad` unit attribute that disables folding.
pub const NOFOLD_ATTRIBUTE: &str = "nofold";

/// Operation trait for `tensor.pad`.
pub trait PadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source tensor.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the mixed static and dynamic low-padding entries.
    fn low(&self) -> Vec<StaticOrDynamicIndex<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `tensor.pad`"));
        let dynamic_start = 1;
        let dynamic_end = dynamic_start + segment_sizes[1] as usize;
        let dynamic_low =
            (dynamic_start..dynamic_end).map(|index| self.operand_value(index).unwrap()).collect::<Vec<_>>();
        let static_low = self
            .attribute(STATIC_LOW_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| panic!("invalid '{STATIC_LOW_ATTRIBUTE}' attribute in `tensor.pad`"));
        let dynamic_index = unsafe { Size::Dynamic.to_c_api() };
        let mut dynamic_low = dynamic_low.into_iter();
        static_low
            .into_iter()
            .map(|index| {
                (index == dynamic_index)
                    .then(|| dynamic_low.next().expect("missing dynamic low-padding operand"))
                    .map_or_else(|| StaticOrDynamicIndex::Static(index), StaticOrDynamicIndex::Dynamic)
            })
            .collect()
    }

    /// Returns the mixed static and dynamic high-padding entries.
    fn high(&self) -> Vec<StaticOrDynamicIndex<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `tensor.pad`"));
        let dynamic_start = 1 + segment_sizes[1] as usize;
        let dynamic_end = dynamic_start + segment_sizes[2] as usize;
        let dynamic_high =
            (dynamic_start..dynamic_end).map(|index| self.operand_value(index).unwrap()).collect::<Vec<_>>();
        let static_high = self
            .attribute(STATIC_HIGH_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| panic!("invalid '{STATIC_HIGH_ATTRIBUTE}' attribute in `tensor.pad`"));
        let dynamic_index = unsafe { Size::Dynamic.to_c_api() };
        let mut dynamic_high = dynamic_high.into_iter();
        static_high
            .into_iter()
            .map(|index| {
                (index == dynamic_index)
                    .then(|| dynamic_high.next().expect("missing dynamic high-padding operand"))
                    .map_or_else(|| StaticOrDynamicIndex::Static(index), StaticOrDynamicIndex::Dynamic)
            })
            .collect()
    }

    /// Returns whether the operation should not be folded.
    fn nofold(&self) -> bool {
        self.has_attribute(NOFOLD_ATTRIBUTE)
    }

    /// Returns the region that yields padding values.
    fn body_region(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }

    /// Returns the padded tensor.
    fn padded(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(Pad);
mlir_op_trait!(Pad, AlwaysSpeculatable);
mlir_op_trait!(Pad, NoMemoryEffect);
mlir_op_trait!(Pad, OneRegion);
mlir_op_trait!(Pad, OneResult);
mlir_op_trait!(Pad, Pure);
mlir_op_trait!(Pad, SingleBlockRegions);
mlir_op_trait!(Pad, ZeroSuccessors);

/// Constructs a new detached/owned [`PadOperation`] at the specified [`Location`].
pub fn pad<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    low: &[StaticOrDynamicIndex<'v, 'c, 't>],
    high: &[StaticOrDynamicIndex<'v, 'c, 't>],
    nofold: bool,
    result_type: T,
    region: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedPadOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::tensor());
    let dynamic_index = unsafe { Size::Dynamic.to_c_api() };
    let static_low = low.iter().map(|index| index.static_value().unwrap_or(dynamic_index)).collect::<Vec<_>>();
    let dynamic_low = low.iter().filter_map(StaticOrDynamicIndex::dynamic_value).collect::<Vec<_>>();
    let static_high = high.iter().map(|index| index.static_value().unwrap_or(dynamic_index)).collect::<Vec<_>>();
    let dynamic_high = high.iter().filter_map(StaticOrDynamicIndex::dynamic_value).collect::<Vec<_>>();
    let mut builder = OperationBuilder::new("tensor.pad", location)
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context
                .dense_i32_array_attribute(&[1, dynamic_low.len() as i32, dynamic_high.len() as i32])
                .unwrap(),
        )
        .add_attribute(STATIC_LOW_ATTRIBUTE, context.dense_i64_array_attribute(&static_low).unwrap())
        .add_attribute(STATIC_HIGH_ATTRIBUTE, context.dense_i64_array_attribute(&static_high).unwrap())
        .add_operand(source)
        .add_operands(&dynamic_low)
        .add_operands(&dynamic_high)
        .add_result(result_type)
        .add_region(region);
    if nofold {
        builder = builder.add_attribute(NOFOLD_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tensor::pad`")
}

/// Operation trait for `tensor.parallel_insert_slice`.
pub trait ParallelInsertSliceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the inserted source tensor.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the destination tensor.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the mixed static and dynamic offsets.
    fn offsets(&self) -> Vec<StaticOrDynamicIndex<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| {
                panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `tensor.parallel_insert_slice`")
            });
        let dynamic_start = 2;
        let dynamic_end = dynamic_start + segment_sizes[2] as usize;
        let dynamic_offsets =
            (dynamic_start..dynamic_end).map(|index| self.operand_value(index).unwrap()).collect::<Vec<_>>();
        let static_offsets = self
            .attribute(STATIC_OFFSETS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| {
                panic!("invalid '{STATIC_OFFSETS_ATTRIBUTE}' attribute in `tensor.parallel_insert_slice`")
            });
        let dynamic_index = unsafe { Size::Dynamic.to_c_api() };
        let mut dynamic_offsets = dynamic_offsets.into_iter();
        static_offsets
            .into_iter()
            .map(|index| {
                (index == dynamic_index)
                    .then(|| dynamic_offsets.next().expect("missing dynamic offset operand"))
                    .map_or_else(|| StaticOrDynamicIndex::Static(index), StaticOrDynamicIndex::Dynamic)
            })
            .collect()
    }

    /// Returns the mixed static and dynamic sizes.
    fn sizes(&self) -> Vec<StaticOrDynamicIndex<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| {
                panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `tensor.parallel_insert_slice`")
            });
        let dynamic_start = 2 + segment_sizes[2] as usize;
        let dynamic_end = dynamic_start + segment_sizes[3] as usize;
        let dynamic_sizes =
            (dynamic_start..dynamic_end).map(|index| self.operand_value(index).unwrap()).collect::<Vec<_>>();
        let static_sizes = self
            .attribute(STATIC_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| {
                panic!("invalid '{STATIC_SIZES_ATTRIBUTE}' attribute in `tensor.parallel_insert_slice`")
            });
        let dynamic_index = unsafe { Size::Dynamic.to_c_api() };
        let mut dynamic_sizes = dynamic_sizes.into_iter();
        static_sizes
            .into_iter()
            .map(|index| {
                (index == dynamic_index)
                    .then(|| dynamic_sizes.next().expect("missing dynamic size operand"))
                    .map_or_else(|| StaticOrDynamicIndex::Static(index), StaticOrDynamicIndex::Dynamic)
            })
            .collect()
    }

    /// Returns the mixed static and dynamic strides.
    fn strides(&self) -> Vec<StaticOrDynamicIndex<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| {
                panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `tensor.parallel_insert_slice`")
            });
        let dynamic_start = 2 + segment_sizes[2] as usize + segment_sizes[3] as usize;
        let dynamic_end = dynamic_start + segment_sizes[4] as usize;
        let dynamic_strides =
            (dynamic_start..dynamic_end).map(|index| self.operand_value(index).unwrap()).collect::<Vec<_>>();
        let static_strides = self
            .attribute(STATIC_STRIDES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger64ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| {
                panic!("invalid '{STATIC_STRIDES_ATTRIBUTE}' attribute in `tensor.parallel_insert_slice`")
            });
        let dynamic_index = unsafe { Size::Dynamic.to_c_api() };
        let mut dynamic_strides = dynamic_strides.into_iter();
        static_strides
            .into_iter()
            .map(|index| {
                (index == dynamic_index)
                    .then(|| dynamic_strides.next().expect("missing dynamic stride operand"))
                    .map_or_else(|| StaticOrDynamicIndex::Static(index), StaticOrDynamicIndex::Dynamic)
            })
            .collect()
    }
}

mlir_op!(ParallelInsertSlice);
mlir_op_trait!(ParallelInsertSlice, ZeroRegions);
mlir_op_trait!(ParallelInsertSlice, ZeroSuccessors);

/// Constructs a new detached/owned [`ParallelInsertSliceOperation`] at the specified [`Location`].
pub fn parallel_insert_slice<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    destination: ValueRef<'v, 'c, 't>,
    offsets: &[StaticOrDynamicIndex<'v, 'c, 't>],
    sizes: &[StaticOrDynamicIndex<'v, 'c, 't>],
    strides: &[StaticOrDynamicIndex<'v, 'c, 't>],
    location: L,
) -> DetachedParallelInsertSliceOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::tensor());
    let dynamic_index = unsafe { Size::Dynamic.to_c_api() };
    let static_offsets = offsets.iter().map(|index| index.static_value().unwrap_or(dynamic_index)).collect::<Vec<_>>();
    let dynamic_offsets = offsets.iter().filter_map(StaticOrDynamicIndex::dynamic_value).collect::<Vec<_>>();
    let static_sizes = sizes.iter().map(|index| index.static_value().unwrap_or(dynamic_index)).collect::<Vec<_>>();
    let dynamic_sizes = sizes.iter().filter_map(StaticOrDynamicIndex::dynamic_value).collect::<Vec<_>>();
    let static_strides = strides.iter().map(|index| index.static_value().unwrap_or(dynamic_index)).collect::<Vec<_>>();
    let dynamic_strides = strides.iter().filter_map(StaticOrDynamicIndex::dynamic_value).collect::<Vec<_>>();
    OperationBuilder::new("tensor.parallel_insert_slice", location)
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context
                .dense_i32_array_attribute(&[
                    1,
                    1,
                    dynamic_offsets.len() as i32,
                    dynamic_sizes.len() as i32,
                    dynamic_strides.len() as i32,
                ])
                .unwrap(),
        )
        .add_attribute(STATIC_OFFSETS_ATTRIBUTE, context.dense_i64_array_attribute(&static_offsets).unwrap())
        .add_attribute(STATIC_SIZES_ATTRIBUTE, context.dense_i64_array_attribute(&static_sizes).unwrap())
        .add_attribute(STATIC_STRIDES_ATTRIBUTE, context.dense_i64_array_attribute(&static_strides).unwrap())
        .add_operand(source)
        .add_operand(destination)
        .add_operands(&dynamic_offsets)
        .add_operands(&dynamic_sizes)
        .add_operands(&dynamic_strides)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tensor::parallel_insert_slice`")
}

/// Name of the `tensor.scatter` dimension-list attribute.
pub const SCATTER_DIMS_ATTRIBUTE: &str = "scatter_dims";

/// Operation trait for `tensor.scatter`.
pub trait ScatterOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the source tensor.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the destination tensor.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the indices tensor.
    fn indices(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the scattered destination dimensions.
    fn scatter_dimensions(&self) -> DenseInteger64ArrayAttributeRef<'c, 't> {
        self.attribute(SCATTER_DIMS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast())
            .unwrap_or_else(|| panic!("invalid '{SCATTER_DIMS_ATTRIBUTE}' attribute in `tensor.scatter`"))
    }

    /// Returns whether coordinates are statically guaranteed to be unique.
    fn unique(&self) -> bool {
        self.has_attribute(UNIQUE_ATTRIBUTE)
    }

    /// Returns the scattered tensor.
    fn scattered(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(Scatter);
mlir_op_trait!(Scatter, AlwaysSpeculatable);
mlir_op_trait!(Scatter, NoMemoryEffect);
mlir_op_trait!(Scatter, OneResult);
mlir_op_trait!(Scatter, Pure);
mlir_op_trait!(Scatter, ZeroRegions);
mlir_op_trait!(Scatter, ZeroSuccessors);

/// Constructs a new detached/owned [`ScatterOperation`] at the specified [`Location`].
pub fn scatter<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    destination: ValueRef<'v, 'c, 't>,
    indices: ValueRef<'v, 'c, 't>,
    scatter_dimensions: &[i64],
    unique: bool,
    location: L,
) -> DetachedScatterOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::tensor());
    let mut builder = OperationBuilder::new("tensor.scatter", location)
        .add_operand(source)
        .add_operand(destination)
        .add_operand(indices)
        .add_attribute(SCATTER_DIMS_ATTRIBUTE, context.dense_i64_array_attribute(scatter_dimensions).unwrap())
        .add_result(destination.r#type());
    if unique {
        builder = builder.add_attribute(UNIQUE_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tensor::scatter`")
}

/// Operation trait for `tensor.splat`.
pub trait SplatOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneResult<'o, 'c, 't> {
    /// Returns the scalar input.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the dynamic size operands.
    fn dynamic_sizes(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        (1..self.operand_count()).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns the splatted tensor.
    fn aggregate(&self) -> ValueRef<'o, 'c, 't> {
        self.output()
    }
}

mlir_op!(Splat);
mlir_op_trait!(Splat, AlwaysSpeculatable);
mlir_op_trait!(Splat, NoMemoryEffect);
mlir_op_trait!(Splat, OneResult);
mlir_op_trait!(Splat, Pure);
mlir_op_trait!(Splat, ZeroRegions);
mlir_op_trait!(Splat, ZeroSuccessors);

/// Constructs a new detached/owned [`SplatOperation`] at the specified [`Location`].
pub fn splat<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    input: ValueRef<'v, 'c, 't>,
    dynamic_sizes: &[ValueRef<'v, 'c, 't>],
    result_type: T,
    location: L,
) -> DetachedSplatOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::tensor());
    OperationBuilder::new("tensor.splat", location)
        .add_operand(input)
        .add_operands(dynamic_sizes)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tensor::splat`")
}

/// Operation trait for `tensor.yield`.
pub trait YieldOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the yielded value.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }
}

mlir_op!(Yield);
mlir_op_trait!(Yield, AlwaysSpeculatable);
mlir_op_trait!(Yield, IsTerminator);
mlir_op_trait!(Yield, NoMemoryEffect);
mlir_op_trait!(Yield, Pure);
mlir_op_trait!(Yield, ReturnLike);
mlir_op_trait!(Yield, SingleBlockRegions);
mlir_op_trait!(Yield, ZeroRegions);
mlir_op_trait!(Yield, ZeroSuccessors);

/// Constructs a new detached/owned [`YieldOperation`] at the specified [`Location`].
pub fn r#yield<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    value: ValueRef<'v, 'c, 't>,
    location: L,
) -> DetachedYieldOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::tensor());
    OperationBuilder::new("tensor.yield", location)
        .add_operand(value)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tensor::yield`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::{func, scf};
    use crate::{Block, Context, Region, Size, Type, Value};

    use super::*;

    #[test]
    fn test_bitcast() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let source_type =
            context.tensor_type(context.signless_integer_type(32), &[Size::Static(2)], None, location).unwrap();
        let result_type =
            context.tensor_type(context.unsigned_integer_type(32), &[Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(source_type, location)]);
            let source = block.argument(0).unwrap().into();
            let op = bitcast(source, result_type, location);
            assert_eq!(op.source(), source);
            assert_eq!(op.destination().r#type(), result_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "bitcast_test",
                func::FuncAttributes {
                    arguments: vec![source_type.into()],
                    results: vec![result_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @bitcast_test(%arg0: tensor<2xi32>) -> tensor<2xui32> {
                    %0 = tensor.bitcast %arg0 : tensor<2xi32> to tensor<2xui32>
                    return %0 : tensor<2xui32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_cast() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let source_type = context.unranked_tensor_type(context.float32_type(), location).unwrap();
        let result_type = context
            .tensor_type(context.float32_type(), &[Size::Dynamic, Size::Dynamic], None, location)
            .unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(source_type, location)]);
            let source = block.argument(0).unwrap().into();
            let op = cast(source, result_type, location);
            assert_eq!(op.source(), source);
            assert_eq!(op.destination().r#type(), result_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "cast_test",
                func::FuncAttributes {
                    arguments: vec![source_type.into()],
                    results: vec![result_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @cast_test(%arg0: tensor<*xf32>) -> tensor<?x?xf32> {
                    %cast = tensor.cast %arg0 : tensor<*xf32> to tensor<?x?xf32>
                    return %cast : tensor<?x?xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_concat() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let lhs_type = context.tensor_type(f32_type, &[Size::Static(2)], None, location).unwrap();
        let rhs_type = context.tensor_type(f32_type, &[Size::Static(3)], None, location).unwrap();
        let result_type = context.tensor_type(f32_type, &[Size::Static(5)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(lhs_type, location), (rhs_type, location)]);
            let lhs = block.argument(0).unwrap().into();
            let rhs = block.argument(1).unwrap().into();
            let op = concat(&[lhs, rhs], 0, result_type, location);
            assert_eq!(op.inputs(), vec![lhs, rhs]);
            assert_eq!(op.dimension(), 0);
            assert_eq!(op.concatenated().r#type(), result_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "concat_test",
                func::FuncAttributes {
                    arguments: vec![lhs_type.into(), rhs_type.into()],
                    results: vec![result_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @concat_test(%arg0: tensor<2xf32>, %arg1: tensor<3xf32>) -> tensor<5xf32> {
                    %concat = tensor.concat dim(0) %arg0, %arg1 : (tensor<2xf32>, tensor<3xf32>) -> tensor<5xf32>
                    return %concat : tensor<5xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_dim() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let tensor_type = context
            .tensor_type(context.float32_type(), &[Size::Static(4), Size::Dynamic], None, location)
            .unwrap();
        let index_type = context.index_type();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type.as_ref(), location), (index_type.as_ref(), location)]);
            let source = block.argument(0).unwrap().into();
            let index = block.argument(1).unwrap().into();
            let op = dim(source, index, location);
            assert_eq!(op.source(), source);
            assert_eq!(op.index(), index);
            assert_eq!(op.size().r#type(), index_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "dim_test",
                func::FuncAttributes {
                    arguments: vec![tensor_type.into(), index_type.into()],
                    results: vec![index_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @dim_test(%arg0: tensor<4x?xf32>, %arg1: index) -> index {
                    %dim = tensor.dim %arg0, %arg1 : tensor<4x?xf32>
                    return %dim : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_empty() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let index_type = context.index_type();
        let result_type = context
            .tensor_type(context.float32_type(), &[Size::Dynamic, Size::Static(4)], None, location)
            .unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(index_type, location)]);
            let dynamic_size = block.argument(0).unwrap().into();
            let op = empty(&[dynamic_size], result_type, location);
            assert_eq!(op.dynamic_sizes(), vec![dynamic_size]);
            assert_eq!(op.tensor().r#type(), result_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "empty_test",
                func::FuncAttributes {
                    arguments: vec![index_type.into()],
                    results: vec![result_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @empty_test(%arg0: index) -> tensor<?x4xf32> {
                    %0 = tensor.empty(%arg0) : tensor<?x4xf32>
                    return %0 : tensor<?x4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_extract() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let index_type = context.index_type();
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(4), Size::Static(4)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (tensor_type.as_ref(), location),
                (index_type.as_ref(), location),
                (index_type.as_ref(), location),
            ]);
            let tensor = block.argument(0).unwrap().into();
            let index_0 = block.argument(1).unwrap().into();
            let index_1 = block.argument(2).unwrap().into();
            let op = extract(tensor, &[index_0, index_1], i32_type, location);
            assert_eq!(op.tensor(), tensor);
            assert_eq!(op.indices(), vec![index_0, index_1]);
            assert_eq!(op.element().r#type(), i32_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "extract_test",
                func::FuncAttributes {
                    arguments: vec![tensor_type.into(), index_type.into(), index_type.into()],
                    results: vec![i32_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @extract_test(%arg0: tensor<4x4xi32>, %arg1: index, %arg2: index) -> i32 {
                    %extracted = tensor.extract %arg0[%arg1, %arg2] : tensor<4x4xi32>
                    return %extracted : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_extract_slice() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let index_type = context.index_type();
        let source_type = context
            .tensor_type(context.float32_type(), &[Size::Dynamic, Size::Static(16)], None, location)
            .unwrap();
        let result_type = context
            .tensor_type(context.float32_type(), &[Size::Static(4), Size::Static(8)], None, location)
            .unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(source_type.as_ref(), location), (index_type.as_ref(), location)]);
            let source = block.argument(0).unwrap().into();
            let dynamic_offset = block.argument(1).unwrap().into();
            let offsets = [StaticOrDynamicIndex::Dynamic(dynamic_offset), StaticOrDynamicIndex::Static(2)];
            let sizes = [StaticOrDynamicIndex::Static(4), StaticOrDynamicIndex::Static(8)];
            let strides = [StaticOrDynamicIndex::Static(1), StaticOrDynamicIndex::Static(1)];
            let op = extract_slice(source, &offsets, &sizes, &strides, result_type, location);
            assert_eq!(op.source(), source);
            assert_eq!(op.offsets(), offsets);
            assert_eq!(op.sizes(), sizes);
            assert_eq!(op.strides(), strides);
            assert_eq!(op.slice().r#type(), result_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "extract_slice_test",
                func::FuncAttributes {
                    arguments: vec![source_type.into(), index_type.into()],
                    results: vec![result_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @extract_slice_test(%arg0: tensor<?x16xf32>, %arg1: index) -> tensor<4x8xf32> {
                    %extracted_slice = tensor.extract_slice %arg0[%arg1, 2] [4, 8] [1, 1] : tensor<?x16xf32> to tensor<4x8xf32>
                    return %extracted_slice : tensor<4x8xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_from_elements() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let result_type = context.tensor_type(i32_type, &[Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location), (i32_type, location)]);
            let element_0 = block.argument(0).unwrap().into();
            let element_1 = block.argument(1).unwrap().into();
            let op = from_elements(&[element_0, element_1], result_type, location);
            assert_eq!(op.elements(), vec![element_0, element_1]);
            assert_eq!(op.tensor().r#type(), result_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "from_elements_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
                    results: vec![result_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @from_elements_test(%arg0: i32, %arg1: i32) -> tensor<2xi32> {
                    %from_elements = tensor.from_elements %arg0, %arg1 : tensor<2xi32>
                    return %from_elements : tensor<2xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_gather() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let index_type = context.index_type();
        let source_type = context.tensor_type(f32_type, &[Size::Static(4), Size::Static(4)], None, location).unwrap();
        let indices_type =
            context.tensor_type(index_type, &[Size::Static(2), Size::Static(1)], None, location).unwrap();
        let result_type = context
            .tensor_type(f32_type, &[Size::Static(2), Size::Static(4), Size::Static(1)], None, location)
            .unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(source_type, location), (indices_type, location)]);
            let source = block.argument(0).unwrap().into();
            let indices = block.argument(1).unwrap().into();
            let op = gather(source, indices, &[1], true, result_type, location);
            assert_eq!(op.source(), source);
            assert_eq!(op.indices(), indices);
            assert_eq!(op.gather_dimensions().values().collect::<Vec<_>>(), vec![1]);
            assert!(op.unique());
            assert_eq!(op.gathered().r#type(), result_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "gather_test",
                func::FuncAttributes {
                    arguments: vec![source_type.into(), indices_type.into()],
                    results: vec![result_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @gather_test(%arg0: tensor<4x4xf32>, %arg1: tensor<2x1xindex>) -> tensor<2x4x1xf32> {
                    %gather = tensor.gather %arg0[%arg1] gather_dims([1]) unique : (tensor<4x4xf32>, tensor<2x1xindex>) -> tensor<2x4x1xf32>
                    return %gather : tensor<2x4x1xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_generate() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let index_type = context.index_type();
        let f32_type = context.float32_type();
        let result_type = context.tensor_type(f32_type, &[Size::Dynamic], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(index_type.as_ref(), location), (f32_type.as_ref(), location)]);
            let dynamic_extent = block.argument(0).unwrap().into();
            let element = block.argument(1).unwrap().into();
            let mut body = context.region();
            let mut body_block = context.block(&[(index_type, location)]);
            let yield_op = r#yield(element, location);
            assert_eq!(yield_op.value(), element);
            body_block.append_operation(yield_op);
            body.append_block(body_block);
            let op = generate(&[dynamic_extent], result_type, body, location);
            assert_eq!(op.dynamic_extents(), vec![dynamic_extent]);
            assert_eq!(op.body_region().blocks().count(), 1);
            assert_eq!(op.tensor().r#type(), result_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "generate_test",
                func::FuncAttributes {
                    arguments: vec![index_type.into(), f32_type.into()],
                    results: vec![result_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @generate_test(%arg0: index, %arg1: f32) -> tensor<?xf32> {
                    %generated = tensor.generate %arg0 {
                    ^bb0(%arg2: index):
                      tensor.yield %arg1 : f32
                    } : tensor<?xf32>
                    return %generated : tensor<?xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_insert() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let index_type = context.index_type();
        let tensor_type = context.tensor_type(i32_type, &[Size::Static(4), Size::Static(4)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (i32_type.as_ref(), location),
                (tensor_type.as_ref(), location),
                (index_type.as_ref(), location),
                (index_type.as_ref(), location),
            ]);
            let scalar = block.argument(0).unwrap().into();
            let destination = block.argument(1).unwrap().into();
            let index_0 = block.argument(2).unwrap().into();
            let index_1 = block.argument(3).unwrap().into();
            let op = insert(scalar, destination, &[index_0, index_1], location);
            assert_eq!(op.scalar(), scalar);
            assert_eq!(op.destination(), destination);
            assert_eq!(op.indices(), vec![index_0, index_1]);
            assert_eq!(op.updated().r#type(), tensor_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "insert_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), tensor_type.into(), index_type.into(), index_type.into()],
                    results: vec![tensor_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @insert_test(%arg0: i32, %arg1: tensor<4x4xi32>, %arg2: index, %arg3: index) -> tensor<4x4xi32> {
                    %inserted = tensor.insert %arg0 into %arg1[%arg2, %arg3] : tensor<4x4xi32>
                    return %inserted : tensor<4x4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_insert_slice() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let index_type = context.index_type();
        let source_type = context
            .tensor_type(context.float32_type(), &[Size::Static(4), Size::Static(8)], None, location)
            .unwrap();
        let destination_type = context
            .tensor_type(context.float32_type(), &[Size::Dynamic, Size::Static(16)], None, location)
            .unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (source_type.as_ref(), location),
                (destination_type.as_ref(), location),
                (index_type.as_ref(), location),
            ]);
            let source = block.argument(0).unwrap().into();
            let destination = block.argument(1).unwrap().into();
            let dynamic_offset = block.argument(2).unwrap().into();
            let offsets = [StaticOrDynamicIndex::Dynamic(dynamic_offset), StaticOrDynamicIndex::Static(2)];
            let sizes = [StaticOrDynamicIndex::Static(4), StaticOrDynamicIndex::Static(8)];
            let strides = [StaticOrDynamicIndex::Static(1), StaticOrDynamicIndex::Static(1)];
            let op = insert_slice(source, destination, &offsets, &sizes, &strides, location);
            assert_eq!(op.source(), source);
            assert_eq!(op.destination(), destination);
            assert_eq!(op.offsets(), offsets);
            assert_eq!(op.sizes(), sizes);
            assert_eq!(op.strides(), strides);
            assert_eq!(op.updated().r#type(), destination_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "insert_slice_test",
                func::FuncAttributes {
                    arguments: vec![source_type.into(), destination_type.into(), index_type.into()],
                    results: vec![destination_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @insert_slice_test(%arg0: tensor<4x8xf32>, %arg1: tensor<?x16xf32>, %arg2: index) -> tensor<?x16xf32> {
                    %inserted_slice = tensor.insert_slice %arg0 into %arg1[%arg2, 2] [4, 8] [1, 1] : tensor<4x8xf32> into tensor<?x16xf32>
                    return %inserted_slice : tensor<?x16xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_rank() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let tensor_type = context.unranked_tensor_type(context.float32_type(), location).unwrap();
        let index_type = context.index_type();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let tensor = block.argument(0).unwrap().into();
            let op = rank(tensor, location);
            assert_eq!(op.tensor(), tensor);
            assert_eq!(op.rank().r#type(), index_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "rank_test",
                func::FuncAttributes {
                    arguments: vec![tensor_type.into()],
                    results: vec![index_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @rank_test(%arg0: tensor<*xf32>) -> index {
                    %rank = tensor.rank %arg0 : tensor<*xf32>
                    return %rank : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_reshape() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let i32_type = context.signless_integer_type(32);
        let source_type = context.tensor_type(f32_type, &[Size::Static(4), Size::Static(1)], None, location).unwrap();
        let shape_type = context.tensor_type(i32_type, &[Size::Static(1)], None, location).unwrap();
        let result_type = context.tensor_type(f32_type, &[Size::Static(4)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(source_type, location), (shape_type, location)]);
            let source = block.argument(0).unwrap().into();
            let shape = block.argument(1).unwrap().into();
            let op = reshape(source, shape, result_type, location);
            assert_eq!(op.source(), source);
            assert_eq!(op.shape(), shape);
            assert_eq!(op.reshaped().r#type(), result_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "reshape_test",
                func::FuncAttributes {
                    arguments: vec![source_type.into(), shape_type.into()],
                    results: vec![result_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @reshape_test(%arg0: tensor<4x1xf32>, %arg1: tensor<1xi32>) -> tensor<4xf32> {
                    %reshape = tensor.reshape %arg0(%arg1) : (tensor<4x1xf32>, tensor<1xi32>) -> tensor<4xf32>
                    return %reshape : tensor<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_expand_shape() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let index_type = context.index_type();
        let f32_type = context.float32_type();
        let source_type = context.tensor_type(f32_type, &[Size::Dynamic, Size::Static(32)], None, location).unwrap();
        let result_type = context
            .tensor_type(f32_type, &[Size::Dynamic, Size::Static(4), Size::Static(32)], None, location)
            .unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(source_type.as_ref(), location), (index_type.as_ref(), location)]);
            let source = block.argument(0).unwrap().into();
            let dynamic_output = block.argument(1).unwrap().into();
            let output_shape = [
                StaticOrDynamicIndex::Dynamic(dynamic_output),
                StaticOrDynamicIndex::Static(4),
                StaticOrDynamicIndex::Static(32),
            ];
            let op = expand_shape(source, &[&[0, 1], &[2]], &output_shape, result_type, location);
            assert_eq!(op.source(), source);
            assert_eq!(op.output_shape(), output_shape);
            assert_eq!(op.reshaped().r#type(), result_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "expand_shape_test",
                func::FuncAttributes {
                    arguments: vec![source_type.into(), index_type.into()],
                    results: vec![result_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @expand_shape_test(%arg0: tensor<?x32xf32>, %arg1: index) -> tensor<?x4x32xf32> {
                    %expanded = tensor.expand_shape %arg0 [[0, 1], [2]] output_shape [%arg1, 4, 32] : tensor<?x32xf32> into tensor<?x4x32xf32>
                    return %expanded : tensor<?x4x32xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_collapse_shape() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let source_type = context
            .tensor_type(f32_type, &[Size::Dynamic, Size::Static(4), Size::Static(32)], None, location)
            .unwrap();
        let result_type = context.tensor_type(f32_type, &[Size::Dynamic, Size::Static(32)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(source_type, location)]);
            let source = block.argument(0).unwrap().into();
            let op = collapse_shape(source, &[&[0, 1], &[2]], result_type, location);
            assert_eq!(op.source(), source);
            assert_eq!(op.reshaped().r#type(), result_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "collapse_shape_test",
                func::FuncAttributes {
                    arguments: vec![source_type.into()],
                    results: vec![result_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @collapse_shape_test(%arg0: tensor<?x4x32xf32>) -> tensor<?x32xf32> {
                    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<?x4x32xf32> into tensor<?x32xf32>
                    return %collapsed : tensor<?x32xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_pad() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let source_type = context.tensor_type(f32_type, &[Size::Static(2)], None, location).unwrap();
        let result_type = context.tensor_type(f32_type, &[Size::Static(4)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(source_type.as_ref(), location), (f32_type.as_ref(), location)]);
            let source = block.argument(0).unwrap().into();
            let value = block.argument(1).unwrap().into();
            let low = [StaticOrDynamicIndex::Static(1)];
            let high = [StaticOrDynamicIndex::Static(1)];
            let mut region = context.region();
            let mut region_block = context.block(&[(context.index_type(), location)]);
            let yield_op = r#yield(value, location);
            assert_eq!(yield_op.value(), value);
            region_block.append_operation(yield_op);
            region.append_block(region_block);
            let op = pad(source, &low, &high, false, result_type, region, location);
            assert_eq!(op.source(), source);
            assert_eq!(op.low(), low);
            assert_eq!(op.high(), high);
            assert!(!op.nofold());
            assert_eq!(op.body_region().blocks().count(), 1);
            assert_eq!(op.padded().r#type(), result_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "pad_test",
                func::FuncAttributes {
                    arguments: vec![source_type.into(), f32_type.into()],
                    results: vec![result_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @pad_test(%arg0: tensor<2xf32>, %arg1: f32) -> tensor<4xf32> {
                    %padded = tensor.pad %arg0 low[1] high[1] {
                    ^bb0(%arg2: index):
                      tensor.yield %arg1 : f32
                    } : tensor<2xf32> to tensor<4xf32>
                    return %padded : tensor<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_parallel_insert_slice() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let source_type = context.tensor_type(f32_type, &[Size::Static(1)], None, location).unwrap();
        let destination_type = context.tensor_type(f32_type, &[Size::Static(4)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(source_type, location), (destination_type, location)]);
            let source = block.argument(0).unwrap().into();
            let destination = block.argument(1).unwrap().into();
            let mut body = context.region();
            let mut body_block =
                context.block(&[(context.index_type().as_ref(), location), (destination_type.as_ref(), location)]);
            let induction_variable = body_block.argument(0).unwrap().into();
            let output_argument = body_block.argument(1).unwrap().into();
            let offsets = [StaticOrDynamicIndex::Dynamic(induction_variable)];
            let sizes = [StaticOrDynamicIndex::Static(1)];
            let strides = [StaticOrDynamicIndex::Static(1)];
            let mut in_parallel_region = context.region();
            let mut in_parallel_block = context.block_with_no_arguments();
            let op = parallel_insert_slice(source, output_argument, &offsets, &sizes, &strides, location);
            assert_eq!(op.source(), source);
            assert_eq!(op.destination(), output_argument);
            assert_eq!(op.offsets(), offsets);
            assert_eq!(op.sizes(), sizes);
            assert_eq!(op.strides(), strides);
            in_parallel_block.append_operation(op);
            in_parallel_region.append_block(in_parallel_block);
            body_block.append_operation(scf::in_parallel(in_parallel_region, location));
            body.append_block(body_block);
            let op = scf::for_all(&[], &[], &[], &[0], &[4], &[1], &[destination], None, body, location);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "parallel_insert_slice_test",
                func::FuncAttributes {
                    arguments: vec![source_type.into(), destination_type.into()],
                    results: vec![destination_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @parallel_insert_slice_test(%arg0: tensor<1xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = scf.forall (%arg2) in (4) shared_outs(%arg3 = %arg1) -> (tensor<4xf32>) {
                      scf.forall.in_parallel {
                        tensor.parallel_insert_slice %arg0 into %arg3[%arg2] [1] [1] : tensor<1xf32> into tensor<4xf32>
                      }
                    }
                    return %0 : tensor<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_scatter() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let index_type = context.index_type();
        let source_type = context
            .tensor_type(f32_type, &[Size::Static(2), Size::Static(4), Size::Static(1)], None, location)
            .unwrap();
        let destination_type =
            context.tensor_type(f32_type, &[Size::Static(4), Size::Static(4)], None, location).unwrap();
        let indices_type =
            context.tensor_type(index_type, &[Size::Static(2), Size::Static(1)], None, location).unwrap();
        module.body().append_operation({
            let mut block =
                context.block(&[(source_type, location), (destination_type, location), (indices_type, location)]);
            let source = block.argument(0).unwrap().into();
            let destination = block.argument(1).unwrap().into();
            let indices = block.argument(2).unwrap().into();
            let op = scatter(source, destination, indices, &[1], true, location);
            assert_eq!(op.source(), source);
            assert_eq!(op.destination(), destination);
            assert_eq!(op.indices(), indices);
            assert_eq!(op.scatter_dimensions().values().collect::<Vec<_>>(), vec![1]);
            assert!(op.unique());
            assert_eq!(op.scattered().r#type(), destination_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "scatter_test",
                func::FuncAttributes {
                    arguments: vec![source_type.into(), destination_type.into(), indices_type.into()],
                    results: vec![destination_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @scatter_test(%arg0: tensor<2x4x1xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<2x1xindex>) -> tensor<4x4xf32> {
                    %scatter = tensor.scatter %arg0 into %arg1[%arg2] scatter_dims([1]) unique : (tensor<2x4x1xf32>, tensor<4x4xf32>, tensor<2x1xindex>) -> tensor<4x4xf32>
                    return %scatter : tensor<4x4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_splat() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let index_type = context.index_type();
        let f32_type = context.float32_type();
        let result_type = context.tensor_type(f32_type, &[Size::Dynamic, Size::Static(4)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location), (index_type.as_ref(), location)]);
            let input = block.argument(0).unwrap().into();
            let dynamic_size = block.argument(1).unwrap().into();
            let op = splat(input, &[dynamic_size], result_type, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.dynamic_sizes(), vec![dynamic_size]);
            assert_eq!(op.aggregate().r#type(), result_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "splat_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), index_type.into()],
                    results: vec![result_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @splat_test(%arg0: f32, %arg1: index) -> tensor<?x4xf32> {
                    %splat = tensor.splat %arg0[%arg1] : tensor<?x4xf32>
                    return %splat : tensor<?x4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_yield() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let result_type = context.tensor_type(f32_type, &[], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let value = block.argument(0).unwrap().into();
            let mut body = context.region();
            let mut body_block = context.block_with_no_arguments();
            let yield_op = r#yield(value, location);
            assert_eq!(yield_op.value(), value);
            body_block.append_operation(yield_op);
            body.append_block(body_block);
            let op = block.append_operation(generate(&[], result_type, body, location));
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "yield_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into()],
                    results: vec![result_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @yield_test(%arg0: f32) -> tensor<f32> {
                    %generated = tensor.generate  {
                      tensor.yield %arg0 : f32
                    } : tensor<f32>
                    return %generated : tensor<f32>
                  }
                }
            "},
        );
    }
}
