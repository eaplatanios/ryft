use crate::dialects::arith::attributes::FastMathFlagsAttributeRef;
use crate::macros::{mlir_op, mlir_op_trait};
use crate::{
    AffineMapAttributeRef, ArrayAttributeRef, Attribute, AttributeRef, BooleanAttributeRef,
    DenseInteger32ArrayAttributeRef, DenseInteger64ArrayAttributeRef, DetachedOp, DetachedRegion, DialectHandle, Error,
    IndexTypeRef, IntegerAttributeRef, Location, Operation, OperationBuilder, OperationResultRef, RegionRef,
    ShapedType, StringAttributeRef, Type, TypeRef, Value, ValueRef, VectorTypeDimension, VectorTypeRef,
};

use super::attributes::{CombiningKindAttributeRef, IteratorTypeArrayAttributeRef, PrintPunctuationAttributeRef};

macro_rules! vector_attribute_accessor {
    // Accessor for an attribute that every well-formed operation carries.
    (required, $method:ident, $attribute_type:ident, $attribute_name:literal) => {
        vector_attribute_accessor!(
            @present,
            $method,
            $attribute_type,
            $attribute_name,
            concat!("Returns the required `", $attribute_name, "` attribute.")
        );
    };
    // Accessor for a default-valued attribute. MLIR populates the default when an operation is created without it,
    // so the attribute is always present and the accessor is infallible for well-formed operations.
    (default, $method:ident, $attribute_type:ident, $attribute_name:literal) => {
        vector_attribute_accessor!(
            @present,
            $method,
            $attribute_type,
            $attribute_name,
            concat!(
                "Returns the `",
                $attribute_name,
                "` attribute. MLIR materializes the default value of this attribute when the operation is created ",
                "without it, so the attribute is always present.",
            )
        );
    };
    // Internal arm that generates the accessor body shared by the `required` and `default` forms.
    (@present, $method:ident, $attribute_type:ident, $attribute_name:literal, $documentation:expr) => {
        #[doc = $documentation]
        fn $method(&self) -> Result<$attribute_type<'c, 't>, Error> {
            self.attribute($attribute_name)?
                .and_then(|attribute| attribute.cast::<$attribute_type>())
                .ok_or_else(|| Error::invalid_argument(concat!("missing or invalid `", $attribute_name, "` attribute")))
        }
    };
    // Accessor for an attribute that may be absent, such as `OptionalAttr` and `DefaultValuedOptionalAttr`.
    (optional, $method:ident, $attribute_type:ident, $attribute_name:literal) => {
        #[doc = concat!("Returns the optional `", $attribute_name, "` attribute.")]
        fn $method(&self) -> Result<Option<$attribute_type<'c, 't>>, Error> {
            self.attribute($attribute_name)?
                .map(|attribute| {
                    attribute
                        .cast::<$attribute_type>()
                        .ok_or_else(|| Error::invalid_argument(concat!("invalid `", $attribute_name, "` attribute")))
                })
                .transpose()
        }
    };
}

/// Vector [`Operation`] that contracts two vectors according to affine indexing maps and combines the result with an
/// accumulator. Iterator types distinguish parallel dimensions from reduction dimensions.
///
/// # Example
///
/// The following is an example of a [`ContractionOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// #map = affine_map<(d0) -> (d0)>
/// #map1 = affine_map<(d0) -> ()>
/// %0 = vector.contract {indexing_maps = [#map, #map, #map1], iterator_types = ["reduction"], kind =
///   #vector.kind<add>} %arg0, %arg1, %arg2 : vector<2xf32>, vector<2xf32> into f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorcontract-vectorcontractionop
pub trait ContractionOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the right operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the accumulator operand.
    fn accumulator(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    vector_attribute_accessor!(required, indexing_maps, ArrayAttributeRef, "indexing_maps");
    vector_attribute_accessor!(required, iterator_types, IteratorTypeArrayAttributeRef, "iterator_types");
    vector_attribute_accessor!(default, kind, CombiningKindAttributeRef, "kind");
    vector_attribute_accessor!(default, fastmath, FastMathFlagsAttributeRef, "fastmath");
}

mlir_op!(Contraction);
mlir_op_trait!(Contraction, ZeroSuccessors);

/// Typed arguments for [`ContractionOperation`].
pub struct ContractionArguments<'v, 'c: 'v, 't: 'c> {
    /// Left operand.
    pub lhs: ValueRef<'v, 'c, 't>,

    /// Right operand.
    pub rhs: ValueRef<'v, 'c, 't>,

    /// Accumulator used to combine the computed value.
    pub accumulator: ValueRef<'v, 'c, 't>,

    /// Type of the resulting value.
    pub result_type: TypeRef<'c, 't>,

    /// Affine indexing maps for the left operand, right operand, and accumulator, in that order.
    pub indexing_maps: ArrayAttributeRef<'c, 't>,

    /// Parallel or reduction iterator kind for each iteration dimension.
    pub iterator_types: IteratorTypeArrayAttributeRef<'c, 't>,

    /// Optional combining operation used by the reduction or contraction.
    pub kind: Option<CombiningKindAttributeRef<'c, 't>>,

    /// Optional floating-point assumptions; omission uses no fast-math assumptions.
    pub fastmath: Option<FastMathFlagsAttributeRef<'c, 't>>,
}

/// Constructs a new detached/owned [`ContractionOperation`] at the specified [`Location`]. Refer to its documentation
/// for the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`ContractionArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn contraction<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: ContractionArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedContractionOperation<'c, 't>, Error> {
    validate_vector_type(arguments.lhs.r#type()?, "left-hand-side operand", "vector.contract")?;
    validate_vector_type(arguments.rhs.r#type()?, "right-hand-side operand", "vector.contract")?;
    if arguments.accumulator.r#type()? != arguments.result_type {
        return Err(Error::invalid_argument("expected accumulator and result types of `vector.contract` to match"));
    }
    let operands = [arguments.lhs, arguments.rhs, arguments.accumulator];
    let result_types = [arguments.result_type];
    let mut attributes = vec![
        ("indexing_maps", arguments.indexing_maps.as_ref()),
        ("iterator_types", arguments.iterator_types.as_ref()),
    ];
    attributes.extend(arguments.kind.map(|attribute| ("kind", attribute.as_ref())));
    attributes.extend(arguments.fastmath.map(|attribute| ("fastmath", attribute.as_ref())));
    unsafe { raw_contraction(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`ContractionOperation`] from MLIR operands, types, attributes, and regions. This
/// interface supports interoperability with components not represented by [`contraction`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`contraction`] for typed construction.
pub unsafe fn raw_contraction<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedContractionOperation<'c, 't>, Error> {
    if operands.len() != 3 {
        return Err(Error::invalid_argument("invalid operand count for `vector.contract`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.contract`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.contract`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.contract", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::contraction`"))
    })
}

/// Vector [`Operation`] that reduces a one-dimensional vector to a scalar using the selected combining kind and an
/// optional accumulator.
///
/// # Example
///
/// The following is an example of a [`ReductionOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %0 = vector.reduction <add>, %arg0 : vector<4xf32> into f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorreduction-vectorreductionop
pub trait ReductionOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the vector operand.
    fn vector(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the optional accumulator operand.
    fn accumulator(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.operand_count() == 2 { self.operand_value(1).map(Some) } else { Ok(None) }
    }

    vector_attribute_accessor!(required, kind, CombiningKindAttributeRef, "kind");
    vector_attribute_accessor!(default, fastmath, FastMathFlagsAttributeRef, "fastmath");
}

mlir_op!(Reduction);
mlir_op_trait!(Reduction, ZeroSuccessors);

/// Typed arguments for [`ReductionOperation`].
pub struct ReductionArguments<'v, 'c: 'v, 't: 'c> {
    /// Vector to reduce.
    pub vector: ValueRef<'v, 'c, 't>,

    /// Optional accumulator used to combine the computed value.
    pub accumulator: Option<ValueRef<'v, 'c, 't>>,

    /// Type of the resulting value.
    pub result_type: TypeRef<'c, 't>,

    /// Combining operation used by the reduction or contraction.
    pub kind: CombiningKindAttributeRef<'c, 't>,

    /// Optional floating-point assumptions; omission uses no fast-math assumptions.
    pub fastmath: Option<FastMathFlagsAttributeRef<'c, 't>>,
}

/// Constructs a new detached/owned [`ReductionOperation`] at the specified [`Location`]. Refer to its documentation for
/// the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`ReductionArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn reduction<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: ReductionArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedReductionOperation<'c, 't>, Error> {
    let vector_type = validate_vector_type(arguments.vector.r#type()?, "source operand", "vector.reduction")?;
    if vector_type.rank() != 1 {
        return Err(Error::invalid_argument("expected `vector.reduction` source operand to have rank one"));
    }
    if let Some(accumulator) = arguments.accumulator
        && accumulator.r#type()? != arguments.result_type
    {
        return Err(Error::invalid_argument("expected accumulator and result types of `vector.reduction` to match"));
    }
    let mut operands = vec![arguments.vector];
    operands.extend(arguments.accumulator);
    let result_types = [arguments.result_type];
    let mut attributes = vec![("kind", arguments.kind.as_ref())];
    attributes.extend(arguments.fastmath.map(|attribute| ("fastmath", attribute.as_ref())));
    unsafe { raw_reduction(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`ReductionOperation`] from MLIR operands, types, attributes, and regions. This interface
/// supports interoperability with components not represented by [`reduction`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`reduction`] for typed construction.
pub unsafe fn raw_reduction<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedReductionOperation<'c, 't>, Error> {
    if !(1..=2).contains(&operands.len()) {
        return Err(Error::invalid_argument("invalid operand count for `vector.reduction`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.reduction`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.reduction`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.reduction", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::reduction`"))
    })
}

/// Vector [`Operation`] that reduces selected dimensions of a vector and combines the remaining values with the
/// accumulator.
///
/// # Example
///
/// The following is an example of a [`MultiDimReductionOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %0 = vector.multi_reduction <add>, %arg0, %arg1 [0] : vector<2x2xf32> to vector<2xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectormulti_reduction-vectormultidimreductionop
pub trait MultiDimReductionOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the accumulator operand.
    fn accumulator(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    vector_attribute_accessor!(required, kind, CombiningKindAttributeRef, "kind");
    vector_attribute_accessor!(required, reduction_dimensions, DenseInteger64ArrayAttributeRef, "reduction_dims");
}

mlir_op!(MultiDimReduction);
mlir_op_trait!(MultiDimReduction, ZeroSuccessors);

/// Typed arguments for [`MultiDimReductionOperation`].
pub struct MultiDimReductionArguments<'v, 'c: 'v, 't: 'c> {
    /// Source value.
    pub source: ValueRef<'v, 'c, 't>,

    /// Accumulator used to combine the computed value.
    pub accumulator: ValueRef<'v, 'c, 't>,

    /// Type of the resulting value.
    pub result_type: TypeRef<'c, 't>,

    /// Combining operation used by the reduction or contraction.
    pub kind: CombiningKindAttributeRef<'c, 't>,

    /// Dimensions to reduce, in source dimension order.
    pub reduction_dimensions: DenseInteger64ArrayAttributeRef<'c, 't>,
}

/// Constructs a new detached/owned [`MultiDimReductionOperation`] at the specified [`Location`]. Refer to its
/// documentation for the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`MultiDimReductionArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn multi_dim_reduction<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: MultiDimReductionArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedMultiDimReductionOperation<'c, 't>, Error> {
    validate_vector_type(arguments.source.r#type()?, "source operand", "vector.multi_reduction")?;
    if arguments.accumulator.r#type()? != arguments.result_type {
        return Err(Error::invalid_argument(
            "expected accumulator and result types of `vector.multi_reduction` to match",
        ));
    }
    let operands = [arguments.source, arguments.accumulator];
    let result_types = [arguments.result_type];
    let attributes = [("kind", arguments.kind.as_ref()), ("reduction_dims", arguments.reduction_dimensions.as_ref())];
    unsafe { raw_multi_dim_reduction(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`MultiDimReductionOperation`] from MLIR operands, types, attributes, and regions. This
/// interface supports interoperability with components not represented by [`multi_dim_reduction`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`multi_dim_reduction`] for typed construction.
pub unsafe fn raw_multi_dim_reduction<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedMultiDimReductionOperation<'c, 't>, Error> {
    if operands.len() != 2 {
        return Err(Error::invalid_argument("invalid operand count for `vector.multi_reduction`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.multi_reduction`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.multi_reduction`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.multi_reduction", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::multi_dim_reduction`"))
    })
}

/// Vector [`Operation`] that broadcasts a scalar or lower-rank vector to a vector with compatible trailing dimensions.
///
/// # Example
///
/// The following is an example of a [`BroadcastOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %0 = vector.broadcast %arg0 : f32 to vector<4xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorbroadcast-vectorbroadcastop
pub trait BroadcastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(Broadcast);
mlir_op_trait!(Broadcast, ZeroSuccessors);

/// Typed arguments for [`BroadcastOperation`].
pub struct BroadcastArguments<'v, 'c: 'v, 't: 'c> {
    /// Source value.
    pub source: ValueRef<'v, 'c, 't>,

    /// Type of the resulting value.
    pub result_type: TypeRef<'c, 't>,
}

/// Constructs a new detached/owned [`BroadcastOperation`] at the specified [`Location`]. Refer to its documentation for
/// the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`BroadcastArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn broadcast<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: BroadcastArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedBroadcastOperation<'c, 't>, Error> {
    validate_vector_type(arguments.result_type, "result", "vector.broadcast")?;
    let operands = [arguments.source];
    let result_types = [arguments.result_type];
    let attributes = [];
    unsafe { raw_broadcast(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`BroadcastOperation`] from MLIR operands, types, attributes, and regions. This interface
/// supports interoperability with components not represented by [`broadcast`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`broadcast`] for typed construction.
pub unsafe fn raw_broadcast<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedBroadcastOperation<'c, 't>, Error> {
    if operands.len() != 1 {
        return Err(Error::invalid_argument("invalid operand count for `vector.broadcast`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.broadcast`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.broadcast`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.broadcast", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::broadcast`"))
    })
}

/// Vector [`Operation`] that selects slices from two vectors using a static mask. Mask indices address the
/// concatenation of the two leading dimensions.
///
/// # Example
///
/// The following is an example of a [`ShuffleOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %0 = vector.shuffle %arg0, %arg1 [0, 5, 2, 7] : vector<4xf32>, vector<4xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorshuffle-vectorshuffleop
pub trait ShuffleOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the first operand.
    fn first(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the second operand.
    fn second(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    vector_attribute_accessor!(required, mask, DenseInteger64ArrayAttributeRef, "mask");
}

mlir_op!(Shuffle);
mlir_op_trait!(Shuffle, ZeroSuccessors);

/// Typed arguments for [`ShuffleOperation`].
pub struct ShuffleArguments<'v, 'c: 'v, 't: 'c> {
    /// First input vector.
    pub first: ValueRef<'v, 'c, 't>,

    /// Second input vector.
    pub second: ValueRef<'v, 'c, 't>,

    /// Type of the resulting value.
    pub result_type: TypeRef<'c, 't>,

    /// Boolean vector selecting active lanes.
    pub mask: DenseInteger64ArrayAttributeRef<'c, 't>,
}

/// Constructs a new detached/owned [`ShuffleOperation`] at the specified [`Location`]. Refer to its documentation for
/// the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`ShuffleArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn shuffle<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: ShuffleArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedShuffleOperation<'c, 't>, Error> {
    let first_type = validate_vector_type(arguments.first.r#type()?, "first operand", "vector.shuffle")?;
    let second_type = validate_vector_type(arguments.second.r#type()?, "second operand", "vector.shuffle")?;
    let result_type = validate_vector_type(arguments.result_type, "result", "vector.shuffle")?;
    if first_type.element_type()? != second_type.element_type()?
        || first_type.element_type()? != result_type.element_type()?
    {
        return Err(Error::invalid_argument("expected `vector.shuffle` element types to match"));
    }
    let operands = [arguments.first, arguments.second];
    let result_types = [arguments.result_type];
    let attributes = [("mask", arguments.mask.as_ref())];
    unsafe { raw_shuffle(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`ShuffleOperation`] from MLIR operands, types, attributes, and regions. This interface
/// supports interoperability with components not represented by [`shuffle`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`shuffle`] for typed construction.
pub unsafe fn raw_shuffle<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedShuffleOperation<'c, 't>, Error> {
    if operands.len() != 2 {
        return Err(Error::invalid_argument("invalid operand count for `vector.shuffle`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.shuffle`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.shuffle`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.shuffle", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::shuffle`"))
    })
}

/// Vector [`Operation`] that alternates elements from two vectors along their trailing dimension, doubling that
/// dimension in the result.
///
/// # Example
///
/// The following is an example of a [`InterleaveOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %0 = vector.interleave %arg0, %arg1 : vector<2xf32> -> vector<4xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorinterleave-vectorinterleaveop
pub trait InterleaveOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the right operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(Interleave);
mlir_op_trait!(Interleave, ZeroSuccessors);

/// Typed arguments for [`InterleaveOperation`].
pub struct InterleaveArguments<'v, 'c: 'v, 't: 'c> {
    /// Left operand.
    pub lhs: ValueRef<'v, 'c, 't>,

    /// Right operand.
    pub rhs: ValueRef<'v, 'c, 't>,

    /// Type of the resulting value.
    pub result_type: TypeRef<'c, 't>,
}

/// Constructs a new detached/owned [`InterleaveOperation`] at the specified [`Location`]. Refer to its documentation
/// for the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`InterleaveArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn interleave<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: InterleaveArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedInterleaveOperation<'c, 't>, Error> {
    if arguments.lhs.r#type()? != arguments.rhs.r#type()? {
        return Err(Error::invalid_argument("expected operand types of `vector.interleave` to match"));
    }
    validate_vector_type(arguments.result_type, "result", "vector.interleave")?;
    let operands = [arguments.lhs, arguments.rhs];
    let result_types = [arguments.result_type];
    let attributes = [];
    unsafe { raw_interleave(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`InterleaveOperation`] from MLIR operands, types, attributes, and regions. This
/// interface supports interoperability with components not represented by [`interleave`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`interleave`] for typed construction.
pub unsafe fn raw_interleave<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedInterleaveOperation<'c, 't>, Error> {
    if operands.len() != 2 {
        return Err(Error::invalid_argument("invalid operand count for `vector.interleave`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.interleave`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.interleave`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.interleave", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::interleave`"))
    })
}

/// Vector [`Operation`] that splits alternating elements of the trailing dimension into two vectors of equal type.
///
/// # Example
///
/// The following is an example of a [`DeinterleaveOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %res1, %res2 = vector.deinterleave %arg0 : vector<4xf32> -> vector<2xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectordeinterleave-vectordeinterleaveop
pub trait DeinterleaveOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the first result.
    fn first_result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the second result.
    fn second_result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(1)
    }
}

mlir_op!(Deinterleave);
mlir_op_trait!(Deinterleave, ZeroSuccessors);

/// Typed arguments for [`DeinterleaveOperation`].
pub struct DeinterleaveArguments<'v, 'c: 'v, 't: 'c> {
    /// Source value.
    pub source: ValueRef<'v, 'c, 't>,

    /// Type of the first result vector.
    pub first_result_type: TypeRef<'c, 't>,

    /// Type of the second result vector.
    pub second_result_type: TypeRef<'c, 't>,
}

/// Constructs a new detached/owned [`DeinterleaveOperation`] at the specified [`Location`]. Refer to its documentation
/// for the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`DeinterleaveArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn deinterleave<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: DeinterleaveArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedDeinterleaveOperation<'c, 't>, Error> {
    validate_vector_type(arguments.source.r#type()?, "source operand", "vector.deinterleave")?;
    if arguments.first_result_type != arguments.second_result_type {
        return Err(Error::invalid_argument("expected result types of `vector.deinterleave` to match"));
    }
    let operands = [arguments.source];
    let result_types = [arguments.first_result_type, arguments.second_result_type];
    let attributes = [];
    unsafe { raw_deinterleave(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`DeinterleaveOperation`] from MLIR operands, types, attributes, and regions. This
/// interface supports interoperability with components not represented by [`deinterleave`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`deinterleave`] for typed construction.
pub unsafe fn raw_deinterleave<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedDeinterleaveOperation<'c, 't>, Error> {
    if operands.len() != 1 {
        return Err(Error::invalid_argument("invalid operand count for `vector.deinterleave`"));
    }
    if result_types.len() != 2 {
        return Err(Error::invalid_argument("invalid result count for `vector.deinterleave`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.deinterleave`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.deinterleave", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::deinterleave`"))
    })
}

/// Vector [`Operation`] that extracts an element or subvector at static or dynamic coordinates in the leading
/// dimensions.
///
/// # Example
///
/// The following is an example of a [`ExtractOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %0 = vector.extract %arg0[1] : vector<4xf32> from vector<2x4xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorextract-vectorextractop
pub trait ExtractOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    vector_attribute_accessor!(required, static_position, DenseInteger64ArrayAttributeRef, "static_position");
}

mlir_op!(Extract);
mlir_op_trait!(Extract, ZeroSuccessors);

/// Typed arguments for [`ExtractOperation`].
pub struct ExtractArguments<'v, 'c: 'v, 't: 'c> {
    /// Source value.
    pub source: ValueRef<'v, 'c, 't>,

    /// Dynamic coordinates, in the order of the dynamic entries in `static_position`.
    pub dynamic_position: Vec<ValueRef<'v, 'c, 't>>,

    /// Type of the resulting value.
    pub result_type: TypeRef<'c, 't>,

    /// Coordinates, with MLIR dynamic-index sentinels for entries supplied by `dynamic_position`.
    pub static_position: DenseInteger64ArrayAttributeRef<'c, 't>,
}

/// Constructs a new detached/owned [`ExtractOperation`] at the specified [`Location`]. Refer to its documentation for
/// the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`ExtractArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn extract<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: ExtractArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedExtractOperation<'c, 't>, Error> {
    let source_type = validate_vector_type(arguments.source.r#type()?, "source operand", "vector.extract")?;
    validate_indices(&arguments.dynamic_position, "vector.extract")?;
    // Dynamic coordinates already occupy entries in `static_position`.
    if arguments.static_position.len() > source_type.rank() {
        return Err(Error::invalid_argument("too many position entries for `vector.extract`"));
    }
    let mut operands = vec![arguments.source];
    operands.extend(arguments.dynamic_position);
    let result_types = [arguments.result_type];
    let attributes = [("static_position", arguments.static_position.as_ref())];
    unsafe { raw_extract(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`ExtractOperation`] from MLIR operands, types, attributes, and regions. This interface
/// supports interoperability with components not represented by [`extract`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`extract`] for typed construction.
pub unsafe fn raw_extract<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedExtractOperation<'c, 't>, Error> {
    if operands.is_empty() {
        return Err(Error::invalid_argument("invalid operand count for `vector.extract`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.extract`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.extract`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.extract", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::extract`"))
    })
}

/// Vector [`Operation`] that computes an element-wise fused multiply-add of two vectors and an accumulator, with a
/// single rounding per element.
///
/// # Example
///
/// The following is an example of a [`FmaOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %0 = vector.fma %arg0, %arg1, %arg2 : vector<4xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorfma-vectorfmaop
pub trait FmaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the right operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the accumulator operand.
    fn accumulator(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(Fma);
mlir_op_trait!(Fma, ZeroSuccessors);

/// Typed arguments for [`FmaOperation`].
pub struct FmaArguments<'v, 'c: 'v, 't: 'c> {
    /// Left operand.
    pub lhs: ValueRef<'v, 'c, 't>,

    /// Right operand.
    pub rhs: ValueRef<'v, 'c, 't>,

    /// Accumulator used to combine the computed value.
    pub accumulator: ValueRef<'v, 'c, 't>,

    /// Type of the resulting value.
    pub result_type: TypeRef<'c, 't>,
}

/// Constructs a new detached/owned [`FmaOperation`] at the specified [`Location`]. Refer to its documentation for the
/// operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`FmaArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn fma<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: FmaArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedFmaOperation<'c, 't>, Error> {
    if arguments.lhs.r#type()? != arguments.rhs.r#type()?
        || arguments.lhs.r#type()? != arguments.accumulator.r#type()?
        || arguments.lhs.r#type()? != arguments.result_type
    {
        return Err(Error::invalid_argument("expected all types of `vector.fma` to match"));
    }
    validate_vector_type(arguments.result_type, "result", "vector.fma")?;
    let operands = [arguments.lhs, arguments.rhs, arguments.accumulator];
    let result_types = [arguments.result_type];
    let attributes = [];
    unsafe { raw_fma(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`FmaOperation`] from MLIR operands, types, attributes, and regions. This interface
/// supports interoperability with components not represented by [`fma`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`fma`] for typed construction.
pub unsafe fn raw_fma<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedFmaOperation<'c, 't>, Error> {
    if operands.len() != 3 {
        return Err(Error::invalid_argument("invalid operand count for `vector.fma`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.fma`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.fma`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.fma", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::fma`"))
    })
}

/// Vector [`Operation`] that decomposes a fixed-size vector into scalar results in row-major element order.
///
/// # Example
///
/// The following is an example of a [`ToElementsOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %0:2 = vector.to_elements %arg0 : vector<2xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorto_elements-vectortoelementsop
pub trait ToElementsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(ToElements);
mlir_op_trait!(ToElements, ZeroSuccessors);

/// Typed arguments for [`ToElementsOperation`].
pub struct ToElementsArguments<'v, 'c: 'v, 't: 'c> {
    /// Source value.
    pub source: ValueRef<'v, 'c, 't>,

    /// Result element types in vector element order.
    pub element_types: Vec<TypeRef<'c, 't>>,
}

/// Constructs a new detached/owned [`ToElementsOperation`] at the specified [`Location`]. Refer to its documentation
/// for the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`ToElementsArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn to_elements<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: ToElementsArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedToElementsOperation<'c, 't>, Error> {
    let source_type = validate_vector_type(arguments.source.r#type()?, "source operand", "vector.to_elements")?;
    let element_type = source_type.element_type()?;
    if arguments.element_types.iter().any(|r#type| *r#type != element_type) {
        return Err(Error::invalid_argument("expected result element types of `vector.to_elements` to match"));
    }
    let operands = [arguments.source];
    let mut result_types = Vec::new();
    result_types.extend(arguments.element_types);
    let attributes = [];
    unsafe { raw_to_elements(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`ToElementsOperation`] from MLIR operands, types, attributes, and regions. This
/// interface supports interoperability with components not represented by [`to_elements`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`to_elements`] for typed construction.
pub unsafe fn raw_to_elements<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedToElementsOperation<'c, 't>, Error> {
    if operands.len() != 1 {
        return Err(Error::invalid_argument("invalid operand count for `vector.to_elements`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.to_elements`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.to_elements", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::to_elements`"))
    })
}

/// Vector [`Operation`] that constructs a fixed-size vector from scalar operands in row-major element order.
///
/// # Example
///
/// The following is an example of a [`FromElementsOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %0 = vector.from_elements %arg0, %arg1 : vector<2xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorfrom_elements-vectorfromelementsop
pub trait FromElementsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(FromElements);
mlir_op_trait!(FromElements, ZeroSuccessors);

/// Typed arguments for [`FromElementsOperation`].
pub struct FromElementsArguments<'v, 'c: 'v, 't: 'c> {
    /// Input elements in vector element order.
    pub elements: Vec<ValueRef<'v, 'c, 't>>,

    /// Type of the resulting value.
    pub result_type: TypeRef<'c, 't>,
}

/// Constructs a new detached/owned [`FromElementsOperation`] at the specified [`Location`]. Refer to its documentation
/// for the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`FromElementsArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn from_elements<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: FromElementsArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedFromElementsOperation<'c, 't>, Error> {
    let result_type = validate_vector_type(arguments.result_type, "result", "vector.from_elements")?;
    let element_type = result_type.element_type()?;
    for element in &arguments.elements {
        if element.r#type()? != element_type {
            return Err(Error::invalid_argument("expected operand element types of `vector.from_elements` to match"));
        }
    }
    let mut operands = Vec::new();
    operands.extend(arguments.elements);
    let result_types = [arguments.result_type];
    let attributes = [];
    unsafe { raw_from_elements(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`FromElementsOperation`] from MLIR operands, types, attributes, and regions. This
/// interface supports interoperability with components not represented by [`from_elements`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`from_elements`] for typed construction.
pub unsafe fn raw_from_elements<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedFromElementsOperation<'c, 't>, Error> {
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.from_elements`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.from_elements`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.from_elements", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::from_elements`"))
    })
}

/// Vector [`Operation`] that inserts an element or subvector at static or dynamic coordinates, preserving the rest of
/// the destination vector.
///
/// # Example
///
/// The following is an example of a [`InsertOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %0 = vector.insert %arg0, %arg1 [1] : vector<4xf32> into vector<2x4xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorinsert-vectorinsertop
pub trait InsertOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the value to store operand.
    fn value_to_store(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the destination operand.
    fn destination(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    vector_attribute_accessor!(required, static_position, DenseInteger64ArrayAttributeRef, "static_position");
}

mlir_op!(Insert);
mlir_op_trait!(Insert, ZeroSuccessors);

/// Typed arguments for [`InsertOperation`].
pub struct InsertArguments<'v, 'c: 'v, 't: 'c> {
    /// Value to insert or store.
    pub value_to_store: ValueRef<'v, 'c, 't>,

    /// Destination vector.
    pub destination: ValueRef<'v, 'c, 't>,

    /// Dynamic coordinates, in the order of the dynamic entries in `static_position`.
    pub dynamic_position: Vec<ValueRef<'v, 'c, 't>>,

    /// Type of the resulting value.
    pub result_type: TypeRef<'c, 't>,

    /// Coordinates, with MLIR dynamic-index sentinels for entries supplied by `dynamic_position`.
    pub static_position: DenseInteger64ArrayAttributeRef<'c, 't>,
}

/// Constructs a new detached/owned [`InsertOperation`] at the specified [`Location`]. Refer to its documentation for
/// the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`InsertArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn insert<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: InsertArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedInsertOperation<'c, 't>, Error> {
    if arguments.destination.r#type()? != arguments.result_type {
        return Err(Error::invalid_argument("expected destination and result types of `vector.insert` to match"));
    }
    validate_vector_type(arguments.result_type, "result", "vector.insert")?;
    validate_indices(&arguments.dynamic_position, "vector.insert")?;
    let mut operands = vec![arguments.value_to_store, arguments.destination];
    operands.extend(arguments.dynamic_position);
    let result_types = [arguments.result_type];
    let attributes = [("static_position", arguments.static_position.as_ref())];
    unsafe { raw_insert(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`InsertOperation`] from MLIR operands, types, attributes, and regions. This interface
/// supports interoperability with components not represented by [`insert`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`insert`] for typed construction.
pub unsafe fn raw_insert<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedInsertOperation<'c, 't>, Error> {
    if operands.len() < 2 {
        return Err(Error::invalid_argument("invalid operand count for `vector.insert`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.insert`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.insert`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.insert", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::insert`"))
    })
}

/// Vector [`Operation`] that inserts a subvector into a scalable vector at a static position.
///
/// # Example
///
/// The following is an example of a [`ScalableInsertOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %0 = vector.scalable.insert %arg0, %arg1[0] : vector<2xf32> into vector<[4]xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorscalableinsert-vectorscalableinsertop
pub trait ScalableInsertOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the value to store operand.
    fn value_to_store(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the destination operand.
    fn destination(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    vector_attribute_accessor!(required, position, IntegerAttributeRef, "pos");
}

mlir_op!(ScalableInsert);
mlir_op_trait!(ScalableInsert, ZeroSuccessors);

/// Typed arguments for [`ScalableInsertOperation`].
pub struct ScalableInsertArguments<'v, 'c: 'v, 't: 'c> {
    /// Value to insert or store.
    pub value_to_store: ValueRef<'v, 'c, 't>,

    /// Destination vector.
    pub destination: ValueRef<'v, 'c, 't>,

    /// Type of the resulting value.
    pub result_type: TypeRef<'c, 't>,

    /// Non-negative position in the scalable vector.
    pub position: IntegerAttributeRef<'c, 't>,
}

/// Constructs a new detached/owned [`ScalableInsertOperation`] at the specified [`Location`]. Refer to its
/// documentation for the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`ScalableInsertArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn scalable_insert<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: ScalableInsertArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedScalableInsertOperation<'c, 't>, Error> {
    if arguments.destination.r#type()? != arguments.result_type {
        return Err(Error::invalid_argument(
            "expected destination and result types of `vector.scalable.insert` to match",
        ));
    }
    if arguments.position.signless_value() < 0 {
        return Err(Error::invalid_argument("expected non-negative `pos` for `vector.scalable.insert`"));
    }
    let operands = [arguments.value_to_store, arguments.destination];
    let result_types = [arguments.result_type];
    let attributes = [("pos", arguments.position.as_ref())];
    unsafe { raw_scalable_insert(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`ScalableInsertOperation`] from MLIR operands, types, attributes, and regions. This
/// interface supports interoperability with components not represented by [`scalable_insert`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`scalable_insert`] for typed construction.
pub unsafe fn raw_scalable_insert<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedScalableInsertOperation<'c, 't>, Error> {
    if operands.len() != 2 {
        return Err(Error::invalid_argument("invalid operand count for `vector.scalable.insert`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.scalable.insert`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.scalable.insert`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.scalable.insert", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::scalable_insert`"))
    })
}

/// Vector [`Operation`] that extracts a subvector from a scalable vector at a static position.
///
/// # Example
///
/// The following is an example of a [`ScalableExtractOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %0 = vector.scalable.extract %arg0[0] : vector<2xf32> from vector<[4]xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorscalableextract-vectorscalableextractop
pub trait ScalableExtractOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    vector_attribute_accessor!(required, position, IntegerAttributeRef, "pos");
}

mlir_op!(ScalableExtract);
mlir_op_trait!(ScalableExtract, ZeroSuccessors);

/// Typed arguments for [`ScalableExtractOperation`].
pub struct ScalableExtractArguments<'v, 'c: 'v, 't: 'c> {
    /// Source value.
    pub source: ValueRef<'v, 'c, 't>,

    /// Type of the resulting value.
    pub result_type: TypeRef<'c, 't>,

    /// Non-negative position in the scalable vector.
    pub position: IntegerAttributeRef<'c, 't>,
}

/// Constructs a new detached/owned [`ScalableExtractOperation`] at the specified [`Location`]. Refer to its
/// documentation for the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`ScalableExtractArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn scalable_extract<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: ScalableExtractArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedScalableExtractOperation<'c, 't>, Error> {
    validate_vector_type(arguments.source.r#type()?, "source operand", "vector.scalable.extract")?;
    validate_vector_type(arguments.result_type, "result", "vector.scalable.extract")?;
    if arguments.position.signless_value() < 0 {
        return Err(Error::invalid_argument("expected non-negative `pos` for `vector.scalable.extract`"));
    }
    let operands = [arguments.source];
    let result_types = [arguments.result_type];
    let attributes = [("pos", arguments.position.as_ref())];
    unsafe { raw_scalable_extract(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`ScalableExtractOperation`] from MLIR operands, types, attributes, and regions. This
/// interface supports interoperability with components not represented by [`scalable_extract`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`scalable_extract`] for typed construction.
pub unsafe fn raw_scalable_extract<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedScalableExtractOperation<'c, 't>, Error> {
    if operands.len() != 1 {
        return Err(Error::invalid_argument("invalid operand count for `vector.scalable.extract`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.scalable.extract`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.scalable.extract`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.scalable.extract", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::scalable_extract`"))
    })
}

/// Vector [`Operation`] that inserts a strided slice into a destination vector, preserving elements outside the slice.
///
/// # Example
///
/// The following is an example of a [`InsertStridedSliceOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %0 = vector.insert_strided_slice %arg0, %arg1 offsets = [1], strides = [1] : vector<2xf32> into vector<4xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorinsert_strided_slice-vectorinsertstridedsliceop
pub trait InsertStridedSliceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the value to store operand.
    fn value_to_store(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the destination operand.
    fn destination(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    vector_attribute_accessor!(required, offsets, ArrayAttributeRef, "offsets");
    vector_attribute_accessor!(required, strides, ArrayAttributeRef, "strides");
}

mlir_op!(InsertStridedSlice);
mlir_op_trait!(InsertStridedSlice, ZeroSuccessors);

/// Typed arguments for [`InsertStridedSliceOperation`].
pub struct InsertStridedSliceArguments<'v, 'c: 'v, 't: 'c> {
    /// Value to insert or store.
    pub value_to_store: ValueRef<'v, 'c, 't>,

    /// Destination vector.
    pub destination: ValueRef<'v, 'c, 't>,

    /// Type of the resulting value.
    pub result_type: TypeRef<'c, 't>,

    /// Starting offsets in dimension order.
    pub offsets: ArrayAttributeRef<'c, 't>,

    /// Slice strides in dimension order.
    pub strides: ArrayAttributeRef<'c, 't>,
}

/// Constructs a new detached/owned [`InsertStridedSliceOperation`] at the specified [`Location`]. Refer to its
/// documentation for the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`InsertStridedSliceArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn insert_strided_slice<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: InsertStridedSliceArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedInsertStridedSliceOperation<'c, 't>, Error> {
    if arguments.destination.r#type()? != arguments.result_type {
        return Err(Error::invalid_argument(
            "expected destination and result types of `vector.insert_strided_slice` to match",
        ));
    }
    let operands = [arguments.value_to_store, arguments.destination];
    let result_types = [arguments.result_type];
    let attributes = [("offsets", arguments.offsets.as_ref()), ("strides", arguments.strides.as_ref())];
    unsafe { raw_insert_strided_slice(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`InsertStridedSliceOperation`] from MLIR operands, types, attributes, and regions. This
/// interface supports interoperability with components not represented by [`insert_strided_slice`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`insert_strided_slice`] for typed construction.
pub unsafe fn raw_insert_strided_slice<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedInsertStridedSliceOperation<'c, 't>, Error> {
    if operands.len() != 2 {
        return Err(Error::invalid_argument("invalid operand count for `vector.insert_strided_slice`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.insert_strided_slice`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.insert_strided_slice`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.insert_strided_slice", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::insert_strided_slice`"))
    })
}

/// Vector [`Operation`] that forms the outer product of its operands and optionally combines it with an accumulator.
///
/// # Example
///
/// The following is an example of a [`OuterProductOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %0 = vector.outerproduct %arg0, %arg1 : vector<2xf32>, vector<3xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorouterproduct-vectorouterproductop
pub trait OuterProductOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the right operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    vector_attribute_accessor!(default, kind, CombiningKindAttributeRef, "kind");
}

mlir_op!(OuterProduct);
mlir_op_trait!(OuterProduct, ZeroSuccessors);

/// Typed arguments for [`OuterProductOperation`].
pub struct OuterProductArguments<'v, 'c: 'v, 't: 'c> {
    /// Left operand.
    pub lhs: ValueRef<'v, 'c, 't>,

    /// Right operand.
    pub rhs: ValueRef<'v, 'c, 't>,

    /// Optional accumulator used to combine the computed value.
    pub accumulator: Option<ValueRef<'v, 'c, 't>>,

    /// Type of the resulting value.
    pub result_type: TypeRef<'c, 't>,

    /// Optional combining operation used by the reduction or contraction.
    pub kind: Option<CombiningKindAttributeRef<'c, 't>>,
}

/// Constructs a new detached/owned [`OuterProductOperation`] at the specified [`Location`]. Refer to its documentation
/// for the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`OuterProductArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn outer_product<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: OuterProductArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedOuterProductOperation<'c, 't>, Error> {
    validate_vector_type(arguments.lhs.r#type()?, "left-hand-side operand", "vector.outerproduct")?;
    validate_vector_type(arguments.rhs.r#type()?, "right-hand-side operand", "vector.outerproduct")?;
    if let Some(accumulator) = arguments.accumulator
        && accumulator.r#type()? != arguments.result_type
    {
        return Err(Error::invalid_argument("expected accumulator and result types of `vector.outerproduct` to match"));
    }
    let mut operands = vec![arguments.lhs, arguments.rhs];
    operands.extend(arguments.accumulator);
    let result_types = [arguments.result_type];
    let mut attributes = Vec::new();
    attributes.extend(arguments.kind.map(|attribute| ("kind", attribute.as_ref())));
    unsafe { raw_outer_product(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`OuterProductOperation`] from MLIR operands, types, attributes, and regions. This
/// interface supports interoperability with components not represented by [`outer_product`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`outer_product`] for typed construction.
pub unsafe fn raw_outer_product<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedOuterProductOperation<'c, 't>, Error> {
    if !(2..=3).contains(&operands.len()) {
        return Err(Error::invalid_argument("invalid operand count for `vector.outerproduct`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.outerproduct`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.outerproduct`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.outerproduct", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::outer_product`"))
    })
}

/// Vector [`Operation`] that extracts a vector slice specified by static offsets, sizes, and strides.
///
/// # Example
///
/// The following is an example of a [`ExtractStridedSliceOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %0 = vector.extract_strided_slice %arg0 offsets = [1], sizes = [2], strides = [1] : vector<4xf32> to vector<2xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorextract_strided_slice-vectorextractstridedsliceop
pub trait ExtractStridedSliceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    vector_attribute_accessor!(required, offsets, ArrayAttributeRef, "offsets");
    vector_attribute_accessor!(required, sizes, ArrayAttributeRef, "sizes");
    vector_attribute_accessor!(required, strides, ArrayAttributeRef, "strides");
}

mlir_op!(ExtractStridedSlice);
mlir_op_trait!(ExtractStridedSlice, ZeroSuccessors);

/// Typed arguments for [`ExtractStridedSliceOperation`].
pub struct ExtractStridedSliceArguments<'v, 'c: 'v, 't: 'c> {
    /// Source value.
    pub source: ValueRef<'v, 'c, 't>,

    /// Type of the resulting value.
    pub result_type: TypeRef<'c, 't>,

    /// Starting offsets in dimension order.
    pub offsets: ArrayAttributeRef<'c, 't>,

    /// Slice sizes in dimension order.
    pub sizes: ArrayAttributeRef<'c, 't>,

    /// Slice strides in dimension order.
    pub strides: ArrayAttributeRef<'c, 't>,
}

/// Constructs a new detached/owned [`ExtractStridedSliceOperation`] at the specified [`Location`]. Refer to its
/// documentation for the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`ExtractStridedSliceArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn extract_strided_slice<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: ExtractStridedSliceArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedExtractStridedSliceOperation<'c, 't>, Error> {
    validate_vector_type(arguments.source.r#type()?, "source operand", "vector.extract_strided_slice")?;
    validate_vector_type(arguments.result_type, "result", "vector.extract_strided_slice")?;
    let operands = [arguments.source];
    let result_types = [arguments.result_type];
    let attributes = [
        ("offsets", arguments.offsets.as_ref()),
        ("sizes", arguments.sizes.as_ref()),
        ("strides", arguments.strides.as_ref()),
    ];
    unsafe { raw_extract_strided_slice(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`ExtractStridedSliceOperation`] from MLIR operands, types, attributes, and regions. This
/// interface supports interoperability with components not represented by [`extract_strided_slice`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`extract_strided_slice`] for typed construction.
pub unsafe fn raw_extract_strided_slice<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedExtractStridedSliceOperation<'c, 't>, Error> {
    if operands.len() != 1 {
        return Err(Error::invalid_argument("invalid operand count for `vector.extract_strided_slice`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.extract_strided_slice`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.extract_strided_slice`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.extract_strided_slice", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::extract_strided_slice`"))
    })
}

/// Vector [`Operation`] that reads a vector from a memory reference or tensor using an affine permutation map. Padding
/// supplies values for out-of-bounds accesses, and an optional mask selects active lanes.
///
/// # Example
///
/// The following is an example of a [`TransferReadOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %0 = vector.transfer_read %arg0[%arg1], %arg2 {in_bounds = [true]} : memref<4xf32>, vector<2xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectortransfer_read-vectortransferreadop
pub trait TransferReadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the base operand.
    fn base(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    vector_attribute_accessor!(required, permutation_map, AffineMapAttributeRef, "permutation_map");
    vector_attribute_accessor!(required, in_bounds, ArrayAttributeRef, "in_bounds");
    vector_attribute_accessor!(required, operand_segment_sizes, DenseInteger32ArrayAttributeRef, "operandSegmentSizes");
}

mlir_op!(TransferRead);
mlir_op_trait!(TransferRead, ZeroSuccessors);

/// Typed arguments for [`TransferReadOperation`].
pub struct TransferReadArguments<'v, 'c: 'v, 't: 'c> {
    /// Source shaped value.
    pub base: ValueRef<'v, 'c, 't>,

    /// Source indices.
    pub indices: Vec<ValueRef<'v, 'c, 't>>,

    /// Padding value.
    pub padding: ValueRef<'v, 'c, 't>,

    /// Optional vector mask.
    pub mask: Option<ValueRef<'v, 'c, 't>>,

    /// Result vector type.
    pub result_type: TypeRef<'c, 't>,

    /// Permutation map.
    pub permutation_map: AffineMapAttributeRef<'c, 't>,

    /// Per-dimension in-bounds promises as an array of boolean attributes.
    pub in_bounds: ArrayAttributeRef<'c, 't>,
}

/// Constructs a new detached/owned [`TransferReadOperation`] at the specified [`Location`]. Refer to its documentation
/// for the operation semantics.
pub fn transfer_read<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: TransferReadArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedTransferReadOperation<'c, 't>, Error> {
    validate_vector_type(arguments.result_type, "result", "vector.transfer_read")?;
    validate_indices(&arguments.indices, "vector.transfer_read")?;
    let context = location.context();
    let operand_segment_sizes = context.dense_i32_array_attribute(&[
        1,
        i32::try_from(arguments.indices.len())
            .map_err(|_| Error::invalid_argument("too many vector transfer indices"))?,
        1,
        i32::from(arguments.mask.is_some()),
    ])?;
    let mut operands = vec![arguments.base];
    operands.extend(arguments.indices);
    operands.push(arguments.padding);
    operands.extend(arguments.mask);
    let attributes = [
        ("permutation_map", arguments.permutation_map.as_ref()),
        ("in_bounds", arguments.in_bounds.as_ref()),
        ("operandSegmentSizes", operand_segment_sizes.as_ref()),
    ];
    unsafe { raw_transfer_read(&operands, &[arguments.result_type], &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`TransferReadOperation`] from MLIR operands, types, attributes, and regions. This
/// interface supports interoperability with components not represented by [`transfer_read`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`transfer_read`] for typed construction.
pub unsafe fn raw_transfer_read<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedTransferReadOperation<'c, 't>, Error> {
    if operands.len() < 2 {
        return Err(Error::invalid_argument("invalid operand count for `vector.transfer_read`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.transfer_read`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.transfer_read`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.transfer_read", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::transfer_read`"))
    })
}

/// Vector [`Operation`] that writes a vector to a memory reference or tensor using an affine permutation map and an
/// optional mask. A tensor destination produces an updated tensor result.
///
/// # Example
///
/// The following is an example of a [`TransferWriteOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// vector.transfer_write %arg0, %arg1[%arg2] {in_bounds = [true]} : vector<2xf32>, memref<4xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectortransfer_write-vectortransferwriteop
pub trait TransferWriteOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the value to store operand.
    fn value_to_store(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the base operand.
    fn base(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    vector_attribute_accessor!(required, permutation_map, AffineMapAttributeRef, "permutation_map");
    vector_attribute_accessor!(required, in_bounds, ArrayAttributeRef, "in_bounds");
    vector_attribute_accessor!(required, operand_segment_sizes, DenseInteger32ArrayAttributeRef, "operandSegmentSizes");
}

mlir_op!(TransferWrite);
mlir_op_trait!(TransferWrite, ZeroSuccessors);

/// Typed arguments for [`TransferWriteOperation`].
pub struct TransferWriteArguments<'v, 'c: 'v, 't: 'c> {
    /// Vector value to store.
    pub value_to_store: ValueRef<'v, 'c, 't>,

    /// Destination shaped value.
    pub base: ValueRef<'v, 'c, 't>,

    /// Destination indices.
    pub indices: Vec<ValueRef<'v, 'c, 't>>,

    /// Optional vector mask.
    pub mask: Option<ValueRef<'v, 'c, 't>>,

    /// Optional tensor result type.
    pub result_type: Option<TypeRef<'c, 't>>,

    /// Permutation map.
    pub permutation_map: AffineMapAttributeRef<'c, 't>,

    /// Per-dimension in-bounds promises as an array of boolean attributes.
    pub in_bounds: ArrayAttributeRef<'c, 't>,
}

/// Constructs a new detached/owned [`TransferWriteOperation`] at the specified [`Location`]. Refer to its documentation
/// for the operation semantics.
pub fn transfer_write<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: TransferWriteArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedTransferWriteOperation<'c, 't>, Error> {
    validate_vector_type(arguments.value_to_store.r#type()?, "value operand", "vector.transfer_write")?;
    validate_indices(&arguments.indices, "vector.transfer_write")?;
    let context = location.context();
    let operand_segment_sizes = context.dense_i32_array_attribute(&[
        1,
        1,
        i32::try_from(arguments.indices.len())
            .map_err(|_| Error::invalid_argument("too many vector transfer indices"))?,
        i32::from(arguments.mask.is_some()),
    ])?;
    let mut operands = vec![arguments.value_to_store, arguments.base];
    operands.extend(arguments.indices);
    operands.extend(arguments.mask);
    let result_types = arguments.result_type.into_iter().collect::<Vec<_>>();
    let attributes = [
        ("permutation_map", arguments.permutation_map.as_ref()),
        ("in_bounds", arguments.in_bounds.as_ref()),
        ("operandSegmentSizes", operand_segment_sizes.as_ref()),
    ];
    unsafe { raw_transfer_write(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`TransferWriteOperation`] from MLIR operands, types, attributes, and regions. This
/// interface supports interoperability with components not represented by [`transfer_write`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`transfer_write`] for typed construction.
pub unsafe fn raw_transfer_write<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedTransferWriteOperation<'c, 't>, Error> {
    if operands.len() < 2 {
        return Err(Error::invalid_argument("invalid operand count for `vector.transfer_write`"));
    }
    if !(0..=1).contains(&result_types.len()) {
        return Err(Error::invalid_argument("invalid result count for `vector.transfer_write`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.transfer_write`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.transfer_write", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::transfer_write`"))
    })
}

/// Vector [`Operation`] that loads a vector from a memory reference at the supplied base indices.
///
/// # Example
///
/// The following is an example of a [`LoadOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %0 = vector.load %arg0[%arg1] : memref<4xf32>, vector<2xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorload-vectorloadop
pub trait LoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the base operand.
    fn base(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    vector_attribute_accessor!(optional, non_temporal, BooleanAttributeRef, "nontemporal");
    vector_attribute_accessor!(optional, alignment, IntegerAttributeRef, "alignment");
}

mlir_op!(Load);
mlir_op_trait!(Load, ZeroSuccessors);

/// Typed arguments for [`LoadOperation`].
pub struct LoadArguments<'v, 'c: 'v, 't: 'c> {
    /// Base memory reference or tensor.
    pub base: ValueRef<'v, 'c, 't>,

    /// Base indices in dimension order.
    pub indices: Vec<ValueRef<'v, 'c, 't>>,

    /// Type of the resulting value.
    pub result_type: TypeRef<'c, 't>,

    /// Optional hint that the memory access has low temporal locality.
    pub non_temporal: Option<BooleanAttributeRef<'c, 't>>,

    /// Optional byte alignment, which must be a positive power of two.
    pub alignment: Option<IntegerAttributeRef<'c, 't>>,
}

/// Constructs a new detached/owned [`LoadOperation`] at the specified [`Location`]. Refer to its documentation for the
/// operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`LoadArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn load<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: LoadArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedLoadOperation<'c, 't>, Error> {
    validate_vector_type(arguments.result_type, "result", "vector.load")?;
    validate_indices(&arguments.indices, "vector.load")?;
    if let Some(alignment) = arguments.alignment {
        let alignment = alignment.signless_value();
        if alignment <= 0 || !(alignment as u64).is_power_of_two() {
            return Err(Error::invalid_argument("expected `alignment` of `vector.load` to be a positive power of two"));
        }
    }
    let mut operands = vec![arguments.base];
    operands.extend(arguments.indices);
    let result_types = [arguments.result_type];
    let mut attributes = Vec::new();
    attributes.extend(arguments.non_temporal.map(|attribute| ("nontemporal", attribute.as_ref())));
    attributes.extend(arguments.alignment.map(|attribute| ("alignment", attribute.as_ref())));
    unsafe { raw_load(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`LoadOperation`] from MLIR operands, types, attributes, and regions. This interface
/// supports interoperability with components not represented by [`load`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`load`] for typed construction.
pub unsafe fn raw_load<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedLoadOperation<'c, 't>, Error> {
    if operands.is_empty() {
        return Err(Error::invalid_argument("invalid operand count for `vector.load`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.load`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.load`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.load", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::load`"))
    })
}

/// Vector [`Operation`] that stores a vector to a memory reference at the supplied base indices.
///
/// # Example
///
/// The following is an example of a [`StoreOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// vector.store %arg0, %arg1[%arg2] : memref<4xf32>, vector<2xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorstore-vectorstoreop
pub trait StoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the value to store operand.
    fn value_to_store(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the base operand.
    fn base(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    vector_attribute_accessor!(optional, non_temporal, BooleanAttributeRef, "nontemporal");
    vector_attribute_accessor!(optional, alignment, IntegerAttributeRef, "alignment");
}

mlir_op!(Store);
mlir_op_trait!(Store, ZeroSuccessors);

/// Typed arguments for [`StoreOperation`].
pub struct StoreArguments<'v, 'c: 'v, 't: 'c> {
    /// Value to insert or store.
    pub value_to_store: ValueRef<'v, 'c, 't>,

    /// Base memory reference or tensor.
    pub base: ValueRef<'v, 'c, 't>,

    /// Base indices in dimension order.
    pub indices: Vec<ValueRef<'v, 'c, 't>>,

    /// Optional hint that the memory access has low temporal locality.
    pub non_temporal: Option<BooleanAttributeRef<'c, 't>>,

    /// Optional byte alignment, which must be a positive power of two.
    pub alignment: Option<IntegerAttributeRef<'c, 't>>,
}

/// Constructs a new detached/owned [`StoreOperation`] at the specified [`Location`]. Refer to its documentation for the
/// operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`StoreArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn store<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: StoreArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedStoreOperation<'c, 't>, Error> {
    validate_vector_type(arguments.value_to_store.r#type()?, "value operand", "vector.store")?;
    validate_indices(&arguments.indices, "vector.store")?;
    if let Some(alignment) = arguments.alignment {
        let alignment = alignment.signless_value();
        if alignment <= 0 || !(alignment as u64).is_power_of_two() {
            return Err(Error::invalid_argument(
                "expected `alignment` of `vector.store` to be a positive power of two",
            ));
        }
    }
    let mut operands = vec![arguments.value_to_store, arguments.base];
    operands.extend(arguments.indices);
    let result_types = [];
    let mut attributes = Vec::new();
    attributes.extend(arguments.non_temporal.map(|attribute| ("nontemporal", attribute.as_ref())));
    attributes.extend(arguments.alignment.map(|attribute| ("alignment", attribute.as_ref())));
    unsafe { raw_store(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`StoreOperation`] from MLIR operands, types, attributes, and regions. This interface
/// supports interoperability with components not represented by [`store`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`store`] for typed construction.
pub unsafe fn raw_store<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedStoreOperation<'c, 't>, Error> {
    if operands.len() < 2 {
        return Err(Error::invalid_argument("invalid operand count for `vector.store`"));
    }
    if !result_types.is_empty() {
        return Err(Error::invalid_argument("invalid result count for `vector.store`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.store`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.store", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::store`"))
    })
}

/// Vector [`Operation`] that loads active lanes from memory and takes inactive lanes from the pass-through vector.
///
/// # Example
///
/// The following is an example of a [`MaskedLoadOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %0 = vector.maskedload %arg0[%arg1], %arg2, %arg3 : memref<4xf32>, vector<2xi1>, vector<2xf32> into vector<2xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectormaskedload-vectormaskedloadop
pub trait MaskedLoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the base operand.
    fn base(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    vector_attribute_accessor!(optional, alignment, IntegerAttributeRef, "alignment");
}

mlir_op!(MaskedLoad);
mlir_op_trait!(MaskedLoad, ZeroSuccessors);

/// Typed arguments for [`MaskedLoadOperation`].
pub struct MaskedLoadArguments<'v, 'c: 'v, 't: 'c> {
    /// Base memory reference or tensor.
    pub base: ValueRef<'v, 'c, 't>,

    /// Base indices in dimension order.
    pub indices: Vec<ValueRef<'v, 'c, 't>>,

    /// Boolean vector selecting active lanes.
    pub mask: ValueRef<'v, 'c, 't>,

    /// Values preserved for inactive lanes.
    pub pass_through: ValueRef<'v, 'c, 't>,

    /// Type of the resulting value.
    pub result_type: TypeRef<'c, 't>,

    /// Optional byte alignment, which must be a positive power of two.
    pub alignment: Option<IntegerAttributeRef<'c, 't>>,
}

/// Constructs a new detached/owned [`MaskedLoadOperation`] at the specified [`Location`]. Refer to its documentation
/// for the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`MaskedLoadArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn masked_load<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: MaskedLoadArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedMaskedLoadOperation<'c, 't>, Error> {
    if arguments.pass_through.r#type()? != arguments.result_type {
        return Err(Error::invalid_argument("expected pass-through and result types of `vector.maskedload` to match"));
    }
    validate_vector_type(arguments.mask.r#type()?, "mask operand", "vector.maskedload")?;
    validate_indices(&arguments.indices, "vector.maskedload")?;
    if let Some(alignment) = arguments.alignment {
        let alignment = alignment.signless_value();
        if alignment <= 0 || !(alignment as u64).is_power_of_two() {
            return Err(Error::invalid_argument(
                "expected `alignment` of `vector.maskedload` to be a positive power of two",
            ));
        }
    }
    let mut operands = vec![arguments.base];
    operands.extend(arguments.indices);
    operands.push(arguments.mask);
    operands.push(arguments.pass_through);
    let result_types = [arguments.result_type];
    let mut attributes = Vec::new();
    attributes.extend(arguments.alignment.map(|attribute| ("alignment", attribute.as_ref())));
    unsafe { raw_masked_load(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`MaskedLoadOperation`] from MLIR operands, types, attributes, and regions. This
/// interface supports interoperability with components not represented by [`masked_load`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`masked_load`] for typed construction.
pub unsafe fn raw_masked_load<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedMaskedLoadOperation<'c, 't>, Error> {
    if operands.len() < 3 {
        return Err(Error::invalid_argument("invalid operand count for `vector.maskedload`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.maskedload`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.maskedload`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.maskedload", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::masked_load`"))
    })
}

/// Vector [`Operation`] that stores active lanes of a vector to memory, leaving inactive memory locations unchanged.
///
/// # Example
///
/// The following is an example of a [`MaskedStoreOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// vector.maskedstore %arg0[%arg1], %arg2, %arg3 : memref<4xf32>, vector<2xi1>, vector<2xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectormaskedstore-vectormaskedstoreop
pub trait MaskedStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the base operand.
    fn base(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    vector_attribute_accessor!(optional, alignment, IntegerAttributeRef, "alignment");
}

mlir_op!(MaskedStore);
mlir_op_trait!(MaskedStore, ZeroSuccessors);

/// Typed arguments for [`MaskedStoreOperation`].
pub struct MaskedStoreArguments<'v, 'c: 'v, 't: 'c> {
    /// Base memory reference or tensor.
    pub base: ValueRef<'v, 'c, 't>,

    /// Base indices in dimension order.
    pub indices: Vec<ValueRef<'v, 'c, 't>>,

    /// Boolean vector selecting active lanes.
    pub mask: ValueRef<'v, 'c, 't>,

    /// Value to insert or store.
    pub value_to_store: ValueRef<'v, 'c, 't>,

    /// Optional byte alignment, which must be a positive power of two.
    pub alignment: Option<IntegerAttributeRef<'c, 't>>,
}

/// Constructs a new detached/owned [`MaskedStoreOperation`] at the specified [`Location`]. Refer to its documentation
/// for the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`MaskedStoreArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn masked_store<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: MaskedStoreArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedMaskedStoreOperation<'c, 't>, Error> {
    validate_vector_type(arguments.mask.r#type()?, "mask operand", "vector.maskedstore")?;
    validate_vector_type(arguments.value_to_store.r#type()?, "value operand", "vector.maskedstore")?;
    validate_indices(&arguments.indices, "vector.maskedstore")?;
    if let Some(alignment) = arguments.alignment {
        let alignment = alignment.signless_value();
        if alignment <= 0 || !(alignment as u64).is_power_of_two() {
            return Err(Error::invalid_argument(
                "expected `alignment` of `vector.maskedstore` to be a positive power of two",
            ));
        }
    }
    let mut operands = vec![arguments.base];
    operands.extend(arguments.indices);
    operands.push(arguments.mask);
    operands.push(arguments.value_to_store);
    let result_types = [];
    let mut attributes = Vec::new();
    attributes.extend(arguments.alignment.map(|attribute| ("alignment", attribute.as_ref())));
    unsafe { raw_masked_store(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`MaskedStoreOperation`] from MLIR operands, types, attributes, and regions. This
/// interface supports interoperability with components not represented by [`masked_store`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`masked_store`] for typed construction.
pub unsafe fn raw_masked_store<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedMaskedStoreOperation<'c, 't>, Error> {
    if operands.len() < 3 {
        return Err(Error::invalid_argument("invalid operand count for `vector.maskedstore`"));
    }
    if !result_types.is_empty() {
        return Err(Error::invalid_argument("invalid result count for `vector.maskedstore`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.maskedstore`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.maskedstore", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::masked_store`"))
    })
}

/// Vector [`Operation`] that loads lanes from addresses selected by an index vector relative to the base indices.
/// Inactive lanes take their values from the pass-through vector.
///
/// # Example
///
/// The following is an example of a [`GatherOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %0 = vector.gather %arg0[%arg1] [%arg2], %arg3, %arg4 : memref<8xf32>, vector<2xi32>, vector<2xi1>,
///   vector<2xf32> into vector<2xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorgather-vectorgatherop
pub trait GatherOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the base operand.
    fn base(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    vector_attribute_accessor!(optional, alignment, IntegerAttributeRef, "alignment");
}

mlir_op!(Gather);
mlir_op_trait!(Gather, ZeroSuccessors);

/// Typed arguments for [`GatherOperation`].
pub struct GatherArguments<'v, 'c: 'v, 't: 'c> {
    /// Base memory reference or tensor.
    pub base: ValueRef<'v, 'c, 't>,

    /// Starting offsets in dimension order.
    pub offsets: Vec<ValueRef<'v, 'c, 't>>,

    /// Vector of offsets added to the base indices.
    pub index_vector: ValueRef<'v, 'c, 't>,

    /// Boolean vector selecting active lanes.
    pub mask: ValueRef<'v, 'c, 't>,

    /// Values preserved for inactive lanes.
    pub pass_through: ValueRef<'v, 'c, 't>,

    /// Type of the resulting value.
    pub result_type: TypeRef<'c, 't>,

    /// Optional byte alignment, which must be a positive power of two.
    pub alignment: Option<IntegerAttributeRef<'c, 't>>,
}

/// Constructs a new detached/owned [`GatherOperation`] at the specified [`Location`]. Refer to its documentation for
/// the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`GatherArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn gather<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: GatherArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedGatherOperation<'c, 't>, Error> {
    if arguments.pass_through.r#type()? != arguments.result_type {
        return Err(Error::invalid_argument("expected pass-through and result types of `vector.gather` to match"));
    }
    validate_vector_type(arguments.index_vector.r#type()?, "index-vector operand", "vector.gather")?;
    validate_vector_type(arguments.mask.r#type()?, "mask operand", "vector.gather")?;
    validate_indices(&arguments.offsets, "vector.gather")?;
    if let Some(alignment) = arguments.alignment {
        let alignment = alignment.signless_value();
        if alignment <= 0 || !(alignment as u64).is_power_of_two() {
            return Err(Error::invalid_argument(
                "expected `alignment` of `vector.gather` to be a positive power of two",
            ));
        }
    }
    let mut operands = vec![arguments.base];
    operands.extend(arguments.offsets);
    operands.push(arguments.index_vector);
    operands.push(arguments.mask);
    operands.push(arguments.pass_through);
    let result_types = [arguments.result_type];
    let mut attributes = Vec::new();
    attributes.extend(arguments.alignment.map(|attribute| ("alignment", attribute.as_ref())));
    unsafe { raw_gather(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`GatherOperation`] from MLIR operands, types, attributes, and regions. This interface
/// supports interoperability with components not represented by [`gather`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`gather`] for typed construction.
pub unsafe fn raw_gather<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedGatherOperation<'c, 't>, Error> {
    if operands.len() < 4 {
        return Err(Error::invalid_argument("invalid operand count for `vector.gather`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.gather`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.gather`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.gather", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::gather`"))
    })
}

/// Vector [`Operation`] that stores active lanes at addresses selected by an index vector relative to the base indices.
///
/// # Example
///
/// The following is an example of a [`ScatterOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// vector.scatter %arg0[%arg1] [%arg2], %arg3, %arg4 : memref<8xf32>, vector<2xi32>, vector<2xi1>, vector<2xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorscatter-vectorscatterop
pub trait ScatterOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the base operand.
    fn base(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    vector_attribute_accessor!(optional, alignment, IntegerAttributeRef, "alignment");
}

mlir_op!(Scatter);
mlir_op_trait!(Scatter, ZeroSuccessors);

/// Typed arguments for [`ScatterOperation`].
pub struct ScatterArguments<'v, 'c: 'v, 't: 'c> {
    /// Base memory reference or tensor.
    pub base: ValueRef<'v, 'c, 't>,

    /// Starting offsets in dimension order.
    pub offsets: Vec<ValueRef<'v, 'c, 't>>,

    /// Vector of offsets added to the base indices.
    pub index_vector: ValueRef<'v, 'c, 't>,

    /// Boolean vector selecting active lanes.
    pub mask: ValueRef<'v, 'c, 't>,

    /// Value to insert or store.
    pub value_to_store: ValueRef<'v, 'c, 't>,

    /// Optional type of the resulting value.
    pub result_type: Option<TypeRef<'c, 't>>,

    /// Optional byte alignment, which must be a positive power of two.
    pub alignment: Option<IntegerAttributeRef<'c, 't>>,
}

/// Constructs a new detached/owned [`ScatterOperation`] at the specified [`Location`]. Refer to its documentation for
/// the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`ScatterArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn scatter<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: ScatterArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedScatterOperation<'c, 't>, Error> {
    validate_vector_type(arguments.index_vector.r#type()?, "index-vector operand", "vector.scatter")?;
    validate_vector_type(arguments.mask.r#type()?, "mask operand", "vector.scatter")?;
    validate_vector_type(arguments.value_to_store.r#type()?, "value operand", "vector.scatter")?;
    validate_indices(&arguments.offsets, "vector.scatter")?;
    if let Some(alignment) = arguments.alignment {
        let alignment = alignment.signless_value();
        if alignment <= 0 || !(alignment as u64).is_power_of_two() {
            return Err(Error::invalid_argument(
                "expected `alignment` of `vector.scatter` to be a positive power of two",
            ));
        }
    }
    let mut operands = vec![arguments.base];
    operands.extend(arguments.offsets);
    operands.push(arguments.index_vector);
    operands.push(arguments.mask);
    operands.push(arguments.value_to_store);
    let mut result_types = Vec::new();
    result_types.extend(arguments.result_type);
    let mut attributes = Vec::new();
    attributes.extend(arguments.alignment.map(|attribute| ("alignment", attribute.as_ref())));
    unsafe { raw_scatter(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`ScatterOperation`] from MLIR operands, types, attributes, and regions. This interface
/// supports interoperability with components not represented by [`scatter`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`scatter`] for typed construction.
pub unsafe fn raw_scatter<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedScatterOperation<'c, 't>, Error> {
    if operands.len() < 4 {
        return Err(Error::invalid_argument("invalid operand count for `vector.scatter`"));
    }
    if !(0..=1).contains(&result_types.len()) {
        return Err(Error::invalid_argument("invalid result count for `vector.scatter`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.scatter`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.scatter", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::scatter`"))
    })
}

/// Vector [`Operation`] that reads a contiguous sequence of memory elements into active vector lanes. Inactive lanes
/// use the pass-through values and do not advance the memory position.
///
/// # Example
///
/// The following is an example of a [`ExpandLoadOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %0 = vector.expandload %arg0[%arg1], %arg2, %arg3 : memref<4xf32>, vector<2xi1>, vector<2xf32> into vector<2xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorexpandload-vectorexpandloadop
pub trait ExpandLoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the base operand.
    fn base(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    vector_attribute_accessor!(optional, alignment, IntegerAttributeRef, "alignment");
}

mlir_op!(ExpandLoad);
mlir_op_trait!(ExpandLoad, ZeroSuccessors);

/// Typed arguments for [`ExpandLoadOperation`].
pub struct ExpandLoadArguments<'v, 'c: 'v, 't: 'c> {
    /// Base memory reference or tensor.
    pub base: ValueRef<'v, 'c, 't>,

    /// Base indices in dimension order.
    pub indices: Vec<ValueRef<'v, 'c, 't>>,

    /// Boolean vector selecting active lanes.
    pub mask: ValueRef<'v, 'c, 't>,

    /// Values preserved for inactive lanes.
    pub pass_through: ValueRef<'v, 'c, 't>,

    /// Type of the resulting value.
    pub result_type: TypeRef<'c, 't>,

    /// Optional byte alignment, which must be a positive power of two.
    pub alignment: Option<IntegerAttributeRef<'c, 't>>,
}

/// Constructs a new detached/owned [`ExpandLoadOperation`] at the specified [`Location`]. Refer to its documentation
/// for the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`ExpandLoadArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn expand_load<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: ExpandLoadArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedExpandLoadOperation<'c, 't>, Error> {
    if arguments.pass_through.r#type()? != arguments.result_type {
        return Err(Error::invalid_argument("expected pass-through and result types of `vector.expandload` to match"));
    }
    validate_vector_type(arguments.mask.r#type()?, "mask operand", "vector.expandload")?;
    validate_indices(&arguments.indices, "vector.expandload")?;
    if let Some(alignment) = arguments.alignment {
        let alignment = alignment.signless_value();
        if alignment <= 0 || !(alignment as u64).is_power_of_two() {
            return Err(Error::invalid_argument(
                "expected `alignment` of `vector.expandload` to be a positive power of two",
            ));
        }
    }
    let mut operands = vec![arguments.base];
    operands.extend(arguments.indices);
    operands.push(arguments.mask);
    operands.push(arguments.pass_through);
    let result_types = [arguments.result_type];
    let mut attributes = Vec::new();
    attributes.extend(arguments.alignment.map(|attribute| ("alignment", attribute.as_ref())));
    unsafe { raw_expand_load(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`ExpandLoadOperation`] from MLIR operands, types, attributes, and regions. This
/// interface supports interoperability with components not represented by [`expand_load`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`expand_load`] for typed construction.
pub unsafe fn raw_expand_load<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedExpandLoadOperation<'c, 't>, Error> {
    if operands.len() < 3 {
        return Err(Error::invalid_argument("invalid operand count for `vector.expandload`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.expandload`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.expandload`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.expandload", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::expand_load`"))
    })
}

/// Vector [`Operation`] that writes active vector lanes to consecutive memory locations. Inactive lanes do not advance
/// the memory position.
///
/// # Example
///
/// The following is an example of a [`CompressStoreOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// vector.compressstore %arg0[%arg1], %arg2, %arg3 : memref<4xf32>, vector<2xi1>, vector<2xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorcompressstore-vectorcompressstoreop
pub trait CompressStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the base operand.
    fn base(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    vector_attribute_accessor!(optional, alignment, IntegerAttributeRef, "alignment");
}

mlir_op!(CompressStore);
mlir_op_trait!(CompressStore, ZeroSuccessors);

/// Typed arguments for [`CompressStoreOperation`].
pub struct CompressStoreArguments<'v, 'c: 'v, 't: 'c> {
    /// Base memory reference or tensor.
    pub base: ValueRef<'v, 'c, 't>,

    /// Base indices in dimension order.
    pub indices: Vec<ValueRef<'v, 'c, 't>>,

    /// Boolean vector selecting active lanes.
    pub mask: ValueRef<'v, 'c, 't>,

    /// Value to insert or store.
    pub value_to_store: ValueRef<'v, 'c, 't>,

    /// Optional byte alignment, which must be a positive power of two.
    pub alignment: Option<IntegerAttributeRef<'c, 't>>,
}

/// Constructs a new detached/owned [`CompressStoreOperation`] at the specified [`Location`]. Refer to its documentation
/// for the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`CompressStoreArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn compress_store<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: CompressStoreArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedCompressStoreOperation<'c, 't>, Error> {
    validate_vector_type(arguments.mask.r#type()?, "mask operand", "vector.compressstore")?;
    validate_vector_type(arguments.value_to_store.r#type()?, "value operand", "vector.compressstore")?;
    validate_indices(&arguments.indices, "vector.compressstore")?;
    if let Some(alignment) = arguments.alignment {
        let alignment = alignment.signless_value();
        if alignment <= 0 || !(alignment as u64).is_power_of_two() {
            return Err(Error::invalid_argument(
                "expected `alignment` of `vector.compressstore` to be a positive power of two",
            ));
        }
    }
    let mut operands = vec![arguments.base];
    operands.extend(arguments.indices);
    operands.push(arguments.mask);
    operands.push(arguments.value_to_store);
    let result_types = [];
    let mut attributes = Vec::new();
    attributes.extend(arguments.alignment.map(|attribute| ("alignment", attribute.as_ref())));
    unsafe { raw_compress_store(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`CompressStoreOperation`] from MLIR operands, types, attributes, and regions. This
/// interface supports interoperability with components not represented by [`compress_store`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`compress_store`] for typed construction.
pub unsafe fn raw_compress_store<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedCompressStoreOperation<'c, 't>, Error> {
    if operands.len() < 3 {
        return Err(Error::invalid_argument("invalid operand count for `vector.compressstore`"));
    }
    if !result_types.is_empty() {
        return Err(Error::invalid_argument("invalid result count for `vector.compressstore`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.compressstore`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.compressstore", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::compress_store`"))
    })
}

/// Vector [`Operation`] that reshapes a vector without changing its element type, element count, or linear element
/// order.
///
/// # Example
///
/// The following is an example of a [`ShapeCastOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %0 = vector.shape_cast %arg0 : vector<2x2xf32> to vector<4xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorshape_cast-vectorshapecastop
pub trait ShapeCastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(ShapeCast);
mlir_op_trait!(ShapeCast, ZeroSuccessors);

/// Typed arguments for [`ShapeCastOperation`].
pub struct ShapeCastArguments<'v, 'c: 'v, 't: 'c> {
    /// Source value.
    pub source: ValueRef<'v, 'c, 't>,

    /// Type of the resulting value.
    pub result_type: TypeRef<'c, 't>,
}

/// Constructs a new detached/owned [`ShapeCastOperation`] at the specified [`Location`]. Refer to its documentation for
/// the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`ShapeCastArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn shape_cast<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: ShapeCastArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedShapeCastOperation<'c, 't>, Error> {
    let source_type = validate_vector_type(arguments.source.r#type()?, "source operand", "vector.shape_cast")?;
    let result_type = validate_vector_type(arguments.result_type, "result", "vector.shape_cast")?;
    if source_type.element_type()? != result_type.element_type()? {
        return Err(Error::invalid_argument("expected element types of `vector.shape_cast` to match"));
    }
    let operands = [arguments.source];
    let result_types = [arguments.result_type];
    let attributes = [];
    unsafe { raw_shape_cast(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`ShapeCastOperation`] from MLIR operands, types, attributes, and regions. This interface
/// supports interoperability with components not represented by [`shape_cast`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`shape_cast`] for typed construction.
pub unsafe fn raw_shape_cast<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedShapeCastOperation<'c, 't>, Error> {
    if operands.len() != 1 {
        return Err(Error::invalid_argument("invalid operand count for `vector.shape_cast`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.shape_cast`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.shape_cast`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.shape_cast", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::shape_cast`"))
    })
}

/// Vector [`Operation`] that reinterprets vector bits using a different element type while preserving the total bit
/// width.
///
/// # Example
///
/// The following is an example of a [`BitCastOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %0 = vector.bitcast %arg0 : vector<2xi32> to vector<4xi16>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorbitcast-vectorbitcastop
pub trait BitCastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(BitCast);
mlir_op_trait!(BitCast, ZeroSuccessors);

/// Typed arguments for [`BitCastOperation`].
pub struct BitCastArguments<'v, 'c: 'v, 't: 'c> {
    /// Source value.
    pub source: ValueRef<'v, 'c, 't>,

    /// Type of the resulting value.
    pub result_type: TypeRef<'c, 't>,
}

/// Constructs a new detached/owned [`BitCastOperation`] at the specified [`Location`]. Refer to its documentation for
/// the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`BitCastArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn bit_cast<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: BitCastArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedBitCastOperation<'c, 't>, Error> {
    let source_type = validate_vector_type(arguments.source.r#type()?, "source operand", "vector.bitcast")?;
    let result_type = validate_vector_type(arguments.result_type, "result", "vector.bitcast")?;
    if source_type.rank() != result_type.rank() {
        return Err(Error::invalid_argument("expected source and result ranks of `vector.bitcast` to match"));
    }
    let operands = [arguments.source];
    let result_types = [arguments.result_type];
    let attributes = [];
    unsafe { raw_bit_cast(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`BitCastOperation`] from MLIR operands, types, attributes, and regions. This interface
/// supports interoperability with components not represented by [`bit_cast`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`bit_cast`] for typed construction.
pub unsafe fn raw_bit_cast<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedBitCastOperation<'c, 't>, Error> {
    if operands.len() != 1 {
        return Err(Error::invalid_argument("invalid operand count for `vector.bitcast`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.bitcast`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.bitcast`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.bitcast", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::bit_cast`"))
    })
}

/// Vector [`Operation`] that reinterprets a memory reference as one with vector-valued elements without copying its
/// data.
///
/// # Example
///
/// The following is an example of a [`TypeCastOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %0 = vector.type_cast %arg0 : memref<2x3xf32> to memref<vector<2x3xf32>>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectortype_cast-vectortypecastop
pub trait TypeCastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(TypeCast);
mlir_op_trait!(TypeCast, ZeroSuccessors);

/// Typed arguments for [`TypeCastOperation`].
pub struct TypeCastArguments<'v, 'c: 'v, 't: 'c> {
    /// Source value.
    pub source: ValueRef<'v, 'c, 't>,

    /// Type of the resulting value.
    pub result_type: TypeRef<'c, 't>,
}

/// Constructs a new detached/owned [`TypeCastOperation`] at the specified [`Location`]. Refer to its documentation for
/// the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`TypeCastArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn type_cast<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: TypeCastArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedTypeCastOperation<'c, 't>, Error> {
    let operands = [arguments.source];
    let result_types = [arguments.result_type];
    let attributes = [];
    unsafe { raw_type_cast(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`TypeCastOperation`] from MLIR operands, types, attributes, and regions. This interface
/// supports interoperability with components not represented by [`type_cast`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`type_cast`] for typed construction.
pub unsafe fn raw_type_cast<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedTypeCastOperation<'c, 't>, Error> {
    if operands.len() != 1 {
        return Err(Error::invalid_argument("invalid operand count for `vector.type_cast`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.type_cast`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.type_cast`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.type_cast", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::type_cast`"))
    })
}

/// Vector [`Operation`] that creates a boolean vector whose active lanes form a statically bounded prefix in each
/// dimension.
///
/// # Example
///
/// The following is an example of a [`ConstantMaskOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %0 = vector.constant_mask [2] : vector<4xi1>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorconstant_mask-vectorconstantmaskop
pub trait ConstantMaskOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    vector_attribute_accessor!(required, mask_dimension_sizes, DenseInteger64ArrayAttributeRef, "mask_dim_sizes");
}

mlir_op!(ConstantMask);
mlir_op_trait!(ConstantMask, ZeroSuccessors);

/// Typed arguments for [`ConstantMaskOperation`].
pub struct ConstantMaskArguments<'c, 't: 'c> {
    /// Result vector type.
    pub result_type: TypeRef<'c, 't>,

    /// Static mask dimension sizes.
    pub mask_dimension_sizes: DenseInteger64ArrayAttributeRef<'c, 't>,
}

/// Constructs a new detached/owned [`ConstantMaskOperation`] at the specified [`Location`]. Refer to its documentation
/// for the operation semantics.
pub fn constant_mask<'c, 't: 'c, L: Location<'c, 't>>(
    arguments: ConstantMaskArguments<'c, 't>,
    location: L,
) -> Result<DetachedConstantMaskOperation<'c, 't>, Error> {
    let result_type = validate_vector_type(arguments.result_type, "result", "vector.constant_mask")?;
    let dimensions = result_type.dimensions().collect::<Vec<_>>();
    let mask_sizes = arguments.mask_dimension_sizes.values().collect::<Vec<_>>();
    if mask_sizes.len() != dimensions.len()
        || mask_sizes.iter().zip(dimensions).any(|(mask_size, dimension)| {
            *mask_size < 0
                || match dimension {
                    VectorTypeDimension::Fixed(size) | VectorTypeDimension::Scalable(size) => *mask_size > size as i64,
                }
        })
    {
        return Err(Error::invalid_argument("invalid mask dimension sizes for `vector.constant_mask`"));
    }
    let attributes = [("mask_dim_sizes", arguments.mask_dimension_sizes.as_ref())];
    unsafe { raw_constant_mask(&[], &[arguments.result_type], &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`ConstantMaskOperation`] from MLIR operands, types, attributes, and regions. This
/// interface supports interoperability with components not represented by [`constant_mask`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`constant_mask`] for typed construction.
pub unsafe fn raw_constant_mask<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedConstantMaskOperation<'c, 't>, Error> {
    if !operands.is_empty() {
        return Err(Error::invalid_argument("invalid operand count for `vector.constant_mask`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.constant_mask`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.constant_mask`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.constant_mask", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::constant_mask`"))
    })
}

/// Vector [`Operation`] that creates a boolean vector whose active prefix in each dimension is bounded by a scalar
/// operand.
///
/// # Example
///
/// The following is an example of a [`CreateMaskOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %0 = vector.create_mask %arg0 : vector<4xi1>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorcreate_mask-vectorcreatemaskop
pub trait CreateMaskOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(CreateMask);
mlir_op_trait!(CreateMask, ZeroSuccessors);

/// Typed arguments for [`CreateMaskOperation`].
pub struct CreateMaskArguments<'v, 'c: 'v, 't: 'c> {
    /// Upper bounds on the active prefix in each dimension.
    pub mask_dimension_sizes: Vec<ValueRef<'v, 'c, 't>>,

    /// Type of the resulting value.
    pub result_type: TypeRef<'c, 't>,
}

/// Constructs a new detached/owned [`CreateMaskOperation`] at the specified [`Location`]. Refer to its documentation
/// for the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`CreateMaskArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn create_mask<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: CreateMaskArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedCreateMaskOperation<'c, 't>, Error> {
    let result_type = validate_vector_type(arguments.result_type, "result", "vector.create_mask")?;
    if arguments.mask_dimension_sizes.len() != result_type.rank() {
        return Err(Error::invalid_argument("expected operand count of `vector.create_mask` to equal its result rank"));
    }
    let mut operands = Vec::new();
    operands.extend(arguments.mask_dimension_sizes);
    let result_types = [arguments.result_type];
    let attributes = [];
    unsafe { raw_create_mask(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`CreateMaskOperation`] from MLIR operands, types, attributes, and regions. This
/// interface supports interoperability with components not represented by [`create_mask`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`create_mask`] for typed construction.
pub unsafe fn raw_create_mask<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedCreateMaskOperation<'c, 't>, Error> {
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.create_mask`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.create_mask`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.create_mask", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::create_mask`"))
    })
}

/// Vector [`Operation`] that executes a maskable operation under a boolean vector mask. Its region yields the results,
/// with an optional pass-through value for inactive lanes.
///
/// # Example
///
/// The following is an example of a [`MaskOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = vector.mask %mask { vector.reduction <add>, %input : vector<4xf32> into f32 } : vector<4xi1> -> f32
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectormask-vectormaskop
pub trait MaskOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the vector mask operand.
    fn mask_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the mask body region.
    fn mask_region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }
}

mlir_op!(Mask);
mlir_op_trait!(Mask, ZeroSuccessors);

/// Typed arguments for [`MaskOperation`].
pub struct MaskArguments<'v, 'c: 'v, 't: 'c> {
    /// Boolean vector selecting active lanes.
    pub mask_value: ValueRef<'v, 'c, 't>,

    /// Optional values preserved for inactive lanes.
    pub pass_through: Option<ValueRef<'v, 'c, 't>>,

    /// Result types, in the order yielded by the body.
    pub result_types: Vec<TypeRef<'c, 't>>,

    /// Body containing the masked operation and its yield terminator.
    pub mask_region: DetachedRegion<'c, 't>,
}

/// Constructs a new detached/owned [`MaskOperation`] at the specified [`Location`]. Refer to its documentation for the
/// operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`MaskArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn mask<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: MaskArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedMaskOperation<'c, 't>, Error> {
    let mask_type = validate_vector_type(arguments.mask_value.r#type()?, "mask operand", "vector.mask")?;
    if mask_type.rank() == 0 {
        return Err(Error::invalid_argument("expected `vector.mask` mask operand to have non-zero rank"));
    }
    if let Some(pass_through) = arguments.pass_through
        && (arguments.result_types.len() != 1 || pass_through.r#type()? != arguments.result_types[0])
    {
        return Err(Error::invalid_argument("expected pass-through and result types of `vector.mask` to match"));
    }
    let mut operands = vec![arguments.mask_value];
    operands.extend(arguments.pass_through);
    let mut result_types = Vec::new();
    result_types.extend(arguments.result_types);
    let attributes = [];
    let regions = vec![arguments.mask_region];
    unsafe { raw_mask(&operands, &result_types, &attributes, regions, location) }
}

/// Assembles a detached/owned [`MaskOperation`] from MLIR operands, types, attributes, and regions.
/// This interface supports interoperability with components not represented by [`mask`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native invariants of `vector.mask`.
/// Only component counts are checked here; prefer [`mask`] for typed construction.
pub unsafe fn raw_mask<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedMaskOperation<'c, 't>, Error> {
    if !(1..=2).contains(&operands.len()) {
        return Err(Error::invalid_argument("invalid operand count for `vector.mask`"));
    }
    if regions.len() != 1 {
        return Err(Error::invalid_argument("invalid region count for `vector.mask`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.mask", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::mask`"))
    })
}

/// Vector [`Operation`] that permutes vector dimensions according to a static permutation.
///
/// # Example
///
/// The following is an example of a [`TransposeOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %0 = vector.transpose %arg0, [1, 0] : vector<2x3xf32> to vector<3x2xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectortranspose-vectortransposeop
pub trait TransposeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    vector_attribute_accessor!(required, permutation, DenseInteger64ArrayAttributeRef, "permutation");
}

mlir_op!(Transpose);
mlir_op_trait!(Transpose, ZeroSuccessors);

/// Typed arguments for [`TransposeOperation`].
pub struct TransposeArguments<'v, 'c: 'v, 't: 'c> {
    /// Source value.
    pub source: ValueRef<'v, 'c, 't>,

    /// Type of the resulting value.
    pub result_type: TypeRef<'c, 't>,

    /// Source dimension for each result dimension.
    pub permutation: DenseInteger64ArrayAttributeRef<'c, 't>,
}

/// Constructs a new detached/owned [`TransposeOperation`] at the specified [`Location`]. Refer to its documentation for
/// the operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`TransposeArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn transpose<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: TransposeArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedTransposeOperation<'c, 't>, Error> {
    let source_type = validate_vector_type(arguments.source.r#type()?, "source operand", "vector.transpose")?;
    let result_type = validate_vector_type(arguments.result_type, "result", "vector.transpose")?;
    if source_type.rank() != result_type.rank() || source_type.element_type()? != result_type.element_type()? {
        return Err(Error::invalid_argument("expected compatible source and result types for `vector.transpose`"));
    }
    let mut permutation = arguments.permutation.values().collect::<Vec<_>>();
    if permutation.len() != source_type.rank() {
        return Err(Error::invalid_argument("expected `vector.transpose` permutation length to match its rank"));
    }
    permutation.sort_unstable();
    if !permutation.into_iter().eq(0..source_type.rank() as i64) {
        return Err(Error::invalid_argument("expected `vector.transpose` permutation to contain each dimension"));
    }
    let operands = [arguments.source];
    let result_types = [arguments.result_type];
    let attributes = [("permutation", arguments.permutation.as_ref())];
    unsafe { raw_transpose(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`TransposeOperation`] from MLIR operands, types, attributes, and regions. This interface
/// supports interoperability with components not represented by [`transpose`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`transpose`] for typed construction.
pub unsafe fn raw_transpose<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedTransposeOperation<'c, 't>, Error> {
    if operands.len() != 1 {
        return Err(Error::invalid_argument("invalid operand count for `vector.transpose`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.transpose`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.transpose`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.transpose", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::transpose`"))
    })
}

/// Vector [`Operation`] that prints a scalar or vector value, or a literal string, followed by the selected
/// punctuation.
///
/// # Example
///
/// The following is an example of a [`PrintOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// vector.print %arg0 : vector<2xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorprint-vectorprintop
pub trait PrintOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    vector_attribute_accessor!(default, punctuation, PrintPunctuationAttributeRef, "punctuation");
    vector_attribute_accessor!(optional, string_literal, StringAttributeRef, "stringLiteral");
}

mlir_op!(Print);
mlir_op_trait!(Print, ZeroSuccessors);

/// Typed arguments for [`PrintOperation`].
pub struct PrintArguments<'v, 'c: 'v, 't: 'c> {
    /// Optional source value.
    pub source: Option<ValueRef<'v, 'c, 't>>,

    /// Optional punctuation appended to the output; omission selects a newline.
    pub punctuation: Option<PrintPunctuationAttributeRef<'c, 't>>,

    /// Optional literal text, mutually exclusive with `source`.
    pub string_literal: Option<StringAttributeRef<'c, 't>>,
}

/// Constructs a new detached/owned [`PrintOperation`] at the specified [`Location`]. Refer to its documentation for the
/// operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`PrintArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn print<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: PrintArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedPrintOperation<'c, 't>, Error> {
    if arguments.string_literal.is_some() && arguments.source.is_some() {
        return Err(Error::invalid_argument("`vector.print` cannot print a value and string simultaneously"));
    }
    let mut operands = Vec::new();
    operands.extend(arguments.source);
    let result_types = [];
    let mut attributes = Vec::new();
    attributes.extend(arguments.punctuation.map(|attribute| ("punctuation", attribute.as_ref())));
    attributes.extend(arguments.string_literal.map(|attribute| ("stringLiteral", attribute.as_ref())));
    unsafe { raw_print(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`PrintOperation`] from MLIR operands, types, attributes, and regions. This interface
/// supports interoperability with components not represented by [`print()`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`print()`] for typed construction.
pub unsafe fn raw_print<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedPrintOperation<'c, 't>, Error> {
    if !(0..=1).contains(&operands.len()) {
        return Err(Error::invalid_argument("invalid operand count for `vector.print`"));
    }
    if !result_types.is_empty() {
        return Err(Error::invalid_argument("invalid result count for `vector.print`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.print`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.print", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::print`"))
    })
}

/// Vector [`Operation`] that returns the runtime scale factor used to determine scalable-vector dimensions.
///
/// # Example
///
/// The following is an example of a [`VectorScaleOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering:
///
/// ```mlir
/// %vscale = vector.vscale
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorvscale-vectorvectorscaleop
pub trait VectorScaleOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(VectorScale);
mlir_op_trait!(VectorScale, ZeroSuccessors);

/// Constructs a new detached/owned [`VectorScaleOperation`] at the specified [`Location`]. Refer to its documentation
/// for the operation semantics.
pub fn vector_scale<'c, 't: 'c, L: Location<'c, 't>>(
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedVectorScaleOperation<'c, 't>, Error> {
    unsafe { raw_vector_scale(&[], &[result_type], &[], Vec::new(), location) }
}

/// Assembles a detached/owned [`VectorScaleOperation`] from MLIR operands, types, attributes, and regions. This
/// interface supports interoperability with components not represented by [`vector_scale`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`vector_scale`] for typed construction.
pub unsafe fn raw_vector_scale<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedVectorScaleOperation<'c, 't>, Error> {
    if !operands.is_empty() {
        return Err(Error::invalid_argument("invalid operand count for `vector.vscale`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.vscale`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.vscale`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.vscale", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::vector_scale`"))
    })
}

/// Vector [`Operation`] that computes prefixes along one vector dimension and returns both the vector of prefixes and
/// the final accumulator. The inclusive flag determines whether a prefix includes its current element.
///
/// # Example
///
/// The following is an example of a [`ScanOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %dest, %accumulated_value = vector.scan <add>, %arg0, %arg1 reduction_dim = 0, inclusive = true :
///   vector<2x4xf32>, vector<4xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorscan-vectorscanop
pub trait ScanOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the initial value operand.
    fn initial_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the destination result.
    fn destination(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the accumulated value result.
    fn accumulated_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(1)
    }

    vector_attribute_accessor!(required, kind, CombiningKindAttributeRef, "kind");
    vector_attribute_accessor!(required, reduction_dimension, IntegerAttributeRef, "reduction_dim");
    vector_attribute_accessor!(required, inclusive, BooleanAttributeRef, "inclusive");
}

mlir_op!(Scan);
mlir_op_trait!(Scan, ZeroSuccessors);

/// Typed arguments for [`ScanOperation`].
pub struct ScanArguments<'v, 'c: 'v, 't: 'c> {
    /// Source value.
    pub source: ValueRef<'v, 'c, 't>,

    /// Initial accumulator value for each scan.
    pub initial_value: ValueRef<'v, 'c, 't>,

    /// Type of the vector containing the scan results.
    pub destination_type: TypeRef<'c, 't>,

    /// Type of the final accumulated value.
    pub accumulated_value_type: TypeRef<'c, 't>,

    /// Combining operation used by the reduction or contraction.
    pub kind: CombiningKindAttributeRef<'c, 't>,

    /// Dimension along which to scan.
    pub reduction_dimension: IntegerAttributeRef<'c, 't>,

    /// Whether each prefix includes the element at its own position.
    pub inclusive: BooleanAttributeRef<'c, 't>,
}

/// Constructs a new detached/owned [`ScanOperation`] at the specified [`Location`]. Refer to its documentation for the
/// operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`ScanArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn scan<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: ScanArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedScanOperation<'c, 't>, Error> {
    if arguments.source.r#type()? != arguments.destination_type
        || arguments.initial_value.r#type()? != arguments.accumulated_value_type
    {
        return Err(Error::invalid_argument(
            "expected corresponding operand and result types of `vector.scan` to match",
        ));
    }
    let source_type = validate_vector_type(arguments.source.r#type()?, "source operand", "vector.scan")?;
    let reduction_dimension = arguments.reduction_dimension.signless_value();
    if reduction_dimension < 0 || reduction_dimension as usize >= source_type.rank() {
        return Err(Error::invalid_argument("`reduction_dim` of `vector.scan` is out of bounds"));
    }
    let operands = [arguments.source, arguments.initial_value];
    let result_types = [arguments.destination_type, arguments.accumulated_value_type];
    let attributes = [
        ("kind", arguments.kind.as_ref()),
        ("reduction_dim", arguments.reduction_dimension.as_ref()),
        ("inclusive", arguments.inclusive.as_ref()),
    ];
    unsafe { raw_scan(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`ScanOperation`] from MLIR operands, types, attributes, and regions. This interface
/// supports interoperability with components not represented by [`scan`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`scan`] for typed construction.
pub unsafe fn raw_scan<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedScanOperation<'c, 't>, Error> {
    if operands.len() != 2 {
        return Err(Error::invalid_argument("invalid operand count for `vector.scan`"));
    }
    if result_types.len() != 2 {
        return Err(Error::invalid_argument("invalid result count for `vector.scan`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.scan`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.scan", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::scan`"))
    })
}

/// Vector [`Operation`] that creates a one-dimensional vector containing consecutive indices beginning at zero.
///
/// # Example
///
/// The following is an example of a [`StepOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %0 = vector.step : vector<4xindex>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectorstep-vectorstepop
pub trait StepOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the result value.
    fn result_value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(Step);
mlir_op_trait!(Step, ZeroSuccessors);

/// Constructs a new detached/owned [`StepOperation`] at the specified [`Location`]. Refer to its documentation for the
/// operation semantics.
pub fn step<'c, 't: 'c, L: Location<'c, 't>>(
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedStepOperation<'c, 't>, Error> {
    let result_type = validate_vector_type(result_type, "result", "vector.step")?;
    if result_type.rank() != 1 {
        return Err(Error::invalid_argument("expected `vector.step` result to have rank one"));
    }
    unsafe { raw_step(&[], &[result_type.as_ref()], &[], Vec::new(), location) }
}

/// Assembles a detached/owned [`StepOperation`] from MLIR operands, types, attributes, and regions. This interface
/// supports interoperability with components not represented by [`step`].
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`step`] for typed construction.
pub unsafe fn raw_step<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedStepOperation<'c, 't>, Error> {
    if !operands.is_empty() {
        return Err(Error::invalid_argument("invalid operand count for `vector.step`"));
    }
    if result_types.len() != 1 {
        return Err(Error::invalid_argument("invalid result count for `vector.step`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.step`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.step", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::step`"))
    })
}

/// Vector [`Operation`] that terminates a vector region and yields its values to the enclosing operation.
///
/// # Example
///
/// The following is an example of a [`YieldOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// vector.yield %value : vector<4xf32>
/// ```
///
/// Refer to the [official MLIR documentation][docs] for more information.
///
/// [docs]: https://mlir.llvm.org/docs/Dialects/Vector/#vectoryield-vectoryieldop
pub trait YieldOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Yield);
mlir_op_trait!(Yield, ZeroSuccessors);

/// Typed arguments for [`YieldOperation`].
pub struct YieldArguments<'v, 'c: 'v, 't: 'c> {
    /// Values yielded to the enclosing operation, in result order.
    pub operands: Vec<ValueRef<'v, 'c, 't>>,
}

/// Constructs a new detached/owned [`YieldOperation`] at the specified [`Location`]. Refer to its documentation for the
/// operation semantics.
///
/// # Parameters
///
///   - `arguments`: Operands, result types, and attributes described by [`YieldArguments`].
///   - `location`: Location assigned to the constructed operation.
pub fn r#yield<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: YieldArguments<'v, 'c, 't>,
    location: L,
) -> Result<DetachedYieldOperation<'c, 't>, Error> {
    let mut operands = Vec::new();
    operands.extend(arguments.operands);
    let result_types = [];
    let attributes = [];
    unsafe { raw_yield(&operands, &result_types, &attributes, Vec::new(), location) }
}

/// Assembles a detached/owned [`YieldOperation`] from MLIR operands, types, attributes, and regions. This interface
/// supports interoperability with components not represented by [`r#yield`](fn@yield).
///
/// # Safety
///
/// The caller must ensure that all components satisfy the native MLIR invariants of the operation. Only component
/// counts are checked here; prefer [`r#yield`](fn@yield) for typed construction.
pub unsafe fn raw_yield<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    regions: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedYieldOperation<'c, 't>, Error> {
    if !result_types.is_empty() {
        return Err(Error::invalid_argument("invalid result count for `vector.yield`"));
    }
    if !regions.is_empty() {
        return Err(Error::invalid_argument("invalid region count for `vector.yield`"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::vector()?)?;
    let mut builder = OperationBuilder::new("vector.yield", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_regions(regions);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `vector::r#yield`"))
    })
}

/// Returns the vector type or reports the operand/result role that has the wrong type.
fn validate_vector_type<'c, 't: 'c>(
    r#type: TypeRef<'c, 't>,
    role: &str,
    operation_name: &str,
) -> Result<VectorTypeRef<'c, 't>, Error> {
    r#type
        .cast::<VectorTypeRef>()
        .ok_or_else(|| Error::invalid_argument(format!("expected {role} of `{operation_name}` to have a vector type")))
}

/// Checks every index type, propagating failures to retrieve operand types.
fn validate_indices(indices: &[ValueRef<'_, '_, '_>], operation_name: &str) -> Result<(), Error> {
    for index in indices {
        if !index.r#type()?.is::<IndexTypeRef>() {
            return Err(Error::invalid_argument(format!("expected index operands for `{operation_name}`")));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::arith::attributes::FastMathFlags;
    use crate::dialects::func;
    use crate::dialects::vector::attributes::{CombiningKind, IteratorType, PrintPunctuation};
    use crate::{Attribute, Block, Context, DialectHandle, Operation, Region, Type, Value};

    use super::*;

    #[test]
    fn test_contraction() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let type_1 = context.float32_type().as_ref();
        let indexing_maps = context.array_attribute(&[
            context.affine_map_attribute(context.identity_affine_map(1)),
            context.affine_map_attribute(context.identity_affine_map(1)),
            context.affine_map_attribute(context.zero_result_affine_map(1, 0)),
        ]);
        let iterator_types = context.vector_iterator_type_array_attribute(&[IteratorType::Reduction]).unwrap();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location), (type_0, location), (type_1, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operand_1 = block.argument(1).unwrap().as_ref();
                let operand_2 = block.argument(2).unwrap().as_ref();
                let operation = contraction(
                    ContractionArguments {
                        lhs: operand_0,
                        rhs: operand_1,
                        accumulator: operand_2,
                        result_type: type_1,
                        indexing_maps,
                        iterator_types,
                        kind: None,
                        fastmath: None,
                    },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 3);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.operand_value(1), Ok(operand_1));
                assert_eq!(operation.operand_value(2), Ok(operand_2));
                assert_eq!(operation.lhs().unwrap(), operand_0);
                assert_eq!(operation.rhs().unwrap(), operand_1);
                assert_eq!(operation.accumulator().unwrap(), operand_2);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_1);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_1));
                assert_eq!(operation.indexing_maps(), Ok(indexing_maps));
                assert_eq!(operation.iterator_types(), Ok(iterator_types));
                assert_eq!(operation.fastmath().unwrap().value(), Ok(FastMathFlags::NONE));
                assert_eq!(operation.kind().unwrap().value(), Ok(CombiningKind::Add));
                assert!(matches!(
                    unsafe {
                        raw_contraction(
                            &[operand_0, operand_1, operand_2],
                            &[type_1],
                            &[
                                ("indexing_maps", indexing_maps.as_ref()),
                                ("iterator_types", iterator_types.as_ref()),
                            ],
                            vec![context.region()],
                            location,
                        )
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.contract`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_contraction",
                    func::FuncAttributes {
                        arguments: vec![type_0.into(), type_0.into(), type_1.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            #map = affine_map<(d0) -> (d0)>
            #map1 = affine_map<(d0) -> ()>
            module {
              func.func @test_contraction(%arg0: vector<2xf32>, %arg1: vector<2xf32>, %arg2: f32) {
                %0 = vector.contract {indexing_maps = [#map, #map, #map1], iterator_types = [\"reduction\"], \
                kind = #vector.kind<add>} %arg0, %arg1, %arg2 : vector<2xf32>, vector<2xf32> into f32
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_reduction() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(4)], location)
            .unwrap()
            .as_ref();
        let type_1 = context.float32_type().as_ref();
        let kind = context.vector_combining_kind_attribute(CombiningKind::Add).unwrap();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operation = reduction(
                    ReductionArguments {
                        vector: operand_0,
                        accumulator: None,
                        result_type: type_1,
                        kind,
                        fastmath: None,
                    },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 1);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.vector().unwrap(), operand_0);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_1);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_1));
                assert_eq!(operation.kind(), Ok(kind));
                assert_eq!(operation.fastmath().unwrap().value(), Ok(FastMathFlags::NONE));
                assert_eq!(operation.accumulator(), Ok(None));
                assert!(matches!(
                    unsafe {
                        raw_reduction(
                            &[operand_0],
                            &[type_1],
                            &[("kind", kind.as_ref())],
                            vec![context.region()],
                            location,
                        )
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.reduction`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_reduction",
                    func::FuncAttributes { arguments: vec![type_0.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_reduction(%arg0: vector<4xf32>) {
                %0 = vector.reduction <add>, %arg0 : vector<4xf32> into f32
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_multi_dim_reduction() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .vector_type(
                context.float32_type().as_ref(),
                &[VectorTypeDimension::Fixed(2), VectorTypeDimension::Fixed(2)],
                location,
            )
            .unwrap()
            .as_ref();
        let type_1 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let kind = context.vector_combining_kind_attribute(CombiningKind::Add).unwrap();
        let reduction_dimensions = context.dense_i64_array_attribute(&[0]).unwrap();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location), (type_1, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operand_1 = block.argument(1).unwrap().as_ref();
                let operation = multi_dim_reduction(
                    MultiDimReductionArguments {
                        source: operand_0,
                        accumulator: operand_1,
                        result_type: type_1,
                        kind,
                        reduction_dimensions,
                    },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 2);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.operand_value(1), Ok(operand_1));
                assert_eq!(operation.source().unwrap(), operand_0);
                assert_eq!(operation.accumulator().unwrap(), operand_1);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_1);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_1));
                assert_eq!(operation.kind(), Ok(kind));
                assert_eq!(operation.reduction_dimensions(), Ok(reduction_dimensions));
                assert!(matches!(
                    unsafe {
                        raw_multi_dim_reduction(
                            &[operand_0, operand_1],
                            &[type_1],
                            &[("kind", kind.as_ref()), ("reduction_dims", reduction_dimensions.as_ref())],
                            vec![context.region()],
                            location,
                        )
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.multi_reduction`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_multi_dim_reduction",
                    func::FuncAttributes { arguments: vec![type_0.into(), type_1.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_multi_dim_reduction(%arg0: vector<2x2xf32>, %arg1: vector<2xf32>) {
                %0 = vector.multi_reduction <add>, %arg0, %arg1 [0] : vector<2x2xf32> to vector<2xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_broadcast() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context.float32_type().as_ref();
        let type_1 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(4)], location)
            .unwrap()
            .as_ref();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operation =
                    broadcast(BroadcastArguments { source: operand_0, result_type: type_1 }, location).unwrap();
                assert_eq!(operation.operand_count(), 1);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.source().unwrap(), operand_0);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_1);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_1));
                assert!(matches!(
                    unsafe { raw_broadcast(&[operand_0], &[type_1], &[], vec![context.region()], location) },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.broadcast`",
                ));
                assert!(matches!(
                    broadcast(BroadcastArguments { source: operand_0, result_type: type_0 }, location),
                    Err(Error::InvalidArgument { message, .. })
                        if message == "expected result of `vector.broadcast` to have a vector type",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_broadcast",
                    func::FuncAttributes { arguments: vec![type_0.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_broadcast(%arg0: f32) {
                %0 = vector.broadcast %arg0 : f32 to vector<4xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_shuffle() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(4)], location)
            .unwrap()
            .as_ref();
        let mask = context.dense_i64_array_attribute(&[0, 5, 2, 7]).unwrap();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location), (type_0, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operand_1 = block.argument(1).unwrap().as_ref();
                let operation = shuffle(
                    ShuffleArguments { first: operand_0, second: operand_1, result_type: type_0, mask },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 2);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.operand_value(1), Ok(operand_1));
                assert_eq!(operation.first().unwrap(), operand_0);
                assert_eq!(operation.second().unwrap(), operand_1);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_0);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_0));
                assert_eq!(operation.mask(), Ok(mask));
                assert!(matches!(
                    unsafe {
                        raw_shuffle(
                            &[operand_0, operand_1],
                            &[type_0],
                            &[("mask", mask.as_ref())],
                            vec![context.region()],
                            location,
                        )
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.shuffle`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_shuffle",
                    func::FuncAttributes { arguments: vec![type_0.into(), type_0.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_shuffle(%arg0: vector<4xf32>, %arg1: vector<4xf32>) {
                %0 = vector.shuffle %arg0, %arg1 [0, 5, 2, 7] : vector<4xf32>, vector<4xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_interleave() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let type_1 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(4)], location)
            .unwrap()
            .as_ref();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location), (type_0, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operand_1 = block.argument(1).unwrap().as_ref();
                let operation =
                    interleave(InterleaveArguments { lhs: operand_0, rhs: operand_1, result_type: type_1 }, location)
                        .unwrap();
                assert_eq!(operation.operand_count(), 2);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.operand_value(1), Ok(operand_1));
                assert_eq!(operation.lhs().unwrap(), operand_0);
                assert_eq!(operation.rhs().unwrap(), operand_1);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_1);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_1));
                assert!(matches!(
                    unsafe {
                        raw_interleave(&[operand_0, operand_1], &[type_1], &[], vec![context.region()], location)
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.interleave`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_interleave",
                    func::FuncAttributes { arguments: vec![type_0.into(), type_0.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_interleave(%arg0: vector<2xf32>, %arg1: vector<2xf32>) {
                %0 = vector.interleave %arg0, %arg1 : vector<2xf32> -> vector<4xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_deinterleave() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(4)], location)
            .unwrap()
            .as_ref();
        let type_1 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operation = deinterleave(
                    DeinterleaveArguments { source: operand_0, first_result_type: type_1, second_result_type: type_1 },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 1);
                assert_eq!(operation.result_count(), 2);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.source().unwrap(), operand_0);
                assert_eq!(operation.first_result().unwrap().r#type().unwrap(), type_1);
                assert_eq!(operation.second_result().unwrap().r#type().unwrap(), type_1);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_1));
                assert_eq!(operation.result(1).unwrap().r#type(), Ok(type_1));
                assert!(matches!(
                    unsafe {
                        raw_deinterleave(&[operand_0], &[type_1, type_1], &[], vec![context.region()], location)
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.deinterleave`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_deinterleave",
                    func::FuncAttributes { arguments: vec![type_0.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_deinterleave(%arg0: vector<4xf32>) {
                %res1, %res2 = vector.deinterleave %arg0 : vector<4xf32> -> vector<2xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_extract() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .vector_type(
                context.float32_type().as_ref(),
                &[VectorTypeDimension::Fixed(2), VectorTypeDimension::Fixed(4)],
                location,
            )
            .unwrap()
            .as_ref();
        let type_1 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(4)], location)
            .unwrap()
            .as_ref();
        let static_position = context.dense_i64_array_attribute(&[1]).unwrap();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operation = extract(
                    ExtractArguments {
                        source: operand_0,
                        dynamic_position: vec![],
                        result_type: type_1,
                        static_position,
                    },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 1);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.source().unwrap(), operand_0);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_1);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_1));
                assert_eq!(operation.static_position(), Ok(static_position));
                assert!(matches!(
                    unsafe {
                        raw_extract(
                            &[operand_0],
                            &[type_1],
                            &[("static_position", static_position.as_ref())],
                            vec![context.region()],
                            location,
                        )
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.extract`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_extract",
                    func::FuncAttributes { arguments: vec![type_0.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_extract(%arg0: vector<2x4xf32>) {
                %0 = vector.extract %arg0[1] : vector<4xf32> from vector<2x4xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_extract_dynamic_position() {
        let context = Context::new();
        let location = context.unknown_location();
        let element_type = context.float32_type();
        let vector_type = context.vector_type(element_type, &[VectorTypeDimension::Fixed(4)], location).unwrap();
        let module = context.module(location).unwrap();
        let mut block = context.block(&[(vector_type.as_ref(), location), (context.index_type().as_ref(), location)]);
        let position = context.dense_i64_array_attribute(&[unsafe { crate::Size::Dynamic.to_c_api() }]).unwrap();
        assert!(matches!(
            extract(
                ExtractArguments {
                    source: block.argument(0).unwrap().as_ref(),
                    dynamic_position: vec![],
                    result_type: element_type.as_ref(),
                    static_position: context.dense_i64_array_attribute(&[0, 0]).unwrap(),
                },
                location,
            ),
            Err(Error::InvalidArgument { message, .. })
                if message == "too many position entries for `vector.extract`",
        ));
        let operation = extract(
            ExtractArguments {
                source: block.argument(0).unwrap().as_ref(),
                dynamic_position: vec![block.argument(1).unwrap().as_ref()],
                result_type: element_type.as_ref(),
                static_position: position,
            },
            location,
        )
        .unwrap();
        assert_eq!(operation.source().unwrap(), block.argument(0).unwrap());
        assert_eq!(operation.operand_value(1).unwrap(), block.argument(1).unwrap());
        assert_eq!(operation.static_position(), Ok(position));
        assert_eq!(operation.result_value().unwrap().r#type().unwrap(), element_type);
        let operation = block.append_operation(operation).unwrap();
        block.append_operation(func::r#return(&[operation.result(0).unwrap()], location).unwrap()).unwrap();
        module
            .body()
            .unwrap()
            .append_operation(
                func::func(
                    "test_extract_dynamic_position",
                    func::FuncAttributes {
                        arguments: vec![vector_type.into(), context.index_type().into()],
                        results: vec![element_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_extract_dynamic_position(%arg0: vector<4xf32>, %arg1: index) -> f32 {
                %0 = vector.extract %arg0[%arg1] : f32 from vector<4xf32>
                return %0 : f32
              }
            }
        "}
        );
    }

    #[test]
    fn test_fma() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(4)], location)
            .unwrap()
            .as_ref();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location), (type_0, location), (type_0, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operand_1 = block.argument(1).unwrap().as_ref();
                let operand_2 = block.argument(2).unwrap().as_ref();
                let operation = fma(
                    FmaArguments { lhs: operand_0, rhs: operand_1, accumulator: operand_2, result_type: type_0 },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 3);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.operand_value(1), Ok(operand_1));
                assert_eq!(operation.operand_value(2), Ok(operand_2));
                assert_eq!(operation.lhs().unwrap(), operand_0);
                assert_eq!(operation.rhs().unwrap(), operand_1);
                assert_eq!(operation.accumulator().unwrap(), operand_2);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_0);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_0));
                assert!(matches!(
                    unsafe {
                        raw_fma(
                            &[operand_0, operand_1, operand_2],
                            &[type_0],
                            &[],
                            vec![context.region()],
                            location,
                        )
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.fma`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_fma",
                    func::FuncAttributes {
                        arguments: vec![type_0.into(), type_0.into(), type_0.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_fma(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: vector<4xf32>) {
                %0 = vector.fma %arg0, %arg1, %arg2 : vector<4xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_to_elements() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let type_1 = context.float32_type().as_ref();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operation = to_elements(
                    ToElementsArguments { source: operand_0, element_types: vec![type_1, type_1] },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 1);
                assert_eq!(operation.result_count(), 2);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.source().unwrap(), operand_0);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_1));
                assert_eq!(operation.result(1).unwrap().r#type(), Ok(type_1));
                assert!(matches!(
                    unsafe {
                        raw_to_elements(&[operand_0], &[type_1, type_1], &[], vec![context.region()], location)
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.to_elements`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_to_elements",
                    func::FuncAttributes { arguments: vec![type_0.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_to_elements(%arg0: vector<2xf32>) {
                %0:2 = vector.to_elements %arg0 : vector<2xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_from_elements() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context.float32_type().as_ref();
        let type_1 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location), (type_0, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operand_1 = block.argument(1).unwrap().as_ref();
                let operation = from_elements(
                    FromElementsArguments { elements: vec![operand_0, operand_1], result_type: type_1 },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 2);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.operand_value(1), Ok(operand_1));
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_1);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_1));
                assert!(matches!(
                    unsafe {
                        raw_from_elements(&[operand_0, operand_1], &[type_1], &[], vec![context.region()], location)
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.from_elements`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_from_elements",
                    func::FuncAttributes { arguments: vec![type_0.into(), type_0.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_from_elements(%arg0: f32, %arg1: f32) {
                %0 = vector.from_elements %arg0, %arg1 : vector<2xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_insert() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(4)], location)
            .unwrap()
            .as_ref();
        let type_1 = context
            .vector_type(
                context.float32_type().as_ref(),
                &[VectorTypeDimension::Fixed(2), VectorTypeDimension::Fixed(4)],
                location,
            )
            .unwrap()
            .as_ref();
        let static_position = context.dense_i64_array_attribute(&[1]).unwrap();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location), (type_1, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operand_1 = block.argument(1).unwrap().as_ref();
                let operation = insert(
                    InsertArguments {
                        value_to_store: operand_0,
                        destination: operand_1,
                        dynamic_position: vec![],
                        result_type: type_1,
                        static_position,
                    },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 2);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.operand_value(1), Ok(operand_1));
                assert_eq!(operation.value_to_store().unwrap(), operand_0);
                assert_eq!(operation.destination().unwrap(), operand_1);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_1);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_1));
                assert_eq!(operation.static_position(), Ok(static_position));
                assert!(matches!(
                    unsafe {
                        raw_insert(
                            &[operand_0, operand_1],
                            &[type_1],
                            &[("static_position", static_position.as_ref())],
                            vec![context.region()],
                            location,
                        )
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.insert`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_insert",
                    func::FuncAttributes { arguments: vec![type_0.into(), type_1.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_insert(%arg0: vector<4xf32>, %arg1: vector<2x4xf32>) {
                %0 = vector.insert %arg0, %arg1 [1] : vector<4xf32> into vector<2x4xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_scalable_insert() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let type_1 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Scalable(4)], location)
            .unwrap()
            .as_ref();
        let position = context.integer_attribute(context.signless_integer_type(64), 0);
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location), (type_1, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operand_1 = block.argument(1).unwrap().as_ref();
                let operation = scalable_insert(
                    ScalableInsertArguments {
                        value_to_store: operand_0,
                        destination: operand_1,
                        result_type: type_1,
                        position,
                    },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 2);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.operand_value(1), Ok(operand_1));
                assert_eq!(operation.value_to_store().unwrap(), operand_0);
                assert_eq!(operation.destination().unwrap(), operand_1);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_1);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_1));
                assert_eq!(operation.position(), Ok(position));
                assert!(matches!(
                    unsafe {
                        raw_scalable_insert(
                            &[operand_0, operand_1],
                            &[type_1],
                            &[("pos", position.as_ref())],
                            vec![context.region()],
                            location,
                        )
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.scalable.insert`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_scalable_insert",
                    func::FuncAttributes { arguments: vec![type_0.into(), type_1.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_scalable_insert(%arg0: vector<2xf32>, %arg1: vector<[4]xf32>) {
                %0 = vector.scalable.insert %arg0, %arg1[0] : vector<2xf32> into vector<[4]xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_scalable_extract() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Scalable(4)], location)
            .unwrap()
            .as_ref();
        let type_1 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let position = context.integer_attribute(context.signless_integer_type(64), 0);
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operation = scalable_extract(
                    ScalableExtractArguments { source: operand_0, result_type: type_1, position },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 1);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.source().unwrap(), operand_0);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_1);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_1));
                assert_eq!(operation.position(), Ok(position));
                assert!(matches!(
                    unsafe {
                        raw_scalable_extract(
                            &[operand_0],
                            &[type_1],
                            &[("pos", position.as_ref())],
                            vec![context.region()],
                            location,
                        )
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.scalable.extract`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_scalable_extract",
                    func::FuncAttributes { arguments: vec![type_0.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_scalable_extract(%arg0: vector<[4]xf32>) {
                %0 = vector.scalable.extract %arg0[0] : vector<2xf32> from vector<[4]xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_insert_strided_slice() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let type_1 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(4)], location)
            .unwrap()
            .as_ref();
        let offsets = context.array_attribute(&[context.integer_attribute(context.signless_integer_type(64), 1)]);
        let strides = context.array_attribute(&[context.integer_attribute(context.signless_integer_type(64), 1)]);
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location), (type_1, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operand_1 = block.argument(1).unwrap().as_ref();
                let operation = insert_strided_slice(
                    InsertStridedSliceArguments {
                        value_to_store: operand_0,
                        destination: operand_1,
                        result_type: type_1,
                        offsets,
                        strides,
                    },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 2);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.operand_value(1), Ok(operand_1));
                assert_eq!(operation.value_to_store().unwrap(), operand_0);
                assert_eq!(operation.destination().unwrap(), operand_1);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_1);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_1));
                assert_eq!(operation.offsets(), Ok(offsets));
                assert_eq!(operation.strides(), Ok(strides));
                assert!(matches!(
                    unsafe {
                        raw_insert_strided_slice(
                            &[operand_0, operand_1],
                            &[type_1],
                            &[("offsets", offsets.as_ref()), ("strides", strides.as_ref())],
                            vec![context.region()],
                            location,
                        )
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.insert_strided_slice`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_insert_strided_slice",
                    func::FuncAttributes { arguments: vec![type_0.into(), type_1.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_insert_strided_slice(%arg0: vector<2xf32>, %arg1: vector<4xf32>) {
                %0 = vector.insert_strided_slice %arg0, %arg1 offsets = [1], strides = [1] : vector<2xf32> into \
                vector<4xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_outer_product() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let type_1 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(3)], location)
            .unwrap()
            .as_ref();
        let type_2 = context
            .vector_type(
                context.float32_type().as_ref(),
                &[VectorTypeDimension::Fixed(2), VectorTypeDimension::Fixed(3)],
                location,
            )
            .unwrap()
            .as_ref();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location), (type_1, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operand_1 = block.argument(1).unwrap().as_ref();
                let operation = outer_product(
                    OuterProductArguments {
                        lhs: operand_0,
                        rhs: operand_1,
                        accumulator: None,
                        result_type: type_2,
                        kind: None,
                    },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 2);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.operand_value(1), Ok(operand_1));
                assert_eq!(operation.lhs().unwrap(), operand_0);
                assert_eq!(operation.rhs().unwrap(), operand_1);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_2);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_2));
                assert_eq!(operation.kind().unwrap().value(), Ok(CombiningKind::Add));
                assert!(matches!(
                    unsafe {
                        raw_outer_product(&[operand_0, operand_1], &[type_2], &[], vec![context.region()], location)
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.outerproduct`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_outer_product",
                    func::FuncAttributes { arguments: vec![type_0.into(), type_1.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_outer_product(%arg0: vector<2xf32>, %arg1: vector<3xf32>) {
                %0 = vector.outerproduct %arg0, %arg1 : vector<2xf32>, vector<3xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_extract_strided_slice() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(4)], location)
            .unwrap()
            .as_ref();
        let type_1 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let offsets = context.array_attribute(&[context.integer_attribute(context.signless_integer_type(64), 1)]);
        let sizes = context.array_attribute(&[context.integer_attribute(context.signless_integer_type(64), 2)]);
        let strides = context.array_attribute(&[context.integer_attribute(context.signless_integer_type(64), 1)]);
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operation = extract_strided_slice(
                    ExtractStridedSliceArguments { source: operand_0, result_type: type_1, offsets, sizes, strides },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 1);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.source().unwrap(), operand_0);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_1);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_1));
                assert_eq!(operation.offsets(), Ok(offsets));
                assert_eq!(operation.sizes(), Ok(sizes));
                assert_eq!(operation.strides(), Ok(strides));
                assert!(matches!(
                    unsafe {
                        raw_extract_strided_slice(
                            &[operand_0],
                            &[type_1],
                            &[
                                ("offsets", offsets.as_ref()),
                                ("sizes", sizes.as_ref()),
                                ("strides", strides.as_ref()),
                            ],
                            vec![context.region()],
                            location,
                        )
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.extract_strided_slice`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_extract_strided_slice",
                    func::FuncAttributes { arguments: vec![type_0.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_extract_strided_slice(%arg0: vector<4xf32>) {
                %0 = vector.extract_strided_slice %arg0 offsets = [1], sizes = [2], strides = [1] : \
                vector<4xf32> to vector<2xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_transfer_read() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .contiguous_mem_ref_type(context.float32_type().as_ref(), &[crate::Size::Static(4)], None, location)
            .unwrap()
            .as_ref();
        let type_1 = context.index_type().as_ref();
        let type_2 = context.float32_type().as_ref();
        let type_3 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let permutation_map = context.affine_map_attribute(context.identity_affine_map(1));
        let in_bounds = context.array_attribute(&[context.boolean_attribute(true)]);
        let operand_segment_sizes = context.dense_i32_array_attribute(&[1, 1, 1, 0]).unwrap();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location), (type_1, location), (type_2, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operand_1 = block.argument(1).unwrap().as_ref();
                let operand_2 = block.argument(2).unwrap().as_ref();
                let operation = transfer_read(
                    TransferReadArguments {
                        base: operand_0,
                        indices: vec![operand_1],
                        padding: operand_2,
                        mask: None,
                        result_type: type_3,
                        permutation_map,
                        in_bounds,
                    },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 3);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.operand_value(1), Ok(operand_1));
                assert_eq!(operation.operand_value(2), Ok(operand_2));
                assert_eq!(operation.base().unwrap(), operand_0);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_3);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_3));
                assert_eq!(operation.permutation_map(), Ok(permutation_map));
                assert_eq!(operation.in_bounds(), Ok(in_bounds));
                assert_eq!(operation.operand_segment_sizes(), Ok(operand_segment_sizes));
                assert!(matches!(
                    unsafe {
                        raw_transfer_read(
                            &[operand_0, operand_1, operand_2],
                            &[type_3],
                            &[
                                ("permutation_map", permutation_map.as_ref()),
                                ("in_bounds", in_bounds.as_ref()),
                                ("operandSegmentSizes", operand_segment_sizes.as_ref()),
                            ],
                            vec![context.region()],
                            location,
                        )
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.transfer_read`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_transfer_read",
                    func::FuncAttributes {
                        arguments: vec![type_0.into(), type_1.into(), type_2.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_transfer_read(%arg0: memref<4xf32>, %arg1: index, %arg2: f32) {
                %0 = vector.transfer_read %arg0[%arg1], %arg2 {in_bounds = [true]} : memref<4xf32>, vector<2xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_transfer_write() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let type_1 = context
            .contiguous_mem_ref_type(context.float32_type().as_ref(), &[crate::Size::Static(4)], None, location)
            .unwrap()
            .as_ref();
        let type_2 = context.index_type().as_ref();
        let permutation_map = context.affine_map_attribute(context.identity_affine_map(1));
        let in_bounds = context.array_attribute(&[context.boolean_attribute(true)]);
        let operand_segment_sizes = context.dense_i32_array_attribute(&[1, 1, 1, 0]).unwrap();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location), (type_1, location), (type_2, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operand_1 = block.argument(1).unwrap().as_ref();
                let operand_2 = block.argument(2).unwrap().as_ref();
                let operation = transfer_write(
                    TransferWriteArguments {
                        value_to_store: operand_0,
                        base: operand_1,
                        indices: vec![operand_2],
                        mask: None,
                        result_type: None,
                        permutation_map,
                        in_bounds,
                    },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 3);
                assert_eq!(operation.result_count(), 0);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.operand_value(1), Ok(operand_1));
                assert_eq!(operation.operand_value(2), Ok(operand_2));
                assert_eq!(operation.value_to_store().unwrap(), operand_0);
                assert_eq!(operation.base().unwrap(), operand_1);
                assert_eq!(operation.permutation_map(), Ok(permutation_map));
                assert_eq!(operation.in_bounds(), Ok(in_bounds));
                assert_eq!(operation.operand_segment_sizes(), Ok(operand_segment_sizes));
                assert!(matches!(
                    unsafe {
                        raw_transfer_write(
                            &[operand_0, operand_1, operand_2],
                            &[],
                            &[
                                ("permutation_map", permutation_map.as_ref()),
                                ("in_bounds", in_bounds.as_ref()),
                                ("operandSegmentSizes", operand_segment_sizes.as_ref()),
                            ],
                            vec![context.region()],
                            location,
                        )
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.transfer_write`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_transfer_write",
                    func::FuncAttributes {
                        arguments: vec![type_0.into(), type_1.into(), type_2.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_transfer_write(%arg0: vector<2xf32>, %arg1: memref<4xf32>, %arg2: index) {
                vector.transfer_write %arg0, %arg1[%arg2] {in_bounds = [true]} : vector<2xf32>, memref<4xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_load() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .contiguous_mem_ref_type(context.float32_type().as_ref(), &[crate::Size::Static(4)], None, location)
            .unwrap()
            .as_ref();
        let type_1 = context.index_type().as_ref();
        let type_2 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location), (type_1, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operand_1 = block.argument(1).unwrap().as_ref();
                let operation = load(
                    LoadArguments {
                        base: operand_0,
                        indices: vec![operand_1],
                        result_type: type_2,
                        non_temporal: None,
                        alignment: None,
                    },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 2);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.operand_value(1), Ok(operand_1));
                assert_eq!(operation.base().unwrap(), operand_0);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_2);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_2));
                assert_eq!(operation.non_temporal(), Ok(None));
                assert_eq!(operation.alignment(), Ok(None));
                assert!(matches!(
                    unsafe {
                        raw_load(&[operand_0, operand_1], &[type_2], &[], vec![context.region()], location)
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.load`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_load",
                    func::FuncAttributes { arguments: vec![type_0.into(), type_1.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_load(%arg0: memref<4xf32>, %arg1: index) {
                %0 = vector.load %arg0[%arg1] : memref<4xf32>, vector<2xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_load_invalid_indices_and_alignment() {
        let context = Context::new();
        let location = context.unknown_location();
        let element_type = context.float32_type();
        let memory_type =
            context.contiguous_mem_ref_type(element_type, &[crate::Size::Static(4)], None, location).unwrap();
        let result_type =
            context.vector_type(element_type, &[VectorTypeDimension::Fixed(2)], location).unwrap().as_ref();
        let block = context.block(&[
            (memory_type.as_ref(), location),
            (context.index_type().as_ref(), location),
            (element_type.as_ref(), location),
        ]);
        assert!(matches!(
            load(
                LoadArguments {
                    base: block.argument(0).unwrap().as_ref(),
                    indices: vec![block.argument(2).unwrap().as_ref()],
                    result_type,
                    non_temporal: None,
                    alignment: None,
                },
                location,
            ),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected index operands for `vector.load`",
        ));
        // Reject zero, negative, and non-power-of-two byte alignments.
        for alignment in [0, -1, 3] {
            assert!(matches!(
                load(
                    LoadArguments {
                        base: block.argument(0).unwrap().as_ref(),
                        indices: vec![block.argument(1).unwrap().as_ref()],
                        result_type,
                        non_temporal: None,
                        alignment: Some(
                            context.integer_attribute(context.signless_integer_type(64), alignment),
                        ),
                    },
                    location,
                ),
                Err(Error::InvalidArgument { message, .. })
                    if message == "expected `alignment` of `vector.load` to be a positive power of two",
            ));
        }
    }

    #[test]
    fn test_store() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let type_1 = context
            .contiguous_mem_ref_type(context.float32_type().as_ref(), &[crate::Size::Static(4)], None, location)
            .unwrap()
            .as_ref();
        let type_2 = context.index_type().as_ref();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location), (type_1, location), (type_2, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operand_1 = block.argument(1).unwrap().as_ref();
                let operand_2 = block.argument(2).unwrap().as_ref();
                let operation = store(
                    StoreArguments {
                        value_to_store: operand_0,
                        base: operand_1,
                        indices: vec![operand_2],
                        non_temporal: None,
                        alignment: None,
                    },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 3);
                assert_eq!(operation.result_count(), 0);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.operand_value(1), Ok(operand_1));
                assert_eq!(operation.operand_value(2), Ok(operand_2));
                assert_eq!(operation.value_to_store().unwrap(), operand_0);
                assert_eq!(operation.base().unwrap(), operand_1);
                assert_eq!(operation.non_temporal(), Ok(None));
                assert_eq!(operation.alignment(), Ok(None));
                assert!(matches!(
                    unsafe {
                        raw_store(&[operand_0, operand_1, operand_2], &[], &[], vec![context.region()], location)
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.store`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_store",
                    func::FuncAttributes {
                        arguments: vec![type_0.into(), type_1.into(), type_2.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_store(%arg0: vector<2xf32>, %arg1: memref<4xf32>, %arg2: index) {
                vector.store %arg0, %arg1[%arg2] : memref<4xf32>, vector<2xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_masked_load() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .contiguous_mem_ref_type(context.float32_type().as_ref(), &[crate::Size::Static(4)], None, location)
            .unwrap()
            .as_ref();
        let type_1 = context.index_type().as_ref();
        let type_2 = context
            .vector_type(context.signless_integer_type(1).as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let type_3 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block =
                    context.block(&[(type_0, location), (type_1, location), (type_2, location), (type_3, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operand_1 = block.argument(1).unwrap().as_ref();
                let operand_2 = block.argument(2).unwrap().as_ref();
                let operand_3 = block.argument(3).unwrap().as_ref();
                let operation = masked_load(
                    MaskedLoadArguments {
                        base: operand_0,
                        indices: vec![operand_1],
                        mask: operand_2,
                        pass_through: operand_3,
                        result_type: type_3,
                        alignment: None,
                    },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 4);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.operand_value(1), Ok(operand_1));
                assert_eq!(operation.operand_value(2), Ok(operand_2));
                assert_eq!(operation.operand_value(3), Ok(operand_3));
                assert_eq!(operation.base().unwrap(), operand_0);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_3);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_3));
                assert_eq!(operation.alignment(), Ok(None));
                assert!(matches!(
                    unsafe {
                        raw_masked_load(
                            &[operand_0, operand_1, operand_2, operand_3],
                            &[type_3],
                            &[],
                            vec![context.region()],
                            location,
                        )
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.maskedload`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_masked_load",
                    func::FuncAttributes {
                        arguments: vec![type_0.into(), type_1.into(), type_2.into(), type_3.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_masked_load(%arg0: memref<4xf32>, %arg1: index, %arg2: vector<2xi1>, %arg3: \
              vector<2xf32>) {
                %0 = vector.maskedload %arg0[%arg1], %arg2, %arg3 : memref<4xf32>, vector<2xi1>, vector<2xf32> \
                into vector<2xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_masked_store() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .contiguous_mem_ref_type(context.float32_type().as_ref(), &[crate::Size::Static(4)], None, location)
            .unwrap()
            .as_ref();
        let type_1 = context.index_type().as_ref();
        let type_2 = context
            .vector_type(context.signless_integer_type(1).as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let type_3 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block =
                    context.block(&[(type_0, location), (type_1, location), (type_2, location), (type_3, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operand_1 = block.argument(1).unwrap().as_ref();
                let operand_2 = block.argument(2).unwrap().as_ref();
                let operand_3 = block.argument(3).unwrap().as_ref();
                let operation = masked_store(
                    MaskedStoreArguments {
                        base: operand_0,
                        indices: vec![operand_1],
                        mask: operand_2,
                        value_to_store: operand_3,
                        alignment: None,
                    },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 4);
                assert_eq!(operation.result_count(), 0);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.operand_value(1), Ok(operand_1));
                assert_eq!(operation.operand_value(2), Ok(operand_2));
                assert_eq!(operation.operand_value(3), Ok(operand_3));
                assert_eq!(operation.base().unwrap(), operand_0);
                assert_eq!(operation.alignment(), Ok(None));
                assert!(matches!(
                    unsafe {
                        raw_masked_store(
                            &[operand_0, operand_1, operand_2, operand_3],
                            &[],
                            &[],
                            vec![context.region()],
                            location,
                        )
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.maskedstore`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_masked_store",
                    func::FuncAttributes {
                        arguments: vec![type_0.into(), type_1.into(), type_2.into(), type_3.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_masked_store(%arg0: memref<4xf32>, %arg1: index, %arg2: vector<2xi1>, %arg3: \
              vector<2xf32>) {
                vector.maskedstore %arg0[%arg1], %arg2, %arg3 : memref<4xf32>, vector<2xi1>, vector<2xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_gather() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .contiguous_mem_ref_type(context.float32_type().as_ref(), &[crate::Size::Static(8)], None, location)
            .unwrap()
            .as_ref();
        let type_1 = context.index_type().as_ref();
        let type_2 = context
            .vector_type(context.signless_integer_type(32).as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let type_3 = context
            .vector_type(context.signless_integer_type(1).as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let type_4 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (type_0, location),
                    (type_1, location),
                    (type_2, location),
                    (type_3, location),
                    (type_4, location),
                ]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operand_1 = block.argument(1).unwrap().as_ref();
                let operand_2 = block.argument(2).unwrap().as_ref();
                let operand_3 = block.argument(3).unwrap().as_ref();
                let operand_4 = block.argument(4).unwrap().as_ref();
                let operation = gather(
                    GatherArguments {
                        base: operand_0,
                        offsets: vec![operand_1],
                        index_vector: operand_2,
                        mask: operand_3,
                        pass_through: operand_4,
                        result_type: type_4,
                        alignment: None,
                    },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 5);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.operand_value(1), Ok(operand_1));
                assert_eq!(operation.operand_value(2), Ok(operand_2));
                assert_eq!(operation.operand_value(3), Ok(operand_3));
                assert_eq!(operation.operand_value(4), Ok(operand_4));
                assert_eq!(operation.base().unwrap(), operand_0);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_4);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_4));
                assert_eq!(operation.alignment(), Ok(None));
                assert!(matches!(
                    unsafe {
                        raw_gather(
                            &[operand_0, operand_1, operand_2, operand_3, operand_4],
                            &[type_4],
                            &[],
                            vec![context.region()],
                            location,
                        )
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.gather`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_gather",
                    func::FuncAttributes {
                        arguments: vec![type_0.into(), type_1.into(), type_2.into(), type_3.into(), type_4.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_gather(%arg0: memref<8xf32>, %arg1: index, %arg2: vector<2xi32>, %arg3: \
              vector<2xi1>, %arg4: vector<2xf32>) {
                %0 = vector.gather %arg0[%arg1] [%arg2], %arg3, %arg4 : memref<8xf32>, vector<2xi32>, \
                vector<2xi1>, vector<2xf32> into vector<2xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_scatter() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .contiguous_mem_ref_type(context.float32_type().as_ref(), &[crate::Size::Static(8)], None, location)
            .unwrap()
            .as_ref();
        let type_1 = context.index_type().as_ref();
        let type_2 = context
            .vector_type(context.signless_integer_type(32).as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let type_3 = context
            .vector_type(context.signless_integer_type(1).as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let type_4 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (type_0, location),
                    (type_1, location),
                    (type_2, location),
                    (type_3, location),
                    (type_4, location),
                ]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operand_1 = block.argument(1).unwrap().as_ref();
                let operand_2 = block.argument(2).unwrap().as_ref();
                let operand_3 = block.argument(3).unwrap().as_ref();
                let operand_4 = block.argument(4).unwrap().as_ref();
                let operation = scatter(
                    ScatterArguments {
                        base: operand_0,
                        offsets: vec![operand_1],
                        index_vector: operand_2,
                        mask: operand_3,
                        value_to_store: operand_4,
                        result_type: None,
                        alignment: None,
                    },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 5);
                assert_eq!(operation.result_count(), 0);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.operand_value(1), Ok(operand_1));
                assert_eq!(operation.operand_value(2), Ok(operand_2));
                assert_eq!(operation.operand_value(3), Ok(operand_3));
                assert_eq!(operation.operand_value(4), Ok(operand_4));
                assert_eq!(operation.base().unwrap(), operand_0);
                assert_eq!(operation.alignment(), Ok(None));
                assert!(matches!(
                    unsafe {
                        raw_scatter(
                            &[operand_0, operand_1, operand_2, operand_3, operand_4],
                            &[],
                            &[],
                            vec![context.region()],
                            location,
                        )
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.scatter`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_scatter",
                    func::FuncAttributes {
                        arguments: vec![type_0.into(), type_1.into(), type_2.into(), type_3.into(), type_4.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_scatter(%arg0: memref<8xf32>, %arg1: index, %arg2: vector<2xi32>, %arg3: \
              vector<2xi1>, %arg4: vector<2xf32>) {
                vector.scatter %arg0[%arg1] [%arg2], %arg3, %arg4 : memref<8xf32>, vector<2xi32>, vector<2xi1>, \
                vector<2xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_expand_load() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .contiguous_mem_ref_type(context.float32_type().as_ref(), &[crate::Size::Static(4)], None, location)
            .unwrap()
            .as_ref();
        let type_1 = context.index_type().as_ref();
        let type_2 = context
            .vector_type(context.signless_integer_type(1).as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let type_3 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block =
                    context.block(&[(type_0, location), (type_1, location), (type_2, location), (type_3, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operand_1 = block.argument(1).unwrap().as_ref();
                let operand_2 = block.argument(2).unwrap().as_ref();
                let operand_3 = block.argument(3).unwrap().as_ref();
                let operation = expand_load(
                    ExpandLoadArguments {
                        base: operand_0,
                        indices: vec![operand_1],
                        mask: operand_2,
                        pass_through: operand_3,
                        result_type: type_3,
                        alignment: None,
                    },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 4);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.operand_value(1), Ok(operand_1));
                assert_eq!(operation.operand_value(2), Ok(operand_2));
                assert_eq!(operation.operand_value(3), Ok(operand_3));
                assert_eq!(operation.base().unwrap(), operand_0);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_3);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_3));
                assert_eq!(operation.alignment(), Ok(None));
                assert!(matches!(
                    unsafe {
                        raw_expand_load(
                            &[operand_0, operand_1, operand_2, operand_3],
                            &[type_3],
                            &[],
                            vec![context.region()],
                            location,
                        )
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.expandload`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_expand_load",
                    func::FuncAttributes {
                        arguments: vec![type_0.into(), type_1.into(), type_2.into(), type_3.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_expand_load(%arg0: memref<4xf32>, %arg1: index, %arg2: vector<2xi1>, %arg3: \
              vector<2xf32>) {
                %0 = vector.expandload %arg0[%arg1], %arg2, %arg3 : memref<4xf32>, vector<2xi1>, vector<2xf32> \
                into vector<2xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_compress_store() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .contiguous_mem_ref_type(context.float32_type().as_ref(), &[crate::Size::Static(4)], None, location)
            .unwrap()
            .as_ref();
        let type_1 = context.index_type().as_ref();
        let type_2 = context
            .vector_type(context.signless_integer_type(1).as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let type_3 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block =
                    context.block(&[(type_0, location), (type_1, location), (type_2, location), (type_3, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operand_1 = block.argument(1).unwrap().as_ref();
                let operand_2 = block.argument(2).unwrap().as_ref();
                let operand_3 = block.argument(3).unwrap().as_ref();
                let operation = compress_store(
                    CompressStoreArguments {
                        base: operand_0,
                        indices: vec![operand_1],
                        mask: operand_2,
                        value_to_store: operand_3,
                        alignment: None,
                    },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 4);
                assert_eq!(operation.result_count(), 0);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.operand_value(1), Ok(operand_1));
                assert_eq!(operation.operand_value(2), Ok(operand_2));
                assert_eq!(operation.operand_value(3), Ok(operand_3));
                assert_eq!(operation.base().unwrap(), operand_0);
                assert_eq!(operation.alignment(), Ok(None));
                assert!(matches!(
                    unsafe {
                        raw_compress_store(
                            &[operand_0, operand_1, operand_2, operand_3],
                            &[],
                            &[],
                            vec![context.region()],
                            location,
                        )
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.compressstore`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_compress_store",
                    func::FuncAttributes {
                        arguments: vec![type_0.into(), type_1.into(), type_2.into(), type_3.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_compress_store(%arg0: memref<4xf32>, %arg1: index, %arg2: vector<2xi1>, %arg3: \
              vector<2xf32>) {
                vector.compressstore %arg0[%arg1], %arg2, %arg3 : memref<4xf32>, vector<2xi1>, vector<2xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_shape_cast() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .vector_type(
                context.float32_type().as_ref(),
                &[VectorTypeDimension::Fixed(2), VectorTypeDimension::Fixed(2)],
                location,
            )
            .unwrap()
            .as_ref();
        let type_1 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(4)], location)
            .unwrap()
            .as_ref();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operation =
                    shape_cast(ShapeCastArguments { source: operand_0, result_type: type_1 }, location).unwrap();
                assert_eq!(operation.operand_count(), 1);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.source().unwrap(), operand_0);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_1);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_1));
                assert!(matches!(
                    unsafe { raw_shape_cast(&[operand_0], &[type_1], &[], vec![context.region()], location) },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.shape_cast`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_shape_cast",
                    func::FuncAttributes { arguments: vec![type_0.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_shape_cast(%arg0: vector<2x2xf32>) {
                %0 = vector.shape_cast %arg0 : vector<2x2xf32> to vector<4xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_bit_cast() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .vector_type(context.signless_integer_type(32).as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let type_1 = context
            .vector_type(context.signless_integer_type(16).as_ref(), &[VectorTypeDimension::Fixed(4)], location)
            .unwrap()
            .as_ref();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operation =
                    bit_cast(BitCastArguments { source: operand_0, result_type: type_1 }, location).unwrap();
                assert_eq!(operation.operand_count(), 1);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.source().unwrap(), operand_0);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_1);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_1));
                assert!(matches!(
                    unsafe { raw_bit_cast(&[operand_0], &[type_1], &[], vec![context.region()], location) },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.bitcast`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_bit_cast",
                    func::FuncAttributes { arguments: vec![type_0.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_bit_cast(%arg0: vector<2xi32>) {
                %0 = vector.bitcast %arg0 : vector<2xi32> to vector<4xi16>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_type_cast() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .contiguous_mem_ref_type(
                context.float32_type().as_ref(),
                &[crate::Size::Static(2), crate::Size::Static(3)],
                None,
                location,
            )
            .unwrap()
            .as_ref();
        let type_1 = context
            .contiguous_mem_ref_type(
                context
                    .vector_type(
                        context.float32_type().as_ref(),
                        &[VectorTypeDimension::Fixed(2), VectorTypeDimension::Fixed(3)],
                        location,
                    )
                    .unwrap()
                    .as_ref(),
                &[],
                None,
                location,
            )
            .unwrap()
            .as_ref();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operation =
                    type_cast(TypeCastArguments { source: operand_0, result_type: type_1 }, location).unwrap();
                assert_eq!(operation.operand_count(), 1);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.source().unwrap(), operand_0);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_1);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_1));
                assert!(matches!(
                    unsafe { raw_type_cast(&[operand_0], &[type_1], &[], vec![context.region()], location) },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.type_cast`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_type_cast",
                    func::FuncAttributes { arguments: vec![type_0.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_type_cast(%arg0: memref<2x3xf32>) {
                %0 = vector.type_cast %arg0 : memref<2x3xf32> to memref<vector<2x3xf32>>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_constant_mask() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .vector_type(context.signless_integer_type(1).as_ref(), &[VectorTypeDimension::Fixed(4)], location)
            .unwrap()
            .as_ref();
        let mask_dimension_sizes = context.dense_i64_array_attribute(&[2]).unwrap();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let operation =
                    constant_mask(ConstantMaskArguments { result_type: type_0, mask_dimension_sizes }, location)
                        .unwrap();
                assert_eq!(operation.operand_count(), 0);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_0);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_0));
                assert_eq!(operation.mask_dimension_sizes(), Ok(mask_dimension_sizes));
                assert!(matches!(
                    unsafe {
                        raw_constant_mask(
                            &[],
                            &[type_0],
                            &[("mask_dim_sizes", mask_dimension_sizes.as_ref())],
                            vec![context.region()],
                            location,
                        )
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.constant_mask`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_constant_mask",
                    func::FuncAttributes { arguments: vec![], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_constant_mask() {
                %0 = vector.constant_mask [2] : vector<4xi1>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_create_mask() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context.index_type().as_ref();
        let type_1 = context
            .vector_type(context.signless_integer_type(1).as_ref(), &[VectorTypeDimension::Fixed(4)], location)
            .unwrap()
            .as_ref();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operation = create_mask(
                    CreateMaskArguments { mask_dimension_sizes: vec![operand_0], result_type: type_1 },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 1);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_1);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_1));
                assert!(matches!(
                    unsafe { raw_create_mask(&[operand_0], &[type_1], &[], vec![context.region()], location) },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.create_mask`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_create_mask",
                    func::FuncAttributes { arguments: vec![type_0.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_create_mask(%arg0: index) {
                %0 = vector.create_mask %arg0 : vector<4xi1>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_mask() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_type = context.parse_type("vector<4xf32>").unwrap();
        let result_type = context.float32_type().as_ref();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(mask_type, location), (vector_type, location)]);
                let mut mask_block = context.block_with_no_arguments();
                let reduction_operation = reduction(
                    ReductionArguments {
                        vector: block.argument(1).unwrap().as_ref(),
                        accumulator: None,
                        result_type,
                        kind: context.vector_combining_kind_attribute(CombiningKind::Add).unwrap(),
                        fastmath: None,
                    },
                    location,
                )
                .unwrap();
                let reduction_operation = mask_block.append_operation(reduction_operation).unwrap();
                mask_block
                    .append_operation(
                        r#yield(
                            YieldArguments { operands: vec![reduction_operation.result(0).unwrap().as_ref()] },
                            location,
                        )
                        .unwrap(),
                    )
                    .unwrap();
                let operation = mask(
                    MaskArguments {
                        mask_value: block.argument(0).unwrap().as_ref(),
                        pass_through: None,
                        result_types: vec![result_type],
                        mask_region: mask_block.try_into().unwrap(),
                    },
                    location,
                )
                .unwrap();
                assert!(matches!(
                    unsafe {
                        raw_mask(&[block.argument(0).unwrap().as_ref()], &[result_type], &[], Vec::new(), location)
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.mask`",
                ));
                assert_eq!(operation.mask_value().unwrap(), block.argument(0).unwrap());
                assert_eq!(operation.mask_region().unwrap().blocks().unwrap().count(), 1);
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "mask",
                    func::FuncAttributes {
                        arguments: vec![mask_type.into(), vector_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @mask(%arg0: vector<4xi1>, %arg1: vector<4xf32>) {
                %0 = vector.mask %arg0 { vector.reduction <add>, %arg1 : vector<4xf32> into f32 } : vector<4xi1> -> f32
                return
              }
            }
        "},
        );
    }
    #[test]
    fn test_transpose() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .vector_type(
                context.float32_type().as_ref(),
                &[VectorTypeDimension::Fixed(2), VectorTypeDimension::Fixed(3)],
                location,
            )
            .unwrap()
            .as_ref();
        let type_1 = context
            .vector_type(
                context.float32_type().as_ref(),
                &[VectorTypeDimension::Fixed(3), VectorTypeDimension::Fixed(2)],
                location,
            )
            .unwrap()
            .as_ref();
        let permutation = context.dense_i64_array_attribute(&[1, 0]).unwrap();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operation =
                    transpose(TransposeArguments { source: operand_0, result_type: type_1, permutation }, location)
                        .unwrap();
                assert_eq!(operation.operand_count(), 1);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.source().unwrap(), operand_0);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_1);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_1));
                assert_eq!(operation.permutation(), Ok(permutation));
                assert!(matches!(
                    unsafe {
                        raw_transpose(
                            &[operand_0],
                            &[type_1],
                            &[("permutation", permutation.as_ref())],
                            vec![context.region()],
                            location,
                        )
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.transpose`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_transpose",
                    func::FuncAttributes { arguments: vec![type_0.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_transpose(%arg0: vector<2x3xf32>) {
                %0 = vector.transpose %arg0, [1, 0] : vector<2x3xf32> to vector<3x2xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_print() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(2)], location)
            .unwrap()
            .as_ref();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operation = print(
                    PrintArguments { source: Some(operand_0), punctuation: None, string_literal: None },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 1);
                assert_eq!(operation.result_count(), 0);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.punctuation().unwrap().value(), Ok(PrintPunctuation::NewLine));
                assert_eq!(operation.string_literal(), Ok(None));
                assert!(matches!(
                    unsafe { raw_print(&[operand_0], &[], &[], vec![context.region()], location) },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.print`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_print",
                    func::FuncAttributes { arguments: vec![type_0.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_print(%arg0: vector<2xf32>) {
                vector.print %arg0 : vector<2xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_vector_scale() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context.index_type().as_ref();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let operation = vector_scale(type_0, location).unwrap();
                assert_eq!(operation.operand_count(), 0);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_0);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_0));
                assert!(matches!(
                    unsafe { raw_vector_scale(&[], &[type_0], &[], vec![context.region()], location) },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.vscale`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_vector_scale",
                    func::FuncAttributes { arguments: vec![], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_vector_scale() {
                %vscale = vector.vscale
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_scan() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .vector_type(
                context.float32_type().as_ref(),
                &[VectorTypeDimension::Fixed(2), VectorTypeDimension::Fixed(4)],
                location,
            )
            .unwrap()
            .as_ref();
        let type_1 = context
            .vector_type(context.float32_type().as_ref(), &[VectorTypeDimension::Fixed(4)], location)
            .unwrap()
            .as_ref();
        let kind = context.vector_combining_kind_attribute(CombiningKind::Add).unwrap();
        let reduction_dimension = context.integer_attribute(context.signless_integer_type(64), 0);
        let inclusive = context.boolean_attribute(true);
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(type_0, location), (type_1, location)]);
                let operand_0 = block.argument(0).unwrap().as_ref();
                let operand_1 = block.argument(1).unwrap().as_ref();
                let operation = scan(
                    ScanArguments {
                        source: operand_0,
                        initial_value: operand_1,
                        destination_type: type_0,
                        accumulated_value_type: type_1,
                        kind,
                        reduction_dimension,
                        inclusive,
                    },
                    location,
                )
                .unwrap();
                assert_eq!(operation.operand_count(), 2);
                assert_eq!(operation.result_count(), 2);
                assert_eq!(operation.operand_value(0), Ok(operand_0));
                assert_eq!(operation.operand_value(1), Ok(operand_1));
                assert_eq!(operation.source().unwrap(), operand_0);
                assert_eq!(operation.initial_value().unwrap(), operand_1);
                assert_eq!(operation.destination().unwrap().r#type().unwrap(), type_0);
                assert_eq!(operation.accumulated_value().unwrap().r#type().unwrap(), type_1);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_0));
                assert_eq!(operation.result(1).unwrap().r#type(), Ok(type_1));
                assert_eq!(operation.kind(), Ok(kind));
                assert_eq!(operation.reduction_dimension(), Ok(reduction_dimension));
                assert_eq!(operation.inclusive(), Ok(inclusive));
                assert!(matches!(
                    unsafe {
                        raw_scan(
                            &[operand_0, operand_1],
                            &[type_0, type_1],
                            &[
                                ("kind", kind.as_ref()),
                                ("reduction_dim", reduction_dimension.as_ref()),
                                ("inclusive", inclusive.as_ref()),
                            ],
                            vec![context.region()],
                            location,
                        )
                    },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.scan`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_scan",
                    func::FuncAttributes { arguments: vec![type_0.into(), type_1.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_scan(%arg0: vector<2x4xf32>, %arg1: vector<4xf32>) {
                %dest, %accumulated_value = vector.scan <add>, %arg0, %arg1 reduction_dim = 0, inclusive = true \
                : vector<2x4xf32>, vector<4xf32>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_step() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let type_0 = context
            .vector_type(context.index_type().as_ref(), &[VectorTypeDimension::Fixed(4)], location)
            .unwrap()
            .as_ref();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let operation = step(type_0, location).unwrap();
                assert_eq!(operation.operand_count(), 0);
                assert_eq!(operation.result_count(), 1);
                assert_eq!(operation.result_value().unwrap().r#type().unwrap(), type_0);
                assert_eq!(operation.result(0).unwrap().r#type(), Ok(type_0));
                assert!(matches!(
                    unsafe { raw_step(&[], &[type_0], &[], vec![context.region()], location) },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.step`",
                ));
                block.append_operation(operation).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "test_step",
                    func::FuncAttributes { arguments: vec![], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @test_step() {
                %0 = vector.step : vector<4xindex>
                return
              }
            }
        "}
        );
    }

    #[test]
    fn test_yield() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::vector().unwrap()).unwrap();
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(mask_type, location)]);
                let mut mask_block = context.block_with_no_arguments();
                let operation = r#yield(YieldArguments { operands: Vec::new() }, location).unwrap();
                assert_eq!(operation.operand_count(), 0);
                assert!(matches!(
                    unsafe { raw_yield(&[], &[], &[], vec![context.region()], location) },
                    Err(Error::InvalidArgument { message, .. })
                        if message == "invalid region count for `vector.yield`",
                ));
                mask_block.append_operation(operation).unwrap();
                block
                    .append_operation(
                        mask(
                            MaskArguments {
                                mask_value: block.argument(0).unwrap().as_ref(),
                                pass_through: None,
                                result_types: Vec::new(),
                                mask_region: mask_block.try_into().unwrap(),
                            },
                            location,
                        )
                        .unwrap(),
                    )
                    .unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "yield",
                    func::FuncAttributes { arguments: vec![mask_type.into()], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert_eq!(module.verify(), Ok(true));
        assert_eq!(
            module.to_string(),
            indoc! {"
            module {
              func.func @yield(%arg0: vector<4xi1>) {
                vector.mask %arg0 { vector.yield } : vector<4xi1>
                return
              }
            }
        "},
        );
    }
}
