use crate::{
    ArrayAttributeRef, Attribute, AttributeRef, BooleanAttributeRef, DenseInteger32ArrayAttributeRef, DetachedOp,
    DetachedRegion, DialectHandle, IntegerAttributeRef, Location, Operation, OperationBuilder, RegionRef,
    SYMBOL_NAME_ATTRIBUTE, StringAttributeRef, StringRef, Type, TypeAttributeRef, TypeRef, ValueRef, mlir_op,
    mlir_op_trait,
};

/// Name of the native PDL function attribute.
pub const NAME_ATTRIBUTE: &str = "name";

/// Name of the `pdl.apply_native_constraint` negation attribute.
pub const IS_NEGATED_ATTRIBUTE: &str = "isNegated";

/// Operation trait for `pdl.apply_native_constraint`.
pub trait ApplyNativeConstraintOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the name of the native constraint function to apply.
    fn native_constraint_name(&self) -> StringRef<'c> {
        self.attribute(NAME_ATTRIBUTE).unwrap().cast::<StringAttributeRef>().unwrap().string()
    }

    /// Returns the PDL entities passed to the native constraint.
    fn arguments(&self) -> impl Iterator<Item = ValueRef<'o, 'c, 't>> {
        self.operand_values()
    }

    /// Returns whether the native constraint result is negated.
    fn is_negated(&self) -> bool {
        self.attribute(IS_NEGATED_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<BooleanAttributeRef>())
            .map(|attribute| attribute.value())
            .unwrap_or(false)
    }
}

mlir_op!(ApplyNativeConstraint);
mlir_op_trait!(ApplyNativeConstraint, ZeroRegions);
mlir_op_trait!(ApplyNativeConstraint, ZeroSuccessors);

/// Constructs a new detached/owned [`ApplyNativeConstraintOperation`] at the specified [`Location`].
pub fn apply_native_constraint<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    name: StringAttributeRef<'c, 't>,
    arguments: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    is_negated: bool,
    location: L,
) -> DetachedApplyNativeConstraintOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::pdl());
    let mut builder = OperationBuilder::new("pdl.apply_native_constraint", location)
        .add_attribute(NAME_ATTRIBUTE, name)
        .add_operands(arguments)
        .add_results(result_types);
    if is_negated {
        builder = builder.add_attribute(IS_NEGATED_ATTRIBUTE, context.boolean_attribute(true));
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `pdl::apply_native_constraint`")
}

/// Operation trait for `pdl.apply_native_rewrite`.
pub trait ApplyNativeRewriteOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the name of the native rewrite function to apply.
    fn native_rewrite_name(&self) -> StringRef<'c> {
        self.attribute(NAME_ATTRIBUTE).unwrap().cast::<StringAttributeRef>().unwrap().string()
    }

    /// Returns the PDL entities passed to the native rewrite.
    fn arguments(&self) -> impl Iterator<Item = ValueRef<'o, 'c, 't>> {
        self.operand_values()
    }
}

mlir_op!(ApplyNativeRewrite);
mlir_op_trait!(ApplyNativeRewrite, ZeroRegions);
mlir_op_trait!(ApplyNativeRewrite, ZeroSuccessors);

/// Constructs a new detached/owned [`ApplyNativeRewriteOperation`] at the specified [`Location`].
pub fn apply_native_rewrite<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    name: StringAttributeRef<'c, 't>,
    arguments: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    location: L,
) -> DetachedApplyNativeRewriteOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::pdl());
    OperationBuilder::new("pdl.apply_native_rewrite", location)
        .add_attribute(NAME_ATTRIBUTE, name)
        .add_operands(arguments)
        .add_results(result_types)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `pdl::apply_native_rewrite`")
}

/// Name of the constant value attribute used by `pdl.attribute`.
pub const VALUE_ATTRIBUTE: &str = "value";

/// Operation trait for `pdl.attribute`.
pub trait AttributeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the optional expected type handle for this attribute.
    fn value_type(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.operand_value(0)
    }

    /// Returns the optional constant attribute value.
    fn value(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute(VALUE_ATTRIBUTE)
    }
}

mlir_op!(Attribute);
mlir_op_trait!(Attribute, OneResult);
mlir_op_trait!(Attribute, ZeroRegions);
mlir_op_trait!(Attribute, ZeroSuccessors);

/// Constructs a new detached/owned [`AttributeOperation`] at the specified [`Location`].
pub fn attribute<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    value_type: Option<ValueRef<'v, 'c, 't>>,
    value: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedAttributeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::pdl());
    let mut builder = OperationBuilder::new("pdl.attribute", location).add_result(context.pdl_attribute_type());
    if let Some(value_type) = value_type {
        builder = builder.add_operand(value_type);
    }
    if let Some(value) = value {
        builder = builder.add_attribute(VALUE_ATTRIBUTE, value);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `pdl::attribute`")
}

/// Operation trait for `pdl.erase`.
pub trait EraseOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the operation handle to erase.
    fn operation(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }
}

mlir_op!(Erase);
mlir_op_trait!(Erase, ZeroRegions);
mlir_op_trait!(Erase, ZeroSuccessors);

/// Constructs a new detached/owned [`EraseOperation`] at the specified [`Location`].
pub fn erase<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operation: ValueRef<'v, 'c, 't>,
    location: L,
) -> DetachedEraseOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::pdl());
    OperationBuilder::new("pdl.erase", location)
        .add_operand(operation)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `pdl::erase`")
}

/// Operation trait for `pdl.operand`.
pub trait OperandOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the optional expected type handle for this operand.
    fn value_type(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.operand_value(0)
    }
}

mlir_op!(Operand);
mlir_op_trait!(Operand, OneResult);
mlir_op_trait!(Operand, ZeroRegions);
mlir_op_trait!(Operand, ZeroSuccessors);

/// Constructs a new detached/owned [`OperandOperation`] at the specified [`Location`].
pub fn operand<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    value_type: Option<ValueRef<'v, 'c, 't>>,
    location: L,
) -> DetachedOperandOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::pdl());
    let mut builder = OperationBuilder::new("pdl.operand", location).add_result(context.pdl_value_type());
    if let Some(value_type) = value_type {
        builder = builder.add_operand(value_type);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `pdl::operand`")
}

/// Operation trait for `pdl.operands`.
pub trait OperandsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the optional expected range-of-types handle for these operands.
    fn value_type(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.operand_value(0)
    }
}

mlir_op!(Operands);
mlir_op_trait!(Operands, OneResult);
mlir_op_trait!(Operands, ZeroRegions);
mlir_op_trait!(Operands, ZeroSuccessors);

/// Constructs a new detached/owned [`OperandsOperation`] at the specified [`Location`].
pub fn operands<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    value_type: Option<ValueRef<'v, 'c, 't>>,
    location: L,
) -> DetachedOperandsOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::pdl());
    let mut builder =
        OperationBuilder::new("pdl.operands", location).add_result(context.pdl_range_type(context.pdl_value_type()));
    if let Some(value_type) = value_type {
        builder = builder.add_operand(value_type);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `pdl::operands`")
}

/// Name of the operand segment-size attribute used by PDL operations with multiple operand groups.
pub const OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operandSegmentSizes";

/// Name of the optional operation name attribute used by `pdl.operation`.
pub const OP_NAME_ATTRIBUTE: &str = "opName";

/// Name of the attribute names array used by `pdl.operation`.
pub const ATTRIBUTE_VALUE_NAMES_ATTRIBUTE: &str = "attributeValueNames";

/// Operation trait for `pdl.operation`.
pub trait OperationOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the optional concrete operation name being matched or created.
    fn op_name(&self) -> Option<StringRef<'c>> {
        self.attribute(OP_NAME_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<StringAttributeRef>().map(|attribute| attribute.string()))
    }

    /// Returns the operand handles attached to this operation handle.
    fn pdl_operand_values(&self) -> impl Iterator<Item = ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(Vec::<i32>::from)
            .unwrap_or_else(|| panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `pdl.operation`"));
        self.operands().map(|operand| operand.value()).take(sizes[0] as usize)
    }

    /// Returns the attribute value handles attached to this operation handle.
    fn attribute_values(&self) -> impl Iterator<Item = ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(Vec::<i32>::from)
            .unwrap_or_else(|| panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `pdl.operation`"));
        self.operands().map(|operand| operand.value()).skip(sizes[0] as usize).take(sizes[1] as usize)
    }

    /// Returns the attribute names corresponding to [`OperationOperation::attribute_values`].
    fn attribute_value_names(&self) -> ArrayAttributeRef<'c, 't> {
        self.attribute(ATTRIBUTE_VALUE_NAMES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<ArrayAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{ATTRIBUTE_VALUE_NAMES_ATTRIBUTE}' attribute in `pdl.operation`"))
    }

    /// Returns the result type handles attached to this operation handle.
    fn type_values(&self) -> impl Iterator<Item = ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(Vec::<i32>::from)
            .unwrap_or_else(|| panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `pdl.operation`"));
        self.operands()
            .map(|operand| operand.value())
            .skip((sizes[0] + sizes[1]) as usize)
            .take(sizes[2] as usize)
    }
}

mlir_op!(Operation);
mlir_op_trait!(Operation, OneResult);
mlir_op_trait!(Operation, ZeroRegions);
mlir_op_trait!(Operation, ZeroSuccessors);

/// Constructs a new detached/owned [`OperationOperation`] at the specified [`Location`].
pub fn operation<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    op_name: Option<StringAttributeRef<'c, 't>>,
    operand_values: &[ValueRef<'v, 'c, 't>],
    attribute_value_names: &[StringAttributeRef<'c, 't>],
    attribute_values: &[ValueRef<'v, 'c, 't>],
    type_values: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> DetachedOperationOperation<'c, 't> {
    assert_eq!(
        attribute_value_names.len(),
        attribute_values.len(),
        "`pdl.operation` attribute name/value counts must match"
    );

    let context = location.context();
    context.load_dialect(DialectHandle::pdl());
    let segment_sizes = [operand_values.len() as i32, attribute_values.len() as i32, type_values.len() as i32];
    let mut builder = OperationBuilder::new("pdl.operation", location)
        .add_operands(operand_values)
        .add_operands(attribute_values)
        .add_operands(type_values)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_attribute(ATTRIBUTE_VALUE_NAMES_ATTRIBUTE, context.array_attribute(attribute_value_names))
        .add_result(context.pdl_operation_type());
    if let Some(op_name) = op_name {
        builder = builder.add_attribute(OP_NAME_ATTRIBUTE, op_name);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `pdl::operation`")
}

/// Name of the `pdl.pattern` benefit attribute.
pub const BENEFIT_ATTRIBUTE: &str = "benefit";

/// Operation trait for `pdl.pattern`.
pub trait PatternOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the rewrite benefit of this pattern.
    fn benefit(&self) -> u16 {
        self.attribute(BENEFIT_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value() as u16)
            .unwrap_or_else(|| panic!("invalid '{BENEFIT_ATTRIBUTE}' attribute in `pdl.pattern`"))
    }

    /// Returns the pattern body region.
    fn body_region(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }
}

mlir_op!(Pattern);
mlir_op_trait!(Pattern, IsolatedFromAbove);
mlir_op_trait!(Pattern, OneRegion);
mlir_op_trait!(Pattern, SingleBlockRegions);
mlir_op_trait!(Pattern, SingleBlock);
mlir_op_trait!(Pattern, Symbol);
mlir_op_trait!(Pattern, ZeroOperands);
mlir_op_trait!(Pattern, ZeroSuccessors);

/// Constructs a new detached/owned [`PatternOperation`] at the specified [`Location`].
pub fn pattern<'c, 't: 'c, L: Location<'c, 't>>(
    benefit: u16,
    symbol_name: Option<StringAttributeRef<'c, 't>>,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedPatternOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::pdl());
    let mut builder = OperationBuilder::new("pdl.pattern", location)
        .add_attribute(BENEFIT_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(16), benefit as i64))
        .add_region(body);
    if let Some(symbol_name) = symbol_name {
        builder = builder.add_attribute(SYMBOL_NAME_ATTRIBUTE, symbol_name);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `pdl::pattern`")
}

/// Operation trait for `pdl.range`.
pub trait RangeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the PDL entities used to construct this range.
    fn arguments(&self) -> impl Iterator<Item = ValueRef<'o, 'c, 't>> {
        self.operand_values()
    }
}

mlir_op!(Range);
mlir_op_trait!(Range, AlwaysSpeculatable);
mlir_op_trait!(Range, NoMemoryEffect);
mlir_op_trait!(Range, OneResult);
mlir_op_trait!(Range, Pure);
mlir_op_trait!(Range, ZeroRegions);
mlir_op_trait!(Range, ZeroSuccessors);

/// Constructs a new detached/owned [`RangeOperation`] at the specified [`Location`].
pub fn range<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    arguments: &[ValueRef<'v, 'c, 't>],
    result_type: T,
    location: L,
) -> DetachedRangeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::pdl());
    OperationBuilder::new("pdl.range", location)
        .add_operands(arguments)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `pdl::range`")
}

/// Operation trait for `pdl.replace`.
pub trait ReplaceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the operation handle to replace.
    fn operation(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional replacement operation handle.
    fn replacement_operation(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(Vec::<i32>::from)
            .unwrap_or_else(|| panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `pdl.replace`"));
        if sizes[1] == 0 { None } else { self.operand_value(1) }
    }

    /// Returns the replacement values or value ranges.
    fn replacement_values(&self) -> impl Iterator<Item = ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(Vec::<i32>::from)
            .unwrap_or_else(|| panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `pdl.replace`"));
        self.operands()
            .map(|operand| operand.value())
            .skip((sizes[0] + sizes[1]) as usize)
            .take(sizes[2] as usize)
    }
}

mlir_op!(Replace);
mlir_op_trait!(Replace, ZeroRegions);
mlir_op_trait!(Replace, ZeroSuccessors);

/// Constructs a new detached/owned [`ReplaceOperation`] at the specified [`Location`].
pub fn replace<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operation: ValueRef<'v, 'c, 't>,
    replacement_operation: Option<ValueRef<'v, 'c, 't>>,
    replacement_values: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> DetachedReplaceOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::pdl());
    let segment_sizes = [1, i32::from(replacement_operation.is_some()), replacement_values.len() as i32];
    let mut builder = OperationBuilder::new("pdl.replace", location)
        .add_operand(operation)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap());
    if let Some(replacement_operation) = replacement_operation {
        builder = builder.add_operand(replacement_operation);
    }
    builder
        .add_operands(replacement_values)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `pdl::replace`")
}

/// Name of the result index attribute used by `pdl.result` and `pdl.results`.
pub const INDEX_ATTRIBUTE: &str = "index";

/// Operation trait for `pdl.result`.
pub trait ResultOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the operation handle that owns the extracted result.
    fn parent(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the zero-based result index to extract.
    fn index(&self) -> u32 {
        self.attribute(INDEX_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value() as u32)
            .unwrap_or_else(|| panic!("invalid '{INDEX_ATTRIBUTE}' attribute in `pdl.result`"))
    }
}

mlir_op!(Result);
mlir_op_trait!(Result, AlwaysSpeculatable);
mlir_op_trait!(Result, NoMemoryEffect);
mlir_op_trait!(Result, OneOperand);
mlir_op_trait!(Result, OneResult);
mlir_op_trait!(Result, Pure);
mlir_op_trait!(Result, ZeroRegions);
mlir_op_trait!(Result, ZeroSuccessors);

/// Constructs a new detached/owned [`ResultOperation`] at the specified [`Location`].
pub fn result<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    parent: ValueRef<'v, 'c, 't>,
    index: u32,
    location: L,
) -> DetachedResultOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::pdl());
    OperationBuilder::new("pdl.result", location)
        .add_operand(parent)
        .add_attribute(INDEX_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(32), index as i64))
        .add_result(context.pdl_value_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `pdl::result`")
}

/// Operation trait for `pdl.results`.
pub trait ResultsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the operation handle that owns the extracted results.
    fn parent(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional ODS result group index to extract.
    fn index(&self) -> Option<u32> {
        self.attribute(INDEX_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value() as u32)
    }
}

mlir_op!(Results);
mlir_op_trait!(Results, AlwaysSpeculatable);
mlir_op_trait!(Results, NoMemoryEffect);
mlir_op_trait!(Results, OneOperand);
mlir_op_trait!(Results, OneResult);
mlir_op_trait!(Results, Pure);
mlir_op_trait!(Results, ZeroRegions);
mlir_op_trait!(Results, ZeroSuccessors);

/// Constructs a new detached/owned [`ResultsOperation`] at the specified [`Location`].
pub fn results<'v, 'c: 'v, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    parent: ValueRef<'v, 'c, 't>,
    index: Option<u32>,
    result_type: T,
    location: L,
) -> DetachedResultsOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::pdl());
    let mut builder = OperationBuilder::new("pdl.results", location).add_operand(parent).add_result(result_type);
    if let Some(index) = index {
        builder = builder
            .add_attribute(INDEX_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(32), index as i64));
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `pdl::results`")
}

/// Operation trait for `pdl.rewrite`.
pub trait RewriteOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the optional root operation handle.
    fn root(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(Vec::<i32>::from)
            .unwrap_or_else(|| panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `pdl.rewrite`"));
        if sizes[0] == 0 { None } else { self.operand_value(0) }
    }

    /// Returns the optional external rewrite function name.
    fn external_rewrite_name(&self) -> Option<StringRef<'c>> {
        self.attribute(NAME_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<StringAttributeRef>().map(|attribute| attribute.string()))
    }

    /// Returns the additional PDL entities passed to an external rewrite.
    fn external_arguments(&self) -> impl Iterator<Item = ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(Vec::<i32>::from)
            .unwrap_or_else(|| panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `pdl.rewrite`"));
        self.operands().map(|operand| operand.value()).skip(sizes[0] as usize).take(sizes[1] as usize)
    }

    /// Returns the rewrite body region.
    fn body_region(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }
}

mlir_op!(Rewrite);
mlir_op_trait!(Rewrite, SingleBlockRegions);
mlir_op_trait!(Rewrite, IsTerminator);
mlir_op_trait!(Rewrite, NoRegionArguments);
mlir_op_trait!(Rewrite, NoTerminator);
mlir_op_trait!(Rewrite, OneRegion);
mlir_op_trait!(Rewrite, SingleBlock);
mlir_op_trait!(Rewrite, ZeroSuccessors);

/// Constructs a new detached/owned [`RewriteOperation`] at the specified [`Location`].
pub fn rewrite<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    root: Option<ValueRef<'v, 'c, 't>>,
    name: Option<StringAttributeRef<'c, 't>>,
    external_arguments: &[ValueRef<'v, 'c, 't>],
    body: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedRewriteOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::pdl());
    let segment_sizes = [i32::from(root.is_some()), external_arguments.len() as i32];
    let mut builder = OperationBuilder::new("pdl.rewrite", location)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_region(body);
    if let Some(root) = root {
        builder = builder.add_operand(root);
    }
    if let Some(name) = name {
        builder = builder.add_attribute(NAME_ATTRIBUTE, name);
    }
    builder
        .add_operands(external_arguments)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `pdl::rewrite`")
}

/// Name of the optional constant type attribute used by `pdl.type`.
pub const CONSTANT_TYPE_ATTRIBUTE: &str = "constantType";

/// Operation trait for `pdl.type`.
pub trait TypeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the optional constant type constraint.
    fn constant_type(&self) -> Option<TypeRef<'c, 't>> {
        self.attribute(CONSTANT_TYPE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TypeAttributeRef>().map(|attribute| attribute.r#type()))
    }
}

mlir_op!(Type);
mlir_op_trait!(Type, OneResult);
mlir_op_trait!(Type, ZeroOperands);
mlir_op_trait!(Type, ZeroRegions);
mlir_op_trait!(Type, ZeroSuccessors);

/// Constructs a new detached/owned [`TypeOperation`] at the specified [`Location`].
pub fn r#type<'c, 't: 'c, L: Location<'c, 't>>(
    constant_type: Option<TypeAttributeRef<'c, 't>>,
    location: L,
) -> DetachedTypeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::pdl());
    let mut builder = OperationBuilder::new("pdl.type", location).add_result(context.pdl_type_type());
    if let Some(constant_type) = constant_type {
        builder = builder.add_attribute(CONSTANT_TYPE_ATTRIBUTE, constant_type);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `pdl::type`")
}

/// Name of the optional constant type array attribute used by `pdl.types`.
pub const CONSTANT_TYPES_ATTRIBUTE: &str = "constantTypes";

/// Operation trait for `pdl.types`.
pub trait TypesOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the optional constant type array constraint.
    fn constant_types(&self) -> Option<ArrayAttributeRef<'c, 't>> {
        self.attribute(CONSTANT_TYPES_ATTRIBUTE).and_then(|attribute| attribute.cast::<ArrayAttributeRef>())
    }
}

mlir_op!(Types);
mlir_op_trait!(Types, OneResult);
mlir_op_trait!(Types, ZeroOperands);
mlir_op_trait!(Types, ZeroRegions);
mlir_op_trait!(Types, ZeroSuccessors);

/// Constructs a new detached/owned [`TypesOperation`] at the specified [`Location`].
pub fn types<'c, 't: 'c, L: Location<'c, 't>>(
    constant_types: Option<ArrayAttributeRef<'c, 't>>,
    location: L,
) -> DetachedTypesOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::pdl());
    let mut builder =
        OperationBuilder::new("pdl.types", location).add_result(context.pdl_range_type(context.pdl_type_type()));
    if let Some(constant_types) = constant_types {
        builder = builder.add_attribute(CONSTANT_TYPES_ATTRIBUTE, constant_types);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `pdl::types`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::{Attribute, Block, Context, OneResult, Operation, Region, Symbol, Type, Value, ValueRef};

    use super::*;

    #[test]
    fn test_apply_native_constraint() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let pdl_attribute_type = context.pdl_attribute_type();
        module.body().append_operation({
            let mut pattern_block = context.block_with_no_arguments();
            let i32_type = context.signless_integer_type(32);
            let type_op = r#type(Some(context.type_attribute(i32_type)), location);
            let type_op = pattern_block.append_operation(type_op);
            let type_value = type_op.result(0).unwrap().as_ref();

            let operand_op = operand(Some(type_value), location);
            let operand_op = pattern_block.append_operation(operand_op);
            let operand_value = operand_op.result(0).unwrap().as_ref();

            let constraint_op = apply_native_constraint(
                context.string_attribute("check"),
                &[operand_value],
                &[pdl_attribute_type.as_ref()],
                true,
                location,
            );
            assert_eq!(constraint_op.native_constraint_name().as_str().unwrap(), "check");
            assert_eq!(constraint_op.arguments().collect::<Vec<_>>(), vec![operand_value]);
            assert!(constraint_op.is_negated());
            assert_eq!(constraint_op.result(0).unwrap().r#type(), pdl_attribute_type.as_ref());
            let constraint_op = pattern_block.append_operation(constraint_op);
            let constraint_value = constraint_op.result(0).unwrap().as_ref();

            let attribute_names = [context.string_attribute("label")];
            let root_op = operation(
                Some(context.string_attribute("test.op")),
                &[operand_value],
                &attribute_names,
                &[constraint_value],
                &[type_value],
                location,
            );
            let root_op = pattern_block.append_operation(root_op);
            let root_value = root_op.result(0).unwrap().as_ref();

            pattern_block.append_operation(rewrite(
                Some(root_value),
                Some(context.string_attribute("finish")),
                &[],
                context.region(),
                location,
            ));

            let pattern_op = pattern(
                1,
                Some(context.string_attribute("apply_native_constraint_test")),
                pattern_block.into(),
                location,
            );
            assert_eq!(pattern_op.benefit(), 1);
            pattern_op
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  pdl.pattern @apply_native_constraint_test : benefit(1) {
                    %0 = type : i32
                    %1 = operand : %0
                    %2 = apply_native_constraint "check"(%1 : !pdl.value) : !pdl.attribute {isNegated = true}
                    %3 = operation "test.op"(%1 : !pdl.value)  {"label" = %2} -> (%0 : !pdl.type)
                    rewrite %3 with "finish"
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_apply_native_rewrite() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let pdl_value_type = context.pdl_value_type();
        module.body().append_operation({
            let mut pattern_block = context.block_with_no_arguments();
            let root_op = operation(Some(context.string_attribute("test.op")), &[], &[], &[], &[], location);
            let root_op = pattern_block.append_operation(root_op);
            let root_value = root_op.result(0).unwrap().as_ref();

            let mut rewrite_block = context.block_with_no_arguments();
            let rewrite_value_op = apply_native_rewrite(
                context.string_attribute("rewrite_value"),
                &[root_value],
                &[pdl_value_type.as_ref()],
                location,
            );
            assert_eq!(rewrite_value_op.native_rewrite_name().as_str().unwrap(), "rewrite_value");
            assert_eq!(rewrite_value_op.arguments().collect::<Vec<_>>(), vec![root_value]);
            assert_eq!(rewrite_value_op.result(0).unwrap().r#type(), pdl_value_type.as_ref());
            rewrite_block.append_operation(rewrite_value_op);

            let rewrite_op = rewrite(Some(root_value), None, &[], rewrite_block.into(), location);
            assert_eq!(rewrite_op.root().unwrap(), root_value);
            assert_eq!(rewrite_op.external_arguments().collect::<Vec<_>>(), Vec::<ValueRef>::new());
            pattern_block.append_operation(rewrite_op);

            pattern(1, Some(context.string_attribute("apply_native_rewrite_test")), pattern_block.into(), location)
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  pdl.pattern @apply_native_rewrite_test : benefit(1) {
                    %0 = operation \"test.op\"\x20
                    rewrite %0 {
                      %1 = apply_native_rewrite \"rewrite_value\"(%0 : !pdl.operation) : !pdl.value
                    }
                  }
                }
            "},
        );
    }

    #[test]
    fn test_attribute() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let pdl_attribute_type = context.pdl_attribute_type();
        module.body().append_operation({
            let mut pattern_block = context.block_with_no_arguments();
            let i32_type = context.signless_integer_type(32);
            let type_op = r#type(Some(context.type_attribute(i32_type)), location);
            let type_op = pattern_block.append_operation(type_op);
            let type_value = type_op.result(0).unwrap().as_ref();

            let attribute_op = attribute(Some(type_value), None, location);
            assert_eq!(attribute_op.value_type().unwrap(), type_value);
            assert!(attribute_op.value().is_none());
            assert_eq!(attribute_op.output_type(), pdl_attribute_type.as_ref());
            let attribute_op = pattern_block.append_operation(attribute_op);
            let attribute_value = attribute_op.result(0).unwrap().as_ref();

            let attribute_names = [context.string_attribute("label")];
            let root_op = operation(
                Some(context.string_attribute("test.op")),
                &[],
                &attribute_names,
                &[attribute_value],
                &[type_value],
                location,
            );
            let root_op = pattern_block.append_operation(root_op);
            let root_value = root_op.result(0).unwrap().as_ref();

            pattern_block.append_operation(rewrite(
                Some(root_value),
                Some(context.string_attribute("finish")),
                &[],
                context.region(),
                location,
            ));

            let pattern_op =
                pattern(1, Some(context.string_attribute("attribute_test")), pattern_block.into(), location);
            assert_eq!(pattern_op.benefit(), 1);
            pattern_op
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  pdl.pattern @attribute_test : benefit(1) {
                    %0 = type : i32
                    %1 = attribute : %0
                    %2 = operation "test.op"  {"label" = %1} -> (%0 : !pdl.type)
                    rewrite %2 with "finish"
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_erase() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        module.body().append_operation({
            let mut pattern_block = context.block_with_no_arguments();
            let root_op = operation(None, &[], &[], &[], &[], location);
            assert!(root_op.op_name().is_none());
            let root_op = pattern_block.append_operation(root_op);
            let root_value = root_op.result(0).unwrap().as_ref();

            let mut rewrite_block = context.block_with_no_arguments();
            let erase_op = erase(root_value, location);
            assert_eq!(erase_op.operation(), root_value);
            rewrite_block.append_operation(erase_op);

            pattern_block.append_operation(rewrite(Some(root_value), None, &[], rewrite_block.into(), location));
            pattern(1, Some(context.string_attribute("erase_test")), pattern_block.into(), location)
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  pdl.pattern @erase_test : benefit(1) {
                    %0 = operation\x20
                    rewrite %0 {
                      erase %0
                    }
                  }
                }
            "},
        );
    }

    #[test]
    fn test_operand() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let pdl_value_type = context.pdl_value_type();
        module.body().append_operation({
            let mut pattern_block = context.block_with_no_arguments();
            let i32_type = context.signless_integer_type(32);
            let type_op = r#type(Some(context.type_attribute(i32_type)), location);
            let type_op = pattern_block.append_operation(type_op);
            let type_value = type_op.result(0).unwrap().as_ref();

            let operand_op = operand(Some(type_value), location);
            assert_eq!(operand_op.value_type().unwrap(), type_value);
            assert_eq!(operand_op.output_type(), pdl_value_type.as_ref());
            let operand_op = pattern_block.append_operation(operand_op);
            let operand_value = operand_op.result(0).unwrap().as_ref();

            let root_op = operation(
                Some(context.string_attribute("test.op")),
                &[operand_value],
                &[],
                &[],
                &[type_value],
                location,
            );
            let root_op = pattern_block.append_operation(root_op);
            let root_value = root_op.result(0).unwrap().as_ref();

            pattern_block.append_operation(rewrite(
                Some(root_value),
                Some(context.string_attribute("finish")),
                &[],
                context.region(),
                location,
            ));

            pattern(1, Some(context.string_attribute("operand_test")), pattern_block.into(), location)
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  pdl.pattern @operand_test : benefit(1) {
                    %0 = type : i32
                    %1 = operand : %0
                    %2 = operation "test.op"(%1 : !pdl.value)  -> (%0 : !pdl.type)
                    rewrite %2 with "finish"
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_operands() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let pdl_range_value_type = context.pdl_range_type(context.pdl_value_type());
        module.body().append_operation({
            let mut pattern_block = context.block_with_no_arguments();
            let i32_type = context.signless_integer_type(32);
            let i64_type = context.signless_integer_type(64);
            let constant_types = context.array_attribute(&[
                context.type_attribute(i32_type).as_ref(),
                context.type_attribute(i64_type).as_ref(),
            ]);
            let types_op = types(Some(constant_types), location);
            let types_op = pattern_block.append_operation(types_op);
            let types_value = types_op.result(0).unwrap().as_ref();

            let operands_op = operands(Some(types_value), location);
            assert_eq!(operands_op.value_type().unwrap(), types_value);
            assert_eq!(operands_op.output_type(), pdl_range_value_type.as_ref());
            let operands_op = pattern_block.append_operation(operands_op);
            let operands_value = operands_op.result(0).unwrap().as_ref();

            let root_op =
                operation(Some(context.string_attribute("test.op")), &[operands_value], &[], &[], &[], location);
            let root_op = pattern_block.append_operation(root_op);
            let root_value = root_op.result(0).unwrap().as_ref();

            pattern_block.append_operation(rewrite(
                Some(root_value),
                Some(context.string_attribute("finish")),
                &[],
                context.region(),
                location,
            ));

            pattern(1, Some(context.string_attribute("operands_test")), pattern_block.into(), location)
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  pdl.pattern @operands_test : benefit(1) {
                    %0 = types : [i32, i64]
                    %1 = operands : %0
                    %2 = operation \"test.op\"(%1 : !pdl.range<value>)\x20
                    rewrite %2 with \x22finish\x22
                  }
                }
            "},
        );
    }

    #[test]
    fn test_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let pdl_operation_type = context.pdl_operation_type();
        module.body().append_operation({
            let mut pattern_block = context.block_with_no_arguments();
            let i32_type = context.signless_integer_type(32);
            let type_op = r#type(Some(context.type_attribute(i32_type)), location);
            let type_op = pattern_block.append_operation(type_op);
            let type_value = type_op.result(0).unwrap().as_ref();

            let operand_op = operand(Some(type_value), location);
            let operand_op = pattern_block.append_operation(operand_op);
            let operand_value = operand_op.result(0).unwrap().as_ref();

            let constant_attribute = context.string_attribute("hello").as_ref();
            let attribute_op = attribute(None, Some(constant_attribute), location);
            let attribute_op = pattern_block.append_operation(attribute_op);
            let attribute_value = attribute_op.result(0).unwrap().as_ref();

            let attribute_names = [context.string_attribute("label")];
            let operation_op = operation(
                Some(context.string_attribute("test.op")),
                &[operand_value],
                &attribute_names,
                &[attribute_value],
                &[type_value],
                location,
            );
            assert_eq!(operation_op.op_name().unwrap().as_str().unwrap(), "test.op");
            assert_eq!(operation_op.pdl_operand_values().collect::<Vec<_>>(), vec![operand_value]);
            assert_eq!(operation_op.attribute_values().collect::<Vec<_>>(), vec![attribute_value]);
            assert_eq!(operation_op.attribute_value_names().len(), 1);
            assert_eq!(operation_op.type_values().collect::<Vec<_>>(), vec![type_value]);
            assert_eq!(operation_op.output_type(), pdl_operation_type.as_ref());
            let operation_op = pattern_block.append_operation(operation_op);
            let operation_value = operation_op.result(0).unwrap().as_ref();

            pattern_block.append_operation(rewrite(
                Some(operation_value),
                Some(context.string_attribute("finish")),
                &[],
                context.region(),
                location,
            ));

            pattern(1, Some(context.string_attribute("operation_test")), pattern_block.into(), location)
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  pdl.pattern @operation_test : benefit(1) {
                    %0 = type : i32
                    %1 = operand : %0
                    %2 = attribute = "hello"
                    %3 = operation "test.op"(%1 : !pdl.value)  {"label" = %2} -> (%0 : !pdl.type)
                    rewrite %3 with "finish"
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_pattern() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        module.body().append_operation({
            let mut pattern_block = context.block_with_no_arguments();
            let root_op = operation(Some(context.string_attribute("test.op")), &[], &[], &[], &[], location);
            let root_op = pattern_block.append_operation(root_op);
            let root_value = root_op.result(0).unwrap().as_ref();
            pattern_block.append_operation(rewrite(
                Some(root_value),
                Some(context.string_attribute("finish")),
                &[],
                context.region(),
                location,
            ));

            let pattern_op = pattern(7, Some(context.string_attribute("pattern_test")), pattern_block.into(), location);
            assert_eq!(pattern_op.benefit(), 7);
            assert_eq!(pattern_op.symbol_name().unwrap().as_str().unwrap(), "pattern_test");
            assert_eq!(pattern_op.regions().count(), 1);
            pattern_op
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  pdl.pattern @pattern_test : benefit(7) {
                    %0 = operation \"test.op\"\x20
                    rewrite %0 with \"finish\"
                  }
                }
            "},
        );
    }

    #[test]
    fn test_range() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let pdl_range_value_type = context.pdl_range_type(context.pdl_value_type());
        module.body().append_operation({
            let mut pattern_block = context.block_with_no_arguments();
            let root_op = operation(Some(context.string_attribute("test.op")), &[], &[], &[], &[], location);
            let root_op = pattern_block.append_operation(root_op);
            let root_value = root_op.result(0).unwrap().as_ref();

            let mut rewrite_block = context.block_with_no_arguments();
            let rewrite_value_op = apply_native_rewrite(
                context.string_attribute("make_value"),
                &[root_value],
                &[context.pdl_value_type().as_ref()],
                location,
            );
            let rewrite_value_op = rewrite_block.append_operation(rewrite_value_op);
            let rewrite_value = rewrite_value_op.result(0).unwrap().as_ref();

            let range_op = range(&[rewrite_value], pdl_range_value_type, location);
            assert_eq!(range_op.arguments().collect::<Vec<_>>(), vec![rewrite_value]);
            assert_eq!(range_op.output_type(), pdl_range_value_type.as_ref());
            let range_op = rewrite_block.append_operation(range_op);
            let range_value = range_op.result(0).unwrap().as_ref();

            let replacement_op =
                operation(Some(context.string_attribute("test.replacement")), &[range_value], &[], &[], &[], location);
            let replacement_op = rewrite_block.append_operation(replacement_op);
            let replacement_value = replacement_op.result(0).unwrap().as_ref();
            rewrite_block.append_operation(replace(root_value, Some(replacement_value), &[], location));

            pattern_block.append_operation(rewrite(Some(root_value), None, &[], rewrite_block.into(), location));

            pattern(1, Some(context.string_attribute("range_test")), pattern_block.into(), location)
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  pdl.pattern @range_test : benefit(1) {
                    %0 = operation \"test.op\"\x20
                    rewrite %0 {
                      %1 = apply_native_rewrite \"make_value\"(%0 : !pdl.operation) : !pdl.value
                      %2 = range %1 : !pdl.value\x20
                      %3 = operation \"test.replacement\"(%2 : !pdl.range<value>)\x20
                      replace %0 with %3
                    }
                  }
                }
            "},
        );
    }

    #[test]
    fn test_replace() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        module.body().append_operation({
            let mut pattern_block = context.block_with_no_arguments();
            let root_op = operation(Some(context.string_attribute("test.op")), &[], &[], &[], &[], location);
            let root_op = pattern_block.append_operation(root_op);
            let root_value = root_op.result(0).unwrap().as_ref();

            let mut rewrite_block = context.block_with_no_arguments();
            let replacement_op =
                operation(Some(context.string_attribute("test.replacement")), &[], &[], &[], &[], location);
            let replacement_op = rewrite_block.append_operation(replacement_op);
            let replacement_value = replacement_op.result(0).unwrap().as_ref();

            let replace_op = replace(root_value, Some(replacement_value), &[], location);
            assert_eq!(replace_op.operation(), root_value);
            assert_eq!(replace_op.replacement_operation().unwrap(), replacement_value);
            assert_eq!(replace_op.replacement_values().collect::<Vec<_>>(), Vec::<ValueRef>::new());
            rewrite_block.append_operation(replace_op);

            pattern_block.append_operation(rewrite(Some(root_value), None, &[], rewrite_block.into(), location));
            pattern(1, Some(context.string_attribute("replace_test")), pattern_block.into(), location)
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  pdl.pattern @replace_test : benefit(1) {
                    %0 = operation \"test.op\"\x20
                    rewrite %0 {
                      %1 = operation \"test.replacement\"\x20
                      replace %0 with %1
                    }
                  }
                }
            "},
        );
    }

    #[test]
    fn test_result() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let pdl_value_type = context.pdl_value_type();
        module.body().append_operation({
            let mut pattern_block = context.block_with_no_arguments();
            let type_op = r#type(Some(context.type_attribute(context.signless_integer_type(32))), location);
            let type_op = pattern_block.append_operation(type_op);
            let type_value = type_op.result(0).unwrap().as_ref();

            let root_op = operation(Some(context.string_attribute("test.op")), &[], &[], &[], &[type_value], location);
            let root_op = pattern_block.append_operation(root_op);
            let root_value = root_op.result(0).unwrap().as_ref();

            let result_op = result(root_value, 0, location);
            assert_eq!(result_op.parent(), root_value);
            assert_eq!(result_op.index(), 0);
            assert_eq!(result_op.output_type(), pdl_value_type.as_ref());
            let result_op = pattern_block.append_operation(result_op);
            let result_value = result_op.result(0).unwrap().as_ref();

            pattern_block.append_operation(rewrite(
                Some(root_value),
                Some(context.string_attribute("consume_result")),
                &[result_value],
                context.region(),
                location,
            ));

            pattern(1, Some(context.string_attribute("result_test")), pattern_block.into(), location)
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  pdl.pattern @result_test : benefit(1) {
                    %0 = type : i32
                    %1 = operation "test.op"  -> (%0 : !pdl.type)
                    %2 = result 0 of %1
                    rewrite %1 with "consume_result"(%2 : !pdl.value)
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_results() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let pdl_range_value_type = context.pdl_range_type(context.pdl_value_type());
        module.body().append_operation({
            let mut pattern_block = context.block_with_no_arguments();
            let type_op = r#type(Some(context.type_attribute(context.signless_integer_type(32))), location);
            let type_op = pattern_block.append_operation(type_op);
            let type_value = type_op.result(0).unwrap().as_ref();

            let root_op = operation(Some(context.string_attribute("test.op")), &[], &[], &[], &[type_value], location);
            let root_op = pattern_block.append_operation(root_op);
            let root_value = root_op.result(0).unwrap().as_ref();

            let results_op = results(root_value, Some(0), pdl_range_value_type, location);
            assert_eq!(results_op.parent(), root_value);
            assert_eq!(results_op.index(), Some(0));
            assert_eq!(results_op.output_type(), pdl_range_value_type.as_ref());
            let results_op = pattern_block.append_operation(results_op);
            let results_value = results_op.result(0).unwrap().as_ref();

            pattern_block.append_operation(rewrite(
                Some(root_value),
                Some(context.string_attribute("consume_results")),
                &[results_value],
                context.region(),
                location,
            ));

            pattern(1, Some(context.string_attribute("results_test")), pattern_block.into(), location)
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  pdl.pattern @results_test : benefit(1) {
                    %0 = type : i32
                    %1 = operation "test.op"  -> (%0 : !pdl.type)
                    %2 = results 0 of %1  -> !pdl.range<value>
                    rewrite %1 with "consume_results"(%2 : !pdl.range<value>)
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_rewrite() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        module.body().append_operation({
            let mut pattern_block = context.block_with_no_arguments();
            let operand_op = operand(None, location);
            assert!(operand_op.value_type().is_none());
            let operand_op = pattern_block.append_operation(operand_op);
            let operand_value = operand_op.result(0).unwrap().as_ref();

            let root_op =
                operation(Some(context.string_attribute("external.op")), &[operand_value], &[], &[], &[], location);
            let root_op = pattern_block.append_operation(root_op);
            let root_value = root_op.result(0).unwrap().as_ref();

            let rewrite_op = rewrite(
                Some(root_value),
                Some(context.string_attribute("external_rewrite")),
                &[operand_value],
                context.region(),
                location,
            );
            assert_eq!(rewrite_op.root().unwrap(), root_value);
            assert_eq!(rewrite_op.external_rewrite_name().unwrap().as_str().unwrap(), "external_rewrite");
            assert_eq!(rewrite_op.external_arguments().collect::<Vec<_>>(), vec![operand_value]);
            assert_eq!(rewrite_op.body_region().blocks().count(), 0);
            pattern_block.append_operation(rewrite_op);

            let pattern_op = pattern(2, Some(context.string_attribute("rewrite_test")), pattern_block.into(), location);
            assert_eq!(pattern_op.benefit(), 2);
            pattern_op
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  pdl.pattern @rewrite_test : benefit(2) {
                    %0 = operand
                    %1 = operation \"external.op\"(%0 : !pdl.value)\x20
                    rewrite %1 with \"external_rewrite\"(%0 : !pdl.value)
                  }
                }
            "},
        );
    }

    #[test]
    fn test_type() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let pdl_type_type = context.pdl_type_type();
        module.body().append_operation({
            let mut pattern_block = context.block_with_no_arguments();
            let i32_type = context.signless_integer_type(32);
            let type_op = r#type(Some(context.type_attribute(i32_type)), location);
            assert_eq!(type_op.constant_type().unwrap(), i32_type.as_ref());
            assert_eq!(type_op.output_type(), pdl_type_type.as_ref());
            let type_op = pattern_block.append_operation(type_op);
            let type_value = type_op.result(0).unwrap().as_ref();

            let root_op = operation(Some(context.string_attribute("test.op")), &[], &[], &[], &[type_value], location);
            let root_op = pattern_block.append_operation(root_op);
            let root_value = root_op.result(0).unwrap().as_ref();

            pattern_block.append_operation(rewrite(
                Some(root_value),
                Some(context.string_attribute("finish")),
                &[],
                context.region(),
                location,
            ));

            pattern(1, Some(context.string_attribute("type_test")), pattern_block.into(), location)
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {r#"
                module {
                  pdl.pattern @type_test : benefit(1) {
                    %0 = type : i32
                    %1 = operation "test.op"  -> (%0 : !pdl.type)
                    rewrite %1 with "finish"
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_types() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let pdl_range_type_type = context.pdl_range_type(context.pdl_type_type());
        module.body().append_operation({
            let mut pattern_block = context.block_with_no_arguments();
            let i32_type = context.signless_integer_type(32);
            let i64_type = context.signless_integer_type(64);
            let constant_types = context.array_attribute(&[
                context.type_attribute(i32_type).as_ref(),
                context.type_attribute(i64_type).as_ref(),
            ]);
            let types_op = types(Some(constant_types), location);
            assert_eq!(types_op.constant_types().unwrap(), constant_types);
            assert_eq!(types_op.output_type(), pdl_range_type_type.as_ref());
            let types_op = pattern_block.append_operation(types_op);
            let types_value = types_op.result(0).unwrap().as_ref();

            let operands_op = operands(Some(types_value), location);
            let operands_op = pattern_block.append_operation(operands_op);
            let operands_value = operands_op.result(0).unwrap().as_ref();

            let root_op =
                operation(Some(context.string_attribute("test.op")), &[operands_value], &[], &[], &[], location);
            let root_op = pattern_block.append_operation(root_op);
            let root_value = root_op.result(0).unwrap().as_ref();

            pattern_block.append_operation(rewrite(
                Some(root_value),
                Some(context.string_attribute("finish")),
                &[],
                context.region(),
                location,
            ));

            pattern(1, Some(context.string_attribute("types_test")), pattern_block.into(), location)
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  pdl.pattern @types_test : benefit(1) {
                    %0 = types : [i32, i64]
                    %1 = operands : %0
                    %2 = operation \"test.op\"(%1 : !pdl.range<value>)\x20
                    rewrite %2 with \"finish\"
                  }
                }
            "},
        );
    }
}
