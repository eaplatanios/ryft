use crate::{
    Attribute, AttributeRef, Block, BlockRef, DenseInteger32ArrayAttributeRef, DetachedOp, DialectHandle, Location,
    Operation, OperationBuilder, TypeRef, Value, ValueRef, mlir_op,
};

/// Canonical MLIR operation name for [`BlockAddressOperation`].
pub const BLOCK_ADDRESS_OPERATION_NAME: &str = "llvm.blockaddress";

/// Operation trait for `llvm.blockaddress`.
pub trait BlockAddressOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        BLOCK_ADDRESS_OPERATION_NAME
    }

    /// Returns the `block_addr` attribute.
    fn block_addr(&self) -> AttributeRef<'c, 't> {
        self.attribute("block_addr").unwrap()
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(BlockAddress);

/// Constructs a new detached `llvm.blockaddress` operation.
pub fn block_address<'c, 't: 'c, L: Location<'c, 't>>(
    result_type: TypeRef<'c, 't>,
    block_addr: AttributeRef<'c, 't>,
    location: L,
) -> DetachedBlockAddressOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(BLOCK_ADDRESS_OPERATION_NAME, location);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("block_addr", block_addr);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::block_address`")
}

/// Canonical MLIR operation name for [`BlockTagOperation`].
pub const BLOCK_TAG_OPERATION_NAME: &str = "llvm.blocktag";

/// Operation trait for `llvm.blocktag`.
pub trait BlockTagOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        BLOCK_TAG_OPERATION_NAME
    }

    /// Returns the `tag` attribute.
    fn tag(&self) -> AttributeRef<'c, 't> {
        self.attribute("tag").unwrap()
    }
}

mlir_op!(BlockTag);

/// Constructs a new detached `llvm.blocktag` operation.
pub fn block_tag<'c, 't: 'c, L: Location<'c, 't>>(
    tag: AttributeRef<'c, 't>,
    location: L,
) -> DetachedBlockTagOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(BLOCK_TAG_OPERATION_NAME, location);
    builder = builder.add_attribute("tag", tag);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::block_tag`")
}

/// Canonical MLIR operation name for [`CondBrOperation`].
pub const COND_BR_OPERATION_NAME: &str = "llvm.cond_br";

/// Operation trait for `llvm.cond_br`.
pub trait CondBrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        COND_BR_OPERATION_NAME
    }

    /// Returns the `condition` operand.
    fn condition(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `true_destination_operands` operands.
    fn true_destination_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let true_destination_operand_count = self
            .attribute("operand_segment_sizes")
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| Vec::<i32>::from(attribute)[1])
            .unwrap_or_else(|| panic!("invalid 'operand_segment_sizes' attribute in `llvm.cond_br`"));
        self.operand_values().skip(1).take(true_destination_operand_count as usize).collect()
    }

    /// Returns the `false_destination_operands` operands.
    fn false_destination_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let true_destination_operand_count = self
            .attribute("operand_segment_sizes")
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| Vec::<i32>::from(attribute)[1])
            .unwrap_or_else(|| panic!("invalid 'operand_segment_sizes' attribute in `llvm.cond_br`"));
        self.operand_values().skip(1 + true_destination_operand_count as usize).collect()
    }

    /// Returns the true destination block.
    fn true_destination(&self) -> BlockRef<'o, 'c, 't> {
        self.successor(0).unwrap()
    }

    /// Returns the false destination block.
    fn false_destination(&self) -> BlockRef<'o, 'c, 't> {
        self.successor(1).unwrap()
    }

    /// Returns the optional `branch_weights` attribute.
    fn branch_weights(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("branch_weights")
    }

    /// Returns the optional `loop_annotation` attribute.
    fn loop_annotation(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("loop_annotation")
    }
}

mlir_op!(CondBr);

/// Constructs a new detached `llvm.cond_br` operation.
pub fn cond_br<
    'b,
    'c: 'b,
    't: 'c,
    V1: Value<'c, 'c, 't>,
    B1: Block<'b, 'c, 't>,
    B2: Block<'b, 'c, 't>,
    L: Location<'c, 't>,
>(
    condition: V1,
    true_destination: &B1,
    false_destination: &B2,
    true_destination_operands: &[ValueRef<'c, 'c, 't>],
    false_destination_operands: &[ValueRef<'c, 'c, 't>],
    branch_weights: Option<AttributeRef<'c, 't>>,
    loop_annotation: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedCondBrOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(COND_BR_OPERATION_NAME, location);
    builder = builder.add_operand(condition);
    builder = builder.add_operands(true_destination_operands);
    builder = builder.add_operands(false_destination_operands);
    builder = builder.add_successor(true_destination);
    builder = builder.add_successor(false_destination);
    builder = builder.add_attribute(
        "operand_segment_sizes",
        context
            .dense_i32_array_attribute(&[
                1,
                true_destination_operands.len() as i32,
                false_destination_operands.len() as i32,
            ])
            .unwrap(),
    );
    if let Some(branch_weights) = branch_weights {
        builder = builder.add_attribute("branch_weights", branch_weights);
    }
    if let Some(loop_annotation) = loop_annotation {
        builder = builder.add_attribute("loop_annotation", loop_annotation);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::cond_br`")
}

/// Canonical MLIR operation name for [`IndirectBrOperation`].
pub const INDIRECT_BR_OPERATION_NAME: &str = "llvm.indirectbr";

/// Operation trait for `llvm.indirectbr`.
pub trait IndirectBrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        INDIRECT_BR_OPERATION_NAME
    }

    /// Returns the `address` operand.
    fn address(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `successor_operands` operands.
    fn successor_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(1).collect()
    }

    /// Returns the destination blocks.
    fn destinations(&self) -> Vec<BlockRef<'o, 'c, 't>> {
        self.successors().collect()
    }

    /// Returns the `indbr_operand_segments` attribute.
    fn indbr_operand_segments(&self) -> AttributeRef<'c, 't> {
        self.attribute("indbr_operand_segments").unwrap()
    }
}

mlir_op!(IndirectBr);

/// Constructs a new detached `llvm.indirectbr` operation.
pub fn indirect_br<'b, 'c: 'b, 't: 'c, V1: Value<'c, 'c, 't>, B: Block<'b, 'c, 't>, L: Location<'c, 't>>(
    address: V1,
    destinations: &[&B],
    successor_operands: &[ValueRef<'c, 'c, 't>],
    indbr_operand_segments: AttributeRef<'c, 't>,
    location: L,
) -> DetachedIndirectBrOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(INDIRECT_BR_OPERATION_NAME, location);
    builder = builder.add_operand(address);
    builder = builder.add_operands(successor_operands);
    builder = builder.add_successors(destinations);
    builder = builder.add_attribute("indbr_operand_segments", indbr_operand_segments);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::indirect_br`")
}

/// Canonical MLIR operation name for [`InvokeOperation`].
pub const INVOKE_OPERATION_NAME: &str = "llvm.invoke";

/// Operation trait for `llvm.invoke`.
pub trait InvokeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        INVOKE_OPERATION_NAME
    }

    /// Returns the `callee_operands` operands.
    fn callee_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute("operand_segment_sizes")
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(Vec::<i32>::from)
            .unwrap_or_else(|| panic!("invalid 'operand_segment_sizes' attribute in `llvm.invoke`"));
        self.operand_values().take(segment_sizes[0] as usize).collect()
    }

    /// Returns the `normal_destination_operands` operands.
    fn normal_destination_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute("operand_segment_sizes")
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(Vec::<i32>::from)
            .unwrap_or_else(|| panic!("invalid 'operand_segment_sizes' attribute in `llvm.invoke`"));
        self.operand_values().skip(segment_sizes[0] as usize).take(segment_sizes[1] as usize).collect()
    }

    /// Returns the `unwind_destination_operands` operands.
    fn unwind_destination_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute("operand_segment_sizes")
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(Vec::<i32>::from)
            .unwrap_or_else(|| panic!("invalid 'operand_segment_sizes' attribute in `llvm.invoke`"));
        self.operand_values()
            .skip((segment_sizes[0] + segment_sizes[1]) as usize)
            .take(segment_sizes[2] as usize)
            .collect()
    }

    /// Returns the `op_bundle_operands` operands.
    fn op_bundle_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute("operand_segment_sizes")
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(Vec::<i32>::from)
            .unwrap_or_else(|| panic!("invalid 'operand_segment_sizes' attribute in `llvm.invoke`"));
        self.operand_values()
            .skip((segment_sizes[0] + segment_sizes[1] + segment_sizes[2]) as usize)
            .collect()
    }

    /// Returns the normal destination block.
    fn normal_destination(&self) -> BlockRef<'o, 'c, 't> {
        self.successor(0).unwrap()
    }

    /// Returns the unwind destination block.
    fn unwind_destination(&self) -> BlockRef<'o, 'c, 't> {
        self.successor(1).unwrap()
    }

    /// Returns the optional `var_callee_type` attribute.
    fn var_callee_type(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("var_callee_type")
    }

    /// Returns the optional `callee` attribute.
    fn callee(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("callee")
    }

    /// Returns the optional `arg_attrs` attribute.
    fn arg_attrs(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("arg_attrs")
    }

    /// Returns the optional `res_attrs` attribute.
    fn res_attrs(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("res_attrs")
    }

    /// Returns the optional `branch_weights` attribute.
    fn branch_weights(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("branch_weights")
    }

    /// Returns the optional `calling_convention` attribute.
    fn calling_convention(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("CConv")
    }

    /// Returns the optional `op_bundle_sizes` attribute.
    fn op_bundle_sizes(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("op_bundle_sizes")
    }

    /// Returns the optional `op_bundle_tags` attribute.
    fn op_bundle_tags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("op_bundle_tags")
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Invoke);

/// Constructs a new detached `llvm.invoke` operation.
pub fn invoke<'b, 'c: 'b, 't: 'c, B1: Block<'b, 'c, 't>, B2: Block<'b, 'c, 't>, L: Location<'c, 't>>(
    callee_operands: &[ValueRef<'c, 'c, 't>],
    normal_destination: &B1,
    normal_destination_operands: &[ValueRef<'c, 'c, 't>],
    unwind_destination: &B2,
    unwind_destination_operands: &[ValueRef<'c, 'c, 't>],
    op_bundle_operands: &[ValueRef<'c, 'c, 't>],
    result_type: TypeRef<'c, 't>,
    var_callee_type: Option<AttributeRef<'c, 't>>,
    callee: Option<AttributeRef<'c, 't>>,
    arg_attrs: Option<AttributeRef<'c, 't>>,
    res_attrs: Option<AttributeRef<'c, 't>>,
    branch_weights: Option<AttributeRef<'c, 't>>,
    calling_convention: Option<AttributeRef<'c, 't>>,
    op_bundle_sizes: Option<AttributeRef<'c, 't>>,
    op_bundle_tags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedInvokeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(INVOKE_OPERATION_NAME, location);
    builder = builder.add_operands(callee_operands);
    builder = builder.add_operands(normal_destination_operands);
    builder = builder.add_operands(unwind_destination_operands);
    builder = builder.add_operands(op_bundle_operands);
    builder = builder.add_successor(normal_destination);
    builder = builder.add_successor(unwind_destination);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute(
        "operand_segment_sizes",
        context
            .dense_i32_array_attribute(&[
                callee_operands.len() as i32,
                normal_destination_operands.len() as i32,
                unwind_destination_operands.len() as i32,
                op_bundle_operands.len() as i32,
            ])
            .unwrap(),
    );
    builder = builder.add_attribute(
        "op_bundle_sizes",
        op_bundle_sizes.unwrap_or_else(|| context.dense_i32_array_attribute(&[]).unwrap().as_ref()),
    );
    if let Some(var_callee_type) = var_callee_type {
        builder = builder.add_attribute("var_callee_type", var_callee_type);
    }
    if let Some(callee) = callee {
        builder = builder.add_attribute("callee", callee);
    }
    if let Some(arg_attrs) = arg_attrs {
        builder = builder.add_attribute("arg_attrs", arg_attrs);
    }
    if let Some(res_attrs) = res_attrs {
        builder = builder.add_attribute("res_attrs", res_attrs);
    }
    if let Some(branch_weights) = branch_weights {
        builder = builder.add_attribute("branch_weights", branch_weights);
    }
    if let Some(calling_convention) = calling_convention {
        builder = builder.add_attribute("CConv", calling_convention);
    }
    if let Some(op_bundle_tags) = op_bundle_tags {
        builder = builder.add_attribute("op_bundle_tags", op_bundle_tags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::invoke`")
}

/// Canonical MLIR operation name for [`ResumeOperation`].
pub const RESUME_OPERATION_NAME: &str = "llvm.resume";

/// Operation trait for `llvm.resume`.
pub trait ResumeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        RESUME_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }
}

mlir_op!(Resume);

/// Constructs a new detached `llvm.resume` operation.
pub fn resume<'c, 't: 'c, V1: Value<'c, 'c, 't>, L: Location<'c, 't>>(
    value: V1,
    location: L,
) -> DetachedResumeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(RESUME_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::resume`")
}

/// Canonical MLIR operation name for [`SwitchOperation`].
pub const SWITCH_OPERATION_NAME: &str = "llvm.switch";

/// Operation trait for `llvm.switch`.
pub trait SwitchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        SWITCH_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `default_operands` operands.
    fn default_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let default_operand_count = self
            .attribute("operand_segment_sizes")
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| Vec::<i32>::from(attribute)[1])
            .unwrap_or_else(|| panic!("invalid 'operand_segment_sizes' attribute in `llvm.switch`"));
        self.operand_values().skip(1).take(default_operand_count as usize).collect()
    }

    /// Returns the `case_operands` operands.
    fn case_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let default_operand_count = self
            .attribute("operand_segment_sizes")
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| Vec::<i32>::from(attribute)[1])
            .unwrap_or_else(|| panic!("invalid 'operand_segment_sizes' attribute in `llvm.switch`"));
        self.operand_values().skip(1 + default_operand_count as usize).collect()
    }

    /// Returns the default destination block.
    fn default_destination(&self) -> BlockRef<'o, 'c, 't> {
        self.successor(0).unwrap()
    }

    /// Returns the case destination blocks.
    fn case_destinations(&self) -> Vec<BlockRef<'o, 'c, 't>> {
        self.successors().skip(1).collect()
    }

    /// Returns the optional `case_values` attribute.
    fn case_values(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("case_values")
    }

    /// Returns the `case_operand_segments` attribute.
    fn case_operand_segments(&self) -> AttributeRef<'c, 't> {
        self.attribute("case_operand_segments").unwrap()
    }

    /// Returns the optional `branch_weights` attribute.
    fn branch_weights(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("branch_weights")
    }
}

mlir_op!(Switch);

/// Constructs a new detached `llvm.switch` operation.
pub fn switch<'b, 'c: 'b, 't: 'c, V1: Value<'c, 'c, 't>, B: Block<'b, 'c, 't>, L: Location<'c, 't>>(
    value: V1,
    default_destination: &B,
    default_operands: &[ValueRef<'c, 'c, 't>],
    case_destinations: &[&B],
    case_operands: &[ValueRef<'c, 'c, 't>],
    case_values: Option<AttributeRef<'c, 't>>,
    case_operand_segments: AttributeRef<'c, 't>,
    branch_weights: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedSwitchOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(SWITCH_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_operands(default_operands);
    builder = builder.add_operands(case_operands);
    builder = builder.add_successor(default_destination);
    builder = builder.add_successors(case_destinations);
    if let Some(case_values) = case_values {
        builder = builder.add_attribute("case_values", case_values);
    }
    builder = builder.add_attribute("case_operand_segments", case_operand_segments);
    builder = builder.add_attribute(
        "operand_segment_sizes",
        context
            .dense_i32_array_attribute(&[1, default_operands.len() as i32, case_operands.len() as i32])
            .unwrap(),
    );
    if let Some(branch_weights) = branch_weights {
        builder = builder.add_attribute("branch_weights", branch_weights);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::switch`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::dialects::llvm::Linkage;
    use crate::dialects::llvm::operations::aggregates::landing_pad;
    use crate::dialects::llvm::operations::core::r#return as llvm_return;
    use crate::dialects::llvm::operations::symbols::func as llvm_func;
    use crate::{Block, BlockRef, Context, Operation, Region, Type, TypeRef, ValueRef};

    use super::*;

    #[test]
    fn test_block_address() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        context.load_dialect(DialectHandle::llvm());
        let pointer_type = context.llvm_pointer_type(0);
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let tag = context.parse_attribute("#llvm.blocktag<id = 1>").unwrap();
            block.append_operation(block_tag(tag, location));
            block.append_operation(llvm_return(None, location));
            llvm_func(
                context.string_attribute("target").as_ref(),
                None,
                context
                    .type_attribute(context.llvm_function_type(context.llvm_void_type(), &[] as &[TypeRef], false))
                    .as_ref(),
                Some(context.llvm_linkage_attribute(Linkage::Internal).as_ref()),
                block.into(),
                location,
            )
        });
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let block_addr = context.parse_attribute("#llvm.blockaddress<function = @target, tag = <id = 1>>").unwrap();
            let op = block_address(pointer_type.as_ref(), block_addr, location);
            assert_eq!(op.operation_name(), "llvm.blockaddress");
            assert_eq!(op.block_addr(), block_addr);
            assert_eq!(op.output_type(), pointer_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_blockaddress_test",
                func::FuncAttributes { arguments: vec![], results: vec![pointer_type.into()], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  llvm.func internal @target() {
                    llvm.blocktag <id = 1>
                    llvm.return
                  }
                  func.func @llvm_blockaddress_test() -> !llvm.ptr {
                    %0 = llvm.blockaddress <function = @target, tag = <id = 1>> : !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_block_tag() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        context.load_dialect(DialectHandle::llvm());
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let tag = context.parse_attribute("#llvm.blocktag<id = 1>").unwrap();
            let op = block_tag(tag, location);
            assert_eq!(op.operation_name(), "llvm.blocktag");
            assert_eq!(op.tag(), tag);
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<ValueRef, _>(&[], location));
            func::func(
                "llvm_blocktag_test",
                func::FuncAttributes { arguments: vec![], results: vec![], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_blocktag_test() {
                    llvm.blocktag <id = 1>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_cond_br() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut entry_block = context.block(&[
                (i1_type.as_ref(), location),
                (i32_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let mut true_block = context.block(&[(i32_type.as_ref(), location)]);
            let mut false_block = context.block(&[(i32_type.as_ref(), location)]);
            let condition = entry_block.argument(0).unwrap();
            let true_value = entry_block.argument(1).unwrap();
            let false_value = entry_block.argument(2).unwrap();
            let op = cond_br(
                condition,
                &true_block,
                &false_block,
                &[true_value.into()],
                &[false_value.into()],
                Some(context.dense_i32_array_attribute(&[13, 21]).unwrap().as_ref()),
                None,
                location,
            );
            assert_eq!(op.operation_name(), "llvm.cond_br");
            assert_eq!(op.condition(), condition);
            assert_eq!(op.true_destination(), BlockRef::from(&true_block));
            assert_eq!(op.false_destination(), BlockRef::from(&false_block));
            assert_eq!(op.true_destination_operands(), vec![true_value]);
            assert_eq!(op.false_destination_operands(), vec![false_value]);
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 0);
            assert_eq!(op.successors().count(), 2);
            entry_block.append_operation(op);
            true_block.append_operation(func::r#return(&[true_block.argument(0).unwrap()], location));
            false_block.append_operation(func::r#return(&[false_block.argument(0).unwrap()], location));
            let mut region = context.region();
            region.append_block(entry_block);
            region.append_block(true_block);
            region.append_block(false_block);
            func::func(
                "llvm_cond_br_test",
                func::FuncAttributes {
                    arguments: vec![i1_type.into(), i32_type.into(), i32_type.into()],
                    results: vec![i32_type.into()],
                    ..Default::default()
                },
                region,
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_cond_br_test(%arg0: i1, %arg1: i32, %arg2: i32) -> i32 {
                    llvm.cond_br %arg0 weights([13, 21]), ^bb1(%arg1 : i32), ^bb2(%arg2 : i32)
                  ^bb1(%0: i32):  // pred: ^bb0
                    return %0 : i32
                  ^bb2(%1: i32):  // pred: ^bb0
                    return %1 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_indirect_br() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut entry_block = context.block(&[(pointer_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let mut destination_block = context.block(&[(i32_type.as_ref(), location)]);
            let address = entry_block.argument(0).unwrap();
            let forwarded = entry_block.argument(1).unwrap();
            let op = indirect_br(
                address,
                &[&destination_block],
                &[forwarded.into()],
                context.dense_i32_array_attribute(&[1]).unwrap().as_ref(),
                location,
            );
            assert_eq!(op.operation_name(), "llvm.indirectbr");
            assert_eq!(op.address(), address);
            assert_eq!(op.destinations(), vec![BlockRef::from(&destination_block)]);
            assert_eq!(op.successor_operands(), vec![forwarded]);
            entry_block.append_operation(op);
            destination_block.append_operation(func::r#return(&[destination_block.argument(0).unwrap()], location));
            let mut region = context.region();
            region.append_block(entry_block);
            region.append_block(destination_block);
            func::func(
                "llvm_indirectbr_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into(), i32_type.into()],
                    results: vec![i32_type.into()],
                    ..Default::default()
                },
                region,
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_indirectbr_test(%arg0: !llvm.ptr, %arg1: i32) -> i32 {
                    llvm.indirectbr %arg0 : !llvm.ptr, [
                    ^bb1(%arg1 : i32)
                    ]
                  ^bb1(%0: i32):  // pred: ^bb0
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_invoke() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i8_type = context.signless_integer_type(8);
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0);
        let exception_type = context.llvm_literal_struct_type(&[pointer_type.as_ref(), i8_type.as_ref()], false);
        let function_type = context.llvm_function_type(i32_type, &[i32_type], false);
        module.body().append_operation(llvm_func(
            context.string_attribute("callee").as_ref(),
            None,
            context.type_attribute(function_type).as_ref(),
            Some(context.llvm_linkage_attribute(Linkage::External).as_ref()),
            context.region(),
            location,
        ));
        module.body().append_operation({
            let mut entry_block = context.block(&[(i32_type.as_ref(), location)]);
            let mut normal_block = context.block(&[(i32_type.as_ref(), location)]);
            let mut unwind_block = context.block(&[(i32_type.as_ref(), location)]);
            let argument = entry_block.argument(0).unwrap();
            let op = invoke(
                &[argument.into()],
                &normal_block,
                &[argument.into()],
                &unwind_block,
                &[argument.into()],
                &[],
                i32_type.as_ref(),
                None,
                Some(context.flat_symbol_ref_attribute("callee").as_ref()),
                None,
                None,
                None,
                None,
                None,
                None,
                location,
            );
            assert_eq!(op.operation_name(), "llvm.invoke");
            assert_eq!(op.callee_operands(), vec![argument]);
            assert_eq!(op.normal_destination(), BlockRef::from(&normal_block));
            assert_eq!(op.unwind_destination(), BlockRef::from(&unwind_block));
            assert_eq!(op.normal_destination_operands(), vec![argument]);
            assert_eq!(op.unwind_destination_operands(), vec![argument]);
            assert!(op.op_bundle_operands().is_empty());
            assert_eq!(op.output_type(), i32_type);
            entry_block.append_operation(op);
            normal_block.append_operation(func::r#return(&[normal_block.argument(0).unwrap()], location));
            unwind_block.append_operation(landing_pad(&[], exception_type.as_ref(), true, location));
            unwind_block.append_operation(func::r#return(&[unwind_block.argument(0).unwrap()], location));
            let mut region = context.region();
            region.append_block(entry_block);
            region.append_block(normal_block);
            region.append_block(unwind_block);
            func::func(
                "llvm_invoke_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into()],
                    results: vec![i32_type.into()],
                    ..Default::default()
                },
                region,
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  llvm.func @callee(i32) -> i32
                  func.func @llvm_invoke_test(%arg0: i32) -> i32 {
                    %0 = llvm.invoke @callee(%arg0) to ^bb1(%arg0 : i32) unwind ^bb2(%arg0 : i32) : (i32) -> i32
                  ^bb1(%1: i32):  // pred: ^bb0
                    return %1 : i32
                  ^bb2(%2: i32):  // pred: ^bb0
                    %3 = llvm.landingpad cleanup : !llvm.struct<(ptr, i8)>
                    return %2 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_resume() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i8_type = context.signless_integer_type(8);
        let pointer_type = context.llvm_pointer_type(0);
        let exception_type = context.llvm_literal_struct_type(&[pointer_type.as_ref(), i8_type.as_ref()], false);
        module.body().append_operation({
            let mut block = context.block(&[(exception_type.as_ref(), location)]);
            let op = resume(block.argument(0).unwrap(), location);
            assert_eq!(op.operation_name(), "llvm.resume");
            assert_eq!(op.value(), block.argument(0).unwrap());
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            func::func(
                "llvm_resume_test",
                func::FuncAttributes { arguments: vec![exception_type.into()], results: vec![], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_resume_test(%arg0: !llvm.struct<(ptr, i8)>) {
                    llvm.resume %arg0 : !llvm.struct<(ptr, i8)>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_switch() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut entry_block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let mut default_block = context.block(&[(i32_type.as_ref(), location)]);
            let mut case_block = context.block(&[(i32_type.as_ref(), location)]);
            let value = entry_block.argument(0).unwrap();
            let forwarded = entry_block.argument(1).unwrap();
            let case_values = context
                .dense_elements_attribute(
                    context.tensor_type(i32_type, &[crate::Size::Static(1)], None, location).unwrap(),
                    &[context.integer_attribute(i32_type, 7)],
                )
                .unwrap();
            let op = switch(
                value,
                &default_block,
                &[forwarded.into()],
                &[&case_block],
                &[forwarded.into()],
                Some(case_values.as_ref()),
                context.dense_i32_array_attribute(&[1]).unwrap().as_ref(),
                None,
                location,
            );
            assert_eq!(op.operation_name(), "llvm.switch");
            assert_eq!(op.value(), value);
            assert_eq!(op.default_destination(), BlockRef::from(&default_block));
            assert_eq!(op.case_destinations(), vec![BlockRef::from(&case_block)]);
            assert_eq!(op.default_operands(), vec![forwarded]);
            assert_eq!(op.case_operands(), vec![forwarded]);
            entry_block.append_operation(op);
            default_block.append_operation(func::r#return(&[default_block.argument(0).unwrap()], location));
            case_block.append_operation(func::r#return(&[case_block.argument(0).unwrap()], location));
            let mut region = context.region();
            region.append_block(entry_block);
            region.append_block(default_block);
            region.append_block(case_block);
            func::func(
                "llvm_switch_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
                    results: vec![i32_type.into()],
                    ..Default::default()
                },
                region,
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_switch_test(%arg0: i32, %arg1: i32) -> i32 {
                    llvm.switch %arg0 : i32, ^bb1(%arg1 : i32) [
                      7: ^bb2(%arg1 : i32)
                    ]
                  ^bb1(%0: i32):  // pred: ^bb0
                    return %0 : i32
                  ^bb2(%1: i32):  // pred: ^bb0
                    return %1 : i32
                  }
                }
            "},
        );
    }
}
