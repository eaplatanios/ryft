use crate::{
    Attribute, AttributeRef, Block, BlockRef, DetachedOp, DialectHandle, Error, Location, Operation, OperationBuilder,
    TypeRef, Value, ValueRef, mlir_op,
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
    fn block_addr(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("block_addr")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "block_addr",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(BlockAddress);

/// Constructs a new detached `llvm.blockaddress` operation.
pub fn block_address<'c, 't: 'c, L: Location<'c, 't>>(
    result_type: TypeRef<'c, 't>,
    block_addr: AttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedBlockAddressOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(BLOCK_ADDRESS_OPERATION_NAME, location);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("block_addr", block_addr);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::block_address`"))
    })
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
    fn tag(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("tag")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "tag",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(BlockTag);

/// Constructs a new detached `llvm.blocktag` operation.
pub fn block_tag<'c, 't: 'c, L: Location<'c, 't>>(
    tag: AttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedBlockTagOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(BLOCK_TAG_OPERATION_NAME, location);
    builder = builder.add_attribute("tag", tag);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::block_tag`"))
    })
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
    fn condition(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `true_destination_operands` operands.
    fn true_destination_operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let true_destination_operand_count =
            self.dense_integer_32_array_attribute_usize_value("operand_segment_sizes", 1)?;
        self.operand_values().skip(1).take(true_destination_operand_count).collect()
    }

    /// Returns the `false_destination_operands` operands.
    fn false_destination_operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let true_destination_operand_count =
            self.dense_integer_32_array_attribute_usize_value("operand_segment_sizes", 1)?;
        self.operand_values().skip(1 + true_destination_operand_count).collect()
    }

    /// Returns the true destination block.
    fn true_destination(&self) -> Result<BlockRef<'o, 'c, 't>, Error> {
        self.successor(0)
    }

    /// Returns the false destination block.
    fn false_destination(&self) -> Result<BlockRef<'o, 'c, 't>, Error> {
        self.successor(1)
    }

    /// Returns the optional `branch_weights` attribute.
    fn branch_weights(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("branch_weights")
    }

    /// Returns the optional `loop_annotation` attribute.
    fn loop_annotation(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
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
) -> Result<DetachedCondBrOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(COND_BR_OPERATION_NAME, location);
    builder = builder.add_operand(condition);
    builder = builder.add_operands(true_destination_operands);
    builder = builder.add_operands(false_destination_operands);
    builder = builder.add_successor(true_destination);
    builder = builder.add_successor(false_destination);
    builder = builder.add_attribute(
        "operand_segment_sizes",
        context.dense_i32_array_attribute(&[
            1,
            true_destination_operands.len() as i32,
            false_destination_operands.len() as i32,
        ])?,
    );
    if let Some(branch_weights) = branch_weights {
        builder = builder.add_attribute("branch_weights", branch_weights);
    }
    if let Some(loop_annotation) = loop_annotation {
        builder = builder.add_attribute("loop_annotation", loop_annotation);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::cond_br`"))
    })
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
    fn address(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `successor_operands` operands.
    fn successor_operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().skip(1).collect()
    }

    /// Returns the destination blocks.
    fn destinations(&self) -> Result<Vec<BlockRef<'o, 'c, 't>>, Error> {
        self.successors().collect()
    }

    /// Returns the `indbr_operand_segments` attribute.
    fn indbr_operand_segments(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("indbr_operand_segments")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "indbr_operand_segments",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
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
) -> Result<DetachedIndirectBrOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(INDIRECT_BR_OPERATION_NAME, location);
    builder = builder.add_operand(address);
    builder = builder.add_operands(successor_operands);
    builder = builder.add_successors(destinations);
    builder = builder.add_attribute("indbr_operand_segments", indbr_operand_segments);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::indirect_br`"))
    })
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
    fn callee_operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let callee_operand_count = self.dense_integer_32_array_attribute_usize_value("operand_segment_sizes", 0)?;
        self.operand_values().take(callee_operand_count).collect()
    }

    /// Returns the `normal_destination_operands` operands.
    fn normal_destination_operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let callee_operand_count = self.dense_integer_32_array_attribute_usize_value("operand_segment_sizes", 0)?;
        let normal_destination_operand_count =
            self.dense_integer_32_array_attribute_usize_value("operand_segment_sizes", 1)?;
        self.operand_values().skip(callee_operand_count).take(normal_destination_operand_count).collect()
    }

    /// Returns the `unwind_destination_operands` operands.
    fn unwind_destination_operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let callee_operand_count = self.dense_integer_32_array_attribute_usize_value("operand_segment_sizes", 0)?;
        let normal_destination_operand_count =
            self.dense_integer_32_array_attribute_usize_value("operand_segment_sizes", 1)?;
        let unwind_destination_operand_count =
            self.dense_integer_32_array_attribute_usize_value("operand_segment_sizes", 2)?;
        self.operand_values()
            .skip(callee_operand_count + normal_destination_operand_count)
            .take(unwind_destination_operand_count)
            .collect()
    }

    /// Returns the `op_bundle_operands` operands.
    fn op_bundle_operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let callee_operand_count = self.dense_integer_32_array_attribute_usize_value("operand_segment_sizes", 0)?;
        let normal_destination_operand_count =
            self.dense_integer_32_array_attribute_usize_value("operand_segment_sizes", 1)?;
        let unwind_destination_operand_count =
            self.dense_integer_32_array_attribute_usize_value("operand_segment_sizes", 2)?;
        self.operand_values()
            .skip(callee_operand_count + normal_destination_operand_count + unwind_destination_operand_count)
            .collect()
    }

    /// Returns the normal destination block.
    fn normal_destination(&self) -> Result<BlockRef<'o, 'c, 't>, Error> {
        self.successor(0)
    }

    /// Returns the unwind destination block.
    fn unwind_destination(&self) -> Result<BlockRef<'o, 'c, 't>, Error> {
        self.successor(1)
    }

    /// Returns the optional `var_callee_type` attribute.
    fn var_callee_type(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("var_callee_type")
    }

    /// Returns the optional `callee` attribute.
    fn callee(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("callee")
    }

    /// Returns the optional `arg_attrs` attribute.
    fn arg_attrs(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("arg_attrs")
    }

    /// Returns the optional `res_attrs` attribute.
    fn res_attrs(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("res_attrs")
    }

    /// Returns the optional `branch_weights` attribute.
    fn branch_weights(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("branch_weights")
    }

    /// Returns the optional `calling_convention` attribute.
    fn calling_convention(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("CConv")
    }

    /// Returns the optional `op_bundle_sizes` attribute.
    fn op_bundle_sizes(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("op_bundle_sizes")
    }

    /// Returns the optional `op_bundle_tags` attribute.
    fn op_bundle_tags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("op_bundle_tags")
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedInvokeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
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
        context.dense_i32_array_attribute(&[
            callee_operands.len() as i32,
            normal_destination_operands.len() as i32,
            unwind_destination_operands.len() as i32,
            op_bundle_operands.len() as i32,
        ])?,
    );
    let empty_op_bundle_sizes = context.dense_i32_array_attribute(&[])?;
    builder =
        builder.add_attribute("op_bundle_sizes", op_bundle_sizes.unwrap_or_else(|| empty_op_bundle_sizes.as_ref()));
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
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::invoke`"))
    })
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
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(Resume);

/// Constructs a new detached `llvm.resume` operation.
pub fn resume<'c, 't: 'c, V1: Value<'c, 'c, 't>, L: Location<'c, 't>>(
    value: V1,
    location: L,
) -> Result<DetachedResumeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(RESUME_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::resume`"))
    })
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
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `default_operands` operands.
    fn default_operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let default_operand_count = self.dense_integer_32_array_attribute_usize_value("operand_segment_sizes", 1)?;
        self.operand_values().skip(1).take(default_operand_count).collect()
    }

    /// Returns the `case_operands` operands.
    fn case_operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let default_operand_count = self.dense_integer_32_array_attribute_usize_value("operand_segment_sizes", 1)?;
        self.operand_values().skip(1 + default_operand_count).collect()
    }

    /// Returns the default destination block.
    fn default_destination(&self) -> Result<BlockRef<'o, 'c, 't>, Error> {
        self.successor(0)
    }

    /// Returns the case destination blocks.
    fn case_destinations(&self) -> Result<Vec<BlockRef<'o, 'c, 't>>, Error> {
        self.successors().skip(1).collect()
    }

    /// Returns the optional `case_values` attribute.
    fn case_values(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("case_values")
    }

    /// Returns the `case_operand_segments` attribute.
    fn case_operand_segments(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("case_operand_segments")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "case_operand_segments",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the optional `branch_weights` attribute.
    fn branch_weights(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
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
) -> Result<DetachedSwitchOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
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
        context.dense_i32_array_attribute(&[1, default_operands.len() as i32, case_operands.len() as i32])?,
    );
    if let Some(branch_weights) = branch_weights {
        builder = builder.add_attribute("branch_weights", branch_weights);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::switch`"))
    })
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
    use crate::{Block, Context, Operation, Region, Type, TypeRef, ValueRef};

    use super::*;

    #[test]
    fn test_block_address() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let tag = context.parse_attribute("#llvm.blocktag<id = 1>").unwrap();
                block.append_operation(block_tag(tag, location).unwrap()).unwrap();
                block.append_operation(llvm_return(None, location).unwrap()).unwrap();
                llvm_func(
                    context.string_attribute("target").as_ref(),
                    None,
                    context
                        .type_attribute(
                            context
                                .llvm_function_type(context.llvm_void_type().unwrap(), &[] as &[TypeRef], false)
                                .unwrap(),
                        )
                        .as_ref(),
                    Some(context.llvm_linkage_attribute(Linkage::Internal).unwrap().as_ref()),
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let block_addr =
                    context.parse_attribute("#llvm.blockaddress<function = @target, tag = <id = 1>>").unwrap();
                let op = block_address(pointer_type.as_ref(), block_addr, location).unwrap();
                assert_eq!(op.operation_name(), "llvm.blockaddress");
                assert_eq!(op.block_addr().unwrap(), block_addr);
                assert_eq!(op.output_type().unwrap(), pointer_type);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_blockaddress_test",
                    func::FuncAttributes {
                        arguments: vec![],
                        results: vec![pointer_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
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
        let module = context.module(location).unwrap();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let tag = context.parse_attribute("#llvm.blocktag<id = 1>").unwrap();
                let op = block_tag(tag, location).unwrap();
                assert_eq!(op.operation_name(), "llvm.blocktag");
                assert_eq!(op.tag().unwrap(), tag);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location).unwrap()).unwrap();
                func::func(
                    "llvm_blocktag_test",
                    func::FuncAttributes { arguments: vec![], results: vec![], ..Default::default() },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
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
        let module = context.module(location).unwrap();
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
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
                )
                .unwrap();
                assert_eq!(op.operation_name(), "llvm.cond_br");
                assert_eq!(op.condition().unwrap(), condition);
                assert_eq!(op.true_destination().unwrap(), true_block.as_ref());
                assert_eq!(op.false_destination().unwrap(), false_block.as_ref());
                assert_eq!(op.true_destination_operands().unwrap(), vec![true_value]);
                assert_eq!(op.false_destination_operands().unwrap(), vec![false_value]);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(op.successors().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                entry_block.append_operation(op).unwrap();
                true_block
                    .append_operation(func::r#return(&[true_block.argument(0).unwrap()], location).unwrap())
                    .unwrap();
                false_block
                    .append_operation(func::r#return(&[false_block.argument(0).unwrap()], location).unwrap())
                    .unwrap();
                let mut region = context.region();
                region.append_block(entry_block).unwrap();
                region.append_block(true_block).unwrap();
                region.append_block(false_block).unwrap();
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
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
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
        let module = context.module(location).unwrap();
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut entry_block =
                    context.block(&[(pointer_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let mut destination_block = context.block(&[(i32_type.as_ref(), location)]);
                let address = entry_block.argument(0).unwrap();
                let forwarded = entry_block.argument(1).unwrap();
                let op = indirect_br(
                    address,
                    &[&destination_block],
                    &[forwarded.into()],
                    context.dense_i32_array_attribute(&[1]).unwrap().as_ref(),
                    location,
                )
                .unwrap();
                assert_eq!(op.operation_name(), "llvm.indirectbr");
                assert_eq!(op.address().unwrap(), address);
                assert_eq!(op.destinations().unwrap(), vec![destination_block.as_ref()]);
                assert_eq!(op.successor_operands().unwrap(), vec![forwarded]);
                entry_block.append_operation(op).unwrap();
                destination_block
                    .append_operation(func::r#return(&[destination_block.argument(0).unwrap()], location).unwrap())
                    .unwrap();
                let mut region = context.region();
                region.append_block(entry_block).unwrap();
                region.append_block(destination_block).unwrap();
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
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
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
        let module = context.module(location).unwrap();
        let i8_type = context.signless_integer_type(8);
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let exception_type =
            context.llvm_literal_struct_type(&[pointer_type.as_ref(), i8_type.as_ref()], false).unwrap();
        let function_type = context.llvm_function_type(i32_type, &[i32_type], false).unwrap();
        module
            .body()
            .unwrap()
            .append_operation(
                llvm_func(
                    context.string_attribute("callee").as_ref(),
                    None,
                    context.type_attribute(function_type).as_ref(),
                    Some(context.llvm_linkage_attribute(Linkage::External).unwrap().as_ref()),
                    context.region(),
                    location,
                )
                .unwrap(),
            )
            .unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
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
                )
                .unwrap();
                assert_eq!(op.operation_name(), "llvm.invoke");
                assert_eq!(op.callee_operands().unwrap(), vec![argument]);
                assert_eq!(op.normal_destination().unwrap(), normal_block.as_ref());
                assert_eq!(op.unwind_destination().unwrap(), unwind_block.as_ref());
                assert_eq!(op.normal_destination_operands().unwrap(), vec![argument]);
                assert_eq!(op.unwind_destination_operands().unwrap(), vec![argument]);
                assert!(op.op_bundle_operands().unwrap().is_empty());
                assert_eq!(op.output_type().unwrap(), i32_type);
                entry_block.append_operation(op).unwrap();
                normal_block
                    .append_operation(func::r#return(&[normal_block.argument(0).unwrap()], location).unwrap())
                    .unwrap();
                unwind_block
                    .append_operation(landing_pad(&[], exception_type.as_ref(), true, location).unwrap())
                    .unwrap();
                unwind_block
                    .append_operation(func::r#return(&[unwind_block.argument(0).unwrap()], location).unwrap())
                    .unwrap();
                let mut region = context.region();
                region.append_block(entry_block).unwrap();
                region.append_block(normal_block).unwrap();
                region.append_block(unwind_block).unwrap();
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
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
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
        let module = context.module(location).unwrap();
        let i8_type = context.signless_integer_type(8);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let exception_type =
            context.llvm_literal_struct_type(&[pointer_type.as_ref(), i8_type.as_ref()], false).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(exception_type.as_ref(), location)]);
                let op = resume(block.argument(0).unwrap(), location).unwrap();
                assert_eq!(op.operation_name(), "llvm.resume");
                assert_eq!(op.value().unwrap(), block.argument(0).unwrap());
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                func::func(
                    "llvm_resume_test",
                    func::FuncAttributes {
                        arguments: vec![exception_type.into()],
                        results: vec![],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
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
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
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
                )
                .unwrap();
                assert_eq!(op.operation_name(), "llvm.switch");
                assert_eq!(op.value().unwrap(), value);
                assert_eq!(op.default_destination().unwrap(), default_block.as_ref());
                assert_eq!(op.case_destinations().unwrap(), vec![case_block.as_ref()]);
                assert_eq!(op.default_operands().unwrap(), vec![forwarded]);
                assert_eq!(op.case_operands().unwrap(), vec![forwarded]);
                entry_block.append_operation(op).unwrap();
                default_block
                    .append_operation(func::r#return(&[default_block.argument(0).unwrap()], location).unwrap())
                    .unwrap();
                case_block
                    .append_operation(func::r#return(&[case_block.argument(0).unwrap()], location).unwrap())
                    .unwrap();
                let mut region = context.region();
                region.append_block(entry_block).unwrap();
                region.append_block(default_block).unwrap();
                region.append_block(case_block).unwrap();
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
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
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
