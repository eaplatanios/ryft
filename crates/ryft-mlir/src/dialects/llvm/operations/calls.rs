use crate::{
    Attribute, AttributeRef, DenseInteger32ArrayAttributeRef, DetachedOp, DialectHandle, Location, Operation,
    OperationBuilder, TypeRef, ValueRef, mlir_op,
};

/// Canonical MLIR operation name for [`CallIntrinsicOperation`].
pub const CALL_INTRINSIC_OPERATION_NAME: &str = "llvm.call_intrinsic";

/// Operation trait for `llvm.call_intrinsic`.
pub trait CallIntrinsicOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CALL_INTRINSIC_OPERATION_NAME
    }

    /// Returns the `arguments` operands.
    fn arguments(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let argument_count = self
            .attribute("operand_segment_sizes")
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| Vec::<i32>::from(attribute)[0])
            .unwrap_or_else(|| panic!("invalid 'operand_segment_sizes' attribute in `llvm.call_intrinsic`"));
        self.operand_values().take(argument_count as usize).collect()
    }

    /// Returns the `op_bundle_operands` operands.
    fn op_bundle_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let argument_count = self
            .attribute("operand_segment_sizes")
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| Vec::<i32>::from(attribute)[0])
            .unwrap_or_else(|| panic!("invalid 'operand_segment_sizes' attribute in `llvm.call_intrinsic`"));
        self.operand_values().skip(argument_count as usize).collect()
    }

    /// Returns the `intrin` attribute.
    fn intrin(&self) -> AttributeRef<'c, 't> {
        self.attribute("intrin").unwrap()
    }

    /// Returns the optional `fastmath_flags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmath_flags")
    }

    /// Returns the optional `op_bundle_sizes` attribute.
    fn op_bundle_sizes(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("op_bundle_sizes")
    }

    /// Returns the optional `op_bundle_tags` attribute.
    fn op_bundle_tags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("op_bundle_tags")
    }

    /// Returns the optional `arg_attrs` attribute.
    fn arg_attrs(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("arg_attrs")
    }

    /// Returns the optional `res_attrs` attribute.
    fn res_attrs(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("res_attrs")
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(CallIntrinsic);

/// Constructs a new detached `llvm.call_intrinsic` operation.
pub fn call_intrinsic<'c, 't: 'c, L: Location<'c, 't>>(
    arguments: &[ValueRef<'c, 'c, 't>],
    op_bundle_operands: &[ValueRef<'c, 'c, 't>],
    result_type: TypeRef<'c, 't>,
    intrin: AttributeRef<'c, 't>,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    op_bundle_sizes: Option<AttributeRef<'c, 't>>,
    op_bundle_tags: Option<AttributeRef<'c, 't>>,
    arg_attrs: Option<AttributeRef<'c, 't>>,
    res_attrs: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedCallIntrinsicOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CALL_INTRINSIC_OPERATION_NAME, location);
    builder = builder.add_operands(arguments);
    builder = builder.add_operands(op_bundle_operands);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("intrin", intrin);
    builder = builder.add_attribute(
        "operand_segment_sizes",
        context
            .dense_i32_array_attribute(&[arguments.len() as i32, op_bundle_operands.len() as i32])
            .unwrap(),
    );
    builder = builder.add_attribute(
        "op_bundle_sizes",
        op_bundle_sizes.unwrap_or_else(|| context.dense_i32_array_attribute(&[]).unwrap().as_ref()),
    );
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmath_flags", fastmath_flags);
    }
    if let Some(op_bundle_tags) = op_bundle_tags {
        builder = builder.add_attribute("op_bundle_tags", op_bundle_tags);
    }
    if let Some(arg_attrs) = arg_attrs {
        builder = builder.add_attribute("arg_attrs", arg_attrs);
    }
    if let Some(res_attrs) = res_attrs {
        builder = builder.add_attribute("res_attrs", res_attrs);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::call_intrinsic`")
}

/// Canonical MLIR operation name for [`CallOperation`].
pub const CALL_OPERATION_NAME: &str = "llvm.call";

/// Operation trait for `llvm.call`.
pub trait CallOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CALL_OPERATION_NAME
    }

    /// Returns the `callee_operands` operands.
    fn callee_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(0).collect()
    }

    /// Returns the optional `callee` attribute.
    fn callee(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("callee")
    }

    /// Returns the optional `var_callee_type` attribute.
    fn var_callee_type(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("var_callee_type")
    }

    /// Returns the optional `fastmath_flags` attribute.
    fn fastmath_flags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("fastmath_flags")
    }

    /// Returns the optional `calling_convention` attribute.
    fn calling_convention(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("CConv")
    }

    /// Returns the `op_bundle_operands` operands.
    fn op_bundle_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let callee_operand_count = self
            .attribute("operand_segment_sizes")
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| Vec::<i32>::from(attribute)[0])
            .unwrap_or_else(|| panic!("invalid 'operand_segment_sizes' attribute in `llvm.call`"));
        self.operand_values().skip(callee_operand_count as usize).collect()
    }

    /// Returns the `op_bundle_sizes` attribute.
    fn op_bundle_sizes(&self) -> AttributeRef<'c, 't> {
        self.attribute("op_bundle_sizes").unwrap()
    }

    /// Returns the optional `op_bundle_tags` attribute.
    fn op_bundle_tags(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("op_bundle_tags")
    }

    /// Returns the optional `arg_attrs` attribute.
    fn arg_attrs(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("arg_attrs")
    }

    /// Returns the optional `res_attrs` attribute.
    fn res_attrs(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("res_attrs")
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Call);

/// Constructs a new detached `llvm.call` operation.
pub fn call<'c, 't: 'c, L: Location<'c, 't>>(
    callee_operands: &[ValueRef<'c, 'c, 't>],
    op_bundle_operands: &[ValueRef<'c, 'c, 't>],
    result_type: TypeRef<'c, 't>,
    var_callee_type: Option<AttributeRef<'c, 't>>,
    callee: Option<AttributeRef<'c, 't>>,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    calling_convention: Option<AttributeRef<'c, 't>>,
    op_bundle_sizes: Option<AttributeRef<'c, 't>>,
    op_bundle_tags: Option<AttributeRef<'c, 't>>,
    arg_attrs: Option<AttributeRef<'c, 't>>,
    res_attrs: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedCallOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CALL_OPERATION_NAME, location);
    builder = builder.add_operands(callee_operands);
    builder = builder.add_operands(op_bundle_operands);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute(
        "operand_segment_sizes",
        context
            .dense_i32_array_attribute(&[callee_operands.len() as i32, op_bundle_operands.len() as i32])
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
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmath_flags", fastmath_flags);
    }
    if let Some(calling_convention) = calling_convention {
        builder = builder.add_attribute("CConv", calling_convention);
    }
    if let Some(op_bundle_tags) = op_bundle_tags {
        builder = builder.add_attribute("op_bundle_tags", op_bundle_tags);
    }
    if let Some(arg_attrs) = arg_attrs {
        builder = builder.add_attribute("arg_attrs", arg_attrs);
    }
    if let Some(res_attrs) = res_attrs {
        builder = builder.add_attribute("res_attrs", res_attrs);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::call`")
}

/// Canonical MLIR operation name for [`InlineAsmOperation`].
pub const INLINE_ASM_OPERATION_NAME: &str = "llvm.inline_asm";

/// Operation trait for `llvm.inline_asm`.
pub trait InlineAsmOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        INLINE_ASM_OPERATION_NAME
    }

    /// Returns the `operands` operands.
    fn operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(0).collect()
    }

    /// Returns the `asm_string` attribute.
    fn asm_string(&self) -> AttributeRef<'c, 't> {
        self.attribute("asm_string").unwrap()
    }

    /// Returns the `constraints` attribute.
    fn constraints(&self) -> AttributeRef<'c, 't> {
        self.attribute("constraints").unwrap()
    }

    /// Returns whether the `has_side_effects` unit attribute is present.
    fn has_side_effects(&self) -> bool {
        self.has_attribute("has_side_effects")
    }

    /// Returns whether the `is_align_stack` unit attribute is present.
    fn is_align_stack(&self) -> bool {
        self.has_attribute("is_align_stack")
    }

    /// Returns the optional `tail_call_kind` attribute.
    fn tail_call_kind(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("tail_call_kind")
    }

    /// Returns the optional `asm_dialect` attribute.
    fn asm_dialect(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("asm_dialect")
    }

    /// Returns the optional `operand_attrs` attribute.
    fn operand_attrs(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute("operand_attrs")
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(InlineAsm);

/// Constructs a new detached `llvm.inline_asm` operation.
pub fn inline_asm<'c, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'c, 'c, 't>],
    result_type: TypeRef<'c, 't>,
    asm_string: AttributeRef<'c, 't>,
    constraints: AttributeRef<'c, 't>,
    has_side_effects: bool,
    is_align_stack: bool,
    tail_call_kind: Option<AttributeRef<'c, 't>>,
    asm_dialect: Option<AttributeRef<'c, 't>>,
    operand_attrs: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedInlineAsmOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(INLINE_ASM_OPERATION_NAME, location);
    builder = builder.add_operands(operands);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("asm_string", asm_string);
    builder = builder.add_attribute("constraints", constraints);
    if has_side_effects {
        builder = builder.add_attribute("has_side_effects", context.unit_attribute());
    }
    if is_align_stack {
        builder = builder.add_attribute("is_align_stack", context.unit_attribute());
    }
    if let Some(tail_call_kind) = tail_call_kind {
        builder = builder.add_attribute("tail_call_kind", tail_call_kind);
    }
    if let Some(asm_dialect) = asm_dialect {
        builder = builder.add_attribute("asm_dialect", asm_dialect);
    }
    if let Some(operand_attrs) = operand_attrs {
        builder = builder.add_attribute("operand_attrs", operand_attrs);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::inline_asm`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::dialects::llvm::Linkage;
    use crate::dialects::llvm::operations::symbols::func as llvm_func;
    use crate::{Block, Context, Operation, Type};

    use super::*;

    #[test]
    fn test_call_intrinsic() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let op = call_intrinsic(
                &[block.argument(0).unwrap().into()],
                &[],
                f32_type.as_ref(),
                context.string_attribute("llvm.sqrt").as_ref(),
                None,
                None,
                None,
                None,
                None,
                location,
            );
            assert_eq!(op.operation_name(), "llvm.call_intrinsic");
            assert_eq!(op.arguments(), vec![block.argument(0).unwrap()]);
            assert!(op.op_bundle_operands().is_empty());
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_call_intrinsic_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into()],
                    results: vec![f32_type.into()],
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
                  func.func @llvm_call_intrinsic_test(%arg0: f32) -> f32 {
                    %0 = llvm.call_intrinsic \"llvm.sqrt\"(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_call() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
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
            let mut block = context.block(&[(i32_type.as_ref(), location)]);
            let op = call(
                &[block.argument(0).unwrap().into()],
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
            assert_eq!(op.operation_name(), "llvm.call");
            assert_eq!(op.callee_operands(), vec![block.argument(0).unwrap()]);
            assert!(op.op_bundle_operands().is_empty());
            assert_eq!(op.output_type(), i32_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_call_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into()],
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
                  llvm.func @callee(i32) -> i32
                  func.func @llvm_call_test(%arg0: i32) -> i32 {
                    %0 = llvm.call @callee(%arg0) : (i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_inline_asm() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location)]);
            let op = inline_asm(
                &[block.argument(0).unwrap().into()],
                i32_type.as_ref(),
                context.string_attribute("mov $0, $1").as_ref(),
                context.string_attribute("=r,r").as_ref(),
                true,
                false,
                None,
                None,
                None,
                location,
            );
            assert_eq!(op.operation_name(), "llvm.inline_asm");
            assert_eq!(InlineAsmOperation::operands(&op), vec![block.argument(0).unwrap()]);
            assert!(op.has_side_effects());
            assert_eq!(op.output_type(), i32_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_inline_asm_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into()],
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
                  func.func @llvm_inline_asm_test(%arg0: i32) -> i32 {
                    %0 = llvm.inline_asm has_side_effects \"mov $0, $1\", \"=r,r\" %arg0 : (i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }
}
