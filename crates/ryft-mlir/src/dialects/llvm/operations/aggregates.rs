use crate::{
    AttributeRef, DetachedOp, DialectHandle, Location, Operation, OperationBuilder, TypeRef, Value, ValueRef, mlir_op,
};

/// Canonical MLIR operation name for [`ExtractValueOperation`].
pub const EXTRACT_VALUE_OPERATION_NAME: &str = "llvm.extractvalue";

/// Operation trait for `llvm.extractvalue`.
pub trait ExtractValueOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        EXTRACT_VALUE_OPERATION_NAME
    }

    /// Returns the `container` operand.
    fn container(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `position` attribute.
    fn position(&self) -> AttributeRef<'c, 't> {
        self.attribute("position").unwrap()
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(ExtractValue);

/// Constructs a new detached `llvm.extractvalue` operation.
pub fn extract_value<'c, 't: 'c, V1: Value<'c, 'c, 't>, L: Location<'c, 't>>(
    container: V1,
    result_type: TypeRef<'c, 't>,
    position: AttributeRef<'c, 't>,
    location: L,
) -> DetachedExtractValueOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(EXTRACT_VALUE_OPERATION_NAME, location);
    builder = builder.add_operand(container);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("position", position);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::extract_value`")
}

/// Canonical MLIR operation name for [`InsertValueOperation`].
pub const INSERT_VALUE_OPERATION_NAME: &str = "llvm.insertvalue";

/// Operation trait for `llvm.insertvalue`.
pub trait InsertValueOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        INSERT_VALUE_OPERATION_NAME
    }

    /// Returns the `container` operand.
    fn container(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `position` attribute.
    fn position(&self) -> AttributeRef<'c, 't> {
        self.attribute("position").unwrap()
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(InsertValue);

/// Constructs a new detached `llvm.insertvalue` operation.
pub fn insert_value<'c, 't: 'c, V1: Value<'c, 'c, 't>, V2: Value<'c, 'c, 't>, L: Location<'c, 't>>(
    container: V1,
    value: V2,
    result_type: TypeRef<'c, 't>,
    position: AttributeRef<'c, 't>,
    location: L,
) -> DetachedInsertValueOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(INSERT_VALUE_OPERATION_NAME, location);
    builder = builder.add_operand(container);
    builder = builder.add_operand(value);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("position", position);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::insert_value`")
}

/// Canonical MLIR operation name for [`LandingpadOperation`].
pub const LANDING_PAD_OPERATION_NAME: &str = "llvm.landingpad";

/// Operation trait for `llvm.landingpad`.
pub trait LandingPadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        LANDING_PAD_OPERATION_NAME
    }

    /// Returns the `clauses` operands.
    fn clauses(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(0).collect()
    }

    /// Returns whether the `cleanup` unit attribute is present.
    fn cleanup(&self) -> bool {
        self.has_attribute("cleanup")
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(LandingPad);

/// Constructs a new detached `llvm.landingpad` operation.
pub fn landing_pad<'c, 't: 'c, L: Location<'c, 't>>(
    clauses: &[ValueRef<'c, 'c, 't>],
    result_type: TypeRef<'c, 't>,
    cleanup: bool,
    location: L,
) -> DetachedLandingPadOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(LANDING_PAD_OPERATION_NAME, location);
    builder = builder.add_operands(clauses);
    builder = builder.add_result(result_type);
    if cleanup {
        builder = builder.add_attribute("cleanup", context.unit_attribute());
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::landing_pad`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Attribute, Block, Context, Operation, Type};

    use super::*;

    #[test]
    fn test_extract_value() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let struct_type = context.llvm_literal_struct_type(&[i32_type.as_ref(), i32_type.as_ref()], false);
        module.body().append_operation({
            let mut block = context.block(&[(struct_type.as_ref(), location)]);
            let position = context.dense_i64_array_attribute(&[1]).unwrap();
            let op = extract_value(block.argument(0).unwrap(), i32_type.as_ref(), position.as_ref(), location);
            assert_eq!(op.operation_name(), "llvm.extractvalue");
            assert_eq!(op.container(), block.argument(0).unwrap());
            assert_eq!(op.position(), position);
            assert_eq!(op.output_type(), i32_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_extractvalue_test",
                func::FuncAttributes {
                    arguments: vec![struct_type.into()],
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
            concat!(
                "module {\n",
                "  func.func @llvm_extractvalue_test(%arg0: !llvm.struct<(i32, i32)>) -> i32 {\n",
                "    %0 = llvm.extractvalue %arg0[1] : !llvm.struct<(i32, i32)> \n",
                "    return %0 : i32\n",
                "  }\n",
                "}\n",
            ),
        );
    }

    #[test]
    fn test_insert_value() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let struct_type = context.llvm_literal_struct_type(&[i32_type.as_ref(), i32_type.as_ref()], false);
        module.body().append_operation({
            let mut block = context.block(&[(struct_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let position = context.dense_i64_array_attribute(&[1]).unwrap();
            let op = insert_value(
                block.argument(0).unwrap(),
                block.argument(1).unwrap(),
                struct_type.as_ref(),
                position.as_ref(),
                location,
            );
            assert_eq!(op.operation_name(), "llvm.insertvalue");
            assert_eq!(op.container(), block.argument(0).unwrap());
            assert_eq!(op.value(), block.argument(1).unwrap());
            assert_eq!(op.position(), position);
            assert_eq!(op.output_type(), struct_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_insert_value_test",
                func::FuncAttributes {
                    arguments: vec![struct_type.into(), i32_type.into()],
                    results: vec![struct_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            concat!(
                "module {\n",
                "  func.func @llvm_insert_value_test(%arg0: !llvm.struct<(i32, i32)>, %arg1: i32) -> !llvm.struct<(i32, i32)> {\n",
                "    %0 = llvm.insertvalue %arg1, %arg0[1] : !llvm.struct<(i32, i32)> \n",
                "    return %0 : !llvm.struct<(i32, i32)>\n",
                "  }\n",
                "}\n",
            ),
        );
    }

    #[test]
    fn test_landing_pad() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i8_type = context.signless_integer_type(8);
        let pointer_type = context.llvm_pointer_type(0);
        let result_type = context.llvm_literal_struct_type(&[pointer_type.as_ref(), i8_type.as_ref()], false);
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = landing_pad(&[], result_type.as_ref(), true, location);
            assert_eq!(op.operation_name(), "llvm.landingpad");
            assert!(op.cleanup());
            assert_eq!(op.output_type(), result_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_landingpad_test",
                func::FuncAttributes { arguments: vec![], results: vec![result_type.into()], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_landingpad_test() -> !llvm.struct<(ptr, i8)> {
                    %0 = llvm.landingpad cleanup : !llvm.struct<(ptr, i8)>
                    return %0 : !llvm.struct<(ptr, i8)>
                  }
                }
            "},
        );
    }
}
