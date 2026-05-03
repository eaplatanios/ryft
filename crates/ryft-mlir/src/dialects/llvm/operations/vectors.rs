use crate::{
    AttributeRef, DetachedOp, DialectHandle, Location, Operation, OperationBuilder, TypeRef, Value, ValueRef, mlir_op,
};

/// Canonical MLIR operation name for [`ExtractElementOperation`].
pub const EXTRACT_ELEMENT_OPERATION_NAME: &str = "llvm.extractelement";

/// Operation trait for `llvm.extractelement`.
pub trait ExtractElementOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        EXTRACT_ELEMENT_OPERATION_NAME
    }

    /// Returns the `vector` operand.
    fn vector(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `position` operand.
    fn position(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(ExtractElement);

/// Constructs a new detached `llvm.extractelement` operation.
pub fn extract_element<'c, 't: 'c, V1: Value<'c, 'c, 't>, V2: Value<'c, 'c, 't>, L: Location<'c, 't>>(
    vector: V1,
    position: V2,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedExtractElementOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(EXTRACT_ELEMENT_OPERATION_NAME, location);
    builder = builder.add_operand(vector);
    builder = builder.add_operand(position);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::extractelement`")
}

/// Canonical MLIR operation name for [`InsertElementOperation`].
pub const INSERT_ELEMENT_OPERATION_NAME: &str = "llvm.insertelement";

/// Operation trait for `llvm.insertelement`.
pub trait InsertElementOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        INSERT_ELEMENT_OPERATION_NAME
    }

    /// Returns the `vector` operand.
    fn vector(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `position` operand.
    fn position(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(InsertElement);

/// Constructs a new detached `llvm.insertelement` operation.
pub fn insert_element<
    'c,
    't: 'c,
    V1: Value<'c, 'c, 't>,
    V2: Value<'c, 'c, 't>,
    V3: Value<'c, 'c, 't>,
    L: Location<'c, 't>,
>(
    vector: V1,
    value: V2,
    position: V3,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedInsertElementOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(INSERT_ELEMENT_OPERATION_NAME, location);
    builder = builder.add_operand(vector);
    builder = builder.add_operand(value);
    builder = builder.add_operand(position);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::insert_element`")
}

/// Canonical MLIR operation name for [`ShuffleVectorOperation`].
pub const SHUFFLE_VECTOR_OPERATION_NAME: &str = "llvm.shufflevector";

/// Operation trait for `llvm.shufflevector`.
pub trait ShuffleVectorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        SHUFFLE_VECTOR_OPERATION_NAME
    }

    /// Returns the `first_vector` operand.
    fn first_vector(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `second_vector` operand.
    fn second_vector(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` attribute.
    fn mask(&self) -> AttributeRef<'c, 't> {
        self.attribute("mask").unwrap()
    }

    /// Returns this operation's first result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(ShuffleVector);

/// Constructs a new detached `llvm.shufflevector` operation.
pub fn shuffle_vector<'c, 't: 'c, V1: Value<'c, 'c, 't>, V2: Value<'c, 'c, 't>, L: Location<'c, 't>>(
    first_vector: V1,
    second_vector: V2,
    result_type: TypeRef<'c, 't>,
    mask: AttributeRef<'c, 't>,
    location: L,
) -> DetachedShuffleVectorOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(SHUFFLE_VECTOR_OPERATION_NAME, location);
    builder = builder.add_operand(first_vector);
    builder = builder.add_operand(second_vector);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("mask", mask);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::shufflevector`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Attribute, Block, Context, Operation, Type};

    use super::*;

    #[test]
    fn test_extract_element() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let vector_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(vector_type, location), (i32_type.as_ref(), location)]);
            let op =
                extract_element(block.argument(0).unwrap(), block.argument(1).unwrap(), i32_type.as_ref(), location);
            assert_eq!(op.operation_name(), "llvm.extractelement");
            assert_eq!(op.vector(), block.argument(0).unwrap());
            assert_eq!(op.position(), block.argument(1).unwrap());
            assert_eq!(op.output_type(), i32_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_extractelement_test",
                func::FuncAttributes {
                    arguments: vec![vector_type.into(), i32_type.into()],
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
                  func.func @llvm_extractelement_test(%arg0: vector<4xi32>, %arg1: i32) -> i32 {
                    %0 = llvm.extractelement %arg0[%arg1 : i32] : vector<4xi32>
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_insert_element() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let vector_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block =
                context.block(&[(vector_type, location), (i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let op = insert_element(
                block.argument(0).unwrap(),
                block.argument(1).unwrap(),
                block.argument(2).unwrap(),
                vector_type,
                location,
            );
            assert_eq!(op.operation_name(), "llvm.insertelement");
            assert_eq!(op.vector(), block.argument(0).unwrap());
            assert_eq!(op.value(), block.argument(1).unwrap());
            assert_eq!(op.position(), block.argument(2).unwrap());
            assert_eq!(op.output_type(), vector_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_insert_element_test",
                func::FuncAttributes {
                    arguments: vec![vector_type.into(), i32_type.into(), i32_type.into()],
                    results: vec![vector_type.into()],
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
                  func.func @llvm_insert_element_test(%arg0: vector<4xi32>, %arg1: i32, %arg2: i32) -> vector<4xi32> {
                    %0 = llvm.insertelement %arg1, %arg0[%arg2 : i32] : vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_shuffle_vector() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let vector_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(vector_type, location), (vector_type, location)]);
            let mask = context.dense_i32_array_attribute(&[0, 1, 4, 5]).unwrap();
            let op = shuffle_vector(
                block.argument(0).unwrap(),
                block.argument(1).unwrap(),
                vector_type,
                mask.as_ref(),
                location,
            );
            assert_eq!(op.operation_name(), "llvm.shufflevector");
            assert_eq!(op.first_vector(), block.argument(0).unwrap());
            assert_eq!(op.second_vector(), block.argument(1).unwrap());
            assert_eq!(op.mask(), mask);
            assert_eq!(op.output_type(), vector_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_shufflevector_test",
                func::FuncAttributes {
                    arguments: vec![vector_type.into(), vector_type.into()],
                    results: vec![vector_type.into()],
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
                "  func.func @llvm_shufflevector_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi32> {\n",
                "    %0 = llvm.shufflevector %arg0, %arg1 [0, 1, 4, 5] : vector<4xi32> \n",
                "    return %0 : vector<4xi32>\n",
                "  }\n",
                "}\n",
            ),
        );
    }
}
