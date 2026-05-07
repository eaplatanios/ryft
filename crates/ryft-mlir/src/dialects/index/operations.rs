use crate::{
    DetachedOp, DialectHandle, Error, Location, Operation, OperationBuilder, Value, ValueRef, mlir_binary_op,
    mlir_generic_unary_op, mlir_op, mlir_op_trait,
};

pub const CONSTANT_VALUE_ATTRIBUTE: &str = "value";

pub trait ConstantOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    fn value(&self) -> Result<usize, Error> {
        Ok(self.integer_attribute(CONSTANT_VALUE_ATTRIBUTE)?.signless_value() as usize)
    }
}

mlir_op!(Constant);
mlir_op_trait!(Constant, ConstantLike);
mlir_op_trait!(Constant, OneResult);
mlir_op_trait!(Constant, ZeroOperands);
mlir_op_trait!(Constant, ZeroRegions);
mlir_op_trait!(Constant, ZeroSuccessors);

pub fn constant<'c, 't: 'c, L: Location<'c, 't>>(
    value: usize,
    location: L,
) -> Result<DetachedConstantOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::index()?);
    OperationBuilder::new("index.constant", location)
        .add_attribute("value", context.integer_attribute(context.index_type(), value as i64))
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `index::constant`"))
        })
}

pub trait BoolConstantOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    fn value(&self) -> Result<bool, Error> {
        Ok(self.boolean_attribute(CONSTANT_VALUE_ATTRIBUTE)?.value())
    }
}

mlir_op!(BoolConstant);
mlir_op_trait!(BoolConstant, ConstantLike);
mlir_op_trait!(BoolConstant, OneResult);
mlir_op_trait!(BoolConstant, ZeroOperands);
mlir_op_trait!(BoolConstant, ZeroRegions);
mlir_op_trait!(BoolConstant, ZeroSuccessors);

pub fn bool_constant<'c, 't: 'c, L: Location<'c, 't>>(
    value: bool,
    location: L,
) -> Result<DetachedBoolConstantOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::index()?);
    OperationBuilder::new("index.bool.constant", location)
        .add_attribute("value", context.boolean_attribute(value))
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `index::bool_constant`"))
        })
}

mlir_generic_unary_op!(index, casts);
mlir_generic_unary_op!(index, castu);

mlir_binary_op!(index, add);
mlir_binary_op!(index, and);
mlir_binary_op!(index, ceildivs);
mlir_binary_op!(index, ceildivu);
mlir_binary_op!(index, divs);
mlir_binary_op!(index, divu);
mlir_binary_op!(index, floordivs);
mlir_binary_op!(index, maxs);
mlir_binary_op!(index, maxu);
mlir_binary_op!(index, mins);
mlir_binary_op!(index, minu);
mlir_binary_op!(index, mul);
mlir_binary_op!(index, or);
mlir_binary_op!(index, rems);
mlir_binary_op!(index, remu);
mlir_binary_op!(index, shl);
mlir_binary_op!(index, shrs);
mlir_binary_op!(index, shru);
mlir_binary_op!(index, sub);
mlir_binary_op!(index, xor);

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComparisonPredicate {
    Equal,
    NotEqual,
    SignedLessThan,
    SignedLessThanOrEqual,
    SignedGreaterThan,
    SignedGreaterThanOrEqual,
    UnsignedLessThan,
    UnsignedLessThanOrEqual,
    UnsignedGreaterThan,
    UnsignedGreaterThanOrEqual,
}

pub const CMP_PREDICATE_ATTRIBUTE: &str = "pred";

pub trait CmpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    fn predicate(&self) -> Result<ComparisonPredicate, Error> {
        self.attribute(CMP_PREDICATE_ATTRIBUTE)
            .and_then(|attribute| match attribute.to_string().as_str() {
                "#index<cmp_predicate eq>" => Some(ComparisonPredicate::Equal),
                "#index<cmp_predicate ne>" => Some(ComparisonPredicate::NotEqual),
                "#index<cmp_predicate slt>" => Some(ComparisonPredicate::SignedLessThan),
                "#index<cmp_predicate sle>" => Some(ComparisonPredicate::SignedLessThanOrEqual),
                "#index<cmp_predicate sgt>" => Some(ComparisonPredicate::SignedGreaterThan),
                "#index<cmp_predicate sge>" => Some(ComparisonPredicate::SignedGreaterThanOrEqual),
                "#index<cmp_predicate ult>" => Some(ComparisonPredicate::UnsignedLessThan),
                "#index<cmp_predicate ule>" => Some(ComparisonPredicate::UnsignedLessThanOrEqual),
                "#index<cmp_predicate ugt>" => Some(ComparisonPredicate::UnsignedGreaterThan),
                "#index<cmp_predicate uge>" => Some(ComparisonPredicate::UnsignedGreaterThanOrEqual),
                _ => None,
            })
            .ok_or_else(|| Error::invalid_argument("invalid `pred` attribute in `index::cmp`"))
    }
}

mlir_op!(Cmp);
mlir_op_trait!(Cmp, OneResult);
mlir_op_trait!(Cmp, ZeroRegions);

pub fn cmp<
    'lhs,
    'rhs,
    'c: 'lhs + 'rhs,
    't: 'c,
    LHS: Value<'lhs, 'c, 't>,
    RHS: Value<'lhs, 'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: LHS,
    rhs: RHS,
    predicate: ComparisonPredicate,
    location: L,
) -> Result<DetachedCmpOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::index()?);
    OperationBuilder::new("index.cmp", location)
        .add_attribute(
            CMP_PREDICATE_ATTRIBUTE,
            context.parse_attribute(match predicate {
                ComparisonPredicate::Equal => "#index<cmp_predicate eq>",
                ComparisonPredicate::NotEqual => "#index<cmp_predicate ne>",
                ComparisonPredicate::SignedLessThan => "#index<cmp_predicate slt>",
                ComparisonPredicate::SignedLessThanOrEqual => "#index<cmp_predicate sle>",
                ComparisonPredicate::SignedGreaterThan => "#index<cmp_predicate sgt>",
                ComparisonPredicate::SignedGreaterThanOrEqual => "#index<cmp_predicate sge>",
                ComparisonPredicate::UnsignedLessThan => "#index<cmp_predicate ult>",
                ComparisonPredicate::UnsignedLessThanOrEqual => "#index<cmp_predicate ule>",
                ComparisonPredicate::UnsignedGreaterThan => "#index<cmp_predicate ugt>",
                ComparisonPredicate::UnsignedGreaterThanOrEqual => "#index<cmp_predicate uge>",
            })?,
        )
        .add_operand(lhs)
        .add_operand(rhs)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `index::cmp`"))
        })
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Block, Context, OneOperand, Operation};

    use super::*;

    #[test]
    fn test_constant() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let op = constant(42, location).unwrap();
                assert_eq!(op.value().unwrap(), 42);
                assert_eq!(op.value().unwrap(), 42);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "constant_test",
                    func::FuncAttributes { arguments: vec![], results: vec![index_type.into()], ..Default::default() },
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
                  func.func @constant_test() -> index {
                    %idx42 = index.constant 42
                    return %idx42 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_bool_constant() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i1_type = context.signless_integer_type(1);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let op = bool_constant(true, location).unwrap();
                assert_eq!(op.value().unwrap(), true);
                assert_eq!(op.value().unwrap(), true);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "bool_constant_test",
                    func::FuncAttributes { arguments: vec![], results: vec![i1_type.into()], ..Default::default() },
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
                  func.func @bool_constant_test() -> i1 {
                    %true = index.bool.constant true
                    return %true : i1
                  }
                }
            "},
        );
    }

    #[test]
    fn test_casts() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type, location)]);
                let input = block.argument(0).unwrap();
                let op = casts(input, index_type, location).unwrap();
                assert_eq!(op.input().unwrap(), input);
                assert_eq!(op.input().unwrap(), input);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "casts_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into()],
                        results: vec![index_type.into()],
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
                  func.func @casts_test(%arg0: i32) -> index {
                    %0 = index.casts %arg0 : i32 to index
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_castu() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type, location)]);
                let input = block.argument(0).unwrap();
                let op = castu(input, index_type, location).unwrap();
                assert_eq!(op.input().unwrap(), input);
                assert_eq!(op.input().unwrap(), input);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "castu_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into()],
                        results: vec![index_type.into()],
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
                  func.func @castu_test(%arg0: i32) -> index {
                    %0 = index.castu %arg0 : i32 to index
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_add() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = add(lhs, rhs, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "add_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
                        results: vec![index_type.into()],
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
                  func.func @add_test(%arg0: index, %arg1: index) -> index {
                    %0 = index.add %arg0, %arg1
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_and() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = and(lhs, rhs, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "and_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
                        results: vec![index_type.into()],
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
                  func.func @and_test(%arg0: index, %arg1: index) -> index {
                    %0 = index.and %arg0, %arg1
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_ceildivs() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = ceildivs(lhs, rhs, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "ceildivs_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
                        results: vec![index_type.into()],
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
                  func.func @ceildivs_test(%arg0: index, %arg1: index) -> index {
                    %0 = index.ceildivs %arg0, %arg1
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_ceildivu() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = ceildivu(lhs, rhs, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "ceildivu_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
                        results: vec![index_type.into()],
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
                  func.func @ceildivu_test(%arg0: index, %arg1: index) -> index {
                    %0 = index.ceildivu %arg0, %arg1
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_divs() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = divs(lhs, rhs, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "divs_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
                        results: vec![index_type.into()],
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
                  func.func @divs_test(%arg0: index, %arg1: index) -> index {
                    %0 = index.divs %arg0, %arg1
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_divu() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = divu(lhs, rhs, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "divu_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
                        results: vec![index_type.into()],
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
                  func.func @divu_test(%arg0: index, %arg1: index) -> index {
                    %0 = index.divu %arg0, %arg1
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_floordivs() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = floordivs(lhs, rhs, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "floordivs_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
                        results: vec![index_type.into()],
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
                  func.func @floordivs_test(%arg0: index, %arg1: index) -> index {
                    %0 = index.floordivs %arg0, %arg1
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_maxs() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = maxs(lhs, rhs, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "maxs_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
                        results: vec![index_type.into()],
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
                  func.func @maxs_test(%arg0: index, %arg1: index) -> index {
                    %0 = index.maxs %arg0, %arg1
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_maxu() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = maxu(lhs, rhs, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "maxu_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
                        results: vec![index_type.into()],
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
                  func.func @maxu_test(%arg0: index, %arg1: index) -> index {
                    %0 = index.maxu %arg0, %arg1
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_mins() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = mins(lhs, rhs, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "mins_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
                        results: vec![index_type.into()],
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
                  func.func @mins_test(%arg0: index, %arg1: index) -> index {
                    %0 = index.mins %arg0, %arg1
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_minu() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = minu(lhs, rhs, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "minu_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
                        results: vec![index_type.into()],
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
                  func.func @minu_test(%arg0: index, %arg1: index) -> index {
                    %0 = index.minu %arg0, %arg1
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_mul() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = mul(lhs, rhs, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "mul_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
                        results: vec![index_type.into()],
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
                  func.func @mul_test(%arg0: index, %arg1: index) -> index {
                    %0 = index.mul %arg0, %arg1
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_or() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = or(lhs, rhs, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "or_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
                        results: vec![index_type.into()],
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
                  func.func @or_test(%arg0: index, %arg1: index) -> index {
                    %0 = index.or %arg0, %arg1
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_rems() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = rems(lhs, rhs, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "rems_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
                        results: vec![index_type.into()],
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
                  func.func @rems_test(%arg0: index, %arg1: index) -> index {
                    %0 = index.rems %arg0, %arg1
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_remu() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = remu(lhs, rhs, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "remu_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
                        results: vec![index_type.into()],
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
                  func.func @remu_test(%arg0: index, %arg1: index) -> index {
                    %0 = index.remu %arg0, %arg1
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_shl() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = shl(lhs, rhs, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "shl_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
                        results: vec![index_type.into()],
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
                  func.func @shl_test(%arg0: index, %arg1: index) -> index {
                    %0 = index.shl %arg0, %arg1
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_shrs() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = shrs(lhs, rhs, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "shrs_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
                        results: vec![index_type.into()],
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
                  func.func @shrs_test(%arg0: index, %arg1: index) -> index {
                    %0 = index.shrs %arg0, %arg1
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_shru() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = shru(lhs, rhs, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "shru_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
                        results: vec![index_type.into()],
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
                  func.func @shru_test(%arg0: index, %arg1: index) -> index {
                    %0 = index.shru %arg0, %arg1
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_sub() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = sub(lhs, rhs, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "sub_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
                        results: vec![index_type.into()],
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
                  func.func @sub_test(%arg0: index, %arg1: index) -> index {
                    %0 = index.sub %arg0, %arg1
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_xor() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = xor(lhs, rhs, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "xor_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
                        results: vec![index_type.into()],
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
                  func.func @xor_test(%arg0: index, %arg1: index) -> index {
                    %0 = index.xor %arg0, %arg1
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_cmp() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i1_type = context.signless_integer_type(1);
        let index_type = context.index_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(index_type, location), (index_type, location)]);
                let lhs = block.argument(0).unwrap();
                let rhs = block.argument(1).unwrap();
                let op = cmp(lhs, rhs, ComparisonPredicate::UnsignedGreaterThanOrEqual, location).unwrap();
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.rhs().unwrap(), rhs);
                assert_eq!(op.predicate().unwrap(), ComparisonPredicate::UnsignedGreaterThanOrEqual);
                assert_eq!(op.lhs().unwrap(), lhs);
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "cmp_test",
                    func::FuncAttributes {
                        arguments: vec![index_type.into(), index_type.into()],
                        results: vec![i1_type.into()],
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
                  func.func @cmp_test(%arg0: index, %arg1: index) -> i1 {
                    %0 = index.cmp uge(%arg0, %arg1)
                    return %0 : i1
                  }
                }
            "},
        );
    }
}
