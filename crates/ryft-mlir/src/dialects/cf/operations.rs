use crate::{
    Attribute, Block, BlockRef, DenseElementsAttributeRef, DenseInteger32ArrayAttributeRef, DetachedOp, DialectHandle,
    ElementsAttribute, FromWithContext, IntegerAttributeRef, IntegerTypeRef, IntoWithContext, Location, Operation,
    OperationBuilder, Size, StringAttributeRef, StringRef, Value, ValueRef, mlir_op, mlir_op_trait,
};

pub const ASSERT_MESSAGE_ATTRIBUTE: &str = "msg";

pub trait AssertOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    fn argument(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    fn message(&self) -> StringRef<'c> {
        self.attribute(ASSERT_MESSAGE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<StringAttributeRef>())
            .map(|attribute| attribute.string())
            .unwrap_or_else(|| panic!("invalid '{ASSERT_MESSAGE_ATTRIBUTE}' attribute in `cf::assert`"))
    }
}

mlir_op!(Assert);
mlir_op_trait!(Assert, ZeroOperands);
mlir_op_trait!(Assert, ZeroRegions);
mlir_op_trait!(Assert, ZeroSuccessors);

pub fn assert<
    'v,
    'c: 'v,
    't: 'c,
    V: Value<'v, 'c, 't>,
    M: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
    L: Location<'c, 't>,
>(
    argument: V,
    message: M,
    location: L,
) -> DetachedAssertOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::cf());
    OperationBuilder::new("cf.assert", location)
        .add_operand(argument)
        .add_attribute("msg", message.into_with_context(context))
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `cf::assert`")
}

pub trait BranchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Branch);
mlir_op_trait!(Branch, ZeroRegions);

pub fn br<'b, 'v, 'c: 'b + 'v, 't: 'c, B: Block<'b, 'c, 't>, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    successor: &B,
    operands: &[V],
    location: L,
) -> DetachedBranchOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::cf());
    OperationBuilder::new("cf.br", location)
        .add_operands(operands)
        .add_successor(successor)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `cf::br`")
}

pub const CONDITIONAL_OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operand_segment_sizes";

pub trait ConditionalBranchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    fn predicate(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    fn on_true_successor_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let true_successor_operand_count = self
            .attribute(CONDITIONAL_OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| Vec::<i32>::from(attribute)[1])
            .unwrap_or_else(|| {
                panic!("invalid '{CONDITIONAL_OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `cf::cond_br`")
            });
        self.operands().skip(1).take(true_successor_operand_count as usize).collect::<Vec<_>>()
    }

    fn on_false_successor_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let true_successor_operand_count = self
            .attribute(CONDITIONAL_OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| Vec::<i32>::from(attribute)[1])
            .unwrap_or_else(|| {
                panic!("invalid '{CONDITIONAL_OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `cf::cond_br`")
            });
        self.operands().skip(1 + true_successor_operand_count as usize).collect::<Vec<_>>()
    }

    fn on_true_successor(&self) -> BlockRef<'o, 'c, 't> {
        self.successor(0).unwrap()
    }

    fn on_false_successor(&self) -> BlockRef<'o, 'c, 't> {
        self.successor(1).unwrap()
    }
}

mlir_op!(ConditionalBranch);
mlir_op_trait!(ConditionalBranch, ZeroRegions);

pub fn cond_br<
    'predicate,
    'on_true_successor,
    'on_false_successor,
    'on_true_operand,
    'on_false_operand,
    'c: 'predicate + 'on_true_successor + 'on_false_successor + 'on_true_operand + 'on_false_operand,
    't: 'c,
    Predicate: Value<'predicate, 'c, 't>,
    OnTrueSuccessor: Block<'on_true_successor, 'c, 't>,
    OnFalseSuccessor: Block<'on_false_successor, 'c, 't>,
    OnTrueOperand: Value<'on_true_operand, 'c, 't>,
    OnFalseOperand: Value<'on_false_operand, 'c, 't>,
    L: Location<'c, 't>,
>(
    predicate: Predicate,
    on_true_successor: &OnTrueSuccessor,
    on_false_successor: &OnFalseSuccessor,
    on_true_successor_operands: &[OnTrueOperand],
    on_false_successor_operands: &[OnFalseOperand],
    location: L,
) -> DetachedConditionalBranchOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::cf());
    OperationBuilder::new("cf.cond_br", location)
        .add_operand(predicate)
        .add_operands(on_true_successor_operands)
        .add_operands(on_false_successor_operands)
        .add_successor(on_true_successor)
        .add_successor(on_false_successor)
        .add_attribute(
            CONDITIONAL_OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            DenseInteger32ArrayAttributeRef::from_with_context(
                &[1, on_true_successor_operands.len() as i32, on_false_successor_operands.len() as i32],
                context,
            ),
        )
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `cf::cond_br`")
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DefaultSwitchBranch<'o, 'c, 't> {
    pub successor: BlockRef<'o, 'c, 't>,
    pub successor_operands: Vec<ValueRef<'o, 'c, 't>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwitchBranch<'o, 'c, 't> {
    pub value: IntegerAttributeRef<'c, 't>,
    pub successor: BlockRef<'o, 'c, 't>,
    pub successor_operands: Vec<ValueRef<'o, 'c, 't>>,
}

pub const SWITCH_CASE_VALUES_ATTRIBUTE: &str = "case_values";
pub const SWITCH_CASE_OPERAND_COUNTS_ATTRIBUTE: &str = "case_operand_segments";
pub const SWITCH_OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operand_segment_sizes";

pub trait SwitchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    fn flag(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    fn default(&self) -> DefaultSwitchBranch<'o, 'c, 't> {
        let default_successor_operand_count = self
            .attribute(SWITCH_OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| Vec::<i32>::from(attribute)[1])
            .unwrap_or_else(|| panic!("invalid '{SWITCH_OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `cf::switch`"));
        DefaultSwitchBranch {
            successor: self.successor(0).unwrap(),
            successor_operands: self.operands().skip(1).take(default_successor_operand_count as usize).collect(),
        }
    }

    fn cases(&self) -> Vec<SwitchBranch<'o, 'c, 't>> {
        let mut case_values = self
            .attribute(SWITCH_CASE_VALUES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseElementsAttributeRef>())
            .map(|attribute| (0..attribute.elements_count()).map(move |i| attribute.element(&[i]).unwrap()))
            .unwrap_or_else(|| panic!("invalid '{SWITCH_CASE_VALUES_ATTRIBUTE}' attribute in `cf::switch`"))
            .flat_map(|element| element.cast::<IntegerAttributeRef>());
        let mut case_successors = self.successors().skip(1);
        let default_successor_operand_count = self
            .attribute(SWITCH_OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| Vec::<i32>::from(attribute)[1])
            .unwrap_or_else(|| panic!("invalid '{SWITCH_OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `cf::switch`"));
        let mut flattened_case_operands = self.operands().skip(1 + default_successor_operand_count as usize);
        let case_operand_counts = self
            .attribute(SWITCH_CASE_OPERAND_COUNTS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(Vec::<i32>::from)
            .unwrap_or_else(|| panic!("invalid '{SWITCH_CASE_OPERAND_COUNTS_ATTRIBUTE}' attribute in `cf::switch`"));
        let mut branches = Vec::new();
        for count in case_operand_counts {
            branches.push(SwitchBranch {
                value: case_values.by_ref().next().unwrap(),
                successor: case_successors.by_ref().next().unwrap(),
                successor_operands: flattened_case_operands.by_ref().take(count as usize).collect(),
            });
        }
        branches
    }
}

mlir_op!(Switch);
mlir_op_trait!(Switch, ZeroRegions);

pub fn switch<
    'flag,
    'default,
    'case,
    'c: 'flag + 'default + 'case,
    't: 'c,
    F: Value<'flag, 'c, 't>,
    L: Location<'c, 't>,
>(
    flag: F,
    flag_type: IntegerTypeRef<'c, 't>,
    default: DefaultSwitchBranch<'default, 'c, 't>,
    cases: &[SwitchBranch<'case, 'c, 't>],
    location: L,
) -> DetachedSwitchOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::cf());
    OperationBuilder::new("cf.switch", location)
        .add_operand(flag)
        .add_operands(default.successor_operands.as_slice())
        .add_operands(
            cases
                .iter()
                .flat_map(|branch| branch.successor_operands.iter().copied())
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .add_successor(&default.successor)
        .add_successors(cases.iter().map(|branch| &branch.successor).collect::<Vec<_>>().as_slice())
        .add_attribute(
            SWITCH_CASE_VALUES_ATTRIBUTE,
            context
                .dense_elements_attribute(
                    context.tensor_type(flag_type, &[Size::Static(cases.len())], None, location).unwrap(),
                    &cases.iter().map(|branch| branch.value).collect::<Vec<_>>(),
                )
                .unwrap(),
        )
        .add_attribute(
            SWITCH_CASE_OPERAND_COUNTS_ATTRIBUTE,
            context
                .dense_i32_array_attribute(
                    &cases.iter().map(|branch| branch.successor_operands.len() as i32).collect::<Vec<_>>(),
                )
                .unwrap(),
        )
        .add_attribute(
            SWITCH_OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            DenseInteger32ArrayAttributeRef::from_with_context(
                &[
                    1,
                    default.successor_operands.len() as i32,
                    cases.iter().map(|branch| branch.successor_operands.len() as i32).sum(),
                ],
                context,
            ),
        )
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `cf::switch`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::{arith, func};
    use crate::{Block, Context, Operation, Region};

    use super::*;

    #[test]
    fn test_assert() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i1_type = context.signless_integer_type(1);
        module.body().append_operation({
            let mut block = context.block(&[(i1_type, location)]);
            let argument = block.argument(0).unwrap();
            let op = assert(argument, "bad stuff", location);
            assert_eq!(op.argument(), argument);
            assert_eq!(op.message().as_str(), Ok("bad stuff"));
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<ValueRef, _>(&[], location));
            func::func(
                "assert_test",
                func::FuncAttributes { arguments: vec![i1_type.into()], results: vec![], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @assert_test(%arg0: i1) {
                    cf.assert %arg0, \"bad stuff\"
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_br() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut entry_block = context.block(&[(i32_type, location)]);
            let mut target_block = context.block(&[(i32_type, location)]);
            let argument = entry_block.argument(0).unwrap();
            let op = br(&target_block, &[argument], location);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 0);
            assert_eq!(op.successors().count(), 1);
            entry_block.append_operation(op);
            target_block.append_operation(func::r#return(&[target_block.argument(0).unwrap()], location));
            let mut region = context.region();
            region.append_block(entry_block);
            region.append_block(target_block);
            func::func(
                "br_test",
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
                  func.func @br_test(%arg0: i32) -> i32 {
                    cf.br ^bb1(%arg0 : i32)
                  ^bb1(%0: i32):  // pred: ^bb0
                    return %0 : i32
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
            let mut entry_block = context.block(&[(i1_type, location), (i32_type, location), (i32_type, location)]);
            let mut true_block = context.block(&[(i32_type, location)]);
            let mut false_block = context.block(&[(i32_type, location)]);
            let predicate = entry_block.argument(0).unwrap();
            let true_value = entry_block.argument(1).unwrap();
            let false_value = entry_block.argument(2).unwrap();
            let op = cond_br(predicate, &true_block, &false_block, &[true_value], &[false_value], location);
            assert_eq!(op.predicate(), predicate);
            assert_eq!(op.on_true_successor_operands(), vec![true_value]);
            assert_eq!(op.on_false_successor_operands(), vec![false_value]);
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
                "cond_br_test",
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
                  func.func @cond_br_test(%arg0: i1, %arg1: i32, %arg2: i32) -> i32 {
                    cf.cond_br %arg0, ^bb1(%arg1 : i32), ^bb2(%arg2 : i32)
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
    fn test_switch() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut entry_block = context.block_with_no_arguments();
            let mut default_block = context.block(&[(i32_type, location)]);
            let mut case_0_block = context.block(&[(i32_type, location)]);
            let mut case_1_block = context.block(&[(i32_type, location)]);
            let flag = entry_block
                .append_operation(arith::constant(context.integer_attribute(i32_type, 1), location))
                .result(0)
                .unwrap();
            let default_branch =
                DefaultSwitchBranch { successor: (&default_block).into(), successor_operands: vec![flag.into()] };
            let case_0_branch = SwitchBranch {
                value: context.integer_attribute(i32_type, 0),
                successor: (&case_0_block).into(),
                successor_operands: vec![flag.into()],
            };
            let case_1_branch = SwitchBranch {
                value: context.integer_attribute(i32_type, 1),
                successor: (&case_1_block).into(),
                successor_operands: vec![flag.into()],
            };
            let op = switch(
                flag,
                i32_type,
                default_branch.clone(),
                &[case_0_branch.clone(), case_1_branch.clone()],
                location,
            );
            assert_eq!(op.flag(), flag);
            assert_eq!(op.default(), default_branch);
            assert_eq!(op.cases(), vec![case_0_branch, case_1_branch]);
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 0);
            assert_eq!(op.successors().count(), 3);
            entry_block.append_operation(op);
            default_block.append_operation(func::r#return(&[default_block.argument(0).unwrap()], location));
            case_0_block.append_operation(func::r#return(&[case_0_block.argument(0).unwrap()], location));
            case_1_block.append_operation(func::r#return(&[case_1_block.argument(0).unwrap()], location));
            let mut region = context.region();
            region.append_block(entry_block);
            region.append_block(default_block);
            region.append_block(case_0_block);
            region.append_block(case_1_block);
            func::func(
                "switch_test",
                func::FuncAttributes { arguments: vec![], results: vec![i32_type.into()], ..Default::default() },
                region,
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @switch_test() -> i32 {
                    %c1_i32 = arith.constant 1 : i32
                    cf.switch %c1_i32 : i32, [
                      default: ^bb1(%c1_i32 : i32),
                      0: ^bb2(%c1_i32 : i32),
                      1: ^bb3(%c1_i32 : i32)
                    ]
                  ^bb1(%0: i32):  // pred: ^bb0
                    return %0 : i32
                  ^bb2(%1: i32):  // pred: ^bb0
                    return %1 : i32
                  ^bb3(%2: i32):  // pred: ^bb0
                    return %2 : i32
                  }
                }
            "},
        );
    }
}
