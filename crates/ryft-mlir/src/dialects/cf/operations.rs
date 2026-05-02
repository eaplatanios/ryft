use crate::{
    Attribute, Block, BlockRef, DenseElementsAttributeRef, DenseInteger32ArrayAttributeRef, DetachedOp, DialectHandle,
    ElementsAttribute, FromWithContext, IntegerAttributeRef, IntegerTypeRef, IntoWithContext, Location, Operation,
    OperationBuilder, Size, StringAttributeRef, StringRef, Value, ValueRef, mlir_op, mlir_op_trait,
};

/// Name of the `cf.assert` message attribute.
pub const ASSERT_MESSAGE_ATTRIBUTE: &str = "msg";

/// Operation trait for `cf.assert`.
///
/// `cf.assert` checks a single boolean operand at runtime. If the operand is false, execution aborts and the message
/// attribute may be surfaced to the user by the runtime.
pub trait AssertOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the boolean value being asserted.
    fn argument(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the runtime error message attached to this assertion.
    fn message(&self) -> StringRef<'c> {
        self.attribute(ASSERT_MESSAGE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<StringAttributeRef>())
            .map(|attribute| attribute.string())
            .unwrap_or_else(|| panic!("invalid '{ASSERT_MESSAGE_ATTRIBUTE}' attribute in `cf::assert`"))
    }
}

mlir_op!(Assert);
mlir_op_trait!(Assert, ZeroRegions);
mlir_op_trait!(Assert, ZeroSuccessors);

/// Constructs a new detached/owned [`AssertOperation`] at the specified [`Location`].
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
        .add_attribute(ASSERT_MESSAGE_ATTRIBUTE, message.into_with_context(context))
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `cf::assert`")
}

/// Operation trait for `cf.br`.
///
/// `cf.br` is an unconditional terminator that transfers control to one destination block and forwards all operands to
/// that block.
pub trait BranchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the destination block.
    fn destination(&self) -> BlockRef<'o, 'c, 't> {
        self.successor(0).unwrap()
    }

    /// Returns the operands forwarded to the destination block.
    fn destination_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }
}

mlir_op!(Branch);
mlir_op_trait!(Branch, AlwaysSpeculatable);
mlir_op_trait!(Branch, IsTerminator);
mlir_op_trait!(Branch, NoMemoryEffect);
mlir_op_trait!(Branch, Pure);
mlir_op_trait!(Branch, SingleBlockRegions);
mlir_op_trait!(Branch, ZeroRegions);

/// Constructs a new detached/owned [`BranchOperation`] at the specified [`Location`].
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

/// Name of the attribute that stores operand segment sizes for `cf.cond_br`.
pub const CONDITIONAL_OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operand_segment_sizes";

/// Name of the optional branch weight attribute for `cf.cond_br`.
pub const CONDITIONAL_BRANCH_WEIGHTS_ATTRIBUTE: &str = "branch_weights";

/// Operation trait for `cf.cond_br`.
///
/// `cf.cond_br` is a conditional terminator that branches to the true destination when the predicate is set and to the
/// false destination otherwise. Each destination has its own forwarded operand segment.
pub trait ConditionalBranchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the boolean predicate controlling which successor is selected.
    fn predicate(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the operands forwarded to the true successor.
    fn on_true_successor_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let true_successor_operand_count = self
            .attribute(CONDITIONAL_OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| Vec::<i32>::from(attribute)[1])
            .unwrap_or_else(|| {
                panic!("invalid '{CONDITIONAL_OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `cf::cond_br`")
            });
        self.operand_values().skip(1).take(true_successor_operand_count as usize).collect::<Vec<_>>()
    }

    /// Returns the operands forwarded to the false successor.
    fn on_false_successor_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let true_successor_operand_count = self
            .attribute(CONDITIONAL_OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| Vec::<i32>::from(attribute)[1])
            .unwrap_or_else(|| {
                panic!("invalid '{CONDITIONAL_OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `cf::cond_br`")
            });
        self.operand_values().skip(1 + true_successor_operand_count as usize).collect::<Vec<_>>()
    }

    /// Returns the true successor block.
    fn on_true_successor(&self) -> BlockRef<'o, 'c, 't> {
        self.successor(0).unwrap()
    }

    /// Returns the false successor block.
    fn on_false_successor(&self) -> BlockRef<'o, 'c, 't> {
        self.successor(1).unwrap()
    }

    /// Returns the optional branch weight attribute.
    fn branch_weights(&self) -> Option<DenseInteger32ArrayAttributeRef<'c, 't>> {
        self.attribute(CONDITIONAL_BRANCH_WEIGHTS_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }
}

mlir_op!(ConditionalBranch);
mlir_op_trait!(ConditionalBranch, AlwaysSpeculatable);
mlir_op_trait!(ConditionalBranch, IsTerminator);
mlir_op_trait!(ConditionalBranch, NoMemoryEffect);
mlir_op_trait!(ConditionalBranch, Pure);
mlir_op_trait!(ConditionalBranch, SingleBlockRegions);
mlir_op_trait!(ConditionalBranch, ZeroRegions);

/// Constructs a new detached/owned [`ConditionalBranchOperation`] at the specified [`Location`].
///
/// Branch weights are omitted from the operation when `branch_weights` is empty. When present, MLIR expects the weights
/// to describe the true and false successor probabilities.
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
    branch_weights: &[i32],
    location: L,
) -> DetachedConditionalBranchOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::cf());
    let mut builder = OperationBuilder::new("cf.cond_br", location)
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
        );
    if !branch_weights.is_empty() {
        builder = builder.add_attribute(
            CONDITIONAL_BRANCH_WEIGHTS_ATTRIBUTE,
            DenseInteger32ArrayAttributeRef::from_with_context(branch_weights, context),
        );
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `cf::cond_br`")
}

/// Default destination and operands for a `cf.switch`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DefaultSwitchBranch<'o, 'c, 't> {
    /// Block reached when no case value matches the switch flag.
    pub successor: BlockRef<'o, 'c, 't>,

    /// Operands forwarded to the default destination block.
    pub successor_operands: Vec<ValueRef<'o, 'c, 't>>,
}

/// Case destination, value, and operands for a `cf.switch`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwitchBranch<'o, 'c, 't> {
    /// Integer value that selects this case.
    pub value: IntegerAttributeRef<'c, 't>,

    /// Block reached when the switch flag matches [`Self::value`].
    pub successor: BlockRef<'o, 'c, 't>,

    /// Operands forwarded to the case destination block.
    pub successor_operands: Vec<ValueRef<'o, 'c, 't>>,
}

/// Name of the optional `cf.switch` case values attribute.
pub const SWITCH_CASE_VALUES_ATTRIBUTE: &str = "case_values";

/// Name of the `cf.switch` case operand segment-size attribute.
pub const SWITCH_CASE_OPERAND_COUNTS_ATTRIBUTE: &str = "case_operand_segments";

/// Name of the attribute that stores operand segment sizes for `cf.switch`.
pub const SWITCH_OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operand_segment_sizes";

/// Operation trait for `cf.switch`.
///
/// `cf.switch` branches on a signless integer flag. It jumps to the first matching case destination, or to the default
/// destination when no case matches.
pub trait SwitchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the integer flag value used for case selection.
    fn flag(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the default destination and its forwarded operands.
    fn default(&self) -> DefaultSwitchBranch<'o, 'c, 't> {
        let default_successor_operand_count = self
            .attribute(SWITCH_OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| Vec::<i32>::from(attribute)[1])
            .unwrap_or_else(|| panic!("invalid '{SWITCH_OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `cf::switch`"));
        DefaultSwitchBranch {
            successor: self.successor(0).unwrap(),
            successor_operands: self.operand_values().skip(1).take(default_successor_operand_count as usize).collect(),
        }
    }

    /// Returns the case destinations, their case values, and their forwarded operands.
    fn cases(&self) -> Vec<SwitchBranch<'o, 'c, 't>> {
        let case_operand_counts = self
            .attribute(SWITCH_CASE_OPERAND_COUNTS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(Vec::<i32>::from)
            .unwrap_or_else(|| panic!("invalid '{SWITCH_CASE_OPERAND_COUNTS_ATTRIBUTE}' attribute in `cf::switch`"));
        let Some(case_values_attribute) = self
            .attribute(SWITCH_CASE_VALUES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseElementsAttributeRef>())
        else {
            if case_operand_counts.is_empty() {
                return Vec::new();
            }
            panic!("invalid '{SWITCH_CASE_VALUES_ATTRIBUTE}' attribute in `cf::switch`")
        };
        let mut case_values =
            (0..case_values_attribute.elements_count()).map(move |i| case_values_attribute.element(&[i]).unwrap());
        let mut case_successors = self.successors().skip(1);
        let default_successor_operand_count = self
            .attribute(SWITCH_OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| Vec::<i32>::from(attribute)[1])
            .unwrap_or_else(|| panic!("invalid '{SWITCH_OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `cf::switch`"));
        let mut flattened_case_operands = self.operand_values().skip(1 + default_successor_operand_count as usize);
        let mut branches = Vec::new();
        for count in case_operand_counts {
            branches.push(SwitchBranch {
                value: case_values.by_ref().next().unwrap().cast::<IntegerAttributeRef>().unwrap(),
                successor: case_successors.by_ref().next().unwrap(),
                successor_operands: flattened_case_operands.by_ref().take(count as usize).collect(),
            });
        }
        branches
    }
}

mlir_op!(Switch);
mlir_op_trait!(Switch, AlwaysSpeculatable);
mlir_op_trait!(Switch, IsTerminator);
mlir_op_trait!(Switch, NoMemoryEffect);
mlir_op_trait!(Switch, Pure);
mlir_op_trait!(Switch, SingleBlockRegions);
mlir_op_trait!(Switch, ZeroRegions);

/// Constructs a new detached/owned [`SwitchOperation`] at the specified [`Location`].
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
    let mut builder = OperationBuilder::new("cf.switch", location)
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
        .add_successors(cases.iter().map(|branch| &branch.successor).collect::<Vec<_>>().as_slice());
    if !cases.is_empty() {
        builder = builder.add_attribute(
            SWITCH_CASE_VALUES_ATTRIBUTE,
            context
                .dense_elements_attribute(
                    context.tensor_type(flag_type, &[Size::Static(cases.len())], None, location).unwrap(),
                    &cases.iter().map(|branch| branch.value).collect::<Vec<_>>(),
                )
                .unwrap(),
        );
    }
    builder
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
            assert_eq!(op.regions().count(), 0);
            assert_eq!(op.successors().count(), 0);
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
            assert_eq!(op.destination(), BlockRef::from(&target_block));
            assert_eq!(op.destination_operands(), vec![argument]);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 0);
            assert_eq!(op.regions().count(), 0);
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
            let unweighted_op =
                cond_br(predicate, &true_block, &false_block, &[true_value], &[false_value], &[], location);
            assert_eq!(unweighted_op.branch_weights(), None);
            let op = cond_br(predicate, &true_block, &false_block, &[true_value], &[false_value], &[13, 21], location);
            assert_eq!(op.predicate(), predicate);
            assert_eq!(op.on_true_successor(), BlockRef::from(&true_block));
            assert_eq!(op.on_false_successor(), BlockRef::from(&false_block));
            assert_eq!(op.on_true_successor_operands(), vec![true_value]);
            assert_eq!(op.on_false_successor_operands(), vec![false_value]);
            assert_eq!(op.branch_weights().map(Vec::<i32>::from), Some(vec![13, 21]));
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 0);
            assert_eq!(op.regions().count(), 0);
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
                    cf.cond_br %arg0 weights([13, 21]), ^bb1(%arg1 : i32), ^bb2(%arg2 : i32)
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
            let default_only_op = switch(flag, i32_type, default_branch.clone(), &[], location);
            assert_eq!(default_only_op.default(), default_branch);
            assert!(default_only_op.cases().is_empty());
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
            assert_eq!(op.regions().count(), 0);
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
