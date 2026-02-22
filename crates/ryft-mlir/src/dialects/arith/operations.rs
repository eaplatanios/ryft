use crate::{
    Attribute, AttributeRef, DetachedOp, DialectHandle, IntegerAttributeRef, Location, Operation, OperationBuilder,
    Value, ValueRef, mlir_binary_op, mlir_generic_unary_op, mlir_op, mlir_op_trait, mlir_unary_op,
};

pub const CONSTANT_VALUE_ATTRIBUTE: &'static str = "value";

pub trait ConstantOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    fn value(&self) -> AttributeRef<'c, 't> {
        self.attribute(CONSTANT_VALUE_ATTRIBUTE)
            .expect(&format!("invalid '{CONSTANT_VALUE_ATTRIBUTE}' attribute in `arith::constant`"))
    }
}

mlir_op!(Constant);
mlir_op_trait!(Constant, ConstantLike);
mlir_op_trait!(Constant, OneResult);
mlir_op_trait!(Constant, ZeroOperands);
mlir_op_trait!(Constant, ZeroRegions);
mlir_op_trait!(Constant, ZeroSuccessors);

pub fn constant<'c, 't: 'c, V: Attribute<'c, 't>, L: Location<'c, 't>>(
    value: V,
    location: L,
) -> DetachedConstantOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::arith());
    OperationBuilder::new("arith.constant", location)
        .add_attribute("value", value)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `arith::constant`")
}

mlir_generic_unary_op!(arith, bitcast);
mlir_generic_unary_op!(arith, extf);
mlir_generic_unary_op!(arith, extsi);
mlir_generic_unary_op!(arith, extui);
mlir_generic_unary_op!(arith, fptosi);
mlir_generic_unary_op!(arith, fptoui);
mlir_generic_unary_op!(arith, index_cast);
mlir_generic_unary_op!(arith, index_castui);
mlir_generic_unary_op!(arith, sitofp);
mlir_generic_unary_op!(arith, truncf);
mlir_generic_unary_op!(arith, trunci);
mlir_generic_unary_op!(arith, uitofp);

mlir_unary_op!(arith, negf);

mlir_binary_op!(arith, addf);
mlir_binary_op!(arith, addi);
mlir_binary_op!(arith, andi);
mlir_binary_op!(arith, ceildivsi);
mlir_binary_op!(arith, ceildivui);
mlir_binary_op!(arith, divf);
mlir_binary_op!(arith, divsi);
mlir_binary_op!(arith, divui);
mlir_binary_op!(arith, floordivsi);
mlir_binary_op!(arith, maximumf);
mlir_binary_op!(arith, maxnumf);
mlir_binary_op!(arith, maxsi);
mlir_binary_op!(arith, maxui);
mlir_binary_op!(arith, minimumf);
mlir_binary_op!(arith, minnumf);
mlir_binary_op!(arith, minsi);
mlir_binary_op!(arith, minui);
mlir_binary_op!(arith, mulf);
mlir_binary_op!(arith, muli);
mlir_binary_op!(arith, ori);
mlir_binary_op!(arith, remf);
mlir_binary_op!(arith, remsi);
mlir_binary_op!(arith, remui);
mlir_binary_op!(arith, shli);
mlir_binary_op!(arith, shrsi);
mlir_binary_op!(arith, shrui);
mlir_binary_op!(arith, subf);
mlir_binary_op!(arith, subi);
mlir_binary_op!(arith, xori);

pub const CMP_PREDICATE_ATTRIBUTE: &'static str = "predicate";

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(i64)]
pub enum FloatingPointComparisonPredicate {
    AlwaysFalse = 0,
    AlwaysTrue = 15,
    Equal = 1,
    NotEqual = 13,
    UnorderedOrEqual = 8,
    NotUnorderedOrEqual = 6,
    GreaterThan = 2,
    UnorderedOrGreaterThan = 9,
    GreaterThanOrEqual = 3,
    UnorderedGreaterThanOrEqual = 10,
    LessThan = 4,
    UnorderedOrLessThan = 11,
    LessThanOrEqual = 5,
    UnorderedLessThanOrEqual = 12,
    Ordered = 7,
    Unordered = 14,
}

pub trait CmpfOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }

    fn predicate(&self) -> FloatingPointComparisonPredicate {
        self.attribute(CMP_PREDICATE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .and_then(|attribute| match attribute.signless_value() {
                0 => Some(FloatingPointComparisonPredicate::AlwaysFalse),
                15 => Some(FloatingPointComparisonPredicate::AlwaysTrue),
                1 => Some(FloatingPointComparisonPredicate::Equal),
                13 => Some(FloatingPointComparisonPredicate::NotEqual),
                8 => Some(FloatingPointComparisonPredicate::UnorderedOrEqual),
                6 => Some(FloatingPointComparisonPredicate::NotUnorderedOrEqual),
                2 => Some(FloatingPointComparisonPredicate::GreaterThan),
                9 => Some(FloatingPointComparisonPredicate::UnorderedOrGreaterThan),
                3 => Some(FloatingPointComparisonPredicate::GreaterThanOrEqual),
                10 => Some(FloatingPointComparisonPredicate::UnorderedGreaterThanOrEqual),
                4 => Some(FloatingPointComparisonPredicate::LessThan),
                11 => Some(FloatingPointComparisonPredicate::UnorderedOrLessThan),
                5 => Some(FloatingPointComparisonPredicate::LessThanOrEqual),
                12 => Some(FloatingPointComparisonPredicate::UnorderedLessThanOrEqual),
                7 => Some(FloatingPointComparisonPredicate::Ordered),
                14 => Some(FloatingPointComparisonPredicate::Unordered),
                _ => None,
            })
            .expect(&format!("invalid '{CMP_PREDICATE_ATTRIBUTE}' attribute in `arith::cmpf`"))
    }
}

mlir_op!(Cmpf);
mlir_op_trait!(Cmpf, OneResult);
mlir_op_trait!(Cmpf, ZeroRegions);

pub fn cmpf<
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
    predicate: FloatingPointComparisonPredicate,
    location: L,
) -> DetachedCmpfOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::arith());
    OperationBuilder::new("arith.cmpf", location)
        .add_attribute(
            CMP_PREDICATE_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), predicate as i64),
        )
        .add_operand(lhs)
        .add_operand(rhs)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `arith::cmpf`")
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(i64)]
pub enum IntegerComparisonPredicate {
    Equal = 0,
    NotEqual = 1,
    SignedLessThan = 2,
    SignedLessThanOrEqual = 3,
    SignedGreaterThan = 4,
    SignedGreaterThanOrEqual = 5,
    UnsignedLessThan = 6,
    UnsignedLessThanOrEqual = 7,
    UnsignedGreaterThan = 8,
    UnsignedGreaterThanOrEqual = 9,
}

pub trait CmpiOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }

    fn predicate(&self) -> IntegerComparisonPredicate {
        self.attribute(CMP_PREDICATE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .and_then(|attribute| match attribute.signless_value() {
                0 => Some(IntegerComparisonPredicate::Equal),
                1 => Some(IntegerComparisonPredicate::NotEqual),
                2 => Some(IntegerComparisonPredicate::SignedLessThan),
                3 => Some(IntegerComparisonPredicate::SignedLessThanOrEqual),
                4 => Some(IntegerComparisonPredicate::SignedGreaterThan),
                5 => Some(IntegerComparisonPredicate::SignedGreaterThanOrEqual),
                6 => Some(IntegerComparisonPredicate::UnsignedLessThan),
                7 => Some(IntegerComparisonPredicate::UnsignedLessThanOrEqual),
                8 => Some(IntegerComparisonPredicate::UnsignedGreaterThan),
                9 => Some(IntegerComparisonPredicate::UnsignedGreaterThanOrEqual),
                _ => None,
            })
            .expect(&format!("invalid '{CMP_PREDICATE_ATTRIBUTE}' attribute in `arith::cmpi`"))
    }
}

mlir_op!(Cmpi);
mlir_op_trait!(Cmpi, OneResult);
mlir_op_trait!(Cmpi, ZeroRegions);

pub fn cmpi<
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
    predicate: IntegerComparisonPredicate,
    location: L,
) -> DetachedCmpiOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::arith());
    OperationBuilder::new("arith.cmpi", location)
        .add_attribute(
            CMP_PREDICATE_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), predicate as i64),
        )
        .add_operand(lhs)
        .add_operand(rhs)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `arith::cmpi`")
}

pub trait SelectOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    fn predicate(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    fn on_true(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }

    fn on_false(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(2).unwrap()
    }
}

mlir_op!(Select);
mlir_op_trait!(Select, OneResult);
mlir_op_trait!(Select, ZeroRegions);

pub fn select<
    'predicate,
    'on_true,
    'on_false,
    'c: 'predicate + 'on_true + 'on_false,
    't: 'c,
    P: Value<'predicate, 'c, 't>,
    T: Value<'on_true, 'c, 't>,
    F: Value<'on_false, 'c, 't>,
    L: Location<'c, 't>,
>(
    predicate: P,
    on_true: T,
    on_false: F,
    location: L,
) -> DetachedSelectOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::arith());
    OperationBuilder::new("arith.select", location)
        .add_operand(predicate)
        .add_operand(on_true)
        .add_operand(on_false)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `arith::select`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::{Block, Context, OneOperand, OneResult, Operation, Type, dialects::func};

    use super::*;

    #[test]
    fn test_constant() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let index_type = context.index_type();
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let value = context.integer_attribute(context.index_type(), 42);
            let op = constant(value, location);
            assert_eq!(op.value(), value);
            assert_eq!(op.output().r#type(), index_type);
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "constant_test",
                func::FuncAttributes { arguments: vec![], results: vec![index_type.into()], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @constant_test() -> index {
                    %c42 = arith.constant 42 : index
                    return %c42 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_cmpf() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i1_type = context.signless_integer_type(1);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type, location), (f32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = cmpf(lhs, rhs, FloatingPointComparisonPredicate::Unordered, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.predicate(), FloatingPointComparisonPredicate::Unordered);
            assert_eq!(op.output().r#type(), i1_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "cmpf_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), f32_type.into()],
                    results: vec![i1_type.into()],
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
                  func.func @cmpf_test(%arg0: f32, %arg1: f32) -> i1 {
                    %0 = arith.cmpf uno, %arg0, %arg1 : f32
                    return %0 : i1
                  }
                }
            "},
        );
    }

    #[test]
    fn test_cmpi() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location), (i32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = cmpi(lhs, rhs, IntegerComparisonPredicate::UnsignedGreaterThanOrEqual, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.predicate(), IntegerComparisonPredicate::UnsignedGreaterThanOrEqual);
            assert_eq!(op.output().r#type(), i1_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "cmpi_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
                    results: vec![i1_type.into()],
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
                  func.func @cmpi_test(%arg0: i32, %arg1: i32) -> i1 {
                    %0 = arith.cmpi uge, %arg0, %arg1 : i32
                    return %0 : i1
                  }
                }
            "},
        );
    }

    #[test]
    fn test_bitcast() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location)]);
            let input = block.argument(0).unwrap();
            let op = bitcast(input, f32_type, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), f32_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "bitcast_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into()],
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
                  func.func @bitcast_test(%arg0: i32) -> f32 {
                    %0 = arith.bitcast %arg0 : i32 to f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_extf() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let f64_type = context.float64_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type, location)]);
            let input = block.argument(0).unwrap();
            let op = extf(input, f64_type, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), f64_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "extf_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into()],
                    results: vec![f64_type.into()],
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
                  func.func @extf_test(%arg0: f32) -> f64 {
                    %0 = arith.extf %arg0 : f32 to f64
                    return %0 : f64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_extsi() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location)]);
            let input = block.argument(0).unwrap();
            let op = extsi(input, i64_type, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), i64_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "extsi_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into()],
                    results: vec![i64_type.into()],
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
                  func.func @extsi_test(%arg0: i32) -> i64 {
                    %0 = arith.extsi %arg0 : i32 to i64
                    return %0 : i64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_extui() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location)]);
            let input = block.argument(0).unwrap();
            let op = extui(input, i64_type, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), i64_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "extui_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into()],
                    results: vec![i64_type.into()],
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
                  func.func @extui_test(%arg0: i32) -> i64 {
                    %0 = arith.extui %arg0 : i32 to i64
                    return %0 : i64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_fptosi() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(f32_type, location)]);
            let input = block.argument(0).unwrap();
            let op = fptosi(input, i32_type, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "fptosi_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into()],
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
                  func.func @fptosi_test(%arg0: f32) -> i32 {
                    %0 = arith.fptosi %arg0 : f32 to i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_fptoui() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(f32_type, location)]);
            let input = block.argument(0).unwrap();
            let op = fptoui(input, i32_type, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "fptoui_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into()],
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
                  func.func @fptoui_test(%arg0: f32) -> i32 {
                    %0 = arith.fptoui %arg0 : f32 to i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_index_cast() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let index_type = context.index_type();
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(index_type, location)]);
            let input = block.argument(0).unwrap();
            let op = index_cast(input, i32_type, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "index_cast_test",
                func::FuncAttributes {
                    arguments: vec![index_type.into()],
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
                  func.func @index_cast_test(%arg0: index) -> i32 {
                    %0 = arith.index_cast %arg0 : index to i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_index_castui() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let index_type = context.index_type();
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(index_type, location)]);
            let input = block.argument(0).unwrap();
            let op = index_castui(input, i32_type, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "index_castui_test",
                func::FuncAttributes {
                    arguments: vec![index_type.into()],
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
                  func.func @index_castui_test(%arg0: index) -> i32 {
                    %0 = arith.index_castui %arg0 : index to i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_sitofp() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location)]);
            let input = block.argument(0).unwrap();
            let op = sitofp(input, f32_type, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), f32_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "sitofp_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into()],
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
                  func.func @sitofp_test(%arg0: i32) -> f32 {
                    %0 = arith.sitofp %arg0 : i32 to f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_truncf() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f64_type = context.float64_type();
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f64_type, location)]);
            let input = block.argument(0).unwrap();
            let op = truncf(input, f32_type, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), f32_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "truncf_test",
                func::FuncAttributes {
                    arguments: vec![f64_type.into()],
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
                  func.func @truncf_test(%arg0: f64) -> f32 {
                    %0 = arith.truncf %arg0 : f64 to f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_trunci() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i64_type, location)]);
            let input = block.argument(0).unwrap();
            let op = trunci(input, i32_type, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "trunci_test",
                func::FuncAttributes {
                    arguments: vec![i64_type.into()],
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
                  func.func @trunci_test(%arg0: i64) -> i32 {
                    %0 = arith.trunci %arg0 : i64 to i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_uitofp() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location)]);
            let input = block.argument(0).unwrap();
            let op = uitofp(input, f32_type, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), f32_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "uitofp_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into()],
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
                  func.func @uitofp_test(%arg0: i32) -> f32 {
                    %0 = arith.uitofp %arg0 : i32 to f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_negf() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type, location)]);
            let input = block.argument(0).unwrap();
            let op = negf(input, location);
            assert_eq!(op.input(), input);
            assert_eq!(op.output().r#type(), f32_type);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "negf_test",
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
                  func.func @negf_test(%arg0: f32) -> f32 {
                    %0 = arith.negf %arg0 : f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_addf() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type, location), (f32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = addf(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), f32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "addf_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), f32_type.into()],
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
                  func.func @addf_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = arith.addf %arg0, %arg1 : f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_addi() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location), (i32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = addi(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "addi_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
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
                  func.func @addi_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = arith.addi %arg0, %arg1 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_andi() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location), (i32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = andi(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "andi_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
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
                  func.func @andi_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = arith.andi %arg0, %arg1 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_ceildivsi() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location), (i32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = ceildivsi(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "ceildivsi_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
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
                  func.func @ceildivsi_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = arith.ceildivsi %arg0, %arg1 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_ceildivui() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location), (i32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = ceildivui(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "ceildivui_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
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
                  func.func @ceildivui_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = arith.ceildivui %arg0, %arg1 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_divf() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type, location), (f32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = divf(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), f32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "divf_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), f32_type.into()],
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
                  func.func @divf_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = arith.divf %arg0, %arg1 : f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_divsi() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location), (i32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = divsi(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "divsi_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
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
                  func.func @divsi_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = arith.divsi %arg0, %arg1 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_divui() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location), (i32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = divui(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "divui_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
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
                  func.func @divui_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = arith.divui %arg0, %arg1 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_floordivsi() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location), (i32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = floordivsi(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "floordivsi_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
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
                  func.func @floordivsi_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = arith.floordivsi %arg0, %arg1 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_maximumf() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type, location), (f32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = maximumf(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), f32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "maximumf_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), f32_type.into()],
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
                  func.func @maximumf_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = arith.maximumf %arg0, %arg1 : f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_maxnumf() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type, location), (f32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = maxnumf(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), f32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "maxnumf_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), f32_type.into()],
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
                  func.func @maxnumf_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = arith.maxnumf %arg0, %arg1 : f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_maxsi() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location), (i32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = maxsi(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "maxsi_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
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
                  func.func @maxsi_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = arith.maxsi %arg0, %arg1 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_maxui() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location), (i32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = maxui(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "maxui_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
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
                  func.func @maxui_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = arith.maxui %arg0, %arg1 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_minimumf() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type, location), (f32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = minimumf(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), f32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "minimumf_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), f32_type.into()],
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
                  func.func @minimumf_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = arith.minimumf %arg0, %arg1 : f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_minnumf() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type, location), (f32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = minnumf(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), f32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "minnumf_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), f32_type.into()],
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
                  func.func @minnumf_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = arith.minnumf %arg0, %arg1 : f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_minsi() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location), (i32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = minsi(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "minsi_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
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
                  func.func @minsi_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = arith.minsi %arg0, %arg1 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_minui() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location), (i32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = minui(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "minui_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
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
                  func.func @minui_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = arith.minui %arg0, %arg1 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_mulf() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type, location), (f32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = mulf(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), f32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "mulf_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), f32_type.into()],
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
                  func.func @mulf_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = arith.mulf %arg0, %arg1 : f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_muli() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location), (i32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = muli(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "muli_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
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
                  func.func @muli_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = arith.muli %arg0, %arg1 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_ori() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location), (i32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = ori(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "ori_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
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
                  func.func @ori_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = arith.ori %arg0, %arg1 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_remf() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type, location), (f32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = remf(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), f32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "remf_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), f32_type.into()],
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
                  func.func @remf_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = arith.remf %arg0, %arg1 : f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_remsi() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location), (i32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = remsi(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "remsi_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
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
                  func.func @remsi_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = arith.remsi %arg0, %arg1 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_remui() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location), (i32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = remui(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "remui_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
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
                  func.func @remui_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = arith.remui %arg0, %arg1 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_shli() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location), (i32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = shli(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "shli_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
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
                  func.func @shli_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = arith.shli %arg0, %arg1 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_shrsi() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location), (i32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = shrsi(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "shrsi_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
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
                  func.func @shrsi_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = arith.shrsi %arg0, %arg1 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_shrui() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location), (i32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = shrui(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "shrui_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
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
                  func.func @shrui_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = arith.shrui %arg0, %arg1 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_subf() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type, location), (f32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = subf(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), f32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "subf_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), f32_type.into()],
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
                  func.func @subf_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = arith.subf %arg0, %arg1 : f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_subi() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location), (i32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = subi(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "subi_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
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
                  func.func @subi_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = arith.subi %arg0, %arg1 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_xori() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location), (i32_type, location)]);
            let lhs = block.argument(0).unwrap();
            let rhs = block.argument(1).unwrap();
            let op = xori(lhs, rhs, location);
            assert_eq!(op.lhs(), lhs);
            assert_eq!(op.rhs(), rhs);
            assert_eq!(op.output().r#type(), i32_type);
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "xori_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
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
                  func.func @xori_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = arith.xori %arg0, %arg1 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_select() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i1_type = context.signless_integer_type(1).as_ref();
        let f32_type = context.float32_type().as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(i1_type, location), (f32_type, location), (f32_type, location)]);
            let predicate = block.argument(0).unwrap();
            let on_true = block.argument(1).unwrap();
            let on_false = block.argument(2).unwrap();
            let op = select(predicate, on_true, on_false, location);
            assert_eq!(op.predicate(), predicate);
            assert_eq!(op.on_true(), on_true);
            assert_eq!(op.on_false(), on_false);
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.result(0).unwrap().r#type(), f32_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "select_test",
                func::FuncAttributes {
                    arguments: vec![i1_type.into(), f32_type.into(), f32_type.into()],
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
                  func.func @select_test(%arg0: i1, %arg1: f32, %arg2: f32) -> f32 {
                    %0 = arith.select %arg0, %arg1, %arg2 : f32
                    return %0 : f32
                  }
                }
            "}
        );
    }
}
