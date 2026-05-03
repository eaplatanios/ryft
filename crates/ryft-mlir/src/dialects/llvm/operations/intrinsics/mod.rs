pub mod constrained;
pub mod coro;
pub mod debug;
pub mod math;
pub mod memory;
pub mod vector;
pub mod vp;

pub use constrained::*;
pub use coro::*;
pub use debug::*;
pub use math::*;
pub use memory::*;
pub use vector::*;
pub use vp::*;

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Attribute, Block, Context, DialectHandle, Operation, Type};

    use super::*;

    #[test]
    fn test_intr_acos() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_acos(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.acos");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_acos_test",
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
                  func.func @llvm_intr_acos_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.acos(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_asin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_asin(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.asin");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_asin_test",
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
                  func.func @llvm_intr_asin_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.asin(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_atan2() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location), (f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_atan2(arg_0, arg_1, f32_type, None, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.atan2");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_atan2_test",
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
                  func.func @llvm_intr_atan2_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = llvm.intr.atan2(%arg0, %arg1) : (f32, f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_atan() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_atan(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.atan");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_atan_test",
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
                  func.func @llvm_intr_atan_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.atan(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_abs() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let is_int_min_poison = context.boolean_attribute(false).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_abs(arg_0, i32_type, is_int_min_poison, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.is_int_min_poison(), is_int_min_poison);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.abs");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_abs_test",
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
                  func.func @llvm_intr_abs_test(%arg0: i32) -> i32 {
                    %0 = \"llvm.intr.abs\"(%arg0) <{is_int_min_poison = false}> : (i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_annotation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0);
        module.body().append_operation({
            let mut block = context.block(&[
                (i32_type.as_ref(), location),
                (pointer_type.as_ref(), location),
                (pointer_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_annotation(arg_0, arg_1, arg_2, arg_3, i32_type, location);
            assert_eq!(op.integer(), arg_0);
            assert_eq!(op.annotation(), arg_1);
            assert_eq!(op.file_name(), arg_2);
            assert_eq!(op.line(), arg_3);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.annotation");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_annotation_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), pointer_type.into(), pointer_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_annotation_test(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32) -> i32 {
                    %0 = \"llvm.intr.annotation\"(%arg0, %arg1, %arg2, %arg3) : (i32, !llvm.ptr, !llvm.ptr, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_assume() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i1_type = context.signless_integer_type(1);
        module.body().append_operation({
            let mut block = context.block(&[(i1_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_assume(arg_0, location);
            assert_eq!(op.condition(), arg_0);
            assert_eq!(op.operation_name(), "llvm.intr.assume");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_assume_test",
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
                  func.func @llvm_intr_assume_test(%arg0: i1) {
                    llvm.intr.assume %arg0 : i1
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_bit_reverse() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_bit_reverse(arg_0, i32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.bitreverse");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_bitreverse_test",
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
                  func.func @llvm_intr_bitreverse_test(%arg0: i32) -> i32 {
                    %0 = llvm.intr.bitreverse(%arg0) : (i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_bswap() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_bswap(arg_0, i32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.bswap");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_bswap_test",
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
                  func.func @llvm_intr_bswap_test(%arg0: i32) -> i32 {
                    %0 = llvm.intr.bswap(%arg0) : (i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_experimental_constrained_fadd() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let f32_type = context.float32_type();
        let roundingmode = context.integer_attribute(i64_type, 1).as_ref();
        let fp_exception_behavior = context.integer_attribute(i64_type, 0).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location), (f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_experimental_constrained_fadd(
                arg_0,
                arg_1,
                f32_type,
                roundingmode,
                fp_exception_behavior,
                location,
            );
            assert_eq!(op.argument_0(), arg_0);
            assert_eq!(op.argument_1(), arg_1);
            assert_eq!(op.roundingmode(), roundingmode);
            assert_eq!(op.fp_exception_behavior(), fp_exception_behavior);
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.experimental.constrained.fadd");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_experimental_constrained_fadd_test",
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
                  func.func @llvm_intr_experimental_constrained_fadd_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = llvm.intr.experimental.constrained.fadd %arg0, %arg1 tonearest ignore : f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_experimental_constrained_fdiv() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let f32_type = context.float32_type();
        let roundingmode = context.integer_attribute(i64_type, 1).as_ref();
        let fp_exception_behavior = context.integer_attribute(i64_type, 0).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location), (f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_experimental_constrained_fdiv(
                arg_0,
                arg_1,
                f32_type,
                roundingmode,
                fp_exception_behavior,
                location,
            );
            assert_eq!(op.argument_0(), arg_0);
            assert_eq!(op.argument_1(), arg_1);
            assert_eq!(op.roundingmode(), roundingmode);
            assert_eq!(op.fp_exception_behavior(), fp_exception_behavior);
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.experimental.constrained.fdiv");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_experimental_constrained_fdiv_test",
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
                  func.func @llvm_intr_experimental_constrained_fdiv_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = llvm.intr.experimental.constrained.fdiv %arg0, %arg1 tonearest ignore : f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_experimental_constrained_fma() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let f32_type = context.float32_type();
        let roundingmode = context.integer_attribute(i64_type, 1).as_ref();
        let fp_exception_behavior = context.integer_attribute(i64_type, 0).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[
                (f32_type.as_ref(), location),
                (f32_type.as_ref(), location),
                (f32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_experimental_constrained_fma(
                arg_0,
                arg_1,
                arg_2,
                f32_type,
                roundingmode,
                fp_exception_behavior,
                location,
            );
            assert_eq!(op.argument_0(), arg_0);
            assert_eq!(op.argument_1(), arg_1);
            assert_eq!(op.argument_2(), arg_2);
            assert_eq!(op.roundingmode(), roundingmode);
            assert_eq!(op.fp_exception_behavior(), fp_exception_behavior);
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.experimental.constrained.fma");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_experimental_constrained_fma_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), f32_type.into(), f32_type.into()],
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
                  func.func @llvm_intr_experimental_constrained_fma_test(%arg0: f32, %arg1: f32, %arg2: f32) -> f32 {
                    %0 = llvm.intr.experimental.constrained.fma %arg0, %arg1, %arg2 tonearest ignore : f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_experimental_constrained_fmuladd() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let f32_type = context.float32_type();
        let roundingmode = context.integer_attribute(i64_type, 1).as_ref();
        let fp_exception_behavior = context.integer_attribute(i64_type, 0).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[
                (f32_type.as_ref(), location),
                (f32_type.as_ref(), location),
                (f32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_experimental_constrained_fmuladd(
                arg_0,
                arg_1,
                arg_2,
                f32_type,
                roundingmode,
                fp_exception_behavior,
                location,
            );
            assert_eq!(op.argument_0(), arg_0);
            assert_eq!(op.argument_1(), arg_1);
            assert_eq!(op.argument_2(), arg_2);
            assert_eq!(op.roundingmode(), roundingmode);
            assert_eq!(op.fp_exception_behavior(), fp_exception_behavior);
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.experimental.constrained.fmuladd");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_experimental_constrained_fmuladd_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), f32_type.into(), f32_type.into()],
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
                  func.func @llvm_intr_experimental_constrained_fmuladd_test(%arg0: f32, %arg1: f32, %arg2: f32) -> f32 {
                    %0 = llvm.intr.experimental.constrained.fmuladd %arg0, %arg1, %arg2 tonearest ignore : f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_experimental_constrained_fmul() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let f32_type = context.float32_type();
        let roundingmode = context.integer_attribute(i64_type, 1).as_ref();
        let fp_exception_behavior = context.integer_attribute(i64_type, 0).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location), (f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_experimental_constrained_fmul(
                arg_0,
                arg_1,
                f32_type,
                roundingmode,
                fp_exception_behavior,
                location,
            );
            assert_eq!(op.argument_0(), arg_0);
            assert_eq!(op.argument_1(), arg_1);
            assert_eq!(op.roundingmode(), roundingmode);
            assert_eq!(op.fp_exception_behavior(), fp_exception_behavior);
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.experimental.constrained.fmul");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_experimental_constrained_fmul_test",
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
                  func.func @llvm_intr_experimental_constrained_fmul_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = llvm.intr.experimental.constrained.fmul %arg0, %arg1 tonearest ignore : f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_experimental_constrained_fpext() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let f32_type = context.float32_type();
        let f64_type = context.float64_type();
        let fp_exception_behavior = context.integer_attribute(i64_type, 0).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_experimental_constrained_fpext(arg_0, f64_type, fp_exception_behavior, location);
            assert_eq!(op.argument_0(), arg_0);
            assert_eq!(op.fp_exception_behavior(), fp_exception_behavior);
            assert_eq!(op.output_type(), f64_type);
            assert_eq!(op.operation_name(), "llvm.intr.experimental.constrained.fpext");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_experimental_constrained_fpext_test",
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
                  func.func @llvm_intr_experimental_constrained_fpext_test(%arg0: f32) -> f64 {
                    %0 = llvm.intr.experimental.constrained.fpext %arg0 ignore : f32 to f64
                    return %0 : f64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_experimental_constrained_fptrunc() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let f32_type = context.float32_type();
        let roundingmode = context.integer_attribute(i64_type, 1).as_ref();
        let fp_exception_behavior = context.integer_attribute(i64_type, 0).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op =
                intr_experimental_constrained_fptrunc(arg_0, f32_type, roundingmode, fp_exception_behavior, location);
            assert_eq!(op.argument_0(), arg_0);
            assert_eq!(op.roundingmode(), roundingmode);
            assert_eq!(op.fp_exception_behavior(), fp_exception_behavior);
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.experimental.constrained.fptrunc");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_experimental_constrained_fptrunc_test",
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
                  func.func @llvm_intr_experimental_constrained_fptrunc_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.experimental.constrained.fptrunc %arg0 tonearest ignore : f32 to f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_experimental_constrained_frem() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let f32_type = context.float32_type();
        let roundingmode = context.integer_attribute(i64_type, 1).as_ref();
        let fp_exception_behavior = context.integer_attribute(i64_type, 0).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location), (f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_experimental_constrained_frem(
                arg_0,
                arg_1,
                f32_type,
                roundingmode,
                fp_exception_behavior,
                location,
            );
            assert_eq!(op.argument_0(), arg_0);
            assert_eq!(op.argument_1(), arg_1);
            assert_eq!(op.roundingmode(), roundingmode);
            assert_eq!(op.fp_exception_behavior(), fp_exception_behavior);
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.experimental.constrained.frem");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_experimental_constrained_frem_test",
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
                  func.func @llvm_intr_experimental_constrained_frem_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = llvm.intr.experimental.constrained.frem %arg0, %arg1 tonearest ignore : f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_experimental_constrained_fsub() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let f32_type = context.float32_type();
        let roundingmode = context.integer_attribute(i64_type, 1).as_ref();
        let fp_exception_behavior = context.integer_attribute(i64_type, 0).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location), (f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_experimental_constrained_fsub(
                arg_0,
                arg_1,
                f32_type,
                roundingmode,
                fp_exception_behavior,
                location,
            );
            assert_eq!(op.argument_0(), arg_0);
            assert_eq!(op.argument_1(), arg_1);
            assert_eq!(op.roundingmode(), roundingmode);
            assert_eq!(op.fp_exception_behavior(), fp_exception_behavior);
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.experimental.constrained.fsub");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_experimental_constrained_fsub_test",
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
                  func.func @llvm_intr_experimental_constrained_fsub_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = llvm.intr.experimental.constrained.fsub %arg0, %arg1 tonearest ignore : f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_experimental_constrained_sito_fp() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let f32_type = context.float32_type();
        let roundingmode = context.integer_attribute(i64_type, 1).as_ref();
        let fp_exception_behavior = context.integer_attribute(i64_type, 0).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op =
                intr_experimental_constrained_sito_fp(arg_0, f32_type, roundingmode, fp_exception_behavior, location);
            assert_eq!(op.argument_0(), arg_0);
            assert_eq!(op.roundingmode(), roundingmode);
            assert_eq!(op.fp_exception_behavior(), fp_exception_behavior);
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.experimental.constrained.sitofp");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_experimental_constrained_sitofp_test",
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
                  func.func @llvm_intr_experimental_constrained_sitofp_test(%arg0: i32) -> f32 {
                    %0 = llvm.intr.experimental.constrained.sitofp %arg0 tonearest ignore : i32 to f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_experimental_constrained_uito_fp() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let f32_type = context.float32_type();
        let roundingmode = context.integer_attribute(i64_type, 1).as_ref();
        let fp_exception_behavior = context.integer_attribute(i64_type, 0).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op =
                intr_experimental_constrained_uito_fp(arg_0, f32_type, roundingmode, fp_exception_behavior, location);
            assert_eq!(op.argument_0(), arg_0);
            assert_eq!(op.roundingmode(), roundingmode);
            assert_eq!(op.fp_exception_behavior(), fp_exception_behavior);
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.experimental.constrained.uitofp");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_experimental_constrained_uitofp_test",
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
                  func.func @llvm_intr_experimental_constrained_uitofp_test(%arg0: i32) -> f32 {
                    %0 = llvm.intr.experimental.constrained.uitofp %arg0 tonearest ignore : i32 to f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_copy_sign() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location), (f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_copy_sign(arg_0, arg_1, f32_type, None, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.copysign");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_copysign_test",
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
                  func.func @llvm_intr_copysign_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = llvm.intr.copysign(%arg0, %arg1) : (f32, f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_coro_align() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = intr_coro_align(i64_type.as_ref(), location);
            assert_eq!(op.output_type(), i64_type);
            assert_eq!(op.operation_name(), "llvm.intr.coro.align");
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_coro_align_test",
                func::FuncAttributes { arguments: vec![], results: vec![i64_type.into()], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_intr_coro_align_test() -> i64 {
                    %0 = llvm.intr.coro.align : i64
                    return %0 : i64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_coro_begin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        let token_type = context.llvm_token_type();
        module.body().append_operation({
            let mut block = context.block(&[(token_type.as_ref(), location), (pointer_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_coro_begin(arg_0, arg_1, pointer_type, location);
            assert_eq!(op.token(), arg_0);
            assert_eq!(op.memory(), arg_1);
            assert_eq!(op.output_type(), pointer_type);
            assert_eq!(op.operation_name(), "llvm.intr.coro.begin");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_coro_begin_test",
                func::FuncAttributes {
                    arguments: vec![token_type.into(), pointer_type.into()],
                    results: vec![pointer_type.into()],
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
                  func.func @llvm_intr_coro_begin_test(%arg0: !llvm.token, %arg1: !llvm.ptr) -> !llvm.ptr {
                    %0 = llvm.intr.coro.begin %arg0, %arg1 : (!llvm.token, !llvm.ptr) -> !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_coro_end() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i1_type = context.signless_integer_type(1);
        let pointer_type = context.llvm_pointer_type(0);
        let token_type = context.llvm_token_type();
        module.body().append_operation({
            let mut block = context.block(&[
                (pointer_type.as_ref(), location),
                (i1_type.as_ref(), location),
                (token_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_coro_end(arg_0, arg_1, arg_2, i1_type, location);
            assert_eq!(op.handle(), arg_0);
            assert_eq!(op.unwind(), arg_1);
            assert_eq!(op.return_values(), arg_2);
            assert_eq!(op.output_type(), i1_type);
            assert_eq!(op.operation_name(), "llvm.intr.coro.end");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_coro_end_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into(), i1_type.into(), token_type.into()],
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
                  func.func @llvm_intr_coro_end_test(%arg0: !llvm.ptr, %arg1: i1, %arg2: !llvm.token) -> i1 {
                    %0 = llvm.intr.coro.end %arg0, %arg1, %arg2 : (!llvm.ptr, i1, !llvm.token) -> i1
                    return %0 : i1
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_coro_free() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        let token_type = context.llvm_token_type();
        module.body().append_operation({
            let mut block = context.block(&[(token_type.as_ref(), location), (pointer_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_coro_free(arg_0, arg_1, pointer_type, location);
            assert_eq!(op.id(), arg_0);
            assert_eq!(op.handle(), arg_1);
            assert_eq!(op.output_type(), pointer_type);
            assert_eq!(op.operation_name(), "llvm.intr.coro.free");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_coro_free_test",
                func::FuncAttributes {
                    arguments: vec![token_type.into(), pointer_type.into()],
                    results: vec![pointer_type.into()],
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
                  func.func @llvm_intr_coro_free_test(%arg0: !llvm.token, %arg1: !llvm.ptr) -> !llvm.ptr {
                    %0 = llvm.intr.coro.free %arg0, %arg1 : (!llvm.token, !llvm.ptr) -> !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_coro_id() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0);
        let token_type = context.llvm_token_type();
        module.body().append_operation({
            let mut block = context.block(&[
                (i32_type.as_ref(), location),
                (pointer_type.as_ref(), location),
                (pointer_type.as_ref(), location),
                (pointer_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_coro_id(arg_0, arg_1, arg_2, arg_3, token_type, location);
            assert_eq!(op.alignment(), arg_0);
            assert_eq!(op.promise(), arg_1);
            assert_eq!(op.coroutine_address(), arg_2);
            assert_eq!(op.function_addresses(), arg_3);
            assert_eq!(op.output_type(), token_type);
            assert_eq!(op.operation_name(), "llvm.intr.coro.id");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_coro_id_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), pointer_type.into(), pointer_type.into(), pointer_type.into()],
                    results: vec![token_type.into()],
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
                  func.func @llvm_intr_coro_id_test(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) -> !llvm.token {
                    %0 = llvm.intr.coro.id %arg0, %arg1, %arg2, %arg3 : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> !llvm.token
                    return %0 : !llvm.token
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_coro_promise() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0);
        module.body().append_operation({
            let mut block = context.block(&[
                (pointer_type.as_ref(), location),
                (i32_type.as_ref(), location),
                (i1_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_coro_promise(arg_0, arg_1, arg_2, pointer_type, location);
            assert_eq!(op.handle(), arg_0);
            assert_eq!(op.alignment(), arg_1);
            assert_eq!(op.from(), arg_2);
            assert_eq!(op.output_type(), pointer_type);
            assert_eq!(op.operation_name(), "llvm.intr.coro.promise");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_coro_promise_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into(), i32_type.into(), i1_type.into()],
                    results: vec![pointer_type.into()],
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
                  func.func @llvm_intr_coro_promise_test(%arg0: !llvm.ptr, %arg1: i32, %arg2: i1) -> !llvm.ptr {
                    %0 = llvm.intr.coro.promise %arg0, %arg1, %arg2 : (!llvm.ptr, i32, i1) -> !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_coro_resume() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_coro_resume(arg_0, location);
            assert_eq!(op.handle(), arg_0);
            assert_eq!(op.operation_name(), "llvm.intr.coro.resume");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_coro_resume_test",
                func::FuncAttributes { arguments: vec![pointer_type.into()], results: vec![], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_intr_coro_resume_test(%arg0: !llvm.ptr) {
                    llvm.intr.coro.resume %arg0 : !llvm.ptr
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_coro_save() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        let token_type = context.llvm_token_type();
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_coro_save(arg_0, token_type, location);
            assert_eq!(op.handle(), arg_0);
            assert_eq!(op.output_type(), token_type);
            assert_eq!(op.operation_name(), "llvm.intr.coro.save");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_coro_save_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into()],
                    results: vec![token_type.into()],
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
                  func.func @llvm_intr_coro_save_test(%arg0: !llvm.ptr) -> !llvm.token {
                    %0 = llvm.intr.coro.save %arg0 : (!llvm.ptr) -> !llvm.token
                    return %0 : !llvm.token
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_coro_size() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = intr_coro_size(i64_type.as_ref(), location);
            assert_eq!(op.output_type(), i64_type);
            assert_eq!(op.operation_name(), "llvm.intr.coro.size");
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_coro_size_test",
                func::FuncAttributes { arguments: vec![], results: vec![i64_type.into()], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_intr_coro_size_test() -> i64 {
                    %0 = llvm.intr.coro.size : i64
                    return %0 : i64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_coro_suspend() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i1_type = context.signless_integer_type(1);
        let i8_type = context.signless_integer_type(8);
        let token_type = context.llvm_token_type();
        module.body().append_operation({
            let mut block = context.block(&[(token_type.as_ref(), location), (i1_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_coro_suspend(arg_0, arg_1, i8_type, location);
            assert_eq!(op.save(), arg_0);
            assert_eq!(op.final_suspend(), arg_1);
            assert_eq!(op.output_type(), i8_type);
            assert_eq!(op.operation_name(), "llvm.intr.coro.suspend");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_coro_suspend_test",
                func::FuncAttributes {
                    arguments: vec![token_type.into(), i1_type.into()],
                    results: vec![i8_type.into()],
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
                  func.func @llvm_intr_coro_suspend_test(%arg0: !llvm.token, %arg1: i1) -> i8 {
                    %0 = llvm.intr.coro.suspend %arg0, %arg1 : i8
                    return %0 : i8
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_cos() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_cos(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.cos");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_cos_test",
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
                  func.func @llvm_intr_cos_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.cos(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_cosh() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_cosh(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.cosh");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_cosh_test",
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
                  func.func @llvm_intr_cosh_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.cosh(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_ctlz() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let is_zero_poison = context.boolean_attribute(false).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_ctlz(arg_0, i32_type, is_zero_poison, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.is_zero_poison(), is_zero_poison);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.ctlz");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_ctlz_test",
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
                  func.func @llvm_intr_ctlz_test(%arg0: i32) -> i32 {
                    %0 = \"llvm.intr.ctlz\"(%arg0) <{is_zero_poison = false}> : (i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_count_trailing_zeros() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let is_zero_poison = context.boolean_attribute(false).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_count_trailing_zeros(arg_0, i32_type, is_zero_poison, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.is_zero_poison(), is_zero_poison);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.cttz");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_cttz_test",
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
                  func.func @llvm_intr_cttz_test(%arg0: i32) -> i32 {
                    %0 = \"llvm.intr.cttz\"(%arg0) <{is_zero_poison = false}> : (i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_ct_pop() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_ct_pop(arg_0, i32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.ctpop");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_ctpop_test",
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
                  func.func @llvm_intr_ctpop_test(%arg0: i32) -> i32 {
                    %0 = llvm.intr.ctpop(%arg0) : (i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_dbg_declare() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        let var_info = context.parse_attribute(r#"#llvm.di_local_variable<scope = #llvm.di_file<"file.c" in "/tmp">, name = "x", file = #llvm.di_file<"file.c" in "/tmp">, line = 1>"#).unwrap();
        let location_expr = context.llvm_di_expression_attribute(&[]).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_dbg_declare(arg_0, var_info, location_expr, location);
            assert_eq!(op.address(), arg_0);
            assert_eq!(op.var_info(), var_info);
            assert_eq!(op.location_expr(), location_expr);
            assert_eq!(op.operation_name(), "llvm.intr.dbg.declare");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_dbg_declare_test",
                func::FuncAttributes { arguments: vec![pointer_type.into()], results: vec![], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                #di_file = #llvm.di_file<\"file.c\" in \"/tmp\">
                #di_local_variable = #llvm.di_local_variable<scope = #di_file, name = \"x\", file = #di_file, line = 1>
                module {
                  func.func @llvm_intr_dbg_declare_test(%arg0: !llvm.ptr) {
                    llvm.intr.dbg.declare #di_local_variable = %arg0 : !llvm.ptr
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_dbg_label() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let label = context.parse_attribute(r#"#llvm.di_label<scope = #llvm.di_file<"file.c" in "/tmp">, name = "label", file = #llvm.di_file<"file.c" in "/tmp">, line = 1>"#).unwrap();
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = intr_dbg_label(label, location);
            assert_eq!(op.label(), label);
            assert_eq!(op.operation_name(), "llvm.intr.dbg.label");
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_dbg_label_test",
                func::FuncAttributes { arguments: vec![], results: vec![], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                #di_file = #llvm.di_file<\"file.c\" in \"/tmp\">
                #di_label = #llvm.di_label<scope = #di_file, name = \"label\", file = #di_file, line = 1>
                module {
                  func.func @llvm_intr_dbg_label_test() {
                    llvm.intr.dbg.label #di_label
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_dbg_value() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let var_info = context.parse_attribute(r#"#llvm.di_local_variable<scope = #llvm.di_file<"file.c" in "/tmp">, name = "x", file = #llvm.di_file<"file.c" in "/tmp">, line = 1>"#).unwrap();
        let location_expr = context.llvm_di_expression_attribute(&[]).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_dbg_value(arg_0, var_info, location_expr, location);
            assert_eq!(op.value(), arg_0);
            assert_eq!(op.var_info(), var_info);
            assert_eq!(op.location_expr(), location_expr);
            assert_eq!(op.operation_name(), "llvm.intr.dbg.value");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_dbg_value_test",
                func::FuncAttributes { arguments: vec![i32_type.into()], results: vec![], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                #di_file = #llvm.di_file<\"file.c\" in \"/tmp\">
                #di_local_variable = #llvm.di_local_variable<scope = #di_file, name = \"x\", file = #di_file, line = 1>
                module {
                  func.func @llvm_intr_dbg_value_test(%arg0: i32) {
                    llvm.intr.dbg.value #di_local_variable = %arg0 : i32
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_debug_trap() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = intr_debug_trap(location);
            assert_eq!(op.operation_name(), "llvm.intr.debugtrap");
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_debugtrap_test",
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
                  func.func @llvm_intr_debugtrap_test() {
                    llvm.intr.debugtrap
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_eh_type_id_for() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0);
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_eh_type_id_for(arg_0, i32_type, location);
            assert_eq!(op.type_info(), arg_0);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.eh.typeid.for");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_eh_typeid_for_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into()],
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
                  func.func @llvm_intr_eh_typeid_for_test(%arg0: !llvm.ptr) -> i32 {
                    %0 = llvm.intr.eh.typeid.for %arg0 : (!llvm.ptr) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_exp10() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_exp10(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.exp10");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_exp10_test",
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
                  func.func @llvm_intr_exp10_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.exp10(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_exp2() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_exp2(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.exp2");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_exp2_test",
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
                  func.func @llvm_intr_exp2_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.exp2(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_exp() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_exp(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.exp");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_exp_test",
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
                  func.func @llvm_intr_exp_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.exp(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_expect() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_expect(arg_0, arg_1, i32_type, location);
            assert_eq!(op.value(), arg_0);
            assert_eq!(op.expected(), arg_1);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.expect");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_expect_test",
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
                  func.func @llvm_intr_expect_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.expect %arg0, %arg1 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_expect_with_probability() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let f64_type = context.float64_type();
        let prob = context.float_attribute(f64_type, 5.000000e-01).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_expect_with_probability(arg_0, arg_1, i32_type, prob, location);
            assert_eq!(op.value(), arg_0);
            assert_eq!(op.expected(), arg_1);
            assert_eq!(op.prob(), prob);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.expect.with.probability");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_expect_with_probability_test",
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
                  func.func @llvm_intr_expect_with_probability_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.expect.with.probability %arg0, %arg1, 5.000000e-01 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_fabs() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_fabs(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.fabs");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_fabs_test",
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
                  func.func @llvm_intr_fabs_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.fabs(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_ceil() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_ceil(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.ceil");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_ceil_test",
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
                  func.func @llvm_intr_ceil_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.ceil(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_floor() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_floor(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.floor");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_floor_test",
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
                  func.func @llvm_intr_floor_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.floor(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_fma() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[
                (f32_type.as_ref(), location),
                (f32_type.as_ref(), location),
                (f32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_fma(arg_0, arg_1, arg_2, f32_type, None, location);
            assert_eq!(op.first(), arg_0);
            assert_eq!(op.second(), arg_1);
            assert_eq!(op.third(), arg_2);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.fma");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_fma_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), f32_type.into(), f32_type.into()],
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
                  func.func @llvm_intr_fma_test(%arg0: f32, %arg1: f32, %arg2: f32) -> f32 {
                    %0 = llvm.intr.fma(%arg0, %arg1, %arg2) : (f32, f32, f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_fmuladd() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[
                (f32_type.as_ref(), location),
                (f32_type.as_ref(), location),
                (f32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_fmuladd(arg_0, arg_1, arg_2, f32_type, None, location);
            assert_eq!(op.first(), arg_0);
            assert_eq!(op.second(), arg_1);
            assert_eq!(op.third(), arg_2);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.fmuladd");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_fmuladd_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), f32_type.into(), f32_type.into()],
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
                  func.func @llvm_intr_fmuladd_test(%arg0: f32, %arg1: f32, %arg2: f32) -> f32 {
                    %0 = llvm.intr.fmuladd(%arg0, %arg1, %arg2) : (f32, f32, f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_trunc() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_trunc(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.trunc");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_trunc_test",
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
                  func.func @llvm_intr_trunc_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.trunc(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_fake_use() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_fake_use(&[arg_0.into(), arg_1.into()], location);
            assert_eq!(op.arguments(), vec![arg_0, arg_1]);
            assert_eq!(op.operation_name(), "llvm.intr.fake.use");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_fake_use_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
                    results: vec![],
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
                  func.func @llvm_intr_fake_use_test(%arg0: i32, %arg1: i32) {
                    llvm.intr.fake.use %arg0, %arg1 : i32, i32
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_frexp() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let result_type = context.llvm_literal_struct_type(&[f32_type.as_ref(), i32_type.as_ref()], false);
            let op = intr_frexp(arg_0, result_type, None, location);
            assert_eq!(op.value(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), result_type);
            assert_eq!(op.operation_name(), "llvm.intr.frexp");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_frexp_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into()],
                    results: vec![result_type.into()],
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
                  func.func @llvm_intr_frexp_test(%arg0: f32) -> !llvm.struct<(f32, i32)> {
                    %0 = llvm.intr.frexp(%arg0) : (f32) -> !llvm.struct<(f32, i32)>
                    return %0 : !llvm.struct<(f32, i32)>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_fshl() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[
                (i32_type.as_ref(), location),
                (i32_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_fshl(arg_0, arg_1, arg_2, i32_type, location);
            assert_eq!(op.first(), arg_0);
            assert_eq!(op.second(), arg_1);
            assert_eq!(op.third(), arg_2);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.fshl");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_fshl_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_fshl_test(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
                    %0 = llvm.intr.fshl(%arg0, %arg1, %arg2) : (i32, i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_fshr() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[
                (i32_type.as_ref(), location),
                (i32_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_fshr(arg_0, arg_1, arg_2, i32_type, location);
            assert_eq!(op.first(), arg_0);
            assert_eq!(op.second(), arg_1);
            assert_eq!(op.third(), arg_2);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.fshr");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_fshr_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_fshr_test(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
                    %0 = llvm.intr.fshr(%arg0, %arg1, %arg2) : (i32, i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_get_active_lane_mask() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_get_active_lane_mask(arg_0, arg_1, mask_type, location);
            assert_eq!(op.base(), arg_0);
            assert_eq!(op.bound(), arg_1);
            assert_eq!(op.output_type(), mask_type);
            assert_eq!(op.operation_name(), "llvm.intr.get.active.lane.mask");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_get_active_lane_mask_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
                    results: vec![mask_type.into()],
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
                  func.func @llvm_intr_get_active_lane_mask_test(%arg0: i32, %arg1: i32) -> vector<4xi1> {
                    %0 = llvm.intr.get.active.lane.mask %arg0, %arg1 : i32, i32 to vector<4xi1>
                    return %0 : vector<4xi1>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_invariant_end() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0);
        let size = context.integer_attribute(i64_type, 1).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location), (pointer_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_invariant_end(arg_0, arg_1, size, location);
            assert_eq!(op.start(), arg_0);
            assert_eq!(op.pointer(), arg_1);
            assert_eq!(op.size(), size);
            assert_eq!(op.operation_name(), "llvm.intr.invariant.end");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_invariant_end_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into(), pointer_type.into()],
                    results: vec![],
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
                  func.func @llvm_intr_invariant_end_test(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
                    llvm.intr.invariant.end %arg0, 1, %arg1 : !llvm.ptr
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_invariant_start() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0);
        let size = context.integer_attribute(i64_type, 1).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_invariant_start(arg_0, pointer_type, size, location);
            assert_eq!(op.pointer(), arg_0);
            assert_eq!(op.size(), size);
            assert_eq!(op.output_type(), pointer_type);
            assert_eq!(op.operation_name(), "llvm.intr.invariant.start");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_invariant_start_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into()],
                    results: vec![pointer_type.into()],
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
                  func.func @llvm_intr_invariant_start_test(%arg0: !llvm.ptr) -> !llvm.ptr {
                    %0 = llvm.intr.invariant.start 1, %arg0 : !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_is_constant() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_is_constant(arg_0, i1_type, location);
            assert_eq!(op.value(), arg_0);
            assert_eq!(op.output_type(), i1_type);
            assert_eq!(op.operation_name(), "llvm.intr.is.constant");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_is_constant_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into()],
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
                  func.func @llvm_intr_is_constant_test(%arg0: i32) -> i1 {
                    %0 = \"llvm.intr.is.constant\"(%arg0) : (i32) -> i1
                    return %0 : i1
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_is_fpclass() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();
        let bit = context.integer_attribute(i32_type, 1).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_is_fpclass(arg_0, i1_type, bit, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.bit(), bit);
            assert_eq!(op.output_type(), i1_type);
            assert_eq!(op.operation_name(), "llvm.intr.is.fpclass");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_is_fpclass_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into()],
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
                  func.func @llvm_intr_is_fpclass_test(%arg0: f32) -> i1 {
                    %0 = \"llvm.intr.is.fpclass\"(%arg0) <{bit = 1 : i32}> : (f32) -> i1
                    return %0 : i1
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_launder_invariant_group() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_launder_invariant_group(arg_0, pointer_type, location);
            assert_eq!(op.pointer(), arg_0);
            assert_eq!(op.output_type(), pointer_type);
            assert_eq!(op.operation_name(), "llvm.intr.launder.invariant.group");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_launder_invariant_group_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into()],
                    results: vec![pointer_type.into()],
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
                  func.func @llvm_intr_launder_invariant_group_test(%arg0: !llvm.ptr) -> !llvm.ptr {
                    %0 = llvm.intr.launder.invariant.group %arg0 : !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_lifetime_end() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_lifetime_end(arg_0, location);
            assert_eq!(op.pointer(), arg_0);
            assert_eq!(op.operation_name(), "llvm.intr.lifetime.end");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_lifetime_end_test",
                func::FuncAttributes { arguments: vec![pointer_type.into()], results: vec![], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_intr_lifetime_end_test(%arg0: !llvm.ptr) {
                    llvm.intr.lifetime.end %arg0 : !llvm.ptr
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_lifetime_start() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_lifetime_start(arg_0, location);
            assert_eq!(op.pointer(), arg_0);
            assert_eq!(op.operation_name(), "llvm.intr.lifetime.start");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_lifetime_start_test",
                func::FuncAttributes { arguments: vec![pointer_type.into()], results: vec![], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_intr_lifetime_start_test(%arg0: !llvm.ptr) {
                    llvm.intr.lifetime.start %arg0 : !llvm.ptr
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_llrint() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_llrint(arg_0, i64_type, location);
            assert_eq!(op.value(), arg_0);
            assert_eq!(op.output_type(), i64_type);
            assert_eq!(op.operation_name(), "llvm.intr.llrint");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_llrint_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into()],
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
                  func.func @llvm_intr_llrint_test(%arg0: f32) -> i64 {
                    %0 = llvm.intr.llrint(%arg0) : (f32) -> i64
                    return %0 : i64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_llround() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_llround(arg_0, i64_type, location);
            assert_eq!(op.value(), arg_0);
            assert_eq!(op.output_type(), i64_type);
            assert_eq!(op.operation_name(), "llvm.intr.llround");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_llround_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into()],
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
                  func.func @llvm_intr_llround_test(%arg0: f32) -> i64 {
                    %0 = llvm.intr.llround(%arg0) : (f32) -> i64
                    return %0 : i64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_ldexp() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_ldexp(arg_0, arg_1, f32_type, None, location);
            assert_eq!(op.value(), arg_0);
            assert_eq!(op.power(), arg_1);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.ldexp");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_ldexp_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_ldexp_test(%arg0: f32, %arg1: i32) -> f32 {
                    %0 = llvm.intr.ldexp(%arg0, %arg1) : (f32, i32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_log10() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_log10(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.log10");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_log10_test",
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
                  func.func @llvm_intr_log10_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.log10(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_log2() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_log2(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.log2");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_log2_test",
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
                  func.func @llvm_intr_log2_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.log2(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_log() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_log(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.log");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_log_test",
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
                  func.func @llvm_intr_log_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.log(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_lrint() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_lrint(arg_0, i64_type, location);
            assert_eq!(op.value(), arg_0);
            assert_eq!(op.output_type(), i64_type);
            assert_eq!(op.operation_name(), "llvm.intr.lrint");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_lrint_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into()],
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
                  func.func @llvm_intr_lrint_test(%arg0: f32) -> i64 {
                    %0 = llvm.intr.lrint(%arg0) : (f32) -> i64
                    return %0 : i64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_lround() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_lround(arg_0, i64_type, location);
            assert_eq!(op.value(), arg_0);
            assert_eq!(op.output_type(), i64_type);
            assert_eq!(op.operation_name(), "llvm.intr.lround");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_lround_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into()],
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
                  func.func @llvm_intr_lround_test(%arg0: f32) -> i64 {
                    %0 = llvm.intr.lround(%arg0) : (f32) -> i64
                    return %0 : i64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_masked_load() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let alignment = context.integer_attribute(i32_type, 1).as_ref();
        let nontemporal = context.unit_attribute().as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location), (mask_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_masked_load(arg_0, arg_1, vector_i32_type, alignment, nontemporal, location);
            assert_eq!(op.data(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.alignment(), alignment);
            assert_eq!(op.nontemporal(), nontemporal);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.masked.load");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_masked_load_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into(), mask_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_masked_load_test(%arg0: !llvm.ptr, %arg1: vector<4xi1>) -> vector<4xi32> {
                    %0 = llvm.intr.masked.load %arg0, %arg1 {alignment = 1 : i32, nontemporal} : (!llvm.ptr, vector<4xi1>) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_masked_store() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let alignment = context.integer_attribute(i32_type, 1).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (pointer_type.as_ref(), location),
                (mask_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_masked_store(arg_0, arg_1, arg_2, alignment, location);
            assert_eq!(op.value(), arg_0);
            assert_eq!(op.data(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.alignment(), alignment);
            assert_eq!(op.operation_name(), "llvm.intr.masked.store");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_masked_store_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), pointer_type.into(), mask_type.into()],
                    results: vec![],
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
                  func.func @llvm_intr_masked_store_test(%arg0: vector<4xi32>, %arg1: !llvm.ptr, %arg2: vector<4xi1>) {
                    llvm.intr.masked.store %arg0, %arg1, %arg2 {alignment = 1 : i32} : vector<4xi32>, vector<4xi1> into !llvm.ptr
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_matrix_column_major_load() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let is_volatile = context.boolean_attribute(false).as_ref();
        let rows = context.integer_attribute(i32_type, 1).as_ref();
        let columns = context.integer_attribute(i32_type, 1).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location), (i64_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_matrix_column_major_load(arg_0, arg_1, vector_i32_type, is_volatile, rows, columns, location);
            assert_eq!(op.data(), arg_0);
            assert_eq!(op.stride(), arg_1);
            assert_eq!(op.is_volatile(), is_volatile);
            assert_eq!(op.rows(), rows);
            assert_eq!(op.columns(), columns);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.matrix.column.major.load");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_matrix_column_major_load_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into(), i64_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_matrix_column_major_load_test(%arg0: !llvm.ptr, %arg1: i64) -> vector<4xi32> {
                    %0 = llvm.intr.matrix.column.major.load %arg0, <stride = %arg1> {columns = 1 : i32, isVolatile = false, rows = 1 : i32} : vector<4xi32> from !llvm.ptr stride i64
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_matrix_column_major_store() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let is_volatile = context.boolean_attribute(false).as_ref();
        let rows = context.integer_attribute(i32_type, 1).as_ref();
        let columns = context.integer_attribute(i32_type, 1).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (pointer_type.as_ref(), location),
                (i64_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_matrix_column_major_store(arg_0, arg_1, arg_2, is_volatile, rows, columns, location);
            assert_eq!(op.matrix(), arg_0);
            assert_eq!(op.data(), arg_1);
            assert_eq!(op.stride(), arg_2);
            assert_eq!(op.is_volatile(), is_volatile);
            assert_eq!(op.rows(), rows);
            assert_eq!(op.columns(), columns);
            assert_eq!(op.operation_name(), "llvm.intr.matrix.column.major.store");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_matrix_column_major_store_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), pointer_type.into(), i64_type.into()],
                    results: vec![],
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
                  func.func @llvm_intr_matrix_column_major_store_test(%arg0: vector<4xi32>, %arg1: !llvm.ptr, %arg2: i64) {
                    llvm.intr.matrix.column.major.store %arg0, %arg1, <stride = %arg2> {columns = 1 : i32, isVolatile = false, rows = 1 : i32} : vector<4xi32> to !llvm.ptr stride i64
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_matrix_multiply() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        let lhs_rows = context.integer_attribute(i32_type, 1).as_ref();
        let lhs_columns = context.integer_attribute(i32_type, 1).as_ref();
        let rhs_columns = context.integer_attribute(i32_type, 1).as_ref();
        module.body().append_operation({
            let mut block =
                context.block(&[(vector_f32_type.as_ref(), location), (vector_f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_matrix_multiply(arg_0, arg_1, vector_f32_type, lhs_rows, lhs_columns, rhs_columns, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.lhs_rows(), lhs_rows);
            assert_eq!(op.lhs_columns(), lhs_columns);
            assert_eq!(op.rhs_columns(), rhs_columns);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.matrix.multiply");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_matrix_multiply_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into(), vector_f32_type.into()],
                    results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_matrix_multiply_test(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> vector<4xf32> {
                    %0 = llvm.intr.matrix.multiply %arg0, %arg1 {lhs_columns = 1 : i32, lhs_rows = 1 : i32, rhs_columns = 1 : i32} : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_matrix_transpose() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        let rows = context.integer_attribute(i32_type, 1).as_ref();
        let columns = context.integer_attribute(i32_type, 1).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(vector_f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_matrix_transpose(arg_0, vector_f32_type, rows, columns, location);
            assert_eq!(op.matrix(), arg_0);
            assert_eq!(op.rows(), rows);
            assert_eq!(op.columns(), columns);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.matrix.transpose");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_matrix_transpose_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into()],
                    results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_matrix_transpose_test(%arg0: vector<4xf32>) -> vector<4xf32> {
                    %0 = llvm.intr.matrix.transpose %arg0 {columns = 1 : i32, rows = 1 : i32} : vector<4xf32> into vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_maxnum() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location), (f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_maxnum(arg_0, arg_1, f32_type, None, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.maxnum");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_maxnum_test",
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
                  func.func @llvm_intr_maxnum_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = llvm.intr.maxnum(%arg0, %arg1) : (f32, f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_maximum() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location), (f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_maximum(arg_0, arg_1, f32_type, None, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.maximum");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_maximum_test",
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
                  func.func @llvm_intr_maximum_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = llvm.intr.maximum(%arg0, %arg1) : (f32, f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_memcpy_inline() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0);
        let len = context.integer_attribute(i64_type, 1).as_ref();
        let is_volatile = context.boolean_attribute(false).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location), (pointer_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_memcpy_inline(arg_0, arg_1, len, is_volatile, location);
            assert_eq!(op.destination(), arg_0);
            assert_eq!(op.source(), arg_1);
            assert_eq!(op.len(), len);
            assert_eq!(op.is_volatile(), is_volatile);
            assert_eq!(op.operation_name(), "llvm.intr.memcpy.inline");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_memcpy_inline_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into(), pointer_type.into()],
                    results: vec![],
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
                  func.func @llvm_intr_memcpy_inline_test(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
                    \"llvm.intr.memcpy.inline\"(%arg0, %arg1) <{isVolatile = false, len = 1 : i64}> : (!llvm.ptr, !llvm.ptr) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_memcpy() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0);
        let is_volatile = context.boolean_attribute(false).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[
                (pointer_type.as_ref(), location),
                (pointer_type.as_ref(), location),
                (i64_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_memcpy(arg_0, arg_1, arg_2, is_volatile, location);
            assert_eq!(op.destination(), arg_0);
            assert_eq!(op.source(), arg_1);
            assert_eq!(op.length(), arg_2);
            assert_eq!(op.is_volatile(), is_volatile);
            assert_eq!(op.operation_name(), "llvm.intr.memcpy");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_memcpy_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into(), pointer_type.into(), i64_type.into()],
                    results: vec![],
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
                  func.func @llvm_intr_memcpy_test(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64) {
                    \"llvm.intr.memcpy\"(%arg0, %arg1, %arg2) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_memmove() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0);
        let is_volatile = context.boolean_attribute(false).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[
                (pointer_type.as_ref(), location),
                (pointer_type.as_ref(), location),
                (i64_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_memmove(arg_0, arg_1, arg_2, is_volatile, location);
            assert_eq!(op.destination(), arg_0);
            assert_eq!(op.source(), arg_1);
            assert_eq!(op.length(), arg_2);
            assert_eq!(op.is_volatile(), is_volatile);
            assert_eq!(op.operation_name(), "llvm.intr.memmove");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_memmove_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into(), pointer_type.into(), i64_type.into()],
                    results: vec![],
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
                  func.func @llvm_intr_memmove_test(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64) {
                    \"llvm.intr.memmove\"(%arg0, %arg1, %arg2) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_memset_inline() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i8_type = context.signless_integer_type(8);
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0);
        let len = context.integer_attribute(i64_type, 1).as_ref();
        let is_volatile = context.boolean_attribute(false).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location), (i8_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_memset_inline(arg_0, arg_1, len, is_volatile, location);
            assert_eq!(op.destination(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.len(), len);
            assert_eq!(op.is_volatile(), is_volatile);
            assert_eq!(op.operation_name(), "llvm.intr.memset.inline");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_memset_inline_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into(), i8_type.into()],
                    results: vec![],
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
                  func.func @llvm_intr_memset_inline_test(%arg0: !llvm.ptr, %arg1: i8) {
                    \"llvm.intr.memset.inline\"(%arg0, %arg1) <{isVolatile = false, len = 1 : i64}> : (!llvm.ptr, i8) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_memset() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i8_type = context.signless_integer_type(8);
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0);
        let is_volatile = context.boolean_attribute(false).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[
                (pointer_type.as_ref(), location),
                (i8_type.as_ref(), location),
                (i64_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_memset(arg_0, arg_1, arg_2, is_volatile, location);
            assert_eq!(op.destination(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.length(), arg_2);
            assert_eq!(op.is_volatile(), is_volatile);
            assert_eq!(op.operation_name(), "llvm.intr.memset");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_memset_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into(), i8_type.into(), i64_type.into()],
                    results: vec![],
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
                  func.func @llvm_intr_memset_test(%arg0: !llvm.ptr, %arg1: i8, %arg2: i64) {
                    \"llvm.intr.memset\"(%arg0, %arg1, %arg2) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_min_num() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location), (f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_min_num(arg_0, arg_1, f32_type, None, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.minnum");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_minnum_test",
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
                  func.func @llvm_intr_minnum_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = llvm.intr.minnum(%arg0, %arg1) : (f32, f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_minimum() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location), (f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_minimum(arg_0, arg_1, f32_type, None, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.minimum");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_minimum_test",
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
                  func.func @llvm_intr_minimum_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = llvm.intr.minimum(%arg0, %arg1) : (f32, f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_nearby_int() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_nearby_int(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.nearbyint");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_nearbyint_test",
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
                  func.func @llvm_intr_nearbyint_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.nearbyint(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_experimental_noalias_scope_decl() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let scope = context.parse_attribute(r#"#llvm.alias_scope<id = "scope", domain = <id = "domain">>"#).unwrap();
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = intr_experimental_noalias_scope_decl(scope, location);
            assert_eq!(op.scope(), scope);
            assert_eq!(op.operation_name(), "llvm.intr.experimental.noalias.scope.decl");
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_experimental_noalias_scope_decl_test",
                func::FuncAttributes { arguments: vec![], results: vec![], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                #alias_scope_domain = #llvm.alias_scope_domain<id = \"domain\">
                #alias_scope = #llvm.alias_scope<id = \"scope\", domain = #alias_scope_domain>
                module {
                  func.func @llvm_intr_experimental_noalias_scope_decl_test() {
                    llvm.intr.experimental.noalias.scope.decl #alias_scope
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_powi() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_powi(arg_0, arg_1, f32_type, None, location);
            assert_eq!(op.value(), arg_0);
            assert_eq!(op.power(), arg_1);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.powi");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_powi_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_powi_test(%arg0: f32, %arg1: i32) -> f32 {
                    %0 = llvm.intr.powi(%arg0, %arg1) : (f32, i32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_pow() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location), (f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_pow(arg_0, arg_1, f32_type, None, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.pow");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_pow_test",
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
                  func.func @llvm_intr_pow_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = llvm.intr.pow(%arg0, %arg1) : (f32, f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_prefetch() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0);
        let rw = context.integer_attribute(i32_type, 1).as_ref();
        let hint = context.integer_attribute(i32_type, 1).as_ref();
        let cache = context.integer_attribute(i32_type, 1).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_prefetch(arg_0, rw, hint, cache, location);
            assert_eq!(op.address(), arg_0);
            assert_eq!(op.rw(), rw);
            assert_eq!(op.hint(), hint);
            assert_eq!(op.cache(), cache);
            assert_eq!(op.operation_name(), "llvm.intr.prefetch");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_prefetch_test",
                func::FuncAttributes { arguments: vec![pointer_type.into()], results: vec![], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_intr_prefetch_test(%arg0: !llvm.ptr) {
                    \"llvm.intr.prefetch\"(%arg0) <{cache = 1 : i32, hint = 1 : i32, rw = 1 : i32}> : (!llvm.ptr) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_ptr_annotation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0);
        module.body().append_operation({
            let mut block = context.block(&[
                (pointer_type.as_ref(), location),
                (pointer_type.as_ref(), location),
                (pointer_type.as_ref(), location),
                (i32_type.as_ref(), location),
                (pointer_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let arg_4 = block.argument(4).unwrap();
            let op = intr_ptr_annotation(arg_0, arg_1, arg_2, arg_3, arg_4, pointer_type, location);
            assert_eq!(op.pointer(), arg_0);
            assert_eq!(op.annotation(), arg_1);
            assert_eq!(op.file_name(), arg_2);
            assert_eq!(op.line(), arg_3);
            assert_eq!(PtrAnnotationOperation::attribute(&op), arg_4);
            assert_eq!(op.output_type(), pointer_type);
            assert_eq!(op.operation_name(), "llvm.intr.ptr.annotation");
            assert_eq!(op.operands().count(), 5);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_ptr_annotation_test",
                func::FuncAttributes {
                    arguments: vec![
                        pointer_type.into(),
                        pointer_type.into(),
                        pointer_type.into(),
                        i32_type.into(),
                        pointer_type.into(),
                    ],
                    results: vec![pointer_type.into()],
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
                  func.func @llvm_intr_ptr_annotation_test(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: !llvm.ptr) -> !llvm.ptr {
                    %0 = \"llvm.intr.ptr.annotation\"(%arg0, %arg1, %arg2, %arg3, %arg4) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_ptrmask() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0);
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location), (i64_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_ptrmask(arg_0, arg_1, pointer_type, location);
            assert_eq!(op.pointer(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.output_type(), pointer_type);
            assert_eq!(op.operation_name(), "llvm.intr.ptrmask");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_ptrmask_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into(), i64_type.into()],
                    results: vec![pointer_type.into()],
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
                  func.func @llvm_intr_ptrmask_test(%arg0: !llvm.ptr, %arg1: i64) -> !llvm.ptr {
                    %0 = llvm.intr.ptrmask %arg0, %arg1 : (!llvm.ptr, i64) -> !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_rint() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_rint(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.rint");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_rint_test",
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
                  func.func @llvm_intr_rint_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.rint(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_round_even() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_round_even(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.roundeven");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_roundeven_test",
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
                  func.func @llvm_intr_roundeven_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.roundeven(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_round() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_round(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.round");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_round_test",
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
                  func.func @llvm_intr_round_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.round(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_sadd_sat() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_sadd_sat(arg_0, arg_1, i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.sadd.sat");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_sadd_sat_test",
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
                  func.func @llvm_intr_sadd_sat_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.sadd.sat(%arg0, %arg1) : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_sadd_with_overflow() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let result_type = context.llvm_literal_struct_type(&[i32_type.as_ref(), i1_type.as_ref()], false);
            let op = intr_sadd_with_overflow(arg_0, arg_1, result_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.output_type(), result_type);
            assert_eq!(op.operation_name(), "llvm.intr.sadd.with.overflow");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_sadd_with_overflow_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
                    results: vec![result_type.into()],
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
                  func.func @llvm_intr_sadd_with_overflow_test(%arg0: i32, %arg1: i32) -> !llvm.struct<(i32, i1)> {
                    %0 = \"llvm.intr.sadd.with.overflow\"(%arg0, %arg1) : (i32, i32) -> !llvm.struct<(i32, i1)>
                    return %0 : !llvm.struct<(i32, i1)>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_scmp() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_scmp(arg_0, arg_1, i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.scmp");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_scmp_test",
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
                  func.func @llvm_intr_scmp_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.scmp(%arg0, %arg1) : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_smax() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_smax(arg_0, arg_1, i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.smax");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_smax_test",
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
                  func.func @llvm_intr_smax_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.smax(%arg0, %arg1) : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_smin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_smin(arg_0, arg_1, i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.smin");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_smin_test",
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
                  func.func @llvm_intr_smin_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.smin(%arg0, %arg1) : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_smul_with_overflow() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let result_type = context.llvm_literal_struct_type(&[i32_type.as_ref(), i1_type.as_ref()], false);
            let op = intr_smul_with_overflow(arg_0, arg_1, result_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.output_type(), result_type);
            assert_eq!(op.operation_name(), "llvm.intr.smul.with.overflow");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_smul_with_overflow_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
                    results: vec![result_type.into()],
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
                  func.func @llvm_intr_smul_with_overflow_test(%arg0: i32, %arg1: i32) -> !llvm.struct<(i32, i1)> {
                    %0 = \"llvm.intr.smul.with.overflow\"(%arg0, %arg1) : (i32, i32) -> !llvm.struct<(i32, i1)>
                    return %0 : !llvm.struct<(i32, i1)>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_ssa_copy() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_ssa_copy(arg_0, i32_type, location);
            assert_eq!(SsaCopyOperation::operand(&op), arg_0);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.ssa.copy");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_ssa_copy_test",
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
                  func.func @llvm_intr_ssa_copy_test(%arg0: i32) -> i32 {
                    %0 = llvm.intr.ssa.copy %arg0 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_sshl_sat() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_sshl_sat(arg_0, arg_1, i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.sshl.sat");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_sshl_sat_test",
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
                  func.func @llvm_intr_sshl_sat_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.sshl.sat(%arg0, %arg1) : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_ssub_sat() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_ssub_sat(arg_0, arg_1, i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.ssub.sat");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_ssub_sat_test",
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
                  func.func @llvm_intr_ssub_sat_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.ssub.sat(%arg0, %arg1) : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_ssub_with_overflow() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let result_type = context.llvm_literal_struct_type(&[i32_type.as_ref(), i1_type.as_ref()], false);
            let op = intr_ssub_with_overflow(arg_0, arg_1, result_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.output_type(), result_type);
            assert_eq!(op.operation_name(), "llvm.intr.ssub.with.overflow");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_ssub_with_overflow_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
                    results: vec![result_type.into()],
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
                  func.func @llvm_intr_ssub_with_overflow_test(%arg0: i32, %arg1: i32) -> !llvm.struct<(i32, i1)> {
                    %0 = \"llvm.intr.ssub.with.overflow\"(%arg0, %arg1) : (i32, i32) -> !llvm.struct<(i32, i1)>
                    return %0 : !llvm.struct<(i32, i1)>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_sin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_sin(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.sin");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_sin_test",
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
                  func.func @llvm_intr_sin_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.sin(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_sincos() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let result_type = context.llvm_literal_struct_type(&[f32_type.as_ref(), f32_type.as_ref()], false);
            let op = intr_sincos(arg_0, result_type, None, location);
            assert_eq!(op.value(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), result_type);
            assert_eq!(op.operation_name(), "llvm.intr.sincos");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_sincos_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into()],
                    results: vec![result_type.into()],
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
                  func.func @llvm_intr_sincos_test(%arg0: f32) -> !llvm.struct<(f32, f32)> {
                    %0 = llvm.intr.sincos(%arg0) : (f32) -> !llvm.struct<(f32, f32)>
                    return %0 : !llvm.struct<(f32, f32)>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_sinh() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_sinh(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.sinh");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_sinh_test",
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
                  func.func @llvm_intr_sinh_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.sinh(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_sqrt() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_sqrt(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.sqrt");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_sqrt_test",
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
                  func.func @llvm_intr_sqrt_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.sqrt(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_stackrestore() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_stackrestore(arg_0, location);
            assert_eq!(op.pointer(), arg_0);
            assert_eq!(op.operation_name(), "llvm.intr.stackrestore");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_stackrestore_test",
                func::FuncAttributes { arguments: vec![pointer_type.into()], results: vec![], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_intr_stackrestore_test(%arg0: !llvm.ptr) {
                    llvm.intr.stackrestore %arg0 : !llvm.ptr
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_stacksave() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = intr_stacksave(pointer_type.as_ref(), location);
            assert_eq!(op.output_type(), pointer_type);
            assert_eq!(op.operation_name(), "llvm.intr.stacksave");
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_stacksave_test",
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
                  func.func @llvm_intr_stacksave_test() -> !llvm.ptr {
                    %0 = llvm.intr.stacksave : !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_stepvector() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = intr_stepvector(vector_i32_type.as_ref(), location);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.stepvector");
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_stepvector_test",
                func::FuncAttributes { arguments: vec![], results: vec![vector_i32_type.into()], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_intr_stepvector_test() -> vector<4xi32> {
                    %0 = llvm.intr.stepvector : vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_strip_invariant_group() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_strip_invariant_group(arg_0, pointer_type, location);
            assert_eq!(op.pointer(), arg_0);
            assert_eq!(op.output_type(), pointer_type);
            assert_eq!(op.operation_name(), "llvm.intr.strip.invariant.group");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_strip_invariant_group_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into()],
                    results: vec![pointer_type.into()],
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
                  func.func @llvm_intr_strip_invariant_group_test(%arg0: !llvm.ptr) -> !llvm.ptr {
                    %0 = llvm.intr.strip.invariant.group %arg0 : !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_tan() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_tan(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.tan");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_tan_test",
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
                  func.func @llvm_intr_tan_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.tan(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_tanh() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_tanh(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.tanh");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_tanh_test",
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
                  func.func @llvm_intr_tanh_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.tanh(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_threadlocal_address() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_threadlocal_address(arg_0, pointer_type, location);
            assert_eq!(op.global(), arg_0);
            assert_eq!(op.output_type(), pointer_type);
            assert_eq!(op.operation_name(), "llvm.intr.threadlocal.address");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_threadlocal_address_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into()],
                    results: vec![pointer_type.into()],
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
                  func.func @llvm_intr_threadlocal_address_test(%arg0: !llvm.ptr) -> !llvm.ptr {
                    %0 = \"llvm.intr.threadlocal.address\"(%arg0) : (!llvm.ptr) -> !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_trap() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = intr_trap(location);
            assert_eq!(op.operation_name(), "llvm.intr.trap");
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_trap_test",
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
                  func.func @llvm_intr_trap_test() {
                    llvm.intr.trap
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_uadd_sat() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_uadd_sat(arg_0, arg_1, i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.uadd.sat");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_uadd_sat_test",
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
                  func.func @llvm_intr_uadd_sat_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.uadd.sat(%arg0, %arg1) : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_uadd_with_overflow() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let result_type = context.llvm_literal_struct_type(&[i32_type.as_ref(), i1_type.as_ref()], false);
            let op = intr_uadd_with_overflow(arg_0, arg_1, result_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.output_type(), result_type);
            assert_eq!(op.operation_name(), "llvm.intr.uadd.with.overflow");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_uadd_with_overflow_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
                    results: vec![result_type.into()],
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
                  func.func @llvm_intr_uadd_with_overflow_test(%arg0: i32, %arg1: i32) -> !llvm.struct<(i32, i1)> {
                    %0 = \"llvm.intr.uadd.with.overflow\"(%arg0, %arg1) : (i32, i32) -> !llvm.struct<(i32, i1)>
                    return %0 : !llvm.struct<(i32, i1)>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_ubsan_trap() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i8_type = context.signless_integer_type(8);
        let failure_kind = context.integer_attribute(i8_type, 1).as_ref();
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = intr_ubsan_trap(failure_kind, location);
            assert_eq!(op.failure_kind(), failure_kind);
            assert_eq!(op.operation_name(), "llvm.intr.ubsantrap");
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_ubsantrap_test",
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
                  func.func @llvm_intr_ubsantrap_test() {
                    llvm.intr.ubsantrap <{failureKind = 1 : i8}>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_ucmp() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_ucmp(arg_0, arg_1, i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.ucmp");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_ucmp_test",
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
                  func.func @llvm_intr_ucmp_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.ucmp(%arg0, %arg1) : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_umax() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_umax(arg_0, arg_1, i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.umax");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_umax_test",
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
                  func.func @llvm_intr_umax_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.umax(%arg0, %arg1) : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_umin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_umin(arg_0, arg_1, i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.umin");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_umin_test",
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
                  func.func @llvm_intr_umin_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.umin(%arg0, %arg1) : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_umul_with_overflow() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let result_type = context.llvm_literal_struct_type(&[i32_type.as_ref(), i1_type.as_ref()], false);
            let op = intr_umul_with_overflow(arg_0, arg_1, result_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.output_type(), result_type);
            assert_eq!(op.operation_name(), "llvm.intr.umul.with.overflow");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_umul_with_overflow_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
                    results: vec![result_type.into()],
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
                  func.func @llvm_intr_umul_with_overflow_test(%arg0: i32, %arg1: i32) -> !llvm.struct<(i32, i1)> {
                    %0 = \"llvm.intr.umul.with.overflow\"(%arg0, %arg1) : (i32, i32) -> !llvm.struct<(i32, i1)>
                    return %0 : !llvm.struct<(i32, i1)>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_ushl_sat() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_ushl_sat(arg_0, arg_1, i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.ushl.sat");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_ushl_sat_test",
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
                  func.func @llvm_intr_ushl_sat_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.ushl.sat(%arg0, %arg1) : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_usub_sat() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_usub_sat(arg_0, arg_1, i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.usub.sat");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_usub_sat_test",
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
                  func.func @llvm_intr_usub_sat_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.usub.sat(%arg0, %arg1) : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_usub_with_overflow() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let result_type = context.llvm_literal_struct_type(&[i32_type.as_ref(), i1_type.as_ref()], false);
            let op = intr_usub_with_overflow(arg_0, arg_1, result_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.output_type(), result_type);
            assert_eq!(op.operation_name(), "llvm.intr.usub.with.overflow");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_usub_with_overflow_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), i32_type.into()],
                    results: vec![result_type.into()],
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
                  func.func @llvm_intr_usub_with_overflow_test(%arg0: i32, %arg1: i32) -> !llvm.struct<(i32, i1)> {
                    %0 = \"llvm.intr.usub.with.overflow\"(%arg0, %arg1) : (i32, i32) -> !llvm.struct<(i32, i1)>
                    return %0 : !llvm.struct<(i32, i1)>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_ashr() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_ashr(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.ashr");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_ashr_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_ashr_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.ashr\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_add() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_add(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.add");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_add_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_add_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.add\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_and() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_and(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.and");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_and_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_and_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.and\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_fadd() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_f32_type.as_ref(), location),
                (vector_f32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_fadd(arg_0, arg_1, arg_2, arg_3, vector_f32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.fadd");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_fadd_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into(), vector_f32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vp_fadd_test(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xf32> {
                    %0 = \"llvm.intr.vp.fadd\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xf32>, vector<4xf32>, vector<4xi1>, i32) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_fdiv() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_f32_type.as_ref(), location),
                (vector_f32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_fdiv(arg_0, arg_1, arg_2, arg_3, vector_f32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.fdiv");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_fdiv_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into(), vector_f32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vp_fdiv_test(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xf32> {
                    %0 = \"llvm.intr.vp.fdiv\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xf32>, vector<4xf32>, vector<4xi1>, i32) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_fmuladd() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_f32_type.as_ref(), location),
                (vector_f32_type.as_ref(), location),
                (vector_f32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let arg_4 = block.argument(4).unwrap();
            let op = intr_vp_fmuladd(arg_0, arg_1, arg_2, arg_3, arg_4, vector_f32_type, location);
            assert_eq!(op.first(), arg_0);
            assert_eq!(op.second(), arg_1);
            assert_eq!(op.third(), arg_2);
            assert_eq!(op.mask(), arg_3);
            assert_eq!(op.explicit_vector_length(), arg_4);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.fmuladd");
            assert_eq!(op.operands().count(), 5);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_fmuladd_test",
                func::FuncAttributes {
                    arguments: vec![
                        vector_f32_type.into(),
                        vector_f32_type.into(),
                        vector_f32_type.into(),
                        mask_type.into(),
                        i32_type.into(),
                    ],
                    results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vp_fmuladd_test(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: vector<4xf32>, %arg3: vector<4xi1>, %arg4: i32) -> vector<4xf32> {
                    %0 = \"llvm.intr.vp.fmuladd\"(%arg0, %arg1, %arg2, %arg3, %arg4) : (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xi1>, i32) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_fmul() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_f32_type.as_ref(), location),
                (vector_f32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_fmul(arg_0, arg_1, arg_2, arg_3, vector_f32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.fmul");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_fmul_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into(), vector_f32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vp_fmul_test(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xf32> {
                    %0 = \"llvm.intr.vp.fmul\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xf32>, vector<4xf32>, vector<4xi1>, i32) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_fneg() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_f32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_fneg(arg_0, arg_1, arg_2, vector_f32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.fneg");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_fneg_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vp_fneg_test(%arg0: vector<4xf32>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xf32> {
                    %0 = \"llvm.intr.vp.fneg\"(%arg0, %arg1, %arg2) : (vector<4xf32>, vector<4xi1>, i32) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_fpext() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_f32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_fpext(arg_0, arg_1, arg_2, vector_f32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.fpext");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_fpext_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vp_fpext_test(%arg0: vector<4xf32>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xf32> {
                    %0 = \"llvm.intr.vp.fpext\"(%arg0, %arg1, %arg2) : (vector<4xf32>, vector<4xi1>, i32) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_fptosi() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_f32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_fptosi(arg_0, arg_1, arg_2, vector_i32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.fptosi");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_fptosi_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_fptosi_test(%arg0: vector<4xf32>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.fptosi\"(%arg0, %arg1, %arg2) : (vector<4xf32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_fptoui() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_f32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_fptoui(arg_0, arg_1, arg_2, vector_i32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.fptoui");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_fptoui_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_fptoui_test(%arg0: vector<4xf32>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.fptoui\"(%arg0, %arg1, %arg2) : (vector<4xf32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_fptrunc() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_f32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_fptrunc(arg_0, arg_1, arg_2, vector_f32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.fptrunc");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_fptrunc_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vp_fptrunc_test(%arg0: vector<4xf32>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xf32> {
                    %0 = \"llvm.intr.vp.fptrunc\"(%arg0, %arg1, %arg2) : (vector<4xf32>, vector<4xi1>, i32) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_frem() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_f32_type.as_ref(), location),
                (vector_f32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_frem(arg_0, arg_1, arg_2, arg_3, vector_f32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.frem");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_frem_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into(), vector_f32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vp_frem_test(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xf32> {
                    %0 = \"llvm.intr.vp.frem\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xf32>, vector<4xf32>, vector<4xi1>, i32) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_fsub() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_f32_type.as_ref(), location),
                (vector_f32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_fsub(arg_0, arg_1, arg_2, arg_3, vector_f32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.fsub");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_fsub_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into(), vector_f32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vp_fsub_test(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xf32> {
                    %0 = \"llvm.intr.vp.fsub\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xf32>, vector<4xf32>, vector<4xi1>, i32) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_fma() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_f32_type.as_ref(), location),
                (vector_f32_type.as_ref(), location),
                (vector_f32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let arg_4 = block.argument(4).unwrap();
            let op = intr_vp_fma(arg_0, arg_1, arg_2, arg_3, arg_4, vector_f32_type, location);
            assert_eq!(op.first(), arg_0);
            assert_eq!(op.second(), arg_1);
            assert_eq!(op.third(), arg_2);
            assert_eq!(op.mask(), arg_3);
            assert_eq!(op.explicit_vector_length(), arg_4);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.fma");
            assert_eq!(op.operands().count(), 5);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_fma_test",
                func::FuncAttributes {
                    arguments: vec![
                        vector_f32_type.into(),
                        vector_f32_type.into(),
                        vector_f32_type.into(),
                        mask_type.into(),
                        i32_type.into(),
                    ],
                    results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vp_fma_test(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: vector<4xf32>, %arg3: vector<4xi1>, %arg4: i32) -> vector<4xf32> {
                    %0 = \"llvm.intr.vp.fma\"(%arg0, %arg1, %arg2, %arg3, %arg4) : (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xi1>, i32) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_inttoptr() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i64_type = context.parse_type("vector<4xi64>").unwrap();
        let vector_pointer_type = context.parse_type("vector<4x!llvm.ptr>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i64_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_inttoptr(arg_0, arg_1, arg_2, vector_pointer_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_pointer_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.inttoptr");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_inttoptr_test",
                func::FuncAttributes {
                    arguments: vec![vector_i64_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_pointer_type.into()],
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
                  func.func @llvm_intr_vp_inttoptr_test(%arg0: vector<4xi64>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4x!llvm.ptr> {
                    %0 = \"llvm.intr.vp.inttoptr\"(%arg0, %arg1, %arg2) : (vector<4xi64>, vector<4xi1>, i32) -> vector<4x!llvm.ptr>
                    return %0 : vector<4x!llvm.ptr>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_lshr() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_lshr(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.lshr");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_lshr_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_lshr_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.lshr\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_load() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (pointer_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_load(arg_0, arg_1, arg_2, vector_i32_type, location);
            assert_eq!(op.pointer(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.load");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_load_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_load_test(%arg0: !llvm.ptr, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.load\"(%arg0, %arg1, %arg2) : (!llvm.ptr, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_merge() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (mask_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_merge(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.condition(), arg_0);
            assert_eq!(op.true_value(), arg_1);
            assert_eq!(op.false_value(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.merge");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_merge_test",
                func::FuncAttributes {
                    arguments: vec![mask_type.into(), vector_i32_type.into(), vector_i32_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_merge_test(%arg0: vector<4xi1>, %arg1: vector<4xi32>, %arg2: vector<4xi32>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.merge\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi1>, vector<4xi32>, vector<4xi32>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_mul() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_mul(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.mul");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_mul_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_mul_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.mul\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_or() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_or(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.or");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_or_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_or_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.or\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_ptrtoint() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i64_type = context.parse_type("vector<4xi64>").unwrap();
        let vector_pointer_type = context.parse_type("vector<4x!llvm.ptr>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_pointer_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_ptrtoint(arg_0, arg_1, arg_2, vector_i64_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_i64_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.ptrtoint");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_ptrtoint_test",
                func::FuncAttributes {
                    arguments: vec![vector_pointer_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i64_type.into()],
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
                  func.func @llvm_intr_vp_ptrtoint_test(%arg0: vector<4x!llvm.ptr>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xi64> {
                    %0 = \"llvm.intr.vp.ptrtoint\"(%arg0, %arg1, %arg2) : (vector<4x!llvm.ptr>, vector<4xi1>, i32) -> vector<4xi64>
                    return %0 : vector<4xi64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_add() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_reduce_add(arg_0, arg_1, arg_2, arg_3, i32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.add");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_add_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_add_test(%arg0: i32, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> i32 {
                    %0 = \"llvm.intr.vp.reduce.add\"(%arg0, %arg1, %arg2, %arg3) : (i32, vector<4xi32>, vector<4xi1>, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_and() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_reduce_and(arg_0, arg_1, arg_2, arg_3, i32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.and");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_and_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_and_test(%arg0: i32, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> i32 {
                    %0 = \"llvm.intr.vp.reduce.and\"(%arg0, %arg1, %arg2, %arg3) : (i32, vector<4xi32>, vector<4xi1>, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_fadd() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (f32_type.as_ref(), location),
                (vector_f32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_reduce_fadd(arg_0, arg_1, arg_2, arg_3, f32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.fadd");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_fadd_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), vector_f32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_fadd_test(%arg0: f32, %arg1: vector<4xf32>, %arg2: vector<4xi1>, %arg3: i32) -> f32 {
                    %0 = \"llvm.intr.vp.reduce.fadd\"(%arg0, %arg1, %arg2, %arg3) : (f32, vector<4xf32>, vector<4xi1>, i32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_fmax() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (f32_type.as_ref(), location),
                (vector_f32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_reduce_fmax(arg_0, arg_1, arg_2, arg_3, f32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.fmax");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_fmax_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), vector_f32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_fmax_test(%arg0: f32, %arg1: vector<4xf32>, %arg2: vector<4xi1>, %arg3: i32) -> f32 {
                    %0 = \"llvm.intr.vp.reduce.fmax\"(%arg0, %arg1, %arg2, %arg3) : (f32, vector<4xf32>, vector<4xi1>, i32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_fmin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (f32_type.as_ref(), location),
                (vector_f32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_reduce_fmin(arg_0, arg_1, arg_2, arg_3, f32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.fmin");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_fmin_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), vector_f32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_fmin_test(%arg0: f32, %arg1: vector<4xf32>, %arg2: vector<4xi1>, %arg3: i32) -> f32 {
                    %0 = \"llvm.intr.vp.reduce.fmin\"(%arg0, %arg1, %arg2, %arg3) : (f32, vector<4xf32>, vector<4xi1>, i32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_fmul() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (f32_type.as_ref(), location),
                (vector_f32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_reduce_fmul(arg_0, arg_1, arg_2, arg_3, f32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.fmul");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_fmul_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), vector_f32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_fmul_test(%arg0: f32, %arg1: vector<4xf32>, %arg2: vector<4xi1>, %arg3: i32) -> f32 {
                    %0 = \"llvm.intr.vp.reduce.fmul\"(%arg0, %arg1, %arg2, %arg3) : (f32, vector<4xf32>, vector<4xi1>, i32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_mul() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_reduce_mul(arg_0, arg_1, arg_2, arg_3, i32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.mul");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_mul_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_mul_test(%arg0: i32, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> i32 {
                    %0 = \"llvm.intr.vp.reduce.mul\"(%arg0, %arg1, %arg2, %arg3) : (i32, vector<4xi32>, vector<4xi1>, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_or() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_reduce_or(arg_0, arg_1, arg_2, arg_3, i32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.or");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_or_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_or_test(%arg0: i32, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> i32 {
                    %0 = \"llvm.intr.vp.reduce.or\"(%arg0, %arg1, %arg2, %arg3) : (i32, vector<4xi32>, vector<4xi1>, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_smax() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_reduce_smax(arg_0, arg_1, arg_2, arg_3, i32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.smax");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_smax_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_smax_test(%arg0: i32, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> i32 {
                    %0 = \"llvm.intr.vp.reduce.smax\"(%arg0, %arg1, %arg2, %arg3) : (i32, vector<4xi32>, vector<4xi1>, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_smin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_reduce_smin(arg_0, arg_1, arg_2, arg_3, i32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.smin");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_smin_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_smin_test(%arg0: i32, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> i32 {
                    %0 = \"llvm.intr.vp.reduce.smin\"(%arg0, %arg1, %arg2, %arg3) : (i32, vector<4xi32>, vector<4xi1>, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_umax() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_reduce_umax(arg_0, arg_1, arg_2, arg_3, i32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.umax");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_umax_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_umax_test(%arg0: i32, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> i32 {
                    %0 = \"llvm.intr.vp.reduce.umax\"(%arg0, %arg1, %arg2, %arg3) : (i32, vector<4xi32>, vector<4xi1>, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_umin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_reduce_umin(arg_0, arg_1, arg_2, arg_3, i32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.umin");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_umin_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_umin_test(%arg0: i32, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> i32 {
                    %0 = \"llvm.intr.vp.reduce.umin\"(%arg0, %arg1, %arg2, %arg3) : (i32, vector<4xi32>, vector<4xi1>, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_xor() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_reduce_xor(arg_0, arg_1, arg_2, arg_3, i32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.xor");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_xor_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_xor_test(%arg0: i32, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> i32 {
                    %0 = \"llvm.intr.vp.reduce.xor\"(%arg0, %arg1, %arg2, %arg3) : (i32, vector<4xi32>, vector<4xi1>, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_sdiv() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_sdiv(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.sdiv");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_sdiv_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_sdiv_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.sdiv\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_sext() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_sext(arg_0, arg_1, arg_2, vector_i32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.sext");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_sext_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_sext_test(%arg0: vector<4xi32>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.sext\"(%arg0, %arg1, %arg2) : (vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_sitofp() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_sitofp(arg_0, arg_1, arg_2, vector_f32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.sitofp");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_sitofp_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vp_sitofp_test(%arg0: vector<4xi32>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xf32> {
                    %0 = \"llvm.intr.vp.sitofp\"(%arg0, %arg1, %arg2) : (vector<4xi32>, vector<4xi1>, i32) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_smax() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_smax(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.smax");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_smax_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_smax_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.smax\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_smin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_smin(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.smin");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_smin_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_smin_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.smin\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_srem() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_srem(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.srem");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_srem_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_srem_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.srem\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_select() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (mask_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_select(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.condition(), arg_0);
            assert_eq!(op.true_value(), arg_1);
            assert_eq!(op.false_value(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.select");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_select_test",
                func::FuncAttributes {
                    arguments: vec![mask_type.into(), vector_i32_type.into(), vector_i32_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_select_test(%arg0: vector<4xi1>, %arg1: vector<4xi32>, %arg2: vector<4xi32>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.select\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi1>, vector<4xi32>, vector<4xi32>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_shl() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_shl(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.shl");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_shl_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_shl_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.shl\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_store() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (pointer_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_store(arg_0, arg_1, arg_2, arg_3, location);
            assert_eq!(op.value(), arg_0);
            assert_eq!(op.pointer(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.operation_name(), "llvm.intr.vp.store");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_vp_store_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), pointer_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![],
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
                  func.func @llvm_intr_vp_store_test(%arg0: vector<4xi32>, %arg1: !llvm.ptr, %arg2: vector<4xi1>, %arg3: i32) {
                    \"llvm.intr.vp.store\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, !llvm.ptr, vector<4xi1>, i32) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_experimental_vp_strided_load() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (pointer_type.as_ref(), location),
                (i64_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_experimental_vp_strided_load(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.pointer(), arg_0);
            assert_eq!(op.stride(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.experimental.vp.strided.load");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_experimental_vp_strided_load_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into(), i64_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_experimental_vp_strided_load_test(%arg0: !llvm.ptr, %arg1: i64, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.experimental.vp.strided.load\"(%arg0, %arg1, %arg2, %arg3) : (!llvm.ptr, i64, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_experimental_vp_strided_store() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (pointer_type.as_ref(), location),
                (i64_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let arg_4 = block.argument(4).unwrap();
            let op = intr_experimental_vp_strided_store(arg_0, arg_1, arg_2, arg_3, arg_4, location);
            assert_eq!(op.value(), arg_0);
            assert_eq!(op.pointer(), arg_1);
            assert_eq!(op.stride(), arg_2);
            assert_eq!(op.mask(), arg_3);
            assert_eq!(op.explicit_vector_length(), arg_4);
            assert_eq!(op.operation_name(), "llvm.intr.experimental.vp.strided.store");
            assert_eq!(op.operands().count(), 5);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_experimental_vp_strided_store_test",
                func::FuncAttributes {
                    arguments: vec![
                        vector_i32_type.into(),
                        pointer_type.into(),
                        i64_type.into(),
                        mask_type.into(),
                        i32_type.into(),
                    ],
                    results: vec![],
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
                  func.func @llvm_intr_experimental_vp_strided_store_test(%arg0: vector<4xi32>, %arg1: !llvm.ptr, %arg2: i64, %arg3: vector<4xi1>, %arg4: i32) {
                    \"llvm.intr.experimental.vp.strided.store\"(%arg0, %arg1, %arg2, %arg3, %arg4) : (vector<4xi32>, !llvm.ptr, i64, vector<4xi1>, i32) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_sub() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_sub(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.sub");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_sub_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_sub_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.sub\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_trunc() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_trunc(arg_0, arg_1, arg_2, vector_i32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.trunc");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_trunc_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_trunc_test(%arg0: vector<4xi32>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.trunc\"(%arg0, %arg1, %arg2) : (vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_udiv() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_udiv(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.udiv");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_udiv_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_udiv_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.udiv\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_uitofp() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_uitofp(arg_0, arg_1, arg_2, vector_f32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.uitofp");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_uitofp_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vp_uitofp_test(%arg0: vector<4xi32>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xf32> {
                    %0 = \"llvm.intr.vp.uitofp\"(%arg0, %arg1, %arg2) : (vector<4xi32>, vector<4xi1>, i32) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_umax() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_umax(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.umax");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_umax_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_umax_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.umax\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_umin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_umin(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.umin");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_umin_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_umin_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.umin\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_urem() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_urem(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.urem");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_urem_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_urem_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.urem\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_xor() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let op = intr_vp_xor(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.xor");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_xor_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_xor_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.xor\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_zext() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_zext(arg_0, arg_1, arg_2, vector_i32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.zext");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_zext_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_zext_test(%arg0: vector<4xi32>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.zext\"(%arg0, %arg1, %arg2) : (vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vacopy() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location), (pointer_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_vacopy(arg_0, arg_1, location);
            assert_eq!(op.destination_list(), arg_0);
            assert_eq!(op.source_list(), arg_1);
            assert_eq!(op.operation_name(), "llvm.intr.vacopy");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_vacopy_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into(), pointer_type.into()],
                    results: vec![],
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
                  func.func @llvm_intr_vacopy_test(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
                    llvm.intr.vacopy %arg1 to %arg0 : !llvm.ptr, !llvm.ptr
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vaend() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_vaend(arg_0, location);
            assert_eq!(op.argument_list(), arg_0);
            assert_eq!(op.operation_name(), "llvm.intr.vaend");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_vaend_test",
                func::FuncAttributes { arguments: vec![pointer_type.into()], results: vec![], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_intr_vaend_test(%arg0: !llvm.ptr) {
                    llvm.intr.vaend %arg0 : !llvm.ptr
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vastart() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        module.body().append_operation({
            let mut block = context.block(&[(pointer_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_vastart(arg_0, location);
            assert_eq!(op.argument_list(), arg_0);
            assert_eq!(op.operation_name(), "llvm.intr.vastart");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_vastart_test",
                func::FuncAttributes { arguments: vec![pointer_type.into()], results: vec![], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_intr_vastart_test(%arg0: !llvm.ptr) {
                    llvm.intr.vastart %arg0 : !llvm.ptr
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_var_annotation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0);
        module.body().append_operation({
            let mut block = context.block(&[
                (pointer_type.as_ref(), location),
                (pointer_type.as_ref(), location),
                (pointer_type.as_ref(), location),
                (i32_type.as_ref(), location),
                (pointer_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let arg_3 = block.argument(3).unwrap();
            let arg_4 = block.argument(4).unwrap();
            let op = intr_var_annotation(arg_0, arg_1, arg_2, arg_3, arg_4, location);
            assert_eq!(op.value(), arg_0);
            assert_eq!(op.annotation(), arg_1);
            assert_eq!(op.file_name(), arg_2);
            assert_eq!(op.line(), arg_3);
            assert_eq!(VarAnnotationOperation::attribute(&op), arg_4);
            assert_eq!(op.operation_name(), "llvm.intr.var.annotation");
            assert_eq!(op.operands().count(), 5);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_var_annotation_test",
                func::FuncAttributes {
                    arguments: vec![
                        pointer_type.into(),
                        pointer_type.into(),
                        pointer_type.into(),
                        i32_type.into(),
                        pointer_type.into(),
                    ],
                    results: vec![],
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
                  func.func @llvm_intr_var_annotation_test(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: !llvm.ptr) {
                    \"llvm.intr.var.annotation\"(%arg0, %arg1, %arg2, %arg3, %arg4) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_masked_compressstore() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (pointer_type.as_ref(), location),
                (mask_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_masked_compressstore(arg_0, arg_1, arg_2, location);
            assert_eq!(op.value(), arg_0);
            assert_eq!(op.pointer(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.operation_name(), "llvm.intr.masked.compressstore");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_masked_compressstore_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), pointer_type.into(), mask_type.into()],
                    results: vec![],
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
                  func.func @llvm_intr_masked_compressstore_test(%arg0: vector<4xi32>, %arg1: !llvm.ptr, %arg2: vector<4xi1>) {
                    \"llvm.intr.masked.compressstore\"(%arg0, %arg1, %arg2) : (vector<4xi32>, !llvm.ptr, vector<4xi1>) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_masked_expandload() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let pointer_type = context.llvm_pointer_type(0);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (pointer_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (vector_i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_masked_expandload(arg_0, arg_1, arg_2, vector_i32_type, location);
            assert_eq!(op.pointer(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.passthru(), arg_2);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.masked.expandload");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_masked_expandload_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into(), mask_type.into(), vector_i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_masked_expandload_test(%arg0: !llvm.ptr, %arg1: vector<4xi1>, %arg2: vector<4xi32>) -> vector<4xi32> {
                    %0 = \"llvm.intr.masked.expandload\"(%arg0, %arg1, %arg2) : (!llvm.ptr, vector<4xi1>, vector<4xi32>) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_masked_gather() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let vector_pointer_type = context.parse_type("vector<4x!llvm.ptr>").unwrap();
        let alignment = context.integer_attribute(i32_type, 1).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(vector_pointer_type.as_ref(), location), (mask_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_masked_gather(arg_0, arg_1, vector_i32_type, alignment, location);
            assert_eq!(op.pointers(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.alignment(), alignment);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.masked.gather");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_masked_gather_test",
                func::FuncAttributes {
                    arguments: vec![vector_pointer_type.into(), mask_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_masked_gather_test(%arg0: vector<4x!llvm.ptr>, %arg1: vector<4xi1>) -> vector<4xi32> {
                    %0 = llvm.intr.masked.gather %arg0, %arg1 {alignment = 1 : i32} : (vector<4x!llvm.ptr>, vector<4xi1>) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_masked_scatter() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let vector_pointer_type = context.parse_type("vector<4x!llvm.ptr>").unwrap();
        let alignment = context.integer_attribute(i32_type, 1).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (vector_pointer_type.as_ref(), location),
                (mask_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_masked_scatter(arg_0, arg_1, arg_2, alignment, location);
            assert_eq!(op.value(), arg_0);
            assert_eq!(op.pointers(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.alignment(), alignment);
            assert_eq!(op.operation_name(), "llvm.intr.masked.scatter");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_masked_scatter_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_pointer_type.into(), mask_type.into()],
                    results: vec![],
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
                  func.func @llvm_intr_masked_scatter_test(%arg0: vector<4xi32>, %arg1: vector<4x!llvm.ptr>, %arg2: vector<4xi1>) {
                    llvm.intr.masked.scatter %arg0, %arg1, %arg2 {alignment = 1 : i32} : vector<4xi32>, vector<4xi1> into vector<4x!llvm.ptr>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_deinterleave2() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let vector8_i32_type = context.parse_type("vector<8xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(vector8_i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_vector_deinterleave2(arg_0, vector_i32_type, location);
            assert_eq!(op.vector(), arg_0);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vector.deinterleave2");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vector_deinterleave2_test",
                func::FuncAttributes {
                    arguments: vec![vector8_i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vector_deinterleave2_test(%arg0: vector<8xi32>) -> vector<4xi32> {
                    %0 = \"llvm.intr.vector.deinterleave2\"(%arg0) : (vector<8xi32>) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_extract() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let pos = context.integer_attribute(i64_type, 1).as_ref();
        module.body().append_operation({
            let mut block = context.block(&[(vector_i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_vector_extract(arg_0, vector_i32_type, pos, location);
            assert_eq!(op.source_vector(), arg_0);
            assert_eq!(op.pos(), pos);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vector.extract");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vector_extract_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vector_extract_test(%arg0: vector<4xi32>) -> vector<4xi32> {
                    %0 = llvm.intr.vector.extract %arg0[1] : vector<4xi32> from vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_insert() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let pos = context.integer_attribute(i64_type, 1).as_ref();
        module.body().append_operation({
            let mut block =
                context.block(&[(vector_i32_type.as_ref(), location), (vector_i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_vector_insert(arg_0, arg_1, vector_i32_type, pos, location);
            assert_eq!(op.destination_vector(), arg_0);
            assert_eq!(op.source_vector(), arg_1);
            assert_eq!(op.pos(), pos);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vector.insert");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vector_insert_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vector_insert_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi32> {
                    %0 = llvm.intr.vector.insert %arg1, %arg0[1] : vector<4xi32> into vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_interleave2() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let vector8_i32_type = context.parse_type("vector<8xi32>").unwrap();
        module.body().append_operation({
            let mut block =
                context.block(&[(vector_i32_type.as_ref(), location), (vector_i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_vector_interleave2(arg_0, arg_1, vector8_i32_type, location);
            assert_eq!(op.first_vector(), arg_0);
            assert_eq!(op.second_vector(), arg_1);
            assert_eq!(op.output_type(), vector8_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vector.interleave2");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vector_interleave2_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into()],
                    results: vec![vector8_i32_type.into()],
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
                  func.func @llvm_intr_vector_interleave2_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<8xi32> {
                    %0 = \"llvm.intr.vector.interleave2\"(%arg0, %arg1) : (vector<4xi32>, vector<4xi32>) -> vector<8xi32>
                    return %0 : vector<8xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_add() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(vector_i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_vector_reduce_add(arg_0, i32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.add");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vector_reduce_add_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_add_test(%arg0: vector<4xi32>) -> i32 {
                    %0 = \"llvm.intr.vector.reduce.add\"(%arg0) : (vector<4xi32>) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_and() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(vector_i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_vector_reduce_and(arg_0, i32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.and");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vector_reduce_and_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_and_test(%arg0: vector<4xi32>) -> i32 {
                    %0 = \"llvm.intr.vector.reduce.and\"(%arg0) : (vector<4xi32>) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_fadd() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location), (vector_f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_vector_reduce_fadd(arg_0, arg_1, f32_type, None, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.input(), arg_1);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.fadd");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vector_reduce_fadd_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), vector_f32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_fadd_test(%arg0: f32, %arg1: vector<4xf32>) -> f32 {
                    %0 = \"llvm.intr.vector.reduce.fadd\"(%arg0, %arg1) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, vector<4xf32>) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_fmax() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(vector_f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_vector_reduce_fmax(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.fmax");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vector_reduce_fmax_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_fmax_test(%arg0: vector<4xf32>) -> f32 {
                    %0 = llvm.intr.vector.reduce.fmax(%arg0) : (vector<4xf32>) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_fmaximum() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(vector_f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_vector_reduce_fmaximum(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.fmaximum");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vector_reduce_fmaximum_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_fmaximum_test(%arg0: vector<4xf32>) -> f32 {
                    %0 = llvm.intr.vector.reduce.fmaximum(%arg0) : (vector<4xf32>) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_fmin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(vector_f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_vector_reduce_fmin(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.fmin");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vector_reduce_fmin_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_fmin_test(%arg0: vector<4xf32>) -> f32 {
                    %0 = llvm.intr.vector.reduce.fmin(%arg0) : (vector<4xf32>) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_fminimum() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(vector_f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_vector_reduce_fminimum(arg_0, f32_type, None, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.fminimum");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vector_reduce_fminimum_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_fminimum_test(%arg0: vector<4xf32>) -> f32 {
                    %0 = llvm.intr.vector.reduce.fminimum(%arg0) : (vector<4xf32>) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_fmul() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type.as_ref(), location), (vector_f32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let op = intr_vector_reduce_fmul(arg_0, arg_1, f32_type, None, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.input(), arg_1);
            assert_eq!(op.fastmath_flags().unwrap().to_string(), "#llvm.fastmath<none>");
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.fmul");
            assert_eq!(op.operands().count(), 2);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vector_reduce_fmul_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), vector_f32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_fmul_test(%arg0: f32, %arg1: vector<4xf32>) -> f32 {
                    %0 = \"llvm.intr.vector.reduce.fmul\"(%arg0, %arg1) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, vector<4xf32>) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_mul() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(vector_i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_vector_reduce_mul(arg_0, i32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.mul");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vector_reduce_mul_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_mul_test(%arg0: vector<4xi32>) -> i32 {
                    %0 = \"llvm.intr.vector.reduce.mul\"(%arg0) : (vector<4xi32>) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_or() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(vector_i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_vector_reduce_or(arg_0, i32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.or");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vector_reduce_or_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_or_test(%arg0: vector<4xi32>) -> i32 {
                    %0 = \"llvm.intr.vector.reduce.or\"(%arg0) : (vector<4xi32>) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_smax() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(vector_i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_vector_reduce_smax(arg_0, i32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.smax");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vector_reduce_smax_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_smax_test(%arg0: vector<4xi32>) -> i32 {
                    %0 = \"llvm.intr.vector.reduce.smax\"(%arg0) : (vector<4xi32>) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_smin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(vector_i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_vector_reduce_smin(arg_0, i32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.smin");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vector_reduce_smin_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_smin_test(%arg0: vector<4xi32>) -> i32 {
                    %0 = \"llvm.intr.vector.reduce.smin\"(%arg0) : (vector<4xi32>) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_umax() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(vector_i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_vector_reduce_umax(arg_0, i32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.umax");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vector_reduce_umax_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_umax_test(%arg0: vector<4xi32>) -> i32 {
                    %0 = \"llvm.intr.vector.reduce.umax\"(%arg0) : (vector<4xi32>) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_umin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(vector_i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_vector_reduce_umin(arg_0, i32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.umin");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vector_reduce_umin_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_umin_test(%arg0: vector<4xi32>) -> i32 {
                    %0 = \"llvm.intr.vector.reduce.umin\"(%arg0) : (vector<4xi32>) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_xor() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(vector_i32_type.as_ref(), location)]);
            let arg_0 = block.argument(0).unwrap();
            let op = intr_vector_reduce_xor(arg_0, i32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.xor");
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vector_reduce_xor_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_xor_test(%arg0: vector<4xi32>) -> i32 {
                    %0 = \"llvm.intr.vector.reduce.xor\"(%arg0) : (vector<4xi32>) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vscale() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = intr_vscale(i32_type.as_ref(), location);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vscale");
            assert_eq!(op.operands().count(), 0);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vscale_test",
                func::FuncAttributes { arguments: vec![], results: vec![i32_type.into()], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @llvm_intr_vscale_test() -> i32 {
                    %0 = \"llvm.intr.vscale\"() : () -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }
}
