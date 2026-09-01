use crate::{
    Attribute, DetachedOp, DialectHandle, Error, Location, Operation, OperationBuilder, Type, mlir_op, mlir_op_trait,
};

use super::PoisonAttributeRef;

/// Name of the poison semantics attribute on `ub.poison`.
pub const VALUE_ATTRIBUTE: &str = "value";

/// Operation trait for `ub.poison`.
pub trait PoisonOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the optional poison semantics attribute.
    fn value(&self) -> Result<Option<PoisonAttributeRef<'c, 't>>, Error> {
        self.attribute(VALUE_ATTRIBUTE)?
            .map(|attribute| {
                attribute
                    .cast::<PoisonAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid `value` attribute in `ub.poison`"))
            })
            .transpose()
    }
}

mlir_op!(Poison);
mlir_op_trait!(Poison, AlwaysSpeculatable);
mlir_op_trait!(Poison, ConstantLike);
mlir_op_trait!(Poison, NoMemoryEffect);
mlir_op_trait!(Poison, OneResult);
mlir_op_trait!(Poison, Pure);
mlir_op_trait!(Poison, ZeroOperands);
mlir_op_trait!(Poison, ZeroRegions);
mlir_op_trait!(Poison, ZeroSuccessors);

/// Constructs a new detached/owned [`PoisonOperation`] with a fully poisoned result.
pub fn poison<'c, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    result_type: T,
    value: Option<PoisonAttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedPoisonOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::ub()?)?;
    let mut builder = OperationBuilder::new("ub.poison", location).add_result(result_type);
    if let Some(value) = value {
        builder = builder.add_attribute(VALUE_ATTRIBUTE, value);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `ub::poison`"))
    })
}

/// Operation trait for `ub.unreachable`.
pub trait UnreachableOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Unreachable);
mlir_op_trait!(Unreachable, SingleBlockRegions);
mlir_op_trait!(Unreachable, IsTerminator);
mlir_op_trait!(Unreachable, ZeroOperands);
mlir_op_trait!(Unreachable, ZeroRegions);
mlir_op_trait!(Unreachable, ZeroSuccessors);

/// Constructs a new detached/owned [`UnreachableOperation`].
pub fn unreachable<'c, 't: 'c, L: Location<'c, 't>>(
    location: L,
) -> Result<DetachedUnreachableOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::ub()?)?;
    OperationBuilder::new("ub.unreachable", location).build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `ub::unreachable`"))
    })
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Block, Context, Operation, Value};

    use super::*;

    #[test]
    fn test_poison() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let result_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let operation = poison(result_type, Some(context.ub_poison_attribute().unwrap()), location).unwrap();
                assert_eq!(operation.operand_count(), 0);
                assert_eq!(operation.result(0).unwrap().r#type().unwrap(), result_type);
                assert_eq!(operation.value().unwrap(), Some(context.ub_poison_attribute().unwrap()));
                let operation = block.append_operation(operation).unwrap();
                block.append_operation(func::r#return(&[operation.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "poison_test",
                    func::FuncAttributes { results: vec![result_type.into()], ..Default::default() },
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
                  func.func @poison_test() -> i32 {
                    %0 = ub.poison <#ub.poison> : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_unreachable() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let operation = unreachable(location).unwrap();
                assert_eq!(operation.operand_count(), 0);
                assert_eq!(operation.result_count(), 0);
                block.append_operation(operation).unwrap();
                func::func("unreachable_test", func::FuncAttributes::default(), block.try_into().unwrap(), location)
                    .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @unreachable_test() {
                    ub.unreachable
                  }
                }
            "},
        );
    }
}
