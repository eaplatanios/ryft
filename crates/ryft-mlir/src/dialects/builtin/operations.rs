use crate::{
    DetachedOp, DetachedRegion, IntoWithContext, Location, Operation, OperationBuilder, SYMBOL_NAME_ATTRIBUTE,
    SYMBOL_VISIBILITY_ATTRIBUTE, SymbolVisibility, Type, Value, mlir_op, mlir_op_trait,
};

use super::StringAttributeRef;

/// Represents a built-in MLIR [module](https://mlir.llvm.org/docs/Dialects/Builtin/#builtinmodule-moduleop).
/// A module represents a top-level container [`Operation`]. It contains a single graph [`Region`](crate::Region)
/// with a single [`Block`](crate::Block) which can contain any number of [`Operation`]s and does not have a
/// [terminator](https://mlir.llvm.org/docs/LangRef/#control-flow-and-ssacfg-regions). Operations within this
/// region cannot implicitly capture values defined outside the module (i.e., modules are
/// [`IsolatedFromAbove`](https://mlir.llvm.org/docs/Traits/#isolatedfromabove)). Furthermore, modules have an optional
/// [symbol name](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/#symbol-table) which can be used to refer to them
/// in [`Operation`]s.
///
/// Note that [`Module`](crate::Module)s are simple wrappers of [`ModuleOperation`]s that are provided for convenience
/// by the MLIR C API.
///
/// # Example
///
/// The following is an example of a [`ModuleOperation`]s represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```text
/// module {
///   func.func @foo()
/// }
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#builtinmodule-moduleop)
/// for more information.
pub trait ModuleOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Module);
mlir_op_trait!(Module, AffineScope);
mlir_op_trait!(Module, HasOnlyGraphRegion);
mlir_op_trait!(Module, IsolatedFromAbove);
mlir_op_trait!(Module, NoRegionArguments);
mlir_op_trait!(Module, NoTerminator);
mlir_op_trait!(Module, OneRegion);
mlir_op_trait!(Module, SingleBlock);
mlir_op_trait!(Module, SingleBlockRegions);
mlir_op_trait!(Module, Symbol);
mlir_op_trait!(Module, SymbolTable);

/// Creates an anonymous [`ModuleOperation`] at the specified [`Location`].
pub fn module<'c, 't: 'c, L: Location<'c, 't>>(
    region: DetachedRegion<'c, 't>,
    location: L,
) -> Option<DetachedModuleOperation<'c, 't>> {
    OperationBuilder::new("builtin.module", location)
        .add_region(region)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
}

/// Creates a named [`ModuleOperation`] at the specified [`Location`].
pub fn named_module<'c, 't: 'c, S: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>, L: Location<'c, 't>>(
    name: S,
    visibility: SymbolVisibility,
    region: DetachedRegion<'c, 't>,
    location: L,
) -> Option<DetachedModuleOperation<'c, 't>> {
    let context = location.context();
    let mut builder = OperationBuilder::new("builtin.module", location);
    builder = builder.add_attribute(SYMBOL_NAME_ATTRIBUTE, name.into_with_context(context));
    if visibility != SymbolVisibility::default() {
        builder = builder.add_attribute(SYMBOL_VISIBILITY_ATTRIBUTE, context.symbol_visibility_attribute(visibility));
    }
    builder.add_region(region).build().and_then(|operation| unsafe { operation.cast() })
}

/// Represents an unrealized conversion from one set of types to another, that is used to enable the inter-mixing of
/// different type systems. This operation should not be attributed any special representational or execution
/// semantics, and is generally only intended to be used to satisfy the temporary intermixing of type systems during
/// the conversion of one type system to another.
///
/// This operation may produce results of arity 1-N, and accept as input operands of arity 0-N.
///
/// # Examples
///
/// The following are example uses of this operation represented as MLIR code:
///
/// ```mlir
/// // An unrealized 0-1 conversion. These types of conversions are useful in cases where a type is removed from the
/// // type system, but not all uses have been converted. For example, imagine we have a tuple type that is expanded to
/// // its element types. If only some uses of an empty tuple type instance are converted we still need an instance of
/// // the tuple type, but have no inputs to the unrealized conversion.
/// %result = unrealized_conversion_cast to !bar.tuple_type<>
///
/// // An unrealized 1-1 conversion.
/// %result1 = unrealized_conversion_cast %operand : !foo.type to !bar.lowered_type
///
/// // An unrealized 1-N conversion.
/// %results2:2 = unrealized_conversion_cast %tuple_operand : !foo.tuple_type<!foo.type, !foo.type> to !foo.type, !foo.type
///
/// // An unrealized N-1 conversion.
/// %result3 = unrealized_conversion_cast %operand, %operand : !foo.type, !foo.type to !bar.tuple_type<!foo.type, !foo.type>
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#builtinunrealized_conversion_cast-unrealizedconversioncastop)
/// for more information.
pub trait UnrealizedConversionCastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(UnrealizedConversionCast);
mlir_op_trait!(UnrealizedConversionCast, AlwaysSpeculatable);
mlir_op_trait!(UnrealizedConversionCast, NoMemoryEffect);
mlir_op_trait!(UnrealizedConversionCast, Pure);

/// Creates a new [`UnrealizedConversionCastOperation`] at the specified [`Location`].
pub fn unrealized_conversion_cast<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    arguments: &[V],
    result_types: &[T],
    location: L,
) -> Option<DetachedUnrealizedConversionCastOperation<'c, 't>> {
    OperationBuilder::new("builtin.unrealized_conversion_cast", location)
        .add_operands(arguments)
        .add_results(result_types)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, Operation, Region, ValueRef};

    use super::*;

    #[test]
    fn test_anonymous_module() {
        let context = Context::new();
        let location = context.unknown_location();
        let mut region = context.region();
        region.append_block(context.block_with_no_arguments());
        let module = module(region, location).unwrap();
        assert_eq!(&context, module.context());
        assert_eq!(module.regions().count(), 1);
        assert!(module.verify());
        assert_eq!(module.to_string(), "module {\n}\n");
    }

    #[test]
    fn test_anonymous_module_parse() {
        let context = Context::new();
        let parsed = context.parse_operation("module {}", "test").unwrap();
        assert!(parsed.verify());
        assert_eq!(parsed.to_string(), "module {\n}\n");
    }

    #[test]
    fn test_named_module() {
        let context = Context::new();
        let location = context.unknown_location();
        let mut region = context.region();
        region.append_block(context.block_with_no_arguments());
        let module = named_module("test_module", SymbolVisibility::Public, region, location).unwrap();
        assert_eq!(&context, module.context());
        assert_eq!(module.regions().count(), 1);
        assert!(module.verify());
        assert_eq!(module.to_string(), "module @test_module {\n}\n");
    }

    #[test]
    fn test_named_module_parse() {
        let context = Context::new();
        let parsed = context.parse_operation("module @test_module {}", "test").unwrap();
        assert!(parsed.verify());
        assert_eq!(parsed.to_string(), "module @test_module {\n}\n");
    }

    #[test]
    fn test_named_module_with_private_visibility() {
        let context = Context::new();
        let location = context.unknown_location();
        let mut region = context.region();
        region.append_block(context.block_with_no_arguments());
        let module = named_module("private_module", SymbolVisibility::Private, region, location).unwrap();
        assert!(module.verify());
        assert_eq!(module.to_string(), "module @private_module attributes {sym_visibility = \"private\"} {\n}\n");
    }

    #[test]
    fn test_named_module_with_private_visibility_parse() {
        let context = Context::new();
        let parsed = context
            .parse_operation("module @private_module attributes {sym_visibility = \"private\"} {}", "test")
            .unwrap();
        assert!(parsed.verify());
        assert_eq!(parsed.to_string(), "module @private_module attributes {sym_visibility = \"private\"} {\n}\n");
    }

    #[test]
    fn test_unrealized_conversion_cast() {
        let context = Context::new();
        let location = context.unknown_location();
        let i32_type = context.signless_integer_type(32);
        let op = unrealized_conversion_cast::<ValueRef, _, _>(&[], &[i32_type], location).unwrap();
        assert_eq!(op.operands().count(), 0);
        assert_eq!(op.results().count(), 1);
        assert_eq!(op.result_type(0).unwrap(), i32_type);
        assert!(op.verify());
        assert_eq!(op.to_string(), "%0 = unrealized_conversion_cast to i32\n");
    }
}
