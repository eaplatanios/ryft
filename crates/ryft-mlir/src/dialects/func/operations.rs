use std::collections::HashMap;

use crate::{
    Attribute, AttributeRef, CALLEE_ATTRIBUTE, Call, Callee, DetachedOp, DetachedRegion, DialectHandle,
    FUNCTION_TYPE_ATTRIBUTE, FlatSymbolRefAttributeRef, FunctionTypeRef, HasCallableArgumentAndResultAttributes,
    IntoWithContext, Location, Operation, OperationBuilder, SYMBOL_NAME_ATTRIBUTE, SYMBOL_VISIBILITY_ATTRIBUTE,
    StringAttributeRef, StringRef, SymbolVisibility, Type, TypeAndAttributes, Value, ValueAndAttributes, ValueRef,
    mlir_op, mlir_op_trait,
};

/// [`Operation`] that represents a direct call to a [`FuncOperation`] that is within the same symbol scope as the call.
/// The operands and result types of the call must match the corresponding function type. The callee is encoded as a
/// [`FlatSymbolRefAttributeRef`] named `callee`.
///
/// # Example
///
/// The following is an example of a [`CallOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %2 = func.call @my_add(%0, %1) : (f32, f32) -> f32
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Func/#funccall-funccallop)
/// for more information.
pub trait CallOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the symbol of the [`Function`](crate::Function) that is being called.
    fn function(&self) -> StringRef<'c> {
        self.attribute(CALLEE_ATTRIBUTE).unwrap().cast::<FlatSymbolRefAttributeRef>().unwrap().reference()
    }

    /// Returns an [`Iterator`] over the argument [`Value`]s (i.e., the operands) of this [`CallOperation`].
    ///
    /// Note that the returned iterator does not hold a borrowed reference to the underlying [`Context`](crate::Context)
    /// because that would make it impossible to perform mutating operations on that context (e.g., from within
    /// [`Pass`](crate::Pass)es) while iterating over the contents of that iterator.
    fn arguments(&self) -> impl Iterator<Item = ValueRef<'o, 'c, 't>> {
        self.operands()
    }

    /// Returns the value of the `no_inline` [`Attribute`] for this [`Operation`]. If `true`, then the compiler will be
    /// instructed to not inline the body of the underlying function in this function call, even if inlining would
    /// otherwise be beneficial from an optimization perspective.
    fn no_inline(&self) -> bool {
        self.has_attribute(FUNCTION_NO_INLINE_ATTRIBUTE)
    }
}

mlir_op!(Call);
mlir_op_trait!(Call, HasCallableArgumentAndResultAttributes);
mlir_op_trait!(Call, MemRefsNormalizable);

impl<'o, 'c: 'o, 't: 'c> Call<'o, 'c, 't> for DetachedCallOperation<'c, 't> {
    fn callee(&self) -> Callee<'o, 'c, 't> {
        let attribute = self.attribute(CALLEE_ATTRIBUTE).unwrap();
        Callee::Symbol(attribute.cast::<FlatSymbolRefAttributeRef>().unwrap())
    }
}

impl<'o, 'c: 'o, 't: 'c> Call<'o, 'c, 't> for CallOperationRef<'o, 'c, 't> {
    fn callee(&self) -> Callee<'o, 'c, 't> {
        let attribute = self.attribute(CALLEE_ATTRIBUTE).unwrap();
        Callee::Symbol(attribute.cast::<FlatSymbolRefAttributeRef>().unwrap())
    }
}

/// Structured representation of the operands, results, and [`Attribute`]s that are attached to a [`CallOperation`].
/// This struct can be used to construct [`CallOperation`]s via [`call`] and provides a [`Default`] implementation
/// making [`CallOperation`] construction more ergonomic.
pub struct CallProperties<'v, 'c, 't, 's> {
    /// [`Vec`] that contains the [`Value`]s of the [`Call`] arguments along with any [`Attribute`]s they may have.
    pub arguments: Vec<ValueAndAttributes<'v, 'c, 't, 's>>,

    /// [`Vec`] that contains the [`Type`]s of the [`Call`] results along with any [`Attribute`]s they may have.
    pub results: Vec<TypeAndAttributes<'c, 't, 's>>,

    /// Refer to [`CallOperation::no_inline`] for information on this property.
    pub no_inline: bool,
}

impl Default for CallProperties<'_, '_, '_, '_> {
    fn default() -> Self {
        Self { arguments: Vec::new(), results: Vec::new(), no_inline: false }
    }
}

/// Constructs a new detached/owned [`CallOperation`] at the specified [`Location`] for the provided callee and
/// [`CallProperties`]. Refer to the documentation of [`CallOperation`] and [`CallProperties`] for more information.
pub fn call<
    'v,
    'c: 'v,
    't: 'c,
    's,
    C: IntoWithContext<'c, 't, FlatSymbolRefAttributeRef<'c, 't>>,
    L: Location<'c, 't>,
>(
    callee: C,
    properties: CallProperties<'v, 'c, 't, 's>,
    location: L,
) -> DetachedCallOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::func());
    let mut builder = OperationBuilder::new("func.call", location)
        .add_attribute(CALLEE_ATTRIBUTE, callee.into_with_context(context))
        .add_operands(&properties.arguments.iter().map(|argument| argument.value).collect::<Vec<_>>())
        .add_results(&properties.results.iter().map(|result| result.r#type).collect::<Vec<_>>());

    if properties.arguments.iter().any(|argument| argument.attributes.is_some()) {
        builder = DetachedCallOperation::<'c, 't>::add_callable_argument_attributes(
            builder,
            properties.arguments.iter().map(|argument| &argument.attributes),
        );
    }

    if properties.results.iter().any(|result| result.attributes.is_some()) {
        builder = DetachedCallOperation::<'c, 't>::add_callable_result_attributes(
            builder,
            properties.results.iter().map(|result| &result.attributes),
        );
    }

    if properties.no_inline {
        builder = builder.add_attribute(FUNCTION_NO_INLINE_ATTRIBUTE, context.unit_attribute());
    }

    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `func::call`")
}

/// [`Operation`] that represents an indirect call to a [`Value`] of function type (i.e., [`FunctionTypeRef`]). The
/// operands and result types of the call must match the corresponding function type. Note that function-typed
/// [`Value`]s can be created using [`func::constant`](constant).
///
/// # Example
///
/// The following is an example of a [`CallIndirectOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %func = func.constant @my_func : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
/// %result = func.call_indirect %func(%0, %1) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Func/#funccall-funccallop)
/// for more information.
pub trait CallIndirectOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the [`Value`] that represents the [`Function`](crate::Function) that is being called.
    fn function(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns an [`Iterator`] over the argument [`Value`]s (i.e., the operands) of this [`CallOperation`].
    ///
    /// Note that the returned iterator does not hold a borrowed reference to the underlying [`Context`](crate::Context)
    /// because that would make it impossible to perform mutating operations on that context (e.g., from within
    /// [`Pass`](crate::Pass)es) while iterating over the contents of that iterator.
    fn arguments(&self) -> impl Iterator<Item = ValueRef<'o, 'c, 't>> {
        self.operands().skip(1)
    }
}

mlir_op!(CallIndirect);
mlir_op_trait!(CallIndirect, HasCallableArgumentAndResultAttributes);

impl<'o, 'c: 'o, 't: 'c> Call<'o, 'c, 't> for DetachedCallIndirectOperation<'c, 't> {
    fn callee(&self) -> Callee<'o, 'c, 't> {
        Callee::Value(self.function())
    }
}

impl<'o, 'c: 'o, 't: 'c> Call<'o, 'c, 't> for CallIndirectOperationRef<'o, 'c, 't> {
    fn callee(&self) -> Callee<'o, 'c, 't> {
        Callee::Value(self.function())
    }
}

/// Structured representation of the operands and results of a [`CallIndirectOperation`]. This struct can be used to
/// construct [`CallIndirectOperation`]s via [`call_indirect`] and provides a [`Default`] implementation making
/// [`CallIndirectOperation`] construction more ergonomic.
pub struct CallIndirectProperties<'v, 'c, 't, 's> {
    /// [`Vec`] that contains the [`Value`]s of the [`Call`] arguments along with any [`Attribute`]s they may have.
    pub arguments: Vec<ValueAndAttributes<'v, 'c, 't, 's>>,

    /// [`Vec`] that contains the [`Type`]s of the [`Call`] results along with any [`Attribute`]s they may have.
    pub results: Vec<TypeAndAttributes<'c, 't, 's>>,
}

impl Default for CallIndirectProperties<'_, '_, '_, '_> {
    fn default() -> Self {
        Self { arguments: Vec::new(), results: Vec::new() }
    }
}

/// Constructs a new detached/owned [`CallIndirectOperation`] at the specified [`Location`] for the provided callee and
/// [`CallIndirectProperties`]. Refer to the documentation of [`CallIndirectOperation`] and [`CallIndirectProperties`]
/// for more information.
pub fn call_indirect<'f, 'v, 'c: 'f + 'v, 't: 'c, 's, C: Value<'f, 'c, 't>, L: Location<'c, 't>>(
    callee: C,
    properties: CallIndirectProperties<'v, 'c, 't, 's>,
    location: L,
) -> DetachedCallIndirectOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::func());
    let mut builder = OperationBuilder::new("func.call_indirect", location)
        .add_operand(callee)
        .add_operands(&properties.arguments.iter().map(|argument| argument.value).collect::<Vec<_>>())
        .add_results(&properties.results.iter().map(|result| result.r#type).collect::<Vec<_>>());

    if properties.arguments.iter().any(|argument| argument.attributes.is_some()) {
        builder = DetachedCallIndirectOperation::<'c, 't>::add_callable_argument_attributes(
            builder,
            properties.arguments.iter().map(|argument| &argument.attributes),
        );
    }

    if properties.results.iter().any(|result| result.attributes.is_some()) {
        builder = DetachedCallIndirectOperation::<'c, 't>::add_callable_result_attributes(
            builder,
            properties.results.iter().map(|result| &result.attributes),
        );
    }

    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `func::call_indirect`")
}

/// Name of the [`Attribute`] that is used to store [`ConstantOperation::function`].
pub const FUNCTION_CONSTANT_VALUE_ATTRIBUTE: &'static str = "value";

/// [`Operation`] that produces an [SSA](https://en.wikipedia.org/wiki/Static_single-assignment_form) value from a
/// symbol reference to a [`FuncOperation`].
///
/// # Example
///
/// The following are examples of [`ConstantOperation`]s represented using their
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // Reference to the `@myfn` function:
/// %2 = func.constant @myfn : (tensor<16xf32>, f32) -> tensor<16xf32>
///
/// // Equivalent generic form:
/// %2 = "func.constant"() { value = @myfn } : () -> ((tensor<16xf32>, f32) -> tensor<16xf32>)
/// ```
///
/// MLIR does not allow direct references to functions in SSA operands because the compiler is multithreaded and
/// disallowing SSA values to directly reference a function simplifies this
/// [rationale](https://mlir.llvm.org/docs/Rationale/Rationale/#multithreading-the-compiler).
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Func/#funcconstant-funcconstantop)
/// for more information.
pub trait ConstantOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the symbol of the underlying [`Function`](crate::Function) value.
    fn function(&self) -> StringRef<'c> {
        self.attribute(FUNCTION_CONSTANT_VALUE_ATTRIBUTE)
            .unwrap()
            .cast::<FlatSymbolRefAttributeRef>()
            .unwrap()
            .reference()
    }

    /// Returns the [`FunctionTypeRef`] of the underlying [`Function`](crate::Function) value.
    fn function_type(&self) -> FunctionTypeRef<'c, 't> {
        self.result_type(0).unwrap().cast().unwrap()
    }
}

mlir_op!(Constant);
mlir_op_trait!(Constant, AlwaysSpeculatable);
mlir_op_trait!(Constant, ConstantLike);
mlir_op_trait!(Constant, NoMemoryEffect);
mlir_op_trait!(Constant, OneResult);
mlir_op_trait!(Constant, Pure);
mlir_op_trait!(Constant, ZeroOperands);

/// Constructs a new detached/owned [`ConstantOperation`] at the specified [`Location`] with the provided underlying
/// [`Function`](crate::Function) reference and type. Refer to the documentation of [`ConstantOperation`] for more
/// information.
pub fn constant<
    'c,
    't: 'c,
    V: IntoWithContext<'c, 't, FlatSymbolRefAttributeRef<'c, 't>>,
    T: IntoWithContext<'c, 't, FunctionTypeRef<'c, 't>>,
    L: Location<'c, 't>,
>(
    function: V,
    function_type: T,
    location: L,
) -> DetachedConstantOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::func());
    OperationBuilder::new("func.constant", location)
        .add_attribute(FUNCTION_CONSTANT_VALUE_ATTRIBUTE, function.into_with_context(context))
        .add_result(function_type.into_with_context(context))
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `func::constant`")
}

/// Name of the [`Attribute`] that is used to store [`FuncOperation::no_inline`].
pub const FUNCTION_NO_INLINE_ATTRIBUTE: &'static str = "no_inline";

/// Name of the [`Attribute`] that is used to store [`FuncOperation::llvm_emit_c_interface`].
pub const FUNCTION_LLVM_EMIT_C_INTERFACE_ATTRIBUTE: &'static str = "llvm.emit_c_interface";

/// [`Operation`] that contains a single [SSACFG](https://mlir.llvm.org/docs/LangRef/#control-flow-and-ssacfg-regions)
/// [`Region`](crate::Region), has a name, and that is a [`Function`](crate::Function). [`Operation`]s within the
/// underlying region are not allowed to capture values defined outside of that region (i.e., [`FuncOperation`]s are
/// [`IsolatedFromAbove`](crate::IsolatedFromAbove)). All external references must use function arguments or
/// [`Attribute`]s that establish a symbolic connection (e.g., symbols referenced by name via a
/// [`FlatSymbolRefAttributeRef`]). _External_ function declarations (used when referring to functions that are declared
/// in some other [`Module`](crate::Module)) are also supported. They are represented as [`FuncOperation`]s with no
/// body. While the MLIR textual representation provides a nice inline syntax for function arguments, they are
/// internally represented as just arguments to the first [`Block`](crate::Block) in the underlying
/// [`Region`](crate::Region).
///
/// # Examples
///
/// The following are examples of [`FuncOperation`]s represented using their [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // External function definitions:
/// func.func private @abort()
/// func.func private @scribble(i32, i64, memref<? x 128 x f32, #layout_map0>) -> f64
///
/// // A function that returns its argument twice:
/// func.func @count(%x: i64) -> (i64, i64)
///   attributes {fruit = "banana"} {
///   return %x, %x: i64, i64
/// }
///
/// // A function with an argument attribute:
/// func.func private @example_fn_arg(%x: i32 {swift.self = unit})
///
/// // A function with a result attribute:
/// func.func private @example_fn_result() -> (f64 {dialectName.attrName = 0 : i64})
///
/// // A function with an attribute:
/// func.func private @example_fn_attr() attributes {dialectName.attrName = false}
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Func/#funcfunc-funcfuncop)
/// for more information.
pub trait FuncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the value of the `no_inline` [`Attribute`] for this [`Operation`]. If `true`, then the compiler will be
    /// instructed to not inline this function into its call sites, even if inlining would otherwise be beneficial from
    /// an optimization perspective.
    fn no_inline(&self) -> bool {
        self.has_attribute(FUNCTION_NO_INLINE_ATTRIBUTE)
    }

    /// Returns the value of the `llvm.emit_c_interface` [`Attribute`] for this [`Operation`]. If `true`, the
    /// MLIR-to-LLVM lowering process will also generate a C-compatible wrapper function for the annotated function
    /// (i.e., a wrapper function which can be called from C code). This is sometimes needed because MLIR functions
    /// often use calling conventions that are not C-compatible. The generated C wrapper function handles the marshaling
    /// between C-style arguments and MLIR's internal representations. You must set this to `true` if: (i) you need to
    /// call MLIR-compiled functions from C/C++ code, (ii) you are creating a library interface that will be used
    /// outside MLIR, or (iii) you want to interoperate with other languages through a C Foreign Function Interface
    /// (FFI). When this is `false`, calling MLIR functions from C requires manually matching MLIR's internal ABI,
    /// which can be complex and fragile.
    fn llvm_emit_c_interface(&self) -> bool {
        self.has_attribute(FUNCTION_LLVM_EMIT_C_INTERFACE_ATTRIBUTE)
    }
}

mlir_op!(Func);
mlir_op_trait!(Func, AffineScope);
mlir_op_trait!(Func, AutomaticAllocationScope);
mlir_op_trait!(Func, Callable);
mlir_op_trait!(Func, Function);
mlir_op_trait!(Func, HasCallableArgumentAndResultAttributes);
mlir_op_trait!(Func, IsolatedFromAbove);
mlir_op_trait!(Func, Symbol);

/// Structured representation of the [`Attribute`]s that are attached to a [`FuncOperation`]. This struct can be used
/// to construct [`FuncOperation`]s via [`func`] and provides a [`Default`] implementation making [`FuncOperation`]
/// construction more ergonomic.
pub struct FuncAttributes<'c, 't, 's> {
    /// [`Vec`] that contains the [`Type`]s of the function arguments along with any [`Attribute`]s they may have.
    pub arguments: Vec<TypeAndAttributes<'c, 't, 's>>,

    /// [`Vec`] that contains the [`Type`]s of the function results along with any [`Attribute`]s they may have.
    pub results: Vec<TypeAndAttributes<'c, 't, 's>>,

    /// [`SymbolVisibility`] for the function. Refer to [`Symbol::symbol_visibility`](crate::Symbol::symbol_visibility)
    /// for more information.
    pub visibility: SymbolVisibility,

    /// Refer to [`FuncOperation::no_inline`] for information on this property.
    pub no_inline: bool,

    /// Refer to [`FuncOperation::llvm_emit_c_interface`] for information on this property.
    pub llvm_emit_c_interface: bool,

    /// Map from names to custom [`Attribute`]s for the function.
    pub other_attributes: HashMap<&'c str, AttributeRef<'c, 't>>,
}

impl Default for FuncAttributes<'_, '_, '_> {
    fn default() -> Self {
        Self {
            arguments: Vec::new(),
            results: Vec::new(),
            visibility: SymbolVisibility::Public,
            no_inline: false,
            llvm_emit_c_interface: false,
            other_attributes: HashMap::new(),
        }
    }
}

/// Constructs a new detached/owned [`FuncOperation`] at the specified [`Location`] with the provided name,
/// [`FuncAttributes`], and body. Refer to the documentation of [`FuncOperation`] and [`FuncAttributes`] for more
/// information.
pub fn func<'c, 't: 'c, 's, N: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>, L: Location<'c, 't>>(
    name: N,
    attributes: FuncAttributes<'c, 't, 's>,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedFuncOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::func());

    let mut builder = OperationBuilder::new("func.func", location)
        .add_attribute(SYMBOL_NAME_ATTRIBUTE, name.into_with_context(context));

    builder = builder.add_attribute(
        FUNCTION_TYPE_ATTRIBUTE,
        context.type_attribute(context.function_type(
            &attributes.arguments.iter().map(|argument| argument.r#type).collect::<Vec<_>>(),
            &attributes.results.iter().map(|result| result.r#type).collect::<Vec<_>>(),
        )),
    );

    if attributes.arguments.iter().any(|argument| argument.attributes.is_some()) {
        builder = DetachedFuncOperation::<'c, 't>::add_callable_argument_attributes(
            builder,
            attributes.arguments.iter().map(|argument| &argument.attributes),
        );
    }

    if attributes.results.iter().any(|result| result.attributes.is_some()) {
        builder = DetachedFuncOperation::<'c, 't>::add_callable_result_attributes(
            builder,
            attributes.results.iter().map(|result| &result.attributes),
        );
    }

    if attributes.visibility != SymbolVisibility::default() {
        builder = builder
            .add_attribute(SYMBOL_VISIBILITY_ATTRIBUTE, context.symbol_visibility_attribute(attributes.visibility));
    }

    if attributes.no_inline {
        builder = builder.add_attribute(FUNCTION_NO_INLINE_ATTRIBUTE, context.unit_attribute());
    }

    if attributes.llvm_emit_c_interface {
        builder = builder.add_attribute(FUNCTION_LLVM_EMIT_C_INTERFACE_ATTRIBUTE, context.unit_attribute());
    }

    for (attribute_name, attribute) in &attributes.other_attributes {
        builder = builder.add_attribute(*attribute_name, *attribute)
    }

    builder
        .add_region(body)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `func::func`")
}

/// [`Operation`] that represents a return operation from within a [`FuncOperation`]. It takes variable number of
/// operands and produces no results. The operand number and types must match the signature of the parent
/// [`FuncOperation`].
///
/// # Example
///
/// The following is an example of a [`ReturnOperation`] represented using its [`Display`](std::fmt::Display) rendering
/// in the context of a [`FuncOperation`]:
///
/// ```mlir
/// func.func @foo() -> (i32, f8) {
///   ...
///   return %0, %1 : i32, f8
/// }
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Func/#funcreturn-funcreturnop)
/// for more information.
pub trait ReturnOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns an [`Iterator`] over the return [`Value`]s (i.e., the operands) of this [`ReturnOperation`].
    ///
    /// Note that the returned iterator does not hold a borrowed reference to the underlying [`Context`](crate::Context)
    /// because that would make it impossible to perform mutating operations on that context (e.g., from within
    /// [`Pass`](crate::Pass)es) while iterating over the contents of that iterator.
    fn values(&self) -> impl Iterator<Item = ValueRef<'o, 'c, 't>> {
        self.operands()
    }
}

mlir_op!(Return);
mlir_op_trait!(Return, AlwaysSpeculatable);
mlir_op_trait!(Return, MemRefsNormalizable);
mlir_op_trait!(Return, NoMemoryEffect);
mlir_op_trait!(Return, Pure);
mlir_op_trait!(Return, ZeroRegions);

/// Constructs a new detached/owned [`ReturnOperation`] at the specified [`Location`] and with the provided operands.
/// Refer to the documentation of [`ReturnOperation`] for more information.
pub fn r#return<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    values: &[V],
    location: L,
) -> DetachedReturnOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::func());
    OperationBuilder::new("func.return", location)
        .add_operands(values)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `func::return`")
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::{Block, Context, Function, OpRef, Operation};

    use super::*;

    fn identity_func<'c, 't, T: Type<'c, 't>, L: Location<'c, 't>>(
        context: &'c Context<'t>,
        r#type: T,
        location: L,
    ) -> DetachedFuncOperation<'c, 't> {
        let mut block = context.block(&[(r#type, location)]);
        block.append_operation(r#return(&[block.argument(0).unwrap()], location));
        func(
            "identity",
            FuncAttributes { arguments: vec![r#type.into()], results: vec![r#type.into()], ..Default::default() },
            vec![block].into_with_context(&context),
            location,
        )
    }

    #[test]
    fn test_simple_func_and_call() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();

        module.body().append_operation(identity_func(&context, context.float32_type(), location));

        // Define a function called `caller` which calls `identity` from within its body.
        module.body().append_operation({
            let mut block = context.block(&[(f32_type, location), (f32_type, location)]);
            let op = call(
                "identity",
                CallProperties {
                    arguments: vec![block.argument(0).unwrap().into()],
                    results: vec![f32_type.into()],
                    no_inline: false,
                },
                location,
            );

            // Check that the `function` accessor of [`CallOperation`] works as expected.
            assert_eq!(op.function().as_str().unwrap(), "identity");

            // Check that the `arguments` accessor of [`CallOperation`] works as expected.
            assert_eq!(op.arguments().collect::<Vec<_>>().len(), 1);

            // Check that the `results` accessor of [`CallOperation`] works as expected.
            assert_eq!(op.results().collect::<Vec<_>>().len(), 1);

            let op = block.append_operation(op);
            block.append_operation(r#return(&[op.result(0).unwrap()], location));
            func(
                "caller",
                FuncAttributes {
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
                  func.func @identity(%arg0: f32) -> f32 {
                    return %arg0 : f32
                  }
                  func.func @caller(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = call @identity(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_recursive_func_and_call() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let index_type = context.index_type();

        // Define a function called `foo` which calls itself from within its body.
        module.body().append_operation({
            let mut block = context.block(&[(index_type, location)]);
            let op = call(
                "foo",
                CallProperties {
                    arguments: vec![block.argument(0).unwrap().into()],
                    results: vec![index_type.into()],
                    ..Default::default()
                },
                location,
            );

            // Check that the `function` accessor of [`CallOperation`] works as expected.
            assert_eq!(op.function().as_str().unwrap(), "foo");

            let op = block.append_operation(op);
            block.append_operation(r#return(&[op.result(0).unwrap()], location));
            func(
                "foo",
                FuncAttributes {
                    arguments: vec![index_type.into()],
                    results: vec![index_type.into()],
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
                  func.func @foo(%arg0: index) -> index {
                    %0 = call @foo(%arg0) : (index) -> index
                    return %0 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_func_and_call_with_no_inline() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);

        module.body().append_operation(identity_func(&context, i32_type, location));

        // Define a function called `caller` which calls `identity` from within its body with the `no_inline` attribute.
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location)]);
            let op = call(
                "identity",
                CallProperties {
                    arguments: vec![block.argument(0).unwrap().into()],
                    results: vec![i32_type.into()],
                    no_inline: true,
                },
                location,
            );
            assert_eq!(
                format!("{:?}", op),
                "DetachedCallOperation[%0 = func.call @identity(<<UNKNOWN SSA VALUE>>) {no_inline} : (i32) -> i32\n]",
            );

            // Test equality and hashing as a fly-by in here.
            let dummy_op = call(
                "identity",
                CallProperties {
                    arguments: vec![block.argument(0).unwrap().into()],
                    results: vec![i32_type.into()],
                    no_inline: true,
                },
                location,
            );
            assert_ne!(op, dummy_op);
            let mut map = HashMap::new();
            map.insert(&op, "op");
            map.insert(&dummy_op, "dummy_op");
            assert_eq!(map.len(), 2);
            assert_eq!(map.get(&op), Some(&"op"));
            assert_eq!(map.get(&dummy_op), Some(&"dummy_op"));
            let mut map = HashMap::new();
            let op_ref = CallOperationRef::from(&op).as_operation_ref();
            let dummy_op_ref = dummy_op.as_operation_ref();
            map.insert(&op_ref, "op");
            map.insert(&dummy_op_ref, "dummy_op");
            assert_eq!(map.len(), 2);
            assert_eq!(map.get(&op_ref), Some(&"op"));
            assert_eq!(map.get(&dummy_op_ref), Some(&"dummy_op"));

            // Check that the `callee` and `function` accessors of [`CallOperation`] work as expected.
            assert!(matches!(op.callee(), Callee::Symbol(_)));
            assert_eq!(op.function().as_str().unwrap(), "identity");

            let op_ref = unsafe { op.as_operation_ref().cast::<CallOperationRef>() }.unwrap();
            assert!(matches!(op_ref.callee(), Callee::Symbol(_)));

            // Check that the `no_inline` accessor of [`CallOperation`] works as expected.
            assert!(op.no_inline());

            // Check that the `arguments` accessor of [`CallOperation`] works as expected.
            assert_eq!(op.arguments().collect::<Vec<_>>().len(), 1);

            // Check that the `results` accessor of [`CallOperation`] works as expected.
            assert_eq!(op.results().collect::<Vec<_>>().len(), 1);

            let op = block.append_operation(op);
            block.append_operation(r#return(&[op.result(0).unwrap()], location));
            func(
                "caller",
                FuncAttributes {
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
                  func.func @identity(%arg0: i32) -> i32 {
                    return %arg0 : i32
                  }
                  func.func @caller(%arg0: i32) -> i32 {
                    %0 = call @identity(%arg0) {no_inline} : (i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_func_with_multiple_arguments() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f64_type = context.float64_type();

        module.body().append_operation(identity_func(&context, f64_type, location));

        // Define a function called `multi_arg` which calls `identity` from within its body.
        module.body().append_operation({
            let mut block = context.block(&[(f64_type, location), (f64_type, location), (f64_type, location)]);
            block.append_operation(call(
                "identity",
                CallProperties {
                    arguments: vec![ValueAndAttributes {
                        value: block.argument(2).unwrap().as_value_ref(),
                        attributes: Some(HashMap::from([(
                            StringRef::from("dummy"),
                            context.string_attribute("42").as_attribute_ref(),
                        )])),
                    }],
                    results: vec![TypeAndAttributes {
                        r#type: f64_type.as_type_ref(),
                        attributes: Some(HashMap::from([(
                            StringRef::from("42"),
                            context.string_attribute("dummy").as_attribute_ref(),
                        )])),
                    }],
                    ..Default::default()
                },
                location,
            ));
            block.append_operation(call(
                "identity",
                CallProperties {
                    arguments: vec![block.argument(1).unwrap().into()],
                    results: vec![f64_type.into()],
                    ..Default::default()
                },
                location,
            ));
            let call_op = block.append_operation(call(
                "identity",
                CallProperties {
                    arguments: vec![block.argument(0).unwrap().into()],
                    results: vec![f64_type.into()],
                    ..Default::default()
                },
                location,
            ));
            block.append_operation(r#return(&[call_op.result(0).unwrap()], location));

            let func_op = func(
                "multi_arg",
                FuncAttributes {
                    arguments: vec![f64_type.into(), f64_type.into(), f64_type.into()],
                    results: vec![f64_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            );

            assert_eq!(func_op.operands().collect::<Vec<_>>().len(), 0);
            assert_eq!(func_op.results().collect::<Vec<_>>().len(), 0);

            // Check that the `arguments` accessor of [`FuncOperation`] works as expected.
            assert_eq!(func_op.function_argument_types().len(), 3);

            // Check that the `results` accessor of [`FuncOperation`] works as expected.
            assert_eq!(func_op.function_result_types().len(), 1);

            // Check that the `no_inline` accessor of [`FuncOperation`] works as expected.
            assert!(!func_op.no_inline());

            // Check that the `llvm_emit_c_interface` accessor of [`FuncOperation`] works as expected.
            assert!(!func_op.llvm_emit_c_interface());

            func_op
        });

        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @identity(%arg0: f64) -> f64 {
                    return %arg0 : f64
                  }
                  func.func @multi_arg(%arg0: f64, %arg1: f64, %arg2: f64) -> f64 {
                    %0 = call @identity(%arg2) {\
                      arg_attrs = [{dummy = \"42\"}], \
                      res_attrs = [{\"42\" = \"dummy\"}]\
                    } : (f64) -> f64
                    %1 = call @identity(%arg1) : (f64) -> f64
                    %2 = call @identity(%arg0) : (f64) -> f64
                    return %2 : f64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_func_and_call_with_multiple_arguments() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f64_type = context.float64_type();

        // Define a function called `multi_arg` which takes three arguments.
        module.body().append_operation({
            let mut block = context.block(&[(f64_type, location), (f64_type, location), (f64_type, location)]);
            block.append_operation(r#return::<ValueRef, _>(&[], location));
            func(
                "multi_arg",
                FuncAttributes {
                    arguments: vec![f64_type.into(), f64_type.into(), f64_type.into()],
                    results: vec![],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });

        // Define a function called `multi_arg_caller` which calls `multi_arg` from within its body.
        module.body().append_operation({
            let mut block = context.block(&[(f64_type, location), (f64_type, location), (f64_type, location)]);
            block.append_operation(call(
                "multi_arg",
                CallProperties {
                    arguments: vec![
                        block.argument(2).unwrap().into(),
                        block.argument(1).unwrap().into(),
                        block.argument(0).unwrap().into(),
                    ],
                    results: vec![],
                    ..Default::default()
                },
                location,
            ));
            block.append_operation(r#return(&[block.argument(1).unwrap()], location));
            func(
                "multi_arg_caller",
                FuncAttributes {
                    arguments: vec![f64_type.into(), f64_type.into(), f64_type.into()],
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
                  func.func @multi_arg(%arg0: f64, %arg1: f64, %arg2: f64) {
                    return
                  }
                  func.func @multi_arg_caller(%arg0: f64, %arg1: f64, %arg2: f64) -> f64 {
                    call @multi_arg(%arg2, %arg1, %arg0) : (f64, f64, f64) -> ()
                    return %arg1 : f64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_func_constant_and_call_indirect() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f64_type = context.float64_type();
        let i64_type = context.signless_integer_type(64);
        let function_type = context.function_type(&[f64_type], &[f64_type]);

        module.body().append_operation(identity_func(&context, f64_type, location));

        // Define a function called `caller` which calls `identity` from within its body.
        module.body().append_operation({
            let mut block = context.block(&[(f64_type.as_type_ref(), location), (i64_type.as_type_ref(), location)]);
            let constant_op = constant("identity", function_type, location);
            assert_eq!(constant_op.function(), StringRef::from("identity"));
            assert_eq!(constant_op.function_type(), function_type);

            let identity = block.append_operation(constant_op);

            let op = call_indirect(
                identity.result(0).unwrap(),
                CallIndirectProperties {
                    arguments: vec![ValueAndAttributes {
                        value: block.argument(0).unwrap().as_value_ref(),
                        attributes: Some(HashMap::from([(
                            StringRef::from("dummy"),
                            context.string_attribute("42").as_attribute_ref(),
                        )])),
                    }],
                    results: vec![TypeAndAttributes {
                        r#type: f64_type.as_type_ref(),
                        attributes: Some(HashMap::from([(
                            StringRef::from("42"),
                            context.string_attribute("dummy").as_attribute_ref(),
                        )])),
                    }],
                    ..Default::default()
                },
                location,
            );

            // Check that the `callee` accessor of [`CallOperation`] works as expected.
            assert!(matches!(op.callee(), Callee::Value(_)));

            let op_ref = unsafe { op.as_operation_ref().cast::<CallIndirectOperationRef>() }.unwrap();
            assert!(matches!(op_ref.callee(), Callee::Value(_)));

            // Check that the `function` accessor of [`CallIndirectOperation`] works as expected.
            assert_eq!(op.function(), identity.result(0).unwrap());

            // Check that the `arguments` accessor of [`CallIndirectOperation`] works as expected.
            assert_eq!(op.arguments().collect::<Vec<_>>().len(), 1);

            block.append_operation(op);
            block.append_operation(r#return(&[block.argument(1).unwrap()], location));
            func(
                "caller",
                FuncAttributes {
                    arguments: vec![f64_type.into(), i64_type.into()],
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
                  func.func @identity(%arg0: f64) -> f64 {
                    return %arg0 : f64
                  }
                  func.func @caller(%arg0: f64, %arg1: i64) -> i64 {
                    %f = constant @identity : (f64) -> f64
                    %0 = call_indirect %f(%arg0) {\
                      arg_attrs = [{dummy = \"42\"}], \
                      res_attrs = [{\"42\" = \"dummy\"}]\
                    } : (f64) -> f64
                    return %arg1 : i64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_func_with_no_inline() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f64_type = context.float64_type();

        // Define a function called `no_inline_function` that has the `no_inline` attribute.
        module.body().append_operation({
            let mut block = context.block(&[(f64_type, location), (f64_type, location), (f64_type, location)]);
            block.append_operation(r#return(&[block.argument(1).unwrap()], location));
            let func_op = func(
                "no_inline_function",
                FuncAttributes {
                    arguments: vec![f64_type.into(), f64_type.into(), f64_type.into()],
                    results: vec![f64_type.into()],
                    no_inline: true,
                    ..Default::default()
                },
                block.into(),
                location,
            );

            assert_eq!(func_op.operands().collect::<Vec<_>>().len(), 0);
            assert_eq!(func_op.results().collect::<Vec<_>>().len(), 0);

            // Check that the `arguments` accessor of [`FuncOperation`] works as expected.
            assert_eq!(func_op.function_argument_types().len(), 3);

            // Check that the `results` accessor of [`FuncOperation`] works as expected.
            assert_eq!(func_op.function_result_types().len(), 1);

            // Check that the `no_inline` accessor of [`FuncOperation`] works as expected.
            assert!(func_op.no_inline());

            // Check that the `llvm_emit_c_interface` accessor of [`FuncOperation`] works as expected.
            assert!(!func_op.llvm_emit_c_interface());

            func_op
        });

        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @no_inline_function(%arg0: f64, %arg1: f64, %arg2: f64) -> f64 attributes {no_inline} {
                    return %arg1 : f64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_func_with_llvm_emit_c_interface() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f64_type = context.float64_type();

        // Define a function called `c_function` that has the `llvm.emit_c_interface` attribute.
        module.body().append_operation({
            let mut block = context.block(&[(f64_type, location), (f64_type, location), (f64_type, location)]);
            block.append_operation(r#return(&[block.argument(1).unwrap()], location));
            let func_op = func(
                "c_function",
                FuncAttributes {
                    arguments: vec![f64_type.into(), f64_type.into(), f64_type.into()],
                    results: vec![f64_type.into()],
                    llvm_emit_c_interface: true,
                    ..Default::default()
                },
                block.into(),
                location,
            );

            assert_eq!(func_op.operands().collect::<Vec<_>>().len(), 0);
            assert_eq!(func_op.results().collect::<Vec<_>>().len(), 0);

            // Check that the `arguments` accessor of [`FuncOperation`] works as expected.
            assert_eq!(func_op.function_argument_types().len(), 3);

            // Check that the `results` accessor of [`FuncOperation`] works as expected.
            assert_eq!(func_op.function_result_types().len(), 1);

            // Check that the `no_inline` accessor of [`FuncOperation`] works as expected.
            assert!(!func_op.no_inline());

            // Check that the `llvm_emit_c_interface` accessor of [`FuncOperation`] works as expected.
            assert!(func_op.llvm_emit_c_interface());

            func_op
        });

        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @c_function(%arg0: f64, %arg1: f64, %arg2: f64) -> f64 attributes {llvm.emit_c_interface} {
                    return %arg1 : f64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_func_with_multiple_attributes() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f64_type = context.float64_type();
        module.body().append_operation({
            let mut block = context.block(&[(f64_type, location), (f64_type, location), (f64_type, location)]);
            block.append_operation(r#return(&[block.argument(1).unwrap()], location));
            func(
                "custom_function",
                FuncAttributes {
                    visibility: SymbolVisibility::Private,
                    arguments: vec![f64_type.into(), f64_type.into(), f64_type.into()],
                    results: vec![f64_type.into()],
                    no_inline: true,
                    llvm_emit_c_interface: true,
                    other_attributes: HashMap::from([
                        ("custom.custom_1", context.unit_attribute().as_attribute_ref()),
                        ("custom.custom_2", context.string_attribute("42").as_attribute_ref()),
                    ]),
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
                  func.func private @custom_function(%arg0: f64, %arg1: f64, %arg2: f64) -> f64 attributes {custom.custom_1, custom.custom_2 = \"42\", llvm.emit_c_interface, no_inline} {
                    return %arg1 : f64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_func_with_argument_and_result_attributes() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f64_type = context.float64_type();
        module.body().append_operation({
            let mut block = context.block(&[(f64_type, location), (f64_type, location), (f64_type, location)]);
            block.append_operation(r#return(&[block.argument(1).unwrap()], location));
            func(
                "custom_function",
                FuncAttributes {
                    visibility: SymbolVisibility::Private,
                    arguments: vec![
                        f64_type.into(),
                        TypeAndAttributes {
                            r#type: f64_type.as_type_ref(),
                            attributes: Some(HashMap::from([
                                ("custom.there".into(), context.unit_attribute().as_attribute_ref()),
                                ("custom.we".into(), context.string_attribute("are").as_attribute_ref()),
                            ])),
                        },
                        f64_type.into(),
                    ],
                    results: vec![TypeAndAttributes {
                        r#type: f64_type.as_type_ref(),
                        attributes: Some(HashMap::from([(
                            "custom.yes".into(),
                            context.boolean_attribute(false).as_attribute_ref(),
                        )])),
                    }],
                    no_inline: true,
                    llvm_emit_c_interface: true,
                    other_attributes: HashMap::from([
                        ("custom.custom_1", context.unit_attribute().as_attribute_ref()),
                        ("custom.custom_2", context.string_attribute("42").as_attribute_ref()),
                    ]),
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
                  func.func private @custom_function(%arg0: f64, %arg1: f64 {custom.there, custom.we = \"are\"}, %arg2: f64) -> (f64 {custom.yes = false}) attributes {custom.custom_1, custom.custom_2 = \"42\", llvm.emit_c_interface, no_inline} {
                    return %arg1 : f64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_external_func() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let i32_type = context.signless_integer_type(32);
        module.body().append_operation(func(
            "external_function",
            FuncAttributes {
                arguments: vec![i32_type.into(), f32_type.into()],
                results: vec![i32_type.into()],
                visibility: SymbolVisibility::Private,
                ..Default::default()
            },
            context.region(),
            location,
        ));
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func private @external_function(i32, f32) -> i32
                }
            "},
        );
    }

    #[test]
    fn test_func_return_void() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32).as_type_ref();
        let f64_type = context.float64_type().as_type_ref();
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location), (f64_type, location)]);
            block.append_operation(r#return::<ValueRef, _>(&[], location));
            func(
                "return_multiple",
                FuncAttributes {
                    arguments: vec![i32_type.into(), f64_type.into()],
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
                  func.func @return_multiple(%arg0: i32, %arg1: f64) {
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_func_return_multiple_values() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32).as_type_ref();
        let f64_type = context.float64_type().as_type_ref();
        module.body().append_operation({
            let mut block = context.block(&[(i32_type, location), (f64_type, location)]);
            let return_op = r#return(&[block.argument(0).unwrap(), block.argument(1).unwrap()], location);
            assert_eq!(return_op.values().collect::<Vec<_>>(), block.arguments().collect::<Vec<_>>());
            block.append_operation(return_op);
            func(
                "return_multiple",
                FuncAttributes {
                    arguments: vec![i32_type.into(), f64_type.into()],
                    results: vec![i32_type.into(), f64_type.into()],
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
                  func.func @return_multiple(%arg0: i32, %arg1: f64) -> (i32, f64) {
                    return %arg0, %arg1 : i32, f64
                  }
                }
            "},
        );
    }
}
