use std::collections::HashMap;

use crate::{
    Attribute, BlockRef, CALLEE_ATTRIBUTE, Call, Callee, DenseInteger32ArrayAttributeRef, DetachedOp, DetachedRegion,
    DialectHandle, FUNCTION_TYPE_ATTRIBUTE, FlatSymbolRefAttributeRef, Function,
    HasCallableArgumentAndResultAttributes, IntoWithContext, Location, Operation, OperationBuilder, RegionRef,
    SYMBOL_NAME_ATTRIBUTE, SYMBOL_VISIBILITY_ATTRIBUTE, StringAttributeRef, StringRef, SymbolVisibility, Type,
    TypeAndAttributes, TypeRef, Value, ValueAndAttributes, ValueRef, mlir_op, mlir_op_trait,
};

use super::ValueTypeRef;

/// Name of the attribute storing operand counts for variadic `async.execute` operand groups.
pub const OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operandSegmentSizes";

/// Name of the attribute storing reference-count deltas on async runtime reference operations.
pub const COUNT_ATTRIBUTE: &str = "count";

/// Operation that starts an asynchronous region once all token dependencies and async value operands are ready.
pub trait ExecuteOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the async token dependencies that gate execution of the body region.
    fn dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let dependency_count = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().next().unwrap())
            .unwrap_or_else(|| panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `async.execute`"));
        self.operand_values().take(dependency_count as usize).collect()
    }

    /// Returns the async body operands unwrapped as block arguments in the body region.
    fn body_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_else(|| panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `async.execute`"));
        self.operand_values().skip(sizes[0] as usize).take(sizes[1] as usize).collect()
    }

    /// Returns the completion token result.
    fn token(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns async value results yielded by the body region.
    fn body_results(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.results().skip(1).map(|result| result.as_ref()).collect()
    }

    /// Returns the asynchronous body region.
    fn body(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }
}

mlir_op!(Execute);
mlir_op_trait!(Execute, AutomaticAllocationScope);
mlir_op_trait!(Execute, OneRegion);
mlir_op_trait!(Execute, ZeroSuccessors);

/// Constructs a new detached [`ExecuteOperation`].
pub fn execute<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    dependencies: &[ValueRef<'v, 'c, 't>],
    body_operands: &[ValueRef<'v, 'c, 't>],
    body_result_types: &[TypeRef<'c, 't>],
    body: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedExecuteOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    let operand_segment_sizes = [dependencies.len() as i32, body_operands.len() as i32];
    let mut result_types = vec![context.async_token_type().as_ref()];
    result_types.extend(body_result_types.iter().map(|r#type| context.async_value_type(*r#type).as_ref()));
    OperationBuilder::new("async.execute", location)
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context.dense_i32_array_attribute(&operand_segment_sizes).unwrap(),
        )
        .add_operands(dependencies)
        .add_operands(body_operands)
        .add_results(&result_types)
        .add_region(body)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::execute`")
}

/// Async function operation that supports non-blocking awaits.
pub trait FuncOperation<'o, 'c: 'o, 't: 'c>: Function<'o, 'c, 't> {
    /// Returns the function body region.
    fn body(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }
}

mlir_op!(Func);
mlir_op_trait!(Func, Callable);
mlir_op_trait!(Func, Function);
mlir_op_trait!(Func, HasCallableArgumentAndResultAttributes);
mlir_op_trait!(Func, IsolatedFromAbove);
mlir_op_trait!(Func, OneRegion);
mlir_op_trait!(Func, Symbol);
mlir_op_trait!(Func, ZeroSuccessors);

/// Structured attributes used to construct an [`FuncOperation`].
pub struct FuncAttributes<'c, 't, 's> {
    /// Function argument types and optional argument attributes.
    pub arguments: Vec<TypeAndAttributes<'c, 't, 's>>,

    /// Async function result types and optional result attributes.
    pub results: Vec<TypeAndAttributes<'c, 't, 's>>,

    /// Symbol visibility of the function.
    pub visibility: SymbolVisibility,

    /// Additional custom attributes attached to the function.
    pub other_attributes: HashMap<&'c str, crate::AttributeRef<'c, 't>>,
}

impl Default for FuncAttributes<'_, '_, '_> {
    fn default() -> Self {
        Self {
            arguments: Vec::new(),
            results: Vec::new(),
            visibility: SymbolVisibility::Public,
            other_attributes: HashMap::new(),
        }
    }
}

/// Constructs a new detached [`FuncOperation`].
pub fn func<'c, 't: 'c, 's, N: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>, L: Location<'c, 't>>(
    name: N,
    attributes: FuncAttributes<'c, 't, 's>,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedFuncOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    let mut builder = OperationBuilder::new("async.func", location)
        .add_attribute(SYMBOL_NAME_ATTRIBUTE, name.into_with_context(context))
        .add_attribute(
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
    for (attribute_name, attribute) in &attributes.other_attributes {
        builder = builder.add_attribute(*attribute_name, *attribute);
    }
    builder
        .add_region(body)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::func`")
}

/// Direct call to an async function in the same symbol scope.
pub trait CallOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the callee symbol name.
    fn function(&self) -> StringRef<'c> {
        self.attribute(CALLEE_ATTRIBUTE).unwrap().cast::<FlatSymbolRefAttributeRef>().unwrap().reference()
    }

    /// Returns the call argument operands.
    fn arguments(&self) -> impl Iterator<Item = ValueRef<'o, 'c, 't>> {
        self.operand_values()
    }
}

mlir_op!(Call);
mlir_op_trait!(Call, HasCallableArgumentAndResultAttributes);
mlir_op_trait!(Call, ZeroRegions);
mlir_op_trait!(Call, ZeroSuccessors);

impl<'o, 'c: 'o, 't: 'c> Call<'o, 'c, 't> for DetachedCallOperation<'c, 't> {
    fn callee(&self) -> Callee<'o, 'c, 't> {
        Callee::Symbol(self.attribute(CALLEE_ATTRIBUTE).unwrap().cast::<FlatSymbolRefAttributeRef>().unwrap())
    }
}

impl<'o, 'c: 'o, 't: 'c> Call<'o, 'c, 't> for CallOperationRef<'o, 'c, 't> {
    fn callee(&self) -> Callee<'o, 'c, 't> {
        Callee::Symbol(self.attribute(CALLEE_ATTRIBUTE).unwrap().cast::<FlatSymbolRefAttributeRef>().unwrap())
    }
}

/// Structured properties used to construct a [`CallOperation`].
#[derive(Default)]
pub struct CallProperties<'v, 'c, 't, 's> {
    /// Call argument values and optional argument attributes.
    pub arguments: Vec<ValueAndAttributes<'v, 'c, 't, 's>>,

    /// Call result types and optional result attributes.
    pub results: Vec<TypeAndAttributes<'c, 't, 's>>,
}

/// Constructs a new detached [`CallOperation`].
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
    context.load_dialect(DialectHandle::r#async());
    let mut builder = OperationBuilder::new("async.call", location)
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
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::call`")
}

/// Return operation for [`FuncOperation`].
pub trait ReturnOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns values returned from the async function body.
    fn values(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }
}

mlir_op!(Return);
mlir_op_trait!(Return, AlwaysSpeculatable);
mlir_op_trait!(Return, NoMemoryEffect);
mlir_op_trait!(Return, Pure);
mlir_op_trait!(Return, ZeroRegions);
mlir_op_trait!(Return, ZeroSuccessors);

/// Constructs a new detached [`ReturnOperation`].
pub fn r#return<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    values: &[V],
    location: L,
) -> DetachedReturnOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    OperationBuilder::new("async.return", location)
        .add_operands(values)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::return`")
}

/// Terminator for an [`ExecuteOperation`] body region.
pub trait YieldOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns values yielded by the execute body region.
    fn values(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }
}

mlir_op!(Yield);
mlir_op_trait!(Yield, AlwaysSpeculatable);
mlir_op_trait!(Yield, NoMemoryEffect);
mlir_op_trait!(Yield, Pure);
mlir_op_trait!(Yield, ZeroRegions);
mlir_op_trait!(Yield, ZeroSuccessors);

/// Constructs a new detached [`YieldOperation`].
pub fn yield_<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    values: &[V],
    location: L,
) -> DetachedYieldOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    OperationBuilder::new("async.yield", location)
        .add_operands(values)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::yield_`")
}

/// Operation that waits for an async token or unwraps an async value.
pub trait AwaitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the awaited async token or value.
    fn operand(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the unwrapped result for `async.value` operands.
    fn value(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(0).map(|result| result.as_ref())
    }
}

mlir_op!(Await);
mlir_op_trait!(Await, ZeroRegions);
mlir_op_trait!(Await, ZeroSuccessors);

/// Constructs a new detached [`AwaitOperation`].
pub fn r#await<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    operand: V,
    location: L,
) -> DetachedAwaitOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    let mut builder = OperationBuilder::new("async.await", location).add_operand(operand);
    if let Some(value_type) = operand.r#type().cast::<ValueTypeRef>() {
        builder = builder.add_result(value_type.value_type());
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::await`")
}

/// Operation that creates an empty async group with a fixed size.
pub trait CreateGroupOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the group size value.
    fn size(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the created async group.
    fn group(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(CreateGroup);
mlir_op_trait!(CreateGroup, AlwaysSpeculatable);
mlir_op_trait!(CreateGroup, NoMemoryEffect);
mlir_op_trait!(CreateGroup, OneOperand);
mlir_op_trait!(CreateGroup, OneResult);
mlir_op_trait!(CreateGroup, Pure);
mlir_op_trait!(CreateGroup, ZeroRegions);
mlir_op_trait!(CreateGroup, ZeroSuccessors);

/// Constructs a new detached [`CreateGroupOperation`].
pub fn create_group<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    size: V,
    location: L,
) -> DetachedCreateGroupOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    OperationBuilder::new("async.create_group", location)
        .add_operand(size)
        .add_result(context.async_group_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::create_group`")
}

/// Operation that adds an async token or value to an async group.
pub trait AddToGroupOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the async token or value added to the group.
    fn operand(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the target async group.
    fn group(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the fixed rank of the added element in the group.
    fn rank(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(AddToGroup);
mlir_op_trait!(AddToGroup, OneResult);
mlir_op_trait!(AddToGroup, ZeroRegions);
mlir_op_trait!(AddToGroup, ZeroSuccessors);

/// Constructs a new detached [`AddToGroupOperation`].
pub fn add_to_group<
    'operand,
    'group,
    'c: 'operand + 'group,
    't: 'c,
    Operand: Value<'operand, 'c, 't>,
    Group: Value<'group, 'c, 't>,
    L: Location<'c, 't>,
>(
    operand: Operand,
    group: Group,
    location: L,
) -> DetachedAddToGroupOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    OperationBuilder::new("async.add_to_group", location)
        .add_operand(operand)
        .add_operand(group)
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::add_to_group`")
}

/// Operation that waits for all elements in an async group to become ready.
pub trait AwaitAllOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the awaited async group.
    fn operand(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }
}

mlir_op!(AwaitAll);
mlir_op_trait!(AwaitAll, ZeroRegions);
mlir_op_trait!(AwaitAll, ZeroSuccessors);

/// Constructs a new detached [`AwaitAllOperation`].
pub fn await_all<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    operand: V,
    location: L,
) -> DetachedAwaitAllOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    OperationBuilder::new("async.await_all", location)
        .add_operand(operand)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::await_all`")
}

/// Operation that creates a switched-resume coroutine identifier.
pub trait CoroIdOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the coroutine identifier.
    fn id(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(CoroId);
mlir_op_trait!(CoroId, OneResult);
mlir_op_trait!(CoroId, ZeroOperands);
mlir_op_trait!(CoroId, ZeroRegions);
mlir_op_trait!(CoroId, ZeroSuccessors);

/// Constructs a new detached [`CoroIdOperation`].
pub fn coro_id<'c, 't: 'c, L: Location<'c, 't>>(location: L) -> DetachedCoroIdOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    OperationBuilder::new("async.coro.id", location)
        .add_result(context.async_coro_id_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::coro_id`")
}

/// Operation that begins a coroutine frame and returns a coroutine handle.
pub trait CoroBeginOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the coroutine identifier operand.
    fn id(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the coroutine handle.
    fn handle(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(CoroBegin);
mlir_op_trait!(CoroBegin, OneOperand);
mlir_op_trait!(CoroBegin, OneResult);
mlir_op_trait!(CoroBegin, ZeroRegions);
mlir_op_trait!(CoroBegin, ZeroSuccessors);

/// Constructs a new detached [`CoroBeginOperation`].
pub fn coro_begin<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    id: V,
    location: L,
) -> DetachedCoroBeginOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    OperationBuilder::new("async.coro.begin", location)
        .add_operand(id)
        .add_result(context.async_coro_handle_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::coro_begin`")
}

/// Operation that deallocates a coroutine frame.
pub trait CoroFreeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the coroutine identifier operand.
    fn id(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the coroutine handle operand.
    fn handle(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }
}

mlir_op!(CoroFree);
mlir_op_trait!(CoroFree, ZeroRegions);
mlir_op_trait!(CoroFree, ZeroSuccessors);

/// Constructs a new detached [`CoroFreeOperation`].
pub fn coro_free<
    'id,
    'handle,
    'c: 'id + 'handle,
    't: 'c,
    Id: Value<'id, 'c, 't>,
    Handle: Value<'handle, 'c, 't>,
    L: Location<'c, 't>,
>(
    id: Id,
    handle: Handle,
    location: L,
) -> DetachedCoroFreeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    OperationBuilder::new("async.coro.free", location)
        .add_operand(id)
        .add_operand(handle)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::coro_free`")
}

/// Operation that marks coroutine completion in a suspend block.
pub trait CoroEndOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the coroutine handle operand.
    fn handle(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }
}

mlir_op!(CoroEnd);
mlir_op_trait!(CoroEnd, ZeroRegions);
mlir_op_trait!(CoroEnd, ZeroSuccessors);

/// Constructs a new detached [`CoroEndOperation`].
pub fn coro_end<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    handle: V,
    location: L,
) -> DetachedCoroEndOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    OperationBuilder::new("async.coro.end", location)
        .add_operand(handle)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::coro_end`")
}

/// Operation that saves a coroutine suspension state.
pub trait CoroSaveOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the coroutine handle operand.
    fn handle(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the saved coroutine state.
    fn state(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(CoroSave);
mlir_op_trait!(CoroSave, OneOperand);
mlir_op_trait!(CoroSave, OneResult);
mlir_op_trait!(CoroSave, ZeroRegions);
mlir_op_trait!(CoroSave, ZeroSuccessors);

/// Constructs a new detached [`CoroSaveOperation`].
pub fn coro_save<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    handle: V,
    location: L,
) -> DetachedCoroSaveOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    OperationBuilder::new("async.coro.save", location)
        .add_operand(handle)
        .add_result(context.async_coro_state_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::coro_save`")
}

/// Terminator that suspends a coroutine and branches to suspend, resume, or cleanup successors.
pub trait CoroSuspendOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the saved coroutine state operand.
    fn state(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the successor reached when the coroutine suspends.
    fn suspend_destination(&self) -> BlockRef<'o, 'c, 't> {
        self.successor(0).unwrap()
    }

    /// Returns the successor reached when the coroutine resumes.
    fn resume_destination(&self) -> BlockRef<'o, 'c, 't> {
        self.successor(1).unwrap()
    }

    /// Returns the successor reached when the coroutine is destroyed.
    fn cleanup_destination(&self) -> BlockRef<'o, 'c, 't> {
        self.successor(2).unwrap()
    }
}

mlir_op!(CoroSuspend);
mlir_op_trait!(CoroSuspend, ZeroRegions);

/// Constructs a new detached [`CoroSuspendOperation`].
pub fn coro_suspend<
    'state,
    'suspend,
    'resume,
    'cleanup,
    'c: 'state + 'suspend + 'resume + 'cleanup,
    't: 'c,
    State: Value<'state, 'c, 't>,
    Suspend: crate::Block<'suspend, 'c, 't>,
    Resume: crate::Block<'resume, 'c, 't>,
    Cleanup: crate::Block<'cleanup, 'c, 't>,
    L: Location<'c, 't>,
>(
    state: State,
    suspend_destination: &Suspend,
    resume_destination: &Resume,
    cleanup_destination: &Cleanup,
    location: L,
) -> DetachedCoroSuspendOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    OperationBuilder::new("async.coro.suspend", location)
        .add_operand(state)
        .add_successor(suspend_destination)
        .add_successor(resume_destination)
        .add_successor(cleanup_destination)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::coro_suspend`")
}

/// Runtime operation that creates an async token or value in the unavailable state.
pub trait RuntimeCreateOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the created async runtime value.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(RuntimeCreate);
mlir_op_trait!(RuntimeCreate, OneResult);
mlir_op_trait!(RuntimeCreate, ZeroOperands);
mlir_op_trait!(RuntimeCreate, ZeroRegions);
mlir_op_trait!(RuntimeCreate, ZeroSuccessors);

/// Constructs a new detached [`RuntimeCreateOperation`].
pub fn runtime_create<'c, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    result_type: T,
    location: L,
) -> DetachedRuntimeCreateOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    OperationBuilder::new("async.runtime.create", location)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::runtime_create`")
}

/// Runtime operation that creates an async group.
pub trait RuntimeCreateGroupOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the group size value.
    fn size(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the created async group.
    fn group(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(RuntimeCreateGroup);
mlir_op_trait!(RuntimeCreateGroup, OneOperand);
mlir_op_trait!(RuntimeCreateGroup, OneResult);
mlir_op_trait!(RuntimeCreateGroup, ZeroRegions);
mlir_op_trait!(RuntimeCreateGroup, ZeroSuccessors);

/// Constructs a new detached [`RuntimeCreateGroupOperation`].
pub fn runtime_create_group<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    size: V,
    location: L,
) -> DetachedRuntimeCreateGroupOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    OperationBuilder::new("async.runtime.create_group", location)
        .add_operand(size)
        .add_result(context.async_group_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::runtime_create_group`")
}

macro_rules! async_runtime_unary_op {
    ($name:ident, $constructor:ident, $operation_name:literal, $description:literal $(,)*) => {
        paste::paste! {
            #[doc = $description]
            pub trait [<$name Operation>]<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
                /// Returns the async runtime operand.
                fn operand(&self) -> ValueRef<'o, 'c, 't> {
                    self.operand_value(0).unwrap()
                }
            }

            mlir_op!($name);
            mlir_op_trait!($name, ZeroRegions);
            mlir_op_trait!($name, ZeroSuccessors);

            #[doc = "Constructs a new detached `"]
            #[doc = $operation_name]
            #[doc = "` operation."]
            pub fn $constructor<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
                operand: V,
                location: L,
            ) -> [<Detached $name Operation>]<'c, 't> {
                let context = location.context();
                context.load_dialect(DialectHandle::r#async());
                OperationBuilder::new($operation_name, location)
                    .add_operand(operand)
                    .build()
                    .and_then(|operation| unsafe { operation.cast() })
                    .expect(concat!("invalid arguments to `async::", stringify!($constructor), "`"))
            }
        }
    };
}

async_runtime_unary_op!(
    RuntimeSetAvailable,
    runtime_set_available,
    "async.runtime.set_available",
    "Runtime operation that marks an async token or value as available.",
);

async_runtime_unary_op!(
    RuntimeSetError,
    runtime_set_error,
    "async.runtime.set_error",
    "Runtime operation that marks an async token or value as failed.",
);

async_runtime_unary_op!(
    RuntimeAwait,
    runtime_await,
    "async.runtime.await",
    "Runtime operation that blocks until an async token, value, or group becomes available.",
);

/// Runtime operation that checks whether an async token, value, or group is in the error state.
pub trait RuntimeIsErrorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the checked async runtime operand.
    fn operand(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the boolean-like error-state result.
    fn is_error(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(RuntimeIsError);
mlir_op_trait!(RuntimeIsError, OneOperand);
mlir_op_trait!(RuntimeIsError, OneResult);
mlir_op_trait!(RuntimeIsError, ZeroRegions);
mlir_op_trait!(RuntimeIsError, ZeroSuccessors);

/// Constructs a new detached [`RuntimeIsErrorOperation`].
pub fn runtime_is_error<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    operand: V,
    location: L,
) -> DetachedRuntimeIsErrorOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    OperationBuilder::new("async.runtime.is_error", location)
        .add_operand(operand)
        .add_result(context.signless_integer_type(1))
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::runtime_is_error`")
}

/// Runtime operation that resumes a coroutine.
pub trait RuntimeResumeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the coroutine handle operand.
    fn handle(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }
}

mlir_op!(RuntimeResume);
mlir_op_trait!(RuntimeResume, ZeroRegions);
mlir_op_trait!(RuntimeResume, ZeroSuccessors);

/// Constructs a new detached [`RuntimeResumeOperation`].
pub fn runtime_resume<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    handle: V,
    location: L,
) -> DetachedRuntimeResumeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    OperationBuilder::new("async.runtime.resume", location)
        .add_operand(handle)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::runtime_resume`")
}

/// Runtime operation that awaits an async runtime value and resumes a coroutine.
pub trait RuntimeAwaitAndResumeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the awaited async runtime operand.
    fn operand(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the coroutine handle operand.
    fn handle(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }
}

mlir_op!(RuntimeAwaitAndResume);
mlir_op_trait!(RuntimeAwaitAndResume, ZeroRegions);
mlir_op_trait!(RuntimeAwaitAndResume, ZeroSuccessors);

/// Constructs a new detached [`RuntimeAwaitAndResumeOperation`].
pub fn runtime_await_and_resume<
    'operand,
    'handle,
    'c: 'operand + 'handle,
    't: 'c,
    Operand: Value<'operand, 'c, 't>,
    Handle: Value<'handle, 'c, 't>,
    L: Location<'c, 't>,
>(
    operand: Operand,
    handle: Handle,
    location: L,
) -> DetachedRuntimeAwaitAndResumeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    OperationBuilder::new("async.runtime.await_and_resume", location)
        .add_operand(operand)
        .add_operand(handle)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::runtime_await_and_resume`")
}

/// Runtime operation that stores an available value into async value storage.
pub trait RuntimeStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the stored value.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the async value storage.
    fn storage(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }
}

mlir_op!(RuntimeStore);
mlir_op_trait!(RuntimeStore, ZeroRegions);
mlir_op_trait!(RuntimeStore, ZeroSuccessors);

/// Constructs a new detached [`RuntimeStoreOperation`].
pub fn runtime_store<
    'value,
    'storage,
    'c: 'value + 'storage,
    't: 'c,
    StoredValue: Value<'value, 'c, 't>,
    Storage: Value<'storage, 'c, 't>,
    L: Location<'c, 't>,
>(
    value: StoredValue,
    storage: Storage,
    location: L,
) -> DetachedRuntimeStoreOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    OperationBuilder::new("async.runtime.store", location)
        .add_operand(value)
        .add_operand(storage)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::runtime_store`")
}

/// Runtime operation that loads an available value from async value storage.
pub trait RuntimeLoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the async value storage.
    fn storage(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the loaded value.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(RuntimeLoad);
mlir_op_trait!(RuntimeLoad, OneOperand);
mlir_op_trait!(RuntimeLoad, OneResult);
mlir_op_trait!(RuntimeLoad, ZeroRegions);
mlir_op_trait!(RuntimeLoad, ZeroSuccessors);

/// Constructs a new detached [`RuntimeLoadOperation`].
pub fn runtime_load<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    storage: V,
    location: L,
) -> DetachedRuntimeLoadOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    let result_type = storage
        .r#type()
        .cast::<ValueTypeRef>()
        .map(|r#type| r#type.value_type())
        .expect("`async.runtime.load` storage must have `!async.value` type");
    OperationBuilder::new("async.runtime.load", location)
        .add_operand(storage)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::runtime_load`")
}

/// Runtime operation that adds an async token or value to a runtime group.
pub trait RuntimeAddToGroupOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the async token or value operand.
    fn operand(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the target async group.
    fn group(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the rank assigned to the added element.
    fn rank(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(RuntimeAddToGroup);
mlir_op_trait!(RuntimeAddToGroup, OneResult);
mlir_op_trait!(RuntimeAddToGroup, ZeroRegions);
mlir_op_trait!(RuntimeAddToGroup, ZeroSuccessors);

/// Constructs a new detached [`RuntimeAddToGroupOperation`].
pub fn runtime_add_to_group<
    'operand,
    'group,
    'c: 'operand + 'group,
    't: 'c,
    Operand: Value<'operand, 'c, 't>,
    Group: Value<'group, 'c, 't>,
    L: Location<'c, 't>,
>(
    operand: Operand,
    group: Group,
    location: L,
) -> DetachedRuntimeAddToGroupOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    OperationBuilder::new("async.runtime.add_to_group", location)
        .add_operand(operand)
        .add_operand(group)
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::runtime_add_to_group`")
}

/// Runtime operation that increments an async runtime value reference count.
pub trait RuntimeAddRefOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the reference-counted operand.
    fn operand(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the reference-count increment.
    fn count(&self) -> i64 {
        self.attribute(COUNT_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<crate::IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value())
            .unwrap_or_else(|| panic!("invalid '{COUNT_ATTRIBUTE}' attribute in `async.runtime.add_ref`"))
    }
}

mlir_op!(RuntimeAddRef);
mlir_op_trait!(RuntimeAddRef, ZeroRegions);
mlir_op_trait!(RuntimeAddRef, ZeroSuccessors);

/// Constructs a new detached [`RuntimeAddRefOperation`].
pub fn runtime_add_ref<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    operand: V,
    count: i64,
    location: L,
) -> DetachedRuntimeAddRefOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    OperationBuilder::new("async.runtime.add_ref", location)
        .add_operand(operand)
        .add_attribute(COUNT_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(64), count))
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::runtime_add_ref`")
}

/// Runtime operation that decrements an async runtime value reference count.
pub trait RuntimeDropRefOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the reference-counted operand.
    fn operand(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the reference-count decrement.
    fn count(&self) -> i64 {
        self.attribute(COUNT_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<crate::IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value())
            .unwrap_or_else(|| panic!("invalid '{COUNT_ATTRIBUTE}' attribute in `async.runtime.drop_ref`"))
    }
}

mlir_op!(RuntimeDropRef);
mlir_op_trait!(RuntimeDropRef, ZeroRegions);
mlir_op_trait!(RuntimeDropRef, ZeroSuccessors);

/// Constructs a new detached [`RuntimeDropRefOperation`].
pub fn runtime_drop_ref<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    operand: V,
    count: i64,
    location: L,
) -> DetachedRuntimeDropRefOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    OperationBuilder::new("async.runtime.drop_ref", location)
        .add_operand(operand)
        .add_attribute(COUNT_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(64), count))
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::runtime_drop_ref`")
}

/// Runtime operation that returns the number of async runtime worker threads.
pub trait RuntimeNumWorkerThreadsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the worker-thread count result.
    fn count(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(RuntimeNumWorkerThreads);
mlir_op_trait!(RuntimeNumWorkerThreads, OneResult);
mlir_op_trait!(RuntimeNumWorkerThreads, ZeroOperands);
mlir_op_trait!(RuntimeNumWorkerThreads, ZeroRegions);
mlir_op_trait!(RuntimeNumWorkerThreads, ZeroSuccessors);

/// Constructs a new detached [`RuntimeNumWorkerThreadsOperation`].
pub fn runtime_num_worker_threads<'c, 't: 'c, L: Location<'c, 't>>(
    location: L,
) -> DetachedRuntimeNumWorkerThreadsOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::r#async());
    OperationBuilder::new("async.runtime.num_worker_threads", location)
        .add_result(context.index_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `async::runtime_num_worker_threads`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::{func as func_dialect, index};
    use crate::{Block, Context, IntoWithContext, Operation, Type};

    use super::*;

    #[test]
    fn test_func_call_await_and_return() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let index_type = context.index_type();
        let async_index_type = context.async_value_type(index_type);

        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let constant = block.append_operation(index::constant(7, location));
            let return_op = r#return(&[constant.result(0).unwrap()], location);
            assert_eq!(return_op.values(), vec![constant.result(0).unwrap().as_ref()]);
            block.append_operation(return_op);
            func(
                "producer",
                FuncAttributes { results: vec![async_index_type.into()], ..Default::default() },
                block.into(),
                location,
            )
        });

        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let call_op = call(
                "producer",
                CallProperties { results: vec![async_index_type.into()], ..Default::default() },
                location,
            );
            assert_eq!(call_op.function().as_str().unwrap(), "producer");
            assert_eq!(call_op.arguments().collect::<Vec<_>>(), Vec::<ValueRef>::new());
            assert_eq!(call_op.result_type(0).unwrap(), async_index_type);
            let call_op = block.append_operation(call_op);
            let await_op = r#await(call_op.result(0).unwrap(), location);
            assert_eq!(AwaitOperation::operand(&await_op), call_op.result(0).unwrap().as_ref());
            assert_eq!(await_op.result_type(0).unwrap(), index_type);
            let await_op = block.append_operation(await_op);
            block.append_operation(r#return(&[await_op.result(0).unwrap()], location));
            func(
                "caller",
                FuncAttributes { results: vec![async_index_type.into()], ..Default::default() },
                block.into(),
                location,
            )
        });

        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  async.func @producer() -> !async.value<index> {
                    %idx7 = index.constant 7
                    return %idx7 : index
                  }
                  async.func @caller() -> !async.value<index> {
                    %0 = call @producer() : () -> !async.value<index>
                    %1 = await %0 : !async.value<index>
                    return %1 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_execute_create_group_add_to_group_and_await_all() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let index_type = context.index_type();
        let token_type = context.async_token_type();
        let async_index_type = context.async_value_type(index_type);

        module.body().append_operation({
            let mut block = context.block(&[(token_type.as_ref(), location), (async_index_type.as_ref(), location)]);
            let dependency = block.argument(0).unwrap();
            let body_operand = block.argument(1).unwrap();
            let mut execute_body = context.block(&[(index_type, location)]);
            let body_argument = execute_body.argument(0).unwrap();
            let yield_op = yield_(&[body_argument], location);
            assert_eq!(yield_op.values(), vec![body_argument.as_ref()]);
            execute_body.append_operation(yield_op);
            let execute_op = execute(
                &[dependency.as_ref()],
                &[body_operand.as_ref()],
                &[index_type.as_ref()],
                execute_body.into(),
                location,
            );
            assert_eq!(execute_op.dependencies(), vec![dependency.as_ref()]);
            assert_eq!(execute_op.body_operands(), vec![body_operand.as_ref()]);
            assert_eq!(execute_op.result_type(0).unwrap(), token_type);
            assert_eq!(execute_op.result_type(1).unwrap(), async_index_type);
            let execute_op = block.append_operation(execute_op);
            block.append_operation(func_dialect::r#return(&[execute_op.result(1).unwrap()], location));
            func_dialect::func(
                "execute_test",
                func_dialect::FuncAttributes {
                    arguments: vec![token_type.into(), async_index_type.into()],
                    results: vec![async_index_type.into()],
                    ..Default::default()
                },
                block.into(),
                location,
            )
        });

        module.body().append_operation({
            let mut block = context.block(&[(index_type.as_ref(), location), (token_type.as_ref(), location)]);
            let size = block.argument(0).unwrap();
            let token = block.argument(1).unwrap();
            let create_group_op = create_group(size, location);
            assert_eq!(create_group_op.size(), size.as_ref());
            assert_eq!(create_group_op.result_type(0).unwrap(), context.async_group_type());
            let group = block.append_operation(create_group_op).result(0).unwrap();
            let add_to_group_op = add_to_group(token, group, location);
            assert_eq!(AddToGroupOperation::operand(&add_to_group_op), token.as_ref());
            assert_eq!(add_to_group_op.group(), group.as_ref());
            assert_eq!(add_to_group_op.result_type(0).unwrap(), index_type);
            let rank = block.append_operation(add_to_group_op).result(0).unwrap();
            let await_all_op = await_all(group, location);
            assert_eq!(AwaitAllOperation::operand(&await_all_op), group.as_ref());
            block.append_operation(await_all_op);
            block.append_operation(func_dialect::r#return(&[rank], location));
            func_dialect::func(
                "group_test",
                func_dialect::FuncAttributes {
                    arguments: vec![index_type.into(), token_type.into()],
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
                  func.func @execute_test(%arg0: !async.token, %arg1: !async.value<index>) -> !async.value<index> {
                    %token, %bodyResults = async.execute [%arg0] (%arg1 as %arg2: !async.value<index>) -> !async.value<index> {
                      async.yield %arg2 : index
                    }
                    return %bodyResults : !async.value<index>
                  }
                  func.func @group_test(%arg0: index, %arg1: !async.token) -> index {
                    %0 = async.create_group %arg0 : !async.group
                    %1 = async.add_to_group %arg1, %0 : !async.token
                    async.await_all %0
                    return %1 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_runtime_operations() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let index_type = context.index_type();
        let token_type = context.async_token_type();
        let async_index_type = context.async_value_type(index_type);
        let handle_type = context.async_coro_handle_type();

        module.body().append_operation({
            let mut block = context.block(&[(index_type.as_ref(), location), (handle_type.as_ref(), location)]);
            let input = block.argument(0).unwrap();
            let handle = block.argument(1).unwrap();
            let created_token = block.append_operation(runtime_create(token_type, location)).result(0).unwrap();
            let storage = block.append_operation(runtime_create(async_index_type, location)).result(0).unwrap();
            block.append_operation(runtime_store(input, storage, location));
            let loaded = block.append_operation(runtime_load(storage, location)).result(0).unwrap();
            let group = block.append_operation(runtime_create_group(input, location)).result(0).unwrap();
            let add_to_group_op = runtime_add_to_group(created_token, group, location);
            assert_eq!(RuntimeAddToGroupOperation::operand(&add_to_group_op), created_token.as_ref());
            assert_eq!(add_to_group_op.group(), group.as_ref());
            block.append_operation(add_to_group_op);
            let add_ref_op = runtime_add_ref(created_token, 1, location);
            assert_eq!(add_ref_op.count(), 1);
            block.append_operation(add_ref_op);
            let drop_ref_op = runtime_drop_ref(created_token, 1, location);
            assert_eq!(drop_ref_op.count(), 1);
            block.append_operation(drop_ref_op);
            block.append_operation(runtime_set_available(created_token, location));
            block.append_operation(runtime_set_error(created_token, location));
            block.append_operation(runtime_is_error(created_token, location));
            block.append_operation(runtime_await(created_token, location));
            block.append_operation(runtime_resume(handle, location));
            block.append_operation(runtime_await_and_resume(created_token, handle, location));
            block.append_operation(runtime_num_worker_threads(location));
            block.append_operation(func_dialect::r#return(&[loaded], location));
            func_dialect::func(
                "runtime_test",
                func_dialect::FuncAttributes {
                    arguments: vec![index_type.into(), handle_type.into()],
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
                  func.func @runtime_test(%arg0: index, %arg1: !async.coro.handle) -> index {
                    %0 = async.runtime.create : !async.token
                    %1 = async.runtime.create : !async.value<index>
                    async.runtime.store %arg0, %1 : <index>
                    %2 = async.runtime.load %1 : <index>
                    %3 = async.runtime.create_group %arg0 : !async.group
                    %4 = async.runtime.add_to_group %0, %3 : !async.token
                    async.runtime.add_ref %0 {count = 1 : i64} : !async.token
                    async.runtime.drop_ref %0 {count = 1 : i64} : !async.token
                    async.runtime.set_available %0 : !async.token
                    async.runtime.set_error %0 : !async.token
                    %5 = async.runtime.is_error %0 : !async.token
                    async.runtime.await %0 : !async.token
                    async.runtime.resume %arg1
                    async.runtime.await_and_resume %0, %arg1 : !async.token
                    %6 = async.runtime.num_worker_threads : index
                    return %2 : index
                  }
                }
            "},
        );
    }

    #[test]
    fn test_coroutine_operations() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);

        module.body().append_operation({
            let mut entry_block = context.block_with_no_arguments();
            let id = entry_block.append_operation(coro_id(location)).result(0).unwrap();
            let begin_op = coro_begin(id, location);
            assert_eq!(begin_op.id(), id.as_ref());
            let handle = entry_block.append_operation(begin_op).result(0).unwrap();
            let save_op = coro_save(handle, location);
            assert_eq!(save_op.handle(), handle.as_ref());
            let state = entry_block.append_operation(save_op).result(0).unwrap();

            let mut suspend_block = context.block_with_no_arguments();
            suspend_block.append_operation(coro_end(handle, location));
            suspend_block.append_operation(func_dialect::r#return::<ValueRef, _>(&[], location));

            let mut resume_block = context.block_with_no_arguments();
            resume_block.append_operation(coro_free(id, handle, location));
            resume_block.append_operation(func_dialect::r#return::<ValueRef, _>(&[], location));

            let mut cleanup_block = context.block_with_no_arguments();
            cleanup_block.append_operation(func_dialect::r#return::<ValueRef, _>(&[], location));

            let suspend_op = coro_suspend(state, &suspend_block, &resume_block, &cleanup_block, location);
            assert_eq!(suspend_op.state(), state.as_ref());
            assert_eq!(suspend_op.suspend_destination(), suspend_block);
            assert_eq!(suspend_op.resume_destination(), resume_block);
            assert_eq!(suspend_op.cleanup_destination(), cleanup_block);
            entry_block.append_operation(suspend_op);

            func_dialect::func(
                "coro_test",
                func_dialect::FuncAttributes::default(),
                vec![entry_block, suspend_block, resume_block, cleanup_block].into_with_context(&context),
                location,
            )
        });

        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @coro_test() {
                    %0 = async.coro.id
                    %1 = async.coro.begin %0
                    %2 = async.coro.save %1
                    async.coro.suspend %2, ^bb1, ^bb2, ^bb3
                  ^bb1:  // pred: ^bb0
                    async.coro.end %1
                    return
                  ^bb2:  // pred: ^bb0
                    async.coro.free %0, %1
                    return
                  ^bb3:  // pred: ^bb0
                    return
                  }
                }
            "},
        );
    }
}
