use std::cell::Ref;
use std::collections::HashMap;
use std::fmt::Display;
use std::marker::PhantomData;

use ryft_xla_sys::bindings::{
    MlirContext, MlirOpOperand, MlirValue, mlirBlockArgumentGetArgNumber, mlirBlockArgumentGetOwner,
    mlirOpOperandGetNextUse, mlirOpOperandGetOperandNumber, mlirOpOperandGetOwner, mlirOpOperandGetValue,
    mlirOpOperandIsNull, mlirOpResultGetOwner, mlirOpResultGetResultNumber, mlirValueDump, mlirValueGetFirstUse,
    mlirValueGetLocation, mlirValueGetType, mlirValueIsABlockArgument, mlirValueIsAOpResult, mlirValuePrintAsOperand,
    mlirValueReplaceAllUsesExcept, mlirValueReplaceAllUsesOfWith, mlirValueSetType,
};

use crate::support::write_to_string_callback;
use crate::{
    AsmState, AttributeRef, BlockRef, Context, Location, LocationRef, Operation, OperationPrintingFlags, OperationRef,
    StringRef, Type, TypeRef, mlir_subtype_trait_impls,
};

/// [`Value`]s represent the arguments of [`Block`](crate::Block)s and the results and operands of [`Operation`]s.
/// We only ever manipulate references to [`Value`]s which are owned by the underlying [`Context`] via types like
/// [`ValueRef`], [`BlockArgumentRef`], [`OperationResultRef`], and [`OperandRef`].
///
/// Note that there are multiple separate lifetime parameters: one for the lifetime of the underlying [`Value`], `'v`,
/// one for the [`Context`] which is associated with it, `'c`, and one for the lifetime of the thread pool used by that
/// [`Context`], `'t`. That is because, [`Value`]s are non-owning references to the underlying MLIR values, which
/// themselves are owned by a [`Block`](crate::Block) or an [`Operation`]. These [`Block`](crate::Block)s and
/// [`Operation`]s are themselves in turn owned by the [`Context`] in which they live, but they may be destroyed
/// before the [`Context`] is destroyed. Therefore, we need to keep track of this lifetime in [`Value`].
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/LangRef/#high-level-structure)
/// for more information.
pub trait Value<'v, 'c: 'v, 't: 'c>: Sized + Copy + Clone + PartialEq + Eq + Display {
    /// Constructs a new [`Value`] of this type from the provided [`MlirValue`] handle that came from a function in the
    /// MLIR C API, and a reference to the [`Context`] that owns it.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    unsafe fn from_c_api(handle: MlirValue, context: &'c Context<'t>) -> Option<Self>;

    /// Returns the [`MlirValue`] that corresponds to this [`Value`] and which can be passed to functions
    /// in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    unsafe fn to_c_api(&self) -> MlirValue;

    /// Returns a reference to the [`Context`] that owns this [`Value`].
    fn context(&self) -> &'c Context<'t>;

    /// Returns `true` if this [`Value`] is an instance of `V`.
    fn is<V: Value<'v, 'c, 't>>(&self) -> bool {
        Self::cast::<V>(self).is_some()
    }

    /// Tries to cast this [`Value`] to an instance of `V` (e.g., an instance of [`OperationResultRef`]).
    /// If it is not an instance of `V`, this function will return [`None`].
    fn cast<V: Value<'v, 'c, 't>>(&self) -> Option<V> {
        unsafe { V::from_c_api(self.to_c_api(), self.context()) }
    }

    /// Up-casts this [`Value`] to an instance of [`ValueRef`] (i.e., the most generic value reference type).
    fn as_ref(&self) -> ValueRef<'v, 'c, 't> {
        self.cast().unwrap()
    }

    /// Returns the name of this [`Value`] (i.e., the ID that would be used to refer to it in MLIR when used as an
    /// operand), if it has one. Values only have names assigned to them if they are owned by some operation.
    /// For example, arguments of [`DetachedBlock`](crate::DetachedBlock)s do not have names.
    ///
    /// Refer to the documentation of [`OperationPrintingFlags`] for more information on the arguments.
    fn name(
        &self,
        use_name_location_as_prefix: bool,
        use_local_scope: bool,
    ) -> Result<Option<String>, std::str::Utf8Error> {
        unsafe {
            // It is not always safe to obtain the name of a value. We perform a few checks to ensure that we not get
            // unexpected crashes at runtime stemming from the underlying MLIR library.
            let handle = self.to_c_api();
            let parent_operation = self.cast::<BlockArgumentRef>().and_then(|a| a.block().parent_operation());
            let operation_result = self.cast::<OperationResultRef>();
            if handle.ptr.is_null() || (operation_result.is_none() && parent_operation.is_none()) {
                Ok(None)
            } else {
                // The following context borrow ensures that access to the underlying MLIR data structures is done
                // safely from Rust. It is maybe more conservative than would be ideal, but that is due to the limited
                // exposure to MLIR internals that we have when working with the MLIR C API.
                let _guard = self.context().borrow();
                let flags = OperationPrintingFlags {
                    use_name_location_as_prefix,
                    use_local_scope,
                    ..OperationPrintingFlags::default()
                };
                let asm_state = AsmState::for_value(*self, flags);
                let mut data = (String::new(), Ok(()));
                mlirValuePrintAsOperand(
                    self.to_c_api(),
                    asm_state.to_c_api(),
                    Some(write_to_string_callback),
                    &mut data as *mut _ as *mut std::ffi::c_void,
                );
                data.1.map(|_| Some(data.0))
            }
        }
    }

    /// Returns the [`Location`] of this value.
    fn location(&self) -> LocationRef<'c, 't> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { LocationRef::from_c_api(mlirValueGetLocation(self.to_c_api()), self.context()).unwrap() }
    }

    /// Returns the [`Type`] of this value.
    fn r#type(&self) -> TypeRef<'c, 't> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { TypeRef::from_c_api(mlirValueGetType(self.to_c_api()), self.context()).unwrap() }
    }

    /// Sets the [`Type`] of this value to the provided [`Type`].
    fn set_type<T: Type<'c, 't>>(&self, r#type: T) {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe { mlirValueSetType(self.to_c_api(), r#type.to_c_api()) }
    }

    /// Returns an [`OperandRefIterator`] over all uses of this [`Value`] (i.e., instances where it appears
    /// as an operand in [`Operation`]s).
    fn uses(&self) -> OperandRefIterator<'v, 'c, 't> {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let context = self.context().borrow();
        OperandRefIterator {
            current_operand: unsafe { OperandRef::from_c_api(mlirValueGetFirstUse(self.to_c_api()), self.context()) },
            _context: context,
        }
    }

    /// Replaces all uses of this [`Value`] in the current [`Context`] intermediate representation (IR) with the
    /// provided `replacement` [`Value`]. After this function is called, no more uses of this [`Value`] will exist.
    fn replace_uses<'r, V: Value<'r, 'c, 't>>(&self, replacement: V)
    where
        'c: 'r,
    {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe { mlirValueReplaceAllUsesOfWith(self.to_c_api(), replacement.to_c_api()) }
    }

    /// Replaces all uses of this [`Value`] in the current [`Context`] intermediate representation (IR) with
    /// the provided `replacement` [`Value`], except for instances appearing in the invocations of any of the
    /// [`Operation`]s in `exceptions`.
    fn replace_uses_except<'r, 'o, V: Value<'r, 'c, 't>, O: Operation<'o, 'c, 't>>(
        &self,
        replacement: V,
        exceptions: &[&O],
    ) where
        'c: 'r + 'o,
    {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow_mut();
        unsafe {
            let exceptions = exceptions.iter().map(|exception| exception.to_c_api()).collect::<Vec<_>>();
            mlirValueReplaceAllUsesExcept(
                self.to_c_api(),
                replacement.to_c_api(),
                exceptions.len().cast_signed(),
                exceptions.as_ptr() as *mut _,
            )
        }
    }

    /// Dumps this [`Value`] to the standard error stream.
    fn dump(&self) {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context().borrow();
        unsafe { mlirValueDump(self.to_c_api()) }
    }
}

/// Reference to an MLIR [`Value`] that is owned by a [`Block`](crate::Block) or an [`Operation`].
///
/// Note that there are multiple separate lifetime parameters: one for the lifetime of the owner of this [`Value`]
/// reference, `'o`, one for the [`Context`] which is associated with it, `'c`, and one for the lifetime of the thread
/// pool used by that [`Context`], `'t`.
#[derive(Copy, Clone)]
pub struct ValueRef<'o, 'c: 'o, 't: 'c> {
    /// Handle that represents this value reference in the MLIR C API.
    handle: MlirValue,

    /// [`Context`] that owns the underlying [`Value`].
    context: &'c Context<'t>,

    /// [`PhantomData`] used to track the lifetime of the owner of this [`Value`] reference.
    owner: PhantomData<&'o ()>,
}

impl<'v, 'o: 'v, 'c, 't> Value<'v, 'c, 't> for ValueRef<'o, 'c, 't> {
    unsafe fn from_c_api(handle: MlirValue, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() { None } else { Some(Self { handle, context, owner: PhantomData }) }
    }

    unsafe fn to_c_api(&self) -> MlirValue {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(ValueRef<'o, 'c, 't> as Value, mlir_type = Value);

/// Reference to an MLIR [`Value`] that represents a [`Block`](crate::Block) argument.
///
/// Note that there are multiple separate lifetime parameters: one for the lifetime of the [`Block`](crate::Block) that
/// owns this [`Value`] reference, `'b`, one for the [`Context`] which is associated with it, `'c`, and one for the
/// lifetime of the thread pool used by that [`Context`], `'t`.
#[derive(Copy, Clone)]
pub struct BlockArgumentRef<'b, 'c: 'b, 't: 'c> {
    /// Handle that represents this value reference in the MLIR C API.
    handle: MlirValue,

    /// [`Context`] that owns the underlying [`Value`].
    context: &'c Context<'t>,

    /// [`PhantomData`] used to track the lifetime of the [`Block`] that owns this [`Value`] reference.
    owner: PhantomData<&'b ()>,
}

impl<'b, 'c, 't> BlockArgumentRef<'b, 'c, 't> {
    /// Returns a reference to the [`Block`](crate::Block) in which this value is defined as an argument.
    pub fn block(&self) -> BlockRef<'b, 'c, 't> {
        let _guard = self.context.borrow();
        unsafe { BlockRef::from_c_api(mlirBlockArgumentGetOwner(self.handle), self.context).unwrap() }
    }

    /// Returns the index of this value in the argument list of its owning [`Block`](crate::Block).
    pub fn argument_index(&self) -> usize {
        let _guard = self.context.borrow();
        unsafe { mlirBlockArgumentGetArgNumber(self.handle).cast_unsigned() }
    }
}

impl<'v, 'b: 'v, 'c, 't> Value<'v, 'c, 't> for BlockArgumentRef<'b, 'c, 't> {
    unsafe fn from_c_api(handle: MlirValue, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirValueIsABlockArgument(handle) } {
            Some(Self { handle, context, owner: PhantomData })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirValue {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(BlockArgumentRef<'b, 'c, 't> as Value, mlir_type = Value);

impl<'o, 'c, 't> From<BlockArgumentRef<'o, 'c, 't>> for ValueRef<'o, 'c, 't> {
    fn from(value: BlockArgumentRef<'o, 'c, 't>) -> Self {
        value.as_ref()
    }
}

/// [`Value`] that represents the result of an [`Operation`].
///
/// Note that there are multiple separate lifetime parameters: one for the lifetime of the [`Operation`] that owns this
/// [`Value`] reference, `'o`, one for the [`Context`] which is associated with it, `'c`, and one for the lifetime
/// of the thread pool used by that [`Context`], `'t`.
#[derive(Copy, Clone)]
pub struct OperationResultRef<'o, 'c: 'o, 't: 'c> {
    /// Handle that represents this value reference in the MLIR C API.
    handle: MlirValue,

    /// [`Context`] that owns the underlying [`Value`].
    context: &'c Context<'t>,

    /// [`PhantomData`] used to track the lifetime of the [`Operation`] that owns this [`Value`] reference.
    owner: PhantomData<&'o ()>,
}

impl<'o, 'c, 't> OperationResultRef<'o, 'c, 't> {
    /// Returns a [`OperationRef`] referencing the [`Operation`] that produced this value as its result.
    pub fn operation(&self) -> OperationRef<'o, 'c, 't> {
        let _guard = self.context.borrow();
        unsafe { OperationRef::from_c_api(mlirOpResultGetOwner(self.handle), self.context).unwrap() }
    }

    /// Returns the index of this value in the results of its owning [`Operation`].
    pub fn result_index(&self) -> usize {
        let _guard = self.context.borrow();
        unsafe { mlirOpResultGetResultNumber(self.handle).cast_unsigned() }
    }
}

impl<'o, 'c, 't> Value<'o, 'c, 't> for OperationResultRef<'o, 'c, 't> {
    unsafe fn from_c_api(handle: MlirValue, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { mlirValueIsAOpResult(handle) } {
            Some(Self { handle, context, owner: PhantomData })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirValue {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(OperationResultRef<'o, 'c, 't> as Value, mlir_type = Value);

impl<'o, 'c, 't> From<OperationResultRef<'o, 'c, 't>> for ValueRef<'o, 'c, 't> {
    fn from(value: OperationResultRef<'o, 'c, 't>) -> Self {
        value.as_ref()
    }
}

/// Reference to a "use" of a [`Value`] as an operand for some [`Operation`].
///
/// Note that there are multiple separate lifetime parameters: one for the lifetime of the [`Operation`] that owns this
/// reference, `'o`, one for the [`Context`] which is associated with it, `'c`, and one for the lifetime of the thread
/// pool used by that [`Context`], `'t`.
#[derive(Copy, Clone)]
pub struct OperandRef<'o, 'c: 'o, 't: 'c> {
    /// Handle that represents this operand reference in the MLIR C API.
    handle: MlirOpOperand,

    /// [`Context`] that owns the underlying [`Value`].
    context: &'c Context<'t>,

    /// [`PhantomData`] used to track the lifetime of the [`Operation`] that owns this operand reference.
    owner: PhantomData<&'o ()>,
}

impl<'o, 'c, 't> OperandRef<'o, 'c, 't> {
    /// Constructs a new [`OperandRef`] from the provided [`MlirOpOperand`] handle that came from a function in the
    /// MLIR C API, and a reference to the [`Context`] that owns it.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn from_c_api(handle: MlirOpOperand, context: &'c Context<'t>) -> Option<Self> {
        unsafe { if mlirOpOperandIsNull(handle) { None } else { Some(Self { handle, context, owner: PhantomData }) } }
    }

    /// Returns the [`MlirOpOperand`] that corresponds to this [`OperandRef`] and which can be passed to functions
    /// in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn to_c_api(&self) -> MlirOpOperand {
        self.handle
    }

    /// Returns a reference to the [`Context`] that owns the underlying [`Value`] of this operand.
    pub fn context(&self) -> &'c Context<'t> {
        self.context
    }

    /// Returns a reference to the underlying [`Value`] of this [`OperandRef`].
    pub fn value(&self) -> ValueRef<'o, 'c, 't> {
        let _guard = self.context.borrow();
        unsafe { ValueRef::from_c_api(mlirOpOperandGetValue(self.to_c_api()), self.context).unwrap() }
    }

    /// Returns a reference to the [`Operation`] that takes this [`OperandRef`] as one of its inputs/operands.
    pub fn operation(&self) -> OperationRef<'o, 'c, 't> {
        let _guard = self.context.borrow();
        unsafe { OperationRef::from_c_api(mlirOpOperandGetOwner(self.to_c_api()), self.context).unwrap() }
    }

    /// Returns the index of this [`OperandRef`] in the [`Operation`] where it is being used.
    pub fn operand_index(&self) -> usize {
        let _guard = self.context.borrow();
        unsafe { mlirOpOperandGetOperandNumber(self.handle) as usize }
    }
}

/// [`Iterator`] over the uses of a [`Value`] as [`OperandRef`]s of [`Operation`]s.
pub struct OperandRefIterator<'o, 'c: 'o, 't: 'c> {
    /// Current [`OperandRef`] in this iterator (i.e., the [`OperandRef`] that will be returned in the next call to
    /// [`OperandRefIterator::next`]). [`Operation`] operands are stored in such a way in MLIR that we can always obtain
    /// the next use of a [`Value`] given the an [`OperandRef`] (that represents one such use).
    current_operand: Option<OperandRef<'o, 'c, 't>>,

    /// [`Context`] reference that, while unused, ensures that the owning context is not modified while
    /// iterating over operands using this iterator.
    _context: Ref<'c, MlirContext>,
}

impl<'o, 'c, 't> Iterator for OperandRefIterator<'o, 'c, 't> {
    type Item = OperandRef<'o, 'c, 't>;

    fn next(&mut self) -> Option<Self::Item> {
        let current_operand = self.current_operand.take();
        self.current_operand = current_operand.as_ref().and_then(|operand| unsafe {
            OperandRef::from_c_api(mlirOpOperandGetNextUse(operand.to_c_api()), operand.context())
        });
        current_operand
    }
}

/// A [`ValueRef`] paired with optional named [`Attribute`](crate::Attribute)s. This is typically used to represent the
/// value and attributes of the arguments to a [`Call`](crate::dialects::func::CallOperation) operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueAndAttributes<'v, 'c, 't, 's> {
    /// Reference to a [`Value`].
    pub value: ValueRef<'v, 'c, 't>,

    /// Optional [`HashMap`] from attribute names to [`Attribute`](crate::Attribute)s.
    pub attributes: Option<HashMap<StringRef<'s>, AttributeRef<'c, 't>>>,
}

impl<'v, 'c, 't, V: Value<'v, 'c, 't>> From<V> for ValueAndAttributes<'v, 'c, 't, '_> {
    fn from(value: V) -> Self {
        Self { value: value.as_ref(), attributes: None }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Attribute, Block, Context, DialectHandle, OperationBuilder};

    use super::*;

    #[test]
    fn test_value() {
        let context = Context::new();
        context.load_dialect(DialectHandle::arith());
        let location = context.unknown_location();
        let index_type = context.index_type();

        // Test using a block argument.
        let block = context.block(&[(index_type, location)]);
        let block_argument = block.argument(0).unwrap();
        assert_eq!(block_argument.context(), &context);
        assert_eq!(block_argument.block(), block);

        // Test using an operation result.
        let op = OperationBuilder::new("arith.constant", location)
            .add_results(&[index_type])
            .add_attribute("value", context.parse_attribute("0 : index").unwrap())
            .build()
            .unwrap();
        let result = op.result(0).unwrap();
        assert_eq!(result.context(), &context);
        assert_eq!(result.location(), location);
        assert_eq!(result.r#type(), index_type);
        assert_eq!(format!("{}", result), "%c0 = arith.constant 0 : index\n");
        assert_eq!(format!("{:?}", result), "OperationResultRef[%c0 = arith.constant 0 : index\n]");

        // Test equality.
        assert_eq!(result, op.result(0).unwrap());
        assert_eq!(result.as_ref(), op.result(0).unwrap());
        assert_eq!(result.as_ref(), result);
        assert_ne!(result, block_argument);
        assert_eq!(block_argument, block_argument);

        // Test null pointer edge case.
        let bad_handle = MlirValue { ptr: std::ptr::null_mut() };
        let value = unsafe { ValueRef::from_c_api(bad_handle, &context) };
        assert!(value.is_none());

        // We are also checking that [`Value::dump`] runs successfully without crashing.
        // Ideally, we would want a way to capture the standard error stream and verify that it printed the right thing.
        result.dump();
    }

    #[test]
    fn test_value_replace_uses() {
        let context = Context::new();
        context.load_dialect(DialectHandle::arith());
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let index_type = context.index_type();
        let op_0 = OperationBuilder::new("arith.constant", location)
            .add_results(&[index_type])
            .add_attribute("value", context.parse_attribute("0 : index").unwrap())
            .build()
            .unwrap();
        let op_1 = OperationBuilder::new("arith.constant", location)
            .add_results(&[index_type])
            .add_attribute("value", context.parse_attribute("1 : index").unwrap())
            .build()
            .unwrap();
        let value_0 = op_0.result(0).unwrap();
        let value_1 = op_1.result(0).unwrap();
        let op_3 = OperationBuilder::new("test.op_3", location).add_operand(value_0).build().unwrap();
        let op_4 = OperationBuilder::new("test.op_4", location).add_operand(value_0).build().unwrap();
        assert_eq!(op_3.operand(0).unwrap(), value_0);
        assert_eq!(op_3.operand(0).unwrap(), value_0);
        value_0.replace_uses(value_1);
        assert_eq!(op_3.operand(0).unwrap(), value_1);
        assert_eq!(op_4.operand(0).unwrap(), value_1);
        value_1.replace_uses_except(value_0, &[&op_3]);
        assert_eq!(op_3.operand(0).unwrap(), value_1);
        assert_eq!(op_4.operand(0).unwrap(), value_0);
    }

    #[test]
    fn test_block_argument() {
        let context = Context::new();
        let location = context.unknown_location();
        let index_type = context.index_type();
        let f64_type = context.float64_type();
        let block = context.block(&[(index_type, location)]);
        let block_argument = block.argument(0).unwrap();
        assert_eq!(block_argument.context(), &context);
        assert_eq!(block_argument.name(false, false).ok().flatten(), None);
        assert_eq!(block_argument.name(true, false).ok().flatten(), None);
        assert_eq!(block_argument.name(false, true).ok().flatten(), None);
        assert_eq!(block_argument.name(true, true).ok().flatten(), None);
        assert_eq!(block_argument.location(), location);
        assert_eq!(block_argument.r#type(), index_type);
        assert_eq!(block_argument.argument_index(), 0);
        assert!(block_argument.is::<ValueRef>());
        assert!(block_argument.is::<BlockArgumentRef>());
        assert!(!block_argument.is::<OperationResultRef>());
        assert_eq!(block_argument.block(), block);
        block_argument.set_type(f64_type);
        assert_eq!(block_argument.r#type(), f64_type);
    }

    #[test]
    fn test_operation_result() {
        let context = Context::new();
        context.load_dialect(DialectHandle::arith());
        let location = context.unknown_location();
        let index_type = context.index_type();
        let op = OperationBuilder::new("arith.constant", location)
            .add_results(&[index_type])
            .add_attribute("value", context.parse_attribute("0 : index").unwrap())
            .build()
            .unwrap();
        let result = op.result(0).unwrap();
        assert_eq!(result.context(), &context);
        assert_eq!(result.name(false, false).ok().flatten(), Some("%c0".to_string()));
        assert_eq!(result.name(true, false).ok().flatten(), Some("%c0".to_string()));
        assert_eq!(result.name(false, true).ok().flatten(), Some("%c0".to_string()));
        assert_eq!(result.name(true, true).ok().flatten(), Some("%c0".to_string()));
        assert_eq!(result.location(), location);
        assert_eq!(result.r#type(), index_type);
        assert_eq!(result.operation(), op);
        assert_eq!(result.result_index(), 0);
        assert!(result.is::<ValueRef>());
        assert!(!result.is::<BlockArgumentRef>());
        assert!(result.is::<OperationResultRef>());
        assert_eq!(format!("{}", result), "%c0 = arith.constant 0 : index\n");
        assert_eq!(format!("{:?}", result), "OperationResultRef[%c0 = arith.constant 0 : index\n]");
    }

    #[test]
    fn test_operand() {
        let context = Context::new();
        context.load_dialect(DialectHandle::arith());
        let location = context.unknown_location();
        let index_type = context.index_type();
        let block = context.block(&[(index_type, location)]);
        let block_argument = block.argument(0).unwrap();
        let op = OperationBuilder::new("arith.constant", location)
            .add_operand(block_argument)
            .add_results(&[index_type])
            .add_attribute("value", context.parse_attribute("0 : index").unwrap())
            .build()
            .unwrap();
        let operand = op.operand(0);
        assert!(operand.is_some());
        let operand = operand.unwrap();
        let block_argument_uses = block_argument.uses().collect::<Vec<_>>();
        assert_eq!(block_argument_uses.len(), 1);
        let block_argument_use = block_argument_uses[0];
        assert_eq!(block_argument_use.context(), &context);
        assert_eq!(block_argument_use.value(), block_argument);
        assert_eq!(block_argument_use.value(), operand);
        assert_eq!(block_argument_use.operation(), op);
        assert_eq!(block_argument_use.operand_index(), 0);
    }

    #[test]
    fn test_value_and_attributes() {
        let context = Context::new();
        let location = context.unknown_location();
        let index_type = context.index_type();
        let block = context.block(&[(index_type, location)]);
        let block_argument = block.argument(0).unwrap();
        let value_and_attributes = ValueAndAttributes::from(block_argument);
        assert_eq!(value_and_attributes.value.to_string(), "<block argument> of type 'index' at index: 0");
        assert!(value_and_attributes.attributes.is_none());
        let value_and_attributes = ValueAndAttributes {
            value: block_argument.as_ref(),
            attributes: Some(HashMap::from([("test_attr".into(), context.unit_attribute().as_ref())])),
        };
        assert_eq!(value_and_attributes.clone().value.to_string(), "<block argument> of type 'index' at index: 0");
        assert!(value_and_attributes.attributes.is_some());
        assert_eq!(value_and_attributes.attributes.unwrap().len(), 1);
    }
}
