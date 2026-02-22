use std::borrow::Cow;

use ryft_xla_sys::bindings::{
    MlirContext, MlirExternalPass, MlirExternalPassCallbacks, MlirLogicalResult, MlirOperation, MlirPass,
    mlirCreateExternalPass, mlirExternalPassSignalFailure,
};

use crate::{Context, ContextRef, DialectHandle, LogicalResult, Operation, OperationRef, StringRef, TypeId};

/// MLIR passes represent the basic infrastructure for transformation and optimization. Refer to the documentation of
/// [`PassManager`](crate::PassManager) and to the [MLIR documentation](https://mlir.llvm.org/docs/PassManagement) for
/// more information.
pub struct Pass {
    /// Handle that represents this [`Pass`] in the MLIR C API.
    handle: MlirPass,
}

impl Pass {
    /// Constructs a new [`Pass`] of this type from the provided handle that came from a function in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn from_c_api(handle: MlirPass) -> Option<Self> {
        if handle.ptr.is_null() { None } else { Some(Self { handle }) }
    }

    /// Returns the [`MlirPass`] that corresponds to this [`Pass`] and which can be passed to functions
    /// in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn to_c_api(&self) -> MlirPass {
        self.handle
    }
}

/// Trait used for supporting [`Pass`]es that are implemented in Rust in the MLIR infrastructure. Specifically,
/// MLIR passes can be developed in Rust by implementing this trait and then using its [`Into<Pass>`] implementation
/// to obtain a [`Pass`] that can be added to a [`PassManager`](crate::PassManager). Refer to [`ClosurePass`] for an
/// example built-in pass that is implemented in Rust and which implements this trait.
pub trait ExternalPass<'c, 't: 'c>: Clone + Sized {
    /// Returns the [`TypeId`] of this [`ExternalPass`].
    fn type_id(&self) -> TypeId<'c>;

    /// Returns the name of this [`ExternalPass`].
    fn name(&self) -> Cow<'_, str>;

    /// Returns the optional command-line argument used when registering this [`ExternalPass`].
    fn command_line_argument(&self) -> Option<Cow<'_, str>>;

    /// Returns the optional command-line description used when registering this [`ExternalPass`].
    fn command_line_description(&self) -> Option<Cow<'_, str>>;

    /// Returns the optional name of the [`Operation`] that this [`ExternalPass`] operates on, or [`None`]
    /// if this is a generic [`Operation`] pass.
    fn operation_name(&self) -> Option<Cow<'_, str>>;

    /// Returns a [`Vec`] that contains [`DialectHandle`]s for all the dependent [`Dialect`](crate::Dialect)s of this
    /// [`ExternalPass`]. These are all the dialects that this pass may create entities (e.g., [`Operation`]s,
    /// [`Type`](crate::Type)s, and [`Attribute`](crate::Attribute)s) for, other than [`Dialect`](crate::Dialect)
    /// that of its inputs. For example, for a pass that translates from dialect `x` to dialect `y` this function
    /// return dialect `y` but not dialect `x`.
    fn dependent_dialects(&self) -> Vec<DialectHandle<'c, 't>>;

    /// Returns the [`Context`] associated with this [`ExternalPass`].
    fn context(&self) -> &'c Context<'t>;

    /// Callback that is called by MLIR right before this [`ExternalPass`] is run. This allows the pass to initialize
    /// any state that is necessary before each run. Returns [`LogicalResult::success`] if it succeeds and
    /// [`LogicalResult::failure`] if it fails.
    fn on_initialization(&mut self, context: ContextRef<'c, 't>) -> LogicalResult;

    /// Callback that is called by MLIR each time this [`ExternalPass`] is run. Returns [`LogicalResult::success`] if
    /// it succeeds and [`LogicalResult::failure`] if it fails.
    fn on_run<'o>(&mut self, operation: OperationRef<'o, 'c, 't>) -> LogicalResult;

    /// Constructs an [`MlirExternalPassCallbacks`] instance that corresponds to this [`ExternalPass`]
    /// and which can be used to provide it to the MLIR C API.
    unsafe fn to_c_api(&self) -> MlirExternalPassCallbacks {
        unsafe extern "C" fn construct<'c, 't: 'c, P: ExternalPass<'c, 't>>(_data: *mut std::ffi::c_void) {}

        unsafe extern "C" fn destruct<'c, 't: 'c, P: ExternalPass<'c, 't>>(data: *mut std::ffi::c_void) {
            unsafe {
                let pass = (data as *mut P).as_mut().expect("encountered invalid external pass");
                std::ptr::drop_in_place(pass);
            }
        }

        unsafe extern "C" fn initialize<'c, 't: 'c, P: ExternalPass<'c, 't>>(
            context: MlirContext,
            data: *mut std::ffi::c_void,
        ) -> MlirLogicalResult {
            unsafe {
                let context = ContextRef::from_c_api(context);
                let pass = (data as *mut P).as_mut().expect("encountered invalid external pass");
                pass.on_initialization(context).to_c_api()
            }
        }

        unsafe extern "C" fn clone<'c, 't: 'c, P: ExternalPass<'c, 't>>(
            data: *mut std::ffi::c_void,
        ) -> *mut std::ffi::c_void {
            unsafe {
                let pass = (data as *mut P).as_ref().expect("encountered invalid external pass");
                Box::<P>::into_raw(Box::new(pass.clone())) as *mut _
            }
        }

        unsafe extern "C" fn run<'c, 't: 'c, P: ExternalPass<'c, 't>>(
            operation: MlirOperation,
            mlir_pass: MlirExternalPass,
            data: *mut std::ffi::c_void,
        ) {
            unsafe {
                let pass = (data as *mut P).as_mut().expect("encountered invalid external pass");
                let context = pass.context();
                let operation = OperationRef::from_c_api(operation, context).expect("encountered invalid operation");
                let result = pass.on_run(operation);
                if result.is_failure() {
                    mlirExternalPassSignalFailure(mlir_pass)
                }
            }
        }

        MlirExternalPassCallbacks {
            construct: Some(construct::<'c, 't, Self>),
            destruct: Some(destruct::<'c, 't, Self>),
            initialize: Some(initialize::<'c, 't, Self>),
            run: Some(run::<'c, 't, Self>),
            clone: Some(clone::<'c, 't, Self>),
        }
    }
}

impl<'c, 't: 'c, P: ExternalPass<'c, 't>> From<P> for Pass {
    fn from(value: P) -> Self {
        let name = value.name();
        let command_line_argument = value.command_line_argument();
        let command_line_description = value.command_line_description();
        let operation_name = value.operation_name();
        let dependent_dialects = value.dependent_dialects();
        unsafe {
            Self::from_c_api(mlirCreateExternalPass(
                value.type_id().to_c_api(),
                StringRef::from(name.as_ref()).to_c_api(),
                StringRef::from(command_line_argument.as_ref().map(|s| s.as_ref()).unwrap_or("")).to_c_api(),
                StringRef::from(command_line_description.as_ref().map(|s| s.as_ref()).unwrap_or("")).to_c_api(),
                StringRef::from(operation_name.as_ref().map(|s| s.as_ref()).unwrap_or("")).to_c_api(),
                dependent_dialects.len().cast_signed(),
                dependent_dialects.as_ptr().cast_mut() as _,
                value.to_c_api(),
                Box::into_raw(Box::new(value)) as _,
            ))
            .unwrap()
        }
    }
}

/// [`ExternalPass`] that wraps a Rust closure (where the closure implements the logic of the pass; i.e., that closure
/// will be invoked on all the [`Operation`]s that this pass is run on).
#[derive(Clone)]
pub struct ClosurePass<'c, 't: 'c, F: Clone + FnMut(OperationRef<'_, 'c, 't>) -> LogicalResult> {
    /// Name of this [`ClosurePass`].
    pub name: String,

    /// Optional command-line argument used when registering this [`ClosurePass`].
    pub command_line_argument: Option<String>,

    /// Optional command-line description used when registering this [`ClosurePass`].
    pub command_line_description: Option<String>,

    /// Optional name of the [`Operation`] that this [`ClosurePass`] operates on, or [`None`] if this is a
    /// generic [`Operation`] pass.
    pub operation_name: Option<String>,

    /// [`Vec`] that contains [`DialectHandle`]s for all the dependent [`Dialect`](crate::Dialect)s of this
    /// [`ClosurePass`]. These are all the dialects that this pass may create entities (e.g., [`Operation`]s,
    /// [`Type`](crate::Type)s, and [`Attribute`](crate::Attribute)s) for, other than [`Dialect`](crate::Dialect)
    /// that of its inputs. For example, for a pass that translates from dialect `x` to dialect `y` this function
    /// return dialect `y` but not dialect `x`.
    pub dependent_dialects: Vec<DialectHandle<'c, 't>>,

    /// [`Context`] associated with this [`ClosurePass`].
    pub context: &'c Context<'t>,

    /// Closure that implements the logic of this [`ClosurePass`]. This closure will be invoked on all the
    /// [`Operation`]s that this pass is run on.
    pub closure: F,
}

impl<'c, 't: 'c, F: Clone + FnMut(OperationRef<'_, 'c, 't>) -> LogicalResult> ExternalPass<'c, 't>
    for ClosurePass<'c, 't, F>
{
    fn type_id(&self) -> TypeId<'c> {
        // We need to make sure that the reference data used to create the [`TypeId`] is 8-byte aligned.
        #[repr(align(8))]
        struct AlignedClosure<F>(F);
        let aligned_closure = AlignedClosure(self.closure.clone());
        TypeId::create(&aligned_closure).unwrap()
    }

    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.name.as_str())
    }

    fn command_line_argument(&self) -> Option<Cow<'_, str>> {
        self.command_line_argument.as_ref().map(|string| Cow::Borrowed(string.as_str()))
    }

    fn command_line_description(&self) -> Option<Cow<'_, str>> {
        self.command_line_description.as_ref().map(|string| Cow::Borrowed(string.as_str()))
    }

    fn operation_name(&self) -> Option<Cow<'_, str>> {
        self.operation_name.as_ref().map(|string| Cow::Borrowed(string.as_str()))
    }

    fn dependent_dialects(&self) -> Vec<DialectHandle<'c, 't>> {
        self.dependent_dialects.clone()
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }

    fn on_initialization(&mut self, _context: ContextRef<'c, 't>) -> LogicalResult {
        LogicalResult::success()
    }

    fn on_run<'o>(&mut self, operation: OperationRef<'o, 'c, 't>) -> LogicalResult {
        (self.closure)(operation)
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Block, Context, Module, Operation, Region, ValueRef};

    use super::*;

    fn test_module<'c, 't>(context: &'c Context<'t>) -> Module<'c, 't> {
        let location = context.unknown_location();
        let module = context.module(location);
        module.body().append_operation(func::func(
            "foo",
            func::FuncAttributes::default(),
            {
                let mut block = context.block_with_no_arguments();
                block.append_operation(func::r#return::<ValueRef, _>(&[], location));
                block.into()
            },
            location,
        ));
        module
    }

    #[test]
    fn test_closure_pass() {
        let context = Context::new();
        let module = test_module(&context);

        // Test using a stateless pass.
        let pass = ClosurePass {
            name: "stateless_pass".into(),
            command_line_argument: Some("custom-arg".into()),
            command_line_description: Some("A pass with custom command line options".into()),
            operation_name: Some("builtin.module".into()),
            dependent_dialects: vec![DialectHandle::func()],
            closure: |op| LogicalResult::from(op.verify()),
            context: &context,
        };
        let _ = pass.type_id();
        assert_eq!(pass.name(), "stateless_pass");
        assert_eq!(pass.command_line_argument(), Some(Cow::Borrowed("custom-arg")));
        assert_eq!(pass.command_line_description(), Some(Cow::Borrowed("A pass with custom command line options")));
        assert_eq!(pass.operation_name(), Some(Cow::Borrowed("builtin.module")));
        assert_eq!(pass.dependent_dialects().len(), 1);
        assert_eq!(pass.context(), &context);

        // Test cloning the pass.
        let cloned_pass = pass.clone();
        assert_eq!(pass.name(), cloned_pass.name());
        assert_eq!(pass.command_line_argument(), cloned_pass.command_line_argument());
        assert_eq!(pass.command_line_description(), cloned_pass.command_line_description());
        assert_eq!(pass.operation_name(), cloned_pass.operation_name());

        // The running the pass.
        let mut pass_manager = context.pass_manager();
        pass_manager.add_pass(pass.into());
        assert!(pass_manager.run(&module.as_operation()).is_success());

        // Test using a stateful pass.
        let counter = Rc::new(RefCell::new(0));
        let counter_clone = counter.clone();
        let pass = ClosurePass {
            name: "stateful_pass".into(),
            command_line_argument: None,
            command_line_description: None,
            operation_name: None,
            dependent_dialects: vec![],
            closure: |op| {
                assert!(op.region_count() > 0);
                assert!(op.region(0).is_some());
                *counter_clone.borrow_mut() += 1;
                LogicalResult::success()
            },
            context: &context,
        };
        let mut pass_manager = context.pass_manager();
        pass_manager.add_pass(pass.clone().into());
        assert_eq!(*counter.borrow(), 0);
        assert!(pass_manager.run(&module.as_operation()).is_success());
        assert_eq!(*counter.borrow(), 1);

        // Test a pass with a failure.
        let pass = ClosurePass {
            name: "failure_pass".into(),
            command_line_argument: None,
            command_line_description: None,
            operation_name: None,
            dependent_dialects: vec![DialectHandle::func()],
            closure: |_| LogicalResult::failure(),
            context: &context,
        };
        let mut pass_manager = context.pass_manager();
        pass_manager.add_pass(pass.into());
        assert!(pass_manager.run(&module.as_operation()).is_failure());
    }

    #[test]
    fn test_external_pass() {
        #[derive(Clone, Debug)]
        struct TestPass<'c, 't: 'c> {
            context: &'c Context<'t>,
            value: i32,
        }

        impl<'c, 't: 'c> ExternalPass<'c, 't> for TestPass<'c, 't> {
            fn type_id(&self) -> TypeId<'c> {
                TypeId::create(&self).unwrap()
            }

            fn name(&self) -> Cow<'_, str> {
                Cow::Borrowed("external_pass")
            }

            fn command_line_argument(&self) -> Option<Cow<'_, str>> {
                Some(Cow::Borrowed("custom-arg"))
            }

            fn command_line_description(&self) -> Option<Cow<'_, str>> {
                Some(Cow::Borrowed("An external test pass"))
            }

            fn operation_name(&self) -> Option<Cow<'_, str>> {
                Some(Cow::Borrowed("builtin.module"))
            }

            fn dependent_dialects(&self) -> Vec<DialectHandle<'c, 't>> {
                vec![DialectHandle::func()]
            }

            fn context(&self) -> &'c Context<'t> {
                self.context
            }

            fn on_initialization(&mut self, _context: ContextRef<'c, 't>) -> LogicalResult {
                self.value = 42;
                LogicalResult::success()
            }

            fn on_run<'o>(&mut self, operation: OperationRef<'o, 'c, 't>) -> LogicalResult {
                assert_eq!(self.value, 42);
                self.value = 30;
                assert_eq!(operation.name().as_str().unwrap(), "builtin.module");
                assert!(operation.verify());
                assert_eq!(
                    operation.region(0).unwrap().blocks().next().unwrap().operations().next().unwrap().name(),
                    self.context.identifier("func.func"),
                );
                LogicalResult::success()
            }
        }

        let context = Context::new();
        let module = test_module(&context);
        let pass = TestPass { context: &context, value: 10 };
        let mut pass_manager = context.pass_manager();
        pass_manager.add_pass(pass.clone().into());
        assert!(pass_manager.run(&module.as_operation()).is_success());

        // Check that the C API callback conversions work as expected. We are only checking `clone`
        // because the rest of the callbacks should have already been checked while running the pass.
        unsafe {
            let callbacks = pass.to_c_api();
            let cloned = callbacks.clone.unwrap()(Box::into_raw(Box::new(pass)) as _);
            let pass = (cloned as *mut TestPass).as_ref().expect("encountered invalid external pass");
            assert_eq!(pass.name(), Cow::Borrowed("external_pass"));
        }
    }
}
