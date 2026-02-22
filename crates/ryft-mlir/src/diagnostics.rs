use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use ryft_xla_sys::bindings::{
    MlirDiagnostic, MlirDiagnosticHandlerID, MlirDiagnosticSeverity_MlirDiagnosticError,
    MlirDiagnosticSeverity_MlirDiagnosticNote, MlirDiagnosticSeverity_MlirDiagnosticRemark,
    MlirDiagnosticSeverity_MlirDiagnosticWarning, MlirLogicalResult, mlirContextAttachDiagnosticHandler,
    mlirContextDetachDiagnosticHandler, mlirDiagnosticGetLocation, mlirDiagnosticGetNote, mlirDiagnosticGetNumNotes,
    mlirDiagnosticGetSeverity, mlirDiagnosticPrint,
};

use crate::support::write_to_formatter_callback;
use crate::{Context, Location, LocationRef, LogicalResult};

/// Severity level of a [`Diagnostic`]. Note that if the underlying native library returns a severity level that
/// is not recognized by this library (e.g., due to a version incompatibility), then that severity level will be
/// represented as an [`DiagnosticSeverity::Unknown`] severity.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Note,
    Remark,

    /// Unknown [`DiagnosticSeverity`] level that holds the [`std::ffi::c_uint`] value used to represent it in the
    /// MLIR C API. This value could be unknown due to a mismatch between the version of MLIR that these Rust bindings
    /// were designed for and the version of the MLIR library that is linked to this executable.
    Unknown(std::ffi::c_uint),
}

impl From<std::ffi::c_uint> for DiagnosticSeverity {
    fn from(value: std::ffi::c_uint) -> Self {
        #[allow(non_upper_case_globals)]
        match value {
            MlirDiagnosticSeverity_MlirDiagnosticError => Self::Error,
            MlirDiagnosticSeverity_MlirDiagnosticWarning => Self::Warning,
            MlirDiagnosticSeverity_MlirDiagnosticNote => Self::Note,
            MlirDiagnosticSeverity_MlirDiagnosticRemark => Self::Remark,
            _ => Self::Unknown(value),
        }
    }
}

/// Opaque reference to a diagnostic that is always owned by a diagnostics handler that is attached to a [`Context`].
/// [`Diagnostic`]s must not be stored outside diagnostic handlers. Refer to [`Context::attach_diagnostics_handler`]
/// for more information.
///
/// Note that there are multiple separate lifetime parameters: one for the lifetime of the owner of this [`Diagnostic`],
/// `'o`, one for the [`Context`] which is associated with it, `'c`, and one for the lifetime of the thread pool used
/// by that [`Context`], `'t`.
pub struct Diagnostic<'o, 'c, 't> {
    /// Handle that represents this [`Diagnostic`] in the MLIR C API.
    handle: MlirDiagnostic,

    /// [`Context`] that owns this [`Diagnostic`].
    context: &'c Context<'t>,

    /// [`PhantomData`] used to track the lifetime of the owner of this [`Diagnostic`].
    owner: PhantomData<&'o ()>,
}

impl<'o, 'c, 't> Diagnostic<'o, 'c, 't> {
    /// Constructs a new [`Diagnostic`] from the provided [`MlirDiagnostic`] handle.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn from_c_api(handle: MlirDiagnostic, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() { None } else { Some(Self { handle, context, owner: PhantomData }) }
    }

    /// Returns a reference to the [`Context`] that owns this [`Diagnostic`].
    pub fn context(&self) -> &'c Context<'t> {
        self.context
    }

    /// Returns the [`DiagnosticSeverity`] of this [`Diagnostic`].
    pub fn severity(&self) -> DiagnosticSeverity {
        DiagnosticSeverity::from(unsafe { mlirDiagnosticGetSeverity(self.handle) })
    }

    /// Returns the [`Location`] at which this [`Diagnostic`] was reported.
    pub fn location(&self) -> LocationRef<'c, 't> {
        unsafe { LocationRef::from_c_api(mlirDiagnosticGetLocation(self.handle), self.context).unwrap() }
    }

    /// Returns the number of notes attached to this [`Diagnostic`].
    pub fn note_count(&self) -> usize {
        unsafe { mlirDiagnosticGetNumNotes(self.handle).cast_unsigned() }
    }

    /// Returns the notes attached to this [`Diagnostic`].
    pub fn notes(&self) -> impl Iterator<Item = Self> {
        (0..self.note_count()).map(|index| self.note(index).unwrap())
    }

    /// Returns the `index`-th note attached to this [`Diagnostic`]. If `index` is larger than
    /// [`Diagnostic::note_count`], then this function will return [`None`].
    pub fn note(&self, index: usize) -> Option<Self> {
        if index < self.note_count() {
            unsafe { Self::from_c_api(mlirDiagnosticGetNote(self.handle, index.cast_signed()), self.context) }
        } else {
            None
        }
    }
}

impl Display for Diagnostic<'_, '_, '_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut data = (formatter, Ok(()));
        let data_ptr = &mut data as *mut _ as *mut std::ffi::c_void;
        unsafe { mlirDiagnosticPrint(self.handle, Some(write_to_formatter_callback), data_ptr) };
        data.1
    }
}

impl Debug for Diagnostic<'_, '_, '_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        Display::fmt(self, formatter)
    }
}

/// ID of a [`Diagnostic`] handler. Refer to [`Context::attach_diagnostics_handler`] for more information.
#[derive(Copy, Clone, Debug)]
pub struct DiagnosticHandlerId {
    pub(crate) id: MlirDiagnosticHandlerID,
}

impl<'t> Context<'t> {
    /// Attaches a [`Diagnostic`]s handler to this [`Context`] and returns a [`DiagnosticHandlerId`] that can be used
    /// to detach the handler later on, if needed, using [`Context::detach_diagnostics_handler`]. Note that handlers
    /// will be invoked in the reverse order of attachment until one of them processes the [`Diagnostic`] completely.
    /// Handlers will be automatically dropped when detached from the [`Context`] or when the [`Context`] they are
    /// attached to is dropped.
    ///
    /// # Parameters
    ///
    ///   * `handler` - Function that accepts a [`Diagnostic`], which is only guaranteed to be live during the call,
    ///     and returns `true` when it processes the input [`Diagnostic`] completely (meaning that no other handler
    ///     will be invoked later on for the same [`Diagnostic`]), and `false` otherwise to let other handlers attempt
    ///     to process the [`Diagnostic`].
    pub fn attach_diagnostics_handler<'c, F: FnMut(Diagnostic<'_, 'c, 't>) -> bool>(
        &'c self,
        handler: F,
    ) -> DiagnosticHandlerId {
        unsafe extern "C" fn handle<'f, 'c: 'f, 't: 'c, F: FnMut(Diagnostic<'_, 'c, 't>) -> bool>(
            diagnostic: MlirDiagnostic,
            user_data: *mut std::ffi::c_void,
        ) -> MlirLogicalResult {
            unsafe {
                let user_data = user_data as *mut (F, &'c Context<'t>);
                let (ref mut handler, context) = *user_data;
                LogicalResult::from((*handler)(Diagnostic::from_c_api(diagnostic, context).unwrap())).to_c_api()
            }
        }

        unsafe extern "C" fn delete<'f, 'c: 'f, 't: 'c, F: FnMut(Diagnostic<'_, 'c, 't>) -> bool>(
            user_data: *mut std::ffi::c_void,
        ) {
            if !user_data.is_null() {
                // When this new [`Box`] goes out of scope, it will be dropped automatically.
                let _ = unsafe { Box::from_raw(user_data as *mut (F, &'c Context<'t>)) };
            }
        }

        DiagnosticHandlerId {
            id: unsafe {
                mlirContextAttachDiagnosticHandler(
                    *self.handle.borrow_mut(),
                    Some(handle::<'_, 'c, 't, F>),
                    Box::into_raw(Box::new((handler, &self))) as *mut _,
                    Some(delete::<'_, 'c, 't, F>),
                )
            },
        }
    }

    /// Detaches a [`Diagnostic`]s handler from this [`Context`]. Note that this will also drop the underlying handler.
    pub fn detach_diagnostics_handler(&self, id: DiagnosticHandlerId) {
        unsafe { mlirContextDetachDiagnosticHandler(*self.handle.borrow_mut(), id.id) }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use pretty_assertions::assert_eq;

    use crate::DialectHandle;

    use super::*;

    #[test]
    fn test_diagnostic_severity_from_c_uint() {
        assert_eq!(DiagnosticSeverity::from(MlirDiagnosticSeverity_MlirDiagnosticError), DiagnosticSeverity::Error);
        assert_eq!(DiagnosticSeverity::from(MlirDiagnosticSeverity_MlirDiagnosticWarning), DiagnosticSeverity::Warning);
        assert_eq!(DiagnosticSeverity::from(MlirDiagnosticSeverity_MlirDiagnosticNote), DiagnosticSeverity::Note);
        assert_eq!(DiagnosticSeverity::from(MlirDiagnosticSeverity_MlirDiagnosticRemark), DiagnosticSeverity::Remark);
        assert_eq!(DiagnosticSeverity::from(9999), DiagnosticSeverity::Unknown(9999));
    }

    #[test]
    fn test_diagnostic_severity_ordering() {
        assert!(DiagnosticSeverity::Error < DiagnosticSeverity::Warning);
        assert!(DiagnosticSeverity::Warning < DiagnosticSeverity::Note);
        assert!(DiagnosticSeverity::Note < DiagnosticSeverity::Remark);
        assert_eq!(DiagnosticSeverity::Error, DiagnosticSeverity::Error);
    }

    #[test]
    fn test_null_diagnostic() {
        let context = Context::new();
        assert!(unsafe { Diagnostic::from_c_api(MlirDiagnostic { ptr: std::ptr::null_mut() }, &context) }.is_none());
    }

    #[test]
    fn test_diagnostics_handler() {
        let context = Context::new();
        context.load_dialect(DialectHandle::func());

        // Create a diagnostics handler that collects a bunch of information to check all accessors.
        let message = Rc::new(RefCell::new(None));
        let severity = Rc::new(RefCell::new(None));
        let location = Rc::new(RefCell::new(None));
        let notes = Rc::new(RefCell::new(Vec::new()));
        let message_clone = message.clone();
        let severity_clone = severity.clone();
        let location_clone = location.clone();
        let notes_clone = notes.clone();

        let diagnostics_handler_id = context.attach_diagnostics_handler(move |diagnostic| {
            // Verify that we can access the context without crashing.
            let _ = diagnostic.context();
            *message_clone.borrow_mut() = Some(diagnostic.to_string());
            *severity_clone.borrow_mut() = Some(diagnostic.severity());
            *location_clone.borrow_mut() = Some(diagnostic.location().to_string());
            *notes_clone.borrow_mut() = diagnostic.notes().map(|note| note.to_string()).collect();
            assert!(diagnostic.note_count() > 0);
            assert_eq!(
                diagnostic.note(0).unwrap().to_string(),
                "see current operation: \"func.return\"(%arg0) : (i32) -> ()".to_string(),
            );
            assert!(diagnostic.note(100).is_none());
            assert_eq!(
                format!("{}", diagnostic),
                "type of return operand 0 ('i32') doesn't match function result type ('i64') in function @test",
            );
            assert_eq!(
                format!("{:?}", diagnostic),
                "type of return operand 0 ('i32') doesn't match function result type ('i64') in function @test",
            );
            true
        });

        // Create a func.func operation with mismatched function signature that will fail verification and
        // produce diagnostics with notes.
        context.parse_module(
            r#"
            module {
                func.func @test(%arg0: i32) -> i64 {
                    func.return %arg0 : i32
                }
            }
            "#,
        );

        // Check that we captured diagnostic information.
        assert_eq!(
            message.borrow().as_ref().unwrap(),
            "type of return operand 0 ('i32') doesn't match function result type ('i64') in function @test",
        );
        assert_eq!(severity.borrow().unwrap(), DiagnosticSeverity::Error);
        assert!(location.borrow().is_some());
        assert_eq!(
            notes.borrow().as_slice(),
            &["see current operation: \"func.return\"(%arg0) : (i32) -> ()".to_string()],
        );

        // Verify that we can detach the handler.
        context.detach_diagnostics_handler(diagnostics_handler_id);
        context.parse_module("foo");
        assert_eq!(
            message.borrow().as_ref().unwrap(),
            "type of return operand 0 ('i32') doesn't match function result type ('i64') in function @test",
        );

        // Now go ahead and attach two other handlers.
        let messages = Rc::new(RefCell::new(Vec::new()));
        let messages_clone_0 = messages.clone();
        let messages_clone_1 = messages.clone();

        // First handler (will be invoked second).
        let diagnostics_handler_id_0 = context.attach_diagnostics_handler(move |diagnostic| {
            messages_clone_0.borrow_mut().push(format!("Handler 0: {}", diagnostic));
            false // Do not consume the diagnostic.
        });

        // Second handler (will be invoked first).
        let diagnostics_handler_id_1 = context.attach_diagnostics_handler(move |diagnostic| {
            messages_clone_1.borrow_mut().push(format!("Handler 1: {}", diagnostic));
            false // Do not consume the diagnostic.
        });

        context.parse_module("foo");
        assert_eq!(
            messages.borrow().as_slice(),
            &[
                "Handler 1: custom op 'foo' is unknown (tried 'builtin.foo' as well)".to_string(),
                "Handler 0: custom op 'foo' is unknown (tried 'builtin.foo' as well)".to_string(),
            ]
        );

        // Now detach the two previous handlers and attach two new ones, where one is consuming the diagnostic.
        context.detach_diagnostics_handler(diagnostics_handler_id_0);
        context.detach_diagnostics_handler(diagnostics_handler_id_1);

        let messages = Rc::new(RefCell::new(Vec::new()));
        let messages_clone_0 = messages.clone();
        let messages_clone_1 = messages.clone();

        // First handler (will be invoked second).
        context.attach_diagnostics_handler(move |diagnostic| {
            messages_clone_0.borrow_mut().push(format!("Handler 0: {}", diagnostic));
            false // Do not consume the diagnostic.
        });

        // Second handler (will be invoked first).
        context.attach_diagnostics_handler(move |diagnostic| {
            messages_clone_1.borrow_mut().push(format!("Handler 1: {}", diagnostic));
            diagnostic.to_string().contains("foo") // Consume the diagnostic if it contains "foo".
        });

        context.parse_module("foo");
        context.parse_module("bar");
        assert_eq!(
            messages.borrow().as_slice(),
            &[
                "Handler 1: custom op 'foo' is unknown (tried 'builtin.foo' as well)".to_string(),
                "Handler 1: custom op 'bar' is unknown (tried 'builtin.bar' as well)".to_string(),
                "Handler 0: custom op 'bar' is unknown (tried 'builtin.bar' as well)".to_string(),
            ],
        );
    }
}
