use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use crate::{Api, Client, Device, Error, Plugin, invoke_pjrt_api_error_fn};

/// Represents a notifying event that is returned by PJRT APIs that enqueue asynchronous work, informing callers when
/// the work is complete and reporting an [`Error`] if something went wrong. Note that [`Event`]s can carry "payload"
/// values that are returned as outputs when the underlying work completes via [`Event::await`] or [`Event::poll`]
/// (e.g., such a payload value could be a host buffer that is being asynchronously populated).
///
/// # Relationship to [`Future`]
///
/// [`Event`]s implement [`Future`] so that they can be seamlessly integrated with asynchronous Rust code.
/// However, while in Rust [`Future`]s typically do not start executing until they are invoked, [`Event`]s
/// represent computations that have already started executing.
pub struct Event<O> {
    /// Handle that represents this [`Event`] in the PJRT C API.
    handle: *mut ffi::PJRT_Event,

    /// State used to track the [`Future`] polling status of this [`Event`]. Specifically, this state is used to make
    /// sure that a callback is registered for this PJRT event such that it can be used as a standard [`Future`].
    state: Arc<EventState>,

    /// "Payload" that this [`Event`] carries. Specifically, this payload will be [`Some`] throughout the lifetime of
    /// this [`Event`] and will be returned when it completes (i.e., when its underlying computation completes) via
    /// [`Event::await`] or [`Event::poll`].
    output: Option<O>,
}

impl<O> Event<O> {
    /// Constructs a new [`Event`] from the provided [`PJRT_Event`] handle that came from a function in the PJRT C API.
    /// The provided `output` represents a "payload" for the resulting [`Event`], that is returned when the underlying
    /// computation finishes executing (e.g., it could be a buffer that is being populated asynchronously and the event
    /// represents the completion of the buffer population).
    pub(crate) unsafe fn from_c_api(handle: *mut ffi::PJRT_Event, api: Api, output: O) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument("the provided PJRT event handle is a null pointer"))
        } else {
            Ok(Self { handle, state: Arc::new(EventState { api, waker: Mutex::new(None) }), output: Some(output) })
        }
    }

    /// Returns the [`PJRT_Event`](ffi::PJRT_Event) that corresponds to this [`Event`] and which can
    /// be passed to functions in the PJRT C API.
    pub(crate) unsafe fn to_c_api(&self) -> *mut ffi::PJRT_Event {
        self.handle
    }

    /// Returns an [`EventHandle`] that can be used to set/trigger this [`Event`] from another thread.
    ///
    /// # Safety
    ///
    /// This function is marked as unsafe because this [`Event`] must remain alive for the lifetime of the returned
    /// [`EventHandle`] (otherwise the latter can become invalid), but there is currently no way to enforce that.
    pub unsafe fn handle(&self) -> EventHandle {
        EventHandle { handle: unsafe { self.to_c_api() }, api: self.state.api }
    }

    /// Sets/triggers this [`Event`] to indicate that the work it represents has completed successfully. If an [`Error`]
    /// is provided, it will be returned when this [`Event`] is polled via [`Event::poll`] or [`Event::await`],
    /// representing that something went wrong while executing the underlying work. Otherwise, the work will be
    /// considered successful.
    ///
    /// # Panic
    ///
    /// Note that if this function is called more than once for a single [`Event`] (or if it is called for an already
    /// fulfilled [`Event`] more generally), it may panic instead of returning an [`Error`]. This behavior, while
    /// undesirable, unfortunately depends on the underlying PJRT [`Plugin`] implementation.
    pub fn set(&mut self, error: Option<Error>) -> Result<(), Error> {
        use ffi::PJRT_Event_Set_Args;
        let error_message = error.as_ref().map(|error| error.message());
        invoke_pjrt_api_error_fn!(
            self.state.api,
            PJRT_Event_Set,
            {
                client = self.to_c_api(),
                error_code = error.as_ref().map(|error| error.code()).unwrap_or(crate::errors::ffi::PJRT_Error_Code_OK),
                error_message = error_message.as_ref().map(|message| message.as_ptr()).unwrap_or(std::ptr::null()),
                error_message_size = error_message.as_ref().map(|message| message.count_bytes()).unwrap_or(0),
            },
        )
    }

    /// Checks if the underlying computation of this [`Event`] has finished executing and returns `true` if it has and
    /// `false` otherwise. Note that an [`Error`] may also be returned if something goes wrong while checking on the
    /// [`Event`]'s status.
    pub fn ready(&self) -> Result<bool, Error> {
        use ffi::PJRT_Event_IsReady_Args;
        invoke_pjrt_api_error_fn!(self.state.api, PJRT_Event_IsReady, { event = self.to_c_api() }, { is_ready })
    }

    /// Registers the provided callback to be invoked when the underlying computation of this [`Event`] finishes
    /// executing. The callback takes an optional [`Error`] as its sole argument whose value depends on whether the
    /// underlying computation produced an error or not.
    pub fn on_ready<F: FnOnce(Option<Error>)>(&self, callback: F) -> Result<(), Error> {
        use ffi::PJRT_Event_OnReady_Args;

        extern "C" fn callback_fn<F: FnOnce(Option<Error>)>(
            error: *mut crate::errors::ffi::PJRT_Error,
            arg: *mut std::ffi::c_void,
        ) {
            let arg = arg as *mut (F, Api);
            let (callback, api) = *unsafe { Box::from_raw(arg) };
            let error = if error.is_null() { None } else { unsafe { Error::from_c_api(error, api).unwrap() } };
            callback(error)
        }

        let callback_fn_arg = Box::into_raw(Box::new((callback, self.state.api)));
        invoke_pjrt_api_error_fn!(
            self.state.api,
            PJRT_Event_OnReady,
            { event = self.to_c_api(), callback = callback_fn::<F>, user_arg = callback_fn_arg as *mut _ },
        )
        .inspect_err(|_| drop(unsafe { Box::from_raw(callback_fn_arg) }))
    }

    /// Blocks the current thread until this [`Event`] is _ready_, returning an [`Error`] if something went wrong.
    pub fn r#await(self) -> Result<O, Error> {
        use ffi::PJRT_Event_Await_Args;

        // It is safe to force-unwrap the output option because it is always going to be `Some` unless
        // this function has been called, and this function consumes `self`.
        let mut event = self;
        invoke_pjrt_api_error_fn!(event.state.api, PJRT_Event_Await, { event = event.handle })
            .map(|_| event.output.take().unwrap())
    }

    /// Returns an [`Error`] that was encountered while waiting for the underlying computation of this [`Event`] to
    /// execute. If the underlying computation has already executed and was successful, this function will return
    /// `Ok(None)`. If the underlying computation has not finished executing yet, this function will return
    /// `Err(Error::FailedPrecondition)`. Otherwise, if the underlying computation has finished executing and ran
    /// into an error, this function will return `Ok(Some(error))`, where `error` is the error that was encountered
    /// by the underlying computation.
    pub fn error(&self) -> Result<Option<Error>, Error> {
        if !self.ready()? {
            Err(Error::failed_precondition("`Event::ready` must return `true` for `Event::error` to be meaningful"))
        } else {
            use ffi::PJRT_Event_Error_Args;
            Ok(invoke_pjrt_api_error_fn!(self.state.api, PJRT_Event_Error, { event = self.to_c_api() }).err())
        }
    }
}

impl<O> Future for Event<O> {
    type Output = Result<O, Error>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // It is safe to force unwrap the output option when ready because it is always going to be `Some`
        // unless this [`Future`] has already returned [`Poll::Ready`] in an earlier call to [`Future::poll`],
        // in which case this function should not have been called and a potential panic is expected.
        match self.ready() {
            Ok(true) => match self.error() {
                Ok(Some(error)) => Poll::Ready(Err(error)),
                Ok(None) => Poll::Ready(Ok(unsafe { self.get_unchecked_mut().output.take().unwrap() })),
                // `Err` is only ever returned by `self.error()` if `self.ready()` is `false` and therefore,
                // the following branch is unreachable.
                Err(_) => unreachable!(),
            },
            Ok(false) => {
                let callback_registration_result = {
                    let mut waker = self.state.waker.lock().unwrap();
                    let has_registered_callback = waker.is_some();
                    *waker = Some(cx.waker().clone());
                    if !has_registered_callback {
                        let data = Arc::into_raw(self.state.clone()) as *mut std::ffi::c_void;
                        Some(self.on_ready(|_| {
                            // We ignore the error because there is nothing we can do with it here,
                            // and if something goes wrong, it should be reflected in [`Event::error`].
                            let state = unsafe { Arc::from_raw(data as *const EventState) };
                            if let Some(waker) = state.waker.lock().unwrap().take() {
                                waker.wake();
                            }
                        }))
                    } else {
                        None
                    }
                };
                match callback_registration_result {
                    Some(Ok(())) => match self.ready() {
                        Ok(true) => match self.error() {
                            Ok(Some(error)) => Poll::Ready(Err(error)),
                            Ok(None) => Poll::Ready(Ok(unsafe { self.get_unchecked_mut().output.take().unwrap() })),
                            // `Err` is only ever returned by `self.error()` if `self.ready()` is `false` and therefore,
                            // the following branch is unreachable.
                            Err(_) => unreachable!(),
                        },
                        _ => Poll::Pending,
                    },
                    Some(Err(error)) => Poll::Ready(Err(error)),
                    None => Poll::Pending,
                }
            }
            Err(error) => Poll::Ready(Err(error)),
        }
    }
}

impl<O> Drop for Event<O> {
    fn drop(&mut self) {
        use ffi::PJRT_Event_Destroy_Args;

        *self.state.waker.lock().unwrap() = None;
        invoke_pjrt_api_error_fn!(self.state.api, PJRT_Event_Destroy, { event = self.to_c_api() })
            .expect("failed to destroy PJRT event");

        // Make sure that we do not leak `self.state` if an "on-ready" callback
        // has been registered but has not been invoked yet.
        while Arc::strong_count(&self.state) > 1 {
            unsafe { Arc::decrement_strong_count(&self.state) };
        }
    }
}

/// Handle associated with an [`Event`] that can be used to set/trigger that event from another thread, but which is
/// inherently unsafe. Refer to the documentation of [`Event::handle`] for a comment on safety. The main feature of this
/// struct is that it is [`Send`] and [`Sync`] while [`Event`] itself is not.
#[derive(Clone)]
pub struct EventHandle {
    /// Handle that represents this [`EventHandle`] in the PJRT C API.
    handle: *mut ffi::PJRT_Event,

    /// Underlying PJRT [`Api`].
    api: Api,
}

impl EventHandle {
    /// Sets/triggers the [`Event`] associated with this [`EventHandle`] to indicate that the work it represents has
    /// completed successfully. If an [`Error`] is provided, it will be returned when the associated [`Event`] is polled
    /// via [`Event::await`] or [`Event::poll`], representing that something went wrong while executing the underlying
    /// work. Otherwise, the work will be considered successful.
    ///
    /// # Safety
    ///
    /// This function is marked as unsafe because it requires that the associated [`Event`] has not been dropped by the
    /// time it is called, and there is no way to enforce that currently.
    ///
    /// # Panic
    ///
    /// Note that if this function is called more than once for a single [`Event`] (or if it is called for an already
    /// fulfilled [`Event`] more generally), it may panic instead of returning an [`Error`]. This behavior, while
    /// undesirable, unfortunately depends on the underlying PJRT [`Plugin`] implementation.
    pub unsafe fn set(&mut self, error: Option<Error>) -> Result<(), Error> {
        use ffi::PJRT_Event_Set_Args;
        let error_message = error.as_ref().map(|error| error.message());
        invoke_pjrt_api_error_fn!(
            self.api,
            PJRT_Event_Set,
            {
                client = self.handle,
                error_code = error.as_ref().map(|error| error.code()).unwrap_or(crate::errors::ffi::PJRT_Error_Code_OK),
                error_message = error_message.as_ref().map(|message| message.as_ptr()).unwrap_or(std::ptr::null()),
                error_message_size = error_message.as_ref().map(|message| message.count_bytes()).unwrap_or(0),
            },
        )
    }
}

unsafe impl Send for EventHandle {}
unsafe impl Sync for EventHandle {}

impl Client<'_> {
    /// Creates a new [`Event`] that carries with it the provided `output` and can be used to track the completion of
    /// an asynchronous computation. `output` will be returned when the underlying computation completes. For example,
    /// `output` could be a buffer and the [`Event`] could be used to track when the work done to populate that buffer
    /// is complete.
    pub fn event<O>(&self, output: O) -> Result<Event<O>, Error> {
        self.api().event(output)
    }
}

impl Plugin {
    /// Creates a new [`Event`] that carries with it the provided `output` and can be used to track the completion of
    /// an asynchronous computation. `output` will be returned when the underlying computation completes. For example,
    /// `output` could be a buffer and the [`Event`] could be used to track when the work done to populate that buffer
    /// is complete.
    pub fn event<O>(&self, output: O) -> Result<Event<O>, Error> {
        self.api().event(output)
    }
}

impl Api {
    /// Creates a new [`Event`] that carries with it the provided `output` and can be used to track the completion of
    /// an asynchronous computation. `output` will be returned when the underlying computation completes. For example,
    /// `output` could be a buffer and the [`Event`] could be used to track when the work done to populate that buffer
    /// is complete.
    pub(crate) fn event<O>(&self, output: O) -> Result<Event<O>, Error> {
        use ffi::PJRT_Event_Create_Args;
        invoke_pjrt_api_error_fn!(*self, PJRT_Event_Create, {}, { event })
            .and_then(|handle| unsafe { Event::from_c_api(handle, *self, output) })
    }
}

/// State used to track the [`Future`] polling status of an [`Event`]. Specifically, this state is used to make sure
/// that a callback is registered for the corresponding PJRT event such that it can be used as a standard [`Future`].
struct EventState {
    /// Underlying PJRT [`Api`].
    api: Api,

    /// [`Waker`] registered for the corresponding [`Event`] in its last [`Event::poll`] invocation.
    waker: Mutex<Option<Waker>>,
}

/// Event that can be used to tell PJRT [`Client`]s about asynchronous actions outside of PJRT. [`AsyncTrackingEvent`]s
/// can be crated using [`Device::async_tracking_event`], and the creation of such an event tells the PJRT [`Client`]
/// that it is creating some outstanding asynchronous work that depends on activities happening on that [`Device`].
/// The caller indicates that the work tracked by an [`AsyncTrackingEvent`] has completed by dropping that event.
/// [`AsyncTrackingEvent`]s are used by some PJRT [`Plugin`] implementations to monitor system-wide dependencies.
pub struct AsyncTrackingEvent {
    /// Handle that represents this [`AsyncTrackingEvent`] in the PJRT C API.
    handle: *mut ffi::PJRT_AsyncTrackingEvent,

    /// Underlying PJRT [`Api`].
    api: Api,
}

impl AsyncTrackingEvent {
    /// Constructs a new [`AsyncTrackingEvent`] from the provided
    /// [`PJRT_AsyncTrackingEvent`](ffi::PJRT_AsyncTrackingEvent) handle
    /// that came from a function in the PJRT C API.
    pub(crate) unsafe fn from_c_api(handle: *mut ffi::PJRT_AsyncTrackingEvent, api: Api) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument("the provided PJRT async tracking event handle is a null pointer"))
        } else {
            Ok(Self { handle, api })
        }
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }
}

impl Drop for AsyncTrackingEvent {
    fn drop(&mut self) {
        use ffi::PJRT_AsyncTrackingEvent_Destroy_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_AsyncTrackingEvent_Destroy, { event = self.handle })
            .expect("failed to destroy PJRT async tracking event");
    }
}

impl Device<'_> {
    /// Creates a new [`AsyncTrackingEvent`] for tracking activities on this [`Device`].
    pub fn async_tracking_event<S: AsRef<str>>(&self, description: S) -> Result<AsyncTrackingEvent, Error> {
        use ffi::PJRT_Device_CreateAsyncTrackingEvent_Args;
        let description = std::ffi::CString::new(description.as_ref()).unwrap();
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Device_CreateAsyncTrackingEvent,
            {
                device = self.to_c_api(),
                description = description.as_ptr(),
                description_size = description.count_bytes(),
            },
            { event },
        )
        .and_then(|handle| unsafe { AsyncTrackingEvent::from_c_api(handle, self.api()) })
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use std::marker::{PhantomData, PhantomPinned};

    use crate::devices::ffi::PJRT_Device;
    use crate::errors::ffi::{PJRT_Error, PJRT_Error_Code};
    use crate::ffi::PJRT_Extension_Base;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_Event {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct PJRT_Event_Create_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub event: *mut PJRT_Event,
    }

    impl PJRT_Event_Create_Args {
        pub fn new() -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), event: std::ptr::null_mut() }
        }
    }

    pub type PJRT_Event_Create = unsafe extern "C" fn(args: *mut PJRT_Event_Create_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Event_Set_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub event: *mut PJRT_Event,
        pub error_code: PJRT_Error_Code,
        pub error_message: *const std::ffi::c_char,
        pub error_message_size: usize,
    }

    impl PJRT_Event_Set_Args {
        pub fn new(
            event: *mut PJRT_Event,
            error_code: PJRT_Error_Code,
            error_message: *const std::ffi::c_char,
            error_message_size: usize,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                event,
                error_code,
                error_message,
                error_message_size,
            }
        }
    }

    pub type PJRT_Event_Set = unsafe extern "C" fn(args: *mut PJRT_Event_Set_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Event_IsReady_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub event: *mut PJRT_Event,
        pub is_ready: bool,
    }

    impl PJRT_Event_IsReady_Args {
        pub fn new(event: *mut PJRT_Event) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), event, is_ready: false }
        }
    }

    pub type PJRT_Event_IsReady = unsafe extern "C" fn(args: *mut PJRT_Event_IsReady_Args) -> *mut PJRT_Error;

    pub type PJRT_Event_OnReadyCallback = unsafe extern "C" fn(error: *mut PJRT_Error, user_arg: *mut std::ffi::c_void);

    #[repr(C)]
    pub struct PJRT_Event_OnReady_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub event: *mut PJRT_Event,
        pub callback: PJRT_Event_OnReadyCallback,
        pub user_arg: *mut std::ffi::c_void,
    }

    impl PJRT_Event_OnReady_Args {
        pub fn new(
            event: *mut PJRT_Event,
            callback: PJRT_Event_OnReadyCallback,
            user_arg: *mut std::ffi::c_void,
        ) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), event, callback, user_arg }
        }
    }

    pub type PJRT_Event_OnReady = unsafe extern "C" fn(args: *mut PJRT_Event_OnReady_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Event_Await_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub event: *mut PJRT_Event,
    }

    impl PJRT_Event_Await_Args {
        pub fn new(event: *mut PJRT_Event) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), event }
        }
    }

    pub type PJRT_Event_Await = unsafe extern "C" fn(args: *mut PJRT_Event_Await_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Event_Error_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub event: *mut PJRT_Event,
    }

    impl PJRT_Event_Error_Args {
        pub fn new(event: *mut PJRT_Event) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), event }
        }
    }

    pub type PJRT_Event_Error = unsafe extern "C" fn(args: *mut PJRT_Event_Error_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Event_Destroy_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub event: *mut PJRT_Event,
    }

    impl PJRT_Event_Destroy_Args {
        pub fn new(event: *mut PJRT_Event) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), event }
        }
    }

    pub type PJRT_Event_Destroy = unsafe extern "C" fn(args: *mut PJRT_Event_Destroy_Args) -> *mut PJRT_Error;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_AsyncTrackingEvent {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct PJRT_Device_CreateAsyncTrackingEvent_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub device: *mut PJRT_Device,
        pub description: *const std::ffi::c_char,
        pub description_size: usize,
        pub event: *mut PJRT_AsyncTrackingEvent,
    }

    impl PJRT_Device_CreateAsyncTrackingEvent_Args {
        pub fn new(device: *mut PJRT_Device, description: *const std::ffi::c_char, description_size: usize) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                device,
                description,
                description_size,
                event: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Device_CreateAsyncTrackingEvent =
        unsafe extern "C" fn(args: *mut PJRT_Device_CreateAsyncTrackingEvent_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_AsyncTrackingEvent_Destroy_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub event: *mut PJRT_AsyncTrackingEvent,
    }

    impl PJRT_AsyncTrackingEvent_Destroy_Args {
        pub fn new(event: *mut PJRT_AsyncTrackingEvent) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), event }
        }
    }

    pub type PJRT_AsyncTrackingEvent_Destroy =
        unsafe extern "C" fn(args: *mut PJRT_AsyncTrackingEvent_Destroy_Args) -> *mut PJRT_Error;
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;

    use crate::tests::test_cpu_client;
    use crate::{Error, Event};

    #[test]
    fn test_event() {
        // Test [`Client::event`].
        let client = test_cpu_client();
        assert!(client.event(42i64).is_ok());
        assert!(client.event("test payload".to_string()).is_ok());
        assert!(client.event(vec![1, 2, 3]).is_ok());
        assert!(client.event(()).is_ok());

        // Test [`Event::set`], [`Event::ready`], [`Event::on_ready`], and [`Event::error`].
        let error = Error::aborted("Test");
        let mut has_error = false;
        let mut event = client.event(42i64).unwrap();
        assert!(event.on_ready(|error| { has_error = error.is_some() }).is_ok());
        assert_eq!(event.ready(), Ok(false));
        assert_eq!(has_error, false);
        assert!(event.set(None).is_ok());
        assert_eq!(event.ready(), Ok(true));
        assert_eq!(has_error, false);

        // Test [`Event::set`], [`Event::ready`], [`Event::on_ready`], and [`Event::error`] with an error.
        let mut event = client.event("test").unwrap();
        assert!(event.on_ready(|error| { has_error = error.is_some() }).is_ok());
        assert_eq!(event.ready(), Ok(false));
        assert_eq!(has_error, false);
        assert!(event.set(Some(error.clone())).is_ok());
        assert_eq!(event.ready(), Ok(true));
        assert_eq!(has_error, true);
        let event_error = event.error().unwrap();
        assert!(event_error.is_some());
        let event_error = event_error.unwrap();
        assert_eq!(event_error.code(), error.code());
        assert_eq!(event_error.message(), error.message());

        // Test [`Event::await`].
        let mut has_invoked_callback = false;
        let event = client.event(42i64).unwrap();
        assert!(event.on_ready(|_| has_invoked_callback = true).is_ok());
        assert_eq!(has_invoked_callback, false);
        assert_eq!(event.ready(), Ok(false));

        let mut event_handle = unsafe { event.handle() };
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            assert!(unsafe { event_handle.set(None) }.is_ok());
        });

        assert_eq!(has_invoked_callback, false);
        assert_eq!(event.r#await(), Ok(42i64));
        assert_eq!(has_invoked_callback, true);

        // Test creating an [`Event`] from a null pointer.
        assert!(matches!(
            unsafe { Event::from_c_api(std::ptr::null_mut(), client.api(), ()) },
            Err(Error::InvalidArgument { message, .. })
                if message == "the provided PJRT event handle is a null pointer",
        ));
    }

    #[test]
    fn test_event_future() {
        let client = test_cpu_client();
        let mut has_invoked_callback = false;
        let mut event = client.event(42i64).unwrap();
        assert!(event.on_ready(|_| has_invoked_callback = true).is_ok());
        assert_eq!(has_invoked_callback, false);
        assert_eq!(event.ready(), Ok(false));

        let mut event_handle = unsafe { event.handle() };
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            assert!(unsafe { event_handle.set(None) }.is_ok());
        });

        assert_eq!(has_invoked_callback, false);
        assert_eq!(block_on(&mut event), Ok(42i64));

        // Adding a short "sleep" to make sure that the "on-ready" callback is invoked before the next check.
        std::thread::sleep(std::time::Duration::from_millis(100));

        assert_eq!(has_invoked_callback, true);
        assert_eq!(event.ready(), Ok(true));
    }

    // TODO(eaplatanios): Add tests for async tracking events.
}
