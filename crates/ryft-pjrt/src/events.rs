use std::cell::Cell;
use std::future::Future;
use std::marker::PhantomData;
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
///
/// # Thread Safety
///
/// [`Event`]s are [`Send`] but intentionally not [`Sync`]. The PJRT C API only requires event implementations to
/// support queries, awaits, and callback registrations that overlap the event's completion, not consumer calls that
/// overlap each other. An [`Event`] can therefore be moved freely across threads but is consumed from one thread at
/// a time, while an [`EventPromise`] can set/trigger it from any other thread.
pub struct Event<O> {
    /// Shared ownership and synchronization [`EventState`] for the underlying PJRT event.
    state: Arc<EventState>,

    /// "Payload" that this [`Event`] carries. Specifically, this payload will be [`Some`] throughout the lifetime of
    /// this [`Event`] and will be returned when it completes (i.e., when its underlying computation completes) via
    /// [`Event::await`] or [`Event::poll`].
    output: Option<O>,

    /// Indicates whether a native "on-ready" callback that wakes the [`Waker`] stored in [`EventState::waker`] has
    /// already been registered for the underlying PJRT event. This is only ever accessed by [`Event::poll`], whose
    /// exclusive borrow of this [`Event`] already serializes it, and it is reset to `false` when a registration attempt
    /// fails so that a later [`Event::poll`] invocation can retry the registration.
    callback_registered: Cell<bool>,

    /// Marker that keeps [`Event`] [`Send`] but not [`Sync`]. Refer to the *Thread Safety* section
    /// of the [`Event`] documentation for more information.
    marker: PhantomData<Cell<()>>,
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
            Ok(Self {
                state: Arc::new(EventState { api, handle: EventHandle(handle), waker: Arc::new(Mutex::new(None)) }),
                output: Some(output),
                callback_registered: Cell::new(false),
                marker: PhantomData,
            })
        }
    }

    /// Checks if the underlying computation of this [`Event`] has finished executing and returns `true` if it has and
    /// `false` otherwise. Note that an [`Error`] may also be returned if something goes wrong while checking on the
    /// [`Event`]'s status.
    pub fn ready(&self) -> Result<bool, Error> {
        use ffi::PJRT_Event_IsReady_Args;
        invoke_pjrt_api_error_fn!(self.state.api, PJRT_Event_IsReady, { event = self.state.handle.0 }, { is_ready })
    }

    /// Registers the provided callback to be invoked when the underlying computation of this [`Event`] finishes
    /// executing. The callback takes an optional [`Error`] as its sole argument whose value depends on whether the
    /// underlying computation produced an error or not. PJRT may invoke the callback on a runtime-owned thread, so
    /// callback state must be safe to send between threads and owned for the full asynchronous lifetime.
    pub fn on_ready<F: FnOnce(Option<Error>) + Send + 'static>(&self, callback: F) -> Result<(), Error> {
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
            { event = self.state.handle.0, callback = callback_fn::<F>, user_arg = callback_fn_arg as *mut _ },
        )
        .inspect_err(|_| drop(unsafe { Box::from_raw(callback_fn_arg) }))
    }

    /// Blocks the current thread until this [`Event`] is _ready_, returning an [`Error`] if something went wrong.
    pub fn r#await(self) -> Result<O, Error> {
        use ffi::PJRT_Event_Await_Args;

        // It is safe to force-unwrap the output option because it is always going to be `Some` unless
        // this function has been called, and this function consumes `self`.
        let mut event = self;
        invoke_pjrt_api_error_fn!(event.state.api, PJRT_Event_Await, { event = event.state.handle.0 })
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
            Ok(invoke_pjrt_api_error_fn!(self.state.api, PJRT_Event_Error, { event = self.state.handle.0 }).err())
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
                *self.state.waker.lock().expect("PJRT event waker mutex poisoned") = Some(cx.waker().clone());
                let callback_registration_result = (!self.callback_registered.get()).then(|| {
                    self.callback_registered.set(true);
                    // The callback must capture only the shared waker slot, never the whole `EventState`. The native
                    // event owns the registered callback, so a callback that owned an `Arc<EventState>` would form a
                    // reference cycle (i.e., `EventState` -> native event -> callback -> `EventState`) that leaks the
                    // native event whenever it never completes (e.g., when a pending `Event` is canceled after its
                    // `EventPromise` was dropped without being set).
                    let waker = self.state.waker.clone();
                    self.on_ready(move |_| {
                        let waker = waker.lock().expect("PJRT event waker mutex poisoned").take();
                        if let Some(waker) = waker {
                            waker.wake();
                        }
                    })
                });
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
                    Some(Err(error)) => {
                        self.callback_registered.set(false);
                        Poll::Ready(Err(error))
                    }
                    None => Poll::Pending,
                }
            }
            Err(error) => Poll::Ready(Err(error)),
        }
    }
}

impl<O> Drop for Event<O> {
    fn drop(&mut self) {
        // We clear any waker registered by `Future::poll` so that an outstanding native "on-ready" callback does not
        // retain a canceled executor task indefinitely if the underlying computation never completes. The native event
        // itself is destroyed by `EventState`'s `Drop` implementation once all owners have been dropped.
        *self.state.waker.lock().expect("PJRT event waker mutex poisoned") = None;
    }
}

/// Promise half of an [`Event`] that can be used to set/trigger that event from another thread (mirroring the
/// `xla::Promise` that backs each event created through `PJRT_Event_Create` in the standard PJRT C API implementation).
/// An [`EventPromise`] shares ownership of the underlying PJRT event, keeping it alive until both the [`Event`] and its
/// promise have been dropped.
///
/// Each [`Event`] created through [`Client::event`] or [`Plugin::event`] is paired with exactly one [`EventPromise`],
/// and [`EventPromise::set`] consumes it, so each event can be set at most once by construction, matching the
/// requirement that the `xla::Promise` backing it is fulfilled at most once. Events returned by other PJRT APIs
/// (e.g., buffer, execution, and transfer completion events) are completed by the PJRT runtime and have no promise,
/// because the PJRT C API restricts `PJRT_Event_Set` to events created through `PJRT_Event_Create`. Note that dropping
/// an [`EventPromise`] without setting it leaves the associated [`Event`] pending forever.
pub struct EventPromise {
    /// Shared ownership and synchronization [`EventState`] for the underlying PJRT event.
    state: Arc<EventState>,
}

impl EventPromise {
    /// Sets/triggers the [`Event`] associated with this [`EventPromise`] to indicate that the work it represents has
    /// completed successfully, consuming this promise. If an [`Error`] is provided, it will be returned when the
    /// associated [`Event`] is polled via [`Event::await`] or [`Event::poll`], representing that something went wrong
    /// while executing the underlying work. Otherwise, the work will be considered successful.
    ///
    /// Note that this promise is consumed even when the underlying PJRT call reports an error, because the PJRT C API
    /// does not guarantee that a failed `PJRT_Event_Set` left the event unfulfilled, and so retrying could fulfill the
    /// backing `xla::Promise` twice.
    pub fn set(self, error: Option<Error>) -> Result<(), Error> {
        use ffi::PJRT_Event_Set_Args;
        let error_message = error.as_ref().map(|error| error.message());
        invoke_pjrt_api_error_fn!(
            self.state.api,
            PJRT_Event_Set,
            {
                client = self.state.handle.0,
                error_code = error.as_ref().map(|error| error.code()).unwrap_or(crate::errors::ffi::PJRT_Error_Code_OK),
                error_message = error_message.as_ref().map(|message| message.as_ptr()).unwrap_or(std::ptr::null()),
                error_message_size = error_message.as_ref().map(|message| message.count_bytes()).unwrap_or(0),
            },
        )
    }
}

impl Client<'_> {
    /// Creates a new [`Event`] that carries with it the provided `output` and can be used to track the completion of an
    /// asynchronous computation, along with the [`EventPromise`] used to set/trigger it. `output` will be returned when
    /// the underlying computation completes. For example, `output` could be a buffer and the [`Event`] could be used to
    /// track when the work done to populate that buffer is complete, with the [`EventPromise`] being set once the work
    /// populating that buffer finishes.
    #[inline]
    pub fn event<O>(&self, output: O) -> Result<(Event<O>, EventPromise), Error> {
        self.api().event(output)
    }
}

impl Plugin {
    /// Creates a new [`Event`] that carries with it the provided `output` and can be used to track the completion of an
    /// asynchronous computation, along with the [`EventPromise`] used to set/trigger it. `output` will be returned when
    /// the underlying computation completes. For example, `output` could be a buffer and the [`Event`] could be used to
    /// track when the work done to populate that buffer is complete, with the [`EventPromise`] being set once the work
    /// populating that buffer finishes.
    pub fn event<O>(&self, output: O) -> Result<(Event<O>, EventPromise), Error> {
        self.api().event(output)
    }
}

impl Api {
    /// Creates a new [`Event`] that carries with it the provided `output` and can be used to track the completion of an
    /// asynchronous computation, along with the [`EventPromise`] used to set/trigger it. `output` will be returned when
    /// the underlying computation completes. For example, `output` could be a buffer and the [`Event`] could be used to
    /// track when the work done to populate that buffer is complete, with the [`EventPromise`] being set once the work
    /// populating that buffer finishes.
    pub(crate) fn event<O>(&self, output: O) -> Result<(Event<O>, EventPromise), Error> {
        use ffi::PJRT_Event_Create_Args;
        // Events created through `PJRT_Event_Create` are the only events that the PJRT C API allows to be
        // set/triggered, and so this is the only place where an `EventPromise` is created.
        let handle = invoke_pjrt_api_error_fn!(*self, PJRT_Event_Create, {}, { event })?;
        let event = unsafe { Event::from_c_api(handle, *self, output) }?;
        let promise = EventPromise { state: event.state.clone() };
        Ok((event, promise))
    }
}

/// Raw PJRT event handle whose ownership is managed by [`EventState`]. This private wrapper narrows the concurrency
/// requirements described in the safety comment below to the raw pointer so that [`EventState`] and [`EventPromise`]
/// obtain [`Send`] and [`Sync`] structurally, while [`Event`] obtains [`Send`] and stays deliberately not [`Sync`]
/// through its own marker field.
#[derive(Copy, Clone)]
struct EventHandle(*mut ffi::PJRT_Event);

// The PJRT C API does not state a blanket thread-safety guarantee for events, but its event semantics require queries,
// awaits, and callback registrations to tolerate overlapping the event's completion: `PJRT_Event_Await` blocks the
// calling thread until the event is made ready by work completing elsewhere (a PJRT runtime thread, or a
// `PJRT_Event_Set` call from another thread), and `PJRT_Event_OnReady` callbacks are invoked from whichever thread
// completes the event. These implementations expose exactly that concurrency and nothing more. Consumer-side native
// calls are serialized because `Event` is deliberately `Send` but not `Sync`, the sole caller-driven completion
// (i.e., `PJRT_Event_Set`) is limited to at most one native call because the non-clonable `EventPromise` is consumed
// by `EventPromise::set`, and destruction is exclusive because it only happens once all owners have been dropped.
unsafe impl Send for EventHandle {}
unsafe impl Sync for EventHandle {}

/// Shared state that owns the native PJRT event on behalf of an [`Event`] and its [`EventPromise`], destroying
/// that native event when the last owner is dropped.
struct EventState {
    /// Underlying PJRT [`Api`].
    api: Api,

    /// [`EventHandle`] for the underlying native PJRT event.
    handle: EventHandle,

    /// [`Waker`] slot shared with the "on-ready" callback that [`Event::poll`] registers, used to integrate the event
    /// with Rust's [`Future`] protocol. It is shared through its own [`Arc`] so that the callback never owns the whole
    /// [`EventState`]. The native event owns that callback, so a callback owning an [`Arc<EventState>`] would form a
    /// reference cycle that leaks the native event whenever it never completes.
    waker: Arc<Mutex<Option<Waker>>>,
}

impl Drop for EventState {
    fn drop(&mut self) {
        use ffi::PJRT_Event_Destroy_Args;
        invoke_pjrt_api_error_fn!(self.api, PJRT_Event_Destroy, { event = self.handle.0 })
            .expect("failed to destroy PJRT event");
    }
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
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::task::{Context, Poll};

    use futures::executor::block_on;
    use futures::task::noop_waker_ref;

    use crate::tests::test_cpu_client;
    use crate::{Error, Event};

    fn assert_send<T: Send>() {}

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn test_event() {
        // `Event` is deliberately `Send` but not `Sync` (refer to its *Thread Safety* documentation section).
        assert_send::<Event<()>>();
        assert_send_sync::<super::EventPromise>();

        // Test `Client::event`.
        let client = test_cpu_client();
        assert!(client.event(42i64).is_ok());
        assert!(client.event("test payload".to_string()).is_ok());
        assert!(client.event(vec![1, 2, 3]).is_ok());
        assert!(client.event(()).is_ok());

        // Test `EventPromise::set`, `Event::ready`, `Event::on_ready`, and `Event::error`.
        let error = Error::aborted("Test");
        let has_error = Arc::new(AtomicBool::new(false));
        let (event, promise) = client.event(42i64).unwrap();
        let callback_has_error = has_error.clone();
        assert!(event.on_ready(move |error| callback_has_error.store(error.is_some(), Ordering::Relaxed)).is_ok());
        assert_eq!(event.ready(), Ok(false));
        assert!(!has_error.load(Ordering::Relaxed));
        assert!(promise.set(None).is_ok());
        assert_eq!(event.ready(), Ok(true));
        assert!(!has_error.load(Ordering::Relaxed));

        // Test `EventPromise::set`, `Event::ready`, `Event::on_ready`, and `Event::error` with an error.
        let (event, promise) = client.event("test").unwrap();
        let callback_has_error = has_error.clone();
        assert!(event.on_ready(move |error| callback_has_error.store(error.is_some(), Ordering::Relaxed)).is_ok());
        assert_eq!(event.ready(), Ok(false));
        assert!(!has_error.load(Ordering::Relaxed));
        assert!(promise.set(Some(error.clone())).is_ok());
        assert_eq!(event.ready(), Ok(true));
        assert!(has_error.load(Ordering::Relaxed));
        let event_error = event.error().unwrap();
        assert!(event_error.is_some());
        let event_error = event_error.unwrap();
        assert_eq!(event_error.code(), error.code());
        assert_eq!(event_error.message(), error.message());

        // Test `Event::await`.
        let has_invoked_callback = Arc::new(AtomicBool::new(false));
        let (event, event_promise) = client.event(42i64).unwrap();
        let callback_invoked = has_invoked_callback.clone();
        assert!(event.on_ready(move |_| callback_invoked.store(true, Ordering::Relaxed)).is_ok());
        assert!(!has_invoked_callback.load(Ordering::Relaxed));
        assert_eq!(event.ready(), Ok(false));

        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            assert!(event_promise.set(None).is_ok());
        });

        assert!(!has_invoked_callback.load(Ordering::Relaxed));
        assert_eq!(event.r#await(), Ok(42i64));
        assert!(has_invoked_callback.load(Ordering::Relaxed));

        // Test creating an `Event` from a null pointer.
        assert!(matches!(
            unsafe { Event::from_c_api(std::ptr::null_mut(), client.api(), ()) },
            Err(Error::InvalidArgument { message, .. })
                if message == "the provided PJRT event handle is a null pointer",
        ));
    }

    #[test]
    fn test_event_promise_keeps_event_and_callback_alive_across_threads() {
        let client = test_cpu_client();
        let (event, promise) = client.event(()).unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        event.on_ready(move |error| sender.send(error).unwrap()).unwrap();
        drop(event);
        std::thread::spawn(move || promise.set(None).unwrap()).join().unwrap();
        assert!(receiver.recv_timeout(std::time::Duration::from_secs(1)).unwrap().is_none());
    }

    #[test]
    fn test_cancelled_pending_event_releases_its_state() {
        let client = test_cpu_client();
        let (mut event, promise) = client.event(()).unwrap();
        let mut context = Context::from_waker(noop_waker_ref());
        assert!(matches!(Pin::new(&mut event).poll(&mut context), Poll::Pending));

        // The "on-ready" callback registered by the poll above must retain only the shared waker slot. If it retained
        // the whole `EventState`, the state, the native event, and the callback would form a reference cycle that leaks
        // the native event when a pending event is cancelled without ever being fulfilled.
        let state = Arc::downgrade(&event.state);
        drop(promise);
        drop(event);
        assert!(state.upgrade().is_none());
    }

    #[test]
    fn test_event_future() {
        let client = test_cpu_client();
        let has_invoked_callback = Arc::new(AtomicBool::new(false));
        let (mut event, event_promise) = client.event(42i64).unwrap();
        let callback_invoked = has_invoked_callback.clone();
        assert!(event.on_ready(move |_| callback_invoked.store(true, Ordering::Relaxed)).is_ok());
        assert!(!has_invoked_callback.load(Ordering::Relaxed));
        assert_eq!(event.ready(), Ok(false));

        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            assert!(event_promise.set(None).is_ok());
        });

        assert!(!has_invoked_callback.load(Ordering::Relaxed));
        assert_eq!(block_on(&mut event), Ok(42i64));

        // Adding a short "sleep" to make sure that the "on-ready" callback is invoked before the next check.
        std::thread::sleep(std::time::Duration::from_millis(100));

        assert!(has_invoked_callback.load(Ordering::Relaxed));
        assert_eq!(event.ready(), Ok(true));
    }

    // TODO(eaplatanios): Add tests for async tracking events.
}
