// //! TODO(eaplatanios): Write a docstring here.
// //!
// //! Refer to the related [JAX documentation page](https://jax.readthedocs.io/en/latest/transfer_guard.html) for more
// //! information on the concept of transfer guards and the implications of the various configuration options.
// 
// use std::backtrace::Backtrace;
// use std::cell::Cell;
// use std::fmt::{Display, Formatter};
// use std::sync::Mutex;
// use tracing::info;
// 
// use crate::errors::Error;
// use crate::status::StatusCode;
// 
// // TODO(eaplatanios): Add support for configuring the transfer guards.
// 
// static GLOBAL_TRANSFER_GUARD_STATE: Mutex<TransferGuardState> = Mutex::new(TransferGuardState {
//     host_to_device: None,
//     device_to_device: None,
//     device_to_host: None,
//     explicit_device_put: false,
//     explicit_device_get: false,
// });
// 
// thread_local! {
//     static THREAD_LOCAL_TRANSFER_GUARD_STATE: Cell<TransferGuardState> = Cell::new(TransferGuardState::default());
// }
// 
// /// Transfer guard level to use.
// #[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
// pub enum TransferGuardLevel {
//     /// Allows both explicit and implicit transfers.
//     #[default]
//     Allow,
// 
//     /// Allows explicit transfers and logs messages for implicit transfers.
//     Log,
// 
//     /// Allows explicit transfers and disallows implicit transfers.
//     Disallow,
// 
//     /// Logs messages for both explicit and implicit transfers.
//     LogExplicit,
// 
//     /// Disallows both explicit and implicit transfers.
//     DisallowExplicit,
// }
// 
// impl TransferGuardLevel {
//     /// Guards a device transfer based on this [TransferGuardLevel] along with whether the transfer is explicit or
//     /// implicit. A device transfer is explicit when it is explicitly initiated by a user with a "device put" or a
//     /// "device get" call, or implicitly via, e.g., a call to print an array.
//     ///
//     /// # Arguments
//     ///
//     ///   * `description` - Description of the transfer to use for logging and error messages.
//     ///   * `explicit_transfer` - Boolean flag indicating whether the transfer to guard is explicit or implicit.
//     fn guard_transfer(&self, description: &str, kind: TransferKind, explicit: bool) -> Result<(), Error> {
//         match self {
//             TransferGuardLevel::Allow => Ok(()),
//             TransferGuardLevel::Log if explicit => Ok(()),
//             TransferGuardLevel::Disallow if explicit => Ok(()),
//             TransferGuardLevel::Log | TransferGuardLevel::LogExplicit => {
//                 let log_message = format!("Allowed '{kind}' transfer.");
//                 info!(log_message, description = description);
//                 Ok(())
//             }
//             TransferGuardLevel::Disallow | TransferGuardLevel::DisallowExplicit => Err(Error::XlaError {
//                 code: StatusCode::InvalidArgument,
//                 message: format!("Disallowed '{kind}' device transfer {description}."),
//                 backtrace: Backtrace::capture().to_string(),
//             }),
//         }
//     }
// }
// 
// /// Kind of a device transfer.
// #[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
// pub enum TransferKind {
//     /// Transfer of data from the host to an accelerator device.
//     HostToDevice,
// 
//     /// Transfer of data from an accelerator device to an accelerator device.
//     DeviceToDevice,
// 
//     /// Transfer of data from an accelerator device to the host.
//     DeviceToHost,
// }
// 
// impl Display for TransferKind {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         match self {
//             TransferKind::HostToDevice => f.write_str("host-to-device"),
//             TransferKind::DeviceToDevice => f.write_str("device-to-device"),
//             TransferKind::DeviceToHost => f.write_str("device-to-host"),
//         }
//     }
// }
// 
// // TODO(eaplatanios): [DOC].
// // Flags for transfer guard levels are controlled by:
// // - a global flag value,
// //   e.g., associated to --jax_transfer_guard_device_to_host
// //   which defaults to TransferGuardLevel::kAllow.
// // - possibly a thread-local value, which initially is std::nullopt and
// //   overrides the global value if set. The thread-local state is used to
// //   implement context managers that locally override the global state.
// //
// // Explicit device_put/device_get contexts are tracked by context managers.
// #[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
// struct TransferGuardState {
//     host_to_device: Option<TransferGuardLevel>,
//     device_to_device: Option<TransferGuardLevel>,
//     device_to_host: Option<TransferGuardLevel>,
//     explicit_device_put: bool,
//     explicit_device_get: bool,
// }
// 
// impl TransferGuardState {
//     fn get_level(kind: TransferKind) -> TransferGuardLevel {
//         match kind {
//             TransferKind::HostToDevice => THREAD_LOCAL_TRANSFER_GUARD_STATE
//                 .get()
//                 .host_to_device
//                 .or_else(|| GLOBAL_TRANSFER_GUARD_STATE.lock().unwrap().host_to_device)
//                 .unwrap_or(TransferGuardLevel::default()),
//             TransferKind::DeviceToDevice => THREAD_LOCAL_TRANSFER_GUARD_STATE
//                 .get()
//                 .device_to_device
//                 .or_else(|| GLOBAL_TRANSFER_GUARD_STATE.lock().unwrap().device_to_device)
//                 .unwrap_or(TransferGuardLevel::default()),
//             TransferKind::DeviceToHost => THREAD_LOCAL_TRANSFER_GUARD_STATE
//                 .get()
//                 .device_to_host
//                 .or_else(|| GLOBAL_TRANSFER_GUARD_STATE.lock().unwrap().device_to_host)
//                 .unwrap_or(TransferGuardLevel::default()),
//         }
//     }
// }
// 
// /// Guards a data transfer, allowing it to take place, optionally logging that it is taking place, or disallowing it.
// /// The last case results in this function returning an [Error].
// ///
// /// # Arguments
// ///
// ///   * `kind` - [TransferKind] of the transfer to guard.
// ///   * `description` - Description of the guarded transfer to be used for logging or error messages.
// pub(crate) fn guard_transfer(kind: TransferKind, description: &str) -> Result<(), Error> {
//     TransferGuardState::get_level(kind).guard_transfer(
//         description,
//         kind,
//         THREAD_LOCAL_TRANSFER_GUARD_STATE.get().explicit_device_put,
//     )
// }
