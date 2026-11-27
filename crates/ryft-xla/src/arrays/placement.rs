//! Placement of arrays on devices, which is the `ryft` analogue of JAX's
//! [`jax.device_put`](https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html).
//!
//! # Placement Entry Points
//!
//!   - [`Session::place`] accepts a [`Placeable`] [`Parameterized`], whose parameters are host values that it uploads,
//!     runtime [`Array`]s, and traced values of compiled, differentiated, or batched functions, and it takes requests
//!     or resolved placements through [`PlacementOptions`].
//!   - [`Array::to_placement`] and [`Array::into_placement`] move one runtime [`Array`], the latter donating its
//!     buffers.
//!   - The core [`Place`](ryft_core::Place) capability places any value that implements it, including traced values
//!     in every context. It records each form of [`PlacementRequest`] as a target or source without resolving it
//!     against a session, including requests given as a [`Placement`], [`Device`], or [`Memory`](ryft_core::Memory).
//!
//! [`PlacementRequest`] defines the target and source forms. [`Session::placement`] pairs a sharding with the
//! session's default mesh. A traced value records a [`PlaceOperation`], whose type rule applies to concrete and
//! abstract traced values alike. The operation resolves device and in-place requests against the array when it
//! executes eagerly, and lowers to sharding, layout, and memory constraints when compiled. A bare memory leaves a
//! value on its devices with its sharding intact, which `jax.device_put(x, jax.memory.Space.Host)` does not guarantee.
//! See [`Session::place`] for tree broadcasting, commitment, and the rules for compiled functions and transforms.
//!
//! Runtime placement usually returns before its transfers complete. See [`crate::arrays::transfers`] for the
//! transfer strategies, blocking cases, and failures reported by readiness and readback functions.

use std::borrow::Cow;
use std::rc::Rc;
use std::sync::{Arc, Weak};

use ryft_core::{
    Array as CoreArray, ArrayType, BatchingPolicy, BatchingTracer, CapturePlacement, CapturePlacementContext, Context,
    Device, DeviceMesh, DifferentiationPolicy, DifferentiationTracer, ImportingContext, LogicalMesh, Parameter,
    ParameterPath, Parameterized, ParameterizedFamily, PartialTracer, PlaceDispatch, PlaceOperation, Placeholder,
    Placement, PlacementRequest, ProgramError, ProjectedValue, ThroughCapture, ThroughParent, Tracer, Typed,
    UnreducedKind, Value as CoreValue,
};
use ryft_macros::Parameter;
use ryft_pjrt::{Buffer, Device as PjrtDevice, Error as PjrtError};

use crate::arrays::host::{
    HostValue, ValueComparison, array_value_digest, combined_digest, dense_host_value_bytes, validate_mesh_devices,
};
use crate::arrays::shards::ArrayShard;
use crate::arrays::transfers::devices_span_processes;
use crate::arrays::{Array, ArrayError};
use crate::compilation::domains::captured_type;
use crate::devices::PjrtDeviceExtension;
use crate::sessions::effects::ensure_effect_dispatch_is_allowed;
use crate::{CompileTracer, Error, FromPjrt, Session, Value};

use sealed::PlaceableParameter;

/// Options of [`Session::place`]. The four type parameters describe whole tree prefixes. Each option can be a single
/// value that applies to the entire input, or a tuple or other parameterized tree that sets different values for
/// different branches. [`Session::place`] broadcasts these prefixes over the input and applies the resulting options
/// leafwise:
///
///   - `Target` describes where to place each input, and `Source` describes its expected placement. Their leaves
///     implement [`SessionPlacementRequestParameter`]. A consumed [`Sharding`](ryft_core::Sharding) resolves against
///     the session's default mesh. To mix resolved placements and explicit requests in a prefix, use
///     [`PlacementRequest::from`].
///   - `Donate` has `bool` leaves. A `true` leaf permits reusing the input array's buffers for the result when donation
///     is possible.
///   - `MayAlias` has leaves that convert into `Option<bool>`. A `false` or `Some(false)` leaf requests a copy, `true`
///     or `Some(true)` permits sharing, and `None` leaves the policy unspecified for that branch.
///
/// Each field is optional, so `None` can omit an entire prefix. Omitting `donate` disables donation, and omitting
/// `may_alias` leaves the policy unspecified. `Donate` and `MayAlias` both default to the single-leaf type `bool`,
/// giving both fields type `Option<bool>` by default. An explicit aliasing tree can use `Option<bool>` leaves to set
/// a policy for some branches and leave others unspecified.
///
/// # Example
///
/// For a two-element input, these options request donation of the first element and a copy of the second:
///
/// ```rust
/// # use ryft_xla::PlacementOptions;
/// let options = PlacementOptions::default()
///     .with_donate((true, false))
///     .with_may_alias((None::<bool>, Some(false)));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlacementOptions<Target = PlacementRequest, Source = PlacementRequest, Donate = bool, MayAlias = bool> {
    /// Target placement tree prefix. When absent, host leaves are uploaded to the default device of the session
    /// (refer to [`Session::with_default_device`]) or, without one, to the first addressable device of its client,
    /// whether or not the session has a default mesh, without being [committed](Array::is_committed) to it, as
    /// devices that this process chose on its own (refer to [`Array::is_locally_assigned`]), and [`Array`] leaves and
    /// traced leaves preserve their current placement and commitment.
    target: Option<Target>,

    /// Source placement tree prefix. Bare sharding leaves resolve against the session's default mesh. The resolved
    /// source is validated for [`Array`] leaves and ignored for eager host leaves.
    source: Option<Source>,

    /// Donation tree prefix. A donated [`Array`] leaf lets a compiled reshard reuse its buffers for the result. The
    /// donation only takes effect when the leaf is the only handle of its buffers. Only concrete [`Array`] leaves
    /// apply it, and host leaves and traced leaves ignore it. A leaf cannot be donated while its aliasing option is
    /// `Some(true)`, and a donated leaf ignores an aliasing option of `Some(false)`, because donation gives the leaf
    /// up, as in JAX (refer to [`Session::place`]).
    donate: Option<Donate>,

    /// Aliasing tree prefix. With `false` or `Some(false)` and without donation, a resulting array shares no buffer
    /// with any other array, which copies the buffers that it would otherwise share. Concrete [`Array`] leaves apply it
    /// when they are placed, and traced leaves stage their placements with a forced copy, which applies it when those
    /// placements execute on concrete arrays (e.g., when a batched or differentiated function places concrete arrays).
    /// Host leaves ignore it, because their uploads never share buffers. Refer to [`Session::place`] for more
    /// information.
    may_alias: Option<MayAlias>,
}

impl<Target, Source, Donate, MayAlias> PlacementOptions<Target, Source, Donate, MayAlias> {
    /// Creates a new [`PlacementOptions`].
    #[inline]
    pub fn new(
        target: Option<Target>,
        source: Option<Source>,
        donate: Option<Donate>,
        may_alias: Option<MayAlias>,
    ) -> Self {
        Self { target, source, donate, may_alias }
    }

    /// Returns these options with the target tree prefix replaced by `target`, which can have another type than the
    /// current one. Each other option keeps its type, so `PlacementOptions::default().with_target(device)` uses the
    /// standard types for the other options without annotations.
    #[inline]
    pub fn with_target<NewTarget>(self, target: NewTarget) -> PlacementOptions<NewTarget, Source, Donate, MayAlias> {
        PlacementOptions::new(Some(target), self.source, self.donate, self.may_alias)
    }

    /// Returns these options with the source tree prefix replaced by `source`, which can have another type than the
    /// current one. The type of each other option is kept, as in [`Self::with_target`].
    #[inline]
    pub fn with_source<NewSource>(self, source: NewSource) -> PlacementOptions<Target, NewSource, Donate, MayAlias> {
        PlacementOptions::new(self.target, Some(source), self.donate, self.may_alias)
    }

    /// Returns these options with the donation tree prefix replaced by `donate`, which can have another type than the
    /// current one. The type of each other option is kept, as in [`Self::with_target`]
    /// (e.g., `PlacementOptions::to(device).with_donate(true)`).
    #[inline]
    pub fn with_donate<NewDonate>(self, donate: NewDonate) -> PlacementOptions<Target, Source, NewDonate, MayAlias> {
        PlacementOptions::new(self.target, self.source, Some(donate), self.may_alias)
    }

    /// Returns these options with the aliasing tree prefix replaced by `may_alias`, which can have another type than
    /// the current one. The type of each other option is kept, as in [`Self::with_target`], so that, for example,
    /// `PlacementOptions::default().with_may_alias(false)` requests a forced copy without a target and without
    /// naming the type of a target. Both `false` and `Some(false)` request a forced copy, and [`None`] leaves
    /// the option unspecified.
    #[inline]
    pub fn with_may_alias<NewMayAlias>(
        self,
        may_alias: NewMayAlias,
    ) -> PlacementOptions<Target, Source, Donate, NewMayAlias> {
        PlacementOptions::new(self.target, self.source, self.donate, Some(may_alias))
    }

    /// Returns the target placement tree prefix, if one was provided.
    #[inline]
    pub fn target(&self) -> Option<&Target> {
        self.target.as_ref()
    }

    /// Returns the source placement tree prefix, if one was provided.
    #[inline]
    pub fn source(&self) -> Option<&Source> {
        self.source.as_ref()
    }

    /// Returns the donation tree prefix, if one was provided.
    #[inline]
    pub fn donate(&self) -> Option<&Donate> {
        self.donate.as_ref()
    }

    /// Returns the aliasing tree prefix, if one was provided.
    #[inline]
    pub fn may_alias(&self) -> Option<&MayAlias> {
        self.may_alias.as_ref()
    }

    /// Returns the target, source, donation, and aliasing tree prefixes of these options.
    fn into_parts(self) -> (Option<Target>, Option<Source>, Option<Donate>, Option<MayAlias>) {
        (self.target, self.source, self.donate, self.may_alias)
    }
}

impl<Target> PlacementOptions<Target> {
    /// Creates [`Session::place`] options with only the target tree prefix `target` (e.g., a [`Device`], a
    /// [`Memory`](ryft_core::Memory), a [`PlacementRequest`], or a tuple of optional placements). The other options
    /// use the standard prefix types and are unset, so this needs no generic type annotations.
    #[inline]
    pub fn to(target: Target) -> Self {
        Self::new(Some(target), None, None, None)
    }
}

// `Default` is implemented only for the standard prefix types. Type parameter defaults do not guide inference in
// expressions, so with an implementation for every choice of prefix types the unannotated `PlacementOptions::default()`
// would have four types to infer and nothing to infer them from. For example, in
// `session.place(value, PlacementOptions::default())`, or before a builder function that replaces a prefix type.
// With this single implementation the compiler has one candidate and picks it. Unset options with other prefix
// types are created with `PlacementOptions::new`.
impl Default for PlacementOptions {
    #[inline]
    fn default() -> Self {
        Self::new(None, None, None, None)
    }
}

/// Represents [`Parameter`] types accepted by [`Session::place`]. A [`Placeable`] consumes one input leaf and
/// produces one [`Self::Output`] leaf. Ryft provides implementations for runtime [`Array`] leaves, host scalars
/// of the [`ArrayElement`](ryft_core::ArrayElement) types that are [`Parameter`]s, and host arrays of the reference
/// backend (i.e., [`ryft_core::Array`]), which are uploaded as dense arrays whatever their host layout is, and all of
/// which produce runtime [`Array`] leaves. [`PlaceInput`] holds either kind of these leaves, including scalars of the
/// other element types (e.g., complex scalars), so that one tree can mix them. Ryft also provides implementations for
/// the traced values of `ryft-core` that can hold arrays of this backend (i.e., [`Tracer`], [`PartialTracer`],
/// [`DifferentiationTracer`], [`BatchingTracer`], and [`ProjectedValue`] over [`ArrayType`], which covers
/// [`CompileTracer`]), so that the same call works inside compiled functions, differentiation, and batching.
/// A traced leaf stays traced and records its placement through the [`PlaceDispatch`] primitive of the
/// [`Place`](ryft_core::Place) capability (i.e., by staging a [`PlaceOperation`] whose copy policy follows the
/// `may_alias` option of the leaf). Omitting the target leaves the leaf where it is. Those implementations require
/// the values that the traced values wrap to be ones whose concrete arrays this crate can find (i.e., [`Array`]s,
/// [`Value`]s, host arrays of the reference backend, and the traced values above), so that a traced leaf over arrays
/// of another PJRT client or of another session can be rejected.
///
/// [`Placeable::place_parameter`] is the per-leaf primitive of [`Session::place`], like
/// [`ReshardDispatch::reshard`](ryft_core::ReshardDispatch::reshard) is for resharding. [`Session::place`] also checks
/// every leaf of a tree before it places any, and it owns the comparison of the values that the processes of a run
/// place (refer to [`Session::place`]), which covers the host values and the locally assigned [`Array`]s. Both need
/// to know every kind of leaf, so this trait is sealed (i.e., only this crate implements it). A type that is made of
/// these leaves is placed as a [`Parameterized`] tree of them.
#[expect(private_bounds, reason = "the sealed supertrait handles arguments internal to placement")]
pub trait Placeable: Parameter + PlaceableParameter<<Self as Placeable>::Output> {
    /// Leaf type that placing this leaf produces, which is [`Array`] for host values and runtime arrays, and the leaf
    /// type itself for traced values.
    type Output: Parameter;

    /// Places this leaf using the provided leafwise placement options. Some combinations of options are invalid for
    /// every leaf (i.e., a `donate` of `true` with a `may_alias` of `Some(true)`, and a [`PlacementRequest::InPlace`]
    /// `target` or `source` that names neither a memory nor a layout). Every implementation rejects them before it
    /// does anything else, with the errors that [`Session::place`] reports for them, which checks them for every leaf
    /// before it places any leaf, so the same options are accepted whether a leaf is placed directly or through
    /// [`Session::place`]. A sharding that is shorter than the rank of the leaf is padded with replicated dimensions
    /// by every implementation, and the rank of a sharding and the other checks that depend on the leaf are handled
    /// by the implementation of this function.
    ///
    /// # Parameters
    ///
    ///   - `session`: Target [`Session`] that supplies the PJRT client and reshard cache. It also identifies
    ///     the session of concrete arrays that a traced leaf may wrap.
    ///   - `target`: Requested placement for this leaf, if one was specified.
    ///   - `source`: Source placement for this leaf, if one was specified.
    ///   - `donate`: Whether a compiled reshard may reuse the buffers of this leaf for the result.
    ///   - `may_alias`: Whether the result may share buffers with other arrays. `Some(false)` forbids it, unless
    ///     `donate` is `true`, which gives this leaf up (refer to [`Session::place`]).
    ///
    /// # Errors
    ///
    /// Implementations return an [`Error`] instead of a placed leaf, and never a partially placed one, when the leaf
    /// cannot be placed by `session` as requested. This covers the invalid options above, which are reported with
    /// [`ArrayError::DonatedArrayMayAlias`] at the root path for a donated leaf that may alias, and with the
    /// [`TypeError`](ryft_core::TypeError) of the type rule of [`PlaceOperation`] for an in-place request that names
    /// nothing (also for the source of a host leaf, which otherwise ignores its source), a leaf whose arrays belong
    /// to another PJRT client than `session`, a traced leaf whose arrays belong to another session (refer to
    /// [`Session::place`] for the rules of concrete and traced leaves), a `target` or `source` that the
    /// type rule of [`PlaceOperation`] rejects for a concrete leaf, except for an uneven split (i.e., a
    /// [`TypeError`](ryft_core::TypeError) of [`PlaceOperation::infer_output_type_allowing_uneven_splits`]),
    /// and a placement that fails to execute or to stage (e.g., a failed transfer, or a target that the rules of
    /// [`PlaceOperation`] reject for a traced leaf). The leaf is consumed either way. [`Session::place`] returns the
    /// first error of any leaf as its own, before it places any leaf when its check of the leaves reports it, and
    /// otherwise after the leaves before it were placed.
    fn place_parameter(
        self,
        session: &Session,
        target: Option<PlacementRequest>,
        source: Option<PlacementRequest>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Self::Output, Error>;
}

impl<T: HostValue + Parameter> PlaceableParameter<Array> for T {
    fn validate_placement(
        &self,
        session: &Session,
        target: Option<&PlacementRequest>,
        source: Option<&PlacementRequest>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<(), Error> {
        validate_parameter_options(ParameterPath::root, target, source, donate, may_alias)?;

        let r#type = self.host_type();
        let target = validate_host_target(&r#type, target.cloned())?;

        // The upload checks its target with the same function before it uploads anything.
        let default_device = || default_upload_device(session, &session.client().addressable_devices()?);
        let (r#type, placement) = host_upload_target(&r#type, target, default_device)?;
        Array::validate_host_upload(session, r#type, placement.mesh(), placement.sharding())
    }

    #[inline]
    fn process_value(&self) -> Option<sealed::ProcessValue<'_, Array>> {
        Some(sealed::ProcessValue::new(|| self.host_value_digest(), |_| self.validate_host_value()))
    }

    #[inline]
    fn communicates(&self, _: &Session, _: Option<&PlacementRequest>) -> bool {
        // Every process uploads its own value onto the devices that it addresses.
        false
    }
}

impl<T: HostValue + Parameter> Placeable for T {
    type Output = Array;

    fn place_parameter(
        self,
        session: &Session,
        target: Option<PlacementRequest>,
        source: Option<PlacementRequest>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Array, Error> {
        // A host value has no placement to validate and no device buffers to donate or share, so it only rejects the
        // options that no leaf accepts.
        validate_parameter_options(ParameterPath::root, target.as_ref(), source.as_ref(), donate, may_alias)?;

        let client = session.client();

        // A value without a target lives on the first device only because nothing chose any, so it is not committed
        // to that device, and operations move it to the devices of their committed inputs. The device is the default
        // device of the session (refer to `default_upload_device`).
        let r#type = self.host_type();
        let host = self.into_host_array();
        let target = validate_host_target(&r#type, target)?;

        // The upload takes dense row-major bytes, which is how a host array without a layout is stored, so only an
        // array with a layout is decoded into a copy. Its storage can be mutated safely, so the decoding and the upload
        // validate it rather than trust it.
        let bytes = dense_host_value_bytes(&host)?;

        // An in-place target names no devices, so it uploads the value to the same device as an upload without a
        // target, in the storage that it names, and it commits the result like every other target. A value without
        // a target, or with one that names only its storage, lives on a device that this process chose on its own
        // (refer to `Array::is_locally_assigned`).
        let committed = target.is_some();
        let locally_assigned = matches!(target, None | Some(PlacementRequest::InPlace { .. }));

        // The upload receives the addressable devices from which the default device is chosen, so that it enumerates
        // the devices of the client once.
        let addressable_devices = client.addressable_devices()?;
        let default_device = || default_upload_device(session, &addressable_devices);
        let (r#type, placement) = host_upload_target(&r#type, target, default_device)?;
        let (mesh, sharding) = (placement.mesh().clone(), placement.sharding().clone());

        Ok(Array::from_host_buffer_on(session, r#type, Some(mesh), Some(sharding), &bytes, addressable_devices)?
            .with_committed(committed)
            .with_locally_assigned(locally_assigned))
    }
}

/// Target or source parameter accepted by [`Session::place`]. This includes [`Placement`], [`PlacementRequest`],
/// [`Device`], [`Memory`](ryft_core::Memory), [`Sharding`](ryft_core::Sharding), and their optional forms. A bare
/// sharding pairs with the default [`DeviceMesh`] of the session when a leaf uses it.
pub trait SessionPlacementRequestParameter: Clone + Parameter + sealed::RequestParameter {}

impl<R: Clone + Parameter + sealed::RequestParameter> SessionPlacementRequestParameter for R {}

/// Contains items that only this crate can name, which seal the traits of this module that depend on them.
mod sealed {
    use ryft_core::{Device, Memory, ParameterPath, Placement, PlacementRequest, PlacementRequestParameter, Sharding};
    use ryft_pjrt::ExecutionFence;

    use crate::arrays::Array;
    use crate::{Error, Session};

    use super::{PlacementArgument, validate_parameter_options};

    /// What [`Session::place`] asks of a leaf that it places as an `Output` before it places any leaf of a tree. Every
    /// [`Placeable`](super::Placeable) implements this trait for its output, which seals that trait, because only this
    /// crate can name this one. [`place_parameter`](super::Placeable::place_parameter) below is the function of that
    /// trait that places the leaf.
    pub(super) trait PlaceableParameter<Output> {
        /// Checks, without placing this leaf, what [`place_parameter`](super::Placeable::place_parameter) would check
        /// from metadata alone before it submits any work for the same options, and returns the same error that it
        /// would return for them. [`Session::place`] calls this function for every leaf of a tree before it places any
        /// leaf, so that no leaf is transferred, uploaded, or donated when a later leaf is rejected. The
        /// implementations check the options that no leaf accepts (refer to [`validate_parameter_options`]) and then:
        ///
        ///   - for a host value, the type rule of [`PlaceOperation`](ryft_core::PlaceOperation) for its target and the
        ///     admission of its upload (e.g., a device that the client of `session` cannot address, unavailable memory
        ///     on an addressable device, or a data type without the additive identity that an unreduced target needs),
        ///   - for an [`Array`], that it belongs to the PJRT client of `session` or has a target that names devices,
        ///     the type rule of [`PlaceOperation`](ryft_core::PlaceOperation) for its target and source with its
        ///     runtime sharding, and the devices of its resolved target or, for an array of another client, the
        ///     transfer onto them (refer to [`Array::to_placement`]), and
        ///   - for a traced value, that the arrays that it wraps belong to `session`, because the type rule of a traced
        ///     value is applied by the context that records or executes its placement.
        ///
        /// The default implementation checks only the options that no leaf accepts.
        ///
        /// # Errors
        ///
        /// Returns the error that [`place_parameter`](super::Placeable::place_parameter) returns for the same options
        /// when one of these checks fails.
        #[inline]
        fn validate_placement(
            &self,
            session: &Session,
            target: Option<&PlacementRequest>,
            source: Option<&PlacementRequest>,
            donate: bool,
            may_alias: Option<bool>,
        ) -> Result<(), Error> {
            let _ = session;
            validate_parameter_options(ParameterPath::root, target, source, donate, may_alias)
        }

        /// Returns the comparison metadata of a concrete leaf. A host value or locally assigned [`Array`] provides a
        /// digest when the session compares its value across processes. The decision to compare is independent of the
        /// value's shape or bytes. The session owns the comparison and attaches its verdict to compared results. The
        /// default implementation returns [`None`], for a leaf whose value is never compared.
        #[inline]
        fn process_value(&self) -> Option<ProcessValue<'_, Output>> {
            None
        }

        /// Returns whether placing this leaf onto `target` can make this process wait for another process of its run
        /// (e.g., for a cross-host transfer), from metadata that every process that makes the same call holds alike,
        /// so that [`Session::place`] admits the whole call on every process before it places such a leaf. This
        /// function describes work submitted directly by this frontend. A tracer delegates to its context, which owns
        /// admission of any concrete placement it dispatches. Processes must agree on whether each array belongs to
        /// their own session's client. The default implementation returns `true`, so that a leaf that does not state
        /// otherwise is admitted with its tree.
        #[inline]
        fn communicates(&self, session: &Session, target: Option<&PlacementRequest>) -> bool {
            let _ = (session, target);
            true
        }

        /// Prepares the parameters of `arguments` in place after [`Session::place`] validates every leaf and before
        /// placing any leaf. A replacement must satisfy its argument's validated options. For example, a mixed
        /// [`PlaceInput`](super::PlaceInput) tree lifts host values into its compilation trace. `path` returns the
        /// path of a leaf by its index. The default implementation leaves the arguments unchanged.
        ///
        /// # Errors
        ///
        /// Returns the error of the first leaf that cannot be prepared, or an error of the whole tree.
        #[inline]
        fn prepare_arguments(
            arguments: &mut [PlacementArgument<Self>],
            session: &Session,
            path: &dyn Fn(usize) -> ParameterPath,
        ) -> Result<(), Error>
        where
            Self: Sized,
        {
            let _ = (arguments, session, path);
            Ok(())
        }
    }

    /// Converts the request forms accepted by the session frontend.
    pub trait RequestParameter {
        /// Converts this leaf to a placement request, resolving a bare sharding on the default mesh of `session`.
        ///
        /// # Errors
        ///
        /// Returns the error of [`Session::placement`] when the sharding cannot resolve.
        fn resolve(self, session: &Session) -> Result<Option<PlacementRequest>, Error>;
    }

    macro_rules! impl_request_parameter {
        ($request:ty) => {
            impl RequestParameter for $request {
                #[inline]
                fn resolve(self, session: &Session) -> Result<Option<PlacementRequest>, Error> {
                    let _ = session;
                    Ok(self.into_request())
                }
            }
        };
    }

    impl_request_parameter!(Placement);
    impl_request_parameter!(PlacementRequest);
    impl_request_parameter!(Device);
    impl_request_parameter!(Memory);

    impl RequestParameter for Sharding {
        #[inline]
        fn resolve(self, session: &Session) -> Result<Option<PlacementRequest>, Error> {
            session.placement(self).map(|placement| Some(placement.into()))
        }
    }

    impl<R: RequestParameter> RequestParameter for Option<R> {
        #[inline]
        fn resolve(self, session: &Session) -> Result<Option<PlacementRequest>, Error> {
            self.map_or(Ok(None), |request| request.resolve(session))
        }
    }

    /// Whether admission checks for known producer failures of an unshared runtime array.
    pub(super) enum ArrayReadiness {
        /// The runtime array keeps its current buffers.
        Unneeded,

        /// The runtime array is moved or copied.
        Required,
    }

    /// A host or locally assigned leaf that can be compared across processes.
    pub struct ProcessValue<'a, Output> {
        /// Computes the digest of a leaf that compares its value. Returning [`None`] poisons the comparison.
        digest: Box<dyn FnOnce() -> Option<[u32; 8]> + 'a>,

        /// Checks an unshared leaf before placement. Host values are always checked before their upload.
        validate: Box<dyn FnOnce(ArrayReadiness) -> Result<(), Error> + 'a>,

        /// Attaches the comparison verdict to the result of placement.
        join: fn(Output, ExecutionFence) -> Output,
    }

    impl<'a> ProcessValue<'a, Array> {
        /// Returns the comparison metadata for a leaf that is placed as an [`Array`].
        #[inline]
        pub(super) fn new<D: 'a + FnOnce() -> Option<[u32; 8]>, V: 'a + FnOnce(ArrayReadiness) -> Result<(), Error>>(
            digest: D,
            validate: V,
        ) -> Self {
            Self {
                digest: Box::new(digest),
                validate: Box::new(validate),
                join: |output, verdict| output.with_value_comparison(Some(verdict)),
            }
        }
    }

    impl<Output> ProcessValue<'_, Output> {
        /// Computes the digest for a leaf that compares its value.
        #[inline]
        pub(super) fn digest(self) -> Option<[u32; 8]> {
            (self.digest)()
        }

        /// Checks a leaf whose value is not compared.
        #[inline]
        pub(super) fn validate(self, readiness: ArrayReadiness) -> Result<(), Error> {
            (self.validate)(readiness)
        }

        /// Returns the function that attaches the comparison verdict.
        #[inline]
        pub(super) fn join(&self) -> fn(Output, ExecutionFence) -> Output {
            self.join
        }
    }
}

impl PlaceableParameter<Array> for Array {
    #[inline]
    fn validate_placement(
        &self,
        session: &Session,
        target: Option<&PlacementRequest>,
        source: Option<&PlacementRequest>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<(), Error> {
        let target = self.validate_placement_requests(session, target.cloned(), source.cloned(), donate, may_alias)?;
        self.validate_array_target(session, target)
    }

    #[inline]
    fn process_value(&self) -> Option<sealed::ProcessValue<'_, Array>> {
        self.is_locally_assigned().then(|| {
            sealed::ProcessValue::new(
                || array_value_digest(self),
                |readiness| {
                    if matches!(readiness, sealed::ArrayReadiness::Required)
                        && let Some(fence) = self.execution_fence()
                    {
                        fence.is_ready()?;
                    }
                    Ok(())
                },
            )
        })
    }

    #[inline]
    fn communicates(&self, session: &Session, target: Option<&PlacementRequest>) -> bool {
        // Each process places a locally assigned value independently. A placement between two clients uses host
        // readback and upload on the target owner. Comparing local values is a separate step.
        if self.is_locally_assigned() || !std::ptr::eq(self.client(), session.client()) {
            return false;
        }

        let Some(target) = target else {
            return false;
        };

        // In-place requests only change storage on the source devices. Explicit targets may move shards.
        let target_devices = match target {
            PlacementRequest::Resolved(placement) => placement.mesh().devices(),
            PlacementRequest::Device { device, .. } => std::slice::from_ref(device),
            PlacementRequest::InPlace { .. } => return false,
        };

        self.resolve_placement_request(target.clone()).map_or_else(
            |_| {
                // If resolution fails, conservatively identify potential peers from the source mesh and the
                // devices named by the request, which every process sees alike.
                devices_span_processes(self.mesh().devices().iter().chain(target_devices))
            },
            |placement| self.placement_communicates(placement.mesh(), placement.sharding()),
        )
    }
}

impl Placeable for Array {
    type Output = Array;

    #[inline]
    fn place_parameter(
        self,
        session: &Session,
        target: Option<PlacementRequest>,
        source: Option<PlacementRequest>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Array, Error> {
        let target = self.validate_placement_requests(session, target, source, donate, may_alias)?;
        self.place_validated(session, target, donate, may_alias)
    }
}

impl Array {
    /// Moves or copies this array to the provided placement without donating the source buffers, so the original
    /// [`Array`] remains fully usable after the call returns.
    ///
    /// When the placement and storage already match, the result is this array (i.e., the same logical array, refer to
    /// the identity rules of [`RetainedArray`](crate::RetainedArray)). If each process chose the devices of this array
    /// on its own, a target that names devices gives a new array on every process instead, which shares the buffers of
    /// this array where they already are, so that independent placements remain distinct when captured. Every other
    /// placement gives a new array. A change of devices or sharding takes the first device strategy that applies of
    /// those that [`crate::arrays::transfers`] describes (i.e., moves of exact shards, shard regions, or a compiled
    /// reshard) and otherwise the host fallback. A source with [locally assigned devices](Self::is_locally_assigned)
    /// (e.g., the result of [`Array::from_host_buffer`] without a mesh) takes none of them, and every process places
    /// its own such array on its own target devices with local work only, as that module describes.
    ///
    /// The host fallback copies the global array to the host as [`Array::copy_to_host`] does and uploads it again as
    /// [`Array::from_host_buffer`] does, so the result has the shards that a host upload of the same values with the
    /// target sharding has. It is used only when the device work fails or is skipped (e.g., when no strategy supports
    /// a manual target, or when a compiled reshard would change the value of an array whose reduction state the
    /// placement changes), when the target mesh is fully addressable from this process, and when this process can
    /// read the source back. That is the case when this process holds a buffer for one replica of every distinct
    /// contribution to the value with elements (i.e., of every tile and, along unreduced axes, of every per-device
    /// value of a tile), whatever the other processes hold, and when no two devices that hold one such contribution
    /// differ along a varying manual mesh axis, because those devices can hold different values for the same slice that
    /// a dense host copy cannot represent. Devices that differ along a manual axis which splits the array hold distinct
    /// tiles, so they read back like any other tiles. A placement that keeps the contributions of a pending reduction
    /// or of manual variation within a tile requires a device strategy, because the host fallback combines
    /// contributions. The readback and the upload continue in the background, and a failure of that work, whose error
    /// also describes the failure of the device work, surfaces through the readiness functions and the readbacks of the
    /// result and through [`Session::synchronize`].
    ///
    /// When the target is shared across processes and this array is locally assigned, every process must hold the same
    /// value, because each process places its own value on the devices that it owns. For uncommitted inputs, a mesh or
    /// sharding target whose devices belong to exactly the processes of a run of more than one process compares the
    /// per-process values based on [`SessionBuilder::value_comparison`](crate::SessionBuilder::value_comparison):
    /// each process reads its value back on the calling thread, hashes it, and submits a comparison before anything
    /// else depends on the value. The comparison uses a compiled all-reduce on a shared native PJRT topology. Explicit
    /// remote device declarations return
    /// [`SessionError::PlacementComparisonUnavailable`](crate::SessionError::PlacementComparisonUnavailable) before
    /// readback when the placement requires a comparison. A digest that differs (including when data types or shapes
    /// differ) or cannot be computed causes a [`ryft_pjrt::Error::FailedPrecondition`] on every process. The same
    /// sequence of comparing placements on every process remains a precondition (refer to [`Session::place`]).
    /// For any other target shared across processes, equal values remain an unchecked precondition.
    ///
    /// The host fallback reads back the combined value of an array with unreduced axes and uploads it again
    /// as [`Array::from_host_buffer`] does, so when the target keeps an unreduced axis, it redistributes the
    /// contributions along that axis. With [`UnreducedKind::Sum`], the devices at coordinate zero along it hold the
    /// whole value and the others hold the additive identity. With [`UnreducedKind::Max`] or [`UnreducedKind::Min`],
    /// every device along it holds the whole value, which is its own extremum. The logical value is the same, but the
    /// shards of the result can then differ from those that a device strategy would produce.
    ///
    /// The result always belongs to `session`, even when it is this array or shares its buffers. An array of another
    /// PJRT client (e.g., of another backend) is read back and uploaded onto the devices that `target` names, because
    /// PJRT copies buffers only within one client, as JAX places an array onto the devices of another backend. That
    /// uses the host path directly. For an array with bytes, the placement must keep no distinct contributions apart,
    /// each target owner must be able to read the value back, and the target must lie on the devices of one process
    /// unless this array is locally assigned. The memory follows the same rule as a transfer within one client.
    /// The request must name the target devices, and the result is [committed](Self::is_committed) to them.
    ///
    /// When the host fallback also cannot satisfy the request, typically because some source shards that the value
    /// needs live on a remote process or because the source varies along manual mesh axes, the original error of the
    /// device work is propagated as the most informative diagnostic.
    ///
    /// The target uses the requested [`Memory`](ryft_core::Memory). When the memory is omitted, a target that names a
    /// single device (i.e., a [`PlacementRequest::Device`]) uses the default memory of that device for every source,
    /// including one in pinned host memory, and a mesh or sharding target keeps the memory of the source. PJRT
    /// cross-host receives land in device memory and can then be copied to the requested memory on that device. An
    /// explicit [`Layout`](ryft_core::Layout) applies to each target shard. When the layout is omitted, the result
    /// keeps the layout of the source only while that layout still describes each target shard (e.g., the strides of a
    /// column shard do not describe the whole value that a gather produces), and declares no layout otherwise. Layout
    /// changes use a compiled conversion on each shard's device when supported. Otherwise, each affected local shard is
    /// read back and uploaded in the requested layout in the background, as for the host fallback. Both paths preserve
    /// the separate contributions along unreduced axes.
    ///
    /// Use [`Self::into_placement`] when the source array is no longer needed after the move. It consumes `self` and
    /// lets PJRT reuse the source buffers for the target where it can, which may avoid an extra allocation.
    ///
    /// # Parameters
    ///
    ///   - `session`: [`Session`] that provides the PJRT client and the compilation cache.
    ///   - `target`: Resolved [`Placement`] or a [`PlacementRequest`] describing where to place the array.
    #[inline]
    pub fn to_placement<P: Into<PlacementRequest>>(&self, session: &Session, target: P) -> Result<Self, Error> {
        self.clone()
            .place_with_request(session, target.into(), false, ValueComparison::of_session(session)?)
    }

    /// Moves this array to the provided placement, consuming `self` and donating its source buffers to the target.
    ///
    /// Donation is best effort. Only the compiled reshard on the mesh of this array donates its input, and PJRT may
    /// then reuse a source buffer for a target shard, which can save the memory of the source array for large training
    /// tensors. Nothing guarantees that saving. A buffer can be reused only when every device holds a shard of the same
    /// type before and after the reshard, which is when the lowering pairs the output of the reshard with its donated
    /// input (e.g., completing a pending sum keeps the buffers of a uniquely owned array on the CPU plugin), and not
    /// when the shards change shape (e.g., moving a uniquely owned `64x64` array from row shards to replicas on four
    /// devices allocates four new buffers). The other strategies (e.g., moves of whole shards, shard regions, a reshard
    /// across meshes, or the host fallback) do not donate, a buffer that another handle shares is never donated, and
    /// PJRT may still allocate new output buffers. Because `self` is consumed at the language boundary, the donated
    /// buffers are not observable to the caller after this function returns.
    ///
    /// The strategies follow [`Self::to_placement`], and donation never changes which of them is chosen, so the
    /// compiled reshard of an array with a pending reduction (i.e., unreduced axes in its runtime sharding) completes
    /// or keeps the reduction on the devices with or without donation, and a change of reduction state that the
    /// compiled reshard cannot make (e.g., introducing a pending maximum, or completing a pending maximum of floating
    /// point values) takes the host fallback from both entry points. Donation changes only what follows a compiled
    /// reshard whose execution fails while this array holds its buffers exclusively, which may have handed them to
    /// PJRT. [`Self::to_placement`] then still falls back to a host transfer, whereas this function returns the error.
    /// When another handle shares the buffers of this array (e.g., a clone that the caller keeps), nothing can be
    /// donated, and this function falls back as [`Self::to_placement`] does.
    ///
    /// # Parameters
    ///
    ///   - `session`: [`Session`] that provides the PJRT client and the compilation cache.
    ///   - `target`: Resolved [`Placement`] or a [`PlacementRequest`] describing where to place the array.
    #[inline]
    pub fn into_placement<P: Into<PlacementRequest>>(self, session: &Session, target: P) -> Result<Self, Error> {
        self.place_with_request(session, target.into(), true, ValueComparison::of_session(session)?)
    }

    /// Places this array as `request` asks, which [`Self::to_placement`] and [`Self::into_placement`]
    /// do with the [`ValueComparison`] of `session`. The request is resolved against this array (refer to
    /// [`Self::resolve_placement_request`]) and placed with [`Self::place_within_client`], or with
    /// [`Self::place_across_clients`] for an array of another PJRT client than that of `session`. Only a request
    /// that names neither devices nor a sharding (i.e., an [`PlacementRequest::InPlace`] request, which names only a
    /// memory or a layout) leaves the result on the devices that this process chose on its own when this array has such
    /// devices, because every other request names the devices of the result (refer to [`Self::is_locally_assigned`]).
    ///
    /// Every process places its own value of such an array, so when `comparison` applies to `request` (refer to
    /// [`ValueComparison::applies`]), this function submits one comparison before executing the placement. It submits
    /// the digest of the value or a poison contribution if the local request is invalid or the readback fails. Each
    /// process participates before returning its local error, and the result joins the verdict. [`Session::place`]
    /// places its leaves with `comparison` set to [`None`], because it compares the values of all of them at once.
    ///
    /// # Parameters
    ///
    ///   - `session`: [`Session`] that provides the PJRT client and the compilation cache.
    ///   - `request`: [`PlacementRequest`] that specifies the target.
    ///   - `donate`: Whether a compiled reshard on the mesh of this array may reuse the buffers of this array
    ///     for its output (refer to [`Self::place_within_client`]).
    ///   - `comparison`: [`ValueComparison`] of the values of the processes, if this placement makes one.
    pub(super) fn place_with_request(
        self,
        session: &Session,
        request: PlacementRequest,
        donate: bool,
        comparison: Option<&ValueComparison>,
    ) -> Result<Self, Error> {
        let in_place = matches!(request, PlacementRequest::InPlace { .. });
        let locally_assigned = self.is_locally_assigned();
        let collective = comparison
            .filter(|comparison| locally_assigned && comparison.applies(Some(&request)))
            .map(|comparison| comparison.collective(session))
            .transpose()?;
        let same_client = std::ptr::eq(self.client(), session.client());
        let target = self
            .validate_placement_client(session, Some(&request))
            .and_then(|_| self.validate_placement_target(session, request, collective.is_some()));
        let verdict = collective
            .map(|collective| {
                let digest = target.as_ref().ok().and_then(|_| combined_digest([array_value_digest(&self)]));
                collective.submit(session, digest)
            })
            .transpose();
        let target = target?;
        let verdict = verdict?;
        let output = match same_client {
            true => self.place_within_client(session, &target, donate)?,
            false => self.place_across_clients(session, &target)?,
        };

        // A target that names devices can already hold the input that this process chose on its own, while another
        // process must move its input there, so such a placement is a new array on every process (refer to the identity
        // rules of `RetainedArray`).
        let output = if locally_assigned && !in_place { output.with_new_identity() } else { output };
        Ok(output.with_locally_assigned(locally_assigned && in_place).with_value_comparison(verdict))
    }

    /// Checks that `session` can place this array onto `request`, which [`Self::place_with_request`] and the
    /// admission of an [`Array`] leaf of [`Session::place`] both check before anything else that concerns this array.
    /// A same-client placement first checks that `session` can bind the source (refer to [`Self::with_session`]).
    /// An array of another PJRT client can be placed only onto a `request` that names devices of the client of
    /// `session` (refer to [`Self::place_across_clients`]).
    fn validate_placement_client(&self, session: &Session, request: Option<&PlacementRequest>) -> Result<(), Error> {
        match request {
            _ if std::ptr::eq(self.client(), session.client()) => self.validate_session_placement(session),
            Some(PlacementRequest::InPlace { .. }) => Err(PjrtError::invalid_argument(
                "cannot place an array of another PJRT client without naming the devices of the target, \
                 because the result must live on devices of the client of the session",
            )
            .into()),
            Some(_) => Ok(()),
            None => self.validate_session_placement(session),
        }
    }

    /// Checks the placement of this array as [`Placeable::place_parameter`] does before it resolves its target,
    /// and returns `target` with a sharding that is padded to the rank of this array. Both [`Placeable`] functions of
    /// [`Array`] call this function, so that they accept the same placements.
    fn validate_placement_requests(
        &self,
        session: &Session,
        target: Option<PlacementRequest>,
        source: Option<PlacementRequest>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Option<PlacementRequest>, Error> {
        // Reject the options that no leaf accepts, as `Session::place` does before it looks at any leaf, and then a
        // foreign PJRT client before validating either placement request, unless the target names the devices onto
        // which such an array is transferred. Dispatch binds the result to `session` when there is a target.
        validate_parameter_options(ParameterPath::root, target.as_ref(), source.as_ref(), donate, may_alias)?;
        self.validate_placement_client(session, target.as_ref())?;

        let rank = self.r#type().rank();
        let target = target.map(|target| target.padded(rank)).transpose()?;
        let source = source.map(|source| source.padded(rank)).transpose()?;

        // Runtime sharding supplies the reduction state and shard shapes along auto axes that the type omits, and an
        // array that is free to be placed elsewhere satisfies the source through its relocatable type, as on every
        // other path that places a concrete value (refer to `PlaceOperation::infer_output_type_of_value`).
        PlaceOperation::new(target.clone(), source).infer_output_type_of_value(self, Some(self.sharding()), false)?;

        Ok(target)
    }

    /// Performs a placement that [`Self::validate_placement_requests`] accepted, using the same copy and completion
    /// rules for both eager entry points.
    fn place_validated(
        self,
        session: &Session,
        target: Option<PlacementRequest>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Array, Error> {
        // Donation takes precedence over a copy requested by `may_alias`. A forced copy must avoid every input
        // buffer, including buffers that only the result holds once placement consumes the leaf. The weak handles
        // let buffers that the placement releases drop normally.
        let copies = may_alias == Some(false) && !donate;
        let buffers: Vec<_> = match copies {
            true => self.addressable_shards().filter_map(ArrayShard::buffer).map(Arc::downgrade).collect(),
            false => Vec::new(),
        };

        // The target resolves against the array in the one place that also keeps the locally assigned devices of an
        // array that an in-place target leaves where it is.
        let output = match target {
            Some(target) => self.place_with_request(session, target, donate, None)?,
            None => self.with_session(session)?.without_send_fence(),
        };

        // The result is this leaf when the placement does not change, and it otherwise shares the buffers of the
        // shards that stay on their devices. Holding the buffers of this leaf that are still alive makes them shared,
        // so that `into_unaliased` copies exactly those, and not the buffers that the placement created.
        if copies {
            // The handles must stay alive until the copy is made, so they are bound to a name and dropped explicitly
            // (`let _ = …` would drop them at once, and the buffers would then look exclusive and not be copied).
            let held = buffers.iter().filter_map(Weak::upgrade).collect::<Vec<_>>();
            let output = output.into_unaliased();
            drop(held);
            output
        } else {
            Ok(output)
        }
    }
}

macro_rules! impl_placeable {
    // Implements `Placeable` for one generic tracer type. The optional `where` clause supplies
    // its type-specific bounds.
    (<$($generic:ident),+> $value:ty $(where $($bounds:tt)+)?) => {
        impl<$($generic),+> PlaceableParameter<$value> for $value
        where
            Self: RuntimeArrays + CoreValue<Type = ArrayType, Dispatch: PlaceDispatch<ArrayType, Self>>,
            $($($bounds)+)?
        {
            #[inline]
            fn validate_placement(
                &self,
                session: &Session,
                target: Option<&PlacementRequest>,
                source: Option<&PlacementRequest>,
                donate: bool,
                may_alias: Option<bool>,
            ) -> Result<(), Error> {
                // Donation only applies to concrete arrays, but a donated leaf that may alias is rejected
                // for every leaf.
                validate_parameter_options(ParameterPath::root, target, source, donate, may_alias)?;

                for array in self.runtime_arrays() {
                    // Placement uses the contexts already attached to the tracer. Rebinding would lose state such as
                    // the residual program owned by a partial evaluation context. Check the PJRT client before the
                    // session because device IDs are scoped to a client. Other handles of the same session are
                    // accepted.
                    array.validate_session_client(session)?;
                    if !array.session().is_same_session(session) {
                        let message = "cannot place a traced value over arrays of another session";
                        return Err(PjrtError::invalid_argument(message).into());
                    }
                }

                Ok(())
            }

            #[inline]
            fn communicates(&self, _: &Session, _: Option<&PlacementRequest>) -> bool {
                false
            }
        }

        impl<$($generic),+> Placeable for $value
        where
            Self: RuntimeArrays + CoreValue<Type = ArrayType, Dispatch: PlaceDispatch<ArrayType, Self>>,
            $($($bounds)+)?
        {
            type Output = Self;

            #[inline]
            fn place_parameter(
                self,
                session: &Session,
                target: Option<PlacementRequest>,
                source: Option<PlacementRequest>,
                donate: bool,
                may_alias: Option<bool>,
            ) -> Result<Self, Error> {
                self.validate_placement(session, target.as_ref(), source.as_ref(), donate, may_alias)?;

                // Pad explicit shardings to the value's rank, which is the per-item rank of a batched value. Keep
                // bare device and in-place requests in their original form so the placement rule can validate them
                // first. It uses a wrapped array's placement to infer or complete the source that transposition
                // uses to return the cotangent (refer to `PlacementRequest::resolve_source`).
                let rank = self.r#type().rank();
                let source = source.map(|source| source.padded(rank)).transpose()?;
                let target = target.map(|target| target.padded(rank)).transpose()?;

                // Record a forced copy even without a target, so the policy applies when the operation executes
                // on concrete arrays. Donation only applies to concrete leaves and takes precedence.
                let may_alias = may_alias != Some(false) || donate;
                Ok(<Self as CoreValue>::Dispatch::place(&self, target.as_ref(), source, may_alias)?)
            }
        }
    };
}

impl_placeable!(<C> Tracer<C> where C: Context<Type = ArrayType>);
impl_placeable!(<C> PartialTracer<C> where C: Context<Type = ArrayType>);
impl_placeable!(<C, P> DifferentiationTracer<C, P> where C: Context<Type = ArrayType>, P: DifferentiationPolicy<C>);
impl_placeable!(<C, P> BatchingTracer<C, P> where C: Context<Type = ArrayType>, P: BatchingPolicy<C>);
impl_placeable!(<V> ProjectedValue<ArrayType, V>);

/// Validates the requested `target` for a host value of `r#type` and returns it with its sharding padded to the rank
/// of the value. An upload places a value that has no tracked sharding and lives in device memory without a declared
/// layout, which is how a host value is uploaded without a target. The type rule of [`PlaceOperation`] validates the
/// target against that type before any upload and permits the target to split a dimension unevenly (refer to
/// [`Session::place`]).
fn validate_host_target(
    r#type: &ArrayType,
    target: Option<PlacementRequest>,
) -> Result<Option<PlacementRequest>, Error> {
    let target = target.map(|target| target.padded(r#type.rank())).transpose()?;
    if let Some(target) = &target {
        PlaceOperation::infer_output_type_allowing_uneven_splits(r#type, None, Some(target), None)?;
    }
    Ok(target)
}

/// Resolves the placement and type of an upload from the `target` returned by [`validate_host_target`]. The type keeps
/// the element type and static shape of `r#type` and takes its memory and layout from the target. An in-place or
/// omitted target uses the device that `default_device` returns, which is called only in those cases.
fn host_upload_target(
    r#type: &ArrayType,
    target: Option<PlacementRequest>,
    default_device: impl FnOnce() -> Result<Device, Error>,
) -> Result<(ArrayType, Placement), Error> {
    let rank = r#type.rank();
    let memory = target.as_ref().and_then(PlacementRequest::target_memory).unwrap_or_default();
    let (placement, layout) = match target {
        Some(PlacementRequest::Resolved(placement)) => {
            let layout = placement.layout().cloned();
            (placement, layout)
        }
        Some(PlacementRequest::Device { device, layout, .. }) => (Placement::single_device(device, rank), layout),
        Some(PlacementRequest::InPlace { layout, .. }) => (Placement::single_device(default_device()?, rank), layout),
        None => (Placement::single_device(default_device()?, rank), None),
    };
    Ok((r#type.clone().with_memory(memory).with_layout(layout), placement))
}

/// Returns the device onto which a host value is uploaded when its target names no devices, which is the default device
/// of `session` (refer to [`Session::with_default_device`]), or otherwise the first of the `addressable_devices` of its
/// client, whether or not `session` has a default mesh. That includes the anonymous mesh of one device that
/// [`Session::with_device_mesh`] gives, which every process shares like any other mesh. The choice reads only what
/// the caller asked for (refer to `Session::default_device_mesh`), so every process uploads such a value onto a
/// device of its own, uncommitted and locally assigned, for both kinds of handle.
fn default_upload_device(session: &Session, addressable_devices: &[PjrtDevice<'static>]) -> Result<Device, Error> {
    if let Some(mesh) = session.default_device_mesh() {
        return Ok(mesh.devices()[0]);
    }
    let device = addressable_devices.first().ok_or_else(|| Error::from(ArrayError::MissingDefaultDevice))?;
    Ok(Device::from_pjrt(device)?)
}

/// Input leaf of [`Session::place`] that holds a host value, a runtime [`Array`], or a [`CompileTracer`].
/// A [`Parameterized`] tree has one leaf type, so use [`PlaceInput`] to mix these values in one call. When `V` is
/// [`Array`], host arrays of the reference backend and scalars of every [`ArrayElement`](ryft_core::ArrayElement) type
/// convert to [`Self::Host`] with [`From`]. Runtime arrays convert to [`Self::Runtime`] with [`From`] and are placed
/// the same way as [`Self::Concrete`]. When `V` is [`CompileTracer`], [`Self::Runtime`] holds a tracer that identifies
/// the compilation trace. The trace receives host values as uploads on the default device and concrete arrays as
/// runtime captures. The imported leaves receive the target, source, and aliasing options of their positions in the
/// tree. Donation and process value comparison do not apply. A tree that spans different compilation traces, or has
/// host or concrete leaves but no tracer to import them, returns a [`ryft_pjrt::Error::InvalidArgument`]. Place host
/// or concrete leaves separately when batching or differentiating a function. A tree of tracers alone can use
/// [`Session::place`] directly without [`PlaceInput`]. A tree whose leaves are all constructed as [`Self::Host`]
/// or [`Self::Concrete`] names no `V`, so it is written `PlaceInput::<Array>::Host(value)`, or its leaves are
/// converted with [`From`] instead.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{ArrayType, DataType};
/// # use ryft_xla::{Array, PlaceInput, PlacementOptions, Platform, Session};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let session = Session::builder().platform(Platform::Cpu).build()?;
/// let r#type = ArrayType::new_static(DataType::F32, [2]);
/// let array = Array::from_host_buffer(&session, r#type, None, None, [1f32, 2.].map(f32::to_ne_bytes).concat())?;
/// let device = array.mesh().devices()[0];
///
/// // The target applies to both leaves, and so would every other option.
/// let options = PlacementOptions::to(device);
/// let (scalar, array) = session.place((PlaceInput::from(2.5f32), PlaceInput::from(array)), options)?;
/// assert_eq!(scalar.mesh().devices(), [device]);
/// assert_eq!(scalar.copy_to_host()?.elements::<f32>()?, vec![2.5]);
/// assert_eq!(array.copy_to_host()?.elements::<f32>()?, vec![1., 2.]);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Parameter)]
pub enum PlaceInput<V = Array> {
    /// Host value, which is uploaded as a dense array (e.g., a scalar, which is a rank-0 array), or which becomes a
    /// constant of the trace of the other leaves of its tree when `V` is a traced value.
    Host(CoreArray),

    /// Runtime array or traced value.
    Runtime(V),

    /// Runtime array, which is placed like [`Self::Runtime`] over runtime arrays or imported into the trace of the
    /// other leaves of its tree when `V` is a tracer.
    Concrete(Array),
}

impl PlaceInput<CompileTracer> {
    /// Checks the type rule of [`PlaceOperation`] and the target of the placement of this leaf, for the type
    /// that the leaf has in a trace whose captures are placed as `capture_placement` states. A host value or a concrete
    /// array joins such a trace as a capture, which keeps its own type under [`CapturePlacement::Preserve`] and is
    /// declared as the compiled function receives it under [`CapturePlacement::FollowCall`] (refer to
    /// [`captured_type`]). A traced leaf already has its type.
    fn validate_placement_in_trace(
        &self,
        session: &Session,
        target: Option<&PlacementRequest>,
        source: Option<&PlacementRequest>,
        capture_placement: CapturePlacement,
    ) -> Result<(), Error> {
        let (r#type, mesh) = match self {
            Self::Host(value) => {
                let default_device = || default_upload_device(session, &session.client().addressable_devices()?);
                let (r#type, placement) = host_upload_target(&value.host_type(), None, default_device)?;
                Array::validate_host_upload(session, r#type.clone(), placement.mesh(), placement.sharding())?;
                (Cow::Owned(r#type), Some(Cow::Owned(placement.mesh().clone())))
            }
            Self::Concrete(value) => {
                value.validate_session_placement(session)?;
                (value.r#type(), Some(Cow::Borrowed(value.mesh())))
            }
            Self::Runtime(value) => (value.r#type(), None),
        };

        let captured = match (self, capture_placement) {
            (Self::Host(_) | Self::Concrete(_), CapturePlacement::FollowCall) => captured_type(&r#type),
            _ => None,
        };

        let r#type = captured.as_ref().unwrap_or(&r#type);
        let rank = r#type.rank();
        let target = target.cloned().map(|request| request.padded(rank)).transpose()?;
        let source = source.cloned().map(|request| request.padded(rank)).transpose()?;
        let operation = PlaceOperation::new(target, source);

        // Check the type before importing any mixed leaves.The compilation session checks targets of traced
        // leaves, while this session checks the targets of the host values and concrete arrays it imports.
        let output = operation.infer_output_type_requiring_even_splits(r#type, None)?;
        if let (Some(input_mesh), Some(request)) = (mesh.as_deref(), operation.target()) {
            // Check the inventory for targets that name devices. An in-place request keeps the capture's
            // devices, which were checked when validating the upload or binding the concrete array to the session.
            let devices = match request {
                PlacementRequest::Resolved(placement) => {
                    validate_mesh_devices(session, placement.mesh())?;
                    placement.mesh().devices()
                }
                PlacementRequest::Device { device, .. } => {
                    let mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![*device])?;
                    validate_mesh_devices(session, &mesh)?;
                    std::slice::from_ref(device)
                }
                PlacementRequest::InPlace { .. } => input_mesh.devices(),
            };

            // Check requested memory before importing the capture. Only this process can inspect its own device
            // memories, and zero values have no buffers that need a memory space.
            if request.explicit_memory().is_some() && !output.data_type().is_zero() {
                let client = session.client();
                let process_index = client.process_index()?;
                for device in devices.iter().filter(|device| device.process_index() == process_index) {
                    client.lookup_device(device.id())?.memory(output.memory())?;
                }
            }
        }

        Ok(())
    }
}

impl PlaceableParameter<Array> for PlaceInput {
    #[inline]
    fn validate_placement(
        &self,
        session: &Session,
        target: Option<&PlacementRequest>,
        source: Option<&PlacementRequest>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<(), Error> {
        match self {
            Self::Host(value) => value.validate_placement(session, target, source, donate, may_alias),
            Self::Runtime(value) | Self::Concrete(value) => {
                value.validate_placement(session, target, source, donate, may_alias)
            }
        }
    }

    #[inline]
    fn process_value(&self) -> Option<sealed::ProcessValue<'_, Array>> {
        match self {
            Self::Host(value) => PlaceableParameter::process_value(value),
            Self::Runtime(value) | Self::Concrete(value) => PlaceableParameter::process_value(value),
        }
    }

    #[inline]
    fn communicates(&self, session: &Session, target: Option<&PlacementRequest>) -> bool {
        match self {
            Self::Host(value) => PlaceableParameter::communicates(value, session, target),
            Self::Runtime(value) | Self::Concrete(value) => PlaceableParameter::communicates(value, session, target),
        }
    }
}

impl Placeable for PlaceInput {
    type Output = Array;

    #[inline]
    fn place_parameter(
        self,
        session: &Session,
        target: Option<PlacementRequest>,
        source: Option<PlacementRequest>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Array, Error> {
        match self {
            Self::Host(value) => value.place_parameter(session, target, source, donate, may_alias),
            Self::Runtime(value) | Self::Concrete(value) => {
                value.place_parameter(session, target, source, donate, may_alias)
            }
        }
    }
}

impl PlaceableParameter<CompileTracer> for PlaceInput<CompileTracer> {
    fn validate_placement(
        &self,
        session: &Session,
        target: Option<&PlacementRequest>,
        source: Option<&PlacementRequest>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<(), Error> {
        validate_parameter_options(ParameterPath::root, target, source, donate, may_alias)?;

        if let Self::Runtime(value) = self {
            value.validate_placement(session, target, source, donate, may_alias)?;
        }

        // One leaf does not see the trace that its tree joins, so this check assumes the capture placement of a
        // compiled function. `Self::prepare_arguments` checks the leaves again for a trace that preserves it.
        self.validate_placement_in_trace(session, target, source, CapturePlacement::FollowCall)
    }

    #[inline]
    fn communicates(&self, session: &Session, target: Option<&PlacementRequest>) -> bool {
        // The leaves that are not traced are rejected before anything is placed unless they joined the trace.
        match self {
            Self::Host(_) | Self::Concrete(_) => false,
            Self::Runtime(value) => PlaceableParameter::communicates(value, session, target),
        }
    }

    fn prepare_arguments(
        arguments: &mut [PlacementArgument<Self>],
        session: &Session,
        path: &dyn Fn(usize) -> ParameterPath,
    ) -> Result<(), Error> {
        if arguments.is_empty() {
            return Ok(());
        }

        // The first tracer supplies the context into which the host and concrete leaves are imported.
        // Every other tracer must share its builder so that all leaves belong to one program.
        let mut traced = arguments.iter().filter_map(|argument| match &argument.parameter {
            Self::Host(_) | Self::Concrete(_) => None,
            Self::Runtime(value) => Some(value),
        });

        let Some(trace) = traced.next() else {
            let error = PjrtError::invalid_argument(
                "cannot place a host value or a runtime array as a traced value without a traced value in the same call",
            );
            return Err(Error::from(error).with_placement_parameter_path(|| path(0)));
        };

        if traced.any(|other| !Rc::ptr_eq(trace.value().builder(), other.value().builder())) {
            return Err(
                PjrtError::invalid_argument("cannot place traced values of different traces in one call").into()
            );
        }

        let context = trace.domain();

        // A trace that preserves the placement of its captures gives a host value or a concrete array its own type,
        // which the check of each leaf did not assume, so a placement that this type rejects is reported before any
        // leaf is uploaded or enters the trace, where a failed placement would stay recorded.
        if context.capture_placement() == CapturePlacement::Preserve {
            for (index, argument) in arguments.iter().enumerate() {
                if matches!(argument.parameter, Self::Host(_) | Self::Concrete(_)) {
                    argument
                        .parameter
                        .validate_placement_in_trace(
                            session,
                            argument.target.as_ref(),
                            argument.source.as_ref(),
                            CapturePlacement::Preserve,
                        )
                        .map_err(|error| error.with_placement_parameter_path(|| path(index)))?;
                }
            }
        }

        // Upload each host leaf on the default device and bind concrete arrays to the session before adding
        // them as captures in the existing compilation trace.
        for (index, argument) in arguments.iter_mut().enumerate() {
            let array = match &argument.parameter {
                Self::Host(value) => value.clone().place_parameter(session, None, None, false, None),
                Self::Concrete(value) => value.clone().place_parameter(session, None, None, false, None),
                Self::Runtime(_) => continue,
            };

            argument.parameter = Self::Runtime(
                array
                    .and_then(|array| {
                        Ok(ImportingContext::<Session, ThroughParent<ThroughCapture>>::import(
                            &context,
                            Value::Array(array),
                        )?)
                    })
                    .map_err(|error| error.with_placement_parameter_path(|| path(index)))?,
            );
        }

        Ok(())
    }
}

impl Placeable for PlaceInput<CompileTracer> {
    type Output = CompileTracer;

    fn place_parameter(
        self,
        session: &Session,
        target: Option<PlacementRequest>,
        source: Option<PlacementRequest>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<CompileTracer, Error> {
        match self {
            Self::Host(_) | Self::Concrete(_) => {
                validate_parameter_options(ParameterPath::root, target.as_ref(), source.as_ref(), donate, may_alias)?;

                // Tree preparation imports these leaves through another leaf's trace. A direct call on a host
                // value or concrete array has no tracer that can supply that context.
                Err(PjrtError::invalid_argument(
                    "cannot place a host value or a runtime array as a traced value without a traced value \
                     in the same call",
                )
                .into())
            }
            Self::Runtime(value) => value.place_parameter(session, target, source, donate, may_alias),
        }
    }
}

impl<T: HostValue> From<T> for PlaceInput {
    #[inline]
    fn from(value: T) -> Self {
        Self::Host(value.into_host_array())
    }
}

impl From<Array> for PlaceInput {
    #[inline]
    fn from(value: Array) -> Self {
        Self::Runtime(value)
    }
}

impl From<CompileTracer> for PlaceInput<CompileTracer> {
    #[inline]
    fn from(value: CompileTracer) -> Self {
        Self::Runtime(value)
    }
}

impl Array {
    /// Returns a new array (refer to the identity rules of [`RetainedArray`](crate::RetainedArray)) with the type, the
    /// placement, and the commitment of this array whose addressable shard buffers no other handle holds. This is how
    /// a placement whose result must not alias its input (i.e., a placement with `may_alias` set to `false` and without
    /// donation) gives that result buffers of its own. The result is a new array on every process, including a process
    /// that holds no buffer of this array or holds each of them alone, because it then has no buffer to copy (refer to
    /// [`Self::with_new_identity`]). A buffer shared with another handle is copied on the same device and in the same
    /// memory (e.g., when a shard stays on its device and the caller still holds the input). Buffers held only by this
    /// array, such as those created by placement, are kept. A clone holds all of this array's buffers, so it makes
    /// this function copy every buffer. A caller that consumed the only handle of the input must therefore hold the
    /// buffers of that input while it calls this function, or the result could return them. The copies are made by
    /// [`Self::with_copied_shard_buffers`], so the execution fence of the result joins the readiness of every copy with
    /// the fence of this array, because a readback reads only one replica of each tile and would otherwise succeed with
    /// a replica whose copy did not fail.
    pub(crate) fn into_unaliased(self) -> Result<Self, Error> {
        // Clones of this array share its parts, and with them every one of its buffers.
        let has_clones = Arc::strong_count(&self.parts) != 1;
        let is_shared = |buffer: &Arc<Buffer<'static>>| has_clones || Arc::strong_count(buffer) != 1;
        if !self.addressable_shards().filter_map(ArrayShard::buffer).any(is_shared) {
            return Ok(self.with_new_identity());
        }
        self.with_copied_shard_buffers(is_shared)
    }
}

impl Session {
    /// Places `x` on devices. This is the Ryft analogue of JAX's
    /// [`jax.device_put`](https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html).
    ///
    /// The input `x` may be one supported leaf or a [`Parameterized`] tree of supported leaves (i.e., of
    /// [`Placeable`]s). Any provided `target`, `source`, `donate`, and `may_alias` options follow tree prefix
    /// broadcasting semantics over `x`. The target and source prefixes accept [`Placement`], [`PlacementRequest`],
    /// bare [`Device`], bare [`Memory`](ryft_core::Memory), and bare [`Sharding`](ryft_core::Sharding) leaves, so a
    /// target can be one device (refer to [`PlacementRequest::Device`]) or a memory that leaves every leaf on its
    /// devices (refer to [`PlacementRequest::InPlace`]). Either can be optional, so a tree can mix explicit placements
    /// with `None` to omit an option on a branch. A tree has one leaf type, so a tree that mixes host values and
    /// [`Array`]s holds them as [`PlaceInput`]s, and so does a tree that mixes host values and traced values inside a
    /// compiled or staged function, whose host leaves become constants of the trace of its traced leaves and are then
    /// placed like those (refer to [`PlaceInput`]). A prefix leaf may also cover only subtrees without leaves, in which
    /// case it places nothing (e.g., one of two targets for an empty [`Vec`] and a scalar, or a single target
    /// for an empty [`Vec`]). A bare sharding that reaches an input leaf requires a compatible default mesh, including
    /// a source for an eager host upload. The upload ignores the resolved source. A host leaf lifted into a trace uses
    /// its source like other traced leaves. Each option tree is validated against the nodes of `x` (refer to
    /// [`Parameterized::paths`] and [`Parameterized::broadcast_to_parameter_structure`]), so every node of an option
    /// tree must be a node of `x`, and every node of `x` must be a node of the option tree or lie at or below one of
    /// its leaves (e.g., the targets `[d]` for an input `[]`, `[[d]]` for `[[x], []]`, `[[d], []]` for `[[x]]`,
    /// and `[[]]` for `[]` are all rejected). An option tree of another type with the same paths (e.g., a [`Vec`] for
    /// an array, or another derived struct with the same parameter fields) matches, and so does a node whose static
    /// metadata differs from that of `x`. Map keys are matched by their [`Debug`](std::fmt::Debug) text. In maps
    /// reached by the prefix, callers must ensure that corresponding keys in the input and option tree render
    /// identically and that different keys render different. An input without leaves is returned as it is
    /// once its options are validated.
    ///
    /// # Targets and Traced Values
    ///
    /// With no target, a host leaf goes to the default device of this handle (refer to
    /// [`Self::with_default_device`]), as `jax.device_put` uploads it under `jax.default_device`, or otherwise to the
    /// first addressable device, whether or not this session has a [default mesh](Self::device_mesh), and an existing
    /// [`Array`] leaf keeps its current placement. A target that names only a memory or a layout uploads a host leaf to
    /// the same device. Use [`Self::placement`] to put a sharding on the session's default mesh. Every [`Array`] placed
    /// with a target is [committed](Array::is_committed) to its devices. An [`Array`] leaf without a target keeps its
    /// commitment and a host leaf without one is uncommitted, so operations with committed inputs can move it to their
    /// devices. Every resulting [`Array`] belongs to this session, including one that keeps its placement. An [`Array`]
    /// leaf of another PJRT client is rejected unless its target names devices, onto which it is read back and uploaded
    /// (refer to [`Array::to_placement`]). A traced leaf executes its placement with the session of the arrays that it
    /// wraps (e.g., the input of a batched or differentiated function, including every primal, tangent, and packed
    /// batch). That session cannot be replaced. A traced leaf whose arrays belong to another PJRT client is therefore
    /// rejected, with or without a target, with the error of such an [`Array`] leaf without a target, and one whose
    /// arrays belong to another session of the same client is rejected with a [`ryft_pjrt::Error::InvalidArgument`]
    /// error. Other handles of this session (e.g., one with another default mesh) are accepted, and the results of
    /// a traced leaf stay attached to the handles of the arrays that it wraps.
    ///
    /// A [`PlacementRequest::Device`] target leaves every leaf wholly on that device. A single-device [`Array`] keeps
    /// its logical mesh and sharding after completing its reduction state. A multi-device [`Array`] gathers onto the
    /// device unless it is unreduced over an axis larger than one, since a placement cannot combine the contributions
    /// of several devices. A bare device also selects the device's default memory. A mesh or sharding target without a
    /// memory keeps the leaf's memory (refer to [`PlacementRequest::target_memory`]). A target without a layout keeps
    /// the leaf's layout only if it still describes each result shard.
    ///
    /// A [`PlacementRequest::InPlace`] target changes only memory or layout, even when the devices and sharding are
    /// unknown during tracing (e.g., how a compiled input is split along auto mesh axes). It commits eager results
    /// to their current devices and commits all public outputs of a compiled program that retains the placement.
    /// Use [`transfer_to_memory`](ryft_core::TransferToMemory) for memory-only moves. With that operation, commitment
    /// depends on the inputs and the surrounding program. Use [`constrain_sharding`](ryft_core::ConstrainSharding)
    /// to constrain auto axes while preserving tracked sharding.
    ///
    /// Inside a batched function, the placement of a batch item (refer to [`Placed`](ryft_core::Placed)) records
    /// how its batch is distributed. A target or source that is such a [`Placement`] keeps that record (refer to
    /// [`Placement::batch_shardings`]), so that placing a batch item on its own placement leaves every batch item
    /// on the devices that hold it.
    ///
    /// Traced leaves stay traced and record requests through [`PlaceDispatch`]. Every leaf follows the type rule of
    /// [`PlaceOperation`] after shorter shardings are padded with replicated dimensions. Eager [`Array`] and host
    /// leaves allow uneven splits and validate requests before moving or uploading data (refer to
    /// [`PlaceOperation::infer_output_type_allowing_uneven_splits`]). A host leaf is validated as a
    /// value without tracked sharding in device memory, so a reduced or unreduced target is rejected.
    ///
    /// The type rule prohibits moves inside or into manual regions and requires targets to preserve reduction state,
    /// subject to the [`PlacementRequest::Device`] rules above. This includes any pending or completed reduction over
    /// an auto axis of a runtime leaf. [`Array::to_placement`] can combine contributions and change reduction state
    /// when its strategies support the target.
    ///
    /// Omitting the target of a traced leaf still stages a placement. It keeps the value on its devices with its
    /// sharding, memory, and layout, as JAX's `device_put` does without a device. That placement records the provided
    /// source. Without one, it records where the value concretely lives wherever it executes on a value that lives
    /// somewhere (e.g., when a differentiated function is evaluated on concrete arrays, refer to
    /// [`Placed`](ryft_core::Placed)). Reverse mode differentiation moves the cotangent back to that source. Such
    /// a placement is staged for every traced leaf, including one without a concrete placement and without a source,
    /// so that its copy policy applies wherever the staged program executes on concrete arrays.
    ///
    /// A placement inside a compiled function must target the device mesh that the function is compiled for (refer
    /// to [`LoweringError::PlacementOutsideProgramMesh`](crate::LoweringError::PlacementOutsideProgramMesh)), so a
    /// compiled function cannot move a value to other devices. The devices of a traced leaf that wraps no concrete
    /// array are not known while it is traced, so a bare device target stages a [`PlacementRequest::Device`] with its
    /// memory and layout, whose type rule selects the default memory of the device when it names none, as it does for
    /// a traced leaf that lives on a mesh of one device. Its type keeps the sharding of the leaf without the reduction
    /// state that one device completes, so it combines with the other values of the function. A compiled function
    /// accepts it only when the device is the only device of the mesh that the function is compiled for (refer to
    /// [`LoweringError::DeviceOutsideProgramMesh`](crate::LoweringError::DeviceOutsideProgramMesh)), and the outputs
    /// of such a function are committed, as an explicit device commits an [`Array`] eagerly. Wherever the staged
    /// placement executes on a concrete array instead (e.g., when a traced program is evaluated on arrays), the device
    /// resolves against that array as it does for an [`Array`] leaf. A bare device source of a traced leaf whose
    /// devices are not known stages a [`PlacementRequest::Device`] source, so that reverse mode differentiation moves
    /// the cotangent back to that device, and transposing the result again moves it to the target once more. Inside a
    /// compiled function, a target must lie on the mesh that the function is compiled for, whereas a source only states
    /// where the input lives, which the type rule checks against the type of the input, and it is not checked against
    /// that mesh, because the function moves nothing to it. The reverse mode of such a function moves the cotangent to
    /// the source, so compiling it rejects a source outside that mesh as a target. Inside a manual region (e.g., a
    /// `shard_map` body), where no value can be moved, a target may name only a memory or a layout, and is otherwise
    /// rejected with [`LoweringError::UnsupportedOperation`](crate::LoweringError::UnsupportedOperation), in compiled
    /// functions and in eager `shard_map` bodies alike. A source may name any devices there, because the forward
    /// placement moves nothing to them. The reverse mode of such a function moves the cotangent to that source, unless
    /// transposition leaves the cotangent where it lives (i.e., the placement has no target or an in-place one, and
    /// the leaf varies along the manual axes of the region or carries reduction state over them, refer to
    /// [`PlaceOperation::transposes_in_place`]), so compiling that program rejects such a source as a target.
    ///
    /// # Donation and Aliasing
    ///
    /// The donation option is only applied to concrete [`Array`] leaves, and it is ignored for host leaves and traced
    /// leaves. Each aliasing option leaf converts into `Option<bool>`, so `false` means `Some(false)` and `true` means
    /// `Some(true)`. When placing a concrete [`Array`] leaf with `Some(false)` and without donation, the result holds
    /// no buffer of the leaf, even when the leaf was the only handle of its buffers. The buffers that the placement
    /// creates are used directly. A traced leaf with `Some(false)` and without donation stages its placement with a
    /// forced copy (refer to [`PlaceOperation::with_may_alias`]), so that when a batched or differentiated function
    /// is evaluated on concrete arrays, its result shares no buffer with any other array. A compiled function has no
    /// observable buffer identity, so the option has no effect inside one. Host leaves ignore the option, because their
    /// uploads never share buffers. Donation decides when both options are given. A donated leaf with `Some(false)` is
    /// placed as a donated leaf without an aliasing option, because the caller gives the leaf up, so there is no alias
    /// for `Some(false)` to forbid, and its result may hold the buffers of the leaf (e.g., the leaf itself when it
    /// already lives at its target). A leaf cannot be donated while its `may_alias` option is `Some(true)`, which
    /// this function reports with [`ArrayError::DonatedArrayMayAlias`] before it places any leaf.
    ///
    /// # Sources
    ///
    /// The source option states where a leaf is expected to live. Its devices are informative on every path, so they
    /// are never compared with the devices that a leaf actually lives on, and reverse mode differentiation sends the
    /// cotangent to the devices that it names. Eager placement checks target devices against its session's inventory.
    /// The compiling session checks an abstract tracer's target. The type rule of [`PlaceOperation`] checks the
    /// declared source sharding, memory, and layout against the concrete [`Array`] leaf's type and runtime sharding,
    /// including the reduction state and shards along auto mesh axes that its type omits. It performs the same check on
    /// a traced leaf that wraps a concrete array (i.e., [`PlaceOperation::infer_output_type_allowing_uneven_splits`]).
    /// A traced leaf wrapping a concrete array then fills in a source that names only memory or layout with the array's
    /// placement. For a single-device array with a source naming a device, it fills in the source with that placement
    /// moved to the named device (refer to [`PlacementRequest::resolve_source`]). Transposition uses this placement to
    /// return the cotangent to its devices and account for its runtime reduction state. An expected layout must match
    /// the shape of each shard determined by the leaf's runtime sharding or its staged type. It must also equal any
    /// layout declared by the leaf's type. Only the type rule checks the source, so it does not inspect physical buffer
    /// layouts when the leaf's type declares no layout.
    ///
    /// # Multi-Process Placement
    ///
    /// In a run with more than one process, every process must make the same call with the same options and tree
    /// structure. For each array leaf, they must agree on whether it is locally assigned and whether it belongs to
    /// their own session's client. Placements communicate across processes in leaf order. With comparison disabled,
    /// each participant must finish its local placements before a subsequent cross-process transfer can complete.
    /// Each process uploads the whole host value to its addressable target devices. When the target spans more than one
    /// process, every process must pass the same logical value for that leaf, or the processes construct conflicting
    /// replicas of one array. Differing values need not produce a conflicting replica, because each process contributes
    /// only the part that its devices hold. For example, inputs `[1, 1]` on one process and `[2, 2]` on another, placed
    /// fully partitioned over one device of each, would form the array `[1, 2]`. The same applies to any locally
    /// assigned [`Array`] leaf, which each process places from its own value. An array is locally assigned when it is
    /// uploaded or created without a target that names devices. A target that names only a memory can commit an array
    /// without ending its local assignment, which lasts until a placement names devices.
    ///
    /// With [`SessionBuilder::value_comparison`](crate::SessionBuilder::value_comparison) enabled, a mesh or sharding
    /// target spanning every process in a multi-process run compares the host or locally assigned values for that
    /// leaf. After the local checks and before placing any leaf, each process hashes its compared leaves with SHA-256
    /// in leaf order. This reads back each compared [`Array`] on the calling thread. The comparison also binds its
    /// position in the session's sequence of comparisons. A process contributes a poison digest if the call's admission
    /// checks fail. The processes compare their digests bit for bit using a compiled all-reduce on a shared native PJRT
    /// device topology. If comparison is required, a session with explicitly declared peers returns
    /// [`SessionError::PlacementComparisonUnavailable`](crate::SessionError::PlacementComparisonUnavailable) for a
    /// call that passes its local checks. A process whose checks fail returns its own error. The comparison fails on
    /// every process with [`ryft_pjrt::Error::FailedPrecondition`] when the values differ, including in data type or
    /// shape, when a process cannot digest one of its values (e.g., one whose bytes do not encode its data type) or
    /// fails its checks, or when the processes compare at different positions. Readiness, readbacks, and operations
    /// that read a result of a compared leaf report a failed comparison.
    ///
    /// Whole-call admission covers the tree presented to one placement call. An error on a rank before that call
    /// reaches admission is reported on that rank only. For example, an enclosing transform can reject a local
    /// input while a peer with a valid input reaches the collective.
    ///
    /// A placement tree with multiple leaves is admitted as a whole when a leaf's placement can make one process wait
    /// for another (e.g., an [`Array`] that moves shards between processes). Each process submits the comparison as
    /// above, even if it has no values to compare, and waits for its verdict before it places any leaf. Admission
    /// checks host encodings and reports known producer failures for locally assigned arrays with an explicit target
    /// or a forced copy. When a leaf fails its checks on one process only (e.g., a host scalar placed onto a split
    /// target on one process and a host vector on another), that process returns its own error and every other process
    /// returns a [`ryft_pjrt::Error::FailedPrecondition`] before any leaf is placed. A failure after admission can
    /// still stop one process before its peers reach the same transfer. This includes backend failures when submitting
    /// work and failures reported asynchronously by target buffers.
    ///
    /// Unlike the `np.equal` of JAX, the comparison is bitwise (e.g., `+0` and `-0` differ, and NaN values with equal
    /// bits agree). What the check cannot turn into an error remains a precondition: every process must make the same
    /// sequence of comparing calls (of this function, of [`Array::to_placement`] and [`Array::into_placement`], and
    /// of eager placements) with trees of the same structure and the same targets, because a process that skips a
    /// comparison or stops calling leaves the others waiting in it. Equal values remain an unchecked precondition
    /// for a target on the devices of some of the processes only, for a device target, and for every placement
    /// of a session that disables the check. [`Placeable::place_parameter`] compares nothing on its own. A host leaf
    /// without a target, or with one that names only its memory or layout, is uploaded independently onto a device
    /// of each process, so those leaves may hold different process-local values (e.g., different batches). Data that
    /// differs across processes and forms one distributed array (e.g., the local part of a value that is split across
    /// processes) is placed with [`Array::from_host_shards`] instead, which takes from each process the bytes of the
    /// shards of its own devices.
    ///
    /// # Example
    ///
    /// The bare device below is the target of the subtree `(b, c)` alone, so `a` keeps its placement and its
    /// commitment. Passing clones keeps `array` for the caller.
    ///
    /// ```rust
    /// # use ryft_core::{ArrayType, DataType};
    /// # use ryft_xla::{Array, PlacementOptions, Platform, Session};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let session = Session::builder().platform(Platform::Cpu).build()?;
    /// let r#type = ArrayType::new_static(DataType::F32, [2]);
    /// let array = Array::from_host_buffer(&session, r#type, None, None, [1f32, 2.].map(f32::to_ne_bytes).concat())?;
    /// let device = array.mesh().devices()[0];
    /// let options = PlacementOptions::to((None, Some(device)));
    /// let (a, (b, c)) = session.place((array.clone(), (array.clone(), array.clone())), options)?;
    /// assert_eq!((a.is_committed(), b.is_committed(), c.is_committed()), (false, true, true));
    /// assert_eq!(b.mesh().devices(), [device]);
    /// assert_eq!(array.copy_to_host()?.elements::<f32>()?, vec![1., 2.]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Parameters
    ///
    ///   - `x`: Leaf or tree of leaves to place.
    ///   - `options`: Target, expected source, donation, and aliasing options for the leaves of `x`.
    ///
    /// # Errors
    ///
    /// Before it places any leaf, this function checks that every option prefix broadcasts over `x`, and then, for
    /// each leaf in order, that it is not donated while it may alias and that its in-place source and target name a
    /// memory or a layout. An empty [`PlacementRequest::InPlace`] produces the [`TypeError`](ryft_core::TypeError)
    /// from [`PlaceOperation`]. The requested storage must also be supported. Mixed leaves from different traces or
    /// an unavailable group comparison can fail the whole call. Before placement, this function checks each leaf's
    /// metadata. For host values and concrete arrays, this includes the source and target type rules, the array's PJRT
    /// client and the session's inventory of target devices. Forward source devices are advisory. The compiling session
    /// checks the targets of abstract tracers. A preflight error leaves every leaf untouched, donated ones included.
    /// Placing a leaf can still fail after preceding leaves have been placed. Causes include failure of submitted work
    /// (e.g., a PJRT transfer, compiled reshard, or allocation), unsupported memory or layout, a move that no strategy
    /// of [`crate::arrays::transfers`] supports (e.g., one that would have to combine the contributions of a pending
    /// reduction), the bytes of a host array that do not encode valid elements, and the errors of the type rule for a
    /// traced leaf, which the context that records or executes its placement applies. Placement drops the results of
    /// the preceding leaves and leaves their submitted transfers and donations to complete, so only such a failure can
    /// have consumed the donated inputs of the leaves that precede the failing one. Every error of a leaf other than
    /// the root of `x` is wrapped in [`ArrayError::FailedParameterPlacement`] with the path of that leaf, except an
    /// [`ArrayError::DonatedArrayMayAlias`] or [`ArrayError::FailedParameterPlacement`] that the leaf returns (e.g.,
    /// from a placement of a tree of its own), whose path is prefixed with the path of that leaf instead.
    #[inline]
    pub fn place<
        P: Placeable,
        Input: Parameterized<P>,
        TargetPlacement: SessionPlacementRequestParameter,
        Target: Parameterized<TargetPlacement>,
        SourcePlacement: SessionPlacementRequestParameter,
        Source: Parameterized<SourcePlacement>,
        Donate: Parameterized<bool>,
        Aliasing: Clone + Parameter + Into<Option<bool>>,
        MayAlias: Parameterized<Aliasing>,
    >(
        &self,
        x: Input,
        options: PlacementOptions<Target, Source, Donate, MayAlias>,
    ) -> Result<<Input as Parameterized<P>>::To<P::Output>, Error>
    where
        <Input as Parameterized<P>>::Family: ParameterizedFamily<P::Output>,
    {
        self.place_with_comparison(x, options, ValueComparison::of_session(self)?)
    }

    /// Implements [`Self::place`] with `comparison`, the comparison of the values of the processes that the call makes
    /// for the leaves that every process places on its own (refer to [`PlaceableParameter::process_value`]), if any.
    pub(crate) fn place_with_comparison<
        P: Placeable,
        Input: Parameterized<P>,
        TargetPlacement: SessionPlacementRequestParameter,
        Target: Parameterized<TargetPlacement>,
        SourcePlacement: SessionPlacementRequestParameter,
        Source: Parameterized<SourcePlacement>,
        Donate: Parameterized<bool>,
        Aliasing: Clone + Parameter + Into<Option<bool>>,
        MayAlias: Parameterized<Aliasing>,
    >(
        &self,
        x: Input,
        options: PlacementOptions<Target, Source, Donate, MayAlias>,
        comparison: Option<&ValueComparison>,
    ) -> Result<<Input as Parameterized<P>>::To<P::Output>, Error>
    where
        <Input as Parameterized<P>>::Family: ParameterizedFamily<P::Output>,
    {
        let structure = x.parameter_structure();
        let (target, source, donate, may_alias) = options.into_parts();
        let target_values = PlacementOptionValues::from_prefix(target, &structure)?;
        let source_values = PlacementOptionValues::from_prefix(source, &structure)?;
        let donate_values = PlacementOptionValues::from_prefix(donate, &structure)?;
        let may_alias_values = PlacementOptionValues::from_prefix(may_alias, &structure)?;

        // Resolve only requests that reached a leaf. Preserve the targets of the other leaves if a request fails,
        // so distributed admission can still determine whether this call would otherwise communicate.
        let path = |index: usize| structure.parameter_paths().nth(index).unwrap();
        let mut checked = Ok(());
        let mut arguments = x
            .into_parameters()
            .enumerate()
            .map(|(index, parameter)| {
                let mut resolve = |result: Result<Option<PlacementRequest>, Error>| match result {
                    Ok(request) => request,
                    Err(error) => {
                        if checked.is_ok() {
                            checked = Err(error.with_placement_parameter_path(|| path(index)));
                        }
                        None
                    }
                };
                let target = resolve(target_values.get(index).cloned().map_or(Ok(None), |r| r.resolve(self)));
                let source = resolve(source_values.get(index).cloned().map_or(Ok(None), |r| r.resolve(self)));
                let donate = donate_values.get(index).copied().unwrap_or_default();
                let may_alias = may_alias_values.get(index).cloned().and_then(Into::into);
                if checked.is_ok() {
                    checked =
                        validate_parameter_options(|| path(index), target.as_ref(), source.as_ref(), donate, may_alias);
                }
                PlacementArgument { parameter, context: (), target, source, donate, may_alias }
            })
            .collect::<Vec<_>>();

        // Validate the options of the whole tree before inspecting its values, and validate every value before
        // preparation can upload host values or import them into a trace.
        let checked = checked
            .and_then(|()| {
                arguments.iter().enumerate().try_for_each(|(index, argument)| {
                    argument
                        .parameter
                        .validate_placement(
                            self,
                            argument.target.as_ref(),
                            argument.source.as_ref(),
                            argument.donate,
                            argument.may_alias,
                        )
                        .map_err(|error| error.with_placement_parameter_path(|| path(index)))
                })
            })
            .and_then(|()| P::prepare_arguments(&mut arguments, self, &path));

        let output_parameters = admit_and_place_arguments(
            self,
            comparison,
            arguments,
            checked,
            |_, argument| {
                argument.parameter.place_parameter(
                    self,
                    argument.target,
                    argument.source,
                    argument.donate,
                    argument.may_alias,
                )
            },
            |index, error| error.with_placement_parameter_path(|| path(index)),
        )?;

        Input::To::<P::Output>::from_parameters(structure, output_parameters).map_err(Into::into)
    }
}

/// Exposes the borrowed [`Array`] handles carried by a value. Traced placement runs in the tracer's existing contexts,
/// so [`Session::place`] checks that those handles belong to the receiving session and its PJRT client.
trait RuntimeArrays {
    /// Returns borrowed handles to the runtime arrays of this value, including the array itself or arrays wrapped
    /// by a tracer (e.g., those in the primal and tangent of a differentiated value). Host values and abstract
    /// tracers return an empty list.
    fn runtime_arrays(&self) -> Vec<&Array>;
}

impl RuntimeArrays for Value {
    #[inline]
    fn runtime_arrays(&self) -> Vec<&Array> {
        match self {
            Self::Array(value) => vec![value],
            Self::Dimension(_) | Self::Reference(_) => Vec::new(),
        }
    }
}

impl<V: RuntimeArrays> RuntimeArrays for ProjectedValue<ArrayType, V> {
    #[inline]
    fn runtime_arrays(&self) -> Vec<&Array> {
        self.value().runtime_arrays()
    }
}

impl RuntimeArrays for Array {
    #[inline]
    fn runtime_arrays(&self) -> Vec<&Array> {
        vec![self]
    }
}

impl RuntimeArrays for CoreArray {
    #[inline]
    fn runtime_arrays(&self) -> Vec<&Array> {
        Vec::new()
    }
}

impl<C: Context> RuntimeArrays for Tracer<C> {
    #[inline]
    fn runtime_arrays(&self) -> Vec<&Array> {
        Vec::new()
    }
}

impl<C: Context<Value: RuntimeArrays>> RuntimeArrays for PartialTracer<C> {
    #[inline]
    fn runtime_arrays(&self) -> Vec<&Array> {
        // An unknown value has no payload yet, and a poisoned value surfaces its deferred error when it is used.
        match self.value().ok().and_then(|value| value.as_known()) {
            Some(value) => value.runtime_arrays(),
            None => Vec::new(),
        }
    }
}

impl<C: Context<Value: RuntimeArrays>, P: BatchingPolicy<C>> RuntimeArrays for BatchingTracer<C, P> {
    #[inline]
    fn runtime_arrays(&self) -> Vec<&Array> {
        P::value(self.batch()).runtime_arrays()
    }
}

impl<C: Context<Value: RuntimeArrays>, P: DifferentiationPolicy<C>> RuntimeArrays for DifferentiationTracer<C, P> {
    #[inline]
    fn runtime_arrays(&self) -> Vec<&Array> {
        let mut arrays = self.primal().runtime_arrays();
        if let Some(tangent) = self.tangent().as_value() {
            arrays.extend(tangent.runtime_arrays());
        }
        arrays
    }
}

/// Option values of the leaves of one [`Session::place`] input, produced by [`Self::from_prefix`]. An omitted
/// prefix and a prefix with one leaf at the root give every leaf the same value, which is stored once.
#[derive(Clone, Debug, PartialEq, Eq)]
enum PlacementOptionValues<V> {
    /// Value of every leaf, or `None` if no prefix was given.
    Uniform(Option<V>),

    /// Values of the leaves, in the order of the leaves.
    PerParameter(Vec<V>),
}

impl<V> PlacementOptionValues<V> {
    /// Expands an optional placement prefix into its value for each input leaf. A prefix with one leaf at the root
    /// covers the whole input, so it needs no path matching or structure cloning, and neither it nor an omitted
    /// prefix allocates per leaf. Structured prefixes use [`Parameterized::broadcast_parameters`] to validate
    /// every node, including empty containers.
    fn from_prefix<P: Parameterized<V>, S: Parameterized<Placeholder>>(
        prefix: Option<P>,
        structure: &S,
    ) -> Result<Self, Error>
    where
        V: Clone + Parameter,
    {
        let Some(prefix) = prefix else {
            return Ok(Self::Uniform(None));
        };

        // A sole parameter below the root still belongs to a structured prefix, so it needs full tree validation.
        if prefix.parameter_count() == 1 && prefix.named_parameters().next().is_some_and(|(path, _)| path.is_root()) {
            return Ok(Self::Uniform(prefix.into_parameters().next()));
        }

        Ok(Self::PerParameter(prefix.broadcast_parameters(structure)?))
    }

    /// Returns the supplied value of the leaf at `index`, if any.
    fn get(&self, index: usize) -> Option<&V> {
        match self {
            Self::Uniform(value) => value.as_ref(),
            Self::PerParameter(values) => Some(&values[index]),
        }
    }
}

/// One input parameter and its expanded placement options. [`place_operation_bindings`] keeps the binding's session
/// in `context`, while [`Session::place`] shares its session across the arguments and uses `()` here.
struct PlacementArgument<P, C = ()> {
    /// Input parameter to place.
    parameter: P,

    /// Session that places `parameter` when the arguments of a call do not share one (i.e., the session of the binding
    /// of a core placement operation), and `()` otherwise.
    context: C,

    /// Target placement of `parameter`, if one was specified.
    target: Option<PlacementRequest>,

    /// Expected source placement of `parameter`, if one was specified.
    source: Option<PlacementRequest>,

    /// Whether a compiled reshard may reuse the buffers of `parameter` for the result.
    donate: bool,

    /// Whether the result may share buffers with other arrays. `Some(false)` forbids it, unless `donate` is `true`,
    /// which gives `parameter` up.
    may_alias: Option<bool>,
}

impl Array {
    /// Resolves a request against this array for physical placement. An in-place request keeps its mesh and sharding.
    /// A device request keeps the logical mesh and sharding of a single-device array, with its reduction state
    /// completed, or gathers a multi-device array onto the canonical single-device mesh. Resolved requests keep
    /// their mesh and are padded to the rank of the array.
    pub(super) fn resolve_placement_request(&self, request: PlacementRequest) -> Result<Placement, Error> {
        validate_parameter_options(ParameterPath::root, Some(&request), None, false, None)?;

        let rank = self.sharding().rank();
        let memory = request.target_memory();
        let (mut placement, layout) = match request.padded(rank)? {
            PlacementRequest::Resolved(placement) => (placement, None),
            PlacementRequest::Device { device, layout, .. } if self.mesh().devices().len() == 1 => {
                // Every axis of a single-device mesh has size one, so its reduction state is complete on the target.
                let mesh = DeviceMesh::new(self.mesh().logical_mesh().clone(), vec![device])?;
                let sharding = self
                    .sharding()
                    .clone()
                    .with_unreduced_axes(Vec::<String>::new())?
                    .with_unreduced_kind(UnreducedKind::Sum)?
                    .with_reduced_axes(Vec::<String>::new())?;
                (Placement::new(mesh, sharding)?, layout)
            }
            PlacementRequest::Device { device, layout, .. } => (Placement::single_device(device, rank), layout),
            PlacementRequest::InPlace { layout, .. } => {
                (Placement::new(self.mesh().clone(), self.sharding().clone())?, layout)
            }
        };

        if let Some(layout) = layout {
            placement = placement.with_layout(layout);
        }

        if let Some(memory) = memory {
            placement = placement.with_memory(memory);
        }

        // Reuse the type rule's rank diagnostic while leaving the remaining checks to the physical placement path,
        // which supports changes to reduction state that the full type rule would reject.
        let r#type = self.r#type();

        if placement.sharding().rank() != r#type.rank() {
            PlaceOperation::infer_output_type_allowing_uneven_splits(
                &r#type,
                None,
                Some(&PlacementRequest::Resolved(placement.clone())),
                None,
            )?;
        }

        Ok(placement)
    }
}

impl Error {
    /// Adds the parameter path to this placement error. A root path returns the original error. A nested
    /// [`ArrayError::DonatedArrayMayAlias`] or [`ArrayError::FailedParameterPlacement`] keeps its variant and source,
    /// with the parameter path prefixed to its existing path. Other errors are wrapped in
    /// [`ArrayError::FailedParameterPlacement`]. Callers supply `path` lazily to resolve it only on failure.
    fn with_placement_parameter_path(self, path: impl FnOnce() -> ParameterPath) -> Self {
        let path = path();
        if path.is_root() {
            return self;
        }
        let prefixed = |inner: ParameterPath| inner.segments().cloned().fold(path.clone(), ParameterPath::with_segment);
        match self {
            Self::Array(array_error) => match *array_error {
                ArrayError::DonatedArrayMayAlias { path: inner } => {
                    ArrayError::DonatedArrayMayAlias { path: prefixed(inner) }.into()
                }
                ArrayError::FailedParameterPlacement { path: inner, source } => {
                    ArrayError::FailedParameterPlacement { path: prefixed(inner), source }.into()
                }
                array_error => {
                    ArrayError::FailedParameterPlacement { path, source: Arc::new(array_error.into()) }.into()
                }
            },
            error => ArrayError::FailedParameterPlacement { path, source: Arc::new(error) }.into(),
        }
    }
}

/// Places the array inputs bound to the given core placement operations as a group. The core operation requires even
/// splits and keeps the input client, while the session frontend also admits uneven splits and explicit cross-client
/// transfers. When the call requires distributed admission, a process whose local check fails still participates
/// before returning its error.
pub(crate) fn place_operation_bindings(
    bindings: Vec<(Session, Array, PlaceOperation)>,
) -> Result<Vec<Array>, ProgramError> {
    ensure_effect_dispatch_is_allowed()?;
    let Some((session, _, _)) = bindings.first() else { return Ok(Vec::new()) };
    let session = session.clone();
    let comparison = if bindings.iter().all(|(_, _, operation)| operation.target().is_none()) {
        None
    } else if bindings.iter().all(|(context, _, _)| context.is_same_session(&session)) {
        ValueComparison::of_session(&session)?
    } else {
        // Each binding keeps its own context. Independent placements can use different contexts.
        // A call that needs agreement requires one comparison sequence shared by all its arguments.
        let communicates =
            bindings.iter().any(|(context, array, operation)| array.communicates(context, operation.target()));
        for (context, array, operation) in &bindings {
            let target = operation.target();
            if ValueComparison::of_session(context)?
                .is_some_and(|comparison| communicates || (array.is_locally_assigned() && comparison.applies(target)))
            {
                return Err(ProgramError::InvalidArgument {
                    message: "distributed core placement requires the leaves of a tree to share one session"
                        .to_string(),
                });
            }
        }
        None
    };

    let arguments = bindings
        .into_iter()
        .map(|(context, parameter, operation)| PlacementArgument {
            parameter,
            context,
            target: operation.target().cloned(),
            source: operation.source().cloned(),
            donate: false,
            may_alias: Some(operation.may_alias()),
        })
        .collect::<Vec<_>>();

    let checked = arguments.iter().try_for_each(|argument| {
        let array = &argument.parameter;
        let context = &argument.context;
        let rank = array.r#type().rank();
        let target = argument.target.clone().map(|target| target.padded(rank)).transpose()?;
        let source = argument.source.clone().map(|source| source.padded(rank)).transpose()?;
        PlaceOperation::new(target.clone(), source).infer_output_type_of_value(array, Some(array.sharding()), true)?;
        context.validate_input_placement(std::slice::from_ref(array), &[])?;
        array.validate_placement_client(context, target.as_ref())?;
        array.validate_array_target(context, target)
    });

    admit_and_place_arguments(
        &session,
        comparison,
        arguments,
        checked,
        |_, argument| {
            let rank = argument.parameter.r#type().rank();
            let target = argument.target.map(|target| target.padded(rank)).transpose()?;
            argument.parameter.place_validated(&argument.context, target, false, argument.may_alias)
        },
        |_, error| error,
    )
    .map_err(Into::into)
}

/// Coordinates any required distributed admission, then places the arguments on every participating process.
fn admit_and_place_arguments<P: Placeable, C>(
    session: &Session,
    comparison: Option<&ValueComparison>,
    arguments: Vec<PlacementArgument<P, C>>,
    checked: Result<(), Error>,
    place: impl Fn(usize, PlacementArgument<P, C>) -> Result<P::Output, Error>,
    map_error: impl Fn(usize, Error) -> Error,
) -> Result<Vec<P::Output>, Error> {
    let Some(comparison) = comparison else {
        checked?;
        return arguments
            .into_iter()
            .enumerate()
            .map(|(index, argument)| place(index, argument).map_err(|error| map_error(index, error)))
            .collect();
    };

    // Ranks decide whether work communicates from shared placement metadata before a local validation can fail.
    let process_values = arguments.iter().map(|argument| argument.parameter.process_value()).collect::<Vec<_>>();
    let admits = arguments.len() > 1
        && arguments.iter().any(|argument| argument.parameter.communicates(session, argument.target.as_ref()));
    let compared = arguments
        .iter()
        .zip(&process_values)
        .map(|(argument, value)| value.is_some() && comparison.applies(argument.target.as_ref()))
        .collect::<Vec<_>>();
    let mut joins = vec![None; arguments.len()];
    let verdict = if admits || compared.iter().any(|compared| *compared) {
        let collective = match comparison.collective(session) {
            Ok(collective) => collective,
            Err(error) => {
                checked?;
                return Err(error.into());
            }
        };

        let mut checked = checked;
        let mut digests = Vec::new();
        for (index, value) in process_values.into_iter().enumerate() {
            let Some(value) = value else { continue };
            if compared[index] {
                joins[index] = Some(value.join());
                digests.push(if checked.is_ok() { value.digest() } else { None });
            } else if admits && checked.is_ok() {
                let argument = &arguments[index];

                // A forced copy requires ready buffers even when the caller supplies no target.
                let copies = argument.may_alias == Some(false) && !argument.donate;

                let readiness = if argument.target.is_some() || copies {
                    sealed::ArrayReadiness::Required
                } else {
                    sealed::ArrayReadiness::Unneeded
                };

                checked = value.validate(readiness).map_err(|error| map_error(index, error));
            }
        }

        let digest = checked.as_ref().ok().and_then(|_| combined_digest(digests));
        let verdict = collective.submit(session, digest);
        checked?;
        let verdict = verdict?;
        if admits {
            verdict.block_until_ready()?;
            None
        } else {
            Some(verdict)
        }
    } else {
        drop(process_values);
        checked?;
        None
    };

    arguments
        .into_iter()
        .enumerate()
        .map(|(index, argument)| {
            let output = place(index, argument).map_err(|error| map_error(index, error))?;
            Ok(match (&verdict, joins[index]) {
                (Some(verdict), Some(join)) => join(output, verdict.clone()),
                _ => output,
            })
        })
        .collect()
}

/// Checks the precondition of [`Placeable::place_parameter`], which rejects the per-leaf options that no leaf accepts
/// and does not depend on the leaf. [`Session::place`] calls this function for every leaf before it places any leaf,
/// and every implementation of [`Placeable::place_parameter`] calls it first, with [`ParameterPath::root`] as `path`,
/// so that a leaf placed directly accepts the same options as one placed through [`Session::place`].
///
/// # Parameters
///
///   - `path`: Returns the path of the leaf, which is only called when an error reports it.
///   - `target`, `source`, `donate`, and `may_alias`: Leafwise options of [`Placeable::place_parameter`].
///
/// # Errors
///
/// Returns [`ArrayError::DonatedArrayMayAlias`] at the path that `path` returns for a `donate` of `true` with a
/// `may_alias` of `Some(true)`, and then a [`TypeError`](ryft_core::TypeError) with the diagnostic of the type rule of
/// [`PlaceOperation`] for an in-place source and then an in-place target that name neither a memory nor a layout. That
/// [`TypeError`](ryft_core::TypeError) is wrapped in [`ArrayError::FailedParameterPlacement`] when `path` returns a
/// path other than the root.
fn validate_parameter_options<P: FnOnce() -> ParameterPath>(
    path: P,
    target: Option<&PlacementRequest>,
    source: Option<&PlacementRequest>,
    donate: bool,
    may_alias: Option<bool>,
) -> Result<(), Error> {
    if donate && may_alias == Some(true) {
        return Err(ArrayError::DonatedArrayMayAlias { path: path() }.into());
    }

    // The type rule of `PlaceOperation` owns the checks of the requests that need no type of the leaf.
    PlaceOperation::validate_requests(target, source)
        .map_err(|error| Error::from(error).with_placement_parameter_path(path))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex, mpsc};
    use std::thread;
    use std::time::Duration;

    #[cfg(feature = "cuda-13")]
    use std::ops::Range;

    use half::bf16;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use ryft_core::{
        Add, BatchAxis, CompilationArtifactExchange, CompilationArtifactExchangeError,
        CompilationArtifactExchangePolicy, CompilationCache, Complex, ConvertElementType, DataType, DeviceMesh,
        DifferentiableType, Dimension, DomainTracingContext, Error as CoreError, Exp, Layout, LogicalMesh, Memory,
        MeshAxis, MeshAxisType, Neg, Operation as CoreOperation, PLACE_OPERATION_NAME, ParameterError, Place,
        PlaceOperation, Placed, ProgramError, Shape, ShardMapOptions, Sharding, ShardingDimension, StaticShape,
        StridedLayout, Tile, TileDimension, TiledLayout, TypeError, Typed, ValueDomainDispatch, ValueProjection, batch,
        differentiate_at, f8e4m3fn, shard_map,
    };
    use ryft_pjrt::{
        BufferSpecification, BufferType, Client, ClientOptions, CpuClientOptions, ExecutionFence, KeyValueStore,
        load_cpu_plugin,
    };

    #[cfg(feature = "cuda-13")]
    use ryft_core::{DimensionBounds, DimensionVariable};

    use crate::arrays::host::participant_mesh_of;
    use crate::arrays::tests::{
        first_device, pending_replicated_vector, place_through_entry_points, process_one_cpu_client, process_outcomes,
        shard_memory_kinds, transfer_managers, two_process_mesh_2x2,
    };
    use crate::distributed::tests::{run_cpu_processes, test_process};
    use crate::sessions::tests::{DeclareRemoteDevices, domain_mesh, two_device_mesh};
    use crate::tests::{
        cpu_client, execution_client, four_device_mesh_x, logical_mesh_2x2, native_buffer_addresses, other_cpu_client,
        session_with_remote_devices, sessions_of_processes, shard_values, values_from_bytes, values_to_bytes,
    };
    use crate::{
        ArrayTracer, CompileTracer, CompiledFunction, CompiledFunctionDispatcher, DistributedRuntime, ExecutionError,
        Options, RetainedArray, SessionError, StagedFunction, StatefulCompileTracer, ToPjrt, TracedProgram, Type,
        Value, compile, compile_with_options, jit, stage, trace,
    };

    #[cfg(feature = "cuda-13")]
    use crate::arrays::host::DIFFERENT_VALUES_MESSAGE;
    #[cfg(feature = "cuda-13")]
    use crate::arrays::tests::{cuda_rank_session, explicit_mesh};
    #[cfg(feature = "cuda-13")]
    use crate::distributed::tests::{read_addressable_f32s, run_as_cuda_rank, run_cuda_rank_pair};

    use super::*;

    /// Returns the path and the underlying error of an `ArrayError::FailedParameterPlacement`, and `None` for any other
    /// error.
    fn failed_parameter(error: &Error) -> Option<(String, &Error)> {
        match error {
            Error::Array(error) => match error.as_ref() {
                ArrayError::FailedParameterPlacement { path, source } => Some((path.to_string(), source.as_ref())),
                _ => None,
            },
            _ => None,
        }
    }

    /// Saves the original placement diagnostic for tests whose transform can defer or wrap it.
    fn record_placement_error<T>(error: &RefCell<Option<String>>, result: Result<T, Error>) -> Result<T, ProgramError> {
        result.map_err(|placement_error| {
            let message = placement_error.to_string();
            error.replace(Some(message.clone()));
            ProgramError::InvalidArgument { message }
        })
    }

    /// Returns the physical dimensions of the buffer that each device of `array` holds, in the device order of its
    /// mesh.
    fn shard_buffer_dimensions(array: &Array) -> Vec<Vec<u64>> {
        array.shards().iter().map(|shard| shard.buffer().unwrap().dimensions().unwrap().to_vec()).collect()
    }

    /// Returns the mesh of `device` alone, with the explicit axis `x` of size 1.
    fn single_device_mesh(device: Device) -> DeviceMesh {
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        DeviceMesh::new(logical_mesh, vec![device]).unwrap()
    }

    /// Every observable fact of one placed array, so that the arrays that different paths produce can be compared.
    #[derive(Clone, Debug, PartialEq)]
    struct Observed {
        devices: Vec<Device>,
        logical_mesh: LogicalMesh,
        sharding: Sharding,
        r#type: ArrayType,
        memory_kinds: Vec<String>,
        values: Vec<f32>,
        committed: bool,
    }

    /// Returns the [`Observed`] facts of `array`.
    fn observe(array: &Array) -> Observed {
        Observed {
            devices: array.mesh().devices().to_vec(),
            logical_mesh: array.mesh().logical_mesh().clone(),
            sharding: array.sharding().clone(),
            r#type: array.r#type().into_owned(),
            memory_kinds: array
                .addressable_shards()
                .map(|shard| shard.buffer().unwrap().memory().unwrap().kind().unwrap().into_owned())
                .collect(),
            values: array.copy_to_host().unwrap().elements::<f32>().unwrap(),
            committed: array.is_committed(),
        }
    }

    /// Traces a function that places a traced leaf of the type of `input` on `session` with `options`, and replays
    /// the traced program on `session` with `input`, which is the staged path of a placement. Returns the placed
    /// array, the error of the placement when it fails while tracing (from which the traced function recovers, refer
    /// to [`record_placement_error`]), and otherwise the error of the trace or of the replay.
    fn replay_placement(session: &Session, input: &Array, options: PlacementOptions) -> Result<Array, String> {
        let error = RefCell::new(None);
        let traced: Result<TracedProgram<ArrayType, ArrayType>, _> = trace(
            |x: ArrayTracer| record_placement_error(&error, session.place(x.clone(), options.clone())).unwrap_or(x),
            input.r#type().into_owned(),
        );
        match (error.take(), traced) {
            (Some(message), _) => Err(message),
            (None, Err(error)) => Err(error.to_string()),
            (None, Ok(traced)) => match traced.interpret_in_context(session, Value::Array(input.clone())) {
                Ok(Value::Array(output)) => Ok(output),
                Ok(_) => Err("placing an array must produce an array".to_string()),
                Err(error) => Err(error.to_string()),
            },
        }
    }

    /// Returns the text that identifies `error` in the outcome of a placement on one rank of a multi-process test:
    /// [`DIFFERENT_VALUES_MESSAGE`] when it reports the failed comparison of the values of the ranks, and otherwise the
    /// first line of the error.
    #[cfg(feature = "cuda-13")]
    fn placement_error_outcome(error: Error) -> String {
        match error {
            Error::Pjrt(PjrtError::FailedPrecondition { message, .. }) if message == DIFFERENT_VALUES_MESSAGE => {
                message
            }
            error => error.to_string().lines().next().unwrap_or("").to_string(),
        }
    }

    /// Returns the outcome of `placed` on this rank: the elements of the shards of this rank once the placed array is
    /// ready, or the [`placement_error_outcome`] of the error of the placement or of its readiness.
    #[cfg(feature = "cuda-13")]
    fn placement_outcome(placed: Result<Array, Error>) -> Result<Vec<Vec<f32>>, String> {
        placed
            .and_then(|placed| placed.block_until_ready().map(|()| placed))
            .map(|placed| read_addressable_f32s(&placed))
            .map_err(placement_error_outcome)
    }

    /// Places `source`, an array on the device of rank 0, replicated onto every device of `mesh`, which spans both
    /// ranks, and returns its [`placement_outcome`]. This ordinary cross-host placement completes only if both ranks
    /// are still in step, so tests run it after a placement that fails on some rank.
    #[cfg(feature = "cuda-13")]
    fn place_replicated_across_ranks(
        session: &Session,
        source: &Array,
        mesh: &DeviceMesh,
    ) -> Result<Vec<Vec<f32>>, String> {
        let target = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap();
        placement_outcome(source.to_placement(session, target))
    }

    /// Places on every rank of the job of `client` a host value with [`Session::place`] and an array that each rank
    /// uploaded without devices with [`Array::to_placement`] onto a split over the two global devices, with the same
    /// values on both ranks or, when `differ` is `true`, with values that differ between them, and then places an
    /// array of rank 0 replicated onto both devices with [`place_replicated_across_ranks`]. Returns the
    /// [`placement_outcome`] of these three placements on this rank, in that order.
    #[cfg(feature = "cuda-13")]
    fn place_values_of_ranks(client: &'static Client<'static>, differ: bool) -> [Result<Vec<Vec<f32>>, String>; 3] {
        let rank = client.process_index().unwrap();
        let devices = client
            .devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = |devices: Vec<Device>| {
            let axis = MeshAxis::new("x", devices.len(), MeshAxisType::Explicit).unwrap();
            DeviceMesh::new(LogicalMesh::new(vec![axis]).unwrap(), devices).unwrap()
        };
        let (process_0_mesh, forward_mesh) = (mesh(vec![devices[0]]), mesh(devices.clone()));
        let split = Sharding::new(forward_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]);
        let split = Placement::new(forward_mesh.clone(), split.unwrap()).unwrap();
        let offset = if differ { 10. * rank as f32 } else { 0. };
        let values = values_to_bytes(&[1f32, 2., 3., 4.].map(|value| value + offset));
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let host = CoreArray::new(r#type.clone(), values.clone()).unwrap();
        let host = placement_outcome(session.place(host, PlacementOptions::to(PlacementRequest::from(split.clone()))));
        let uploaded = Array::from_host_buffer(&session, r#type.clone(), None, None, values).unwrap();
        let uploaded = placement_outcome(uploaded.to_placement(&session, split));

        let r#type = r#type.with_sharding(Sharding::replicated(process_0_mesh.logical_mesh().clone(), 1)).unwrap();
        let values = values_to_bytes(&[17f32, 18., 19., 20.]);
        let source = Array::from_host_buffer(&session, r#type, process_0_mesh, None, values).unwrap();
        [host, uploaded, place_replicated_across_ranks(&session, &source, &forward_mesh)]
    }

    /// Runs `check` in one of the two CPU processes of a placement test (refer to [`run_cpu_processes`]), and returns
    /// immediately in a process that no launcher started. `check` receives the client of its process, which lists only
    /// the device of its own process, and the devices of the two processes in process order. Both processes stay
    /// connected until each of them has finished its check.
    fn run_as_cpu_process(check: impl FnOnce(&'static Client<'static>, [Device; 2])) {
        let Some((rank, address)) = test_process() else {
            return;
        };
        let plugin = load_cpu_plugin().unwrap();
        let runtime = Box::leak(Box::new(DistributedRuntime::initialize(&plugin, &address, 2, rank).unwrap()));
        let options = ClientOptions::CPU(CpuClientOptions { device_count: Some(1), process_id: Some(rank) });
        let client: &'static Client<'static> = Box::leak(Box::new(runtime.new_client(&plugin, options).unwrap()));
        assert_eq!(client.devices().unwrap().len(), 1);
        let local = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let key = |rank| format!("ryft/test/placement/device/{rank}");
        let store = runtime.key_value_store();
        let timeout = Duration::from_secs(5);
        store.put(key(rank).as_bytes(), &local.id().to_le_bytes()).unwrap();
        let devices = [0, 1].map(|rank| {
            let bytes = store.get(key(rank).as_bytes(), timeout).unwrap();
            Device::new(usize::from_le_bytes(bytes.try_into().unwrap()), rank)
        });
        check(client, devices);
        let finished = |rank| format!("ryft/test/placement/finished/{rank}");
        store.put(finished(rank).as_bytes(), b"ready").unwrap();
        store.get(finished(1 - rank).as_bytes(), timeout).unwrap();
        runtime.shutdown().unwrap();
    }

    /// Compilation artifact exchange of process 0 with one fictitious peer process. Process 0 leads every compilation
    /// of a program over its own devices, so it publishes such programs and never waits for its peer.
    struct PeerCompilationExchange;

    impl CompilationArtifactExchange for PeerCompilationExchange {
        fn process_index(&self) -> usize {
            0
        }

        fn process_count(&self) -> usize {
            2
        }

        fn publish(&self, _: &[u8], _: &[u8]) -> Result<(), CompilationArtifactExchangeError> {
            Ok(())
        }

        fn receive(&self, _: &[u8], _: Duration) -> Result<Option<Vec<u8>>, CompilationArtifactExchangeError> {
            panic!("process 0 leads the compilations of programs over its own devices and never waits for a peer");
        }
    }

    /// Returns a session of `client` that declares `remote_devices` and whose compilation cache must share every
    /// artifact with a peer process.
    fn peer_exchange_session<D: IntoIterator<Item = Device>>(
        client: &'static ryft_pjrt::Client<'static>,
        remote_devices: D,
    ) -> Session {
        let cache = CompilationCache::new().with_artifact_exchange(
            Arc::new(PeerCompilationExchange),
            CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_secs(1) },
        );
        Session::builder()
            .client(client)
            .compilation_cache(cache)
            .remote_devices(remote_devices)
            .build()
            .unwrap()
    }

    /// Slice, PJRT memory kind, and row-major elements of an addressable shard of an array.
    #[cfg(feature = "cuda-13")]
    type ShardObservation = (Vec<Range<usize>>, String, Vec<f32>);

    /// Places `source` on `target` with [`Array::to_placement`] for the step `step` of a test that both ranks of a
    /// CUDA job run, waits for the result, and returns a [`ShardObservation`] of every shard of the result that this
    /// rank addresses. Asserts that the result is committed with the mesh and the sharding of `target` and that its
    /// type renders as `expected_type`, which both ranks must agree on whichever of them holds the shards of the
    /// result. A failed placement, readiness check, readback, or assertion panics with the rank and the step, which the
    /// report of [`run_cuda_rank_pair`] shows for both ranks.
    #[cfg(feature = "cuda-13")]
    fn place_step(
        rank: usize,
        step: &str,
        session: &Session,
        source: &Array,
        target: Placement,
        expected_type: &str,
    ) -> Vec<ShardObservation> {
        let placed = source
            .to_placement(session, target.clone())
            .and_then(|placed| placed.block_until_ready().map(|()| placed))
            .unwrap_or_else(|error| panic!("rank {rank} failed to place the source in the step `{step}`: {error}"));
        assert_eq!(placed.r#type().to_string(), expected_type, "rank {rank}, step `{step}`");
        assert_eq!(placed.mesh(), target.mesh(), "rank {rank}, step `{step}`");
        assert_eq!(placed.sharding(), target.sharding(), "rank {rank}, step `{step}`");
        assert!(placed.is_committed(), "rank {rank} placed an uncommitted result in the step `{step}`");
        placed
            .addressable_shards()
            .map(|shard| {
                let kind = shard.buffer().unwrap().memory().unwrap().kind().unwrap().into_owned();
                let bytes = shard.copy_to_host_bytes().unwrap_or_else(|error| {
                    panic!("rank {rank} failed to read a shard placed in the step `{step}`: {error}")
                });
                (shard.slice().to_vec(), kind, values_from_bytes::<f32>(&bytes))
            })
            .collect()
    }

    #[test]
    fn test_placement_options_new() {
        let target = PlacementRequest::from(Device::new(3, 0));
        let options: PlacementOptions = PlacementOptions::new(Some(target.clone()), None, Some(true), Some(false));
        assert_eq!(options.target(), Some(&target));
        assert_eq!(options.source(), None);
        assert_eq!(options.donate(), Some(&true));
        assert_eq!(options.may_alias(), Some(&false));
        assert_eq!(options.into_parts(), (Some(target), None, Some(true), Some(false)));

        // Unset options with another prefix type than the standard ones come from the constructor.
        let options = PlacementOptions::<Device>::new(None, None, None, None);
        assert_eq!(options.into_parts(), (None, None, None, None));
    }

    #[test]
    fn test_placement_options_with_target() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();

        // The target changes its type, and the other options keep theirs.
        let options = PlacementOptions::default().with_may_alias(Some(false)).with_target((Some(devices[1]), None));
        assert_eq!(options.clone().into_parts(), (Some((Some(devices[1]), None)), None, None, Some(Some(false))));
        let (placed, uploaded) = session.place((1f32, 2f32), options).unwrap();
        assert_eq!(placed.mesh().devices(), [devices[1]]);
        assert!(placed.is_committed());
        assert_eq!(uploaded.mesh().devices(), [devices[0]]);
        assert!(!uploaded.is_committed());
    }

    #[test]
    fn test_placement_options_with_source() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let array = session.place(2.5f32, PlacementOptions::to(devices[0])).unwrap();

        // The source changes its type, and it is validated against the type of the array leaf, whereas the devices that
        // it names are informative.
        let options = PlacementOptions::to(devices[1]).with_source(devices[0]);
        assert_eq!(options.clone().into_parts(), (Some(devices[1]), Some(devices[0]), None, None));
        let moved = session.place(array.clone(), options).unwrap();
        assert_eq!(moved.mesh().devices(), [devices[1]]);
        let kept = session.place(array.clone(), PlacementOptions::default().with_source(devices[1])).unwrap();
        assert_eq!(kept.mesh().devices(), [devices[0]]);
        let pinned = PlacementRequest::from(Memory::Host { pinned: true });
        let error = session.place(array.clone(), PlacementOptions::default().with_source(pinned)).unwrap_err();
        assert!(matches!(&error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
        assert_eq!(
            error.to_string(),
            format!("`place` source memory `Host[Pinned]` does not match the input type `{}`", array.r#type()),
        );
    }

    #[test]
    fn test_placement_options_with_donate() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[1]).unwrap();
        let array = session.place(2.5f32, PlacementOptions::default()).unwrap();
        let buffer = Arc::downgrade(array.shards()[0].buffer().unwrap());

        // The donation changes its type, and a donated leaf is consumed.
        let options = PlacementOptions::to(device).with_donate(true);
        assert_eq!(options.clone().into_parts(), (Some(device), None, Some(true), None));
        let placed = session.place(array, options).unwrap();
        assert_eq!(placed.mesh().devices(), [device]);
        assert_eq!(shard_values(&placed), vec![vec![2.5]]);
        assert!(buffer.upgrade().is_none());
        let options = PlacementOptions::default().with_donate((true, false));
        assert_eq!(options.into_parts(), (None, None, Some((true, false)), None));
    }

    #[test]
    fn test_placement_options_with_may_alias() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let array = session.place(2.5f32, PlacementOptions::default()).unwrap();

        let shares_buffer =
            |placed: &Array| Arc::ptr_eq(placed.shards()[0].buffer().unwrap(), array.shards()[0].buffer().unwrap());

        // A forced copy needs no target, and the copy shares no buffer with the input.
        let options = PlacementOptions::default().with_may_alias(Some(false));
        assert_eq!(options.clone().into_parts(), (None, None, None, Some(Some(false))));
        let copied = session.place(array.clone(), options).unwrap();
        assert!(!shares_buffer(&copied));
        let kept = session.place(array.clone(), PlacementOptions::default().with_may_alias(Some(true))).unwrap();
        assert!(shares_buffer(&kept));

        // Plain `bool` aliasing options are accepted as well, and they mean what their `Some` forms mean.
        let options = PlacementOptions::default().with_may_alias(false);
        assert_eq!(options.clone().into_parts(), (None, None, None, Some(false)));
        let copied = session.place(array.clone(), options).unwrap();
        assert!(!shares_buffer(&copied));
        let kept = session.place(array.clone(), PlacementOptions::default().with_may_alias(true)).unwrap();
        assert!(shares_buffer(&kept));

        // So are tuple prefixes of plain `bool` options.
        let options = PlacementOptions::default().with_may_alias((false, true));
        let (copied, kept) = session.place((array.clone(), array.clone()), options).unwrap();
        assert!(!shares_buffer(&copied));
        assert!(shares_buffer(&kept));

        // An optional leaf can leave its branch unspecified while the other branch requests a copy.
        let options = PlacementOptions::default().with_may_alias((None, Some(false)));
        let (kept, copied) = session.place((array.clone(), array.clone()), options).unwrap();
        assert!(shares_buffer(&kept));
        assert!(!shares_buffer(&copied));
    }

    #[test]
    fn test_placement_options_to() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        assert_eq!(PlacementOptions::to(devices[1]).into_parts(), (Some(devices[1]), None, None, None));

        // The target alone fixes the types of the other options, whether it is a raw device or a placement request.
        let scalar = session.place(2.5f32, PlacementOptions::to(devices[1])).unwrap();
        assert_eq!(scalar.mesh().devices(), [devices[1]]);
        assert!(scalar.is_committed());
        let moved = session.place(scalar, PlacementOptions::to(PlacementRequest::from(devices[0]))).unwrap();
        assert_eq!(moved.mesh().devices(), [devices[0]]);
        assert_eq!(moved.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![2.5]);
    }

    #[test]
    fn test_placement_options_default() {
        assert_eq!(PlacementOptions::default().into_parts(), (None, None, None, None));

        // The default has the standard prefix types, so a call site needs no annotation, and a builder function then
        // replaces the prefix type of the option that it sets.
        let options = PlacementOptions::default().with_target(Device::new(1, 0));
        assert_eq!(options.into_parts(), (Some(Device::new(1, 0)), None, None, None));

        // Both ordinary downstream spellings of the default compile without generic arguments and place a value.
        let session = Session::builder().client(execution_client()).build().unwrap();
        let placed = session.place(3f32, PlacementOptions::default()).unwrap();
        assert_eq!(placed.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![3.]);
        assert!(!placed.is_committed());
        let options: PlacementOptions = Default::default();
        let placed = session.place(3f32, options).unwrap();
        assert_eq!(placed.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![3.]);
    }

    #[test]
    fn test_placeable_place_parameter_rejects_options_that_no_leaf_accepts() {
        let session = Session::builder().client(execution_client()).build().unwrap();
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]),
            None,
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();
        let empty = PlacementRequest::InPlace { memory: None, layout: None };
        let donation_error = "array `$` cannot be donated while `may_alias` is `Some(true)`";
        let source_error = "`place` in-place source must name a memory or a layout";
        let target_error = "`place` in-place target must name a memory or a layout";
        let cases = [
            (None, None, true, Some(true), donation_error),
            (None, Some(empty.clone()), true, Some(true), donation_error),
            (None, Some(empty.clone()), false, None, source_error),
            (Some(empty.clone()), Some(empty.clone()), false, Some(false), source_error),
            (Some(empty.clone()), None, true, None, target_error),
        ];

        // The options that no leaf accepts are rejected with the same error whether a leaf is placed through
        // `Session::place` or directly through its primitive, for host values (which otherwise ignore their source,
        // donation, and aliasing options), runtime arrays, mixed inputs, and traced values (which otherwise ignore
        // their donation option).
        let place = |target: &Option<PlacementRequest>, source: &Option<PlacementRequest>, donate, may_alias| {
            PlacementOptions::new(target.clone(), source.clone(), Some(donate), Some(may_alias))
        };
        for (target, source, donate, may_alias, expected) in &cases {
            let (donate, may_alias) = (*donate, *may_alias);
            let options = || place(target, source, donate, may_alias);
            let primitive_errors = [
                1f32.place_parameter(&session, target.clone(), source.clone(), donate, may_alias).unwrap_err(),
                array
                    .clone()
                    .place_parameter(&session, target.clone(), source.clone(), donate, may_alias)
                    .unwrap_err(),
                PlaceInput::from(1f32)
                    .place_parameter(&session, target.clone(), source.clone(), donate, may_alias)
                    .unwrap_err(),
            ];
            let session_errors = [
                session.place(1f32, options()).unwrap_err(),
                session.place(array.clone(), options()).unwrap_err(),
                session.place(PlaceInput::from(1f32), options()).unwrap_err(),
            ];
            for error in primitive_errors.iter().chain(&session_errors) {
                assert_eq!(error.to_string(), *expected);
            }

            let traced_errors = RefCell::new(Vec::new());
            let _: Array = batch(
                |x| {
                    let mut errors = traced_errors.borrow_mut();
                    errors.push(
                        x.clone().place_parameter(&session, target.clone(), source.clone(), donate, may_alias).err(),
                    );
                    errors.push(session.place(x.clone(), options()).err());
                    Ok(x)
                },
                array.clone(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
            let traced_errors = traced_errors
                .into_inner()
                .into_iter()
                .map(|error| error.map(|error| error.to_string()))
                .collect::<Vec<_>>();
            assert_eq!(traced_errors, vec![Some(expected.to_string()); 2]);
        }
    }

    #[test]
    fn test_array_to_placement() {
        // A value on one device is split over two devices, one of which is the device that holds it.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![first_device(client)],
        )
        .unwrap();
        let source_sharding = Sharding::replicated(source_mesh.logical_mesh().clone(), 1);
        let source_values = [0.0f32, 1.0, 2.0, 3.0];
        let source_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(source_sharding).unwrap();
        let source_array = Array::from_host_buffer(
            &session,
            source_type,
            source_mesh,
            None,
            values_to_bytes::<f32>(source_values.as_slice()).as_slice(),
        )
        .unwrap();

        let target_mesh = domain_mesh(client, "x", 2);
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let moved_array = source_array
            .to_placement(&session, Placement::new(target_mesh.clone(), target_sharding).unwrap())
            .unwrap();

        assert_eq!(moved_array.addressable_shards().count(), 2);
        assert_eq!(
            moved_array.sharding().visualize().unwrap().render(false),
            indoc! {"
                    ┌─────┬─────┐
                    │  0  │  1  │
                    └─────┴─────┘
                "}
            .trim_end()
            .to_string()
        );
        assert_eq!(shard_values(&moved_array), vec![vec![0.0, 1.0], vec![2.0, 3.0]]);
    }

    #[test]
    fn test_array_to_placement_splits_unevenly() {
        // Exact local shapes require differently sized slices for the split and a concatenation for the gather.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let shard_values = |array: &Array| {
            array
                .shards()
                .iter()
                .map(|shard| values_from_bytes::<i32>(&shard.copy_to_host_bytes().unwrap()))
                .collect::<Vec<_>>()
        };
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Auto] {
            let mesh = two_device_mesh(client, axis_type);
            let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
            let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
            let source = Array::from_host_buffer(
                &session,
                ArrayType::new_static(DataType::I32, [5]),
                mesh.clone(),
                Some(replicated.clone()),
                values_to_bytes(&[0i32, 1, 2, 3, 4]),
            )
            .unwrap();
            let target = Placement::new(mesh.clone(), sharded.clone()).unwrap();
            let split = source.to_placement(&session, target.clone()).unwrap();
            let on_devices = source.clone().reshard_on_devices(&session, &mesh, &sharded, false).unwrap();
            let donated = source.into_placement(&session, target).unwrap();
            for placed in [&split, &donated, &on_devices] {
                assert_eq!(placed.sharding(), &sharded);
                assert_eq!(shard_values(placed), vec![vec![0, 1, 2], vec![3, 4]]);
            }
            let gathered = split.reshard_on_devices(&session, &mesh, &replicated, false).unwrap();
            assert_eq!(gathered.sharding(), &replicated);
            assert_eq!(shard_values(&gathered), vec![vec![0, 1, 2, 3, 4]; 2]);
        }
    }

    #[test]
    fn test_array_to_placement_gathers_disjoint_manual_slices_of_pending_sums() {
        // A `u8[2]` with a pending sum along `x` whose summands `[1, 2]` and `[3, 4]` are split along the manual axis
        // `m` is gathered onto replicas of each summand, which moves its contributions, whether or not the source
        // varies along `m`, because the devices along `m` hold disjoint slices, so no two values compete for one
        // slice.
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
        let split = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["m"])]).unwrap();
        let split = split.with_unreduced_axes(["x"]).unwrap();
        let summands = Sharding::replicated(logical_mesh, 1).with_unreduced_axes(["x"]).unwrap();
        let target = Placement::new(mesh.clone(), summands).unwrap();
        for sharding in [split.clone(), split.with_varying_manual_axes(["m"]).unwrap()] {
            let shards = devices.iter().zip(1u8..).map(|(device, value)| (device.id(), Arc::<[u8]>::from([value])));
            let r#type = ArrayType::new_static(DataType::U8, [2]);
            let source = Array::from_host_shards(&session, r#type, mesh.clone(), Some(sharding.clone()), shards);
            let source = source.unwrap();
            let consumed = source.clone().into_placement(&session, target.clone());
            for placed in [source.to_placement(&session, target.clone()), consumed] {
                let placed = placed.unwrap();
                let shards = placed.addressable_shards().map(|shard| shard.copy_to_host_bytes().unwrap());
                let expected = [vec![1u8, 2], vec![1, 2], vec![3, 4], vec![3, 4]];
                assert_eq!(shards.collect::<Vec<_>>(), expected, "{sharding:?}");
                assert_eq!(placed.copy_to_host_bytes().unwrap(), [4u8, 6], "{sharding:?}");
            }
        }
    }

    #[test]
    fn test_array_to_placement_splits_unevenly_with_and_without_artifact_sharing() {
        // A replicated `u8[5]` on a device of this process and a device of another process is split into shards of
        // three and two elements, which a compiled reshard cannot produce. Each process slices its own replica on its
        // own device and leads the compilation of that slice, so sharing compilation artifacts with the other process
        // changes nothing. Neither does a declared layout that the buffers already have, nor a memory other than device
        // memory, which each process stages its replica out of and moves its shard back into on its own devices, nor a
        // column-major `u8[5, 2]` split along its rows, whose shards are converted to the dense row-major layout on
        // their devices. The result keeps the memory and the layout of the source.
        let client = cpu_client(1);
        let local = first_device(client);
        let remote = Device::new(local.id() + 1, local.process_index() + 1);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![local, remote]).unwrap();
        let sharded = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let rows = vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()];
        let rows = Sharding::new(logical_mesh, rows).unwrap();
        let vector = ArrayType::new_static(DataType::U8, [5]);
        let default_layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let matrix = ArrayType::new_static(DataType::U8, [5, 2]).with_layout(column_major);
        let (vector_values, matrix_values) = (vec![1u8, 2, 3, 4, 5], (0u8..10).collect::<Vec<_>>());
        let cases = [
            (vector.clone(), &vector_values, &sharded, "device", vec![1u8, 2, 3]),
            (vector.clone().with_layout(default_layout), &vector_values, &sharded, "device", vec![1, 2, 3]),
            (vector.with_memory(Memory::Host { pinned: true }), &vector_values, &sharded, "pinned_host", vec![1, 2, 3]),
            (matrix, &matrix_values, &rows, "device", vec![0, 1, 2, 3, 4, 5]),
        ];
        let timeout = Duration::from_secs(1);
        for policy in [
            CompilationArtifactExchangePolicy::Disabled,
            CompilationArtifactExchangePolicy::PreferSharing { timeout },
            CompilationArtifactExchangePolicy::RequireSharing { timeout },
        ] {
            let cache = CompilationCache::new().with_artifact_exchange(Arc::new(PeerCompilationExchange), policy);
            let session =
                Session::builder().client(client).compilation_cache(cache).remote_devices([remote]).build().unwrap();
            for (r#type, values, sharding, kind, expected) in &cases {
                let source = Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), None, *values).unwrap();
                let target = Placement::new(mesh.clone(), (*sharding).clone()).unwrap();
                for placed in [
                    source.to_placement(&session, target.clone()),
                    session.place(source.clone(), PlacementOptions::to(target)),
                ] {
                    let placed = placed.unwrap();
                    let shard = placed.addressable_shards().next().unwrap();
                    assert_eq!(&shard.copy_to_host_bytes().unwrap(), expected, "{policy:?} {}", r#type);
                    assert_eq!(placed.r#type().memory(), r#type.memory(), "{policy:?} {}", r#type);
                    assert_eq!(placed.layout(), r#type.layout(), "{policy:?} {}", r#type);
                    assert_eq!(shard.buffer().unwrap().memory().unwrap().kind().unwrap(), *kind);
                }
            }
        }
    }

    #[test]
    fn test_array_to_placement_returns_array_of_session() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let meshed_session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let first_device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![mesh.devices()[0]],
        )
        .unwrap();
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]),
            first_device_mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();

        // The identity placement and the placement that copies whole shards both return arrays of the receiving
        // session, which has a default mesh that the session of the input lacks.
        let unchanged = Placement::new(
            first_device_mesh.clone(),
            Sharding::replicated(first_device_mesh.logical_mesh().clone(), 1),
        )
        .unwrap();
        let copied = Placement::new(mesh.clone(), replicated).unwrap();
        for target in [unchanged, copied] {
            let placed = array.to_placement(&meshed_session, target.clone()).unwrap();
            let donated = array.clone().into_placement(&meshed_session, target).unwrap();
            for placed in [placed, donated] {
                assert!(placed.session().is_same_session(&meshed_session));
                assert_eq!(placed.session().device_mesh().unwrap(), &mesh);
            }
        }
    }

    #[test]
    fn test_array_to_placement_transfers_arrays_of_another_client() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let other_client = other_cpu_client(2);
        let other_session = Session::builder().client(other_client).build().unwrap();
        let other_mesh = domain_mesh(other_client, "x", 2);
        let devices = mesh.devices().to_vec();
        let split = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let values = values_to_bytes(&[1f32, 2., 3., 4.]);
        let other_split =
            Sharding::new(other_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let source =
            Array::from_host_buffer(&other_session, r#type.clone(), other_mesh.clone(), Some(other_split), &values)
                .unwrap();
        let uncommitted = Array::from_host_buffer(&other_session, r#type.clone(), None, None, &values).unwrap();
        assert!(uncommitted.is_locally_assigned());
        assert_eq!((source.mesh(), source.sharding()), (&mesh, &split));

        // A device or a sharding of this client receives the value read back from the other client, committed, not
        // locally assigned, in the default memory of its devices, and as an array of this session and its client.
        // The devices of both clients have the same identifiers, so the sharding is also the placement of the source,
        // which still moves. Donation changes nothing, and an uncommitted array of the other client moves alike.
        let targets = [
            (PlacementRequest::from(devices[1]), vec![devices[1]], vec![vec![1f32, 2., 3., 4.]]),
            (
                PlacementRequest::from(Placement::new(mesh.clone(), split.clone()).unwrap()),
                devices.clone(),
                vec![vec![1f32, 2.], vec![3., 4.]],
            ),
        ];
        for (target, expected_devices, expected_values) in targets {
            for input in [&source, &uncommitted] {
                let donated = input.clone().into_placement(&session, target.clone());
                for placed in [input.to_placement(&session, target.clone()), donated] {
                    let placed = placed.unwrap();
                    assert!(placed.session().is_same_session(&session));
                    assert!(std::ptr::eq(placed.client(), client));
                    assert_eq!(placed.mesh().devices(), expected_devices.as_slice());
                    assert_eq!((placed.is_committed(), placed.is_locally_assigned()), (true, false));
                    assert_eq!(placed.r#type().memory(), Memory::Device);
                    assert_eq!(shard_values(&placed), expected_values);
                }
            }
        }

        // `Session::place` admits and transfers the same arrays, while an in-place target and no target keep
        // rejecting them, because a result of this session cannot live on the devices of the other client.
        let placed = session.place(source.clone(), PlacementOptions::to(devices[0])).unwrap();
        assert_eq!((placed.mesh().devices(), placed.is_committed()), (&devices[..1], true));
        assert_eq!(shard_values(&placed), vec![vec![1f32, 2., 3., 4.]]);
        let in_place = PlacementRequest::InPlace { memory: Some(Memory::Device), layout: None };
        let in_place_message = "cannot place an array of another PJRT client without naming the devices of the \
                                target, because the result must live on devices of the client of the session";
        for result in [
            source.to_placement(&session, in_place.clone()),
            session.place(source.clone(), PlacementOptions::to(in_place.clone())),
        ] {
            assert!(matches!(
                result,
                Err(Error::Pjrt(PjrtError::InvalidArgument { message, .. })) if message == in_place_message,
            ));
        }
        assert!(matches!(
            session.place(source.clone(), PlacementOptions::default()),
            Err(Error::Pjrt(PjrtError::InvalidArgument { message, .. }))
                if message == "cannot associate an array with a session of a different PJRT client",
        ));

        // A pending sum is completed by the readback, so a target that keeps it is rejected before anything is
        // read back, both by `Session::place` and by the placement itself.
        let pending = Sharding::replicated(other_mesh.logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let contributions = other_mesh
            .devices()
            .iter()
            .zip([[1f32, 2., 3., 4.], [10., 20., 30., 40.]])
            .map(|(device, values)| (device.id(), Arc::<[u8]>::from(values_to_bytes(&values))))
            .collect::<Vec<_>>();
        let summands =
            Array::from_host_shards(&other_session, r#type.clone(), other_mesh.clone(), Some(pending), contributions)
                .unwrap();
        let replicated = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap();
        let completed = summands.to_placement(&session, replicated.clone()).unwrap();
        assert_eq!(shard_values(&completed), vec![vec![11f32, 22., 33., 44.]; 2]);
        let kept = replicated
            .clone()
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let kept_message = "cannot place an array of another PJRT client onto a target that keeps the \
                            contributions of its pending reduction or manual variation apart, because the transfer \
                            between the clients reads back their combined value";
        for result in [
            summands.to_placement(&session, kept.clone()),
            session.place(summands.clone(), PlacementOptions::to(PlacementRequest::from(kept.clone()))),
        ] {
            assert!(matches!(
                result,
                Err(Error::Pjrt(PjrtError::InvalidArgument { message, .. })) if message == kept_message,
            ));
        }

        // The upload waits for the work that produces the source on the other client and reports its failure through
        // the readiness and the readback of the result.
        let (failing, managers) = pending_replicated_vector(&other_session, &other_mesh);
        let placed = failing.to_placement(&session, devices[0]).unwrap();
        managers[0].set_error(0, PjrtError::aborted("first replica failed")).unwrap();
        managers[1].set_error(0, PjrtError::aborted("second replica failed")).unwrap();
        let expected = format!(
            "the transfer of an array from another PJRT client failed with: {}",
            Error::Pjrt(PjrtError::aborted("first replica failed")),
        );
        for result in [placed.block_until_ready(), placed.copy_to_host_bytes().map(drop)] {
            assert!(matches!(
                result,
                Err(Error::Pjrt(PjrtError::Internal { message, .. })) if message == expected,
            ));
        }
    }

    #[test]
    fn test_array_to_placement_rebinding_checks_session_devices() {
        let client = cpu_client(1);
        let remote = first_device(process_one_cpu_client(1));
        let session = session_with_remote_devices(client, [remote]);
        let known_alias = session_with_remote_devices(client, [remote]);
        let unknown_alias = session_with_remote_devices(client, []);
        let source = session.place(7u8, PlacementOptions::to(remote)).unwrap();
        let target = Placement::new(source.mesh().clone(), source.sharding().clone()).unwrap();
        let retained = RetainedArray::new(&source);
        let place = |session: &Session| {
            [
                source.to_placement(session, target.clone()),
                source.clone().into_placement(session, target.clone()),
                source.to_placement(session, Memory::Device),
                session.place(source.clone(), PlacementOptions::to(target.clone())),
                session.place(source.clone(), PlacementOptions::default()),
            ]
        };
        let expected = ArrayError::UnknownDevice { device_id: remote.id(), process_index: remote.process_index() };
        for result in place(&unknown_alias) {
            let error = result.unwrap_err();
            assert!(matches!(
                error,
                Error::Array(error) if matches!(
                    *error,
                    ArrayError::UnknownDevice { device_id, process_index }
                        if device_id == remote.id() && process_index == remote.process_index(),
                ),
            ));
        }
        // The error keeps its cause through the shared conversion, which compares by the identity of that cause.
        let error = retained.to_array(&unknown_alias).unwrap_err();
        assert!(matches!(error, ProgramError::Custom(_)));
        assert_eq!(error.to_string(), expected.to_string());
        for placed in place(&known_alias)
            .map(Result::unwrap)
            .into_iter()
            .chain([retained.to_array(&known_alias).unwrap()])
        {
            assert!(placed.session().is_same_session(&known_alias));
            assert_eq!(placed.mesh(), target.mesh());
            assert_eq!(placed.addressable_shards().count(), 0);
            placed.block_until_ready().unwrap();
        }
    }

    #[test]
    fn test_array_to_placement_transfers_size_one_pending_reductions_between_clients() {
        let client = cpu_client(1);
        let other_client = other_cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let other_session = Session::builder().client(other_client).build().unwrap();
        let source = domain_mesh(other_client, "x", 1);
        let target_mesh = domain_mesh(client, "x", 1);
        let sharding = Sharding::replicated(source.logical_mesh().clone(), 0).with_unreduced_axes(["x"]).unwrap();
        let target = Placement::new(target_mesh, sharding.clone()).unwrap();
        for (data_type, bytes) in [(DataType::U8, vec![7]), (DataType::Zero, vec![])] {
            let value = Array::from_host_buffer(
                &other_session,
                ArrayType::scalar(data_type),
                source.clone(),
                Some(sharding.clone()),
                bytes.as_slice(),
            )
            .unwrap();
            let same_client = Placement::new(source.clone(), sharding.clone()).unwrap();
            assert_eq!(value.to_placement(&other_session, same_client).unwrap().copy_to_host_bytes().unwrap(), bytes);
            for placed in place_through_entry_points(&session, &value, target.clone()).map(Result::unwrap) {
                assert_eq!(placed.sharding(), &sharding);
                assert_eq!(placed.copy_to_host_bytes().unwrap(), bytes);
            }
        }
    }

    #[test]
    fn test_array_to_placement_transfers_zero_between_clients_on_shared_mesh() {
        let sources = [other_cpu_client(1), process_one_cpu_client(1)];
        let targets = [cpu_client(1), process_one_cpu_client(1)];
        let source_sessions = sessions_of_processes(sources);
        let target_sessions = sessions_of_processes(targets);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let source_mesh = DeviceMesh::new(logical_mesh.clone(), sources.map(first_device).to_vec()).unwrap();
        let target_mesh = DeviceMesh::new(logical_mesh.clone(), targets.map(first_device).to_vec()).unwrap();
        let sharding = Sharding::replicated(logical_mesh, 0).with_unreduced_axes(["x"]).unwrap();
        let target = Placement::new(target_mesh, sharding.clone()).unwrap();
        for (source, session) in source_sessions.iter().zip(&target_sessions) {
            let (event, producer) = source.client().event(()).unwrap();
            let value = Array::from_host_buffer(
                source,
                ArrayType::scalar(DataType::Zero),
                source_mesh.clone(),
                Some(sharding.clone()),
                [],
            )
            .unwrap()
            .with_execution_fence(ExecutionFence::new(vec![event]));
            assert!(!value.is_locally_assigned());
            producer.set(Some(PjrtError::aborted("zero source failed"))).unwrap();
            for placed in place_through_entry_points(session, &value, target.clone()).map(Result::unwrap) {
                assert_eq!(placed.sharding(), &sharding);
                assert_eq!(placed.mesh(), target.mesh());
                assert_eq!(placed.addressable_shards().count(), 1);
                assert!(placed.addressable_shards().all(|shard| shard.buffer().is_none()));
                assert!(matches!(
                    placed.block_until_ready(),
                    Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "zero source failed",
                ));
            }

            let empty_sharding =
                Sharding::replicated(source_mesh.logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap();
            let empty = Array::from_host_buffer(
                source,
                ArrayType::new_static(DataType::U8, [0]),
                source_mesh.clone(),
                Some(empty_sharding.clone()),
                [],
            )
            .unwrap();
            let empty_target = target.clone().with_sharding(empty_sharding.clone()).unwrap();
            let placed = session.place(empty, PlacementOptions::to(empty_target)).unwrap();
            assert_eq!(placed.sharding(), &empty_sharding);
            assert!(placed.copy_to_host_bytes().unwrap().is_empty());
        }
    }

    #[test]
    fn test_array_to_placement_transfers_local_shards_between_clients_with_different_ranks() {
        let clients = [cpu_client(1), process_one_cpu_client(1)];
        for (source_client, client) in [(clients[0], clients[1]), (clients[1], clients[0])] {
            let source = Session::builder().client(source_client).build().unwrap();
            let session = Session::builder().client(client).build().unwrap();
            let value = Array::from_host_buffer(
                &source,
                ArrayType::new_static(DataType::U8, [1]),
                domain_mesh(source_client, "x", 1),
                None,
                [17u8],
            )
            .unwrap();
            assert!(!value.is_locally_assigned());
            let device = first_device(client);
            for placed in place_through_entry_points(&session, &value, device).map(Result::unwrap) {
                assert_eq!(placed.mesh().devices(), &[device]);
                assert_eq!(placed.copy_to_host_bytes().unwrap(), [17u8]);
            }
        }
    }

    #[test]
    fn test_array_to_placement_from_another_client_agrees_across_processes() {
        // Each of two processes sees a source client and a target client with one device per process, and places
        // `u8[2] = [7, 11]` onto the target device of process 1, which reads the value back. A source that only
        // process 0 holds is the same error on both processes, before anything is read back, also on process 0, which
        // owns no target device. A source that process 1 also holds is placed by process 1, and process 0 returns
        // the same metadata without buffers. Each entry point agrees.
        let source_clients = [cpu_client(1), process_one_cpu_client(1)];
        let target_clients = [other_cpu_client(1), process_one_cpu_client(1)];
        let source_devices = source_clients.map(first_device);
        let target_device = first_device(target_clients[1]);
        let target = Placement::new(
            DeviceMesh::new(LogicalMesh::single_device(), vec![target_device]).unwrap(),
            Sharding::replicated(LogicalMesh::single_device(), 1),
        )
        .unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let process_0_mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![source_devices[0]]).unwrap();
        let both_processes_mesh = DeviceMesh::new(logical_mesh, source_devices.to_vec()).unwrap();
        let r#type = ArrayType::new_static(DataType::U8, [2]);
        let expected_message = "cannot place an array of another PJRT client onto the devices of process 1, which \
                                holds no replica of shard #0 of the array, because the transfer between the clients \
                                reads back the value of the array on that process";
        let mut outcomes = Vec::new();
        for process in 0..2 {
            let remote = [source_devices[1 - process]];
            let source_session = session_with_remote_devices(source_clients[process], remote);
            let session = session_with_remote_devices(target_clients[process], remote);
            let place = |mesh: &DeviceMesh| {
                let source = Array::from_host_buffer(&source_session, r#type.clone(), mesh.clone(), None, [7u8, 11]);
                place_through_entry_points(&session, &source.unwrap(), target.clone())
            };
            for result in place(&process_0_mesh) {
                assert!(matches!(
                    result,
                    Err(Error::Pjrt(PjrtError::InvalidArgument { message, .. })) if message == expected_message,
                ));
            }
            for result in place(&both_processes_mesh) {
                let placed = result.unwrap();
                placed.block_until_ready().unwrap();
                let bytes = placed.addressable_shards().map(|shard| shard.copy_to_host_bytes().unwrap());
                let bytes = bytes.collect::<Vec<_>>();
                let metadata = (placed.mesh().clone(), placed.sharding().clone(), placed.r#type().into_owned());
                outcomes.push((metadata, (placed.is_committed(), placed.is_locally_assigned()), bytes));
            }
        }
        let metadata = outcomes[0].0.clone();
        assert_eq!(metadata.0, target.mesh().clone());
        let expected = |bytes: Vec<Vec<u8>>| (metadata.clone(), (true, false), bytes);
        assert_eq!(outcomes, [vec![expected(vec![]); 3], vec![expected(vec![vec![7, 11]]); 3]].concat());
    }

    #[test]
    fn test_array_to_placement_from_another_client_keeps_memory_of_source() {
        // A target that omits its memory keeps the pinned host memory of a source of another client, as within
        // one client, and a request for a device selects its default memory.
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let other_session = Session::builder().client(other_cpu_client(1)).build().unwrap();
        let device = first_device(client);
        let pinned = Memory::Host { pinned: true };
        let r#type = ArrayType::new_static(DataType::U8, [1]).with_memory(pinned);
        let source = Array::from_host_buffer(&other_session, r#type, None, None, [13u8]).unwrap();
        let target = Placement::new(
            DeviceMesh::new(LogicalMesh::single_device(), vec![device]).unwrap(),
            Sharding::replicated(LogicalMesh::single_device(), 1),
        )
        .unwrap();
        let devices = client.addressable_devices().unwrap();
        let device_kind = devices[0].default_memory().unwrap().kind().unwrap().into_owned();
        let rows = [
            (PlacementRequest::from(target), pinned, "pinned_host".to_string()),
            (PlacementRequest::from(device), Memory::Device, device_kind),
        ];
        for (request, memory, kind) in rows {
            for placed in place_through_entry_points(&session, &source, request) {
                let placed = placed.unwrap();
                assert_eq!(placed.r#type().memory(), memory);
                assert_eq!(shard_memory_kinds(&placed), vec![kind.clone()]);
                assert_eq!(placed.copy_to_host_bytes().unwrap(), vec![13u8]);
            }
        }
    }

    #[test]
    fn test_array_to_placement_reports_failed_sources_without_buffers_across_clients() {
        // A source of another client is replicated over a device of process 0 and a device of process 1, and its
        // replica on process 0 is the buffer of a transfer manager. Process 0 owns no device of the target on
        // process 1, so it reads nothing back, and its result without buffers reports the failure of that replica
        // through each entry point.
        let other_client = other_cpu_client(1);
        let remote_device = first_device(process_one_cpu_client(1));
        let session = session_with_remote_devices(cpu_client(1), [remote_device]);
        let other_session = session_with_remote_devices(other_client, [remote_device]);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let source_devices = vec![first_device(other_client), remote_device];
        let source_mesh = DeviceMesh::new(logical_mesh.clone(), source_devices).unwrap();
        let other_devices = other_client.addressable_devices().unwrap();
        let memory = other_devices[0].default_memory().unwrap();
        let specification = BufferSpecification::new(BufferType::F32, [4u64]);
        let manager = other_client.host_to_device_transfer_manager(vec![specification], memory).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let replicated = Some(Sharding::replicated(logical_mesh, 1));
        let buffers = vec![manager.retrieve_buffer(0).unwrap()];
        let source = Array::from_addressable_buffers(&other_session, r#type, source_mesh, replicated, buffers).unwrap();
        let placed = place_through_entry_points(&session, &source, remote_device).map(Result::unwrap);
        manager.set_error(0, PjrtError::aborted("replica failed")).unwrap();
        for placed in placed {
            assert_eq!((placed.mesh().devices(), placed.addressable_shards().count()), ([remote_device].as_slice(), 0));
            assert!(matches!(
                placed.block_until_ready(),
                Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "replica failed",
            ));
        }
    }

    #[test]
    fn test_array_to_placement_from_another_client_returns_bufferless_results_without_waiting() {
        // A `DataType::Zero` array has no bytes to read back or upload, so placing it onto another client returns
        // before the work that produces it completes, without reading it back on the calling thread, and the result
        // reports that work, through each entry point.
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let other_client = other_cpu_client(1);
        let other_session = Session::builder().client(other_client).build().unwrap();
        let device = first_device(client);
        let (event, producer) = other_client.event(()).unwrap();
        let r#type = ArrayType::new_static(DataType::Zero, [6]);
        let source = Array::from_host_buffer(&other_session, r#type, None, None, [0u8; 0])
            .unwrap()
            .with_execution_fence(ExecutionFence::new(vec![event]));
        let (sender, receiver) = mpsc::channel();
        let worker = thread::spawn(move || {
            sender.send(place_through_entry_points(&session, &source, device)).unwrap();
        });
        let placed = receiver.recv_timeout(Duration::from_secs(5));
        producer.set(Some(PjrtError::aborted("producer failed"))).unwrap();
        worker.join().unwrap();
        for placed in placed.unwrap() {
            let placed = placed.unwrap();
            assert_eq!(placed.data_type(), DataType::Zero);
            assert_eq!((placed.mesh().devices(), placed.addressable_shards().count()), ([device].as_slice(), 1));
            assert!(matches!(
                placed.block_until_ready(),
                Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "producer failed",
            ));
        }
    }

    #[test]
    fn test_array_to_placement_ignores_earlier_sends_onto_the_same_devices() {
        // The session frontend and core dispatch each place an array that issued a send which fails later. Each
        // places without a target and onto the source device, with and without a forced copy. The new
        // placements send nothing, so only the source reports the failure.
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = first_device(client);
        let (send, finish) = client.event(()).unwrap();
        let mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![device]).unwrap();
        let sent = Array::from_host_buffer(&session, ArrayType::new_static(DataType::U8, [1]), mesh, None, [17u8])
            .unwrap()
            .with_send_fence(ExecutionFence::new(vec![send]));
        let placed = [
            PlacementOptions::default(),
            PlacementOptions::default().with_may_alias(false),
            PlacementOptions::to(PlacementRequest::from(device)),
            PlacementOptions::to(PlacementRequest::from(device)).with_may_alias(false),
        ]
        .map(|options| session.place(sent.clone(), options).unwrap());
        let target_request = PlacementRequest::from(device);
        let core = [(None, true), (None, false), (Some(&target_request), true), (Some(&target_request), false)].map(
            |(target, may_alias)| {
                <ValueDomainDispatch as PlaceDispatch<ArrayType, Array>>::place(&sent, target, None, may_alias).unwrap()
            },
        );
        finish.set(Some(PjrtError::aborted("send failed"))).unwrap();
        assert!(matches!(
            sent.block_until_ready(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "send failed",
        ));
        for placed in placed.into_iter().chain(core) {
            assert_eq!(placed.copy_to_host_bytes().unwrap(), vec![17u8]);
            assert!(matches!(placed.block_until_ready(), Ok(())));
        }
    }

    #[test]
    fn test_array_to_placement_keeps_local_assignment_only_in_place() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let source = Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(&[1f32, 2.])).unwrap();
        assert!(source.is_locally_assigned());
        assert!(!source.is_committed());
        let own_mesh = source.mesh().clone();
        let pinned = Memory::Host { pinned: true };

        // A target that names only a memory or a layout commits the result, as in JAX, but leaves it on the
        // devices that this process chose on its own.
        for placed in [
            source.to_placement(&session, pinned).unwrap(),
            source.clone().into_placement(&session, pinned).unwrap(),
            source
                .to_placement(
                    &session,
                    PlacementRequest::InPlace {
                        memory: None,
                        layout: Some(Layout::Tiled(TiledLayout::new(vec![0], vec![]))),
                    },
                )
                .unwrap(),
        ] {
            assert_eq!(placed.mesh(), &own_mesh);
            assert!(placed.is_committed());
            assert!(placed.is_locally_assigned());
        }

        // A target that names devices or a sharding names the devices of the result, even when they are the
        // devices of the source, and commits it, while the source stays uncommitted.
        let devices = session.addressable_devices().unwrap();
        let same_devices = Placement::new(own_mesh.clone(), source.sharding().clone()).unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let other_devices = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap();
        for (target, expected_devices) in [
            (PlacementRequest::from(same_devices), own_mesh.devices()),
            (PlacementRequest::from(other_devices), &devices[..]),
            (PlacementRequest::from(devices[0]), &devices[..1]),
            (PlacementRequest::from(devices[1]), &devices[1..]),
        ] {
            for placed in
                [source.to_placement(&session, target.clone()), source.clone().into_placement(&session, target)]
            {
                let placed = placed.unwrap();
                assert_eq!(placed.mesh().devices(), expected_devices);
                assert!(placed.is_committed());
                assert!(!placed.is_locally_assigned());
            }
        }
        assert!(!source.is_committed());
        session.synchronize().unwrap();
    }

    #[test]
    fn test_array_to_placement_copies_matching_local_shards_without_full_source_addressability() {
        let client = cpu_client(1);
        let local_device = client.addressable_devices().unwrap().remove(0);
        let local_device_id = local_device.id().unwrap();
        let remote_device_id = local_device_id + 1;
        let session = session_with_remote_devices(client, [Device::new(remote_device_id, 1)]);
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(local_device_id, local_device.process_index().unwrap()), Device::new(remote_device_id, 1)],
        )
        .unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let local_source_buffer = client
            .buffer(values_to_bytes::<f32>(&[0.0, 1.0]).as_slice(), BufferType::F32, [2u64], None, local_device, None)
            .unwrap();
        let source_array = Array::from_addressable_buffers(
            &session,
            ArrayType::new_static(DataType::F32, [4]),
            mesh.clone(),
            Some(sharding.clone()),
            vec![local_source_buffer],
        )
        .unwrap();

        let copied_array =
            source_array.to_placement(&session, Placement::new(mesh.clone(), sharding).unwrap()).unwrap();
        let expected_visualization =
            format!("┌─────┬─────┐\n│{:^5}│{:^5}│\n└─────┴─────┘", local_device_id, remote_device_id);

        assert_eq!(copied_array.addressable_shards().count(), 1);
        assert_eq!(copied_array.sharding().visualize().unwrap().render(false), expected_visualization);
        let copied_shard_bytes = copied_array.device_shard(local_device_id).unwrap().copy_to_host_bytes().unwrap();
        assert_eq!(values_from_bytes::<f32>(copied_shard_bytes.as_slice()), vec![0.0, 1.0]);
        assert!(copied_array.device_shard(remote_device_id).unwrap().buffer().is_none());
    }

    #[test]
    fn test_array_to_placement_copies_computed_pinned_host_replicas() {
        let clients = [cpu_client(2), process_one_cpu_client(2)];
        let devices: [Device; 2] =
            std::array::from_fn(|rank| Device::from_pjrt(&clients[rank].addressable_devices().unwrap()[rank]).unwrap());
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices.to_vec()).unwrap();
        let memory = Memory::Host { pinned: true };
        let target = Placement::new(mesh.clone(), Sharding::replicated(logical_mesh, 2)).unwrap().with_memory(memory);

        for (rank, client) in clients.into_iter().enumerate() {
            let session = session_with_remote_devices(client, [devices[1 - rank]]);
            let r#type = ArrayType::new_static(DataType::I8, [2, 4]).with_memory(memory);
            let source = Array::from_host_buffer(&session, r#type, None, None, [0u8, 1, 2, 3, 4, 5, 6, 7])
                .unwrap()
                .neg()
                .unwrap();
            assert!(source.is_locally_assigned());
            assert_eq!(shard_memory_kinds(&source), ["pinned_host"]);

            // Rank 0 can reuse its replica. Rank 1 must copy the computed buffer to its second device.
            let placed = source.to_placement(&session, target.clone()).unwrap();
            let placed_by_session = session.place(source.clone(), PlacementOptions::to(target.clone())).unwrap();
            for output in [placed, placed_by_session] {
                output.block_until_ready().unwrap();
                assert_eq!(output.mesh(), &mesh);
                assert_eq!(shard_memory_kinds(&output), ["pinned_host"]);
                let shard = output.addressable_shards().next().unwrap();
                assert_eq!(shard.device(), devices[rank]);
                let buffer = shard.buffer().unwrap();
                assert_eq!(Arc::ptr_eq(source.shards()[0].buffer().unwrap(), buffer), rank == 0);
                assert_eq!(
                    values_from_bytes::<i8>(&shard.copy_to_host_bytes().unwrap()),
                    [0, -1, -2, -3, -4, -5, -6, -7]
                );
            }
            session.synchronize().unwrap();
        }
    }

    #[test]
    fn test_array_to_placement_copies_available_local_replica() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let devices = session.addressable_devices().unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let buffer = client.buffer([7u8].as_slice(), BufferType::U8, [1u64], None, client_devices[0].clone(), None);
        let source = Array::from_addressable_buffers(
            &session,
            ArrayType::new_static(DataType::U8, [1]),
            mesh.clone(),
            Some(replicated),
            vec![buffer.unwrap()],
        )
        .unwrap();
        assert!(source.device_shard(devices[1].id()).unwrap().buffer().is_none());
        assert_eq!(source.copy_to_host_bytes().unwrap(), [7]);

        // The plan names the replica on the target device, which this array lacks. Only this process takes part
        // in the move, so the replica on the other device of this process serves it instead.
        let source_buffer = Arc::clone(source.device_shard(devices[0].id()).unwrap().buffer().unwrap());
        for placed in place_through_entry_points(&session, &source, devices[1]) {
            let placed = placed.unwrap();
            assert_eq!(placed.mesh().devices(), [devices[1]]);
            let buffer = placed.shards()[0].buffer().unwrap();
            assert!(!Arc::ptr_eq(buffer, &source_buffer));
            assert_eq!(buffer.device().unwrap().id().unwrap(), devices[1].id());
            assert_eq!(buffer.dimensions().unwrap(), [1]);
            assert_eq!(placed.block_until_ready().map_err(|error| error.to_string()), Ok(()));
            assert_eq!(placed.copy_to_host_bytes().unwrap(), [7]);
        }

        // A compiled reshard to another mesh moves the replicas of a replicated source as whole shards, and no program
        // reads the source itself, so a missing replica is served by the other one here too, rather than ruling out
        // the device work. The declared layout rules out shard regions, and the reversed mesh names the replica on
        // the first device for its second shard, which this array lacks in the second case.
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let logical_mesh = mesh.logical_mesh().clone();
        let reversed = DeviceMesh::new(logical_mesh.clone(), vec![devices[1], devices[0]]).unwrap();
        let rows = Sharding::new(
            logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        for held in [vec![0, 1], vec![1]] {
            let session = Session::builder().client(client).build().unwrap();
            let buffers = held
                .iter()
                .map(|&index: &usize| {
                    let memory = client_devices[index].default_memory().unwrap();
                    let values = values_to_bytes(&[1f32, 2., 3., 4.]);
                    let layout = Some(column_major.to_pjrt().unwrap());
                    client.buffer(values.as_slice(), BufferType::F32, [2u64, 2], None, memory, layout).unwrap()
                })
                .collect();
            let r#type = ArrayType::new_static(DataType::F32, [2, 2]).with_layout(column_major.clone());
            let replicated = Some(Sharding::replicated(logical_mesh.clone(), 2));
            let source = Array::from_addressable_buffers(&session, r#type, mesh.clone(), replicated, buffers).unwrap();
            let placed = source.to_placement(&session, Placement::new(reversed.clone(), rows.clone()).unwrap());
            let placed = placed.unwrap();
            assert_eq!(placed.block_until_ready().map_err(|error| error.to_string()), Ok(()));
            assert_eq!(placed.layout(), Some(&column_major));
            assert_eq!(shard_values(&placed), [vec![1., 2.], vec![3., 4.]]);
            // The programs include one storage conversion for each device that prepares an argument in another layout
            // than the one it is held in, which is cached for that device and not repeated by later placements.
            assert_eq!(session.cached_program_count(), 5, "the device work must place a source with replicas {held:?}");
        }

        // A missing shard whose slice has no other replica on this process is still an error.
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let buffer = client.buffer([7u8].as_slice(), BufferType::U8, [1u64], None, client_devices[0].clone(), None);
        let source = Array::from_addressable_buffers(
            &session,
            ArrayType::new_static(DataType::U8, [2]),
            mesh.clone(),
            Some(sharded.clone()),
            vec![buffer.unwrap()],
        )
        .unwrap();
        let reversed = DeviceMesh::new(mesh.logical_mesh().clone(), vec![devices[1], devices[0]]).unwrap();
        let error = source.to_placement(&session, Placement::new(reversed, sharded).unwrap()).unwrap_err();
        assert_eq!(
            error.to_string(),
            format!("array shard #1 on device {} has no addressable buffer", devices[1].id()),
        );
    }

    #[test]
    fn test_array_to_placement_moves_pending_contributions_onto_a_transposed_mesh() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let source_mesh = DeviceMesh::new(logical_mesh_2x2(), devices.clone()).unwrap();
        let transposed = LogicalMesh::new(vec![
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let target_mesh = DeviceMesh::new(transposed, vec![devices[0], devices[2], devices[1], devices[3]]).unwrap();
        let pending = |mesh: &DeviceMesh, axes: &[&str]| {
            let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
            sharding.with_unreduced_axes(axes.iter().map(|axis| axis.to_string())).unwrap()
        };
        let import = |data_type: DataType, mesh: &DeviceMesh, axes: &[&str], contributions: [Vec<u8>; 4]| {
            Array::from_host_shards(
                &session,
                ArrayType::scalar(data_type),
                mesh.clone(),
                Some(pending(mesh, axes)),
                devices.iter().zip(contributions).map(|(device, bytes)| (device.id(), Arc::<[u8]>::from(bytes))),
            )
            .unwrap()
        };
        // Returns the bytes that each device holds, in the order of `devices`.
        let device_bytes = |array: &Array| {
            devices
                .iter()
                .map(|device| array.device_shard(device.id()).unwrap().copy_to_host_bytes().unwrap())
                .collect::<Vec<_>>()
        };

        // Every device keeps its coordinate along `x`, whatever the position of `x` in either mesh, so the placement
        // only moves contributions and keeps the sum of two smallest subnormal `f32` values.
        let contribution = vec![1u8, 0, 0, 0];
        let source = import(DataType::F32, &source_mesh, &["x"], std::array::from_fn(|_| contribution.clone()));
        let target = Placement::new(target_mesh.clone(), pending(&target_mesh, &["x"])).unwrap();
        let placed = source.to_placement(&session, target).unwrap();
        for shard in placed.shards() {
            assert_eq!(shard.copy_to_host_bytes().unwrap(), contribution);
        }
        assert_eq!(placed.copy_to_host_bytes().unwrap(), vec![2u8, 0, 0, 0]);

        // With pending sums along both axes, the contributions correspond by their ordinals (i.e., the k-th unreduced
        // axis of the source in mesh order corresponds to the k-th one of the target when their sizes are equal),
        // so the contributions at `(x, y) = (0, 1)` and `(1, 0)` swap devices and the readback combines every
        // contribution in the same order, which keeps the sum of `[MAX, -MAX, MAX, -MAX]` at zero and the sum of
        // `[2^24, 1, -2^24, 1]` at one.
        let both = ["x", "y"];
        for (contributions, sum) in
            [([f32::MAX, -f32::MAX, f32::MAX, -f32::MAX], 0f32), ([16_777_216f32, 1., -16_777_216., 1.], 1.)]
        {
            let bytes = contributions.map(|value| values_to_bytes(&[value]));
            let source = import(DataType::F32, &source_mesh, &both, bytes);
            let before = source.copy_to_host_bytes().unwrap();
            let target = Placement::new(target_mesh.clone(), pending(&target_mesh, &["y", "x"])).unwrap();
            let placed = source.to_placement(&session, target).unwrap();
            let moved = [contributions[0], contributions[2], contributions[1], contributions[3]];
            assert_eq!(device_bytes(&placed), moved.map(|value| values_to_bytes(&[value])));
            assert_eq!(values_from_bytes::<f32>(&placed.copy_to_host_bytes().unwrap()), [sum]);
            assert_eq!(placed.copy_to_host_bytes().unwrap(), before);
        }

        // Only `y` distinguishes the contributions, so the mesh `(y, z)` takes them along its first axis, and every
        // other axis holds replicas whatever its name.
        let yz = LogicalMesh::new(vec![
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let yz = DeviceMesh::new(yz, vec![devices[0], devices[2], devices[1], devices[3]]).unwrap();
        let source = import(DataType::U8, &source_mesh, &["y"], [vec![1], vec![2], vec![1], vec![2]]);
        let placed = source.to_placement(&session, Placement::new(yz.clone(), pending(&yz, &["y"])).unwrap()).unwrap();
        assert_eq!(device_bytes(&placed), [vec![1], vec![2], vec![1], vec![2]]);
        assert_eq!(placed.copy_to_host_bytes().unwrap(), vec![3]);

        // Six contributions on `(x=2, y=3)` correspond by their ordinals to those on `(y=3, x=2)`, so the contribution
        // of each device moves to the device with the same ordinal, and the readback combines them in the same order,
        // as JAX keeps the flattened order of the shards, which keeps the sum of `[MAX, -MAX, 0, MAX, -MAX, 0]` at zero
        // and that of `[2^24, 1, 0, -2^24, 1, 0]` at one.
        let session = Session::builder().client(cpu_client(6)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let logical_mesh = |axes: [(&str, usize); 2]| {
            let axes = axes.iter().map(|(name, size)| MeshAxis::new(*name, *size, MeshAxisType::Explicit).unwrap());
            LogicalMesh::new(axes.collect()).unwrap()
        };
        let source_mesh = DeviceMesh::new(logical_mesh([("x", 2), ("y", 3)]), devices.clone()).unwrap();
        let target_devices = [0, 3, 1, 4, 2, 5].map(|index| devices[index]).to_vec();
        let target_mesh = DeviceMesh::new(logical_mesh([("y", 3), ("x", 2)]), target_devices).unwrap();
        let target = Placement::new(target_mesh.clone(), pending(&target_mesh, &["y", "x"])).unwrap();
        for (contributions, sum) in [
            ([f32::MAX, -f32::MAX, 0., f32::MAX, -f32::MAX, 0.], 0f32),
            ([16_777_216f32, 1., 0., -16_777_216., 1., 0.], 1.),
        ] {
            let source = Array::from_host_shards(
                &session,
                ArrayType::scalar(DataType::F32),
                source_mesh.clone(),
                Some(pending(&source_mesh, &["x", "y"])),
                devices
                    .iter()
                    .zip(contributions)
                    .map(|(device, value)| (device.id(), Arc::<[u8]>::from(values_to_bytes(&[value])))),
            )
            .unwrap();
            let before = source.copy_to_host_bytes().unwrap();
            assert_eq!(values_from_bytes::<f32>(&before), [sum]);
            for placed in place_through_entry_points(&session, &source, target.clone()) {
                let placed = placed.unwrap();
                // The device at position `k` of the target mesh holds the contribution with ordinal `k`.
                let moved = target_mesh
                    .devices()
                    .iter()
                    .map(|device| placed.device_shard(device.id()).unwrap().copy_to_host_bytes().unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(moved, contributions.map(|value| values_to_bytes(&[value])));
                assert_eq!(placed.copy_to_host_bytes().unwrap(), before);
            }
        }
    }

    #[test]
    fn test_array_to_placement_slices_available_local_replica() {
        let client = cpu_client(2);
        let client_devices = client.addressable_devices().unwrap();
        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let remote = Device::new(devices[1].id() + 1, devices[1].process_index() + 1);
        let session = session_with_remote_devices(client, [remote]);
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 3, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![devices[0], devices[1], remote],
        )
        .unwrap();
        let target_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("target", 2, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![devices[0], remote],
        )
        .unwrap();
        let replicated = Sharding::replicated(source_mesh.logical_mesh().clone(), 1);
        let split =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["target"])]).unwrap();
        let target = Placement::new(target_mesh.clone(), split.clone()).unwrap();

        // The source holds only one of its local replicas, and the plan names the replica on the target device.
        // Only this process takes part in the region of that device, so whichever local replica holds a buffer serves
        // it, sliced on its own device and copied to the target device when that differs.
        for (holder, holder_device) in client_devices.iter().zip(&devices) {
            let buffer =
                client.buffer([10u8, 20, 30, 40].as_slice(), BufferType::U8, [4u64], None, holder.clone(), None);
            let source = Array::from_addressable_buffers(
                &session,
                ArrayType::new_static(DataType::U8, [4]),
                source_mesh.clone(),
                Some(replicated.clone()),
                vec![buffer.unwrap()],
            )
            .unwrap();
            assert_eq!(source.copy_to_host_bytes().unwrap(), [10, 20, 30, 40]);
            let placed = source.to_placement(&session, target.clone()).unwrap();
            assert_eq!(placed.mesh(), &target_mesh);
            assert_eq!(placed.sharding(), &split);
            assert!(placed.is_committed());
            assert_eq!(placed.addressable_shards().count(), 1);
            assert!(placed.device_shard(remote.id()).unwrap().buffer().is_none());
            let buffer = placed.device_shard(devices[0].id()).unwrap().buffer().unwrap();
            assert_eq!(buffer.device().unwrap().id().unwrap(), devices[0].id());
            assert_eq!(buffer.dimensions().unwrap(), [2]);
            assert_eq!(placed.block_until_ready().map_err(|error| error.to_string()), Ok(()));
            assert_eq!(placed.device_shard(devices[0].id()).unwrap().copy_to_host_bytes().unwrap(), [10, 20]);
            assert_eq!(
                source.device_shard(holder_device.id()).unwrap().copy_to_host_bytes().unwrap(),
                [10, 20, 30, 40],
            );
        }
    }

    #[test]
    fn test_array_to_placement_rejects_missing_source_shard_buffers() {
        // Shard #0 of the source lies on the addressable device 0 but has no buffer, and no other device of this
        // process holds a replica of it.
        let session = session_with_remote_devices(cpu_client(1), [Device::new(1, 1)]);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let source_mesh = DeviceMesh::new(logical_mesh.clone(), vec![Device::new(0, 0), Device::new(1, 1)]).unwrap();
        let sharding = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let source =
            Array::from_addressable_buffers(&session, r#type, source_mesh, Some(sharding), Vec::new()).unwrap();
        let error = source.to_placement(&session, Device::new(0, 0)).unwrap_err();
        assert!(matches!(
            &error,
            Error::Array(error) if matches!(error.as_ref(),
                ArrayError::MissingShardBuffer { shard_index: 0, device_id: 0 },
            ),
        ));
        assert_eq!(error.to_string(), "array shard #0 on device 0 has no addressable buffer");
    }

    #[test]
    fn test_array_to_placement_combines_unreduced_device_values() {
        // The devices along the unreduced axis `x` hold summands of the logical value, and `y` tiles its only
        // dimension. Every target shard has the slices of a source shard, but a copy of that shard would hold
        // one summand instead of the logical value, so the placement completes the sum along `x` while `y` keeps
        // tiling the value.
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client.addressable_devices().unwrap();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), session.addressable_devices().unwrap()).unwrap();
        let tiled = Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["y"])]).unwrap();
        let unreduced = tiled.clone().with_unreduced_axes(["x"]).unwrap();
        let buffers = [[1f32, 2.], [3., 4.], [10., 20.], [30., 40.]]
            .iter()
            .zip(&devices)
            .map(|(values, device)| {
                client
                    .buffer(values_to_bytes(values).as_slice(), BufferType::F32, [2], None, device.clone(), None)
                    .unwrap()
            })
            .collect();
        let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(unreduced).unwrap();
        let summands = Array::from_addressable_buffers(&session, r#type, mesh.clone(), None, buffers).unwrap();
        let reduced = summands.to_placement(&session, Placement::new(mesh, tiled.clone()).unwrap()).unwrap();
        assert_eq!(reduced.sharding(), &tiled);
        assert_eq!(shard_values(&reduced), vec![vec![11., 22.], vec![33., 44.], vec![11., 22.], vec![33., 44.]]);
    }

    #[test]
    fn test_array_to_placement_completes_pending_reduction_with_and_without_donation() {
        // Two devices hold the contributions `[1, 5, 3, 0]` and `[2, 4, 3, 7]` to an `f32[4]` value with a pending
        // reduction of each kind, over an auto and over an explicit mesh axis, which a placement splits along that
        // axis, on the same mesh and onto two other devices. Completing a pending sum (e.g., of the partial sums of a
        // gradient) is a capability of the placement of an array, whether or not it donates its source, and it
        // completes on the devices for either axis type, because the host round trip needs a target that this
        // process fully addresses. On the same mesh, one compiled reshard completes it (i.e., one program with and one
        // without a donated input). Across meshes, the gather of the source completes it before its replicas move.
        // XLA's maximum and minimum reductions drop NaN values, so a pending maximum or minimum of floating point
        // values takes the host round trip from both entry points and compiles nothing (refer to
        // `test_array_to_placement_preserves_value_across_reduction_state_changes`). Each axis type has a session of
        // its own, so that every cell that compiles a program adds it to the count, and both axis types must give the
        // same rows.
        let client = cpu_client(4);
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let mut outcomes_by_axis_type = Vec::new();
        for axis_type in [MeshAxisType::Auto, MeshAxisType::Explicit] {
            let mut outcomes = Vec::new();
            let session = Session::builder().client(client).build().unwrap();
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let mesh = DeviceMesh::new(logical_mesh.clone(), devices[..2].to_vec()).unwrap();
            let other_mesh = DeviceMesh::new(logical_mesh.clone(), devices[2..].to_vec()).unwrap();
            let split = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
            for kind in [UnreducedKind::Sum, UnreducedKind::Max, UnreducedKind::Min] {
                let unreduced = Sharding::replicated(logical_mesh.clone(), 1)
                    .with_unreduced_axes(["x"])
                    .unwrap()
                    .with_unreduced_kind(kind)
                    .unwrap();
                let source = || {
                    let contributions = devices
                        .iter()
                        .zip([[1f32, 5., 3., 0.], [2., 4., 3., 7.]])
                        .map(|(device, values)| (device.id(), Arc::<[u8]>::from(values_to_bytes(&values))))
                        .collect::<Vec<_>>();
                    let r#type = ArrayType::new_static(DataType::F32, [4]);
                    let sharding = Some(unreduced.clone());
                    Array::from_host_shards(&session, r#type, mesh.clone(), sharding, contributions).unwrap()
                };
                for (entry_point, target_mesh, consume) in
                    [("borrowed", &mesh, false), ("consumed", &mesh, true), ("across meshes", &other_mesh, false)]
                {
                    let programs = session.cached_program_count();
                    let target = Placement::new(target_mesh.clone(), split.clone()).unwrap();
                    let placed = if consume {
                        source().into_placement(&session, target)
                    } else {
                        source().to_placement(&session, target)
                    };
                    let placed = placed.map(|placed| {
                        assert_eq!((placed.mesh(), placed.sharding()), (target_mesh, &split));
                        shard_values(&placed)
                    });
                    let programs = session.cached_program_count() - programs;
                    outcomes.push(format!("{kind:?} {entry_point}: +{programs} {placed:?}"));
                }
            }
            outcomes_by_axis_type.push(outcomes);
        }
        assert_eq!(outcomes_by_axis_type[0], outcomes_by_axis_type[1], "both axis types must give the same rows");
        assert_eq!(
            outcomes_by_axis_type[0].join("\n"),
            indoc! {"
                Sum borrowed: +1 Ok([[3.0, 9.0], [6.0, 7.0]])
                Sum consumed: +1 Ok([[3.0, 9.0], [6.0, 7.0]])
                Sum across meshes: +2 Ok([[3.0, 9.0], [6.0, 7.0]])
                Max borrowed: +0 Ok([[2.0, 5.0], [3.0, 7.0]])
                Max consumed: +0 Ok([[2.0, 5.0], [3.0, 7.0]])
                Max across meshes: +0 Ok([[2.0, 5.0], [3.0, 7.0]])
                Min borrowed: +0 Ok([[1.0, 4.0], [3.0, 0.0]])
                Min consumed: +0 Ok([[1.0, 4.0], [3.0, 0.0]])
                Min across meshes: +0 Ok([[1.0, 4.0], [3.0, 0.0]])"},
        );
    }

    #[test]
    fn test_array_to_placement_keeps_or_declines_reduction_state_changes() {
        // Placements of an `f32[4]` value on a 2x2 mesh whose axes `x` and `y` are explicit or auto, from and to
        // shardings with reduction state, with the number of programs that each entry point compiles in a session of
        // its own (i.e., on the same mesh with and without donation, and onto the same devices in reverse order). A
        // pending reduction that a target keeps moves as regions of the shards of each contribution, on the same
        // mesh too, so that sparse and dense sources behave alike, which slice or join the shards of every device
        // (`+4`) and complete nothing, whatever the type of the axes. On the same mesh, the compiled reshard is the
        // fallback when regions do not apply. A compiled reshard makes every other change over auto axes (i.e., it
        // completes a pending reduction and introduces a pending sum or a completed reduction). XLA cannot introduce a
        // pending maximum or minimum, so a target with one that the source does not keep compiles nothing and
        // takes the host round trip (e.g., across meshes, where the gather completes the reduction that the final
        // reshard would have to introduce again). Every cell reads back the logical value of its source with the
        // target sharding, and every combination of axis types must give the same rows.
        let client = cpu_client(4);
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let (a, b, c, d) = ([1f32, 5., 3., 0.], [2f32, 4., 3., 7.], [10f32, 20., 30., 40.], [100f32, 200., 300., 400.]);
        let combine = |kind: UnreducedKind, values: &[[f32; 4]]| -> Vec<f32> {
            (0..4)
                .map(|index| {
                    let column = values.iter().map(|value| value[index]);
                    match kind {
                        UnreducedKind::Sum => column.sum(),
                        UnreducedKind::Max => column.fold(f32::MIN, f32::max),
                        UnreducedKind::Min => column.fold(f32::MAX, f32::min),
                    }
                })
                .collect()
        };
        // The contributions of the devices in mesh order of a pending reduction along `x` or along `y`.
        let along_x = vec![a.to_vec(), a.to_vec(), b.to_vec(), b.to_vec()];
        let along_y = vec![a.to_vec(), b.to_vec(), a.to_vec(), b.to_vec()];
        let mut outcomes_by_axis_types = Vec::new();
        for (x_type, y_type) in [
            (MeshAxisType::Explicit, MeshAxisType::Explicit),
            (MeshAxisType::Auto, MeshAxisType::Auto),
            (MeshAxisType::Explicit, MeshAxisType::Auto),
            (MeshAxisType::Auto, MeshAxisType::Explicit),
        ] {
            let logical_mesh =
                LogicalMesh::new(vec![MeshAxis::new("x", 2, x_type).unwrap(), MeshAxis::new("y", 2, y_type).unwrap()])
                    .unwrap();
            let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
            let reversed_mesh = DeviceMesh::new(logical_mesh.clone(), devices.iter().rev().copied().collect()).unwrap();
            let replicated = Sharding::replicated(logical_mesh.clone(), 1);
            let split_y = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
            let unconstrained = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::unconstrained()]).unwrap();
            let unreduced = |sharding: &Sharding, axes: &[&str], kind: UnreducedKind| {
                sharding
                    .clone()
                    .with_unreduced_axes(axes.iter().copied())
                    .unwrap()
                    .with_unreduced_kind(kind)
                    .unwrap()
            };
            let reduced = |sharding: &Sharding| sharding.clone().with_reduced_axes(["x"]).unwrap();
            let maximum = unreduced(&replicated, &["x"], UnreducedKind::Max);
            // Each source is its sharding, the contributions of the devices in mesh order, and its logical value.
            let sum_x = (
                unreduced(&replicated, &["x"], UnreducedKind::Sum),
                along_x.clone(),
                combine(UnreducedKind::Sum, &[a, b]),
            );
            let sum_x_split_y = (
                unreduced(&split_y, &["x"], UnreducedKind::Sum),
                vec![a[..2].to_vec(), a[2..].to_vec(), b[..2].to_vec(), b[2..].to_vec()],
                combine(UnreducedKind::Sum, &[a, b]),
            );
            let max_x = (maximum.clone(), along_x.clone(), combine(UnreducedKind::Max, &[a, b]));
            let min_x = (
                unreduced(&replicated, &["x"], UnreducedKind::Min),
                along_x.clone(),
                combine(UnreducedKind::Min, &[a, b]),
            );
            let sum_y = (
                unreduced(&replicated, &["y"], UnreducedKind::Sum),
                along_y.clone(),
                combine(UnreducedKind::Sum, &[a, b]),
            );
            let sum_x_y = (
                unreduced(&replicated, &["x", "y"], UnreducedKind::Sum),
                vec![a.to_vec(), b.to_vec(), c.to_vec(), d.to_vec()],
                combine(UnreducedKind::Sum, &[a, b, c, d]),
            );
            let plain = (replicated.clone(), vec![c.to_vec(); 4], c.to_vec());
            let plain_split_y =
                (split_y.clone(), vec![c[..2].to_vec(), c[2..].to_vec(), c[..2].to_vec(), c[2..].to_vec()], c.to_vec());
            let reduced_x = (reduced(&replicated), vec![c.to_vec(); 4], c.to_vec());
            let cells = [
                // Pure moves that keep reduction state.
                ("sum(x) -> sum(x) split(y)", &sum_x, unreduced(&split_y, &["x"], UnreducedKind::Sum)),
                ("sum(x) split(y) -> sum(x)", &sum_x_split_y, unreduced(&replicated, &["x"], UnreducedKind::Sum)),
                // An unconstrained dimension is stored replicated, so this one only moves contributions, as whole
                // shards on either mesh, and compiles nothing.
                ("sum(x) -> sum(x) {?}", &sum_x, unreduced(&unconstrained, &["x"], UnreducedKind::Sum)),
                ("max(x) -> max(x) split(y)", &max_x, unreduced(&split_y, &["x"], UnreducedKind::Max)),
                ("min(x) -> min(x) split(y)", &min_x, unreduced(&split_y, &["x"], UnreducedKind::Min)),
                ("reduced(x) -> reduced(x) split(y)", &reduced_x, reduced(&split_y)),
                // A completed reduction along `x` and a kept one along `y`.
                ("sum(x, y) -> sum(y)", &sum_x_y, unreduced(&replicated, &["y"], UnreducedKind::Sum)),
                // Targets with a pending maximum.
                ("sum(x) -> max(x)", &sum_x, maximum.clone()),
                ("max(x) -> max(x)", &max_x, maximum.clone()),
                ("min(x) -> max(x)", &min_x, maximum.clone()),
                ("sum(y) -> max(x)", &sum_y, maximum.clone()),
                ("sum(x, y) -> max(x)", &sum_x_y, maximum.clone()),
                ("replicated -> max(x)", &plain, maximum.clone()),
                ("split(y) -> max(x)", &plain_split_y, maximum.clone()),
                ("reduced(x) -> max(x)", &reduced_x, maximum.clone()),
            ];
            let mut outcomes = Vec::new();
            for (name, (source_sharding, contributions, logical_value), target_sharding) in cells {
                let context = format!("{x_type:?}/{y_type:?} {name}");
                let mut counts = Vec::new();
                for (target_mesh, consume) in [(&mesh, false), (&mesh, true), (&reversed_mesh, false)] {
                    let session = Session::builder().client(client).build().unwrap();
                    let contributions = devices
                        .iter()
                        .zip(contributions)
                        .map(|(device, values)| (device.id(), Arc::<[u8]>::from(values_to_bytes(values))))
                        .collect::<Vec<_>>();
                    let r#type = ArrayType::new_static(DataType::F32, [4]);
                    let sharding = Some(source_sharding.clone());
                    let source =
                        Array::from_host_shards(&session, r#type, mesh.clone(), sharding, contributions).unwrap();
                    let target = Placement::new(target_mesh.clone(), target_sharding.clone()).unwrap();
                    let placed = if consume {
                        source.into_placement(&session, target)
                    } else {
                        source.to_placement(&session, target)
                    };
                    let placed = placed.unwrap();
                    assert_eq!((placed.mesh(), placed.sharding()), (target_mesh, &target_sharding), "{context}");
                    assert_eq!(&placed.copy_to_host().unwrap().elements::<f32>().unwrap(), logical_value, "{context}");
                    counts.push(format!("+{}", session.cached_program_count()));
                }
                outcomes.push(format!("{name}: {}", counts.join(" ")));
            }
            outcomes_by_axis_types.push(outcomes);
        }
        for outcomes in &outcomes_by_axis_types[1..] {
            assert_eq!(outcomes, &outcomes_by_axis_types[0], "every combination of axis types must give the same rows");
        }
        assert_eq!(
            outcomes_by_axis_types[0].join("\n"),
            indoc! {"
                sum(x) -> sum(x) split(y): +4 +4 +4
                sum(x) split(y) -> sum(x): +4 +4 +4
                sum(x) -> sum(x) {?}: +0 +0 +0
                max(x) -> max(x) split(y): +4 +4 +4
                min(x) -> min(x) split(y): +4 +4 +4
                reduced(x) -> reduced(x) split(y): +1 +1 +4
                sum(x, y) -> sum(y): +1 +1 +2
                sum(x) -> max(x): +0 +0 +0
                max(x) -> max(x): +0 +0 +0
                min(x) -> max(x): +0 +0 +0
                sum(y) -> max(x): +0 +0 +0
                sum(x, y) -> max(x): +0 +0 +0
                replicated -> max(x): +0 +0 +0
                split(y) -> max(x): +0 +0 +0
                reduced(x) -> max(x): +0 +0 +0"},
        );
    }

    #[test]
    fn test_array_to_placement_introduces_pending_sums_only_on_devices_at_coordinate_zero() {
        // A placement that introduces a pending sum places the value only on the devices at coordinate zero along the
        // introduced axes (here `y`, so devices 0 and 2 of the 2x2 mesh) and gives the others the additive identity
        // (negative zero for `f32`), uploaded on their own. A source that already holds the value there with the
        // target placement along the other axes (e.g., replicated, or with a pending sum along `x` that the
        // target keeps) needs no move and no program, so those devices keep the buffers of the source and the
        // placement creates only its result.
        let client = cpu_client(4);
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
        let replicated = Sharding::replicated(logical_mesh.clone(), 1);
        let split_y = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let pending = |axes: &[&str]| replicated.clone().with_unreduced_axes(axes.iter().copied()).unwrap();
        let single_mesh = LogicalMesh::new(vec![MeshAxis::new("z", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let single = DeviceMesh::new(single_mesh.clone(), vec![devices[0]]).unwrap();
        let (value, other, zeros) = (vec![1f32, 2., 3., 4.], vec![10f32, 20., 30., 40.], vec![0f32; 4]);
        let (first_half, second_half) = (value[..2].to_vec(), value[2..].to_vec());
        // Each row is the name of the source, its mesh, its sharding, and the values of its devices in mesh order,
        // and the target sharding and the values that the devices hold after the placement in mesh order.
        let rows = [
            (
                "replicated",
                &mesh,
                replicated.clone(),
                vec![value.clone(); 4],
                pending(&["y"]),
                vec![value.clone(), zeros.clone(), value.clone(), zeros.clone()],
            ),
            (
                "sum along x",
                &mesh,
                pending(&["x"]),
                vec![value.clone(), value.clone(), other.clone(), other.clone()],
                pending(&["x", "y"]),
                vec![value.clone(), zeros.clone(), other.clone(), zeros.clone()],
            ),
            (
                "split along y",
                &mesh,
                split_y,
                vec![first_half.clone(), second_half.clone(), first_half, second_half],
                pending(&["y"]),
                vec![value.clone(), zeros.clone(), value.clone(), zeros.clone()],
            ),
            (
                "on device 0",
                &single,
                Sharding::replicated(single_mesh, 1),
                vec![value.clone()],
                pending(&["x", "y"]),
                vec![value.clone(), zeros.clone(), zeros.clone(), zeros],
            ),
        ];
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let mut outcomes = Vec::new();
        for (name, source_mesh, source_sharding, source_values, target, expected) in rows {
            let session = Session::builder().client(client).build().unwrap();
            let shards = devices
                .iter()
                .zip(&source_values)
                .map(|(device, values)| (device.id(), Arc::<[u8]>::from(values_to_bytes(values))));
            let source = Array::from_host_shards(
                &session,
                r#type.clone(),
                source_mesh.clone(),
                Some(source_sharding),
                shards.collect::<Vec<_>>(),
            )
            .unwrap();
            let source_buffers =
                source.addressable_shards().filter_map(ArrayShard::buffer).cloned().collect::<Vec<_>>();
            let (programs, arrays) = (session.cached_program_count(), session.array_statistics().created_arrays);
            let placed = source.to_placement(&session, Placement::new(mesh.clone(), target.clone()).unwrap());
            let placed = placed.unwrap();
            assert_eq!((placed.mesh(), placed.sharding()), (&mesh, &target), "{name}");
            let shared = placed
                .addressable_shards()
                .enumerate()
                .filter(|(_, shard)| {
                    let buffer = shard.buffer().unwrap();
                    source_buffers.iter().any(|source| Arc::ptr_eq(source, buffer))
                })
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            outcomes.push(format!(
                "{name}: +{} programs, +{} arrays, devices sharing a source buffer {shared:?}",
                session.cached_program_count() - programs,
                session.array_statistics().created_arrays - arrays,
            ));
            assert_eq!(shard_values(&placed), expected, "{name}");
            let bits = values_from_bytes::<f32>(&placed.shards()[1].copy_to_host_bytes().unwrap())
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
            assert_eq!(bits, vec![0x8000_0000; 4], "{name}");
        }
        assert_eq!(
            outcomes.join("\n"),
            indoc! {"
                replicated: +0 programs, +1 arrays, devices sharing a source buffer [0, 2]
                sum along x: +0 programs, +1 arrays, devices sharing a source buffer [0, 2]
                split along y: +2 programs, +8 arrays, devices sharing a source buffer []
                on device 0: +0 programs, +2 arrays, devices sharing a source buffer [0]"},
        );
    }

    #[test]
    fn test_array_to_placement_introduces_pending_sums_from_local_replicas() {
        // Process 0 of two holds a replicated value over the 2x2 mesh of `[d0, R0, d1, R1]` with a buffer on `d1` only.
        // Introducing a pending sum along `y` keeps the value on `d0` and `d1`, so `d0` is served from the local
        // replica on `d1`, as exact shards serve a missing replica, and nothing is compiled. Without any local buffer,
        // the placement fails with a missing shard buffer.
        let (session, mesh) = two_process_mesh_2x2();
        let client = session.client();
        let local = session.addressable_devices().unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let target = replicated.clone().with_unreduced_axes(["y"]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let values = [1f32, 2., 3., 4.];
        let device = client.addressable_devices().unwrap()[1].clone();
        let buffer = client.buffer(values_to_bytes(&values).as_slice(), BufferType::F32, [4u64], None, device, None);
        let buffers = vec![buffer.unwrap()];
        let source =
            Array::from_addressable_buffers(&session, r#type.clone(), mesh.clone(), Some(replicated.clone()), buffers);
        let source = source.unwrap();
        let source_buffer = Arc::clone(source.shards()[2].buffer().unwrap());
        let programs = session.cached_program_count();
        let placed = source.to_placement(&session, Placement::new(mesh.clone(), target.clone()).unwrap()).unwrap();
        assert_eq!((placed.sharding(), session.cached_program_count() - programs), (&target, 0));
        let held = placed
            .addressable_shards()
            .map(|shard| {
                let buffer = shard.buffer().unwrap();
                let values = values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap());
                (shard.device(), values, Arc::ptr_eq(buffer, &source_buffer))
            })
            .collect::<Vec<_>>();
        assert_eq!(held, vec![(local[0], values.to_vec(), false), (local[1], values.to_vec(), true)]);

        let empty = Array::from_addressable_buffers(&session, r#type, mesh.clone(), Some(replicated), Vec::new());
        let placed = empty.unwrap().to_placement(&session, Placement::new(mesh, target).unwrap());
        assert_eq!(
            placed.map(|_| ()).map_err(|error| error.to_string()),
            Err(format!("array shard #0 on device {} has no addressable buffer", local[0].id())),
        );
    }

    #[test]
    fn test_array_to_placement_introduces_pending_sums_without_buffers_for_empty_shards() {
        // Process 0 of two holds a `u8[1]` split along `x`, or a `u8[1, 2]` split into rows along `x`, over the 2x2
        // mesh of `[d0, R0, d1, R1]` with a buffer on `d0` only, since the slice of `d1` is empty. Introducing a
        // pending sum along `y`, with the vector kept and the columns of the matrix split along `x`, restricts the
        // value to the devices at `y = 0`, where the empty slice of `d1` needs no buffer, so `d1` gets an empty shard
        // of the vector and a column of the matrix.
        let (session, mesh) = two_process_mesh_2x2();
        let client = session.client();
        let logical_mesh = mesh.logical_mesh().clone();
        let split = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let rows = vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated];
        let columns = vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])];
        let columns = Sharding::new(logical_mesh.clone(), columns).unwrap().with_unreduced_axes(["y"]).unwrap();
        let cases = [
            (
                vec![1u64],
                vec![42u8],
                split.clone(),
                split.with_unreduced_axes(["y"]).unwrap(),
                vec![vec![42u8], vec![]],
            ),
            (vec![1, 2], vec![42, 43], Sharding::new(logical_mesh, rows).unwrap(), columns, vec![vec![42], vec![43]]),
        ];
        for (dimensions, bytes, source_sharding, target_sharding, expected) in cases {
            let device = client.addressable_devices().unwrap()[0].clone();
            let buffer = client.buffer(bytes.as_slice(), BufferType::U8, dimensions.clone(), None, device, None);
            let r#type =
                ArrayType::new_static(DataType::U8, dimensions.iter().map(|&size| size as usize).collect::<Vec<_>>());
            let source = Array::from_addressable_buffers(
                &session,
                r#type,
                mesh.clone(),
                Some(source_sharding),
                vec![buffer.unwrap()],
            );
            let source = source.unwrap();
            let target = Placement::new(mesh.clone(), target_sharding).unwrap();
            for placed in
                [source.to_placement(&session, target.clone()), source.clone().into_placement(&session, target)]
            {
                let held = placed
                    .unwrap()
                    .addressable_shards()
                    .map(|shard| shard.copy_to_host_bytes().unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(held, expected, "{dimensions:?}");
            }
        }
    }

    #[test]
    fn test_array_to_placement_introduces_pending_sums_reporting_failures_of_empty_shards() {
        // A replicated `u8[0]` whose pending uploads fail after it is placed into a pending sum along `x`. The sum
        // replaces its empty buffers, which nothing reads, and still reports their failure, as an exact move does.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let managers = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| {
                let specification = BufferSpecification::new(BufferType::U8, [0u64]);
                client
                    .host_to_device_transfer_manager(vec![specification], device.default_memory().unwrap())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let buffers = managers.iter().map(|manager| manager.retrieve_buffer(0).unwrap()).collect();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let r#type = ArrayType::new_static(DataType::U8, [0]);
        let source =
            Array::from_addressable_buffers(&session, r#type, mesh.clone(), Some(replicated.clone()), buffers).unwrap();
        let target = Placement::new(mesh, replicated.with_unreduced_axes(["x"]).unwrap()).unwrap();
        let placed = source.to_placement(&session, target).unwrap();
        for manager in &managers {
            manager.set_error(0, PjrtError::aborted("empty upload failed")).unwrap();
        }
        assert!(matches!(
            placed.block_until_ready(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "empty upload failed",
        ));
        assert!(matches!(
            placed.copy_to_host_bytes(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "empty upload failed",
        ));
    }

    #[test]
    fn test_array_to_placement_completes_pending_sums_in_the_order_and_precision_of_readback() {
        // Four contributions whose sum depends on the order and the precision of accumulation, rotated so that every
        // element sees another order (element `e` of device `d` holds `contributions[(d + e) % 4]`). The collective of
        // the CPU plugin accumulates in ascending order of the coordinates, `bf16` sums in `f32`, and the sums of the
        // eight-bit formats in `f16`, which the readback reproduces, so a readback and a placement that completes the
        // sum agree on both axis types (e.g., on `2` everywhere for the `bf16` and `f8e4m3fn` rows, whose sums in
        // the element type would give `[1, 2, 0, 0]`).
        let session = Session::builder().client(cpu_client(4)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let rows = [
            (DataType::BF16, 256.0, [2.0, 2.0, 2.0, 2.0]),
            (DataType::BF16, 16777216.0, [1.0, 2.0, 2.0, 2.0]),
            (DataType::F32, 16777216.0, [1.0, 2.0, 2.0, 2.0]),
            (DataType::F16, 2048.0, [1.0, 2.0, 2.0, 2.0]),
            (DataType::F8E4M3FN, 256.0, [2.0, 2.0, 2.0, 2.0]),
            (DataType::F8E5M2, 57344.0, [1.0, 2.0, 0.0, 0.0]),
        ];
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Auto] {
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, axis_type).unwrap()]).unwrap();
            let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
            let pending = Sharding::replicated(logical_mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap();
            let replicated = Placement::new(mesh.clone(), Sharding::replicated(logical_mesh, 1)).unwrap();
            for (data_type, power, expected) in rows {
                let r#type = ArrayType::new_static(data_type, [4]);
                let contributions = [1.0f64, power, -power, 1.0];
                let encode = |value: f64| {
                    CoreArray::scalar(value).unwrap().convert_element_type(data_type).unwrap().logical_bytes()
                };
                let shards = (0..4).map(|device| {
                    let bytes = (0..4).flat_map(|element| encode(contributions[(device + element) % 4]));
                    (devices[device].id(), Arc::<[u8]>::from(bytes.collect::<Vec<_>>()))
                });
                let decode = |bytes: Vec<u8>| {
                    let array = CoreArray::new(r#type.clone(), bytes).unwrap();
                    array.convert_element_type(DataType::F64).unwrap().elements::<f64>().unwrap()
                };
                let source =
                    Array::from_host_shards(&session, r#type.clone(), mesh.clone(), Some(pending.clone()), shards)
                        .unwrap();
                let readback = decode(source.copy_to_host_bytes().unwrap());
                let placed =
                    decode(source.to_placement(&session, replicated.clone()).unwrap().copy_to_host_bytes().unwrap());
                assert_eq!(
                    (data_type, axis_type, readback, placed),
                    (data_type, axis_type, expected.to_vec(), expected.to_vec())
                );
            }
        }
    }

    #[test]
    fn test_array_to_placement_keeps_partial_sums_of_sparse_imports_onto_meshes_with_unused_axes() {
        // An `f32[4]` value with a pending sum along `x` and `y` of a 2x2x2 mesh, whose devices along `r` hold
        // replicas, imported with the buffers of the devices at `r = 1` only, completes `y` onto the same devices with
        // an unused axis `u` of size one, which keeps the partial sums `MAX` and `0` (or `7` and `11`) along `x`. The
        // gather of the compiled reshard across meshes serves the absent buffers from local replicas, and no host
        // fallback, which would combine the partial sums, is admitted for a placement that keeps some contributions.
        let client = cpu_client(8);
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();
        let devices = session.addressable_devices().unwrap();
        let mesh_on = |axes: &[(&str, usize)]| {
            let axes = axes.iter().map(|&(name, size)| MeshAxis::new(name, size, MeshAxisType::Explicit).unwrap());
            DeviceMesh::new(LogicalMesh::new(axes.collect()).unwrap(), devices.clone()).unwrap()
        };
        let source_mesh = mesh_on(&[("x", 2), ("y", 2), ("r", 2)]);
        let target_mesh = mesh_on(&[("x", 2), ("y", 2), ("r", 2), ("u", 1)]);
        let source_sharding =
            Sharding::replicated(source_mesh.logical_mesh().clone(), 1).with_unreduced_axes(["x", "y"]).unwrap();
        let target_sharding =
            Sharding::replicated(target_mesh.logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let rows =
            [([f32::MAX, 0., f32::MAX, -f32::MAX], [f32::MAX, 0.], f32::MAX), ([3., 4., 5., 6.], [7., 11.], 18.)];
        for (values, partial_sums, value) in rows {
            let buffers = (1..8)
                .step_by(2)
                .map(|index| {
                    let bytes = values_to_bytes(&[values[index / 2]; 4]);
                    let device = client_devices[index].clone();
                    client.buffer(bytes.as_slice(), BufferType::F32, [4u64], None, device, None).unwrap()
                })
                .collect();
            let r#type = ArrayType::new_static(DataType::F32, [4]);
            let sharding = Some(source_sharding.clone());
            let source = Array::from_addressable_buffers(&session, r#type, source_mesh.clone(), sharding, buffers);
            let source = source.unwrap();
            let target = Placement::new(target_mesh.clone(), target_sharding.clone()).unwrap();
            let mut expected = vec![vec![partial_sums[0]; 4]; 4];
            expected.extend(vec![vec![partial_sums[1]; 4]; 4]);
            let borrowed = source.to_placement(&session, target.clone());
            for placed in [borrowed, source.clone().into_placement(&session, target)] {
                let placed = placed.unwrap();
                assert_eq!(placed.sharding(), &target_sharding);
                assert_eq!(shard_values(&placed), expected);
                assert_eq!(values_from_bytes::<f32>(&placed.copy_to_host_bytes().unwrap()), [value; 4]);
            }
        }
    }

    #[test]
    fn test_array_to_placement_admits_process_local_values_alike_on_every_process() {
        // Two processes, each with eight devices, upload the same value without naming a mesh, so that each places it
        // on its own first devices, and place it on devices of process 0. Process 1 owns no target device and
        // returns the metadata of the result, and both processes return the same result or the same error, which
        // every process derives from metadata alone:
        //   - `u8[n] = [3; n]` with a pending sum along `x` and `y` onto a target that keeps `x` and splits the
        //     value along `y`, which needs a compiled reshard that cannot produce uneven shards, while a host round
        //     trip would combine the contributions along `x`, so `n = 5` is rejected. The value is on the four
        //     target devices, on them in reversed order (so process 0 first moves its value onto them), or on
        //     eight devices along a further replicated axis `r` (so every process plans it on a view with devices
        //     beyond the target). Process 1 takes part in the admission, and it returns an array without shards
        //     when the placement is admitted.
        //   - `f8e8m0fnu[1]`, which has no additive identity, onto two devices that introduce a pending sum along `x`,
        //     which process 0 uploads through its host and process 1 checks alike.
        let sessions = sessions_of_processes([cpu_client(8), process_one_cpu_client(8)]);
        let devices = sessions[0].addressable_devices().unwrap();
        let logical_mesh = |axes: &[(&str, usize)]| {
            let axes = axes.iter().map(|&(name, size)| MeshAxis::new(name, size, MeshAxisType::Auto).unwrap());
            LogicalMesh::new(axes.collect()).unwrap()
        };
        let xy = logical_mesh(&[("x", 2), ("y", 2)]);
        let xyr = logical_mesh(&[("x", 2), ("y", 2), ("r", 2)]);
        let x = logical_mesh(&[("x", 2)]);
        let pending = |mesh: &LogicalMesh| Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["x", "y"]);
        let split = Sharding::new(xy.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
        let split = split.with_unreduced_axes(["x"]).unwrap();
        let split = Placement::new(DeviceMesh::new(xy.clone(), devices[..4].to_vec()).unwrap(), split).unwrap();
        let reversed_devices = devices[..4].iter().rev().copied().collect::<Vec<_>>();
        let reversed = Placement::new(DeviceMesh::new(xy.clone(), reversed_devices).unwrap(), split.sharding().clone());
        let reversed = reversed.unwrap();
        let summands = Sharding::replicated(x.clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let summands = Placement::new(DeviceMesh::new(x, devices[..2].to_vec()).unwrap(), summands).unwrap();
        let uneven = "cannot place an array whose pending reduction the placement keeps along the mesh axes `x` and \
                      changes along others when no device strategy applies (compiled reshard cannot produce the shards \
                      of different sizes of shape `[5]` with sharding `{mesh<['x'=2:auto, 'y'=2:auto]>, [{'y'}], \
                      unreduced={'x'}}`), because the host round trip that remains would combine the contributions \
                      that the placement keeps; complete the reduction first, or place the array onto an even split"
            .to_string();
        let identity = Error::from(ArrayError::MissingUnreducedSumIdentity { data_type: DataType::F8E8M0FNU });
        let identity = identity.to_string();
        let rows = [
            ("u8[4] onto its devices", DataType::U8, 4, Some(pending(&xy).unwrap()), &split, &uneven),
            ("u8[5] onto its devices", DataType::U8, 5, Some(pending(&xy).unwrap()), &split, &uneven),
            ("u8[4] onto its devices reversed", DataType::U8, 4, Some(pending(&xy).unwrap()), &reversed, &uneven),
            ("u8[5] onto its devices reversed", DataType::U8, 5, Some(pending(&xy).unwrap()), &reversed, &uneven),
            ("u8[4] from more devices", DataType::U8, 4, Some(pending(&xyr).unwrap()), &split, &uneven),
            ("u8[5] from more devices", DataType::U8, 5, Some(pending(&xyr).unwrap()), &split, &uneven),
            ("f8e8m0fnu[1] into summands", DataType::F8E8M0FNU, 1, None, &summands, &identity),
        ];
        let mut outcomes = Vec::new();
        for (name, data_type, size, source_sharding, target, rejected) in rows {
            let r#type = ArrayType::new_static(data_type, [size]);
            let bytes = if data_type == DataType::U8 { vec![3u8; size] } else { vec![0x7f; size] };
            let mut placements = Vec::new();
            for session in &sessions {
                let source = Array::from_host_buffer(session, r#type.clone(), None, source_sharding.clone(), &bytes);
                let source = source.unwrap();
                assert!(source.is_locally_assigned(), "{name}");
                // `Session::place` changes no reduction state, so both entry points are those of `Array`.
                let borrowed = source.to_placement(session, target.clone());
                let consumed = source.into_placement(session, target.clone());
                placements.extend([borrowed, consumed].map(|placed| placed.map_err(|error| error.to_string())));
            }
            let (owner, other) = process_outcomes(&placements, name);
            let outcome = format!("{name}: process 0 {owner}; process 1 {other}");
            outcomes.push(outcome.replace(&format!("error: {rejected}"), "error"));
        }
        assert_eq!(
            outcomes.join("\n"),
            indoc! {"
                u8[4] onto its devices: process 0 [[3, 3], [3, 3], [0, 0], [0, 0]]; process 1 []
                u8[5] onto its devices: process 0 error; process 1 error
                u8[4] onto its devices reversed: process 0 [[3, 3], [3, 3], [0, 0], [0, 0]]; process 1 []
                u8[5] onto its devices reversed: process 0 error; process 1 error
                u8[4] from more devices: process 0 [[3, 3], [3, 3], [0, 0], [0, 0]]; process 1 []
                u8[5] from more devices: process 0 error; process 1 error
                f8e8m0fnu[1] into summands: process 0 error; process 1 error"},
        );
    }

    #[test]
    fn test_array_to_placement_admits_what_the_owner_runs() {
        // Two processes, each with four devices, upload `u8[1] = [3]` replicated and varying along the manual axis `m`
        // without naming a mesh, so that each places it on its own devices, and place it on devices of process 0.
        // Every process derives the strategy that process 0 runs from the view of the value on those devices, and
        // process 1 checks only what that strategy checks. Each row lists the shard bytes that each process holds after
        // `Array::to_placement` and `Array::into_placement`, which agree:
        //   - onto the same sharding on two devices, in either order, which keeps both values of the slice with exact
        //     shards on process 0, with no host readback of a value that varies along `m`.
        //   - onto four devices that introduce a pending sum along `x`, in either order, which writes the additive
        //     identity on process 0 without a readback.
        let sessions = sessions_of_processes([cpu_client(4), process_one_cpu_client(4)]);
        let devices = sessions[0].addressable_devices().unwrap();
        let logical_mesh = |axes: &[(&str, MeshAxisType)]| {
            let axes = axes.iter().map(|&(name, r#type)| MeshAxis::new(name, 2, r#type).unwrap());
            LogicalMesh::new(axes.collect()).unwrap()
        };
        let m = logical_mesh(&[("m", MeshAxisType::Manual)]);
        let mx = logical_mesh(&[("m", MeshAxisType::Manual), ("x", MeshAxisType::Auto)]);
        let varying =
            |mesh: &LogicalMesh| Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["m"]).unwrap();
        let summands = varying(&mx).with_unreduced_axes(["x"]).unwrap();
        let target = |mesh: &LogicalMesh, devices: Vec<Device>, sharding: Sharding| {
            Placement::new(DeviceMesh::new(mesh.clone(), devices).unwrap(), sharding).unwrap()
        };
        let reversed = |devices: &[Device]| devices.iter().rev().copied().collect::<Vec<_>>();
        let rows = [
            ("onto two devices", &m, target(&m, devices[..2].to_vec(), varying(&m))),
            ("onto two devices reversed", &m, target(&m, reversed(&devices[..2]), varying(&m))),
            ("into summands", &mx, target(&mx, devices[..4].to_vec(), summands.clone())),
            ("into summands reversed", &mx, target(&mx, reversed(&devices[..4]), summands)),
        ];
        let mut outcomes = Vec::new();
        for (name, source_mesh, target) in rows {
            let mut placements = Vec::new();
            for session in &sessions {
                let r#type = ArrayType::new_static(DataType::U8, [1]);
                let source = Array::from_host_buffer(session, r#type, None, Some(varying(source_mesh)), [3u8]).unwrap();
                assert!(source.is_locally_assigned(), "{name}");
                let borrowed = source.to_placement(session, target.clone());
                let consumed = source.into_placement(session, target.clone());
                placements.extend([borrowed, consumed].map(|placed| placed.map_err(|error| error.to_string())));
            }
            let (owner, other) = process_outcomes(&placements, name);
            outcomes.push(format!("{name}: process 0 {owner}; process 1 {other}"));
        }
        assert_eq!(
            outcomes.join("\n"),
            indoc! {"
                onto two devices: process 0 [[3], [3]]; process 1 []
                onto two devices reversed: process 0 [[3], [3]]; process 1 []
                into summands: process 0 [[3], [0], [3], [0]]; process 1 []
                into summands reversed: process 0 [[3], [0], [3], [0]]; process 1 []"},
        );
    }

    #[test]
    fn test_array_to_placement_reports_failed_sources_without_buffers_across_processes() {
        // Process 1 holds the only buffer of a replica of an `f32` scalar with a pending maximum along `x`, which a
        // PJRT transfer manager has not filled yet, and places it on a device of process 0, which completes the maximum
        // with a host round trip. Process 1 takes no part and returns the metadata of the result, which must still
        // report that its source upload failed, although the source has no execution fence.
        let remote_client = process_one_cpu_client(2);
        let (local_devices, remote_devices) =
            (cpu_client(2).addressable_devices().unwrap(), remote_client.addressable_devices().unwrap());
        let local = local_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let remote = remote_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("replicas", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let source_mesh =
            DeviceMesh::new(logical_mesh.clone(), vec![local[0], remote[0], local[1], remote[1]]).unwrap();
        let pending = Sharding::replicated(logical_mesh, 0)
            .with_unreduced_axes(["x"])
            .unwrap()
            .with_unreduced_kind(UnreducedKind::Max)
            .unwrap();
        let target_mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![local[0]]).unwrap();
        let target_sharding = Sharding::replicated(target_mesh.logical_mesh().clone(), 0);
        let target = Placement::new(target_mesh, target_sharding.clone()).unwrap();
        let session = session_with_remote_devices(remote_client, local.clone());
        let specification = BufferSpecification::new(BufferType::F32, Vec::<u64>::new());
        let manager = remote_client
            .host_to_device_transfer_manager(vec![specification], remote_devices[0].default_memory().unwrap());
        let manager = manager.unwrap();
        let buffers = vec![manager.retrieve_buffer(0).unwrap()];
        let r#type = ArrayType::scalar(DataType::F32);
        let source = Array::from_addressable_buffers(&session, r#type, source_mesh, Some(pending), buffers).unwrap();
        let placed = source.to_placement(&session, target).unwrap();
        assert_eq!(placed.sharding(), &target_sharding);
        assert_eq!(placed.addressable_shards().count(), 0);
        manager.set_error(0, PjrtError::aborted("source upload failed")).unwrap();
        for array in [&source, &placed] {
            assert!(matches!(
                array.block_until_ready(),
                Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "source upload failed",
            ));
        }
    }

    #[test]
    fn test_array_to_placement_splits_boolean_values_into_unreduced_summands() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let r#type = ArrayType::new_static(DataType::Boolean, [2]);
        let value = Array::from_host_buffer(&session, r#type, mesh.clone(), Some(replicated), [1u8, 0]).unwrap();

        // Boolean summands combine with logical or, so the device away from coordinate zero along `x` holds `false`.
        let summands = value.to_placement(&session, Placement::new(mesh, unreduced.clone()).unwrap()).unwrap();
        assert_eq!(summands.sharding(), &unreduced);
        assert_eq!(
            summands.shards().iter().map(|shard| shard.copy_to_host_bytes().unwrap()).collect::<Vec<_>>(),
            vec![vec![1, 0], vec![0, 0]],
        );
        assert_eq!(summands.copy_to_host_bytes().unwrap(), vec![1, 0]);
    }

    #[test]
    fn test_array_to_placement_pads_short_shardings() {
        // A sharding shorter than the rank of the array leaves the omitted trailing dimensions replicated, when
        // borrowing or consuming the array.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let logical_mesh = mesh.logical_mesh().clone();
        let split = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let split_matrix = Sharding::new(
            logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let values = (0..12).map(|value| value as f32).collect::<Vec<_>>();
        let r#type = ArrayType::new_static(DataType::F32, [4, 3]);
        let replicated = Some(Sharding::replicated(logical_mesh, 2));
        let source = Array::from_host_buffer(&session, r#type, mesh.clone(), replicated, values_to_bytes(&values));
        let source = source.unwrap();
        let target = Placement::new(mesh, split).unwrap();
        let borrowed = source.to_placement(&session, target.clone()).unwrap();
        let consumed = source.into_placement(&session, target).unwrap();
        for output in [borrowed, consumed] {
            assert_eq!(output.sharding(), &split_matrix);
            let dimensions = output.shards().iter().map(|shard| shard.buffer().unwrap().dimensions().unwrap().to_vec());
            assert_eq!(dimensions.collect::<Vec<_>>(), vec![vec![2, 3], vec![2, 3]]);
            assert_eq!(shard_values(&output), vec![values[..6].to_vec(), values[6..].to_vec()]);
        }
    }

    #[test]
    fn test_array_to_placement_preserves_memory() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = four_device_mesh_x(client);
        let single_device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![mesh.devices()[0]],
        )
        .unwrap();
        let sub_mesh = two_device_mesh(client, MeshAxisType::Auto);
        let mesh_of_type = |axis_type| {
            let axis = MeshAxis::new("x", 4, axis_type).unwrap();
            DeviceMesh::new(LogicalMesh::new(vec![axis]).unwrap(), mesh.devices().to_vec()).unwrap()
        };
        let (explicit, manual_mesh) = (mesh_of_type(MeshAxisType::Explicit), mesh_of_type(MeshAxisType::Manual));
        let sharding = |mesh: &DeviceMesh, sharded: bool| match sharded {
            true => Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
            false => Sharding::replicated(mesh.logical_mesh().clone(), 1),
        };
        let values = [1f32, 2., 3., 4.];
        let replicas = vec![values.to_vec(); 4];
        let tiles = vec![vec![1f32], vec![2.], vec![3.], vec![4.]];

        // Each row places a source on a mesh, replicated or split along `x`, onto a mesh, replicated or split along
        // `x`, in every memory, with and without donating the source, and the result keeps the memory of the source.
        // Whole shards that move to other devices are copied to the same memory of those devices, while the shard
        // that stays on its device shares its buffer, and no program is compiled for them. The compiled reshard on
        // one mesh, with auto or explicit axes, declares its input and its output in the memory of the source. A
        // replicated source on another mesh is copied to the memory of the source on the target devices before it is
        // resharded there, and a split source on another mesh is first gathered on its own mesh, which also keeps
        // its memory. A manual placement uploads to the memory of the source.
        let cases = [
            ("whole shards", &single_device_mesh, false, &mesh, false, &replicas, true),
            ("compiled reshard to a split", &mesh, false, &mesh, true, &tiles, false),
            ("compiled reshard to replicas", &mesh, true, &mesh, false, &replicas, false),
            ("compiled reshard on explicit axes", &explicit, false, &explicit, true, &tiles, false),
            ("replicas on another mesh", &single_device_mesh, false, &mesh, true, &tiles, false),
            ("split on another mesh", &sub_mesh, true, &mesh, true, &tiles, false),
            ("split on another mesh to replicas", &sub_mesh, true, &mesh, false, &replicas, false),
            ("manual placement", &manual_mesh, false, &manual_mesh, true, &tiles, false),
        ];
        for (memory, kind) in [
            (Memory::Device, "device"),
            (Memory::Host { pinned: true }, "pinned_host"),
            (Memory::Host { pinned: false }, "unpinned_host"),
        ] {
            for (name, source_mesh, source_sharded, target_mesh, target_sharded, expected, compiles_nothing) in &cases {
                for donate in [false, true] {
                    let case = format!("{memory:?} {name}, donate={donate}");
                    let r#type = ArrayType::new_static(DataType::F32, [4]).with_memory(memory);
                    let source_sharding = Some(sharding(source_mesh, *source_sharded));
                    let bytes = values_to_bytes(&values);
                    let source =
                        Array::from_host_buffer(&session, r#type, (*source_mesh).clone(), source_sharding, bytes);
                    let source = source.unwrap();
                    let target =
                        Placement::new((*target_mesh).clone(), sharding(target_mesh, *target_sharded)).unwrap();
                    let programs = session.cached_program_count();
                    let placed = match donate {
                        true => source.into_placement(&session, target),
                        false => source.to_placement(&session, target),
                    };
                    let placed = placed.unwrap();
                    if *compiles_nothing {
                        assert_eq!(session.cached_program_count(), programs, "{case}");
                    }
                    assert_eq!(placed.r#type().memory(), memory, "{case}");
                    assert_eq!(shard_memory_kinds(&placed), vec![kind; 4], "{case}");
                    assert_eq!(&shard_values(&placed), *expected, "{case}");
                }
            }
        }
    }

    #[test]
    fn test_array_to_placement_changes_memory_and_layout() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 2);
        let sharded = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let row_major = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let memory = Memory::Host { pinned: true };
        let values = [1f32, 2., 3., 4., 5., 6., 7., 8.];
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4, 2]).with_layout(row_major.clone()),
            mesh.clone(),
            Some(replicated.clone()),
            values_to_bytes(&values),
        )
        .unwrap();

        // Changing just the memory on the same mesh keeps the source layout and needs no reshard program.
        let programs = session.cached_program_count();
        let copied = input
            .to_placement(&session, Placement::new(mesh.clone(), replicated).unwrap().with_memory(memory))
            .unwrap();
        assert_eq!(session.cached_program_count(), programs);
        assert_eq!(copied.r#type().memory(), memory);
        assert_eq!(copied.layout(), Some(&row_major));
        assert_eq!(shard_memory_kinds(&copied), ["pinned_host"; 2]);
        assert_eq!(copied.copy_to_host().unwrap().elements::<f32>().unwrap(), values);

        // The storage request also applies after a compiled reshard changes the slices.
        let placed = input
            .into_placement(
                &session,
                Placement::new(mesh.clone(), sharded.clone())
                    .unwrap()
                    .with_layout(column_major.clone())
                    .with_memory(memory),
            )
            .unwrap();
        assert_eq!(placed.sharding(), &sharded);
        assert_eq!(placed.r#type().memory(), memory);
        assert_eq!(placed.layout(), Some(&column_major));
        assert_eq!(shard_memory_kinds(&placed), ["pinned_host"; 2]);
        for shard in placed.addressable_shards() {
            assert_eq!(shard.buffer().unwrap().specification().unwrap().layout, column_major.to_pjrt().unwrap());
        }
        assert_eq!(placed.copy_to_host().unwrap().elements::<f32>().unwrap(), values);

        // Whole shards can move directly to the requested memory on other devices.
        let reversed =
            DeviceMesh::new(mesh.logical_mesh().clone(), mesh.devices().iter().rev().copied().collect()).unwrap();
        let moved = placed
            .to_placement(
                &session,
                Placement::new(reversed.clone(), sharded.clone()).unwrap().with_memory(Memory::Device),
            )
            .unwrap();
        assert_eq!(moved.mesh(), &reversed);
        assert_eq!(moved.r#type().memory(), Memory::Device);
        for shard in moved.addressable_shards() {
            let buffer = shard.buffer().unwrap();
            assert_eq!(buffer.memory().unwrap(), buffer.device().unwrap().default_memory().unwrap());
        }
        assert_eq!(moved.copy_to_host().unwrap().elements::<f32>().unwrap(), values);

        // When crossing meshes also changes the slices, the compiled reshard can use device intermediates.
        let unpinned = Memory::Host { pinned: false };
        let moved = copied
            .into_placement(&session, Placement::new(reversed.clone(), sharded).unwrap().with_memory(unpinned))
            .unwrap();
        assert_eq!(moved.mesh(), &reversed);
        assert_eq!(moved.r#type().memory(), unpinned);
        assert_eq!(shard_memory_kinds(&moved), ["unpinned_host"; 2]);
        assert_eq!(moved.copy_to_host().unwrap().elements::<f32>().unwrap(), values);
    }

    #[test]
    fn test_array_to_placement_changes_memory_fails_readback_of_failed_replica() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let other_mesh = domain_mesh(client, "y", 2);

        // A move to pinned host memory copies each replica on its device, as a change of storage on the same mesh and
        // as a copy of whole shards to another mesh of the same devices. Only the copy of the second replica fails,
        // and a readback reads one replica of each tile, so the result must report the failure through its fence.
        for target_mesh in [mesh.clone(), other_mesh] {
            let (source, managers) = pending_replicated_vector(&session, &mesh);
            let replicated = Sharding::replicated(target_mesh.logical_mesh().clone(), 1);
            let target = Placement::new(target_mesh, replicated).unwrap().with_memory(Memory::Host { pinned: true });
            let placed = source.to_placement(&session, target).unwrap();
            assert_eq!(shard_memory_kinds(&placed), ["pinned_host"; 2]);
            managers[0]
                .transfer_data(0, Arc::new(values_to_bytes(&[1f32, 2., 3., 4.])), 0, true)
                .unwrap()
                .r#await()
                .unwrap();
            managers[1].set_error(0, PjrtError::aborted("second replica failed")).unwrap();
            assert!(matches!(
                placed.block_until_ready(),
                Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "second replica failed",
            ));
            assert!(matches!(
                placed.copy_to_host_bytes(),
                Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "second replica failed",
            ));
            assert!(matches!(
                session.synchronize(),
                Err(Error::Execution(ExecutionError::SynchronizationFailures { failures }))
                    if matches!(&failures[..], [(_, Error::Pjrt(PjrtError::Aborted { message, .. }))]
                        if message == "second replica failed"),
            ));
        }
    }

    #[test]
    fn test_array_to_placement_compiled_reshard_across_meshes_fails_readback_of_failed_replica() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let other_mesh = domain_mesh(client, "y", 2);

        // The source is in pinned host memory in a declared column-major layout, which rules out shard regions, and its
        // second replica comes from a copy that fails. A compiled reshard to a split over another mesh of the same
        // devices copies each replica to the device memory of its own device and reads each copy on its device, so
        // the failed replica fails both its copy and the program that reads that copy. The fence of the result reports
        // the first failure that it observes, which is either of the two, so the readiness and the readback of the
        // result both fail with the cause of the failed replica, as the copy reports it or as the program reports it.
        // A failed replica never yields a result that is ready or a readback that succeeds.
        let failure = |result: Result<(), Error>| match result {
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) => Some(("aborted", message)),
            Err(Error::Pjrt(PjrtError::Internal { message, .. })) => Some(("internal", message)),
            _ => None,
        };
        let replica_failures = [
            Some(("aborted", "second replica failed".to_string())),
            Some(("internal", "Error dispatching computation: second replica failed".to_string())),
        ];
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let specification =
            BufferSpecification::with_layout(BufferType::F32, [2u64, 2], column_major.to_pjrt().unwrap());
        let managers = transfer_managers(client, specification);
        let buffers = managers.iter().map(|manager| manager.retrieve_buffer(0).unwrap()).collect();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 2);
        let r#type = ArrayType::new_static(DataType::F32, [2, 2]).with_layout(column_major);
        let source = Array::from_addressable_buffers(&session, r#type, mesh.clone(), Some(replicated.clone()), buffers);
        let pinned = Placement::new(mesh, replicated).unwrap().with_memory(Memory::Host { pinned: true });
        let source = source.unwrap().to_placement(&session, pinned).unwrap();
        let sharded = Sharding::new(
            other_mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["y"]), ShardingDimension::replicated()],
        )
        .unwrap();
        // The compiled reshard keeps the column-major layout of its input, and the result is requested in the row-major
        // layout, so a conversion on each device reads the result of the reshard.
        let row_major = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let target = Placement::new(other_mesh.clone(), sharded.clone())
            .unwrap()
            .with_memory(Memory::Device)
            .with_layout(row_major);
        let placed = source.to_placement(&session, target).unwrap();
        assert_eq!(placed.mesh(), &other_mesh);
        assert_eq!(placed.sharding(), &sharded);
        assert_eq!(shard_memory_kinds(&placed), ["device"; 2]);
        assert_eq!(session.cached_program_count(), 3);
        managers[0]
            .transfer_data(0, Arc::new(values_to_bytes(&[1f32, 2., 3., 4.])), 0, true)
            .unwrap()
            .r#await()
            .unwrap();
        managers[1].set_error(0, PjrtError::aborted("second replica failed")).unwrap();
        assert!(matches!(
            source.block_until_ready(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "second replica failed",
        ));
        let ready = failure(placed.block_until_ready());
        assert!(replica_failures.contains(&ready), "unexpected readiness of the result: {ready:?}");
        let readback = failure(placed.copy_to_host_bytes().map(|_| ()));
        assert!(replica_failures.contains(&readback), "unexpected readback of the result: {readback:?}");

        // The session also reports the failed replica, once for each failed submission in registration order. The
        // copies that read the failed replica report its cause as the copy reports it, the execution of the reshard
        // that reads those copies reports it as the failed execution reports it, and the conversions that read the
        // failed result of the reshard wrap that report once more.
        let Err(Error::Execution(ExecutionError::SynchronizationFailures { failures })) = session.synchronize() else {
            panic!("expected the session to report the failed replica");
        };
        let failures = failures
            .into_iter()
            .map(|(identity, error)| (identity, failure(Err(error)).unwrap()))
            .collect::<Vec<_>>();
        let message = "Error dispatching computation: Error dispatching computation: second replica failed";
        let converted_failure = ("internal", message.to_string());
        assert_eq!(
            failures,
            [
                (0, replica_failures[0].clone().unwrap()),
                (1, replica_failures[0].clone().unwrap()),
                (4, replica_failures[1].clone().unwrap()),
                (5, converted_failure.clone()),
                (7, converted_failure),
            ],
        );
    }

    #[test]
    fn test_array_to_placement_relayouts_addressable_unreduced_shards() {
        let client = cpu_client(2);
        let devices = client.addressable_devices().unwrap();
        let mut mesh_devices = devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let remote_device =
            Device::new(mesh_devices.iter().map(Device::id).max().unwrap() + 1, client.process_index().unwrap() + 1);
        mesh_devices.push(remote_device);
        let session = session_with_remote_devices(client, [remote_device]);
        let shared_session = peer_exchange_session(client, [remote_device]);
        let logical_mesh =
            LogicalMesh::new(vec![MeshAxis::new("x", mesh_devices.len(), MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), mesh_devices).unwrap();
        let sharding = Sharding::replicated(logical_mesh, 2).with_unreduced_axes(["x"]).unwrap();
        let summands = [[1f32, 2., 3., 4.], [10f32, 20., 30., 40.]];
        let buffers = summands
            .iter()
            .zip(devices)
            .map(|(values, device)| {
                client
                    .buffer(values_to_bytes(values).as_slice(), BufferType::F32, [2, 2], None, device, None)
                    .unwrap()
            })
            .collect();
        let (producer_event, producer) = client.event(()).unwrap();
        let (send_event, send) = client.event(()).unwrap();
        let source = Array::from_addressable_buffers(
            &session,
            ArrayType::new_static(DataType::F32, [2, 2]),
            mesh.clone(),
            Some(sharding.clone()),
            buffers,
        )
        .unwrap()
        .with_execution_fence(ExecutionFence::new(vec![producer_event]))
        .with_send_fence(ExecutionFence::new(vec![send_event]));
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let target = Placement::new(mesh, sharding.clone()).unwrap().with_layout(layout.clone());
        let placed = source.to_placement(&session, target.clone()).unwrap();
        // A session that shares compilation artifacts leads the conversions of the shards of its own process, so it
        // converts them on their devices as well.
        let shared = source.to_placement(&shared_session, target).unwrap();
        assert_eq!(session.cached_program_count(), 2);
        assert_eq!(shared_session.cached_program_count(), 2);

        for output in [&placed, &shared] {
            assert_eq!(output.sharding(), &sharding);
            assert_eq!(output.layout(), Some(&layout));
            assert!(output.device_shard(remote_device.id()).unwrap().buffer().is_none());
            assert!(!output.execution_fence().unwrap().is_ready().unwrap());
            for (shard, summand) in output.addressable_shards().zip(summands) {
                assert_eq!(shard.buffer().unwrap().specification().unwrap().layout, layout.to_pjrt().unwrap());
                assert_eq!(values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()), summand);
            }
        }

        // Every placement path retains the pending producer. The sends of the source belong to the placement that
        // produced it, so a later conversion of its layout does not wait for them or report them.
        producer.set(None).unwrap();
        send.set(Some(PjrtError::aborted("source transfer failed"))).unwrap();
        assert_eq!(source.block_until_ready().unwrap_err().to_string(), "source transfer failed");
        for output in [placed, shared] {
            output.block_until_ready().unwrap();
        }
    }

    #[test]
    fn test_array_to_placement_zero_space_is_bufferless_and_preserves_type_metadata() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = four_device_mesh_x(client);
        let source_sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let layout = Layout::Tiled(TiledLayout::new(vec![0], Vec::new()));
        let source_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(4)]))
            .with_layout(layout.clone())
            .with_memory(Memory::Host { pinned: true })
            .with_sharding(source_sharding)
            .unwrap();
        let source = Array::from_host_buffer(&session, source_type, mesh.clone(), None, []).unwrap();
        let target_sharding =
            Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let resharded = source
            .to_placement(&session, Placement::new(mesh.clone(), target_sharding.clone()).unwrap())
            .unwrap();

        assert_eq!(resharded.sharding(), &target_sharding);
        assert_eq!(resharded.layout(), Some(&layout));
        assert_eq!(resharded.r#type().memory(), Memory::Host { pinned: true });
        assert_eq!(resharded.addressable_shards().count(), mesh.devices().len());
        assert!(resharded.addressable_shards().all(|shard| shard.buffer().is_none()));
    }

    #[test]
    fn test_array_to_placement_zero_space_rejects_invalid_layout() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let source =
            Array::from_host_buffer(&session, ArrayType::new_static(DataType::Zero, [4]), mesh.clone(), None, [])
                .unwrap();
        let layout = Layout::Tiled(TiledLayout::new(vec![1, 0], Vec::new()));
        let message = "`place` target layout is invalid: tiled layout rank 2 does not match array rank 1";

        // A `Zero` array has no buffers whose layout could be checked, so the layout is validated against its shape,
        // both when the placement keeps its devices and sharding and when it changes them.
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        for sharding in [source.sharding().clone(), sharded] {
            let target = Placement::new(mesh.clone(), sharding).unwrap().with_layout(layout.clone());
            let error = source.to_placement(&session, target).unwrap_err();
            assert!(matches!(
                &error,
                Error::Core(error) if **error == CoreError::Type(TypeError::Invalid { message: message.to_string() }),
            ));
            assert_eq!(error.to_string(), message);
        }

        // A layout of the right rank is recorded on the bufferless result.
        let layout = Layout::Tiled(TiledLayout::new(vec![0], Vec::new()));
        let target = Placement::new(mesh.clone(), source.sharding().clone()).unwrap().with_layout(layout.clone());
        let placed = source.to_placement(&session, target).unwrap();
        assert_eq!(placed.layout(), Some(&layout));
        assert!(placed.addressable_shards().all(|shard| shard.buffer().is_none()));
    }

    #[test]
    fn test_array_to_placement_validates_layout_of_each_shard() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let columns = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
        )
        .unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [4, 4]);
        let values = (0..16).map(|value| value as f32).collect::<Vec<_>>();
        let source = Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(&values)).unwrap();
        let strided = |strides: Vec<isize>| Layout::Strided(StridedLayout::new(strides));

        // A target layout describes each target shard, so the strides `[8, 4]` of a dense `f32[4, 2]` column
        // shard pass shape validation for the column shards of a global `[4, 4]` value, which they would overlap.
        // The CPU plugin cannot create buffers in that layout, so the placement rejects it before device work.
        let target = Placement::new(mesh.clone(), columns.clone()).unwrap().with_layout(strided(vec![8, 4]));
        assert!(matches!(
            source.to_placement(&session, target),
            Err(Error::Pjrt(PjrtError::InvalidArgument { message, .. }))
                if message == "cpu PJRT does not support strided device layouts",
        ));

        // The same strides overlap the rows of a replicated shard, and strides that overlap a column shard are
        // rejected.
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 2);
        for (sharding, strides, axis) in [(replicated, vec![8, 4], 0), (columns, vec![4, 8], 1)] {
            let target = Placement::new(mesh.clone(), sharding).unwrap().with_layout(strided(strides));
            let error = source.to_placement(&session, target).unwrap_err();
            assert_eq!(
                error.to_string(),
                format!(
                    "`{PLACE_OPERATION_NAME}` target layout is invalid: strided layout stride 8 on axis {axis} is \
                     smaller than the 16-byte span occupied by more minor axes and may alias array elements",
                ),
            );
        }
    }

    #[test]
    fn test_array_to_placement_zero_space_preserves_producer_completion() {
        let client = crate::tests::execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let device = client.addressable_devices().unwrap().remove(0);
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::from_pjrt(&device).unwrap()],
        )
        .unwrap();
        let target_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("target", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            source_mesh.devices().to_vec(),
        )
        .unwrap();
        let target_sharding = Sharding::replicated(target_mesh.logical_mesh().clone(), 1);
        let target = Placement::new(target_mesh.clone(), target_sharding.clone()).unwrap();

        for fails in [false, true] {
            let (event, promise) = client.event(()).unwrap();
            let source = Array::from_host_buffer(
                &session,
                ArrayType::new_static(DataType::Zero, [2]),
                source_mesh.clone(),
                None,
                [],
            )
            .unwrap()
            .with_execution_fence(ExecutionFence::new(vec![event]));
            let outputs = [
                source.to_placement(&session, target.clone()).unwrap(),
                source.clone().into_placement(&session, target.clone()).unwrap(),
                source.clone().reshard_on_devices(&session, &target_mesh, &target_sharding, false).unwrap(),
            ];
            // A move that only changes the representation must neither wait for nor discard the producer's completion.
            // Complete the event before asserting so that even a failing regression releases its pending native
            // callbacks.
            let pending = outputs
                .iter()
                .map(|output| matches!(output.execution_fence().map(ExecutionFence::is_ready), Some(Ok(false))))
                .collect::<Vec<_>>();
            promise.set(fails.then(|| PjrtError::invalid_argument("zero producer failed"))).unwrap();
            assert_eq!(pending, vec![true; 3]);
            for output in outputs {
                let result = output.block_until_ready();
                if fails {
                    assert!(matches!(
                        result,
                        Err(Error::Pjrt(PjrtError::InvalidArgument { message, .. }))
                            if message == "zero producer failed",
                    ));
                } else {
                    assert_eq!(result.map_err(|error| error.to_string()), Ok(()));
                }
            }
        }
    }

    #[test]
    fn test_array_to_placement_compiled_reshard_on_same_mesh_preserves_producer_failure() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let target = Placement::new(mesh.clone(), sharded.clone()).unwrap();

        // Both replicas of the source are readable, but the execution that produced it fails afterwards (e.g., a copy
        // of a replica that an earlier placement made and joined into its fence). The compiled reshard reads only the
        // readable buffers, so its result must carry the fence of the source across the launch, with and without
        // donating the source.
        for donate in [false, true] {
            let (producer_event, producer) = client.event(()).unwrap();
            let source = Array::from_host_buffer(
                &session,
                ArrayType::new_static(DataType::F32, [4]),
                mesh.clone(),
                Some(replicated.clone()),
                values_to_bytes(&[1f32, 2., 3., 4.]),
            )
            .unwrap()
            .with_execution_fence(ExecutionFence::new(vec![producer_event]));
            source
                .addressable_shards()
                .for_each(|shard| shard.buffer().unwrap().ready().unwrap().r#await().unwrap());
            let programs = session.cached_program_count();
            let placed = if donate {
                source.into_placement(&session, target.clone()).unwrap()
            } else {
                source.to_placement(&session, target.clone()).unwrap()
            };
            assert_eq!(session.cached_program_count(), programs + 1);
            assert_eq!(placed.mesh(), &mesh);
            assert_eq!(placed.sharding(), &sharded);
            assert_eq!(
                placed
                    .addressable_shards()
                    .map(|shard| shard.buffer().unwrap().dimensions().unwrap().to_vec())
                    .collect::<Vec<_>>(),
                [[2], [2]],
            );
            assert_eq!(shard_values(&placed), vec![vec![1., 2.], vec![3., 4.]]);
            assert!(matches!(placed.execution_fence().map(ExecutionFence::is_ready), Some(Ok(false))));
            producer.set(Some(PjrtError::aborted("replica copy failed"))).unwrap();
            assert!(matches!(
                placed.block_until_ready(),
                Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "replica copy failed",
            ));
            assert!(matches!(
                placed.copy_to_host_bytes(),
                Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "replica copy failed",
            ));
        }
    }

    #[test]
    fn test_array_to_placement_compiled_reshard_sharded_to_differently_sharded_on_same_mesh() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(logical_mesh_2x2(), devices).unwrap();

        let row_values = [
            100.0f32, 101.0, 102.0, 103.0, 110.0, 111.0, 112.0, 113.0, 120.0, 121.0, 122.0, 123.0, 130.0, 131.0, 132.0,
            133.0,
        ];

        let sharded_along_x = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(4)]));
        let source_array = Array::from_host_buffer(
            &session,
            source_type,
            mesh.clone(),
            Some(sharded_along_x),
            values_to_bytes::<f32>(&row_values).as_slice(),
        )
        .unwrap();

        let sharded_along_y = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["y"])],
        )
        .unwrap();
        let resharded =
            source_array.to_placement(&session, Placement::new(mesh.clone(), sharded_along_y).unwrap()).unwrap();
        assert_eq!(resharded.addressable_shards().count(), 4);

        // Every device holds two whole columns, and the devices along `x` hold copies of the same columns.
        assert_eq!(
            shard_values(&resharded),
            vec![
                vec![100.0, 101.0, 110.0, 111.0, 120.0, 121.0, 130.0, 131.0],
                vec![102.0, 103.0, 112.0, 113.0, 122.0, 123.0, 132.0, 133.0],
                vec![100.0, 101.0, 110.0, 111.0, 120.0, 121.0, 130.0, 131.0],
                vec![102.0, 103.0, 112.0, 113.0, 122.0, 123.0, 132.0, 133.0],
            ],
        );
    }

    #[test]
    fn test_array_to_placement_compiled_reshard_shares_artifact_without_process_zero() {
        // Compilation artifact exchange of process 1 of three that records its calls and never receives an artifact.
        struct RecordingExchange(Mutex<Vec<String>>);

        impl CompilationArtifactExchange for RecordingExchange {
            fn process_index(&self) -> usize {
                1
            }

            fn process_count(&self) -> usize {
                3
            }

            fn validate_compatibility(
                &self,
                _key: &[u8],
                leader: usize,
                _timeout: Duration,
            ) -> Result<(), CompilationArtifactExchangeError> {
                self.0.lock().unwrap().push(format!("validate with leader {leader}"));
                Ok(())
            }

            fn publish(&self, _key: &[u8], _artifact: &[u8]) -> Result<(), CompilationArtifactExchangeError> {
                self.0.lock().unwrap().push("publish".to_string());
                Ok(())
            }

            fn receive(&self, _: &[u8], _: Duration) -> Result<Option<Vec<u8>>, CompilationArtifactExchangeError> {
                self.0.lock().unwrap().push("receive".to_string());
                Ok(None)
            }
        }

        // This client is process 1 of a job of three processes, and both of its devices form the mesh of the source
        // and of the target, so it runs the compiled reshard, as process 2 would on a mesh that spans the devices
        // of both, while process 0 owns no device of that mesh and compiles nothing. Process 1 is therefore the leader
        // of the compilation, which compiles and publishes the program rather than waiting for process 0.
        let client = process_one_cpu_client(2);
        let exchange = Arc::new(RecordingExchange(Mutex::new(Vec::new())));
        let cache = CompilationCache::new().with_artifact_exchange(
            Arc::clone(&exchange) as Arc<dyn CompilationArtifactExchange>,
            CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_millis(10) },
        );
        let session = Session::builder().client(client).compilation_cache(cache).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        assert_eq!(mesh.devices().iter().map(Device::process_index).collect::<Vec<_>>(), [1, 1]);
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let source =
            Array::from_host_buffer(&session, r#type, mesh.clone(), None, values_to_bytes(&[1f32, 2., 3., 4.]));
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let placed = source.unwrap().to_placement(&session, Placement::new(mesh, sharded).unwrap()).unwrap();
        assert_eq!(shard_values(&placed), [vec![1., 2.], vec![3., 4.]]);
        assert_eq!(*exchange.0.lock().unwrap(), ["validate with leader 1", "publish"]);
    }

    #[test]
    fn test_array_to_placement_compiled_reshard_caches_executable_across_calls() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = four_device_mesh_x(client);
        let replicated = Some(Sharding::replicated(mesh.logical_mesh().clone(), 1));
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let values = values_to_bytes(&[30f32, 31., 32., 33.]);
        let source = Array::from_host_buffer(&session, r#type, mesh.clone(), replicated, values).unwrap();
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let target = Placement::new(mesh, sharded).unwrap();

        assert_eq!(session.cached_program_count(), 0);
        let first = source.to_placement(&session, target.clone()).unwrap();
        assert_eq!(session.cached_program_count(), 1);
        let cold = session.compilation_cache().statistics();
        assert_eq!((cold.misses, cold.compilations, cold.memory_hits), (1, 1, 0));
        let second = source.to_placement(&session, target).unwrap();
        assert_eq!(session.cached_program_count(), 1);
        // The second reshard reaches the executable cache with lowered IR and reuses the first PJRT executable.
        let warm = session.compilation_cache().statistics();
        assert_eq!((warm.misses, warm.compilations, warm.memory_hits), (1, 1, 1));

        assert_eq!(shard_values(&first), vec![vec![30.], vec![31.], vec![32.], vec![33.]]);
        assert_eq!(shard_values(&second), vec![vec![30.], vec![31.], vec![32.], vec![33.]]);
    }

    #[test]
    fn test_array_to_placement_preserves_missing_cross_host_extension() {
        let client = cpu_client(1);
        let local = first_device(client);
        let remote = Device::new(local.id() + 1, 1);
        let session = session_with_remote_devices(client, [remote]);

        // The source array is on the local device, replicated on a 1-device submesh.
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![local],
        )
        .unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let sharding = Some(Sharding::replicated(source_mesh.logical_mesh().clone(), 1));
        let values = values_to_bytes(&[100f32, 101., 102., 103.]);
        let source = Array::from_host_buffer(&session, r#type, source_mesh, sharding, values).unwrap();

        // Sending to the remote target requires the extension even though local copies are possible.
        assert!(matches!(client.cross_host_transfers_extension(), Err(PjrtError::Unimplemented { .. })));
        let target_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![local, remote],
        )
        .unwrap();
        let target_sharding = Sharding::replicated(target_mesh.logical_mesh().clone(), 1);
        assert!(matches!(
            source.to_placement(&session, Placement::new(target_mesh.clone(), target_sharding).unwrap()),
            Err(Error::Pjrt(PjrtError::Unimplemented { message, .. }))
                if message == "the cross-host transfers extension is not provided by the PJRT plugin",
        ));

        // Moving a proper region also requires the extension, even when the source device can slice it locally.
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert!(matches!(
            source.to_placement(&session, Placement::new(target_mesh, target_sharding).unwrap()),
            Err(Error::Pjrt(PjrtError::Unimplemented { message, .. }))
                if message == "the cross-host transfers extension is not provided by the PJRT plugin",
        ));
    }

    #[test]
    fn test_array_to_placement_preserves_missing_cross_host_extension_for_receiver() {
        let client = cpu_client(1);
        assert!(matches!(client.cross_host_transfers_extension(), Err(PjrtError::Unimplemented { .. })));
        let local_device = first_device(client);
        let remote_device = Device::new(local_device.id() + 1, local_device.process_index() + 1);
        let session = session_with_remote_devices(client, [remote_device]);
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![remote_device],
        )
        .unwrap();
        let source_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(source_mesh.logical_mesh().clone(), 1))
            .unwrap();
        let source = Array::from_addressable_buffers(&session, source_type, source_mesh, None, Vec::new()).unwrap();
        assert_eq!(source.addressable_shards().count(), 0);
        let target_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("target", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![local_device],
        )
        .unwrap();
        let target_sharding = Sharding::replicated(target_mesh.logical_mesh().clone(), 1);

        // This process only receives. Every target is local, so there is no remote target to blame for the
        // missing extension. Preserve that error instead of searching for a nonexistent remote target and
        // panicking.
        assert!(matches!(
            source.to_placement(&session, Placement::new(target_mesh, target_sharding).unwrap()),
            Err(Error::Pjrt(PjrtError::Unimplemented { message, .. }))
                if message == "the cross-host transfers extension is not provided by the PJRT plugin",
        ));
    }

    #[test]
    fn test_array_to_placement_places_uncommitted_value_on_devices_of_this_process() {
        let client = cpu_client(2);
        assert!(matches!(client.cross_host_transfers_extension(), Err(PjrtError::Unimplemented { .. })));
        let local_devices = Session::builder().client(client).build().unwrap().addressable_devices().unwrap();
        let remote_device =
            Device::new(local_devices.iter().map(Device::id).max().unwrap() + 1, client.process_index().unwrap() + 1);
        let session = session_with_remote_devices(client, [remote_device]);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 3, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh =
            DeviceMesh::new(logical_mesh.clone(), vec![local_devices[0], local_devices[1], remote_device]).unwrap();
        let values = [1f32, 2., 3., 4., 5., 6.];

        // Each process uploads the same value without a target, so its array lives uncommitted on its own first
        // device. A remote process would have chosen its own device, so the source mesh is not shared, and each
        // process places its own value on its own target devices without planning any cross-host transfer. Both
        // the borrowing and the consuming placement do so, as does the implicit reshard of a compiled input.
        let replicated = Sharding::replicated(logical_mesh.clone(), 1);
        let sharded = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        for (sharding, expected) in [
            (&replicated, vec![values.to_vec(), values.to_vec()]),
            (&sharded, vec![vec![1., 2.], vec![3., 4.]]),
            (&unreduced, vec![values.to_vec(), vec![-0.; 6]]),
        ] {
            let source = Array::from_host_buffer(
                &session,
                ArrayType::new_static(DataType::F32, [6]),
                None,
                None,
                values_to_bytes(&values),
            )
            .unwrap();
            assert!(!source.is_committed());
            assert_eq!(source.mesh().devices(), [local_devices[0]]);
            let target = Placement::new(mesh.clone(), sharding.clone()).unwrap();
            let outputs = [
                source.to_placement(&session, target.clone()).unwrap(),
                source.clone().into_placement(&session, target).unwrap(),
                source.clone().reshard_on_devices(&session, &mesh, sharding, false).unwrap(),
            ];
            for output in outputs {
                assert_eq!(output.mesh(), &mesh);
                assert_eq!(output.sharding(), sharding);
                assert!(output.is_committed());
                assert_eq!(output.r#type().memory(), Memory::Device);
                assert!(output.device_shard(remote_device.id()).unwrap().buffer().is_none());
                assert_eq!(output.block_until_ready().map_err(|error| error.to_string()), Ok(()));
                assert_eq!(
                    output
                        .addressable_shards()
                        .map(|shard| shard.buffer().unwrap().dimensions().unwrap().to_vec())
                        .collect::<Vec<_>>(),
                    expected.iter().map(|values| vec![values.len() as u64]).collect::<Vec<_>>(),
                );
                let shards = output
                    .addressable_shards()
                    .map(|shard| values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()))
                    .collect::<Vec<_>>();
                assert_eq!(
                    shards.iter().flatten().map(|value| value.to_bits()).collect::<Vec<_>>(),
                    expected.iter().flatten().map(|value| value.to_bits()).collect::<Vec<_>>(),
                );
            }
        }

        // The value placed into a memory of its device is committed to it, but its device is still one that this
        // process chose on its own, so it is placed with local work only as well. The same array on a device that every
        // process knows plans a transfer to the remote device, which needs the cross-host transfers extension.
        let uploaded = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [6]),
            None,
            None,
            values_to_bytes(&values),
        )
        .unwrap();
        let target = Placement::new(mesh.clone(), replicated.clone()).unwrap();
        let in_memory = uploaded.to_placement(&session, Memory::Device).unwrap();
        assert!(in_memory.is_committed());
        let placed = in_memory.to_placement(&session, target.clone()).unwrap();
        let placed_values = placed
            .addressable_shards()
            .map(|shard| values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(placed_values, [values.to_vec(), values.to_vec()]);
        let shared = uploaded.with_locally_assigned(false);
        assert!(matches!(
            shared.to_placement(&session, target),
            Err(Error::Pjrt(PjrtError::Unimplemented { message, .. }))
                if message == "the cross-host transfers extension is not provided by the PJRT plugin",
        ));

        // The upload along an unreduced target axis uses the layout that the placement records, which is the
        // requested one here, so it lands in that layout directly, rather than in the default layout followed by a
        // compiled conversion of every shard, and every process records that layout whichever strategy it takes.
        let session = session_with_remote_devices(client, [remote_device]);
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let r#type = ArrayType::new_static(DataType::F32, [2, 3]);
        let uploaded = Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(&values)).unwrap();
        let unreduced = Sharding::replicated(logical_mesh, 2).with_unreduced_axes(["x"]).unwrap();
        let target = Placement::new(mesh.clone(), unreduced).unwrap().with_layout(column_major.clone());
        let placed = uploaded.to_placement(&session, target).unwrap();
        assert_eq!(placed.layout(), Some(&column_major));
        assert_eq!(placed.block_until_ready().map_err(|error| error.to_string()), Ok(()));
        assert_eq!(
            placed
                .addressable_shards()
                .map(|shard| shard.buffer().unwrap().specification().unwrap().layout)
                .collect::<Vec<_>>(),
            [column_major.to_pjrt().unwrap(), column_major.to_pjrt().unwrap()],
        );
        let shards = placed
            .addressable_shards()
            .flat_map(|shard| values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()))
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(shards, values.iter().chain(&[-0f32; 6]).map(|value| value.to_bits()).collect::<Vec<_>>());
        assert_eq!(session.cached_program_count(), 0);
    }

    #[test]
    fn test_array_to_placement_places_pending_uncommitted_value_on_remote_devices() {
        let client = cpu_client(1);
        let local_device = first_device(client);
        let remote_device = Device::new(local_device.id() + 1, local_device.process_index() + 1);
        let session = session_with_remote_devices(client, [remote_device]);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![remote_device]).unwrap();
        let unreduced = Sharding::replicated(logical_mesh, 1).with_unreduced_axes(["x"]).unwrap();
        let target = Placement::new(mesh.clone(), unreduced.clone()).unwrap();
        let pending_source = || {
            let (event, producer) = client.event(()).unwrap();
            let r#type = ArrayType::new_static(DataType::F32, [2]);
            let source = Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(&[1f32, 2.]))
                .unwrap()
                .with_execution_fence(ExecutionFence::new(vec![event]));
            assert!(!source.is_committed());
            (source, producer)
        };

        // The value of this process goes through a host upload along the unreduced target axis, and no
        // target device belongs to this process. The result still waits for and reports the producer of the
        // source, which the uploads of the processes that hold the target devices wait for.
        let (source, producer) = pending_source();
        let placed = source.to_placement(&session, target.clone()).unwrap();
        assert_eq!(placed.mesh(), &mesh);
        assert_eq!(placed.sharding(), &unreduced);
        assert!(placed.is_committed());
        assert_eq!(placed.addressable_shards().count(), 0);
        assert!(placed.shards()[0].buffer().is_none());
        assert!(matches!(placed.execution_fence().map(ExecutionFence::is_ready), Some(Ok(false))));
        producer.set(None).unwrap();
        assert_eq!(placed.block_until_ready().map_err(|error| error.to_string()), Ok(()));

        let (source, producer) = pending_source();
        let placed = source.to_placement(&session, target).unwrap();
        producer.set(Some(PjrtError::aborted("producer failed"))).unwrap();
        assert!(matches!(
            placed.block_until_ready(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "producer failed",
        ));

        // Along a replicated target, the value of this process is shared or copied on devices, and no target
        // device belongs to this process. A source whose only completion is the readiness of its pending buffer (e.g.,
        // a host upload) has no execution fence, so the result waits for and reports that buffer.
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let target = Placement::new(mesh.clone(), replicated.clone()).unwrap();
        let manager = transfer_managers(client, BufferSpecification::new(BufferType::F32, [2u64])).remove(0);
        let source_mesh = domain_mesh(client, "x", 1);
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let buffers = vec![manager.retrieve_buffer(0).unwrap()];
        let source = Array::from_addressable_buffers(&session, r#type, source_mesh, None, buffers).unwrap();
        let source = source.with_locally_assigned(true);
        assert!(source.execution_fence().is_none());
        let placed = source.to_placement(&session, target).unwrap();
        assert_eq!(placed.sharding(), &replicated);
        assert_eq!(placed.addressable_shards().count(), 0);
        manager.set_error(0, PjrtError::aborted("upload failed")).unwrap();
        assert!(matches!(
            placed.block_until_ready(),
            Err(Error::Pjrt(PjrtError::Aborted { message, .. })) if message == "upload failed",
        ));
    }

    #[test]
    fn test_array_to_placement_rejects_misattributed_target_devices() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let local_device = first_device(client);
        let (device_id, process_index) = (local_device.id(), local_device.process_index());
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let misattributed = DeviceMesh::new(logical_mesh.clone(), vec![Device::new(device_id, process_index + 1)]);
        let target = Placement::new(misattributed.unwrap(), Sharding::replicated(logical_mesh.clone(), 0)).unwrap();
        let bytes = values_to_bytes(&[7f32]);
        let source = Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, bytes).unwrap();

        // The target assigns the device of this process to another process, so every process would leave its
        // shard to another one. Both placements reject it before any device work, and the source stays readable.
        let message =
            format!("device {device_id} reports process index {process_index}, but expected {}", process_index + 1);
        for error in [
            source.to_placement(&session, target.clone()).unwrap_err(),
            source.clone().into_placement(&session, target).unwrap_err(),
        ] {
            assert!(matches!(
                &error,
                Error::Array(error) if matches!(
                    error.as_ref(),
                    ArrayError::DeviceProcessIndexMismatch { device_id: id, expected, actual }
                        if *id == device_id && *expected == process_index + 1 && *actual == process_index,
                ),
            ));
            assert_eq!(error.to_string(), message);
        }
        assert_eq!(values_from_bytes::<f32>(&source.copy_to_host_bytes().unwrap()), [7.]);

        // A target device of this process that the client does not address would receive no buffer either, and
        // both placements reject it as an import or an upload onto that mesh does.
        let missing = DeviceMesh::new(logical_mesh.clone(), vec![Device::new(device_id + 999, process_index)]);
        let target = Placement::new(missing.unwrap(), Sharding::replicated(logical_mesh, 0)).unwrap();
        let message =
            format!("device {} is not addressable by the PJRT client in process {process_index}", device_id + 999);
        for error in [
            source.to_placement(&session, target.clone()).unwrap_err(),
            source.clone().into_placement(&session, target).unwrap_err(),
        ] {
            assert!(matches!(
                &error,
                Error::Array(error) if matches!(
                    error.as_ref(),
                    ArrayError::NonAddressableDevice { device_id: id, process_index: index }
                        if *id == device_id + 999 && *index == process_index,
                ),
            ));
            assert_eq!(error.to_string(), message);
        }
        session.synchronize().unwrap();
    }

    #[test]
    fn test_array_to_placement_with_manual_mesh_axes_uses_shard_regions() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let manual_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();

        let values = [40.0f32, 41.0, 42.0, 43.0];
        let replicated_sharding = Sharding::replicated(manual_mesh.logical_mesh().clone(), 1);
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(values.len())]))
            .with_sharding(replicated_sharding)
            .unwrap();
        let source_array = Array::from_host_buffer(
            &session,
            source_type,
            manual_mesh.clone(),
            None,
            values_to_bytes::<f32>(&values).as_slice(),
        )
        .unwrap();

        // Each device can slice its own replica even though the SPMD partitioner cannot plan manual mesh axes.
        let sharded_target =
            Sharding::new(manual_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let resharded = source_array
            .to_placement(&session, Placement::new(manual_mesh.clone(), sharded_target.clone()).unwrap())
            .unwrap();
        assert_eq!(resharded.sharding(), &sharded_target);
        assert_eq!(resharded.addressable_shards().count(), 4);

        // Each device-local shard now owns one element of the input vector.
        assert_eq!(shard_values(&resharded), vec![vec![40.0], vec![41.0], vec![42.0], vec![43.0]]);

        assert_ne!(session.eager_dispatch_statistics().misses, 0);
        let on_devices = source_array.reshard_on_devices(&session, &manual_mesh, &sharded_target, false).unwrap();
        assert_eq!(shard_values(&on_devices), shard_values(&resharded));
    }

    #[test]
    fn test_array_to_placement_host_fallback_returns_before_readback() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Manual);
        let values = [0f32, 1., 2., 3., 4., 5., 6., 7.];
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 2);
        let r#type = ArrayType::new_static(DataType::F32, [2, 4]);
        let (producer_event, producer) = client.event(()).unwrap();
        let source =
            Array::from_host_buffer(&session, r#type, mesh.clone(), Some(replicated), values_to_bytes(&values))
                .unwrap()
                .with_execution_fence(ExecutionFence::new(vec![producer_event]));

        // A target with unreduced axes on a manual mesh has no device strategy, because shard regions need one
        // value per slice and a compiled reshard needs a mesh without manual axes, so the host fallback places it. The
        // readback waits for the pending producer of the source, so the placement can only return because the readback
        // and the upload continue in the background.
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 2).with_unreduced_axes(["x"]).unwrap();
        let memory = Memory::Host { pinned: true };
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let target = Placement::new(mesh.clone(), sharding.clone())
            .unwrap()
            .with_memory(memory)
            .with_layout(layout.clone());
        let placed = source.to_placement(&session, target).unwrap();
        assert_eq!(placed.data_type(), DataType::F32);
        assert_eq!(placed.shape(), StaticShape::new(vec![2, 4]));
        assert_eq!(placed.mesh(), &mesh);
        assert_eq!(placed.sharding(), &sharding);
        assert_eq!(placed.r#type().memory(), memory);
        assert_eq!(placed.layout(), Some(&layout));
        assert!(matches!(placed.execution_fence().map(ExecutionFence::is_ready), Some(Ok(false))));
        assert_eq!(shard_memory_kinds(&placed), ["pinned_host"; 2]);
        for shard in placed.addressable_shards() {
            let buffer = shard.buffer().unwrap();
            assert_eq!(buffer.specification().unwrap().layout, layout.to_pjrt().unwrap());
            assert_eq!(buffer.ready().unwrap().ready(), Ok(false));
        }

        // The device at coordinate zero along the unreduced axis holds the whole value, and the other one holds the
        // additive identity.
        producer.set(None).unwrap();
        placed.block_until_ready().unwrap();
        assert_eq!(shard_values(&placed), vec![values.to_vec(), vec![0.; 8]]);
        assert_eq!(placed.copy_to_host().unwrap().elements::<f32>().unwrap(), values);
        session.synchronize().unwrap();
    }

    #[test]
    fn test_array_to_placement_host_fallback_poisons_result_on_readback_failure() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Manual);
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let (producer_event, producer) = client.event(()).unwrap();
        let source = Array::from_host_buffer(
            &session,
            r#type,
            mesh.clone(),
            Some(replicated),
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap()
        .with_execution_fence(ExecutionFence::new(vec![producer_event]));
        // A target with unreduced axes on a manual mesh has no device strategy (refer to
        // `test_array_to_placement_host_fallback_returns_before_readback`), so the host fallback places it.
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let placed = source.to_placement(&session, Placement::new(mesh, sharding).unwrap()).unwrap();

        // The failure of the producer fails the readback, which poisons every target buffer with an error that
        // also carries the failure of the device work.
        producer.set(Some(PjrtError::aborted("producer failed"))).unwrap();
        let Err(Error::Pjrt(PjrtError::Internal { message, .. })) = placed.block_until_ready() else {
            panic!("the result of a failed host fallback must be poisoned");
        };
        assert_eq!(
            message,
            "the host fallback of an array placement failed with: producer failed (the device work that it replaced \
             failed with: compiled reshard does not support `Manual` mesh axis x)",
        );
        for shard in placed.addressable_shards() {
            assert!(matches!(
                shard.buffer().unwrap().ready().unwrap().r#await(),
                Err(PjrtError::Internal { message: shard_message, .. }) if shard_message == message,
            ));
        }
        assert!(matches!(
            session.synchronize(),
            Err(Error::Execution(ExecutionError::SynchronizationFailures { failures }))
                if matches!(&failures[..], [(_, Error::Pjrt(PjrtError::Internal { message: failure, .. }))]
                    if *failure == message),
        ));
    }

    #[test]
    fn test_array_to_placement_host_fallback_rejects_varying_manual_axes() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Manual);
        let (logical_mesh, devices) = (mesh.logical_mesh().clone(), mesh.devices().to_vec());
        let sharding = Sharding::replicated(logical_mesh.clone(), 0).with_varying_manual_axes(["x"]).unwrap();
        let first_bytes = values_to_bytes(&[11f32]);
        let second_bytes = values_to_bytes(&[22f32]);
        let source = Array::from_host_shards(
            &session,
            ArrayType::scalar(DataType::F32),
            mesh,
            Some(sharding.clone()),
            [(devices[0].id(), first_bytes.as_slice()), (devices[1].id(), second_bytes.as_slice())],
        )
        .unwrap();

        // A placement that keeps the variation moves each value as it is, so the value at coordinate zero moves to
        // the first device of the reversed mesh.
        let reversed_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1], devices[0]]).unwrap();
        let target = Placement::new(reversed_mesh.clone(), sharding).unwrap();
        assert_eq!(shard_values(&source.to_placement(&session, target).unwrap()), vec![vec![11.], vec![22.]]);

        // The devices hold different values for the same slice, which a dense host copy cannot represent, so the
        // host fallback of a placement that drops the variation reports the error of the device work instead of
        // uploading one of those values everywhere.
        let target = Placement::new(reversed_mesh, Sharding::replicated(logical_mesh, 0)).unwrap();
        assert!(matches!(
            source.to_placement(&session, target),
            Err(Error::Array(error))
                if matches!(error.as_ref(), ArrayError::ManualReshardingAxis { axis_name } if axis_name == "x"),
        ));
        assert_eq!(shard_values(&source), vec![vec![11.], vec![22.]]);
    }

    #[test]
    fn test_array_to_placement_copies_resharded_host_shards_through_host() {
        // On two CPUs, a replicated `u8[2, 4]` in pinned or unpinned host memory resharded into column shards holds
        // buffers that a compiled reshard produced in host memory in a layout that the CPU plugin can neither copy on
        // the device nor copy to the other CPU. Moving them to device memory, copying them without aliasing, and
        // reversing their devices while keeping the column split, into the same memory or into device memory, copy
        // each shard through host memory from its own buffer, through every placement entry point.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let devices = mesh.devices().to_vec();
        let reversed = DeviceMesh::new(mesh.logical_mesh().clone(), vec![devices[1], devices[0]]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 2);
        let columns = vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])];
        let columns = Sharding::new(mesh.logical_mesh().clone(), columns).unwrap();
        let values = (0u8..8).collect::<Vec<_>>();
        let expected = |devices: [Device; 2]| vec![(devices[0], vec![0u8, 1, 4, 5]), (devices[1], vec![2, 3, 6, 7])];
        let shard_bytes = |array: &Array| {
            let shards = array.shards().iter();
            shards.map(|shard| (shard.device(), shard.copy_to_host_bytes().unwrap())).collect::<Vec<_>>()
        };
        for memory in [Memory::Host { pinned: true }, Memory::Host { pinned: false }] {
            let r#type = ArrayType::new_static(DataType::U8, [2, 4]).with_memory(memory);
            let sharding = Some(replicated.clone());
            let source = Array::from_host_buffer(&session, r#type, mesh.clone(), sharding, values.as_slice()).unwrap();
            let split = Placement::new(mesh.clone(), columns.clone()).unwrap();
            let resharded = source.to_placement(&session, split).unwrap();
            assert_eq!(shard_bytes(&resharded), expected([devices[0], devices[1]]), "{memory:?}");

            let targets = [
                (&mesh, Memory::Device, [devices[0], devices[1]]),
                (&reversed, memory, [devices[1], devices[0]]),
                (&reversed, Memory::Device, [devices[1], devices[0]]),
            ];
            for (target_mesh, target_memory, target_devices) in targets {
                let target = Placement::new(target_mesh.clone(), columns.clone()).unwrap().with_memory(target_memory);
                for placed in place_through_entry_points(&session, &resharded, target) {
                    let placed = placed.unwrap();
                    let observed = (placed.mesh().clone(), placed.r#type().memory(), shard_bytes(&placed));
                    let case = format!("{memory:?} -> {target_memory:?} on {target_devices:?}");
                    assert_eq!(observed, (target_mesh.clone(), target_memory, expected(target_devices)), "{case}");
                    assert_eq!(placed.copy_to_host_bytes().unwrap(), values, "{case}");
                }
            }

            let copied = session.place(resharded.clone(), PlacementOptions::default().with_may_alias(false)).unwrap();
            let shares = copied
                .shards()
                .iter()
                .zip(resharded.shards())
                .map(|(copied, resharded)| Arc::ptr_eq(copied.buffer().unwrap(), resharded.buffer().unwrap()));
            assert_eq!(shares.collect::<Vec<_>>(), [false; 2], "{memory:?}");
            let observed = (copied.r#type().memory(), shard_bytes(&copied));
            assert_eq!(observed, (memory, expected([devices[0], devices[1]])), "{memory:?}");
            assert_eq!(copied.copy_to_host_bytes().unwrap(), values, "{memory:?}");
        }
    }

    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_array_to_placement_between_clients_cpu_process_helper() {
        let Some((rank, address)) = test_process() else {
            return;
        };
        let plugin = load_cpu_plugin().unwrap();
        let runtime: &'static DistributedRuntime =
            Box::leak(Box::new(DistributedRuntime::initialize(&plugin, &address, 2, rank).unwrap()));

        // One of the two clients names another process than the runtime has, which `DistributedRuntime::new_client`
        // rejects, so both are created from the key-value store of the runtime directly.
        let client = |process_id| {
            let options = ClientOptions::CPU(CpuClientOptions { device_count: Some(1), process_id: Some(process_id) });
            &*Box::leak(Box::new(plugin.client_with_key_value_store(options, runtime.key_value_store()).unwrap()))
        };
        let source_client = client(1 - rank);
        let target_client = client(rank);
        assert_eq!(source_client.process_index().unwrap(), 1 - rank);
        assert_eq!(target_client.process_index().unwrap(), rank);
        let store = runtime.key_value_store();
        let timeout = Duration::from_secs(5);
        let owner_device = |key: &[u8], client: &'static Client<'static>, process| {
            if rank == 0 {
                let addressable = client.addressable_devices().unwrap();
                store.put(key, &addressable[0].id().unwrap().to_le_bytes()).unwrap();
            }
            let id = usize::from_le_bytes(store.get(key, timeout).unwrap().try_into().unwrap());
            Device::new(id, process)
        };
        let source_device = owner_device(b"ryft/test/host-bridge/source", source_client, 1);
        let target_device = owner_device(b"ryft/test/host-bridge/target", target_client, 0);
        let session = |client, device| {
            Session::builder()
                .client(client)
                .remote_devices((rank == 1).then_some(device))
                .value_comparison(false)
                .build()
                .unwrap()
        };
        let source_session = session(source_client, source_device);
        let target_session = session(target_client, target_device);
        let source_mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![source_device]).unwrap();
        let r#type = ArrayType::new_static(DataType::U8, [1]);
        let source = Array::from_host_buffer(&source_session, r#type, source_mesh, None, [17u8]).unwrap();
        if rank == 0 {
            assert_eq!(source.copy_to_host_bytes().unwrap(), [17u8]);
        }
        let error = source.to_placement(&target_session, target_device).unwrap_err();
        assert!(matches!(
            error,
            Error::Pjrt(PjrtError::InvalidArgument { message, .. })
                if message == "cannot place an array of another PJRT client onto the devices of process 0, which \
                               holds no replica of shard #0 of the array, because the transfer between the clients \
                               reads back the value of the array on that process",
        ));

        let finished = |rank| format!("ryft/test/host-bridge/finished/{rank}");
        store.put(finished(rank).as_bytes(), b"ready").unwrap();
        store.get(finished(1 - rank).as_bytes(), timeout).unwrap();
        runtime.shutdown().unwrap();
    }

    #[test]
    fn test_array_to_placement_between_clients_on_two_cpu_processes() {
        run_cpu_processes("arrays::placement::tests::test_array_to_placement_between_clients_cpu_process_helper", 2);
    }

    // Worker of `test_array_to_placement_on_two_cuda_processes`, which returns immediately unless that test launched it
    // as one of its two ranks. It runs the placements of `cross_host_placements` with `Array::to_placement`, checking
    // that each source keeps its shards, and then with `Array::into_placement`, and asserts the outcomes of both.
    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_array_to_placement_cuda_process_helper() {
        // Places arrays of `[1, 2, 3, 4]` across the two ranks of a CUDA job, whose devices are `devices`, with
        // `place`, which calls `Array::to_placement` or `Array::into_placement`, and returns the outcome of each
        // placement, labeled with its name, as the elements of the shards that this rank addresses once the result is
        // ready, or as the first line of the error of the placement or of its readiness check. Every rank of the job
        // must call this function, because each placement transfers shards between the ranks or executes a compiled
        // reshard over the devices of both.
        fn cross_host_placements(
            devices: &[Device],
            session: &Session,
            place: impl Fn(Array, &Session, Placement) -> Result<Array, Error>,
        ) -> Vec<(&'static str, Result<Vec<Vec<f32>>, String>)> {
            assert_eq!(devices.iter().map(Device::process_index).collect::<Vec<_>>(), vec![0, 1]);
            let replicated = |mesh: &DeviceMesh| Sharding::replicated(mesh.logical_mesh().clone(), 1);
            let sharded = |mesh: &DeviceMesh| {
                Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()
            };
            let array = |mesh: &DeviceMesh, sharding: Sharding, values: [f32; 4]| {
                let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding).unwrap();
                Array::from_host_buffer(session, r#type, mesh.clone(), None, values_to_bytes(&values)).unwrap()
            };
            let process_0_mesh = explicit_mesh(&[devices[0]]);
            let forward_mesh = explicit_mesh(devices);
            let reversed_mesh = explicit_mesh(&[devices[1], devices[0]]);

            // This checks whole-shard copies, slicing a region before sending it, and joining local and received
            // regions on each target. CUDA pairs sends with receives in issue order for each pair of devices.
            // The last copy exposes an unmatched send from an earlier placement by receiving its old bytes.
            let placements = [
                (
                    "sharded_to_reversed_sharded",
                    array(&forward_mesh, sharded(&forward_mesh), [9.0, 10.0, 11.0, 12.0]),
                    Placement::new(reversed_mesh.clone(), sharded(&reversed_mesh)).unwrap(),
                ),
                (
                    "replicated_to_replicated",
                    array(&process_0_mesh, replicated(&process_0_mesh), [1.0, 2.0, 3.0, 4.0]),
                    Placement::new(forward_mesh.clone(), replicated(&forward_mesh)).unwrap(),
                ),
                (
                    "replicated_to_sharded",
                    array(&process_0_mesh, replicated(&process_0_mesh), [5.0, 6.0, 7.0, 8.0]),
                    Placement::new(forward_mesh.clone(), sharded(&forward_mesh)).unwrap(),
                ),
                (
                    "sharded_to_reversed_replicated",
                    array(&forward_mesh, sharded(&forward_mesh), [13.0, 14.0, 15.0, 16.0]),
                    Placement::new(reversed_mesh.clone(), replicated(&reversed_mesh)).unwrap(),
                ),
                (
                    "replicated_to_replicated_again",
                    array(&process_0_mesh, replicated(&process_0_mesh), [17.0, 18.0, 19.0, 20.0]),
                    Placement::new(forward_mesh.clone(), replicated(&forward_mesh)).unwrap(),
                ),
            ];
            placements
                .into_iter()
                .map(|(name, source, target)| {
                    let placed = place(source, session, target)
                        .and_then(|placed| placed.block_until_ready().map(|()| placed))
                        .map(|placed| read_addressable_f32s(&placed))
                        .map_err(|error| error.to_string().lines().next().unwrap_or("").to_string());
                    (name, placed)
                })
                .collect()
        }

        run_as_cuda_rank(|client| {
            let (rank, devices, session) = cuda_rank_session(client);
            client.cross_host_transfers_extension().unwrap();
            let to_placement = cross_host_placements(&devices, &session, |source, session, target| {
                let source_shards = read_addressable_f32s(&source);
                let placed = source.to_placement(session, target)?;
                assert_eq!(read_addressable_f32s(&source), source_shards);
                Ok(placed)
            });
            let into_placement = cross_host_placements(&devices, &session, |source, session, target| {
                source.into_placement(session, target)
            });

            // Each rank holds the shards that the target places on its own device, which is the second device of the
            // reversed mesh for rank 0, and both entry points place alike.
            let expected: [Vec<(&str, Result<Vec<Vec<f32>>, String>)>; 2] = [
                vec![
                    ("sharded_to_reversed_sharded", Ok(vec![vec![11.0, 12.0]])),
                    ("replicated_to_replicated", Ok(vec![vec![1.0, 2.0, 3.0, 4.0]])),
                    ("replicated_to_sharded", Ok(vec![vec![5.0, 6.0]])),
                    ("sharded_to_reversed_replicated", Ok(vec![vec![13.0, 14.0, 15.0, 16.0]])),
                    ("replicated_to_replicated_again", Ok(vec![vec![17.0, 18.0, 19.0, 20.0]])),
                ],
                vec![
                    ("sharded_to_reversed_sharded", Ok(vec![vec![9.0, 10.0]])),
                    ("replicated_to_replicated", Ok(vec![vec![1.0, 2.0, 3.0, 4.0]])),
                    ("replicated_to_sharded", Ok(vec![vec![7.0, 8.0]])),
                    ("sharded_to_reversed_replicated", Ok(vec![vec![13.0, 14.0, 15.0, 16.0]])),
                    ("replicated_to_replicated_again", Ok(vec![vec![17.0, 18.0, 19.0, 20.0]])),
                ],
            ];
            assert_eq!(to_placement, expected[rank]);
            assert_eq!(into_placement, expected[rank]);
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_to_placement_on_two_cuda_processes() {
        run_cuda_rank_pair("arrays::placement::tests::test_array_to_placement_cuda_process_helper");
    }

    // Worker of `test_array_to_placement_relayouts_cross_host_replica_on_two_cuda_processes`, which returns immediately
    // unless that test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_array_to_placement_relayouts_cross_host_replica_cuda_process_helper() {
        run_as_cuda_rank(|client| {
            let (rank, devices, session) = cuda_rank_session(client);
            let source_mesh = explicit_mesh(&devices[..1]);
            let target_mesh = explicit_mesh(&devices);
            let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));

            // A column-major replica of `[[1, 2, 3], [4, 5, 6]]` on the device of rank 0 is split by rows over both
            // ranks. Neither whole shards nor shard regions apply, so rank 0 sends its replica to rank 1 for a
            // compiled reshard on the target mesh, and it must convert the replica to the row-major layout that
            // the receive of rank 1 allocates, or rank 1 would decode its row as `[5, 3, 6]`.
            let r#type = ArrayType::new_static(DataType::F32, [2, 3])
                .with_layout(column_major.clone())
                .with_sharding(Sharding::replicated(source_mesh.logical_mesh().clone(), 2))
                .unwrap();
            let values = values_to_bytes(&[1f32, 2., 3., 4., 5., 6.]);
            let source = Array::from_host_buffer(&session, r#type, source_mesh, None, values).unwrap();
            let rows = Sharding::new(
                target_mesh.logical_mesh().clone(),
                vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
            )
            .unwrap();
            let placed = source.to_placement(&session, Placement::new(target_mesh.clone(), rows.clone()).unwrap());
            let placed = placed.unwrap();
            placed.block_until_ready().unwrap();
            assert_eq!(placed.mesh(), &target_mesh);
            assert_eq!(placed.sharding(), &rows);
            assert_eq!(placed.layout(), Some(&column_major));

            // Each rank holds the row of the target on its own device in a column-major buffer, which is
            // described by its slice, whether its physical layout is column-major, and its dense row-major values.
            let shards = placed
                .addressable_shards()
                .map(|shard| {
                    let layout = shard.buffer().unwrap().specification().unwrap().layout;
                    let column_major = layout == column_major.to_pjrt().unwrap();
                    (
                        shard.slice().to_vec(),
                        column_major,
                        values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()),
                    )
                })
                .collect::<Vec<_>>();
            let expected: [Vec<(Vec<Range<usize>>, bool, Vec<f32>)>; 2] = [
                vec![(vec![0..1, 0..3], true, vec![1.0, 2.0, 3.0])],
                vec![(vec![1..2, 0..3], true, vec![4.0, 5.0, 6.0])],
            ];
            assert_eq!(shards, expected[rank]);
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_to_placement_relayouts_cross_host_replica_on_two_cuda_processes() {
        run_cuda_rank_pair(
            "arrays::placement::tests::test_array_to_placement_relayouts_cross_host_replica_cuda_process_helper",
        );
    }

    // Worker of `test_array_to_placement_agrees_on_strategy_on_two_cuda_processes`, which returns immediately unless
    // that test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_array_to_placement_agrees_on_strategy_cuda_process_helper() {
        // Compilation artifact exchange that requires sharing every artifact with the other rank. A rank leads the
        // compilation of a program over its own devices, so it publishes the conversion of its own shard and never
        // waits for an artifact of the other rank. A placement that switched to a program over both ranks would wait.
        struct SharedCompilationExchange {
            process_index: usize,
        }

        impl CompilationArtifactExchange for SharedCompilationExchange {
            fn process_index(&self) -> usize {
                self.process_index
            }

            fn process_count(&self) -> usize {
                2
            }

            fn publish(&self, _: &[u8], _: &[u8]) -> Result<(), CompilationArtifactExchangeError> {
                Ok(())
            }

            fn receive(&self, _: &[u8], _: Duration) -> Result<Option<Vec<u8>>, CompilationArtifactExchangeError> {
                panic!("a placement of whole shards must not wait for a compiled program");
            }
        }

        run_as_cuda_rank(|client| {
            let rank = client.process_index().unwrap();
            let devices = client
                .devices()
                .unwrap()
                .iter()
                .map(|device| Device::from_pjrt(device).unwrap())
                .collect::<Vec<_>>();
            let cache = CompilationCache::new().with_artifact_exchange(
                Arc::new(SharedCompilationExchange { process_index: rank }),
                CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_secs(60) },
            );
            let session = Session::builder().client(client).compilation_cache(cache).build().unwrap();
            let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
            let replicated = Sharding::replicated(logical_mesh.clone(), 2);
            let source_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[0]]).unwrap();
            let target_mesh = DeviceMesh::new(logical_mesh, vec![devices[1]]).unwrap();

            // Rank 0 holds `[[1, 2, 3], [4, 5, 6]]` in a column-major buffer, which the type does not declare, so only
            // rank 0 knows that its buffer is not in the row-major layout that the receive of rank 1 allocates. Rank 1
            // only receives. Both ranks choose whole shards from the shared metadata, and rank 0 converts its buffer
            // with a program over its own device, which it leads and publishes, rather than switching to a compiled
            // reshard that rank 1 would never join.
            let buffers = if rank == 0 {
                let device = client.addressable_devices().unwrap().remove(0);
                let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])).to_pjrt().unwrap();
                let values = values_to_bytes(&[1f32, 2., 3., 4., 5., 6.]);
                let memory = device.default_memory().unwrap();
                let buffer =
                    client.buffer(values.as_slice(), BufferType::F32, [2u64, 3], None, memory, Some(column_major));
                let buffer = buffer.unwrap();
                assert_ne!(
                    buffer.specification().unwrap().layout,
                    Layout::Tiled(TiledLayout::new(vec![1, 0], vec![])).to_pjrt().unwrap(),
                );
                vec![buffer]
            } else {
                Vec::new()
            };
            let r#type = ArrayType::new_static(DataType::F32, [2, 3]).with_sharding(replicated.clone()).unwrap();
            let source = Array::from_addressable_buffers(&session, r#type, source_mesh, None, buffers).unwrap();
            let target = Placement::new(target_mesh.clone(), replicated).unwrap();
            let placed = source.to_placement(&session, target).unwrap();
            placed.block_until_ready().unwrap();
            assert_eq!(placed.mesh(), &target_mesh);
            assert_eq!(session.cached_program_count(), if rank == 0 { 1 } else { 0 });
            let shards =
                placed.addressable_shards().map(|shard| shard.copy_to_host_bytes().unwrap()).collect::<Vec<_>>();
            if rank == 0 {
                assert_eq!(shards.len(), 0);
            } else {
                assert_eq!(
                    shards.iter().map(|bytes| values_from_bytes::<f32>(bytes)).collect::<Vec<_>>(),
                    [vec![1., 2., 3., 4., 5., 6.]]
                );
            }
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_to_placement_agrees_on_strategy_on_two_cuda_processes() {
        run_cuda_rank_pair("arrays::placement::tests::test_array_to_placement_agrees_on_strategy_cuda_process_helper");
    }

    // Worker of `test_array_to_placement_swaps_shards_into_memory_on_two_cuda_processes`, which returns immediately
    // unless that test launched it as one of its two ranks. It swaps the halves of `[1, 2, 3, 4]` between the ranks
    // into each memory and asserts the result of each swap.
    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_array_to_placement_swaps_shards_into_memory_cuda_process_helper() {
        run_as_cuda_rank(|client| {
            let (rank, devices, session) = cuda_rank_session(client);
            let forward_mesh = explicit_mesh(&devices);
            let reversed_mesh = explicit_mesh(&[devices[1], devices[0]]);
            let sharded =
                Sharding::new(forward_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
            let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharded.clone()).unwrap();
            let values = values_to_bytes(&[1f32, 2., 3., 4.]);
            let source = Array::from_host_buffer(&session, r#type, forward_mesh, None, values).unwrap();

            // Each rank sends its half to the other one and receives the other half into default device memory, which
            // it then copies to the requested memory. Each rank looks up that memory on its own device only after it
            // issued both of its transfers, so a memory that its device lacks fails that rank after the swap is paired,
            // and every later swap still pairs its transfers. A swap that left a transfer unmatched would make the next
            // swap between the same devices read stale halves. Both memories exist on CUDA devices, so this does not
            // exercise such a failure, and a rank that failed so would not reach the transfers of a later placement of
            // a tree. Each swap returns the memory kind and the elements of every shard that this rank addresses once
            // the result is ready, or the first line of the error of the placement or of its readiness check.
            let pinned = Memory::Host { pinned: true };
            let unpinned = Memory::Host { pinned: false };
            let swaps = [pinned, unpinned, Memory::Device]
                .into_iter()
                .map(|memory| {
                    let target = Placement::new(reversed_mesh.clone(), sharded.clone()).unwrap().with_memory(memory);
                    let placed = source
                        .to_placement(&session, target)
                        .and_then(|placed| placed.block_until_ready().map(|()| placed))
                        .map(|placed| {
                            placed
                                .addressable_shards()
                                .map(|shard| {
                                    let kind = shard.buffer().unwrap().memory().unwrap().kind().unwrap().into_owned();
                                    (kind, values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()))
                                })
                                .collect::<Vec<_>>()
                        })
                        .map_err(|error| error.to_string().lines().next().unwrap_or("").to_string());
                    (memory, placed)
                })
                .collect::<Vec<_>>();
            let expected: [Vec<(Memory, Result<Vec<(String, Vec<f32>)>, String>)>; 2] = [
                vec![
                    (pinned, Ok(vec![("pinned_host".to_string(), vec![3.0, 4.0])])),
                    (unpinned, Ok(vec![("unpinned_host".to_string(), vec![3.0, 4.0])])),
                    (Memory::Device, Ok(vec![("device".to_string(), vec![3.0, 4.0])])),
                ],
                vec![
                    (pinned, Ok(vec![("pinned_host".to_string(), vec![1.0, 2.0])])),
                    (unpinned, Ok(vec![("unpinned_host".to_string(), vec![1.0, 2.0])])),
                    (Memory::Device, Ok(vec![("device".to_string(), vec![1.0, 2.0])])),
                ],
            ];
            assert_eq!(swaps, expected[rank]);
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_to_placement_swaps_shards_into_memory_on_two_cuda_processes() {
        run_cuda_rank_pair(
            "arrays::placement::tests::test_array_to_placement_swaps_shards_into_memory_cuda_process_helper",
        );
    }

    // Worker of `test_array_to_placement_transfers_between_memories_on_two_cuda_processes`, which returns immediately
    // unless that test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_array_to_placement_transfers_between_memories_cuda_process_helper() {
        run_as_cuda_rank(|client| {
            let (rank, devices, session) = cuda_rank_session(client);
            let source_mesh = explicit_mesh(&devices[..1]);
            let target_mesh = explicit_mesh(&devices[1..]);
            let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
            let values = values_to_bytes(&[1f32, 2., 3., 4.]);

            // Rank 0 only sends `[1, 2, 3, 4]` from its device, as an `f32[4]` or as an `f32[2, 2]` in a declared
            // column-major layout, and rank 1 only receives it on its own device. The CUDA cross-host transfer reads a
            // device buffer in the row-major layout that the receive allocates, so rank 0 stages a source in host
            // memory in device memory and converts a source in a declared layout to that layout before it sends it.
            // Rank 1 receives into default device memory, then converts the received buffer to the declared layout,
            // if any, and copies it to the requested memory, or to the memory of the source when the target keeps it,
            // only after it issued its receive. A memory that only the receiver checks therefore never leaves the
            // sender waiting, and rank 0, which holds no target shard, records the same metadata without any work. A
            // placement that left a transfer unmatched would make the next one between the same devices read stale
            // values. Each step lists its name, the memory of the source, whether the source declares a column-major
            // layout, the memory that the target requests, if any (i.e., none when the target keeps the memory of the
            // source), and the type of the result that both ranks record.
            let pinned = Memory::Host { pinned: true };
            let unpinned = Memory::Host { pinned: false };
            let steps = [
                (
                    "f32[4] from Device into Device",
                    Memory::Device,
                    false,
                    Some(Memory::Device),
                    "f32[4][sharding={mesh<['x'=1:explicit]>, [{}]}]",
                ),
                (
                    "f32[4] from Device into Host { pinned: true }",
                    Memory::Device,
                    false,
                    Some(pinned),
                    "f32[4][sharding={mesh<['x'=1:explicit]>, [{}]}]@Host[Pinned]",
                ),
                (
                    "f32[4] from Device into Host { pinned: false }",
                    Memory::Device,
                    false,
                    Some(unpinned),
                    "f32[4][sharding={mesh<['x'=1:explicit]>, [{}]}]@Host[Unpinned]",
                ),
                (
                    "column-major f32[2, 2] from Device into Device",
                    Memory::Device,
                    true,
                    Some(Memory::Device),
                    "f32[2, 2][layout=tiled{0,1}][sharding={mesh<['x'=1:explicit]>, [{}, {}]}]",
                ),
                (
                    "column-major f32[2, 2] from Device into Host { pinned: true }",
                    Memory::Device,
                    true,
                    Some(pinned),
                    "f32[2, 2][layout=tiled{0,1}][sharding={mesh<['x'=1:explicit]>, [{}, {}]}]@Host[Pinned]",
                ),
                (
                    "column-major f32[2, 2] from Device into Host { pinned: false }",
                    Memory::Device,
                    true,
                    Some(unpinned),
                    "f32[2, 2][layout=tiled{0,1}][sharding={mesh<['x'=1:explicit]>, [{}, {}]}]@Host[Unpinned]",
                ),
                (
                    "f32[4] from Host { pinned: true } into the same memory",
                    pinned,
                    false,
                    None,
                    "f32[4][sharding={mesh<['x'=1:explicit]>, [{}]}]@Host[Pinned]",
                ),
                (
                    "f32[4] from Host { pinned: false } into the same memory",
                    unpinned,
                    false,
                    None,
                    "f32[4][sharding={mesh<['x'=1:explicit]>, [{}]}]@Host[Unpinned]",
                ),
            ];
            let mut shards = Vec::new();
            for (step, memory, declared, target_memory, expected_type) in steps {
                let (r#type, dimensions) = if declared {
                    (ArrayType::new_static(DataType::F32, [2, 2]).with_layout(column_major.clone()), 2)
                } else {
                    (ArrayType::new_static(DataType::F32, [4]), 1)
                };
                let replicated = Sharding::replicated(source_mesh.logical_mesh().clone(), dimensions);
                let r#type = r#type.with_memory(memory).with_sharding(replicated.clone()).unwrap();
                let source =
                    Array::from_host_buffer(&session, r#type, source_mesh.clone(), None, values.as_slice()).unwrap();
                let target = Placement::new(target_mesh.clone(), replicated).unwrap();
                let target = match target_memory {
                    Some(target_memory) => target.with_memory(target_memory),
                    None => target,
                };
                shards.push((step, place_step(rank, step, &session, &source, target, expected_type)));
            }
            let expected: [Vec<(&str, Vec<ShardObservation>)>; 2] = [
                vec![
                    ("f32[4] from Device into Device", vec![]),
                    ("f32[4] from Device into Host { pinned: true }", vec![]),
                    ("f32[4] from Device into Host { pinned: false }", vec![]),
                    ("column-major f32[2, 2] from Device into Device", vec![]),
                    ("column-major f32[2, 2] from Device into Host { pinned: true }", vec![]),
                    ("column-major f32[2, 2] from Device into Host { pinned: false }", vec![]),
                    ("f32[4] from Host { pinned: true } into the same memory", vec![]),
                    ("f32[4] from Host { pinned: false } into the same memory", vec![]),
                ],
                vec![
                    (
                        "f32[4] from Device into Device",
                        vec![(vec![0..4], "device".to_string(), vec![1.0, 2.0, 3.0, 4.0])],
                    ),
                    (
                        "f32[4] from Device into Host { pinned: true }",
                        vec![(vec![0..4], "pinned_host".to_string(), vec![1.0, 2.0, 3.0, 4.0])],
                    ),
                    (
                        "f32[4] from Device into Host { pinned: false }",
                        vec![(vec![0..4], "unpinned_host".to_string(), vec![1.0, 2.0, 3.0, 4.0])],
                    ),
                    (
                        "column-major f32[2, 2] from Device into Device",
                        vec![(vec![0..2, 0..2], "device".to_string(), vec![1.0, 2.0, 3.0, 4.0])],
                    ),
                    (
                        "column-major f32[2, 2] from Device into Host { pinned: true }",
                        vec![(vec![0..2, 0..2], "pinned_host".to_string(), vec![1.0, 2.0, 3.0, 4.0])],
                    ),
                    (
                        "column-major f32[2, 2] from Device into Host { pinned: false }",
                        vec![(vec![0..2, 0..2], "unpinned_host".to_string(), vec![1.0, 2.0, 3.0, 4.0])],
                    ),
                    (
                        "f32[4] from Host { pinned: true } into the same memory",
                        vec![(vec![0..4], "pinned_host".to_string(), vec![1.0, 2.0, 3.0, 4.0])],
                    ),
                    (
                        "f32[4] from Host { pinned: false } into the same memory",
                        vec![(vec![0..4], "unpinned_host".to_string(), vec![1.0, 2.0, 3.0, 4.0])],
                    ),
                ],
            ];
            assert_eq!(shards, expected[rank]);
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_to_placement_transfers_between_memories_on_two_cuda_processes() {
        run_cuda_rank_pair(
            "arrays::placement::tests::test_array_to_placement_transfers_between_memories_cuda_process_helper",
        );
    }

    // Worker of `test_array_to_placement_sends_padded_bounded_dynamic_shard_on_two_cuda_processes`, which returns
    // immediately unless that test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_array_to_placement_sends_padded_bounded_dynamic_shard_cuda_process_helper() {
        run_as_cuda_rank(|client| {
            let (rank, devices, session) = cuda_rank_session(client);
            let source_mesh = explicit_mesh(&devices[..1]);
            let target_mesh = explicit_mesh(&devices[1..]);
            let replicated = Sharding::replicated(source_mesh.logical_mesh().clone(), 1);

            // Rank 0 runs a bounded-dynamic identity whose result buffer is allocated for the bound of its dimension
            // with a trailer that records its extent, and rank 1 holds only the metadata of that result. The CUDA
            // plugin sends as many bytes as the allocation of the sender holds, while the receiver allocates a static
            // buffer of the shard, so rank 0 must send a static copy of exactly that size. This holds for an extent
            // below the bound, where the physical dimensions of the dynamic buffer differ from the shape of the shard,
            // and at the maximum extent, where they are equal but the allocation still holds the trailer. Each step
            // records, on rank 0, the unpadded and dynamic dimensions of the padded source buffer and whether its
            // allocation is larger than the elements of the shard, and, for every shard that this rank addresses, the
            // dimensions and the allocation size of the received buffer, the number of received elements, and whether
            // they are `0, 1, 2, ...`. A failure names the rank and step.
            let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(1024)).unwrap());
            let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
            let identity: crate::CompiledFunction<ArrayType, ArrayType> =
                crate::compile_with_options(|input| input, dynamic_type, &session, Options::new(source_mesh.clone()))
                    .unwrap_or_else(|error| {
                        panic!("rank {rank} failed to compile the bounded-dynamic identity: {error}")
                    });
            let mut steps = Vec::new();
            for size in [2usize, 1023] {
                let step = format!("extent {size}");
                let r#type = ArrayType::new_static(DataType::F32, [size]);
                let values = (0..size).map(|value| value as f32).collect::<Vec<_>>();
                let (source, padded_source) = if rank == 0 {
                    let input = Array::from_host_buffer(
                        &session,
                        r#type.clone(),
                        source_mesh.clone(),
                        Some(replicated.clone()),
                        values_to_bytes(&values),
                    )
                    .unwrap_or_else(|error| panic!("rank {rank} failed to upload the input of the {step}: {error}"));
                    let padded = identity.executable().execute(&session, input).unwrap_or_else(|error| {
                        panic!("rank {rank} failed to run the identity for the {step}: {error}")
                    });
                    let buffer = padded.shards()[0].buffer().unwrap();
                    let padded_source = (
                        buffer.unpadded_dimensions().unwrap().to_vec(),
                        buffer.dynamic_dimensions().unwrap().to_vec(),
                        buffer.on_device_size_in_bytes().unwrap() > 4 * size,
                    );
                    (padded, Some(padded_source))
                } else {
                    let sharding = Some(replicated.clone());
                    let source =
                        Array::from_addressable_buffers(&session, r#type, source_mesh.clone(), sharding, Vec::new())
                            .unwrap_or_else(|error| {
                                panic!("rank {rank} failed to import the source of the {step}: {error}")
                            });
                    (source, None)
                };
                let target = Placement::new(target_mesh.clone(), replicated.clone()).unwrap();
                let placed = source
                    .to_placement(&session, target)
                    .and_then(|placed| placed.block_until_ready().map(|()| placed))
                    .unwrap_or_else(|error| {
                        panic!("rank {rank} failed to send the dynamic shard of the {step}: {error}")
                    });
                let shards = placed
                    .addressable_shards()
                    .map(|shard| {
                        let buffer = shard.buffer().unwrap();
                        let bytes = shard.copy_to_host_bytes().unwrap_or_else(|error| {
                            panic!("rank {rank} failed to read the received shard of the {step}: {error}")
                        });
                        let received = values_from_bytes::<f32>(&bytes);
                        (
                            buffer.dimensions().unwrap().to_vec(),
                            buffer.on_device_size_in_bytes().unwrap(),
                            received.len(),
                            received == values,
                        )
                    })
                    .collect::<Vec<_>>();
                steps.push((size, padded_source, shards));
                session
                    .synchronize()
                    .unwrap_or_else(|error| panic!("rank {rank} failed to synchronize after the {step}: {error}"));
            }
            let expected: [Vec<(usize, Option<(Vec<u64>, Vec<u64>, bool)>, Vec<(Vec<u64>, usize, usize, bool)>)>; 2] = [
                vec![(2, Some((vec![2], vec![0], true)), vec![]), (1023, Some((vec![1023], vec![0], true)), vec![])],
                vec![(2, None, vec![(vec![2], 8, 2, true)]), (1023, None, vec![(vec![1023], 4092, 1023, true)])],
            ];
            assert_eq!(steps, expected[rank]);
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_to_placement_sends_padded_bounded_dynamic_shard_on_two_cuda_processes() {
        run_cuda_rank_pair(
            "arrays::placement::tests::test_array_to_placement_sends_padded_bounded_dynamic_shard_cuda_process_helper",
        );
    }

    // Worker of `test_array_to_placement_compiles_on_mesh_of_one_rank_on_two_cuda_processes`, which returns immediately
    // unless that test launched it as one of its two ranks. It places `[[1, 2], [3, 4]]`, which is split by rows over
    // the devices of both ranks in a column-major layout, replicated onto the device of each rank in turn.
    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_array_to_placement_compiles_on_mesh_of_one_rank_cuda_process_helper() {
        run_as_cuda_rank(|client| {
            let (rank, devices, session) = cuda_rank_session(client);
            let source_mesh = explicit_mesh(&devices);
            let rows = Sharding::new(
                source_mesh.logical_mesh().clone(),
                vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
            )
            .unwrap();
            let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
            let r#type =
                ArrayType::new_static(DataType::F32, [2, 2]).with_layout(column_major).with_sharding(rows).unwrap();
            let values = values_to_bytes(&[1f32, 2., 3., 4.]);
            let source = Array::from_host_buffer(&session, r#type, source_mesh, None, values).unwrap();

            // The declared layout rules out shard regions, so each placement gathers the source on the devices of both
            // ranks, copies the gathered value to the target device on the rank that owns it, and reshards it
            // there. Only that rank compiles and runs the final reshard, and the other rank returns the same metadata
            // without buffers. A rank that compiled for the device of its peer would fail, and a rank that skipped the
            // gather would leave its peer waiting in it.
            let mut shards = Vec::new();
            for (owner, step) in [(0, "onto rank 0"), (1, "onto rank 1")] {
                let mesh = explicit_mesh(&devices[owner..owner + 1]);
                let target =
                    Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 2)).unwrap();
                let r#type = "f32[2, 2][layout=tiled{0,1}][sharding={mesh<['x'=1:explicit]>, [{}, {}]}]";
                shards.push((step, place_step(rank, step, &session, &source, target, r#type)));
            }
            let expected: [Vec<(&str, Vec<ShardObservation>)>; 2] = [
                vec![
                    ("onto rank 0", vec![(vec![0..2, 0..2], "device".to_string(), vec![1.0, 2.0, 3.0, 4.0])]),
                    ("onto rank 1", vec![]),
                ],
                vec![
                    ("onto rank 0", vec![]),
                    ("onto rank 1", vec![(vec![0..2, 0..2], "device".to_string(), vec![1.0, 2.0, 3.0, 4.0])]),
                ],
            ];
            assert_eq!(shards, expected[rank]);
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_to_placement_compiles_on_mesh_of_one_rank_on_two_cuda_processes() {
        run_cuda_rank_pair(
            "arrays::placement::tests::test_array_to_placement_compiles_on_mesh_of_one_rank_cuda_process_helper",
        );
    }

    // Worker of `test_array_to_placement_mixes_process_local_strategies_on_two_cuda_processes`, which returns
    // immediately unless that test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_array_to_placement_mixes_process_local_strategies_cuda_process_helper() {
        run_as_cuda_rank(|client| {
            let (rank, devices, session) = cuda_rank_session(client);
            let mut steps = [
                "onto rank 0",
                "onto rank 0 in pinned host memory",
                "onto rank 1",
                "onto rank 1 in pinned host memory",
                "replicated over both ranks",
                "split over both ranks",
                "in device memory, replicated over both ranks",
                "in device memory, split over both ranks",
                "committed",
            ]
            .into_iter();
            let mut shards = Vec::new();

            // Each rank uploads `[1, 2, 3, 4]` without a target, so its array lives uncommitted on its own
            // device, which the other rank does not know. A target on the device of one rank is placed by that
            // rank with the strategy of a shared source, which moves nothing between the ranks, and by the other rank
            // with local work only, which places nothing on its devices, and both ranks must still record the same
            // metadata.
            let r#type = ArrayType::new_static(DataType::F32, [4]);
            let values = values_to_bytes(&[1f32, 2., 3., 4.]);
            let source = Array::from_host_buffer(&session, r#type, None, None, values).unwrap();
            assert!(!source.is_committed(), "rank {rank} uploaded a committed value");
            assert!(
                source.is_locally_assigned(),
                "rank {rank} uploaded a value whose devices were not locally assigned"
            );
            assert_eq!(source.mesh().devices(), [devices[rank]]);
            for owner in [0, 1] {
                let mesh = explicit_mesh(&devices[owner..owner + 1]);
                let target = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 1));
                let target = target.unwrap();
                let step = steps.next().unwrap();
                let r#type = "f32[4][sharding={mesh<['x'=1:explicit]>, [{}]}]";
                shards.push((step, place_step(rank, step, &session, &source, target.clone(), r#type)));
                let target = target.with_memory(Memory::Host { pinned: true });
                let step = steps.next().unwrap();
                let r#type = "f32[4][sharding={mesh<['x'=1:explicit]>, [{}]}]@Host[Pinned]";
                shards.push((step, place_step(rank, step, &session, &source, target, r#type)));
            }

            // A target over the devices of both ranks is placed by each rank with local work only, also once the
            // value is placed into device memory, which commits it to its own device, because each rank still chose
            // that device on its own.
            let mesh = explicit_mesh(&devices);
            let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
            let split = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
            let in_memory = source.to_placement(&session, Memory::Device).unwrap();
            assert!(in_memory.is_committed(), "rank {rank} placed a value into memory without committing it");
            assert_eq!(in_memory.mesh().devices(), [devices[rank]]);
            for source in [&source, &in_memory] {
                for (sharding, r#type) in [
                    (&replicated, "f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}]"),
                    (&split, "f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}]"),
                ] {
                    let target = Placement::new(mesh.clone(), sharding.clone()).unwrap();
                    let step = steps.next().unwrap();
                    shards.push((step, place_step(rank, step, &session, source, target, r#type)));
                }
            }

            // A committed value of rank 0 then moves to rank 1, which would receive the bytes of an earlier send that
            // no receive matched.
            let source_mesh = explicit_mesh(&devices[..1]);
            let r#type = ArrayType::new_static(DataType::F32, [4]);
            let values = values_to_bytes(&[5f32, 6., 7., 8.]);
            let committed = Array::from_host_buffer(&session, r#type, source_mesh, None, values).unwrap();
            let target = Placement::new(mesh, replicated).unwrap();
            let step = steps.next().unwrap();
            let r#type = "f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}]";
            shards.push((step, place_step(rank, step, &session, &committed, target, r#type)));
            assert!(steps.next().is_none());

            // The rank that owns the device of a target on one device holds its shard, and each rank holds the
            // shard of its own device of a target over the devices of both ranks.
            let expected: [Vec<(&str, Vec<ShardObservation>)>; 2] = [
                vec![
                    ("onto rank 0", vec![(vec![0..4], "device".to_string(), vec![1.0, 2.0, 3.0, 4.0])]),
                    (
                        "onto rank 0 in pinned host memory",
                        vec![(vec![0..4], "pinned_host".to_string(), vec![1.0, 2.0, 3.0, 4.0])],
                    ),
                    ("onto rank 1", vec![]),
                    ("onto rank 1 in pinned host memory", vec![]),
                    ("replicated over both ranks", vec![(vec![0..4], "device".to_string(), vec![1.0, 2.0, 3.0, 4.0])]),
                    ("split over both ranks", vec![(vec![0..2], "device".to_string(), vec![1.0, 2.0])]),
                    (
                        "in device memory, replicated over both ranks",
                        vec![(vec![0..4], "device".to_string(), vec![1.0, 2.0, 3.0, 4.0])],
                    ),
                    (
                        "in device memory, split over both ranks",
                        vec![(vec![0..2], "device".to_string(), vec![1.0, 2.0])],
                    ),
                    ("committed", vec![(vec![0..4], "device".to_string(), vec![5.0, 6.0, 7.0, 8.0])]),
                ],
                vec![
                    ("onto rank 0", vec![]),
                    ("onto rank 0 in pinned host memory", vec![]),
                    ("onto rank 1", vec![(vec![0..4], "device".to_string(), vec![1.0, 2.0, 3.0, 4.0])]),
                    (
                        "onto rank 1 in pinned host memory",
                        vec![(vec![0..4], "pinned_host".to_string(), vec![1.0, 2.0, 3.0, 4.0])],
                    ),
                    ("replicated over both ranks", vec![(vec![0..4], "device".to_string(), vec![1.0, 2.0, 3.0, 4.0])]),
                    ("split over both ranks", vec![(vec![2..4], "device".to_string(), vec![3.0, 4.0])]),
                    (
                        "in device memory, replicated over both ranks",
                        vec![(vec![0..4], "device".to_string(), vec![1.0, 2.0, 3.0, 4.0])],
                    ),
                    (
                        "in device memory, split over both ranks",
                        vec![(vec![2..4], "device".to_string(), vec![3.0, 4.0])],
                    ),
                    ("committed", vec![(vec![0..4], "device".to_string(), vec![5.0, 6.0, 7.0, 8.0])]),
                ],
            ];
            assert_eq!(shards, expected[rank]);
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_to_placement_mixes_process_local_strategies_on_two_cuda_processes() {
        run_cuda_rank_pair(
            "arrays::placement::tests::test_array_to_placement_mixes_process_local_strategies_cuda_process_helper",
        );
    }

    // Worker of `test_array_to_placement_reshards_across_meshes_on_two_cuda_processes`, which returns immediately
    // unless that test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_array_to_placement_reshards_across_meshes_cuda_process_helper() {
        run_as_cuda_rank(|client| {
            let (rank, devices, session) = cuda_rank_session(client);
            let rows = |mesh: &DeviceMesh| {
                let dimensions = vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()];
                Sharding::new(mesh.logical_mesh().clone(), dimensions).unwrap()
            };
            let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
            let meshes = [explicit_mesh(&devices), explicit_mesh(&[devices[1], devices[0]])];
            let mut steps = [
                "from rank 0 onto the forward mesh",
                "from rank 0 onto the reversed mesh",
                "from rank 1 onto the forward mesh",
                "from rank 1 onto the reversed mesh",
            ]
            .into_iter();
            let mut shards = Vec::new();

            // `[[1, 2], [3, 4]]` is split by rows over the device of one rank alone, in a column-major layout. Its
            // shards differ from those of a split over both ranks, and the declared layout rules out shard regions, so
            // the owner gathers the value on its own device, keeps that replica for its own device of the target
            // mesh and sends it to the other rank, which only receives it, and both ranks then run the final reshard
            // on the target mesh. `test_array_device_placement_strategy_agrees_across_processes` plans the same
            // strategy for the row `shared sharded with a declared layout -> sharded over both processes`.
            for owner in [0, 1] {
                let source_mesh = explicit_mesh(&devices[owner..owner + 1]);
                let r#type = ArrayType::new_static(DataType::F32, [2, 2])
                    .with_layout(column_major.clone())
                    .with_sharding(rows(&source_mesh))
                    .unwrap();
                let values = values_to_bytes(&[1f32, 2., 3., 4.]);
                let source = Array::from_host_buffer(&session, r#type, source_mesh, None, values).unwrap();
                for mesh in &meshes {
                    let target = Placement::new(mesh.clone(), rows(mesh)).unwrap();
                    let step = steps.next().unwrap();
                    let r#type = "f32[2, 2][layout=tiled{0,1}][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}]";
                    shards.push((step, place_step(rank, step, &session, &source, target, r#type)));
                }
            }
            assert!(steps.next().is_none());

            // Each rank holds the row of the target on its own device, which is the first row on the forward mesh
            // and the second one on the reversed mesh for rank 0, whichever rank owned the source.
            let expected: [Vec<(&str, Vec<ShardObservation>)>; 2] = [
                vec![
                    (
                        "from rank 0 onto the forward mesh",
                        vec![(vec![0..1, 0..2], "device".to_string(), vec![1.0, 2.0])],
                    ),
                    (
                        "from rank 0 onto the reversed mesh",
                        vec![(vec![1..2, 0..2], "device".to_string(), vec![3.0, 4.0])],
                    ),
                    (
                        "from rank 1 onto the forward mesh",
                        vec![(vec![0..1, 0..2], "device".to_string(), vec![1.0, 2.0])],
                    ),
                    (
                        "from rank 1 onto the reversed mesh",
                        vec![(vec![1..2, 0..2], "device".to_string(), vec![3.0, 4.0])],
                    ),
                ],
                vec![
                    (
                        "from rank 0 onto the forward mesh",
                        vec![(vec![1..2, 0..2], "device".to_string(), vec![3.0, 4.0])],
                    ),
                    (
                        "from rank 0 onto the reversed mesh",
                        vec![(vec![0..1, 0..2], "device".to_string(), vec![1.0, 2.0])],
                    ),
                    (
                        "from rank 1 onto the forward mesh",
                        vec![(vec![1..2, 0..2], "device".to_string(), vec![3.0, 4.0])],
                    ),
                    (
                        "from rank 1 onto the reversed mesh",
                        vec![(vec![0..1, 0..2], "device".to_string(), vec![1.0, 2.0])],
                    ),
                ],
            ];
            assert_eq!(shards, expected[rank]);
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_array_to_placement_reshards_across_meshes_on_two_cuda_processes() {
        run_cuda_rank_pair(
            "arrays::placement::tests::test_array_to_placement_reshards_across_meshes_cuda_process_helper",
        );
    }

    #[test]
    fn test_array_into_placement() {
        // A value on one device is consumed and split over two devices, one of which is the device that held it.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![first_device(client)],
        )
        .unwrap();
        let source_sharding = Sharding::replicated(source_mesh.logical_mesh().clone(), 1);
        let source_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(source_sharding).unwrap();
        let source_bytes = values_to_bytes(&[0f32, 1., 2., 3.]);
        let source =
            Array::from_host_buffer(&session, source_type, source_mesh, None, source_bytes.as_slice()).unwrap();
        let target_mesh = domain_mesh(client, "x", 2);
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let target = Placement::new(target_mesh.clone(), target_sharding.clone()).unwrap();

        let moved = source.into_placement(&session, target).unwrap();

        assert_eq!(moved.mesh(), &target_mesh);
        assert_eq!(moved.sharding(), &target_sharding);
        assert_eq!(moved.addressable_shards().count(), 2);
        assert_eq!(shard_values(&moved), vec![vec![0.0, 1.0], vec![2.0, 3.0]]);
    }

    #[test]
    fn test_array_into_placement_preserves_same_partially_addressable_placement() {
        let client = cpu_client(1);
        let local = first_device(client);
        let remote = Device::new(local.id() + 1, 1);
        let session = session_with_remote_devices(client, [remote]);
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![local, remote],
        )
        .unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let local_device = client.addressable_devices().unwrap().remove(0);
        let bytes = values_to_bytes(&[0f32, 1.]);
        let buffer = client.buffer(bytes.as_slice(), BufferType::F32, [2u64], None, local_device, None).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let source =
            Array::from_addressable_buffers(&session, r#type, mesh.clone(), Some(sharding.clone()), vec![buffer]);

        let copied = source.unwrap().into_placement(&session, Placement::new(mesh, sharding).unwrap()).unwrap();
        let expected_visualization = format!("┌─────┬─────┐\n│{:^5}│{:^5}│\n└─────┴─────┘", local.id(), remote.id());
        assert_eq!(copied.addressable_shards().count(), 1);
        assert_eq!(copied.sharding().visualize().unwrap().render(false), expected_visualization);
        let copied_shard_bytes = copied.device_shard(local.id()).unwrap().copy_to_host_bytes().unwrap();
        assert_eq!(values_from_bytes::<f32>(copied_shard_bytes.as_slice()), vec![0.0, 1.0]);
        assert!(copied.device_shard(remote.id()).unwrap().buffer().is_none());
    }

    #[test]
    fn test_array_into_placement_agrees_with_to_placement_on_a_declared_layout_with_a_pending_sum() {
        // Completing a pending sum of a column-major `f32[4, 4]` over two devices, with `to_placement` and with
        // `into_placement` of a clone (so that nothing can be donated), reads back the same value through both.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mut outcomes = Vec::new();
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Auto] {
            let mesh = two_device_mesh(client, axis_type);
            let pending = Sharding::replicated(mesh.logical_mesh().clone(), 2).with_unreduced_axes(["x"]).unwrap();
            let r#type = ArrayType::new_static(DataType::F32, [4, 4])
                .with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])));
            let values = (0..16).map(|value| value as f32).collect::<Vec<_>>();
            let source =
                Array::from_host_buffer(&session, r#type, mesh.clone(), Some(pending), values_to_bytes(&values));
            let source = source.unwrap();
            let target = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 2)).unwrap();
            let borrowed = source.to_placement(&session, target.clone()).and_then(|placed| placed.copy_to_host_bytes());
            let consumed =
                source.clone().into_placement(&session, target).and_then(|placed| placed.copy_to_host_bytes());
            let (borrowed, consumed) = (borrowed.unwrap(), consumed.unwrap());
            assert_eq!(values_from_bytes::<f32>(&borrowed), values, "{axis_type:?}");
            outcomes.push((axis_type, borrowed == consumed));
        }
        assert_eq!(outcomes, [(MeshAxisType::Explicit, true), (MeshAxisType::Auto, true)]);
    }

    #[test]
    fn test_place_input_place_parameter() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let upload = || {
            let r#type = ArrayType::new_static(DataType::F32, [2]);
            Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(&[1f32, 2.])).unwrap()
        };

        // One tree mixes host values and arrays, and the target prefix broadcasts over both kinds of leaves.
        let array = upload();
        let buffer = Arc::downgrade(array.shards()[0].buffer().unwrap());
        let options = PlacementOptions::<PlacementRequest, PlacementRequest, bool, _>::new(
            Some(PlacementRequest::from(devices[1])),
            None,
            None,
            Some(Some(false)),
        );
        let (scalar, vector, placed) = session
            .place(
                (
                    PlaceInput::from(2.5f32),
                    PlaceInput::from(CoreArray::vector(vec![3i32, 4]).unwrap()),
                    PlaceInput::from(array),
                ),
                options,
            )
            .unwrap();
        assert_eq!(scalar.mesh().devices(), [devices[1]]);
        assert_eq!(scalar.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![2.5]);
        assert_eq!(vector.mesh().devices(), [devices[1]]);
        assert_eq!(vector.copy_to_host().unwrap().elements::<i32>().unwrap(), vec![3, 4]);
        assert_eq!(placed.mesh().devices(), [devices[1]]);
        assert_eq!(shard_values(&placed), vec![vec![1., 2.]]);
        assert!(buffer.upgrade().is_none());

        // The source prefix also broadcasts over both kinds of leaves, and only the array leaf validates it.
        let options = |source: PlacementRequest| {
            PlacementOptions::<PlacementRequest, PlacementRequest>::new(None, Some(source), None, None)
        };
        let (scalar, kept) = session
            .place((PlaceInput::from(1u8), PlaceInput::from(upload())), options(PlacementRequest::from(devices[0])))
            .unwrap();
        assert_eq!(scalar.mesh().devices(), [devices[0]]);
        assert_eq!(scalar.copy_to_host_bytes().unwrap(), vec![1u8]);
        assert_eq!(kept.mesh().devices(), [devices[0]]);
        let pinned = PlacementRequest::from(Memory::Host { pinned: true });
        let array = upload();
        let message =
            format!("`place` source memory `Host[Pinned]` does not match the input type `{}`", array.r#type());
        let error = session.place((PlaceInput::from(1u8), PlaceInput::from(array)), options(pinned)).unwrap_err();
        let (path, error) = failed_parameter(&error).unwrap();
        assert_eq!(path, "$.1");
        assert!(matches!(error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
        assert_eq!(error.to_string(), message);
    }

    #[test]
    fn test_place_input_from() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let array = Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(&[1f32, 2.])).unwrap();
        let complex = Complex::new(1.5f32, -2.);
        let low_precision = f8e4m3fn::from_f32(0.5).unwrap();

        // A host value converts into a host input and an array into a runtime input.
        assert!(matches!(PlaceInput::from(true), PlaceInput::Host(host) if host == CoreArray::scalar(true).unwrap()));
        assert!(matches!(PlaceInput::from(array.clone()), PlaceInput::Runtime(_)));

        // Every scalar that a host array can hold converts into a host value, including complex and low precision
        // scalars, which are not leaves on their own, so one tree can place them together with arrays.
        assert!(matches!(
            PlaceInput::from(complex),
            PlaceInput::Host(host) if host == CoreArray::scalar(complex).unwrap(),
        ));
        let (complex_array, low_precision_array, array) = session
            .place(
                (PlaceInput::from(complex), PlaceInput::from(low_precision), PlaceInput::from(array)),
                PlacementOptions::default(),
            )
            .unwrap();
        assert_eq!(complex_array.data_type(), DataType::C64);
        assert_eq!(complex_array.copy_to_host().unwrap().elements::<Complex<f32>>().unwrap(), vec![complex]);
        assert_eq!(low_precision_array.data_type(), DataType::F8E4M3FN);
        assert_eq!(low_precision_array.copy_to_host().unwrap().elements::<f8e4m3fn>().unwrap(), vec![low_precision]);
        assert_eq!(shard_values(&array), vec![vec![1., 2.]]);
    }

    #[test]
    fn test_array_into_unaliased() {
        let client = cpu_client(2);
        let mesh = domain_mesh(client, "x", 2);
        let session = Session::builder().client(client).build().unwrap();
        let upload = || {
            let r#type = ArrayType::new_static(DataType::F32, [2]);
            Array::from_host_buffer(&session, r#type, mesh.clone(), None, values_to_bytes(&[1f32, 2.])).unwrap()
        };
        let buffer_addresses =
            |array: &Array| array.shards().iter().map(|shard| Arc::as_ptr(shard.buffer().unwrap())).collect::<Vec<_>>();

        // An array that is the only handle of its buffers is returned as it is.
        let array = upload();
        let addresses = buffer_addresses(&array);
        let unaliased = array.into_unaliased().unwrap();
        assert_eq!(buffer_addresses(&unaliased), addresses);

        // Only the buffers that another handle shares are copied, and the others are kept.
        let array = upload();
        let addresses = buffer_addresses(&array);
        let native_addresses = native_buffer_addresses(&array);
        let held = Arc::clone(array.shards()[0].buffer().unwrap());
        let unaliased = array.into_unaliased().unwrap();
        assert!(!Arc::ptr_eq(unaliased.shards()[0].buffer().unwrap(), &held));
        assert_ne!(native_buffer_addresses(&unaliased)[0], native_addresses[0]);
        assert_eq!(buffer_addresses(&unaliased)[1], addresses[1]);
        assert_eq!(native_buffer_addresses(&unaliased)[1], native_addresses[1]);
        assert_eq!(shard_values(&unaliased), vec![vec![1., 2.]; 2]);

        // A clone shares every buffer, so all of them are copied.
        let array = upload();
        let unaliased = array.clone().into_unaliased().unwrap();
        let shares_buffers = unaliased
            .shards()
            .iter()
            .zip(array.shards())
            .map(|(unaliased_shard, shard)| Arc::ptr_eq(unaliased_shard.buffer().unwrap(), shard.buffer().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(shares_buffers, vec![false, false]);
        let (unaliased_addresses, addresses) = (native_buffer_addresses(&unaliased), native_buffer_addresses(&array));
        let shares_addresses = unaliased_addresses.iter().zip(&addresses).map(|(a, b)| a == b).collect::<Vec<_>>();
        assert_eq!(shares_addresses, vec![false, false]);
        assert_eq!(unaliased.r#type(), array.r#type());
        assert_eq!(unaliased.mesh(), array.mesh());
        assert_eq!(unaliased.sharding(), array.sharding());
        assert_eq!(shard_values(&unaliased), vec![vec![1., 2.]; 2]);

        // The copies keep the commitment of the array.
        let array = upload().with_committed(false);
        assert!(!array.clone().into_unaliased().unwrap().is_committed());
        let array = upload().with_committed(true);
        assert!(array.clone().into_unaliased().unwrap().is_committed());
    }

    #[test]
    fn test_session_place() {
        // A host value that `Session::place` uploads without a target (or with one that names only its storage) goes
        // to the first addressable device of the client, whether or not the session has a default mesh, and is a
        // choice of this process alone (i.e., locally assigned). It is uncommitted without a target, and committed in
        // the memory that a target names. The default mesh of a session is the analogue of `jax.set_mesh`,
        // under which JAX 0.12.0.1 keeps such a value on its default device, uncommitted, for a mesh of one device and
        // of two (`(Explicit,) device_put: SingleDeviceSharding(device=CpuDevice(id=0)) False`), while eager results
        // without array inputs go to the mesh, committed, as `Array::from_host_buffer` without a mesh does here.
        // Uploading onto the default mesh would treat the value of each process (e.g., its own data batch) as one
        // global value that every process supplied.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let single = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let describe = |array: Array| {
            let devices = array.mesh().devices().iter().map(Device::id).collect::<Vec<_>>();
            assert_eq!(array.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1., 2.]);
            let memory = array.r#type().memory();
            let kind = array.shards()[0].buffer().unwrap().memory().unwrap().kind().unwrap().to_string();
            let (committed, locally_assigned) = (array.is_committed(), array.is_locally_assigned());
            let committed = if committed { "committed" } else { "uncommitted" };
            let locally_assigned = if locally_assigned { "locally assigned" } else { "not locally assigned" };
            format!("devices {devices:?}, {memory} ({kind}), {committed}, {locally_assigned}")
        };
        let mut outcomes = Vec::new();
        for (name, session) in [
            ("no mesh", session.clone()),
            ("mesh on CPU 1", session.with_device_mesh(DeviceMesh::new(single, vec![devices[1]]).unwrap())),
            ("mesh on both CPUs", session.with_device_mesh(domain_mesh(client, "x", 2))),
        ] {
            let placed = session.place(CoreArray::vector(vec![1f32, 2.]).unwrap(), PlacementOptions::default());
            let pinned = PlacementOptions::to(Memory::Host { pinned: true });
            let in_place = session.place(CoreArray::vector(vec![1f32, 2.]).unwrap(), pinned);
            let r#type = ArrayType::new_static(DataType::F32, [2]);
            let uploaded = Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(&[1f32, 2.]));
            outcomes.push(format!("{name}: place {}", describe(placed.unwrap())));
            outcomes.push(format!("{name}: place in place {}", describe(in_place.unwrap())));
            outcomes.push(format!("{name}: from_host_buffer {}", describe(uploaded.unwrap())));
        }
        assert_eq!(
            outcomes.join("\n"),
            indoc! {"
                no mesh: place devices [0], Device (device), uncommitted, locally assigned
                no mesh: place in place devices [0], Host[Pinned] (pinned_host), committed, locally assigned
                no mesh: from_host_buffer devices [0], Device (device), uncommitted, locally assigned
                mesh on CPU 1: place devices [0], Device (device), uncommitted, locally assigned
                mesh on CPU 1: place in place devices [0], Host[Pinned] (pinned_host), committed, locally assigned
                mesh on CPU 1: from_host_buffer devices [1], Device (device), committed, not locally assigned
                mesh on both CPUs: place devices [0], Device (device), uncommitted, locally assigned
                mesh on both CPUs: place in place devices [0], Host[Pinned] (pinned_host), committed, locally assigned
                mesh on both CPUs: from_host_buffer devices [0, 1], Device (device), committed, not locally assigned"},
        );

        // A host scalar keeps its data type, and a target device places it on that device, committed.
        let scalar = session.place(2.5f32, PlacementOptions::default()).unwrap();
        assert_eq!(scalar.r#type().into_owned().with_sharding(None).unwrap(), ArrayType::scalar(DataType::F32));
        assert_eq!(scalar.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![2.5]);
        let flag = session.place(true, PlacementOptions::default()).unwrap();
        assert_eq!(flag.data_type(), DataType::Boolean);
        assert_eq!(flag.copy_to_host_bytes().unwrap(), vec![1u8]);
        let placed = session.place(bf16::from_f32(1.5), PlacementOptions::to(devices[1])).unwrap();
        assert_eq!(placed.mesh().devices(), [devices[1]]);
        assert!(placed.is_committed());
        assert_eq!(placed.copy_to_host_bytes().unwrap(), bf16::from_f32(1.5).to_bits().to_ne_bytes().to_vec());
    }

    #[test]
    fn test_session_place_uploads_reference_backend_arrays() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let matrix = CoreArray::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap();
        let array = session.place(matrix, PlacementOptions::default()).unwrap();
        assert_eq!(
            array.r#type().into_owned().with_sharding(None).unwrap(),
            ArrayType::new_static(DataType::F32, [2, 3]),
        );
        assert_eq!(array.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1., 2., 3., 4., 5., 6.]);

        // A token is a valid value of the reference backend, but effect tokens are not arrays, so its upload is
        // rejected before anything is allocated or uploaded, also when it is given as a `PlaceInput`.
        let token = || CoreArray::new(ArrayType::scalar(DataType::Token), Vec::new()).unwrap();
        let message =
            "tokens are created by effectful operations and cannot be constructed, uploaded, or placed as arrays";
        assert_eq!(
            session.place(token(), PlacementOptions::default()).map(|_| ()).map_err(|error| error.to_string()),
            Err(message.to_string()),
        );
        assert_eq!(
            session
                .place(PlaceInput::<Array>::Host(token()), PlacementOptions::default())
                .map(|_| ())
                .map_err(|error| error.to_string()),
            Err(message.to_string()),
        );
    }

    #[test]
    fn test_session_place_validates_host_storage() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();

        // The storage of a host array can be mutated safely, so its upload validates the encoding of its elements
        // rather than trusting it, with and without an explicit layout, and reports an invalid Boolean byte.
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        for r#type in [
            ArrayType::new_static(DataType::Boolean, [1]),
            ArrayType::new_static(DataType::Boolean, [1]).with_layout(layout),
        ] {
            let mut host = CoreArray::new(r#type, vec![1]).unwrap();
            host.storage_bytes_mut()[0] = 2;
            let error = session.place(host, PlacementOptions::default()).unwrap_err();
            assert_eq!(error.to_string(), "array element 0 has invalid `bool` byte encoding [2]");
        }
    }

    #[test]
    fn test_session_place_tracks_local_assignment() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let device = session.target().unwrap().devices()[1].device();
        let pinned = Memory::Host { pinned: true };

        // A host value uploaded with a target that names a device lives on a device that this process did not choose
        // on its own (refer to `test_session_place` for the uploads that it chooses).
        assert!(!session.place(1f32, PlacementOptions::to(device)).unwrap().is_locally_assigned());

        // An array keeps the fact without a target and with an in-place one, eagerly and when a staged in-place
        // placement executes on it, and a target that names a device clears it.
        let local = session.place(1f32, PlacementOptions::default()).unwrap();
        assert!(session.place(local.clone(), PlacementOptions::default()).unwrap().is_locally_assigned());
        assert!(session.place(local.clone(), PlacementOptions::to(pinned)).unwrap().is_locally_assigned());
        let staged =
            session.bind(PlaceOperation::new(pinned, None), Vec::new(), &[Value::Array(local.clone())]).unwrap();
        let [Value::Array(staged)] = staged.as_slice() else { panic!("placing an array must produce an array") };
        assert!(staged.is_locally_assigned());
        assert!(!session.place(local, PlacementOptions::to(device)).unwrap().is_locally_assigned());
    }

    #[test]
    fn test_session_place_applies_memory_and_layout() {
        let client = cpu_client(2);
        let mesh = domain_mesh(client, "x", 2);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let memory = Memory::Host { pinned: true };
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let values = vec![1f32, 2., 3., 4., 5., 6., 7., 8.];
        let placement = Placement::new(mesh.clone(), sharding.clone())
            .unwrap()
            .with_layout(layout.clone())
            .with_memory(memory);
        let uploaded = session
            .place(CoreArray::matrix(4, 2, values.clone()).unwrap(), PlacementOptions::to(placement.clone()))
            .unwrap();

        assert_eq!(uploaded.mesh(), &mesh);
        assert_eq!(uploaded.sharding(), &sharding);
        assert_eq!(uploaded.r#type().memory(), memory);
        assert_eq!(uploaded.layout(), Some(&layout));
        for shard in uploaded.addressable_shards() {
            let buffer = shard.buffer().unwrap();
            assert_eq!(buffer.memory().unwrap().kind().unwrap(), "pinned_host");
            assert_eq!(buffer.specification().unwrap().layout, layout.to_pjrt().unwrap());
        }
        assert_eq!(uploaded.copy_to_host().unwrap().elements::<f32>().unwrap(), values);

        // An existing leaf can change both options without changing its mesh or shard slices.
        let layout = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let target = PlacementRequest::from(
            session.placement(sharding.clone()).unwrap().with_memory(Memory::Device).with_layout(layout.clone()),
        );
        let moved = session
            .place(uploaded, PlacementOptions::<_, _>::new(Some(target), Some(placement), None, None))
            .unwrap();
        assert_eq!(moved.mesh(), &mesh);
        assert_eq!(moved.sharding(), &sharding);
        assert_eq!(moved.r#type().memory(), Memory::Device);
        assert_eq!(moved.layout(), Some(&layout));
        for shard in moved.addressable_shards() {
            let buffer = shard.buffer().unwrap();
            assert_eq!(buffer.memory().unwrap(), buffer.device().unwrap().default_memory().unwrap());
            assert_eq!(buffer.specification().unwrap().layout, layout.to_pjrt().unwrap());
        }
        assert_eq!(moved.copy_to_host().unwrap().elements::<f32>().unwrap(), values);
    }

    #[test]
    fn test_session_place_pads_short_shardings() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let logical_mesh = mesh.logical_mesh().clone();
        let split = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let placement = session.placement(split.clone()).unwrap();
        let split_matrix = Sharding::new(
            logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let vector = CoreArray::vector(vec![1f32, 2., 3., 4.]).unwrap();
        let matrix = CoreArray::matrix(4, 3, (0..12).map(|value| value as f32).collect()).unwrap();
        let halves = vec![vec![0f32, 1., 2., 3., 4., 5.], vec![6., 7., 8., 9., 10., 11.]];

        // One sharding shorter than the rank of a leaf leaves the omitted trailing dimensions replicated, so one
        // target splits a vector and a matrix along their first dimension, and a rank zero sharding replicates
        // both. The resolved shardings have the rank of each leaf.
        let (placed_vector, placed_matrix) =
            session.place((vector.clone(), matrix.clone()), PlacementOptions::to(placement.clone())).unwrap();
        assert_eq!(placed_vector.sharding(), &split);
        assert_eq!(shard_buffer_dimensions(&placed_vector), vec![vec![2], vec![2]]);
        assert_eq!(shard_values(&placed_vector), vec![vec![1., 2.], vec![3., 4.]]);
        assert!(placed_vector.is_committed());
        assert_eq!(placed_matrix.sharding(), &split_matrix);
        assert_eq!(shard_buffer_dimensions(&placed_matrix), vec![vec![2, 3], vec![2, 3]]);
        assert_eq!(shard_values(&placed_matrix), halves);
        assert!(placed_matrix.is_committed());
        let replicated = Sharding::replicated(logical_mesh.clone(), 0);
        let (replicated_vector, replicated_matrix) = session
            .place((vector.clone(), matrix.clone()), PlacementOptions::to(session.placement(replicated).unwrap()))
            .unwrap();
        assert_eq!(replicated_vector.sharding(), &Sharding::replicated(logical_mesh.clone(), 1));
        assert_eq!(shard_buffer_dimensions(&replicated_vector), vec![vec![4], vec![4]]);
        assert_eq!(replicated_matrix.sharding(), &Sharding::replicated(logical_mesh.clone(), 2));
        assert_eq!(shard_buffer_dimensions(&replicated_matrix), vec![vec![4, 3], vec![4, 3]]);
        assert_eq!(shard_values(&replicated_matrix), vec![halves.concat(); 2]);

        // Arrays resolve short targets and sources in the same way, also from a resolved placement.
        let resplit = session
            .place(
                replicated_matrix.clone(),
                PlacementOptions::<Placement, Placement>::new(
                    Some(placement.clone()),
                    Some(session.placement(Sharding::replicated(logical_mesh.clone(), 1)).unwrap()),
                    None,
                    None,
                ),
            )
            .unwrap();
        assert_eq!(resplit.mesh(), &mesh);
        assert_eq!(resplit.sharding(), &split_matrix);
        assert_eq!(shard_buffer_dimensions(&resplit), vec![vec![2, 3], vec![2, 3]]);
        assert_eq!(shard_values(&resplit), halves);
        assert!(resplit.is_committed());
        let checked = session
            .place(
                resplit.clone(),
                PlacementOptions::<PlacementRequest, Placement>::new(None, Some(placement.clone()), None, None),
            )
            .unwrap();
        assert!(Arc::ptr_eq(checked.shards()[0].buffer().unwrap(), resplit.shards()[0].buffer().unwrap()));

        // Direct placement also pads a short request when borrowing or consuming an array.
        let request = PlacementRequest::from(placement.clone());
        let borrowed = replicated_matrix.to_placement(&session, request.clone()).unwrap();
        let consumed = replicated_matrix.clone().into_placement(&session, request).unwrap();
        for output in [borrowed, consumed] {
            assert_eq!(output.sharding(), &split_matrix);
            assert_eq!(shard_buffer_dimensions(&output), vec![vec![2, 3], vec![2, 3]]);
            assert_eq!(shard_values(&output), halves);
        }

        // A traced leaf resolves a short target for its rank.
        let input_type = ArrayType::new_static(DataType::F32, [4, 3])
            .with_sharding(Sharding::replicated(logical_mesh.clone(), 2))
            .unwrap();
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile(
            |input: CompileTracer| session.place(input, PlacementOptions::to(placement)).unwrap(),
            input_type,
            &session,
            mesh.clone(),
        )
        .unwrap();
        let output = compiled.executable().execute(&session, replicated_matrix).unwrap();
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(output.sharding(), &split_matrix);
        assert_eq!(shard_buffer_dimensions(&output), vec![vec![2, 3], vec![2, 3]]);
        assert_eq!(shard_values(&output), halves);

        // A sharding with more dimensions than a leaf is still rejected, with the error of the type rule of `place`.
        let error = session.place(vector, PlacementOptions::to(session.placement(split_matrix).unwrap())).unwrap_err();
        assert!(matches!(&error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
        assert_eq!(error.to_string(), "`place` target sharding rank (2) does not match the input rank (1)");
    }

    #[test]
    fn test_session_place_selects_default_memory_for_device_targets() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let source_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[0]]).unwrap();
        let pinned = Memory::Host { pinned: true };
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]).with_memory(pinned),
            source_mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();
        assert_eq!(array.shards()[0].buffer().unwrap().memory().unwrap().kind().unwrap(), "pinned_host");
        let is_in_default_memory = |array: &Array| {
            let buffer = array.shards()[0].buffer().unwrap();
            buffer.memory().unwrap() == buffer.device().unwrap().default_memory().unwrap()
        };

        // A bare device target selects the default memory of that device, both on another device and on the
        // device that the array already lives on. JAX 0.12.0.1 on CPU does so for another device, and for the same
        // device only with `may_alias=False`, whereas under default options it raises "Memory kind mismatch with
        // xla::PjRtBuffers" for the pinned host buffer.
        for device in [devices[1], devices[0]] {
            let moved = session.place(array.clone(), PlacementOptions::to(device)).unwrap();
            assert_eq!(moved.mesh().devices(), [device]);
            assert_eq!(moved.r#type().memory(), Memory::Device);
            assert!(is_in_default_memory(&moved));
            assert_eq!(shard_values(&moved), vec![vec![1., 2.]]);
            let moved = array.to_placement(&session, device).unwrap();
            assert_eq!(moved.r#type().memory(), Memory::Device);
            assert!(is_in_default_memory(&moved));
        }

        // An explicit memory still applies, and a mesh target without a memory keeps the memory of the array.
        let moved = session
            .place(
                array.clone(),
                PlacementOptions::to(PlacementRequest::Device {
                    device: devices[1],
                    memory: Some(pinned),
                    layout: None,
                }),
            )
            .unwrap();
        assert_eq!(moved.r#type().memory(), pinned);
        assert_eq!(moved.shards()[0].buffer().unwrap().memory().unwrap().kind().unwrap(), "pinned_host");
        let target_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let target =
            PlacementRequest::from(Placement::new(target_mesh.clone(), Sharding::replicated(logical_mesh, 1)).unwrap());
        let moved = session.place(array.clone(), PlacementOptions::to(target)).unwrap();
        assert_eq!(moved.mesh(), &target_mesh);
        assert_eq!(moved.r#type().memory(), pinned);

        // A host value placed on a bare device is uploaded to the default memory of that device.
        let uploaded = session.place(3f32, PlacementOptions::to(devices[1])).unwrap();
        assert_eq!(uploaded.r#type().memory(), Memory::Device);
        assert!(is_in_default_memory(&uploaded));

        // A traced value placed on a bare device also selects the default memory of that device.
        let (value, _) = differentiate_at(array.clone())
            .vjp(|x| Ok(session.place(x, PlacementOptions::to(devices[1])).unwrap()))
            .unwrap();
        assert_eq!(value.mesh().devices(), [devices[1]]);
        assert_eq!(value.r#type().memory(), Memory::Device);
        assert!(is_in_default_memory(&value));

        // The core `Place` capability follows the same contract, eagerly and traced, because the type rule of a bare
        // device request selects that memory for every path.
        let moved = array.place(&devices[1], None).unwrap();
        assert_eq!(moved.r#type().memory(), Memory::Device);
        assert!(is_in_default_memory(&moved));
        let (value, _) = differentiate_at(array.clone()).vjp(|x| x.place(&devices[1], None)).unwrap();
        assert_eq!(value.r#type().memory(), Memory::Device);
        assert!(is_in_default_memory(&value));
    }

    #[test]
    fn test_session_place_moves_arrays_to_memory_targets() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let pinned = Memory::Host { pinned: true };
        let memory_kinds = |array: &Array| {
            array
                .shards()
                .iter()
                .map(|shard| shard.buffer().unwrap().memory().unwrap().kind().unwrap().to_string())
                .collect::<Vec<_>>()
        };
        let is_in_default_memory = |array: &Array| {
            array.shards().iter().all(|shard| {
                let buffer = shard.buffer().unwrap();
                buffer.memory().unwrap() == buffer.device().unwrap().default_memory().unwrap()
            })
        };

        // A bare memory leaves an array that is split over two devices on its mesh with its sharding and its values,
        // and moves its buffers to that memory. The result is committed, as JAX commits the result of `device_put`
        // onto a memory space.
        let sharded = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]),
            mesh.clone(),
            Some(sharding.clone()),
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        let moved = session.place(sharded, PlacementOptions::to(pinned)).unwrap();
        assert_eq!(moved.mesh(), &mesh);
        assert_eq!(moved.sharding(), &sharding);
        assert_eq!(moved.r#type().memory(), pinned);
        assert_eq!(memory_kinds(&moved), vec!["pinned_host"; 2]);
        assert_eq!(shard_buffer_dimensions(&moved), vec![vec![2], vec![2]]);
        assert_eq!(shard_values(&moved), vec![vec![1., 2.], vec![3., 4.]]);
        assert!(moved.is_committed());

        // The default memory moves it back.
        let restored = session.place(moved, PlacementOptions::to(Memory::Device)).unwrap();
        assert_eq!(restored.mesh(), &mesh);
        assert_eq!(restored.sharding(), &sharding);
        assert_eq!(restored.r#type().memory(), Memory::Device);
        assert!(is_in_default_memory(&restored));
        assert_eq!(shard_values(&restored), vec![vec![1., 2.], vec![3., 4.]]);
        assert!(restored.is_committed());

        // An uncommitted array on one device stays on that device and is committed, and `Array::to_placement` accepts
        // a bare memory as well.
        let single = session.place(CoreArray::vector(vec![1f32, 2.]).unwrap(), PlacementOptions::default()).unwrap();
        assert!(!single.is_committed());
        for moved in [
            session.place(single.clone(), PlacementOptions::to(pinned)).unwrap(),
            single.to_placement(&session, pinned).unwrap(),
        ] {
            assert_eq!(moved.mesh(), single.mesh());
            assert_eq!(moved.sharding(), single.sharding());
            assert_eq!(moved.r#type().memory(), pinned);
            assert_eq!(memory_kinds(&moved), vec!["pinned_host"]);
            assert_eq!(shard_values(&moved), vec![vec![1., 2.]]);
            assert!(moved.is_committed());
        }
    }

    #[test]
    fn test_session_place_combines_memory_targets_with_other_targets() {
        // One branch of a tree can change only the memory of its leaves while another moves its leaves to a device.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let pinned = Memory::Host { pinned: true };
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]),
            mesh.clone(),
            Some(sharding.clone()),
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        let options = PlacementOptions::to((PlacementRequest::from(pinned), PlacementRequest::from(mesh.devices()[1])));
        let (in_place, on_device) = session.place((array.clone(), array), options).unwrap();
        assert_eq!(in_place.mesh(), &mesh);
        assert_eq!(in_place.sharding(), &sharding);
        assert_eq!(in_place.r#type().memory(), pinned);
        assert_eq!(shard_values(&in_place), vec![vec![1., 2.], vec![3., 4.]]);
        assert_eq!(on_device.mesh().devices(), [mesh.devices()[1]]);
        assert_eq!(on_device.r#type().memory(), Memory::Device);
        assert_eq!(shard_values(&on_device), vec![vec![1., 2., 3., 4.]]);
    }

    #[test]
    fn test_session_place_resolves_bare_sharding_prefixes() {
        let client = cpu_client(2);
        let mesh = domain_mesh(client, "x", 2);
        let session = Session::builder().client(client).build().unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert!(matches!(
            session.place(1f32, PlacementOptions::to(sharding.clone())),
            Err(Error::Session(SessionError::MissingMesh)),
        ));

        let session = session.with_device_mesh(mesh.clone());
        let input = CoreArray::vector(vec![1f32, 2., 3., 4.]).unwrap();
        let (placed, default) = session
            .place((input, CoreArray::scalar(5f32).unwrap()), PlacementOptions::to((Some(sharding.clone()), None)))
            .unwrap();
        assert_eq!(placed.mesh(), &mesh);
        assert_eq!(placed.sharding(), &sharding);
        assert_eq!(shard_values(&placed), vec![vec![1., 2.], vec![3., 4.]]);
        assert!(!default.is_committed());

        let uploaded = session.place(5f32, PlacementOptions::default().with_source(sharding.clone())).unwrap();
        assert_eq!(shard_values(&uploaded), vec![vec![5.]]);

        let checked = session.place(placed, PlacementOptions::default().with_source(sharding)).unwrap();
        assert_eq!(shard_values(&checked), vec![vec![1., 2.], vec![3., 4.]]);
    }

    #[test]
    fn test_session_place_resolves_only_consumed_bare_shardings() {
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        let device = session.devices().unwrap()[1];
        let sharding = Sharding::replicated(LogicalMesh::single_device(), 0);

        let empty = BTreeMap::<&str, f32>::new();
        let options = PlacementOptions::to(sharding.clone()).with_source(sharding.clone());
        assert!(session.place(empty.clone(), options).unwrap().is_empty());

        let options = PlacementOptions::to(BTreeMap::from([("absent", sharding.clone())]));
        assert!(matches!(
            session.place(empty.clone(), options),
            Err(Error::Core(error)) if matches!(
                error.as_ref(),
                CoreError::Parameter(ParameterError::UnusedParameters { paths: Some(paths) })
                    if paths == &[r#"$["absent"]"#.to_string()],
            ),
        ));

        let options = PlacementOptions::to((Some(sharding.clone()), None)).with_source((Some(sharding.clone()), None));
        let (empty, host) = session.place((empty, 2f32), options).unwrap();
        assert!(empty.is_empty());
        assert_eq!(shard_values(&host), vec![vec![2.]]);

        let options = PlacementOptions::to(device).with_source(sharding.clone());
        assert!(matches!(session.place(2f32, options), Err(Error::Session(SessionError::MissingMesh))));

        let input = (PlaceInput::from(3f32), PlaceInput::from(host));
        let options = PlacementOptions::default().with_source((None, Some(sharding)));
        let error = session.place(input, options).unwrap_err();
        let (path, source) = failed_parameter(&error).unwrap();
        assert_eq!(path, "$.1");
        assert!(matches!(source, Error::Session(SessionError::MissingMesh)));
    }

    #[test]
    fn test_session_place_commits_array_leaves_with_a_target() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let uncommitted = Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(&[1f32, 2.])).unwrap();
        let options = |target: Option<PlacementRequest>, may_alias: Option<bool>| {
            PlacementOptions::new(target, Option::<PlacementRequest>::None, Option::<bool>::None, Some(may_alias))
        };
        assert!(!uncommitted.is_committed());

        // Without a target, an array leaf keeps its commitment, also when it is copied.
        assert!(!session.place(uncommitted.clone(), options(None, None)).unwrap().is_committed());
        assert!(!session.place(uncommitted.clone(), options(None, Some(false))).unwrap().is_committed());
        let committed = uncommitted.clone().with_committed(true);
        assert!(session.place(committed, options(None, Some(false))).unwrap().is_committed());

        // A target commits the result, even one that leaves the leaf where it is.
        let device = uncommitted.mesh().devices()[0];
        let placed = session.place(uncommitted.clone(), options(Some(PlacementRequest::from(device)), None)).unwrap();
        assert_eq!(placed.mesh(), uncommitted.mesh());
        assert!(placed.is_committed());
        let other_device = Device::from_pjrt(&client.addressable_devices().unwrap()[1]).unwrap();
        let placed = session.place(uncommitted, options(Some(PlacementRequest::from(other_device)), None)).unwrap();
        assert_eq!(placed.mesh().devices(), [other_device]);
        assert!(placed.is_committed());
    }

    #[test]
    fn test_session_place_completes_pending_reduction_on_single_device_target() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let source_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[0]]).unwrap();
        let replicated = Sharding::replicated(logical_mesh.clone(), 1);
        let r#type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(replicated.clone().with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let array =
            Array::from_host_buffer(&session, r#type.clone(), source_mesh, None, values_to_bytes(&[1f32, 2.])).unwrap();

        // A device names neither a mesh nor a sharding, so the array keeps its logical mesh over the target device,
        // whether it already lives on that device or not. Its pending reduction along `x`, which has size one, is
        // complete there, so the result is an ordinary replicated array to which `exp` applies. That is the type that
        // the type rule of a staged device target records.
        for device in [devices[0], devices[1]] {
            let placed = session.place(array.clone(), PlacementOptions::to(device)).unwrap();
            assert_eq!(placed.mesh(), &DeviceMesh::new(logical_mesh.clone(), vec![device]).unwrap());
            assert_eq!(placed.sharding(), &replicated);
            assert_eq!(
                placed.r#type().into_owned(),
                CoreOperation::infer_output_types(&PlaceOperation::new(device, None), &[r#type.clone()], &[])
                    .unwrap()
                    .remove(0),
            );
            assert_eq!(
                placed.r#type().as_ref(),
                &ArrayType::new_static(DataType::F32, [2]).with_sharding(replicated.clone()).unwrap(),
            );
            assert!(placed.is_committed());
            assert_eq!(shard_values(&placed.exp().unwrap()), vec![vec![1f32.exp(), 2f32.exp()]]);
        }

        // An ordinary array on one device keeps its logical mesh as well, whatever its axes are named (e.g., that of
        // the mesh that an upload without a mesh creates).
        let uploaded = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]),
            None,
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();
        let placed = session.place(uploaded.clone(), PlacementOptions::to(devices[1])).unwrap();
        assert_eq!(placed.mesh(), &DeviceMesh::new(uploaded.mesh().logical_mesh().clone(), vec![devices[1]]).unwrap());
        assert_eq!(placed.sharding(), uploaded.sharding());
        assert_eq!(placed.r#type(), uploaded.r#type());
        assert_eq!(shard_values(&placed), vec![vec![1., 2.]]);
    }

    #[test]
    fn test_session_place_combines_gathered_values_with_uploads_on_their_device() {
        // A value that a device target gathers lives on the canonical logical mesh of one device, which is also
        // the mesh of an upload without a mesh, so the two combine on that device eagerly and in a staged program
        // that is replayed on arrays, as values with `SingleDeviceSharding`s on one device do in JAX.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let device = mesh.devices()[0];
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let split_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding).unwrap();
        let split =
            Array::from_host_buffer(&session, split_type.clone(), mesh, None, values_to_bytes(&[1f32, 2., 3., 4.]))
                .unwrap();
        let upload = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]),
            None,
            None,
            values_to_bytes(&[1f32, 1., 1., 1.]),
        )
        .unwrap();
        let canonical_mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![device]).unwrap();
        assert_eq!(upload.mesh(), &canonical_mesh);

        let gathered = session.place(split.clone(), PlacementOptions::to(device)).unwrap();
        assert_eq!(gathered.mesh(), &canonical_mesh);
        let sum = gathered.add(&upload).unwrap();
        assert_eq!((sum.mesh(), shard_values(&sum)), (&canonical_mesh, vec![vec![2., 3., 4., 5.]]));
        let placed_upload = session.place(upload.clone(), PlacementOptions::to(device)).unwrap();
        let sum = gathered.add(&placed_upload).unwrap();
        assert_eq!((sum.mesh(), shard_values(&sum)), (&canonical_mesh, vec![vec![2., 3., 4., 5.]]));

        let traced: TracedProgram<(ArrayType, ArrayType), ArrayType> = trace(
            |(x, y): (ArrayTracer, ArrayTracer)| {
                let x = session.place(x, PlacementOptions::to(device)).unwrap();
                let y = session.place(y, PlacementOptions::to(device)).unwrap();
                x.add(&y).unwrap()
            },
            (split_type, upload.r#type().into_owned()),
        )
        .unwrap();
        assert_eq!(
            traced.to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}], %1:f32[4] .
                let %2:f32[4] = place [target=device(0)] %0
                    %3:f32[4] = place [target=device(0)] %1
                    %4:f32[4] = add %2 %3
                in (%4)"
            },
        );
        let inputs = (Value::Array(split), Value::Array(upload));
        let Value::Array(sum) = traced.interpret_in_context(&session, inputs).unwrap() else {
            panic!("adding two arrays must produce an array");
        };
        assert_eq!((sum.mesh(), shard_values(&sum)), (&canonical_mesh, vec![vec![2., 3., 4., 5.]]));
    }

    #[test]
    fn test_session_place_preserves_partially_addressable_array_when_target_is_absent() {
        let client = cpu_client(1);
        let local_device = client.addressable_devices().unwrap().remove(0);
        let local_device_id = local_device.id().unwrap();
        let remote_device_id = local_device_id + 1;
        let session = session_with_remote_devices(client, [Device::new(remote_device_id, 1)]);
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(local_device_id, local_device.process_index().unwrap()), Device::new(remote_device_id, 1)],
        )
        .unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let local_source_buffer = client
            .buffer(values_to_bytes::<f32>(&[0.0, 1.0]).as_slice(), BufferType::F32, [2u64], None, local_device, None)
            .unwrap();
        let source_array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let source_array = Array::from_addressable_buffers(
            &session,
            source_array_type,
            mesh.clone(),
            Some(sharding.clone()),
            vec![local_source_buffer],
        )
        .unwrap();

        // Without a target, an array that this process addresses only in part is kept as it is, with its local shard
        // and its sharding over the remote device.
        let placed = session.place(source_array.clone(), PlacementOptions::default()).unwrap();
        assert_eq!(placed.addressable_shards().count(), 1);
        assert_eq!(
            placed.sharding().visualize().unwrap().render(false),
            format!("┌─────┬─────┐\n│{local_device_id:^5}│{remote_device_id:^5}│\n└─────┴─────┘"),
        );
        let local_shard = placed.device_shard(local_device_id).unwrap();
        let source_shard = source_array.device_shard(local_device_id).unwrap();
        assert!(Arc::ptr_eq(local_shard.buffer().unwrap(), source_shard.buffer().unwrap()));
        assert_eq!(values_from_bytes::<f32>(&local_shard.copy_to_host_bytes().unwrap()), vec![0.0, 1.0]);
        assert!(placed.device_shard(remote_device_id).unwrap().buffer().is_none());
    }

    #[test]
    fn test_session_place_returns_arrays_of_receiving_session() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let meshed_session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let upload = |session: &Session| {
            let r#type = ArrayType::new_static(DataType::F32, [2]);
            Array::from_host_buffer(session, r#type, None, None, values_to_bytes(&[1f32, 2.])).unwrap()
        };
        let options =
            |target: Option<PlacementRequest>| PlacementOptions::<PlacementRequest>::new(target, None, None, None);

        // An array of another session of the same client comes back owned by the receiving session, which has a
        // default mesh that the session of the input lacks, whether the placement keeps the array where it is or not.
        let array = upload(&session);
        for target in [None, Some(PlacementRequest::from(device))] {
            let placed = meshed_session.place(array.clone(), options(target)).unwrap();
            assert!(placed.session().is_same_session(&meshed_session));
            assert_eq!(placed.session().device_mesh().unwrap(), &mesh);
        }

        // An array of another client that lives on a device with the same ID is rejected without a target, and
        // transferred onto a device target as a committed array of the receiving session.
        let other_array = upload(&Session::builder().client(other_cpu_client(1)).build().unwrap());
        assert_eq!(other_array.mesh().devices(), [device]);
        assert!(matches!(
            session.place(other_array.clone(), options(None)),
            Err(Error::Pjrt(ryft_pjrt::Error::InvalidArgument { message, .. }))
                if message == "cannot associate an array with a session of a different PJRT client",
        ));
        let placed = meshed_session.place(other_array, options(Some(PlacementRequest::from(device)))).unwrap();
        assert!(placed.session().is_same_session(&meshed_session));
        assert!(std::ptr::eq(placed.client(), client));
        assert_eq!((placed.mesh().devices(), placed.is_committed()), ([device].as_slice(), true));
        assert_eq!(values_from_bytes::<f32>(&placed.copy_to_host_bytes().unwrap()), vec![1f32, 2.]);
    }

    #[test]
    fn test_session_place_validates_source_storage_on_every_path() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let array = session.place(CoreArray::vector(vec![1f32, 2.]).unwrap(), PlacementOptions::default()).unwrap();
        let device = array.mesh().devices()[0];
        let pinned = Memory::Host { pinned: true };
        let options = |source: &PlacementRequest| {
            PlacementOptions::<PlacementRequest>::new(None, Some(source.clone()), None, None)
        };

        // A source states the storage in which a value is expected, whatever its form, so a source in another memory
        // than the memory of the array is rejected with the error of the type rule of `place` for an array leaf, for a
        // traced leaf of a staged function, and for a traced leaf of a differentiated function that wraps the array.
        let error = format!(
            "`{PLACE_OPERATION_NAME}` source memory `Host[Pinned]` does not match the input type `{}`",
            array.r#type(),
        );
        for source in [
            PlacementRequest::from(array.placement().unwrap().with_memory(pinned)),
            PlacementRequest::Device { device, memory: Some(pinned), layout: None },
            PlacementRequest::from(pinned),
        ] {
            let eager = session.place(array.clone(), options(&source));
            assert_eq!(eager.map(|_| ()).map_err(|error| error.to_string()), Err(error.clone()));
            // A staging context reports the error of the placement that it stages when tracing completes.
            let staged = RefCell::new(None);
            let traced: Result<TracedProgram<ArrayType, ArrayType>, _> = trace(
                |x: ArrayTracer| {
                    staged.replace(session.place(x.clone(), options(&source)).err().map(|error| error.to_string()));
                    x
                },
                array.r#type().into_owned(),
            );
            assert_eq!(staged.take().or(traced.err().map(|error| error.to_string())), Some(error.clone()));
            let differentiated = RefCell::new(None);
            differentiate_at(array.clone())
                .jvp(array.clone(), |x| {
                    let result = session.place(x.clone(), options(&source));
                    differentiated.replace(result.err().map(|error| error.to_string()));
                    Ok(x)
                })
                .unwrap();
            assert_eq!(differentiated.take(), Some(error.clone()));
        }

        // A source in the memory of the array is accepted, and a host leaf ignores its source.
        let placed = session.place(array.clone(), options(&PlacementRequest::from(Memory::Device))).unwrap();
        assert!(Arc::ptr_eq(placed.shards()[0].buffer().unwrap(), array.shards()[0].buffer().unwrap()));
        let uploaded = session.place(3f32, options(&PlacementRequest::from(pinned))).unwrap();
        assert_eq!(uploaded.r#type().memory(), Memory::Device);
    }

    #[test]
    fn test_session_place_validates_source_layout_against_type_on_every_path() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let session = Session::builder().client(client).build().unwrap();
        let row_major = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let rank_1 = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let r#type = ArrayType::new_static(DataType::F32, [2, 3]);
        let values = [1f32, 2., 3., 4., 5., 6.];
        let options = |layout: &Layout| {
            PlacementOptions::<PlacementRequest, PlacementRequest>::new(
                None,
                Some(PlacementRequest::InPlace { memory: None, layout: Some(layout.clone()) }),
                None,
                None,
            )
        };
        // Two batch items with the values of `array`, packed along a new leading dimension, whose type, like that of
        // every batch item of a packed value, declares no layout.
        let packed = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2, 2, 3]),
            mesh.clone(),
            None,
            values_to_bytes(&values.repeat(2)),
        )
        .unwrap();

        // Places `array` with the source layout `layout` eagerly, by the replay of a staged program, in a compiled
        // function, under forward and reverse mode differentiation, and for each batch item of `packed` under batching,
        // and returns the values (of one batch item for batching) or the error of each path.
        let place_on_every_path = |array: &Array, layout: &Layout| {
            let error = RefCell::new(None::<String>);
            let observe = |result: Result<Array, String>| {
                result.map(|array| array.copy_to_host().unwrap().elements::<f32>().unwrap())
            };
            let eager = observe(session.place(array.clone(), options(layout)).map_err(|error| error.to_string()));
            let traced: Result<TracedProgram<ArrayType, ArrayType>, _> = trace(
                |x: ArrayTracer| record_placement_error(&error, session.place(x.clone(), options(layout))).unwrap_or(x),
                array.r#type().into_owned(),
            );
            let staged = match (error.take(), traced) {
                (Some(message), _) => Err(message),
                (None, Err(error)) => Err(error.to_string()),
                (None, Ok(traced)) => match traced.interpret_in_context(&session, Value::Array(array.clone())) {
                    Ok(Value::Array(output)) => Ok(output),
                    Ok(_) => Err("placing an array must produce an array".to_string()),
                    Err(error) => Err(error.to_string()),
                },
            };
            let compiled: Result<CompiledFunction<ArrayType, ArrayType>, _> = compile(
                |x: CompileTracer| {
                    record_placement_error(&error, session.place(x.clone(), options(layout))).unwrap_or(x)
                },
                array.r#type().into_owned(),
                &session,
                mesh.clone(),
            );
            let compiled = match (error.take(), compiled) {
                (Some(message), _) => Err(message),
                (None, Err(error)) => Err(error.to_string()),
                (None, Ok(compiled)) => {
                    compiled.executable().execute(&session, array.clone()).map_err(|error| error.to_string())
                }
            };
            let jvp = differentiate_at(array.clone())
                .jvp(array.clone(), |x| record_placement_error(&error, session.place(x, options(layout))))
                .map(|(primal, _)| primal)
                .map_err(|other| error.take().unwrap_or(other.to_string()));
            let vjp = differentiate_at(array.clone())
                .vjp(|x| record_placement_error(&error, session.place(x, options(layout))))
                .map(|(primal, _)| primal)
                .map_err(|other| error.take().unwrap_or(other.to_string()));
            let batched: Result<Array, _> = batch(
                |x| record_placement_error(&error, session.place(x, options(layout))),
                packed.clone(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            );
            let batched = observe(batched.map_err(|other| error.take().unwrap_or(other.to_string()))).map(|values| {
                let (first, second) = values.split_at(values.len() / 2);
                assert_eq!(first, second);
                first.to_vec()
            });
            [eager, observe(staged), observe(compiled), observe(jvp), observe(vjp), batched]
        };

        // A source is checked against the type of a value only, so an expected layout is accepted on every path for an
        // array whose type declares no layout, whatever the physical layouts of its buffers are (here column-major).
        let buffers = client
            .addressable_devices()
            .unwrap()
            .into_iter()
            .map(|device| {
                let memory = device.default_memory().unwrap();
                let layout = Some(column_major.to_pjrt().unwrap());
                client.buffer(values_to_bytes(&values).as_slice(), BufferType::F32, [2u64, 3], None, memory, layout)
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let undeclared =
            Array::from_addressable_buffers(&session, r#type.clone(), mesh.clone(), None, buffers).unwrap();
        assert_eq!(undeclared.layout(), None);
        assert_eq!(place_on_every_path(&undeclared, &row_major), [0; 6].map(|_| Ok(values.to_vec())));

        // An expected layout that does not address the shape of the value is rejected on every path with the error of
        // the type rule of `place`, which validates the source of a batched placement against a batch item before the
        // batching rule drops its layout.
        let error = format!(
            "`{PLACE_OPERATION_NAME}` source layout is invalid: tiled layout rank 1 does not match array rank 2"
        );
        assert_eq!(place_on_every_path(&undeclared, &rank_1), [0; 6].map(|_| Err(error.clone())));

        // An expected layout that contradicts the layout that the type of an array declares is rejected on every path
        // with the error of the type rule of `place`. The batch items of `packed` declare no layout, so the batched
        // placement accepts it.
        let declared = Array::from_host_buffer(
            &session,
            r#type.with_layout(column_major.clone()),
            mesh.clone(),
            None,
            values_to_bytes(&values),
        )
        .unwrap();
        let error = format!(
            "`{PLACE_OPERATION_NAME}` source layout `{row_major}` does not match the input type `{}`",
            declared.r#type(),
        );
        assert_eq!(
            place_on_every_path(&declared, &row_major),
            [
                Err(error.clone()),
                Err(error.clone()),
                Err(error.clone()),
                Err(error.clone()),
                Err(error),
                Ok(values.to_vec())
            ]
        );

        // An expected layout that matches the layout that the type of an array declares is accepted, and the array
        // keeps its buffers.
        let checked = session.place(declared.clone(), options(&column_major)).unwrap();
        assert!(Arc::ptr_eq(checked.shards()[0].buffer().unwrap(), declared.shards()[0].buffer().unwrap()));
    }

    #[test]
    fn test_session_place_validates_source_layout_against_shape_of_array_without_buffers() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let array = Array::zero(&session, ArrayType::new_static(DataType::Zero, [3]), None, mesh.clone()).unwrap();
        let source = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap();
        let place = |source: PlacementRequest| {
            session
                .place(array.clone(), PlacementOptions::default().with_source(source))
                .map(|placed| placed.r#type().into_owned())
        };

        // A rank-1 array of the zero data type has no buffers, and an expected layout is checked against its shape by
        // the type rule of `place`, as for every other value, whether the source is a placement or a device.
        let rank_2 = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let message = format!(
            "`{PLACE_OPERATION_NAME}` source layout is invalid: tiled layout rank 2 does not match array rank 1",
        );
        assert_eq!(
            place(PlacementRequest::from(source.clone().with_layout(rank_2.clone())))
                .map_err(|error| error.to_string()),
            Err(message.clone()),
        );
        assert_eq!(
            place(PlacementRequest::Device { device: mesh.devices()[0], memory: None, layout: Some(rank_2) })
                .map_err(|error| error.to_string()),
            Err(message),
        );
        let rank_1 = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        assert_eq!(place(PlacementRequest::from(source.with_layout(rank_1))).unwrap(), array.r#type().into_owned());
    }

    #[test]
    fn test_session_place_accepts_device_source_for_array_on_that_device() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();

        // An eager upload without a mesh lives on the canonical mesh of one device, as a device target does.
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]),
            None,
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();
        assert_eq!(array.mesh().devices(), [device]);
        let checked = session.place(array.clone(), PlacementOptions::default().with_source(device)).unwrap();
        assert!(Arc::ptr_eq(checked.shards()[0].buffer().unwrap(), array.shards()[0].buffer().unwrap()));

        // A bare device source names no memory, so it also accepts an array in another memory of that device.
        let pinned = Memory::Host { pinned: true };
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]).with_memory(pinned),
            domain_mesh(client, "x", 1),
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();
        let checked = session.place(array.clone(), PlacementOptions::default().with_source(device)).unwrap();
        assert!(Arc::ptr_eq(checked.shards()[0].buffer().unwrap(), array.shards()[0].buffer().unwrap()));
        assert_eq!(checked.r#type().memory(), pinned);
    }

    #[test]
    fn test_session_place_treats_source_devices_as_informative_on_every_path() {
        // The devices that a source names are informative, as in JAX, which never compares them with where a value
        // lives. An array on the first device that is placed on the second one with a source on the third one is
        // therefore placed alike eagerly, by the replay of a staged program, and under forward and reverse mode
        // differentiation, and reverse mode sends the cotangent to the devices that the source names.
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]),
            None,
            None,
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        assert_eq!(input.mesh().devices(), [devices[0]]);
        let options = || {
            PlacementOptions::new(
                Some(PlacementRequest::from(devices[1])),
                Some(PlacementRequest::from(devices[2])),
                Option::<bool>::None,
                Option::<Option<bool>>::None,
            )
        };
        let observe =
            |array: &Array| (array.mesh().devices().to_vec(), array.r#type().into_owned(), shard_values(array));
        let expected = (vec![devices[1]], input.r#type().into_owned(), vec![vec![1., 2., 3., 4.]]);

        let eager = session.place(input.clone(), options()).unwrap();
        assert_eq!(observe(&eager), expected);

        let traced: TracedProgram<ArrayType, ArrayType> =
            trace(|x: ArrayTracer| session.place(x, options()).unwrap(), input.r#type().into_owned()).unwrap();
        assert_eq!(
            traced.to_string(),
            indoc! {"
                lambda %0:f32[4] .
                let %1:f32[4] = place [source=device(2), target=device(1)] %0
                in (%1)"},
        );
        let Value::Array(staged) = traced.interpret_in_context(&session, Value::Array(input.clone())).unwrap() else {
            panic!("placing an array must produce an array");
        };
        assert_eq!(observe(&staged), expected);

        let (primal, tangent) = differentiate_at(input.clone())
            .jvp(input.clone(), |x| Ok(session.place(x, options()).unwrap()))
            .unwrap();
        assert_eq!(observe(&primal), expected);
        assert_eq!(observe(&tangent), expected);

        let (primal, pullback) =
            differentiate_at(input.clone()).vjp(|x| Ok(session.place(x, options()).unwrap())).unwrap();
        assert_eq!(observe(&primal), expected);
        let seed = Array::from_host_buffer(
            &session,
            primal.r#type().into_owned(),
            primal.mesh().clone(),
            None,
            values_to_bytes(&[5f32, 6., 7., 8.]),
        )
        .unwrap();
        let cotangent = pullback.apply_materialized(seed).unwrap();
        assert_eq!(
            observe(&cotangent),
            (vec![devices[2]], input.r#type().cotangent().unwrap(), vec![vec![5., 6., 7., 8.]]),
        );
    }

    #[test]
    fn test_session_place_rejects_reduction_state_changes_of_relocatable_sources() {
        // An uncommitted upload with a pending minimum (and a pending sum) over the auto axis of its anonymous mesh,
        // which its type omits, is relocatable. A replicated target on the other device would complete that reduction,
        // which a placement cannot do, and a source that names only a memory must not change that through
        // `Session::place` or `Place::place`, because the relocatable type relaxes only the match of the source.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let second_device = Device::from_pjrt(&client.addressable_devices().unwrap()[1]).unwrap();
        let logical_mesh = LogicalMesh::single_device();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![second_device]).unwrap();
        let replicated = Sharding::replicated(logical_mesh, 1);
        let target = Placement::new(mesh, replicated.clone()).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let rejection =
            "`place` cannot change reduction state; use a reduction (or the corresponding collective) instead";
        for kind in [UnreducedKind::Min, UnreducedKind::Sum] {
            let sharding =
                replicated.clone().with_unreduced_axes(["device"]).unwrap().with_unreduced_kind(kind).unwrap();
            let bytes = values_to_bytes(&[1f32, 2.]);
            let input = Array::from_host_buffer(&session, r#type.clone(), None, Some(sharding), bytes).unwrap();
            assert_eq!(CoreValue::relocatable_type(&input), Some(r#type.clone()));
            let outcomes = vec![
                session
                    .place(input.clone(), PlacementOptions::to(target.clone()))
                    .map(|_| ())
                    .map_err(|error| error.to_string()),
                session
                    .place(input.clone(), PlacementOptions::to(target.clone()).with_source(Memory::Device))
                    .map(|_| ())
                    .map_err(|error| error.to_string()),
                input.place(&target, &Memory::Device).map(|_| ()).map_err(|error| error.to_string()),
            ];
            assert_eq!(outcomes, vec![Err(rejection.to_string()); 3]);
        }
    }

    #[test]
    fn test_session_place_preflights_missing_whole_shard_sources() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![devices[0]]).unwrap();
        let missing =
            Array::from_addressable_buffers(&session, ArrayType::scalar(DataType::U8), mesh, None, vec![]).unwrap();
        let device_id = devices[0].id();
        let direct = missing.to_placement(&session, devices[1]).unwrap_err();
        assert!(matches!(
            direct,
            Error::Array(error) if matches!(
                *error,
                ArrayError::MissingShardBuffer { shard_index: 0, device_id: id } if id == device_id,
            ),
        ));

        let created = session.array_statistics().created_arrays;
        let error = session
            .place((PlaceInput::from(3u8), PlaceInput::from(missing)), PlacementOptions::to(devices[1]))
            .unwrap_err();
        let (path, source) = failed_parameter(&error).unwrap();
        assert_eq!(path, "$.1");
        assert!(matches!(
            source,
            Error::Array(error) if matches!(
                **error,
                ArrayError::MissingShardBuffer { shard_index: 0, device_id: id } if id == device_id,
            ),
        ));
        assert_eq!(session.array_statistics().created_arrays, created);
    }

    #[test]
    fn test_session_place_applies_donation_and_aliasing_options() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![devices[0]],
        )
        .unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let source = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]),
            source_mesh,
            None,
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        let shares_source_buffer = |array: &Array| {
            array
                .shards()
                .iter()
                .map(|shard| Arc::ptr_eq(shard.buffer().unwrap(), source.shards()[0].buffer().unwrap()))
                .collect::<Vec<_>>()
        };
        let options = |target: Option<PlacementRequest>, donate: bool, may_alias: Option<bool>| {
            PlacementOptions::new(target, Option::<PlacementRequest>::None, Some(donate), Some(may_alias))
        };

        // Without a new placement, the result is the leaf itself unless aliasing is forbidden.
        let unchanged = session.place(source.clone(), options(None, false, None)).unwrap();
        assert_eq!(shares_source_buffer(&unchanged), vec![true]);
        let copied = session.place(source.clone(), options(None, false, Some(false))).unwrap();
        assert_eq!(copied.r#type(), source.r#type());
        assert_eq!(copied.mesh(), source.mesh());
        assert_eq!(shares_source_buffer(&copied), vec![false]);
        assert_ne!(native_buffer_addresses(&copied), native_buffer_addresses(&source));
        assert_eq!(shard_values(&copied), vec![vec![1., 2., 3., 4.]]);

        // A forbidden alias copies every shard within its own memory, so an array in host memory stays there.
        let host = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]).with_memory(Memory::Host { pinned: true }),
            source.mesh().clone(),
            None,
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        let copied = session.place(host.clone(), options(None, false, Some(false))).unwrap();
        assert_eq!(copied.r#type(), host.r#type());
        assert!(!Arc::ptr_eq(copied.shards()[0].buffer().unwrap(), host.shards()[0].buffer().unwrap()));
        assert_ne!(native_buffer_addresses(&copied), native_buffer_addresses(&host));
        assert_eq!(copied.shards()[0].buffer().unwrap().memory().unwrap().kind().unwrap(), "pinned_host");
        assert_eq!(shard_values(&copied), vec![vec![1., 2., 3., 4.]]);

        // A shard that stays on its device shares its buffer unless aliasing is forbidden.
        let target = PlacementRequest::from(Placement::new(mesh.clone(), replicated).unwrap());
        let placed = session.place(source.clone(), options(Some(target.clone()), false, None)).unwrap();
        assert_eq!(shares_source_buffer(&placed), vec![true, false]);
        assert_eq!(shard_values(&placed), vec![vec![1., 2., 3., 4.]; 2]);
        let placed = session.place(source.clone(), options(Some(target), false, Some(false))).unwrap();
        assert_eq!(shares_source_buffer(&placed), vec![false, false]);
        assert_eq!(shard_values(&placed), vec![vec![1., 2., 3., 4.]; 2]);

        // A donated leaf is consumed. This one shares its buffers with `source`, so the donation does not take effect
        // and `source` stays readable.
        let target = PlacementRequest::from(Placement::new(mesh, sharded.clone()).unwrap());
        let donated = session.place(source.clone(), options(Some(target), true, None)).unwrap();
        assert_eq!(donated.sharding(), &sharded);
        assert_eq!(shard_values(&donated), vec![vec![1., 2.], vec![3., 4.]]);
        assert_eq!(shard_values(&source), vec![vec![1., 2., 3., 4.]]);

        // Donation decides over a forbidden alias, as in JAX, where `donate=True` with `may_alias=False` selects
        // `DONATE_INPUT` and returns the buffer of an array that already lives at its target. A donated leaf is
        // therefore placed as without an aliasing option, whether or not another handle shares its buffers, and only
        // the leaf without donation above is copied.
        let shared = session.place(source.clone(), options(None, true, Some(false))).unwrap();
        assert_eq!(shares_source_buffer(&shared), vec![true]);
        let exclusive = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]),
            source.mesh().clone(),
            None,
            values_to_bytes(&[5f32, 6., 7., 8.]),
        )
        .unwrap();
        let addresses = native_buffer_addresses(&exclusive);
        let donated = session.place(exclusive, options(None, true, Some(false))).unwrap();
        assert_eq!(native_buffer_addresses(&donated), addresses);
        assert_eq!(shard_values(&donated), vec![vec![5., 6., 7., 8.]]);
    }

    #[test]
    fn test_session_place_copies_only_handle_of_leaf_that_may_not_alias() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let upload = || {
            let r#type = ArrayType::new_static(DataType::F32, [2]);
            Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(&[1f32, 2.])).unwrap()
        };
        let options = |target: Option<PlacementRequest>| {
            PlacementOptions::<PlacementRequest, PlacementRequest, bool, _>::new(target, None, None, Some(Some(false)))
        };

        // The leaf is the only handle of its buffer, which a forced copy still must not return, whether the placement
        // leaves the leaf where it is or keeps its shard on its device while adding a replica on another device.
        let replicated = PlacementRequest::from(
            Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap(),
        );
        for (target, values) in [(None, vec![vec![1f32, 2.]]), (Some(replicated), vec![vec![1., 2.]; 2])] {
            let array = upload();
            let buffer = Arc::downgrade(array.shards()[0].buffer().unwrap());
            let placed = session.place(array, options(target)).unwrap();
            assert!(buffer.upgrade().is_none());
            assert_eq!(shard_values(&placed), values);
        }
    }

    #[test]
    fn test_session_place_rejects_donated_leaf_that_may_alias() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]),
            None,
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();

        // The donation and aliasing options of every leaf are checked before any leaf is placed, and the error names
        // the first leaf that is donated while it may alias.
        let error = session
            .place(
                (array.clone(), array.clone()),
                PlacementOptions::<PlacementRequest, PlacementRequest, _, _>::new(
                    None,
                    None,
                    Some((false, true)),
                    Some(Some(true)),
                ),
            )
            .unwrap_err();
        assert!(matches!(
            &error,
            Error::Array(error) if matches!(
                error.as_ref(),
                ArrayError::DonatedArrayMayAlias { path } if path.to_string() == "$.1",
            ),
        ));
        assert_eq!(error.to_string(), "array `$.1` cannot be donated while `may_alias` is `Some(true)`");
    }

    #[test]
    fn test_session_place_admits_a_tree_of_leaves_that_can_communicate() {
        struct Wrapper(Array);
        impl Parameter for Wrapper {}
        impl PlaceableParameter<Array> for Wrapper {
            fn validate_placement(
                &self,
                session: &Session,
                target: Option<&PlacementRequest>,
                source: Option<&PlacementRequest>,
                donate: bool,
                may_alias: Option<bool>,
            ) -> Result<(), Error> {
                self.0.validate_placement(session, target, source, donate, may_alias)
            }
        }
        impl Placeable for Wrapper {
            type Output = Array;

            fn place_parameter(
                self,
                session: &Session,
                target: Option<PlacementRequest>,
                source: Option<PlacementRequest>,
                donate: bool,
                may_alias: Option<bool>,
            ) -> Result<Array, Error> {
                self.0.place_parameter(session, target, source, donate, may_alias)
            }
        }

        let client = cpu_client(1);
        let remote = Device::new(1000, 1);
        let session = Session::builder().client(client).remote_devices([remote]).build().unwrap();
        let local = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let committed = session.place(7u8, PlacementOptions::to(local)).unwrap();
        let scalar = session.place(9u8, PlacementOptions::default()).unwrap();
        let vector =
            Array::from_host_buffer(&session, ArrayType::new_static(DataType::U8, [2]), None, None, [8, 9]).unwrap();
        let mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![local]).unwrap();
        let placement = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap();
        let options = PlacementOptions::to((PlacementRequest::from(remote), PlacementRequest::from(placement)));

        // One rank would encounter the invalid second leaf. A rank with a valid second leaf must attempt admission
        // before it can submit the cross-host transfer in the first wrapper.
        let error = session.place((Wrapper(committed.clone()), Wrapper(scalar)), options.clone()).unwrap_err();
        assert_eq!(
            error.to_string(),
            "failed to place the parameter at `$.1`: \
             `place` target sharding rank (1) does not match the input rank (0)",
        );
        let error = session.place((Wrapper(committed), Wrapper(vector)), options).unwrap_err();
        assert!(matches!(error, Error::Session(SessionError::PlacementComparisonUnavailable)));
    }

    #[test]
    fn test_session_place_rejects_traced_leaves_over_arrays_of_another_client() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let other_client = other_cpu_client(2);
        let other_session = Session::builder().client(other_client).build().unwrap();
        let other_device = other_session.addressable_devices().unwrap()[1];
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2, 2]),
            None,
            None,
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        let other_tangent = Array::from_host_buffer(
            &other_session,
            ArrayType::new_static(DataType::F32, [2, 2]),
            None,
            None,
            values_to_bytes(&[5f32, 6., 7., 8.]),
        )
        .unwrap();
        let rejection = Some("cannot associate an array with a session of a different PJRT client".to_string());

        // A traced leaf over arrays of another client is rejected like those arrays are, with or without a target,
        // because the devices of the receiving session that a target names are not the devices of those arrays.
        for target in [None, Some(PlacementRequest::from(other_device))] {
            let options = || PlacementOptions::<PlacementRequest>::new(target.clone(), None, None, None);
            let error = RefCell::new(None);

            // A batched leaf wraps the packed array.
            let _: Result<Array, _> = batch(
                |x| record_placement_error(&error, other_session.place(x, options())),
                input.clone(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            );
            assert_eq!(error.take(), rejection);

            // A differentiated leaf wraps the primal array through a partially evaluated value.
            let _ = differentiate_at(input.clone())
                .vjp(|x| record_placement_error(&error, other_session.place(x, options())));
            assert_eq!(error.take(), rejection);

            // A differentiated leaf also wraps its tangent, which is checked like its primal, so a tangent of another
            // client is rejected even when the primal is an array of the receiving session.
            let _: Result<(Array, Array), _> = differentiate_at(input.clone())
                .jvp(other_tangent.clone(), |x| record_placement_error(&error, session.place(x, options())));
            assert_eq!(error.take(), rejection);
        }

        // A staged leaf wraps no arrays, so any session can place it.
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            |input: ArrayTracer| other_session.place(input, PlacementOptions::default()).unwrap(),
            ArrayType::new_static(DataType::F32, [2]),
        )
        .unwrap();
        assert_eq!(
            traced.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = place %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_session_place_rejects_traced_leaves_over_arrays_of_another_session() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let other_session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let device = mesh.devices()[0];
        let upload = |session: &Session| {
            let r#type = ArrayType::new_static(DataType::F32, [2, 2]);
            Array::from_host_buffer(session, r#type, None, None, values_to_bytes(&[1f32, 2., 3., 4.])).unwrap()
        };
        let options =
            |target: Option<PlacementRequest>| PlacementOptions::<PlacementRequest>::new(target, None, None, None);
        let rejection = Some("cannot place a traced value over arrays of another session".to_string());

        // A traced leaf places its arrays with the session that they belong to, which cannot become the receiving
        // session, so a traced leaf over arrays of another session of the same client is rejected, with or without
        // a target.
        let input = upload(&other_session);
        for target in [None, Some(PlacementRequest::from(device))] {
            let error = RefCell::new(None);
            let _: Result<Array, _> = batch(
                |x| record_placement_error(&error, session.place(x, options(target.clone()))),
                input.clone(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            );
            assert_eq!(error.take(), rejection);
            let _ = differentiate_at(input.clone())
                .vjp(|x| record_placement_error(&error, session.place(x, options(target.clone()))));
            assert_eq!(error.take(), rejection);
        }

        // Another handle of the session of the arrays (e.g., one with a default mesh) places the leaf, whose result
        // stays attached to the handle of those arrays.
        let input = upload(&session);
        let meshed_session = session.with_device_mesh(mesh);
        let output: Array = batch(
            |x| Ok(meshed_session.place(x, options(Some(PlacementRequest::from(device)))).unwrap()),
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert!(output.session().is_same_session(&session));
        assert!(matches!(output.session().device_mesh(), Err(Error::Session(SessionError::MissingMesh))));
        assert_eq!(shard_values(&output), vec![vec![1., 2., 3., 4.]]);
    }

    #[test]
    fn test_session_place_places_traced_leaves_in_compiled_functions() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let input_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        // The traced input stays traced, and its placement on the mesh of the compiled function is staged in it.
        let target = PlacementRequest::from(session.placement(sharding.clone()).unwrap());
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile(
            |input: CompileTracer| session.place(input, PlacementOptions::to(target)).unwrap(),
            input_type.clone(),
            &session,
            mesh.clone(),
        )
        .unwrap();
        assert_eq!(
            compiled.source_program().program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = place [
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                ] %0
                in (%1)"
            },
        );
        let input = Array::from_host_buffer(
            &session,
            input_type.clone(),
            mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        let output = compiled.executable().execute(&session, input).unwrap();
        assert_eq!(output.r#type().as_ref(), &input_type.with_sharding(sharding.clone()).unwrap());
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(output.sharding(), &sharding);
        assert_eq!(shard_values(&output), vec![vec![1., 2.], vec![3., 4.]]);
    }

    #[test]
    fn test_session_place_places_traced_leaves_in_place_in_compiled_functions() {
        // A traced input of a function that is compiled for a mesh of two devices has no concrete placement, and its
        // type does not say how it is split along the auto axis `x`, which only the boundary of the compiled function
        // states, so the traced function cannot name its placement, and a bare memory stages an in-place target. The
        // result lives on the devices of the input, with the distribution of the input, in that memory.
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let session = Session::builder().client(client).build().unwrap();
        let pinned = Memory::Host { pinned: true };
        let input_type = ArrayType::new_static(DataType::F32, [4]);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
            |input: CompileTracer| session.place(input, PlacementOptions::to(pinned)).unwrap(),
            input_type.clone(),
            &session,
            Options::new(mesh.clone()).with_input_shardings(vec![sharding.clone()]),
        )
        .unwrap();
        assert_eq!(
            compiled.source_program().program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:auto]>, [{}]}] .
                let %1:f32[4][sharding={mesh<['x'=2:auto]>, [{}]}]@Host[Pinned] = place [\
                    target=in_place(memory=Host[Pinned])] %0
                in (%1)"
            },
        );
        let input = Array::from_host_buffer(
            &session,
            input_type,
            mesh.clone(),
            Some(sharding.clone()),
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        let output = compiled.executable().execute(&session, input).unwrap();
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(output.sharding(), &sharding);
        assert_eq!(output.r#type().memory(), pinned);
        for shard in output.shards() {
            assert_eq!(shard.buffer().unwrap().memory().unwrap().kind().unwrap(), "pinned_host");
        }
        assert_eq!(shard_values(&output), vec![vec![1., 2.], vec![3., 4.]]);
        assert!(output.is_committed());
    }

    #[test]
    fn test_session_place_places_traced_leaves_on_sole_device_of_program_mesh() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let mesh = single_device_mesh(devices[1]);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let input_type = ArrayType::new_static(DataType::F32, [2]).with_sharding(sharding.clone()).unwrap();

        // The devices of a traced input are not known while it is traced, so a bare device stages a device target whose
        // type rule selects the default memory of the device, which keeps the sharding of the input on the mesh of the
        // function. That does not depend on the default mesh of the session, which here is either absent or another
        // mesh.
        for session in [session.clone(), session.with_device_mesh(two_device_mesh(client, MeshAxisType::Explicit))] {
            let compiled: CompiledFunction<ArrayType, ArrayType> = compile(
                |input: CompileTracer| session.place(input, PlacementOptions::to(devices[1])).unwrap(),
                input_type.clone(),
                &session,
                mesh.clone(),
            )
            .unwrap();
            assert_eq!(
                compiled.source_program().program().to_string(),
                indoc! {"
                    lambda %0:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                    let %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [target=device(1)] %0
                    in (%1)"
                },
            );

            // The output lives on that device in its default memory, and it is committed, as an explicit device
            // commits an array eagerly.
            let input =
                Array::from_host_buffer(&session, input_type.clone(), mesh.clone(), None, values_to_bytes(&[1f32, 2.]))
                    .unwrap();
            let output = compiled.executable().execute(&session, input).unwrap();
            assert_eq!(output.mesh(), &mesh);
            assert_eq!(output.sharding(), &sharding);
            assert_eq!(output.r#type().memory(), Memory::Device);
            assert!(output.is_committed());
            let buffer = output.shards()[0].buffer().unwrap().clone();
            assert_eq!(buffer.memory().unwrap(), buffer.device().unwrap().default_memory().unwrap());
            assert_eq!(shard_values(&output), vec![vec![1., 2.]]);
        }

        // The placed value keeps the sharding that the input is tracked with over the mesh of the function, so it can
        // be added to another input of that function.
        let compiled: CompiledFunction<(ArrayType, ArrayType), ArrayType> = compile(
            |(x, y): (CompileTracer, CompileTracer)| {
                session.place(x, PlacementOptions::to(devices[1])).unwrap().add(&y).unwrap()
            },
            (input_type.clone(), input_type.clone()),
            &session,
            mesh.clone(),
        )
        .unwrap();
        assert_eq!(
            compiled.source_program().program().to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}], \
                    %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %2:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [target=device(1)] %0
                    %3:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = add %2 %1
                in (%3)"
            },
        );
        let x = Array::from_host_buffer(&session, input_type.clone(), mesh.clone(), None, values_to_bytes(&[1f32, 2.]))
            .unwrap();
        let y =
            Array::from_host_buffer(&session, input_type, mesh.clone(), None, values_to_bytes(&[3f32, 4.])).unwrap();
        let output = compiled.executable().execute(&session, (x, y)).unwrap();
        assert_eq!(output.mesh(), &mesh);
        assert!(output.is_committed());
        assert_eq!(shard_values(&output), vec![vec![4., 6.]]);
    }

    #[test]
    fn test_session_place_completes_pending_reduction_of_traced_leaves_on_sole_device_of_program_mesh() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let mesh = single_device_mesh(devices[1]);
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(replicated.clone().with_unreduced_axes(["x"]).unwrap())
            .unwrap();

        // The pending reduction along `x`, which has size one, is complete on the device, so the staged device target
        // records a type without it, `exp` applies to that value, and the compiled function places its output as a
        // direct placement of the input array on the device does.
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile(
            |input: CompileTracer| session.place(input, PlacementOptions::to(devices[1])).unwrap().exp().unwrap(),
            input_type.clone(),
            &session,
            mesh.clone(),
        )
        .unwrap();
        assert_eq!(
            compiled.source_program().program().to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=1:explicit]>, [{}], unreduced={'x'}}] .
                let %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [target=device(1)] %0
                    %2:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = exp %1
                in (%2)"
            },
        );
        let input =
            Array::from_host_buffer(&session, input_type, mesh.clone(), None, values_to_bytes(&[1f32, 2.])).unwrap();
        let output = compiled.executable().execute(&session, input.clone()).unwrap();
        let direct = session.place(input, PlacementOptions::to(devices[1])).unwrap().exp().unwrap();
        assert_eq!(
            (output.mesh(), output.sharding(), output.r#type()),
            (direct.mesh(), direct.sharding(), direct.r#type())
        );
        assert_eq!((output.mesh(), output.sharding()), (&mesh, &replicated));
        assert!(output.is_committed());
        assert_eq!(shard_values(&output), vec![vec![1f32.exp(), 2f32.exp()]]);
    }

    #[test]
    fn test_session_place_places_traced_leaves_in_staged_and_jit_functions() {
        let session = Session::builder().client(cpu_client(1)).build().unwrap();
        let device = session.addressable_devices().unwrap()[0];
        let mesh = single_device_mesh(device);
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let input =
            Array::from_host_buffer(&session, input_type.clone(), mesh.clone(), None, values_to_bytes(&[1f32, 2.]))
                .unwrap();
        let memory = Memory::Host { pinned: true };
        let target = PlacementRequest::Device { device, memory: Some(memory), layout: None };

        // A function that is staged and compiled later traces its input without a concrete placement as well.
        let staged: StagedFunction<ArrayType, ArrayType> = stage(
            |input: CompileTracer| session.place(input, PlacementOptions::to(target.clone())).unwrap(),
            input_type.clone(),
            Options::new(mesh.clone()),
        )
        .unwrap();
        assert_eq!(
            staged.source_program().program().to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}]@Host[Pinned] = place [\
                    target=device(0, memory=Host[Pinned])] %0
                in (%1)"
            },
        );
        let output = session.compile_staged_function(staged).unwrap().executable().execute(&session, input.clone());
        let output = output.unwrap();
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(output.r#type().memory(), memory);
        assert!(output.is_committed());
        assert_eq!(shard_values(&output), vec![vec![1., 2.]]);

        // So does a function that is dispatched with `jit`, which traces it on its first call.
        let function: CompiledFunctionDispatcher<_, (), ArrayType, ArrayType> = jit(
            |_, input: CompileTracer| Ok(session.place(input, PlacementOptions::to(target.clone())).unwrap()),
            &session,
            mesh.clone(),
        );
        let output = function.execute((), input).unwrap();
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(output.r#type().memory(), memory);
        assert!(output.is_committed());
        assert_eq!(shard_values(&output), vec![vec![1., 2.]]);
    }

    #[test]
    fn test_session_place_defers_device_checks_of_traced_leaves_to_compilation() {
        // A traced leaf, bare or wrapped in a `PlaceInput`, stages its requests without checking their devices against
        // the session that stages it, and the session that compiles the function resolves them. A target can name a
        // device that only the compiling session has.
        let staging_session = Session::builder().client(cpu_client(1)).build().unwrap();
        let compilation_session = Session::builder().client(cpu_client(2)).build().unwrap();
        let device = compilation_session.devices().unwrap()[1];
        assert!(!staging_session.devices().unwrap().contains(&device));
        let input = compilation_session
            .place(CoreArray::vector(vec![3f32]).unwrap(), PlacementOptions::to(device))
            .unwrap();
        let staged: StagedFunction<ArrayType, (ArrayType, ArrayType)> = stage(
            |input: CompileTracer| {
                let direct = staging_session.place(input.clone(), PlacementOptions::to(device)).unwrap();
                let wrapped = staging_session.place(PlaceInput::from(input), PlacementOptions::to(device)).unwrap();
                (direct, wrapped)
            },
            input.r#type().into_owned(),
            Options::new(input.mesh().clone()),
        )
        .unwrap();
        assert_eq!(
            staged.source_program().program().to_string(),
            indoc! {"
                lambda %0:f32[1] .
                let %1:f32[1] = place [target=device(1)] %0
                    %2:f32[1] = place [target=device(1)] %0
                in (%1, %2)"
            },
        );
        let compiled = compilation_session.compile_staged_function(staged).unwrap();
        let (direct, wrapped) = compiled.executable().execute(&compilation_session, input).unwrap();
        for output in [direct, wrapped] {
            assert_eq!(output.mesh().devices(), [device]);
            assert_eq!(shard_values(&output), vec![vec![3.]]);
        }

        // The devices of a source are informative, so an eager placement accepts a source device that no session
        // knows, and a traced leaf stages it, which the compiled function then ignores.
        let session = Session::builder().client(cpu_client(1)).build().unwrap();
        let unknown = Device::new(999, 0);
        let input = session.place(CoreArray::vector(vec![1f32, 2.]).unwrap(), PlacementOptions::default()).unwrap();
        let options = || PlacementOptions::default().with_source(unknown);
        let eager = session.place(input.clone(), options()).unwrap();
        assert_eq!(shard_values(&eager), vec![vec![1., 2.]]);
        let compiled: CompiledFunction<ArrayType, (ArrayType, ArrayType)> = compile(
            |input: CompileTracer| {
                let direct = session.place(input.clone(), options()).unwrap();
                let wrapped = session.place(PlaceInput::from(input), options()).unwrap();
                (direct, wrapped)
            },
            input.r#type().into_owned(),
            &session,
            input.mesh().clone(),
        )
        .unwrap();
        assert_eq!(
            compiled.source_program().program().to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = place [source=device(999)] %0
                    %2:f32[2] = place [source=device(999)] %0
                in (%1, %2)"
            },
        );
        let (direct, wrapped) = compiled.executable().execute(&session, input).unwrap();
        assert_eq!(shard_values(&direct), vec![vec![1., 2.]]);
        assert_eq!(shard_values(&wrapped), vec![vec![1., 2.]]);
    }

    #[test]
    fn test_session_place_applies_memory_and_layout_of_devices_to_traced_leaves_in_compiled_functions() {
        let session = Session::builder().client(cpu_client(1)).build().unwrap();
        let device = session.addressable_devices().unwrap()[0];
        let mesh = single_device_mesh(device);
        let input_type = ArrayType::new_static(DataType::F32, [2, 2]);
        let memory = Memory::Host { pinned: true };
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let target = PlacementRequest::Device { device, memory: Some(memory), layout: Some(layout.clone()) };
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile(
            |input: CompileTracer| session.place(input, PlacementOptions::to(target)).unwrap(),
            input_type.clone(),
            &session,
            mesh.clone(),
        )
        .unwrap();
        assert_eq!(
            compiled.source_program().program().to_string(),
            indoc! {"
                lambda %0:f32[2, 2] .
                let %1:f32[2, 2][layout=tiled{0,1}]@Host[Pinned] = place [\
                    target=device(0, memory=Host[Pinned], layout=tiled{0,1})] %0
                in (%1)"
            },
        );
        let input =
            Array::from_host_buffer(&session, input_type, mesh.clone(), None, values_to_bytes(&[1f32, 2., 3., 4.]))
                .unwrap();
        let output = compiled.executable().execute(&session, input).unwrap();
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(output.r#type().memory(), memory);
        assert_eq!(output.layout(), Some(&layout));
        let buffer = output.shards()[0].buffer().unwrap().clone();
        assert_eq!(buffer.memory().unwrap().kind().unwrap(), "pinned_host");
        assert_eq!(buffer.specification().unwrap().layout, layout.to_pjrt().unwrap());
        assert!(output.is_committed());
        assert_eq!(output.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1., 2., 3., 4.]);
    }

    #[test]
    fn test_session_place_places_traced_leaves_in_differentiation() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let source_mesh = single_device_mesh(devices[0]);
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [3]),
            source_mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2., 3.]),
        )
        .unwrap();
        let cotangent = |value: &Array| {
            let r#type = value.r#type().into_owned();
            Array::from_host_buffer(&session, r#type, value.mesh().clone(), None, values_to_bytes(&[4f32, 5., 6.]))
                .unwrap()
        };

        // The primal moves to the target device, and the pullback moves the cotangent back to the source mesh. The
        // donation option does not apply to the traced leaf, so the input stays readable.
        let options = PlacementOptions::to(devices[1]).with_donate(true).with_may_alias(Some(false));
        let (value, pullback) =
            differentiate_at(input.clone()).vjp(|x| Ok(session.place(x, options).unwrap())).unwrap();
        assert_eq!(value.mesh().devices(), [devices[1]]);
        assert_eq!(shard_values(&value), vec![vec![1., 2., 3.]]);
        assert_eq!(shard_values(&input), vec![vec![1., 2., 3.]]);
        let input_cotangent = pullback.apply_materialized(cotangent(&value)).unwrap();
        assert_eq!(input_cotangent.mesh(), &source_mesh);
        assert_eq!(input_cotangent.r#type(), input.r#type());
        assert_eq!(shard_values(&input_cotangent), vec![vec![4., 5., 6.]]);

        // A device names a device and not a mesh, so a traced leaf over an array on that device resolves the device
        // to the mesh of the array, both as a source, to which the pullback returns the cotangent, and as a target
        // that keeps the leaf where it is.
        let options = PlacementOptions::to(devices[1]).with_source(devices[0]);
        let (value, pullback) =
            differentiate_at(input.clone()).vjp(|x| Ok(session.place(x, options).unwrap())).unwrap();
        assert_eq!(value.mesh().devices(), [devices[1]]);
        let input_cotangent = pullback.apply_materialized(cotangent(&value)).unwrap();
        assert_eq!(input_cotangent.mesh(), &source_mesh);
        assert_eq!(shard_values(&input_cotangent), vec![vec![4., 5., 6.]]);
        let (value, _) = differentiate_at(input.clone())
            .vjp(|x| Ok(session.place(x, PlacementOptions::to(devices[0])).unwrap()))
            .unwrap();
        assert_eq!(value.mesh(), &source_mesh);
        assert_eq!(shard_values(&value), vec![vec![1., 2., 3.]]);
    }

    #[test]
    fn test_session_place_places_traced_leaves_in_place_in_differentiation() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let pinned = Memory::Host { pinned: true };
        let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap();
        let array = |values: &[f32]| {
            Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), None, values_to_bytes(values)).unwrap()
        };

        // The primal and the tangent both move to the memory on the devices where they live.
        let (primal, tangent) = differentiate_at(array(&[1., 2., 3., 4.]))
            .jvp(array(&[5., 6., 7., 8.]), |x| Ok(session.place(x, PlacementOptions::to(pinned)).unwrap()))
            .unwrap();
        for (output, values) in
            [(&primal, vec![vec![1., 2.], vec![3., 4.]]), (&tangent, vec![vec![5., 6.], vec![7., 8.]])]
        {
            assert_eq!(output.mesh(), &mesh);
            assert_eq!(output.sharding(), &sharding);
            assert_eq!(output.r#type().memory(), pinned);
            assert_eq!(output.shards()[0].buffer().unwrap().memory().unwrap().kind().unwrap(), "pinned_host");
            assert_eq!(shard_values(output), values);
        }

        // The cotangent returns to the memory of the input, on the same devices.
        let (value, pullback) = differentiate_at(array(&[1., 2., 3., 4.]))
            .vjp(|x| Ok(session.place(x, PlacementOptions::to(pinned)).unwrap()))
            .unwrap();
        assert_eq!(value.r#type().memory(), pinned);
        let cotangent = pullback.apply_materialized(value).unwrap();
        assert_eq!(cotangent.mesh(), &mesh);
        assert_eq!(cotangent.sharding(), &sharding);
        assert_eq!(cotangent.r#type().memory(), Memory::Device);
        let buffer = cotangent.shards()[0].buffer().unwrap();
        assert_eq!(buffer.memory().unwrap(), buffer.device().unwrap().default_memory().unwrap());
        assert_eq!(shard_values(&cotangent), vec![vec![1., 2.], vec![3., 4.]]);
    }

    #[test]
    fn test_session_place_completes_pending_reduction_of_traced_leaves_in_differentiation() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let (source_mesh, target_mesh) = (single_device_mesh(devices[0]), single_device_mesh(devices[1]));
        let replicated = Sharding::replicated(source_mesh.logical_mesh().clone(), 1);
        let r#type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(replicated.clone().with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let upload = |values: &[f32]| {
            Array::from_host_buffer(&session, r#type.clone(), source_mesh.clone(), None, values_to_bytes(values))
                .unwrap()
        };
        let (primal, tangent) = (upload(&[1., 2.]), upload(&[3., 4.]));

        // The primal and the tangent of a traced leaf on a mesh of one device move to another device as a direct
        // placement of an array moves them, with their logical mesh over that device and without their pending
        // reduction along `x`, which has size one, so `exp` applies to them.
        let (value, tangent): (Array, Array) = differentiate_at(primal.clone())
            .jvp(tangent, |x| session.place(x, PlacementOptions::to(devices[1])).unwrap().exp())
            .unwrap();
        let direct = session.place(primal, PlacementOptions::to(devices[1])).unwrap().exp().unwrap();
        assert_eq!(
            (value.mesh(), value.sharding(), value.r#type()),
            (direct.mesh(), direct.sharding(), direct.r#type())
        );
        assert_eq!((value.mesh(), value.sharding()), (&target_mesh, &replicated));
        assert_eq!(
            (tangent.mesh(), tangent.sharding(), tangent.r#type()),
            (direct.mesh(), direct.sharding(), direct.r#type())
        );
        assert_eq!(shard_values(&value), vec![vec![1f32.exp(), 2f32.exp()]]);
        assert_eq!(shard_values(&tangent), vec![vec![3. * 1f32.exp(), 4. * 2f32.exp()]]);
    }

    #[test]
    fn test_session_place_without_target_in_differentiation() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let (first_mesh, second_mesh) = (single_device_mesh(devices[0]), single_device_mesh(devices[1]));
        let sharding = Sharding::replicated(first_mesh.logical_mesh().clone(), 1);
        let r#type = ArrayType::new_static(DataType::F32, [3]).with_sharding(sharding.clone()).unwrap();
        let input =
            Array::from_host_buffer(&session, r#type, first_mesh.clone(), None, values_to_bytes(&[1f32, 2., 3.]))
                .unwrap();
        let source = Placement::new(second_mesh.clone(), sharding).unwrap();

        // Without a target, the primal stays where it lives, and the pullback moves the cotangent to the source, which
        // is inferred from the primal when it is absent and can name another device than the one of the primal.
        let cases = [(None, &second_mesh, &first_mesh), (Some(source), &first_mesh, &second_mesh)];
        for (source, cotangent_mesh, input_cotangent_mesh) in cases {
            let options = PlacementOptions::<PlacementRequest, Placement>::new(None, source, None, None);
            let (value, pullback) =
                differentiate_at(input.clone()).vjp(|x| Ok(session.place(x, options).unwrap())).unwrap();
            assert_eq!(value.mesh(), &first_mesh);
            assert_eq!(shard_values(&value), vec![vec![1., 2., 3.]]);
            let cotangent = Array::from_host_buffer(
                &session,
                value.r#type().into_owned(),
                cotangent_mesh.clone(),
                None,
                values_to_bytes(&[4f32, 5., 6.]),
            )
            .unwrap();
            let input_cotangent = pullback.apply_materialized(cotangent).unwrap();
            assert_eq!(input_cotangent.mesh(), input_cotangent_mesh);
            assert_eq!(input_cotangent.r#type(), input.r#type());
            assert_eq!(shard_values(&input_cotangent), vec![vec![4., 5., 6.]]);
        }
    }

    #[test]
    fn test_session_place_stages_traced_leaves_without_target() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let (first_mesh, second_mesh) = (single_device_mesh(devices[0]), single_device_mesh(devices[1]));
        let sharding = Sharding::replicated(first_mesh.logical_mesh().clone(), 1);
        let r#type = ArrayType::new_static(DataType::F32, [2]).with_sharding(sharding.clone()).unwrap();
        let source = Placement::new(second_mesh, sharding).unwrap();
        let options = || PlacementOptions::<PlacementRequest, Placement>::new(None, Some(source.clone()), None, None);

        // Without a target, the placement is staged without one and with the source, which only the transposition of
        // the placement uses, so replaying it on an array that lives on the first device leaves the array there, even
        // though the source names the second device.
        let traced: TracedProgram<ArrayType, ArrayType> =
            trace(|input: ArrayTracer| session.place(input, options()).unwrap(), r#type.clone()).unwrap();
        assert_eq!(
            traced.to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [
                    source={mesh<['x'=1:explicit]>, devices=[1], sharding={mesh<['x'=1:explicit]>, [{}]}},
                ] %0
                in (%1)"
            },
        );
        let input =
            Array::from_host_buffer(&session, r#type.clone(), first_mesh.clone(), None, values_to_bytes(&[1f32, 2.]))
                .unwrap();
        let Value::Array(output) = traced.interpret_in_context(&session, Value::Array(input)).unwrap() else {
            panic!("placing an array must produce an array");
        };
        assert_eq!(output.mesh(), &first_mesh);
        assert_eq!(shard_values(&output), vec![vec![1., 2.]]);

        // The pullback places the cotangent on the source, with the memory of the primal and without a source of its
        // own, because the forward placement has no target.
        let traced: TracedProgram<(ArrayType, ArrayType), ArrayType> = trace(
            |(input, cotangent): (ArrayTracer, ArrayTracer)| {
                let (_, pullback) = differentiate_at(input).vjp(|x| Ok(session.place(x, options()).unwrap())).unwrap();
                pullback.apply_materialized(cotangent).unwrap()
            },
            (r#type.clone(), r#type),
        )
        .unwrap();
        assert_eq!(
            traced.to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}], \
                    %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %2:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [
                    target={mesh<['x'=1:explicit]>, devices=[1], \
                        sharding={mesh<['x'=1:explicit]>, [{}]}, memory=Device},
                    may_alias=false,
                ] %1
                in (%2)"
            },
        );
    }

    #[test]
    fn test_session_place_stages_device_source_of_traced_leaves_for_pullback() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let (first_mesh, second_mesh) = (single_device_mesh(devices[0]), single_device_mesh(devices[1]));
        let r#type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(first_mesh.logical_mesh().clone(), 1))
            .unwrap();
        let options = || PlacementOptions::to(devices[1]).with_source(devices[0]);
        let upload = |values: &[f32], mesh: &DeviceMesh| {
            let array = Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), None, values_to_bytes(values));
            Value::Array(array.unwrap())
        };

        // The devices of a traced input are not known, so a bare device source stays a device, and the pullback moves
        // the cotangent from the target back onto it over the logical mesh of the input, as JAX's transposition of
        // `device_put` returns it to that device. Interpreting the pullback on an input on the first device and a
        // cotangent on the second one returns the gradient on the first device.
        let traced: TracedProgram<(ArrayType, ArrayType), ArrayType> = trace(
            |(input, cotangent): (ArrayTracer, ArrayTracer)| {
                let (_, pullback) = differentiate_at(input).vjp(|x| Ok(session.place(x, options()).unwrap())).unwrap();
                pullback.apply_materialized(cotangent).unwrap()
            },
            (r#type.clone(), r#type.clone()),
        )
        .unwrap();
        assert_eq!(
            traced.to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}], \
                    %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %2:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [
                    inferred_source=device(1),
                    target={mesh<['x'=1:explicit]>, devices=[0], \
                        sharding={mesh<['x'=1:explicit]>, [{}]}, memory=Device},
                    may_alias=false,
                ] %1
                in (%2)"
            },
        );
        let inputs = (upload(&[1., 2.], &first_mesh), upload(&[3., 4.], &second_mesh));
        let Value::Array(gradient) = traced.interpret_in_context(&session, inputs).unwrap() else {
            panic!("placing an array must produce an array");
        };
        assert_eq!(gradient.mesh(), &first_mesh);
        assert_eq!(gradient.r#type().as_ref(), &r#type);
        assert_eq!(shard_values(&gradient), vec![vec![3., 4.]]);

        // Transposing the pullback again swaps the two devices back, so the cotangent of the gradient moves onto the
        // second device, which is the target of the forward placement.
        let traced: TracedProgram<(ArrayType, ArrayType), ArrayType> = trace(
            |(input, cotangent): (ArrayTracer, ArrayTracer)| {
                let (_, pullback) = differentiate_at((input, cotangent.clone()))
                    .vjp(|(x, cotangent)| {
                        let (_, pullback) = differentiate_at(x).vjp(|x| Ok(session.place(x, options()).unwrap()))?;
                        pullback.apply_materialized(cotangent)
                    })
                    .unwrap();
                pullback.apply_materialized(cotangent).unwrap().1
            },
            (r#type.clone(), r#type.clone()),
        )
        .unwrap();
        assert_eq!(
            traced.to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}], \
                    %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %2:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [
                    inferred_source={mesh<['x'=1:explicit]>, devices=[0], \
                        sharding={mesh<['x'=1:explicit]>, [{}]}},
                    target={mesh<['x'=1:explicit]>, devices=[1], \
                        sharding={mesh<['x'=1:explicit]>, [{}]}, memory=Device},
                    may_alias=false,
                ] %1
                in (%2)"
            },
        );
        let inputs = (upload(&[1., 2.], &first_mesh), upload(&[5., 6.], &first_mesh));
        let Value::Array(cotangent) = traced.interpret_in_context(&session, inputs).unwrap() else {
            panic!("placing an array must produce an array");
        };
        assert_eq!(cotangent.mesh(), &second_mesh);
        assert_eq!(cotangent.r#type().as_ref(), &r#type);
        assert_eq!(shard_values(&cotangent), vec![vec![5., 6.]]);
    }

    #[test]
    fn test_session_place_compiles_traced_leaves_without_target() {
        let session = Session::builder().client(cpu_client(1)).build().unwrap();
        let device = session.addressable_devices().unwrap()[0];
        let mesh = single_device_mesh(device);
        let input_type = ArrayType::new_static(DataType::F32, [2]);

        // A bare device does not determine a placement for a traced leaf whose devices are not known, so the source
        // stays a device, which only transposition uses, and the placement without a target compiles and leaves the
        // value where it is.
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile(
            |input: CompileTracer| session.place(input, PlacementOptions::default().with_source(device)).unwrap(),
            input_type.clone(),
            &session,
            mesh.clone(),
        )
        .unwrap();
        assert_eq!(
            compiled.source_program().program().to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = place [source=device(0)] %0
                in (%1)"
            },
        );
        let input =
            Array::from_host_buffer(&session, input_type, mesh.clone(), None, values_to_bytes(&[1f32, 2.])).unwrap();
        let output = compiled.executable().execute(&session, input).unwrap();
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(shard_values(&output), vec![vec![1., 2.]]);
    }

    #[test]
    fn test_session_place_places_traced_leaves_in_batching() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let device = session.addressable_devices().unwrap()[1];
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2, 3]),
            None,
            None,
            values_to_bytes(&[1f32, 2., 3., 4., 5., 6.]),
        )
        .unwrap();

        // The target resolves for the rank of each batch item, and the packed result lives on the target device,
        // committed to it, with the type of the input.
        let output: Array = batch(
            |x| Ok(session.place(x, PlacementOptions::to(device)).unwrap()),
            input.clone(),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(output.r#type(), input.r#type());
        assert_eq!(output.mesh().devices(), [device]);
        assert!(output.is_committed());
        assert_eq!(shard_values(&output), vec![vec![1., 2., 3., 4., 5., 6.]]);
    }

    #[test]
    fn test_session_place_places_traced_leaves_in_batching_on_their_own_placement() {
        let client = cpu_client(2);
        let mesh = domain_mesh(client, "x", 2);
        let session = Session::builder().client(client).build().unwrap();
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2, 3]),
            mesh.clone(),
            Some(sharding.clone()),
            values_to_bytes(&[1f32, 2., 3., 4., 5., 6.]),
        )
        .unwrap();

        // Each device holds one batch item, which the placement of a batch item records as the sharding of its batch,
        // so placing every batch item on its own placement leaves each of them on its device.
        let output: Array = batch(
            |x| {
                let placement = x.placement().unwrap();
                assert_eq!(placement.batch_shardings(), [ShardingDimension::sharded(["x"])]);
                Ok(session.place(x, PlacementOptions::<Placement>::new(Some(placement), None, None, None)).unwrap())
            },
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(output.sharding(), &sharding);
        assert_eq!(shard_values(&output), vec![vec![1., 2., 3.], vec![4., 5., 6.]]);
    }

    #[test]
    fn test_session_place_copies_traced_leaves_that_may_not_alias() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2, 2]),
            None,
            None,
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        let shares_input_buffer =
            |array: &Array| Arc::ptr_eq(array.shards()[0].buffer().unwrap(), input.shards()[0].buffer().unwrap());
        let options = |may_alias: Option<bool>| {
            PlacementOptions::<PlacementRequest, PlacementRequest, bool, _>::new(None, None, None, Some(may_alias))
        };

        // A placement without a target and a source that must not alias its input is still staged, so replaying it
        // while the caller holds the input returns a copy of the buffer of the input.
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            |input: ArrayTracer| session.place(input, options(Some(false))).unwrap(),
            input.r#type().into_owned(),
        )
        .unwrap();
        assert_eq!(
            traced.to_string(),
            indoc! {"
                lambda %0:f32[2, 2] .
                let %1:f32[2, 2] = place [may_alias=false] %0
                in (%1)"
            },
        );
        let Value::Array(output) = traced.interpret_in_context(&session, Value::Array(input.clone())).unwrap() else {
            panic!("placing an array must produce an array");
        };
        assert!(!shares_input_buffer(&output));
        assert_ne!(native_buffer_addresses(&output), native_buffer_addresses(&input));
        assert_eq!(output.r#type(), input.r#type());
        assert_eq!(shard_values(&output), vec![vec![1., 2., 3., 4.]]);

        // The caller keeps a live handle of the input, so a placement that leaves each batch item where it is returns
        // the buffer of the input by default and a copy of it when the result must not alias the input.
        for (may_alias, shares_buffer) in [(None, true), (Some(false), false)] {
            let output: Array = batch(
                |x| Ok(session.place(x, options(may_alias)).unwrap()),
                input.clone(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
            assert_eq!(shares_input_buffer(&output), shares_buffer);
            assert_eq!(output.r#type(), input.r#type());
            assert_eq!(shard_values(&output), vec![vec![1., 2., 3., 4.]]);
        }

        // The primal of a differentiated placement follows the same rule.
        for (may_alias, shares_buffer) in [(None, true), (Some(false), false)] {
            let (value, _) =
                differentiate_at(input.clone()).vjp(|x| Ok(session.place(x, options(may_alias)).unwrap())).unwrap();
            assert_eq!(shares_input_buffer(&value), shares_buffer);
            assert_eq!(value.r#type(), input.r#type());
            assert_eq!(shard_values(&value), vec![vec![1., 2., 3., 4.]]);
        }
    }

    #[test]
    fn test_session_place_lifts_host_leaves_and_arrays_of_traced_trees() {
        // `jax.jit(lambda x: jax.device_put((x, 2.0, np.ones(3), a), jax.devices()[1]))` stages its host leaves and the
        // concrete array `a` that it closes over as values that its `device_put` places like `x` (JAX 0.12.0.1 accepts
        // the tree in one call), and a tree of `PlaceInput`s over traced leaves does the same in a compiled and in a
        // staged function. The same tree over runtime arrays uploads its host leaves.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let mesh = single_device_mesh(devices[1]);
        let input_type = ArrayType::new_static(DataType::F32, [])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let scalar = || CoreArray::scalar(2f32).unwrap();
        let vector = || CoreArray::vector(vec![1f32; 3]).unwrap();
        let captured_type = ArrayType::new_static(DataType::F32, [2]);
        let captured = Array::from_host_buffer(&session, captured_type, None, None, values_to_bytes(&[3f32, 4.]));
        let captured = captured.unwrap();
        let place = |x: CompileTracer| {
            let tree = (
                PlaceInput::from(x),
                PlaceInput::Host(scalar()),
                PlaceInput::Host(vector()),
                PlaceInput::Concrete(captured.clone()),
            );
            session.place(tree, PlacementOptions::to(devices[1])).unwrap()
        };
        let rendering = indoc! {"
            lambda %0:f32[][sharding={mesh<['x'=1:explicit]>, []}] .
            let %1:f32[][sharding={mesh<['x'=1:explicit]>, []}] = place [target=device(1)] %0
                %2:f32[] = const session:f32[]
                %3:f32[] = place [target=device(1)] %2
                %4:f32[3] = const session:f32[3]
                %5:f32[3] = place [target=device(1)] %4
                %6:f32[2] = const session:f32[2]
                %7:f32[2] = place [target=device(1)] %6
            in (%1, %3, %5, %7)"
        };
        let check = |(x, scalar, vector, captured): (Array, Array, Array, Array)| {
            for output in [&x, &scalar, &vector, &captured] {
                assert_eq!(output.mesh().devices(), [devices[1]]);
                assert!(output.is_committed());
            }
            assert_eq!(
                (shard_values(&x), shard_values(&scalar), shard_values(&vector), shard_values(&captured)),
                (vec![vec![1.]], vec![vec![2.]], vec![vec![1., 1., 1.]], vec![vec![3., 4.]]),
            );
        };
        let input = Array::from_host_buffer(&session, input_type.clone(), mesh.clone(), None, values_to_bytes(&[1f32]))
            .unwrap();

        type Outputs = (ArrayType, ArrayType, ArrayType, ArrayType);
        let compiled: CompiledFunction<ArrayType, Outputs> =
            compile(place, input_type.clone(), &session, mesh.clone()).unwrap();
        assert_eq!(compiled.source_program().program().to_string(), rendering);
        check(compiled.executable().execute(&session, input.clone()).unwrap());

        let staged: StagedFunction<ArrayType, Outputs> = stage(place, input_type, Options::new(mesh.clone())).unwrap();
        assert_eq!(staged.source_program().program().to_string(), rendering);
        check(
            session
                .compile_staged_function(staged)
                .unwrap()
                .executable()
                .execute(&session, input.clone())
                .unwrap(),
        );

        let tree = (
            PlaceInput::from(input),
            PlaceInput::from(2f32),
            PlaceInput::from(vector()),
            PlaceInput::Concrete(captured),
        );
        check(session.place(tree, PlacementOptions::to(devices[1])).unwrap());
    }

    #[test]
    fn test_session_place_validates_traced_leaves_before_uploading_host_leaves() {
        // The rank one sharding of the target of the traced leaf does not match its rank, so the call fails before it
        // uploads the host leaf of the same tree, and the staged function holds no placement.
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let rank_one = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let error = RefCell::new(None);
        let before = session.array_statistics();
        let staged: StagedFunction<ArrayType, ArrayType> = stage(
            |scalar: CompileTracer| {
                let tree = (PlaceInput::Host(CoreArray::scalar(2f32).unwrap()), PlaceInput::from(scalar.clone()));
                let options = PlacementOptions::to((None, Some(rank_one.clone())));
                *error.borrow_mut() = session.place(tree, options).err();
                scalar
            },
            ArrayType::scalar(DataType::F32),
            Options::new(mesh),
        )
        .unwrap();
        assert_eq!(
            staged.source_program().program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                in (%0)"
            },
        );
        let error = error.into_inner().unwrap();
        let (path, source) = failed_parameter(&error).unwrap();
        assert_eq!(path, "$.1");
        assert!(matches!(source, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
        assert_eq!(source.to_string(), "`place` target sharding rank (1) does not match the input rank (0)");
        assert_eq!(session.array_statistics(), before);
    }

    #[test]
    fn test_session_place_broadcasts_options_over_host_leaves_of_traced_trees() {
        // Each host leaf of a tree of traced leaves is placed with the options of its position in the option trees, as
        // `jax.device_put((x, 2.0, np.ones(3)), (None, device, None))` gives each leaf its own device inside `jit`.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let mesh = single_device_mesh(devices[1]);
        let input_type = ArrayType::new_static(DataType::F32, [])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 0))
            .unwrap();
        let unset = None::<bool>;
        let options =
            PlacementOptions::to((None, Some(devices[1]), None)).with_may_alias((unset, unset, (Some(false), unset)));
        type Outputs = (ArrayType, ArrayType, (ArrayType, ArrayType));
        let compiled: CompiledFunction<(ArrayType, ArrayType), Outputs> = compile(
            |(x, y): (CompileTracer, CompileTracer)| {
                let tree = (
                    PlaceInput::from(x),
                    PlaceInput::Host(CoreArray::scalar(2f32).unwrap()),
                    (PlaceInput::Host(CoreArray::vector(vec![1f32; 3]).unwrap()), PlaceInput::from(y)),
                );
                session.place(tree, options).unwrap()
            },
            (input_type.clone(), input_type.clone()),
            &session,
            mesh.clone(),
        )
        .unwrap();
        assert_eq!(
            compiled.source_program().program().to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=1:explicit]>, []}], \
                    %1:f32[][sharding={mesh<['x'=1:explicit]>, []}] .
                let %2:f32[][sharding={mesh<['x'=1:explicit]>, []}] = place %0
                    %3:f32[] = const session:f32[]
                    %4:f32[] = place [target=device(1)] %3
                    %5:f32[3] = const session:f32[3]
                    %6:f32[3] = place [may_alias=false] %5
                    %7:f32[][sharding={mesh<['x'=1:explicit]>, []}] = place %1
                in (%2, %4, %6, %7)"
            },
        );
        let upload = |value: f32| {
            let bytes = values_to_bytes(&[value]);
            Array::from_host_buffer(&session, input_type.clone(), mesh.clone(), None, bytes).unwrap()
        };
        let (x, scalar, (vector, y)) = compiled.executable().execute(&session, (upload(1.), upload(3.))).unwrap();
        for output in [&x, &scalar, &vector, &y] {
            assert_eq!(output.mesh().devices(), [devices[1]]);
            assert!(output.is_committed());
        }
        assert_eq!(
            (shard_values(&x), shard_values(&scalar), shard_values(&vector), shard_values(&y)),
            (vec![vec![1.]], vec![vec![2.]], vec![vec![1., 1., 1.]], vec![vec![3.]]),
        );
    }

    #[test]
    fn test_session_place_rejects_host_leaves_of_traced_trees_without_one_trace() {
        let session = Session::builder().client(cpu_client(1)).build().unwrap();
        let device = session.addressable_devices().unwrap()[0];
        let mesh = single_device_mesh(device);
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let message = |error: Option<Error>| match error {
            Some(Error::Pjrt(ryft_pjrt::Error::InvalidArgument { message, .. })) => message,
            error => panic!("expected an invalid argument error, got {error:?}"),
        };

        // A host leaf joins the trace of the traced leaves of its tree, so they must belong to one trace, whereas JAX
        // lifts the input of an enclosing function into the trace of a nested one.
        let error = RefCell::new(None);
        let inner_rendering = RefCell::new(None);
        let outer: CompiledFunction<ArrayType, ArrayType> = compile(
            |x: CompileTracer| {
                let inner: CompiledFunction<ArrayType, ArrayType> = compile(
                    |y: CompileTracer| {
                        let host = PlaceInput::Host(CoreArray::scalar(2f32).unwrap());
                        let tree = (PlaceInput::from(x.clone()), PlaceInput::from(y.clone()), host);
                        *error.borrow_mut() = session.place(tree, PlacementOptions::to(device)).err();
                        y
                    },
                    input_type.clone(),
                    &session,
                    mesh.clone(),
                )
                .unwrap();
                inner_rendering.replace(Some(inner.source_program().program().to_string()));
                x
            },
            input_type.clone(),
            &session,
            mesh.clone(),
        )
        .unwrap();
        assert_eq!(message(error.take()), "cannot place traced values of different traces in one call");
        let rendering = indoc! {"
            lambda %0:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] .
            in (%0)"
        };
        assert_eq!(inner_rendering.take().unwrap(), rendering);
        assert_eq!(outer.source_program().program().to_string(), rendering);

        // A host leaf without traced leaves in its tree has no trace to join.
        let host = PlaceInput::<CompileTracer>::Host(CoreArray::scalar(2f32).unwrap());
        assert_eq!(
            message(session.place(host, PlacementOptions::to(device)).err()),
            "cannot place a host value or a runtime array as a traced value without a traced value in the same call",
        );
    }

    #[test]
    fn test_session_place_checks_arrays_of_traced_trees_for_traces_that_preserve_capture_placement() {
        // A plain trace preserves the placement of its captures, so a concrete array that is replicated over two
        // devices joins it with that sharding, which a source on one device does not describe. The placement is
        // rejected before the array enters the trace, so a caller that handles the error can still finish its trace.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.devices().unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
        let target = Placement::new(mesh, Sharding::replicated(logical_mesh, 0)).unwrap();
        let captured: Array = session.place(5f32, PlacementOptions::to(target)).unwrap();
        let input_type = ArrayType::scalar(DataType::F32);
        let (_, program) = DomainTracingContext::<Session, Value>::trace_with_context_and_named_axes(
            |context, input: StatefulCompileTracer| {
                let tree = (
                    PlaceInput::Runtime(CompileTracer::new(input.clone(), input_type.clone())),
                    PlaceInput::Concrete(captured.clone()),
                );
                let options = PlacementOptions::default().with_source((None, Some(devices[0])));
                assert_eq!(
                    session.place(tree, options).map(|_| ()).map_err(|error| error.to_string()),
                    Err("failed to place the parameter at `$.1`: `place` device source names one device, but the \
                         input type `f32[][sharding={mesh<['x'=2:explicit]>, []}]` is tracked over 2 devices"
                        .to_string()),
                );
                assert_eq!(context.captures().borrow().len(), 0);
                Ok(input)
            },
            Type::Array(input_type.clone()),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[] .
                in (%0)"
            },
        );
    }

    #[test]
    fn test_session_place_broadcasts_root_placement_over_array_tuple() {
        // One placement at the root moves both arrays of a tuple from a mesh of one device to a split over two.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![devices[0]],
        )
        .unwrap();
        let source_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(source_mesh.logical_mesh().clone(), 1))
            .unwrap();
        let upload = |values: &[f32]| {
            Array::from_host_buffer(&session, source_type.clone(), source_mesh.clone(), None, values_to_bytes(values))
                .unwrap()
        };
        let target_mesh = domain_mesh(client, "x", 2);
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let target = PlacementRequest::from(Placement::new(target_mesh.clone(), target_sharding.clone()).unwrap());
        let (first, second) = session
            .place((upload(&[0., 1., 2., 3.]), upload(&[10., 11., 12., 13.])), PlacementOptions::to(target))
            .unwrap();
        assert_eq!((first.mesh(), first.sharding()), (&target_mesh, &target_sharding));
        assert_eq!((second.mesh(), second.sharding()), (&target_mesh, &target_sharding));
        assert_eq!(shard_values(&first), vec![vec![0., 1.], vec![2., 3.]]);
        assert_eq!(shard_values(&second), vec![vec![10., 11.], vec![12., 13.]]);
    }

    #[test]
    fn test_session_place_repeats_root_options_over_large_inputs() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let array = session.place(2.5f32, PlacementOptions::to(devices[0])).unwrap();
        let array_count = 1_000;
        let input = vec![array.clone(); array_count];

        // Options at the root give every leaf what options that repeat them for every leaf give it.
        let root_options = PlacementOptions::new(Some(devices[1]), Some(devices[0]), Some(false), Some(Some(false)));
        let repeated_options = PlacementOptions::new(
            Some(vec![devices[1]; array_count]),
            Some(vec![devices[0]; array_count]),
            Some(vec![false; array_count]),
            Some(vec![Some(false); array_count]),
        );
        let placed = session.place(input.clone(), root_options).unwrap();
        let expected = session.place(input.clone(), repeated_options).unwrap();
        assert_eq!(placed.len(), array_count);
        for (placed, expected) in placed.iter().zip(&expected) {
            assert_eq!(placed.mesh().devices(), [devices[1]]);
            assert_eq!(placed.mesh(), expected.mesh());
            assert_eq!(placed.sharding(), expected.sharding());
            assert_eq!(placed.r#type(), expected.r#type());
            assert!(placed.is_committed());
            assert_eq!(shard_values(placed), vec![vec![2.5]]);
        }

        // The checks of root options that do not depend on paths still run, before and while each leaf is placed.
        let options =
            PlacementOptions::<PlacementRequest, PlacementRequest, _, _>::new(None, None, Some(true), Some(Some(true)));
        assert_eq!(
            session.place(input.clone(), options).map(|_| ()).map_err(|error| error.to_string()),
            Err("array `$[0]` cannot be donated while `may_alias` is `Some(true)`".to_string()),
        );
        let error = session.place(input, PlacementOptions::default().with_source(Memory::Host { pinned: true }));
        let error = error.map(|_| ()).unwrap_err();
        let (path, error) = failed_parameter(&error).unwrap();
        assert_eq!(path, "$[0]");
        assert!(matches!(error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
    }

    #[test]
    fn test_session_place_keeps_omitted_options_uniform_over_large_inputs() {
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        let device = session.addressable_devices().unwrap()[1];
        let array = session.place(2.5f32, PlacementOptions::to(device)).unwrap();
        let array_count = 4_000;
        let input = vec![array.clone(); array_count];

        // Already resident leaves without any option stay where they are.
        let placed = session.place(input.clone(), PlacementOptions::default()).unwrap();
        assert_eq!(placed.len(), array_count);
        for placed in &placed {
            assert_eq!(placed.mesh(), array.mesh());
            assert_eq!(placed.sharding(), array.sharding());
            assert!(placed.is_committed());
        }
        assert_eq!(shard_values(&placed[array_count - 1]), vec![vec![2.5]]);

        // The path of a leaf is still reported when one of its options is invalid.
        let mut donate = vec![false; array_count];
        donate[array_count - 1] = true;
        let options = PlacementOptions::default().with_donate(donate).with_may_alias(Some(true));
        assert_eq!(
            session.place(input, options).map(|_| ()).map_err(|error| error.to_string()),
            Err("array `$[3999]` cannot be donated while `may_alias` is `Some(true)`".to_string()),
        );
    }

    #[test]
    fn test_session_place_matches_core_place_over_borrowed_map_keys() {
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        let devices = session.devices().unwrap();
        let left = String::from("left");
        let right = String::from("right");
        let input = BTreeMap::from([
            (left.as_str(), session.place(3f32, PlacementOptions::default()).unwrap()),
            (right.as_str(), session.place(7f32, PlacementOptions::default()).unwrap()),
        ]);
        let targets = BTreeMap::from([(left.as_str(), devices[1]), (right.as_str(), devices[0])]);
        let exact = input.place(&targets, None).unwrap();
        let owned_targets = BTreeMap::from([(left.clone(), devices[1]), (right.clone(), devices[0])]);
        let prefix = session.place(input, PlacementOptions::to(owned_targets)).unwrap();
        for output in [exact, prefix] {
            for (key, device, value) in [(left.as_str(), devices[1], 3f32), (right.as_str(), devices[0], 7f32)] {
                let array = &output[key];
                assert_eq!(array.mesh().devices(), &[device]);
                assert!(array.is_committed());
                assert_eq!(array.copy_to_host().unwrap().elements::<f32>().unwrap(), [value]);
            }
        }
    }

    #[test]
    fn test_session_place_broadcasts_optional_placement_prefixes() {
        let client = cpu_client(2);
        let mesh = domain_mesh(client, "x", 2);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let values = vec![1f32, 2., 3., 4.];
        let host = CoreArray::vector(values.clone()).unwrap();

        // An explicit sharding uses the session mesh. The None prefix sends both leaves to the first local device.
        let (partitioned, [local, other_local]) = session
            .place(
                (host.clone(), [host.clone(), host]),
                PlacementOptions::<_, PlacementRequest>::new(
                    Some((Some(PlacementRequest::from(session.placement(sharded.clone()).unwrap())), None)),
                    None,
                    None,
                    None,
                ),
            )
            .unwrap();
        assert_eq!(partitioned.mesh(), &mesh);
        assert_eq!(partitioned.sharding(), &sharded);
        assert_eq!(shard_values(&partitioned), vec![vec![1., 2.], vec![3., 4.]]);
        for array in [&local, &other_local] {
            assert_eq!(array.mesh().devices(), &mesh.devices()[..1]);
            assert_eq!(shard_values(array), vec![values.clone()]);
        }

        // For existing arrays, None leaves preserve their target or omit their source check independently.
        let (moved, kept) = session
            .place(
                (local, partitioned.clone()),
                PlacementOptions::<_, _>::new(
                    Some((Some(Placement::new(mesh.clone(), replicated.clone()).unwrap()), None)),
                    Some((None, Some(Placement::new(mesh.clone(), sharded.clone()).unwrap()))),
                    None,
                    None,
                ),
            )
            .unwrap();
        assert_eq!(moved.mesh(), &mesh);
        assert_eq!(moved.sharding(), &replicated);
        assert_eq!(shard_values(&moved), vec![values.clone(); 2]);
        assert_eq!(kept.mesh(), partitioned.mesh());
        assert_eq!(kept.sharding(), partitioned.sharding());
        for (kept_shard, source_shard) in kept.shards().iter().zip(partitioned.shards()) {
            assert!(Arc::ptr_eq(kept_shard.buffer().unwrap(), source_shard.buffer().unwrap()));
        }

        // A placement on the session mesh also works as an optional prefix leaf.
        let partitioned = session
            .place(
                CoreArray::vector(values.clone()).unwrap(),
                PlacementOptions::<_, PlacementRequest>::new(
                    Some(Some(session.placement(sharded.clone()).unwrap())),
                    None,
                    None,
                    None,
                ),
            )
            .unwrap();
        assert_eq!(partitioned.mesh(), &mesh);
        assert_eq!(partitioned.sharding(), &sharded);
        assert_eq!(shard_values(&partitioned), vec![vec![1., 2.], vec![3., 4.]]);

        // A bare device is a placement on that device, also as a prefix leaf.
        let device = mesh.devices()[1];
        let placed = session
            .place(
                CoreArray::vector(values.clone()).unwrap(),
                PlacementOptions::<_, PlacementRequest>::new(Some(Some(device)), None, None, None),
            )
            .unwrap();
        assert_eq!(placed.mesh().devices(), &[device]);
        assert_eq!(shard_values(&placed), vec![values]);
    }

    #[test]
    fn test_session_place_validates_option_prefixes_of_empty_input() {
        let session = Session::builder().client(cpu_client(1)).build().unwrap();
        let device = PlacementRequest::from(session.addressable_devices().unwrap()[0]);

        // Root option prefixes cover the input without leaves, which is returned as it is.
        let options = PlacementOptions::new(Some(device.clone()), Some(device.clone()), Some(true), Some(Some(false)));
        assert_eq!(session.place(Vec::<Array>::new(), options).unwrap().len(), 0);
        assert_eq!(session.place(Vec::<f32>::new(), PlacementOptions::default()).unwrap().len(), 0);
        assert_eq!(session.place(Vec::<f32>::new(), PlacementOptions::to(device.clone())).unwrap().len(), 0);

        // An option prefix with an element for an input without elements matches no node of that input.
        let options = PlacementOptions::<Vec<PlacementRequest>>::new(Some(vec![device]), None, None, None);
        assert!(matches!(
            session.place(Vec::<f32>::new(), options),
            Err(Error::Core(error)) if matches!(
                error.as_ref(),
                CoreError::Parameter(ParameterError::UnusedParameters { paths: Some(paths) })
                    if paths == &vec!["$[0]".to_string()],
            ),
        ));
    }

    #[test]
    fn test_session_place_validates_empty_nodes_of_option_prefixes() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let parameter_error = |result: Result<Vec<Vec<Array>>, Error>| match result {
            Err(Error::Core(error)) => match error.as_ref() {
                CoreError::Parameter(error) => error.clone(),
                error => panic!("expected a parameter error, got `{error}`"),
            },
            result => panic!("expected a parameter error, got `{result:?}`"),
        };
        let missing =
            |path: &str| ParameterError::MissingParameters { expected_count: 1, paths: Some(vec![path.to_string()]) };
        let unused = |path: &str| ParameterError::UnusedParameters { paths: Some(vec![path.to_string()]) };

        // An empty node of the input must be covered by an option leaf or matched by an empty option node.
        assert_eq!(
            parameter_error(session.place(vec![vec![1f32], Vec::new()], PlacementOptions::to(vec![vec![devices[1]]]))),
            missing("$[1]"),
        );

        // An empty option node must match an empty node of the input at the same path. This holds for every option.
        let options = PlacementOptions::to(vec![vec![devices[1]], Vec::new()]);
        assert_eq!(parameter_error(session.place(vec![vec![1f32]], options)), unused("$[1]"));
        let options = PlacementOptions::default().with_source(vec![vec![devices[0]], Vec::new()]);
        assert_eq!(parameter_error(session.place(vec![vec![1f32]], options)), unused("$[1]"));
        let options = PlacementOptions::default().with_donate(vec![vec![false], Vec::new()]);
        assert_eq!(parameter_error(session.place(vec![vec![1f32]], options)), unused("$[1]"));
        let options = PlacementOptions::default().with_may_alias(vec![vec![None], Vec::new()]);
        assert_eq!(parameter_error(session.place(vec![vec![1f32]], options)), unused("$[1]"));

        // An empty option node does not match an input without that node, even when neither has leaves.
        assert_eq!(
            parameter_error(session.place(Vec::<Vec<f32>>::new(), PlacementOptions::to(vec![Vec::<Device>::new()]))),
            unused("$[0]"),
        );

        // Matching empty nodes on both sides are valid, and so is an option leaf over an empty node of the input.
        let placed = session
            .place(vec![vec![1f32], Vec::new()], PlacementOptions::to(vec![vec![devices[1]], Vec::new()]))
            .unwrap();
        assert_eq!(placed[0][0].mesh().devices(), [devices[1]]);
        assert_eq!(placed[1].len(), 0);
        let placed = session
            .place(vec![vec![1f32], Vec::new()], PlacementOptions::to(vec![devices[1], devices[0]]))
            .unwrap();
        assert_eq!(placed[0][0].mesh().devices(), [devices[1]]);
        assert!(placed[0][0].is_committed());
        assert_eq!(placed[0][0].copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1.]);
        assert_eq!(placed[1].len(), 0);
        assert_eq!(session.place(Vec::<Vec<f32>>::new(), PlacementOptions::to(Vec::<Device>::new())).unwrap().len(), 0);

        // Nodes match by path, so an option tree of another container type with the same paths matches.
        let [placed, empty] = session
            .place([vec![1f32], Vec::new()], PlacementOptions::to(vec![vec![devices[1]], Vec::new()]))
            .unwrap();
        assert_eq!(placed[0].mesh().devices(), [devices[1]]);
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn test_session_place_prepares_arguments_before_placing_leaves() {
        struct Leaf(u8);

        impl Parameter for Leaf {}

        impl Placeable for Leaf {
            type Output = u8;

            fn place_parameter(
                self,
                session: &Session,
                target: Option<PlacementRequest>,
                source: Option<PlacementRequest>,
                donate: bool,
                may_alias: Option<bool>,
            ) -> Result<u8, Error> {
                let _ = (session, target, source, donate, may_alias);
                Ok(self.0)
            }
        }

        impl PlaceableParameter<u8> for Leaf {
            fn prepare_arguments(
                arguments: &mut [PlacementArgument<Self>],
                session: &Session,
                path: &dyn Fn(usize) -> ParameterPath,
            ) -> Result<(), Error> {
                let _ = (session, path);
                for argument in arguments {
                    argument.parameter.0 += 1;
                }
                Ok(())
            }
        }

        let session = Session::builder().client(cpu_client(1)).build().unwrap();
        assert_eq!(session.place((Leaf(0), Leaf(2)), PlacementOptions::default()).unwrap(), (1, 3));
    }

    #[test]
    fn test_session_place_validates_every_leaf_before_placing_any() {
        // The first leaf is donated and moves to the second device, and the second leaf fails a check that needs no
        // work to be submitted. Every such check runs for every leaf before any leaf is placed, so the call creates no
        // array and the first leaf keeps its value and its device.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let other_session = Session::builder().client(other_cpu_client(1)).build().unwrap();
        let upload = |session: &Session| {
            let r#type = ArrayType::new_static(DataType::F32, [2]);
            Array::from_host_buffer(session, r#type, None, None, values_to_bytes(&[1f32, 2.])).unwrap()
        };
        let first = upload(&session);
        let devices = session.addressable_devices().unwrap();
        let process_index = devices[0].process_index();
        let mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![devices[0]]).unwrap();
        let rank_two = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 2)).unwrap();
        let unknown = PlacementRequest::from(Device::new(999, process_index));
        let pinned = PlacementRequest::from(Memory::Host { pinned: true });
        let cases = [
            ("type rule of a host leaf", PlaceInput::from(2u8), Some(PlacementRequest::from(rank_two)), None),
            ("unknown device of a host leaf", PlaceInput::from(2u8), Some(unknown.clone()), None),
            ("unknown device of an array leaf", PlaceInput::from(upload(&session)), Some(unknown), None),
            ("source of an array leaf", PlaceInput::from(upload(&session)), None, Some(pinned.clone())),
            ("client of an array leaf", PlaceInput::from(upload(&other_session)), None, None),
            ("in-place target of a foreign array leaf", PlaceInput::from(upload(&other_session)), Some(pinned), None),
        ];
        let mut outcomes = Vec::new();
        for (name, second, target, source) in cases {
            let statistics = session.array_statistics();
            let options = PlacementOptions::new(
                Some((Some(PlacementRequest::from(devices[1])), target)),
                Some((None, source)),
                Some((true, false)),
                None::<bool>,
            );
            let error = session.place((PlaceInput::from(first.clone()), second), options).unwrap_err();
            let created = session.array_statistics().created_arrays - statistics.created_arrays;
            outcomes.push((name, error.to_string(), created));
        }
        let failed = |message: &str| format!("failed to place the parameter at `$.1`: {message}");
        assert_eq!(
            outcomes,
            [
                (
                    "type rule of a host leaf",
                    failed("`place` target sharding rank (2) does not match the input rank (0)"),
                    0,
                ),
                (
                    "unknown device of a host leaf",
                    failed("device 999 is not addressable by the PJRT client in process 0"),
                    0
                ),
                (
                    "unknown device of an array leaf",
                    failed("device 999 is not addressable by the PJRT client in process 0"),
                    0
                ),
                (
                    "source of an array leaf",
                    failed("`place` source memory `Host[Pinned]` does not match the input type `f32[2]`"),
                    0
                ),
                (
                    "client of an array leaf",
                    failed("cannot associate an array with a session of a different PJRT client"),
                    0
                ),
                (
                    "in-place target of a foreign array leaf",
                    failed(
                        "cannot place an array of another PJRT client without naming the devices of the target, \
                         because the result must live on devices of the client of the session",
                    ),
                    0,
                ),
            ],
        );
        assert_eq!(first.mesh().devices(), [devices[0]]);
        assert_eq!(first.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1., 2.]);
    }

    #[test]
    fn test_session_place_checks_options_of_every_leaf_before_values() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let session = Session::builder().client(client).build().unwrap();
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]),
            None,
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();
        let options = |target: PlacementRequest| {
            PlacementOptions::<_, _>::new(
                Some((None, Some(target))),
                Some((Some(PlacementRequest::from(Memory::Host { pinned: true })), None)),
                None,
                None,
            )
        };

        // The first leaf has a source memory that does not match its value, and the second leaf has a target that names
        // nothing. The options of every leaf are checked before any value is, so the error names the second leaf.
        let target = PlacementRequest::InPlace { memory: None, layout: None };
        let error = session.place((array.clone(), array.clone()), options(target)).unwrap_err();
        assert_eq!(
            error.to_string(),
            "failed to place the parameter at `$.1`: `place` in-place target must name a memory or a layout",
        );

        // With valid options, the source memory of the first leaf is reported at its path.
        let target = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap();
        let message =
            format!("`place` source memory `Host[Pinned]` does not match the input type `{}`", array.r#type());
        let error = session.place((array.clone(), array), options(target.into())).unwrap_err();
        let (path, error) = failed_parameter(&error).unwrap();
        assert_eq!(path, "$.0");
        assert!(matches!(error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
        assert_eq!(error.to_string(), message);
    }

    #[test]
    fn test_session_place_reports_the_first_type_rule_error() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let unreduced = Sharding::replicated(mesh.logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(unreduced).unwrap();
        let input = Array::from_host_buffer(&session, r#type, mesh.clone(), None, values_to_bytes(&[1f32, 2., 3., 4.]))
            .unwrap();
        let target = PlacementRequest::from(
            Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap(),
        );
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));

        // Each target and each source below is rejected, so the error of the eager placement must be the first one that
        // the type rule reports for the requests that the frontend stages for the same array. The target changes the
        // reduction state of the value, except for the empty in-place target.
        let reduction_error = "`place` cannot change reduction state; use a reduction (or the corresponding \
                               collective) instead";
        let cases = [
            (
                target.clone(),
                PlacementRequest::from(
                    Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 2)).unwrap(),
                ),
                "`place` source sharding rank (2) does not match the input rank (1)",
            ),
            (
                target.clone(),
                PlacementRequest::InPlace { memory: None, layout: None },
                "`place` in-place source must name a memory or a layout",
            ),
            (
                PlacementRequest::InPlace { memory: None, layout: None },
                PlacementRequest::InPlace { memory: None, layout: None },
                "`place` in-place source must name a memory or a layout",
            ),
            (target.clone(), PlacementRequest::InPlace { memory: None, layout: Some(layout) }, reduction_error),
            (target, PlacementRequest::from(Memory::Host { pinned: true }), reduction_error),
        ];
        for (target, source, expected_error) in cases {
            let options = PlacementOptions::to(target.clone()).with_source(source.clone());
            let eager = session.place(input.clone(), options).err().unwrap().to_string();
            let operation = PlaceOperation::new(target.padded(1).unwrap(), source.padded(1).unwrap());
            let inferred = operation.infer_output_types(&[input.r#type().into_owned()], &[]).err().unwrap();
            let inferred = Error::from(inferred).to_string();
            assert_eq!((eager, inferred), (expected_error.to_string(), expected_error.to_string()));
        }

        // A host value is uploaded as the placement of a value without a tracked sharding in device memory, so its
        // target is rejected with the first error of the type rule for such a value, as the target of an upload of the
        // same value without a mesh is, and as the target of a traced leaf of that type is, whose trace then fails
        // with the same error even though the traced function recovers from the failed placement.
        let manual_mesh = two_device_mesh(client, MeshAxisType::Manual);
        let host = CoreArray::from_elements(ArrayType::new_static(DataType::F32, [2]), &[1f32, 2.]).unwrap();
        let replicated = || Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let rank_2 = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let cases = [
            (
                PlacementRequest::from(
                    Placement::new(mesh.clone(), replicated().with_unreduced_axes(["x"]).unwrap()).unwrap(),
                ),
                reduction_error.to_string(),
            ),
            (
                PlacementRequest::from(
                    Placement::new(mesh.clone(), replicated().with_reduced_axes(["x"]).unwrap()).unwrap(),
                ),
                reduction_error.to_string(),
            ),
            (
                PlacementRequest::from(
                    Placement::new(
                        manual_mesh.clone(),
                        Sharding::replicated(manual_mesh.logical_mesh().clone(), 1)
                            .with_varying_manual_axes(["x"])
                            .unwrap(),
                    )
                    .unwrap(),
                ),
                "`place` cannot move values that vary along manual mesh axes or carry reduction state over them"
                    .to_string(),
            ),
            (
                PlacementRequest::Device { device: mesh.devices()[1], memory: None, layout: Some(rank_2.clone()) },
                "`place` target layout is invalid: tiled layout rank 2 does not match array rank 1".to_string(),
            ),
            (
                PlacementRequest::from(
                    Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 2)).unwrap(),
                ),
                "`place` target sharding rank (2) does not match the input rank (1)".to_string(),
            ),
        ];
        for (target, expected_error) in cases {
            let options = || PlacementOptions::to(target.clone());
            let uploaded = Array::from_host_buffer(
                &session,
                ArrayType::new_static(DataType::F32, [2]),
                None,
                None,
                values_to_bytes(&[1f32, 2.]),
            )
            .unwrap();
            let host = session.place(host.clone(), options()).err().unwrap().to_string();
            let array = session.place(uploaded, options()).err().unwrap().to_string();
            let error = RefCell::new(None);
            let traced: Result<TracedProgram<ArrayType, ArrayType>, _> = trace(
                |x: ArrayTracer| record_placement_error(&error, session.place(x.clone(), options())).unwrap_or(x),
                ArrayType::new_static(DataType::F32, [2]),
            );
            let traced = traced.map(|traced| traced.to_string()).map_err(|error| error.to_string());
            assert_eq!(traced, Err(expected_error.clone()));
            let recorded = error.take().unwrap();
            assert_eq!((host, array, recorded), (expected_error.clone(), expected_error.clone(), expected_error));
        }
    }

    #[test]
    fn test_session_place_agrees_across_paths_for_targets_and_sources() {
        /// Input state of the matrix, which is uploaded onto `mesh` with `sharding` and then placed in `memory`, or
        /// uploaded without a mesh when `sharding` is `None`.
        struct InputState {
            name: &'static str,
            mesh: DeviceMesh,
            sharding: Option<Sharding>,
            memory: Memory,
        }

        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let pinned = Memory::Host { pinned: true };
        let single_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let double_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        // Uploads `values` with the shape `shape`, the sharding `sharding` over the logical mesh of `mesh`, and in
        // `memory`, which a placement onto that memory selects after the upload. Without a sharding, the values are
        // uploaded without a mesh, so the array is uncommitted on the canonical mesh of the first device.
        let upload =
            |shape: &[usize], sharding: Option<&Sharding>, mesh: &DeviceMesh, memory: Memory, values: &[f32]| {
                let r#type = ArrayType::new_static(DataType::F32, shape.to_vec());
                let array = match sharding {
                    Some(sharding) => {
                        let r#type = r#type.with_sharding(sharding.clone()).unwrap();
                        Array::from_host_buffer(&session, r#type, mesh.clone(), None, values_to_bytes(values)).unwrap()
                    }
                    None => Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(values)).unwrap(),
                };
                match memory {
                    Memory::Device => array,
                    memory => session.place(array, PlacementOptions::to(memory)).unwrap(),
                }
            };

        // The input states, each with the meshes of its own devices and of the devices that the other paths use.
        let single = DeviceMesh::new(single_mesh.clone(), vec![devices[0]]).unwrap();
        let double = DeviceMesh::new(double_mesh.clone(), devices[..2].to_vec()).unwrap();
        let other_mesh = |mesh: &DeviceMesh| {
            let other_devices = devices[2..2 + mesh.devices().len()].to_vec();
            DeviceMesh::new(mesh.logical_mesh().clone(), other_devices).unwrap()
        };
        let replicated = |mesh: &LogicalMesh| Sharding::replicated(mesh.clone(), 1);
        let unconstrained =
            |mesh: &LogicalMesh| Sharding::new(mesh.clone(), vec![ShardingDimension::Unconstrained]).unwrap();
        let canonical = DeviceMesh::new(LogicalMesh::single_device(), vec![devices[0]]).unwrap();
        let inputs = [
            InputState {
                name: "single device",
                mesh: single.clone(),
                sharding: Some(replicated(&single_mesh)),
                memory: Memory::Device,
            },
            InputState {
                name: "uncommitted on the canonical mesh",
                mesh: canonical,
                sharding: None,
                memory: Memory::Device,
            },
            InputState {
                name: "sharded over two devices",
                mesh: double.clone(),
                sharding: Some(Sharding::new(double_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()),
                memory: Memory::Device,
            },
            InputState {
                name: "replicated over two devices",
                mesh: double.clone(),
                sharding: Some(replicated(&double_mesh)),
                memory: Memory::Device,
            },
            InputState {
                name: "reduced over an axis of size two",
                mesh: double.clone(),
                sharding: Some(replicated(&double_mesh).with_reduced_axes(["x"]).unwrap()),
                memory: Memory::Device,
            },
            InputState {
                name: "unreduced over an axis of size one",
                mesh: single.clone(),
                sharding: Some(replicated(&single_mesh).with_unreduced_axes(["x"]).unwrap()),
                memory: Memory::Device,
            },
            InputState {
                name: "unreduced over an axis of size two",
                mesh: double.clone(),
                sharding: Some(replicated(&double_mesh).with_unreduced_axes(["x"]).unwrap()),
                memory: Memory::Device,
            },
            InputState {
                name: "pinned host memory",
                mesh: single.clone(),
                sharding: Some(replicated(&single_mesh)),
                memory: pinned,
            },
        ];
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));

        let mut disagreements = Vec::new();
        for InputState { name: input_name, mesh, sharding, memory } in inputs {
            let input = upload(&[4], sharding.as_ref(), &mesh, memory, &[1., 2., 3., 4.]);
            assert_eq!(input.mesh(), &mesh);
            let other_memory = if memory == Memory::Device { pinned } else { Memory::Device };
            // A sharding target resolves over the default mesh of the session handle that places the value. A target on
            // another logical mesh over the devices of the value (i.e., its mesh with another axis `'y'=1:explicit`,
            // which restates its reduction state) moves the value onto a mesh of another topology where it lives (e.g.,
            // an uncommitted value from the canonical mesh of its device onto `['device'=1:auto, 'y'=1:explicit]`). A
            // device target on the device of the value is the sole device of the mesh of a value on one device.
            let meshed_session = session.clone().with_device_mesh(other_mesh(&mesh));
            let input_sharding = input.sharding();
            let mut renamed_axes = input_sharding.mesh().axes().to_vec();
            renamed_axes.push(MeshAxis::new("y", 1, MeshAxisType::Explicit).unwrap());
            let renamed_mesh = LogicalMesh::new(renamed_axes).unwrap();
            let renamed = DeviceMesh::new(renamed_mesh.clone(), mesh.devices().to_vec()).unwrap();
            let renamed_sharding = Sharding::new(renamed_mesh, input_sharding.dimensions().to_vec())
                .and_then(|sharding| sharding.with_unreduced_axes(input_sharding.unreduced_axes().clone()))
                .and_then(|sharding| sharding.with_reduced_axes(input_sharding.reduced_axes().clone()))
                .unwrap();
            // A placement never changes reduction state on any path, so a target that does not restate the reduction
            // state of the value (i.e., the mesh of the value replicated, or a mesh that does not define the axis of
            // that state) is rejected on every path, as a device target is for a value that is unreduced over a larger
            // axis.
            let undefined_mesh =
                LogicalMesh::new(vec![MeshAxis::new("y", mesh.devices().len(), MeshAxisType::Explicit).unwrap()])
                    .unwrap();
            let undefined = DeviceMesh::new(undefined_mesh.clone(), other_mesh(&mesh).devices().to_vec()).unwrap();
            let targets = [
                (
                    "mesh",
                    Some(PlacementRequest::from(Placement::new(other_mesh(&mesh), input.sharding().clone()).unwrap())),
                ),
                (
                    "replicated",
                    Some(PlacementRequest::from(
                        Placement::new(mesh.clone(), replicated(mesh.logical_mesh())).unwrap(),
                    )),
                ),
                (
                    "undefined axis",
                    Some(PlacementRequest::from(Placement::new(undefined, replicated(&undefined_mesh)).unwrap())),
                ),
                ("sharding", Some(PlacementRequest::from(meshed_session.placement(input.sharding().clone()).unwrap()))),
                (
                    "other logical mesh",
                    Some(PlacementRequest::from(Placement::new(renamed, renamed_sharding).unwrap())),
                ),
                (
                    "unconstrained",
                    Some(PlacementRequest::from(
                        Placement::new(mesh.clone(), unconstrained(mesh.logical_mesh())).unwrap(),
                    )),
                ),
                ("device", Some(PlacementRequest::from(devices[2]))),
                ("own device", Some(PlacementRequest::from(mesh.devices()[0]))),
                ("memory", Some(PlacementRequest::from(other_memory))),
                ("none", None),
            ];
            // A device source states that the value lives on that device alone, so it is rejected on every path for a
            // value that is tracked over two devices. The devices of a source are informative, so a source on other
            // devices than the value is accepted on every path but a compiled function, whose program must contain
            // every device that a `place` names, and reverse mode sends the cotangent there. A source layout is checked
            // against the type of the value only, which declares none here, so it is accepted on every path, including
            // the batched one, where it describes a batch item.
            let resolved = Placement::new(input.mesh().clone(), input.sharding().clone()).unwrap();
            let sources = [
                ("none", None),
                ("inferred", Some(PlacementRequest::from(input.placement().unwrap()))),
                ("resolved", Some(resolved.clone().into())),
                (
                    "other devices",
                    Some(PlacementRequest::from(Placement::new(other_mesh(&mesh), input.sharding().clone()).unwrap())),
                ),
                ("memory", Some(PlacementRequest::from(memory))),
                ("device", Some(PlacementRequest::from(mesh.devices()[0]))),
                ("layout", Some(PlacementRequest::InPlace { memory: None, layout: Some(layout.clone()) })),
                ("resolved with layout", Some(resolved.with_layout(layout.clone()).into())),
                (
                    "device with layout",
                    Some(PlacementRequest::Device {
                        device: mesh.devices()[0],
                        memory: None,
                        layout: Some(layout.clone()),
                    }),
                ),
            ];
            for (target_name, target) in &targets {
                let session = if *target_name == "sharding" { &meshed_session } else { &session };
                for (source_name, source) in &sources {
                    let cell = format!("{input_name}, target {target_name}, source {source_name}");
                    let options = || PlacementOptions::new(target.clone(), source.clone(), None::<bool>, None::<bool>);
                    let error = RefCell::new(None::<String>);
                    let failure = |other: String| error.take().unwrap_or(other);

                    // The eager placement is what every other path must agree with.
                    let eager = session.place(input.clone(), options()).map(|output| observe(&output));
                    let eager = eager.map_err(|error| error.to_string());

                    // A value that is unreduced over an axis of size two holds a contribution on each device, which no
                    // placement combines, so the targets that do not restate that state are rejected with the error of
                    // the type rule, whatever the source, and every other path must report the same error.
                    let rejection = match (input_name, *target_name) {
                        ("unreduced over an axis of size two", "replicated") => Some(format!(
                            "`{PLACE_OPERATION_NAME}` cannot change reduction state; use a reduction (or the \
                             corresponding collective) instead",
                        )),
                        ("unreduced over an axis of size two", "undefined axis") => Some(format!(
                            "`{PLACE_OPERATION_NAME}` cannot redefine preserved explicit mesh axis `'x'=2:explicit` as \
                             an undefined axis",
                        )),
                        ("unreduced over an axis of size two", "device" | "own device") => Some(format!(
                            "`{PLACE_OPERATION_NAME}` cannot move a value with reduction state over mesh axes larger \
                             than one (i.e., `x`) onto a single device; use a reduction (or the corresponding \
                             collective) first",
                        )),
                        _ => None,
                    };
                    if let Some(rejection) = rejection
                        && eager != Err(rejection.clone())
                    {
                        disagreements.push(format!("{cell}, eager: {eager:?} != rejection: {rejection:?}"));
                    }

                    // The core `Place` capability places the array with the requests that the frontend stages for a
                    // traced value, through the same type rule and resolver.
                    let request = |placement: &Option<PlacementRequest>| {
                        placement.clone().map(|placement| placement.padded(1)).transpose()
                    };
                    let core = match (request(target), request(source)) {
                        (Ok(target), Ok(source)) => input
                            .place(&target, &source)
                            .map(|output| observe(&output))
                            .map_err(|error| Error::from(error).to_string()),
                        (Err(error), _) | (_, Err(error)) => Err(error.to_string()),
                    };
                    if core != eager {
                        disagreements.push(format!("{cell}, core: {core:?} != eager: {eager:?}"));
                    }

                    // The type rule owns the type of a placed value, so the array that the eager placement produces has
                    // the type that the type rule infers for the request as it is given.
                    if let (Ok(eager), Ok(target), Ok(source)) = (&eager, request(target), request(source)) {
                        let r#type = input.r#type();
                        let sharding = Some(input.sharding());
                        let inferred = PlaceOperation::infer_output_type_allowing_uneven_splits(
                            &r#type,
                            sharding,
                            target.as_ref(),
                            source.as_ref(),
                        );
                        if inferred.as_ref() != Ok(&eager.r#type) {
                            disagreements.push(format!("{cell}, type rule: {inferred:?} != eager: {:?}", eager.r#type));
                        }
                    }

                    let staged = replay_placement(session, &input, options()).map(|output| observe(&output));
                    if staged != eager {
                        disagreements.push(format!("{cell}, staged: {staged:?} != eager: {eager:?}"));
                    }

                    // A function compiled for the mesh of the input, where the decisions allow the target (i.e., one
                    // that moves no value to other devices, or a device target on the sole device of that mesh),
                    // returns the eager result. A source on other devices is admitted there too, because the function
                    // moves nothing to it.
                    let is_sole_device = *target_name == "own device" && mesh.devices().len() == 1;
                    if matches!(*target_name, "memory" | "none" | "unconstrained") || is_sole_device {
                        let compiled: Result<CompiledFunction<ArrayType, ArrayType>, _> = compile(
                            |x: CompileTracer| {
                                record_placement_error(&error, session.place(x.clone(), options())).unwrap_or(x)
                            },
                            input.r#type().into_owned(),
                            &session,
                            input.mesh().clone(),
                        );
                        let compiled = match (error.take(), compiled) {
                            (Some(message), _) => Err(message),
                            (None, Err(error)) => Err(error.to_string()),
                            (None, Ok(compiled)) => compiled
                                .executable()
                                .execute(&session, input.clone())
                                .map(|output| observe(&output))
                                .map_err(|error| error.to_string()),
                        };
                        if compiled != eager {
                            disagreements.push(format!("{cell}, compiled: {compiled:?} != eager: {eager:?}"));
                        }
                    }

                    // Eager forward mode differentiation places the primal and the tangent alike.
                    let jvp = differentiate_at(input.clone())
                        .jvp(input.clone(), |x| record_placement_error(&error, session.place(x, options())))
                        .map_err(|error| failure(error.to_string()));
                    let (primal, tangent) = match jvp {
                        Ok((primal, tangent)) => (Ok(observe(&primal)), Ok(observe(&tangent))),
                        Err(error) => (Err(error.clone()), Err(error)),
                    };
                    if primal != eager {
                        disagreements.push(format!("{cell}, jvp primal: {primal:?} != eager: {eager:?}"));
                    }
                    if tangent != eager {
                        disagreements.push(format!("{cell}, jvp tangent: {tangent:?} != eager: {eager:?}"));
                    }

                    // Reverse mode, batched or not, rejects one combination by design. The cotangent of a device
                    // target onto a value that is reduced over a larger axis is unreduced over it, which a placement
                    // cannot produce. A memory source of a concrete value is completed with where the value lives, so
                    // the cotangent of a value that a device target gathers returns to its devices with every source.
                    let is_gathered = mesh.devices().len() > 1 && matches!(*target_name, "device" | "own device");
                    let rejected_in_reverse_mode =
                        if eager.is_ok() && is_gathered && input_name == "reduced over an axis of size two" {
                            Some(format!(
                                "cannot transpose `{PLACE_OPERATION_NAME}` onto a single device for a value that is \
                                 reduced over mesh axes larger than one (i.e., `x`), because its cotangent is \
                                 unreduced over them",
                            ))
                        } else {
                            None
                        };

                    // Returns the disagreement of the result of a reverse mode path with the rejection `expected`, if
                    // there is one. A result that is rejected as expected agrees.
                    let check_rejection =
                        |path: &str, expected: &Option<String>, result: &Result<(), String>| match expected {
                            Some(expected) if result.as_ref().err() != Some(expected) => {
                                Some(format!("{cell}, {path}: {result:?} does not report `{expected}`"))
                            }
                            _ => None,
                        };

                    // Returns a seed for the cotangent of `output`, with the values `values` and the shape `shape` of
                    // `output`, which lives on other devices than `output` in the memory of `output`.
                    let seed = |output: &Observed, shape: &[usize], sharding: &Sharding, values: &[f32]| {
                        let seed_devices = devices
                            .iter()
                            .copied()
                            .filter(|device| !output.devices.contains(device))
                            .take(output.devices.len())
                            .collect::<Vec<_>>();
                        let seed_mesh = DeviceMesh::new(output.logical_mesh.clone(), seed_devices).unwrap();
                        upload(shape, Some(&sharding.cotangent().unwrap()), &seed_mesh, output.r#type.memory(), values)
                    };

                    // Returns the disagreement of a cotangent with the cotangent of `primal` that its pullback must
                    // return for a seed whose values are `values`. It returns to the memory of the primal
                    // and to the devices that the source names, which are those of the primal, except for a source
                    // on other devices. A memory source names no devices, and it is completed with the devices of the
                    // concrete primal, so the cotangent returns to them, also from a seed on other devices.
                    let check_cotangent = |path: &str, cotangent: &Array, primal: &Array, values: Vec<f32>| {
                        let cotangent = observe(cotangent);
                        let devices = match *source_name {
                            "other devices" => other_mesh(&mesh).devices().to_vec(),
                            _ => primal.mesh().devices().to_vec(),
                        };
                        let expected =
                            (devices, observe(primal).memory_kinds, primal.r#type().cotangent().unwrap(), values);
                        let actual = (cotangent.devices, cotangent.memory_kinds, cotangent.r#type, cotangent.values);
                        (actual != expected).then(|| format!("{cell}, {path} cotangent: {actual:?} != {expected:?}"))
                    };

                    // Eager reverse mode differentiation, whose pullback is applied to a seed that lives on other
                    // devices than the output, returns the cotangent where `check_cotangent` says.
                    let vjp = differentiate_at(input.clone())
                        .vjp(|x| record_placement_error(&error, session.place(x, options())))
                        .map_err(|error| failure(error.to_string()));
                    match vjp {
                        Ok((value, pullback)) => {
                            let value = observe(&value);
                            if Ok(&value) != eager.as_ref() {
                                disagreements.push(format!("{cell}, vjp primal: {value:?} != eager: {eager:?}"));
                            }
                            let seed = seed(&value, &[4], &value.sharding, &[5., 6., 7., 8.]);
                            let cotangent =
                                pullback.apply_materialized(seed.clone()).map_err(|error| error.to_string());
                            let result = cotangent.as_ref().map(|_| ()).map_err(Clone::clone);
                            match (check_rejection("vjp", &rejected_in_reverse_mode, &result), cotangent) {
                                (Some(disagreement), _) => disagreements.push(disagreement),
                                (None, _) if rejected_in_reverse_mode.is_some() => {}
                                (None, Ok(cotangent)) => {
                                    let values = observe(&seed).values;
                                    disagreements.extend(check_cotangent("vjp", &cotangent, &input, values));
                                }
                                (None, Err(error)) => disagreements.push(format!("{cell}, vjp cotangent: {error}")),
                            }
                        }
                        Err(error) => {
                            if Err(&error) != eager.as_ref().map(|_| ()) {
                                disagreements.push(format!("{cell}, vjp: {error} != eager: {eager:?}"));
                            }
                        }
                    }

                    // Batched reverse mode differentiation places every batch item where the eager placement places the
                    // input, and its pullback returns the cotangent of every batch item where `check_cotangent` says.
                    // The seed is mapped when the output lives on the logical mesh of the input, and it is the same for
                    // every batch item otherwise, because the batches of all mapped inputs live on one logical mesh.
                    let Ok(eager) = &eager else { continue };
                    let packed_sharding =
                        sharding.as_ref().map(|sharding| sharding.batched(0, ShardingDimension::Replicated).unwrap());
                    let packed = upload(
                        &[2, 4],
                        packed_sharding.as_ref(),
                        input.mesh(),
                        memory,
                        &[1., 2., 3., 4., 5., 6., 7., 8.],
                    );
                    let eager_sharding = &eager.sharding;
                    let (packed_seed, seed_axis, seed_repeats) = if &eager.logical_mesh == input.mesh().logical_mesh() {
                        let sharding = eager_sharding.batched(0, ShardingDimension::Replicated).unwrap();
                        let values = [9., 10., 11., 12., 13., 14., 15., 16.];
                        (seed(eager, &[2, 4], &sharding, &values), BatchAxis::new(0), 1)
                    } else {
                        (seed(eager, &[4], eager_sharding, &[9., 10., 11., 12.]), BatchAxis::replicated(), 2)
                    };
                    let batched: Result<(Array, Array), _> = batch(
                        |(x, seed)| {
                            let (value, pullback) = differentiate_at(x)
                                .vjp(|x| record_placement_error(&error, session.place(x, options())))?;
                            Ok((value, pullback.apply_materialized(seed)?))
                        },
                        (packed.clone(), packed_seed.clone()),
                        (BatchAxis::new(0), seed_axis),
                        (BatchAxis::new(0), BatchAxis::new(0)),
                        None,
                    );
                    let batched = batched.map_err(|error| failure(error.to_string()));
                    let result = batched.as_ref().map(|_| ()).map_err(Clone::clone);
                    match (check_rejection("batched vjp", &rejected_in_reverse_mode, &result), batched) {
                        (Some(disagreement), _) => disagreements.push(disagreement),
                        (None, _) if rejected_in_reverse_mode.is_some() => {}
                        (None, Ok((value, cotangent))) => {
                            let value = observe(&value);
                            let replicated = ShardingDimension::Replicated;
                            let expected = Observed {
                                devices: eager.devices.clone(),
                                logical_mesh: eager.logical_mesh.clone(),
                                sharding: eager.sharding.batched(0, replicated.clone()).unwrap(),
                                r#type: eager.r#type.batched(0, Dimension::Static(2), replicated).unwrap(),
                                memory_kinds: eager.memory_kinds.clone(),
                                values: observe(&packed).values,
                                committed: eager.committed,
                            };
                            let actual = &value;
                            if actual != &expected {
                                disagreements.push(format!("{cell}, batched vjp value: {actual:?} != {expected:?}"));
                            }
                            let values = observe(&packed_seed).values.repeat(seed_repeats);
                            disagreements.extend(check_cotangent("batched vjp", &cotangent, &packed, values));
                        }
                        (None, Err(error)) => disagreements.push(format!("{cell}, batched vjp: {error}")),
                    }
                }
            }
        }

        assert!(disagreements.is_empty(), "{} disagreements:\n{}", disagreements.len(), disagreements.join("\n"));
    }

    #[test]
    fn test_session_place_agrees_across_paths_for_runtime_reduction_state_over_auto_axes() {
        // The type of an array omits its reduction state over auto mesh axes, which its runtime sharding records (e.g.,
        // a scalar that is imported as the contributions 1 and 2 of a pending sum over `'x'=2:auto`, or that is reduced
        // over that axis), so the eager placement, the core `Place` capability, and forward mode validate a concrete
        // array against its runtime sharding. A target that restates that state (i.e., the placement of the array, or
        // the same state over other devices) is accepted and keeps the value, whereas a target that drops it and a
        // device target of a pending sum over a larger axis are rejected, as for the same state over an explicit axis.
        // The type rule with the runtime sharding of the array agrees with the eager placement on every cell. A staged
        // value has only its type, so the replay of a staged program rejects the targets that restate a state over an
        // auto axis when it is traced, and it agrees with the eager placement on every other target. Each input also
        // has a packed counterpart of type `f32[2]` with the same state, whose batch items are such scalars, so that an
        // eager `batch` of the placement of each item validates the item against its runtime sharding too, and accepts
        // the identity target while it rejects a target that drops the state with the same error.
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let pinned = Memory::Host { pinned: true };
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let auto = DeviceMesh::new(auto_mesh.clone(), devices[..2].to_vec()).unwrap();
        let other_auto = DeviceMesh::new(auto_mesh.clone(), devices[2..].to_vec()).unwrap();
        let auto_replicated = Sharding::replicated(auto_mesh.clone(), 0);
        let scalar = ArrayType::scalar(DataType::F32);
        let pair = ArrayType::new_static(DataType::F32, [2]);
        let changed = format!(
            "`{PLACE_OPERATION_NAME}` cannot change reduction state; use a reduction (or the corresponding collective) \
             instead",
        );
        let unreduced_auto = auto_replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let reduced_auto = auto_replicated.clone().with_reduced_axes(["x"]).unwrap();
        let auto_inputs = [
            (
                "unreduced over an auto axis",
                Array::from_host_shards(
                    &session,
                    scalar.clone(),
                    auto.clone(),
                    Some(unreduced_auto.clone()),
                    [
                        (devices[0].id(), values_to_bytes(&[1f32]).as_slice()),
                        (devices[1].id(), values_to_bytes(&[2f32]).as_slice()),
                    ],
                )
                .unwrap(),
                Array::from_host_shards(
                    &session,
                    pair.clone(),
                    auto.clone(),
                    Some(Sharding::replicated(auto_mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap()),
                    [
                        (devices[0].id(), values_to_bytes(&[1f32, 2.]).as_slice()),
                        (devices[1].id(), values_to_bytes(&[10f32, 20.]).as_slice()),
                    ],
                )
                .unwrap(),
            ),
            (
                "reduced over an auto axis",
                Array::from_host_buffer(
                    &session,
                    scalar.clone(),
                    auto.clone(),
                    Some(reduced_auto),
                    values_to_bytes(&[3f32]),
                )
                .unwrap(),
                Array::from_host_buffer(
                    &session,
                    pair.clone(),
                    auto.clone(),
                    Some(Sharding::replicated(auto_mesh.clone(), 1).with_reduced_axes(["x"]).unwrap()),
                    values_to_bytes(&[3f32, 30.]),
                )
                .unwrap(),
            ),
        ];
        let mut disagreements = Vec::new();
        for (input_name, input, packed) in auto_inputs {
            assert_eq!(
                (observe(&input).values, input.r#type().into_owned()),
                (vec![3.], scalar.clone().with_sharding(auto_replicated.clone()).unwrap())
            );
            let state = input.sharding().clone();
            let is_unreduced = !state.unreduced_axes().is_empty();
            let targets = [
                ("identity", Some(PlacementRequest::from(Placement::new(auto.clone(), state.clone()).unwrap())), None),
                (
                    "other devices",
                    Some(PlacementRequest::from(Placement::new(other_auto.clone(), state.clone()).unwrap())),
                    None,
                ),
                (
                    "replicated",
                    Some(PlacementRequest::from(Placement::new(auto.clone(), auto_replicated.clone()).unwrap())),
                    Some(changed.clone()),
                ),
                (
                    "device",
                    Some(PlacementRequest::from(devices[2])),
                    is_unreduced.then(|| {
                        format!(
                            "`{PLACE_OPERATION_NAME}` cannot move a value with reduction state over mesh axes larger \
                             than one (i.e., `x`) onto a single device; use a reduction (or the corresponding \
                             collective) first",
                        )
                    }),
                ),
                ("memory", Some(PlacementRequest::from(pinned)), None),
                ("none", None, None),
            ];
            for (target_name, target, rejection) in targets {
                let cell = format!("{input_name}, target {target_name}");
                let options =
                    || PlacementOptions::new(target.clone(), None::<PlacementRequest>, None::<bool>, None::<bool>);
                let error = RefCell::new(None::<String>);
                let eager = session.place(input.clone(), options()).map(|output| observe(&output));
                let eager = eager.map_err(|error| error.to_string());
                match (&rejection, &eager) {
                    (Some(rejection), _) if eager.as_ref().err() != Some(rejection) => {
                        disagreements.push(format!("{cell}, eager: {eager:?} != rejection: {rejection:?}"));
                    }
                    (None, Err(_)) => disagreements.push(format!("{cell}, eager: {eager:?}")),
                    (None, Ok(output)) if output.values != vec![3.] => {
                        disagreements.push(format!("{cell}, eager values: {:?} != [3.0]", output.values));
                    }
                    (None, Ok(output)) if target_name == "identity" && output.sharding != state => {
                        disagreements.push(format!("{cell}, eager sharding: {} != {state}", output.sharding));
                    }
                    _ => {}
                }

                let request = target.clone().map(|target| target.padded(0)).transpose().unwrap();
                let inferred = PlaceOperation::infer_output_type_allowing_uneven_splits(
                    &input.r#type(),
                    Some(&state),
                    request.as_ref(),
                    None,
                );
                let inferred = inferred.map_err(|error| Error::from(error).to_string());
                if inferred != eager.clone().map(|output| output.r#type) {
                    disagreements.push(format!("{cell}, type rule: {inferred:?} != eager: {eager:?}"));
                }

                let core = input.place(&request, None).map(|output| observe(&output));
                let core = core.map_err(|error| Error::from(error).to_string());
                let jvp = differentiate_at(input.clone())
                    .jvp(input.clone(), |x| record_placement_error(&error, session.place(x, options())))
                    .map(|(primal, _)| observe(&primal))
                    .map_err(|other| error.take().unwrap_or(other.to_string()));
                let staged = replay_placement(&session, &input, options()).map(|output| observe(&output));
                let expected_staged = match target_name {
                    "identity" | "other devices" => Err(changed.clone()),
                    _ => eager.clone(),
                };
                for (path, result, expected) in
                    [("core", core, &eager), ("jvp", jvp, &eager), ("staged", staged, &expected_staged)]
                {
                    if &result != expected {
                        disagreements.push(format!("{cell}, {path}: {result:?} != {expected:?}"));
                    }
                }

                // The identity target keeps the packed value exactly (i.e., its runtime sharding and the contribution
                // of each device), and a target that drops the state is rejected with the error of the other paths.
                let expected_batched = match target_name {
                    "identity" => Ok((packed.sharding().clone(), shard_values(&packed))),
                    "replicated" => Err(changed.clone()),
                    _ => continue,
                };
                let batched: Result<Array, _> = batch(
                    |x| record_placement_error(&error, session.place(x, options())),
                    packed.clone(),
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                );
                let batched = batched
                    .map(|output| (output.sharding().clone(), shard_values(&output)))
                    .map_err(|other| error.take().unwrap_or(other.to_string()));
                if batched != expected_batched {
                    disagreements.push(format!("{cell}, batched: {batched:?} != {expected_batched:?}"));
                }
            }
        }

        assert!(disagreements.is_empty(), "{} disagreements:\n{}", disagreements.len(), disagreements.join("\n"));
    }

    #[test]
    fn test_session_place_agrees_across_paths_for_shard_layouts() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let mut disagreements = Vec::new();

        // The runtime sharding of an array replicated along an auto axis determines its shards, which its type does
        // not, so an expected source layout that overlaps them is rejected by the eager placement as forward mode
        // rejects it after it completes the source with that runtime placement, with the error of the type rule.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let auto = DeviceMesh::new(auto_mesh.clone(), devices[..2].to_vec()).unwrap();
        let auto_matrix = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4, 4]),
            auto,
            Some(Sharding::replicated(auto_mesh, 2)),
            values_to_bytes(&[42f32; 16]),
        )
        .unwrap();
        let source =
            PlacementRequest::InPlace { memory: None, layout: Some(Layout::Strided(StridedLayout::new(vec![8, 4]))) };
        let options = || PlacementOptions::default().with_source(source.clone());
        let source_request = source.clone();
        let error = RefCell::new(None::<String>);
        let invalid = Err(format!(
            "`{PLACE_OPERATION_NAME}` source layout is invalid: strided layout stride 8 on axis 0 is smaller than the \
             16-byte span occupied by more minor axes and may alias array elements",
        ));
        let results = [
            ("eager", session.place(auto_matrix.clone(), options()).map(|_| ()).map_err(|error| error.to_string())),
            (
                "type rule",
                PlaceOperation::infer_output_type_allowing_uneven_splits(
                    &auto_matrix.r#type(),
                    Some(auto_matrix.sharding()),
                    None,
                    Some(&source_request),
                )
                .map(|_| ())
                .map_err(|error| Error::from(error).to_string()),
            ),
            (
                "core",
                auto_matrix
                    .place(&None::<PlacementRequest>, &Some(source_request.clone()))
                    .map(|_| ())
                    .map_err(|error| Error::from(error).to_string()),
            ),
            (
                "jvp",
                differentiate_at(auto_matrix.clone())
                    .jvp(auto_matrix.clone(), |x| record_placement_error(&error, session.place(x, options())))
                    .map(|_| ())
                    .map_err(|other| error.take().unwrap_or(other.to_string())),
            ),
        ];
        for (path, result) in results {
            if result != invalid {
                disagreements.push(format!("auto source layout, {path}: {result:?} != {invalid:?}"));
            }
        }

        // A layout describes each shard, so the byte strides `[8, 4]` of a dense `f32[4, 2]` column shard of an
        // `f32[4, 4]` value that is split into two column shards pass the type rule (also with the runtime sharding of
        // the array), although they would overlap the rows of the whole value. No backend here can execute the accepted
        // type, and each says why: the CPU plugin cannot create strided buffers, which every path that places the array
        // reports alike, and the reference backend stores a whole value in one buffer, which it reports as a shard
        // layout that it cannot materialize.
        let double_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let columns =
            Sharding::new(double_mesh.clone(), vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])])
                .unwrap();
        let values = (1..=16).map(|value| value as f32).collect::<Vec<_>>();
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4, 4]).with_sharding(columns).unwrap(),
            DeviceMesh::new(double_mesh, devices[..2].to_vec()).unwrap(),
            None,
            values_to_bytes(&values),
        )
        .unwrap();
        let layout = Layout::Strided(StridedLayout::new(vec![8, 4]));
        let options = || PlacementOptions::to(PlacementRequest::InPlace { memory: None, layout: Some(layout.clone()) });
        let eager = session
            .place(input.clone(), options())
            .map(|output| observe(&output))
            .map_err(|error| error.to_string());
        let staged = replay_placement(&session, &input, options()).map(|output| observe(&output));
        let error = RefCell::new(None::<String>);
        let jvp = differentiate_at(input.clone())
            .jvp(input.clone(), |x| record_placement_error(&error, session.place(x, options())))
            .map(|(primal, _)| observe(&primal))
            .map_err(|other| error.take().unwrap_or(other.to_string()));
        let request = PlacementRequest::InPlace { memory: None, layout: Some(layout.clone()) };
        let inferred = PlaceOperation::infer_output_type_allowing_uneven_splits(
            &input.r#type(),
            Some(input.sharding()),
            Some(&request),
            None,
        );
        let output_type = input.r#type().into_owned().with_layout(layout.clone());
        if inferred.as_ref() != Ok(&output_type) {
            disagreements.push(format!("per-shard layout, type rule: {inferred:?} != {output_type:?}"));
        }
        let unsupported = Err("cpu PJRT does not support strided device layouts".to_string());
        for (path, result) in [("eager", &eager), ("staged", &staged), ("jvp", &jvp)] {
            if result != &unsupported {
                disagreements.push(format!("per-shard layout, {path}: {result:?} != {unsupported:?}"));
            }
        }
        let reference = CoreArray::from_elements(input.r#type().into_owned(), &values).unwrap();
        let reference = reference.place(&request, None).map_err(|error| error.to_string());
        let whole_value = Err(format!(
            "the reference `Array` backend stores a whole value in one buffer, so it cannot materialize the shard \
             layout of `{output_type}` for it: strided layout stride 8 on axis 0 is smaller than the 16-byte span occupied by \
             more minor axes and may alias array elements",
        ));
        if reference != whole_value {
            disagreements.push(format!("per-shard layout, reference: {reference:?} != {whole_value:?}"));
        }

        // Without a layout, a target keeps the layout of a value while it still describes each target shard, and the
        // type rule says the same. The CPU plugin cannot create the strided layouts above, so the case of a layout
        // that does not describe the gathered shard is covered by the type rule and the reference backend, and here a
        // tiled layout, which describes every shard, is gathered from the column shards onto one device.
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let tiled = PlacementOptions::to(PlacementRequest::InPlace { memory: None, layout: Some(column_major) });
        let tiled = session.place(input.clone(), tiled).unwrap();
        let gather = || PlacementOptions::to(PlacementRequest::from(devices[2]));
        let inferred = PlaceOperation::new(devices[2], None).infer_output_types(&[tiled.r#type().into_owned()], &[]);
        let inferred = inferred.map(|mut types| types.remove(0)).map_err(|error| error.to_string());
        let eager = session.place(tiled.clone(), gather()).map(|output| observe(&output));
        let eager = eager.map_err(|error| error.to_string());
        let core = tiled.place(&devices[2], None).map(|output| observe(&output));
        let core = core.map_err(|error| Error::from(error).to_string());
        let staged = replay_placement(&session, &tiled, gather()).map(|output| observe(&output));
        for (path, result) in [("core", &core), ("staged", &staged)] {
            if result != &eager {
                disagreements.push(format!("gathered layout, {path}: {result:?} != eager: {eager:?}"));
            }
        }
        if eager.as_ref().map(|output| &output.r#type) != inferred.as_ref() {
            disagreements.push(format!("gathered layout, eager: {eager:?} != type rule: {inferred:?}"));
        }

        assert!(disagreements.is_empty(), "{} disagreements:\n{}", disagreements.len(), disagreements.join("\n"));
    }

    #[test]
    fn test_session_place_agrees_across_paths_for_values_inside_manual_regions() {
        // Values inside a manual region (i.e., that vary along the manual axis `m`, or that are reduced over it) can
        // only stay on their devices, and nothing can be moved into such a region, so every target that moves a value
        // is rejected on every path with the error of the type rule, even a target that is the placement of the value.
        // Reverse mode and batching of an eager value do not apply to these values, so they are compared on the eager,
        // core, staged, and forward mode paths.
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let pinned = Memory::Host { pinned: true };
        let observe_shards =
            |array: &Array| (array.mesh().devices().to_vec(), array.r#type().into_owned(), shard_values(array));
        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let manual = DeviceMesh::new(manual_mesh.clone(), devices[..2].to_vec()).unwrap();
        let other_manual = DeviceMesh::new(manual_mesh.clone(), devices[2..].to_vec()).unwrap();
        let varying = Sharding::replicated(manual_mesh.clone(), 1).with_varying_manual_axes(["m"]).unwrap();
        let reduced = Sharding::replicated(manual_mesh.clone(), 1).with_reduced_axes(["m"]).unwrap();
        let vector = ArrayType::new_static(DataType::F32, [2]);
        let varying_input = Array::from_host_shards(
            &session,
            vector.clone(),
            manual.clone(),
            Some(varying.clone()),
            [
                (devices[0].id(), values_to_bytes(&[1f32, 2.]).as_slice()),
                (devices[1].id(), values_to_bytes(&[10f32, 20.]).as_slice()),
            ],
        )
        .unwrap();
        let reduced_type = vector.clone().with_sharding(reduced.clone()).unwrap();
        let reduced_input =
            Array::from_host_buffer(&session, reduced_type, manual.clone(), None, values_to_bytes(&[1f32, 2.]))
                .unwrap();
        let plain =
            Array::from_host_buffer(&session, vector.clone(), None, None, values_to_bytes(&[1f32, 2.])).unwrap();
        let manual_error =
            "`place` cannot move values that vary along manual mesh axes or carry reduction state over them"
                .to_string();
        let region_targets = |sharding: &Sharding| {
            vec![
                ("device", Some(PlacementRequest::from(devices[2])), Some(manual_error.clone())),
                ("own device", Some(PlacementRequest::from(devices[0])), Some(manual_error.clone())),
                (
                    "other manual mesh",
                    Some(PlacementRequest::from(Placement::new(other_manual.clone(), sharding.clone()).unwrap())),
                    Some(manual_error.clone()),
                ),
                (
                    "own placement",
                    Some(PlacementRequest::from(Placement::new(manual.clone(), sharding.clone()).unwrap())),
                    Some(manual_error.clone()),
                ),
                ("memory", Some(PlacementRequest::from(pinned)), None),
                ("none", None, None),
            ]
        };
        let inputs = [
            ("varying along a manual axis", varying_input, region_targets(&varying)),
            ("reduced over a manual axis", reduced_input, region_targets(&reduced)),
            (
                "outside any manual region",
                plain,
                vec![(
                    "varying along a manual axis",
                    Some(PlacementRequest::from(Placement::new(manual.clone(), varying.clone()).unwrap())),
                    Some(manual_error.clone()),
                )],
            ),
        ];
        let mut disagreements = Vec::new();
        for (input_name, input, targets) in inputs {
            for (target_name, target, rejection) in targets {
                let cell = format!("{input_name}, target {target_name}");
                let options =
                    || PlacementOptions::new(target.clone(), None::<PlacementRequest>, None::<bool>, None::<bool>);
                let error = RefCell::new(None::<String>);
                let eager = session.place(input.clone(), options()).map(|output| observe_shards(&output));
                let eager = eager.map_err(|error| error.to_string());
                if let Some(rejection) = &rejection
                    && eager.as_ref().err() != Some(rejection)
                {
                    disagreements.push(format!("{cell}, eager: {eager:?} != rejection: {rejection:?}"));
                }
                if rejection.is_none() && eager.is_err() {
                    disagreements.push(format!("{cell}, eager: {eager:?}"));
                }
                let request = target.clone().map(|target| target.padded(1)).transpose().unwrap();
                let core = input.place(&request, None).map(|output| observe_shards(&output));
                let core = core.map_err(|error| Error::from(error).to_string());
                let staged = replay_placement(&session, &input, options()).map(|output| observe_shards(&output));
                let jvp = differentiate_at(input.clone())
                    .jvp(input.clone(), |x| record_placement_error(&error, session.place(x, options())))
                    .map(|(primal, _)| observe_shards(&primal))
                    .map_err(|other| error.take().unwrap_or(other.to_string()));
                for (path, result) in [("core", core), ("staged", staged), ("jvp", jvp)] {
                    if result != eager {
                        disagreements.push(format!("{cell}, {path}: {result:?} != eager: {eager:?}"));
                    }
                }
            }
        }

        assert!(disagreements.is_empty(), "{} disagreements:\n{}", disagreements.len(), disagreements.join("\n"));
    }

    #[test]
    fn test_session_place_agrees_across_paths_for_host_values() {
        // A host value is uploaded onto a target as the placement of a value without a tracked sharding in device
        // memory, so it is rejected or placed as an upload of the same value without a mesh is (a host value ignores
        // sources, so none is given), with the first error of the type rule for such a value, except that both split a
        // dimension unevenly. A traced leaf of that type follows the type rule with the even split included. A
        // sharding target resolves over the default mesh of the session handle that places the value.
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let pinned = Memory::Host { pinned: true };
        let observe_shards =
            |array: &Array| (array.mesh().devices().to_vec(), array.r#type().into_owned(), shard_values(array));
        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let manual = DeviceMesh::new(manual_mesh.clone(), devices[..2].to_vec()).unwrap();
        let varying = Sharding::replicated(manual_mesh, 1).with_varying_manual_axes(["m"]).unwrap();
        let explicit = two_device_mesh(client, MeshAxisType::Explicit);
        let explicit_sharding =
            |dimensions: Vec<ShardingDimension>| Sharding::new(explicit.logical_mesh().clone(), dimensions).unwrap();
        let other_device_mesh = single_device_mesh(devices[2]);
        let other_replicated = Sharding::replicated(other_device_mesh.logical_mesh().clone(), 1);
        let meshed_session = session.clone().with_device_mesh(other_device_mesh.clone());
        let rank_2 = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let targets = [
            (
                "unreduced",
                Some(PlacementRequest::from(
                    Placement::new(
                        explicit.clone(),
                        explicit_sharding(vec![ShardingDimension::Replicated]).with_unreduced_axes(["x"]).unwrap(),
                    )
                    .unwrap(),
                )),
            ),
            (
                "reduced",
                Some(PlacementRequest::from(
                    Placement::new(
                        explicit.clone(),
                        explicit_sharding(vec![ShardingDimension::Replicated]).with_reduced_axes(["x"]).unwrap(),
                    )
                    .unwrap(),
                )),
            ),
            (
                "varying along a manual axis",
                Some(PlacementRequest::from(Placement::new(manual.clone(), varying.clone()).unwrap())),
            ),
            (
                "device with a layout of rank 2",
                Some(PlacementRequest::Device { device: devices[1], memory: None, layout: Some(rank_2.clone()) }),
            ),
            (
                "rank 2",
                Some(PlacementRequest::from(
                    Placement::new(explicit.clone(), explicit_sharding(vec![ShardingDimension::Replicated; 2]))
                        .unwrap(),
                )),
            ),
            (
                "unconstrained",
                Some(PlacementRequest::from(
                    Placement::new(explicit.clone(), explicit_sharding(vec![ShardingDimension::Unconstrained]))
                        .unwrap(),
                )),
            ),
            (
                "uneven split",
                Some(PlacementRequest::from(
                    Placement::new(explicit.clone(), explicit_sharding(vec![ShardingDimension::sharded(["x"])]))
                        .unwrap(),
                )),
            ),
            ("device", Some(PlacementRequest::from(devices[1]))),
            ("memory", Some(PlacementRequest::from(pinned))),
            (
                "mesh",
                Some(PlacementRequest::from(Placement::new(other_device_mesh, other_replicated.clone()).unwrap())),
            ),
            ("sharding", Some(PlacementRequest::from(meshed_session.placement(other_replicated).unwrap()))),
            ("none", None),
        ];
        let host_type = ArrayType::new_static(DataType::F32, [3]);
        let host = CoreArray::from_elements(host_type.clone(), &[1f32, 2., 3.]).unwrap();
        let mut disagreements = Vec::new();
        for (target_name, target) in targets {
            let cell = format!("host value, target {target_name}");
            let session = if target_name == "sharding" { &meshed_session } else { &session };
            let options =
                || PlacementOptions::new(target.clone(), None::<PlacementRequest>, None::<bool>, None::<bool>);
            let uploaded =
                Array::from_host_buffer(session, host_type.clone(), None, None, values_to_bytes(&[1f32, 2., 3.]))
                    .unwrap();
            let host = session
                .place(host.clone(), options())
                .map(|output| (observe_shards(&output), output.is_committed()));
            let host = host.map_err(|error| error.to_string());
            let array =
                session.place(uploaded, options()).map(|output| (observe_shards(&output), output.is_committed()));
            let array = array.map_err(|error| error.to_string());
            if host != array {
                disagreements.push(format!("{cell}: {host:?} != array: {array:?}"));
            }
            let request = target.clone().map(|target| target.padded(1)).transpose().unwrap();
            let inferred =
                PlaceOperation::infer_output_type_allowing_uneven_splits(&host_type, None, request.as_ref(), None);
            let inferred = inferred.map_err(|error| Error::from(error).to_string());
            if host.as_ref().map(|_| ()) != inferred.as_ref().map(|_| ()) {
                disagreements.push(format!("{cell}: {host:?} != type rule: {inferred:?}"));
            }
            let error = RefCell::new(None::<String>);
            let _: Result<TracedProgram<ArrayType, ArrayType>, _> = trace(
                |x: ArrayTracer| record_placement_error(&error, session.place(x.clone(), options())).unwrap_or(x),
                host_type.clone(),
            );
            let traced = error.take();
            let expected = match (target_name, &host) {
                ("uneven split", _) => Some(
                    "`place` target sharding splits dimension #0 with extent 3 into 2 shards, which do not divide it \
                     evenly"
                        .to_string(),
                ),
                (_, Err(error)) => Some(error.clone()),
                (_, Ok(_)) => None,
            };
            if traced != expected {
                disagreements.push(format!("{cell}, traced: {traced:?} != {expected:?}"));
            }
        }

        assert!(disagreements.is_empty(), "{} disagreements:\n{}", disagreements.len(), disagreements.join("\n"));
    }

    #[test]
    fn test_session_place_agrees_across_paths_for_sources_inside_shard_map_bodies() {
        // Inside a `shard_map` body, a source that names devices (i.e., the placement of the local value over the
        // devices of the region, or the sole device of a region of one device) would become the target of the
        // transposed placement of an invariant local value (i.e., one that does not vary along the manual axis), which
        // would move its cotangent. The forward function moves nothing to that source, so it compiles and runs eagerly,
        // and the program of reverse mode, which keeps only the transposed placement, is rejected for its target. The
        // cotangent of a varying local value stays where it lives (refer to `PlaceOperation::transposes_in_place`), so
        // every source compiles and executes on every path for it, as does every other source for an invariant value.
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let pinned = Memory::Host { pinned: true };
        let pinned_seed = |r#type: &ArrayType| r#type.clone().with_memory(pinned);
        let moved_error = format!(
            "cannot lower `{PLACE_OPERATION_NAME}`: a value inside a manual region cannot be moved to another \
             placement",
        );
        let mut disagreements = Vec::new();
        for count in [1usize, 2] {
            let program_logical_mesh =
                LogicalMesh::new(vec![MeshAxis::new("x", count, MeshAxisType::Explicit).unwrap()]).unwrap();
            let program_mesh = DeviceMesh::new(program_logical_mesh.clone(), devices[..count].to_vec()).unwrap();
            let region_mesh = LogicalMesh::new(vec![MeshAxis::new("x", count, MeshAxisType::Manual).unwrap()]).unwrap();
            let region_devices = DeviceMesh::new(region_mesh.clone(), devices[..count].to_vec()).unwrap();
            let r#type = ArrayType::new_static(DataType::F32, [4])
                .with_sharding(Sharding::replicated(program_logical_mesh.clone(), 1))
                .unwrap();
            for is_varying in [false, true] {
                // A specification that splits the value along the manual axis makes the local value vary along it.
                let (specification, local_sharding, output_type) = if is_varying {
                    let split =
                        |mesh: &LogicalMesh| Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]);
                    let varying = Sharding::replicated(region_mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap();
                    let output_type = r#type.clone().with_sharding(split(&program_logical_mesh).unwrap()).unwrap();
                    (split(&region_mesh).unwrap(), varying, output_type)
                } else {
                    let specification = Sharding::replicated(region_mesh.clone(), 1);
                    (specification.clone(), specification, r#type.clone())
                };
                let mut sources = vec![
                    ("none", None, false),
                    ("memory", Some(PlacementRequest::from(Memory::Device)), false),
                    (
                        "layout",
                        Some(PlacementRequest::InPlace {
                            memory: None,
                            layout: Some(Layout::Tiled(TiledLayout::new(vec![0], vec![]))),
                        }),
                        false,
                    ),
                    (
                        "resolved",
                        Some(PlacementRequest::from(
                            Placement::new(region_devices.clone(), local_sharding.clone()).unwrap(),
                        )),
                        !is_varying,
                    ),
                ];
                if count == 1 {
                    sources.push(("device", Some(PlacementRequest::from(devices[0])), !is_varying));
                }
                for (source_name, source, is_rejected) in sources {
                    for (target_name, target) in [("memory", Some(PlacementRequest::from(pinned))), ("none", None)] {
                        let input_name = if is_varying { "varying" } else { "invariant" };
                        let cell = format!(
                            "`shard_map` over {count} devices, {input_name} value, target {target_name}, source \
                             {source_name}",
                        );
                        let options =
                            || PlacementOptions::new(target.clone(), source.clone(), None::<bool>, None::<bool>);
                        let forward: Result<CompiledFunction<ArrayType, ArrayType>, _> = compile(
                            |x: CompileTracer| {
                                shard_map(
                                    |local| {
                                        record_placement_error(&RefCell::new(None), session.place(local, options()))
                                    },
                                    x,
                                    region_mesh.clone(),
                                    specification.clone(),
                                    specification.clone(),
                                    ShardMapOptions::new(),
                                )
                                .unwrap()
                            },
                            r#type.clone(),
                            &session,
                            program_mesh.clone(),
                        );
                        let seed_type = if target.is_some() { pinned_seed(&output_type) } else { output_type.clone() };
                        let reverse: Result<CompiledFunction<(ArrayType, ArrayType), ArrayType>, _> = compile(
                            |(x, seed): (CompileTracer, CompileTracer)| {
                                let (_, pullback) = differentiate_at(x.into_value())
                                    .vjp(|x| {
                                        let x = ValueProjection::<ArrayType>::into_projected(x)?;
                                        let output = shard_map(
                                            |local| {
                                                record_placement_error(
                                                    &RefCell::new(None),
                                                    session.place(local, options()),
                                                )
                                            },
                                            x,
                                            region_mesh.clone(),
                                            specification.clone(),
                                            specification.clone(),
                                            ShardMapOptions::new(),
                                        )
                                        .map_err(|error| {
                                            ProgramError::InvalidArgument { message: error.to_string() }
                                        })?;
                                        Ok(output.into_value())
                                    })
                                    .unwrap();
                                let cotangent = pullback.apply_materialized(seed.into_value()).unwrap();
                                ValueProjection::<ArrayType>::into_projected(cotangent).unwrap()
                            },
                            (r#type.clone(), seed_type),
                            &session,
                            program_mesh.clone(),
                        );
                        let input = Array::from_host_buffer(
                            &session,
                            r#type.clone(),
                            program_mesh.clone(),
                            None,
                            values_to_bytes(&[1f32, 2., 3., 4.]),
                        )
                        .unwrap();
                        let eager = shard_map(
                            |local| record_placement_error(&RefCell::new(None), session.place(local, options())),
                            input,
                            region_mesh.clone(),
                            specification.clone(),
                            specification.clone(),
                            ShardMapOptions::new(),
                        );
                        let results = [
                            ("forward", forward.map(|_| ()).map_err(|error| error.to_string())),
                            ("reverse", reverse.map(|_| ()).map_err(|error| error.to_string())),
                            ("eager", eager.map(|_| ()).map_err(|error| error.to_string())),
                        ];
                        for (path, result) in results {
                            let expected =
                                if is_rejected && path == "reverse" { Err(moved_error.clone()) } else { Ok(()) };
                            if result != expected {
                                disagreements.push(format!("{cell}, {path}: {result:?} != {expected:?}"));
                            }
                        }
                    }
                }
            }
        }

        assert!(disagreements.is_empty(), "{} disagreements:\n{}", disagreements.len(), disagreements.join("\n"));
    }

    // Worker of `test_session_place_rejects_unavailable_value_comparisons_on_two_cpu_processes`, which returns
    // immediately unless that test launched it as one of its two processes.
    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_session_place_rejects_unavailable_value_comparisons_cpu_process_helper() {
        run_as_cpu_process(|client, devices| {
            let rank = client.process_index().unwrap();
            let local = devices[rank];
            let session = Session::builder().client(client).remote_devices([devices[1 - rank]]).build().unwrap();
            assert_eq!(session.devices().unwrap(), devices);
            let mesh = session.mesh([MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
            assert_eq!(mesh.devices(), devices);
            let logical_mesh = mesh.logical_mesh().clone();
            let target = Placement::new(mesh, Sharding::replicated(logical_mesh, 1)).unwrap();
            let r#type = ArrayType::new_static(DataType::U8, [2]);
            let host = || CoreArray::new(r#type.clone(), vec![rank as u8; 2]).unwrap();
            let check_rejected = |output: Result<Array, Error>| {
                assert!(matches!(output, Err(Error::Session(SessionError::PlacementComparisonUnavailable))));
            };

            // The declarations make both ranks reject even when one candidate can see every device.
            let comparison = ValueComparison::new(target.mesh().clone(), vec![local]);
            let candidate = (rank == 0).then_some(&comparison);
            let comparison = candidate.or(ValueComparison::of_session(&session).unwrap());
            check_rejected(session.place_with_comparison(host(), PlacementOptions::to(target.clone()), comparison));
            check_rejected(session.place(host(), PlacementOptions::to(target.clone())));
            let uploaded = Array::from_host_buffer(&session, r#type.clone(), None, None, [rank as u8; 2]).unwrap();
            let core_error = uploaded.place(&target, None).unwrap_err();
            assert_eq!(
                core_error.downcast_custom::<SessionError>(),
                Some(&SessionError::PlacementComparisonUnavailable)
            );
            check_rejected(uploaded.to_placement(&session, target.clone()));
            check_rejected(uploaded.into_placement(&session, target.clone()));
        });
    }

    #[test]
    fn test_session_place_rejects_unavailable_value_comparisons_on_two_cpu_processes() {
        run_cpu_processes(
            "arrays::placement::tests::test_session_place_rejects_unavailable_value_comparisons_cpu_process_helper",
            2,
        );
    }

    // Worker of `test_session_place_skips_value_comparisons_on_two_cpu_processes`, which returns immediately unless
    // that test launched it as one of its two processes.
    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_session_place_skips_value_comparisons_cpu_process_helper() {
        run_as_cpu_process(|client, devices| {
            let rank = client.process_index().unwrap();
            // Placements onto a device of this process and placements of a session that disables the comparison compare
            // nothing, so they succeed although the declared devices make the comparison unavailable.
            let new_session = || Session::builder().client(client).remote_devices([devices[1 - rank]]);
            let session = new_session().build().unwrap();
            let mesh = session.mesh([MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
            let logical_mesh = mesh.logical_mesh().clone();
            let target = Placement::new(mesh, Sharding::replicated(logical_mesh, 1)).unwrap();
            let r#type = ArrayType::new_static(DataType::U8, [2]);
            let host = CoreArray::new(r#type, vec![rank as u8; 2]).unwrap();
            let local_placement = session.place(3u8, PlacementOptions::to(devices[rank])).unwrap();
            assert_eq!(local_placement.copy_to_host_bytes().unwrap(), [3]);
            let unchecked = new_session().value_comparison(false).build().unwrap();
            let placed = unchecked.place(host, PlacementOptions::to(target)).unwrap();
            assert_eq!(placed.copy_to_host_bytes().unwrap(), [rank as u8; 2]);
        });
    }

    #[test]
    fn test_session_place_skips_value_comparisons_on_two_cpu_processes() {
        run_cpu_processes("arrays::placement::tests::test_session_place_skips_value_comparisons_cpu_process_helper", 2);
    }

    // Worker of `test_session_place_rejects_unsupported_layouts_on_two_cpu_processes`, which returns immediately
    // unless that test launched it as one of its two processes.
    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_session_place_rejects_unsupported_layouts_cpu_process_helper() {
        run_as_cpu_process(|client, devices| {
            let rank = client.process_index().unwrap();
            let session = Session::builder().client(client).remote_devices([devices[1 - rank]]);
            let unchecked = session.value_comparison(false).build().unwrap();
            let r#type = ArrayType::new_static(DataType::U8, [2]);
            let host = || CoreArray::new(r#type.clone(), vec![rank as u8; 2]).unwrap();

            // CPU rejects the unsupported layout on the owner and nonowner before either can submit a cross-host leaf.
            let committed = unchecked.place(host(), PlacementOptions::to(devices[0])).unwrap();
            assert_eq!(committed.addressable_shards().count(), usize::from(rank == 0));
            let layouts = [
                (Layout::Strided(StridedLayout::new(vec![2])), "cpu PJRT does not support strided device layouts"),
                (
                    Layout::Tiled(TiledLayout::new(vec![0], vec![Tile::new(vec![TileDimension::Sized(2)])])),
                    "cpu PJRT does not support device layouts with tiles",
                ),
            ];
            for (layout, layout_error) in layouts {
                let upload =
                    PlacementRequest::Device { device: devices[0], memory: None, layout: Some(layout.clone()) };
                let relayout = PlacementRequest::InPlace { memory: None, layout: Some(layout.clone()) };
                let transfer = PlacementRequest::from(devices[1]);
                let unsupported = |error: &Error| {
                    matches!(error, Error::Pjrt(PjrtError::InvalidArgument { message, .. })
                        if message == layout_error)
                };
                let assert_leaf_error = |error: Error, expected_path: &str| {
                    assert!(
                        matches!(&error, Error::Array(error) if matches!(error.as_ref(),
                            ArrayError::FailedParameterPlacement { source, path }
                                if path.to_string() == expected_path && unsupported(source)
                        )),
                        "{error:?}"
                    );
                };
                let statistics = unchecked.array_statistics();
                let error = unchecked
                    .place(
                        (PlaceInput::from(host()), PlaceInput::from(committed.clone())),
                        PlacementOptions::to((upload, transfer.clone())),
                    )
                    .unwrap_err();
                assert_leaf_error(error, "$.0");
                let error = unchecked
                    .place(
                        (committed.clone(), committed.clone()),
                        PlacementOptions::to((relayout.clone(), transfer.clone())),
                    )
                    .unwrap_err();
                assert_leaf_error(error, "$.0");
                let error = unchecked
                    .place((committed.clone(), committed.clone()), PlacementOptions::to((transfer, relayout.clone())))
                    .unwrap_err();
                assert_leaf_error(error, "$.1");

                // `Array::to_placement`, `Array::place_storage`, and the host constructors reject the same layouts with
                // the same errors, also on the rank that uploads no shard, and create no array. They share this worker
                // so that the job needs no second pair of processes.
                let source_mesh = committed.mesh().clone();
                let bytes = [7u8, 9];
                let target_type = r#type.clone().with_layout(layout.clone());
                let dense = Array::from_host_buffer(&unchecked, target_type.clone(), source_mesh.clone(), None, &bytes);
                let shard = (rank == 0).then_some((devices[0].id(), &bytes));
                let sharded = Array::from_host_shards(&unchecked, target_type, source_mesh.clone(), None, shard);
                let directly_placed = committed.to_placement(&unchecked, relayout.clone());
                let converted = committed.clone().place_storage(Memory::Device, Some(layout.clone()));
                for rejected in [dense, sharded, directly_placed, converted] {
                    let error = rejected.unwrap_err();
                    assert!(unsupported(&error), "{error:?}");
                }
                assert_eq!(unchecked.array_statistics().created_arrays, statistics.created_arrays);

                let unchanged = committed.to_placement(&unchecked, devices[0]).unwrap();
                assert_eq!(unchanged, committed);

                let zero_type = ArrayType::new_static(DataType::Zero, [2]);
                let zero =
                    Array::from_host_buffer(&unchecked, zero_type.clone(), source_mesh.clone(), None, []).unwrap();
                let placed = unchecked.place(zero, PlacementOptions::to(relayout)).unwrap();
                let zero_type = zero_type.with_layout(layout.clone());
                let zero_shard = (rank == 0).then_some((devices[0].id(), &[] as &[u8]));
                let sharded =
                    Array::from_host_shards(&unchecked, zero_type.clone(), source_mesh.clone(), None, zero_shard)
                        .unwrap();
                let dense = Array::from_host_buffer(&unchecked, zero_type, source_mesh, None, []).unwrap();
                for zero in [placed, sharded, dense] {
                    assert_eq!(zero.layout(), Some(&layout));
                    assert!(zero.addressable_shards().all(|shard| shard.buffer().is_none()));
                    zero.block_until_ready().unwrap();
                }
            }
        });
    }

    #[test]
    fn test_session_place_rejects_unsupported_layouts_on_two_cpu_processes() {
        run_cpu_processes(
            "arrays::placement::tests::test_session_place_rejects_unsupported_layouts_cpu_process_helper",
            2,
        );
    }

    // Worker of `test_session_place_compares_equal_values_on_two_cuda_processes`, which returns immediately unless that
    // test launched it as one of its two ranks, and asserts the outcomes of its rank.
    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_session_place_compares_equal_values_cuda_process_helper() {
        run_as_cuda_rank(|client| {
            // Equal values pass the comparison, and each rank holds its shard of the value.
            let rank = client.process_index().unwrap();
            let shard: Result<Vec<Vec<f32>>, String> = Ok([vec![vec![1., 2.]], vec![vec![3., 4.]]][rank].clone());
            assert_eq!(
                place_values_of_ranks(client, false),
                [shard.clone(), shard, Ok(vec![vec![17., 18., 19., 20.]])],
            );
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_session_place_compares_equal_values_on_two_cuda_processes() {
        run_cuda_rank_pair("arrays::placement::tests::test_session_place_compares_equal_values_cuda_process_helper");
    }

    // Worker of `test_session_place_rejects_different_values_on_two_cuda_processes`, which returns immediately unless
    // that test launched it as one of its two ranks, and asserts the outcomes of its rank.
    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_session_place_rejects_different_values_cuda_process_helper() {
        run_as_cuda_rank(|client| {
            // Different values fail the results of both ranks alike without a hang, and the ranks stay in step.
            let different_values: Result<Vec<Vec<f32>>, String> = Err(DIFFERENT_VALUES_MESSAGE.to_string());
            assert_eq!(
                place_values_of_ranks(client, true),
                [different_values.clone(), different_values, Ok(vec![vec![17., 18., 19., 20.]])],
            );
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_session_place_rejects_different_values_on_two_cuda_processes() {
        run_cuda_rank_pair("arrays::placement::tests::test_session_place_rejects_different_values_cuda_process_helper");
    }

    // Worker of `test_session_place_reports_an_invalid_leaf_on_two_cuda_processes`, which returns immediately unless
    // that test launched it as one of its two ranks, and asserts the outcomes of its rank.
    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_session_place_reports_an_invalid_leaf_cuda_process_helper() {
        run_as_cuda_rank(|client| {
            // Places a tree of two `bool[2]` host values with `Session::place` onto a split over the two global
            // devices, whose first leaf on rank 0 holds a byte that does not encode a `bool`, and then places an array
            // of rank 0 replicated onto both devices.
            let rank = client.process_index().unwrap();
            let devices = client
                .devices()
                .unwrap()
                .iter()
                .map(|device| Device::from_pjrt(device).unwrap())
                .collect::<Vec<_>>();
            let session = Session::builder().client(client).build().unwrap();
            let axis = |size: usize| MeshAxis::new("x", size, MeshAxisType::Explicit).unwrap();
            let forward_mesh = DeviceMesh::new(LogicalMesh::new(vec![axis(2)]).unwrap(), devices.clone()).unwrap();
            let process_0_mesh = DeviceMesh::new(LogicalMesh::new(vec![axis(1)]).unwrap(), vec![devices[0]]).unwrap();
            let split = Sharding::new(forward_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]);
            let split = Placement::new(forward_mesh.clone(), split.unwrap()).unwrap();
            let bools = ArrayType::new_static(DataType::Boolean, [2]);
            let mut first = CoreArray::new(bools.clone(), vec![1, 0]).unwrap();
            if rank == 0 {
                first.storage_bytes_mut()[0] = 2;
            }
            let tree = vec![first, CoreArray::new(bools, vec![0, 1]).unwrap()];
            let placed = session.place(tree, PlacementOptions::to(PlacementRequest::from(split)));
            let placed = placed
                .map(|placed| {
                    let ready = placed.iter().map(|placed| placed.block_until_ready().map_err(placement_error_outcome));
                    ready.collect::<Vec<_>>()
                })
                .map_err(placement_error_outcome);

            // The rank whose first leaf is invalid returns its error after the one comparison of the call, the other
            // rank sees the mismatch on both of its results, and both stay in step.
            let invalid =
                "failed to place the parameter at `$[0]`: array element 0 has invalid `bool` byte encoding [2]";
            let different_values: Result<(), String> = Err(DIFFERENT_VALUES_MESSAGE.to_string());
            assert_eq!(placed, [Err(invalid.to_string()), Ok(vec![different_values.clone(), different_values])][rank]);
            let r#type = ArrayType::new_static(DataType::F32, [4]);
            let r#type = r#type.with_sharding(Sharding::replicated(process_0_mesh.logical_mesh().clone(), 1)).unwrap();
            let values = values_to_bytes(&[17f32, 18., 19., 20.]);
            let source = Array::from_host_buffer(&session, r#type, process_0_mesh, None, values).unwrap();
            assert_eq!(
                place_replicated_across_ranks(&session, &source, &forward_mesh),
                Ok(vec![vec![17., 18., 19., 20.]]),
            );
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_session_place_reports_an_invalid_leaf_on_two_cuda_processes() {
        run_cuda_rank_pair("arrays::placement::tests::test_session_place_reports_an_invalid_leaf_cuda_process_helper");
    }

    // Worker of `test_session_place_rejects_a_call_with_an_invalid_leaf_on_two_cuda_processes`, which returns
    // immediately unless that test launched it as one of its two ranks, and asserts the outcomes of its rank.
    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_session_place_rejects_a_call_with_an_invalid_leaf_cuda_process_helper() {
        run_as_cuda_rank(|client| {
            // Places with one call of `Session::place` a host value split over the two global devices, which is a
            // scalar on rank 0 (so that only rank 0 rejects it) and a vector on rank 1, and an array of rank 0
            // replicated onto both devices, which rank 1 can only receive from rank 0. Then places that array once
            // more on its own.
            let rank = client.process_index().unwrap();
            let devices = client
                .devices()
                .unwrap()
                .iter()
                .map(|device| Device::from_pjrt(device).unwrap())
                .collect::<Vec<_>>();
            let session = Session::builder().client(client).build().unwrap();
            let axis = |size: usize| MeshAxis::new("x", size, MeshAxisType::Explicit).unwrap();
            let forward_mesh = DeviceMesh::new(LogicalMesh::new(vec![axis(2)]).unwrap(), devices.clone()).unwrap();
            let process_0_mesh = DeviceMesh::new(LogicalMesh::new(vec![axis(1)]).unwrap(), vec![devices[0]]).unwrap();
            let split = Sharding::new(forward_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]);
            let split = PlacementRequest::from(Placement::new(forward_mesh.clone(), split.unwrap()).unwrap());
            let replicated = Sharding::replicated(forward_mesh.logical_mesh().clone(), 1);
            let replicated = PlacementRequest::from(Placement::new(forward_mesh.clone(), replicated).unwrap());
            let r#type = ArrayType::new_static(DataType::F32, [4]);
            let values = values_to_bytes(&[17f32, 18., 19., 20.]);
            let source = Array::from_host_buffer(&session, r#type, process_0_mesh, None, values).unwrap();
            let host = match rank {
                0 => CoreArray::scalar(1f32).unwrap(),
                _ => CoreArray::vector(vec![1f32, 2.]).unwrap(),
            };
            let tree = vec![PlaceInput::Host(host), PlaceInput::Runtime(source.clone())];
            let options = PlacementOptions::to(vec![split, replicated]);
            let placed = session.place(tree, options).and_then(|placed| {
                placed.iter().try_for_each(Array::block_until_ready)?;
                Ok(placed)
            });
            let placed = placed.map(|placed| placed.iter().map(read_addressable_f32s).collect::<Vec<_>>());

            // The first leaf is invalid on rank 0 only, and the second leaf of the same call is a cross-host move that
            // rank 1 could only complete with rank 0. Both ranks return without placing either leaf, rank 0 with its
            // own error and rank 1 with the mismatch, and both stay in step for the placement that follows.
            let leaf = "failed to place the parameter at `$[0]`";
            let error = "`place` target sharding rank (1) does not match the input rank (0)";
            let errors = [format!("{leaf}: {error}"), DIFFERENT_VALUES_MESSAGE.to_string()];
            assert_eq!(placed.map_err(placement_error_outcome), Err(errors[rank].clone()));
            assert_eq!(
                place_replicated_across_ranks(&session, &source, &forward_mesh),
                Ok(vec![vec![17., 18., 19., 20.]]),
            );
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_session_place_rejects_a_call_with_an_invalid_leaf_on_two_cuda_processes() {
        run_cuda_rank_pair(
            "arrays::placement::tests::test_session_place_rejects_a_call_with_an_invalid_leaf_cuda_process_helper",
        );
    }

    #[test]
    fn test_placement_option_values_from_prefix() {
        let session = Session::builder().client(cpu_client(1)).build().unwrap();
        let array = session.place(2.5f32, PlacementOptions::default()).unwrap();
        let structure = vec![array; 4_000].parameter_structure();

        // Omitted and root options are stored once even when the input has many parameters.
        let expansion = PlacementOptionValues::from_prefix(None::<PlacementRequest>, &structure);
        assert!(matches!(expansion, Ok(PlacementOptionValues::Uniform(None))));
        let expansion = PlacementOptionValues::from_prefix(Some(false), &structure);
        assert!(matches!(expansion, Ok(PlacementOptionValues::Uniform(Some(false)))));
    }

    #[test]
    fn test_array_resolve_placement_request() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![devices[0]],
        )
        .unwrap();
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]),
            mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();

        // A device target on the device that the array lives on keeps the mesh of the array, even though that mesh
        // names its axis differently from the mesh that a device target resolves to, and its sharding without the
        // pending reduction along its axis of size one, which one device completes.
        let memory = Memory::Host { pinned: true };
        let target = |device| PlacementRequest::Device { device, memory: Some(memory), layout: None };
        let replicated = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap();
        assert_eq!(
            array.resolve_placement_request(target(devices[0])).unwrap(),
            replicated.clone().with_memory(memory),
        );
        let unreduced = Sharding::replicated(mesh.logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let unreduced_array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]).with_sharding(unreduced).unwrap(),
            mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();
        assert_eq!(
            unreduced_array.resolve_placement_request(target(devices[0])).unwrap(),
            replicated.clone().with_memory(memory),
        );

        // Any other device keeps the logical mesh and the sharding of an array on a mesh of one device, laid over
        // that device, so that the array keeps how its dimensions are sharded.
        let other_mesh = DeviceMesh::new(mesh.logical_mesh().clone(), vec![devices[1]]).unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let sharded_array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]).with_sharding(sharding.clone()).unwrap(),
            mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();
        assert_eq!(
            sharded_array.resolve_placement_request(target(devices[1])).unwrap(),
            Placement::new(other_mesh.clone(), sharding).unwrap().with_memory(memory),
        );

        // An array on more devices resolves like a value of its rank, which gathers it onto the device.
        let two_device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap(),
            devices.clone(),
        )
        .unwrap();
        let split_array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]),
            two_device_mesh,
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();
        assert_eq!(
            split_array.resolve_placement_request(target(devices[1])).unwrap(),
            Placement::single_device(devices[1], 1).with_memory(memory),
        );

        // A device target without a memory selects the default memory of its device, as JAX's `device_put` does,
        // whereas a sharding target without a memory selects none, so the array keeps its own memory.
        assert_eq!(
            array.resolve_placement_request(devices[0].into()).unwrap(),
            replicated.clone().with_memory(Memory::Device),
        );
        assert_eq!(
            array.resolve_placement_request(devices[1].into()).unwrap(),
            Placement::new(other_mesh, Sharding::replicated(mesh.logical_mesh().clone(), 1))
                .unwrap()
                .with_memory(Memory::Device),
        );
        let configured = session.with_device_mesh(mesh.clone());
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        assert_eq!(
            array.resolve_placement_request(configured.placement(sharding).unwrap().into()).unwrap(),
            replicated
        );
    }

    #[test]
    fn test_array_resolve_placement_request_in_place() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]),
            mesh.clone(),
            Some(sharding.clone()),
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();

        // An in-place target resolves to the mesh and the sharding of the array, with the memory and the layout that
        // it names, and without them the array keeps its own.
        let memory = Memory::Host { pinned: true };
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let placement = Placement::new(mesh.clone(), sharding).unwrap();
        assert_eq!(
            array
                .resolve_placement_request(PlacementRequest::InPlace {
                    memory: Some(memory),
                    layout: Some(layout.clone()),
                })
                .unwrap(),
            placement.clone().with_memory(memory).with_layout(layout.clone()),
        );
        assert_eq!(
            array
                .resolve_placement_request(PlacementRequest::InPlace { memory: None, layout: Some(layout.clone()) })
                .unwrap(),
            placement.with_layout(layout),
        );

        // An in-place target that names neither a memory nor a layout changes nothing, so it is rejected.
        let error = array
            .resolve_placement_request(PlacementRequest::InPlace { memory: None, layout: None })
            .unwrap_err();
        assert!(matches!(&error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
        assert_eq!(error.to_string(), "`place` in-place target must name a memory or a layout");
    }

    #[test]
    fn test_array_resolve_placement_request_completes_reduction_state_of_device_targets() {
        let session = Session::builder().client(cpu_client(1)).build().unwrap();
        let device = session.addressable_devices().unwrap()[0];
        let upload = |session: &Session, sharding, value: f32| {
            Array::from_host_buffer(
                session,
                ArrayType::scalar(DataType::F32),
                None,
                sharding,
                values_to_bytes(&[value]),
            )
            .unwrap()
        };
        let anonymous = upload(&session, None, 3.0);
        assert!(!anonymous.is_committed());
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![device]).unwrap();
        let session = session.with_device_mesh(mesh.clone());
        let replicated = Sharding::replicated(logical_mesh, 0);
        let plain = upload(&session, Some(replicated.clone()), 1.0);
        let observe = |array: &Array| {
            assert_eq!(array.mesh(), &mesh);
            assert_eq!(array.sharding(), &replicated);
            assert!(array.is_committed());
            shard_values(array)
        };
        let (saved_plain, plain_pushforward) =
            differentiate_at(plain.clone()).linearize(|x| x.place(&device, None)).unwrap();
        assert_eq!(observe(&saved_plain), vec![vec![1.0f32]]);
        assert_eq!(observe(&plain_pushforward.apply(plain.clone()).unwrap()), vec![vec![1.0f32]]);

        for sharding in [
            replicated.clone().with_reduced_axes(["x"]).unwrap(),
            replicated.clone().with_unreduced_axes(["x"]).unwrap(),
        ] {
            let input = upload(&session, Some(sharding.clone()), 2.0);
            assert_eq!(input.sharding(), &sharding);
            assert_eq!(input.r#type(), plain.r#type());
            let eager = input.place(&device, None).unwrap();
            let (value, derivative) =
                differentiate_at(input.clone()).jvp(input.clone(), |x| x.place(&device, None)).unwrap();
            let (saved_value, pushforward) =
                differentiate_at(input.clone()).linearize(|x| x.place(&device, None)).unwrap();
            let saved_derivative = pushforward.apply(input.clone()).unwrap();
            for output in [eager, value, derivative, saved_value, saved_derivative] {
                assert_eq!(observe(&output), vec![vec![2.0f32]]);
            }

            // The plain primal's saved target cannot clear the tangent's physical reduction state.
            let (value, derivative) =
                differentiate_at(plain.clone()).jvp(input.clone(), |x| x.place(&device, None)).unwrap();
            assert_eq!(observe(&value), vec![vec![1.0f32]]);
            assert_eq!(observe(&derivative), vec![vec![2.0f32]]);
            assert_eq!(
                plain_pushforward.apply(input.clone()).unwrap_err(),
                ProgramError::Type(TypeError::invalid(
                    "`place` cannot change reduction state; use a reduction (or the corresponding collective) instead",
                )),
            );

            // The reduced primal's saved target can clear state, but it retains the anonymous tangent's mesh.
            let (value, derivative) =
                differentiate_at(input).jvp(anonymous.clone(), |x| x.place(&device, None)).unwrap();
            assert_eq!(observe(&value), vec![vec![2.0f32]]);
            assert_eq!(observe(&derivative), vec![vec![3.0f32]]);
            assert_eq!(
                pushforward.apply(anonymous.clone()).unwrap_err(),
                ProgramError::Type(TypeError::invalid(format!(
                    "encountered output type {} which is incompatible with the region's declared type {}",
                    anonymous.r#type(),
                    plain.r#type(),
                ))),
            );
        }
    }

    #[test]
    fn test_session_place_names_failed_parameter() {
        let session = Session::builder().client(cpu_client(1)).build().unwrap();
        let upload = |shape: &[usize]| {
            let count = shape.iter().product::<usize>();
            Array::from_host_buffer(
                &session,
                ArrayType::new_static(DataType::F32, shape.to_vec()),
                None,
                None,
                values_to_bytes(&vec![1f32; count]),
            )
            .unwrap()
        };
        let (matrix, vector) = (upload(&[2, 2]), upload(&[2]));
        let matrix_placement = Placement::single_device(vector.mesh().devices()[0], 2);
        let empty = PlacementRequest::InPlace { memory: None, layout: None };
        let target_message = "`place` in-place target must name a memory or a layout";
        let rank_message = "`place` target sharding rank (2) does not match the input rank (1)";
        let path_of = |error: &Error| failed_parameter(error).map(|(path, source)| (path, source.to_string()));

        // An option that the validation pass rejects for the second leaf names that leaf.
        let error = session
            .place(
                (matrix.clone(), vector.clone()),
                PlacementOptions::<_, PlacementRequest, bool, bool>::new(
                    Some((None, Some(empty.clone()))),
                    None,
                    None,
                    None,
                ),
            )
            .unwrap_err();
        assert_eq!(path_of(&error), Some(("$.1".to_string(), target_message.to_string())));
        assert_eq!(error.to_string(), format!("failed to place the parameter at `$.1`: {target_message}"));

        // An error of the type rule for the second leaf names that leaf, and it is reported before any leaf is placed.
        let options = PlacementOptions::to(matrix_placement.clone());
        let error = session.place((matrix, vector.clone()), options).unwrap_err();
        assert_eq!(path_of(&error), Some(("$.1".to_string(), rank_message.to_string())));
        assert_eq!(error.to_string(), format!("failed to place the parameter at `$.1`: {rank_message}"));

        // The same failures of a lone value at the root keep their own errors.
        let error = session.place(vector.clone(), PlacementOptions::to(empty)).unwrap_err();
        assert_eq!((path_of(&error), error.to_string()), (None, target_message.to_string()));
        let error = session.place(vector, PlacementOptions::to(matrix_placement)).unwrap_err();
        assert!(matches!(&error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
        assert_eq!(error.to_string(), rank_message);
    }

    #[test]
    fn test_session_place_prefixes_paths_of_nested_placements() {
        /// Leaf that places a pair of copies of `array` of its own through `Session::place`, which fails at the
        /// second element of that pair. When `donate` is `true`, that element is donated while it may alias.
        /// Otherwise, it has an in-place target that names nothing.
        struct PairPlacingLeaf {
            array: Array,
            donate: bool,
        }

        impl Parameter for PairPlacingLeaf {}

        impl PlaceableParameter<Array> for PairPlacingLeaf {}

        impl Placeable for PairPlacingLeaf {
            type Output = Array;

            fn place_parameter(
                self,
                session: &Session,
                target: Option<PlacementRequest>,
                source: Option<PlacementRequest>,
                donate: bool,
                may_alias: Option<bool>,
            ) -> Result<Array, Error> {
                validate_parameter_options(ParameterPath::root, target.as_ref(), source.as_ref(), donate, may_alias)?;
                let pair = (self.array.clone(), self.array);
                let (first, _) = if self.donate {
                    let options = PlacementOptions::<PlacementRequest, PlacementRequest, _, _>::new(
                        None,
                        None,
                        Some((false, true)),
                        Some(Some(true)),
                    );
                    session.place(pair, options)?
                } else {
                    let empty = PlacementRequest::InPlace { memory: None, layout: None };
                    let target = Some((None, Some(empty)));
                    let options = PlacementOptions::<_, PlacementRequest, bool, bool>::new(target, None, None, None);
                    session.place(pair, options)?
                };
                Ok(first)
            }
        }

        let session = Session::builder().client(cpu_client(1)).build().unwrap();
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]),
            None,
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();
        let place = |donate| {
            let input = BTreeMap::from([("weights", PairPlacingLeaf { array: array.clone(), donate })]);
            session.place(input, PlacementOptions::default()).unwrap_err()
        };

        // A failure of a placement that a leaf makes of a tree of its own keeps its typed error, and its path names
        // the leaf followed by the element of that tree.
        let error = place(false);
        let target_message = "`place` in-place target must name a memory or a layout";
        let (path, source) = failed_parameter(&error).unwrap();
        assert_eq!(path, "$[\"weights\"].1");
        assert!(matches!(source, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
        assert_eq!(source.to_string(), target_message);
        assert_eq!(error.to_string(), format!("failed to place the parameter at `$[\"weights\"].1`: {target_message}"));

        let error = place(true);
        assert!(matches!(
            &error,
            Error::Array(error) if matches!(
                error.as_ref(),
                ArrayError::DonatedArrayMayAlias { path } if path.to_string() == "$[\"weights\"].1",
            ),
        ));
        assert_eq!(error.to_string(), "array `$[\"weights\"].1` cannot be donated while `may_alias` is `Some(true)`");
    }

    #[test]
    fn test_place_operation_bindings_pads_short_shardings() {
        // The core `Place` capability pads a sharding with fewer dimensions than an array with trailing replicated
        // dimensions, as `Session::place` and `Array::to_placement` do and as pinned JAX pads a short partition
        // specification, so a split of the first dimension applies to a matrix.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let logical_mesh = mesh.logical_mesh().clone();
        let split = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let split_matrix =
            Sharding::new(logical_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2, 2]);
        let matrix =
            Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(&[1f32, 2., 3., 4.])).unwrap();
        let placed = matrix.place(&Placement::new(mesh.clone(), split).unwrap(), None).unwrap();
        assert_eq!(placed.mesh(), &mesh);
        assert_eq!(placed.sharding(), &split_matrix);
        assert_eq!(shard_values(&placed), vec![vec![1., 2.], vec![3., 4.]]);
        assert!(placed.is_committed());
    }

    // Worker of `test_place_operation_bindings_admits_an_invalid_leaf_on_two_cuda_processes`, which returns
    // immediately unless that test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_place_operation_bindings_admits_an_invalid_leaf_cuda_process_helper() {
        run_as_cuda_rank(|client| {
            let rank = client.process_index().unwrap();
            let session = Session::builder().client(client).build().unwrap();
            let mesh = session.mesh([MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
            let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
            let target = Placement::new(mesh, sharding).unwrap();
            let different_values = |output: Array| {
                assert!(matches!(
                    output.block_until_ready(),
                    Err(Error::Pjrt(PjrtError::FailedPrecondition { message, .. }))
                        if message == DIFFERENT_VALUES_MESSAGE,
                ));
            };

            // The core `Place` of an array whose shape differs between the ranks reaches concrete dispatch on both.
            // Rank 0 rejects its scalar for the split target, but it still takes part in the two value comparisons of
            // admission before it returns its error, so rank 1 sees the mismatch when its result is ready.
            let shape = if rank == 0 { vec![] } else { vec![2] };
            let input = Array::from_host_buffer(
                &session,
                ArrayType::new_static(DataType::U8, shape),
                None,
                None,
                vec![1u8; 1 + rank],
            )
            .unwrap();
            assert!(input.is_locally_assigned());
            let before = session.next_value_comparison();
            let result = input.place(&target, None);
            if rank == 0 {
                assert_eq!(
                    result.unwrap_err(),
                    ProgramError::Type(TypeError::invalid(
                        "`place` target sharding rank (1) does not match the input rank (0)",
                    )),
                );
            } else {
                different_values(result.unwrap());
            }
            assert_eq!(session.next_value_comparison(), before + 2);

            // A placement of equal values afterwards takes two comparisons too and places the shard of each rank.
            let input =
                Array::from_host_buffer(&session, ArrayType::new_static(DataType::U8, [4]), None, None, [2u8; 4])
                    .unwrap();
            let before = session.next_value_comparison();
            let output = input.place(&target, None).unwrap();
            output.block_until_ready().unwrap();
            let shards = output.addressable_shards().map(|shard| shard.copy_to_host_bytes().unwrap());
            assert_eq!(shards.collect::<Vec<_>>(), [vec![2, 2]]);
            assert_eq!(session.next_value_comparison(), before + 2);
        });
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_place_operation_bindings_admits_an_invalid_leaf_on_two_cuda_processes() {
        run_cuda_rank_pair(
            "arrays::placement::tests::test_place_operation_bindings_admits_an_invalid_leaf_cuda_process_helper",
        );
    }

    #[test]
    fn test_admit_and_place_arguments_checks_only_data_needed_by_an_unshared_leaf() {
        let client = cpu_client(2);
        let remote = Device::new(1000, 1);
        let session = session_with_remote_devices(client, [remote]);
        let devices = session.addressable_devices().unwrap();
        let comparison = ValueComparison::new(participant_mesh_of(devices.clone()).unwrap(), devices.clone());
        let committed = session.place(7u8, PlacementOptions::to(devices[0])).unwrap();
        let (event, promise) = client.event(()).unwrap();
        promise.set(Some(PjrtError::invalid_argument("unshared producer failed"))).unwrap();
        let fence = ExecutionFence::new(vec![event]);
        assert_eq!(fence.block_until_ready().unwrap_err().to_string(), "unshared producer failed");
        let unshared = session.place(9u8, PlacementOptions::default()).unwrap().with_execution_fence(fence);
        assert!(unshared.is_locally_assigned());

        let run = |parameter: PlaceInput<Array>, target, may_alias| {
            let arguments = vec![
                PlacementArgument { parameter, context: (), target, source: None, donate: false, may_alias },
                PlacementArgument {
                    parameter: PlaceInput::from(committed.clone()),
                    context: (),
                    target: Some(PlacementRequest::from(remote)),
                    source: None,
                    donate: false,
                    may_alias: None,
                },
            ];
            admit_and_place_arguments(
                &session,
                Some(&comparison),
                arguments,
                Ok(()),
                // The declared peer cannot receive native sends in this test. Stop at dispatch of its argument.
                |index, argument| match index {
                    0 => argument.parameter.place_parameter(&session, argument.target, None, false, argument.may_alias),
                    _ => Ok(committed.clone()),
                },
                |_, error| error,
            )
        };

        // A leaf that this process placed on its own and that stays where it is needs none of its data, so the
        // failure of its producer does not stop admission, which takes its two value comparisons, and surfaces only
        // when the result is ready. A copy of that leaf needs its data, so the failure is the error of the call.
        let position = session.next_value_comparison();
        let placed = run(PlaceInput::from(unshared.clone()), None, None).unwrap();
        assert_eq!(session.next_value_comparison(), position + 2);
        assert_eq!(placed[0].block_until_ready().unwrap_err().to_string(), "unshared producer failed");
        assert_eq!(placed[1].copy_to_host_bytes().unwrap(), [7]);
        assert_eq!(
            run(PlaceInput::from(unshared), None, Some(false)).unwrap_err().to_string(),
            "unshared producer failed",
        );

        // A copy of an array that varies along a manual axis keeps the values of each of its devices, as when it is
        // placed on its own.
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
        let sharding = Sharding::replicated(logical_mesh, 1).with_varying_manual_axes(["x"]).unwrap();
        let varying = Array::from_host_shards(
            &session,
            ArrayType::new_static(DataType::U8, [1]),
            mesh,
            Some(sharding),
            [(devices[0].id(), [7u8].as_slice()), (devices[1].id(), [9u8].as_slice())],
        )
        .unwrap()
        .with_locally_assigned(true);
        let direct = session.place(varying.clone(), PlacementOptions::default().with_may_alias(false)).unwrap();
        let grouped = run(PlaceInput::from(varying), None, Some(false)).unwrap();
        for output in [&direct, &grouped[0]] {
            output.block_until_ready().unwrap();
            let bytes = output.shards().iter().map(|shard| shard.copy_to_host_bytes().unwrap()).collect::<Vec<_>>();
            assert_eq!(bytes, [vec![7], vec![9]]);
        }

        // The upload of a host leaf needs its data, so an invalid encoding is the error of the call.
        let mut host = CoreArray::scalar(true).unwrap();
        host.storage_bytes_mut()[0] = 2;
        assert_eq!(
            run(PlaceInput::from(host), None, None).unwrap_err().to_string(),
            "array element 0 has invalid `bool` byte encoding [2]",
        );
    }
}
