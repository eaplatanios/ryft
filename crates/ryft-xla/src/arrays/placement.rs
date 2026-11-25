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
//! Runtime placement usually returns before its transfers complete. See [`crate::arrays::resharding`] for the
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
use crate::arrays::resharding::devices_span_processes;
use crate::arrays::shards::ArrayShard;
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
///   - `Target` specifies the destination, and `Source` describes the expected input placement. Their leaves implement
///     [`SessionPlacementRequestParameter`]. A consumed [`Sharding`](ryft_core::Sharding) resolves against the
///     session's default mesh. To mix resolved placements and explicit requests in a prefix, use
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
    /// Destination placement tree prefix. When absent, host leaves are uploaded to the default device of the session
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

    /// Returns these options with the destination tree prefix replaced by `target`, which can have another type than
    /// the current one. Each other option keeps its type, so `PlacementOptions::default().with_target(device)` uses
    /// the standard types for the other options without annotations.
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
    /// `PlacementOptions::default().with_may_alias(false)` requests a forced copy without a destination and without
    /// naming the type of a destination. Both `false` and `Some(false)` request a forced copy, and [`None`] leaves
    /// the option unspecified.
    #[inline]
    pub fn with_may_alias<NewMayAlias>(
        self,
        may_alias: NewMayAlias,
    ) -> PlacementOptions<Target, Source, Donate, NewMayAlias> {
        PlacementOptions::new(self.target, self.source, self.donate, Some(may_alias))
    }

    /// Returns the destination placement tree prefix, if one was provided.
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

    /// Returns the destination, source, donation, and aliasing tree prefixes of these options.
    fn into_parts(self) -> (Option<Target>, Option<Source>, Option<Donate>, Option<MayAlias>) {
        (self.target, self.source, self.donate, self.may_alias)
    }
}

impl<Target> PlacementOptions<Target> {
    /// Creates [`Session::place`] options with only the destination tree prefix `target` (e.g., a [`Device`], a
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
/// `may_alias` option of the leaf), and without a destination, that operation has no target and leaves the leaf where
/// it is. Those implementations require the values that the traced values wrap to be ones whose concrete arrays this
/// crate can find (i.e., [`Array`]s, [`Value`]s, host arrays of the reference backend, and the traced values above),
/// so that a traced leaf over arrays of another PJRT client or of another session can be rejected.
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
    ///   - `session`: Destination [`Session`] that supplies the PJRT client and reshard cache. It also identifies
    ///     the session of concrete arrays that a traced leaf may wrap.
    ///   - `target`: Destination placement for this leaf, if one was specified.
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

        // The upload checks its destination with the same function before it uploads anything.
        let default_device = || default_upload_device(session, &session.client().addressable_devices()?);
        let (r#type, placement) = host_upload_destination(&r#type, target, default_device)?;
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

        // A value without a destination lives on the first device only because nothing chose any, so it is not
        // committed to that device, and operations move it to the devices of their committed inputs. The device is
        // the default device of the session (refer to `default_upload_device`).
        let r#type = self.host_type();
        let host = self.into_host_array();
        let target = validate_host_target(&r#type, target)?;

        // The upload takes dense row-major bytes, which is how a host array without a layout is stored, so only an
        // array with a layout is decoded into a copy. Its storage can be mutated safely, so the decoding and the upload
        // validate it rather than trust it.
        let bytes = dense_host_value_bytes(&host)?;

        // An in-place destination names no devices, so it uploads the value to the same device as an upload without
        // a destination, in the storage that it names, and it commits the result like every other destination. A value
        // without a destination, or with one that names only its storage, lives on a device that this process chose on
        // its own (refer to `Array::is_locally_assigned`).
        let committed = target.is_some();
        let locally_assigned = matches!(target, None | Some(PlacementRequest::InPlace { .. }));

        // The upload receives the addressable devices from which the default device is chosen, so that it enumerates
        // the devices of the client once.
        let addressable_devices = client.addressable_devices()?;
        let default_device = || default_upload_device(session, &addressable_devices);
        let (r#type, placement) = host_upload_destination(&r#type, target, default_device)?;
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
        ///   - for a host value, the type rule of [`PlaceOperation`](ryft_core::PlaceOperation) for its destination
        ///     and the admission of its upload (e.g., a device that the client of `session` cannot address, unavailable
        ///     memory on an addressable device, or a data type without the additive identity that an unreduced
        ///     destination needs),
        ///   - for an [`Array`], that it belongs to the PJRT client of `session` or has a destination that names
        ///     devices, the type rule of [`PlaceOperation`](ryft_core::PlaceOperation) for its destination and source
        ///     with its runtime sharding, and the devices of its resolved destination or, for an array of another
        ///     client, the transfer onto them (refer to [`Array::to_placement`]), and
        ///   - for a traced value, that the arrays that it wraps belong to `session`, because the type rule of a
        ///     traced value is applied by the context that records or executes its placement.
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
        self.validate_array_destination(session, target)
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
        // readback and upload on the destination owner. Comparing local values is a separate step.
        if self.is_locally_assigned() || !std::ptr::eq(self.client(), session.client()) {
            return false;
        }

        let Some(target) = target else {
            return false;
        };

        // In-place requests only change storage on the source devices. Explicit destinations may move shards.
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
    /// Checks the placement of this array as [`Placeable::place_parameter`] does before it resolves its destination,
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

        // The destination resolves against the array in the one place that also keeps the locally assigned devices
        // of an array that an in-place destination leaves where it is.
        let output = match target {
            Some(target) => self.place_with_request(session, target, donate, None)?,
            None => self.with_session(session)?.without_previous_sends(),
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

                // Record a forced copy even without a destination, so the policy applies when the operation
                // executes on concrete arrays. Donation only applies to concrete leaves and takes precedence.
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

/// Validates the destination `target` of the upload of a host value whose element type and static shape `r#type`
/// describes, and returns it with a sharding that is padded to the rank of the value. The upload of a host value onto
/// a destination is the placement of a value that has no tracked sharding and lives in device memory without a declared
/// layout, which is how a host value is uploaded without a destination, so the type rule of [`PlaceOperation`]
/// validates the destination against that type before anything is uploaded, except that the destination may
/// split a dimension unevenly (refer to [`Session::place`]).
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

/// Returns the type, with the memory and the layout of the destination, and the placement of the upload of a host value
/// whose element type and static shape `r#type` describes onto the destination `target`, which [`validate_host_target`]
/// returned. A destination that names no devices (i.e., an in-place one or none) uploads the value onto the device that
/// `default_device` returns, which is only called for such a destination.
fn host_upload_destination(
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

/// Returns the device onto which a host value is uploaded when its destination names no devices, which is the default
/// device of `session` (refer to [`Session::with_default_device`]), or otherwise the first of the `addressable_devices`
/// of its client, whether or not `session` has a default mesh. That includes the anonymous mesh of one device that
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
/// // The destination applies to both leaves, and so would every other option.
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
    /// Checks the type rule of [`PlaceOperation`] and the destination of the placement of this leaf, for the type
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
                let (r#type, placement) = host_upload_destination(&value.host_type(), None, default_device)?;
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

        // Check the type before importing any mixed leaves.The compilation session checks destinations of traced
        // leaves, while this session checks the destinations of the host values and concrete arrays it imports.
        let output = operation.infer_output_type_requiring_even_splits(r#type, None)?;
        if let (Some(input_mesh), Some(request)) = (mesh.as_deref(), operation.target()) {
            // Check the inventory for destinations that name devices. An in-place request keeps the capture's
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
    /// broadcasting semantics over `x`. The destination and source prefixes accept [`Placement`], [`PlacementRequest`],
    /// bare [`Device`], bare [`Memory`](ryft_core::Memory), and bare [`Sharding`](ryft_core::Sharding) leaves, so a
    /// destination can be one device (refer to [`PlacementRequest::Device`]) or a memory that leaves every leaf on its
    /// devices (refer to [`PlacementRequest::InPlace`]). Either can be optional, so a tree can mix explicit placements
    /// with `None` to omit an option on a branch. A tree has one leaf type, so a tree that mixes host values and
    /// [`Array`]s holds them as [`PlaceInput`]s, and so does a tree that mixes host values and traced values inside a
    /// compiled or staged function, whose host leaves become constants of the trace of its traced leaves and are then
    /// placed like those (refer to [`PlaceInput`]). A prefix leaf may also cover only subtrees without leaves, in which
    /// case it places nothing (e.g., one of two destinations for an empty [`Vec`] and a scalar, or a single destination
    /// for an empty [`Vec`]). A bare sharding that reaches an input leaf requires a compatible default mesh, including
    /// a source for an eager host upload. The upload ignores the resolved source. A host leaf lifted into a trace uses
    /// its source like other traced leaves. Each option tree is validated against the nodes of `x` (refer to
    /// [`Parameterized::paths`] and [`Parameterized::broadcast_to_parameter_structure`]), so every node of an option
    /// tree must be a node of `x`, and every node of `x` must be a node of the option tree or lie at or below one of
    /// its leaves (e.g., the destinations `[d]` for an input `[]`, `[[d]]` for `[[x], []]`, `[[d], []]` for `[[x]]`,
    /// and `[[]]` for `[]` are all rejected). An option tree of another type with the same paths (e.g., a [`Vec`] for
    /// an array, or another derived struct with the same parameter fields) matches, and so does a node whose static
    /// metadata differs from that of `x`. Map keys are matched by their [`Debug`](std::fmt::Debug) text. In maps
    /// reached by the prefix, callers must ensure that corresponding keys in the input and option tree render
    /// identically and that different keys render different. An input without leaves is returned as it is
    /// once its options are validated.
    ///
    /// # Destinations and Traced Values
    ///
    /// With no destination, a host leaf goes to the default device of this handle (refer to
    /// [`Self::with_default_device`]), as `jax.device_put` uploads it under `jax.default_device`, or otherwise to the
    /// first addressable device, whether or not this session has a [default mesh](Self::device_mesh), and an existing
    /// [`Array`] leaf keeps its current placement. A destination that names only a memory or a layout uploads a host
    /// leaf to the same device. Use [`Self::placement`] to put a sharding on the session's default mesh. Every
    /// [`Array`] placed with a destination is [committed](Array::is_committed) to its devices. An [`Array`] leaf
    /// without a destination keeps its commitment and a host leaf without one is uncommitted, so operations with
    /// committed inputs can move it to their devices. Every resulting [`Array`] belongs to this session, including
    /// one that keeps its placement. An [`Array`] leaf of another PJRT client is rejected unless its destination names
    /// devices, onto which it is read back and uploaded (refer to [`Array::to_placement`]). A traced leaf executes its
    /// placement with the session of the arrays that it wraps (e.g., the input of a batched or differentiated function,
    /// including every primal, tangent, and packed batch). That session cannot be replaced. A traced leaf whose arrays
    /// belong to another PJRT client is therefore rejected, with or without a destination, with the error of such an
    /// [`Array`] leaf without a destination, and one whose arrays belong to another session of the same client is
    /// rejected with a [`ryft_pjrt::Error::InvalidArgument`] error. Other handles of this session (e.g., one with
    /// another default mesh) are accepted, and the results of a traced leaf stay attached to the handles of the
    /// arrays that it wraps.
    ///
    /// A [`PlacementRequest::Device`] destination leaves every leaf wholly on that device. A single-device [`Array`]
    /// keeps its logical mesh and sharding after completing its reduction state. A multi-device [`Array`] gathers
    /// onto the device unless it is unreduced over an axis larger than one, since a placement cannot combine the
    /// contributions of several devices. A bare device also selects the device's default memory. A mesh or sharding
    /// destination without a memory keeps the leaf's memory (refer to [`PlacementRequest::target_memory`]).
    /// A destination without a layout keeps the leaf's layout only if it still describes each result shard.
    ///
    /// A [`PlacementRequest::InPlace`] destination changes only memory or layout, even when the devices and sharding
    /// are unknown during tracing (e.g., how a compiled input is split along auto mesh axes). Its results are
    /// committed, as JAX commits the result of `device_put` onto a memory space eagerly and inside `jit`.
    ///
    /// Inside a batched function, the placement of a batch item (refer to [`Placed`](ryft_core::Placed)) records how
    /// its batch is distributed. A destination or source that is such a [`Placement`] keeps that record (refer to
    /// [`Placement::batch_shardings`]), so that placing a batch item on its own placement leaves every batch item
    /// on the devices that hold it.
    ///
    /// Traced leaves stay traced and record requests through [`PlaceDispatch`]. Every leaf follows the type rule of
    /// [`PlaceOperation`] after shorter shardings are padded with replicated dimensions. Eager [`Array`] and host
    /// leaves allow uneven splits and validate requests before moving or uploading data (refer to
    /// [`PlaceOperation::infer_output_type_allowing_uneven_splits`]). A host leaf is validated as a
    /// value without tracked sharding in device memory, so a reduced or unreduced destination is rejected.
    ///
    /// The type rule prohibits moves inside or into manual regions and requires destinations to preserve reduction
    /// state, subject to the [`PlacementRequest::Device`] rules above. This includes any pending or completed reduction
    /// over an auto axis of a runtime leaf. [`Array::to_placement`] can combine contributions and change reduction
    /// state when its strategies support the destination.
    ///
    /// A traced leaf without a destination stages a placement without a target, which leaves the value where it is
    /// (i.e., its devices, sharding, memory, and layout are unchanged), as JAX's `device_put` does without a device.
    /// That placement records the provided source. Without one, it records where the value concretely lives wherever
    /// it executes on a value that lives somewhere (e.g., when a differentiated function is evaluated on concrete
    /// arrays, refer to [`Placed`](ryft_core::Placed)). Reverse mode differentiation moves the cotangent back to
    /// that source. Such a placement is staged for every traced leaf, including one without a concrete placement and
    /// without a source, so that its copy policy applies wherever the staged program executes on concrete arrays.
    ///
    /// A placement inside a compiled function must target the device mesh that the function is compiled for (refer
    /// to [`LoweringError::PlacementOutsideProgramMesh`](crate::LoweringError::PlacementOutsideProgramMesh)), so a
    /// compiled function cannot move a value to other devices. The devices of a traced leaf that wraps no concrete
    /// array are not known while it is traced, so a bare device destination stages a [`PlacementRequest::Device`] with
    /// its memory and layout, whose type rule selects the default memory of the device when it names none, as it does
    /// for a traced leaf that lives on a mesh of one device. Its type keeps the sharding of the leaf without the
    /// reduction state that one device completes, so it combines with the other values of the function. A compiled
    /// function accepts it only when the device is the only device of the mesh that the function is compiled for (refer
    /// to [`LoweringError::DeviceOutsideProgramMesh`](crate::LoweringError::DeviceOutsideProgramMesh)), and the outputs
    /// of such a function are committed, as an explicit device commits an [`Array`] eagerly. Wherever the staged
    /// placement executes on a concrete array instead (e.g., when a traced program is evaluated on arrays), the device
    /// resolves against that array as it does for an [`Array`] leaf. A bare device source of a traced leaf whose
    /// devices are not known stages a [`PlacementRequest::Device`] source, so that reverse mode differentiation moves
    /// the cotangent back to that device, and transposing the result again moves it to the destination once more.
    /// Inside a compiled function, a destination must lie on the mesh that the function is compiled for, whereas a
    /// source only states where the input lives, which the type rule checks against the type of the input, and it is
    /// not checked against that mesh, because the function moves nothing to it. The reverse mode of such a function
    /// moves the cotangent to the source, so compiling it rejects a source outside that mesh as a destination. Inside
    /// a manual region (e.g., a `shard_map` body), where no value can be moved, a destination may name only a memory
    /// or a layout, and is otherwise rejected with
    /// [`LoweringError::UnsupportedOperation`](crate::LoweringError::UnsupportedOperation), in compiled functions and
    /// in eager `shard_map` bodies alike. A source may name any devices there, because the forward placement moves
    /// nothing to them. The reverse mode of such a function moves the cotangent to that source, unless transposition
    /// leaves the cotangent where it lives (i.e., the placement has no destination or an in-place one, and the leaf
    /// varies along the manual axes of the region or carries reduction state over them, refer to
    /// [`PlaceOperation::transposes_in_place`]), so compiling that program rejects such a source as a destination.
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
    /// already lives at its destination). A leaf cannot be donated while its `may_alias` option is `Some(true)`,
    /// which this function reports with [`ArrayError::DonatedArrayMayAlias`] before it places any leaf.
    ///
    /// # Sources
    ///
    /// The source option states where a leaf is expected to live. Its devices are informative on every path, so they
    /// are never compared with the devices that a leaf actually lives on, and reverse mode differentiation sends the
    /// cotangent to the devices that it names. Eager placement checks destination devices against its session's
    /// inventory. The compiling session checks an abstract tracer's destination. The type rule of [`PlaceOperation`]
    /// checks the declared source sharding, memory, and layout against the concrete [`Array`] leaf's type and runtime
    /// sharding, including the reduction state and shards along auto mesh axes that its type omits. It performs the
    /// same check on a traced leaf that wraps a concrete array (refer to
    /// [`PlaceOperation::infer_output_type_allowing_uneven_splits`]). A traced leaf wrapping a concrete array then
    /// fills in a source that names only memory or layout with the array's placement. For a single-device array with
    /// a source naming a device, it fills in the source with that placement moved to the named device (refer to
    /// [`PlacementRequest::resolve_source`]). Transposition uses this placement to return the cotangent to its devices
    /// and account for its runtime reduction state. An expected layout must match the shape of each shard determined by
    /// the leaf's runtime sharding or its staged type. It must also equal any layout declared by the leaf's type. Only
    /// the type rule checks the source, so it does not inspect physical buffer layouts when the leaf's type declares
    /// no layout.
    ///
    /// # Multi-Process Placement
    ///
    /// In a run with more than one process, every process must make the same call with the same options and tree
    /// structure. For each array leaf, they must agree on whether it is locally assigned and whether it belongs to
    /// their own session's client. Placements communicate across processes in leaf order. With comparison disabled,
    /// each participant must finish its local placements before a subsequent cross-process transfer can complete.
    /// Each process uploads the whole host value to its addressable destination devices. When the destination spans
    /// more than one process, every process must pass the same logical value for that leaf, or the processes construct
    /// conflicting replicas of one array. Differing values need not produce a conflicting replica, because each process
    /// contributes only the part that its devices hold. For example, inputs `[1, 1]` on one process and `[2, 2]` on
    /// another, placed fully partitioned over one device of each, would form the array `[1, 2]`. The same applies to
    /// any locally assigned [`Array`] leaf, which each process places from its own value. An array is locally assigned
    /// when it is uploaded or created without a destination that names devices. A destination that names only a memory
    /// can commit an array without ending its local assignment, which lasts until a placement names devices.
    ///
    /// With [`SessionBuilder::value_comparison`](crate::SessionBuilder::value_comparison) enabled, a mesh or sharding
    /// destination spanning every process in a multi-process run compares the host or locally assigned values for that
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
    /// checks host encodings and reports known producer failures for locally assigned arrays with an explicit
    /// destination or a forced copy. When a leaf fails its checks on one process only (e.g., a host scalar placed onto
    /// a split destination on one process and a host vector on another), that process returns its own error and every
    /// other process returns a [`ryft_pjrt::Error::FailedPrecondition`] before any leaf is placed. A failure after
    /// admission can still stop one process before its peers reach the same transfer. This includes backend failures
    /// when submitting work and failures reported asynchronously by destination buffers.
    ///
    /// Unlike the `np.equal` of JAX, the comparison is bitwise (e.g., `+0` and `-0` differ, and NaN values with equal
    /// bits agree). What the check cannot turn into an error remains a precondition: every process must make the same
    /// sequence of comparing calls (of this function, of [`Array::to_placement`] and [`Array::into_placement`], and of
    /// eager placements) with trees of the same structure and the same destinations, because a process that skips a
    /// comparison or stops calling leaves the others waiting in it. Equal values remain an unchecked precondition for
    /// a destination on the devices of some of the processes only, for a device destination, and for every placement
    /// of a session that disables the check. [`Placeable::place_parameter`] compares nothing on its own. A host leaf
    /// without a destination, or with one that names only its memory or layout, is uploaded independently onto a device
    /// of each process, so those leaves may hold different process-local values (e.g., different batches). Data that
    /// differs across processes and forms one distributed array (e.g., the local part of a value that is split across
    /// processes) is placed with [`Array::from_host_shards`] instead, which takes from each process the bytes of the
    /// shards of its own devices.
    ///
    /// # Example
    ///
    /// The bare device below is the destination of the subtree `(b, c)` alone, so `a` keeps its placement and its
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
    ///   - `options`: Destination, expected source, donation, and aliasing options for the leaves of `x`.
    ///
    /// # Errors
    ///
    /// Before it places any leaf, this function checks that every option prefix broadcasts over `x`, and then, for each
    /// leaf in order, that it is not donated while it may alias and that its in-place source and destination name a
    /// memory or a layout. An empty [`PlacementRequest::InPlace`] produces the [`TypeError`](ryft_core::TypeError)
    /// from [`PlaceOperation`]. The requested storage must also be supported. Mixed leaves from different traces or
    /// an unavailable group comparison can fail the whole call. Before placement, this function checks each leaf's
    /// metadata. For host values and concrete arrays, this includes the source and destination type rules, the
    /// array's PJRT client and the session's inventory of destination devices. Forward source devices are advisory. The
    /// compiling session checks the destinations of abstract tracers. A preflight error leaves every leaf untouched,
    /// donated ones included. Placing a leaf can still fail after preceding leaves have been placed. Causes include
    /// failure of submitted work (e.g., a PJRT transfer, compiled reshard, or allocation), unsupported memory or layout,
    /// a move that no strategy of [`crate::arrays::resharding`] supports (e.g., one that would have to combine the
    /// contributions of a pending reduction), the bytes of a host array that do not encode valid elements, and the
    /// errors of the type rule for a traced leaf, which the context that records or executes its placement applies.
    /// Placement drops the results of the preceding leaves and leaves their submitted transfers and donations to
    /// complete, so only such a failure can have consumed the donated inputs of the leaves that precede the failing
    /// one. Every error of a leaf other than the root of `x` is wrapped in [`ArrayError::FailedParameterPlacement`]
    /// with the path of that leaf, except an [`ArrayError::DonatedArrayMayAlias`] or
    /// [`ArrayError::FailedParameterPlacement`] that the leaf returns (e.g., from a placement of a tree of its own),
    /// whose path is prefixed with the path of that leaf instead.
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
enum PlacementOptionValues<Value> {
    /// Value of every leaf, or `None` if no prefix was given.
    Uniform(Option<Value>),

    /// Values of the leaves, in the order of the leaves.
    PerParameter(Vec<Value>),
}

impl<Value> PlacementOptionValues<Value> {
    /// Expands an optional placement prefix into its value for each input leaf. A prefix with one leaf at the root
    /// covers the whole input, so it needs no path matching or structure cloning, and neither it nor an omitted
    /// prefix allocates per leaf. Structured prefixes use [`Parameterized::broadcast_parameters`] to validate
    /// every node, including empty containers.
    fn from_prefix<P: Parameterized<Value>, S: Parameterized<Placeholder>>(
        prefix: Option<P>,
        structure: &S,
    ) -> Result<Self, Error>
    where
        Value: Clone + Parameter,
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
    fn get(&self, index: usize) -> Option<&Value> {
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

    /// Destination placement of `parameter`, if one was specified.
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
        array.validate_array_destination(context, target)
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

                // A forced copy requires ready buffers even when the caller supplies no destination.
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

// TODO(eaplatanios): Review from here onwards.

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::{BTreeMap, HashSet};
    use std::sync::Arc;
    use std::time::Duration;

    use half::bf16;
    #[cfg(feature = "cuda-13")]
    use indoc::formatdoc;
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Add, ArrayIrType, ArrayIrValue, BatchAxis, Complex, DataType, DeviceMesh, DifferentiableType, Differentiate,
        DifferentiationDual, DifferentiationError, Dimension, DimensionBounds, DimensionVariable, DomainTracingContext,
        EmptyRegionDriver, Error as CoreError, Exp, InterpretableOperation, Layout, LogicalMesh, Memory, MeshAxis,
        MeshAxisType, Neg, Operation as CoreOperation, PLACE_OPERATION_NAME, ParameterError, Place, PlaceOperation,
        Placed, ProgramError, Shape, ShardMapOptions, Sharding, ShardingDimension, ShardingError, Sin, StridedLayout,
        Tile, TileDimension, TiledLayout, TypeError, Typed, ValueProjection, batch, differentiate_at, f8e4m3fn,
        shard_map,
    };
    use ryft_pjrt::{
        BufferType, Client, ClientOptions, CpuClientOptions, ExecutionFence, KeyValueStore, load_cpu_plugin,
    };

    #[cfg(feature = "cuda-13")]
    use crate::arrays::host::DIFFERENT_VALUES_MESSAGE;
    use crate::arrays::host::participant_mesh_of;
    #[cfg(feature = "cuda-13")]
    use crate::distributed::tests::{
        read_addressable_f32s, run_as_cuda_rank, run_cuda_rank_pair, run_cuda_rank_pair_probes,
    };
    use crate::distributed::tests::{run_cpu_processes, test_process};
    use crate::sessions::tests::{DeclareRemoteDevices, domain_mesh, two_device_mesh};
    use crate::tests::{
        cpu_client, device_mesh_2x2, execution_client, other_cpu_client, session_with_remote_devices,
        values_from_bytes, values_to_bytes,
    };
    use crate::{
        ArrayTracer, CompileTracer, CompiledFunction, CompiledFunctionDispatcher, DistributedRuntime, LoweringError,
        Options, SessionError, StagedFunction, StatefulCompileTracer, ToPjrt, TracedProgram, Type, Value, compile,
        compile_statefully, compile_with_options, jit, stage, trace,
    };

    use super::*;

    /// Uploads an uncommitted two-element vector for a placement derivative test.
    fn upload_place_jvp_vector(session: &Session, values: &[f32; 2]) -> Array {
        Array::from_host_buffer(session, ArrayType::new_static(DataType::F32, [2]), None, None, values_to_bytes(values))
            .unwrap()
    }

    /// Uploads a named singleton placement primal and a tangent on the session's default device.
    fn place_jvp_inputs(session: &Session, device: Device) -> (Array, Array) {
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("other", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![device]).unwrap();
        let placement = Placement::new(mesh, Sharding::replicated(logical_mesh, 1)).unwrap();
        let primal = upload_place_jvp_vector(session, &[1., 2.]).place(&placement, None).unwrap();
        (primal, upload_place_jvp_vector(session, &[3., 4.]))
    }

    /// Reads the value of a single-device derivative result alongside its placement.
    fn observe_place_jvp(array: &Array) -> (Device, bool, Vec<f32>) {
        let [device] = array.mesh().devices() else { panic!("expected one device") };
        (*device, array.is_committed(), array.copy_to_host().unwrap().elements::<f32>().unwrap())
    }

    // Returns the path and the underlying error of an `ArrayError::FailedParameterPlacement`, and `None` for any other
    // error.
    fn failed_parameter(error: &Error) -> Option<(String, &Error)> {
        match error {
            Error::Array(error) => match error.as_ref() {
                ArrayError::FailedParameterPlacement { path, source } => Some((path.to_string(), source.as_ref())),
                _ => None,
            },
            _ => None,
        }
    }

    // Saves the original placement diagnostic for tests whose transform can defer or wrap it.
    fn record_placement_error<T>(error: &RefCell<Option<String>>, result: Result<T, Error>) -> Result<T, ProgramError> {
        result.map_err(|placement_error| {
            let message = placement_error.to_string();
            error.replace(Some(message.clone()));
            ProgramError::InvalidArgument { message }
        })
    }

    // Returns the `f32` values that the devices of `array` hold, in the device order of its mesh.
    fn shard_values(array: &Array) -> Vec<Vec<f32>> {
        array
            .shards()
            .iter()
            .map(|shard| values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()))
            .collect()
    }

    // Returns the native addresses of the buffers that the devices of `array` hold, in the device order of its mesh,
    // once they are ready. Two arrays that are both alive share storage exactly when they share such an address, which
    // the identity of the handles of their buffers does not show.
    fn native_buffer_addresses(array: &Array) -> Vec<usize> {
        array
            .shards()
            .iter()
            .map(|shard| {
                let buffer = shard.buffer().unwrap();
                buffer.ready().unwrap().r#await().unwrap();
                // The buffer is ready and stays alive while `array` does, and its address is only compared and never
                // dereferenced.
                unsafe { buffer.unsafe_pointer().unwrap() }
            })
            .collect()
    }

    // Returns the physical dimensions of the buffer that each device of `array` holds, in the device order of its mesh.
    fn shard_buffer_dimensions(array: &Array) -> Vec<Vec<u64>> {
        array.shards().iter().map(|shard| shard.buffer().unwrap().dimensions().unwrap().to_vec()).collect()
    }

    // Leaf that places a pair of copies of `array` of its own through `Session::place`, which fails at the second
    // element of that pair, because that element is donated while it may alias when `donate` is `true`, and because it
    // has an in-place target that names nothing otherwise.
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

    /// Places on every rank of the job of `client` a host value with [`Session::place`] and an array that each rank
    /// uploaded without devices with [`Array::to_placement`] onto a split over the two global devices, with the same
    /// values on both ranks or, when `differ` is `true`, with values that differ between them, and then places an
    /// array of rank 0 replicated onto both devices. Prints one `probe:` line for each placement with the elements of
    /// the shards of this rank once its result is ready, `different values` when its readiness reports the failed
    /// comparison of the values of the ranks, or the first line of any other error.
    #[cfg(feature = "cuda-13")]
    fn probe_value_comparison(client: &'static Client<'static>, differ: bool) {
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
        let replicated = |mesh: &DeviceMesh| Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let report = |name: &str, placed: Result<Array, Error>| {
            let outcome = match placed.and_then(|placed| placed.block_until_ready().map(|()| placed)) {
                Ok(placed) => format!("{:?}", read_addressable_f32s(&placed)),
                Err(Error::Pjrt(ryft_pjrt::Error::FailedPrecondition { message, .. }))
                    if message == DIFFERENT_VALUES_MESSAGE =>
                {
                    "different values".to_string()
                }
                Err(error) => error.to_string().lines().next().unwrap_or("").to_string(),
            };
            println!("probe: rank {rank} {name}: {outcome}");
        };

        let offset = if differ { 10. * rank as f32 } else { 0. };
        let values = values_to_bytes(&[1f32, 2., 3., 4.].map(|value| value + offset));
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let host = CoreArray::new(r#type.clone(), values.clone()).unwrap();
        report("host value", session.place(host, PlacementOptions::to(PlacementRequest::from(split.clone()))));
        let uploaded = Array::from_host_buffer(&session, r#type.clone(), None, None, values).unwrap();
        report("uploaded array", uploaded.to_placement(&session, split));

        // An ordinary cross-host placement afterwards completes only if both ranks are still in step.
        let r#type = r#type.with_sharding(replicated(&process_0_mesh)).unwrap();
        let values = values_to_bytes(&[17f32, 18., 19., 20.]);
        let source = Array::from_host_buffer(&session, r#type, process_0_mesh.clone(), None, values).unwrap();
        let target = Placement::new(forward_mesh.clone(), replicated(&forward_mesh)).unwrap();
        report("cross-host placement", source.to_placement(&session, target));
    }

    /// Places on every rank of the job of `client` a tree of two `bool[2]` host values with [`Session::place`] onto a
    /// split over the two global devices, whose first leaf on rank 0 holds a byte that does not encode a `bool`, and
    /// then places an array of rank 0 replicated onto both devices. Prints one `probe:` line for the tree, with
    /// `invalid value` when the call returns the error of that leaf and otherwise `different values` for each leaf
    /// whose readiness reports the failed comparison, and one for the cross-host placement, as
    /// [`probe_value_comparison`] does.
    #[cfg(feature = "cuda-13")]
    fn probe_value_comparison_with_invalid_leaf(client: &'static Client<'static>) {
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
        let outcome = match session.place(tree, PlacementOptions::to(PlacementRequest::from(split))) {
            Ok(placed) => placed
                .iter()
                .map(|placed| match placed.block_until_ready() {
                    Ok(()) => "ready".to_string(),
                    Err(Error::Pjrt(ryft_pjrt::Error::FailedPrecondition { message, .. }))
                        if message == DIFFERENT_VALUES_MESSAGE =>
                    {
                        "different values".to_string()
                    }
                    Err(error) => error.to_string().lines().next().unwrap_or("").to_string(),
                })
                .collect::<Vec<_>>()
                .join(", "),
            Err(Error::Array(error))
                if matches!(error.as_ref(), ArrayError::FailedParameterPlacement { source, .. }
                    if source.to_string() == "array element 0 has invalid `bool` byte encoding [2]") =>
            {
                "invalid value".to_string()
            }
            Err(error) => error.to_string().lines().next().unwrap_or("").to_string(),
        };
        println!("probe: rank {rank} tree: {outcome}");

        // An ordinary cross-host placement afterwards completes only if both ranks are still in step.
        let replicated = |mesh: &DeviceMesh| Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(replicated(&process_0_mesh)).unwrap();
        let values = values_to_bytes(&[17f32, 18., 19., 20.]);
        let source = Array::from_host_buffer(&session, r#type, process_0_mesh.clone(), None, values).unwrap();
        let target = Placement::new(forward_mesh.clone(), replicated(&forward_mesh)).unwrap();
        let placed = source.to_placement(&session, target);
        let placed = placed.and_then(|placed| placed.block_until_ready().map(|()| placed));
        match placed {
            Ok(placed) => println!("probe: rank {rank} cross-host placement: {:?}", read_addressable_f32s(&placed)),
            Err(error) => {
                let error = error.to_string();
                println!("probe: rank {rank} cross-host placement: {}", error.lines().next().unwrap_or(""));
            }
        }
    }

    /// Places on every rank of the job of `client`, with one call of [`Session::place`], a host value split over the
    /// two global devices, which is a scalar on rank 0 (so that only rank 0 rejects it) and a vector on rank 1, and an
    /// array of rank 0 replicated onto both devices, which rank 1 can only receive from rank 0. Then places that array
    /// once more on its own. Prints one `probe:` line for the call, with `different values` when it returns the error
    /// of the failed comparison or otherwise the first line of its error or the elements of the shards of this rank,
    /// and one for the placement that follows it.
    #[cfg(feature = "cuda-13")]
    fn probe_call_admission_with_invalid_leaf(client: &'static Client<'static>) {
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
        let options = PlacementOptions::to(vec![split, replicated.clone()]);
        let placed = session.place(tree, options).and_then(|placed| {
            placed.iter().try_for_each(Array::block_until_ready)?;
            Ok(placed)
        });
        let outcome = match placed {
            Ok(placed) => format!("{:?}", placed.iter().map(read_addressable_f32s).collect::<Vec<_>>()),
            Err(Error::Pjrt(ryft_pjrt::Error::FailedPrecondition { message, .. }))
                if message == DIFFERENT_VALUES_MESSAGE =>
            {
                "different values".to_string()
            }
            Err(error) => error.to_string().lines().next().unwrap_or("").to_string(),
        };
        println!("probe: rank {rank} tree: {outcome}");

        // An ordinary cross-host placement afterwards completes only if both ranks are still in step.
        let placed = source.to_placement(&session, replicated);
        match placed.and_then(|placed| placed.block_until_ready().map(|()| placed)) {
            Ok(placed) => println!("probe: rank {rank} cross-host placement: {:?}", read_addressable_f32s(&placed)),
            Err(error) => {
                let error = error.to_string();
                println!("probe: rank {rank} cross-host placement: {}", error.lines().next().unwrap_or(""));
            }
        }
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

    #[test]
    fn test_session_placement() {
        let session = Session::builder().client(cpu_client(1)).build().unwrap();
        let mesh = device_mesh_2x2();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert!(matches!(session.placement(sharding.clone()), Err(Error::Session(SessionError::MissingMesh))));
        let configured = session.with_device_mesh(mesh.clone());
        assert_eq!(configured.placement(sharding.clone()).unwrap(), Placement::new(mesh.clone(), sharding).unwrap());

        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("z", 4, MeshAxisType::Auto).unwrap()]).unwrap();
        let error = configured.placement(Sharding::replicated(other_mesh.clone(), 1)).unwrap_err();
        assert!(matches!(
            &error,
            Error::Core(error) if matches!(
                error.as_ref(),
                CoreError::Sharding(ShardingError::MeshMismatch { expected, actual })
                    if expected == mesh.logical_mesh() && actual == &other_mesh,
            ),
        ));
        assert_eq!(
            error.to_string(),
            format!("mesh mismatch; expected `{:?}` but got `{other_mesh:?}`", mesh.logical_mesh()),
        );
    }

    #[test]
    fn test_array_resolve_placement_request() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
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
        let destination = |device| PlacementRequest::Device { device, memory: Some(memory), layout: None };
        let replicated = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap();
        assert_eq!(
            array.resolve_placement_request(destination(devices[0])).unwrap(),
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
            unreduced_array.resolve_placement_request(destination(devices[0])).unwrap(),
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
            sharded_array.resolve_placement_request(destination(devices[1])).unwrap(),
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
            split_array.resolve_placement_request(destination(devices[1])).unwrap(),
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
    fn test_placement_options_new() {
        let target = PlacementRequest::from(Device::new(3, 0));
        let options: PlacementOptions = PlacementOptions::new(Some(target.clone()), None, Some(true), Some(false));
        assert_eq!(options.target(), Some(&target));
        assert_eq!(options.source(), None);
        assert_eq!(options.donate(), Some(&true));
        assert_eq!(options.may_alias(), Some(&false));
        assert_eq!(options.into_parts(), (Some(target), None, Some(true), Some(false)));
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

        // Unset options with another prefix type come from the constructor, as the default has the standard types.
        let options = PlacementOptions::<Device>::new(None, None, None, None);
        assert_eq!(options.into_parts(), (None, None, None, None));
    }

    #[test]
    fn test_placement_options_with_target() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();

        // The destination changes its type, and the other options keep theirs.
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
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
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

        // A forced copy needs no destination, and the copy shares no buffer with the input.
        let options = PlacementOptions::default().with_may_alias(Some(false));
        assert_eq!(options.clone().into_parts(), (None, None, None, Some(Some(false))));
        let copied = session.place(array.clone(), options).unwrap();
        assert!(!shares_buffer(&copied));
        assert_eq!(copied.mesh(), array.mesh());
        assert_eq!(copied.r#type(), array.r#type());
        assert_eq!(shard_values(&copied), vec![vec![2.5]]);
        let kept = session.place(array.clone(), PlacementOptions::default().with_may_alias(Some(true))).unwrap();
        assert!(shares_buffer(&kept));

        // Plain `bool` aliasing options are accepted as well, and they mean what their `Some` forms mean.
        let options = PlacementOptions::default().with_may_alias(false);
        assert_eq!(options.clone().into_parts(), (None, None, None, Some(false)));
        let copied = session.place(array.clone(), options).unwrap();
        assert!(!shares_buffer(&copied));
        assert_eq!(copied.mesh(), array.mesh());
        assert_eq!(shard_values(&copied), vec![vec![2.5]]);
        let kept = session.place(array.clone(), PlacementOptions::default().with_may_alias(true)).unwrap();
        assert!(shares_buffer(&kept));

        // So are tuple prefixes of plain `bool` options.
        let options = PlacementOptions::default().with_may_alias((false, true));
        let (copied, kept) = session.place((array.clone(), array.clone()), options).unwrap();
        assert!(!shares_buffer(&copied));
        assert_eq!(shard_values(&copied), vec![vec![2.5]]);
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
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(PlacementOptions::to(devices[1]).into_parts(), (Some(devices[1]), None, None, None));

        // The destination alone fixes the types of the other options, whether it is a raw device, a placement request,
        // or a tuple prefix.
        let scalar = session.place(2.5f32, PlacementOptions::to(devices[1])).unwrap();
        assert_eq!(scalar.mesh().devices(), [devices[1]]);
        assert!(scalar.is_committed());
        let moved = session.place(scalar, PlacementOptions::to(PlacementRequest::from(devices[0]))).unwrap();
        assert_eq!(moved.mesh().devices(), [devices[0]]);
        assert_eq!(moved.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![2.5]);
        let (placed, uploaded) = session.place((1f32, 2f32), PlacementOptions::to((Some(devices[1]), None))).unwrap();
        assert_eq!(placed.mesh().devices(), [devices[1]]);
        assert!(placed.is_committed());
        assert_eq!(uploaded.mesh().devices(), [devices[0]]);
        assert!(!uploaded.is_committed());
    }

    #[test]
    fn test_place_input_from() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let array = Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(&[1f32, 2.])).unwrap();
        let complex = Complex::new(1.5f32, -2.);
        let low_precision = f8e4m3fn::from_f32(0.5).unwrap();

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
    fn test_place_input_place_parameter() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let upload = || {
            let r#type = ArrayType::new_static(DataType::F32, [2]);
            Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(&[1f32, 2.])).unwrap()
        };
        assert!(matches!(PlaceInput::from(true), PlaceInput::Host(host) if host == CoreArray::scalar(true).unwrap()));
        assert!(matches!(PlaceInput::from(upload()), PlaceInput::Runtime(_)));

        // One tree mixes host values and arrays, and the destination prefix broadcasts over both kinds of leaves.
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
        let error = session.place((PlaceInput::from(1u8), PlaceInput::from(upload())), options(pinned)).unwrap_err();
        let (path, error) = failed_parameter(&error).unwrap();
        assert_eq!(path, "$.1");
        assert!(matches!(error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
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
        for (unaliased_shard, shard) in unaliased.shards().iter().zip(array.shards()) {
            assert!(!Arc::ptr_eq(unaliased_shard.buffer().unwrap(), shard.buffer().unwrap()));
        }
        for (unaliased_address, address) in
            native_buffer_addresses(&unaliased).into_iter().zip(native_buffer_addresses(&array))
        {
            assert_ne!(unaliased_address, address);
        }
        assert_eq!(unaliased.r#type(), array.r#type());
        assert_eq!(unaliased.mesh(), array.mesh());
        assert_eq!(unaliased.sharding(), array.sharding());
        assert_eq!(shard_values(&unaliased), vec![vec![1., 2.]; 2]);

        // The copies keep the commitment of the array.
        let array = upload().with_committed(false);
        let unaliased = array.clone().into_unaliased().unwrap();
        assert!(!unaliased.is_committed());
        assert!(array.into_unaliased().unwrap().with_committed(true).is_committed());
    }

    #[test]
    fn test_array_relocatable_type_matches_value_array() {
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let (named_primal, movable) = place_jvp_inputs(&session, devices[1]);
        let named_tangent = movable.place(&named_primal.placement().unwrap(), None).unwrap();
        let primal = upload_place_jvp_vector(&session, &[1., 2.]);
        let committed = primal.place(&devices[1], None).unwrap();

        // The saved identity keeps whichever tangent its declaration admits. A movable primal leaves its
        // declaration open, while a committed anonymous primal requires an anonymous or relocatable tangent.
        let check = |primal: Array, tangent: Array, expected| {
            let (_, pushforward) = differentiate_at(primal.clone()).linearize(Ok).unwrap();
            let through_array = pushforward.apply(tangent.clone()).map(|array| observe_place_jvp(&array));

            let (_, pushforward) = session.differentiate_at(Value::Array(primal)).linearize(Ok).unwrap();
            let through_value = pushforward.apply(Value::Array(tangent)).and_then(|output| {
                let array = ValueProjection::<ArrayType>::into_projected(output)?;
                Ok(observe_place_jvp(&array))
            });
            assert_eq!(through_array, expected);
            assert_eq!(through_value, expected);
        };
        check(primal, named_tangent.clone(), Ok((devices[1], true, vec![3., 4.])));
        check(committed.clone(), movable, Ok((devices[0], false, vec![3., 4.])));
        check(
            committed,
            named_tangent,
            Err(ProgramError::MalformedProgram(
                "pushforward tangent 0 has type f32[2][sharding={mesh<['other'=1:explicit]>, [{}]}] \
                 but its primal boundary requires tangent type f32[2]"
                    .into(),
            )),
        );
    }

    #[test]
    fn test_array_relocatable_type_survives_generic_array_ir_wrapping() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let second = Device::from_pjrt(&client.addressable_devices().unwrap()[1]).unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 1);
        let mesh = DeviceMesh::new(mesh, vec![second]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let upload = |mesh, sharding| {
            Array::from_host_buffer(&session, r#type.clone(), mesh, sharding, values_to_bytes(&[1f32, 2.])).unwrap()
        };
        let named = upload(Some(mesh), Some(sharding));
        let anonymous = upload(None, None);
        for movable in [anonymous.clone(), named.clone().with_committed(false)] {
            let declaration = CoreValue::relocatable_type(&movable).unwrap();
            let movable = ArrayIrValue::Array(movable);
            assert_eq!(CoreValue::relocatable_type(&movable), Some(ArrayIrType::Array(declaration)));
            let primal = ArrayIrValue::Array(named.clone());
            let dual = DifferentiationDual::new(primal, movable.clone()).unwrap();
            assert_eq!(dual.tangent(), &ryft_core::MaybeZero::Value(movable));
        }

        let committed = ArrayIrValue::Array(anonymous.place(&second, None).unwrap());
        let named = ArrayIrValue::Array(named);
        for (primal, tangent, message) in [
            (
                named.clone(),
                committed.clone(),
                "tangent type f32[2] does not match type f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] \
                 required by primal type f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}]",
            ),
            (
                committed,
                named,
                "tangent type f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] does not match type f32[2] \
                 required by primal type f32[2]",
            ),
        ] {
            assert!(matches!(
                DifferentiationDual::new(primal, tangent),
                Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message: actual })))
                    if actual == message,
            ));
        }
    }

    #[test]
    fn test_place_operation_vjp_accepts_relocatable_seeds() {
        // The VJP of placing an uncommitted upload on the first device onto a named mesh of the second device accepts
        // an uncommitted seed on the first device, which may be placed as the cotangent type states, as well as a seed
        // on that mesh, through an `Array` and through its `Value::Array` representation, and returns the cotangent
        // to the first device, as JAX does. A seed committed to the first device has the same anonymous abstract
        // sharding. The pullback accepts it and replays the transposed placement on its actual devices (refer to
        // `test_array_place_pullback_uses_the_seed_on_its_own_devices`).
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("q", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let target = Placement::new(mesh.clone(), Sharding::replicated(logical_mesh.clone(), 1)).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let upload = |values: &[f32]| {
            Array::from_host_buffer(&session, r#type.clone(), None, None, values_to_bytes(values)).unwrap()
        };
        let seeds = [
            upload(&[5., 6.]),
            Array::from_host_buffer(
                &session,
                r#type.clone(),
                mesh,
                Some(Sharding::replicated(logical_mesh, 1)),
                values_to_bytes(&[5f32, 6.]),
            )
            .unwrap(),
            upload(&[5., 6.]).to_placement(&session, devices[0]).unwrap(),
        ];
        let observe = |array: Array| {
            let devices = array.mesh().devices().iter().map(|device| device.id()).collect::<Vec<_>>();
            (devices, array.copy_to_host().unwrap().elements::<f32>().unwrap(), array.is_committed())
        };
        let mut outcomes = Vec::new();
        for seed in seeds {
            let through_array = differentiate_at(upload(&[1., 2.]))
                .vjp(|x| x.place(&target, None))
                .map_err(|error| error.to_string())
                .and_then(|(_, pullback)| pullback.apply_materialized(seed.clone()).map_err(|error| error.to_string()))
                .map(observe);
            let through_value = session
                .differentiate_at(Value::Array(upload(&[1., 2.])))
                .vjp(|x| x.place(&target, None))
                .map_err(|error| error.to_string())
                .and_then(|(_, pullback)| {
                    let cotangent =
                        pullback.apply_materialized(Value::Array(seed)).map_err(|error| error.to_string())?;
                    ValueProjection::<ArrayType>::into_projected(cotangent).map_err(|error| error.to_string())
                })
                .map(observe);
            outcomes.push(vec![through_array, through_value]);
        }
        // The transposed placement has a target (where the primal lived), so its result is committed to it.
        let accepted = Ok((vec![devices[0].id()], vec![5., 6.], true));
        assert_eq!(outcomes, vec![vec![accepted.clone(), accepted.clone()]; 3]);

        // `Session::place` admits a source on that mesh for the uncommitted upload by the same rule and leaves it where
        // it lives, while it still rejects that source for the committed array.
        let source = PlacementRequest::from(target);
        let options = || PlacementOptions::default().with_source(source.clone());
        assert_eq!(
            session.place(upload(&[1., 2.]), options()).map(observe).ok(),
            Some((vec![devices[0].id()], vec![1., 2.], false)),
        );
        let committed = upload(&[1., 2.]).to_placement(&session, devices[0]).unwrap();
        assert_eq!(
            session.place(committed, options()).unwrap_err().to_string(),
            "`place` source sharding `{mesh<['q'=1:explicit]>, [{}]}` does not match the input sharding \
             `{mesh<['device'=1:auto]>, [{}]}`",
        );
    }

    #[test]
    fn test_place_operation_keeps_reduction_state_checks_for_relocatable_sources() {
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
    fn test_place_operation_admits_relocatable_sources_on_every_path() {
        // A targetless placement of an uncommitted upload on the first device with a source on a named mesh of the
        // second device is admitted, because the upload may be placed there first, and leaves the upload where it is,
        // uncommitted. Eager placement, linearization, the fused JVP, batching, and the direct interpretation of the
        // operation all admit it by the same rule, as pinned JAX admits `device_put` with such a source eagerly, under
        // `jvp`, and under `vmap`.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("other", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let source = Placement::new(mesh, Sharding::replicated(logical_mesh, 1)).unwrap();
        let upload = |shape: &[usize], values: &[f32]| {
            let r#type = ArrayType::new_static(DataType::F32, shape.to_vec());
            Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(values)).unwrap()
        };
        let observe = |array: Array| {
            let devices = array.mesh().devices().iter().map(|device| device.id()).collect::<Vec<_>>();
            (devices, array.copy_to_host().unwrap().elements::<f32>().unwrap(), array.is_committed())
        };
        let first = devices[0].id();

        let options = PlacementOptions::default().with_source(source.clone());
        assert_eq!(
            session.place(upload(&[2], &[3., 4.]), options).map(observe).map_err(|error| error.to_string()),
            Ok((vec![first], vec![3., 4.], false)),
        );
        assert_eq!(
            upload(&[2], &[3., 4.])
                .place(&Option::<Placement>::None, &source)
                .map(observe)
                .map_err(|error| error.to_string()),
            Ok((vec![first], vec![3., 4.], false)),
        );
        let (primal, pushforward) = differentiate_at(upload(&[2], &[3., 4.]))
            .linearize(|x| x.place(&Option::<Placement>::None, &source))
            .unwrap();
        assert_eq!(observe(primal), (vec![first], vec![3., 4.], false));
        assert_eq!(observe(pushforward.apply(upload(&[2], &[5., 6.])).unwrap()), (vec![first], vec![5., 6.], false));
        let (primal, tangent) = differentiate_at(upload(&[2], &[3., 4.]))
            .jvp(upload(&[2], &[5., 6.]), |x| x.place(&Option::<Placement>::None, &source))
            .unwrap();
        assert_eq!(
            (observe(primal), observe(tangent)),
            ((vec![first], vec![3., 4.], false), (vec![first], vec![5., 6.], false)),
        );
        let batched = batch(
            |x| x.place(&Option::<Placement>::None, &source),
            upload(&[2, 2], &[3., 4., 5., 6.]),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        );
        assert_eq!(
            batched.map(observe).map_err(|error| error.to_string()),
            Ok((vec![first], vec![3., 4., 5., 6.], false)),
        );
        let input = upload(&[2], &[3., 4.]);
        let operation = PlaceOperation::new(None, source.clone());
        let interpreted = InterpretableOperation::interpret(&operation, &input.domain(), &EmptyRegionDriver, &[input])
            .map(|outputs| outputs.into_iter().map(observe).collect::<Vec<_>>())
            .map_err(|error| error.to_string());
        assert_eq!(interpreted, Ok(vec![(vec![first], vec![3., 4.], false)]));

        let committed = upload(&[2], &[3., 4.]).to_placement(&session, devices[0]).unwrap();
        let rejection = format!(
            "`{PLACE_OPERATION_NAME}` source sharding `{}` does not match the input sharding `{}`",
            source.sharding(),
            committed.sharding(),
        );
        assert_eq!(
            differentiate_at(committed.clone())
                .linearize(|x| x.place(&Option::<Placement>::None, &source))
                .map(|_| ())
                .map_err(|error| error.to_string()),
            Err(rejection.clone()),
        );
        assert_eq!(
            committed.place(&Option::<Placement>::None, &source).map(|_| ()).map_err(|error| error.to_string()),
            Err(rejection),
        );
    }

    #[test]
    fn test_session_place_uploads_host_leaves() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[1]).unwrap();

        // Without a destination, a host leaf is uploaded to the first local device without being committed to it, also
        // when the session has a mesh.
        let scalar = session.place(2.5f32, PlacementOptions::default()).unwrap();
        let flag = session.place(true, PlacementOptions::default()).unwrap();
        assert_eq!(scalar.r#type().into_owned().with_sharding(None).unwrap(), ArrayType::scalar(DataType::F32));
        assert_eq!(scalar.mesh().devices(), [Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap()]);
        assert!(!scalar.is_committed());
        let meshed_session = session.with_device_mesh(domain_mesh(client, "x", 2));
        assert!(!meshed_session.place(2.5f32, PlacementOptions::default()).unwrap().is_committed());
        assert_eq!(scalar.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![2.5]);
        assert_eq!(flag.data_type(), DataType::Boolean);
        assert_eq!(flag.copy_to_host_bytes().unwrap(), vec![1u8]);

        // A destination device places the leaf on that device.
        let placed = session
            .place(
                bf16::from_f32(1.5),
                PlacementOptions::new(
                    Some(PlacementRequest::from(device)),
                    Option::<PlacementRequest>::None,
                    Option::<bool>::None,
                    Option::<Option<bool>>::None,
                ),
            )
            .unwrap();
        assert_eq!(placed.mesh().devices(), [device]);
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

        // A host value uploaded without a destination, or with one that names only its storage, lives on a device that
        // this process chose on its own, and a destination that names a device does not.
        let local = session.place(1f32, PlacementOptions::default()).unwrap();
        assert!(local.is_locally_assigned());
        assert!(session.place(1f32, PlacementOptions::to(pinned)).unwrap().is_locally_assigned());
        assert!(!session.place(1f32, PlacementOptions::to(device)).unwrap().is_locally_assigned());

        // An array keeps the fact without a destination and with an in-place one, eagerly and when a staged in-place
        // placement executes on it, and a destination that names a device clears it.
        assert!(session.place(local.clone(), PlacementOptions::default()).unwrap().is_locally_assigned());
        assert!(session.place(local.clone(), PlacementOptions::to(pinned)).unwrap().is_locally_assigned());
        let staged =
            session.bind(PlaceOperation::new(pinned, None), Vec::new(), &[Value::Array(local.clone())]).unwrap();
        let [Value::Array(staged)] = staged.as_slice() else { panic!("placing an array must produce an array") };
        assert!(staged.is_locally_assigned());
        assert!(!session.place(local, PlacementOptions::to(device)).unwrap().is_locally_assigned());
    }

    #[test]
    fn test_session_place_uploads_host_values_without_target_to_first_device_of_client() {
        // A host value that `Session::place` uploads without a destination (or with one that names only its storage)
        // goes to the first addressable device of the client, whether or not the session has a default mesh, and is a
        // choice of this process alone (i.e., locally assigned), uncommitted without a destination. The default mesh of
        // a session is the analogue of `jax.set_mesh`, under which JAX 0.12.0.1 keeps such a value on its default
        // device, uncommitted, for a mesh of one device and of two (`(Explicit,) device_put:
        // SingleDeviceSharding(device=CpuDevice(id=0)) False`), while eager results without array inputs go to the
        // mesh, committed, as `Array::from_host_buffer` without a mesh does here. Uploading onto the default mesh would
        // treat the value of each process (e.g., its own data batch) as one global value that every process supplied.
        let client = cpu_client(2);
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let single = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let session = Session::builder().client(client).build().unwrap();
        let describe = |array: Array| {
            let devices = array.mesh().devices().iter().map(Device::id).collect::<Vec<_>>();
            assert_eq!(array.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1., 2.]);
            let (committed, locally_assigned) = (array.is_committed(), array.is_locally_assigned());
            format!("devices {devices:?}, committed {committed}, locally assigned {locally_assigned}")
        };
        let mut outcomes = Vec::new();
        for (name, session) in [
            ("no default mesh", session.clone()),
            ("default mesh on CPU 1", session.with_device_mesh(DeviceMesh::new(single, vec![devices[1]]).unwrap())),
            ("default mesh on both CPUs", session.with_device_mesh(domain_mesh(client, "x", 2))),
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
                no default mesh: place devices [0], committed false, locally assigned true
                no default mesh: place in place devices [0], committed true, locally assigned true
                no default mesh: from_host_buffer devices [0], committed false, locally assigned true
                default mesh on CPU 1: place devices [0], committed false, locally assigned true
                default mesh on CPU 1: place in place devices [0], committed true, locally assigned true
                default mesh on CPU 1: from_host_buffer devices [1], committed true, locally assigned false
                default mesh on both CPUs: place devices [0], committed false, locally assigned true
                default mesh on both CPUs: place in place devices [0], committed true, locally assigned true
                default mesh on both CPUs: from_host_buffer devices [0, 1], committed true, locally assigned false"},
        );
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
            .place(
                CoreArray::matrix(4, 2, values.clone()).unwrap(),
                PlacementOptions::<Placement>::new(Some(placement.clone()), None, None, None),
            )
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
    fn test_placement_pads_short_shardings_with_replicated_dimensions() {
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
        let vector = CoreArray::vector(vec![1i32, 2, 3, 4]).unwrap();
        let matrix = CoreArray::matrix(4, 3, (0..12).collect::<Vec<i32>>()).unwrap();
        let shard_elements = |array: &Array| {
            array
                .shards()
                .iter()
                .map(|shard| values_from_bytes::<i32>(&shard.copy_to_host_bytes().unwrap()))
                .collect::<Vec<_>>()
        };

        // One sharding shorter than the rank of a leaf leaves the omitted trailing dimensions replicated, so one
        // destination splits a vector and a matrix along their first dimension, and a rank zero sharding replicates
        // both. The resolved shardings have the rank of each leaf.
        let (placed_vector, placed_matrix) =
            session.place((vector.clone(), matrix.clone()), PlacementOptions::to(placement.clone())).unwrap();
        assert_eq!(placed_vector.sharding(), &split);
        assert_eq!(shard_buffer_dimensions(&placed_vector), vec![vec![2], vec![2]]);
        assert_eq!(shard_elements(&placed_vector), vec![vec![1, 2], vec![3, 4]]);
        assert!(placed_vector.is_committed());
        assert_eq!(placed_matrix.sharding(), &split_matrix);
        assert_eq!(shard_buffer_dimensions(&placed_matrix), vec![vec![2, 3], vec![2, 3]]);
        assert_eq!(shard_elements(&placed_matrix), vec![(0..6).collect::<Vec<_>>(), (6..12).collect()]);
        assert!(placed_matrix.is_committed());
        let replicated = Sharding::replicated(logical_mesh.clone(), 0);
        let (replicated_vector, replicated_matrix) = session
            .place((vector.clone(), matrix.clone()), PlacementOptions::to(session.placement(replicated).unwrap()))
            .unwrap();
        assert_eq!(replicated_vector.sharding(), &Sharding::replicated(logical_mesh.clone(), 1));
        assert_eq!(shard_buffer_dimensions(&replicated_vector), vec![vec![4], vec![4]]);
        assert_eq!(replicated_matrix.sharding(), &Sharding::replicated(logical_mesh.clone(), 2));
        assert_eq!(shard_buffer_dimensions(&replicated_matrix), vec![vec![4, 3], vec![4, 3]]);
        assert_eq!(shard_elements(&replicated_matrix), vec![(0..12).collect::<Vec<_>>(); 2]);

        // Arrays resolve short destinations and sources in the same way, also from a resolved placement.
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
        assert_eq!(shard_elements(&resplit), vec![(0..6).collect::<Vec<_>>(), (6..12).collect()]);
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
            assert_eq!(shard_elements(&output), shard_elements(&resplit));
        }

        // A traced leaf resolves a short destination for its rank.
        let input_type = ArrayType::new_static(DataType::I32, [4, 3])
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
        assert_eq!(shard_elements(&output), vec![(0..6).collect::<Vec<_>>(), (6..12).collect()]);

        // A sharding with more dimensions than a leaf is still rejected, with the error of the type rule of `place`
        // (refer to `test_session_place_reports_type_rule_error_for_destinations_of_higher_rank`).
        let error = session.place(vector, PlacementOptions::to(session.placement(split_matrix).unwrap())).unwrap_err();
        assert!(matches!(&error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
        assert_eq!(error.to_string(), "`place` target sharding rank (2) does not match the input rank (1)");
    }

    #[test]
    fn test_session_place_reports_type_rule_error_for_destinations_of_higher_rank() {
        // A destination whose sharding has more dimensions than a leaf is rejected with the error of the type rule of
        // `place` for every kind of leaf, which a source reports as the error of a source.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let placement = || {
            Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 3))
                .map(PlacementRequest::from)
        };
        let target = || PlacementOptions::to(placement().unwrap());
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let values = [1f32, 2., 3., 4.];
        let message = format!("`{PLACE_OPERATION_NAME}` target sharding rank (3) does not match the input rank (1)");

        let host = CoreArray::from_elements(r#type.clone(), &values).unwrap();
        assert_eq!(session.place(host, target()).map(|_| ()).map_err(|error| error.to_string()), Err(message.clone()));
        let array = Array::from_host_buffer(&session, r#type.clone(), None, None, values_to_bytes(&values)).unwrap();
        let error = session.place(array.clone(), target()).unwrap_err();
        assert!(matches!(&error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
        assert_eq!(error.to_string(), message);
        let traced = RefCell::new(None);
        let _: Result<TracedProgram<ArrayType, ArrayType>, _> = trace(
            |x: ArrayTracer| {
                traced.replace(Some(session.place(x.clone(), target()).map(|_| ()).map_err(|error| error.to_string())));
                x
            },
            r#type,
        );
        assert_eq!(traced.take(), Some(Err(message)));

        let source =
            PlacementOptions::<PlacementRequest, PlacementRequest>::new(None, Some(placement().unwrap()), None, None);
        assert_eq!(
            session.place(array, source).map(|_| ()).map_err(|error| error.to_string()),
            Err(format!("`{PLACE_OPERATION_NAME}` source sharding rank (3) does not match the input rank (1)")),
        );
    }

    #[test]
    fn test_session_place_selects_default_memory_for_device_destinations() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
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

        // A bare device destination selects the default memory of that device, both on another device and on the
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

        // An explicit memory still applies, and a mesh destination without a memory keeps the memory of the array.
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

        // A bare device source accepts the memory of an array on that device.
        let checked = session
            .place(
                array.clone(),
                PlacementOptions::<PlacementRequest, PlacementRequest>::new(
                    None,
                    Some(PlacementRequest::from(devices[0])),
                    None,
                    None,
                ),
            )
            .unwrap();
        assert!(Arc::ptr_eq(checked.shards()[0].buffer().unwrap(), array.shards()[0].buffer().unwrap()));
        assert_eq!(checked.r#type().memory(), pinned);
    }

    #[test]
    fn test_session_place_moves_arrays_to_memory_destinations() {
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
    fn test_session_place_uploads_host_values_to_memory_destinations() {
        // A bare memory names no devices, so a host value is uploaded to the first addressable device, as without a
        // destination, in that memory, and the result is committed like the result of every destination.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let pinned = Memory::Host { pinned: true };
        let uploaded = session.place(CoreArray::vector(vec![1f32, 2.]).unwrap(), PlacementOptions::to(pinned)).unwrap();
        assert_eq!(uploaded.mesh().devices(), [Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap()]);
        assert_eq!(uploaded.r#type().memory(), pinned);
        assert_eq!(uploaded.shards()[0].buffer().unwrap().memory().unwrap().kind().unwrap(), "pinned_host");
        assert_eq!(uploaded.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1., 2.]);
        assert!(uploaded.is_committed());
    }

    #[test]
    fn test_session_place_combines_memory_destinations_with_other_destinations() {
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

        // An in-place source or destination that names neither a memory nor a layout says nothing, so it is rejected
        // for every kind of leaf before any leaf is placed.
        let error = session
            .place(array.clone(), options(&PlacementRequest::InPlace { memory: None, layout: None }))
            .unwrap_err();
        assert!(matches!(&error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
        assert_eq!(error.to_string(), "`place` in-place source must name a memory or a layout");
        let host_error =
            session.place(3f32, options(&PlacementRequest::InPlace { memory: None, layout: None })).unwrap_err();
        assert_eq!(host_error.to_string(), error.to_string());
        let error = session
            .place(array, PlacementOptions::to(PlacementRequest::InPlace { memory: None, layout: None }))
            .unwrap_err();
        assert!(matches!(&error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
        assert_eq!(error.to_string(), "`place` in-place target must name a memory or a layout");
    }

    #[test]
    fn test_session_place_broadcasts_root_placement_over_array_tuple() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(client_devices[0].id().unwrap(), client_devices[0].process_index().unwrap())],
        )
        .unwrap();
        let source_sharding = Sharding::replicated(source_mesh.logical_mesh().clone(), 1);
        let first_source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(source_sharding.clone())
            .unwrap();
        let first_source_array = Array::from_host_buffer(
            &session,
            first_source_type,
            source_mesh.clone(),
            None,
            values_to_bytes::<f32>(&[0.0, 1.0, 2.0, 3.0]).as_slice(),
        )
        .unwrap();
        let second_source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(source_sharding)
            .unwrap();
        let second_source_array = Array::from_host_buffer(
            &session,
            second_source_type,
            source_mesh,
            None,
            values_to_bytes::<f32>(&[10.0, 11.0, 12.0, 13.0]).as_slice(),
        )
        .unwrap();

        let target_mesh = domain_mesh(client, "x", 2);
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let moved_arrays = session
            .place(
                (first_source_array, second_source_array),
                PlacementOptions::new(
                    Some(PlacementRequest::from(Placement::new(target_mesh.clone(), target_sharding.clone()).unwrap())),
                    Option::<PlacementRequest>::None,
                    Option::<bool>::None,
                    Option::<Option<bool>>::None,
                ),
            )
            .unwrap();

        assert_eq!(
            moved_arrays.0.sharding().visualize().unwrap().render(false),
            indoc! {"
                    ┌─────┬─────┐
                    │  0  │  1  │
                    └─────┴─────┘
                "}
            .trim_end()
            .to_string()
        );
        assert_eq!(
            moved_arrays.1.sharding().visualize().unwrap().render(false),
            indoc! {"
                    ┌─────┬─────┐
                    │  0  │  1  │
                    └─────┴─────┘
                "}
            .trim_end()
            .to_string()
        );
        assert_eq!(shard_values(&moved_arrays.0), vec![vec![0.0, 1.0], vec![2.0, 3.0]]);
        assert_eq!(shard_values(&moved_arrays.1), vec![vec![10.0, 11.0], vec![12.0, 13.0]]);
    }

    #[test]
    fn test_session_place_repeats_root_options_over_large_inputs() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
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
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[1]).unwrap();
        let array = session.place(2.5f32, PlacementOptions::to(device)).unwrap();
        let array_count = 4_000;
        let input = vec![array.clone(); array_count];

        // Omitted and root options are stored once even when the input has many parameters.
        let structure = input.parameter_structure();
        let expansion = PlacementOptionValues::from_prefix(None::<PlacementRequest>, &structure);
        assert!(matches!(expansion, Ok(PlacementOptionValues::Uniform(None))));
        let expansion = PlacementOptionValues::from_prefix(Some(false), &structure);
        assert!(matches!(expansion, Ok(PlacementOptionValues::Uniform(Some(false)))));

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
    fn test_session_place_prepares_the_existing_leaves() {
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
    fn test_session_and_core_place_match_borrowed_map_keys() {
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

        // For existing arrays, None leaves preserve their destination or omit their source check independently.
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
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = PlacementRequest::from(Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap());

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
    fn test_session_place_broadcasts_option_prefixes_over_empty_subtrees() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();

        // The first destination covers the empty vector, which places nothing, and the second one places the scalar.
        let (empty, scalar) = session
            .place(
                (Vec::<f32>::new(), 1f32),
                PlacementOptions::<_, PlacementRequest>::new(
                    Some((PlacementRequest::from(devices[0]), PlacementRequest::from(devices[1]))),
                    None,
                    None,
                    None,
                ),
            )
            .unwrap();
        assert_eq!(empty.len(), 0);
        assert_eq!(scalar.mesh().devices(), [devices[1]]);
        assert!(scalar.is_committed());
        assert_eq!(scalar.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1.]);
    }

    #[test]
    fn test_session_place_validates_empty_nodes_of_option_prefixes() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
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
    fn test_session_place_donation_and_aliasing() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
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
        // `DONATE_INPUT` and returns the buffer of an array that already lives at its destination. A donated leaf is
        // therefore placed as without an aliasing option, whether or not another handle shares its buffers, and only
        // the leaf without donation above is copied. Before, the forbidden alias won and both results were copies.
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
    fn test_session_place_commits_array_leaves_with_a_destination() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let uncommitted = Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(&[1f32, 2.])).unwrap();
        let options = |target: Option<PlacementRequest>, may_alias: Option<bool>| {
            PlacementOptions::new(target, Option::<PlacementRequest>::None, Option::<bool>::None, Some(may_alias))
        };
        assert!(!uncommitted.is_committed());

        // Without a destination, an array leaf keeps its commitment, also when it is copied.
        assert!(!session.place(uncommitted.clone(), options(None, None)).unwrap().is_committed());
        assert!(!session.place(uncommitted.clone(), options(None, Some(false))).unwrap().is_committed());
        let committed = uncommitted.clone().with_committed(true);
        assert!(session.place(committed, options(None, Some(false))).unwrap().is_committed());

        // A destination commits the result, even one that leaves the leaf where it is.
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
    fn test_session_place_replicates_array_leaf_on_its_own_device() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![device],
        )
        .unwrap();
        let unreduced = Sharding::replicated(mesh.logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]).with_sharding(unreduced).unwrap();
        let array =
            Array::from_host_buffer(&session, r#type, mesh.clone(), None, values_to_bytes(&[1f32, 2.])).unwrap();

        // A device names neither a mesh nor a sharding, so an array that already lives on that device keeps its mesh
        // but becomes an ordinary replicated array, without the pending reduction along its unreduced axis.
        let target = PlacementRequest::from(device);
        let placed = session
            .place(array, PlacementOptions::<PlacementRequest>::new(Some(target), None, None, None))
            .unwrap();
        assert_eq!(placed.mesh(), &mesh);
        assert_eq!(placed.sharding(), &Sharding::replicated(mesh.logical_mesh().clone(), 1));
        assert_eq!(shard_values(&placed.exp().unwrap()), vec![vec![1f32.exp(), 2f32.exp()]]);
    }

    #[test]
    fn test_session_place_moves_array_on_single_device_mesh_onto_device() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let source_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[0]]).unwrap();
        let target_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let replicated = Sharding::replicated(logical_mesh, 1);
        let r#type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(replicated.clone().with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let array =
            Array::from_host_buffer(&session, r#type.clone(), source_mesh, None, values_to_bytes(&[1f32, 2.])).unwrap();

        // The array keeps its logical mesh over the device, and its pending reduction along `x`, which has size one,
        // is complete there, so `exp` applies to the result. That is the type that the type rule of a staged device
        // target records.
        let placed = session.place(array, PlacementOptions::to(devices[1])).unwrap();
        assert_eq!(placed.mesh(), &target_mesh);
        assert_eq!(placed.sharding(), &replicated);
        assert_eq!(
            placed.r#type().into_owned(),
            CoreOperation::infer_output_types(&PlaceOperation::new(devices[1], None), &[r#type], &[])
                .unwrap()
                .remove(0),
        );
        assert_eq!(
            placed.r#type().as_ref(),
            &ArrayType::new_static(DataType::F32, [2]).with_sharding(replicated).unwrap()
        );
        assert!(placed.is_committed());
        assert_eq!(shard_values(&placed.exp().unwrap()), vec![vec![1f32.exp(), 2f32.exp()]]);

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

        // Host leaves are rejected in the same way, even though they ignore both options otherwise.
        let error = session
            .place(
                1f32,
                PlacementOptions::<PlacementRequest, PlacementRequest, _, _>::new(
                    None,
                    None,
                    Some(true),
                    Some(Some(true)),
                ),
            )
            .unwrap_err();
        assert_eq!(error.to_string(), "array `$` cannot be donated while `may_alias` is `Some(true)`");
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
    fn test_session_place_validates_every_leaf_before_placing_any() {
        // The first leaf is donated and moves to the second device, and the second leaf fails a check that needs no
        // work to be submitted. Every such check runs for every leaf before any leaf is placed, so the call creates no
        // array and the first leaf keeps its value and its device. Before, the first leaf was placed (i.e., the session
        // created one array and could consume the donated leaf) before the second leaf was rejected.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let other_session = Session::builder().client(other_cpu_client(1)).build().unwrap();
        let upload = |session: &Session| {
            let r#type = ArrayType::new_static(DataType::F32, [2]);
            Array::from_host_buffer(session, r#type, None, None, values_to_bytes(&[1f32, 2.])).unwrap()
        };
        let first = upload(&session);
        let devices = client.addressable_devices().unwrap();
        let devices = devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
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
            let direct = second.clone().place_parameter(&session, target.clone(), source.clone(), false, None);
            let expected = format!("failed to place the parameter at `$.1`: {}", direct.unwrap_err());
            let statistics = session.array_statistics();
            let options = PlacementOptions::new(
                Some((Some(PlacementRequest::from(devices[1])), target)),
                Some((None, source)),
                Some((true, false)),
                None::<bool>,
            );
            let error = session.place((PlaceInput::from(first.clone()), second), options).unwrap_err();
            let created = session.array_statistics().created_arrays - statistics.created_arrays;
            outcomes.push((name, error.to_string() == expected, created));
        }
        assert_eq!(
            outcomes,
            [
                ("type rule of a host leaf", true, 0),
                ("unknown device of a host leaf", true, 0),
                ("unknown device of an array leaf", true, 0),
                ("source of an array leaf", true, 0),
                ("client of an array leaf", true, 0),
                ("in-place target of a foreign array leaf", true, 0),
            ],
        );

        // The case of the review reports the error of the type rule for the scalar at its path.
        let rank_two = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 2)).unwrap();
        let options = PlacementOptions::new(
            Some((Some(PlacementRequest::from(devices[1])), Some(PlacementRequest::from(rank_two)))),
            None::<PlacementRequest>,
            Some((true, false)),
            None::<bool>,
        );
        let error = session.place((PlaceInput::from(first.clone()), PlaceInput::from(2u8)), options).unwrap_err();
        assert_eq!(
            error.to_string(),
            "failed to place the parameter at `$.1`: \
             `place` target sharding rank (2) does not match the input rank (0)",
        );
        assert_eq!(first.mesh().devices(), [devices[0]]);
        assert_eq!(first.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1., 2.]);
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
            "failed to place the parameter at `$.1`: `place` target sharding rank (1) does not match the input rank (0)",
        );
        let error = session.place((Wrapper(committed), Wrapper(vector)), options).unwrap_err();
        assert!(matches!(error, Error::Session(SessionError::PlacementComparisonUnavailable)));
    }

    #[test]
    fn test_core_placement_error_preserves_backend_array_errors() {
        let session = Session::builder().client(cpu_client(1)).build().unwrap();
        let input = session.place(1u8, PlacementOptions::default()).unwrap();
        let device = Device::new(2048, input.mesh().devices()[0].process_index() + 1);
        let error = input.place(&device, None).unwrap_err();
        let expected = ArrayError::UnknownDevice { device_id: device.id(), process_index: device.process_index() };
        assert_eq!(error.to_string(), expected.to_string());
        assert!(std::error::Error::source(&error).is_none());
        assert!(HashSet::from([error.clone()]).contains(&error));
    }

    #[test]
    fn test_session_place_admission_checks_only_data_needed_by_an_unshared_leaf() {
        let client = cpu_client(2);
        let remote = Device::new(1000, 1);
        let session = session_with_remote_devices(client, [remote]);
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
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

        let position = session.next_value_comparison();
        let placed = run(PlaceInput::from(unshared.clone()), None, None).unwrap();
        assert_eq!(session.next_value_comparison(), position + 2);
        assert_eq!(placed[0].block_until_ready().unwrap_err().to_string(), "unshared producer failed");
        assert_eq!(placed[1].copy_to_host_bytes().unwrap(), [7]);
        assert_eq!(
            run(PlaceInput::from(unshared), None, Some(false)).unwrap_err().to_string(),
            "unshared producer failed",
        );

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

        let mut host = CoreArray::scalar(true).unwrap();
        host.storage_bytes_mut()[0] = 2;
        assert_eq!(
            run(PlaceInput::from(host), None, None).unwrap_err().to_string(),
            "array element 0 has invalid `bool` byte encoding [2]",
        );
    }

    #[test]
    fn test_session_place_preflights_missing_whole_shard_sources() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client.addressable_devices().unwrap();
        let devices = devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![devices[0]]).unwrap();
        let missing =
            Array::from_addressable_buffers(&session, ArrayType::scalar(DataType::U8), mesh, None, vec![]).unwrap();
        let device_id = devices[0].id();
        let direct = missing.to_placement(&session, devices[1]).unwrap_err();
        assert!(matches!(
            direct,
            Error::Array(error) if matches!(*error, ArrayError::MissingShardBuffer { shard_index: 0, device_id: id } if id == device_id),
        ));

        let created = session.array_statistics().created_arrays;
        let error = session
            .place((PlaceInput::from(3u8), PlaceInput::from(missing)), PlacementOptions::to(devices[1]))
            .unwrap_err();
        let (path, source) = failed_parameter(&error).unwrap();
        assert_eq!(path, "$.1");
        assert!(matches!(
            source,
            Error::Array(error) if matches!(**error, ArrayError::MissingShardBuffer { shard_index: 0, device_id: id } if id == device_id),
        ));
        assert_eq!(session.array_statistics().created_arrays, created);
    }

    #[test]
    fn test_session_place_prefixes_paths_of_nested_placements() {
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
    fn test_placeable_place_parameter_rejects_options_that_session_place_rejects() {
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
    fn test_session_place_validates_in_place_targets_before_placing_any_input() {
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

        // The first leaf has the wrong source memory, which is only found when it is placed. The invalid target of the
        // second leaf is rejected first, before any leaf is placed.
        let target = PlacementRequest::InPlace { memory: None, layout: None };
        let error = session.place((array.clone(), array.clone()), options(target)).unwrap_err();
        assert_eq!(
            error.to_string(),
            "failed to place the parameter at `$.1`: `place` in-place target must name a memory or a layout",
        );

        let target = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap();
        let error = session.place((array.clone(), array), options(target.into())).unwrap_err();
        let (path, error) = failed_parameter(&error).unwrap();
        assert_eq!(path, "$.0");
        assert!(matches!(error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
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

        let copied_array = session.place(source_array, PlacementOptions::default()).unwrap();
        let expected_visualization =
            format!("┌─────┬─────┐\n│{:^5}│{:^5}│\n└─────┴─────┘", local_device_id, remote_device_id);

        assert_eq!(copied_array.addressable_shards().count(), 1);
        assert_eq!(copied_array.sharding().visualize().unwrap().render(false), expected_visualization);
        let copied_shard_bytes = copied_array.device_shard(local_device_id).unwrap().copy_to_host_bytes().unwrap();
        assert_eq!(values_from_bytes::<f32>(copied_shard_bytes.as_slice()), vec![0.0, 1.0]);
        assert!(copied_array.device_shard(remote_device_id).unwrap().buffer().is_none());
    }

    #[test]
    fn test_session_place_treats_source_devices_as_informative_on_every_path() {
        // The devices that a source names are informative, as in JAX, which never compares them with where a value
        // lives. An array on the first device that is placed on the second one with a source on the third one is
        // therefore placed alike eagerly, by the replay of a staged program, and under forward and reverse mode
        // differentiation, and reverse mode sends the cotangent to the devices that the source names.
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
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
        let Value::Array(staged) =
            traced.program().interpret_in_context(&session, Value::Array(input.clone())).unwrap()
        else {
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
    fn test_session_place_validates_source_storage() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 2);
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let r#type = ArrayType::new_static(DataType::F32, [2, 2]).with_layout(layout.clone());
        let source = Placement::new(mesh.clone(), sharding).unwrap();
        let array =
            Array::from_host_buffer(&session, r#type, mesh, None, values_to_bytes(&[1f32, 2., 3., 4.])).unwrap();
        let place = |source| {
            session.place(
                array.clone(),
                PlacementOptions::<PlacementRequest, Placement>::new(None, Some(source), None, None),
            )
        };
        let checked = place(source.clone().with_memory(Memory::Device).with_layout(layout.clone())).unwrap();
        assert!(Arc::ptr_eq(checked.shards()[0].buffer().unwrap(), array.shards()[0].buffer().unwrap()));

        // The memory and the layout that the array declares are those of its type, so they are checked by the type
        // rule of `place`, as they are on every other path (refer to
        // `test_session_place_validates_source_storage_on_every_path`).
        let memory = Memory::Host { pinned: true };
        let error = place(source.clone().with_memory(memory)).unwrap_err();
        assert!(matches!(&error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
        assert_eq!(
            error.to_string(),
            format!(
                "`{PLACE_OPERATION_NAME}` source memory `Host[Pinned]` does not match the input type `{}`",
                array.r#type(),
            ),
        );

        let other_layout = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let error = place(source.with_layout(other_layout.clone())).unwrap_err();
        assert!(matches!(&error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
        assert_eq!(
            error.to_string(),
            format!(
                "`{PLACE_OPERATION_NAME}` source layout `{other_layout}` does not match the input type `{}`",
                array.r#type(),
            ),
        );
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
                (None, Ok(traced)) => {
                    match traced.program().interpret_in_context(&session, Value::Array(array.clone())) {
                        Ok(Value::Array(output)) => Ok(output),
                        Ok(_) => Err("placing an array must produce an array".to_string()),
                        Err(error) => Err(error.to_string()),
                    }
                }
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

        // An eager upload without a mesh lives on the canonical mesh of one device, as a device destination does.
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]),
            None,
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();
        assert_eq!(array.mesh().devices(), [device]);
        let checked = session
            .place(
                array.clone(),
                PlacementOptions::<PlacementRequest, PlacementRequest>::new(
                    None,
                    Some(PlacementRequest::from(device)),
                    None,
                    None,
                ),
            )
            .unwrap();
        assert!(Arc::ptr_eq(checked.shards()[0].buffer().unwrap(), array.shards()[0].buffer().unwrap()));
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

        // An array of another client that lives on a device with the same ID is rejected without a destination, and
        // transferred onto a device destination as a committed array of the receiving session.
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
    fn test_session_place_rejects_traced_leaves_over_arrays_of_another_client() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let other_client = other_cpu_client(2);
        let other_session = Session::builder().client(other_client).build().unwrap();
        let other_device = Device::from_pjrt(&other_client.addressable_devices().unwrap()[1]).unwrap();
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
        let is_rejected = |error: &Option<Error>| {
            matches!(
                error,
                Some(Error::Pjrt(ryft_pjrt::Error::InvalidArgument { message, .. }))
                    if message == "cannot associate an array with a session of a different PJRT client",
            )
        };

        // A traced leaf over arrays of another client is rejected like those arrays are, with or without a target,
        // because the devices of the receiving session that a target names are not the devices of those arrays.
        for target in [None, Some(PlacementRequest::from(other_device))] {
            let options = || PlacementOptions::<PlacementRequest>::new(target.clone(), None, None, None);

            // A batched leaf wraps the packed array.
            let mut error = None;
            let _: Array = batch(
                |x| match other_session.place(x.clone(), options()) {
                    Ok(output) => Ok(output),
                    Err(place_error) => {
                        error = Some(place_error);
                        Ok(x)
                    }
                },
                input.clone(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
            assert!(is_rejected(&error));

            // A differentiated leaf wraps the primal array through a partially evaluated value.
            let mut error = None;
            let _ = differentiate_at(input.clone())
                .vjp(|x| match other_session.place(x.clone(), options()) {
                    Ok(output) => Ok(output),
                    Err(place_error) => {
                        error = Some(place_error);
                        Ok(x)
                    }
                })
                .unwrap();
            assert!(is_rejected(&error));

            // A differentiated leaf also wraps its tangent, which is checked like its primal, so a tangent of another
            // client is rejected even when the primal is an array of the receiving session.
            let mut error = None;
            let _: (Array, Array) = differentiate_at(input.clone())
                .jvp(other_tangent.clone(), |x| match session.place(x.clone(), options()) {
                    Ok(output) => Ok(output),
                    Err(place_error) => {
                        error = Some(place_error);
                        Ok(x)
                    }
                })
                .unwrap();
            assert!(is_rejected(&error));
        }

        // A staged leaf wraps no arrays, so any session can place it.
        let traced: Result<TracedProgram<ArrayType, ArrayType>, _> = trace(
            |input: ArrayTracer| other_session.place(input, PlacementOptions::default()).unwrap(),
            ArrayType::new_static(DataType::F32, [2]),
        );
        assert!(traced.is_ok());
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
        let is_rejected = |error: &Option<Error>| {
            matches!(
                error,
                Some(Error::Pjrt(ryft_pjrt::Error::InvalidArgument { message, .. }))
                    if message == "cannot place a traced value over arrays of another session",
            )
        };

        // A traced leaf places its arrays with the session that they belong to, which cannot become the receiving
        // session, so a traced leaf over arrays of another session of the same client is rejected, with or without
        // a target.
        let input = upload(&other_session);
        for target in [None, Some(PlacementRequest::from(device))] {
            let mut error = None;
            let _: Array = batch(
                |x| match session.place(x.clone(), options(target.clone())) {
                    Ok(output) => Ok(output),
                    Err(place_error) => {
                        error = Some(place_error);
                        Ok(x)
                    }
                },
                input.clone(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
            assert!(is_rejected(&error));

            let mut error = None;
            let _ = differentiate_at(input.clone())
                .vjp(|x| match session.place(x.clone(), options(target.clone())) {
                    Ok(output) => Ok(output),
                    Err(place_error) => {
                        error = Some(place_error);
                        Ok(x)
                    }
                })
                .unwrap();
            assert!(is_rejected(&error));
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
        assert!(output.session().device_mesh().is_err());
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
            |input: CompileTracer| {
                session
                    .place(input, PlacementOptions::<PlacementRequest>::new(Some(target), None, None, None))
                    .unwrap()
            },
            input_type.clone(),
            &session,
            mesh.clone(),
        )
        .unwrap();
        let input =
            Array::from_host_buffer(&session, input_type, mesh.clone(), None, values_to_bytes(&[1f32, 2., 3., 4.]))
                .unwrap();
        let output = compiled.executable().execute(&session, input).unwrap();
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(output.sharding(), &sharding);
        assert_eq!(shard_values(&output), vec![vec![1., 2.], vec![3., 4.]]);
    }

    #[test]
    fn test_session_place_places_traced_values_in_place_in_compiled_functions() {
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
    fn test_session_place_places_traced_leaves_on_sole_device_of_compiled_mesh() {
        let client = cpu_client(2);
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let sharding = Sharding::replicated(logical_mesh, 1);
        let input_type = ArrayType::new_static(DataType::F32, [2]).with_sharding(sharding.clone()).unwrap();
        let session = Session::builder().client(client).build().unwrap();

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
    }

    #[test]
    fn test_session_place_completes_pending_reduction_of_traced_values_on_sole_device_of_compiled_mesh() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let replicated = Sharding::replicated(logical_mesh, 1);
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
    fn test_session_place_combines_traced_values_on_sole_device_with_other_values() {
        let client = cpu_client(1);
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![device]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(logical_mesh, 1))
            .unwrap();
        let session = Session::builder().client(client).build().unwrap();

        // The placed value keeps the sharding that the input is tracked with over the mesh of the function, so it can
        // be added to another input of that function.
        let compiled: CompiledFunction<(ArrayType, ArrayType), ArrayType> = compile(
            |(x, y): (CompileTracer, CompileTracer)| {
                session.place(x, PlacementOptions::to(device)).unwrap().add(&y).unwrap()
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
                let %2:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [target=device(0)] %0
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
    fn test_session_place_combines_gathered_values_with_uploads_on_their_device() {
        // A value that a device destination gathers lives on the canonical logical mesh of one device, which is also
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
        let inputs = (Value::Array(split), Value::Array(upload));
        let Value::Array(sum) = traced.program().interpret_in_context(&session, inputs).unwrap() else {
            panic!("adding two arrays must produce an array");
        };
        assert_eq!((sum.mesh(), shard_values(&sum)), (&canonical_mesh, vec![vec![2., 3., 4., 5.]]));
    }

    #[test]
    fn test_session_place_places_traced_values_on_sole_device_through_every_staging_entry_point() {
        let client = cpu_client(1);
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![device]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(logical_mesh, 1))
            .unwrap();
        let session = Session::builder().client(client).build().unwrap();
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
        let output = function.execute((), input.clone()).unwrap();
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(output.r#type().memory(), memory);
        assert!(output.is_committed());
        assert_eq!(shard_values(&output), vec![vec![1., 2.]]);

        // A stateful function traces values of the array IR type universe, which `Session::place` does not accept,
        // so it stages the device target through the `PlaceDispatch` primitive of `Place` directly.
        let device_target = PlacementRequest::Device { device, memory: Some(memory), layout: None };
        let compiled = compile_statefully::<_, Type, Type>(
            |input| {
                Ok(<StatefulCompileTracer as CoreValue>::Dispatch::place(&input, Some(&device_target), None, true)?)
            },
            Type::Array(input_type),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let Value::Array(output) = compiled.execute_statefully_and_wait(&session, Value::Array(input)).unwrap() else {
            panic!("placing an array must produce an array");
        };
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(output.r#type().memory(), memory);
        assert!(output.is_committed());
        assert_eq!(shard_values(&output), vec![vec![1., 2.]]);
    }

    #[test]
    fn test_session_place_lifts_host_leaves_and_arrays_of_traced_trees() {
        // `jax.jit(lambda x: jax.device_put((x, 2.0, np.ones(3), a), jax.devices()[1]))` stages its host leaves and the
        // concrete array `a` that it closes over as values that its `device_put` places like `x` (JAX 0.12.0.1 accepts
        // the tree in one call), and a tree of `PlaceInput`s over traced values does the same in a compiled and in a
        // staged function. The same tree over runtime arrays uploads its host leaves.
        let client = cpu_client(2);
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [])
            .with_sharding(Sharding::replicated(logical_mesh, 0))
            .unwrap();
        let session = Session::builder().client(client).build().unwrap();
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
    fn test_session_place_rejects_invalid_mixed_trace_before_uploading_host_leaves() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let rank_one = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let error = RefCell::new(None);
        let before = session.array_statistics();
        let _: StagedFunction<ArrayType, ArrayType> = stage(
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
        let error = error.into_inner().unwrap();
        let (path, source) = failed_parameter(&error).unwrap();
        assert_eq!(path, "$.1");
        assert!(matches!(source, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
        assert_eq!(session.array_statistics(), before);
    }

    #[test]
    fn test_session_place_broadcasts_options_over_host_leaves_of_traced_trees() {
        // Each host leaf of a tree of traced values is placed with the options of its position in the option trees, as
        // `jax.device_put((x, 2.0, np.ones(3)), (None, device, None))` gives each leaf its own device inside `jit`.
        let client = cpu_client(2);
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [])
            .with_sharding(Sharding::replicated(logical_mesh, 0))
            .unwrap();
        let session = Session::builder().client(client).build().unwrap();
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
        let client = cpu_client(1);
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![device]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(logical_mesh, 1))
            .unwrap();
        let session = Session::builder().client(client).build().unwrap();
        let message = |error: Option<Error>| match error {
            Some(Error::Pjrt(ryft_pjrt::Error::InvalidArgument { message, .. })) => message,
            error => panic!("expected an invalid argument error, got {error:?}"),
        };

        // A host leaf joins the trace of the traced leaves of its tree, so they must belong to one trace, whereas JAX
        // lifts the input of an enclosing function into the trace of a nested one.
        let error = RefCell::new(None);
        let _: CompiledFunction<ArrayType, ArrayType> = compile(
            |x: CompileTracer| {
                let _: CompiledFunction<ArrayType, ArrayType> = compile(
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
                x
            },
            input_type.clone(),
            &session,
            mesh.clone(),
        )
        .unwrap();
        assert_eq!(message(error.take()), "cannot place traced values of different traces in one call");

        // A host leaf without traced leaves in its tree has no trace to join.
        let host = PlaceInput::<CompileTracer>::Host(CoreArray::scalar(2f32).unwrap());
        assert_eq!(
            message(session.place(host, PlacementOptions::to(device)).err()),
            "cannot place a host value or a runtime array as a traced value without a traced value in the same call",
        );
    }

    #[test]
    fn test_session_place_checks_wrapped_tracer_destinations_in_the_compilation_session() {
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
    }

    #[test]
    fn test_session_place_accepts_unregistered_source_devices_for_compile_tracers() {
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
    fn test_session_place_rejects_traced_targets_outside_compiled_mesh_and_admits_sources_there() {
        let client = cpu_client(2);
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let session = Session::builder().client(client).build().unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2]);
        let place_error = |device: Device, mesh: &DeviceMesh| {
            let compiled: Result<CompiledFunction<ArrayType, ArrayType>, _> = compile(
                |input: CompileTracer| session.place(input, PlacementOptions::to(device)).unwrap(),
                input_type.clone(),
                &session,
                mesh.clone(),
            );
            compiled.err().unwrap()
        };

        // A device that is not the device of the mesh of the function is rejected.
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![devices[0]],
        )
        .unwrap();
        let error = place_error(devices[1], &mesh);
        assert!(matches!(
            &error,
            Error::Lowering(LoweringError::DeviceOutsideProgramMesh { device, mesh: program_mesh })
                if device == &devices[1] && program_mesh == &mesh,
        ));
        assert_eq!(
            error.to_string(),
            format!(
                "`place` cannot move a value to the device {} inside a program compiled for the devices [{}] of mesh \
                 `{}`",
                devices[1].id(),
                devices[0].id(),
                mesh.logical_mesh(),
            ),
        );

        // So is a device of a mesh with more than one device, because the value is not placed on that device alone.
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let error = place_error(devices[0], &mesh);
        assert!(matches!(
            &error,
            Error::Lowering(LoweringError::DeviceOutsideProgramMesh { device, mesh: program_mesh })
                if device == &devices[0] && program_mesh == &mesh,
        ));
        assert_eq!(
            error.to_string(),
            format!(
                "`place` cannot move a value to the device {} inside a program compiled for the devices [{}, {}] of \
                 mesh `{}`",
                devices[0].id(),
                devices[0].id(),
                devices[1].id(),
                mesh.logical_mesh(),
            ),
        );

        // A source only states where the input lives, and the forward function moves nothing to it, so a device source
        // of a value without a tracked sharding that is not the only device of the program mesh compiles, with or
        // without a target, and the function returns its input on the program mesh. Before, it was rejected because
        // transposition would return the cotangent to that device.
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let observe = |array: &Array| {
            let values = array.copy_to_host().unwrap().elements::<f32>().unwrap();
            (array.mesh().devices().to_vec(), array.is_committed(), values)
        };
        let bytes = values_to_bytes(&[1f32, 2.]);
        let input = Array::from_host_buffer(&session, input_type.clone(), mesh.clone(), None, &bytes).unwrap();
        for target in [Some(PlacementRequest::from(Memory::Device)), None] {
            let options = || {
                PlacementOptions::new(
                    target.clone(),
                    Some(PlacementRequest::from(devices[0])),
                    Option::<bool>::None,
                    Option::<Option<bool>>::None,
                )
            };
            let compiled: CompiledFunction<ArrayType, ArrayType> = compile(
                |input: CompileTracer| session.place(input, options()).unwrap(),
                input_type.clone(),
                &session,
                mesh.clone(),
            )
            .unwrap();
            let output = compiled.executable().execute(&session, input.clone()).unwrap();
            assert_eq!(observe(&output), (devices[..2].to_vec(), true, vec![1., 2.]));
        }

        // So does a resolved source on devices outside the program mesh, and the function returns what the eager
        // placement returns. Its reverse mode derivative inside the same compiled function holds only the transposed
        // placement, whose target is that source, so it is still rejected as a target outside the program mesh.
        let first_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![devices[0]],
        )
        .unwrap();
        let second_mesh = DeviceMesh::new(first_mesh.logical_mesh().clone(), vec![devices[1]]).unwrap();
        let source = PlacementRequest::from(
            Placement::new(second_mesh.clone(), Sharding::replicated(second_mesh.logical_mesh().clone(), 1)).unwrap(),
        );
        let options = || {
            PlacementOptions::new(
                Some(PlacementRequest::from(Memory::Device)),
                Some(source.clone()),
                Option::<bool>::None,
                Option::<Option<bool>>::None,
            )
        };
        let reverse: Result<CompiledFunction<(ArrayType, ArrayType), ArrayType>, _> = compile(
            |(x, seed): (CompileTracer, CompileTracer)| {
                let (_, pullback) = differentiate_at(x).vjp(|x| Ok(session.place(x, options()).unwrap())).unwrap();
                pullback.apply_materialized(seed).unwrap()
            },
            (input_type.clone(), input_type.clone()),
            &session,
            first_mesh.clone(),
        );
        let forward: CompiledFunction<ArrayType, ArrayType> = compile(
            |input: CompileTracer| session.place(input, options()).unwrap(),
            input_type.clone(),
            &session,
            first_mesh.clone(),
        )
        .unwrap();
        let input_type = input_type.with_sharding(Sharding::replicated(first_mesh.logical_mesh().clone(), 1)).unwrap();
        let input = Array::from_host_buffer(&session, input_type, first_mesh.clone(), None, &bytes).unwrap();
        let compiled = observe(&forward.executable().execute(&session, input.clone()).unwrap());
        let eager = observe(&session.place(input, options()).unwrap());
        assert_eq!((compiled, eager.clone()), (eager, (vec![devices[0]], true, vec![1., 2.])));
        let error = reverse.err().unwrap();
        assert!(matches!(
            &error,
            Error::Lowering(LoweringError::PlacementOutsideProgramMesh { target, mesh: program_mesh })
                if target == &second_mesh && program_mesh == &first_mesh,
        ));
    }

    #[test]
    fn test_session_place_admits_split_sources_of_traced_values_in_bounded_dynamic_compiled_functions() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let extent = DimensionVariable::new("n", DimensionBounds::new(1, Some(8)).unwrap());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.into()]));
        let place_onto = |target: PlacementRequest, source: Option<PlacementRequest>| {
            let options =
                PlacementOptions::new(Some(target), source, Option::<bool>::None, Option::<Option<bool>>::None);
            compile::<_, ArrayType, ArrayType>(
                |input: CompileTracer| session.place(input, options.clone()).unwrap(),
                input_type.clone(),
                &session,
                mesh.clone(),
            )
        };
        let place = |sharding: Sharding| {
            place_onto(
                PlacementRequest::from(Memory::Device),
                Some(PlacementRequest::from(Placement::new(mesh.clone(), sharding).unwrap())),
            )
        };

        // Bounded dynamic programs run one fully replicated replica per device, so a split target would be ignored and
        // is rejected (refer to `test_place_operation_lower_rejects_splits_in_bounded_dynamic_programs`), but a
        // source only states where the input lives and the program moves nothing to it, so a split source compiles.
        // Before, it was rejected because transposition would turn it into the target of the cotangent.
        let split = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(place(split).map(|_| ()).map_err(|error| error.to_string()), Ok(()));

        // A replicated source is what every replica holds anyway, and so is a source or a target that leaves a
        // dimension unconstrained, which an array stores replicated, as the unconstrained sharding override of a call
        // is.
        assert!(place(Sharding::replicated(mesh.logical_mesh().clone(), 1)).is_ok());
        let unconstrained = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::Unconstrained]).unwrap();
        assert!(place(unconstrained.clone()).is_ok());
        assert!(place_onto(PlacementRequest::from(Placement::new(mesh.clone(), unconstrained).unwrap()), None).is_ok());
    }

    #[test]
    fn test_session_place_places_traced_values_with_sources_in_compiled_mesh() {
        let client = cpu_client(2);
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let session = Session::builder().client(client).build().unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("a", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let upload = |values: &[f32]| {
            Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), None, values_to_bytes(values)).unwrap()
        };

        // A device source on the sole device of the program mesh, and a resolved source over the program mesh, are
        // accepted in the forward function and in its reverse mode derivative inside the same compiled function, whose
        // transposed placement moves the cotangent onto those devices.
        let sources = [
            PlacementRequest::from(devices[1]),
            PlacementRequest::from(Placement::new(mesh.clone(), Sharding::replicated(logical_mesh, 1)).unwrap()),
        ];
        for source in sources {
            let options = || {
                PlacementOptions::new(
                    Some(PlacementRequest::from(Memory::Device)),
                    Some(source.clone()),
                    Option::<bool>::None,
                    Option::<Option<bool>>::None,
                )
            };
            let forward: CompiledFunction<ArrayType, ArrayType> = compile(
                |x: CompileTracer| session.place(x, options()).unwrap(),
                r#type.clone(),
                &session,
                mesh.clone(),
            )
            .unwrap();
            let output = forward.executable().execute(&session, upload(&[1., 2., 3., 4.])).unwrap();
            assert_eq!(output.mesh(), &mesh);
            assert_eq!(shard_values(&output), vec![vec![1., 2., 3., 4.]]);

            let reverse: CompiledFunction<(ArrayType, ArrayType), ArrayType> = compile(
                |(x, seed): (CompileTracer, CompileTracer)| {
                    let (_, pullback) = differentiate_at(x).vjp(|x| Ok(session.place(x, options()).unwrap())).unwrap();
                    pullback.apply_materialized(seed).unwrap()
                },
                (r#type.clone(), r#type.clone()),
                &session,
                mesh.clone(),
            )
            .unwrap();
            let cotangent = reverse
                .executable()
                .execute(&session, (upload(&[1., 2., 3., 4.]), upload(&[5., 6., 7., 8.])))
                .unwrap();
            assert_eq!(cotangent.mesh(), &mesh);
            assert_eq!(shard_values(&cotangent), vec![vec![5., 6., 7., 8.]]);
        }
    }

    #[test]
    fn test_session_place_applies_memory_and_layout_of_devices_to_traced_values_in_compiled_functions() {
        let client = cpu_client(1);
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![device],
        )
        .unwrap();
        let session = Session::builder().client(client).build().unwrap();
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
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![devices[0]],
        )
        .unwrap();
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [3]),
            source_mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2., 3.]),
        )
        .unwrap();

        // The primal moves to the target device, and the pullback moves the cotangent back to the source mesh. The
        // donation option does not apply to the traced leaf, and a forced copy of a moved primal is that move itself.
        let options = PlacementOptions::new(
            Some(PlacementRequest::from(devices[1])),
            Option::<PlacementRequest>::None,
            Some(true),
            Some(Some(false)),
        );
        let (value, pullback) =
            differentiate_at(input.clone()).vjp(|x| Ok(session.place(x, options).unwrap())).unwrap();
        assert_eq!(value.mesh().devices(), [devices[1]]);
        assert_eq!(shard_values(&value), vec![vec![1., 2., 3.]]);
        let cotangent = Array::from_host_buffer(
            &session,
            value.r#type().into_owned(),
            value.mesh().clone(),
            None,
            values_to_bytes(&[4f32, 5., 6.]),
        )
        .unwrap();
        let input_cotangent = pullback.apply_materialized(cotangent).unwrap();
        assert_eq!(input_cotangent.mesh(), &source_mesh);
        assert_eq!(input_cotangent.r#type(), input.r#type());
        assert_eq!(shard_values(&input_cotangent), vec![vec![4., 5., 6.]]);

        // A device names a device and not a mesh, so a traced leaf over an array on that device resolves the device
        // to the mesh of the array, both as a source and as a destination that keeps the leaf where it is.
        let options = PlacementOptions::new(
            Some(PlacementRequest::from(devices[1])),
            Some(PlacementRequest::from(devices[0])),
            Option::<bool>::None,
            Option::<Option<bool>>::None,
        );
        let (value, _) = differentiate_at(input.clone()).vjp(|x| Ok(session.place(x, options).unwrap())).unwrap();
        assert_eq!(value.mesh().devices(), [devices[1]]);
        let options = PlacementOptions::new(
            Some(PlacementRequest::from(devices[0])),
            Option::<PlacementRequest>::None,
            Option::<bool>::None,
            Option::<Option<bool>>::None,
        );
        let (value, _) = differentiate_at(input.clone()).vjp(|x| Ok(session.place(x, options).unwrap())).unwrap();
        assert_eq!(value.mesh(), &source_mesh);
        assert_eq!(shard_values(&value), vec![vec![1., 2., 3.]]);
    }

    #[test]
    fn test_session_place_places_traced_values_in_place_in_differentiation() {
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
    fn test_session_place_completes_pending_reduction_of_traced_values_in_differentiation() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let source_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[0]]).unwrap();
        let target_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let replicated = Sharding::replicated(logical_mesh, 1);
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
    fn test_session_place_places_traced_leaves_in_differentiation_without_destination() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let source_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[0]]).unwrap();
        let cotangent_mesh = DeviceMesh::new(logical_mesh, vec![devices[1]]).unwrap();
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [3]),
            source_mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2., 3.]),
        )
        .unwrap();

        // Without a destination, the primal stays where it lives, and the pullback moves a cotangent that lives on
        // another device back to the primal, both with the inferred source and with an explicit one.
        for source in [None, Some(PlacementRequest::from(devices[0]))] {
            let options = PlacementOptions::<PlacementRequest>::new(None, source, None, None);
            let (value, pullback) =
                differentiate_at(input.clone()).vjp(|x| Ok(session.place(x, options).unwrap())).unwrap();
            assert_eq!(value.mesh(), &source_mesh);
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
            assert_eq!(input_cotangent.mesh(), &source_mesh);
            assert_eq!(shard_values(&input_cotangent), vec![vec![4., 5., 6.]]);
        }
    }

    #[test]
    fn test_session_place_stages_explicit_source_of_traced_leaves_without_destination() {
        let client = cpu_client(1);
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap()],
        )
        .unwrap();
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let r#type = ArrayType::new_static(DataType::F32, [2]).with_sharding(sharding.clone()).unwrap();
        let source = PlacementRequest::from(session.placement(sharding).unwrap());

        // Without a destination, the placement is staged without a target, so the value stays where it is, and with
        // the source, which only the transposition of the placement uses.
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            |input: ArrayTracer| {
                let options = PlacementOptions::<PlacementRequest>::new(None, Some(source.clone()), None, None);
                session.place(input, options).unwrap()
            },
            r#type,
        )
        .unwrap();
        assert_eq!(
            traced.program().to_string(),
            indoc! {r#"
                lambda %0:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [
                    source={mesh<['x'=1:explicit]>, devices=[0], sharding={mesh<['x'=1:explicit]>, [{}]}},
                ] %0
                in (%1)
            "#}
            .trim_end(),
        );
    }

    #[test]
    fn test_session_place_stages_explicit_source_of_traced_leaves_without_destination_for_pullback() {
        let client = cpu_client(1);
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap()],
        )
        .unwrap();
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let r#type = ArrayType::new_static(DataType::F32, [2]).with_sharding(sharding.clone()).unwrap();
        let source = PlacementRequest::from(session.placement(sharding).unwrap());

        // The pullback places the cotangent on the source, with the memory of the primal and without a source of its
        // own, because the forward placement has no target.
        let traced: TracedProgram<(ArrayType, ArrayType), ArrayType> = trace(
            |(input, cotangent): (ArrayTracer, ArrayTracer)| {
                let options = PlacementOptions::<PlacementRequest>::new(None, Some(source.clone()), None, None);
                let (_, pullback) = differentiate_at(input).vjp(|x| Ok(session.place(x, options).unwrap())).unwrap();
                pullback.apply_materialized(cotangent).unwrap()
            },
            (r#type.clone(), r#type),
        )
        .unwrap();
        assert_eq!(
            traced.program().to_string(),
            indoc! {r#"
                lambda %0:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}], %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %2:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [
                    target={mesh<['x'=1:explicit]>, devices=[0], sharding={mesh<['x'=1:explicit]>, [{}]}, memory=Device},
                    may_alias=false,
                ] %1
                in (%2)
            "#}
            .trim_end(),
        );
    }

    #[test]
    fn test_session_place_stages_device_source_of_traced_values_for_pullback() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let meshes = devices
            .iter()
            .map(|&device| DeviceMesh::new(logical_mesh.clone(), vec![device]).unwrap())
            .collect::<Vec<_>>();
        let r#type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(logical_mesh.clone(), 1))
            .unwrap();
        let options =
            || PlacementOptions::to(PlacementRequest::from(devices[1])).with_source(PlacementRequest::from(devices[0]));
        let interpret = |traced: &TracedProgram<(ArrayType, ArrayType), ArrayType>, values: [(&[f32], usize); 2]| {
            let [first, second] = values.map(|(values, mesh): (&[f32], usize)| {
                Value::Array(
                    Array::from_host_buffer(
                        &session,
                        r#type.clone(),
                        meshes[mesh].clone(),
                        None,
                        values_to_bytes(values),
                    )
                    .unwrap(),
                )
            });
            let Value::Array(output) = traced.program().interpret_in_context(&session, (first, second)).unwrap() else {
                panic!("placing an array must produce an array");
            };
            output
        };

        // The devices of a traced input are not known, so a bare device source stays a device, and the pullback moves
        // the cotangent from the destination back onto it over the logical mesh of the input, as JAX's transposition of
        // `device_put` returns it to that device. Interpreting
        // the pullback on an input on the first device and a cotangent on the second one returns the gradient on the
        // first device.
        let traced: TracedProgram<(ArrayType, ArrayType), ArrayType> = trace(
            |(input, cotangent): (ArrayTracer, ArrayTracer)| {
                let (_, pullback) = differentiate_at(input).vjp(|x| Ok(session.place(x, options()).unwrap())).unwrap();
                pullback.apply_materialized(cotangent).unwrap()
            },
            (r#type.clone(), r#type.clone()),
        )
        .unwrap();
        assert_eq!(
            traced.program().to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}], %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %2:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [
                    inferred_source=device(1),
                    target={mesh<['x'=1:explicit]>, devices=[0], \
                        sharding={mesh<['x'=1:explicit]>, [{}]}, memory=Device},
                    may_alias=false,
                ] %1
                in (%2)"
            },
        );
        let gradient = interpret(&traced, [(&[1., 2.], 0), (&[3., 4.], 1)]);
        assert_eq!(gradient.mesh(), &meshes[0]);
        assert_eq!(gradient.r#type().as_ref(), &r#type);
        assert_eq!(shard_values(&gradient), vec![vec![3., 4.]]);

        // Transposing the pullback again swaps the two devices back, so the cotangent of the gradient moves onto the
        // second device, which is the destination of the forward placement.
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
            traced.program().to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}], %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] .
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
        let cotangent = interpret(&traced, [(&[1., 2.], 0), (&[5., 6.], 0)]);
        assert_eq!(cotangent.mesh(), &meshes[1]);
        assert_eq!(cotangent.r#type().as_ref(), &r#type);
        assert_eq!(shard_values(&cotangent), vec![vec![5., 6.]]);
    }

    #[test]
    fn test_session_place_keeps_traced_leaves_without_destination_where_they_live() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let first_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[0]]).unwrap();
        let second_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let sharding = Sharding::replicated(logical_mesh, 1);
        let r#type = ArrayType::new_static(DataType::F32, [3]).with_sharding(sharding.clone()).unwrap();
        let source = Placement::new(second_mesh.clone(), sharding).unwrap();
        let input = Array::from_host_buffer(
            &session,
            r#type.clone(),
            first_mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2., 3.]),
        )
        .unwrap();
        let options = || PlacementOptions::<PlacementRequest, Placement>::new(None, Some(source.clone()), None, None);

        // A placement with only a source is staged without a target, so replaying it on an array that lives on the
        // first device leaves the array there, even though the source names the second device.
        let traced: TracedProgram<ArrayType, ArrayType> =
            trace(|input: ArrayTracer| session.place(input, options()).unwrap(), r#type).unwrap();
        let Value::Array(output) =
            traced.program().interpret_in_context(&session, Value::Array(input.clone())).unwrap()
        else {
            panic!("placing an array must produce an array");
        };
        assert_eq!(output.mesh(), &first_mesh);
        assert_eq!(shard_values(&output), vec![vec![1., 2., 3.]]);

        // Its pullback moves the cotangent to the source.
        let (value, pullback) =
            differentiate_at(input.clone()).vjp(|x| Ok(session.place(x, options()).unwrap())).unwrap();
        assert_eq!(value.mesh(), &first_mesh);
        assert_eq!(shard_values(&value), vec![vec![1., 2., 3.]]);
        let cotangent = Array::from_host_buffer(
            &session,
            value.r#type().into_owned(),
            first_mesh,
            None,
            values_to_bytes(&[4f32, 5., 6.]),
        )
        .unwrap();
        let input_cotangent = pullback.apply_materialized(cotangent).unwrap();
        assert_eq!(input_cotangent.mesh(), &second_mesh);
        assert_eq!(input_cotangent.r#type(), input.r#type());
        assert_eq!(shard_values(&input_cotangent), vec![vec![4., 5., 6.]]);
    }

    #[test]
    fn test_session_place_compiles_traced_leaves_with_only_a_source() {
        let client = cpu_client(1);
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![device],
        )
        .unwrap();
        let session = Session::builder().client(client).build().unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2]);

        // A bare device does not determine a placement for a traced value whose devices are not known, so the source
        // stays a device, which only transposition uses, and the placement without a target compiles and leaves the
        // value where it is.
        let source = PlacementRequest::from(device);
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile(
            |input: CompileTracer| {
                session
                    .place(input, PlacementOptions::<PlacementRequest>::new(None, Some(source), None, None))
                    .unwrap()
            },
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
    fn test_session_place_copies_traced_leaves_without_destination_and_source() {
        let client = cpu_client(1);
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap()],
        )
        .unwrap();
        let session = Session::builder().client(client).build().unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let input =
            Array::from_host_buffer(&session, r#type.clone(), mesh, None, values_to_bytes(&[1f32, 2.])).unwrap();

        // A placement without a destination and a source that must not alias its input is still staged, so replaying
        // it while the caller holds the input returns a copy of the buffer of the input.
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            |input: ArrayTracer| {
                let options = PlacementOptions::<PlacementRequest, PlacementRequest, bool, _>::new(
                    None,
                    None,
                    None,
                    Some(Some(false)),
                );
                session.place(input, options).unwrap()
            },
            r#type,
        )
        .unwrap();
        assert_eq!(
            traced.program().to_string(),
            indoc! {r#"
                lambda %0:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [may_alias=false] %0
                in (%1)
            "#}
            .trim_end(),
        );
        let Value::Array(output) =
            traced.program().interpret_in_context(&session, Value::Array(input.clone())).unwrap()
        else {
            panic!("placing an array must produce an array");
        };
        assert!(!Arc::ptr_eq(output.shards()[0].buffer().unwrap(), input.shards()[0].buffer().unwrap()));
        assert_ne!(native_buffer_addresses(&output), native_buffer_addresses(&input));
        assert_eq!(output.r#type(), input.r#type());
        assert_eq!(shard_values(&output), vec![vec![1., 2.]]);
    }

    #[test]
    fn test_session_place_places_traced_leaves_in_batching() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[1]).unwrap();
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2, 3]),
            None,
            None,
            values_to_bytes(&[1f32, 2., 3., 4., 5., 6.]),
        )
        .unwrap();

        // The target resolves for the rank of each batch item, and the packed result lives on the target device.
        let options = PlacementOptions::<PlacementRequest>::new(Some(PlacementRequest::from(device)), None, None, None);
        let output: Array =
            batch(|x| Ok(session.place(x, options).unwrap()), input, BatchAxis::new(0), BatchAxis::new(0), None)
                .unwrap();
        assert_eq!(output.mesh().devices(), [device]);
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
    fn test_session_place_reports_errors_of_arrays_in_the_order_of_the_type_rule() {
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
            let options = PlacementOptions::new(
                Some(target.clone()),
                Some(source.clone()),
                Option::<bool>::None,
                Option::<Option<bool>>::None,
            );
            let eager = session.place(input.clone(), options).err().unwrap().to_string();
            let operation = PlaceOperation::new(target.padded(1).unwrap(), source.padded(1).unwrap());
            let inferred = operation.infer_output_types(&[input.r#type().into_owned()], &[]).err().unwrap();
            let inferred = Error::from(inferred).to_string();
            assert_eq!((eager, inferred), (expected_error.to_string(), expected_error.to_string()));
        }

        // A host value is uploaded as the placement of a value without a tracked sharding in device memory, so its
        // destination is rejected with the first error of the type rule for such a value, as the destination of an
        // upload of the same value without a mesh is, and as the destination of a traced value of that type is.
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
            let _: Result<TracedProgram<ArrayType, ArrayType>, _> = trace(
                |x: ArrayTracer| {
                    session.place(x.clone(), options()).unwrap_or_else(|placement_error| {
                        error.replace(Some(placement_error.to_string()));
                        x
                    })
                },
                ArrayType::new_static(DataType::F32, [2]),
            );
            let traced = error.take().unwrap();
            assert_eq!((host, array, traced), (expected_error.clone(), expected_error.clone(), expected_error));
        }
    }

    #[test]
    fn test_session_place_agrees_across_eager_staged_compiled_and_transformed_paths() {
        // Every observable fact of one placed array, so that the arrays that different paths produce can be compared.
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

        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let pinned = Memory::Host { pinned: true };
        let single_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let double_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let observe = |array: &Array| Observed {
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
        };
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
        let single = (DeviceMesh::new(single_mesh.clone(), vec![devices[0]]).unwrap(), single_mesh.clone());
        let double = (DeviceMesh::new(double_mesh.clone(), devices[..2].to_vec()).unwrap(), double_mesh.clone());
        let other_mesh = |mesh: &DeviceMesh| {
            let other_devices = devices[2..2 + mesh.devices().len()].to_vec();
            DeviceMesh::new(mesh.logical_mesh().clone(), other_devices).unwrap()
        };
        let replicated = |mesh: &LogicalMesh| Sharding::replicated(mesh.clone(), 1);
        let unconstrained =
            |mesh: &LogicalMesh| Sharding::new(mesh.clone(), vec![ShardingDimension::Unconstrained]).unwrap();
        let canonical = DeviceMesh::new(LogicalMesh::single_device(), vec![devices[0]]).unwrap();
        let inputs = [
            ("single device", single.0.clone(), Some(replicated(&single.1)), Memory::Device),
            ("uncommitted on the canonical mesh", canonical, None, Memory::Device),
            (
                "sharded over two devices",
                double.0.clone(),
                Some(Sharding::new(double.1.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()),
                Memory::Device,
            ),
            ("replicated over two devices", double.0.clone(), Some(replicated(&double.1)), Memory::Device),
            (
                "reduced over an axis of size two",
                double.0.clone(),
                Some(replicated(&double.1).with_reduced_axes(["x"]).unwrap()),
                Memory::Device,
            ),
            (
                "unreduced over an axis of size one",
                single.0.clone(),
                Some(replicated(&single.1).with_unreduced_axes(["x"]).unwrap()),
                Memory::Device,
            ),
            (
                "unreduced over an axis of size two",
                double.0.clone(),
                Some(replicated(&double.1).with_unreduced_axes(["x"]).unwrap()),
                Memory::Device,
            ),
            ("pinned host memory", single.0.clone(), Some(replicated(&single.1)), pinned),
        ];
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));

        let mut disagreements = Vec::new();
        for (input_name, mesh, sharding, memory) in inputs {
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
                    let options = || {
                        PlacementOptions::new(
                            target.clone(),
                            source.clone(),
                            Option::<bool>::None,
                            Option::<Option<bool>>::None,
                        )
                    };
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

                    // A traced function interpreted on the session (i.e., the replay of a staged program).
                    let traced: Result<TracedProgram<ArrayType, ArrayType>, _> = trace(
                        |x: ArrayTracer| {
                            record_placement_error(&error, session.place(x.clone(), options())).unwrap_or(x)
                        },
                        input.r#type().into_owned(),
                    );
                    let staged = match (error.take(), traced) {
                        (Some(message), _) => Err(message),
                        (None, Err(error)) => Err(error.to_string()),
                        (None, Ok(traced)) => {
                            match traced.program().interpret_in_context(session, Value::Array(input.clone())) {
                                Ok(Value::Array(output)) => Ok(observe(&output)),
                                Ok(_) => Err("placing an array must produce an array".to_string()),
                                Err(error) => Err(error.to_string()),
                            }
                        }
                    };
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

                    // Reverse mode rejects one combination by design. The cotangent of a device target onto a value
                    // that is reduced over a larger axis is unreduced over it, which a placement cannot produce. A
                    // memory source of a concrete value is completed with where the value lives, so the cotangent of
                    // a value that a device target gathers returns to its devices with every source.
                    let is_gathered = mesh.devices().len() > 1 && matches!(*target_name, "device" | "own device");
                    let rejected_in_reverse_mode = match () {
                        _ if eager.is_err() || !is_gathered => None,
                        _ if input_name == "reduced over an axis of size two" => Some(format!(
                            "cannot transpose `{PLACE_OPERATION_NAME}` onto a single device for a value that is \
                             reduced over mesh axes larger than one (i.e., `x`), because its cotangent is unreduced \
                             over them",
                        )),
                        _ => None,
                    };
                    let rejected_in_batched_reverse_mode = rejected_in_reverse_mode.clone();

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
                    match (check_rejection("batched vjp", &rejected_in_batched_reverse_mode, &result), batched) {
                        (Some(disagreement), _) => disagreements.push(disagreement),
                        (None, _) if rejected_in_batched_reverse_mode.is_some() => {}
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
        // A host value is uploaded where a destination says, which is where the same destination places an upload of
        // that value without a mesh, so both kinds of leaves agree (a host value ignores sources, so none is given).
        let host = CoreArray::from_elements(ArrayType::new_static(DataType::F32, [4]), &[1f32, 2., 3., 4.]).unwrap();
        let meshed_session = session.clone().with_device_mesh(other_mesh(&single.0));
        for (target_name, target) in [
            (
                "mesh",
                Some(PlacementRequest::from(Placement::new(other_mesh(&single.0), replicated(&single.1)).unwrap())),
            ),
            ("sharding", Some(PlacementRequest::from(meshed_session.placement(replicated(&single.1)).unwrap()))),
            ("device", Some(PlacementRequest::from(devices[2]))),
            ("memory", Some(PlacementRequest::from(pinned))),
            ("none", None),
        ] {
            let session = if target_name == "sharding" { &meshed_session } else { &session };
            let options = || {
                PlacementOptions::new(
                    target.clone(),
                    Option::<PlacementRequest>::None,
                    Option::<bool>::None,
                    Option::<Option<bool>>::None,
                )
            };
            let uploaded = Array::from_host_buffer(
                session,
                ArrayType::new_static(DataType::F32, [4]),
                None,
                None,
                values_to_bytes(&[1f32, 2., 3., 4.]),
            )
            .unwrap();
            let host = session.place(host.clone(), options()).map(|output| observe(&output));
            let array = session.place(uploaded, options()).map(|output| observe(&output));
            let (host, array) = (host.map_err(|error| error.to_string()), array.map_err(|error| error.to_string()));
            if host != array {
                disagreements.push(format!("host value, target {target_name}: {host:?} != array: {array:?}"));
            }
        }

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
                let traced: Result<TracedProgram<ArrayType, ArrayType>, _> = trace(
                    |x: ArrayTracer| record_placement_error(&error, session.place(x.clone(), options())).unwrap_or(x),
                    input.r#type().into_owned(),
                );
                let staged = match (error.take(), traced) {
                    (Some(message), _) => Err(message),
                    (None, Err(error)) => Err(error.to_string()),
                    (None, Ok(traced)) => {
                        match traced.program().interpret_in_context(&session, Value::Array(input.clone())) {
                            Ok(Value::Array(output)) => Ok(observe(&output)),
                            Ok(_) => Err("placing an array must produce an array".to_string()),
                            Err(error) => Err(error.to_string()),
                        }
                    }
                };
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

        // The runtime sharding of an array replicated along an auto axis determines its shards, which its type does
        // not, so an expected source layout that overlaps them is rejected by the eager placement as forward mode
        // rejects it after it completes the source with that runtime placement, with the error of the type rule.
        let auto_matrix = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4, 4]),
            auto.clone(),
            Some(Sharding::replicated(auto_mesh.clone(), 2)),
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
        let columns =
            Sharding::new(double_mesh.clone(), vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])])
                .unwrap();
        let input = upload(
            &[4, 4],
            Some(&columns),
            &double.0,
            Memory::Device,
            &(1..=16).map(|value| value as f32).collect::<Vec<_>>(),
        );
        let layout = Layout::Strided(StridedLayout::new(vec![8, 4]));
        let options = || PlacementOptions::to(PlacementRequest::InPlace { memory: None, layout: Some(layout.clone()) });
        let eager = session
            .place(input.clone(), options())
            .map(|output| observe(&output))
            .map_err(|error| error.to_string());
        let error = RefCell::new(None::<String>);
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            |x: ArrayTracer| record_placement_error(&error, session.place(x.clone(), options())).unwrap_or(x),
            input.r#type().into_owned(),
        )
        .unwrap();
        let staged = match error.take() {
            Some(message) => Err(message),
            None => match traced.program().interpret_in_context(&session, Value::Array(input.clone())) {
                Ok(Value::Array(output)) => Ok(observe(&output)),
                Ok(_) => Err("placing an array must produce an array".to_string()),
                Err(error) => Err(error.to_string()),
            },
        };
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
        let values = (1..=16).map(|value| value as f32).collect::<Vec<_>>();
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

        // Without a layout, a destination keeps the layout of a value while it still describes each destination shard,
        // and the type rule says the same. The CPU plugin cannot create the strided layouts above, so the case of a
        // layout that does not describe the gathered shard is covered by the type rule and the reference backend, and
        // here a tiled layout, which describes every shard, is gathered from the column shards onto one device.
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
        let traced: TracedProgram<ArrayType, ArrayType> =
            trace(|x: ArrayTracer| session.place(x, gather()).unwrap(), tiled.r#type().into_owned()).unwrap();
        let staged = match traced.program().interpret_in_context(&session, Value::Array(tiled.clone())) {
            Ok(Value::Array(output)) => Ok(observe(&output)),
            Ok(_) => Err("placing an array must produce an array".to_string()),
            Err(error) => Err(error.to_string()),
        };
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
    fn test_session_place_agrees_across_paths_for_manual_regions_and_host_values() {
        // The matrix of `test_session_place_agrees_across_eager_staged_compiled_and_transformed_paths` covers values
        // outside manual regions on every path, including reverse mode and batching. The values of this matrix (i.e.,
        // values inside a manual region, values that are moved into one, host values, and the values of a `shard_map`
        // body) are rejected wherever they would move, and reverse mode and batching of an eager value do not apply to
        // them, so they are compared on the paths that do (i.e., eager, core, staged, forward mode, and compiled).
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let pinned = Memory::Host { pinned: true };
        let observe =
            |array: &Array| (array.mesh().devices().to_vec(), array.r#type().into_owned(), shard_values(array));
        let mut disagreements = Vec::new();

        // Values inside a manual region (i.e., that vary along the manual axis `m`, or that are reduced over it) can
        // only stay on their devices, and nothing can be moved into such a region, so every target that moves a value
        // is rejected on every path with the error of the type rule, even a target that is the placement of the value.
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
        for (input_name, input, targets) in inputs {
            for (target_name, target, rejection) in targets {
                let cell = format!("{input_name}, target {target_name}");
                let options =
                    || PlacementOptions::new(target.clone(), None::<PlacementRequest>, None::<bool>, None::<bool>);
                let error = RefCell::new(None::<String>);
                let eager = session.place(input.clone(), options()).map(|output| observe(&output));
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
                let core = input.place(&request, None).map(|output| observe(&output));
                let core = core.map_err(|error| Error::from(error).to_string());
                let traced: Result<TracedProgram<ArrayType, ArrayType>, _> = trace(
                    |x: ArrayTracer| record_placement_error(&error, session.place(x.clone(), options())).unwrap_or(x),
                    input.r#type().into_owned(),
                );
                let staged = match (error.take(), traced) {
                    (Some(message), _) => Err(message),
                    (None, Err(error)) => Err(error.to_string()),
                    (None, Ok(traced)) => {
                        match traced.program().interpret_in_context(&session, Value::Array(input.clone())) {
                            Ok(Value::Array(output)) => Ok(observe(&output)),
                            Ok(_) => Err("placing an array must produce an array".to_string()),
                            Err(error) => Err(error.to_string()),
                        }
                    }
                };
                let jvp = differentiate_at(input.clone())
                    .jvp(input.clone(), |x| record_placement_error(&error, session.place(x, options())))
                    .map(|(primal, _)| observe(&primal))
                    .map_err(|other| error.take().unwrap_or(other.to_string()));
                for (path, result) in [("core", core), ("staged", staged), ("jvp", jvp)] {
                    if result != eager {
                        disagreements.push(format!("{cell}, {path}: {result:?} != eager: {eager:?}"));
                    }
                }
            }
        }

        // A host value is uploaded onto a destination as the placement of a value without a tracked sharding in device
        // memory, so it is rejected or placed as an upload of the same value without a mesh is (a host value ignores
        // sources, so none is given), with the first error of the type rule for such a value, except that both split a
        // dimension unevenly. A traced value of that type follows the type rule with the even split included.
        let explicit = two_device_mesh(client, MeshAxisType::Explicit);
        let explicit_sharding =
            |dimensions: Vec<ShardingDimension>| Sharding::new(explicit.logical_mesh().clone(), dimensions).unwrap();
        let rank_2 = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let host_targets = [
            (
                "unreduced",
                PlacementRequest::from(
                    Placement::new(
                        explicit.clone(),
                        explicit_sharding(vec![ShardingDimension::Replicated]).with_unreduced_axes(["x"]).unwrap(),
                    )
                    .unwrap(),
                ),
            ),
            (
                "reduced",
                PlacementRequest::from(
                    Placement::new(
                        explicit.clone(),
                        explicit_sharding(vec![ShardingDimension::Replicated]).with_reduced_axes(["x"]).unwrap(),
                    )
                    .unwrap(),
                ),
            ),
            (
                "varying along a manual axis",
                PlacementRequest::from(Placement::new(manual.clone(), varying.clone()).unwrap()),
            ),
            (
                "device with a layout of rank 2",
                PlacementRequest::Device { device: devices[1], memory: None, layout: Some(rank_2.clone()) },
            ),
            (
                "rank 2",
                PlacementRequest::from(
                    Placement::new(explicit.clone(), explicit_sharding(vec![ShardingDimension::Replicated; 2]))
                        .unwrap(),
                ),
            ),
            (
                "unconstrained",
                PlacementRequest::from(
                    Placement::new(explicit.clone(), explicit_sharding(vec![ShardingDimension::Unconstrained]))
                        .unwrap(),
                ),
            ),
            (
                "uneven split",
                PlacementRequest::from(
                    Placement::new(explicit.clone(), explicit_sharding(vec![ShardingDimension::sharded(["x"])]))
                        .unwrap(),
                ),
            ),
            ("device", PlacementRequest::from(devices[1])),
            ("memory", PlacementRequest::from(pinned)),
        ];
        let host_type = ArrayType::new_static(DataType::F32, [3]);
        let host = CoreArray::from_elements(host_type.clone(), &[1f32, 2., 3.]).unwrap();
        for (target_name, target) in host_targets {
            let cell = format!("host value, target {target_name}");
            let options = || PlacementOptions::to(target.clone());
            let uploaded =
                Array::from_host_buffer(&session, host_type.clone(), None, None, values_to_bytes(&[1f32, 2., 3.]))
                    .unwrap();
            let host = session.place(host.clone(), options()).map(|output| (observe(&output), output.is_committed()));
            let host = host.map_err(|error| error.to_string());
            let array = session.place(uploaded, options()).map(|output| (observe(&output), output.is_committed()));
            let array = array.map_err(|error| error.to_string());
            if host != array {
                disagreements.push(format!("{cell}: {host:?} != array: {array:?}"));
            }
            let request = target.clone().padded(1).unwrap();
            let inferred =
                PlaceOperation::infer_output_type_allowing_uneven_splits(&host_type, None, Some(&request), None);
            let inferred = inferred.map_err(|error| Error::from(error).to_string());
            if let (Err(host), Err(inferred)) = (&host, &inferred)
                && host != inferred
            {
                disagreements.push(format!("{cell}: {host:?} != type rule: {inferred:?}"));
            }
            if host.is_ok() != inferred.is_ok() {
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

        // Inside a `shard_map` body, a source that names devices (i.e., the placement of the local value over the
        // devices of the region, or the sole device of a region of one device) would become the target of the
        // transposed placement of an invariant local value (i.e., one that does not vary along the manual axis), which
        // would move its cotangent. The forward function moves nothing to that source, so it compiles and runs eagerly,
        // and the program of reverse mode, which keeps only the transposed placement, is rejected for its target. The
        // cotangent of a varying local value stays where it lives (refer to `PlaceOperation::transposes_in_place`), so
        // every source compiles and executes on every path for it, as does every other source for an invariant value.
        let pinned_seed = |r#type: &ArrayType| r#type.clone().with_memory(pinned);
        let moved_error = format!(
            "cannot lower `{PLACE_OPERATION_NAME}`: a value inside a manual region cannot be moved to another \
             placement",
        );
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
                        let body = |local| record_placement_error(&RefCell::new(None), session.place(local, options()));
                        let forward: Result<CompiledFunction<ArrayType, ArrayType>, _> = compile(
                            |x: CompileTracer| {
                                let options = ShardMapOptions::new();
                                shard_map(
                                    body,
                                    x,
                                    region_mesh.clone(),
                                    specification.clone(),
                                    specification.clone(),
                                    options,
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
                            let expected = match (is_rejected, path) {
                                (false, _) => Ok(()),
                                (true, "reverse") => Err(moved_error.clone()),
                                (true, _) => Ok(()),
                            };
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

                // Direct storage conversion and host constructors share the format check for empty upload lists too.
                let source_mesh = committed.mesh().clone();
                let bytes = [7u8, 9];
                let destination_type = r#type.clone().with_layout(layout.clone());
                let dense =
                    Array::from_host_buffer(&unchecked, destination_type.clone(), source_mesh.clone(), None, &bytes);
                let shard = (rank == 0).then_some((devices[0].id(), &bytes));
                let sharded = Array::from_host_shards(&unchecked, destination_type, source_mesh.clone(), None, shard);
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
    // test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_session_place_compares_equal_values_cuda_process_helper() {
        run_as_cuda_rank(|client| probe_value_comparison(client, false));
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_session_place_compares_equal_values_on_two_cuda_processes() {
        // Equal values pass the comparison, and each rank holds its shard of the value.
        let helper = "arrays::placement::tests::test_session_place_compares_equal_values_cuda_process_helper";
        let probes = run_cuda_rank_pair_probes(helper);
        assert_eq!(
            probes,
            indoc! {"
                rank 0 host value: [[1.0, 2.0]]
                rank 0 uploaded array: [[1.0, 2.0]]
                rank 0 cross-host placement: [[17.0, 18.0, 19.0, 20.0]]
                rank 1 host value: [[3.0, 4.0]]
                rank 1 uploaded array: [[3.0, 4.0]]
                rank 1 cross-host placement: [[17.0, 18.0, 19.0, 20.0]]"},
        );
    }

    // Worker of `test_session_place_rejects_different_values_on_two_cuda_processes`, which returns immediately unless
    // that test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_session_place_rejects_different_values_cuda_process_helper() {
        run_as_cuda_rank(|client| probe_value_comparison(client, true));
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_session_place_rejects_different_values_on_two_cuda_processes() {
        // Different values fail the results of both ranks alike without a hang, and the ranks stay in step.
        let helper = "arrays::placement::tests::test_session_place_rejects_different_values_cuda_process_helper";
        let probes = run_cuda_rank_pair_probes(helper);
        assert_eq!(
            probes,
            indoc! {"
                rank 0 host value: different values
                rank 0 uploaded array: different values
                rank 0 cross-host placement: [[17.0, 18.0, 19.0, 20.0]]
                rank 1 host value: different values
                rank 1 uploaded array: different values
                rank 1 cross-host placement: [[17.0, 18.0, 19.0, 20.0]]"},
        );
    }

    // Worker of `test_session_place_reports_an_invalid_leaf_on_two_cuda_processes`, which returns immediately unless
    // that test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_session_place_reports_an_invalid_leaf_cuda_process_helper() {
        run_as_cuda_rank(probe_value_comparison_with_invalid_leaf);
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_session_place_reports_an_invalid_leaf_on_two_cuda_processes() {
        // The rank whose first leaf is invalid returns its error after the one comparison of the call, the other rank
        // sees the mismatch on both of its results, and both stay in step.
        let helper = "arrays::placement::tests::test_session_place_reports_an_invalid_leaf_cuda_process_helper";
        let probes = run_cuda_rank_pair_probes(helper);
        assert_eq!(
            probes,
            indoc! {"
                rank 0 tree: invalid value
                rank 0 cross-host placement: [[17.0, 18.0, 19.0, 20.0]]
                rank 1 tree: different values, different values
                rank 1 cross-host placement: [[17.0, 18.0, 19.0, 20.0]]"},
        );
    }

    // Worker of `test_session_place_admits_a_call_with_an_invalid_leaf_on_two_cuda_processes`, which returns
    // immediately unless that test launched it as one of its two ranks.
    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_session_place_admits_a_call_with_an_invalid_leaf_cuda_process_helper() {
        run_as_cuda_rank(probe_call_admission_with_invalid_leaf);
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "requires one CUDA GPU with NCCL multi-rank support and CUDA MPS"]
    fn test_session_place_admits_a_call_with_an_invalid_leaf_on_two_cuda_processes() {
        // The first leaf is invalid on rank 0 only, and the second leaf of the same call is a cross-host move that rank
        // 1 could only complete with rank 0. Both ranks return without placing either leaf, rank 0 with its own error
        // and rank 1 with the mismatch, and both stay in step for the placement that follows.
        let helper =
            "arrays::placement::tests::test_session_place_admits_a_call_with_an_invalid_leaf_cuda_process_helper";
        let probes = run_cuda_rank_pair_probes(helper);
        let leaf = "failed to place the parameter at `$[0]`";
        let error = "`place` target sharding rank (1) does not match the input rank (0)";
        assert_eq!(
            probes,
            formatdoc! {"
                rank 0 tree: {leaf}: {error}
                rank 0 cross-host placement: [[17.0, 18.0, 19.0, 20.0]]
                rank 1 tree: different values
                rank 1 cross-host placement: [[17.0, 18.0, 19.0, 20.0]]"},
        );
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    #[ignore = "entry point of a process of a multi-process test"]
    fn test_place_core_operations_cuda_process_helper() {
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

            // Both ranks reach concrete dispatch. The rank with an invalid leaf still takes part in admission.
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
    fn test_place_core_operations_on_two_cuda_processes() {
        run_cuda_rank_pair("arrays::placement::tests::test_place_core_operations_cuda_process_helper");
    }

    #[test]
    fn test_place_operation_differentiation_keeps_saved_return_placement() {
        // A saved linearization of a placement from the first device onto the third records the first device as its
        // return placement. Rebinding that linear map at a tangent on the second device (i.e., pulling it back or
        // linearizing it again) keeps that return placement, so every cotangent goes back to the first device, as
        // pinned JAX returns it there in all three cases.
        let client = cpu_client(3);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let upload = Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(&[1f32, 2.])).unwrap();
        let observe = |array: Array| {
            let devices = array.mesh().devices().iter().map(|device| device.id()).collect::<Vec<_>>();
            (devices, array.copy_to_host().unwrap().elements::<f32>().unwrap(), array.is_committed())
        };
        let x = upload.to_placement(&session, devices[0]).unwrap();
        let t = upload.to_placement(&session, devices[1]).unwrap();
        let (primal, linear) = differentiate_at(x).linearize(|x| x.place(&devices[2], None)).unwrap();
        let returned = (vec![devices[0].id()], vec![1., 2.], true);
        let transposed = linear.program().transpose().unwrap();
        let direct = transposed.interpret_in_context(&upload.domain(), vec![primal.clone()]).unwrap();
        assert_eq!(direct.into_iter().map(observe).collect::<Vec<_>>(), vec![returned.clone()]);

        let (_, pullback) = differentiate_at(t.clone()).vjp(|z| linear.apply_in_context(&z.domain(), z)).unwrap();
        assert_eq!(observe(pullback.apply_materialized(primal).unwrap()), returned);

        let (result, second) = differentiate_at(t).linearize(|z| linear.apply_in_context(&z.domain(), z)).unwrap();
        assert_eq!(observe(result), (vec![devices[2].id()], vec![1., 2.], true));
        assert_eq!(
            second.program().to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = place [
                    inferred_source={mesh<['device'=1:auto]>, devices=[0], \
                        sharding={mesh<['device'=1:auto]>, [{}]}, memory=Device},
                    target=device(2),
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_admits_relocatable_sources_under_nested_transforms() {
        // A targetless placement of an uncommitted upload on the first device with a source on a named mesh of the
        // second device is admitted under two nested batching levels, batching around a JVP, and a JVP around
        // batching, as it is eagerly (refer to `test_place_operation_admits_relocatable_sources_on_every_path`),
        // because every transform value projects the relocatable type of the value that it wraps. Each leaves the
        // upload where it is, uncommitted, as pinned JAX does.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("other", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let source = Placement::new(mesh, Sharding::replicated(logical_mesh, 1)).unwrap();
        let upload = |shape: &[usize], values: &[f32]| {
            let r#type = ArrayType::new_static(DataType::F32, shape.to_vec());
            Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(values)).unwrap()
        };
        let observe = |array: Array| {
            let devices = array.mesh().devices().iter().map(|device| device.id()).collect::<Vec<_>>();
            (devices, array.copy_to_host().unwrap().elements::<f32>().unwrap(), array.is_committed())
        };
        let first = devices[0].id();

        let nested = batch(
            |outer| {
                Ok(batch(
                    |item| item.place(&Option::<Placement>::None, &source),
                    outer,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )?)
            },
            upload(&[2, 1, 2], &[3., 4., 5., 6.]),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        );
        assert_eq!(
            nested.map(observe).map_err(|error| error.to_string()),
            Ok((vec![first], vec![3., 4., 5., 6.], false)),
        );
        let batched_jvp = batch(
            |outer| {
                let (_, tangent) = differentiate_at(outer.clone())
                    .jvp(outer, |item| item.place(&Option::<Placement>::None, &source))?;
                Ok(tangent)
            },
            upload(&[2, 2], &[3., 4., 5., 6.]),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        );
        assert_eq!(
            batched_jvp.map(observe).map_err(|error| error.to_string()),
            Ok((vec![first], vec![3., 4., 5., 6.], false)),
        );
        let jvp_of_batch = differentiate_at(upload(&[2, 2], &[1., 2., 3., 4.]))
            .jvp(upload(&[2, 2], &[5., 6., 7., 8.]), |x| {
                Ok(batch(
                    |item| item.place(&Option::<Placement>::None, &source),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )?)
            })
            .map(|(primal, tangent)| (observe(primal), observe(tangent)))
            .map_err(|error| error.to_string());
        assert_eq!(
            jvp_of_batch,
            Ok(((vec![first], vec![1., 2., 3., 4.], false), (vec![first], vec![5., 6., 7., 8.], false))),
        );

        // The same three compositions through the `Value::Array` representation of the uploads, whose batching levels
        // (refer to `ArrayIrBatchingPolicy`) and array views (i.e., `ProjectedValue`s) project the relocatable type of
        // an array member, give the same results.
        let observe_value = |value: Value| observe(ValueProjection::<ArrayType>::into_projected(value).unwrap());
        let nested = batch(
            |outer| {
                Ok(batch(
                    |item| item.place(&Option::<Placement>::None, &source),
                    outer,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )?)
            },
            Value::Array(upload(&[2, 1, 2], &[3., 4., 5., 6.])),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        );
        let batched_jvp = batch(
            |outer| {
                let (_, tangent) = differentiate_at(outer.clone())
                    .jvp(outer, |item| item.place(&Option::<Placement>::None, &source))?;
                Ok(tangent)
            },
            Value::Array(upload(&[2, 2], &[3., 4., 5., 6.])),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        );
        let jvp_of_batch = differentiate_at(Value::Array(upload(&[2, 2], &[1., 2., 3., 4.]))).jvp(
            Value::Array(upload(&[2, 2], &[5., 6., 7., 8.])),
            |x| {
                Ok(batch(
                    |item| item.place(&Option::<Placement>::None, &source),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )?)
            },
        );
        let outcomes = vec![
            nested.map(|value| vec![observe_value(value)]).map_err(|error| error.to_string()),
            batched_jvp.map(|value| vec![observe_value(value)]).map_err(|error| error.to_string()),
            jvp_of_batch
                .map(|(primal, tangent)| vec![observe_value(primal), observe_value(tangent)])
                .map_err(|error| error.to_string()),
        ];
        assert_eq!(
            outcomes,
            vec![
                Ok(vec![(vec![first], vec![3., 4., 5., 6.], false)]),
                Ok(vec![(vec![first], vec![3., 4., 5., 6.], false)]),
                Ok(vec![(vec![first], vec![1., 2., 3., 4.], false), (vec![first], vec![5., 6., 7., 8.], false)]),
            ],
        );
    }

    #[test]
    fn test_place_pads_short_shardings() {
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

    #[test]
    fn test_array_place_pullback_uses_the_seed_on_its_own_devices() {
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let split = |name: &str, mesh_devices: &[Device]| {
            let axis = MeshAxis::new(name, mesh_devices.len(), MeshAxisType::Explicit).unwrap();
            let logical_mesh = LogicalMesh::new(vec![axis]).unwrap();
            let sharding = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded([name])]).unwrap();
            Placement::new(DeviceMesh::new(logical_mesh, mesh_devices.to_vec()).unwrap(), sharding).unwrap()
        };
        let target = split("q", &devices[..2]);
        let upload = |shape: &[usize], values: &[f32], placement: PlacementRequest| {
            let r#type = ArrayType::new_static(DataType::F32, shape.to_vec());
            let array = Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(values)).unwrap();
            array.to_placement(&session, placement).unwrap()
        };
        let observe = |array: Array| {
            let devices = array.mesh().devices().iter().map(|device| device.id()).collect::<Vec<_>>();
            (devices, array.copy_to_host().unwrap().elements::<f32>().unwrap(), array.is_committed())
        };
        let x = upload(&[4], &[1., 2., 3., 4.], devices[0].into());
        let [bare_seed, matching_seed, mismatched_seed] = [
            upload(&[4], &[5., 6., 7., 8.], devices[2].into()),
            upload(&[4], &[5., 6., 7., 8.], split("q", &devices[2..]).into()),
            upload(&[4], &[5., 6., 7., 8.], split("r", &devices[2..]).into()),
        ];
        let source = devices[0];

        // Exercise both the array operation family and its wrapper in the session operation family.
        macro_rules! check_pullbacks {
            ($differentiate:expr, $lift:expr, $into_array:expr) => {{
                let differentiate = $differentiate;
                let lift = $lift;
                let into_array = $into_array;
                let (output, pullback) = differentiate().vjp(|x| x.place(&target, &source)).unwrap();
                let output = into_array(output);
                assert_eq!(output.mesh().devices(), &devices[..2]);
                assert_eq!(output.sharding(), target.sharding());

                // An anonymous seed and the same logical sharding on other devices both pass the boundary.
                for seed in [&bare_seed, &matching_seed] {
                    let gradient = pullback.apply_materialized(lift(seed.clone())).map(into_array).map(observe);
                    assert_eq!(gradient, Ok((vec![source.id()], vec![5., 6., 7., 8.], true)));
                }
                let gradient = pullback.apply_materialized(lift(mismatched_seed.clone())).map(into_array).map(observe);
                assert_eq!(
                    gradient,
                    Err(ProgramError::MalformedProgram(format!(
                        "pullback cotangent 0 has type {} but its primal boundary requires cotangent type {}",
                        mismatched_seed.r#type(),
                        output.r#type().cotangent().unwrap(),
                    ))),
                );

                // Negation runs where the anonymous seed lives and the placement sends the result to the source.
                let (_, pullback) = differentiate().vjp(|x| x.place(&target, &source)?.neg()).unwrap();
                let gradient = pullback.apply_materialized(lift(bare_seed.clone())).map(into_array).map(observe);
                assert_eq!(gradient, Ok((vec![source.id()], vec![-5., -6., -7., -8.], true)));

                // Sine's transpose multiplies the seed by the cosine evaluated on the forward output devices.
                let (_, pullback) = differentiate().vjp(|x| x.place(&target, &source)?.sin()).unwrap();
                let gradient = pullback.apply_materialized(lift(bare_seed.clone())).map(into_array).map(observe);
                assert_eq!(
                    gradient,
                    Err(ProgramError::InvalidArgument {
                        message: "received incompatible devices for eager XLA execution: input #1 is placed on \
                                  devices [2] but input #0 is placed on devices [0, 1]"
                            .to_string(),
                    }),
                );
            }};
        }
        check_pullbacks!(|| differentiate_at(x.clone()), std::convert::identity, std::convert::identity);
        check_pullbacks!(|| session.differentiate_at(Value::Array(x.clone())), Value::Array, |value| {
            ValueProjection::<ArrayType>::into_projected(value).unwrap()
        });

        // Under `vmap`, the seed is the same for every batch item, and the gradient of every item returns to the first
        // device.
        let packed = upload(&[2, 4], &[1., 2., 3., 4., 5., 6., 7., 8.], devices[0].into());
        let batched: Result<Array, _> = batch(
            |(x, seed)| {
                let (_, pullback) = differentiate_at(x).vjp(|x| x.place(&target, None))?;
                pullback.apply_materialized(seed)
            },
            (packed, bare_seed.clone()),
            (BatchAxis::new(0), BatchAxis::replicated()),
            BatchAxis::new(0),
            None,
        );
        assert_eq!(
            batched.map(observe).map_err(|error| error.to_string()),
            Ok((vec![devices[0].id()], vec![5., 6., 7., 8., 5., 6., 7., 8.], true)),
        );

        // Transposing the pullback again places the cotangent of the inner seed on the forward destination,
        // including when that seed lived on a single device outside the mesh.
        let cotangents = [&matching_seed, &bare_seed].map(|inner_seed| -> Result<_, String> {
            let (_, pullback) = differentiate_at((x.clone(), inner_seed.clone()))
                .vjp(|(x, cotangent)| {
                    let (_, pullback) = differentiate_at(x).vjp(|x| x.place(&target, None))?;
                    pullback.apply_materialized(cotangent)
                })
                .map_err(|error| error.to_string())?;
            let seed = upload(&[4], &[9., 10., 11., 12.], devices[3].into());
            let cotangents = pullback.apply_materialized(seed).map_err(|error| error.to_string())?;
            assert_eq!(cotangents.1.sharding(), target.sharding());
            assert_eq!(shard_values(&cotangents.1), vec![vec![9., 10.], vec![11., 12.]]);
            Ok(observe(cotangents.1))
        });
        assert_eq!(
            cotangents,
            [
                Ok((vec![devices[0].id(), devices[1].id()], vec![9., 10., 11., 12.], true)),
                Ok((vec![devices[0].id(), devices[1].id()], vec![9., 10., 11., 12.], true)),
            ],
        );
    }

    #[test]
    fn test_place_operation_jvp_admits_relocatable_tangents() {
        let session = Session::builder().client(cpu_client(3)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let (primal, tangent) = place_jvp_inputs(&session, devices[1]);
        let aligned = tangent.place(&primal.placement().unwrap(), None).unwrap();
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let laid_out = aligned
            .place(&PlacementRequest::InPlace { memory: None, layout: Some(layout.clone()) }, None)
            .unwrap();
        assert_eq!(laid_out.r#type().layout(), Some(&layout));

        let target = devices[2];
        let (_, pushforward) = differentiate_at(primal.clone()).linearize(|x| x.place(&target, None)).unwrap();
        for tangent in [tangent, aligned, laid_out] {
            let (value, derivative) =
                differentiate_at(primal.clone()).jvp(tangent.clone(), |x| x.place(&target, None)).unwrap();
            assert_eq!(observe_place_jvp(&value), (target, true, vec![1., 2.]));
            assert_eq!(observe_place_jvp(&derivative), (target, true, vec![3., 4.]));
            assert_eq!(observe_place_jvp(&pushforward.apply(tangent).unwrap()), (target, true, vec![3., 4.]));
        }
    }

    #[test]
    fn test_place_operation_jvp_moves_committed_inputs_to_an_explicit_device() {
        let session = Session::builder().client(cpu_client(3)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let primal = upload_place_jvp_vector(&session, &[1., 2.]).place(&devices[1], None).unwrap();
        let tangent = upload_place_jvp_vector(&session, &[3., 4.]).place(&devices[0], None).unwrap();
        let target = devices[2];
        let (value, derivative) =
            differentiate_at(primal.clone()).jvp(tangent.clone(), |x| x.place(&target, None)).unwrap();
        let (_, pushforward) = differentiate_at(primal).linearize(|x| x.place(&target, None)).unwrap();
        assert_eq!(observe_place_jvp(&value), (target, true, vec![1., 2.]));
        assert_eq!(observe_place_jvp(&derivative), (target, true, vec![3., 4.]));
        assert_eq!(observe_place_jvp(&pushforward.apply(tangent).unwrap()), (target, true, vec![3., 4.]));
    }

    #[test]
    fn test_place_operation_jvp_on_explicit_device_feeds_elementwise_arithmetic() {
        let session = Session::builder().client(cpu_client(3)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let (primal, tangent) = place_jvp_inputs(&session, devices[1]);
        let target = devices[2];
        let (value, derivative) = differentiate_at(primal)
            .jvp(tangent, |x| {
                let placed = x.place(&target, None)?;
                placed.add(&placed)
            })
            .unwrap();
        assert_eq!(observe_place_jvp(&value), (target, true, vec![2., 4.]));
        assert_eq!(observe_place_jvp(&derivative), (target, true, vec![6., 8.]));
    }

    #[test]
    fn test_place_operation_jvp_preserves_devices_for_memory_targets() {
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let primal = upload_place_jvp_vector(&session, &[1., 2.]).place(&devices[1], None).unwrap();
        let tangent = upload_place_jvp_vector(&session, &[3., 4.]);
        let committed = tangent.place(&devices[0], None).unwrap();
        let pinned = Memory::Host { pinned: true };
        let (_, pushforward) = differentiate_at(primal.clone()).linearize(|x| x.place(&pinned, None)).unwrap();

        for tangent in [tangent, committed] {
            let (value, derivative) =
                differentiate_at(primal.clone()).jvp(tangent.clone(), |x| x.place(&pinned, None)).unwrap();
            let saved = pushforward.apply(tangent).unwrap();
            assert_eq!(value.r#type().memory(), pinned);
            assert_eq!(observe_place_jvp(&value), (devices[1], true, vec![1., 2.]));
            for result in [derivative, saved] {
                assert_eq!(result.r#type().memory(), pinned);
                assert_eq!(observe_place_jvp(&result), (devices[0], true, vec![3., 4.]));
            }
        }
    }

    #[test]
    fn test_place_operation_jvp_keeps_memory_placements_for_independent_arithmetic() {
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let primal = upload_place_jvp_vector(&session, &[1., 2.]).place(&devices[1], None).unwrap();
        let tangent = upload_place_jvp_vector(&session, &[3., 4.]);
        let (value, derivative) = differentiate_at(primal.clone())
            .jvp(tangent.clone(), |x| {
                let placed = x.place(&Memory::Device, None)?;
                placed.add(&placed)
            })
            .unwrap();
        assert_eq!(observe_place_jvp(&value), (devices[1], true, vec![2., 4.]));
        assert_eq!(observe_place_jvp(&derivative), (devices[0], true, vec![6., 8.]));
        let (_, pushforward) = differentiate_at(primal)
            .linearize(|x| {
                let placed = x.place(&Memory::Device, None)?;
                placed.add(&placed)
            })
            .unwrap();
        assert_eq!(observe_place_jvp(&pushforward.apply(tangent).unwrap()), (devices[0], true, vec![6., 8.]));
    }

    #[test]
    fn test_place_operation_pushforward_rejects_incompatible_devices_after_memory_placement() {
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let (primal, tangent) = place_jvp_inputs(&session, devices[1]);
        let (_, pushforward) = differentiate_at(primal).linearize(|x| x.place(&Memory::Device, None)?.sin()).unwrap();
        assert!(matches!(
            pushforward.apply(tangent),
            Err(ProgramError::InvalidArgument { message })
                if message == "received incompatible devices for eager XLA execution: input #1 is placed on \
                               devices [0] but input #0 is placed on devices [1]",
        ));
    }

    #[test]
    fn test_place_operation_jvp_rejects_anonymous_tangent_outputs_for_named_memory_placement() {
        let session = Session::builder().client(cpu_client(2)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let (primal, tangent) = place_jvp_inputs(&session, devices[1]);
        let pinned = Memory::Host { pinned: true };
        let (_, pushforward) = differentiate_at(primal.clone()).linearize(|x| x.place(&pinned, None)).unwrap();
        assert!(matches!(
            differentiate_at(primal).jvp(tangent.clone(), |x| x.place(&pinned, None)),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "tangent type f32[2]@Host[Pinned] does not match type \
                               f32[2][sharding={mesh<['other'=1:explicit]>, [{}]}]@Host[Pinned] required by primal type \
                               f32[2][sharding={mesh<['other'=1:explicit]>, [{}]}]@Host[Pinned]",
        ));
        assert!(matches!(
            pushforward.apply(tangent),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "encountered output type f32[2]@Host[Pinned] which is incompatible with the region's \
                               declared type f32[2][sharding={mesh<['other'=1:explicit]>, [{}]}]@Host[Pinned]",
        ));
    }

    #[test]
    fn test_place_operation_jvp_bare_device_target_uses_default_memory() {
        let session = Session::builder().client(cpu_client(3)).build().unwrap();
        let devices = session.addressable_devices().unwrap();
        let primal = upload_place_jvp_vector(&session, &[1., 2.]).place(&devices[1], None).unwrap();
        let tangent = upload_place_jvp_vector(&session, &[3., 4.]);
        let pinned = Memory::Host { pinned: true };
        let primal = primal.place(&pinned, None).unwrap();
        let tangent = tangent.place(&pinned, None).unwrap();
        let target = devices[2];
        let (value, derivative) =
            differentiate_at(primal.clone()).jvp(tangent.clone(), |x| x.place(&target, None)).unwrap();
        let (_, pushforward) = differentiate_at(primal).linearize(|x| x.place(&target, None)).unwrap();
        let saved = pushforward.apply(tangent).unwrap();
        for output in [value, derivative, saved] {
            assert_eq!(output.r#type().memory(), Memory::Device);
            assert_eq!(output.mesh().devices(), &[target]);
            assert!(output.is_committed());
        }
    }

    #[test]
    fn test_array_place_jvp_bare_device_target_completes_reduction_state() {
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
    fn test_array_place_partial_evaluation_uses_declared_reduction_state() {
        let session = Session::builder().client(cpu_client(1)).build().unwrap();
        let device = session.addressable_devices().unwrap()[0];
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![device]).unwrap();
        let session = session.with_device_mesh(mesh);
        let sharding = Sharding::replicated(logical_mesh, 0).with_reduced_axes(["x"]).unwrap();
        let input = Array::from_host_buffer(
            &session,
            ArrayType::scalar(DataType::F32),
            None,
            Some(sharding.clone()),
            values_to_bytes(&[2.0f32]),
        )
        .unwrap();
        assert_eq!(input.sharding(), &sharding);
        assert_eq!(input.r#type().sharding(), Some(&sharding.without_auto_axes()));
        let target = input.placement().unwrap();
        let eager = input.place(&target, None).unwrap();
        assert_eq!(eager.placement(), Some(target.clone()));
        assert_eq!(shard_values(&eager), vec![vec![2.0f32]]);

        // The shared fold validates the staged type before it can execute a known placement. That type omits the
        // physical reduction over the automatic axis, so the resolved target cannot restate that reduction here.
        let error = ProgramError::Type(TypeError::invalid(
            "`place` cannot change reduction state; use a reduction (or the corresponding collective) instead",
        ));
        let context = ryft_core::PartialEvaluationContext::new(CoreValue::domain(&input));
        let known = PartialTracer::new(context.clone(), context.known_value(input.clone()));
        assert_eq!(known.place(&target, None).unwrap_err(), error);

        let context = ryft_core::PartialEvaluationContext::new(session);
        let known = PartialTracer::new(context.clone(), context.known_value(Value::Array(input)));
        assert_eq!(known.place(&target, None).unwrap_err(), error);
    }

    #[test]
    fn test_place_operation_admits_relocatable_known_partial_values() {
        // A known partial value keeps the relocatability of the uncommitted upload that it carries, so a targetless
        // placement with a source on a named mesh of the second device admits it through `Place`, direct
        // interpretation, and batching, and leaves it where it is, uncommitted, as it does for the upload itself. An
        // unknown partial value with a declared sharding names a residual program variable, which is staged,
        // so it is validated strictly against that declaration.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("other", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let source = Placement::new(mesh, Sharding::replicated(logical_mesh, 1)).unwrap();
        let upload = |shape: &[usize], values: &[f32]| {
            let r#type = ArrayType::new_static(DataType::F32, shape.to_vec());
            Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(values)).unwrap()
        };
        let observe = |value: PartialTracer<ryft_core::ProjectedContext<Session, ArrayType>>| {
            let array = value.value().unwrap().as_known().unwrap().clone();
            let devices = array.mesh().devices().iter().map(|device| device.id()).collect::<Vec<_>>();
            (devices, array.copy_to_host().unwrap().elements::<f32>().unwrap(), array.is_committed())
        };
        let array = upload(&[2], &[1., 2.]);
        let context = ryft_core::PartialEvaluationContext::new(CoreValue::domain(&array));
        let known = PartialTracer::new(context.clone(), context.known_value(array.clone()));
        let packed = PartialTracer::new(context.clone(), context.known_value(upload(&[1, 2], &[1., 2.])));
        let operation = PlaceOperation::new(None, source.clone());
        let outcomes = vec![
            known.place(&Option::<Placement>::None, &source).map(observe).map_err(|error| error.to_string()),
            InterpretableOperation::interpret(&operation, &context, &EmptyRegionDriver, &[known.clone()])
                .map(|mut outputs| observe(outputs.remove(0)))
                .map_err(|error| error.to_string()),
            batch(
                |item| item.place(&Option::<Placement>::None, &source),
                packed,
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .map(observe)
            .map_err(|error| error.to_string()),
        ];
        let accepted = Ok((vec![devices[0].id()], vec![1., 2.], false));
        assert_eq!(outcomes, vec![accepted.clone(), accepted.clone(), accepted]);

        let context = ryft_core::PartialEvaluationContext::new(CoreValue::domain(&array));
        let initial = LogicalMesh::new(vec![MeshAxis::new("initial", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let declaration = array.r#type().into_owned().with_sharding(Sharding::replicated(initial, 1)).unwrap();
        let unknown = PartialTracer::new(context.clone(), context.unknown_input(declaration.clone(), 0));
        let placed = unknown
            .place(&Option::<Placement>::None, &source)
            .map_err(|error| error.to_string())
            .and_then(|value| value.value().map(|_| ()).map_err(|error| error.to_string()));
        assert_eq!(
            placed,
            Err(format!(
                "`{PLACE_OPERATION_NAME}` source sharding `{}` does not match the input type `{}`",
                source.sharding(),
                declaration,
            )),
        );
    }

    #[test]
    fn test_array_place_folds_and_residualizes_relocatable_values() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let second = Device::from_pjrt(&client.addressable_devices().unwrap()[1]).unwrap();
        let named_mesh =
            |axis| LogicalMesh::new(vec![MeshAxis::new(axis, 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let initial = Sharding::replicated(named_mesh("first"), 1);
        let source_mesh = named_mesh("second");
        let source = Placement::new(
            DeviceMesh::new(source_mesh.clone(), vec![second]).unwrap(),
            Sharding::replicated(source_mesh, 1),
        )
        .unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let values = [1f32, 2.];
        let anonymous =
            Array::from_host_buffer(&session, r#type.clone(), None, None, values_to_bytes(&values)).unwrap();
        let named =
            Array::from_host_buffer(&session, r#type.clone(), None, Some(initial), values_to_bytes(&values)).unwrap();
        let copy = PlaceOperation::new(None, source).with_may_alias(false);
        assert!(!named.is_committed());
        assert!(copy.infer_output_types(&[named.r#type().into_owned()], &[]).is_err());

        for input in [anonymous, named] {
            let domain = CoreValue::domain(&input);
            let context = ryft_core::PartialEvaluationContext::new(domain.clone());
            let known = PartialTracer::new(context.clone(), context.known_value(input.clone()));
            let [output] = InterpretableOperation::interpret(
                &copy.clone().with_may_alias(true),
                &context,
                &EmptyRegionDriver,
                &[known],
            )
            .unwrap()
            .try_into()
            .unwrap();
            assert_eq!(output.value().unwrap().declared_type(), None);
            assert_eq!(output.r#type(), input.r#type());
            assert_eq!(output.value().unwrap().as_known().unwrap().placement(), input.placement());

            let context = ryft_core::PartialEvaluationContext::new(domain.clone()).with_allow_copy_folding(false);
            let known = PartialTracer::new(context.clone(), context.known_value(input.clone()));
            let [output] = InterpretableOperation::interpret(&copy, &context, &EmptyRegionDriver, &[known])
                .unwrap()
                .try_into()
                .unwrap();
            let residual = output.value().unwrap().clone();
            assert!(residual.is_unknown());
            drop(output);
            let evaluation = context.into_evaluation(vec![residual]).unwrap();
            assert_eq!(
                evaluation.program().to_string(),
                format!("lambda %0:{0} .\nlet %1:{0} = {copy} %0\nin (%1)", r#type),
            );

            let first_copy = evaluation.interpret(&domain, &[]).unwrap().remove(0);
            let second_copy = evaluation.interpret(&domain, &[]).unwrap().remove(0);
            assert_eq!(first_copy.mesh(), input.mesh());
            assert_eq!(second_copy.mesh(), input.mesh());
            assert_eq!(shard_values(&first_copy), vec![values.to_vec()]);
            assert_eq!(shard_values(&second_copy), vec![values.to_vec()]);
            let input_addresses = native_buffer_addresses(&input);
            let first_addresses = native_buffer_addresses(&first_copy);
            let second_addresses = native_buffer_addresses(&second_copy);
            assert_ne!(first_addresses, input_addresses);
            assert_ne!(second_addresses, input_addresses);
            assert_ne!(first_addresses, second_addresses);

            // The same rule runs through the projected array member of the full session family.
            let context = ryft_core::PartialEvaluationContext::new(session.clone()).with_allow_copy_folding(false);
            let known = PartialTracer::new(context.clone(), context.known_value(Value::Array(input.clone())));
            let [output] = context.bind(copy.clone(), Vec::new(), &[known]).unwrap().try_into().unwrap();
            let residual = output.value().unwrap().clone();
            assert!(residual.is_unknown());
            drop(output);
            let evaluation = context.into_evaluation(vec![residual]).unwrap();
            assert_eq!(
                evaluation.program().to_string(),
                format!("lambda %0:{0} .\nlet %1:{0} = {copy} %0\nin (%1)", r#type),
            );
            let output = evaluation.interpret(&session, &[]).unwrap().remove(0);
            let output = ValueProjection::<ArrayType>::into_projected(output).unwrap();
            assert_eq!(output.mesh(), input.mesh());
            assert_eq!(shard_values(&output), vec![values.to_vec()]);
            assert_ne!(native_buffer_addresses(&output), input_addresses);
        }
    }
}
