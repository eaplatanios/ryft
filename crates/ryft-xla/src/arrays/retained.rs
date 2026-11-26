use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

use ryft_core::{
    ArrayType, Device, DeviceMesh, Parameter, ProgramError, Sharding, Type, TypeError, TypeIdentityRenaming, Typed,
};
use ryft_macros::Parameter;
use ryft_pjrt::{Client, ExecutionFence};

use crate::arrays::host::validate_mesh_devices;
use crate::arrays::metadata::ArrayMetadata;
use crate::arrays::shards::{ArrayShard, ShardContribution};
use crate::arrays::transfers::PlacementPlanningView;
use crate::arrays::{Array, ArrayError, ArrayParts};
use crate::{Error, Session};

/// Storage of an [`Array`] that is kept alive without the [`Session`] of that array, so that staged
/// programs can hold it as a constant. A trace whose capture table cannot hold runtime arrays (e.g., the body of a
/// [`rematerialize`](ryft_core::rematerialize) call or of a [`composite`](ryft_core::composite) and its rules, which
/// are traced over constants alone) still needs to refer to the runtime arrays that a compiled function it calls has
/// captured. Importing such an array through [`ThroughSession`](crate::ThroughSession) retains it and stages it as an
/// immediate [`Constant::Session`](crate::Constant::Session), which can appear in any region of any program, including
/// lazily traced derivative rules. Lifting that constant in a [`Session`] turns it back into an [`Array`], and lowering
/// turns every distinct retained array of a program into a hidden executable argument that follows the captures of the
/// program, so that compiled executables depend only on the array's type and never on its data.
///
/// # Differences from [`Array`]
///
/// A retained array refers to the same device buffers as the array that it was created from, but it is not a value
/// that operations can be applied to:
///
///   - **No Session:** An [`Array`] carries the [`Session`] that it belongs to, which is what lets operations on it
///     find their execution context. A retained array carries none, so it must first be turned back into an array
///     by lifting its constant in a session, which is also the session that the new array belongs to.
///   - **Usable as a Key:** A retained array implements [`Eq`] and [`Hash`] by its storage identity, declared type,
///     and commitment. Retaining the same array twice yields equal constants. Handles with different commitment
///     remain distinct because an eager operation whose regions use the constant (e.g., an eager `condition`) runs
///     where a committed constant is, as it runs where a committed operand is, whereas a call of a compiled function
///     never runs where the arrays that its callee captures are, nor commits its results because of them, committed or
///     not. Commitment is therefore part of the identity of a program that holds the constant and of its cache keys,
///     so two programs that differ only in the commitment of a constant compile separately, which costs compilations
///     but never changes results.
///
/// The first difference is what makes it safe to store in a program. Programs live in the caches of a session, so
/// a program that held an [`Array`] would keep alive the session whose caches keep the program alive, and neither
/// would ever be released, including the device buffers.
///
/// # Logical Identity
///
/// Two retained arrays are equal when they retain the same logical array with the same declared type and the same
/// commitment. The same array imported twice into one function with one type is one constant, and therefore one hidden
/// argument, whereas the same array declared with two types (e.g., imported by a function that is compiled for a mesh
/// of another topology than the mesh of the array, which declares it over its own mesh, and by one that is not) is two
/// constants, each of which its executable receives with its own type. The declared type depends on the type and the
/// mesh topology of the array alone, which every process shares. Lowering passes each distinct retained array of a
/// program as one hidden argument of its executable, and the identity of a program and its cache keys include its
/// constants, so every process of a distributed program must decide alike whether two retained arrays are equal. The
/// logical identity of an array is therefore decided by the operation that produced it, never by whether this process
/// held buffers to copy or to reuse, and every function that returns an array from another array follows these rules
/// on every process:
///
///   - **Same Array:** A clone, a change of the session of an array (e.g., [`Session::place`] of an [`Array`] without
///     a destination and without a forced copy), and a placement whose destination is the mesh, the sharding, the
///     memory, and the layout that the array already has, including an in-place destination that names its current
///     storage, return the array itself, unless the array has locally assigned devices and the destination names
///     devices. Whether a destination is the placement of an array reads only its type and placement metadata, which
///     every process shares for an array whose devices it did not choose on its own. A change of commitment keeps the
///     array as well, and retaining it then gives another retained array, because commitment is part of the identity
///     of a retained array (refer to [`RetainedArray`]).
///   - **New Array:** A forced copy (i.e., a placement with `may_alias` set to `false` of [`Session::place`] or of a
///     staged [`PlaceOperation`](ryft_core::PlaceOperation)), also on a process that holds no buffer of its input or
///     holds every one of them alone, a placement of an array with locally assigned devices on a destination that
///     names devices, also on a process whose devices already hold that destination, every other placement (e.g., one
///     that changes the devices, the sharding, the memory, or the layout, whether it shares, copies, transfers, or
///     uploads buffers, and on a process that holds no buffer of its result), the move of an input to the devices that
///     an operation runs on, and an array that a retained array is turned back into ([`RetainedArray::to_array`]) are
///     new arrays.
///   - **New Array with Shared Placements:** An array with an additional execution dependency (i.e., one that also
///     waits for the completion of another execution) has its own capture identity and shares placements of the
///     underlying buffers. Each capture observes the completion of its own producer.
///
/// Donation never changes these rules, because it only consumes the input. An operation whose result is a new array
/// but that would return the storage of its input on some process gives that result storage of its own instead, on
/// every process.
///
/// # Ownership
///
/// A retained array owns its shards, and its clones share them. The programs, regions, derivative rules, and caches
/// that hold one therefore keep the device buffers alive, and the buffers are released together with the last value
/// that refers to them. Any session of the PJRT client that owns the buffers can turn a retained array back into an
/// [`Array`]. The array that was retained and every retention of it share its copies on other devices, which stay
/// alive as long as any of them does, even after the functions that needed those copies are gone.
#[derive(Clone, Parameter)]
pub struct RetainedArray {
    /// [`ArrayType`] that this retained array declares as a constant of a program, which is what its [`Typed`]
    /// implementation returns. It is the type in [`Self::metadata`] until [`Self::rename_type_identities`] renames
    /// the identities in it (i.e., the dimension variables of a bounded dynamic shape), which changes only this field.
    declared_type: ArrayType,

    /// Type and runtime placement of the [`Array`] that was retained, shared with it. It never changes, and the
    /// arrays that [`Self::to_array`] creates share it as well.
    metadata: Arc<ArrayMetadata>,

    /// Identity of the storage of the [`Array`] that was retained, which is the same for every retention of that
    /// array. It keeps the address of that storage reserved without keeping the storage alive.
    identity: Weak<ArrayParts>,

    /// [`ArrayShard`]s of the retained array, which refer to its device buffers.
    shards: Arc<[ArrayShard]>,

    /// Whole-execution completion fence of the launch that produced the retained array, if any.
    execution_fence: Option<ExecutionFence>,

    /// Completion fence of the cross-host sends of the placement that produced the retained array, if any.
    /// Arrays that [`Self::to_array`] creates share it with the retained array.
    send_fence: Option<ExecutionFence>,

    /// Whether this retained array is committed to its devices (refer to [`Array::is_committed`]). It is part
    /// of the identity of a constant and is preserved when [`Self::to_array`] reconstructs the array.
    committed: bool,

    /// Whether the process chose the devices of the retained array on its own, which the arrays that [`Self::to_array`]
    /// creates keep. Local assignment is excluded from the identity of this retained array. Refer to
    /// [`Array::is_locally_assigned`] for more information.
    locally_assigned: bool,

    /// PJRT [`Client`] that owns the buffers of the retained array.
    client: &'static Client<'static>,

    /// Cache of the placements of this array on other devices, keyed by the destination mesh and sharding and by
    /// whether the source is locally assigned (which decides whether a transfer communicates between processes).
    /// [`Self::placed`] fills it, so that a function that receives the array on other devices places it once rather
    /// than once for each lowering, execution, or interpretation. Every retention of the same [`Array`] and their
    /// clones share this cache, which lives as long as any of them, and a handle without storage (i.e., one for which
    /// [`Self::without_storage`] returns `true`) has an empty one of its own. It takes no part in the identity of the
    /// retained array.
    ///
    /// Each entry is one placement attempt. The first caller that misses records the attempt before it places
    /// anything, and every other caller of that entry waits for it rather than submitting its own. An attempt keeps
    /// its outcome for the lifetime of the cache, except for a failure of a placement that cannot communicate, which
    /// is removed (refer to [`Self::placed`]).
    ///
    /// The lock of this cache is a leaf lock. It is held only to look up, record, or remove an attempt, or to copy
    /// out a completed one (refer to [`Self::completed_placement_on`]), and never while an attempt is awaited, while
    /// readiness is queried, or while another lock is taken. A running attempt holds only its own once-cell while it
    /// compiles and submits work, and it never waits for another attempt of a retained array. Every critical section
    /// leaves the cache valid, so a poisoned lock is recovered rather than propagated.
    placements: Arc<PlacementIndex>,
}

impl RetainedArray {
    /// Retains `array` without its session. Its device buffers stay alive for as long as the returned value or any of
    /// its clones exists. Retaining one array more than once returns equal values, so that repeatedly importing the
    /// same captured array (e.g., the weights that a compiled function captures) stages the same constant every time,
    /// and every retention shares the placements of the array.
    #[inline]
    pub fn new(array: &Array) -> Self {
        Self {
            declared_type: array.parts.metadata.r#type().clone(),
            metadata: Arc::clone(&array.parts.metadata),
            identity: Arc::downgrade(&array.parts),
            shards: Arc::from(array.parts.shards.as_slice()),
            execution_fence: array.execution_fence.clone(),
            send_fence: array.send_fence.clone(),
            committed: array.is_committed(),
            locally_assigned: array.is_locally_assigned(),
            client: array.client(),
            placements: Arc::clone(array.parts.placements.get_or_init(Arc::default)),
        }
    }

    /// Returns the runtime [`Sharding`] of the retained array, including placement over auto mesh axes that its
    /// declared [`ArrayType`] does not record.
    #[inline]
    pub fn sharding(&self) -> &Sharding {
        self.metadata.sharding()
    }

    /// Returns the concrete [`DeviceMesh`] of the retained array.
    #[inline]
    pub fn mesh(&self) -> &DeviceMesh {
        self.metadata.mesh()
    }

    /// Returns whether the retained array is [committed to its devices](Array::is_committed).
    #[inline]
    pub fn is_committed(&self) -> bool {
        self.committed
    }

    /// Returns whether the process chose the devices of the retained array on its own (refer to
    /// [`Array::is_locally_assigned`]).
    #[inline]
    pub(crate) fn is_locally_assigned(&self) -> bool {
        self.locally_assigned
    }

    /// Returns a new [`Array`] that refers to the device buffers of this retained array and belongs to `session`.
    /// The array keeps the commitment and the local assignment of this retained array.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::Type`] if the buffers belong to a different PJRT client, or a backend error if
    /// the session cannot validate the retained array's devices and process assignments against its inventory.
    pub fn to_array(&self, session: &Session) -> Result<Array, ProgramError> {
        self.validate_client(session.client())?;
        validate_mesh_devices(session, self.mesh())?;

        // The new array has storage of its own, and its shards refer to the same device buffers.
        // The parts need no validation because they come from an array that was already validated.
        Ok(Array {
            parts: Arc::new(ArrayParts::new(session, Arc::clone(&self.metadata), self.shards.to_vec())),
            execution_fence: self.execution_fence.clone(),
            send_fence: self.send_fence.clone(),
            session: session.clone(),
            committed: self.committed,
            locally_assigned: self.locally_assigned,
        })
    }

    /// Returns this retained array with the identities of its type renamed by `renaming`.
    #[inline]
    pub fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayType as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        Ok(Self { declared_type: self.declared_type.rename_identities(renaming)?, ..self.clone() })
    }

    /// Returns this retained array declared with `declared_type`, which must describe the same array. The declared
    /// type is part of the identity of a retained array, so two retentions of one array with different declared types
    /// are different constants. A function that is compiled for a mesh of another topology than the mesh of an array
    /// that it imports declares that array with the type with which the function receives it on its own mesh (refer
    /// to `imported_constant`). Lowering binds its hidden argument slot to the array placed on that mesh.
    #[inline]
    pub(crate) fn with_declared_type(&self, declared_type: ArrayType) -> Self {
        Self { declared_type, ..self.clone() }
    }

    /// Returns a handle of this retained array that is equal to it (refer to its [`PartialEq`] implementation) but
    /// keeps none of its buffers or placements alive, for keys of caches that must not keep a constant alive after the
    /// programs that hold it are released. The weak identity of the storage keeps its address reserved, so a key with
    /// such a handle never matches a constant of another array. It cannot be turned back into an array.
    pub(crate) fn without_storage(&self) -> Self {
        Self {
            shards: Arc::from([]),
            execution_fence: None,
            send_fence: None,
            placements: Arc::default(),
            ..self.clone()
        }
    }

    /// Returns a retained copy of the array that `place` puts on `mesh` as `sharding`. Every retention of the source
    /// array and their clones share the physical placement, including clones with a different declared type. The
    /// placement records the type and commitment of the array that `place` returns. Each source keeps its own declared
    /// type and identity for use in the program. The source passed to `place` carries only buffer readiness, so the
    /// attempt records completion of the placement itself. Each returned handle joins that completion with the
    /// requesting source's execution fence. An earlier placement's sends belong to that placement.
    ///
    /// Exactly one `place` runs for each source ownership, mesh, and sharding on every process, whatever the
    /// interleaving of the threads that request it and however many retentions of the array request it. A placement may
    /// communicate (e.g., a cross-host transfer, which PJRT pairs between processes by the order of submission), so a
    /// transfer that one process submits twice and its peer once would wait forever or pair with later work. The first
    /// caller that misses records an attempt before it places anything, every other caller waits for that attempt
    /// without holding any lock that `place` takes, and placements of other arrays, meshes, or shardings proceed
    /// independently. The placed copy is kept until the array and every retention of it are gone.
    ///
    /// Whether a placement can communicate is decided alike on every process (i.e., [`Self::placement_communicates`]).
    /// A key of a placement that can communicate has one attempt for the lifetime of the array on every process, so
    /// its outcome is kept, failure included. This is the exception to the rule that no cache of this crate keeps an
    /// error, because the outcome is local while the work communicates: a placement can succeed on the process that
    /// sends and fail on the process that receives (e.g., in its copy to the requested memory after the receive), and
    /// a retry by that process alone would submit a receive that no send pairs with. Such a failed attempt is therefore
    /// reported to every later caller, and the array must be created again (on every process) to retry it. A failed
    /// attempt of a placement that cannot communicate is reported to the callers that waited for it and then removed,
    /// so that a later request runs `place` again. A `place` that panics may already have submitted a transfer, so its
    /// attempt records a failure as well, and the panic then continues on the thread that ran it. A failure of the
    /// placed array after a successful submission (e.g., of its transfer) is reported by that array itself (i.e., by
    /// its readiness and its readbacks), which stays cached for the same reason. `place` must not request a placement
    /// of this array (i.e., its source), which it would wait for forever, but it may look for a completed placement of
    /// it on other meshes or shardings (refer to [`Self::completed_placement_on`]), which never waits and holds the
    /// lock of the placements only to copy out their handles, since `place` runs without that lock. A caller therefore
    /// consults its exact key before any other source is chosen, and a caller that finds its key never waits for other
    /// work.
    ///
    /// # Errors
    ///
    /// Returns the error of `place` to every caller of a failed attempt of a placement that cannot communicate, and
    /// [`ArrayError::FailedCapturePlacement`] with the error of `place` as its source to every caller of a failed
    /// attempt of one that can, the caller that ran `place` included, which says that the capture must be created
    /// again to retry it.
    pub(crate) fn placed(
        &self,
        mesh: &DeviceMesh,
        sharding: &Sharding,
        place: impl FnOnce(Self) -> Result<Array, Error>,
    ) -> Result<Self, Error> {
        let communicates = self.placement_communicates(mesh, sharding);
        let key = || (mesh.clone(), sharding.clone(), self.locally_assigned);
        let device_key = || (mesh.devices().to_vec(), self.locally_assigned);
        let attempt = {
            let mut placements = self.placements.lock().unwrap_or_else(PoisonError::into_inner);
            let PlacementEntries { attempts, attempts_by_devices } = &mut *placements;
            let attempt = attempts.entry(key()).or_insert_with(|| {
                let attempt = PlacementAttempt::default();
                attempts_by_devices.entry(device_key()).or_default().push(Arc::clone(&attempt));
                attempt
            });
            Arc::clone(attempt)
        };

        // The once-cell runs `place` on the first thread that reaches it and blocks every other one until it finishes.
        // A panic is caught so that the cell records a failure rather than staying empty for another `place`. Which
        // caller runs `place` depends on the interleaving, so the cell records the failure as every caller receives it.
        let mut panic = None;
        let placed = attempt
            .get_or_init(|| {
                let source = Self { execution_fence: None, send_fence: None, ..self.clone() };
                let placed = match catch_unwind(AssertUnwindSafe(|| place(source))) {
                    Ok(placed) => placed,
                    Err(payload) => {
                        panic = Some(payload);
                        let message = "the placement panicked".to_string();
                        Err(Error::from(ProgramError::InvalidArgument { message }))
                    }
                };
                placed.map(|array| RetainedArray::new(&array)).map_err(|error| {
                    if !communicates {
                        return error;
                    }
                    let source = Arc::new(error);
                    Error::from(ArrayError::FailedCapturePlacement { sharding: sharding.clone(), source })
                })
            })
            .clone();

        // A failed attempt of a placement that cannot communicate is removed once it has completed, unless a later
        // request already replaced it, so that the next request places the array again.
        if !communicates && placed.is_err() {
            let mut placements = self.placements.lock().unwrap_or_else(PoisonError::into_inner);
            let key = key();
            if placements.attempts.get(&key).is_some_and(|recorded| Arc::ptr_eq(recorded, &attempt)) {
                placements.attempts.remove(&key);
                let device_key = device_key();
                if let Some(recorded) = placements.attempts_by_devices.get_mut(&device_key) {
                    recorded.retain(|recorded| !Arc::ptr_eq(recorded, &attempt));
                    if recorded.is_empty() {
                        placements.attempts_by_devices.remove(&device_key);
                    }
                }
            }
        }

        if let Some(payload) = panic {
            resume_unwind(payload);
        }

        placed.map(|mut placed| {
            if let Some(source_fence) = &self.execution_fence {
                placed.execution_fence =
                    Some(ExecutionFence::join(std::iter::once(source_fence.clone()).chain(placed.execution_fence)));
            }
            placed
        })
    }

    /// Returns whether placing this retained array on `mesh` with `sharding` can involve other processes (i.e.,
    /// whether it may send or receive shards across processes, or run a collective with them). It returns `false`
    /// when this process can carry the placement out on its own, which is the case when the array has locally
    /// assigned devices, when it is a structural zero (which has no buffers), when it already has that placement,
    /// or when its devices and those of `mesh` all belong to one process. The answer depends only on the type and
    /// the placement of the array and on the destination, never on its buffers or on a client, so every process of a
    /// distributed run computes the same answer. It errs on the side of `true`, so a placement for which it returns
    /// `true` may end up communicating nothing.
    #[inline]
    pub(crate) fn placement_communicates(&self, mesh: &DeviceMesh, sharding: &Sharding) -> bool {
        PlacementPlanningView::new(&self.metadata, self.shards.iter().map(ArrayShard::descriptor)).communicates(
            self.locally_assigned,
            mesh,
            sharding,
        )
    }

    /// Returns a completed placement of this retained array on `devices`, if one was recorded (refer to
    /// [`Self::placed`]), so that a caller can place from that copy rather than from this array. The candidate is the
    /// first recorded placement, in the order in which they were recorded, that succeeded, whose devices are `devices`
    /// in that order, whose memory and declared layout are those of this array, and which `accepts`.
    ///
    /// The candidate must keep the memory and the declared layout of this array, because a placement keeps the memory
    /// of its source and derives its layout from the layout of its source and the destination alone. A placement from
    /// such a copy therefore has the type that a placement from this array has, whereas a copy that dropped the layout
    /// (e.g., a replicated copy of column shards) would drop it from every placement made from it as well.
    ///
    /// This function never waits. It skips attempts that are still running, and it returns [`None`] unless the work of
    /// the candidate has already completed without an error, because a submitted placement can still fail and a copy
    /// made from it would share its failed buffers. Only that first candidate is queried, so that a pending one does
    /// not make every later lookup query the readiness of every earlier candidate. Returning [`None`] is always
    /// correct, because the caller then places from this array itself. The lock of the placements is held only
    /// to copy out one candidate at a time, and it is released before readiness is queried.
    pub(crate) fn completed_placement_on(&self, devices: &[Device], accepts: impl Fn(&Self) -> bool) -> Option<Self> {
        let r#type = self.metadata.r#type();
        let keeps_storage = |placed: &Self| {
            let placed_type = placed.metadata.r#type();
            placed_type.memory() == r#type.memory() && placed_type.layout() == r#type.layout()
        };

        let key = (devices.to_vec(), self.locally_assigned);
        let candidate = (0..)
            .map_while(|position| {
                let placements = self.placements.lock().unwrap_or_else(PoisonError::into_inner);
                placements.attempts_by_devices.get(&key)?.get(position).map(Arc::clone)
            })
            .find_map(|attempt| match attempt.get() {
                Some(Ok(placed)) if keeps_storage(placed) && accepts(placed) => Some(placed.clone()),
                _ => None,
            })?;

        // The work that produced the candidate must have completed without an error, which is checked without waiting
        // for it. Pending work and a failed readiness query count as not completed.
        let fence_completed =
            |fence: &Option<ExecutionFence>| fence.as_ref().is_none_or(|fence| matches!(fence.is_ready(), Ok(true)));
        let completed = fence_completed(&candidate.execution_fence)
            && fence_completed(&candidate.send_fence)
            && candidate
                .shards
                .iter()
                .filter_map(ArrayShard::buffer)
                .all(|buffer| buffer.ready().is_ok_and(|event| matches!(event.error(), Ok(None))));
        completed.then_some(candidate)
    }

    /// Returns whether this retained array lacks the buffer of some shard on a device of this process, and this process
    /// can supply every such buffer on its own. A missing buffer can be supplied when its shard has no elements, for
    /// which an empty buffer suffices, or when another shard on this process holds the same contribution with a buffer,
    /// which can be copied. It returns `false` when no local buffer is missing, and for an array of the zero data type,
    /// which never needs buffers.
    ///
    /// A placement of such an array onto its own mesh and sharding supplies the missing buffers. Execution would
    /// otherwise supply them again on every call, by the same rule, so a caller places the array once instead. The
    /// check reads only the placement metadata and the buffers of this process, and supplying the buffers moves no
    /// shard to another device, so it involves no other process, whatever buffers the other processes hold.
    pub(crate) fn can_fill_missing_local_buffers(&self) -> Result<bool, Error> {
        let process_index = self.client.process_index()?;
        let local = |shard: &&ArrayShard| shard.device().process_index() == process_index;
        let missing = |shard: &&ArrayShard| shard.buffer().is_none();
        let is_empty = |shard: &ArrayShard| shard.slice().iter().any(|range| range.is_empty());
        let has_missing = self.shards.iter().filter(local).any(|shard| missing(&shard));
        if !has_missing || self.metadata.r#type().data_type().is_zero() {
            return Ok(false);
        }
        let mesh = self.mesh();
        let sharding = self.sharding();
        let varying_manual_axes = mesh
            .logical_mesh()
            .axes()
            .iter()
            .map(|axis| axis.name())
            .filter(|name| sharding.varying_manual_axes().contains(*name))
            .collect::<Vec<_>>();
        let contribution =
            |shard: &ArrayShard| ShardContribution::ordered(mesh, sharding, shard.descriptor(), &varying_manual_axes);
        let held = self
            .shards
            .iter()
            .filter(local)
            .filter(|shard| shard.buffer().is_some())
            .map(contribution)
            .collect::<HashSet<_>>();
        let is_served = |shard: &ArrayShard| is_empty(shard) || held.contains(&contribution(shard));
        Ok(self.shards.iter().filter(local).filter(missing).all(is_served))
    }

    /// Returns this retained array as a placement of it onto the devices where it already lives, which a program names
    /// for every process. The result shares its buffers and is committed to its devices. Its assignment is known to
    /// every process, so an executable sees the same placement facts whether the source already lived there or moved.
    pub(crate) fn on_named_devices(&self) -> Self {
        Self { committed: true, locally_assigned: false, ..self.clone() }
    }

    /// Validates that `client` owns the buffers of this retained array.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError::Type`] error if the buffers belong to a different PJRT client.
    pub(crate) fn validate_client(&self, client: &Client<'_>) -> Result<(), ProgramError> {
        if !std::ptr::eq(self.client, client) {
            return Err(TypeError::invalid(format!(
                "session constant `{self}` retains an array of a different PJRT client \
                 than the client of this session",
            ))
            .into());
        }
        Ok(())
    }
}

impl Debug for RetainedArray {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Show the fields that distinguish retained constants. The shards themselves have no useful debug format.
        formatter
            .debug_struct("RetainedArray")
            .field("type", &self.declared_type)
            .field("storage", &self.identity.as_ptr())
            .field("committed", &self.committed)
            .finish()
    }
}

impl Display for RetainedArray {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "session:{}", self.declared_type)
    }
}

impl PartialEq for RetainedArray {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // The weak handle keeps the storage address reserved for the lifetime of the constant.
        Weak::ptr_eq(&self.identity, &other.identity)
            && self.declared_type == other.declared_type
            && self.committed == other.committed
    }
}

impl Eq for RetainedArray {}

impl Hash for RetainedArray {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.identity.as_ptr().hash(state);
        self.declared_type.hash(state);
        self.committed.hash(state);
    }
}

impl Typed for RetainedArray {
    type Type = ArrayType;

    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayType> {
        Cow::Borrowed(&self.declared_type)
    }
}

/// Placement index of the storage of an [`Array`], which [`RetainedArray::placed`] keeps for every retention of it.
pub(super) type PlacementIndex = Mutex<PlacementEntries>;

/// Entries of a [`PlacementIndex`], grouped by the source's local assignment as well as the destination. The device
/// index keeps attempts in submission order for [`RetainedArray::completed_placement_on`].
#[derive(Default)]
pub(super) struct PlacementEntries {
    attempts: HashMap<(DeviceMesh, Sharding, bool), PlacementAttempt>,
    attempts_by_devices: HashMap<(Vec<Device>, bool), Vec<PlacementAttempt>>,
}

/// Attempt of [`RetainedArray::placed`] to place a retained array on one mesh and sharding, which every caller that
/// requests that placement while the attempt runs or after it waits for and shares, whether it succeeds or fails.
type PlacementAttempt = Arc<OnceLock<Result<RetainedArray, Error>>>;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    use pretty_assertions::assert_eq;

    use ryft_core::{
        DataType, Device, DimensionBounds, DimensionVariable, Error as CoreError, LogicalMesh, Memory, MeshAxis,
        MeshAxisType, Placement, ShardingDimension,
    };
    use ryft_pjrt::{Buffer, BufferType, ClientOptions, CpuClientOptions, load_cpu_plugin};

    use crate::sessions::tests::{domain_mesh, f32_scalar, replicated_scalar_type, shares_effect_scope};
    use crate::tests::{
        cpu_client, execution_client, hash_of, other_cpu_client, peer_devices, session_with_remote_devices,
        values_to_bytes,
    };
    use crate::{FromPjrt, PlacementOptions};

    use super::*;

    /// Returns a mesh with axis `name` over the first device of `client` and a device of another process, which device
    /// metadata alone describes, so that a placement onto it can communicate.
    fn communicating_mesh(client: &Client<'static>, name: &str) -> DeviceMesh {
        let local = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let remote = Device::new(local.id() + 1, local.process_index() + 1);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new(name, 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        DeviceMesh::new(logical_mesh, vec![local, remote]).unwrap()
    }

    #[test]
    fn test_retained_array_new() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let array = Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 1f32.to_ne_bytes())
            .unwrap();

        // Retaining one array twice yields equal retained arrays, which share the type of the array and keep its
        // device buffers alive after the array itself is dropped.
        let retained = RetainedArray::new(&array);
        assert_eq!(RetainedArray::new(&array), retained);
        assert_eq!(&retained.declared_type, array.parts.metadata.r#type());
        assert!(Arc::ptr_eq(&retained.metadata, &array.parts.metadata));
        drop(array);
        let restored = retained.to_array(&session).unwrap();
        assert_eq!(restored.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1f32]);

        // An array that was created from a retained array has storage of its own, so retaining it yields a different
        // retained array.
        assert_ne!(RetainedArray::new(&restored), retained);
    }

    #[test]
    fn test_retained_array_sharding() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]),
            mesh,
            Some(sharding.clone()),
            values_to_bytes(&[1f32, 2f32]),
        )
        .unwrap();
        assert_eq!(RetainedArray::new(&array).sharding(), &sharding);
    }

    #[test]
    fn test_retained_array_mesh() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let array = f32_scalar(&session, &mesh, 1f32);
        assert_eq!(RetainedArray::new(&array).mesh(), &mesh);
    }

    #[test]
    fn test_retained_array_is_committed() {
        let session = Session::builder().client(cpu_client(1)).build().unwrap();
        let array = Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 1f32.to_ne_bytes())
            .unwrap();
        assert!(!RetainedArray::new(&array).is_committed());
        assert!(RetainedArray::new(&array.with_committed(true)).is_committed());
    }

    #[test]
    fn test_retained_array_is_locally_assigned() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let r#type = ArrayType::scalar(DataType::F32);
        let array = Array::from_host_buffer(&session, r#type.clone(), None, None, 1f32.to_ne_bytes()).unwrap();
        let retained = RetainedArray::new(&array);
        assert!(retained.is_locally_assigned());

        // Local assignment is preserved independently of retained-array identity.
        let cleared = RetainedArray::new(&array.with_locally_assigned(false));
        assert!(!cleared.is_locally_assigned());
        assert_eq!(cleared, retained);
        let shared = Array::from_host_buffer(&session, r#type, domain_mesh(client, "x", 1), None, 1f32.to_ne_bytes());
        assert!(!RetainedArray::new(&shared.unwrap()).is_locally_assigned());
    }

    #[test]
    fn test_retained_array_to_array() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let array = Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 1f32.to_ne_bytes())
            .unwrap();
        let retained = RetainedArray::new(&array);

        // The new array refers to the retained device buffers and belongs to the session handle that created it.
        let fork = session.with_new_effect_scope();
        let restored = retained.to_array(&fork).unwrap();
        assert_eq!(restored, array);
        assert!(shares_effect_scope(restored.session(), &fork));

        // The new array keeps the commitment of the retained array, even on a session whose mesh
        // would commit an upload.
        assert!(!retained.to_array(&session.with_device_mesh(domain_mesh(client, "x", 1))).unwrap().is_committed());
        assert!(RetainedArray::new(&array.clone().with_committed(true)).to_array(&fork).unwrap().is_committed());

        // The new array also keeps whether the process chose the devices of the retained array on its own.
        assert!(retained.to_array(&fork).unwrap().is_locally_assigned());
        let cleared = RetainedArray::new(&array.clone().with_locally_assigned(false));
        assert!(!cleared.to_array(&fork).unwrap().is_locally_assigned());

        // Every session of the client that owns the buffers can create the array, and the buffers outlive the session
        // of the array that was retained.
        let other_session = Session::builder().client(client).build().unwrap();
        drop((array, restored, fork, session));
        let restored = retained.to_array(&other_session).unwrap();
        assert!(restored.session().is_same_session(&other_session));
        assert_eq!(restored.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1f32]);

        // Sessions of another client cannot use the buffers.
        let foreign_session = Session::builder().client(other_cpu_client(1)).build().unwrap();
        assert_eq!(
            retained.to_array(&foreign_session),
            Err(TypeError::invalid(
                "session constant `session:f32[]` retains an array of a \
                 different PJRT client than the client of this session",
            )
            .into()),
        );
    }

    #[test]
    fn test_retained_array_rename_type_identities() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let array = Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 1f32.to_ne_bytes())
            .unwrap();
        let retained = RetainedArray::new(&array);

        // Runtime arrays have static shapes, so their types have no identities and a renaming leaves the declared type
        // as it is. The renamed value refers to the same array.
        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let mut renaming = TypeIdentityRenaming::new();
        renaming
            .insert(DimensionVariable::new("source", bounds), DimensionVariable::new("target", bounds))
            .unwrap();
        let renamed = retained.rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed, retained);
        assert_eq!(renamed.declared_type, retained.declared_type);
        assert!(Arc::ptr_eq(&renamed.metadata, &retained.metadata));
        assert!(Arc::ptr_eq(&renamed.shards, &retained.shards));
    }

    #[test]
    fn test_retained_array_debug() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let array = Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 1f32.to_ne_bytes())
            .unwrap();
        assert_eq!(
            format!("{:?}", RetainedArray::new(&array)),
            format!(
                "RetainedArray {{ type: {:?}, storage: {:?}, committed: false }}",
                array.parts.metadata.r#type(),
                Arc::as_ptr(&array.parts),
            ),
        );
    }

    #[test]
    fn test_retained_array_display() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let array = Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 1f32.to_ne_bytes())
            .unwrap();
        assert_eq!(RetainedArray::new(&array).to_string(), "session:f32[]");
    }

    #[test]
    fn test_retained_array_eq() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let first = Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 1f32.to_ne_bytes())
            .unwrap();
        let second =
            Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 1f32.to_ne_bytes())
                .unwrap();
        let retained = RetainedArray::new(&first);

        // Retained arrays are equal when they were created from the same array, and never because two arrays have
        // the same type and the same contents.
        assert_eq!(retained, retained.clone());
        assert_eq!(retained, RetainedArray::new(&first));
        assert_ne!(retained, RetainedArray::new(&second));
        assert_ne!(retained, RetainedArray::new(&first.with_committed(true)));
    }

    #[test]
    fn test_retained_array_placed() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let upload =
            || Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 1f32.to_ne_bytes());

        // A capture with locally assigned devices never communicates (refer to
        // `RetainedArray::placement_communicates`), so the source names its devices for every process.
        let source = upload().unwrap().with_locally_assigned(false);
        let retained = RetainedArray::new(&source);
        let attempts = AtomicUsize::new(0);
        let place = |outcome: Result<(), &str>| {
            attempts.fetch_add(1, Ordering::SeqCst);
            match outcome {
                Ok(()) => Ok(upload()?),
                Err(message) => Err(Error::from(ProgramError::InvalidArgument { message: message.to_string() })),
            }
        };
        let cause = |message: &str| CoreError::from(ProgramError::InvalidArgument { message: message.to_string() });
        let replicated = |mesh: &DeviceMesh| Sharding::replicated(mesh.logical_mesh().clone(), 0);

        // A placement that can communicate has one attempt for the lifetime of the array, whose failure every later
        // caller receives without placing the array again, because a retry by one process alone would unpair the
        // cross-host transfers of the processes. The capture must be created again to retry. Every caller, the one
        // that ran the attempt included, receives the same typed failure, whose source is the original error of the
        // attempt, so that callers can match on its cause.
        let failure = |outcome: Result<RetainedArray, Error>| {
            let Err(error) = outcome else { panic!("expected a failed placement") };
            let message = error.to_string();
            let Error::Array(error) = error else { panic!("expected an array error, got {message}") };
            let ArrayError::FailedCapturePlacement { sharding, source } = *error else {
                panic!("expected a failed capture placement, got {message}");
            };
            (message, sharding, source)
        };
        let message = |sharding: &Sharding, cause: &str| {
            format!(
                "the placement of this captured array as `{sharding}` failed with: {cause}; a failed placement of a \
                 capture is not retried, because a retry by one process alone would unpair its cross-host transfers, \
                 so create the capture again on every process to retry it",
            )
        };
        let mesh = communicating_mesh(client, "x");
        let sharding = replicated(&mesh);
        assert_eq!(
            (
                retained.placement_communicates(&mesh, &sharding),
                RetainedArray::new(&upload().unwrap()).placement_communicates(&mesh, &sharding),
            ),
            (true, false),
        );
        let first = failure(retained.placed(&mesh, &sharding, |_| place(Err("placement failed"))));
        let later = failure(RetainedArray::new(&source).placed(&mesh, &sharding, |_| place(Ok(()))));
        assert_eq!((&first.0, &first.1), (&message(&sharding, "placement failed"), &sharding));
        assert_eq!((&later.0, &later.1), (&first.0, &first.1));
        assert!(Arc::ptr_eq(&first.2, &later.2));
        assert!(matches!(first.2.as_ref(), Error::Core(error) if **error == cause("placement failed")));

        // A placement whose producer panics may already have submitted a transfer, so its attempt records a failure
        // as well, and the panic continues on the thread that ran it.
        let other_mesh = communicating_mesh(client, "y");
        let other_sharding = replicated(&other_mesh);
        let panicked = catch_unwind(AssertUnwindSafe(|| {
            retained.placed(&other_mesh, &other_sharding, |_| panic!("producer panicked"))
        }));
        assert!(panicked.is_err());
        let later = failure(retained.placed(&other_mesh, &other_sharding, |_| place(Ok(()))));
        assert_eq!((&later.0, &later.1), (&message(&other_sharding, "the placement panicked"), &other_sharding));
        assert!(matches!(later.2.as_ref(), Error::Core(error) if **error == cause("the placement panicked")));
        assert_eq!(attempts.load(Ordering::SeqCst), 1);

        // A placement whose devices all belong to one process cannot communicate, so its failure is returned as it
        // is, its attempt is removed once it completes, and the next request, from any retention, places the array
        // again. A panic is retried as well.
        let single_process = |outcome: Result<RetainedArray, Error>| match outcome {
            Err(Error::Core(error)) => *error,
            Err(error) => panic!("expected the error of the placement, got {error}"),
            Ok(_) => panic!("expected a failed placement"),
        };
        let local_mesh = domain_mesh(client, "x", 2);
        let local_sharding = replicated(&local_mesh);
        let failed = retained.placed(&local_mesh, &local_sharding, |_| place(Err("placement failed")));
        assert_eq!(single_process(failed), cause("placement failed"));
        let panicked = catch_unwind(AssertUnwindSafe(|| {
            retained.placed(&local_mesh, &local_sharding, |_| panic!("producer panicked"))
        }));
        assert!(panicked.is_err());
        let failed = retained.placed(&local_mesh, &local_sharding, |_| place(Err("placement failed again")));
        assert_eq!(single_process(failed), cause("placement failed again"));
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
        let placed = RetainedArray::new(&source).placed(&local_mesh, &local_sharding, |_| place(Ok(()))).unwrap();
        assert_eq!(retained.placed(&local_mesh, &local_sharding, |_| place(Ok(()))).unwrap(), placed);
        assert_eq!(attempts.load(Ordering::SeqCst), 4);

        // A successful placement is recorded and shared by every clone of the retention, including a clone with
        // another declared type.
        let third_mesh = domain_mesh(client, "z", 2);
        let third_sharding = replicated(&third_mesh);
        let placed = retained.placed(&third_mesh, &third_sharding, |_| place(Ok(()))).unwrap();
        let declared_type = ArrayType::scalar(DataType::F32).with_sharding(Some(third_sharding.clone())).unwrap();
        let shared = retained.with_declared_type(declared_type).placed(&third_mesh, &third_sharding, |_| place(Ok(())));
        assert_eq!(shared.unwrap(), placed);
        assert_eq!(attempts.load(Ordering::SeqCst), 5);

        // The two ownership modes can submit different cross-process work, even for identical destination keys.
        let local = RetainedArray::new(&source.with_locally_assigned(true));
        assert!(!local.placement_communicates(&mesh, &sharding));
        local.placed(&mesh, &sharding, |_| place(Ok(()))).unwrap();
        assert_eq!(attempts.load(Ordering::SeqCst), 6);
        let later = failure(retained.placed(&mesh, &sharding, |_| place(Ok(()))));
        assert!(Arc::ptr_eq(&first.2, &later.2));
    }

    #[test]
    fn test_retained_array_placed_retries_a_local_replica_fill_on_a_shared_mesh() {
        let client = cpu_client(2);
        let local = client.addressable_devices().unwrap();
        let first = Device::from_pjrt(&local[0]).unwrap();
        let second = Device::from_pjrt(&local[1]).unwrap();
        let remote = Device::new(second.id() + 1, second.process_index() + 1);
        let session = session_with_remote_devices(client, [remote]);
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 3, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![first, second, remote]).unwrap();
        let sharding = Sharding::replicated(logical_mesh, 1);
        let buffer = client.buffer([7u8].as_slice(), BufferType::U8, [1u64], None, local[0].clone(), None).unwrap();
        let array = Array::from_addressable_buffers(
            &session,
            ArrayType::new_static(DataType::U8, [1]),
            mesh.clone(),
            Some(sharding.clone()),
            vec![buffer],
        )
        .unwrap();
        let retained = RetainedArray::new(&array);
        assert!(retained.can_fill_missing_local_buffers().unwrap());
        assert!(!retained.placement_communicates(&mesh, &sharding));
        let failure = CoreError::from(ProgramError::InvalidArgument { message: "replica fill failed".to_string() });
        let failed = retained.placed(&mesh, &sharding, |_| Err(failure.clone().into()));
        assert!(matches!(failed, Err(Error::Core(error)) if *error == failure));
        let retry = RetainedArray::new(&array)
            .placed(&mesh, &sharding, |source| {
                source.to_array(&session)?.reshard_on_devices(&session, &mesh, &sharding, false)
            })
            .unwrap();
        let output = retry.to_array(&session).unwrap();
        output.block_until_ready().unwrap();
        let local = output.addressable_shards().map(|shard| shard.copy_to_host_bytes().unwrap()).collect::<Vec<_>>();
        assert_eq!(local, [vec![7], vec![7]]);
        assert!(output.device_shard(remote.id()).unwrap().buffer().is_none());

        // Moving off a shared mesh can still use a local replica. Its failed copy must also be retried.
        let singleton = |device| {
            let mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![device]).unwrap();
            let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
            Placement::new(mesh, sharding).unwrap()
        };
        let local = singleton(second);
        let distant = singleton(Device::new(remote.id() + 1, remote.process_index() + 1));
        for (target, communicates) in [(&local, false), (&distant, true)] {
            assert_eq!(array.placement_communicates(target.mesh(), target.sharding()), communicates);
            assert_eq!(retained.placement_communicates(target.mesh(), target.sharding()), communicates);
        }
        let failed = retained.placed(local.mesh(), local.sharding(), |_| Err(failure.clone().into()));
        assert!(matches!(failed, Err(Error::Core(error)) if *error == failure));
        let retry = retained
            .placed(local.mesh(), local.sharding(), |source| {
                source.to_array(&session)?.reshard_on_devices(&session, local.mesh(), local.sharding(), false)
            })
            .unwrap()
            .to_array(&session)
            .unwrap();
        retry.block_until_ready().unwrap();
        assert_eq!(retry.mesh(), local.mesh());
        assert_eq!(retry.copy_to_host_bytes().unwrap(), [7]);
    }

    #[test]
    fn test_retained_array_placed_shares_copies_between_retentions_for_the_lifetime_of_the_array() {
        // Independent retentions of one array share one placement for each destination, whose copy lives until the
        // array and every retention of it are gone, in whichever order they are dropped.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let r#type = ArrayType::scalar(DataType::F32);
        let submissions = AtomicUsize::new(0);
        let place = |_| {
            submissions.fetch_add(1, Ordering::SeqCst);
            let (mesh, sharding) = (mesh.clone(), Some(sharding.clone()));
            Ok::<_, Error>(Array::from_host_buffer(&session, r#type.clone(), mesh, sharding, 1f32.to_ne_bytes())?)
        };
        let buffers = |array: &RetainedArray| {
            array.shards.iter().map(|shard| Arc::clone(shard.buffer().unwrap())).collect::<Vec<_>>()
        };
        for array_dropped_first in [false, true] {
            let mut source =
                Some(Array::from_host_buffer(&session, r#type.clone(), None, None, 1f32.to_ne_bytes()).unwrap());
            let mut retentions = Some((0..4).map(|_| RetainedArray::new(source.as_ref().unwrap())).collect::<Vec<_>>());
            let copies = retentions
                .iter()
                .flatten()
                .map(|retained| retained.placed(&mesh, &sharding, place).unwrap())
                .collect::<Vec<_>>();
            assert!(copies.iter().all(|copy| *copy == copies[0]));
            assert!(copies.iter().all(|copy| {
                buffers(copy).iter().zip(buffers(&copies[0])).all(|(buffer, first)| Arc::ptr_eq(buffer, &first))
            }));
            let copy_buffers = buffers(&copies[0]).iter().map(Arc::downgrade).collect::<Vec<Weak<Buffer<'static>>>>();
            let live = || copy_buffers.iter().filter(|buffer| buffer.strong_count() > 0).count();
            assert_eq!(live(), 2);
            drop(copies);
            if array_dropped_first {
                source = None;
            } else {
                retentions = None;
            }
            assert_eq!(live(), 2);
            drop((source, retentions));
            assert_eq!(live(), 0);
        }
        assert_eq!(submissions.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_retained_array_placed_starts_over_for_a_new_identity_that_keeps_its_storage() {
        // A new logical array that keeps uniquely owned storage (refer to `Array::with_new_identity`) inherits none of
        // the placements of the old one, so it places again on every process, whether that process kept its storage
        // or allocated new storage, as a process without the buffers of the array does.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 0);
        let r#type = ArrayType::scalar(DataType::F32);
        let submissions = AtomicUsize::new(0);
        let place = |_| {
            submissions.fetch_add(1, Ordering::SeqCst);
            let (mesh, sharding) = (mesh.clone(), Some(sharding.clone()));
            Ok::<_, Error>(Array::from_host_buffer(&session, r#type.clone(), mesh, sharding, 1f32.to_ne_bytes())?)
        };
        let array = Array::from_host_buffer(&session, r#type.clone(), None, None, 1f32.to_ne_bytes()).unwrap();
        let retained = RetainedArray::new(&array);
        retained.placed(&mesh, &sharding, place).unwrap();
        RetainedArray::new(&array).placed(&mesh, &sharding, place).unwrap();
        assert_eq!(submissions.load(Ordering::SeqCst), 1);
        drop(retained);
        let storage = Arc::as_ptr(&array.parts);
        let array = array.with_new_identity();
        assert_eq!(Arc::as_ptr(&array.parts), storage);
        RetainedArray::new(&array).placed(&mesh, &sharding, place).unwrap();
        assert_eq!(submissions.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_retained_array_placed_shares_one_attempt_between_concurrent_callers() {
        // The first caller records the attempt before it places anything, and its `place` blocks until it is
        // released, while four other callers, two of them through another retention of the same array, request the
        // same placement. It is released only once every waiter holds the recorded attempt, which a waiter takes before
        // it waits for that attempt, so every waiter arrives while the attempt runs. Each shares its outcome and never
        // submits a placement of its own, for a success and for a failure alike.
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let local_mesh = domain_mesh(client, "x", 1);
        let remote_mesh = communicating_mesh(client, "x");
        let upload =
            || Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 1f32.to_ne_bytes());
        let outcomes = |succeed: bool, mesh: &DeviceMesh| {
            let sharding = &Sharding::replicated(mesh.logical_mesh().clone(), 0);
            let array = upload().unwrap().with_locally_assigned(false);
            let retentions = [RetainedArray::new(&array), RetainedArray::new(&array)];
            let submission_count = AtomicUsize::new(0);
            let submissions = &submission_count;
            let (started_sender, started) = mpsc::channel();
            let (release, released) = mpsc::channel::<()>();
            let place = move |_| {
                submissions.fetch_add(1, Ordering::SeqCst);
                started_sender.send(()).unwrap();
                released.recv().unwrap();
                if succeed {
                    Ok(upload()?)
                } else {
                    Err(Error::from(ProgramError::InvalidArgument { message: "placement failed".to_string() }))
                }
            };
            let outcomes = std::thread::scope(|scope| {
                let first = scope.spawn(|| retentions[0].placed(mesh, sharding, place));

                // The attempt is recorded before `place` runs, so every waiter below finds it.
                started.recv().unwrap();
                let waiters = (0..4)
                    .map(|index| {
                        let retained = &retentions[index % 2];
                        scope.spawn(move || {
                            retained.placed(mesh, sharding, |_| {
                                submissions.fetch_add(1, Ordering::SeqCst);
                                Ok(upload()?)
                            })
                        })
                    })
                    .collect::<Vec<_>>();

                // The two records of the placements (by key and by devices), the first caller, and the four waiters
                // hold the attempt, which none of them releases before the attempt finishes.
                let attempt_holders = || {
                    let placements = retentions[1].placements.lock().unwrap();
                    Arc::strong_count(placements.attempts.values().next().unwrap())
                };

                // A regression that keeps a waiter from the attempt fails the test rather than hanging it, after the
                // producer is released so that every thread of the scope finishes.
                let deadline = Instant::now() + Duration::from_secs(60);
                while attempt_holders() < 7 {
                    if Instant::now() >= deadline {
                        release.send(()).unwrap();
                        panic!("the waiters did not all hold the attempt within 60 seconds");
                    }
                    std::thread::yield_now();
                }
                release.send(()).unwrap();
                [first].into_iter().chain(waiters).map(|thread| thread.join().unwrap()).collect::<Vec<_>>()
            });
            let recorded = {
                let placements = retentions[0].placements.lock().unwrap();
                (placements.attempts.len(), placements.attempts_by_devices.values().map(Vec::len).sum::<usize>())
            };
            (submission_count.load(Ordering::SeqCst), recorded, outcomes)
        };

        for mesh in [&local_mesh, &remote_mesh] {
            let (submissions, recorded, placed) = outcomes(true, mesh);
            assert_eq!((submissions, recorded), (1, (1, 1)));
            let placed = placed.into_iter().map(Result::unwrap).collect::<Vec<_>>();
            assert!(placed.iter().all(|array| *array == placed[0]));
        }

        // Every caller of a placement that can communicate receives the same typed failure, which shares the original
        // error of the attempt as its source, and the attempt stays recorded.
        let cause = CoreError::from(ProgramError::InvalidArgument { message: "placement failed".to_string() });
        let (submissions, recorded, failed) = outcomes(false, &remote_mesh);
        assert_eq!((submissions, recorded), (1, (1, 1)));
        let sharding = Sharding::replicated(remote_mesh.logical_mesh().clone(), 0);
        let sources = failed
            .into_iter()
            .map(|outcome| match outcome.map(|_| ()) {
                Err(Error::Array(error)) => match *error {
                    ArrayError::FailedCapturePlacement { sharding: failed_sharding, source } => {
                        assert_eq!(failed_sharding, sharding);
                        source
                    }
                    error => panic!("expected a failed capture placement, got {error}"),
                },
                outcome => panic!("expected an array error, got {outcome:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(sources.len(), 5);
        assert!(sources.iter().all(|source| Arc::ptr_eq(source, &sources[0])));
        assert!(matches!(sources[0].as_ref(), Error::Core(error) if **error == cause));

        // Every caller of a placement that cannot communicate receives the error of the attempt as it is, and the
        // attempt is removed once it completes.
        let (submissions, recorded, failed) = outcomes(false, &local_mesh);
        assert_eq!((submissions, recorded), (1, (0, 0)));
        let errors = failed
            .into_iter()
            .map(|outcome| match outcome.map(|_| ()) {
                Err(Error::Core(error)) => *error,
                outcome => panic!("expected the error of the placement, got {outcome:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(errors, vec![cause; 5]);
    }

    #[test]
    fn test_retained_array_eq_follows_the_producing_operation_on_every_process() {
        // Two single-device clients stand for two processes. Each places a scalar on the device of process 0, so that
        // only process 0 holds a buffer of it, and derives arrays from it with every function that returns an array
        // from an array. Whether a result is the same retained array as its input must follow from the function alone.
        let clients = [0, 1].map(|process| {
            let options = CpuClientOptions { device_count: Some(1), process_id: Some(process) };
            let client = load_cpu_plugin().unwrap().client(ClientOptions::CPU(options)).unwrap();
            let client: &'static Client<'static> = Box::leak(Box::new(client));
            client
        });
        let device = Device::from_pjrt(&clients[0].addressable_devices().unwrap()[0]).unwrap();
        let rows = clients.map(|client| {
            let session = session_with_remote_devices(client, peer_devices(client, &clients));
            let local = session.place(4f32, PlacementOptions::default()).unwrap();
            let array = session.place(local.clone(), PlacementOptions::to(device)).unwrap();
            let holds_buffer = client.process_index().unwrap() == 0;
            assert_eq!(array.addressable_shards().count(), usize::from(holds_buffer));
            let copy = PlacementOptions::default().with_may_alias(false);
            let copy_on_device = PlacementOptions::to(device).with_may_alias(false);
            let donated_copy = PlacementOptions::default().with_donate(true).with_may_alias(false);

            // Commitment is part of the identity of a retained array but not of the logical array, so the input is
            // compared with the commitment of each result.
            let same = |input: &Array, output: &Array| {
                RetainedArray::new(output) == RetainedArray::new(&input.clone().with_committed(output.is_committed()))
            };
            let mut rows = vec![
                ("clone", same(&array, &array.clone())),
                (
                    "place without a destination",
                    same(&array, &session.place(array.clone(), PlacementOptions::default()).unwrap()),
                ),
                (
                    "place on its own placement",
                    same(&array, &session.place(array.clone(), PlacementOptions::to(device)).unwrap()),
                ),
                ("to_placement on its own placement", same(&array, &array.to_placement(&session, device).unwrap())),
                (
                    "into_placement on its own placement",
                    same(&array, &array.clone().into_placement(&session, device).unwrap()),
                ),
                (
                    "in-place placement in its own memory",
                    same(&array, &array.to_placement(&session, Memory::Device).unwrap()),
                ),
                (
                    "in-place placement of a locally assigned array",
                    same(&local, &local.to_placement(&session, Memory::Device).unwrap()),
                ),
                ("forced copy", same(&array, &session.place(array.clone(), copy.clone()).unwrap())),
                (
                    "forced copy on its own placement",
                    same(&array, &session.place(array.clone(), copy_on_device).unwrap()),
                ),
                ("donated without alias", same(&array, &session.place(array.clone(), donated_copy).unwrap())),
                ("unaliased", same(&array, &array.clone().into_unaliased().unwrap())),
                ("copied shard buffers", same(&array, &array.with_copied_shard_buffers(|_| true).unwrap())),
                (
                    "reshard on its own devices",
                    same(
                        &array,
                        &array.clone().reshard_on_devices(&session, array.mesh(), array.sharding(), false).unwrap(),
                    ),
                ),
                ("restored retained array", same(&array, &RetainedArray::new(&array).to_array(&session).unwrap())),
                (
                    "place of a locally assigned array",
                    same(&local, &session.place(local.clone(), PlacementOptions::to(device)).unwrap()),
                ),
                (
                    "to_placement of a locally assigned array",
                    same(&local, &local.to_placement(&session, device).unwrap()),
                ),
            ];

            // A forced copy of the only handle of an array is a new array as well, when a retained array refers to it.
            let retained = RetainedArray::new(&array);
            rows.push((
                "forced copy of the only handle",
                retained == RetainedArray::new(&session.place(array, copy).unwrap()),
            ));
            rows
        });

        assert_eq!(rows[0], rows[1]);
        assert_eq!(
            rows[0],
            vec![
                ("clone", true),
                ("place without a destination", true),
                ("place on its own placement", true),
                ("to_placement on its own placement", true),
                ("into_placement on its own placement", true),
                ("in-place placement in its own memory", true),
                ("in-place placement of a locally assigned array", true),
                ("forced copy", false),
                ("forced copy on its own placement", false),
                ("donated without alias", true),
                ("unaliased", false),
                ("copied shard buffers", false),
                ("reshard on its own devices", false),
                ("restored retained array", false),
                ("place of a locally assigned array", false),
                ("to_placement of a locally assigned array", false),
                ("forced copy of the only handle", false),
            ],
        );
    }

    #[test]
    fn test_retained_array_hash() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let first = Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 1f32.to_ne_bytes())
            .unwrap();
        let second =
            Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 1f32.to_ne_bytes())
                .unwrap();
        let retained = RetainedArray::new(&first);
        let other = RetainedArray::new(&second);
        assert_eq!(hash_of(&retained), hash_of(&RetainedArray::new(&first)));
        assert_ne!(hash_of(&retained), hash_of(&other));
        let committed = RetainedArray::new(&first.clone().with_committed(true));
        assert_ne!(hash_of(&retained), hash_of(&committed));

        // Commitment distinguishes keys even when they share their storage and declared type.
        #[allow(clippy::mutable_key_type)]
        let constants =
            HashMap::from([(retained, "first"), (other.clone(), "second"), (committed.clone(), "committed")]);
        assert_eq!(constants.get(&RetainedArray::new(&first)), Some(&"first"));
        assert_eq!(constants.get(&other), Some(&"second"));
        assert_eq!(constants.get(&committed), Some(&"committed"));
    }

    #[test]
    fn test_retained_array_type() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap();
        let r#type = replicated_scalar_type(&mesh, DataType::F32);
        let array = Array::from_host_buffer(&session, r#type.clone(), None, None, 1f32.to_ne_bytes()).unwrap();
        assert_eq!(RetainedArray::new(&array).r#type().into_owned(), r#type);
    }
}
