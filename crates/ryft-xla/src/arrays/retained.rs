use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Weak};

use ryft_core::{
    ArrayType, DeviceMesh, Parameter, ProgramError, Sharding, Type, TypeError, TypeIdentityRenaming, Typed,
};
use ryft_macros::Parameter;
use ryft_pjrt::{Client, ExecutionFence};

use crate::Session;
use crate::arrays::metadata::ArrayMetadata;
use crate::arrays::shards::ArrayShard;
use crate::arrays::{Array, ArrayParts};

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
/// Two retained arrays are equal when they retain the same logical array with the same commitment. Lowering passes
/// each distinct retained array of a program as one hidden argument of its executable, and the identity of a program
/// and its cache keys include its constants, so every process of a distributed program must decide alike whether two
/// retained arrays are equal. The logical identity of an array is therefore decided by the operation that produced it,
/// never by whether this process held buffers to copy or to reuse, and every function that returns an array from
/// another array follows these rules on every process:
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
///
/// Donation never changes these rules, because it only consumes the input. An operation whose result is a new array
/// but that would return the storage of its input on some process gets storage of its own from one function,
/// `Array::with_new_identity`.
///
/// # Ownership
///
/// A retained array owns its shards, and its clones share them. The programs, regions, derivative rules, and caches
/// that hold one therefore keep the device buffers alive, and the buffers are released together with the last value
/// that refers to them. Any session of the PJRT client that owns the buffers can turn a retained array back into an
/// [`Array`].
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

    /// Whether the retained array is committed to its devices, which the arrays that [`Self::to_array`] creates keep.
    /// It is part of the identity of this constant. Refer to [`Array::is_committed`] for more information.
    committed: bool,

    /// Whether the process chose the devices of the retained array on its own, which the arrays that [`Self::to_array`]
    /// creates keep. Local assignment is excluded from the identity of this retained array. Refer to
    /// [`Array::is_locally_assigned`] for more information.
    locally_assigned: bool,

    /// PJRT [`Client`] that owns the buffers of the retained array.
    client: &'static Client<'static>,
}

impl RetainedArray {
    /// Retains `array` without its session. Its device buffers stay alive for as long as the returned value or any of
    /// its clones exists. Retaining one array more than once returns equal values, so that repeatedly importing the
    /// same captured array (e.g., the weights that a compiled function captures) stages the same constant every time.
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
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError::Type`] error if the buffers are owned by another PJRT client than the one
    /// of `session`, which cannot use them.
    pub fn to_array(&self, session: &Session) -> Result<Array, ProgramError> {
        self.validate_client(session.client())?;

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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;

    use ryft_core::{DataType, Device, DimensionBounds, DimensionVariable, Memory, ShardingDimension};
    use ryft_pjrt::{ClientOptions, CpuClientOptions, load_cpu_plugin};

    use crate::sessions::tests::{domain_mesh, f32_scalar, replicated_scalar_type, shares_effect_scope};
    use crate::tests::{cpu_client, execution_client, hash_of, other_cpu_client, values_to_bytes};
    use crate::{FromPjrt, PlacementOptions};

    use super::*;

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
                "session constant `session:f32[][sharding={mesh<['device'=1:auto]>, []}]` retains an array of a \
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
        assert_eq!(RetainedArray::new(&array).to_string(), "session:f32[][sharding={mesh<['device'=1:auto]>, []}]");
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
            let session = Session::builder().client(client).build().unwrap();
            let local = session.place(4f32, PlacementOptions::defaults()).unwrap();
            let array = session.place(local.clone(), PlacementOptions::to(device)).unwrap();
            let holds_buffer = client.process_index().unwrap() == 0;
            assert_eq!(array.addressable_shards().count(), usize::from(holds_buffer));
            let copy = PlacementOptions::defaults().with_may_alias(false);
            let copy_on_device = PlacementOptions::to(device).with_may_alias(false);
            let donated_copy = PlacementOptions::defaults().with_donate(true).with_may_alias(false);

            // Commitment is part of the identity of a retained array but not of the logical array, so the input is
            // compared with the commitment of each result.
            let same = |input: &Array, output: &Array| {
                RetainedArray::new(output) == RetainedArray::new(&input.clone().with_committed(output.is_committed()))
            };
            let mut rows = vec![
                ("clone", same(&array, &array.clone())),
                (
                    "place without a destination",
                    same(&array, &session.place(array.clone(), PlacementOptions::defaults()).unwrap()),
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
                ("donated forced copy", same(&array, &session.place(array.clone(), donated_copy).unwrap())),
                ("unaliased", same(&array, &array.clone().into_unaliased(&session).unwrap())),
                ("copied shard buffers", same(&array, &array.with_copied_shard_buffers(&session, |_| true).unwrap())),
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
                ("donated forced copy", false),
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
