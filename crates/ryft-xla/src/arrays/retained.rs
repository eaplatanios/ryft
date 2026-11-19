//! Arrays that are kept alive without their sessions, for use as constants of staged programs.
//!
//! An [`Array`] carries the [`Session`] that it belongs to and caches of values derived from it, so a program that
//! held one would keep alive the session whose caches hold the program. [`RetainedArray`] is the part of an array
//! that is safe to store in a program. [`RetainedArray::new`] takes it from an array, and
//! [`RetainedArray::to_array`] turns it back into an array of a chosen session.

// TODO(eaplatanios): Review the module docstring.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Weak};

use ryft_core::{
    ArrayType, DeviceMesh, Parameter, ProgramError, Sharding, Type as CoreType, TypeError, TypeIdentityRenaming, Typed,
};
use ryft_macros::Parameter;
use ryft_pjrt::{Client, DeviceId, ExecutionFence};

use crate::Session;
use crate::arrays::{Array, ArrayParts, ArrayShard, ShardIndex};

/// Storage of an [`Array`] that is kept alive without the [`Session`] of that array, so that staged programs
/// can hold it as a constant. A trace whose capture table cannot hold runtime arrays (e.g., the body of a
/// [`rematerialize`](ryft_core::rematerialize) call or of a [`composite`](ryft_core::composite) and its rules, which
/// are traced over constants alone) still needs to refer to the runtime arrays that a compiled function it calls has
/// captured. Importing such an array through [`ThroughSession`](crate::ThroughSession) retains it and stages it as an immediate
/// [`Constant::Session`](crate::Constant::Session), which can appear in any region of any program, including lazily traced derivative rules.
/// Lifting that constant in a [`Session`] turns it back into an [`Array`], and lowering turns every distinct retained
/// array of a program into a hidden executable argument that follows the captures of the program, so that compiled
/// executables depend only on the array's type and never on its data.
///
/// # Differences from [`Array`]
///
/// A retained array refers to the same device buffers as the array that it was created from, but it is not a value
/// that operations can be applied to:
///
///   - **No Session:** An [`Array`] carries the [`Session`] that it belongs to, which is what lets operations on it
///     find their execution context. A retained array carries none, so it must first be turned back into an array
///     by lifting its constant in a session, which is also the session that the new array belongs to.
///   - **No Caches:** An [`Array`] owns caches of values derived from it (e.g., padded materializations), which are
///     arrays themselves. A retained array holds only the type, the placement, the shards, and the completion fences
///     of its array.
///   - **Usable as a Key:** A retained array implements [`Eq`] and [`Hash`] by the identity of the storage that it
///     was created from and by its type, so retaining one array twice yields equal values and programs that refer to
///     the same array share their compiled forms.
///
/// The first two differences are what make it safe to store in a program. Programs live in the caches of a session,
/// so a program that held an [`Array`] would keep alive the session whose caches keep the program alive, and neither
/// would ever be released, including the device buffers.
///
/// # Ownership
///
/// A retained array owns its shards, and its clones share them. The programs, regions, derivative rules, and caches
/// that hold one therefore keep the device buffers alive, and the buffers are released together with the last value
/// that refers to them. Any session of the PJRT client that owns the buffers can turn a retained array back into an
/// [`Array`].
#[derive(Clone, Parameter)]
pub struct RetainedArray {
    /// [`ArrayType`] that this retained array declares as a constant of a program, which is what [`Typed::r#type`]
    /// returns. It is [`Self::actual_type`] until [`Self::rename_type_identities`] renames the identities in it
    /// (i.e., the dimension variables of a bounded dynamic shape), which changes only this field.
    declared_type: Arc<ArrayType>,

    /// [`ArrayType`] of the [`Array`] that was retained, shared with it. It never changes, and the arrays that
    /// [`Self::to_array`] creates have it.
    actual_type: Arc<ArrayType>,

    /// Identity of the storage of the [`Array`] that was retained, which is the same for every retention of that
    /// array. It keeps the address of that storage reserved without keeping the storage or its caches alive.
    identity: Weak<ArrayParts>,

    /// Runtime placement of the retained array over all mesh axes.
    sharding: Arc<Sharding>,

    /// [`ArrayShard`]s of the retained array, which refer to its device buffers.
    shards: Arc<[ArrayShard]>,

    /// Lookup table that maps each [`DeviceId`] to the [`ShardIndex`] of its shard in [`Self::shards`].
    shard_index_by_device: Arc<HashMap<DeviceId, ShardIndex>>,

    /// Concrete [`DeviceMesh`] that the retained array is placed on.
    mesh: Arc<DeviceMesh>,

    /// Whole-execution completion fence of the launch that produced the retained array, if any.
    execution_fence: Option<ExecutionFence>,

    /// Completion fence of the cross-host sends of the placement that produced the retained array, if any.
    /// Arrays that [`Self::to_array`] creates share it with the retained array.
    send_fence: Option<ExecutionFence>,

    /// PJRT [`Client`] that owns the buffers of the retained array.
    client: &'static Client<'static>,
}

// TODO(eaplatanios): Review from here onwards.

impl RetainedArray {
    /// Retains `array` without its session and its caches. Its device buffers stay alive for as long as the returned
    /// value or any of its clones exists. Retaining one array more than once returns equal values, so that
    /// repeatedly importing the same captured array (e.g., the weights that a compiled function captures) stages the
    /// same constant every time.
    pub(crate) fn new(array: &Array) -> Self {
        Self {
            declared_type: Arc::clone(&array.parts.r#type),
            actual_type: Arc::clone(&array.parts.r#type),
            identity: Arc::downgrade(&array.parts),
            sharding: Arc::clone(&array.parts.sharding),
            shards: Arc::from(array.parts.shards.as_slice()),
            shard_index_by_device: Arc::clone(&array.parts.shard_index_by_device),
            mesh: Arc::clone(&array.parts.mesh),
            execution_fence: array.execution_fence.clone(),
            send_fence: array.send_fence.clone(),
            client: array.client(),
        }
    }

    /// Returns a new [`Array`] that refers to the device buffers of this retained array and belongs to `session`.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError::Type`] error if the buffers are owned by another PJRT client than the one of
    /// `session`, which cannot use them.
    pub(crate) fn to_array(&self, session: &Session) -> Result<Array, ProgramError> {
        if !std::ptr::eq(self.client, session.client()) {
            return Err(TypeError::invalid(format!(
                "session constant `{self}` retains an array of a different PJRT client than the client of this \
                 session",
            ))
            .into());
        }
        // The new array has storage of its own, with empty value-local caches, and its shards refer to the same device
        // buffers. The parts need no validation because they come from an array that was already validated.
        crate::telemetry::array_constructed();
        Ok(Array {
            parts: Arc::new(ArrayParts::new(
                Arc::clone(&self.actual_type),
                Arc::clone(&self.sharding),
                self.shards.to_vec(),
                Arc::clone(&self.shard_index_by_device),
                Arc::clone(&self.mesh),
            )),
            execution_fence: self.execution_fence.clone(),
            send_fence: self.send_fence.clone(),
            session: session.clone(),
        })
    }

    /// Returns this retained array with the identities of its type renamed by `renaming`.
    pub(crate) fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayType as CoreType>::Identity>,
    ) -> Result<Self, TypeError> {
        Ok(Self { declared_type: Arc::new(self.declared_type.rename_identities(renaming)?), ..self.clone() })
    }
}

// The shards have no useful `Debug` representation, so retained arrays are described by their type and by the
// identity of the array storage that they were created from, which is what distinguishes them.
impl Debug for RetainedArray {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RetainedArray")
            .field("type", &self.declared_type)
            .field("storage", &self.identity.as_ptr())
            .finish()
    }
}

// Retained arrays are identified by the array storage that they were created from and by their type. The identity
// of that storage stays reserved for as long as any of them holds it, so equal identities always mean the same array.
impl PartialEq for RetainedArray {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.identity, &other.identity) && self.declared_type == other.declared_type
    }
}

impl Eq for RetainedArray {}

impl Hash for RetainedArray {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.identity.as_ptr().hash(state);
        self.declared_type.hash(state);
    }
}

impl Display for RetainedArray {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "session:{}", self.declared_type)
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
    use std::sync::Arc;

    use pretty_assertions::assert_eq;
    use ryft_core::{ArrayType, DataType, TypeError, TypeIdentityRenaming, Typed};

    use crate::arrays::Array;
    use crate::sessions::tests::{domain_mesh, replicated_scalar_type, session_with_client, shares_effect_scope};
    use crate::tests::{execution_client, hash_of, other_cpu_client, values_from_bytes};

    use super::RetainedArray;

    #[test]
    fn test_retained_array_new() {
        let client = execution_client();
        let session = session_with_client(client);
        let array = Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 1f32.to_ne_bytes())
            .unwrap();

        // Retaining one array twice yields equal handles, and the handles keep the array alive after the array itself
        // is dropped.
        let handle = RetainedArray::new(&array);
        let duplicate = RetainedArray::new(&array);
        assert_eq!(duplicate, handle);
        drop((array, duplicate));
        let resolved = handle.to_array(&session).unwrap();
        assert_eq!(values_from_bytes::<f32>(&resolved.copy_to_host().unwrap()), vec![1f32]);

        // An array that was resolved from a handle has storage of its own, so retaining it yields another handle.
        assert_ne!(RetainedArray::new(&resolved), handle);
    }

    #[test]
    fn test_retained_array_to_array() {
        let client = execution_client();
        let session = session_with_client(client);
        let array = Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 1f32.to_ne_bytes())
            .unwrap();
        let handle = RetainedArray::new(&array);

        // The resolved array shares the retained storage and belongs to the resolving handle.
        let fork = session.with_new_effect_scope();
        let resolved = handle.to_array(&fork).unwrap();
        assert_eq!(resolved, array);
        assert!(shares_effect_scope(resolved.session(), &fork));

        // Every session of the client that owns the array resolves the handle, and the array outlives its session.
        let other_session = session_with_client(client);
        drop((array, resolved, fork, session));
        let resolved = handle.to_array(&other_session).unwrap();
        assert!(resolved.session().is_same_session(&other_session));
        assert_eq!(values_from_bytes::<f32>(&resolved.copy_to_host().unwrap()), vec![1f32]);

        // Sessions of another client cannot use the buffers of the array.
        let foreign_session = session_with_client(other_cpu_client(1));
        assert_eq!(
            handle.to_array(&foreign_session),
            Err(TypeError::invalid(format!(
                "session constant `{handle}` retains an array of a different PJRT client than the client of this \
                 session",
            ))
            .into()),
        );
    }

    #[test]
    fn test_retained_array_rename_type_identities() {
        // Renaming keeps the storage of a handle and only renames its type.
        let client = execution_client();
        let session = session_with_client(client);
        let array = Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 1f32.to_ne_bytes())
            .unwrap();
        let handle = RetainedArray::new(&array);
        let renamed = handle.rename_type_identities(&TypeIdentityRenaming::new()).unwrap();
        assert_eq!(renamed, handle);
        assert!(Arc::ptr_eq(&renamed.shards, &handle.shards));
    }

    #[test]
    fn test_retained_array_eq() {
        let client = execution_client();
        let session = session_with_client(client);
        let other_session = session_with_client(client);
        let first = Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 1f32.to_ne_bytes())
            .unwrap();
        let second =
            Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 2f32.to_ne_bytes())
                .unwrap();
        let other =
            Array::from_host_buffer(&other_session, ArrayType::scalar(DataType::F32), None, None, 1f32.to_ne_bytes())
                .unwrap();
        let handle = RetainedArray::new(&first);

        // Handles are equal when they refer to the same retained array of the same session with the same type.
        assert_eq!(handle, handle.clone());
        assert_eq!(handle, RetainedArray::new(&first));
        assert_ne!(handle, RetainedArray::new(&second));
        assert_ne!(handle, RetainedArray::new(&other));
    }

    #[test]
    fn test_retained_array_hash() {
        let client = execution_client();
        let session = session_with_client(client);
        let first = Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 1f32.to_ne_bytes())
            .unwrap();
        let second =
            Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 2f32.to_ne_bytes())
                .unwrap();
        let handle = RetainedArray::new(&first);
        let other = RetainedArray::new(&second);
        assert_eq!(hash_of(&handle), hash_of(&RetainedArray::new(&first)));
        assert_ne!(hash_of(&handle), hash_of(&other));
        let constants = HashMap::from([(handle.clone(), "first"), (other.clone(), "second")]);
        assert_eq!(constants.get(&RetainedArray::new(&first)), Some(&"first"));
        assert_eq!(constants.get(&other), Some(&"second"));
    }

    #[test]
    fn test_retained_array_display() {
        let client = execution_client();
        let session = session_with_client(client);
        let array = Array::from_host_buffer(&session, ArrayType::scalar(DataType::F32), None, None, 1f32.to_ne_bytes())
            .unwrap();
        let handle = RetainedArray::new(&array);
        assert_eq!(handle.to_string(), "session:f32[][sharding={mesh<['x'=1:auto]>, []}]");
    }

    #[test]
    fn test_retained_array_type() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = session_with_client(client);
        let array = Array::from_host_buffer(
            &session,
            replicated_scalar_type(&mesh, DataType::F32),
            None,
            None,
            1f32.to_ne_bytes(),
        )
        .unwrap();
        assert_eq!(RetainedArray::new(&array).r#type().into_owned(), replicated_scalar_type(&mesh, DataType::F32));
    }
}
