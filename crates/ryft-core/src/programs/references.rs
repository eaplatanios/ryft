//! Structural reference types, operation-level reference semantics descriptors, and the eager reference holder.
//!
//! References give programs mutable state with explicit runtime identity. This module owns the program-facing
//! surface of that feature:
//!
//! - [`ReferenceType`] is the structural [`Type`] of a reference. It carries only referent metadata: runtime
//!   identity belongs to [`Reference`] and never participates in structural equality, hashing, or retained-program
//!   specialization.
//! - [`Reference`] is the generic cloneable identity-bearing root holder. Clones alias one holder, reads return
//!   immutable value snapshots, replacement is atomic and preserves the exact declared referent type, handle-local
//!   type-identity mappings reconstruct values bidirectionally at the holder boundary, and a consuming freeze
//!   invalidates the complete alias family. Array indexing and slicing live in the array-owned
//!   [`ArrayReference`](crate::arrays::ArrayReference) wrapper rather than this generic layer.
//! - [`ReferenceOperationSemantics`], together with [`ReferenceOutputSemantics`], [`ReferenceInputAccess`], and
//!   [`ReferenceAccessMode`], is the descriptor vocabulary through which an operation declares which of its outputs
//!   define new reference roots or alias input roots and how each reference input is accessed. Descriptors speak
//!   only in operation-local operand/result index space; program-level analysis resolves them to canonical roots.
//! - [`ReferenceError`] reports failed accesses to an eager holder.
//!
//! References are second-class program values: they may appear as instruction intermediates, inputs, or captures,
//! but never as public program outputs or in ordinary numeric use.
//! [`ReferenceAnalysis`](crate::arrays::ReferenceAnalysis) enforces that static root, lifetime, and second-class
//! boundary contract before eager program replay or later discharge. The reference operations themselves —
//! allocation, snapshot read, replacement, ordered additive update, and consuming freeze — are independent operation
//! payloads defined in [`crate::operations::references`] rather than one homogeneous reference operation family, and
//! binding-level sugar such as `write` is defined over `swap` instead of adding IR operations.
//!
//! Array reference views are statically validated and eliminated by canonical slice, reshape, and update-slice
//! discharge. Local references compose with the supported program transforms only after that discharge. External
//! holders use guarded state transactions around compiled execution. Views remain root-local across structured-region
//! boundaries: a region must receive the root handle and recreate any index or slice view inside its own body.

// TODO(eaplatanios): Review this module.

use std::borrow::{Borrow, Cow};
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, MutexGuard};

use ryft_macros::Parameter;
use thiserror::Error;

use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::programs::identities::{TypeIdentityPosition, TypeIdentityRenaming};
use crate::programs::types::{Type, TypeError, TypeRefinements, Typed};
use crate::programs::values::Value;

/// Error produced while accessing a [`Reference`]'s eager holder.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
#[non_exhaustive]
pub enum ReferenceError {
    /// The reference and its complete alias family were invalidated by a consuming freeze.
    #[error("reference is frozen")]
    Frozen,

    /// The holder's synchronization primitive was poisoned by a panic during an earlier access.
    #[error("reference holder is poisoned")]
    Poisoned,

    /// The holder was invalidated after a stateful backend invocation crossed its irreversible execution boundary.
    #[error("reference state is poisoned: {reason}")]
    ExecutionPoisoned {
        /// Backend-owned reason the state can no longer be used safely.
        reason: String,
    },

    /// A guarded transaction attempted an operation that is incompatible with the holder's current state.
    #[error("reference holder already has an extracted transaction value")]
    TransactionInProgress,

    /// A prepared replacement belongs to a different holder transaction.
    #[error("prepared reference value belongs to a different holder")]
    TransactionHolderMismatch,

    /// A replacement or update result did not preserve the holder's exact declared referent type.
    #[error("reference value type `{actual}` must exactly match declared referent type `{expected}`")]
    ReferentTypeMismatch {
        /// Exact declared referent type.
        expected: String,

        /// Actual replacement or update-result type.
        actual: String,
    },

    /// A handle-local metadata mapping could not reconstruct a value crossing the shared-holder boundary.
    #[error("reference value reconstruction failed: {message}")]
    ValueReconstruction {
        /// Underlying value-family reconstruction diagnostic.
        message: String,
    },
}

/// Meaning of one reference input access performed by an operation.
///
/// The modes form a deliberately small vocabulary mirroring JAX's `ReadEffect`/`WriteEffect`/`AccumEffect` reference
/// effects, plus consumption as a lifetime event. There is intentionally no `ReadWrite` mode: `Write` asserts that
/// the access writes, not that no read occurs (see [`ReferenceAccessMode::Write`]), which matches JAX classifying
/// `swap` as a plain write.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ReferenceAccessMode {
    /// Reads the current state without replacing it.
    Read,

    /// Replaces the current state. `Write` is an over-approximation floor: it asserts that the access *writes*, not
    /// that no read occurs. An operation may still observe the previous state through its results (e.g.,
    /// `reference_swap` returns the old value), so generic analyses must treat a `Write` conservatively as possibly
    /// reading prior state through its results — in particular, any future dead-store elimination on a state chain
    /// must not remove an earlier write based on this mode alone. Whether a specific write actually reads is an
    /// operation-specific question (e.g., kernel lowering of `reference_swap` knows its own old-value result and can
    /// emit a plain store when that result is dead); this descriptor deliberately carries no access-to-result
    /// mapping until a generic analysis genuinely needs one.
    Write,

    /// Combines an update with the current state as an *ordered* additive accumulation. Accumulation stays distinct
    /// from [`Write`](ReferenceAccessMode::Write) because it is linear in the update operand and therefore
    /// transposable, unlike a replacement. It carries no commutativity or atomicity promise: same-root accumulations
    /// execute in program order (floating-point addition cannot generally be reordered while preserving results),
    /// and atomic/commutative accumulation is a separate future semantics.
    Accumulate,

    /// Consumes the root: after this access the root and its entire alias family are invalid. Consumption is a
    /// lifetime event, not a memory-access flavor — `freeze` is the consuming access that also returns the final
    /// value, and a future `free_reference` would consume without producing a result.
    Consume,
}

/// Kind of one root-preserving reference alias edge.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ReferenceAliasKind {
    /// The alias preserves the input handle's exact referent type and mapping.
    Identity,

    /// The alias adds operation-owned view metadata validated by an array-specialized analysis entry.
    View,
}

/// Reference classification of one operation output: either a fresh canonical root or an alias of one input's root.
///
/// The two cases are mutually exclusive by construction, so an output can never be declared as both a new root and
/// an alias. [`Alias`](ReferenceOutputSemantics::Alias) carries exactly one `input_index` on purpose: every reference
/// operand must resolve to exactly one canonical root, so multi-source aliases (e.g., a hypothetical
/// `select_reference(a, b)`) are structurally unrepresentable rather than merely rejected. [`ReferenceAliasKind`]
/// distinguishes an identity-preserving edge from an operation-owned view edge. Generic root analysis needs only that
/// marker; array-specialized analysis obtains and validates the exact coordinate transform from its operation family.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ReferenceOutputSemantics {
    /// The output allocates a fresh resource and defines a new canonical root.
    NewRoot {
        /// Operation output index defining the new root.
        output_index: usize,
    },

    /// The output aliases the canonical root of one reference input.
    Alias {
        /// Operation output index producing the alias.
        output_index: usize,

        /// Operation input index whose canonical root is preserved.
        input_index: usize,

        /// Whether this edge preserves the exact handle or adds an operation-owned view mapping.
        kind: ReferenceAliasKind,
    },
}

impl ReferenceOutputSemantics {
    /// Returns the operation output index this classification applies to.
    #[inline]
    pub const fn output_index(self) -> usize {
        match self {
            Self::NewRoot { output_index } | Self::Alias { output_index, .. } => output_index,
        }
    }
}

/// Reference access performed through one operation input.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReferenceInputAccess {
    /// Operation input index being accessed.
    input_index: usize,

    /// Semantic mode of the access.
    mode: ReferenceAccessMode,
}

impl ReferenceInputAccess {
    /// Creates an access of `mode` through `input_index`.
    #[inline]
    pub const fn new(input_index: usize, mode: ReferenceAccessMode) -> Self {
        Self { input_index, mode }
    }

    /// Returns the operation input index being accessed.
    #[inline]
    pub const fn input_index(self) -> usize {
        self.input_index
    }

    /// Returns the semantic mode of this access.
    #[inline]
    pub const fn mode(self) -> ReferenceAccessMode {
        self.mode
    }
}

/// Operation-local reference semantics: output root/alias classification plus input accesses, expressed in
/// operand/result index space.
///
/// All indices are *operation-local operand and result positions*, never resource identifiers. Program-level
/// analysis later resolves them to canonical roots (an entry input, a capture, or an allocation instruction), so the
/// descriptor intentionally contains no runtime resource IDs. The empty default describes operations that neither
/// create, alias, nor access references.
///
/// # Examples
///
/// The array-reference vocabulary declares the following semantics:
///
/// ```text
/// new_reference(x) -> r
///     outputs  = [NewRoot { output_index: 0 }]
///     accesses = []
///
/// reference_read(r) -> x
///     outputs  = []
///     accesses = [ReferenceInputAccess { input_index: 0, mode: Read }]
///
/// reference_swap(r, x) -> old
///     outputs  = []
///     accesses = [ReferenceInputAccess { input_index: 0, mode: Write }]
///
/// reference_add_update(r, x) -> ()
///     outputs  = []
///     accesses = [ReferenceInputAccess { input_index: 0, mode: Accumulate }]
///
/// freeze_reference(r) -> x
///     outputs  = []
///     accesses = [ReferenceInputAccess { input_index: 0, mode: Consume }]
///
/// reference_index(r, axis, index) -> view
/// reference_slice(r, axes) -> view
///     outputs  = [Alias { output_index: 0, input_index: 0, kind: View }]
///     accesses = []
/// ```
///
/// Note that `reference_swap` declares `Write` even though its result observes the previous value: `Write` does not
/// assert the absence of a read (refer to [`ReferenceAccessMode::Write`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReferenceOperationSemantics {
    /// Reference classifications of operation outputs.
    outputs: Vec<ReferenceOutputSemantics>,

    /// State accesses through reference inputs.
    accesses: Vec<ReferenceInputAccess>,
}

// Shared empty descriptor returned by `ReferenceOperationSemantics::empty` so that the `Operation` trait default can
// hand out a borrow without allocating (`Vec::new` is `const`, so this static needs no lazy initialization).
static EMPTY_REFERENCE_OPERATION_SEMANTICS: ReferenceOperationSemantics =
    ReferenceOperationSemantics { outputs: Vec::new(), accesses: Vec::new() };

impl ReferenceOperationSemantics {
    /// Creates a reference semantics descriptor from its ordered components.
    ///
    /// # Panics
    ///
    /// Panics when one output index receives two classifications or one input index receives two accesses. These are
    /// operation-author contract violations: the documented mutual exclusivity of [`ReferenceOutputSemantics`] holds
    /// only if each operand/result position appears at most once, and program-level analysis trusts that invariant.
    /// Index ranges cannot be checked here because the descriptor carries no arity information; program-level
    /// reference analysis validates them against each instruction's actual operand/result arity.
    pub fn new(outputs: Vec<ReferenceOutputSemantics>, accesses: Vec<ReferenceInputAccess>) -> Self {
        for (index, output) in outputs.iter().enumerate() {
            let output_index = output.output_index();
            assert!(
                outputs[..index].iter().all(|previous| previous.output_index() != output_index),
                "output {output_index} received two reference classifications",
            );
        }
        for (index, access) in accesses.iter().enumerate() {
            let input_index = access.input_index();
            assert!(
                accesses[..index].iter().all(|previous| previous.input_index() != input_index),
                "input {input_index} received two reference accesses",
            );
        }
        Self { outputs, accesses }
    }

    /// Returns the shared empty descriptor used by operations that neither create, alias, nor access references.
    #[inline]
    pub fn empty() -> &'static Self {
        &EMPTY_REFERENCE_OPERATION_SEMANTICS
    }

    /// Returns whether this descriptor declares no reference outputs and no reference accesses.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.outputs.is_empty() && self.accesses.is_empty()
    }

    /// Returns output reference classifications in deterministic operation-defined order.
    #[inline]
    pub fn outputs(&self) -> &[ReferenceOutputSemantics] {
        self.outputs.as_slice()
    }

    /// Returns reference accesses in deterministic operation-defined order.
    #[inline]
    pub fn accesses(&self) -> &[ReferenceInputAccess] {
        self.accesses.as_slice()
    }
}

/// Structural [`Type`] of a reference to a value whose type is `T`.
///
/// A reference type contains only referent metadata. Runtime resource identity belongs to [`Reference`] and therefore
/// does not affect structural equality, hashing, or retained-program specialization. Compatibility is exact because a
/// reference cannot implicitly broadcast or promote its storage, while refinement and identity handling delegate to
/// the referent type. Exactness deliberately spans the referent's optional layout, sharding, and memory metadata as
/// well: the first XLA slice requires exact physical referent compatibility for mutation, so a metadata-tolerant
/// relation would overpromise. Revisit this if a compatibility consumer ever needs the tolerant reading.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct ReferenceType<T: Type> {
    /// Structural type of the referenced value.
    referent: T,
}

impl<T: Type> ReferenceType<T> {
    /// Creates a reference type for `referent`.
    #[inline]
    pub fn new(referent: T) -> Self {
        Self { referent }
    }

    /// Returns the structural type of the referenced value.
    #[inline]
    pub fn referent(&self) -> &T {
        &self.referent
    }
}

impl<T: Type> Display for ReferenceType<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "ref<{}>", self.referent)
    }
}

impl<T: Type> Type for ReferenceType<T> {
    type Identity = T::Identity;
    type Refinements = ReferenceTypeRefinements<T>;

    #[inline]
    fn identities(&self) -> impl Iterator<Item = (TypeIdentityPosition, &Self::Identity)> {
        self.referent.identities()
    }

    fn derive_identity_renaming(
        declared: &[Self],
        actual: &[Self],
    ) -> Result<TypeIdentityRenaming<Self::Identity>, TypeError> {
        let declared = declared.iter().map(|r#type| r#type.referent.clone()).collect::<Vec<_>>();
        let actual = actual.iter().map(|r#type| r#type.referent.clone()).collect::<Vec<_>>();
        T::derive_identity_renaming(&declared, &actual)
    }

    #[inline]
    fn rename_identities(&self, renaming: &TypeIdentityRenaming<Self::Identity>) -> Result<Self, TypeError> {
        Ok(Self::new(self.referent.rename_identities(renaming)?))
    }

    #[inline]
    fn is_compatible_with(&self, other: &Self) -> bool {
        self == other
    }

    #[inline]
    fn is_refined_by(&self, other: &Self) -> bool {
        self.referent.is_refined_by(&other.referent)
    }

    #[inline]
    fn is_scalar(&self) -> bool {
        false
    }

    #[inline]
    fn is_complex(&self) -> bool {
        false
    }

    #[inline]
    fn is_reference(&self) -> bool {
        true
    }
}

/// Cross-occurrence refinements established for a complete [`ReferenceType`] signature.
#[derive(Clone, Debug)]
pub struct ReferenceTypeRefinements<T: Type> {
    /// Referent refinement state shared across every reference in the signature.
    referents: T::Refinements,
}

impl<T: Type> Default for ReferenceTypeRefinements<T> {
    #[inline]
    fn default() -> Self {
        Self { referents: T::Refinements::default() }
    }
}

impl<T: Type> TypeRefinements<ReferenceType<T>> for ReferenceTypeRefinements<T> {
    fn establish<D: IntoIterator, A: IntoIterator>(declared: D, actual: A) -> Result<Self, TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<ReferenceType<T>>,
        A::Item: Borrow<ReferenceType<T>>,
    {
        // Collecting the items is a shallow move; the referents themselves are delegated by borrow (`&T` satisfies
        // the `Borrow<T>` item bound) so no referent is ever cloned on this type-inference path.
        let declared = declared.into_iter().collect::<Vec<_>>();
        let actual = actual.into_iter().collect::<Vec<_>>();
        let declared = declared.iter().map(|r#type| &r#type.borrow().referent);
        let actual = actual.iter().map(|r#type| &r#type.borrow().referent);
        Ok(Self { referents: T::Refinements::establish(declared, actual)? })
    }

    fn validate<D: IntoIterator, A: IntoIterator>(
        &self,
        declared: D,
        actual: A,
        closed_identities: &[T::Identity],
    ) -> Result<(), TypeError>
    where
        D::IntoIter: ExactSizeIterator,
        A::IntoIter: ExactSizeIterator,
        D::Item: Borrow<ReferenceType<T>>,
        A::Item: Borrow<ReferenceType<T>>,
    {
        // Refer to the borrow-based delegation note in `establish` above.
        let declared = declared.into_iter().collect::<Vec<_>>();
        let actual = actual.into_iter().collect::<Vec<_>>();
        let declared = declared.iter().map(|r#type| &r#type.borrow().referent);
        let actual = actual.iter().map(|r#type| &r#type.borrow().referent);
        self.referents.validate(declared, actual, closed_identities)
    }
}

/// Opaque process-local identity that remains stable for the lifetime of one eager [`Reference`] holder.
///
/// The identity supports alias-identity checks and diagnostics inside one process. It carries no structural type
/// information, is never serialized into a program or compilation key, and may be reused after the last handle for
/// the original holder is dropped.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReferenceId(usize);

/// Cloneable identity-bearing holder for a referenced [`Value`].
///
/// Cloning a reference aliases the same holder. Reading clones the current value. Reference referents are expected to
/// have immutable-value clone semantics so that later replacement or update cannot change an initializer, read
/// result, or swap result retained by the caller. Array IR admits only array referents, whose SSA/copy-on-write
/// semantics satisfy this requirement; resource handles such as references are not valid referents there.
///
/// A directly created eager handle has no program-owned scope: it remains valid until an explicit freeze or until the
/// last handle in its alias family is dropped. Statically validated local program roots that are never frozen are
/// implicitly discarded when their nonescaping interpretation environment is released.
pub struct Reference<V: Value> {
    /// Shared mutable holder whose allocation defines this reference's runtime identity.
    holder: Arc<Mutex<ReferenceState<V>>>,

    /// Handle-local structural referent type.
    r#type: ReferenceType<V::Type>,

    /// Structural referent type of values stored in the shared holder.
    root_type: V::Type,

    /// Identity mapping applied when a stored value crosses into this handle.
    root_to_handle: TypeIdentityRenaming<<V::Type as Type>::Identity>,

    /// Inverse identity mapping applied before a handle-local value enters the shared holder.
    handle_to_root: TypeIdentityRenaming<<V::Type as Type>::Identity>,
}

/// Lifecycle state shared by every handle in one reference alias family.
enum ReferenceState<V: Value> {
    /// Live reference containing its current immutable value snapshot.
    Ready(V),

    /// Value temporarily extracted by a synchronous backend transaction holding this holder's mutex.
    Taken,

    /// Value that may have been consumed by an irreversible failed backend invocation.
    ExecutionPoisoned(Arc<str>),

    /// Consumed reference whose value was returned by `freeze`.
    Frozen,
}

impl<V: Value> Reference<V> {
    /// Creates a new independent reference initialized with `value`.
    pub fn new(value: V) -> Self {
        let root_type = value.r#type().into_owned();
        let r#type = ReferenceType::new(root_type.clone());
        Self {
            holder: Arc::new(Mutex::new(ReferenceState::Ready(value))),
            r#type,
            root_type,
            root_to_handle: TypeIdentityRenaming::new(),
            handle_to_root: TypeIdentityRenaming::new(),
        }
    }

    /// Returns this holder's process-local identity, which remains stable while any alias is alive.
    #[inline]
    pub fn id(&self) -> ReferenceId {
        ReferenceId(Arc::as_ptr(&self.holder) as usize)
    }

    /// Returns whether this handle exposes the holder's root type without a handle-local identity mapping.
    #[doc(hidden)]
    pub fn is_root_handle(&self) -> bool {
        self.root_to_handle.is_identity() && self.handle_to_root.is_identity()
    }

    /// Locks this holder for one synchronous stateful backend transaction.
    #[doc(hidden)]
    pub fn lock(&self) -> Result<ReferenceGuard<'_, V>, ReferenceError> {
        Ok(ReferenceGuard {
            id: self.id(),
            r#type: &self.r#type,
            root_type: &self.root_type,
            root_to_handle: &self.root_to_handle,
            handle_to_root: &self.handle_to_root,
            state: self.holder.lock().map_err(|_| ReferenceError::Poisoned)?,
        })
    }

    /// Returns a clone of the currently stored value, which is an immutable snapshot for a valid reference referent.
    pub fn read(&self) -> Result<V, ReferenceError> {
        match &*self.holder.lock().map_err(|_| ReferenceError::Poisoned)? {
            ReferenceState::Ready(value) => value
                .rename_type_identities(&self.root_to_handle)
                .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() }),
            ReferenceState::Frozen => Err(ReferenceError::Frozen),
            ReferenceState::ExecutionPoisoned(reason) => {
                Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() })
            }
            ReferenceState::Taken => Err(ReferenceError::TransactionInProgress),
        }
    }

    /// Atomically replaces the stored value and returns the previous referent value.
    ///
    /// The replacement must have exactly the declared referent type. A rejected replacement leaves the live holder
    /// unchanged.
    pub fn swap(&self, replacement: V) -> Result<V, ReferenceError> {
        let mut state = self.holder.lock().map_err(|_| ReferenceError::Poisoned)?;
        match &mut *state {
            ReferenceState::Ready(current) => {
                self.validate_referent_type(&replacement)?;
                let stored_replacement = replacement
                    .rename_type_identities(&self.handle_to_root)
                    .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() })?;
                self.validate_root_type(&stored_replacement)?;
                let old = current
                    .rename_type_identities(&self.root_to_handle)
                    .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() })?;
                *current = stored_replacement;
                Ok(old)
            }
            ReferenceState::Frozen => Err(ReferenceError::Frozen),
            ReferenceState::ExecutionPoisoned(reason) => {
                Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() })
            }
            ReferenceState::Taken => Err(ReferenceError::TransactionInProgress),
        }
    }

    /// Atomically computes and installs an updated value while retaining the old value on every failure.
    ///
    /// This crate-visible primitive keeps value-family-specific update logic (such as array addition) outside the
    /// generic holder while ensuring no other access can interleave between reading the old state and installing the
    /// new one.
    pub(crate) fn update_with(&self, update: impl FnOnce(&V) -> Result<V, ProgramError>) -> Result<(), ProgramError> {
        self.update_with_result(|current| Ok((update(current)?, ())))
    }

    /// Atomically maps this handle's current value to a replacement and an operation result.
    ///
    /// Both handle-local reconstruction directions complete before the shared state is committed, so every failure
    /// leaves the live holder unchanged.
    pub(crate) fn update_with_result<R>(
        &self,
        update: impl FnOnce(&V) -> Result<(V, R), ProgramError>,
    ) -> Result<R, ProgramError> {
        let mut state = self.holder.lock().map_err(|_| ProgramError::custom(ReferenceError::Poisoned))?;
        match &mut *state {
            ReferenceState::Ready(current) => {
                let local = current.rename_type_identities(&self.root_to_handle).map_err(|error| {
                    ProgramError::custom(ReferenceError::ValueReconstruction { message: error.to_string() })
                })?;
                let (updated, result) = update(&local)?;
                self.validate_referent_type(&updated).map_err(ProgramError::custom)?;
                let stored = updated.rename_type_identities(&self.handle_to_root).map_err(|error| {
                    ProgramError::custom(ReferenceError::ValueReconstruction { message: error.to_string() })
                })?;
                self.validate_root_type(&stored).map_err(ProgramError::custom)?;
                *current = stored;
                Ok(result)
            }
            ReferenceState::Frozen => Err(ProgramError::custom(ReferenceError::Frozen)),
            ReferenceState::ExecutionPoisoned(reason) => {
                Err(ProgramError::custom(ReferenceError::ExecutionPoisoned { reason: reason.to_string() }))
            }
            ReferenceState::Taken => Err(ProgramError::custom(ReferenceError::TransactionInProgress)),
        }
    }

    /// Consumes this reference's current value and invalidates every handle in its alias family.
    pub fn freeze(&self) -> Result<V, ReferenceError> {
        let mut state = self.holder.lock().map_err(|_| ReferenceError::Poisoned)?;
        match &*state {
            ReferenceState::Ready(value) => {
                let value = value
                    .rename_type_identities(&self.root_to_handle)
                    .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() })?;
                *state = ReferenceState::Frozen;
                Ok(value)
            }
            ReferenceState::Frozen => Err(ReferenceError::Frozen),
            ReferenceState::ExecutionPoisoned(reason) => {
                Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() })
            }
            ReferenceState::Taken => Err(ReferenceError::TransactionInProgress),
        }
    }

    /// Returns a handle-local identity-renamed view of this same shared holder.
    pub(crate) fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<V::Type as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        let current_type = self.r#type.referent();
        let renamed_type = current_type.rename_identities(renaming)?;
        let inverse_step =
            V::Type::derive_identity_renaming(std::slice::from_ref(&renamed_type), std::slice::from_ref(current_type))?;
        let root_to_handle = Self::compose_renamings(
            &self.root_to_handle,
            renaming,
            self.root_type.identities().map(|(_, identity)| identity),
        )?;
        let handle_to_root = Self::compose_renamings(
            &inverse_step,
            &self.handle_to_root,
            renamed_type.identities().map(|(_, identity)| identity),
        )?;
        if self.root_type.rename_identities(&root_to_handle)? != renamed_type
            || renamed_type.rename_identities(&handle_to_root)? != self.root_type
        {
            return Err(TypeError::invalid(
                "reference identity renaming must admit an exact bidirectional value reconstruction",
            ));
        }
        Ok(Self {
            holder: self.holder.clone(),
            r#type: ReferenceType::new(renamed_type),
            root_type: self.root_type.clone(),
            root_to_handle,
            handle_to_root,
        })
    }

    /// Returns whether this holder is still live without reading or changing its value.
    pub(crate) fn validate_live(&self) -> Result<(), ReferenceError> {
        match &*self.holder.lock().map_err(|_| ReferenceError::Poisoned)? {
            ReferenceState::Ready(_) => Ok(()),
            ReferenceState::Frozen => Err(ReferenceError::Frozen),
            ReferenceState::ExecutionPoisoned(reason) => {
                Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() })
            }
            ReferenceState::Taken => Err(ReferenceError::TransactionInProgress),
        }
    }

    /// Composes two simultaneous identity mappings over the provided source identities.
    fn compose_renamings<'a>(
        first: &TypeIdentityRenaming<<V::Type as Type>::Identity>,
        second: &TypeIdentityRenaming<<V::Type as Type>::Identity>,
        sources: impl Iterator<Item = &'a <V::Type as Type>::Identity>,
    ) -> Result<TypeIdentityRenaming<<V::Type as Type>::Identity>, TypeError>
    where
        <V::Type as Type>::Identity: 'a,
    {
        let mut result = TypeIdentityRenaming::new();
        for source in sources {
            result.insert(source.clone(), second.rename(&first.rename(source)))?;
        }
        Ok(result)
    }

    /// Validates that `value` preserves this holder's exact declared referent type.
    fn validate_referent_type(&self, value: &V) -> Result<(), ReferenceError> {
        let actual = value.r#type();
        if actual.as_ref() == self.r#type.referent() {
            return Ok(());
        }
        Err(ReferenceError::ReferentTypeMismatch {
            expected: self.r#type.referent().to_string(),
            actual: actual.to_string(),
        })
    }

    /// Validates the exact value type stored behind every handle-local mapping.
    fn validate_root_type(&self, value: &V) -> Result<(), ReferenceError> {
        let actual = value.r#type();
        if actual.as_ref() == &self.root_type {
            return Ok(());
        }
        Err(ReferenceError::ReferentTypeMismatch { expected: self.root_type.to_string(), actual: actual.to_string() })
    }
}

/// Exclusive holder guard used by synchronous stateful compilation backends.
///
/// A backend may extract the current value, but must then either install a type-compatible replacement or poison the
/// holder before dropping the guard. Dropping a guard with an extracted value poisons the holder defensively.
#[doc(hidden)]
pub struct ReferenceGuard<'a, V: Value> {
    /// Identity of the locked shared holder.
    id: ReferenceId,

    /// Handle-local referent type.
    r#type: &'a ReferenceType<V::Type>,

    /// Structural type stored by the shared root holder.
    root_type: &'a V::Type,

    /// Mapping applied when a root value crosses into this handle.
    root_to_handle: &'a TypeIdentityRenaming<<V::Type as Type>::Identity>,

    /// Mapping applied when a handle-local value enters the root holder.
    handle_to_root: &'a TypeIdentityRenaming<<V::Type as Type>::Identity>,

    /// Locked holder lifecycle state.
    state: MutexGuard<'a, ReferenceState<V>>,
}

/// Root-normalized holder value whose fallible reconstruction and type validation have completed.
#[doc(hidden)]
pub struct PreparedReferenceValue<V: Value> {
    /// Identity of the holder against which this value was prepared.
    reference_id: ReferenceId,

    /// Value represented in the shared root holder's type identity space.
    value: V,
}

impl<V: Value> ReferenceGuard<'_, V> {
    /// Returns a handle-local immutable snapshot without extracting holder state.
    pub fn snapshot(&self) -> Result<V, ReferenceError> {
        match &*self.state {
            ReferenceState::Ready(value) => value
                .rename_type_identities(self.root_to_handle)
                .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() }),
            ReferenceState::Frozen => Err(ReferenceError::Frozen),
            ReferenceState::ExecutionPoisoned(reason) => {
                Err(ReferenceError::ExecutionPoisoned { reason: reason.to_string() })
            }
            ReferenceState::Taken => Err(ReferenceError::TransactionInProgress),
        }
    }

    /// Extracts the handle-local current value for a potentially donating backend invocation.
    pub fn take(&mut self) -> Result<V, ReferenceError> {
        let local = self.snapshot()?;
        match std::mem::replace(&mut *self.state, ReferenceState::Taken) {
            ReferenceState::Ready(_) => Ok(local),
            state => {
                *self.state = state;
                Err(ReferenceError::TransactionInProgress)
            }
        }
    }

    /// Validates a prospective restored or installed value without changing holder state.
    pub fn prepare(&self, value: V) -> Result<PreparedReferenceValue<V>, ReferenceError> {
        let actual = value.r#type();
        if actual.as_ref() != self.r#type.referent() {
            return Err(ReferenceError::ReferentTypeMismatch {
                expected: self.r#type.referent().to_string(),
                actual: actual.to_string(),
            });
        }
        let stored = value
            .rename_type_identities(self.handle_to_root)
            .map_err(|error| ReferenceError::ValueReconstruction { message: error.to_string() })?;
        let actual = stored.r#type();
        if actual.as_ref() != self.root_type {
            return Err(ReferenceError::ReferentTypeMismatch {
                expected: self.root_type.to_string(),
                actual: actual.to_string(),
            });
        }
        Ok(PreparedReferenceValue { reference_id: self.id, value: stored })
    }

    /// Validates that `value` was prepared against this exact holder.
    pub fn accepts(&self, value: &PreparedReferenceValue<V>) -> Result<(), ReferenceError> {
        if value.reference_id == self.id { Ok(()) } else { Err(ReferenceError::TransactionHolderMismatch) }
    }

    /// Installs a value whose reconstruction and type checks completed through [`Self::prepare`].
    ///
    /// Installation fails when `value` was prepared for another holder or this guard does not own an extracted value.
    pub fn install(&mut self, value: PreparedReferenceValue<V>) -> Result<(), ReferenceError> {
        self.accepts(&value)?;
        if !matches!(*self.state, ReferenceState::Taken) {
            return Err(ReferenceError::TransactionInProgress);
        }
        *self.state = ReferenceState::Ready(value.value);
        Ok(())
    }

    /// Invalidates an extracted value after an irreversible backend failure.
    pub fn poison(&mut self, reason: impl Into<Arc<str>>) -> Result<(), ReferenceError> {
        if !matches!(*self.state, ReferenceState::Taken) {
            return Err(ReferenceError::TransactionInProgress);
        }
        *self.state = ReferenceState::ExecutionPoisoned(reason.into());
        Ok(())
    }
}

impl<V: Value> Drop for ReferenceGuard<'_, V> {
    fn drop(&mut self) {
        if matches!(*self.state, ReferenceState::Taken) {
            *self.state =
                ReferenceState::ExecutionPoisoned("stateful transaction ended without restoring state".into());
        }
    }
}

impl<V: Value> Clone for Reference<V> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            holder: self.holder.clone(),
            r#type: self.r#type.clone(),
            root_type: self.root_type.clone(),
            root_to_handle: self.root_to_handle.clone(),
            handle_to_root: self.handle_to_root.clone(),
        }
    }
}

impl<V: Value> Debug for Reference<V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Reference").field("id", &self.id()).field("type", &self.r#type).finish()
    }
}

// `Display` deliberately renders only the handle-local type: the Value rendering contract requires deterministic
// output (renderings back diagnostics, rendered-program tests, and the debug-assertions transform-cache determinism
// recheck), so the process-local holder address must not leak here. Runtime identity remains visible through `Debug`.
impl<V: Value> Display for Reference<V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.r#type, formatter)
    }
}

impl<V: Value> PartialEq for Reference<V> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.holder, &other.holder)
    }
}

impl<V: Value> Eq for Reference<V> {}

impl<V: Value> Hash for Reference<V> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.holder).hash(state);
    }
}

impl<V: Value> Parameter for Reference<V> {}

impl<V: Value> Typed for Reference<V> {
    type Type = ReferenceType<V::Type>;

    #[inline]
    fn r#type(&self) -> Cow<'_, Self::Type> {
        Cow::Borrowed(&self.r#type)
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::cell::Cell;
    use std::collections::HashMap;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayType, DataType, Dimension, DimensionBounds, DimensionError, DimensionVariable, Shape,
    };
    use crate::operations::Add;
    use crate::programs::operations::Operation;
    use crate::programs::regions::RegionInterface;

    use super::*;

    #[test]
    fn test_reference_type_delegates_identity_and_refinement_without_implicit_compatibility() {
        let declared = DimensionVariable::new("declared", DimensionBounds::positive(Some(9)).unwrap());
        let actual = DimensionVariable::new("actual", DimensionBounds::positive(Some(5)).unwrap());
        let declared_type =
            ReferenceType::new(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(declared.clone())])));
        let actual_type =
            ReferenceType::new(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(actual.clone())])));
        let renaming = ReferenceType::derive_identity_renaming(
            std::slice::from_ref(&declared_type),
            std::slice::from_ref(&actual_type),
        )
        .unwrap();
        assert_eq!(renaming.rename(&declared), actual);

        let static_two = ReferenceType::new(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)])));
        let static_three = ReferenceType::new(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)])));
        assert!(declared_type.is_refined_by(&static_two));
        assert!(!declared_type.is_compatible_with(&static_two));
        assert!(!static_two.is_compatible_with(&static_three));
        assert!(static_two.is_reference());
        assert!(!static_two.is_scalar());
        assert!(!static_two.is_complex());
        assert_eq!(static_two.to_string(), "ref<f32[2]>");
        assert_eq!(format!("{static_two:?}"), format!("ReferenceType {{ referent: {:?} }}", static_two.referent()),);
        let refinements = ReferenceTypeRefinements::establish(
            [declared_type.clone(), declared_type.clone()],
            [static_two.clone(), static_two.clone()],
        )
        .unwrap();
        assert_eq!(refinements.validate([declared_type.clone()], [static_two.clone()], &[]), Ok(()),);
        let error = ReferenceTypeRefinements::establish(
            [
                ReferenceType::new(ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(declared.clone())]),
                )),
                ReferenceType::new(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(declared)]))),
            ],
            [static_two, static_three],
        )
        .unwrap_err();
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::InputDimensionMismatch { dimension: "declared".to_string(), expected: 2, actual: 3 }),
        );
    }

    #[test]
    fn test_reference_clones_alias_one_holder_and_reads_snapshots() {
        let initial = Array::vector(vec![1.0_f32, 2.0]);
        let reference = Reference::new(initial.clone());
        let alias = reference.clone();
        let distinct = Reference::new(initial);
        assert_eq!(reference, alias);
        assert_ne!(reference, distinct);
        assert_eq!(reference.id(), alias.id());
        assert_eq!(reference.read().unwrap(), Array::vector(vec![1.0_f32, 2.0]));
        assert_eq!(reference.r#type(), alias.r#type());
        assert_eq!(reference.r#type(), distinct.r#type());
        // `Display` is deterministic and type-based (the holder address would leak nondeterminism into diagnostics
        // and program renderings); runtime identity remains visible through `Debug`.
        assert_eq!(reference.to_string(), "ref<f32[2]>");
        assert_eq!(reference.to_string(), distinct.to_string());
        assert_eq!(
            format!("{reference:?}"),
            format!("Reference {{ id: {:?}, type: {:?} }}", reference.id(), reference.r#type()),
        );

        let mut reference_type_hasher = DefaultHasher::new();
        reference.r#type().hash(&mut reference_type_hasher);
        let mut distinct_type_hasher = DefaultHasher::new();
        distinct.r#type().hash(&mut distinct_type_hasher);
        assert_eq!(reference_type_hasher.finish(), distinct_type_hasher.finish());

        let references = HashMap::from([(reference.clone(), "root")]);
        assert_eq!(references.get(&alias), Some(&"root"));
        assert_eq!(references.get(&distinct), None);
    }

    #[test]
    fn test_reference_freeze_invalidates_the_complete_alias_family() {
        let reference = Reference::new(Array::vector(vec![1.0_f32, 2.0]));
        let alias = reference.clone();
        assert_eq!(reference.freeze(), Ok(Array::vector(vec![1.0_f32, 2.0])));

        assert_eq!(alias.read(), Err(ReferenceError::Frozen));
        assert_eq!(alias.swap(Array::vector(vec![3.0_f32, 4.0])), Err(ReferenceError::Frozen));
        assert_eq!(reference.freeze(), Err(ReferenceError::Frozen));

        // A rejected update must not invoke value-family code after the shared holder has been consumed.
        let update_executed = Cell::new(false);
        let error = alias
            .update_with(|_| {
                update_executed.set(true);
                Ok(Array::vector(vec![3.0_f32, 4.0]))
            })
            .unwrap_err();
        assert!(!update_executed.get());
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_reference_rejected_replacements_and_updates_leave_the_holder_unchanged() {
        let initial = Array::vector(vec![1.0_f32, 2.0]);
        let reference = Reference::new(initial.clone());

        assert_eq!(
            reference.swap(Array::vector(vec![3.0_f32, 4.0, 5.0])),
            Err(ReferenceError::ReferentTypeMismatch { expected: "f32[2]".to_string(), actual: "f32[3]".to_string() }),
        );
        assert_eq!(reference.read(), Ok(initial.clone()));

        let update_error = ProgramError::InvalidArgument { message: "test update failed".to_string() };
        assert_eq!(reference.update_with(|_| Err(update_error.clone())), Err(update_error));
        assert_eq!(reference.read(), Ok(initial.clone()));

        let error = reference.update_with(|_| Ok(Array::vector(vec![3.0_f32, 4.0, 5.0]))).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceError>(),
            Some(&ReferenceError::ReferentTypeMismatch {
                expected: "f32[2]".to_string(),
                actual: "f32[3]".to_string(),
            }),
        );
        assert_eq!(reference.read(), Ok(initial));
    }

    #[test]
    fn test_reference_mutation_preserves_snapshots_and_independent_roots() {
        let initializer = Array::vector(vec![1.0_f32, 2.0]);
        let first = Reference::new(initializer.clone());
        let second = Reference::new(initializer.clone());
        let read_snapshot = first.read().unwrap();
        let replacement = Array::vector(vec![3.0_f32, 4.0]);
        let retained_replacement = replacement.clone();

        let swapped_snapshot = first.swap(replacement).unwrap();
        first.update_with(|current| current.add(&Array::vector(vec![10.0_f32, 20.0]))).unwrap();
        assert_eq!(second.read(), Ok(initializer.clone()));
        assert_eq!(second.swap(Array::vector(vec![5.0_f32, 6.0])), Ok(initializer.clone()));

        assert_eq!(initializer, Array::vector(vec![1.0_f32, 2.0]));
        assert_eq!(read_snapshot, Array::vector(vec![1.0_f32, 2.0]));
        assert_eq!(swapped_snapshot, Array::vector(vec![1.0_f32, 2.0]));
        assert_eq!(retained_replacement, Array::vector(vec![3.0_f32, 4.0]));
        assert_eq!(first.read(), Ok(Array::vector(vec![13.0_f32, 24.0])));
        assert_eq!(second.read(), Ok(Array::vector(vec![5.0_f32, 6.0])));
    }

    #[test]
    fn test_reference_read_reports_a_poisoned_holder() {
        let reference = Reference::new(Array::scalar(1.0_f32));
        let holder = Arc::clone(&reference.holder);
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let _guard = holder.lock().unwrap();
                panic!("poison reference holder");
            }))
            .is_err(),
        );
        assert_eq!(reference.read(), Err(ReferenceError::Poisoned));
    }

    #[test]
    fn test_reference_guard_prepares_transaction_values_for_the_exact_holder() {
        let first = Reference::new(Array::scalar(1.0_f32));
        let second = Reference::new(Array::scalar(2.0_f32));
        let mut first_guard = first.lock().unwrap();
        let second_guard = second.lock().unwrap();
        assert_eq!(first_guard.take(), Ok(Array::scalar(1.0_f32)));
        let prepared = first_guard.prepare(Array::scalar(3.0_f32)).unwrap();
        assert_eq!(second_guard.accepts(&prepared), Err(ReferenceError::TransactionHolderMismatch));
        first_guard.install(prepared).unwrap();
        drop(first_guard);
        drop(second_guard);
        assert_eq!(first.read(), Ok(Array::scalar(3.0_f32)));
        assert_eq!(second.read(), Ok(Array::scalar(2.0_f32)));
    }

    #[test]
    fn test_reference_guard_poison_isolated_to_the_extracted_holder() {
        let first = Reference::new(Array::scalar(1.0_f32));
        let second = Reference::new(Array::scalar(2.0_f32));
        let mut first_guard = first.lock().unwrap();
        let second_guard = second.lock().unwrap();
        first_guard.take().unwrap();
        first_guard.poison("test execution failed").unwrap();
        drop(first_guard);
        drop(second_guard);
        assert_eq!(
            first.read(),
            Err(ReferenceError::ExecutionPoisoned { reason: "test execution failed".to_string() }),
        );
        assert_eq!(second.read(), Ok(Array::scalar(2.0_f32)));
    }

    #[test]
    fn test_dropping_reference_guard_poisons_extracted_holder() {
        let reference = Reference::new(Array::scalar(1.0_f32));
        let mut guard = reference.lock().unwrap();
        assert_eq!(guard.take(), Ok(Array::scalar(1.0_f32)));
        drop(guard);
        assert_eq!(
            reference.read(),
            Err(ReferenceError::ExecutionPoisoned {
                reason: "stateful transaction ended without restoring state".to_string(),
            }),
        );
    }

    #[test]
    fn test_reference_semantics_describes_an_aliasing_view_without_resource_identity() {
        /// Test operation whose result aliases its input without defining a new reference root.
        #[derive(Clone)]
        struct TestAliasingViewOperation;

        impl Operation for TestAliasingViewOperation {
            type Type = ReferenceType<ArrayType>;

            fn name(&self) -> &'static str {
                "test_aliasing_view"
            }

            fn infer_output_types(
                &self,
                input_types: &[ReferenceType<ArrayType>],
                _region_interfaces: &[RegionInterface<ReferenceType<ArrayType>>],
            ) -> Result<Vec<ReferenceType<ArrayType>>, TypeError> {
                Ok(input_types.to_vec())
            }

            fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
                Cow::Owned(ReferenceOperationSemantics::new(
                    vec![ReferenceOutputSemantics::Alias {
                        output_index: 0,
                        input_index: 0,
                        kind: ReferenceAliasKind::Identity,
                    }],
                    Vec::new(),
                ))
            }
        }

        let semantics = TestAliasingViewOperation.reference_semantics();
        assert!(semantics.accesses().is_empty());
        assert!(!semantics.is_empty());
        assert_eq!(
            semantics.outputs(),
            &[ReferenceOutputSemantics::Alias { output_index: 0, input_index: 0, kind: ReferenceAliasKind::Identity }],
        );
        assert_eq!(ReferenceOperationSemantics::empty(), &ReferenceOperationSemantics::default());
        assert!(ReferenceOperationSemantics::empty().is_empty());

        // Boxed operations must forward reference semantics rather than fall back to the empty trait default.
        let boxed: Box<TestAliasingViewOperation> = Box::new(TestAliasingViewOperation);
        assert_eq!(boxed.reference_semantics(), TestAliasingViewOperation.reference_semantics());
    }

    #[test]
    #[should_panic(expected = "output 0 received two reference classifications")]
    fn test_reference_semantics_rejects_two_classifications_for_one_output() {
        ReferenceOperationSemantics::new(
            vec![
                ReferenceOutputSemantics::NewRoot { output_index: 0 },
                ReferenceOutputSemantics::Alias { output_index: 0, input_index: 0, kind: ReferenceAliasKind::Identity },
            ],
            Vec::new(),
        );
    }

    #[test]
    #[should_panic(expected = "input 0 received two reference accesses")]
    fn test_reference_semantics_rejects_two_accesses_for_one_input() {
        ReferenceOperationSemantics::new(
            Vec::new(),
            vec![
                ReferenceInputAccess::new(0, ReferenceAccessMode::Read),
                ReferenceInputAccess::new(0, ReferenceAccessMode::Write),
            ],
        );
    }
}
