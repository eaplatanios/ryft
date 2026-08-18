//! Structural reference types, operation-level reference semantics descriptors, and the eager reference holder.
//!
//! References give programs mutable state with explicit runtime identity. This module owns the program-facing
//! surface of that feature:
//!
//! - [`ReferenceType`] is the structural [`Type`] of a reference. It carries only referent metadata: runtime
//!   identity belongs to [`Reference`] and never participates in structural equality, hashing, or retained-program
//!   specialization.
//! - [`Reference`] is the cloneable identity-bearing eager holder. Clones alias one holder, reads return immutable
//!   value snapshots, replacement and additive update are atomic and preserve the exact declared referent type, and
//!   a consuming freeze invalidates the complete alias family.
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
//! Views, discharge, automatic differentiation, batching, rematerialization, external program-holder mutation, and
//! backend lowering are not yet supported. The complete validation, runtime, and kernel roadmap is tracked in the
//! repository's `plan-references.md`.

// TODO(eaplatanios): Review this module.

use std::borrow::{Borrow, Cow};
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

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

    /// A replacement or update result did not preserve the holder's exact declared referent type.
    #[error("reference value type `{actual}` must exactly match declared referent type `{expected}`")]
    ReferentTypeMismatch {
        /// Exact declared referent type.
        expected: String,

        /// Actual replacement or update-result type.
        actual: String,
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

/// Reference classification of one operation output: either a fresh canonical root or an alias of one input's root.
///
/// The two cases are mutually exclusive by construction, so an output can never be declared as both a new root and
/// an alias. [`Alias`](ReferenceOutputSemantics::Alias) carries exactly one `input_index` on purpose: every reference
/// operand must resolve to exactly one canonical root, so multi-source aliases (e.g., a hypothetical
/// `select_reference(a, b)`) are structurally unrepresentable rather than merely rejected. View operations will
/// attach their ordered coordinate-transform stacks to the alias case when the view representation lands; the alias
/// edge itself is all that root resolution needs until then.
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
/// The whole-array reference vocabulary declares the following semantics:
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
/// future_view(r) -> view
///     outputs  = [Alias { output_index: 0, input_index: 0 }]
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
pub(crate) struct ReferenceId(usize);

impl Display for ReferenceId {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "0x{:x}", self.0)
    }
}

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
}

/// Lifecycle state shared by every handle in one reference alias family.
enum ReferenceState<V: Value> {
    /// Live reference containing its current immutable value snapshot.
    Ready(V),

    /// Consumed reference whose value was returned by `freeze`.
    Frozen,
}

impl<V: Value> Reference<V> {
    /// Creates a new independent reference initialized with `value`.
    pub fn new(value: V) -> Self {
        let r#type = ReferenceType::new(value.r#type().into_owned());
        Self { holder: Arc::new(Mutex::new(ReferenceState::Ready(value))), r#type }
    }

    /// Returns this holder's process-local identity, which remains stable while any alias is alive.
    #[inline]
    pub(crate) fn id(&self) -> ReferenceId {
        ReferenceId(Arc::as_ptr(&self.holder) as usize)
    }

    /// Returns a clone of the currently stored value, which is an immutable snapshot for a valid reference referent.
    pub fn read(&self) -> Result<V, ReferenceError> {
        match &*self.holder.lock().map_err(|_| ReferenceError::Poisoned)? {
            ReferenceState::Ready(value) => Ok(value.clone()),
            ReferenceState::Frozen => Err(ReferenceError::Frozen),
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
                Ok(std::mem::replace(current, replacement))
            }
            ReferenceState::Frozen => Err(ReferenceError::Frozen),
        }
    }

    /// Atomically computes and installs an updated value while retaining the old value on every failure.
    ///
    /// This crate-visible primitive keeps value-family-specific update logic (such as array addition) outside the
    /// generic holder while ensuring no other access can interleave between reading the old state and installing the
    /// new one.
    pub(crate) fn update_with(&self, update: impl FnOnce(&V) -> Result<V, ProgramError>) -> Result<(), ProgramError> {
        let mut state = self.holder.lock().map_err(|_| ProgramError::custom(ReferenceError::Poisoned))?;
        match &mut *state {
            ReferenceState::Ready(current) => {
                let updated = update(current)?;
                self.validate_referent_type(&updated).map_err(ProgramError::custom)?;
                *current = updated;
                Ok(())
            }
            ReferenceState::Frozen => Err(ProgramError::custom(ReferenceError::Frozen)),
        }
    }

    /// Consumes this reference's current value and invalidates every handle in its alias family.
    pub fn freeze(&self) -> Result<V, ReferenceError> {
        let mut state = self.holder.lock().map_err(|_| ReferenceError::Poisoned)?;
        match std::mem::replace(&mut *state, ReferenceState::Frozen) {
            ReferenceState::Ready(value) => Ok(value),
            ReferenceState::Frozen => Err(ReferenceError::Frozen),
        }
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
}

impl<V: Value> Clone for Reference<V> {
    #[inline]
    fn clone(&self) -> Self {
        Self { holder: self.holder.clone(), r#type: self.r#type.clone() }
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
                    vec![ReferenceOutputSemantics::Alias { output_index: 0, input_index: 0 }],
                    Vec::new(),
                ))
            }
        }

        let semantics = TestAliasingViewOperation.reference_semantics();
        assert!(semantics.accesses().is_empty());
        assert_eq!(semantics.outputs(), &[ReferenceOutputSemantics::Alias { output_index: 0, input_index: 0 }]);
        assert_eq!(ReferenceOperationSemantics::empty(), &ReferenceOperationSemantics::default());

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
                ReferenceOutputSemantics::Alias { output_index: 0, input_index: 0 },
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
