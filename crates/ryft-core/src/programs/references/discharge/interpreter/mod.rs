mod regions;
mod rules;

pub use regions::{
    RecursiveReferenceDischargeDriver, ReferenceDischargeRegionDestination, ReferenceRegionDischargeBoundary,
    ReferenceRegionDischargeFork, ReferenceRegionStateInsertion, ReferenceRegionSummary, ReferenceStateWidening,
    discharge_positional_region_operation,
};
pub(super) use rules::region_closure_touches_references;
pub use rules::{
    ReferenceDischargeDriver, ReferenceDischargeableOperation, discharge_preserved_access,
    discharge_reference_free_operation,
};
use rules::{validate_discharged_value_type, validate_preserved_value};

use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::contexts::Domain;
use crate::programs::ProgramError;
use crate::programs::instructions::InstructionId;
use crate::programs::types::{Type, Typed};

use super::super::types::ReferenceType;
use super::policies::{ReferenceAccumulationPolicy, ReferenceDischargePolicy};
use super::results::ReferenceSource;
use super::selection::{ReferenceDischargeSelection, ReferenceDischargeSite};

/// Identity of one reference root inside a running reference discharge.
///
/// Handles are minted by [`ReferenceDischargeContext`] as roots enter its environment, so they are interpreter
/// identities rather than source-program coordinates: they exist only for the duration of one discharge and are
/// meaningful only against the environment that produced them. Pre-transform identity for caller-facing selection is
/// [`ReferenceDischargeSite`] instead.
///
/// Each handle records which environment minted it, so a handle from an unrelated discharge is reported rather than
/// silently addressing whichever root happens to occupy the same position. That is also what isolates a structured
/// rule's region fork: the fork mints its own environment, so a caller handle cannot address a fork root and a fork
/// handle cannot address a caller root. The one table relating the two lives inside
/// [`ReferenceDischargeDriver::discharge_region_program`], which reports its results in caller terms.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReferenceRootHandle {
    /// Environment that minted this handle.
    environment: ReferenceDischargeEnvironmentId,

    /// Position of the root in that environment.
    index: usize,
}

impl Display for ReferenceRootHandle {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "reference root {}:{}", self.environment.0, self.index)
    }
}

/// Identity of one reference discharge root environment, shared by every clone of the context that owns it and
/// distinct for every environment a structured rule's region fork mints.
///
/// This is private because no caller ever names it: it exists to make [`ReferenceRootHandle`] addressable only in
/// the environment that minted it, and a handle is obtained from the context rather than constructed.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ReferenceDischargeEnvironmentId(usize);

impl ReferenceDischargeEnvironmentId {
    /// Returns a fresh environment identity, distinct from every identity handed out so far in this process.
    fn next() -> Self {
        static NEXT_ENVIRONMENT_ID: AtomicUsize = AtomicUsize::new(0);
        Self(NEXT_ENVIRONMENT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// Environment entry describing what one live reference root became during reference discharge.
#[derive(Debug)]
enum ReferenceRootState<V> {
    /// Root selected for discharge, which threads through the destination program as immutable state.
    Discharged {
        /// Current immutable state of the whole root.
        current: V,

        /// Whether any ordered write or accumulation has been applied to this root. Read-only roots are pruned from
        /// hidden outputs and from structured-operation widening, so this is the fact that pruning consults.
        mutated: bool,
    },

    /// Root not selected for discharge, which survives in the destination program as a reference value. This is the
    /// root's own destination reference value and is what boundary threading uses; a handle derived from it through
    /// a view carries its own exact destination value instead.
    Preserved {
        /// Destination reference-typed value denoting the root.
        reference: V,
    },
}

/// Reference roots the capture prefix of one discharge scope binds.
///
/// A capture-lifted program names its caller's references through constants rather than through its own boundary: the
/// entry boundary carries the lifted capture prefix, and an attached region inside that program names the very same
/// references through capture constants. Resolving one is therefore a property of the scope a region discharges
/// under, not of any rule, so the scope rides on [`ReferenceDischargeContext`] beside the root environment and is
/// recomputed at every region boundary — inherited by default, and replaced by a fresh prefix wherever an operation
/// declares one through [`Operation::region_capture_input_count`].
///
/// Recognizing a capture is a *constant-family* question, and the interpreter deliberately serves families that are
/// not capture-bearing at all, so the seam is a function pointer supplied by the entry point that knows the family
/// rather than a [`CaptureConstant`] bound on the whole architecture. The [`Default`] scope recognizes nothing and
/// binds nothing, which is exactly the behavior of a program that has no captures.
pub(super) struct ReferenceCaptureScope<Constant> {
    /// Capture position a constant names, or [`None`] when it is an ordinary constant of its family.
    capture_index: fn(&Constant) -> Option<usize>,

    /// Root each capture position binds, or [`None`] when that position carries an ordinary value rather than a
    /// reference. A capture position past the end of this list binds nothing.
    roots: Rc<[Option<ReferenceRootHandle>]>,
}

impl<Constant> ReferenceCaptureScope<Constant> {
    /// Creates a capture scope.
    ///
    /// # Parameters
    ///
    ///   - `capture_index`: Seam reporting the capture position a constant of this family names.
    ///   - `roots`: Root each capture position binds, in capture order.
    #[inline]
    pub(super) fn new(capture_index: fn(&Constant) -> Option<usize>, roots: Vec<Option<ReferenceRootHandle>>) -> Self {
        Self { capture_index, roots: roots.into() }
    }

    /// Returns the root each capture position binds, in capture order.
    #[inline]
    fn roots(&self) -> &[Option<ReferenceRootHandle>] {
        self.roots.as_ref()
    }

    /// Returns the root one constant denotes, or [`None`] when the constant names no capture position or that
    /// position binds no root. A constant this scope cannot resolve is an ordinary constant of its family, and a
    /// reference-typed one that no scope resolves is rejected where it is lifted.
    #[inline]
    fn resolve(&self, constant: &Constant) -> Option<ReferenceRootHandle> {
        (self.capture_index)(constant).and_then(|index| self.roots.get(index).copied().flatten())
    }

    /// Returns this scope's seam over a different set of bound roots, which is how a nested region's scope and a
    /// region fork's remapped scope are built without restating the constant family's recognition rule.
    #[inline]
    fn with_roots(&self, roots: Vec<Option<ReferenceRootHandle>>) -> Self {
        Self { capture_index: self.capture_index, roots: roots.into() }
    }
}

impl<Constant> Default for ReferenceCaptureScope<Constant> {
    #[inline]
    fn default() -> Self {
        Self { capture_index: |_| None, roots: Rc::from([]) }
    }
}

impl<Constant> Clone for ReferenceCaptureScope<Constant> {
    #[inline]
    fn clone(&self) -> Self {
        Self { capture_index: self.capture_index, roots: Rc::clone(&self.roots) }
    }
}

impl<Constant> Debug for ReferenceCaptureScope<Constant> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("ReferenceCaptureScope").field("roots", &self.roots).finish_non_exhaustive()
    }
}

/// Destination fate a [`ReferenceDischargeReference`] handle carries for its root.
///
/// The binding is fixed when the handle is constructed and always agrees with the root's environment state, because
/// roots never move between the discharged and preserved fates after they are bound: carrying the fate on the handle
/// makes a handle/environment disagreement unrepresentable rather than defensively re-checked at every access.
#[derive(Clone, Debug, PartialEq)]
enum ReferenceDischargeBinding<V> {
    /// The root became explicit immutable state, so accesses through this handle rewrite into state reads and
    /// writes against the environment.
    Discharged,

    /// The root survives in the destination program, and this exact handle denotes `reference` there.
    ///
    /// A preserved handle must consume this value rather than re-deriving its view chain per access, because
    /// re-deriving would duplicate and reorder the replayed view operations in the destination program.
    Preserved {
        /// Exact destination reference value this handle denotes.
        reference: V,
    },
}

/// Handle to one live reference root flowing through reference discharge.
///
/// The fields are private and only [`ReferenceDischargeContext`] constructs them, so a rule can read a handle but
/// cannot fabricate a root, an alias, a derived type, or a preserved destination value. That keeps root identity and
/// view composition checked even though the rule trait is open to third-party operations.
pub struct ReferenceDischargeReference<C: Domain, P: ReferenceDischargePolicy<C>> {
    /// Identity of the root this handle denotes.
    root: ReferenceRootHandle,

    /// Whether this handle denotes the complete root rather than any derived view of it.
    denotes_whole_root: bool,

    /// Composed policy-owned view chain from the root to this handle.
    alias: P::Alias,

    /// Reference type this exact handle exposes, which differs from the root's type under a composed view.
    r#type: ReferenceType<P::Referent>,

    /// Destination fate of this handle's root, fixed at construction.
    binding: ReferenceDischargeBinding<C::Value>,
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> ReferenceDischargeReference<C, P> {
    /// Returns the identity of the root this handle denotes.
    #[inline]
    pub const fn root(&self) -> ReferenceRootHandle {
        self.root
    }

    /// Returns whether this handle denotes the complete root rather than a derived view.
    #[inline]
    const fn denotes_whole_root(&self) -> bool {
        self.denotes_whole_root
    }

    /// Returns the composed view chain from the root to this handle.
    #[inline]
    pub const fn alias(&self) -> &P::Alias {
        &self.alias
    }

    /// Returns the reference type this exact handle exposes.
    #[inline]
    pub const fn r#type(&self) -> &ReferenceType<P::Referent> {
        &self.r#type
    }

    /// Returns the exact destination reference value of a preserved handle, or [`None`] when the root was
    /// discharged.
    #[inline]
    pub const fn preserved(&self) -> Option<&C::Value> {
        match &self.binding {
            ReferenceDischargeBinding::Discharged => None,
            ReferenceDischargeBinding::Preserved { reference } => Some(reference),
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Clone for ReferenceDischargeReference<C, P> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            root: self.root,
            denotes_whole_root: self.denotes_whole_root,
            alias: self.alias.clone(),
            r#type: self.r#type.clone(),
            binding: self.binding.clone(),
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Debug for ReferenceDischargeReference<C, P> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReferenceDischargeReference")
            .field("root", &self.root)
            .field("denotes_whole_root", &self.denotes_whole_root)
            .field("alias", &self.alias)
            .field("type", &self.r#type)
            .field("binding", &self.binding)
            .finish()
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Display for ReferenceDischargeReference<C, P> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{} {}", self.root, self.r#type)
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> PartialEq for ReferenceDischargeReference<C, P>
where
    C::Value: PartialEq,
    P::Alias: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root
            && self.denotes_whole_root == other.denotes_whole_root
            && self.alias == other.alias
            && self.r#type == other.r#type
            && self.binding == other.binding
    }
}

/// Context-free carrier flowing through reference discharge.
///
/// Rules receive and return carriers; the context that owns the root environment travels separately as an explicit
/// rule argument rather than being stamped onto every value. It is public because the rule trait names it, and
/// because enum variant fields are always as public as their enum, the reference payload is the opaque
/// [`ReferenceDischargeReference`] rather than inline fields.
pub enum ReferenceDischargeValue<C: Domain, P: ReferenceDischargePolicy<C>> {
    /// Ordinary destination value, carrying no reference and replayed as-is.
    Ordinary(C::Value),

    /// Handle to one live reference root.
    Reference(ReferenceDischargeReference<C, P>),
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> ReferenceDischargeValue<C, P> {
    /// Returns the ordinary destination value this carrier holds, or an error naming `expectation` when it holds a
    /// reference handle instead.
    ///
    /// # Parameters
    ///
    ///   - `expectation`: Description of the operand the caller expected, used in the diagnostic.
    pub fn expect_ordinary(&self, expectation: &str) -> Result<&C::Value, ProgramError> {
        match self {
            Self::Ordinary(value) => Ok(value),
            Self::Reference(reference) => Err(ProgramError::MalformedProgram(format!(
                "reference discharge expected {expectation} but received {reference}",
            ))),
        }
    }

    /// Returns the reference handle this carrier holds, or an error naming `expectation` when it holds an ordinary
    /// value instead.
    ///
    /// # Parameters
    ///
    ///   - `expectation`: Description of the operand the caller expected, used in the diagnostic.
    pub fn expect_reference(&self, expectation: &str) -> Result<&ReferenceDischargeReference<C, P>, ProgramError> {
        match self {
            Self::Reference(reference) => Ok(reference),
            Self::Ordinary(_) => Err(ProgramError::MalformedProgram(format!(
                "reference discharge expected {expectation} but received an ordinary value",
            ))),
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Clone for ReferenceDischargeValue<C, P> {
    #[inline]
    fn clone(&self) -> Self {
        match self {
            Self::Ordinary(value) => Self::Ordinary(value.clone()),
            Self::Reference(reference) => Self::Reference(reference.clone()),
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Debug for ReferenceDischargeValue<C, P> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ordinary(value) => formatter.debug_tuple("Ordinary").field(value).finish(),
            Self::Reference(reference) => formatter.debug_tuple("Reference").field(reference).finish(),
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Display for ReferenceDischargeValue<C, P> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ordinary(value) => Display::fmt(value, formatter),
            Self::Reference(reference) => Display::fmt(reference, formatter),
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> PartialEq for ReferenceDischargeValue<C, P>
where
    C::Value: PartialEq,
    P::Alias: PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Ordinary(value), Self::Ordinary(other)) => value == other,
            (Self::Reference(reference), Self::Reference(other)) => reference == other,
            _ => false,
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Typed for ReferenceDischargeValue<C, P> {
    type Type = C::Type;

    #[inline]
    fn r#type(&self) -> Cow<'_, C::Type> {
        match self {
            Self::Ordinary(value) => value.r#type(),
            Self::Reference(reference) => Cow::Owned(P::lift_reference_type(reference.r#type().clone())),
        }
    }
}

/// Complete environment record of one live root: the reference type the whole root exposes, and what discharge turned
/// that root into.
///
/// The reference type is recorded because a root's identity outlives every handle that denotes it. A structured rule
/// threading an inherited root through a rebuilt region boundary holds only that root's handle, never a handle it
/// could read a type off, so the environment is where the whole-root type has to live.
struct ReferenceRootEntry<T: Type, V> {
    /// Reference type of the whole root, whose referent types the immutable state a discharged root threads.
    r#type: ReferenceType<T>,

    /// What discharge turned this root into.
    state: ReferenceRootState<V>,
}

/// Live root environment of one reference discharge, shared by every clone of its context.
struct ReferenceDischargeEnvironment<T: Type, V> {
    /// Identity that every handle minted from this environment records.
    id: ReferenceDischargeEnvironmentId,

    /// State of every root minted so far, indexed by [`ReferenceRootHandle`]. A consumed root keeps its slot and
    /// becomes [`None`], so a use-after-consume is reported against the exact root rather than as an unknown handle.
    roots: Vec<Option<ReferenceRootEntry<T, V>>>,
}

impl<T: Type, V> ReferenceDischargeEnvironment<T, V> {
    /// Returns the state slot that `root` names, or an error when the handle belongs to another environment or names
    /// a position this environment never minted.
    fn slot(&self, root: ReferenceRootHandle) -> Result<&Option<ReferenceRootEntry<T, V>>, ProgramError> {
        if root.environment != self.id {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge accessed {root}, which belongs to an environment other than the active `{}`",
                self.id.0,
            )));
        }
        self.roots
            .get(root.index)
            .ok_or_else(|| ProgramError::MalformedProgram(format!("reference discharge accessed never-bound {root}")))
    }
}

/// Active state of one reference discharge, owning the live root environment that its flowing values refer to.
///
/// Discharge is a single program-to-program interpretation driven region by region through
/// [`ReferenceDischargeDriver`]: each replayed instruction dispatches to its
/// [`ReferenceDischargeableOperation`] rule with this context as an explicit argument, and rules bind the rewritten,
/// reference-free work through [`parent`](Self::parent).
///
/// Its state lives here rather than in the flowing values because a reference is an identity, not a payload: several
/// handles can denote the same root through different views, and every one of them must observe the same current
/// state. Clones therefore share one environment, exactly as every other stateful context in Ryft shares one active
/// builder. A structured rule that must rebuild an attached region instead runs it against an isolated environment
/// through [`ReferenceDischargeDriver::discharge_region_program`], which commits nothing here.
pub struct ReferenceDischargeContext<C: Domain, P: ReferenceDischargePolicy<C>> {
    /// Destination context that owns the discharged values and executes or stages the rewritten work.
    parent: C,

    /// Live root environment shared by every clone of this context.
    environment: Rc<RefCell<ReferenceDischargeEnvironment<P::Referent, C::Value>>>,

    /// Roots the capture prefix of the scope this context discharges binds. A region that inherits its parent's
    /// capture prefix discharges under the same scope; a region fork rebuilds the scope in its own root terms.
    captures: ReferenceCaptureScope<C::Constant>,

    /// Reference sites this discharge normalizes into immutable state. Every root the selection omits is preserved,
    /// and the selection is shared unchanged by every clone and by every region fork, because a site names a source
    /// coordinate that means the same thing wherever the replay reaches it.
    selection: ReferenceDischargeSelection,
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> ReferenceDischargeContext<C, P> {
    /// Creates a discharge context with an empty root environment and an empty capture scope over the provided
    /// destination context, discharging every reference it reaches.
    ///
    /// The capture scope is populated afterwards rather than here, because the roots it binds are minted by this very
    /// context as its boundary is threaded. Partial discharge is requested through
    /// The [`partial-discharge entry point`](crate::Program::partially_discharge_references_with_policy) must be used
    /// instead of constructing a context, so that a selection is always validated against the program whose
    /// coordinates it names.
    #[inline]
    pub fn new(parent: C) -> Self {
        Self::new_selecting(parent, ReferenceDischargeSelection::everything())
    }

    /// Creates a discharge context with an empty root environment and an empty capture scope over the provided
    /// destination context, discharging exactly the references `selection` names.
    #[inline]
    pub(super) fn new_selecting(parent: C, selection: ReferenceDischargeSelection) -> Self {
        Self {
            parent,
            environment: Rc::new(RefCell::new(ReferenceDischargeEnvironment {
                id: ReferenceDischargeEnvironmentId::next(),
                roots: Vec::new(),
            })),
            captures: ReferenceCaptureScope::default(),
            selection,
        }
    }

    /// Returns whether the allocation an operation application performs was selected for discharge, which is what an
    /// allocation rule asks before deciding between a discharged root and one that survives in the destination.
    ///
    /// An application that did not come from a replayed instruction — a region-free rule invocation through
    /// [`EmptyRegionDriver`](crate::programs::EmptyRegionDriver) — has no source coordinate and is always
    /// discharged: no [`ReferenceDischargeSite`] can name it, so declining it would express nothing about the
    /// caller's choice.
    ///
    /// This is the only selection question a rule ever asks, which is why it is the only one exposed. Whether an
    /// *entry-boundary* root was selected is decided once, by the program-level entry point that threads the boundary,
    /// and no rule is in a position to ask it.
    ///
    /// # Parameters
    ///
    ///   - `instruction`: Replay position of the application, from [`ReferenceDischargeDriver::instruction`].
    ///   - `output_index`: Output position at which the application defines the fresh root.
    #[inline]
    pub fn selects_allocation(&self, instruction: Option<InstructionId>, output_index: usize) -> bool {
        instruction.is_none_or(|instruction| {
            self.selection.selects(ReferenceDischargeSite::Allocation { instruction, output_index })
        })
    }

    /// Returns whether one entry-boundary root was selected for discharge.
    #[inline]
    pub(super) fn selects_external(&self, source: ReferenceSource) -> bool {
        self.selection.selects(ReferenceDischargeSite::External(source))
    }

    /// Returns the destination context that owns the discharged values.
    #[inline]
    pub const fn parent(&self) -> &C {
        &self.parent
    }

    /// Returns the capture scope this context discharges under.
    #[inline]
    const fn captures(&self) -> &ReferenceCaptureScope<C::Constant> {
        &self.captures
    }

    /// Returns this context discharging under a different capture scope, sharing its live root environment.
    ///
    /// A region fork reaches its own scope this way, because the roots that scope binds are minted by the fork itself
    /// and therefore exist only once its boundary has been threaded.
    #[inline]
    pub(super) fn with_captures(&self, captures: ReferenceCaptureScope<C::Constant>) -> Self
    where
        C: Clone,
    {
        Self {
            parent: self.parent.clone(),
            environment: Rc::clone(&self.environment),
            captures,
            selection: self.selection.clone(),
        }
    }

    /// Binds a fresh root that threads as immutable state and returns the unviewed handle denoting it.
    ///
    /// # Parameters
    ///
    ///   - `r#type`: Reference type of the fresh root, normally derived from the allocating operation's inferred
    ///     output type through [`ReferenceDischargePolicy::project_reference_type`].
    ///   - `initial`: Destination value that becomes the root's initial immutable state.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `initial` does not carry the lifted referent type of `r#type`.
    pub fn allocate_discharged(
        &self,
        r#type: ReferenceType<P::Referent>,
        initial: C::Value,
    ) -> Result<ReferenceDischargeValue<C, P>, ProgramError> {
        validate_discharged_value_type::<C, P>(&initial, &r#type)?;
        Ok(self.bind_root_value(
            r#type,
            ReferenceRootState::Discharged { current: initial, mutated: false },
            ReferenceDischargeBinding::Discharged,
        ))
    }

    /// Binds a root that survives in the destination program and returns the unviewed handle denoting it.
    ///
    /// # Parameters
    ///
    ///   - `r#type`: Reference type of the root.
    ///   - `reference`: Destination reference-typed value denoting the root.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `reference` does not carry the reference type `r#type`.
    pub fn bind_preserved(
        &self,
        r#type: ReferenceType<P::Referent>,
        reference: C::Value,
    ) -> Result<ReferenceDischargeValue<C, P>, ProgramError> {
        validate_preserved_value::<C, P>(&reference, &r#type)?;
        Ok(self.bind_root_value(
            r#type,
            ReferenceRootState::Preserved { reference: reference.clone() },
            ReferenceDischargeBinding::Preserved { reference },
        ))
    }

    /// Returns a handle that composes `alias` onto `reference`, denoting the same root through a derived view.
    ///
    /// The composed alias is the authoritative view chain for the derived handle, so callers pass the complete chain
    /// rather than a single step. When the root is preserved, `derive_preserved` receives the parent handle's exact
    /// destination reference value and must return the destination value of the derived handle — normally by
    /// replaying the view operation — so that later accesses consume that exact value instead of re-deriving the
    /// chain. When the root is discharged, `derive_preserved` is never called, so a handle whose binding disagrees
    /// with its root's fate cannot be constructed.
    ///
    /// # Parameters
    ///
    ///   - `reference`: Handle the view is composed onto.
    ///   - `alias`: Complete composed view chain of the derived handle.
    ///   - `r#type`: Reference type the derived handle exposes.
    ///   - `derive_preserved`: Produces the derived handle's destination reference value from the parent handle's,
    ///     invoked only when the root is preserved.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the root is no longer live or when the derived destination
    /// value does not carry the reference type `r#type`, and propagates every `derive_preserved` failure.
    pub fn derive(
        &self,
        reference: &ReferenceDischargeReference<C, P>,
        alias: P::Alias,
        r#type: ReferenceType<P::Referent>,
        derive_preserved: impl FnOnce(&C::Value) -> Result<C::Value, ProgramError>,
    ) -> Result<ReferenceDischargeValue<C, P>, ProgramError> {
        let root = reference.root();
        self.validate_live_root(root)?;
        let binding = match &reference.binding {
            ReferenceDischargeBinding::Discharged => ReferenceDischargeBinding::Discharged,
            ReferenceDischargeBinding::Preserved { reference: parent } => {
                let derived = derive_preserved(parent)?;
                validate_preserved_value::<C, P>(&derived, &r#type)?;
                ReferenceDischargeBinding::Preserved { reference: derived }
            }
        };
        Ok(ReferenceDischargeValue::Reference(ReferenceDischargeReference {
            root,
            denotes_whole_root: false,
            alias,
            r#type,
            binding,
        }))
    }

    /// Returns the reference type of one live root as a whole, which is the type a handle denoting the complete root
    /// exposes and whose referent types the root's immutable state. Region rebuilding is the consumer: a threaded
    /// root arrives as a bare identity, and this is how the rebuilt boundary recovers the type its position carries.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the root was consumed or was never bound in this context.
    fn root_reference_type(&self, root: ReferenceRootHandle) -> Result<ReferenceType<P::Referent>, ProgramError> {
        self.with_root_entry(root, |entry| entry.r#type.clone())
    }

    /// Returns the current immutable state of one discharged root.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the root is not live or was preserved rather than discharged.
    pub fn discharged_state(&self, root: ReferenceRootHandle) -> Result<C::Value, ProgramError> {
        self.with_root_entry(root, |entry| match &entry.state {
            ReferenceRootState::Discharged { current, .. } => Ok(current.clone()),
            ReferenceRootState::Preserved { .. } => Err(ProgramError::MalformedProgram(format!(
                "reference discharge requested the discharged state of preserved {root}",
            ))),
        })?
    }

    /// Replaces the immutable state of one discharged root and records that the root was mutated.
    ///
    /// The context's own write, replace, and accumulate services use this for the successor they just computed, and
    /// the positional rewrite uses it for the appended final-state outputs it publishes. A structured rule merges a
    /// boundary's returned state through [`merge_boundary_state`](Self::merge_boundary_state) instead, because a
    /// symmetric boundary returns a successor state for roots it never wrote and the mutation flag decides whether a
    /// root
    /// publishes a hidden final-state output.
    ///
    /// # Parameters
    ///
    ///   - `root`: Discharged root whose state is being replaced.
    ///   - `current`: Successor immutable state of the whole root.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the root is not live or was preserved rather than discharged.
    fn set_discharged_state(&self, root: ReferenceRootHandle, current: C::Value) -> Result<(), ProgramError> {
        self.replace_discharged_state(root, current, true)
    }

    /// Replaces the state of one discharged root with the state carried back out of a boundary, recording a mutation
    /// only when that boundary's closure actually wrote it.
    ///
    /// A loop-shaped boundary is symmetric: it returns a successor state for every root it carries, including roots
    /// its closure only read. The value that comes back for such a root equals the one that entered, so re-threading
    /// it keeps the destination consistent — but recording it as a write would not, because the mutation flag is what
    /// decides whether an external root publishes a hidden final-state output and therefore whether its caller updates
    /// the shared reference state. A read-only loop must leave its caller's reference state unchanged.
    ///
    /// # Parameters
    ///
    ///   - `root`: Discharged root whose state is being merged.
    ///   - `current`: State the boundary carried back out, which for an unwritten root equals the entering state.
    ///   - `mutated`: Whether the boundary's closure wrote or accumulated into the root.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the root is not live or was preserved rather than discharged.
    fn merge_discharged_state(
        &self,
        root: ReferenceRootHandle,
        current: C::Value,
        mutated: bool,
    ) -> Result<(), ProgramError> {
        self.replace_discharged_state(root, current, mutated)
    }

    /// Returns whether any ordered write or accumulation has been applied to one discharged root.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the root is not live or was preserved rather than discharged.
    pub(in crate::programs::references) fn is_mutated(&self, root: ReferenceRootHandle) -> Result<bool, ProgramError> {
        self.with_root_entry(root, |entry| match &entry.state {
            ReferenceRootState::Discharged { mutated, .. } => Ok(*mutated),
            ReferenceRootState::Preserved { .. } => Err(ProgramError::MalformedProgram(format!(
                "reference discharge queried mutation of preserved {root}",
            ))),
        })?
    }

    /// Returns every root that is still live in this context's environment, in binding order.
    pub fn live_roots(&self) -> Vec<ReferenceRootHandle> {
        let environment = self.environment.borrow();
        environment
            .roots
            .iter()
            .enumerate()
            .filter(|(_, state)| state.is_some())
            .map(|(index, _)| ReferenceRootHandle { environment: environment.id, index })
            .collect()
    }

    /// Reads the coordinates that `reference` selects from its root's current state.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the root is not live, and propagates the policy's error when
    /// the alias cannot be applied. Reading a preserved root through this function is rejected, because a preserved
    /// access must replay verbatim in the destination instead.
    pub fn read(&self, reference: &ReferenceDischargeReference<C, P>) -> Result<C::Value, ProgramError> {
        let current = self.discharged_state(reference.root())?;
        P::read(&self.parent, &current, reference.alias())
    }

    /// Replaces the coordinates that `reference` selects without observing their previous contents.
    ///
    /// # Parameters
    ///
    ///   - `reference`: Handle selecting the coordinates to replace.
    ///   - `replacement`: Value written into the selected coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the root is not live or was preserved, and propagates the
    /// policy's error when the write cannot be applied.
    pub fn write(
        &self,
        reference: &ReferenceDischargeReference<C, P>,
        replacement: C::Value,
    ) -> Result<(), ProgramError> {
        let root = reference.root();
        let current = self.discharged_state(root)?;
        let successor = P::write(&self.parent, &current, replacement, reference.alias())?;
        self.set_discharged_state(root, successor)
    }

    /// Replaces the coordinates that `reference` selects and returns their previous contents.
    ///
    /// # Parameters
    ///
    ///   - `reference`: Handle selecting the coordinates to replace.
    ///   - `replacement`: Value written into the selected coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the root is not live or was preserved, and propagates the
    /// policy's error when the alias cannot be applied.
    pub fn replace(
        &self,
        reference: &ReferenceDischargeReference<C, P>,
        replacement: C::Value,
    ) -> Result<C::Value, ProgramError> {
        let root = reference.root();
        let current = self.discharged_state(root)?;
        let (previous, successor) = P::replace(&self.parent, &current, replacement, reference.alias())?;
        self.set_discharged_state(root, successor)?;
        Ok(previous)
    }

    /// Accumulates `update` into the coordinates that `reference` selects.
    ///
    /// # Parameters
    ///
    ///   - `reference`: Handle selecting the coordinates to accumulate into.
    ///   - `update`: Value added into the selected coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the root is not live or was preserved, and propagates the
    /// policy's error when the alias cannot be applied or the universe forbids accumulation.
    pub fn accumulate(
        &self,
        reference: &ReferenceDischargeReference<C, P>,
        update: C::Value,
    ) -> Result<(), ProgramError>
    where
        P: ReferenceAccumulationPolicy<C>,
    {
        let root = reference.root();
        let current = self.discharged_state(root)?;
        let successor = P::accumulate(&self.parent, &current, update, reference.alias())?;
        self.set_discharged_state(root, successor)
    }

    /// Yields the current whole-root state of `reference`'s root and unbinds the root, so that every later access to
    /// it is reported as a use-after-consume.
    ///
    /// Consumption is a whole-root event and always yields the whole root's state, so the handle's alias is
    /// deliberately not applied. A derived handle therefore cannot name a consumption, even when its referent type
    /// happens to equal the root's. The invariant is enforced at the state transition where it is relied upon.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the root is not live, was preserved rather than discharged, or
    /// is named through a derived handle rather than the original whole-root handle.
    pub fn consume(&self, reference: &ReferenceDischargeReference<C, P>) -> Result<C::Value, ProgramError> {
        let root = reference.root();
        let current_type = self.with_root_entry(root, |entry| match &entry.state {
            ReferenceRootState::Discharged { current, .. } => Ok(current.r#type().into_owned()),
            ReferenceRootState::Preserved { .. } => Err(ProgramError::MalformedProgram(format!(
                "reference discharge requested the discharged state of preserved {root}",
            ))),
        })??;
        if !reference.denotes_whole_root() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot consume {root} through the derived view `{}`; consumption yields the \
                 whole root, whose referent is `{}`",
                reference.r#type(),
                current_type,
            )));
        }
        let mut environment = self.environment.borrow_mut();
        // The inspection above proved that this handle belongs to this environment and names a live discharged root.
        let entry = environment.roots[root.index].take().unwrap();
        let ReferenceRootState::Discharged { current, .. } = entry.state else { unreachable!() };
        Ok(current)
    }

    /// Unbinds one preserved root, so that every later access to it is reported as a use-after-consume.
    ///
    /// This is [`consume`](Self::consume)'s counterpart for a root that survives in the destination. It yields no
    /// value, because the consuming operation was replayed verbatim and its own result is what the destination
    /// produced; all that remains is to stop the discharge environment from handing the root out again. Consumption is
    /// still a whole-root event, so a derived handle cannot name one even when its referent type equals the root's.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the root is not live, was discharged rather than preserved, or
    /// is named through a derived handle rather than the original whole-root handle.
    pub fn unbind_preserved(&self, reference: &ReferenceDischargeReference<C, P>) -> Result<(), ProgramError> {
        let root = reference.root();
        let whole = self.root_reference_type(root)?;
        if self.root_is_discharged(root)? {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge unbound discharged {root} as a preserved root",
            )));
        }
        if !reference.denotes_whole_root() {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot consume {root} through the derived view `{}`; consumption yields the \
                 whole root, whose reference type is `{}`",
                reference.r#type(),
                whole,
            )));
        }

        // `root_reference_type` already proved that this handle belongs to this environment and names a live root.
        self.environment.borrow_mut().roots[root.index] = None;
        Ok(())
    }

    /// Returns the unviewed handle denoting one live root of this environment.
    ///
    /// This mints no root: it re-exposes one the environment already holds, which is what resolving a capture-scoped
    /// reference constant needs. A preserved root's handle carries the root's own destination reference value,
    /// exactly as [`bind_preserved`](Self::bind_preserved) produced it.
    fn root_handle(&self, root: ReferenceRootHandle) -> Result<ReferenceDischargeValue<C, P>, ProgramError> {
        let (r#type, binding) = self.with_root_entry(root, |entry| {
            let binding = match &entry.state {
                ReferenceRootState::Discharged { .. } => ReferenceDischargeBinding::Discharged,
                ReferenceRootState::Preserved { reference } => {
                    ReferenceDischargeBinding::Preserved { reference: reference.clone() }
                }
            };
            (entry.r#type.clone(), binding)
        })?;
        let alias = P::root_alias(r#type.referent());
        Ok(ReferenceDischargeValue::Reference(ReferenceDischargeReference {
            root,
            denotes_whole_root: true,
            alias,
            r#type,
            binding,
        }))
    }

    /// Applies `use_entry` to one live root while holding the environment's immutable borrow.
    ///
    /// Callers must clone only the fields they need and must not invoke policy or destination operations from the
    /// callback, keeping the [`RefCell`] borrow local to this query.
    fn with_root_entry<R>(
        &self,
        root: ReferenceRootHandle,
        use_entry: impl FnOnce(&ReferenceRootEntry<P::Referent, C::Value>) -> R,
    ) -> Result<R, ProgramError> {
        let environment = self.environment.borrow();
        let entry = environment
            .slot(root)?
            .as_ref()
            .ok_or_else(|| ProgramError::MalformedProgram(format!("reference discharge accessed consumed {root}")))?;
        Ok(use_entry(entry))
    }

    /// Returns whether one live root is discharged rather than preserved.
    fn root_is_discharged(&self, root: ReferenceRootHandle) -> Result<bool, ProgramError> {
        self.with_root_entry(root, |entry| matches!(entry.state, ReferenceRootState::Discharged { .. }))
    }

    /// Validates that `root` belongs to this environment and remains live.
    pub(super) fn validate_live_root(&self, root: ReferenceRootHandle) -> Result<(), ProgramError> {
        self.with_root_entry(root, |_| ())
    }

    /// Validates that `current` carries the lifted referent type of `root` without mutating the environment.
    fn validate_discharged_state_type(
        &self,
        root: ReferenceRootHandle,
        current: &C::Value,
    ) -> Result<(), ProgramError> {
        let r#type = self.root_reference_type(root)?;
        validate_discharged_value_type::<C, P>(current, &r#type)
    }

    /// Appends one root record to the environment and returns the handle that denotes it.
    fn bind_root(
        &self,
        r#type: ReferenceType<P::Referent>,
        state: ReferenceRootState<C::Value>,
    ) -> ReferenceRootHandle {
        let mut environment = self.environment.borrow_mut();
        environment.roots.push(Some(ReferenceRootEntry { r#type, state }));
        ReferenceRootHandle { environment: environment.id, index: environment.roots.len() - 1 }
    }

    /// Binds one fresh whole-root carrier from an already validated environment state.
    fn bind_root_value(
        &self,
        r#type: ReferenceType<P::Referent>,
        state: ReferenceRootState<C::Value>,
        binding: ReferenceDischargeBinding<C::Value>,
    ) -> ReferenceDischargeValue<C, P> {
        let alias = P::root_alias(r#type.referent());
        let root = self.bind_root(r#type.clone(), state);
        ReferenceDischargeValue::Reference(ReferenceDischargeReference {
            root,
            denotes_whole_root: true,
            alias,
            r#type,
            binding,
        })
    }

    /// Replaces one live discharged root's state with its successor and merges the mutation fact.
    fn replace_discharged_state(
        &self,
        root: ReferenceRootHandle,
        current: C::Value,
        mutated: bool,
    ) -> Result<(), ProgramError> {
        self.validate_discharged_state_type(root, &current)?;
        let mut environment = self.environment.borrow_mut();
        environment.slot(root)?;
        match environment.roots[root.index].as_mut().map(|entry| &mut entry.state) {
            Some(ReferenceRootState::Discharged { current: state, mutated: previous_mutated }) => {
                *state = current;
                *previous_mutated |= mutated;
                Ok(())
            }
            Some(ReferenceRootState::Preserved { .. }) => Err(ProgramError::MalformedProgram(format!(
                "reference discharge replaced the state of preserved {root}",
            ))),
            None => Err(ProgramError::MalformedProgram(format!("reference discharge accessed consumed {root}"))),
        }
    }
}

impl<C: Clone + Domain, P: ReferenceDischargePolicy<C>> Clone for ReferenceDischargeContext<C, P> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            parent: self.parent.clone(),
            environment: Rc::clone(&self.environment),
            captures: self.captures.clone(),
            selection: self.selection.clone(),
        }
    }
}

impl<C: Domain, P: ReferenceDischargePolicy<C>> Debug for ReferenceDischargeContext<C, P> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let live_root_count = self.environment.borrow().roots.iter().filter(|root| root.is_some()).count();
        formatter
            .debug_struct("ReferenceDischargeContext")
            .field("live_roots", &live_root_count)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {

    use pretty_assertions::assert_eq;

    use crate::programs::ProgramError;

    use crate::programs::operations::Operation;

    use crate::programs::references::discharge::tests::*;

    use crate::programs::references::types::ReferenceType;

    use crate::programs::types::Typed;

    use super::*;

    #[test]
    fn test_reference_discharge_value_reports_operand_kind_mismatches() {
        // A rule that receives the wrong carrier kind gets a diagnostic naming what it expected, which is what keeps
        // an open set of third-party rules diagnosable without each of them inventing its own message.
        let context = ListDischargeContext::new(ListDestination::new());
        let reference_type = ReferenceType::new(ListType { length: 1 });
        let allocated = context.allocate_discharged(reference_type, ListIrValue::List(vec![1])).unwrap();
        let root = allocated.expect_reference("the allocated root").unwrap().root();
        let ordinary: ListDischargeValue = ReferenceDischargeValue::Ordinary(ListIrValue::List(vec![1]));

        assert_eq!(ordinary.expect_ordinary("an update value"), Ok(&ListIrValue::List(vec![1])));
        assert_eq!(
            allocated.expect_ordinary("an update value"),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge expected an update value but received {root} ref<list<1>>",
            ))),
        );
        assert_eq!(
            ordinary.expect_reference("a reference to read"),
            Err(ProgramError::MalformedProgram(
                "reference discharge expected a reference to read but received an ordinary value".to_string(),
            )),
        );
    }

    #[test]
    fn test_reference_capture_scope() {
        // A scope binds one root per capture position. Positions carrying an ordinary value, positions past the end of
        // the scope, and constants that name no capture position at all all resolve to nothing, which is what leaves
        // an unresolvable reference-typed constant to the rejection at the lift site.
        let context = ListDischargeContext::new(ListDestination::new());
        let allocated = context
            .allocate_discharged(ReferenceType::new(ListType { length: 2 }), ListIrValue::List(vec![1, 2]))
            .unwrap();
        let root = allocated.expect_reference("the captured root").unwrap().root();

        let empty = ReferenceCaptureScope::<ListIrValue>::default();
        assert_eq!(empty.roots(), &[]);
        assert_eq!(empty.resolve(&ListIrValue::Reference(ReferenceType::new(ListType { length: 2 }))), None);

        let scope = ReferenceCaptureScope::new(list_capture_position, vec![None, None, Some(root)]);
        assert_eq!(scope.roots(), &[None, None, Some(root)]);
        assert_eq!(scope.resolve(&ListIrValue::Reference(ReferenceType::new(ListType { length: 2 }))), Some(root));
        assert_eq!(scope.resolve(&ListIrValue::Reference(ReferenceType::new(ListType { length: 1 }))), None);
        assert_eq!(scope.resolve(&ListIrValue::Reference(ReferenceType::new(ListType { length: 9 }))), None);
        assert_eq!(scope.resolve(&ListIrValue::List(vec![1, 2])), None);

        // Rebinding keeps the seam, which is how a nested region's scope and a fork's remapped scope are built.
        let rebound = scope.with_roots(vec![Some(root)]);
        assert_eq!(rebound.roots(), &[Some(root)]);
        assert_eq!(rebound.resolve(&ListIrValue::Reference(ReferenceType::new(ListType { length: 0 }))), Some(root));
    }

    #[test]
    fn test_reference_discharge_context_threads_discharged_root_state() {
        let context = ListDischargeContext::new(ListDestination::new());
        let reference_type = ReferenceType::new(ListType { length: 4 });
        let allocated =
            context.allocate_discharged(reference_type.clone(), ListIrValue::List(vec![1, 2, 3, 4])).unwrap();
        let reference = allocated.expect_reference("the allocated root").unwrap().clone();
        let root = reference.root();

        // A fresh root starts unmutated, exposes its identity alias and reference type, and carries no destination
        // reference value because it was discharged rather than preserved.
        assert_eq!(context.live_roots(), vec![root]);
        assert_eq!(context.is_mutated(root), Ok(false));
        assert_eq!(reference.alias(), &ListAlias { offset: 0, length: 4 });
        assert_eq!(reference.r#type(), &reference_type);
        assert_eq!(reference.preserved(), None);
        assert_eq!(context.read(&reference), Ok(ListIrValue::List(vec![1, 2, 3, 4])));

        // A derived handle narrows the view without touching the root's identity, and its accesses act only on the
        // coordinates it selects.
        let view = context
            .derive(&reference, ListAlias { offset: 1, length: 2 }, ReferenceType::new(ListType { length: 2 }), |_| {
                unreachable!("the root is discharged")
            })
            .unwrap();
        let view = view.expect_reference("the derived view").unwrap().clone();
        assert_eq!(view.root(), root);
        assert_eq!(context.read(&view), Ok(ListIrValue::List(vec![2, 3])));
        assert_eq!(context.replace(&view, ListIrValue::List(vec![20, 30])), Ok(ListIrValue::List(vec![2, 3])));
        assert_eq!(context.read(&reference), Ok(ListIrValue::List(vec![1, 20, 30, 4])));
        assert_eq!(context.is_mutated(root), Ok(true));
        assert_eq!(context.accumulate(&view, ListIrValue::List(vec![1, 1])), Ok(()));
        assert_eq!(context.read(&reference), Ok(ListIrValue::List(vec![1, 21, 31, 4])));

        // Consumption is a whole-root event. Provenance, not type equality, distinguishes the root handle from a
        // derived view: a policy may derive a view whose referent happens to have the root's exact type.
        let same_type_view = context
            .derive(&reference, ListAlias { offset: 0, length: 4 }, reference_type.clone(), |_| {
                unreachable!("the root is discharged")
            })
            .unwrap();
        let same_type_view = same_type_view.expect_reference("the same-type derived view").unwrap();
        assert_eq!(
            context
                .operand_root(&ReferenceDischargeValue::Reference(same_type_view.clone()), ListOperation::Call.name(),),
            Err(ProgramError::MalformedProgram(format!(
                "operation `list.call` passes the derived view `ref<list<4>>` of {root} across a region boundary, \
                 which carries the whole root `ref<list<4>>`; derive the view inside the region instead",
            ))),
        );
        assert_eq!(
            context.consume(same_type_view),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot consume {root} through the derived view `ref<list<4>>`; consumption \
                 yields the whole root, whose referent is `list<4>`",
            ))),
        );

        // A narrower derived view is rejected by the same provenance check rather than silently yielding the whole
        // root's value under the view's type.
        assert_eq!(
            context.consume(&view),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot consume {root} through the derived view `ref<list<2>>`; consumption \
                 yields the whole root, whose referent is `list<4>`",
            ))),
        );

        // Through the root handle it yields the complete state and unbinds the root, so every later access through
        // any handle of that root is reported against the exact root.
        assert_eq!(context.consume(&reference), Ok(ListIrValue::List(vec![1, 21, 31, 4])));
        assert_eq!(context.live_roots(), Vec::new());
        let consumed = ProgramError::MalformedProgram(format!("reference discharge accessed consumed {root}"));
        assert_eq!(context.read(&reference), Err(consumed.clone()));
        assert_eq!(context.root_reference_type(root), Err(consumed.clone()));
        assert_eq!(context.set_discharged_state(root, ListIrValue::List(vec![0; 4])), Err(consumed));

        // A handle minted by an unrelated discharge is reported instead of silently addressing whichever root
        // occupies the same position here.
        let other = ListDischargeContext::new(ListDestination::new());
        let foreign = other.allocate_discharged(reference_type, ListIrValue::List(vec![0; 4])).unwrap();
        let foreign = foreign.expect_reference("the unrelated root").unwrap().root();
        let prefix =
            format!("reference discharge accessed {foreign}, which belongs to an environment other than the active");
        assert!(matches!(
            context.root_reference_type(foreign),
            Err(ProgramError::MalformedProgram(message)) if message.starts_with(&prefix),
        ));
    }

    #[test]
    fn test_reference_discharge_context_validates_root_state_types_before_mutation() {
        let context = ListDischargeContext::new(ListDestination::new());
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let wrong_state = ListIrValue::List(vec![1]);
        let error = ProgramError::MalformedProgram(
            "reference discharge state has type `list<1>` but root `ref<list<2>>` requires `list<2>`".to_string(),
        );

        // A malformed allocation is rejected before a root is inserted into the environment.
        assert_eq!(context.allocate_discharged(reference_type.clone(), wrong_state.clone()), Err(error.clone()));
        assert_eq!(context.live_roots(), Vec::new());

        let allocated = context.allocate_discharged(reference_type, ListIrValue::List(vec![1, 2])).unwrap();
        let reference = allocated.expect_reference("the allocated root").unwrap();
        let root = reference.root();

        // Both state-replacement paths validate before taking the mutable environment borrow, so failure preserves
        // the prior state and mutation bit.
        assert_eq!(context.set_discharged_state(root, wrong_state.clone()), Err(error.clone()));
        assert_eq!(context.read(reference), Ok(ListIrValue::List(vec![1, 2])));
        assert_eq!(context.is_mutated(root), Ok(false));
        assert_eq!(context.merge_discharged_state(root, wrong_state, true), Err(error));
        assert_eq!(context.read(reference), Ok(ListIrValue::List(vec![1, 2])));
        assert_eq!(context.is_mutated(root), Ok(false));
    }

    #[test]
    fn test_reference_discharge_context_clones_share_root_state() {
        let context = ListDischargeContext::new(ListDestination::new());
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let allocated = context.allocate_discharged(reference_type, ListIrValue::List(vec![1, 2])).unwrap();
        let reference = allocated.expect_reference("the allocated root").unwrap().clone();

        // A clone shares the environment rather than copying it, which is the contract every stateful Ryft context
        // follows: several handles can denote one root, and every one of them must observe the same current state.
        // Isolation is therefore never implicit — a structured rule that must not commit rebuilds its region against
        // an environment of its own through `discharge_region_program`.
        let clone = context.clone();
        clone.accumulate(&reference, ListIrValue::List(vec![10, 10])).unwrap();
        assert_eq!(context.read(&reference), Ok(ListIrValue::List(vec![11, 12])));
        context.accumulate(&reference, ListIrValue::List(vec![1, 1])).unwrap();
        assert_eq!(clone.read(&reference), Ok(ListIrValue::List(vec![12, 13])));
    }

    #[test]
    fn test_reference_discharge_context_binds_preserved_roots() {
        let context = ListDischargeContext::new(ListDestination::new());
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let destination_reference = ListIrValue::Reference(reference_type.clone());
        let bound = context.bind_preserved(reference_type.clone(), destination_reference.clone()).unwrap();
        let reference = bound.expect_reference("the preserved root").unwrap().clone();
        let root = reference.root();

        // A preserved root keeps its destination reference value on the handle, so a later access can replay
        // verbatim instead of re-deriving the handle.
        assert_eq!(context.root_reference_type(root), Ok(reference_type));
        assert_eq!(reference.preserved(), Some(&destination_reference));
        assert_eq!(
            context.operand_value(&ReferenceDischargeValue::Reference(reference.clone())),
            Ok(destination_reference.clone()),
        );

        // Every discharged-state service rejects a preserved root by name rather than silently treating it as state.
        assert_eq!(
            context.read(&reference),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge requested the discharged state of preserved {root}",
            ))),
        );
        assert_eq!(
            context.is_mutated(root),
            Err(ProgramError::MalformedProgram(format!("reference discharge queried mutation of preserved {root}"))),
        );
        assert_eq!(
            context.set_discharged_state(root, ListIrValue::List(vec![0, 0])),
            Err(ProgramError::MalformedProgram(format!("reference discharge replaced the state of preserved {root}",))),
        );

        // Deriving on a preserved root hands the closure the parent handle's exact destination value, so the derived
        // handle cannot disagree with its root's fate by construction.
        let view_type = ReferenceType::new(ListType { length: 1 });
        let view_alias = ListAlias { offset: 0, length: 1 };
        let view = context
            .derive(&reference, view_alias, view_type.clone(), |parent| {
                assert_eq!(parent, &ListIrValue::Reference(ReferenceType::new(ListType { length: 2 })));
                Ok(ListIrValue::Reference(view_type.clone()))
            })
            .unwrap();
        let view = view.expect_reference("the derived view").unwrap();
        assert_eq!(view.root(), root);
        assert_eq!(view.preserved(), Some(&ListIrValue::Reference(view_type)));
    }

    #[test]
    fn test_reference_discharge_value_reports_its_type_and_display() {
        let context = ListDischargeContext::new(ListDestination::new());
        let ordinary =
            ReferenceDischargeValue::<ListDestination, ListReferenceDischarge>::Ordinary(ListIrValue::List(vec![1]));
        let reference_type = ReferenceType::new(ListType { length: 2 });
        let reference = context.allocate_discharged(reference_type.clone(), ListIrValue::List(vec![1, 2])).unwrap();

        // An ordinary carrier reports the wrapped destination value's type, while a reference handle reports its own
        // reference type lifted into the destination universe.
        assert_eq!(ordinary.r#type().into_owned(), ListIrType::List(ListType { length: 1 }));
        assert_eq!(reference.r#type().into_owned(), ListIrType::Reference(reference_type));
        assert_eq!(ordinary.to_string(), "[1]");
        let root = reference.expect_reference("the allocated root").unwrap().root();
        assert_eq!(reference.to_string(), format!("{root} ref<list<2>>"));
        assert_eq!(
            format!("{reference:?}"),
            format!(
                "Reference(ReferenceDischargeReference {{ root: {root:?}, denotes_whole_root: true, alias: ListAlias \
                 {{ offset: 0, length: 2 }}, type: ReferenceType {{ referent: ListType {{ length: 2 }} }}, \
                 binding: Discharged }})",
            ),
        );
    }

    #[test]
    fn test_reference_discharge_context_unbinds_preserved_roots() {
        // Consuming a preserved root yields no state — the replayed operation already produced the destination's own
        // result — but it must still stop the environment from handing the root out again, and only a handle denoting
        // the whole root can name a consumption.
        let context = ListDischargeContext::new(ListDestination::new());
        let referent = ListType { length: 2 };
        let preserved = context
            .bind_preserved(
                ReferenceType::new(referent.clone()),
                ListIrValue::Reference(ReferenceType::new(referent.clone())),
            )
            .unwrap();
        let reference = preserved.expect_reference("the preserved root").unwrap().clone();
        let discharged =
            context.allocate_discharged(ReferenceType::new(referent), ListIrValue::List(vec![1, 2])).unwrap();
        let discharged_root = discharged.expect_reference("the discharged root").unwrap().root();

        let same_type_view = context
            .derive(&reference, ListAlias { offset: 0, length: 2 }, reference.r#type().clone(), |_| {
                Ok(ListIrValue::Reference(reference.r#type().clone()))
            })
            .unwrap();
        let same_type_view = same_type_view.expect_reference("the same-type preserved view").unwrap();
        assert_eq!(
            context.unbind_preserved(same_type_view),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot consume {} through the derived view `ref<list<2>>`; consumption yields \
                 the whole root, whose reference type is `ref<list<2>>`",
                reference.root(),
            ))),
        );

        let view = context
            .derive(&reference, ListAlias { offset: 0, length: 1 }, ReferenceType::new(ListType { length: 1 }), |_| {
                Ok(ListIrValue::Reference(ReferenceType::new(ListType { length: 1 })))
            })
            .unwrap();
        let view = view.expect_reference("the derived preserved view").unwrap().clone();
        assert_eq!(
            context.unbind_preserved(&view),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge cannot consume {} through the derived view `ref<list<1>>`; consumption yields \
                 the whole root, whose reference type is `ref<list<2>>`",
                reference.root(),
            ))),
        );
        assert_eq!(context.unbind_preserved(&reference), Ok(()));
        assert_eq!(context.live_roots(), vec![discharged_root]);
        assert_eq!(
            context.unbind_preserved(&reference),
            Err(ProgramError::MalformedProgram(format!("reference discharge accessed consumed {}", reference.root()))),
        );

        // A discharged root is not unbound through the preserved path, which is what keeps the two states from being
        // confused by a rule that dispatched on the wrong one.
        let discharged = discharged.expect_reference("the discharged root").unwrap();
        assert_eq!(
            context.unbind_preserved(discharged),
            Err(ProgramError::MalformedProgram(format!(
                "reference discharge unbound discharged {discharged_root} as a preserved root",
            ))),
        );
    }
}
