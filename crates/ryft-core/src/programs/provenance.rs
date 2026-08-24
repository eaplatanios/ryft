use std::cell::RefCell;
use std::fmt::Display;
use std::sync::Arc;

/// Named provenance scope describing one level of the logical origin of an [`Instruction`](crate::Instruction) (e.g.,
/// the framework facility or user computation that staged it). Scope names are single path segments preserved verbatim.
/// Namespacing is expressed structurally by nesting scopes, rendered as `::`-separated paths such as
/// `ryft::differentiation::coordinate_basis`, rather than by separator characters inside one name. Construction
/// is infallible and performs no validation because names have no correctness role, but names should be non-empty and
/// identifier-like (i.e., matching `[A-Za-z_][A-Za-z0-9_]*`) so that they render bare; any other name renders quoted
/// and escaped as Rust string literals. The root scope name `ryft` is typically reserved for framework-owned scopes,
/// which nest as `ryft::<subsystem>::<concept>`. User scopes should use their own namespace. Each scope maps onto
/// one MLIR [`NameLoc`](https://mlir.llvm.org/docs/Dialects/Builtin/#nameloc) when a program is lowered into MLIR
/// using Ryft's XLA backend.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProvenanceScope(Arc<str>);

impl ProvenanceScope {
    /// Creates a new [`ProvenanceScope`] with the provided name, preserving the name verbatim.
    #[inline]
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        Self(name.into())
    }

    /// Returns the name of this [`ProvenanceScope`].
    #[inline]
    pub fn name(&self) -> &str {
        &self.0
    }
}

/// Internal shared node representation backing non-unknown [`Provenance`] values. Unknown provenance is represented
/// using [`None`] values stored internally in [`Provenance`] rather than an explicit [`ProvenanceNode`] variant in
/// order to keep them allocation-free. That is because every [`Instruction`](crate::Instruction) carries a
/// [`Provenance`] and unknown provenance typically dominates in practice, which is what [`Provenance`] is
/// optimized for.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ProvenanceNode {
    /// One named [`ProvenanceScope`] above an inner origin.
    Scope {
        /// Name of this scope level.
        scope: ProvenanceScope,

        /// Origin below this scope level.
        origin: Provenance,
    },

    /// Several source origins intentionally merged into one generated instruction. This is normalized at construction
    /// by [`Provenance::fused`]; constituents are never unknown, never themselves fused, and never structural
    /// duplicates, and there are always at least two of them.
    Fused {
        /// Merged source origins, in first-occurrence order.
        origins: Arc<[Provenance]>,
    },
}

impl Display for ProvenanceNode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scope { scope, origin } => {
                // Identifier-like scope names render bare, while all other names render quoted and escaped
                // deterministically as Rust string literals, so arbitrary name content (i.e., including `::`,
                // brackets, quotes, and newlines) can never make the rendering ambiguous.
                let name = scope.name();
                let mut characters = name.chars();
                let bare =
                    characters.next().is_some_and(|character| character.is_ascii_alphabetic() || character == '_')
                        && characters.all(|character| character.is_ascii_alphanumeric() || character == '_');
                if bare {
                    write!(formatter, "{name}")?;
                } else {
                    write!(formatter, "{name:?}")?;
                }
                if let Some(origin) = origin.0.as_deref() {
                    write!(formatter, "::")?;
                    origin.fmt(formatter)?;
                }
                Ok(())
            }
            Self::Fused { origins } => {
                write!(formatter, "fused[")?;
                for (index, origin) in origins.iter().enumerate() {
                    if index > 0 {
                        write!(formatter, ", ")?;
                    }
                    write!(formatter, "{origin}")?;
                }
                write!(formatter, "]")
            }
        }
    }
}

/// Persistent, hierarchical, non-semantic origin of one [`Instruction`](crate::Instruction). Despite the related
/// name, this is unrelated to [`OutputRegionProvenance`](crate::OutputRegionProvenance), which describes the _semantic_
/// dataflow origin of an operation output. Provenance records _where_ an instruction came from (e.g., the framework
/// facility that staged it, or the source instructions a transform generated it from) without adding a Single Static
/// Assignment (SSA) value, a data dependency, or any semantic behavior. It is purely diagnostic (i.e., type inference,
/// effects, differentiation, batching, interpretation, optimization legality, and the canonical semantic
/// [`Program`](crate::Program) rendering must never depend on it), and behavior must remain correct if all provenance
/// is removed. Consequently, [`unknown`](Self::unknown) is always a correct value as dropping provenance degrades
/// diagnostics and should not change program behavior.
///
/// A provenance value is an immutable tree over three shapes:
///
///   - **Unknown:** No recorded origin, which is the default for manually constructed instructions.
///   - **Scope:** One named [`ProvenanceScope`] above an inner origin, forming outermost-first scope chains.
///   - **Fused:** Several source origins intentionally merged into one generated instruction.
///
/// Values are backed by shared immutable nodes, so cloning is one [`Arc`] clone, all instructions staged under one
/// active scope share the same allocation, and unknown provenance allocates nothing. Equality and hashing are
/// structural. When a program is lowered to MLIR using Ryft's XLA backend, scopes become
/// [`NameLoc`](https://mlir.llvm.org/docs/Dialects/Builtin/#nameloc)s and fused origins become
/// [`FusedLoc`](https://mlir.llvm.org/docs/Dialects/Builtin/#fusedloc)s.
///
/// The [`Display`] output renders scope chains as `::`-separated paths (e.g., `outer::inner`), fused origins as
/// bracketed lists that only ever terminate a chain (e.g., `top::fused[a, outer::inner]`), and unknown provenance as
/// `unknown` (program renderings omit the provenance of such instructions entirely instead of printing this token).
/// Identifier-like scope names render bare, while all other names render quoted and escaped as Rust string literals,
/// so arbitrary name content can never make the rendering ambiguous.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Provenance(Option<Arc<ProvenanceNode>>);

impl Provenance {
    /// Returns the [`Provenance`] of an [`Instruction`](crate::Instruction) with no recorded origin.
    #[inline]
    pub fn unknown() -> Self {
        Self(None)
    }

    /// Creates a new [`Provenance`] that attaches one named scope above the provided origin. No deduplication or
    /// common-prefix factoring is performed. The stored representation stays purely structural, and visualizers may
    /// factor common scope prefixes at display time.
    ///
    /// # Parameters
    ///
    ///   - `scope`: Scope to attach as the new outermost level.
    ///   - `origin`: Origin recorded below the new scope (possibly [`unknown`](Self::unknown)).
    #[inline]
    pub fn scope(scope: ProvenanceScope, origin: Provenance) -> Self {
        Self(Some(Arc::new(ProvenanceNode::Scope { scope, origin })))
    }

    /// Creates a new [`Provenance`] for a generated [`Instruction`](crate::Instruction) with multiple source origins
    /// (e.g., several source instructions intentionally merged into one by a transform). The provided origins are
    /// normalized at construction:
    ///
    ///   - unknown origins are discarded when another origin is present,
    ///   - nested fused origins are flattened,
    ///   - structurally duplicate origins are removed while preserving first-occurrence order, and
    ///   - zero remaining origins return [`unknown`](Self::unknown) and one remaining origin is returned directly.
    pub fn fused<O: IntoIterator<Item = Provenance>>(origins: O) -> Self {
        let mut normalized = Vec::<Provenance>::new();
        for origin in origins {
            match origin.0.as_deref() {
                // Fused nodes are normalized at construction, so their constituents are never unknown and never
                // themselves fused, and flattening one level is sufficient.
                Some(ProvenanceNode::Fused { origins }) => {
                    for origin in origins.iter() {
                        if !normalized.contains(origin) {
                            normalized.push(origin.clone());
                        }
                    }
                }
                Some(_) => {
                    if !normalized.contains(&origin) {
                        normalized.push(origin);
                    }
                }
                None => {}
            }
        }
        match normalized.len() {
            0 => Self::unknown(),
            1 => normalized.pop().unwrap(),
            _ => Self(Some(Arc::new(ProvenanceNode::Fused { origins: normalized.into() }))),
        }
    }

    /// Returns `true` if this [`Provenance`] records no origin.
    #[inline]
    pub fn is_unknown(&self) -> bool {
        self.0.is_none()
    }

    /// Returns the outermost scope of this [`Provenance`] together with the origin recorded below it when this
    /// provenance is a scope, and [`None`] if it is unknown or fused. Walking `as_scope` repeatedly recovers the
    /// complete outermost-first scope path and the provenance below it. Together with [`is_unknown`](Self::is_unknown)
    /// and [`as_fused`](Self::as_fused), this mirrors the three provenance shapes one-to-one and lets a visualizer
    /// traverse the complete provenance tree.
    #[inline]
    pub fn as_scope(&self) -> Option<(&ProvenanceScope, &Provenance)> {
        match self.0.as_deref() {
            Some(ProvenanceNode::Scope { scope, origin }) => Some((scope, origin)),
            _ => None,
        }
    }

    /// Returns the merged source origins of this [`Provenance`] when it is a fused provenance, and [`None`] otherwise.
    /// Fused constituents are normalized at construction; they are never unknown, never themselves fused, and never
    /// structural duplicates, and there are always at least two of them.
    #[inline]
    pub fn as_fused(&self) -> Option<&[Provenance]> {
        match self.0.as_deref() {
            Some(ProvenanceNode::Fused { origins }) => Some(origins),
            _ => None,
        }
    }
}

impl Default for Provenance {
    #[inline]
    fn default() -> Self {
        Self::unknown()
    }
}

impl Display for Provenance {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0.as_deref() {
            None => write!(formatter, "unknown"),
            Some(node) => node.fmt(formatter),
        }
    }
}

/// Active provenance state owned by a [`Context`](crate::Context) that is a staging boundary (i.e., one that owns the
/// [`ProgramBuilder`](crate::ProgramBuilder) instructions are emitted into, such as a tracing or partial evaluation
/// context). The state is shared across clones of the owning context (typically behind an [`Rc`](std::rc::Rc)) so
/// that every clone observes the same active scopes, while independent traces own independent states and cannot leak
/// scopes into each other.
///
/// The state keeps the innermost active origin, the scope frames entered after it (outermost first), and a cached
/// fully composed [`Provenance`] that folds those frames over that origin:
///
///   - Entering a scope pushes one frame. Entering an origin replaces the active origin and clears the frame list,
///     so frames that were already active at that entry are the enclosing transform's ambient context, not part of
///     the instruction's origin (they are restored when the origin is left). Replaying a source instruction under an
///     ambient scope that its provenance already records therefore preserves the source provenance exactly, with no
///     double wrap, while newly synthesized work staged with no origin still receives the ambient scopes folded over
///     unknown provenance.
///   - Entering an origin while another origin is active fuses the new origin with the provenance _composed at that
///     moment_ (i.e., the outer origin with all frames entered after it already folded). Two consequences of this fused
///     model are deliberate: (i) entering an _unknown_ origin inside an active origin keeps the enclosing composition
///     (fused normalization discards unknown constituents), so a nested replay of an unlabeled instruction attributes
///     its rewrite to the enclosing source, exactly like an inlined operation inheriting its call site's location, and
///     (ii) a [`seeded`](Self::seeded) boundary fuses replayed source origins with its seed, so residual and nested
///     trace instructions record both where the boundary came from and where each instruction came from.
///
/// The cache is recomputed only when entering a scope or origin, while leaving one restores the entry snapshot and
/// performs no recomposition. [`current`](Self::current) and instruction staging clone the cached value and never
/// rebuild nodes. Entry side recomposition rebuilds the chain above the change point, which requires `O(scope depth)`
/// node allocations per entry. That is accepted because scope nesting is shallow in practice, and the requirement is
/// precisely that composition happens only at scope and origin transitions, never per staged instruction.
///
/// Both [`with_scope`](Self::invoke_with_scope) and [`with_origin`](Self::invoke_with_origin) restore the previous state on ordinary
/// return, error return, and panic unwinding through internal Resource Acquisition Is Initialization (RAII) guards, and
/// no [`RefCell`] borrow is held while the provided closure runs.
#[derive(Debug)]
pub struct ProvenanceState {
    /// Interior-mutable state. [`ProvenanceState`] is shared by reference across context clones, so all mutation
    /// happens through short-lived borrows that are never held while user closures run.
    inner: RefCell<ProvenanceStateInner>,
}

impl ProvenanceState {
    /// Creates a new empty [`ProvenanceState`] with no active scopes or origins and unknown composed [`Provenance`].
    #[inline]
    pub fn new() -> Self {
        Self::seeded(Provenance::unknown())
    }

    /// Creates a new [`ProvenanceState`] seeded with the provided origin and no scope frames. A nested staging
    /// boundary (e.g., a nested tracing context) seeds its state from its parent context's current provenance, so that
    /// instructions staged in the nested program record where the nested program itself came from, while later
    /// replaying those instructions under the same ambient scopes preserves each instruction's provenance exactly
    /// instead of wrapping or fusing the shared scopes twice.
    #[inline]
    pub fn seeded(origin: Provenance) -> Self {
        // With no scopes entered yet, the composed provenance is the seed itself.
        Self {
            inner: RefCell::new(ProvenanceStateInner {
                origin: if origin.is_unknown() { None } else { Some(origin.clone()) },
                scopes: Vec::new(),
                composed: origin,
            }),
        }
    }

    /// Returns the currently composed [`Provenance`]. This clones the cached composition and performs no node
    /// construction, so it is cheap enough to call once per staged instruction.
    #[inline]
    pub fn current(&self) -> Provenance {
        self.inner.borrow().composed.clone()
    }

    /// Invokes `function` with the provided origin [`Provenance`] entered as the active origin, restoring the previous
    /// state afterwards (including on error returns and panic unwinding). When another origin is already active, the
    /// new origin is fused with the provenance composed at this moment. Either way the frame list is cleared, so only
    /// scopes entered after this call fold over the new origin.
    pub fn invoke_with_origin<R, F: FnOnce() -> R>(&self, origin: Provenance, function: F) -> R {
        let _guard = self.enter(|inner| {
            inner.origin =
                Some(if inner.origin.is_some() { Provenance::fused([inner.composed.clone(), origin]) } else { origin });
            inner.scopes.clear();
        });
        function()
    }

    /// Invokes `function` with the provided [`ProvenanceScope`] entered as the innermost active scope frame, restoring
    /// the previous state afterwards (including on error returns and panic unwinding).
    #[inline]
    pub fn invoke_with_scope<R, F: FnOnce() -> R>(&self, scope: ProvenanceScope, function: F) -> R {
        let _guard = self.enter(|inner| inner.scopes.push(scope));
        function()
    }

    /// Snapshots the current state into a [`ProvenanceRestorationGuard`], applies `enter` to record one scope or origin
    /// entry, and recomputes the cached composition, all under one short-lived borrow that ends before the guard is
    /// returned. Restoring the snapshot on drop is definitionally identical to undoing the entry and recomposing,
    /// because the composition is a pure function of the origin and frames, so leaving performs no recomposition.
    fn enter<F: FnOnce(&mut ProvenanceStateInner)>(&self, enter: F) -> ProvenanceRestorationGuard<'_> {
        let mut inner = self.inner.borrow_mut();
        let snapshot = inner.clone();
        enter(&mut inner);
        inner.recompute_composed_provenance();
        // The guard exists only after the borrow ends, so even a panic inside `enter` or `recompose` can never make
        // its `Drop` re-borrow the state while this borrow is still live.
        drop(inner);
        ProvenanceRestorationGuard { state: self, snapshot }
    }
}

impl Default for ProvenanceState {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Interior state of a [`ProvenanceState`].
#[derive(Clone, Debug, Default)]
struct ProvenanceStateInner {
    /// Innermost effective origin [`Provenance`]. This is the entered origin provenance, fused with the provenance
    /// composed at entry time when another origin was already active, or [`None`] when no origin is active. Note that
    /// an active _unknown_ origin is distinct from [`None`] in that it still suppresses the ambient frames cleared at
    /// its entry.
    origin: Option<Provenance>,

    /// [`ProvenanceScope`] frames entered after the active origin [`Provenance`], outermost first. Entering an origin
    /// provenance clears this list, which is what keeps ambient frames out of replayed instructions' provenance.
    scopes: Vec<ProvenanceScope>,

    /// Cached fully composed [`Provenance`], recomputed only at scope/origin transitions.
    composed: Provenance,
}

impl ProvenanceStateInner {
    /// Recomputes the cached fully composed [`Provenance`] by folding the [`ProvenanceScope`] frames outermost-first
    /// over the active origin provenance or over unknown provenance when no origin is active.
    #[inline]
    fn recompute_composed_provenance(&mut self) {
        let base = self.origin.clone().unwrap_or_else(Provenance::unknown);
        self.composed = self.scopes.iter().rev().fold(base, |origin, scope| Provenance::scope(scope.clone(), origin));
    }
}

/// Resource Acquisition Is Initialization (RAII) guard that restores the [`ProvenanceState`] snapshot taken when its
/// entry was recorded, on ordinary return, error return, and panic unwinding alike. The guard holds no [`RefCell`]
/// borrow while the user closure runs. It re-borrows only inside [`Drop`].
struct ProvenanceRestorationGuard<'s> {
    /// [`ProvenanceState`] to restore.
    state: &'s ProvenanceState,

    /// Complete [`ProvenanceStateInner`] snapshot at entry.
    snapshot: ProvenanceStateInner,
}

impl Drop for ProvenanceRestorationGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        *self.state.inner.borrow_mut() = std::mem::take(&mut self.snapshot);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_provenance_unknown() {
        let unknown = Provenance::unknown();
        assert!(unknown.is_unknown());
        assert_eq!(unknown.as_scope(), None);
        assert_eq!(unknown.as_fused(), None);
    }

    #[test]
    fn test_provenance_scope() {
        let scope = ProvenanceScope::new("coordinate_basis");
        assert_eq!(scope.name(), "coordinate_basis");
        let provenance = Provenance::scope(scope.clone(), Provenance::unknown());
        assert!(!provenance.is_unknown());
        assert_eq!(provenance.as_scope(), Some((&scope, &Provenance::unknown())));
        assert_eq!(provenance.as_fused(), None);
    }

    #[test]
    fn test_provenance_nested_scopes() {
        let outer = ProvenanceScope::new("outer");
        let inner = ProvenanceScope::new("inner");
        let provenance = Provenance::scope(outer.clone(), Provenance::scope(inner.clone(), Provenance::unknown()));

        // Walking `as_scope` repeatedly recovers the complete outermost-first scope path and the origin below it.
        let mut path = Vec::new();
        let mut current = &provenance;
        while let Some((scope, origin)) = current.as_scope() {
            path.push(scope);
            current = origin;
        }
        assert_eq!(path, vec![&outer, &inner]);
        assert!(current.is_unknown());

        // A scope chain above a fused origin stops at the fused node, which is reached through `as_scope()` and
        // traversed through `as_fused()`.
        let first = Provenance::scope(ProvenanceScope::new("first"), Provenance::unknown());
        let second = Provenance::scope(ProvenanceScope::new("second"), Provenance::unknown());
        let fused = Provenance::fused([first.clone(), second.clone()]);
        let provenance = Provenance::scope(outer.clone(), fused.clone());
        assert_eq!(provenance.as_scope(), Some((&outer, &fused)));
        assert_eq!(fused.as_fused(), Some([first, second].as_slice()));

        // `Provenance::scope` performs no deduplication or common-prefix factoring.
        let repeated = Provenance::scope(outer.clone(), Provenance::scope(outer.clone(), Provenance::unknown()));
        let (first_scope, below) = repeated.as_scope().unwrap();
        assert_eq!(first_scope, &outer);
        assert_eq!(below.as_scope(), Some((&outer, &Provenance::unknown())));
    }

    #[test]
    fn test_provenance_fused_normalization() {
        let a = Provenance::scope(ProvenanceScope::new("a"), Provenance::unknown());
        let b = Provenance::scope(ProvenanceScope::new("b"), Provenance::unknown());
        let c = Provenance::scope(ProvenanceScope::new("c"), Provenance::unknown());

        // Zero origins return unknown and one origin is returned directly.
        assert_eq!(Provenance::fused([]), Provenance::unknown());
        assert_eq!(Provenance::fused([a.clone()]), a);

        // Unknown origins are discarded when another origin is present.
        assert_eq!(Provenance::fused([Provenance::unknown(), Provenance::unknown()]), Provenance::unknown());
        assert_eq!(Provenance::fused([Provenance::unknown(), a.clone()]), a);
        assert_eq!(
            Provenance::fused([a.clone(), Provenance::unknown(), b.clone()]),
            Provenance::fused([a.clone(), b.clone()]),
        );

        // Nested fused origins are flattened.
        assert_eq!(
            Provenance::fused([Provenance::fused([a.clone(), b.clone()]), c.clone()]),
            Provenance::fused([a.clone(), b.clone(), c.clone()]),
        );

        // Structurally duplicate origins are removed while preserving first-occurrence order.
        let deduplicated = Provenance::fused([b.clone(), a.clone(), b.clone()]);
        assert_eq!(deduplicated.as_fused(), Some([b.clone(), a.clone()].as_slice()));
        assert_eq!(
            Provenance::fused([Provenance::fused([a.clone(), b.clone()]), b.clone()]),
            Provenance::fused([a.clone(), b.clone()]),
        );
    }

    #[test]
    fn test_provenance_equality() {
        let scope = || ProvenanceScope::new("scope");
        assert_eq!(ProvenanceScope::new("scope"), ProvenanceScope::new("scope"));
        assert_ne!(ProvenanceScope::new("scope"), ProvenanceScope::new("other"));

        // Equality is structural across independently constructed values.
        assert_eq!(
            Provenance::scope(scope(), Provenance::unknown()),
            Provenance::scope(scope(), Provenance::unknown()),
        );
        assert_ne!(Provenance::scope(scope(), Provenance::unknown()), Provenance::unknown());
        assert_ne!(
            Provenance::scope(scope(), Provenance::scope(scope(), Provenance::unknown())),
            Provenance::scope(scope(), Provenance::unknown()),
        );
    }

    #[test]
    fn test_provenance_hashing() {
        let build = || {
            let a = Provenance::scope(ProvenanceScope::new("a"), Provenance::unknown());
            let b = Provenance::scope(ProvenanceScope::new("b"), Provenance::unknown());
            Provenance::scope(ProvenanceScope::new("outer"), Provenance::fused([a, b]))
        };
        let mut set = HashSet::new();
        set.insert(build());
        set.insert(build());
        set.insert(Provenance::unknown());
        set.insert(Provenance::unknown());
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_provenance_display() {
        assert_eq!(Provenance::unknown().to_string(), "unknown");

        let inner = Provenance::scope(ProvenanceScope::new("inner"), Provenance::unknown());
        let outer = Provenance::scope(ProvenanceScope::new("outer"), inner.clone());
        assert_eq!(inner.to_string(), "inner");
        assert_eq!(outer.to_string(), "outer::inner");

        let a = Provenance::scope(ProvenanceScope::new("a"), Provenance::unknown());
        let fused = Provenance::fused([a.clone(), outer.clone()]);
        assert_eq!(fused.to_string(), "fused[a, outer::inner]");
        assert_eq!(Provenance::scope(ProvenanceScope::new("top"), fused).to_string(), "top::fused[a, outer::inner]",);

        // Non-identifier scope names render quoted and escaped deterministically as Rust string literals, so
        // arbitrary name content can never make the rendering ambiguous.
        let escaped = Provenance::scope(ProvenanceScope::new("quo\"te\nline"), Provenance::unknown());
        assert_eq!(escaped.to_string(), "\"quo\\\"te\\nline\"");
        let mixed = Provenance::scope(
            ProvenanceScope::new("layer 3"),
            Provenance::scope(ProvenanceScope::new("attention"), Provenance::unknown()),
        );
        assert_eq!(mixed.to_string(), "\"layer 3\"::attention");
        assert_eq!(Provenance::scope(ProvenanceScope::new("a::b"), Provenance::unknown()).to_string(), "\"a::b\"");
        assert_eq!(Provenance::scope(ProvenanceScope::new("0"), Provenance::unknown()).to_string(), "\"0\"");
        assert_eq!(Provenance::scope(ProvenanceScope::new(""), Provenance::unknown()).to_string(), "\"\"");
    }

    #[test]
    fn test_provenance_state_scopes() {
        let outer = ProvenanceScope::new("outer");
        let inner = ProvenanceScope::new("inner");
        let state = ProvenanceState::new();
        assert!(state.current().is_unknown());
        state.invoke_with_scope(outer.clone(), || {
            assert_eq!(state.current(), Provenance::scope(outer.clone(), Provenance::unknown()));
            state.invoke_with_scope(inner.clone(), || {
                // The earlier-entered scope is the outermost node.
                assert_eq!(
                    state.current(),
                    Provenance::scope(outer.clone(), Provenance::scope(inner.clone(), Provenance::unknown())),
                );
            });
            assert_eq!(state.current(), Provenance::scope(outer.clone(), Provenance::unknown()));
        });
        assert!(state.current().is_unknown());
    }

    #[test]
    fn test_provenance_state_origin_scope_boundary() {
        let ambient = ProvenanceScope::new("ambient");
        let transform = ProvenanceScope::new("transform");
        let nested_transform = ProvenanceScope::new("nested_transform");
        let origin = Provenance::scope(ProvenanceScope::new("source"), Provenance::unknown());
        let state = ProvenanceState::new();
        state.invoke_with_scope(ambient.clone(), || {
            state.invoke_with_origin(origin.clone(), || {
                // Scopes entered before the origin do not fold over it, so a 1-to-1 replay under an ambient scope
                // preserves the source provenance exactly.
                assert_eq!(state.current(), origin);
                state.invoke_with_scope(transform.clone(), || {
                    // Scopes entered after the origin do fold over it.
                    assert_eq!(state.current(), Provenance::scope(transform.clone(), origin.clone()));
                    state.invoke_with_scope(nested_transform.clone(), || {
                        // Active scopes remain outside the source provenance's own scope chain and preserve their
                        // entry order.
                        assert_eq!(
                            state.current(),
                            Provenance::scope(
                                transform.clone(),
                                Provenance::scope(nested_transform.clone(), origin.clone()),
                            ),
                        );
                    });
                });
            });
            // Work staged with no origin still receives the ambient scope folded over unknown provenance.
            assert_eq!(state.current(), Provenance::scope(ambient.clone(), Provenance::unknown()));
        });
    }

    #[test]
    fn test_provenance_state_nested_origins_fuse() {
        let a = Provenance::scope(ProvenanceScope::new("a"), Provenance::unknown());
        let b = Provenance::scope(ProvenanceScope::new("b"), Provenance::unknown());
        let intermediate = ProvenanceScope::new("s");
        let state = ProvenanceState::new();
        state.invoke_with_origin(a.clone(), || {
            state.invoke_with_scope(intermediate.clone(), || {
                let composed = Provenance::scope(intermediate.clone(), a.clone());
                assert_eq!(state.current(), composed);
                state.invoke_with_origin(b.clone(), || {
                    // The nested origin fuses with the provenance composed at entry, not with the raw outer origin,
                    // so the intermediate scope `s` is retained.
                    assert_eq!(state.current(), Provenance::fused([composed.clone(), b.clone()]));
                });
                // Leaving the nested origin restores the previous composition.
                assert_eq!(state.current(), composed);
            });
            assert_eq!(state.current(), a);
        });
        assert!(state.current().is_unknown());
    }

    #[test]
    fn test_provenance_state_unknown_origin_inside_active_origin() {
        // Entering an unknown origin inside an active origin keeps the enclosing composition, because fused
        // normalization discards unknown constituents: a nested replay of an unlabeled instruction attributes
        // its rewrite to the enclosing source instruction.
        let a = Provenance::scope(ProvenanceScope::new("a"), Provenance::unknown());
        let state = ProvenanceState::new();
        state.invoke_with_origin(a.clone(), || {
            state.invoke_with_origin(Provenance::unknown(), || {
                assert_eq!(state.current(), a);
            });
        });

        // Without an enclosing origin, an unknown origin composes to unknown, so replaying an unlabeled instruction
        // under ambient scopes preserves its unknown provenance exactly.
        let ambient = ProvenanceScope::new("ambient");
        state.invoke_with_scope(ambient.clone(), || {
            state.invoke_with_origin(Provenance::unknown(), || {
                assert!(state.current().is_unknown());
            });
        });
    }

    #[test]
    fn test_provenance_state_seeded() {
        let seed = Provenance::scope(ProvenanceScope::new("seed"), Provenance::unknown());
        let nested = ProvenanceScope::new("nested");
        let state = ProvenanceState::seeded(seed.clone());
        assert_eq!(state.current(), seed);
        state.invoke_with_scope(nested.clone(), || {
            assert_eq!(state.current(), Provenance::scope(nested.clone(), seed.clone()));
        });
        assert_eq!(state.current(), seed);
        assert!(ProvenanceState::seeded(Provenance::unknown()).current().is_unknown());

        // A seeded boundary fuses replayed source origins with its seed, recording both where the boundary came from
        // and where each replayed instruction came from.
        let source = Provenance::scope(ProvenanceScope::new("source"), Provenance::unknown());
        state.invoke_with_origin(source.clone(), || {
            assert_eq!(state.current(), Provenance::fused([seed.clone(), source.clone()]));
        });
    }

    #[test]
    fn test_provenance_state_restores_after_error_and_panic() {
        let scope = ProvenanceScope::new("scope");
        let origin = Provenance::scope(ProvenanceScope::new("origin"), Provenance::unknown());
        let state = ProvenanceState::new();

        // Restoration after an ordinary error return.
        let result: Result<(), &str> = state.invoke_with_scope(scope.clone(), || Err("failure"));
        assert_eq!(result, Err("failure"));
        assert!(state.current().is_unknown());

        // Restoration after panic unwinding, for both scope and origin frames.
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state.invoke_with_scope(scope.clone(), || state.invoke_with_origin(origin.clone(), || panic!("unwind")))
        }));
        assert!(panic.is_err());
        assert!(state.current().is_unknown());
        state.invoke_with_scope(scope.clone(), || {
            assert_eq!(state.current(), Provenance::scope(scope.clone(), Provenance::unknown()));
        });
    }
}
