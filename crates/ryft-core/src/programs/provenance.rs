use std::fmt::Display;
use std::sync::Arc;

/// Named provenance scope describing one level of the logical origin of an [`Instruction`](crate::Instruction) (e.g.,
/// the framework facility or user computation that staged it). Scope names are stable, fully-qualified strings that are
/// preserved verbatim. Construction is infallible and performs no validation because names have no correctness role.
/// Names should be non-empty, and the `ryft.` prefix is reserved for framework-owned scopes following the
/// `ryft.<subsystem>.<concept>` convention. User scopes should use their own namespace. Each scope maps onto
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

    /// Several source origins intentionally merged into one generated instruction. Normalized at construction by
    /// [`Provenance::fused`]: constituents are never unknown, never themselves fused, and never structural
    /// duplicates, and there are always at least two of them.
    Fused {
        /// Merged source origins, in first-occurrence order.
        origins: Arc<[Provenance]>,
    },
}

/// Persistent, hierarchical, non-semantic origin of one [`Instruction`](crate::Instruction). Provenance records _where_
/// an instruction came from (e.g., the framework facility that staged it, or the source instructions a transform
/// generated it from) without adding a Single Static Assignment (SSA) value, a data dependency, or any semantic
/// behavior. It is purely diagnostic (i.e., type inference, effects, differentiation, batching, interpretation,
/// optimization legality, and the canonical semantic [`Program`](crate::Program) rendering must never depend on it),
/// and behavior must remain correct if all provenance is removed. Consequently, [`unknown`](Self::unknown) is always a
/// correct value as dropping provenance degrades diagnostics and should not change program behavior.
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
/// The [`Display`] output uses the diagnostic expression grammar with scope names escaped as Rust string literals:
/// `"outer"("inner")` for a scope chain, `fused["a", "b"]` for fused origins, and `unknown` for unknown provenance
/// (program renderings omit the provenance of such instructions entirely instead of printing this token).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Provenance(Option<Arc<ProvenanceNode>>);

// TODO(eaplatanios): Review from here onwards.

impl Provenance {
    /// Returns the [`Provenance`] of an instruction with no recorded origin. This value is allocation-free.
    pub fn unknown() -> Self {
        Self(None)
    }

    /// Returns `true` if this [`Provenance`] records no origin. This gives renderers, lowerers, serialization
    /// boundaries, and tests a direct way to make that distinction without reconstructing it from traversal results.
    pub fn is_unknown(&self) -> bool {
        self.0.is_none()
    }

    /// Creates a new [`Provenance`] that attaches one named scope above the provided origin. No deduplication or
    /// common-prefix factoring is performed: the stored representation stays purely structural, and visualizers may
    /// factor common scope prefixes at display time.
    ///
    /// # Parameters
    ///
    ///   - `scope`: Scope to attach as the new outermost level.
    ///   - `origin`: Origin recorded below the new scope (possibly [`unknown`](Self::unknown)).
    pub fn scope(scope: ProvenanceScope, origin: Provenance) -> Self {
        Self(Some(Arc::new(ProvenanceNode::Scope { scope, origin })))
    }

    /// Creates a new [`Provenance`] for a generated instruction with multiple source origins (e.g., several source
    /// instructions intentionally merged into one by a transform). The provided origins are normalized at
    /// construction:
    ///
    ///   - unknown origins are discarded when another origin is present;
    ///   - nested fused origins are flattened;
    ///   - structurally duplicate origins are removed while preserving first-occurrence order; and
    ///   - zero remaining origins return [`unknown`](Self::unknown) and one remaining origin is returned directly.
    pub fn fused(origins: impl IntoIterator<Item = Provenance>) -> Self {
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

    /// Returns the scope names of this [`Provenance`], from the outermost scope down to the first non-scope node.
    /// Returns an empty path for unknown provenance and for fused roots, whose constituents are reached through
    /// [`origins`](Self::origins). The provenance below the innermost returned scope is reached through
    /// [`origin`](Self::origin).
    pub fn scope_path(&self) -> Vec<&ProvenanceScope> {
        let mut path = Vec::new();
        let mut current = self;
        while let Some(ProvenanceNode::Scope { scope, origin }) = current.0.as_deref() {
            path.push(scope);
            current = origin;
        }
        path
    }

    /// Returns the provenance below the innermost scope named by [`scope_path`](Self::scope_path) when this
    /// [`Provenance`] is a scope chain, and `None` for unknown provenance and fused roots. Returning `None` instead
    /// of `self` as a sentinel prevents accidental non-terminating traversal. Together with
    /// [`is_unknown`](Self::is_unknown), [`scope_path`](Self::scope_path), and [`origins`](Self::origins), this lets
    /// a visualizer traverse the complete provenance tree.
    pub fn origin(&self) -> Option<&Provenance> {
        match self.0.as_deref() {
            Some(ProvenanceNode::Scope { origin, .. }) => {
                let mut current = origin;
                while let Some(ProvenanceNode::Scope { origin, .. }) = current.0.as_deref() {
                    current = origin;
                }
                Some(current)
            }
            _ => None,
        }
    }

    /// Returns the merged source origins of this [`Provenance`] when it is a fused root, and an empty slice
    /// otherwise. Fused constituents are normalized at construction: they are never unknown, never themselves fused,
    /// and never structural duplicates.
    pub fn origins(&self) -> &[Provenance] {
        match self.0.as_deref() {
            Some(ProvenanceNode::Fused { origins }) => origins,
            _ => &[],
        }
    }
}

impl Display for Provenance {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0.as_deref() {
            None => write!(formatter, "unknown"),
            Some(ProvenanceNode::Scope { scope, origin }) => {
                // Scope names are preserved verbatim in storage and escaped deterministically at render time as Rust
                // string literals, which handles quotes, newlines, and delimiters unambiguously.
                write!(formatter, "{:?}", scope.name())?;
                if !origin.is_unknown() {
                    write!(formatter, "({origin})")?;
                }
                Ok(())
            }
            Some(ProvenanceNode::Fused { origins }) => {
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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_provenance_unknown() {
        let unknown = Provenance::unknown();
        assert!(unknown.is_unknown());
        assert_eq!(unknown.scope_path(), Vec::<&ProvenanceScope>::new());
        assert_eq!(unknown.origin(), None);
        assert_eq!(unknown.origins(), &[]);
    }

    #[test]
    fn test_provenance_scope() {
        let scope = ProvenanceScope::new("ryft.differentiation.coordinate_basis");
        assert_eq!(scope.name(), "ryft.differentiation.coordinate_basis");
        let provenance = Provenance::scope(scope.clone(), Provenance::unknown());
        assert!(!provenance.is_unknown());
        assert_eq!(provenance.scope_path(), vec![&scope]);
        assert_eq!(provenance.origin(), Some(&Provenance::unknown()));
        assert_eq!(provenance.origins(), &[]);
    }

    #[test]
    fn test_provenance_nested_scopes() {
        let outer = ProvenanceScope::new("outer");
        let inner = ProvenanceScope::new("inner");
        let provenance = Provenance::scope(outer.clone(), Provenance::scope(inner.clone(), Provenance::unknown()));
        assert_eq!(provenance.scope_path(), vec![&outer, &inner]);
        assert_eq!(provenance.origin(), Some(&Provenance::unknown()));

        // A scope chain above a fused origin stops at the fused node, which is reached through `origin()` and
        // traversed through `origins()`.
        let first = Provenance::scope(ProvenanceScope::new("first"), Provenance::unknown());
        let second = Provenance::scope(ProvenanceScope::new("second"), Provenance::unknown());
        let fused = Provenance::fused([first.clone(), second.clone()]);
        let provenance = Provenance::scope(outer.clone(), fused.clone());
        assert_eq!(provenance.scope_path(), vec![&outer]);
        assert_eq!(provenance.origin(), Some(&fused));
        assert_eq!(provenance.origin().unwrap().origins(), &[first, second]);

        // `Provenance::scope` performs no deduplication or common-prefix factoring.
        let repeated = Provenance::scope(outer.clone(), Provenance::scope(outer.clone(), Provenance::unknown()));
        assert_eq!(repeated.scope_path(), vec![&outer, &outer]);
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
        assert_eq!(deduplicated.origins(), &[b.clone(), a.clone()]);
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
        assert_eq!(inner.to_string(), "\"inner\"");
        assert_eq!(outer.to_string(), "\"outer\"(\"inner\")");

        let a = Provenance::scope(ProvenanceScope::new("a"), Provenance::unknown());
        let fused = Provenance::fused([a.clone(), outer.clone()]);
        assert_eq!(fused.to_string(), "fused[\"a\", \"outer\"(\"inner\")]");
        assert_eq!(
            Provenance::scope(ProvenanceScope::new("top"), fused).to_string(),
            "\"top\"(fused[\"a\", \"outer\"(\"inner\")])",
        );

        // Scope names are escaped deterministically as Rust string literals.
        let escaped = Provenance::scope(ProvenanceScope::new("quo\"te\nline"), Provenance::unknown());
        assert_eq!(escaped.to_string(), "\"quo\\\"te\\nline\"");
    }
}
