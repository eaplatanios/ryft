//! Identifiers for the function/primitive being compiled.

use std::panic::Location;

/// Identifier for the function or primitive being compiled.
///
/// Two compilations with the same [`FunctionFingerprint`] and the same input type signatures
/// (and the same backend-specific options) produce the same executable and share a cache entry.
/// This mirrors how JAX's compile cache combines a function fingerprint with abstract input
/// value signatures.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum FunctionFingerprint {
    /// A `ryft` built-in primitive identified by a static name. Reserved for internal use cases
    /// like the compiled-reshard path.
    Primitive(&'static str),

    /// A user function identified by the source location of its outer entry point (for example
    /// the call site of [`compile_and_execute_with_options`](super::compile_and_execute_with_options)).
    /// Construct via [`FunctionFingerprint::from_caller`].
    ///
    /// JAX uses the Python function's identity plus closure-captured cells as the fingerprint;
    /// Rust's closures don't expose an equivalent stable identity, so `ryft` uses the call-site
    /// location as a best-effort proxy. The core pipeline pairs this with a hash of the input
    /// tree's structural shape (see [`FunctionFingerprint::Composite`]) so that two calls at the
    /// same source line with structurally-different inputs still get distinct cache entries.
    SourceLocation {
        /// Source file the call site lives in.
        file: &'static str,

        /// Line number of the call site.
        line: u32,

        /// Column of the call site.
        column: u32,
    },

    /// A composite fingerprint: a base fingerprint mixed with an opaque 64-bit hash of additional
    /// state that uniquely identifies a function instance. The core pipeline uses this to fold a
    /// hash of the input tree's
    /// [`ParameterStructure`](crate::parameters::Parameterized::ParameterStructure) into the
    /// call-site fingerprint, so non-`Parameter` fields (hyperparameters, mode flags, ...) baked
    /// into the user's input tree partition the cache automatically.
    Composite {
        /// Base fingerprint that the extra state is mixed into.
        base: Box<FunctionFingerprint>,

        /// Opaque 64-bit hash of the additional state.
        extra: u64,
    },
}

impl FunctionFingerprint {
    /// Constructs a [`FunctionFingerprint::SourceLocation`] from the call site that invokes this
    /// function. The call-site location is captured at compile time via `#[track_caller]` and is
    /// stable as long as the call site doesn't move.
    #[inline]
    #[track_caller]
    pub fn from_caller() -> Self {
        let location = Location::caller();
        Self::SourceLocation { file: location.file(), line: location.line(), column: location.column() }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use super::*;

    fn hash_of<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[track_caller]
    fn caller_fingerprint() -> FunctionFingerprint {
        FunctionFingerprint::from_caller()
    }

    #[test]
    fn test_from_caller_captures_call_site() {
        let fingerprint = caller_fingerprint();
        match fingerprint {
            FunctionFingerprint::SourceLocation { file, line, column } => {
                assert!(file.ends_with("fingerprint.rs"), "unexpected file: {file}");
                assert!(line > 0);
                assert!(column > 0);
            }
            other => panic!("expected SourceLocation, got {other:?}"),
        }
    }

    #[test]
    fn test_from_caller_distinguishes_distinct_call_sites() {
        let first = caller_fingerprint();
        let second = caller_fingerprint();
        assert_ne!(first, second, "fingerprints from distinct call sites should differ");
    }

    #[test]
    fn test_composite_distinguishes_extras() {
        let base = FunctionFingerprint::Primitive("ryft.test");
        let a = FunctionFingerprint::Composite { base: Box::new(base.clone()), extra: 1 };
        let b = FunctionFingerprint::Composite { base: Box::new(base), extra: 2 };
        assert_ne!(a, b);
        assert_ne!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn test_primitive_round_trips() {
        let fingerprint = FunctionFingerprint::Primitive("ryft.test");
        let clone = fingerprint.clone();
        assert_eq!(fingerprint, clone);
        assert_eq!(hash_of(&fingerprint), hash_of(&clone));
    }
}
